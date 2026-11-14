import json
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hm2.data import CLASS_NAMES
from hm2.grouped_validation import (
    assess_candidate,
    assign_video_folds,
    clip_video,
    evaluate_groups,
    planned_candidates,
    select_validation_pool,
    sign_flip_p_value,
    subset_reversal_rate,
    summarize_det_overlap,
    write_coco_groups,
)
from hm2.yolo_pipeline import OPTIONAL_INFERENCE_KEYS, apply_experiment, load_pipeline_settings

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = {
    "min_improved_share": 0.67,
    "max_p_value": 0.1,
    "max_group_loss": 0.02,
    "subset_fraction": 0.3,
    "subset_draws": 500,
    "seed": 2026,
}


def _frames(clips: dict[str, int]) -> pl.DataFrame:
    return pl.from_dicts(
        [
            {
                "split": "train",
                "clip": clip,
                "frame_id": frame_id,
                "target_count": 2,
                "is_labeled": True,
            }
            for clip, count in clips.items()
            for frame_id in range(1, count + 1)
        ]
    )


def test_clip_video_reads_the_raw_video_number() -> None:
    assert clip_video("uav0000013_01073_v") == "0000013"


def test_det_overlap_separates_frame_and_video_contamination() -> None:
    frames = _frames({"uav0000001_00100_v": 10, "uav0000002_00000_v": 10, "uav0000003_00000_v": 10})
    index = {
        "videos": {
            "0000001": {"train": [105], "test-dev": [1]},
            "0000002": {"val": [40]},
            "0000003": {"test-dev": [5]},
        }
    }
    overlap = summarize_det_overlap(frames, index).sort("clip")
    assert overlap["video"].to_list() == ["0000001", "0000002", "0000003"]
    assert overlap["det_images_inside_clip"].to_list() == [1, 0, 0]
    assert overlap["det_nearest_frame_gap"].to_list() == [0, 31, None]
    assert overlap["det_test_dev_images_same_video"].to_list() == [1, 0, 1]
    assert overlap["is_frame_clean"].to_list() == [False, True, True]
    assert overlap["is_video_clean"].to_list() == [False, False, True]
    assert overlap["objects"].to_list() == [20, 20, 20]


def test_pool_drops_clean_clips_whose_video_has_a_contaminated_clip() -> None:
    frames = _frames({"uav0000001_00000_v": 5, "uav0000001_00100_v": 5, "uav0000002_00000_v": 5})
    index = {"videos": {"0000001": {"train": [102]}}}
    pool = select_validation_pool(summarize_det_overlap(frames, index))
    assert pool["clip"].to_list() == ["uav0000002_00000_v"]


def test_folds_keep_videos_whole_and_balance_objects_greedily() -> None:
    pool = pl.DataFrame(
        {
            "clip": ["a1", "a2", "b", "c", "d"],
            "video": ["A", "A", "B", "C", "D"],
            "objects": [30, 30, 50, 40, 20],
        }
    )
    folds = assign_video_folds(pool, 2)
    by_clip = dict(zip(folds["clip"], folds["fold"], strict=True))
    assert by_clip["a1"] == by_clip["a2"] == 0
    assert by_clip["b"] == 1
    assert by_clip["c"] == 1
    assert by_clip["d"] == 0
    totals = folds.group_by("fold").agg(pl.col("objects").sum()).sort("fold")["objects"]
    assert totals.to_list() == [80, 90]


def test_sign_flip_p_value_is_exact() -> None:
    assert sign_flip_p_value(np.array([0.01, 0.02, 0.03])) == pytest.approx(1 / 8)
    assert sign_flip_p_value(np.array([0.01, -0.01])) == pytest.approx(3 / 4)


def test_reversal_rate_is_zero_for_uniform_gains_and_high_for_mixed_signs() -> None:
    assert subset_reversal_rate(np.full(10, 0.01), 0.3, 200, 0) == 0
    mixed = np.array([0.05, -0.04, 0.03, -0.05, 0.04, -0.03])
    assert subset_reversal_rate(mixed, 0.3, 2000, 0) > 0.3


def test_candidate_assessment_requires_broad_and_not_single_class_gains() -> None:
    broad = np.array([0.01, 0.02, 0.015, 0.01, 0.005, 0.02, 0.01])
    classes = dict.fromkeys(CLASS_NAMES, 0.01)
    adopted = assess_candidate(broad, classes, CRITERIA)
    assert adopted["should_adopt"]
    one_class = {**dict.fromkeys(CLASS_NAMES, -0.001), "van": 0.08}
    assessed = assess_candidate(broad, one_class, CRITERIA)
    assert assessed["is_mean_positive"] and not assessed["is_not_one_class"]
    assert not assessed["should_adopt"]
    one_video = np.array([0.2, -0.01, -0.005, -0.01, 0.0, -0.002, -0.01])
    assessed = assess_candidate(one_video, classes, CRITERIA)
    assert assessed["mean_difference"] > 0 and not assessed["is_improvement_broad"]
    assert not assessed["should_adopt"]


def test_group_evaluation_scores_each_group_with_its_own_classes(tmp_path: Path) -> None:
    images = [
        {"id": index, "width": 200, "height": 100, "file_name": f"{index}.jpg"} for index in (1, 2)
    ]
    boxes = {1: (10.0, 10.0, 40.0, 30.0, 3), 2: (50.0, 20.0, 40.0, 40.0, 1)}
    annotations = [
        {
            "id": image_id,
            "image_id": image_id,
            "category_id": category,
            "bbox": [x, y, width, height],
            "area": width * height,
            "iscrowd": 0,
        }
        for image_id, (x, y, width, height, category) in boxes.items()
    ]
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)]
    source = tmp_path / "pool.json"
    source.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )
    groups = {"first": [1], "second": [2]}
    paths = write_coco_groups(source, groups, tmp_path / "groups")
    records = [
        {"image_id": 1, "category_id": 3, "bbox": [10.0, 10.0, 40.0, 30.0], "score": 0.9},
        {"image_id": 2, "category_id": 3, "bbox": [50.0, 20.0, 40.0, 40.0], "score": 0.8},
    ]
    scores = evaluate_groups(paths, groups, records).sort("group")
    assert scores["map"].to_list() == pytest.approx([1.0, 0.0])
    assert scores["ap_car"][0] == pytest.approx(1.0)
    assert scores["ap_person"][0] is None
    assert scores["ap_person"][1] == pytest.approx(0.0)


def test_grouped_cv_plan_overrides_only_known_settings() -> None:
    plan = tomllib.loads((ROOT / "config/yolo11s_grouped_cv.toml").read_text())
    base = load_pipeline_settings(ROOT / "config" / plan["base_config"])
    settings = apply_experiment(base, {"dataset": plan["dataset"]})
    assert settings["dataset"]["validation_clips"] == []
    for recipe in plan["recipes"].values():
        for seed in recipe["seeds"]:
            trained = apply_experiment(settings, {"training": {"seed": seed, **recipe["training"]}})
            assert trained["training"]["seed"] == seed
    for variant in plan["inference_variants"].values():
        assert set(variant["inference"]) <= set(OPTIONAL_INFERENCE_KEYS)
        apply_experiment(settings, {"inference": variant["inference"]})
    assert set(plan["inference_checks"]["stress_variants"]) <= set(plan["inference_variants"])
    candidates = planned_candidates(plan)
    named = [
        plan["candidates"]["control"],
        *plan["candidates"]["primary"],
        *plan["inference_checks"]["checkpoints"],
    ]
    assert set(named) <= set(candidates)
    assert candidates["low_learning_rate_soup_epoch1"] == [
        ("low_learning_rate", seed) for seed in (2026, 2027, 2028)
    ]
    assert candidates["interpolation_alpha0.5"] == [("control", 2026)]
    index = json.loads((ROOT / "config" / plan["det_frame_index"]).read_text())
    assert {source["images"] for source in index["sources"].values()} == {6471, 548, 1610}
