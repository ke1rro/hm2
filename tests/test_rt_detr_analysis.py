import json
from pathlib import Path

import pytest

from hm2.data import CLASS_NAMES
from hm2.rt_detr_analysis import (
    bin_images_by_density,
    count_near_duplicates,
    count_objects_per_image,
    evaluate_area_bins,
    evaluate_density_bins,
    load_detections,
    load_ground_truth,
    read_validation_curve,
    run_comparison_analysis,
    summarize_convergence,
    suppress_duplicates,
)

BOXES = {
    1: [[10, 10, 6, 6], [40, 40, 20, 20]],
    2: [[10, 10, 100, 100], [200, 10, 12, 12], [300, 10, 30, 30]],
}


def _write_annotations(path: Path) -> Path:
    images = [{"id": image_id, "width": 640, "height": 480} for image_id in BOXES]
    annotations = [
        {
            "id": len(images) * 10 + index + image_id * 100,
            "image_id": image_id,
            "category_id": 3,
            "bbox": box,
            "area": box[2] * box[3],
            "iscrowd": 0,
        }
        for image_id, boxes in BOXES.items()
        for index, box in enumerate(boxes)
    ]
    annotations.append(
        {
            "id": 999,
            "image_id": 2,
            "category_id": 3,
            "bbox": [500, 400, 100, 50],
            "area": 5000,
            "iscrowd": 1,
        }
    )
    path.write_text(
        json.dumps(
            {
                "info": {},
                "images": images,
                "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES, 1)],
                "annotations": annotations,
            }
        )
    )
    return path


def _perfect_predictions() -> list[dict]:
    return [
        {"image_id": image_id, "category_id": 3, "bbox": box, "score": 0.9}
        for image_id, boxes in BOXES.items()
        for box in boxes
    ]


def test_area_bins_score_perfect_predictions_and_skip_empty_bins(tmp_path: Path) -> None:
    """Catch size bins that use the wrong area ranges or report empty bins as zero."""
    ground_truth = load_ground_truth(_write_annotations(tmp_path / "gt.json"))
    detections = load_detections(ground_truth, _perfect_predictions())
    bins = {"all": (0.0, 1e10), "tiny": (0.0, 64.0), "huge": (1e6, 1e10)}
    result = evaluate_area_bins(ground_truth, detections, [1, 2], bins)
    assert result["all"] == pytest.approx(1.0)
    assert result["tiny"] == pytest.approx(1.0)
    assert result["huge"] is None
    empty = evaluate_area_bins(ground_truth, load_detections(ground_truth, []), [1, 2], bins)
    assert empty["all"] == pytest.approx(0.0)


def test_density_bins_count_targets_without_crowds(tmp_path: Path) -> None:
    """Catch crowd records inflating the number of objects that defines a dense scene."""
    ground_truth = load_ground_truth(_write_annotations(tmp_path / "gt.json"))
    counts = count_objects_per_image(ground_truth)
    assert counts == {1: 2, 2: 3}
    bins = {"two": (1, 2), "three_or_more": (3, 10**9)}
    assert bin_images_by_density(counts, bins) == {"two": [1], "three_or_more": [2]}
    detections = load_detections(ground_truth, _perfect_predictions()[:2])
    result = evaluate_density_bins(ground_truth, detections, [1, 2], bins)
    assert result["two"] == {"images": 1, "objects": 2, "map": 1.0, "ap_small": 1.0}
    assert result["three_or_more"]["map"] == pytest.approx(0.0)


def test_near_duplicates_count_only_same_class_confident_overlaps() -> None:
    """Catch duplicate counts that mix classes or include low-score detections."""
    predictions = [
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 3, "bbox": [1, 0, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 4, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 3, "bbox": [0, 1, 10, 10], "score": 0.1},
        {"image_id": 2, "category_id": 3, "bbox": [50, 50, 10, 10], "score": 0.9},
    ]
    result = count_near_duplicates(predictions, 0.3, 0.5)
    assert result["detections"] == 4
    assert result["duplicate_pairs"] == 1
    assert result["images_with_pairs"] == 1
    assert result["pairs_per_100_detections"] == pytest.approx(25.0)
    assert count_near_duplicates(predictions, 0.05, 0.5)["duplicate_pairs"] == 3


def test_post_hoc_nms_keeps_the_best_of_each_same_class_cluster() -> None:
    """Catch NMS that suppresses across classes or keeps the lower-scoring duplicate."""
    predictions = [
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 3, "bbox": [1, 0, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 4, "bbox": [1, 0, 10, 10], "score": 0.7},
    ]
    kept = suppress_duplicates(predictions, 0.5)
    assert sorted(record["score"] for record in kept) == [0.7, 0.9]


def test_validation_curves_read_both_record_layouts(tmp_path: Path) -> None:
    """Catch the comparison plot losing RetinaNet or B1 epochs because of their formats."""
    nested = tmp_path / "nested"
    flat = tmp_path / "flat"
    nested.mkdir()
    flat.mkdir()
    metrics = {"map": 0.2, "ap50": 0.4, "ap_small": 0.1}
    (nested / "metrics.json").write_text(
        json.dumps(
            [
                {
                    "epoch": 1,
                    "metrics": {"max_dets_100": metrics, "max_dets_500": {}},
                    "image_exposures": 4868,
                    "wall_train_seconds": 900,
                }
            ]
        )
    )
    (flat / "metrics.json").write_text(
        json.dumps(
            [{"epoch": 1, "metrics": metrics, "image_exposures": 4868, "wall_train_seconds": 720}]
        )
    )
    expected = {"epoch": 1, "image_exposures": 4868, "map": 0.2, "ap50": 0.4, "ap_small": 0.1}
    assert read_validation_curve(nested) == [{**expected, "train_minutes": 15.0}]
    assert read_validation_curve(flat) == [{**expected, "train_minutes": 12.0}]
    assert read_validation_curve(tmp_path / "missing") == []


def test_convergence_summary_flags_a_model_still_improving() -> None:
    """Catch a convergence statement that hides a last-epoch gain."""
    curve = [{"epoch": epoch, "map": value} for epoch, value in ((1, 0.05), (2, 0.1), (3, 0.12))]
    history = [
        {"epoch": 2, "loss": 4.0},
        {"epoch": 2, "loss": 2.0},
        {"epoch": 3, "loss": 2.4},
        {"epoch": 3, "loss": float("nan")},
    ]
    summary = summarize_convergence(curve, history)
    assert summary["last_epoch_gain"] == pytest.approx(0.02)
    assert summary["is_still_improving"] is True
    assert summary["best_epoch"] == 3
    assert summary["last_epoch_loss_change"] == pytest.approx(-0.2)
    plateau = summarize_convergence([{"epoch": 1, "map": 0.19}, {"epoch": 2, "map": 0.18}])
    assert plateau["is_still_improving"] is False and plateau["best_epoch"] == 1
    assert summarize_convergence([]) == {"status": "Not run"}


def test_comparison_analysis_marks_missing_detectors_as_not_run(tmp_path: Path) -> None:
    """Catch a comparison that crashes or invents values when a detector has no records."""
    annotation_path = _write_annotations(tmp_path / "gt.json")
    measured = tmp_path / "rt_detr"
    measured.mkdir()
    (measured / "metrics.json").write_text(
        json.dumps([{"epoch": 1, "metrics": {"map": 1.0, "ap50": 1.0, "ap_small": 1.0}}])
    )
    (measured / "val_predictions.json").write_text(json.dumps(_perfect_predictions()))
    detectors = [
        {
            "key": "rt_detr",
            "label": "RT-DETR",
            "family": "query",
            "directory": str(measured),
            "predictions_file": "val_predictions.json",
            "is_query_based": True,
        },
        {
            "key": "missing",
            "label": "Missing",
            "family": "cnn",
            "directory": str(tmp_path / "missing"),
            "predictions_file": "predictions.json",
            "is_query_based": False,
        },
    ]
    result = run_comparison_analysis(annotation_path, detectors, tmp_path / "analysis")
    first, second = result["detectors"]
    assert first["status"] == "Measured"
    assert first["predictions"]["area_bins"]["all"] == pytest.approx(1.0)
    assert [row["map"] for row in first["predictions"]["post_hoc_nms"]] == [
        pytest.approx(1.0),
        pytest.approx(1.0),
    ]
    assert second["status"] == "Not run" and "predictions" not in second
    assert json.loads((tmp_path / "analysis/analysis.json").read_text())["image_count"] == 2
