import copy
import csv
import inspect
import json
import shutil
from pathlib import Path

import polars as pl
import pytest
import torch
from PIL import Image

from hm2.data import CLASS_NAMES
from hm2.experiment_checkpoint import write_json
from hm2.faster_rcnn import IgnoreAwareRoIHeads
from hm2.faster_rcnn_experiments import (
    RECORDED_MODULES,
    _build_training_state,
    aggregate_loss_curve,
    build_experiment_config,
    compute_source_identity,
    describe_environment,
    export_test_submission,
    find_competition_root,
    find_previous_output,
    hash_sources,
    load_batch_records,
    load_final_model,
    measure_inference_speed,
    prepare_experiment_tables,
    resolve_model_arguments,
    resolve_training_recipe,
    restore_previous_output,
    run_experiment,
    run_timing_probe,
    select_experiment_frames,
    stage_images,
    summarize_experiment,
    verify_selection,
)

WIDTH, HEIGHT = 160, 96
CLIPS = {
    "train": {"uav0000001_00000_v": 3, "uav0000002_00000_v": 3},
    "val": {"uav0000003_00000_v": 2},
}
SETTINGS: dict = {
    "dataset": {
        "frame_stride": 1,
        "expected_selected_frames": 6,
        "expected_selected_clips": 2,
        "expected_validation_frames": 2,
        "should_stage_images": False,
    },
    "model": {"pretrained": False, "min_size": 64, "max_size": 96, "detections_per_img": 50},
    "training": {
        "seed": 2026,
        "epochs": 2,
        "batch_size": 2,
        "num_workers": 0,
        "learning_rate": 0.001,
        "warmup_updates": 2,
        "flip_probability": 0.5,
        "checkpoint_seconds": 600,
    },
    "evaluation": {"batch_size": 2, "num_workers": 0, "speed_batch_size": 1, "speed_frames": 2},
    "probe": {"updates": 2, "validation_frames": 2},
}


@pytest.fixture(autouse=True)
def limited_cpu_threads() -> None:
    """Keep tiny detector tests from oversubscribing the CPU."""
    torch.set_num_threads(2)


def _write_split(root: Path, split: str, clips: dict[str, int]) -> None:
    images, annotations = [], []
    for video_id, (clip, count) in enumerate(clips.items(), 1):
        (root / split / "images" / clip).mkdir(parents=True)
        for frame in range(1, count + 1):
            image_id = len(images) + 1
            name = f"{frame:07d}.jpg"
            Image.new("RGB", (WIDTH, HEIGHT), (frame * 20, 80, 120)).save(
                root / split / "images" / clip / name
            )
            images.append(
                {
                    "id": image_id,
                    "file_name": f"images/{clip}/{name}",
                    "width": WIDTH,
                    "height": HEIGHT,
                    "video_id": video_id,
                    "video": clip,
                    "frame_id": frame,
                }
            )
            for category, box in ((3, [10, 50, 30, 20]), (1, [80, 55, 12, 25])):
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": category,
                        "bbox": box,
                        "area": box[2] * box[3],
                        "iscrowd": 0,
                        "truncation": 0,
                        "occlusion": 0,
                        "track_id": len(annotations) + 1,
                        "orig_category_id": category,
                    }
                )
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [0, 0, WIDTH, 40],
                    "area": WIDTH * 40,
                    "iscrowd": 1,
                    "ignore": 1,
                    "orig_category_id": 0,
                }
            )
    document = {
        "info": {},
        "categories": [{"id": index, "name": name} for index, name in enumerate(CLASS_NAMES, 1)],
        "images": images,
        "annotations": annotations,
    }
    (root / split / "annotations_coco.json").write_text(json.dumps(document))


def _write_competition(root: Path) -> None:
    for split, clips in CLIPS.items():
        _write_split(root, split, clips)
    (root / "test/images/clip_001").mkdir(parents=True)
    rows = []
    for frame in (1, 2):
        name = f"{frame:07d}.jpg"
        Image.new("RGB", (WIDTH, HEIGHT), (200, 100, frame * 30)).save(
            root / "test/images/clip_001" / name
        )
        rows.append(
            [
                f"clip_001_{frame:07d}",
                f"test/images/clip_001/{name}",
                "clip_001",
                frame,
                WIDTH,
                HEIGHT,
            ]
        )
    with (root / "test_images.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "file_name", "clip", "frame", "width", "height"])
        writer.writerows(rows)


@pytest.fixture(scope="module")
def competition(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Build a tiny competition directory and its prepared tables once for the module."""
    root = tmp_path_factory.mktemp("competition")
    _write_competition(root)
    frames, objects, ignored = prepare_experiment_tables(root, root / "metadata")
    return {"root": root, "frames": frames, "objects": objects, "ignored": ignored}


@pytest.fixture(scope="module")
def finished_run(competition: dict, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Train in two sessions, pausing after the first epoch, and keep the records."""
    batch_directory = tmp_path_factory.mktemp("batch")
    output = batch_directory / "baseline_stride5"

    from hm2.colab_drive import restore_checkpoint_from_drive, sync_checkpoint_to_drive

    drive = batch_directory / "drive"

    def persist_to_drive(directory: Path) -> None:
        sync_checkpoint_to_drive(directory, drive)

    arguments = (
        competition["root"],
        competition["root"],
        competition["frames"],
        competition["objects"],
        competition["ignored"],
        output,
        SETTINGS,
        "baseline_stride5",
    )
    first = run_experiment(*arguments, stop_after_epoch=1, on_checkpoint=persist_to_drive)
    shutil.rmtree(output)
    restored = restore_checkpoint_from_drive(drive, output, {})
    assert restored["epochs_completed"] == 1
    assert not restored["pending_evaluation"]
    final = run_experiment(*arguments)
    return {"batch_directory": batch_directory, "output": output, "first": first, "final": final}


def test_resolved_model_arguments_include_defaults_and_reject_unknown_keys() -> None:
    """Catch changed builder defaults escaping the resume check and silent typos."""
    resolved = resolve_model_arguments({"min_size": 640})
    assert resolved["min_size"] == 640
    assert resolved["anchor_sizes"] == [32, 64, 128, 256, 512]
    assert resolved["detections_per_img"] == 500
    assert "seed" not in resolved and "allow_download" not in resolved
    with pytest.raises(ValueError, match="mispelled"):
        resolve_model_arguments({"mispelled": 1})


def test_tables_are_cached_and_selection_follows_the_stride_rule(competition: dict) -> None:
    """Catch a drifting frame selection, wrong test row identifiers and shared flights."""
    frames = competition["frames"]
    test_rows = frames.filter(pl.col("split") == "test")
    assert test_rows["image_id"].to_list() == ["0", "1"]
    assert test_rows["submission_id"].to_list() == ["clip_001_0000001", "clip_001_0000002"]
    cached = prepare_experiment_tables(Path("/nonexistent"), competition["root"] / "metadata")
    assert cached[0].equals(frames)

    selected, validation = select_experiment_frames(frames, {"frame_stride": 2})
    assert selected["frame_id"].to_list() == [1, 3, 1, 3]
    assert validation.height == 2
    full, validation = select_experiment_frames(frames, SETTINGS["dataset"])
    verify_selection(full, validation, SETTINGS["dataset"])
    with pytest.raises(ValueError, match="selected frames"):
        verify_selection(selected, validation, SETTINGS["dataset"])
    overlapping = validation.with_columns(pl.lit("uav0000001_09999_v").alias("clip"))
    with pytest.raises(ValueError, match="uav0000001"):
        verify_selection(full, overlapping, SETTINGS["dataset"])

    keys = full["image_key"].to_list()
    identity = compute_source_identity(competition["root"], keys)
    assert identity != compute_source_identity(competition["root"], keys[:-1])


def test_competition_root_must_be_unique(tmp_path: Path) -> None:
    """Catch an ambiguous or missing attached dataset."""
    with pytest.raises(FileNotFoundError):
        find_competition_root(tmp_path)
    (tmp_path / "competition").mkdir()
    (tmp_path / "competition/test_images.csv").write_text("")
    assert find_competition_root(tmp_path) == tmp_path / "competition"
    (tmp_path / "other/nested").mkdir(parents=True)
    (tmp_path / "other/nested/test_images.csv").write_text("")
    with pytest.raises(FileNotFoundError):
        find_competition_root(tmp_path)


def test_staging_copies_only_listed_images_and_skips_existing(
    competition: dict, tmp_path: Path
) -> None:
    """Catch copying the whole dataset or repeating finished copies."""
    frames = competition["frames"]
    wanted = frames.filter(pl.col("split") == "val")["relative_path"].to_list()
    first = stage_images(competition["root"], tmp_path, wanted)
    assert first["files"] == first["copied_files"] == 2
    assert sorted(path.name for path in tmp_path.rglob("*.jpg")) == ["0000001.jpg", "0000002.jpg"]
    assert not list(tmp_path.rglob("*.incoming"))
    second = stage_images(competition["root"], tmp_path, wanted)
    assert second["copied_files"] == 0 and second["megabytes_per_second"] is None


def test_training_pauses_resumes_and_keeps_the_final_epoch(finished_run: dict) -> None:
    """Catch repeated epochs, a lost cursor, best-on-validation selection and missing records."""
    output = finished_run["output"]
    first, final = finished_run["first"], finished_run["final"]
    assert (first["epochs_completed"], first["global_step"], first["is_completed"]) == (1, 3, False)
    assert not first["is_stopped"]
    assert not first["pending_evaluation"]
    assert (final["epochs_completed"], final["global_step"], final["is_completed"]) == (2, 6, True)
    assert final["image_exposures"] == 12

    metrics = json.loads((output / "metrics.json").read_text())
    assert [row["epoch"] for row in metrics] == [1, 2]
    assert set(metrics[-1]["metrics"]) == {"max_dets_100", "max_dets_500"}
    history = json.loads((output / "history.json").read_text())
    assert [row["step"] for row in history] == [1, 2, 3, 4, 5, 6]
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in history)

    assert (output / "latest.pt").is_file() and not (output / "best.pt").exists()
    config = json.loads((output / "config.json").read_text())
    assert "never chosen on validation" in config["checkpoint_selection"]
    assert json.loads((output / "val_predictions.json").read_text()) is not None
    sessions = json.loads((output / "sessions.json").read_text())
    assert [session["resumed_from_update"] for session in sessions] == [0, 3]
    environment = json.loads((output / "environment.json").read_text())
    assert set(environment["source_sha256"]) == set(RECORDED_MODULES)
    assert json.loads((output / "model.json").read_text())["batch_norm_layers"] > 0


def test_resume_rejects_a_changed_recipe(finished_run: dict, competition: dict) -> None:
    """Catch resuming a checkpoint under different settings."""
    changed = copy.deepcopy(SETTINGS)
    changed["training"]["learning_rate"] = 0.01
    with pytest.raises(ValueError, match="does not match"):
        run_experiment(
            competition["root"],
            competition["root"],
            competition["frames"],
            competition["objects"],
            competition["ignored"],
            finished_run["output"],
            changed,
            "baseline_stride5",
        )
    with pytest.raises(FileExistsError):
        run_experiment(
            competition["root"],
            competition["root"],
            competition["frames"],
            competition["objects"],
            competition["ignored"],
            finished_run["output"],
            SETTINGS,
            "baseline_stride5",
            resume=False,
        )


def test_timing_probe_measures_and_checks_ignore_handling(
    competition: dict, tmp_path: Path
) -> None:
    """Catch a probe that skips the ignore check, writes checkpoints or reports nothing."""
    result = run_timing_probe(
        competition["root"],
        competition["root"],
        competition["frames"],
        competition["objects"],
        competition["ignored"],
        tmp_path,
        SETTINGS,
    )
    assert result["train_images_per_second"] > 0 and result["loader_images_per_second"] > 0
    assert result["ignore_check"]["anchor_negatives_excluded"] > 0
    assert set(result["inference"]) == {"batch_1", "batch_2"}
    assert result["projection"]["total_minutes"] > 0
    assert json.loads((tmp_path / "probe.json").read_text())["updates"] == 2
    assert not list(tmp_path.glob("*.pt"))


def test_final_model_speed_and_unverified_submission(finished_run: dict, competition: dict) -> None:
    """Catch a checkpoint that cannot be restored and submissions that fail validation."""
    device = torch.device("cpu")
    model, config = load_final_model(finished_run["output"], device)
    heads = model.roi_heads
    assert isinstance(heads, IgnoreAwareRoIHeads) and heads.detections_per_img == 50
    assert config["epochs"] == 2
    validation = competition["frames"].filter(pl.col("split") == "val")
    speed = measure_inference_speed(
        model,
        competition["root"],
        validation,
        competition["objects"],
        competition["ignored"],
        SETTINGS["evaluation"],
        device,
    )
    assert speed["batch_size"] == 1 and speed["milliseconds_per_frame"] > 0
    record = export_test_submission(
        model,
        competition["root"],
        competition["frames"],
        finished_run["output"],
        SETTINGS["evaluation"],
        device,
    )
    assert record["images"] == 2 and record["official_sample_verified"] is False
    table = pl.read_csv(
        finished_run["output"] / record["file"], schema_overrides={"image_id": pl.String}
    )
    assert table["image_id"].to_list() == ["clip_001_0000001", "clip_001_0000002"]


def test_records_summarize_not_run_partial_and_completed_states(
    finished_run: dict, tmp_path: Path
) -> None:
    """Catch results tables that hide unfinished or missing experiments."""
    completed = summarize_experiment(finished_run["output"])
    assert completed["status"] == "Completed" and len(completed["epochs"]) == 2
    assert set(completed["per_class_ap_500"]) == set(CLASS_NAMES)
    assert summarize_experiment(tmp_path / "missing")["status"] == "Not run"

    partial = tmp_path / "batch/partial_run"
    partial.mkdir(parents=True)
    for name in ("status.json", "config.json", "metrics.json"):
        (partial / name).write_text((finished_run["output"] / name).read_text())
    status = json.loads((partial / "status.json").read_text())
    write_json(partial / "status.json", {**status, "is_completed": False})
    (tmp_path / "batch/_timing_probe").mkdir()
    write_json(tmp_path / "batch/_timing_probe/probe.json", {"updates": 3})
    batch = load_batch_records(tmp_path / "batch")
    assert [item["status"] for item in batch["experiments"]] == ["Partial"]
    assert batch["probe"] == {"updates": 3}


def test_previous_output_is_found_and_restored_without_overwriting(tmp_path: Path) -> None:
    """Catch resuming from the wrong directory or replacing a newer checkpoint."""
    source = tmp_path / "input/previous_version/runs/faster_rcnn/batch/baseline_stride5"
    source.mkdir(parents=True)
    (source / "latest.pt").write_bytes(b"old")
    assert find_previous_output(tmp_path / "input", "batch", "baseline_stride5") == source
    assert find_previous_output(tmp_path / "input", "other", "baseline_stride5") is None
    destination = tmp_path / "working/baseline_stride5"
    assert restore_previous_output(source, destination) is True
    (destination / "latest.pt").write_bytes(b"newer")
    assert restore_previous_output(source, destination) is False
    assert (destination / "latest.pt").read_bytes() == b"newer"


def test_source_hashes_cover_the_shared_modules() -> None:
    """Catch run records that omit the shared training loop or evaluator."""
    hashes = hash_sources()
    assert set(hashes) == set(RECORDED_MODULES)
    assert "hm2.retinanet" in hashes and all(len(value) == 64 for value in hashes.values())


def test_loss_curve_windows_do_not_cross_epochs(finished_run: dict) -> None:
    """Catch loss curves that blend the end of one epoch with the start of the next."""
    history = json.loads((finished_run["output"] / "history.json").read_text())
    curve = aggregate_loss_curve(history, window_size=2)
    assert curve["epoch"].to_list() == [1, 1, 2, 2]
    assert curve["batches"].to_list() == [2, 1, 1, 2]
    assert curve["step"].to_list() == [2, 3, 4, 6]
    assert curve["loss"].is_finite().all()


def test_default_recipe_and_builder_are_the_faster_rcnn_baseline() -> None:
    """Catch the shared-loop refactor changing what the B1 run records, builds or hashes."""
    keys = ["train/clip/0000001", "train/clip/0000006"]
    config, builder = resolve_training_recipe(SETTINGS, "baseline_stride5", keys, "digest")
    assert config == build_experiment_config(SETTINGS, "baseline_stride5", keys, "digest")
    assert builder is _build_training_state
    custom = {"experiment_id": "other"}

    def other_builder(settings: dict, batches: int, device: torch.device) -> tuple:
        return ()

    assert resolve_training_recipe(SETTINGS, "other", keys, "digest", custom, other_builder) == (
        custom,
        other_builder,
    )
    run_parameters = inspect.signature(run_experiment).parameters
    assert run_parameters["experiment_config"].default is None
    assert run_parameters["build_training_state"].default is None
    assert run_parameters["describe_step"].default is None
    assert run_parameters["recorded_modules"].default == RECORDED_MODULES
    assert inspect.signature(describe_environment).parameters["modules"].default == (
        RECORDED_MODULES
    )
    assert inspect.signature(find_previous_output).parameters["family"].default == "faster_rcnn"
    assert hash_sources() == hash_sources(RECORDED_MODULES)
