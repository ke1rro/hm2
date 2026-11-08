import copy
from pathlib import Path

import polars as pl
import pytest
import torch
from PIL import Image
from torchvision.models.detection.transform import resize_boxes

from experiments.input_resolution import resized_dimensions
from experiments.photometric_augmentation import PhotometricDetectionDataset, apply_photometric
from experiments.repeat_factor_sampling import (
    class_presence_statistics,
    image_repeat_factors,
    sample_epoch_indices,
)
from hm2.faster_rcnn import IgnoreAwareDetectionDataset, IgnoreAwareTransform
from hm2.retinanet_experiments import epoch_order


def _tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    frames = pl.DataFrame(
        {
            "image_key": ["train/a/1", "train/a/2", "train/b/1", "train/b/2"],
            "split": ["train"] * 4,
            "clip": ["a", "a", "b", "b"],
            "frame_id": [1, 2, 1, 2],
            "image_id": ["1", "2", "3", "4"],
            "relative_path": ["image.png"] * 4,
            "image_width": [20] * 4,
            "image_height": [10] * 4,
        }
    )
    objects = pl.DataFrame(
        {
            "image_key": ["train/a/1", "train/a/1", "train/a/2", "train/b/1"],
            "category_id": [1, 1, 1, 7],
            "x": [1.0] * 4,
            "y": [2.0] * 4,
            "width": [3.0] * 4,
            "height": [4.0] * 4,
        }
    )
    return frames, objects, objects.head(1).drop("category_id")


def test_photometric_replay_and_geometry(tmp_path: Path) -> None:
    """Keep target and ignored geometry unchanged while changing only float RGB appearance."""
    Image.new("RGB", (20, 10), (80, 110, 150)).save(tmp_path / "image.png")
    frames, objects, ignored = _tables()
    settings = {
        "brightness": [0.85, 1.15],
        "contrast": [0.85, 1.15],
        "saturation": [0.9, 1.1],
        "probability": 1.0,
    }
    baseline = IgnoreAwareDetectionDataset(tmp_path, frames, objects, ignored, 0.5, 2026)
    augmented = PhotometricDetectionDataset(
        tmp_path, frames, objects, ignored, 0.5, 2026, photometric=settings
    )
    image, target = baseline[0]
    changed, actual = augmented[0]
    assert not torch.equal(image, changed)
    assert torch.equal(changed, augmented[0][0])
    assert changed.dtype == torch.float32 and 0 <= changed.min() <= changed.max() <= 1
    assert all(torch.equal(target[key], actual[key]) for key in target)
    disabled = {**settings, "probability": 0.0}
    state = torch.get_rng_state().clone()
    assert torch.equal(image, apply_photometric(image, 2026, disabled))
    assert torch.equal(state, torch.get_rng_state())


def test_resolution_round_trip_including_ignored_boxes() -> None:
    """Use the detector's aspect-ratio resize and restore original pixel coordinates."""
    target = {
        "boxes": torch.tensor([[10.0, 20.0, 80.0, 90.0]]),
        "ignore_boxes": torch.tensor([[1.0, 2.0, 9.0, 8.0]]),
    }
    original = copy.deepcopy(target)
    transform = IgnoreAwareTransform(80, 128, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    transform.eval()
    image, transformed = transform.resize(torch.zeros(3, 100, 200), target)
    assert tuple(image.shape[-2:]) == resized_dimensions(100, 200, 80, 128) == (64, 128)
    assert transformed is not None
    for key in original:
        restored = resize_boxes(transformed[key], list(image.shape[-2:]), [100, 200])
        torch.testing.assert_close(restored, original[key])
    assert resized_dimensions(1080, 1920, 800, 1280) == (720, 1280)


def test_rfs_image_presence_and_fixed_budget() -> None:
    """Ignore repeated annotations and use a deterministic, bounded fixed-length epoch."""
    frames, objects, _ = _tables()
    stats = class_presence_statistics(frames, objects)
    assert stats.filter(pl.col("category_id") == 1)["frame_frequency"].item() == 0.5
    assert stats.filter(pl.col("category_id") == 7)["frame_frequency"].item() == 0.25
    assert stats.filter(pl.col("category_id") == 8)["frame_frequency"].item() == 0
    factors = image_repeat_factors(frames, objects, threshold=1.0, max_repeat=1.5)
    assert factors["repeat_factor"].max() == 1.5
    order = sample_epoch_indices(factors["repeat_factor"].to_list(), 2026, 2)
    assert len(order) == frames.height
    assert order == sample_epoch_indices(factors["repeat_factor"].to_list(), 2026, 2)
    assert order[2:] == sample_epoch_indices(factors["repeat_factor"].to_list(), 2026, 2)[2:]
    assert sample_epoch_indices([1.0] * 4, 2026, 2) == epoch_order(4, 2026, 2)
    assert image_repeat_factors(frames, objects, 0.01)["repeat_factor"].to_list() == [1.0] * 4
    with pytest.raises(ValueError, match="training"):
        class_presence_statistics(frames.with_columns(pl.lit("val").alias("split")), objects)


def test_disabled_recipes_and_missing_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve disabled ablations to the exact baseline and expose honest missing states."""
    from experiments import protocol
    from experiments.protocol import load_records, load_settings
    from hm2.faster_rcnn_experiments import load_experiment_settings

    root = Path(__file__).parents[1]
    baseline = load_experiment_settings(root / "config/faster_rcnn_baseline.toml")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    frames, objects, ignored = _tables()
    for name, table in (("frames", frames), ("objects", objects), ("ignored_regions", ignored)):
        table.write_parquet(metadata / f"{name}.parquet")
    calls = []

    def capture(*args, **kwargs) -> dict:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(protocol, "verify_selection", lambda *args: None)
    monkeypatch.setattr(protocol, "run_experiment", capture)
    for name in ("photometric_augmentation", "input_resolution", "repeat_factor_sampling"):
        disabled = load_settings(root, name, is_enabled=False)
        assert disabled == baseline
        protocol.train_experiment(name, tmp_path, metadata, tmp_path / name, disabled)
        assert calls[-1]["dataset_factory"] is None
        assert calls[-1]["order_factory"] is None
        records = load_records(tmp_path / name)
        assert records["status"] == "Not run" and records["epochs"] == []
    yolo = load_settings(root, "yolo_baseline")
    assert yolo["model"]["backend"] == "ultralytics"
    assert yolo["model"]["model_id"] == "yolo11s.pt"
    assert yolo["loss"] == {"box": 7.5, "cls": 0.5, "dfl": 1.5}


def test_training_hooks_preserve_cursor_budget_and_best_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume a repeated-image order mid-epoch using real atomic optimizer checkpoints."""
    import json

    from hm2 import faster_rcnn_experiments
    from hm2.retinanet import make_optimizer

    class TinyDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.forward = self._forward

        def _forward(self, images: list[torch.Tensor], targets: list[dict]) -> dict:
            return {"loss": self.weight.square() * images[0].mean()}

    def build_state(settings: dict, batches: int, device: torch.device) -> tuple:
        model = TinyDetector()
        optimizer, scheduler = make_optimizer(model, 0.01, batches * 2, 1)
        return model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False)

    Image.new("RGB", (20, 10), (80, 110, 150)).save(tmp_path / "image.png")
    frames, objects, ignored = _tables()
    validation = frames.head(1).with_columns(
        pl.lit("val/v/1").alias("image_key"),
        pl.lit("val").alias("split"),
        pl.lit("v").alias("clip"),
        pl.lit("5").alias("image_id"),
    )
    frames = pl.concat([frames, validation])
    settings = {
        "dataset": {"frame_stride": 1},
        "model": {"pretrained": False},
        "training": {
            "seed": 2026,
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.01,
            "warmup_updates": 1,
            "flip_probability": 0.5,
            "checkpoint_seconds": 0,
            "num_workers": 0,
        },
        "evaluation": {"batch_size": 2, "num_workers": 0},
        "data_experiment": {"name": "sampling_cpu_test"},
    }
    monkeypatch.setattr(faster_rcnn_experiments, "_build_training_state", build_state)
    monkeypatch.setattr(
        faster_rcnn_experiments,
        "predict_loader",
        lambda *args: ([], [5], {"milliseconds_per_frame": 1.0}),
    )
    monkeypatch.setattr(faster_rcnn_experiments, "predictions_to_coco", lambda *args: [])
    scores = iter([0.2, 0.1])
    monkeypatch.setattr(
        faster_rcnn_experiments,
        "evaluate_detections",
        lambda *args: {"max_dets_100": {"map": next(scores)}},
    )
    calls = []

    def factory(root: Path, rows: pl.DataFrame, *args, **kwargs) -> IgnoreAwareDetectionDataset:
        calls.append(rows["image_key"].to_list())
        return IgnoreAwareDetectionDataset(root, rows, *args, **kwargs)

    def order(count: int, seed: int, epoch: int) -> list[int]:
        return [2, 2, 0, 3]

    output = tmp_path / "run"

    def pause(path: Path) -> None:
        status = json.loads((path / "status.json").read_text())
        if status["epochs_completed"] == 0 and status["batch_cursor"] == 1:
            (path / "STOP").touch()

    arguments = (tmp_path, tmp_path, frames, objects, ignored, output, settings, "rfs_test")
    first = faster_rcnn_experiments.run_experiment(
        *arguments, dataset_factory=factory, order_factory=order, on_checkpoint=pause
    )
    assert first["batch_cursor"] == 1
    (output / "STOP").unlink()
    final = faster_rcnn_experiments.run_experiment(
        *arguments, dataset_factory=factory, order_factory=order
    )
    assert final["is_completed"] and final["global_step"] == 4
    assert final["image_exposures"] == 8
    assert calls[1] == ["train/a/1", "train/b/2"]
    best = torch.load(output / "best.pt", weights_only=True)
    assert best["extra_state"]["epochs_completed"] == 1
    assert (
        json.loads((output / "best_metrics.json").read_text())["metrics"]["max_dets_100"]["map"]
        == 0.2
    )


@pytest.mark.parametrize(
    "number,name",
    [
        ("05", "photometric_augmentation"),
        ("06", "input_resolution"),
        ("07", "yolo_baseline"),
        ("08", "repeat_factor_sampling"),
    ],
)
def test_research_notebook_missing_results_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, number: str, name: str
) -> None:
    """Execute every research cell without inputs and forbid accidental training."""
    import IPython.display
    import nbformat

    from experiments import protocol

    def forbidden(*args, **kwargs) -> None:
        raise AssertionError("Research Run All must not launch training")

    monkeypatch.setenv("HM2_METADATA_DIR", str(tmp_path / "missing-metadata"))
    monkeypatch.setenv("HM2_BASELINE_RESULTS", str(tmp_path / "missing-baseline"))
    monkeypatch.setenv("HM2_DATA_EXPERIMENT_OUTPUT", str(tmp_path / "missing-variant"))
    monkeypatch.setattr(protocol, "train_experiment", forbidden)
    monkeypatch.setattr(IPython.display, "display", lambda *args, **kwargs: None)
    notebook = nbformat.read(Path(__file__).parents[1] / f"notebooks/{number}_{name}.ipynb", 4)
    nbformat.validate(notebook)
    namespace: dict = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{name}-research", "exec"), namespace)
    assert namespace["variant"]["status"] == "Not run"
    assert namespace["SHOULD_TRAIN"] is False


@pytest.mark.parametrize("is_inference_only", [False, True])
def test_kaggle_data_staging_cell_uses_original_metadata_and_real_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_inference_only: bool
) -> None:
    """Stage real small images and source annotations in an empty simulated hosted runtime."""
    import logging
    import shutil

    import nbformat

    from hm2.detection_data import select_frames
    from hm2.faster_rcnn_experiments import find_competition_root, stage_images

    source = tmp_path / "input/competition"
    source.mkdir(parents=True)
    pl.DataFrame(
        {"image_id": ["test_1"], "file_name": ["test.png"], "width": [20], "height": [10]}
    ).write_csv(source / "test_images.csv")
    Image.new("RGB", (20, 10)).save(source / "test.png")
    for split in ("train", "val"):
        (source / split).mkdir()
        (source / split / "annotations_coco.json").write_text('{"annotations": []}')
    Image.new("RGB", (20, 10), (80, 110, 150)).save(source / "image.png")
    frames, objects, ignored = _tables()
    validation = frames.head(1).with_columns(
        pl.lit("val/v/1").alias("image_key"),
        pl.lit("val").alias("split"),
        pl.lit("v").alias("clip"),
        pl.lit("5").alias("image_id"),
    )
    test = validation.with_columns(
        pl.lit("test").alias("split"),
        pl.lit("test/1").alias("image_key"),
        pl.lit("test_1").alias("image_id"),
        pl.lit("test.png").alias("relative_path"),
    )
    frames = pl.concat([frames, validation, test])
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for name, table in (("frames", frames), ("objects", objects), ("ignored_regions", ignored)):
        table.write_parquet(metadata / f"{name}.parquet")
    output = tmp_path / "output"
    output.mkdir()
    notebook = nbformat.read(
        Path(__file__).parents[1] / "experiments/03_data_experiments_kaggle.ipynb", 4
    )
    stage = notebook.cells[5].source.replace(
        "/kaggle/temp/data_experiment_images", str(tmp_path / "staged")
    )
    namespace = {
        "Path": Path,
        "pl": pl,
        "torch": torch,
        "shutil": shutil,
        "LOGGER": logging.getLogger(__name__),
        "display": lambda *args: None,
        "find_competition_root": find_competition_root,
        "select_frames": select_frames,
        "verify_selection": lambda *args: None,
        "stage_images": stage_images,
        "METADATA": metadata,
        "KAGGLE_INPUT": tmp_path / "input",
        "LOCAL_BATCH": output,
        "SHOULD_TRAIN": not is_inference_only,
        "SHOULD_EXPORT_SUBMISSIONS": is_inference_only,
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    exec(compile(stage, "kaggle-data-staging", "exec"), namespace)
    staged = namespace["IMAGE_ROOT"]
    assert isinstance(staged, Path)
    if is_inference_only:
        assert (staged / "test.png").read_bytes() == (source / "test.png").read_bytes()
        assert (staged / "test_images.csv").read_bytes() == (
            source / "test_images.csv"
        ).read_bytes()
        assert not (staged / "image.png").exists()
    else:
        assert (staged / "image.png").read_bytes() == (source / "image.png").read_bytes()
        assert not (staged / "test.png").exists()
    for split in ("train", "val"):
        assert (staged / split / "annotations_coco.json").read_bytes() == (
            source / split / "annotations_coco.json"
        ).read_bytes()
    assert pl.read_parquet(metadata / "frames.parquet").equals(frames)


@pytest.mark.parametrize(
    "scenario", ["complete", "skip_completed", "pause", "pending_validation", "deadline"]
)
def test_kaggle_queue_runs_independent_faster_rcnn_experiments(
    tmp_path: Path, scenario: str
) -> None:
    """Run the actual execution cell with bounded training and isolated checkpoint destinations."""
    import logging
    from types import SimpleNamespace

    import nbformat

    from experiments.protocol import load_settings

    names = ("photometric_augmentation", "repeat_factor_sampling", "input_resolution")
    outputs = {name: tmp_path / name for name in names}
    folders = {name: f"drive/{name}" for name in names}
    progress = {}
    if scenario in {"skip_completed", "pending_validation"}:
        progress[names[0]] = {
            "epochs_completed": 5,
            "pending_evaluation": scenario == "pending_validation",
        }
    calls, saves = [], []

    def sync(api: object, output: Path, folder: str) -> None:
        saves.append((output.name, folder))

    def train(
        name: str, root: Path, metadata: Path, output: Path, settings: dict, **kwargs
    ) -> dict:
        calls.append((name, output, settings["model"]["min_size"], kwargs["seconds"]))
        kwargs["on_checkpoint"](output)
        return {"is_completed": not (scenario == "pause" and name == names[1])}

    clock = iter([100.0, 100.0, 110.0, 120.0])
    namespace = {
        "SHOULD_TRAIN": True,
        "SHOULD_EXPORT_SUBMISSIONS": False,
        "EXPERIMENTS": names,
        "SETTINGS_BY_EXPERIMENT": {
            name: load_settings(Path(__file__).parents[1], name) for name in names
        },
        "OUTPUTS": outputs,
        "RESTORED_PROGRESS": progress,
        "RUN_FOLDERS": folders,
        "SESSION_HOURS": 0 if scenario == "deadline" else 10,
        "IMAGE_ROOT": tmp_path,
        "METADATA": tmp_path,
        "Path": Path,
        "LOGGER": logging.getLogger(__name__),
        "time": SimpleNamespace(monotonic=lambda: next(clock)),
        "api": None,
        "display": lambda *args: None,
        "train_experiment": train,
        "sync_checkpoint_to_drive_api": sync,
        "sync_results_to_drive_api": lambda *args: None,
        "publish_snapshot": lambda *args: None,
    }
    notebook = nbformat.read(
        Path(__file__).parents[1] / "experiments/03_data_experiments_kaggle.ipynb", 4
    )
    exec(compile(notebook.cells[9].source, "kaggle-queue", "exec"), namespace)
    expected_names = list(names)
    if scenario == "skip_completed":
        expected_names = list(names[1:])
    elif scenario == "pause":
        expected_names = list(names[:2])
    elif scenario == "deadline":
        expected_names = []
    assert [row[0] for row in calls] == expected_names
    assert all(output == outputs[name] for name, output, _, _ in calls)
    assert all(folder == folders[name] for name, folder in saves)
    assert all(not (output / "probe.json").exists() for output in outputs.values())
    if scenario == "complete":
        assert [row[2] for row in calls] == [640, 640, 800]
        assert [row[3] for row in calls] == [36000, 35990, 35980]


def test_kaggle_execution_default_exports_three_completed_runs() -> None:
    """Keep the execution entry point configured to export the three completed ablations."""
    import ast

    import nbformat

    notebook = nbformat.read(
        Path(__file__).parents[1] / "experiments/03_data_experiments_kaggle.ipynb", 4
    )
    assignments: list[ast.stmt] = [
        node
        for node in ast.parse(notebook.cells[3].source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in {"EXPERIMENTS", "SHOULD_TRAIN", "SHOULD_EXPORT_SUBMISSIONS"}
            for target in node.targets
        )
    ]
    namespace = {}
    exec(
        compile(ast.Module(body=assignments, type_ignores=[]), "launch-controls", "exec"), namespace
    )
    assert namespace["SHOULD_TRAIN"] is False
    assert namespace["SHOULD_EXPORT_SUBMISSIONS"] is True
    assert namespace["EXPERIMENTS"] == (
        "photometric_augmentation",
        "repeat_factor_sampling",
        "input_resolution",
    )


def test_completed_checkpoint_exports_and_reuses_submission_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Export real test rows and pixel predictions, then reuse only a matching finished CSV."""
    import hashlib
    import json

    from experiments import protocol

    class Detector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.forward = self._forward

        def _forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
            return [
                {
                    "boxes": torch.tensor([[1.0, 2.0, 9.0, 8.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([8]),
                }
                for _ in images
            ]

    output = tmp_path / "run"
    output.mkdir()
    Image.new("RGB", (20, 10)).save(tmp_path / "test.png")
    pl.DataFrame(
        {
            "image_id": ["clip_1", "clip_2"],
            "file_name": ["test.png"] * 2,
            "width": [20, 20],
            "height": [10, 10],
        }
    ).write_csv(tmp_path / "test_images.csv")
    settings = {
        "experiment_id": "input_resolution",
        "seed": 2026,
        "epochs": 5,
        "model": {"pretrained": True, "min_size": 800, "max_size": 1280},
    }
    torch.save(
        {
            "model": Detector().state_dict(),
            "config": settings,
            "extra_state": {"epochs_completed": 5, "pending_evaluation": False},
        },
        output / "latest.pt",
    )
    monkeypatch.setattr(protocol, "build_faster_rcnn", lambda **kwargs: Detector())
    record = protocol.export_submission(
        "input_resolution",
        tmp_path,
        output,
        {"batch_size": 2, "num_workers": 0},
        device=torch.device("cpu"),
    )
    csv = output / "faster_rcnn_input_resolution_stride5.csv"
    actual = pl.read_csv(csv)
    assert actual["image_id"].to_list() == ["clip_1", "clip_2"]
    assert actual["PredictionString"][0].split()[0] == "7"
    assert list(map(float, actual["PredictionString"][0].split()[2:])) == [1, 2, 8, 6]
    assert record["images"] == 2 and record["epoch"] == 5
    assert (
        record["checkpoint_sha256"]
        == hashlib.sha256((output / "latest.pt").read_bytes()).hexdigest()
    )

    def forbidden(**kwargs) -> None:
        raise AssertionError("A finished matching export must not initialize another model")

    monkeypatch.setattr(protocol, "build_faster_rcnn", forbidden)
    reused = protocol.export_submission(
        "input_resolution",
        tmp_path,
        output,
        {"batch_size": 2, "num_workers": 0},
        device=torch.device("cpu"),
    )
    assert reused["csv_sha256"] == record["csv_sha256"]
    metadata = json.loads((output / "submission_metadata.json").read_text())
    assert metadata["file"] == csv.name and metadata["official_sample_verified"] is False


def test_kaggle_exports_completed_runs_without_training(tmp_path: Path) -> None:
    """Run inference-only for all completed experiments and copy their CSVs for download."""
    import logging
    import shutil
    from types import SimpleNamespace

    import nbformat

    from experiments.protocol import load_settings

    names = ("photometric_augmentation", "repeat_factor_sampling", "input_resolution")
    outputs = {name: tmp_path / name for name in names}
    for output in outputs.values():
        output.mkdir()
    exports = []

    def forbidden(*args, **kwargs) -> dict:
        raise AssertionError("Inference-only must never call training")

    def export(name: str, root: Path, output: Path, settings: dict) -> dict:
        filename = f"faster_rcnn_{name}_stride5.csv"
        (output / filename).write_text("image_id,PredictionString\nclip_1,\n")
        exports.append(name)
        return {"file": filename, "images": 1}

    notebook = nbformat.read(
        Path(__file__).parents[1] / "experiments/03_data_experiments_kaggle.ipynb", 4
    )
    namespace = {
        "SHOULD_TRAIN": False,
        "SHOULD_EXPORT_SUBMISSIONS": True,
        "EXPERIMENTS": names,
        "OUTPUTS": outputs,
        "SETTINGS_BY_EXPERIMENT": {
            name: load_settings(Path(__file__).parents[1], name) for name in names
        },
        "RESTORED_PROGRESS": {
            name: {"epochs_completed": 5, "pending_evaluation": False} for name in names
        },
        "RUN_FOLDERS": {name: f"drive/{name}" for name in names},
        "SUBMISSIONS_ROOT": tmp_path / "submissions",
        "SESSION_HOURS": 10,
        "IMAGE_ROOT": tmp_path,
        "METADATA": tmp_path,
        "Path": Path,
        "time": SimpleNamespace(monotonic=lambda: 100),
        "shutil": shutil,
        "LOGGER": logging.getLogger(__name__),
        "api": None,
        "display": lambda *args: None,
        "train_experiment": forbidden,
        "export_submission": export,
        "sync_checkpoint_to_drive_api": lambda *args: None,
        "sync_results_to_drive_api": lambda *args: None,
        "publish_snapshot": lambda *args: None,
    }
    exec(compile(notebook.cells[9].source, "kaggle-inference-only", "exec"), namespace)
    assert exports == list(names)
    assert len(list((tmp_path / "submissions").glob("*.csv"))) == 3
