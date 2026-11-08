import copy
import hashlib
import json
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import polars as pl
import torch
from PIL import Image, ImageStat

from experiments.photometric_augmentation import PhotometricDetectionDataset
from experiments.repeat_factor_sampling import image_repeat_factors, sample_epoch_indices
from hm2.detection_data import select_frames
from hm2.experiment_checkpoint import write_json
from hm2.faster_rcnn import build_faster_rcnn
from hm2.faster_rcnn_experiments import (
    export_test_submission,
    load_experiment_settings,
    run_experiment,
    verify_selection,
)
from hm2.submission import validate_submission

LOGGER = logging.getLogger(__name__)
NAMES = (
    "photometric_augmentation",
    "input_resolution",
    "yolo_baseline",
    "repeat_factor_sampling",
)
METRICS = ("map", "ap50", "ap75", "ap_small", "ap_medium", "ap_large")
LABELS = {
    "map": "mAP",
    "ap50": "AP50",
    "ap75": "AP75",
    "ap_small": "AP small",
    "ap_medium": "AP medium",
    "ap_large": "AP large",
}
BATCH_ID = "controlled_data_stride5_5epochs_seed2026"


def load_settings(root: Path, name: str, is_enabled: bool = True) -> dict:
    """Load one visible TOML recipe or reproduce its unchanged baseline when disabled.

    Args:
        root: Project root containing config/.
        name: Descriptive data experiment name.
        is_enabled: Whether to apply the experiment's single changed factor.

    Returns:
        Exact detector-loop settings, including recorded ablation identity when enabled.
    """
    if name not in NAMES:
        raise ValueError(f"Choose an experiment from {NAMES}.")
    if name == "yolo_baseline":
        settings = load_experiment_settings(root / "config/yolo_baseline.toml")
        settings["data_experiment"] = {
            "name": name,
            "kind": "Anchor-free architecture baseline",
            "ignore_policy": "Exclude negative points in ignored regions; preserve positives",
            "ultralytics_version": "8.3.235",
            "best_checkpoint": "Highest full-validation COCO mAP; final epoch is primary",
        }
        return settings
    settings = load_experiment_settings(
        root / (f"config/data_{name}.toml" if is_enabled else "config/faster_rcnn_baseline.toml")
    )
    if is_enabled:
        settings["data_experiment"] = {
            "name": name,
            "best_checkpoint": "Highest full-validation COCO mAP; final epoch is primary",
            **{key: settings[key] for key in ("photometric", "sampling") if key in settings},
        }
    return settings


def baseline_directory(root: Path, name: str) -> Path:
    """Locate the original result records without creating or downloading data.

    Args:
        root: Project root.
        name: Ablation name identifying the corresponding detector family.

    Returns:
        Existing canonical baseline record directory, possibly not restored locally.
    """
    return root / "runs/faster_rcnn/faster_rcnn_stride5_5epochs_seed2026/baseline_stride5"


def load_records(directory: Path) -> dict:
    """Read saved measurements, including partial and missing runs, without model execution.

    Args:
        directory: Local restored or newly generated experiment directory.

    Returns:
        Uniform status, exact config, canonical epoch metrics, history and environment.
    """
    status_path = directory / "status.json"
    if not status_path.is_file():
        return {
            "status": "Not run",
            "config": {},
            "epochs": [],
            "history": [],
            "environment": {},
            "probe": {},
            "compact_history": [],
            "final": None,
            "best": None,
        }
    status = json.loads(status_path.read_text())
    config = json.loads((directory / "config.json").read_text())
    raw = json.loads((directory / "metrics.json").read_text())
    epochs = [{**row, "metrics": row["metrics"].get("max_dets_100", row["metrics"])} for row in raw]
    is_completed = status.get("is_completed", status.get("completed", False))
    available = [row for row in epochs if row["metrics"]["map"] is not None]
    history = directory / "history.json"
    environment = directory / "environment.json"
    compact = directory / "history_summary.json"
    probe = directory / "probe.json"
    return {
        "status": "Completed" if is_completed else "Partial",
        "config": config,
        "epochs": epochs,
        "history": json.loads(history.read_text()) if history.exists() else [],
        "compact_history": json.loads(compact.read_text()) if compact.exists() else [],
        "environment": json.loads(environment.read_text()) if environment.exists() else {},
        "probe": json.loads(probe.read_text()) if probe.exists() else {},
        "final": epochs[-1] if is_completed and epochs else None,
        "best": max(available, key=lambda row: row["metrics"]["map"]) if available else None,
        "progress": status,
    }


def metric_table(records: dict[str, dict]) -> pl.DataFrame:
    """Present final and diagnostic best-epoch metrics without inventing missing values.

    Args:
        records: Short readable run labels mapped to loaded result records.

    Returns:
        Final and best-validation rows with nullable measurements for missing runs.
    """
    rows = []
    for label, record in records.items():
        for selection in ("final", "best"):
            row = record[selection]
            rows.append(
                {
                    "Run": label,
                    "Status": record["status"],
                    "Selection": "Final epoch" if selection == "final" else "Best validation epoch",
                    "Epoch": row["epoch"] if row else None,
                    **{LABELS[key]: row["metrics"].get(key) if row else None for key in METRICS},
                    "Training min": row.get("wall_train_seconds", 0) / 60 if row else None,
                    "Validation ms/frame": (
                        row.get("inference", {}).get("milliseconds_per_frame") if row else None
                    ),
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)


def epoch_table(records: dict[str, dict]) -> pl.DataFrame:
    """Collect canonical per-epoch metrics for comparable detector runs.

    Args:
        records: Readable labels mapped to loaded results.

    Returns:
        Long-form metric rows, empty when no evaluation exists.
    """
    rows = [
        {"Run": label, "Epoch": row["epoch"], "Metric": LABELS[key], "AP": row["metrics"].get(key)}
        for label, record in records.items()
        for row in record["epochs"]
        for key in METRICS
    ]
    return pl.from_dicts(rows) if rows else pl.DataFrame()


def class_table(records: dict[str, dict]) -> pl.DataFrame:
    """Collect per-class AP for the last completed validation, clearly marking its epoch.

    Args:
        records: Readable labels mapped to loaded results.

    Returns:
        Long-form class measurements, empty when no evaluation exists.
    """
    rows = [
        {"Run": label, "Class": name, "AP": value, "Epoch": record["epochs"][-1]["epoch"]}
        for label, record in records.items()
        if record["epochs"]
        for name, value in record["epochs"][-1]["metrics"]["per_class_ap"].items()
    ]
    return pl.from_dicts(rows) if rows else pl.DataFrame()


def loss_table(record: dict) -> pl.DataFrame:
    """Aggregate training batches into small within-epoch windows for readable loss curves.

    Args:
        record: Loaded experiment with original training history.

    Returns:
        Mean loss, training throughput and last LR for each 100-attempt window.
    """
    if not record["history"]:
        if record.get("compact_history"):
            return pl.from_dicts(record["compact_history"]).select(
                "epoch",
                pl.col("step").alias("Updates"),
                pl.col("loss").alias("Loss"),
                pl.col("learning_rate").alias("Learning rate"),
            )
        return pl.DataFrame()
    return (
        pl.from_dicts(record["history"], infer_schema_length=None)
        .with_row_index("index")
        .with_columns((pl.col("index") // 100).alias("window"))
        .group_by("epoch", "window")
        .agg(
            pl.col("step").last().alias("Updates"),
            pl.col("loss").mean().alias("Loss"),
            pl.col("images_per_second").mean().alias("Images/s"),
            pl.col("learning_rate").last().alias("Learning rate"),
        )
        .sort("epoch", "Updates")
    )


def choose_example_frames(
    root: Path,
    selected: pl.DataFrame,
    objects: pl.DataFrame,
) -> list[dict]:
    """Choose dark, crowded and tiny-object training examples without reading all images.

    Args:
        root: Existing local dataset root.
        selected: Selected training metadata.
        objects: Original GT table.

    Returns:
        Three captioned frame records; darkness is relative to a fixed 24-image sample.
    """
    sample = selected.sample(n=min(24, selected.height), seed=2026).to_dicts()
    luminance = []
    for row in sample:
        with Image.open(root / row["relative_path"]) as image:
            image = image.convert("L")
            image.thumbnail((64, 64))
            luminance.append(ImageStat.Stat(image).mean[0])
    smallest = objects.join(selected.select("image_key"), on="image_key", how="semi").sort(
        "area" if "area" in objects.columns else "width"
    )["image_key"][0]
    return [
        {**sample[min(range(len(sample)), key=luminance.__getitem__)], "Reason": "Dark sample"},
        {**selected.sort("target_count", descending=True).row(0, named=True), "Reason": "Crowded"},
        {
            **selected.filter(pl.col("image_key") == smallest).row(0, named=True),
            "Reason": "Small objects",
        },
    ]


def train_experiment(
    name: str,
    root: Path,
    metadata: Path,
    output: Path,
    settings: dict,
    seconds: int = 36000,
    on_checkpoint: Callable[[Path], None] | None = None,
) -> dict:
    """Run one explicitly authorized ablation through its existing detector loop.

    The caller owns the visible training switch, dataset staging and Drive setup. No
    baseline is retrained. New experiment paths must not point to historical batches.

    Args:
        name: One descriptive data experiment name.
        root: Existing fast local dataset with original annotations.
        metadata: Prepared original parquet tables.
        output: Dedicated new experiment directory.
        settings: Exact visible TOML recipe.
        seconds: Bounded session training time before a resumable pause.
        on_checkpoint: Existing synchronous verified Drive persistence callback.

    Returns:
        Actual status from the unmodified detector loop's persistence contract.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Full experiments require explicitly authorized CUDA execution.")
    frames, objects, ignored = (
        pl.read_parquet(metadata / f"{key}.parquet")
        for key in ("frames", "objects", "ignored_regions")
    )
    selected, validation = select_frames(frames, "train", 5), select_frames(frames, "val")
    verify_selection(
        selected,
        validation,
        {
            "expected_selected_frames": 4868,
            "expected_selected_clips": 56,
            "expected_validation_frames": 2846,
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    write_json(
        output / "experiment_source.json",
        {
            "sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "settings": settings,
        },
    )
    effective = copy.deepcopy(settings)
    dataset_factory = None
    order_hook = None
    model_hook = None
    if name == "yolo_baseline":
        from experiments.yolo_baseline import build_yolo

        model_hook = build_yolo
    if name == "photometric_augmentation" and "photometric" in settings:
        dataset_factory = partial(PhotometricDetectionDataset, photometric=settings["photometric"])
    if name == "repeat_factor_sampling" and "sampling" in settings:
        factors = image_repeat_factors(selected, objects, **settings["sampling"])
        weights = factors["repeat_factor"].to_list()
        identity = hashlib.sha256(
            json.dumps(factors.to_dicts(), sort_keys=True).encode()
        ).hexdigest()
        write_json(
            output / "sampling_factors.json", {"sha256": identity, "frames": factors.to_dicts()}
        )

        def weighted_order(count: int, seed: int, epoch: int) -> list[int]:
            return sample_epoch_indices(weights, seed, epoch, count)

        order_hook = weighted_order

    return run_experiment(
        root,
        root,
        frames,
        objects,
        ignored,
        output,
        effective,
        name,
        seconds=seconds,
        on_checkpoint=on_checkpoint,
        dataset_factory=dataset_factory,
        order_factory=order_hook,
        model_factory=model_hook,
    )


def baseline_protocol_table(record: dict, baseline_settings: dict) -> pl.DataFrame:
    """Check restored baseline scientific settings before using its measurements.

    Args:
        record: Loaded original baseline records.
        baseline_settings: Disabled-feature recipe for the same detector.

    Returns:
        Field-wise expected and recorded values, with null checks when not restored.
    """
    if "training" in baseline_settings:
        expected = {
            key: baseline_settings["training"][key]
            for key in (
                "seed",
                "epochs",
                "batch_size",
                "learning_rate",
                "warmup_updates",
                "flip_probability",
            )
        }
        expected.update(
            {
                "frame_stride": baseline_settings["dataset"]["frame_stride"],
                "min_size": baseline_settings["model"]["min_size"],
                "max_size": baseline_settings["model"]["max_size"],
            }
        )
    else:
        expected = {
            key: baseline_settings[key]
            for key in (
                "seed",
                "epochs",
                "batch_size",
                "learning_rate",
                "frame_stride",
                "min_size",
                "max_size",
                "flip_probability",
                "warmup_epochs",
            )
        }
    config = record["config"]
    rows = []
    for key, value in expected.items():
        actual = config.get(key, config.get("model", {}).get(key))
        rows.append(
            {
                "Setting": key,
                "Expected": str(value),
                "Recorded": str(actual) if actual is not None else "Not restored",
                "Matches": actual == value if actual is not None else None,
            }
        )
    validation = record["epochs"][-1]["metrics"] if record["epochs"] else {}
    rows.append(
        {
            "Setting": "Validation frames",
            "Expected": "2846",
            "Recorded": str(validation.get("image_count", "Not restored")),
            "Matches": validation.get("image_count") == 2846 if validation else None,
        }
    )
    return pl.from_dicts(rows)


def runtime_table(records: dict[str, dict]) -> pl.DataFrame:
    """Show measured timing and GPU memory without treating missing history as zero usage.

    Args:
        records: Readable labels mapped to original saved records.

    Returns:
        Per-run training and validation cost, throughput and optional recorded GPU peaks.
    """
    rows = []
    for label, record in records.items():
        progress = record.get("progress", {})
        seconds = progress.get("wall_train_seconds")
        history = record["history"]
        peaks = [
            row["gpu_peak_allocated_bytes"] for row in history if "gpu_peak_allocated_bytes" in row
        ]
        free = [row["gpu_free_bytes"] for row in history if "gpu_free_bytes" in row]
        rows.append(
            {
                "Run": label,
                "Status": record["status"],
                "Training min": seconds / 60 if seconds is not None else None,
                "Validation min": (
                    sum(row["evaluation_seconds"] for row in record["epochs"]) / 60
                    if record["epochs"]
                    else None
                ),
                "Images/s": progress.get("image_exposures", 0) / seconds if seconds else None,
                "Peak allocated MiB": max(peaks) / 1024**2 if peaks else None,
                "Minimum free MiB": min(free) / 1024**2 if free else None,
                "GPU": record["environment"].get("gpu"),
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def export_submission(
    name: str,
    root: Path,
    output: Path,
    evaluation: dict,
    device: torch.device | None = None,
) -> dict:
    """Export a completed final checkpoint on Kaggle and reuse matching finished CSVs.

    Verify the checkpoint and test-table identities, load only model weights and run
    the existing test exporter. Preserve the trained resize and NMS settings. Keep
    generated CSVs and provenance beside checkpoints for verified Drive publication.

    Args:
        name: Descriptive Faster R-CNN experiment name.
        root: Staged test images and original test_images.csv.
        output: Restored experiment checkpoint and result directory.
        evaluation: Existing inference batch size and worker settings.
        device: CUDA by default; an explicit CPU device supports small functional tests.

    Returns:
        Submission counts, checkpoint identity and the descriptive CSV filename.
    """
    checkpoint = output / "latest.pt"
    with checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    test_path = root / "test_images.csv"
    test_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    csv_path = output / f"faster_rcnn_{name}_stride5.csv"
    metadata_path = output / "submission_metadata.json"
    if csv_path.is_file() and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text())
        if (
            previous.get("checkpoint_sha256") == checkpoint_sha256
            and previous.get("test_metadata_sha256") == test_sha256
            and previous.get("csv_sha256") == hashlib.sha256(csv_path.read_bytes()).hexdigest()
        ):
            LOGGER.info("Reuse finished submission for %s", name)
            return previous
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config, progress = state["config"], state["extra_state"]
    if (
        config["experiment_id"] != name
        or progress["epochs_completed"] != config["epochs"]
        or progress["pending_evaluation"]
    ):
        raise ValueError("Submission requires this experiment's completed final checkpoint.")
    test = pl.read_csv(test_path, schema_overrides={"image_id": pl.String})
    if test["image_id"].n_unique() != test.height:
        raise ValueError("Test image identifiers must be unique.")
    frames = test.select(
        pl.lit("test").alias("split"),
        pl.col("image_id").alias("submission_id"),
        pl.col("image_id").alias("image_key"),
        pl.col("file_name").alias("relative_path"),
        pl.col("width").alias("image_width"),
        pl.col("height").alias("image_height"),
    ).with_row_index("image_id")
    selected_device = torch.device("cuda") if device is None else device
    model_arguments: dict = dict(config["model"])
    model_arguments["pretrained"] = False
    model = build_faster_rcnn(**model_arguments, allow_download=False, seed=config["seed"])
    model.load_state_dict(state["model"], strict=True)
    del state
    model.to(selected_device)
    LOGGER.info("Infer %s test images for %s on %s", test.height, name, selected_device)
    record = export_test_submission(model, root, frames, output, evaluation, selected_device)
    generated = output / record["file"]
    sample_path = root / "sample_submission.csv"
    if sample_path.is_file():
        sample = pl.read_csv(sample_path, schema_overrides={"image_id": pl.String})
        submission = pl.read_csv(
            generated, schema_overrides={"image_id": pl.String, "PredictionString": pl.String}
        )
        validation = validate_submission(submission, sample)
        submission = sample.select("image_id").join(
            submission, on="image_id", how="left", maintain_order="left"
        )
        submission.write_csv(generated)
        record.update(validation)
        record["official_sample_verified"] = True
    generated.replace(csv_path)
    record.update(
        file=csv_path.name,
        experiment_id=name,
        epoch=progress["epochs_completed"],
        checkpoint_selection="Configured final epoch",
        checkpoint_sha256=checkpoint_sha256,
        test_metadata_sha256=test_sha256,
        csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        config=config,
    )
    write_json(metadata_path, record)
    LOGGER.info(
        "Saved %s: %s images, %s detections", csv_path, record["images"], record["detections"]
    )
    return record
