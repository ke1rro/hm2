import hashlib
import importlib
import inspect
import json
import logging
import math
import platform
import shutil
import statistics
import time
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import torch
import torchvision
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.models.detection import FasterRCNN

from hm2.data import load_tables
from hm2.dense_evaluation import evaluate_detections
from hm2.detection_data import make_loader, select_frames
from hm2.evaluation import predictions_to_coco
from hm2.experiment_checkpoint import load_checkpoint, publish_snapshot, save_checkpoint, write_json
from hm2.faster_rcnn import (
    IgnoreAwareDetectionDataset,
    build_faster_rcnn,
    summarize_ignore_assignment,
)
from hm2.retinanet import make_optimizer, predict_loader, train_steps
from hm2.retinanet_experiments import epoch_order
from hm2.submission import build_submission, validate_submission
from hm2.tiny_subset import flight_prefix

LOGGER = logging.getLogger(__name__)

EXPERIMENT_LABELS = {"baseline_stride5": "Faster R-CNN baseline (every fifth frame)"}
RECORDED_MODULES = (
    "hm2.data",
    "hm2.dense_evaluation",
    "hm2.detection_data",
    "hm2.evaluation",
    "hm2.experiment_checkpoint",
    "hm2.faster_rcnn",
    "hm2.faster_rcnn_experiments",
    "hm2.retinanet",
    "hm2.retinanet_experiments",
    "hm2.submission",
    "hm2.tiny_subset",
)
TABLE_NAMES = ("frames", "objects", "ignored_regions")
LOSS_COLUMNS = ("loss", "loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg")
EMPTY_OBJECT_SCHEMA = {
    "image_key": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "width": pl.Float64,
    "height": pl.Float64,
    "category_id": pl.Int64,
}


def experiment_label(name: str) -> str:
    """Return the readable name used in notebook tables and captions.

    Args:
        name: Descriptive experiment name, such as baseline_stride5.

    Returns:
        The readable label, or the name itself when it has no label.
    """
    return EXPERIMENT_LABELS.get(name, name)


def load_experiment_settings(path: Path) -> dict:
    """Read an experiment configuration with dataset, model, training and other tables.

    Args:
        path: TOML file in the config directory.

    Returns:
        The configuration tables keyed by table name.
    """
    return tomllib.loads(path.read_text())


def resolve_model_arguments(overrides: dict) -> dict:
    """Merge configured model settings with the builder defaults into one explicit record.

    The result is part of the experiment identity, so a changed default is detected on
    resume. Unknown keys fail early, because a misspelled setting would be ignored.

    Args:
        overrides: The model table of an experiment configuration.

    Returns:
        Every build_faster_rcnn argument except the download switch and the seed.
    """
    parameters = inspect.signature(build_faster_rcnn).parameters
    recorded = {
        name: value for name, value in parameters.items() if name not in {"allow_download", "seed"}
    }
    unknown = set(overrides) - set(recorded)
    if unknown:
        raise ValueError(f"Unknown model settings: {sorted(unknown)}")
    resolved = {name: overrides.get(name, value.default) for name, value in recorded.items()}
    return {
        name: list(value) if isinstance(value, tuple) else value for name, value in resolved.items()
    }


def find_competition_root(input_root: Path) -> Path:
    """Find the attached competition data without scanning every file.

    Args:
        input_root: The read-only input directory of the hosted notebook.

    Returns:
        The single directory, one or two levels down, that holds test_images.csv.
    """
    candidates = sorted(
        {
            path.parent
            for pattern in ("*/test_images.csv", "*/*/test_images.csv")
            for path in input_root.glob(pattern)
        }
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one competition dataset in {input_root}: {candidates}")
    return candidates[0]


def prepare_experiment_tables(
    root: Path, metadata_directory: Path
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build the frame, target and ignored-region tables once and reuse them afterwards.

    The tables come from the shared load_tables, so they match the ones the RetinaNet
    runs used. Test rows get a numeric image_id and keep the submission identifier in
    submission_id, because the loader and the evaluator need integer image ids.

    Args:
        root: Competition directory with the train and val COCO files and test_images.csv.
        metadata_directory: Writable directory that caches the three parquet tables.

    Returns:
        Frames, target objects and ignored regions in original-image pixels.
    """
    paths = [metadata_directory / f"{name}.parquet" for name in TABLE_NAMES]
    if all(path.is_file() for path in paths):
        frames, objects, ignored = (pl.read_parquet(path) for path in paths)
        return frames, objects, ignored
    frames, objects, ignored = load_tables(root)
    is_test = pl.col("split") == "test"
    labeled = frames.filter(~is_test).with_columns(
        pl.lit(None, dtype=pl.String).alias("submission_id")
    )
    test = frames.filter(is_test).with_columns(
        pl.col("image_id").alias("submission_id"),
        pl.int_range(pl.len()).cast(pl.String).alias("image_id"),
    )
    frames = pl.concat([labeled, test], how="vertical_relaxed")
    metadata_directory.mkdir(parents=True, exist_ok=True)
    for table, path in zip((frames, objects, ignored), paths, strict=True):
        table.write_parquet(path)
    return frames, objects, ignored


def select_experiment_frames(
    frames: pl.DataFrame, dataset_settings: dict
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Select the training frames by the shared stride rule and all validation frames.

    Args:
        frames: Prepared frame metadata.
        dataset_settings: The dataset table, with frame_stride.

    Returns:
        The selected training frames and the complete official validation frames.
    """
    return (
        select_frames(frames, "train", dataset_settings["frame_stride"]),
        select_frames(frames, "val"),
    )


def verify_selection(
    selected: pl.DataFrame, validation: pl.DataFrame, dataset_settings: dict
) -> None:
    """Check the selection against the expected counts and the flight separation.

    Args:
        selected: Selected training frames.
        validation: Official validation frames.
        dataset_settings: The dataset table, with the expected_* counts.
    """
    checks = {
        "selected frames": (selected.height, dataset_settings["expected_selected_frames"]),
        "selected clips": (
            selected["clip"].n_unique(),
            dataset_settings["expected_selected_clips"],
        ),
        "validation frames": (validation.height, dataset_settings["expected_validation_frames"]),
    }
    wrong = {name: pair for name, pair in checks.items() if pair[0] != pair[1]}
    if wrong:
        raise ValueError(f"Selection differs from the protocol (found, expected): {wrong}")
    shared = {flight_prefix(clip) for clip in selected["clip"]} & {
        flight_prefix(clip) for clip in validation["clip"]
    }
    if shared:
        raise ValueError(f"Train and validation share flights: {sorted(shared)}")


def stage_images(source_root: Path, destination_root: Path, relative_paths: list[str]) -> dict:
    """Copy only the listed images to fast local storage, keeping their relative paths.

    Files that already exist with the same size are skipped, so an interrupted copy can
    simply be repeated. Each file is copied under a temporary name and then renamed.

    Args:
        source_root: Directory holding the images, possibly read-only network storage.
        destination_root: Writable local directory that receives the same layout.
        relative_paths: Image paths relative to both roots.

    Returns:
        File counts, megabytes and seconds, and the copy rate in megabytes per second.
    """
    started = time.perf_counter()
    copied_files, copied_bytes, total_bytes = 0, 0, 0
    unique_paths = sorted(set(relative_paths))
    for relative_path in unique_paths:
        source, target = source_root / relative_path, destination_root / relative_path
        size = source.stat().st_size
        total_bytes += size
        if target.is_file() and target.stat().st_size == size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".incoming")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        copied_files += 1
        copied_bytes += size
    seconds = time.perf_counter() - started
    return {
        "files": len(unique_paths),
        "copied_files": copied_files,
        "megabytes": total_bytes / 1e6,
        "copied_megabytes": copied_bytes / 1e6,
        "seconds": seconds,
        "megabytes_per_second": copied_bytes / 1e6 / seconds if copied_bytes else None,
    }


def hash_sources() -> dict[str, str]:
    """Hash every source file that shapes a run, as a stand-in for a commit identifier.

    Returns:
        SHA256 digests keyed by module name, covering the shared RetinaNet modules too.
    """
    return {
        name: hashlib.sha256(
            Path(inspect.getfile(importlib.import_module(name))).read_bytes()
        ).hexdigest()
        for name in RECORDED_MODULES
    }


def compute_source_identity(root: Path, selected_keys: list[str]) -> str:
    """Identify the annotation files and the exact training selection.

    Args:
        root: Directory that may hold train and val COCO files.
        selected_keys: Ordered image keys of the training selection.

    Returns:
        A SHA256 digest that changes when the annotations or the selection change.
    """
    identity = hashlib.sha256()
    for name in ("train/annotations_coco.json", "val/annotations_coco.json"):
        path = root / name
        if path.is_file():
            with path.open("rb") as stream:
                identity.update(hashlib.file_digest(stream, "sha256").digest())
    identity.update(json.dumps(selected_keys).encode())
    return identity.hexdigest()


def build_experiment_config(
    settings: dict, experiment_name: str, selected_keys: list[str], source_identity: str
) -> dict:
    """Describe the recipe that a checkpoint must match before a run may resume.

    Runtime knobs such as worker counts and checkpoint timing are left out, so changing
    them never blocks a resume.

    Args:
        settings: The complete experiment configuration.
        experiment_name: Descriptive experiment name.
        selected_keys: Ordered image keys of the training selection.
        source_identity: Digest from compute_source_identity.

    Returns:
        A JSON-compatible dictionary of lists, numbers and strings.
    """
    training = settings["training"]
    return {
        "experiment_id": experiment_name,
        "seed": training["seed"],
        "epochs": training["epochs"],
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
        "warmup_updates": training["warmup_updates"],
        "flip_probability": training["flip_probability"],
        "frame_stride": settings["dataset"]["frame_stride"],
        "selected_frame_count": len(selected_keys),
        "dataset_selection_sha256": source_identity,
        "model": resolve_model_arguments(settings["model"]),
        "optimizer": {"backend": "SGD", "momentum": 0.9, "weight_decay": 1e-4},
        "loss": {
            "rpn_objectness": "binary_cross_entropy",
            "rpn_box_regression": "smooth_l1",
            "roi_classification": "cross_entropy",
            "roi_box_regression": "smooth_l1",
        },
        "checkpoint_selection": "final epoch, never chosen on validation",
    }


def summarize_model(model: torch.nn.Module) -> dict:
    """Count trainable parameters and the BatchNorm layers that train with batch statistics.

    Args:
        model: Detector to inspect.

    Returns:
        Parameter and BatchNorm counts for the run records.
    """
    batch_norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "batch_norm_layers": len(batch_norms),
        "batch_norm_layers_with_frozen_weights": sum(
            not next(module.parameters()).requires_grad for module in batch_norms
        ),
    }


def describe_environment(device: torch.device) -> dict:
    """Record software, hardware and source hashes for one execution.

    Args:
        device: Device that trains the model.

    Returns:
        A JSON-compatible description of the runtime.
    """
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_sha256": hash_sources(),
        "reproducibility": "Fixed epoch order and frame flips; CUDA may be nondeterministic.",
    }


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _attach_execution_log(output: Path, experiment_name: str) -> logging.Logger:
    logger = logging.getLogger(f"{__name__}.{experiment_name}")
    log_path = (output / "execution.log").resolve()
    if not any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path)
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _write_compact_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _new_progress() -> dict:
    return {
        "epochs_completed": 0,
        "batch_cursor": 0,
        "global_step": 0,
        "image_exposures": 0,
        "history": [],
        "metrics": [],
        "wall_train_seconds": 0.0,
        "pending_evaluation": False,
    }


def _progress_signature(progress: dict) -> tuple:
    return (
        progress["epochs_completed"],
        progress["batch_cursor"],
        progress["global_step"],
        len(progress["metrics"]),
        progress["pending_evaluation"],
    )


def _record_session(output: Path, environment: dict, resumed_step: int) -> None:
    path = output / "sessions.json"
    sessions = json.loads(path.read_text()) if path.is_file() else []
    sessions.append(
        {
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "resumed_from_update": resumed_step,
            **{key: value for key, value in environment.items() if key != "source_sha256"},
        }
    )
    write_json(path, sessions)


def _build_training_state(
    settings: dict, batches: int, device: torch.device
) -> tuple[FasterRCNN, Optimizer, LRScheduler, torch.amp.GradScaler]:
    training = settings["training"]
    model = build_faster_rcnn(allow_download=True, seed=training["seed"], **settings["model"])
    model.to(device)
    optimizer, scheduler = make_optimizer(
        model,
        training["learning_rate"],
        batches * training["epochs"],
        training["warmup_updates"],
    )
    return model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def run_experiment(
    image_root: Path,
    source_root: Path,
    frames: pl.DataFrame,
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
    output: Path,
    settings: dict,
    experiment_name: str,
    resume: bool = True,
    seconds: float | None = None,
    on_checkpoint: Callable[[Path], None] | None = None,
) -> dict:
    """Train for the configured epochs, validate after each one and keep resumable records.

    The loop mirrors the RetinaNet protocol: a seeded epoch order, SGD with warmup and two
    step decays, mixed precision on CUDA, and a checkpoint at least every checkpoint_seconds
    and at every epoch boundary. The model after the final epoch is the result; no epoch is
    chosen on validation, and per-epoch validation is recorded only as information. A rerun
    restores latest.pt, rejects a changed recipe, skips finished epochs, and finishes
    pending validation. Creating a STOP file in the output directory pauses the run.

    Args:
        image_root: Directory that holds the images, local or staged.
        source_root: Directory that holds the COCO annotation files.
        frames: Prepared frame metadata.
        objects: Prepared target objects.
        ignored: Prepared ignored regions.
        output: Experiment directory for checkpoints and JSON records.
        settings: The complete experiment configuration.
        experiment_name: Descriptive experiment name.
        resume: Whether an existing checkpoint may be restored.
        seconds: Optional training time budget for this call.
        on_checkpoint: Optional hook called with the output directory after each checkpoint.

    Returns:
        The run status, including whether every epoch and its validation are complete.
    """
    output.mkdir(parents=True, exist_ok=True)
    logger = _attach_execution_log(output, experiment_name)
    training, evaluation = settings["training"], settings["evaluation"]
    selected, validation = select_experiment_frames(frames, settings["dataset"])
    selected_keys = selected["image_key"].to_list()
    config = build_experiment_config(
        settings,
        experiment_name,
        selected_keys,
        compute_source_identity(source_root, selected_keys),
    )
    device = _select_device()
    batches = math.ceil(selected.height / training["batch_size"])
    model, optimizer, scheduler, scaler = _build_training_state(settings, batches, device)
    progress = _new_progress()
    latest = output / "latest.pt"
    saved_signature = None
    if latest.exists():
        if not resume:
            raise FileExistsError("Existing experiment checkpoint requires resume=True.")
        progress = load_checkpoint(latest, model, optimizer, scheduler, scaler, config)[
            "extra_state"
        ]
        saved_signature = _progress_signature(progress)
    environment = describe_environment(device)
    write_json(output / "config.json", config)
    write_json(output / "model.json", summarize_model(model))
    write_json(output / "environment.json", environment)
    write_json(
        output / "selected_frames.json",
        {"train": selected_keys, "val": validation["image_key"].to_list()},
    )
    _record_session(output, environment, progress["global_step"])
    started = time.monotonic()
    last_save = started

    def status_record() -> dict:
        return {
            **{key: value for key, value in progress.items() if key not in {"history", "metrics"}},
            "experiment_id": experiment_name,
            "train_frames": selected.height,
            "is_completed": progress["epochs_completed"] == training["epochs"]
            and not progress["pending_evaluation"],
            "is_stopped": (output / "STOP").exists(),
            "chunk_seconds": time.monotonic() - started,
        }

    def should_pause() -> bool:
        return (output / "STOP").exists() or (
            seconds is not None and time.monotonic() - started >= seconds
        )

    def persist() -> None:
        nonlocal saved_signature, last_save
        signature = _progress_signature(progress)
        if signature == saved_signature:
            return
        save_checkpoint(
            latest, model, optimizer, scheduler, progress["global_step"], config, scaler, progress
        )
        write_json(output / "history.json", progress["history"])
        write_json(output / "metrics.json", progress["metrics"])
        write_json(output / "status.json", status_record())
        logger.info(
            "Checkpoint epoch=%s cursor=%s updates=%s",
            progress["epochs_completed"],
            progress["batch_cursor"],
            progress["global_step"],
        )
        for handler in logger.handlers:
            handler.flush()
        if on_checkpoint is not None:
            publish_snapshot(output)
            on_checkpoint(output)
        saved_signature = signature
        last_save = time.monotonic()

    if saved_signature is not None and on_checkpoint is not None:
        publish_snapshot(output)
        on_checkpoint(output)
    persist()
    while progress["epochs_completed"] < training["epochs"] or progress["pending_evaluation"]:
        if progress["pending_evaluation"]:
            epoch_number = progress["epochs_completed"]
            logger.info("Validate epoch %s on %s frames", epoch_number, validation.height)
            evaluation_started = time.monotonic()
            validation_dataset = IgnoreAwareDetectionDataset(
                image_root, validation, objects, ignored
            )
            validation_loader = make_loader(
                validation_dataset, evaluation["batch_size"], evaluation["num_workers"], device
            )
            predictions, image_ids, inference = predict_loader(model, validation_loader, device)
            records = predictions_to_coco(predictions, image_ids)
            metrics = evaluate_detections(
                source_root / "val/annotations_coco.json", records, image_ids
            )
            row = {
                "epoch": epoch_number,
                "metrics": metrics,
                "inference": inference,
                "evaluation_seconds": time.monotonic() - evaluation_started,
                "global_step": progress["global_step"],
                "image_exposures": progress["image_exposures"],
                "wall_train_seconds": progress["wall_train_seconds"],
            }
            progress["metrics"].append(row)
            progress["pending_evaluation"] = False
            write_json(output / f"epoch_{epoch_number}_metrics.json", row)
            if epoch_number == training["epochs"]:
                _write_compact_json(output / "val_predictions.json", records)
            del predictions, records
            persist()
            if epoch_number == training["epochs"]:
                break
        if should_pause():
            break
        epoch = progress["epochs_completed"]
        ordered = selected[epoch_order(selected.height, training["seed"], epoch)]
        remaining = ordered.slice(progress["batch_cursor"] * training["batch_size"])
        loader = make_loader(
            IgnoreAwareDetectionDataset(
                image_root,
                remaining,
                objects,
                ignored,
                training["flip_probability"],
                augmentation_seed=training["seed"] + epoch,
            ),
            training["batch_size"],
            training["num_workers"],
            device,
            seed=training["seed"] + epoch,
        )

        def on_step(row: dict) -> bool:
            progress["history"].append(
                {**row, "epoch": epoch + 1, "attempt": len(progress["history"]) + 1}
            )
            progress["batch_cursor"] += 1
            progress["global_step"] = row["step"]
            progress["image_exposures"] += row["image_count"]
            progress["wall_train_seconds"] += row["seconds"]
            if progress["global_step"] % 100 == 0:
                logger.info(
                    "epoch=%s update=%s loss=%.4f lr=%.6f images_per_second=%.2f",
                    epoch + 1,
                    progress["global_step"],
                    row["loss"],
                    row["learning_rate"],
                    row["images_per_second"],
                )
            is_pausing = should_pause()
            if is_pausing or time.monotonic() - last_save >= training["checkpoint_seconds"]:
                persist()
            return is_pausing

        train_steps(
            model,
            loader,
            optimizer,
            scheduler,
            device,
            max_steps=batches,
            start_step=progress["global_step"],
            scaler=scaler,
            max_batches=batches - progress["batch_cursor"],
            on_step=on_step,
        )
        if progress["batch_cursor"] != batches:
            break
        progress["epochs_completed"] += 1
        progress["batch_cursor"] = 0
        progress["pending_evaluation"] = True
        persist()
    persist()
    write_json(output / "status.json", status_record())
    return status_record()


def select_probe_frames(
    selected: pl.DataFrame, validation: pl.DataFrame, ignored: pl.DataFrame, settings: dict
) -> dict[str, pl.DataFrame]:
    """Choose the few frames the timing probe uses, deterministically.

    Args:
        selected: Selected training frames.
        validation: Official validation frames.
        ignored: Prepared ignored regions.
        settings: The complete experiment configuration, with the probe table.

    Returns:
        The first training frames of epoch zero, the selected frame with the most ignored
        regions, and validation frames spread evenly over all clips.
    """
    probe, training = settings["probe"], settings["training"]
    order = epoch_order(selected.height, training["seed"], 0)[
        : probe["updates"] * training["batch_size"]
    ]
    most_ignored = (
        ignored.join(selected.select("image_key"), on="image_key", how="semi")
        .group_by("image_key")
        .len()
        .sort(["len", "image_key"], descending=[True, False])["image_key"][0]
    )
    step = max(1, validation.height // probe["validation_frames"])
    return {
        "train": selected[order],
        "ignore": selected.filter(pl.col("image_key") == most_ignored),
        "validation": validation.gather_every(step).head(probe["validation_frames"]),
    }


def run_timing_probe(
    image_root: Path,
    source_root: Path,
    frames: pl.DataFrame,
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
    output: Path,
    settings: dict,
) -> dict:
    """Measure the real cost of the experiment on this machine with a few updates.

    Train a handful of updates with the real recipe, then time inference at batch size 1 and
    at the validation batch size, evaluate once, and check that ignore-aware sampling still
    excludes background and keeps positives under mixed precision. Nothing is checkpointed.
    The projections are linear extrapolations from these short measurements, not results.

    Args:
        image_root: Directory that holds the images, local or staged.
        source_root: Directory that holds the COCO annotation files.
        frames: Prepared frame metadata.
        objects: Prepared target objects.
        ignored: Prepared ignored regions.
        output: Directory that receives probe.json.
        settings: The complete experiment configuration.

    Returns:
        The measurements and projections that were also written to probe.json.
    """
    training, evaluation, probe = settings["training"], settings["evaluation"], settings["probe"]
    selected, validation = select_experiment_frames(frames, settings["dataset"])
    probe_frames = select_probe_frames(selected, validation, ignored, settings)
    device = _select_device()
    batches = math.ceil(selected.height / training["batch_size"])
    model, optimizer, scheduler, scaler = _build_training_state(settings, batches, device)
    loader = make_loader(
        IgnoreAwareDetectionDataset(
            image_root,
            probe_frames["train"],
            objects,
            ignored,
            training["flip_probability"],
            augmentation_seed=training["seed"],
        ),
        training["batch_size"],
        training["num_workers"],
        device,
        seed=training["seed"],
    )
    decode_started = time.perf_counter()
    decoded = sum(len(images) for images, _ in loader)
    decode_images_per_second = decoded / (time.perf_counter() - decode_started)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = train_steps(
        model, loader, optimizer, scheduler, device, probe["updates"], scaler=scaler
    )
    steady = history[min(3, len(history) - 1) :]
    images_per_second = statistics.median(row["images_per_second"] for row in steady)
    ignore_dataset = IgnoreAwareDetectionDataset(
        image_root, probe_frames["ignore"], objects, ignored
    )
    image, target = ignore_dataset[0]
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        ignore_summary = summarize_ignore_assignment(
            model, image.to(device), {key: value.to(device) for key, value in target.items()}
        )
    if (
        ignore_summary["anchor_negatives_excluded"] <= 0
        or ignore_summary["anchor_positives_aware"] != ignore_summary["anchor_positives_stock"]
        or ignore_summary["proposal_positives_aware"] != ignore_summary["proposal_positives_stock"]
    ):
        raise RuntimeError(f"Ignore-aware sampling failed its check: {ignore_summary}")
    validation_dataset = IgnoreAwareDetectionDataset(
        image_root, probe_frames["validation"], objects, ignored
    )
    inference = {}
    for batch_size in sorted({1, evaluation["batch_size"]}):
        validation_loader = make_loader(
            validation_dataset, batch_size, evaluation["num_workers"], device
        )
        predictions, image_ids, inference[f"batch_{batch_size}"] = predict_loader(
            model, validation_loader, device
        )
    evaluation_started = time.perf_counter()
    records = predictions_to_coco(predictions, image_ids)
    evaluate_detections(source_root / "val/annotations_coco.json", records, image_ids)
    evaluation_seconds_per_frame = (time.perf_counter() - evaluation_started) / len(image_ids)
    validation_ms = inference[f"batch_{evaluation['batch_size']}"]["milliseconds_per_frame"]
    train_minutes = training["epochs"] * batches * training["batch_size"] / images_per_second / 60
    validation_minutes = (
        training["epochs"]
        * validation.height
        * (validation_ms / 1000 + evaluation_seconds_per_frame)
        / 60
    )
    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "updates": probe["updates"],
        "batch_size": training["batch_size"],
        "losses_are_finite": True,
        "train_images_per_second": images_per_second,
        "train_seconds_per_update": training["batch_size"] / images_per_second,
        "loader_images_per_second": decode_images_per_second,
        "peak_gpu_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "ignore_check": ignore_summary,
        "inference": inference,
        "evaluation_seconds_per_frame": evaluation_seconds_per_frame,
        "detections_per_frame_mean": len(records) / len(image_ids),
        "projection": {
            "train_minutes": train_minutes,
            "validation_minutes": validation_minutes,
            "total_minutes": train_minutes + validation_minutes,
            "note": "Linear extrapolation from the probe; an untrained model emits more boxes.",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "probe.json", result)
    return result


def load_final_model(output: Path, device: torch.device) -> tuple[FasterRCNN, dict]:
    """Restore the model saved in latest.pt without downloading any weights.

    Args:
        output: Experiment directory with latest.pt.
        device: Device that receives the model.

    Returns:
        The model in evaluation mode and the configuration saved with the checkpoint.
    """
    state = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
    config = state["config"]
    model_arguments: dict = {**config["model"], "pretrained": False}
    model = build_faster_rcnn(**model_arguments)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), config


def measure_inference_speed(
    model: torch.nn.Module,
    image_root: Path,
    validation: pl.DataFrame,
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
    evaluation_settings: dict,
    device: torch.device,
) -> dict:
    """Time end-to-end inference on validation frames spread over all clips.

    Args:
        model: Trained detector.
        image_root: Directory that holds the images, local or staged.
        validation: Official validation frames.
        objects: Prepared target objects.
        ignored: Prepared ignored regions.
        evaluation_settings: The evaluation table, with speed_frames and speed_batch_size.
        device: Device that runs the model.

    Returns:
        The timing record from predict_loader.
    """
    step = max(1, validation.height // evaluation_settings["speed_frames"])
    subset = validation.gather_every(step).head(evaluation_settings["speed_frames"])
    loader = make_loader(
        IgnoreAwareDetectionDataset(image_root, subset, objects, ignored),
        evaluation_settings["speed_batch_size"],
        evaluation_settings["num_workers"],
        device,
    )
    return predict_loader(model, loader, device)[2]


def export_test_submission(
    model: torch.nn.Module,
    image_root: Path,
    frames: pl.DataFrame,
    output: Path,
    evaluation_settings: dict,
    device: torch.device,
) -> dict:
    """Predict every test frame and write a submission file without submitting it.

    The official sample file is not part of the data, so the rows are validated against the
    membership of test_images.csv only and the file name says so.

    Args:
        model: Trained detector.
        image_root: Directory that holds the test images, local or staged.
        frames: Prepared frame metadata including the test rows.
        output: Experiment directory that receives the CSV and its metadata.
        evaluation_settings: The evaluation table, with batch_size and num_workers.
        device: Device that runs the model.

    Returns:
        Validated image, detection and empty-prediction counts plus the inference timing.
    """
    test_frames = frames.filter(pl.col("split") == "test")
    no_objects = pl.DataFrame(schema=EMPTY_OBJECT_SCHEMA)
    loader = make_loader(
        IgnoreAwareDetectionDataset(
            image_root, test_frames, no_objects, no_objects.drop("category_id")
        ),
        evaluation_settings["batch_size"],
        evaluation_settings["num_workers"],
        device,
    )
    predictions, image_ids, inference = predict_loader(model, loader, device)
    if image_ids != list(range(test_frames.height)):
        raise ValueError("Prediction order differs from the test frame table.")
    sample = test_frames.select(
        pl.col("submission_id").alias("image_id"), pl.lit("").alias("PredictionString")
    )
    path = output / "submission_unverified.csv"
    build_submission(predictions, test_frames, sample).write_csv(path)
    validation = validate_submission(
        pl.read_csv(path, schema_overrides={"image_id": pl.String, "PredictionString": pl.String}),
        sample,
    )
    record = {
        **validation,
        "official_sample_verified": False,
        "file": path.name,
        "inference": inference,
    }
    write_json(output / "submission_metadata.json", record)
    return record


def find_previous_output(input_root: Path, batch_id: str, experiment_name: str) -> Path | None:
    """Find the experiment directory of a previous notebook version attached as input.

    Args:
        input_root: The read-only input directory of the hosted notebook.
        batch_id: Batch identifier of the run.
        experiment_name: Descriptive experiment name.

    Returns:
        The attached directory that holds latest.pt, or None when nothing is attached.
    """
    for child in sorted(input_root.iterdir()):
        candidate = child / "runs/faster_rcnn" / batch_id / experiment_name
        if (candidate / "latest.pt").is_file():
            return candidate
    return None


def restore_previous_output(previous: Path, destination: Path) -> bool:
    """Copy a previous version's records into the writable output so that a run can resume.

    Args:
        previous: Experiment directory from find_previous_output.
        destination: Writable experiment directory of the current run.

    Returns:
        Whether files were copied; an existing checkpoint in the destination wins.
    """
    if (destination / "latest.pt").is_file():
        return False
    shutil.copytree(previous, destination, dirs_exist_ok=True)
    return True


def _flatten_epoch(row: dict) -> dict:
    canonical, dense = row["metrics"]["max_dets_100"], row["metrics"]["max_dets_500"]
    return {
        "epoch": row["epoch"],
        "map_100": canonical["map"],
        "ap50_100": canonical["ap50"],
        "ap_small_100": canonical["ap_small"],
        "map_500": dense["map"],
        "ap50_500": dense["ap50"],
        "ap_small_500": dense["ap_small"],
        "validation_ms_per_frame": row["inference"]["milliseconds_per_frame"],
        "evaluation_seconds": row["evaluation_seconds"],
        "global_step": row["global_step"],
        "wall_train_seconds": row["wall_train_seconds"],
    }


def summarize_experiment(directory: Path) -> dict:
    """Summarize one experiment directory from its JSON records.

    Args:
        directory: Directory written by run_experiment.

    Returns:
        The status (Not run, Partial or Completed), per-epoch metrics, the final per-class
        AP at both detection caps, and the speed record when it exists.
    """
    status_path = directory / "status.json"
    if not status_path.is_file():
        return {"experiment_id": directory.name, "status": "Not run", "epochs": []}
    status = json.loads(status_path.read_text())
    config = json.loads((directory / "config.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    speed_path = directory / "speed.json"
    return {
        "experiment_id": directory.name,
        "status": "Completed" if status["is_completed"] else "Partial",
        "epochs_total": config["epochs"],
        "epochs_completed": status["epochs_completed"],
        "global_step": status["global_step"],
        "train_frames": status["train_frames"],
        "image_exposures": status["image_exposures"],
        "wall_train_seconds": status["wall_train_seconds"],
        "epochs": [_flatten_epoch(row) for row in metrics],
        "per_class_ap_100": (
            metrics[-1]["metrics"]["max_dets_100"]["per_class_ap"] if metrics else None
        ),
        "per_class_ap_500": (
            metrics[-1]["metrics"]["max_dets_500"]["per_class_ap"] if metrics else None
        ),
        "speed": json.loads(speed_path.read_text()) if speed_path.is_file() else None,
    }


def load_batch_records(batch_directory: Path) -> dict:
    """Summarize every experiment of a batch and read its timing probe.

    Args:
        batch_directory: runs/faster_rcnn/<batch_id> directory.

    Returns:
        The experiment summaries and the probe record, or None when no probe exists.
    """
    probe_path = batch_directory / "_timing_probe/probe.json"
    experiments = [
        summarize_experiment(path)
        for path in sorted(batch_directory.iterdir())
        if path.is_dir() and not path.name.startswith("_")
    ]
    return {
        "experiments": experiments,
        "probe": json.loads(probe_path.read_text()) if probe_path.is_file() else None,
    }


def aggregate_loss_curve(history: list[dict], window_size: int = 100) -> pl.DataFrame:
    """Average the four Faster R-CNN losses and their sum in windows inside each epoch.

    Args:
        history: Step records from history.json, with epoch, attempt, step and the losses.
        window_size: Maximum consecutive batch attempts represented by one curve point.

    Returns:
        One row per window with the last update number and the mean of every loss.
    """
    return (
        pl.from_dicts(history, infer_schema_length=None)
        .with_columns(((pl.col("attempt") - 1) // window_size).alias("window"))
        .group_by("epoch", "window", maintain_order=True)
        .agg(
            pl.col("step").last(),
            pl.len().alias("batches"),
            *[pl.col(name).mean() for name in LOSS_COLUMNS],
        )
        .drop("window")
    )
