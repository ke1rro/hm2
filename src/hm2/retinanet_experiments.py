import json
import logging
import math
import pickle
import platform
import time
from collections.abc import Callable
from pathlib import Path

import polars as pl
import torch
import torchvision

from hm2.detection_data import DetectionDataset, make_loader, select_frames
from hm2.evaluation import evaluate_coco, predictions_to_coco
from hm2.experiment_checkpoint import load_checkpoint, publish_snapshot, save_checkpoint, write_json
from hm2.experiment_names import descriptive_name, experiment_label
from hm2.experiment_setup import dataset_identity
from hm2.retinanet import build_retinanet, make_optimizer, predict_loader, train_steps


def epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    """Return a reproducible frame permutation independent of loader worker state.

    Args:
        count: Number of selected training frames.
        seed: Shared experiment seed.
        epoch: Zero-based training epoch.

    Returns:
        Selected frame indices in epoch order.
    """
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def verify_resume(root: Path, metadata: Path, output: Path, config: dict) -> dict:
    """Restore a saved RetinaNet training state and infer one validation image on CUDA.

    Args:
        root: Original train/validation dataset.
        metadata: Prepared frame and object tables.
        output: Experiment directory with latest.pt and its configuration sidecar.
        config: Requested experiment recipe, already checked against its batch configuration.

    Returns:
        Restored optimizer step, detection count and actual device without training.
    """
    frames = pl.read_parquet(metadata / "frames.parquet")
    objects = pl.read_parquet(metadata / "objects.parquet")
    device = torch.device("cuda:0")
    model = build_retinanet(
        config["variant"],
        allow_download=True,
        min_size=config["min_size"],
        max_size=config["max_size"],
        seed=config["seed"],
    ).to(device)
    batches = math.ceil(
        select_frames(frames, "train", config["frame_stride"]).height / config["batch_size"]
    )
    optimizer, scheduler = make_optimizer(
        model,
        config["learning_rate"],
        batches * config["epochs"],
        round(batches * config["warmup_epochs"]),
    )
    scaler = torch.amp.GradScaler("cuda")
    path = output / "latest.pt"
    saved_config = json.loads(path.with_suffix(".json").read_text())
    state = load_checkpoint(path, model, optimizer, scheduler, scaler, saved_config)
    model.eval()
    image, _ = DetectionDataset(root, select_frames(frames, "val").head(1), objects)[0]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        predictions = model([image.to(device)])
    result = {
        "step": state["step"],
        "detections": len(predictions[0]["boxes"]),
        "device": str(device),
    }
    del model, optimizer, scheduler, scaler, predictions
    torch.cuda.empty_cache()
    return result


def _checkpoint_signature(progress: dict) -> tuple:
    return (
        progress["epochs_completed"],
        progress["batch_cursor"],
        progress["global_step"],
        progress["image_exposures"],
        progress["wall_train_seconds"],
        len(progress["history"]),
        len(progress["metrics"]),
        progress["pending_evaluation"],
        progress["best_map"],
        progress["rollback"],
    )


def run_chunk(
    root: Path,
    metadata: Path,
    output: Path,
    config: dict,
    resume: bool = True,
    seconds: int = 600,
    on_checkpoint: Callable[[Path], None] | None = None,
) -> dict:
    """Train a bounded epoch segment and publish resumable snapshots for concurrent local transfer.

    Complete validation after every epoch, including pending validation on resume.
    Full validation can exceed the chunk time budget. The budget bounds training
    between batches; an in-flight batch and atomic checkpoint must finish first.

    Args:
        root: Staged original dataset with validation COCO annotations.
        metadata: Directory containing frames, objects, and ignored_regions parquet tables.
        output: Runtime experiment directory periodically downloaded by the host.
        config: Flat shared recipe and experiment identity.
        resume: Whether to restore compatible existing state.
        seconds: Training wall time budget for this call.
        on_checkpoint: Optional synchronous persistence hook after a local snapshot is complete.

    Returns:
        Published status with completion, progress, and rollback information.
    """
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{__name__}.{descriptive_name(config['experiment_id'])}")
    log_path = (output / "execution.log").resolve()
    if not any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path)
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    frames = pl.read_parquet(metadata / "frames.parquet")
    objects = pl.read_parquet(metadata / "objects.parquet")
    selected = select_frames(frames, "train", config["frame_stride"])
    validation = select_frames(frames, "val")
    selected_keys = selected["image_key"].to_list()
    identity = dataset_identity(root, metadata, selected_keys)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_retinanet(
        config["variant"],
        allow_download=True,
        min_size=config["min_size"],
        max_size=config["max_size"],
        seed=config["seed"],
    ).to(device)
    batches = math.ceil(selected.height / config["batch_size"])
    optimizer, scheduler = make_optimizer(
        model,
        config["learning_rate"],
        batches * config["epochs"],
        round(batches * config["warmup_epochs"]),
    )
    effective = {
        **config,
        "dataset_selection_sha256": identity,
        "selected_frame_count": selected.height,
        "validation_frame_count": validation.height,
        "anchor_sizes": [list(level) for level in model.anchor_generator.sizes],
        "anchor_aspect_ratios": [list(level) for level in model.anchor_generator.aspect_ratios],
        "optimizer": {"backend": "SGD", "momentum": 0.9, "weight_decay": 1e-4},
        "loss": {"classification": "focal", "regression": "giou"},
    }
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    progress: dict = {
        "epochs_completed": 0,
        "batch_cursor": 0,
        "global_step": 0,
        "image_exposures": 0,
        "history": [],
        "metrics": [],
        "wall_train_seconds": 0.0,
        "pending_evaluation": False,
        "best_map": None,
        "rollback": None,
    }
    latest = output / "latest.pt"
    saved_signature = None
    if not resume and latest.exists():
        raise FileExistsError("Existing experiment checkpoint requires resume=True.")
    if resume:
        checkpoint = latest if latest.exists() else output / "best.pt"
        if checkpoint.exists():
            try:
                restored = load_checkpoint(
                    checkpoint, model, optimizer, scheduler, scaler, effective
                )
                if checkpoint == latest:
                    saved_signature = _checkpoint_signature(restored["extra_state"])
            except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as error:
                fallback = output / "best.pt"
                if checkpoint == fallback or not fallback.exists():
                    raise
                restored = load_checkpoint(fallback, model, optimizer, scheduler, scaler, effective)
                restored["extra_state"]["rollback"] = str(error)
                logger.warning("Restored compatible best checkpoint after latest failed: %s", error)
            progress = restored["extra_state"]
            if checkpoint != latest:
                progress["rollback"] = "latest.pt missing; restored best.pt"
    write_json(output / "config.json", effective)
    write_json(
        output / "selected_frames.json",
        {
            "train": selected_keys,
            "val": validation["image_key"].to_list(),
        },
    )
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "reproducibility": "Fixed epoch order and frame flips; CUDA may be nondeterministic.",
            "validation_cost": "Full official validation every epoch may exceed chunk time budget.",
            "wall_train_seconds_scope": (
                "Sum of end-to-end training batch latency; excludes model and metadata setup, "
                "iterator construction, checkpoints, and validation."
            ),
            "snapshot_retention": (
                "Retain at least three generations and all generations younger than 30 minutes. "
                "Host preserves its last verified snapshot when a download is interrupted."
            ),
            "training_ignore_policy": "Stock RetinaNet does not suppress ignored-region negatives.",
        },
    )
    started = time.monotonic()
    last_save = started

    def status_record() -> dict:
        status = {
            key: value for key, value in progress.items() if key not in {"history", "metrics"}
        }
        status.update(
            {
                "experiment_id": config["experiment_id"],
                "train_frames": selected.height,
                "completed": progress["epochs_completed"] == config["epochs"]
                and not progress["pending_evaluation"],
                "stopped": (output / "STOP").exists(),
                "chunk_seconds": time.monotonic() - started,
            }
        )
        return status

    def persist() -> None:
        nonlocal last_save, saved_signature
        signature = _checkpoint_signature(progress)
        if signature == saved_signature:
            return
        save_checkpoint(
            latest,
            model,
            optimizer,
            scheduler,
            progress["global_step"],
            effective,
            scaler,
            progress,
        )
        write_json(output / "history.json", progress["history"])
        write_json(output / "metrics.json", progress["metrics"])
        write_json(output / "status.json", status_record())
        logger.info(
            "Checkpoint epoch=%s cursor=%s updates=%s exposures=%s",
            progress["epochs_completed"],
            progress["batch_cursor"],
            progress["global_step"],
            progress["image_exposures"],
        )
        for handler in logger.handlers:
            handler.flush()
        publish_snapshot(output)
        if on_checkpoint is not None:
            on_checkpoint(output)
        saved_signature = signature
        last_save = time.monotonic()

    if saved_signature is None:
        persist()
    else:
        ready = output / "sync/ready.json"
        manifest = json.loads(ready.read_text()) if ready.is_file() else {}
        snapshot = output / "sync" / manifest.get("generation", "") / "latest.pt"
        if not snapshot.is_file() or not snapshot.samefile(latest):
            write_json(output / "history.json", progress["history"])
            write_json(output / "metrics.json", progress["metrics"])
            write_json(output / "status.json", status_record())
            publish_snapshot(output)
        if on_checkpoint is not None:
            on_checkpoint(output)
    while progress["epochs_completed"] < config["epochs"] or progress["pending_evaluation"]:
        if progress["pending_evaluation"]:
            logger.info(
                "%s: full validation epoch %s",
                experiment_label(config["experiment_id"]),
                progress["epochs_completed"],
            )
            evaluation_started = time.monotonic()
            val_loader = make_loader(
                DetectionDataset(root, validation, objects),
                config["batch_size"],
                config["num_workers"],
                device,
            )
            predictions, image_ids, inference = predict_loader(model, val_loader, device)
            records = predictions_to_coco(predictions, image_ids)
            metrics = evaluate_coco(root / "val/annotations_coco.json", records, image_ids)
            is_best = metrics["map"] is not None and (
                progress["best_map"] is None or metrics["map"] > progress["best_map"]
            )
            row = {
                "epoch": progress["epochs_completed"],
                "metrics": metrics,
                "inference": inference,
                "evaluation_seconds": time.monotonic() - evaluation_started,
                "global_step": progress["global_step"],
                "is_best": is_best,
                "image_exposures": progress["image_exposures"],
                "wall_train_seconds": progress["wall_train_seconds"],
            }
            progress["metrics"].append(row)
            progress["pending_evaluation"] = False
            write_json(output / "predictions.json", records)
            write_json(output / f"epoch_{progress['epochs_completed']}_metrics.json", row)
            if is_best:
                progress["best_map"] = metrics["map"]
                save_checkpoint(
                    output / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    progress["global_step"],
                    effective,
                    scaler,
                    progress,
                )
            persist()
            if progress["epochs_completed"] == config["epochs"]:
                break
        if (output / "STOP").exists() or time.monotonic() - started >= seconds:
            break
        epoch = progress["epochs_completed"]
        ordered = selected[epoch_order(selected.height, config["seed"], epoch)]
        remaining = ordered.slice(progress["batch_cursor"] * config["batch_size"])
        loader = make_loader(
            DetectionDataset(
                root,
                remaining,
                objects,
                config["flip_probability"],
                augmentation_seed=config["seed"] + epoch,
            ),
            config["batch_size"],
            config["num_workers"],
            device,
            seed=config["seed"] + epoch,
        )

        def on_step(row: dict) -> bool:
            progress["history"].append(
                {**row, "epoch": epoch + 1, "attempt": len(progress["history"]) + 1}
            )
            progress["batch_cursor"] += 1
            progress["global_step"] = row["step"]
            progress["image_exposures"] += row["image_count"]
            progress["wall_train_seconds"] += row["seconds"]
            should_stop = (output / "STOP").exists() or time.monotonic() - started >= seconds
            if should_stop or time.monotonic() - last_save >= config["checkpoint_seconds"]:
                persist()
            return should_stop

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
        if progress["batch_cursor"] == batches:
            progress["epochs_completed"] += 1
            progress["batch_cursor"] = 0
            progress["pending_evaluation"] = True
            persist()
        else:
            break
    persist()
    return status_record()
