import importlib
import logging
import math
import statistics
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from hm2.dense_evaluation import evaluate_detections
from hm2.detection_data import make_loader
from hm2.drive_api import DriveApi
from hm2.evaluation import predictions_to_coco
from hm2.experiment_checkpoint import write_json
from hm2.faster_rcnn import IgnoreAwareDetectionDataset
from hm2.faster_rcnn_experiments import (
    RECORDED_MODULES,
    compute_source_identity,
    run_experiment,
    select_experiment_frames,
    select_probe_frames,
)
from hm2.retinanet import predict_loader, train_steps
from hm2.rt_detr import (
    GradientClippedAdamW,
    QueryDetector,
    build_rt_detr,
    compare_with_reference_loss,
    make_rt_detr_optimizer,
)

LOGGER = logging.getLogger(__name__)

RUN_ORDER = ("baseline_q300", "nwd_loss", "queries_500", "denoising_300")
OPTIONAL_EXPERIMENTS = ("shared_lr",)
EXPERIMENT_LABELS = {
    "baseline_q300": "RT-DETRv2 baseline (300 queries)",
    "nwd_loss": "RT-DETRv2 with NWD box loss",
    "queries_500": "RT-DETRv2 with 500 queries",
    "denoising_300": "RT-DETRv2 with 300 denoising queries",
    "shared_lr": "RT-DETRv2 with one learning rate",
}
RT_DETR_MODULES = (*RECORDED_MODULES, "hm2.rt_detr", "hm2.rt_detr_experiments")
RECORD_FILES = (
    "status.json",
    "config.json",
    "metrics.json",
    "history.json",
    "model.json",
    "environment.json",
    "speed.json",
    "submission_metadata.json",
    "val_predictions.json",
)


def experiment_label(name: str) -> str:
    """Return the readable name used in notebook tables and captions.

    Args:
        name: Descriptive experiment name, such as baseline_q300.

    Returns:
        The readable label, or the name itself when it has no label.
    """
    return EXPERIMENT_LABELS.get(name, name)


def load_rt_detr_settings(path: Path, experiment_name: str) -> dict:
    """Read the shared RT-DETR recipe and apply one experiment's overrides.

    Each experiment table may only replace keys that already exist in the shared tables,
    so a misspelled override fails instead of being ignored.

    Args:
        path: TOML file with dataset, model, loss, matcher, training, evaluation, probe,
            export and experiments tables.
        experiment_name: Name of a table under experiments.

    Returns:
        The configuration tables of that experiment, without the experiments table.
    """
    document = tomllib.loads(path.read_text())
    experiments = document.pop("experiments")
    if experiment_name not in experiments:
        raise ValueError(f"Unknown experiment {experiment_name!r}: {sorted(experiments)}")
    for table, values in experiments[experiment_name].items():
        unknown = set(values) - set(document.get(table, {}))
        if unknown:
            raise ValueError(f"Unknown {table} settings in {experiment_name}: {sorted(unknown)}")
        document[table] = {**document[table], **values}
    return document


def list_experiments(path: Path) -> list[str]:
    """Return the experiment names in the order they appear in the configuration.

    Args:
        path: TOML file with an experiments table.

    Returns:
        Experiment names.
    """
    return list(tomllib.loads(path.read_text())["experiments"])


def build_rt_detr_experiment_config(
    settings: dict, experiment_name: str, selected_keys: list[str], source_identity: str
) -> dict:
    """Describe the recipe that a checkpoint must match before a run may resume.

    Args:
        settings: The configuration of one experiment from load_rt_detr_settings.
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
        "backbone_learning_rate": training["backbone_learning_rate"],
        "weight_decay": training["weight_decay"],
        "gradient_clip_norm": training["gradient_clip_norm"],
        "warmup_updates": training["warmup_updates"],
        "flip_probability": training["flip_probability"],
        "frame_stride": settings["dataset"]["frame_stride"],
        "selected_frame_count": len(selected_keys),
        "dataset_selection_sha256": source_identity,
        "model": settings["model"],
        "loss": settings["loss"],
        "matcher": settings["matcher"],
        "optimizer": {
            "backend": "AdamW",
            "groups": "backbone at backbone_learning_rate; no weight decay on other 1D params",
            "gradient_clipping": "global L2 norm before every update",
        },
        "schedule": "linear warmup from 1%, then x0.1 at 2/3 and at 90% of the updates",
        "postprocess": "top-k (query, class) pairs by sigmoid score, no NMS",
        "checkpoint_selection": "final epoch, never chosen on validation",
    }


def build_rt_detr_training_state(
    settings: dict, batches: int, device: torch.device
) -> tuple[QueryDetector, Optimizer, LRScheduler, torch.amp.GradScaler]:
    """Build the pretrained detector, the clipped AdamW optimizer, the schedule and the scaler.

    Args:
        settings: The configuration of one experiment.
        batches: Optimizer updates per epoch.
        device: Device that trains the model.

    Returns:
        The model on the device, the optimizer, the schedule and the AMP gradient scaler.
    """
    training = settings["training"]
    model = build_rt_detr(
        settings["model"], settings["loss"], settings["matcher"], training["seed"]
    ).to(device)
    optimizer, scheduler = make_rt_detr_optimizer(model, training, batches * training["epochs"])
    return model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def describe_query_step(model: torch.nn.Module, optimizer: Optimizer) -> dict:
    """Return the per-step diagnostics added to every history row.

    Args:
        model: The wrapped detector after its training call.
        optimizer: The clipped AdamW optimizer after its update.

    Returns:
        Matched and excluded query counts, denoising groups and the gradient norm.
    """
    return {
        **getattr(model, "last_diagnostics", {}),
        "gradient_norm": (
            optimizer.last_gradient_norm if isinstance(optimizer, GradientClippedAdamW) else None
        ),
    }


def run_rt_detr_experiment(
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
    """Train and validate one RT-DETR experiment with the shared Faster R-CNN run loop.

    The loop, the epoch order, the flips, the per-epoch validation at maxDets 100 and 500,
    the checkpoints and the final-epoch rule are those of the B1 run; only the recipe and
    the training-state builder differ.

    Args:
        image_root: Directory that holds the images, local or staged.
        source_root: Directory that holds the COCO annotation files.
        frames: Prepared frame metadata.
        objects: Prepared target objects.
        ignored: Prepared ignored regions.
        output: Experiment directory for checkpoints and JSON records.
        settings: The configuration of one experiment.
        experiment_name: Descriptive experiment name.
        resume: Whether an existing checkpoint may be restored.
        seconds: Optional training time budget for this call.
        on_checkpoint: Optional hook called with the output directory after each checkpoint.

    Returns:
        The run status, including whether every epoch and its validation are complete.
    """
    selected, _ = select_experiment_frames(frames, settings["dataset"])
    selected_keys = selected["image_key"].to_list()
    config = build_rt_detr_experiment_config(
        settings,
        experiment_name,
        selected_keys,
        compute_source_identity(source_root, selected_keys),
    )
    return run_experiment(
        image_root,
        source_root,
        frames,
        objects,
        ignored,
        output,
        settings,
        experiment_name,
        resume=resume,
        seconds=seconds,
        on_checkpoint=on_checkpoint,
        experiment_config=config,
        build_training_state=build_rt_detr_training_state,
        describe_step=describe_query_step,
        recorded_modules=RT_DETR_MODULES,
    )


def summarize_object_sizes(
    selected: pl.DataFrame, objects: pl.DataFrame, model_settings: dict
) -> dict:
    """Measure the object size in model pixels, which the NWD constant should resemble.

    Args:
        selected: Selected training frames with image_width and image_height.
        objects: Prepared target objects with width and height in original pixels.
        model_settings: The model table, with min_size and max_size.

    Returns:
        Mean, median and 10th and 90th percentiles of sqrt(width * height) in model pixels.
    """
    width, height = pl.col("image_width"), pl.col("image_height")
    scale = pl.min_horizontal(
        model_settings["min_size"] / pl.min_horizontal(width, height),
        model_settings["max_size"] / pl.max_horizontal(width, height),
    )
    sizes = (
        objects.join(
            selected.select("image_key", scale.alias("scale")), on="image_key", how="inner"
        )
        .select(((pl.col("width") * pl.col("height")).sqrt() * pl.col("scale")).alias("size"))[
            "size"
        ]
        .to_numpy()
    )
    if sizes.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None, "objects": 0}
    return {
        "mean": float(np.mean(sizes)),
        "median": float(np.median(sizes)),
        "p10": float(np.quantile(sizes, 0.1)),
        "p90": float(np.quantile(sizes, 0.9)),
        "objects": int(sizes.size),
    }


def run_rt_detr_timing_probe(
    image_root: Path,
    source_root: Path,
    frames: pl.DataFrame,
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
    output: Path,
    settings: dict,
) -> dict:
    """Measure the real cost of one RT-DETR experiment and check its conventions on the GPU.

    Train a handful of updates with the real recipe, time inference at batch size 1 and at
    the validation batch size and evaluate once. Check that the checkpoint configuration
    follows each input size, that our loss equals the Hugging Face loss on a shared forward
    pass, and that matching keeps every object while the ignore rule drops queries on the
    frame with the most ignored regions. Nothing is checkpointed; the projections are
    linear extrapolations from a few updates, not results.

    Args:
        image_root: Directory that holds the images, local or staged.
        source_root: Directory that holds the COCO annotation files.
        frames: Prepared frame metadata.
        objects: Prepared target objects.
        ignored: Prepared ignored regions.
        output: Directory that receives probe.json.
        settings: The configuration of one experiment.

    Returns:
        The measurements, checks and projections that were also written to probe.json.
    """
    transformers = importlib.import_module("transformers")
    training, evaluation, probe = settings["training"], settings["evaluation"], settings["probe"]
    selected, validation = select_experiment_frames(frames, settings["dataset"])
    probe_frames = select_probe_frames(selected, validation, ignored, settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = math.ceil(selected.height / training["batch_size"])
    model, optimizer, scheduler, scaler = build_rt_detr_training_state(settings, batches, device)
    detector_config = getattr(model.detector, "config")
    if detector_config.anchor_image_size is not None or detector_config.eval_size is not None:
        raise RuntimeError("The detector still uses fixed 640 by 640 anchors or embeddings.")
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
    first_images, first_targets = next(iter(loader))
    reference = compare_with_reference_loss(
        model,
        [image.to(device) for image in first_images],
        [{key: value.to(device) for key, value in target.items()} for target in first_targets],
    )
    if reference["relative_difference"] > 1e-3:
        raise RuntimeError(f"Our loss differs from the Hugging Face loss: {reference}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = train_steps(
        model, loader, optimizer, scheduler, device, probe["updates"], scaler=scaler
    )
    steady = history[min(3, len(history) - 1) :]
    images_per_second = statistics.median(row["images_per_second"] for row in steady)
    image, target = IgnoreAwareDetectionDataset(
        image_root, probe_frames["ignore"], objects, ignored
    )[0]
    model.train()
    with (
        torch.no_grad(),
        torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"),
    ):
        model([image.to(device)], [{key: value.to(device) for key, value in target.items()}])
    ignore_check = {
        **model.last_diagnostics,
        "objects": int(target["labels"].numel()),
        "ignored_regions": int(target["ignore_boxes"].shape[0]),
        "expected_matched_queries": min(
            int(target["labels"].numel()), settings["model"]["num_queries"]
        ),
    }
    if ignore_check["matched_queries"] != ignore_check["expected_matched_queries"]:
        raise RuntimeError(f"Matching lost objects on the ignore probe frame: {ignore_check}")
    validation_dataset = IgnoreAwareDetectionDataset(
        image_root, probe_frames["validation"], objects, ignored
    )
    inference = {}
    predictions, image_ids = [], []
    for batch_size in sorted({1, evaluation["batch_size"]}):
        predictions, image_ids, inference[f"batch_{batch_size}"] = predict_loader(
            model,
            make_loader(validation_dataset, batch_size, evaluation["num_workers"], device),
            device,
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
        "transformers": str(transformers.__version__),
        "model_id": settings["model"]["model_id"],
        "model_revision": settings["model"]["model_revision"],
        "model_commit": getattr(detector_config, "_commit_hash", None),
        "anchor_image_size": detector_config.anchor_image_size,
        "eval_size": detector_config.eval_size,
        "distinct_decoder_class_heads": len(
            {
                layer.weight.data_ptr()
                for layer in getattr(model.detector, "model").decoder.class_embed
            }
        ),
        "image_mean": model.image_mean,
        "image_std": model.image_std,
        "updates": probe["updates"],
        "batch_size": training["batch_size"],
        "losses_are_finite": True,
        "reference_loss_check": reference,
        "train_images_per_second": images_per_second,
        "train_seconds_per_update": training["batch_size"] / images_per_second,
        "peak_gpu_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "ignore_check": ignore_check,
        "object_size_model_pixels": summarize_object_sizes(selected, objects, settings["model"]),
        "nwd_constant": settings["loss"]["nwd_constant"],
        "inference": inference,
        "evaluation_seconds_per_frame": evaluation_seconds_per_frame,
        "detections_per_frame_mean": len(records) / len(image_ids),
        "projection": {
            "train_minutes": train_minutes,
            "validation_minutes": validation_minutes,
            "total_minutes": train_minutes + validation_minutes,
            "note": "Linear extrapolation from the probe; not a result.",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "probe.json", result)
    return result


def load_final_rt_detr_model(output: Path, device: torch.device) -> tuple[QueryDetector, dict]:
    """Restore the model saved in latest.pt; only the configuration files are fetched.

    Args:
        output: Experiment directory with latest.pt.
        device: Device that receives the model.

    Returns:
        The model in evaluation mode and the configuration saved with the checkpoint.
    """
    state = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
    config = state["config"]
    model = build_rt_detr(
        config["model"], config["loss"], config["matcher"], config["seed"], pretrained=False
    )
    model.load_state_dict(state["model"])
    return model.to(device).eval(), config


def find_drive_folder(api: DriveApi, parent: str, names: list[str]) -> str | None:
    """Follow existing Drive folder names without creating any folder.

    Args:
        api: Authenticated Drive file transport.
        parent: Starting Drive folder identifier.
        names: Folder names from the parent downwards.

    Returns:
        The identifier of the last folder, or None when any folder is missing.
    """
    current = parent
    for name in names:
        child = api.children(current).get(name)
        if child is None:
            return None
        current = child["id"]
    return current


def download_committed_files(
    api: DriveApi, folder_id: str, names: tuple[str, ...], destination: Path
) -> list[str]:
    """Download selected files of the committed snapshot of one experiment folder.

    Args:
        api: Authenticated Drive file transport.
        folder_id: Experiment folder with CURRENT.json and generations.
        names: File names to fetch when the snapshot contains them.
        destination: Local directory that receives the files.

    Returns:
        The names that were downloaded and verified against the manifest.
    """
    children = api.children(folder_id)
    if "CURRENT.json" not in children:
        return []
    manifest = api.read_json(children["CURRENT.json"]["id"])
    generations = api.children(children["generations"]["id"])
    files = api.children(generations[manifest["generation"]]["id"])
    downloaded = []
    for name in names:
        if name in manifest["files"] and name in files:
            api.download(files[name]["id"], destination / name, manifest["files"][name])
            downloaded.append(name)
    return downloaded
