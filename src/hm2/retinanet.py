import inspect
import json
import logging
import random
import time
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torchvision
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler, MultiStepLR, SequentialLR
from torchvision.models.detection import (
    RetinaNet,
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection._utils import _box_loss
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from tqdm.notebook import tqdm

from hm2.detection_data import DetectionDataset, make_loader
from hm2.evaluation import evaluate_coco, predictions_to_coco
from hm2.experiment_checkpoint import load_checkpoint, save_checkpoint
from hm2.experiment_names import VARIANTS, descriptive_name

__all__ = [
    "build_retinanet",
    "load_checkpoint",
    "make_optimizer",
    "predict_loader",
    "run_smoke_test",
    "save_checkpoint",
    "train_steps",
]

LOGGER = logging.getLogger(__name__)


def build_retinanet(
    variant: str,
    pretrained: bool = True,
    allow_download: bool = False,
    min_size: int = 640,
    max_size: int = 1024,
    seed: int = 2026,
) -> RetinaNet:
    """Build one controlled RetinaNet variant with the same seeded nine-class initialization.

    Load COCO detector weights before replacing only the class head. Retain the
    pretrained class tower and regression head. Random initialization is available
    explicitly for architecture unit tests, never as a pretrained experiment substitute.

    Args:
        variant: standard_anchors, smaller_anchors, or diou_regression.
        pretrained: Whether to load COCO detector weights.
        allow_download: Whether an explicitly authorized missing-weight download is allowed.
        min_size: Shared shorter-side resize in model pixels.
        max_size: Shared maximum longer-side resize in model pixels.
        seed: Shared model initialization seed.

    Returns:
        A nine-class TorchVision RetinaNet with the requested isolated modification.
    """
    variant = descriptive_name(variant)
    if variant not in VARIANTS:
        raise ValueError(f"Expected a RetinaNet variant from {VARIANTS}.")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    if weights is not None:
        cached = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
        if not cached.is_file() and not allow_download:
            raise FileNotFoundError(
                "COCO RetinaNet V2 weights are not cached. Authorize the official weight "
                "download before setting allow_download=True."
            )
    model = retinanet_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        topk_candidates=1000,
    )
    previous_head = model.head.classification_head
    anchors_per_location = model.anchor_generator.num_anchors_per_location()[0]
    head = RetinaNetClassificationHead(
        model.backbone.out_channels,
        anchors_per_location,
        9,
        norm_layer=partial(torch.nn.GroupNorm, 32),
    )
    head.conv.load_state_dict(previous_head.conv.state_dict())
    model.head.classification_head = head
    if variant == "smaller_anchors":
        smaller_sizes = tuple(
            tuple(size // 2 for size in level) for level in model.anchor_generator.sizes
        )
        model.anchor_generator = AnchorGenerator(
            smaller_sizes, model.anchor_generator.aspect_ratios
        )
    if variant == "diou_regression":
        version = tuple(int(part) for part in torchvision.__version__.split("+")[0].split(".")[:2])
        if version < (0, 16) or '"diou"' not in inspect.getsource(_box_loss):
            raise RuntimeError("This TorchVision box-loss implementation has not exposed DIoU.")
        model.head.regression_head._loss_type = "diou"
    return model


def make_optimizer(
    model: torch.nn.Module, lr: float, total_steps: int, warmup_steps: int = 100
) -> tuple[Optimizer, LRScheduler]:
    """Use one SGD momentum recipe with linear warmup and two shared LR reductions.

    Args:
        model: Detector whose trainable parameters are optimized.
        lr: Base learning rate shared across runs.
        total_steps: Shared optimization budget.
        warmup_steps: Initial linear warmup steps.

    Returns:
        SGD optimizer and resumable step-based schedule.
    """
    optimizer = SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    warmup = min(warmup_steps, total_steps - 1)
    decay = MultiStepLR(
        optimizer,
        milestones=[
            max(1, int(total_steps * 2 / 3)) - warmup,
            max(2, int(total_steps * 0.9)) - warmup,
        ],
        gamma=0.1,
    )
    scheduler = (
        SequentialLR(
            optimizer,
            [LinearLR(optimizer, start_factor=0.01, total_iters=warmup), decay],
            milestones=[warmup],
        )
        if warmup
        else decay
    )
    return optimizer, scheduler


def train_steps(
    model: torch.nn.Module,
    loader: Iterable[tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    max_steps: int,
    start_step: int = 0,
    scaler: torch.amp.GradScaler | None = None,
    should_show_progress: bool = False,
    checkpoint_path: Path | None = None,
    config: dict | None = None,
    checkpoint_interval: int = 250,
    max_attempts: int | None = None,
    max_batches: int | None = None,
    on_step: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Run a bounded optimization loop and record end-to-end step latency and losses.

    Reuse the loader across epochs, transfer pinned batches asynchronously on CUDA,
    and use fp16 autocast with gradient scaling there. Synchronize CUDA timing and
    advance the schedule only when the scaler actually performs an optimizer update.

    Args:
        model: Detector to train.
        loader: Reiterable batches of variable-size images and targets.
        optimizer: Shared optimizer.
        scheduler: Shared step-based learning-rate schedule.
        device: Explicit CPU or CUDA device.
        max_steps: Number of successful diagnostic or authorized graded optimizer updates.
        start_step: Successful optimizer updates restored from a checkpoint.
        scaler: Persistent gradient scaler, disabled for CPU.
        should_show_progress: Whether to show a notebook progress bar.
        checkpoint_path: Optional resumable checkpoint destination.
        config: Exact experiment configuration saved with checkpoints.
        checkpoint_interval: Number of loop steps between checkpoint saves.
        max_attempts: Optional hard attempt limit for a bounded smoke diagnostic.
        max_batches: Stop after this many consumed batches without repeating the loader.
        on_step: Callback after each consumed batch; return True to stop.

    Returns:
        Per-step loss, latency, throughput, LR, and optimizer-update records.
    """
    model.train()
    is_cuda = device.type == "cuda"
    scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=is_cuda)
    iterator = iter(loader)
    history = []
    completed = start_step
    attempts = 0
    progress = tqdm(total=max_steps, disable=not should_show_progress)
    while completed < start_step + max_steps:
        if max_batches is not None and attempts >= max_batches:
            break
        if max_attempts is not None and attempts >= max_attempts:
            raise RuntimeError(
                "Smoke attempt limit reached before the requested optimizer updates."
            )
        attempts += 1
        if is_cuda:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            images, targets = next(iterator)
        except StopIteration:
            if max_batches is not None:
                break
            iterator = iter(loader)
            images, targets = next(iterator)
        nonblocking = is_cuda and bool(getattr(loader, "pin_memory", False))
        images = [image.to(device, non_blocking=nonblocking) for image in images]
        targets = [
            {key: value.to(device, non_blocking=nonblocking) for key, value in target.items()}
            for target in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=is_cuda):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite loss before optimizer update {completed + 1}.")
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        is_updated = scaler.get_scale() >= scale_before
        if is_updated:
            scheduler.step()
            completed += 1
            progress.update(1)
        values = (
            torch.stack([loss.detach(), *[value.detach() for value in losses.values()]])
            .cpu()
            .tolist()
        )
        if is_cuda:
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        history.append(
            {
                "step": completed,
                "attempt": attempts,
                "loss": values[0],
                **dict(zip(losses, values[1:], strict=True)),
                "seconds": seconds,
                "images_per_second": len(images) / seconds,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "optimizer_updated": is_updated,
                "image_count": len(images),
            }
        )
        if on_step is not None and on_step(history[-1]):
            break
        if checkpoint_path is not None and is_updated and completed % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, completed, config or {}, scaler
            )
    progress.close()
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            completed,
            config or {},
            scaler,
        )
    return history


def predict_loader(
    model: torch.nn.Module,
    loader: Iterable[tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]],
    device: torch.device,
    warmup: int = 1,
) -> tuple[list[dict], list[int], dict]:
    """Predict source-resolution boxes and time decoding, transfers, model, and CPU outputs.

    Args:
        model: Detector to evaluate in inference mode.
        loader: Validation loader with deterministic frame order.
        device: Explicit evaluation device.
        warmup: Untimed inference passes on the first batch.

    Returns:
        CPU predictions, source image IDs, and synchronized end-to-end speed metadata.
    """
    model.eval()
    is_cuda = device.type == "cuda"
    nonblocking = is_cuda and bool(getattr(loader, "pin_memory", False))
    with torch.inference_mode():
        first_images, _ = next(iter(loader))
        for _ in range(warmup):
            with torch.autocast(device.type, dtype=torch.float16, enabled=is_cuda):
                model([image.to(device, non_blocking=nonblocking) for image in first_images])
        if is_cuda:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        predictions, image_ids = [], []
        for images, targets in loader:
            with torch.autocast(device.type, dtype=torch.float16, enabled=is_cuda):
                batch = model([image.to(device, non_blocking=nonblocking) for image in images])
            predictions.extend(
                {key: value.cpu() for key, value in prediction.items()} for prediction in batch
            )
            image_ids.extend(int(target["image_id"].item()) for target in targets)
        if is_cuda:
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
    return (
        predictions,
        image_ids,
        {
            "frames_per_second": len(image_ids) / seconds,
            "milliseconds_per_frame": seconds * 1000 / len(image_ids),
            "image_count": len(image_ids),
            "warmup_batches": warmup,
            "batch_size": getattr(loader, "batch_size", None),
            "device": str(device),
            "amp": is_cuda,
            "timing_scope": (
                "Decode, collation, transfer, model transform, inference, NMS, CPU outputs. "
                "Excludes COCO evaluation."
            ),
        },
    )


def run_smoke_test(
    root: Path,
    train_frames: pl.DataFrame,
    val_frames: pl.DataFrame,
    objects: pl.DataFrame,
    training: dict,
    smoke: dict,
    output_directory: Path,
    allow_download: bool = False,
) -> dict:
    """Exercise the notebook's bounded pretrained CUDA path for all three configurations.

    Benchmark a few decoding batches, then perform at most ten total optimizer steps.
    Use identical small selections and resize settings for each variant. Save only
    diagnostic JSON records, not graded results or large diagnostic model checkpoints.

    Args:
        root: Original or explicitly staged diagnostic dataset root.
        train_frames: At most eight real official training frames.
        val_frames: At most eight real official validation frames.
        objects: Prepared ordinary GT, including targets that overlap ignored regions.
        training: Shared optimizer, schedule, flip probability, and seed settings.
        smoke: Small resize, batch size, worker count, and per-run step settings.
        output_directory: Ignored diagnostic output directory.
        allow_download: Whether the official pretrained weight download is authorized.

    Returns:
        Device details, data-loader benchmarks, and measured diagnostic run records.
    """
    steps = smoke["steps_per_run"]
    if not 1 <= steps <= 3 or train_frames.height > 8 or val_frames.height > 8:
        raise ValueError("Smoke tests require 1 to 3 steps per run and at most 8 frames per split.")
    if not torch.cuda.is_available():
        raise RuntimeError("The pretrained GPU smoke test requires a connected CUDA kernel.")
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    dataset = DetectionDataset(root, train_frames, objects, training["flip_probability"])
    validation = DetectionDataset(root, val_frames, objects)
    benchmarks = []
    for workers in (0, smoke["num_workers"]):
        loader = make_loader(dataset, smoke["batch_size"], workers, device, seed=training["seed"])
        started = time.perf_counter()
        iterator = iter(loader)
        next(iterator)
        startup_seconds = time.perf_counter() - started
        started = time.perf_counter()
        count = sum(len(images) for images, _ in iterator)
        elapsed = time.perf_counter() - started
        benchmarks.append(
            {
                "num_workers": workers,
                "startup_seconds": startup_seconds,
                "measured_images": count,
                "images_per_second": count / elapsed,
                "scope": "Image decode and CPU collation only",
            }
        )
        del iterator, loader
    output_directory.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for run in VARIANTS:
        LOGGER.info("Run %s: start bounded pretrained GPU smoke", run)
        model = build_retinanet(
            run,
            allow_download=allow_download,
            min_size=smoke["min_size"],
            max_size=smoke["max_size"],
            seed=training["seed"],
        ).to(device)
        optimizer, scheduler = make_optimizer(
            model, training["learning_rate"], training["max_steps"], training["warmup_steps"]
        )
        loader = make_loader(
            dataset,
            smoke["batch_size"],
            smoke["num_workers"],
            device,
            should_shuffle=True,
            seed=training["seed"],
        )
        val_loader = make_loader(validation, 1, smoke["num_workers"], device)
        scaler = torch.amp.GradScaler("cuda")
        initial = model.head.regression_head.bbox_reg.weight.detach().clone()
        torch.cuda.reset_peak_memory_stats(device)
        history = train_steps(
            model, loader, optimizer, scheduler, device, steps, scaler=scaler, max_attempts=3
        )
        has_parameter_update = not torch.equal(initial, model.head.regression_head.bbox_reg.weight)
        if not has_parameter_update or not any(row["optimizer_updated"] for row in history):
            raise RuntimeError(f"No optimizer parameter update was observed for {run}.")
        predictions, image_ids, speed = predict_loader(model, val_loader, device)
        for prediction, record in zip(predictions, validation.records, strict=True):
            boxes = prediction["boxes"]
            has_valid_boxes = (
                torch.isfinite(boxes).all()
                & (boxes >= 0).all()
                & (boxes[:, 0::2] <= record["image_width"]).all()
                & (boxes[:, 1::2] <= record["image_height"]).all()
            )
            if not has_valid_boxes:
                raise ValueError("Smoke predictions are not finite original-image pixel boxes.")
        records = predictions_to_coco(predictions, image_ids)
        metrics = evaluate_coco(root / "val/annotations_coco.json", records, image_ids)
        measured = history[1:]
        result = {
            "run": run,
            "status": "Diagnostic only",
            "initialization": "COCO-pretrained",
            "training_config": training,
            "smoke_config": smoke,
            "training_steps": steps,
            "optimizer_updates": sum(row["optimizer_updated"] for row in history),
            "has_parameter_update": has_parameter_update,
            "history": history,
            "measured_steps_after_warmup": len(measured),
            "training_images_per_second": (
                (len(measured) * smoke["batch_size"] / sum(row["seconds"] for row in measured))
                if measured
                else None
            ),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "inference": speed,
            "diagnostic_metrics": metrics,
            "train_image_keys": train_frames["image_key"].to_list(),
            "val_image_keys": val_frames["image_key"].to_list(),
            "input_shapes": [
                [3, row["image_height"], row["image_width"]]
                for row in train_frames.iter_rows(named=True)
            ],
            "prediction_shapes": [list(prediction["boxes"].shape) for prediction in predictions],
            "predicted_foreground_labels": sorted({record["category_id"] for record in records}),
            "background_predictions_filtered": sum(
                int((prediction["labels"] == 0).sum()) for prediction in predictions
            ),
        }
        (output_directory / f"{run}_predictions.json").write_text(json.dumps(records) + "\n")
        (output_directory / f"{run}_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        LOGGER.info("Run %s: %s optimizer updates verified", run, result["optimizer_updates"])
        del model, optimizer, scheduler, scaler, initial, loader, val_loader
        torch.cuda.empty_cache()
    summary = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "loader_benchmarks": benchmarks,
        "runs": results,
        "attempted_training_steps": sum(len(result["history"]) for result in results),
        "total_optimizer_steps": sum(result["optimizer_updates"] for result in results),
        "graded_training": "Not run",
    }
    (output_directory / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
