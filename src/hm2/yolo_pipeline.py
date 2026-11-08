import hashlib
import json
import logging
import math
import shutil
import threading
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import polars as pl
import torch
import ultralytics
from ultralytics.nn.tasks import DetectionModel

from hm2.data import CLASS_NAMES
from hm2.dense_evaluation import SUBMISSION_DETECTION_CAP, evaluate_coco_max_dets
from hm2.submission import validate_submission
from hm2.yolo_data import resize_long_side

LOGGER = logging.getLogger(__name__)
REQUIRED_TABLES = ("dataset", "model", "training", "inference")
PASSTHROUGH_TRAINING_KEYS = (
    "seed",
    "batch",
    "workers",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_bias_lr",
    "cos_lr",
    "close_mosaic",
    "patience",
    "amp",
    "deterministic",
    "mosaic",
    "mixup",
    "copy_paste",
    "scale",
    "translate",
    "degrees",
    "shear",
    "perspective",
    "fliplr",
    "flipud",
    "hsv_h",
    "hsv_s",
    "hsv_v",
)


class Detector(Protocol):
    """Object detector with the Ultralytics prediction interface used by this pipeline."""

    def predict(
        self,
        source: list[np.ndarray],
        *,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        half: bool,
        device: int | str,
        verbose: bool,
    ) -> list:
        """Return one result with boxes.xyxy, boxes.conf and boxes.cls per input image."""
        ...


def load_pipeline_settings(path: Path) -> dict:
    """Read the YOLO pipeline configuration and require its four tables.

    Args:
        path: TOML configuration file in config/.

    Returns:
        Configuration tables keyed by dataset, model, training and inference.
    """
    settings = tomllib.loads(path.read_text())
    missing = [table for table in REQUIRED_TABLES if table not in settings]
    if missing:
        raise ValueError(f"Missing configuration tables: {missing}")
    if settings["model"]["backend"] != "ultralytics":
        raise ValueError("This pipeline uses the native Ultralytics backend.")
    return settings


def training_arguments(
    settings: dict, data_yaml: Path, project: Path, name: str, epochs: int
) -> dict:
    """Map the visible configuration to native Ultralytics training arguments.

    Every recipe value is passed explicitly, so a changed library default cannot alter
    the run silently. The prepared images already have the training size, the run
    directory is fixed for resumption, and caching stays disabled to bound memory.

    Args:
        settings: Pipeline configuration tables.
        data_yaml: Prepared Ultralytics dataset description.
        project: Directory containing native run directories.
        name: Run directory name.
        epochs: Planned number of epochs.

    Returns:
        Keyword arguments for ultralytics.YOLO.train.
    """
    training = settings["training"]
    return {
        "data": str(data_yaml),
        "imgsz": settings["dataset"]["image_size"],
        "epochs": epochs,
        "device": training["device"],
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "cache": False,
        "rect": False,
        "val": True,
        "save": True,
        "plots": True,
        **{key: training[key] for key in PASSTHROUGH_TRAINING_KEYS},
    }


def plan_epochs(
    train_seconds_per_image: float,
    validation_seconds: float,
    train_images: int,
    training_settings: dict,
) -> int:
    """Choose the largest epoch count whose measured cost fits the training budget.

    One epoch costs one pass over the training images plus one full validation. The
    validation time is inflated by a margin because later checkpoints keep more boxes.
    A budget below the configured minimum is an error rather than a silent reduction.

    Args:
        train_seconds_per_image: Measured training seconds per image on the target GPUs.
        validation_seconds: Measured seconds for one full validation pass.
        train_images: Number of prepared training images.
        training_settings: The training table with budget_hours, min_epochs, max_epochs
            and validation_time_margin.

    Returns:
        Planned number of epochs.
    """
    epoch_seconds = (
        train_seconds_per_image * train_images
        + validation_seconds * training_settings["validation_time_margin"]
    )
    epochs = math.floor(training_settings["budget_hours"] * 3600 / epoch_seconds)
    if epochs < training_settings["min_epochs"]:
        raise ValueError(
            f"The budget fits {epochs} epochs of {epoch_seconds:.0f} s, below min_epochs. "
            "Increase budget_hours or use faster hardware."
        )
    return min(epochs, training_settings["max_epochs"])


def _load_resized(path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return resize_long_side(image, image_size)


def predict_frames(
    model: Detector,
    frames: pl.DataFrame,
    image_root: Path,
    image_size: int,
    inference_settings: dict,
    device: int | str,
    loader_threads: int = 4,
) -> list[dict[str, np.ndarray]]:
    """Detect objects on original frames and return boxes in original-image pixels.

    Frames are resized with the same rule as the prepared training images and grouped by
    shape for minimal letterbox padding. The next batch is decoded in parallel threads
    while the current one runs on the GPU.
    Ultralytics returns boxes in resized coordinates; we scale each axis back by its own
    exact factor and clip to the original image.

    Args:
        model: Loaded ultralytics.YOLO detector.
        frames: Rows with relative_path, image_width and image_height.
        image_root: Directory against which relative_path is resolved.
        image_size: Longer side of the network input in pixels.
        inference_settings: The inference table with confidence, iou, max_detections,
            batch and half.
        device: CUDA device index or "cpu".
        loader_threads: Threads used to decode and resize frames.

    Returns:
        One mapping per input row, in input order, with xyxy boxes, scores and classes 0-7.
    """
    ordered = frames.with_row_index("row").sort("image_width", "image_height", "row")
    batches = [
        group.slice(start, inference_settings["batch"])
        for _, group in ordered.group_by(["image_width", "image_height"], maintain_order=True)
        for start in range(0, group.height, inference_settings["batch"])
    ]
    results: list[dict[str, np.ndarray]] = [{} for _ in range(frames.height)]
    with ThreadPoolExecutor(max_workers=loader_threads) as executor:

        def submit(batch: pl.DataFrame) -> list[Future[np.ndarray]]:
            return [
                executor.submit(_load_resized, image_root / path, image_size)
                for path in batch["relative_path"]
            ]

        pending = submit(batches[0]) if batches else []
        for index, batch in enumerate(batches):
            images = [future.result() for future in pending]
            if index + 1 < len(batches):
                pending = submit(batches[index + 1])
            outputs = model.predict(
                images,
                imgsz=image_size,
                conf=inference_settings["confidence"],
                iou=inference_settings["iou"],
                max_det=inference_settings["max_detections"],
                half=inference_settings["half"],
                device=device,
                verbose=False,
            )
            for row, image, output in zip(
                batch.iter_rows(named=True), images, outputs, strict=True
            ):
                width, height = row["image_width"], row["image_height"]
                scale = np.array(
                    [width / image.shape[1], height / image.shape[0]] * 2, dtype=np.float32
                )
                boxes = output.boxes.xyxy.cpu().numpy().astype(np.float32) * scale
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
                results[row["row"]] = {
                    "boxes": boxes,
                    "scores": output.boxes.conf.cpu().numpy().astype(np.float32),
                    "classes": output.boxes.cls.cpu().numpy().astype(np.int64),
                }
    return results


def coco_detections(predictions: list[dict[str, np.ndarray]], image_ids: list[int]) -> list[dict]:
    """Convert original-pixel predictions to COCO detection records with category IDs 1-8.

    Args:
        predictions: Outputs of predict_frames.
        image_ids: COCO image identifiers aligned with the predictions.

    Returns:
        COCO detections with top-left xywh boxes, omitting degenerate boxes.
    """
    records = []
    for prediction, image_id in zip(predictions, image_ids, strict=True):
        boxes = prediction["boxes"]
        sizes = boxes[:, 2:] - boxes[:, :2]
        valid = (sizes > 0).all(axis=1)
        records.extend(
            {
                "image_id": image_id,
                "category_id": int(label) + 1,
                "bbox": [float(x), float(y), float(width), float(height)],
                "score": float(score),
            }
            for (x, y), (width, height), score, label in zip(
                boxes[valid, :2],
                sizes[valid],
                prediction["scores"][valid],
                prediction["classes"][valid],
                strict=True,
            )
        )
    return records


def evaluate_predictions(annotation_path: Path, records: list[dict], image_ids: list[int]) -> dict:
    """Evaluate COCO AP with crowd regions ignored at the canonical and submission caps.

    Args:
        annotation_path: Original COCO JSON of the official split.
        records: COCO detections in original-image pixels.
        image_ids: Every evaluated COCO image identifier, including frames without boxes.

    Returns:
        Metrics keyed by max_dets_100 and max_dets_500.
    """
    return {
        f"max_dets_{cap}": evaluate_coco_max_dets(
            annotation_path, records, image_ids, cap, backend="faster_coco_eval"
        )
        for cap in (100, SUBMISSION_DETECTION_CAP)
    }


def build_yolo_submission(
    predictions: list[dict[str, np.ndarray]], test_frames: pl.DataFrame
) -> pl.DataFrame:
    """Format predictions as competition rows in test_images.csv order and validate them.

    Each row keeps at most 500 highest-scoring boxes as class, score and top-left pixel
    x, y, width, height rounded to 0.01 pixels. Boxes whose rounded width or height is
    zero are dropped, so every written box is valid. Frames without detections keep an
    empty string.

    Args:
        predictions: Outputs of predict_frames aligned with test_frames.
        test_frames: The original test_images.csv table.

    Returns:
        A validated image_id and PredictionString table covering every test image.
    """
    rows = []
    for prediction in predictions:
        corners = np.round(prediction["boxes"][:, :2].astype(np.float64), 2)
        sizes = np.round((prediction["boxes"][:, 2:] - prediction["boxes"][:, :2]), 2)
        valid = np.flatnonzero((sizes > 0).all(axis=1))
        order = valid[np.argsort(-prediction["scores"][valid], kind="stable")]
        order = order[:SUBMISSION_DETECTION_CAP]
        tokens = [
            f"{prediction['classes'][index]} {prediction['scores'][index]:.5f} "
            f"{corners[index, 0]:.2f} {corners[index, 1]:.2f} "
            f"{sizes[index, 0]:.2f} {sizes[index, 1]:.2f}"
            for index in order
        ]
        rows.append(" ".join(tokens))
    submission = pl.DataFrame(
        {"image_id": test_frames["image_id"], "PredictionString": rows},
        schema={"image_id": pl.String, "PredictionString": pl.String},
    )
    validate_submission(submission, test_frames.select("image_id"))
    return submission


def _copy_atomically(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".incoming")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


class CheckpointMirror:
    """Background publisher of completed native Ultralytics checkpoints and results.

    The native trainer may run in a DDP subprocess, where notebook callbacks are not
    available. The mirror polls weights/last.pt, waits until it is no longer being
    written, copies the run files into a flat directory and calls publish on it.

    Attributes:
        errors: Exceptions raised while copying or publishing, reported after training.

    Args:
        run_directory: Native Ultralytics run directory.
        mirror_directory: Flat directory passed to publish.
        publish: Callable that durably stores the mirror directory.
        interval_seconds: Polling interval.
        settle_seconds: Minimum checkpoint age before it is copied.
    """

    errors: list[Exception]

    def __init__(
        self,
        run_directory: Path,
        mirror_directory: Path,
        publish: Callable[[Path], None],
        interval_seconds: float = 120.0,
        settle_seconds: float = 15.0,
    ) -> None:
        self.run_directory = run_directory
        self.mirror_directory = mirror_directory
        self.publish = publish
        self.interval_seconds = interval_seconds
        self.settle_seconds = settle_seconds
        self.errors = []
        self._published: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def mirror_once(self) -> bool:
        """Copy and publish the current checkpoint when it is new and completely written.

        Returns:
            Whether a new checkpoint generation was published.
        """
        last = self.run_directory / "weights" / "last.pt"
        if not last.is_file():
            return False
        status = last.stat()
        signature = (status.st_mtime_ns, status.st_size)
        if signature == self._published or time.time() - status.st_mtime < self.settle_seconds:
            return False
        self.mirror_directory.mkdir(parents=True, exist_ok=True)
        sources = [
            last,
            self.run_directory / "weights" / "best.pt",
            self.run_directory / "results.csv",
        ]
        for source in sources:
            if source.is_file():
                _copy_atomically(source, self.mirror_directory / source.name)
        status = last.stat()
        if (status.st_mtime_ns, status.st_size) != signature:
            return False
        self.publish(self.mirror_directory)
        self._published = signature
        LOGGER.info("Published checkpoint mirror of %s", last)
        return True

    def _poll(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.mirror_once()
            except Exception as error:
                LOGGER.exception("Checkpoint mirror failed")
                self.errors.append(error)
                return

    def start(self) -> None:
        """Start polling in a daemon thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop polling, then publish the final checkpoint and raise any recorded failure."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        if self.errors:
            raise RuntimeError("Checkpoint mirroring failed during training.") from self.errors[0]
        self.settle_seconds = 0.0
        self.mirror_once()


def _pairwise_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (bottom_right - top_left).clip(0).prod(axis=2)
    first_area = (first[:, 2:] - first[:, :2]).prod(axis=1)
    second_area = (second[:, 2:] - second[:, :2]).prod(axis=1)
    return intersection / (first_area[:, None] + second_area[None, :] - intersection)


def classify_ground_truth_outcomes(
    predictions: list[dict[str, np.ndarray]],
    frames: pl.DataFrame,
    objects: pl.DataFrame,
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> pl.DataFrame:
    """Label every ground-truth object as detected, mislocalized, misclassified or missed.

    Confident predictions are matched greedily by descending score to unmatched objects
    of the same class at the IoU threshold. An unmatched object is misclassified when a
    confident box of another class reaches the threshold, mislocalized when a same-class
    box overlaps it with IoU of at least 0.1, and missed otherwise.

    Args:
        predictions: Outputs of predict_frames aligned with frames.
        frames: Evaluated frame metadata with image_key.
        objects: Target annotations in original-image pixels with class_id and area.
        score_threshold: Minimum confidence of a counted prediction.
        iou_threshold: IoU required for a correct detection.

    Returns:
        One row per ground-truth object with image_key, class_id, area and outcome.
    """
    grouped = objects.partition_by("image_key", as_dict=True)
    rows = []
    for prediction, image_key in zip(predictions, frames["image_key"], strict=True):
        truth = grouped.get((image_key,))
        if truth is None:
            continue
        boxes = truth.select("x", "y", "width", "height").to_numpy().astype(np.float64)
        boxes[:, 2:] += boxes[:, :2]
        classes = truth["class_id"].to_numpy()
        keep = prediction["scores"] >= score_threshold
        order = np.argsort(-prediction["scores"][keep], kind="stable")
        predicted_boxes = prediction["boxes"][keep][order]
        predicted_classes = prediction["classes"][keep][order]
        overlaps = (
            _pairwise_iou(predicted_boxes, boxes)
            if len(predicted_boxes)
            else np.zeros((0, len(boxes)))
        )
        same_class = predicted_classes[:, None] == classes[None, :]
        is_matched = np.zeros(len(boxes), dtype=bool)
        for index in range(len(predicted_boxes)):
            candidates = np.where(same_class[index] & ~is_matched, overlaps[index], 0.0)
            best = int(candidates.argmax()) if len(boxes) else 0
            if len(boxes) and candidates[best] >= iou_threshold:
                is_matched[best] = True
        best_same = np.where(same_class, overlaps, 0.0).max(axis=0, initial=0.0)
        best_other = np.where(~same_class, overlaps, 0.0).max(axis=0, initial=0.0)
        outcome = np.select(
            [is_matched, best_other >= iou_threshold, best_same >= 0.1],
            ["Detected", "Misclassified", "Mislocalized"],
            default="Missed",
        )
        rows.append(
            pl.DataFrame(
                {
                    "image_key": image_key,
                    "class_id": classes,
                    "area": truth["area"].to_numpy(),
                    "outcome": outcome,
                }
            )
        )
    return pl.concat(rows)


def remap_detection_head(
    source: Path, destination: Path, source_channels: tuple[int, ...], model_config: str
) -> dict:
    """Convert a detector checkpoint to the eight competition classes by selecting head channels.

    Every tensor except the final classification convolutions is copied unchanged. Each
    competition class takes the weights and bias of one source class channel, so the
    converted network reproduces the source scores for the selected classes exactly.

    Args:
        source: Ultralytics detection checkpoint with a larger class head.
        destination: Path of the converted checkpoint.
        source_channels: Source class index for each competition class, in class-ID order.
        model_config: Ultralytics architecture YAML matching the source network.

    Returns:
        Provenance of the conversion: source SHA256, channel mapping and source class names.
    """
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    network = (checkpoint.get("ema") or checkpoint["model"]).float()
    converted = DetectionModel(model_config, nc=len(CLASS_NAMES), verbose=False)
    target_state = converted.state_dict()
    source_state = network.state_dict()
    state = {}
    for name, value in target_state.items():
        original = source_state[name]
        if original.shape == value.shape:
            state[name] = original
        elif original.shape[1:] == value.shape[1:] and ".cv3." in name:
            state[name] = original[list(source_channels)]
        else:
            raise ValueError(f"Incompatible tensor {name}: {original.shape} -> {value.shape}")
    converted.load_state_dict(state, strict=True)
    converted.names = dict(enumerate(CLASS_NAMES))
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {
        "source_sha256": digest,
        "source_channels": list(source_channels),
        "source_names": {int(key): value for key, value in network.names.items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": converted.half(),
            "train_args": {"task": "detect", "single_cls": False},
            "epoch": -1,
            "date": None,
            "version": ultralytics.__version__,
            "head_remap": provenance,
        },
        destination,
    )
    return provenance


def write_coco_subset(source_root: Path, frames: pl.DataFrame, destination: Path) -> list[int]:
    """Write one COCO file for frames from either official split, keeping crowd regions.

    Images are renumbered in the order of frames, and every annotation of a selected image,
    including category-specific iscrowd regions, is copied with a new unique ID.

    Args:
        source_root: Competition directory with train/ and val/ COCO annotations.
        frames: Selected frame metadata with split and the original COCO image_id.
        destination: Path of the written COCO JSON.

    Returns:
        New COCO image IDs aligned with the rows of frames.
    """
    keys = [(row["split"], int(row["image_id"])) for row in frames.iter_rows(named=True)]
    new_ids = {key: index for index, key in enumerate(keys, 1)}
    images, annotations, categories = [], [], []
    for split in sorted({split for split, _ in keys}):
        document = json.loads((source_root / split / "annotations_coco.json").read_text())
        categories = document["categories"]
        images.extend(
            {**image, "id": new_ids[(split, image["id"])]}
            for image in document["images"]
            if (split, image["id"]) in new_ids
        )
        annotations.extend(
            {**annotation, "image_id": new_ids[(split, annotation["image_id"])]}
            for annotation in document["annotations"]
            if (split, annotation["image_id"]) in new_ids
        )
    for index, annotation in enumerate(annotations, 1):
        annotation["id"] = index
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "images": sorted(images, key=lambda image: image["id"]),
                "annotations": annotations,
                "categories": categories,
            }
        )
    )
    return [new_ids[key] for key in keys]
