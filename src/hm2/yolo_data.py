import hashlib
import logging
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from PIL import Image

from hm2.data import CLASS_NAMES
from hm2.detection_data import select_frames

LOGGER = logging.getLogger(__name__)
SPLITS = ("train", "val")


@dataclass(frozen=True)
class RenderTask:
    """One labeled frame to resize, optionally mask and write in YOLO layout.

    Attributes:
        split: Official split of the frame, train or val.
        source_image: Original competition JPEG.
        source_label: Original normalized YOLO label file for the same frame.
        target_image: Destination JPEG inside the prepared dataset.
        target_label: Destination label file inside the prepared dataset.
        width: Original image width recorded in metadata.
        height: Original image height recorded in metadata.
        ignored_boxes: Original-pixel x, y, width, height of ignored regions.
        target_boxes: Original-pixel x, y, width, height of labeled objects.
    """

    split: str
    source_image: Path
    source_label: Path
    target_image: Path
    target_label: Path
    width: int
    height: int
    ignored_boxes: tuple[tuple[float, float, float, float], ...]
    target_boxes: tuple[tuple[float, float, float, float], ...]


def resize_long_side(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an image so its longer side equals the model input size.

    Downscaling uses area interpolation and upscaling uses bilinear interpolation, which
    matches the Ultralytics loader without augmentation. Training, validation and test
    images therefore share one deterministic resampling rule.

    Args:
        image: Image array with height and width as the first two dimensions.
        size: Target length of the longer side in pixels.

    Returns:
        The resized image, or the input itself when it already has the target size.
    """
    height, width = image.shape[:2]
    ratio = size / max(height, width)
    if ratio == 1:
        return image
    resized = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, resized, interpolation=interpolation)


def _pixel_bounds(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = box
    return (
        max(0, math.floor(x)),
        max(0, math.floor(y)),
        min(width, math.ceil(x + box_width)),
        min(height, math.ceil(y + box_height)),
    )


def mask_ignored_regions(
    image: np.ndarray,
    ignored_boxes: tuple[tuple[float, float, float, float], ...],
    target_boxes: tuple[tuple[float, float, float, float], ...],
    fill: int,
) -> np.ndarray:
    """Paint don't-care regions with a constant value while keeping every labeled object.

    Ignored regions mark crowds and unannotated objects. Painting them removes pixels that
    would otherwise be supervised as background. Labeled boxes that overlap an ignored
    region are restored afterwards, so no annotated object loses its pixels.

    Args:
        image: Original-resolution image array.
        ignored_boxes: Original-pixel x, y, width, height of ignored regions.
        target_boxes: Original-pixel x, y, width, height of labeled objects.
        fill: Constant intensity written into ignored pixels.

    Returns:
        A masked copy, or the input itself when the frame has no ignored regions.
    """
    if not ignored_boxes:
        return image
    height, width = image.shape[:2]
    masked = image.copy()
    for box in ignored_boxes:
        left, top, right, bottom = _pixel_bounds(box, width, height)
        masked[top:bottom, left:right] = fill
    for box in target_boxes:
        left, top, right, bottom = _pixel_bounds(box, width, height)
        masked[top:bottom, left:right] = image[top:bottom, left:right]
    return masked


def parse_yolo_label(text: str) -> np.ndarray:
    """Parse and validate normalized YOLO rows with class IDs 0 to 7.

    Args:
        text: Contents of one YOLO label file, possibly empty.

    Returns:
        An array of shape (N, 5) with class, center x, center y, width and height.
    """
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows:
        return np.empty((0, 5))
    if any(len(row) != 5 for row in rows):
        raise ValueError("YOLO label rows need exactly five values.")
    values = np.asarray(rows, dtype=float)
    classes, centers, sizes = values[:, 0], values[:, 1:3], values[:, 3:5]
    tolerance = 1e-4
    if (
        not np.isin(classes, np.arange(len(CLASS_NAMES))).all()
        or (sizes <= 0).any()
        or (centers - sizes / 2 < -tolerance).any()
        or (centers + sizes / 2 > 1 + tolerance).any()
    ):
        raise ValueError("YOLO labels need classes 0 to 7 and positive boxes inside the image.")
    return values


def render_frame(task: RenderTask, image_size: int, jpeg_quality: int, fill: int | None) -> dict:
    """Write one resized, optionally masked frame and its validated original label file.

    Normalized YOLO coordinates are invariant to the aspect-preserving resize, so the
    original label text is copied byte for byte after validation.

    Args:
        task: Source and destination paths with the frame's annotation geometry.
        image_size: Longer side of the written image in pixels.
        jpeg_quality: JPEG quality of the written image.
        fill: Ignored-region intensity, or None to keep ignored pixels unchanged.

    Returns:
        Frame-level counts: written width and height, labels and masked regions.
    """
    image = cv2.imread(str(task.source_image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {task.source_image}")
    if image.shape[:2] != (task.height, task.width):
        raise ValueError(f"Image size differs from metadata: {task.source_image}")
    label_text = task.source_label.read_text()
    labels = parse_yolo_label(label_text)
    if len(labels) != len(task.target_boxes):
        raise ValueError(f"YOLO and COCO object counts differ: {task.source_label}")
    if fill is not None:
        image = mask_ignored_regions(image, task.ignored_boxes, task.target_boxes, fill)
    image = resize_long_side(image, image_size)
    task.target_image.parent.mkdir(parents=True, exist_ok=True)
    task.target_label.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(task.target_image), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
        raise OSError(f"Could not write {task.target_image}")
    task.target_label.write_text(label_text)
    return {
        "written_width": image.shape[1],
        "written_height": image.shape[0],
        "labels": len(labels),
        "masked_regions": len(task.ignored_boxes) if fill is not None else 0,
    }


def select_dataset_frames(
    frames: pl.DataFrame, frame_stride: int, validation_clips: list[str]
) -> dict[str, pl.DataFrame]:
    """Assign whole labeled clips to training or validation and subsample training frames.

    Validation keeps every frame of the listed clips, which may come from either official
    split. Training uses every Nth frame of all other labeled clips, so no clip is shared.

    Args:
        frames: Image metadata from hm2.data.load_tables.
        frame_stride: Positive temporal stride applied independently inside every clip.
        validation_clips: Labeled clips held out for validation.

    Returns:
        Selected frame metadata keyed by train and val, in deterministic clip and frame order.
    """
    training = pl.concat([select_frames(frames, split, frame_stride) for split in SPLITS])
    validation = pl.concat([select_frames(frames, split) for split in SPLITS])
    return {
        "train": training.filter(~pl.col("clip").is_in(validation_clips)),
        "val": validation.filter(pl.col("clip").is_in(validation_clips)),
    }


def selection_digest(selection: dict[str, pl.DataFrame]) -> str:
    """Hash the ordered selected image keys of every split.

    Args:
        selection: Selected frame metadata keyed by split.

    Returns:
        A SHA256 hex digest identifying the exact frame selection.
    """
    keys = "\n".join(key for split in SPLITS for key in selection[split]["image_key"].to_list())
    return hashlib.sha256(keys.encode()).hexdigest()


def _boxes_by_image(
    table: pl.DataFrame,
) -> dict[str, tuple[tuple[float, float, float, float], ...]]:
    grouped = table.group_by("image_key").agg("x", "y", "width", "height")
    return {
        row["image_key"]: tuple(zip(row["x"], row["y"], row["width"], row["height"], strict=True))
        for row in grouped.iter_rows(named=True)
    }


def build_render_tasks(
    source_root: Path,
    output_root: Path,
    selection: dict[str, pl.DataFrame],
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
) -> list[RenderTask]:
    """Create one render task for every selected train and validation frame.

    Args:
        source_root: Competition directory containing train/ and val/.
        output_root: Writable prepared dataset directory.
        selection: Selected frame metadata keyed by split.
        objects: Target annotations in original-image pixels.
        ignored: Unique ignored regions in original-image pixels.

    Returns:
        Render tasks in split, clip and frame order.
    """
    target_boxes = _boxes_by_image(objects)
    ignored_boxes = _boxes_by_image(ignored)
    tasks = []
    for split in SPLITS:
        for row in selection[split].iter_rows(named=True):
            relative = Path(row["clip"]) / row["frame"]
            tasks.append(
                RenderTask(
                    split=split,
                    source_image=source_root / row["relative_path"],
                    source_label=source_root
                    / row["split"]
                    / "labels"
                    / relative.with_suffix(".txt"),
                    target_image=output_root / "images" / split / relative.with_suffix(".jpg"),
                    target_label=output_root / "labels" / split / relative.with_suffix(".txt"),
                    width=row["image_width"],
                    height=row["image_height"],
                    ignored_boxes=ignored_boxes.get(row["image_key"], ()),
                    target_boxes=target_boxes.get(row["image_key"], ()),
                )
            )
    return tasks


def write_data_yaml(output_root: Path) -> Path:
    """Write the Ultralytics dataset description for the prepared directory.

    Args:
        output_root: Prepared dataset directory with images/ and labels/.

    Returns:
        Path to the written data.yaml file.
    """
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    path = output_root / "data.yaml"
    path.write_text(f"path: {output_root}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return path


def prepare_yolo_dataset(
    source_root: Path,
    output_root: Path,
    frames: pl.DataFrame,
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
    dataset_settings: dict,
    workers: int,
) -> dict:
    """Render the selected frames into a fresh Ultralytics dataset and describe it.

    The output directory is replaced, so stale frames from another recipe cannot leak
    into training. Images are written with threads because OpenCV releases the GIL
    while decoding, resizing and encoding.

    Args:
        source_root: Read-only competition directory.
        output_root: Writable destination, replaced if it already exists.
        frames: Image metadata from hm2.data.load_tables.
        objects: Target annotations in original-image pixels.
        ignored: Unique ignored regions in original-image pixels.
        dataset_settings: The dataset table of the pipeline configuration.
        workers: Number of rendering threads.

    Returns:
        A JSON-compatible manifest with the recipe, selection digest and frame counts.
    """
    started = time.monotonic()
    selection = select_dataset_frames(
        frames, dataset_settings["frame_stride"], dataset_settings["validation_clips"]
    )
    tasks = build_render_tasks(source_root, output_root, selection, objects, ignored)
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True)
    fill = dataset_settings["ignore_fill"] if dataset_settings["should_mask_ignored"] else None
    LOGGER.info("Rendering %d frames into %s with %d threads", len(tasks), output_root, workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                lambda task: render_frame(
                    task, dataset_settings["image_size"], dataset_settings["jpeg_quality"], fill
                ),
                tasks,
            )
        )
    data_yaml = write_data_yaml(output_root)
    split_counts = {}
    for split in SPLITS:
        split_results = [
            result for task, result in zip(tasks, results, strict=True) if task.split == split
        ]
        split_counts[split] = {
            "frames": len(split_results),
            "clips": selection[split]["clip"].n_unique(),
            "labels": sum(result["labels"] for result in split_results),
            "masked_regions": sum(result["masked_regions"] for result in split_results),
            "frames_with_masks": sum(result["masked_regions"] > 0 for result in split_results),
        }
    LOGGER.info("Prepared dataset in %.1f s", time.monotonic() - started)
    return {
        "dataset": dataset_settings,
        "selection_sha256": selection_digest(selection),
        "data_yaml": str(data_yaml),
        "splits": split_counts,
        "render_seconds": time.monotonic() - started,
    }


def verify_yolo_dataset(
    output_root: Path,
    frames: pl.DataFrame,
    objects: pl.DataFrame,
    dataset_settings: dict,
    manifest: dict,
) -> pl.DataFrame:
    """Check the prepared dataset against metadata before any GPU time is spent.

    Verify frame and clip counts, image-label pairing, written image sizes, label
    validity, per-class box counts against COCO metadata, split separation and the
    reproducibility of the frame selection.

    Args:
        output_root: Prepared dataset directory.
        frames: Image metadata from hm2.data.load_tables.
        objects: Target annotations in original-image pixels.
        dataset_settings: The dataset table of the pipeline configuration.
        manifest: Manifest returned by prepare_yolo_dataset.

    Returns:
        Named checks with observed and expected values and a pass indicator.
    """
    selection = select_dataset_frames(
        frames, dataset_settings["frame_stride"], dataset_settings["validation_clips"]
    )
    checks = [
        (
            "Selection digest reproduces",
            manifest["selection_sha256"],
            selection_digest(selection),
        ),
        (
            "Train and validation clips are disjoint",
            0,
            len(set(selection["train"]["clip"]) & set(selection["val"]["clip"])),
        ),
    ]
    expected = {
        "train": (
            dataset_settings["expected_train_frames"],
            dataset_settings["expected_train_clips"],
        ),
        "val": (
            dataset_settings["expected_validation_frames"],
            dataset_settings["expected_validation_clips"],
        ),
    }
    for split in SPLITS:
        images = sorted((output_root / "images" / split).glob("*/*.jpg"))
        labels = sorted((output_root / "labels" / split).glob("*/*.txt"))
        image_stems = {
            path.relative_to(output_root / "images" / split).with_suffix("") for path in images
        }
        label_stems = {
            path.relative_to(output_root / "labels" / split).with_suffix("") for path in labels
        }
        class_counts = np.zeros(len(CLASS_NAMES), dtype=int)
        for path in labels:
            values = parse_yolo_label(path.read_text())
            class_counts += np.bincount(values[:, 0].astype(int), minlength=len(CLASS_NAMES))
        keys = selection[split]["image_key"].implode()
        metadata_counts = (
            objects.filter(pl.col("image_key").is_in(keys))
            .group_by("class_id")
            .len()
            .sort("class_id")
        )
        reference = np.zeros(len(CLASS_NAMES), dtype=int)
        reference[metadata_counts["class_id"].to_numpy()] = metadata_counts["len"].to_numpy()
        long_sides, readable = set(), 0
        for path in images:
            with Image.open(path) as image:
                long_sides.add(max(image.size))
            readable += cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8) is not None
        checks.extend(
            [
                (f"{split}: frames", expected[split][0], len(images)),
                (f"{split}: clips", expected[split][1], len({path.parent.name for path in images})),
                (f"{split}: images without labels", 0, len(image_stems - label_stems)),
                (f"{split}: labels without images", 0, len(label_stems - image_stems)),
                (f"{split}: readable images", len(images), readable),
                (
                    f"{split}: image long sides",
                    [dataset_settings["image_size"]],
                    sorted(long_sides),
                ),
                (f"{split}: boxes per class match COCO", reference.tolist(), class_counts.tolist()),
            ]
        )
    return pl.DataFrame(
        {
            "check": [name for name, _, _ in checks],
            "expected": [str(value) for _, value, _ in checks],
            "observed": [str(value) for _, _, value in checks],
        }
    ).with_columns((pl.col("expected") == pl.col("observed")).alias("is_pass"))
