import json
import logging
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import polars as pl

CLASS_NAMES = ("person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor")
COMPETITION = "multiple-object-video-detection-hw-2-ucu-ai-2026"
LOGGER = logging.getLogger(__name__)


def find_dataset_root(directory: Path) -> Path | None:
    """Find the dataset root independently of an archive's wrapper directory.

    Args:
        directory: Directory containing extracted competition files.

    Returns:
        The unique directory containing test metadata, or None when none exists.
    """
    roots = sorted({path.parent for path in directory.rglob("test_images.csv")})
    if len(roots) > 1:
        raise ValueError("Multiple datasets found. Set HM2_DATA_DIR to one dataset root.")
    return roots[0] if roots else None


def extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract a ZIP without replacing existing files or writing outside its destination.

    Args:
        archive_path: Local competition archive.
        destination: Directory in which to preserve the archive's structure.
    """
    destination = destination.resolve()
    with ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive path escapes its destination: {entry.filename}")
            if not entry.is_dir() and target.exists():
                raise FileExistsError(f"Refusing to overwrite {target}. Use an empty destination.")
        LOGGER.info("Extracting %s into %s", archive_path.name, destination)
        archive.extractall(destination)


def yolo_to_xywh(boxes: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Convert normalized center boxes to original-pixel top-left boxes.

    Args:
        boxes: An array of shape (N, 4) containing normalized cx, cy, width, height.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        An array of shape (N, 4) containing pixel x, y, width, height.
    """
    pixels = np.asarray(boxes, dtype=float) * ([image_width, image_height] * 2)
    pixels[:, :2] -= pixels[:, 2:] / 2
    return pixels


def xywh_to_yolo(boxes: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Convert original-pixel top-left boxes to normalized center boxes.

    Args:
        boxes: An array of shape (N, 4) containing pixel x, y, width, height.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        An array of shape (N, 4) containing normalized cx, cy, width, height.
    """
    normalized = np.asarray(boxes, dtype=float).copy()
    normalized[:, :2] += normalized[:, 2:] / 2
    return normalized / ([image_width, image_height] * 2)


def read_yolo_boxes(
    path: Path, image_width: int, image_height: int, is_ignore: bool = False
) -> np.ndarray:
    """Read normalized labels or optional ignored regions as pixel boxes.

    Args:
        path: A YOLO label file or an optional ignored-region file.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.
        is_ignore: Whether rows omit the class and a missing file means no regions.

    Returns:
        Rows of class ID, x, y, width, height, using -1 for ignored-region classes.
    """
    if is_ignore and not path.exists():
        return np.empty((0, 5))
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return np.empty((0, 5))
    expected_columns = 4 if is_ignore else 5
    if any(len(row) != expected_columns for row in rows):
        raise ValueError(f"Expected {expected_columns} columns in {path}.")
    values = np.asarray(rows, dtype=float)
    classes = np.full(len(values), -1) if is_ignore else values[:, 0]
    if not is_ignore and not np.isin(classes, np.arange(8)).all():
        raise ValueError(f"Unexpected target class in {path}.")
    return np.column_stack((classes, yolo_to_xywh(values[:, -4:], image_width, image_height)))


def _read_coco(root: Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    LOGGER.info("Reading %s COCO metadata", split)
    document = json.loads((root / split / "annotations_coco.json").read_text())
    categories = {
        category["id"]: category["name"]
        for category in document["categories"]
        if category["id"] in range(1, 9)
    }
    if categories != dict(enumerate(CLASS_NAMES, 1)):
        raise ValueError(f"Unexpected COCO category mapping in {split}: {categories}")
    image_rows = []
    for record in document["images"]:
        filename = Path(record["file_name"])
        clip = filename.parent.name
        relative_path = str(Path(split) / "images" / clip / filename.name)
        image_rows.append(
            {
                "split": split,
                "image_id": str(record["id"]),
                "image_key": f"{split}/{clip}/{filename.stem}",
                "clip": clip,
                "frame": filename.stem,
                "frame_id": int(record.get("frame_id", filename.stem)),
                "relative_path": relative_path,
                "image_width": record["width"],
                "image_height": record["height"],
                "video": str(record.get("video", clip)),
                "video_id": record.get("video_id"),
                "is_labeled": True,
            }
        )
    frames = pl.from_dicts(image_rows).with_columns(pl.col("video_id").cast(pl.Int64))
    annotations = (
        pl.from_dicts(
            document["annotations"],
            schema={
                "id": pl.Int64,
                "image_id": pl.Int64,
                "category_id": pl.Int64,
                "bbox": pl.List(pl.Float64),
                "iscrowd": pl.Int64,
                "occlusion": pl.Int64,
                "truncation": pl.Int64,
                "track_id": pl.Int64,
                "orig_category_id": pl.Int64,
            },
        )
        .with_columns(
            pl.lit(split).alias("split"),
            pl.col("image_id").cast(pl.String),
            pl.col("iscrowd").fill_null(0).cast(pl.Boolean).alias("is_crowd"),
            *[
                pl.col("bbox").list.get(i).alias(name)
                for i, name in enumerate(("x", "y", "width", "height"))
            ],
        )
        .drop("bbox", "iscrowd")
        .rename({"id": "annotation_id"})
    )
    annotations = annotations.join(frames, on=["split", "image_id"], how="left").with_columns(
        (pl.col("width") * pl.col("height")).alias("area"),
        (pl.col("width") / pl.col("height")).alias("aspect_ratio"),
    )
    return frames, annotations


def load_tables(root: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build compact image, target-object, and ignored-region tables from source metadata.

    Read COCO metadata for labeled splits and CSV metadata for test images. Keep
    crowd regions separate and keep target counts null for unlabeled test frames.
    Group repeated COCO crowd geometry across categories and retain all source IDs.
    Preserve raw attributes and original coordinates without clipping annotations.

    Args:
        root: Extracted competition directory containing train, val, and test metadata.

    Returns:
        Image metadata, target annotations, and ignored regions as three Polars tables.
    """
    labeled = [_read_coco(root, split) for split in ("train", "val")]
    test = pl.read_csv(root / "test_images.csv", schema_overrides={"image_id": pl.String})
    test_frames = test.select(
        pl.lit("test").alias("split"),
        "image_id",
        (
            pl.lit("test/")
            + pl.col("clip")
            + "/"
            + pl.col("file_name").str.split("/").list.last().str.strip_suffix(".jpg")
        ).alias("image_key"),
        "clip",
        pl.col("file_name").str.split("/").list.last().str.strip_suffix(".jpg").alias("frame"),
        pl.col("frame").cast(pl.Int64).alias("frame_id"),
        pl.col("file_name").alias("relative_path"),
        pl.col("width").alias("image_width"),
        pl.col("height").alias("image_height"),
        pl.col("clip").alias("video"),
        pl.lit(None, dtype=pl.Int64).alias("video_id"),
        pl.lit(False).alias("is_labeled"),
    )
    frames = pl.concat([pair[0] for pair in labeled] + [test_frames], how="vertical_relaxed")
    annotations = pl.concat([pair[1] for pair in labeled], how="vertical_relaxed")
    raw_ignored = annotations.filter(pl.col("is_crowd"))
    region_columns = [
        column for column in raw_ignored.columns if column not in ("annotation_id", "category_id")
    ]
    ignored = raw_ignored.group_by(region_columns, maintain_order=True).agg(
        pl.col("annotation_id").alias("coco_annotation_ids"),
        pl.col("category_id").alias("coco_category_ids"),
        pl.len().alias("coco_replication_count"),
    )
    objects = (
        annotations.filter(~pl.col("is_crowd"))
        .with_columns(
            (pl.col("category_id") - 1).alias("class_id"),
        )
        .with_columns(
            pl.col("class_id").replace_strict(dict(enumerate(CLASS_NAMES))).alias("class_name"),
            pl.when(pl.col("area") < 32**2)
            .then(pl.lit("Small"))
            .when(pl.col("area") < 96**2)
            .then(pl.lit("Medium"))
            .otherwise(pl.lit("Large"))
            .alias("size_group"),
        )
    )
    for table, name in ((objects, "target_count"), (ignored, "ignore_count")):
        counts = table.group_by("image_key").len(name=name)
        frames = frames.join(counts, on="image_key", how="left").with_columns(
            pl.when(pl.col("is_labeled"))
            .then(pl.col(name).fill_null(0))
            .otherwise(None)
            .alias(name)
        )
    return frames.sort("split", "clip", "frame_id"), objects, ignored


def audit_dataset(
    root: Path, frames: pl.DataFrame, objects: pl.DataFrame, ignored: pl.DataFrame
) -> pl.DataFrame:
    """Report essential identifier, file, coordinate, and split-contract violations.

    Compare paths and identifiers against the source inventory and count geometric
    anomalies without editing annotations. Report a missing sample submission
    explicitly and leave its unavailable identifier checks unexecuted.

    Args:
        root: Dataset directory against which image paths are resolved.
        frames: Image metadata including unlabeled test records.
        objects: Target annotations in original-image pixels.
        ignored: Separate COCO crowd annotations in original-image pixels.

    Returns:
        Named checks with anomaly counts and pass indicators, without modifying inputs.
    """
    annotations = pl.concat(
        [
            objects.select("split", "image_key", "annotation_id"),
            ignored.select(
                "split", "image_key", pl.col("coco_annotation_ids").alias("annotation_id")
            ).explode("annotation_id"),
        ]
    )
    expected_paths = {row for row in frames["relative_path"].to_list()}
    actual_paths = {
        str(path.relative_to(root))
        for split in ("train", "val", "test")
        for path in (root / split / "images").glob("*/*.jpg")
    }
    labeled = frames.filter(pl.col("is_labeled"))
    missing_labels = sum(
        not (root / row["split"] / "labels" / row["clip"] / f"{row['frame']}.txt").is_file()
        for row in labeled.iter_rows(named=True)
    )
    train_clips = set(frames.filter(pl.col("split") == "train")["clip"])
    val_clips = set(frames.filter(pl.col("split") == "val")["clip"])
    test_ids = set(frames.filter(pl.col("split") == "test")["image_id"])
    findings = [
        ("Duplicate image keys", frames.height - frames["image_key"].n_unique()),
        ("Duplicate split image IDs", frames.height - frames.unique(["split", "image_id"]).height),
        (
            "Duplicate annotation IDs",
            annotations.height - annotations.unique(["split", "annotation_id"]).height,
        ),
        ("Annotations with unknown image IDs", annotations["image_key"].null_count()),
        ("Missing image files", len(expected_paths - actual_paths)),
        ("Images absent from metadata", len(actual_paths - expected_paths)),
        ("Missing target label files", missing_labels),
        ("Overlapping train and validation clips", len(train_clips & val_clips)),
        ("Missing sample submission", int(not (root / "sample_submission.csv").is_file())),
        (
            "Invalid image dimensions",
            frames.filter((pl.col("image_width") <= 0) | (pl.col("image_height") <= 0)).height,
        ),
    ]
    if (root / "sample_submission.csv").is_file():
        submission = pl.read_csv(
            root / "sample_submission.csv", schema_overrides={"image_id": pl.String}
        )
        findings.extend(
            [
                (
                    "Test versus sample image ID differences",
                    len(test_ids ^ set(submission["image_id"])),
                ),
                (
                    "Duplicate sample image IDs",
                    submission.height - submission["image_id"].n_unique(),
                ),
            ]
        )
    for name, table in (("Target", objects), ("Ignored-region", ignored)):
        invalid = pl.any_horizontal(
            *[~pl.col(column).is_finite() for column in ("x", "y", "width", "height")],
            pl.col("width") <= 0,
            pl.col("height") <= 0,
        )
        outside = (
            (pl.col("x") < 0)
            | (pl.col("y") < 0)
            | (pl.col("x") + pl.col("width") > pl.col("image_width") + 0.01)
            | (pl.col("y") + pl.col("height") > pl.col("image_height") + 0.01)
        )
        findings.extend(
            [
                (f"{name} invalid box dimensions", table.filter(invalid).height),
                (f"{name} boxes outside image", table.filter(outside).height),
            ]
        )
    return pl.DataFrame(findings, schema=["check", "count"], orient="row").with_columns(
        (pl.col("count") == 0).alias("is_pass")
    )


def _match_boxes(expected: np.ndarray, actual: np.ndarray) -> float:
    if expected.shape != actual.shape:
        return float("inf")
    remaining = actual.copy()
    maximum_error = 0.0
    for box in expected:
        distances = np.abs(remaining - box).max(axis=1)
        distances[remaining[:, 0] != box[0]] = np.inf
        nearest = int(distances.argmin())
        maximum_error = max(maximum_error, float(distances[nearest]))
        remaining = np.delete(remaining, nearest, axis=0)
    return maximum_error


def compare_yolo_coco(
    root: Path,
    frames: pl.DataFrame,
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
    sample_size: int = 12,
    seed: int = 2026,
) -> pl.DataFrame:
    """Match sampled YOLO and COCO annotations by full image identity and class.

    Args:
        root: Dataset directory containing normalized label and ignore files.
        frames: Image metadata with original dimensions.
        objects: Target objects with zero-based class IDs.
        ignored: Separate ignored regions from COCO metadata.
        sample_size: Maximum number of labeled images to compare.
        seed: Fixed seed for reproducible image selection.

    Returns:
        Image keys, target and region counts, maximum coordinate errors, and match flags.
    """
    labeled = frames.filter(pl.col("is_labeled")).sort("image_key")
    sample = labeled.sample(n=min(sample_size, labeled.height), seed=seed)
    records = []
    for row in sample.iter_rows(named=True):
        key = row["image_key"]
        width, height = row["image_width"], row["image_height"]
        label_path = root / row["split"] / "labels" / row["clip"] / f"{row['frame']}.txt"
        ignore_path = root / row["split"] / "ignore" / row["clip"] / f"{row['frame']}.txt"
        yolo_objects = read_yolo_boxes(label_path, width, height)
        yolo_ignored = read_yolo_boxes(ignore_path, width, height, is_ignore=True)
        coco_objects = (
            objects.filter(pl.col("image_key") == key)
            .select("class_id", "x", "y", "width", "height")
            .to_numpy()
        )
        coco_ignored = (
            ignored.filter(pl.col("image_key") == key)
            .select(pl.lit(-1).alias("class_id"), "x", "y", "width", "height")
            .to_numpy()
        )
        error = max(
            _match_boxes(coco_objects, yolo_objects), _match_boxes(coco_ignored, yolo_ignored)
        )
        records.append(
            {
                "image_key": key,
                "targets": len(yolo_objects),
                "ignored_regions": len(yolo_ignored),
                "max_error_px": error,
                "is_match": error <= 0.01,
            }
        )
    return pl.from_dicts(records)
