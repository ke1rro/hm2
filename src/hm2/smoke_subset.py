import csv
import io
import json
import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

import polars as pl
from PIL import Image

from hm2.data import read_yolo_boxes

LOGGER = logging.getLogger(__name__)

FRAME_SCHEMA = {
    "split": pl.String,
    "image_id": pl.String,
    "image_key": pl.String,
    "clip": pl.String,
    "frame": pl.String,
    "frame_id": pl.Int64,
    "relative_path": pl.String,
    "image_width": pl.Int64,
    "image_height": pl.Int64,
    "is_labeled": pl.Boolean,
    "submission_id": pl.String,
}
OBJECT_SCHEMA = {
    "split": pl.String,
    "image_key": pl.String,
    "class_id": pl.Int64,
    "category_id": pl.Int64,
    "x": pl.Float64,
    "y": pl.Float64,
    "width": pl.Float64,
    "height": pl.Float64,
    "area": pl.Float64,
}
REGION_SCHEMA = {
    "split": pl.String,
    "image_key": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "width": pl.Float64,
    "height": pl.Float64,
}


def flight_prefix(clip: str) -> str:
    """Return the VisDrone flight identifier shared by clips cut from one flight.

    Args:
        clip: Clip name such as uav0000013_00000_v.

    Returns:
        The flight prefix, such as uav0000013.
    """
    return clip.split("_")[0]


def check_flight_separation(selection: dict[str, dict[str, list[int]]]) -> None:
    """Reject a selection in which one flight contributes to both train and validation.

    Args:
        selection: Frame numbers keyed by split and then by clip name.
    """
    shared = {flight_prefix(clip) for clip in selection["train"]} & {
        flight_prefix(clip) for clip in selection["val"]
    }
    if shared:
        raise ValueError(f"Train and validation share flights: {sorted(shared)}")


@contextmanager
def _source_reader(source: Path) -> Generator[Callable[[str], bytes | None]]:
    if source.is_file():
        with ZipFile(source) as archive:
            names = set(archive.namelist())
            yield lambda name: archive.read(name) if name in names else None
    else:
        yield lambda name: (source / name).read_bytes() if (source / name).is_file() else None


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _frame_name(frame_id: int) -> str:
    return f"{frame_id:07d}"


def _copy_labeled_files(
    read: Callable[[str], bytes | None], destination: Path, split: str, clips: dict
) -> None:
    for clip, frame_ids in clips.items():
        for frame in map(_frame_name, frame_ids):
            for kind, suffix, is_required in (
                ("images", ".jpg", True),
                ("labels", ".txt", True),
                ("ignore", ".txt", False),
            ):
                name = f"{split}/{kind}/{clip}/{frame}{suffix}"
                content = read(name)
                if content is None and is_required:
                    raise FileNotFoundError(f"Missing {name} in the dataset source.")
                if content is not None:
                    _write(destination / name, content)


def _train_tables(destination: Path, clips: dict) -> tuple[list, list, list]:
    frames, objects, regions = [], [], []
    for clip, frame_ids in clips.items():
        for frame_id in frame_ids:
            frame = _frame_name(frame_id)
            key = f"train/{clip}/{frame}"
            relative_path = f"train/images/{clip}/{frame}.jpg"
            with Image.open(destination / relative_path) as image:
                width, height = image.size
            frames.append(
                {
                    "split": "train",
                    "image_id": str(len(frames) + 1),
                    "image_key": key,
                    "clip": clip,
                    "frame": frame,
                    "frame_id": frame_id,
                    "relative_path": relative_path,
                    "image_width": width,
                    "image_height": height,
                    "is_labeled": True,
                    "submission_id": None,
                }
            )
            labels = read_yolo_boxes(
                destination / f"train/labels/{clip}/{frame}.txt", width, height
            )
            objects.extend(
                {
                    "split": "train",
                    "image_key": key,
                    "class_id": int(class_id),
                    "category_id": int(class_id) + 1,
                    "x": x,
                    "y": y,
                    "width": box_width,
                    "height": box_height,
                    "area": box_width * box_height,
                }
                for class_id, x, y, box_width, box_height in labels.tolist()
            )
            ignored = read_yolo_boxes(
                destination / f"train/ignore/{clip}/{frame}.txt", width, height, is_ignore=True
            )
            regions.extend(
                {"split": "train", "image_key": key, "x": x, "y": y, "width": w, "height": h}
                for _, x, y, w, h in ignored.tolist()
            )
    return frames, objects, regions


def _validation_tables(document: dict, clips: dict) -> tuple[dict, list, list, list]:
    wanted = {
        f"images/{clip}/{_frame_name(frame_id)}.jpg": (clip, frame_id)
        for clip, frame_ids in clips.items()
        for frame_id in frame_ids
    }
    images = [image for image in document["images"] if image["file_name"] in wanted]
    if len(images) != len(wanted):
        raise ValueError("Some selected validation frames are absent from the COCO file.")
    keys = {}
    frames = []
    for image in images:
        clip, frame_id = wanted[image["file_name"]]
        frame = _frame_name(frame_id)
        keys[image["id"]] = f"val/{clip}/{frame}"
        frames.append(
            {
                "split": "val",
                "image_id": str(image["id"]),
                "image_key": keys[image["id"]],
                "clip": clip,
                "frame": frame,
                "frame_id": frame_id,
                "relative_path": f"val/images/{clip}/{frame}.jpg",
                "image_width": image["width"],
                "image_height": image["height"],
                "is_labeled": True,
                "submission_id": None,
            }
        )
    annotations = [item for item in document["annotations"] if item["image_id"] in keys]
    objects = [
        {
            "split": "val",
            "image_key": keys[item["image_id"]],
            "class_id": item["category_id"] - 1,
            "category_id": item["category_id"],
            "x": float(item["bbox"][0]),
            "y": float(item["bbox"][1]),
            "width": float(item["bbox"][2]),
            "height": float(item["bbox"][3]),
            "area": float(item["area"]),
        }
        for item in annotations
        if not item.get("iscrowd")
    ]
    unique_regions = {
        (item["image_id"], *map(float, item["bbox"])) for item in annotations if item.get("iscrowd")
    }
    regions = [
        {"split": "val", "image_key": keys[image_id], "x": x, "y": y, "width": w, "height": h}
        for image_id, x, y, w, h in sorted(unique_regions)
    ]
    subset = {key: value for key, value in document.items() if key not in {"images", "annotations"}}
    subset.update({"images": images, "annotations": annotations})
    return subset, frames, objects, regions


def _test_frames(
    read: Callable[[str], bytes | None], destination: Path, clips: dict
) -> tuple[list, list[dict]]:
    content = read("test_images.csv")
    if content is None:
        raise FileNotFoundError("Missing test_images.csv in the dataset source.")
    wanted = {(clip, frame_id) for clip, frame_ids in clips.items() for frame_id in frame_ids}
    rows = [
        row
        for row in csv.DictReader(io.StringIO(content.decode()))
        if (row["clip"], int(row["frame"])) in wanted
    ]
    if len(rows) != len(wanted):
        raise ValueError("Some selected test frames are absent from test_images.csv.")
    frames = []
    for index, row in enumerate(rows):
        image = read(row["file_name"])
        if image is None:
            raise FileNotFoundError(f"Missing {row['file_name']} in the dataset source.")
        _write(destination / row["file_name"], image)
        frame = Path(row["file_name"]).stem
        frames.append(
            {
                "split": "test",
                "image_id": str(index),
                "image_key": f"test/{row['clip']}/{frame}",
                "clip": row["clip"],
                "frame": frame,
                "frame_id": int(row["frame"]),
                "relative_path": row["file_name"],
                "image_width": int(row["width"]),
                "image_height": int(row["height"]),
                "is_labeled": False,
                "submission_id": row["image_id"],
            }
        )
    return frames, rows


def prepare_smoke_subset(
    source: Path, destination: Path, selection: dict[str, dict[str, list[int]]]
) -> Path:
    """Copy a few selected frames and build small annotation tables without extraction.

    Read only the selected images, YOLO labels and ignore files from the competition ZIP
    or an extracted competition directory. Validation targets and crowds come from the
    original COCO file, filtered to the selected frames; training targets come from the
    YOLO and ignore files. Reuse a finished subset only when its selection is identical.

    Args:
        source: Competition ZIP or dataset root containing train, val and test_images.csv.
        destination: Empty or previously prepared directory outside the repository.
        selection: Frame numbers keyed by split (train, val, test) and then by clip name.

    Returns:
        The prepared dataset root, with tables in its metadata directory.
    """
    check_flight_separation(selection)
    manifest = destination / "selection.json"
    if manifest.is_file():
        if json.loads(manifest.read_text()) != selection:
            raise ValueError(f"{destination} holds a different selection; use another directory.")
        LOGGER.info("Reuse prepared subset %s", destination)
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty {destination}.")
    with _source_reader(source) as read:
        for split in ("train", "val"):
            _copy_labeled_files(read, destination, split, selection[split])
        test_frames, test_rows = _test_frames(read, destination, selection["test"])
        document = read("val/annotations_coco.json")
        if document is None:
            raise FileNotFoundError("Missing val/annotations_coco.json in the dataset source.")
    coco_subset, val_frames, val_objects, val_regions = _validation_tables(
        json.loads(document), selection["val"]
    )
    del document
    (destination / "val/annotations_coco.json").write_text(json.dumps(coco_subset))
    with (destination / "test_images.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    train_frames, train_objects, train_regions = _train_tables(destination, selection["train"])
    metadata = destination / "metadata"
    metadata.mkdir(exist_ok=True)
    pl.DataFrame(train_frames + val_frames + test_frames, schema=FRAME_SCHEMA).write_parquet(
        metadata / "frames.parquet"
    )
    pl.DataFrame(train_objects + val_objects, schema=OBJECT_SCHEMA).write_parquet(
        metadata / "objects.parquet"
    )
    pl.DataFrame(train_regions + val_regions, schema=REGION_SCHEMA).write_parquet(
        metadata / "ignored_regions.parquet"
    )
    manifest.write_text(json.dumps(selection, indent=2) + "\n")
    LOGGER.info("Prepared subset %s", destination)
    return destination


def load_smoke_tables(root: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Read the frame, target-object and ignored-region tables of a prepared subset.

    Args:
        root: Directory returned by prepare_smoke_subset.

    Returns:
        Frames, objects and ignored regions in original-image pixels.
    """
    metadata = root / "metadata"
    return (
        pl.read_parquet(metadata / "frames.parquet"),
        pl.read_parquet(metadata / "objects.parquet"),
        pl.read_parquet(metadata / "ignored_regions.parquet"),
    )
