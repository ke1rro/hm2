import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import polars as pl
import pytest
from PIL import Image

from hm2.data import (
    CLASS_NAMES,
    audit_dataset,
    compare_yolo_coco,
    extract_archive,
    find_dataset_root,
    load_tables,
    xywh_to_yolo,
    yolo_to_xywh,
)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Create a small dataset with repeated frame names and a crowd annotation."""
    for split, clips in (("train", ["a", "b"]), ("val", ["c"])):
        image_records = []
        for image_id, clip in enumerate(clips, 1):
            image_path = tmp_path / split / "images" / clip / "0000001.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (200, 100), "gray").save(image_path)
            label_path = tmp_path / split / "labels" / clip / "0000001.txt"
            label_path.parent.mkdir(parents=True)
            label_path.write_text("0 0.5 0.25 0.2 0.1\n" if image_id == 1 else "")
            image_records.append(
                {
                    "id": image_id,
                    "file_name": f"images/{clip}/0000001.jpg",
                    "width": 200,
                    "height": 100,
                    "video": clip,
                    "frame_id": 1,
                }
            )
        document = {
            "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)],
            "images": image_records,
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [80, 20, 40, 10],
                    "iscrowd": 0,
                    "occlusion": 1,
                },
            ],
        }
        if split == "train":
            document["annotations"].extend(
                {
                    "id": category_id + 1,
                    "image_id": 1,
                    "category_id": category_id,
                    "bbox": [0, 0, 20, 20],
                    "iscrowd": 1,
                }
                for category_id in range(1, 9)
            )
            ignore_path = tmp_path / split / "ignore" / "a" / "0000001.txt"
            ignore_path.parent.mkdir(parents=True)
            ignore_path.write_text("0.05 0.1 0.1 0.2\n")
        (tmp_path / split / "annotations_coco.json").write_text(json.dumps(document))
        (tmp_path / split / "data.yaml").write_text("names: []\n")
    test_path = tmp_path / "test" / "images" / "t" / "0000001.jpg"
    test_path.parent.mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(test_path)
    (tmp_path / "test_images.csv").write_text(
        "image_id,file_name,clip,frame,width,height\n"
        "t_0000001,test/images/t/0000001.jpg,t,1,200,100\n"
    )
    (tmp_path / "sample_submission.csv").write_text("image_id,PredictionString\nt_0000001,\n")
    return tmp_path


def test_box_conversions_use_original_dimensions() -> None:
    """Catch swapped dimensions or center versus top-left coordinate mistakes."""
    normalized = np.array([[0.5, 0.25, 0.2, 0.1]])
    pixels = yolo_to_xywh(normalized, 200, 100)
    np.testing.assert_allclose(pixels, [[80, 20, 40, 10]])
    np.testing.assert_allclose(xywh_to_yolo(pixels, 200, 100), normalized)


def test_tables_separate_crowds_and_unlabeled_frames(dataset_root: Path) -> None:
    """Catch class offsets and counting crowds or unlabeled frames as targets."""
    frames, objects, ignored = load_tables(dataset_root)
    assert objects["class_id"].to_list() == [0, 0]
    assert objects["class_name"].to_list() == ["person", "person"]
    assert ignored.height == 1
    assert ignored["coco_replication_count"].to_list() == [8]
    assert sorted(ignored["coco_category_ids"][0].to_list()) == list(range(1, 9))
    assert frames.filter(pl.col("split") == "test")["target_count"].to_list() == [None]
    assert frames.filter(pl.col("clip") == "b")["target_count"].to_list() == [0]
    assert frames["image_key"].n_unique() == 4


def test_yolo_matching_uses_full_image_identity(dataset_root: Path) -> None:
    """Catch frame-name collisions and treating absent ignore files as errors."""
    frames, objects, ignored = load_tables(dataset_root)
    comparison = compare_yolo_coco(dataset_root, frames, objects, ignored, sample_size=10)
    assert comparison.height == 3
    assert comparison["is_match"].all()
    np.testing.assert_array_less(comparison["max_error_px"].to_numpy(), 0.01)


def test_archive_wrapper_and_existing_files(tmp_path: Path) -> None:
    """Catch assuming a flat archive or overwriting existing extracted files."""
    archive_path = tmp_path / "dataset.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("wrapper/test_images.csv", "image_id\nexample\n")
        archive.writestr("wrapper/sample_submission.csv", "image_id,PredictionString\n")
    destination = tmp_path / "extracted"
    extract_archive(archive_path, destination)
    assert find_dataset_root(destination) == destination / "wrapper"
    existing_path = destination / "wrapper/test_images.csv"
    existing_path.write_text("user data")
    with pytest.raises(FileExistsError):
        extract_archive(archive_path, destination)
    assert existing_path.read_text() == "user data"


def test_audit_reports_bad_boxes_without_changing_them(dataset_root: Path) -> None:
    """Catch silently clipping invalid coordinates or omitting missing images."""
    frames, objects, ignored = load_tables(dataset_root)
    objects = objects.with_columns(pl.lit(-5.0).alias("x"))
    (dataset_root / "train/images/a/0000001.jpg").unlink()
    (dataset_root / "sample_submission.csv").unlink()
    findings = audit_dataset(dataset_root, frames, objects, ignored)
    assert findings.filter(pl.col("check") == "Target boxes outside image")["count"][0] == 2
    assert findings.filter(pl.col("check") == "Missing image files")["count"][0] == 1
    assert findings.filter(pl.col("check") == "Missing sample submission")["count"][0] == 1
    assert objects["x"].to_list() == [-5.0, -5.0]
