import json
from pathlib import Path

import pytest
from PIL import Image

from hm2.smoke_subset import check_flight_separation, load_smoke_tables, prepare_smoke_subset


def _write_source(root: Path) -> None:
    for split, clip in (("train", "uav0000001_00000_v"), ("val", "uav0000002_00000_v")):
        (root / split / "images" / clip).mkdir(parents=True)
        (root / split / "labels" / clip).mkdir(parents=True)
        Image.new("RGB", (100, 50)).save(root / split / "images" / clip / "0000001.jpg")
        (root / split / "labels" / clip / "0000001.txt").write_text("2 0.5 0.5 0.2 0.4\n")
    (root / "train/ignore/uav0000001_00000_v").mkdir(parents=True)
    (root / "train/ignore/uav0000001_00000_v/0000001.txt").write_text("0.1 0.1 0.2 0.2\n")
    crowd = {"image_id": 5, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 1}
    document = {
        "info": {},
        "categories": [{"id": index, "name": str(index)} for index in range(1, 9)],
        "images": [
            {
                "id": 5,
                "file_name": "images/uav0000002_00000_v/0000001.jpg",
                "width": 100,
                "height": 50,
            },
            {
                "id": 6,
                "file_name": "images/uav0000002_00000_v/0000002.jpg",
                "width": 100,
                "height": 50,
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 5,
                "category_id": 3,
                "bbox": [40, 15, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            *[{"id": 2 + index, "category_id": index + 1, **crowd} for index in range(8)],
            {
                "id": 20,
                "image_id": 6,
                "category_id": 1,
                "bbox": [1, 1, 2, 2],
                "area": 4,
                "iscrowd": 0,
            },
        ],
    }
    (root / "val/annotations_coco.json").write_text(json.dumps(document))
    (root / "test/images/clip_001").mkdir(parents=True)
    Image.new("RGB", (80, 40)).save(root / "test/images/clip_001/0000001.jpg")
    (root / "test_images.csv").write_text(
        "image_id,file_name,clip,frame,width,height\n"
        "clip_001_0000001,test/images/clip_001/0000001.jpg,clip_001,1,80,40\n"
        "clip_001_0000002,test/images/clip_001/0000002.jpg,clip_001,2,80,40\n"
    )


def test_selection_rejects_a_flight_shared_by_train_and_validation() -> None:
    with pytest.raises(ValueError, match="uav0000013"):
        check_flight_separation(
            {"train": {"uav0000013_00000_v": [1]}, "val": {"uav0000013_01073_v": [1]}, "test": {}}
        )


def test_subset_copies_only_selected_frames_and_keeps_original_crowds(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)
    selection = {
        "train": {"uav0000001_00000_v": [1]},
        "val": {"uav0000002_00000_v": [1]},
        "test": {"clip_001": [1]},
    }
    root = prepare_smoke_subset(source, tmp_path / "subset", selection)
    frames, objects, ignored = load_smoke_tables(root)
    assert frames["split"].to_list() == ["train", "val", "test"]
    assert frames.filter(frames["split"] == "test")["submission_id"].to_list() == [
        "clip_001_0000001"
    ]
    train_object = objects.filter(objects["split"] == "train").row(0, named=True)
    assert (train_object["category_id"], train_object["x"], train_object["width"]) == (
        3,
        40.0,
        20.0,
    )
    assert ignored.height == 2
    coco = json.loads((root / "val/annotations_coco.json").read_text())
    assert [image["id"] for image in coco["images"]] == [5]
    assert sum(item["iscrowd"] for item in coco["annotations"]) == 8
    assert not (root / "val/images/uav0000002_00000_v/0000002.jpg").exists()
    assert prepare_smoke_subset(source, tmp_path / "subset", selection) == root
    with pytest.raises(ValueError):
        prepare_smoke_subset(source, tmp_path / "subset", {**selection, "test": {"clip_001": [2]}})
