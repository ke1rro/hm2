import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from PIL import Image

from hm2.detection_data import DetectionDataset, detection_collate, make_loader, select_frames
from hm2.evaluation import evaluate_coco, predictions_to_coco


def test_dataset_preserves_labels_coordinates_identity_and_empty_targets(tmp_path: Path) -> None:
    """Catch frame-name collisions, class offsets, resizing, or dropped overlapping GT."""
    paths = ["train/images/a/0001.jpg", "train/images/b/0001.jpg", "val/images/c/0001.jpg"]
    for relative_path in paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True)
        Image.new("RGB", (40, 20), "white").save(path)
    frames = pl.DataFrame(
        {
            "image_key": ["train/a/0001", "train/b/0001", "val/c/0001"],
            "image_id": ["1", "2", "1"],
            "relative_path": paths,
            "image_width": [40, 40, 40],
            "image_height": [20, 20, 20],
            "split": ["train", "train", "val"],
            "clip": ["a", "b", "c"],
            "frame_id": [1, 1, 1],
        }
    )
    objects = pl.DataFrame(
        {
            "image_key": ["train/a/0001", "train/a/0001", "val/c/0001"],
            "category_id": [1, 8, 8],
            "x": [2.0, 5.0, 10.0],
            "y": [3.0, 4.0, 2.0],
            "width": [8.0, 10.0, 4.0],
            "height": [7.0, 8.0, 5.0],
        }
    )
    dataset = DetectionDataset(tmp_path, frames, objects)
    image, target = dataset[0]
    assert image.shape == (3, 20, 40)
    assert image.dtype == torch.float32 and image.min() == image.max() == 1
    torch.testing.assert_close(
        target["boxes"], torch.tensor([[2.0, 3.0, 10.0, 10.0], [5.0, 4.0, 15.0, 12.0]])
    )
    assert target["labels"].tolist() == [1, 8]
    assert dataset[1][1]["boxes"].shape == (0, 4)
    assert dataset[1][1]["labels"].shape == (0,)
    assert dataset.frames["image_key"].n_unique() == 3
    flipped = DetectionDataset(tmp_path, frames, objects, flip_probability=1.0)[0][1]
    torch.testing.assert_close(
        flipped["boxes"], torch.tensor([[30.0, 3.0, 38.0, 10.0], [25.0, 4.0, 35.0, 12.0]])
    )
    loader = make_loader(dataset, batch_size=2, num_workers=0, device=torch.device("cpu"))
    images, targets = next(iter(loader))
    assert len(images) == len(targets) == 2
    assert detection_collate([dataset[1]])[1][0]["labels"].dtype == torch.int64


def test_selection_is_per_clip_and_excludes_test_frames() -> None:
    """Catch global stride selection that omits short clips or mixes evaluation splits."""
    frames = pl.DataFrame(
        {
            "split": ["train"] * 5 + ["val"],
            "clip": ["a", "a", "a", "b", "b", "c"],
            "frame_id": [1, 2, 3, 10, 20, 1],
        }
    )
    selected = select_frames(frames, "train", stride=2)
    assert selected.select("clip", "frame_id").rows() == [("a", 1), ("a", 3), ("b", 10)]


def test_coco_conversion_and_crowd_evaluation_use_original_pixels(tmp_path: Path) -> None:
    """Catch predicted background labels, wrong COCO offsets, or lost source crowds."""
    prediction = {
        "boxes": torch.tensor(
            [[2.0, 3.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0], [1.0, 1.0, 2.0, 2.0]]
        ),
        "labels": torch.tensor([8, 1, 0]),
        "scores": torch.tensor([0.99, 0.9, 0.8]),
    }
    records = predictions_to_coco([prediction], [17])
    assert records[0]["category_id"] == 8 and records[0]["bbox"] == [2.0, 3.0, 8.0, 7.0]
    assert len(records) == 2 and {row["image_id"] for row in records} == {17}
    document = {
        "images": [{"id": 17, "width": 40, "height": 20, "file_name": "c/0001.jpg"}],
        "categories": [{"id": i, "name": str(i)} for i in range(1, 9)],
        "annotations": [
            {
                "id": 1,
                "image_id": 17,
                "category_id": 8,
                "bbox": [2, 3, 8, 7],
                "area": 56,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 17,
                "category_id": 1,
                "bbox": [0, 0, 40, 20],
                "area": 800,
                "iscrowd": 1,
            },
            {
                "id": 3,
                "image_id": 17,
                "category_id": 8,
                "bbox": [0, 0, 40, 20],
                "area": 800,
                "iscrowd": 1,
            },
        ],
    }
    path = tmp_path / "annotations_coco.json"
    path.write_text(json.dumps(document))
    metrics = evaluate_coco(path, records, [17])
    assert metrics["map"] == pytest.approx(1.0)
    assert metrics["ap_small"] == pytest.approx(1.0)
    assert metrics["max_dets"] == [1, 10, 100]
    assert metrics["per_class_ap"]["motor"] == pytest.approx(1.0)
    assert metrics["per_class_ap"]["person"] is None
    empty = evaluate_coco(path, [], [17])
    assert empty["map"] == pytest.approx(0.0)
    assert np.isfinite(empty["map"])
