import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RegionProposalNetwork

from hm2.data import read_yolo_boxes
from hm2.dense_evaluation import evaluate_coco_max_dets, evaluate_detections
from hm2.evaluation import evaluate_coco
from hm2.faster_rcnn import (
    IgnoreAwareDetectionDataset,
    IgnoreAwareRegionProposalNetwork,
    IgnoreAwareRoIHeads,
    IgnoreAwareTransform,
    build_faster_rcnn,
)
from hm2.submission import build_submission


def _tables(key: str, objects: list[dict], regions: list[dict]) -> tuple:
    frames = pl.DataFrame(
        {
            "image_key": [key],
            "image_id": ["7"],
            "relative_path": ["image.jpg"],
            "image_width": [40],
            "image_height": [20],
        }
    )
    objects_table = pl.DataFrame(
        [{"image_key": key, **row} for row in objects],
        schema={
            "image_key": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
            "width": pl.Float64,
            "height": pl.Float64,
            "category_id": pl.Int64,
        },
    )
    regions_table = pl.DataFrame(
        [{"image_key": key, **row} for row in regions],
        schema={
            "image_key": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
            "width": pl.Float64,
            "height": pl.Float64,
        },
    )
    return frames, objects_table, regions_table


def test_yolo_label_round_trips_to_torchvision_target_and_submission(tmp_path: Path) -> None:
    label = tmp_path / "0000001.txt"
    label.write_text("0 0.328497 0.776455 0.026042 0.116402\n")
    class_id, x, y, width, height = read_yolo_boxes(label, 1344, 756)[0].tolist()
    assert np.allclose([x, y, width, height], [424, 543, 35, 88], atol=0.01)

    Image.new("RGB", (1344, 756)).save(tmp_path / "image.jpg")
    frames, objects, regions = _tables(
        "train/clip/0000001",
        [{"x": x, "y": y, "width": width, "height": height, "category_id": int(class_id) + 1}],
        [],
    )
    frames = frames.with_columns(
        pl.lit(1344).alias("image_width"), pl.lit(756).alias("image_height")
    )
    _, target = IgnoreAwareDetectionDataset(tmp_path, frames, objects, regions)[0]
    assert target["labels"].tolist() == [1]
    assert np.allclose(target["boxes"][0].tolist(), [424, 543, 459, 631], atol=0.01)

    prediction = {
        "boxes": target["boxes"],
        "labels": target["labels"],
        "scores": torch.tensor([0.9]),
    }
    sample = pl.DataFrame({"image_id": ["clip_001_0000001"], "PredictionString": [""]})
    test_frames = frames.with_columns(pl.lit("clip_001_0000001").alias("submission_id"))
    submission = build_submission([prediction], test_frames, sample)
    tokens = submission["PredictionString"][0].split()
    assert tokens[0] == "0"
    assert np.allclose([float(value) for value in tokens[2:]], [424, 543, 35, 88], atol=0.01)


def test_flip_moves_targets_and_ignored_regions_together(tmp_path: Path) -> None:
    Image.new("RGB", (40, 20)).save(tmp_path / "image.jpg")
    frames, objects, regions = _tables(
        "train/clip/0000001",
        [{"x": 2.0, "y": 3.0, "width": 8.0, "height": 4.0, "category_id": 3}],
        [{"x": 30.0, "y": 0.0, "width": 10.0, "height": 5.0}],
    )
    _, target = IgnoreAwareDetectionDataset(
        tmp_path, frames, objects, regions, flip_probability=1.0
    )[0]
    assert target["boxes"].tolist() == [[30.0, 3.0, 38.0, 7.0]]
    assert target["ignore_boxes"].tolist() == [[0.0, 0.0, 10.0, 5.0]]


def test_transform_rescales_ignored_regions_like_targets() -> None:
    transform = IgnoreAwareTransform(200, 400, [0.0] * 3, [1.0] * 3)
    box = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
    target = {"boxes": box.clone(), "labels": torch.tensor([1]), "ignore_boxes": box.clone()}
    images, targets = transform([torch.rand(3, 100, 200)], [target])
    assert images.image_sizes == [(200, 400)]
    assert torch.allclose(targets[0]["ignore_boxes"], targets[0]["boxes"])
    assert torch.allclose(targets[0]["boxes"], box * 2)


def test_rpn_excludes_only_negative_anchors_inside_ignored_regions() -> None:
    rpn = IgnoreAwareRegionProposalNetwork(
        None,
        torch.nn.Identity(),
        0.7,
        0.3,
        256,
        0.5,
        {"training": 1, "testing": 1},
        {"training": 1, "testing": 1},
        0.7,
    )
    anchors = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [100.0, 100.0, 110.0, 110.0],
            [195.0, 100.0, 205.0, 110.0],
            [196.0, 100.0, 206.0, 110.0],
            [300.0, 300.0, 310.0, 310.0],
        ]
    )
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([1]),
        "ignore_boxes": torch.tensor([[0.0, 0.0, 200.0, 200.0]]),
    }
    stock, _ = RegionProposalNetwork.assign_targets_to_anchors(rpn, [anchors], [target])
    aware, _ = rpn.assign_targets_to_anchors([anchors], [target])
    assert stock[0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert aware[0].tolist() == [1.0, -1.0, -1.0, 0.0, 0.0]


def test_roi_sampling_never_returns_proposals_inside_ignored_regions() -> None:
    heads = IgnoreAwareRoIHeads(
        torch.nn.Identity(),
        torch.nn.Identity(),
        torch.nn.Identity(),
        0.5,
        0.5,
        512,
        0.25,
        None,
        0.05,
        0.5,
        100,
    )
    proposals = [torch.tensor([[100.0, 100.0, 110.0, 110.0], [300.0, 300.0, 310.0, 310.0]])]
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([4]),
        "ignore_boxes": torch.tensor([[50.0, 50.0, 200.0, 200.0]]),
    }
    sampled, _, labels, _ = heads.select_training_samples([proposals[0].clone()], [target])
    assert heads._ignore_boxes is None
    kept = {tuple(box) for box in sampled[0].tolist()}
    assert (100.0, 100.0, 110.0, 110.0) not in kept
    assert (300.0, 300.0, 310.0, 310.0) in kept
    assert sorted(labels[0].tolist()) == [0, 4]


def test_random_model_trains_and_predicts_eight_class_labels() -> None:
    model = build_faster_rcnn(pretrained=False, min_size=128, max_size=160, detections_per_img=500)
    predictor = model.roi_heads.box_predictor
    assert isinstance(predictor, FastRCNNPredictor)
    assert predictor.cls_score.out_features == 9
    assert model.roi_heads.detections_per_img == 500
    image = torch.rand(3, 120, 160)
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 40.0, 50.0], [60.0, 20.0, 90.0, 60.0]]),
        "labels": torch.tensor([1, 8]),
        "ignore_boxes": torch.tensor([[100.0, 0.0, 160.0, 60.0]]),
    }
    model.train()
    losses = model([image], [target])
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    model.eval()
    with torch.inference_mode():
        prediction = model([image])[0]
    assert len(prediction["boxes"]) <= 500
    assert ((prediction["labels"] >= 1) & (prediction["labels"] <= 8)).all()


def test_submission_cap_recovers_dense_frames_lost_at_hundred_detections(tmp_path: Path) -> None:
    boxes = [
        [float(25 * (index % 15)), float(25 * (index // 15)), 20.0, 20.0] for index in range(150)
    ]
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 400, "height": 300, "file_name": "a.jpg"}],
                "categories": [{"id": index, "name": str(index)} for index in range(1, 9)],
                "annotations": [
                    {
                        "id": index + 1,
                        "image_id": 1,
                        "category_id": 3,
                        "bbox": box,
                        "area": 400.0,
                        "iscrowd": 0,
                    }
                    for index, box in enumerate(boxes)
                ],
            }
        )
    )
    predictions = [
        {"image_id": 1, "category_id": 3, "bbox": box, "score": 1.0 - index / 1000}
        for index, box in enumerate(boxes)
    ]
    noisy = [
        {"image_id": 1, "category_id": 3, "bbox": [x + index % 7, y, w, h], "score": 0.5}
        for index, (x, y, w, h) in enumerate(boxes)
    ]
    metrics = evaluate_detections(annotation_path, predictions, [1])
    assert metrics["max_dets_500"]["map"] == 1.0
    assert metrics["max_dets_500"]["ap_small"] == 1.0
    assert metrics["max_dets_100"]["map"] < 0.7
    assert metrics["max_dets_100"]["max_dets"] == [1, 10, 100]
    canonical = evaluate_coco(annotation_path, noisy, [1])
    recomputed = evaluate_coco_max_dets(annotation_path, noisy, [1], 100)
    for name in ("map", "ap50", "ap75", "ap_small"):
        assert recomputed[name] == pytest.approx(canonical[name])
