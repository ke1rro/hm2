from pathlib import Path

import pytest
import torch
from ultralytics.utils.loss import v8DetectionLoss

from experiments.yolo_baseline import YoloDetector, classification_keep_mask


def test_ignore_locations_preserve_all_positive_assignments() -> None:
    """Remove only negative points in ignored regions, including empty-GT images."""
    points = torch.tensor([[5.0, 5.0], [15.0, 15.0], [25.0, 25.0]])
    foreground = torch.tensor([[True, False, False], [False, False, False]])
    ignored = [torch.tensor([[0.0, 0.0, 20.0, 20.0]]), torch.empty(0, 4)]
    keep = classification_keep_mask(foreground, points, ignored)
    assert keep.tolist() == [[True, False, True], [True, True, True]]
    assert (keep | ~foreground).all()
    assert torch.equal(
        classification_keep_mask(foreground, points, ignored, False), torch.ones_like(foreground)
    )


@pytest.fixture(scope="module")
def detector() -> YoloDetector:
    """Build an eight-class YOLO11s without pretrained weights or downloads."""
    torch.set_num_threads(2)
    return YoloDetector(pretrained=False, min_size=64, max_size=96)


def test_yolo_native_loss_equivalence_and_cpu_gradient(detector: YoloDetector) -> None:
    """Match native CIoU/BCE/DFL without exclusions and backpropagate genuine model losses."""
    from experiments.yolo_baseline import IgnoreAwareYoloLoss

    detector.train()
    images = torch.rand(2, 3, 64, 96)
    predictions = detector.network(images)
    native_batch = {
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [7.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.4, 0.4, 0.3, 0.3]]),
    }
    batch = {**native_batch, "ignore_boxes": [torch.empty(0, 4), torch.empty(0, 4)]}
    expected = v8DetectionLoss(detector.network)(predictions, native_batch)[0]
    actual = IgnoreAwareYoloLoss(detector.network)(predictions, batch)[0]
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (3,) and torch.isfinite(actual).all()
    masked_batch = {**batch, "ignore_boxes": [torch.tensor([[0.0, 0.0, 96.0, 64.0]])] * 2}
    masked = IgnoreAwareYoloLoss(detector.network)(predictions, masked_batch)[0]
    torch.testing.assert_close(masked[[0, 2]], expected[[0, 2]])
    assert masked[1] < expected[1]
    actual.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in detector.network.parameters())
    assert detector.network.yaml["scale"] == "s"
    assert detector.network.model[-1].nc == 8


def test_yolo_original_coordinates_labels_and_checkpoint_roundtrip(
    detector: YoloDetector, tmp_path: Path
) -> None:
    """Return source pixels and restore the shared optimizer, scaler, RNG and batch cursor."""
    from hm2.experiment_checkpoint import load_checkpoint, save_checkpoint
    from hm2.retinanet import make_optimizer

    detector.eval()
    images = [torch.rand(3, 40, 80), torch.rand(3, 80, 40)]
    with torch.no_grad():
        outputs = detector(images)
    assert len(outputs) == 2
    for output, image in zip(outputs, images, strict=True):
        assert set(output) == {"boxes", "scores", "labels"}
        assert ((output["labels"] >= 1) & (output["labels"] <= 8)).all()
        assert (output["boxes"][:, [0, 2]] <= image.shape[-1]).all()
        assert (output["boxes"][:, [1, 3]] <= image.shape[-2]).all()
    known = detector.decode_predictions(
        [torch.tensor([[0.0, 0.0, 96.0, 48.0, 0.8, 0.0]])], [(48, 96)], [(40, 80)]
    )[0]
    torch.testing.assert_close(known["boxes"], torch.tensor([[0.0, 0.0, 80.0, 40.0]]))
    assert known["labels"].tolist() == [1]
    optimizer, scheduler = make_optimizer(detector, 0.0025, 10, 1)
    path = tmp_path / "latest.pt"
    config = {"model": "yolo11s", "seed": 2026}
    save_checkpoint(
        path, detector, optimizer, scheduler, 2, config, extra_state={"batch_cursor": 2}
    )
    state = load_checkpoint(path, detector, optimizer, scheduler, expected_config=config)
    assert state["extra_state"]["batch_cursor"] == 2 and state["step"] == 2


def test_yolo_shared_loop_resumes_after_consumed_batch(tmp_path: Path) -> None:
    """Resume real CPU YOLO optimization with its original schedule and full validation state."""
    import json

    import polars as pl
    from PIL import Image

    from experiments.protocol import load_settings
    from experiments.yolo_baseline import build_yolo
    from hm2.data import CLASS_NAMES
    from hm2.faster_rcnn_experiments import run_experiment

    frames, objects = [], []
    for index in range(5):
        split = "train" if index < 4 else "val"
        path = f"{index}.png"
        Image.new("RGB", (96, 64), (80, 110, 150)).save(tmp_path / path)
        key = f"{split}/{index}"
        frames.append(
            {
                "split": split,
                "image_key": key,
                "image_id": str(index),
                "clip": split,
                "frame_id": index,
                "relative_path": path,
                "image_width": 96,
                "image_height": 64,
            }
        )
        objects.append(
            {
                "image_key": key,
                "category_id": 1,
                "x": 20.0,
                "y": 20.0,
                "width": 20.0,
                "height": 20.0,
            }
        )
    (tmp_path / "val").mkdir()
    (tmp_path / "val/annotations_coco.json").write_text(
        json.dumps(
            {
                "images": [{"id": 4, "width": 96, "height": 64, "file_name": "4.png"}],
                "categories": [
                    {"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 4,
                        "category_id": 1,
                        "bbox": [20, 20, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    }
                ],
            }
        )
    )
    frames, objects = pl.from_dicts(frames), pl.from_dicts(objects)
    ignored = objects.head(0).drop("category_id")
    settings = load_settings(Path(__file__).parents[1], "yolo_baseline")
    settings["dataset"]["frame_stride"] = 1
    settings["training"].update(epochs=1, num_workers=0, checkpoint_seconds=0, warmup_updates=1)
    settings["model"].update(pretrained=False, min_size=64, max_size=96)
    settings["evaluation"]["num_workers"] = 0
    output = tmp_path / "run"

    def pause(directory: Path) -> None:
        state = json.loads((directory / "status.json").read_text())
        if state["batch_cursor"] == 1:
            (directory / "STOP").touch()

    args = (tmp_path, tmp_path, frames, objects, ignored, output, settings, "yolo_baseline")
    first = run_experiment(*args, model_factory=build_yolo, on_checkpoint=pause)
    assert first["global_step"] == first["batch_cursor"] == 1
    (output / "STOP").unlink()
    final = run_experiment(*args, model_factory=build_yolo)
    assert final["is_completed"] and final["global_step"] == 2
    assert final["image_exposures"] == 4
    assert (output / "best.pt").exists()
    config = json.loads((output / "config.json").read_text())
    assert config["model"]["backend"] == "ultralytics"
    assert config["loss"]["box_regression"] == "ciou"
    state = torch.load(output / "latest.pt", weights_only=True)
    assert state["optimizer"]["state"]
    assert state["extra_state"]["epochs_completed"] == 1
