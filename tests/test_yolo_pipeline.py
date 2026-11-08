import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import polars as pl
import pytest
import torch
from ultralytics.nn.tasks import DetectionModel

from hm2.data import CLASS_NAMES, load_tables
from hm2.dense_evaluation import evaluate_coco_max_dets
from hm2.yolo_data import (
    mask_ignored_regions,
    parse_yolo_label,
    prepare_yolo_dataset,
    resize_long_side,
    verify_yolo_dataset,
)
from hm2.yolo_pipeline import (
    PASSTHROUGH_TRAINING_KEYS,
    CheckpointMirror,
    build_yolo_submission,
    classify_ground_truth_outcomes,
    coco_detections,
    load_pipeline_settings,
    plan_epochs,
    predict_frames,
    remap_detection_head,
    training_arguments,
    write_coco_subset,
)

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 400, 224
TARGET = (100.0, 60.0, 60.0, 40.0)
IGNORED = (90.0, 50.0, 140.0, 100.0)


def _settings() -> dict:
    return load_pipeline_settings(ROOT / "config/yolo11s_native_1280.toml")


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Create two training clips, one validation clip and a small test listing."""
    root = tmp_path / "competition"
    annotation_id = 1
    for split, clips in (("train", ["clip_a", "clip_b"]), ("val", ["clip_c"])):
        images, annotations = [], []
        for clip in clips:
            for frame_id in range(1, 4):
                image_id = len(images) + 1
                relative = Path(clip) / f"{frame_id:07d}"
                image = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)
                image_path = root / split / "images" / relative.with_suffix(".jpg")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path), image)
                x, y, width, height = TARGET
                label_path = root / split / "labels" / relative.with_suffix(".txt")
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(
                    f"2 {(x + width / 2) / WIDTH:.6f} {(y + height / 2) / HEIGHT:.6f} "
                    f"{width / WIDTH:.6f} {height / HEIGHT:.6f}\n"
                )
                images.append(
                    {
                        "id": image_id,
                        "file_name": f"images/{relative}.jpg",
                        "width": WIDTH,
                        "height": HEIGHT,
                        "video": clip,
                        "frame_id": frame_id,
                    }
                )
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 3,
                        "bbox": list(TARGET),
                        "area": TARGET[2] * TARGET[3],
                        "iscrowd": 0,
                        "track_id": 1,
                    }
                )
                annotations.append(
                    {
                        "id": annotation_id + 1,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": list(IGNORED),
                        "area": IGNORED[2] * IGNORED[3],
                        "iscrowd": 1,
                    }
                )
                annotation_id += 2
        document = {
            "categories": [
                {"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)
            ],
            "images": images,
            "annotations": annotations,
        }
        (root / split / "annotations_coco.json").write_text(json.dumps(document))
    (root / "test" / "images").mkdir(parents=True)
    pl.DataFrame(
        {
            "image_id": ["clip_t_0000001", "clip_t_0000002"],
            "file_name": ["test/images/clip_t/0000001.jpg", "test/images/clip_t/0000002.jpg"],
            "clip": ["clip_t", "clip_t"],
            "frame": [1, 2],
            "width": [WIDTH, 200],
            "height": [HEIGHT, 112],
        }
    ).write_csv(root / "test_images.csv")
    return root


def test_prepared_dataset_masks_ignored_pixels_and_keeps_labels(
    dataset_root: Path, tmp_path: Path
) -> None:
    frames, objects, ignored = load_tables(dataset_root)
    settings = {
        **_settings()["dataset"],
        "frame_stride": 2,
        "image_size": 200,
        "validation_clips": ["clip_b"],
        "expected_train_frames": 4,
        "expected_train_clips": 2,
        "expected_validation_frames": 3,
        "expected_validation_clips": 1,
    }
    output = tmp_path / "prepared"
    manifest = prepare_yolo_dataset(dataset_root, output, frames, objects, ignored, settings, 2)
    checks = verify_yolo_dataset(output, frames, objects, settings, manifest)
    assert checks["is_pass"].all()
    assert manifest["splits"]["train"] == {
        "frames": 4,
        "clips": 2,
        "labels": 4,
        "masked_regions": 4,
        "frames_with_masks": 4,
    }
    moved_label = output / "labels/train/clip_c/0000003.txt"
    assert moved_label.read_text() == (dataset_root / "val/labels/clip_c/0000003.txt").read_text()
    assert sorted(path.name for path in (output / "images/val/clip_b").iterdir()) == [
        "0000001.jpg",
        "0000002.jpg",
        "0000003.jpg",
    ]
    source_label = dataset_root / "train/labels/clip_a/0000001.txt"
    written_label = output / "labels/train/clip_a/0000001.txt"
    assert written_label.read_text() == source_label.read_text()
    image = cv2.imread(str(output / "images/train/clip_a/0000001.jpg"))
    assert image is not None
    assert image.shape[:2] == (112, 200)
    assert abs(int(image[65, 100, 0]) - 114) <= 3
    assert abs(int(image[40, 65, 0]) - 200) <= 3
    assert abs(int(image[5, 5, 0]) - 200) <= 3


def test_masking_restores_annotated_pixels_inside_ignored_regions() -> None:
    image = np.arange(20 * 30 * 3, dtype=np.uint32).reshape(20, 30, 3).astype(np.uint8)
    masked = mask_ignored_regions(image, ((2.5, 2.0, 20.0, 10.0),), ((5.0, 4.0, 4.0, 3.0),), 114)
    assert (masked[2:12, 12:22] == 114).all()
    np.testing.assert_array_equal(masked[4:7, 5:9], image[4:7, 5:9])
    np.testing.assert_array_equal(masked[15:, :], image[15:, :])
    assert mask_ignored_regions(image, (), (), 114) is image


def test_label_parser_rejects_classes_and_boxes_outside_contract() -> None:
    assert parse_yolo_label("").shape == (0, 5)
    assert parse_yolo_label("7 0.5 0.5 0.2 0.2\n").shape == (1, 5)
    for text in ("8 0.5 0.5 0.2 0.2", "0 0.95 0.5 0.2 0.2", "0 0.5 0.5 0 0.1"):
        with pytest.raises(ValueError):
            parse_yolo_label(text)


class _FakeDetector:
    """Return one box at fixed fractions of each resized input image."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[int, int]]] = []

    def predict(self, source: list[np.ndarray], **_: object) -> list[SimpleNamespace]:
        self.calls.append([image.shape[:2] for image in source])
        outputs = []
        for image in source:
            height, width = image.shape[:2]
            boxes = torch.tensor(
                [
                    [0.25 * width, 0.5 * height, 0.5 * width, 0.75 * height],
                    [-5.0, -5, 2 * width, 10],
                ]
            )
            outputs.append(
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=boxes, conf=torch.tensor([0.9, 0.1]), cls=torch.tensor([2.0, 0.0])
                    )
                )
            )
        return outputs


def test_predictions_return_to_original_pixels_in_input_order(dataset_root: Path) -> None:
    test = pl.read_csv(dataset_root / "test_images.csv")
    for row in test.iter_rows(named=True):
        path = dataset_root / row["file_name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.zeros((row["height"], row["width"], 3), dtype=np.uint8))
    frames = test.rename(
        {"file_name": "relative_path", "width": "image_width", "height": "image_height"}
    )
    reversed_frames = pl.concat([frames, frames]).reverse()
    detector = _FakeDetector()
    inference = {**_settings()["inference"], "batch": 2}
    predictions = predict_frames(detector, reversed_frames, dataset_root, 160, inference, "cpu")
    assert all(len({shape for shape in call}) == 1 for call in detector.calls)
    for prediction, row in zip(predictions, reversed_frames.iter_rows(named=True), strict=True):
        width, height = row["image_width"], row["image_height"]
        resized_height = round(height * 160 / max(width, height))
        np.testing.assert_allclose(
            prediction["boxes"][0], [0.25 * width, 0.5 * height, 0.5 * width, 0.75 * height]
        )
        np.testing.assert_allclose(
            prediction["boxes"][1], [0, 0, width, 10 * height / resized_height], rtol=1e-6
        )
        assert prediction["classes"].tolist() == [2, 0]


def test_resize_uses_longer_side_without_changing_aspect() -> None:
    image = np.zeros((1071, 1904, 3), dtype=np.uint8)
    assert resize_long_side(image, 1280).shape == (720, 1280, 3)
    assert resize_long_side(np.zeros((382, 680, 3), dtype=np.uint8), 1280).shape == (719, 1280, 3)
    assert resize_long_side(image[:720, :1280], 1280).shape == (720, 1280, 3)


def test_submission_keeps_order_cap_and_original_coordinates() -> None:
    test = pl.DataFrame({"image_id": ["b", "a", "c"]})
    dense = {
        "boxes": np.tile(np.array([[10, 20, 40, 60]], dtype=np.float32), (600, 1)),
        "scores": np.linspace(0.001, 0.9, 600, dtype=np.float32),
        "classes": np.full(600, 7),
    }
    single = {
        "boxes": np.array(
            [[1.25, 2.5, 11.25, 22.5], [5, 5, 5, 9], [7, 7, 7.004, 9]], dtype=np.float32
        ),
        "scores": np.array([0.5, 0.9, 0.95], dtype=np.float32),
        "classes": np.array([0, 3, 4]),
    }
    empty = {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0), "classes": np.zeros(0)}
    submission = build_yolo_submission([single, dense, empty], test)
    assert submission["image_id"].to_list() == ["b", "a", "c"]
    assert submission["PredictionString"][0] == "0 0.50000 1.25 2.50 10.00 20.00"
    tokens = submission["PredictionString"][1].split()
    assert len(tokens) == 500 * 6
    assert tokens[:2] == ["7", "0.90000"]
    assert submission["PredictionString"][2] == ""


def test_coco_records_use_one_based_categories_and_xywh() -> None:
    prediction = {
        "boxes": np.array([[10, 20, 40, 60], [3, 3, 3, 8]], dtype=np.float32),
        "scores": np.array([0.7, 0.2], dtype=np.float32),
        "classes": np.array([7, 1]),
    }
    records = coco_detections([prediction], [42])
    assert records == [
        {
            "image_id": 42,
            "category_id": 8,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "score": pytest.approx(0.7),
        }
    ]


def test_epoch_plan_fits_budget_and_rejects_short_budgets() -> None:
    training = {
        **_settings()["training"],
        "budget_hours": 2.0,
        "min_epochs": 5,
        "max_epochs": 50,
        "validation_time_margin": 1.0,
    }
    assert plan_epochs(0.02, 60.0, 6000, training) == 40
    assert plan_epochs(0.001, 0.0, 6000, training) == 50
    with pytest.raises(ValueError):
        plan_epochs(1.0, 60.0, 6000, training)


def test_training_arguments_pass_every_recipe_value() -> None:
    settings = _settings()
    arguments = training_arguments(settings, Path("data.yaml"), Path("runs"), "run", 40)
    assert arguments["imgsz"] == settings["dataset"]["image_size"]
    assert arguments["epochs"] == 40
    assert arguments["exist_ok"] is True
    assert all(arguments[key] == settings["training"][key] for key in PASSTHROUGH_TRAINING_KEYS)


def test_ground_truth_outcomes_separate_error_types() -> None:
    frames = pl.DataFrame({"image_key": ["val/c/1"]})
    objects = pl.DataFrame(
        {
            "image_key": ["val/c/1"] * 4,
            "class_id": [0, 2, 3, 4],
            "x": [0.0, 100.0, 200.0, 300.0],
            "y": [0.0] * 4,
            "width": [10.0] * 4,
            "height": [10.0] * 4,
            "area": [100.0] * 4,
        }
    )
    prediction = {
        "boxes": np.array([[0, 0, 10, 10], [100, 0, 110, 10], [205, 0, 215, 10]], np.float32),
        "scores": np.array([0.9, 0.8, 0.7], np.float32),
        "classes": np.array([0, 5, 3]),
    }
    outcomes = classify_ground_truth_outcomes([prediction], frames, objects, 0.3)
    assert outcomes["outcome"].to_list() == ["Detected", "Misclassified", "Mislocalized", "Missed"]


def test_checkpoint_mirror_publishes_only_settled_new_checkpoints(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    published = []
    mirror = CheckpointMirror(
        run, tmp_path / "mirror", lambda directory: published.append(sorted(directory.iterdir()))
    )
    assert not mirror.mirror_once()
    (run / "weights" / "last.pt").write_bytes(b"epoch-1")
    (run / "results.csv").write_text("epoch\n1\n")
    assert not mirror.mirror_once()
    mirror.settle_seconds = 0.0
    assert mirror.mirror_once()
    assert not mirror.mirror_once()
    assert [path.name for path in published[0]] == ["last.pt", "results.csv"]
    assert (tmp_path / "mirror" / "last.pt").read_bytes() == b"epoch-1"


def test_fast_and_reference_coco_backends_agree_with_crowds(dataset_root: Path) -> None:
    annotation = dataset_root / "val/annotations_coco.json"
    image_ids = [1, 2, 3]
    x, y, width, height = TARGET
    predictions = [
        {"image_id": 1, "category_id": 3, "bbox": [x + 1, y, width, height], "score": 0.9},
        {"image_id": 2, "category_id": 3, "bbox": [x + 12, y + 8, width, height], "score": 0.8},
        {"image_id": 2, "category_id": 1, "bbox": [95.0, 55.0, 20.0, 20.0], "score": 0.7},
        {"image_id": 3, "category_id": 5, "bbox": [300.0, 150.0, 30.0, 30.0], "score": 0.6},
    ]
    for records in (predictions, []):
        reference = evaluate_coco_max_dets(annotation, records, image_ids, 500)
        fast = evaluate_coco_max_dets(annotation, records, image_ids, 500, "faster_coco_eval")
        for key in ("map", "ap50", "ap75", "ap_small", "ap_medium", "ap_large"):
            assert fast[key] == pytest.approx(reference[key], abs=1e-9)
        assert fast["per_class_ap"] == pytest.approx(reference["per_class_ap"], abs=1e-9)


def test_head_remap_reproduces_selected_source_class_scores(tmp_path: Path) -> None:
    torch.manual_seed(2026)
    source_network = DetectionModel("yolo11n.yaml", nc=11, verbose=False).eval()
    source = tmp_path / "source.pt"
    torch.save({"model": source_network, "epoch": -1}, source)
    channels = (0, 2, 3, 4, 5, 6, 8, 9)
    destination = tmp_path / "converted.pt"
    provenance = remap_detection_head(source, destination, channels, "yolo11n.yaml")
    converted = torch.load(destination, map_location="cpu", weights_only=False)["model"].float()
    converted.eval()
    image = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        expected = source_network(image)[0]
        actual = converted(image)[0]
    torch.testing.assert_close(actual[:, :4], expected[:, :4], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        actual[:, 4:], expected[:, 4:][:, list(channels)], rtol=1e-2, atol=1e-3
    )
    assert converted.names == dict(enumerate(CLASS_NAMES))
    assert provenance["source_channels"] == list(channels)


def test_coco_subset_merges_splits_with_crowds_and_aligned_ids(
    dataset_root: Path, tmp_path: Path
) -> None:
    frames, _, _ = load_tables(dataset_root)
    subset = frames.filter(
        ((pl.col("split") == "val") & (pl.col("frame_id") == 2))
        | ((pl.col("clip") == "clip_b") & (pl.col("frame_id") <= 2))
    ).sort("split", "clip", "frame_id")
    destination = tmp_path / "subset.json"
    image_ids = write_coco_subset(dataset_root, subset, destination)
    document = json.loads(destination.read_text())
    assert image_ids == [1, 2, 3]
    assert [image["id"] for image in document["images"]] == [1, 2, 3]
    assert [image["file_name"] for image in document["images"]] == [
        "images/clip_b/0000001.jpg",
        "images/clip_b/0000002.jpg",
        "images/clip_c/0000002.jpg",
    ]
    assert (
        sorted(annotation["iscrowd"] for annotation in document["annotations"]) == [0] * 3 + [1] * 3
    )
    assert len({annotation["id"] for annotation in document["annotations"]}) == 6
    x, y, width, height = TARGET
    predictions = [
        {"image_id": image_id, "category_id": 3, "bbox": [x, y, width, height], "score": 0.9}
        for image_id in image_ids
    ]
    metrics = evaluate_coco_max_dets(destination, predictions, image_ids, 100, "faster_coco_eval")
    assert metrics["map"] == pytest.approx(1.0)
