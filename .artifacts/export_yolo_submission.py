import hashlib
import json
import logging
import sys
from pathlib import Path

import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.yolo_baseline import YoloDetector
from hm2.detection_data import DetectionDataset, make_loader
from hm2.experiment_checkpoint import write_json
from hm2.retinanet import predict_loader
from hm2.submission import build_submission, validate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
LOGGER = logging.getLogger("yolo_submission")
CACHE = ROOT / "runs/submissions/yolo_baseline_stride5"
SOURCE = CACHE / "source"
DATASET = (
    Path.home() / ".cache/kagglehub/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026"
)
CHECKPOINT = SOURCE / "latest.pt"
with CHECKPOINT.open("rb") as stream:
    checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
manifest = json.loads((SOURCE / "manifest.json").read_text())
if checkpoint_sha256 != manifest["files"]["latest.pt"]:
    raise ValueError("Checkpoint differs from the verified Drive generation.")
state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
if state["extra_state"]["epochs_completed"] != 5:
    raise ValueError("Expected the completed fifth-epoch checkpoint.")
settings = state["config"]
arguments = {key: value for key, value in settings["model"].items() if key != "backend"}
arguments["pretrained"] = False
device = torch.device("mps")
torch.set_num_threads(2)
model = YoloDetector(**arguments, seed=settings["seed"], loss_gains=settings["loss"]["gains"])
model.load_state_dict(state["model"], strict=True)
model.to(device)
model.eval()
del state

test = pl.read_csv(DATASET / "test_images.csv", schema_overrides={"image_id": pl.String})
frames = test.select(
    pl.col("image_id").alias("submission_id"),
    pl.col("image_id").alias("image_key"),
    pl.col("file_name").alias("relative_path"),
    pl.col("width").alias("image_width"),
    pl.col("height").alias("image_height"),
).with_row_index("image_id")
if frames.height != 3811 or frames["submission_id"].n_unique() != 3811:
    raise ValueError("Expected all 3811 unique official test images.")
for relative_path in frames["relative_path"]:
    if not (DATASET / relative_path).is_file():
        raise FileNotFoundError(relative_path)
objects = pl.DataFrame(
    schema={"image_key": pl.String, "x": pl.Float64, "y": pl.Float64,
            "width": pl.Float64, "height": pl.Float64, "category_id": pl.Int64}
)
predictions, speeds = [], []
for start in range(0, frames.height, 100):
    selected = frames.slice(start, 100)
    path = CACHE / f"test_predictions_{start:04d}.pt"
    identity = {"checkpoint_sha256": checkpoint_sha256, "image_ids": selected["image_id"].to_list()}
    if path.exists():
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if saved["identity"] != identity:
            raise ValueError("Prediction chunk belongs to another checkpoint or test selection.")
    else:
        loader = make_loader(DetectionDataset(DATASET, selected, objects), 2, 0, device)
        batch_predictions, image_ids, speed = predict_loader(model, loader, device, warmup=0)
        if image_ids != identity["image_ids"]:
            raise ValueError("Inference order differs from the official test metadata.")
        saved = {"identity": identity, "predictions": batch_predictions, "speed": speed}
        temporary = path.with_suffix(".incoming")
        torch.save(saved, temporary)
        temporary.replace(path)
    predictions.extend(saved["predictions"])
    speeds.append(saved["speed"])
    LOGGER.info("Test inference %d/%d, %.2f images/s", len(predictions), frames.height,
                saved["speed"]["frames_per_second"])

sample_path = DATASET / "sample_submission.csv"
has_official_sample = sample_path.is_file()
sample = (
    pl.read_csv(sample_path, schema_overrides={"image_id": pl.String}) if has_official_sample
    else test.select("image_id").with_columns(pl.lit("").alias("PredictionString"))
)
submission = build_submission(predictions, frames, sample)
destination = ROOT / "submissions/yolo_baseline_stride5.csv"
temporary = destination.with_suffix(".csv.incoming")
submission.write_csv(temporary)
validation = validate_submission(
    pl.read_csv(temporary, schema_overrides={"image_id": pl.String, "PredictionString": pl.String}),
    sample,
)
temporary.replace(destination)
with destination.open("rb") as stream:
    csv_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
record = {
    "experiment_id": "yolo_baseline",
    "epoch": 5,
    "checkpoint_selection": "Final epoch; also highest full-validation COCO mAP",
    "checkpoint_sha256": checkpoint_sha256,
    "source_generation": manifest["generation"],
    "source_drive_folder": "17zwMD_8UMmdhv1UhHpacDD2SZt6HKAWU",
    "config": settings,
    "validation": validation,
    "official_sample_verified": has_official_sample,
    "test_metadata_sha256": hashlib.sha256((DATASET / "test_images.csv").read_bytes()).hexdigest(),
    "csv_sha256": csv_sha256,
    "device": str(device),
    "amp": False,
    "batch_size": 2,
    "inference_chunks": speeds,
    "box_format": "Original-image pixel top-left xywh",
    "class_ids": list(range(8)),
    "maximum_detections_per_image": 500,
}
write_json(CACHE / "submission_metadata.json", record)
LOGGER.info("Saved %s: %s", destination, validation)
