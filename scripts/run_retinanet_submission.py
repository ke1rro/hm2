import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import polars as pl
import torch

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Infer all cached test images locally with a trained checkpoint and export a submission.

    Reuse complete cached predictions on repeated calls. Without the official sample,
    export an explicitly unverified CSV against test_images.csv membership; rerunning
    with the sample validates cached outputs and produces submission.csv without inference.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hm2.detection_data import DetectionDataset, make_loader
    from hm2.experiment_names import result_directory
    from hm2.retinanet import build_retinanet, predict_loader
    from hm2.submission import build_submission, validate_submission

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(
        description="Export RetinaNet standard-anchor stride-5 test predictions without training."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=result_directory(Path("runs/submissions"), "standard_anchors_stride5") / "best.pt",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.home()
        / ".cache/kagglehub/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=result_directory(Path("runs/submissions"), "standard_anchors_stride5"),
    )
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    test = pl.read_csv(
        arguments.dataset_root / "test_images.csv", schema_overrides={"image_id": pl.String}
    )
    frames = test.select(
        pl.col("image_id").alias("submission_id"),
        pl.col("image_id").alias("image_key"),
        pl.col("file_name").alias("relative_path"),
        pl.col("width").alias("image_width"),
        pl.col("height").alias("image_height"),
    ).with_row_index("image_id")
    if frames.height != 3811 or frames["submission_id"].n_unique() != 3811:
        raise ValueError("Expected all 3811 unique official test frames.")
    missing = [
        path for path in frames["relative_path"] if not (arguments.dataset_root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing test images, including {missing[:3]}")
    sample_path = arguments.sample_submission or arguments.dataset_root / "sample_submission.csv"
    has_official_sample = sample_path.is_file()
    sample = (
        pl.read_csv(sample_path, schema_overrides={"image_id": pl.String})
        if has_official_sample
        else test.select("image_id").with_columns(pl.lit("").alias("PredictionString"))
    )
    with arguments.checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    identity = {"checkpoint_sha256": checkpoint_sha256, "image_ids": test["image_id"].to_list()}
    cache = arguments.output / "test_predictions.pt"
    saved: dict
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Cached predictions belong to another checkpoint or test selection.")
        LOGGER.info("Reuse verified prediction cache %s", cache)
    else:
        state = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
        settings = state["config"]
        device = torch.device(arguments.device)
        model = build_retinanet(
            settings["variant"],
            pretrained=False,
            min_size=settings["min_size"],
            max_size=settings["max_size"],
        )
        model.load_state_dict(state["model"])
        model.to(device)
        objects = pl.DataFrame(
            schema={
                "image_key": pl.String,
                "x": pl.Float64,
                "y": pl.Float64,
                "width": pl.Float64,
                "height": pl.Float64,
                "category_id": pl.Int64,
            }
        )
        loader = make_loader(
            DetectionDataset(arguments.dataset_root, frames, objects), 2, 0, device
        )
        LOGGER.info(
            "Infer %s test frames using %s, checkpoint epoch %s",
            frames.height,
            device,
            state["extra_state"]["epochs_completed"],
        )
        predictions, image_ids, speed = predict_loader(model, loader, device)
        if image_ids != list(range(frames.height)):
            raise ValueError("Inference image order differs from test metadata.")
        saved = {
            "identity": identity,
            "predictions": predictions,
            "speed": speed,
            "config": settings,
        }
        temporary = cache.with_suffix(".incoming")
        torch.save(saved, temporary)
        temporary.replace(cache)
    submission = build_submission(saved["predictions"], frames, sample)
    filename = "submission.csv" if has_official_sample else "submission_unverified.csv"
    path = arguments.output / filename
    submission.write_csv(path)
    validation = validate_submission(
        pl.read_csv(path, schema_overrides={"image_id": pl.String}), sample
    )
    record = {
        **validation,
        "official_sample_verified": has_official_sample,
        "checkpoint_sha256": checkpoint_sha256,
        "config": saved["config"],
        "inference": saved["speed"],
    }
    (arguments.output / "submission_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    LOGGER.info("Saved %s: %s", path, validation)
    if not has_official_sample:
        LOGGER.warning(
            "Official sample is absent. Only test_images.csv membership is verified; "
            "rerun with --sample-submission before submitting."
        )


if __name__ == "__main__":
    main()
