import math

import polars as pl
import torch


def validate_submission(submission: pl.DataFrame, sample: pl.DataFrame) -> dict:
    """Validate competition image membership and pixel-coordinate prediction strings.

    Args:
        submission: Candidate CSV contents with image_id and PredictionString columns.
        sample: Original competition sample defining the complete image membership.

    Returns:
        Validated image, detection and empty-prediction counts.
    """
    if submission.columns != ["image_id", "PredictionString"]:
        raise ValueError("Submission needs image_id and PredictionString columns in that order.")
    if (
        submission.height != sample.height
        or submission["image_id"].n_unique() != submission.height
        or set(submission["image_id"]) != set(sample["image_id"])
    ):
        raise ValueError("Submission image identifiers must match the complete sample exactly.")
    total, empty = 0, 0
    for value in submission["PredictionString"]:
        tokens = (value or "").split()
        if len(tokens) % 6 or len(tokens) > 500 * 6:
            raise ValueError("Each image needs at most 500 groups of six detection values.")
        empty += not tokens
        total += len(tokens) // 6
        for index in range(0, len(tokens), 6):
            label = int(tokens[index])
            score, x, y, width, height = map(float, tokens[index + 1 : index + 6])
            if not 0 <= label <= 7:
                raise ValueError("Submission class IDs must be integers from 0 to 7.")
            if (
                not all(math.isfinite(number) for number in (score, x, y, width, height))
                or not 0 <= score <= 1
                or min(x, y) < 0
                or min(width, height) <= 0
            ):
                raise ValueError("Submission scores and pixel boxes must be finite and valid.")
    return {"images": submission.height, "detections": total, "empty_images": empty}


def build_submission(
    predictions: list[dict], frames: pl.DataFrame, sample: pl.DataFrame
) -> pl.DataFrame:
    """Convert RetinaNet outputs to submission rows in the official sample's order.

    Use original-image pixel xywh, map foreground labels 1–8 to 0–7, retain empty
    rows and cap each image at 500 highest-scoring detections. No threshold is tuned.

    Args:
        predictions: CPU detector outputs aligned with the frame rows.
        frames: Metadata with submission_id, image_width and image_height.
        sample: Original competition sample submission.

    Returns:
        Validated image_id and PredictionString table, including all sample images.
    """
    if len(predictions) != frames.height:
        raise ValueError("Predictions must cover every test image exactly once.")
    rows = []
    for prediction, frame in zip(predictions, frames.iter_rows(named=True), strict=True):
        boxes = prediction["boxes"].clone().float()
        scores, labels = prediction["scores"], prediction["labels"]
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise ValueError("Detector produced non-finite predictions.")
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, frame["image_width"])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, frame["image_height"])
        boxes[:, 2:] -= boxes[:, :2]
        valid = (labels >= 1) & (labels <= 8) & (boxes[:, 2:] > 0).all(dim=1)
        indices = valid.nonzero().flatten()
        indices = indices[scores[indices].argsort(descending=True)[:500]]
        tokens = []
        for index in indices.tolist():
            tokens.extend(
                [
                    str(int(labels[index]) - 1),
                    f"{float(scores[index]):.8g}",
                    *(f"{value:.8g}" for value in boxes[index].tolist()),
                ]
            )
        rows.append({"image_id": frame["submission_id"], "PredictionString": " ".join(tokens)})
    result = pl.from_dicts(rows, schema={"image_id": pl.String, "PredictionString": pl.String})
    validate_submission(result, sample)
    return sample.select("image_id").join(result, on="image_id", how="left", maintain_order="left")
