import polars as pl
import pytest
import torch

from hm2.submission import build_submission, validate_submission


def test_submission_preserves_sample_order_and_original_pixel_coordinates() -> None:
    """Convert foreground IDs and xyxy boxes while retaining empty image rows."""
    sample = pl.DataFrame({"image_id": ["clip_b_1", "clip_a_1"], "PredictionString": ["", ""]})
    frames = pl.DataFrame(
        {
            "submission_id": ["clip_a_1", "clip_b_1"],
            "image_width": [100, 100],
            "image_height": [50, 50],
        }
    )
    predictions = [
        {
            "boxes": torch.tensor(
                [[2.0, 3.0, 10.0, 10.0], [-2.0, 40.0, 105.0, 60.0], [0.0, 0.0, 10.0, 10.0]]
            ),
            "labels": torch.tensor([8, 1, 0]),
            "scores": torch.tensor([0.9, 0.8, 0.99]),
        },
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty(0, dtype=torch.int64),
            "scores": torch.empty(0),
        },
    ]
    result = build_submission(predictions, frames, sample)
    assert result["image_id"].to_list() == ["clip_b_1", "clip_a_1"]
    assert result["PredictionString"][0] == ""
    values = result["PredictionString"][1].split()
    assert [int(values[0]), int(values[6])] == [7, 0]
    assert list(map(float, values[2:6])) == [2.0, 3.0, 8.0, 7.0]
    assert list(map(float, values[8:12])) == [0.0, 40.0, 100.0, 10.0]
    assert validate_submission(result, sample)["images"] == 2


def test_submission_caps_detections_and_rejects_missing_images() -> None:
    """Enforce the competition row and detection-count contracts."""
    sample = pl.DataFrame({"image_id": ["test"], "PredictionString": [""]})
    frames = pl.DataFrame({"submission_id": ["test"], "image_width": [100], "image_height": [50]})
    prediction = {
        "boxes": torch.tensor([[1.0, 2.0, 10.0, 20.0]]).repeat(501, 1),
        "labels": torch.ones(501, dtype=torch.int64),
        "scores": torch.full((501,), 0.5),
    }
    result = build_submission([prediction], frames, sample)
    assert len(result["PredictionString"][0].split()) == 500 * 6
    with pytest.raises(ValueError, match="image"):
        validate_submission(result.head(0), sample)
    invalid = result.with_columns(pl.lit("8 0.5 0 0 10 10").alias("PredictionString"))
    with pytest.raises(ValueError, match="class"):
        validate_submission(invalid, sample)
