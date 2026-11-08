import math

import polars as pl
import torch

from hm2.data import CLASS_NAMES
from hm2.retinanet_experiments import epoch_order


def _presence(frames: pl.DataFrame, objects: pl.DataFrame) -> pl.DataFrame:
    if frames.is_empty() or set(frames["split"]) != {"train"}:
        raise ValueError("RFS requires a nonempty training selection only.")
    return (
        objects.select("image_key", "category_id")
        .unique()
        .join(frames.select("image_key", "clip"), on="image_key", how="inner")
    )


def class_presence_statistics(frames: pl.DataFrame, objects: pl.DataFrame) -> pl.DataFrame:
    """Count training-image presence and clip concentration for all eight classes.

    Args:
        frames: Unique selected training frames, including frames with no GT.
        objects: Valid training labels; duplicate annotations do not change presence.

    Returns:
        Per-class image frequencies, clip counts and the largest clip's presence share.
    """
    presence = _presence(frames, objects)
    by_clip = presence.group_by("category_id", "clip").len(name="clip_images")
    counts = by_clip.group_by("category_id").agg(
        pl.col("clip_images").sum().alias("image_count"),
        pl.len().alias("clip_count"),
        (pl.col("clip_images").max() / pl.col("clip_images").sum()).alias("max_clip_share"),
    )
    return (
        pl.DataFrame({"category_id": range(1, 9), "class_name": CLASS_NAMES})
        .join(counts, on="category_id", how="left")
        .with_columns(
            pl.col("image_count", "clip_count").fill_null(0), pl.col("max_clip_share").fill_null(0)
        )
        .with_columns((pl.col("image_count") / frames.height).alias("frame_frequency"))
        .sort("category_id")
    )


def image_repeat_factors(
    frames: pl.DataFrame,
    objects: pl.DataFrame,
    threshold: float = 0.01,
    max_repeat: float = 4.0,
) -> pl.DataFrame:
    """Assign each training image its largest capped class-presence repeat factor.

    Args:
        frames: Unique training frames in baseline selection order.
        objects: Valid foreground labels.
        threshold: Image-presence threshold in (0, 1].
        max_repeat: Explicit upper bound, at least one.

    Returns:
        Aligned image_key, index and repeat_factor columns, with one for empty images.
    """
    if not 0 < threshold <= 1 or not math.isfinite(max_repeat) or max_repeat < 1:
        raise ValueError("Use a threshold in (0, 1] and a finite repeat cap >= 1.")
    classes = class_presence_statistics(frames, objects).with_columns(
        pl.when(pl.col("frame_frequency") > 0)
        .then((threshold / pl.col("frame_frequency")).sqrt().clip(1, max_repeat))
        .otherwise(1.0)
        .alias("repeat_factor")
    )
    factors = (
        _presence(frames, objects)
        .join(classes.select("category_id", "repeat_factor"), on="category_id")
        .group_by("image_key")
        .agg(pl.col("repeat_factor").max())
    )
    return (
        frames.select("image_key")
        .with_row_index("index")
        .join(factors, on="image_key", how="left", maintain_order="left")
        .with_columns(pl.col("repeat_factor").fill_null(1.0))
    )


def sample_epoch_indices(
    repeat_factors: list[float],
    seed: int,
    epoch: int,
    sample_count: int | None = None,
) -> list[int]:
    """Draw a fixed-length weighted epoch, or the exact baseline permutation for unit weights.

    Normalize image repeat factors into sampling probabilities rather than expanding
    an epoch. Replacement permits repeats while keeping the baseline batch budget.

    Args:
        repeat_factors: Positive bounded image factors in baseline selection order.
        seed: Shared experiment seed.
        epoch: Zero-based epoch for deterministic restart and cursor slicing.
        sample_count: Requested number of image draws, defaulting to the selection size.

    Returns:
        Exactly sample_count indices; all-unit default weights preserve the baseline order.
    """
    count = len(repeat_factors) if sample_count is None else sample_count
    if not repeat_factors or count < 1:
        raise ValueError("Sampling needs positive frame and draw counts.")
    weights = torch.tensor(repeat_factors, dtype=torch.float64)
    if not torch.isfinite(weights).all() or (weights < 1).any():
        raise ValueError("Repeat factors must be finite and at least one.")
    if (weights == 1).all() and count == len(repeat_factors):
        return epoch_order(count, seed, epoch)
    return torch.multinomial(
        weights, count, replacement=True, generator=torch.Generator().manual_seed(seed + epoch)
    ).tolist()


def sampling_exposure(
    frames: pl.DataFrame,
    objects: pl.DataFrame,
    factors: pl.DataFrame,
    indices: list[int],
) -> pl.DataFrame:
    """Compare baseline, expected and realized class-bearing image draws for one epoch.

    Args:
        frames: Selected training frames in baseline order.
        objects: Valid labels used only for image-level class presence.
        factors: Aligned image repeat factors.
        indices: Realized fixed-length sampling order.

    Returns:
        Image draws per class; classes co-occurring in a frame share its exposure.
    """
    draws = pl.DataFrame({"image_key": frames["image_key"].gather(indices)})
    probabilities = factors.select(
        "image_key", (pl.col("repeat_factor") / pl.col("repeat_factor").sum()).alias("p")
    )
    presence = _presence(frames, objects)
    expected = (
        presence.join(probabilities, on="image_key")
        .group_by("category_id")
        .agg((pl.col("p").sum() * len(indices)).alias("expected_draws"))
    )
    actual = presence.join(draws, on="image_key").group_by("category_id").len(name="actual_draws")
    return (
        class_presence_statistics(frames, objects)
        .join(expected, on="category_id", how="left")
        .join(actual, on="category_id", how="left")
        .with_columns(pl.col("expected_draws", "actual_draws").fill_null(0))
    )
