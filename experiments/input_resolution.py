import math

import polars as pl


def resized_dimensions(height: int, width: int, min_size: int, max_size: int) -> tuple[int, int]:
    """Calculate the aspect-ratio-preserving dimensions of the detector's resize.

    Args:
        height: Original image height.
        width: Original image width.
        min_size: Target shorter side.
        max_size: Upper limit on the longer side.

    Returns:
        Integer height and width using the detector's interpolation scale convention.
    """
    scale = min(min_size / min(height, width), max_size / max(height, width))
    return math.floor(height * scale), math.floor(width * scale)


def resized_box_statistics(
    frames: pl.DataFrame,
    objects: pl.DataFrame,
    min_size: int,
    max_size: int,
) -> pl.DataFrame:
    """Project GT sizes into resized pixels using actual integer height and width ratios.

    Args:
        frames: Selected source frame dimensions.
        objects: Valid GT boxes in original pixels.
        min_size: Shorter-side target.
        max_size: Longer-side cap.

    Returns:
        Per-object resized width, height and area; these are not COCO AP size bins.
    """
    dimensions = (
        frames.select("image_key", "image_width", "image_height")
        .with_columns(
            pl.min_horizontal(
                min_size / pl.min_horizontal("image_width", "image_height"),
                max_size / pl.max_horizontal("image_width", "image_height"),
            ).alias("scale")
        )
        .with_columns(
            (pl.col("image_width") * pl.col("scale")).floor().alias("resized_width"),
            (pl.col("image_height") * pl.col("scale")).floor().alias("resized_height"),
        )
    )
    return (
        objects.select("image_key", "category_id", "width", "height")
        .join(dimensions, on="image_key", how="inner")
        .with_columns(
            (pl.col("width") * pl.col("resized_width") / pl.col("image_width")).alias("box_width"),
            (pl.col("height") * pl.col("resized_height") / pl.col("image_height")).alias(
                "box_height"
            ),
        )
        .with_columns((pl.col("box_width") * pl.col("box_height")).alias("box_area"))
    )
