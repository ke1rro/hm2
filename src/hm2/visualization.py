from pathlib import Path

import polars as pl
from PIL import Image, ImageDraw, ImageFont

from hm2.data import CLASS_NAMES

CLASS_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#56B4E9",
    "#CC79A7",
    "#8C6D31",
    "#6F4C9B",
)
IGNORE_COLOR = "#FF00A8"


def draw_frame(
    image_path: Path,
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
    max_width: int = 1000,
    should_label: bool = True,
) -> Image.Image:
    """Draw target boxes and distinct translucent ignored regions on a display copy.

    Args:
        image_path: Original image to open without modifying it.
        objects: Target annotations for this image in original pixel coordinates.
        ignored: Separate ignored regions for this image in original pixel coordinates.
        max_width: Maximum display width in pixels.
        should_label: Whether to add compact class labels to target boxes.

    Returns:
        A resized RGB display image with a class and ignored-region legend.
    """
    with Image.open(image_path) as source:
        scale = min(1.0, max_width / source.width)
        canvas = source.convert("RGB").resize(
            (round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS
        )
    drawing = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default(size=13)
    for region in ignored.iter_rows(named=True):
        x, y, width, height = (region[key] * scale for key in ("x", "y", "width", "height"))
        drawing.rectangle(
            (x, y, x + width, y + height), fill=(255, 0, 168, 45), outline=IGNORE_COLOR, width=3
        )
    for box in objects.iter_rows(named=True):
        x, y, width, height = (box[key] * scale for key in ("x", "y", "width", "height"))
        color = CLASS_COLORS[box["class_id"]]
        drawing.rectangle((x, y, x + width, y + height), outline=color, width=2)
        if should_label:
            label = CLASS_NAMES[box["class_id"]]
            label_y = max(0, y - 15)
            text_box = drawing.textbbox((x, label_y), label, font=font)
            drawing.rectangle(text_box, fill=(0, 0, 0, 190))
            drawing.text((x, label_y), label, fill="white", font=font)
    legend_height = 54
    result = Image.new("RGB", (canvas.width, canvas.height + legend_height), "white")
    result.paste(canvas, (0, legend_height))
    legend = ImageDraw.Draw(result)
    entries = list(zip(CLASS_NAMES, CLASS_COLORS)) + [("Ignored region", IGNORE_COLOR)]
    for index, (name, color) in enumerate(entries):
        x = 8 + (index % 5) * (canvas.width // 5)
        y = 6 + (index // 5) * 24
        legend.rectangle((x, y + 2, x + 10, y + 12), fill=color)
        legend.text((x + 14, y), name, font=font, fill="black")
    return result


def draw_frame_pair(
    image_path: Path,
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
    panel_width: int = 760,
    should_label: bool = True,
) -> Image.Image:
    """Place an unchanged frame view next to its annotated display copy.

    Args:
        image_path: Source frame on disk.
        objects: Target annotations for this frame.
        ignored: Separate ignored regions for this frame.
        panel_width: Maximum width of each panel.
        should_label: Whether target boxes receive compact class labels.

    Returns:
        A labeled RGB comparison image containing original and annotated panels.
    """
    annotated = draw_frame(image_path, objects, ignored, panel_width, should_label)
    with Image.open(image_path) as source:
        original = source.convert("RGB").resize(
            (annotated.width, annotated.height - 54), Image.Resampling.LANCZOS
        )
    result = Image.new("RGB", (annotated.width * 2, annotated.height + 24), "white")
    result.paste(original, (0, 78))
    result.paste(annotated, (annotated.width, 24))
    drawing = ImageDraw.Draw(result)
    drawing.text((8, 4), "Original frame", fill="black", font=ImageFont.load_default(size=15))
    drawing.text(
        (annotated.width + 8, 4),
        "Ground truth and ignored regions",
        fill="black",
        font=ImageFont.load_default(size=15),
    )
    return result


def draw_prediction_pair(
    image_path: Path,
    objects: pl.DataFrame,
    ignored: pl.DataFrame,
    predictions: pl.DataFrame,
    panel_width: int = 700,
) -> Image.Image:
    """Compare ground truth with scored predictions using the same class and ignore colors.

    Args:
        image_path: Original source image.
        objects: Ground truth with zero-based visualization class IDs.
        ignored: Unique ignored-region geometries for this frame.
        predictions: Display-filtered predictions with zero-based class IDs and pixel xywh.
        panel_width: Maximum width of each panel.

    Returns:
        A ground-truth panel beside a prediction panel, with ignored areas visible in both.
    """
    ground_truth = draw_frame(image_path, objects, ignored, panel_width, objects.height <= 40)
    predicted = draw_frame(image_path, predictions, ignored, panel_width, predictions.height <= 40)
    result = Image.new("RGB", (ground_truth.width * 2, ground_truth.height + 24), "white")
    result.paste(ground_truth, (0, 24))
    result.paste(predicted, (ground_truth.width, 24))
    drawing = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=15)
    drawing.text((8, 3), "Ground truth", fill="black", font=font)
    drawing.text((ground_truth.width + 8, 3), "Predictions", fill="black", font=font)
    return result


def draw_crop(
    image_path: Path,
    box: tuple[float, float, float, float],
    label: str,
    zoom: int = 4,
    padding: int = 12,
) -> Image.Image:
    """Enlarge an object crop while retaining its source location and scale.

    Args:
        image_path: Original source frame.
        box: Original-pixel x, y, width, height of the target.
        label: Target class name.
        zoom: Integer display magnification.
        padding: Context margin in original-image pixels.

    Returns:
        An enlarged nearest-neighbor crop with the target outline and source coordinates.
    """
    x, y, width, height = box
    with Image.open(image_path) as source:
        left, top = max(0, int(x) - padding), max(0, int(y) - padding)
        right = min(source.width, round(x + width) + padding)
        bottom = min(source.height, round(y + height) + padding)
        crop = source.convert("RGB").crop((left, top, right, bottom))
    crop = crop.resize((crop.width * zoom, crop.height * zoom), Image.Resampling.NEAREST)
    drawing = ImageDraw.Draw(crop)
    drawing.rectangle(
        ((x - left) * zoom, (y - top) * zoom, (x + width - left) * zoom, (y + height - top) * zoom),
        outline="#FF3030",
        width=2,
    )
    result = Image.new("RGB", (max(320, crop.width), crop.height + 52), "white")
    result.paste(crop, (0, 52))
    drawing = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=13)
    drawing.text(
        (5, 3), f"{label}, {width:.1f} x {height:.1f} px, {zoom}x view", fill="black", font=font
    )
    drawing.text(
        (5, 23), f"Source x={x:.1f}, y={y:.1f}, crop ({left}, {top})", fill="black", font=font
    )
    return result
