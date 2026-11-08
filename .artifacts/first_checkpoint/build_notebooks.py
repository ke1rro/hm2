import ast
import logging
import sys
from pathlib import Path
from textwrap import dedent

import nbformat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def _markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(text).strip())


def _code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(text).strip())


def _write(name: str, cells: list[nbformat.NotebookNode]) -> None:
    rendered_cells = []
    for cell in cells:
        if cell.cell_type != "code":
            rendered_cells.append(cell)
            continue
        pending = []
        for statement in ast.parse(cell.source).body:
            expression = statement.value if isinstance(statement, ast.Expr) else None
            if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name) and expression.func.id == "display":
                for argument in expression.args:
                    pending.append(ast.unparse(argument))
                    rendered_cells.append(_code("\n\n".join(pending)))
                    pending = []
            elif isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute) and expression.func.attr == "show":
                pending.append(ast.unparse(expression.func.value))
                rendered_cells.append(_code("\n\n".join(pending)))
                pending = []
            else:
                source = ast.unparse(statement)
                if (
                    isinstance(statement, (ast.Import, ast.ImportFrom))
                    and pending
                    and isinstance(ast.parse(pending[-1]).body[-1], (ast.Import, ast.ImportFrom))
                ):
                    pending[-1] += "\n" + source
                else:
                    pending.append(source)
        if pending:
            rendered_cells.append(_code("\n\n".join(pending)))
    notebook = nbformat.v4.new_notebook(cells=rendered_cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3 (hm2)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
        "checkpoint": "Data understanding and preprocessing",
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / "notebooks" / name)
    LOGGER.info("Created %s with %s cells", name, len(rendered_cells))


ROOT_CELL = """
working_directory = Path.cwd().resolve()
REPOSITORY_ROOT = next(
    directory for directory in (working_directory, *working_directory.parents)
    if (directory / "pyproject.toml").is_file() and (directory / "src/hm2").is_dir()
)
METADATA_DIRECTORY = REPOSITORY_ROOT / "data/metadata"
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
SEED = 2026
pl.Config.set_tbl_rows(12)
pl.Config.set_tbl_cols(12)
"""

data_cells = [
    _markdown("""
    # Data

    ## 1. Paths and prerequisites

    We keep the raw competition files unchanged and prepare small metadata tables for EDA.
    This notebook uses CPU metadata parsing and opens only a sample of image headers.

    Run `uv sync --locked` first and select the project kernel. The first download is
    about 10 GB. Local files take priority. Set `HM2_DATA_DIR` in `.env` to use an existing
    directory or ZIP. Otherwise, [KaggleHub](https://github.com/Kaggle/kagglehub) reuses its
    cache. Accept the competition rules in the browser and configure Kaggle authentication
    or `KAGGLE_API_TOKEN` in `.env` before downloading. Credentials are never displayed.

    The expected inventory comes from [the assignment context](../docs/assignment_context.md).
    """),
    _code(dedent("""
    import json
    import logging
    import sys
    from pathlib import Path
    from zipfile import ZipFile

    import kagglehub
    import polars as pl
    from decouple import config
    from IPython.display import Markdown, display
    from kagglehub.config import set_kaggle_api_token
    from kagglehub.exceptions import KaggleApiHTTPError, UnauthenticatedError
    from PIL import Image
    from tqdm.notebook import tqdm

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    """) + ROOT_CELL + dedent("""
    from hm2.data import (
        CLASS_NAMES, COMPETITION, audit_dataset, compare_yolo_coco, extract_archive,
        find_dataset_root, load_tables,
    )
    """)),
    _markdown("""
    ## 2. Locate or download the data

    We check for extracted metadata before considering an archive or a download.
    Extraction preserves wrapper directories and refuses to overwrite existing files.
    An incomplete local dataset is reported by the checks below rather than overwritten.
    """),
    _code("""
    configured_directory = config("HM2_DATA_DIR", default=None)
    DATA_DIRECTORY = Path(configured_directory or REPOSITORY_ROOT / "data").expanduser()
    dataset_root = find_dataset_root(DATA_DIRECTORY)
    archives = [DATA_DIRECTORY] if DATA_DIRECTORY.is_file() else sorted(DATA_DIRECTORY.glob("*.zip"))
    manifest_path = METADATA_DIRECTORY / "manifest.json"
    if dataset_root is None and not configured_directory and manifest_path.is_file():
        previous_root = Path(json.loads(manifest_path.read_text())["dataset_root"])
        dataset_root = find_dataset_root(previous_root)

    if dataset_root is None and archives:
        if len(archives) != 1:
            raise ValueError("More than one ZIP found. Set HM2_DATA_DIR to the intended ZIP.")
        with ZipFile(archives[0]) as archive:
            archive_summary = pl.DataFrame({
                "Archive": [archives[0].name], "Files": [len(archive.infolist())],
                "Uncompressed GiB": [sum(item.file_size for item in archive.infolist()) / 1024**3],
            })
            display(archive_summary)
            display(pl.DataFrame({"Example path": archive.namelist()[:8]}))
        extraction_directory = archives[0].parent / "extracted" / archives[0].stem
        if find_dataset_root(extraction_directory) is None:
            extract_archive(archives[0], extraction_directory)
        dataset_root = find_dataset_root(extraction_directory)

    if dataset_root is None:
        token = config("KAGGLE_API_TOKEN", default=None)
        if token:
            set_kaggle_api_token(token)
        del token
        try:
            downloaded_path = Path(kagglehub.competition_download(COMPETITION))
        except (KaggleApiHTTPError, UnauthenticatedError):
            raise RuntimeError(
                "Kaggle access failed. Check KAGGLE_API_TOKEN in .env or your existing "
                "Kaggle authentication, accept the competition rules, or set HM2_DATA_DIR "
                "to an existing local dataset."
            ) from None
        dataset_root = find_dataset_root(downloaded_path)
    if dataset_root is None:
        raise FileNotFoundError("No test_images.csv found. Inspect the extracted archive layout.")
    dataset_root = dataset_root.resolve()
    display(pl.DataFrame({"Competition": [COMPETITION], "Dataset root": [str(dataset_root)]}))
    """),
    _markdown("""
    ## 3. Dataset structure and inventory

    We read train and validation COCO metadata and the test CSV. Target categories are
    checked against the eight-class mapping. COCO crowd annotations remain in a separate
    ignored-region table. A test frame has no target count because its labels are unknown.
    """),
    _code("""
    layout = []
    for split in ("train", "val", "test"):
        for directory in (dataset_root / split).iterdir():
            layout.append({"Split": split, "Path": str(directory.relative_to(dataset_root)),
                           "Kind": "Directory" if directory.is_dir() else "File"})
    display(pl.from_dicts(layout))
    frames, objects, ignored_regions = load_tables(dataset_root)
    display(pl.DataFrame({"YOLO class ID": range(8), "COCO category ID": range(1, 9),
                          "Class": CLASS_NAMES}))
    """),
    _code("""
    expected_inventory = pl.DataFrame({
        "split": ["train", "val", "test"],
        "expected_frames": [24201, 2846, 3811], "expected_clips": [56, 7, 48],
    })
    inventory = frames.group_by("split").agg(
        pl.len().alias("metadata_frames"), pl.col("clip").n_unique().alias("clips"),
        pl.col("image_width").min().alias("min_width_px"),
        pl.col("image_width").max().alias("max_width_px"),
        pl.col("image_height").min().alias("min_height_px"),
        pl.col("image_height").max().alias("max_height_px"),
    )
    file_inventory = []
    for split in ("train", "val", "test"):
        paths = list((dataset_root / split / "images").glob("*/*.jpg"))
        file_inventory.append({
            "split": split, "image_files": len(paths),
            "file_clips": len({path.parent.name for path in paths}),
            "has_annotations": (dataset_root / split / "annotations_coco.json").is_file(),
            "label_files": len(list((dataset_root / split / "labels").glob("*/*.txt"))),
            "ignore_files": len(list((dataset_root / split / "ignore").glob("*/*.txt"))),
        })
    inventory = inventory.join(pl.from_dicts(file_inventory), on="split").join(
        expected_inventory, on="split"
    ).with_columns(
        ((pl.col("metadata_frames") == pl.col("expected_frames"))
         & (pl.col("image_files") == pl.col("expected_frames"))
         & (pl.col("clips") == pl.col("expected_clips"))
         & (pl.col("file_clips") == pl.col("expected_clips"))).alias("is_complete")
    ).sort("split")
    display(inventory)
    display(Markdown(
        f"**Observation.** The inventory contains {frames.height:,} image records. "
        f"{inventory.filter(pl.col('is_complete')).height} of 3 splits match the published "
        "frame and clip counts. Any mismatch needs investigation before experiments."
    ))
    """),
    _markdown("""
    ## 4. Annotation records and file integrity

    COCO stores pixel top-left `x y width height` with category IDs 1 to 8. YOLO stores
    normalized center `cx cy width height` with class IDs 0 to 7. Ignored-region text files
    omit the class. We keep their coordinates and raw attributes without clipping boxes.

    The downloaded COCO files repeat each ignored region for all eight target categories.
    We group identical crowd geometry for region counts and overlays, retaining all source
    annotation IDs and category IDs in list columns. This does not change the raw files.
    """),
    _code("""
    test_metadata = pl.read_csv(dataset_root / "test_images.csv")
    display(test_metadata.head(3))
    if (dataset_root / "sample_submission.csv").is_file():
        sample_submission = pl.read_csv(dataset_root / "sample_submission.csv")
        display(sample_submission.head(3))
    else:
        display(Markdown(
            "**Inventory mismatch.** `sample_submission.csv` is absent from the downloaded "
            "competition files. Sample-row validation is unavailable. We have not generated "
            "a replacement or any submission predictions."
        ))
    display(objects.select(
        "image_key", "category_id", "class_id", "class_name", "x", "y", "width", "height",
        "occlusion", "truncation", "track_id", "orig_category_id",
    ).head(5))
    display(ignored_regions.select("image_key", "x", "y", "width", "height",
                                   "coco_replication_count", "coco_category_ids").head(3))
    integrity = audit_dataset(dataset_root, frames, objects, ignored_regions)
    display(integrity)
    """),
    _code("""
    header_checks = []
    sampled_frames = frames.sample(n=min(18, frames.height), seed=SEED)
    resolution_extremes = frames.sort("image_width").group_by("split", maintain_order=True).agg(
        pl.col("image_key").first().alias("smallest"), pl.col("image_key").last().alias("largest")
    )
    extreme_keys = resolution_extremes["smallest"].to_list() + resolution_extremes["largest"].to_list()
    sampled_frames = pl.concat([sampled_frames, frames.filter(pl.col("image_key").is_in(extreme_keys))]).unique("image_key")
    for record in tqdm(sampled_frames.iter_rows(named=True), total=sampled_frames.height):
        with Image.open(dataset_root / record["relative_path"]) as image:
            header_checks.append({
                "image_key": record["image_key"], "header_width": image.width,
                "header_height": image.height,
                "is_match": image.size == (record["image_width"], record["image_height"]),
            })
    header_checks = pl.from_dicts(header_checks)
    coordinate_checks = compare_yolo_coco(dataset_root, frames, objects, ignored_regions,
                                         sample_size=24, seed=SEED)
    display(header_checks, coordinate_checks.head(8))
    display(Markdown(
        f"**Resolution check.** Source metadata reaches {frames['image_width'].max()} × "
        f"{frames['image_height'].max()} px, above the 1920 × 1080 maximum in the supplied "
        "context. The header sample includes resolution extremes from every split."
    ))
    display(Markdown(
        f"**Observation.** {header_checks.filter(~pl.col('is_match')).height} sampled image "
        f"headers and {coordinate_checks.filter(~pl.col('is_match')).height} sampled "
        "YOLO/COCO comparisons disagree. The comparison tolerance is 0.01 original-image "
        "pixels to allow for six-decimal YOLO rounding."
    ))
    """),
    _markdown("""
    ## 5. Save metadata for EDA

    Parquet avoids repeatedly parsing the COCO JSON in the analysis notebook. The manifest
    records the source metadata sizes and modification times, the resolved dataset path,
    and the seed. `01_eda.ipynb` checks this manifest before loading the tables.
    """),
    _code("""
    METADATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, table in (("frames", frames), ("objects", objects), ("ignored_regions", ignored_regions)):
        output_path = METADATA_DIRECTORY / f"{name}.parquet"
        table.write_parquet(output_path)
        outputs.append({"Table": name, "Rows": table.height, "Size MiB": output_path.stat().st_size / 1024**2})
    source_names = ["train/annotations_coco.json", "val/annotations_coco.json",
                    "test_images.csv", "sample_submission.csv"]
    source_names = [name for name in source_names if (dataset_root / name).is_file()]
    manifest = {
        "dataset_root": str(dataset_root), "seed": SEED,
        "sources": {name: {"size": (dataset_root / name).stat().st_size,
                           "mtime_ns": (dataset_root / name).stat().st_mtime_ns} for name in source_names},
        "has_inventory_mismatches": not inventory["is_complete"].all(),
        "integrity_anomaly_count": int(integrity["count"].sum()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\\n")
    integrity.write_csv(METADATA_DIRECTORY / "integrity.csv")
    inventory.write_csv(METADATA_DIRECTORY / "inventory.csv")
    display(pl.from_dicts(outputs))
    """),
    _markdown("""
    The outputs are `data/metadata/frames.parquet`, `objects.parquet`,
    `ignored_regions.parquet`, the source manifest, and inventory checks. Raw images,
    labels, and ignored regions have not been resized or changed.

    Open `01_eda.ipynb` next. It requires these tables and the original images at the path
    recorded in the manifest. Review any reported anomalies before using the data for training.
    """),
]
_write("00_data.ipynb", data_cells)

eda_cells = [
    _markdown("""
    # EDA

    ## 1. Load the data

    We use the metadata prepared in `00_data.ipynb`. Target counts exclude ignored regions,
    and test frames remain unlabeled. All sizes below refer to original-image pixels.
    """),
    _code(dedent("""
    import json
    import sys
    from io import BytesIO
    from pathlib import Path

    import numpy as np
    import plotly.express as px
    import plotly.io
    import polars as pl
    from IPython.display import Image as DisplayImage
    from IPython.display import Markdown, display
    from PIL import Image, ImageDraw

    """) + ROOT_CELL + dedent("""
    from hm2.data import CLASS_NAMES, compare_yolo_coco, read_yolo_boxes, xywh_to_yolo, yolo_to_xywh
    from hm2.visualization import CLASS_COLORS, draw_crop, draw_frame, draw_frame_pair
    manifest_path = METADATA_DIRECTORY / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run 00_data.ipynb before this notebook.")
    manifest = json.loads(manifest_path.read_text())
    dataset_root = Path(manifest["dataset_root"])
    for name, signature in manifest["sources"].items():
        stat = (dataset_root / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (signature["size"], signature["mtime_ns"]):
            raise ValueError(f"Source metadata changed: {name}. Rerun 00_data.ipynb.")
    frames = pl.read_parquet(METADATA_DIRECTORY / "frames.parquet")
    objects = pl.read_parquet(METADATA_DIRECTORY / "objects.parquet")
    ignored_regions = pl.read_parquet(METADATA_DIRECTORY / "ignored_regions.parquet")
    labeled_frames = frames.filter(pl.col("is_labeled"))
    colors = dict(zip(CLASS_NAMES, CLASS_COLORS))
    plotly.io.renderers.default = "plotly_mimetype"

    def display_image(image: Image.Image) -> None:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=50, optimize=True)
        display(DisplayImage(data=buffer.getvalue()))

    display(frames.group_by("split").agg(pl.len().alias("Frames"),
             pl.col("clip").n_unique().alias("Clips"), pl.col("is_labeled").first()).sort("split"))
    display(pl.DataFrame({"Class ID": range(8), "COCO category ID": range(1, 9), "Class": CLASS_NAMES}))
    display(pl.read_csv(METADATA_DIRECTORY / "integrity.csv"))
    """)),
    _code("""
    train_clips = set(frames.filter(pl.col("split") == "train")["clip"])
    val_clips = set(frames.filter(pl.col("split") == "val")["clip"])
    overlapping_clips = sorted(train_clips & val_clips)
    display(pl.DataFrame({"Train clips": [len(train_clips)], "Validation clips": [len(val_clips)],
                          "Overlapping clips": [len(overlapping_clips)]}))
    if overlapping_clips:
        raise ValueError("Train and validation share clips. Investigate before experiments.")
    """),
    _markdown("""
    The official validation split stays separate. Neighboring frames can show the same
    objects and background, so a random frame split could leak sequence information.
    Any later internal folds must group by clip.

    ## 2. Visual overview

    We choose examples from metadata with seed 2026. The criteria cover sparse and crowded
    scenes, tiny boxes, recorded occlusion, ignored regions, and the rarest observed class.
    Duplicate selections are shown once. Magenta translucent regions are ignored areas.
    They are not background negatives. Dense views use colors without repeated text labels.
    """),
    _code("""
    frame_sizes = objects.group_by("image_key").agg(
        pl.col("area").min().alias("min_area"), pl.col("occlusion").max().alias("max_occlusion"),
    )
    candidates = labeled_frames.join(frame_sizes, on="image_key", how="left").sample(
        fraction=1.0, shuffle=True, seed=SEED
    )
    rare_class = objects.group_by("class_name").len().sort("len", "class_name")["class_name"][0]
    rare_keys = objects.filter(pl.col("class_name") == rare_class)["image_key"].unique().sort()
    selections = [
        ("Sparse scene", candidates.filter(pl.col("target_count") > 0).sort("target_count").head(1)),
        ("Crowded scene", candidates.sort("target_count", descending=True).head(1)),
        ("Tiny objects", candidates.filter(pl.col("min_area").is_not_null()).sort("min_area").head(1)),
        ("High recorded occlusion code", candidates.filter(pl.col("max_occlusion").is_not_null())
         .sort("max_occlusion", descending=True).head(1)),
        ("Ignored regions", candidates.filter(pl.col("ignore_count") > 0)
         .sort("ignore_count", descending=True).head(1)),
        (f"Rare class: {rare_class}", candidates.filter(pl.col("image_key").is_in(rare_keys.implode())).head(1)),
    ]
    shown_keys = set()
    selected_records = []
    for reason, selection in selections:
        if selection.is_empty():
            continue
        record = selection.row(0, named=True)
        if record["image_key"] in shown_keys:
            continue
        shown_keys.add(record["image_key"])
        selected_records.append({"Reason": reason, "Image": record["image_key"],
                                 "Targets": record["target_count"], "Ignored regions": record["ignore_count"]})
        display(Markdown(f"**{reason}.** `{record['image_key']}`, "
                         f"{record['image_width']} × {record['image_height']} px, "
                         f"{record['target_count']} targets."))
        display_image(draw_frame_pair(
            dataset_root / record["relative_path"],
            objects.filter(pl.col("image_key") == record["image_key"]),
            ignored_regions.filter(pl.col("image_key") == record["image_key"]),
            panel_width=560, should_label=record["target_count"] <= 60,
        ))
    display(pl.from_dicts(selected_records))
    """),
    _markdown("""
    ## 3. Class distribution

    We count target boxes, not unique tracked objects. The same tracked object can appear
    in many frames. The table retains all eight classes, including a zero count if needed.
    """),
    _code("""
    classes = pl.DataFrame({"class_id": range(8), "class_name": CLASS_NAMES})
    split_classes = pl.DataFrame({"split": ["train", "val"]}).join(classes, how="cross")
    class_counts = split_classes.join(
        objects.group_by("split", "class_id", "class_name").len(name="count"),
        on=["split", "class_id", "class_name"], how="left",
    ).with_columns(pl.col("count").fill_null(0)).with_columns(
        (pl.col("count") / pl.col("count").sum().over("split")).alias("proportion")
    ).sort("split", "class_id")
    display(class_counts)
    px.bar(class_counts, x="class_name", y="count", color="split", barmode="group",
           category_orders={"class_name": list(CLASS_NAMES)},
           labels={"class_name": "Class", "count": "Target annotations", "split": "Split"},
           title="Target class counts by split").show()
    overall_classes = classes.join(objects.group_by("class_id").len(name="count"),
                                    on="class_id", how="left").with_columns(pl.col("count").fill_null(0))
    overall_classes = overall_classes.with_columns((pl.col("count") / pl.col("count").sum()).alias("proportion"))
    display(overall_classes)
    """),
    _code("""
    clip_class_counts = objects.group_by("split", "clip", "class_name").len(name="count")
    clip_grid = labeled_frames.select("split", "clip").unique().join(
        classes.select("class_name"), how="cross"
    )
    clip_proportions = clip_grid.join(clip_class_counts, on=["split", "clip", "class_name"],
                                      how="left").with_columns(pl.col("count").fill_null(0)).with_columns(
        (pl.col("count") / pl.col("count").sum().over("split", "clip")).fill_nan(0).alias("proportion"),
        (pl.col("split") + "/" + pl.col("clip")).alias("clip_label"),
    ).sort("clip_label")
    px.density_heatmap(clip_proportions, x="class_name", y="clip_label", z="proportion",
                       histfunc="sum", color_continuous_scale="Blues", height=950,
                       category_orders={"class_name": list(CLASS_NAMES)},
                       labels={"class_name": "Class", "clip_label": "Clip", "proportion": "Within-clip fraction"},
                       title="Class proportions vary across labeled clips").show()
    """),
    _markdown("""
    ## 4. Object sizes

    We compare width, height, area, and aspect ratio in original-image pixels. The area
    distribution is aggregated before plotting, so the notebook does not embed every box.
    Small means area below 32² px², medium means 32² to below 96² px², and large means
    at least 96² px². These bins describe source boxes and do not measure detector quality.
    """),
    _code("""
    size_quantiles = objects.group_by("class_name").agg(
        pl.len().alias("Objects"), pl.col("width").median().alias("Median width px"),
        pl.col("height").median().alias("Median height px"),
        pl.col("area").quantile(0.1).alias("Area p10 px²"),
        pl.col("area").median().alias("Median area px²"),
        pl.col("area").quantile(0.9).alias("Area p90 px²"),
        pl.col("aspect_ratio").quantile(0.1).alias("Aspect ratio p10"),
        pl.col("aspect_ratio").median().alias("Median aspect ratio"),
        pl.col("aspect_ratio").quantile(0.9).alias("Aspect ratio p90"),
    ).sort("Objects", descending=True)
    display(size_quantiles)
    size_fractions = objects.group_by("split", "size_group").len(name="count").with_columns(
        (pl.col("count") / pl.col("count").sum().over("split")).alias("fraction")
    )
    display(size_fractions.sort("split", "size_group"))
    area_distribution = objects.filter(pl.col("area") > 0).with_columns(
        ((pl.col("area").log10() * 4).floor() / 4).alias("log_bin")
    ).group_by("split", "log_bin").len(name="count").with_columns(
        (10 ** (pl.col("log_bin") + 0.125)).alias("area_px2")
    ).sort("split", "area_px2")
    px.line(area_distribution, x="area_px2", y="count", color="split", log_x=True,
            labels={"area_px2": "Box area (px², logarithmic bins)", "count": "Target annotations", "split": "Split"},
            title="Box area distribution in original images").show()
    """),
    _code("""
    shape_distribution = objects.filter((pl.col("width") > 0) & (pl.col("height") > 0)).with_columns(
        (pl.col("width").log10() * 4).floor().alias("width_bin"),
        (pl.col("height").log10() * 4).floor().alias("height_bin"),
    ).group_by("width_bin", "height_bin").len(name="count").with_columns(
        (10 ** ((pl.col("width_bin") + 0.5) / 4)).alias("width_px"),
        (10 ** ((pl.col("height_bin") + 0.5) / 4)).alias("height_px"),
    )
    px.scatter(shape_distribution, x="width_px", y="height_px", size="count", color="count",
               log_x=True, log_y=True, color_continuous_scale="Blues", size_max=30,
               labels={"width_px": "Box width (px)", "height_px": "Box height (px)", "count": "Annotations in bin"},
               title="Box widths and heights across labeled frames").show()
    per_clip_sizes = objects.group_by("split", "clip").agg(
        pl.col("area").median().alias("Median area px²"),
        (pl.col("size_group") == "Small").mean().alias("Small-object fraction"),
    )
    display(per_clip_sizes.sort("Small-object fraction", descending=True).head(8))
    """),
    _markdown("""
    We also inspect typical small targets at 4× magnification. For each represented class,
    We choose the small box nearest that class's median small-object area. Each crop retains
    source pixel coordinates and a red target outline. Magnification adds no image detail.
    """),
    _code("""
    tiny_objects = objects.filter((pl.col("area") > 0) & (pl.col("area") < 32**2)).with_columns(
        (pl.col("area") - pl.col("area").median().over("class_name")).abs().alias("median_distance")
    ).sort("median_distance", "image_key", "annotation_id").unique(
        "class_name", keep="first", maintain_order=True
    ).head(6)
    crops = []
    for record in tiny_objects.iter_rows(named=True):
        crops.append(draw_crop(dataset_root / record["relative_path"],
                     tuple(record[key] for key in ("x", "y", "width", "height")), record["class_name"]))
    if crops:
        cell_width = max(crop.width for crop in crops)
        cell_height = max(crop.height for crop in crops)
        crop_grid = Image.new("RGB", (cell_width * 3, cell_height * 2), "white")
        for index, crop in enumerate(crops):
            crop_grid.paste(crop, ((index % 3) * cell_width, (index // 3) * cell_height))
        display_image(crop_grid)
    display(tiny_objects.select("class_name", "image_key", "x", "y", "width", "height"))
    """),
    _markdown("""
    ## 5. Objects per frame

    The counts include labeled frames with zero targets. We compare the center and tail
    of the distribution because dense scenes may need different later sampling and
    detection limits. These are dataset properties, not measured detector failures.
    """),
    _code("""
    density_summary = labeled_frames.group_by("split").agg(
        pl.len().alias("Frames"), pl.col("target_count").mean().alias("Mean targets"),
        pl.col("target_count").median().alias("Median targets"),
        pl.col("target_count").quantile(0.95).alias("Targets p95"),
        pl.col("target_count").quantile(0.99).alias("Targets p99"),
        pl.col("target_count").max().alias("Maximum targets"),
        (pl.col("target_count") == 0).sum().alias("Empty target frames"),
    ).sort("split")
    display(density_summary)
    density_bins = labeled_frames.with_columns(
        ((pl.col("target_count") // 25) * 25).alias("bin_start")
    ).group_by("split", "bin_start").len(name="frame_count").sort("split", "bin_start")
    px.bar(density_bins, x="bin_start", y="frame_count", color="split", barmode="group",
           labels={"bin_start": "Targets per frame (25-object bins)", "frame_count": "Frames", "split": "Split"},
           title="Scene density including zero-target frames").show()
    clip_density = labeled_frames.group_by("split", "clip").agg(
        pl.len().alias("Frames"), pl.col("target_count").mean().alias("Mean targets"),
        pl.col("target_count").median().alias("Median targets"),
        pl.col("target_count").quantile(0.95).alias("Targets p95"),
        pl.col("target_count").max().alias("Maximum targets"),
    )
    display(clip_density.sort("Mean targets", descending=True).head(8))
    """),
    _markdown("""
    ## 6. Resolution and split differences

    We compare image dimensions for all three splits, then compare label-dependent statistics
    only for train and validation. Test resolution and appearance can be inspected without
    inventing test labels or selecting model settings from test results.
    """),
    _code("""
    resolution_counts = frames.group_by("split", "image_width", "image_height").len(name="count")
    px.scatter(resolution_counts, x="image_width", y="image_height", color="split", size="count",
               size_max=45, labels={"image_width": "Image width (px)", "image_height": "Image height (px)",
                                    "count": "Frames", "split": "Split"},
               title="Frame resolutions by split").show()
    display(frames.with_columns((pl.col("image_width") / pl.col("image_height")).alias("aspect_ratio"))
            .group_by("split").agg(pl.col("aspect_ratio").min().alias("Min aspect ratio"),
                                  pl.col("aspect_ratio").median().alias("Median aspect ratio"),
                                  pl.col("aspect_ratio").max().alias("Max aspect ratio")))
    split_sizes = objects.group_by("split").agg(
        pl.col("area").median().alias("Median area px²"),
        (pl.col("size_group") == "Small").mean().alias("Small-object fraction"),
    )
    display(density_summary.join(split_sizes, on="split"))
    test_example = frames.filter(pl.col("split") == "test").sort("image_key").sample(n=1, seed=SEED).row(0, named=True)
    display(Markdown(f"**Unlabeled test example.** `{test_example['image_key']}`, "
                     f"{test_example['image_width']} × {test_example['image_height']} px."))
    with Image.open(dataset_root / test_example["relative_path"]) as source:
        test_view = source.convert("RGB")
        test_view.thumbnail((700, 450))
    display_image(test_view)
    """),
    _markdown("""
    ## 7. Ignored regions, occlusion, and truncation

    The magenta gallery regions are unique COCO crowd geometries. The source COCO repeats
    each region across categories, and the table retains those source IDs separately.
    We compare the geometries with the
    optional ignore text files later. Individual region areas do not measure union coverage,
    since regions can overlap. Occlusion and truncation codes below are raw metadata values.
    We do not assign ordinal meanings without verifying the source encoding.
    """),
    _code("""
    display(ignored_regions.group_by("split").agg(
        pl.len().alias("Ignored regions"), pl.col("image_key").n_unique().alias("Frames with regions"),
        pl.col("clip").n_unique().alias("Clips with regions"),
        pl.col("area").median().alias("Median region area px²"),
        pl.col("area").quantile(0.95).alias("Region area p95 px²"),
    ))
    attribute_counts = pl.concat([
        objects.group_by("split", attribute).len(name="count").rename({attribute: "code"})
        .with_columns(pl.col("code").cast(pl.String).fill_null("Missing"),
                      pl.lit(attribute.capitalize()).alias("attribute"))
        for attribute in ("occlusion", "truncation")
    ]).with_columns((pl.col("count") / pl.col("count").sum().over("split", "attribute")).alias("fraction"))
    display(attribute_counts.sort("attribute", "split", "code"))
    px.bar(attribute_counts, x="code", y="fraction", color="split", facet_col="attribute", barmode="group",
           labels={"code": "Raw annotation code", "fraction": "Target-object fraction", "split": "Split"},
           title="Recorded occlusion and truncation codes").show()
    """),
    _markdown("""
    ## 8. Video structure and annotation agreement

    We inspect clip lengths and adjacent frames, then compare sampled YOLO labels with COCO
    using full split, clip, and frame identities. Equal frame filenames in different clips
    are not the same image. Text rounding allows a tolerance of 0.01 original-image pixels.
    """),
    _code("""
    clip_structure = frames.group_by("split", "clip").agg(
        pl.len().alias("Frames"), pl.col("frame_id").min().alias("First frame"),
        pl.col("frame_id").max().alias("Last frame"),
        pl.col("frame_id").n_unique().alias("Unique frame IDs"),
    ).with_columns((pl.col("Last frame") - pl.col("First frame") + 1 - pl.col("Unique frame IDs"))
                   .alias("Frame ID gaps"))
    display(clip_structure.group_by("split").agg(
        pl.col("Frames").min().alias("Shortest clip"), pl.col("Frames").median().alias("Median clip length"),
        pl.col("Frames").max().alias("Longest clip"), pl.col("Frame ID gaps").sum().alias("Frame ID gaps"),
    ))
    display(clip_structure.filter((pl.col("Frame ID gaps") > 0) | (pl.col("Frames") != pl.col("Unique frame IDs"))).head(8))
    px.scatter(clip_density, x="Frames", y="Mean targets", color="split", size="Targets p95", hover_name="clip",
               labels={"Frames": "Frames per clip", "Mean targets": "Mean targets per frame", "split": "Split"},
               title="Clip length and scene density").show()
    ordered_frames = frames.filter(pl.col("split") == "train").sort("clip", "frame_id")
    consecutive_starts = ordered_frames.filter(
        pl.col("frame_id").shift(-3).over("clip") == pl.col("frame_id") + 3
    )
    if not consecutive_starts.is_empty():
        start = consecutive_starts.sample(n=1, seed=SEED).row(0, named=True)
        temporal_frames = ordered_frames.filter(
            (pl.col("clip") == start["clip"]) & pl.col("frame_id").is_between(start["frame_id"], start["frame_id"] + 3)
        )
    else:
        clip = ordered_frames["clip"][0]
        temporal_frames = ordered_frames.filter(pl.col("clip") == clip).head(4)
    temporal_panels = []
    for record in temporal_frames.iter_rows(named=True):
        with Image.open(dataset_root / record["relative_path"]) as source:
            panel = source.convert("RGB")
            panel.thumbnail((400, 260))
        labeled_panel = Image.new("RGB", (400, 285), "white")
        labeled_panel.paste(panel, (0, 25))
        ImageDraw.Draw(labeled_panel).text((5, 5), f"{record['clip']} / {record['frame']}", fill="black")
        temporal_panels.append(labeled_panel)
    temporal_grid = Image.new("RGB", (800, 570), "white")
    for index, panel in enumerate(temporal_panels):
        temporal_grid.paste(panel, ((index % 2) * 400, (index // 2) * 285))
    display_image(temporal_grid)
    display(temporal_frames.select("clip", "frame", "frame_id"))
    """),
    _code("""
    coordinate_checks = compare_yolo_coco(dataset_root, frames, objects, ignored_regions, sample_size=48, seed=SEED)
    display(coordinate_checks.head(8))
    display(coordinate_checks.select(pl.len().alias("Sampled frames"),
            (~pl.col("is_match")).sum().alias("Mismatched frames"),
            pl.col("max_error_px").max().alias("Maximum error px")))
    roundtrip_errors = []
    sampled_boxes = objects.sample(n=min(100, objects.height), seed=SEED)
    for dimensions, group in sampled_boxes.group_by("image_width", "image_height"):
        width, height = dimensions
        original = group.select("x", "y", "width", "height").to_numpy()
        recovered = yolo_to_xywh(xywh_to_yolo(original, width, height), width, height)
        np.testing.assert_allclose(recovered, original, atol=1e-8)
        roundtrip_errors.append(float(np.abs(original - recovered).max()))
    display(pl.DataFrame({"Maximum round-trip error px": [max(roundtrip_errors, default=0.0)]}))
    visual_key = coordinate_checks["image_key"][0]
    visual_record = frames.filter(pl.col("image_key") == visual_key).row(0, named=True)
    yolo_path = dataset_root / visual_record["split"] / "labels" / visual_record["clip"] / f"{visual_record['frame']}.txt"
    yolo_pixels = read_yolo_boxes(yolo_path, visual_record["image_width"], visual_record["image_height"])
    yolo_table = pl.DataFrame(yolo_pixels, schema=["class_id", "x", "y", "width", "height"], orient="row").with_columns(pl.col("class_id").cast(pl.Int64))
    display(Markdown(f"**Converted YOLO ground truth.** `{visual_key}`. Compare this view with the numeric COCO agreement check."))
    display_image(draw_frame(dataset_root / visual_record["relative_path"], yolo_table,
                             ignored_regions.filter(pl.col("image_key") == visual_key),
                             max_width=700, should_label=len(yolo_pixels) <= 60))
    """),
    _markdown("""
    ## 9. Preprocessing context

    The setup has not resized or modified raw data. The tables retain original image
    dimensions, pixel boxes, sequence identity, and ignored regions. Any later resize,
    padding, tiling, or augmentation must transform boxes explicitly and preserve the
    distinction between targets and ignored regions.

    We will treat input resolution, small-object sampling, and density-aware inference
    limits as hypotheses for later controlled experiments. No detector has been trained
    or evaluated here. Test metadata does not determine preprocessing or hyperparameters.
    """),
    _code("""
    small_fraction = objects.select((pl.col("size_group") == "Small").mean()).item()
    median_density = labeled_frames["target_count"].median()
    max_density = labeled_frames["target_count"].max()
    largest_class = overall_classes.sort("count", descending=True).row(0, named=True)
    display(Markdown(
        f"**Observation.** Small boxes account for {small_fraction:.1%} of target annotations. "
        f"Labeled frames have a median of {median_density:g} targets and a maximum of {max_density}. "
        f"The most frequent class is {largest_class['class_name']} at {largest_class['proportion']:.1%}. "
        "These are annotation statistics, with repeated appearances of tracked objects counted separately."
    ))
    display(Markdown(
        f"**Validation.** Train and validation share {len(overlapping_clips)} clip IDs. "
        "We preserve the official validation set and will group any later internal splits by clip. "
        f"All randomized examples use seed {SEED}."
    ))
    """),
]
_write("01_eda.ipynb", eda_cells)
