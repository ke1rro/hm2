import json
import logging
from pathlib import Path

import numpy as np
import plotly.express as px
import polars as pl

from hm2.experiment_names import (
    EXPERIMENTS,
    describe_text,
    descriptive_name,
    experiment_label,
    result_directory,
)
from hm2.visualization import draw_prediction_pair

LOGGER = logging.getLogger(__name__)
RUN_ORDER = tuple(EXPERIMENTS)
METRICS = ("map", "ap50", "ap75", "ap_small", "ap_medium", "ap_large")
CONTRASTS = {
    "anchors_stride": ("smaller_anchors_stride5", "standard_anchors_stride5"),
    "anchors_full": ("smaller_anchors_full", "standard_anchors_full"),
    "full_minus_stride_baseline": ("standard_anchors_full", "standard_anchors_stride5"),
    "full_minus_stride_tuned": ("smaller_anchors_full", "smaller_anchors_stride5"),
}


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _charts(batch_dir: Path, rows: list[dict], histories: list[dict]) -> list[str]:
    paths = []
    labels = {name: experiment_label(name) for name in RUN_ORDER}
    table = pl.from_dicts(rows, infer_schema_length=None).with_columns(
        pl.col("experiment_id").replace_strict(labels)
    )
    specifications = [
        ("metrics", table.unpivot(on=list(METRICS), index="experiment_id"), "variable", "value"),
        ("ap_small", table, "experiment_id", "ap_small"),
        ("time", table, "experiment_id", "train_seconds"),
    ]
    per_class = [
        {"experiment_id": experiment_label(row["experiment_id"]), "Class": name, "AP": value}
        for row in rows
        for name, value in row["per_class_ap"].items()
    ]
    if per_class:
        specifications.append(("per_class", pl.from_dicts(per_class), "Class", "AP"))
    for name, values, x, y in specifications:
        figure = px.bar(
            values.filter(pl.col(y).is_not_null()),
            x=x,
            y=y,
            color="experiment_id",
            barmode="group",
            title=name.replace("_", " ").capitalize(),
            labels={
                x: x.replace("_", " ").capitalize(),
                y: y.replace("_", " ").capitalize(),
                "experiment_id": "Experiment",
            },
        )
        path = batch_dir / f"{name}.html"
        figure.write_html(path, include_plotlyjs=True)
        figure.write_json(path.with_suffix(".json"))
        paths.append(str(path))
    if histories:
        history = pl.from_dicts(histories, infer_schema_length=None).with_columns(
            pl.col("experiment_id").replace_strict(labels)
        )
        for column in ("loss", "learning_rate", "classification", "bbox_regression"):
            if column not in history.columns:
                continue
            figure = px.line(
                history,
                x="global_step",
                y=column,
                color="experiment_id",
                title=f'Training {column.replace("_", " ")}',
                labels={
                    "global_step": "Optimizer step",
                    column: column.capitalize(),
                    "experiment_id": "Experiment",
                },
            )
            path = batch_dir / f"{column}.html"
            figure.write_html(path, include_plotlyjs=True)
            figure.write_json(path.with_suffix(".json"))
            paths.append(str(path))
    return paths


def _missed(objects: pl.DataFrame, predictions: list[dict]) -> int:
    boxes = objects.select("x", "y", "width", "height").to_numpy().astype(float)
    if not len(boxes):
        return 0
    if not predictions:
        return len(boxes)
    predicted = np.array([record["bbox"] for record in predictions], dtype=float)
    right = np.minimum(
        boxes[:, None, :2] + boxes[:, None, 2:], predicted[None, :, :2] + predicted[None, :, 2:]
    )
    left = np.maximum(boxes[:, None, :2], predicted[None, :, :2])
    intersection = np.maximum(right - left, 0).prod(axis=2)
    union = boxes[:, 2:].prod(axis=1)[:, None] + predicted[:, 2:].prod(axis=1) - intersection
    quality = intersection / np.maximum(union, 1e-12)
    labels = np.array([record["category_id"] for record in predictions])
    quality *= objects["category_id"].to_numpy()[:, None] == labels[None, :]
    matched = np.zeros(len(boxes), dtype=bool)
    for index in np.argsort([-record["score"] for record in predictions]):
        eligible = quality[:, index].copy()
        eligible[matched] = -1
        best = int(eligible.argmax())
        if eligible[best] >= 0.5:
            matched[best] = True
    return int((~matched).sum())


def _previews(batch_dir: Path, dataset_root: Path, metadata: Path) -> list[dict]:
    frames = pl.read_parquet(metadata / "frames.parquet").filter(pl.col("split") == "val")
    objects = pl.read_parquet(metadata / "objects.parquet").filter(pl.col("split") == "val")
    ignored = pl.read_parquet(metadata / "ignored_regions.parquet").filter(pl.col("split") == "val")
    objects_by_key = objects.partition_by("image_key", as_dict=True)
    ignored_by_key = ignored.partition_by("image_key", as_dict=True)
    predictions_by_run = {}
    for run in RUN_ORDER:
        path = result_directory(batch_dir, run) / "predictions.json"
        if path.exists():
            indexed = {}
            for prediction in _read(path):
                if prediction["score"] >= 0.3:
                    indexed.setdefault(int(prediction["image_id"]), []).append(prediction)
            predictions_by_run[run] = indexed
    if not predictions_by_run or frames.is_empty() or objects.is_empty():
        return []
    frame_lookup = {row["image_key"]: row for row in frames.iter_rows(named=True)}
    selection_path = batch_dir / "gallery_selection.json"
    if selection_path.exists():
        choices = json.loads(selection_path.read_text())
    else:
        reference = next(iter(predictions_by_run))
        candidates = (
            objects.group_by("image_key")
            .agg(pl.len().alias("targets"), pl.col("area").min().alias("min_area"))
            .sort("image_key")
        )
        choices = [
            ("Dense scene", candidates.sort("targets", descending=True)["image_key"][0]),
            ("Tiny objects", candidates.sort("min_area")["image_key"][0]),
        ]
        missed = [
            (
                _missed(
                    targets,
                    predictions_by_run[reference].get(int(frame_lookup[key[0]]["image_id"]), []),
                ),
                key[0],
            )
            for key, targets in objects_by_key.items()
        ]
        count, key = max(missed)
        if count > 0:
            choices.append((f"Missed GT in {experiment_label(reference)} at IoU 0.5", key))
        _save(selection_path, choices)
    directory = batch_dir / "previews"
    directory.mkdir(exist_ok=True)
    records = []
    for index, (reason, key) in enumerate(choices):
        row = frame_lookup[key]
        for run, indexed in predictions_by_run.items():
            predictions = indexed.get(int(row["image_id"]), [])
            predicted = pl.DataFrame(
                {
                    "class_id": [record["category_id"] - 1 for record in predictions],
                    **{
                        name: [record["bbox"][position] for record in predictions]
                        for position, name in enumerate(("x", "y", "width", "height"))
                    },
                }
            )
            path = directory / f"{index}_{run}.jpg"
            draw_prediction_pair(
                dataset_root / row["relative_path"],
                objects_by_key[(key,)],
                ignored_by_key.get((key,), ignored.head(0)),
                predicted,
            ).save(path, quality=80)
            records.append(
                {
                    "experiment_id": run,
                    "image_key": key,
                    "reason": describe_text(reason),
                    "path": str(path),
                    "score_threshold": 0.3,
                }
            )
    _save(batch_dir / "preview_selection.json", records)
    return records


def summarize_batch(
    batch_dir: Path, dataset_root: Path | None = None, metadata: Path | None = None
) -> dict:
    """Aggregate measured experiment outputs without executing a detector.

    Args:
        batch_dir: Batch directory containing the four experiment directories.
        dataset_root: Optional original image directory for diagnostic previews.
        metadata: Optional directory containing the three raw metadata tables.

    Returns:
        Comparison rows, four measured contrasts, chart paths, and optional preview records.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    rows, histories, epoch_metrics = [], [], []
    for run in RUN_ORDER:
        directory = result_directory(batch_dir, run)
        status = _read(directory / "status.json")
        config = _read(directory / "config.json")
        if not isinstance(status, dict) or not isinstance(config, dict):
            raise ValueError(f"Expected status and configuration objects in {directory}.")
        metric_records = _read(directory / "metrics.json")
        latest = metric_records[-1] if isinstance(metric_records, list) and metric_records else {}
        metrics = latest.get("metrics", metric_records if isinstance(metric_records, dict) else {})
        inference = latest.get("inference", {})
        if isinstance(metric_records, list):
            epoch_metrics.extend(
                {"experiment_id": run, "epoch": record["epoch"], metric: value}
                for record in metric_records
                for metric, value in record["metrics"].items()
                if metric in METRICS
            )
        row = {
            "experiment_id": run,
            "status": (
                "Completed" if status.get("completed") else ("Partial" if status else "Not run")
            ),
            **config,
            **{name: metrics.get(name) for name in METRICS},
            "per_class_ap": metrics.get("per_class_ap", {}),
            **{
                name: status.get(name)
                for name in (
                    "epoch",
                    "global_step",
                    "image_exposures",
                    "train_seconds",
                    "train_frames",
                )
            },
            "epoch": latest.get("epoch", status.get("epochs_completed")),
            "train_seconds": status.get("wall_train_seconds"),
            "inference_seconds": (
                inference["milliseconds_per_frame"] * inference["image_count"] / 1000
                if "milliseconds_per_frame" in inference
                else None
            ),
            "fps": inference.get("frames_per_second"),
            "validation_protocol": metrics.get("metric"),
            "ignore_policy": metrics.get("ignore_policy"),
            "validation_images": metrics.get("image_count"),
        }
        row["experiment_id"] = run
        if "variant" in config:
            row["variant"] = descriptive_name(config["variant"])
        rows.append(row)
        history = _read(directory / "history.json")
        if isinstance(history, list):
            histories.extend(
                {
                    "attempt": record.get("attempt", 0),
                    "global_step": record.get("step"),
                    **record,
                    "experiment_id": run,
                }
                for record in history
            )
    lookup = {row["experiment_id"]: row for row in rows}
    differences = {}
    for name, (changed, baseline) in CONTRASTS.items():
        first, second = lookup[changed], lookup[baseline]
        values = {
            metric: (
                first[metric] - second[metric]
                if first[metric] is not None and second[metric] is not None
                else None
            )
            for metric in (*METRICS, "train_seconds", "inference_seconds", "fps")
        }
        values["per_class_ap"] = {
            label: (
                first["per_class_ap"][label] - second["per_class_ap"][label]
                if first["per_class_ap"].get(label) is not None
                and second["per_class_ap"].get(label) is not None
                else None
            )
            for label in first["per_class_ap"].keys() | second["per_class_ap"].keys()
        }
        differences[name] = {
            "changed": changed,
            "baseline": baseline,
            "completed_pair": first["status"] == second["status"] == "Completed",
            **values,
        }
    _save(batch_dir / "comparison.json", rows)
    pl.from_dicts(
        [
            {
                **{
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                    if key != "per_class_ap"
                },
                **{f"ap_{name}": value for name, value in row["per_class_ap"].items()},
            }
            for row in rows
        ],
        infer_schema_length=None,
    ).write_csv(batch_dir / "comparison.csv")
    _save(batch_dir / "differences.json", differences)
    charts = _charts(batch_dir, rows, histories)
    if epoch_metrics:
        values = (
            pl.from_dicts(epoch_metrics, infer_schema_length=None)
            .with_columns(
                pl.col("experiment_id").replace_strict(
                    {name: experiment_label(name) for name in RUN_ORDER}
                )
            )
            .unpivot(
                on=[name for name in METRICS if any(name in row for row in epoch_metrics)],
                index=["experiment_id", "epoch"],
            )
            .filter(pl.col("value").is_not_null())
        )
        figure = px.line(
            values,
            x="epoch",
            y="value",
            color="experiment_id",
            facet_col="variable",
            markers=True,
            title="Validation metrics by completed epoch",
            labels={
                "epoch": "Completed epoch",
                "value": "AP",
                "experiment_id": "Experiment",
                "variable": "Metric",
            },
        )
        path = batch_dir / "validation_curves.html"
        figure.write_html(path, include_plotlyjs=True)
        figure.write_json(path.with_suffix(".json"))
        charts.append(str(path))
    previews = (
        _previews(batch_dir, dataset_root, metadata)
        if (dataset_root is not None and metadata is not None)
        else []
    )
    LOGGER.info("Saved experiment comparisons in %s", batch_dir)
    return {"comparison": rows, "differences": differences, "charts": charts, "previews": previews}
