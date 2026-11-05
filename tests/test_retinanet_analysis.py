import json
from pathlib import Path

import plotly.io

from hm2.experiment_names import (
    EXPERIMENTS,
    describe_text,
    descriptive_name,
    experiment_label,
    result_directory,
    storage_id,
)
from hm2.retinanet_analysis import summarize_batch


def test_descriptive_names_preserve_checkpoint_paths(tmp_path: Path) -> None:
    """Resolve descriptive names without moving existing experiment checkpoints."""
    for name in EXPERIMENTS:
        legacy = tmp_path / storage_id(name)
        legacy.mkdir()
        (legacy / "latest.pt").write_bytes(b"existing checkpoint")
        assert descriptive_name(storage_id(name)) == name
        assert result_directory(tmp_path, name) == legacy
        assert (
            result_directory(tmp_path, name) / "latest.pt"
        ).read_bytes() == b"existing checkpoint"
        assert describe_text(storage_id(name)) == experiment_label(name)
    assert result_directory(tmp_path, "diou_regression") == tmp_path / "diou_regression"
    assert describe_text("R1-S minus R0-S") == (
        "RetinaNet with smaller anchors (every fifth frame) minus "
        "RetinaNet with standard anchors (every fifth frame)"
    )


def test_partial_batch_preserves_missing_metrics(tmp_path: Path) -> None:
    """Keep absent results null and calculate only measured fixture contrasts."""
    for run, score in [("R0-S", 0.2), ("R1-S", 0.3)]:
        directory = tmp_path / run
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "experiment_id": run,
                    "variant": "R0" if run == "R0-S" else "R1",
                    "optimizer": {"lr": 0.0025},
                    "loss": {"regression": "giou"},
                    "anchor_sizes": [[32, 64], [128, 256]],
                }
            )
        )
        (directory / "status.json").write_text(json.dumps({"completed": run == "R0-S"}))
        (directory / "metrics.json").write_text(
            json.dumps([{"epoch": 1, "metrics": {"map": score, "per_class_ap": {"car": score}}}])
        )
    summary = summarize_batch(tmp_path)
    assert summary["comparison"][0]["experiment_id"] == "standard_anchors_stride5"
    assert summary["comparison"][1]["variant"] == "smaller_anchors"
    assert summary["differences"]["anchors_stride"]["baseline"] == "standard_anchors_stride5"
    assert '"R0-S"' not in (tmp_path / "comparison.json").read_text()
    assert summary["comparison"][2]["status"] == "Not run"
    assert summary["comparison"][2]["map"] is None
    assert abs(summary["differences"]["anchors_stride"]["map"] - 0.1) < 1e-12
    assert summary["differences"]["anchors_stride"]["completed_pair"] is False
    assert summary["differences"]["anchors_full"]["map"] is None
    assert (tmp_path / "comparison.csv").exists()
    for chart in summary["charts"]:
        figure = plotly.io.read_json(Path(chart).with_suffix(".json"))
        bundle = plotly.io.renderers["plotly_mimetype"].to_mimebundle(figure.to_dict())
        assert "application/vnd.plotly.v1+json" in bundle
        assert len(json.dumps(bundle).encode()) < 100_000
    metric_figure = plotly.io.read_json(tmp_path / "metrics.json")
    assert metric_figure.data[0].name == experiment_label("standard_anchors_stride5")
    assert metric_figure.data[0].x[0] == "map"
