import tomllib
from pathlib import Path

import polars as pl
import pytest
import torch
from torch import nn

from hm2.drive_api import DriveApi
from hm2.rt_detr import GradientClippedAdamW
from hm2.rt_detr_experiments import (
    RT_DETR_MODULES,
    RUN_ORDER,
    build_rt_detr_experiment_config,
    describe_query_step,
    download_committed_files,
    experiment_label,
    find_drive_folder,
    list_experiments,
    load_rt_detr_settings,
    summarize_object_sizes,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/rt_detr_baseline.toml"
B1_CONFIG = ROOT / "config/faster_rcnn_baseline.toml"


def test_configuration_lists_the_experiments_in_the_agreed_order() -> None:
    """Catch a run order that differs from baseline, NWD, queries, denoising."""
    assert list_experiments(CONFIG) == list(RUN_ORDER)
    assert all(experiment_label(name) != name for name in list_experiments(CONFIG))


def test_each_experiment_changes_exactly_its_own_setting() -> None:
    """Catch an experiment that silently changes more than one factor against the baseline."""
    baseline = load_rt_detr_settings(CONFIG, "baseline_q300")
    changes = {
        "nwd_loss": ("loss", "box_loss", "nwd"),
        "queries_500": ("model", "num_queries", 500),
        "denoising_300": ("model", "num_denoising", 300),
    }
    for name, (table, key, value) in changes.items():
        settings = load_rt_detr_settings(CONFIG, name)
        assert settings[table][key] == value
        differences = {
            (section, option)
            for section in baseline
            for option in baseline[section]
            if baseline[section][option] != settings[section][option]
        }
        assert differences == {(table, key)}


def test_unknown_experiment_or_override_key_fails(tmp_path: Path) -> None:
    """Catch a misspelled override that would otherwise be ignored."""
    with pytest.raises(ValueError, match="Unknown experiment"):
        load_rt_detr_settings(CONFIG, "missing")
    broken = tmp_path / "broken.toml"
    broken.write_text(CONFIG.read_text() + "\n[experiments.typo.model]\nnum_querys = 500\n")
    with pytest.raises(ValueError, match="num_querys"):
        load_rt_detr_settings(broken, "typo")


def test_recipe_keeps_the_b1_protocol() -> None:
    """Catch a data, resize, schedule-length or evaluation difference from the B1 run."""
    settings = load_rt_detr_settings(CONFIG, "baseline_q300")
    b1 = tomllib.loads(B1_CONFIG.read_text())
    assert settings["dataset"] == b1["dataset"]
    assert settings["evaluation"] == b1["evaluation"]
    assert settings["probe"] == b1["probe"]
    assert settings["export"] == b1["export"]
    for key in ("seed", "epochs", "batch_size", "flip_probability", "checkpoint_seconds"):
        assert settings["training"][key] == b1["training"][key]
    for key in ("min_size", "max_size", "detections_per_img", "ignore_threshold"):
        assert settings["model"][key] == b1["model"][key]


def test_experiment_config_records_everything_that_changes_a_run() -> None:
    """Catch a changed recipe resuming from another experiment's checkpoint."""
    keys = ["train/clip/0000001"]
    configs = {
        name: build_rt_detr_experiment_config(
            load_rt_detr_settings(CONFIG, name), name, keys, "digest"
        )
        for name in list_experiments(CONFIG)
    }
    serialized = {str(sorted(config.items())) for config in configs.values()}
    assert len(serialized) == len(configs)
    baseline = configs["baseline_q300"]
    assert baseline["checkpoint_selection"] == "final epoch, never chosen on validation"
    assert baseline["model"]["model_id"] == "PekingU/rtdetr_v2_r50vd"
    assert baseline["selected_frame_count"] == 1
    assert "hm2.rt_detr" in RT_DETR_MODULES and "hm2.faster_rcnn" in RT_DETR_MODULES


class _DiagnosedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_diagnostics: dict[str, float] = {
            "matched_queries": 3.0,
            "excluded_queries_final": 1.0,
        }


def test_step_description_adds_diagnostics_and_gradient_norm() -> None:
    """Catch history rows that lose the matching counts or the gradient norm."""
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = GradientClippedAdamW([parameter], max_norm=0.1)
    optimizer.last_gradient_norm = 2.5
    assert describe_query_step(_DiagnosedModel(), optimizer) == {
        "matched_queries": 3.0,
        "excluded_queries_final": 1.0,
        "gradient_norm": 2.5,
    }


def test_object_sizes_are_measured_in_model_pixels() -> None:
    """Catch an NWD constant check that measures original pixels instead of model pixels."""
    selected = pl.DataFrame(
        {"image_key": ["a", "b"], "image_width": [1360, 1920], "image_height": [765, 1080]}
    )
    objects = pl.DataFrame(
        {"image_key": ["a", "b", "c"], "width": [40.0, 60.0, 1.0], "height": [40.0, 60.0, 1.0]}
    )
    sizes = summarize_object_sizes(selected, objects, {"min_size": 640, "max_size": 1024})
    assert sizes["objects"] == 2
    assert sizes["mean"] == pytest.approx((40 * 1024 / 1360 + 60 * 1024 / 1920) / 2)


class _FakeDrive(DriveApi):
    def __init__(self) -> None:
        self.downloads: list[tuple[str, Path, str]] = []
        self.folders = {
            "root": {"batches": {"id": "batches"}},
            "batches": {"run": {"id": "run"}},
            "run": {"CURRENT.json": {"id": "pointer"}, "generations": {"id": "generations"}},
            "generations": {"7": {"id": "generation"}},
            "generation": {"metrics.json": {"id": "metrics"}, "latest.pt": {"id": "weights"}},
        }

    def children(self, parent: str) -> dict:
        return self.folders.get(parent, {})

    def read_json(self, file_id: str) -> dict:
        return {"generation": "7", "files": {"metrics.json": "abc", "latest.pt": "def"}}

    def download(self, file_id: str, destination: Path, sha256: str | None = None) -> None:
        self.downloads.append((file_id, destination, sha256 or ""))


def test_drive_records_are_read_from_the_committed_generation(tmp_path: Path) -> None:
    """Catch reading a stale generation, fetching weights, or creating teammate folders."""
    drive = _FakeDrive()
    folder = find_drive_folder(drive, "root", ["batches", "run"])
    assert folder == "run"
    assert find_drive_folder(drive, "root", ["batches", "missing"]) is None
    names = download_committed_files(drive, "run", ("metrics.json", "history.json"), tmp_path)
    assert names == ["metrics.json"]
    assert drive.downloads == [("metrics", tmp_path / "metrics.json", "abc")]
    assert download_committed_files(drive, "batches", ("metrics.json",), tmp_path) == []
