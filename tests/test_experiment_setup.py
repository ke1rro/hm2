import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from hm2.experiment_setup import save_experiment_config, stage_dataset


def test_staging_reuses_extracted_inputs_and_keeps_source_archive(tmp_path: Path) -> None:
    """Reuse an extracted dataset without requiring another transfer or changing inputs."""
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("wrapper/train/images/clip/frame.jpg", b"image")
        bundle.writestr("wrapper/val/annotations_coco.json", "{}")
    destination = tmp_path / "dataset"
    local_archive = tmp_path / "staging.zip"
    root = stage_dataset(archive, destination, local_archive, reserve_bytes=0)
    assert root == destination / "wrapper"
    assert archive.is_file()
    assert not local_archive.exists()
    frame = root / "train/images/clip/frame.jpg"
    modified = frame.stat().st_mtime_ns
    assert stage_dataset(tmp_path / "missing.zip", destination) == root
    assert frame.stat().st_mtime_ns == modified


def test_existing_experiment_config_is_checked_before_replacement(tmp_path: Path) -> None:
    """Prevent a notebook restart from replacing the original scientific recipe."""
    path = tmp_path / "batch_config.json"
    recipe = [{"experiment_id": "R0-S", "epochs": 5, "seed": 2026}]
    save_experiment_config(path, recipe)
    save_experiment_config(path, recipe)
    with pytest.raises(ValueError, match="configuration"):
        save_experiment_config(path, [{"experiment_id": "R0-S", "epochs": 3}])
    assert json.loads(path.read_text()) == recipe
