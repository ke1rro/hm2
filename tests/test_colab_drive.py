import json
from pathlib import Path

import pytest
import torch

from hm2 import colab_drive
from hm2.experiment_checkpoint import publish_snapshot, save_checkpoint
from hm2.retinanet import make_optimizer


def _snapshot(directory: Path, step: int, experiment: str = "R0-S") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model = torch.nn.Linear(2, 1)
    optimizer, scheduler = make_optimizer(model, 0.01, 4, 1)
    save_checkpoint(
        directory / "latest.pt",
        model,
        optimizer,
        scheduler,
        step,
        {"experiment_id": experiment},
        extra_state={"epochs_completed": step, "batch_cursor": 0, "global_step": step},
    )
    (directory / "status.json").write_text(json.dumps({"epochs_completed": step}))
    publish_snapshot(directory)


def test_drive_checkpoint_round_trip_and_compatibility(tmp_path: Path) -> None:
    """Restore the last saved training state and reject another experiment's checkpoint."""
    source, drive, restored = (tmp_path / name for name in ("source", "drive", "restored"))
    _snapshot(source, 1)
    colab_drive.sync_checkpoint_to_drive(source, drive)
    progress = colab_drive.restore_checkpoint_from_drive(drive, restored, {"experiment_id": "R0-S"})
    assert progress["global_step"] == 1
    state = torch.load(restored / "latest.pt", weights_only=True)
    assert state["step"] == 1
    assert state["extra_state"]["epochs_completed"] == 1
    with pytest.raises(ValueError, match="configuration"):
        colab_drive.restore_checkpoint_from_drive(
            drive, tmp_path / "wrong", {"experiment_id": "R1-S"}
        )
    assert not (tmp_path / "wrong/latest.pt").exists()
    _snapshot(restored, 2)
    progress = colab_drive.restore_checkpoint_from_drive(drive, restored, {"experiment_id": "R0-S"})
    assert progress["global_step"] == 2
    assert (
        colab_drive.restore_checkpoint_from_drive(
            drive, tmp_path / "fresh_runtime", {"experiment_id": "R0-S"}
        )["global_step"]
        == 2
    )
    assert colab_drive.verify_drive_checkpoint(tmp_path / "probe", tmp_path / "probe_drive")[
        "verified"
    ]


def test_failed_drive_copy_retains_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave the committed Drive generation unchanged if the replacement is corrupt."""
    source, drive = tmp_path / "source", tmp_path / "drive"
    _snapshot(source, 1)
    colab_drive.sync_checkpoint_to_drive(source, drive)
    previous = (drive / "CURRENT.json").read_text()
    interrupted = drive / "generations/interrupted"
    interrupted.mkdir()
    (interrupted / "latest.pt.incoming").write_bytes(b"abandoned transfer")
    _snapshot(source, 2)
    original = colab_drive.shutil.copyfile

    def corrupt_copy(source: Path, destination: Path) -> Path:
        result = original(source, destination)
        if source.name == "latest.pt":
            Path(destination).write_bytes(b"corrupt transfer")
        return result

    monkeypatch.setattr(colab_drive.shutil, "copyfile", corrupt_copy)
    with pytest.raises(ValueError, match="checksum"):
        colab_drive.sync_checkpoint_to_drive(source, drive)
    assert (drive / "CURRENT.json").read_text() == previous
    assert len(list((drive / "generations").iterdir())) == 1
    monkeypatch.undo()
    progress = colab_drive.restore_checkpoint_from_drive(
        drive, tmp_path / "restored", {"experiment_id": "R0-S"}
    )
    assert progress["global_step"] == 1


def test_completed_local_evaluation_is_not_rolled_back(tmp_path: Path) -> None:
    """Preserve newly evaluated metrics when Drive still has the same pre-evaluation step."""
    local, drive = tmp_path / "local", tmp_path / "drive"
    _snapshot(local, 1)
    path = local / "latest.pt"
    state = torch.load(path, weights_only=True)
    state["extra_state"]["pending_evaluation"] = True
    state["extra_state"]["metrics"] = []
    torch.save(state, path)
    publish_snapshot(local)
    colab_drive.sync_checkpoint_to_drive(local, drive)
    state["extra_state"]["pending_evaluation"] = False
    state["extra_state"]["metrics"] = [{"epoch": 1, "metrics": {"map": 0.1}}]
    temporary = local / "checkpoint.incoming"
    torch.save(state, temporary)
    temporary.replace(path)
    publish_snapshot(local)
    progress = colab_drive.restore_checkpoint_from_drive(drive, local, {"experiment_id": "R0-S"})
    assert not progress["pending_evaluation"]
    assert progress["metrics"][0]["metrics"]["map"] == 0.1
