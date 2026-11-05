import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
from googleapiclient.discovery import build
from googleapiclient.http import HttpMockSequence

from hm2 import drive_api
from hm2.experiment_checkpoint import publish_snapshot, save_checkpoint
from hm2.retinanet import make_optimizer


def _snapshot(directory: Path, epoch: int, experiment: str) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer, scheduler = make_optimizer(model, 0.01, 10, 1)
    model(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    scheduler.step()
    save_checkpoint(
        directory / "latest.pt",
        model,
        optimizer,
        scheduler,
        epoch,
        {"experiment_id": experiment},
        extra_state={"epochs_completed": epoch, "batch_cursor": 0, "global_step": epoch},
    )
    publish_snapshot(directory)


class LocalDrive(drive_api.DriveApi):
    """Filesystem transport substitute for tests of real snapshot transactions."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.should_fail = False

    def children(self, parent: str) -> dict[str, dict]:
        """List local file metadata using Drive's binary checksum field."""
        result = {}
        for path in Path(parent).iterdir():
            result[path.name] = {"id": str(path), "name": path.name}
        return result

    def folder(self, parent: str, name: str) -> str:
        """Create or return a local folder."""
        path = Path(parent) / name
        path.mkdir(exist_ok=True)
        return str(path)

    def read_json(self, file_id: str) -> dict:
        """Read a JSON object from the transport."""
        return json.loads(Path(file_id).read_text())

    def upload(self, source: Path, parent: str) -> dict:
        """Copy a completed object or simulate a failed checkpoint transfer."""
        if self.should_fail and source.name == "latest.pt":
            raise OSError("interrupted upload")
        destination = Path(parent) / source.name
        shutil.copyfile(source, destination)
        return {"id": str(destination)}

    def download(self, file_id: str, destination: Path, sha256: str | None = None) -> None:
        """Copy an object through the production integrity check."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file_id, destination)
        drive_api.verify_checksum(
            destination,
            {
                "sha256Checksum": sha256,
                "md5Checksum": hashlib.md5(Path(file_id).read_bytes()).hexdigest(),
            },
        )

    def copy(self, file_id: str, parent: str, name: str) -> dict:
        """Copy an unchanged object without uploading its contents again."""
        destination = Path(parent) / name
        shutil.copyfile(file_id, destination)
        return {"sha256Checksum": hashlib.sha256(destination.read_bytes()).hexdigest()}

    def discard(self, file_id: str) -> None:
        """Remove an expired test generation."""
        shutil.rmtree(file_id)


def test_api_snapshot_resume_and_failed_replacement(tmp_path: Path) -> None:
    """Keep the committed state resumable when the next remote upload fails."""
    source = tmp_path / "source"
    remote = tmp_path / "remote"
    remote.mkdir()
    api = LocalDrive(remote)
    _snapshot(source, 3, "R0-F")
    manifest = drive_api.sync_checkpoint_to_drive_api(api, source, str(remote))
    restored = tmp_path / "restored"
    progress = drive_api.restore_checkpoint_from_drive_api(
        api, str(remote), restored, {"experiment_id": "R0-F"}, tmp_path / "cache"
    )
    assert progress["epochs_completed"] == 3
    assert torch.load(restored / "latest.pt", weights_only=True)["step"] == 3
    _snapshot(source, 4, "R0-F")
    api.should_fail = True
    with pytest.raises(OSError, match="interrupted"):
        drive_api.sync_checkpoint_to_drive_api(api, source, str(remote))
    assert json.loads((remote / "CURRENT.json").read_text()) == manifest
    api.should_fail = False
    drive_api.sync_checkpoint_to_drive_api(api, source, str(remote))
    latest = drive_api.restore_checkpoint_from_drive_api(
        api, str(remote), tmp_path / "restarted", {"experiment_id": "R0-F"}, tmp_path / "cache2"
    )
    assert latest["epochs_completed"] == 4
    assert drive_api.verify_drive_api_checkpoint(api, tmp_path / "probe", str(remote))["verified"]
    with pytest.raises(ValueError, match="configuration"):
        drive_api.restore_checkpoint_from_drive_api(
            api, str(remote), tmp_path / "wrong", {"experiment_id": "R1-F"}, tmp_path / "cache3"
        )


def test_remote_checksum_rejects_corruption(tmp_path: Path) -> None:
    """Reject a remote object's mismatch before promoting a checkpoint pointer."""
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"valid checkpoint")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    drive_api.verify_checksum(path, {"sha256Checksum": checksum})
    path.write_bytes(b"corrupt checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        drive_api.verify_checksum(path, {"sha256Checksum": checksum})


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_api_restore_uses_previous_generation_and_persists_newer_local_state(
    tmp_path: Path,
    damage: str,
) -> None:
    """Recover a retained valid epoch and persist completed local work before it can be skipped."""
    source, remote = tmp_path / "source", tmp_path / "remote"
    remote.mkdir()
    api = LocalDrive(remote)
    _snapshot(source, 3, "R0-F")
    previous = drive_api.sync_checkpoint_to_drive_api(api, source, str(remote))
    _snapshot(source, 4, "R0-F")
    current = drive_api.sync_checkpoint_to_drive_api(api, source, str(remote))
    if damage == "corrupt":
        (remote / "generations" / current["generation"] / "latest.pt").write_bytes(b"corrupt")
    else:
        shutil.rmtree(remote / "generations" / current["generation"])
    restored = tmp_path / "restored"
    progress = drive_api.restore_checkpoint_from_drive_api(
        api, str(remote), restored, {"experiment_id": "R0-F"}, tmp_path / "cache"
    )
    assert progress["epochs_completed"] == 3
    assert (remote / "generations" / previous["generation"] / "latest.pt").exists()
    _snapshot(restored, 5, "R0-F")
    progress = drive_api.restore_checkpoint_from_drive_api(
        api, str(remote), restored, {"experiment_id": "R0-F"}, tmp_path / "cache"
    )
    assert progress["epochs_completed"] == 5
    final = drive_api.restore_checkpoint_from_drive_api(
        api, str(remote), tmp_path / "fresh", {"experiment_id": "R0-F"}, tmp_path / "fresh_cache"
    )
    assert final["epochs_completed"] == 5


def test_download_keeps_old_copy_when_transport_returns_corrupt_bytes(tmp_path: Path) -> None:
    """Exercise real Drive client download handling without replacing a good local checkpoint."""
    expected = b"new checkpoint"
    metadata = {
        "size": str(len(expected)),
        "md5Checksum": hashlib.md5(expected).hexdigest(),
    }
    http = HttpMockSequence(
        [
            ({"status": "200"}, json.dumps(metadata)),
            ({"status": "200", "content-length": str(len(expected))}, b"bad checkpoint"),
        ]
    )
    api = drive_api.DriveApi(build("drive", "v3", http=http, cache_discovery=False))
    destination = tmp_path / "latest.pt"
    destination.write_bytes(b"previous good checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        api.download("checkpoint-id", destination)
    assert destination.read_bytes() == b"previous good checkpoint"
