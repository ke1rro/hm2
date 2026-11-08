import hashlib
import json
import logging
import shutil
from pathlib import Path

import pytest
import torch
from googleapiclient.discovery import build
from googleapiclient.http import HttpMockSequence

from hm2 import drive_api
from hm2.experiment_checkpoint import publish_snapshot, save_checkpoint
from hm2.retinanet import make_optimizer
from hm2.yolo_pipeline import CheckpointMirror


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


@pytest.mark.parametrize(
    "notebook_name",
    [
        "02_faster_rcnn_experiments_kaggle.ipynb",
        "04_data_experiments_kaggle.ipynb",
    ],
)
def test_faster_rcnn_notebook_drive_setup_and_snapshot_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notebook_name: str
) -> None:
    """Execute the notebook's Drive setup with real checkpoint IO and a local transport."""
    from types import SimpleNamespace

    import nbformat
    import polars as pl

    remote = tmp_path / "remote"
    remote.mkdir()
    api = LocalDrive(remote)
    monkeypatch.setattr(drive_api, "DriveApi", lambda service: api)
    notebook = nbformat.read(
        Path(__file__).parents[1] / "experiments" / notebook_name,
        as_version=4,
    )
    source = notebook.cells[7].source.split("status_path =")[0]
    source = source.replace("/kaggle/temp/drive_cache", str(tmp_path / "cache"))
    frames = pl.DataFrame({"image_key": ["train/clip/0000001"]})
    settings = {"dataset": {}}
    config = {"experiment_id": "baseline_stride5"}
    output = tmp_path / "output"
    names = (
        ["photometric_augmentation", "repeat_factor_sampling", "input_resolution"]
        if notebook_name.startswith("04_")
        else ["baseline_stride5"]
    )
    outputs = {name: output / name for name in names}
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    namespace: dict = {
        "Path": Path,
        "json": json,
        "LOGGER": logging.getLogger(__name__),
        "verify_selection": lambda *args: None,
        "compute_source_identity": lambda *args: "selection-hash",
        "build_experiment_config": lambda settings, name, *args: {"experiment_id": name},
        "find_previous_output": lambda *args: None,
        "selected": frames,
        "validation": frames,
        "SETTINGS": settings,
        "EXPERIMENT": "baseline_stride5",
        "EXPERIMENTS": names,
        "SETTINGS_BY_EXPERIMENT": {name: settings for name in names},
        "OUTPUTS": outputs,
        "SOURCE_ROOT": tmp_path,
        "IMAGE_ROOT": tmp_path,
        "STAGE": "full",
        "IS_KAGGLE": True,
        "KAGGLE_INPUT": inputs,
        "BATCH_ID": "faster_rcnn_test",
        "OUTPUT_DIRECTORY": output,
        "OUTPUT": output,
        "SHOULD_TRAIN": not notebook_name.startswith("04_"),
        "SHOULD_EXPORT_SUBMISSIONS": notebook_name.startswith("04_"),
        "SHOULD_RUN_GPU_PROBE": False,
        "drive_root": str(remote),
        "service": SimpleNamespace(
            about=lambda: SimpleNamespace(
                get=lambda **kwargs: SimpleNamespace(execute=lambda **kwargs: {"storageQuota": {}})
            )
        ),
    }
    exec(compile(source, "notebook-drive-setup", "exec"), namespace)
    if notebook_name.startswith("04_"):
        assert not (remote / "batches/faster_rcnn_test/_probe_api").exists()
        assert namespace["RESTORED_PROGRESS"] == {name: {} for name in names}
        for epoch, name in enumerate(names, 1):
            _snapshot(outputs[name], epoch, name)
            folder = namespace["RUN_FOLDERS"][name]
            drive_api.sync_checkpoint_to_drive_api(api, outputs[name], folder)
            progress = drive_api.restore_checkpoint_from_drive_api(
                api,
                folder,
                tmp_path / "restored" / name,
                {"experiment_id": name},
                tmp_path / "restore-cache" / name,
            )
            assert progress["epochs_completed"] == epoch
        exec(compile(source, "notebook-drive-setup", "exec"), namespace)
        assert {
            name: progress["global_step"]
            for name, progress in namespace["RESTORED_PROGRESS"].items()
        } == {"photometric_augmentation": 1, "repeat_factor_sampling": 2, "input_resolution": 3}
    else:
        assert namespace["probe_result"]["verified"]
        assert namespace["progress"] == {}
        _snapshot(output, 3, "baseline_stride5")
        namespace["persist_checkpoint"](output)
        progress = drive_api.restore_checkpoint_from_drive_api(
            api, namespace["run_folder"], tmp_path / "restored", config, tmp_path / "restore-cache"
        )
        assert progress["epochs_completed"] == 3
        exec(compile(source, "notebook-drive-setup", "exec"), namespace)
        assert namespace["progress"]["global_step"] == 3


@pytest.mark.parametrize(
    "notebook_name,archive_name",
    [
        ("02_faster_rcnn_experiments_kaggle.ipynb", "faster_rcnn_epoch1_project.zip"),
        ("04_data_experiments_kaggle.ipynb", "controlled_data_submissions_project.zip"),
    ],
)
def test_faster_rcnn_notebook_bootstrap_downloads_into_a_new_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notebook_name: str, archive_name: str
) -> None:
    """Run the notebook bootstrap against simulated Google HTTP responses on an empty disk."""
    import ast
    import importlib
    import logging
    import sys
    from io import BytesIO
    from types import SimpleNamespace
    from zipfile import ZipFile

    import googleapiclient.discovery
    import nbformat

    contents = BytesIO()
    required = (
        "src/hm2/faster_rcnn.py",
        "src/hm2/faster_rcnn_experiments.py",
        "src/hm2/drive_api.py",
        "config/faster_rcnn_baseline.toml",
    )
    if notebook_name.startswith("04_"):
        required += (
            "config/data_photometric_augmentation.toml",
            "config/data_input_resolution.toml",
            "config/yolo_baseline.toml",
            "config/data_repeat_factor_sampling.toml",
            "experiments/protocol.py",
            "experiments/yolo_baseline.py",
            "experiments/photometric_augmentation.py",
            "experiments/input_resolution.py",
            "experiments/repeat_factor_sampling.py",
            "data/metadata/frames.parquet",
            "data/metadata/objects.parquet",
            "data/metadata/ignored_regions.parquet",
        )
    with ZipFile(contents, "w") as archive:
        for name in required:
            archive.writestr(name, b"test source")
    payload = contents.getvalue()
    metadata = {
        "name": archive_name,
        "id": "project",
        "size": str(len(payload)),
        "md5Checksum": hashlib.md5(payload).hexdigest(),
    }
    http = HttpMockSequence(
        [
            ({"status": "200"}, json.dumps({"files": [{"id": "inputs", "name": "inputs"}]})),
            (
                {"status": "200"},
                json.dumps(
                    {"files": [{"id": "old-project", "name": "faster_rcnn_project.zip"}, metadata]}
                ),
            ),
            ({"status": "200", "content-length": str(len(payload))}, payload),
        ]
    )
    service = build("drive", "v3", http=http, cache_discovery=False)
    monkeypatch.setattr(googleapiclient.discovery, "build", lambda *args, **kwargs: service)
    token = {"refresh_token": "test", "client_id": "test", "client_secret": "test"}
    monkeypatch.setitem(
        sys.modules,
        "kaggle_secrets",
        SimpleNamespace(
            UserSecretsClient=lambda: SimpleNamespace(get_secret=lambda name: json.dumps(token))
        ),
    )
    notebook = nbformat.read(
        Path(__file__).parents[1] / "experiments" / notebook_name,
        as_version=4,
    )
    source = notebook.cells[3].source.replace("/kaggle/temp/", str(tmp_path / "runtime") + "/")
    bootstrap = ast.Module(body=[ast.parse(source).body[0]], type_ignores=[])
    namespace: dict = {
        "IS_KAGGLE": True,
        "Path": Path,
        "ZipFile": ZipFile,
        "importlib": importlib,
        "json": json,
        "hashlib": hashlib,
        "LOGGER": logging.getLogger(__name__),
    }
    exec(compile(bootstrap, "notebook-bootstrap", "exec"), namespace)
    for name in required:
        assert (namespace["REPOSITORY_ROOT"] / name).read_bytes() == b"test source"


def test_submission_csv_is_restored_with_completed_checkpoint(tmp_path: Path) -> None:
    """Keep the exported CSV in immutable snapshots so a fresh runtime can reuse it."""
    source, remote = tmp_path / "source", tmp_path / "remote"
    remote.mkdir()
    _snapshot(source, 5, "input_resolution")
    csv = source / "faster_rcnn_input_resolution_stride5.csv"
    csv.write_text("image_id,PredictionString\nclip_1,\n")
    publish_snapshot(source)
    api = LocalDrive(remote)
    drive_api.sync_checkpoint_to_drive_api(api, source, str(remote))
    restored = tmp_path / "restored"
    drive_api.restore_checkpoint_from_drive_api(
        api, str(remote), restored, {"experiment_id": "input_resolution"}, tmp_path / "cache"
    )
    assert (restored / csv.name).read_bytes() == csv.read_bytes()


def test_native_yolo_mirror_round_trips_through_verified_generations(tmp_path: Path) -> None:
    """Publish native checkpoints through the snapshot transport and restore the newest one."""
    run, remote = tmp_path / "run", tmp_path / "remote"
    (run / "weights").mkdir(parents=True)
    remote.mkdir()
    api = LocalDrive(remote)

    def publish(directory: Path) -> None:
        publish_snapshot(directory)
        drive_api.sync_checkpoint_to_drive_api(api, directory, str(remote))

    mirror = CheckpointMirror(run, tmp_path / "mirror", publish, settle_seconds=0.0)
    for epoch in (1, 2):
        (run / "weights" / "last.pt").write_bytes(f"epoch {epoch}".encode())
        (run / "results.csv").write_text(f"epoch\n{epoch}\n")
        assert mirror.mirror_once()
    restored = drive_api.download_current_generation(api, str(remote), tmp_path / "restored")
    assert set(restored["files"]) == {"last.pt", "results.csv"}
    assert (tmp_path / "restored" / "last.pt").read_bytes() == b"epoch 2"
    assert (
        drive_api.download_current_generation(api, str(tmp_path / "run"), tmp_path / "none") == {}
    )
