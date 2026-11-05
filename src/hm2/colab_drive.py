import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

import torch

from hm2.experiment_checkpoint import load_checkpoint, publish_snapshot, save_checkpoint, write_json
from hm2.retinanet import make_optimizer

LOGGER = logging.getLogger(__name__)


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _copy_verified(source: Path, destination: Path, digest: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _digest(destination) == digest:
        return
    temporary = destination.with_name(destination.name + ".incoming")
    shutil.copyfile(source, temporary)
    if _digest(temporary) != digest:
        raise ValueError(f"Drive transfer checksum mismatch: {destination.name}")
    temporary.replace(destination)


def _commit(source: Path, destination: Path, manifest: dict) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    pointer = destination / "CURRENT.json"
    if pointer.exists() and json.loads(pointer.read_text()) == manifest:
        return manifest
    generations = destination / "generations"
    generations.mkdir(exist_ok=True)
    for interrupted in generations.iterdir():
        if interrupted.is_dir() and not (interrupted / "manifest.json").is_file():
            shutil.rmtree(interrupted)
    generation = generations / manifest["generation"]
    generation.mkdir(parents=True, exist_ok=True)
    has_verified_generation = False
    try:
        for name, digest in manifest["files"].items():
            _copy_verified(source / name, generation / name, digest)
        write_json(generation / "manifest.json", manifest)
        has_verified_generation = True
    finally:
        if not has_verified_generation:
            shutil.rmtree(generation)
    write_json(pointer, manifest)
    if json.loads(pointer.read_text()) != manifest:
        raise ValueError("Drive checkpoint pointer failed read-back verification.")
    committed = sorted(
        path for path in generation.parent.iterdir() if (path / "manifest.json").is_file()
    )
    for previous in committed[:-2]:
        if previous != generation:
            shutil.rmtree(previous)
    LOGGER.info("Verified Drive snapshot %s in %s", manifest["generation"], destination)
    return manifest


def sync_checkpoint_to_drive(local: Path, drive: Path) -> dict:
    """Copy a completed local checkpoint snapshot before promoting its Drive pointer.

    Verify every copied file by SHA256 on the destination filesystem, then publish
    CURRENT.json. Retain the preceding committed generation until the new one passes.
    A failed copy raises immediately so training cannot continue without persistence.

    Args:
        local: Experiment output with the existing runner's sync/ready.json snapshot.
        drive: Experiment directory on the mounted Drive.

    Returns:
        The committed generation and its file checksums.
    """
    manifest = json.loads((local / "sync/ready.json").read_text())
    return _commit(local / "sync" / manifest["generation"], drive, manifest)


def _progress(state: dict) -> tuple[int, int, int, int, bool]:
    progress = state["extra_state"]
    return (
        progress["epochs_completed"],
        progress["batch_cursor"],
        state["step"],
        len(progress.get("metrics", [])),
        not progress.get("pending_evaluation", False),
    )


def restore_checkpoint_from_drive(drive: Path, local: Path, config: dict) -> dict:
    """Restore the most recent verified Drive checkpoint compatible with the requested recipe.

    Verify a complete generation and its configuration before replacing local files.
    Preserve newer compatible local progress when restarting only the notebook kernel.
    The existing epoch runner checks full dataset hashes and actual anchors before loading.

    Args:
        drive: Mounted Drive experiment directory with committed generations.
        local: Fast local Colab experiment output directory.
        config: Exact requested shared recipe and experiment identity.

    Returns:
        Saved epoch, cursor and optimization progress, or an empty mapping for a new run.
    """
    manifests = sorted(drive.glob("generations/*/manifest.json"), reverse=True)
    if not manifests:
        return {}
    selected = None
    failure: Exception | None = None
    for path in manifests:
        source = path.parent
        try:
            manifest = json.loads(path.read_text())
            has_valid_files = all(
                _digest(source / name) == digest for name, digest in manifest["files"].items()
            )
        except (OSError, json.JSONDecodeError) as error:
            failure = error
            LOGGER.warning("Reject unreadable Drive generation %s", source)
            continue
        if not has_valid_files:
            failure = ValueError(f"Drive checkpoint checksum mismatch in {source}")
            LOGGER.warning("Reject corrupt Drive generation %s", source)
            continue
        selected = source, manifest
        break
    if selected is None:
        raise ValueError("No verified Drive checkpoint remains.") from failure
    source, manifest = selected
    state = torch.load(source / "latest.pt", map_location="cpu", weights_only=True)
    if any(state["config"].get(key) != value for key, value in config.items()):
        raise ValueError("Drive checkpoint configuration differs from the requested experiment.")
    checkpoint = local / "latest.pt"
    if checkpoint.exists():
        current = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if any(current["config"].get(key) != value for key, value in config.items()):
            raise ValueError(
                "Local checkpoint configuration differs from the requested experiment."
            )
        if _progress(current) > _progress(state):
            publish_snapshot(local)
            sync_checkpoint_to_drive(local, drive)
            return current["extra_state"]
    for name, digest in manifest["files"].items():
        _copy_verified(source / name, local / name, digest)
    LOGGER.info("Restored Drive checkpoint %s, step %s", source, state["step"])
    return state["extra_state"]


def sync_results_to_drive(local: Path, drive: Path) -> dict:
    """Persist comparison tables, plots and previews as a verified Drive generation.

    Args:
        local: Batch root containing generated summaries and optional preview images.
        drive: Drive batch root whose summary directory receives these outputs.

    Returns:
        The summary generation and file checksums.
    """
    files = [path for path in local.iterdir() if path.suffix in {".json", ".csv", ".html"}]
    files.extend((local / "previews").glob("*.jpg"))
    manifest = {
        "generation": str(time.time_ns()),
        "files": {path.relative_to(local).as_posix(): _digest(path) for path in files},
    }
    return _commit(local, drive / "summary", manifest)


def verify_drive_checkpoint(local: Path, drive: Path) -> dict:
    """Exercise a real CPU checkpoint write, Drive copy, read and state restoration.

    Args:
        local: Temporary fast local directory for the tiny probe checkpoint.
        drive: Mounted Drive directory where the probe is written and read back.

    Returns:
        Verification status and matching checkpoint checksum.
    """
    model = torch.nn.Linear(2, 1)
    optimizer, scheduler = make_optimizer(model, 0.01, 4, 1)
    model(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    scheduler.step()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    config = {"purpose": "drive_checkpoint_probe"}
    path = local / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, 1, config, scaler)
    digest = _digest(path)
    target = drive / "checkpoint.pt"
    _copy_verified(path, target, digest)
    restored = torch.nn.Linear(2, 1)
    restored_optimizer, restored_scheduler = make_optimizer(restored, 0.01, 4, 1)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = load_checkpoint(
        target, restored, restored_optimizer, restored_scheduler, restored_scaler, config
    )
    if (
        state["step"] != 1
        or not all(
            torch.equal(value, restored.state_dict()[key])
            for key, value in model.state_dict().items()
        )
        or restored_scheduler.state_dict() != scheduler.state_dict()
        or not restored_optimizer.state
        or restored_scaler.state_dict() != scaler.state_dict()
    ):
        raise ValueError("Drive checkpoint state restoration failed.")
    return {"verified": True, "sha256": digest, "checkpoint": str(target)}
