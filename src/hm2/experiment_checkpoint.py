import hashlib
import json
import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

LOGGER = logging.getLogger(__name__)


def write_json(path: Path, value: object) -> None:
    """Write JSON through an atomic local replacement.

    Args:
        path: Destination JSON pathname.
        value: JSON-compatible configuration or result record.
    """
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def publish_snapshot(output: Path) -> None:
    """Publish an immutable checkpoint generation for verified durable synchronization.

    Hard-link completed model files, copy result records, and publish ready.json last.
    Retain at least three local generations and all generations younger than 30 minutes.

    Args:
        output: Experiment directory containing completed checkpoints, JSON, CSV and logs.
    """
    sync = output / "sync"
    sync.mkdir(exist_ok=True)
    generation = str(time.time_ns())
    snapshot = sync / generation
    snapshot.mkdir()
    files = {}
    for source in sorted(output.iterdir()):
        if source.suffix not in {".pt", ".json", ".log", ".csv"} or not source.is_file():
            continue
        destination = snapshot / source.name
        if source.suffix == ".pt":
            destination.hardlink_to(source)
        else:
            shutil.copy2(source, destination)
        with destination.open("rb") as stream:
            files[source.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json(sync / "ready.json", {"generation": generation, "files": files})
    generations = sorted(path for path in sync.iterdir() if path.is_dir())
    for expired in generations[:-3]:
        if time.time_ns() - int(expired.name) >= 30 * 60 * 1_000_000_000:
            shutil.rmtree(expired)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    step: int,
    config: dict,
    scaler: torch.amp.GradScaler | None = None,
    extra_state: dict | None = None,
) -> None:
    """Save model, optimizer, schedule, AMP, configuration, and RNG states atomically.

    Args:
        path: Ignored project checkpoint path.
        model: Model whose state is saved.
        optimizer: Stateful training optimizer.
        scheduler: Stateful learning-rate schedule.
        step: Number of completed loop steps.
        config: JSON-compatible experiment settings.
        scaler: Optional persistent AMP scaler.
        extra_state: Optional durable experiment progress.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_state = np.random.get_state()
    state = {
        "extra_state": extra_state or {},
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": config,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    write_json(path.with_suffix(".json"), config)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler | None = None,
    expected_config: dict | None = None,
) -> dict:
    """Restore training states and random generators from a trusted local checkpoint.

    Args:
        path: Previously saved local checkpoint.
        model: Matching experiment architecture.
        optimizer: Matching optimizer.
        scheduler: Matching learning-rate schedule.
        scaler: Optional gradient scaler to restore.
        expected_config: Exact configuration to validate before mutating training state.

    Returns:
        Completed step and exact saved configuration for checking the resumed run.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if expected_config is not None and state["config"] != expected_config:
        raise ValueError("Checkpoint configuration does not match the requested experiment.")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    numpy_state = state["numpy_rng"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        available_devices = torch.cuda.device_count()
        if len(state["cuda_rng"]) != available_devices:
            LOGGER.info(
                "Restore available GPU RNG states: checkpoint=%s, runtime=%s",
                len(state["cuda_rng"]),
                available_devices,
            )
        torch.cuda.set_rng_state_all(state["cuda_rng"][:available_devices])
    return {
        "step": state["step"],
        "config": state["config"],
        "extra_state": state.get("extra_state", {}),
    }
