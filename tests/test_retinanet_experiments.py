import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import torch
from PIL import Image

from hm2 import retinanet_experiments
from hm2.detection_data import DetectionDataset
from hm2.experiment_checkpoint import load_checkpoint, publish_snapshot, save_checkpoint
from hm2.retinanet import make_optimizer, train_steps
from hm2.retinanet_experiments import epoch_order, run_chunk


class TinyDetector(torch.nn.Module):
    """Minimal differentiable detector for CPU persistence checks."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.forward = self._forward
        self.anchor_generator = SimpleNamespace(sizes=((32,),), aspect_ratios=((1.0,),))

    def _forward(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        return {"loss": self.weight.square() * sum(image.mean() for image in images)}


def test_checkpoint_rejects_configuration_before_loading(tmp_path: Path) -> None:
    """Keep incompatible checkpoints from mutating any model state."""
    model = TinyDetector()
    optimizer, scheduler = make_optimizer(model, 0.01, 4, 1)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, optimizer, scheduler, 0, {"variant": "R0"})
    model.weight.data.fill_(3)
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, model, optimizer, scheduler, expected_config={"variant": "R1"})
    assert model.weight.item() == 3


def test_bounded_batches_and_checkpoint_resume(tmp_path: Path) -> None:
    """Consume each batch once and restore real CPU optimizer and cursor state."""
    model = TinyDetector()
    optimizer, scheduler = make_optimizer(model, 0.01, 4, 1)
    loader = [([torch.ones(3, 2, 2)], [{}])] * 2
    history = train_steps(
        model, loader, optimizer, scheduler, torch.device("cpu"), 10, max_batches=2
    )
    assert len(history) == 2
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, optimizer, scheduler, 2, {}, extra_state={"batch_cursor": 2})
    restored = TinyDetector()
    new_optimizer, new_scheduler = make_optimizer(restored, 0.01, 4, 1)
    state = load_checkpoint(path, restored, new_optimizer, new_scheduler, expected_config={})
    assert state["step"] == 2
    assert state["extra_state"]["batch_cursor"] == 2
    torch.testing.assert_close(
        optimizer.state[model.weight]["momentum_buffer"],
        new_optimizer.state[restored.weight]["momentum_buffer"],
    )
    torch.testing.assert_close(model.weight, restored.weight)
    assert optimizer.state_dict()["param_groups"] == new_optimizer.state_dict()["param_groups"]
    assert scheduler.state_dict() == new_scheduler.state_dict()


def test_resume_keeps_active_rng_when_moving_from_two_gpus_to_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restore the active GPU state without addressing an absent second Kaggle device."""
    model = TinyDetector()
    optimizer, scheduler = make_optimizer(model, 0.01, 4, 1)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, optimizer, scheduler, 3, {})
    state = torch.load(path, weights_only=True)
    state["cuda_rng"] = [
        torch.tensor([11], dtype=torch.uint8),
        torch.tensor([22], dtype=torch.uint8),
    ]
    torch.save(state, path)
    restored_rng = []

    def set_rng(value: torch.Tensor, device: int) -> None:
        if device != 0:
            raise IndexError("GPU no longer exists")
        restored_rng.append(value)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda.random, "set_rng_state", set_rng)
    result = load_checkpoint(path, model, optimizer, scheduler)
    assert result["step"] == 3
    assert len(restored_rng) == 1
    assert restored_rng[0].tolist() == [11]


def _fixture_data(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "data"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (root / "val").mkdir(parents=True)
    (root / "val/annotations_coco.json").write_text("{}")
    rows = []
    for index, split in enumerate(["train", "train", "train", "val"]):
        filename = f"{index}.png"
        Image.new("RGB", (8, 8), (100, 150, 200)).save(root / filename)
        rows.append(
            {
                "image_key": str(index),
                "image_id": index,
                "split": split,
                "clip": split,
                "frame_id": index,
                "relative_path": filename,
                "image_width": 8,
                "image_height": 8,
            }
        )
    pl.DataFrame(rows).write_parquet(metadata / "frames.parquet")
    pl.DataFrame(
        {
            "image_key": ["0", "1", "2", "3"],
            "x": [1.0] * 4,
            "y": [1.0] * 4,
            "width": [2.0] * 4,
            "height": [2.0] * 4,
            "category_id": [1] * 4,
        }
    ).write_parquet(metadata / "objects.parquet")
    pl.DataFrame({"image_key": []}, schema={"image_key": pl.String}).write_parquet(
        metadata / "ignored_regions.parquet"
    )
    return root, metadata


def test_epoch_three_resume_and_deterministic_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume epoch three to five without repeating frames or evaluations."""
    root, metadata = _fixture_data(tmp_path)
    frames = pl.read_parquet(metadata / "frames.parquet").filter(pl.col("split") == "train")
    objects = pl.read_parquet(metadata / "objects.parquet")
    order = epoch_order(3, 2026, 2)
    full = DetectionDataset(root, frames[order], objects, 0.5, augmentation_seed=2028)
    tail = DetectionDataset(root, frames[order].slice(2), objects, 0.5, augmentation_seed=2028)
    torch.testing.assert_close(full[2][1]["boxes"], tail[0][1]["boxes"])
    assert epoch_order(3, 2026, 2) == order
    monkeypatch.setattr(retinanet_experiments, "build_retinanet", lambda *a, **k: TinyDetector())
    monkeypatch.setattr(retinanet_experiments, "predict_loader", lambda *a: ([], [3], {}))
    monkeypatch.setattr(retinanet_experiments, "predictions_to_coco", lambda *a: [])
    output = tmp_path / "output"
    evaluations = []

    def evaluate(*args: object) -> dict:
        evaluations.append(len(evaluations) + 1)
        if len(evaluations) == 3:
            (output / "STOP").touch()
        return {"map": 0.1}

    monkeypatch.setattr(retinanet_experiments, "evaluate_coco", evaluate)
    config = {
        "experiment_id": "R0-F",
        "variant": "R0",
        "frame_stride": 1,
        "epochs": 5,
        "seed": 2026,
        "batch_size": 2,
        "num_workers": 0,
        "min_size": 640,
        "max_size": 1024,
        "learning_rate": 0.0025,
        "warmup_epochs": 0.5,
        "flip_probability": 0.5,
        "checkpoint_seconds": 600,
    }
    saved = []
    first = run_chunk(
        root,
        metadata,
        output,
        config,
        on_checkpoint=lambda path: saved.append((path / "sync/ready.json").exists()),
    )
    assert saved and all(saved)
    assert first["epochs_completed"] == 3
    assert first["image_exposures"] == 9
    assert not first["completed"]
    (output / "STOP").unlink()
    final = run_chunk(root, metadata, output, config)
    assert final["completed"]
    assert final["epochs_completed"] == 5
    assert final["global_step"] == 10
    assert final["image_exposures"] == 15
    assert len(evaluations) == 5
    checkpoint_modified = (output / "latest.pt").stat().st_mtime_ns
    again = run_chunk(root, metadata, output, config)
    assert again["global_step"] == 10
    assert len(evaluations) == 5
    assert (output / "latest.pt").stat().st_mtime_ns == checkpoint_modified


def test_skipped_amp_updates_do_not_repeat_epoch() -> None:
    """Count consumed frames even when AMP declines the first optimizer update."""
    model = TinyDetector()
    optimizer, scheduler = make_optimizer(model, 0.01, 4, 1)
    calls = 0

    def gradient_hook(gradient: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return gradient * float("inf") if calls == 1 else gradient

    model.weight.register_hook(gradient_hook)
    history = train_steps(
        model,
        [([torch.ones(3, 2, 2)], [{}])] * 2,
        optimizer,
        scheduler,
        torch.device("cpu"),
        2,
        scaler=torch.amp.GradScaler("cpu"),
        max_batches=2,
    )
    assert [row["optimizer_updated"] for row in history] == [False, True]
    assert len(history) == 2
    assert sum(row["image_count"] for row in history) == 2


@pytest.mark.parametrize("snapshot_state", ["existing", "missing", "placeholder"])
def test_resume_retries_failed_persistence_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snapshot_state: str
) -> None:
    """Retry a failed durable copy without rewriting or training past the saved checkpoint."""
    root, metadata = _fixture_data(tmp_path)
    monkeypatch.setattr(retinanet_experiments, "build_retinanet", lambda *a, **k: TinyDetector())
    config = {
        "experiment_id": "R0-S",
        "variant": "R0",
        "frame_stride": 1,
        "epochs": 5,
        "seed": 2026,
        "batch_size": 2,
        "num_workers": 0,
        "min_size": 640,
        "max_size": 1024,
        "learning_rate": 0.0025,
        "warmup_epochs": 0.5,
        "flip_probability": 0.5,
        "checkpoint_seconds": 600,
    }
    output = tmp_path / "output"

    def fail_copy(path: Path) -> None:
        raise OSError("Drive copy failed")

    with pytest.raises(OSError, match="Drive copy failed"):
        run_chunk(root, metadata, output, config, seconds=0, on_checkpoint=fail_copy)
    modified = (output / "latest.pt").stat().st_mtime_ns
    ready = output / "sync/ready.json"
    if snapshot_state == "missing":
        ready.unlink()
    elif snapshot_state == "placeholder":
        ready.write_text(json.dumps({"generation": "", "files": {}}))
    with pytest.raises(OSError, match="Drive copy failed"):
        run_chunk(root, metadata, output, config, seconds=0, on_checkpoint=fail_copy)
    callbacks = []
    status = run_chunk(
        root, metadata, output, config, seconds=0, on_checkpoint=lambda path: callbacks.append(path)
    )
    assert callbacks == [output]
    assert status["global_step"] == 0
    assert (output / "latest.pt").stat().st_mtime_ns == modified


def test_transfer_snapshots_are_immutable_and_keep_three_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish hashes last while preserving old checkpoint and mutable log contents."""
    import hashlib
    import json

    output = tmp_path / "output"
    output.mkdir()
    (output / "latest.pt").write_bytes(b"first checkpoint")
    (output / "execution.log").write_text("first log\n")
    (output / "status.json").write_text('{"global_step": 1}')
    publish_snapshot(output)
    manifest = json.loads((output / "sync/ready.json").read_text())
    first = output / "sync" / manifest["generation"]
    for filename, expected in manifest["files"].items():
        assert hashlib.sha256((first / filename).read_bytes()).hexdigest() == expected
    temporary = output / "new.pt"
    temporary.write_bytes(b"second checkpoint")
    temporary.replace(output / "latest.pt")
    (output / "execution.log").write_text("second log\n")
    publish_snapshot(output)
    assert (first / "latest.pt").read_bytes() == b"first checkpoint"
    assert (first / "execution.log").read_text() == "first log\n"
    publish_snapshot(output)
    assert first.exists()
    publish_snapshot(output)
    assert first.exists()
    generation_time = int(manifest["generation"]) + 31 * 60 * 1_000_000_000
    monkeypatch.setattr(retinanet_experiments.time, "time_ns", lambda: generation_time)
    publish_snapshot(output)
    assert not first.exists()
    assert len([path for path in (output / "sync").iterdir() if path.is_dir()]) == 3
