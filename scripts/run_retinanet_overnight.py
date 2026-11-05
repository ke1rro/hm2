import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import tomllib
import uuid
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import polars as pl
import torch

LOGGER = logging.getLogger(__name__)
REPOSITORY = Path(__file__).resolve().parents[1]
REMOTE = "/content/hm2_retinanet_overnight"


def _sanitize(text: str) -> str:
    return re.sub(r"https?://[^\s]+", lambda match: match.group().split("?", 1)[0], text)


def _cli(arguments: list[str], code: str | None = None, timeout: int = 7400) -> None:
    LOGGER.info("Colab: %s", " ".join(arguments[:2]))
    result = subprocess.run(
        ["colab", *arguments], input=code, text=True, capture_output=True, timeout=timeout
    )
    output = _sanitize(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Colab {arguments[0]} failed ({result.returncode}): {output[-4000:]}")
    if output.strip():
        LOGGER.info("%s", output.strip())


def _remote(session: str, batch: Path, code: str) -> dict:
    identifier = uuid.uuid4().hex
    remote_path = f"/content/hm2_response_{identifier}.json"
    local_path = batch / "state" / f"response_{identifier}.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "import json, traceback\nfrom pathlib import Path\ntry:\n"
        + textwrap.indent(code, "    ")
        + "\n    response = {'ok': True, 'result': result}\n"
        "except Exception:\n    response = {'ok': False, 'error': traceback.format_exc()}\n"
        + f"Path({remote_path!r}).write_text(json.dumps(response))\n"
    )
    _cli(["exec", "-s", session, "--timeout", "300"], script, timeout=330)
    _cli(["download", "-s", session, remote_path, str(local_path)])
    response = json.loads(local_path.read_text())
    local_path.unlink()
    if not response["ok"]:
        raise RuntimeError(response["error"])
    return response["result"]


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _upload_file(args: argparse.Namespace, batch: Path, path: Path, remote: str) -> None:
    part_size = 32 * 1024**2
    parent = str(Path(remote).parent)
    _remote(
        args.session,
        batch,
        f"Path({parent!r}).mkdir(parents=True, exist_ok=True)\nresult = {{'ready': True}}",
    )
    if path.stat().st_size <= part_size:
        _cli(["upload", "-s", args.session, str(path), remote])
        return
    identifier = uuid.uuid4().hex
    parts_directory = parent + "/.upload_" + identifier
    _remote(args.session, batch, f"Path({parts_directory!r}).mkdir()\nresult = {{'ready': True}}")
    temporary = batch / "state" / f"upload_{identifier}.part"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with path.open("rb") as source:
            while chunk := source.read(part_size):
                temporary.write_bytes(chunk)
                _cli(
                    [
                        "upload",
                        "-s",
                        args.session,
                        str(temporary),
                        f"{parts_directory}/{count:05d}.part",
                    ]
                )
                count += 1
        digest = _digest(path)
        _remote(
            args.session,
            batch,
            "import hashlib, shutil\n"
            f"parts = Path({parts_directory!r})\n"
            f"incoming = Path({remote + '.assembling'!r})\n"
            "with incoming.open('wb') as target:\n"
            f"    for index in range({count}):\n"
            "        with (parts / f'{index:05d}.part').open('rb') as source:\n"
            "            shutil.copyfileobj(source, target, length=1024**2)\n"
            "with incoming.open('rb') as source:\n"
            "    digest = hashlib.file_digest(source, 'sha256').hexdigest()\n"
            f"if digest != {digest!r}:\n"
            "    raise ValueError('Multipart upload checksum mismatch')\n"
            f"incoming.replace(Path({remote!r}))\n"
            "shutil.rmtree(parts)\nresult = {'verified': True}",
        )
    finally:
        temporary.unlink(missing_ok=True)


def _settings(batch: Path, resume: bool) -> tuple[Path, list[dict], pl.DataFrame]:
    sys.path.insert(0, str(REPOSITORY / "src"))
    from hm2.detection_data import select_frames
    from hm2.experiment_names import EXPERIMENTS, storage_id

    metadata = REPOSITORY / "data/metadata"
    manifest = json.loads((metadata / "manifest.json").read_text())
    root = Path(manifest["dataset_root"])
    for name, signature in manifest["sources"].items():
        stat = (root / name).stat()
        if [stat.st_size, stat.st_mtime_ns] != [signature["size"], signature["mtime_ns"]]:
            raise ValueError(f"Metadata source changed: {name}")
    frames = pl.read_parquet(metadata / "frames.parquet")
    train = select_frames(frames, "train")
    validation = select_frames(frames, "val")
    subset = select_frames(frames, "train", 5)
    if train.height != 24201 or validation.height != 2846 or subset["clip"].n_unique() != 56:
        raise ValueError("Official dataset counts or clip coverage differ from the experiment.")
    if set(train["clip"]) & set(validation["clip"]):
        raise ValueError("Official train/validation clips overlap.")
    settings = tomllib.loads((REPOSITORY / "config/retinanet_overnight.toml").read_text())
    if settings["epochs"] != 5 or settings["subset_stride"] != 5:
        raise ValueError("This batch requires five epochs and the stride-five subset.")
    configurations = []
    for experiment_name, (variant, frame_stride) in EXPERIMENTS.items():
        name = storage_id(experiment_name)
        selected = subset if frame_stride == 5 else train
        configurations.append(
            {
                **{key: value for key, value in settings.items() if key != "subset_stride"},
                "experiment_id": name,
                "variant": storage_id(variant),
                "frame_stride": frame_stride,
                "warmup_epochs": 100 / math.ceil(subset.height / settings["batch_size"]),
            }
        )
        keys = selected["image_key"].to_list()
        path = batch / "manifests" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != keys:
            raise ValueError(f"Selection changed for {name}.")
        path.write_text(json.dumps(keys) + "\n")
    config_path = batch / "batch_config.json"
    if config_path.exists():
        if not resume:
            raise ValueError("Batch already exists. Use --resume to preserve its progress.")
        if json.loads(config_path.read_text()) != configurations:
            raise ValueError("Requested recipe differs from the saved batch configuration.")
    config_path.write_text(json.dumps(configurations, indent=2) + "\n")
    return root, configurations, pl.concat([train, validation])


def _connect(args: argparse.Namespace, batch: Path) -> dict:
    _cli(["sessions"])
    probe = _remote(
        args.session,
        batch,
        "import importlib.util, platform, shutil, torch, torchvision\n"
        "result = {'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None, "
        "'python': platform.python_version(), 'torch': torch.__version__, "
        "'torchvision': torchvision.__version__, 'disk_free': shutil.disk_usage('/content').free, "
        "'missing': [name for name in ('polars', 'pycocotools', 'PIL', 'tqdm') "
        "if importlib.util.find_spec(name) is None]}",
    )
    if not probe["gpu"]:
        raise RuntimeError("The selected Colab session has no CUDA GPU.")
    if probe["missing"]:
        raise RuntimeError(f"Remote dependencies are missing: {probe['missing']}")
    if probe["disk_free"] < 15 * 1024**3:
        raise OSError("Colab needs at least 15 GiB of free staging/training disk.")
    (batch / "environment.json").write_text(json.dumps(probe, indent=2) + "\n")
    return probe


def _stage_code(args: argparse.Namespace, batch: Path, root: Path, frames: pl.DataFrame) -> None:
    bundle = batch / "state/code_metadata.zip"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(bundle, "w") as archive:
        for path in (REPOSITORY / "src/hm2").glob("*.py"):
            archive.write(path, "src/hm2/" + path.name)
        for name in ("frames.parquet", "objects.parquet", "ignored_regions.parquet"):
            archive.write(REPOSITORY / "data/metadata" / name, "metadata/" + name)
        archive.write(root / "val/annotations_coco.json", "data/val/annotations_coco.json")
        for split in ("train", "val"):
            for row in frames.filter(pl.col("split") == split).head(2).iter_rows(named=True):
                archive.write(root / row["relative_path"], "data/" + row["relative_path"])
    _upload_file(args, batch, bundle, "/content/hm2_code.zip")
    _remote(
        args.session,
        batch,
        "import sys\nfrom zipfile import ZipFile\n"
        f"workspace = Path({REMOTE!r})\n"
        "with ZipFile('/content/hm2_code.zip') as archive:\n"
        "    archive.extractall(workspace)\n"
        "sys.path.insert(0, str(workspace / 'src'))\nresult = {'staged': True}",
    )


def _preflight(args: argparse.Namespace, batch: Path) -> dict:
    (batch / "preflight.json").write_text(json.dumps({"restore_marker": False}) + "\n")
    output = f"{REMOTE}/outputs/{batch.name}"
    result = _remote(
        args.session,
        batch,
        "import logging, sys, time, torch, polars as pl\n"
        f"sys.path.insert(0, {REMOTE + '/src'!r})\n"
        "from hm2.detection_data import DetectionDataset, make_loader, select_frames\n"
        "from hm2.retinanet import build_retinanet, make_optimizer, train_steps, "
        "predict_loader\n"
        "from hm2.experiment_checkpoint import save_checkpoint, load_checkpoint\n"
        f"workspace = Path({REMOTE!r})\n"
        "frames = pl.read_parquet(workspace / 'metadata/frames.parquet')\n"
        "objects = pl.read_parquet(workspace / 'metadata/objects.parquet')\n"
        "device = torch.device('cuda')\ntorch.set_num_threads(2)\n"
        "model = build_retinanet('standard_anchors', allow_download=True).to(device)\n"
        "optimizer, scheduler = make_optimizer(model, .0025, 20, 2)\n"
        "scaler = torch.amp.GradScaler('cuda')\n"
        "dataset = DetectionDataset(workspace / 'data', select_frames(frames, 'train').head(2), "
        "objects, .5)\n"
        "loader = make_loader(dataset, 2, 2, device)\n"
        "history = train_steps(model, loader, optimizer, scheduler, device, 3, scaler=scaler, "
        "max_attempts=6)\n"
        "val = DetectionDataset(workspace / 'data', "
        "select_frames(frames, 'val').head(2), objects)\n"
        "predictions, image_ids, speed = predict_loader(model, "
        "make_loader(val, 1, 2, device), device)\n"
        f"path = Path({output + '/preflight.pt'!r})\n"
        "save_checkpoint(path, model, optimizer, scheduler, 3, {'diagnostic': True}, scaler)\n"
        "restored = load_checkpoint(path, model, optimizer, scheduler, scaler)\n"
        "measured = history[1:]\n"
        "result = {'checkpoint_step': restored['step'], "
        "'training_images_per_second': len(measured)*2/sum(row['seconds'] for row in measured), "
        "'validation_milliseconds_per_frame': speed['milliseconds_per_frame'], "
        "'prediction_counts': [len(item['boxes']) for item in predictions]}\n"
        "del model, optimizer, scheduler, scaler, loader\ntorch.cuda.empty_cache()",
    )
    temporary = batch / "preflight.pt.download"
    _cli(["download", "-s", args.session, output + "/preflight.pt", str(temporary)])
    state = torch.load(temporary, map_location="cpu", weights_only=True)
    if state["step"] != 3 or not state["optimizer"]["state"] or not state["scaler"]:
        raise RuntimeError("Downloaded preflight checkpoint has incomplete training state.")
    temporary.replace(batch / "preflight.pt")
    _upload_file(args, batch, batch / "preflight.pt", output + "/roundtrip.pt")
    restored = _remote(
        args.session,
        batch,
        "import torch, sys\n"
        f"sys.path.insert(0, {REMOTE + '/src'!r})\n"
        "from hm2.retinanet import build_retinanet, make_optimizer\n"
        "from hm2.experiment_checkpoint import load_checkpoint\n"
        "device = torch.device('cuda')\n"
        "model = build_retinanet('standard_anchors', allow_download=True).to(device)\n"
        "optimizer, scheduler = make_optimizer(model, .0025, 20, 2)\n"
        "scaler = torch.amp.GradScaler('cuda')\n"
        f"path = Path({output + '/roundtrip.pt'!r})\n"
        "restored = load_checkpoint(path, model, optimizer, scheduler, scaler, "
        "expected_config={'diagnostic': True})\n"
        "saved = torch.load(path, map_location='cpu', weights_only=True)\n"
        "if restored['step'] != 3 or scheduler.last_epoch != 3:\n"
        "    raise ValueError('Round-trip optimizer or scheduler position differs')\n"
        "if not optimizer.state or not any('momentum_buffer' in item "
        "for item in optimizer.state.values()):\n"
        "    raise ValueError('Round-trip momentum state is missing')\n"
        "if not scaler.state_dict() or scaler.state_dict() != saved['scaler']:\n"
        "    raise ValueError('Round-trip scaler state differs')\n"
        "if not all(torch.equal(value.cpu(), saved['model'][name]) "
        "for name,value in model.state_dict().items()):\n"
        "    raise ValueError('Round-trip model weights differ')\n"
        "result = {'restore_marker': True}\n"
        "del model, optimizer, scheduler, scaler, saved\ntorch.cuda.empty_cache()",
    )
    result.update(restored)
    (batch / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _stage_images(args: argparse.Namespace, batch: Path, root: Path, frames: pl.DataFrame) -> None:
    records = [
        {"path": path, "size": (root / path).stat().st_size} for path in frames["relative_path"]
    ]
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    total = sum(record["size"] for record in records)
    if shutil.disk_usage(batch).free < total + 5 * 1024**3:
        raise OSError("Insufficient local space for temporary staging and retrieved checkpoints.")
    LOGGER.info(
        "Required train+val transfer %.2f GiB, dataset key %s", total / 1024**3, fingerprint
    )
    started = time.perf_counter()
    groups: list[list[dict]] = [[]]
    size = 0
    for record in records:
        if size + record["size"] > 128 * 1024**2:
            groups.append([])
            size = 0
        groups[-1].append(record)
        size += record["size"]
    for index, group in enumerate(groups):
        name = f"images_{index:03d}.zip"
        expected = {record["path"]: record["size"] for record in group}
        available = _remote(
            args.session,
            batch,
            f"destination = Path({REMOTE + '/data'!r})\n"
            f"expected = json.loads({json.dumps(expected)!r})\n"
            "result = {'cached': all((destination / name).is_file() and "
            "(destination / name).stat().st_size == size for name,size in expected.items())}",
        )
        if available["cached"]:
            continue
        archive_path = batch / "state" / name
        with ZipFile(archive_path, "w", compression=ZIP_STORED) as archive:
            for record in group:
                archive.write(root / record["path"], record["path"])
        digest = _digest(archive_path)
        _upload_file(args, batch, archive_path, "/content/hm2_images.zip")
        _remote(
            args.session,
            batch,
            "import hashlib\nfrom zipfile import ZipFile\n"
            "source = Path('/content/hm2_images.zip')\n"
            "with source.open('rb') as stream:\n"
            "    digest = hashlib.file_digest(stream, 'sha256').hexdigest()\n"
            f"if digest != {digest!r}:\n    raise ValueError('Dataset upload checksum mismatch')\n"
            f"destination = Path({REMOTE + '/data'!r})\n"
            "with ZipFile(source) as archive:\n"
            "    if archive.testzip() is not None:\n"
            "        raise ValueError('Dataset archive failed CRC verification')\n"
            "    archive.extractall(destination)\n"
            f"expected = json.loads({json.dumps(expected)!r})\n"
            "if not all((destination / name).stat().st_size == size "
            "for name,size in expected.items()):\n"
            "    raise ValueError('Incomplete extracted dataset shard')\n"
            "source.unlink()\nresult = {'verified': True}",
        )
        archive_path.unlink()
        LOGGER.info("Dataset shard %s/%s verified", index + 1, len(groups))
    (batch / "staging.json").write_text(
        json.dumps(
            {"bytes": total, "seconds": time.perf_counter() - started, "fingerprint": fingerprint}
        )
        + "\n"
    )


def _sessions() -> str:
    return subprocess.run(
        ["colab", "sessions"], capture_output=True, text=True, check=True, timeout=60
    ).stdout


def _ensure_session(args: argparse.Namespace) -> None:
    sessions = _sessions()
    if f"[{args.session}]" not in sessions:
        _cli(["new", "-s", args.session, "--gpu", "T4"])
        if f"[{args.session}]" not in _sessions():
            raise RuntimeError("Requested Colab session was not created.")


def _validate_checkpoint(path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not {"model", "optimizer", "scheduler", "scaler", "config", "step"} <= state.keys():
        raise ValueError(f"Incomplete checkpoint {path.name}")


def _retrieve(args: argparse.Namespace, batch: Path, name: str) -> dict:
    remote = f"{REMOTE}/outputs/{batch.name}/{name}/sync"
    local = batch / name
    local.mkdir(exist_ok=True)
    ready_path = local / "ready.download.json"
    _cli(["download", "-s", args.session, remote + "/ready.json", str(ready_path)])
    ready = json.loads(ready_path.read_text())
    ready_path.unlink()
    manifest = ready["files"]
    if not manifest:
        status_path = local / "status.json"
        return json.loads(status_path.read_text()) if status_path.exists() else {"completed": False}
    generation = ready["generation"]
    if Path(generation).name != generation:
        raise ValueError("Invalid snapshot generation.")
    current = local / "CURRENT"
    previous = local / "snapshots" / current.read_text().strip() if current.exists() else None
    if previous is not None and previous.name == generation:
        status_path = previous / "status.json"
        return json.loads(status_path.read_text()) if status_path.exists() else {"completed": False}
    old_manifest = (
        json.loads((previous / "retrieval_manifest.json").read_text())
        if (previous is not None)
        else {}
    )
    snapshot = local / "snapshots" / generation
    snapshot.mkdir(parents=True, exist_ok=True)
    for filename, digest in manifest.items():
        if Path(filename).name != filename:
            raise ValueError("Invalid snapshot filename.")
        path = snapshot / filename
        if path.exists() and _digest(path) == digest:
            continue
        if previous is not None and old_manifest.get(filename) == digest:
            path.unlink(missing_ok=True)
            path.hardlink_to(previous / filename)
            continue
        temporary = path.with_suffix(path.suffix + ".download")
        _cli(
            [
                "download",
                "-s",
                args.session,
                remote + "/" + ready["generation"] + "/" + filename,
                str(temporary),
            ]
        )
        if _digest(temporary) != digest:
            raise ValueError(f"Retrieved checksum mismatch: {filename}")
        if filename.endswith(".pt"):
            _validate_checkpoint(temporary)
        temporary.replace(path)
    (snapshot / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pointer = local / "CURRENT.incoming"
    pointer.write_text(generation + "\n")
    pointer.replace(local / "CURRENT")
    for filename in manifest:
        temporary = local / (filename + ".incoming")
        temporary.unlink(missing_ok=True)
        temporary.hardlink_to(snapshot / filename)
        temporary.replace(local / filename)
    (local / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    generations = sorted(path for path in (local / "snapshots").iterdir() if path.is_dir())
    for old in generations[:-2]:
        if old.name != generation:
            shutil.rmtree(old)
    path = local / "status.json"
    return json.loads(path.read_text()) if path.exists() else {"completed": False}


def _restore(args: argparse.Namespace, batch: Path, name: str) -> None:
    local = batch / name
    pointer = local / "CURRENT"
    if pointer.exists():
        local = local / "snapshots" / pointer.read_text().strip()
    manifest_path = local / "retrieval_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    remote = f"{REMOTE}/outputs/{batch.name}/{name}"
    remote_state = _remote(
        args.session,
        batch,
        f"path = Path({remote + '/latest.pt'!r})\n"
        "import torch\n"
        "state = torch.load(path, map_location='cpu', weights_only=True) if path.exists() else {}\n"
        "progress = state.get('extra_state', {})\n"
        "result = {'progress': [progress.get('epochs_completed', -1), "
        "progress.get('batch_cursor', -1), state.get('step', -1)]}",
    )
    checkpoint = local / "latest.pt"
    if not checkpoint.exists():
        return
    _validate_checkpoint(checkpoint)
    if _digest(checkpoint) != manifest.get("latest.pt"):
        raise ValueError("Local resume checkpoint differs from verified retrieval manifest.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    progress = state["extra_state"]
    cursor = [progress["epochs_completed"], progress["batch_cursor"], state["step"]]
    if remote_state["progress"] >= cursor:
        return
    for filename, digest in manifest.items():
        path = local / filename
        if not path.exists() or _digest(path) != digest:
            raise ValueError(f"Unverified local restore artifact: {filename}")
        if filename.endswith(".pt"):
            _validate_checkpoint(path)
        incoming = remote + "/" + filename + ".incoming"
        _upload_file(args, batch, path, incoming)
        _remote(
            args.session,
            batch,
            "import hashlib\n"
            f"path = Path({incoming!r})\n"
            "with path.open('rb') as stream:\n"
            "    digest = hashlib.file_digest(stream, 'sha256').hexdigest()\n"
            f"if digest != {digest!r}:\n    raise ValueError('Resume upload checksum mismatch')\n"
            f"path.replace(Path({remote + '/' + filename!r}))\nresult = {{'restored': True}}",
        )


def _run_chunk(args: argparse.Namespace, batch: Path, config: dict) -> dict:
    name = config["experiment_id"]
    output = f"{REMOTE}/outputs/{batch.name}/{name}"
    response = f"{output}/response_{uuid.uuid4().hex}.json"
    _remote(
        args.session,
        batch,
        f"sync = Path({output + '/sync'!r})\nsync.mkdir(parents=True, exist_ok=True)\n"
        "if not (sync / 'ready.json').exists():\n"
        "    (sync / 'ready.json').write_text(json.dumps({'generation': '', 'files': {}}))\n"
        "result = {'ready': True}",
    )
    script = (
        "import sys, json, traceback\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {REMOTE + '/src'!r})\n"
        "try:\n"
        "    from hm2.retinanet_experiments import run_chunk\n"
        f"    result = run_chunk(Path({REMOTE + '/data'!r}), "
        f"Path({REMOTE + '/metadata'!r}), Path({output!r}), "
        f"json.loads({json.dumps(config)!r}), resume=True, seconds=600)\n"
        "    response = {'ok': True, 'result': result}\n"
        "except Exception:\n    response = {'ok': False, 'error': traceback.format_exc()}\n"
        f"Path({response!r}).write_text(json.dumps(response))\n"
    )
    local = batch / name
    local.mkdir(exist_ok=True)
    with (local / "remote_cli.log").open("a") as log:
        process = subprocess.Popen(
            ["colab", "exec", "-s", args.session, "--timeout", "7200"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def forward_output() -> None:
            if process.stdout is not None:
                for line in process.stdout:
                    log.write(_sanitize(line))
                    log.flush()

        reader = threading.Thread(target=forward_output, daemon=True)
        reader.start()
        if process.stdin is None:
            raise RuntimeError("Remote process has no input pipe.")
        process.stdin.write(script)
        process.stdin.close()
        while process.poll() is None:
            time.sleep(30)
            _retrieve(args, batch, name)
        reader.join(timeout=10)
        if process.returncode != 0:
            raise RuntimeError(f"Colab execution failed with exit {process.returncode}.")
    response_path = local / "response.download.json"
    response_path.unlink(missing_ok=True)
    _cli(["download", "-s", args.session, response, str(response_path)])
    result = json.loads(response_path.read_text())
    response_path.unlink()
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return _retrieve(args, batch, name)


def _work(args: argparse.Namespace, batch: Path, root: Path, configs: list[dict]) -> None:
    sys.path.insert(0, str(REPOSITORY / "src"))
    from hm2.experiment_names import experiment_label
    from hm2.retinanet_analysis import summarize_batch

    frames = pl.read_parquet(REPOSITORY / "data/metadata/frames.parquet").filter(
        pl.col("split").is_in(["train", "val"])
    )
    probe = json.loads((batch / "preflight.json").read_text())
    if probe.get("restore_marker") is not True:
        raise RuntimeError("Full local checkpoint round-trip preflight has not passed.")
    _stage_images(args, batch, root, frames)
    exposures = sum(
        len(json.loads((batch / "manifests" / f"{item['experiment_id']}.json").read_text()))
        * item["epochs"]
        for item in configs
    )
    estimate = {
        "training_hours": exposures / probe["training_images_per_second"] / 3600,
        "validation_hours": 20 * 2846 * probe["validation_milliseconds_per_frame"] / 3_600_000,
        "staging": json.loads((batch / "staging.json").read_text()),
        "limitations": "Short diagnostic throughput only; excludes checkpoint and download I/O. "
        "Sustained speed and runtime quota are uncertain; overnight completion is not guaranteed.",
    }
    (batch / "estimate.json").write_text(json.dumps(estimate, indent=2) + "\n")
    LOGGER.info("Planning estimate: %s", estimate)
    recoveries = 0
    for config in configs:
        name = config["experiment_id"]
        status_path = batch / name / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("completed"):
            for filename in ("latest.pt", "best.pt"):
                _validate_checkpoint(batch / name / filename)
                saved = torch.load(batch / name / filename, map_location="cpu", weights_only=True)[
                    "config"
                ]
                for key, value in config.items():
                    if saved.get(key) != value:
                        raise ValueError(f"Completed checkpoint configuration differs: {key}")
            continue
        _restore(args, batch, name)
        while not status.get("completed"):
            if (batch / "STOP").exists():
                _retrieve(args, batch, name)
                _cli(["stop", "-s", args.session])
                return
            try:
                previous = (
                    _digest(batch / name / "metrics.json")
                    if (batch / name / "metrics.json").exists()
                    else None
                )
                status = _run_chunk(args, batch, config)
            except subprocess.SubprocessError, RuntimeError, OSError:
                if f"[{args.session}]" in _sessions() or recoveries >= 2:
                    raise
                recoveries += 1
                LOGGER.warning(
                    "Runtime lost; restoring verified local state, recovery %s", recoveries
                )
                _ensure_session(args)
                _connect(args, batch)
                _stage_code(args, batch, root, frames)
                _stage_images(args, batch, root, frames)
                _restore(args, batch, name)
                continue
            metrics_path = batch / name / "metrics.json"
            changed = metrics_path.exists() and _digest(metrics_path) != previous
            summarize_batch(
                batch, root if changed else None, REPOSITORY / "data/metadata" if changed else None
            )
            LOGGER.info(
                "Verified local progress %s: %s",
                experiment_label(name),
                {**status, "experiment_id": experiment_label(name)},
            )
    summarize_batch(batch, root, REPOSITORY / "data/metadata")
    (batch / "COMPLETED").write_text(datetime.now().isoformat() + "\n")
    _cli(["stop", "-s", args.session])


def _claim_lock(batch: Path) -> None:
    path = batch / "controller.pid"
    if path.exists():
        pid = int(path.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink()
        else:
            raise RuntimeError(f"Batch already has live controller PID {pid}.")
    with path.open("x") as destination:
        destination.write(str(os.getpid()) + "\n")


def main() -> None:
    """Prepare, detach, resume, or safely stop the fixed four-run Colab experiment batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--session", default="hm2-retinanet-overnight")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--new-session", action="store_true")
    args = parser.parse_args()
    batch = (
        args.batch_dir
        or Path("runs/retinanet_overnight") / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    batch.mkdir(parents=True, exist_ok=True)
    if args.stop:
        (batch / "STOP").touch()
        LOGGER.info(
            "Stop requested for %s; controller persists current chunk before stopping", batch
        )
        return
    _claim_lock(batch)
    root, configs, frames = _settings(batch, args.resume)
    if not args.worker:
        for command in ("uv", "colab", "caffeinate"):
            if shutil.which(command) is None:
                raise FileNotFoundError(f"Required command is not installed: {command}")
        for command in ("new", "exec", "upload", "download"):
            _cli([command, "--help"])
        if args.new_session:
            _cli(["new", "-s", args.session, "--gpu", "T4"])
        _ensure_session(args)
        _connect(args, batch)
        _stage_code(args, batch, root, frames)
        _preflight(args, batch)
        if args.prepare_only:
            return
        (batch / "STOP").unlink(missing_ok=True)
    if args.detach:
        (batch / "controller.pid").unlink(missing_ok=True)
        command = [
            "caffeinate",
            "-ims",
            "uv",
            "run",
            "--locked",
            "python",
            str(Path(__file__).resolve()),
            "--worker",
            "--resume",
            "--batch-dir",
            str(batch),
            "--session",
            args.session,
        ]
        with (batch / "controller.log").open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (batch / "launcher.pid").write_text(str(process.pid) + "\n")
        deadline = time.monotonic() + 15
        while (
            process.poll() is None
            and not (batch / "controller.pid").exists()
            and time.monotonic() < deadline
        ):
            time.sleep(0.5)
        if process.poll() is not None or not (batch / "controller.pid").exists():
            raise RuntimeError("Detached controller failed to start; inspect controller.log.")
        os.kill(int((batch / "controller.pid").read_text()), 0)
        LOGGER.info(
            "Detached batch %s, launcher PID %s, log %s",
            batch.name,
            process.pid,
            batch / "controller.log",
        )
    else:
        _work(args, batch, root, configs)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    main()
