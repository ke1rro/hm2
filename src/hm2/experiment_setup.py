import hashlib
import json
import logging
import shutil
from pathlib import Path
from zipfile import ZipFile

import polars as pl

from hm2.data import extract_archive
from hm2.detection_data import DetectionDataset, select_frames
from hm2.experiment_checkpoint import write_json

LOGGER = logging.getLogger(__name__)


def find_training_dataset(directory: Path) -> Path | None:
    """Locate a single original train/validation dataset, including archive wrappers.

    Args:
        directory: Existing or proposed extraction directory.

    Returns:
        Dataset root, or None when staging is still required.
    """
    roots = [
        path.parent.parent
        for path in directory.rglob("val/annotations_coco.json")
        if (path.parent.parent / "train/images").is_dir()
    ]
    if len(roots) > 1:
        raise ValueError("Set the dataset path to exactly one original train/val dataset.")
    return roots[0] if roots else None


def stage_dataset(
    archive: Path,
    destination: Path,
    local_archive: Path | None = None,
    reserve_bytes: int = 4 * 1024**3,
) -> Path:
    """Reuse staged data or extract an archive on fast runtime storage with a disk check.

    Args:
        archive: Existing source ZIP, potentially on mounted Drive.
        destination: Fast local extraction directory.
        local_archive: Optional temporary ZIP copy removed after successful extraction.
        reserve_bytes: Free space reserved for checkpoints and outputs.

    Returns:
        Extracted train/validation dataset root, preserving the source archive.
    """
    existing = find_training_dataset(destination)
    if existing is not None:
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as bundle:
        needed = sum(entry.file_size for entry in bundle.infolist()) + reserve_bytes
    if local_archive is not None and local_archive.resolve() != archive.resolve():
        needed += archive.stat().st_size
    if shutil.disk_usage(destination).free < needed:
        raise OSError("Insufficient local disk for dataset staging and checkpoints.")
    staged = archive
    if local_archive is not None and local_archive.resolve() != archive.resolve():
        local_archive.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Copy dataset ZIP to %s", local_archive)
        shutil.copyfile(archive, local_archive)
        staged = local_archive
    extract_archive(staged, destination)
    root = find_training_dataset(destination)
    if root is None:
        raise ValueError("The archive contains no original train/val dataset.")
    if staged != archive:
        staged.unlink()
    return root


def verify_dataset(
    root: Path,
    metadata: Path,
    expected_frames: dict[str, int],
    expected_clips: dict[str, int],
    source_manifest: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Verify split identity, image coverage, real RGB pairs and original annotations.

    Args:
        root: Original train/validation dataset.
        metadata: Prepared frames, objects and ignored-region tables.
        expected_frames: Expected image count for each labeled split.
        expected_clips: Expected clip count for each labeled split.
        source_manifest: Prepared JSON containing validation_annotations_sha256.

    Returns:
        Verified frame and object metadata tables for notebook inspection.
    """
    frames = pl.read_parquet(metadata / "frames.parquet")
    objects = pl.read_parquet(metadata / "objects.parquet")
    train = select_frames(frames, "train")
    validation = select_frames(frames, "val")
    if set(train["clip"]) & set(validation["clip"]):
        raise ValueError("Train and validation clips overlap.")
    for split, count in expected_frames.items():
        selected = select_frames(frames, split)
        if selected.height != count or selected["clip"].n_unique() != expected_clips[split]:
            raise ValueError(f"Dataset split {split} differs from the expected inventory.")
        missing = [path for path in selected["relative_path"] if not (root / path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} images, including {missing[:3]}")
        image, target = DetectionDataset(root, selected.head(1), objects)[0]
        LOGGER.info(
            "Verified %s: image %s, target boxes %s",
            split,
            list(image.shape),
            list(target["boxes"].shape),
        )
    inputs = json.loads(source_manifest.read_text())
    with (root / "val/annotations_coco.json").open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != inputs["validation_annotations_sha256"]:
        raise ValueError("Original validation annotations differ from the prepared metadata.")
    return frames, objects


def dataset_identity(root: Path, metadata: Path, selected_keys: list[str]) -> str:
    """Hash annotations and the exact training selection for checkpoint compatibility.

    Args:
        root: Dataset containing the original validation COCO annotations.
        metadata: Prepared frames, objects and ignored-region parquet directory.
        selected_keys: Ordered training frame identifiers used by the experiment.

    Returns:
        SHA256 identity compatible with previously saved experiment checkpoints.
    """
    identity = hashlib.sha256()
    for path in (
        metadata / "frames.parquet",
        metadata / "objects.parquet",
        metadata / "ignored_regions.parquet",
        root / "val/annotations_coco.json",
    ):
        identity.update(path.read_bytes())
    identity.update(json.dumps(selected_keys).encode())
    return identity.hexdigest()


def save_experiment_config(path: Path, configuration: dict | list[dict]) -> None:
    """Persist a new configuration atomically and reject changes to an existing batch.

    Args:
        path: Runtime or durable batch configuration pathname.
        configuration: Exact recipe or list of experiment recipes to preserve on resume.
    """
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError("Existing experiment configuration differs from the requested recipe.")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, configuration)
    if json.loads(path.read_text()) != configuration:
        raise ValueError("Experiment configuration failed write/read verification.")
