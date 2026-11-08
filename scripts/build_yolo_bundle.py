import argparse
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

LOGGER = logging.getLogger(__name__)
BUNDLE_NAME = "yolo_native_project.zip"
CONFIG_NAME = "config/yolo11s_native_1280.toml"
VISDRONE_URL = (
    "https://huggingface.co/dronefreak/visdrone-yolo11s/resolve/"
    "c1c9c8f559a7ebdc5271f8d5695b026dd22420cc/best.pt"
)
VISDRONE_SHA256 = "3134756bf29e761550991c0f1e5d91b262736e1c383082172691503227f645e7"
VISDRONE_CHANNELS = (0, 2, 3, 4, 5, 6, 8, 9)
INIT_NAME = "yolo11s_visdrone_det_8cls.pt"
NOTEBOOK_IMPORTS = (
    "hm2.data",
    "hm2.drive_api",
    "hm2.experiment_checkpoint",
    "hm2.faster_rcnn_experiments",
    "hm2.visualization",
    "hm2.yolo_data",
    "hm2.yolo_pipeline",
)
OFFLINE_PACKAGES = ("ultralytics==8.3.235", "ultralytics-thop>=2.0", "faster-coco-eval>=1.8.0")


def build_initial_checkpoint(root: Path, directory: Path) -> dict:
    """Download the pinned VisDrone-DET YOLO11s checkpoint and convert it to eight classes.

    The source has eleven VisDrone classes. Person takes the pedestrian channel and
    tricycle the tricycle channel; the other six competition classes map one to one.

    Args:
        root: Repository root whose src/ provides the conversion function.
        directory: Staging directory for the converted checkpoint.

    Returns:
        Conversion provenance, including the source URL and both SHA256 digests.
    """
    sys.path.insert(0, str(root / "src"))
    from hm2.yolo_pipeline import remap_detection_head

    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "visdrone_yolo11s_source.pt"
    if not source.is_file():
        urllib.request.urlretrieve(VISDRONE_URL, source)
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != VISDRONE_SHA256:
            raise ValueError("The downloaded VisDrone checkpoint does not match its pinned SHA256.")
    destination = directory / INIT_NAME
    provenance = remap_detection_head(source, destination, VISDRONE_CHANNELS, "yolo11s.yaml")
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {**provenance, "source_url": VISDRONE_URL, "converted_sha256": digest}


def build_bundle(root: Path, output: Path, initial_checkpoint: Path) -> dict:
    """Build and verify the code ZIP read by the YOLO pipeline notebook.

    The archive holds src/hm2, the pipeline configuration and the initial checkpoint under
    weights/. Every entry is compared with its source bytes, and the notebook's imports are
    resolved from a fresh extraction so a missing module fails here instead of on Kaggle.

    Args:
        root: Repository root.
        output: Destination ZIP path; no cloud files are changed.
        initial_checkpoint: Converted eight-class checkpoint used to start training.

    Returns:
        ZIP path, size, entry count and SHA256.
    """
    entries = {
        path.relative_to(root).as_posix(): path for path in sorted((root / "src/hm2").glob("*.py"))
    }
    entries[CONFIG_NAME] = root / CONFIG_NAME
    entries[f"weights/{initial_checkpoint.name}"] = initial_checkpoint
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".incoming")
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in entries.items()}
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for name, path in entries.items():
            archive.write(path, name)
        archive.writestr("bundle_manifest.json", json.dumps(hashes, sort_keys=True))
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Bundle CRC verification failed.")
        for name, path in entries.items():
            if archive.read(name) != path.read_bytes():
                raise ValueError(f"Bundle differs from current source: {path}")
        with tempfile.TemporaryDirectory(prefix="hm2-yolo-bundle-") as extracted:
            archive.extractall(extracted)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import importlib, sys; sys.path.insert(0, sys.argv[1]); "
                    "[importlib.import_module(name) for name in sys.argv[2:]]",
                    str(Path(extracted) / "src"),
                    *NOTEBOOK_IMPORTS,
                ],
                check=True,
                cwd=extracted,
            )
    temporary.replace(output)
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "file": str(output),
        "bytes": output.stat().st_size,
        "entries": len(hashes),
        "sha256": digest,
    }


def stage_offline_wheels(directory: Path, python_version: str) -> list[str]:
    """Download the wheels needed to install the pipeline without Internet on Kaggle.

    Files are written next to the code bundle without subdirectories, because Kaggle
    dataset uploads skip directories by default.

    Args:
        directory: Dataset directory that also holds the code bundle.
        python_version: Target interpreter version of the Kaggle image, such as 3.13.

    Returns:
        Downloaded wheel filenames.
    """
    for stale in directory.glob("*.whl"):
        stale.unlink()
    subprocess.run(
        [
            "uvx",
            "--from",
            "pip",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            *(
                argument
                for platform in ("manylinux2014_x86_64", "manylinux_2_28_x86_64", "any")
                for argument in ("--platform", platform)
            ),
            "--python-version",
            python_version,
            "--dest",
            str(directory),
            *OFFLINE_PACKAGES,
        ],
        check=True,
    )
    return sorted(path.name for path in directory.glob("*.whl"))


def main() -> None:
    """Build the initial checkpoint, the verified code bundle and optional offline wheels."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build verified YOLO pipeline inputs.")
    parser.add_argument("--output-directory", type=Path, default=root / "dist/yolo_native")
    parser.add_argument("--staging-directory", type=Path, default=root / "data/weights")
    parser.add_argument("--offline-assets", action="store_true")
    parser.add_argument("--python-version", default="3.13")
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    LOGGER.info(
        "Initial checkpoint: %s", build_initial_checkpoint(root, arguments.staging_directory)
    )
    bundle = build_bundle(
        root,
        arguments.output_directory / BUNDLE_NAME,
        arguments.staging_directory / INIT_NAME,
    )
    LOGGER.info("Verified bundle: %s", bundle)
    if arguments.offline_assets:
        LOGGER.info(
            "Offline wheels: %s",
            stage_offline_wheels(arguments.output_directory, arguments.python_version),
        )


if __name__ == "__main__":
    main()
