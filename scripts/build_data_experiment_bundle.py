import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

LOGGER = logging.getLogger(__name__)


def build_bundle(root: Path, output: Path) -> dict:
    """Build and verify a code bundle with byte-identical prepared metadata.

    Args:
        root: Repository root with src, experiments, config and prepared parquet metadata.
        output: Local ZIP destination; no cloud files are changed.

    Returns:
        ZIP path, size and entry count after CRC and source-byte verification.
    """
    paths = sorted((root / "src/hm2").glob("*.py"))
    paths += sorted((root / "experiments").glob("*.py"))
    paths += sorted((root / "config").glob("*.toml"))
    paths += [
        root / f"data/metadata/{name}.parquet" for name in ("frames", "objects", "ignored_regions")
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Prepare the original metadata before bundling: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".incoming")
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("bundle_manifest.json", json.dumps(hashes, sort_keys=True))
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Bundle CRC verification failed.")
        for path in paths:
            name = path.relative_to(root).as_posix()
            if archive.read(name) != path.read_bytes():
                raise ValueError(f"Bundle differs from current source: {name}")
    temporary.replace(output)
    return {"file": str(output), "bytes": output.stat().st_size, "entries": len(hashes)}


def main() -> None:
    """Build the local code ZIP for the controlled-data Kaggle entry notebook."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build verified controlled-data experiment inputs."
    )
    parser.add_argument(
        "--output", type=Path, default=root / "data/exports/controlled_data_submissions_project.zip"
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    LOGGER.info("Verified bundle: %s", build_bundle(root, arguments.output))


if __name__ == "__main__":
    main()
