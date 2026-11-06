import argparse
import logging
import sys
import tomllib
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Copy the configured tiny-mode frames from the competition data without extraction."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hm2.tiny_subset import prepare_tiny_subset

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Competition ZIP or root.")
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/faster_rcnn_tiny.toml",
    )
    arguments = parser.parse_args()
    settings = tomllib.loads(arguments.config.read_text())["dataset"]
    root = prepare_tiny_subset(
        arguments.source,
        arguments.destination,
        {split: settings[f"{split}_frames"] for split in ("train", "val", "test")},
    )
    LOGGER.info("Smoke subset ready at %s", root)


if __name__ == "__main__":
    main()
