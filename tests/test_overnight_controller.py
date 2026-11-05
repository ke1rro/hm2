import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


def test_invalid_snapshot_preserves_committed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain committed state when a newly downloaded snapshot fails its checksum."""
    path = Path(__file__).resolve().parents[1] / "scripts/run_retinanet_overnight.py"
    specification = importlib.util.spec_from_file_location("overnight_controller", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load the controller fixture.")
    controller = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(controller)
    local = tmp_path / "R0-S"
    local.mkdir()
    (local / "latest.pt").write_bytes(b"known-good-fixture")
    (local / "CURRENT").write_text("previous")
    previous = local / "snapshots" / "previous"
    previous.mkdir(parents=True)
    (previous / "retrieval_manifest.json").write_text("{}")

    def download(arguments: list[str]) -> None:
        destination = Path(arguments[-1])
        if arguments[-2].endswith("ready.json"):
            destination.write_text(
                json.dumps({"generation": "next", "files": {"latest.pt": "not-a-valid-checksum"}})
            )
        else:
            destination.write_bytes(b"corrupt-fixture")

    monkeypatch.setattr(controller, "_cli", download)
    with pytest.raises(ValueError, match="checksum mismatch"):
        controller._retrieve(Namespace(session="fixture"), tmp_path, "R0-S")
    assert (local / "latest.pt").read_bytes() == b"known-good-fixture"
    assert (local / "CURRENT").read_text() == "previous"
