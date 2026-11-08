import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hm2.experiment_checkpoint import write_json
from hm2.experiment_names import EXPERIMENTS


def publish_csv(name: str, cache: Path, records: dict) -> None:
    source = cache / "submission_unverified.csv"
    source_metadata = json.loads((cache / "submission_metadata.json").read_text())
    checkpoint = records["runs"][name]["source_files"]["best.pt"]
    if source_metadata["checkpoint_sha256"] != checkpoint["sha256"]:
        raise ValueError(f"Unexpected checkpoint for {name}")
    destination = ROOT / "submissions" / f"{name}.csv"
    temporary = destination.with_suffix(".csv.incoming")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    temporary.replace(destination)
    metrics = json.loads((RESULTS / name / "metrics.json").read_text())
    best = max(metrics, key=lambda row: row["metrics"]["map"])
    record = {
        "experiment_id": name,
        "selected_epoch": best["epoch"],
        "selection_criterion": "Highest full-validation COCO mAP",
        "validation_map": best["metrics"]["map"],
        "checkpoint": checkpoint,
        "csv": {"path": destination.name, "sha256": digest, "size": destination.stat().st_size},
        "validation": {
            key: source_metadata[key]
            for key in ("images", "detections", "empty_images", "official_sample_verified")
        },
        "inference": source_metadata["inference"],
        "box_format": "Original-image pixel top-left xywh",
        "class_ids": list(range(8)),
        "max_detections_per_image": 500,
    }
    write_json(destination.with_suffix(".metadata.json"), record)
    logging.info("Published %s: %s", destination, record["validation"])


RESULTS = ROOT / "results/retinanet/retinanet_stride5_full_5epochs_seed2026"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    records = json.loads((RESULTS / "provenance.json").read_text())
    (ROOT / "submissions").mkdir(exist_ok=True)
    for name in EXPERIMENTS:
        cache = ROOT / "runs/submissions" / name
        if name == "standard_anchors_stride5":
            cache = ROOT / "runs/submissions/R0-S"
        logging.info("Start or reuse test inference: %s", name)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_retinanet_submission.py"),
                "--checkpoint",
                str(cache / "best.pt"),
                "--output",
                str(cache),
            ],
            cwd=ROOT,
            check=True,
        )
        publish_csv(name, cache, records)
    (ROOT / "runs/submissions/EXPORT_COMPLETED").write_text("All four CSVs exported.\n")
    logging.info("All four submissions completed")


if __name__ == "__main__":
    main()
