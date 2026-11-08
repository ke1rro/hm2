import json
import logging
import platform
import sys
from pathlib import Path
from zipfile import ZipFile

import polars as pl
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
workspace = Path("/content/hm2_retinanet_smoke")
with ZipFile("/content/hm2_retinanet_smoke.zip") as archive:
    archive.extractall(workspace)
sys.path.insert(0, str(workspace / "src"))

from hm2.retinanet import run_smoke_test

root = workspace / "data"
settings = json.loads((root / "settings.json").read_text())
summary = run_smoke_test(
    root,
    pl.read_parquet(root / "train_frames.parquet"),
    pl.read_parquet(root / "val_frames.parquet"),
    pl.read_parquet(root / "objects.parquet"),
    settings["training"],
    settings["smoke"],
    workspace / "results",
    allow_download=True,
)
summary["python"] = platform.python_version()
summary["access"] = "Official colab CLI 0.7.5, colab exec"
(workspace / "results/smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with ZipFile("/content/hm2_retinanet_smoke_results.zip", "w") as archive:
    for path in (workspace / "results").glob("*.json"):
        archive.write(path, path.name)
logging.info(
    "Smoke finished: %s optimizer updates, runs=%s",
    summary["total_optimizer_steps"],
    [result["run"] for result in summary["runs"]],
)
