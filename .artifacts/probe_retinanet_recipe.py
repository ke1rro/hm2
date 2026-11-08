import json
import logging
import sys
from pathlib import Path

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
sys.path.insert(0, str(workspace / "src"))
from hm2.detection_data import DetectionDataset, make_loader
from hm2.retinanet import build_retinanet, make_optimizer, train_steps

root = workspace / "data"
settings = json.loads((root / "settings.json").read_text())["training"]
device = torch.device("cuda")
dataset = DetectionDataset(
    root,
    pl.read_parquet(root / "train_frames.parquet"),
    pl.read_parquet(root / "objects.parquet"),
    settings["flip_probability"],
)
model = build_retinanet(
    "R0", min_size=settings["min_size"], max_size=settings["max_size"], seed=settings["seed"]
).to(device)
optimizer, scheduler = make_optimizer(
    model, settings["learning_rate"], settings["max_steps"], settings["warmup_steps"]
)
loader = make_loader(
    dataset, settings["batch_size"], settings["num_workers"], device, True, settings["seed"]
)
torch.cuda.reset_peak_memory_stats()
history = train_steps(model, loader, optimizer, scheduler, device, 2, max_attempts=4)
measured = history[1:]
throughput = len(measured) * settings["batch_size"] / sum(row["seconds"] for row in measured)
result = {
    "status": "Diagnostic only, R0 recipe timing, not a graded experiment",
    "run": "R0",
    "device": torch.cuda.get_device_name(),
    "training_config": settings,
    "history": history,
    "warmup_attempts": 1,
    "measured_attempts": len(measured),
    "optimizer_updates": sum(row["optimizer_updated"] for row in history),
    "training_images_per_second": throughput,
    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    "estimated_training_minutes_per_run": settings["max_steps"] * settings["batch_size"] / throughput / 60,
    "estimate_limitations": "One measured R0 batch after warmup on four real frames. Excludes validation, checkpoint saves, and AMP retries. Clip/image shapes, density, and decoding vary. R1/R2 are not timed at recipe resize. This is a noisy planning estimate, not a measured full-run duration.",
}
(workspace / "results/recipe_probe.json").write_text(json.dumps(result, indent=2) + "\n")
logging.info("Recipe probe: %s", json.dumps(result))
del model, optimizer, scheduler, loader
torch.cuda.empty_cache()
