import logging
import sys
from pathlib import Path
from textwrap import dedent

import nbformat

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
cells = []


def markdown(source: str) -> None:
    cells.append(nbformat.v4.new_markdown_cell(dedent(source).strip()))


def code(source: str) -> None:
    cells.append(nbformat.v4.new_code_cell(dedent(source).strip()))


markdown("""
# RetinaNet experiments in Google Colab

We execute the existing detector and epoch runner through the native Colab notebook.
No Colab CLI, external experiment service, or Mac connection is required. We preserve
the shared seed 2026, batch size 2, resize 640/1024, SGD, focal classification and GIoU
regression. The official validation split has 2,846 frames and canonical COCO maxDets=100.

| Run | Anchors | Train selection | Epochs |
| --- | --- | --- | --- |
| R0-S | Original | Every fifth frame within each clip | 5 |
| R1-S | Existing smaller sizes | Same stride-5 selection | 5 |
| R0-F | Original | All 24,201 train frames | 5 |
| R1-F | Existing smaller sizes | All 24,201 train frames | 5 |

Five full-data epochs expose roughly five times as many images as five subset epochs.
Differences compare training recipes and do not isolate diversity from update count.
Ordinary GT remains intact inside ignored regions. Evaluation retains original COCO
crowds. Stock training does not suppress ignored-area negative anchors, and the exact
competition ignore semantics remain unverified. No test-set tuning is performed.

## 1. Open and prepare

Open this file through **colab.research.google.com → File → Upload notebook**. Select
**Runtime → Change runtime type → GPU**, then run the setup cells in order. You complete
the interactive Google Drive authorization yourself.

Upload the two prepared files from the Mac's `data/exports/` into
**My Drive/hm2_retinanet/inputs/**:

* `retinanet_colab_project.zip`: the current `src/hm2/` code, TOML configuration and
  prepared metadata. This includes uncommitted code, so cloning an older repository
  revision alone is insufficient.
* `visdrone_train_val.zip`: cached official train and validation images plus original
  validation annotations, about 9.1 GiB. No dataset redownload is required.

The Mac paths are not accessible to Colab. These uploads are required on the first run.
You can instead set `DATASET_ROOT` to a matching extracted copy already in Colab. Use
fast `/content` storage for images. Keep several GiB of Drive quota free for checkpoints.
""")
code("""
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import platform
import shutil
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path
from zipfile import ZipFile

if sys.version_info < (3, 12):
    raise RuntimeError("Use a current Colab Python runtime with Python 3.12 or newer.")

requirements = {
    "polars": "polars>=1.31",
    "plotly": "plotly>=6",
    "pycocotools": "pycocotools>=2.0.11",
    "decouple": "python-decouple>=3.8",
    "PIL": "Pillow>=11",
    "tqdm": "tqdm>=4.67",
    "ipywidgets": "ipywidgets>=8",
}
missing = [package for module, package in requirements.items() if importlib.util.find_spec(module) is None]
if missing:
    uv = shutil.which("uv")
    if uv is None:
        installer = Path("/content/install_uv.sh")
        urllib.request.urlretrieve("https://astral.sh/uv/install.sh", installer)
        subprocess.run(["sh", str(installer)], check=True)
        uv = str(Path.home() / ".local/bin/uv")
    subprocess.run([uv, "pip", "install", "--system", "--python", sys.executable, *missing], check=True)

import polars as pl
import torch
import torchvision
from IPython.display import display

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
LOGGER = logging.getLogger("retinanet_colab")
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU runtime before continuing.")
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False
pl.from_dicts([{
    "Python": platform.python_version(),
    "PyTorch": torch.__version__,
    "TorchVision": torchvision.__version__,
    "CUDA": torch.version.cuda,
    "GPU": torch.cuda.get_device_name(),
    "Free VRAM GiB": torch.cuda.mem_get_info()[0] / 1024**3,
}])
""")
markdown("""
We retain Colab's paired CUDA PyTorch and TorchVision installation. We install only
missing auxiliary packages with uv, using its [official installer](https://docs.astral.sh/uv/getting-started/installation/)
when necessary. We do not apply the local Python 3.14 lock to the hosted kernel.
The prior T4 verification used Python 3.13.16, torch 2.11.0+cu130 and torchvision
0.26.0+cu130. The table above records the actual runtime used here.

## 2. Mount Drive and choose paths

Keep the same `BATCH_ID` when resuming. Choose a new ID only for an independent batch.
The next cell calls native `google.colab.drive.mount()` and may request authorization.
""")
code("""
drive = importlib.import_module("google.colab.drive")
drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/hm2_retinanet")
PROJECT_ARCHIVE = DRIVE_ROOT / "inputs/retinanet_colab_project.zip"
DATASET_ARCHIVE = DRIVE_ROOT / "inputs/visdrone_train_val.zip"
PROJECT_ROOT = Path("/content/hm2")
DATASET_ROOT = Path("/content/visdrone")
BATCH_ID = "retinanet_stride5_full_5epochs_seed2026"
LOCAL_BATCH = PROJECT_ROOT / "runs/retinanet_overnight" / BATCH_ID
DRIVE_BATCH = DRIVE_ROOT / "batches" / BATCH_ID
LOCAL_BATCH.mkdir(parents=True, exist_ok=True)
DRIVE_BATCH.mkdir(parents=True, exist_ok=True)
{"Local outputs": str(LOCAL_BATCH), "Durable outputs": str(DRIVE_BATCH)}
""")
markdown("""
## 3. Stage project and existing dataset

The project archive supplies the existing implementation, configuration and prepared
metadata. We copy the dataset ZIP from Drive to local storage before extraction.
An already extracted matching dataset is reused. The first staging pass requires about
20 GiB of free Colab disk, including the temporary ZIP, which is deleted after extraction.
""")
code("""
if not PROJECT_ARCHIVE.is_file():
    raise FileNotFoundError(f"Upload retinanet_colab_project.zip to {PROJECT_ARCHIVE.parent}.")
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
with ZipFile(PROJECT_ARCHIVE) as archive:
    for entry in archive.infolist():
        if not (PROJECT_ROOT / entry.filename).resolve().is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError("Project archive contains a path outside the project directory.")
    archive.extractall(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hm2.colab_drive import (
    restore_checkpoint_from_drive,
    sync_checkpoint_to_drive,
    sync_results_to_drive,
    verify_drive_checkpoint,
)
from hm2.data import extract_archive
from hm2.detection_data import DetectionDataset, select_frames
from hm2.retinanet_analysis import summarize_batch
from hm2.retinanet import build_retinanet
from hm2.retinanet_experiments import run_chunk

roots = [
    path.parent.parent for path in DATASET_ROOT.rglob("val/annotations_coco.json")
    if (path.parent.parent / "train/images").is_dir()
]
if not roots:
    if not DATASET_ARCHIVE.is_file():
        raise FileNotFoundError(f"Upload visdrone_train_val.zip to {DATASET_ARCHIVE.parent}.")
    with ZipFile(DATASET_ARCHIVE) as archive:
        extracted_bytes = sum(entry.file_size for entry in archive.infolist())
    if shutil.disk_usage("/content").free < extracted_bytes + DATASET_ARCHIVE.stat().st_size + 4 * 1024**3:
        raise OSError("Insufficient Colab disk for dataset staging and checkpoints.")
    local_archive = Path("/content/visdrone_train_val.zip")
    LOGGER.info("Copy dataset archive from Drive, then extract to %s", DATASET_ROOT)
    shutil.copyfile(DATASET_ARCHIVE, local_archive)
    extract_archive(local_archive, DATASET_ROOT)
    local_archive.unlink()
    roots = [
        path.parent.parent for path in DATASET_ROOT.rglob("val/annotations_coco.json")
        if (path.parent.parent / "train/images").is_dir()
    ]
if len(roots) != 1:
    raise ValueError("Set DATASET_ROOT to the directory containing exactly one train/val dataset.")
DATASET_ROOT = roots[0]
METADATA = PROJECT_ROOT / "data/metadata"
{"Dataset": str(DATASET_ROOT), "Metadata": str(METADATA)}
""")
markdown("""
## 4. Verify inputs, recipe and Drive checkpoint storage

We check all required image paths and decode one real image/target pair from each split.
The tiny CPU probe writes a real model/optimizer/scheduler/scaler checkpoint locally,
copies it to mounted Drive, verifies the copy, and restores its state before training.
The real experiment's initial checkpoint is also synchronized before its first update.

Ordinary source annotations, class IDs and image coordinates use the existing code.
Five epochs, original GIoU, preprocessing, augmentations and validation stay unchanged.
""")
code("""
frames = pl.read_parquet(METADATA / "frames.parquet")
objects = pl.read_parquet(METADATA / "objects.parquet")
train_frames = select_frames(frames, "train")
val_frames = select_frames(frames, "val")
settings = tomllib.loads((PROJECT_ROOT / "config/retinanet_overnight.toml").read_text())
subset = select_frames(frames, "train", settings["subset_stride"])
if (
    train_frames.height != 24201 or val_frames.height != 2846
    or train_frames["clip"].n_unique() != 56 or val_frames["clip"].n_unique() != 7
    or subset["clip"].n_unique() != 56 or settings["subset_stride"] != 5 or settings["epochs"] != 5
):
    raise ValueError("The staged inputs or recipe do not match the four planned experiments.")
if set(train_frames["clip"]) & set(val_frames["clip"]):
    raise ValueError("Train and validation clips overlap.")
required_frames = pl.concat([train_frames, val_frames])
missing_images = [path for path in required_frames["relative_path"] if not (DATASET_ROOT / path).is_file()]
if missing_images:
    raise FileNotFoundError(f"Missing {len(missing_images)} train/val images, including {missing_images[:3]}.")
for selected in (train_frames, val_frames):
    image, target = DetectionDataset(DATASET_ROOT, selected.head(1), objects)[0]
    LOGGER.info("Verified %s: image %s, target boxes %s", selected["split"][0], list(image.shape), list(target["boxes"].shape))
inputs = json.loads((METADATA / "colab_inputs.json").read_text())
with (DATASET_ROOT / "val/annotations_coco.json").open("rb") as stream:
    actual_annotations_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
if actual_annotations_sha256 != inputs["validation_annotations_sha256"]:
    raise ValueError("The original validation annotations differ from the prepared metadata bundle.")

dataset_identity = hashlib.sha256()
for path in (
    METADATA / "frames.parquet", METADATA / "objects.parquet",
    METADATA / "ignored_regions.parquet", DATASET_ROOT / "val/annotations_coco.json",
):
    dataset_identity.update(path.read_bytes())
selection_identities = {}
for stride in (5, 1):
    identity = dataset_identity.copy()
    identity.update(json.dumps(select_frames(frames, "train", stride)["image_key"].to_list()).encode())
    selection_identities[stride] = identity.hexdigest()
anchor_reference = build_retinanet("R0", pretrained=False, min_size=64, max_size=96)
baseline_sizes = anchor_reference.anchor_generator.sizes
baseline_ratios = anchor_reference.anchor_generator.aspect_ratios
del anchor_reference

EXPERIMENTS: list[dict] = [
    {
        **{key: value for key, value in settings.items() if key != "subset_stride"},
        "experiment_id": name,
        "variant": name[:2],
        "frame_stride": 5 if name.endswith("S") else 1,
        "warmup_epochs": 100 / math.ceil(subset.height / settings["batch_size"]),
        "dataset_selection_sha256": selection_identities[5 if name.endswith("S") else 1],
        "anchor_sizes": [
            [size // 2 if name.startswith("R1") else size for size in level]
            for level in baseline_sizes
        ],
        "anchor_aspect_ratios": [list(level) for level in baseline_ratios],
    }
    for name in ("R0-S", "R1-S", "R0-F", "R1-F")
]
configuration_path = DRIVE_BATCH / "batch_config.json"
if configuration_path.exists() and json.loads(configuration_path.read_text()) != EXPERIMENTS:
    raise ValueError("This BATCH_ID has another recipe. Restore that configuration or use a new ID.")
configuration_text = json.dumps(EXPERIMENTS, indent=2) + "\\n"
temporary = configuration_path.with_suffix(".incoming")
temporary.write_text(configuration_text)
temporary.replace(configuration_path)
if configuration_path.read_text() != configuration_text:
    raise ValueError("Drive configuration write/read verification failed.")
(LOCAL_BATCH / "batch_config.json").write_text(configuration_text)
probe = verify_drive_checkpoint(LOCAL_BATCH / "_probe", DRIVE_BATCH / "_probe")
LOGGER.info("Drive checkpoint write/read/restore verified: %s", probe["sha256"])
pl.from_dicts([
    {
        "Run": item["experiment_id"],
        "Anchors": "Original" if item["variant"] == "R0" else "Smaller",
        "Train frames": subset.height if item["frame_stride"] == 5 else train_frames.height,
        "Epochs": item["epochs"],
        "Batch size": item["batch_size"],
        "Resize": f"{item['min_size']}/{item['max_size']}",
    } for item in EXPERIMENTS
])
""")
markdown("""
## 5. Run experiments

**This is the only training cell.** Execute it manually after setup passes. It loads
the official COCO weights into the Torch cache if absent. All four runs start from
that same pretrained initialization. Compatible existing checkpoints are restored first.
Completed runs are skipped. One experiment runs at a time, in the table's order.

Each checkpoint is saved to local Colab storage first. A synchronous callback copies
the complete immutable snapshot to Drive and checks every destination SHA256 before
publishing `CURRENT.json`. The preceding verified generation remains until replacement
is confirmed. Saves occur every approximately 10 minutes, each epoch and completion.
A Drive error stops the run rather than silently continuing without durable persistence.

Full validation after every epoch selects `best.pt` by highest canonical mAP. `latest.pt`
contains resumable training progress. Final metrics, per-class AP, logs, comparisons,
curves and consistent previews are synchronized after every completed experiment.
""")
code("""
for experiment in EXPERIMENTS:
    name = experiment["experiment_id"]
    local_run = LOCAL_BATCH / name
    drive_run = DRIVE_BATCH / name
    progress = restore_checkpoint_from_drive(drive_run, local_run, experiment)
    has_completed = (
        progress.get("epochs_completed") == experiment["epochs"]
        and not progress.get("pending_evaluation", False)
    )
    if has_completed:
        LOGGER.info("Skip completed experiment %s", name)
    else:
        LOGGER.info("Start or resume %s", name)

        def persist_checkpoint(source: Path) -> None:
            sync_checkpoint_to_drive(source, drive_run)

        while True:
            status = run_chunk(
                DATASET_ROOT, METADATA, local_run, experiment, resume=True,
                seconds=experiment["checkpoint_seconds"], on_checkpoint=persist_checkpoint,
            )
            display(pl.from_dicts([{
                "Run": name, "Epochs completed": status["epochs_completed"],
                "Batch cursor": status["batch_cursor"], "Updates": status["global_step"],
                "Image exposures": status["image_exposures"], "Completed": status["completed"],
            }]))
            if status["stopped"]:
                raise InterruptedError(f"Remove {local_run / 'STOP'} before resuming.")
            if status["completed"]:
                break
    summary = summarize_batch(LOCAL_BATCH, DATASET_ROOT, METADATA)
    sync_results_to_drive(LOCAL_BATCH, DRIVE_BATCH)
    display(pl.from_dicts(summary["comparison"], infer_schema_length=None).select(
        "experiment_id", "status", "epoch", "train_frames", "global_step",
        "map", "ap50", "ap_small", "train_seconds",
    ))
LOGGER.info("All four experiments completed and synchronized to %s", DRIVE_BATCH)
""")
markdown("""
## 6. Monitor and resume

Monitor the Run cell's checkpoint and validation messages while it is active. Full
validation and Drive copying can take several minutes between progress updates.
In Drive, each experiment's `CURRENT.json` identifies its verified
`generations/<generation>/` directory. Open `status.json`, `execution.log`, `metrics.json`,
`latest.pt` or `best.pt` there. Summary tables, plots and previews live under
`summary/generations/`, identified by `summary/CURRENT.json`.

To resume after a runtime reset, select GPU, rerun setup and the training cell with the
**same BATCH_ID and recipe**. Cached `/content` files may be gone, so staging repeats
from the archives already on Drive. The latest verified compatible checkpoint restores
model, optimizer, scheduler, AMP scaler, epoch, batch cursor, history and RNG state.
No completed epoch or completed experiment is intentionally retrained. Work since the
last successfully synchronized checkpoint can be repeated. CUDA is not guaranteed
bit-for-bit reproducible. Browser disconnection and Colab runtime limits can still stop
execution, so completion overnight is not promised.

The cell below only reads saved progress. Execute it after interruption or between runs.
Read the live Run cell output or open the Drive files while the kernel is training.
The tuned Faster R-CNN comparison remains an integration point until compatible
measured teammate results are available. Metric differences are descriptive, not claims
of statistical significance.
""")
code("""
rows = []
for name in ("R0-S", "R1-S", "R0-F", "R1-F"):
    pointer = DRIVE_BATCH / name / "CURRENT.json"
    if pointer.is_file():
        committed = json.loads(pointer.read_text())
        directory = pointer.parent / "generations" / committed["generation"]
        status = json.loads((directory / "status.json").read_text())
        rows.append({
            "Run": name, "Epochs completed": status["epochs_completed"],
            "Updates": status["global_step"], "Completed": status["completed"],
            "Drive generation": committed["generation"],
        })
    else:
        rows.append({"Run": name, "Epochs completed": None, "Updates": None, "Completed": False, "Drive generation": "Not run"})
pl.from_dicts(rows)
""")

notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
    "accelerator": "GPU",
    "colab": {"name": "03_retinanet_experiments_colab.ipynb"},
}
nbformat.validate(notebook)
for index, cell in enumerate(cells):
    if cell.cell_type == "code":
        compile(cell.source, f"Colab cell {index}", "exec")
nbformat.write(notebook, "notebooks/03_retinanet_experiments_colab.ipynb")
logging.info("Created Colab entry point with %s code cells", sum(c.cell_type == "code" for c in cells))
