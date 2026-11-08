# RetinaNet checkpoint, 2026-11-03

## Implementation and decisions

- Combined both RetinaNet prompt specifications. No full training, final pipeline,
  report authoring, branches, staging, commits, or pushes.
- Notebook `notebooks/02_retinanet.ipynb` presents R0 COCO-pretrained V2, R1 half-size
  anchors only, and R2 DIoU only. All use the same seed 2026 and initialization.
- Shared planned recipe: every eighth frame per train clip, 3050 frames / 56 clips,
  full official validation 2846 frames / 7 clips, 3000 successful updates, SGD
  momentum 0.9, weight decay 0.0001, LR 0.0025, 100-update warmup, decay at
  updates 2000 and 2700, batch size 2, resize 640/1024, horizontal flip 0.5.
- TorchVision detector weights load before replacing the class output head. The
  compatible pretrained class tower and regression weights are preserved. Head
  classes 9, foreground labels 1 through 8, reserved prediction label 0 removed.
- Ordinary GT remains intact even within ignored regions. The evaluator uses
  original category-specific COCO crowd records and maxDets [1, 10, 100]. Stock
  training does not suppress negative anchors in ignored areas. Exact competition
  custom-scoring semantics remain unresolved.
- Lazy image loading and precomputed target lookup avoid per-item table scans.
  CUDA uses fp16 autocast/scaling, pinning/nonblocking transfers, two persistent
  workers and prefetch 2. No compilation, channels-last, or optimizer ablation.
- Independent review found and verified fixes for AMP skips incorrectly consuming
  update budgets and stale results being compared after training-config changes.
- Checkpoints restore model, optimizer, scheduler, scaler and RNG states. Resume
  restarts data iteration; worker prefetch/augmentation are not replayed bit for bit.

## Measured Colab execution

Used only official Colab CLI 0.7.5 after the user's CLI-only instruction. Session
`hm2-retinanet-smoke` was created on a Tesla T4 and stopped after retrieving results.
Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0 are the
documented hosted-GPU exception to the local Python 3.14 environment.

Commands included `colab sessions`, `colab new --session hm2-retinanet-smoke --gpu T4`,
`colab upload`, `colab exec --session hm2-retinanet-smoke --file ...`, `colab download`,
and `colab stop --session hm2-retinanet-smoke`. No full dataset download. Uploaded
3.34 MiB containing source and four real train plus four real validation images,
selected metadata, and all original annotations/crowds for those validation images.
User explicitly authorized the official 146 MiB COCO detector weight download.

Main smoke: 2 successful updates each for R0/R1/R2, 6 attempts total, no AMP skips.
Resize 320/512, batch size 1, inference four real validation frames per variant.
All losses finite, parameters changed, original-pixel prediction bounds valid,
foreground labels in 1..8 and real COCO evaluation completed.

| Run | Training images/s | Inference ms/frame | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| R0 | 7.679 | 135.8 | 0.692 | 1.078 |
| R1 | 12.750 | 150.2 | 0.692 | 1.078 |
| R2 | 12.063 | 175.5 | 0.692 | 1.078 |

Training throughput uses one measured step after one warmup step per run. Inference
includes decode, collation, transfer, model preprocessing/forward/NMS/postprocessing,
and CPU outputs, with CUDA synchronization and one untimed first-batch warmup.
These noisy timings cannot establish an optimization benefit or model ranking.
All three tiny-subset diagnostic AP metrics were actually zero after two updates
of newly initialized class outputs. They are not graded results or useful quality evidence.

The decode-only benchmark measured three images after first-batch startup:
workers 0: 16.25 images/s, startup 0.138 s; workers 2: 36.72 images/s, startup
0.229 s. Too few frames to establish a general improvement.

Separate R0 recipe probe: two further successful updates, resize 640/1024, batch
size 2. Total diagnostic budget actually consumed: **8 updates, 8 attempts**.
One measured batch after warmup: 7.402 images/s, 1.712 GiB peak allocated,
1.875 GiB peak reserved. At that observed rate, 3000 updates of batch size 2
would take about 13.51 training minutes. This is only a noisy four-frame planning
estimate, excluding validation, checkpoint writes, AMP retries, and variation in
image dimensions/density. R1/R2 were not timed at the graded resize.

Machine-readable outputs: `data/experiments/retinanet/smoke/`, including
`smoke_summary.json`, three prediction/diagnostic JSON pairs, and `recipe_probe.json`.
No graded `R0/R1/R2/result.json` exists. All full experiments remain **Not run**.

## Verification

- `uv run --locked pytest -q`: 10 passed in 5.92 seconds, including intentional
  scaler overflow, successful-update accounting, scaler checkpoint restoration,
  subsequent resumed optimization, actual CPU DIoU backward and source-pixel inference.
- `uv run --locked isort --check-only .`: passed.
- `uv run --locked black --check .`: passed.
- `uv run --locked flake8 src/ tests/`: passed. No scripts directory exists.
- `uv run --locked ty check`: passed.
- Notebook JSON/schema and every Python cell compiled. Fresh local Jupyter kernel
  ran all 16 code cells with smoke, recipe timing, graded training, and download
  flags false. Real metadata/image contracts and saved diagnostics/gallery rendered.
  No notebook errors. Two image pairs cover dense, tiny, and missed-GT examples.
- `uv run --locked pre-commit run --files` on all seven changed checkpoint files:
  passed. Full `--all-files`: only existing `report.md` Markdown formatting fails.
  The report was not modified.
- Existing user modifications in `notebooks/01_eda.ipynb` and unrelated files left intact.

The final notebook defaults all execution/download switches to false. It contains
explicit smoke, two-update recipe timing, and later-authorized graded-run calls.
