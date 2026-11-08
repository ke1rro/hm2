# Controlled data experiments

Research notebooks 05 through 08 read metadata and measured results with training
disabled by default. The Kaggle execution notebook currently exports submissions for the three completed Faster R-CNN ablations without training. No new results are claimed before execution.

| Experiment                 | Research notebook                                    | Single changed factor                                                  | Detector baseline                     |
| -------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------- |
| `photometric_augmentation` | [05](../notebooks/05_photometric_augmentation.ipynb) | Mild brightness, contrast and saturation with probability 0.30         | Faster R-CNN stride 5                 |
| `input_resolution`         | [06](../notebooks/06_input_resolution.ipynb)         | Resize 800/1280 instead of 640/1024                                    | Faster R-CNN stride 5                 |
| `yolo_baseline`            | [07](../notebooks/07_yolo_baseline.ipynb)            | YOLO11s anchor-free architecture baseline; existing SGD/AMP loop       | Faster R-CNN and RetinaNet references |
| `repeat_factor_sampling`   | [08](../notebooks/08_repeat_factor_sampling.ipynb)   | Fixed-budget image sampling using capped class-presence repeat factors | Faster R-CNN stride 5                 |

The Python modules here contain experiment-specific logic. `protocol.py` only joins
these functions to the existing detector loops and loads saved measurements.
The existing execution notebooks 00 through 02 remain available for the original runs.

## YOLO11s baseline

The unrun ignore-aware RetinaNet proposal has been removed. Completed RetinaNet
experiments and research notebooks remain available for the anchor-based rubric.
YOLO11s is a separate anchor-free architecture baseline, not a replacement label
for old RetinaNet measurements. The three Faster R-CNN data ablations are unchanged.

`yolo_baseline.py` wraps the native eight-class YOLO11s network and CIoU/BCE/DFL
criterion in the existing image/target contract. It reuses the existing SGD, AMP,
sampler, full-validation loop, immutable snapshots and verified Google Drive transport.
Checkpoints are the project's complete `latest.pt`/`best.pt`, including optimizer,
scheduler, scaler, RNG and consumed-batch cursor; they are not native trainer exports.
Mosaic, mixup, HSV, random affine transforms and native optimizer auto-selection do
not run because the native Ultralytics trainer is not called.

Input RGB remains in [0, 1], with 640/1024 aspect-ratio resize and right/bottom padding
114/255 to stride 32. Output boxes are clipped to resized image bounds and transformed
back to original pixels. Native classes 0–7 become evaluator labels 1–8. Negative
candidate points inside ignored regions are masked only in BCE; native positive
assignment, quality normalization, CIoU and DFL remain unchanged. Native loss gains
are box=7.5, cls=0.5, dfl=1.5. This differs from the other architectures' native losses
and normalization, so cross-family differences are not a single-factor ablation.

Ultralytics is pinned to 8.3.235 for the tested criterion contract and to avoid newer
platform integrations. Hosted installation preserves the runtime CUDA PyTorch pair
and existing OpenCV. Locally we depend on opencv-python, which Ultralytics requires,
instead of installing both OpenCV distributions into the same cv2 namespace.
Matplotlib is a transitive library dependency; research charts still use Plotly.

The completed fifth epoch scored validation mAP 0.154990 and Kaggle Public 0.14348,
as observed in the user's 7 November submission screenshot. The Faster R-CNN final
reference scored 0.20369 Public. YOLO validation mAP rose through epoch five; the shared
short budget and SGD schedule warrant investigation before blaming architecture.
A training or inference defect has not been established. We retain this baseline but
exclude it from the new queue; no new YOLO run is started.

Sources: [YOLO11](https://docs.ultralytics.com/models/yolo11/) and
[the pinned loss source](https://github.com/ultralytics/ultralytics/blob/v8.3.235/ultralytics/utils/loss.py).

## Protocol

All ablations use the same 4,868 stride-five training frames, full 2,846-frame
validation, seed 2026, five epochs, batch size 2, original pretrained initialization,
SGD LR 0.0025, momentum 0.9, weight decay 0.0001, original warmup/decays and CUDA AMP.
The baseline horizontal flip probability remains 0.5. Validation is deterministic.
Resizing, normalization, original-pixel prediction postprocessing and evaluators
remain in the detector implementation. RetinaNet retains its original 300-detection
inference cap and canonical COCO evaluation. Faster R-CNN retains its 500-detection
inference cap and both 100/500 maxDets evaluations. Primary comparisons use maxDets 100
within the same detector family.

New runs retain both latest.pt and best.pt. Final-epoch metrics are the primary
comparison. Best-validation metrics are shown separately. The original Faster R-CNN
baseline keeps its existing final-epoch policy, and its best historical epoch metric
does not imply that checkpoint is available. Validation and Public scores can diverge.

RFS starts with an analysis of threshold 0.01, which yields only unit factors here.
Bus is present in 1,333 of the 4,868 selected images, despite its low box-count share.
The configured threshold 0.5 is chosen from training presence only, not validation or
Public feedback, with repeat factors capped at 4. We normalize these factors into
image probabilities and sample exactly 4,868 images per epoch with replacement.
This is a fixed-budget weighted RFS adaptation, not an expanded repeat-factor epoch.
When all factors are one, the exact baseline permutation is used. Resume reconstructs
the epoch order and slices it using the saved consumed-batch cursor.
The nominal budget remains 12,170 batch attempts; AMP can skip optimizer updates,
whose actual count is recorded rather than compensated with extra training.

## Local CPU analysis

Prepare or restore the original metadata using notebook 00. No automatic data
or weight download occurs in the four research notebooks. Launch one independently:

```bash
uv run jupyter lab notebooks/05_photometric_augmentation.ipynb
uv run jupyter lab notebooks/06_input_resolution.ipynb
uv run jupyter lab notebooks/07_yolo_baseline.ipynb
uv run jupyter lab notebooks/08_repeat_factor_sampling.ipynb
```

Use `HM2_DATA_DIR` for an existing extracted dataset, `HM2_METADATA_DIR` for the
original parquet directory, `HM2_BASELINE_RESULTS` for restored baseline records,
and `HM2_DATA_EXPERIMENT_OUTPUT` for one ablation's records. Defaults match the
existing project directories. Missing data or results are displayed explicitly.
The baseline protocol table checks restored configuration and validation coverage.
Do not interpret an incompatible or missing baseline as a verified comparison.
Original JSON records stay on Drive and local copies remain gitignored.

The YOLO baseline is verified with short untrained CPU model and checkpoint tests.
No model initialization is triggered by its default research-notebook execution.

## Kaggle execution, after authorization

1. Build the small code and original-metadata bundle from the repository root:

   ```bash
   uv run python scripts/build_data_experiment_bundle.py
   ```

   The output is `data/exports/controlled_data_submissions_project.zip`. The builder checks CRC
   and byte-for-byte agreement, including original metadata. It excludes secrets,
   weights, images and execution notebooks. Upload this as a separate file to
   `My Drive/hm2_retinanet/inputs/`. Preserve the existing baseline archives.

2. Import [03_data_experiments_kaggle.ipynb](03_data_experiments_kaggle.ipynb) into a
   private Kaggle notebook. Start a fresh session, enable Internet, attach the same
   competition dataset and enable access to secret `HM2_DRIVE_TOKEN`. Follow the
   existing OAuth setup in [AGENTS.md](../AGENTS.md#hosted-experiment-execution).

3. The notebook defaults to this ordered queue:

   ```python
   EXPERIMENTS = (
       "photometric_augmentation",
       "repeat_factor_sampling",
       "input_resolution",
   )
   SHOULD_TRAIN = False
   SHOULD_EXPORT_SUBMISSIONS = True
   SESSION_HOURS = 10.0
   ```

   Keep `BATCH_ID = "controlled_data_stride5_5epochs_seed2026"` for compatible restarts.
   Each experiment retains its own configuration, five-epoch budget, optimizer and
   Drive directory. Completed YOLO11s is excluded. Its code remains for reproducibility.

4. Enable one CUDA GPU, then use Save Version and Save & Run All. The default is
   inference-only: stage all 3,811 test images, restore each completed final checkpoint
   and export one CSV per model. No training or synthetic probes run. Completed CSVs are
   reused when their checkpoint, test metadata and file hashes match.

5. Download files from `/kaggle/working/submissions/`. Each experiment also saves its CSV
   and `submission_metadata.json` beside its checkpoint and into verified Drive snapshots.
   Normal snapshot restore includes these exports, so another session can skip them.
   No submission is sent to Kaggle automatically.

6. For future training set `SHOULD_TRAIN = True` and leave export enabled. The notebook
   trains or resumes each run, then immediately infers test images on the same Kaggle GPU.
   The session deadline covers the full queue. A partial training run stops before export;
   rerun the same batch to continue. Stop other writers before starting a background run.

This change adds Python export code and CSV snapshot support. Upload the new separate
`controlled_data_submissions_project.zip` to `hm2_retinanet/inputs/` and import the updated
execution notebook together. Preserve the existing training code archives.

Local outputs are
`runs/data_experiments/controlled_data_stride5_5epochs_seed2026/<experiment>/`.
Durable outputs use the same run directory under
`My Drive/hm2_retinanet/batches/controlled_data_stride5_5epochs_seed2026/`.

Checkpoints contain model, optimizer, scheduler, AMP scaler, RNG, exact recipe,
epoch, batch cursor, successful updates, image exposures and history. Complete
snapshots and measured JSON are verified on Drive synchronously. A storage failure
stops execution. A paused run resumes pending validation and skips completed work.
Keep final and validation-selected results distinct when comparing experiments.
The bundle and uploaded notebook must both be updated after relevant source changes.

## Verification and limitations

Use the existing project checks plus `uv run flake8 experiments/`. Focused tests are
in `tests/test_data_experiments.py` and `tests/test_yolo_baseline.py`; hosted setup cells are exercised with simulated
HTTP and real local checkpoint serialization in `tests/test_drive_api.py`.

GPU throughput, memory feasibility and ablation quality remain unmeasured until an
explicit run. Batch size and accumulation are not adjusted automatically. Timing depends
on runtime hardware. There is one seed and seven validation videos; class AP, mAP and
size-specific AP must be assessed together. Historical GPU memory may be unavailable.
The YOLO loss adapter is version-sensitive. It checks the pinned library version;
CPU tests verify its native-loss equivalence and unchanged regression before GPU execution.

RFS follows the class-presence idea in the
[Detectron2 sampler](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/distributed_sampler.py)
with the fixed-length adaptation described above. Photometric adjustments and detector
coordinate transforms use TorchVision directly.
