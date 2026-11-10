# Submission CSVs

We export one CSV for each completed experiment, selecting its checkpoint by the
highest full-validation COCO mAP:

| CSV                            | Training selection | Anchors  | Selected epoch |
| ------------------------------ | ------------------ | -------- | -------------: |
| `standard_anchors_stride5.csv` | Every fifth frame  | Standard |              5 |
| `smaller_anchors_stride5.csv`  | Every fifth frame  | Smaller  |              5 |
| `standard_anchors_full.csv`    | Full training set  | Standard |              1 |
| `smaller_anchors_full.csv`     | Full training set  | Smaller  |              4 |

Each file has `image_id,PredictionString` and covers all 3,811 test images.
Prediction groups use `class score x y width height`, with class IDs 0 to 7 and
pixel top-left coordinates. Empty predictions are allowed. The exporter clips
boxes to image bounds, sorts by score and retains at most 500 valid detections.
We use the trained detector's existing inference settings without test-set tuning.

The CSVs, their metadata and `manifest.json` are stored together in the
[Drive submission batch](https://drive.google.com/drive/folders/1NBGbuvGnWHgeikhZFh5KPa1X1gRfuL-F).
The local CSVs are gitignored. Generated metadata is kept on Drive, with checksums,
selected checkpoints, validation scores and measured inference timing.

Training measurements and source checkpoint IDs are in the
[Drive result archive](https://drive.google.com/file/d/1e5AybR4U8T-SMt2nJ8XcipM_2ye5COUn/view).
The [results notebook](../notebooks/03_retinanet_results.ipynb) contains the measured
tables and plots, plus instructions for downloading the records when rerunning.

To regenerate a CSV, download that experiment's `best.pt` using the file ID and
SHA256 in the results provenance, then run the existing exporter. For example:

```bash
uv run python scripts/run_retinanet_submission.py \
  --checkpoint runs/submissions/smaller_anchors_full/best.pt \
  --output runs/submissions/smaller_anchors_full
```

Copy the exported CSV into this directory using the experiment name above.
The exporter reuses predictions when their checkpoint hash and test-frame list
match. Supply `--sample-submission /path/to/sample_submission.csv` when the
official template is available. Current exports validate image membership
against `test_images.csv` and the supplied competition format; the API returns
404 for the official sample. `official_sample_verified` records that distinction.

The CSVs are generated deliverables. No Kaggle submission is made automatically.

## Faster R-CNN

`faster_rcnn_baseline_stride5.csv` contains the completed ignore-aware baseline's
final-epoch predictions. We restored the original Kaggle export from its
[verified Drive summary generation](https://drive.google.com/drive/folders/11-kfw9cFvQkfI4KdmAfDTGWmLy0Ha-hj)
and checked its SHA256 against `CURRENT.json`. The file covers all 3,811 test
images in `test_images.csv` order, with 287,745 detections and at most 142 per image.
We validated class IDs, scores and pixel-coordinate bounds. The official sample
was unavailable locally, so `official_sample_verified` remains false.

The checkpoint is epoch 5, selected as the configured final epoch. Training used
4,868 frames, every fifth frame within each clip. The
[results notebook](../notebooks/04_faster_rcnn_results.ipynb) reads the downloaded
records under `runs/faster_rcnn/faster_rcnn_stride5_5epochs_seed2026/`. These records,
weights and CSVs are gitignored; original records remain on Drive.

### Epoch-one replication

`faster_rcnn_baseline_stride5_epoch1.csv` contains predictions from the separate
`faster_rcnn_stride5_epoch1_recovery_seed2026` run, paused after the first epoch
and full validation. We retain the original five-epoch learning-rate schedule,
batch size 2, seed 2026 and every-fifth-frame training selection. This is a new
replication, not the deleted epoch-one checkpoint from the original run.

The new checkpoint has 2,433 optimizer updates and 14.48 minutes of training.
Full validation on 2,846 images gives COCO mAP 0.194534, AP50 0.410924 and
AP small 0.086361 at maxDets 100. Its Kaggle score has not been measured.

We restored the CSV from the
[verified Drive summary generation](https://drive.google.com/drive/folders/1-j0PGOrdLlnhk5c5FQWqCT2YmMcLWPyc).
It covers all 3,811 test images in `test_images.csv` order, with 377,822 detections,
at most 206 per image and no empty predictions. Image membership, classes,
scores and original-image coordinate bounds were validated. The official sample
was unavailable locally, so `official_sample_verified` remains false.

The CSV SHA256 is
`3afc51092471eb50e4cbb3064580dd94d86a2ab8f353defc9cca26443f2343d7`.
The immutable summary manifest and checkpoint pointer agree on the export
metadata and epoch-one records. Source JSON and weights remain on Drive.

## YOLO11s

`yolo_baseline_stride5.csv` contains the completed YOLO11s baseline's final epoch-five
predictions. This epoch also has the highest validation mAP in the five-epoch run:
mAP 0.154990, AP50 0.297651 and AP small 0.092178 at maxDets 100. Training used the
same 4,868 stride-five frames and took 23.71 minutes, excluding validation and transfers.

We restored the checkpoint from verified Drive generation `1794081756651770183` and
ran test inference locally on MPS in FP32, using the saved resize, NMS and confidence
settings. The CSV covers all 3,811 test images in `test_images.csv` order, with 256,500
detections, at most 216 per image and no empty rows. Class IDs, scores and original-image
pixel bounds were validated. The official sample was unavailable locally, so
`official_sample_verified` remains false. The user submitted this CSV and reported
Kaggle Public **0.14348**, visible in the 7 November 2026 submissions screenshot.
This is below the Faster R-CNN final-epoch Public reference of 0.20369; the cause
has not been established.

The [CSV](https://drive.google.com/file/d/1vsPmYGTtLBT14FEkj3ah_vICpr3qt87V/view)
and [export metadata](https://drive.google.com/file/d/1HlYixXJ9ZqpooNG5mI-T-uXW8OhGpbn5/view)
are stored in the existing YOLO experiment folder on Drive. The CSV SHA256 is
`4380b7c2c989e72f49529f6d07dae4582b7b7e8c9648da91c685c51ba3f08397`.

## Faster R-CNN data experiments

We downloaded all three completed Kaggle CUDA/AMP exports from their verified Drive
generations. Each CSV uses the final fifth-epoch checkpoint and covers all 3,811 test
images in test_images.csv order, with no empty predictions. We verified checkpoint and
CSV SHA256 identities, class IDs 0–7, confidence values, original-pixel box bounds and
the 500-detection limit. The official sample was unavailable; metadata records validation
against test_images.csv. No competition submission was sent during download.

| Experiment               | CSV                                                | Detections | Maximum per image | Drive CSV                                                                      |
| ------------------------ | -------------------------------------------------- | ---------: | ----------------: | ------------------------------------------------------------------------------ |
| Photometric augmentation | `faster_rcnn_photometric_augmentation_stride5.csv` |     291999 |               147 | [File](https://drive.google.com/file/d/1oFh9w80g_t6JAm0-5mr6EzObT-S3ji58/view) |
| Repeat factor sampling   | `faster_rcnn_repeat_factor_sampling_stride5.csv`   |     287076 |               144 | [File](https://drive.google.com/file/d/1d9ctinCKY50jX81SPgE-R3Fu8JoUYunk/view) |
| Higher resolution        | `faster_rcnn_input_resolution_stride5.csv`         |     291863 |               157 | [File](https://drive.google.com/file/d/1BZNjWZ24JnLpW74zVmboTBCR4GgnsIQR/view) |

The corresponding immutable generations are `1794131500682713072`,
`1794131823144531125` and `1794132219893485612`. Checkpoint hashes and inference timing
remain in each generation's submission_metadata.json on Drive. Temporary downloaded
records were removed after verification; local CSVs remain gitignored.
