# RetinaNet submission CSVs

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
