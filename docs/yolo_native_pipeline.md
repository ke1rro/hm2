# YOLO11s pipeline from VisDrone-DET: audit, design and run instructions

Snapshot: 2026-11-08. Competition deadline: 2026-11-09 23:55 UTC (5 submissions per day).
This is an engineering handoff for the final pipeline, not the student-authored report.
The pipeline ran once on Kaggle (2 × T4) on 2026-11-08 and scored **0.34389** on the Public
leaderboard (Section 7). Records are under
`batches/yolo11s_native_1280_stride4_seed2026/` on Drive.

Files: `config/yolo11s_native_1280.toml`, `src/hm2/yolo_data.py`,
`src/hm2/yolo_pipeline.py`, `scripts/build_yolo_bundle.py`,
`notebooks/10_yolo11s_native_pipeline.ipynb`, `tests/test_yolo_pipeline.py`.

## 1. What earlier experiments established

The metric is standard COCO mAP over IoU 0.50:0.05:0.95. Our local evaluator is the
same: pycocotools COCOeval with category-specific `iscrowd` ignore regions, and
faster_coco_eval as an equivalent fast backend.

| Model (official validation, 2,846 frames, maxDets 100)   |               Val mAP |                Public |
| -------------------------------------------------------- | --------------------: | --------------------: |
| RetinaNet, four anchor/sampling variants                 |           0.149–0.159 |           0.142–0.168 |
| Faster R-CNN, stride 5, epoch 5                          |                 0.175 |               0.20369 |
| Faster R-CNN + photometric / repeat-factor / 800–1280 px | 0.179 / 0.182 / 0.209 | 0.199 / 0.196 / 0.194 |
| YOLO11s in the shared training loop                      |                 0.155 |               0.14348 |
| RT-DETRv2-R50 q300, 5 epochs (merged to `main`)          |                 0.246 |               0.25323 |
| Leaderboard: baseline / first place (2026-11-08)         |                     — |         0.22 / 0.3675 |

Verified facts:

- **The local evaluator reproduces recorded values.** Re-evaluating the stored YOLO11s
  validation predictions gives mAP 0.154990, AP50 0.297651 and AP_small 0.092178, with
  both backends.
- **The old YOLO11s result measured an unfinished recipe.** It used the shared loop
  (batch 2, SGD at lr 0.0025, five epochs, flip only, no mosaic, HSV, affine or EMA), and
  mAP was still rising at epoch five.
- **Validation and Public disagree because of the data, not the metric.** Both use the
  same metric, yet the Faster R-CNN ablations moved them in opposite directions. Seven
  validation clips and about 14 Public clips are small samples.
- **No data-integrity defect was found.** Labels, class mapping, coordinates and
  sequence-level splits are consistent. `sample_submission.csv` is absent from the
  archive, so submissions are validated against `test_images.csv`.

## 2. Dataset measurements that drive the design

| Property                           | Measurement                                                                                                     | Consequence                                                      |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Effective size after resize, √(wh) | At a 640 px long side, 54% of boxes are below 16 px (median 15 px); at 1280 px, 9.8% (median 30 px)             | Train and infer at 1280 px                                       |
| Test resolution                    | Test 680–1920 px wide; train 1344–2720 px; validation up to 3840 px                                             | Resize by the long side with one rule for every split            |
| Relative scale on test             | Confident test boxes 1.9–2.7% of frame width; ground truth 2.2–2.3%                                             | Scale normalization transfers to test                            |
| Temporal redundancy                | Same-track IoU between t and t+s: 0.90 (s=1), 0.66 (s=4), 0.44 (s=8)                                            | Stride 4 keeps 99.4% of tracks                                   |
| Ignored regions                    | 75% of train frames; 3.5% of boxes touch one                                                                    | Mask ignored pixels and restore annotated boxes                  |
| Class weighting                    | COCO mAP averages the 8 classes equally; bus is 0.9% of train boxes and appears in one official validation clip | Rare classes matter as much as car; validation must contain them |

## 3. VisDrone-DET initialization

A public YOLO11s trained on VisDrone2019-DET at 1280 px is available as
[dronefreak/visdrone-yolo11s](https://huggingface.co/dronefreak/visdrone-yolo11s)
(revision `c1c9c8f`, SHA256 `3134756b…`). Its training arguments are stored in the
checkpoint: YOLO11s from COCO, 87 epochs, 1280 px. Its 11 VisDrone classes map exactly to
ours: person = pedestrian + people, tricycle = tricycle + awning-tricycle, and the other
six are one to one.

**Overlap check.** We compared a normalized 64×36 grayscale descriptor of every
VisDrone-DET train and val image (7,019) with every frame of our splits:

- 6 of 7 official validation clips and 41 of 56 training clips have exact frames in
  DET, with NCC up to 0.9999. DET file names carry the VID sequence IDs.
- No test frame has a near duplicate: best NCC 0.928, and visual inspection shows the
  same bridge from another viewpoint, not the same frame.
- Our own train clips against test reach at most 0.86.

The Kaggle rules forbid only hand labels or predictions of validation or test records.
Using DET is therefore legitimate for the test set, provided it is cited. The official
validation split, however, cannot evaluate a DET-initialized model.

**Zero-shot quality.** We remapped classes after inference, with no training on our
data. On clips that DET never contained:

| Data                                                     |   VisDrone-DET YOLO11s | Faster R-CNN | RT-DETRv2 | YOLO11s shared loop |
| -------------------------------------------------------- | ---------------------: | -----------: | --------: | ------------------: |
| Unseen validation clip `uav0000268_05773_v` (978 frames) |              **0.159** |        0.112 |     0.100 |               0.104 |
| 11 unseen training clips (773 frames, every 4th)         | **0.333** (AP50 0.606) |            — |         — |                   — |

The unseen validation clip is a sparse 4K scene where every model scores low. The
training-clip value is the closest analogue of the test situation: no identical frames,
but similar locations.

## 4. Selected pipeline

**Initial weights.** `scripts/build_yolo_bundle.py` downloads the pinned checkpoint,
checks its SHA256 and converts the head with `remap_detection_head`. Each competition
class takes one source classification channel (person ← pedestrian, tricycle ←
tricycle); every other tensor is copied. A test confirms that the converted network
reproduces the selected source scores. The checkpoint stores its provenance and ships
inside the code bundle.

**Data split.** Clips are assigned whole. Validation is a clean fold of six clips that
neither DET nor our training contains: `uav0000268_05773_v` plus five unseen training
clips (`150_02310`, `361_02323`, `143_02250`, `278_00001`, `020_00406`). It has 3,269
frames, 14–109 objects per frame, widths 1344–3840 px and all eight classes (790 bus
boxes). Training uses every fourth frame of the other 51 training clips and of the six
DET-contaminated validation clips: 5,966 frames from 57 clips. All frames are written at
a 1280 px long side with ignored regions masked; labels are copied unchanged.

**Recipe.** We use the native Ultralytics 8.3.235 trainer with explicit arguments:

- **Optimizer.** SGD with momentum 0.937 and weight decay 0.0005, nominal batch 64,
  batch 16 on 2×T4 with DDP, AMP and EMA.
- **Schedule.** The source finished training at a learning rate of about 6e-4, so we
  restart gently: lr0 0.002, a one-epoch warmup with `warmup_bias_lr` 0, and linear decay
  to 0.1·lr0.
- **Augmentation.** The source model's augmentation: mosaic 1.0 switched off for the
  last 5 epochs, scale 0.3, translation 0.1, horizontal flip and HSV.
- **Training length.** At most 30 epochs within a 5 h budget, planned from measured
  throughput, with early stopping after 10 epochs without improvement. A short schedule
  limits forgetting of DET's scene diversity.

**Selection.** The candidates are the initial checkpoint (person channel = pedestrian
only, a conservative reference), `best.pt` and `last.pt`. Each is evaluated on the
original clean-fold frames with COCO mAP at maxDets 100 and crowd regions ignored, using
a merged two-split COCO file. The highest mAP is chosen, so fine-tuning that hurts
cannot be selected.

**Inference and export.** 1280 px, confidence 0.001, class-aware NMS at IoU 0.7, at most
500 boxes per frame, boxes mapped back per axis to original pixels. The CSV is validated
against `test_images.csv` and written atomically with metadata.

**Zero-shot fallback.** `runs/visdrone_det_zero_shot/visdrone_det_yolo11s_zero_shot_1280.csv`
uses the unconverted 11-class model with classes merged after inference and NMS
repeated. It was produced on CPU and needs no GPU.

## 5. Verification performed (CPU only)

| Check                                           | Result                                                                                                                                                   |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset preparation, clean fold                 | 9,235 frames in 24 s with 16 threads; all 16 checks pass, including per-class box counts equal to COCO and disjoint clips                                |
| Head conversion                                 | Converted 8-class network reproduces the selected source channels (unit test)                                                                            |
| COCO subset for the fold                        | Merges both splits with crowd regions and aligned IDs; a perfect prediction scores 1.0 (unit test)                                                       |
| Coordinate mapping                              | Matches Ultralytics' own prediction on all six test resolutions (median IoU 0.95–1.00)                                                                   |
| Evaluator                                       | faster_coco_eval equals pycocotools to 6 decimals                                                                                                        |
| Native training contract and notebook rehearsal | All cells executed on CPU with tiny overrides, including the VisDrone-initialized training                                                               |
| Kaggle bootstrap                                | Attached zip, Kaggle-extracted bundle, missing bundle and the real Drive download behave as intended; offline wheels install in a fresh Python 3.13 venv |
| Repository checks                               | isort, black, flake8, ty, mdformat and pytest (101 tests) pass                                                                                           |

The Kaggle run then verified the GPU path: DDP on two T4 GPUs, 8–13 GB peak memory per
GPU at batch 8, about 4.7 minutes per epoch including validation, and the throughput-based
plan of 30 epochs.

## 6. Running on Kaggle

1. Attach the competition data and the private dataset **ke1rro/hm2-yolo-native**. It
   holds the code bundle with the converted checkpoint and the wheels. The same bundle is
   in `My Drive/hm2_retinanet/inputs/`. After code changes, run
   `uv run python scripts/build_yolo_bundle.py --offline-assets`, then
   `kaggle datasets version -p dist/yolo_native -m "..."`.
2. Import `notebooks/10_yolo11s_native_pipeline.ipynb` and choose **GPU T4 x2**.
3. With the default `SHOULD_USE_DRIVE = True`, enable the `HM2_DRIVE_TOKEN` secret, and
   refresh it if it is older than seven days. Set it to `False` to keep outputs only in
   `/kaggle/working`.
4. Set `SHOULD_TRAIN = True` and start **Save Version → Save & Run All**. The run takes
   about 6 h:
   - data preparation, a few minutes;
   - throughput check, about 8 minutes;
   - training, at most 5 h;
   - selection on the clean fold, about 10 minutes;
   - export to `/kaggle/working/submissions/yolo11s_native_1280_stride4.csv`.
5. **After an interruption,** rerun the same version. With Drive enabled it resumes from
   the last published `last.pt`.
6. **On a CUDA out-of-memory error,** set `batch = 12`. For a single T4, set
   `device = "0"`.

Submit CSVs manually. Choose the final two leaderboard entries using clean-fold mAP
rather than Public score alone.

## 7. Kaggle run results

| Quantity                                        | Value                                                                                        |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Throughput check                                | 0.0457 s per training image, 60 s per validation, 30 planned epochs (2.9 h)                  |
| Epochs trained                                  | 11, stopped early: native fitness peaked after epoch 1 and declined while training loss fell |
| Native validation mAP50-95 (clean fold, masked) | 0.382 after epoch 1, 0.341 after epoch 11                                                    |
| Training time                                   | 0.89 h; the whole session about 73 minutes                                                   |

Selection on the clean fold (COCO, maxDets 100, original frames, crowd regions ignored):

| Candidate                                                  |        mAP |  AP50 | AP small | Kaggle Public |
| ---------------------------------------------------------- | ---------: | ----: | -------: | ------------: |
| Initial VisDrone checkpoint (person from pedestrian only)  |     0.3025 | 0.563 |    0.095 |             — |
| Same model, classes merged after inference (zero-shot CSV) |          — |     — |        — |       0.27998 |
| `best.pt`, epoch 1 (selected)                              | **0.3553** | 0.598 |    0.142 |   **0.34389** |
| `last.pt`, epoch 11                                        |     0.3154 | 0.538 |    0.122 |             — |

Per-class AP of the selected checkpoint: car 0.520, bus 0.664, van 0.392, tricycle 0.340,
truck 0.315, person 0.263, motor 0.220, bicycle 0.129.

Observed facts: one fine-tuning epoch at a warmup learning rate improved both the clean
fold (+5.3 points over the converted initial model) and the Public score (+6.4 points over
the zero-shot CSV). Longer fine-tuning lowered clean-fold mAP, consistent with forgetting
the scene diversity of VisDrone-DET. The clean fold and the Public leaderboard moved in
the same direction, unlike the official validation split in the earlier Faster R-CNN
ablations. A lower learning rate or a frozen backbone for a few epochs would test the
forgetting explanation; we have not run it.

## Sources

- VisDrone-DET YOLO11s checkpoint: https://huggingface.co/dronefreak/visdrone-yolo11s
  (DetectionBench, AGPL-3.0); dataset VisDrone2019-DET, CC BY-NC-SA 3.0,
  https://arxiv.org/abs/2001.06303
- Ultralytics YOLO11 and training arguments: https://docs.ultralytics.com/models/yolo11/,
  https://docs.ultralytics.com/modes/train/
- COCO evaluation and crowd handling: https://cocodataset.org/#detection-eval
- faster-coco-eval: https://github.com/MiXaiLL76/faster_coco_eval
- Small-object detection survey: https://arxiv.org/abs/2207.14096
