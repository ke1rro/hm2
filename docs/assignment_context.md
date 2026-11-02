# Multiple Object Video Detection — HW2 | UCU AI 2026

> **Document purpose:** Reference context and assignment specification for contributors and coding agents. This document describes the task, data contracts, evaluation expectations, submission constraints, and source material. It is **not** an implementation plan, development workflow, or technical report.
>
> **Source of truth:** [Official Kaggle competition](https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026) and its provided dataset files. This document consolidates the Dataset Description and competition Overview supplied by the student. If they differ from the current competition page, the official instructions take precedence.

## 1. Task definition

The assignment is a computer-vision **multiple-object detection in drone video** problem based on **VisDrone2019-VID** footage. The goal is to predict object **class, confidence score, and bounding box** for every relevant object in every frame of the **48 unlabeled test video clips**.

The primary prediction unit is a **video frame**. The data retain video/sequence membership and frame ordering, but the required submission contains one row of detections per test image. The assignment also evaluates understanding of multiple detector families, controlled experimentation, and the resulting technical report.

The detection environment has several salient properties:

- Objects are frequently **small** (often approximately **10–30 pixels**).
- Scenes can be **dense**, with multiple adjacent or overlapping objects.
- Objects may be partially or heavily **occluded** and **truncated**.
- Input resolutions vary across clips: **680 × 382** to **1920 × 1080** pixels.
- Neighboring video frames are highly correlated; **video clip identity** is relevant to validation and generalization.
- Some areas are deliberately **ignored / don't-care** rather than exhaustively annotated.

### Target classes

The task uses **8 classes** with zero-based identifiers in YOLO labels and submissions:

|  ID | Class   |  ID | Class    |
| --: | ------- | --: | -------- |
|   0 | person  |   4 | truck    |
|   1 | bicycle |   5 | tricycle |
|   2 | car     |   6 | bus      |
|   3 | van     |   7 | motor    |

**Important:** This competition's eight-class target space is not interchangeable with the original VisDrone ten-category convention. The COCO annotations also use a different numeric offset (see below).

## 2. Dataset inventory and layout

| Split / file            |   Clips |                   Frames | Annotation availability                  |
| ----------------------- | ------: | -----------------------: | ---------------------------------------- |
| `train/`                |      56 |                   24,201 | Labeled                                  |
| `val/`                  |       7 |                    2,846 | Labeled                                  |
| `test/images/`          |      48 |                    3,811 | Unlabeled                                |
| **Total**               | **111** |               **30,858** | —                                        |
| `test_images.csv`       |       — | 3,811 test-image records | Test-image metadata                      |
| `sample_submission.csv` |       — |   One row per test image | Valid empty-prediction example (score 0) |

`train/` and `val/` share the following directory layout:

```text
train/ or val/
├── images/<clip>/<frame>.jpg
├── labels/<clip>/<frame>.txt
├── ignore/<clip>/<frame>.txt
├── annotations_coco.json
└── data.yaml
```

- `images/`: individual video frames, grouped by clip.
- `labels/`: YOLO-format labels, one text file per labeled frame.
- `ignore/`: don't-care regions; files exist only for frames that have such regions.
- `annotations_coco.json`: the same target annotations in COCO format, plus metadata.
- `data.yaml`: Ultralytics-compatible dataset configuration supplied with each labeled split.

The test split includes `test/images/<clip>/<frame>.jpg` but **no ground-truth annotations**.

## 3. Annotation contracts

### 3.1 YOLO labels

Each nonempty line in `labels/<clip>/<frame>.txt` has:

```text
class_id center_x center_y width height
```

Example supplied in the dataset description:

```text
0 0.328497 0.776455 0.026042 0.116402
```

- Class IDs: **0–7**.
- `center_x`, `center_y`, `width`, `height`: floating-point values **normalized by the original image width and height** (range 0–1).
- `center_x` / `center_y` represent the **center** of the box.
- `width` / `height` represent box dimensions, not the bottom-right corner.

### 3.2 Ignored regions

Each line in `ignore/<clip>/<frame>.txt` has:

```text
center_x center_y width height
```

- Coordinates and dimensions are normalized (same convention as YOLO boxes).
- No class ID is present.
- These regions may represent crowds, dense parking areas, too-small/too-dense objects, or VisDrone's original `others` category.
- The dataset description states that **detections in these regions are not penalized in scoring**.
- The annotations represent **don't-care** areas, not ordinary labeled background.

The exact geometric matching rule used by the competition scorer for ignored regions is **not specified** in the provided description; consult the official evaluation/scoring implementation for any exact threshold or overlap behavior.

### 3.3 COCO annotations

`annotations_coco.json` stores boxes in original-image **pixel coordinates**:

```text
bbox = [x, y, width, height]
```

- `x`, `y`: **top-left** corner.
- COCO `category_id`: **1–8**, where **1 = person** and **8 = motor**.
- YOLO and submission IDs: **0–7**. These are not numerically interchangeable with COCO IDs.
- Ignored regions are represented as annotations with `iscrowd = 1`.
- Additional annotation fields: `truncation`, `occlusion`, `track_id`, and `orig_category_id`.
- `track_id`: identity of the same object across video frames.
- `orig_category_id`: category in the original ten-class VisDrone dataset.
- Image records include `video` and `frame_id` metadata.

### Coordinate conventions at a glance

| Format                        | Class IDs | Bounding-box representation | Coordinate units      |
| ----------------------------- | --------- | --------------------------- | --------------------- |
| YOLO `labels/*.txt`           | 0–7       | `cx cy w h`                 | Normalized to 0–1     |
| `ignore/*.txt`                | No class  | `cx cy w h`                 | Normalized to 0–1     |
| COCO `annotations_coco.json`  | 1–8       | `x y w h` (top-left)        | Original-image pixels |
| Submission `PredictionString` | 0–7       | `x y w h` (top-left)        | Original-image pixels |

## 4. Test metadata and submission contract

### 4.1 `test_images.csv`

| Column      | Meaning                                                                    |
| ----------- | -------------------------------------------------------------------------- |
| `image_id`  | Submission row identifier, e.g. `clip_001_0000001`                         |
| `file_name` | Path relative to the dataset root, e.g. `test/images/clip_001/0000001.jpg` |
| `clip`      | Video sequence identifier                                                  |
| `frame`     | Consecutive frame number within the clip                                   |
| `width`     | Image width in pixels                                                      |
| `height`    | Image height in pixels                                                     |

The competition description specifies a **clip-level** public/private leaderboard split, not a per-frame split.

### 4.2 Submission schema

The submission is a CSV with **exactly one row per test image** and the columns:

```csv
image_id,PredictionString
clip_001_0000001,2 0.91 1366 580 26 44 0 0.55 809 625 29 66
clip_001_0000002,
```

Each prediction is a **space-separated group of six values**:

```text
class_id confidence x y width height
```

- `class_id`: integer **0–7**.
- `confidence`: detection confidence score.
- `x`, `y`: box **top-left** corner.
- `width`, `height`: box dimensions.
- All spatial values refer to the **original unresized image**, in **pixels**.
- Multiple detections are concatenated into a single space-separated string in a single cell.
- An image with no detections has an **empty** `PredictionString`.
- **At most 500 detections per image are scored.**

The `sample_submission.csv` contains valid rows with empty predictions, which score 0.

## 5. Assessment and required deliverables

### 5.1 Grading overview

| Component            |          Points |
| -------------------- | --------------: |
| Private leaderboard  |            2.00 |
| Code and experiments |            6.00 |
| Technical report     |            2.00 |
| **Core total**       |       **10.00** |
| Optional bonus tasks | **Up to +5.00** |

**Critical condition:** The competition description states that if a submission does **not beat the baseline**, the homework grade is canceled.

The final submission requires **two separate deliverables**:

1. **Complete, reproducible code**, organized in one or more Jupyter notebooks and/or scripts.
2. **A technical report** documenting methodology, motivation, experiments, results, and final selection.

The official **GitHub repository template** must be used. The report must be published via **GitHub Pages** according to the template instructions. Both the code and report must be submitted through **LMS**, and the repository must be public and accessible for grading. A notebook with saved outputs does not replace the report.

### 5.2 Code checkpoints (6.00 points)

The following are **assignment requirements**, not recommendations for a particular implementation.

| Checkpoint                           | Points | Expected scope                                                                                                                                                                                                                |
| ------------------------------------ | -----: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data Understanding and Preprocessing |   1.00 | Class distribution; object sizes; objects/frame; variation by sequence; annotation formats and ignored regions; visually verified labels; sequence-level validation; documented preprocessing.                                |
| Two-stage anchor-based detection     |   1.00 | Pretrained Faster R-CNN / Cascade R-CNN or comparable detector; at least two relevant training-parameter investigations; mAP, AP50, AP_small, inference speed; at least three failure cases.                                  |
| One-stage anchor-based detection     |   1.00 | RetinaNet / SSD or comparable detector; relevant anchor, focal-loss, or assignment settings; comparison with two-stage results.                                                                                               |
| One-stage anchor-free detection      |   1.00 | FCOS / CenterNet / suitable recent YOLO or comparable detector; at least two relevant parameter investigations; size-specific comparisons against anchor-based detectors.                                                     |
| Query-based detection                |   0.75 | RT-DETR / Deformable DETR or comparable detector, with comparable compute budget; query/matching/loss/learning-rate investigations; mAP-vs-training comparison with CNN detector; analysis of dense scenes and small objects. |
| Your Pipeline                        |   1.25 | Final approach backed by earlier results; both data/inference strategies and additional modeling experiments; clear intermediate comparisons; reproducible final results and model summary.                                   |

Cross-cutting requirements stated in the competition:

- Use **sequence-level**, not random frame-level, validation splits to avoid temporal leakage.
- Use proper **cross-validation / out-of-fold evaluation** and consistent metrics across model experiments.
- Keep model comparisons on a consistent evaluation protocol and explain material differences.
- For the four detector-family checkpoints, include **at least two experiments beyond baseline involving hyperparameter tuning and loss modification**.
- Show both successful and unsuccessful experiments; correctness and evidence matter beyond a single leaderboard improvement.
- Standard requested detection metrics include **mAP, AP50, AP_small**, and, for the two-stage checkpoint, **inference speed**.
- Technical understanding and decision justification are assessed in an individual defense.

**Specification boundary:** The provided `Your Pipeline` subsection refers to “both parts below” but does not separately enumerate their names/details in the supplied text. No additional hidden rubric is inferred here. Check the current Kaggle assignment wording if this matters.

### 5.3 Technical report (2.00 points)

| Component                        | Points | Requirement                                                                                                       |
| -------------------------------- | -----: | ----------------------------------------------------------------------------------------------------------------- |
| Methodology and motivation       |   0.75 | Data preparation, validation design, models, tuning strategy, and reasoning for key choices.                      |
| Results and analysis             |   0.75 | Consistent comparisons; incremental effects; unsuccessful experiments; conclusions supported by observed results. |
| Final model and run instructions |   0.50 | Selected final model/ensemble; reproducible training and prediction instructions, dependencies, settings, seeds.  |

The competition recommends a **compact report of around 1,500 words**. Excessive verbosity may be penalized. Long unfiltered notebook outputs, runtime errors, or poorly structured code may also reduce the grade.

**Academic integrity:** AI assistance with **code** is permitted, but the supplied assignment explicitly prohibits **AI-written reports**. Reports must communicate the students' own conclusions and reasoning. Ideas or code derived from external sources require proper citation. Code sharing or plagiarism between participants is prohibited.

### 5.4 Optional bonus tasks (up to 5.00 points)

| Bonus category                 |      Points | What is evaluated                                                                                                                                                  |
| ------------------------------ | ----------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Data Temporality               |       +1.00 | Improvements using temporal relationships across video frames, e.g. linked/propagated detections, tracking, motion compensation, temporal features.                |
| Visualization                  |       +1.00 | Meaningful visual explanation of model behavior/failure via embeddings, transformer attention, Grad-CAM, etc.; interpretations are more important than plot count. |
| Custom Architectural Decisions | Up to +3.00 | Evidence-supported changes to an architecture for this specific dataset, with comprehensive experiments, failures, and explanations.                               |

For the **Visualization** bonus, the provided rubric explicitly requires **Weights & Biases** logging, including training curves, configurations, and validation images with predicted boxes for each required model, plus a grader-accessible W&B project/report link. Every bonus task requires an additional individual discussion.

### 5.5 Collaboration and submission constraints

- The assignment explicitly allows and encourages **teams of two**; each participant has an **individual defense** and must understand all aspects of the solution. The supplied wording does not describe larger teams.
- Both teammates must be collaborators on the repository and have meaningful commits.
- The official repository template and public GitHub repository requirements apply.
- Both teammates must submit the repository link through LMS.
- Do not alter commit timestamps or manipulate commit history.
- The provided competition text contains **inconsistent late-submission wording**: one section describes a 50% penalty for submissions up to a week late; another states no late submissions are accepted. Treat the **current official Timeline/course rules** as authoritative rather than assuming either is definitive.
- Relative site counters such as “7 days to go” are not stable deadlines; verify the actual closing date on Kaggle.

## 6. Evaluation terminology and unresolved specification details

- **mAP / AP50 / AP_small**: standard object-detection evaluation concepts explicitly requested for internal model comparisons. See the linked COCO evaluation references for definitions.
- **Private leaderboard**: contributes two core grading points; the public/private split is by video clip according to the dataset description.
- **Competition-specific scoring**: the provided material identifies a custom metric but does **not** include its complete mathematical implementation or exact treatment of ignored-area overlap. Do not assume ordinary COCO evaluation is identical to the Kaggle leaderboard scorer without checking the official Evaluation page or scoring implementation.
- **Per-size AP** and video-level validation are relevant because targets are frequently small and frames within a clip are strongly correlated.

## 7. Reference sources

The following sources are **linked in the competition's provided reading list**. They are included for provenance, terminology, and later verification; listing them does not prescribe any model, library, or implementation choice.

### Primary task and dataset sources

1. **Official UCU HW2 competition** — https://www.kaggle.com/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026
   Authoritative for the current assignment wording, dataset files, scoring, template, submission, and timeline.
2. **VisDrone benchmark paper** — https://arxiv.org/abs/2001.06303
   Drone dataset and benchmark background, including data collection and annotation conventions.
3. **COCO benchmark** — https://cocodataset.org/#detection-eval
   Evaluation terminology such as AP and object-size categories.
4. **COCO paper** — https://arxiv.org/abs/1405.0312
   Dataset and evaluation context.
5. **COCO API / pycocotools** — https://github.com/cocodataset/cocoapi
   Reference tooling and evaluation behavior.

### Detector families explicitly covered by the rubric

06. **Faster R-CNN** — https://arxiv.org/abs/1506.01497
    Two-stage anchor-based detection and region proposals.
07. **Feature Pyramid Networks (FPN)** — https://arxiv.org/abs/1612.03144
    Multiscale features relevant to small-object detection.
08. **RetinaNet / Focal Loss** — https://arxiv.org/abs/1708.02002
    One-stage anchor-based detector and class imbalance.
09. **FCOS** — https://arxiv.org/abs/1904.01355
    One-stage anchor-free detector.
10. **DETR** — https://arxiv.org/abs/2005.12872
    Query-based set prediction and bipartite matching.
11. **Deformable DETR** — https://arxiv.org/abs/2010.04159
    Multiscale query-based detection.
12. **RT-DETR** — https://arxiv.org/abs/2304.08069
    Real-time query-based detector.

### Dataset-specific challenges and analysis

13. **Towards Large-Scale Small Object Detection: Survey and Benchmarks** — https://arxiv.org/abs/2207.14096
    Small-target detection literature.
14. **Normalized Wasserstein Distance for Tiny Object Detection** — https://arxiv.org/abs/2110.13389
    Tiny-box localization and loss behavior.
15. **TIDE: A General Toolbox for Identifying Object Detection Errors** — https://arxiv.org/abs/2008.08115
    Detection error taxonomy for failure analysis.
16. **SAHI: Slicing Aided Hyper Inference** — https://arxiv.org/abs/2202.06934
    Small-object detection at high image resolutions.

### Temporal and visualization bonus background

17. **Seq-NMS** — https://arxiv.org/abs/1602.08465
    Video-sequence-based rescoring.
18. **ByteTrack** — https://arxiv.org/abs/2110.06864
    Tracking-by-detection across frames.
19. **Grad-CAM** — https://arxiv.org/abs/1610.02391
    Visual attribution for neural network predictions.
20. **Weights & Biases documentation** — https://docs.wandb.ai/
    Logging platform referenced by the visualization bonus.

### Relevant framework documentation cited in the assignment

- **TorchVision detection tutorial:** https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- **MMDetection:** https://github.com/open-mmlab/mmdetection
- **Ultralytics:** https://docs.ultralytics.com/
- **Hugging Face object detection:** https://huggingface.co/docs/transformers/tasks/object_detection
- **Albumentations:** https://albumentations.ai/docs/
- **scikit-learn GroupKFold:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html

## 8. Provenance and scope of this file

- **Dataset statistics, class IDs, annotation formats, ignored regions, image resolutions, and submission format** come from the student-provided *Dataset Description*.
- **Checkpoints, point values, bonus requirements, policies, deliverables, and suggested references** come from the student-provided copy of the *Kaggle competition Overview*.
- This document does **not** contain any measured EDA results, completed model experiments, leaderboard scores, claims about actual model performance, or chosen implementation design.
- This document is **not** the technical report requested by the assignment and must not be submitted as a substitute for a student-authored report.
