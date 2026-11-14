---
title: Homework 2 -- Multiple Object Video Detection
author: Lenyk Nikita, Hayevska Yustyna
course: Artificial Intelligence · UCU
date: 2026-11-11
---

## EDA (Exploratory Data Analysis)

During the EDA phase, our goal was to understand the dataset and make sure that nothing in it would silently invalidate training or evaluation. 

Remark about coordinate systems:

The dataset uses three conventions. 

This could cause issues, as a conversion error still produces a valid file that scores near zero. 

YOLO labels store the box center and size normalized to the image width and height, with class IDs 0 to 7.

COCO annotations store the top-left corner and size in the original pixels, with category IDs 1 to 8.

The submission expects the COCO layout in original pixels but the YOLO class IDs. We convert with $x = (c_x - w/2) W$, $y = (c_y - h/2) H$, width $w W$, height $h H$ and class = category - 1. Ignored regions use the YOLO layout without a class. Our models see a frame resized so that its longer side is 1280 px. 


### EDA Findings

Object size: 24.4% of the boxes are COCO small (by COCO convention, below 32x32 px), with median areas from 1 344 $\mathrm{px}^2$ for person to 12 470 $\mathrm{px}^2$ for bus ([01_eda](notebooks/01_eda.ipynb), Section 4).

The relevant size is the one after resizing, however. With a 640 px long side, as in our first detectors, 54% of the boxes have a side below 16 px (median 15 px). With a 1280 px long side, the median is 30 px and 9.5% are below 16 px ([10_yolo11s_native_pipeline](notebooks/10_yolo11s_native_pipeline.ipynb), Section 4). The final model therefore uses 1280 px.

Density: A training frame contains 45.7 objects on average, with a median of 43, a 99th percentile of 142 and a maximum of 180 (01_eda, Section 5). We use a confidence threshold of 0.001 and up to 500 detections per frame. Raising the threshold to 0.05 lowered mAP by 0.39 points ([11_yolo11s_tuning_experiments](notebooks/11_yolo11s_tuning_experiments.ipynb), Section 6).

Class imbalance: Car accounts for 44.0% of the boxes, person 31.1% and bus for 0.8% (01_eda, Section 3).

On the clean validation fold, AP follows object size more than frequency: bus reaches 0.66 and bicycle 0.13 (10_yolo11s_native_pipeline, Section 10). Repeat-factor sampling did not give a consistent gain ([09_repeat_factor_sampling](notebooks/09_repeat_factor_sampling.ipynb)). The failure analysis shows that among predictions with score at least 0.3, 24 to 29% of van, truck and bicycle objects are assigned to another class (10_yolo11s_native_pipeline, Section 11).

Train and test: labeled frames are 1344 to 3840 px wide, while test frames are 680 to 1920 px (01_eda, Section 6).

Independence: the same object keeps a median IoU of 0.90 between neighboring frames (10_yolo11s_native_pipeline, Section 4), so every split keeps whole clips and the final validation is grouped by video ([12_yolo11s_grouped_cv](notebooks/12_yolo11s_grouped_cv.ipynb)).
