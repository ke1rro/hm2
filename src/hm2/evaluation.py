import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert

from hm2.data import CLASS_NAMES


def predictions_to_coco(
    predictions: list[dict[str, torch.Tensor]], image_ids: list[int]
) -> list[dict]:
    """Convert postprocessed source-resolution predictions to COCO detection records.

    Args:
        predictions: TorchVision xyxy predictions already returned to original resolution.
        image_ids: Corresponding source COCO image IDs within one official split.

    Returns:
        Pixel xywh detections with foreground category IDs 1 to 8, omitting label 0.
    """
    records = []
    for prediction, image_id in zip(predictions, image_ids, strict=True):
        labels = prediction["labels"].detach().cpu()
        if ((labels < 0) | (labels > 8)).any():
            raise ValueError("Predicted labels must be in the configured 0 to 8 head range.")
        foreground = labels > 0
        boxes = box_convert(prediction["boxes"].detach().cpu()[foreground], "xyxy", "xywh")
        scores = prediction["scores"].detach().cpu()[foreground]
        records.extend(
            {"image_id": image_id, "category_id": label, "bbox": box, "score": score}
            for label, box, score in zip(
                labels[foreground].tolist(), boxes.tolist(), scores.tolist(), strict=True
            )
        )
    return records


def evaluate_coco(annotation_path: Path, predictions: list[dict], image_ids: list[int]) -> dict:
    """Evaluate a selected official split using original category-specific COCO crowds.

    Args:
        annotation_path: Original COCO JSON, not the deduplicated visualization table.
        predictions: COCO-format detections in original-image pixels.
        image_ids: Exact evaluated source image IDs, including frames with no predictions.

    Returns:
        Canonical AP metrics, nullable per-class AP, and the explicit evaluation protocol.
    """
    with redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation_path))
        ground_truth.dataset.setdefault("info", {})
        if predictions:
            detections = ground_truth.loadRes(predictions)
        else:
            detections = COCO()
            detections.dataset = {
                "images": ground_truth.dataset["images"],
                "categories": ground_truth.dataset["categories"],
                "annotations": [],
            }
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.catIds = list(range(1, 9))
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = np.asarray(evaluator.eval["precision"])
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        values = precision[:, :, index, 0, 2]
        values = values[values > -1]
        per_class[name] = float(np.mean(values)) if values.size else None
    return {
        "map": float(evaluator.stats[0]) if evaluator.stats[0] >= 0 else None,
        "ap50": float(evaluator.stats[1]) if evaluator.stats[1] >= 0 else None,
        "ap_small": float(evaluator.stats[3]) if evaluator.stats[3] >= 0 else None,
        "per_class_ap": per_class,
        "max_dets": [1, 10, 100],
        "iou_thresholds": evaluator.params.iouThrs.tolist(),
        "image_count": len(image_ids),
        "category_ids": list(range(1, 9)),
        "ignore_policy": "Original category-specific COCO iscrowd annotations",
        "coordinate_system": "Original-image pixel xywh",
        "metric": "COCO bbox AP, not the Kaggle custom metric",
    }
