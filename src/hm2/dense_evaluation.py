import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from faster_coco_eval import COCO as FasterCOCO
from faster_coco_eval import COCOeval_faster
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hm2.data import CLASS_NAMES
from hm2.evaluation import evaluate_coco

SUBMISSION_DETECTION_CAP = 500


def _mean_precision(values: np.ndarray) -> float | None:
    values = values[values > -1]
    return float(np.mean(values)) if values.size else None


def _set_empty_detections(detections: COCO | FasterCOCO, ground_truth: COCO | FasterCOCO) -> None:
    detections.dataset = {
        "images": ground_truth.dataset["images"],
        "categories": ground_truth.dataset["categories"],
        "annotations": [],
    }
    detections.createIndex()


def evaluate_coco_max_dets(
    annotation_path: Path,
    predictions: list[dict],
    image_ids: list[int],
    max_detections: int,
    backend: str = "pycocotools",
) -> dict:
    """Evaluate COCO bbox AP with a configurable per-image detection cap.

    The protocol matches hm2.evaluation.evaluate_coco except for the last maxDets value.
    AP values are averaged directly from the accumulated precision array, because
    COCOeval.summarize computes overall mAP only at a hard-coded cap of 100 detections.
    Object size bins use the ground-truth area field, which is width times height in
    original-image pixels, and crowd annotations keep detections inside them unpenalized.
    The faster_coco_eval backend is a C++ reimplementation of the same COCOeval protocol
    for dense prediction sets; both backends read the same precision array.

    Args:
        annotation_path: Original COCO JSON with category-specific crowd annotations.
        predictions: COCO-format detections in original-image pixel xywh.
        image_ids: Exact evaluated source image IDs, including frames with no predictions.
        max_detections: Highest-scoring detections per image that are evaluated.
        backend: COCOeval implementation, either pycocotools or faster_coco_eval.

    Returns:
        COCO AP metrics, nullable per-class AP and the explicit evaluation protocol.
    """
    max_dets = [1, 10, max_detections]
    with redirect_stdout(io.StringIO()):
        if backend == "faster_coco_eval":
            faster_truth = FasterCOCO(str(annotation_path))
            faster_detections = faster_truth.loadRes(predictions) if predictions else FasterCOCO()
            if not predictions:
                _set_empty_detections(faster_detections, faster_truth)
            evaluator = COCOeval_faster(faster_truth, faster_detections, "bbox")
        else:
            ground_truth = COCO(str(annotation_path))
            ground_truth.dataset.setdefault("info", {})
            detections = ground_truth.loadRes(predictions) if predictions else COCO()
            if not predictions:
                _set_empty_detections(detections, ground_truth)
            evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.catIds = list(range(1, 9))
        evaluator.params.maxDets = max_dets
        evaluator.evaluate()
        evaluator.accumulate()
    precision = np.asarray(evaluator.eval["precision"])[..., -1]
    thresholds = evaluator.params.iouThrs
    at_50 = int(np.argmin(np.abs(thresholds - 0.5)))
    at_75 = int(np.argmin(np.abs(thresholds - 0.75)))
    areas = {label: index for index, label in enumerate(evaluator.params.areaRngLbl)}
    return {
        "map": _mean_precision(precision[:, :, :, areas["all"]]),
        "ap50": _mean_precision(precision[at_50, :, :, areas["all"]]),
        "ap75": _mean_precision(precision[at_75, :, :, areas["all"]]),
        "ap_small": _mean_precision(precision[:, :, :, areas["small"]]),
        "ap_medium": _mean_precision(precision[:, :, :, areas["medium"]]),
        "ap_large": _mean_precision(precision[:, :, :, areas["large"]]),
        "per_class_ap": {
            name: _mean_precision(precision[:, :, index, areas["all"]])
            for index, name in enumerate(CLASS_NAMES)
        },
        "max_dets": max_dets,
        "iou_thresholds": evaluator.params.iouThrs.tolist(),
        "image_count": len(image_ids),
        "category_ids": list(range(1, 9)),
        "ignore_policy": "Original category-specific COCO iscrowd annotations",
        "coordinate_system": "Original-image pixel xywh",
        "area_definition": "COCO ground-truth area in original-image pixels",
        "metric": "COCO bbox AP; the competition metric is not defined in the originals",
        "backend": backend,
    }


def evaluate_detections(
    annotation_path: Path, predictions: list[dict], image_ids: list[int]
) -> dict:
    """Report COCO AP at the canonical cap of 100 and at the submission cap of 500.

    The cap of 100 reuses hm2.evaluation.evaluate_coco unchanged, so its values compare
    directly with the RetinaNet results. Dense frames can contain more than 100 targets,
    and the submission scores up to 500 detections per image.

    Args:
        annotation_path: Original COCO JSON with category-specific crowd annotations.
        predictions: COCO-format detections in original-image pixel xywh.
        image_ids: Exact evaluated source image IDs, including frames with no predictions.

    Returns:
        Metrics keyed by max_dets_100 and max_dets_500.
    """
    return {
        "max_dets_100": evaluate_coco(annotation_path, predictions, image_ids),
        f"max_dets_{SUBMISSION_DETECTION_CAP}": evaluate_coco_max_dets(
            annotation_path, predictions, image_ids, SUBMISSION_DETECTION_CAP
        ),
    }
