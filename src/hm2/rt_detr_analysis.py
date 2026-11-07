import io
import json
import math
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import batched_nms, box_convert, box_iou

AREA_BINS = {
    "all": (0.0, 1e10),
    "below_8": (0.0, 8.0**2),
    "8_to_16": (8.0**2, 16.0**2),
    "16_to_32": (16.0**2, 32.0**2),
    "32_to_96": (32.0**2, 96.0**2),
    "above_96": (96.0**2, 1e10),
}
DENSITY_BINS = {
    "1_to_20": (1, 20),
    "21_to_50": (21, 50),
    "51_to_100": (51, 100),
    "above_100": (101, 10**9),
}
DUPLICATE_SCORE_THRESHOLDS = (0.05, 0.3)
DUPLICATE_IOU_THRESHOLDS = (0.5, 0.7)
NMS_IOU_THRESHOLDS = (0.5, 0.7)
CONVERGENCE_GAIN = 0.005


def load_ground_truth(annotation_path: Path) -> COCO:
    """Load the original COCO annotations once for several evaluations.

    Args:
        annotation_path: Original COCO JSON with category-specific crowd annotations.

    Returns:
        The pycocotools ground-truth object.
    """
    with redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation_path))
    ground_truth.dataset.setdefault("info", {})
    return ground_truth


def load_detections(ground_truth: COCO, predictions: list[dict]) -> COCO:
    """Index COCO-format detections against the ground truth, allowing an empty list.

    Args:
        ground_truth: The pycocotools ground-truth object.
        predictions: COCO-format detections in original-image pixel xywh.

    Returns:
        The pycocotools detection object.
    """
    with redirect_stdout(io.StringIO()):
        if predictions:
            return ground_truth.loadRes(predictions)
        detections = COCO()
        detections.dataset = {
            "images": ground_truth.dataset["images"],
            "categories": ground_truth.dataset["categories"],
            "annotations": [],
        }
        detections.createIndex()
    return detections


def evaluate_area_bins(
    ground_truth: COCO,
    detections: COCO,
    image_ids: list[int],
    area_bins: dict[str, tuple[float, float]],
    max_detections: int = 100,
) -> dict[str, float | None]:
    """Return COCO mAP over IoU 0.50 to 0.95 for each ground-truth area range.

    The protocol is the canonical COCO one with category-specific crowds; only the area
    ranges differ, so the all range at maxDets 100 equals the canonical mAP.

    Args:
        ground_truth: The pycocotools ground-truth object.
        detections: The pycocotools detection object.
        image_ids: Evaluated image ids.
        area_bins: Area ranges in original-image square pixels keyed by name.
        max_detections: Detections per image and category that are evaluated.

    Returns:
        mAP keyed by area range name, None where the range has no objects.
    """
    with redirect_stdout(io.StringIO()):
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.catIds = list(range(1, 9))
        evaluator.params.maxDets = [1, 10, max_detections]
        evaluator.params.areaRng = [list(bounds) for bounds in area_bins.values()]
        evaluator.params.areaRngLbl = list(area_bins)
        evaluator.evaluate()
        evaluator.accumulate()
    precision = np.asarray(evaluator.eval["precision"])[..., -1]
    results = {}
    for index, name in enumerate(area_bins):
        values = precision[:, :, :, index]
        values = values[values > -1]
        results[name] = float(np.mean(values)) if values.size else None
    return results


def count_objects_per_image(ground_truth: COCO) -> dict[int, int]:
    """Count target objects per image, leaving out crowd and ignored-region records.

    Args:
        ground_truth: The pycocotools ground-truth object.

    Returns:
        Object counts keyed by image id, including images without objects.
    """
    counts = {image_id: 0 for image_id in ground_truth.getImgIds()}
    for annotation in ground_truth.dataset["annotations"]:
        if not annotation.get("iscrowd", 0) and 1 <= annotation["category_id"] <= 8:
            counts[annotation["image_id"]] += 1
    return counts


def bin_images_by_density(
    counts: dict[int, int], density_bins: dict[str, tuple[int, int]]
) -> dict[str, list[int]]:
    """Group image ids by their number of target objects.

    Args:
        counts: Object counts keyed by image id.
        density_bins: Inclusive object-count ranges keyed by name.

    Returns:
        Sorted image ids keyed by bin name; images outside every range are left out.
    """
    return {
        name: sorted(image_id for image_id, count in counts.items() if low <= count <= high)
        for name, (low, high) in density_bins.items()
    }


def evaluate_density_bins(
    ground_truth: COCO,
    detections: COCO,
    image_ids: list[int],
    density_bins: dict[str, tuple[int, int]],
    max_detections: int = 100,
) -> dict[str, dict]:
    """Return COCO mAP and AP_small on the images of each scene-density range.

    Args:
        ground_truth: The pycocotools ground-truth object.
        detections: The pycocotools detection object.
        image_ids: Evaluated image ids.
        density_bins: Inclusive object-count ranges keyed by name.
        max_detections: Detections per image and category that are evaluated.

    Returns:
        Per bin, the image count, the object count, mAP and AP_small.
    """
    counts = count_objects_per_image(ground_truth)
    selected = set(image_ids)
    results = {}
    for name, ids in bin_images_by_density(counts, density_bins).items():
        ids = [image_id for image_id in ids if image_id in selected]
        metrics = (
            evaluate_area_bins(
                ground_truth,
                detections,
                ids,
                {"all": AREA_BINS["all"], "small": (0.0, 32.0**2)},
                max_detections,
            )
            if ids
            else {"all": None, "small": None}
        )
        results[name] = {
            "images": len(ids),
            "objects": sum(counts[image_id] for image_id in ids),
            "map": metrics["all"],
            "ap_small": metrics["small"],
        }
    return results


def group_predictions(predictions: list[dict]) -> dict[int, dict[str, torch.Tensor]]:
    """Collect COCO-format detections per image as xyxy tensors.

    Args:
        predictions: COCO-format detections in original-image pixel xywh.

    Returns:
        Per image id, boxes as xyxy, scores and category ids.
    """
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in predictions:
        grouped[record["image_id"]].append(record)
    return {
        image_id: {
            "boxes": box_convert(
                torch.tensor([record["bbox"] for record in records], dtype=torch.float32),
                "xywh",
                "xyxy",
            ),
            "scores": torch.tensor([record["score"] for record in records], dtype=torch.float32),
            "categories": torch.tensor([record["category_id"] for record in records]),
        }
        for image_id, records in grouped.items()
    }


def count_near_duplicates(
    predictions: list[dict], score_threshold: float, iou_threshold: float
) -> dict:
    """Count pairs of same-class detections that overlap like duplicates.

    A detector with class-wise NMS at IoU t cannot produce a same-class pair above t.
    A set-prediction detector has no such step, so the count shows whether bipartite
    matching taught its queries not to predict the same object twice.

    Args:
        predictions: COCO-format detections in original-image pixel xywh.
        score_threshold: Minimum score of both detections of a pair.
        iou_threshold: Minimum IoU of a counted pair.

    Returns:
        Kept detections, duplicate pairs, images with any pair and pairs per 100 detections.
    """
    detections, pairs, images_with_pairs = 0, 0, 0
    for image in group_predictions(predictions).values():
        is_kept = image["scores"] >= score_threshold
        boxes, categories = image["boxes"][is_kept], image["categories"][is_kept]
        detections += int(is_kept.sum())
        image_pairs = 0
        for category in categories.unique().tolist():
            class_boxes = boxes[categories == category]
            overlaps = box_iou(class_boxes, class_boxes).triu(diagonal=1)
            image_pairs += int((overlaps >= iou_threshold).sum())
        pairs += image_pairs
        images_with_pairs += image_pairs > 0
    return {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "detections": detections,
        "duplicate_pairs": pairs,
        "images_with_pairs": images_with_pairs,
        "pairs_per_100_detections": 100 * pairs / detections if detections else None,
    }


def suppress_duplicates(predictions: list[dict], iou_threshold: float) -> list[dict]:
    """Apply class-wise NMS after the fact to COCO-format detections.

    Args:
        predictions: COCO-format detections in original-image pixel xywh.
        iou_threshold: IoU above which the lower-scoring same-class detection is removed.

    Returns:
        The kept detections, in their original record form.
    """
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in predictions:
        grouped[record["image_id"]].append(record)
    kept = []
    for records in grouped.values():
        boxes = box_convert(
            torch.tensor([record["bbox"] for record in records], dtype=torch.float32),
            "xywh",
            "xyxy",
        )
        scores = torch.tensor([record["score"] for record in records], dtype=torch.float32)
        categories = torch.tensor([record["category_id"] for record in records])
        kept.extend(
            records[index]
            for index in batched_nms(boxes, scores, categories, iou_threshold).tolist()
        )
    return kept


def read_validation_curve(directory: Path) -> list[dict]:
    """Read per-epoch validation from Faster R-CNN, RetinaNet or RT-DETR metrics.json.

    Faster R-CNN and RT-DETR records nest the canonical metrics under max_dets_100;
    RetinaNet records keep them at the top level. Both keep the cumulative image
    exposures and the summed training batch time of each epoch.

    Args:
        directory: Experiment directory with metrics.json.

    Returns:
        One row per validation with epoch, exposures, training minutes, mAP, AP50 and
        AP_small, or an empty list when the file is missing.
    """
    path = directory / "metrics.json"
    if not path.is_file():
        return []
    rows = []
    for record in json.loads(path.read_text()):
        metrics = record["metrics"].get("max_dets_100", record["metrics"])
        rows.append(
            {
                "epoch": record["epoch"],
                "image_exposures": record.get("image_exposures"),
                "train_minutes": (
                    record["wall_train_seconds"] / 60
                    if record.get("wall_train_seconds") is not None
                    else None
                ),
                "map": metrics["map"],
                "ap50": metrics["ap50"],
                "ap_small": metrics["ap_small"],
            }
        )
    return rows


def summarize_convergence(curve: list[dict], history: list[dict] | None = None) -> dict:
    """Summarize whether validation and training loss were still moving at the end.

    Args:
        curve: Rows from read_validation_curve.
        history: Optional step records with epoch and loss.

    Returns:
        Final and best mAP, the best epoch, the mAP change over the last epoch, the
        relative change of the mean training loss between the last two epochs, and
        is_still_improving when the last epoch added more than half a point of mAP.
    """
    if not curve:
        return {"status": "Not run"}
    values = [row["map"] for row in curve]
    best = max(range(len(values)), key=lambda index: values[index])
    gain = values[-1] - values[-2] if len(values) > 1 else None
    loss_change = None
    if history:
        by_epoch: dict[int, list[float]] = defaultdict(list)
        for row in history:
            if math.isfinite(row["loss"]):
                by_epoch[row["epoch"]].append(row["loss"])
        epochs = sorted(by_epoch)
        if len(epochs) > 1:
            previous = float(np.mean(by_epoch[epochs[-2]]))
            last = float(np.mean(by_epoch[epochs[-1]]))
            loss_change = (last - previous) / previous
    return {
        "status": "Measured",
        "epochs": len(values),
        "final_map": values[-1],
        "best_map": values[best],
        "best_epoch": curve[best]["epoch"],
        "last_epoch_gain": gain,
        "last_epoch_loss_change": loss_change,
        "is_still_improving": gain is not None and gain > CONVERGENCE_GAIN,
    }


def analyze_predictions(
    ground_truth: COCO, predictions: list[dict], image_ids: list[int], is_query_based: bool
) -> dict:
    """Run the size, density and duplicate analyses on one detector's validation output.

    For a query-based detector, also evaluate it after class-wise NMS at each threshold;
    if matching already removed duplicates, NMS cannot raise its mAP.

    Args:
        ground_truth: The pycocotools ground-truth object.
        predictions: COCO-format detections of the final model on the validation frames.
        image_ids: Evaluated image ids.
        is_query_based: Whether to add the post-hoc NMS evaluation.

    Returns:
        Area-bin mAP, density-bin metrics, duplicate counts and the optional NMS results.
    """
    detections = load_detections(ground_truth, predictions)
    result = {
        "detections_per_image": len(predictions) / len(image_ids),
        "area_bins": evaluate_area_bins(ground_truth, detections, image_ids, AREA_BINS),
        "density_bins": evaluate_density_bins(ground_truth, detections, image_ids, DENSITY_BINS),
        "near_duplicates": [
            count_near_duplicates(predictions, score, iou)
            for score in DUPLICATE_SCORE_THRESHOLDS
            for iou in DUPLICATE_IOU_THRESHOLDS
        ],
    }
    if is_query_based:
        result["post_hoc_nms"] = [
            {
                "iou_threshold": threshold,
                "detections_per_image": len(suppressed) / len(image_ids),
                "map": evaluate_area_bins(
                    ground_truth,
                    load_detections(ground_truth, suppressed),
                    image_ids,
                    {"all": AREA_BINS["all"]},
                )["all"],
            }
            for threshold in NMS_IOU_THRESHOLDS
            for suppressed in [suppress_duplicates(predictions, threshold)]
        ]
    return result


def run_comparison_analysis(annotation_path: Path, detectors: list[dict], output: Path) -> dict:
    """Analyze every available detector and write the comparison record.

    Args:
        annotation_path: Original validation COCO JSON.
        detectors: Per detector, key, label, family, directory, predictions file name and
            is_query_based.
        output: Directory that receives analysis.json.

    Returns:
        Per detector, its status, validation curve, convergence summary and, where its
        predictions exist, the prediction analyses.
    """
    ground_truth = load_ground_truth(annotation_path)
    image_ids = sorted(ground_truth.getImgIds())
    records = []
    for detector in detectors:
        directory = Path(detector["directory"])
        curve = read_validation_curve(directory)
        history_path = directory / "history.json"
        history = json.loads(history_path.read_text()) if history_path.is_file() else None
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        record = {
            "key": detector["key"],
            "label": detector["label"],
            "family": detector["family"],
            "is_query_based": detector["is_query_based"],
            "status": "Not run" if not curve else "Measured",
            "image_exposures": status.get("image_exposures"),
            "train_minutes": (
                status["wall_train_seconds"] / 60 if "wall_train_seconds" in status else None
            ),
            "curve": curve,
            "convergence": summarize_convergence(curve, history),
        }
        predictions_path = directory / detector["predictions_file"]
        if predictions_path.is_file():
            predictions = json.loads(predictions_path.read_text())
            record["predictions"] = analyze_predictions(
                ground_truth, predictions, image_ids, detector["is_query_based"]
            )
            del predictions
        records.append(record)
    result = {
        "annotation_file": annotation_path.name,
        "image_count": len(image_ids),
        "area_bins": AREA_BINS,
        "density_bins": DENSITY_BINS,
        "max_detections": 100,
        "detectors": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
