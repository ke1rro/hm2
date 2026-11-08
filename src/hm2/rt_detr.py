import importlib
import math
import random
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LinearLR, LRScheduler, MultiStepLR, SequentialLR
from torchvision.ops import box_convert, generalized_box_iou

from hm2.data import CLASS_NAMES
from hm2.faster_rcnn import ignore_coverage

NUM_CLASSES = len(CLASS_NAMES)
CLASS_PRIOR_PROBABILITY = 0.01
BACKBONE_PREFIX = "detector.model.backbone."
LOSS_PARTS = ("final", "auxiliary", "encoder", "denoising")
LOSS_TERMS = ("class", "l1", "box")
LOSS_COLUMNS = ("loss", *(f"loss_{term}_{part}" for part in LOSS_PARTS for term in LOSS_TERMS))
DIAGNOSTIC_COLUMNS = (
    "matched_queries",
    "excluded_queries_final",
    "excluded_queries_all_layers",
    "denoising_groups",
)
Indices = list[tuple[Tensor, Tensor]]


def compute_resize_size(height: int, width: int, min_size: int, max_size: int) -> tuple[int, int]:
    """Return the model-input size that keeps the aspect ratio under the B1 resize rule.

    The shorter side becomes min_size unless the longer side would exceed max_size, in
    which case the longer side becomes max_size, as in torchvision detection transforms.

    Args:
        height: Original image height in pixels.
        width: Original image width in pixels.
        min_size: Target shorter side in model pixels.
        max_size: Maximum longer side in model pixels.

    Returns:
        Resized height and width in model pixels.
    """
    scale = min(min_size / min(height, width), max_size / max(height, width))
    return round(height * scale), round(width * scale)


def padded_canvas_size(sizes: list[tuple[int, int]], size_divisor: int) -> tuple[int, int]:
    """Return the batch canvas that holds every resized image, rounded up to a stride multiple.

    Args:
        sizes: Resized height and width of each image in the batch.
        size_divisor: Largest backbone stride that the canvas must be divisible by.

    Returns:
        Canvas height and width in model pixels; images sit at the top-left corner.
    """
    height = max(size[0] for size in sizes)
    width = max(size[1] for size in sizes)
    return (
        math.ceil(height / size_divisor) * size_divisor,
        math.ceil(width / size_divisor) * size_divisor,
    )


def labels_to_detr_classes(labels: Tensor) -> Tensor:
    """Map torchvision foreground labels 1 to 8 to DETR class ids 0 to 7.

    DETR scores each class with its own sigmoid, so it has no background class. The ids
    equal the submission class ids.

    Args:
        labels: Torchvision labels where 0 would be background.

    Returns:
        Class ids for the DETR head.
    """
    return labels - 1


def detr_classes_to_labels(classes: Tensor) -> Tensor:
    """Map DETR class ids 0 to 7 back to torchvision foreground labels 1 to 8.

    Args:
        classes: Class ids predicted by the DETR head.

    Returns:
        Labels in the convention of predictions_to_coco and build_submission.
    """
    return classes + 1


def scale_boxes(boxes: Tensor, scale_x: float, scale_y: float) -> Tensor:
    """Scale xyxy pixel boxes from the original image to the resized image.

    Args:
        boxes: Boxes as xyxy original-image pixels with shape (N, 4).
        scale_x: Resized width divided by original width.
        scale_y: Resized height divided by original height.

    Returns:
        Boxes as xyxy resized-image pixels.
    """
    return boxes * boxes.new_tensor([scale_x, scale_y, scale_x, scale_y])


def boxes_to_normalized_cxcywh(
    boxes: Tensor, scale_x: float, scale_y: float, canvas: tuple[int, int]
) -> Tensor:
    """Convert original-pixel xyxy targets to DETR targets on the padded canvas.

    RT-DETR predicts boxes relative to the whole input tensor, so the normalization uses
    the padded canvas, not the resized image or the original image.

    Args:
        boxes: Boxes as xyxy original-image pixels with shape (N, 4).
        scale_x: Resized width divided by original width.
        scale_y: Resized height divided by original height.
        canvas: Padded canvas height and width in model pixels.

    Returns:
        Center x, center y, width and height divided by the canvas width and height.
    """
    height, width = canvas
    scaled = box_convert(scale_boxes(boxes, scale_x, scale_y), "xyxy", "cxcywh")
    return scaled / scaled.new_tensor([width, height, width, height])


def normalized_cxcywh_to_pixels(boxes: Tensor, canvas: tuple[int, int]) -> Tensor:
    """Convert normalized canvas cxcywh boxes to xyxy canvas pixels.

    Args:
        boxes: Normalized cxcywh boxes with shape (..., 4).
        canvas: Padded canvas height and width in model pixels.

    Returns:
        Boxes as xyxy canvas pixels.
    """
    height, width = canvas
    pixels = boxes * boxes.new_tensor([width, height, width, height])
    return box_convert(pixels, "cxcywh", "xyxy")


def decode_predictions(
    logits: Tensor,
    boxes: Tensor,
    canvas: tuple[int, int],
    resized_sizes: list[tuple[int, int]],
    original_sizes: list[tuple[int, int]],
    max_detections: int,
) -> list[dict[str, Tensor]]:
    """Turn final-layer DETR outputs into original-pixel detections without NMS.

    Every (query, class) pair is a candidate scored by its sigmoid probability, and the
    highest max_detections candidates are kept, as in the official RT-DETR postprocess.

    Args:
        logits: Class logits with shape (B, Q, 8).
        boxes: Normalized canvas cxcywh boxes with shape (B, Q, 4).
        canvas: Padded canvas height and width in model pixels.
        resized_sizes: Resized height and width of each image.
        original_sizes: Original height and width of each image.
        max_detections: Maximum detections kept per image.

    Returns:
        One dictionary per image with xyxy original-pixel boxes, scores and labels 1 to 8.
    """
    predictions = []
    num_classes = logits.shape[-1]
    for image_logits, image_boxes, resized, original in zip(
        logits.float(), boxes.float(), resized_sizes, original_sizes, strict=True
    ):
        scores, flat_indices = (
            image_logits.sigmoid().flatten().topk(min(max_detections, image_logits.numel()))
        )
        queries = flat_indices // num_classes
        classes = flat_indices % num_classes
        pixels = normalized_cxcywh_to_pixels(image_boxes, canvas)[queries]
        scale_x, scale_y = resized[1] / original[1], resized[0] / original[0]
        pixels = pixels / pixels.new_tensor([scale_x, scale_y, scale_x, scale_y])
        pixels[:, [0, 2]] = pixels[:, [0, 2]].clamp(0, original[1])
        pixels[:, [1, 3]] = pixels[:, [1, 3]].clamp(0, original[0])
        predictions.append(
            {"boxes": pixels, "scores": scores, "labels": detr_classes_to_labels(classes)}
        )
    return predictions


def paired_iou(first: Tensor, second: Tensor) -> Tensor:
    """Return the IoU of each box with the box at the same index.

    Args:
        first: Boxes as xyxy with shape (N, 4).
        second: Boxes as xyxy with shape (N, 4).

    Returns:
        IoU values with shape (N,).
    """
    top_left = torch.max(first[:, :2], second[:, :2])
    bottom_right = torch.min(first[:, 2:], second[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    union = (
        (first[:, 2:] - first[:, :2]).prod(dim=1)
        + (second[:, 2:] - second[:, :2]).prod(dim=1)
        - intersection
    )
    return intersection / union


def paired_giou(first: Tensor, second: Tensor) -> Tensor:
    """Return the generalized IoU of each box with the box at the same index.

    Args:
        first: Boxes as xyxy with shape (N, 4).
        second: Boxes as xyxy with shape (N, 4).

    Returns:
        GIoU values in [-1, 1] with shape (N,).
    """
    top_left = torch.max(first[:, :2], second[:, :2])
    bottom_right = torch.min(first[:, 2:], second[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    union = (
        (first[:, 2:] - first[:, :2]).prod(dim=1)
        + (second[:, 2:] - second[:, :2]).prod(dim=1)
        - intersection
    )
    enclosing = (
        torch.max(first[:, 2:], second[:, 2:]) - torch.min(first[:, :2], second[:, :2])
    ).prod(dim=1)
    return intersection / union - (enclosing - union) / enclosing


def paired_nwd(first: Tensor, second: Tensor, constant: float) -> Tensor:
    """Return the normalized Wasserstein distance similarity of paired boxes.

    Each box is modelled as a 2D Gaussian with its center as mean and half its width and
    height as standard deviations. The 2-Wasserstein distance between two such Gaussians
    is the Euclidean distance between (cx, cy, w / 2, h / 2) vectors, and the similarity is
    exp(-distance / constant). Unlike IoU it stays informative for tiny or disjoint boxes.

    Args:
        first: Boxes as cxcywh pixels with shape (N, 4).
        second: Boxes as cxcywh pixels with shape (N, 4).
        constant: Normalizing size in the same pixels, close to a typical object size.

    Returns:
        Similarity values in (0, 1] with shape (N,).
    """
    center = (first[:, :2] - second[:, :2]).pow(2).sum(dim=1)
    shape = ((first[:, 2:] - second[:, 2:]) / 2).pow(2).sum(dim=1)
    return torch.exp(-torch.sqrt(center + shape + 1e-7) / constant)


def match_queries(
    logits: Tensor, boxes: Tensor, targets: list[dict[str, Tensor]], matcher_settings: dict
) -> Indices:
    """Assign each ground-truth object to one query by Hungarian matching.

    The cost is the RT-DETR cost: a focal classification cost, the L1 distance between
    normalized cxcywh boxes and the negative GIoU, each multiplied by its weight.

    Args:
        logits: Class logits with shape (B, Q, 8).
        boxes: Normalized canvas cxcywh boxes with shape (B, Q, 4).
        targets: Per image, class_labels 0 to 7 and normalized canvas cxcywh boxes.
        matcher_settings: class_cost, l1_cost, giou_cost, focal_alpha and focal_gamma.

    Returns:
        Per image, matched query indices and the matched target indices.
    """
    batch_size, num_queries = logits.shape[:2]
    sizes = [len(target["class_labels"]) for target in targets]
    empty = torch.zeros(0, dtype=torch.int64)
    if sum(sizes) == 0:
        return [(empty, empty) for _ in sizes]
    with torch.no_grad():
        alpha, gamma = matcher_settings["focal_alpha"], matcher_settings["focal_gamma"]
        probability = logits.flatten(0, 1).float().sigmoid()
        probability = probability[:, torch.cat([target["class_labels"] for target in targets])]
        negative = (1 - alpha) * probability.pow(gamma) * -(1 - probability + 1e-8).log()
        positive = alpha * (1 - probability).pow(gamma) * -(probability + 1e-8).log()
        predicted = boxes.flatten(0, 1).float()
        target_boxes = torch.cat([target["boxes"] for target in targets]).float()
        cost = (
            matcher_settings["l1_cost"] * torch.cdist(predicted, target_boxes, p=1)
            + matcher_settings["class_cost"] * (positive - negative)
            + matcher_settings["giou_cost"]
            * -generalized_box_iou(
                box_convert(predicted, "cxcywh", "xyxy"),
                box_convert(target_boxes, "cxcywh", "xyxy"),
            )
        )
        maximum = torch.finfo(cost.dtype).max
        cost = torch.nan_to_num(cost, nan=maximum, posinf=maximum, neginf=maximum)
        cost = cost.view(batch_size, num_queries, -1).cpu()
    indices = []
    for image, block in enumerate(cost.split(sizes, -1)):
        rows, columns = linear_sum_assignment(block[image].numpy())
        indices.append(
            (torch.as_tensor(rows, dtype=torch.int64), torch.as_tensor(columns, dtype=torch.int64))
        )
    return indices


def denoising_indices(denoising_meta: dict, targets: list[dict[str, Tensor]]) -> Indices:
    """Pair each positive denoising query with the object it was built from.

    Args:
        denoising_meta: The denoising_meta_values of the RT-DETR output.
        targets: Per image, class_labels and boxes.

    Returns:
        Per image, positive denoising query indices and the repeated target indices.
    """
    groups = denoising_meta["dn_num_group"]
    indices = []
    for positives, target in zip(denoising_meta["dn_positive_idx"], targets, strict=True):
        count = len(target["class_labels"])
        indices.append(
            (positives.cpu().to(torch.int64), torch.arange(count, dtype=torch.int64).tile(groups))
        )
    return indices


def unmatched_ignore_keep(
    boxes: Tensor,
    indices: Indices,
    ignore_boxes: list[Tensor],
    canvas: tuple[int, int],
    threshold: float,
) -> Tensor:
    """Mark unmatched queries whose predicted box lies mostly inside one ignored region.

    The classification loss of a marked query is dropped: it is neither pushed towards
    background nor treated as an object. Matched queries are never marked, so the
    matcher and every positive target stay unchanged. The rule mirrors the Faster R-CNN
    one, with the query's own detached box in place of a fixed anchor.

    Args:
        boxes: Normalized canvas cxcywh boxes with shape (B, Q, 4).
        indices: Matched query and target indices for each image.
        ignore_boxes: Ignored regions as xyxy canvas pixels for each image.
        canvas: Padded canvas height and width in model pixels.
        threshold: Minimum covered fraction of the box area that drops the query.

    Returns:
        Float weights with shape (B, Q): 0 for dropped queries and 1 otherwise.
    """
    keep = boxes.new_ones(boxes.shape[:2], dtype=torch.float32)
    for image, (image_boxes, (queries, _), regions) in enumerate(
        zip(boxes.detach(), indices, ignore_boxes, strict=True)
    ):
        if regions.numel() == 0:
            continue
        is_ignored = (
            ignore_coverage(normalized_cxcywh_to_pixels(image_boxes.float(), canvas), regions)
            >= threshold
        )
        is_ignored[queries.to(is_ignored.device)] = False
        keep[image, is_ignored] = 0.0
    return keep


def compute_layer_losses(
    logits: Tensor,
    boxes: Tensor,
    targets: list[dict[str, Tensor]],
    indices: Indices,
    num_boxes: float,
    loss_settings: dict,
    canvas: tuple[int, int],
    keep: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the unweighted varifocal, L1 and box losses of one decoder output.

    The varifocal loss uses the IoU of each matched pair as the soft target of its class,
    and an alpha * p^gamma weight on every negative, as in RT-DETR. The optional keep
    weights remove dropped queries from the classification sum. The box loss is GIoU, or
    1 - NWD when loss_settings selects nwd, computed in canvas pixels.

    Args:
        logits: Class logits with shape (B, Q, 8).
        boxes: Normalized canvas cxcywh boxes with shape (B, Q, 4).
        targets: Per image, class_labels 0 to 7 and normalized canvas cxcywh boxes.
        indices: Matched query and target indices for each image.
        num_boxes: Normalizer, the number of objects in the batch times the group count.
        loss_settings: focal_alpha, focal_gamma, box_loss and nwd_constant.
        canvas: Padded canvas height and width in model pixels.
        keep: Optional (B, Q) weights for the classification loss.

    Returns:
        Classification, L1 and box losses as scalars.
    """
    logits, boxes = logits.float(), boxes.float()
    device = logits.device
    batch_index = torch.cat(
        [torch.full_like(queries, image) for image, (queries, _) in enumerate(indices)]
    ).to(device)
    query_index = torch.cat([queries for queries, _ in indices]).to(device)
    target_boxes = torch.cat(
        [target["boxes"][columns.to(device)] for target, (_, columns) in zip(targets, indices)]
    ).float()
    target_classes = torch.cat(
        [
            target["class_labels"][columns.to(device)]
            for target, (_, columns) in zip(targets, indices)
        ]
    )
    matched_boxes = boxes[batch_index, query_index]
    matched_corners = box_convert(matched_boxes, "cxcywh", "xyxy")
    target_corners = box_convert(target_boxes, "cxcywh", "xyxy")
    ious = paired_iou(matched_corners.detach(), target_corners)
    one_hot = torch.zeros_like(logits)
    one_hot[batch_index, query_index, target_classes] = 1.0
    soft_target = torch.zeros_like(logits)
    soft_target[batch_index, query_index, target_classes] = ious
    probability = logits.detach().sigmoid()
    weight = (
        loss_settings["focal_alpha"] * probability.pow(loss_settings["focal_gamma"]) * (1 - one_hot)
        + soft_target
    )
    if keep is not None:
        weight = weight * keep[..., None]
    class_loss = (
        F.binary_cross_entropy_with_logits(logits, soft_target, weight=weight, reduction="sum")
        / num_boxes
    )
    l1_loss = (matched_boxes - target_boxes).abs().sum() / num_boxes
    if loss_settings["box_loss"] == "nwd":
        height, width = canvas
        size = matched_boxes.new_tensor([width, height, width, height])
        similarity = paired_nwd(
            matched_boxes * size, target_boxes * size, loss_settings["nwd_constant"]
        )
    else:
        similarity = paired_giou(matched_corners, target_corners)
    box_loss = (1 - similarity).sum() / num_boxes
    return class_loss, l1_loss, box_loss


def compute_detection_losses(
    intermediate_logits: Tensor,
    intermediate_boxes: Tensor,
    encoder_logits: Tensor,
    encoder_boxes: Tensor,
    denoising_meta: dict | None,
    targets: list[dict[str, Tensor]],
    ignore_boxes: list[Tensor],
    canvas: tuple[int, int],
    loss_settings: dict,
    matcher_settings: dict,
    ignore_threshold: float | None,
) -> tuple[dict[str, Tensor], dict[str, float]]:
    """Compute the weighted RT-DETR set-prediction losses with optional ignore handling.

    Split the decoder outputs into denoising and matching queries. Match the final layer,
    every intermediate layer and the encoder top-k proposals to the objects independently
    and add their varifocal, L1 and box losses, normalized by the object count. Denoising
    queries use their known objects instead of matching, normalized by the object count
    times the group count. With an ignore threshold, unmatched matching queries inside
    ignored regions get no classification loss; denoising queries are never dropped.
    Matching and losses run in float32 even under mixed precision.

    Args:
        intermediate_logits: Decoder logits with shape (B, L, Q_total, 8).
        intermediate_boxes: Decoder normalized cxcywh boxes with shape (B, L, Q_total, 4).
        encoder_logits: Encoder top-k logits with shape (B, Q, 8).
        encoder_boxes: Encoder top-k normalized cxcywh boxes with shape (B, Q, 4).
        denoising_meta: The denoising_meta_values of the output, or None without denoising.
        targets: Per image, class_labels 0 to 7 and normalized canvas cxcywh boxes.
        ignore_boxes: Ignored regions as xyxy canvas pixels for each image.
        canvas: Padded canvas height and width in model pixels.
        loss_settings: class_weight, l1_weight, box_weight, focal_alpha, focal_gamma,
            box_loss and nwd_constant.
        matcher_settings: class_cost, l1_cost, giou_cost, focal_alpha and focal_gamma.
        ignore_threshold: Covered fraction that drops an unmatched query, or None.

    Returns:
        Weighted losses keyed loss_<term>_<part>, and per-step diagnostic counts.
    """
    with torch.autocast(intermediate_logits.device.type, enabled=False):
        logits, boxes = intermediate_logits.float(), intermediate_boxes.float()
        denoising_parts = []
        if denoising_meta is not None:
            denoising_logits, logits = logits.split(denoising_meta["dn_num_split"], dim=2)
            denoising_boxes, boxes = boxes.split(denoising_meta["dn_num_split"], dim=2)
            denoising_parts = [
                (denoising_logits[:, layer], denoising_boxes[:, layer])
                for layer in range(denoising_logits.shape[1])
            ]
        layers = logits.shape[1]
        parts = [
            ("final" if layer == layers - 1 else "auxiliary", logits[:, layer], boxes[:, layer])
            for layer in range(layers)
        ]
        parts.append(("encoder", encoder_logits.float(), encoder_boxes.float()))
        num_boxes = float(max(sum(len(target["class_labels"]) for target in targets), 1))
        weights = (
            loss_settings["class_weight"],
            loss_settings["l1_weight"],
            loss_settings["box_weight"],
        )
        zero = logits.new_zeros(())
        losses = {f"loss_{term}_{part}": zero for part in LOSS_PARTS for term in LOSS_TERMS}
        diagnostics = {name: 0.0 for name in DIAGNOSTIC_COLUMNS}
        for part, part_logits, part_boxes in parts:
            indices = match_queries(part_logits, part_boxes, targets, matcher_settings)
            keep = (
                unmatched_ignore_keep(part_boxes, indices, ignore_boxes, canvas, ignore_threshold)
                if ignore_threshold is not None
                else None
            )
            values = compute_layer_losses(
                part_logits, part_boxes, targets, indices, num_boxes, loss_settings, canvas, keep
            )
            for term, value, weight in zip(LOSS_TERMS, values, weights, strict=True):
                losses[f"loss_{term}_{part}"] = losses[f"loss_{term}_{part}"] + weight * value
            excluded = float((keep == 0).sum()) if keep is not None else 0.0
            diagnostics["excluded_queries_all_layers"] += excluded
            if part == "final":
                diagnostics["matched_queries"] = float(sum(len(rows) for rows, _ in indices))
                diagnostics["excluded_queries_final"] = excluded
        if denoising_meta is not None:
            indices = denoising_indices(denoising_meta, targets)
            groups = denoising_meta["dn_num_group"]
            diagnostics["denoising_groups"] = float(groups)
            for part_logits, part_boxes in denoising_parts:
                values = compute_layer_losses(
                    part_logits,
                    part_boxes,
                    targets,
                    indices,
                    num_boxes * groups,
                    loss_settings,
                    canvas,
                )
                for term, value, weight in zip(LOSS_TERMS, values, weights, strict=True):
                    losses[f"loss_{term}_denoising"] = (
                        losses[f"loss_{term}_denoising"] + weight * value
                    )
    return losses, diagnostics


class QueryDetector(nn.Module):
    """RT-DETR wrapped in the torchvision detection interface used by the shared training loop.

    In training mode, a call with images and targets returns the weighted loss dictionary;
    in evaluation mode, a call with images returns original-pixel detections. Images are
    float CHW tensors in [0, 1] and original pixels; targets hold xyxy original-pixel boxes,
    labels 1 to 8 and optional ignore_boxes, as produced by IgnoreAwareDetectionDataset.

    Attributes:
        detector: The Hugging Face RTDetrV2ForObjectDetection model.
        last_diagnostics: Diagnostic counts of the most recent training call.

    Args:
        detector: The Hugging Face RTDetrV2ForObjectDetection model.
        model_settings: min_size, max_size, size_divisor, detections_per_img,
            is_ignore_aware and ignore_threshold.
        loss_settings: Loss weights and options passed to compute_detection_losses.
        matcher_settings: Matching cost weights passed to match_queries.
        image_mean: Optional per-channel mean subtracted after the [0, 1] rescale.
        image_std: Optional per-channel standard deviation used with image_mean.
    """

    def __init__(
        self,
        detector: nn.Module,
        model_settings: dict,
        loss_settings: dict,
        matcher_settings: dict,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.min_size = model_settings["min_size"]
        self.max_size = model_settings["max_size"]
        self.size_divisor = model_settings["size_divisor"]
        self.detections_per_img = model_settings["detections_per_img"]
        self.ignore_threshold = (
            model_settings["ignore_threshold"] if model_settings["is_ignore_aware"] else None
        )
        self.loss_settings = loss_settings
        self.matcher_settings = matcher_settings
        self.image_mean = image_mean
        self.image_std = image_std
        self.last_diagnostics: dict[str, float] = {}

    def prepare_batch(
        self, images: list[Tensor], targets: list[dict[str, Tensor]] | None = None
    ) -> dict:
        """Resize, normalize and pad images, and convert targets to the DETR convention.

        Args:
            images: Float CHW images in [0, 1] and original pixels.
            targets: Optional targets with xyxy original-pixel boxes, labels 1 to 8 and
                optional ignore_boxes.

        Returns:
            pixel_values, canvas, resized_sizes, original_sizes and, with targets, labels
            (class ids 0 to 7 and normalized canvas cxcywh boxes) and ignore_boxes (xyxy
            canvas pixels).
        """
        original_sizes = [(int(image.shape[-2]), int(image.shape[-1])) for image in images]
        resized_sizes = [
            compute_resize_size(height, width, self.min_size, self.max_size)
            for height, width in original_sizes
        ]
        canvas = padded_canvas_size(resized_sizes, self.size_divisor)
        pixel_values = images[0].new_zeros((len(images), 3, *canvas))
        for index, (image, size) in enumerate(zip(images, resized_sizes, strict=True)):
            resized = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]
            if self.image_mean is not None and self.image_std is not None:
                mean = resized.new_tensor(self.image_mean)[:, None, None]
                std = resized.new_tensor(self.image_std)[:, None, None]
                resized = (resized - mean) / std
            pixel_values[index, :, : size[0], : size[1]] = resized
        batch = {
            "pixel_values": pixel_values,
            "canvas": canvas,
            "resized_sizes": resized_sizes,
            "original_sizes": original_sizes,
        }
        if targets is None:
            return batch
        scales = [
            (resized[1] / original[1], resized[0] / original[0])
            for resized, original in zip(resized_sizes, original_sizes, strict=True)
        ]
        batch["labels"] = [
            {
                "class_labels": labels_to_detr_classes(target["labels"]),
                "boxes": boxes_to_normalized_cxcywh(target["boxes"], *scale, canvas),
            }
            for target, scale in zip(targets, scales, strict=True)
        ]
        batch["ignore_boxes"] = [
            scale_boxes(target.get("ignore_boxes", target["boxes"].new_zeros((0, 4))), *scale)
            for target, scale in zip(targets, scales, strict=True)
        ]
        return batch

    def detect(
        self, images: list[Tensor], targets: list[dict[str, Tensor]] | None = None
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Return training losses, or detections in evaluation mode; this is the forward pass.

        Args:
            images: Float CHW images in [0, 1] and original pixels.
            targets: Targets for training; ignored in evaluation mode.

        Returns:
            The weighted loss dictionary in training mode, otherwise one detection
            dictionary per image with xyxy original-pixel boxes, scores and labels 1 to 8.
        """
        base_model = getattr(self.detector, "model")
        if self.training:
            if targets is None:
                raise ValueError("Training mode needs targets.")
            batch = self.prepare_batch(images, targets)
            outputs = base_model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            losses, self.last_diagnostics = compute_detection_losses(
                outputs.intermediate_logits,
                outputs.intermediate_reference_points,
                outputs.enc_topk_logits,
                outputs.enc_topk_bboxes,
                outputs.denoising_meta_values,
                batch["labels"],
                batch["ignore_boxes"],
                batch["canvas"],
                self.loss_settings,
                self.matcher_settings,
                self.ignore_threshold,
            )
            return losses
        batch = self.prepare_batch(images)
        outputs = base_model(pixel_values=batch["pixel_values"])
        return decode_predictions(
            outputs.intermediate_logits[:, -1],
            outputs.intermediate_reference_points[:, -1],
            batch["canvas"],
            batch["resized_sizes"],
            batch["original_sizes"],
            self.detections_per_img,
        )

    forward = detect


def reset_classification_heads(detector: nn.Module, seed: int) -> list[str]:
    """Initialize every eight-class head and the denoising label embedding with a fixed seed.

    Use the RT-DETR initialization: Xavier-uniform weights and a bias that makes every
    class start at probability 0.01, and a Xavier-uniform denoising label embedding.

    Args:
        detector: The RTDetrV2ForObjectDetection model with eight-class heads.
        seed: Seed of the initialization.

    Returns:
        Names of the reinitialized modules, for the run records.
    """
    torch.manual_seed(seed)
    bias = -math.log((1 - CLASS_PRIOR_PROBABILITY) / CLASS_PRIOR_PROBABILITY)
    names = []
    with torch.no_grad():
        for name, module in detector.named_modules():
            if isinstance(module, nn.Linear) and module.out_features == NUM_CLASSES:
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, bias)
                names.append(name)
            elif isinstance(module, nn.Embedding) and module.num_embeddings == NUM_CLASSES + 1:
                nn.init.xavier_uniform_(module.weight)
                names.append(name)
    return names


def build_detector_overrides(
    model_settings: dict, loss_settings: dict, matcher_settings: dict
) -> dict:
    """Translate the TOML settings into Hugging Face RTDetrV2Config overrides.

    The anchor and position-embedding sizes are unset so that both follow each input
    size; the checkpoint fixes them at 640 by 640, which would be wrong at 1024 by 576.
    The Hugging Face loss settings mirror ours so that the probe can compare the two.

    Args:
        model_settings: The model table.
        loss_settings: The loss table.
        matcher_settings: The matcher table.

    Returns:
        Keyword overrides for from_pretrained or RTDetrV2Config.from_pretrained.
    """
    return {
        "num_labels": NUM_CLASSES,
        "id2label": dict(enumerate(CLASS_NAMES)),
        "label2id": {name: index for index, name in enumerate(CLASS_NAMES)},
        "num_queries": model_settings["num_queries"],
        "num_denoising": model_settings["num_denoising"],
        "label_noise_ratio": model_settings["label_noise_ratio"],
        "box_noise_scale": model_settings["box_noise_scale"],
        "anchor_image_size": None,
        "eval_size": None,
        "initializer_bias_prior_prob": CLASS_PRIOR_PROBABILITY,
        "matcher_class_cost": matcher_settings["class_cost"],
        "matcher_bbox_cost": matcher_settings["l1_cost"],
        "matcher_giou_cost": matcher_settings["giou_cost"],
        "matcher_alpha": matcher_settings["focal_alpha"],
        "matcher_gamma": matcher_settings["focal_gamma"],
        "focal_loss_alpha": loss_settings["focal_alpha"],
        "focal_loss_gamma": loss_settings["focal_gamma"],
        "weight_loss_vfl": loss_settings["class_weight"],
        "weight_loss_bbox": loss_settings["l1_weight"],
        "weight_loss_giou": loss_settings["box_weight"],
    }


def build_rt_detr(
    model_settings: dict,
    loss_settings: dict,
    matcher_settings: dict,
    seed: int = 2026,
    pretrained: bool = True,
) -> QueryDetector:
    """Build RT-DETRv2 from the official COCO checkpoint with new eight-class heads.

    Load the checkpoint named by model_id and model_revision, keep every pretrained weight
    except the class heads, which are reinitialized with the seed, and read the image
    normalization from the checkpoint's preprocessor configuration. Without pretrained,
    only the configuration is loaded, for restoring a saved training checkpoint. The
    Hugging Face library is imported here, so tensor-only code does not need it.

    Args:
        model_settings: The model table, with model_id, model_revision and the sizes.
        loss_settings: The loss table.
        matcher_settings: The matcher table.
        seed: Seed of the new heads.
        pretrained: Whether to load the official COCO weights.

    Returns:
        The wrapped detector.
    """
    transformers = importlib.import_module("transformers")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model_id, revision = model_settings["model_id"], model_settings["model_revision"]
    overrides = build_detector_overrides(model_settings, loss_settings, matcher_settings)
    if pretrained:
        detector = transformers.RTDetrV2ForObjectDetection.from_pretrained(
            model_id, revision=revision, ignore_mismatched_sizes=True, **overrides
        )
        reset_classification_heads(detector, seed)
    else:
        config = transformers.RTDetrV2Config.from_pretrained(
            model_id, revision=revision, **overrides
        )
        detector = transformers.RTDetrV2ForObjectDetection(config)
    processor = transformers.AutoImageProcessor.from_pretrained(model_id, revision=revision)
    is_normalized = bool(getattr(processor, "do_normalize", False))
    return QueryDetector(
        detector,
        model_settings,
        loss_settings,
        matcher_settings,
        list(processor.image_mean) if is_normalized else None,
        list(processor.image_std) if is_normalized else None,
    )


def split_parameter_groups(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    learning_rate: float,
    backbone_learning_rate: float,
    weight_decay: float,
    backbone_prefix: str = BACKBONE_PREFIX,
) -> list[dict]:
    """Group trainable parameters by the RT-DETR recipe.

    Backbone parameters use the backbone learning rate with weight decay. Other biases and
    normalization weights, the one-dimensional parameters, use no weight decay. All other
    parameters use the main learning rate and weight decay. Frozen parameters are skipped.

    Args:
        named_parameters: Parameter names and parameters, as from named_parameters().
        learning_rate: Learning rate of the encoder, decoder and heads.
        backbone_learning_rate: Learning rate of the backbone.
        weight_decay: AdamW weight decay of decayed parameters.
        backbone_prefix: Name prefix of backbone parameters.

    Returns:
        Non-empty AdamW parameter groups, each with a name, params, lr and weight_decay.
    """
    groups: dict[str, dict] = {
        "backbone": {"params": [], "lr": backbone_learning_rate, "weight_decay": weight_decay},
        "decay": {"params": [], "lr": learning_rate, "weight_decay": weight_decay},
        "no_decay": {"params": [], "lr": learning_rate, "weight_decay": 0.0},
    }
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if name.startswith(backbone_prefix):
            group = "backbone"
        elif parameter.ndim <= 1:
            group = "no_decay"
        else:
            group = "decay"
        groups[group]["params"].append(parameter)
    return [{"name": name, **group} for name, group in groups.items() if group["params"]]


class GradientClippedAdamW(AdamW):
    """AdamW that clips the global gradient norm right before every update.

    The clipping is a step pre-hook. GradScaler.step unscales the gradients before calling
    step, so the clipping sees true gradients under mixed precision, and a skipped AMP
    update skips the clipping too.

    Attributes:
        max_norm: Maximum global L2 norm of the gradients.
        last_gradient_norm: Norm before clipping of the latest update, or None.

    Args:
        params: Parameter groups.
        max_norm: Maximum global L2 norm of the gradients.
        **kwargs: AdamW keyword arguments.
    """

    def __init__(self, params: Iterable, max_norm: float, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.max_norm = max_norm
        self.last_gradient_norm: float | None = None
        self.register_step_pre_hook(self._clip_gradients)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients and the recorded norm of the previous update.

        Args:
            set_to_none: Whether gradients are set to None instead of zero.
        """
        self.last_gradient_norm = None
        super().zero_grad(set_to_none)

    def _clip_gradients(self, optimizer: Optimizer, args: tuple, kwargs: dict) -> None:
        parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        self.last_gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, self.max_norm))


def make_rt_detr_optimizer(
    model: nn.Module, training_settings: dict, total_steps: int
) -> tuple[Optimizer, LRScheduler]:
    """Build the clipped AdamW optimizer with the same schedule shape as the CNN runs.

    The schedule warms up linearly from 1% of every group's learning rate and multiplies
    it by 0.1 at two thirds and at 90% of the updates, as the Faster R-CNN and RetinaNet
    runs do, so that the detector families differ in the optimizer but not the schedule.

    Args:
        model: The wrapped detector.
        training_settings: The training table, with learning_rate, backbone_learning_rate,
            weight_decay, gradient_clip_norm and warmup_updates.
        total_steps: Number of optimizer updates of the run.

    Returns:
        The optimizer and the step-based schedule.
    """
    optimizer = GradientClippedAdamW(
        split_parameter_groups(
            model.named_parameters(),
            training_settings["learning_rate"],
            training_settings["backbone_learning_rate"],
            training_settings["weight_decay"],
        ),
        max_norm=training_settings["gradient_clip_norm"],
    )
    warmup = min(training_settings["warmup_updates"], total_steps - 1)
    decay = MultiStepLR(
        optimizer,
        milestones=[
            max(1, int(total_steps * 2 / 3)) - warmup,
            max(2, int(total_steps * 0.9)) - warmup,
        ],
        gamma=0.1,
    )
    scheduler = (
        SequentialLR(
            optimizer,
            [LinearLR(optimizer, start_factor=0.01, total_iters=warmup), decay],
            milestones=[warmup],
        )
        if warmup
        else decay
    )
    return optimizer, scheduler


def compare_with_reference_loss(
    model: QueryDetector, images: list[Tensor], targets: list[dict]
) -> dict:
    """Compare our loss with the Hugging Face loss on one shared forward pass.

    Run the full Hugging Face detector once in training mode and float32 with labels, so
    both losses see the same outputs and the same denoising noise, and evaluate our loss
    with GIoU and without ignore handling, which is what the library implements.

    Args:
        model: The wrapped detector, on its training device.
        images: Float CHW images in [0, 1] and original pixels, on the same device.
        targets: Targets with xyxy original-pixel boxes and labels 1 to 8.

    Returns:
        Both totals, their relative difference and whether denoising was active.
    """
    was_training = model.training
    model.train()
    batch = model.prepare_batch(images, targets)
    with torch.no_grad():
        outputs = model.detector(pixel_values=batch["pixel_values"], labels=batch["labels"])
        losses, _ = compute_detection_losses(
            outputs.intermediate_logits,
            outputs.intermediate_reference_points,
            outputs.enc_topk_logits,
            outputs.enc_topk_bboxes,
            outputs.denoising_meta_values,
            batch["labels"],
            batch["ignore_boxes"],
            batch["canvas"],
            {**model.loss_settings, "box_loss": "giou"},
            model.matcher_settings,
            None,
        )
    model.train(was_training)
    reference, ours = float(outputs.loss), float(sum(losses.values()))
    return {
        "reference_loss": reference,
        "our_loss": ours,
        "relative_difference": abs(ours - reference) / max(abs(reference), 1e-12),
        "has_denoising": outputs.denoising_meta_values is not None,
    }
