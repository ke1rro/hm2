import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import ultralytics
from torchvision.models.detection.transform import resize_boxes
from torchvision.ops import box_convert, clip_boxes_to_image
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.tal import make_anchors

from hm2.data import CLASS_NAMES
from hm2.faster_rcnn import IgnoreAwareTransform

ULTRALYTICS_VERSION = "8.3.235"


def classification_keep_mask(
    foreground: torch.Tensor,
    points: torch.Tensor,
    ignored: list[torch.Tensor],
    is_enabled: bool = True,
) -> torch.Tensor:
    """Exclude negative anchor-free locations inside ignored regions, preserving positives.

    Args:
        foreground: Native task-aligned positive assignment mask with shape B by A.
        points: Candidate center locations in resized-image pixels, shared across the batch.
        ignored: Per-image ignored XYXY regions in the same resized coordinate frame.
        is_enabled: Whether to apply the don't-care mask.

    Returns:
        Classification-valid locations, with every native positive retained.
    """
    keep = torch.ones_like(foreground, dtype=torch.bool)
    if is_enabled:
        for index, regions in enumerate(ignored):
            if regions.numel():
                inside = (
                    (
                        (points[:, None] >= regions[None, :, :2])
                        & (points[:, None] <= regions[None, :, 2:])
                    )
                    .all(-1)
                    .any(-1)
                )
                keep[index] = foreground[index] | ~inside
    return keep


class IgnoreAwareYoloLoss(v8DetectionLoss):
    """Pinned native YOLO11 CIoU/BCE/DFL with classification-only ignored-location masking.

    Args:
        model: Eight-class Ultralytics DetectionModel with explicit loss gains.
        is_ignore_aware: Whether to exclude negative classification locations in ignored regions.
    """

    def __init__(self, model: DetectionModel, is_ignore_aware: bool = True) -> None:
        super().__init__(model)
        self.is_ignore_aware = is_ignore_aware

    def __call__(
        self, preds: list[torch.Tensor] | tuple, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retain native assignment and box losses while masking only don't-care negatives.

        The calculation follows Ultralytics 8.3.235 v8DetectionLoss. Only the BCE
        reduction receives a spatial mask; positive quality normalization and CIoU/DFL
        use the original foreground mask and targets.

        Args:
            preds: Native multi-scale training head outputs.
            batch: Native normalized targets plus optional original resized ignore_boxes.

        Returns:
            Differentiable box, classification and DFL components scaled by batch size,
            and the detached unscaled components, matching the native criterion contract.
        """
        loss = torch.zeros(3, device=self.device)
        features = preds[1] if isinstance(preds, tuple) else preds
        distribution, scores = torch.cat(
            [feature.view(features[0].shape[0], self.no, -1) for feature in features], 2
        ).split((self.reg_max * 4, self.nc), 1)
        scores = scores.permute(0, 2, 1).contiguous()
        distribution = distribution.permute(0, 2, 1).contiguous()
        batch_size = scores.shape[0]
        image_size = (
            torch.tensor(features[0].shape[2:], device=self.device, dtype=scores.dtype)
            * self.stride[0]
        )
        points, strides = make_anchors(features, self.stride, 0.5)
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets, batch_size, scale_tensor=image_size[[1, 0, 1, 0]])
        labels, boxes = targets.split((1, 4), 2)
        valid_gt = boxes.sum(2, keepdim=True).gt_(0.0)
        predicted_boxes = self.bbox_decode(points, distribution)
        _, assigned_boxes, assigned_scores, foreground, _ = self.assigner(
            scores.detach().sigmoid(),
            (predicted_boxes.detach() * strides).type(boxes.dtype),
            points * strides,
            labels,
            boxes,
            valid_gt,
        )
        normalizer = max(assigned_scores.sum(), 1)
        ignored = batch.get("ignore_boxes", [points.new_empty((0, 4))] * batch_size)
        keep = classification_keep_mask(foreground, points * strides, ignored, self.is_ignore_aware)
        loss[1] = (
            self.bce(scores, assigned_scores.to(scores.dtype)) * keep.unsqueeze(-1)
        ).sum() / normalizer
        if foreground.sum():
            loss[0], loss[2] = self.bbox_loss(
                distribution,
                predicted_boxes,
                points,
                assigned_boxes / strides,
                assigned_scores,
                normalizer,
                foreground,
            )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss * batch_size, loss.detach()


class YoloDetector(torch.nn.Module):
    """YOLO11s adapted to the project's image/target, loss and checkpoint interfaces.

    Args:
        model_id: Official YOLO11s pretrained filename or a local checkpoint of that name.
        pretrained: Whether to transfer original COCO weights.
        allow_download: Explicit permission for a missing official weight download.
        min_size: Shorter-side resize target, matching the reference image geometry.
        max_size: Longer-side resize cap.
        num_classes: Eight foreground categories, zero-based inside the native head.
        score_thresh: Fixed inference confidence floor.
        nms_thresh: Fixed class-aware NMS IoU threshold.
        detections_per_img: Maximum returned detections.
        is_ignore_aware: Whether to suppress negative classification locations in ignored regions.
        seed: Shared initialization seed.
        loss_gains: Explicit native box, classification and DFL weights.
    """

    def __init__(
        self,
        model_id: str = "yolo11s.pt",
        pretrained: bool = True,
        allow_download: bool = False,
        min_size: int = 640,
        max_size: int = 1024,
        num_classes: int = 8,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 500,
        is_ignore_aware: bool = True,
        seed: int = 2026,
        loss_gains: dict | None = None,
    ) -> None:
        super().__init__()
        if ultralytics.__version__ != ULTRALYTICS_VERSION:
            raise RuntimeError(
                f"Use ultralytics=={ULTRALYTICS_VERSION} for this tested loss contract."
            )
        if Path(model_id).name != "yolo11s.pt" or num_classes != len(CLASS_NAMES):
            raise ValueError("This baseline is YOLO11s with exactly eight foreground classes.")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.network = DetectionModel("yolo11s.yaml", nc=num_classes, verbose=False)
        if pretrained:
            if not Path(model_id).is_file() and not allow_download:
                raise FileNotFoundError(
                    "YOLO11s weights are missing; authorize their download first."
                )
            self.network.load(YOLO(model_id).model, verbose=False)
        gains = loss_gains or {"box": 7.5, "cls": 0.5, "dfl": 1.5}
        vars(self.network)["args"] = SimpleNamespace(**gains)
        self.network.names = dict(enumerate(CLASS_NAMES))
        self.transform = IgnoreAwareTransform(min_size, max_size, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.is_ignore_aware = is_ignore_aware
        self._criterion: IgnoreAwareYoloLoss | None = None
        self.forward = self._forward

    def _forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        original_sizes = [(image.shape[-2], image.shape[-1]) for image in images]
        transformed, targets = self.transform(images, targets)
        for index, (height, width) in enumerate(transformed.image_sizes):
            transformed.tensors[index, :, height:, :] = 114 / 255
            transformed.tensors[index, :, :height, width:] = 114 / 255
        predictions = self.network(transformed.tensors)
        if self.training:
            if targets is None:
                raise ValueError("Training requires foreground targets and ignored regions.")
            if self._criterion is None or self._criterion.device != transformed.tensors.device:
                self._criterion = IgnoreAwareYoloLoss(self.network, self.is_ignore_aware)
            height, width = transformed.tensors.shape[-2:]
            boxes = torch.cat([target["boxes"] for target in targets])
            labels = torch.cat([target["labels"] for target in targets])
            batch = {
                "batch_idx": torch.cat(
                    [
                        boxes.new_full((len(target["boxes"]),), index)
                        for index, target in enumerate(targets)
                    ]
                ),
                "cls": (labels - 1).to(boxes.dtype).view(-1, 1),
                "bboxes": box_convert(boxes, "xyxy", "cxcywh")
                / boxes.new_tensor([width, height, width, height]),
                "ignore_boxes": [target.get("ignore_boxes", boxes[:0]) for target in targets],
            }
            components = self._criterion(predictions, batch)[0]
            return dict(
                zip(("loss_box", "loss_classification", "loss_dfl"), components, strict=True)
            )
        detections = non_max_suppression(
            predictions,
            conf_thres=self.score_thresh,
            iou_thres=self.nms_thresh,
            max_det=self.detections_per_img,
            nc=self.num_classes,
        )
        return self.decode_predictions(detections, transformed.image_sizes, original_sizes)

    def decode_predictions(
        self,
        detections: list[torch.Tensor],
        resized_sizes: list[tuple[int, int]],
        original_sizes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        """Clip padding predictions, map zero-based labels and restore original image pixels.

        Args:
            detections: Native post-NMS XYXY, confidence and zero-based class rows.
            resized_sizes: Actual aspect-ratio resized image sizes before padding.
            original_sizes: Source image sizes before resizing.

        Returns:
            Source-pixel boxes with foreground labels 1–8 for the shared COCO evaluator.
        """
        results = []
        for prediction, size, original in zip(
            detections, resized_sizes, original_sizes, strict=True
        ):
            boxes = clip_boxes_to_image(prediction[:, :4], size)
            valid = (boxes[:, 2:] > boxes[:, :2]).all(1)
            results.append(
                {
                    "boxes": resize_boxes(boxes[valid], list(size), list(original)),
                    "scores": prediction[valid, 4],
                    "labels": prediction[valid, 5].long() + 1,
                }
            )
        return results


def build_yolo(settings: dict) -> YoloDetector:
    """Construct the explicit YOLO baseline for the existing supervised training loop.

    Args:
        settings: Visible model, training and native loss configuration tables.

    Returns:
        Fresh pretrained YOLO11s or the same architecture when restoring its own state.
    """
    arguments = {key: value for key, value in settings["model"].items() if key != "backend"}
    return YoloDetector(
        **arguments,
        allow_download=True,
        seed=settings["training"]["seed"],
        loss_gains=settings["loss"],
    )
