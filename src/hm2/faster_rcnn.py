import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch import Tensor
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes
from torchvision.ops import box_area, box_convert
from torchvision.transforms.functional import pil_to_tensor

from hm2.detection_data import DetectionDataset

NUM_CLASSES = 9
DEFAULT_ANCHOR_SIZES = (32, 64, 128, 256, 512)
ANCHOR_ASPECT_RATIOS = (0.5, 1.0, 2.0)


def ignore_coverage(boxes: Tensor, ignore_boxes: Tensor) -> Tensor:
    """Return the largest fraction of each box's area that lies inside one ignored region.

    Args:
        boxes: Candidate anchors or proposals as xyxy pixels with shape (N, 4).
        ignore_boxes: Ignored regions as xyxy pixels with shape (M, 4).

    Returns:
        Coverage values in [0, 1] with shape (N,), zero when there are no regions.
    """
    if boxes.numel() == 0 or ignore_boxes.numel() == 0:
        return boxes.new_zeros(boxes.shape[0])
    ignore_boxes = ignore_boxes.to(boxes)
    top_left = torch.max(boxes[:, None, :2], ignore_boxes[None, :, :2])
    bottom_right = torch.min(boxes[:, None, 2:], ignore_boxes[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=2)
    return (intersection / box_area(boxes).clamp(min=1e-6)[:, None]).max(dim=1).values


def suppress_ignored_negatives(
    labels: Tensor, boxes: Tensor, ignore_boxes: Tensor, threshold: float
) -> Tensor:
    """Mark background samples that mostly lie inside ignored regions as excluded.

    Torchvision samplers draw positives from labels >= 1 and negatives from label 0,
    and never sample label -1. Positive and already excluded samples are unchanged.

    Args:
        labels: Sampler labels for anchors or proposals.
        boxes: Matching anchors or proposals as xyxy pixels.
        ignore_boxes: Ignored regions as xyxy pixels in the same coordinate frame.
        threshold: Minimum covered area fraction that excludes a negative sample.

    Returns:
        A copy of the labels in which covered negatives are set to -1.
    """
    is_ignored = (labels == 0) & (ignore_coverage(boxes, ignore_boxes) >= threshold)
    return torch.where(is_ignored, torch.full_like(labels, -1), labels)


class IgnoreAwareRegionProposalNetwork(RegionProposalNetwork):
    """Region proposal network that never samples anchors inside ignored regions as background.

    Args:
        *args: Positional RegionProposalNetwork arguments.
        ignore_threshold: Covered area fraction that excludes a negative anchor, or None.
        **kwargs: Keyword RegionProposalNetwork arguments.
    """

    def __init__(self, *args, ignore_threshold: float | None = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ignore_threshold = ignore_threshold

    def assign_targets_to_anchors(
        self, anchors: list[Tensor], targets: list[dict[str, Tensor]]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Assign anchors with the stock matcher, then exclude covered negatives.

        Args:
            anchors: Anchors for each image in resized-image pixels.
            targets: Resized targets, optionally containing ignore_boxes.

        Returns:
            Anchor labels and matched ground-truth boxes for each image.
        """
        labels, matched_boxes = super().assign_targets_to_anchors(anchors, targets)
        if self.ignore_threshold is None:
            return labels, matched_boxes
        labels = [
            (
                suppress_ignored_negatives(
                    image_labels, image_anchors, target["ignore_boxes"], self.ignore_threshold
                )
                if "ignore_boxes" in target
                else image_labels
            )
            for image_labels, image_anchors, target in zip(labels, anchors, targets, strict=True)
        ]
        return labels, matched_boxes


class IgnoreAwareRoIHeads(RoIHeads):
    """Second-stage heads that never sample proposals inside ignored regions as background.

    Args:
        *args: Positional RoIHeads arguments.
        ignore_threshold: Covered area fraction that excludes a negative proposal, or None.
        **kwargs: Keyword RoIHeads arguments.
    """

    def __init__(self, *args, ignore_threshold: float | None = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ignore_threshold = ignore_threshold
        self._ignore_boxes: list[Tensor] | None = None

    def select_training_samples(
        self, proposals: list[Tensor], targets: list[dict[str, Tensor]] | None
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        """Run stock proposal sampling while the current ignored regions are available.

        Args:
            proposals: RPN proposals for each image in resized-image pixels.
            targets: Resized targets, optionally containing ignore_boxes.

        Returns:
            Sampled proposals, matched indices, labels and regression targets.
        """
        if (
            self.ignore_threshold is not None
            and targets is not None
            and all("ignore_boxes" in target for target in targets)
        ):
            self._ignore_boxes = [target["ignore_boxes"] for target in targets]
        try:
            return super().select_training_samples(proposals, targets)
        finally:
            self._ignore_boxes = None

    def assign_targets_to_proposals(
        self, proposals: list[Tensor], gt_boxes: list[Tensor], gt_labels: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Assign proposals with the stock matcher, then exclude covered negatives.

        Args:
            proposals: Proposals for each image, including appended ground-truth boxes.
            gt_boxes: Ground-truth boxes for each image.
            gt_labels: Ground-truth labels 1 to 8 for each image.

        Returns:
            Matched ground-truth indices and sampler labels for each image.
        """
        matched_indices, labels = super().assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels
        )
        if self._ignore_boxes is None or self.ignore_threshold is None:
            return matched_indices, labels
        labels = [
            suppress_ignored_negatives(
                image_labels, image_proposals, regions, self.ignore_threshold
            )
            for image_labels, image_proposals, regions in zip(
                labels, proposals, self._ignore_boxes, strict=True
            )
        ]
        return matched_indices, labels


class IgnoreAwareTransform(GeneralizedRCNNTransform):
    """Torchvision detection transform that also rescales ignored regions with the image."""

    def resize(
        self, image: Tensor, target: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        """Resize an image and its targets, including optional ignore_boxes.

        Args:
            image: CHW float image in original pixels.
            target: Optional target with xyxy boxes and ignore_boxes in original pixels.

        Returns:
            The resized image and target in resized-image pixels.
        """
        original_size = list(image.shape[-2:])
        image, target = super().resize(image, target)
        if target is not None and "ignore_boxes" in target:
            target["ignore_boxes"] = resize_boxes(
                target["ignore_boxes"], original_size, list(image.shape[-2:])
            )
        return image, target


def build_faster_rcnn(
    pretrained: bool = True,
    allow_download: bool = False,
    min_size: int = 800,
    max_size: int = 1333,
    anchor_sizes: tuple[int, ...] | list[int] = DEFAULT_ANCHOR_SIZES,
    rpn_pre_nms_top_n_train: int = 2000,
    rpn_pre_nms_top_n_test: int = 1000,
    rpn_post_nms_top_n_train: int = 2000,
    rpn_post_nms_top_n_test: int = 1000,
    box_batch_size_per_image: int = 512,
    box_positive_fraction: float = 0.25,
    detections_per_img: int = 500,
    is_ignore_aware: bool = True,
    ignore_threshold: float = 0.5,
    trainable_backbone_layers: int = 3,
    seed: int = 2026,
) -> FasterRCNN:
    """Build a COCO-pretrained Faster R-CNN ResNet-50 FPN V2 for the eight target classes.

    Load the official torchvision COCO weights, then replace the RPN and RoI heads with
    ignore-aware subclasses that reuse the pretrained RPN head, RoI pooling and box head.
    Only the final box predictor is new: nine outputs, background plus eight classes,
    initialized with the given seed. Random initialization exists only for unit tests.

    Args:
        pretrained: Whether to load the official COCO detector weights.
        allow_download: Whether a missing official weight file may be downloaded.
        min_size: Shorter-side resize target in model pixels.
        max_size: Maximum longer side in model pixels.
        anchor_sizes: One anchor size per FPN level, P2 to P6.
        rpn_pre_nms_top_n_train: Proposals per level kept before NMS in training.
        rpn_pre_nms_top_n_test: Proposals per level kept before NMS in inference.
        rpn_post_nms_top_n_train: Proposals per image kept after NMS in training.
        rpn_post_nms_top_n_test: Proposals per image kept after NMS in inference.
        box_batch_size_per_image: Sampled RoIs per image for the second-stage loss.
        box_positive_fraction: Target fraction of positive sampled RoIs.
        detections_per_img: Maximum detections returned per image.
        is_ignore_aware: Whether negatives inside ignored regions are excluded from sampling.
        ignore_threshold: Covered area fraction that excludes a negative anchor or proposal.
        trainable_backbone_layers: Number of trainable ResNet stages for pretrained weights.
        seed: Seed for the new box predictor initialization.

    Returns:
        A nine-class torchvision Faster R-CNN with ignore-aware sampling.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    if weights is not None:
        cached = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
        if not cached.is_file() and not allow_download:
            raise FileNotFoundError(
                "COCO Faster R-CNN V2 weights are not cached. Set allow_download=True to "
                "download the official torchvision file."
            )
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=trainable_backbone_layers if pretrained else None,
    )
    threshold = ignore_threshold if is_ignore_aware else None
    anchor_generator = AnchorGenerator(
        tuple((size,) for size in anchor_sizes), (ANCHOR_ASPECT_RATIOS,) * len(anchor_sizes)
    )
    model.rpn = IgnoreAwareRegionProposalNetwork(
        anchor_generator,
        model.rpn.head,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n={"training": rpn_pre_nms_top_n_train, "testing": rpn_pre_nms_top_n_test},
        post_nms_top_n={
            "training": rpn_post_nms_top_n_train,
            "testing": rpn_post_nms_top_n_test,
        },
        nms_thresh=0.7,
        ignore_threshold=threshold,
    )
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES)
    model.roi_heads = IgnoreAwareRoIHeads(
        model.roi_heads.box_roi_pool,
        model.roi_heads.box_head,
        predictor,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=box_batch_size_per_image,
        positive_fraction=box_positive_fraction,
        bbox_reg_weights=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=detections_per_img,
        ignore_threshold=threshold,
    )
    model.transform = IgnoreAwareTransform(
        min_size, max_size, model.transform.image_mean, model.transform.image_std
    )
    return model


class IgnoreAwareDetectionDataset(DetectionDataset):
    """DetectionDataset frames that also return ignored regions in original-image pixels.

    Targets are built by DetectionDataset: xyxy float boxes and labels 1 to 8, where 0 is
    background. Ignored regions are returned as ignore_boxes and flipped together with
    the image and the targets, using the same flip draw as DetectionDataset.

    Args:
        root: Dataset directory containing the frames' relative paths.
        frames: Selected frame metadata with image_key, image_id, relative_path and sizes.
        objects: Target objects with image_key, pixel x, y, width, height and category_id.
        ignored: Ignored regions with image_key and pixel x, y, width, height.
        flip_probability: Training-only horizontal flip probability.
        augmentation_seed: Optional epoch seed for flips independent of worker scheduling.
    """

    def __init__(
        self,
        root: Path,
        frames: pl.DataFrame,
        objects: pl.DataFrame,
        ignored: pl.DataFrame,
        flip_probability: float = 0.0,
        augmentation_seed: int | None = None,
    ) -> None:
        super().__init__(root, frames, objects, flip_probability, augmentation_seed)
        grouped = (
            ignored.join(frames.select("image_key"), on="image_key", how="semi")
            .group_by("image_key")
            .agg(pl.concat_list("x", "y", "width", "height").alias("boxes"))
        )
        self.ignore_boxes = {
            row["image_key"]: box_convert(
                torch.tensor(row["boxes"], dtype=torch.float32), "xywh", "xyxy"
            )
            for row in grouped.iter_rows(named=True)
        }

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        record = self.records[index]
        key = record["image_key"]
        with Image.open(self.root / record["relative_path"]) as source:
            image = pil_to_tensor(source.convert("RGB")).to(torch.float32).div_(255)
        boxes, labels = self.targets.get(
            key, (torch.empty((0, 4)), torch.empty(0, dtype=torch.int64))
        )
        boxes = boxes.clone()
        ignore_boxes = self.ignore_boxes.get(key, torch.empty((0, 4))).clone()
        flip_draw = (
            random.Random(f"{self.augmentation_seed}:{key}").random()
            if self.augmentation_seed is not None
            else torch.rand(()).item()
        )
        if self.flip_probability and flip_draw < self.flip_probability:
            image = image.flip(-1)
            width = image.shape[-1]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            ignore_boxes[:, [0, 2]] = width - ignore_boxes[:, [2, 0]]
        return image, {
            "boxes": boxes,
            "labels": labels,
            "ignore_boxes": ignore_boxes,
            "image_id": torch.tensor(int(record["image_id"])),
            "original_size": torch.tensor([record["image_height"], record["image_width"]]),
        }


def summarize_ignore_assignment(
    model: FasterRCNN, image: Tensor, target: dict[str, Tensor]
) -> dict[str, int]:
    """Count anchors and proposals that ignore-aware sampling removes from the background.

    Run the model's transform, backbone and RPN on one image, then compare stock torchvision
    label assignment with the ignore-aware assignment for anchors and proposals.

    Args:
        model: Faster R-CNN built by build_faster_rcnn with ignore-aware sampling.
        image: CHW float image in original pixels on the model's device.
        target: Target with boxes, labels and ignore_boxes in original pixels.

    Returns:
        Totals, stock negatives, newly excluded negatives and positives for both stages.
    """
    transform, rpn, heads = model.transform, model.rpn, model.roi_heads
    if not (
        isinstance(transform, IgnoreAwareTransform)
        and isinstance(rpn, IgnoreAwareRegionProposalNetwork)
        and isinstance(heads, IgnoreAwareRoIHeads)
    ):
        raise TypeError("Build the model with build_faster_rcnn to inspect ignore handling.")
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        images, targets = transform([image], [{key: value for key, value in target.items()}])
        features = model.backbone(images.tensors)
        anchors = rpn.anchor_generator(images, list(features.values()))
        stock_anchor_labels, _ = RegionProposalNetwork.assign_targets_to_anchors(
            rpn, anchors, targets
        )
        aware_anchor_labels, _ = rpn.assign_targets_to_anchors(anchors, targets)
        proposals, _ = rpn(images, features)
        gt_boxes = [targets[0]["boxes"]]
        gt_labels = [targets[0]["labels"]]
        proposals = heads.add_gt_proposals(proposals, gt_boxes)
        _, stock_proposal_labels = RoIHeads.assign_targets_to_proposals(
            heads, proposals, gt_boxes, gt_labels
        )
        heads._ignore_boxes = [targets[0]["ignore_boxes"]]
        try:
            _, aware_proposal_labels = heads.assign_targets_to_proposals(
                proposals, gt_boxes, gt_labels
            )
        finally:
            heads._ignore_boxes = None
    model.train(was_training)
    stock_anchors, aware_anchors = stock_anchor_labels[0], aware_anchor_labels[0]
    stock_proposals, aware_proposals = stock_proposal_labels[0], aware_proposal_labels[0]
    return {
        "ignored_regions": int(targets[0]["ignore_boxes"].shape[0]),
        "anchors": int(stock_anchors.numel()),
        "anchor_negatives_stock": int((stock_anchors == 0).sum()),
        "anchor_negatives_excluded": int(((stock_anchors == 0) & (aware_anchors == -1)).sum()),
        "anchor_positives_stock": int((stock_anchors == 1).sum()),
        "anchor_positives_aware": int((aware_anchors == 1).sum()),
        "proposals": int(stock_proposals.numel()),
        "proposal_negatives_stock": int((stock_proposals == 0).sum()),
        "proposal_negatives_excluded": int(
            ((stock_proposals == 0) & (aware_proposals == -1)).sum()
        ),
        "proposal_positives_stock": int((stock_proposals >= 1).sum()),
        "proposal_positives_aware": int((aware_proposals >= 1).sum()),
    }


def compare_loss_windows(history: list[dict], window: int) -> dict[str, float]:
    """Compare mean total loss over the first and last training windows.

    Args:
        history: Step records with a total loss, as returned by train_steps.
        window: Number of steps per window, such as one pass over a fixed subset.

    Returns:
        First-window mean, last-window mean and their ratio.
    """
    losses = [row["loss"] for row in history]
    if len(losses) < 2 * window:
        raise ValueError("The history needs at least two complete windows.")
    first = float(np.mean(losses[:window]))
    last = float(np.mean(losses[-window:]))
    return {"first_window_loss": first, "last_window_loss": last, "ratio": last / first}
