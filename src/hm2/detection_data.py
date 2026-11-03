import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert
from torchvision.transforms.functional import pil_to_tensor


def select_frames(frames: pl.DataFrame, split: str, stride: int = 1) -> pl.DataFrame:
    """Select every Nth available frame independently within each official clip.

    Args:
        frames: Prepared image metadata with split, clip, and frame identifiers.
        split: Official labeled split to select.
        stride: Positive sampling stride within each clip.

    Returns:
        Deterministically ordered selected metadata rows.
    """
    if split not in {"train", "val"} or stride < 1:
        raise ValueError("Use a labeled split and a positive frame stride.")
    return (
        frames.filter(pl.col("split") == split)
        .sort("clip", "frame_id")
        .with_columns((pl.col("frame_id").rank("ordinal").over("clip") - 1).alias("clip_index"))
        .filter(pl.col("clip_index") % stride == 0)
        .drop("clip_index")
    )


class DetectionDataset(Dataset):
    """On-demand RGB frames with precomputed foreground targets in source pixels.

    Ordinary GT is preserved, including GT overlapping ignored regions. Ignore-area
    negative-anchor suppression is not implemented by this dataset or stock RetinaNet.

    Args:
        root: Original dataset directory.
        frames: Selected labeled frame metadata.
        objects: Prepared target-object table, excluding crowd replicas.
        flip_probability: Training-only horizontal flip probability.
    """

    def __init__(
        self, root: Path, frames: pl.DataFrame, objects: pl.DataFrame, flip_probability: float = 0.0
    ) -> None:
        self.root = root
        self.frames = frames
        self.records = frames.to_dicts()
        self.flip_probability = flip_probability
        grouped = (
            objects.join(frames.select("image_key"), on="image_key", how="semi")
            .group_by("image_key")
            .agg(
                pl.concat_list("x", "y", "width", "height").alias("boxes"),
                pl.col("category_id").alias("labels"),
            )
        )
        self.targets = {
            row["image_key"]: (
                box_convert(torch.tensor(row["boxes"], dtype=torch.float32), "xywh", "xyxy"),
                torch.tensor(row["labels"], dtype=torch.int64),
            )
            for row in grouped.iter_rows(named=True)
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        with Image.open(self.root / record["relative_path"]) as source:
            image = pil_to_tensor(source.convert("RGB")).to(torch.float32).div_(255)
        boxes, labels = self.targets.get(
            record["image_key"], (torch.empty((0, 4)), torch.empty(0, dtype=torch.int64))
        )
        boxes = boxes.clone()
        if self.flip_probability and torch.rand(()) < self.flip_probability:
            image = image.flip(-1)
            boxes[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
        return image, {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(int(record["image_id"])),
            "original_size": torch.tensor([record["image_height"], record["image_width"]]),
        }


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Keep variable-size images and targets as separate lists.

    Args:
        batch: Decoded image and target pairs.

    Returns:
        Parallel lists of images and target dictionaries.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(
    dataset: DetectionDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    should_shuffle: bool = False,
    seed: int = 2026,
) -> DataLoader:
    """Build a seeded loader with bounded prefetch and CUDA-aware pinned memory.

    Args:
        dataset: Lazy image dataset with cached annotation lookups.
        batch_size: Images per batch.
        num_workers: CPU decoding workers, with zero supported for debugging.
        device: Target compute device.
        should_shuffle: Whether to shuffle training frames.
        seed: Reproducible shuffle and worker seed.

    Returns:
        A configured variable-image-size DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers else None,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
