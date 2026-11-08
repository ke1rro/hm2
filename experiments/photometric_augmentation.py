import hashlib
import random
from pathlib import Path

import polars as pl
import torch
from torchvision.transforms import functional

from hm2.faster_rcnn import IgnoreAwareDetectionDataset


def apply_photometric(image: torch.Tensor, seed: int, settings: dict) -> torch.Tensor:
    """Apply seeded mild color changes to an unnormalized float RGB image.

    Args:
        image: CHW float RGB values in [0, 1], before detector normalization.
        seed: Local random seed, independent of worker scheduling and global RNG state.
        settings: Probability and brightness, contrast and saturation factor intervals.

    Returns:
        An appearance-only transformed image with unchanged dimensions.
    """
    probability = settings["probability"]
    if not 0 <= probability <= 1:
        raise ValueError("Photometric probability must lie in [0, 1].")
    generator = random.Random(seed)
    if generator.random() >= probability:
        return image
    adjustments = [
        functional.adjust_brightness,
        functional.adjust_contrast,
        functional.adjust_saturation,
    ]
    factors = [
        generator.uniform(*settings[name]) for name in ("brightness", "contrast", "saturation")
    ]
    order = list(range(3))
    generator.shuffle(order)
    for index in order:
        image = adjustments[index](image, factors[index])
    return image


class PhotometricDetectionDataset(IgnoreAwareDetectionDataset):
    """Ignore-aware training frames with reproducible appearance-only augmentation.

    Args:
        root: Original image root.
        frames: Selected training frames in consumed order.
        objects: Valid GT objects.
        ignored: Ignored-region geometry.
        flip_probability: Existing horizontal flip probability.
        augmentation_seed: Epoch seed used by both augmentation policies.
        photometric: Probability and factor intervals for the photometric experiment.
    """

    def __init__(
        self,
        root: Path,
        frames: pl.DataFrame,
        objects: pl.DataFrame,
        ignored: pl.DataFrame,
        flip_probability: float = 0.0,
        augmentation_seed: int | None = None,
        photometric: dict | None = None,
    ) -> None:
        super().__init__(root, frames, objects, ignored, flip_probability, augmentation_seed)
        self.photometric = photometric
        if photometric and photometric["probability"] and augmentation_seed is None:
            raise ValueError("Enabled photometric augmentation needs an explicit epoch seed.")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, target = super().__getitem__(index)
        if self.photometric:
            identity = f"photometric:{self.augmentation_seed}:{self.records[index]['image_key']}"
            seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
            image = apply_photometric(image, seed, self.photometric)
        return image, target
