import re
from pathlib import Path

VARIANTS = ("standard_anchors", "smaller_anchors", "diou_regression")
EXPERIMENTS = {
    "standard_anchors_stride5": ("standard_anchors", 5),
    "smaller_anchors_stride5": ("smaller_anchors", 5),
    "standard_anchors_full": ("standard_anchors", 1),
    "smaller_anchors_full": ("smaller_anchors", 1),
}
STORAGE_IDS = {
    "standard_anchors": "R0",
    "smaller_anchors": "R1",
    "diou_regression": "R2",
    "standard_anchors_stride5": "R0-S",
    "smaller_anchors_stride5": "R1-S",
    "standard_anchors_full": "R0-F",
    "smaller_anchors_full": "R1-F",
}
DESCRIPTIVE_NAMES = {value: key for key, value in STORAGE_IDS.items()}
LABELS = {
    "standard_anchors": "RetinaNet with standard anchors",
    "smaller_anchors": "RetinaNet with smaller anchors",
    "diou_regression": "RetinaNet with DIoU regression",
    **{
        name: f"RetinaNet with {'standard' if variant == 'standard_anchors' else 'smaller'} "
        f"anchors ({'every fifth frame' if stride == 5 else 'full training set'})"
        for name, (variant, stride) in EXPERIMENTS.items()
    },
}


def descriptive_name(name: str) -> str:
    """Resolve a historical checkpoint ID to its descriptive experiment or variant name.

    Args:
        name: Current name, legacy storage ID, or another model's existing identifier.

    Returns:
        Descriptive name, preserving identifiers outside this mapping.
    """
    return DESCRIPTIVE_NAMES.get(name, name)


def storage_id(name: str) -> str:
    """Resolve a descriptive name to the existing checkpoint and directory identity.

    Args:
        name: Public experiment name or already persisted identifier.

    Returns:
        Existing storage identity without migrating checkpoints or configurations.
    """
    return STORAGE_IDS.get(name, name)


def experiment_label(name: str) -> str:
    """Return the readable name used in notebook tables, legends and captions.

    Args:
        name: Public name or historical identifier.

    Returns:
        Human-readable model description, retaining unknown model names.
    """
    return LABELS.get(descriptive_name(name), name)


def result_directory(root: Path, name: str) -> Path:
    """Read existing legacy results while using descriptive paths for new standalone outputs.

    Args:
        root: Parent directory containing experiment outputs.
        name: Public experiment or variant name.

    Returns:
        Existing descriptive or legacy directory, or the new descriptive pathname.
    """
    canonical = root / descriptive_name(name)
    legacy = root / storage_id(name)
    return canonical if canonical.exists() or not legacy.exists() else legacy


def describe_text(text: str) -> str:
    """Translate historical experiment tokens in saved notebook output and captions.

    Args:
        text: Human-facing text loaded from older experiment outputs.

    Returns:
        Text with historical tokens replaced by readable model descriptions.
    """
    pattern = (
        r"(?<![A-Za-z0-9_])(?:"
        + "|".join(re.escape(name) for name in sorted(DESCRIPTIVE_NAMES, key=len, reverse=True))
        + r")(?![A-Za-z0-9_])"
    )
    return re.sub(pattern, lambda match: experiment_label(match.group()), text)
