import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl

from hm2.data import CLASS_NAMES
from hm2.dense_evaluation import evaluate_coco_max_dets

PRETRAINING_SPLITS = ("train", "val")


def clip_video(clip: str) -> str:
    """Return the seven-digit raw VisDrone video number of a VID clip name.

    Args:
        clip: VID clip name such as uav0000013_00000_v.

    Returns:
        The raw video number, such as 0000013.
    """
    return clip[3:10]


def checkpoint_name(recipe: str, seed: int, epoch: int) -> str:
    """Name the checkpoint of one training run after a number of completed epochs.

    Args:
        recipe: Recipe name from the grouped cross-validation plan.
        seed: Training seed.
        epoch: Number of completed epochs.

    Returns:
        A candidate name such as control_seed2026_epoch1.
    """
    return f"{recipe}_seed{seed}_epoch{epoch}"


def interpolation_name(alpha: float) -> str:
    """Name the interpolation that gives the fine-tuned weights the coefficient alpha.

    Args:
        alpha: Coefficient of the fine-tuned checkpoint; the initial one gets 1 - alpha.

    Returns:
        A candidate name such as interpolation_alpha0.5.
    """
    return f"interpolation_alpha{alpha}"


def soup_name(recipe: str, epoch: int) -> str:
    """Name the uniform weight average of every seed of one recipe at one epoch.

    Args:
        recipe: Recipe name from the grouped cross-validation plan.
        epoch: Number of completed epochs of the averaged checkpoints.

    Returns:
        A candidate name such as low_learning_rate_soup_epoch1.
    """
    return f"{recipe}_soup_epoch{epoch}"


def planned_candidates(plan: dict) -> dict[str, list[tuple[str, int]]]:
    """List every checkpoint candidate of one fold with the training runs it depends on.

    Args:
        plan: The grouped cross-validation plan from config/yolo11s_grouped_cv.toml.

    Returns:
        Required (recipe, seed) runs keyed by candidate name, starting with the untrained
        initial checkpoint.
    """
    candidates: dict[str, list[tuple[str, int]]] = {"initial": []}
    for recipe, settings in plan["recipes"].items():
        for seed in settings["seeds"]:
            for epoch in range(1, plan["cross_validation"]["epochs"] + 1):
                candidates[checkpoint_name(recipe, seed, epoch)] = [(recipe, seed)]
    choice = plan["candidates"]
    source = candidates[choice["interpolation_source"]]
    for alpha in choice["interpolation_alphas"]:
        candidates[interpolation_name(alpha)] = source
    recipe = choice["soup_recipe"]
    candidates[soup_name(recipe, choice["soup_epoch"])] = [
        (recipe, seed) for seed in plan["recipes"][recipe]["seeds"]
    ]
    return candidates


def summarize_det_overlap(frames: pl.DataFrame, det_index: dict) -> pl.DataFrame:
    """Locate VisDrone-DET images relative to every labeled VID clip.

    A clip uav0000VVV_SSSSS_v covers raw frames SSSSS to SSSSS + n - 1 of video VVV. A clip
    is frame-clean when no DET train or val image of its video falls inside that range, and
    video-clean when DET train and val contain no image of its video at all. DET test-dev
    images are counted separately because the pretrained checkpoint never trained on them.

    Args:
        frames: Image metadata from hm2.data.load_tables.
        det_index: Frame index written by scripts/build_det_frame_index.py.

    Returns:
        One row per labeled clip with its video, frame and object counts and DET overlap.
    """
    clips = (
        frames.filter(pl.col("is_labeled"))
        .group_by("split", "clip")
        .agg(
            pl.len().alias("frames"),
            pl.col("frame_id").min().alias("first_frame_id"),
            pl.col("frame_id").max().alias("last_frame_id"),
            pl.col("target_count").sum().alias("objects"),
        )
        .sort("clip")
    )
    rows = []
    for row in clips.iter_rows(named=True):
        video = clip_video(row["clip"])
        start = int(row["clip"][11:16])
        first, last = start + row["first_frame_id"] - 1, start + row["last_frame_id"] - 1
        splits = det_index["videos"].get(video, {})
        pretraining = [frame for split in PRETRAINING_SPLITS for frame in splits.get(split, [])]
        gaps = [max(first - frame, frame - last, 0) for frame in pretraining]
        rows.append(
            {
                **{key: row[key] for key in ("clip", "split", "frames", "objects")},
                "video": video,
                "det_images_same_video": len(pretraining),
                "det_images_inside_clip": sum(gap == 0 for gap in gaps),
                "det_nearest_frame_gap": min(gaps) if gaps else None,
                "det_test_dev_images_same_video": len(splits.get("test-dev", [])),
            }
        )
    return pl.from_dicts(rows).with_columns(
        (pl.col("det_images_inside_clip") == 0).alias("is_frame_clean"),
        (pl.col("det_images_same_video") == 0).alias("is_video_clean"),
    )


def select_validation_pool(overlap: pl.DataFrame) -> pl.DataFrame:
    """Keep the clips of raw videos whose every labeled clip is frame-clean.

    A frame-clean clip whose video also has a DET-contaminated clip is excluded, because
    that sibling clip must train and would show the same flight to the model.

    Args:
        overlap: Output of summarize_det_overlap.

    Returns:
        The clips eligible for grouped validation.
    """
    videos = overlap.group_by("video").agg(pl.col("is_frame_clean").all())
    clean = videos.filter(pl.col("is_frame_clean"))["video"].implode()
    return overlap.filter(pl.col("video").is_in(clean))


def assign_video_folds(pool: pl.DataFrame, fold_count: int) -> pl.DataFrame:
    """Assign whole videos to folds with a deterministic greedy balance of object counts.

    Videos are taken in decreasing object count, ties broken by video number, and each goes
    to the fold with the fewest objects so far, ties broken by the lower fold index.

    Args:
        pool: Clips eligible for validation with video and objects columns.
        fold_count: Number of folds.

    Returns:
        The pool with a zero-based fold column.
    """
    videos = (
        pool.group_by("video")
        .agg(pl.col("objects").sum())
        .sort(["objects", "video"], descending=[True, False])
    )
    totals = [0] * fold_count
    assignment = {}
    for video, objects in videos.iter_rows():
        fold = min(range(fold_count), key=lambda index: (totals[index], index))
        assignment[video] = fold
        totals[fold] += objects
    return pool.with_columns(pl.col("video").replace_strict(assignment).alias("fold"))


def write_coco_groups(
    annotation_path: Path, image_groups: dict[str, list[int]], directory: Path
) -> dict[str, Path]:
    """Split one COCO file into one file per group of image IDs, keeping every ID unchanged.

    Args:
        annotation_path: COCO file with the images of every group, including crowd regions.
        image_groups: COCO image IDs keyed by group name.
        directory: Destination of the per-group files.

    Returns:
        Written COCO paths keyed by group name.
    """
    document = json.loads(annotation_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for group, image_ids in image_groups.items():
        selected = set(image_ids)
        path = directory / f"{group}.json"
        path.write_text(
            json.dumps(
                {
                    **document,
                    "images": [image for image in document["images"] if image["id"] in selected],
                    "annotations": [
                        annotation
                        for annotation in document["annotations"]
                        if annotation["image_id"] in selected
                    ],
                }
            )
        )
        paths[group] = path
    return paths


def evaluate_groups(
    group_annotations: dict[str, Path],
    image_groups: dict[str, list[int]],
    records: list[dict],
) -> pl.DataFrame:
    """Evaluate COCO AP at 100 detections separately for every group of images.

    Args:
        group_annotations: Per-group COCO files from write_coco_groups.
        image_groups: COCO image IDs keyed by the same group names.
        records: COCO detections covering at least the images of every group.

    Returns:
        One row per group with mAP, AP50, AP75, size APs and per-class AP columns, where a
        class absent from the group has a null AP.
    """
    by_image: dict[int, list[dict]] = {}
    for record in records:
        by_image.setdefault(record["image_id"], []).append(record)
    rows = []
    for group, image_ids in image_groups.items():
        detections = [record for image_id in image_ids for record in by_image.get(image_id, [])]
        metrics = evaluate_coco_max_dets(
            group_annotations[group], detections, image_ids, 100, backend="faster_coco_eval"
        )
        rows.append(
            {
                "group": group,
                **{
                    key: metrics[key]
                    for key in ("map", "ap50", "ap75", "ap_small", "ap_medium", "ap_large")
                },
                **{f"ap_{name}": metrics["per_class_ap"][name] for name in CLASS_NAMES},
            }
        )
    return pl.from_dicts(rows, infer_schema_length=None)


def sign_flip_p_value(differences: np.ndarray) -> float:
    """Return the exact one-sided sign-flip p-value for a positive mean paired difference.

    Under the null hypothesis each group's difference is symmetric around zero, so all 2^n
    sign assignments are equally likely. The p-value is the share of assignments whose mean
    is at least the observed mean. The enumeration is exact and practical for up to about
    twenty groups.

    Args:
        differences: One paired difference per independent group.

    Returns:
        The one-sided p-value.
    """
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(differences))))
    means = signs @ differences / len(differences)
    return float(np.mean(means >= differences.mean() - 1e-12))


def subset_reversal_rate(
    differences: np.ndarray, subset_fraction: float, draws: int, seed: int
) -> float:
    """Estimate how often a random subset of groups and its complement disagree in sign.

    This mimics a clip-level Public/Private split: each draw assigns a random subset_fraction
    of the groups to one part and the rest to the other, and counts a reversal when the mean
    paired difference has a different sign in the two parts.

    Args:
        differences: One paired difference per independent group.
        subset_fraction: Share of groups in the smaller part, such as 0.3.
        draws: Number of random splits.
        seed: Seed of the random generator.

    Returns:
        The share of draws with a sign reversal.
    """
    count = len(differences)
    size = min(max(round(subset_fraction * count), 1), count - 1)
    generator = np.random.default_rng(seed)
    orders = generator.permuted(np.tile(np.arange(count), (draws, 1)), axis=1)
    subset = differences[orders[:, :size]].mean(axis=1)
    complement = differences[orders[:, size:]].mean(axis=1)
    return float(np.mean(np.sign(subset) != np.sign(complement)))


def assess_candidate(
    differences: np.ndarray,
    class_differences: dict[str, float],
    criteria: dict,
) -> dict:
    """Apply the preregistered rule that decides whether a candidate replaces the control.

    The candidate passes only when the mean paired difference over groups is positive, the
    share of improved groups and the sign-flip p-value meet their thresholds, the mean class
    difference stays positive after removing the class with the largest gain, and no group
    loses more than the tolerated amount. The Public/Private-style reversal rate is reported
    for interpretation but is not part of the rule.

    Args:
        differences: Candidate minus control mAP for every independent group.
        class_differences: Candidate minus control pooled AP keyed by class name.
        criteria: The decision table with min_improved_share, max_p_value,
            max_group_loss, subset_fraction, subset_draws and seed.

    Returns:
        The statistics, the result of every criterion and the overall decision.
    """
    gains = sorted(class_differences.values())
    statistics = {
        "groups": len(differences),
        "mean_difference": float(differences.mean()),
        "improved_share": float(np.mean(differences > 0)),
        "p_value": sign_flip_p_value(differences),
        "worst_group_difference": float(differences.min()),
        "mean_class_difference": float(np.mean(gains)),
        "mean_class_difference_without_best": float(np.mean(gains[:-1])),
        "reversal_rate": subset_reversal_rate(
            differences, criteria["subset_fraction"], criteria["subset_draws"], criteria["seed"]
        ),
    }
    checks = {
        "is_mean_positive": statistics["mean_difference"] > 0,
        "is_improvement_broad": statistics["improved_share"] >= criteria["min_improved_share"],
        "is_significant": statistics["p_value"] <= criteria["max_p_value"],
        "is_not_one_class": statistics["mean_class_difference_without_best"] > 0,
        "is_worst_group_tolerable": statistics["worst_group_difference"]
        >= -criteria["max_group_loss"],
    }
    return {**statistics, **checks, "should_adopt": all(checks.values())}
