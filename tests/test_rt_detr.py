import math

import polars as pl
import pytest
import torch
from torch import nn

from hm2.rt_detr import (
    CLASS_PRIOR_PROBABILITY,
    LOSS_PARTS,
    LOSS_TERMS,
    GradientClippedAdamW,
    QueryDetector,
    boxes_to_normalized_cxcywh,
    build_detector_overrides,
    compute_detection_losses,
    compute_layer_losses,
    compute_resize_size,
    decode_predictions,
    detr_classes_to_labels,
    labels_to_detr_classes,
    make_rt_detr_optimizer,
    match_queries,
    padded_canvas_size,
    paired_giou,
    paired_iou,
    paired_nwd,
    reset_classification_heads,
    split_parameter_groups,
    unmatched_ignore_keep,
)
from hm2.submission import build_submission

MODEL = {
    "min_size": 640,
    "max_size": 1024,
    "size_divisor": 32,
    "detections_per_img": 500,
    "is_ignore_aware": True,
    "ignore_threshold": 0.5,
    "num_queries": 300,
    "num_denoising": 100,
    "label_noise_ratio": 0.5,
    "box_noise_scale": 1.0,
}
LOSS = {
    "class_weight": 1.0,
    "l1_weight": 5.0,
    "box_weight": 2.0,
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
    "box_loss": "giou",
    "nwd_constant": 12.0,
}
MATCHER = {
    "class_cost": 2.0,
    "l1_cost": 5.0,
    "giou_cost": 2.0,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
}


def test_resize_keeps_aspect_ratio_under_the_b1_rule() -> None:
    """Catch a resize that differs from the 640/1024 rule of the CNN runs."""
    assert compute_resize_size(765, 1360, 640, 1024) == (576, 1024)
    assert compute_resize_size(1080, 1920, 640, 1024) == (576, 1024)
    assert compute_resize_size(1050, 1400, 640, 1024) == (640, 853)


def test_canvas_holds_every_image_and_is_a_stride_multiple() -> None:
    """Catch a canvas that crops an image or breaks the 32-pixel backbone stride."""
    assert padded_canvas_size([(576, 1024), (640, 853)], 32) == (640, 1024)
    assert padded_canvas_size([(575, 1023)], 32) == (576, 1024)


def test_labels_map_to_detr_and_submission_class_ids() -> None:
    """Catch an off-by-one between torchvision labels, DETR classes and submission ids."""
    labels = torch.tensor([1, 3, 8])
    assert labels_to_detr_classes(labels).tolist() == [0, 2, 7]
    assert detr_classes_to_labels(labels_to_detr_classes(labels)).tolist() == [1, 3, 8]


def test_targets_are_normalized_cxcywh_on_the_padded_canvas() -> None:
    """Catch normalizing by the original image or the unpadded size instead of the canvas."""
    boxes = torch.tensor([[136.0, 76.5, 272.0, 153.0]])
    normalized = boxes_to_normalized_cxcywh(boxes, 1024 / 1360, 576 / 765, (640, 1024))
    assert torch.allclose(normalized, torch.tensor([[0.15, 0.135, 0.1, 0.09]]), atol=1e-6)


def test_batch_preparation_resizes_pads_and_converts_targets() -> None:
    """Catch images or targets that do not line up on the shared canvas."""
    wrapper = QueryDetector(nn.Identity(), MODEL, LOSS, MATCHER)
    images = [torch.ones(3, 765, 1360), torch.ones(3, 1050, 1400)]
    targets = [
        {
            "boxes": torch.tensor([[136.0, 76.5, 272.0, 153.0]]),
            "labels": torch.tensor([8]),
            "ignore_boxes": torch.tensor([[0.0, 0.0, 680.0, 382.5]]),
        },
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)},
    ]
    batch = wrapper.prepare_batch(images, targets)
    assert batch["canvas"] == (640, 1024)
    assert batch["resized_sizes"] == [(576, 1024), (640, 853)]
    assert batch["pixel_values"].shape == (2, 3, 640, 1024)
    assert batch["pixel_values"][0, :, 576:].abs().sum() == 0
    assert batch["pixel_values"][1, :, :, 853:].abs().sum() == 0
    assert batch["pixel_values"][0, :, :576, :1024].min() == 1
    assert batch["labels"][0]["class_labels"].tolist() == [7]
    assert torch.allclose(
        batch["labels"][0]["boxes"], torch.tensor([[0.15, 0.135, 0.1, 0.09]]), atol=1e-6
    )
    assert torch.allclose(batch["ignore_boxes"][0], torch.tensor([[0.0, 0.0, 512.0, 288.0]]))
    assert batch["ignore_boxes"][1].shape == (0, 4)


def test_decoding_returns_original_pixels_ranked_by_query_and_class() -> None:
    """Catch wrong ranking, a lost label shift, or boxes left in canvas coordinates."""
    logits = torch.full((1, 2, 8), -10.0)
    logits[0, 0, 3], logits[0, 1, 7], logits[0, 0, 5] = 2.0, 1.0, 0.0
    boxes = torch.tensor([[[0.15, 0.135, 0.1, 0.09], [0.99, 0.5, 0.1, 0.1]]])
    prediction = decode_predictions(logits, boxes, (640, 1024), [(576, 1024)], [(765, 1360)], 3)[0]
    assert prediction["labels"].tolist() == [4, 8, 6]
    assert torch.allclose(
        prediction["scores"], torch.sigmoid(torch.tensor([2.0, 1.0, 0.0])), atol=1e-6
    )
    assert torch.allclose(prediction["boxes"][0], torch.tensor([136.0, 76.5, 272.0, 153.0]))
    assert torch.allclose(prediction["boxes"][2], prediction["boxes"][0])
    assert prediction["boxes"][1, 2] == 1360


def test_decoded_detr_class_seven_reaches_the_submission_as_seven() -> None:
    """Catch the last class being dropped or shifted on its way to the CSV."""
    logits = torch.full((1, 1, 8), -10.0)
    logits[0, 0, 7] = 3.0
    boxes = torch.tensor([[[0.15, 0.135, 0.1, 0.09]]])
    predictions = decode_predictions(logits, boxes, (640, 1024), [(576, 1024)], [(765, 1360)], 1)
    frames = pl.DataFrame(
        {"submission_id": ["clip_001_0000001"], "image_width": [1360], "image_height": [765]}
    )
    sample = pl.DataFrame({"image_id": ["clip_001_0000001"], "PredictionString": [""]})
    tokens = build_submission(predictions, frames, sample)["PredictionString"][0].split()
    assert tokens[0] == "7"
    assert [float(value) for value in tokens[2:]] == pytest.approx([136.0, 76.5, 136.0, 76.5])


def test_matcher_pairs_each_object_with_its_closest_query() -> None:
    """Catch swapped indices or a matcher that ignores the box costs."""
    logits = torch.zeros((1, 3, 8))
    boxes = torch.tensor([[[0.9, 0.9, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1], [0.6, 0.6, 0.2, 0.2]]])
    targets = [
        {
            "class_labels": torch.tensor([2, 5]),
            "boxes": torch.tensor([[0.6, 0.6, 0.2, 0.2], [0.31, 0.3, 0.1, 0.1]]),
        }
    ]
    queries, objects = match_queries(logits, boxes, targets, MATCHER)[0]
    assert dict(zip(objects.tolist(), queries.tolist(), strict=True)) == {0: 2, 1: 1}


def test_matcher_handles_images_without_objects() -> None:
    """Catch a crash on a batch whose frames contain no objects."""
    targets = [{"class_labels": torch.zeros(0, dtype=torch.int64), "boxes": torch.zeros((0, 4))}]
    queries, objects = match_queries(
        torch.zeros((1, 3, 8)), torch.rand((1, 3, 4)), targets, MATCHER
    )[0]
    assert queries.numel() == objects.numel() == 0


def test_ignore_rule_drops_only_unmatched_queries_mostly_inside_a_region() -> None:
    """Catch the rule touching matched queries or queries outside ignored regions."""
    boxes = torch.tensor(
        [
            [
                [0.2, 0.2, 0.1, 0.1],
                [0.3, 0.3, 0.1, 0.1],
                [0.8, 0.8, 0.1, 0.1],
                [0.5, 0.2, 0.2, 0.1],
                [0.55, 0.2, 0.2, 0.1],
            ]
        ]
    )
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    regions = [torch.tensor([[0.0, 0.0, 50.0, 50.0]])]
    keep = unmatched_ignore_keep(boxes, indices, regions, (100, 100), 0.5)
    assert keep.tolist() == [[1.0, 0.0, 1.0, 0.0, 1.0]]
    no_regions = unmatched_ignore_keep(boxes, indices, [torch.zeros((0, 4))], (100, 100), 0.5)
    assert no_regions.tolist() == [[1.0] * 5]


def test_dropped_query_gets_no_classification_loss_or_gradient() -> None:
    """Catch an ignored query that is still pushed towards background."""
    logits = torch.zeros((1, 3, 8), requires_grad=True)
    boxes = torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]])
    targets = [{"class_labels": torch.tensor([2]), "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1]])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    keep = torch.tensor([[1.0, 0.0, 1.0]])
    class_loss, _, _ = compute_layer_losses(
        logits, boxes, targets, indices, 1.0, LOSS, (100, 100), keep
    )
    class_loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1].abs().sum() == 0
    assert logits.grad[0, 0].abs().sum() > 0 and logits.grad[0, 2].abs().sum() > 0
    full_loss, _, _ = compute_layer_losses(logits, boxes, targets, indices, 1.0, LOSS, (100, 100))
    assert full_loss > class_loss


def test_layer_losses_equal_hand_computed_values() -> None:
    """Catch a wrong varifocal weight, soft target or normalization."""
    logits = torch.zeros((1, 1, 8))
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    targets = [{"class_labels": torch.tensor([3]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    class_loss, l1_loss, box_loss = compute_layer_losses(
        logits, boxes, targets, indices, 2.0, LOSS, (100, 100)
    )
    negative_weight = 0.75 * 0.5**2
    expected = (1 + 7 * negative_weight) * math.log(2) / 2
    assert class_loss.item() == pytest.approx(expected, rel=1e-5)
    assert l1_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert box_loss.item() == pytest.approx(0.0, abs=1e-6)


def test_paired_overlaps_match_hand_values() -> None:
    """Catch an IoU or GIoU formula error in the box loss."""
    first = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    second = torch.tensor([[1.0, 0.0, 3.0, 2.0]])
    assert paired_iou(first, second).item() == pytest.approx(2 / 6)
    disjoint = torch.tensor([[4.0, 0.0, 6.0, 2.0]])
    assert paired_giou(first, disjoint).item() == pytest.approx(0 - (12 - 8) / 12)


def test_nwd_matches_the_gaussian_wasserstein_formula() -> None:
    """Catch a wrong NWD distance, which the loss-modification experiment depends on."""
    box = torch.tensor([[50.0, 50.0, 10.0, 10.0]])
    assert paired_nwd(box, box, 12.0).item() == pytest.approx(1.0, abs=1e-4)
    shifted = torch.tensor([[53.0, 50.0, 10.0, 10.0]])
    assert paired_nwd(box, shifted, 12.0).item() == pytest.approx(math.exp(-3 / 12), rel=1e-5)
    wider = torch.tensor([[50.0, 50.0, 16.0, 10.0]])
    assert paired_nwd(box, wider, 12.0).item() == pytest.approx(math.exp(-3 / 12), rel=1e-5)


def test_nwd_loss_stays_informative_for_disjoint_tiny_boxes() -> None:
    """Catch a box loss that gives the same value for near and far misses of tiny boxes."""
    target = [{"class_labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.01, 0.01]])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    settings = {**LOSS, "box_loss": "nwd"}
    near = torch.tensor([[[0.52, 0.5, 0.01, 0.01]]])
    far = torch.tensor([[[0.6, 0.5, 0.01, 0.01]]])
    near_loss = compute_layer_losses(
        torch.zeros((1, 1, 8)), near, target, indices, 1.0, settings, (100, 100)
    )[2]
    far_loss = compute_layer_losses(
        torch.zeros((1, 1, 8)), far, target, indices, 1.0, settings, (100, 100)
    )[2]
    assert 0 < near_loss < far_loss < 1


def _denoising_outputs(objects: int, queries: int, layers: int) -> tuple:
    torch.manual_seed(0)
    total = 2 * objects + queries
    logits = torch.randn((1, layers, total, 8))
    boxes = torch.rand((1, layers, total, 4)) * 0.2 + 0.1
    meta = {
        "dn_positive_idx": [torch.arange(objects)],
        "dn_num_group": 1,
        "dn_num_split": [2 * objects, queries],
    }
    return (
        logits,
        boxes,
        torch.randn((1, queries, 8)),
        torch.rand((1, queries, 4)) * 0.2 + 0.1,
        meta,
    )


def test_detection_losses_cover_every_part_and_count_dropped_queries() -> None:
    """Catch a missing loss part or an ignore rule that also removes matched queries."""
    logits, boxes, encoder_logits, encoder_boxes, meta = _denoising_outputs(2, 5, 3)
    targets = [
        {
            "class_labels": torch.tensor([0, 4]),
            "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.25, 0.2, 0.05, 0.1]]),
        }
    ]
    regions = [torch.tensor([[0.0, 0.0, 100.0, 100.0]])]
    losses, diagnostics = compute_detection_losses(
        logits,
        boxes,
        encoder_logits,
        encoder_boxes,
        meta,
        targets,
        regions,
        (100, 100),
        LOSS,
        MATCHER,
        0.5,
    )
    assert set(losses) == {f"loss_{term}_{part}" for part in LOSS_PARTS for term in LOSS_TERMS}
    assert all(torch.isfinite(value) and value > 0 for value in losses.values())
    assert diagnostics["matched_queries"] == 2
    assert diagnostics["excluded_queries_final"] == 3
    assert diagnostics["excluded_queries_all_layers"] == 3 * 4
    assert diagnostics["denoising_groups"] == 1
    plain, plain_diagnostics = compute_detection_losses(
        logits,
        boxes,
        encoder_logits,
        encoder_boxes,
        meta,
        targets,
        regions,
        (100, 100),
        LOSS,
        MATCHER,
        None,
    )
    assert plain_diagnostics["excluded_queries_final"] == 0
    assert plain["loss_class_final"] > losses["loss_class_final"]
    assert plain["loss_class_denoising"] == losses["loss_class_denoising"]
    assert plain["loss_l1_final"] == losses["loss_l1_final"]


def test_parameter_groups_follow_the_rt_detr_recipe() -> None:
    """Catch the backbone training at the transformer rate or decayed biases and norms."""
    backbone = nn.Parameter(torch.zeros(2, 2))
    weight = nn.Parameter(torch.zeros(2, 2))
    bias = nn.Parameter(torch.zeros(2))
    frozen = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    groups = split_parameter_groups(
        [
            ("detector.model.backbone.conv.weight", backbone),
            ("detector.model.decoder.linear.weight", weight),
            ("detector.model.decoder.linear.bias", bias),
            ("detector.model.encoder.frozen", frozen),
        ],
        1e-4,
        1e-5,
        1e-4,
    )
    by_name = {group["name"]: group for group in groups}
    assert by_name["backbone"]["params"] == [backbone] and by_name["backbone"]["lr"] == 1e-5
    assert by_name["decay"]["params"] == [weight] and by_name["decay"]["weight_decay"] == 1e-4
    assert by_name["no_decay"]["params"] == [bias] and by_name["no_decay"]["weight_decay"] == 0
    assert all(parameter is not frozen for group in groups for parameter in group["params"])


def test_clipping_sees_unscaled_gradients_and_records_their_norm() -> None:
    """Catch clipping applied to AMP-scaled gradients or a missing norm record."""
    parameter = nn.Parameter(torch.zeros(2))
    optimizer = GradientClippedAdamW([parameter], max_norm=1.0, lr=0.1, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    optimizer.zero_grad()
    scaler.scale((parameter * torch.tensor([3.0, 4.0])).sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    assert optimizer.last_gradient_norm == pytest.approx(5.0)
    assert parameter.grad is not None
    assert torch.allclose(parameter.grad, torch.tensor([0.6, 0.8]))
    optimizer.zero_grad()
    assert optimizer.last_gradient_norm is None


def test_schedule_warms_up_then_decays_like_the_cnn_runs() -> None:
    """Catch a schedule whose shape differs from the Faster R-CNN and RetinaNet runs."""
    parameters = nn.ParameterList([nn.Parameter(torch.zeros(2, 2))])
    training = {
        "learning_rate": 1e-4,
        "backbone_learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 0.1,
        "warmup_updates": 3,
    }
    optimizer, scheduler = make_rt_detr_optimizer(parameters, training, 30)
    rates = []
    for _ in range(30):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] == pytest.approx(1e-6)
    assert rates[3] == pytest.approx(1e-4)
    assert rates[19] == pytest.approx(1e-4)
    assert rates[20] == pytest.approx(1e-5)
    assert rates[27] == pytest.approx(1e-6)


def test_head_reset_uses_the_class_prior_and_skips_other_layers() -> None:
    """Catch new class heads starting at a high score or other layers being reset."""
    class_head, label_embed, box = nn.Linear(4, 8), nn.Embedding(9, 4), nn.Linear(4, 5)
    layers = nn.ModuleDict({"class_head": class_head, "label_embed": label_embed, "box": box})
    box_weight = box.weight.detach().clone()
    names = reset_classification_heads(layers, 2026)
    assert names == ["class_head", "label_embed"]
    prior = torch.sigmoid(class_head.bias)
    assert torch.allclose(prior, torch.full((8,), CLASS_PRIOR_PROBABILITY))
    assert torch.equal(box.weight, box_weight)


def test_detector_overrides_free_the_fixed_640_anchor_size() -> None:
    """Catch evaluation at 1024 by 576 with anchors generated for 640 by 640."""
    overrides = build_detector_overrides({**MODEL, "num_queries": 500}, LOSS, MATCHER)
    assert overrides["anchor_image_size"] is None and overrides["eval_size"] is None
    assert overrides["num_labels"] == 8 and overrides["num_queries"] == 500
    assert overrides["id2label"][7] == "motor"
    assert overrides["weight_loss_giou"] == LOSS["box_weight"]
