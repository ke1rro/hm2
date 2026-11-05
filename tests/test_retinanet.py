from pathlib import Path

import pytest
import torch

from hm2.experiment_checkpoint import save_checkpoint
from hm2.retinanet import build_retinanet, load_checkpoint, make_optimizer, train_steps


@pytest.fixture(autouse=True)
def limited_cpu_threads() -> None:
    """Keep tiny architecture tests from oversubscribing the CPU."""
    torch.set_num_threads(2)


def test_three_runs_change_only_intended_model_settings() -> None:
    """Catch incompatible anchor counts, head initialization drift, or stacked ablations."""
    baseline = build_retinanet("standard_anchors", pretrained=False, min_size=64, max_size=96)
    anchors = build_retinanet("smaller_anchors", pretrained=False, min_size=64, max_size=96)
    diou = build_retinanet("diou_regression", pretrained=False, min_size=64, max_size=96)
    assert baseline.head.classification_head.num_classes == 9
    assert baseline.head.classification_head.cls_logits.out_channels == 81
    assert baseline.anchor_generator.num_anchors_per_location() == [9] * 5
    assert anchors.anchor_generator.sizes[0] == (16, 20, 25)
    assert baseline.anchor_generator.sizes[0] == (32, 40, 50)
    assert anchors.anchor_generator.aspect_ratios == baseline.anchor_generator.aspect_ratios
    assert diou.anchor_generator.sizes == baseline.anchor_generator.sizes
    assert (
        baseline.head.regression_head._loss_type
        == anchors.head.regression_head._loss_type
        == "giou"
    )
    assert diou.head.regression_head._loss_type == "diou"
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, anchors.state_dict()[name])
        torch.testing.assert_close(tensor, diou.state_dict()[name])


def test_real_diou_forward_backward_and_checkpoint_resume(tmp_path: Path) -> None:
    """Catch inactive DIoU, nonfinite updates, coordinate scaling, or incomplete resume."""
    model = build_retinanet("R2", pretrained=False, min_size=64, max_size=96)
    optimizer, scheduler = make_optimizer(model, lr=0.001, total_steps=6, warmup_steps=1)
    image = torch.rand(3, 80, 120)
    target = {
        "boxes": torch.tensor([[12.0, 10.0, 60.0, 55.0]]),
        "labels": torch.tensor([8]),
        "image_id": torch.tensor(7),
    }
    before = model.head.regression_head.bbox_reg.weight.detach().clone()
    gradient_calls = 0

    def _skip_first_update(gradient: torch.Tensor) -> torch.Tensor:
        nonlocal gradient_calls
        gradient_calls += 1
        return gradient * float("inf") if gradient_calls == 1 else gradient

    hook = model.head.regression_head.bbox_reg.weight.register_hook(_skip_first_update)
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0)
    history = train_steps(
        model,
        [([image], [target])],
        optimizer,
        scheduler,
        torch.device("cpu"),
        max_steps=1,
        scaler=scaler,
        max_attempts=2,
    )
    hook.remove()
    assert history[0]["loss"] > 0
    assert [row["optimizer_updated"] for row in history] == [False, True]
    assert [row["step"] for row in history] == [0, 1]
    assert not torch.equal(before, model.head.regression_head.bbox_reg.weight)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model, optimizer, scheduler, step=1, config={"run": "R2", "seed": 2026}, scaler=scaler
    )
    restored = build_retinanet("R2", pretrained=False, min_size=64, max_size=96)
    resumed_optimizer, resumed_scheduler = make_optimizer(
        restored, lr=0.001, total_steps=6, warmup_steps=1
    )
    resumed_scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    state = load_checkpoint(path, restored, resumed_optimizer, resumed_scheduler, resumed_scaler)
    assert state["step"] == 1 and state["config"]["run"] == "R2"
    torch.testing.assert_close(
        restored.head.regression_head.bbox_reg.weight, model.head.regression_head.bbox_reg.weight
    )
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    assert resumed_scaler.state_dict() == scaler.state_dict()
    resumed_history = train_steps(
        restored,
        [([image], [target])],
        resumed_optimizer,
        resumed_scheduler,
        torch.device("cpu"),
        max_steps=1,
        start_step=state["step"],
        scaler=resumed_scaler,
    )
    assert resumed_history[0]["step"] == 2
    assert not torch.equal(
        model.head.regression_head.bbox_reg.weight, restored.head.regression_head.bbox_reg.weight
    )
    restored.eval()
    restored.score_thresh = 0.0
    with torch.inference_mode():
        prediction = restored([image])[0]
    assert prediction["boxes"].shape[1] == 4
    assert prediction["boxes"][:, 0::2].max() <= 120
    assert prediction["boxes"][:, 1::2].max() <= 80
