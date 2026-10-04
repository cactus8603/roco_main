from __future__ import annotations

import pytest

from stablebridge.physical_repair.uncertainty_aware_flow import (
    BoundedUncertaintyFlowRefinerV1,
)
from stablebridge.physical_repair.uncertainty_refinement_training import (
    UncertaintyRefinerLossPolicyV1,
    uncertainty_refiner_loss_v1,
)


def test_refiner_preserves_invalid_pixels_and_respects_pointwise_cap():
    torch = pytest.importorskip("torch")
    refiner = BoundedUncertaintyFlowRefinerV1(3, maximum_update_px=0.2)
    features = torch.zeros((1, 3, 2, 3))
    base = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    log_variance = torch.full((1, 1, 2, 3), 10.0)
    valid = torch.tensor([[[True, False, True], [False, True, True]]])
    result = refiner(features, base, log_variance, valid)
    invalid = (~valid)[:, None].expand_as(base)
    assert torch.equal(result[invalid], base[invalid])
    delta_norm = torch.linalg.vector_norm(result - base, dim=1)
    assert float(delta_norm.max()) <= 0.2 + 1e-6


def test_refiner_rejects_nonboolean_valid_mask():
    torch = pytest.importorskip("torch")
    refiner = BoundedUncertaintyFlowRefinerV1(1)
    with pytest.raises(ValueError, match="boolean"):
        refiner(
            torch.zeros((1, 1, 2, 2)),
            torch.zeros((1, 2, 2, 2)),
            torch.zeros((1, 1, 2, 2)),
            torch.ones((1, 2, 2)),
        )


def test_u1_loss_penalizes_harm_relative_to_frozen_base():
    torch = pytest.importorskip("torch")
    base = torch.zeros((1, 2, 2, 2), requires_grad=True)
    truth = torch.zeros_like(base)
    refined = torch.ones_like(base, requires_grad=True)
    valid = torch.ones((1, 2, 2), dtype=torch.bool)
    result = uncertainty_refiner_loss_v1(refined, base, truth, valid)
    assert result.harm.item() > 0
    assert result.anchor.item() > 0
    result.total.backward()
    assert refined.grad is not None
    assert base.grad is None


def test_u1_loss_ignores_invalid_pixels_and_requires_support():
    torch = pytest.importorskip("torch")
    base = torch.zeros((1, 2, 1, 2))
    truth = torch.zeros_like(base)
    refined = torch.zeros_like(base)
    refined[:, :, :, 1] = 100.0
    valid = torch.tensor([[[True, False]]])
    result = uncertainty_refiner_loss_v1(refined, base, truth, valid)
    assert result.task.item() == pytest.approx(0.0)
    assert result.harm.item() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="empty"):
        uncertainty_refiner_loss_v1(
            refined, base, truth, torch.zeros_like(valid, dtype=torch.bool),
        )


def test_u1_loss_policy_rejects_unconstrained_objective():
    with pytest.raises(ValueError, match="positive task or harm"):
        UncertaintyRefinerLossPolicyV1(task_weight=0.0, harm_weight=0.0)
