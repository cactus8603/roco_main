"""Losses for the optional U1 bounded flow-refinement stage.

U0 observes a frozen matcher.  U1 is deliberately a separate stage: it keeps
the matcher and calibrated uncertainty provider frozen, predicts only a
bounded residual, and is trained on held-out ground-truth flow.  This module
contains no runtime activation path.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import torch
except ImportError:  # pragma: no cover - contract readers need no torch
    torch = None


@dataclass(frozen=True)
class UncertaintyRefinerLossPolicyV1:
    """Weights for a conservative, action-bank-independent U1 objective.

    The harm term penalizes degradation relative to the frozen native flow on
    every labelled pixel.  The anchor and total-variation terms regularize the
    residual itself; neither can authorize runtime use without calibration.
    """

    task_weight: float = 1.0
    harm_weight: float = 2.0
    anchor_weight: float = 0.05
    smoothness_weight: float = 0.01
    harm_margin_px: float = 0.0
    charbonnier_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        names = (
            "task_weight", "harm_weight", "anchor_weight", "smoothness_weight",
            "harm_margin_px", "charbonnier_epsilon",
        )
        for name in names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite and nonnegative")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.charbonnier_epsilon <= 0.0:
            raise ValueError("charbonnier epsilon must be positive")
        if self.task_weight + self.harm_weight <= 0.0:
            raise ValueError("U1 needs a positive task or harm objective")


@dataclass(frozen=True)
class UncertaintyRefinerLossV1:
    total: object
    task: object
    harm: object
    anchor: object
    smoothness: object
    valid_count: int


def _valid_4d(mask, reference):
    if mask.shape == reference[:, 0].shape:
        result = mask[:, None]
    elif mask.shape == reference[:, :1].shape:
        result = mask
    else:
        raise ValueError("valid mask must have shape NHW or N1HW")
    if result.dtype != torch.bool:
        raise ValueError("valid mask must be boolean")
    if not bool(result.any()):
        raise ValueError("U1 loss has empty valid support")
    return result


def _masked_total_variation(delta, valid):
    horizontal_valid = valid[..., :, 1:] & valid[..., :, :-1]
    vertical_valid = valid[..., 1:, :] & valid[..., :-1, :]
    terms = []
    if bool(horizontal_valid.any()):
        horizontal = (delta[..., :, 1:] - delta[..., :, :-1]).abs()
        terms.append(horizontal[horizontal_valid.expand_as(horizontal)].mean())
    if bool(vertical_valid.any()):
        vertical = (delta[..., 1:, :] - delta[..., :-1, :]).abs()
        terms.append(vertical[vertical_valid.expand_as(vertical)].mean())
    if not terms:
        return delta.sum() * 0.0
    return torch.stack(terms).mean()


def uncertainty_refiner_loss_v1(
    refined_flow,
    base_flow,
    ground_truth_flow,
    valid_mask,
    *,
    policy: UncertaintyRefinerLossPolicyV1 = UncertaintyRefinerLossPolicyV1(),
) -> UncertaintyRefinerLossV1:
    """Compute supervised task, per-pixel harm, anchor and TV losses.

    ``base_flow`` is detached in all reference terms so this objective cannot
    train the production matcher accidentally.  Callers must additionally
    freeze the matcher and U0 provider parameters.
    """

    if torch is None:
        raise ImportError("uncertainty refiner loss requires torch")
    if not isinstance(policy, UncertaintyRefinerLossPolicyV1):
        raise ValueError("U1 loss needs a typed policy")
    if refined_flow.shape != base_flow.shape or refined_flow.shape != ground_truth_flow.shape:
        raise ValueError("refined, base and ground-truth flow shapes must match")
    if refined_flow.ndim != 4 or refined_flow.shape[1] != 2:
        raise ValueError("U1 flows must have shape N2HW")
    if not all(bool(torch.isfinite(value).all()) for value in (
        refined_flow, base_flow, ground_truth_flow,
    )):
        raise ValueError("U1 flows must be finite")
    valid = _valid_4d(valid_mask, refined_flow)
    pixel_valid = valid[:, 0]
    frozen_base = base_flow.detach()
    refined_error = torch.linalg.vector_norm(
        refined_flow - ground_truth_flow, dim=1,
    )
    base_error = torch.linalg.vector_norm(
        frozen_base - ground_truth_flow, dim=1,
    )
    selected_error = refined_error[pixel_valid]
    epsilon = policy.charbonnier_epsilon
    task = (torch.sqrt(selected_error.square() + epsilon * epsilon) - epsilon).mean()
    degradation = torch.relu(
        selected_error - base_error[pixel_valid].detach() - policy.harm_margin_px
    )
    harm = degradation.square().mean()
    delta = refined_flow - frozen_base
    anchor = torch.linalg.vector_norm(delta, dim=1)[pixel_valid].mean()
    smoothness = _masked_total_variation(delta, valid)
    total = (
        policy.task_weight * task
        + policy.harm_weight * harm
        + policy.anchor_weight * anchor
        + policy.smoothness_weight * smoothness
    )
    return UncertaintyRefinerLossV1(
        total=total,
        task=task,
        harm=harm,
        anchor=anchor,
        smoothness=smoothness,
        valid_count=int(pixel_valid.sum().item()),
    )


__all__ = [
    "UncertaintyRefinerLossPolicyV1",
    "UncertaintyRefinerLossV1",
    "uncertainty_refiner_loss_v1",
]
