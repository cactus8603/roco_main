"""Paper-aligned uncertainty supervision and recurrent SEA-RAFT refinement.

The earlier U0 observer is intentionally retained as a post-hoc baseline.  The
components in this module implement the missing U2Flow mechanism: uncertainty
is predicted from every recurrent hidden state, detached before it conditions
the flow residual, and trained with an augmentation-consistency likelihood
whose residual target cannot backpropagate into the flow branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentUncertaintyVariantV2(str, Enum):
    HEAD_ONLY = "head_only"
    REFINEMENT_WITHOUT_UNCERTAINTY = "refinement_without_uncertainty"
    REFINEMENT_WITH_UNCERTAINTY = "refinement_with_uncertainty"


class TrainableScopeV2(str, Enum):
    UNCERTAINTY_ONLY = "uncertainty_only"
    REFINEMENT_HEADS = "refinement_heads"
    ALL = "all"


class UncertaintyAwareSeaRaftFlowHeadV2(nn.Module):
    """Wrap SEA-RAFT's six-channel head without changing its info channels.

    The correction branch is zero-initialized, so installing this module is an
    exact flow-preserving operation before optimization.  The two uncertainty
    inputs to the correction branch are detached, matching U2Flow's gradient
    boundary.  ``REFINEMENT_WITHOUT_UNCERTAINTY`` is the parameter-matched
    dummy-map ablation from the paper.
    """

    def __init__(
        self,
        base_head: nn.Module,
        *,
        hidden_channels: int,
        refinement_channels: int = 64,
        variant: RecurrentUncertaintyVariantV2 | str = (
            RecurrentUncertaintyVariantV2.REFINEMENT_WITH_UNCERTAINTY
        ),
    ) -> None:
        super().__init__()
        if not isinstance(base_head, nn.Module):
            raise TypeError("base_head must be a torch module")
        if hidden_channels <= 0 or refinement_channels <= 0:
            raise ValueError("head channel counts must be positive")
        self.base_head = base_head
        self.hidden_channels = int(hidden_channels)
        self.variant = RecurrentUncertaintyVariantV2(variant)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(self.hidden_channels, 1, 3, padding=1),
        )
        self.refinement_head = nn.Sequential(
            nn.Conv2d(5, refinement_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(refinement_channels, 2, 3, padding=1),
        )
        # Preserve the pretrained matcher exactly at initialization.
        nn.init.zeros_(self.refinement_head[-1].weight)
        nn.init.zeros_(self.refinement_head[-1].bias)
        self._capture = False
        self._captured_log_variances: list[torch.Tensor] = []

    def begin_capture(self) -> None:
        self._captured_log_variances = []
        self._capture = True

    def end_capture(self) -> tuple[torch.Tensor, ...]:
        result = tuple(self._captured_log_variances)
        self._captured_log_variances = []
        self._capture = False
        return result

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base = self.base_head(hidden)
        if base.ndim != 4 or base.shape[1] < 2:
            raise ValueError("SEA-RAFT base flow head must return NCHW with C>=2")
        alpha = self.uncertainty_head(hidden)
        if self._capture:
            self._captured_log_variances.append(alpha)
        if self.variant is RecurrentUncertaintyVariantV2.HEAD_ONLY:
            return base
        flow_feature = base[:, :2]
        if self.variant is RecurrentUncertaintyVariantV2.REFINEMENT_WITH_UNCERTAINTY:
            detached_alpha = alpha.detach()
            reliability = torch.sigmoid(-detached_alpha)
        else:
            detached_alpha = torch.zeros_like(alpha)
            reliability = torch.ones_like(alpha)
        weighted = flow_feature * reliability
        correction = self.refinement_head(torch.cat(
            (flow_feature, weighted, detached_alpha), dim=1,
        ))
        return torch.cat((flow_feature + correction, base[:, 2:]), dim=1)


class UncertaintyAwareSeaRaftV2(nn.Module):
    """Install an uncertainty-aware head at every SEA-RAFT recurrent update."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        variant: RecurrentUncertaintyVariantV2 | str,
        trainable_scope: TrainableScopeV2 | str = TrainableScopeV2.ALL,
        refinement_channels: int = 64,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module) or not hasattr(backbone, "flow_head"):
            raise TypeError("SEA-RAFT backbone must expose flow_head")
        hidden_channels = int(getattr(getattr(backbone, "args", None), "dim", 0))
        if hidden_channels <= 0:
            # Tiny test doubles can expose the channel count directly.
            hidden_channels = int(getattr(backbone, "hidden_channels", 0))
        if hidden_channels <= 0:
            raise ValueError("cannot infer SEA-RAFT recurrent hidden channels")
        original_head = backbone.flow_head
        self.recurrent_head = UncertaintyAwareSeaRaftFlowHeadV2(
            original_head,
            hidden_channels=hidden_channels,
            refinement_channels=refinement_channels,
            variant=variant,
        )
        backbone.flow_head = self.recurrent_head
        self.backbone = backbone
        self.variant = RecurrentUncertaintyVariantV2(variant)
        self.trainable_scope = TrainableScopeV2(trainable_scope)
        self._apply_trainable_scope()

    def _apply_trainable_scope(self) -> None:
        self.requires_grad_(self.trainable_scope is TrainableScopeV2.ALL)
        if self.trainable_scope is not TrainableScopeV2.ALL:
            self.recurrent_head.uncertainty_head.requires_grad_(True)
        if self.trainable_scope is TrainableScopeV2.REFINEMENT_HEADS:
            self.recurrent_head.refinement_head.requires_grad_(
                self.variant is not RecurrentUncertaintyVariantV2.HEAD_ONLY
            )
        if self.trainable_scope is TrainableScopeV2.UNCERTAINTY_ONLY:
            self.recurrent_head.refinement_head.requires_grad_(False)
        # The pretrained six-channel head is never silently included in a
        # heads-only optimizer.
        if self.trainable_scope is not TrainableScopeV2.ALL:
            self.recurrent_head.base_head.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.trainable_scope is not TrainableScopeV2.ALL:
            self.backbone.eval()
        return self

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, **kwargs):
        masks: list[torch.Tensor] = []
        handle = None
        upsample_weight = getattr(self.backbone, "upsample_weight", None)
        if isinstance(upsample_weight, nn.Module):
            handle = upsample_weight.register_forward_hook(
                lambda _module, _inputs, output: masks.append(0.25 * output)
            )
        self.recurrent_head.begin_capture()
        try:
            output = self.backbone(image1, image2, **kwargs)
            captured = self.recurrent_head.end_capture()
        except BaseException:
            self.recurrent_head.end_capture()
            raise
        finally:
            if handle is not None:
                handle.remove()
        if not isinstance(output, Mapping):
            raise TypeError("SEA-RAFT output must be a mapping")
        flows = output.get("flow")
        if not isinstance(flows, (tuple, list)) or len(flows) != len(captured):
            raise ValueError("flow and recurrent uncertainty sequences must align")
        full_resolution = []
        use_convex = (
            len(masks) == len(captured)
            and callable(getattr(self.backbone, "upsample_data", None))
        )
        for index, (alpha, flow) in enumerate(zip(captured, flows)):
            if use_convex:
                dummy_flow = torch.zeros(
                    alpha.shape[0], 2, *alpha.shape[-2:],
                    dtype=alpha.dtype, device=alpha.device,
                )
                _unused, upsampled = self.backbone.upsample_data(
                    dummy_flow, alpha, masks[index].detach(),
                )
                if upsampled.shape[-2:] != flow.shape[-2:]:
                    delta_h = upsampled.shape[-2] - flow.shape[-2]
                    delta_w = upsampled.shape[-1] - flow.shape[-1]
                    if delta_h < 0 or delta_w < 0:
                        raise ValueError("upsampled uncertainty is smaller than unpadded flow")
                    top, left = delta_h // 2, delta_w // 2
                    upsampled = upsampled[
                        ..., top : top + flow.shape[-2], left : left + flow.shape[-1]
                    ]
            else:
                upsampled = F.interpolate(
                    alpha, size=flow.shape[-2:], mode="bilinear", align_corners=True,
                )
            full_resolution.append(upsampled)
        result = dict(output)
        result["uncertainty_log_variance"] = tuple(full_resolution)
        return result


def make_pinned_uncertainty_aware_sea_raft_v2(
    *,
    device: str,
    variant: str,
    trainable_scope: str = "all",
    refinement_channels: int = 64,
    vendor_root: str | None = None,
    config_path: str | None = None,
    checkpoint: str | None = None,
) -> UncertaintyAwareSeaRaftV2:
    """JSON-friendly loader for the pinned Spring-M SEA-RAFT checkpoint."""

    from .sea_raft_uncertainty_adapter import load_pinned_sea_raft_model

    model, _report = load_pinned_sea_raft_model(
        device=device,
        vendor_root=None if vendor_root is None else Path(vendor_root),
        config_path=None if config_path is None else Path(config_path),
        checkpoint=None if checkpoint is None else Path(checkpoint),
    )
    return UncertaintyAwareSeaRaftV2(
        model,
        variant=variant,
        trainable_scope=trainable_scope,
        refinement_channels=refinement_channels,
    ).to(torch.device(device))


def transform_flow_affine_v2(
    flow: torch.Tensor,
    native_to_augmented: torch.Tensor,
    output_hw: Sequence[int],
    *,
    native_valid: torch.Tensor | None = None,
    augmented_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport dense flow through the same affine applied to both endpoints.

    For an output pixel ``q``, the native vector is sampled at ``A^-1 q`` and
    transformed by the linear part of ``A``.  Support requires both the source
    point and its displaced endpoint to remain inside the native lattice.
    """

    if flow.ndim != 4 or flow.shape[1] != 2 or not flow.is_floating_point():
        raise ValueError("flow must be floating N2HW")
    if native_to_augmented.shape != (flow.shape[0], 3, 3):
        raise ValueError("native_to_augmented must have shape N33")
    if not bool(torch.isfinite(flow).all()) or not bool(torch.isfinite(native_to_augmented).all()):
        raise ValueError("flow transport inputs must be finite")
    if len(tuple(output_hw)) != 2:
        raise ValueError("output_hw must contain height and width")
    out_h, out_w = (int(value) for value in output_hw)
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output dimensions must be positive")
    batch, _, in_h, in_w = flow.shape
    dtype, device = flow.dtype, flow.device
    matrix = native_to_augmented.to(device=device, dtype=dtype)
    inverse = torch.linalg.inv(matrix)
    yy, xx = torch.meshgrid(
        torch.arange(out_h, device=device, dtype=dtype),
        torch.arange(out_w, device=device, dtype=dtype),
        indexing="ij",
    )
    points = torch.stack((xx, yy, torch.ones_like(xx)), dim=0).reshape(1, 3, -1)
    source = torch.bmm(inverse, points.expand(batch, -1, -1))
    source_x = source[:, 0].reshape(batch, out_h, out_w)
    source_y = source[:, 1].reshape(batch, out_h, out_w)
    if in_w > 1:
        grid_x = 2.0 * source_x / float(in_w - 1) - 1.0
    else:
        grid_x = torch.zeros_like(source_x)
    if in_h > 1:
        grid_y = 2.0 * source_y / float(in_h - 1) - 1.0
    else:
        grid_y = torch.zeros_like(source_y)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    sampled = F.grid_sample(
        flow, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    linear = matrix[:, :2, :2]
    transported = torch.einsum("bij,bjhw->bihw", linear, sampled)
    endpoint_x = source_x + sampled[:, 0]
    endpoint_y = source_y + sampled[:, 1]
    valid = (
        (source_x >= 0.0) & (source_x <= in_w - 1.0)
        & (source_y >= 0.0) & (source_y <= in_h - 1.0)
        & (endpoint_x >= 0.0) & (endpoint_x <= in_w - 1.0)
        & (endpoint_y >= 0.0) & (endpoint_y <= in_h - 1.0)
    )[:, None]
    if native_valid is not None:
        if native_valid.shape not in {flow[:, :1].shape, flow[:, 0].shape}:
            raise ValueError("native_valid must have shape N1HW or NHW")
        source_valid = native_valid if native_valid.ndim == 4 else native_valid[:, None]
        sampled_valid = F.grid_sample(
            source_valid.to(dtype), grid, mode="nearest", padding_mode="zeros",
            align_corners=True,
        ) >= 0.5
        valid &= sampled_valid
    if augmented_valid is not None:
        candidate = augmented_valid if augmented_valid.ndim == 4 else augmented_valid[:, None]
        if candidate.shape != valid.shape:
            raise ValueError("augmented_valid must match transported output support")
        valid &= candidate.to(device=device, dtype=torch.bool)
    return torch.where(valid.expand_as(transported), transported, torch.zeros_like(transported)), valid


@dataclass(frozen=True)
class DecoupledFlowLossPolicyV2:
    task_weight: float = 1.0
    augmentation_weight: float = 0.02
    uncertainty_weight: float = 0.005
    gamma: float = 0.8

    def __post_init__(self) -> None:
        for name in ("task_weight", "augmentation_weight", "uncertainty_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        gamma = float(self.gamma)
        if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must lie in (0,1]")
        object.__setattr__(self, "gamma", gamma)


@dataclass(frozen=True)
class DecoupledFlowLossV2:
    total: torch.Tensor
    task: torch.Tensor
    augmentation: torch.Tensor
    uncertainty: torch.Tensor


def _mask4(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    result = mask[:, None] if mask.ndim == 3 else mask
    if result.shape != reference[:, :1].shape or result.dtype is not torch.bool:
        raise ValueError("valid mask must be boolean N1HW or NHW")
    if not bool(result.any()):
        raise ValueError("loss has empty valid support")
    return result


def _weighted_sequence_mean(terms: Sequence[torch.Tensor], gamma: float) -> torch.Tensor:
    if not terms:
        raise ValueError("sequence loss needs predictions")
    count = len(terms)
    return sum(gamma ** (count - index - 1) * term for index, term in enumerate(terms))


def decoupled_uncertainty_flow_loss_v2(
    native_flow_predictions: Sequence[torch.Tensor],
    augmented_flow_predictions: Sequence[torch.Tensor],
    augmented_log_variances: Sequence[torch.Tensor],
    ground_truth_flow: torch.Tensor | None,
    native_valid: torch.Tensor | None,
    transformed_teacher_flow: torch.Tensor,
    consistency_valid: torch.Tensor,
    *,
    policy: DecoupledFlowLossPolicyV2 = DecoupledFlowLossPolicyV2(),
) -> DecoupledFlowLossV2:
    """Supervised task carrier plus U2Flow's decoupled AR/uncertainty losses."""

    native = tuple(native_flow_predictions)
    augmented = tuple(augmented_flow_predictions)
    uncertainty = tuple(augmented_log_variances)
    if not native or len(augmented) != len(uncertainty) or not augmented:
        raise ValueError("flow and uncertainty sequences must be nonempty and aligned")
    consistency_mask = _mask4(consistency_valid, transformed_teacher_flow)
    teacher = transformed_teacher_flow.detach()

    if (ground_truth_flow is None) != (native_valid is None):
        raise ValueError("ground-truth flow and validity must both be present or absent")
    if ground_truth_flow is None:
        if policy.task_weight != 0.0:
            raise ValueError("ground truth is required when task_weight is nonzero")
        # Preserve a tensor-valued loss record without granting the missing
        # label path any training role.  Native-only U0 uses this branch.
        task = native[-1].sum() * 0.0
    else:
        assert native_valid is not None
        native_mask = _mask4(native_valid, ground_truth_flow)
        task_terms = []
        for prediction in native:
            if prediction.shape != ground_truth_flow.shape:
                raise ValueError("native prediction and ground truth shapes must match")
            component = F.smooth_l1_loss(prediction, ground_truth_flow, reduction="none")
            task_terms.append(component[native_mask.expand_as(component)].mean())
        task = _weighted_sequence_mean(task_terms, policy.gamma)

    ar_terms, uncertainty_terms = [], []
    for prediction, alpha in zip(augmented, uncertainty):
        if prediction.shape != teacher.shape or alpha.shape != teacher[:, :1].shape:
            raise ValueError("augmented flow/uncertainty shapes must match teacher")
        difference = (prediction - teacher).abs()
        ar_component = F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction="none")
        ar_terms.append(ar_component[consistency_mask.expand_as(ar_component)].mean())
        # Paper Eq. 9: D is detached so this term cannot optimize flow through
        # the likelihood shortcut.  L_ar above remains the explicit flow path.
        discrepancy = difference.sum(dim=1, keepdim=True).detach()
        bounded_alpha = alpha.clamp(-30.0, 30.0)
        nll = math.sqrt(2.0) * torch.exp(-0.5 * bounded_alpha) * discrepancy + 0.5 * bounded_alpha
        uncertainty_terms.append(nll[consistency_mask].mean())

    augmentation = _weighted_sequence_mean(ar_terms, policy.gamma)
    uncertainty_loss = _weighted_sequence_mean(uncertainty_terms, policy.gamma)
    total = (
        policy.task_weight * task
        + policy.augmentation_weight * augmentation
        + policy.uncertainty_weight * uncertainty_loss
    )
    return DecoupledFlowLossV2(total, task, augmentation, uncertainty_loss)


__all__ = [
    "DecoupledFlowLossPolicyV2",
    "DecoupledFlowLossV2",
    "RecurrentUncertaintyVariantV2",
    "TrainableScopeV2",
    "UncertaintyAwareSeaRaftFlowHeadV2",
    "UncertaintyAwareSeaRaftV2",
    "decoupled_uncertainty_flow_loss_v2",
    "make_pinned_uncertainty_aware_sea_raft_v2",
    "transform_flow_affine_v2",
]
