"""WAFT sequence mixture-of-Laplace loss with one GT-defined denominator.

The returned numerator can be accumulated across microbatches. Invalid labels
are indexed out before arithmetic, so NaN * 0 cannot contaminate the graph.
Nonfinite predictions at supervised pixels are errors, not dropped labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class LossSettings:
    gamma: float = 0.85
    var_min: float = 0.0
    var_max: float = 10.0
    max_flow: float = 40000.0

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.gamma, self.var_min, self.var_max, self.max_flow)):
            raise ValueError("Loss settings must be finite")
        if not 0 < self.gamma <= 1 or self.var_min > 0 or self.var_max < 0 or self.max_flow <= 0:
            raise ValueError("Invalid sequence MoL settings")


def supervision_mask(flow_gt: torch.Tensor, valid: torch.Tensor, max_flow: float) -> torch.Tensor:
    if flow_gt.ndim != 4 or flow_gt.shape[1] != 2 or valid.shape != flow_gt.shape[:1] + flow_gt.shape[2:]:
        raise ValueError("Expected GT B2HW and valid BHW")
    if flow_gt.dtype not in (torch.float32, torch.float64) or valid.dtype != torch.bool:
        raise ValueError("Loss requires FP32/FP64 GT and boolean valid")
    if not torch.isfinite(flow_gt.permute(0, 2, 3, 1)[valid]).all():
        raise ValueError("Nonfinite GT marked valid")
    # Double precision avoids overflow while deciding the fixed label population.
    magnitude = torch.linalg.vector_norm(flow_gt.detach().double(), dim=1)
    return valid & (magnitude < max_flow)


def sequence_mol_terms(output: dict, flow_gt: torch.Tensor, valid: torch.Tensor,
                       settings: LossSettings = LossSettings()) -> tuple[torch.Tensor, int]:
    """Return weighted sequence loss SUM and valid scalar-component COUNT.

    Matches the pinned official loss for finite admitted predictions: two
    mixture components share weights/scales for u/v; normalize over 2*N valid
    components, then sum gamma-weighted recurrent losses. No image mean or
    per-microbatch mean is taken here.
    """
    flows, infos = output.get("flow"), output.get("info")
    if not isinstance(flows, (list, tuple)) or not flows or not isinstance(infos, (list, tuple)) or len(flows) != len(infos):
        raise ValueError("Aligned nonempty flow/info prediction sequences required")
    mask = supervision_mask(flow_gt, valid, settings.max_flow)
    count = 2 * int(mask.sum())
    target = flow_gt.detach().permute(0, 2, 3, 1)[mask]
    numerator = None
    for i, (prediction, info) in enumerate(zip(flows, infos)):
        if prediction.shape != flow_gt.shape or info.shape != (flow_gt.shape[0], 4, *flow_gt.shape[2:]):
            raise ValueError("Prediction/info must share the original GT grid")
        if prediction.dtype not in (torch.float32, torch.float64) or info.dtype != prediction.dtype:
            raise ValueError("Unvalidated loss precision")
        predicted = prediction.permute(0, 2, 3, 1)[mask]
        parameters = info.permute(0, 2, 3, 1)[mask]
        if not torch.isfinite(predicted).all() or not torch.isfinite(parameters).all():
            raise FloatingPointError("Nonfinite prediction at a supervised pixel")
        if not count:
            # Empty indexing keeps a finite, connected zero graph even if raw
            # invalid positions contain NaNs. The trainer skips an empty window.
            loss_sum = predicted.sum() * 0 + parameters.sum() * 0
        else:
            weights = parameters[:, :2]
            raw_scale = parameters[:, 2:]
            log_scale = torch.stack((raw_scale[:, 0].clamp(0, settings.var_max),
                                     raw_scale[:, 1].clamp(settings.var_min, 0)), dim=1)
            distance = (target.to(predicted.dtype) - predicted).abs()
            log_components = weights - math.log(2.0) - log_scale
            log_components = log_components[:, None, :] - distance[:, :, None] * torch.exp(-log_scale)[:, None, :]
            nll = torch.logsumexp(weights, dim=1)[:, None] - torch.logsumexp(log_components, dim=2)
            loss_sum = nll.sum()
        term = settings.gamma ** (len(flows) - i - 1) * loss_sum
        numerator = term if numerator is None else numerator + term
    return numerator, count


def sequence_mol_loss(output: dict, flow_gt: torch.Tensor, valid: torch.Tensor,
                      settings: LossSettings = LossSettings()) -> torch.Tensor:
    numerator, count = sequence_mol_terms(output, flow_gt, valid, settings)
    return numerator / count if count else numerator
