"""Optional U2Flow-style three-frame bidirectional flow fusion.

The recurrent flow provider remains a two-frame model.  Given
``[previous, current, next]``, this module evaluates ``current -> next`` and
``current -> previous`` separately.  A fresh tiny motion model is then fitted
per sample on pixels where both directions are uncertainty-reliable.  Its
forward estimate replaces only pixels where the forward prediction is
unreliable while the backward prediction is reliable.

Fusion is deliberately disabled by default.  It is an inference/validation
post-process and never contributes gradients to the flow or uncertainty
provider.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class U2FlowBidirectionalFusionPolicyV1:
    """Configuration matching the public Sintel fusion implementation."""

    enabled: bool = False
    optimization_steps: int = 2700
    uncertainty_variance_threshold: float = 45.0
    learning_rate: float = 0.01
    learning_rate_decay: float = 0.8
    variance_minimum: float = 1e-3
    variance_maximum: float = 200.0
    random_seed: int = 8603

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("fusion enabled must be boolean")
        if (
            isinstance(self.optimization_steps, bool)
            or not isinstance(self.optimization_steps, int)
            or self.optimization_steps < 1
        ):
            raise ValueError("fusion optimization_steps must be a positive integer")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("fusion random_seed must be a nonnegative integer")
        for name in (
            "uncertainty_variance_threshold",
            "learning_rate",
            "learning_rate_decay",
            "variance_minimum",
            "variance_maximum",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"fusion {name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.variance_minimum > self.variance_maximum:
            raise ValueError("fusion variance bounds are reversed")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None,
    ) -> "U2FlowBidirectionalFusionPolicyV1 | None":
        if value is None:
            return None
        expected = {
            "enabled", "optimization_steps", "uncertainty_variance_threshold",
            "learning_rate", "learning_rate_decay", "variance_minimum",
            "variance_maximum", "random_seed",
        }
        if set(value) != expected:
            raise ValueError("U2Flow fusion policy fields drift")
        return cls(**dict(value))


class U2FlowTinyMotionModelV1(nn.Module):
    """Three-layer motion prior used by the public U2Flow fusion script."""

    def __init__(self, *, generator: torch.Generator | None = None) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=3, padding=1, bias=True),
        )
        for module in self.layers.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(-value)


@dataclass(frozen=True)
class U2FlowBidirectionalFusionResultV1:
    fused_flow: torch.Tensor
    forward_flow: torch.Tensor
    predicted_forward_from_backward: torch.Tensor
    forward_reliable: torch.Tensor | None
    backward_reliable: torch.Tensor | None
    joint_fit_mask: torch.Tensor | None
    replacement_mask: torch.Tensor
    fusion_applied: bool


def _validate_flow_and_uncertainty(
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    forward_log_variance: torch.Tensor,
    backward_log_variance: torch.Tensor,
) -> None:
    if (
        forward_flow.ndim != 4
        or forward_flow.shape[1] != 2
        or backward_flow.shape != forward_flow.shape
    ):
        raise ValueError("fusion flows must share shape N2HW")
    expected_uncertainty = forward_flow[:, :1].shape
    if (
        forward_log_variance.shape != expected_uncertainty
        or backward_log_variance.shape != expected_uncertainty
    ):
        raise ValueError("fusion log variances must share shape N1HW")
    values = (
        forward_flow, backward_flow,
        forward_log_variance, backward_log_variance,
    )
    if not all(value.is_floating_point() for value in values):
        raise ValueError("fusion inputs must be floating tensors")
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("fusion inputs must be finite")
    if any(value.device != forward_flow.device for value in values):
        raise ValueError("fusion inputs must share one device")


def _centered_coordinates(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    # Preserve the public implementation's (row, column) channel order.
    return torch.stack((grid_y, grid_x), dim=0).unsqueeze(0)


def u2flow_bidirectional_fusion_v1(
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    forward_log_variance: torch.Tensor,
    backward_log_variance: torch.Tensor,
    *,
    policy: U2FlowBidirectionalFusionPolicyV1 = U2FlowBidirectionalFusionPolicyV1(),
) -> U2FlowBidirectionalFusionResultV1:
    """Fuse detached forward/backward predictions without touching the provider."""

    if not isinstance(policy, U2FlowBidirectionalFusionPolicyV1):
        raise TypeError("fusion policy must be typed")
    _validate_flow_and_uncertainty(
        forward_flow, backward_flow,
        forward_log_variance, backward_log_variance,
    )
    detached_forward = forward_flow.detach()
    detached_backward = backward_flow.detach()
    empty_replacement = torch.zeros_like(forward_log_variance, dtype=torch.bool)
    if not policy.enabled:
        return U2FlowBidirectionalFusionResultV1(
            fused_flow=detached_forward,
            forward_flow=detached_forward,
            predicted_forward_from_backward=detached_forward,
            forward_reliable=None,
            backward_reliable=None,
            joint_fit_mask=None,
            replacement_mask=empty_replacement,
            fusion_applied=False,
        )

    forward_variance = forward_log_variance.detach().float().exp().clamp(
        min=policy.variance_minimum, max=policy.variance_maximum,
    )
    backward_variance = backward_log_variance.detach().float().exp().clamp(
        min=policy.variance_minimum, max=policy.variance_maximum,
    )
    forward_reliable = forward_variance < policy.uncertainty_variance_threshold
    backward_reliable = backward_variance < policy.uncertainty_variance_threshold
    joint_fit = forward_reliable & backward_reliable
    predicted_samples: list[torch.Tensor] = []
    fused_samples: list[torch.Tensor] = []
    replacement_samples: list[torch.Tensor] = []

    original_dtype = detached_forward.dtype
    for sample_index in range(detached_forward.shape[0]):
        target = detached_forward[sample_index : sample_index + 1].float()
        backward = detached_backward[sample_index : sample_index + 1].float()
        fit_mask = joint_fit[sample_index : sample_index + 1]
        replacement = (
            ~forward_reliable[sample_index : sample_index + 1]
            & backward_reliable[sample_index : sample_index + 1]
        )
        if not bool(fit_mask.any()):
            predicted_samples.append(target.to(dtype=original_dtype))
            fused_samples.append(target.to(dtype=original_dtype))
            replacement_samples.append(torch.zeros_like(replacement))
            continue

        coordinates = _centered_coordinates(
            target.shape[-2], target.shape[-1],
            device=target.device, dtype=target.dtype,
        )
        motion_input = torch.cat((backward, coordinates), dim=1)
        # Construct on CPU with an isolated generator, then move to the flow
        # device.  Conv2d's own reset_parameters consumes the global CPU RNG,
        # so fork/restore it even though the final Xavier weights use our
        # explicit generator.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(policy.random_seed + sample_index)
        with torch.random.fork_rng(devices=[]):
            motion_model = U2FlowTinyMotionModelV1(generator=generator)
        motion_model = motion_model.to(target.device)
        with torch.enable_grad():
            optimizer = torch.optim.Adam(
                motion_model.parameters(), lr=policy.learning_rate,
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=policy.learning_rate_decay,
            )
            decay_interval = max(policy.optimization_steps // 20, 1)
            for step in range(policy.optimization_steps):
                optimizer.zero_grad(set_to_none=True)
                prediction = motion_model(motion_input)
                endpoint_error = torch.linalg.vector_norm(
                    prediction - target, dim=1, keepdim=True,
                )
                loss = endpoint_error[fit_mask].mean()
                loss.backward()
                optimizer.step()
                if (step + 1) % decay_interval == 0:
                    scheduler.step()
            motion_model.eval()
            with torch.no_grad():
                predicted = motion_model(motion_input)
        predicted = torch.nan_to_num(predicted, nan=0.0).to(dtype=original_dtype)
        selected = replacement.expand_as(target)
        fused = torch.where(selected, predicted, target.to(dtype=original_dtype))
        predicted_samples.append(predicted)
        fused_samples.append(fused)
        replacement_samples.append(replacement)

    return U2FlowBidirectionalFusionResultV1(
        fused_flow=torch.cat(fused_samples, dim=0),
        forward_flow=detached_forward,
        predicted_forward_from_backward=torch.cat(predicted_samples, dim=0),
        forward_reliable=forward_reliable,
        backward_reliable=backward_reliable,
        joint_fit_mask=joint_fit,
        replacement_mask=torch.cat(replacement_samples, dim=0),
        fusion_applied=True,
    )


def _last_output_tensor(output: Mapping[str, Any], key: str) -> torch.Tensor:
    values = output.get(key)
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError(f"flow provider output needs a nonempty {key} sequence")
    result = values[-1]
    if not isinstance(result, torch.Tensor):
        raise ValueError(f"flow provider {key} must contain tensors")
    return result


def fuse_u2flow_triplet_prediction_v1(
    model: nn.Module,
    triplet: torch.Tensor,
    forward_flow: torch.Tensor,
    forward_log_variance: torch.Tensor,
    *,
    matcher_iterations: int,
    policy: U2FlowBidirectionalFusionPolicyV1 = U2FlowBidirectionalFusionPolicyV1(),
    image_scale: float = 255.0,
) -> U2FlowBidirectionalFusionResultV1:
    """Use an existing current-to-next prediction and add the reverse pass."""

    if triplet.ndim != 5 or triplet.shape[1] != 3 or triplet.shape[2] != 3:
        raise ValueError("fusion triplet must have shape N3CHW with RGB channels")
    if triplet.shape[0] != forward_flow.shape[0] or triplet.shape[-2:] != forward_flow.shape[-2:]:
        raise ValueError("fusion triplet and forward prediction geometry differ")
    if isinstance(matcher_iterations, bool) or not isinstance(matcher_iterations, int) or matcher_iterations < 1:
        raise ValueError("matcher_iterations must be a positive integer")
    image_scale = float(image_scale)
    if not math.isfinite(image_scale) or image_scale <= 0.0:
        raise ValueError("image_scale must be finite and positive")
    if not policy.enabled:
        return u2flow_bidirectional_fusion_v1(
            forward_flow,
            torch.zeros_like(forward_flow),
            forward_log_variance,
            torch.zeros_like(forward_log_variance),
            policy=policy,
        )
    with torch.no_grad():
        backward_output = model(
            triplet[:, 1].float() * image_scale,
            triplet[:, 0].float() * image_scale,
            iters=matcher_iterations,
            test_mode=True,
        )
    if not isinstance(backward_output, Mapping):
        raise TypeError("flow provider output must be a mapping")
    return u2flow_bidirectional_fusion_v1(
        forward_flow,
        _last_output_tensor(backward_output, "flow"),
        forward_log_variance,
        _last_output_tensor(backward_output, "uncertainty_log_variance"),
        policy=policy,
    )


def predict_u2flow_triplet_fusion_v1(
    model: nn.Module,
    triplet: torch.Tensor,
    *,
    matcher_iterations: int,
    policy: U2FlowBidirectionalFusionPolicyV1 = U2FlowBidirectionalFusionPolicyV1(),
    image_scale: float = 255.0,
) -> U2FlowBidirectionalFusionResultV1:
    """Run current-to-next and, only when enabled, current-to-previous fusion."""

    if triplet.ndim != 5 or triplet.shape[1:3] != (3, 3):
        raise ValueError("fusion triplet must have shape N3CHW with RGB channels")
    with torch.no_grad():
        forward_output = model(
            triplet[:, 1].float() * image_scale,
            triplet[:, 2].float() * image_scale,
            iters=matcher_iterations,
            test_mode=True,
        )
    if not isinstance(forward_output, Mapping):
        raise TypeError("flow provider output must be a mapping")
    return fuse_u2flow_triplet_prediction_v1(
        model,
        triplet,
        _last_output_tensor(forward_output, "flow"),
        _last_output_tensor(forward_output, "uncertainty_log_variance"),
        matcher_iterations=matcher_iterations,
        policy=policy,
        image_scale=image_scale,
    )


__all__ = [
    "U2FlowBidirectionalFusionPolicyV1",
    "U2FlowBidirectionalFusionResultV1",
    "U2FlowTinyMotionModelV1",
    "fuse_u2flow_triplet_prediction_v1",
    "predict_u2flow_triplet_fusion_v1",
    "u2flow_bidirectional_fusion_v1",
]
