"""Outcome-blind observable features for the SEA-RAFT U0 observer.

The channel order is the frozen Work-B schema.  This torch implementation
keeps feature construction on the GPU while the SEA-RAFT matcher and its
native mixture uncertainty remain detached.
"""
from __future__ import annotations

import math

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


FEATURE_CHANNELS_V1 = (
    "i1_r",
    "i1_g",
    "i1_b",
    "warped_i2_r",
    "warped_i2_g",
    "warped_i2_b",
    "flow_x_over_diagonal",
    "flow_y_over_diagonal",
    "existing_risk",
    "photometric_l1_rgb_mean",
    "flow_gradient_magnitude_over_diagonal",
    "runtime_warp_validity",
)


def _require_tensor(value, *, channels: int, name: str):
    if torch is None:
        raise ImportError("uncertainty feature construction requires torch")
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[1] != channels:
        raise ValueError(f"{name} must have shape N{channels}HW")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def warp_second_to_first_torch_v1(image2_01, flow_xy):
    """Sample ``image2`` at ``p + flow(p)`` and return a strict valid mask."""

    image = _require_tensor(image2_01, channels=3, name="second image")
    flow = _require_tensor(flow_xy, channels=2, name="native flow")
    if image.shape[0] != flow.shape[0] or image.shape[2:] != flow.shape[2:]:
        raise ValueError("image and flow must share N,H,W")
    n, _, height, width = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    sample_x = xx[None] + flow[:, 0]
    sample_y = yy[None] + flow[:, 1]
    valid = (
        (sample_x >= 0)
        & (sample_x <= width - 1)
        & (sample_y >= 0)
        & (sample_y <= height - 1)
    )
    if width > 1:
        grid_x = 2.0 * sample_x / float(width - 1) - 1.0
    else:  # pragma: no cover - real crops are wider than one pixel
        grid_x = torch.zeros_like(sample_x)
    if height > 1:
        grid_y = 2.0 * sample_y / float(height - 1) - 1.0
    else:  # pragma: no cover
        grid_y = torch.zeros_like(sample_y)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    warped = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    warped = torch.where(valid[:, None], warped, torch.zeros_like(warped))
    return warped, valid[:, None]


def _flow_gradient_magnitude(flow):
    dx = torch.zeros_like(flow)
    dy = torch.zeros_like(flow)
    if flow.shape[-1] > 1:
        dx[..., :-1] = flow[..., 1:] - flow[..., :-1]
        dx[..., -1] = dx[..., -2]
    if flow.shape[-2] > 1:
        dy[..., :-1, :] = flow[..., 1:, :] - flow[..., :-1, :]
        dy[..., -1, :] = dy[..., -2, :]
    return torch.sqrt(torch.sum(dx.square() + dy.square(), dim=1, keepdim=True))


def build_searaft_observable_features_v1(
    image1_01,
    image2_01,
    native_flow_xy,
    native_log_scale,
):
    """Build the frozen NCHW 12-channel observer input.

    Inputs are runtime-available only.  Ground truth, augmentation identity,
    action labels, and outcomes are deliberately absent.
    """

    first = _require_tensor(image1_01, channels=3, name="first image")
    second = _require_tensor(image2_01, channels=3, name="second image")
    flow = _require_tensor(native_flow_xy, channels=2, name="native flow")
    risk = _require_tensor(native_log_scale, channels=1, name="native log scale")
    shape = (first.shape[0], first.shape[2], first.shape[3])
    if any((value.shape[0], value.shape[2], value.shape[3]) != shape for value in (second, flow, risk)):
        raise ValueError("all uncertainty features must share N,H,W")
    if bool((first < 0).any()) or bool((first > 1).any()) or bool((second < 0).any()) or bool((second > 1).any()):
        raise ValueError("observer RGB inputs must lie in [0,1]")
    warped, warp_valid = warp_second_to_first_torch_v1(second, flow)
    height, width = first.shape[-2:]
    diagonal = math.hypot(height, width)
    photo = torch.mean(torch.abs(first - warped), dim=1, keepdim=True)
    photo = torch.where(warp_valid, photo, torch.zeros_like(photo))
    gradient = _flow_gradient_magnitude(flow) / diagonal
    features = torch.cat(
        (
            first,
            warped,
            flow / diagonal,
            risk,
            photo,
            gradient,
            warp_valid.to(dtype=first.dtype),
        ),
        dim=1,
    ).detach()
    if features.shape[1] != len(FEATURE_CHANNELS_V1) or not bool(torch.isfinite(features).all()):
        raise RuntimeError("SEA-RAFT uncertainty feature invariant failed")
    return features, warp_valid


__all__ = [
    "FEATURE_CHANNELS_V1",
    "build_searaft_observable_features_v1",
    "warp_second_to_first_torch_v1",
]
