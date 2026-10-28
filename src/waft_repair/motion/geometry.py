"""Coordinates for same-frame feature grids and cross-frame evidence reads.

Coordinates ending in ``xy`` are in the sampled tensor's index units, unless
explicitly named physical. All sampling uses align_corners=True. A feature
lattice is defined by physical_position = origin + stride * index; tensor
shape alone never defines alignment.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def dinov3_grid_hw(padded_hw):
    """Common max-side640/no-upsample patch16 resize policy, using exact integers.

    Rounding down to the patch lattice introduces a small, explicitly mapped
    per-axis scale difference. No crop or hidden classification transform is used.
    """
    if (not isinstance(padded_hw, (tuple, list)) or len(padded_hw) != 2
            or any(type(x) is not int or x <= 0 for x in padded_hw)):
        raise ValueError('Positive integer padded H,W required')
    hp, wp = padded_hw
    longest = max(hp, wp)
    if longest <= 640:
        return max(1, hp // 16), max(1, wp // 16)
    return max(1, hp * 640 // (longest * 16)), max(1, wp * 640 // (longest * 16))


def _xy(value):
    return (float(value), float(value)) if isinstance(value, (int, float)) else tuple(float(x) for x in value)


def pixel_grid(hw, *, device=None, dtype=torch.float32):
    """Return H,W,2 in x,y order."""
    yy, xx = torch.meshgrid(torch.arange(hw[0], device=device, dtype=dtype),
                            torch.arange(hw[1], device=device, dtype=dtype), indexing="ij")
    return torch.stack((xx, yy), -1)


def grid_to_physical(xy: Tensor, stride, origin=0.0):
    return xy * xy.new_tensor(_xy(stride)) + xy.new_tensor(_xy(origin))


def physical_to_grid(xy: Tensor, stride, origin=0.0):
    return (xy - xy.new_tensor(_xy(origin))) / xy.new_tensor(_xy(stride))


def normalized_grid(xy: Tensor, source_hw):
    """Normalize indices for align_corners=True.

    A singleton axis has no invertible normalized coordinate map, so this
    low-level helper places it at zero. ``sample_at_xy`` first embeds such
    axes in an explicit zero halo; this helper alone does not define their
    out-of-bounds sampling behavior.
    """
    h, w = source_hw
    x = 2 * xy[..., 0] / (w - 1) - 1 if w > 1 else torch.zeros_like(xy[..., 0])
    y = 2 * xy[..., 1] / (h - 1) - 1 if h > 1 else torch.zeros_like(xy[..., 1])
    return torch.stack((x, y), -1)


def sample_at_xy(x: Tensor, xy: Tensor):
    """Sample B,C,H,W at B,...,2 indices, returning B,C,... .

    This function reads features, not GT labels. The caller must separately
    check native/corner support; zeros outside the tensor are not evidence.
    """
    if x.ndim != 4 or xy.ndim < 3 or xy.shape[0] != x.shape[0] or xy.shape[-1] != 2:
        raise ValueError("Expected B,C,H,W and B,...,2 coordinates")
    shape = xy.shape[1:-1]
    # align_corners=True collapses every normalized coordinate to index zero
    # on a singleton axis, even coordinates far outside that axis. An explicit
    # one-cell zero halo restores ordinary zero-extended bilinear sampling:
    # index0 returns the value, +/-0.5 returns half, and |index|>=1 returns zero.
    # Non-singleton grids take the original, byte-identical sampling path.
    singleton_h, singleton_w = x.shape[-2] == 1, x.shape[-1] == 1
    if singleton_h or singleton_w:
        x = F.pad(x, (int(singleton_w), int(singleton_w),
                      int(singleton_h), int(singleton_h)))
        xy = xy + xy.new_tensor((int(singleton_w), int(singleton_h)))
    grid = normalized_grid(xy, x.shape[-2:]).reshape(x.shape[0], -1, 1, 2)
    result = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return result.reshape(x.shape[0], x.shape[1], *shape)


def sampling_grid(source_hw, source_stride, source_origin, target_hw, target_stride,
                  target_origin, *, device=None, dtype=torch.float32):
    physical = grid_to_physical(pixel_grid(target_hw, device=device, dtype=dtype),
                                target_stride, target_origin)
    return physical_to_grid(physical, source_stride, source_origin)


def resample_grid(x: Tensor, source_stride, source_origin, target_stride, target_origin, target_hw):
    """Resample a feature field between declared physical lattices."""
    xy = sampling_grid(x.shape[-2:], source_stride, source_origin, target_hw,
                       target_stride, target_origin, device=x.device, dtype=x.dtype)
    return sample_at_xy(x, xy[None].expand(x.shape[0], -1, -1, -1))


def native_support(native_hw, padding, grid_hw, stride, origin=0.0, *, device=None):
    """Return 1,1,H,W bool indicating lattice centers inside the native image.

    This is center support, not a claim that a receptive field excludes padding.
    Sampling this mask and requiring one additionally checks all used corners.
    """
    if len(padding) != 4 or min(padding) < 0 or min(native_hw) <= 0:
        raise ValueError("Expected native H,W and nonnegative left,right,top,bottom padding")
    left, _, top, _ = padding
    points = grid_to_physical(pixel_grid(grid_hw, device=device), stride, origin)
    x, y = points[..., 0] - left, points[..., 1] - top
    return ((x >= 0) & (x <= native_hw[1] - 1) & (y >= 0) & (y <= native_hw[0] - 1))[None, None]


def sample_support(mask: Tensor, xy: Tensor):
    """Require tensor bounds AND complete bilinear native-center support."""
    h, w = mask.shape[-2:]
    in_bounds = ((xy[..., 0] >= 0) & (xy[..., 0] <= w - 1)
                 & (xy[..., 1] >= 0) & (xy[..., 1] <= h - 1))
    sampled = sample_at_xy(mask.to(dtype=xy.dtype).expand(xy.shape[0], -1, -1, -1), xy)[:, 0]
    return in_bounds & (sampled >= 1 - 1e-6)


def nearest_parent_hypotheses(target_xy_s4: Tensor, weights_s4: Tensor, output_hw):
    """Propagate candidate SETS to S2 without interpolating candidate ranks.

    Each S2 query takes its nearest S4 parent's hypotheses, then translates
    each endpoint by the source subcell displacement. This is a local motion
    proposal, not an assertion that motion is constant across a boundary.
    """
    b, h4, w4, k, _ = target_xy_s4.shape
    query2 = pixel_grid(output_hw, device=target_xy_s4.device, dtype=target_xy_s4.dtype).reshape(-1, 2)
    parent4 = torch.floor(query2 / 2 + .5).long()
    parent4[:, 0].clamp_(0, w4 - 1)
    parent4[:, 1].clamp_(0, h4 - 1)
    index = parent4[:, 1] * w4 + parent4[:, 0]
    endpoints = target_xy_s4.reshape(b, -1, k, 2)[:, index] * 2
    endpoints = endpoints + (query2 - parent4.to(query2.dtype) * 2)[None, :, None]
    weights = weights_s4.reshape(b, -1, k)[:, index]
    return endpoints, weights, query2
