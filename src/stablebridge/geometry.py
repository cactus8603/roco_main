"""Native-pixel geometry shared by flow, stereo, and association matching.

Coordinates are [..., 2] in (x, y) order, at integer pixel centres. Sampling
returns an explicit validity mask; zeros from padding are never observations.
No function in this module reads labels or assumes a correspondence is true.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


def pixel_grid(height: int, width: int, origin_xy=(0, 0), dtype=np.float32):
    """Return [H,W,2] native pixel centres, with an optional crop origin."""
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive")
    y, x = np.mgrid[:height, :width]
    return np.stack((x + origin_xy[0], y + origin_xy[1]), axis=-1).astype(dtype)


@dataclass(frozen=True)
class PixelTransform:
    """Crop, resize (half-pixel convention), then left/top padding.

    ``scale_xy`` is resized_pixels / original_pixels for each axis. ``pad_xy``
    is measured in resized pixels. Stride describes a regular patch grid whose
    first token centre is (stride-1)/2 in the padded local image.
    """

    origin_xy: tuple = (0.0, 0.0)
    scale_xy: tuple = (1.0, 1.0)
    pad_xy: tuple = (0.0, 0.0)
    native_hw: Optional[tuple] = None

    def __post_init__(self):
        for name in ("origin_xy", "scale_xy", "pad_xy"):
            value = getattr(self, name)
            if len(value) != 2 or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain two finite values")
        if min(self.scale_xy) <= 0:
            raise ValueError("Resize scales must be positive")
        if self.native_hw is not None and (len(self.native_hw) != 2 or min(self.native_hw) < 1):
            raise ValueError("native_hw must contain positive height and width")

    def native_to_local(self, xy):
        xy = np.asarray(xy)
        return (xy - self.origin_xy + .5) * self.scale_xy - .5 + self.pad_xy

    def local_to_native(self, xy):
        xy = np.asarray(xy)
        return (xy - self.pad_xy + .5) / self.scale_xy - .5 + self.origin_xy

    def native_to_patch(self, xy, stride=16):
        sx, sy = _stride_xy(stride)
        return (self.native_to_local(xy) + .5) / (sx, sy) - .5

    def patch_to_native(self, xy, stride=16):
        sx, sy = _stride_xy(stride)
        return self.local_to_native((np.asarray(xy) + .5) * (sx, sy) - .5)

    def native_valid(self, xy):
        if self.native_hw is None:
            raise ValueError("native_hw is required to determine observation validity")
        relative = np.asarray(xy) - self.origin_xy
        return in_bounds(relative, self.native_hw)

    def as_dict(self):
        return {"origin_xy": list(self.origin_xy), "scale_xy": list(self.scale_xy),
                "pad_xy": list(self.pad_xy),
                "native_hw": None if self.native_hw is None else list(self.native_hw),
                "coordinate_convention": "zero_based_pixel_centres_half_pixel_resize"}


def _stride_xy(stride):
    values = (stride, stride) if np.isscalar(stride) else tuple(stride)
    if len(values) != 2 or not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("stride must be positive scalar or (x,y)")
    return values


def in_bounds(xy, hw):
    """Strict observed-centre support, inclusive of first and last centre."""
    xy = np.asarray(xy)
    if xy.shape[-1] != 2:
        raise ValueError("Coordinates must end in (x,y)")
    h, w = hw
    return (np.isfinite(xy).all(axis=-1) & (xy[..., 0] >= 0) &
            (xy[..., 0] <= w - 1) & (xy[..., 1] >= 0) & (xy[..., 1] <= h - 1))


def native_displacement_from_local(source_xy, target_xy, source_transform, target_transform):
    """Map BOTH endpoints before subtracting (different crops/scales allowed)."""
    return target_transform.local_to_native(target_xy) - source_transform.local_to_native(source_xy)


def sample_numpy(values, xy, valid_mask=None):
    """Bilinear sample CxHxW or HxW at [...,2] local pixel centres.

    Returns (samples, valid), with samples [C,...] (or [...] for scalar input).
    Every nonzero-weight contributing corner must be observed and finite.
    Invalid samples are zeroed; callers MUST consult the returned mask.
    """
    values = np.asarray(values)
    scalar = values.ndim == 2
    if scalar:
        values = values[None]
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("values must be CxHxW or HxW")
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[-1] != 2:
        raise ValueError("Coordinates must end in (x,y)")
    _, h, w = values.shape
    valid = in_bounds(xy, (h, w))
    observation = np.isfinite(values).all(axis=0)
    if valid_mask is not None:
        if np.shape(valid_mask) != (h, w):
            raise ValueError("valid_mask must match HxW")
        observation &= np.asarray(valid_mask, dtype=bool)
    safe = np.where(np.isfinite(xy), xy, 0)
    x = np.clip(safe[..., 0], 0, w - 1)
    y = np.clip(safe[..., 1], 0, h - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    output = np.zeros((values.shape[0],) + xy.shape[:-1], dtype=np.result_type(values.dtype, np.float32))
    safe_values = np.where(observation[None], values, 0)
    for yy, xx, weight in ((y0, x0, (1-fx)*(1-fy)), (y0, x1, fx*(1-fy)),
                           (y1, x0, (1-fx)*fy), (y1, x1, fx*fy)):
        output += safe_values[:, yy, xx] * weight
        valid &= observation[yy, xx] | (weight == 0)
    output = np.where(valid[None], output, 0)
    return (output[0] if scalar else output), valid


def sample_torch(values, xy, valid_mask=None):
    """Torch equivalent for BxCxHxW values and Bx...x2 coordinates.

    Output is BxCx... plus Bx... boolean validity. Uses align_corners=False
    and validates true pixel-centre support independently of grid_sample.
    """
    import torch
    import torch.nn.functional as functional

    if values.ndim != 4 or xy.ndim < 3 or xy.shape[0] != values.shape[0] or xy.shape[-1] != 2:
        raise ValueError("Expected BxCxHxW values and Bx...x2 coordinates")
    if not values.is_floating_point():
        raise ValueError("Torch values must be floating point")
    b, c, h, w = values.shape
    shape = tuple(xy.shape[1:-1])
    xy = xy.to(device=values.device, dtype=values.dtype)
    valid = (torch.isfinite(xy).all(dim=-1) & (xy[..., 0] >= 0) & (xy[..., 0] <= w-1) &
             (xy[..., 1] >= 0) & (xy[..., 1] <= h-1))
    observed = torch.isfinite(values).all(dim=1)
    if valid_mask is not None:
        if tuple(valid_mask.shape) != (b, h, w):
            raise ValueError("valid_mask must be BxHxW")
        observed &= valid_mask.to(device=values.device, dtype=torch.bool)
    safe_xy = torch.where(torch.isfinite(xy), xy, torch.zeros_like(xy))
    grid = (safe_xy + .5) * safe_xy.new_tensor((2/w, 2/h)) - 1
    grid = grid.reshape(b, -1, 1, 2)
    safe_values = torch.where(observed[:, None], values, torch.zeros_like(values))
    sampled = functional.grid_sample(safe_values, grid, mode="bilinear", padding_mode="zeros",
                                     align_corners=False).reshape((b, c) + shape)
    # Exact corner logic avoids treating a small contribution from a padded
    # token as observed because of a tolerance on a interpolated validity mask.
    x, y = safe_xy[..., 0].clamp(0, w-1), safe_xy[..., 1].clamp(0, h-1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0+1).clamp(max=w-1), (y0+1).clamp(max=h-1)
    fx, fy = x-x0, y-y0
    batch = torch.arange(b, device=values.device).reshape((b,) + (1,)*len(shape))
    for yy, xx, weight in ((y0,x0,(1-fx)*(1-fy)), (y0,x1,fx*(1-fy)),
                           (y1,x0,(1-fx)*fy), (y1,x1,fx*fy)):
        valid &= observed[batch, yy, xx] | (weight == 0)
    sampled = torch.where(valid[:, None], sampled, torch.zeros_like(sampled))
    return sampled, valid
