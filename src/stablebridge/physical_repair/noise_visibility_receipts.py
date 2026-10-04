"""Bidirectional-geometry receipts for sparse median interventions.

These receipts qualify an exact median action response with an observable
forward/backward consistency mask.  The mask is a visibility proxy, not
ground-truth occlusion.  No corruption identity, task truth, or matcher outcome
is accepted by the API.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import cv2
import numpy as np

from .operators import impulse_support


def _mask_hash(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(mask, dtype=bool)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _flow_hash(flow: np.ndarray) -> str:
    value = np.ascontiguousarray(flow, dtype=np.float32)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _image(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.uint8 or result.ndim != 3 or result.shape[2] != 3:
        raise ValueError("expected uint8 RGB")
    return np.ascontiguousarray(result)


def _flow(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (*shape, 2) or not np.isfinite(result).all():
        raise ValueError("expected finite HxWx2 flow")
    return np.ascontiguousarray(result)


def _warp(value: np.ndarray, flow: np.ndarray, interpolation: int
          ) -> tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx + flow[..., 0]
    map_y = yy + flow[..., 1]
    valid = ((map_x >= 1.0) & (map_x <= width - 2.0)
             & (map_y >= 1.0) & (map_y <= height - 2.0))
    warped = cv2.remap(
        np.asarray(value), map_x, map_y, interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    return np.ascontiguousarray(warped), np.ascontiguousarray(valid)


def _lower_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    mean = float(values.mean())
    if values.size == 1:
        return mean
    return mean - float(values.std(ddof=1) / math.sqrt(values.size))


@dataclass(frozen=True)
class BidirectionalVisibilityReceipt:
    method: str
    absolute_tolerance_px: float
    relative_tolerance: float
    forward_flow_hash: str
    backward_flow_hash: str
    first_valid_pixels: int
    first_visible_pixels: int
    first_visible_fraction: float
    first_fb_error_p50_px: float
    first_fb_error_p95_px: float
    first_visible_mask: np.ndarray
    first_visible_hash: str
    second_valid_pixels: int
    second_visible_pixels: int
    second_visible_fraction: float
    second_fb_error_p50_px: float
    second_fb_error_p95_px: float
    second_visible_mask: np.ndarray
    second_visible_hash: str
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if self.method != "forward_backward_consistency_v1":
            raise ValueError("unknown visibility method")
        if self.absolute_tolerance_px <= 0.0 or self.relative_tolerance < 0.0:
            raise ValueError("invalid visibility tolerance")
        for endpoint in ("first", "second"):
            mask = np.ascontiguousarray(getattr(self, f"{endpoint}_visible_mask"))
            if mask.dtype != np.bool_ or mask.ndim != 2:
                raise ValueError("visibility masks must be boolean HxW")
            if _mask_hash(mask) != getattr(self, f"{endpoint}_visible_hash"):
                raise ValueError("visibility support hash mismatch")
            valid = int(getattr(self, f"{endpoint}_valid_pixels"))
            visible = int(getattr(self, f"{endpoint}_visible_pixels"))
            fraction = float(getattr(self, f"{endpoint}_visible_fraction"))
            if valid < 1 or not 0 <= visible <= valid:
                raise ValueError("invalid visibility pixel accounting")
            if not np.isclose(fraction, visible / valid, rtol=0.0, atol=1e-15):
                raise ValueError("visibility fraction drift")
            if int(mask.sum()) != visible:
                raise ValueError("visibility mask count drift")
            if not all(np.isfinite((
                getattr(self, f"{endpoint}_fb_error_p50_px"),
                getattr(self, f"{endpoint}_fb_error_p95_px"),
            ))):
                raise ValueError("visibility error quantiles must be finite")
            object.__setattr__(self, f"{endpoint}_visible_mask", mask)
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("visibility receipt cannot read labels or outcomes")


@dataclass(frozen=True)
class VisibilityQualifiedMedianResponseReceipt:
    endpoint: str
    coordinate_frame: str
    tile_size: int
    pre_visibility_support_pixels: int
    visible_support_pixels: int
    visible_support_fraction: float
    fit_tiles: int
    check_tiles: int
    fit_gain_lower_255: float
    check_gain_lower_255: float
    mean_gain_255: float
    reference_structure_fraction: float
    changed_visible_support_mask: np.ndarray
    support_hash: str
    response_supported: bool
    visibility_proxy_applied: bool = True
    true_occlusion_ground_truth_available: bool = False
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if self.endpoint not in {"first", "second"}:
            raise ValueError("median response needs a concrete endpoint")
        if self.coordinate_frame != "first_flow_native":
            raise ValueError("visibility-qualified response coordinate mismatch")
        if self.tile_size < 16:
            raise ValueError("tile size must be at least 16")
        if not 0 <= self.visible_support_pixels <= self.pre_visibility_support_pixels:
            raise ValueError("invalid visible support accounting")
        expected_fraction = self.visible_support_pixels / max(
            self.pre_visibility_support_pixels, 1
        )
        if not np.isclose(
            self.visible_support_fraction, expected_fraction, rtol=0.0, atol=1e-15
        ):
            raise ValueError("visible support fraction drift")
        values = (
            self.fit_gain_lower_255, self.check_gain_lower_255,
            self.mean_gain_255, self.reference_structure_fraction,
        )
        if not all(np.isfinite(values)):
            raise ValueError("median response values must be finite")
        if not 0.0 <= self.reference_structure_fraction <= 1.0:
            raise ValueError("reference structure fraction must lie in [0,1]")
        mask = np.ascontiguousarray(self.changed_visible_support_mask)
        if mask.dtype != np.bool_ or mask.ndim != 2:
            raise ValueError("median response support must be boolean HxW")
        if int(mask.sum()) != self.visible_support_pixels:
            raise ValueError("median response support count drift")
        if _mask_hash(mask) != self.support_hash:
            raise ValueError("median response support hash mismatch")
        expected = bool(
            self.fit_tiles > 0 and self.check_tiles > 0
            and self.fit_gain_lower_255 > 0.0
            and self.check_gain_lower_255 > 0.0
        )
        if self.response_supported != expected:
            raise ValueError("median response decision drift")
        if not self.visibility_proxy_applied or self.true_occlusion_ground_truth_available:
            raise ValueError("visibility semantics drift")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("median response cannot read labels or outcomes")
        object.__setattr__(self, "changed_visible_support_mask", mask)


def _one_direction_visibility(
    flow: np.ndarray,
    reverse: np.ndarray,
    absolute_tolerance_px: float,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    warped_reverse, valid = _warp(reverse, flow, cv2.INTER_LINEAR)
    residual = np.linalg.norm(flow + warped_reverse, axis=2)
    threshold = absolute_tolerance_px + relative_tolerance * np.linalg.norm(flow, axis=2)
    visible = valid & np.isfinite(residual) & (residual <= threshold)
    return np.ascontiguousarray(visible), np.ascontiguousarray(valid), residual


def bidirectional_visibility_receipt(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    *,
    absolute_tolerance_px: float = 1.0,
    relative_tolerance: float = 0.05,
) -> BidirectionalVisibilityReceipt:
    forward = np.asarray(forward_flow, dtype=np.float32)
    if forward.ndim != 3 or forward.shape[2] != 2:
        raise ValueError("expected finite HxWx2 flow")
    forward = _flow(forward, forward.shape[:2])
    backward = _flow(backward_flow, forward.shape[:2])
    if absolute_tolerance_px <= 0.0 or relative_tolerance < 0.0:
        raise ValueError("invalid visibility tolerance")
    first, first_valid, first_error = _one_direction_visibility(
        forward, backward, absolute_tolerance_px, relative_tolerance,
    )
    second, second_valid, second_error = _one_direction_visibility(
        backward, forward, absolute_tolerance_px, relative_tolerance,
    )

    def quantile(error: np.ndarray, valid: np.ndarray, value: float) -> float:
        selected = error[valid]
        return float(np.quantile(selected, value)) if selected.size else 0.0

    return BidirectionalVisibilityReceipt(
        method="forward_backward_consistency_v1",
        absolute_tolerance_px=float(absolute_tolerance_px),
        relative_tolerance=float(relative_tolerance),
        forward_flow_hash=_flow_hash(forward),
        backward_flow_hash=_flow_hash(backward),
        first_valid_pixels=int(first_valid.sum()),
        first_visible_pixels=int(first.sum()),
        first_visible_fraction=float(first.sum() / max(first_valid.sum(), 1)),
        first_fb_error_p50_px=quantile(first_error, first_valid, 0.5),
        first_fb_error_p95_px=quantile(first_error, first_valid, 0.95),
        first_visible_mask=first,
        first_visible_hash=_mask_hash(first),
        second_valid_pixels=int(second_valid.sum()),
        second_visible_pixels=int(second.sum()),
        second_visible_fraction=float(second.sum() / max(second_valid.sum(), 1)),
        second_fb_error_p50_px=quantile(second_error, second_valid, 0.5),
        second_fb_error_p95_px=quantile(second_error, second_valid, 0.95),
        second_visible_mask=second,
        second_visible_hash=_mask_hash(second),
    )


def dis_bidirectional_flow(
    first: np.ndarray,
    second: np.ndarray,
    *,
    preset: str = "ultrafast",
    spatial_propagation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a fixed, outcome-free CPU alignment proxy in both directions."""
    first = _image(first)
    second = _image(second)
    if first.shape != second.shape:
        raise ValueError("endpoint shapes differ")
    presets = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }
    if preset not in presets:
        raise ValueError("unknown DIS preset")
    first_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)

    def compute(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        estimator = cv2.DISOpticalFlow_create(presets[preset])
        estimator.setUseSpatialPropagation(bool(spatial_propagation))
        return np.ascontiguousarray(estimator.calc(source, target, None), dtype=np.float32)

    return compute(first_gray, second_gray), compute(second_gray, first_gray)


def visibility_qualified_median_responses(
    first: np.ndarray,
    second: np.ndarray,
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    *,
    tile_size: int = 32,
    absolute_tolerance_px: float = 1.0,
    relative_tolerance: float = 0.05,
) -> tuple[BidirectionalVisibilityReceipt, dict[str, VisibilityQualifiedMedianResponseReceipt]]:
    first = _image(first)
    second = _image(second)
    if first.shape != second.shape:
        raise ValueError("endpoint shapes differ")
    forward = _flow(forward_flow, first.shape[:2])
    backward = _flow(backward_flow, first.shape[:2])
    if tile_size < 16:
        raise ValueError("tile size must be at least 16")
    visibility = bidirectional_visibility_receipt(
        forward, backward,
        absolute_tolerance_px=absolute_tolerance_px,
        relative_tolerance=relative_tolerance,
    )
    first_support = impulse_support(first) > 0.0
    second_support = impulse_support(second) > 0.0
    first_median = cv2.medianBlur(first, 3)
    second_median = cv2.medianBlur(second, 3)
    repaired_first = first.copy()
    repaired_second = second.copy()
    repaired_first[first_support] = first_median[first_support]
    repaired_second[second_support] = second_median[second_support]
    warped_second, valid = _warp(second.astype(np.float32), forward, cv2.INTER_LINEAR)
    warped_repaired_second, valid_repaired = _warp(
        repaired_second.astype(np.float32), forward, cv2.INTER_LINEAR,
    )
    warped_second_support, valid_support = _warp(
        second_support.astype(np.float32), forward, cv2.INTER_NEAREST,
    )
    warped_second_median, valid_median = _warp(
        second_median.astype(np.float32), forward, cv2.INTER_LINEAR,
    )
    valid_all = valid & valid_repaired & valid_support & valid_median
    responses = {}
    for endpoint in ("first", "second"):
        if endpoint == "first":
            native = first.astype(np.float32)
            candidate = repaired_first.astype(np.float32)
            reference = warped_second.astype(np.float32)
            pre_support = first_support & valid_all
            reference_delta = np.mean(
                np.abs(warped_second_median - warped_second.astype(np.float32)), axis=2,
            )
        else:
            native = warped_second.astype(np.float32)
            candidate = warped_repaired_second.astype(np.float32)
            reference = first.astype(np.float32)
            pre_support = (warped_second_support > 0.5) & valid_all
            reference_delta = np.mean(
                np.abs(first_median.astype(np.float32) - first.astype(np.float32)), axis=2,
            )
        support = pre_support & visibility.first_visible_mask
        before = np.mean(np.abs(native - reference), axis=2)
        after = np.mean(np.abs(candidate - reference), axis=2)
        signed_response = before - after
        endpoint_delta = np.mean(np.abs(candidate - native), axis=2)
        folds: list[list[float]] = [[], []]
        height, width = support.shape
        for top in range(0, height - tile_size + 1, tile_size):
            for left in range(0, width - tile_size + 1, tile_size):
                ys = slice(top, top + tile_size)
                xs = slice(left, left + tile_size)
                selected = support[ys, xs]
                if not np.any(selected):
                    continue
                fold = ((top // tile_size) + (left // tile_size)) % 2
                folds[fold].append(float(signed_response[ys, xs][selected].mean()))
        fit = np.asarray(folds[0], dtype=np.float64)
        check = np.asarray(folds[1], dtype=np.float64)
        visible_pixels = int(support.sum())
        pre_pixels = int(pre_support.sum())
        structure = int(np.sum(
            support & (reference_delta >= np.maximum(endpoint_delta, 1.0))
        ))
        fit_lower = _lower_mean(fit)
        check_lower = _lower_mean(check)
        responses[endpoint] = VisibilityQualifiedMedianResponseReceipt(
            endpoint=endpoint, coordinate_frame="first_flow_native",
            tile_size=tile_size, pre_visibility_support_pixels=pre_pixels,
            visible_support_pixels=visible_pixels,
            visible_support_fraction=float(visible_pixels / max(pre_pixels, 1)),
            fit_tiles=int(fit.size), check_tiles=int(check.size),
            fit_gain_lower_255=fit_lower,
            check_gain_lower_255=check_lower,
            mean_gain_255=float(signed_response[support].mean()) if visible_pixels else 0.0,
            reference_structure_fraction=float(structure / max(visible_pixels, 1)),
            changed_visible_support_mask=np.ascontiguousarray(support),
            support_hash=_mask_hash(support),
            response_supported=bool(
                fit.size > 0 and check.size > 0
                and fit_lower > 0.0 and check_lower > 0.0
            ),
        )
    return visibility, responses
