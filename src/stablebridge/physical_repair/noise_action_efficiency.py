"""Support-aligned clipping and normalized exact-action response receipts.

The clipping null measures the same event as ``impulse_support``: an exact
black/white RGB value whose local 3x3 median is not the same extreme.  The
action response is normalized by the exact median intervention magnitude, so
its unitless lower bound measures how much of the action amplitude becomes
paired-agreement improvement.  Neither API accepts corruption labels, clean
counterparts, task truth, or outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import cv2
import numpy as np
from scipy.special import ndtr
from scipy.stats import t as student_t

from .noise_visibility_receipts import (
    BidirectionalVisibilityReceipt,
    bidirectional_visibility_receipt,
)
from .operators import impulse_support


def _mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=bool).tobytes()).hexdigest()


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


def _noise_sigma_rgb(image: np.ndarray) -> tuple[float, float, float]:
    source = image.astype(np.float32)
    output = []
    for channel in range(3):
        laplacian = cv2.Laplacian(
            source[..., channel], cv2.CV_32F, ksize=1,
            borderType=cv2.BORDER_REFLECT_101,
        )
        center = float(np.median(laplacian))
        sigma = float(
            np.median(np.abs(laplacian - center))
            / 0.6744897501960817 / math.sqrt(20.0)
        )
        output.append(max(sigma, 0.5))
    return tuple(output)  # type: ignore[return-value]


def _simultaneous_lower(values: np.ndarray, alpha: float,
                        hypotheses_tested: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return -1.0
    quantile = float(student_t.ppf(
        1.0 - alpha / (2.0 * hypotheses_tested), values.size - 1,
    ))
    return float(values.mean() - quantile * values.std(ddof=1) / math.sqrt(values.size))


@dataclass(frozen=True)
class SupportAlignedClippingNullReceipt:
    endpoint: str
    alpha: float
    hypotheses_tested: int
    valid_pixels: int
    black_opportunity_pixels: int
    white_opportunity_pixels: int
    observed_isolated_extrema: int
    expected_event_upper_mean: float
    simultaneous_count_upper: float
    excess_count_lower: float
    sigma_rgb_255: tuple[float, float, float]
    support_mask: np.ndarray
    support_hash: str
    supported: bool
    event_id: str = "exact_extreme_and_local_median_not_same_extreme_v1"
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if self.endpoint not in {"first", "second"}:
            raise ValueError("clipping null needs a concrete endpoint")
        if not 0.0 < self.alpha < 1.0 or self.hypotheses_tested < 1:
            raise ValueError("invalid clipping-null multiplicity")
        if not 0 <= self.observed_isolated_extrema <= self.valid_pixels:
            raise ValueError("invalid clipping-null count")
        if not all(np.isfinite((
            self.expected_event_upper_mean, self.simultaneous_count_upper,
            self.excess_count_lower, *self.sigma_rgb_255,
        ))):
            raise ValueError("clipping-null values must be finite")
        mask = np.ascontiguousarray(self.support_mask)
        if mask.dtype != np.bool_ or mask.ndim != 2:
            raise ValueError("clipping support must be boolean HxW")
        if int(mask.sum()) != self.observed_isolated_extrema:
            raise ValueError("clipping support count drift")
        if _mask_hash(mask) != self.support_hash:
            raise ValueError("clipping support hash mismatch")
        if self.supported != (self.excess_count_lower > 0.0):
            raise ValueError("clipping support decision drift")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("clipping null cannot read labels or outcomes")
        object.__setattr__(self, "support_mask", mask)


@dataclass(frozen=True)
class MedianActionEfficiencyReceipt:
    endpoint: str
    coordinate_frame: str
    tile_size: int
    alpha: float
    hypotheses_tested: int
    efficiency_floor: float
    visible_support_pixels: int
    fit_tiles: int
    check_tiles: int
    fit_efficiency_mean: float
    check_efficiency_mean: float
    fit_efficiency_lower: float
    check_efficiency_lower: float
    global_efficiency: float
    mean_action_magnitude_255: float
    mean_signed_gain_255: float
    changed_visible_support_mask: np.ndarray
    support_hash: str
    supported: bool
    visibility_proxy_applied: bool = True
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if self.endpoint not in {"first", "second"}:
            raise ValueError("action efficiency needs a concrete endpoint")
        if self.coordinate_frame != "first_flow_native":
            raise ValueError("action efficiency coordinate mismatch")
        if self.tile_size < 16 or not 0.0 < self.alpha < 1.0:
            raise ValueError("invalid efficiency confidence configuration")
        if self.hypotheses_tested < 1 or not 0.0 < self.efficiency_floor < 1.0:
            raise ValueError("invalid efficiency decision configuration")
        values = (
            self.fit_efficiency_mean, self.check_efficiency_mean,
            self.fit_efficiency_lower, self.check_efficiency_lower,
            self.global_efficiency, self.mean_action_magnitude_255,
            self.mean_signed_gain_255,
        )
        if not all(np.isfinite(values)):
            raise ValueError("action efficiency values must be finite")
        mask = np.ascontiguousarray(self.changed_visible_support_mask)
        if mask.dtype != np.bool_ or mask.ndim != 2:
            raise ValueError("action efficiency support must be boolean HxW")
        if int(mask.sum()) != self.visible_support_pixels:
            raise ValueError("action efficiency support count drift")
        if _mask_hash(mask) != self.support_hash:
            raise ValueError("action efficiency support hash mismatch")
        expected = bool(
            self.fit_tiles >= 2 and self.check_tiles >= 2
            and self.fit_efficiency_lower > self.efficiency_floor
            and self.check_efficiency_lower > self.efficiency_floor
        )
        if self.supported != expected:
            raise ValueError("action efficiency decision drift")
        if not self.visibility_proxy_applied:
            raise ValueError("visibility proxy must be applied")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("action efficiency cannot read labels or outcomes")
        object.__setattr__(self, "changed_visible_support_mask", mask)


def support_aligned_clipping_null_receipt(
    image: np.ndarray,
    *,
    endpoint: str,
    alpha: float = 0.05,
    hypotheses_tested: int = 8,
) -> SupportAlignedClippingNullReceipt:
    image = _image(image)
    if endpoint not in {"first", "second"}:
        raise ValueError("clipping null needs a concrete endpoint")
    height, width = image.shape[:2]
    valid = np.zeros((height, width), dtype=bool)
    valid[1:-1, 1:-1] = True
    support = (impulse_support(image) > 0.0) & valid
    median = cv2.medianBlur(image, 3).astype(np.float64)
    median_black = np.all(median <= 1.0, axis=2)
    median_white = np.all(median >= 254.0, axis=2)
    sigma = np.asarray(_noise_sigma_rgb(image), dtype=np.float64)
    black_marginal = ndtr((1.5 - median) / sigma.reshape(1, 1, 3))
    white_marginal = ndtr((median - 253.5) / sigma.reshape(1, 1, 3))
    # Match the observed event exactly: an extreme only counts when the local
    # median is not already the same extreme.
    probability_upper = np.minimum(
        np.min(black_marginal, axis=2) * (~median_black)
        + np.min(white_marginal, axis=2) * (~median_white),
        1.0,
    )
    expected = float(probability_upper[valid].sum())
    log_term = math.log(float(hypotheses_tested) / float(alpha))
    simultaneous_upper = min(
        float(valid.sum()),
        expected + math.sqrt(2.0 * expected * log_term) + 2.0 * log_term / 3.0,
    )
    observed = int(support.sum())
    return SupportAlignedClippingNullReceipt(
        endpoint=endpoint, alpha=float(alpha),
        hypotheses_tested=int(hypotheses_tested), valid_pixels=int(valid.sum()),
        black_opportunity_pixels=int((valid & ~median_black).sum()),
        white_opportunity_pixels=int((valid & ~median_white).sum()),
        observed_isolated_extrema=observed,
        expected_event_upper_mean=expected,
        simultaneous_count_upper=simultaneous_upper,
        excess_count_lower=float(observed - simultaneous_upper),
        sigma_rgb_255=tuple(float(value) for value in sigma),
        support_mask=np.ascontiguousarray(support),
        support_hash=_mask_hash(support),
        supported=observed > simultaneous_upper,
    )


def visibility_qualified_median_efficiency(
    first: np.ndarray,
    second: np.ndarray,
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    *,
    tile_size: int = 32,
    alpha: float = 0.05,
    hypotheses_tested: int = 8,
    efficiency_floor: float = 0.5,
    absolute_tolerance_px: float = 1.0,
    relative_tolerance: float = 0.05,
) -> tuple[BidirectionalVisibilityReceipt, dict[str, MedianActionEfficiencyReceipt]]:
    first = _image(first)
    second = _image(second)
    if first.shape != second.shape:
        raise ValueError("endpoint shapes differ")
    forward = _flow(forward_flow, first.shape[:2])
    backward = _flow(backward_flow, first.shape[:2])
    if tile_size < 16:
        raise ValueError("tile size must be at least 16")
    visibility = bidirectional_visibility_receipt(
        forward, backward, absolute_tolerance_px=absolute_tolerance_px,
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
    valid_all = valid & valid_repaired & valid_support & visibility.first_visible_mask
    receipts = {}
    for endpoint in ("first", "second"):
        if endpoint == "first":
            native = first.astype(np.float32)
            candidate = repaired_first.astype(np.float32)
            reference = warped_second.astype(np.float32)
            support = first_support & valid_all
        else:
            native = warped_second.astype(np.float32)
            candidate = warped_repaired_second.astype(np.float32)
            reference = first.astype(np.float32)
            support = (warped_second_support > 0.5) & valid_all
        before = np.mean(np.abs(native - reference), axis=2)
        after = np.mean(np.abs(candidate - reference), axis=2)
        gain = before - after
        magnitude = np.mean(np.abs(candidate - native), axis=2)
        folds: list[list[float]] = [[], []]
        height, width = support.shape
        for top in range(0, height - tile_size + 1, tile_size):
            for left in range(0, width - tile_size + 1, tile_size):
                ys = slice(top, top + tile_size)
                xs = slice(left, left + tile_size)
                selected = support[ys, xs]
                if not np.any(selected):
                    continue
                denominator = float(magnitude[ys, xs][selected].sum())
                if denominator <= 1e-12:
                    continue
                efficiency = float(gain[ys, xs][selected].sum() / denominator)
                folds[((top // tile_size) + (left // tile_size)) % 2].append(
                    float(np.clip(efficiency, -1.0, 1.0))
                )
        fit = np.asarray(folds[0], dtype=np.float64)
        check = np.asarray(folds[1], dtype=np.float64)
        fit_lower = _simultaneous_lower(fit, alpha, hypotheses_tested)
        check_lower = _simultaneous_lower(check, alpha, hypotheses_tested)
        pixels = int(support.sum())
        total_magnitude = float(magnitude[support].sum()) if pixels else 0.0
        receipts[endpoint] = MedianActionEfficiencyReceipt(
            endpoint=endpoint, coordinate_frame="first_flow_native",
            tile_size=tile_size, alpha=float(alpha),
            hypotheses_tested=int(hypotheses_tested),
            efficiency_floor=float(efficiency_floor),
            visible_support_pixels=pixels, fit_tiles=int(fit.size),
            check_tiles=int(check.size),
            fit_efficiency_mean=float(fit.mean()) if fit.size else 0.0,
            check_efficiency_mean=float(check.mean()) if check.size else 0.0,
            fit_efficiency_lower=fit_lower,
            check_efficiency_lower=check_lower,
            global_efficiency=float(
                gain[support].sum() / max(total_magnitude, 1e-12)
            ) if pixels else 0.0,
            mean_action_magnitude_255=float(magnitude[support].mean()) if pixels else 0.0,
            mean_signed_gain_255=float(gain[support].mean()) if pixels else 0.0,
            changed_visible_support_mask=np.ascontiguousarray(support),
            support_hash=_mask_hash(support),
            supported=bool(
                fit.size >= 2 and check.size >= 2
                and fit_lower > efficiency_floor and check_lower > efficiency_floor
            ),
        )
    return visibility, receipts

