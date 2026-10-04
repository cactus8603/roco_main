"""Measurement-derived laws for competing noise repair actions.

The receipts in this module are evidence generators, not a task selector.  In
particular they keep three questions separate:

* whether isolated exact extrema exceed a conservative diffuse-clipping null;
* whether paired residual variance is closer to constant, mean-proportional,
  or mean-squared, and whether Anscombe/log coordinates flatten it; and
* whether the exact sparse median intervention improves paired agreement on
  held-out spatial tiles.

No corruption label or task outcome is accepted.  Visibility remains explicit:
one forward flow establishes coordinate validity and fold orientation, but it
cannot prove occlusion-free correspondence by itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import cv2
import numpy as np
from scipy.optimize import nnls
from scipy.special import ndtr
from scipy.stats import spearmanr

from .operators import impulse_support


RAW_VARIANCE_MODELS = ("constant", "mean", "mean_squared")


def _mask_hash(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(mask, dtype=bool)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_image(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("expected uint8 RGB")
    return np.ascontiguousarray(value)


@dataclass(frozen=True)
class DiffuseClippingNullReceipt:
    """Conservative all-channel clipping count under a diffuse Gaussian null."""

    endpoint: str
    coordinate_frame: str
    alpha: float
    hypotheses_tested: int
    valid_pixels: int
    observed_isolated_extrema: int
    expected_extrema_upper_mean: float
    simultaneous_count_upper: float
    excess_count_lower: float
    standardized_excess_lower: float
    sigma_rgb_255: tuple[float, float, float]
    support_mask: np.ndarray
    support_hash: str
    supported: bool
    null_id: str = "diffuse_gaussian_clipping_frechet_upper_v1"
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if self.endpoint not in {"first", "second"}:
            raise ValueError("clipping null needs a concrete endpoint")
        if self.coordinate_frame != f"{self.endpoint}_native":
            raise ValueError("clipping-null coordinate frame mismatch")
        if not 0.0 < float(self.alpha) < 1.0 or self.hypotheses_tested < 1:
            raise ValueError("invalid simultaneous clipping-null level")
        if self.valid_pixels < 1 or not 0 <= self.observed_isolated_extrema <= self.valid_pixels:
            raise ValueError("invalid clipping-null count")
        numeric = (
            self.expected_extrema_upper_mean,
            self.simultaneous_count_upper,
            self.excess_count_lower,
            self.standardized_excess_lower,
            *self.sigma_rgb_255,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("clipping-null values must be finite")
        if self.expected_extrema_upper_mean < 0.0 or self.simultaneous_count_upper < 0.0:
            raise ValueError("clipping-null expectations must be nonnegative")
        if any(value <= 0.0 for value in self.sigma_rgb_255):
            raise ValueError("clipping-null sigma must be positive")
        mask = np.asarray(self.support_mask)
        if mask.dtype != np.bool_ or mask.ndim != 2:
            raise ValueError("clipping-null support must be a boolean HxW mask")
        mask = np.ascontiguousarray(mask)
        if self.support_hash != _mask_hash(mask):
            raise ValueError("clipping-null support hash mismatch")
        if bool(self.supported) != bool(self.excess_count_lower > 0.0):
            raise ValueError("clipping-null support decision drift")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("clipping null cannot read labels or outcomes")
        object.__setattr__(self, "support_mask", mask)


@dataclass(frozen=True)
class VarianceLawReceipt:
    """Held-out paired residual law and variance-stabilization diagnostics."""

    endpoint: str
    coordinate_frame: str
    tile_size: int
    fit_tiles: int
    check_tiles: int
    raw_model_check_losses: tuple[tuple[str, float], ...]
    raw_model_coefficients: tuple[tuple[str, tuple[float, ...]], ...]
    raw_best_model: str
    raw_best_to_second_loss_margin: float
    raw_variance_intensity_spearman: float
    anscombe_variance_intensity_spearman: float
    log_variance_intensity_spearman: float
    raw_constant_check_loss: float
    anscombe_constant_check_loss: float
    log_constant_check_loss: float
    anscombe_flattening_gain_lower: float
    log_flattening_gain_lower: float
    valid_support_mask: np.ndarray
    support_hash: str
    law_hint: str
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if self.endpoint not in {"first", "second"}:
            raise ValueError("variance law needs a concrete endpoint")
        if self.coordinate_frame != "flow_native":
            raise ValueError("variance-law statistics must use flow coordinates")
        if self.tile_size < 16 or min(self.fit_tiles, self.check_tiles) < 1:
            raise ValueError("invalid variance-law tile accounting")
        losses = dict(self.raw_model_check_losses)
        coefficients = dict(self.raw_model_coefficients)
        if tuple(losses) != RAW_VARIANCE_MODELS or tuple(coefficients) != RAW_VARIANCE_MODELS:
            raise ValueError("variance-law model registry drift")
        if self.raw_best_model not in RAW_VARIANCE_MODELS:
            raise ValueError("unknown best variance law")
        if self.law_hint not in {"wiener3", "anscombe_wiener3", "log_wiener3"}:
            raise ValueError("unknown variance-law action hint")
        values = (
            *losses.values(),
            self.raw_best_to_second_loss_margin,
            self.raw_variance_intensity_spearman,
            self.anscombe_variance_intensity_spearman,
            self.log_variance_intensity_spearman,
            self.raw_constant_check_loss,
            self.anscombe_constant_check_loss,
            self.log_constant_check_loss,
            self.anscombe_flattening_gain_lower,
            self.log_flattening_gain_lower,
            *(item for values in coefficients.values() for item in values),
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("variance-law values must be finite")
        if any(value < 0.0 for value in losses.values()):
            raise ValueError("variance-law losses must be nonnegative")
        mask = np.asarray(self.valid_support_mask)
        if mask.dtype != np.bool_ or mask.ndim != 2:
            raise ValueError("variance-law support must be a boolean HxW mask")
        mask = np.ascontiguousarray(mask)
        if self.support_hash != _mask_hash(mask):
            raise ValueError("variance-law support hash mismatch")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("variance law cannot read labels or outcomes")
        object.__setattr__(self, "valid_support_mask", mask)


@dataclass(frozen=True)
class ExactMedianResponseReceipt:
    """Paired action response of the exact sparse median intervention."""

    endpoint: str
    coordinate_frame: str
    tile_size: int
    support_pixels: int
    fit_tiles: int
    check_tiles: int
    fit_gain_lower_255: float
    check_gain_lower_255: float
    mean_gain_255: float
    reference_structure_fraction: float
    changed_support_mask: np.ndarray
    support_hash: str
    response_supported: bool
    visibility_assumption_resolved: bool
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if self.endpoint not in {"first", "second"}:
            raise ValueError("median response needs a concrete endpoint")
        if self.coordinate_frame != "flow_native":
            raise ValueError("median response must use flow coordinates")
        if self.tile_size < 16 or min(self.support_pixels, self.fit_tiles, self.check_tiles) < 0:
            raise ValueError("invalid median-response accounting")
        for value in (
            self.fit_gain_lower_255, self.check_gain_lower_255,
            self.mean_gain_255, self.reference_structure_fraction,
        ):
            _finite(value, "median-response value")
        if not 0.0 <= self.reference_structure_fraction <= 1.0:
            raise ValueError("reference structure fraction must lie in [0,1]")
        mask = np.asarray(self.changed_support_mask)
        if mask.dtype != np.bool_ or mask.ndim != 2:
            raise ValueError("median-response support must be a boolean HxW mask")
        mask = np.ascontiguousarray(mask)
        if self.support_hash != _mask_hash(mask):
            raise ValueError("median-response support hash mismatch")
        expected = bool(
            self.fit_tiles > 0 and self.check_tiles > 0
            and self.fit_gain_lower_255 > 0.0 and self.check_gain_lower_255 > 0.0
        )
        if bool(self.response_supported) != expected:
            raise ValueError("median-response support decision drift")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("median response cannot read labels or outcomes")
        object.__setattr__(self, "changed_support_mask", mask)


@dataclass(frozen=True)
class PairedNoiseLawReceipt:
    endpoint: str
    clipping_null: DiffuseClippingNullReceipt
    variance_law: VarianceLawReceipt
    median_response: ExactMedianResponseReceipt
    median_feasible_before_visibility: bool
    unresolved_vetoes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.endpoint not in {"first", "second"}:
            raise ValueError("paired noise receipt needs a concrete endpoint")
        if any(row.endpoint != self.endpoint for row in (
            self.clipping_null, self.variance_law, self.median_response,
        )):
            raise ValueError("paired noise subreceipt endpoint mismatch")
        expected = bool(self.clipping_null.supported and self.median_response.response_supported)
        if bool(self.median_feasible_before_visibility) != expected:
            raise ValueError("median feasible-set decision drift")
        if not self.unresolved_vetoes or len(set(self.unresolved_vetoes)) != len(
            self.unresolved_vetoes
        ):
            raise ValueError("unresolved paired-noise vetoes must be explicit")


def _noise_sigma_rgb(image: np.ndarray) -> tuple[float, float, float]:
    source = image.astype(np.float32)
    values = []
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
        values.append(max(sigma, 0.5))
    return tuple(values)  # type: ignore[return-value]


def diffuse_clipping_null_receipt(
    image: np.ndarray,
    *,
    endpoint: str,
    alpha: float = 0.05,
    hypotheses_tested: int = 8,
) -> DiffuseClippingNullReceipt:
    """Bound isolated all-RGB extrema expected from diffuse clipped noise.

    Per-channel Gaussian tail probabilities are combined with a Frechet upper
    bound (the minimum marginal probability), not an independence product.
    This is deliberately conservative under unknown RGB noise correlation.
    """
    image = _validate_image(image)
    if endpoint not in {"first", "second"}:
        raise ValueError("clipping null needs a concrete endpoint")
    if not 0.0 < float(alpha) < 1.0 or hypotheses_tested < 1:
        raise ValueError("invalid clipping-null multiplicity")
    height, width = image.shape[:2]
    valid = np.zeros((height, width), dtype=bool)
    valid[1:-1, 1:-1] = True
    support = (impulse_support(image) > 0.0) & valid
    local_center = cv2.medianBlur(image, 3).astype(np.float64)
    sigma = np.asarray(_noise_sigma_rgb(image), dtype=np.float64)
    black_marginal = ndtr((1.5 - local_center) / sigma.reshape(1, 1, 3))
    white_marginal = ndtr((local_center - 253.5) / sigma.reshape(1, 1, 3))
    # For an all-channel event with unknown dependence, P(intersection) is at
    # most the smallest marginal probability.  Summing black and white remains
    # a valid union upper bound after clipping at one.
    probability_upper = np.minimum(
        np.min(black_marginal, axis=2) + np.min(white_marginal, axis=2), 1.0,
    )
    expected = float(probability_upper[valid].sum())
    log_term = math.log(float(hypotheses_tested) / float(alpha))
    simultaneous_upper = min(
        float(valid.sum()),
        expected + math.sqrt(2.0 * expected * log_term) + 2.0 * log_term / 3.0,
    )
    observed = int(support.sum())
    excess = float(observed - simultaneous_upper)
    standardized = excess / math.sqrt(max(expected, 1.0))
    return DiffuseClippingNullReceipt(
        endpoint=endpoint, coordinate_frame=f"{endpoint}_native",
        alpha=float(alpha), hypotheses_tested=int(hypotheses_tested),
        valid_pixels=int(valid.sum()), observed_isolated_extrema=observed,
        expected_extrema_upper_mean=expected,
        simultaneous_count_upper=simultaneous_upper,
        excess_count_lower=excess,
        standardized_excess_lower=standardized,
        sigma_rgb_255=tuple(float(value) for value in sigma),
        support_mask=np.ascontiguousarray(support),
        support_hash=_mask_hash(support), supported=excess > 0.0,
    )


def _warp_second(value: np.ndarray, flow: np.ndarray, interpolation: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
    valid = ((map_x >= 1.0) & (map_x <= width - 2.0)
             & (map_y >= 1.0) & (map_y <= height - 2.0))
    warped = cv2.remap(
        np.asarray(value), map_x, map_y, interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    return np.ascontiguousarray(warped), np.ascontiguousarray(valid)


def _highpass(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    low = cv2.GaussianBlur(
        source, (0, 0), 1.2, borderType=cv2.BORDER_REFLECT_101,
    )
    return np.ascontiguousarray(source - low)


def _robust_variance(value: np.ndarray) -> float:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    center = float(np.median(flat))
    sigma = float(np.median(np.abs(flat - center)) / 0.6744897501960817)
    return sigma * sigma


def _lower_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    mean = float(values.mean())
    if values.size == 1:
        return mean
    return mean - float(values.std(ddof=1) / math.sqrt(values.size))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return 0.0
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else 0.0


def _design(mean: np.ndarray, model: str) -> np.ndarray:
    if model == "constant":
        return np.ones((mean.size, 1), dtype=np.float64)
    if model == "mean":
        return np.stack((np.ones_like(mean), mean), axis=1)
    if model == "mean_squared":
        return np.stack((np.ones_like(mean), mean * mean), axis=1)
    raise ValueError(model)


def _fit_and_losses(mean: np.ndarray, variance: np.ndarray, fit: np.ndarray,
                    check: np.ndarray
                    ) -> tuple[dict[str, tuple[float, ...]], dict[str, np.ndarray]]:
    scale = max(float(np.median(variance[fit])), float(np.mean(variance[fit])) * 0.1, 1e-12)
    coefficients: dict[str, tuple[float, ...]] = {}
    losses: dict[str, np.ndarray] = {}
    for model in RAW_VARIANCE_MODELS:
        matrix_fit = _design(mean[fit], model)
        coefficient, _ = nnls(matrix_fit, variance[fit])
        prediction = _design(mean[check], model) @ coefficient
        losses[model] = ((variance[check] - prediction) / scale) ** 2
        coefficients[model] = tuple(float(value) for value in coefficient)
    return coefficients, losses


def _constant_losses(values: np.ndarray, fit: np.ndarray, check: np.ndarray
                     ) -> np.ndarray:
    center = float(np.mean(values[fit]))
    scale = max(float(np.median(values[fit])), center * 0.1, 1e-12)
    return ((values[check] - center) / scale) ** 2


def _variance_law_receipt(
    *, endpoint: str, first: np.ndarray, warped_second: np.ndarray,
    valid: np.ndarray, tile_size: int,
) -> VarianceLawReceipt:
    raw_first = first.astype(np.float32)
    raw_second = warped_second.astype(np.float32)
    anscombe_first = 2.0 * np.sqrt(raw_first + np.float32(3.0 / 8.0))
    anscombe_second = 2.0 * np.sqrt(raw_second + np.float32(3.0 / 8.0))
    log_first = np.log1p(raw_first)
    log_second = np.log1p(raw_second)
    residuals = {
        "raw": _highpass(raw_first) - _highpass(raw_second),
        "anscombe": _highpass(anscombe_first) - _highpass(anscombe_second),
        "log": _highpass(log_first) - _highpass(log_second),
    }
    midpoint = 0.5 * (raw_first + raw_second)
    height, width = valid.shape
    means: list[float] = []
    raw_variance: list[float] = []
    anscombe_variance: list[float] = []
    log_variance: list[float] = []
    parity: list[int] = []
    used_support = np.zeros_like(valid)
    for top in range(0, height - tile_size + 1, tile_size):
        for left in range(0, width - tile_size + 1, tile_size):
            ys = slice(top, top + tile_size)
            xs = slice(left, left + tile_size)
            tile_valid = valid[ys, xs]
            if float(tile_valid.mean()) < 0.85:
                continue
            choose = np.repeat(tile_valid[..., None], 3, axis=2)
            means.append(float(midpoint[ys, xs][choose].mean()) / 255.0)
            raw_variance.append(_robust_variance(residuals["raw"][ys, xs][choose]))
            anscombe_variance.append(_robust_variance(
                residuals["anscombe"][ys, xs][choose]
            ))
            log_variance.append(_robust_variance(residuals["log"][ys, xs][choose]))
            parity.append(((top // tile_size) + (left // tile_size)) % 2)
            used_support[ys, xs] = tile_valid
    mean = np.asarray(means, dtype=np.float64)
    raw = np.asarray(raw_variance, dtype=np.float64)
    anscombe = np.asarray(anscombe_variance, dtype=np.float64)
    logarithmic = np.asarray(log_variance, dtype=np.float64)
    parity_array = np.asarray(parity, dtype=np.int64)
    fit = parity_array == 0
    check = parity_array == 1
    if min(int(fit.sum()), int(check.sum())) < 4:
        raise ValueError("insufficient paired tiles for variance law")
    coefficients, loss_vectors = _fit_and_losses(mean, raw, fit, check)
    losses = {name: float(values.mean()) for name, values in loss_vectors.items()}
    ordered = sorted(losses, key=losses.get)
    best, second = ordered[:2]
    raw_constant = loss_vectors["constant"]
    anscombe_constant = _constant_losses(anscombe, fit, check)
    log_constant = _constant_losses(logarithmic, fit, check)
    anscombe_gain = _lower_mean(raw_constant - anscombe_constant)
    log_gain = _lower_mean(raw_constant - log_constant)
    hint = {
        "constant": "wiener3",
        "mean": "anscombe_wiener3",
        "mean_squared": "log_wiener3",
    }[best]
    return VarianceLawReceipt(
        endpoint=endpoint, coordinate_frame="flow_native", tile_size=tile_size,
        fit_tiles=int(fit.sum()), check_tiles=int(check.sum()),
        raw_model_check_losses=tuple((name, losses[name]) for name in RAW_VARIANCE_MODELS),
        raw_model_coefficients=tuple(
            (name, coefficients[name]) for name in RAW_VARIANCE_MODELS
        ),
        raw_best_model=best,
        raw_best_to_second_loss_margin=float(losses[second] - losses[best]),
        raw_variance_intensity_spearman=_safe_spearman(mean[check], raw[check]),
        anscombe_variance_intensity_spearman=_safe_spearman(
            mean[check], anscombe[check]
        ),
        log_variance_intensity_spearman=_safe_spearman(
            mean[check], logarithmic[check]
        ),
        raw_constant_check_loss=float(raw_constant.mean()),
        anscombe_constant_check_loss=float(anscombe_constant.mean()),
        log_constant_check_loss=float(log_constant.mean()),
        anscombe_flattening_gain_lower=anscombe_gain,
        log_flattening_gain_lower=log_gain,
        valid_support_mask=np.ascontiguousarray(used_support),
        support_hash=_mask_hash(used_support), law_hint=hint,
    )


def _median_response_receipt(
    *, endpoint: str, first: np.ndarray, second: np.ndarray,
    warped_second: np.ndarray, flow: np.ndarray, valid: np.ndarray,
    tile_size: int,
) -> ExactMedianResponseReceipt:
    first_support = impulse_support(first) > 0.0
    second_support_native = impulse_support(second) > 0.0
    first_median = cv2.medianBlur(first, 3)
    second_median = cv2.medianBlur(second, 3)
    repaired_first = first.copy()
    repaired_first[first_support] = first_median[first_support]
    repaired_second = second.copy()
    repaired_second[second_support_native] = second_median[second_support_native]
    warped_repaired_second, valid_repaired = _warp_second(
        repaired_second.astype(np.float32), flow, cv2.INTER_LINEAR,
    )
    warped_second_support, valid_support = _warp_second(
        second_support_native.astype(np.float32), flow, cv2.INTER_NEAREST,
    )
    warped_second_median, valid_reference_median = _warp_second(
        second_median.astype(np.float32), flow, cv2.INTER_LINEAR,
    )
    valid_all = valid & valid_repaired & valid_support & valid_reference_median
    if endpoint == "first":
        native_endpoint = first.astype(np.float32)
        candidate_endpoint = repaired_first.astype(np.float32)
        reference = warped_second.astype(np.float32)
        support = first_support & valid_all
        reference_delta = np.mean(
            np.abs(warped_second_median - warped_second.astype(np.float32)), axis=2,
        )
    else:
        native_endpoint = warped_second.astype(np.float32)
        candidate_endpoint = warped_repaired_second.astype(np.float32)
        reference = first.astype(np.float32)
        support = (warped_second_support > 0.5) & valid_all
        reference_delta = np.mean(
            np.abs(first_median.astype(np.float32) - first.astype(np.float32)), axis=2,
        )
    before = np.mean(np.abs(native_endpoint - reference), axis=2)
    after = np.mean(np.abs(candidate_endpoint - reference), axis=2)
    response = before - after
    endpoint_delta = np.mean(np.abs(candidate_endpoint - native_endpoint), axis=2)
    height, width = support.shape
    fold_values: list[list[float]] = [[], []]
    reference_structure = 0
    support_pixels = int(support.sum())
    if support_pixels:
        reference_structure = int(np.sum(
            support & (reference_delta >= np.maximum(endpoint_delta, 1.0))
        ))
    for top in range(0, height - tile_size + 1, tile_size):
        for left in range(0, width - tile_size + 1, tile_size):
            ys = slice(top, top + tile_size)
            xs = slice(left, left + tile_size)
            selected = support[ys, xs]
            if not np.any(selected):
                continue
            fold = ((top // tile_size) + (left // tile_size)) % 2
            fold_values[fold].append(float(response[ys, xs][selected].mean()))
    fit_values = np.asarray(fold_values[0], dtype=np.float64)
    check_values = np.asarray(fold_values[1], dtype=np.float64)
    fit_lower = _lower_mean(fit_values)
    check_lower = _lower_mean(check_values)
    response_supported = bool(
        fit_values.size > 0 and check_values.size > 0
        and fit_lower > 0.0 and check_lower > 0.0
    )
    return ExactMedianResponseReceipt(
        endpoint=endpoint, coordinate_frame="flow_native", tile_size=tile_size,
        support_pixels=support_pixels, fit_tiles=int(fit_values.size),
        check_tiles=int(check_values.size), fit_gain_lower_255=fit_lower,
        check_gain_lower_255=check_lower,
        mean_gain_255=float(response[support].mean()) if support_pixels else 0.0,
        reference_structure_fraction=float(reference_structure / max(support_pixels, 1)),
        changed_support_mask=np.ascontiguousarray(support),
        support_hash=_mask_hash(support), response_supported=response_supported,
        # One forward flow cannot establish occlusion-free correspondence.
        visibility_assumption_resolved=False,
    )


def paired_noise_law_receipts(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    *,
    tile_size: int = 32,
    alpha: float = 0.05,
    hypotheses_tested: int = 8,
) -> dict[str, PairedNoiseLawReceipt]:
    """Build action-specific noise-law receipts without labels or outcomes."""
    first = _validate_image(first)
    second = _validate_image(second)
    flow = np.asarray(flow, dtype=np.float32)
    if (first.shape != second.shape or flow.shape != (*first.shape[:2], 2)
            or not np.isfinite(flow).all()):
        raise ValueError("expected equal RGB endpoints and finite HxWx2 flow")
    if tile_size < 16:
        raise ValueError("noise-law tile size must be at least 16")
    warped_second, valid = _warp_second(
        second.astype(np.float32), flow, cv2.INTER_LINEAR,
    )
    clipping = {
        endpoint: diffuse_clipping_null_receipt(
            image, endpoint=endpoint, alpha=alpha,
            hypotheses_tested=hypotheses_tested,
        )
        for endpoint, image in (("first", first), ("second", second))
    }
    rows = {}
    for endpoint in ("first", "second"):
        variance = _variance_law_receipt(
            endpoint=endpoint, first=first, warped_second=warped_second,
            valid=valid, tile_size=tile_size,
        )
        response = _median_response_receipt(
            endpoint=endpoint, first=first, second=second,
            warped_second=warped_second, flow=flow, valid=valid,
            tile_size=tile_size,
        )
        rows[endpoint] = PairedNoiseLawReceipt(
            endpoint=endpoint, clipping_null=clipping[endpoint],
            variance_law=variance, median_response=response,
            median_feasible_before_visibility=bool(
                clipping[endpoint].supported and response.response_supported
            ),
            unresolved_vetoes=(
                "forward_backward_visibility_unresolved",
                "same_support_signed_task_utility_unresolved",
                "selection_aware_tail_risk_unresolved",
            ),
        )
    return rows
