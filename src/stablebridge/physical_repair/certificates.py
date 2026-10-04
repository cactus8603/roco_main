"""Action-specific physical applicability certificates.

The current implementation covers the highest-risk common-passband motion
action.  It compares motion with identity and isotropic Gaussian explanations,
requires a one-sided bandwidth loss, and verifies the fitted kernel on spatial
holdout tiles.  The certificate establishes operator compatibility only; it
does not assert task-error improvement.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from .contracts import ActionSpec, PhysicalCertificate


@dataclass(frozen=True)
class CommonPassbandConfig:
    motion_lengths: tuple[int, ...] = (5, 9, 13, 17, 21)
    motion_angles: tuple[int, ...] = (0, 30, 60, 90, 120, 150)
    gaussian_sigmas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    zoom_maxima: tuple[float, ...] = (1.08, 1.12, 1.16, 1.20, 1.24, 1.28, 1.32)
    tile_size: int = 32
    minimum_valid_fraction: float = 0.25
    minimum_support_pixels: int = 512
    minimum_gradient_deficit: float = 0.03
    minimum_spectral_deficit: float = 0.02
    minimum_holdout_relative_gain: float = 0.02
    minimum_motion_specificity: float = 0.005
    minimum_fit_stability: float = 0.55
    calibration_version: str = "physical-null-v1"

    def __post_init__(self) -> None:
        if (not self.motion_lengths or not self.motion_angles or not self.gaussian_sigmas
                or not self.zoom_maxima
                or min(self.motion_lengths) < 3 or any(length % 2 == 0 for length in self.motion_lengths)
                or self.tile_size < 8 or self.minimum_support_pixels < 1):
            raise ValueError("invalid common-passband candidate grid")
        for name in ("minimum_valid_fraction", "minimum_gradient_deficit",
                     "minimum_spectral_deficit", "minimum_holdout_relative_gain",
                     "minimum_motion_specificity", "minimum_fit_stability"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must lie in [0,1]")


def _validate(first: np.ndarray, second: np.ndarray,
              flow: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(first); second = np.asarray(second); flow = np.asarray(flow)
    if (first.dtype != np.uint8 or second.dtype != np.uint8
            or first.ndim != 3 or first.shape != second.shape or first.shape[2] != 3
            or flow.shape != (*first.shape[:2], 2) or not np.isfinite(flow).all()):
        raise ValueError("expected matching uint8 RGB pair and finite HxWx2 flow")
    return np.ascontiguousarray(first), np.ascontiguousarray(second), np.ascontiguousarray(flow)


def _warp_second(second: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = second.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx + flow[..., 0].astype(np.float32)
    map_y = yy + flow[..., 1].astype(np.float32)
    valid = ((map_x >= 0.0) & (map_x <= width - 1.0)
             & (map_y >= 0.0) & (map_y <= height - 1.0))
    warped = cv2.remap(second.astype(np.float32), map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return np.ascontiguousarray(warped), valid


def _motion_kernel(length: int, angle: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    theta = math.radians(float(angle))
    dx, dy = math.cos(theta) * center, math.sin(theta) * center
    cv2.line(kernel,
             (int(round(center - dx)), int(round(center - dy))),
             (int(round(center + dx)), int(round(center + dy))), 1.0, 1)
    return kernel / max(float(kernel.sum()), 1.0)


def _motion(image: np.ndarray, length: int, angle: float) -> np.ndarray:
    return cv2.filter2D(image.astype(np.float32), -1,
                        _motion_kernel(length, angle),
                        borderType=cv2.BORDER_REFLECT_101)


def _gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigmaX=sigma,
                            sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def _zoom(image: np.ndarray, factor: float) -> np.ndarray:
    height, width = image.shape[:2]
    resized = cv2.resize(image, None, fx=factor, fy=factor,
                         interpolation=cv2.INTER_LINEAR)
    y0 = (resized.shape[0] - height) // 2
    x0 = (resized.shape[1] - width) // 2
    return resized[y0:y0 + height, x0:x0 + width]


def _zoom_blur(image: np.ndarray, maximum: float) -> np.ndarray:
    source = image.astype(np.float32)
    total = source.copy()
    factors = np.arange(1.02, float(maximum) + 1e-6, 0.02)
    for factor in factors:
        total += _zoom(source, float(factor))
    return total / float(len(factors) + 1)


def _checkerboard(shape: tuple[int, int], tile_size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    fit = ((yy // tile_size + xx // tile_size) % 2) == 0
    return fit, ~fit


def _subsample(values: np.ndarray, maximum: int = 100_000) -> np.ndarray:
    if len(values) <= maximum:
        return values
    return values[::max(1, len(values) // maximum)][:maximum]


def _affine(candidate: np.ndarray, target: np.ndarray,
            support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the same low-dimensional radiometric nuisance for every model."""
    gain = np.ones(3, dtype=np.float64)
    offset = np.zeros(3, dtype=np.float64)
    for channel in range(3):
        x = _subsample(candidate[..., channel][support].astype(np.float64))
        y = _subsample(target[..., channel][support].astype(np.float64))
        if len(x) < 16:
            continue
        center_x, center_y = np.median(x), np.median(y)
        variance = float(np.mean((x - center_x) ** 2))
        slope = (float(np.mean((x - center_x) * (y - center_y)))
                 / max(variance, 1e-6))
        gain[channel] = np.clip(slope, 0.5, 2.0)
        offset[channel] = np.clip(center_y - gain[channel] * center_x, -64.0, 64.0)
    return gain, offset


def _score(candidate: np.ndarray, target: np.ndarray, fit: np.ndarray,
           support: np.ndarray) -> tuple[float, float]:
    gain, offset = _affine(candidate, target, fit)
    adjusted = candidate.astype(np.float32) * gain.astype(np.float32)
    adjusted += offset.astype(np.float32)
    residual = np.mean(np.abs(adjusted - target.astype(np.float32)), axis=2)
    values = residual[support]
    return ((float(np.median(values)) if values.size else math.inf),
            float(np.mean(values)) if values.size else math.inf)


def _gradient_p90(image: np.ndarray) -> float:
    gray = cv2.cvtColor(np.clip(np.rint(image), 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.quantile(np.sqrt(gx * gx + gy * gy), 0.90))


def _spectral_high(image: np.ndarray) -> float:
    gray = cv2.cvtColor(np.clip(np.rint(image), 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)
    small = cv2.resize(gray, (256, 128), interpolation=cv2.INTER_AREA)
    small -= float(small.mean())
    power = np.abs(np.fft.rfft2(small)) ** 2
    fy = np.fft.fftfreq(small.shape[0])[:, None]
    fx = np.fft.rfftfreq(small.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    total = float(power[radius > 0].sum())
    return float(power[radius >= 0.25].sum()) / max(total, 1e-12)


def _angle_distance(left: float, right: float) -> float:
    delta = abs(float(left) - float(right)) % 180.0
    return min(delta, 180.0 - delta)


def _best_motion(sharp: np.ndarray, target: np.ndarray, fit: np.ndarray,
                 check: np.ndarray, config: CommonPassbandConfig):
    rows = []
    for length in config.motion_lengths:
        for angle in config.motion_angles:
            candidate = _motion(sharp, length, angle)
            fit_score, _ = _score(candidate, target, fit, fit)
            check_score, _ = _score(candidate, target, fit, check)
            rows.append((fit_score, check_score, length, angle))
    fit_best = min(rows, key=lambda row: row[0])
    check_best = min(rows, key=lambda row: row[1])
    angle_stability = 1.0 - _angle_distance(fit_best[3], check_best[3]) / 90.0
    length_span = max(config.motion_lengths) - min(config.motion_lengths)
    length_stability = 1.0 - abs(fit_best[2] - check_best[2]) / max(length_span, 1)
    stability = float(np.clip(0.5 * (angle_stability + length_stability), 0.0, 1.0))
    return fit_best, check_best, stability, rows


def _best_zoom(sharp: np.ndarray, target: np.ndarray, fit: np.ndarray,
               check: np.ndarray, config: CommonPassbandConfig):
    rows = []
    for maximum in config.zoom_maxima:
        candidate = _zoom_blur(sharp, maximum)
        fit_score, _ = _score(candidate, target, fit, fit)
        check_score, _ = _score(candidate, target, fit, check)
        rows.append((fit_score, check_score, maximum))
    fit_best = min(rows, key=lambda row: row[0])
    check_best = min(rows, key=lambda row: row[1])
    span = max(config.zoom_maxima) - min(config.zoom_maxima)
    stability = 1.0 - abs(fit_best[2] - check_best[2]) / max(span, 1e-6)
    return fit_best, check_best, float(np.clip(stability, 0.0, 1.0)), rows


def common_passband_certificate(first: np.ndarray, second: np.ndarray,
                                native_flow: np.ndarray, *,
                                degraded_endpoint: str,
                                config: CommonPassbandConfig | None = None
                                ) -> PhysicalCertificate:
    """Test whether a one-sided motion common-passband action is supported."""
    config = config or CommonPassbandConfig()
    first, second, native_flow = _validate(first, second, native_flow)
    if degraded_endpoint not in {"first", "second"}:
        raise ValueError("common-passband certificate needs first or second")
    started = time.perf_counter()
    warped_second, valid = _warp_second(second, native_flow)
    if degraded_endpoint == "first":
        sharp, target = warped_second, first.astype(np.float32)
        sharp_native, target_native = second, first
        modified = "second"
        coordinate = "second_native"
    else:
        sharp, target = first.astype(np.float32), warped_second
        sharp_native, target_native = first, second
        modified = "first"
        coordinate = "first_native"
    radius = max(config.motion_lengths) // 2 + 1
    valid_inner = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                            iterations=max(1, radius // 3)).astype(bool)
    fit_tiles, check_tiles = _checkerboard(valid.shape, config.tile_size)
    fit, check = valid_inner & fit_tiles, valid_inner & check_tiles
    reasons = []
    valid_fraction = float(valid_inner.mean())
    if valid_fraction < config.minimum_valid_fraction:
        reasons.append("insufficient_observation_support")
    if int(fit.sum()) < config.minimum_support_pixels or int(check.sum()) < config.minimum_support_pixels:
        reasons.append("insufficient_spatial_holdout_support")
    action = ActionSpec(
        operator_id="blind_motion_common", operator_version="v2-null-holdout",
        domain="image", hypothesized_degraded_endpoint=degraded_endpoint,
        modified_endpoint=modified, coordinate_frame=coordinate,
        execution_semantics="from_zero",
    )
    if reasons:
        elapsed = time.perf_counter() - started
        return PhysicalCertificate(
            action=action, status="unsupported",
            observation_support_fraction=valid_fraction,
            identifiable_support_fraction=0.0, null_score=0.0, action_score=0.0,
            spatial_holdout_gain=0.0, parameter_uncertainty=1.0,
            fit_stability=0.0, rejection_reasons=tuple(reasons),
            calibration_version=config.calibration_version,
            measured_probe_seconds=elapsed,
        )
    null_fit, _ = _score(sharp, target, fit, fit)
    null_check, _ = _score(sharp, target, fit, check)
    motion_fit, motion_check_oracle, stability, _ = _best_motion(
        sharp, target, fit, check, config,
    )
    # The action parameter is selected only on fit tiles.  Its check score is
    # the second item in motion_fit; motion_check_oracle is diagnostic only.
    action_check = float(motion_fit[1])
    gaussian_rows = []
    for sigma in config.gaussian_sigmas:
        candidate = _gaussian(sharp, sigma)
        gaussian_rows.append((*_score(candidate, target, fit, check), sigma))
    gaussian_best = min(gaussian_rows, key=lambda row: row[0])
    gaussian_check = float(gaussian_best[1])
    gradient_sharp, gradient_target = _gradient_p90(sharp_native), _gradient_p90(target_native)
    spectral_sharp, spectral_target = _spectral_high(sharp_native), _spectral_high(target_native)
    gradient_deficit = (gradient_sharp - gradient_target) / max(gradient_sharp, 1e-6)
    spectral_deficit = (spectral_sharp - spectral_target) / max(spectral_sharp, 1e-9)
    holdout_gain = null_check - action_check
    relative_gain = holdout_gain / max(null_check, 1e-6)
    specificity = (gaussian_check - action_check) / max(null_check, 1e-6)
    bandwidth_votes = int(gradient_deficit >= config.minimum_gradient_deficit)
    bandwidth_votes += int(spectral_deficit >= config.minimum_spectral_deficit)
    if bandwidth_votes == 0:
        reasons.append("no_one_sided_bandwidth_loss")
    if relative_gain < config.minimum_holdout_relative_gain:
        reasons.append("motion_does_not_beat_identity_on_holdout")
    if specificity < config.minimum_motion_specificity:
        reasons.append("motion_not_specific_against_isotropic_blur")
    if stability < config.minimum_fit_stability:
        reasons.append("motion_parameters_not_spatially_stable")
    status = "supported" if not reasons else "rejected"
    elapsed = time.perf_counter() - started
    return PhysicalCertificate(
        action=action, status=status,
        observation_support_fraction=valid_fraction,
        identifiable_support_fraction=float(min(fit.sum(), check.sum()) / valid.size),
        null_score=float(null_check), action_score=action_check,
        spatial_holdout_gain=float(holdout_gain),
        parameter_uncertainty=float(1.0 - stability), fit_stability=stability,
        competing_model_scores={
            "identity_check": float(null_check),
            "gaussian_check": gaussian_check,
            "motion_check_oracle": float(motion_check_oracle[1]),
        },
        estimated_parameters={
            "motion_length": float(motion_fit[2]),
            "motion_angle_degrees_mod_180": float(motion_fit[3]),
            "gaussian_sigma": float(gaussian_best[2]),
        },
        diagnostics={
            "null_fit": float(null_fit), "motion_fit": float(motion_fit[0]),
            "gradient_deficit": float(gradient_deficit),
            "spectral_deficit": float(spectral_deficit),
            "bandwidth_votes": float(bandwidth_votes),
            "holdout_relative_gain": float(relative_gain),
            "motion_specificity": float(specificity),
            "check_selected_length": float(motion_check_oracle[2]),
            "check_selected_angle": float(motion_check_oracle[3]),
        },
        rejection_reasons=tuple(reasons),
        calibration_version=config.calibration_version,
        measured_probe_seconds=elapsed,
    )


def zoom_common_passband_certificate(first: np.ndarray, second: np.ndarray,
                                     native_flow: np.ndarray, *,
                                     degraded_endpoint: str,
                                     config: CommonPassbandConfig | None = None
                                     ) -> PhysicalCertificate:
    """Test a one-sided, image-centred radial zoom common-passband action."""
    config = config or CommonPassbandConfig()
    first, second, native_flow = _validate(first, second, native_flow)
    if degraded_endpoint not in {"first", "second"}:
        raise ValueError("zoom certificate needs first or second")
    started = time.perf_counter()
    warped_second, valid = _warp_second(second, native_flow)
    if degraded_endpoint == "first":
        sharp, target = warped_second, first.astype(np.float32)
        sharp_native, target_native = second, first
        modified, coordinate = "second", "second_native"
    else:
        sharp, target = first.astype(np.float32), warped_second
        sharp_native, target_native = first, second
        modified, coordinate = "first", "first_native"
    valid_inner = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8),
                            iterations=2).astype(bool)
    fit_tiles, check_tiles = _checkerboard(valid.shape, config.tile_size)
    fit, check = valid_inner & fit_tiles, valid_inner & check_tiles
    reasons = []
    valid_fraction = float(valid_inner.mean())
    if valid_fraction < config.minimum_valid_fraction:
        reasons.append("insufficient_observation_support")
    if int(fit.sum()) < config.minimum_support_pixels or int(check.sum()) < config.minimum_support_pixels:
        reasons.append("insufficient_spatial_holdout_support")
    action = ActionSpec(
        operator_id="blind_zoom_common", operator_version="v2-null-holdout",
        domain="image", hypothesized_degraded_endpoint=degraded_endpoint,
        modified_endpoint=modified, coordinate_frame=coordinate,
        execution_semantics="from_zero",
    )
    if reasons:
        return PhysicalCertificate(
            action=action, status="unsupported",
            observation_support_fraction=valid_fraction,
            identifiable_support_fraction=0.0, null_score=0.0, action_score=0.0,
            spatial_holdout_gain=0.0, parameter_uncertainty=1.0,
            fit_stability=0.0, rejection_reasons=tuple(reasons),
            calibration_version=config.calibration_version,
            measured_probe_seconds=time.perf_counter() - started,
        )
    null_fit, _ = _score(sharp, target, fit, fit)
    null_check, _ = _score(sharp, target, fit, check)
    zoom_fit, zoom_check_oracle, stability, _ = _best_zoom(
        sharp, target, fit, check, config,
    )
    action_check = float(zoom_fit[1])
    gaussian_rows = []
    for sigma in config.gaussian_sigmas:
        candidate = _gaussian(sharp, sigma)
        gaussian_rows.append((*_score(candidate, target, fit, check), sigma))
    gaussian_best = min(gaussian_rows, key=lambda row: row[0])
    # Motion is a competing non-radial PSF.  Select it on fit support only.
    motion_fit, _, _, _ = _best_motion(sharp, target, fit, check, config)
    competitor_check = min(float(gaussian_best[1]), float(motion_fit[1]))
    gradient_sharp, gradient_target = _gradient_p90(sharp_native), _gradient_p90(target_native)
    spectral_sharp, spectral_target = _spectral_high(sharp_native), _spectral_high(target_native)
    gradient_deficit = (gradient_sharp - gradient_target) / max(gradient_sharp, 1e-6)
    spectral_deficit = (spectral_sharp - spectral_target) / max(spectral_sharp, 1e-9)
    bandwidth_votes = int(gradient_deficit >= config.minimum_gradient_deficit)
    bandwidth_votes += int(spectral_deficit >= config.minimum_spectral_deficit)
    holdout_gain = null_check - action_check
    relative_gain = holdout_gain / max(null_check, 1e-6)
    specificity = (competitor_check - action_check) / max(null_check, 1e-6)
    if bandwidth_votes == 0:
        reasons.append("no_one_sided_bandwidth_loss")
    if relative_gain < config.minimum_holdout_relative_gain:
        reasons.append("zoom_does_not_beat_identity_on_holdout")
    if specificity < config.minimum_motion_specificity:
        reasons.append("zoom_not_specific_against_nonradial_models")
    if stability < config.minimum_fit_stability:
        reasons.append("zoom_parameter_not_spatially_stable")
    status = "supported" if not reasons else "rejected"
    return PhysicalCertificate(
        action=action, status=status,
        observation_support_fraction=valid_fraction,
        identifiable_support_fraction=float(min(fit.sum(), check.sum()) / valid.size),
        null_score=float(null_check), action_score=action_check,
        spatial_holdout_gain=float(holdout_gain),
        parameter_uncertainty=float(1.0 - stability), fit_stability=stability,
        competing_model_scores={
            "identity_check": float(null_check),
            "gaussian_check": float(gaussian_best[1]),
            "motion_check": float(motion_fit[1]),
            "zoom_check_oracle": float(zoom_check_oracle[1]),
        },
        estimated_parameters={
            "zoom_maximum": float(zoom_fit[2]),
            "gaussian_sigma": float(gaussian_best[2]),
            "motion_length": float(motion_fit[2]),
            "motion_angle_degrees_mod_180": float(motion_fit[3]),
        },
        diagnostics={
            "null_fit": float(null_fit), "zoom_fit": float(zoom_fit[0]),
            "gradient_deficit": float(gradient_deficit),
            "spectral_deficit": float(spectral_deficit),
            "bandwidth_votes": float(bandwidth_votes),
            "holdout_relative_gain": float(relative_gain),
            "zoom_specificity": float(specificity),
            "check_selected_zoom": float(zoom_check_oracle[2]),
        },
        rejection_reasons=tuple(reasons),
        calibration_version=config.calibration_version,
        measured_probe_seconds=time.perf_counter() - started,
    )
