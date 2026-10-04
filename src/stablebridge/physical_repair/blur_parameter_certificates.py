"""Independent A/B-fold forward-model certificates for blur parameters.

Unlike E134, this module does not receive a parameter proposed by a generic
router.  Even spatial tiles estimate one parameter per physical family;
checkerboard-odd tiles test each fixed family/endpoint hypothesis against the
identity forward model. The decision uses held-out spatial re-degradation;
windowed transfer-ratio spectra are retained as diagnostics because boundary
leakage can bias their family ranking. The test bound is simultaneous over all
three families and two endpoints. Corruption labels, task outcomes, and ground
truth are not inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from statistics import NormalDist
from typing import Mapping, Sequence

import cv2
import numpy as np

from .contracts import ActionSpec, PhysicalCertificate


TILE = 64
FAMILIES = ("disk", "gaussian", "motion")
ENDPOINTS = ("first", "second")
FAMILYWISE_ALPHA = 0.05
SIMULTANEOUS_HYPOTHESES = len(FAMILIES) * len(ENDPOINTS)
MIN_BLOCKS_PER_SPLIT = 4
MIN_VALID_FRACTION = 0.90
MIN_TEXTURE_GRADIENT_255 = 2.0
MIN_INFORMATION_RETENTION = 0.05
MIN_ENDPOINT_GAIN = 1.0 / 255.0
FREQUENCY_FLOOR = 1e-3


@dataclass(frozen=True)
class BlurFamilyEvidence:
    """Replayable parameter fit and independent physical check for one action."""

    family: str
    endpoint: str
    selected_parameter: tuple[float, ...]
    identifiable_parameters: tuple[tuple[float, ...], ...]
    fit_score_curve: Mapping[str, float]
    fit_support_hash: str
    check_support_hash: str
    check_blocks: int
    own_null_improvement_median: float
    own_null_improvement_lcb: float
    own_null_spectral_improvement_lcb: float
    pairwise_improvement_lcb: Mapping[str, float]
    endpoint_improvement_lcb: float
    information_retention_median: float
    recoverable_regions: tuple[tuple[int, int], ...]
    recoverability_status: str
    certificate: PhysicalCertificate

    def __post_init__(self) -> None:
        if self.family not in FAMILIES or self.endpoint not in ENDPOINTS:
            raise ValueError("invalid blur family or endpoint")
        if not self.selected_parameter or not self.identifiable_parameters:
            raise ValueError("blur evidence needs a selected and identifiable set")
        if self.selected_parameter not in self.identifiable_parameters:
            raise ValueError("selected parameter must lie in identifiable set")
        if not self.fit_score_curve or not self.fit_support_hash or not self.check_support_hash:
            raise ValueError("blur evidence needs score curve and split hashes")
        if self.fit_support_hash == self.check_support_hash:
            raise ValueError("fit and check support must differ")
        if self.check_blocks < 0:
            raise ValueError("check block count cannot be negative")
        if self.recoverability_status not in {"supported", "unsupported"}:
            raise ValueError("invalid recoverability status")
        if not 0.0 <= self.information_retention_median <= 1.0:
            raise ValueError("information retention must lie in [0,1]")

    @property
    def action_key(self) -> str:
        return f"common_{self.family}@{self.endpoint}"

    @property
    def direct_pairwise_winner(self) -> bool:
        spatial = [value for name, value in self.pairwise_improvement_lcb.items()
                   if name.startswith("spatial_vs_")]
        return bool(self.certificate.status == "supported" and spatial
                    and all(value > 0.0 for value in spatial)
                    and self.endpoint_improvement_lcb > MIN_ENDPOINT_GAIN)


def simultaneous_z(alpha: float = FAMILYWISE_ALPHA,
                   hypotheses: int = SIMULTANEOUS_HYPOTHESES) -> float:
    if not 0.0 < float(alpha) < 1.0 or hypotheses < 1:
        raise ValueError("invalid familywise alpha or hypothesis count")
    return float(NormalDist().inv_cdf(1.0 - float(alpha) / (2.0 * hypotheses)))


def _validate(first: np.ndarray, second: np.ndarray, flow: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(first)
    second = np.asarray(second)
    flow = np.asarray(flow, dtype=np.float32)
    if (first.dtype != np.uint8 or second.dtype != np.uint8
            or first.ndim != 3 or first.shape != second.shape or first.shape[2] != 3
            or flow.shape != (*first.shape[:2], 2) or not np.isfinite(flow).all()):
        raise ValueError("expected matching uint8 RGB pair and finite HxWx2 flow")
    return np.ascontiguousarray(first), np.ascontiguousarray(second), flow


def _warp_second(second: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = second.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
    valid = ((map_x >= 0.0) & (map_x <= width - 1.0)
             & (map_y >= 0.0) & (map_y <= height - 1.0))
    warped = cv2.remap(
        second.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    return np.ascontiguousarray(warped), np.ascontiguousarray(valid)


def _gray(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return cv2.cvtColor(clipped, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _blocks(valid: np.ndarray, sharp: np.ndarray,
            target: np.ndarray) -> list[tuple[int, int, int]]:
    gradients = []
    for image in (sharp, target):
        gray = _gray(image)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradients.append(np.sqrt(gx * gx + gy * gy))
    texture = np.maximum(gradients[0], gradients[1])
    output = []
    height, width = valid.shape
    for y0 in range(0, height - TILE + 1, TILE):
        for x0 in range(0, width - TILE + 1, TILE):
            ys, xs = slice(y0, y0 + TILE), slice(x0, x0 + TILE)
            local = valid[ys, xs]
            if (float(local.mean()) < MIN_VALID_FRACTION
                    or float(np.median(texture[ys, xs][local]))
                    < MIN_TEXTURE_GRADIENT_255):
                continue
            output.append((y0, x0, (y0 // TILE + x0 // TILE) & 1))
    return output


def _support_hash(endpoint: str, parity: int,
                  blocks: Sequence[tuple[int, int, int]]) -> str:
    payload = {
        "endpoint": endpoint, "parity": parity,
        "tile": TILE,
        "blocks": [[y, x] for y, x, value in blocks if value == parity],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@lru_cache(maxsize=1)
def _frequency_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fy = np.fft.fftfreq(TILE)[:, None]
    fx = np.fft.rfftfreq(TILE)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    band = (radius >= 1.0 / TILE) & (radius <= 0.45)
    window = np.outer(np.hanning(TILE), np.hanning(TILE)).astype(np.float64)
    return radius, band, window


def _observation(sharp: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    radius, band, window = _frequency_geometry()
    left = window * (sharp.astype(np.float64) - float(np.mean(sharp)))
    right = window * (target.astype(np.float64) - float(np.mean(target)))
    x, y = np.fft.rfft2(left), np.fft.rfft2(right)
    abs_x, abs_y = np.abs(x), np.abs(y)
    epsilon = max(float(np.median(abs_x[band])) * 1e-3, 1e-9)
    empirical = abs_y / np.maximum(abs_x, epsilon)
    weights = np.sqrt(abs_x * abs_y) * band
    low = band & (radius <= 0.08)
    low_weights = weights * low
    scale = float(np.sum(low_weights * empirical) / max(float(low_weights.sum()), 1e-12))
    log_empirical = np.log(np.clip(
        empirical / max(scale, 1e-9), FREQUENCY_FLOOR, 1.0 / FREQUENCY_FLOOR,
    ))
    fisher = radius * radius * abs_x * abs_x * band
    return {"log_empirical": log_empirical, "weights": weights, "fisher": fisher}


def _disk_kernel(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    return kernel / float(kernel.sum())


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    vector = cv2.getGaussianKernel(2 * radius + 1, sigma, cv2.CV_32F)
    kernel = vector @ vector.T
    return kernel / float(kernel.sum())


def _motion_kernel(length: int, angle: float) -> np.ndarray:
    size = int(length) | 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    theta = math.radians(float(angle))
    dx, dy = math.cos(theta) * center, math.sin(theta) * center
    cv2.line(
        kernel,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        1.0, 1,
    )
    return kernel / max(float(kernel.sum()), 1.0)


def _kernel(family: str, parameter: tuple[float, ...]) -> np.ndarray:
    if family == "disk":
        return _disk_kernel(int(round(parameter[0])))
    if family == "gaussian":
        return _gaussian_kernel(float(parameter[0]))
    if family == "motion":
        return _motion_kernel(int(round(parameter[0])), float(parameter[1]))
    raise ValueError("unknown blur family")


@lru_cache(maxsize=512)
def _otf(family: str, parameter: tuple[float, ...]) -> np.ndarray:
    kernel = _kernel(family, parameter)
    if max(kernel.shape) > TILE:
        raise ValueError("kernel exceeds spectral tile")
    padded = np.zeros((TILE, TILE), dtype=np.float64)
    cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            padded[(y - cy) % TILE, (x - cx) % TILE] += float(kernel[y, x])
    return np.fft.rfft2(padded)


def parameter_grid(family: str) -> tuple[tuple[float, ...], ...]:
    if family == "disk":
        return tuple((float(radius),) for radius in range(1, 9))
    if family == "gaussian":
        return tuple((float(index) / 2.0,) for index in range(1, 17))
    if family == "motion":
        return tuple(
            (float(length), float(angle))
            for length in (3, 7, 11, 15, 17)
            for angle in range(0, 180, 30)
        )
    raise ValueError("unknown blur family")


def _model_scores(observations: Sequence[dict[str, np.ndarray]],
                  family: str, parameter: tuple[float, ...]
                  ) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.abs(_otf(family, parameter))
    log_model = np.log(np.clip(magnitude, FREQUENCY_FLOOR, 1.0))
    scores, information = [], []
    for item in observations:
        weights = item["weights"]
        denominator = max(float(weights.sum()), 1e-12)
        scores.append(math.sqrt(float(np.sum(
            weights * (item["log_empirical"] - log_model) ** 2
        )) / denominator))
        fisher = item["fisher"]
        information.append(float(
            np.sum(fisher * magnitude * magnitude)
            / max(float(np.sum(fisher)), 1e-12)
        ))
    return np.asarray(scores, np.float64), np.asarray(information, np.float64)


def _apply(image: np.ndarray, family: str,
           parameter: tuple[float, ...]) -> np.ndarray:
    return cv2.filter2D(
        np.asarray(image, dtype=np.float32), -1, _kernel(family, parameter),
        borderType=cv2.BORDER_REFLECT_101,
    )


def _fit_mask(shape: tuple[int, int], valid: np.ndarray,
              blocks: Sequence[tuple[int, int, int]]) -> np.ndarray:
    support = np.zeros(shape, dtype=bool)
    for y0, x0, parity in blocks:
        if parity == 0:
            ys, xs = slice(y0, y0 + TILE), slice(x0, x0 + TILE)
            support[ys, xs] |= valid[ys, xs]
    return support


def _affine(candidate: np.ndarray, target: np.ndarray,
            support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gain = np.ones(3, dtype=np.float64)
    offset = np.zeros(3, dtype=np.float64)
    for channel in range(3):
        x = candidate[..., channel][support].astype(np.float64)
        y = target[..., channel][support].astype(np.float64)
        if x.size < 16:
            continue
        center_x, center_y = np.median(x), np.median(y)
        variance = float(np.mean((x - center_x) ** 2))
        slope = float(np.mean((x - center_x) * (y - center_y))) / max(variance, 1e-6)
        gain[channel] = np.clip(slope, 0.5, 2.0)
        offset[channel] = np.clip(center_y - gain[channel] * center_x, -64.0, 64.0)
    return gain, offset


def _spatial_scores(candidate: np.ndarray, target: np.ndarray,
                    support: np.ndarray,
                    blocks: Sequence[tuple[int, int, int]]) -> np.ndarray:
    gain, offset = _affine(candidate, target, support)
    adjusted = candidate.astype(np.float32) * gain.astype(np.float32)
    adjusted += offset.astype(np.float32)
    target = target.astype(np.float32)
    radius = 2
    scores = []
    # Registration error is a nuisance competitor, not blur evidence.  Give
    # identity and every action the same nine local displacement hypotheses and
    # compare their best held-out residuals on an interior crop.
    for y0, x0, _ in blocks:
        target_patch = target[
            y0 + radius:y0 + TILE - radius,
            x0 + radius:x0 + TILE - radius,
        ]
        alternatives = []
        for dy in (-radius, 0, radius):
            for dx in (-radius, 0, radius):
                candidate_patch = adjusted[
                    y0 + radius + dy:y0 + TILE - radius + dy,
                    x0 + radius + dx:x0 + TILE - radius + dx,
                ]
                alternatives.append(float(np.median(np.mean(
                    np.abs(target_patch - candidate_patch), axis=2,
                ))) / 255.0)
        scores.append(min(alternatives))
    return np.asarray(scores, dtype=np.float64)


def _identity_scores(observations: Sequence[dict[str, np.ndarray]]) -> np.ndarray:
    output = []
    for item in observations:
        weights = item["weights"]
        output.append(math.sqrt(float(np.sum(
            weights * item["log_empirical"] ** 2
        )) / max(float(weights.sum()), 1e-12)))
    return np.asarray(output, np.float64)


def _robust_se(values: np.ndarray) -> float:
    if values.size < 2:
        return math.inf
    center = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - center)) / math.sqrt(values.size))


def _lcb(values: np.ndarray, z_value: float) -> float:
    if values.size < MIN_BLOCKS_PER_SPLIT:
        return -math.inf
    return float(np.median(values) - z_value * _robust_se(values))


def _parameter_key(parameter: tuple[float, ...]) -> str:
    return ",".join(f"{value:g}" for value in parameter)


def _estimated_parameters(family: str,
                          parameter: tuple[float, ...]) -> dict[str, float]:
    if family == "disk":
        return {"radius": parameter[0]}
    if family == "gaussian":
        return {"sigma": parameter[0]}
    return {"length": parameter[0], "angle_degrees": parameter[1]}


def blur_parameter_certificates(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    *,
    alpha: float = FAMILYWISE_ALPHA,
) -> dict[str, BlurFamilyEvidence]:
    """Fit each blur family on A tiles and certify it on disjoint B tiles."""
    first, second, flow = _validate(first, second, flow)
    warped_second, valid = _warp_second(second, flow)
    z_value = simultaneous_z(alpha, SIMULTANEOUS_HYPOTHESES)
    pairwise_z = simultaneous_z(alpha, 9)
    intermediates: dict[tuple[str, str], dict[str, object]] = {}

    for endpoint in ENDPOINTS:
        sharp, target = (
            (warped_second, first.astype(np.float32)) if endpoint == "first"
            else (first.astype(np.float32), warped_second)
        )
        blocks = _blocks(valid, sharp, target)
        gray_sharp, gray_target = _gray(sharp), _gray(target)
        observations = []
        for y0, x0, parity in blocks:
            observations.append((
                y0, x0, parity,
                _observation(
                    gray_sharp[y0:y0 + TILE, x0:x0 + TILE],
                    gray_target[y0:y0 + TILE, x0:x0 + TILE],
                ),
            ))
        fit_indices = [index for index, row in enumerate(observations) if row[2] == 0]
        check_indices = [index for index, row in enumerate(observations) if row[2] == 1]
        all_observations = [row[3] for row in observations]
        identity = _identity_scores(all_observations)
        fit_mask = _fit_mask(valid.shape, valid, blocks)
        identity_spatial = _spatial_scores(sharp, target, fit_mask, blocks)
        for family in FAMILIES:
            curve: dict[tuple[float, ...], float] = {}
            score_cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
            spatial_cache: dict[tuple[float, ...], np.ndarray] = {}

            def evaluate_parameter(parameter: tuple[float, ...]) -> None:
                if parameter in score_cache:
                    return
                scores, information = _model_scores(all_observations, family, parameter)
                score_cache[parameter] = (scores, information)
                spatial = _spatial_scores(
                    _apply(sharp, family, parameter), target, fit_mask, blocks,
                )
                spatial_cache[parameter] = spatial
                curve[parameter] = float(np.median(spatial[fit_indices])) if fit_indices else math.inf

            for parameter in parameter_grid(family):
                evaluate_parameter(parameter)
            if family == "motion" and curve:
                coarse = min(curve, key=lambda parameter: (
                    curve[parameter], parameter,
                ))
                length, angle = int(round(coarse[0])), int(round(coarse[1]))
                refinements = {
                    (float(candidate_length), float(candidate_angle % 180))
                    for candidate_length in range(max(3, length - 4), min(17, length + 4) + 1, 2)
                    for candidate_angle in (angle - 15, angle, angle + 15)
                }
                for parameter in sorted(refinements):
                    evaluate_parameter(parameter)
            selected = min(curve, key=lambda parameter: (
                curve[parameter],
                float(np.median(score_cache[parameter][0][fit_indices]))
                if fit_indices else math.inf,
                parameter,
            ))
            selected_scores, selected_information = score_cache[selected]
            selected_spatial = spatial_cache[selected]
            # A-fold near-optimal parameters form an uncertainty set; B remains
            # untouched.  The selected parameter is always retained even when
            # robust fit variance vanishes.
            identifiable = []
            selected_fit = selected_spatial[fit_indices]
            for parameter in sorted(curve):
                candidate_fit = spatial_cache[parameter][fit_indices]
                difference = candidate_fit - selected_fit
                if (parameter == selected or not fit_indices
                        or float(np.median(difference))
                        <= z_value * _robust_se(difference)):
                    identifiable.append(parameter)
            check_scores = selected_scores[check_indices]
            check_spatial = selected_spatial[check_indices]
            check_information = selected_information[check_indices]
            own_spectral_delta = identity[check_indices] - check_scores
            own_spatial_delta = identity_spatial[check_indices] - check_spatial
            own_spectral_lcb = _lcb(own_spectral_delta, z_value)
            own_spatial_lcb = _lcb(own_spatial_delta, z_value)
            regions = tuple(
                (observations[index][0], observations[index][1])
                for local, index in enumerate(check_indices)
                if own_spatial_delta[local] > 0.0
                and check_information[local] >= MIN_INFORMATION_RETENTION
            )
            intermediates[(family, endpoint)] = {
                "blocks": blocks,
                "fit_indices": fit_indices,
                "check_indices": check_indices,
                "curve": curve,
                "selected": selected,
                "identifiable": tuple(identifiable),
                "scores": selected_scores,
                "spatial_scores": selected_spatial,
                "information": selected_information,
                "identity": identity,
                "identity_spatial": identity_spatial,
                "own_spectral_delta": own_spectral_delta,
                "own_spatial_delta": own_spatial_delta,
                "own_spectral_lcb": own_spectral_lcb,
                "own_spatial_lcb": own_spatial_lcb,
                "regions": regions,
            }

    output: dict[str, BlurFamilyEvidence] = {}
    for family in FAMILIES:
        for endpoint in ENDPOINTS:
            item = intermediates[(family, endpoint)]
            check_indices = item["check_indices"]
            pairwise: dict[str, float] = {}
            own_scores = item["scores"][check_indices]
            own_spatial = item["spatial_scores"][check_indices]
            for competitor in FAMILIES:
                if competitor == family:
                    continue
                other = intermediates[(competitor, endpoint)]
                # Positive means this action has lower held-out spectral error.
                spectral_delta = other["scores"][check_indices] - own_scores
                spatial_delta = other["spatial_scores"][check_indices] - own_spatial
                pairwise[f"spectral_vs_{competitor}"] = _lcb(
                    spectral_delta, pairwise_z,
                )
                pairwise[f"spatial_vs_{competitor}"] = _lcb(
                    spatial_delta, pairwise_z,
                )
            other_endpoint = "second" if endpoint == "first" else "first"
            endpoint_competitor = intermediates[(family, other_endpoint)]
            endpoint_delta = (
                endpoint_competitor["spatial_scores"][check_indices] - own_spatial
            )
            endpoint_lcb = _lcb(endpoint_delta, pairwise_z)
            check_information = item["information"][check_indices]
            information_median = (
                float(np.median(check_information)) if len(check_indices) else 0.0
            )
            own_spatial_lcb = float(item["own_spatial_lcb"])
            own_spectral_lcb = float(item["own_spectral_lcb"])
            enough = len(check_indices) >= MIN_BLOCKS_PER_SPLIT
            status = "supported" if enough and own_spatial_lcb > 0.0 else "rejected"
            recoverability = (
                "supported" if len(item["regions"]) >= MIN_BLOCKS_PER_SPLIT
                and information_median >= MIN_INFORMATION_RETENTION
                else "unsupported"
            )
            modified_endpoint = "second" if endpoint == "first" else "first"
            action = ActionSpec(
                operator_id=f"common_{family}",
                operator_version="v1-independent-ab-spectral-fit",
                domain="image", hypothesized_degraded_endpoint=endpoint,
                modified_endpoint=modified_endpoint, coordinate_frame="flow_native",
            )
            reasons = []
            if not enough:
                reasons.append("insufficient_independent_check_blocks")
            if own_spatial_lcb <= 0.0:
                reasons.append("simultaneous_spatial_own_null_not_rejected")
            selected = item["selected"]
            uncertainty = max(
                np.linalg.norm(np.asarray(parameter) - np.asarray(selected))
                for parameter in item["identifiable"]
            )
            certificate = PhysicalCertificate(
                action=action, status=status,
                observation_support_fraction=float(len(item["blocks"]) * TILE * TILE
                                                   / max(first.shape[0] * first.shape[1], 1)),
                identifiable_support_fraction=float(len(item["regions"])
                                                    / max(len(check_indices), 1)),
                null_score=float(np.median(item["identity_spatial"][check_indices]))
                if len(check_indices) else 0.0,
                action_score=float(np.median(own_spatial)) if len(check_indices) else 0.0,
                spatial_holdout_gain=float(np.median(item["own_spatial_delta"]))
                if len(check_indices) else 0.0,
                parameter_uncertainty=float(uncertainty),
                fit_stability=float(np.mean(item["own_spatial_delta"] > 0.0))
                if len(check_indices) else 0.0,
                competing_model_scores={
                    f"pairwise_lcb_vs_{name}": float(value)
                    for name, value in pairwise.items()
                } | {"endpoint_margin_lcb": endpoint_lcb},
                estimated_parameters=_estimated_parameters(family, selected),
                diagnostics={
                    "fit_blocks": float(len(item["fit_indices"])),
                    "check_blocks": float(len(check_indices)),
                    "simultaneous_z": z_value,
                    "familywise_alpha": float(alpha),
                    "hypotheses": float(SIMULTANEOUS_HYPOTHESES),
                    "information_retention_median": information_median,
                    "recoverable_regions": float(len(item["regions"])),
                    "spectral_own_null_lcb": own_spectral_lcb,
                    "geometry_nuisance_offsets": 9.0,
                    "pairwise_simultaneous_z": pairwise_z,
                    "minimum_endpoint_gain": MIN_ENDPOINT_GAIN,
                },
                rejection_reasons=tuple(reasons),
                calibration_version="blur-spatial-forward-ab-geometry9-bonferroni-6-v1",
            )
            evidence = BlurFamilyEvidence(
                family=family, endpoint=endpoint,
                selected_parameter=selected,
                identifiable_parameters=item["identifiable"],
                fit_score_curve={
                    _parameter_key(parameter): float(score)
                    for parameter, score in item["curve"].items()
                },
                fit_support_hash=_support_hash(endpoint, 0, item["blocks"]),
                check_support_hash=_support_hash(endpoint, 1, item["blocks"]),
                check_blocks=len(check_indices),
                own_null_improvement_median=float(np.median(item["own_spatial_delta"]))
                if len(check_indices) else 0.0,
                own_null_improvement_lcb=own_spatial_lcb,
                own_null_spectral_improvement_lcb=own_spectral_lcb,
                pairwise_improvement_lcb=pairwise,
                endpoint_improvement_lcb=endpoint_lcb,
                information_retention_median=information_median,
                recoverable_regions=item["regions"],
                recoverability_status=recoverability,
                certificate=certificate,
            )
            output[evidence.action_key] = evidence
    return output
