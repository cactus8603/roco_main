"""Interpretable certificates for sparse impulse and locally additive noise."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .contracts import ActionSpec, PhysicalCertificate
from .operators import impulse_support


def _action(operator_id: str, endpoint: str, version: str) -> ActionSpec:
    return ActionSpec(
        operator_id=operator_id, operator_version=version, domain="image",
        hypothesized_degraded_endpoint=endpoint, modified_endpoint=endpoint,
        coordinate_frame=f"{endpoint}_native", execution_semantics="from_zero",
    )


def impulse_noise_certificate(image: np.ndarray, *, endpoint: str,
                              minimum_pixels: int = 32,
                              minimum_fraction: float = 1e-4,
                              maximum_fraction: float = 0.15) -> PhysicalCertificate:
    """Require sparse extrema that disagree with their local median.

    The rule rejects large saturated areas and extrema that are also supported
    by the 3x3 neighbourhood.  Thresholds express sparsity/count assumptions,
    not a corruption-family score fitted to task outcomes.
    """
    started = time.perf_counter()
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected uint8 RGB")
    if endpoint not in {"first", "second"}:
        raise ValueError("impulse certificate needs a concrete endpoint")
    exact = impulse_support(image) > 0.0
    count, fraction = int(exact.sum()), float(exact.mean())
    raw_extreme = np.all(image <= 1, axis=2) | np.all(image >= 254, axis=2)
    raw_fraction = float(raw_extreme.mean())
    reasons = []
    if count < minimum_pixels or fraction < minimum_fraction:
        reasons.append("insufficient_sparse_extrema")
    if fraction > maximum_fraction:
        reasons.append("extrema_not_sparse")
    status = "supported" if not reasons else "rejected"
    return PhysicalCertificate(
        action=_action("impulse_exact_median3", endpoint, "v1-sparse-local-median"),
        status=status, observation_support_fraction=1.0,
        identifiable_support_fraction=fraction,
        null_score=raw_fraction, action_score=fraction,
        spatial_holdout_gain=max(raw_fraction - fraction, 0.0),
        parameter_uncertainty=float(1.0 / math.sqrt(max(count, 1))),
        fit_stability=float(np.clip(count / max(4 * minimum_pixels, 1), 0.0, 1.0)),
        competing_model_scores={"raw_saturation_fraction": raw_fraction},
        estimated_parameters={"impulse_density": fraction},
        diagnostics={"detected_pixels": float(count)},
        rejection_reasons=tuple(reasons), calibration_version="physical-sparsity-v1",
        measured_probe_seconds=time.perf_counter() - started,
    )


def _tile_noise(image: np.ndarray, tile_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = gray.shape
    rows, columns = math.ceil(height / tile_size), math.ceil(width / tile_size)
    values = np.zeros((rows, columns), dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            tile = gray[row * tile_size:min((row + 1) * tile_size, height),
                        column * tile_size:min((column + 1) * tile_size, width)]
            lap = cv2.Laplacian(tile, cv2.CV_32F, ksize=1,
                                borderType=cv2.BORDER_REFLECT_101)
            center = float(np.median(lap))
            values[row, column] = float(
                np.median(np.abs(lap - center)) / 0.6744897501960817 / math.sqrt(20.0)
            )
    return values, (height, width)


def additive_noise_certificate(image: np.ndarray, *, endpoint: str,
                               tile_size: int = 32, robust_z: float = 4.0,
                               absolute_sigma_255: float = 2.0,
                               minimum_connected_tiles: int = 2) -> PhysicalCertificate:
    """Detect a spatially coherent excess in a robust additive-noise estimate."""
    started = time.perf_counter()
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected uint8 RGB")
    if endpoint not in {"first", "second"}:
        raise ValueError("additive certificate needs a concrete endpoint")
    tile_sigma, shape = _tile_noise(image, tile_size)
    baseline = float(np.median(tile_sigma))
    mad = float(np.median(np.abs(tile_sigma - baseline)))
    robust_scale = max(1.482602218505602 * mad, 0.5)
    z = (tile_sigma - baseline) / robust_scale
    active = (z >= robust_z) & (tile_sigma >= absolute_sigma_255)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(
        active.astype(np.uint8), connectivity=8,
    )
    largest = int(stats[1:, cv2.CC_STAT_AREA].max(initial=0)) if components > 1 else 0
    active_count = int(active.sum())
    active_fraction = float(active.mean())
    coherence = float(largest / max(active_count, 1))
    reasons = []
    if largest < minimum_connected_tiles:
        reasons.append("no_spatially_coherent_noise_excess")
    if active_fraction > 0.80:
        reasons.append("no_clean_within_image_baseline")
    status = "supported" if not reasons else "rejected"
    return PhysicalCertificate(
        action=_action("wiener3", endpoint, "v1-robust-tile-excess"),
        status=status, observation_support_fraction=1.0,
        identifiable_support_fraction=active_fraction,
        null_score=baseline, action_score=float(tile_sigma[active].mean()
                                                if np.any(active) else baseline),
        spatial_holdout_gain=max(float(tile_sigma.max(initial=baseline)) - baseline, 0.0),
        parameter_uncertainty=float(1.0 / math.sqrt(max(active_count, 1))),
        fit_stability=coherence,
        competing_model_scores={"within_image_baseline_sigma_255": baseline},
        estimated_parameters={
            "active_mean_sigma_255": float(tile_sigma[active].mean()
                                           if np.any(active) else baseline),
            "robust_scale_sigma_255": robust_scale,
        },
        diagnostics={
            "active_tile_fraction": active_fraction,
            "active_tile_count": float(active_count),
            "largest_connected_tiles": float(largest),
            "largest_component_fraction": coherence,
            "maximum_robust_z": float(z.max(initial=0.0)),
        },
        rejection_reasons=tuple(reasons), calibration_version="physical-noise-excess-v1",
        measured_probe_seconds=time.perf_counter() - started,
    )


def select_noise_action(first: np.ndarray, second: np.ndarray) -> dict:
    """Fail-closed two-action selector; impulse has priority over additive noise."""
    certificates = {
        "impulse_first": impulse_noise_certificate(first, endpoint="first"),
        "impulse_second": impulse_noise_certificate(second, endpoint="second"),
        "wiener_first": additive_noise_certificate(first, endpoint="first"),
        "wiener_second": additive_noise_certificate(second, endpoint="second"),
    }
    for family, names, operator in (
        ("impulse", ("impulse_first", "impulse_second"), "impulse_exact_median3"),
        ("additive", ("wiener_first", "wiener_second"), "wiener3"),
    ):
        supported = [name.rsplit("_", 1)[1] for name in names
                     if certificates[name].status == "supported"]
        if supported:
            endpoint = "both" if len(supported) == 2 else supported[0]
            return {"action": operator, "endpoint": endpoint, "family": family,
                    "certificates": certificates}
    return {"action": "native", "endpoint": "unknown", "family": "none",
            "certificates": certificates}

