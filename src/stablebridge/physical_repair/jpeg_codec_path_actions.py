"""Closure-adaptive JPEG repair on a fixed convex intervention path.

E146 showed that a full quantization-cell projection is physically valid in
luma but often too strong for the complete RGB/YCbCr/re-encode pipeline.  This
successor does not add an operator or relax closure.  It treats action strength
as an identifiable parameter: even macro-tiles select the first member of a
fixed dyadic native-to-repair path that satisfies codec closure, while odd
macro-tiles independently certify the selected candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
import time

import cv2
import numpy as np

from .contracts import ActionSpec, PhysicalCertificate
from .jpeg_action_certificates import (
    CODEC_NONINFERIORITY_255,
    FAMILYWISE_ALPHA,
    MACRO_TILE,
    MIN_CHECK_TILES,
    _boundary_excess,
    _bounds,
    _macro_tiles,
)
from .jpeg_quantization_actions import (
    BLOCK,
    DCT_ROUNDING_BOUND,
    _block_view,
    _forward_dct,
    _jpeg_roundtrip,
    jpeg_luma_quant_table,
    jpeg_quantization_projected_deblock,
)


DYADIC_ACTION_STRENGTHS = (1.0, 0.5, 0.25, 0.125, 0.0625)


@dataclass(frozen=True)
class JPEGCodecPathAction:
    image: np.ndarray
    footprint: np.ndarray
    strength: float
    attempted_strengths: tuple[float, ...]
    quantization_violation_max: float


def _candidate(image: np.ndarray, proposal: np.ndarray, strength: float) -> np.ndarray:
    value = image.astype(np.float32) + strength * (
        proposal.astype(np.float32) - image.astype(np.float32)
    )
    return np.ascontiguousarray(np.clip(np.rint(value), 0, 255).astype(np.uint8))


def _quantization_violation(
    observed: np.ndarray,
    candidate: np.ndarray,
    estimated_quality: int,
) -> float:
    table = jpeg_luma_quant_table(estimated_quality)
    observed_y = cv2.cvtColor(observed, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
    candidate_y = cv2.cvtColor(candidate, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
    observed_blocks, _, _ = _block_view(observed_y)
    candidate_blocks, _, _ = _block_view(candidate_y)
    centers = np.rint(_forward_dct(observed_blocks) / table[None]) * table[None]
    lower, upper = centers - 0.5 * table[None], centers + 0.5 * table[None]
    coefficients = _forward_dct(candidate_blocks)
    violation = np.maximum(lower - coefficients, 0.0)
    violation = np.maximum(violation, coefficients - upper)
    return float(np.max(violation))


def _tile_statistics(
    image: np.ndarray,
    candidate: np.ndarray,
    reencoded_observed: np.ndarray,
    reencoded_candidate: np.ndarray,
    *,
    parity: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    observed_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gains, closure, before_values, after_values = [], [], [], []
    for ys, xs in _macro_tiles(image.shape[:2], parity=parity):
        before = _boundary_excess(observed_gray[ys, xs])
        after = _boundary_excess(candidate_gray[ys, xs])
        before_values.append(before)
        after_values.append(after)
        gains.append(before - after)
        baseline_error = float(np.mean(np.abs(
            reencoded_observed[ys, xs].astype(np.float32)
            - image[ys, xs].astype(np.float32)
        )))
        candidate_error = float(np.mean(np.abs(
            reencoded_candidate[ys, xs].astype(np.float32)
            - image[ys, xs].astype(np.float32)
        )))
        closure.append(candidate_error - baseline_error)
    return (
        np.asarray(gains, dtype=np.float64),
        np.asarray(closure, dtype=np.float64),
        before_values,
        after_values,
    )


def jpeg_codec_path_action(
    image: np.ndarray,
    *,
    estimated_quality: int,
) -> JPEGCodecPathAction:
    """Select action strength on even tiles; reserve odd tiles for certification."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected uint8 RGB")
    proposal = jpeg_quantization_projected_deblock(
        image, estimated_quality=estimated_quality,
    )
    observed_reencoded = _jpeg_roundtrip(image, estimated_quality)
    fit_z = float(NormalDist().inv_cdf(
        1.0 - FAMILYWISE_ALPHA / (2.0 * len(DYADIC_ACTION_STRENGTHS))
    ))
    attempted = []
    selected = image.copy()
    selected_strength = 0.0
    selected_violation = 0.0
    for strength in DYADIC_ACTION_STRENGTHS:
        attempted.append(strength)
        candidate = _candidate(image, proposal.image, strength)
        if np.array_equal(candidate, image):
            continue
        violation = _quantization_violation(image, candidate, estimated_quality)
        if violation > DCT_ROUNDING_BOUND:
            continue
        reencoded = _jpeg_roundtrip(candidate, estimated_quality)
        gains, closure, _, _ = _tile_statistics(
            image, candidate, observed_reencoded, reencoded, parity=0,
        )
        boundary_lcb, _ = _bounds(gains, fit_z)
        # Closure is a codec invariant rather than an average visual score.
        # Require every fit tile to stay inside the same one-level allowance;
        # the independent check fold still uses a simultaneous upper bound.
        closure_ub = float(np.max(closure)) if len(closure) else float("inf")
        if (len(gains) >= MIN_CHECK_TILES
                and boundary_lcb > 0.0
                and closure_ub <= CODEC_NONINFERIORITY_255):
            selected = candidate
            selected_strength = strength
            selected_violation = violation
            break
    footprint = proposal.footprint.copy()
    selected[~footprint] = image[~footprint]
    return JPEGCodecPathAction(
        image=np.ascontiguousarray(selected), footprint=footprint,
        strength=float(selected_strength), attempted_strengths=tuple(attempted),
        quantization_violation_max=float(selected_violation),
    )


def jpeg_codec_path_action_certificate(
    image: np.ndarray,
    *,
    endpoint: str,
    estimated_quality: int,
    presence_lcb: float,
    alpha: float = FAMILYWISE_ALPHA,
) -> PhysicalCertificate:
    """Independently certify the A-fold-selected codec-path action on B tiles."""
    started = time.perf_counter()
    image = np.asarray(image)
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 2 * MACRO_TILE):
        raise ValueError("expected sufficiently large uint8 RGB")
    if endpoint not in {"first", "second"}:
        raise ValueError("JPEG efficacy needs a concrete endpoint")
    if not np.isfinite(presence_lcb) or not 0.0 < alpha < 1.0:
        raise ValueError("invalid JPEG presence bound or alpha")
    result = jpeg_codec_path_action(image, estimated_quality=estimated_quality)
    candidate = result.image
    changed = np.any(candidate != image, axis=2)
    outside_exact = bool(np.array_equal(candidate[~result.footprint], image[~result.footprint]))
    observed_reencoded = _jpeg_roundtrip(image, estimated_quality)
    candidate_reencoded = _jpeg_roundtrip(candidate, estimated_quality)
    gains, closure, before_values, after_values = _tile_statistics(
        image, candidate, observed_reencoded, candidate_reencoded, parity=1,
    )
    z_value = float(NormalDist().inv_cdf(1.0 - alpha / (2.0 * 5.0)))
    boundary_lcb, _ = _bounds(gains, z_value)
    _, closure_ub = _bounds(closure, z_value)
    reasons = []
    if presence_lcb <= 0.0:
        reasons.append("jpeg_presence_not_supported")
    if result.strength <= 0.0:
        reasons.append("no_fit_fold_codec_feasible_strength")
    if len(gains) < MIN_CHECK_TILES:
        reasons.append("insufficient_independent_check_tiles")
    if boundary_lcb <= 0.0:
        reasons.append("no_heldout_boundary_excess_reduction")
    if closure_ub > CODEC_NONINFERIORITY_255:
        reasons.append("codec_closure_not_noninferior")
    if result.quantization_violation_max > DCT_ROUNDING_BOUND:
        reasons.append("quantization_interval_violation")
    if not outside_exact:
        reasons.append("action_changed_outside_declared_footprint")
    if not np.any(changed):
        reasons.append("action_has_no_effect")
    before_median = float(np.median(before_values)) if before_values else 0.0
    after_median = float(np.median(after_values)) if after_values else 0.0
    return PhysicalCertificate(
        action=ActionSpec(
            operator_id="jpeg_deblock",
            operator_version="v4-codec-path-ab",
            domain="image", hypothesized_degraded_endpoint=endpoint,
            modified_endpoint=endpoint, coordinate_frame=f"{endpoint}_native",
        ),
        status="supported" if not reasons else "rejected",
        observation_support_fraction=1.0,
        identifiable_support_fraction=float(changed.mean()),
        null_score=max(before_median / 255.0, 0.0),
        action_score=max(after_median / 255.0, 0.0),
        spatial_holdout_gain=float(np.median(gains) / 255.0) if len(gains) else 0.0,
        parameter_uncertainty=float(1.0 / math.sqrt(max(len(gains), 1))),
        fit_stability=float(np.mean(gains > 0.0)) if len(gains) else 0.0,
        competing_model_scores={
            "jpeg_presence_lcb": float(presence_lcb),
            "boundary_reduction_lcb_255": float(boundary_lcb),
            "codec_closure_excess_ub_255": float(closure_ub),
            "quantization_violation_max_dct": result.quantization_violation_max,
        },
        estimated_parameters={
            "ijg_quality": float(estimated_quality), "block": float(BLOCK),
            "action_strength": result.strength,
            "attempted_strengths": float(len(result.attempted_strengths)),
        },
        diagnostics={
            "check_tiles": float(len(gains)), "simultaneous_z": z_value,
            "changed_fraction": float(changed.mean()),
            "declared_footprint_fraction": float(result.footprint.mean()),
            "outside_footprint_exact": float(outside_exact),
            "codec_noninferiority_margin_255": CODEC_NONINFERIORITY_255,
            "dct_rounding_bound": DCT_ROUNDING_BOUND,
        },
        rejection_reasons=tuple(reasons),
        calibration_version="jpeg-codec-path-ab-closure-v4",
        measured_probe_seconds=time.perf_counter() - started,
    )
