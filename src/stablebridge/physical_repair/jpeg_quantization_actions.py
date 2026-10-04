"""Quantization-consistent JPEG deblocking by alternating projections.

The fixed two-pixel interpolation used by the first JPEG action audit reduced
block boundaries but usually left the observed JPEG quantization cell.  This
successor alternates a deterministic boundary-smoothing projection with a DCT
quantization-interval projection.  It is an interpretable codec action, not a
learned restoration network and not a claim that lost texture is recovered.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import math
from statistics import NormalDist
import time

import cv2
import numpy as np
from PIL import Image

from .contracts import ActionSpec, PhysicalCertificate
from .jpeg_action_certificates import (
    BLOCK,
    CODEC_NONINFERIORITY_255,
    FAMILYWISE_ALPHA,
    MACRO_TILE,
    MIN_CHECK_TILES,
    _boundary_excess,
    _bounds,
    _macro_tiles,
)


DCT_ROUNDING_BOUND = 4.0
MAX_PROJECTION_ITERATIONS = 64
CONVERGENCE_255 = 0.05

_JPEG_LUMA = np.asarray([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)

_DCT = np.asarray([
    [
        (np.sqrt(1.0 / BLOCK) if frequency == 0 else np.sqrt(2.0 / BLOCK))
        * np.cos(np.pi * (2 * position + 1) * frequency / (2 * BLOCK))
        for position in range(BLOCK)
    ]
    for frequency in range(BLOCK)
], dtype=np.float32)


def jpeg_luma_quant_table(quality: int) -> np.ndarray:
    if type(quality) is not int or not 1 <= quality <= 95:
        raise ValueError("estimated IJG quality must be an integer in [1,95]")
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    table = np.floor((_JPEG_LUMA * scale + 50.0) / 100.0)
    return np.clip(table, 1.0, 255.0).astype(np.float32)


def _block_view(plane: np.ndarray) -> tuple[np.ndarray, int, int]:
    height = (plane.shape[0] // BLOCK) * BLOCK
    width = (plane.shape[1] // BLOCK) * BLOCK
    blocks = (
        plane[:height, :width]
        .reshape(height // BLOCK, BLOCK, width // BLOCK, BLOCK)
        .transpose(0, 2, 1, 3)
        .reshape(-1, BLOCK, BLOCK)
    )
    return np.ascontiguousarray(blocks, dtype=np.float32), height, width


def _unblock(blocks: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.ascontiguousarray(
        blocks.reshape(height // BLOCK, width // BLOCK, BLOCK, BLOCK)
        .transpose(0, 2, 1, 3)
        .reshape(height, width)
    )


def _forward_dct(blocks: np.ndarray) -> np.ndarray:
    return np.einsum("ui,nij,vj->nuv", _DCT, blocks - 128.0, _DCT, optimize=True)


def _inverse_dct(coefficients: np.ndarray) -> np.ndarray:
    return np.einsum(
        "ui,nuv,vj->nij", _DCT, coefficients, _DCT, optimize=True,
    ) + 128.0


def _smooth_boundaries(plane: np.ndarray) -> np.ndarray:
    value = np.asarray(plane, dtype=np.float32).copy()
    height, width = value.shape
    for boundary in range(BLOCK, width, BLOCK):
        if boundary < 2 or boundary + 1 >= width:
            continue
        left, right = value[:, boundary - 2].copy(), value[:, boundary + 1].copy()
        value[:, boundary - 1] = (2.0 * left + right) / 3.0
        value[:, boundary] = (left + 2.0 * right) / 3.0
    for boundary in range(BLOCK, height, BLOCK):
        if boundary < 2 or boundary + 1 >= height:
            continue
        top, bottom = value[boundary - 2].copy(), value[boundary + 1].copy()
        value[boundary - 1] = (2.0 * top + bottom) / 3.0
        value[boundary] = (top + 2.0 * bottom) / 3.0
    return value


@dataclass(frozen=True)
class JPEGQuantizationAction:
    image: np.ndarray
    footprint: np.ndarray
    iterations: int
    quantization_violation_max: float


def jpeg_quantization_projected_deblock(
    image: np.ndarray,
    *,
    estimated_quality: int,
) -> JPEGQuantizationAction:
    """Alternate boundary smoothing and luma quantization-cell projection."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected uint8 RGB")
    table = jpeg_luma_quant_table(estimated_quality)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    observed = ycrcb[..., 0].astype(np.float32)
    observed_blocks, height, width = _block_view(observed)
    observed_coefficients = _forward_dct(observed_blocks)
    centers = np.rint(observed_coefficients / table[None]) * table[None]
    # The closed quantization cell is the codec-consistent set.  Pixel/channel
    # rounding is accounted for after RGB realization by the deterministic
    # DCT_ROUNDING_BOUND, rather than by shrinking small cells until they are
    # empty.
    lower = centers - 0.5 * table[None]
    upper = centers + 0.5 * table[None]
    rgb = image[:height, :width].astype(np.float32)
    feasible_lower = -np.min(rgb, axis=2)
    feasible_upper = 255.0 - np.max(rgb, axis=2)
    luma_lower = observed[:height, :width] + feasible_lower
    luma_upper = observed[:height, :width] + feasible_upper
    # Boundary smoothing proposes a point.  The loop then alternates only the
    # two convex feasibility projections: JPEG DCT cells and RGB-gamut luma.
    # Reapplying smoothing inside the loop would prevent POCS convergence.
    current = _smooth_boundaries(observed[:height, :width])
    iterations = 0
    for iteration in range(1, MAX_PROJECTION_ITERATIONS + 1):
        proposal_blocks, _, _ = _block_view(current)
        coefficients = np.clip(_forward_dct(proposal_blocks), lower, upper)
        projected = np.clip(_unblock(_inverse_dct(coefficients), height, width), 0.0, 255.0)
        projected = np.clip(projected, luma_lower, luma_upper)
        change = float(np.max(np.abs(projected - current)))
        current = projected
        iterations = iteration
        if change <= CONVERGENCE_255:
            break
    # A direct YCrCb->RGB conversion can leave the projected luma cell when
    # the fixed chroma is outside the RGB gamut.  Real deployment stores RGB,
    # so that conversion is part of the action rather than harmless I/O.  Move
    # all three RGB channels by the same amount instead: this preserves chroma,
    # changes luma by the same amount, and exposes the exact per-pixel gamut
    # interval before rounding.  Clipped/saturated pixels naturally fall back
    # toward the observed value instead of silently invalidating the DCT proof.
    requested_delta = current - observed[:height, :width]
    realized_delta = np.clip(requested_delta, feasible_lower, feasible_upper)
    output = image.copy()
    output[:height, :width] = np.clip(
        np.rint(rgb + realized_delta[..., None]), 0, 255,
    ).astype(np.uint8)
    footprint = np.zeros(image.shape[:2], dtype=bool)
    footprint[:height, :width] = True
    output[~footprint] = image[~footprint]
    output = np.ascontiguousarray(output)
    realized = cv2.cvtColor(output, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
    realized_blocks, _, _ = _block_view(realized)
    realized_coefficients = _forward_dct(realized_blocks)
    violation = np.maximum(lower - realized_coefficients, 0.0)
    violation = np.maximum(violation, realized_coefficients - upper)
    return JPEGQuantizationAction(
        image=output, footprint=footprint, iterations=iterations,
        quantization_violation_max=float(np.max(violation)),
    )


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="JPEG", quality=quality, subsampling=2)
    stream.seek(0)
    return np.ascontiguousarray(
        np.array(Image.open(stream).convert("RGB"), dtype=np.uint8, copy=True)
    )


def jpeg_quantization_action_certificate(
    image: np.ndarray,
    *,
    endpoint: str,
    estimated_quality: int,
    presence_lcb: float,
    alpha: float = FAMILYWISE_ALPHA,
) -> PhysicalCertificate:
    """Certify the projected action on held-out tiles without task outcome."""
    started = time.perf_counter()
    image = np.asarray(image)
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 2 * MACRO_TILE):
        raise ValueError("expected sufficiently large uint8 RGB")
    if endpoint not in {"first", "second"}:
        raise ValueError("JPEG efficacy needs a concrete endpoint")
    if not np.isfinite(presence_lcb) or not 0.0 < alpha < 1.0:
        raise ValueError("invalid JPEG presence bound or alpha")
    result = jpeg_quantization_projected_deblock(
        image, estimated_quality=estimated_quality,
    )
    candidate = result.image
    changed = np.any(candidate != image, axis=2)
    outside_exact = bool(np.array_equal(candidate[~result.footprint], image[~result.footprint]))
    observed_reencoded = _jpeg_roundtrip(image, estimated_quality)
    candidate_reencoded = _jpeg_roundtrip(candidate, estimated_quality)
    observed_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tiles = _macro_tiles(image.shape[:2], parity=1)
    gains, closure_excess, before_values, after_values = [], [], [], []
    for ys, xs in tiles:
        before = _boundary_excess(observed_gray[ys, xs])
        after = _boundary_excess(candidate_gray[ys, xs])
        before_values.append(before)
        after_values.append(after)
        gains.append(before - after)
        baseline_error = float(np.mean(np.abs(
            observed_reencoded[ys, xs].astype(np.float32)
            - image[ys, xs].astype(np.float32)
        )))
        candidate_error = float(np.mean(np.abs(
            candidate_reencoded[ys, xs].astype(np.float32)
            - image[ys, xs].astype(np.float32)
        )))
        closure_excess.append(candidate_error - baseline_error)
    gains = np.asarray(gains, dtype=np.float64)
    closure_excess = np.asarray(closure_excess, dtype=np.float64)
    z_value = float(NormalDist().inv_cdf(1.0 - alpha / (2.0 * 5.0)))
    boundary_lcb, _ = _bounds(gains, z_value)
    _, closure_ub = _bounds(closure_excess, z_value)
    reasons = []
    if presence_lcb <= 0.0:
        reasons.append("jpeg_presence_not_supported")
    if len(tiles) < MIN_CHECK_TILES:
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
            operator_version="v3-quantization-projected-pocs",
            domain="image", hypothesized_degraded_endpoint=endpoint,
            modified_endpoint=endpoint, coordinate_frame=f"{endpoint}_native",
        ),
        status="supported" if not reasons else "rejected",
        observation_support_fraction=1.0,
        identifiable_support_fraction=float(changed.mean()),
        null_score=max(before_median / 255.0, 0.0),
        action_score=max(after_median / 255.0, 0.0),
        spatial_holdout_gain=float(np.median(gains) / 255.0) if len(gains) else 0.0,
        parameter_uncertainty=float(1.0 / math.sqrt(max(len(tiles), 1))),
        fit_stability=float(np.mean(gains > 0.0)) if len(gains) else 0.0,
        competing_model_scores={
            "jpeg_presence_lcb": float(presence_lcb),
            "boundary_reduction_lcb_255": float(boundary_lcb),
            "codec_closure_excess_ub_255": float(closure_ub),
            "quantization_violation_max_dct": result.quantization_violation_max,
        },
        estimated_parameters={
            "ijg_quality": float(estimated_quality), "block": float(BLOCK),
            "projection_iterations": float(result.iterations),
        },
        diagnostics={
            "check_tiles": float(len(tiles)), "simultaneous_z": z_value,
            "changed_fraction": float(changed.mean()),
            "declared_footprint_fraction": float(result.footprint.mean()),
            "outside_footprint_exact": float(outside_exact),
            "codec_noninferiority_margin_255": CODEC_NONINFERIORITY_255,
            "dct_rounding_bound": DCT_ROUNDING_BOUND,
        },
        rejection_reasons=tuple(reasons),
        calibration_version="jpeg-action-qcell-pocs-ab-closure-v3",
        measured_probe_seconds=time.perf_counter() - started,
    )
