"""Held-out physical efficacy certificate for the fixed JPEG deblock action.

JPEG lattice presence is not evidence that deblocking helps.  This module
checks the action itself: it must reduce aligned 8x8 boundary excess on held-out
macro-tiles, remain in the same codec equivalence class under re-encoding, and
leave every pixel outside its declared two-pixel boundary footprint bit exact.
The caller supplies a label-free JPEG presence margin and estimated IJG quality
from an independent fit split.
"""
from __future__ import annotations

import io
import math
from statistics import NormalDist
import time

import cv2
import numpy as np
from PIL import Image

from .contracts import ActionSpec, PhysicalCertificate


BLOCK = 8
MACRO_TILE = 64
FAMILYWISE_ALPHA = 0.05
MIN_CHECK_TILES = 4
CODEC_NONINFERIORITY_255 = 1.0


def jpeg_block_linear(image: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Apply the fixed, strength-free two-pixel boundary interpolation."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected uint8 RGB")
    value = image.astype(np.float32).copy()
    height, width = value.shape[:2]
    for boundary in range(block, width, block):
        if boundary < 2 or boundary + 1 >= width:
            continue
        left, right = value[:, boundary - 2].copy(), value[:, boundary + 1].copy()
        value[:, boundary - 1] = (2.0 * left + right) / 3.0
        value[:, boundary] = (left + 2.0 * right) / 3.0
    for boundary in range(block, height, block):
        if boundary < 2 or boundary + 1 >= height:
            continue
        top, bottom = value[boundary - 2].copy(), value[boundary + 1].copy()
        value[boundary - 1] = (2.0 * top + bottom) / 3.0
        value[boundary] = (top + 2.0 * bottom) / 3.0
    return np.ascontiguousarray(np.clip(np.rint(value), 0, 255).astype(np.uint8))


def jpeg_action_footprint(shape: tuple[int, int], block: int = BLOCK) -> np.ndarray:
    height, width = shape
    output = np.zeros((height, width), dtype=bool)
    for boundary in range(block, width, block):
        if boundary < 2 or boundary + 1 >= width:
            continue
        output[:, boundary - 1:boundary + 1] = True
    for boundary in range(block, height, block):
        if boundary < 2 or boundary + 1 >= height:
            continue
        output[boundary - 1:boundary + 1, :] = True
    return output


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    if type(quality) is not int or not 1 <= quality <= 95:
        raise ValueError("estimated IJG quality must be an integer in [1,95]")
    stream = io.BytesIO()
    Image.fromarray(image).save(
        stream, format="JPEG", quality=quality, subsampling=2,
    )
    stream.seek(0)
    return np.ascontiguousarray(
        np.array(Image.open(stream).convert("RGB"), dtype=np.uint8, copy=True)
    )


def _macro_tiles(shape: tuple[int, int], parity: int) -> list[tuple[slice, slice]]:
    height, width = shape
    output = []
    for y0 in range(0, height - MACRO_TILE + 1, MACRO_TILE):
        for x0 in range(0, width - MACRO_TILE + 1, MACRO_TILE):
            if (y0 // MACRO_TILE + x0 // MACRO_TILE) % 2 == parity:
                output.append((slice(y0, y0 + MACRO_TILE),
                               slice(x0, x0 + MACRO_TILE)))
    return output


def _boundary_excess(gray: np.ndarray) -> float:
    values = []
    for boundary in range(BLOCK, gray.shape[1], BLOCK):
        if boundary < 2 or boundary + 1 >= gray.shape[1]:
            continue
        edge = np.abs(gray[:, boundary - 1] - gray[:, boundary])
        flank = 0.5 * (
            np.abs(gray[:, boundary - 2] - gray[:, boundary - 1])
            + np.abs(gray[:, boundary] - gray[:, boundary + 1])
        )
        values.append(edge - flank)
    for boundary in range(BLOCK, gray.shape[0], BLOCK):
        if boundary < 2 or boundary + 1 >= gray.shape[0]:
            continue
        edge = np.abs(gray[boundary - 1] - gray[boundary])
        flank = 0.5 * (
            np.abs(gray[boundary - 2] - gray[boundary - 1])
            + np.abs(gray[boundary] - gray[boundary + 1])
        )
        values.append(edge - flank)
    if not values:
        return 0.0
    return float(np.mean(np.concatenate([value.ravel() for value in values])))


def _robust_se(values: np.ndarray) -> float:
    if values.size < 2:
        return math.inf
    center = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - center)) / math.sqrt(values.size))


def _bounds(values: np.ndarray, z_value: float) -> tuple[float, float]:
    if values.size < MIN_CHECK_TILES:
        return -math.inf, math.inf
    center, radius = float(np.median(values)), z_value * _robust_se(values)
    return center - radius, center + radius


def jpeg_deblock_action_certificate(
    image: np.ndarray,
    *,
    endpoint: str,
    estimated_quality: int,
    presence_lcb: float,
    candidate: np.ndarray | None = None,
    alpha: float = FAMILYWISE_ALPHA,
) -> PhysicalCertificate:
    """Certify a fixed JPEG action using independent presence evidence."""
    started = time.perf_counter()
    image = np.asarray(image)
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 2 * MACRO_TILE):
        raise ValueError("expected sufficiently large uint8 RGB")
    if endpoint not in {"first", "second"}:
        raise ValueError("JPEG efficacy needs a concrete endpoint")
    if not np.isfinite(presence_lcb):
        raise ValueError("presence LCB must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    candidate = jpeg_block_linear(image) if candidate is None else np.asarray(candidate)
    if candidate.dtype != np.uint8 or candidate.shape != image.shape:
        raise ValueError("candidate must match the observed RGB image")
    candidate = np.ascontiguousarray(candidate)
    footprint = jpeg_action_footprint(image.shape[:2])
    changed = np.any(candidate != image, axis=2)
    outside_exact = bool(np.array_equal(candidate[~footprint], image[~footprint]))

    observed_reencoded = _jpeg_roundtrip(image, estimated_quality)
    candidate_reencoded = _jpeg_roundtrip(candidate, estimated_quality)
    observed_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tiles = _macro_tiles(image.shape[:2], parity=1)
    boundary_gains, closure_excess = [], []
    before_values, after_values = [], []
    for ys, xs in tiles:
        before = _boundary_excess(observed_gray[ys, xs])
        after = _boundary_excess(candidate_gray[ys, xs])
        before_values.append(before)
        after_values.append(after)
        boundary_gains.append(before - after)
        baseline_error = float(np.mean(np.abs(
            observed_reencoded[ys, xs].astype(np.float32)
            - image[ys, xs].astype(np.float32)
        )))
        candidate_error = float(np.mean(np.abs(
            candidate_reencoded[ys, xs].astype(np.float32)
            - image[ys, xs].astype(np.float32)
        )))
        closure_excess.append(candidate_error - baseline_error)
    boundary_gains = np.asarray(boundary_gains, dtype=np.float64)
    closure_excess = np.asarray(closure_excess, dtype=np.float64)
    z_value = float(NormalDist().inv_cdf(1.0 - alpha / (2.0 * 4.0)))
    boundary_lcb, _ = _bounds(boundary_gains, z_value)
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
    if not outside_exact:
        reasons.append("action_changed_outside_declared_footprint")
    if not np.any(changed):
        reasons.append("action_has_no_effect")
    status = "supported" if not reasons else "rejected"
    before_median = float(np.median(before_values)) if before_values else 0.0
    after_median = float(np.median(after_values)) if after_values else 0.0
    return PhysicalCertificate(
        action=ActionSpec(
            operator_id="jpeg_deblock", operator_version="v2-heldout-codec-closure",
            domain="image", hypothesized_degraded_endpoint=endpoint,
            modified_endpoint=endpoint, coordinate_frame=f"{endpoint}_native",
        ),
        status=status,
        observation_support_fraction=1.0,
        identifiable_support_fraction=float(changed.mean()),
        null_score=max(before_median / 255.0, 0.0),
        action_score=max(after_median / 255.0, 0.0),
        spatial_holdout_gain=float(np.median(boundary_gains) / 255.0)
        if len(boundary_gains) else 0.0,
        parameter_uncertainty=float(1.0 / math.sqrt(max(len(tiles), 1))),
        fit_stability=float(np.mean(boundary_gains > 0.0))
        if len(boundary_gains) else 0.0,
        competing_model_scores={
            "jpeg_presence_lcb": float(presence_lcb),
            "boundary_reduction_lcb_255": float(boundary_lcb),
            "codec_closure_excess_ub_255": float(closure_ub),
        },
        estimated_parameters={
            "ijg_quality": float(estimated_quality), "block": float(BLOCK),
        },
        diagnostics={
            "check_tiles": float(len(tiles)), "simultaneous_z": z_value,
            "changed_fraction": float(changed.mean()),
            "declared_footprint_fraction": float(footprint.mean()),
            "outside_footprint_exact": float(outside_exact),
            "codec_noninferiority_margin_255": CODEC_NONINFERIORITY_255,
        },
        rejection_reasons=tuple(reasons),
        calibration_version="jpeg-action-efficacy-ab-codec-closure-v2",
        measured_probe_seconds=time.perf_counter() - started,
    )

