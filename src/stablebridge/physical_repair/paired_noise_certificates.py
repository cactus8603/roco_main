"""Paired null models for noise-action selection."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .contracts import ActionSpec, PhysicalCertificate
from .noise_certificates import additive_noise_certificate
from .operators import impulse_support


def _warp_second(value: np.ndarray, flow: np.ndarray, interpolation: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    height, width = value.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
    valid = ((map_x >= 0.0) & (map_x <= width - 1.0)
             & (map_y >= 0.0) & (map_y <= height - 1.0))
    warped = cv2.remap(value, map_x, map_y, interpolation,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return warped, valid


def _certificate(endpoint: str, evidence: np.ndarray, valid: np.ndarray, *,
                 raw_fraction: float, difference_threshold_255: float,
                 minimum_pixels: int, minimum_occupied_tiles: int,
                 started: float) -> PhysicalCertificate:
    height, width = evidence.shape
    count = int(evidence.sum())
    fraction = float(count / max(int(valid.sum()), 1))
    tile = 32
    occupied = 0
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            occupied += int(np.any(evidence[y:y + tile, x:x + tile]))
    reasons = []
    if count < minimum_pixels:
        reasons.append("insufficient_paired_unexplained_extrema")
    if occupied < minimum_occupied_tiles:
        reasons.append("unexplained_extrema_not_spatially_distributed")
    status = "supported" if not reasons else "rejected"
    action = ActionSpec(
        operator_id="impulse_exact_median3", operator_version="v2-paired-null",
        domain="image", hypothesized_degraded_endpoint=endpoint,
        modified_endpoint=endpoint, coordinate_frame=f"{endpoint}_native",
    )
    stability = float(np.clip(occupied / max(2 * minimum_occupied_tiles, 1), 0.0, 1.0))
    return PhysicalCertificate(
        action=action, status=status, observation_support_fraction=float(valid.mean()),
        identifiable_support_fraction=fraction, null_score=raw_fraction,
        action_score=fraction, spatial_holdout_gain=max(raw_fraction - fraction, 0.0),
        parameter_uncertainty=float(1.0 / math.sqrt(max(count, 1))),
        fit_stability=stability,
        competing_model_scores={"persistent_or_correspondence_null_fraction":
                                max(raw_fraction - fraction, 0.0)},
        estimated_parameters={"paired_unexplained_impulse_density": fraction},
        diagnostics={
            "detected_pixels": float(count), "occupied_32px_tiles": float(occupied),
            "paired_difference_threshold_255": float(difference_threshold_255),
        },
        rejection_reasons=tuple(reasons), calibration_version="paired-impulse-null-v2",
        measured_probe_seconds=time.perf_counter() - started,
    )


def paired_impulse_certificates(first: np.ndarray, second: np.ndarray, flow: np.ndarray, *,
                                difference_threshold_255: float = 48.0,
                                minimum_pixels: int = 32,
                                minimum_occupied_tiles: int = 4
                                ) -> tuple[PhysicalCertificate, PhysicalCertificate]:
    """Compare isolated extrema against the other endpoint at native correspondence."""
    started = time.perf_counter()
    first = np.asarray(first); second = np.asarray(second); flow = np.asarray(flow, np.float32)
    if (first.dtype != np.uint8 or second.dtype != np.uint8 or first.shape != second.shape
            or first.ndim != 3 or first.shape[2] != 3
            or flow.shape != (*first.shape[:2], 2) or not np.isfinite(flow).all()):
        raise ValueError("invalid observed pair or flow")
    first_mask = impulse_support(first) > 0.0
    second_mask = impulse_support(second).astype(np.float32)
    warped_second, valid = _warp_second(second.astype(np.float32), flow, cv2.INTER_LINEAR)
    warped_second_mask, mask_valid = _warp_second(second_mask, flow, cv2.INTER_NEAREST)
    valid &= mask_valid
    difference = np.mean(np.abs(first.astype(np.float32) - warped_second), axis=2)
    first_evidence = first_mask & valid & (difference >= difference_threshold_255)
    second_evidence_flow = (warped_second_mask > 0.5) & valid & (
        difference >= difference_threshold_255
    )
    first_cert = _certificate(
        "first", first_evidence, valid, raw_fraction=float(first_mask.mean()),
        difference_threshold_255=difference_threshold_255,
        minimum_pixels=minimum_pixels, minimum_occupied_tiles=minimum_occupied_tiles,
        started=started,
    )
    second_cert = _certificate(
        "second", second_evidence_flow, valid,
        raw_fraction=float((warped_second_mask[valid] > 0.5).mean() if np.any(valid) else 0.0),
        difference_threshold_255=difference_threshold_255,
        minimum_pixels=minimum_pixels, minimum_occupied_tiles=minimum_occupied_tiles,
        started=started,
    )
    return first_cert, second_cert


def select_paired_noise_action(first: np.ndarray, second: np.ndarray, flow: np.ndarray) -> dict:
    impulse_first, impulse_second = paired_impulse_certificates(first, second, flow)
    certificates = {
        "impulse_first": impulse_first,
        "impulse_second": impulse_second,
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
            return {
                "action": operator,
                "endpoint": "both" if len(supported) == 2 else supported[0],
                "family": family, "certificates": certificates,
            }
    return {"action": "native", "endpoint": "unknown", "family": "none",
            "certificates": certificates}

