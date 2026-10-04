"""Action-response certificates with observable correspondence competitors.

These certificates do not classify a named corruption.  They ask whether a
specific local image action has enough visible, unique correspondence support
and whether its held-out response is better than the identity explanation.
The returned competitor scores keep radiometry, local warp error, and relative
bandwidth loss available to a later action-relative utility model.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .contracts import ActionSpec, PhysicalCertificate
from .endpoint_support import observable_endpoint_applicability
from .operators import apply_local_image_action


_POPCOUNT = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
JOINT_ACTIONS = tuple(
    (operator, endpoint)
    for operator in ("impulse_exact_median3", "wiener3")
    for endpoint in ("first", "second", "both")
)


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


def _census(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    center = padded[1:-1, 1:-1]
    output = np.zeros(center.shape, dtype=np.uint8)
    bit = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbour = padded[1 + dy:1 + dy + center.shape[0],
                               1 + dx:1 + dx + center.shape[1]]
            output |= ((neighbour >= center).astype(np.uint8) << bit)
            bit += 1
    return output


def _warp(value: np.ndarray, flow: np.ndarray, *, dx: float = 0.0, dy: float = 0.0,
          interpolation: int = cv2.INTER_LINEAR) -> tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx + flow[..., 0] + np.float32(dx)
    map_y = yy + flow[..., 1] + np.float32(dy)
    valid = ((map_x >= 0.0) & (map_x <= width - 1.0)
             & (map_y >= 0.0) & (map_y <= height - 1.0))
    warped = cv2.remap(
        np.asarray(value), map_x, map_y, interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped, valid


def _matching_evidence(first: np.ndarray, second: np.ndarray, flow: np.ndarray
                       ) -> dict[str, np.ndarray]:
    first_desc, second_desc = _census(first), _census(second)
    scores = []
    validities = []
    offsets = tuple((dy, dx) for dy in (-2, 0, 2) for dx in (-2, 0, 2))
    for dy, dx in offsets:
        warped, valid = _warp(
            second_desc, flow, dx=float(dx), dy=float(dy),
            interpolation=cv2.INTER_NEAREST,
        )
        score = _POPCOUNT[np.bitwise_xor(first_desc, warped)].astype(np.float32) / 8.0
        score[~valid] = 2.0
        scores.append(score)
        validities.append(valid)
    stacked = np.stack(scores)
    ordered = np.partition(stacked, kth=1, axis=0)
    best, second_best = ordered[0], ordered[1]
    native_index = offsets.index((0, 0))
    native = stacked[native_index]
    native_valid = validities[native_index]
    margin = np.maximum(second_best - best, 0.0)
    # At most three of eight Census bits may disagree and the best local match
    # must beat the runner-up by at least one bit.  These are descriptor-level
    # identifiability requirements, not family-specific fitted thresholds.
    unique = native_valid & (best <= 3.0 / 8.0) & (margin >= 1.0 / 8.0)
    return {
        "native": native, "best": best, "margin": margin,
        "valid": native_valid, "unique": unique,
    }


def _action_pair(first: np.ndarray, second: np.ndarray, flow: np.ndarray, *,
                 operator_id: str, endpoint: str
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    applicability = observable_endpoint_applicability(
        first, second, flow, operator_id=operator_id, endpoint=endpoint,
    )
    left, right = first.copy(), second.copy()
    diagnostics: dict[str, float] = {}
    if endpoint in {"first", "both"}:
        left, detail = apply_local_image_action(
            first, applicability.input_first,
            operator_id=operator_id, endpoint="first",
        )
        diagnostics.update({f"first_{key}": float(value)
                            for key, value in detail.diagnostics.items()})
        diagnostics["first_changed_fraction"] = detail.changed_fraction
    if endpoint in {"second", "both"}:
        right, detail = apply_local_image_action(
            second, applicability.input_second,
            operator_id=operator_id, endpoint="second",
        )
        diagnostics.update({f"second_{key}": float(value)
                            for key, value in detail.diagnostics.items()})
        diagnostics["second_changed_fraction"] = detail.changed_fraction
    return left, right, applicability.output_flow, diagnostics


def _residual_maps(first: np.ndarray, second: np.ndarray, flow: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    warped_rgb, valid = _warp(second.astype(np.float32), flow)
    rgb = np.mean(np.abs(first.astype(np.float32) - warped_rgb), axis=2) / 255.0
    first_desc, second_desc = _census(first), _census(second)
    warped_desc, desc_valid = _warp(
        second_desc, flow, interpolation=cv2.INTER_NEAREST,
    )
    census = _POPCOUNT[np.bitwise_xor(first_desc, warped_desc)].astype(np.float32) / 8.0
    return rgb, census, valid & desc_valid


def _radiometric_residual(first: np.ndarray, warped_second: np.ndarray,
                          chosen: np.ndarray) -> float:
    if int(chosen.sum()) < 8:
        return 1.0
    x = warped_second[chosen].astype(np.float64)
    y = first[chosen].astype(np.float64)
    residuals = []
    for channel in range(3):
        design = np.column_stack((x[:, channel], np.ones(len(x))))
        coefficients, *_ = np.linalg.lstsq(design, y[:, channel], rcond=None)
        gain = float(np.clip(coefficients[0], 0.25, 4.0))
        offset = float(np.clip(coefficients[1], -255.0, 255.0))
        prediction = gain * x[:, channel] + offset
        residuals.append(np.abs(y[:, channel] - prediction))
    return float(np.mean(np.column_stack(residuals)) / 255.0)


def _gradient_deficit(first: np.ndarray, second: np.ndarray, flow: np.ndarray,
                      chosen: np.ndarray, endpoint: str) -> float:
    def magnitude(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)

    first_gradient = magnitude(first)
    second_gradient, valid = _warp(magnitude(second), flow)
    use = chosen & valid
    if not np.any(use):
        return 0.0
    first_mean = float(first_gradient[use].mean())
    second_mean = float(second_gradient[use].mean())
    scale = max(first_mean, second_mean, 1e-6)
    if endpoint == "first":
        return max(second_mean - first_mean, 0.0) / scale
    if endpoint == "second":
        return max(first_mean - second_mean, 0.0) / scale
    return abs(first_mean - second_mean) / scale


def joint_action_certificate(first: np.ndarray, second: np.ndarray, flow: np.ndarray, *,
                             operator_id: str, endpoint: str,
                             minimum_evidence_pixels: int = 32
                             ) -> PhysicalCertificate:
    """Score one executable action against identity and observable competitors."""
    started = time.perf_counter()
    first, second, flow = _validate(first, second, flow)
    if operator_id not in {name for name, _ in JOINT_ACTIONS}:
        raise ValueError(f"unsupported joint action: {operator_id}")
    if endpoint not in {"first", "second", "both"}:
        raise ValueError("endpoint must be first, second, or both")
    matching = _matching_evidence(first, second, flow)
    repaired_first, repaired_second, output_support, action_diagnostics = _action_pair(
        first, second, flow, operator_id=operator_id, endpoint=endpoint,
    )
    before_rgb, before_census, before_valid = _residual_maps(first, second, flow)
    after_rgb, after_census, after_valid = _residual_maps(
        repaired_first, repaired_second, flow,
    )
    support = output_support > 0.05
    observable = support & before_valid & after_valid
    identifiable = observable & matching["unique"]
    count = int(identifiable.sum())
    changed = (np.any(repaired_first != first, axis=2)
               | np.any(repaired_second != second, axis=2))
    effective = identifiable & changed
    effective_count = int(effective.sum())
    chosen = effective if effective_count else identifiable
    if np.any(chosen):
        null_map = 0.5 * (before_rgb + before_census)
        action_map = 0.5 * (after_rgb + after_census)
        null_score = float(null_map[chosen].mean())
        action_score = float(action_map[chosen].mean())
        rgb_gain = float((before_rgb[chosen] - after_rgb[chosen]).mean())
        census_gain = float((before_census[chosen] - after_census[chosen]).mean())
        warp_offset_gain = float(
            (matching["native"][chosen] - matching["best"][chosen]).mean()
        )
        margin = float(matching["margin"][chosen].mean())
    else:
        null_score = action_score = rgb_gain = census_gain = warp_offset_gain = margin = 0.0
    height, width = support.shape
    yy, xx = np.mgrid[:height, :width]
    partition = ((yy // 64 + xx // 64) % 2).astype(bool)
    holdout_gains = []
    for parity in (False, True):
        subset = chosen & (partition == parity)
        if np.any(subset):
            combined_before = 0.5 * (before_rgb[subset] + before_census[subset])
            combined_after = 0.5 * (after_rgb[subset] + after_census[subset])
            holdout_gains.append(float(np.mean(combined_before - combined_after)))
    spatial_gain = float(min(holdout_gains, default=0.0))
    fit_stability = float(np.mean(np.asarray(holdout_gains) > 0.0)) if holdout_gains else 0.0
    warped_second, warp_valid = _warp(second.astype(np.float32), flow)
    affine_use = identifiable & warp_valid
    affine_residual = _radiometric_residual(first, warped_second, affine_use)
    identity_rgb = float(before_rgb[affine_use].mean()) if np.any(affine_use) else 1.0
    affine_gain = identity_rgb - affine_residual
    blur_score = _gradient_deficit(first, second, flow, identifiable, endpoint)
    reasons = []
    if int(observable.sum()) < minimum_evidence_pixels:
        reasons.append("insufficient_observable_action_support")
    if count < minimum_evidence_pixels:
        reasons.append("insufficient_unique_correspondence_support")
    if effective_count < minimum_evidence_pixels:
        reasons.append("insufficient_effective_action_response")
    if null_score - action_score <= 0.0:
        reasons.append("no_identity_residual_improvement")
    if len(holdout_gains) < 2 or spatial_gain <= 0.0:
        reasons.append("no_two_partition_holdout_improvement")
    status = "supported" if not reasons else "rejected"
    action = ActionSpec(
        operator_id=operator_id, operator_version="v3-joint-response",
        domain="image", hypothesized_degraded_endpoint=endpoint,
        modified_endpoint=endpoint, coordinate_frame="flow_native",
    )
    return PhysicalCertificate(
        action=action, status=status,
        observation_support_fraction=float(observable.mean()),
        identifiable_support_fraction=float(identifiable.mean()),
        null_score=max(null_score, 0.0), action_score=max(action_score, 0.0),
        spatial_holdout_gain=spatial_gain,
        parameter_uncertainty=float(1.0 / math.sqrt(max(effective_count, 1))),
        fit_stability=fit_stability,
        competing_model_scores={
            "affine_radiometry_gain": affine_gain,
            "local_warp_offset_gain": warp_offset_gain,
            "relative_gradient_deficit": blur_score,
            "identity_rgb_residual": identity_rgb,
            "affine_rgb_residual": affine_residual,
        },
        estimated_parameters={
            "combined_residual_gain": null_score - action_score,
            "rgb_residual_gain": rgb_gain,
            "census_residual_gain": census_gain,
            "mean_matching_uniqueness": margin,
            "changed_fraction": float(changed.mean()),
            "support_fraction": float(support.mean()),
        },
        diagnostics={
            **action_diagnostics,
            "observable_pixels": float(observable.sum()),
            "unique_pixels": float(count),
            "effective_pixels": float(effective_count),
            "holdout_partitions": float(len(holdout_gains)),
        },
        rejection_reasons=tuple(reasons),
        calibration_version="descriptor-identifiability-v3",
        measured_probe_seconds=time.perf_counter() - started,
    )


def joint_action_certificates(first: np.ndarray, second: np.ndarray,
                              flow: np.ndarray) -> dict[str, PhysicalCertificate]:
    """Return all six certificates without selecting from outcome labels."""
    return {
        f"{operator}@{endpoint}": joint_action_certificate(
            first, second, flow, operator_id=operator, endpoint=endpoint,
        )
        for operator, endpoint in JOINT_ACTIONS
    }

