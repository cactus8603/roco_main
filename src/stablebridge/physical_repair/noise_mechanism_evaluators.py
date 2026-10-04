"""Outcome-blind physical-mechanism evaluators for Wave-1 noise actions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import cv2
import numpy as np


NOISE_MECHANISM_RECEIPT_SCHEMA_V1 = "stablebridge-noise-mechanism-evaluation/v1"
_SUPPORTED = {
    "paired_impulse_median3.endpoint_supported_v1": (
        "impulse_anchor_deviation_reduction_v1"
    ),
    "paired_additive_wiener3.local_tile_sigma_v2": (
        "active_tile_excess_noise_power_reduction_v1"
    ),
}


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _validate(
    before: Sequence[np.ndarray],
    after: Sequence[np.ndarray],
    supports: Sequence[np.ndarray],
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    if len(before) != 2 or len(after) != 2 or len(supports) != 2:
        raise ValueError("mechanism evaluator needs two endpoints")
    before = tuple(np.asarray(value) for value in before)
    after = tuple(np.asarray(value) for value in after)
    supports = tuple(np.asarray(value) for value in supports)
    if (
        before[0].dtype != np.uint8 or before[1].dtype != np.uint8
        or before[0].ndim != 3 or before[0].shape != before[1].shape
        or before[0].shape[2] != 3
        or any(value.dtype != np.uint8 or value.shape != before[0].shape for value in after)
        or any(value.dtype != np.bool_ or value.shape != before[0].shape[:2] for value in supports)
    ):
        raise ValueError("invalid mechanism before/after/support arrays")
    return (
        tuple(np.ascontiguousarray(value) for value in before),
        tuple(np.ascontiguousarray(value) for value in after),
        tuple(np.ascontiguousarray(value) for value in supports),
    )


def _relative_reduction(before: float, after: float) -> float:
    if not math.isfinite(before) or not math.isfinite(after) or before <= 1e-12:
        raise RuntimeError("mechanism signal is unestimable")
    return float((before - after) / before)


def _impulse_endpoint(
    before: np.ndarray, after: np.ndarray, support: np.ndarray,
) -> dict[str, Any]:
    if not bool(support.any()):
        raise RuntimeError("impulse mechanism support is empty")
    anchor = cv2.medianBlur(before, 3)
    before_signal = float(np.mean(
        np.abs(before[support].astype(np.float64) - anchor[support].astype(np.float64))
    ) / 255.0)
    after_signal = float(np.mean(
        np.abs(after[support].astype(np.float64) - anchor[support].astype(np.float64))
    ) / 255.0)
    outside_exact = bool(np.array_equal(before[~support], after[~support]))
    if not outside_exact:
        raise RuntimeError("impulse mechanism output escaped support")
    reduction = _relative_reduction(before_signal, after_signal)
    return {
        "active_samples_rgb": int(support.sum()) * 3,
        "frozen_anchor_sha256": _array_sha256(anchor),
        "active_signal_before": before_signal,
        "active_signal_after": after_signal,
        "active_relative_reduction": reduction,
        "negative_control": "NO_OP_REPLAY_ON_SAME_SUPPORT",
        "negative_control_relative_reduction": 0.0,
        "specificity_difference": reduction,
        "outside_support_byte_identity": outside_exact,
    }


def _tile_high_band_power(image: np.ndarray, tile_size: int = 32) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (3, 3), borderType=cv2.BORDER_REFLECT_101)
    high = np.square(gray - mean, dtype=np.float32)
    rows = math.ceil(gray.shape[0] / tile_size)
    columns = math.ceil(gray.shape[1] / tile_size)
    powers = np.empty((rows, columns), dtype=np.float64)
    for row in range(rows):
        for column in range(columns):
            tile = high[
                row * tile_size:min((row + 1) * tile_size, gray.shape[0]),
                column * tile_size:min((column + 1) * tile_size, gray.shape[1]),
            ]
            powers[row, column] = float(np.mean(tile, dtype=np.float64))
    return powers


def _tile_support(mask: np.ndarray, tile_size: int = 32) -> np.ndarray:
    rows = math.ceil(mask.shape[0] / tile_size)
    columns = math.ceil(mask.shape[1] / tile_size)
    result = np.zeros((rows, columns), dtype=bool)
    for row in range(rows):
        for column in range(columns):
            result[row, column] = bool(np.any(mask[
                row * tile_size:min((row + 1) * tile_size, mask.shape[0]),
                column * tile_size:min((column + 1) * tile_size, mask.shape[1]),
            ]))
    return result


def _wiener_endpoint(
    before: np.ndarray, after: np.ndarray, support: np.ndarray,
) -> dict[str, Any]:
    active = _tile_support(support)
    if not bool(active.any()):
        raise RuntimeError("Wiener mechanism support has no active tile")
    # The v2 support is whole-tile by construction.  Reject partial support so
    # mechanism evidence cannot silently change its denominator.
    expanded = np.repeat(np.repeat(active, 32, axis=0), 32, axis=1)[
        :support.shape[0], :support.shape[1]
    ]
    if not np.array_equal(expanded, support):
        raise RuntimeError("Wiener support is not the frozen whole-tile support")
    before_power = _tile_high_band_power(before)
    after_power = _tile_high_band_power(after)
    inactive = ~active
    # The high-band statistic has a one-pixel read halo.  A directly adjacent
    # tile can therefore respond to changed active pixels despite byte-exact
    # inactive output.  Freeze a one-tile guard band before selecting controls.
    guarded_active = cv2.dilate(
        active.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1,
    ).astype(bool)
    control_eligible = ~guarded_active
    if not bool(control_eligible.any()):
        raise RuntimeError("Wiener mechanism needs an inactive before-only noise floor")
    floor = float(np.median(before_power[inactive]))
    active_before_values = np.maximum(before_power[active] - floor, 0.0)
    active_after_values = np.maximum(after_power[active] - floor, 0.0)
    active_before = float(np.mean(active_before_values))
    active_after = float(np.mean(active_after_values))
    active_reduction = _relative_reduction(active_before, active_after)

    coordinates = np.argwhere(control_eligible)
    target = float(np.median(before_power[active]))
    ranked = sorted(
        (
            abs(float(before_power[row, column]) - target),
            int(row), int(column),
        )
        for row, column in coordinates
    )
    matched = ranked[:int(active.sum())]
    if len(matched) != int(active.sum()):
        raise RuntimeError("insufficient matched inactive Wiener tiles")
    matched_coordinates = [(row, column) for _, row, column in matched]
    inactive_before = float(np.mean([
        max(float(before_power[row, column]) - floor, 0.0)
        for row, column in matched_coordinates
    ]))
    inactive_after = float(np.mean([
        max(float(after_power[row, column]) - floor, 0.0)
        for row, column in matched_coordinates
    ]))
    inactive_reduction = (
        0.0 if inactive_before <= 1e-12
        else float((inactive_before - inactive_after) / inactive_before)
    )
    outside_exact = bool(np.array_equal(before[~support], after[~support]))
    if not outside_exact:
        raise RuntimeError("Wiener mechanism output escaped support")
    return {
        "tile_size_px": 32,
        "active_tiles": int(active.sum()),
        "matched_inactive_tiles": [list(value) for value in matched_coordinates],
        "inactive_control_guard": "ONE_TILE_CHEBYSHEV_GUARD_FOR_3X3_METRIC_HALO",
        "frozen_inactive_noise_floor": floor,
        "active_signal_before": active_before,
        "active_signal_after": active_after,
        "active_relative_reduction": active_reduction,
        "matched_inactive_signal_before": inactive_before,
        "matched_inactive_signal_after": inactive_after,
        "negative_control_relative_reduction": inactive_reduction,
        "specificity_difference": float(active_reduction - inactive_reduction),
        "outside_support_byte_identity": outside_exact,
    }


@dataclass(frozen=True)
class NoiseMechanismEvaluationReceiptV1:
    action_id: str
    metric_id: str
    physical_pair_sha256: str
    endpoint_records: tuple[dict[str, Any], ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.action_id not in _SUPPORTED or self.metric_id != _SUPPORTED[self.action_id]:
            raise ValueError("noise mechanism action/metric identity drift")
        if not self.endpoint_records:
            raise ValueError("mechanism receipt needs an evaluated endpoint")
        payload = self.as_dict(include_hash=False)
        expected = _canonical_sha256(payload)
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("mechanism receipt hash drift")
        object.__setattr__(self, "receipt_sha256", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema": NOISE_MECHANISM_RECEIPT_SCHEMA_V1,
            "action_id": self.action_id,
            "metric_id": self.metric_id,
            "physical_pair_sha256": self.physical_pair_sha256,
            "endpoint_records": list(self.endpoint_records),
            "runtime_inputs": ["before_pair", "after_pair", "endpoint_supports"],
            "ground_truth_read": False,
            "task_outcome_read": False,
            "scientific_qualification": False,
            "selector_admission": False,
            "production_authority": False,
        }
        if include_hash:
            result["receipt_sha256"] = self.receipt_sha256
        return result


def evaluate_noise_mechanism_v1(
    *,
    action_id: str,
    physical_pair_sha256: str,
    before_pair: Sequence[np.ndarray],
    after_pair: Sequence[np.ndarray],
    endpoint_supports: Sequence[np.ndarray],
) -> NoiseMechanismEvaluationReceiptV1:
    if action_id not in _SUPPORTED:
        raise ValueError("unsupported noise mechanism action")
    before, after, supports = _validate(before_pair, after_pair, endpoint_supports)
    rows = []
    for index, endpoint in enumerate(("first", "second")):
        support = supports[index]
        if not bool(support.any()):
            if not np.array_equal(before[index], after[index]):
                raise RuntimeError("empty-support endpoint changed")
            continue
        values = (
            _impulse_endpoint(before[index], after[index], support)
            if action_id == "paired_impulse_median3.endpoint_supported_v1"
            else _wiener_endpoint(before[index], after[index], support)
        )
        rows.append({
            "endpoint": endpoint,
            "support_sha256": _array_sha256(support),
            "support_pixels": int(support.sum()),
            **values,
        })
    return NoiseMechanismEvaluationReceiptV1(
        action_id=action_id,
        metric_id=_SUPPORTED[action_id],
        physical_pair_sha256=physical_pair_sha256,
        endpoint_records=tuple(rows),
    )


__all__ = [
    "NOISE_MECHANISM_RECEIPT_SCHEMA_V1",
    "NoiseMechanismEvaluationReceiptV1",
    "evaluate_noise_mechanism_v1",
]
