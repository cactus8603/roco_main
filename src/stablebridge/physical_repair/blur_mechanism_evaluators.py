"""Outcome-blind branch-specific mechanism evaluators for local blur v2."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import cv2
import numpy as np

from .blur_parameter_certificates import TILE, _kernel
from .local_blur_successor import (
    LOCAL_BLUR_ACTION_ID_V2,
    LOCAL_BLUR_OPERATOR_ID_V2,
    _flow_support,
    _proposal,
)
from .operators import blend_local_proposal
from .support import transport_flow_mask_to_second


BLUR_MECHANISM_RECEIPT_SCHEMA_V1 = "stablebridge-blur-mechanism-evaluation/v1"
_METRICS = {
    "disk": "disk_otf_endpoint_discrepancy_reduction_v1",
    "gaussian": "gaussian_otf_endpoint_discrepancy_reduction_v1",
    "motion": "motion_otf_endpoint_discrepancy_reduction_v1",
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
    flow: np.ndarray,
) -> tuple[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray], np.ndarray,
]:
    if len(before) != 2 or len(after) != 2 or len(supports) != 2:
        raise ValueError("blur evaluator needs two endpoints")
    before = tuple(np.asarray(value) for value in before)
    after = tuple(np.asarray(value) for value in after)
    supports = tuple(np.asarray(value) for value in supports)
    flow = np.asarray(flow)
    if (
        before[0].dtype != np.uint8 or before[1].dtype != np.uint8
        or before[0].ndim != 3 or before[0].shape != before[1].shape
        or before[0].shape[2] != 3
        or any(value.dtype != np.uint8 or value.shape != before[0].shape for value in after)
        or any(
            value.dtype != np.bool_ or value.shape != before[0].shape[:2]
            for value in supports
        )
        or flow.dtype != np.float32 or flow.shape != (*before[0].shape[:2], 2)
        or not np.isfinite(flow).all()
    ):
        raise ValueError("invalid blur mechanism arrays")
    return (
        tuple(np.ascontiguousarray(value) for value in before),
        tuple(np.ascontiguousarray(value) for value in after),
        tuple(np.ascontiguousarray(value) for value in supports),
        np.ascontiguousarray(flow),
    )


def _warp_second(image: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx + flow[..., 0]
    map_y = yy + flow[..., 1]
    valid = (
        (map_x >= 0.0) & (map_x <= width - 1.0)
        & (map_y >= 0.0) & (map_y <= height - 1.0)
    )
    warped = cv2.remap(
        image.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    return np.ascontiguousarray(warped), np.ascontiguousarray(valid)


def _gray_float(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return cv2.cvtColor(clipped, cv2.COLOR_RGB2GRAY).astype(np.float64)


def _spectrum(
    image: np.ndarray, valid: np.ndarray, window: np.ndarray,
) -> np.ndarray:
    gray = _gray_float(image)
    spatial_weight = window * valid.astype(np.float64)
    denominator = max(float(spatial_weight.sum()), 1e-12)
    center = float(np.sum(gray * spatial_weight) / denominator)
    return np.abs(np.fft.rfft2((gray - center) * spatial_weight))


def _tile_discrepancies(
    target: np.ndarray,
    native_candidate: np.ndarray,
    action_candidate: np.ndarray,
    negative_candidate: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float, float, dict[str, Any]]:
    window = np.outer(np.hanning(TILE), np.hanning(TILE)).astype(np.float64)
    fy = np.fft.fftfreq(TILE)[:, None]
    fx = np.fft.rfftfreq(TILE)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    band = (radius >= 1.0 / TILE) & (radius <= 0.45)
    low = band & (radius <= 0.08)
    spectra = [
        _spectrum(value, valid, window)
        for value in (target, native_candidate, action_candidate, negative_candidate)
    ]
    target_spectrum, native_spectrum = spectra[:2]
    common_scale = max(
        float(np.median(np.concatenate((
            target_spectrum[band], native_spectrum[band],
        )))) * 1e-6,
        1e-9,
    )

    def normalize(value: np.ndarray) -> np.ndarray:
        return value / max(float(value[low].sum()), common_scale)

    normalized = [normalize(value) for value in spectra]
    frozen_weights = np.sqrt(normalized[0] * normalized[1]) * band
    weight_sum = max(float(frozen_weights.sum()), 1e-12)

    def distance(value: np.ndarray) -> float:
        delta = np.abs(
            np.log(np.maximum(normalized[0], common_scale))
            - np.log(np.maximum(value, common_scale))
        )
        return float(np.sum(frozen_weights * delta) / weight_sum)

    values = tuple(distance(value) for value in normalized[1:])
    return (*values, {
        "valid_fraction": float(valid.mean()),
        "frozen_frequency_weight_sha256": _array_sha256(frozen_weights),
        "native_target_spectrum_sha256": _array_sha256(target_spectrum),
        "native_opposite_spectrum_sha256": _array_sha256(native_spectrum),
    })


def _kernel_second_moment(kernel: np.ndarray) -> float:
    yy, xx = np.mgrid[:kernel.shape[0], :kernel.shape[1]]
    yy = yy.astype(np.float64) - (kernel.shape[0] - 1.0) / 2.0
    xx = xx.astype(np.float64) - (kernel.shape[1] - 1.0) / 2.0
    return float(np.sum(kernel.astype(np.float64) * (xx * xx + yy * yy)))


def _negative_control(
    family: str, parameter: tuple[float, ...],
) -> tuple[str, tuple[float, ...], dict[str, Any]]:
    selected_moment = _kernel_second_moment(_kernel(family, parameter))
    if family == "disk":
        control_family = "gaussian"
        control_parameter = (math.sqrt(max(selected_moment, 1e-12) / 2.0),)
        rule = "GAUSSIAN_SIGMA_FROM_SELECTED_DISK_DISCRETE_SECOND_MOMENT"
    elif family == "gaussian":
        candidates = [
            (
                abs(_kernel_second_moment(_kernel("disk", (float(radius),)))
                    - selected_moment),
                radius,
            )
            for radius in range(1, 9)
        ]
        _, radius = min(candidates)
        control_family = "disk"
        control_parameter = (float(radius),)
        rule = "CLOSEST_DISCRETE_DISK_SECOND_MOMENT_RADIUS_1_TO_8"
    elif family == "motion":
        control_family = "motion"
        control_parameter = (parameter[0], float((parameter[1] + 90.0) % 180.0))
        rule = "SAME_LENGTH_ORTHOGONAL_ORIENTATION"
    else:
        raise ValueError("unknown blur mechanism family")
    control_moment = _kernel_second_moment(_kernel(control_family, control_parameter))
    return control_family, control_parameter, {
        "rule": rule,
        "selected_second_moment_px2": selected_moment,
        "control_second_moment_px2": control_moment,
        "absolute_second_moment_mismatch_px2": abs(control_moment - selected_moment),
    }


@dataclass(frozen=True)
class BlurMechanismEvaluationReceiptV1:
    action_id: str
    family_id: str
    metric_id: str
    physical_pair_sha256: str
    winner_id: str
    identified_endpoint: str
    modified_endpoint: str
    selected_parameter: tuple[float, ...]
    negative_control_family: str
    negative_control_parameter: tuple[float, ...]
    endpoint_records: tuple[dict[str, Any], ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.action_id != LOCAL_BLUR_ACTION_ID_V2:
            raise ValueError("blur mechanism action identity drift")
        if self.family_id not in _METRICS or self.metric_id != _METRICS[self.family_id]:
            raise ValueError("blur mechanism family/metric identity drift")
        if self.winner_id != f"common_{self.family_id}@{self.identified_endpoint}":
            raise ValueError("blur mechanism winner identity drift")
        expected = "second" if self.identified_endpoint == "first" else "first"
        if self.modified_endpoint != expected:
            raise ValueError("blur mechanism endpoint direction drift")
        if len(self.endpoint_records) != 1:
            raise ValueError("blur mechanism receipt needs one routed endpoint record")
        payload = self.as_dict(include_hash=False)
        expected_hash = _canonical_sha256(payload)
        if self.receipt_sha256 and self.receipt_sha256 != expected_hash:
            raise ValueError("blur mechanism receipt hash drift")
        object.__setattr__(self, "receipt_sha256", expected_hash)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema": BLUR_MECHANISM_RECEIPT_SCHEMA_V1,
            "action_id": self.action_id,
            "family_id": self.family_id,
            "metric_id": self.metric_id,
            "physical_pair_sha256": self.physical_pair_sha256,
            "winner_id": self.winner_id,
            "identified_endpoint": self.identified_endpoint,
            "modified_endpoint": self.modified_endpoint,
            "selected_parameter": list(self.selected_parameter),
            "negative_control_family": self.negative_control_family,
            "negative_control_parameter": list(self.negative_control_parameter),
            "endpoint_records": list(self.endpoint_records),
            "runtime_inputs": [
                "before_pair", "after_pair", "endpoint_supports",
                "observed_native_flow", "recoverable_regions",
            ],
            "ground_truth_read": False,
            "task_outcome_read": False,
            "scientific_qualification": False,
            "selector_admission": False,
            "production_authority": False,
        }
        if include_hash:
            result["receipt_sha256"] = self.receipt_sha256
        return result


def evaluate_blur_mechanism_v1(
    *,
    physical_pair_sha256: str,
    before_pair: Sequence[np.ndarray],
    after_pair: Sequence[np.ndarray],
    endpoint_supports: Sequence[np.ndarray],
    observed_native_flow: np.ndarray,
    winner_id: str,
    family: str,
    identified_endpoint: str,
    selected_parameter: Sequence[float],
    recoverable_regions: Sequence[Sequence[int]],
) -> BlurMechanismEvaluationReceiptV1:
    """Evaluate selected blur PSF against its frozen matched negative control."""

    before, after, supports, flow = _validate(
        before_pair, after_pair, endpoint_supports, observed_native_flow,
    )
    if family not in _METRICS or identified_endpoint not in {"first", "second"}:
        raise ValueError("invalid routed blur family or endpoint")
    if winner_id != f"common_{family}@{identified_endpoint}":
        raise ValueError("winner/family/endpoint drift")
    parameter = tuple(float(value) for value in selected_parameter)
    if not parameter or not all(math.isfinite(value) for value in parameter):
        raise ValueError("selected blur parameter must be finite")
    regions = tuple(tuple(map(int, value)) for value in recoverable_regions)
    if not regions:
        raise RuntimeError("blur mechanism recoverable support is empty")
    flow_weight = _flow_support(before[0].shape[:2], regions)
    if identified_endpoint == "first":
        modified_endpoint = "second"
        endpoint_weight, _ = transport_flow_mask_to_second(flow_weight, flow)
        expected_supports = (
            np.zeros(flow_weight.shape, dtype=bool), endpoint_weight > 0.0,
        )
        modified_index = 1
    else:
        modified_endpoint = "first"
        endpoint_weight = flow_weight
        expected_supports = (
            endpoint_weight > 0.0, np.zeros(flow_weight.shape, dtype=bool),
        )
        modified_index = 0
    if any(
        not np.array_equal(actual, expected)
        for actual, expected in zip(supports, expected_supports)
    ):
        raise RuntimeError("blur mechanism endpoint support drift")
    if not np.array_equal(before[1 - modified_index], after[1 - modified_index]):
        raise RuntimeError("blur mechanism changed its identified endpoint")
    if not np.array_equal(
        before[modified_index][~supports[modified_index]],
        after[modified_index][~supports[modified_index]],
    ):
        raise RuntimeError("blur mechanism output escaped support")

    negative_family, negative_parameter, negative_match = _negative_control(
        family, parameter,
    )
    negative_modified, negative_record = blend_local_proposal(
        before[modified_index],
        _proposal(before[modified_index], negative_family, negative_parameter),
        endpoint_weight,
        operator_id=f"{LOCAL_BLUR_OPERATOR_ID_V2}.negative_control",
        endpoint=modified_endpoint,
        feather_sigma=1.0,
    )
    negative_pair = [before[0].copy(), before[1].copy()]
    negative_pair[modified_index] = negative_modified

    warped_native_second, valid = _warp_second(before[1], flow)
    warped_action_second, valid_action = _warp_second(after[1], flow)
    warped_negative_second, valid_negative = _warp_second(negative_pair[1], flow)
    if not np.array_equal(valid, valid_action) or not np.array_equal(valid, valid_negative):
        raise RuntimeError("blur mechanism warp-validity drift")
    region_rows = []
    for y0, x0 in regions:
        ys, xs = slice(y0, y0 + TILE), slice(x0, x0 + TILE)
        local_valid = valid[ys, xs]
        if float(local_valid.mean()) < 0.90:
            raise RuntimeError("blur mechanism region lost frozen valid support")
        if identified_endpoint == "first":
            target = before[0][ys, xs].astype(np.float32)
            native_candidate = warped_native_second[ys, xs]
            action_candidate = warped_action_second[ys, xs]
            negative_candidate = warped_negative_second[ys, xs]
        else:
            target = warped_native_second[ys, xs]
            native_candidate = before[0][ys, xs].astype(np.float32)
            action_candidate = after[0][ys, xs].astype(np.float32)
            negative_candidate = negative_pair[0][ys, xs].astype(np.float32)
        native_signal, action_signal, negative_signal, diagnostics = _tile_discrepancies(
            target, native_candidate, action_candidate, negative_candidate,
            local_valid,
        )
        region_rows.append({
            "region_y0_x0": [y0, x0],
            "native_signal": native_signal,
            "action_signal": action_signal,
            "negative_control_signal": negative_signal,
            **diagnostics,
        })
    before_signal = float(np.mean([row["native_signal"] for row in region_rows]))
    after_signal = float(np.mean([row["action_signal"] for row in region_rows]))
    negative_signal = float(np.mean([
        row["negative_control_signal"] for row in region_rows
    ]))
    if before_signal <= 1e-12 or not all(map(
        math.isfinite, (before_signal, after_signal, negative_signal),
    )):
        raise RuntimeError("blur mechanism signal is unestimable")
    active_reduction = float((before_signal - after_signal) / before_signal)
    negative_reduction = float((before_signal - negative_signal) / before_signal)
    specificity = float(active_reduction - negative_reduction)
    endpoint_record = {
        "endpoint": modified_endpoint,
        "identified_endpoint": identified_endpoint,
        "support_sha256": _array_sha256(supports[modified_index]),
        "support_pixels": int(supports[modified_index].sum()),
        "recoverable_region_count": len(regions),
        "active_signal_before": before_signal,
        "active_signal_after": after_signal,
        "active_relative_reduction": active_reduction,
        "negative_control_signal_after": negative_signal,
        "negative_control_relative_reduction": negative_reduction,
        "specificity_difference": specificity,
        "outside_support_byte_identity": True,
        "negative_control_match": negative_match,
        "negative_control_local_action": {
            "operator_id": negative_record.operator_id,
            "endpoint": negative_record.endpoint,
            "support_fraction": negative_record.support_fraction,
            "changed_fraction": negative_record.changed_fraction,
        },
        "region_records": region_rows,
    }
    return BlurMechanismEvaluationReceiptV1(
        action_id=LOCAL_BLUR_ACTION_ID_V2,
        family_id=family,
        metric_id=_METRICS[family],
        physical_pair_sha256=physical_pair_sha256,
        winner_id=winner_id,
        identified_endpoint=identified_endpoint,
        modified_endpoint=modified_endpoint,
        selected_parameter=parameter,
        negative_control_family=negative_family,
        negative_control_parameter=negative_parameter,
        endpoint_records=(endpoint_record,),
    )


__all__ = [
    "BLUR_MECHANISM_RECEIPT_SCHEMA_V1",
    "BlurMechanismEvaluationReceiptV1",
    "evaluate_blur_mechanism_v1",
]
