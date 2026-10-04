"""Mechanism evaluator for block-exact JPEG local successors v3."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import cv2
import numpy as np

from .jpeg_action_certificates import (
    CODEC_NONINFERIORITY_255,
    MACRO_TILE,
    _boundary_excess,
)
from .jpeg_codec_path_actions import _quantization_violation
from .jpeg_mechanism_evaluators import _shifted_phase_proposal, _validate
from .jpeg_quantization_actions import (
    DCT_ROUNDING_BOUND,
    _jpeg_roundtrip,
)
from .local_jpeg_successors import jpeg_macroblock_support_v2
from .local_jpeg_successors_v3 import (
    LOCAL_JPEG_CODEC_ACTION_ID_V3,
    LOCAL_JPEG_CODEC_OPERATOR_ID_V3,
    LOCAL_JPEG_QCELL_ACTION_ID_V3,
    LOCAL_JPEG_QCELL_OPERATOR_ID_V3,
    _block_exact_composite,
)


JPEG_MECHANISM_RECEIPT_SCHEMA_V3 = "stablebridge-jpeg-mechanism-evaluation/v3"
_POLICIES = {
    LOCAL_JPEG_QCELL_ACTION_ID_V3: {
        "policy": "qcell",
        "operator_id": LOCAL_JPEG_QCELL_OPERATOR_ID_V3,
        "metric_id": "jpeg_qcell_block_exact_boundary_excess_reduction_v3",
    },
    LOCAL_JPEG_CODEC_ACTION_ID_V3: {
        "policy": "codec",
        "operator_id": LOCAL_JPEG_CODEC_OPERATOR_ID_V3,
        "metric_id": "jpeg_codec_path_block_exact_boundary_excess_reduction_v3",
    },
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


@dataclass(frozen=True)
class JPEGMechanismEvaluationReceiptV3:
    action_id: str
    metric_id: str
    physical_pair_sha256: str
    endpoint: str
    estimated_quality: int
    selected_strength: float
    endpoint_records: tuple[dict[str, Any], ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        policy = _POLICIES.get(self.action_id)
        if policy is None or self.metric_id != policy["metric_id"]:
            raise ValueError("JPEG v3 mechanism action/metric identity drift")
        if self.endpoint not in {"first", "second"} or len(self.endpoint_records) != 1:
            raise ValueError("JPEG v3 mechanism endpoint record drift")
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("JPEG v3 mechanism receipt hash drift")
        object.__setattr__(self, "receipt_sha256", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema": JPEG_MECHANISM_RECEIPT_SCHEMA_V3,
            "action_id": self.action_id,
            "metric_id": self.metric_id,
            "physical_pair_sha256": self.physical_pair_sha256,
            "endpoint": self.endpoint,
            "estimated_ijg_quality": self.estimated_quality,
            "selected_strength": self.selected_strength,
            "endpoint_records": list(self.endpoint_records),
            "estimand": {
                "efficacy_signal": "aligned_8px_block_boundary_excess_255",
                "negative_control": (
                    "same proposal path at block origin (4,4), composed by "
                    "the same native-grid block-exact support"
                ),
                "mandatory_safety": [
                    "observed qcell containment",
                    "codec re-encode closure for codec policy",
                    "outside-support byte identity",
                ],
            },
            "composition_policy": "ALIGNED_8PX_BLOCK_EXACT_SELECTION_V1",
            "runtime_inputs": [
                "before_endpoint", "after_endpoint", "before_only_support",
                "receipt_bound_quality", "receipt_bound_strength",
            ],
            "ground_truth_read": False,
            "task_outcome_read": False,
            "scientific_qualification": False,
            "selector_admission": False,
            "production_authority": False,
        }
        if include_hash:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def evaluate_jpeg_mechanism_v3(
    *,
    action_id: str,
    physical_pair_sha256: str,
    endpoint: str,
    before_endpoint: np.ndarray,
    after_endpoint: np.ndarray,
    endpoint_support: np.ndarray,
    estimated_quality: int,
    selected_strength: float,
) -> JPEGMechanismEvaluationReceiptV3:
    policy = _POLICIES.get(action_id)
    if policy is None or endpoint not in {"first", "second"}:
        raise ValueError("unsupported JPEG v3 mechanism policy or endpoint")
    if type(estimated_quality) is not int or not 1 <= estimated_quality <= 95:
        raise ValueError("invalid receipt-bound JPEG quality")
    before, after, support = _validate(
        before_endpoint, after_endpoint, endpoint_support,
    )
    expected_support, tile_evidence = jpeg_macroblock_support_v2(before)
    if not np.array_equal(support, expected_support):
        raise RuntimeError("JPEG v3 mechanism before-only support drift")
    if not support.any():
        raise RuntimeError("JPEG v3 mechanism support is empty")
    if not np.array_equal(before[~support], after[~support]):
        raise RuntimeError("JPEG v3 mechanism output escaped support")

    shifted = _shifted_phase_proposal(
        before, policy=str(policy["policy"]), quality=estimated_quality,
        strength=float(selected_strength),
    )
    negative = _block_exact_composite(before, shifted, support)
    gray_before = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_after = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_negative = cv2.cvtColor(negative, cv2.COLOR_RGB2GRAY).astype(np.float32)
    region_rows = []
    for tile in tile_evidence:
        if not bool(tile["active"]):
            continue
        y0, x0 = map(int, tile["origin_y_x"])
        ys, xs = slice(y0, y0 + MACRO_TILE), slice(x0, x0 + MACRO_TILE)
        region_rows.append({
            "origin_y_x": [y0, x0],
            "native_boundary_excess_255": max(float(
                _boundary_excess(gray_before[ys, xs])
            ), 0.0),
            "action_boundary_excess_255": max(float(
                _boundary_excess(gray_after[ys, xs])
            ), 0.0),
            "shifted_control_boundary_excess_255": max(float(
                _boundary_excess(gray_negative[ys, xs])
            ), 0.0),
        })
    before_signal = float(np.mean([
        row["native_boundary_excess_255"] for row in region_rows
    ]))
    after_signal = float(np.mean([
        row["action_boundary_excess_255"] for row in region_rows
    ]))
    negative_signal = float(np.mean([
        row["shifted_control_boundary_excess_255"] for row in region_rows
    ]))
    if before_signal <= 1e-12 or not all(map(
        math.isfinite, (before_signal, after_signal, negative_signal),
    )):
        raise RuntimeError("JPEG v3 mechanism signal is unestimable")
    active_reduction = float((before_signal - after_signal) / before_signal)
    negative_reduction = float((before_signal - negative_signal) / before_signal)
    specificity = float(active_reduction - negative_reduction)
    violation = _quantization_violation(before, after, estimated_quality)
    if violation > DCT_ROUNDING_BOUND:
        raise RuntimeError("JPEG v3 action violated its frozen quantization cells")

    reencoded_before = _jpeg_roundtrip(before, estimated_quality)
    reencoded_after = _jpeg_roundtrip(after, estimated_quality)
    closure_excess = []
    for region in region_rows:
        y0, x0 = map(int, region["origin_y_x"])
        ys, xs = slice(y0, y0 + MACRO_TILE), slice(x0, x0 + MACRO_TILE)
        baseline = float(np.mean(np.abs(
            reencoded_before[ys, xs].astype(np.float32)
            - before[ys, xs].astype(np.float32)
        )))
        action = float(np.mean(np.abs(
            reencoded_after[ys, xs].astype(np.float32)
            - before[ys, xs].astype(np.float32)
        )))
        closure_excess.append(action - baseline)
    maximum_closure_excess = float(max(closure_excess))
    if (
        policy["policy"] == "codec"
        and maximum_closure_excess > CODEC_NONINFERIORITY_255
    ):
        raise RuntimeError("JPEG v3 codec path violated re-encode closure")
    endpoint_record = {
        "endpoint": endpoint,
        "support_sha256": _array_sha256(support),
        "support_pixels": int(support.sum()),
        "active_macro_tiles": len(region_rows),
        "active_signal_before": before_signal,
        "active_signal_after": after_signal,
        "active_relative_reduction": active_reduction,
        "negative_control_signal_after": negative_signal,
        "negative_control_relative_reduction": negative_reduction,
        "specificity_difference": specificity,
        "quantization_violation_max_dct": violation,
        "quantization_rounding_bound_dct": DCT_ROUNDING_BOUND,
        "codec_closure_excess_max_255": maximum_closure_excess,
        "codec_closure_bound_255": CODEC_NONINFERIORITY_255,
        "outside_support_byte_identity": True,
        "negative_control_composition": "ALIGNED_8PX_BLOCK_EXACT_SELECTION_V1",
        "region_records": region_rows,
    }
    return JPEGMechanismEvaluationReceiptV3(
        action_id=action_id,
        metric_id=str(policy["metric_id"]),
        physical_pair_sha256=physical_pair_sha256,
        endpoint=endpoint,
        estimated_quality=estimated_quality,
        selected_strength=float(selected_strength),
        endpoint_records=(endpoint_record,),
    )


__all__ = [
    "JPEG_MECHANISM_RECEIPT_SCHEMA_V3",
    "JPEGMechanismEvaluationReceiptV3",
    "evaluate_jpeg_mechanism_v3",
]
