"""Versioned local successor for the routed blur equalization policy.

The frozen v3 certificate identifies one degraded endpoint, one physical PSF
family/parameter, and held-out 64 px regions where forward re-degradation is
recoverable.  The historical Work-E adapter retained the branch decision but
blurred the entire opposite endpoint.  This module preserves the decision and
operator while restricting the intervention to those recoverable regions.

This is an execution/mechanism primitive only.  It grants no scientific,
selector, cost, or production authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .blur_parameter_certificates import TILE, _apply, _kernel
from .blur_parameter_certificates_v3 import (
    action_specific_direct_winners,
    blur_parameter_certificates_v3,
)
from .operators import blend_local_proposal
from .support import transport_flow_mask_to_second


LOCAL_BLUR_ACTION_ID_V2 = "blur_v3_cross_endpoint.local_recoverable_v2"
LOCAL_BLUR_OPERATOR_ID_V2 = "blur_v3_local_recoverable_cross_endpoint_v2"
LOCAL_BLUR_SCHEMA_V2 = "stablebridge-local-blur-successor/v2"
_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_BINDINGS = {
    "blur_base": (
        _ROOT / "src/stablebridge/physical_repair/blur_parameter_certificates.py",
        "80a924cab1b5dcb804906072c49f947742a70c71c8e95d288241b261fec3bb39",
    ),
    "blur_v2": (
        _ROOT / "src/stablebridge/physical_repair/blur_parameter_certificates_v2.py",
        "0a08990e75d3bfb5970d0de27d31c12ea456648ffa54d64df5878f4f09cb7dab",
    ),
    "blur_v3": (
        _ROOT / "src/stablebridge/physical_repair/blur_parameter_certificates_v3.py",
        "9012877a9d1ed58caba8bc21fb6835fa0f3912249b16a0dddfac337f31f3c815",
    ),
    "local_compositor": (
        _ROOT / "src/stablebridge/physical_repair/operators.py",
        "d638c3c0d7ba7f497b96749b4c81895a35055c16a2f1d85becd0a8b00f766c2e",
    ),
    "support_transport": (
        _ROOT / "src/stablebridge/physical_repair/support.py",
        "61f234b7c04a0763d2d5101ae2172be1ce262e5db7904a35e75cdd4a0ec097fa",
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


def _verify_sources() -> dict[str, str]:
    result = {}
    root = _ROOT.resolve()
    for name, (path, expected) in _SOURCE_BINDINGS.items():
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"local blur source escaped workspace: {name}")
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"local blur source drift: {name}; expected {expected}, got {actual}"
            )
        result[name] = actual
    return result


def _validate(
    first: np.ndarray, second: np.ndarray, flow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(first)
    second = np.asarray(second)
    flow = np.asarray(flow)
    if (
        first.dtype != np.uint8 or second.dtype != np.uint8
        or first.ndim != 3 or first.shape != second.shape or first.shape[2] != 3
        or min(first.shape[:2]) <= 0
        or flow.dtype != np.float32 or flow.shape != (*first.shape[:2], 2)
        or not np.isfinite(flow).all()
    ):
        raise ValueError("expected paired uint8 RGB and finite float32 HxWx2 flow")
    return (
        np.ascontiguousarray(first), np.ascontiguousarray(second),
        np.ascontiguousarray(flow),
    )


def _flow_support(
    shape: tuple[int, int], regions: tuple[tuple[int, int], ...],
) -> np.ndarray:
    support = np.zeros(shape, dtype=np.float32)
    for region in regions:
        if len(region) != 2:
            raise RuntimeError("blur recoverable region geometry drift")
        y0, x0 = map(int, region)
        if (
            y0 < 0 or x0 < 0 or y0 + TILE > shape[0]
            or x0 + TILE > shape[1] or y0 % TILE or x0 % TILE
        ):
            raise RuntimeError("blur recoverable region escaped the 64px lattice")
        support[y0:y0 + TILE, x0:x0 + TILE] = 1.0
    return np.ascontiguousarray(support)


def _proposal(image: np.ndarray, family: str, parameter: tuple[float, ...]) -> np.ndarray:
    filtered = _apply(image.astype(np.float32), family, parameter)
    return np.ascontiguousarray(np.clip(np.rint(filtered), 0, 255).astype(np.uint8))


def _base_receipt(
    *, pair: tuple[np.ndarray, np.ndarray], flow: np.ndarray,
    source_hashes: Mapping[str, str], winner_count: int,
) -> dict[str, Any]:
    return {
        "schema": LOCAL_BLUR_SCHEMA_V2,
        "action_id": LOCAL_BLUR_ACTION_ID_V2,
        "operator_id": LOCAL_BLUR_OPERATOR_ID_V2,
        "input_sha256s": {
            "first": _array_sha256(pair[0]),
            "second": _array_sha256(pair[1]),
            "native_flow": _array_sha256(flow),
        },
        "direct_winner_count": int(winner_count),
        "source_manifest_sha256": _canonical_sha256(dict(source_hashes)),
        "runtime_inputs": [
            "observed_first_rgb", "observed_second_rgb", "observed_native_flow",
        ],
        "ground_truth_read": False,
        "task_outcome_read": False,
        "scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
    }


@dataclass(frozen=True)
class LocalBlurSuccessorV2:
    status: str
    exact_endpoint: str | None
    winner_id: str | None
    family: str | None
    identified_endpoint: str | None
    selected_parameter: tuple[float, ...] | None
    recoverable_regions: tuple[tuple[int, int], ...]
    first_rgb: np.ndarray
    second_rgb: np.ndarray
    first_support: np.ndarray
    second_support: np.ndarray
    flow_support: np.ndarray
    receipt: Mapping[str, Any]


def build_local_blur_successor_v2(
    observed_first_rgb: np.ndarray,
    observed_second_rgb: np.ndarray,
    observed_native_flow: np.ndarray,
) -> LocalBlurSuccessorV2:
    """Route v3 once and locally equalize only its recoverable regions."""

    source_hashes = _verify_sources()
    first, second, flow = _validate(
        observed_first_rgb, observed_second_rgb, observed_native_flow,
    )
    pair = (first, second)
    try:
        evidence = blur_parameter_certificates_v3(first, second, flow)
        winners = action_specific_direct_winners(evidence)
    except (ValueError, RuntimeError) as exc:
        receipt = _base_receipt(
            pair=pair, flow=flow, source_hashes=source_hashes, winner_count=0,
        )
        receipt.update({
            "status": "INVALID_CERTIFICATE_V3",
            "direct_winner_count_available": False,
            "bounded_error_type": type(exc).__name__,
        })
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        empty = np.zeros(first.shape[:2], dtype=bool)
        return LocalBlurSuccessorV2(
            status=str(receipt["status"]), exact_endpoint=None,
            winner_id=None, family=None, identified_endpoint=None,
            selected_parameter=None, recoverable_regions=(),
            first_rgb=first.copy(), second_rgb=second.copy(),
            first_support=empty.copy(), second_support=empty.copy(),
            flow_support=empty.copy(), receipt=receipt,
        )
    receipt = _base_receipt(
        pair=pair, flow=flow, source_hashes=source_hashes,
        winner_count=len(winners),
    )
    empty = np.zeros(first.shape[:2], dtype=bool)
    empty_flow = np.zeros(first.shape[:2], dtype=bool)
    if len(winners) != 1:
        receipt.update({
            "status": "UNSUPPORTED_DIRECT_WINNER_COUNT",
            "direct_winners": list(winners),
        })
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        return LocalBlurSuccessorV2(
            status=str(receipt["status"]), exact_endpoint=None,
            winner_id=None, family=None, identified_endpoint=None,
            selected_parameter=None, recoverable_regions=(),
            first_rgb=first.copy(), second_rgb=second.copy(),
            first_support=empty.copy(), second_support=empty.copy(),
            flow_support=empty_flow, receipt=receipt,
        )

    winner_id = winners[0]
    item = evidence[winner_id]
    regions = tuple(tuple(map(int, value)) for value in item.recoverable_regions)
    if item.recoverability_status != "supported" or not regions:
        receipt.update({
            "status": "UNSUPPORTED_RECOVERABLE_REGIONS",
            "winner_id": winner_id,
            "recoverability_status": item.recoverability_status,
            "recoverable_regions": [list(value) for value in regions],
        })
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        return LocalBlurSuccessorV2(
            status=str(receipt["status"]), exact_endpoint=None,
            winner_id=winner_id, family=item.family,
            identified_endpoint=item.endpoint,
            selected_parameter=tuple(map(float, item.selected_parameter)),
            recoverable_regions=regions,
            first_rgb=first.copy(), second_rgb=second.copy(),
            first_support=empty.copy(), second_support=empty.copy(),
            flow_support=empty_flow, receipt=receipt,
        )

    parameter = tuple(map(float, item.selected_parameter))
    flow_weight = _flow_support(first.shape[:2], regions)
    output_first, output_second = first.copy(), second.copy()
    first_weight = np.zeros(first.shape[:2], dtype=np.float32)
    second_weight = np.zeros(first.shape[:2], dtype=np.float32)
    if item.endpoint == "first":
        exact_endpoint = "second"
        second_weight, splat_weight = transport_flow_mask_to_second(flow_weight, flow)
        hard_weight = second_weight
        candidate = _proposal(second, item.family, parameter)
        output_second, action_record = blend_local_proposal(
            second, candidate, hard_weight,
            operator_id=LOCAL_BLUR_OPERATOR_ID_V2,
            endpoint="second", feather_sigma=1.0,
        )
        collision_fraction = float(np.mean(splat_weight > 1.00001))
        uncovered_fraction = float(np.mean(splat_weight <= 0.0))
    elif item.endpoint == "second":
        exact_endpoint = "first"
        first_weight = flow_weight.copy()
        candidate = _proposal(first, item.family, parameter)
        output_first, action_record = blend_local_proposal(
            first, candidate, first_weight,
            operator_id=LOCAL_BLUR_OPERATOR_ID_V2,
            endpoint="first", feather_sigma=1.0,
        )
        collision_fraction = 0.0
        uncovered_fraction = 0.0
    else:  # defensive: BlurFamilyEvidence already validates this.
        raise RuntimeError("blur winner endpoint drift")

    first_support = np.ascontiguousarray(first_weight > 0.0)
    second_support = np.ascontiguousarray(second_weight > 0.0)
    flow_support = np.ascontiguousarray(flow_weight > 0.0)
    changed_first = np.any(output_first != first, axis=2)
    changed_second = np.any(output_second != second, axis=2)
    if (
        np.any(changed_first & ~first_support)
        or np.any(changed_second & ~second_support)
    ):
        raise RuntimeError("local blur successor escaped declared support")
    total_changed = int(changed_first.sum() + changed_second.sum())
    status = (
        "EXECUTED_LOCAL_RECOVERABLE_V2"
        if total_changed > 0 else "IDENTITY_ONLY_LOCAL_RECOVERABLE_V2"
    )
    kernel = _kernel(item.family, parameter)
    winner_evidence = {
        "family": item.family,
        "identified_endpoint": item.endpoint,
        "selected_parameter": list(parameter),
        "fit_support_sha256": item.fit_support_hash,
        "check_support_sha256": item.check_support_hash,
        "recoverability_status": item.recoverability_status,
        "recoverable_regions": [list(value) for value in regions],
        "certificate": asdict(item.certificate),
    }
    receipt.update({
        "status": status,
        "winner_id": winner_id,
        "family": item.family,
        "identified_endpoint": item.endpoint,
        "modified_endpoint": exact_endpoint,
        "selected_parameter": list(parameter),
        "selected_parameter_sha256": _canonical_sha256(list(parameter)),
        "winner_evidence_sha256": _canonical_sha256(winner_evidence),
        "recoverable_regions": [list(value) for value in regions],
        "tile_size_px": TILE,
        "flow_support_sha256": _array_sha256(flow_support),
        "flow_support_pixels": int(flow_support.sum()),
        "endpoint_support_sha256s": {
            "first": _array_sha256(first_support),
            "second": _array_sha256(second_support),
        },
        "endpoint_support_pixels": {
            "first": int(first_support.sum()),
            "second": int(second_support.sum()),
        },
        "output_sha256s": {
            "first": _array_sha256(output_first),
            "second": _array_sha256(output_second),
        },
        "changed_mask_sha256s": {
            "first": _array_sha256(changed_first),
            "second": _array_sha256(changed_second),
        },
        "changed_pixels": {
            "first": int(changed_first.sum()),
            "second": int(changed_second.sum()),
        },
        "outside_support_byte_identity": bool(
            np.array_equal(output_first[~first_support], first[~first_support])
            and np.array_equal(output_second[~second_support], second[~second_support])
        ),
        "operator_kernel_radius_px": int(max(kernel.shape) // 2),
        "support_feather_sigma_px": 1.0,
        "support_transport_policy": (
            "FLOW_NATIVE_RECOVERABLE_TILES_BILINEAR_SPLAT_TO_SECOND_V1"
            if exact_endpoint == "second"
            else "FLOW_NATIVE_RECOVERABLE_TILES_DIRECT_FIRST_V1"
        ),
        "second_transport_collision_fraction": collision_fraction,
        "second_transport_uncovered_fraction": uncovered_fraction,
        "local_action_record": asdict(action_record),
    })
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return LocalBlurSuccessorV2(
        status=status,
        exact_endpoint=exact_endpoint if total_changed > 0 else None,
        winner_id=winner_id,
        family=item.family,
        identified_endpoint=item.endpoint,
        selected_parameter=parameter,
        recoverable_regions=regions,
        first_rgb=np.ascontiguousarray(output_first),
        second_rgb=np.ascontiguousarray(output_second),
        first_support=first_support,
        second_support=second_support,
        flow_support=flow_support,
        receipt=receipt,
    )


__all__ = [
    "LOCAL_BLUR_ACTION_ID_V2",
    "LOCAL_BLUR_OPERATOR_ID_V2",
    "LOCAL_BLUR_SCHEMA_V2",
    "LocalBlurSuccessorV2",
    "build_local_blur_successor_v2",
]
