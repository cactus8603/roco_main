"""Versioned local successor for the E261 two-arm blur design.

This module leaves the historical local-blur v2 path untouched.  It consumes
the outcome-blind v6 certificate, requires exactly one supported public arm,
and applies its selected disk, Gaussian, or motion PSF only to held-out 64 px
recoverable regions on the opposite endpoint.

It is a materialization primitive only.  Cost ceilings, child quartet
observables, scientific qualification, and selector admission remain separate
gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .blur_parameter_certificates import TILE, _kernel
from .blur_parameter_certificates_v6 import (
    ISOTROPIC_ARM_V6,
    MOTION_ARM_V6,
    action_specific_direct_winners_v6,
    blur_parameter_certificates_v6,
)
from .local_blur_successor import (
    _array_sha256,
    _canonical_sha256,
    _flow_support,
    _proposal,
    _validate,
)
from .operators import blend_local_proposal
from .support import transport_flow_mask_to_second


LOCAL_BLUR_DISPATCHER_ID_V3 = "blur_v6_two_arm.local_recoverable_v3"
LOCAL_BLUR_ACTION_IDS_V3 = {
    ISOTROPIC_ARM_V6: "common_isotropic.local_recoverable_v3",
    MOTION_ARM_V6: "common_motion.local_recoverable_v3",
}
LOCAL_BLUR_OPERATOR_IDS_V3 = {
    ISOTROPIC_ARM_V6: "common_isotropic_local_recoverable_cross_endpoint_v3",
    MOTION_ARM_V6: "common_motion_local_recoverable_cross_endpoint_v3",
}
LOCAL_BLUR_SUPPORT_POLICY_ID_V3 = "blur_v6_recoverable_region_support_v3"
LOCAL_BLUR_SCHEMA_V3 = "stablebridge-local-blur-successor/v3"

_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_BINDINGS = {
    "blur_v6": (
        _ROOT / "src/stablebridge/physical_repair/blur_parameter_certificates_v6.py",
        "4f4eb082196d52d5f3862fc3ec972a4c9a7aa851af7860e1e085ce78cc8ef7b4",
    ),
    "local_blur_v2_helpers": (
        _ROOT / "src/stablebridge/physical_repair/local_blur_successor.py",
        "a95bd3716ca63d0ecda668faf0eb8025c0d7db3e59f8575438a96b68d19380b9",
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


def _verify_sources() -> dict[str, str]:
    result = {}
    root = _ROOT.resolve()
    for name, (path, expected) in _SOURCE_BINDINGS.items():
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"local blur v3 source escaped workspace: {name}")
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"local blur v3 source drift: {name}; "
                f"expected {expected}, got {actual}"
            )
        result[name] = actual
    return result


def _base_receipt(
    *,
    pair: tuple[np.ndarray, np.ndarray],
    flow: np.ndarray,
    source_hashes: Mapping[str, str],
    winner_count: int,
) -> dict[str, Any]:
    return {
        "schema": LOCAL_BLUR_SCHEMA_V3,
        "dispatcher_id": LOCAL_BLUR_DISPATCHER_ID_V3,
        "candidate_action_ids": dict(LOCAL_BLUR_ACTION_IDS_V3),
        "candidate_operator_ids": dict(LOCAL_BLUR_OPERATOR_IDS_V3),
        "support_policy_id": LOCAL_BLUR_SUPPORT_POLICY_ID_V3,
        "action_id": None,
        "operator_id": None,
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
        "availability": "MATERIALIZABLE_TEST_ONLY",
        "cost_status": "UNBOUND",
        "ground_truth_read": False,
        "task_outcome_read": False,
        "scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
    }


@dataclass(frozen=True)
class LocalBlurSuccessorV3:
    status: str
    action_id: str | None
    operator_id: str | None
    public_arm: str | None
    exact_endpoint: str | None
    winner_id: str | None
    internal_family: str | None
    identified_endpoint: str | None
    selected_parameter: tuple[float, ...] | None
    recoverable_regions: tuple[tuple[int, int], ...]
    first_rgb: np.ndarray
    second_rgb: np.ndarray
    first_support: np.ndarray
    second_support: np.ndarray
    flow_support: np.ndarray
    first_read_support: np.ndarray
    second_read_support: np.ndarray
    receipt: Mapping[str, Any]


def _abstention(
    *,
    status: str,
    first: np.ndarray,
    second: np.ndarray,
    receipt: dict[str, Any],
    winner_id: str | None = None,
    public_arm: str | None = None,
    internal_family: str | None = None,
    identified_endpoint: str | None = None,
    selected_parameter: tuple[float, ...] | None = None,
    recoverable_regions: tuple[tuple[int, int], ...] = (),
) -> LocalBlurSuccessorV3:
    receipt["status"] = status
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    empty = np.zeros(first.shape[:2], dtype=bool)
    return LocalBlurSuccessorV3(
        status=status,
        action_id=None,
        operator_id=None,
        public_arm=public_arm,
        exact_endpoint=None,
        winner_id=winner_id,
        internal_family=internal_family,
        identified_endpoint=identified_endpoint,
        selected_parameter=selected_parameter,
        recoverable_regions=recoverable_regions,
        first_rgb=first.copy(),
        second_rgb=second.copy(),
        first_support=empty.copy(),
        second_support=empty.copy(),
        flow_support=empty.copy(),
        first_read_support=empty.copy(),
        second_read_support=empty.copy(),
        receipt=receipt,
    )


def _read_halo(write_support: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return the exact in-frame pixels read by a centered PSF write support."""
    footprint = np.ascontiguousarray(np.abs(kernel) > 0.0, dtype=np.uint8)
    expanded = cv2.dilate(
        np.ascontiguousarray(write_support, dtype=np.uint8),
        footprint,
        anchor=(kernel.shape[1] // 2, kernel.shape[0] // 2),
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.ascontiguousarray(expanded > 0)


def build_local_blur_successor_v3(
    observed_first_rgb: np.ndarray,
    observed_second_rgb: np.ndarray,
    observed_native_flow: np.ndarray,
) -> LocalBlurSuccessorV3:
    """Route the v6 two-arm certificate and materialize one local action."""

    source_hashes = _verify_sources()
    first, second, flow = _validate(
        observed_first_rgb, observed_second_rgb, observed_native_flow,
    )
    pair = (first, second)
    try:
        evidence = blur_parameter_certificates_v6(first, second, flow)
        winners = action_specific_direct_winners_v6(evidence)
    except (ValueError, RuntimeError) as exc:
        receipt = _base_receipt(
            pair=pair, flow=flow, source_hashes=source_hashes, winner_count=0,
        )
        receipt.update({
            "direct_winner_count_available": False,
            "bounded_error_type": type(exc).__name__,
        })
        return _abstention(
            status="INVALID_CERTIFICATE_V6",
            first=first,
            second=second,
            receipt=receipt,
        )

    receipt = _base_receipt(
        pair=pair, flow=flow, source_hashes=source_hashes,
        winner_count=len(winners),
    )
    if len(winners) != 1:
        receipt["direct_winners"] = list(winners)
        return _abstention(
            status="UNSUPPORTED_DIRECT_WINNER_COUNT",
            first=first,
            second=second,
            receipt=receipt,
        )

    winner_id = winners[0]
    item = evidence[winner_id]
    regions = tuple(tuple(map(int, value)) for value in item.recoverable_regions)
    parameter = tuple(map(float, item.selected_parameter))
    if item.status != "supported" or len(regions) < 4:
        receipt.update({
            "winner_id": winner_id,
            "public_arm": item.arm,
            "internal_operator_mode": item.internal_family,
            "recoverable_regions": [list(value) for value in regions],
            "rejection_reasons": list(item.rejection_reasons),
        })
        return _abstention(
            status="UNSUPPORTED_RECOVERABLE_REGIONS",
            first=first,
            second=second,
            receipt=receipt,
            winner_id=winner_id,
            public_arm=item.arm,
            internal_family=item.internal_family,
            identified_endpoint=item.endpoint,
            selected_parameter=parameter,
            recoverable_regions=regions,
        )

    action_id = LOCAL_BLUR_ACTION_IDS_V3[item.arm]
    operator_id = LOCAL_BLUR_OPERATOR_IDS_V3[item.arm]
    flow_weight = _flow_support(first.shape[:2], regions)
    output_first, output_second = first.copy(), second.copy()
    first_weight = np.zeros(first.shape[:2], dtype=np.float32)
    second_weight = np.zeros(first.shape[:2], dtype=np.float32)
    if item.endpoint == "first":
        exact_endpoint = "second"
        second_weight, splat_weight = transport_flow_mask_to_second(flow_weight, flow)
        candidate = _proposal(second, item.internal_family, parameter)
        output_second, action_record = blend_local_proposal(
            second,
            candidate,
            second_weight,
            operator_id=operator_id,
            endpoint="second",
            feather_sigma=1.0,
        )
        collision_fraction = float(np.mean(splat_weight > 1.00001))
        uncovered_fraction = float(np.mean(splat_weight <= 0.0))
    elif item.endpoint == "second":
        exact_endpoint = "first"
        first_weight = flow_weight.copy()
        candidate = _proposal(first, item.internal_family, parameter)
        output_first, action_record = blend_local_proposal(
            first,
            candidate,
            first_weight,
            operator_id=operator_id,
            endpoint="first",
            feather_sigma=1.0,
        )
        collision_fraction = 0.0
        uncovered_fraction = 0.0
    else:  # pragma: no cover - typed evidence already prevents this.
        raise RuntimeError("v6 blur winner endpoint drift")

    first_support = np.ascontiguousarray(first_weight > 0.0)
    second_support = np.ascontiguousarray(second_weight > 0.0)
    flow_support = np.ascontiguousarray(flow_weight > 0.0)
    changed_first = np.any(output_first != first, axis=2)
    changed_second = np.any(output_second != second, axis=2)
    if (
        np.any(changed_first & ~first_support)
        or np.any(changed_second & ~second_support)
    ):
        raise RuntimeError("local blur v3 successor escaped declared support")
    total_changed = int(changed_first.sum() + changed_second.sum())
    status = (
        "EXECUTED_LOCAL_RECOVERABLE_V3"
        if total_changed > 0 else "IDENTITY_ONLY_LOCAL_RECOVERABLE_V3"
    )
    kernel = _kernel(item.internal_family, parameter)
    first_read_support = _read_halo(first_support, kernel)
    second_read_support = _read_halo(second_support, kernel)
    if (
        np.any(first_support & ~first_read_support)
        or np.any(second_support & ~second_read_support)
    ):
        raise RuntimeError("local blur v3 write support escaped its read halo")
    winner_evidence = {
        "public_arm": item.arm,
        "internal_operator_mode": item.internal_family,
        "identified_endpoint": item.endpoint,
        "selected_parameter": list(parameter),
        "source_action_key": item.source_action_key,
        "fit_support_sha256": item.fit_support_sha256,
        "check_support_sha256": item.check_support_sha256,
        "fit_score": item.fit_score,
        "recoverable_regions": [list(value) for value in regions],
        "certificate": asdict(item.certificate),
    }
    receipt.update({
        "status": status,
        "action_id": action_id,
        "operator_id": operator_id,
        "winner_id": winner_id,
        "public_arm": item.arm,
        "internal_operator_mode": item.internal_family,
        "source_action_key": item.source_action_key,
        "fit_support_sha256": item.fit_support_sha256,
        "check_support_sha256": item.check_support_sha256,
        "identified_endpoint": item.endpoint,
        "modified_endpoint": exact_endpoint,
        "strength": {
            "status": "CASE_BOUND_PHYSICAL_PARAMETER",
            "parameter": list(parameter),
        },
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
        "endpoint_read_support_sha256s": {
            "first": _array_sha256(first_read_support),
            "second": _array_sha256(second_read_support),
        },
        "endpoint_read_support_pixels": {
            "first": int(first_read_support.sum()),
            "second": int(second_read_support.sum()),
        },
        "kernel_support_sha256": _array_sha256(
            np.ascontiguousarray(np.abs(kernel) > 0.0)
        ),
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
    return LocalBlurSuccessorV3(
        status=status,
        action_id=action_id,
        operator_id=operator_id,
        public_arm=item.arm,
        exact_endpoint=exact_endpoint if total_changed > 0 else None,
        winner_id=winner_id,
        internal_family=item.internal_family,
        identified_endpoint=item.endpoint,
        selected_parameter=parameter,
        recoverable_regions=regions,
        first_rgb=np.ascontiguousarray(output_first),
        second_rgb=np.ascontiguousarray(output_second),
        first_support=first_support,
        second_support=second_support,
        flow_support=flow_support,
        first_read_support=first_read_support,
        second_read_support=second_read_support,
        receipt=receipt,
    )


__all__ = [
    "LOCAL_BLUR_ACTION_IDS_V3",
    "LOCAL_BLUR_DISPATCHER_ID_V3",
    "LOCAL_BLUR_OPERATOR_IDS_V3",
    "LOCAL_BLUR_SCHEMA_V3",
    "LOCAL_BLUR_SUPPORT_POLICY_ID_V3",
    "LocalBlurSuccessorV3",
    "build_local_blur_successor_v3",
]
