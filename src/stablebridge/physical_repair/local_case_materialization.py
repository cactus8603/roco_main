"""Typed local-output receipts for Wave-1 local action sentinels.

This module proves a narrow execution property: a discovered impulse or Wiener
or routed-blur control produces a full-strength child pair whose changed pixels are contained
in its exact before-only endpoint supports and whose complement is byte-exact.
It does not grant scientific, selector, cost, or production authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .action_bank_contracts import ActionEndpointV2, CaseBindingPrerequisitesV2


LOCAL_CASE_MATERIALIZATION_SCHEMA_V1 = "stablebridge-local-case-materialization/v1"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_BINDINGS = {
    "local_operator": (
        _ROOT / "src/stablebridge/physical_repair/operators.py",
        "d638c3c0d7ba7f497b96749b4c81895a35055c16a2f1d85becd0a8b00f766c2e",
    ),
    "support_transport": (
        _ROOT / "src/stablebridge/physical_repair/support.py",
        "61f234b7c04a0763d2d5101ae2172be1ce262e5db7904a35e75cdd4a0ec097fa",
    ),
    "noise_certificate": (
        _ROOT / "src/stablebridge/physical_repair/noise_certificates.py",
        "459d1e97b7b11cbb9bc58eed3ea9052f9d2f53d5b0437007a0cb99daab2bba55",
    ),
    "paired_noise_certificate": (
        _ROOT / "src/stablebridge/physical_repair/paired_noise_certificates.py",
        "6c7031197f81862d1f6959561f5b0d2d600bc235172ab2b38b8db054a0237848",
    ),
    "exact_control_executor": (
        _ROOT / "research/region_aware_action_learning_20261001/work_package_E/executor_adapter.py",
        "28b7742ad7a9baf44f7373fc5bc108d60643f8a25980dc258c42df6529e50a1e",
    ),
    "local_wiener_successor": (
        _ROOT / "src/stablebridge/physical_repair/local_wiener_successor.py",
        "11e23d5449c66629e31fa665e704e860831fe6a0092623937ae4836cd05ec673",
    ),
    "local_blur_successor": (
        _ROOT / "src/stablebridge/physical_repair/local_blur_successor.py",
        "a95bd3716ca63d0ecda668faf0eb8025c0d7db3e59f8575438a96b68d19380b9",
    ),
    "local_jpeg_successors": (
        _ROOT / "src/stablebridge/physical_repair/local_jpeg_successors.py",
        "ba675f0e5e557d3104c41f86c22616c0113f88e0837ba11a35193eb8eb3c9505",
    ),
}
_POLICIES = {
    "paired_impulse_median3.endpoint_supported_v1": {
        "control_id": "paired_impulse_median3",
        "operator_id": "impulse_exact_median3",
        "image_read_dependency": "FINITE_3X3_RADIUS_1_PX",
        "support_read_dependency": (
            "PAIRED_FLOW_WARP_EXACT_EXTREMA_AND_ABSOLUTE_DIFFERENCE_GE_48"
        ),
        "support_transport_policy": "WORK_E_IMPULSE_MAPS_V1",
    },
    "paired_additive_wiener3.endpoint_supported_v1": {
        "control_id": "paired_additive_wiener3",
        "operator_id": "wiener3",
        "image_read_dependency": (
            "FULL_ENDPOINT_ROBUST_NOISE_ESTIMATE_PLUS_3X3_RADIUS_1_PX_FILTER"
        ),
        "support_read_dependency": "ENDPOINT_NATIVE_32PX_TILE_ROBUST_Z_GE_4_SIGMA_GE_2",
        "support_transport_policy": "ENDPOINT_NATIVE_NO_OUTPUT_TRANSPORT",
    },
    "paired_additive_wiener3.local_tile_sigma_v2": {
        "control_id": "paired_additive_wiener3.local_tile_sigma_v2",
        "operator_id": "wiener3_local_tile_sigma_v2",
        "image_read_dependency": (
            "ACTIVE_32PX_TILE_ROBUST_SIGMA_PLUS_3X3_RADIUS_1_PX_FILTER"
        ),
        "support_read_dependency": "ENDPOINT_NATIVE_32PX_TILE_ROBUST_Z_GE_4_SIGMA_GE_2",
        "support_transport_policy": "ENDPOINT_NATIVE_NO_OUTPUT_TRANSPORT",
    },
    "blur_v3_cross_endpoint.local_recoverable_v2": {
        "control_id": "blur_v3_cross_endpoint.local_recoverable_v2",
        "operator_id": "blur_v3_local_recoverable_cross_endpoint_v2",
        "image_read_dependency": (
            "CASE_SELECTED_PSF_FINITE_KERNEL_RADIUS_ON_OPPOSITE_ENDPOINT"
        ),
        "support_read_dependency": (
            "V3_HELD_OUT_64PX_RECOVERABLE_REGIONS_AFTER_UNIQUE_DIRECT_WINNER"
        ),
        "support_transport_policy": (
            "FLOW_NATIVE_RECOVERABLE_TILES_DIRECT_FIRST_OR_BILINEAR_SPLAT_SECOND_V1"
        ),
    },
    "jpeg_qcell_v3.local_macroblock_v2": {
        "control_id": "jpeg_qcell_v3.local_macroblock_v2",
        "operator_id": "jpeg_qcell_local_macroblock_v2",
        "image_read_dependency": (
            "FULL_ENDPOINT_QCELL_PROPOSAL_THEN_64PX_MACROBLOCK_HARD_REMASK"
        ),
        "support_read_dependency": (
            "PRE_ACTION_ALIGNED_64PX_MACRO_TILE_8PX_BOUNDARY_EXCESS_GT_1_255"
        ),
        "support_transport_policy": "ENDPOINT_NATIVE_NO_OUTPUT_TRANSPORT",
    },
    "jpeg_codec_path_v4.local_macroblock_v2": {
        "control_id": "jpeg_codec_path_v4.local_macroblock_v2",
        "operator_id": "jpeg_codec_path_local_macroblock_v2",
        "image_read_dependency": (
            "FULL_ENDPOINT_DYADIC_CODEC_PATH_SELECTION_THEN_64PX_MACROBLOCK_HARD_REMASK"
        ),
        "support_read_dependency": (
            "PRE_ACTION_ALIGNED_64PX_MACRO_TILE_8PX_BOUNDARY_EXCESS_GT_1_255"
        ),
        "support_transport_policy": "ENDPOINT_NATIVE_NO_OUTPUT_TRANSPORT",
    },
}


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _role_sha256(role: str, value: np.ndarray) -> str:
    return _canonical_sha256({
        "role": role,
        "array_sha256": _array_sha256(value),
        "hash_contract": "dtype-shape-bytes/v1",
    })


def _require_sha(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _verify_sources() -> dict[str, str]:
    result = {}
    root = _ROOT.resolve()
    for name, (path, expected) in _SOURCE_BINDINGS.items():
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"local materializer source escaped workspace: {name}")
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"local materializer source drift: {name}; expected {expected}, got {actual}"
            )
        result[name] = actual
    return result


def _validate_arrays(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    output_first: np.ndarray,
    output_second: np.ndarray,
    supports: Sequence[np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    first = np.asarray(first)
    second = np.asarray(second)
    flow = np.asarray(flow)
    output_first = np.asarray(output_first)
    output_second = np.asarray(output_second)
    if (
        first.dtype != np.uint8 or second.dtype != np.uint8
        or first.ndim != 3 or first.shape != second.shape or first.shape[2] != 3
        or flow.dtype != np.float32 or flow.shape != (*first.shape[:2], 2)
        or not np.isfinite(flow).all()
        or output_first.dtype != np.uint8 or output_second.dtype != np.uint8
        or output_first.shape != first.shape or output_second.shape != second.shape
        or len(supports) != 2
    ):
        raise ValueError("invalid local materialization arrays")
    support_pair = tuple(np.asarray(value) for value in supports)
    if any(
        value.dtype != np.bool_ or value.shape != first.shape[:2]
        for value in support_pair
    ):
        raise ValueError("local supports must be two HxW boolean arrays")
    return (
        (np.ascontiguousarray(first), np.ascontiguousarray(second)),
        np.ascontiguousarray(flow),
        (np.ascontiguousarray(output_first), np.ascontiguousarray(output_second)),
        tuple(np.ascontiguousarray(value) for value in support_pair),
    )


@dataclass(frozen=True)
class LocalCaseMaterializationReceiptV1:
    qualification_action_id: str
    exact_control_id: str
    operator_id: str
    case_id: str
    physical_pair_sha256: str
    exact_endpoint: str
    endpoint_sha256: str
    source_input_sha256s: tuple[str, str, str]
    support_realization_sha256: str
    rollback_sha256: str
    materialization_recipe_sha256: str
    endpoint_records: tuple[Mapping[str, Any], Mapping[str, Any]]
    image_read_dependency: str
    support_read_dependency: str
    support_transport_policy: str
    source_manifest_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.qualification_action_id not in _POLICIES:
            raise ValueError("unsupported local Wave-1 policy")
        policy = _POLICIES[self.qualification_action_id]
        if self.exact_control_id != policy["control_id"]:
            raise ValueError("exact control does not match local policy")
        if self.operator_id != policy["operator_id"]:
            raise ValueError("operator does not match local policy")
        if self.exact_endpoint not in {"first", "second", "both"}:
            raise ValueError("invalid exact endpoint")
        for name in (
            "physical_pair_sha256", "endpoint_sha256", "support_realization_sha256",
            "rollback_sha256", "materialization_recipe_sha256",
            "source_manifest_sha256",
        ):
            _require_sha(getattr(self, name), name)
        if len(self.source_input_sha256s) != 3:
            raise ValueError("three role-bound source hashes are required")
        for index, value in enumerate(self.source_input_sha256s):
            _require_sha(value, f"source input {index}")
        if len(set(self.source_input_sha256s)) != 3:
            raise ValueError("role-bound source hashes must be unique")
        if len(self.endpoint_records) != 2:
            raise ValueError("two endpoint records are required")
        payload = self.as_dict(include_hash=False)
        expected = _canonical_sha256(payload)
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("local materialization receipt hash drift")
        object.__setattr__(self, "receipt_sha256", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema": LOCAL_CASE_MATERIALIZATION_SCHEMA_V1,
            "qualification_action_id": self.qualification_action_id,
            "exact_control_id": self.exact_control_id,
            "operator_id": self.operator_id,
            "case_id": self.case_id,
            "physical_pair_sha256": self.physical_pair_sha256,
            "exact_endpoint": self.exact_endpoint,
            "endpoint_sha256": self.endpoint_sha256,
            "source_input_sha256s": list(self.source_input_sha256s),
            "support_realization_sha256": self.support_realization_sha256,
            "rollback_sha256": self.rollback_sha256,
            "materialization_recipe_sha256": self.materialization_recipe_sha256,
            "endpoint_records": [dict(value) for value in self.endpoint_records],
            "image_read_dependency": self.image_read_dependency,
            "support_read_dependency": self.support_read_dependency,
            "support_transport_policy": self.support_transport_policy,
            "source_manifest_sha256": self.source_manifest_sha256,
            "composition": "FULL_STRENGTH_LOCAL_OUTPUT_HARD_REMASK_FEATHER_SIGMA_1_V1",
            "input_strength": 1.0,
            "outside_support_byte_identity": True,
            "case_binding_prerequisites_ready": False,
            "case_binding_missing": ["prospective_cost_ceiling_receipt"],
            "scientific_qualification": False,
            "selector_admission": False,
            "production_authority": False,
        }
        if include_hash:
            result["receipt_sha256"] = self.receipt_sha256
        return result

    def to_case_binding_prerequisites_v2(
        self, *, cost_ceiling_receipt_sha256: str,
    ) -> CaseBindingPrerequisitesV2:
        """Close the last prerequisite field after a prospective cost freeze."""

        _require_sha(cost_ceiling_receipt_sha256, "cost ceiling receipt")
        endpoint = {
            "first": ActionEndpointV2.FIRST,
            "second": ActionEndpointV2.SECOND,
            "both": ActionEndpointV2.BOTH,
        }[self.exact_endpoint]
        return CaseBindingPrerequisitesV2(
            case_id=self.case_id,
            exact_endpoint=endpoint,
            source_input_sha256s=self.source_input_sha256s,
            endpoint_sha256=self.endpoint_sha256,
            support_realization_sha256=self.support_realization_sha256,
            rollback_sha256=self.rollback_sha256,
            materialization_recipe_sha256=self.materialization_recipe_sha256,
            cost_ceiling_receipt_sha256=cost_ceiling_receipt_sha256,
        )


def build_local_case_materialization_v1(
    *,
    qualification_action_id: str,
    exact_control_id: str,
    operator_id: str,
    observed_first_rgb: np.ndarray,
    observed_second_rgb: np.ndarray,
    observed_native_flow: np.ndarray,
    output_first_rgb: np.ndarray,
    output_second_rgb: np.ndarray,
    endpoint_supports: Sequence[np.ndarray],
    modified_endpoints: Sequence[str],
    physical_pair_sha256: str,
    endpoint_policy_sha256: str,
) -> LocalCaseMaterializationReceiptV1:
    """Build a strict local receipt for a Wave-1 full-strength local output."""

    source_hashes = _verify_sources()
    policy = _POLICIES.get(qualification_action_id)
    if policy is None:
        raise ValueError("unsupported local Wave-1 sentinel")
    if exact_control_id != policy["control_id"] or operator_id != policy["operator_id"]:
        raise ValueError("local policy/control/operator identity drift")
    _require_sha(physical_pair_sha256, "physical pair")
    _require_sha(endpoint_policy_sha256, "endpoint policy")
    pair, flow, outputs, supports = _validate_arrays(
        observed_first_rgb, observed_second_rgb, observed_native_flow,
        output_first_rgb, output_second_rgb, endpoint_supports,
    )
    modified = tuple(str(value) for value in modified_endpoints)
    if not modified or len(modified) != len(set(modified)):
        raise ValueError("modified endpoints must be nonempty and unique")
    if any(value not in {"first", "second"} for value in modified):
        raise ValueError("unknown modified endpoint")
    exact_endpoint = modified[0] if len(modified) == 1 else "both"

    endpoint_records = []
    names = ("first", "second")
    total_changed = 0
    for name, observed, output, support in zip(names, pair, outputs, supports):
        changed = np.any(output != observed, axis=2)
        changed_count = int(changed.sum())
        total_changed += changed_count
        outside_exact = bool(np.array_equal(output[~support], observed[~support]))
        if not outside_exact or np.any(changed & ~support):
            raise RuntimeError("local output escaped declared support")
        is_modified = name in modified
        if is_modified and (not bool(support.any()) or changed_count == 0):
            raise RuntimeError("modified endpoint has no acting local support")
        if not is_modified and (bool(support.any()) or changed_count != 0):
            raise RuntimeError("unmodified endpoint changed or declared support")
        endpoint_records.append({
            "endpoint": name,
            "coordinate_frame": f"{name}_native",
            "modified": is_modified,
            "source_sha256": _array_sha256(observed),
            "output_sha256": _array_sha256(output),
            "support_sha256": _array_sha256(support),
            "support_pixels": int(support.sum()),
            "changed_mask_sha256": _array_sha256(changed),
            "changed_pixels": changed_count,
            "changed_subset_of_support": True,
            "outside_support_byte_identity": outside_exact,
        })
    if total_changed == 0:
        raise RuntimeError("local action is identity-only")

    role_hashes = (
        _role_sha256("observed_first_rgb", pair[0]),
        _role_sha256("observed_second_rgb", pair[1]),
        _role_sha256("observed_native_flow", flow),
    )
    endpoint_sha = _canonical_sha256({
        "policy_sha256": endpoint_policy_sha256,
        "physical_pair_sha256": physical_pair_sha256,
        "exact_endpoint": exact_endpoint,
        "modified_endpoints": list(modified),
    })
    support_sha = _canonical_sha256({
        "physical_pair_sha256": physical_pair_sha256,
        "support_transport_policy": policy["support_transport_policy"],
        "endpoint_supports": [
            {"endpoint": row["endpoint"], "support_sha256": row["support_sha256"]}
            for row in endpoint_records
        ],
        "flow_sha256": _array_sha256(flow),
    })
    rollback_sha = _canonical_sha256({
        "rule": "RESTORE_ROLE_ORDERED_IMMUTABLE_OBSERVED_PAIR_BYTE_EXACT_V1",
        "first_sha256": _array_sha256(pair[0]),
        "second_sha256": _array_sha256(pair[1]),
    })
    recipe_sha = _canonical_sha256({
        "operator_id": operator_id,
        "composition": "blend_local_proposal",
        "feather_sigma_px": 1.0,
        "hard_remask": True,
        "image_read_dependency": policy["image_read_dependency"],
        "support_read_dependency": policy["support_read_dependency"],
        "source_hashes": source_hashes,
    })
    source_manifest_sha = _canonical_sha256(source_hashes)
    return LocalCaseMaterializationReceiptV1(
        qualification_action_id=qualification_action_id,
        exact_control_id=exact_control_id,
        operator_id=operator_id,
        case_id=f"physical-pair:{physical_pair_sha256}",
        physical_pair_sha256=physical_pair_sha256,
        exact_endpoint=exact_endpoint,
        endpoint_sha256=endpoint_sha,
        source_input_sha256s=role_hashes,
        support_realization_sha256=support_sha,
        rollback_sha256=rollback_sha,
        materialization_recipe_sha256=recipe_sha,
        endpoint_records=tuple(endpoint_records),
        image_read_dependency=policy["image_read_dependency"],
        support_read_dependency=policy["support_read_dependency"],
        support_transport_policy=policy["support_transport_policy"],
        source_manifest_sha256=source_manifest_sha,
    )


__all__ = [
    "LOCAL_CASE_MATERIALIZATION_SCHEMA_V1",
    "LocalCaseMaterializationReceiptV1",
    "build_local_case_materialization_v1",
]
