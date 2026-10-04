"""Case-bound lineage receipt for block-exact local JPEG successors v3."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .action_bank_contracts import ActionEndpointV2, CaseBindingPrerequisitesV2
from .local_blur_successor import _array_sha256, _canonical_sha256, _validate
from .local_jpeg_successors_v3 import (
    LOCAL_JPEG_CODEC_ACTION_ID_V3,
    LOCAL_JPEG_CODEC_OPERATOR_ID_V3,
    LOCAL_JPEG_QCELL_ACTION_ID_V3,
    LOCAL_JPEG_QCELL_OPERATOR_ID_V3,
    LocalJPEGSuccessorV3,
)


LOCAL_JPEG_CASE_SCHEMA_V3 = "stablebridge-local-jpeg-case-materialization/v3"
LOCAL_JPEG_SUPPORT_POLICY_ID_V3 = "jpeg_aligned_block_exact_support_v3"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_BINDINGS = {
    "action_contract": (
        _ROOT / "src/stablebridge/physical_repair/action_bank_contracts.py",
        "055d221ddb94cfdc947006f38c4c17d9428e66c98ad1e2ea24219bc04e8f9cf8",
    ),
    "local_jpeg_v3": (
        _ROOT / "src/stablebridge/physical_repair/local_jpeg_successors_v3.py",
        "051a930cf41ecbe2db64ab871d4f0f6529c73e4a7323723ed3c7da9790138b2d",
    ),
}
_ACTIONS = {
    LOCAL_JPEG_QCELL_ACTION_ID_V3: LOCAL_JPEG_QCELL_OPERATOR_ID_V3,
    LOCAL_JPEG_CODEC_ACTION_ID_V3: LOCAL_JPEG_CODEC_OPERATOR_ID_V3,
}


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
            raise RuntimeError(f"JPEG v3 case source escaped workspace: {name}")
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"JPEG v3 case source drift: {name}; expected {expected}, got {actual}"
            )
        result[name] = actual
    return result


def _role_sha256(role: str, value: np.ndarray) -> str:
    return _canonical_sha256({
        "role": role,
        "array_sha256": _array_sha256(value),
        "hash_contract": "dtype-shape-bytes/v1",
    })


@dataclass(frozen=True)
class LocalJPEGCaseMaterializationReceiptV3:
    qualification_action_id: str
    exact_control_id: str
    operator_id: str
    case_id: str
    physical_pair_sha256: str
    exact_endpoint: str
    endpoint_sha256: str
    exact_parameter_sha256: str
    source_input_sha256s: tuple[str, str, str]
    write_support_realization_sha256: str
    read_support_realization_sha256: str
    rollback_sha256: str
    materialization_recipe_sha256: str
    endpoint_records: tuple[Mapping[str, Any], Mapping[str, Any]]
    source_manifest_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        expected_operator = _ACTIONS.get(self.qualification_action_id)
        if expected_operator is None:
            raise ValueError("unknown JPEG v3 case action")
        if self.exact_control_id != self.qualification_action_id:
            raise ValueError("JPEG v3 case control identity drift")
        if self.operator_id != expected_operator:
            raise ValueError("JPEG v3 case operator identity drift")
        if self.exact_endpoint not in {"first", "second", "both"}:
            raise ValueError("invalid JPEG v3 exact endpoint")
        if not self.case_id or self.case_id != self.case_id.strip():
            raise ValueError("JPEG v3 case needs canonical case id")
        for name in (
            "physical_pair_sha256", "endpoint_sha256", "exact_parameter_sha256",
            "write_support_realization_sha256", "read_support_realization_sha256",
            "rollback_sha256", "materialization_recipe_sha256",
            "source_manifest_sha256",
        ):
            _require_sha(getattr(self, name), name)
        if len(self.source_input_sha256s) != 3:
            raise ValueError("JPEG v3 case needs three role-bound source hashes")
        for index, value in enumerate(self.source_input_sha256s):
            _require_sha(value, f"source input {index}")
        if len(set(self.source_input_sha256s)) != 3:
            raise ValueError("JPEG v3 role-bound source hashes must be unique")
        if len(self.endpoint_records) != 2:
            raise ValueError("JPEG v3 case needs two endpoint records")
        expected_hash = _canonical_sha256(self.as_dict(include_hash=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected_hash:
            raise ValueError("JPEG v3 case receipt hash drift")
        object.__setattr__(self, "receipt_sha256", expected_hash)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema": LOCAL_JPEG_CASE_SCHEMA_V3,
            "qualification_action_id": self.qualification_action_id,
            "exact_control_id": self.exact_control_id,
            "operator_id": self.operator_id,
            "case_id": self.case_id,
            "physical_pair_sha256": self.physical_pair_sha256,
            "exact_endpoint": self.exact_endpoint,
            "endpoint_sha256": self.endpoint_sha256,
            "exact_parameter_sha256": self.exact_parameter_sha256,
            "source_input_sha256s": list(self.source_input_sha256s),
            "write_support_realization_sha256": self.write_support_realization_sha256,
            "read_support_realization_sha256": self.read_support_realization_sha256,
            "rollback_sha256": self.rollback_sha256,
            "materialization_recipe_sha256": self.materialization_recipe_sha256,
            "endpoint_records": [dict(row) for row in self.endpoint_records],
            "support_policy_id": LOCAL_JPEG_SUPPORT_POLICY_ID_V3,
            "image_read_dependency": (
                "FULL_ENDPOINT_FROZEN_PROPOSAL;ALIGNED_8PX_BLOCK_EXACT_WRITE"
            ),
            "support_transport_policy": "ENDPOINT_NATIVE_NO_OUTPUT_TRANSPORT",
            "composition": "ALIGNED_8PX_BLOCK_EXACT_SELECTION_V1",
            "input_strength": 1.0,
            "output_beta": 1.0,
            "source_manifest_sha256": self.source_manifest_sha256,
            "outside_write_support_byte_identity": True,
            "case_binding_prerequisites_ready": False,
            "case_binding_missing": ["prospective_cost_ceiling_receipt"],
            "child_observables_ready": False,
            "child_observables_missing": [
                "region_projected_CC_RR_CR_RC_receipt"
            ],
            "scientific_qualification": False,
            "selector_admission": False,
            "production_authority": False,
        }
        if include_hash:
            value["receipt_sha256"] = self.receipt_sha256
        return value

    def to_case_binding_prerequisites_v2(
        self, *, cost_ceiling_receipt_sha256: str,
    ) -> CaseBindingPrerequisitesV2:
        _require_sha(cost_ceiling_receipt_sha256, "cost ceiling receipt")
        return CaseBindingPrerequisitesV2(
            case_id=self.case_id,
            exact_endpoint={
                "first": ActionEndpointV2.FIRST,
                "second": ActionEndpointV2.SECOND,
                "both": ActionEndpointV2.BOTH,
            }[self.exact_endpoint],
            source_input_sha256s=self.source_input_sha256s,
            endpoint_sha256=self.endpoint_sha256,
            support_realization_sha256=_canonical_sha256({
                "write": self.write_support_realization_sha256,
                "read": self.read_support_realization_sha256,
            }),
            rollback_sha256=self.rollback_sha256,
            materialization_recipe_sha256=self.materialization_recipe_sha256,
            cost_ceiling_receipt_sha256=cost_ceiling_receipt_sha256,
        )


def build_local_jpeg_case_materialization_v3(
    *,
    action_id: str,
    case_id: str,
    physical_pair_sha256: str,
    endpoint_policy_sha256: str,
    observed_first_rgb: np.ndarray,
    observed_second_rgb: np.ndarray,
    observed_native_flow: np.ndarray,
    successors: Mapping[str, LocalJPEGSuccessorV3],
) -> LocalJPEGCaseMaterializationReceiptV3:
    """Bind every executed endpoint of one JPEG policy to the parent pair."""
    source_hashes = _verify_sources()
    operator_id = _ACTIONS.get(action_id)
    if operator_id is None:
        raise ValueError("unknown JPEG v3 action")
    _require_sha(physical_pair_sha256, "physical pair")
    _require_sha(endpoint_policy_sha256, "endpoint policy")
    first, second, flow = _validate(
        observed_first_rgb, observed_second_rgb, observed_native_flow,
    )
    if not successors or set(successors) - {"first", "second"}:
        raise ValueError("JPEG v3 case needs one or two named endpoint successors")
    pair = {"first": first, "second": second}
    endpoint_records = []
    parameters = {}
    outputs = {}
    writes = {}
    reads = {}
    for endpoint in ("first", "second"):
        observed = pair[endpoint]
        successor = successors.get(endpoint)
        if successor is None:
            output = observed.copy()
            write = np.zeros(observed.shape[:2], dtype=bool)
            read = write.copy()
            modified = False
        else:
            if successor.status != "EXECUTED_LOCAL_BLOCK_EXACT_V3":
                raise ValueError("JPEG v3 case accepts only executed successors")
            if successor.action_id != action_id or successor.operator_id != operator_id:
                raise ValueError("JPEG v3 case successor identity drift")
            if successor.receipt["endpoint"] != endpoint:
                raise ValueError("JPEG v3 case successor endpoint drift")
            if successor.receipt["input_sha256"] != _array_sha256(observed):
                raise RuntimeError("JPEG v3 case source input drift")
            output = np.ascontiguousarray(successor.output_rgb)
            write = np.ascontiguousarray(successor.support)
            # Proposal generation includes a full endpoint JPEG/DCT path even
            # though composition writes only aligned selected blocks.
            read = np.ones(observed.shape[:2], dtype=bool)
            modified = True
            parameters[endpoint] = {
                "estimated_ijg_quality": successor.estimated_quality,
                "selected_strength": successor.selected_strength,
                "active_macro_tiles": [
                    list(value) for value in successor.active_macro_tiles
                ],
                "successor_receipt_sha256": successor.receipt["receipt_sha256"],
            }
        changed = np.any(output != observed, axis=2)
        if np.any(changed & ~write) or not np.array_equal(
            output[~write], observed[~write]
        ):
            raise RuntimeError("JPEG v3 case escaped write support")
        if modified and (not write.any() or not changed.any()):
            raise RuntimeError("JPEG v3 case modified endpoint has no action")
        if not modified and (write.any() or read.any() or changed.any()):
            raise RuntimeError("JPEG v3 case inactive endpoint is not exact")
        outputs[endpoint] = output
        writes[endpoint] = write
        reads[endpoint] = read
        endpoint_records.append({
            "endpoint": endpoint,
            "modified": modified,
            "source_sha256": _array_sha256(observed),
            "child_sha256": _array_sha256(output),
            "write_support_sha256": _array_sha256(write),
            "write_support_pixels": int(write.sum()),
            "read_support_sha256": _array_sha256(read),
            "read_support_pixels": int(read.sum()),
            "changed_mask_sha256": _array_sha256(changed),
            "changed_pixels": int(changed.sum()),
            "outside_write_support_byte_identity": True,
        })
    active = tuple(endpoint for endpoint in ("first", "second") if endpoint in successors)
    exact_endpoint = active[0] if len(active) == 1 else "both"
    write_hash = _canonical_sha256({
        endpoint: _array_sha256(writes[endpoint]) for endpoint in ("first", "second")
    })
    read_hash = _canonical_sha256({
        endpoint: _array_sha256(reads[endpoint]) for endpoint in ("first", "second")
    })
    parameter_hash = _canonical_sha256({
        "action_id": action_id,
        "operator_id": operator_id,
        "endpoints": parameters,
    })
    rollback_hash = _canonical_sha256({
        "first": _array_sha256(first),
        "second": _array_sha256(second),
        "policy": "RESTORE_EXACT_PARENT_PAIR_V1",
    })
    recipe_hash = _canonical_sha256({
        "action_id": action_id,
        "operator_id": operator_id,
        "exact_parameter_sha256": parameter_hash,
        "write_support_realization_sha256": write_hash,
        "read_support_realization_sha256": read_hash,
        "composition": "ALIGNED_8PX_BLOCK_EXACT_SELECTION_V1",
        "support_transport": "ENDPOINT_NATIVE_NO_OUTPUT_TRANSPORT",
    })
    endpoint_hash = _canonical_sha256({
        "policy_sha256": endpoint_policy_sha256,
        "modified_endpoints": list(active),
    })
    return LocalJPEGCaseMaterializationReceiptV3(
        qualification_action_id=action_id,
        exact_control_id=action_id,
        operator_id=operator_id,
        case_id=case_id,
        physical_pair_sha256=physical_pair_sha256,
        exact_endpoint=exact_endpoint,
        endpoint_sha256=endpoint_hash,
        exact_parameter_sha256=parameter_hash,
        source_input_sha256s=(
            _role_sha256("OBSERVED_FIRST_RGB", first),
            _role_sha256("OBSERVED_SECOND_RGB", second),
            _role_sha256("OBSERVED_NATIVE_FLOW", flow),
        ),
        write_support_realization_sha256=write_hash,
        read_support_realization_sha256=read_hash,
        rollback_sha256=rollback_hash,
        materialization_recipe_sha256=recipe_hash,
        endpoint_records=tuple(endpoint_records),  # type: ignore[arg-type]
        source_manifest_sha256=_canonical_sha256(source_hashes),
    )


__all__ = [
    "LOCAL_JPEG_CASE_SCHEMA_V3",
    "LOCAL_JPEG_SUPPORT_POLICY_ID_V3",
    "LocalJPEGCaseMaterializationReceiptV3",
    "build_local_jpeg_case_materialization_v3",
]
