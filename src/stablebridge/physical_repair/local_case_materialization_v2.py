"""Typed case-bound receipt for the versioned two-arm local blur successor.

The receipt binds the selected public arm and internal PSF parameter to exact
source images, write support, read halo, flow transport, outputs, and rollback.
It is the parent-lineage record required before region-projected CC/RR/CR/RC
child observables can be produced.

Cost, child observables, scientific qualification, selector admission, and
production authority remain explicitly absent.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .action_bank_contracts import ActionEndpointV2, CaseBindingPrerequisitesV2
from .local_blur_successor import _array_sha256, _canonical_sha256, _validate
from .local_blur_successor_v3 import (
    LOCAL_BLUR_ACTION_IDS_V3,
    LOCAL_BLUR_OPERATOR_IDS_V3,
    LOCAL_BLUR_SUPPORT_POLICY_ID_V3,
    LocalBlurSuccessorV3,
)


LOCAL_CASE_MATERIALIZATION_SCHEMA_V2 = "stablebridge-local-case-materialization/v2"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_BINDINGS = {
    "action_contract": (
        _ROOT / "src/stablebridge/physical_repair/action_bank_contracts.py",
        "055d221ddb94cfdc947006f38c4c17d9428e66c98ad1e2ea24219bc04e8f9cf8",
    ),
    "local_blur_v3": (
        _ROOT / "src/stablebridge/physical_repair/local_blur_successor_v3.py",
        "092f519659e247171271208c5b5e9bd3fc2c2abb40bc3eabc5419299175f9021",
    ),
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
            raise RuntimeError(f"local materialization v2 source escaped: {name}")
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"local materialization v2 source drift: {name}; "
                f"expected {expected}, got {actual}"
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
class LocalCaseMaterializationReceiptV2:
    qualification_action_id: str
    exact_control_id: str
    operator_id: str
    public_arm: str
    internal_family: str
    case_id: str
    physical_pair_sha256: str
    identified_endpoint: str
    exact_endpoint: str
    endpoint_sha256: str
    exact_parameter_sha256: str
    source_input_sha256s: tuple[str, str, str]
    write_support_realization_sha256: str
    read_support_realization_sha256: str
    flow_transport_sha256: str
    rollback_sha256: str
    materialization_recipe_sha256: str
    endpoint_records: tuple[Mapping[str, Any], Mapping[str, Any]]
    source_manifest_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        expected_action = LOCAL_BLUR_ACTION_IDS_V3.get(self.public_arm)
        expected_operator = LOCAL_BLUR_OPERATOR_IDS_V3.get(self.public_arm)
        if self.qualification_action_id != expected_action:
            raise ValueError("local materialization v2 action identity drift")
        if self.exact_control_id != expected_action:
            raise ValueError("local materialization v2 control identity drift")
        if self.operator_id != expected_operator:
            raise ValueError("local materialization v2 operator identity drift")
        if self.public_arm == "common_isotropic":
            if self.internal_family not in {"disk", "gaussian"}:
                raise ValueError("isotropic case needs disk/Gaussian internal mode")
        elif self.public_arm == "common_motion":
            if self.internal_family != "motion":
                raise ValueError("motion case needs motion internal mode")
        else:
            raise ValueError("unknown local blur v3 public arm")
        if self.identified_endpoint not in {"first", "second"}:
            raise ValueError("invalid identified endpoint")
        expected_endpoint = (
            "second" if self.identified_endpoint == "first" else "first"
        )
        if self.exact_endpoint != expected_endpoint:
            raise ValueError("local materialization v2 endpoint direction drift")
        if not self.case_id or self.case_id != self.case_id.strip():
            raise ValueError("local materialization v2 needs canonical case id")
        for name in (
            "physical_pair_sha256",
            "endpoint_sha256",
            "exact_parameter_sha256",
            "write_support_realization_sha256",
            "read_support_realization_sha256",
            "flow_transport_sha256",
            "rollback_sha256",
            "materialization_recipe_sha256",
            "source_manifest_sha256",
        ):
            _require_sha(getattr(self, name), name)
        if len(self.source_input_sha256s) != 3:
            raise ValueError("local materialization v2 needs three source hashes")
        for index, value in enumerate(self.source_input_sha256s):
            _require_sha(value, f"source input {index}")
        if len(set(self.source_input_sha256s)) != 3:
            raise ValueError("role-bound source hashes must be unique")
        if len(self.endpoint_records) != 2:
            raise ValueError("local materialization v2 needs two endpoint records")
        expected_hash = _canonical_sha256(self.as_dict(include_hash=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected_hash:
            raise ValueError("local materialization v2 receipt hash drift")
        object.__setattr__(self, "receipt_sha256", expected_hash)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema": LOCAL_CASE_MATERIALIZATION_SCHEMA_V2,
            "qualification_action_id": self.qualification_action_id,
            "exact_control_id": self.exact_control_id,
            "operator_id": self.operator_id,
            "public_arm": self.public_arm,
            "internal_family": self.internal_family,
            "case_id": self.case_id,
            "physical_pair_sha256": self.physical_pair_sha256,
            "identified_endpoint": self.identified_endpoint,
            "exact_endpoint": self.exact_endpoint,
            "endpoint_sha256": self.endpoint_sha256,
            "exact_parameter_sha256": self.exact_parameter_sha256,
            "source_input_sha256s": list(self.source_input_sha256s),
            "write_support_realization_sha256": (
                self.write_support_realization_sha256
            ),
            "read_support_realization_sha256": (
                self.read_support_realization_sha256
            ),
            "flow_transport_sha256": self.flow_transport_sha256,
            "rollback_sha256": self.rollback_sha256,
            "materialization_recipe_sha256": self.materialization_recipe_sha256,
            "endpoint_records": [dict(value) for value in self.endpoint_records],
            "support_policy_id": LOCAL_BLUR_SUPPORT_POLICY_ID_V3,
            "composition": "FULL_STRENGTH_LOCAL_OUTPUT_HARD_REMASK_FEATHER_SIGMA_1_V2",
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
            result["receipt_sha256"] = self.receipt_sha256
        return result

    def to_case_binding_prerequisites_v2(
        self, *, cost_ceiling_receipt_sha256: str,
    ) -> CaseBindingPrerequisitesV2:
        _require_sha(cost_ceiling_receipt_sha256, "cost ceiling receipt")
        endpoint = {
            "first": ActionEndpointV2.FIRST,
            "second": ActionEndpointV2.SECOND,
        }[self.exact_endpoint]
        combined_support = _canonical_sha256({
            "write": self.write_support_realization_sha256,
            "read": self.read_support_realization_sha256,
            "transport": self.flow_transport_sha256,
        })
        return CaseBindingPrerequisitesV2(
            case_id=self.case_id,
            exact_endpoint=endpoint,
            source_input_sha256s=self.source_input_sha256s,
            endpoint_sha256=self.endpoint_sha256,
            support_realization_sha256=combined_support,
            rollback_sha256=self.rollback_sha256,
            materialization_recipe_sha256=self.materialization_recipe_sha256,
            cost_ceiling_receipt_sha256=cost_ceiling_receipt_sha256,
        )


def build_local_case_materialization_v2(
    *,
    case_id: str,
    physical_pair_sha256: str,
    endpoint_policy_sha256: str,
    observed_first_rgb: np.ndarray,
    observed_second_rgb: np.ndarray,
    observed_native_flow: np.ndarray,
    successor: LocalBlurSuccessorV3,
) -> LocalCaseMaterializationReceiptV2:
    """Bind one executed v3 successor to exact local parent/child lineage."""
    source_hashes = _verify_sources()
    _require_sha(physical_pair_sha256, "physical pair")
    _require_sha(endpoint_policy_sha256, "endpoint policy")
    first, second, flow = _validate(
        observed_first_rgb, observed_second_rgb, observed_native_flow,
    )
    if successor.status != "EXECUTED_LOCAL_RECOVERABLE_V3":
        raise ValueError("local materialization v2 needs an executed successor")
    if any(value is None for value in (
        successor.action_id,
        successor.operator_id,
        successor.public_arm,
        successor.internal_family,
        successor.identified_endpoint,
        successor.exact_endpoint,
        successor.selected_parameter,
    )):
        raise ValueError("executed successor v3 has incomplete identity")
    receipt = successor.receipt
    observed_hashes = {
        "first": _array_sha256(first),
        "second": _array_sha256(second),
        "native_flow": _array_sha256(flow),
    }
    if receipt["input_sha256s"] != observed_hashes:
        raise RuntimeError("local materialization v2 source input drift")

    outputs = (successor.first_rgb, successor.second_rgb)
    writes = (successor.first_support, successor.second_support)
    reads = (successor.first_read_support, successor.second_read_support)
    names = ("first", "second")
    endpoint_records = []
    for name, observed, output, write, read in zip(
        names, (first, second), outputs, writes, reads,
    ):
        if (
            output.dtype != np.uint8 or output.shape != observed.shape
            or write.dtype != np.bool_ or write.shape != observed.shape[:2]
            or read.dtype != np.bool_ or read.shape != observed.shape[:2]
        ):
            raise ValueError("local materialization v2 child geometry drift")
        changed = np.any(output != observed, axis=2)
        if np.any(changed & ~write) or not np.array_equal(
            output[~write], observed[~write]
        ):
            raise RuntimeError("local materialization v2 escaped write support")
        if np.any(write & ~read):
            raise RuntimeError("local materialization v2 write escaped read halo")
        endpoint_records.append({
            "endpoint": name,
            "modified": name == successor.exact_endpoint,
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
    modified_rows = [row for row in endpoint_records if row["modified"]]
    if len(modified_rows) != 1 or modified_rows[0]["changed_pixels"] <= 0:
        raise RuntimeError("local materialization v2 needs one changed endpoint")
    unmodified = [row for row in endpoint_records if not row["modified"]][0]
    if (
        unmodified["changed_pixels"] != 0
        or unmodified["write_support_pixels"] != 0
        or unmodified["read_support_pixels"] != 0
    ):
        raise RuntimeError("local materialization v2 changed inactive endpoint")

    public_arm = str(successor.public_arm)
    internal_family = str(successor.internal_family)
    identified = str(successor.identified_endpoint)
    exact_endpoint = str(successor.exact_endpoint)
    parameter = tuple(map(float, successor.selected_parameter or ()))
    exact_parameter_sha256 = _canonical_sha256({
        "public_arm": public_arm,
        "internal_family": internal_family,
        "selected_parameter": list(parameter),
        "identified_endpoint": identified,
        "modified_endpoint": exact_endpoint,
        "source_action_key": receipt["source_action_key"],
        "fit_support_sha256": receipt["fit_support_sha256"],
        "check_support_sha256": receipt["check_support_sha256"],
    })
    write_support_sha256 = _canonical_sha256({
        row["endpoint"]: row["write_support_sha256"]
        for row in endpoint_records
    })
    read_support_sha256 = _canonical_sha256({
        row["endpoint"]: row["read_support_sha256"]
        for row in endpoint_records
    })
    flow_transport_sha256 = _canonical_sha256({
        "native_flow_sha256": observed_hashes["native_flow"],
        "flow_support_sha256": receipt["flow_support_sha256"],
        "policy": receipt["support_transport_policy"],
        "endpoint_write_supports": receipt["endpoint_support_sha256s"],
    })
    rollback_sha256 = _canonical_sha256({
        "first": observed_hashes["first"],
        "second": observed_hashes["second"],
        "policy": "RESTORE_EXACT_PARENT_PAIR_V1",
    })
    recipe_sha256 = _canonical_sha256({
        "action_id": successor.action_id,
        "operator_id": successor.operator_id,
        "public_arm": public_arm,
        "internal_family": internal_family,
        "exact_parameter_sha256": exact_parameter_sha256,
        "write_support_realization_sha256": write_support_sha256,
        "read_support_realization_sha256": read_support_sha256,
        "flow_transport_sha256": flow_transport_sha256,
        "composition": "PSF_PROPOSAL_FEATHER_SIGMA_1_HARD_REMASK_V3",
        "successor_receipt_sha256": receipt["receipt_sha256"],
    })
    endpoint_sha256 = _canonical_sha256({
        "policy_sha256": endpoint_policy_sha256,
        "identified_endpoint": identified,
        "modified_endpoint": exact_endpoint,
    })
    source_input_sha256s = (
        _role_sha256("OBSERVED_FIRST_RGB", first),
        _role_sha256("OBSERVED_SECOND_RGB", second),
        _role_sha256("OBSERVED_NATIVE_FLOW", flow),
    )
    return LocalCaseMaterializationReceiptV2(
        qualification_action_id=str(successor.action_id),
        exact_control_id=str(successor.action_id),
        operator_id=str(successor.operator_id),
        public_arm=public_arm,
        internal_family=internal_family,
        case_id=case_id,
        physical_pair_sha256=physical_pair_sha256,
        identified_endpoint=identified,
        exact_endpoint=exact_endpoint,
        endpoint_sha256=endpoint_sha256,
        exact_parameter_sha256=exact_parameter_sha256,
        source_input_sha256s=source_input_sha256s,
        write_support_realization_sha256=write_support_sha256,
        read_support_realization_sha256=read_support_sha256,
        flow_transport_sha256=flow_transport_sha256,
        rollback_sha256=rollback_sha256,
        materialization_recipe_sha256=recipe_sha256,
        endpoint_records=tuple(endpoint_records),  # type: ignore[arg-type]
        source_manifest_sha256=_canonical_sha256(source_hashes),
    )


__all__ = [
    "LOCAL_CASE_MATERIALIZATION_SCHEMA_V2",
    "LocalCaseMaterializationReceiptV2",
    "build_local_case_materialization_v2",
]
