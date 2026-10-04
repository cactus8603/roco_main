"""Bridge frozen bank-v2 rows and exact local executions to child receipts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .child_observables import ActionBindingV1, canonical_sha256


_STATIC_MECHANISMS = {
    "paired_impulse_median3.endpoint_supported_v1": "paired_impulse_median3",
    "paired_additive_wiener3.local_tile_sigma_v2": "paired_additive_wiener3",
    "common_motion.local_recoverable_v3": "common_motion",
    "jpeg_qcell_v3.local_macroblock_v3": "jpeg_qcell_v3",
    "jpeg_codec_path_v4.local_macroblock_v3": "jpeg_codec_path_v4",
}


def _bank_identity(bank: Mapping[str, Any]) -> tuple[str, str]:
    if bank.get("schema") != "stablebridge-foundation-action-bank/v2":
        raise ValueError("qualification binding needs foundation action bank v2")
    bank_id = bank.get("bank_id")
    bank_sha = bank.get("bank_sha256")
    if not isinstance(bank_id, str) or not bank_id:
        raise ValueError("action bank has no canonical bank_id")
    payload = {key: value for key, value in bank.items() if key != "bank_sha256"}
    if bank_sha != canonical_sha256(payload):
        raise ValueError("action bank semantic SHA-256 drift")
    return bank_id, bank_sha


def _descriptor(bank: Mapping[str, Any], action_id: str) -> Mapping[str, Any]:
    matches = [row for row in bank.get("actions", ()) if row.get("action_id") == action_id]
    if len(matches) != 1:
        raise ValueError("action must have exactly one frozen bank descriptor")
    row = matches[0]
    if not isinstance(row.get("descriptor_hash"), str):
        raise ValueError("bank descriptor has no hash")
    return row


def _mechanism_id(row: Mapping[str, Any]) -> str:
    action_id = str(row["qualification_action_id"])
    if action_id == "common_isotropic.local_recoverable_v3":
        branch = row.get("blur_branch")
        family = branch.get("internal_family") if isinstance(branch, Mapping) else None
        if family not in {"disk", "gaussian"}:
            raise ValueError("isotropic binding needs exact disk/Gaussian internal mode")
        return f"common_{family}"
    mechanism = _STATIC_MECHANISMS.get(action_id)
    if mechanism is None:
        raise ValueError("qualification action has no frozen mechanism mapping")
    return mechanism


def _local_value(local_receipt: object, name: str) -> Any:
    if not hasattr(local_receipt, name):
        raise ValueError(f"local execution receipt is missing {name}")
    return getattr(local_receipt, name)


def _support_realization_sha256(local_receipt: object) -> str:
    if hasattr(local_receipt, "write_support_realization_sha256"):
        payload = {
            "write": _local_value(local_receipt, "write_support_realization_sha256"),
            "read": _local_value(local_receipt, "read_support_realization_sha256"),
        }
        if hasattr(local_receipt, "flow_transport_sha256"):
            payload["transport"] = _local_value(
                local_receipt, "flow_transport_sha256",
            )
        return canonical_sha256(payload)
    return str(_local_value(local_receipt, "support_realization_sha256"))


@dataclass(frozen=True)
class QualificationActionBindingBundleV1:
    action: ActionBindingV1
    action_execution_receipt_sha256: str
    local_materialization_recipe_sha256: str
    local_support_realization_sha256: str


def build_qualification_action_binding_v1(
    *,
    bank: Mapping[str, Any],
    qualification_row: Mapping[str, Any],
    local_receipt: object,
) -> QualificationActionBindingBundleV1:
    """Bind one executed nonnative matrix row to its exact local lineage."""

    bank_id, bank_sha = _bank_identity(bank)
    if qualification_row.get("qualification_status") != "ELIGIBLE_EXECUTED":
        raise ValueError("only an executed qualification row can bind a child")
    action_id = str(qualification_row.get("qualification_action_id"))
    if action_id == "native":
        raise ValueError("native binding uses build_native_action_binding_v1")
    descriptor = _descriptor(bank, action_id)
    if qualification_row.get("action_descriptor_sha256") != descriptor["descriptor_hash"]:
        raise ValueError("qualification row descriptor binding drift")
    if _local_value(local_receipt, "qualification_action_id") != action_id:
        raise ValueError("local receipt action identity drift")
    local_hash = str(_local_value(local_receipt, "receipt_sha256"))
    if qualification_row.get("local_case_materialization_receipt_sha256") != local_hash:
        raise ValueError("qualification row local receipt hash drift")
    operator_id = str(_local_value(local_receipt, "operator_id"))
    if operator_id != descriptor["operator"]["id"]:
        raise ValueError("local receipt operator drift from bank descriptor")
    endpoint = str(_local_value(local_receipt, "exact_endpoint"))
    if qualification_row.get("exact_endpoint") != endpoint:
        raise ValueError("qualification row endpoint drift")
    row_parameter = str(qualification_row.get("parameter_sha256"))
    if hasattr(local_receipt, "exact_parameter_sha256"):
        if row_parameter != _local_value(local_receipt, "exact_parameter_sha256"):
            raise ValueError("qualification row exact parameter drift")
    recipe_sha = str(_local_value(local_receipt, "materialization_recipe_sha256"))
    support_sha = _support_realization_sha256(local_receipt)
    action = ActionBindingV1(
        bank_id=bank_id,
        bank_sha256=bank_sha,
        action_id=action_id,
        mechanism_id=_mechanism_id(qualification_row),
        operator_id=operator_id,
        control_id=str(qualification_row["exact_control_id"]),
        endpoint=endpoint,
        strength=float(qualification_row["input_strength"]),
        exact_parameter_sha256=row_parameter,
        support_policy_id=str(descriptor["support"]["policy_id"]),
        support_sha256=support_sha,
        composition_id=f"local-materialization-recipe:{recipe_sha}",
        action_descriptor_sha256=str(descriptor["descriptor_hash"]),
    )
    return QualificationActionBindingBundleV1(
        action=action,
        action_execution_receipt_sha256=local_hash,
        local_materialization_recipe_sha256=recipe_sha,
        local_support_realization_sha256=support_sha,
    )


def build_native_action_binding_v1(
    *,
    bank: Mapping[str, Any],
    case_id: str,
    physical_pair_sha256: str,
) -> tuple[ActionBindingV1, Mapping[str, Any]]:
    """Create the exact zero-action control binding for a qualification case."""

    bank_id, bank_sha = _bank_identity(bank)
    descriptor = _descriptor(bank, "native")
    parameter_sha = canonical_sha256({
        "operator": "identity",
        "physical_pair_sha256": physical_pair_sha256,
        "input_strength": 0.0,
    })
    support_sha = canonical_sha256({
        "policy_id": descriptor["support"]["policy_id"],
        "physical_pair_sha256": physical_pair_sha256,
        "support_pixels": 0,
    })
    execution_payload = {
        "schema": "stablebridge-native-action-execution/v1",
        "case_id": case_id,
        "physical_pair_sha256": physical_pair_sha256,
        "action_id": "native",
        "operator_id": "identity",
        "endpoint": "none",
        "input_strength": 0.0,
        "exact_parameter_sha256": parameter_sha,
        "support_realization_sha256": support_sha,
        "outside_support_byte_identity": True,
        "GT_read": False,
        "H2_read": False,
        "outcome_read": False,
    }
    execution_hash = canonical_sha256(execution_payload)
    execution_record = {
        **execution_payload,
        "receipt_sha256": execution_hash,
    }
    return ActionBindingV1(
        bank_id=bank_id,
        bank_sha256=bank_sha,
        action_id="native",
        mechanism_id="native_identity",
        operator_id="identity",
        control_id="native",
        endpoint="none",
        strength=0.0,
        exact_parameter_sha256=parameter_sha,
        support_policy_id=str(descriptor["support"]["policy_id"]),
        support_sha256=support_sha,
        composition_id="identity-byte-exact-v1",
        action_descriptor_sha256=str(descriptor["descriptor_hash"]),
    ), execution_record


__all__ = [
    "QualificationActionBindingBundleV1",
    "build_native_action_binding_v1",
    "build_qualification_action_binding_v1",
]
