"""Fail-closed bridge from action descriptor v2 to ``CandidateArmV7``.

``ActionDescriptorV2`` and ``CaseBindingPrerequisitesV2`` intentionally do not
contain every field required by Selector-v7.  In particular, the selector also
needs an action index, an exact control, materialization identity, provenance,
and two fully typed cost accounts.  ``FrozenCaseArmBindingV2`` supplies exactly
those missing values and seals them before adaptation; this module never
invents defaults for a missing scientific or accounting receipt.

The operator parameter is not mapped to selector strength.  It remains sealed
inside ``ActionDescriptorV2.descriptor_hash`` while Selector-v7 receives only
the descriptor's normalized ``input_strength`` and ``output_beta``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re

from .action_bank_contracts import (
    NATIVE_ACTION_ID_V2,
    ActionCostStatusV2,
    ActionDescriptorV2,
    ControlValueStatusV2,
    CaseBindingPrerequisitesV2,
    validate_case_binding_prerequisites_v2,
)
from .selector_v7 import (
    CandidateArmV7,
    CandidateScopeV7,
    CostCountersV7,
)


CASE_ARM_BINDING_SCHEMA_V2 = "stablebridge-case-arm-binding/v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty canonical text")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 binding")
    return value


def _require_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _counter_payload(counter: CostCountersV7) -> dict[str, object]:
    if not isinstance(counter, CostCountersV7):
        raise ValueError("case-arm cost ceilings must use CostCountersV7")
    return {
        "natural_forwards": counter.natural_forwards,
        "reused_forwards": counter.reused_forwards,
        "candidate_forwards": counter.candidate_forwards,
        "reverse_forwards": counter.reverse_forwards,
        "cpu_seconds": counter.cpu_seconds,
        "gpu_seconds": counter.gpu_seconds,
        "wall_seconds": counter.wall_seconds,
        "bytes_moved": counter.bytes_moved,
        "peak_allocated_bytes": counter.peak_allocated_bytes,
        "peak_reserved_bytes": counter.peak_reserved_bytes,
        "matcher_trajectories": counter.matcher_trajectories,
        "cache_state": counter.cache_state,
    }


def case_binding_prerequisites_hash_v2(
    prerequisites: CaseBindingPrerequisitesV2,
) -> str:
    """Hash every prerequisite without erasing source-endpoint order."""

    if not isinstance(prerequisites, CaseBindingPrerequisitesV2):
        raise ValueError("case binding prerequisites must be typed")
    return _sha256({
        "case_id": prerequisites.case_id,
        "exact_endpoint": prerequisites.exact_endpoint.value,
        "source_input_sha256s": list(prerequisites.source_input_sha256s),
        "endpoint_sha256": prerequisites.endpoint_sha256,
        "support_realization_sha256": (
            prerequisites.support_realization_sha256
        ),
        "rollback_sha256": prerequisites.rollback_sha256,
        "materialization_recipe_sha256": (
            prerequisites.materialization_recipe_sha256
        ),
        "cost_ceiling_receipt_sha256": (
            prerequisites.cost_ceiling_receipt_sha256
        ),
    })


@dataclass(frozen=True)
class FrozenCaseArmBindingV2:
    """Additional frozen values needed for an exact CASE_ATOMIC arm.

    ``descriptor_sha256`` and ``prerequisites_sha256`` prevent a binding from
    being replayed against another template or case.  ``production_authority``
    is explicit and permanently false; building a selector candidate is not a
    production authorization event.
    """

    descriptor_sha256: str
    prerequisites_sha256: str
    candidate_id: str
    action_index: int
    exact_control_id: str
    control_sha256: str
    materialization_recipe_id: str
    materialization_recipe_version: str
    source_provenance_sha256: str
    offline_bank_acquisition_cost_ceiling: float
    offline_bank_acquisition_cost_counter_ceiling: CostCountersV7
    prospective_runtime_cost_counter_ceiling: CostCountersV7
    production_authority: bool = False
    binding_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        _require_sha256(self.descriptor_sha256, "bound descriptor hash")
        _require_sha256(
            self.prerequisites_sha256, "bound prerequisites hash"
        )
        _require_text(self.candidate_id, "candidate id")
        if (
            isinstance(self.action_index, bool)
            or not isinstance(self.action_index, int)
            or self.action_index < 0
        ):
            raise ValueError(
                "case-bound action index must be a nonnegative integer"
            )
        _require_text(self.exact_control_id, "exact control id")
        _require_sha256(self.control_sha256, "exact control hash")
        _require_text(
            self.materialization_recipe_id, "materialization recipe id"
        )
        _require_text(
            self.materialization_recipe_version,
            "materialization recipe version",
        )
        _require_sha256(
            self.source_provenance_sha256, "source provenance hash"
        )
        offline = _require_nonnegative(
            self.offline_bank_acquisition_cost_ceiling,
            "offline-bank acquisition cost ceiling",
        )
        _counter_payload(self.offline_bank_acquisition_cost_counter_ceiling)
        _counter_payload(self.prospective_runtime_cost_counter_ceiling)
        if not isinstance(self.production_authority, bool):
            raise ValueError("binding production authority must be boolean")
        if self.production_authority:
            raise ValueError(
                "case-arm bindings cannot grant production authority"
            )
        object.__setattr__(
            self,
            "offline_bank_acquisition_cost_ceiling",
            offline,
        )
        expected = _sha256(self.as_dict(include_hash=False))
        if self.binding_hash:
            _require_sha256(self.binding_hash, "case-arm binding hash")
            if self.binding_hash != expected:
                raise ValueError("case-arm binding hash drift")
        object.__setattr__(self, "binding_hash", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": CASE_ARM_BINDING_SCHEMA_V2,
            "descriptor_sha256": self.descriptor_sha256,
            "prerequisites_sha256": self.prerequisites_sha256,
            "candidate_id": self.candidate_id,
            "action_index": self.action_index,
            "exact_control_id": self.exact_control_id,
            "control_sha256": self.control_sha256,
            "materialization_recipe_id": self.materialization_recipe_id,
            "materialization_recipe_version": (
                self.materialization_recipe_version
            ),
            "source_provenance_sha256": self.source_provenance_sha256,
            "offline_bank_acquisition_cost_ceiling": (
                self.offline_bank_acquisition_cost_ceiling
            ),
            "offline_bank_acquisition_cost_counter_ceiling": _counter_payload(
                self.offline_bank_acquisition_cost_counter_ceiling
            ),
            "prospective_runtime_cost_counter_ceiling": _counter_payload(
                self.prospective_runtime_cost_counter_ceiling
            ),
            "production_authority": self.production_authority,
        }
        if include_hash:
            result["binding_hash"] = self.binding_hash
        return result


def _validate_runtime_cost_binding(
    descriptor: ActionDescriptorV2,
    counters: CostCountersV7,
) -> None:
    cost = descriptor.cost
    if cost.status is not ActionCostStatusV2.BOUND:
        raise ValueError("candidate adaptation needs bound cost ceilings")
    # The None cases are excluded by ActionCostV2, but retaining this guard
    # makes this boundary independently fail closed if that contract changes.
    if (
        cost.runtime_wall_seconds_ceiling is None
        or cost.matcher_trajectories_ceiling is None
        or cost.peak_vram_bytes_ceiling is None
    ):
        raise ValueError("candidate adaptation has an incomplete cost ceiling")
    if counters.wall_seconds != cost.runtime_wall_seconds_ceiling:
        raise ValueError(
            "runtime counter wall-seconds do not match descriptor ceiling"
        )
    if counters.matcher_trajectories != cost.matcher_trajectories_ceiling:
        raise ValueError(
            "runtime matcher trajectories do not match descriptor ceiling"
        )
    if counters.peak_reserved_bytes != cost.peak_vram_bytes_ceiling:
        raise ValueError(
            "runtime reserved VRAM does not match descriptor ceiling"
        )


def bind_action_descriptor_v2_to_candidate_arm_v7(
    descriptor: ActionDescriptorV2,
    prerequisites: CaseBindingPrerequisitesV2,
    binding: FrozenCaseArmBindingV2,
) -> CandidateArmV7:
    """Create one exact Selector-v7 arm from fully frozen, receipt-bound input."""

    if not isinstance(descriptor, ActionDescriptorV2):
        raise ValueError("candidate adaptation needs a typed v2 descriptor")
    if not isinstance(prerequisites, CaseBindingPrerequisitesV2):
        raise ValueError("candidate adaptation needs typed case prerequisites")
    if not isinstance(binding, FrozenCaseArmBindingV2):
        raise ValueError("candidate adaptation needs a frozen case-arm binding")
    validate_case_binding_prerequisites_v2(descriptor, prerequisites)
    if binding.descriptor_sha256 != descriptor.descriptor_hash:
        raise ValueError("case-arm binding descriptor hash mismatch")
    expected_prerequisites = case_binding_prerequisites_hash_v2(prerequisites)
    if binding.prerequisites_sha256 != expected_prerequisites:
        raise ValueError("case-arm binding prerequisites hash mismatch")
    if (
        descriptor.availability.production_authority
        or descriptor.receipt.production_authority
        or binding.production_authority
    ):
        raise ValueError("candidate adaptation cannot grant production authority")
    _validate_runtime_cost_binding(
        descriptor,
        binding.prospective_runtime_cost_counter_ceiling,
    )
    if (
        descriptor.input_strength_status is not ControlValueStatusV2.BOUND
        or descriptor.output_beta_status is not ControlValueStatusV2.BOUND
        or descriptor.input_strength is None
        or descriptor.output_beta is None
    ):
        raise ValueError("candidate adaptation needs bound normalized controls")

    native = descriptor.action_id == NATIVE_ACTION_ID_V2
    if native:
        if binding.action_index != 0 or binding.exact_control_id != "native":
            raise ValueError(
                "native adaptation needs action index zero and native control"
            )
    elif binding.action_index == 0:
        raise ValueError("only the native descriptor may use action index zero")

    runtime_cost = descriptor.cost.runtime_wall_seconds_ceiling
    if runtime_cost is None:  # pragma: no cover - guarded above
        raise ValueError("candidate adaptation has no runtime cost ceiling")
    return CandidateArmV7(
        candidate_id=binding.candidate_id,
        action_index=binding.action_index,
        action_identity=descriptor.action_id,
        mechanism_id=descriptor.operator.operator_id,
        operator_version=descriptor.operator.version,
        exact_control_id=binding.exact_control_id,
        endpoint_id=prerequisites.exact_endpoint.value,
        input_strength=descriptor.input_strength,
        output_beta=descriptor.output_beta,
        action_hash=descriptor.descriptor_hash,
        control_hash=binding.control_sha256,
        endpoint_hash=prerequisites.endpoint_sha256,
        support_hash=prerequisites.support_realization_sha256,
        support_policy_hash=descriptor.support.policy_sha256,
        rollback_hash=prerequisites.rollback_sha256,
        source_input_hashes=prerequisites.source_input_sha256s,
        materialization_recipe_id=binding.materialization_recipe_id,
        materialization_recipe_version=(
            binding.materialization_recipe_version
        ),
        materialization_recipe_hash=(
            prerequisites.materialization_recipe_sha256
        ),
        # The binding hash seals source provenance plus the descriptor,
        # prerequisite, cost, and non-authority receipts into the arm.
        provenance_hash=binding.binding_hash,
        scope=(
            CandidateScopeV7.NATIVE if native else CandidateScopeV7.CASE_ATOMIC
        ),
        unit_ids=(prerequisites.case_id,),
        interaction_receipts=(),
        offline_bank_acquisition_cost_ceiling=(
            binding.offline_bank_acquisition_cost_ceiling
        ),
        offline_bank_acquisition_cost_counter_ceiling=(
            binding.offline_bank_acquisition_cost_counter_ceiling
        ),
        prospective_runtime_cost_ceiling=runtime_cost,
        prospective_runtime_cost_counter_ceiling=(
            binding.prospective_runtime_cost_counter_ceiling
        ),
        hard_legal=True,
        hard_reject_reasons=(),
    )


__all__ = [
    "CASE_ARM_BINDING_SCHEMA_V2",
    "FrozenCaseArmBindingV2",
    "bind_action_descriptor_v2_to_candidate_arm_v7",
    "case_binding_prerequisites_hash_v2",
]
