"""Append-only SR1 SelectorV7-v3 contract successor.

The reviewed v1 module remains byte-for-byte available as ``selector_v7``.
This successor closes SR1 root-review findings R01--R05 without reading an
outcome, executing an action, fitting a model, or granting an authority seal.
All code here is deterministic, immutable, and suitable for synthetic replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Sequence

from . import selector_v7 as v1
from . import selector_v7_v2 as v2


SELECTOR_V7_V3_SCHEMA_VERSION = "selector-v7-contract/v4.39-sr1-v3-draft"
SELECTOR_V7_V3_CANONICALIZATION_VERSION = "selector-v7-v3-canonical-json/v1"
SELECTOR_V7_V2_REVIEWED_SHA256 = (
    "8fbc8e3751ccb2febfaf7b8a4291be4a65ced4cfe56238c55bcf56657a1526c6"
)
SELECTOR_V7_V2_REVIEWED_TEST_SHA256 = (
    "ec6ebf6ead9ff9770c12ced68b7c18126a9cd8ff546dc75df57a397ddbe5c30f"
)
SELECTOR_V7_V1_FAILED_REVIEW_SHA256 = (
    "3299aa0c36f1b765291b894608f6b65945dd2df81f604634ae8224e4f757aeb5"
)
SELECTOR_V7_V1_FAILED_TEST_SHA256 = (
    "e48df3e376d8a934f408ea7d83f2d777af9266412251f86c5ffb607d1216e144"
)
SR1_ROOT_REVIEW_V1_SHA256 = (
    "4e1d804b2fc4a300ace248be963bafa5165fabe31e2bfe230834b4b353bf46ed"
)
PLANNER_ALGORITHM_ID_V7V3 = "group-weighted-ridge"
PLANNER_ALGORITHM_VERSION_V7V3 = "v1"
PLANNER_TARGET_ID_V7V3 = "G_hat_equals_B_hat_minus_H_hat"
PLANNER_CONFIG_HASH_V7V3 = hashlib.sha256(
    b'{"algorithm":"group-weighted-ridge","l2":1.0,"standardization":"fit-mean-std"}'
).hexdigest()


def _sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sealed(provided: str, payload: object, name: str) -> str:
    expected = _sha256(payload)
    if provided and provided != expected:
        raise ValueError(f"{name} does not match canonical content")
    return expected


def _rehash_valid(value: object, _active: set[int] | None = None) -> bool:
    """Reconstruct every nested dataclass so stale inner hashes cannot hide drift."""
    active = set() if _active is None else _active
    if isinstance(value, (tuple, list, set, frozenset)):
        return all(_rehash_valid(row, active) for row in value)
    if isinstance(value, dict):
        return all(
            _rehash_valid(key, active) and _rehash_valid(row, active)
            for key, row in value.items()
        )
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is None:
        return True
    marker = id(value)
    if marker in active:
        return False
    active.add(marker)
    try:
        if any(
            not _rehash_valid(getattr(value, name), active)
            for name in fields
        ):
            return False
        type(value)(**{name: getattr(value, name) for name in fields})
    except Exception:
        return False
    finally:
        active.remove(marker)
    return True


def _sorted_unique(values: Sequence[str], name: str) -> tuple[str, ...]:
    rows = tuple(values)
    if any(not isinstance(row, str) or not row for row in rows):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(rows) != len(set(rows)):
        raise ValueError(f"{name} must be unique")
    return tuple(sorted(rows))


def _sum_counters(rows: Sequence[v1.CostCountersV7], *,
                  cache_state: str = "cold") -> v1.CostCountersV7:
    total = v1.CostCountersV7.zero(cache_state=cache_state)
    for row in rows:
        total = total.plus(row)
    return total


class DecisionReasonV7V3(str, Enum):
    SAFE_SET_MAXIMUM = "safe_set_maximum"
    NO_SAFE_CANDIDATE = "no_safe_candidate"
    NATIVE_IDENTITY_INVALID = "native_identity_invalid"
    SCORE_BANK_INCOMPLETE = "score_bank_incomplete"
    PROPOSAL_REPLAY_MISMATCH = "proposal_replay_mismatch"
    PLANNER_AUTHORITY_MISMATCH = "planner_authority_mismatch"
    ASSESSOR_CALIBRATION_MISMATCH = "assessor_calibration_mismatch"
    CALIBRATION_FAMILY_INCOMPLETE = "calibration_family_incomplete"
    ALIAS_CHRONOLOGY_INVALID = "alias_chronology_invalid"
    ACTION_BINDING_MISMATCH = "action_binding_mismatch"
    FEATURE_INELIGIBLE = "feature_ineligible"
    RISK_BUDGET_EXCEEDED = "risk_budget_exceeded"
    NONPOSITIVE_CONSERVATIVE_UTILITY = "nonpositive_conservative_utility"
    COST_BUDGET_EXCEEDED = "cost_budget_exceeded"
    GLOBAL_PARTIAL_COMMIT = "global_partial_commit"
    LOCAL_SUPPORT_MISMATCH = "local_support_mismatch"
    AUDIT_BUDGET_EXHAUSTED = "audit_budget_exhausted"
    CHILD_BUDGET_EXHAUSTED = "child_budget_exhausted"
    MALFORMED_INPUT = "malformed_input"


class AliasChronologyModeV7V3(str, Enum):
    DIRECT_NO_ALIAS = "direct_no_alias"
    PRE_RUN_FROZEN_REUSE = "pre_run_frozen_reuse"
    POST_HOC_DISCOVERY = "post_hoc_discovery"
    SAME_RUN_DUPLICATE = "same_run_duplicate"


class RuntimeOperationV7V3(str, Enum):
    EXECUTE_ARM = "execute_arm"
    VERIFY_ALIAS_REUSE = "verify_alias_reuse"


class ObservationSourceV7V3(str, Enum):
    FROZEN_FAMILY_OBSERVATION = "frozen_family_observation"
    PROSPECTIVE_RUNTIME_OBSERVATION = "prospective_runtime_observation"


@dataclass(frozen=True)
class SelectorV7V3Disposition:
    predecessor_schema: str = v2.SELECTOR_V7_V2_SCHEMA_VERSION
    predecessor_sha256: str = SELECTOR_V7_V2_REVIEWED_SHA256
    predecessor_review_status: str = "FAIL_SR1_V2_NEW_CONTRACT_BLOCKERS_NO_SEAL"
    successor_schema: str = SELECTOR_V7_V3_SCHEMA_VERSION
    authority: str = "DRAFT_SYNTHETIC_ONLY_NO_AUTHORITY_SEAL"


@dataclass(frozen=True)
class PlannerFitReceiptV7V3:
    fit_fold_role: v1.FoldRoleV7
    target_fold_role: v1.FoldRoleV7
    fit_fold_ids: tuple[str, ...]
    fit_data_hash: str
    fit_target_hash: str
    group_weighting_hash: str
    availability: v1.AvailabilityV7
    availability_reason: str
    frozen_before_target_fold: bool
    provenance_hash: str
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        if self.fit_fold_role is not v1.FoldRoleV7.FIT:
            raise ValueError("planner FIT receipt must have FIT source role")
        if not isinstance(self.target_fold_role, v1.FoldRoleV7):
            raise ValueError("planner FIT target role is untyped")
        folds = _sorted_unique(self.fit_fold_ids, "planner FIT fold ids")
        for name, value in (
            ("planner FIT data hash", self.fit_data_hash),
            ("planner FIT target hash", self.fit_target_hash),
            ("planner FIT group-weighting hash", self.group_weighting_hash),
            ("planner FIT availability reason", self.availability_reason),
            ("planner FIT provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        if not isinstance(self.availability, v1.AvailabilityV7):
            raise ValueError("planner FIT availability is untyped")
        if self.availability is not v1.AvailabilityV7.AVAILABLE:
            raise ValueError("primary planner FIT receipt must be available")
        if self.frozen_before_target_fold is not True:
            raise ValueError("planner FIT receipt must precede target-fold scoring")
        payload = {
            "fit_fold_role": self.fit_fold_role.value,
            "target_fold_role": self.target_fold_role.value,
            "fit_fold_ids": list(folds),
            "fit_data_hash": self.fit_data_hash,
            "fit_target_hash": self.fit_target_hash,
            "group_weighting_hash": self.group_weighting_hash,
            "availability": self.availability.value,
            "availability_reason": self.availability_reason,
            "frozen_before_target_fold": self.frozen_before_target_fold,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "fit_fold_ids", folds)
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "planner FIT receipt hash",
        ))


@dataclass(frozen=True)
class PlannerModelReceiptV7V3:
    fit_receipt: PlannerFitReceiptV7V3
    feature_registry: v1.FeatureAllowlistsV7
    model_hash: str
    model_provenance_hash: str
    feature_schema_hash: str
    algorithm_id: str
    algorithm_version: str
    config_hash: str
    target_id: str
    frozen_before_scoring: bool
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.fit_receipt, PlannerFitReceiptV7V3):
            raise ValueError("planner model needs a typed FIT receipt")
        if not isinstance(self.feature_registry, v1.FeatureAllowlistsV7):
            raise ValueError("planner model needs a typed feature registry")
        for name, value in (
            ("planner model hash", self.model_hash),
            ("planner model provenance hash", self.model_provenance_hash),
            ("planner model feature-schema hash", self.feature_schema_hash),
            ("planner algorithm id", self.algorithm_id),
            ("planner algorithm version", self.algorithm_version),
            ("planner config hash", self.config_hash),
            ("planner target id", self.target_id),
        ):
            _text(value, name)
        if (self.algorithm_id != PLANNER_ALGORITHM_ID_V7V3
                or self.algorithm_version != PLANNER_ALGORITHM_VERSION_V7V3
                or self.config_hash != PLANNER_CONFIG_HASH_V7V3
                or self.target_id != PLANNER_TARGET_ID_V7V3):
            raise ValueError("planner algorithm/config/target authority drifted")
        if self.frozen_before_scoring is not True:
            raise ValueError("planner model must be frozen before scoring")
        payload = {
            "fit_receipt_hash": self.fit_receipt.receipt_hash,
            "feature_registry_fingerprint": (
                self.feature_registry.combined_fingerprint
            ),
            "planner_allowlist_hash": (
                self.feature_registry.planner_allowlist_hash
            ),
            "model_hash": self.model_hash,
            "model_provenance_hash": self.model_provenance_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "config_hash": self.config_hash,
            "target_id": self.target_id,
            "frozen_before_scoring": self.frozen_before_scoring,
        }
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "planner model receipt hash",
        ))


@dataclass(frozen=True)
class ImmutableNativeReceiptV7V3:
    source_hash: str
    source_materialization_hash: str
    endpoint_id: str
    execution_unit_id: str
    support_hash: str
    support_policy_hash: str
    producer_hash: str
    frozen_before_manifest: bool
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("native source hash", self.source_hash),
            ("native source materialization hash", self.source_materialization_hash),
            ("native endpoint id", self.endpoint_id),
            ("native execution unit id", self.execution_unit_id),
            ("native support hash", self.support_hash),
            ("native support-policy hash", self.support_policy_hash),
            ("native producer hash", self.producer_hash),
        ):
            _text(value, name)
        if self.frozen_before_manifest is not True:
            raise ValueError("native source receipt must be frozen before manifest")
        payload = {
            name: getattr(self, name)
            for name in (
                "source_hash", "source_materialization_hash", "endpoint_id",
                "execution_unit_id", "support_hash", "support_policy_hash",
                "producer_hash", "frozen_before_manifest",
            )
        }
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "immutable native receipt hash",
        ))


@dataclass(frozen=True)
class ManifestFreezeReceiptV7V3:
    base_manifest_hash: str
    candidate_family_hash: str
    source_hash: str
    target_fold_role: v1.FoldRoleV7
    sr0a_manifest_receipt_hash: str
    freeze_sequence: int
    outcome_capability_absent: bool
    provenance_hash: str
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("freeze base-manifest hash", self.base_manifest_hash),
            ("freeze candidate-family hash", self.candidate_family_hash),
            ("freeze source hash", self.source_hash),
            ("SR0A manifest receipt hash", self.sr0a_manifest_receipt_hash),
            ("freeze provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        if not isinstance(self.target_fold_role, v1.FoldRoleV7):
            raise ValueError("manifest freeze target fold is untyped")
        _integer(self.freeze_sequence, "manifest freeze sequence", minimum=1)
        if self.outcome_capability_absent is not True:
            raise ValueError("manifest freeze requires absent outcome capability")
        payload = {
            "base_manifest_hash": self.base_manifest_hash,
            "candidate_family_hash": self.candidate_family_hash,
            "source_hash": self.source_hash,
            "target_fold_role": self.target_fold_role.value,
            "sr0a_manifest_receipt_hash": self.sr0a_manifest_receipt_hash,
            "freeze_sequence": self.freeze_sequence,
            "outcome_capability_absent": self.outcome_capability_absent,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "manifest freeze receipt hash",
        ))


def _validate_canonical_native(
    manifest: v1.CandidateManifestV7,
    receipt: ImmutableNativeReceiptV7V3,
) -> None:
    if not _rehash_valid(manifest) or not _rehash_valid(receipt):
        raise ValueError("manifest or native receipt is malformed")
    native = manifest.native
    expected_action_hash = _sha256(["native", receipt.endpoint_id])
    expected_control_hash = _sha256(["native-control", receipt.endpoint_id])
    expected_endpoint_hash = _sha256(["endpoint", receipt.endpoint_id])
    zero = v1.CostCountersV7.zero()
    exact = (
        native.candidate_id == "action0:native"
        and native.action_index == 0
        and native.action_identity == "native"
        and native.mechanism_id == "native"
        and native.operator_version == "immutable-v1"
        and native.exact_control_id == "native"
        and native.endpoint_id == receipt.endpoint_id
        and native.input_strength == 0.0
        and native.output_beta == 0.0
        and native.action_hash == expected_action_hash
        and native.control_hash == expected_control_hash
        and native.endpoint_hash == expected_endpoint_hash
        and native.support_hash == receipt.support_hash
        and native.support_policy_hash == receipt.support_policy_hash
        and native.rollback_hash == receipt.source_materialization_hash
        and native.source_input_hashes == (receipt.source_hash,)
        and native.materialization_recipe_id == "immutable_native"
        and native.materialization_recipe_version == "v1"
        and native.materialization_recipe_hash
        == receipt.source_materialization_hash
        and native.scope is v1.CandidateScopeV7.NATIVE
        and native.unit_ids == (receipt.execution_unit_id,)
        and not native.interaction_receipts
        and native.offline_bank_acquisition_cost_ceiling == 0.0
        and native.offline_bank_acquisition_cost_counter_ceiling.content_hash
        == zero.content_hash
        and native.prospective_runtime_cost_ceiling == 0.0
        and native.prospective_runtime_cost_counter_ceiling.content_hash
        == zero.content_hash
        and native.hard_legal
        and not native.hard_reject_reasons
        and manifest.source_hash == receipt.source_hash
        and manifest.support_policy_hash == receipt.support_policy_hash
        and manifest.execution_unit_id == receipt.execution_unit_id
    )
    if not exact:
        raise ValueError("action-0 native identity is not canonical and immutable")


@dataclass(frozen=True)
class CandidateManifestV7V3:
    base_manifest: v1.CandidateManifestV7
    immutable_native_receipt: ImmutableNativeReceiptV7V3
    freeze_receipt: ManifestFreezeReceiptV7V3
    planner_model_receipt: PlannerModelReceiptV7V3
    schema_version: str = SELECTOR_V7_V3_SCHEMA_VERSION
    predecessor_failed_review_sha256: str = SELECTOR_V7_V2_REVIEWED_SHA256
    manifest_v3_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SELECTOR_V7_V3_SCHEMA_VERSION:
            raise ValueError("SelectorV7-v3 schema version drifted")
        if self.predecessor_failed_review_sha256 != (
            SELECTOR_V7_V2_REVIEWED_SHA256
        ):
            raise ValueError("SelectorV7-v3 predecessor disposition drifted")
        if not isinstance(self.base_manifest, v1.CandidateManifestV7):
            raise ValueError("v2 manifest needs a typed v1 base manifest")
        if not isinstance(self.immutable_native_receipt, ImmutableNativeReceiptV7V3):
            raise ValueError("v3 manifest needs immutable native receipt")
        if not isinstance(self.freeze_receipt, ManifestFreezeReceiptV7V3):
            raise ValueError("v3 manifest needs typed outcome-blind freeze receipt")
        if not isinstance(self.planner_model_receipt, PlannerModelReceiptV7V3):
            raise ValueError("v3 manifest needs typed planner model authority")
        _validate_canonical_native(
            self.base_manifest, self.immutable_native_receipt,
        )
        freeze = self.freeze_receipt
        if (self.base_manifest.frozen_before_outcome is not True
                or freeze.base_manifest_hash != self.base_manifest.manifest_hash
                or freeze.candidate_family_hash
                != self.base_manifest.candidate_family_hash
                or freeze.source_hash != self.base_manifest.source_hash
                or freeze.target_fold_role is not self.base_manifest.fold_role):
            raise ValueError("manifest was not frozen outcome-blind by SR0A")
        model = self.planner_model_receipt
        if (model.fit_receipt.target_fold_role is not self.base_manifest.fold_role
                or model.feature_registry.planner_allowlist_hash
                != self.base_manifest.planner_allowlist_hash):
            raise ValueError("planner model authority does not bind manifest")
        payload = {
            "schema_version": self.schema_version,
            "predecessor_failed_review_sha256": (
                self.predecessor_failed_review_sha256
            ),
            "base_manifest_hash": self.base_manifest.manifest_hash,
            "immutable_native_receipt_hash": (
                self.immutable_native_receipt.receipt_hash
            ),
            "freeze_receipt_hash": freeze.receipt_hash,
            "planner_model_receipt_hash": model.receipt_hash,
        }
        object.__setattr__(self, "manifest_v3_hash", _sealed(
            self.manifest_v3_hash, payload, "SelectorV7-v3 manifest hash",
        ))

    @property
    def native(self) -> v1.CandidateArmV7:
        return self.base_manifest.native


@dataclass(frozen=True)
class PreRunAliasSealV7V3:
    manifest_v3_hash: str
    representative_planned_arm_id: str
    alias_planned_arm_ids: tuple[str, ...]
    alias_receipt_bindings: tuple[tuple[str, str], ...]
    sealed_at_ns: int
    reuse_policy_hash: str
    provenance_hash: str
    seal_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("alias-seal manifest hash", self.manifest_v3_hash),
            ("alias-seal representative id", self.representative_planned_arm_id),
            ("alias-seal reuse policy hash", self.reuse_policy_hash),
            ("alias-seal provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        aliases = _sorted_unique(
            self.alias_planned_arm_ids, "pre-run alias planned ids",
        )
        if not aliases or self.representative_planned_arm_id in aliases:
            raise ValueError("pre-run alias seal needs nonrepresentative aliases")
        bindings = tuple(sorted(self.alias_receipt_bindings))
        if tuple(row[0] for row in bindings) != aliases:
            raise ValueError("pre-run alias seal must bind every alias receipt")
        if any(not isinstance(row, tuple) or len(row) != 2
               or not row[0] or not row[1] for row in bindings):
            raise ValueError("pre-run alias bindings are malformed")
        _integer(self.sealed_at_ns, "alias seal timestamp")
        payload = {
            "manifest_v3_hash": self.manifest_v3_hash,
            "representative_planned_arm_id": self.representative_planned_arm_id,
            "alias_planned_arm_ids": list(aliases),
            "alias_receipt_bindings": [list(row) for row in bindings],
            "sealed_at_ns": self.sealed_at_ns,
            "reuse_policy_hash": self.reuse_policy_hash,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "alias_planned_arm_ids", aliases)
        object.__setattr__(self, "alias_receipt_bindings", bindings)
        object.__setattr__(self, "seal_hash", _sealed(
            self.seal_hash, payload, "pre-run alias seal hash",
        ))


@dataclass(frozen=True)
class AliasRunPlanV7V3:
    manifest_v3_hash: str
    runtime_run_id: str
    representative_planned_arm_id: str
    alias_planned_arm_ids: tuple[str, ...]
    alias_receipt_bindings: tuple[tuple[str, str], ...]
    mode: AliasChronologyModeV7V3
    earliest_run_start_ns: int
    pre_run_alias_seal: PreRunAliasSealV7V3 | None
    runtime_policy_hash: str
    provenance_hash: str
    plan_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("run-plan manifest hash", self.manifest_v3_hash),
            ("runtime run id", self.runtime_run_id),
            ("run-plan representative id", self.representative_planned_arm_id),
            ("runtime policy hash", self.runtime_policy_hash),
            ("run-plan provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        if not isinstance(self.mode, AliasChronologyModeV7V3):
            raise ValueError("run-plan alias chronology mode is untyped")
        aliases = _sorted_unique(self.alias_planned_arm_ids, "run-plan aliases")
        bindings = tuple(sorted(self.alias_receipt_bindings))
        if tuple(row[0] for row in bindings) != aliases:
            raise ValueError("run plan must bind each alias receipt exactly once")
        _integer(self.earliest_run_start_ns, "earliest run start")
        if self.mode is AliasChronologyModeV7V3.DIRECT_NO_ALIAS:
            if aliases or bindings or self.pre_run_alias_seal is not None:
                raise ValueError("direct run plan cannot carry aliases or reuse seal")
        elif not aliases:
            raise ValueError("alias chronology mode needs at least one alias")
        if self.mode is AliasChronologyModeV7V3.PRE_RUN_FROZEN_REUSE:
            seal = self.pre_run_alias_seal
            if not isinstance(seal, PreRunAliasSealV7V3):
                raise ValueError("pre-run reuse needs a typed frozen alias seal")
            if (seal.manifest_v3_hash != self.manifest_v3_hash
                    or seal.representative_planned_arm_id
                    != self.representative_planned_arm_id
                    or seal.alias_planned_arm_ids != aliases
                    or seal.alias_receipt_bindings != bindings
                    or seal.sealed_at_ns >= self.earliest_run_start_ns):
                raise ValueError("alias reuse seal was not frozen before runtime")
        elif self.pre_run_alias_seal is not None:
            raise ValueError("post-hoc/duplicate modes cannot claim pre-run reuse")
        payload = {
            "manifest_v3_hash": self.manifest_v3_hash,
            "runtime_run_id": self.runtime_run_id,
            "representative_planned_arm_id": self.representative_planned_arm_id,
            "alias_planned_arm_ids": list(aliases),
            "alias_receipt_bindings": [list(row) for row in bindings],
            "mode": self.mode.value,
            "earliest_run_start_ns": self.earliest_run_start_ns,
            "pre_run_alias_seal_hash": (
                self.pre_run_alias_seal.seal_hash
                if self.pre_run_alias_seal is not None else None
            ),
            "runtime_policy_hash": self.runtime_policy_hash,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "alias_planned_arm_ids", aliases)
        object.__setattr__(self, "alias_receipt_bindings", bindings)
        object.__setattr__(self, "plan_hash", _sealed(
            self.plan_hash, payload, "alias run-plan hash",
        ))


@dataclass(frozen=True)
class RuntimeExecutionReceiptV7V3:
    manifest_v3_hash: str
    proposal_hash: str
    alias_run_plan_hash: str
    runtime_run_id: str
    planned_arm_id: str
    realization_receipt_hash: str
    executable_bytes_hash: str
    operation: RuntimeOperationV7V3
    reused_from_planned_arm_id: str | None
    started_at_ns: int
    finished_at_ns: int
    actual_prospective_runtime_cost: float
    actual_prospective_runtime_cost_counters: v1.CostCountersV7
    cost_account: v1.CostAccountV7
    provenance_hash: str
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("runtime manifest hash", self.manifest_v3_hash),
            ("runtime proposal hash", self.proposal_hash),
            ("runtime alias-plan hash", self.alias_run_plan_hash),
            ("runtime run id", self.runtime_run_id),
            ("runtime planned arm id", self.planned_arm_id),
            ("runtime realization receipt hash", self.realization_receipt_hash),
            ("runtime executable bytes hash", self.executable_bytes_hash),
            ("runtime provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        if not isinstance(self.operation, RuntimeOperationV7V3):
            raise ValueError("runtime operation is untyped")
        start = _integer(self.started_at_ns, "runtime start timestamp")
        finish = _integer(self.finished_at_ns, "runtime finish timestamp")
        if finish < start:
            raise ValueError("runtime execution finishes before it starts")
        _nonnegative(
            self.actual_prospective_runtime_cost,
            "actual prospective-runtime cost",
        )
        if not isinstance(
            self.actual_prospective_runtime_cost_counters,
            v1.CostCountersV7,
        ):
            raise ValueError("runtime execution needs typed cost counters")
        if self.cost_account is not v1.CostAccountV7.PROSPECTIVE_RUNTIME:
            raise ValueError("runtime execution has the wrong cost account")
        if self.operation is RuntimeOperationV7V3.EXECUTE_ARM:
            if self.reused_from_planned_arm_id is not None:
                raise ValueError("executed arm cannot claim alias reuse")
        else:
            _text(self.reused_from_planned_arm_id, "alias reuse source id")
            forbidden = (
                "natural_forwards", "reused_forwards", "candidate_forwards",
                "reverse_forwards", "gpu_seconds", "matcher_trajectories",
            )
            if any(getattr(
                self.actual_prospective_runtime_cost_counters, name,
            ) != 0 for name in forbidden):
                raise ValueError("alias verification cannot hide an execution")
        payload = {
            "manifest_v3_hash": self.manifest_v3_hash,
            "proposal_hash": self.proposal_hash,
            "alias_run_plan_hash": self.alias_run_plan_hash,
            "runtime_run_id": self.runtime_run_id,
            "planned_arm_id": self.planned_arm_id,
            "realization_receipt_hash": self.realization_receipt_hash,
            "executable_bytes_hash": self.executable_bytes_hash,
            "operation": self.operation.value,
            "reused_from_planned_arm_id": self.reused_from_planned_arm_id,
            "started_at_ns": start,
            "finished_at_ns": finish,
            "actual_prospective_runtime_cost": (
                self.actual_prospective_runtime_cost
            ),
            "actual_prospective_runtime_cost_counters_hash": (
                self.actual_prospective_runtime_cost_counters.content_hash
            ),
            "cost_account": self.cost_account.value,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "runtime execution receipt hash",
        ))


@dataclass(frozen=True)
class RuntimeCostLedgerV7V3:
    manifest_v3_hash: str
    proposal_hash: str
    alias_run_plan_hash: str
    runtime_run_id: str
    representative_planned_arm_id: str
    alias_planned_arm_ids: tuple[str, ...]
    mode: AliasChronologyModeV7V3
    execution_receipts: tuple[RuntimeExecutionReceiptV7V3, ...]
    post_hoc_discovered_at_ns: int | None
    charged_prospective_runtime_cost: float
    charged_prospective_runtime_cost_counters: v1.CostCountersV7
    cost_account: v1.CostAccountV7
    provenance_hash: str
    ledger_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("ledger manifest hash", self.manifest_v3_hash),
            ("ledger proposal hash", self.proposal_hash),
            ("ledger alias-plan hash", self.alias_run_plan_hash),
            ("ledger run id", self.runtime_run_id),
            ("ledger representative id", self.representative_planned_arm_id),
            ("ledger provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        aliases = _sorted_unique(self.alias_planned_arm_ids, "ledger aliases")
        if not isinstance(self.mode, AliasChronologyModeV7V3):
            raise ValueError("ledger alias chronology mode is untyped")
        receipts = tuple(self.execution_receipts)
        if not receipts or any(not isinstance(
            row, RuntimeExecutionReceiptV7V3,
        ) for row in receipts):
            raise ValueError("runtime ledger needs typed execution receipts")
        if len({row.receipt_hash for row in receipts}) != len(receipts):
            raise ValueError("runtime execution receipts must be unique")
        if any(
            row.manifest_v3_hash != self.manifest_v3_hash
            or row.proposal_hash != self.proposal_hash
            or row.alias_run_plan_hash != self.alias_run_plan_hash
            or row.runtime_run_id != self.runtime_run_id
            for row in receipts
        ):
            raise ValueError("runtime ledger receipt binding drifted")
        charged = sum(row.actual_prospective_runtime_cost for row in receipts)
        if not math.isclose(
            charged, _nonnegative(
                self.charged_prospective_runtime_cost,
                "ledger charged prospective-runtime cost",
            ), rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("runtime ledger scalar cost does not sum receipts")
        if not isinstance(
            self.charged_prospective_runtime_cost_counters,
            v1.CostCountersV7,
        ):
            raise ValueError("runtime ledger needs typed charged counters")
        summed = _sum_counters(
            tuple(row.actual_prospective_runtime_cost_counters for row in receipts),
            cache_state=receipts[0].actual_prospective_runtime_cost_counters.cache_state,
        )
        if summed.content_hash != (
            self.charged_prospective_runtime_cost_counters.content_hash
        ):
            raise ValueError("runtime ledger counters do not sum receipts")
        if self.cost_account is not v1.CostAccountV7.PROSPECTIVE_RUNTIME:
            raise ValueError("runtime ledger has the wrong cost account")
        payload = {
            "manifest_v3_hash": self.manifest_v3_hash,
            "proposal_hash": self.proposal_hash,
            "alias_run_plan_hash": self.alias_run_plan_hash,
            "runtime_run_id": self.runtime_run_id,
            "representative_planned_arm_id": self.representative_planned_arm_id,
            "alias_planned_arm_ids": list(aliases),
            "mode": self.mode.value,
            "execution_receipt_hashes": sorted(
                row.receipt_hash for row in receipts
            ),
            "post_hoc_discovered_at_ns": self.post_hoc_discovered_at_ns,
            "charged_prospective_runtime_cost": charged,
            "charged_prospective_runtime_cost_counters_hash": summed.content_hash,
            "cost_account": self.cost_account.value,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "alias_planned_arm_ids", aliases)
        object.__setattr__(self, "ledger_hash", _sealed(
            self.ledger_hash, payload, "runtime cost ledger hash",
        ))


@dataclass(frozen=True)
class PlannerScoreReceiptV7V3:
    base_score: v1.PlannerCandidateV7
    planner_model_receipt_hash: str
    planner_model_hash: str
    planner_fit_receipt_hash: str
    target_fold_role: v1.FoldRoleV7
    feature_registry_fingerprint: str
    feature_schema_hash: str
    score_provenance_hash: str
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.base_score, v1.PlannerCandidateV7):
            raise ValueError("planner score receipt needs typed base score")
        for name, value in (
            ("score planner-model receipt hash", self.planner_model_receipt_hash),
            ("score planner-model hash", self.planner_model_hash),
            ("score planner FIT receipt hash", self.planner_fit_receipt_hash),
            ("score feature-registry fingerprint",
             self.feature_registry_fingerprint),
            ("score feature-schema hash", self.feature_schema_hash),
            ("score provenance hash", self.score_provenance_hash),
        ):
            _text(value, name)
        if not isinstance(self.target_fold_role, v1.FoldRoleV7):
            raise ValueError("planner score target fold is untyped")
        score_payload = {
            name: (
                getattr(self.base_score, name).value
                if isinstance(getattr(self.base_score, name), Enum)
                else getattr(self.base_score, name)
            )
            for name in self.base_score.__dataclass_fields__
        }
        payload = {
            "base_score": score_payload,
            "planner_model_receipt_hash": self.planner_model_receipt_hash,
            "planner_model_hash": self.planner_model_hash,
            "planner_fit_receipt_hash": self.planner_fit_receipt_hash,
            "target_fold_role": self.target_fold_role.value,
            "feature_registry_fingerprint": self.feature_registry_fingerprint,
            "feature_schema_hash": self.feature_schema_hash,
            "score_provenance_hash": self.score_provenance_hash,
        }
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "planner score receipt hash",
        ))


@dataclass(frozen=True)
class PlannerScoreBankV7V3:
    manifest_v3_hash: str
    candidate_family_hash: str
    planner_allowlist_hash: str
    planner_model_receipt: PlannerModelReceiptV7V3
    universe_digest: str
    universe_cardinality: int
    scores: tuple[v1.PlannerCandidateV7, ...]
    frozen_before_observation: bool
    score_receipts: tuple[PlannerScoreReceiptV7V3, ...] = ()
    score_bank_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("score-bank manifest hash", self.manifest_v3_hash),
            ("score-bank candidate-family hash", self.candidate_family_hash),
            ("score-bank planner allowlist hash", self.planner_allowlist_hash),
            ("score-bank universe digest", self.universe_digest),
        ):
            _text(value, name)
        if not isinstance(self.planner_model_receipt, PlannerModelReceiptV7V3):
            raise ValueError("score bank needs typed planner model authority")
        if (self.planner_allowlist_hash
                != self.planner_model_receipt.feature_registry.planner_allowlist_hash):
            raise ValueError("score bank allowlist differs from planner authority")
        count = _integer(
            self.universe_cardinality, "score-bank universe cardinality",
        )
        rows = tuple(self.scores)
        if any(not isinstance(row, v1.PlannerCandidateV7) for row in rows):
            raise ValueError("score bank needs typed planner rows")
        if len({row.candidate_id for row in rows}) != len(rows):
            raise ValueError("score bank candidate ids must be unique")
        if len({row.planned_arm_id for row in rows}) != len(rows):
            raise ValueError("score bank planned ids must be unique")
        if len(rows) != count:
            raise ValueError("score bank cardinality does not match its rows")
        if self.frozen_before_observation is not True:
            raise ValueError("score bank must be frozen before observation")
        authority = self.planner_model_receipt
        receipts = tuple(self.score_receipts) or tuple(
            PlannerScoreReceiptV7V3(
                base_score=row,
                planner_model_receipt_hash=authority.receipt_hash,
                planner_model_hash=authority.model_hash,
                planner_fit_receipt_hash=authority.fit_receipt.receipt_hash,
                target_fold_role=authority.fit_receipt.target_fold_role,
                feature_registry_fingerprint=(
                    authority.feature_registry.combined_fingerprint
                ),
                feature_schema_hash=authority.feature_schema_hash,
                score_provenance_hash=authority.model_provenance_hash,
            )
            for row in rows
        )
        by_receipt_id = {
            row.base_score.planned_arm_id: row for row in receipts
        }
        if (len(by_receipt_id) != len(receipts)
                or set(by_receipt_id)
                != {row.planned_arm_id for row in rows}):
            raise ValueError("planner score receipt family is incomplete")
        for score in rows:
            receipt = by_receipt_id[score.planned_arm_id]
            if (receipt.base_score != score
                    or receipt.planner_model_receipt_hash
                    != authority.receipt_hash
                    or receipt.planner_model_hash != authority.model_hash
                    or receipt.planner_fit_receipt_hash
                    != authority.fit_receipt.receipt_hash
                    or receipt.target_fold_role
                    is not authority.fit_receipt.target_fold_role
                    or receipt.feature_registry_fingerprint
                    != authority.feature_registry.combined_fingerprint
                    or receipt.feature_schema_hash
                    != authority.feature_schema_hash
                    or receipt.score_provenance_hash
                    != authority.model_provenance_hash):
                raise ValueError("planner score differs from model/FIT authority")
        payload = {
            "manifest_v3_hash": self.manifest_v3_hash,
            "candidate_family_hash": self.candidate_family_hash,
            "planner_allowlist_hash": self.planner_allowlist_hash,
            "planner_model_receipt_hash": (
                self.planner_model_receipt.receipt_hash
            ),
            "planner_model_hash": self.planner_model_receipt.model_hash,
            "planner_fit_receipt_hash": (
                self.planner_model_receipt.fit_receipt.receipt_hash
            ),
            "planner_feature_registry_fingerprint": (
                self.planner_model_receipt.feature_registry.combined_fingerprint
            ),
            "universe_digest": self.universe_digest,
            "universe_cardinality": count,
            "frozen_before_observation": self.frozen_before_observation,
            "scores": [
                {
                    name: (
                        getattr(row, name).value
                        if isinstance(getattr(row, name), Enum)
                        else getattr(row, name)
                    )
                    for name in row.__dataclass_fields__
                }
                for row in sorted(rows, key=lambda item: item.planned_arm_id)
            ],
            "score_receipt_hashes": [
                row.receipt_hash for row in sorted(
                    receipts,
                    key=lambda item: item.base_score.planned_arm_id,
                )
            ],
        }
        object.__setattr__(self, "score_receipts", receipts)
        object.__setattr__(self, "score_bank_hash", _sealed(
            self.score_bank_hash, payload, "planner score-bank hash",
        ))


@dataclass(frozen=True)
class PlannerProposalV7V3:
    manifest_v3_hash: str
    candidate_family_hash: str
    score_bank: PlannerScoreBankV7V3
    alias_run_plans: tuple[AliasRunPlanV7V3, ...]
    alias_run_plan_digest: str
    planner_algorithm_id: str
    planner_algorithm_version: str
    planner_config_hash: str
    planner_provenance_hash: str
    profile: v1.PlannerProfileV7
    top_k: int
    maximum_prospective_runtime_cost: float
    maximum_prospective_runtime_cost_counters: v1.CostCountersV7
    reserved_prospective_runtime_cost: float
    reserved_prospective_runtime_cost_counters: v1.CostCountersV7
    deterministic_ranked_planned_arm_ids: tuple[str, ...]
    selected_planned_arm_ids: tuple[str, ...]
    frozen_before_observation: bool
    proposal_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("proposal manifest hash", self.manifest_v3_hash),
            ("proposal candidate-family hash", self.candidate_family_hash),
            ("proposal run-plan digest", self.alias_run_plan_digest),
            ("planner algorithm id", self.planner_algorithm_id),
            ("planner algorithm version", self.planner_algorithm_version),
            ("planner config hash", self.planner_config_hash),
            ("planner provenance hash", self.planner_provenance_hash),
        ):
            _text(value, name)
        if not isinstance(self.score_bank, PlannerScoreBankV7V3):
            raise ValueError("proposal needs a typed complete score bank")
        authority = self.score_bank.planner_model_receipt
        if (self.planner_algorithm_id != authority.algorithm_id
                or self.planner_algorithm_version != authority.algorithm_version
                or self.planner_config_hash != authority.config_hash
                or self.planner_provenance_hash
                != authority.model_provenance_hash):
            raise ValueError("proposal metadata differs from planner authority")
        plans = tuple(self.alias_run_plans)
        if any(not isinstance(row, AliasRunPlanV7V3) for row in plans):
            raise ValueError("proposal needs typed alias run plans")
        if len({row.representative_planned_arm_id for row in plans}) != len(plans):
            raise ValueError("proposal run plans must have unique representatives")
        if self.profile not in {v1.PlannerProfileV7.K1_ABLATION,
                                v1.PlannerProfileV7.K2_PRIMARY}:
            raise ValueError("proposal profile is invalid")
        expected_k = 1 if self.profile is v1.PlannerProfileV7.K1_ABLATION else 2
        if self.top_k != expected_k:
            raise ValueError("proposal profile and top-K disagree")
        ceiling = _nonnegative(
            self.maximum_prospective_runtime_cost,
            "proposal prospective-runtime ceiling",
        )
        reserved = _nonnegative(
            self.reserved_prospective_runtime_cost,
            "proposal prospective-runtime reservation",
        )
        if reserved > ceiling + 1e-12:
            raise ValueError("proposal reservation exceeds scalar ceiling")
        if not isinstance(self.maximum_prospective_runtime_cost_counters,
                          v1.CostCountersV7) or not isinstance(
            self.reserved_prospective_runtime_cost_counters,
            v1.CostCountersV7,
        ):
            raise ValueError("proposal needs typed cost-counter ceilings")
        if not self.reserved_prospective_runtime_cost_counters.within(
            self.maximum_prospective_runtime_cost_counters,
        ):
            raise ValueError("proposal counter reservation exceeds ceiling")
        ranked = _sorted_unique(
            self.deterministic_ranked_planned_arm_ids,
            "proposal deterministic rank ids",
        )
        # Preserve rank order while separately checking uniqueness.
        ranked = tuple(self.deterministic_ranked_planned_arm_ids)
        if len(ranked) != len(set(ranked)) or any(not row for row in ranked):
            raise ValueError("proposal deterministic rank must be unique")
        selected = tuple(self.selected_planned_arm_ids)
        if len(selected) > self.top_k or len(selected) != len(set(selected)):
            raise ValueError("proposal selected ids violate top-K")
        if not set(selected).issubset(ranked):
            raise ValueError("proposal selected ids lie outside deterministic rank")
        if self.frozen_before_observation is not True:
            raise ValueError("proposal must be frozen before observation")
        payload = {
            "manifest_v3_hash": self.manifest_v3_hash,
            "candidate_family_hash": self.candidate_family_hash,
            "score_bank_hash": self.score_bank.score_bank_hash,
            "planner_model_receipt_hash": authority.receipt_hash,
            "alias_run_plan_digest": self.alias_run_plan_digest,
            "planner_algorithm_id": self.planner_algorithm_id,
            "planner_algorithm_version": self.planner_algorithm_version,
            "planner_config_hash": self.planner_config_hash,
            "planner_provenance_hash": self.planner_provenance_hash,
            "profile": self.profile.value,
            "top_k": self.top_k,
            "maximum_prospective_runtime_cost": ceiling,
            "maximum_prospective_runtime_cost_counters_hash": (
                self.maximum_prospective_runtime_cost_counters.content_hash
            ),
            "reserved_prospective_runtime_cost": reserved,
            "reserved_prospective_runtime_cost_counters_hash": (
                self.reserved_prospective_runtime_cost_counters.content_hash
            ),
            "deterministic_ranked_planned_arm_ids": list(ranked),
            "selected_planned_arm_ids": list(selected),
            "frozen_before_observation": self.frozen_before_observation,
        }
        object.__setattr__(self, "proposal_hash", _sealed(
            self.proposal_hash, payload, "SelectorV7-v3 proposal hash",
        ))


def _manifest_arm_maps(manifest: CandidateManifestV7V3) -> tuple[
    dict[str, v1.CandidateArmV7], dict[str, v1.CandidateArmV7]
]:
    by_planned = {
        row.planned_arm_id: row for row in manifest.base_manifest.candidates
    }
    by_candidate = {
        row.candidate_id: row for row in manifest.base_manifest.candidates
    }
    return by_planned, by_candidate


def _realization_state(
    manifest: CandidateManifestV7V3,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
) -> tuple[
    dict[str, v1.ArmRealizationReceiptV7],
    dict[str, str],
    dict[str, tuple[str, ...]],
    dict[str, v1.ArmAliasReceiptV7],
]:
    _validate_canonical_native(
        manifest.base_manifest, manifest.immutable_native_receipt,
    )
    by_realization, representatives = v1._validated_realization_aliases_v7(
        manifest.base_manifest, realizations, aliases,
    )
    groups: dict[str, list[str]] = {}
    for planned_id, receipt in by_realization.items():
        arm = next(row for row in manifest.base_manifest.candidates
                   if row.planned_arm_id == planned_id)
        if (arm.action_index != 0
                and receipt.status is v1.ArmRealizationStatusV7.COMPLETE):
            groups.setdefault(representatives[planned_id], []).append(planned_id)
    frozen_groups = {
        rep: tuple(sorted(members)) for rep, members in groups.items()
    }
    alias_by_id = {row.alias_planned_arm_id: row for row in aliases}
    return by_realization, representatives, frozen_groups, alias_by_id


def _validate_run_plan(
    manifest: CandidateManifestV7V3,
    plan: AliasRunPlanV7V3,
    group_members: tuple[str, ...],
    alias_by_id: dict[str, v1.ArmAliasReceiptV7],
) -> None:
    representative = min(group_members)
    aliases = tuple(row for row in group_members if row != representative)
    expected_bindings = tuple(sorted(
        (row, alias_by_id[row].receipt_hash) for row in aliases
    ))
    if (plan.manifest_v3_hash != manifest.manifest_v3_hash
            or plan.representative_planned_arm_id != representative
            or plan.alias_planned_arm_ids != aliases
            or plan.alias_receipt_bindings != expected_bindings):
        raise ValueError("alias run plan does not match exact canonical group")
    if not aliases and plan.mode is not AliasChronologyModeV7V3.DIRECT_NO_ALIAS:
        raise ValueError("non-alias arm must use direct chronology")
    if aliases and plan.mode is AliasChronologyModeV7V3.DIRECT_NO_ALIAS:
        raise ValueError("alias group cannot use direct chronology")


def _group_cost_ceiling(
    representative: str,
    group_members: tuple[str, ...],
    plan: AliasRunPlanV7V3,
    by_arm: dict[str, v1.CandidateArmV7],
    alias_by_id: dict[str, v1.ArmAliasReceiptV7],
) -> tuple[float, v1.CostCountersV7]:
    if plan.mode is AliasChronologyModeV7V3.PRE_RUN_FROZEN_REUSE:
        scalar = by_arm[representative].prospective_runtime_cost_ceiling
        counters = by_arm[representative].prospective_runtime_cost_counter_ceiling
        for planned_id in group_members:
            if planned_id == representative:
                continue
            alias = alias_by_id[planned_id]
            scalar += alias.prospective_runtime_verification_cost
            counters = counters.plus(
                alias.prospective_runtime_verification_cost_counters
            )
        return float(scalar), counters
    rows = tuple(by_arm[planned_id] for planned_id in group_members)
    return (
        sum(row.prospective_runtime_cost_ceiling for row in rows),
        _sum_counters(tuple(
            row.prospective_runtime_cost_counter_ceiling for row in rows
        ), cache_state=rows[0].prospective_runtime_cost_counter_ceiling.cache_state),
    )


def _proposal_context(
    manifest: CandidateManifestV7V3,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    run_plans: Sequence[AliasRunPlanV7V3],
    maximum_cost: float,
    maximum_counters: v1.CostCountersV7,
) -> tuple[
    tuple[v1.CandidateArmV7, ...],
    dict[str, tuple[float, v1.CostCountersV7]],
]:
    by_realization, _, groups, alias_by_id = _realization_state(
        manifest, realizations, aliases,
    )
    by_arm, _ = _manifest_arm_maps(manifest)
    plans = {row.representative_planned_arm_id: row for row in run_plans}
    if len(plans) != len(tuple(run_plans)) or set(plans) != set(groups):
        raise ValueError("run-plan family is incomplete or injected")
    costs: dict[str, tuple[float, v1.CostCountersV7]] = {}
    universe: list[v1.CandidateArmV7] = []
    for representative, members in groups.items():
        plan = plans[representative]
        _validate_run_plan(manifest, plan, members, alias_by_id)
        arm = by_arm[representative]
        cost = _group_cost_ceiling(
            representative, members, plan, by_arm, alias_by_id,
        )
        costs[representative] = cost
        if (arm.hard_legal and arm.composition_reason() is None
                and cost[0] <= maximum_cost + 1e-12
                and cost[1].within(maximum_counters)
                and by_realization[representative].status
                is v1.ArmRealizationStatusV7.COMPLETE):
            universe.append(arm)
    return tuple(sorted(universe, key=lambda row: row.planned_arm_id)), costs


def propose_candidates_v7_v3(
    manifest: CandidateManifestV7V3,
    scores: Sequence[v1.PlannerCandidateV7],
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    alias_run_plans: Sequence[AliasRunPlanV7V3],
    *,
    top_k: int,
    maximum_prospective_runtime_cost: float,
    maximum_prospective_runtime_cost_counters: v1.CostCountersV7,
    planner_algorithm_id: str,
    planner_algorithm_version: str,
    planner_config_hash: str,
    planner_provenance_hash: str,
) -> PlannerProposalV7V3:
    """Seal a complete score bank, then deterministically replay K1/K2."""
    if not _rehash_valid(manifest):
        raise ValueError("SelectorV7-v3 manifest is malformed")
    if top_k not in {1, 2}:
        raise ValueError("SelectorV7-v3 top-K must be one or two")
    authority = manifest.planner_model_receipt
    if (planner_algorithm_id != authority.algorithm_id
            or planner_algorithm_version != authority.algorithm_version
            or planner_config_hash != authority.config_hash
            or planner_provenance_hash != authority.model_provenance_hash
            or authority.feature_registry.planner_allowlist_hash
            != manifest.base_manifest.planner_allowlist_hash):
        raise ValueError("planner metadata is not manifest-authorized")
    ceiling = _nonnegative(
        maximum_prospective_runtime_cost, "planner runtime cost ceiling",
    )
    if not isinstance(maximum_prospective_runtime_cost_counters,
                      v1.CostCountersV7):
        raise ValueError("planner needs typed runtime counter ceiling")
    universe, costs = _proposal_context(
        manifest, realizations, aliases, alias_run_plans,
        ceiling, maximum_prospective_runtime_cost_counters,
    )
    universe_ids = tuple(row.planned_arm_id for row in universe)
    universe_digest = _sha256({
        "manifest_v3_hash": manifest.manifest_v3_hash,
        "planned_arm_ids": list(universe_ids),
    })
    rows = tuple(scores)
    by_score = {row.planned_arm_id: row for row in rows}
    if len(by_score) != len(rows) or set(by_score) != set(universe_ids):
        raise ValueError("complete planner score bank is missing or injected rows")
    by_arm = {row.planned_arm_id: row for row in universe}
    for planned_id, score in by_score.items():
        arm = by_arm[planned_id]
        if (score.candidate_id != arm.candidate_id
                or score.cost_hash != arm.cost_hash
                or score.planner_allowlist_hash
                != manifest.base_manifest.planner_allowlist_hash):
            raise ValueError("planner score binding drifted")
    score_bank = PlannerScoreBankV7V3(
        manifest_v3_hash=manifest.manifest_v3_hash,
        candidate_family_hash=manifest.base_manifest.candidate_family_hash,
        planner_allowlist_hash=manifest.base_manifest.planner_allowlist_hash,
        planner_model_receipt=authority,
        universe_digest=universe_digest,
        universe_cardinality=len(universe),
        scores=rows,
        frozen_before_observation=True,
    )
    eligible = [
        (score, by_arm[planned_id])
        for planned_id, score in by_score.items()
        if score.before_features_complete
    ]
    eligible.sort(key=lambda pair: (
        -pair[0].planner_predicted_gain_raw_px,
        costs[pair[1].planned_arm_id][0],
        pair[1].planned_arm_id,
    ))
    ranked = tuple(row.planned_arm_id for _, row in eligible)
    selected: list[str] = []
    reserved_scalar = 0.0
    reserved_counters = v1.CostCountersV7.zero(
        cache_state=maximum_prospective_runtime_cost_counters.cache_state,
    )
    for _, arm in eligible:
        cost, counters = costs[arm.planned_arm_id]
        next_counters = reserved_counters.plus(counters)
        if (reserved_scalar + cost > ceiling + 1e-12
                or not next_counters.within(
                    maximum_prospective_runtime_cost_counters
                )):
            continue
        selected.append(arm.planned_arm_id)
        reserved_scalar += cost
        reserved_counters = next_counters
        if len(selected) == top_k:
            break
    plan_digest = _sha256([
        row.plan_hash for row in sorted(
            alias_run_plans,
            key=lambda item: item.representative_planned_arm_id,
        )
    ])
    return PlannerProposalV7V3(
        manifest_v3_hash=manifest.manifest_v3_hash,
        candidate_family_hash=manifest.base_manifest.candidate_family_hash,
        score_bank=score_bank,
        alias_run_plans=tuple(alias_run_plans),
        alias_run_plan_digest=plan_digest,
        planner_algorithm_id=planner_algorithm_id,
        planner_algorithm_version=planner_algorithm_version,
        planner_config_hash=planner_config_hash,
        planner_provenance_hash=planner_provenance_hash,
        profile=(v1.PlannerProfileV7.K1_ABLATION if top_k == 1
                 else v1.PlannerProfileV7.K2_PRIMARY),
        top_k=top_k,
        maximum_prospective_runtime_cost=ceiling,
        maximum_prospective_runtime_cost_counters=(
            maximum_prospective_runtime_cost_counters
        ),
        reserved_prospective_runtime_cost=reserved_scalar,
        reserved_prospective_runtime_cost_counters=reserved_counters,
        deterministic_ranked_planned_arm_ids=ranked,
        selected_planned_arm_ids=tuple(selected),
        frozen_before_observation=True,
    )


@dataclass(frozen=True)
class CandidateObservationV7V3:
    manifest_v3_hash: str
    proposal_hash: str
    candidate_id: str
    planned_arm_id: str
    realization_receipt_hash: str
    action_hash: str
    control_hash: str
    endpoint_hash: str
    support_hash: str
    realized_support_hash: str
    interaction_hash: str
    cost_hash: str
    source_hash: str
    planner_allowlist_hash: str
    assessor_allowlist_hash: str
    preaction_shared_feature_hash: str
    feature_schema_hash: str
    provenance_hash: str
    availability: v1.AvailabilityV7
    availability_reason: str
    runtime_features: tuple[v1.RuntimeFeatureV7, ...]
    evidence_blocks: tuple[v1.EvidenceBlockReceiptV7, ...]
    evidence_source_hashes: tuple[str, ...]
    observed_unit_ids: tuple[str, ...]
    atomically_observed: bool
    observation_source: ObservationSourceV7V3
    runtime_cost_ledger_hash: str | None
    observation_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("observation manifest hash", self.manifest_v3_hash),
            ("observation proposal hash", self.proposal_hash),
            ("observation candidate id", self.candidate_id),
            ("observation planned arm id", self.planned_arm_id),
            ("observation realization receipt hash", self.realization_receipt_hash),
            ("observation action hash", self.action_hash),
            ("observation control hash", self.control_hash),
            ("observation endpoint hash", self.endpoint_hash),
            ("observation support hash", self.support_hash),
            ("observation realized support hash", self.realized_support_hash),
            ("observation interaction hash", self.interaction_hash),
            ("observation cost hash", self.cost_hash),
            ("observation source hash", self.source_hash),
            ("observation planner allowlist hash", self.planner_allowlist_hash),
            ("observation assessor allowlist hash", self.assessor_allowlist_hash),
            ("observation shared-feature hash", self.preaction_shared_feature_hash),
            ("observation feature-schema hash", self.feature_schema_hash),
            ("observation provenance hash", self.provenance_hash),
            ("observation availability reason", self.availability_reason),
        ):
            _text(value, name)
        if not isinstance(self.availability, v1.AvailabilityV7):
            raise ValueError("observation availability is untyped")
        if not isinstance(self.observation_source, ObservationSourceV7V3):
            raise ValueError("observation source is untyped")
        features = tuple(self.runtime_features)
        if ({row.name for row in features}
                != v1.ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7
                or len(features)
                != len(v1.ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7)):
            raise ValueError("observation post-action feature family drifted")
        blocks = tuple(self.evidence_blocks)
        if ({row.block for row in blocks} != v1.PRIMARY_REQUIRED_EVIDENCE_BLOCKS_V7
                or len(blocks) != len(v1.PRIMARY_REQUIRED_EVIDENCE_BLOCKS_V7)):
            raise ValueError("observation evidence-block family drifted")
        sources = _sorted_unique(
            self.evidence_source_hashes, "observation evidence source hashes",
        )
        units = _sorted_unique(self.observed_unit_ids, "observed unit ids")
        if not isinstance(self.atomically_observed, bool):
            raise ValueError("observation atomic flag must be boolean")
        if self.observation_source is (
            ObservationSourceV7V3.PROSPECTIVE_RUNTIME_OBSERVATION
        ):
            _text(self.runtime_cost_ledger_hash, "runtime cost ledger hash")
        elif self.runtime_cost_ledger_hash is not None:
            raise ValueError("frozen family observation cannot claim runtime cost")
        payload = {
            "manifest_v3_hash": self.manifest_v3_hash,
            "proposal_hash": self.proposal_hash,
            "candidate_id": self.candidate_id,
            "planned_arm_id": self.planned_arm_id,
            "realization_receipt_hash": self.realization_receipt_hash,
            "bindings": [
                self.action_hash, self.control_hash, self.endpoint_hash,
                self.support_hash, self.realized_support_hash,
                self.interaction_hash, self.cost_hash,
            ],
            "source_hash": self.source_hash,
            "planner_allowlist_hash": self.planner_allowlist_hash,
            "assessor_allowlist_hash": self.assessor_allowlist_hash,
            "preaction_shared_feature_hash": self.preaction_shared_feature_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "provenance_hash": self.provenance_hash,
            "availability": self.availability.value,
            "availability_reason": self.availability_reason,
            "runtime_features": sorted(
                [row.name, row.value] for row in features
            ),
            "evidence_blocks": sorted([
                row.block.value, row.availability.value,
                row.source_hash, row.receipt_hash,
            ] for row in blocks),
            "evidence_source_hashes": list(sources),
            "observed_unit_ids": list(units),
            "atomically_observed": self.atomically_observed,
            "observation_source": self.observation_source.value,
            "runtime_cost_ledger_hash": self.runtime_cost_ledger_hash,
        }
        object.__setattr__(self, "observation_hash", _sealed(
            self.observation_hash, payload, "SelectorV7-v3 observation hash",
        ))


def calibration_choice_family_digest_v7_v3(
    manifest: CandidateManifestV7V3,
    proposal: PlannerProposalV7V3,
) -> str:
    """CAL-time family identity; deliberately excludes runtime observations."""
    return _sha256({
        "manifest_v3_hash": manifest.manifest_v3_hash,
        "candidate_family_hash": manifest.base_manifest.candidate_family_hash,
        "score_bank_universe_digest": proposal.score_bank.universe_digest,
        "score_bank_universe_cardinality": (
            proposal.score_bank.universe_cardinality
        ),
        "covered_planned_arm_ids": sorted(
            row.planned_arm_id for row in proposal.score_bank.scores
        ),
    })


@dataclass(frozen=True)
class CalibrationChoiceFamilyReceiptV7V3:
    manifest_v3_hash: str
    choice_family_digest: str
    choice_family_cardinality: int
    covered_planned_arm_ids: tuple[str, ...]
    score_bank_universe_digest: str
    calibration_fold_id: str
    calibration_data_hash: str
    assessor_model_hash: str
    simultaneous_correction_id: str
    frozen_before_evaluation_runtime: bool
    provenance_hash: str
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("CAL-family manifest hash", self.manifest_v3_hash),
            ("CAL-family digest", self.choice_family_digest),
            ("CAL-family score universe digest", self.score_bank_universe_digest),
            ("CAL-family fold id", self.calibration_fold_id),
            ("CAL-family data hash", self.calibration_data_hash),
            ("CAL-family assessor model hash", self.assessor_model_hash),
            ("CAL-family simultaneous correction id",
             self.simultaneous_correction_id),
            ("CAL-family provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        covered = _sorted_unique(
            self.covered_planned_arm_ids, "CAL-family planned-arm ids",
        )
        count = _integer(
            self.choice_family_cardinality, "CAL-family cardinality",
        )
        if count != len(covered):
            raise ValueError("CAL-family cardinality differs from coverage")
        if self.frozen_before_evaluation_runtime is not True:
            raise ValueError("CAL-family authority must predate evaluation runtime")
        payload = {
            "manifest_v3_hash": self.manifest_v3_hash,
            "choice_family_digest": self.choice_family_digest,
            "choice_family_cardinality": count,
            "covered_planned_arm_ids": list(covered),
            "score_bank_universe_digest": self.score_bank_universe_digest,
            "calibration_fold_id": self.calibration_fold_id,
            "calibration_data_hash": self.calibration_data_hash,
            "assessor_model_hash": self.assessor_model_hash,
            "simultaneous_correction_id": self.simultaneous_correction_id,
            "frozen_before_evaluation_runtime": (
                self.frozen_before_evaluation_runtime
            ),
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "covered_planned_arm_ids", covered)
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "CAL-time choice-family receipt hash",
        ))


@dataclass(frozen=True)
class CalibrationReceiptV7V3:
    manifest_v3_hash: str
    base_receipt: v1.CalibrationReceiptV7
    choice_family_receipt: CalibrationChoiceFamilyReceiptV7V3
    assessor_model_provenance_hash: str
    assessor_algorithm_id: str
    assessor_algorithm_version: str
    assessor_feature_schema_hash: str
    assessor_config_hash: str
    simultaneous_correction_id: str
    frozen_before_decision: bool
    calibration_v3_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("calibration manifest hash", self.manifest_v3_hash),
            ("calibration assessor model provenance hash",
             self.assessor_model_provenance_hash),
            ("calibration assessor algorithm id", self.assessor_algorithm_id),
            ("calibration assessor algorithm version",
             self.assessor_algorithm_version),
            ("calibration feature-schema hash",
             self.assessor_feature_schema_hash),
            ("calibration assessor config hash", self.assessor_config_hash),
            ("calibration simultaneous correction id",
             self.simultaneous_correction_id),
        ):
            _text(value, name)
        if not isinstance(self.base_receipt, v1.CalibrationReceiptV7):
            raise ValueError("v3 calibration needs typed v1 receipt")
        family = self.choice_family_receipt
        if not isinstance(family, CalibrationChoiceFamilyReceiptV7V3):
            raise ValueError("v3 calibration needs typed CAL-family authority")
        if family.assessor_model_hash != self.base_receipt.model_hash:
            raise ValueError("CAL-family model hash does not bind base receipt")
        if self.assessor_feature_schema_hash != (
            self.base_receipt.feature_schema_hash
        ):
            raise ValueError("calibration feature schema does not bind base receipt")
        if (set(family.covered_planned_arm_ids)
                != set(self.base_receipt.covered_planned_arm_ids)):
            raise ValueError("base and v3 calibration family coverage differ")
        if (family.calibration_fold_id != self.base_receipt.calibration_fold_id
                or family.calibration_data_hash
                != self.base_receipt.calibration_data_hash
                or family.simultaneous_correction_id
                != self.simultaneous_correction_id
                or family.manifest_v3_hash != self.manifest_v3_hash):
            raise ValueError("CAL-family authority differs from calibration")
        if self.frozen_before_decision is not True:
            raise ValueError("v3 calibration must be frozen before decision")
        payload = {
            "manifest_v3_hash": self.manifest_v3_hash,
            "base_receipt_hash": self.base_receipt.receipt_hash,
            "choice_family_receipt_hash": family.receipt_hash,
            "assessor_model_hash": self.base_receipt.model_hash,
            "assessor_model_provenance_hash": (
                self.assessor_model_provenance_hash
            ),
            "assessor_algorithm_id": self.assessor_algorithm_id,
            "assessor_algorithm_version": self.assessor_algorithm_version,
            "assessor_feature_schema_hash": self.assessor_feature_schema_hash,
            "assessor_config_hash": self.assessor_config_hash,
            "simultaneous_correction_id": self.simultaneous_correction_id,
            "frozen_before_decision": self.frozen_before_decision,
        }
        object.__setattr__(self, "calibration_v3_hash", _sealed(
            self.calibration_v3_hash, payload, "v3 calibration receipt hash",
        ))

    @property
    def assessor_model_hash(self) -> str:
        return self.base_receipt.model_hash

    @property
    def choice_family_digest(self) -> str:
        return self.choice_family_receipt.choice_family_digest

    @property
    def choice_family_cardinality(self) -> int:
        return self.choice_family_receipt.choice_family_cardinality

    @property
    def covered_planned_arm_ids(self) -> tuple[str, ...]:
        return self.choice_family_receipt.covered_planned_arm_ids


@dataclass(frozen=True)
class ActionRiskVectorV7V3:
    base_vector: v1.ActionRiskVectorV7
    assessor_model_hash: str
    assessor_model_provenance_hash: str
    assessor_algorithm_id: str
    assessor_algorithm_version: str
    assessor_feature_schema_hash: str
    assessor_config_hash: str
    calibration_v3_hash: str
    risk_vector_v3_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.base_vector, v1.ActionRiskVectorV7):
            raise ValueError("v2 risk vector needs typed v1 five-head vector")
        for name, value in (
            ("risk assessor model hash", self.assessor_model_hash),
            ("risk assessor model provenance hash",
             self.assessor_model_provenance_hash),
            ("risk assessor algorithm id", self.assessor_algorithm_id),
            ("risk assessor algorithm version", self.assessor_algorithm_version),
            ("risk assessor feature-schema hash",
             self.assessor_feature_schema_hash),
            ("risk assessor config hash", self.assessor_config_hash),
            ("risk v2 calibration hash", self.calibration_v3_hash),
        ):
            _text(value, name)
        if self.base_vector.assessor_hash != self.assessor_model_hash:
            raise ValueError("risk vector model hash differs from its assessor hash")
        if (self.base_vector.assessor_provenance_hash
                != self.assessor_model_provenance_hash):
            raise ValueError("risk vector model provenance binding differs")
        payload = {
            "base_risk_vector_hash": self.base_vector.risk_vector_hash,
            "assessor_model_hash": self.assessor_model_hash,
            "assessor_model_provenance_hash": (
                self.assessor_model_provenance_hash
            ),
            "assessor_algorithm_id": self.assessor_algorithm_id,
            "assessor_algorithm_version": self.assessor_algorithm_version,
            "assessor_feature_schema_hash": self.assessor_feature_schema_hash,
            "assessor_config_hash": self.assessor_config_hash,
            "calibration_v3_hash": self.calibration_v3_hash,
        }
        object.__setattr__(self, "risk_vector_v3_hash", _sealed(
            self.risk_vector_v3_hash, payload, "v2 risk-vector hash",
        ))


@dataclass(frozen=True)
class PortfolioDecisionV7V3:
    state: v1.PortfolioStateV7
    manifest_v3_hash: str
    proposal_hash: str
    calibration_v3_hash: str
    native_candidate_id: str
    selected_candidate_id: str | None
    safe_candidate_ids: tuple[str, ...]
    reason: DecisionReasonV7V3
    audit_count: int
    child_count: int
    decision_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.state, v1.PortfolioStateV7):
            raise ValueError("v2 portfolio state is untyped")
        if not isinstance(self.reason, DecisionReasonV7V3):
            raise ValueError("v2 decision reason is untyped")
        for name, value in (
            ("decision manifest hash", self.manifest_v3_hash),
            ("decision proposal hash", self.proposal_hash),
            ("decision calibration hash", self.calibration_v3_hash),
            ("decision native candidate id", self.native_candidate_id),
        ):
            _text(value, name)
        if self.audit_count not in {0, 1} or self.child_count not in {0, 1}:
            raise ValueError("v2 audit/child counts are bounded by one")
        if self.state is v1.PortfolioStateV7.COMMIT:
            if (not self.selected_candidate_id
                    or self.selected_candidate_id == self.native_candidate_id):
                raise ValueError("v2 commit needs one nonnative selection")
        elif self.state is v1.PortfolioStateV7.NATIVE:
            if self.selected_candidate_id != self.native_candidate_id:
                raise ValueError("v2 native decision must select canonical native")
        payload = {
            "state": self.state.value,
            "manifest_v3_hash": self.manifest_v3_hash,
            "proposal_hash": self.proposal_hash,
            "calibration_v3_hash": self.calibration_v3_hash,
            "native_candidate_id": self.native_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "safe_candidate_ids": list(self.safe_candidate_ids),
            "reason": self.reason.value,
            "audit_count": self.audit_count,
            "child_count": self.child_count,
        }
        object.__setattr__(self, "decision_hash", _sealed(
            self.decision_hash, payload, "v2 portfolio decision hash",
        ))


def _native_decision(
    manifest: object,
    proposal: object,
    calibration: object,
    reason: DecisionReasonV7V3,
    *,
    audit_count: int = 0,
    child_count: int = 0,
) -> PortfolioDecisionV7V3:
    try:
        manifest_hash = str(manifest.manifest_v3_hash)
        native_id = str(manifest.native.candidate_id)
    except Exception:
        manifest_hash = "unbound-v2-manifest"
        native_id = "action0:native"
    proposal_hash = str(getattr(proposal, "proposal_hash", "") or
                        "unbound-v2-proposal")
    calibration_hash = str(
        getattr(calibration, "calibration_v3_hash", "")
        or "unbound-v2-calibration"
    )
    return PortfolioDecisionV7V3(
        state=v1.PortfolioStateV7.NATIVE,
        manifest_v3_hash=manifest_hash,
        proposal_hash=proposal_hash,
        calibration_v3_hash=calibration_hash,
        native_candidate_id=native_id,
        selected_candidate_id=native_id,
        safe_candidate_ids=(),
        reason=reason,
        audit_count=audit_count if audit_count in {0, 1} else 1,
        child_count=child_count if child_count in {0, 1} else 1,
    )


def _validate_runtime_ledger(
    manifest: CandidateManifestV7V3,
    proposal: PlannerProposalV7V3,
    ledger: RuntimeCostLedgerV7V3,
    plan: AliasRunPlanV7V3,
    group_members: tuple[str, ...],
    by_realization: dict[str, v1.ArmRealizationReceiptV7],
    by_arm: dict[str, v1.CandidateArmV7],
    alias_by_id: dict[str, v1.ArmAliasReceiptV7],
) -> None:
    representative = plan.representative_planned_arm_id
    aliases = tuple(row for row in group_members if row != representative)
    if (ledger.manifest_v3_hash != manifest.manifest_v3_hash
            or ledger.proposal_hash != proposal.proposal_hash
            or ledger.alias_run_plan_hash != plan.plan_hash
            or ledger.runtime_run_id != plan.runtime_run_id
            or ledger.representative_planned_arm_id != representative
            or ledger.alias_planned_arm_ids != aliases
            or ledger.mode is not plan.mode):
        raise ValueError("runtime ledger and frozen alias run plan differ")
    receipts = tuple(ledger.execution_receipts)
    if any(row.started_at_ns < plan.earliest_run_start_ns for row in receipts):
        raise ValueError("runtime execution predates its frozen run plan")
    by_operation: dict[tuple[str, RuntimeOperationV7V3], list[
        RuntimeExecutionReceiptV7V3
    ]] = {}
    for receipt in receipts:
        by_operation.setdefault(
            (receipt.planned_arm_id, receipt.operation), [],
        ).append(receipt)
        realization = by_realization.get(receipt.planned_arm_id)
        if (realization is None
                or realization.status is not v1.ArmRealizationStatusV7.COMPLETE
                or receipt.realization_receipt_hash != realization.receipt_hash
                or receipt.executable_bytes_hash
                != realization.executable_bytes_hash):
            raise ValueError("runtime receipt does not bind exact realization")
        if receipt.operation is RuntimeOperationV7V3.EXECUTE_ARM:
            arm = by_arm[receipt.planned_arm_id]
            if (receipt.actual_prospective_runtime_cost
                    > arm.prospective_runtime_cost_ceiling + 1e-12
                    or not receipt.actual_prospective_runtime_cost_counters.within(
                        arm.prospective_runtime_cost_counter_ceiling
                    )):
                raise ValueError("runtime arm execution exceeded its ceiling")
        else:
            alias = alias_by_id.get(receipt.planned_arm_id)
            if (alias is None
                    or receipt.reused_from_planned_arm_id != representative
                    or receipt.actual_prospective_runtime_cost
                    > alias.prospective_runtime_verification_cost + 1e-12
                    or not receipt.actual_prospective_runtime_cost_counters.within(
                        alias.prospective_runtime_verification_cost_counters
                    )):
                raise ValueError("runtime alias verification exceeded its ceiling")
    execute_ids = {
        planned_id for (planned_id, operation), values in by_operation.items()
        if operation is RuntimeOperationV7V3.EXECUTE_ARM and len(values) == 1
    }
    verify_ids = {
        planned_id for (planned_id, operation), values in by_operation.items()
        if operation is RuntimeOperationV7V3.VERIFY_ALIAS_REUSE and len(values) == 1
    }
    if sum(len(values) for values in by_operation.values()) != len(by_operation):
        raise ValueError("runtime arm operation was charged more than once")
    if plan.mode is AliasChronologyModeV7V3.PRE_RUN_FROZEN_REUSE:
        seal = plan.pre_run_alias_seal
        if (seal is None
                or seal.sealed_at_ns >= min(row.started_at_ns for row in receipts)
                or execute_ids != {representative}
                or verify_ids != set(aliases)
                or ledger.post_hoc_discovered_at_ns is not None):
            raise ValueError("pre-run reuse chronology is invalid")
    elif plan.mode in {
        AliasChronologyModeV7V3.POST_HOC_DISCOVERY,
        AliasChronologyModeV7V3.SAME_RUN_DUPLICATE,
    }:
        if execute_ids != set(group_members) or verify_ids:
            raise ValueError("duplicate chronology must charge every executed arm")
        if plan.mode is AliasChronologyModeV7V3.POST_HOC_DISCOVERY:
            discovery = _integer(
                ledger.post_hoc_discovered_at_ns,
                "post-hoc alias discovery timestamp",
            )
            if discovery < max(row.finished_at_ns for row in receipts):
                raise ValueError("post-hoc alias discovery predates execution finish")
        elif ledger.post_hoc_discovered_at_ns is not None:
            raise ValueError("same-run duplicate cannot claim post-hoc discovery")
    else:
        if (execute_ids != {representative} or verify_ids
                or ledger.post_hoc_discovered_at_ns is not None):
            raise ValueError("direct chronology needs exactly one execution")


def select_portfolio_v7_v3(
    manifest: CandidateManifestV7V3,
    proposal: PlannerProposalV7V3,
    observations: Sequence[CandidateObservationV7V3],
    risks: Sequence[ActionRiskVectorV7V3],
    calibration: CalibrationReceiptV7V3,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    runtime_cost_ledgers: Sequence[RuntimeCostLedgerV7V3],
    *,
    audit_count: int = 0,
    child_count: int = 0,
) -> PortfolioDecisionV7V3:
    """Replay frozen authorities and inspect only selected top-K runtime rows."""
    if (isinstance(audit_count, bool) or not isinstance(audit_count, int)
            or audit_count < 0 or audit_count > 1):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.AUDIT_BUDGET_EXHAUSTED,
            audit_count=1, child_count=child_count,
        )
    if (isinstance(child_count, bool) or not isinstance(child_count, int)
            or child_count < 0 or child_count > 1):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.CHILD_BUDGET_EXHAUSTED,
            audit_count=audit_count, child_count=1,
        )
    try:
        selected_ids = set(proposal.selected_planned_arm_ids)
        selected_observations = tuple(
            row for row in observations
            if getattr(row, "planned_arm_id", None) in selected_ids
        )
        selected_risks = tuple(
            row for row in risks
            if getattr(getattr(row, "base_vector", None),
                       "planned_arm_id", None) in selected_ids
        )
        selected_ledgers = tuple(
            row for row in runtime_cost_ledgers
            if getattr(row, "representative_planned_arm_id", None)
            in selected_ids
        )
        objects = (
            manifest, proposal, calibration, proposal.score_bank,
            *proposal.score_bank.scores, *proposal.alias_run_plans,
            *realizations, *aliases, *selected_observations, *selected_risks,
            *selected_ledgers,
        )
    except Exception:
        return _native_decision(
            manifest, proposal, calibration, DecisionReasonV7V3.MALFORMED_INPUT,
            audit_count=audit_count, child_count=child_count,
        )
    if any(not _rehash_valid(row) for row in objects):
        return _native_decision(
            manifest, proposal, calibration, DecisionReasonV7V3.MALFORMED_INPUT,
            audit_count=audit_count, child_count=child_count,
        )
    try:
        _validate_canonical_native(
            manifest.base_manifest, manifest.immutable_native_receipt,
        )
        freeze = manifest.freeze_receipt
        if (manifest.base_manifest.frozen_before_outcome is not True
                or freeze.base_manifest_hash
                != manifest.base_manifest.manifest_hash
                or freeze.candidate_family_hash
                != manifest.base_manifest.candidate_family_hash
                or freeze.source_hash != manifest.base_manifest.source_hash
                or freeze.target_fold_role is not manifest.base_manifest.fold_role
                or freeze.outcome_capability_absent is not True):
            raise ValueError("manifest freeze authority drifted")
    except Exception:
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.NATIVE_IDENTITY_INVALID,
            audit_count=audit_count, child_count=child_count,
        )
    authority = manifest.planner_model_receipt
    if (proposal.score_bank.planner_model_receipt.receipt_hash
            != authority.receipt_hash
            or proposal.score_bank.planner_model_receipt.model_hash
            != authority.model_hash
            or proposal.score_bank.planner_allowlist_hash
            != authority.feature_registry.planner_allowlist_hash
            or proposal.score_bank.planner_allowlist_hash
            != manifest.base_manifest.planner_allowlist_hash
            or proposal.planner_algorithm_id != authority.algorithm_id
            or proposal.planner_algorithm_version != authority.algorithm_version
            or proposal.planner_config_hash != authority.config_hash
            or proposal.planner_provenance_hash
            != authority.model_provenance_hash
            or authority.fit_receipt.target_fold_role
            is not manifest.base_manifest.fold_role):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.PLANNER_AUTHORITY_MISMATCH,
            audit_count=audit_count, child_count=child_count,
        )
    try:
        replayed = propose_candidates_v7_v3(
            manifest, proposal.score_bank.scores, realizations, aliases,
            proposal.alias_run_plans, top_k=proposal.top_k,
            maximum_prospective_runtime_cost=(
                proposal.maximum_prospective_runtime_cost
            ),
            maximum_prospective_runtime_cost_counters=(
                proposal.maximum_prospective_runtime_cost_counters
            ),
            planner_algorithm_id=proposal.planner_algorithm_id,
            planner_algorithm_version=proposal.planner_algorithm_version,
            planner_config_hash=proposal.planner_config_hash,
            planner_provenance_hash=proposal.planner_provenance_hash,
        )
    except ValueError as error:
        reason = (
            DecisionReasonV7V3.SCORE_BANK_INCOMPLETE
            if "score bank" in str(error) else
            DecisionReasonV7V3.PLANNER_AUTHORITY_MISMATCH
            if "planner" in str(error) and "author" in str(error) else
            DecisionReasonV7V3.PROPOSAL_REPLAY_MISMATCH
        )
        return _native_decision(
            manifest, proposal, calibration, reason,
            audit_count=audit_count, child_count=child_count,
        )
    if replayed.proposal_hash != proposal.proposal_hash:
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.PROPOSAL_REPLAY_MISMATCH,
            audit_count=audit_count, child_count=child_count,
        )
    try:
        by_realization, _, groups, alias_by_id = _realization_state(
            manifest, realizations, aliases,
        )
        universe, group_costs = _proposal_context(
            manifest, realizations, aliases, proposal.alias_run_plans,
            proposal.maximum_prospective_runtime_cost,
            proposal.maximum_prospective_runtime_cost_counters,
        )
    except Exception:
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.PROPOSAL_REPLAY_MISMATCH,
            audit_count=audit_count, child_count=child_count,
        )
    by_arm, _ = _manifest_arm_maps(manifest)
    universe_ids = {row.planned_arm_id for row in universe}
    observation_by_id = {
        row.planned_arm_id: row for row in selected_observations
    }
    risk_by_id = {
        row.base_vector.planned_arm_id: row for row in selected_risks
    }
    if (len(observation_by_id) != len(selected_observations)
            or len(risk_by_id) != len(selected_risks)
            or set(observation_by_id) != selected_ids
            or set(risk_by_id) != selected_ids):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.FEATURE_INELIGIBLE,
            audit_count=audit_count, child_count=child_count,
        )
    expected_family_digest = calibration_choice_family_digest_v7_v3(
        manifest, proposal,
    )
    base_calibration = calibration.base_receipt
    family = calibration.choice_family_receipt
    if (calibration.manifest_v3_hash != manifest.manifest_v3_hash
            or base_calibration.manifest_hash
            != manifest.base_manifest.manifest_hash
            or base_calibration.candidate_family_hash
            != manifest.base_manifest.candidate_family_hash
            or base_calibration.target_fold_role
            is not manifest.base_manifest.fold_role
            or base_calibration.planner_allowlist_hash
            != manifest.base_manifest.planner_allowlist_hash
            or base_calibration.assessor_allowlist_hash
            != manifest.base_manifest.assessor_allowlist_hash
            or base_calibration.assessor_feature_profile
            != v1.ASSESSOR_FEATURE_PROFILE_V7
            or tuple(base_calibration.head_names)
            != v1.PRIMARY_ASSESSOR_HEADS_V7
            or base_calibration.severe_estimand
            is not v1.SevereEstimandV7.EXECUTION_UNIT_ANY_ROW_SEVERE
            or base_calibration.selection_method
            is not v1.CalibrationMethodV7.COMPONENT_MAX_RESIDUAL
            or base_calibration.frozen_before_decision is not True
            or base_calibration.model_hash != calibration.assessor_model_hash
            or base_calibration.feature_schema_hash
            != calibration.assessor_feature_schema_hash
            or family.manifest_v3_hash != manifest.manifest_v3_hash
            or calibration.choice_family_digest != expected_family_digest
            or calibration.choice_family_cardinality != len(universe_ids)
            or set(calibration.covered_planned_arm_ids) != universe_ids
            or set(base_calibration.covered_planned_arm_ids) != universe_ids
            or family.score_bank_universe_digest
            != proposal.score_bank.universe_digest
            or family.calibration_fold_id
            != base_calibration.calibration_fold_id
            or family.calibration_data_hash
            != base_calibration.calibration_data_hash
            or family.assessor_model_hash != base_calibration.model_hash
            or family.simultaneous_correction_id
            != calibration.simultaneous_correction_id
            or family.frozen_before_evaluation_runtime is not True):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.CALIBRATION_FAMILY_INCOMPLETE,
            audit_count=audit_count, child_count=child_count,
        )
    if (base_calibration.availability is not v1.AvailabilityV7.AVAILABLE
            or not base_calibration.frozen_before_decision):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.ASSESSOR_CALIBRATION_MISMATCH,
            audit_count=audit_count, child_count=child_count,
        )
    ledgers = {
        row.representative_planned_arm_id: row
        for row in selected_ledgers
    }
    if (len(ledgers) != len(selected_ledgers)
            or set(ledgers) != selected_ids):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.ALIAS_CHRONOLOGY_INVALID,
            audit_count=audit_count, child_count=child_count,
        )
    plans = {
        row.representative_planned_arm_id: row
        for row in proposal.alias_run_plans
    }
    try:
        for planned_id, ledger in ledgers.items():
            _validate_runtime_ledger(
                manifest, proposal, ledger, plans[planned_id],
                groups[planned_id], by_realization, by_arm, alias_by_id,
            )
    except Exception:
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.ALIAS_CHRONOLOGY_INVALID,
            audit_count=audit_count, child_count=child_count,
        )
    try:
        actual_scalar = sum(
            row.charged_prospective_runtime_cost for row in ledgers.values()
        )
        actual_counters = _sum_counters(
            tuple(row.charged_prospective_runtime_cost_counters
                  for row in ledgers.values()),
            cache_state=proposal.maximum_prospective_runtime_cost_counters.cache_state,
        )
    except Exception:
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.COST_BUDGET_EXCEEDED,
            audit_count=audit_count, child_count=child_count,
        )
    if (actual_scalar > proposal.maximum_prospective_runtime_cost + 1e-12
            or not actual_counters.within(
                proposal.maximum_prospective_runtime_cost_counters
            ) or proposal.maximum_prospective_runtime_cost
            > float(calibration.base_receipt.maximum_prospective_runtime_cost)
            + 1e-12
            or not proposal.maximum_prospective_runtime_cost_counters.within(
                calibration.base_receipt.maximum_prospective_runtime_cost_counters
            )):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.COST_BUDGET_EXCEEDED,
            audit_count=audit_count, child_count=child_count,
        )
    score_by_id = {
        row.planned_arm_id: row for row in proposal.score_bank.scores
    }
    for planned_id in proposal.selected_planned_arm_ids:
        arm = by_arm[planned_id]
        observation = observation_by_id[planned_id]
        risk = risk_by_id[planned_id]
        base_risk = risk.base_vector
        realization = by_realization[planned_id]
        if base_risk.availability is not v1.AvailabilityV7.AVAILABLE:
            return _native_decision(
                manifest, proposal, calibration,
                DecisionReasonV7V3.FEATURE_INELIGIBLE,
                audit_count=audit_count, child_count=child_count,
            )
        if (observation.manifest_v3_hash != manifest.manifest_v3_hash
                or observation.proposal_hash != proposal.proposal_hash
                or observation.candidate_id != arm.candidate_id
                or observation.realization_receipt_hash != realization.receipt_hash
                or observation.realized_support_hash
                != realization.realized_support_hash
                or observation.action_hash != arm.action_hash
                or observation.control_hash != arm.control_hash
                or observation.endpoint_hash != arm.endpoint_hash
                or observation.support_hash != arm.support_hash
                or observation.interaction_hash != arm.interaction_hash
                or observation.cost_hash != arm.cost_hash
                or observation.source_hash != manifest.base_manifest.source_hash
                or observation.planner_allowlist_hash
                != manifest.base_manifest.planner_allowlist_hash
                or observation.assessor_allowlist_hash
                != manifest.base_manifest.assessor_allowlist_hash
                or observation.preaction_shared_feature_hash
                != score_by_id[planned_id].before_feature_hash
                or observation.observation_source is not
                ObservationSourceV7V3.PROSPECTIVE_RUNTIME_OBSERVATION
                or observation.runtime_cost_ledger_hash
                != ledgers[planned_id].ledger_hash):
            return _native_decision(
                manifest, proposal, calibration,
                DecisionReasonV7V3.ACTION_BINDING_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
        if (observation.availability is not v1.AvailabilityV7.AVAILABLE
                or any(row.availability is not v1.AvailabilityV7.AVAILABLE
                       for row in observation.evidence_blocks)):
            return _native_decision(
                manifest, proposal, calibration,
                DecisionReasonV7V3.FEATURE_INELIGIBLE,
                audit_count=audit_count, child_count=child_count,
            )
        if arm.scope is v1.CandidateScopeV7.CASE_ATOMIC:
            if (not observation.atomically_observed
                    or set(observation.observed_unit_ids) != set(arm.unit_ids)):
                return _native_decision(
                    manifest, proposal, calibration,
                    DecisionReasonV7V3.GLOBAL_PARTIAL_COMMIT,
                    audit_count=audit_count, child_count=child_count,
                )
        elif (arm.scope is not v1.CandidateScopeV7.CERTIFIED_LOCAL
              or len(arm.unit_ids) != 1
              or set(observation.observed_unit_ids) != set(arm.unit_ids)):
            return _native_decision(
                manifest, proposal, calibration,
                DecisionReasonV7V3.LOCAL_SUPPORT_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
        model_binding = (
            base_risk.assessor_hash == calibration.assessor_model_hash
            and risk.assessor_model_hash == calibration.assessor_model_hash
            and base_risk.assessor_provenance_hash
            == calibration.assessor_model_provenance_hash
            and risk.assessor_model_provenance_hash
            == calibration.assessor_model_provenance_hash
            and risk.assessor_algorithm_id
            == calibration.assessor_algorithm_id
            and risk.assessor_algorithm_version
            == calibration.assessor_algorithm_version
            and risk.assessor_feature_schema_hash
            == calibration.assessor_feature_schema_hash
            and risk.assessor_config_hash == calibration.assessor_config_hash
            and base_risk.calibration_receipt_hash
            == calibration.calibration_v3_hash
            and risk.calibration_v3_hash == calibration.calibration_v3_hash
            and observation.feature_schema_hash
            == calibration.assessor_feature_schema_hash
            and base_risk.assessor_allowlist_hash
            == manifest.base_manifest.assessor_allowlist_hash
            and base_risk.assessor_allowlist_hash
            == base_calibration.assessor_allowlist_hash
            and base_risk.assessor_feature_profile
            == v1.ASSESSOR_FEATURE_PROFILE_V7
        )
        if not model_binding:
            return _native_decision(
                manifest, proposal, calibration,
                DecisionReasonV7V3.ASSESSOR_CALIBRATION_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
        if (base_risk.manifest_hash != manifest.base_manifest.manifest_hash
                or base_risk.proposal_hash != proposal.proposal_hash
                or base_risk.observation_hash != observation.observation_hash
                or base_risk.candidate_id != arm.candidate_id
                or base_risk.action_hash != arm.action_hash
                or base_risk.control_hash != arm.control_hash
                or base_risk.endpoint_hash != arm.endpoint_hash
                or base_risk.support_hash != arm.support_hash
                or base_risk.interaction_hash != arm.interaction_hash
                or base_risk.cost_hash != arm.cost_hash):
            return _native_decision(
                manifest, proposal, calibration,
                DecisionReasonV7V3.ACTION_BINDING_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
    if not selected_ids:
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V3.NO_SAFE_CANDIDATE,
            audit_count=audit_count, child_count=child_count,
        )
    safe: list[tuple[v1.CandidateArmV7, v1.ActionRiskVectorV7, float]] = []
    risk_rejected = False
    utility_rejected = False
    for planned_id in proposal.selected_planned_arm_ids:
        arm = by_arm[planned_id]
        risk = risk_by_id[planned_id].base_vector
        if (not risk.feature_eligible or not risk.calibration_valid
                or not risk.bound_bookkeeping_valid):
            return _native_decision(
                manifest, proposal, calibration,
                DecisionReasonV7V3.FEATURE_INELIGIBLE,
                audit_count=audit_count, child_count=child_count,
            )
        separate_risk_pass = (
            float(risk.any_row_severe_probability_upper)
            <= float(calibration.base_receipt.rho_any_row_severe)
            and float(risk.harm_upper)
            <= float(calibration.base_receipt.maximum_harm)
            and float(risk.harmed_pixel_fraction_upper)
            <= float(calibration.base_receipt.maximum_harmed_fraction)
            and float(risk.pixel_harm_cvar95_upper)
            <= float(calibration.base_receipt.maximum_cvar95)
        )
        objective = (
            float(risk.net_gain_lower)
            - v1.LAMBDA_COST_V7 * group_costs[planned_id][0]
        )
        if separate_risk_pass and objective > 0.0:
            safe.append((arm, risk, objective))
        else:
            risk_rejected = risk_rejected or not separate_risk_pass
            utility_rejected = utility_rejected or objective <= 0.0
    safe.sort(key=lambda item: (
        -item[2], float(item[1].any_row_severe_probability_upper),
        float(item[1].pixel_harm_cvar95_upper),
        group_costs[item[0].planned_arm_id][0],
        item[0].action_identity, item[0].planned_arm_id,
    ))
    if not safe:
        reason = (
            DecisionReasonV7V3.RISK_BUDGET_EXCEEDED if risk_rejected
            else DecisionReasonV7V3.NONPOSITIVE_CONSERVATIVE_UTILITY
            if utility_rejected else DecisionReasonV7V3.NO_SAFE_CANDIDATE
        )
        return _native_decision(
            manifest, proposal, calibration, reason,
            audit_count=audit_count, child_count=child_count,
        )
    selected = safe[0][0]
    return PortfolioDecisionV7V3(
        state=v1.PortfolioStateV7.COMMIT,
        manifest_v3_hash=manifest.manifest_v3_hash,
        proposal_hash=proposal.proposal_hash,
        calibration_v3_hash=calibration.calibration_v3_hash,
        native_candidate_id=manifest.native.candidate_id,
        selected_candidate_id=selected.candidate_id,
        safe_candidate_ids=tuple(row[0].candidate_id for row in safe),
        reason=DecisionReasonV7V3.SAFE_SET_MAXIMUM,
        audit_count=audit_count,
        child_count=child_count,
    )


def selector_v7_v3_schema_fingerprint() -> str:
    classes = (
        SelectorV7V3Disposition, PlannerFitReceiptV7V3,
        PlannerModelReceiptV7V3, ImmutableNativeReceiptV7V3,
        ManifestFreezeReceiptV7V3, CalibrationChoiceFamilyReceiptV7V3,
        CandidateManifestV7V3, PreRunAliasSealV7V3, AliasRunPlanV7V3,
        RuntimeExecutionReceiptV7V3, RuntimeCostLedgerV7V3,
        PlannerScoreReceiptV7V3, PlannerScoreBankV7V3, PlannerProposalV7V3,
        CandidateObservationV7V3, CalibrationReceiptV7V3,
        ActionRiskVectorV7V3, PortfolioDecisionV7V3,
    )
    enums = (
        DecisionReasonV7V3, AliasChronologyModeV7V3,
        RuntimeOperationV7V3, ObservationSourceV7V3,
    )
    return _sha256({
        "schema_version": SELECTOR_V7_V3_SCHEMA_VERSION,
        "canonicalization_version": SELECTOR_V7_V3_CANONICALIZATION_VERSION,
        "predecessor_failed_review_sha256": (
            SELECTOR_V7_V2_REVIEWED_SHA256
        ),
        "sr1_root_review_v1_sha256": SR1_ROOT_REVIEW_V1_SHA256,
        "classes": {
            cls.__name__: list(cls.__dataclass_fields__) for cls in classes
        },
        "enums": {
            enum.__name__: [[row.name, row.value] for row in enum]
            for enum in enums
        },
        "base_v1_schema_fingerprint": v1.selector_v7_schema_fingerprint(),
        "base_v2_schema_fingerprint": v2.selector_v7_v2_schema_fingerprint(),
    })


__all__ = [
    "ActionRiskVectorV7V3",
    "AliasChronologyModeV7V3",
    "AliasRunPlanV7V3",
    "CalibrationChoiceFamilyReceiptV7V3",
    "CalibrationReceiptV7V3",
    "CandidateManifestV7V3",
    "CandidateObservationV7V3",
    "DecisionReasonV7V3",
    "ImmutableNativeReceiptV7V3",
    "ManifestFreezeReceiptV7V3",
    "ObservationSourceV7V3",
    "PlannerProposalV7V3",
    "PlannerFitReceiptV7V3",
    "PlannerModelReceiptV7V3",
    "PlannerScoreBankV7V3",
    "PlannerScoreReceiptV7V3",
    "PortfolioDecisionV7V3",
    "PreRunAliasSealV7V3",
    "RuntimeCostLedgerV7V3",
    "RuntimeExecutionReceiptV7V3",
    "RuntimeOperationV7V3",
    "SELECTOR_V7_V1_FAILED_REVIEW_SHA256",
    "SELECTOR_V7_V1_FAILED_TEST_SHA256",
    "SELECTOR_V7_V2_REVIEWED_SHA256",
    "SELECTOR_V7_V2_REVIEWED_TEST_SHA256",
    "SELECTOR_V7_V3_CANONICALIZATION_VERSION",
    "SELECTOR_V7_V3_SCHEMA_VERSION",
    "SR1_ROOT_REVIEW_V1_SHA256",
    "PLANNER_ALGORITHM_ID_V7V3",
    "PLANNER_ALGORITHM_VERSION_V7V3",
    "PLANNER_CONFIG_HASH_V7V3",
    "PLANNER_TARGET_ID_V7V3",
    "SelectorV7V3Disposition",
    "calibration_choice_family_digest_v7_v3",
    "propose_candidates_v7_v3",
    "select_portfolio_v7_v3",
    "selector_v7_v3_schema_fingerprint",
]
