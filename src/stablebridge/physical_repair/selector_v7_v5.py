"""Append-only SR1 SelectorV7-v5 contract successor.

The reviewed v1 module remains byte-for-byte available as ``selector_v7``.
This successor closes SR1 root-review findings R01--R14 without reading an
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
from . import selector_v7_v3 as v3
from . import selector_v7_v4 as v4


SELECTOR_V7_V5_SCHEMA_VERSION = "selector-v7-contract/v4.39-sr1-v5-draft"
SELECTOR_V7_V5_CANONICALIZATION_VERSION = "selector-v7-v5-canonical-json/v1"
SELECTOR_V7_V4_REVIEWED_SHA256 = (
    "fb8bb892f07f9e735646d45b489c895c31c2fbb200f1650369fd7f5ee94a155e"
)
SELECTOR_V7_V4_REVIEWED_TEST_SHA256 = (
    "07ade36dd674299200990704090ca0c456213fff9dc913511636590b49e0516f"
)
SR1_ROOT_REVIEW_V4_SHA256 = (
    "7a3037bcb2fc7f28d6ccf4b3f05fb85640f40e7a9f0e43e2f8e07e03fa96d36c"
)
SR1_ROOT_REVIEW_V4_STATUS = (
    "FAIL_SR1_V4_TEST_ONLY_PROVENANCE_AND_EXACT_FOLD_BYPASSES_NO_SEAL"
)
SELECTOR_V7_V3_REVIEWED_SHA256 = (
    "abe48316abcfb4f1a5f3a1386d4b435323adc36198a167beb7b0043455caf6db"
)
SELECTOR_V7_V3_REVIEWED_TEST_SHA256 = (
    "22311a6ed2570a49f8df33f1f4840e0138662dec65ab924c89e44824efbb661b"
)
SR1_ROOT_REVIEW_V3_SHA256 = (
    "95658a24d4b72d120562e94871757ca83007e7d3f444878868863f02a141a945"
)
SR1_ROOT_REVIEW_V3_STATUS = (
    "FAIL_SR1_V3_PLANNER_AUTHORITY_AND_BENEFIT_PRESERVATION_BYPASSES_NO_SEAL"
)
SR1_ROOT_REVIEW_V2_SHA256 = (
    "048513754c61dfd278020fdae3a104cf8af38ce78bde1097107227906e2f04ae"
)
SR1_ROOT_REVIEW_V2_STATUS = "FAIL_SR1_V2_NEW_CONTRACT_BYPASSES_NO_SEAL"
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
PLANNER_ALGORITHM_ID_V7V5 = "group-weighted-ridge"
PLANNER_ALGORITHM_VERSION_V7V5 = "v1"
PLANNER_TARGET_ID_V7V5 = "G_hat_equals_B_hat_minus_H_hat"
PLANNER_CONFIG_HASH_V7V5 = hashlib.sha256(
    b'{"algorithm":"group-weighted-ridge","l2":1.0,"standardization":"fit-mean-std"}'
).hexdigest()
PLANNER_SCHEMA_PRODUCER_CODE_HASH_V7V5 = hashlib.sha256(
    b"selector-v7-v5-canonical-planner-schema-producer/v1"
).hexdigest()
PLANNER_SCHEMA_CONFIG_HASH_V7V5 = hashlib.sha256(
    b"selector-v7-v5-preaction-planner-schema-config/v1"
).hexdigest()
PLANNER_MODEL_SCHEMA_HASH_V7V5 = hashlib.sha256(
    b"selector-v7-v5-group-weighted-ridge-model-schema/v1"
).hexdigest()
E235_OUTPUT_REPLAY_SCHEMA_HASH_V7V5 = hashlib.sha256(
    b"selector-v7-v5-e235-output-replay-schema/v1"
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


class DecisionReasonV7V5(str, Enum):
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


class AliasChronologyModeV7V5(str, Enum):
    DIRECT_NO_ALIAS = "direct_no_alias"
    PRE_RUN_FROZEN_REUSE = "pre_run_frozen_reuse"
    POST_HOC_DISCOVERY = "post_hoc_discovery"
    SAME_RUN_DUPLICATE = "same_run_duplicate"


class RuntimeOperationV7V5(str, Enum):
    EXECUTE_ARM = "execute_arm"
    VERIFY_ALIAS_REUSE = "verify_alias_reuse"


class ObservationSourceV7V5(str, Enum):
    FROZEN_FAMILY_OBSERVATION = "frozen_family_observation"
    PROSPECTIVE_RUNTIME_OBSERVATION = "prospective_runtime_observation"


class ExternalReplayAuthorityModeV7V5(str, Enum):
    PRODUCTION_TRUSTED = "production_trusted"
    TEST_ONLY = "test_only"
    UNVERIFIED = "unverified"


class ProductionEligibilityV7V5(str, Enum):
    PRODUCTION_ELIGIBLE = "production_eligible"
    TEST_ONLY_NONAUTHORITY = "test_only_nonauthority"
    UNVERIFIED_FAIL_CLOSED = "unverified_fail_closed"


class DecisionStateV7V5(str, Enum):
    COMMIT = "commit"
    AUDIT = "audit"
    NATIVE = "native"
    TEST_ONLY_WOULD_COMMIT = "test_only_would_commit"
    TEST_ONLY_WOULD_NATIVE = "test_only_would_native"


@dataclass(frozen=True)
class SelectorV7V5Disposition:
    predecessor_schema: str = v4.SELECTOR_V7_V4_SCHEMA_VERSION
    predecessor_sha256: str = SELECTOR_V7_V4_REVIEWED_SHA256
    predecessor_test_sha256: str = SELECTOR_V7_V4_REVIEWED_TEST_SHA256
    predecessor_review_sha256: str = SR1_ROOT_REVIEW_V4_SHA256
    predecessor_review_status: str = SR1_ROOT_REVIEW_V4_STATUS
    v2_review_sha256: str = SR1_ROOT_REVIEW_V2_SHA256
    v2_review_status: str = SR1_ROOT_REVIEW_V2_STATUS
    successor_schema: str = SELECTOR_V7_V5_SCHEMA_VERSION
    authority: str = "DRAFT_SYNTHETIC_ONLY_NO_AUTHORITY_SEAL"


@dataclass(frozen=True)
class FiveFoldPartitionV7V5:
    fold_universe_ids: tuple[str, ...]
    target_fold_id: str
    calibration_fold_id: str
    fit_fold_ids: tuple[str, ...]
    evaluation_fold_id: str
    partition_hash: str = ""

    def __post_init__(self) -> None:
        universe = _sorted_unique(
            self.fold_universe_ids, "partition five-fold universe",
        )
        fit = _sorted_unique(self.fit_fold_ids, "partition FIT folds")
        _text(self.target_fold_id, "partition target fold")
        _text(self.calibration_fold_id, "partition CAL fold")
        _text(self.evaluation_fold_id, "partition EVAL fold")
        if len(universe) != 5 or len(fit) != 3:
            raise ValueError("partition requires exact five-fold 3/1/1 roles")
        if self.evaluation_fold_id != self.target_fold_id:
            raise ValueError("partition EVAL fold must equal its target fold")
        if (self.calibration_fold_id == self.target_fold_id
                or set(fit) | {
                    self.calibration_fold_id, self.evaluation_fold_id,
                } != set(universe)
                or set(fit) & {
                    self.calibration_fold_id, self.evaluation_fold_id,
                }):
            raise ValueError("partition roles do not exactly cover five folds")
        payload = {
            "fold_universe_ids": list(universe),
            "target_fold_id": self.target_fold_id,
            "calibration_fold_id": self.calibration_fold_id,
            "fit_fold_ids": list(fit),
            "evaluation_fold_id": self.evaluation_fold_id,
        }
        object.__setattr__(self, "fold_universe_ids", universe)
        object.__setattr__(self, "fit_fold_ids", fit)
        object.__setattr__(self, "partition_hash", _sealed(
            self.partition_hash, payload, "five-fold partition hash",
        ))


@dataclass(frozen=True)
class FiveFoldSplitAuthorityRootV7V5:
    sr0a_manifest_receipt_hash: str
    fold_universe_ids: tuple[str, ...]
    partitions: tuple[FiveFoldPartitionV7V5, ...]
    split_policy_hash: str
    frozen_before_fit: bool
    provenance_hash: str
    fold_universe_hash: str = ""
    root_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("split-root SR0A manifest receipt hash",
             self.sr0a_manifest_receipt_hash),
            ("split-root policy hash", self.split_policy_hash),
            ("split-root provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        universe = _sorted_unique(
            self.fold_universe_ids, "split-root five-fold universe",
        )
        partitions = tuple(sorted(
            self.partitions, key=lambda row: row.target_fold_id,
        ))
        if (len(universe) != 5 or len(partitions) != 5
                or any(not isinstance(row, FiveFoldPartitionV7V5)
                       for row in partitions)):
            raise ValueError("split root needs five typed target partitions")
        if ({row.target_fold_id for row in partitions} != set(universe)
                or {row.calibration_fold_id for row in partitions}
                != set(universe)
                or any(row.fold_universe_ids != universe for row in partitions)
                or len({row.partition_hash for row in partitions}) != 5):
            raise ValueError("split root does not cover every target/CAL fold once")
        if self.frozen_before_fit is not True:
            raise ValueError("split root must be frozen before FIT")
        universe_hash = _sealed(
            self.fold_universe_hash,
            {"fold_universe_ids": list(universe)},
            "split-root five-fold universe hash",
        )
        payload = {
            "sr0a_manifest_receipt_hash": self.sr0a_manifest_receipt_hash,
            "fold_universe_ids": list(universe),
            "fold_universe_hash": universe_hash,
            "partition_hashes": [row.partition_hash for row in partitions],
            "split_policy_hash": self.split_policy_hash,
            "frozen_before_fit": self.frozen_before_fit,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "fold_universe_ids", universe)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "fold_universe_hash", universe_hash)
        object.__setattr__(self, "root_hash", _sealed(
            self.root_hash, payload, "five-fold split-authority root hash",
        ))


@dataclass(frozen=True)
class SplitAuthorityReceiptV7V5:
    outer_split_id: str
    authority_root_receipt: FiveFoldSplitAuthorityRootV7V5
    fold_universe_ids: tuple[str, ...]
    target_fold_id: str
    calibration_fold_id: str
    fit_fold_ids: tuple[str, ...]
    calibration_fold_ids: tuple[str, ...]
    evaluation_fold_ids: tuple[str, ...]
    sr0a_manifest_receipt_hash: str
    split_authority_root_hash: str
    partition_root_hash: str
    assignment_hash: str
    split_policy_hash: str
    frozen_before_fit: bool
    provenance_hash: str
    fold_universe_hash: str = ""
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("outer split id", self.outer_split_id),
            ("split target fold id", self.target_fold_id),
            ("split calibration fold id", self.calibration_fold_id),
            ("split SR0A manifest receipt hash",
             self.sr0a_manifest_receipt_hash),
            ("split authority root hash", self.split_authority_root_hash),
            ("split partition root hash", self.partition_root_hash),
            ("split assignment hash", self.assignment_hash),
            ("split policy hash", self.split_policy_hash),
            ("split provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        if not isinstance(
            self.authority_root_receipt, FiveFoldSplitAuthorityRootV7V5,
        ):
            raise ValueError("split receipt needs typed five-fold authority root")
        universe = _sorted_unique(
            self.fold_universe_ids, "five-fold universe ids",
        )
        fit = _sorted_unique(self.fit_fold_ids, "FIT fold ids")
        cal = _sorted_unique(self.calibration_fold_ids, "CAL fold ids")
        evaluation = _sorted_unique(self.evaluation_fold_ids, "EVAL fold ids")
        if (len(universe) != 5 or len(fit) != 3
                or len(cal) != 1 or len(evaluation) != 1):
            raise ValueError(
                "split needs exact five-fold universe and 3 FIT/1 CAL/1 EVAL"
            )
        if set(fit) & set(cal) or set(fit) & set(evaluation) or set(cal) & set(evaluation):
            raise ValueError("FIT/CAL/EVAL fold ids must be pairwise disjoint")
        if (self.target_fold_id not in evaluation
                or self.calibration_fold_id not in cal):
            raise ValueError("target/CAL ids must belong to exact split families")
        if set(fit) | set(cal) | set(evaluation) != set(universe):
            raise ValueError("FIT/CAL/EVAL must exactly partition five-fold universe")
        root = self.authority_root_receipt
        partition = next(
            (row for row in root.partitions
             if row.target_fold_id == self.target_fold_id),
            None,
        )
        if (root.fold_universe_ids != universe
                or root.sr0a_manifest_receipt_hash
                != self.sr0a_manifest_receipt_hash
                or root.root_hash != self.split_authority_root_hash
                or root.split_policy_hash != self.split_policy_hash
                or partition is None
                or partition.calibration_fold_id != self.calibration_fold_id
                or partition.fit_fold_ids != fit
                or partition.evaluation_fold_id != evaluation[0]
                or partition.partition_hash != self.partition_root_hash
                or partition.partition_hash != self.assignment_hash):
            raise ValueError("split receipt differs from exact five-fold authority root")
        if self.frozen_before_fit is not True:
            raise ValueError("split authority must be frozen before FIT")
        universe_hash = _sealed(
            self.fold_universe_hash,
            {"fold_universe_ids": list(universe)},
            "five-fold universe hash",
        )
        payload = {
            "outer_split_id": self.outer_split_id,
            "authority_root_receipt_hash": root.root_hash,
            "fold_universe_ids": list(universe),
            "fold_universe_hash": universe_hash,
            "target_fold_id": self.target_fold_id,
            "calibration_fold_id": self.calibration_fold_id,
            "fit_fold_ids": list(fit),
            "calibration_fold_ids": list(cal),
            "evaluation_fold_ids": list(evaluation),
            "sr0a_manifest_receipt_hash": self.sr0a_manifest_receipt_hash,
            "split_authority_root_hash": self.split_authority_root_hash,
            "partition_root_hash": self.partition_root_hash,
            "assignment_hash": self.assignment_hash,
            "split_policy_hash": self.split_policy_hash,
            "frozen_before_fit": self.frozen_before_fit,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "fold_universe_ids", universe)
        object.__setattr__(self, "fit_fold_ids", fit)
        object.__setattr__(self, "calibration_fold_ids", cal)
        object.__setattr__(self, "evaluation_fold_ids", evaluation)
        object.__setattr__(self, "fold_universe_hash", universe_hash)
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "split authority receipt hash",
        ))


def _planner_feature_schema_payload_v7_v5(
    registry: v1.FeatureAllowlistsV7,
) -> dict[str, object]:
    fields = tuple(sorted(
        registry.planner_fields,
        key=lambda row: (row.canonical_field_id, row.name),
    ))
    return {
        "planner_profile": registry.planner_profile,
        "planner_allowlist_hash": registry.planner_allowlist_hash,
        "feature_registry_fingerprint": registry.combined_fingerprint,
        "fields": [
            {
                "canonical_field_id": row.canonical_field_id,
                "name": row.name,
                "stage": row.stage.value,
                "available_at_stage": row.available_at_stage.value,
                "source_hash": row.source_hash,
                "dependency_kind": row.dependency_kind.value,
                "dependency_hash": row.dependency_hash,
            }
            for row in fields
        ],
    }


@dataclass(frozen=True)
class PlannerFeatureSchemaReceiptV7V5:
    feature_registry: v1.FeatureAllowlistsV7
    schema_producer_code_hash: str
    schema_config_hash: str
    frozen_before_fit: bool
    provenance_hash: str
    schema_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.feature_registry, v1.FeatureAllowlistsV7):
            raise ValueError("planner schema needs typed feature registry")
        for name, value in (
            ("planner schema producer-code hash", self.schema_producer_code_hash),
            ("planner schema config hash", self.schema_config_hash),
            ("planner schema provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        if self.frozen_before_fit is not True:
            raise ValueError("planner feature schema must be frozen before FIT")
        if (self.schema_producer_code_hash
                != PLANNER_SCHEMA_PRODUCER_CODE_HASH_V7V5
                or self.schema_config_hash != PLANNER_SCHEMA_CONFIG_HASH_V7V5):
            raise ValueError("planner feature schema producer/config is unauthorized")
        payload = _planner_feature_schema_payload_v7_v5(self.feature_registry)
        payload.update({
            "schema_producer_code_hash": self.schema_producer_code_hash,
            "schema_config_hash": self.schema_config_hash,
            "frozen_before_fit": self.frozen_before_fit,
            "provenance_hash": self.provenance_hash,
        })
        object.__setattr__(self, "schema_hash", _sealed(
            self.schema_hash, payload, "canonical planner feature-schema hash",
        ))


@dataclass(frozen=True)
class ExternalInferenceReplayAuthorityV7V5:
    """E235 package authority frozen before any target-fold score exists.

    The receipt is necessary but is not a trust anchor by itself.  Production
    proposal/commit also require the root-review seal and the exact post-run
    verification receipt to be supplied through caller-owned trusted sets.
    """

    integration_package_id: str
    integration_package_version: str
    integration_package_hash: str
    package_freeze_receipt_hash: str
    integration_allowlist_hash: str
    root_review_seal_hash: str
    root_review_status: str
    executor_code_hash: str
    producer_code_hash: str
    producer_config_hash: str
    runtime_hash: str
    planner_model_bytes_hash: str
    planner_feature_schema_hash: str
    planner_model_schema_hash: str
    output_replay_schema_hash: str
    frozen_before_score: bool
    mode: ExternalReplayAuthorityModeV7V5
    authority_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("E235 integration package id", self.integration_package_id),
            ("E235 integration package version", self.integration_package_version),
            ("E235 integration package hash", self.integration_package_hash),
            ("E235 package freeze receipt hash", self.package_freeze_receipt_hash),
            ("E235 integration allowlist hash", self.integration_allowlist_hash),
            ("E235 root-review seal hash", self.root_review_seal_hash),
            ("E235 root-review status", self.root_review_status),
            ("E235 executor-code hash", self.executor_code_hash),
            ("E235 producer-code hash", self.producer_code_hash),
            ("E235 producer-config hash", self.producer_config_hash),
            ("E235 runtime hash", self.runtime_hash),
            ("E235 planner-model bytes hash", self.planner_model_bytes_hash),
            ("E235 planner feature-schema hash", self.planner_feature_schema_hash),
            ("E235 planner model-schema hash", self.planner_model_schema_hash),
            ("E235 output-replay schema hash", self.output_replay_schema_hash),
        ):
            _text(value, name)
        if not isinstance(self.mode, ExternalReplayAuthorityModeV7V5):
            raise ValueError("E235 authority mode is untyped")
        if self.frozen_before_score is not True:
            raise ValueError("E235 package authority must be frozen before score")
        if self.output_replay_schema_hash != E235_OUTPUT_REPLAY_SCHEMA_HASH_V7V5:
            raise ValueError("E235 output-replay schema is not the frozen SR1 schema")
        if self.planner_model_schema_hash != PLANNER_MODEL_SCHEMA_HASH_V7V5:
            raise ValueError("E235 planner-model schema is not the frozen SR1 schema")
        if self.mode is ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED:
            if self.root_review_status != "PASS":
                raise ValueError("production E235 authority needs exact PASS review")
        elif self.mode is ExternalReplayAuthorityModeV7V5.TEST_ONLY:
            if (self.root_review_status != "TEST_ONLY"
                    or self.root_review_seal_hash != "TEST_ONLY_NO_ROOT_SEAL"):
                raise ValueError("synthetic E235 authority must be visibly TEST_ONLY")
        else:
            raise ValueError("UNVERIFIED is a decision disposition, not authority")
        payload = {
            name: (
                getattr(self, name).value
                if isinstance(getattr(self, name), Enum)
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
            if name != "authority_hash"
        }
        object.__setattr__(self, "authority_hash", _sealed(
            self.authority_hash, payload, "external E235 authority hash",
        ))


@dataclass(frozen=True)
class PlannerFitReceiptV7V5:
    fit_fold_role: v1.FoldRoleV7
    target_fold_role: v1.FoldRoleV7
    target_fold_id: str
    fit_fold_ids: tuple[str, ...]
    split_authority: SplitAuthorityReceiptV7V5
    fit_data_hash: str
    fit_target_hash: str
    group_weighting_hash: str
    standardization_artifact_hash: str
    standardization_receipt_hash: str
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
        _text(self.target_fold_id, "planner FIT target fold id")
        if not isinstance(self.split_authority, SplitAuthorityReceiptV7V5):
            raise ValueError("planner FIT needs typed exact split authority")
        folds = _sorted_unique(self.fit_fold_ids, "planner FIT fold ids")
        if not folds:
            raise ValueError("planner FIT fold ids cannot be empty")
        if folds != self.split_authority.fit_fold_ids:
            raise ValueError("planner FIT folds differ from split authority")
        target_family = (
            self.split_authority.evaluation_fold_ids
            if self.target_fold_role in {v1.FoldRoleV7.EVAL, v1.FoldRoleV7.RUNTIME}
            else self.split_authority.calibration_fold_ids
            if self.target_fold_role is v1.FoldRoleV7.CAL else ()
        )
        if (self.target_fold_id not in target_family
                or self.target_fold_id != self.split_authority.target_fold_id
                or self.target_fold_id in folds):
            raise ValueError("planner target fold is not disjoint from FIT authority")
        for name, value in (
            ("planner FIT data hash", self.fit_data_hash),
            ("planner FIT target hash", self.fit_target_hash),
            ("planner FIT group-weighting hash", self.group_weighting_hash),
            ("planner FIT standardization artifact hash",
             self.standardization_artifact_hash),
            ("planner FIT standardization receipt hash",
             self.standardization_receipt_hash),
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
            "target_fold_id": self.target_fold_id,
            "fit_fold_ids": list(folds),
            "split_authority_hash": self.split_authority.receipt_hash,
            "fit_data_hash": self.fit_data_hash,
            "fit_target_hash": self.fit_target_hash,
            "group_weighting_hash": self.group_weighting_hash,
            "standardization_artifact_hash": self.standardization_artifact_hash,
            "standardization_receipt_hash": self.standardization_receipt_hash,
            "availability": self.availability.value,
            "availability_reason": self.availability_reason,
            "frozen_before_target_fold": self.frozen_before_target_fold,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "fit_fold_ids", folds)
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "planner FIT receipt hash",
        ))


def planner_model_identity_hash_v7_v5(
    fit_receipt: PlannerFitReceiptV7V5,
    feature_schema_receipt: PlannerFeatureSchemaReceiptV7V5,
    model_bytes_hash: str,
    required_e235_replay_authority_hash: str,
) -> str:
    _text(model_bytes_hash, "planner model bytes hash")
    _text(required_e235_replay_authority_hash, "E235 replay authority hash")
    return _sha256({
        "fit_receipt_hash": fit_receipt.receipt_hash,
        "feature_schema_hash": feature_schema_receipt.schema_hash,
        "model_bytes_hash": model_bytes_hash,
        "required_e235_replay_authority_hash": (
            required_e235_replay_authority_hash
        ),
        "algorithm_id": PLANNER_ALGORITHM_ID_V7V5,
        "algorithm_version": PLANNER_ALGORITHM_VERSION_V7V5,
        "config_hash": PLANNER_CONFIG_HASH_V7V5,
        "target_id": PLANNER_TARGET_ID_V7V5,
    })


@dataclass(frozen=True)
class PlannerModelReceiptV7V5:
    fit_receipt: PlannerFitReceiptV7V5
    feature_registry: v1.FeatureAllowlistsV7
    feature_schema_receipt: PlannerFeatureSchemaReceiptV7V5
    model_bytes_hash: str
    required_e235_replay_authority_hash: str
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
        if not isinstance(self.fit_receipt, PlannerFitReceiptV7V5):
            raise ValueError("planner model needs a typed FIT receipt")
        if not isinstance(self.feature_registry, v1.FeatureAllowlistsV7):
            raise ValueError("planner model needs a typed feature registry")
        if not isinstance(
            self.feature_schema_receipt, PlannerFeatureSchemaReceiptV7V5,
        ):
            raise ValueError("planner model needs typed feature-schema receipt")
        for name, value in (
            ("planner model bytes hash", self.model_bytes_hash),
            ("required E235 replay authority hash",
             self.required_e235_replay_authority_hash),
            ("planner model hash", self.model_hash),
            ("planner model provenance hash", self.model_provenance_hash),
            ("planner model feature-schema hash", self.feature_schema_hash),
            ("planner algorithm id", self.algorithm_id),
            ("planner algorithm version", self.algorithm_version),
            ("planner config hash", self.config_hash),
            ("planner target id", self.target_id),
        ):
            _text(value, name)
        if (self.algorithm_id != PLANNER_ALGORITHM_ID_V7V5
                or self.algorithm_version != PLANNER_ALGORITHM_VERSION_V7V5
                or self.config_hash != PLANNER_CONFIG_HASH_V7V5
                or self.target_id != PLANNER_TARGET_ID_V7V5):
            raise ValueError("planner algorithm/config/target authority drifted")
        schema = self.feature_schema_receipt
        if (schema.feature_registry != self.feature_registry
                or schema.feature_registry.combined_fingerprint
                != self.feature_registry.combined_fingerprint
                or self.feature_schema_hash != schema.schema_hash):
            raise ValueError("planner feature schema is not canonical/registry-bound")
        if self.model_hash != planner_model_identity_hash_v7_v5(
            self.fit_receipt, schema, self.model_bytes_hash,
            self.required_e235_replay_authority_hash,
        ):
            raise ValueError("planner model hash does not bind exact model bytes")
        if self.frozen_before_scoring is not True:
            raise ValueError("planner model must be frozen before scoring")
        payload = {
            "fit_receipt_hash": self.fit_receipt.receipt_hash,
            "feature_registry_fingerprint": (
                self.feature_registry.combined_fingerprint
            ),
            "feature_schema_receipt_hash": schema.schema_hash,
            "planner_allowlist_hash": (
                self.feature_registry.planner_allowlist_hash
            ),
            "model_hash": self.model_hash,
            "model_bytes_hash": self.model_bytes_hash,
            "required_e235_replay_authority_hash": (
                self.required_e235_replay_authority_hash
            ),
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
class ImmutableNativeReceiptV7V5:
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
class ManifestFreezeReceiptV7V5:
    base_manifest_hash: str
    candidate_family_hash: str
    source_hash: str
    target_fold_role: v1.FoldRoleV7
    sr0a_manifest_receipt_hash: str
    split_authority_root_hash: str
    fold_universe_hash: str
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
            ("frozen split-authority root hash",
             self.split_authority_root_hash),
            ("frozen five-fold universe hash", self.fold_universe_hash),
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
            "split_authority_root_hash": self.split_authority_root_hash,
            "fold_universe_hash": self.fold_universe_hash,
            "freeze_sequence": self.freeze_sequence,
            "outcome_capability_absent": self.outcome_capability_absent,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "manifest freeze receipt hash",
        ))


def _validate_canonical_native(
    manifest: v1.CandidateManifestV7,
    receipt: ImmutableNativeReceiptV7V5,
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
class CandidateManifestV7V5:
    base_manifest: v1.CandidateManifestV7
    immutable_native_receipt: ImmutableNativeReceiptV7V5
    freeze_receipt: ManifestFreezeReceiptV7V5
    planner_model_receipt: PlannerModelReceiptV7V5
    schema_version: str = SELECTOR_V7_V5_SCHEMA_VERSION
    predecessor_failed_review_sha256: str = SELECTOR_V7_V4_REVIEWED_SHA256
    manifest_v5_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SELECTOR_V7_V5_SCHEMA_VERSION:
            raise ValueError("SelectorV7-v5 schema version drifted")
        if self.predecessor_failed_review_sha256 != (
            SELECTOR_V7_V4_REVIEWED_SHA256
        ):
            raise ValueError("SelectorV7-v5 predecessor disposition drifted")
        if not isinstance(self.base_manifest, v1.CandidateManifestV7):
            raise ValueError("v2 manifest needs a typed v1 base manifest")
        if not isinstance(self.immutable_native_receipt, ImmutableNativeReceiptV7V5):
            raise ValueError("v3 manifest needs immutable native receipt")
        if not isinstance(self.freeze_receipt, ManifestFreezeReceiptV7V5):
            raise ValueError("v3 manifest needs typed outcome-blind freeze receipt")
        if not isinstance(self.planner_model_receipt, PlannerModelReceiptV7V5):
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
                or model.fit_receipt.split_authority.sr0a_manifest_receipt_hash
                != freeze.sr0a_manifest_receipt_hash
                or model.fit_receipt.split_authority.split_authority_root_hash
                != freeze.split_authority_root_hash
                or model.fit_receipt.split_authority.fold_universe_hash
                != freeze.fold_universe_hash
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
        object.__setattr__(self, "manifest_v5_hash", _sealed(
            self.manifest_v5_hash, payload, "SelectorV7-v5 manifest hash",
        ))

    @property
    def native(self) -> v1.CandidateArmV7:
        return self.base_manifest.native


@dataclass(frozen=True)
class PreRunAliasSealV7V5:
    manifest_v5_hash: str
    representative_planned_arm_id: str
    alias_planned_arm_ids: tuple[str, ...]
    alias_receipt_bindings: tuple[tuple[str, str], ...]
    sealed_at_ns: int
    reuse_policy_hash: str
    provenance_hash: str
    seal_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("alias-seal manifest hash", self.manifest_v5_hash),
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
            "manifest_v5_hash": self.manifest_v5_hash,
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
class AliasRunPlanV7V5:
    manifest_v5_hash: str
    runtime_run_id: str
    representative_planned_arm_id: str
    alias_planned_arm_ids: tuple[str, ...]
    alias_receipt_bindings: tuple[tuple[str, str], ...]
    mode: AliasChronologyModeV7V5
    earliest_run_start_ns: int
    pre_run_alias_seal: PreRunAliasSealV7V5 | None
    runtime_policy_hash: str
    provenance_hash: str
    plan_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("run-plan manifest hash", self.manifest_v5_hash),
            ("runtime run id", self.runtime_run_id),
            ("run-plan representative id", self.representative_planned_arm_id),
            ("runtime policy hash", self.runtime_policy_hash),
            ("run-plan provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        if not isinstance(self.mode, AliasChronologyModeV7V5):
            raise ValueError("run-plan alias chronology mode is untyped")
        aliases = _sorted_unique(self.alias_planned_arm_ids, "run-plan aliases")
        bindings = tuple(sorted(self.alias_receipt_bindings))
        if tuple(row[0] for row in bindings) != aliases:
            raise ValueError("run plan must bind each alias receipt exactly once")
        _integer(self.earliest_run_start_ns, "earliest run start")
        if self.mode is AliasChronologyModeV7V5.DIRECT_NO_ALIAS:
            if aliases or bindings or self.pre_run_alias_seal is not None:
                raise ValueError("direct run plan cannot carry aliases or reuse seal")
        elif not aliases:
            raise ValueError("alias chronology mode needs at least one alias")
        if self.mode is AliasChronologyModeV7V5.PRE_RUN_FROZEN_REUSE:
            seal = self.pre_run_alias_seal
            if not isinstance(seal, PreRunAliasSealV7V5):
                raise ValueError("pre-run reuse needs a typed frozen alias seal")
            if (seal.manifest_v5_hash != self.manifest_v5_hash
                    or seal.representative_planned_arm_id
                    != self.representative_planned_arm_id
                    or seal.alias_planned_arm_ids != aliases
                    or seal.alias_receipt_bindings != bindings
                    or seal.sealed_at_ns >= self.earliest_run_start_ns):
                raise ValueError("alias reuse seal was not frozen before runtime")
        elif self.pre_run_alias_seal is not None:
            raise ValueError("post-hoc/duplicate modes cannot claim pre-run reuse")
        payload = {
            "manifest_v5_hash": self.manifest_v5_hash,
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
class RuntimeExecutionReceiptV7V5:
    manifest_v5_hash: str
    proposal_hash: str
    alias_run_plan_hash: str
    runtime_run_id: str
    planned_arm_id: str
    realization_receipt_hash: str
    executable_bytes_hash: str
    operation: RuntimeOperationV7V5
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
            ("runtime manifest hash", self.manifest_v5_hash),
            ("runtime proposal hash", self.proposal_hash),
            ("runtime alias-plan hash", self.alias_run_plan_hash),
            ("runtime run id", self.runtime_run_id),
            ("runtime planned arm id", self.planned_arm_id),
            ("runtime realization receipt hash", self.realization_receipt_hash),
            ("runtime executable bytes hash", self.executable_bytes_hash),
            ("runtime provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        if not isinstance(self.operation, RuntimeOperationV7V5):
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
        if self.operation is RuntimeOperationV7V5.EXECUTE_ARM:
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
            "manifest_v5_hash": self.manifest_v5_hash,
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
class RuntimeCostLedgerV7V5:
    manifest_v5_hash: str
    proposal_hash: str
    alias_run_plan_hash: str
    runtime_run_id: str
    representative_planned_arm_id: str
    alias_planned_arm_ids: tuple[str, ...]
    mode: AliasChronologyModeV7V5
    execution_receipts: tuple[RuntimeExecutionReceiptV7V5, ...]
    post_hoc_discovered_at_ns: int | None
    charged_prospective_runtime_cost: float
    charged_prospective_runtime_cost_counters: v1.CostCountersV7
    cost_account: v1.CostAccountV7
    provenance_hash: str
    ledger_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("ledger manifest hash", self.manifest_v5_hash),
            ("ledger proposal hash", self.proposal_hash),
            ("ledger alias-plan hash", self.alias_run_plan_hash),
            ("ledger run id", self.runtime_run_id),
            ("ledger representative id", self.representative_planned_arm_id),
            ("ledger provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        aliases = _sorted_unique(self.alias_planned_arm_ids, "ledger aliases")
        if not isinstance(self.mode, AliasChronologyModeV7V5):
            raise ValueError("ledger alias chronology mode is untyped")
        receipts = tuple(self.execution_receipts)
        if not receipts or any(not isinstance(
            row, RuntimeExecutionReceiptV7V5,
        ) for row in receipts):
            raise ValueError("runtime ledger needs typed execution receipts")
        if len({row.receipt_hash for row in receipts}) != len(receipts):
            raise ValueError("runtime execution receipts must be unique")
        if any(
            row.manifest_v5_hash != self.manifest_v5_hash
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
            "manifest_v5_hash": self.manifest_v5_hash,
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
class BeforeFeatureArtifactReceiptV7V5:
    base_manifest_hash: str
    candidate_id: str
    planned_arm_id: str
    before_feature_hash: str
    feature_artifact_bytes_hash: str
    planner_feature_schema_hash: str
    feature_registry_fingerprint: str
    source_input_hash: str
    producer_code_hash: str
    producer_config_hash: str
    runtime_hash: str
    frozen_before_inference: bool
    provenance_hash: str
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("before-feature manifest hash", self.base_manifest_hash),
            ("before-feature candidate id", self.candidate_id),
            ("before-feature planned-arm id", self.planned_arm_id),
            ("before-feature logical hash", self.before_feature_hash),
            ("before-feature artifact bytes hash", self.feature_artifact_bytes_hash),
            ("before-feature planner schema hash", self.planner_feature_schema_hash),
            ("before-feature registry fingerprint", self.feature_registry_fingerprint),
            ("before-feature source-input hash", self.source_input_hash),
            ("before-feature producer-code hash", self.producer_code_hash),
            ("before-feature producer-config hash", self.producer_config_hash),
            ("before-feature runtime hash", self.runtime_hash),
            ("before-feature provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        if self.frozen_before_inference is not True:
            raise ValueError("before-feature artifact must predate inference")
        payload = {
            name: getattr(self, name)
            for name in (
                "base_manifest_hash", "candidate_id", "planned_arm_id",
                "before_feature_hash", "feature_artifact_bytes_hash",
                "planner_feature_schema_hash", "feature_registry_fingerprint",
                "source_input_hash", "producer_code_hash",
                "producer_config_hash", "runtime_hash",
                "frozen_before_inference", "provenance_hash",
            )
        }
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "before-feature artifact receipt hash",
        ))


@dataclass(frozen=True)
class PlannerInferenceExecutionReceiptV7V5:
    candidate_id: str
    planned_arm_id: str
    planner_model_receipt_hash: str
    planner_model_hash: str
    planner_model_bytes_hash: str
    planner_fit_receipt_hash: str
    target_fold_role: v1.FoldRoleV7
    target_fold_id: str
    before_feature_artifact_receipt_hash: str
    before_feature_artifact_bytes_hash: str
    input_identity_hash: str
    producer_code_hash: str
    producer_config_hash: str
    runtime_hash: str
    inference_run_id: str
    predicted_gain_raw_px: float
    output_score_bytes_hash: str
    required_e235_replay_authority_hash: str
    external_replay_receipt_hash: str
    completed_before_observation: bool
    provenance_hash: str
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("inference candidate id", self.candidate_id),
            ("inference planned-arm id", self.planned_arm_id),
            ("score planner-model receipt hash", self.planner_model_receipt_hash),
            ("score planner-model hash", self.planner_model_hash),
            ("score planner-model bytes hash", self.planner_model_bytes_hash),
            ("score planner FIT receipt hash", self.planner_fit_receipt_hash),
            ("inference target fold id", self.target_fold_id),
            ("before-feature artifact receipt hash",
             self.before_feature_artifact_receipt_hash),
            ("before-feature artifact bytes hash",
             self.before_feature_artifact_bytes_hash),
            ("inference input identity hash", self.input_identity_hash),
            ("inference producer-code hash", self.producer_code_hash),
            ("inference producer-config hash", self.producer_config_hash),
            ("inference runtime hash", self.runtime_hash),
            ("inference run id", self.inference_run_id),
            ("output score bytes hash", self.output_score_bytes_hash),
            ("required E235 replay authority hash",
             self.required_e235_replay_authority_hash),
            ("external E235 replay receipt hash",
             self.external_replay_receipt_hash),
            ("inference provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        if not isinstance(self.target_fold_role, v1.FoldRoleV7):
            raise ValueError("planner inference target fold is untyped")
        score = _finite(self.predicted_gain_raw_px, "inference predicted gain")
        expected_output = _sha256({"float64_hex": score.hex()})
        if self.output_score_bytes_hash != expected_output:
            raise ValueError("output score bytes do not bind predicted gain")
        if self.completed_before_observation is not True:
            raise ValueError("planner inference must precede observation")
        payload = {
            "candidate_id": self.candidate_id,
            "planned_arm_id": self.planned_arm_id,
            "planner_model_receipt_hash": self.planner_model_receipt_hash,
            "planner_model_hash": self.planner_model_hash,
            "planner_model_bytes_hash": self.planner_model_bytes_hash,
            "planner_fit_receipt_hash": self.planner_fit_receipt_hash,
            "target_fold_role": self.target_fold_role.value,
            "target_fold_id": self.target_fold_id,
            "before_feature_artifact_receipt_hash": (
                self.before_feature_artifact_receipt_hash
            ),
            "before_feature_artifact_bytes_hash": (
                self.before_feature_artifact_bytes_hash
            ),
            "input_identity_hash": self.input_identity_hash,
            "producer_code_hash": self.producer_code_hash,
            "producer_config_hash": self.producer_config_hash,
            "runtime_hash": self.runtime_hash,
            "inference_run_id": self.inference_run_id,
            "predicted_gain_raw_px": score,
            "output_score_bytes_hash": self.output_score_bytes_hash,
            "required_e235_replay_authority_hash": (
                self.required_e235_replay_authority_hash
            ),
            "external_replay_receipt_hash": self.external_replay_receipt_hash,
            "completed_before_observation": self.completed_before_observation,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "planner inference execution receipt hash",
        ))


@dataclass(frozen=True)
class PlannerScoreReceiptV7V5:
    base_score: v1.PlannerCandidateV7
    before_feature_artifact: BeforeFeatureArtifactReceiptV7V5
    inference_execution: PlannerInferenceExecutionReceiptV7V5
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.base_score, v1.PlannerCandidateV7):
            raise ValueError("planner score receipt needs typed base score")
        if not isinstance(
            self.before_feature_artifact, BeforeFeatureArtifactReceiptV7V5,
        ) or not isinstance(
            self.inference_execution, PlannerInferenceExecutionReceiptV7V5,
        ):
            raise ValueError("planner score needs explicit artifact/inference receipts")
        artifact = self.before_feature_artifact
        execution = self.inference_execution
        if (self.base_score.candidate_id != artifact.candidate_id
                or self.base_score.candidate_id != execution.candidate_id
                or self.base_score.planned_arm_id != artifact.planned_arm_id
                or self.base_score.planned_arm_id != execution.planned_arm_id
                or self.base_score.before_feature_hash
                != artifact.before_feature_hash
                or execution.before_feature_artifact_receipt_hash
                != artifact.receipt_hash
                or execution.before_feature_artifact_bytes_hash
                != artifact.feature_artifact_bytes_hash
                or not math.isclose(
                    self.base_score.planner_predicted_gain_raw_px,
                    execution.predicted_gain_raw_px,
                    rel_tol=0.0, abs_tol=0.0,
                )):
            raise ValueError("planner score differs from explicit inference execution")
        payload = {
            "base_score": {
                name: (
                    getattr(self.base_score, name).value
                    if isinstance(getattr(self.base_score, name), Enum)
                    else getattr(self.base_score, name)
                )
                for name in self.base_score.__dataclass_fields__
            },
            "before_feature_artifact_receipt_hash": artifact.receipt_hash,
            "inference_execution_receipt_hash": execution.receipt_hash,
        }
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash, payload, "planner score receipt hash",
        ))


@dataclass(frozen=True)
class ExternalInferenceReplayVerificationV7V5:
    """Post-inference E235 verification, trusted only through an external set."""

    authority_hash: str
    base_manifest_hash: str
    planner_model_hash: str
    planner_model_bytes_hash: str
    planner_feature_schema_hash: str
    integration_allowlist_hash: str
    root_review_seal_hash: str
    root_review_status: str
    output_replay_schema_hash: str
    execution_bindings: tuple[tuple[str, str, str, str], ...]
    verified_before_proposal: bool
    mode: ExternalReplayAuthorityModeV7V5
    provenance_hash: str
    verification_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("verification authority hash", self.authority_hash),
            ("verification base-manifest hash", self.base_manifest_hash),
            ("verification planner-model hash", self.planner_model_hash),
            ("verification planner-model bytes hash", self.planner_model_bytes_hash),
            ("verification planner feature-schema hash",
             self.planner_feature_schema_hash),
            ("verification integration allowlist hash",
             self.integration_allowlist_hash),
            ("verification root-review seal hash", self.root_review_seal_hash),
            ("verification root-review status", self.root_review_status),
            ("verification output-replay schema hash",
             self.output_replay_schema_hash),
            ("verification provenance hash", self.provenance_hash),
        ):
            _text(value, name)
        if not isinstance(self.mode, ExternalReplayAuthorityModeV7V5):
            raise ValueError("E235 verification mode is untyped")
        if self.verified_before_proposal is not True:
            raise ValueError("E235 inference must be verified before proposal")
        if self.output_replay_schema_hash != E235_OUTPUT_REPLAY_SCHEMA_HASH_V7V5:
            raise ValueError("E235 verification schema drifted")
        bindings = tuple(sorted(self.execution_bindings))
        if (not bindings or len(bindings) != len({row[0] for row in bindings})
                or any(not isinstance(row, tuple) or len(row) != 4
                       or any(not isinstance(value, str) or not value
                              for value in row)
                       for row in bindings)):
            raise ValueError("E235 verification needs one exact binding per score")
        if self.mode is ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED:
            if self.root_review_status != "PASS":
                raise ValueError("production E235 verification needs PASS status")
        elif self.mode is ExternalReplayAuthorityModeV7V5.TEST_ONLY:
            if (self.root_review_status != "TEST_ONLY"
                    or self.root_review_seal_hash != "TEST_ONLY_NO_ROOT_SEAL"):
                raise ValueError("synthetic E235 verification must be TEST_ONLY")
        else:
            raise ValueError("UNVERIFIED cannot sign inference verification")
        payload = {
            "authority_hash": self.authority_hash,
            "base_manifest_hash": self.base_manifest_hash,
            "planner_model_hash": self.planner_model_hash,
            "planner_model_bytes_hash": self.planner_model_bytes_hash,
            "planner_feature_schema_hash": self.planner_feature_schema_hash,
            "integration_allowlist_hash": self.integration_allowlist_hash,
            "root_review_seal_hash": self.root_review_seal_hash,
            "root_review_status": self.root_review_status,
            "output_replay_schema_hash": self.output_replay_schema_hash,
            "execution_bindings": [list(row) for row in bindings],
            "verified_before_proposal": self.verified_before_proposal,
            "mode": self.mode.value,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "execution_bindings", bindings)
        object.__setattr__(self, "verification_hash", _sealed(
            self.verification_hash, payload, "external E235 verification hash",
        ))


@dataclass(frozen=True)
class PlannerScoreBankV7V5:
    manifest_v5_hash: str
    candidate_family_hash: str
    planner_allowlist_hash: str
    planner_model_receipt: PlannerModelReceiptV7V5
    universe_digest: str
    universe_cardinality: int
    scores: tuple[v1.PlannerCandidateV7, ...]
    frozen_before_observation: bool
    score_receipts: tuple[PlannerScoreReceiptV7V5, ...]
    score_bank_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("score-bank manifest hash", self.manifest_v5_hash),
            ("score-bank candidate-family hash", self.candidate_family_hash),
            ("score-bank planner allowlist hash", self.planner_allowlist_hash),
            ("score-bank universe digest", self.universe_digest),
        ):
            _text(value, name)
        if not isinstance(self.planner_model_receipt, PlannerModelReceiptV7V5):
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
        receipts = tuple(self.score_receipts)
        if any(not isinstance(row, PlannerScoreReceiptV7V5) for row in receipts):
            raise ValueError("score bank requires explicit typed inference receipts")
        by_receipt_id = {
            row.base_score.planned_arm_id: row for row in receipts
        }
        if (len(by_receipt_id) != len(receipts)
                or set(by_receipt_id)
                != {row.planned_arm_id for row in rows}):
            raise ValueError("planner score receipt family is incomplete")
        for score in rows:
            receipt = by_receipt_id[score.planned_arm_id]
            artifact = receipt.before_feature_artifact
            execution = receipt.inference_execution
            if (receipt.base_score != score
                    or execution.planner_model_receipt_hash
                    != authority.receipt_hash
                    or execution.planner_model_hash != authority.model_hash
                    or execution.planner_model_bytes_hash
                    != authority.model_bytes_hash
                    or execution.planner_fit_receipt_hash
                    != authority.fit_receipt.receipt_hash
                    or execution.target_fold_role
                    is not authority.fit_receipt.target_fold_role
                    or execution.target_fold_id
                    != authority.fit_receipt.target_fold_id
                    or artifact.feature_registry_fingerprint
                    != authority.feature_registry.combined_fingerprint
                    or artifact.planner_feature_schema_hash
                    != authority.feature_schema_hash
                    or execution.required_e235_replay_authority_hash
                    != authority.required_e235_replay_authority_hash
                    or not execution.external_replay_receipt_hash):
                raise ValueError("planner score differs from model/FIT authority")
        payload = {
            "manifest_v5_hash": self.manifest_v5_hash,
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
class PlannerProposalV7V5:
    manifest_v5_hash: str
    candidate_family_hash: str
    score_bank: PlannerScoreBankV7V5
    alias_run_plans: tuple[AliasRunPlanV7V5, ...]
    alias_run_plan_digest: str
    planner_algorithm_id: str
    planner_algorithm_version: str
    planner_config_hash: str
    planner_provenance_hash: str
    external_authority_mode: ExternalReplayAuthorityModeV7V5
    external_authority_hash: str
    external_verification_hash: str
    production_eligibility: ProductionEligibilityV7V5
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
            ("proposal manifest hash", self.manifest_v5_hash),
            ("proposal candidate-family hash", self.candidate_family_hash),
            ("proposal run-plan digest", self.alias_run_plan_digest),
            ("planner algorithm id", self.planner_algorithm_id),
            ("planner algorithm version", self.planner_algorithm_version),
            ("planner config hash", self.planner_config_hash),
            ("planner provenance hash", self.planner_provenance_hash),
            ("proposal external authority hash", self.external_authority_hash),
            ("proposal external verification hash",
             self.external_verification_hash),
        ):
            _text(value, name)
        if not isinstance(self.score_bank, PlannerScoreBankV7V5):
            raise ValueError("proposal needs a typed complete score bank")
        if not isinstance(
            self.external_authority_mode, ExternalReplayAuthorityModeV7V5,
        ) or not isinstance(
            self.production_eligibility, ProductionEligibilityV7V5,
        ):
            raise ValueError("proposal external authority disposition is untyped")
        if self.external_authority_mode is (
            ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED
        ):
            if self.production_eligibility is not (
                ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE
            ):
                raise ValueError("production proposal must be production-eligible")
        elif self.external_authority_mode is (
            ExternalReplayAuthorityModeV7V5.TEST_ONLY
        ):
            if self.production_eligibility is not (
                ProductionEligibilityV7V5.TEST_ONLY_NONAUTHORITY
            ):
                raise ValueError("TEST_ONLY proposal must be marked nonauthority")
        else:
            raise ValueError("unverified authority cannot produce a proposal")
        authority = self.score_bank.planner_model_receipt
        if (self.planner_algorithm_id != authority.algorithm_id
                or self.planner_algorithm_version != authority.algorithm_version
                or self.planner_config_hash != authority.config_hash
                or self.planner_provenance_hash
                != authority.model_provenance_hash):
            raise ValueError("proposal metadata differs from planner authority")
        plans = tuple(self.alias_run_plans)
        if any(not isinstance(row, AliasRunPlanV7V5) for row in plans):
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
            "manifest_v5_hash": self.manifest_v5_hash,
            "candidate_family_hash": self.candidate_family_hash,
            "score_bank_hash": self.score_bank.score_bank_hash,
            "planner_model_receipt_hash": authority.receipt_hash,
            "alias_run_plan_digest": self.alias_run_plan_digest,
            "planner_algorithm_id": self.planner_algorithm_id,
            "planner_algorithm_version": self.planner_algorithm_version,
            "planner_config_hash": self.planner_config_hash,
            "planner_provenance_hash": self.planner_provenance_hash,
            "external_authority_mode": self.external_authority_mode.value,
            "external_authority_hash": self.external_authority_hash,
            "external_verification_hash": self.external_verification_hash,
            "production_eligibility": self.production_eligibility.value,
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
            self.proposal_hash, payload, "SelectorV7-v5 proposal hash",
        ))


def _manifest_arm_maps(manifest: CandidateManifestV7V5) -> tuple[
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
    manifest: CandidateManifestV7V5,
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
    manifest: CandidateManifestV7V5,
    plan: AliasRunPlanV7V5,
    group_members: tuple[str, ...],
    alias_by_id: dict[str, v1.ArmAliasReceiptV7],
) -> None:
    representative = min(group_members)
    aliases = tuple(row for row in group_members if row != representative)
    expected_bindings = tuple(sorted(
        (row, alias_by_id[row].receipt_hash) for row in aliases
    ))
    if (plan.manifest_v5_hash != manifest.manifest_v5_hash
            or plan.representative_planned_arm_id != representative
            or plan.alias_planned_arm_ids != aliases
            or plan.alias_receipt_bindings != expected_bindings):
        raise ValueError("alias run plan does not match exact canonical group")
    if not aliases and plan.mode is not AliasChronologyModeV7V5.DIRECT_NO_ALIAS:
        raise ValueError("non-alias arm must use direct chronology")
    if aliases and plan.mode is AliasChronologyModeV7V5.DIRECT_NO_ALIAS:
        raise ValueError("alias group cannot use direct chronology")


def _group_cost_ceiling(
    representative: str,
    group_members: tuple[str, ...],
    plan: AliasRunPlanV7V5,
    by_arm: dict[str, v1.CandidateArmV7],
    alias_by_id: dict[str, v1.ArmAliasReceiptV7],
) -> tuple[float, v1.CostCountersV7]:
    if plan.mode is AliasChronologyModeV7V5.PRE_RUN_FROZEN_REUSE:
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
    manifest: CandidateManifestV7V5,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    run_plans: Sequence[AliasRunPlanV7V5],
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


def planner_inference_input_identity_hash_v7_v5(
    manifest: CandidateManifestV7V5,
    arm: v1.CandidateArmV7,
) -> str:
    return _sha256({
        "base_manifest_hash": manifest.base_manifest.manifest_hash,
        "source_hash": manifest.base_manifest.source_hash,
        "candidate_id": arm.candidate_id,
        "planned_arm_id": arm.planned_arm_id,
        "action_hash": arm.action_hash,
        "control_hash": arm.control_hash,
        "endpoint_hash": arm.endpoint_hash,
        "support_hash": arm.support_hash,
        "interaction_hash": arm.interaction_hash,
        "cost_hash": arm.cost_hash,
    })


def planner_score_output_bytes_hash_v7_v5(value: float) -> str:
    finite = _finite(value, "planner score output")
    return _sha256({"float64_hex": finite.hex()})


def make_planner_score_receipt_v7_v5(
    manifest: CandidateManifestV7V5,
    arm: v1.CandidateArmV7,
    score: v1.PlannerCandidateV7,
    *,
    feature_artifact_bytes_hash: str,
    feature_producer_code_hash: str,
    feature_producer_config_hash: str,
    inference_producer_code_hash: str,
    inference_runtime_hash: str,
    inference_run_id: str,
    external_replay_receipt_hash: str,
    provenance_hash: str,
) -> PlannerScoreReceiptV7V5:
    if (score.candidate_id != arm.candidate_id
            or score.planned_arm_id != arm.planned_arm_id):
        raise ValueError("planner score does not bind requested arm")
    model = manifest.planner_model_receipt
    artifact = BeforeFeatureArtifactReceiptV7V5(
        base_manifest_hash=manifest.base_manifest.manifest_hash,
        candidate_id=arm.candidate_id,
        planned_arm_id=arm.planned_arm_id,
        before_feature_hash=score.before_feature_hash,
        feature_artifact_bytes_hash=feature_artifact_bytes_hash,
        planner_feature_schema_hash=model.feature_schema_hash,
        feature_registry_fingerprint=(
            model.feature_registry.combined_fingerprint
        ),
        source_input_hash=manifest.base_manifest.source_hash,
        producer_code_hash=feature_producer_code_hash,
        producer_config_hash=feature_producer_config_hash,
        runtime_hash=inference_runtime_hash,
        frozen_before_inference=True,
        provenance_hash=provenance_hash,
    )
    execution = PlannerInferenceExecutionReceiptV7V5(
        candidate_id=arm.candidate_id,
        planned_arm_id=arm.planned_arm_id,
        planner_model_receipt_hash=model.receipt_hash,
        planner_model_hash=model.model_hash,
        planner_model_bytes_hash=model.model_bytes_hash,
        planner_fit_receipt_hash=model.fit_receipt.receipt_hash,
        target_fold_role=model.fit_receipt.target_fold_role,
        target_fold_id=model.fit_receipt.target_fold_id,
        before_feature_artifact_receipt_hash=artifact.receipt_hash,
        before_feature_artifact_bytes_hash=artifact.feature_artifact_bytes_hash,
        input_identity_hash=planner_inference_input_identity_hash_v7_v5(
            manifest, arm,
        ),
        producer_code_hash=inference_producer_code_hash,
        producer_config_hash=model.config_hash,
        runtime_hash=inference_runtime_hash,
        inference_run_id=inference_run_id,
        predicted_gain_raw_px=score.planner_predicted_gain_raw_px,
        output_score_bytes_hash=planner_score_output_bytes_hash_v7_v5(
            score.planner_predicted_gain_raw_px,
        ),
        required_e235_replay_authority_hash=(
            model.required_e235_replay_authority_hash
        ),
        external_replay_receipt_hash=external_replay_receipt_hash,
        completed_before_observation=True,
        provenance_hash=provenance_hash,
    )
    return PlannerScoreReceiptV7V5(score, artifact, execution)


def make_test_only_replay_verification_v7_v5(
    manifest: CandidateManifestV7V5,
    authority: ExternalInferenceReplayAuthorityV7V5,
    score_receipts: Sequence[PlannerScoreReceiptV7V5],
    *,
    provenance_hash: str,
) -> ExternalInferenceReplayVerificationV7V5:
    """Build a visibly synthetic receipt; production entry points reject it."""
    if authority.mode is not ExternalReplayAuthorityModeV7V5.TEST_ONLY:
        raise ValueError("test-only verifier cannot sign production authority")
    return ExternalInferenceReplayVerificationV7V5(
        authority_hash=authority.authority_hash,
        base_manifest_hash=manifest.base_manifest.manifest_hash,
        planner_model_hash=manifest.planner_model_receipt.model_hash,
        planner_model_bytes_hash=manifest.planner_model_receipt.model_bytes_hash,
        planner_feature_schema_hash=(
            manifest.planner_model_receipt.feature_schema_hash
        ),
        integration_allowlist_hash=authority.integration_allowlist_hash,
        root_review_seal_hash=authority.root_review_seal_hash,
        root_review_status="TEST_ONLY",
        output_replay_schema_hash=authority.output_replay_schema_hash,
        execution_bindings=tuple(
            (
                row.base_score.planned_arm_id,
                row.inference_execution.receipt_hash,
                row.inference_execution.external_replay_receipt_hash,
                row.inference_execution.output_score_bytes_hash,
            )
            for row in score_receipts
        ),
        verified_before_proposal=True,
        mode=ExternalReplayAuthorityModeV7V5.TEST_ONLY,
        provenance_hash=provenance_hash,
    )


def _validate_external_replay_v7_v5(
    manifest: CandidateManifestV7V5,
    score_receipts: Sequence[PlannerScoreReceiptV7V5],
    authority: ExternalInferenceReplayAuthorityV7V5,
    verification: ExternalInferenceReplayVerificationV7V5,
    *,
    trusted_root_review_seal_hashes: frozenset[str],
    trusted_verification_hashes: frozenset[str],
    allow_test_only: bool,
) -> None:
    if (not isinstance(authority, ExternalInferenceReplayAuthorityV7V5)
            or not isinstance(
                verification, ExternalInferenceReplayVerificationV7V5,
            ) or not _rehash_valid(authority)
            or not _rehash_valid(verification)):
        raise ValueError("external E235 replay authority is missing or malformed")
    model = manifest.planner_model_receipt
    exact_authority = (
        model.required_e235_replay_authority_hash == authority.authority_hash
        and authority.planner_model_bytes_hash == model.model_bytes_hash
        and authority.planner_feature_schema_hash == model.feature_schema_hash
        and authority.planner_model_schema_hash == PLANNER_MODEL_SCHEMA_HASH_V7V5
        and verification.authority_hash == authority.authority_hash
        and verification.base_manifest_hash == manifest.base_manifest.manifest_hash
        and verification.planner_model_hash == model.model_hash
        and verification.planner_model_bytes_hash == model.model_bytes_hash
        and verification.planner_feature_schema_hash == model.feature_schema_hash
        and verification.integration_allowlist_hash
        == authority.integration_allowlist_hash
        and verification.root_review_seal_hash == authority.root_review_seal_hash
        and verification.root_review_status == authority.root_review_status
        and verification.output_replay_schema_hash
        == authority.output_replay_schema_hash
        and verification.mode is authority.mode
    )
    if not exact_authority:
        raise ValueError("external E235 replay authority binding drifted")
    expected_bindings: list[tuple[str, str, str, str]] = []
    for receipt in score_receipts:
        artifact = receipt.before_feature_artifact
        execution = receipt.inference_execution
        if (artifact.producer_code_hash != authority.executor_code_hash
                or artifact.producer_config_hash
                != authority.producer_config_hash
                or artifact.runtime_hash != authority.runtime_hash
                or execution.producer_code_hash != authority.producer_code_hash
                or execution.producer_config_hash
                != authority.producer_config_hash
                or execution.runtime_hash != authority.runtime_hash
                or execution.required_e235_replay_authority_hash
                != authority.authority_hash):
            raise ValueError("score was not produced by the E235-authorized stack")
        expected_bindings.append((
            receipt.base_score.planned_arm_id,
            execution.receipt_hash,
            execution.external_replay_receipt_hash,
            execution.output_score_bytes_hash,
        ))
    if tuple(sorted(expected_bindings)) != verification.execution_bindings:
        raise ValueError("external E235 replay did not verify the exact score bank")
    if authority.mode is ExternalReplayAuthorityModeV7V5.TEST_ONLY:
        if not allow_test_only:
            raise ValueError("TEST_ONLY E235 authority is forbidden in production")
        return
    if allow_test_only:
        raise ValueError("synthetic entry point cannot grant production authority")
    if (authority.root_review_status != "PASS"
            or authority.root_review_seal_hash
            not in trusted_root_review_seal_hashes
            or verification.verification_hash
            not in trusted_verification_hashes):
        raise ValueError("external E235 authority lacks a caller-trusted PASS root")


def _propose_candidates_v7_v5_impl(
    manifest: CandidateManifestV7V5,
    score_receipts: Sequence[PlannerScoreReceiptV7V5],
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    alias_run_plans: Sequence[AliasRunPlanV7V5],
    *,
    top_k: int,
    maximum_prospective_runtime_cost: float,
    maximum_prospective_runtime_cost_counters: v1.CostCountersV7,
    planner_algorithm_id: str,
    planner_algorithm_version: str,
    planner_config_hash: str,
    planner_provenance_hash: str,
    external_replay_authority: ExternalInferenceReplayAuthorityV7V5,
    external_replay_verification: ExternalInferenceReplayVerificationV7V5,
    trusted_root_review_seal_hashes: frozenset[str],
    trusted_verification_hashes: frozenset[str],
    allow_test_only: bool,
) -> PlannerProposalV7V5:
    """Seal a complete score bank, then deterministically replay K1/K2."""
    if not _rehash_valid(manifest):
        raise ValueError("SelectorV7-v5 manifest is malformed")
    if top_k not in {1, 2}:
        raise ValueError("SelectorV7-v5 top-K must be one or two")
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
        "manifest_v5_hash": manifest.manifest_v5_hash,
        "planned_arm_ids": list(universe_ids),
    })
    receipts_by_score = tuple(score_receipts)
    if any(not isinstance(row, PlannerScoreReceiptV7V5)
           for row in receipts_by_score):
        raise ValueError("planner requires explicit typed inference receipts")
    rows = tuple(row.base_score for row in receipts_by_score)
    by_score = {row.planned_arm_id: row for row in rows}
    if len(by_score) != len(rows) or set(by_score) != set(universe_ids):
        raise ValueError("complete planner score bank is missing or injected rows")
    _validate_external_replay_v7_v5(
        manifest, receipts_by_score, external_replay_authority,
        external_replay_verification,
        trusted_root_review_seal_hashes=trusted_root_review_seal_hashes,
        trusted_verification_hashes=trusted_verification_hashes,
        allow_test_only=allow_test_only,
    )
    by_arm = {row.planned_arm_id: row for row in universe}
    for planned_id, score in by_score.items():
        arm = by_arm[planned_id]
        receipt = next(
            row for row in receipts_by_score
            if row.base_score.planned_arm_id == planned_id
        )
        if (score.candidate_id != arm.candidate_id
                or score.cost_hash != arm.cost_hash
                or score.planner_allowlist_hash
                != manifest.base_manifest.planner_allowlist_hash):
            raise ValueError("planner score binding drifted")
        if (receipt.inference_execution.input_identity_hash
                != planner_inference_input_identity_hash_v7_v5(manifest, arm)):
            raise ValueError("planner inference input identity drifted")
    score_bank = PlannerScoreBankV7V5(
        manifest_v5_hash=manifest.manifest_v5_hash,
        candidate_family_hash=manifest.base_manifest.candidate_family_hash,
        planner_allowlist_hash=manifest.base_manifest.planner_allowlist_hash,
        planner_model_receipt=authority,
        universe_digest=universe_digest,
        universe_cardinality=len(universe),
        scores=rows,
        frozen_before_observation=True,
        score_receipts=receipts_by_score,
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
    return PlannerProposalV7V5(
        manifest_v5_hash=manifest.manifest_v5_hash,
        candidate_family_hash=manifest.base_manifest.candidate_family_hash,
        score_bank=score_bank,
        alias_run_plans=tuple(alias_run_plans),
        alias_run_plan_digest=plan_digest,
        planner_algorithm_id=planner_algorithm_id,
        planner_algorithm_version=planner_algorithm_version,
        planner_config_hash=planner_config_hash,
        planner_provenance_hash=planner_provenance_hash,
        external_authority_mode=external_replay_authority.mode,
        external_authority_hash=external_replay_authority.authority_hash,
        external_verification_hash=(
            external_replay_verification.verification_hash
        ),
        production_eligibility=(
            ProductionEligibilityV7V5.TEST_ONLY_NONAUTHORITY
            if allow_test_only else
            ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE
        ),
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


def propose_candidates_v7_v5(
    manifest: CandidateManifestV7V5,
    score_receipts: Sequence[PlannerScoreReceiptV7V5],
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    alias_run_plans: Sequence[AliasRunPlanV7V5],
    *,
    top_k: int,
    maximum_prospective_runtime_cost: float,
    maximum_prospective_runtime_cost_counters: v1.CostCountersV7,
    planner_algorithm_id: str,
    planner_algorithm_version: str,
    planner_config_hash: str,
    planner_provenance_hash: str,
    external_replay_authority: ExternalInferenceReplayAuthorityV7V5,
    external_replay_verification: ExternalInferenceReplayVerificationV7V5,
    trusted_root_review_seal_hashes: frozenset[str] = frozenset(),
    trusted_verification_hashes: frozenset[str] = frozenset(),
) -> PlannerProposalV7V5:
    """Production planner entry point; trust is external and fail-closed."""
    return _propose_candidates_v7_v5_impl(
        manifest, score_receipts, realizations, aliases, alias_run_plans,
        top_k=top_k,
        maximum_prospective_runtime_cost=maximum_prospective_runtime_cost,
        maximum_prospective_runtime_cost_counters=(
            maximum_prospective_runtime_cost_counters
        ),
        planner_algorithm_id=planner_algorithm_id,
        planner_algorithm_version=planner_algorithm_version,
        planner_config_hash=planner_config_hash,
        planner_provenance_hash=planner_provenance_hash,
        external_replay_authority=external_replay_authority,
        external_replay_verification=external_replay_verification,
        trusted_root_review_seal_hashes=trusted_root_review_seal_hashes,
        trusted_verification_hashes=trusted_verification_hashes,
        allow_test_only=False,
    )


def propose_candidates_v7_v5_test_only(
    manifest: CandidateManifestV7V5,
    score_receipts: Sequence[PlannerScoreReceiptV7V5],
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    alias_run_plans: Sequence[AliasRunPlanV7V5],
    **kwargs: object,
) -> PlannerProposalV7V5:
    """Synthetic contract replay.  Never grants a production authority seal."""
    return _propose_candidates_v7_v5_impl(
        manifest, score_receipts, realizations, aliases, alias_run_plans,
        trusted_root_review_seal_hashes=frozenset(),
        trusted_verification_hashes=frozenset(),
        allow_test_only=True,
        **kwargs,
    )


@dataclass(frozen=True)
class CandidateObservationV7V5:
    manifest_v5_hash: str
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
    observation_source: ObservationSourceV7V5
    runtime_cost_ledger_hash: str | None
    observation_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("observation manifest hash", self.manifest_v5_hash),
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
        if not isinstance(self.observation_source, ObservationSourceV7V5):
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
            ObservationSourceV7V5.PROSPECTIVE_RUNTIME_OBSERVATION
        ):
            _text(self.runtime_cost_ledger_hash, "runtime cost ledger hash")
        elif self.runtime_cost_ledger_hash is not None:
            raise ValueError("frozen family observation cannot claim runtime cost")
        payload = {
            "manifest_v5_hash": self.manifest_v5_hash,
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
            self.observation_hash, payload, "SelectorV7-v5 observation hash",
        ))


def calibration_choice_family_digest_v7_v5(
    manifest: CandidateManifestV7V5,
    proposal: PlannerProposalV7V5,
) -> str:
    """CAL-time family identity; deliberately excludes runtime observations."""
    return _sha256({
        "manifest_v5_hash": manifest.manifest_v5_hash,
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
class CalibrationChoiceFamilyReceiptV7V5:
    manifest_v5_hash: str
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
            ("CAL-family manifest hash", self.manifest_v5_hash),
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
            "manifest_v5_hash": self.manifest_v5_hash,
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
class CalibrationReceiptV7V5:
    manifest_v5_hash: str
    base_receipt: v1.CalibrationReceiptV7
    choice_family_receipt: CalibrationChoiceFamilyReceiptV7V5
    assessor_model_provenance_hash: str
    assessor_algorithm_id: str
    assessor_algorithm_version: str
    assessor_feature_schema_hash: str
    assessor_config_hash: str
    simultaneous_correction_id: str
    frozen_before_decision: bool
    calibration_v5_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("calibration manifest hash", self.manifest_v5_hash),
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
        if not isinstance(family, CalibrationChoiceFamilyReceiptV7V5):
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
                or family.manifest_v5_hash != self.manifest_v5_hash):
            raise ValueError("CAL-family authority differs from calibration")
        if self.frozen_before_decision is not True:
            raise ValueError("v3 calibration must be frozen before decision")
        payload = {
            "manifest_v5_hash": self.manifest_v5_hash,
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
        object.__setattr__(self, "calibration_v5_hash", _sealed(
            self.calibration_v5_hash, payload, "v3 calibration receipt hash",
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
class ActionRiskVectorV7V5:
    base_vector: v1.ActionRiskVectorV7
    assessor_model_hash: str
    assessor_model_provenance_hash: str
    assessor_algorithm_id: str
    assessor_algorithm_version: str
    assessor_feature_schema_hash: str
    assessor_config_hash: str
    calibration_v5_hash: str
    risk_vector_v5_hash: str = ""

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
            ("risk v2 calibration hash", self.calibration_v5_hash),
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
            "calibration_v5_hash": self.calibration_v5_hash,
        }
        object.__setattr__(self, "risk_vector_v5_hash", _sealed(
            self.risk_vector_v5_hash, payload, "v2 risk-vector hash",
        ))


@dataclass(frozen=True)
class PortfolioDecisionV7V5:
    state: DecisionStateV7V5
    manifest_v5_hash: str
    proposal_hash: str
    calibration_v5_hash: str
    native_candidate_id: str
    selected_candidate_id: str | None
    safe_candidate_ids: tuple[str, ...]
    arm_reasons: tuple[tuple[str, DecisionReasonV7V5], ...]
    arm_audit_states: tuple[tuple[str, bool], ...]
    external_authority_mode: ExternalReplayAuthorityModeV7V5
    external_authority_hash: str
    external_verification_hash: str
    production_eligibility: ProductionEligibilityV7V5
    reason: DecisionReasonV7V5
    audit_count: int
    child_count: int
    decision_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.state, DecisionStateV7V5):
            raise ValueError("v5 portfolio state is untyped")
        if not isinstance(self.reason, DecisionReasonV7V5):
            raise ValueError("v2 decision reason is untyped")
        for name, value in (
            ("decision manifest hash", self.manifest_v5_hash),
            ("decision proposal hash", self.proposal_hash),
            ("decision calibration hash", self.calibration_v5_hash),
            ("decision native candidate id", self.native_candidate_id),
            ("decision external authority hash", self.external_authority_hash),
            ("decision external verification hash",
             self.external_verification_hash),
        ):
            _text(value, name)
        if not isinstance(
            self.external_authority_mode, ExternalReplayAuthorityModeV7V5,
        ) or not isinstance(
            self.production_eligibility, ProductionEligibilityV7V5,
        ):
            raise ValueError("decision authority disposition is untyped")
        if self.audit_count not in {0, 1} or self.child_count not in {0, 1}:
            raise ValueError("v2 audit/child counts are bounded by one")
        if (len({row[0] for row in self.arm_reasons}) != len(self.arm_reasons)
                or any(not row[0] or not isinstance(row[1], DecisionReasonV7V5)
                       for row in self.arm_reasons)):
            raise ValueError("per-arm reasons must be typed and unique")
        if (len({row[0] for row in self.arm_audit_states})
                != len(self.arm_audit_states)
                or any(not row[0] or not isinstance(row[1], bool)
                       for row in self.arm_audit_states)):
            raise ValueError("per-arm audit states must be boolean and unique")
        if sum(value for _, value in self.arm_audit_states) > 1:
            raise ValueError("at most one selected arm may receive audit")
        if self.state is DecisionStateV7V5.COMMIT:
            if (not self.selected_candidate_id
                    or self.selected_candidate_id == self.native_candidate_id):
                raise ValueError("v2 commit needs one nonnative selection")
            if (self.external_authority_mode is not
                    ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED
                    or self.production_eligibility is not
                    ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE):
                raise ValueError("COMMIT requires production-trusted authority")
        elif self.state is DecisionStateV7V5.NATIVE:
            if self.selected_candidate_id != self.native_candidate_id:
                raise ValueError("v2 native decision must select canonical native")
            if not (
                (self.external_authority_mode is
                 ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED
                 and self.production_eligibility is
                 ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE)
                or (self.external_authority_mode is
                    ExternalReplayAuthorityModeV7V5.UNVERIFIED
                    and self.production_eligibility is
                    ProductionEligibilityV7V5.UNVERIFIED_FAIL_CLOSED)
            ):
                raise ValueError("production NATIVE authority disposition is invalid")
        elif self.state is DecisionStateV7V5.AUDIT:
            if (self.external_authority_mode is not
                    ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED
                    or self.production_eligibility is not
                    ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE):
                raise ValueError("AUDIT requires production-trusted authority")
        elif self.state in {
            DecisionStateV7V5.TEST_ONLY_WOULD_COMMIT,
            DecisionStateV7V5.TEST_ONLY_WOULD_NATIVE,
        }:
            if (self.external_authority_mode is not
                    ExternalReplayAuthorityModeV7V5.TEST_ONLY
                    or self.production_eligibility is not
                    ProductionEligibilityV7V5.TEST_ONLY_NONAUTHORITY):
                raise ValueError("synthetic decision lacks TEST_ONLY disposition")
            if (self.state is DecisionStateV7V5.TEST_ONLY_WOULD_COMMIT
                    and (not self.selected_candidate_id
                         or self.selected_candidate_id
                         == self.native_candidate_id)):
                raise ValueError("synthetic would-commit needs nonnative selection")
            if (self.state is DecisionStateV7V5.TEST_ONLY_WOULD_NATIVE
                    and self.selected_candidate_id != self.native_candidate_id):
                raise ValueError("synthetic would-native must select native")
        payload = {
            "state": self.state.value,
            "manifest_v5_hash": self.manifest_v5_hash,
            "proposal_hash": self.proposal_hash,
            "calibration_v5_hash": self.calibration_v5_hash,
            "native_candidate_id": self.native_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "safe_candidate_ids": list(self.safe_candidate_ids),
            "arm_reasons": [[key, value.value] for key, value in self.arm_reasons],
            "arm_audit_states": [list(row) for row in self.arm_audit_states],
            "external_authority_mode": self.external_authority_mode.value,
            "external_authority_hash": self.external_authority_hash,
            "external_verification_hash": self.external_verification_hash,
            "production_eligibility": self.production_eligibility.value,
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
    reason: DecisionReasonV7V5,
    *,
    audit_count: int = 0,
    child_count: int = 0,
    arm_reasons: tuple[tuple[str, DecisionReasonV7V5], ...] = (),
    arm_audit_states: tuple[tuple[str, bool], ...] = (),
    force_unverified: bool = False,
) -> PortfolioDecisionV7V5:
    try:
        manifest_hash = str(manifest.manifest_v5_hash)
        native_id = str(manifest.native.candidate_id)
    except Exception:
        manifest_hash = "unbound-v2-manifest"
        native_id = "action0:native"
    proposal_hash = str(getattr(proposal, "proposal_hash", "") or
                        "unbound-v2-proposal")
    calibration_hash = str(
        getattr(calibration, "calibration_v5_hash", "")
        or "unbound-v2-calibration"
    )
    authority_mode = getattr(
        proposal, "external_authority_mode",
        ExternalReplayAuthorityModeV7V5.UNVERIFIED,
    )
    authority_hash = str(
        getattr(proposal, "external_authority_hash", "")
        or "unverified-external-authority"
    )
    verification_hash = str(
        getattr(proposal, "external_verification_hash", "")
        or "unverified-external-verification"
    )
    if force_unverified:
        state = DecisionStateV7V5.NATIVE
        authority_mode = ExternalReplayAuthorityModeV7V5.UNVERIFIED
        authority_hash = "rejected-nonauthority-proposal"
        verification_hash = "rejected-nonauthority-verification"
        eligibility = ProductionEligibilityV7V5.UNVERIFIED_FAIL_CLOSED
    elif authority_mode is ExternalReplayAuthorityModeV7V5.TEST_ONLY:
        state = DecisionStateV7V5.TEST_ONLY_WOULD_NATIVE
        eligibility = ProductionEligibilityV7V5.TEST_ONLY_NONAUTHORITY
    elif authority_mode is ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED:
        state = DecisionStateV7V5.NATIVE
        eligibility = ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE
    else:
        state = DecisionStateV7V5.NATIVE
        authority_mode = ExternalReplayAuthorityModeV7V5.UNVERIFIED
        eligibility = ProductionEligibilityV7V5.UNVERIFIED_FAIL_CLOSED
    return PortfolioDecisionV7V5(
        state=state,
        manifest_v5_hash=manifest_hash,
        proposal_hash=proposal_hash,
        calibration_v5_hash=calibration_hash,
        native_candidate_id=native_id,
        selected_candidate_id=native_id,
        safe_candidate_ids=(),
        arm_reasons=arm_reasons,
        arm_audit_states=arm_audit_states,
        external_authority_mode=authority_mode,
        external_authority_hash=authority_hash,
        external_verification_hash=verification_hash,
        production_eligibility=eligibility,
        reason=reason,
        audit_count=audit_count if audit_count in {0, 1} else 1,
        child_count=child_count if child_count in {0, 1} else 1,
    )


def _validate_runtime_ledger(
    manifest: CandidateManifestV7V5,
    proposal: PlannerProposalV7V5,
    ledger: RuntimeCostLedgerV7V5,
    plan: AliasRunPlanV7V5,
    group_members: tuple[str, ...],
    by_realization: dict[str, v1.ArmRealizationReceiptV7],
    by_arm: dict[str, v1.CandidateArmV7],
    alias_by_id: dict[str, v1.ArmAliasReceiptV7],
) -> None:
    representative = plan.representative_planned_arm_id
    aliases = tuple(row for row in group_members if row != representative)
    if (ledger.manifest_v5_hash != manifest.manifest_v5_hash
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
    by_operation: dict[tuple[str, RuntimeOperationV7V5], list[
        RuntimeExecutionReceiptV7V5
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
        if receipt.operation is RuntimeOperationV7V5.EXECUTE_ARM:
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
        if operation is RuntimeOperationV7V5.EXECUTE_ARM and len(values) == 1
    }
    verify_ids = {
        planned_id for (planned_id, operation), values in by_operation.items()
        if operation is RuntimeOperationV7V5.VERIFY_ALIAS_REUSE and len(values) == 1
    }
    if sum(len(values) for values in by_operation.values()) != len(by_operation):
        raise ValueError("runtime arm operation was charged more than once")
    if plan.mode is AliasChronologyModeV7V5.PRE_RUN_FROZEN_REUSE:
        seal = plan.pre_run_alias_seal
        if (seal is None
                or seal.sealed_at_ns >= min(row.started_at_ns for row in receipts)
                or execute_ids != {representative}
                or verify_ids != set(aliases)
                or ledger.post_hoc_discovered_at_ns is not None):
            raise ValueError("pre-run reuse chronology is invalid")
    elif plan.mode in {
        AliasChronologyModeV7V5.POST_HOC_DISCOVERY,
        AliasChronologyModeV7V5.SAME_RUN_DUPLICATE,
    }:
        if execute_ids != set(group_members) or verify_ids:
            raise ValueError("duplicate chronology must charge every executed arm")
        if plan.mode is AliasChronologyModeV7V5.POST_HOC_DISCOVERY:
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


def _select_portfolio_v7_v5_impl(
    manifest: CandidateManifestV7V5,
    proposal: PlannerProposalV7V5,
    observations: Sequence[CandidateObservationV7V5],
    risks: Sequence[ActionRiskVectorV7V5],
    calibration: CalibrationReceiptV7V5,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    runtime_cost_ledgers: Sequence[RuntimeCostLedgerV7V5],
    *,
    audit_count: int = 0,
    child_count: int = 0,
    external_replay_authority: ExternalInferenceReplayAuthorityV7V5 | None,
    external_replay_verification: (
        ExternalInferenceReplayVerificationV7V5 | None
    ),
    trusted_root_review_seal_hashes: frozenset[str],
    trusted_verification_hashes: frozenset[str],
    allow_test_only: bool,
) -> PortfolioDecisionV7V5:
    """Replay frozen authorities and inspect only selected top-K runtime rows."""
    if (isinstance(audit_count, bool) or not isinstance(audit_count, int)
            or audit_count < 0 or audit_count > 1):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V5.AUDIT_BUDGET_EXHAUSTED,
            audit_count=1, child_count=child_count,
        )
    if (isinstance(child_count, bool) or not isinstance(child_count, int)
            or child_count < 0 or child_count > 1):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V5.CHILD_BUDGET_EXHAUSTED,
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
            *realizations, *aliases, *selected_ledgers,
        )
    except Exception:
        return _native_decision(
            manifest, proposal, calibration, DecisionReasonV7V5.MALFORMED_INPUT,
            audit_count=audit_count, child_count=child_count,
        )
    if any(not _rehash_valid(row) for row in objects):
        return _native_decision(
            manifest, proposal, calibration, DecisionReasonV7V5.MALFORMED_INPUT,
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
            DecisionReasonV7V5.NATIVE_IDENTITY_INVALID,
            audit_count=audit_count, child_count=child_count,
        )
    authority = manifest.planner_model_receipt
    try:
        _validate_external_replay_v7_v5(
            manifest, proposal.score_bank.score_receipts,
            external_replay_authority,  # type: ignore[arg-type]
            external_replay_verification,  # type: ignore[arg-type]
            trusted_root_review_seal_hashes=trusted_root_review_seal_hashes,
            trusted_verification_hashes=trusted_verification_hashes,
            allow_test_only=allow_test_only,
        )
    except Exception:
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V5.PLANNER_AUTHORITY_MISMATCH,
            audit_count=audit_count, child_count=child_count,
        )
    if (proposal.score_bank.planner_model_receipt.receipt_hash
            != authority.receipt_hash
            or proposal.external_authority_mode
            is not external_replay_authority.mode
            or proposal.external_authority_hash
            != external_replay_authority.authority_hash
            or proposal.external_verification_hash
            != external_replay_verification.verification_hash
            or proposal.production_eligibility is not (
                ProductionEligibilityV7V5.TEST_ONLY_NONAUTHORITY
                if allow_test_only else
                ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE
            )
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
            is not manifest.base_manifest.fold_role
            or authority.feature_schema_hash
            != authority.feature_schema_receipt.schema_hash
            or authority.required_e235_replay_authority_hash == ""):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V5.PLANNER_AUTHORITY_MISMATCH,
            audit_count=audit_count, child_count=child_count,
        )
    try:
        replayed = _propose_candidates_v7_v5_impl(
            manifest, proposal.score_bank.score_receipts, realizations, aliases,
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
            external_replay_authority=external_replay_authority,
            external_replay_verification=external_replay_verification,
            trusted_root_review_seal_hashes=trusted_root_review_seal_hashes,
            trusted_verification_hashes=trusted_verification_hashes,
            allow_test_only=allow_test_only,
        )
    except ValueError as error:
        reason = (
            DecisionReasonV7V5.SCORE_BANK_INCOMPLETE
            if "score bank" in str(error) else
            DecisionReasonV7V5.PLANNER_AUTHORITY_MISMATCH
            if "planner" in str(error) and "author" in str(error) else
            DecisionReasonV7V5.PROPOSAL_REPLAY_MISMATCH
        )
        return _native_decision(
            manifest, proposal, calibration, reason,
            audit_count=audit_count, child_count=child_count,
        )
    if replayed.proposal_hash != proposal.proposal_hash:
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V5.PROPOSAL_REPLAY_MISMATCH,
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
            DecisionReasonV7V5.PROPOSAL_REPLAY_MISMATCH,
            audit_count=audit_count, child_count=child_count,
        )
    by_arm, _ = _manifest_arm_maps(manifest)
    universe_ids = {row.planned_arm_id for row in universe}
    observations_by_id: dict[str, list[CandidateObservationV7V5]] = {}
    for row in selected_observations:
        observations_by_id.setdefault(row.planned_arm_id, []).append(row)
    risks_by_id: dict[str, list[ActionRiskVectorV7V5]] = {}
    for row in selected_risks:
        risks_by_id.setdefault(row.base_vector.planned_arm_id, []).append(row)
    expected_family_digest = calibration_choice_family_digest_v7_v5(
        manifest, proposal,
    )
    base_calibration = calibration.base_receipt
    family = calibration.choice_family_receipt
    if (calibration.manifest_v5_hash != manifest.manifest_v5_hash
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
            or family.manifest_v5_hash != manifest.manifest_v5_hash
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
            or family.frozen_before_evaluation_runtime is not True
            or base_calibration.calibration_fold_id not in
            authority.fit_receipt.split_authority.calibration_fold_ids
            or base_calibration.calibration_fold_id
            != authority.fit_receipt.split_authority.calibration_fold_id
            or authority.fit_receipt.target_fold_id not in
            authority.fit_receipt.split_authority.evaluation_fold_ids):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V5.CALIBRATION_FAMILY_INCOMPLETE,
            audit_count=audit_count, child_count=child_count,
        )
    if (base_calibration.availability is not v1.AvailabilityV7.AVAILABLE
            or not base_calibration.frozen_before_decision):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V5.ASSESSOR_CALIBRATION_MISMATCH,
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
            DecisionReasonV7V5.ALIAS_CHRONOLOGY_INVALID,
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
            DecisionReasonV7V5.ALIAS_CHRONOLOGY_INVALID,
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
            DecisionReasonV7V5.COST_BUDGET_EXCEEDED,
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
            DecisionReasonV7V5.COST_BUDGET_EXCEEDED,
            audit_count=audit_count, child_count=child_count,
        )
    score_by_id = {
        row.planned_arm_id: row for row in proposal.score_bank.scores
    }
    arm_reasons: dict[str, DecisionReasonV7V5] = {}
    arm_audit_states: dict[str, bool] = {}
    eligible_risks: dict[str, v1.ActionRiskVectorV7] = {}
    for planned_id in proposal.selected_planned_arm_ids:
        arm = by_arm[planned_id]
        observation_rows = observations_by_id.get(planned_id, [])
        risk_rows = risks_by_id.get(planned_id, [])
        if len(observation_rows) != 1 or len(risk_rows) != 1:
            arm_reasons[planned_id] = DecisionReasonV7V5.FEATURE_INELIGIBLE
            arm_audit_states[planned_id] = audit_count == 0
            continue
        observation = observation_rows[0]
        risk = risk_rows[0]
        if not _rehash_valid(observation) or not _rehash_valid(risk):
            arm_reasons[planned_id] = DecisionReasonV7V5.FEATURE_INELIGIBLE
            arm_audit_states[planned_id] = audit_count == 0
            continue
        base_risk = risk.base_vector
        realization = by_realization[planned_id]
        if base_risk.availability is not v1.AvailabilityV7.AVAILABLE:
            arm_reasons[planned_id] = DecisionReasonV7V5.FEATURE_INELIGIBLE
            arm_audit_states[planned_id] = audit_count == 0
            continue
        if (observation.manifest_v5_hash != manifest.manifest_v5_hash
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
                ObservationSourceV7V5.PROSPECTIVE_RUNTIME_OBSERVATION
                or observation.runtime_cost_ledger_hash
                != ledgers[planned_id].ledger_hash):
            return _native_decision(
                manifest, proposal, calibration,
                DecisionReasonV7V5.ACTION_BINDING_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
        if (observation.availability is not v1.AvailabilityV7.AVAILABLE
                or any(row.availability is not v1.AvailabilityV7.AVAILABLE
                       for row in observation.evidence_blocks)):
            arm_reasons[planned_id] = DecisionReasonV7V5.FEATURE_INELIGIBLE
            arm_audit_states[planned_id] = audit_count == 0
            continue
        if arm.scope is v1.CandidateScopeV7.CASE_ATOMIC:
            if (not observation.atomically_observed
                    or set(observation.observed_unit_ids) != set(arm.unit_ids)):
                arm_reasons[planned_id] = (
                    DecisionReasonV7V5.GLOBAL_PARTIAL_COMMIT
                )
                arm_audit_states[planned_id] = audit_count == 0
                continue
        elif (arm.scope is not v1.CandidateScopeV7.CERTIFIED_LOCAL
              or len(arm.unit_ids) != 1
              or set(observation.observed_unit_ids) != set(arm.unit_ids)):
            arm_reasons[planned_id] = DecisionReasonV7V5.LOCAL_SUPPORT_MISMATCH
            arm_audit_states[planned_id] = audit_count == 0
            continue
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
            == calibration.calibration_v5_hash
            and risk.calibration_v5_hash == calibration.calibration_v5_hash
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
                DecisionReasonV7V5.ASSESSOR_CALIBRATION_MISMATCH,
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
                DecisionReasonV7V5.ACTION_BINDING_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
        if (not base_risk.feature_eligible or not base_risk.calibration_valid
                or not base_risk.bound_bookkeeping_valid):
            arm_reasons[planned_id] = DecisionReasonV7V5.FEATURE_INELIGIBLE
            arm_audit_states[planned_id] = audit_count == 0
            continue
        eligible_risks[planned_id] = base_risk
    allocated_audit = False
    for planned_id in proposal.selected_planned_arm_ids:
        if arm_audit_states.get(planned_id, False):
            if allocated_audit:
                arm_audit_states[planned_id] = False
            else:
                allocated_audit = True
    if not selected_ids:
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V5.NO_SAFE_CANDIDATE,
            audit_count=audit_count, child_count=child_count,
        )
    safe: list[tuple[v1.CandidateArmV7, v1.ActionRiskVectorV7, float]] = []
    risk_rejected = False
    utility_rejected = False
    for planned_id in proposal.selected_planned_arm_ids:
        if planned_id not in eligible_risks:
            continue
        arm = by_arm[planned_id]
        risk = eligible_risks[planned_id]
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
            arm_reasons[planned_id] = (
                DecisionReasonV7V5.RISK_BUDGET_EXCEEDED
                if not separate_risk_pass else
                DecisionReasonV7V5.NONPOSITIVE_CONSERVATIVE_UTILITY
            )
            arm_audit_states[planned_id] = False
    safe.sort(key=lambda item: (
        -item[2], float(item[1].any_row_severe_probability_upper),
        float(item[1].pixel_harm_cvar95_upper),
        group_costs[item[0].planned_arm_id][0],
        item[0].action_identity, item[0].planned_arm_id,
    ))
    if not safe:
        reason = (
            DecisionReasonV7V5.RISK_BUDGET_EXCEEDED if risk_rejected
            else DecisionReasonV7V5.NONPOSITIVE_CONSERVATIVE_UTILITY
            if utility_rejected else DecisionReasonV7V5.NO_SAFE_CANDIDATE
        )
        return _native_decision(
            manifest, proposal, calibration, reason,
            audit_count=audit_count, child_count=child_count,
            arm_reasons=tuple(sorted(arm_reasons.items())),
            arm_audit_states=tuple(sorted(arm_audit_states.items())),
        )
    selected = safe[0][0]
    is_test_only = proposal.external_authority_mode is (
        ExternalReplayAuthorityModeV7V5.TEST_ONLY
    )
    return PortfolioDecisionV7V5(
        state=(
            DecisionStateV7V5.TEST_ONLY_WOULD_COMMIT
            if is_test_only else DecisionStateV7V5.COMMIT
        ),
        manifest_v5_hash=manifest.manifest_v5_hash,
        proposal_hash=proposal.proposal_hash,
        calibration_v5_hash=calibration.calibration_v5_hash,
        native_candidate_id=manifest.native.candidate_id,
        selected_candidate_id=selected.candidate_id,
        safe_candidate_ids=tuple(row[0].candidate_id for row in safe),
        arm_reasons=tuple(sorted(arm_reasons.items())),
        arm_audit_states=tuple(sorted(arm_audit_states.items())),
        external_authority_mode=proposal.external_authority_mode,
        external_authority_hash=proposal.external_authority_hash,
        external_verification_hash=proposal.external_verification_hash,
        production_eligibility=proposal.production_eligibility,
        reason=DecisionReasonV7V5.SAFE_SET_MAXIMUM,
        audit_count=audit_count,
        child_count=child_count,
    )


def select_portfolio_v7_v5(
    manifest: CandidateManifestV7V5,
    proposal: PlannerProposalV7V5,
    observations: Sequence[CandidateObservationV7V5],
    risks: Sequence[ActionRiskVectorV7V5],
    calibration: CalibrationReceiptV7V5,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    runtime_cost_ledgers: Sequence[RuntimeCostLedgerV7V5],
    *,
    audit_count: int = 0,
    child_count: int = 0,
    external_replay_authority: (
        ExternalInferenceReplayAuthorityV7V5 | None
    ) = None,
    external_replay_verification: (
        ExternalInferenceReplayVerificationV7V5 | None
    ) = None,
    trusted_root_review_seal_hashes: frozenset[str] = frozenset(),
    trusted_verification_hashes: frozenset[str] = frozenset(),
) -> PortfolioDecisionV7V5:
    """Production commit boundary; absent external E235 trust returns native."""
    if (not isinstance(proposal, PlannerProposalV7V5)
            or proposal.external_authority_mode is not
            ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED
            or proposal.production_eligibility is not
            ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE):
        return _native_decision(
            manifest, proposal, calibration,
            DecisionReasonV7V5.PLANNER_AUTHORITY_MISMATCH,
            audit_count=audit_count, child_count=child_count,
            force_unverified=True,
        )
    return _select_portfolio_v7_v5_impl(
        manifest, proposal, observations, risks, calibration, realizations,
        aliases, runtime_cost_ledgers, audit_count=audit_count,
        child_count=child_count,
        external_replay_authority=external_replay_authority,
        external_replay_verification=external_replay_verification,
        trusted_root_review_seal_hashes=trusted_root_review_seal_hashes,
        trusted_verification_hashes=trusted_verification_hashes,
        allow_test_only=False,
    )


def select_portfolio_v7_v5_test_only(
    manifest: CandidateManifestV7V5,
    proposal: PlannerProposalV7V5,
    observations: Sequence[CandidateObservationV7V5],
    risks: Sequence[ActionRiskVectorV7V5],
    calibration: CalibrationReceiptV7V5,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    runtime_cost_ledgers: Sequence[RuntimeCostLedgerV7V5],
    *,
    external_replay_authority: ExternalInferenceReplayAuthorityV7V5,
    external_replay_verification: ExternalInferenceReplayVerificationV7V5,
    audit_count: int = 0,
    child_count: int = 0,
) -> PortfolioDecisionV7V5:
    """Synthetic-only selector; the production entry point rejects its receipts."""
    if (proposal.external_authority_mode is not
            ExternalReplayAuthorityModeV7V5.TEST_ONLY
            or proposal.production_eligibility is not
            ProductionEligibilityV7V5.TEST_ONLY_NONAUTHORITY):
        raise ValueError("test-only selector requires TEST_ONLY_NONAUTHORITY proposal")
    return _select_portfolio_v7_v5_impl(
        manifest, proposal, observations, risks, calibration, realizations,
        aliases, runtime_cost_ledgers, audit_count=audit_count,
        child_count=child_count,
        external_replay_authority=external_replay_authority,
        external_replay_verification=external_replay_verification,
        trusted_root_review_seal_hashes=frozenset(),
        trusted_verification_hashes=frozenset(),
        allow_test_only=True,
    )


def verify_production_proposal_v7_v5(
    proposal: object,
    *,
    trusted_authority_hashes: frozenset[str],
    trusted_verification_hashes: frozenset[str],
) -> bool:
    """Production consumer gate; TEST_ONLY bytes are categorically rejected."""
    return bool(
        isinstance(proposal, PlannerProposalV7V5)
        and _rehash_valid(proposal)
        and proposal.external_authority_mode is
        ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED
        and proposal.production_eligibility is
        ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE
        and proposal.external_authority_hash in trusted_authority_hashes
        and proposal.external_verification_hash in trusted_verification_hashes
    )


def serialize_production_proposal_v7_v5(
    proposal: object,
    *,
    trusted_authority_hashes: frozenset[str],
    trusted_verification_hashes: frozenset[str],
) -> bytes:
    if not verify_production_proposal_v7_v5(
        proposal,
        trusted_authority_hashes=trusted_authority_hashes,
        trusted_verification_hashes=trusted_verification_hashes,
    ):
        raise ValueError("proposal is not trusted production authority")
    assert isinstance(proposal, PlannerProposalV7V5)
    return json.dumps({
        "schema_version": SELECTOR_V7_V5_SCHEMA_VERSION,
        "proposal_hash": proposal.proposal_hash,
        "external_authority_mode": proposal.external_authority_mode.value,
        "external_authority_hash": proposal.external_authority_hash,
        "external_verification_hash": proposal.external_verification_hash,
        "production_eligibility": proposal.production_eligibility.value,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def verify_production_decision_v7_v5(
    decision: object,
    *,
    trusted_proposal_hashes: frozenset[str],
    trusted_authority_hashes: frozenset[str],
    trusted_verification_hashes: frozenset[str],
) -> bool:
    """Reject synthetic outcomes even if their remaining hashes are well formed."""
    return bool(
        isinstance(decision, PortfolioDecisionV7V5)
        and _rehash_valid(decision)
        and decision.state in {
            DecisionStateV7V5.COMMIT,
            DecisionStateV7V5.AUDIT,
            DecisionStateV7V5.NATIVE,
        }
        and decision.external_authority_mode is
        ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED
        and decision.production_eligibility is
        ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE
        and decision.proposal_hash in trusted_proposal_hashes
        and decision.external_authority_hash in trusted_authority_hashes
        and decision.external_verification_hash in trusted_verification_hashes
    )


def serialize_production_decision_v7_v5(
    decision: object,
    *,
    trusted_proposal_hashes: frozenset[str],
    trusted_authority_hashes: frozenset[str],
    trusted_verification_hashes: frozenset[str],
) -> bytes:
    if not verify_production_decision_v7_v5(
        decision,
        trusted_proposal_hashes=trusted_proposal_hashes,
        trusted_authority_hashes=trusted_authority_hashes,
        trusted_verification_hashes=trusted_verification_hashes,
    ):
        raise ValueError("decision is not trusted production authority")
    assert isinstance(decision, PortfolioDecisionV7V5)
    return json.dumps({
        "schema_version": SELECTOR_V7_V5_SCHEMA_VERSION,
        "state": decision.state.value,
        "decision_hash": decision.decision_hash,
        "proposal_hash": decision.proposal_hash,
        "external_authority_mode": decision.external_authority_mode.value,
        "external_authority_hash": decision.external_authority_hash,
        "external_verification_hash": decision.external_verification_hash,
        "production_eligibility": decision.production_eligibility.value,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def selector_v7_v5_schema_fingerprint() -> str:
    classes = (
        SelectorV7V5Disposition, FiveFoldPartitionV7V5,
        FiveFoldSplitAuthorityRootV7V5, SplitAuthorityReceiptV7V5,
        PlannerFeatureSchemaReceiptV7V5,
        ExternalInferenceReplayAuthorityV7V5, PlannerFitReceiptV7V5,
        PlannerModelReceiptV7V5, ImmutableNativeReceiptV7V5,
        ManifestFreezeReceiptV7V5, CalibrationChoiceFamilyReceiptV7V5,
        CandidateManifestV7V5, PreRunAliasSealV7V5, AliasRunPlanV7V5,
        RuntimeExecutionReceiptV7V5, RuntimeCostLedgerV7V5,
        BeforeFeatureArtifactReceiptV7V5,
        PlannerInferenceExecutionReceiptV7V5,
        PlannerScoreReceiptV7V5, ExternalInferenceReplayVerificationV7V5,
        PlannerScoreBankV7V5, PlannerProposalV7V5,
        CandidateObservationV7V5, CalibrationReceiptV7V5,
        ActionRiskVectorV7V5, PortfolioDecisionV7V5,
    )
    enums = (
        DecisionReasonV7V5, AliasChronologyModeV7V5,
        RuntimeOperationV7V5, ObservationSourceV7V5,
        ExternalReplayAuthorityModeV7V5, ProductionEligibilityV7V5,
        DecisionStateV7V5,
    )
    return _sha256({
        "schema_version": SELECTOR_V7_V5_SCHEMA_VERSION,
        "canonicalization_version": SELECTOR_V7_V5_CANONICALIZATION_VERSION,
        "predecessor_failed_review_sha256": (
            SELECTOR_V7_V4_REVIEWED_SHA256
        ),
        "predecessor_failed_test_sha256": SELECTOR_V7_V4_REVIEWED_TEST_SHA256,
        "v4_root_review_sha256": SR1_ROOT_REVIEW_V4_SHA256,
        "v4_root_review_status": SR1_ROOT_REVIEW_V4_STATUS,
        "v2_root_review_sha256": SR1_ROOT_REVIEW_V2_SHA256,
        "v2_root_review_status": SR1_ROOT_REVIEW_V2_STATUS,
        "v3_root_review_sha256": SR1_ROOT_REVIEW_V3_SHA256,
        "v3_root_review_status": SR1_ROOT_REVIEW_V3_STATUS,
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
        "base_v3_schema_fingerprint": v3.selector_v7_v3_schema_fingerprint(),
        "base_v4_schema_fingerprint": v4.selector_v7_v4_schema_fingerprint(),
    })


__all__ = [
    "ActionRiskVectorV7V5",
    "AliasChronologyModeV7V5",
    "AliasRunPlanV7V5",
    "BeforeFeatureArtifactReceiptV7V5",
    "CalibrationChoiceFamilyReceiptV7V5",
    "CalibrationReceiptV7V5",
    "CandidateManifestV7V5",
    "CandidateObservationV7V5",
    "DecisionReasonV7V5",
    "DecisionStateV7V5",
    "E235_OUTPUT_REPLAY_SCHEMA_HASH_V7V5",
    "ExternalInferenceReplayAuthorityV7V5",
    "ExternalInferenceReplayVerificationV7V5",
    "ExternalReplayAuthorityModeV7V5",
    "FiveFoldPartitionV7V5",
    "FiveFoldSplitAuthorityRootV7V5",
    "ImmutableNativeReceiptV7V5",
    "ManifestFreezeReceiptV7V5",
    "ObservationSourceV7V5",
    "PlannerProposalV7V5",
    "PlannerFitReceiptV7V5",
    "PlannerFeatureSchemaReceiptV7V5",
    "PlannerInferenceExecutionReceiptV7V5",
    "PlannerModelReceiptV7V5",
    "PlannerScoreBankV7V5",
    "PlannerScoreReceiptV7V5",
    "PortfolioDecisionV7V5",
    "ProductionEligibilityV7V5",
    "PreRunAliasSealV7V5",
    "RuntimeCostLedgerV7V5",
    "RuntimeExecutionReceiptV7V5",
    "RuntimeOperationV7V5",
    "SplitAuthorityReceiptV7V5",
    "SELECTOR_V7_V1_FAILED_REVIEW_SHA256",
    "SELECTOR_V7_V1_FAILED_TEST_SHA256",
    "SELECTOR_V7_V2_REVIEWED_SHA256",
    "SELECTOR_V7_V2_REVIEWED_TEST_SHA256",
    "SELECTOR_V7_V3_REVIEWED_SHA256",
    "SELECTOR_V7_V3_REVIEWED_TEST_SHA256",
    "SELECTOR_V7_V4_REVIEWED_SHA256",
    "SELECTOR_V7_V4_REVIEWED_TEST_SHA256",
    "SELECTOR_V7_V5_CANONICALIZATION_VERSION",
    "SELECTOR_V7_V5_SCHEMA_VERSION",
    "SR1_ROOT_REVIEW_V1_SHA256",
    "SR1_ROOT_REVIEW_V2_SHA256",
    "SR1_ROOT_REVIEW_V2_STATUS",
    "SR1_ROOT_REVIEW_V3_SHA256",
    "SR1_ROOT_REVIEW_V3_STATUS",
    "SR1_ROOT_REVIEW_V4_SHA256",
    "SR1_ROOT_REVIEW_V4_STATUS",
    "PLANNER_ALGORITHM_ID_V7V5",
    "PLANNER_ALGORITHM_VERSION_V7V5",
    "PLANNER_CONFIG_HASH_V7V5",
    "PLANNER_SCHEMA_CONFIG_HASH_V7V5",
    "PLANNER_SCHEMA_PRODUCER_CODE_HASH_V7V5",
    "PLANNER_MODEL_SCHEMA_HASH_V7V5",
    "PLANNER_TARGET_ID_V7V5",
    "SelectorV7V5Disposition",
    "calibration_choice_family_digest_v7_v5",
    "make_planner_score_receipt_v7_v5",
    "planner_inference_input_identity_hash_v7_v5",
    "planner_model_identity_hash_v7_v5",
    "planner_score_output_bytes_hash_v7_v5",
    "propose_candidates_v7_v5",
    "select_portfolio_v7_v5",
    "serialize_production_decision_v7_v5",
    "serialize_production_proposal_v7_v5",
    "selector_v7_v5_schema_fingerprint",
    "verify_production_decision_v7_v5",
    "verify_production_proposal_v7_v5",
]
