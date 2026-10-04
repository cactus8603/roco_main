"""Outcome-blind contracts for the Selector-v7 plan/observe/commit boundary.

This module implements the v4.39 amendment as a pure CPU contract.  It does not execute an
action, decode a target, train an assessor, or infer a corruption family.  A
planner may spend observation cost on at most two arms from an immutable
manifest.  The committer then re-ranks only those arms with separately
calibrated benefit, harm, severe-tail, harmed-fraction, and CVaR bounds.

Native is a mandatory zero-cost action-0 arm.  Any malformed binding, missing
required evidence, non-finite number, unknown interaction, partial global
observation, or cost overflow makes the whole decision fail closed to native.
Hard-rejected arms can never be reopened by a planner or assessor.  Primary
execution units are one CASE_ATOMIC unit or one CERTIFIED_LOCAL unit; v7.39
does not deliver multi-local compositions.  Its numerical safety thresholds,
component-max calibration rule, K1/K2 profiles, and matched-cost counters are
frozen in this module rather than accepted as tunable runtime policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
from itertools import permutations
from typing import Iterable, Sequence


class AvailabilityV7(str, Enum):
    AVAILABLE = "available"
    UNKNOWN_UNOBSERVED_ARM = "unknown_unobserved_arm"
    TYPED_MISSING = "typed_missing"
    OOD = "ood"
    LOW_SUPPORT = "low_support"


class FoldRoleV7(str, Enum):
    FIT = "fit"
    CAL = "cal"
    EVAL = "eval"
    RUNTIME = "runtime"


class CandidateScopeV7(str, Enum):
    NATIVE = "native"
    CASE_ATOMIC = "case_atomic"
    CERTIFIED_LOCAL = "certified_local"


class CandidateExclusionStateV7(str, Enum):
    HARD_ILLEGAL_PREEXECUTION = "hard_illegal_preexecution"
    SOFT_NOT_PROPOSED = "soft_not_proposed"
    OBSERVED_BUT_FEATURE_INELIGIBLE = "observed_but_feature_ineligible"


class PlannerProfileV7(str, Enum):
    K1_ABLATION = "k1_ablation"
    K2_PRIMARY = "k2_primary"


class PredictionKindV7(str, Enum):
    PLANNER_MODEL_PREDICTION_PREDECISION = (
        "planner_model_prediction_predecision"
    )
    ASSESSOR_MODEL_PREDICTION_PREDECISION = (
        "assessor_model_prediction_predecision"
    )


class SevereEstimandV7(str, Enum):
    EXECUTION_UNIT_ANY_ROW_SEVERE = "execution_unit_any_row_severe"


class EvidenceBlockV7(str, Enum):
    RESPONSE_ALPHA = "response_alpha"
    CC_TRANSITION = "cc_transition"
    E233_CYCLE_G0 = "e233_cycle_g0"
    PHYSICAL_SUPPORT = "physical_support"


class FeatureStageV7(str, Enum):
    PREACTION_SHARED = "preaction_shared"
    PLANNER_EXTENSION = "planner_extension"
    ASSESSOR_POSTACTION_EXTENSION = "assessor_postaction_extension"


class FeatureDependencyKindV7(str, Enum):
    TYPED_ACTION = "typed_action"
    SOURCE_INPUT = "source_input"
    NATIVE_OBSERVATION = "native_observation"
    PREEXECUTION_RECEIPT = "preexecution_receipt"
    ARM_REALIZATION = "arm_realization"
    POST_ACTION_OBSERVATION = "post_action_observation"


class ArmRealizationStatusV7(str, Enum):
    COMPLETE = "complete"
    UNKNOWN_UNOBSERVED_ARM = "unknown_unobserved_arm"
    FAILED = "failed"


class CostAccountV7(str, Enum):
    OFFLINE_BANK_ACQUISITION = "offline_bank_acquisition"
    PROSPECTIVE_RUNTIME = "prospective_runtime"
    TRAINING = "training"


class InteractionKindV7(str, Enum):
    DISJOINT = "disjoint"
    BOUNDED = "bounded"


class CalibrationMethodV7(str, Enum):
    FIXED_SINGLE = "fixed_single"
    SIMULTANEOUS_BOUND = "simultaneous_bound"
    WORST_COMPONENT = "worst_component"
    COMPONENT_MAX_RESIDUAL = "component_max_standardized_one_sided_residual"


class PortfolioStateV7(str, Enum):
    COMMIT = "commit"
    AUDIT = "audit"
    NATIVE = "native"


class HardRejectReasonV7(str, Enum):
    ILLEGAL_LINEAGE = "illegal_lineage"
    HASH_DRIFT = "hash_drift"
    NONRECONSTRUCTIBLE = "nonreconstructible_from_native"
    OUTSIDE_SUPPORT = "outside_support"
    UNKNOWN_INTERACTION = "unknown_interaction"
    COST_UNPROVEN = "cost_unproven"
    REQUIRED_EVIDENCE_MISSING = "required_evidence_missing"
    FORBIDDEN_RUNTIME_FIELD = "forbidden_runtime_field"
    PRIMARY_MULTI_LOCAL_FORBIDDEN = "primary_multi_local_forbidden"


class DecisionReasonV7(str, Enum):
    SAFE_SET_MAXIMUM = "safe_set_maximum"
    AMBIGUITY_AUDIT = "ambiguity_audit"
    EMPTY_PROPOSAL = "empty_proposal"
    NO_SAFE_CANDIDATE = "no_safe_candidate"
    RISK_BUDGET_EXCEEDED = "risk_budget_exceeded"
    ANY_ROW_SEVERE_BUDGET_EXCEEDED = "any_row_severe_budget_exceeded"
    NONPOSITIVE_CONSERVATIVE_UTILITY = "nonpositive_conservative_utility"
    MANIFEST_NOT_FROZEN = "manifest_not_frozen"
    MANIFEST_DRIFT = "manifest_drift"
    PROPOSAL_NOT_FROZEN = "proposal_not_frozen"
    PROPOSAL_OVERFLOW = "proposal_overflow"
    CANDIDATE_INJECTION = "candidate_injection"
    HARD_REJECT_REOPEN_ATTEMPT = "hard_reject_reopen_attempt"
    ACTION_BINDING_MISMATCH = "action_binding_mismatch"
    CALIBRATION_BINDING_MISMATCH = "calibration_binding_mismatch"
    CALIBRATION_SELECTION_FAMILY_INCOMPLETE = (
        "calibration_selection_family_incomplete"
    )
    TYPED_MISSING = "typed_missing"
    UNKNOWN_UNOBSERVED_ARM = "unknown_unobserved_arm"
    OOD = "ood"
    LOW_SUPPORT = "low_support"
    FEATURE_INELIGIBLE = "feature_ineligible"
    CALIBRATION_INVALID = "calibration_invalid"
    NONFINITE_VALUE = "nonfinite_value"
    UNKNOWN_INTERACTION = "unknown_interaction"
    GLOBAL_PARTIAL_COMMIT = "global_partial_commit"
    LOCAL_SUPPORT_MISMATCH = "local_support_mismatch"
    COST_BUDGET_EXCEEDED = "cost_budget_exceeded"
    AUDIT_BUDGET_EXHAUSTED = "audit_budget_exhausted"
    CHILD_BUDGET_EXHAUSTED = "child_budget_exhausted"
    MALFORMED_INPUT = "malformed_input"


PLANNER_FEATURE_PROFILE_V7 = "PLANNER_BEFORE_ONLY_V7"
ASSESSOR_FEATURE_PROFILE_V7 = "ASSESSOR_POST_ACTION_NO_WORKB_V7"
V7_PRIMARY_FEATURE_PROFILE = ASSESSOR_FEATURE_PROFILE_V7
SELECTOR_V7_SCHEMA_VERSION = "selector-v7-contract/v4.39-draft"
SELECTOR_V7_CANONICALIZATION_VERSION = "selector-v7-canonical-json/v3"
RHO_ANY_ROW_SEVERE_V7 = 0.05
TAU_H_MEAN_RAW_PX_V7 = 0.05
TAU_HARMED_FRACTION_V7 = 0.05
TAU_CVAR95_RAW_PX_V7 = 0.25
ROW_SEVERE_CUTOFF_G_RAW_PX_V7 = -0.25
CVAR_LEVEL_V7 = 0.95
LAMBDA_COST_V7 = 0.0
CALIBRATION_ALPHA_COMPONENT_V7 = 0.10
MINIMUM_GLOBAL_CALIBRATION_COMPONENTS_V7 = 9
MINIMUM_FIT_COMPONENTS_PER_PRIMARY_CELL_V7 = 3
MINIMUM_CAL_COMPONENTS_PER_PRIMARY_CELL_V7 = 1
PRIMARY_HEAD_SCALE_PX_V7 = 0.25
PRIMARY_HEAD_SCALE_PROBABILITY_V7 = 1.0
PRIMARY_REQUIRED_EVIDENCE_BLOCKS_V7 = frozenset(EvidenceBlockV7)
PRIMARY_ASSESSOR_HEADS_V7 = (
    "benefit_mean_lower",
    "harm_mean_upper",
    "any_row_severe_probability_upper",
    "harmed_pixel_fraction_upper",
    "pixel_harm_CVaR95_upper",
)
PREACTION_SHARED_FIELDS_V7 = frozenset({
    "action_identity",
    "control_identity",
    "endpoint_identity",
    "input_strength",
    "requested_output_beta",
    "before_image_observable",
    "native_flow_observable",
    "cheap_before_evidence",
    "physical_preexecution_legality",
    "planned_prospective_runtime_cost",
})
PLANNER_EXTENSION_FIELDS_V7: frozenset[str] = frozenset()
ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7 = frozenset({
    "realized_response_alpha",
    "cc_before_full_delta",
    "e233_cycle_g0",
    "physical_support_realization",
})
PLANNER_BEFORE_ONLY_FIELDS_V7 = (
    PREACTION_SHARED_FIELDS_V7 | PLANNER_EXTENSION_FIELDS_V7
)
ASSESSOR_POST_ACTION_FIELDS_V7 = (
    PREACTION_SHARED_FIELDS_V7 | ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7
)

_FORBIDDEN_RUNTIME_TOKENS = frozenset({
    "corruption", "corrupt", "dataset", "scene", "path", "filepath",
    "file", "filename", "groundtruth", "ground_truth", "gt", "outcome",
    "target", "label", "severity_label",
})

_NONPRIMARY_RUNTIME_TOKENS = frozenset({
    "workb", "rr", "cr", "rc", "census", "v1", "learned_uncertainty",
})


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    item = float(value)
    if not math.isfinite(item):
        raise ValueError(f"{name} must be finite")
    return item


def _nonnegative(value: object, name: str) -> float:
    item = _finite(value, name)
    if item < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return item


def _fraction(value: object, name: str) -> float:
    item = _finite(value, name)
    if not 0.0 <= item <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return item


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _enum(value: object, cls: type[Enum], name: str) -> Enum:
    if not isinstance(value, cls):
        raise ValueError(f"{name} must be {cls.__name__}")
    return value


def _unique_nonempty(values: Sequence[str], name: str) -> tuple[str, ...]:
    rows = tuple(values)
    if not rows or any(not isinstance(item, str) or not item for item in rows):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(rows) != len(set(rows)):
        raise ValueError(f"{name} must be unique")
    return rows


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sealed_hash(provided: str, payload: object, name: str) -> str:
    expected = _sha256(payload)
    if provided and provided != expected:
        raise ValueError(f"{name} does not match its sealed content")
    return expected


def _forbidden_runtime_name(name: str) -> bool:
    normalized = name.strip().lower()
    tokens = set(filter(None, re.split(r"[^a-z0-9]+", normalized)))
    compact = "_".join(filter(None, re.split(r"[^a-z0-9]+", normalized)))
    return (
        bool(tokens & (_FORBIDDEN_RUNTIME_TOKENS | _NONPRIMARY_RUNTIME_TOKENS))
        or compact in (_FORBIDDEN_RUNTIME_TOKENS | _NONPRIMARY_RUNTIME_TOKENS)
    )


@dataclass(frozen=True)
class CostCountersV7:
    """Frozen counters for one explicitly named cost account."""

    natural_forwards: int
    reused_forwards: int
    candidate_forwards: int
    reverse_forwards: int
    cpu_seconds: float
    gpu_seconds: float
    wall_seconds: float
    bytes_moved: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    matcher_trajectories: int
    cache_state: str

    def __post_init__(self) -> None:
        for name in (
            "natural_forwards", "reused_forwards", "candidate_forwards",
            "reverse_forwards", "bytes_moved", "peak_allocated_bytes",
            "peak_reserved_bytes", "matcher_trajectories",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("cpu_seconds", "gpu_seconds", "wall_seconds"):
            _nonnegative(getattr(self, name), name)
        if self.peak_allocated_bytes > self.peak_reserved_bytes:
            raise ValueError("peak allocated bytes cannot exceed reserved bytes")
        if self.cache_state not in {"warm", "cold"}:
            raise ValueError("cache state must be frozen as warm or cold")

    def within(self, ceiling: "CostCountersV7") -> bool:
        names = (
            "natural_forwards", "reused_forwards", "candidate_forwards",
            "reverse_forwards", "cpu_seconds", "gpu_seconds", "wall_seconds",
            "bytes_moved", "peak_allocated_bytes", "peak_reserved_bytes",
            "matcher_trajectories",
        )
        return (
            self.cache_state == ceiling.cache_state
            and all(getattr(self, name) <= getattr(ceiling, name) + 1e-12
                    for name in names)
        )

    def plus(self, other: "CostCountersV7") -> "CostCountersV7":
        """Compose sequential costs; memory peaks remain maxima, not sums."""
        if not isinstance(other, CostCountersV7):
            raise ValueError("cost counters can only compose with cost counters")
        if self.cache_state != other.cache_state:
            raise ValueError("cost counters with different cache states cannot compose")
        return CostCountersV7(
            natural_forwards=self.natural_forwards + other.natural_forwards,
            reused_forwards=self.reused_forwards + other.reused_forwards,
            candidate_forwards=self.candidate_forwards + other.candidate_forwards,
            reverse_forwards=self.reverse_forwards + other.reverse_forwards,
            cpu_seconds=self.cpu_seconds + other.cpu_seconds,
            gpu_seconds=self.gpu_seconds + other.gpu_seconds,
            wall_seconds=self.wall_seconds + other.wall_seconds,
            bytes_moved=self.bytes_moved + other.bytes_moved,
            peak_allocated_bytes=max(
                self.peak_allocated_bytes, other.peak_allocated_bytes,
            ),
            peak_reserved_bytes=max(
                self.peak_reserved_bytes, other.peak_reserved_bytes,
            ),
            matcher_trajectories=(
                self.matcher_trajectories + other.matcher_trajectories
            ),
            cache_state=self.cache_state,
        )

    @classmethod
    def zero(cls, *, cache_state: str = "cold") -> "CostCountersV7":
        return cls(
            natural_forwards=0, reused_forwards=0, candidate_forwards=0,
            reverse_forwards=0, cpu_seconds=0.0, gpu_seconds=0.0,
            wall_seconds=0.0, bytes_moved=0, peak_allocated_bytes=0,
            peak_reserved_bytes=0, matcher_trajectories=0,
            cache_state=cache_state,
        )

    @property
    def content_hash(self) -> str:
        return _sha256({name: getattr(self, name) for name in self.__dataclass_fields__})


@dataclass(frozen=True)
class EvidenceBlockReceiptV7:
    block: EvidenceBlockV7
    availability: AvailabilityV7
    source_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        _enum(self.block, EvidenceBlockV7, "evidence block")
        _enum(self.availability, AvailabilityV7, "evidence block availability")
        _require_text(self.source_hash, "evidence block source hash")
        _require_text(self.receipt_hash, "evidence block receipt hash")


@dataclass(frozen=True)
class FeatureFieldV7:
    canonical_field_id: str
    name: str
    stage: FeatureStageV7
    available_at_stage: FeatureStageV7
    source_hash: str
    dependency_kind: FeatureDependencyKindV7
    dependency_hash: str

    def __post_init__(self) -> None:
        _require_text(self.canonical_field_id, "canonical feature field id")
        _require_text(self.name, "feature field name")
        if _forbidden_runtime_name(self.name):
            raise ValueError("feature allowlist contains a forbidden runtime field")
        _enum(self.stage, FeatureStageV7, "feature stage")
        _enum(self.available_at_stage, FeatureStageV7, "feature availability stage")
        _enum(self.dependency_kind, FeatureDependencyKindV7, "dependency kind")
        _require_text(self.source_hash, "feature source hash")
        _require_text(self.dependency_hash, "feature dependency hash")
        if self.stage is not self.available_at_stage:
            raise ValueError("feature stage and availability stage differ")


@dataclass(frozen=True)
class FeatureAllowlistsV7:
    preaction_shared_fields: tuple[FeatureFieldV7, ...]
    planner_extension_fields: tuple[FeatureFieldV7, ...]
    assessor_postaction_extension_fields: tuple[FeatureFieldV7, ...]
    planner_profile: str = PLANNER_FEATURE_PROFILE_V7
    assessor_profile: str = ASSESSOR_FEATURE_PROFILE_V7
    planner_allowlist_hash: str = ""
    assessor_allowlist_hash: str = ""
    combined_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.planner_profile != PLANNER_FEATURE_PROFILE_V7:
            raise ValueError("planner feature profile name drifted")
        if self.assessor_profile != ASSESSOR_FEATURE_PROFILE_V7:
            raise ValueError("assessor feature profile name drifted")
        shared = tuple(self.preaction_shared_fields)
        planner_extension = tuple(self.planner_extension_fields)
        assessor_extension = tuple(self.assessor_postaction_extension_fields)
        groups = (shared, planner_extension, assessor_extension)
        if any(any(not isinstance(row, FeatureFieldV7) for row in rows)
               for rows in groups):
            raise ValueError("feature allowlists require typed fields")
        if ({row.name for row in shared} != PREACTION_SHARED_FIELDS_V7
                or len(shared) != len(PREACTION_SHARED_FIELDS_V7)):
            raise ValueError("preaction shared allowlist is incomplete or drifted")
        if ({row.name for row in planner_extension}
                != PLANNER_EXTENSION_FIELDS_V7
                or len(planner_extension) != len(PLANNER_EXTENSION_FIELDS_V7)):
            raise ValueError("planner extension must remain the frozen empty set")
        if ({row.name for row in assessor_extension}
                != ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7
                or len(assessor_extension)
                != len(ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7)):
            raise ValueError("assessor post-action extension is incomplete or drifted")
        names = [set(row.name for row in rows) for rows in groups]
        if (names[0] & names[1] or names[0] & names[2]
                or names[1] & names[2]):
            raise ValueError("shared and extension feature rows must not be duplicated")
        ids = [set(row.canonical_field_id for row in rows) for rows in groups]
        if any(len(group_ids) != len(rows)
               for group_ids, rows in zip(ids, groups)):
            raise ValueError("canonical feature ids must be unique within each set")
        if (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]):
            raise ValueError("canonical feature ids cannot alias across stored sets")
        if any(row.stage is not FeatureStageV7.PREACTION_SHARED
               for row in shared):
            raise ValueError("shared field has incorrect stage metadata")
        if any(row.stage is not FeatureStageV7.PLANNER_EXTENSION
               for row in planner_extension):
            raise ValueError("planner extension field has incorrect stage metadata")
        if any(row.stage is not FeatureStageV7.ASSESSOR_POSTACTION_EXTENSION
               for row in assessor_extension):
            raise ValueError("assessor extension field has incorrect stage metadata")
        if any(row.dependency_kind in {
            FeatureDependencyKindV7.ARM_REALIZATION,
            FeatureDependencyKindV7.POST_ACTION_OBSERVATION,
        } for row in (*shared, *planner_extension)):
            raise ValueError("post-action dependency entered preaction features")
        if any(row.dependency_kind not in {
            FeatureDependencyKindV7.ARM_REALIZATION,
            FeatureDependencyKindV7.POST_ACTION_OBSERVATION,
        } for row in assessor_extension):
            raise ValueError("assessor extension needs a post-action dependency")

        def rows_payload(rows: Sequence[FeatureFieldV7]) -> list[dict[str, str]]:
            return [
                {
                    "canonical_field_id": row.canonical_field_id,
                    "name": row.name,
                    "stage": row.stage.value,
                    "available_at_stage": row.available_at_stage.value,
                    "source_hash": row.source_hash,
                    "dependency_kind": row.dependency_kind.value,
                    "dependency_hash": row.dependency_hash,
                }
                for row in sorted(rows, key=lambda item: item.name)
            ]

        shared_payload = rows_payload(shared)
        planner_payload = rows_payload(planner_extension)
        assessor_extension_payload = rows_payload(assessor_extension)
        planner_hash = _sealed_hash(
            self.planner_allowlist_hash,
            {
                "profile": self.planner_profile,
                "shared_fields": shared_payload,
                "extension_fields": planner_payload,
            },
            "planner allowlist hash",
        )
        assessor_hash = _sealed_hash(
            self.assessor_allowlist_hash,
            {
                "profile": self.assessor_profile,
                "shared_fields": shared_payload,
                "extension_fields": assessor_extension_payload,
            },
            "assessor allowlist hash",
        )
        fingerprint = _sealed_hash(
            self.combined_fingerprint,
            {
                "planner_allowlist_hash": planner_hash,
                "assessor_allowlist_hash": assessor_hash,
            },
            "combined feature allowlist fingerprint",
        )
        object.__setattr__(self, "planner_allowlist_hash", planner_hash)
        object.__setattr__(self, "assessor_allowlist_hash", assessor_hash)
        object.__setattr__(self, "combined_fingerprint", fingerprint)

    @property
    def assessor_model_fields(self) -> frozenset[str]:
        return ASSESSOR_POST_ACTION_FIELDS_V7

    @property
    def planner_fields(self) -> tuple[FeatureFieldV7, ...]:
        return self.preaction_shared_fields + self.planner_extension_fields

    @property
    def assessor_fields(self) -> tuple[FeatureFieldV7, ...]:
        return (
            self.preaction_shared_fields
            + self.assessor_postaction_extension_fields
        )


@dataclass(frozen=True)
class ArmRealizationReceiptV7:
    """Append-only SR2 realization bound to an immutable planned arm id."""

    manifest_hash: str
    planned_arm_id: str
    source_hash: str
    status: ArmRealizationStatusV7
    status_reason: str
    realized_input_hash: str | None
    realized_output_hash: str | None
    realized_support_hash: str | None
    realized_operator_hash: str | None
    tensor_shape: tuple[int, ...]
    tensor_dtype: str | None
    realized_bytes: int | None
    offline_bank_acquisition_cost: float | None
    offline_bank_acquisition_cost_counters: CostCountersV7 | None
    cost_account: CostAccountV7
    provenance_hash: str
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("realization manifest hash", self.manifest_hash),
            ("realization planned arm id", self.planned_arm_id),
            ("realization source hash", self.source_hash),
            ("realization status reason", self.status_reason),
            ("realization provenance hash", self.provenance_hash),
        ):
            _require_text(value, name)
        _enum(self.status, ArmRealizationStatusV7, "arm realization status")
        realized = (
            self.realized_input_hash, self.realized_output_hash,
            self.realized_support_hash, self.realized_operator_hash,
            self.tensor_dtype, self.realized_bytes,
            self.offline_bank_acquisition_cost,
            self.offline_bank_acquisition_cost_counters,
        )
        _enum(self.cost_account, CostAccountV7, "realization cost account")
        if self.cost_account is not CostAccountV7.OFFLINE_BANK_ACQUISITION:
            raise ValueError("realization costs must use offline-bank account")
        if self.status is ArmRealizationStatusV7.COMPLETE:
            for name, value in (
                ("realized input hash", self.realized_input_hash),
                ("realized output hash", self.realized_output_hash),
                ("realized support hash", self.realized_support_hash),
                ("realized operator hash", self.realized_operator_hash),
                ("realized tensor dtype", self.tensor_dtype),
            ):
                _require_text(value, name)
            if (not self.tensor_shape
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           or value <= 0 for value in self.tensor_shape)):
                raise ValueError("complete realization needs a positive tensor shape")
            if (isinstance(self.realized_bytes, bool)
                    or not isinstance(self.realized_bytes, int)
                    or self.realized_bytes <= 0):
                raise ValueError("complete realization needs positive realized bytes")
            _nonnegative(
                self.offline_bank_acquisition_cost,
                "offline bank acquisition cost",
            )
            if not isinstance(
                self.offline_bank_acquisition_cost_counters, CostCountersV7,
            ):
                raise ValueError("complete realization needs offline cost counters")
        else:
            if any(value is not None for value in realized):
                raise ValueError("incomplete realization fields must be typed missing")
            if self.tensor_shape:
                raise ValueError("incomplete realization cannot claim a tensor shape")
        payload = {
            "manifest_hash": self.manifest_hash,
            "planned_arm_id": self.planned_arm_id,
            "source_hash": self.source_hash,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "realized_input_hash": self.realized_input_hash,
            "realized_output_hash": self.realized_output_hash,
            "realized_support_hash": self.realized_support_hash,
            "realized_operator_hash": self.realized_operator_hash,
            "tensor_shape": list(self.tensor_shape),
            "tensor_dtype": self.tensor_dtype,
            "realized_bytes": self.realized_bytes,
            "offline_bank_acquisition_cost": self.offline_bank_acquisition_cost,
            "offline_bank_acquisition_cost_hash": (
                self.offline_bank_acquisition_cost_counters.content_hash
                if self.offline_bank_acquisition_cost_counters is not None else None
            ),
            "cost_account": self.cost_account.value,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "receipt_hash", _sealed_hash(
            self.receipt_hash, payload, "arm realization receipt hash",
        ))

    @property
    def executable_bytes_hash(self) -> str | None:
        if self.status is not ArmRealizationStatusV7.COMPLETE:
            return None
        return _sha256({
            "realized_input_hash": self.realized_input_hash,
            "realized_output_hash": self.realized_output_hash,
            "realized_support_hash": self.realized_support_hash,
            "realized_operator_hash": self.realized_operator_hash,
            "tensor_shape": list(self.tensor_shape),
            "tensor_dtype": self.tensor_dtype,
            "realized_bytes": self.realized_bytes,
        })


@dataclass(frozen=True)
class ArmAliasReceiptV7:
    """Append-only byte-alias record; aliases retain their planned lineage."""

    manifest_hash: str
    representative_planned_arm_id: str
    alias_planned_arm_id: str
    representative_realization_receipt_hash: str
    alias_realization_receipt_hash: str
    executable_bytes_hash: str
    realized_operator_hash: str
    prospective_runtime_verification_cost: float
    prospective_runtime_verification_cost_counters: CostCountersV7
    cost_account: CostAccountV7
    provenance_hash: str
    reason_code: str = "byte_identical_executable"
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("alias manifest hash", self.manifest_hash),
            ("representative planned arm id", self.representative_planned_arm_id),
            ("alias planned arm id", self.alias_planned_arm_id),
            ("representative realization receipt hash",
             self.representative_realization_receipt_hash),
            ("alias realization receipt hash", self.alias_realization_receipt_hash),
            ("alias executable bytes hash", self.executable_bytes_hash),
            ("alias realized operator hash", self.realized_operator_hash),
            ("alias provenance hash", self.provenance_hash),
            ("alias reason code", self.reason_code),
        ):
            _require_text(value, name)
        if self.representative_planned_arm_id >= self.alias_planned_arm_id:
            raise ValueError("alias representative must be lexical planned-id minimum")
        if self.reason_code != "byte_identical_executable":
            raise ValueError("unknown arm alias reason")
        _nonnegative(
            self.prospective_runtime_verification_cost,
            "prospective alias verification cost",
        )
        if not isinstance(
            self.prospective_runtime_verification_cost_counters,
            CostCountersV7,
        ):
            raise ValueError("alias verification needs typed cost counters")
        _enum(self.cost_account, CostAccountV7, "alias cost account")
        if self.cost_account is not CostAccountV7.PROSPECTIVE_RUNTIME:
            raise ValueError("alias verification must use prospective-runtime account")
        forbidden_execution_counters = (
            "natural_forwards", "reused_forwards", "candidate_forwards",
            "reverse_forwards", "gpu_seconds", "matcher_trajectories",
        )
        if any(
            getattr(self.prospective_runtime_verification_cost_counters, name)
            != 0
            for name in forbidden_execution_counters
        ):
            raise ValueError("alias verification cost cannot hide model execution")
        payload = {
            name: getattr(self, name)
            for name in (
                "manifest_hash", "representative_planned_arm_id",
                "alias_planned_arm_id",
                "representative_realization_receipt_hash",
                "alias_realization_receipt_hash", "executable_bytes_hash",
                "realized_operator_hash", "prospective_runtime_verification_cost",
                "cost_account", "provenance_hash", "reason_code",
            )
        }
        payload["prospective_runtime_verification_cost_counters_hash"] = (
            self.prospective_runtime_verification_cost_counters.content_hash
        )
        payload["cost_account"] = self.cost_account.value
        object.__setattr__(self, "receipt_hash", _sealed_hash(
            self.receipt_hash, payload, "arm alias receipt hash",
        ))


@dataclass(frozen=True)
class InteractionReceiptV7:
    """One directed side of a symmetric local-interaction certificate."""

    source_unit_id: str
    target_unit_id: str
    kind: InteractionKindV7
    interaction_upper: float
    receipt_hash: str

    def __post_init__(self) -> None:
        _require_text(self.source_unit_id, "interaction source unit")
        _require_text(self.target_unit_id, "interaction target unit")
        if self.source_unit_id == self.target_unit_id:
            raise ValueError("an interaction receipt must join distinct units")
        _enum(self.kind, InteractionKindV7, "interaction kind")
        bound = _nonnegative(self.interaction_upper, "interaction upper")
        if self.kind is InteractionKindV7.DISJOINT and bound != 0.0:
            raise ValueError("a disjoint interaction must have zero upper bound")
        _require_text(self.receipt_hash, "interaction receipt hash")


@dataclass(frozen=True)
class CandidateArmV7:
    """One exact action/control/endpoint arm in a frozen manifest."""

    candidate_id: str
    action_index: int
    action_identity: str
    mechanism_id: str
    operator_version: str
    exact_control_id: str
    endpoint_id: str
    input_strength: float
    output_beta: float
    action_hash: str
    control_hash: str
    endpoint_hash: str
    support_hash: str
    support_policy_hash: str
    rollback_hash: str
    source_input_hashes: tuple[str, ...]
    materialization_recipe_id: str
    materialization_recipe_version: str
    materialization_recipe_hash: str
    provenance_hash: str
    scope: CandidateScopeV7
    unit_ids: tuple[str, ...]
    interaction_receipts: tuple[InteractionReceiptV7, ...]
    offline_bank_acquisition_cost_ceiling: float
    offline_bank_acquisition_cost_counter_ceiling: CostCountersV7
    prospective_runtime_cost_ceiling: float
    prospective_runtime_cost_counter_ceiling: CostCountersV7
    hard_legal: bool
    hard_reject_reasons: tuple[HardRejectReasonV7, ...] = ()
    interaction_hash: str = ""
    cost_hash: str = ""
    planned_arm_id: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate id", self.candidate_id),
            ("action identity", self.action_identity),
            ("mechanism id", self.mechanism_id),
            ("operator version", self.operator_version),
            ("exact control id", self.exact_control_id),
            ("endpoint id", self.endpoint_id),
            ("action hash", self.action_hash),
            ("control hash", self.control_hash),
            ("endpoint hash", self.endpoint_hash),
            ("support hash", self.support_hash),
            ("support policy hash", self.support_policy_hash),
            ("rollback hash", self.rollback_hash),
            ("materialization recipe id", self.materialization_recipe_id),
            ("materialization recipe version", self.materialization_recipe_version),
            ("materialization recipe hash", self.materialization_recipe_hash),
            ("provenance hash", self.provenance_hash),
        ):
            _require_text(value, name)
        if isinstance(self.action_index, bool) or not isinstance(self.action_index, int):
            raise ValueError("action index must be an integer")
        if self.action_index < 0:
            raise ValueError("action index must be nonnegative")
        _enum(self.scope, CandidateScopeV7, "candidate scope")
        units = _unique_nonempty(self.unit_ids, "candidate unit ids")
        sources = _unique_nonempty(self.source_input_hashes, "source input hashes")
        strength = _fraction(self.input_strength, "input strength")
        beta = _fraction(self.output_beta, "output beta")
        offline_cost = _nonnegative(
            self.offline_bank_acquisition_cost_ceiling,
            "offline bank acquisition cost ceiling",
        )
        runtime_cost = _nonnegative(
            self.prospective_runtime_cost_ceiling,
            "prospective runtime cost ceiling",
        )
        if not isinstance(
            self.offline_bank_acquisition_cost_counter_ceiling,
            CostCountersV7,
        ):
            raise ValueError("candidate needs offline cost-counter ceiling")
        if not isinstance(
            self.prospective_runtime_cost_counter_ceiling,
            CostCountersV7,
        ):
            raise ValueError("candidate needs prospective cost-counter ceiling")
        if not isinstance(self.hard_legal, bool):
            raise ValueError("hard legality must be boolean")
        if any(not isinstance(item, HardRejectReasonV7)
               for item in self.hard_reject_reasons):
            raise ValueError("hard rejection reasons must be typed")
        if len(set(self.hard_reject_reasons)) != len(self.hard_reject_reasons):
            raise ValueError("hard rejection reasons must be unique")
        if self.hard_legal and self.hard_reject_reasons:
            raise ValueError("a hard-legal arm cannot carry hard rejection reasons")
        if not self.hard_legal and not self.hard_reject_reasons:
            raise ValueError("a hard-rejected arm needs a typed reason")
        receipts = tuple(self.interaction_receipts)
        keys = tuple((row.source_unit_id, row.target_unit_id) for row in receipts)
        if len(keys) != len(set(keys)):
            raise ValueError("interaction directions must be unique")
        if any(row.source_unit_id not in units or row.target_unit_id not in units
               for row in receipts):
            raise ValueError("interaction receipt lies outside candidate units")
        if self.scope is not CandidateScopeV7.CERTIFIED_LOCAL and receipts:
            raise ValueError("only local candidates may carry interaction receipts")

        native = self.action_index == 0
        if native:
            if (self.scope is not CandidateScopeV7.NATIVE
                    or self.action_identity != "native"
                    or self.exact_control_id != "native"
                    or strength != 0.0 or beta != 0.0
                    or offline_cost != 0.0 or runtime_cost != 0.0
                    or not self.hard_legal):
                raise ValueError("action 0 must be feasible zero-cost native")
            zero_counter_names = (
                "natural_forwards", "reused_forwards", "candidate_forwards",
                "reverse_forwards", "cpu_seconds", "gpu_seconds",
                "wall_seconds", "bytes_moved", "peak_allocated_bytes",
                "peak_reserved_bytes", "matcher_trajectories",
            )
            if any(
                getattr(counters, name) != 0
                for counters in (
                    self.offline_bank_acquisition_cost_counter_ceiling,
                    self.prospective_runtime_cost_counter_ceiling,
                )
                for name in zero_counter_names
            ):
                raise ValueError("native must have zero cost counters")
        elif self.scope is CandidateScopeV7.NATIVE:
            raise ValueError("only action 0 may have native scope")
        elif strength <= 0.0 or beta <= 0.0:
            raise ValueError("nonnative input strength and output beta must be positive")
        if self.scope is CandidateScopeV7.CASE_ATOMIC and len(units) != 1:
            raise ValueError("CASE_ATOMIC must name exactly one case execution unit")
        if self.scope is CandidateScopeV7.CERTIFIED_LOCAL and len(units) != 1:
            if self.hard_legal:
                raise ValueError("v7.39 primary forbids multi-local commit")
            if not ({
                HardRejectReasonV7.UNKNOWN_INTERACTION,
                HardRejectReasonV7.PRIMARY_MULTI_LOCAL_FORBIDDEN,
            } & set(self.hard_reject_reasons)):
                raise ValueError("multi-local arm needs a permanent hard reason")

        interaction_payload = [
            {
                "source": row.source_unit_id,
                "target": row.target_unit_id,
                "kind": row.kind.value,
                "upper": float(row.interaction_upper),
                "receipt_hash": row.receipt_hash,
            }
            for row in sorted(
                receipts, key=lambda x: (x.source_unit_id, x.target_unit_id)
            )
        ]
        interaction_hash = _sealed_hash(
            self.interaction_hash, interaction_payload, "interaction hash",
        )
        cost_hash = _sealed_hash(
            self.cost_hash,
            {
                "offline_bank_acquisition_cost_ceiling": offline_cost,
                "offline_bank_acquisition_cost_counter_ceiling_hash": (
                    self.offline_bank_acquisition_cost_counter_ceiling.content_hash
                ),
                "prospective_runtime_cost_ceiling": runtime_cost,
                "prospective_runtime_cost_counter_ceiling_hash": (
                    self.prospective_runtime_cost_counter_ceiling.content_hash
                ),
            },
            "cost hash",
        )
        planned_payload = {
            "mechanism_id": self.mechanism_id,
            "operator_version": self.operator_version,
            "endpoint_id": self.endpoint_id,
            "input_strength": strength,
            "output_beta": beta,
            "support_policy_hash": self.support_policy_hash,
            "source_input_hashes": list(sources),
            "materialization_recipe_id": self.materialization_recipe_id,
            "materialization_recipe_version": self.materialization_recipe_version,
            "materialization_recipe_hash": self.materialization_recipe_hash,
            "execution_unit_ids": list(units),
        }
        planned_arm_id = _sealed_hash(
            self.planned_arm_id,
            planned_payload,
            "planned arm id",
        )
        object.__setattr__(self, "interaction_hash", interaction_hash)
        object.__setattr__(self, "cost_hash", cost_hash)
        object.__setattr__(self, "planned_arm_id", planned_arm_id)

    @property
    def manifest_binding_hash(self) -> str:
        return _sha256({
            "candidate_id": self.candidate_id,
            "action_index": self.action_index,
            "planned_arm_id": self.planned_arm_id,
            "action_identity": self.action_identity,
            "exact_control_id": self.exact_control_id,
            "action_hash": self.action_hash,
            "control_hash": self.control_hash,
            "endpoint_hash": self.endpoint_hash,
            "support_hash": self.support_hash,
            "interaction_hash": self.interaction_hash,
            "cost_hash": self.cost_hash,
            "materialization_recipe_hash": self.materialization_recipe_hash,
            "provenance_hash": self.provenance_hash,
            "scope": self.scope.value,
            "unit_ids": list(self.unit_ids),
            "hard_legal": self.hard_legal,
            "hard_reject_reasons": [row.value for row in self.hard_reject_reasons],
        })

    def composition_reason(self) -> DecisionReasonV7 | None:
        if (self.scope is not CandidateScopeV7.CERTIFIED_LOCAL
                or len(self.unit_ids) == 1):
            return None
        by_direction = {
            (row.source_unit_id, row.target_unit_id): row
            for row in self.interaction_receipts
        }
        for left, right in permutations(self.unit_ids, 2):
            row = by_direction.get((left, right))
            reverse = by_direction.get((right, left))
            if row is None or reverse is None:
                return DecisionReasonV7.UNKNOWN_INTERACTION
            if row.kind is not reverse.kind:
                return DecisionReasonV7.UNKNOWN_INTERACTION
            if row.receipt_hash != reverse.receipt_hash:
                return DecisionReasonV7.UNKNOWN_INTERACTION
            if not math.isclose(
                row.interaction_upper, reverse.interaction_upper,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                return DecisionReasonV7.UNKNOWN_INTERACTION
        return None


@dataclass(frozen=True)
class CandidateManifestV7:
    """Outcome-blind, immutable finite candidate bank for one decision unit."""

    case_id: str
    execution_unit_id: str
    fold_role: FoldRoleV7
    source_hash: str
    support_policy_hash: str
    planner_allowlist_hash: str
    assessor_allowlist_hash: str
    schema_version: str
    provenance_hash: str
    frozen_before_outcome: bool
    candidates: tuple[CandidateArmV7, ...]
    candidate_family_hash: str = ""
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("case id", self.case_id),
            ("execution unit id", self.execution_unit_id),
            ("manifest source hash", self.source_hash),
            ("support policy hash", self.support_policy_hash),
            ("planner allowlist hash", self.planner_allowlist_hash),
            ("assessor allowlist hash", self.assessor_allowlist_hash),
            ("manifest provenance hash", self.provenance_hash),
        ):
            _require_text(value, name)
        _enum(self.fold_role, FoldRoleV7, "manifest fold role")
        if self.schema_version != SELECTOR_V7_SCHEMA_VERSION:
            raise ValueError("manifest SelectorV7 schema version drifted")
        if self.planner_allowlist_hash == self.assessor_allowlist_hash:
            raise ValueError("planner and assessor allowlists cannot alias")
        if not isinstance(self.frozen_before_outcome, bool):
            raise ValueError("manifest frozen flag must be boolean")
        rows = tuple(self.candidates)
        if not rows or any(not isinstance(row, CandidateArmV7) for row in rows):
            raise ValueError("manifest needs typed candidate arms")
        if len({row.candidate_id for row in rows}) != len(rows):
            raise ValueError("candidate ids must be unique")
        if len({row.action_index for row in rows}) != len(rows):
            raise ValueError("candidate action indices must be unique")
        if len({row.planned_arm_id for row in rows}) != len(rows):
            raise ValueError("planned arm ids must be unique")
        native = [row for row in rows if row.action_index == 0]
        if len(native) != 1:
            raise ValueError("manifest must contain exactly one action-0 native")
        for row in rows:
            if row.support_policy_hash != self.support_policy_hash:
                raise ValueError("candidate and manifest support policies differ")
            if self.source_hash not in row.source_input_hashes:
                raise ValueError("candidate does not bind the manifest source")
            reason = row.composition_reason()
            if reason is not None and row.hard_legal:
                raise ValueError("unknown local interaction cannot be hard legal")
            if (reason is not None
                    and HardRejectReasonV7.UNKNOWN_INTERACTION
                    not in row.hard_reject_reasons):
                raise ValueError("unknown local interaction needs a hard reason")
        identities = [
            row.planned_arm_id
            for row in sorted(rows, key=lambda item: item.action_index)
        ]
        family_hash = _sealed_hash(
            self.candidate_family_hash, identities, "candidate family hash",
        )
        payload = {
            "case_id": self.case_id,
            "execution_unit_id": self.execution_unit_id,
            "fold_role": self.fold_role.value,
            "source_hash": self.source_hash,
            "support_policy_hash": self.support_policy_hash,
            "planner_allowlist_hash": self.planner_allowlist_hash,
            "assessor_allowlist_hash": self.assessor_allowlist_hash,
            "schema_version": self.schema_version,
            "provenance_hash": self.provenance_hash,
            "frozen_before_outcome": self.frozen_before_outcome,
            "candidate_family_hash": family_hash,
            "planned_arm_ids": identities,
            "candidate_manifest_binding_hashes": [
                row.manifest_binding_hash
                for row in sorted(rows, key=lambda item: item.action_index)
            ],
        }
        manifest_hash = _sealed_hash(
            self.manifest_hash, payload, "manifest hash",
        )
        object.__setattr__(self, "candidate_family_hash", family_hash)
        object.__setattr__(self, "manifest_hash", manifest_hash)

    @property
    def native(self) -> CandidateArmV7:
        return next(row for row in self.candidates if row.action_index == 0)

    def candidate(self, candidate_id: str) -> CandidateArmV7 | None:
        return next(
            (row for row in self.candidates if row.candidate_id == candidate_id),
            None,
        )

    def exclusion_state(self, candidate_id: str,
                        proposed_ids: Iterable[str] = ()) -> CandidateExclusionStateV7 | None:
        row = self.candidate(candidate_id)
        if row is None:
            raise ValueError("candidate is outside the frozen manifest")
        if not row.hard_legal:
            return CandidateExclusionStateV7.HARD_ILLEGAL_PREEXECUTION
        if candidate_id not in set(proposed_ids) and row.action_index != 0:
            return CandidateExclusionStateV7.SOFT_NOT_PROPOSED
        return None


@dataclass(frozen=True)
class PlannerCandidateV7:
    candidate_id: str
    planned_arm_id: str
    cost_hash: str
    planner_allowlist_hash: str
    before_feature_hash: str
    prediction_kind: PredictionKindV7
    planner_predicted_gain_raw_px: float
    before_features_complete: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("planner candidate id", self.candidate_id),
            ("planner planned arm id", self.planned_arm_id),
            ("planner cost hash", self.cost_hash),
            ("planner allowlist hash", self.planner_allowlist_hash),
            ("planner before-feature hash", self.before_feature_hash),
        ):
            _require_text(value, name)
        _finite(self.planner_predicted_gain_raw_px, "planner predicted G")
        if (self.prediction_kind
                is not PredictionKindV7.PLANNER_MODEL_PREDICTION_PREDECISION):
            raise ValueError("planner score is not an allowed predecision prediction")
        if not isinstance(self.before_features_complete, bool):
            raise ValueError("before-feature completeness must be boolean")


@dataclass(frozen=True)
class PlannerProposalV7:
    manifest_hash: str
    candidate_family_hash: str
    source_hash: str
    fold_role: FoldRoleV7
    planner_allowlist_hash: str
    planner_policy_hash: str
    planner_provenance_hash: str
    profile: PlannerProfileV7
    top_k: int
    maximum_prospective_runtime_cost: float
    reserved_prospective_runtime_cost: float
    maximum_prospective_runtime_cost_counters: CostCountersV7
    reserved_prospective_runtime_cost_counters: CostCountersV7
    frozen_before_observation: bool
    candidates: tuple[PlannerCandidateV7, ...]
    proposal_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("proposal manifest hash", self.manifest_hash),
            ("proposal candidate-family hash", self.candidate_family_hash),
            ("proposal source hash", self.source_hash),
            ("proposal planner allowlist hash", self.planner_allowlist_hash),
            ("planner policy hash", self.planner_policy_hash),
            ("planner provenance hash", self.planner_provenance_hash),
        ):
            _require_text(value, name)
        _enum(self.fold_role, FoldRoleV7, "proposal fold role")
        _enum(self.profile, PlannerProfileV7, "planner profile")
        if self.top_k not in {1, 2}:
            raise ValueError("Selector-v7 top K must be exactly 1 or 2")
        expected_k = (
            1 if self.profile is PlannerProfileV7.K1_ABLATION else 2
        )
        if self.top_k != expected_k:
            raise ValueError("planner profile and K disagree")
        ceiling = _nonnegative(
            self.maximum_prospective_runtime_cost,
            "proposal prospective-runtime cost ceiling",
        )
        reserved = _nonnegative(
            self.reserved_prospective_runtime_cost,
            "proposal reserved prospective-runtime cost",
        )
        if reserved > ceiling:
            raise ValueError("proposal reservation exceeds its frozen cost ceiling")
        if not isinstance(
            self.maximum_prospective_runtime_cost_counters, CostCountersV7,
        ) or not isinstance(
            self.reserved_prospective_runtime_cost_counters, CostCountersV7,
        ):
            raise ValueError("proposal needs typed prospective-runtime counters")
        if not self.reserved_prospective_runtime_cost_counters.within(
            self.maximum_prospective_runtime_cost_counters,
        ):
            raise ValueError("proposal counter reservation exceeds its ceiling")
        if not isinstance(self.frozen_before_observation, bool):
            raise ValueError("proposal frozen flag must be boolean")
        rows = tuple(self.candidates)
        if any(not isinstance(row, PlannerCandidateV7) for row in rows):
            raise ValueError("proposal needs typed planner candidates")
        if len({row.candidate_id for row in rows}) != len(rows):
            raise ValueError("proposal candidate ids must be unique")
        payload = {
            "manifest_hash": self.manifest_hash,
            "candidate_family_hash": self.candidate_family_hash,
            "source_hash": self.source_hash,
            "planner_allowlist_hash": self.planner_allowlist_hash,
            "fold_role": self.fold_role.value,
            "planner_policy_hash": self.planner_policy_hash,
            "planner_provenance_hash": self.planner_provenance_hash,
            "profile": self.profile.value,
            "top_k": self.top_k,
            "maximum_prospective_runtime_cost": ceiling,
            "reserved_prospective_runtime_cost": reserved,
            "maximum_prospective_runtime_cost_counters_hash": (
                self.maximum_prospective_runtime_cost_counters.content_hash
            ),
            "reserved_prospective_runtime_cost_counters_hash": (
                self.reserved_prospective_runtime_cost_counters.content_hash
            ),
            "frozen_before_observation": self.frozen_before_observation,
            "candidate_rows": [
                {
                    "candidate_id": row.candidate_id,
                    "planned_arm_id": row.planned_arm_id,
                    "cost_hash": row.cost_hash,
                    "planner_allowlist_hash": row.planner_allowlist_hash,
                    "before_feature_hash": row.before_feature_hash,
                    "prediction_kind": row.prediction_kind.value,
                    "planner_predicted_gain_raw_px": (
                        row.planner_predicted_gain_raw_px
                    ),
                    "before_features_complete": row.before_features_complete,
                }
                for row in rows
            ],
        }
        object.__setattr__(self, "proposal_hash", _sealed_hash(
            self.proposal_hash, payload, "proposal hash",
        ))


@dataclass(frozen=True)
class RuntimeFeatureV7:
    name: str
    value: float

    def __post_init__(self) -> None:
        _require_text(self.name, "runtime feature name")
        if _forbidden_runtime_name(self.name):
            raise ValueError("forbidden dataset/scene/path/GT/outcome runtime field")
        _finite(self.value, f"runtime feature {self.name}")


@dataclass(frozen=True)
class CandidateObservationV7:
    manifest_hash: str
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
    planner_feature_profile: str
    assessor_feature_profile: str
    planner_allowlist_hash: str
    assessor_allowlist_hash: str
    preaction_shared_feature_hash: str
    feature_schema_hash: str
    provenance_hash: str
    availability: AvailabilityV7
    availability_reason: str
    runtime_features: tuple[RuntimeFeatureV7, ...]
    evidence_blocks: tuple[EvidenceBlockReceiptV7, ...]
    evidence_source_hashes: tuple[str, ...]
    observed_unit_ids: tuple[str, ...]
    atomically_observed: bool
    observed_prospective_runtime_cost: float
    observed_prospective_runtime_cost_counters: CostCountersV7
    cost_account: CostAccountV7
    cost_receipt_hash: str
    observation_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("observation manifest hash", self.manifest_hash),
            ("observation proposal hash", self.proposal_hash),
            ("observation candidate id", self.candidate_id),
            ("observation planned arm id", self.planned_arm_id),
            ("observation realization receipt hash",
             self.realization_receipt_hash),
            ("observation action hash", self.action_hash),
            ("observation control hash", self.control_hash),
            ("observation endpoint hash", self.endpoint_hash),
            ("observation support hash", self.support_hash),
            ("observation realized support hash", self.realized_support_hash),
            ("observation interaction hash", self.interaction_hash),
            ("observation cost hash", self.cost_hash),
            ("observation source hash", self.source_hash),
            ("planner feature profile", self.planner_feature_profile),
            ("assessor feature profile", self.assessor_feature_profile),
            ("observation planner allowlist hash", self.planner_allowlist_hash),
            ("observation assessor allowlist hash", self.assessor_allowlist_hash),
            ("preaction shared feature hash", self.preaction_shared_feature_hash),
            ("feature schema hash", self.feature_schema_hash),
            ("observation provenance hash", self.provenance_hash),
            ("availability reason", self.availability_reason),
            ("cost receipt hash", self.cost_receipt_hash),
        ):
            _require_text(value, name)
        _enum(self.availability, AvailabilityV7, "observation availability")
        _enum(self.cost_account, CostAccountV7, "observation cost account")
        if self.cost_account is not CostAccountV7.PROSPECTIVE_RUNTIME:
            raise ValueError("observation cost must use prospective-runtime account")
        if self.planner_feature_profile != PLANNER_FEATURE_PROFILE_V7:
            raise ValueError("observation planner stage profile drifted")
        if self.assessor_feature_profile != ASSESSOR_FEATURE_PROFILE_V7:
            raise ValueError("observation assessor stage profile drifted")
        features = tuple(self.runtime_features)
        if any(not isinstance(row, RuntimeFeatureV7) for row in features):
            raise ValueError("runtime features must be typed")
        if len({row.name for row in features}) != len(features):
            raise ValueError("runtime feature names must be unique")
        if ({row.name for row in features}
                != ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7):
            raise ValueError("runtime features must be exact post-action extension")
        blocks = tuple(self.evidence_blocks)
        if any(not isinstance(row, EvidenceBlockReceiptV7) for row in blocks):
            raise ValueError("evidence blocks must be typed")
        if {row.block for row in blocks} != PRIMARY_REQUIRED_EVIDENCE_BLOCKS_V7:
            raise ValueError("primary observation needs its exact required blocks")
        if len(blocks) != len(PRIMARY_REQUIRED_EVIDENCE_BLOCKS_V7):
            raise ValueError("primary evidence blocks must be unique")
        sources = _unique_nonempty(
            self.evidence_source_hashes, "evidence source hashes",
        )
        units = _unique_nonempty(self.observed_unit_ids, "observed unit ids")
        if not isinstance(self.atomically_observed, bool):
            raise ValueError("atomic observation flag must be boolean")
        cost = _nonnegative(
            self.observed_prospective_runtime_cost,
            "observed prospective-runtime cost",
        )
        if not isinstance(
            self.observed_prospective_runtime_cost_counters, CostCountersV7,
        ):
            raise ValueError("observation needs typed cost counters")
        if self.availability is AvailabilityV7.AVAILABLE and not features:
            raise ValueError("available observation needs runtime features")
        payload = {
            "manifest_hash": self.manifest_hash,
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
            "planner_feature_profile": self.planner_feature_profile,
            "assessor_feature_profile": self.assessor_feature_profile,
            "planner_allowlist_hash": self.planner_allowlist_hash,
            "assessor_allowlist_hash": self.assessor_allowlist_hash,
            "preaction_shared_feature_hash": self.preaction_shared_feature_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "provenance_hash": self.provenance_hash,
            "availability": self.availability.value,
            "availability_reason": self.availability_reason,
            "runtime_features": [
                [row.name, row.value] for row in features
            ],
            "evidence_blocks": [
                [row.block.value, row.availability.value, row.source_hash,
                 row.receipt_hash]
                for row in sorted(blocks, key=lambda item: item.block.value)
            ],
            "evidence_source_hashes": list(sources),
            "observed_unit_ids": list(units),
            "atomically_observed": self.atomically_observed,
            "observed_prospective_runtime_cost": cost,
            "observed_prospective_runtime_cost_counters_hash": (
                self.observed_prospective_runtime_cost_counters.content_hash
            ),
            "cost_account": self.cost_account.value,
            "cost_receipt_hash": self.cost_receipt_hash,
        }
        object.__setattr__(self, "observation_hash", _sealed_hash(
            self.observation_hash, payload, "observation hash",
        ))


@dataclass(frozen=True)
class ActionRiskVectorV7:
    """Exactly five v4.39 assessor heads and their simultaneous bounds."""

    manifest_hash: str
    proposal_hash: str
    observation_hash: str
    candidate_id: str
    planned_arm_id: str
    action_hash: str
    control_hash: str
    endpoint_hash: str
    support_hash: str
    interaction_hash: str
    cost_hash: str
    assessor_hash: str
    assessor_provenance_hash: str
    assessor_feature_profile: str
    assessor_allowlist_hash: str
    prediction_kind: PredictionKindV7
    calibration_receipt_hash: str
    availability: AvailabilityV7
    availability_reason: str
    feature_eligible: bool
    calibration_valid: bool
    bound_bookkeeping_valid: bool
    bookkeeping_receipt_hash: str
    predicted_benefit: float | None
    benefit_lower: float | None
    predicted_harm: float | None
    harm_upper: float | None
    predicted_any_row_severe_probability: float | None
    any_row_severe_probability_upper: float | None
    predicted_harmed_pixel_fraction: float | None
    harmed_pixel_fraction_upper: float | None
    predicted_pixel_harm_cvar95: float | None
    pixel_harm_cvar95_upper: float | None
    risk_vector_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("risk manifest hash", self.manifest_hash),
            ("risk proposal hash", self.proposal_hash),
            ("risk observation hash", self.observation_hash),
            ("risk candidate id", self.candidate_id),
            ("risk planned arm id", self.planned_arm_id),
            ("risk action hash", self.action_hash),
            ("risk control hash", self.control_hash),
            ("risk endpoint hash", self.endpoint_hash),
            ("risk support hash", self.support_hash),
            ("risk interaction hash", self.interaction_hash),
            ("risk cost hash", self.cost_hash),
            ("assessor hash", self.assessor_hash),
            ("assessor provenance hash", self.assessor_provenance_hash),
            ("risk assessor feature profile", self.assessor_feature_profile),
            ("risk assessor allowlist hash", self.assessor_allowlist_hash),
            ("risk calibration receipt hash", self.calibration_receipt_hash),
            ("risk availability reason", self.availability_reason),
            ("bookkeeping receipt hash", self.bookkeeping_receipt_hash),
        ):
            _require_text(value, name)
        _enum(self.availability, AvailabilityV7, "risk availability")
        if self.assessor_feature_profile != ASSESSOR_FEATURE_PROFILE_V7:
            raise ValueError("risk vector assessor feature profile drifted")
        if (self.prediction_kind
                is not PredictionKindV7.ASSESSOR_MODEL_PREDICTION_PREDECISION):
            raise ValueError("assessor outputs must be predecision model predictions")
        if not isinstance(self.feature_eligible, bool):
            raise ValueError("feature eligibility must be boolean")
        if not isinstance(self.calibration_valid, bool):
            raise ValueError("calibration validity must be boolean")
        if not isinstance(self.bound_bookkeeping_valid, bool):
            raise ValueError("bound bookkeeping validity must be boolean")
        numeric = (
            self.predicted_benefit, self.benefit_lower,
            self.predicted_harm, self.harm_upper,
            self.predicted_any_row_severe_probability,
            self.any_row_severe_probability_upper,
            self.predicted_harmed_pixel_fraction,
            self.harmed_pixel_fraction_upper,
            self.predicted_pixel_harm_cvar95,
            self.pixel_harm_cvar95_upper,
        )
        if self.availability is AvailabilityV7.AVAILABLE:
            if any(item is None for item in numeric):
                raise ValueError("available risk vector has typed-missing values")
            _nonnegative(self.benefit_lower, "benefit lower")
            _nonnegative(self.harm_upper, "harm upper")
            _nonnegative(self.predicted_benefit, "predicted benefit")
            _nonnegative(self.predicted_harm, "predicted harm")
            _fraction(
                self.predicted_any_row_severe_probability,
                "predicted any-row-severe probability",
            )
            _fraction(
                self.any_row_severe_probability_upper,
                "any-row-severe probability upper",
            )
            _fraction(
                self.predicted_harmed_pixel_fraction,
                "predicted harmed-pixel fraction",
            )
            _fraction(
                self.harmed_pixel_fraction_upper,
                "harmed-pixel fraction upper",
            )
            _nonnegative(
                self.predicted_pixel_harm_cvar95,
                "predicted pixel-harm CVaR95",
            )
            _nonnegative(
                self.pixel_harm_cvar95_upper,
                "pixel-harm CVaR95 upper",
            )
            ordered_pairs = (
                (self.benefit_lower, self.predicted_benefit,
                 "benefit lower exceeds point prediction"),
                (self.predicted_harm, self.harm_upper,
                 "harm upper lies below point prediction"),
                (self.predicted_any_row_severe_probability,
                 self.any_row_severe_probability_upper,
                 "any-row-severe upper lies below point prediction"),
                (self.predicted_harmed_pixel_fraction,
                 self.harmed_pixel_fraction_upper,
                 "harmed-fraction upper lies below point prediction"),
                (self.predicted_pixel_harm_cvar95,
                 self.pixel_harm_cvar95_upper,
                 "CVaR upper lies below point prediction"),
            )
            for lower, upper, message in ordered_pairs:
                if float(lower) > float(upper) + 1e-12:
                    raise ValueError(message)
        else:
            if (self.feature_eligible or self.calibration_valid
                    or self.bound_bookkeeping_valid):
                raise ValueError("unavailable risk vector cannot be eligible")
            if any(item is not None for item in numeric):
                raise ValueError("unavailable risk values must be typed missing")
        payload = {
            "manifest_hash": self.manifest_hash,
            "proposal_hash": self.proposal_hash,
            "observation_hash": self.observation_hash,
            "candidate_id": self.candidate_id,
            "planned_arm_id": self.planned_arm_id,
            "bindings": [
                self.action_hash, self.control_hash, self.endpoint_hash,
                self.support_hash, self.interaction_hash, self.cost_hash,
            ],
            "assessor_hash": self.assessor_hash,
            "assessor_provenance_hash": self.assessor_provenance_hash,
            "assessor_feature_profile": self.assessor_feature_profile,
            "assessor_allowlist_hash": self.assessor_allowlist_hash,
            "prediction_kind": self.prediction_kind.value,
            "calibration_receipt_hash": self.calibration_receipt_hash,
            "availability": self.availability.value,
            "availability_reason": self.availability_reason,
            "feature_eligible": self.feature_eligible,
            "calibration_valid": self.calibration_valid,
            "bound_bookkeeping_valid": self.bound_bookkeeping_valid,
            "bookkeeping_receipt_hash": self.bookkeeping_receipt_hash,
            "five_heads": list(numeric),
        }
        object.__setattr__(self, "risk_vector_hash", _sealed_hash(
            self.risk_vector_hash, payload, "risk vector hash",
        ))

    @property
    def L_B(self) -> float | None:
        return self.benefit_lower

    @property
    def U_H(self) -> float | None:
        return self.harm_upper

    @property
    def G_hat(self) -> float | None:
        if self.predicted_benefit is None or self.predicted_harm is None:
            return None
        return float(self.predicted_benefit) - float(self.predicted_harm)

    @property
    def net_gain_lower(self) -> float | None:
        if self.benefit_lower is None or self.harm_upper is None:
            return None
        return float(self.benefit_lower) - float(self.harm_upper)

    @property
    def U_p_any_row_severe(self) -> float | None:
        return self.any_row_severe_probability_upper

    @property
    def U_harmed_frac(self) -> float | None:
        return self.harmed_pixel_fraction_upper

    @property
    def U_CVaR95(self) -> float | None:
        return self.pixel_harm_cvar95_upper

    @property
    def head_count(self) -> int:
        return 5


@dataclass(frozen=True)
class AuditEligibilityReceiptV7:
    """Independent, predeclared S6 evidence request; not an assessor head."""

    manifest_hash: str
    proposal_hash: str
    observation_hash: str
    candidate_id: str
    planned_arm_id: str
    independent_evidence_id: str
    independent_evidence_policy_hash: str
    interval_crosses_commit_boundary: bool
    value_of_information_lower: float
    prospective_runtime_evidence_cost: float
    prospective_runtime_evidence_cost_counters: CostCountersV7
    cost_account: CostAccountV7
    frozen_before_decision: bool
    provenance_hash: str
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("audit manifest hash", self.manifest_hash),
            ("audit proposal hash", self.proposal_hash),
            ("audit observation hash", self.observation_hash),
            ("audit candidate id", self.candidate_id),
            ("audit planned arm id", self.planned_arm_id),
            ("independent evidence id", self.independent_evidence_id),
            ("independent evidence policy hash",
             self.independent_evidence_policy_hash),
            ("audit provenance hash", self.provenance_hash),
        ):
            _require_text(value, name)
        if not isinstance(self.interval_crosses_commit_boundary, bool):
            raise ValueError("audit boundary-crossing flag must be boolean")
        if not isinstance(self.frozen_before_decision, bool):
            raise ValueError("audit frozen flag must be boolean")
        _nonnegative(self.value_of_information_lower, "audit VOI lower")
        _nonnegative(
            self.prospective_runtime_evidence_cost,
            "audit prospective-runtime evidence cost",
        )
        if not isinstance(
            self.prospective_runtime_evidence_cost_counters, CostCountersV7,
        ):
            raise ValueError("audit receipt needs typed cost counters")
        _enum(self.cost_account, CostAccountV7, "audit cost account")
        if self.cost_account is not CostAccountV7.PROSPECTIVE_RUNTIME:
            raise ValueError("audit evidence must use prospective-runtime account")
        payload = {
            "manifest_hash": self.manifest_hash,
            "proposal_hash": self.proposal_hash,
            "observation_hash": self.observation_hash,
            "candidate_id": self.candidate_id,
            "planned_arm_id": self.planned_arm_id,
            "independent_evidence_id": self.independent_evidence_id,
            "independent_evidence_policy_hash": (
                self.independent_evidence_policy_hash
            ),
            "interval_crosses_commit_boundary": (
                self.interval_crosses_commit_boundary
            ),
            "value_of_information_lower": self.value_of_information_lower,
            "prospective_runtime_evidence_cost": (
                self.prospective_runtime_evidence_cost
            ),
            "prospective_runtime_evidence_cost_counters_hash": (
                self.prospective_runtime_evidence_cost_counters.content_hash
            ),
            "cost_account": self.cost_account.value,
            "frozen_before_decision": self.frozen_before_decision,
            "provenance_hash": self.provenance_hash,
        }
        object.__setattr__(self, "receipt_hash", _sealed_hash(
            self.receipt_hash, payload, "audit eligibility receipt hash",
        ))


@dataclass(frozen=True)
class CalibrationReceiptV7:
    manifest_hash: str
    candidate_family_hash: str
    target_fold_role: FoldRoleV7
    calibration_fold_id: str
    model_hash: str
    feature_schema_hash: str
    calibration_data_hash: str
    calibration_policy_hash: str
    calibration_provenance_hash: str
    planner_allowlist_hash: str
    assessor_allowlist_hash: str
    assessor_feature_profile: str
    head_names: tuple[str, ...]
    severe_estimand: SevereEstimandV7
    selection_method: CalibrationMethodV7
    covered_planned_arm_ids: tuple[str, ...]
    frozen_before_decision: bool
    availability: AvailabilityV7
    availability_reason: str
    effective_components: int | None
    minimum_effective_components: int | None
    confidence_level: float | None
    familywise_error_rate: float | None
    calibration_quantile_index: int | None
    px_head_scale: float | None
    probability_head_scale: float | None
    minimum_fit_components_per_cell: int | None
    minimum_calibration_components_per_cell: int | None
    observed_fit_components_per_cell: int | None
    observed_calibration_components_per_cell: int | None
    row_severe_cutoff_g_raw_px: float | None
    cvar_level: float | None
    rho_any_row_severe: float | None
    maximum_harm: float | None
    maximum_harmed_fraction: float | None
    maximum_cvar95: float | None
    cost_penalty: float | None
    maximum_prospective_runtime_cost: float | None
    maximum_prospective_runtime_cost_counters: CostCountersV7 | None
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("calibration manifest hash", self.manifest_hash),
            ("calibration candidate-family hash", self.candidate_family_hash),
            ("calibration fold id", self.calibration_fold_id),
            ("calibration model hash", self.model_hash),
            ("calibration feature schema hash", self.feature_schema_hash),
            ("calibration data hash", self.calibration_data_hash),
            ("calibration policy hash", self.calibration_policy_hash),
            ("calibration provenance hash", self.calibration_provenance_hash),
            ("calibration planner allowlist hash", self.planner_allowlist_hash),
            ("calibration assessor allowlist hash", self.assessor_allowlist_hash),
            ("calibration assessor feature profile", self.assessor_feature_profile),
            ("calibration availability reason", self.availability_reason),
        ):
            _require_text(value, name)
        _enum(self.target_fold_role, FoldRoleV7, "calibration target fold role")
        _enum(self.selection_method, CalibrationMethodV7, "calibration method")
        _enum(self.availability, AvailabilityV7, "calibration availability")
        if self.assessor_feature_profile != ASSESSOR_FEATURE_PROFILE_V7:
            raise ValueError("calibration assessor feature profile drifted")
        if self.planner_allowlist_hash == self.assessor_allowlist_hash:
            raise ValueError("calibration stage allowlists cannot alias")
        if tuple(self.head_names) != PRIMARY_ASSESSOR_HEADS_V7:
            raise ValueError("v4.39 calibration cardinality must be exactly five")
        if self.severe_estimand is not SevereEstimandV7.EXECUTION_UNIT_ANY_ROW_SEVERE:
            raise ValueError("calibration severe estimand must be any-row-severe")
        covered = _unique_nonempty(
            self.covered_planned_arm_ids,
            "covered planned arm ids",
        )
        if not isinstance(self.frozen_before_decision, bool):
            raise ValueError("calibration frozen flag must be boolean")
        numeric = (
            self.effective_components, self.minimum_effective_components,
            self.confidence_level, self.familywise_error_rate,
            self.calibration_quantile_index, self.px_head_scale,
            self.probability_head_scale, self.minimum_fit_components_per_cell,
            self.minimum_calibration_components_per_cell,
            self.observed_fit_components_per_cell,
            self.observed_calibration_components_per_cell,
            self.row_severe_cutoff_g_raw_px, self.cvar_level,
            self.rho_any_row_severe, self.maximum_harm,
            self.maximum_harmed_fraction, self.maximum_cvar95,
            self.cost_penalty, self.maximum_prospective_runtime_cost,
        )
        if self.availability is AvailabilityV7.AVAILABLE:
            if any(item is None for item in numeric):
                raise ValueError("available calibration has typed-missing values")
            count_fields = (
                "effective_components", "minimum_effective_components",
                "calibration_quantile_index",
                "minimum_fit_components_per_cell",
                "minimum_calibration_components_per_cell",
                "observed_fit_components_per_cell",
                "observed_calibration_components_per_cell",
            )
            for name in count_fields:
                value = getattr(self, name)
                if (isinstance(value, bool) or not isinstance(value, int)
                        or value < 0):
                    raise ValueError(f"{name} must be a nonnegative integer")
            effective = int(self.effective_components)
            minimum = int(self.minimum_effective_components)
            if effective < minimum:
                raise ValueError("calibration has low support")
            if minimum != MINIMUM_GLOBAL_CALIBRATION_COMPONENTS_V7:
                raise ValueError("minimum global calibration count is frozen at 9")
            confidence = _fraction(self.confidence_level, "confidence level")
            alpha = _fraction(self.familywise_error_rate, "familywise error rate")
            if not 0.0 < confidence < 1.0 or not 0.0 < alpha < 1.0:
                raise ValueError("confidence and familywise error must be interior")
            if not math.isclose(
                alpha, CALIBRATION_ALPHA_COMPONENT_V7,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError("component calibration alpha is frozen at 0.10")
            expected_k = math.ceil((effective + 1) * (1.0 - alpha))
            if self.calibration_quantile_index != expected_k or expected_k > effective:
                raise ValueError("invalid or unreachable calibration quantile index")
            if not math.isclose(float(self.px_head_scale), PRIMARY_HEAD_SCALE_PX_V7,
                                rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("px head scale is frozen at 0.25")
            if not math.isclose(
                float(self.probability_head_scale),
                PRIMARY_HEAD_SCALE_PROBABILITY_V7,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError("probability head scale is frozen at 1.0")
            if self.minimum_fit_components_per_cell != (
                MINIMUM_FIT_COMPONENTS_PER_PRIMARY_CELL_V7
            ) or self.minimum_calibration_components_per_cell != (
                MINIMUM_CAL_COMPONENTS_PER_PRIMARY_CELL_V7
            ):
                raise ValueError("per-cell component minima are frozen")
            if (int(self.observed_fit_components_per_cell)
                    < int(self.minimum_fit_components_per_cell)
                    or int(self.observed_calibration_components_per_cell)
                    < int(self.minimum_calibration_components_per_cell)):
                raise ValueError("calibration cell has low support")
            if not math.isclose(
                float(self.row_severe_cutoff_g_raw_px),
                ROW_SEVERE_CUTOFF_G_RAW_PX_V7,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError("severe cutoff is frozen at -0.25 raw px")
            if not math.isclose(float(self.cvar_level), CVAR_LEVEL_V7,
                                rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("CVaR level is frozen at 0.95")
            _fraction(
                self.rho_any_row_severe,
                "rho any-row-severe",
            )
            _nonnegative(self.maximum_harm, "maximum harm")
            _fraction(self.maximum_harmed_fraction, "maximum harmed fraction")
            _nonnegative(self.maximum_cvar95, "maximum CVaR95")
            _nonnegative(self.cost_penalty, "cost penalty")
            _nonnegative(
                self.maximum_prospective_runtime_cost,
                "maximum prospective-runtime cost",
            )
            if not isinstance(
                self.maximum_prospective_runtime_cost_counters,
                CostCountersV7,
            ):
                raise ValueError(
                    "available calibration needs prospective-runtime counters"
                )
            constants = (
                (self.rho_any_row_severe, RHO_ANY_ROW_SEVERE_V7,
                 "rho any-row-severe"),
                (self.maximum_harm, TAU_H_MEAN_RAW_PX_V7, "tau H"),
                (self.maximum_harmed_fraction, TAU_HARMED_FRACTION_V7,
                 "tau harmed fraction"),
                (self.maximum_cvar95, TAU_CVAR95_RAW_PX_V7, "tau CVaR95"),
                (self.cost_penalty, LAMBDA_COST_V7, "lambda cost"),
            )
            for observed, expected, name in constants:
                if not math.isclose(float(observed), expected,
                                    rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"{name} differs from frozen v7.39 constant")
        else:
            if any(item is not None for item in numeric):
                raise ValueError("unavailable calibration values must be typed missing")
            if self.maximum_prospective_runtime_cost_counters is not None:
                raise ValueError("unavailable calibration counters must be typed missing")
        if self.selection_method is not CalibrationMethodV7.COMPONENT_MAX_RESIDUAL:
            raise ValueError("v7.39 primary requires the unique max-residual calibration")
        payload = {
            "manifest_hash": self.manifest_hash,
            "candidate_family_hash": self.candidate_family_hash,
            "target_fold_role": self.target_fold_role.value,
            "calibration_fold_id": self.calibration_fold_id,
            "model_hash": self.model_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "calibration_data_hash": self.calibration_data_hash,
            "calibration_policy_hash": self.calibration_policy_hash,
            "calibration_provenance_hash": self.calibration_provenance_hash,
            "planner_allowlist_hash": self.planner_allowlist_hash,
            "assessor_allowlist_hash": self.assessor_allowlist_hash,
            "assessor_feature_profile": self.assessor_feature_profile,
            "head_names": list(self.head_names),
            "severe_estimand": self.severe_estimand.value,
            "selection_method": self.selection_method.value,
            "covered_planned_arm_ids": list(covered),
            "frozen_before_decision": self.frozen_before_decision,
            "availability": self.availability.value,
            "availability_reason": self.availability_reason,
            "numeric": list(numeric),
            "maximum_prospective_runtime_cost_counters_hash": (
                self.maximum_prospective_runtime_cost_counters.content_hash
                if self.maximum_prospective_runtime_cost_counters is not None
                else None
            ),
        }
        object.__setattr__(self, "receipt_hash", _sealed_hash(
            self.receipt_hash, payload, "calibration receipt hash",
        ))


@dataclass(frozen=True)
class PortfolioDecisionV7:
    state: PortfolioStateV7
    manifest_hash: str
    proposal_hash: str
    calibration_receipt_hash: str
    native_candidate_id: str
    selected_candidate_id: str | None
    audit_candidate_id: str | None
    committed_unit_ids: tuple[str, ...]
    safe_candidate_ids: tuple[str, ...]
    conservative_utilities: tuple[tuple[str, float], ...]
    exclusion_states: tuple[tuple[str, CandidateExclusionStateV7], ...]
    reasons: tuple[DecisionReasonV7, ...]
    audit_count: int
    child_count: int

    def __post_init__(self) -> None:
        _enum(self.state, PortfolioStateV7, "portfolio state")
        for name, value in (
            ("decision manifest hash", self.manifest_hash),
            ("decision proposal hash", self.proposal_hash),
            ("decision calibration hash", self.calibration_receipt_hash),
            ("decision native candidate id", self.native_candidate_id),
        ):
            _require_text(value, name)
        if not self.reasons or any(not isinstance(row, DecisionReasonV7)
                                   for row in self.reasons):
            raise ValueError("decision needs typed reason codes")
        if self.audit_count not in {0, 1} or self.child_count not in {0, 1}:
            raise ValueError("decision audit and child counters are bounded by one")
        if (len({candidate for candidate, _ in self.exclusion_states})
                != len(self.exclusion_states)
                or any(not isinstance(stage, CandidateExclusionStateV7)
                       for _, stage in self.exclusion_states)):
            raise ValueError("candidate exclusion stages must be typed and unique")
        if self.state is PortfolioStateV7.COMMIT:
            if not self.selected_candidate_id or self.audit_candidate_id is not None:
                raise ValueError("commit decision needs exactly one selected candidate")
            if self.selected_candidate_id == self.native_candidate_id:
                raise ValueError("commit cannot name native")
            _unique_nonempty(self.committed_unit_ids, "committed unit ids")
        elif self.state is PortfolioStateV7.AUDIT:
            if self.selected_candidate_id is not None or not self.audit_candidate_id:
                raise ValueError("audit decision needs exactly one audit candidate")
            if self.committed_unit_ids:
                raise ValueError("audit cannot commit units")
        else:
            if self.selected_candidate_id != self.native_candidate_id:
                raise ValueError("native decision must select action-0 native")
            if self.audit_candidate_id is not None or self.committed_unit_ids:
                raise ValueError("native decision cannot audit or commit units")


def make_native_arm_v7(
    *, execution_unit_id: str, endpoint_id: str, support_hash: str,
    support_policy_hash: str, source_hash: str, provenance_hash: str,
    source_materialization_hash: str,
) -> CandidateArmV7:
    """Build the required action-0 arm for a manifest."""
    _require_text(execution_unit_id, "execution unit id")
    return CandidateArmV7(
        candidate_id="action0:native",
        action_index=0,
        action_identity="native",
        mechanism_id="native",
        operator_version="immutable-v1",
        exact_control_id="native",
        endpoint_id=endpoint_id,
        input_strength=0.0,
        output_beta=0.0,
        action_hash=_sha256(["native", endpoint_id]),
        control_hash=_sha256(["native-control", endpoint_id]),
        endpoint_hash=_sha256(["endpoint", endpoint_id]),
        support_hash=support_hash,
        support_policy_hash=support_policy_hash,
        rollback_hash=source_materialization_hash,
        source_input_hashes=(source_hash,),
        materialization_recipe_id="immutable_native",
        materialization_recipe_version="v1",
        materialization_recipe_hash=source_materialization_hash,
        provenance_hash=provenance_hash,
        scope=CandidateScopeV7.NATIVE,
        unit_ids=(execution_unit_id,),
        interaction_receipts=(),
        offline_bank_acquisition_cost_ceiling=0.0,
        offline_bank_acquisition_cost_counter_ceiling=CostCountersV7.zero(),
        prospective_runtime_cost_ceiling=0.0,
        prospective_runtime_cost_counter_ceiling=CostCountersV7.zero(),
        hard_legal=True,
    )


def _validated_realization_aliases_v7(
    manifest: CandidateManifestV7,
    realizations: Sequence[ArmRealizationReceiptV7],
    aliases: Sequence[ArmAliasReceiptV7],
) -> tuple[dict[str, ArmRealizationReceiptV7], dict[str, str]]:
    by_planned: dict[str, ArmRealizationReceiptV7] = {}
    for receipt in realizations:
        if receipt.planned_arm_id in by_planned:
            raise ValueError("planned arm has multiple realization receipts")
        arm = next(
            (row for row in manifest.candidates
             if row.planned_arm_id == receipt.planned_arm_id),
            None,
        )
        if arm is None:
            raise ValueError("realization receipt injected an unplanned arm")
        if (receipt.manifest_hash != manifest.manifest_hash
                or receipt.source_hash != manifest.source_hash):
            raise ValueError("realization receipt and manifest bindings differ")
        if receipt.status is ArmRealizationStatusV7.COMPLETE:
            if (receipt.realized_operator_hash != arm.action_hash
                    or float(receipt.offline_bank_acquisition_cost)
                    > arm.offline_bank_acquisition_cost_ceiling + 1e-12
                    or not receipt.offline_bank_acquisition_cost_counters.within(
                        arm.offline_bank_acquisition_cost_counter_ceiling
                    )):
                raise ValueError(
                    "offline realization binding or cost exceeds its planned ceiling"
                )
        by_planned[receipt.planned_arm_id] = receipt

    nonnative_complete = {
        planned_id: receipt
        for planned_id, receipt in by_planned.items()
        if receipt.status is ArmRealizationStatusV7.COMPLETE
        and next(row for row in manifest.candidates
                 if row.planned_arm_id == planned_id).action_index != 0
    }
    by_executable: dict[str, list[str]] = {}
    for planned_id, receipt in nonnative_complete.items():
        by_executable.setdefault(
            str(receipt.executable_bytes_hash), [],
        ).append(planned_id)
    expected_aliases: dict[str, str] = {}
    for members in by_executable.values():
        representative = min(members)
        for planned_id in members:
            if planned_id != representative:
                expected_aliases[planned_id] = representative

    observed_aliases: dict[str, str] = {}
    for receipt in aliases:
        if receipt.manifest_hash != manifest.manifest_hash:
            raise ValueError("alias receipt belongs to another manifest")
        if receipt.alias_planned_arm_id in observed_aliases:
            raise ValueError("planned arm has multiple alias receipts")
        representative = by_planned.get(receipt.representative_planned_arm_id)
        alias = by_planned.get(receipt.alias_planned_arm_id)
        if (representative is None or alias is None
                or representative.status is not ArmRealizationStatusV7.COMPLETE
                or alias.status is not ArmRealizationStatusV7.COMPLETE):
            raise ValueError("alias receipt requires two complete realizations")
        if (receipt.representative_realization_receipt_hash
                != representative.receipt_hash
                or receipt.alias_realization_receipt_hash != alias.receipt_hash
                or receipt.executable_bytes_hash
                != representative.executable_bytes_hash
                or receipt.executable_bytes_hash != alias.executable_bytes_hash
                or receipt.realized_operator_hash
                != representative.realized_operator_hash
                or receipt.realized_operator_hash != alias.realized_operator_hash):
            raise ValueError("alias receipt does not prove byte identity")
        observed_aliases[receipt.alias_planned_arm_id] = (
            receipt.representative_planned_arm_id
        )
    if observed_aliases != expected_aliases:
        raise ValueError("byte-identical realizations need exact canonical aliases")
    representatives = {
        planned_id: expected_aliases.get(planned_id, planned_id)
        for planned_id in by_planned
    }
    return by_planned, representatives


def _prospective_group_cost_v7(
    representative: CandidateArmV7,
    aliases: Sequence[ArmAliasReceiptV7],
) -> tuple[float, CostCountersV7]:
    """One future execution plus hash verification for each frozen alias."""
    scalar = float(representative.prospective_runtime_cost_ceiling)
    counters = representative.prospective_runtime_cost_counter_ceiling
    for alias in aliases:
        if alias.representative_planned_arm_id == representative.planned_arm_id:
            scalar += float(alias.prospective_runtime_verification_cost)
            counters = counters.plus(
                alias.prospective_runtime_verification_cost_counters,
            )
    return scalar, counters


def propose_candidates_v7(
    manifest: CandidateManifestV7,
    scores: Sequence[PlannerCandidateV7],
    realizations: Sequence[ArmRealizationReceiptV7],
    aliases: Sequence[ArmAliasReceiptV7],
    *,
    top_k: int,
    maximum_prospective_runtime_cost: float,
    maximum_prospective_runtime_cost_counters: CostCountersV7,
    planner_policy_hash: str,
    planner_provenance_hash: str,
) -> PlannerProposalV7:
    """Create a deterministic before-only K1/K2 soft proposal.

    Hard-rejected, native, and non-positive optimistic arms are ineligible.
    Proposal rank is optimistic utility, evidence value, lower cost, then
    lexicographic action identity/candidate id.  The committer may later
    re-rank the selected finite set, but cannot add another arm.
    """
    if top_k not in {1, 2}:
        raise ValueError("Selector-v7 top K must be exactly 1 or 2")
    ceiling = _nonnegative(
        maximum_prospective_runtime_cost,
        "planner prospective-runtime cost ceiling",
    )
    if not isinstance(maximum_prospective_runtime_cost_counters, CostCountersV7):
        raise ValueError("planner needs prospective-runtime counter ceiling")
    if not manifest.frozen_before_outcome:
        raise ValueError("planner requires a manifest frozen before outcomes")
    realization_by_id, representatives = _validated_realization_aliases_v7(
        manifest, realizations, aliases,
    )
    rows = tuple(scores)
    if len({row.candidate_id for row in rows}) != len(rows):
        raise ValueError("planner scores contain duplicate candidates")
    admitted: list[tuple[PlannerCandidateV7, CandidateArmV7]] = []
    for score in rows:
        arm = manifest.candidate(score.candidate_id)
        if arm is None:
            raise ValueError("planner attempted candidate injection")
        if (score.planned_arm_id != arm.planned_arm_id
                or score.cost_hash != arm.cost_hash
                or score.planner_allowlist_hash != manifest.planner_allowlist_hash):
            raise ValueError("planner candidate binding drifted")
        if arm.action_index == 0 or not arm.hard_legal:
            continue
        realization = realization_by_id.get(arm.planned_arm_id)
        if (realization is None
                or realization.status is not ArmRealizationStatusV7.COMPLETE):
            continue
        if representatives[arm.planned_arm_id] != arm.planned_arm_id:
            continue
        if arm.composition_reason() is not None:
            continue
        if not score.before_features_complete:
            continue
        admitted.append((score, arm))
    admitted.sort(key=lambda pair: (
        -pair[0].planner_predicted_gain_raw_px,
        _prospective_group_cost_v7(pair[1], aliases)[0],
        pair[1].planned_arm_id,
    ))
    selected_rows: list[PlannerCandidateV7] = []
    reserved = 0.0
    reserved_counters = CostCountersV7.zero(
        cache_state=maximum_prospective_runtime_cost_counters.cache_state,
    )
    seen_exact_arms: set[str] = set()
    for candidate, candidate_arm in admitted:
        if candidate_arm.planned_arm_id in seen_exact_arms:
            continue
        group_cost, group_counters = _prospective_group_cost_v7(
            candidate_arm, aliases,
        )
        next_counters = reserved_counters.plus(group_counters)
        if (reserved + group_cost > ceiling + 1e-12
                or not next_counters.within(
                    maximum_prospective_runtime_cost_counters,
                )):
            continue
        selected_rows.append(candidate)
        seen_exact_arms.add(candidate_arm.planned_arm_id)
        reserved += group_cost
        reserved_counters = next_counters
        if len(selected_rows) == top_k:
            break
    selected = tuple(selected_rows)
    return PlannerProposalV7(
        manifest_hash=manifest.manifest_hash,
        candidate_family_hash=manifest.candidate_family_hash,
        source_hash=manifest.source_hash,
        fold_role=manifest.fold_role,
        planner_allowlist_hash=manifest.planner_allowlist_hash,
        planner_policy_hash=planner_policy_hash,
        planner_provenance_hash=planner_provenance_hash,
        profile=(
            PlannerProfileV7.K1_ABLATION if top_k == 1
            else PlannerProfileV7.K2_PRIMARY
        ),
        top_k=top_k,
        maximum_prospective_runtime_cost=ceiling,
        reserved_prospective_runtime_cost=reserved,
        maximum_prospective_runtime_cost_counters=(
            maximum_prospective_runtime_cost_counters
        ),
        reserved_prospective_runtime_cost_counters=reserved_counters,
        frozen_before_observation=True,
        candidates=selected,
    )


def _availability_reason(status: AvailabilityV7) -> DecisionReasonV7:
    return {
        AvailabilityV7.UNKNOWN_UNOBSERVED_ARM: (
            DecisionReasonV7.UNKNOWN_UNOBSERVED_ARM
        ),
        AvailabilityV7.TYPED_MISSING: DecisionReasonV7.TYPED_MISSING,
        AvailabilityV7.OOD: DecisionReasonV7.OOD,
        AvailabilityV7.LOW_SUPPORT: DecisionReasonV7.LOW_SUPPORT,
        AvailabilityV7.AVAILABLE: DecisionReasonV7.MALFORMED_INPUT,
    }[status]


def _decision(
    *, state: PortfolioStateV7, manifest: object, proposal: object,
    calibration: object, reason: DecisionReasonV7,
    audit_count: int, child_count: int,
    selected: CandidateArmV7 | None = None,
    audit_candidate: CandidateArmV7 | None = None,
    safe: Sequence[CandidateArmV7] = (),
    utilities: Sequence[tuple[str, float]] = (),
    feature_ineligible_ids: Sequence[str] = (),
) -> PortfolioDecisionV7:
    manifest_hash = getattr(manifest, "manifest_hash", None) or "unbound-manifest"
    proposal_hash = getattr(proposal, "proposal_hash", None) or "unbound-proposal"
    calibration_hash = (
        getattr(calibration, "receipt_hash", None) or "unbound-calibration"
    )
    try:
        native = manifest.native
        native_id = native.candidate_id
    except Exception:
        native_id = "action0:native"
    bounded_audit = audit_count if audit_count in {0, 1} else 1
    bounded_child = child_count if child_count in {0, 1} else 1
    exclusions: list[tuple[str, CandidateExclusionStateV7]] = []
    try:
        proposed_ids = {row.candidate_id for row in proposal.candidates}
        ineligible = set(feature_ineligible_ids)
        for row in manifest.candidates:
            if row.action_index == 0:
                continue
            if not row.hard_legal:
                exclusions.append((
                    row.candidate_id,
                    CandidateExclusionStateV7.HARD_ILLEGAL_PREEXECUTION,
                ))
            elif row.candidate_id not in proposed_ids:
                exclusions.append((
                    row.candidate_id,
                    CandidateExclusionStateV7.SOFT_NOT_PROPOSED,
                ))
            elif row.candidate_id in ineligible:
                exclusions.append((
                    row.candidate_id,
                    CandidateExclusionStateV7.OBSERVED_BUT_FEATURE_INELIGIBLE,
                ))
    except Exception:
        exclusions = []
    return PortfolioDecisionV7(
        state=state,
        manifest_hash=str(manifest_hash),
        proposal_hash=str(proposal_hash),
        calibration_receipt_hash=str(calibration_hash),
        native_candidate_id=native_id,
        selected_candidate_id=(
            selected.candidate_id if selected is not None
            else native_id if state is PortfolioStateV7.NATIVE else None
        ),
        audit_candidate_id=(
            audit_candidate.candidate_id if audit_candidate is not None else None
        ),
        committed_unit_ids=(selected.unit_ids if selected is not None else ()),
        safe_candidate_ids=tuple(row.candidate_id for row in safe),
        conservative_utilities=tuple(utilities),
        exclusion_states=tuple(exclusions),
        reasons=(reason,),
        audit_count=bounded_audit,
        child_count=bounded_child,
    )


def _same_arm_binding(
    arm: CandidateArmV7,
    row: CandidateObservationV7 | ActionRiskVectorV7,
) -> bool:
    return (
        row.candidate_id == arm.candidate_id
        and row.planned_arm_id == arm.planned_arm_id
        and row.action_hash == arm.action_hash
        and row.control_hash == arm.control_hash
        and row.endpoint_hash == arm.endpoint_hash
        and row.support_hash == arm.support_hash
        and row.interaction_hash == arm.interaction_hash
        and row.cost_hash == arm.cost_hash
    )


def _rehash_valid(row: object) -> bool:
    """Re-run a frozen dataclass validator to detect post-decode mutation."""
    try:
        values = {
            field: getattr(row, field)
            for field in row.__dataclass_fields__  # type: ignore[attr-defined]
        }
        type(row)(**values)
    except Exception:
        return False
    return True


def select_portfolio_v7(
    manifest: CandidateManifestV7,
    proposal: PlannerProposalV7,
    observations: Sequence[CandidateObservationV7],
    risks: Sequence[ActionRiskVectorV7],
    calibration: CalibrationReceiptV7,
    realizations: Sequence[ArmRealizationReceiptV7],
    aliases: Sequence[ArmAliasReceiptV7],
    *,
    audit_count: int = 0,
    child_count: int = 0,
    audit_receipts: Sequence[AuditEligibilityReceiptV7] = (),
) -> PortfolioDecisionV7:
    """Choose the best conservative arm or fail closed to action-0 native."""
    if (isinstance(audit_count, bool) or not isinstance(audit_count, int)
            or audit_count < 0 or audit_count > 1):
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration,
            reason=DecisionReasonV7.AUDIT_BUDGET_EXHAUSTED,
            audit_count=1, child_count=child_count,
        )
    if (isinstance(child_count, bool) or not isinstance(child_count, int)
            or child_count < 0 or child_count > 1):
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration,
            reason=DecisionReasonV7.CHILD_BUDGET_EXHAUSTED,
            audit_count=audit_count, child_count=1,
        )
    objects = (
        manifest, proposal, calibration, *realizations, *aliases,
        *observations, *risks, *audit_receipts,
    )
    if any(not _rehash_valid(row) for row in objects):
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration, reason=DecisionReasonV7.MALFORMED_INPUT,
            audit_count=audit_count, child_count=child_count,
        )
    if not manifest.frozen_before_outcome:
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration, reason=DecisionReasonV7.MANIFEST_NOT_FROZEN,
            audit_count=audit_count, child_count=child_count,
        )
    try:
        realization_by_id, representatives = _validated_realization_aliases_v7(
            manifest, realizations, aliases,
        )
    except Exception:
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration, reason=DecisionReasonV7.MALFORMED_INPUT,
            audit_count=audit_count, child_count=child_count,
        )
    if (proposal.manifest_hash != manifest.manifest_hash
            or proposal.candidate_family_hash != manifest.candidate_family_hash
            or proposal.source_hash != manifest.source_hash
            or proposal.fold_role is not manifest.fold_role
            or proposal.planner_allowlist_hash
            != manifest.planner_allowlist_hash):
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration, reason=DecisionReasonV7.MANIFEST_DRIFT,
            audit_count=audit_count, child_count=child_count,
        )
    if not proposal.frozen_before_observation:
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration, reason=DecisionReasonV7.PROPOSAL_NOT_FROZEN,
            audit_count=audit_count, child_count=child_count,
        )
    if proposal.top_k not in {1, 2} or len(proposal.candidates) > proposal.top_k:
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration, reason=DecisionReasonV7.PROPOSAL_OVERFLOW,
            audit_count=audit_count, child_count=child_count,
        )
    expected_profile = (
        PlannerProfileV7.K1_ABLATION if proposal.top_k == 1
        else PlannerProfileV7.K2_PRIMARY
    )
    if proposal.profile is not expected_profile:
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration, reason=DecisionReasonV7.PROPOSAL_OVERFLOW,
            audit_count=audit_count, child_count=child_count,
        )
    proposed: dict[str, CandidateArmV7] = {}
    proposal_score_by_id: dict[str, PlannerCandidateV7] = {}
    for score in proposal.candidates:
        arm = manifest.candidate(score.candidate_id)
        if arm is None or arm.action_index == 0:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.CANDIDATE_INJECTION,
                audit_count=audit_count, child_count=child_count,
            )
        if arm.composition_reason() is not None:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.UNKNOWN_INTERACTION,
                audit_count=audit_count, child_count=child_count,
            )
        if not arm.hard_legal:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.HARD_REJECT_REOPEN_ATTEMPT,
                audit_count=audit_count, child_count=child_count,
            )
        realization = realization_by_id.get(arm.planned_arm_id)
        if (realization is None
                or realization.status is not ArmRealizationStatusV7.COMPLETE):
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.UNKNOWN_UNOBSERVED_ARM,
                audit_count=audit_count, child_count=child_count,
            )
        if representatives[arm.planned_arm_id] != arm.planned_arm_id:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.CANDIDATE_INJECTION,
                audit_count=audit_count, child_count=child_count,
            )
        if (score.planned_arm_id != arm.planned_arm_id
                or score.cost_hash != arm.cost_hash
                or score.planner_allowlist_hash
                != manifest.planner_allowlist_hash):
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.ACTION_BINDING_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
        proposed[arm.candidate_id] = arm
        proposal_score_by_id[arm.candidate_id] = score
    if not proposed:
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration, reason=DecisionReasonV7.EMPTY_PROPOSAL,
            audit_count=audit_count, child_count=child_count,
        )
    if (calibration.manifest_hash != manifest.manifest_hash
            or calibration.candidate_family_hash != manifest.candidate_family_hash
            or calibration.target_fold_role is not manifest.fold_role
            or calibration.planner_allowlist_hash
            != manifest.planner_allowlist_hash
            or calibration.assessor_allowlist_hash
            != manifest.assessor_allowlist_hash
            or not calibration.frozen_before_decision):
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration,
            reason=DecisionReasonV7.CALIBRATION_BINDING_MISMATCH,
            audit_count=audit_count, child_count=child_count,
        )
    if calibration.availability is not AvailabilityV7.AVAILABLE:
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration,
            reason=_availability_reason(calibration.availability),
            audit_count=audit_count, child_count=child_count,
        )
    identities = {arm.planned_arm_id for arm in proposed.values()}
    if not identities.issubset(calibration.covered_planned_arm_ids):
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration,
            reason=DecisionReasonV7.CALIBRATION_SELECTION_FAMILY_INCOMPLETE,
            audit_count=audit_count, child_count=child_count,
        )
    if calibration.selection_method is not CalibrationMethodV7.COMPONENT_MAX_RESIDUAL:
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration,
            reason=DecisionReasonV7.CALIBRATION_SELECTION_FAMILY_INCOMPLETE,
            audit_count=audit_count, child_count=child_count,
        )
    try:
        planned_groups = {
            row.candidate_id: _prospective_group_cost_v7(row, aliases)
            for row in proposed.values()
        }
    except ValueError:
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration,
            reason=DecisionReasonV7.COST_BUDGET_EXCEEDED,
            audit_count=audit_count, child_count=child_count,
        )
    reserved = sum(group[0] for group in planned_groups.values())
    reserved_counters = CostCountersV7.zero(
        cache_state=proposal.maximum_prospective_runtime_cost_counters.cache_state,
    )
    try:
        for _, counters in planned_groups.values():
            reserved_counters = reserved_counters.plus(counters)
    except ValueError:
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration,
            reason=DecisionReasonV7.COST_BUDGET_EXCEEDED,
            audit_count=audit_count, child_count=child_count,
        )
    if (not math.isclose(
            reserved, proposal.reserved_prospective_runtime_cost,
            rel_tol=0.0, abs_tol=1e-12,
        ) or reserved > proposal.maximum_prospective_runtime_cost + 1e-12
            or reserved_counters.content_hash
            != proposal.reserved_prospective_runtime_cost_counters.content_hash
            or not reserved_counters.within(
                proposal.maximum_prospective_runtime_cost_counters
            )
            or proposal.maximum_prospective_runtime_cost
            > float(calibration.maximum_prospective_runtime_cost) + 1e-12
            or not proposal.maximum_prospective_runtime_cost_counters.within(
                calibration.maximum_prospective_runtime_cost_counters
            )):
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration,
            reason=DecisionReasonV7.COST_BUDGET_EXCEEDED,
            audit_count=audit_count, child_count=child_count,
        )

    observation_by_id = {row.candidate_id: row for row in observations}
    risk_by_id = {row.candidate_id: row for row in risks}
    proposed_ids = set(proposed)
    if (len(observation_by_id) != len(observations)
            or len(risk_by_id) != len(risks)
            or set(observation_by_id) != proposed_ids
            or set(risk_by_id) != proposed_ids):
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration, reason=DecisionReasonV7.CANDIDATE_INJECTION,
            audit_count=audit_count, child_count=child_count,
        )

    safe_rows: list[tuple[CandidateArmV7, ActionRiskVectorV7, float]] = []
    risk_rejected = False
    any_row_severe_rejected = False
    utility_rejected = False
    for candidate_id, arm in proposed.items():
        observation = observation_by_id[candidate_id]
        risk = risk_by_id[candidate_id]
        realization = realization_by_id[arm.planned_arm_id]
        if (observation.manifest_hash != manifest.manifest_hash
                or observation.proposal_hash != proposal.proposal_hash
                or risk.manifest_hash != manifest.manifest_hash
                or risk.proposal_hash != proposal.proposal_hash
                or risk.observation_hash != observation.observation_hash
                or observation.planner_allowlist_hash
                != manifest.planner_allowlist_hash
                or observation.assessor_allowlist_hash
                != manifest.assessor_allowlist_hash
                or risk.assessor_allowlist_hash
                != manifest.assessor_allowlist_hash
                or observation.preaction_shared_feature_hash
                != proposal_score_by_id[candidate_id].before_feature_hash
                or not _same_arm_binding(arm, observation)
                or not _same_arm_binding(arm, risk)):
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.ACTION_BINDING_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
        if (observation.realization_receipt_hash != realization.receipt_hash
                or observation.realized_support_hash
                != realization.realized_support_hash):
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.ACTION_BINDING_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
        if risk.calibration_receipt_hash != calibration.receipt_hash:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.CALIBRATION_BINDING_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
        if observation.source_hash != manifest.source_hash:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.MANIFEST_DRIFT,
                audit_count=audit_count, child_count=child_count,
            )
        if observation.feature_schema_hash != calibration.feature_schema_hash:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.CALIBRATION_BINDING_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
        if observation.availability is not AvailabilityV7.AVAILABLE:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=_availability_reason(observation.availability),
                audit_count=audit_count, child_count=child_count,
                feature_ineligible_ids=(candidate_id,),
            )
        bad_blocks = [
            block for block in observation.evidence_blocks
            if block.availability is not AvailabilityV7.AVAILABLE
        ]
        if bad_blocks:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=_availability_reason(bad_blocks[0].availability),
                audit_count=audit_count, child_count=child_count,
                feature_ineligible_ids=(candidate_id,),
            )
        if risk.availability is not AvailabilityV7.AVAILABLE:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=_availability_reason(risk.availability),
                audit_count=audit_count, child_count=child_count,
                feature_ineligible_ids=(candidate_id,),
            )
        expected_units = set(arm.unit_ids)
        observed_units = set(observation.observed_unit_ids)
        if arm.scope is CandidateScopeV7.CASE_ATOMIC:
            if (not observation.atomically_observed
                    or observed_units != expected_units):
                return _decision(
                    state=PortfolioStateV7.NATIVE, manifest=manifest,
                    proposal=proposal, calibration=calibration,
                    reason=DecisionReasonV7.GLOBAL_PARTIAL_COMMIT,
                    audit_count=audit_count, child_count=child_count,
                )
        elif (arm.scope is not CandidateScopeV7.CERTIFIED_LOCAL
              or len(expected_units) != 1 or observed_units != expected_units):
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.LOCAL_SUPPORT_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
        group_cost, group_counters = planned_groups[candidate_id]
        if (observation.observed_prospective_runtime_cost
                > group_cost + 1e-12
                or not observation.observed_prospective_runtime_cost_counters.within(
                    group_counters
                )
                or group_cost
                > float(calibration.maximum_prospective_runtime_cost)):
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.COST_BUDGET_EXCEEDED,
                audit_count=audit_count, child_count=child_count,
            )
        if not risk.feature_eligible:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.FEATURE_INELIGIBLE,
                audit_count=audit_count, child_count=child_count,
                feature_ineligible_ids=(candidate_id,),
            )
        if not risk.calibration_valid:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.CALIBRATION_INVALID,
                audit_count=audit_count, child_count=child_count,
                feature_ineligible_ids=(candidate_id,),
            )
        if not risk.bound_bookkeeping_valid:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.FEATURE_INELIGIBLE,
                audit_count=audit_count, child_count=child_count,
                feature_ineligible_ids=(candidate_id,),
            )
        separate_risk_pass = (
            float(risk.any_row_severe_probability_upper)
            <= float(calibration.rho_any_row_severe)
            and float(risk.harm_upper) <= float(calibration.maximum_harm)
            and float(risk.harmed_pixel_fraction_upper)
            <= float(calibration.maximum_harmed_fraction)
            and float(risk.pixel_harm_cvar95_upper)
            <= float(calibration.maximum_cvar95)
        )
        objective = (
            float(risk.net_gain_lower)
            - LAMBDA_COST_V7 * planned_groups[candidate_id][0]
        )
        if separate_risk_pass and objective > 0.0:
            safe_rows.append((arm, risk, objective))
        else:
            risk_rejected = risk_rejected or not separate_risk_pass
            any_row_severe_rejected = (
                any_row_severe_rejected
                or float(risk.any_row_severe_probability_upper)
                > float(calibration.rho_any_row_severe)
            )
            utility_rejected = utility_rejected or objective <= 0.0

    safe_rows.sort(key=lambda item: (
        -item[2],
        float(item[1].any_row_severe_probability_upper),
        float(item[1].pixel_harm_cvar95_upper),
        planned_groups[item[0].candidate_id][0],
        item[0].action_identity,
        item[0].candidate_id,
    ))
    if safe_rows:
        selected = safe_rows[0][0]
        safe_arms = tuple(item[0] for item in safe_rows)
        utilities = tuple((item[0].candidate_id, item[2]) for item in safe_rows)
        return _decision(
            state=PortfolioStateV7.COMMIT, manifest=manifest,
            proposal=proposal, calibration=calibration,
            reason=DecisionReasonV7.SAFE_SET_MAXIMUM,
            audit_count=audit_count, child_count=child_count,
            selected=selected, safe=safe_arms, utilities=utilities,
        )
    audit_rows = tuple(audit_receipts)
    if len({row.candidate_id for row in audit_rows}) != len(audit_rows):
        return _decision(
            state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
            calibration=calibration, reason=DecisionReasonV7.MALFORMED_INPUT,
            audit_count=audit_count, child_count=child_count,
        )
    eligible_audits: list[
        tuple[float, CandidateArmV7, AuditEligibilityReceiptV7]
    ] = []
    for audit in audit_rows:
        arm = proposed.get(audit.candidate_id)
        seen = observation_by_id.get(audit.candidate_id)
        assessed = risk_by_id.get(audit.candidate_id)
        if arm is None or seen is None or assessed is None:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.CANDIDATE_INJECTION,
                audit_count=audit_count, child_count=child_count,
            )
        if (audit.manifest_hash != manifest.manifest_hash
                or audit.proposal_hash != proposal.proposal_hash
                or audit.observation_hash != seen.observation_hash
                or audit.planned_arm_id != arm.planned_arm_id):
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.ACTION_BINDING_MISMATCH,
                audit_count=audit_count, child_count=child_count,
            )
        try:
            audit_counters = (
                proposal.reserved_prospective_runtime_cost_counters.plus(
                    audit.prospective_runtime_evidence_cost_counters
                )
            )
        except ValueError:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.COST_BUDGET_EXCEEDED,
                audit_count=audit_count, child_count=child_count,
            )
        net_voi = (
            float(audit.value_of_information_lower)
            - float(audit.prospective_runtime_evidence_cost)
        )
        budget_ok = (
            proposal.reserved_prospective_runtime_cost
            + audit.prospective_runtime_evidence_cost
            <= float(calibration.maximum_prospective_runtime_cost) + 1e-12
            and audit_counters.within(
                calibration.maximum_prospective_runtime_cost_counters
            )
        )
        crosses = (
            audit.interval_crosses_commit_boundary
            and assessed.G_hat is not None and float(assessed.G_hat) > 0.0
            and assessed.net_gain_lower is not None
            and float(assessed.net_gain_lower) <= 0.0
        )
        other_risk_pass = (
            float(assessed.any_row_severe_probability_upper)
            <= float(calibration.rho_any_row_severe)
            and float(assessed.harm_upper) <= float(calibration.maximum_harm)
            and float(assessed.harmed_pixel_fraction_upper)
            <= float(calibration.maximum_harmed_fraction)
            and float(assessed.pixel_harm_cvar95_upper)
            <= float(calibration.maximum_cvar95)
        )
        if (audit.frozen_before_decision and crosses and other_risk_pass
                and net_voi > 0.0
                and budget_ok):
            eligible_audits.append((net_voi, arm, audit))
    if eligible_audits:
        if audit_count >= 1:
            return _decision(
                state=PortfolioStateV7.NATIVE, manifest=manifest,
                proposal=proposal, calibration=calibration,
                reason=DecisionReasonV7.AUDIT_BUDGET_EXHAUSTED,
                audit_count=audit_count, child_count=child_count,
            )
        eligible_audits.sort(key=lambda item: (
            -item[0], item[1].action_identity, item[1].candidate_id,
        ))
        return _decision(
            state=PortfolioStateV7.AUDIT, manifest=manifest,
            proposal=proposal, calibration=calibration,
            reason=DecisionReasonV7.AMBIGUITY_AUDIT,
            audit_count=audit_count + 1, child_count=child_count,
            audit_candidate=eligible_audits[0][1],
        )
    reason = (
        DecisionReasonV7.ANY_ROW_SEVERE_BUDGET_EXCEEDED
        if any_row_severe_rejected
        else DecisionReasonV7.RISK_BUDGET_EXCEEDED if risk_rejected
        else DecisionReasonV7.NONPOSITIVE_CONSERVATIVE_UTILITY
        if utility_rejected else DecisionReasonV7.NO_SAFE_CANDIDATE
    )
    return _decision(
        state=PortfolioStateV7.NATIVE, manifest=manifest, proposal=proposal,
        calibration=calibration, reason=reason,
        audit_count=audit_count, child_count=child_count,
    )


class CapacityIntervalStatusV7(str, Enum):
    ESTIMABLE = "estimable"
    NOT_ESTIMABLE = "not_estimable"
    NO_CURRENT_SEVERE_DENOMINATOR = "no_current_severe_denominator"
    CURRENT_ACTION_MEMBERSHIP_INCOMPLETE = (
        "current_action_membership_incomplete"
    )


class CapacityRouteV7(str, Enum):
    ACTION_BANK_BOTTLENECK = "action_bank_bottleneck"
    CAPACITY_SUFFICIENT_FOR_SELECTOR_TEST = (
        "capacity_sufficient_for_selector_test"
    )
    INCONCLUSIVE_CAPACITY_OR_MISSING_ARMS = (
        "inconclusive_capacity_or_missing_arms"
    )
    NO_CURRENT_SEVERE_CAPACITY_ROUTE_NOT_APPLICABLE = (
        "no_current_severe_capacity_route_not_applicable"
    )
    INCONCLUSIVE_CURRENT_ACTION_MEMBERSHIP = (
        "inconclusive_current_action_membership"
    )


class CapacityReasonV7(str, Enum):
    POSSIBLE_UPPER_BELOW_BANK_THRESHOLD = (
        "possible_upper_below_bank_threshold"
    )
    CONFIRMED_LOWER_AT_LEAST_SELECTOR_THRESHOLD = (
        "confirmed_lower_at_least_selector_threshold"
    )
    BOUNDS_INCONCLUSIVE = "bounds_inconclusive"
    CURRENT_ACTION_OUTCOME_MASK_OR_LINEAGE_UNKNOWN = (
        "current_action_outcome_mask_or_lineage_unknown"
    )
    NO_CURRENT_SEVERE_DENOMINATOR = "no_current_severe_denominator"
    NO_ELIGIBLE_COMPONENTS = "no_eligible_components"
    ZERO_VALID_MASK = "zero_valid_mask"
    MISSING_LINEAGE = "missing_lineage"
    INSUFFICIENT_VALID_BOOTSTRAPS = "insufficient_valid_bootstraps"


@dataclass(frozen=True)
class CapacityIntervalV7:
    status: CapacityIntervalStatusV7
    estimate: float | None
    lower: float | None
    upper: float | None
    valid_resamples: int
    total_resamples: int = 10_000
    bootstrap_seed: int = 20_261_003
    quantile_method: str = "inverted_cdf_nearest_rank"
    quantile_index_formula: str = "ceil(q*m)-1_clamped_to_[0,m-1]"
    lower_quantile: float = 0.025
    upper_quantile: float = 0.975
    resampling_unit: str = "independent_component_with_all_units"
    resample_size_rule: str = "n_original_components_with_replacement"
    invalid_resample_rule: str = "sampled_current_severe_denominator_equals_zero"

    def __post_init__(self) -> None:
        _enum(self.status, CapacityIntervalStatusV7, "capacity interval status")
        for name in ("valid_resamples", "total_resamples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.total_resamples != 10_000:
            raise ValueError("capacity bootstrap is frozen at 10,000 resamples")
        if self.bootstrap_seed != 20_261_003:
            raise ValueError("capacity bootstrap seed is frozen at 20261003")
        if self.quantile_method != "inverted_cdf_nearest_rank":
            raise ValueError("capacity CI requires inverted-CDF nearest rank")
        if self.quantile_index_formula != "ceil(q*m)-1_clamped_to_[0,m-1]":
            raise ValueError("capacity nearest-rank index formula drifted")
        if self.resampling_unit != "independent_component_with_all_units":
            raise ValueError("capacity bootstrap must resample components")
        if self.resample_size_rule != "n_original_components_with_replacement":
            raise ValueError("capacity bootstrap resample-size rule drifted")
        if self.invalid_resample_rule != (
            "sampled_current_severe_denominator_equals_zero"
        ):
            raise ValueError("capacity invalid-resample rule drifted")
        if (not math.isclose(self.lower_quantile, 0.025, abs_tol=1e-12)
                or not math.isclose(self.upper_quantile, 0.975, abs_tol=1e-12)):
            raise ValueError("capacity CI quantiles are frozen at 0.025/0.975")
        if self.valid_resamples > self.total_resamples:
            raise ValueError("valid bootstrap resamples exceed total resamples")
        values = (self.estimate, self.lower, self.upper)
        if self.status is CapacityIntervalStatusV7.ESTIMABLE:
            if self.valid_resamples < 9_500 or any(value is None for value in values):
                raise ValueError("estimable capacity requires at least 9,500 resamples")
            estimate, lower, upper = (
                _fraction(value, "capacity interval value") for value in values
            )
            if not lower <= estimate <= upper:
                raise ValueError("capacity interval ordering is invalid")
        else:
            if any(value is not None for value in values):
                raise ValueError("non-estimable capacity interval must be typed missing")
            if (self.status is CapacityIntervalStatusV7.NOT_ESTIMABLE
                    and self.valid_resamples >= 9_500):
                raise ValueError("NOT_ESTIMABLE needs fewer than 9,500 valid resamples")
            if (self.status
                    is CapacityIntervalStatusV7.NO_CURRENT_SEVERE_DENOMINATOR
                    and self.valid_resamples != 0):
                raise ValueError("zero denominator must have zero valid resamples")
            if (self.status is
                    CapacityIntervalStatusV7.CURRENT_ACTION_MEMBERSHIP_INCOMPLETE
                    and self.valid_resamples != 0):
                raise ValueError("unknown membership cannot run a bootstrap")


@dataclass(frozen=True)
class CapacityReportV7:
    """SR0B schema only; it neither decodes targets nor performs bootstrap."""

    fixed_current_any_row_severe_denominator: int | None
    eligible_independent_components: int
    confirmed_safe_capacity_numerator: int | None
    possible_safe_capacity_numerator: int | None
    confirmed_interval: CapacityIntervalV7
    possible_interval: CapacityIntervalV7
    current_action_membership_complete: bool
    lineage_complete: bool
    valid_masks_nonzero: bool
    route: CapacityRouteV7
    reason_code: CapacityReasonV7
    confirmed_unknown_policy: str = "unknown_not_confirmed"
    possible_unknown_policy: str = "hard_legal_unknown_may_be_safe"

    def __post_init__(self) -> None:
        for name in ("eligible_independent_components",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.confirmed_interval, CapacityIntervalV7) or not isinstance(
            self.possible_interval, CapacityIntervalV7
        ):
            raise ValueError("capacity report needs typed confirmed/possible intervals")
        if (not isinstance(self.current_action_membership_complete, bool)
                or not isinstance(self.lineage_complete, bool)
                or not isinstance(self.valid_masks_nonzero, bool)):
            raise ValueError("capacity lineage/mask flags must be boolean")
        _enum(self.route, CapacityRouteV7, "capacity route")
        _enum(self.reason_code, CapacityReasonV7, "capacity reason code")
        if self.confirmed_unknown_policy != "unknown_not_confirmed":
            raise ValueError("confirmed capacity unknown policy drifted")
        if self.possible_unknown_policy != "hard_legal_unknown_may_be_safe":
            raise ValueError("possible capacity unknown policy drifted")
        if not self.current_action_membership_complete:
            if any(value is not None for value in (
                self.fixed_current_any_row_severe_denominator,
                self.confirmed_safe_capacity_numerator,
                self.possible_safe_capacity_numerator,
            )):
                raise ValueError("unknown membership counts must be typed missing")
            if (self.route
                    is not CapacityRouteV7.INCONCLUSIVE_CURRENT_ACTION_MEMBERSHIP
                    or self.reason_code is not
                    CapacityReasonV7.CURRENT_ACTION_OUTCOME_MASK_OR_LINEAGE_UNKNOWN
                    or self.confirmed_interval.status is not
                    CapacityIntervalStatusV7.CURRENT_ACTION_MEMBERSHIP_INCOMPLETE
                    or self.possible_interval.status is not
                    CapacityIntervalStatusV7.CURRENT_ACTION_MEMBERSHIP_INCOMPLETE):
                raise ValueError("unknown current-action membership must fail closed")
            return
        for name in (
            "fixed_current_any_row_severe_denominator",
            "confirmed_safe_capacity_numerator",
            "possible_safe_capacity_numerator",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        denominator = int(self.fixed_current_any_row_severe_denominator)
        if not (0 <= int(self.confirmed_safe_capacity_numerator)
                <= int(self.possible_safe_capacity_numerator) <= denominator):
            raise ValueError("confirmed/possible capacity bounds are inconsistent")
        if denominator == 0:
            expected_route = (
                CapacityRouteV7.NO_CURRENT_SEVERE_CAPACITY_ROUTE_NOT_APPLICABLE
            )
            expected_status = (
                CapacityIntervalStatusV7.NO_CURRENT_SEVERE_DENOMINATOR
            )
            if (self.route is not expected_route
                    or self.reason_code is not
                    CapacityReasonV7.NO_CURRENT_SEVERE_DENOMINATOR
                    or self.confirmed_interval.status is not expected_status
                    or self.possible_interval.status is not expected_status):
                raise ValueError("zero denominator route must be not applicable")
            return
        if self.eligible_independent_components == 0:
            if (self.route is not
                    CapacityRouteV7.INCONCLUSIVE_CAPACITY_OR_MISSING_ARMS
                    or self.reason_code is not
                    CapacityReasonV7.NO_ELIGIBLE_COMPONENTS
                    or self.confirmed_interval.status is not
                    CapacityIntervalStatusV7.NOT_ESTIMABLE
                    or self.possible_interval.status is not
                    CapacityIntervalStatusV7.NOT_ESTIMABLE):
                raise ValueError("no eligible components must route inconclusive")
            return
        if not self.lineage_complete:
            if (self.route is not
                    CapacityRouteV7.INCONCLUSIVE_CAPACITY_OR_MISSING_ARMS
                    or self.reason_code is not CapacityReasonV7.MISSING_LINEAGE
                    or self.confirmed_interval.status is not
                    CapacityIntervalStatusV7.NOT_ESTIMABLE
                    or self.possible_interval.status is not
                    CapacityIntervalStatusV7.NOT_ESTIMABLE):
                raise ValueError("missing lineage must route inconclusive")
            return
        if not self.valid_masks_nonzero:
            if (self.route is not
                    CapacityRouteV7.INCONCLUSIVE_CAPACITY_OR_MISSING_ARMS
                    or self.reason_code is not CapacityReasonV7.ZERO_VALID_MASK
                    or self.confirmed_interval.status is not
                    CapacityIntervalStatusV7.NOT_ESTIMABLE
                    or self.possible_interval.status is not
                    CapacityIntervalStatusV7.NOT_ESTIMABLE):
                raise ValueError("zero valid masks must route inconclusive")
            return
        estimable = (
            self.confirmed_interval.status is CapacityIntervalStatusV7.ESTIMABLE
            and self.possible_interval.status is CapacityIntervalStatusV7.ESTIMABLE
            and self.lineage_complete and self.valid_masks_nonzero
        )
        if not estimable:
            if (self.route is not
                    CapacityRouteV7.INCONCLUSIVE_CAPACITY_OR_MISSING_ARMS
                    or self.reason_code is not
                    CapacityReasonV7.INSUFFICIENT_VALID_BOOTSTRAPS):
                raise ValueError("non-estimable capacity must route inconclusive")
            return
        confirmed_estimate = (
            int(self.confirmed_safe_capacity_numerator) / denominator
        )
        possible_estimate = (
            int(self.possible_safe_capacity_numerator) / denominator
        )
        if (not math.isclose(
                float(self.confirmed_interval.estimate), confirmed_estimate,
                rel_tol=0.0, abs_tol=1e-12,
            ) or not math.isclose(
                float(self.possible_interval.estimate), possible_estimate,
                rel_tol=0.0, abs_tol=1e-12,
            )):
            raise ValueError("capacity estimates disagree with fixed-denominator counts")
        if float(self.possible_interval.upper) < 0.20:
            expected = CapacityRouteV7.ACTION_BANK_BOTTLENECK
            expected_reason = CapacityReasonV7.POSSIBLE_UPPER_BELOW_BANK_THRESHOLD
        elif float(self.confirmed_interval.lower) >= 0.50:
            expected = CapacityRouteV7.CAPACITY_SUFFICIENT_FOR_SELECTOR_TEST
            expected_reason = (
                CapacityReasonV7.CONFIRMED_LOWER_AT_LEAST_SELECTOR_THRESHOLD
            )
        else:
            expected = CapacityRouteV7.INCONCLUSIVE_CAPACITY_OR_MISSING_ARMS
            expected_reason = CapacityReasonV7.BOUNDS_INCONCLUSIVE
        if self.route is not expected or self.reason_code is not expected_reason:
            raise ValueError("capacity route disagrees with frozen bounds")


@dataclass(frozen=True)
class ConfirmatoryInferenceContractV7:
    """Schema fingerprint only; SR1 does not execute the SR4 bootstrap."""

    hypotheses: tuple[str, str] = ("H1_S5_vs_S1", "H2_S5_vs_S3M_B4_K2")
    component_difference: str = "baseline_severe_rows_minus_S5_severe_rows"
    eligibility_requirement: str = "n_gt_0_and_all_component_differences_finite"
    centering: str = "subtract_grand_component_mean"
    bootstrap_statistic: str = "mean_resampled_centered_component_residual"
    resampling_unit: str = "eligible_independent_component"
    resample_size_rule: str = "n_original_components_with_replacement"
    resamples: int = 10_000
    minimum_valid_draws: int = 10_000
    seed: int = 20_261_003
    shared_rng_seed_stream: bool = True
    raw_p_formula: str = "(1+count(T_boot>=T_observed))/10001"
    familywise_alpha: float = 0.05
    holm_first_threshold: float = 0.025
    holm_second_threshold: float = 0.05
    tie_order: tuple[str, str] = ("H1", "H2")
    both_rejections_required_for_sr5: bool = True
    sr5_authority: str = "point_conjunction_and_both_holm_stepdown_rejections"
    adjusted_p_formula: str = (
        "first=min(1,2*p1);second=min(1,max(2*p1,p2));restore_original_order"
    )

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            if getattr(self, name) != field.default:
                raise ValueError(f"v4.39 inference contract field drifted: {name}")


@dataclass(frozen=True)
class NumericProvenanceV7:
    constant_name: str
    value: float
    provenance: str
    estimand_semantics: str


NUMERIC_PROVENANCE_V7 = (
    NumericProvenanceV7(
        "rho_any_row_severe", RHO_ANY_ROW_SEVERE_V7,
        "selector_v6_numeric_default_only_new_v7_estimand_no_v6_authority",
        "execution_unit_any_eligible_delivered_row_G_below_minus_0.25",
    ),
    NumericProvenanceV7(
        "tau_H_mean_raw_px", TAU_H_MEAN_RAW_PX_V7,
        "numeric_reference_selector_v6", "execution_unit_pixel_mean_harm",
    ),
    NumericProvenanceV7(
        "tau_pixel_harm_CVaR95_raw_px", TAU_CVAR95_RAW_PX_V7,
        "numeric_reference_selector_v6", "union_valid_pixel_harm_cvar95",
    ),
    NumericProvenanceV7(
        "pixel_harm_CVaR_level", CVAR_LEVEL_V7,
        "numeric_reference_selector_v6", "union_valid_pixel_harm_cvar_level",
    ),
    NumericProvenanceV7(
        "tau_harmed_fraction", TAU_HARMED_FRACTION_V7,
        "new_v7_preregistration", "union_valid_harmed_pixel_fraction",
    ),
    NumericProvenanceV7(
        "row_severe_cutoff_G_raw_px", ROW_SEVERE_CUTOFF_G_RAW_PX_V7,
        "new_v7_preregistered_net_gain_semantics",
        "row_same_mask_G_strictly_below_minus_0.25",
    ),
)


PLANNED_ARM_ID_FIELDS_V7 = (
    "mechanism_id", "operator_version", "endpoint_id", "input_strength",
    "output_beta", "support_policy_hash", "source_input_hashes",
    "materialization_recipe_id", "materialization_recipe_version",
    "materialization_recipe_hash", "execution_unit_ids",
)


def selector_v7_schema_fingerprint() -> str:
    schema_classes = (
        CandidateArmV7, ArmRealizationReceiptV7, ArmAliasReceiptV7,
        CandidateManifestV7, PlannerCandidateV7, PlannerProposalV7,
        CandidateObservationV7, ActionRiskVectorV7, AuditEligibilityReceiptV7,
        CalibrationReceiptV7,
        PortfolioDecisionV7, CapacityIntervalV7, CapacityReportV7,
        FeatureFieldV7, FeatureAllowlistsV7,
        ConfirmatoryInferenceContractV7,
    )
    enums = (
        AvailabilityV7, CandidateExclusionStateV7, CandidateScopeV7,
        DecisionReasonV7, FeatureStageV7, FeatureDependencyKindV7,
        ArmRealizationStatusV7, SevereEstimandV7, CostAccountV7,
        CapacityIntervalStatusV7, CapacityRouteV7, CapacityReasonV7,
    )
    payload = {
        "schema_version": SELECTOR_V7_SCHEMA_VERSION,
        "canonicalization_version": SELECTOR_V7_CANONICALIZATION_VERSION,
        "classes": {
            cls.__name__: list(cls.__dataclass_fields__)
            for cls in schema_classes
        },
        "enums": {
            enum.__name__: [[row.name, row.value] for row in enum]
            for enum in enums
        },
        "planned_arm_id_fields": list(PLANNED_ARM_ID_FIELDS_V7),
        "assessor_heads": list(PRIMARY_ASSESSOR_HEADS_V7),
        "preaction_shared_fields": sorted(PREACTION_SHARED_FIELDS_V7),
        "planner_extension_fields": sorted(PLANNER_EXTENSION_FIELDS_V7),
        "assessor_postaction_extension_fields": sorted(
            ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7
        ),
        "planner_fields": sorted(PLANNER_BEFORE_ONLY_FIELDS_V7),
        "assessor_fields": sorted(ASSESSOR_POST_ACTION_FIELDS_V7),
        "numeric_provenance": [
            [row.constant_name, row.value, row.provenance, row.estimand_semantics]
            for row in NUMERIC_PROVENANCE_V7
        ],
    }
    return _sha256(payload)


@dataclass(frozen=True)
class SelectorV7IntegrationFingerprint:
    schema_version: str
    canonicalization_version: str
    schema_fingerprint: str
    reason_enum_hash: str
    stage_allowlist_name_hash: str
    workplan_schema: str
    inference_contract_hash: str


def selector_v7_integration_fingerprint() -> SelectorV7IntegrationFingerprint:
    return SelectorV7IntegrationFingerprint(
        schema_version=SELECTOR_V7_SCHEMA_VERSION,
        canonicalization_version=SELECTOR_V7_CANONICALIZATION_VERSION,
        schema_fingerprint=selector_v7_schema_fingerprint(),
        reason_enum_hash=_sha256([
            [enum.__name__, row.name, row.value]
            for enum in (
                DecisionReasonV7, HardRejectReasonV7, CapacityReasonV7,
                CapacityRouteV7,
            )
            for row in enum
        ]),
        stage_allowlist_name_hash=_sha256({
            "shared": sorted(PREACTION_SHARED_FIELDS_V7),
            "planner_extension": sorted(PLANNER_EXTENSION_FIELDS_V7),
            "assessor_postaction_extension": sorted(
                ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7
            ),
        }),
        workplan_schema="selector-v7-redesign-workplan/v4",
        inference_contract_hash=_sha256({
            name: getattr(ConfirmatoryInferenceContractV7(), name)
            for name in ConfirmatoryInferenceContractV7.__dataclass_fields__
        }),
    )


__all__ = [
    "ActionRiskVectorV7",
    "ArmAliasReceiptV7",
    "ArmRealizationReceiptV7",
    "ArmRealizationStatusV7",
    "AuditEligibilityReceiptV7",
    "ASSESSOR_FEATURE_PROFILE_V7",
    "ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7",
    "ASSESSOR_POST_ACTION_FIELDS_V7",
    "AvailabilityV7",
    "CALIBRATION_ALPHA_COMPONENT_V7",
    "CVAR_LEVEL_V7",
    "CalibrationMethodV7",
    "CalibrationReceiptV7",
    "CapacityIntervalStatusV7",
    "CapacityIntervalV7",
    "CapacityReasonV7",
    "CapacityReportV7",
    "CapacityRouteV7",
    "CandidateArmV7",
    "CandidateExclusionStateV7",
    "CandidateManifestV7",
    "CandidateObservationV7",
    "CandidateScopeV7",
    "ConfirmatoryInferenceContractV7",
    "CostAccountV7",
    "CostCountersV7",
    "DecisionReasonV7",
    "EvidenceBlockReceiptV7",
    "EvidenceBlockV7",
    "FeatureAllowlistsV7",
    "FeatureDependencyKindV7",
    "FeatureFieldV7",
    "FeatureStageV7",
    "FoldRoleV7",
    "HardRejectReasonV7",
    "InteractionKindV7",
    "InteractionReceiptV7",
    "LAMBDA_COST_V7",
    "MINIMUM_CAL_COMPONENTS_PER_PRIMARY_CELL_V7",
    "MINIMUM_FIT_COMPONENTS_PER_PRIMARY_CELL_V7",
    "MINIMUM_GLOBAL_CALIBRATION_COMPONENTS_V7",
    "NUMERIC_PROVENANCE_V7",
    "NumericProvenanceV7",
    "PRIMARY_HEAD_SCALE_PROBABILITY_V7",
    "PRIMARY_HEAD_SCALE_PX_V7",
    "PRIMARY_ASSESSOR_HEADS_V7",
    "PRIMARY_REQUIRED_EVIDENCE_BLOCKS_V7",
    "PLANNED_ARM_ID_FIELDS_V7",
    "PLANNER_BEFORE_ONLY_FIELDS_V7",
    "PLANNER_EXTENSION_FIELDS_V7",
    "PLANNER_FEATURE_PROFILE_V7",
    "PlannerCandidateV7",
    "PlannerProfileV7",
    "PlannerProposalV7",
    "PortfolioDecisionV7",
    "PortfolioStateV7",
    "PredictionKindV7",
    "PREACTION_SHARED_FIELDS_V7",
    "RuntimeFeatureV7",
    "RHO_ANY_ROW_SEVERE_V7",
    "ROW_SEVERE_CUTOFF_G_RAW_PX_V7",
    "SELECTOR_V7_CANONICALIZATION_VERSION",
    "SELECTOR_V7_SCHEMA_VERSION",
    "SelectorV7IntegrationFingerprint",
    "SevereEstimandV7",
    "TAU_CVAR95_RAW_PX_V7",
    "TAU_HARMED_FRACTION_V7",
    "TAU_H_MEAN_RAW_PX_V7",
    "V7_PRIMARY_FEATURE_PROFILE",
    "make_native_arm_v7",
    "propose_candidates_v7",
    "select_portfolio_v7",
    "selector_v7_integration_fingerprint",
    "selector_v7_schema_fingerprint",
]
