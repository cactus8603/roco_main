"""Auditable uncertainty bindings around family action competition.

This successor layer keeps the frozen Selector-v7 feature schema intact while
proving that a planner score consumed a particular native uncertainty receipt
and that a post-action assessor consumed a particular native/child delta.
Provider-specific names (for example Work B or U2Flow) never enter the selector
contract; only sealed provider-neutral receipts do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Sequence

from .family_action_competition import (
    ActionFamilyManifestV1,
    FamilyAwareDecisionV1,
    FamilyAwareProposalV1,
    propose_family_competition_v1,
    select_family_portfolio_v1,
)
from .selector_v7 import (
    ActionRiskVectorV7,
    ArmAliasReceiptV7,
    ArmRealizationReceiptV7,
    AuditEligibilityReceiptV7,
    CalibrationReceiptV7,
    CandidateManifestV7,
    CandidateObservationV7,
    CostCountersV7,
    PlannerCandidateV7,
)
from .uncertainty_aware_flow import (
    UncertaintyAvailabilityV1,
    UncertaintyReceiptV1,
)


UNCERTAINTY_FAMILY_COMPETITION_SCHEMA_V1 = (
    "stablebridge-uncertainty-family-competition/v1"
)


def _hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty canonical text")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PreActionUncertaintyArtifactV1:
    candidate_id: str
    planned_arm_id: str
    base_before_feature_hash: str
    native_uncertainty_receipt_hash: str
    uncertainty_spatial_feature_hash: str
    feature_schema_hash: str
    availability: UncertaintyAvailabilityV1
    availability_reason: str
    artifact_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate id", self.candidate_id),
            ("planned arm id", self.planned_arm_id),
            ("base before feature hash", self.base_before_feature_hash),
            ("native uncertainty receipt hash", self.native_uncertainty_receipt_hash),
            ("uncertainty spatial feature hash", self.uncertainty_spatial_feature_hash),
            ("uncertainty feature schema hash", self.feature_schema_hash),
            ("uncertainty availability reason", self.availability_reason),
        ):
            _text(value, name)
        if not isinstance(self.availability, UncertaintyAvailabilityV1):
            raise ValueError("pre-action uncertainty availability must be typed")
        expected = _hash({
            "candidate_id": self.candidate_id,
            "planned_arm_id": self.planned_arm_id,
            "base_before_feature_hash": self.base_before_feature_hash,
            "native_uncertainty_receipt_hash": self.native_uncertainty_receipt_hash,
            "uncertainty_spatial_feature_hash": self.uncertainty_spatial_feature_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "availability": self.availability.value,
            "availability_reason": self.availability_reason,
        })
        if self.artifact_hash and self.artifact_hash != expected:
            raise ValueError("pre-action uncertainty artifact hash drift")
        object.__setattr__(self, "artifact_hash", expected)


@dataclass(frozen=True)
class UncertaintyBoundPlannerScoreV1:
    score: PlannerCandidateV7
    artifact: PreActionUncertaintyArtifactV1
    model_input_artifact_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.score, PlannerCandidateV7):
            raise ValueError("uncertainty planner score must be typed")
        if not isinstance(self.artifact, PreActionUncertaintyArtifactV1):
            raise ValueError("uncertainty planner artifact must be typed")
        if (
            self.score.candidate_id != self.artifact.candidate_id
            or self.score.planned_arm_id != self.artifact.planned_arm_id
        ):
            raise ValueError("uncertainty planner candidate binding drift")
        if (
            self.score.before_feature_hash != self.artifact.artifact_hash
            or self.model_input_artifact_hash != self.artifact.artifact_hash
        ):
            raise ValueError("planner model did not bind the uncertainty artifact")
        if (
            self.artifact.availability is not UncertaintyAvailabilityV1.AVAILABLE
            and self.score.before_features_complete
        ):
            raise ValueError("missing uncertainty cannot produce a complete planner score")


def preaction_uncertainty_artifact_v1(
    *,
    candidate_id: str,
    planned_arm_id: str,
    base_before_feature_hash: str,
    native_uncertainty: UncertaintyReceiptV1,
    uncertainty_spatial_feature_hash: str,
    feature_schema_hash: str,
) -> PreActionUncertaintyArtifactV1:
    if native_uncertainty.observation_role != "native":
        raise ValueError("pre-action artifact needs native uncertainty")
    return PreActionUncertaintyArtifactV1(
        candidate_id=candidate_id,
        planned_arm_id=planned_arm_id,
        base_before_feature_hash=base_before_feature_hash,
        native_uncertainty_receipt_hash=native_uncertainty.receipt_hash,
        uncertainty_spatial_feature_hash=uncertainty_spatial_feature_hash,
        feature_schema_hash=feature_schema_hash,
        availability=native_uncertainty.availability,
        availability_reason=native_uncertainty.availability_reason,
    )


def propose_uncertainty_aware_family_competition_v1(
    selector_manifest: CandidateManifestV7,
    family_manifest: ActionFamilyManifestV1,
    bound_scores: Sequence[UncertaintyBoundPlannerScoreV1],
    realizations: Sequence[ArmRealizationReceiptV7],
    aliases: Sequence[ArmAliasReceiptV7],
    *,
    top_k: int,
    maximum_prospective_runtime_cost: float,
    maximum_prospective_runtime_cost_counters: CostCountersV7,
    cost_penalty_raw_px_per_cost_unit: float,
    planner_policy_hash: str,
    base_planner_provenance_hash: str,
) -> FamilyAwareProposalV1:
    rows = tuple(bound_scores)
    if not rows or any(not isinstance(row, UncertaintyBoundPlannerScoreV1) for row in rows):
        raise ValueError("uncertainty-aware proposal needs bound planner scores")
    provenance = _hash({
        "base_planner_provenance_hash": _text(
            base_planner_provenance_hash, "base planner provenance hash"
        ),
        "uncertainty_artifact_hashes": [
            row.artifact.artifact_hash
            for row in sorted(rows, key=lambda item: item.score.candidate_id)
        ],
    })
    return propose_family_competition_v1(
        selector_manifest,
        family_manifest,
        tuple(row.score for row in rows),
        realizations,
        aliases,
        top_k=top_k,
        maximum_prospective_runtime_cost=maximum_prospective_runtime_cost,
        maximum_prospective_runtime_cost_counters=(
            maximum_prospective_runtime_cost_counters
        ),
        cost_penalty_raw_px_per_cost_unit=cost_penalty_raw_px_per_cost_unit,
        planner_policy_hash=planner_policy_hash,
        planner_provenance_hash=provenance,
    )


@dataclass(frozen=True)
class PostActionUncertaintyDeltaV1:
    candidate_id: str
    observation_hash: str
    native_uncertainty_receipt_hash: str
    child_uncertainty_receipt_hash: str
    mean_delta: float
    p90_delta: float
    p99_delta: float
    resolved_fraction: float
    newly_uncertain_fraction: float
    delta_spatial_feature_hash: str
    evidence_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate id", self.candidate_id),
            ("observation hash", self.observation_hash),
            ("native uncertainty receipt hash", self.native_uncertainty_receipt_hash),
            ("child uncertainty receipt hash", self.child_uncertainty_receipt_hash),
            ("delta spatial feature hash", self.delta_spatial_feature_hash),
        ):
            _text(value, name)
        deltas = tuple(_finite(value, name) for name, value in (
            ("uncertainty mean delta", self.mean_delta),
            ("uncertainty p90 delta", self.p90_delta),
            ("uncertainty p99 delta", self.p99_delta),
            ("resolved uncertainty fraction", self.resolved_fraction),
            ("newly uncertain fraction", self.newly_uncertain_fraction),
        ))
        if not 0.0 <= deltas[3] <= 1.0 or not 0.0 <= deltas[4] <= 1.0:
            raise ValueError("uncertainty delta fractions must lie in [0,1]")
        expected = _hash({
            "candidate_id": self.candidate_id,
            "observation_hash": self.observation_hash,
            "native_uncertainty_receipt_hash": self.native_uncertainty_receipt_hash,
            "child_uncertainty_receipt_hash": self.child_uncertainty_receipt_hash,
            "deltas": list(deltas),
            "delta_spatial_feature_hash": self.delta_spatial_feature_hash,
        })
        if self.evidence_hash and self.evidence_hash != expected:
            raise ValueError("post-action uncertainty evidence hash drift")
        object.__setattr__(self, "mean_delta", deltas[0])
        object.__setattr__(self, "p90_delta", deltas[1])
        object.__setattr__(self, "p99_delta", deltas[2])
        object.__setattr__(self, "resolved_fraction", deltas[3])
        object.__setattr__(self, "newly_uncertain_fraction", deltas[4])
        object.__setattr__(self, "evidence_hash", expected)


def postaction_uncertainty_delta_v1(
    *,
    candidate_id: str,
    observation_hash: str,
    native_uncertainty: UncertaintyReceiptV1,
    child_uncertainty: UncertaintyReceiptV1,
    mean_delta: float,
    p90_delta: float,
    p99_delta: float,
    resolved_fraction: float,
    newly_uncertain_fraction: float,
    delta_spatial_feature_hash: str,
) -> PostActionUncertaintyDeltaV1:
    if native_uncertainty.observation_role != "native":
        raise ValueError("post-action delta needs native baseline uncertainty")
    if child_uncertainty.observation_role != "executed_child":
        raise ValueError("post-action delta needs executed-child uncertainty")
    if native_uncertainty.task != child_uncertainty.task:
        raise ValueError("native and child uncertainty task drift")
    if (
        native_uncertainty.availability is not UncertaintyAvailabilityV1.AVAILABLE
        or child_uncertainty.availability is not UncertaintyAvailabilityV1.AVAILABLE
    ):
        raise ValueError("post-action uncertainty delta needs available maps")
    return PostActionUncertaintyDeltaV1(
        candidate_id=candidate_id,
        observation_hash=observation_hash,
        native_uncertainty_receipt_hash=native_uncertainty.receipt_hash,
        child_uncertainty_receipt_hash=child_uncertainty.receipt_hash,
        mean_delta=mean_delta,
        p90_delta=p90_delta,
        p99_delta=p99_delta,
        resolved_fraction=resolved_fraction,
        newly_uncertain_fraction=newly_uncertain_fraction,
        delta_spatial_feature_hash=delta_spatial_feature_hash,
    )


def uncertainty_assessor_provenance_hash_v1(
    base_assessor_provenance_hash: str,
    evidence: PostActionUncertaintyDeltaV1,
) -> str:
    return _hash({
        "base_assessor_provenance_hash": _text(
            base_assessor_provenance_hash, "base assessor provenance hash"
        ),
        "post_action_uncertainty_evidence_hash": evidence.evidence_hash,
    })


@dataclass(frozen=True)
class UncertaintyBoundRiskV1:
    risk: ActionRiskVectorV7
    evidence: PostActionUncertaintyDeltaV1
    base_assessor_provenance_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.risk, ActionRiskVectorV7):
            raise ValueError("uncertainty-bound risk must be typed")
        if not isinstance(self.evidence, PostActionUncertaintyDeltaV1):
            raise ValueError("post-action uncertainty evidence must be typed")
        if (
            self.risk.candidate_id != self.evidence.candidate_id
            or self.risk.observation_hash != self.evidence.observation_hash
        ):
            raise ValueError("post-action uncertainty risk binding drift")
        expected = uncertainty_assessor_provenance_hash_v1(
            self.base_assessor_provenance_hash, self.evidence,
        )
        if self.risk.assessor_provenance_hash != expected:
            raise ValueError("assessor model did not bind post-action uncertainty")


def select_uncertainty_aware_family_portfolio_v1(
    manifest: CandidateManifestV7,
    family_manifest: ActionFamilyManifestV1,
    family_proposal: FamilyAwareProposalV1,
    observations: Sequence[CandidateObservationV7],
    bound_risks: Sequence[UncertaintyBoundRiskV1],
    calibration: CalibrationReceiptV7,
    realizations: Sequence[ArmRealizationReceiptV7],
    aliases: Sequence[ArmAliasReceiptV7],
    *,
    audit_count: int = 0,
    child_count: int = 0,
    audit_receipts: Sequence[AuditEligibilityReceiptV7] = (),
) -> FamilyAwareDecisionV1:
    rows = tuple(bound_risks)
    if any(not isinstance(row, UncertaintyBoundRiskV1) for row in rows):
        raise ValueError("uncertainty-aware selection needs bound risks")
    return select_family_portfolio_v1(
        manifest,
        family_manifest,
        family_proposal,
        observations,
        tuple(row.risk for row in rows),
        calibration,
        realizations,
        aliases,
        audit_count=audit_count,
        child_count=child_count,
        audit_receipts=audit_receipts,
    )


__all__ = [
    "UNCERTAINTY_FAMILY_COMPETITION_SCHEMA_V1",
    "PostActionUncertaintyDeltaV1",
    "PreActionUncertaintyArtifactV1",
    "UncertaintyBoundPlannerScoreV1",
    "UncertaintyBoundRiskV1",
    "postaction_uncertainty_delta_v1",
    "preaction_uncertainty_artifact_v1",
    "propose_uncertainty_aware_family_competition_v1",
    "select_uncertainty_aware_family_portfolio_v1",
    "uncertainty_assessor_provenance_hash_v1",
]
