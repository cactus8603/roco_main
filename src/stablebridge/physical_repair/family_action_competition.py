"""Family-aware exact-control competition layered on frozen Selector-v7.

The action bank remains a complete bank of exact action/strength controls.
This module adds a frozen action-family registry, jointly chooses at most one
control from each family under the K1/K2 cost budget, and then delegates the
post-action safety decision to Selector-v7.  It does not change Selector-v7's
sealed implementation or its native fail-closed behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from itertools import combinations
import json
import math
from typing import Sequence

from .selector_v7 import (
    ActionRiskVectorV7,
    ArmAliasReceiptV7,
    ArmRealizationReceiptV7,
    ArmRealizationStatusV7,
    AuditEligibilityReceiptV7,
    CalibrationReceiptV7,
    CandidateArmV7,
    CandidateManifestV7,
    CandidateObservationV7,
    CostCountersV7,
    DecisionReasonV7,
    PlannerCandidateV7,
    PlannerProposalV7,
    PortfolioDecisionV7,
    PortfolioStateV7,
    propose_candidates_v7,
    select_portfolio_v7,
)


FAMILY_ACTION_COMPETITION_SCHEMA_V1 = "stablebridge-family-action-competition/v1"
FAMILY_ACTION_COMPETITION_ALGORITHM_V1 = (
    "joint-distinct-family-exact-control-competition-under-k1-k2-budget"
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty canonical text")
    return value


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FrozenActionFamilyDefinitionV1:
    """One registry row tying a family to exactly one action identity."""

    action_family_id: str
    member_action_identities: tuple[str, ...]
    registry_source_hash: str
    definition_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        _text(self.action_family_id, "action family id")
        _text(self.registry_source_hash, "family registry source hash")
        members = tuple(self.member_action_identities)
        if len(members) != 1:
            raise ValueError("one action family must bind exactly one action identity")
        _text(members[0], "family member action identity")
        if members[0] != self.action_family_id:
            raise ValueError("action family id must equal its frozen action identity")
        expected = _sha256({
            "action_family_id": self.action_family_id,
            "member_action_identities": list(members),
            "registry_source_hash": self.registry_source_hash,
        })
        if self.definition_hash and self.definition_hash != expected:
            raise ValueError("action-family definition hash drift")
        object.__setattr__(self, "member_action_identities", members)
        object.__setattr__(self, "definition_hash", expected)


@dataclass(frozen=True)
class ExactControlFamilyBindingV1:
    """Sealed mapping from one Selector-v7 exact arm to its family metadata."""

    candidate_id: str
    planned_arm_id: str
    action_identity: str
    exact_control_id: str
    action_family_id: str
    strength_id: str
    variant_id: str
    beta_id: str
    input_strength: float
    output_beta: float
    binding_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate id", self.candidate_id),
            ("planned arm id", self.planned_arm_id),
            ("action identity", self.action_identity),
            ("exact control id", self.exact_control_id),
            ("action family id", self.action_family_id),
            ("strength id", self.strength_id),
            ("variant id", self.variant_id),
            ("beta id", self.beta_id),
        ):
            _text(value, name)
        strength = _fraction(self.input_strength, "family input strength")
        beta = _fraction(self.output_beta, "family output beta")
        if strength <= 0.0 or beta <= 0.0:
            raise ValueError("family bindings are only for positive non-native controls")
        expected = _sha256({
            "candidate_id": self.candidate_id,
            "planned_arm_id": self.planned_arm_id,
            "action_identity": self.action_identity,
            "exact_control_id": self.exact_control_id,
            "action_family_id": self.action_family_id,
            "strength_id": self.strength_id,
            "variant_id": self.variant_id,
            "beta_id": self.beta_id,
            "input_strength": strength,
            "output_beta": beta,
        })
        if self.binding_hash and self.binding_hash != expected:
            raise ValueError("exact-control family binding hash drift")
        object.__setattr__(self, "input_strength", strength)
        object.__setattr__(self, "output_beta", beta)
        object.__setattr__(self, "binding_hash", expected)


@dataclass(frozen=True)
class ActionFamilyManifestV1:
    """Complete, frozen partition of one Selector-v7 non-native action bank."""

    selector_manifest_hash: str
    selector_candidate_family_hash: str
    frozen_before_outcome: bool
    definitions: tuple[FrozenActionFamilyDefinitionV1, ...]
    bindings: tuple[ExactControlFamilyBindingV1, ...]
    schema: str = FAMILY_ACTION_COMPETITION_SCHEMA_V1
    family_manifest_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        _text(self.selector_manifest_hash, "selector manifest hash")
        _text(self.selector_candidate_family_hash, "selector candidate-family hash")
        if self.schema != FAMILY_ACTION_COMPETITION_SCHEMA_V1:
            raise ValueError("family manifest schema drift")
        if self.frozen_before_outcome is not True:
            raise ValueError("family manifest must be frozen before outcomes")
        definitions = tuple(self.definitions)
        bindings = tuple(self.bindings)
        if not definitions or any(
            not isinstance(row, FrozenActionFamilyDefinitionV1)
            for row in definitions
        ):
            raise ValueError("family manifest needs typed registry definitions")
        if not bindings or any(
            not isinstance(row, ExactControlFamilyBindingV1) for row in bindings
        ):
            raise ValueError("family manifest needs typed non-native bindings")
        if len({row.action_family_id for row in definitions}) != len(definitions):
            raise ValueError("family registry ids must be unique")
        if len({row.registry_source_hash for row in definitions}) != 1:
            raise ValueError("all family definitions must share one frozen registry source")
        members = [member for row in definitions for member in row.member_action_identities]
        if len(set(members)) != len(members):
            raise ValueError("action identities cannot be split across families")
        if len({row.candidate_id for row in bindings}) != len(bindings):
            raise ValueError("family binding candidate ids must be unique")
        if len({row.planned_arm_id for row in bindings}) != len(bindings):
            raise ValueError("family binding planned-arm ids must be unique")
        family_ids = {row.action_family_id for row in definitions}
        if any(row.action_family_id not in family_ids for row in bindings):
            raise ValueError("family binding references an undefined family")
        payload = self._payload(definitions, bindings)
        expected = _sha256(payload)
        if self.family_manifest_hash and self.family_manifest_hash != expected:
            raise ValueError("action-family manifest hash drift")
        object.__setattr__(self, "definitions", definitions)
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "family_manifest_hash", expected)

    def _payload(
        self,
        definitions: Sequence[FrozenActionFamilyDefinitionV1] | None = None,
        bindings: Sequence[ExactControlFamilyBindingV1] | None = None,
    ) -> dict[str, object]:
        definition_rows = tuple(self.definitions if definitions is None else definitions)
        binding_rows = tuple(self.bindings if bindings is None else bindings)
        return {
            "schema": self.schema,
            "selector_manifest_hash": self.selector_manifest_hash,
            "selector_candidate_family_hash": self.selector_candidate_family_hash,
            "frozen_before_outcome": self.frozen_before_outcome,
            "definitions": [
                row.definition_hash
                for row in sorted(definition_rows, key=lambda item: item.action_family_id)
            ],
            "bindings": [
                row.binding_hash
                for row in sorted(binding_rows, key=lambda item: item.candidate_id)
            ],
        }

    def binding(self, candidate_id: str) -> ExactControlFamilyBindingV1 | None:
        return next((row for row in self.bindings if row.candidate_id == candidate_id), None)


@dataclass(frozen=True)
class FamilyStrengthWinnerV1:
    action_family_id: str
    candidate_id: str
    planned_arm_id: str
    strength_id: str
    variant_id: str
    beta_id: str
    planner_predicted_gain_raw_px: float
    prospective_runtime_cost: float
    conservative_objective: float

    def __post_init__(self) -> None:
        for name, value in (
            ("winner action family id", self.action_family_id),
            ("winner candidate id", self.candidate_id),
            ("winner planned arm id", self.planned_arm_id),
            ("winner strength id", self.strength_id),
            ("winner variant id", self.variant_id),
            ("winner beta id", self.beta_id),
        ):
            _text(value, name)
        gain = float(self.planner_predicted_gain_raw_px)
        cost = _nonnegative(self.prospective_runtime_cost, "winner runtime cost")
        objective = float(self.conservative_objective)
        if not math.isfinite(gain) or gain <= 0.0:
            raise ValueError("family winner must have positive finite predicted gain")
        if not math.isfinite(objective) or objective <= 0.0:
            raise ValueError("family winner must have positive conservative objective")
        object.__setattr__(self, "planner_predicted_gain_raw_px", gain)
        object.__setattr__(self, "prospective_runtime_cost", cost)
        object.__setattr__(self, "conservative_objective", objective)


@dataclass(frozen=True)
class FamilyAwareProposalV1:
    family_manifest_hash: str
    effective_planner_policy_hash: str
    input_score_family_hash: str
    family_winners: tuple[FamilyStrengthWinnerV1, ...]
    suppressed_candidate_ids: tuple[str, ...]
    proposal: PlannerProposalV7
    selection_receipt_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        for name, value in (
            ("family-aware manifest hash", self.family_manifest_hash),
            ("family-aware effective policy hash", self.effective_planner_policy_hash),
            ("family-aware input score hash", self.input_score_family_hash),
        ):
            _text(value, name)
        winners = tuple(self.family_winners)
        suppressed = tuple(self.suppressed_candidate_ids)
        if any(not isinstance(row, FamilyStrengthWinnerV1) for row in winners):
            raise ValueError("family-aware proposal winners must be typed")
        if len({row.action_family_id for row in winners}) != len(winners):
            raise ValueError("family-aware proposal contains duplicate families")
        if len(set(suppressed)) != len(suppressed):
            raise ValueError("suppressed candidate ids must be unique")
        if not isinstance(self.proposal, PlannerProposalV7):
            raise ValueError("family-aware result needs a Selector-v7 proposal")
        if self.proposal.planner_policy_hash != self.effective_planner_policy_hash:
            raise ValueError("inner proposal policy is not family-aware")
        proposed_ids = tuple(row.candidate_id for row in self.proposal.candidates)
        winner_ids = tuple(row.candidate_id for row in winners)
        if proposed_ids != winner_ids:
            raise ValueError("inner proposal must exactly equal family winners")
        if set(proposed_ids).intersection(suppressed):
            raise ValueError("a family winner cannot also be suppressed")
        payload = {
            "family_manifest_hash": self.family_manifest_hash,
            "effective_planner_policy_hash": self.effective_planner_policy_hash,
            "input_score_family_hash": self.input_score_family_hash,
            "family_winners": [
                {
                    "action_family_id": row.action_family_id,
                    "candidate_id": row.candidate_id,
                    "planned_arm_id": row.planned_arm_id,
                    "strength_id": row.strength_id,
                    "variant_id": row.variant_id,
                    "beta_id": row.beta_id,
                    "gain": row.planner_predicted_gain_raw_px,
                    "cost": row.prospective_runtime_cost,
                    "objective": row.conservative_objective,
                }
                for row in winners
            ],
            "suppressed_candidate_ids": list(suppressed),
            "proposal_hash": self.proposal.proposal_hash,
        }
        expected = _sha256(payload)
        if self.selection_receipt_hash and self.selection_receipt_hash != expected:
            raise ValueError("family selection receipt hash drift")
        object.__setattr__(self, "family_winners", winners)
        object.__setattr__(self, "suppressed_candidate_ids", suppressed)
        object.__setattr__(self, "selection_receipt_hash", expected)


class FamilyCommitGateV1(str, Enum):
    PASS = "pass"
    FAMILY_BINDING_INVALID = "family_binding_invalid"
    PROPOSAL_BINDING_INVALID = "proposal_binding_invalid"
    CALIBRATION_CHOICE_FAMILY_INCOMPLETE = "calibration_choice_family_incomplete"


@dataclass(frozen=True)
class FamilyAwareDecisionV1:
    gate: FamilyCommitGateV1
    required_calibration_planned_arm_ids: tuple[str, ...]
    decision: PortfolioDecisionV7

    def __post_init__(self) -> None:
        if not isinstance(self.gate, FamilyCommitGateV1):
            raise ValueError("family-aware decision needs a typed gate")
        required = tuple(self.required_calibration_planned_arm_ids)
        if len(set(required)) != len(required):
            raise ValueError("required calibration arm ids must be unique")
        if not isinstance(self.decision, PortfolioDecisionV7):
            raise ValueError("family-aware decision needs a Selector-v7 decision")
        object.__setattr__(self, "required_calibration_planned_arm_ids", required)


@dataclass(frozen=True)
class _EligibleExactControl:
    score: PlannerCandidateV7
    binding: ExactControlFamilyBindingV1
    cost: float
    counters: CostCountersV7
    objective: float


def _rehash_family_manifest(manifest: ActionFamilyManifestV1) -> bool:
    if _sha256(manifest._payload()) != manifest.family_manifest_hash:
        return False
    try:
        for row in manifest.definitions:
            FrozenActionFamilyDefinitionV1(
                action_family_id=row.action_family_id,
                member_action_identities=row.member_action_identities,
                registry_source_hash=row.registry_source_hash,
                definition_hash=row.definition_hash,
            )
        for row in manifest.bindings:
            ExactControlFamilyBindingV1(
                candidate_id=row.candidate_id,
                planned_arm_id=row.planned_arm_id,
                action_identity=row.action_identity,
                exact_control_id=row.exact_control_id,
                action_family_id=row.action_family_id,
                strength_id=row.strength_id,
                variant_id=row.variant_id,
                beta_id=row.beta_id,
                input_strength=row.input_strength,
                output_beta=row.output_beta,
                binding_hash=row.binding_hash,
            )
    except (TypeError, ValueError):
        return False
    return True


def validate_action_family_manifest_v1(
    selector_manifest: CandidateManifestV7,
    family_manifest: ActionFamilyManifestV1,
) -> None:
    """Prove complete exact-arm coverage and frozen action-family ownership."""

    if not isinstance(selector_manifest, CandidateManifestV7):
        raise ValueError("family validation needs a Selector-v7 manifest")
    if not isinstance(family_manifest, ActionFamilyManifestV1):
        raise ValueError("family validation needs a typed family manifest")
    if not _rehash_family_manifest(family_manifest):
        raise ValueError("family manifest failed sealed revalidation")
    if (
        family_manifest.selector_manifest_hash != selector_manifest.manifest_hash
        or family_manifest.selector_candidate_family_hash
        != selector_manifest.candidate_family_hash
        or family_manifest.frozen_before_outcome != selector_manifest.frozen_before_outcome
    ):
        raise ValueError("family manifest is bound to another Selector-v7 bank")
    nonnative = {
        row.candidate_id: row
        for row in selector_manifest.candidates
        if row.action_index != 0
    }
    bindings = {row.candidate_id: row for row in family_manifest.bindings}
    if set(bindings) != set(nonnative):
        raise ValueError("family manifest must cover every non-native exact arm")
    definitions = {row.action_family_id: row for row in family_manifest.definitions}
    action_identities = {row.action_identity for row in nonnative.values()}
    registry_identities = {
        member for row in family_manifest.definitions for member in row.member_action_identities
    }
    if action_identities != registry_identities:
        raise ValueError("family registry must exactly cover bank action identities")
    for candidate_id, arm in nonnative.items():
        binding = bindings[candidate_id]
        definition = definitions[binding.action_family_id]
        if (
            binding.planned_arm_id != arm.planned_arm_id
            or binding.action_identity != arm.action_identity
            or binding.exact_control_id != arm.exact_control_id
        ):
            raise ValueError("family binding exact-arm identity drifted")
        if binding.action_identity not in definition.member_action_identities:
            raise ValueError("family binding action identity has wrong owner")
        if not math.isclose(binding.input_strength, arm.input_strength, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("family binding input strength drifted")
        if not math.isclose(binding.output_beta, arm.output_beta, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("family binding output beta drifted")


def _prospective_group_cost(
    arm: CandidateArmV7,
    aliases: Sequence[ArmAliasReceiptV7],
) -> tuple[float, CostCountersV7]:
    scalar = float(arm.prospective_runtime_cost_ceiling)
    counters = arm.prospective_runtime_cost_counter_ceiling
    for alias in aliases:
        if alias.representative_planned_arm_id == arm.planned_arm_id:
            scalar += float(alias.prospective_runtime_verification_cost)
            counters = counters.plus(alias.prospective_runtime_verification_cost_counters)
    return scalar, counters


def _score_family_hash(scores: Sequence[PlannerCandidateV7]) -> str:
    return _sha256([
        {
            "candidate_id": row.candidate_id,
            "planned_arm_id": row.planned_arm_id,
            "cost_hash": row.cost_hash,
            "planner_allowlist_hash": row.planner_allowlist_hash,
            "before_feature_hash": row.before_feature_hash,
            "prediction_kind": row.prediction_kind.value,
            "gain": row.planner_predicted_gain_raw_px,
            "before_features_complete": row.before_features_complete,
        }
        for row in sorted(scores, key=lambda item: item.candidate_id)
    ])


def propose_family_competition_v1(
    selector_manifest: CandidateManifestV7,
    family_manifest: ActionFamilyManifestV1,
    scores: Sequence[PlannerCandidateV7],
    realizations: Sequence[ArmRealizationReceiptV7],
    aliases: Sequence[ArmAliasReceiptV7],
    *,
    top_k: int,
    maximum_prospective_runtime_cost: float,
    maximum_prospective_runtime_cost_counters: CostCountersV7,
    cost_penalty_raw_px_per_cost_unit: float,
    planner_policy_hash: str,
    planner_provenance_hash: str,
) -> FamilyAwareProposalV1:
    """Jointly select the best budget-feasible distinct-family portfolio."""

    validate_action_family_manifest_v1(selector_manifest, family_manifest)
    _text(planner_policy_hash, "base planner policy hash")
    _text(planner_provenance_hash, "planner provenance hash")
    if top_k not in {1, 2}:
        raise ValueError("family-aware Selector-v7 top K must be 1 or 2")
    ceiling = _nonnegative(maximum_prospective_runtime_cost, "runtime-cost ceiling")
    penalty = _nonnegative(cost_penalty_raw_px_per_cost_unit, "cost penalty")
    if not isinstance(maximum_prospective_runtime_cost_counters, CostCountersV7):
        raise ValueError("family-aware proposal needs typed cost counters")

    score_rows = tuple(scores)
    realization_rows = tuple(realizations)
    alias_rows = tuple(aliases)
    if any(not isinstance(row, PlannerCandidateV7) for row in score_rows):
        raise ValueError("family-aware planner scores must be typed")
    if len({row.candidate_id for row in score_rows}) != len(score_rows):
        raise ValueError("family-aware planner scores contain duplicate candidates")
    hard_legal_ids = {
        row.candidate_id
        for row in selector_manifest.candidates
        if row.action_index != 0 and row.hard_legal
    }
    if {row.candidate_id for row in score_rows} != hard_legal_ids:
        raise ValueError("family-aware planner needs exactly one score per hard-legal arm")
    arms = {row.candidate_id: row for row in selector_manifest.candidates}
    for score in score_rows:
        arm = arms[score.candidate_id]
        if (
            score.planned_arm_id != arm.planned_arm_id
            or score.cost_hash != arm.cost_hash
            or score.planner_allowlist_hash != selector_manifest.planner_allowlist_hash
        ):
            raise ValueError("family-aware planner score binding drifted")

    if len({row.planned_arm_id for row in realization_rows}) != len(realization_rows):
        raise ValueError("family-aware realizations contain duplicates")
    realization_by_arm = {row.planned_arm_id: row for row in realization_rows}
    alias_planned_ids = {row.alias_planned_arm_id for row in alias_rows}
    eligible: list[_EligibleExactControl] = []
    for score in score_rows:
        arm = arms[score.candidate_id]
        binding = family_manifest.binding(score.candidate_id)
        if binding is None:
            raise ValueError("family binding disappeared")
        receipt = realization_by_arm.get(arm.planned_arm_id)
        if (
            not score.before_features_complete
            or score.planner_predicted_gain_raw_px <= 0.0
            or receipt is None
            or receipt.status is not ArmRealizationStatusV7.COMPLETE
            or arm.planned_arm_id in alias_planned_ids
            or arm.composition_reason() is not None
        ):
            continue
        cost, counters = _prospective_group_cost(arm, alias_rows)
        objective = score.planner_predicted_gain_raw_px - penalty * cost
        if objective <= 0.0:
            continue
        eligible.append(_EligibleExactControl(score, binding, cost, counters, objective))

    feasible: list[tuple[tuple[float, float, float, tuple[str, ...]], tuple[_EligibleExactControl, ...]]] = []
    for size in range(1, min(top_k, len(eligible)) + 1):
        for rows in combinations(eligible, size):
            if len({row.binding.action_family_id for row in rows}) != size:
                continue
            total_cost = sum(row.cost for row in rows)
            counters = CostCountersV7.zero(
                cache_state=maximum_prospective_runtime_cost_counters.cache_state
            )
            try:
                for row in rows:
                    counters = counters.plus(row.counters)
            except ValueError:
                continue
            if total_cost > ceiling + 1e-12 or not counters.within(
                maximum_prospective_runtime_cost_counters
            ):
                continue
            ids = tuple(sorted(row.score.planned_arm_id for row in rows))
            rank = (
                -sum(row.objective for row in rows),
                -sum(row.score.planner_predicted_gain_raw_px for row in rows),
                total_cost,
                ids,
            )
            feasible.append((rank, rows))
    chosen = min(feasible, key=lambda item: item[0])[1] if feasible else ()
    chosen = tuple(sorted(chosen, key=lambda row: (
        -row.score.planner_predicted_gain_raw_px,
        row.cost,
        row.score.planned_arm_id,
    )))
    chosen_scores = tuple(row.score for row in chosen)
    selected_ids = {row.score.candidate_id for row in chosen}
    all_nonnative_ids = {
        row.candidate_id for row in selector_manifest.candidates if row.action_index != 0
    }
    suppressed = tuple(sorted(all_nonnative_ids - selected_ids))
    input_score_hash = _score_family_hash(score_rows)
    effective_policy_hash = _sha256({
        "algorithm": FAMILY_ACTION_COMPETITION_ALGORITHM_V1,
        "base_planner_policy_hash": planner_policy_hash,
        "family_manifest_hash": family_manifest.family_manifest_hash,
        "selector_candidate_family_hash": selector_manifest.candidate_family_hash,
        "input_score_family_hash": input_score_hash,
        "top_k": top_k,
        "maximum_prospective_runtime_cost": ceiling,
        "maximum_prospective_runtime_cost_counters_hash": (
            maximum_prospective_runtime_cost_counters.content_hash
        ),
        "cost_penalty_raw_px_per_cost_unit": penalty,
    })
    proposal = propose_candidates_v7(
        selector_manifest,
        chosen_scores,
        realization_rows,
        alias_rows,
        top_k=top_k,
        maximum_prospective_runtime_cost=ceiling,
        maximum_prospective_runtime_cost_counters=maximum_prospective_runtime_cost_counters,
        planner_policy_hash=effective_policy_hash,
        planner_provenance_hash=planner_provenance_hash,
    )
    if tuple(row.candidate_id for row in proposal.candidates) != tuple(
        row.candidate_id for row in chosen_scores
    ):
        raise RuntimeError("Selector-v7 changed the jointly selected family portfolio")
    winners = tuple(FamilyStrengthWinnerV1(
        action_family_id=row.binding.action_family_id,
        candidate_id=row.score.candidate_id,
        planned_arm_id=row.score.planned_arm_id,
        strength_id=row.binding.strength_id,
        variant_id=row.binding.variant_id,
        beta_id=row.binding.beta_id,
        planner_predicted_gain_raw_px=row.score.planner_predicted_gain_raw_px,
        prospective_runtime_cost=row.cost,
        conservative_objective=row.objective,
    ) for row in chosen)
    return FamilyAwareProposalV1(
        family_manifest_hash=family_manifest.family_manifest_hash,
        effective_planner_policy_hash=effective_policy_hash,
        input_score_family_hash=input_score_hash,
        family_winners=winners,
        suppressed_candidate_ids=suppressed,
        proposal=proposal,
    )


def _native_family_gate_decision(
    manifest: CandidateManifestV7,
    proposal: PlannerProposalV7,
    calibration: CalibrationReceiptV7,
    reason: DecisionReasonV7,
    audit_count: int,
    child_count: int,
) -> PortfolioDecisionV7:
    bounded_audit = audit_count if audit_count in {0, 1} else 1
    bounded_child = child_count if child_count in {0, 1} else 1
    return PortfolioDecisionV7(
        state=PortfolioStateV7.NATIVE,
        manifest_hash=manifest.manifest_hash,
        proposal_hash=proposal.proposal_hash,
        calibration_receipt_hash=calibration.receipt_hash,
        native_candidate_id=manifest.native.candidate_id,
        selected_candidate_id=manifest.native.candidate_id,
        audit_candidate_id=None,
        committed_unit_ids=(),
        safe_candidate_ids=(),
        conservative_utilities=(),
        exclusion_states=(),
        reasons=(reason,),
        audit_count=bounded_audit,
        child_count=bounded_child,
    )


def select_family_portfolio_v1(
    manifest: CandidateManifestV7,
    family_manifest: ActionFamilyManifestV1,
    family_proposal: FamilyAwareProposalV1,
    observations: Sequence[CandidateObservationV7],
    risks: Sequence[ActionRiskVectorV7],
    calibration: CalibrationReceiptV7,
    realizations: Sequence[ArmRealizationReceiptV7],
    aliases: Sequence[ArmAliasReceiptV7],
    *,
    audit_count: int = 0,
    child_count: int = 0,
    audit_receipts: Sequence[AuditEligibilityReceiptV7] = (),
) -> FamilyAwareDecisionV1:
    """Commit through v7 only when calibration covers the full choice family."""

    required = tuple(sorted(
        row.planned_arm_id
        for row in manifest.candidates
        if row.action_index != 0 and row.hard_legal
    ))
    try:
        validate_action_family_manifest_v1(manifest, family_manifest)
    except (TypeError, ValueError):
        decision = _native_family_gate_decision(
            manifest, family_proposal.proposal, calibration,
            DecisionReasonV7.MALFORMED_INPUT, audit_count, child_count,
        )
        return FamilyAwareDecisionV1(FamilyCommitGateV1.FAMILY_BINDING_INVALID, required, decision)
    try:
        if (
            family_proposal.family_manifest_hash != family_manifest.family_manifest_hash
            or family_proposal.proposal.planner_policy_hash
            != family_proposal.effective_planner_policy_hash
        ):
            raise ValueError("family proposal binding drift")
        PlannerProposalV7(**{
            name: getattr(family_proposal.proposal, name)
            for name in family_proposal.proposal.__dataclass_fields__
        })
        FamilyAwareProposalV1(**{
            name: getattr(family_proposal, name)
            for name in family_proposal.__dataclass_fields__
        })
    except (TypeError, ValueError):
        decision = _native_family_gate_decision(
            manifest, family_proposal.proposal, calibration,
            DecisionReasonV7.MALFORMED_INPUT, audit_count, child_count,
        )
        return FamilyAwareDecisionV1(FamilyCommitGateV1.PROPOSAL_BINDING_INVALID, required, decision)
    if not set(required).issubset(set(calibration.covered_planned_arm_ids)):
        decision = _native_family_gate_decision(
            manifest, family_proposal.proposal, calibration,
            DecisionReasonV7.CALIBRATION_SELECTION_FAMILY_INCOMPLETE,
            audit_count, child_count,
        )
        return FamilyAwareDecisionV1(
            FamilyCommitGateV1.CALIBRATION_CHOICE_FAMILY_INCOMPLETE,
            required,
            decision,
        )
    decision = select_portfolio_v7(
        manifest,
        family_proposal.proposal,
        tuple(observations),
        tuple(risks),
        calibration,
        tuple(realizations),
        tuple(aliases),
        audit_count=audit_count,
        child_count=child_count,
        audit_receipts=tuple(audit_receipts),
    )
    return FamilyAwareDecisionV1(FamilyCommitGateV1.PASS, required, decision)


__all__ = [
    "FAMILY_ACTION_COMPETITION_ALGORITHM_V1",
    "FAMILY_ACTION_COMPETITION_SCHEMA_V1",
    "ActionFamilyManifestV1",
    "ExactControlFamilyBindingV1",
    "FamilyAwareDecisionV1",
    "FamilyAwareProposalV1",
    "FamilyCommitGateV1",
    "FamilyStrengthWinnerV1",
    "FrozenActionFamilyDefinitionV1",
    "propose_family_competition_v1",
    "select_family_portfolio_v1",
    "validate_action_family_manifest_v1",
]
