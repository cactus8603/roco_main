"""Certificate-gated, bounded action planning with exact rollback.

This module is deliberately a small-horizon controller, not a generic agent.
It plans at most three predeclared exact controls for one region and commits at
most one repair.  Every candidate is executed speculatively from an immutable
checkpoint, restored exactly after observation, and admitted only through the
same typed physical, support, EPE and tail contracts used by Selector v6.

The planner never consumes a corruption label, task truth or an outcome.  A
high-level proposal can therefore be learned or heuristic, while the commit
authority remains an independently recomputed proof-carrying decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .selector_v6 import (
    ActionHypothesisV6,
    CertifiedControlEnumerationV6,
    CertifiedControlOptionV6,
    SelectorV6Config,
    enumerate_certified_control_options_v6,
)
from .selector_v6_evidence import TypedPhysicalLawReceipt


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
PLAN_STATES = frozenset({"running", "stop_native", "stop_prefix"})
STEP_KINDS = frozenset({"repair_candidate", "stop_native", "stop_prefix"})
EVENT_KINDS = frozenset({"initialized", "dispatch", "rollback", "replan", "commit", "stop"})
FAILURE_REASONS = frozenset({
    "nonfinite_candidate",
    "coordinate_mismatch",
    "postcheck_failed",
    "certificate_invalidated",
    "support_empty",
    "execution_failed",
})


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def typed_law_receipt_set_sha256(
    receipts: Sequence[TypedPhysicalLawReceipt],
) -> str:
    """Hash a complete typed-law set independently of serialization order."""

    rows = sorted(
        (asdict(receipt) for receipt in receipts),
        key=lambda row: (row["role"], row["law_id"]),
    )
    return _canonical_sha256(rows)


@dataclass(frozen=True)
class PlanCheckpointV1:
    checkpoint_id: str
    root_source_input_hash: str
    current_input_hash: str
    native_output_hash: str
    current_output_hash: str
    delivery_commitment_sha256: str
    output_trust: float
    delivery_support_hash: str | None
    parent_checkpoint_id: str | None
    accepted_path_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        identities = (
            self.checkpoint_id, self.root_source_input_hash,
            self.current_input_hash, self.native_output_hash,
            self.current_output_hash, self.delivery_commitment_sha256,
        )
        if any(not item for item in identities):
            raise ValueError("checkpoint identities are required")
        if len(set(self.accepted_path_keys)) != len(self.accepted_path_keys):
            raise ValueError("checkpoint paths must be unique")
        expected = _checkpoint_id(
            root_source_input_hash=self.root_source_input_hash,
            current_input_hash=self.current_input_hash,
            native_output_hash=self.native_output_hash,
            current_output_hash=self.current_output_hash,
            delivery_commitment_sha256=self.delivery_commitment_sha256,
            output_trust=self.output_trust,
            delivery_support_hash=self.delivery_support_hash,
            parent_checkpoint_id=self.parent_checkpoint_id,
            accepted_path_keys=self.accepted_path_keys,
        )
        if self.checkpoint_id != expected:
            raise ValueError("checkpoint content hash drift")
        if self.parent_checkpoint_id is None:
            if (self.accepted_path_keys
                    or self.current_input_hash != self.root_source_input_hash
                    or self.current_output_hash != self.native_output_hash
                    or self.output_trust != 0.0
                    or self.delivery_support_hash is not None):
                raise ValueError("native checkpoint is not immutable root state")
        else:
            if (len(self.accepted_path_keys) != 1
                    or not 0.0 < self.output_trust <= 1.0
                    or not self.delivery_support_hash):
                raise ValueError("ActionPlan-v1 commit lacks exact delivery binding")


def _checkpoint_id(*, root_source_input_hash: str, current_input_hash: str,
                   native_output_hash: str, current_output_hash: str,
                   delivery_commitment_sha256: str, output_trust: float,
                   delivery_support_hash: str | None,
                   parent_checkpoint_id: str | None,
                   accepted_path_keys: tuple[str, ...]) -> str:
    return _canonical_sha256({
        "schema": "certificate-action-checkpoint/v1",
        "root_source_input_hash": root_source_input_hash,
        "current_input_hash": current_input_hash,
        "native_output_hash": native_output_hash,
        "current_output_hash": current_output_hash,
        "delivery_commitment_sha256": delivery_commitment_sha256,
        "output_trust": output_trust,
        "delivery_support_hash": delivery_support_hash,
        "parent_checkpoint_id": parent_checkpoint_id,
        "accepted_path_keys": list(accepted_path_keys),
    })


def make_native_checkpoint_v1(
    *, source_input_hash: str, native_output_hash: str,
) -> PlanCheckpointV1:
    if not source_input_hash or not native_output_hash:
        raise ValueError("native source and output hashes are required")
    delivery_commitment = _canonical_sha256({
        "schema": "bounded-delivery-commitment/v1",
        "native_output_hash": native_output_hash,
        "candidate_output_hash": native_output_hash,
        "delivery_support_hash": None,
        "output_trust": 0.0,
        "fallback": "bit_exact_native",
    })
    checkpoint_id = _checkpoint_id(
        root_source_input_hash=source_input_hash,
        current_input_hash=source_input_hash,
        native_output_hash=native_output_hash,
        current_output_hash=native_output_hash,
        delivery_commitment_sha256=delivery_commitment,
        output_trust=0.0,
        delivery_support_hash=None,
        parent_checkpoint_id=None,
        accepted_path_keys=(),
    )
    return PlanCheckpointV1(
        checkpoint_id=checkpoint_id,
        root_source_input_hash=source_input_hash,
        current_input_hash=source_input_hash,
        native_output_hash=native_output_hash,
        current_output_hash=native_output_hash,
        delivery_commitment_sha256=delivery_commitment,
        output_trust=0.0,
        delivery_support_hash=None,
        parent_checkpoint_id=None,
        accepted_path_keys=(),
    )


@dataclass(frozen=True)
class ActionPlanCandidateV1:
    """A predeclared exact action/control to execute from one checkpoint."""

    hypothesis_key: str
    control_key: str
    action_identity: str
    region_key: str
    source_checkpoint_id: str
    source_input_hash: str
    admission_support_hash: str
    latent_set_hash: str
    law_receipt_set_sha256: str
    support_policy_hash: str
    input_path_id: str
    control_parameter_hash: str
    expected_information_gain_lower: float
    maximum_compute_seconds: float
    maximum_matcher_trajectories: int
    probe_rank: int
    mandatory: bool

    def __post_init__(self) -> None:
        identities = (
            self.hypothesis_key, self.control_key, self.action_identity,
            self.region_key, self.source_checkpoint_id, self.source_input_hash,
            self.admission_support_hash, self.latent_set_hash,
            self.law_receipt_set_sha256, self.support_policy_hash,
            self.input_path_id, self.control_parameter_hash,
        )
        if any(not item for item in identities):
            raise ValueError("planned control identity is incomplete")
        if not self.hypothesis_key.startswith(f"{self.action_identity}#"):
            raise ValueError("planned action and hypothesis identities differ")
        if SHA_RE.fullmatch(self.law_receipt_set_sha256) is None:
            raise ValueError("typed-law set needs a canonical SHA-256")
        _nonnegative(
            self.expected_information_gain_lower,
            "expected information gain lower",
        )
        _nonnegative(self.maximum_compute_seconds, "maximum candidate compute")
        if (isinstance(self.maximum_matcher_trajectories, bool)
                or self.maximum_matcher_trajectories < 0):
            raise ValueError("maximum matcher trajectories must be nonnegative")
        if isinstance(self.probe_rank, bool) or self.probe_rank < 0:
            raise ValueError("probe rank must be nonnegative")
        if not isinstance(self.mandatory, bool):
            raise ValueError("mandatory must be Boolean")

    @property
    def path_key(self) -> str:
        return f"{self.hypothesis_key}/{self.control_key}"


@dataclass(frozen=True)
class ActionPlanConfigV1:
    selector: SelectorV6Config = field(default_factory=SelectorV6Config)
    maximum_compute_seconds: float = 30.0
    maximum_matcher_trajectories: int = 3
    cost_penalty_px_per_second: float = 0.0
    minimum_selection_gap_px: float = 0.0
    maximum_predeclared_candidates: int = 3
    maximum_committed_actions: int = 1

    def __post_init__(self) -> None:
        _nonnegative(self.maximum_compute_seconds, "plan compute budget")
        _nonnegative(self.cost_penalty_px_per_second, "plan cost penalty")
        _nonnegative(self.minimum_selection_gap_px, "selection gap")
        if (isinstance(self.maximum_matcher_trajectories, bool)
                or not 1 <= self.maximum_matcher_trajectories <= 3):
            raise ValueError("ActionPlan-v1 trajectory budget must lie in [1,3]")
        if not 1 <= self.maximum_predeclared_candidates <= 3:
            raise ValueError("ActionPlan-v1 is fixed to at most K=2+1 controls")
        if self.maximum_committed_actions != 1:
            raise ValueError("ActionPlan-v1 commits at most one action")


@dataclass(frozen=True)
class ActionCandidateResultV1:
    proposal_path_key: str
    source_checkpoint_id: str
    hypothesis: ActionHypothesisV6

    def __post_init__(self) -> None:
        if not self.proposal_path_key or not self.source_checkpoint_id:
            raise ValueError("candidate result provenance is required")
        if len(self.hypothesis.controls) != 1:
            raise ValueError("one speculative result must contain one exact control")

    @property
    def control(self):
        return self.hypothesis.controls[0]


@dataclass(frozen=True)
class ActionCandidateFailureV1:
    proposal_path_key: str
    source_checkpoint_id: str
    reason: str
    spent_compute_seconds: float
    spent_matcher_trajectories: int

    def __post_init__(self) -> None:
        if self.reason not in FAILURE_REASONS:
            raise ValueError("candidate failure reason is not predeclared")
        _nonnegative(self.spent_compute_seconds, "failed candidate compute")
        if (isinstance(self.spent_matcher_trajectories, bool)
                or self.spent_matcher_trajectories < 0):
            raise ValueError("failed candidate trajectories must be nonnegative")


@dataclass(frozen=True)
class ActionPlanEventV1:
    ordinal: int
    kind: str
    from_checkpoint_id: str
    to_checkpoint_id: str
    proposal_path_key: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.ordinal < 0 or self.kind not in EVENT_KINDS:
            raise ValueError("invalid plan event")
        if not self.from_checkpoint_id or not self.to_checkpoint_id or not self.reason:
            raise ValueError("plan event provenance is incomplete")
        if self.kind in {"dispatch", "rollback", "commit"} and not self.proposal_path_key:
            raise ValueError("candidate event needs an exact path")


@dataclass(frozen=True)
class ActionPlanStepV1:
    kind: str
    plan_id: str
    source_checkpoint_id: str
    proposal_path_key: str | None
    action_identity: str | None
    region_key: str | None
    expected_information_gain_lower: float
    maximum_compute_seconds: float
    maximum_matcher_trajectories: int
    required_law_receipt_set_sha256: str | None
    stop_reason: str

    def __post_init__(self) -> None:
        if self.kind not in STEP_KINDS or not self.plan_id or not self.stop_reason:
            raise ValueError("invalid action-plan step")
        if self.kind == "repair_candidate":
            required = (
                self.proposal_path_key, self.action_identity, self.region_key,
                self.required_law_receipt_set_sha256,
            )
            if any(not value for value in required):
                raise ValueError("repair step lacks proof obligations")
        elif any(value is not None for value in (
            self.proposal_path_key, self.action_identity, self.region_key,
            self.required_law_receipt_set_sha256,
        )):
            raise ValueError("terminal step cannot name a candidate")


@dataclass(frozen=True)
class ActionPlanStateV1:
    plan_id: str
    native_checkpoint: PlanCheckpointV1
    current_checkpoint: PlanCheckpointV1
    candidates: tuple[ActionPlanCandidateV1, ...]
    config: ActionPlanConfigV1
    results: tuple[ActionCandidateResultV1, ...]
    failures: tuple[ActionCandidateFailureV1, ...]
    history: tuple[ActionPlanEventV1, ...]
    inflight_path_key: str | None
    state: str
    selected_path_key: str | None
    terminal_reason: str | None

    @property
    def attempted_path_keys(self) -> frozenset[str]:
        return frozenset(
            [item.proposal_path_key for item in self.results]
            + [item.proposal_path_key for item in self.failures]
        )

    @property
    def spent_compute_seconds(self) -> float:
        return float(
            sum(item.control.expected_compute_seconds for item in self.results)
            + sum(item.spent_compute_seconds for item in self.failures)
        )

    @property
    def spent_matcher_trajectories(self) -> int:
        return int(
            sum(item.control.extra_matcher_trajectories for item in self.results)
            + sum(item.spent_matcher_trajectories for item in self.failures)
        )


def _plan_id(root: PlanCheckpointV1,
             candidates: Sequence[ActionPlanCandidateV1],
             config: ActionPlanConfigV1) -> str:
    return _canonical_sha256({
        "schema": "certificate-action-plan/v1",
        "native_checkpoint_id": root.checkpoint_id,
        "candidates": [asdict(item) for item in candidates],
        "config": asdict(config),
    })


def initialize_action_plan_v1(
    native_checkpoint: PlanCheckpointV1,
    candidates: Sequence[ActionPlanCandidateV1],
    *,
    config: ActionPlanConfigV1 = ActionPlanConfigV1(),
) -> ActionPlanStateV1:
    rows = tuple(sorted(
        candidates,
        key=lambda row: (
            row.probe_rank, -row.expected_information_gain_lower, row.path_key,
        ),
    ))
    if not rows or len(rows) > config.maximum_predeclared_candidates:
        raise ValueError("plan needs one to three predeclared candidates")
    if len({row.path_key for row in rows}) != len(rows):
        raise ValueError("predeclared candidate paths must be unique")
    if not any(row.mandatory for row in rows):
        raise ValueError("plan needs at least one mandatory candidate")
    if len({row.region_key for row in rows}) != 1:
        raise ValueError("ActionPlan-v1 resolves one region at a time")
    for row in rows:
        if (row.source_checkpoint_id != native_checkpoint.checkpoint_id
                or row.source_input_hash
                != native_checkpoint.root_source_input_hash):
            raise ValueError("candidate does not fork from immutable native checkpoint")
    identifier = _plan_id(native_checkpoint, rows, config)
    initial = ActionPlanEventV1(
        ordinal=0,
        kind="initialized",
        from_checkpoint_id=native_checkpoint.checkpoint_id,
        to_checkpoint_id=native_checkpoint.checkpoint_id,
        proposal_path_key=None,
        reason="predeclared_choice_family_frozen",
    )
    state = ActionPlanStateV1(
        plan_id=identifier,
        native_checkpoint=native_checkpoint,
        current_checkpoint=native_checkpoint,
        candidates=rows,
        config=config,
        results=(),
        failures=(),
        history=(initial,),
        inflight_path_key=None,
        state="running",
        selected_path_key=None,
        terminal_reason=None,
    )
    validate_action_plan_state_v1(state)
    return state


def _candidate_map(state: ActionPlanStateV1) -> dict[str, ActionPlanCandidateV1]:
    return {row.path_key: row for row in state.candidates}


def _validate_result(state: ActionPlanStateV1,
                     result: ActionCandidateResultV1) -> None:
    candidate = _candidate_map(state).get(result.proposal_path_key)
    if candidate is None:
        raise ValueError("candidate result was not predeclared")
    row = result.hypothesis
    control = result.control
    endpoint = row.action.hypothesized_degraded_endpoint
    action_identity = f"{row.action.operator_id}@{endpoint}"
    checks = (
        result.source_checkpoint_id == candidate.source_checkpoint_id,
        row.key == candidate.hypothesis_key,
        control.key == candidate.control_key,
        action_identity == candidate.action_identity,
        row.region_key == candidate.region_key,
        row.source_input_hash == candidate.source_input_hash,
        row.admission_support_hash == candidate.admission_support_hash,
        row.latent_set_hash == candidate.latent_set_hash,
        typed_law_receipt_set_sha256(row.law_receipts)
        == candidate.law_receipt_set_sha256,
        control.support.support_policy_hash == candidate.support_policy_hash,
        control.input_path_id == candidate.input_path_id,
        control.control_parameter_hash == candidate.control_parameter_hash,
        control.expected_compute_seconds <= candidate.maximum_compute_seconds,
        control.extra_matcher_trajectories
        <= candidate.maximum_matcher_trajectories,
        control.native_output_hash == state.native_checkpoint.native_output_hash,
    )
    if not all(checks):
        raise ValueError("speculative result does not satisfy its frozen proposal")


def validate_action_plan_state_v1(state: ActionPlanStateV1) -> None:
    if state.state not in PLAN_STATES or not state.plan_id:
        raise ValueError("invalid action-plan state")
    if state.plan_id != _plan_id(
        state.native_checkpoint, state.candidates, state.config,
    ):
        raise ValueError("action-plan identity drift")
    candidates = _candidate_map(state)
    if len(candidates) != len(state.candidates):
        raise ValueError("action-plan candidate duplication")
    result_keys = tuple(item.proposal_path_key for item in state.results)
    failure_keys = tuple(item.proposal_path_key for item in state.failures)
    if (len(result_keys) != len(set(result_keys))
            or len(failure_keys) != len(set(failure_keys))
            or set(result_keys) & set(failure_keys)
            or not (set(result_keys) | set(failure_keys)).issubset(candidates)):
        raise ValueError("attempted candidate ledger drift")
    for result in state.results:
        _validate_result(state, result)
    for failure in state.failures:
        candidate = candidates[failure.proposal_path_key]
        if (failure.source_checkpoint_id != candidate.source_checkpoint_id
                or failure.spent_compute_seconds > candidate.maximum_compute_seconds
                or failure.spent_matcher_trajectories
                > candidate.maximum_matcher_trajectories):
            raise ValueError("failed-candidate budget/provenance drift")
    if state.spent_compute_seconds > state.config.maximum_compute_seconds:
        raise ValueError("action plan exceeded its compute budget")
    if state.spent_matcher_trajectories > state.config.maximum_matcher_trajectories:
        raise ValueError("action plan exceeded its trajectory budget")
    if state.inflight_path_key is not None:
        if (state.state != "running"
                or state.inflight_path_key not in candidates
                or state.inflight_path_key in state.attempted_path_keys):
            raise ValueError("inflight candidate state drift")
    if tuple(event.ordinal for event in state.history) != tuple(range(len(state.history))):
        raise ValueError("action-plan history ordinal drift")
    if state.state == "running":
        if (state.current_checkpoint != state.native_checkpoint
                or state.selected_path_key is not None
                or state.terminal_reason is not None):
            raise ValueError("running plan must remain on immutable checkpoint")
    elif state.state == "stop_native":
        if (state.current_checkpoint != state.native_checkpoint
                or state.selected_path_key is not None
                or not state.terminal_reason):
            raise ValueError("native terminal state drift")
    else:
        if (state.current_checkpoint.parent_checkpoint_id
                != state.native_checkpoint.checkpoint_id
                or state.current_checkpoint.accepted_path_keys
                != (state.selected_path_key,)
                or not state.terminal_reason):
            raise ValueError("committed safe-prefix state drift")
        selected, reason = _best_separated_option(state)
        if (selected is None
                or selected.option.path_key != state.selected_path_key
                or reason != state.terminal_reason
                or state.current_checkpoint != _checkpoint_for_option(state, selected)):
            raise ValueError("committed checkpoint is not verifier-derived")


def _append_event(state: ActionPlanStateV1, *, kind: str,
                  from_checkpoint_id: str, to_checkpoint_id: str,
                  proposal_path_key: str | None, reason: str,
                  ) -> tuple[ActionPlanEventV1, ...]:
    event = ActionPlanEventV1(
        ordinal=len(state.history),
        kind=kind,
        from_checkpoint_id=from_checkpoint_id,
        to_checkpoint_id=to_checkpoint_id,
        proposal_path_key=proposal_path_key,
        reason=reason,
    )
    return state.history + (event,)


def record_action_candidate_v1(
    state: ActionPlanStateV1,
    result: ActionCandidateResultV1,
) -> ActionPlanStateV1:
    validate_action_plan_state_v1(state)
    if state.inflight_path_key != result.proposal_path_key:
        raise ValueError("result does not match the dispatched candidate")
    _validate_result(state, result)
    updated = replace(
        state,
        results=state.results + (result,),
        history=_append_event(
            state,
            kind="rollback",
            from_checkpoint_id=state.native_checkpoint.checkpoint_id,
            to_checkpoint_id=state.native_checkpoint.checkpoint_id,
            proposal_path_key=result.proposal_path_key,
            reason="speculative_candidate_observed_then_exact_checkpoint_restored",
        ),
        inflight_path_key=None,
    )
    validate_action_plan_state_v1(updated)
    return updated


def record_action_failure_v1(
    state: ActionPlanStateV1,
    failure: ActionCandidateFailureV1,
) -> ActionPlanStateV1:
    validate_action_plan_state_v1(state)
    if state.inflight_path_key != failure.proposal_path_key:
        raise ValueError("failure does not match the dispatched candidate")
    candidate = _candidate_map(state)[failure.proposal_path_key]
    if (failure.source_checkpoint_id != candidate.source_checkpoint_id
            or failure.spent_compute_seconds > candidate.maximum_compute_seconds
            or failure.spent_matcher_trajectories
            > candidate.maximum_matcher_trajectories):
        raise ValueError("failed candidate exceeded its frozen proposal")
    updated = replace(
        state,
        failures=state.failures + (failure,),
        history=_append_event(
            state,
            kind="rollback",
            from_checkpoint_id=state.native_checkpoint.checkpoint_id,
            to_checkpoint_id=state.native_checkpoint.checkpoint_id,
            proposal_path_key=failure.proposal_path_key,
            reason=failure.reason,
        ),
        inflight_path_key=None,
    )
    validate_action_plan_state_v1(updated)
    return updated


def _merge_results(
    results: Sequence[ActionCandidateResultV1],
) -> tuple[ActionHypothesisV6, ...]:
    grouped: dict[str, list[ActionHypothesisV6]] = {}
    for result in results:
        grouped.setdefault(result.hypothesis.key, []).append(result.hypothesis)
    merged: list[ActionHypothesisV6] = []
    for key, rows in sorted(grouped.items()):
        first = rows[0]
        controls = []
        for row in rows:
            same = (
                row.action == first.action
                and row.region_key == first.region_key
                and row.source_input_hash == first.source_input_hash
                and row.admission_support_hash == first.admission_support_hash
                and row.latent_set_hash == first.latent_set_hash
                and row.latent_parameter_keys == first.latent_parameter_keys
                and row.physical_status == first.physical_status
                and row.law_receipts == first.law_receipts
            )
            if not same:
                raise ValueError("results cannot be merged across physical hypotheses")
            controls.extend(row.controls)
        if len({control.key for control in controls}) != len(controls):
            raise ValueError("duplicate exact control result")
        merged.append(replace(
            first, controls=tuple(sorted(controls, key=lambda item: item.key)),
        ))
    return tuple(merged)


@dataclass(frozen=True)
class _PlanOption:
    option: CertifiedControlOptionV6
    objective_lower: float
    objective_upper: float


def _enumerate_plan_options(
    state: ActionPlanStateV1,
) -> tuple[CertifiedControlEnumerationV6, tuple[_PlanOption, ...]]:
    declared = frozenset(row.path_key for row in state.candidates)
    merged = _merge_results(state.results)
    enumeration = enumerate_certified_control_options_v6(
        merged,
        config=state.config.selector,
        predeclared_path_keys=declared,
    )
    options = tuple(
        _PlanOption(
            option=item,
            objective_lower=(
                item.utility_lower_px * item.delivery_fraction
                - state.config.cost_penalty_px_per_second * item.cost_seconds
            ),
            objective_upper=(
                item.utility_upper_px * item.delivery_fraction
                - state.config.cost_penalty_px_per_second * item.cost_seconds
            ),
        )
        for item in enumeration.options
    )
    return enumeration, options


def _best_separated_option(
    state: ActionPlanStateV1,
) -> tuple[_PlanOption | None, str]:
    if not state.results:
        return None, "no_verified_candidate"
    _, options = _enumerate_plan_options(state)
    if not options:
        return None, "no_candidate_passed_complete_certificate_stack"
    best_by_path: dict[str, _PlanOption] = {}
    for item in options:
        path = item.option.path_key
        incumbent = best_by_path.get(path)
        key = (item.objective_lower, item.option.utility_lower_px, item.option.beta)
        if incumbent is None or key > (
            incumbent.objective_lower,
            incumbent.option.utility_lower_px,
            incumbent.option.beta,
        ):
            best_by_path[path] = item
    ordered = sorted(
        best_by_path.values(),
        key=lambda item: (
            item.objective_lower,
            item.option.utility_lower_px,
            item.option.beta,
            tuple(reversed(item.option.path_key)),
        ),
        reverse=True,
    )
    best = ordered[0]
    if best.objective_lower <= 0.0:
        return None, "no_positive_cost_adjusted_lower_bound"
    rivals = [item for item in ordered[1:] if item.option.path_key != best.option.path_key]
    if rivals and best.objective_lower <= (
        max(item.objective_upper for item in rivals)
        + state.config.minimum_selection_gap_px
    ):
        return None, "safe_action_intervals_overlap"
    return best, "unique_safe_cost_adjusted_action"


def _remaining_budget_fits(
    state: ActionPlanStateV1, candidate: ActionPlanCandidateV1,
) -> bool:
    return bool(
        state.spent_compute_seconds + candidate.maximum_compute_seconds
        <= state.config.maximum_compute_seconds
        and state.spent_matcher_trajectories
        + candidate.maximum_matcher_trajectories
        <= state.config.maximum_matcher_trajectories
    )


def _dispatch(
    state: ActionPlanStateV1,
    candidate: ActionPlanCandidateV1,
    *, reason: str, replan: bool,
) -> tuple[ActionPlanStateV1, ActionPlanStepV1]:
    history = state.history
    if replan:
        history = history + (ActionPlanEventV1(
            ordinal=len(history),
            kind="replan",
            from_checkpoint_id=state.native_checkpoint.checkpoint_id,
            to_checkpoint_id=state.native_checkpoint.checkpoint_id,
            proposal_path_key=None,
            reason=reason,
        ),)
    interim = replace(state, history=history)
    history = _append_event(
        interim,
        kind="dispatch",
        from_checkpoint_id=state.native_checkpoint.checkpoint_id,
        to_checkpoint_id=state.native_checkpoint.checkpoint_id,
        proposal_path_key=candidate.path_key,
        reason=reason,
    )
    updated = replace(
        interim,
        history=history,
        inflight_path_key=candidate.path_key,
    )
    step = ActionPlanStepV1(
        kind="repair_candidate",
        plan_id=state.plan_id,
        source_checkpoint_id=candidate.source_checkpoint_id,
        proposal_path_key=candidate.path_key,
        action_identity=candidate.action_identity,
        region_key=candidate.region_key,
        expected_information_gain_lower=candidate.expected_information_gain_lower,
        maximum_compute_seconds=candidate.maximum_compute_seconds,
        maximum_matcher_trajectories=candidate.maximum_matcher_trajectories,
        required_law_receipt_set_sha256=candidate.law_receipt_set_sha256,
        stop_reason=reason,
    )
    validate_action_plan_state_v1(updated)
    return updated, step


def _stop_native(
    state: ActionPlanStateV1, reason: str,
) -> tuple[ActionPlanStateV1, ActionPlanStepV1]:
    updated = replace(
        state,
        state="stop_native",
        terminal_reason=reason,
        history=_append_event(
            state,
            kind="stop",
            from_checkpoint_id=state.native_checkpoint.checkpoint_id,
            to_checkpoint_id=state.native_checkpoint.checkpoint_id,
            proposal_path_key=None,
            reason=reason,
        ),
    )
    step = ActionPlanStepV1(
        kind="stop_native",
        plan_id=state.plan_id,
        source_checkpoint_id=state.native_checkpoint.checkpoint_id,
        proposal_path_key=None,
        action_identity=None,
        region_key=None,
        expected_information_gain_lower=0.0,
        maximum_compute_seconds=0.0,
        maximum_matcher_trajectories=0,
        required_law_receipt_set_sha256=None,
        stop_reason=reason,
    )
    validate_action_plan_state_v1(updated)
    return updated, step


def _checkpoint_for_option(
    state: ActionPlanStateV1, selected: _PlanOption,
) -> PlanCheckpointV1:
    item = selected.option
    delivery_commitment = _canonical_sha256({
        "schema": "bounded-delivery-commitment/v1",
        "native_output_hash": state.native_checkpoint.native_output_hash,
        "candidate_output_hash": item.control.candidate_output_hash,
        "delivery_support_hash": item.control.support.delivery_support_hash,
        "output_trust": item.beta,
        "metric_id": item.control.task_bound.metric_id,
        "control_selection_receipt_hash": (
            item.control.task_bound.control_selection_receipt_hash
        ),
        "tail_calibration_version": item.control.tail_risk.calibration_version,
        "tail_calibration_data_hash": item.control.tail_risk.calibration_data_hash,
        "fallback": "bit_exact_native_outside_delivery_support",
    })
    checkpoint_id = _checkpoint_id(
        root_source_input_hash=state.native_checkpoint.root_source_input_hash,
        current_input_hash=item.control.repaired_input_hash,
        native_output_hash=state.native_checkpoint.native_output_hash,
        current_output_hash=item.control.candidate_output_hash,
        delivery_commitment_sha256=delivery_commitment,
        output_trust=item.beta,
        delivery_support_hash=item.control.support.delivery_support_hash,
        parent_checkpoint_id=state.native_checkpoint.checkpoint_id,
        accepted_path_keys=(item.path_key,),
    )
    return PlanCheckpointV1(
        checkpoint_id=checkpoint_id,
        root_source_input_hash=state.native_checkpoint.root_source_input_hash,
        current_input_hash=item.control.repaired_input_hash,
        native_output_hash=state.native_checkpoint.native_output_hash,
        current_output_hash=item.control.candidate_output_hash,
        delivery_commitment_sha256=delivery_commitment,
        output_trust=item.beta,
        delivery_support_hash=item.control.support.delivery_support_hash,
        parent_checkpoint_id=state.native_checkpoint.checkpoint_id,
        accepted_path_keys=(item.path_key,),
    )


def _commit(
    state: ActionPlanStateV1, selected: _PlanOption, reason: str,
) -> tuple[ActionPlanStateV1, ActionPlanStepV1]:
    item = selected.option
    checkpoint = _checkpoint_for_option(state, selected)
    updated = replace(
        state,
        current_checkpoint=checkpoint,
        state="stop_prefix",
        selected_path_key=item.path_key,
        terminal_reason=reason,
        history=_append_event(
            state,
            kind="commit",
            from_checkpoint_id=state.native_checkpoint.checkpoint_id,
            to_checkpoint_id=checkpoint.checkpoint_id,
            proposal_path_key=item.path_key,
            reason=reason,
        ),
    )
    step = ActionPlanStepV1(
        kind="stop_prefix",
        plan_id=state.plan_id,
        source_checkpoint_id=checkpoint.checkpoint_id,
        proposal_path_key=None,
        action_identity=None,
        region_key=None,
        expected_information_gain_lower=0.0,
        maximum_compute_seconds=0.0,
        maximum_matcher_trajectories=0,
        required_law_receipt_set_sha256=None,
        stop_reason=reason,
    )
    validate_action_plan_state_v1(updated)
    return updated, step


def advance_action_plan_v1(
    state: ActionPlanStateV1,
) -> tuple[ActionPlanStateV1, ActionPlanStepV1]:
    """Dispatch one speculative action or terminate with safe prefix/native."""

    validate_action_plan_state_v1(state)
    if state.inflight_path_key is not None:
        raise ValueError("resolve the inflight candidate before advancing")
    if state.state != "running":
        kind = "stop_prefix" if state.state == "stop_prefix" else "stop_native"
        return state, ActionPlanStepV1(
            kind=kind,
            plan_id=state.plan_id,
            source_checkpoint_id=state.current_checkpoint.checkpoint_id,
            proposal_path_key=None,
            action_identity=None,
            region_key=None,
            expected_information_gain_lower=0.0,
            maximum_compute_seconds=0.0,
            maximum_matcher_trajectories=0,
            required_law_receipt_set_sha256=None,
            stop_reason=state.terminal_reason or state.state,
        )

    attempted = state.attempted_path_keys
    mandatory = [
        row for row in state.candidates
        if row.mandatory and row.path_key not in attempted
    ]
    affordable = [row for row in mandatory if _remaining_budget_fits(state, row)]
    if mandatory:
        if not affordable:
            return _stop_native(state, "mandatory_probe_exceeds_remaining_budget")
        return _dispatch(
            state, affordable[0],
            reason="execute_predeclared_mandatory_probe",
            replan=False,
        )

    selected, selection_reason = _best_separated_option(state)
    if selected is not None:
        return _commit(state, selected, selection_reason)

    optional = [
        row for row in state.candidates
        if not row.mandatory and row.path_key not in attempted
        and _remaining_budget_fits(state, row)
    ]
    if optional:
        return _dispatch(
            state, optional[0],
            reason=f"optional_probe_for:{selection_reason}",
            replan=True,
        )
    terminal_reason = (
        "optional_probe_exceeds_remaining_budget"
        if any(not row.mandatory and row.path_key not in attempted
               for row in state.candidates)
        else selection_reason
    )
    return _stop_native(state, terminal_reason)


def action_plan_receipt_v1(state: ActionPlanStateV1) -> dict[str, Any]:
    """Return a compact, outcome-blind replay receipt for one terminal plan."""

    validate_action_plan_state_v1(state)
    if state.state == "running" or state.inflight_path_key is not None:
        raise ValueError("only a terminal action plan has a final receipt")
    events = [asdict(event) for event in state.history]
    return {
        "schema": "certificate-action-plan-receipt/v1",
        "status": state.state,
        "plan_id": state.plan_id,
        "native_checkpoint_id": state.native_checkpoint.checkpoint_id,
        "final_checkpoint_id": state.current_checkpoint.checkpoint_id,
        "final_delivery_commitment_sha256": (
            state.current_checkpoint.delivery_commitment_sha256
        ),
        "final_output_trust": state.current_checkpoint.output_trust,
        "final_delivery_support_hash": state.current_checkpoint.delivery_support_hash,
        "selected_path_key": state.selected_path_key,
        "terminal_reason": state.terminal_reason,
        "predeclared_path_keys": sorted(row.path_key for row in state.candidates),
        "attempted_path_keys": sorted(state.attempted_path_keys),
        "spent_compute_seconds": state.spent_compute_seconds,
        "spent_matcher_trajectories": state.spent_matcher_trajectories,
        "history_sha256": _canonical_sha256(events),
        "rollback_count": sum(event.kind == "rollback" for event in state.history),
        "replan_count": sum(event.kind == "replan" for event in state.history),
        "commit_count": sum(event.kind == "commit" for event in state.history),
        "corruption_label_used": False,
        "GT_read": False,
        "H2_read": False,
        "outcome_read": False,
    }
