"""Opt-in sequential orchestration over the frozen one-step family selector.

The compatibility path is intentionally boring: with the default policy this
module tail-calls :func:`select_family_portfolio_v1` and returns that exact
object.  Multi-action state and receipts are created only after an explicit
enable flag plus frozen retry/transition/calibration authorities are supplied.

The module is wiring, not deployment authority.  The repository's active
policy remains single-action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Sequence

from .family_action_competition import (
    ActionFamilyManifestV1,
    FamilyAwareDecisionV1,
    FamilyAwareProposalV1,
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
    PortfolioStateV7,
)


MULTI_ACTION_ORCHESTRATOR_SCHEMA_V1 = "stablebridge-multi-action-orchestrator/v1"


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


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


class MultiActionModeV1(str, Enum):
    ROOT_RETRY = "root_retry"
    COMPOSED_CHAIN = "composed_chain"


class MultiActionRunStateV1(str, Enum):
    READY = "ready"
    AUDIT_PENDING = "audit_pending"
    RETRY_ROOT = "retry_root"
    CONTINUE_CHAIN = "continue_chain"
    STOP_CURRENT = "stop_current"
    COMPLETE = "complete"


@dataclass(frozen=True)
class MultiActionPolicyV1:
    """Frozen horizon and authority.  Defaults preserve single-action behavior."""

    enabled: bool = False
    mode: MultiActionModeV1 = MultiActionModeV1.ROOT_RETRY
    maximum_steps: int = 1
    maximum_commits: int = 1
    maximum_root_retries: int = 0
    total_runtime_cost_ceiling: float = 0.0
    total_runtime_counter_ceiling: CostCountersV7 = field(
        default_factory=CostCountersV7.zero
    )
    retry_schedule_hash: str | None = None
    transition_registry_hash: str | None = None
    sequential_calibration_authority_hash: str | None = None
    policy_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, MultiActionModeV1):
            raise ValueError("multi-action mode must be typed")
        for name in ("maximum_steps", "maximum_commits", "maximum_root_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        ceiling = _nonnegative(
            self.total_runtime_cost_ceiling, "multi-action runtime cost ceiling"
        )
        if not isinstance(self.total_runtime_counter_ceiling, CostCountersV7):
            raise ValueError("multi-action policy needs typed cost counters")
        authorities = (
            self.retry_schedule_hash,
            self.transition_registry_hash,
            self.sequential_calibration_authority_hash,
        )
        if not self.enabled:
            if (
                self.maximum_steps != 1
                or self.maximum_commits != 1
                or self.maximum_root_retries != 0
                or ceiling != 0.0
                or any(value is not None for value in authorities)
            ):
                raise ValueError("disabled multi-action policy must use compatibility defaults")
        else:
            if self.maximum_steps < 2:
                raise ValueError("enabled multi-action policy needs at least two steps")
            if self.maximum_commits < 1 or ceiling <= 0.0:
                raise ValueError("enabled multi-action policy needs commit and cost budgets")
            _text(self.retry_schedule_hash, "retry schedule hash")
            _text(
                self.sequential_calibration_authority_hash,
                "sequential calibration authority hash",
            )
            if self.mode is MultiActionModeV1.ROOT_RETRY:
                if self.maximum_root_retries < 1:
                    raise ValueError("root retry mode needs a retry allowance")
            else:
                _text(self.transition_registry_hash, "transition registry hash")
        expected = _hash({
            "enabled": self.enabled,
            "mode": self.mode.value,
            "maximum_steps": self.maximum_steps,
            "maximum_commits": self.maximum_commits,
            "maximum_root_retries": self.maximum_root_retries,
            "total_runtime_cost_ceiling": ceiling,
            "total_runtime_counter_ceiling_hash": (
                self.total_runtime_counter_ceiling.content_hash
            ),
            "retry_schedule_hash": self.retry_schedule_hash,
            "transition_registry_hash": self.transition_registry_hash,
            "sequential_calibration_authority_hash": (
                self.sequential_calibration_authority_hash
            ),
        })
        if self.policy_hash and self.policy_hash != expected:
            raise ValueError("multi-action policy hash drift")
        object.__setattr__(self, "total_runtime_cost_ceiling", ceiling)
        object.__setattr__(self, "policy_hash", expected)


@dataclass(frozen=True)
class SequentialCostEntryV1:
    ordinal: int
    kind: str
    scalar_cost: float
    counters: CostCountersV7
    execution_receipt_hash: str
    entry_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or self.ordinal < 0:
            raise ValueError("cost entry ordinal must be nonnegative")
        if self.kind not in {
            "probe", "audit", "delivery", "rollback", "failed_execution",
        }:
            raise ValueError("unknown sequential cost kind")
        cost = _nonnegative(self.scalar_cost, "sequential scalar cost")
        if not isinstance(self.counters, CostCountersV7):
            raise ValueError("sequential cost entry needs typed counters")
        _text(self.execution_receipt_hash, "execution receipt hash")
        expected = _hash({
            "ordinal": self.ordinal,
            "kind": self.kind,
            "scalar_cost": cost,
            "counters_hash": self.counters.content_hash,
            "execution_receipt_hash": self.execution_receipt_hash,
        })
        if self.entry_hash and self.entry_hash != expected:
            raise ValueError("sequential cost entry hash drift")
        object.__setattr__(self, "scalar_cost", cost)
        object.__setattr__(self, "entry_hash", expected)


@dataclass(frozen=True)
class SequentialCostLedgerV1:
    entries: tuple[SequentialCostEntryV1, ...]
    cache_state: str = "cold"
    ledger_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(not isinstance(row, SequentialCostEntryV1) for row in entries):
            raise ValueError("sequential ledger entries must be typed")
        if tuple(row.ordinal for row in entries) != tuple(range(len(entries))):
            raise ValueError("sequential ledger ordinals must be contiguous")
        if len({row.execution_receipt_hash for row in entries}) != len(entries):
            raise ValueError("sequential execution receipt replay detected")
        if any(row.counters.cache_state != self.cache_state for row in entries):
            raise ValueError("sequential ledger cache state drift")
        expected = _hash({
            "cache_state": self.cache_state,
            "entry_hashes": [row.entry_hash for row in entries],
        })
        if self.ledger_hash and self.ledger_hash != expected:
            raise ValueError("sequential cost ledger hash drift")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "ledger_hash", expected)

    @property
    def scalar_cost(self) -> float:
        return float(sum(row.scalar_cost for row in self.entries))

    @property
    def counters(self) -> CostCountersV7:
        value = CostCountersV7.zero(cache_state=self.cache_state)
        for row in self.entries:
            value = value.plus(row.counters)
        return value

    def append(self, entry: SequentialCostEntryV1) -> "SequentialCostLedgerV1":
        if entry.ordinal != len(self.entries):
            raise ValueError("appended sequential cost has the wrong ordinal")
        return SequentialCostLedgerV1((*self.entries, entry), self.cache_state)


@dataclass(frozen=True)
class TemporalCompositionReceiptV1:
    parent_checkpoint_hash: str
    ordered_prefix_planned_arm_ids: tuple[str, ...]
    next_planned_arm_id: str
    transition_registry_hash: str
    sequential_calibration_authority_hash: str
    interaction_harm_upper: float
    eligible: bool
    receipt_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        for name, value in (
            ("parent checkpoint hash", self.parent_checkpoint_hash),
            ("next planned arm id", self.next_planned_arm_id),
            ("transition registry hash", self.transition_registry_hash),
            ("sequential calibration authority hash", self.sequential_calibration_authority_hash),
        ):
            _text(value, name)
        prefix = tuple(self.ordered_prefix_planned_arm_ids)
        if not prefix or any(not isinstance(row, str) or not row for row in prefix):
            raise ValueError("temporal composition needs a nonempty ordered prefix")
        harm = _nonnegative(self.interaction_harm_upper, "interaction harm upper")
        expected = _hash({
            "parent_checkpoint_hash": self.parent_checkpoint_hash,
            "ordered_prefix_planned_arm_ids": list(prefix),
            "next_planned_arm_id": self.next_planned_arm_id,
            "transition_registry_hash": self.transition_registry_hash,
            "sequential_calibration_authority_hash": self.sequential_calibration_authority_hash,
            "interaction_harm_upper": harm,
            "eligible": self.eligible,
        })
        if self.receipt_hash and self.receipt_hash != expected:
            raise ValueError("temporal composition receipt hash drift")
        object.__setattr__(self, "ordered_prefix_planned_arm_ids", prefix)
        object.__setattr__(self, "interaction_harm_upper", harm)
        object.__setattr__(self, "receipt_hash", expected)


@dataclass(frozen=True)
class CumulativeRiskReceiptV1:
    root_source_hash: str
    checkpoint_hash: str
    ordered_planned_arm_ids: tuple[str, ...]
    net_gain_lower: float
    harm_upper: float
    severe_probability_upper: float
    harmed_fraction_upper: float
    cvar95_upper: float
    passed: bool
    calibration_authority_hash: str
    receipt_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        for name, value in (
            ("root source hash", self.root_source_hash),
            ("checkpoint hash", self.checkpoint_hash),
            ("cumulative calibration authority hash", self.calibration_authority_hash),
        ):
            _text(value, name)
        arms = tuple(self.ordered_planned_arm_ids)
        if not arms or any(not isinstance(row, str) or not row for row in arms):
            raise ValueError("cumulative risk needs an ordered action prefix")
        gain = _nonnegative(self.net_gain_lower, "cumulative net gain lower")
        values = tuple(_nonnegative(value, name) for name, value in (
            ("cumulative harm upper", self.harm_upper),
            ("cumulative severe upper", self.severe_probability_upper),
            ("cumulative harmed fraction upper", self.harmed_fraction_upper),
            ("cumulative CVaR95 upper", self.cvar95_upper),
        ))
        if values[1] > 1.0 or values[2] > 1.0:
            raise ValueError("cumulative probabilities must lie in [0,1]")
        expected = _hash({
            "root_source_hash": self.root_source_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "ordered_planned_arm_ids": list(arms),
            "net_gain_lower": gain,
            "risk_uppers": list(values),
            "passed": self.passed,
            "calibration_authority_hash": self.calibration_authority_hash,
        })
        if self.receipt_hash and self.receipt_hash != expected:
            raise ValueError("cumulative risk receipt hash drift")
        object.__setattr__(self, "ordered_planned_arm_ids", arms)
        object.__setattr__(self, "net_gain_lower", gain)
        object.__setattr__(self, "harm_upper", values[0])
        object.__setattr__(self, "severe_probability_upper", values[1])
        object.__setattr__(self, "harmed_fraction_upper", values[2])
        object.__setattr__(self, "cvar95_upper", values[3])
        object.__setattr__(self, "receipt_hash", expected)


@dataclass(frozen=True)
class MultiActionStateV1:
    policy_hash: str
    root_source_hash: str
    current_source_hash: str
    current_checkpoint_hash: str
    run_state: MultiActionRunStateV1
    step_count: int
    commit_count: int
    retry_count: int
    attempted_planned_arm_ids: tuple[str, ...]
    committed_planned_arm_ids: tuple[str, ...]
    cost_ledger: SequentialCostLedgerV1
    state_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        for name, value in (
            ("multi-action policy hash", self.policy_hash),
            ("root source hash", self.root_source_hash),
            ("current source hash", self.current_source_hash),
            ("current checkpoint hash", self.current_checkpoint_hash),
        ):
            _text(value, name)
        if not isinstance(self.run_state, MultiActionRunStateV1):
            raise ValueError("multi-action run state must be typed")
        for name in ("step_count", "commit_count", "retry_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        attempted = tuple(self.attempted_planned_arm_ids)
        committed = tuple(self.committed_planned_arm_ids)
        if len(set(attempted)) != len(attempted):
            raise ValueError("multi-action candidate replay detected")
        if any(row not in attempted for row in committed):
            raise ValueError("committed action was never attempted")
        if self.commit_count != len(committed):
            raise ValueError("multi-action commit count drift")
        expected = _hash({
            "policy_hash": self.policy_hash,
            "root_source_hash": self.root_source_hash,
            "current_source_hash": self.current_source_hash,
            "current_checkpoint_hash": self.current_checkpoint_hash,
            "run_state": self.run_state.value,
            "step_count": self.step_count,
            "commit_count": self.commit_count,
            "retry_count": self.retry_count,
            "attempted_planned_arm_ids": list(attempted),
            "committed_planned_arm_ids": list(committed),
            "cost_ledger_hash": self.cost_ledger.ledger_hash,
        })
        if self.state_hash and self.state_hash != expected:
            raise ValueError("multi-action state hash drift")
        object.__setattr__(self, "attempted_planned_arm_ids", attempted)
        object.__setattr__(self, "committed_planned_arm_ids", committed)
        object.__setattr__(self, "state_hash", expected)


def initial_multi_action_state_v1(
    policy: MultiActionPolicyV1,
    *,
    root_source_hash: str,
    root_checkpoint_hash: str,
) -> MultiActionStateV1:
    if not policy.enabled:
        raise ValueError("disabled compatibility policy creates no multi-action state")
    return MultiActionStateV1(
        policy_hash=policy.policy_hash,
        root_source_hash=root_source_hash,
        current_source_hash=root_source_hash,
        current_checkpoint_hash=root_checkpoint_hash,
        run_state=MultiActionRunStateV1.READY,
        step_count=0,
        commit_count=0,
        retry_count=0,
        attempted_planned_arm_ids=(),
        committed_planned_arm_ids=(),
        cost_ledger=SequentialCostLedgerV1(
            (), policy.total_runtime_counter_ceiling.cache_state
        ),
    )


def select_with_multi_action_v1(
    manifest: CandidateManifestV7,
    family_manifest: ActionFamilyManifestV1,
    family_proposal: FamilyAwareProposalV1,
    observations: Sequence[CandidateObservationV7],
    risks: Sequence[ActionRiskVectorV7],
    calibration: CalibrationReceiptV7,
    realizations: Sequence[ArmRealizationReceiptV7],
    aliases: Sequence[ArmAliasReceiptV7],
    *,
    policy: MultiActionPolicyV1 = MultiActionPolicyV1(),
    audit_count: int = 0,
    child_count: int = 0,
    audit_receipts: Sequence[AuditEligibilityReceiptV7] = (),
) -> FamilyAwareDecisionV1:
    """Exact compatibility entrypoint; enabled callers use ``advance`` below."""

    if policy.enabled:
        raise ValueError("enabled multi-action execution requires explicit state and receipts")
    return select_family_portfolio_v1(
        manifest, family_manifest, family_proposal, observations, risks,
        calibration, realizations, aliases, audit_count=audit_count,
        child_count=child_count, audit_receipts=audit_receipts,
    )


@dataclass(frozen=True)
class MultiActionStepResultV1:
    inner_decision: FamilyAwareDecisionV1
    state: MultiActionStateV1


def advance_multi_action_v1(
    state: MultiActionStateV1,
    policy: MultiActionPolicyV1,
    manifest: CandidateManifestV7,
    family_manifest: ActionFamilyManifestV1,
    family_proposal: FamilyAwareProposalV1,
    observations: Sequence[CandidateObservationV7],
    risks: Sequence[ActionRiskVectorV7],
    calibration: CalibrationReceiptV7,
    realizations: Sequence[ArmRealizationReceiptV7],
    aliases: Sequence[ArmAliasReceiptV7],
    *,
    step_cost: SequentialCostEntryV1,
    committed_checkpoint_hash: str | None = None,
    cumulative_risk: CumulativeRiskReceiptV1 | None = None,
    temporal_composition: TemporalCompositionReceiptV1 | None = None,
    audit_receipts: Sequence[AuditEligibilityReceiptV7] = (),
) -> MultiActionStepResultV1:
    """Run one immutable v7 decision and advance the opt-in outer state."""

    if not policy.enabled or state.policy_hash != policy.policy_hash:
        raise ValueError("multi-action policy is disabled or mismatched")
    if state.run_state not in {
        MultiActionRunStateV1.READY,
        MultiActionRunStateV1.RETRY_ROOT,
        MultiActionRunStateV1.CONTINUE_CHAIN,
    }:
        raise ValueError("multi-action state is not ready for a new step")
    if state.step_count >= policy.maximum_steps:
        raise ValueError("multi-action step budget exhausted")
    expected_source = (
        state.root_source_hash
        if policy.mode is MultiActionModeV1.ROOT_RETRY
        else state.current_source_hash
    )
    if manifest.source_hash != expected_source:
        raise ValueError("multi-action manifest is bound to the wrong checkpoint")
    proposal_ids = tuple(family_proposal.proposal.proposed_candidate_ids)
    planned_by_candidate = {
        row.candidate_id: row.planned_arm_id for row in manifest.candidates
    }
    attempted = tuple(planned_by_candidate[row] for row in proposal_ids)
    if set(attempted) & set(state.attempted_planned_arm_ids):
        raise ValueError("multi-action planned arm replay detected")
    ledger = state.cost_ledger.append(step_cost)
    if (
        ledger.scalar_cost > policy.total_runtime_cost_ceiling + 1e-12
        or not ledger.counters.within(policy.total_runtime_counter_ceiling)
    ):
        raise ValueError("multi-action cumulative cost budget exceeded")
    decision = select_family_portfolio_v1(
        manifest, family_manifest, family_proposal, observations, risks,
        calibration, realizations, aliases, audit_count=0, child_count=0,
        audit_receipts=audit_receipts,
    )
    inner = decision.decision
    all_attempted = (*state.attempted_planned_arm_ids, *attempted)
    committed = state.committed_planned_arm_ids
    current_source = state.current_source_hash
    current_checkpoint = state.current_checkpoint_hash
    commits = state.commit_count
    retries = state.retry_count

    if inner.state is PortfolioStateV7.AUDIT:
        run_state = MultiActionRunStateV1.AUDIT_PENDING
    elif inner.state is PortfolioStateV7.NATIVE:
        if (
            policy.mode is MultiActionModeV1.ROOT_RETRY
            and retries < policy.maximum_root_retries
            and state.step_count + 1 < policy.maximum_steps
        ):
            retries += 1
            run_state = MultiActionRunStateV1.RETRY_ROOT
        else:
            run_state = MultiActionRunStateV1.STOP_CURRENT
    else:
        selected = inner.selected_candidate_id
        planned = planned_by_candidate.get(selected or "")
        if planned is None:
            raise ValueError("inner commit selected an unknown arm")
        _text(committed_checkpoint_hash, "committed checkpoint hash")
        if cumulative_risk is None or not cumulative_risk.passed:
            raise ValueError("multi-action commit needs passing cumulative risk")
        expected_prefix = (*committed, planned)
        if (
            cumulative_risk.root_source_hash != state.root_source_hash
            or cumulative_risk.checkpoint_hash != committed_checkpoint_hash
            or cumulative_risk.ordered_planned_arm_ids != expected_prefix
            or cumulative_risk.calibration_authority_hash
            != policy.sequential_calibration_authority_hash
        ):
            raise ValueError("cumulative risk receipt binding drift")
        if policy.mode is MultiActionModeV1.COMPOSED_CHAIN and committed:
            if temporal_composition is None or not temporal_composition.eligible:
                raise ValueError("composed chain needs an eligible transition receipt")
            if (
                temporal_composition.parent_checkpoint_hash != state.current_checkpoint_hash
                or temporal_composition.ordered_prefix_planned_arm_ids != committed
                or temporal_composition.next_planned_arm_id != planned
                or temporal_composition.transition_registry_hash
                != policy.transition_registry_hash
                or temporal_composition.sequential_calibration_authority_hash
                != policy.sequential_calibration_authority_hash
            ):
                raise ValueError("temporal composition receipt binding drift")
        committed = expected_prefix
        commits += 1
        current_source = committed_checkpoint_hash  # next manifest must bind this
        current_checkpoint = committed_checkpoint_hash
        can_continue = (
            policy.mode is MultiActionModeV1.COMPOSED_CHAIN
            and commits < policy.maximum_commits
            and state.step_count + 1 < policy.maximum_steps
        )
        run_state = (
            MultiActionRunStateV1.CONTINUE_CHAIN
            if can_continue else MultiActionRunStateV1.COMPLETE
        )

    next_state = MultiActionStateV1(
        policy_hash=policy.policy_hash,
        root_source_hash=state.root_source_hash,
        current_source_hash=current_source,
        current_checkpoint_hash=current_checkpoint,
        run_state=run_state,
        step_count=state.step_count + 1,
        commit_count=commits,
        retry_count=retries,
        attempted_planned_arm_ids=all_attempted,
        committed_planned_arm_ids=committed,
        cost_ledger=ledger,
    )
    return MultiActionStepResultV1(decision, next_state)


__all__ = [
    "MULTI_ACTION_ORCHESTRATOR_SCHEMA_V1",
    "CumulativeRiskReceiptV1",
    "MultiActionModeV1",
    "MultiActionPolicyV1",
    "MultiActionRunStateV1",
    "MultiActionStateV1",
    "MultiActionStepResultV1",
    "SequentialCostEntryV1",
    "SequentialCostLedgerV1",
    "TemporalCompositionReceiptV1",
    "advance_multi_action_v1",
    "initial_multi_action_state_v1",
    "select_with_multi_action_v1",
]
