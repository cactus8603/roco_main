from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import stablebridge.physical_repair.multi_action_orchestrator as module
from stablebridge.physical_repair.multi_action_orchestrator import (
    CumulativeRiskReceiptV1,
    MultiActionModeV1,
    MultiActionPolicyV1,
    MultiActionRunStateV1,
    SequentialCostEntryV1,
    SequentialCostLedgerV1,
    TemporalCompositionReceiptV1,
    advance_multi_action_v1,
    initial_multi_action_state_v1,
    select_with_multi_action_v1,
)
from stablebridge.physical_repair.selector_v7 import CostCountersV7, PortfolioStateV7


def counters(candidate_forwards: int = 0, wall: float = 0.0) -> CostCountersV7:
    return CostCountersV7(
        natural_forwards=0,
        reused_forwards=0,
        candidate_forwards=candidate_forwards,
        reverse_forwards=0,
        cpu_seconds=wall / 2,
        gpu_seconds=wall / 2,
        wall_seconds=wall,
        bytes_moved=100 * candidate_forwards,
        peak_allocated_bytes=10 * candidate_forwards,
        peak_reserved_bytes=20 * candidate_forwards,
        matcher_trajectories=candidate_forwards,
        cache_state="cold",
    )


def test_disabled_policy_is_strict_compatibility_default():
    policy = MultiActionPolicyV1()
    assert policy.enabled is False
    with pytest.raises(ValueError, match="compatibility defaults"):
        replace(policy, maximum_steps=2)
    with pytest.raises(ValueError, match="creates no"):
        initial_multi_action_state_v1(
            policy, root_source_hash="root", root_checkpoint_hash="checkpoint",
        )


def test_disabled_entrypoint_tail_calls_existing_selector(monkeypatch):
    sentinel = object()
    observed = {}

    def fake(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(module, "select_family_portfolio_v1", fake)
    result = select_with_multi_action_v1(
        "manifest", "families", "proposal", ("observation",), ("risk",),
        "calibration", ("realization",), ("alias",), audit_count=1,
        child_count=1, audit_receipts=("audit",),
    )
    assert result is sentinel
    assert observed["args"][:3] == ("manifest", "families", "proposal")
    assert observed["kwargs"] == {
        "audit_count": 1,
        "child_count": 1,
        "audit_receipts": ("audit",),
    }


def test_enabled_policy_requires_frozen_authorities_and_builds_state():
    ceiling = counters(3, 3.0)
    policy = MultiActionPolicyV1(
        enabled=True,
        mode=MultiActionModeV1.ROOT_RETRY,
        maximum_steps=2,
        maximum_commits=1,
        maximum_root_retries=1,
        total_runtime_cost_ceiling=3.0,
        total_runtime_counter_ceiling=ceiling,
        retry_schedule_hash="retry-schedule",
        sequential_calibration_authority_hash="sequence-calibration",
    )
    state = initial_multi_action_state_v1(
        policy, root_source_hash="root", root_checkpoint_hash="root-checkpoint",
    )
    assert state.step_count == 0
    assert state.cost_ledger.scalar_cost == 0
    assert state.current_source_hash == "root"


def test_composed_chain_requires_transition_registry():
    with pytest.raises(ValueError, match="transition registry"):
        MultiActionPolicyV1(
            enabled=True,
            mode=MultiActionModeV1.COMPOSED_CHAIN,
            maximum_steps=2,
            maximum_commits=2,
            total_runtime_cost_ceiling=2.0,
            total_runtime_counter_ceiling=counters(2, 2.0),
            retry_schedule_hash="frozen-candidate-schedule",
            sequential_calibration_authority_hash="sequence-calibration",
        )


def test_ledger_is_append_only_sums_cost_and_rejects_replay():
    first = SequentialCostEntryV1(0, "probe", 1.0, counters(1, 1.0), "exec-1")
    second = SequentialCostEntryV1(
        1, "failed_execution", 0.5, counters(1, 0.5), "exec-2",
    )
    ledger = SequentialCostLedgerV1(()).append(first).append(second)
    assert ledger.scalar_cost == pytest.approx(1.5)
    assert ledger.counters.candidate_forwards == 2
    assert ledger.counters.peak_allocated_bytes == 10
    replay = SequentialCostEntryV1(
        1, "failed_execution", 0.5, counters(1, 0.5), "exec-1",
    )
    with pytest.raises(ValueError, match="replay"):
        SequentialCostLedgerV1((first, replay))


def fake_step_inputs(source: str, candidate: str, planned_arm: str):
    manifest = SimpleNamespace(
        source_hash=source,
        candidates=(
            SimpleNamespace(candidate_id=candidate, planned_arm_id=planned_arm),
        ),
    )
    proposal = SimpleNamespace(
        proposal=SimpleNamespace(proposed_candidate_ids=(candidate,)),
    )
    return manifest, proposal


def install_fake_decision(monkeypatch, state: PortfolioStateV7, candidate=None):
    result = SimpleNamespace(
        decision=SimpleNamespace(state=state, selected_candidate_id=candidate),
    )
    monkeypatch.setattr(module, "select_family_portfolio_v1", lambda *a, **k: result)
    return result


def test_root_retry_retains_failed_probe_cost(monkeypatch):
    policy = MultiActionPolicyV1(
        enabled=True,
        mode=MultiActionModeV1.ROOT_RETRY,
        maximum_steps=2,
        maximum_commits=1,
        maximum_root_retries=1,
        total_runtime_cost_ceiling=2.0,
        total_runtime_counter_ceiling=counters(2, 2.0),
        retry_schedule_hash="frozen-root-retries",
        sequential_calibration_authority_hash="sequence-calibration",
    )
    state = initial_multi_action_state_v1(
        policy, root_source_hash="root", root_checkpoint_hash="root-checkpoint",
    )
    manifest, proposal = fake_step_inputs("root", "candidate-a", "planned-a")
    install_fake_decision(monkeypatch, PortfolioStateV7.NATIVE)
    result = advance_multi_action_v1(
        state, policy, manifest, "families", proposal, (), (), "calibration",
        (), (), step_cost=SequentialCostEntryV1(
            0, "failed_execution", 0.5, counters(1, 0.5), "exec-a",
        ),
    )
    assert result.state.run_state is MultiActionRunStateV1.RETRY_ROOT
    assert result.state.retry_count == 1
    assert result.state.cost_ledger.scalar_cost == pytest.approx(0.5)
    assert result.state.current_checkpoint_hash == "root-checkpoint"


def test_composed_chain_requires_temporal_receipt_after_first_commit(monkeypatch):
    policy = MultiActionPolicyV1(
        enabled=True,
        mode=MultiActionModeV1.COMPOSED_CHAIN,
        maximum_steps=2,
        maximum_commits=2,
        maximum_root_retries=0,
        total_runtime_cost_ceiling=2.0,
        total_runtime_counter_ceiling=counters(2, 2.0),
        retry_schedule_hash="frozen-chain-candidates",
        transition_registry_hash="transition-registry",
        sequential_calibration_authority_hash="sequence-calibration",
    )
    state = initial_multi_action_state_v1(
        policy, root_source_hash="root", root_checkpoint_hash="root-checkpoint",
    )
    manifest_a, proposal_a = fake_step_inputs("root", "candidate-a", "planned-a")
    install_fake_decision(monkeypatch, PortfolioStateV7.COMMIT, "candidate-a")
    first_risk = CumulativeRiskReceiptV1(
        root_source_hash="root",
        checkpoint_hash="checkpoint-a",
        ordered_planned_arm_ids=("planned-a",),
        net_gain_lower=0.1,
        harm_upper=0.01,
        severe_probability_upper=0.001,
        harmed_fraction_upper=0.01,
        cvar95_upper=0.02,
        passed=True,
        calibration_authority_hash="sequence-calibration",
    )
    first = advance_multi_action_v1(
        state, policy, manifest_a, "families", proposal_a, (), (), "calibration",
        (), (), step_cost=SequentialCostEntryV1(
            0, "probe", 0.5, counters(1, 0.5), "exec-a",
        ), committed_checkpoint_hash="checkpoint-a", cumulative_risk=first_risk,
    )
    assert first.state.run_state is MultiActionRunStateV1.CONTINUE_CHAIN
    assert first.state.current_source_hash == "checkpoint-a"

    manifest_b, proposal_b = fake_step_inputs(
        "checkpoint-a", "candidate-b", "planned-b",
    )
    install_fake_decision(monkeypatch, PortfolioStateV7.COMMIT, "candidate-b")
    second_risk = CumulativeRiskReceiptV1(
        root_source_hash="root",
        checkpoint_hash="checkpoint-b",
        ordered_planned_arm_ids=("planned-a", "planned-b"),
        net_gain_lower=0.12,
        harm_upper=0.015,
        severe_probability_upper=0.002,
        harmed_fraction_upper=0.012,
        cvar95_upper=0.025,
        passed=True,
        calibration_authority_hash="sequence-calibration",
    )
    second_cost = SequentialCostEntryV1(
        1, "probe", 0.5, counters(1, 0.5), "exec-b",
    )
    with pytest.raises(ValueError, match="transition receipt"):
        advance_multi_action_v1(
            first.state, policy, manifest_b, "families", proposal_b, (), (),
            "calibration", (), (), step_cost=second_cost,
            committed_checkpoint_hash="checkpoint-b", cumulative_risk=second_risk,
        )

    transition = TemporalCompositionReceiptV1(
        parent_checkpoint_hash="checkpoint-a",
        ordered_prefix_planned_arm_ids=("planned-a",),
        next_planned_arm_id="planned-b",
        transition_registry_hash="transition-registry",
        sequential_calibration_authority_hash="sequence-calibration",
        interaction_harm_upper=0.01,
        eligible=True,
    )
    second = advance_multi_action_v1(
        first.state, policy, manifest_b, "families", proposal_b, (), (),
        "calibration", (), (), step_cost=second_cost,
        committed_checkpoint_hash="checkpoint-b", cumulative_risk=second_risk,
        temporal_composition=transition,
    )
    assert second.state.run_state is MultiActionRunStateV1.COMPLETE
    assert second.state.committed_planned_arm_ids == ("planned-a", "planned-b")
    assert second.state.cost_ledger.scalar_cost == pytest.approx(1.0)
