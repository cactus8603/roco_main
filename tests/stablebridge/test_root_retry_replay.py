from __future__ import annotations

from dataclasses import replace

import pytest

from stablebridge.physical_repair.root_retry_replay import (
    AttemptDecisionV1,
    AttemptOutcomeScopeV1,
    AttemptOutcomeV1,
    FrozenRootAttemptOutcomesV1,
    OfflineReplayModeV1,
    RootRetryReplayConfigV1,
    evaluate_single_vs_root_retry_v1,
    grouped_bootstrap_root_retry_differences_v1,
    replay_root_retry_case_v1,
)


def attempt(
    ordinal: int,
    decision: AttemptDecisionV1,
    *,
    root: str = "root-hash",
    cost: float = 1.0,
    gain: float | None = None,
    harm: float | None = None,
    scope: AttemptOutcomeScopeV1 = AttemptOutcomeScopeV1.ROOT,
) -> AttemptOutcomeV1:
    return AttemptOutcomeV1(
        ordinal=ordinal,
        attempt_id=f"attempt-{ordinal}",
        parent_source_hash=root,
        decision=decision,
        cost=cost,
        outcome_receipt_hash=f"receipt-{ordinal}",
        gain=gain,
        harm=harm,
        scope=scope,
    )


def case(
    case_id: str,
    scene: str,
    *attempts: AttemptOutcomeV1,
    root: str = "root-hash",
) -> FrozenRootAttemptOutcomesV1:
    return FrozenRootAttemptOutcomesV1(
        case_id=case_id,
        scene_group=scene,
        root_source_hash=root,
        attempts=attempts,
    )


def config(maximum_attempts: int = 2) -> RootRetryReplayConfigV1:
    return RootRetryReplayConfigV1(
        maximum_attempts=maximum_attempts,
        severe_harm_threshold=0.25,
    )


def test_stop_retries_from_root_commit_terminates_and_sunk_cost_remains():
    outcomes = case(
        "case-a", "scene-a",
        attempt(0, AttemptDecisionV1.STOP, cost=2.0),
        attempt(1, AttemptDecisionV1.COMMIT, cost=3.0, gain=1.2, harm=0.1),
        attempt(2, AttemptDecisionV1.COMMIT, cost=99.0, gain=9.0, harm=0.0),
    )
    replay = replay_root_retry_case_v1(outcomes, config(3))
    assert replay.attempted_ids == ("attempt-0", "attempt-1")
    assert replay.terminal_decision is AttemptDecisionV1.COMMIT
    assert replay.committed_attempt_id == "attempt-1"
    assert replay.gain == pytest.approx(1.2)
    assert replay.harm == pytest.approx(0.1)
    assert replay.severe_harm is False
    assert replay.total_cost == pytest.approx(5.0)


def test_final_stop_returns_native_metrics_but_keeps_all_attempt_costs():
    outcomes = case(
        "case-a", "scene-a",
        attempt(0, AttemptDecisionV1.STOP, cost=2.0, gain=-2.0, harm=2.0),
        attempt(1, AttemptDecisionV1.STOP, cost=4.0),
        attempt(2, AttemptDecisionV1.COMMIT, cost=8.0, gain=3.0, harm=0.0),
    )
    replay = replay_root_retry_case_v1(outcomes, config(2))
    assert replay.terminal_decision is AttemptDecisionV1.STOP
    assert replay.committed_attempt_id is None
    assert replay.gain == replay.harm == 0.0
    assert replay.severe_harm is replay.adopted is False
    assert replay.total_cost == pytest.approx(6.0)


def test_first_commit_makes_single_and_retry_identical_and_ignores_later_cost():
    outcomes = case(
        "case-a", "scene-a",
        attempt(0, AttemptDecisionV1.COMMIT, cost=1.5, gain=0.5, harm=0.3),
        attempt(1, AttemptDecisionV1.COMMIT, cost=50.0, gain=2.0, harm=0.0),
    )
    evaluation = evaluate_single_vs_root_retry_v1((outcomes,), config())
    assert evaluation.single_attempt == evaluation.retry
    assert evaluation.retry.severe_harm_rate == pytest.approx(1.0)
    assert evaluation.retry.total_cost == pytest.approx(1.5)
    assert evaluation.retry_minus_single.mean_gain == 0.0
    assert evaluation.retry_minus_single.total_cost == 0.0


def test_single_vs_retry_reports_paired_gain_harm_severe_adoption_and_cost():
    outcomes = case(
        "case-a", "scene-a",
        attempt(0, AttemptDecisionV1.STOP, cost=2.0),
        attempt(1, AttemptDecisionV1.COMMIT, cost=3.0, gain=1.2, harm=0.3),
    )
    evaluation = evaluate_single_vs_root_retry_v1((outcomes,), config())
    assert evaluation.single_attempt.mean_gain == 0.0
    assert evaluation.single_attempt.adoption_rate == 0.0
    assert evaluation.single_attempt.total_cost == pytest.approx(2.0)
    assert evaluation.retry.mean_gain == pytest.approx(1.2)
    assert evaluation.retry.mean_harm == pytest.approx(0.3)
    assert evaluation.retry.severe_harm_rate == pytest.approx(1.0)
    assert evaluation.retry.adoption_rate == pytest.approx(1.0)
    assert evaluation.retry.total_cost == pytest.approx(5.0)
    delta = evaluation.retry_minus_single
    assert (
        delta.mean_gain,
        delta.mean_harm,
        delta.severe_harm_rate,
        delta.adoption_rate,
        delta.mean_cost,
    ) == pytest.approx((1.2, 0.3, 1.0, 1.0, 3.0))


def test_frozen_order_is_contiguous_hashed_and_not_resorted():
    first = attempt(0, AttemptDecisionV1.STOP)
    second = attempt(1, AttemptDecisionV1.STOP)
    outcomes = case("case-a", "scene-a", first, second)
    assert len(outcomes.frozen_order_hash) == 64
    with pytest.raises(ValueError, match="frozen order"):
        case("case-a", "scene-a", second, first)
    with pytest.raises(ValueError, match="hash drift"):
        replace(outcomes, frozen_order_hash="wrong")


def test_composed_chain_mode_and_outcomes_are_explicitly_forbidden():
    with pytest.raises(ValueError, match="COMPOSED_CHAIN"):
        RootRetryReplayConfigV1(
            maximum_attempts=2,
            severe_harm_threshold=0.25,
            mode=OfflineReplayModeV1.COMPOSED_CHAIN,
        )
    with pytest.raises(ValueError, match="COMPOSED_CHAIN outcome"):
        case(
            "case-a", "scene-a",
            attempt(
                0, AttemptDecisionV1.COMMIT, gain=1.0, harm=0.0,
                scope=AttemptOutcomeScopeV1.COMPOSED_CHAIN,
            ),
        )


def test_every_visited_attempt_must_be_bound_to_same_root():
    with pytest.raises(ValueError, match="immutable root"):
        case(
            "case-a", "scene-a",
            attempt(0, AttemptDecisionV1.STOP),
            attempt(
                1, AttemptDecisionV1.COMMIT, root="child-hash",
                gain=1.0, harm=0.0,
            ),
        )


def test_commit_requires_metrics_but_failed_stop_may_have_none():
    with pytest.raises(ValueError, match="needs finite gain and harm"):
        attempt(0, AttemptDecisionV1.COMMIT, gain=None, harm=None)
    stopped = attempt(0, AttemptDecisionV1.STOP, cost=0.5)
    assert replay_root_retry_case_v1(
        case("case-a", "scene-a", stopped), config(),
    ).total_cost == pytest.approx(0.5)


def test_group_bootstrap_reuses_scene_groups_for_paired_differences():
    cases = (
        case(
            "a1", "scene-a",
            attempt(0, AttemptDecisionV1.STOP, cost=1.0),
            attempt(1, AttemptDecisionV1.COMMIT, cost=2.0, gain=2.0, harm=0.0),
        ),
        case(
            "a2", "scene-a",
            attempt(0, AttemptDecisionV1.STOP, cost=1.0),
            attempt(1, AttemptDecisionV1.COMMIT, cost=2.0, gain=0.0, harm=0.5),
        ),
        case(
            "b1", "scene-b",
            attempt(0, AttemptDecisionV1.STOP, cost=1.0),
            attempt(1, AttemptDecisionV1.STOP, cost=3.0),
        ),
    )
    evaluation = evaluate_single_vs_root_retry_v1(cases, config())
    bootstrap = grouped_bootstrap_root_retry_differences_v1(
        evaluation, resamples=200, seed=7,
    )
    assert bootstrap.gain_difference.point == pytest.approx(2.0 / 3.0)
    assert bootstrap.harm_difference.point == pytest.approx(0.5 / 3.0)
    assert bootstrap.adoption_rate_difference.point == pytest.approx(2.0 / 3.0)
    assert bootstrap.cost_difference.point == pytest.approx(7.0 / 3.0)
    assert bootstrap.gain_difference.group_count == 2


def test_evaluator_is_pure_and_has_no_storage_or_gpu_surface(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline replay attempted external I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    outcomes = case(
        "case-a", "scene-a",
        attempt(0, AttemptDecisionV1.STOP),
        attempt(1, AttemptDecisionV1.COMMIT, gain=1.0, harm=0.0),
    )
    result = evaluate_single_vs_root_retry_v1((outcomes,), config())
    assert result.retry.adoption_rate == 1.0


def test_duplicate_cases_and_non_retry_comparison_are_rejected():
    outcomes = case(
        "case-a", "scene-a",
        attempt(0, AttemptDecisionV1.STOP),
        attempt(1, AttemptDecisionV1.STOP),
    )
    with pytest.raises(ValueError, match="duplicate case"):
        evaluate_single_vs_root_retry_v1((outcomes, outcomes), config())
    with pytest.raises(ValueError, match="at least two"):
        evaluate_single_vs_root_retry_v1((outcomes,), config(1))
