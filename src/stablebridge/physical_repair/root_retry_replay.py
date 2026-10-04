"""Outcome-only offline replay for a frozen ROOT_RETRY policy.

This module is action-bank independent and has no action executor, GPU call,
filesystem operation, or path input.  It consumes already materialized
root-level attempt outcomes in their frozen order.  It must not be used to
approximate ``COMPOSED_CHAIN``: a child-of-action outcome is a different
estimand and is rejected explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Sequence

from .uncertainty_evaluation import (
    GroupBootstrapIntervalV1,
    grouped_bootstrap_mean_v1,
)


ROOT_RETRY_REPLAY_SCHEMA_V1 = "stablebridge-root-retry-offline-replay/v1"


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty canonical text")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain NUL")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


class OfflineReplayModeV1(str, Enum):
    ROOT_RETRY = "root_retry"
    COMPOSED_CHAIN = "composed_chain"


class AttemptDecisionV1(str, Enum):
    COMMIT = "commit"
    STOP = "stop"


class AttemptOutcomeScopeV1(str, Enum):
    ROOT = "root"
    COMPOSED_CHAIN = "composed_chain"


@dataclass(frozen=True)
class RootRetryReplayConfigV1:
    """Frozen evaluation policy; composed chains are outside this evaluator."""

    maximum_attempts: int
    severe_harm_threshold: float
    mode: OfflineReplayModeV1 = OfflineReplayModeV1.ROOT_RETRY

    def __post_init__(self) -> None:
        if not isinstance(self.mode, OfflineReplayModeV1):
            raise ValueError("offline replay mode must be typed")
        if self.mode is OfflineReplayModeV1.COMPOSED_CHAIN:
            raise ValueError(
                "COMPOSED_CHAIN cannot be replayed from root-level outcomes"
            )
        if (
            isinstance(self.maximum_attempts, bool)
            or not isinstance(self.maximum_attempts, int)
            or self.maximum_attempts < 1
        ):
            raise ValueError("maximum_attempts must be a positive integer")
        threshold = _nonnegative(
            self.severe_harm_threshold, "severe harm threshold",
        )
        object.__setattr__(self, "severe_harm_threshold", threshold)


@dataclass(frozen=True)
class AttemptOutcomeV1:
    """Previously observed outcome of one attempt against an immutable root.

    Outcome metrics may be absent for STOP (for example, failed execution),
    but a COMMIT must carry finite gain and harm.  ``cost`` is always charged
    once the attempt is visited, irrespective of STOP or COMMIT.
    """

    ordinal: int
    attempt_id: str
    parent_source_hash: str
    decision: AttemptDecisionV1
    cost: float
    outcome_receipt_hash: str
    gain: float | None = None
    harm: float | None = None
    scope: AttemptOutcomeScopeV1 = AttemptOutcomeScopeV1.ROOT

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("attempt ordinal must be a nonnegative integer")
        _text(self.attempt_id, "attempt id")
        _text(self.parent_source_hash, "attempt parent source hash")
        _text(self.outcome_receipt_hash, "outcome receipt hash")
        if not isinstance(self.decision, AttemptDecisionV1):
            raise ValueError("attempt decision must be typed")
        if not isinstance(self.scope, AttemptOutcomeScopeV1):
            raise ValueError("attempt outcome scope must be typed")
        cost = _nonnegative(self.cost, "attempt cost")
        gain = None if self.gain is None else _finite(self.gain, "attempt gain")
        harm = None if self.harm is None else _nonnegative(self.harm, "attempt harm")
        if self.decision is AttemptDecisionV1.COMMIT and (gain is None or harm is None):
            raise ValueError("COMMIT attempt needs finite gain and harm outcomes")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "harm", harm)


@dataclass(frozen=True)
class FrozenRootAttemptOutcomesV1:
    """One case's root outcomes in an outcome-independent frozen order."""

    case_id: str
    scene_group: str
    root_source_hash: str
    attempts: tuple[AttemptOutcomeV1, ...]
    frozen_order_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        _text(self.case_id, "case id")
        _text(self.scene_group, "scene group")
        _text(self.root_source_hash, "root source hash")
        attempts = tuple(self.attempts)
        if not attempts or any(not isinstance(row, AttemptOutcomeV1) for row in attempts):
            raise ValueError("case needs typed root attempt outcomes")
        if tuple(row.ordinal for row in attempts) != tuple(range(len(attempts))):
            raise ValueError("attempts must follow contiguous frozen order")
        if len({row.attempt_id for row in attempts}) != len(attempts):
            raise ValueError("attempt id replay detected")
        if len({row.outcome_receipt_hash for row in attempts}) != len(attempts):
            raise ValueError("outcome receipt replay detected")
        if any(row.scope is AttemptOutcomeScopeV1.COMPOSED_CHAIN for row in attempts):
            raise ValueError(
                "COMPOSED_CHAIN outcome cannot enter a root-attempt replay"
            )
        if any(row.parent_source_hash != self.root_source_hash for row in attempts):
            raise ValueError("attempt is not bound to the immutable root source")
        expected = _hash({
            "schema": ROOT_RETRY_REPLAY_SCHEMA_V1,
            "case_id": self.case_id,
            "root_source_hash": self.root_source_hash,
            "ordered_attempt_ids": [row.attempt_id for row in attempts],
        })
        if self.frozen_order_hash and self.frozen_order_hash != expected:
            raise ValueError("frozen attempt order hash drift")
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "frozen_order_hash", expected)


@dataclass(frozen=True)
class RootRetryCaseReplayV1:
    case_id: str
    scene_group: str
    maximum_attempts: int
    attempted_ids: tuple[str, ...]
    terminal_decision: AttemptDecisionV1
    committed_attempt_id: str | None
    gain: float
    harm: float
    severe_harm: bool
    adopted: bool
    total_cost: float
    frozen_order_hash: str


@dataclass(frozen=True)
class RootRetryCaseComparisonV1:
    case_id: str
    scene_group: str
    single_attempt: RootRetryCaseReplayV1
    retry: RootRetryCaseReplayV1
    gain_difference: float
    harm_difference: float
    severe_rate_difference: float
    adoption_rate_difference: float
    cost_difference: float


@dataclass(frozen=True)
class RootRetryPolicyMetricsV1:
    case_count: int
    mean_gain: float
    mean_harm: float
    severe_harm_rate: float
    adoption_rate: float
    mean_cost: float
    total_cost: float


@dataclass(frozen=True)
class RootRetryMetricDifferencesV1:
    mean_gain: float
    mean_harm: float
    severe_harm_rate: float
    adoption_rate: float
    mean_cost: float
    total_cost: float


@dataclass(frozen=True)
class RootRetryEvaluationV1:
    single_attempt: RootRetryPolicyMetricsV1
    retry: RootRetryPolicyMetricsV1
    retry_minus_single: RootRetryMetricDifferencesV1
    cases: tuple[RootRetryCaseComparisonV1, ...]


@dataclass(frozen=True)
class RootRetryGroupedBootstrapV1:
    """Scene-group bootstrap intervals for retry-minus-single means."""

    gain_difference: GroupBootstrapIntervalV1
    harm_difference: GroupBootstrapIntervalV1
    severe_rate_difference: GroupBootstrapIntervalV1
    adoption_rate_difference: GroupBootstrapIntervalV1
    cost_difference: GroupBootstrapIntervalV1


def replay_root_retry_case_v1(
    case: FrozenRootAttemptOutcomesV1,
    config: RootRetryReplayConfigV1,
) -> RootRetryCaseReplayV1:
    """Replay one case without executing or materializing any action."""

    if not isinstance(case, FrozenRootAttemptOutcomesV1):
        raise ValueError("root retry replay needs typed frozen case outcomes")
    if not isinstance(config, RootRetryReplayConfigV1):
        raise ValueError("root retry replay needs a typed config")
    attempted: list[str] = []
    total_cost = 0.0
    committed: AttemptOutcomeV1 | None = None
    for outcome in case.attempts[:config.maximum_attempts]:
        attempted.append(outcome.attempt_id)
        total_cost += outcome.cost
        if outcome.decision is AttemptDecisionV1.COMMIT:
            committed = outcome
            break

    if committed is None:
        terminal = AttemptDecisionV1.STOP
        committed_id = None
        gain = harm = 0.0
        severe = adopted = False
    else:
        terminal = AttemptDecisionV1.COMMIT
        committed_id = committed.attempt_id
        # COMMIT completeness is established by AttemptOutcomeV1.
        assert committed.gain is not None and committed.harm is not None
        gain, harm = committed.gain, committed.harm
        severe = harm >= config.severe_harm_threshold
        adopted = True
    return RootRetryCaseReplayV1(
        case_id=case.case_id,
        scene_group=case.scene_group,
        maximum_attempts=config.maximum_attempts,
        attempted_ids=tuple(attempted),
        terminal_decision=terminal,
        committed_attempt_id=committed_id,
        gain=gain,
        harm=harm,
        severe_harm=severe,
        adopted=adopted,
        total_cost=float(total_cost),
        frozen_order_hash=case.frozen_order_hash,
    )


def _policy_metrics(rows: Sequence[RootRetryCaseReplayV1]) -> RootRetryPolicyMetricsV1:
    count = len(rows)
    if count == 0:
        raise ValueError("root retry evaluation needs at least one case")
    total_cost = sum(row.total_cost for row in rows)
    return RootRetryPolicyMetricsV1(
        case_count=count,
        mean_gain=float(sum(row.gain for row in rows) / count),
        mean_harm=float(sum(row.harm for row in rows) / count),
        severe_harm_rate=float(sum(row.severe_harm for row in rows) / count),
        adoption_rate=float(sum(row.adopted for row in rows) / count),
        mean_cost=float(total_cost / count),
        total_cost=float(total_cost),
    )


def evaluate_single_vs_root_retry_v1(
    cases: Sequence[FrozenRootAttemptOutcomesV1],
    retry_config: RootRetryReplayConfigV1,
) -> RootRetryEvaluationV1:
    """Compare maximum-one-attempt and ROOT_RETRY on identical frozen cases."""

    if not isinstance(retry_config, RootRetryReplayConfigV1):
        raise ValueError("root retry evaluation needs a typed config")
    if retry_config.maximum_attempts < 2:
        raise ValueError("single-vs-retry evaluation needs at least two attempts")
    typed_cases = tuple(cases)
    if not typed_cases or any(
        not isinstance(case, FrozenRootAttemptOutcomesV1) for case in typed_cases
    ):
        raise ValueError("root retry evaluation needs typed frozen cases")
    case_ids = [case.case_id for case in typed_cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate case id in root retry evaluation")
    single_config = RootRetryReplayConfigV1(
        maximum_attempts=1,
        severe_harm_threshold=retry_config.severe_harm_threshold,
    )
    comparisons: list[RootRetryCaseComparisonV1] = []
    singles: list[RootRetryCaseReplayV1] = []
    retries: list[RootRetryCaseReplayV1] = []
    for case in typed_cases:
        single = replay_root_retry_case_v1(case, single_config)
        retry = replay_root_retry_case_v1(case, retry_config)
        singles.append(single)
        retries.append(retry)
        comparisons.append(RootRetryCaseComparisonV1(
            case_id=case.case_id,
            scene_group=case.scene_group,
            single_attempt=single,
            retry=retry,
            gain_difference=retry.gain - single.gain,
            harm_difference=retry.harm - single.harm,
            severe_rate_difference=float(retry.severe_harm) - float(single.severe_harm),
            adoption_rate_difference=float(retry.adopted) - float(single.adopted),
            cost_difference=retry.total_cost - single.total_cost,
        ))
    single_metrics = _policy_metrics(singles)
    retry_metrics = _policy_metrics(retries)
    return RootRetryEvaluationV1(
        single_attempt=single_metrics,
        retry=retry_metrics,
        retry_minus_single=RootRetryMetricDifferencesV1(
            mean_gain=retry_metrics.mean_gain - single_metrics.mean_gain,
            mean_harm=retry_metrics.mean_harm - single_metrics.mean_harm,
            severe_harm_rate=(
                retry_metrics.severe_harm_rate - single_metrics.severe_harm_rate
            ),
            adoption_rate=retry_metrics.adoption_rate - single_metrics.adoption_rate,
            mean_cost=retry_metrics.mean_cost - single_metrics.mean_cost,
            total_cost=retry_metrics.total_cost - single_metrics.total_cost,
        ),
        cases=tuple(comparisons),
    )


def grouped_bootstrap_root_retry_differences_v1(
    evaluation: RootRetryEvaluationV1,
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> RootRetryGroupedBootstrapV1:
    """Bootstrap paired policy differences by scene rather than by case row."""

    if not isinstance(evaluation, RootRetryEvaluationV1):
        raise ValueError("group bootstrap needs a typed root retry evaluation")
    rows = evaluation.cases
    groups = [row.scene_group for row in rows]

    def interval(attribute: str) -> GroupBootstrapIntervalV1:
        return grouped_bootstrap_mean_v1(
            [getattr(row, attribute) for row in rows],
            groups,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        )

    return RootRetryGroupedBootstrapV1(
        gain_difference=interval("gain_difference"),
        harm_difference=interval("harm_difference"),
        severe_rate_difference=interval("severe_rate_difference"),
        adoption_rate_difference=interval("adoption_rate_difference"),
        cost_difference=interval("cost_difference"),
    )


__all__ = [
    "AttemptDecisionV1",
    "AttemptOutcomeScopeV1",
    "AttemptOutcomeV1",
    "FrozenRootAttemptOutcomesV1",
    "OfflineReplayModeV1",
    "ROOT_RETRY_REPLAY_SCHEMA_V1",
    "RootRetryCaseComparisonV1",
    "RootRetryCaseReplayV1",
    "RootRetryEvaluationV1",
    "RootRetryGroupedBootstrapV1",
    "RootRetryMetricDifferencesV1",
    "RootRetryPolicyMetricsV1",
    "RootRetryReplayConfigV1",
    "evaluate_single_vs_root_retry_v1",
    "grouped_bootstrap_root_retry_differences_v1",
    "replay_root_retry_case_v1",
]
