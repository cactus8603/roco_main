"""Profile-driven, region-aware selector contract.

This module turns the auditable action profiles into executable constraints.
It does not classify renderer corruptions.  Each hypothesis is an action,
endpoint, region, and *set* of still-identifiable parameters.  Physical
evidence admits a hypothesis; only an independent signed task witness can
authorize delivery.  Native output remains the zero-cost feasible decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from .action_profiles import ACTION_PHYSICS_PROFILES
from .contracts import PhysicalCertificate
from .selector_v2 import EvidenceSplit, RecoverabilityEvidence


V3_STATES = frozenset({"native", "probe", "ambiguous", "selected"})
WITNESS_KINDS = frozenset({"statistical", "deterministic"})


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _fraction(value: float, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


@dataclass(frozen=True)
class WitnessBound:
    """A numerically verified physical inequality on frozen B support.

    Witness names alone are not evidence.  Every decision/closure witness has
    to expose a conservative signed margin (positive means that the required
    inequality passed), the exact support on which it was checked, and the
    complete parameter subset covered by the bound.
    """

    name: str
    margin_lower: float
    support_hash: str
    parameter_keys: tuple[str, ...]
    kind: str = "statistical"
    effective_n: float = 0.0
    method: str = "fixed_holdout"

    def __post_init__(self) -> None:
        if not self.name or not self.support_hash or not self.method:
            raise ValueError("witness identity, support, and method are required")
        if self.kind not in WITNESS_KINDS:
            raise ValueError("unknown witness kind")
        _finite(self.margin_lower, "witness margin lower")
        _nonnegative(self.effective_n, "witness effective n")
        if not self.parameter_keys or len(set(self.parameter_keys)) != len(
            self.parameter_keys
        ):
            raise ValueError("witness needs unique covered parameter keys")


@dataclass(frozen=True)
class ParameterState:
    """Worst-case inputs for one member of an identifiable parameter set."""

    key: str
    physical_margin_lower: float
    information_retention_lower: float
    predicted_utility_mean: float | None = None
    utility_radius: float | None = None
    predicted_harm_upper: float | None = None
    severe_harm_score: float | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("parameter state needs a key")
        _finite(self.physical_margin_lower, "physical margin lower")
        _fraction(self.information_retention_lower, "information retention lower")
        task = (
            self.predicted_utility_mean,
            self.utility_radius,
            self.predicted_harm_upper,
            self.severe_harm_score,
        )
        if any(value is None for value in task):
            if not all(value is None for value in task):
                raise ValueError("parameter task evidence must be all present or all absent")
        else:
            _finite(self.predicted_utility_mean, "predicted utility mean")
            _nonnegative(self.utility_radius, "utility radius")
            _nonnegative(self.predicted_harm_upper, "predicted harm upper")
            _nonnegative(self.severe_harm_score, "severe harm score")

    @property
    def has_task_witness(self) -> bool:
        return self.predicted_utility_mean is not None

    @property
    def utility_lower(self) -> float:
        if not self.has_task_witness:
            raise ValueError("task witness is absent")
        assert self.predicted_utility_mean is not None
        assert self.utility_radius is not None
        return float(self.predicted_utility_mean - self.utility_radius)

    @property
    def utility_upper(self) -> float:
        if not self.has_task_witness:
            raise ValueError("task witness is absent")
        assert self.predicted_utility_mean is not None
        assert self.utility_radius is not None
        return float(self.predicted_utility_mean + self.utility_radius)


@dataclass(frozen=True)
class SequentialEvidence:
    attempts_seen: int = 1
    anytime_valid: bool = False
    method: str = "fixed_once"
    alpha_spent: float = 0.05

    def __post_init__(self) -> None:
        if self.attempts_seen < 1 or not self.method:
            raise ValueError("invalid sequential evidence metadata")
        _fraction(self.alpha_spent, "alpha spent")


@dataclass(frozen=True)
class RegionActionHypothesis:
    key: str
    certificate: PhysicalCertificate
    evidence_split: EvidenceSplit
    recoverability: RecoverabilityEvidence
    region_key: str
    parameters: tuple[ParameterState, ...]
    decision_witnesses: tuple[WitnessBound, ...]
    closure_witnesses: tuple[WitnessBound, ...]
    fit_effective_n: float
    check_effective_n: float
    pixel_fraction: float
    task_support_retention: float
    task_verifier_support_hash: str | None
    task_comparison_support_hash: str | None
    expected_compute_seconds: float
    extra_matcher_trajectories: int
    sequential: SequentialEvidence = SequentialEvidence()
    conflict_keys: tuple[str, ...] = ()
    probe_value_upper: float = 0.0
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if not self.key or not self.region_key:
            raise ValueError("hypothesis and region keys are required")
        operator = self.certificate.action.operator_id
        if operator not in ACTION_PHYSICS_PROFILES:
            raise ValueError("hypothesis operator lacks an action profile")
        endpoint = self.certificate.action.hypothesized_degraded_endpoint
        expected_prefix = f"{operator}@{endpoint}#"
        if not self.key.startswith(expected_prefix):
            raise ValueError("hypothesis key does not match action identity")
        if not self.parameters or len({row.key for row in self.parameters}) != len(
            self.parameters
        ):
            raise ValueError("hypothesis needs unique parameter states")
        profile = ACTION_PHYSICS_PROFILES[operator]
        if profile.parameter_geometry == "fixed" and len(self.parameters) != 1:
            raise ValueError("fixed actions need exactly one parameter state")
        task_presence = {row.has_task_witness for row in self.parameters}
        if len(task_presence) != 1:
            raise ValueError("task evidence must cover the complete parameter set")
        has_task = next(iter(task_presence))
        task_hashes = (
            self.task_verifier_support_hash,
            self.task_comparison_support_hash,
        )
        if has_task and any(not value for value in task_hashes):
            raise ValueError("task evidence needs verifier and comparison support hashes")
        if not has_task and any(value is not None for value in task_hashes):
            raise ValueError("task support hashes require task evidence")
        decision_names = tuple(item.name for item in self.decision_witnesses)
        closure_names = tuple(item.name for item in self.closure_witnesses)
        if len(set(decision_names)) != len(decision_names):
            raise ValueError("decision witnesses must be unique")
        if len(set(closure_names)) != len(closure_names):
            raise ValueError("closure witnesses must be unique")
        parameter_keys = {item.key for item in self.parameters}
        for witness in (*self.decision_witnesses, *self.closure_witnesses):
            if set(witness.parameter_keys) != parameter_keys:
                raise ValueError(
                    "every physical witness must cover the complete parameter set"
                )
        _nonnegative(self.fit_effective_n, "fit effective n")
        _nonnegative(self.check_effective_n, "check effective n")
        _fraction(self.pixel_fraction, "pixel fraction")
        _fraction(self.task_support_retention, "task support retention")
        _nonnegative(self.expected_compute_seconds, "expected compute seconds")
        if self.extra_matcher_trajectories < 0:
            raise ValueError("matcher trajectory count must be nonnegative")
        _nonnegative(self.probe_value_upper, "probe value upper")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("selector hypotheses cannot read labels, truth, or outcomes")

    @property
    def has_task_witness(self) -> bool:
        return self.parameters[0].has_task_witness


@dataclass(frozen=True)
class SelectorV3Config:
    maximum_familywise_error_rate: float = 0.05
    minimum_effective_samples: float = 4.0
    minimum_information_retention: float = 0.05
    minimum_task_support_retention: float = 0.5
    minimum_utility_lower: float = 0.0
    small_harm_budget: float = 0.05
    severe_harm_threshold: float = 0.25
    minimum_trust_alpha: float = 0.05
    maximum_total_weighted_harm: float = 0.05
    maximum_total_pixel_fraction: float = 1.0
    maximum_matcher_trajectories: int = 3
    maximum_compute_seconds: float = 30.0
    trajectory_cost_seconds: float = 0.0
    cost_penalty_per_second: float = 0.0
    separation_margin: float = 0.0
    maximum_exact_hypotheses: int = 20

    def __post_init__(self) -> None:
        for value, name in (
            (self.maximum_familywise_error_rate, "maximum familywise error rate"),
            (self.minimum_information_retention, "minimum information retention"),
            (self.minimum_task_support_retention, "minimum task support retention"),
            (self.minimum_trust_alpha, "minimum trust alpha"),
            (self.maximum_total_pixel_fraction, "maximum total pixel fraction"),
        ):
            _fraction(value, name)
        for value, name in (
            (self.minimum_effective_samples, "minimum effective samples"),
            (self.small_harm_budget, "small harm budget"),
            (self.severe_harm_threshold, "severe harm threshold"),
            (self.maximum_total_weighted_harm, "maximum weighted harm"),
            (self.maximum_compute_seconds, "maximum compute seconds"),
            (self.trajectory_cost_seconds, "trajectory cost seconds"),
            (self.cost_penalty_per_second, "cost penalty per second"),
            (self.separation_margin, "separation margin"),
        ):
            _nonnegative(value, name)
        _finite(self.minimum_utility_lower, "minimum utility lower")
        if self.maximum_matcher_trajectories < 0 or self.maximum_exact_hypotheses < 1:
            raise ValueError("invalid trajectory or exact-selection budget")
        if self.small_harm_budget > self.severe_harm_threshold:
            raise ValueError("small harm budget cannot exceed severe threshold")


@dataclass(frozen=True)
class _Choice:
    row: RegionActionHypothesis
    trust_alpha: float
    robust_mean: float
    robust_lower: float
    robust_upper: float
    harm_upper: float
    objective: float


@dataclass(frozen=True)
class SelectorV3Decision:
    state: str
    admitted_hypotheses: tuple[str, ...]
    selected_hypotheses: tuple[str, ...]
    probe_hypotheses: tuple[str, ...]
    ambiguous_hypotheses: tuple[str, ...]
    rejected_hypotheses: Mapping[str, tuple[str, ...]]
    trust_scales: Mapping[str, float]
    robust_utility_intervals: Mapping[str, tuple[float, float]]
    total_objective: float
    total_weighted_harm: float
    total_compute_seconds: float
    total_matcher_trajectories: int
    stop_reason: str

    def __post_init__(self) -> None:
        if self.state not in V3_STATES or not self.stop_reason:
            raise ValueError("invalid Selector-v3 decision")
        _finite(self.total_objective, "total objective")
        _nonnegative(self.total_weighted_harm, "total weighted harm")
        _nonnegative(self.total_compute_seconds, "total compute seconds")
        if self.total_matcher_trajectories < 0:
            raise ValueError("negative matcher trajectory count")
        if self.state == "selected" and not self.selected_hypotheses:
            raise ValueError("selected state needs at least one hypothesis")
        if self.state != "selected" and self.selected_hypotheses:
            raise ValueError("non-selected state cannot deliver interventions")


def _physical_reasons(row: RegionActionHypothesis,
                      config: SelectorV3Config) -> list[str]:
    reasons: list[str] = []
    profile = ACTION_PHYSICS_PROFILES[row.certificate.action.operator_id]
    if row.certificate.status != "supported":
        reasons.append(f"physical_{row.certificate.status}")
    split = row.evidence_split
    if not split.multiplicity_controlled:
        reasons.append("multiplicity_uncontrolled")
    if split.familywise_error_rate > config.maximum_familywise_error_rate:
        reasons.append("familywise_error_too_large")
    if row.fit_effective_n < config.minimum_effective_samples:
        reasons.append("insufficient_fit_effective_samples")
    if row.check_effective_n < config.minimum_effective_samples:
        reasons.append("insufficient_check_effective_samples")
    if not profile.decision_witnesses:
        reasons.append("action_profile_has_no_decision_witness")
    decision = {item.name: item for item in row.decision_witnesses}
    closure = {item.name: item for item in row.closure_witnesses}
    missing_decision = sorted(set(profile.decision_witnesses) - set(decision))
    reasons.extend(f"missing_decision_witness:{name}" for name in missing_decision)
    required_closure = set(profile.closure_witnesses) - {"task_signed_utility"}
    missing_closure = sorted(required_closure - set(closure))
    reasons.extend(f"missing_closure_witness:{name}" for name in missing_closure)
    for stage, required, observed in (
        ("decision", set(profile.decision_witnesses), decision),
        ("closure", required_closure, closure),
    ):
        for name in sorted(required & set(observed)):
            witness = observed[name]
            if witness.support_hash != split.certificate_support_hash:
                reasons.append(f"{stage}_witness_support_changed:{name}")
            if witness.margin_lower <= 0.0:
                reasons.append(f"{stage}_witness_bound_not_positive:{name}")
            if (witness.kind == "statistical"
                    and witness.effective_n < config.minimum_effective_samples):
                reasons.append(f"{stage}_witness_effective_n_insufficient:{name}")
    if row.sequential.attempts_seen > 1 and not row.sequential.anytime_valid:
        reasons.append("optional_stopping_uncontrolled")
    if row.sequential.alpha_spent > config.maximum_familywise_error_rate:
        reasons.append("sequential_alpha_budget_exceeded")
    if any(parameter.physical_margin_lower <= 0.0 for parameter in row.parameters):
        reasons.append("parameter_set_physical_lower_not_positive")
    if any(
        parameter.information_retention_lower
        <= config.minimum_information_retention
        for parameter in row.parameters
    ):
        reasons.append("parameter_set_information_insufficient")
    recoverability = row.recoverability
    if recoverability.status != "supported":
        reasons.append(f"recoverability_{recoverability.status}")
    if not recoverability.visibility_certified:
        reasons.append("visibility_uncertified")
    if (recoverability.collision_fraction > 0.0
            and recoverability.collision_policy == "unresolved"):
        reasons.append("collision_unresolved")
    if (recoverability.information_retention_lower
            <= config.minimum_information_retention):
        reasons.append("recoverability_information_insufficient")
    if recoverability.output_support_fraction <= 0.0 or row.pixel_fraction <= 0.0:
        reasons.append("empty_output_support")
    return reasons


def _task_choice(row: RegionActionHypothesis,
                 config: SelectorV3Config) -> tuple[_Choice | None, list[str]]:
    if not row.has_task_witness:
        return None, ["task_witness_missing"]
    reasons: list[str] = []
    if row.evidence_split.verifier_support_hash is None:
        reasons.append("verifier_partition_missing")
    elif row.task_verifier_support_hash != row.evidence_split.verifier_support_hash:
        reasons.append("verifier_partition_changed")
    if row.task_comparison_support_hash != row.recoverability.output_support_hash:
        reasons.append("comparison_support_changed")
    if row.task_support_retention < config.minimum_task_support_retention:
        reasons.append("task_support_retention")
    means = [float(parameter.predicted_utility_mean) for parameter in row.parameters]
    lowers = [parameter.utility_lower for parameter in row.parameters]
    uppers = [parameter.utility_upper for parameter in row.parameters]
    harms = [float(parameter.predicted_harm_upper) for parameter in row.parameters]
    severe = [float(parameter.severe_harm_score) for parameter in row.parameters]
    robust_mean = min(means)
    robust_lower = min(lowers)
    robust_upper = min(uppers)
    harm_upper = max(harms)
    if max(severe) >= config.severe_harm_threshold or harm_upper >= config.severe_harm_threshold:
        reasons.append("severe_harm_guard")
    if robust_mean <= 0.0:
        reasons.append("nonpositive_parameter_set_utility")
    if reasons:
        return None, reasons
    if robust_lower > config.minimum_utility_lower:
        alpha = 1.0
    elif harm_upper <= config.small_harm_budget:
        radius = max(
            float(parameter.utility_radius) for parameter in row.parameters
        )
        confidence_scale = robust_mean / (robust_mean + radius + 1e-12)
        harm_scale = 1.0 if harm_upper <= 0.0 else min(
            1.0, config.small_harm_budget / harm_upper,
        )
        alpha = float(confidence_scale * harm_scale)
        if alpha < config.minimum_trust_alpha:
            return None, ["shrunk_trust_below_minimum"]
    else:
        return None, ["utility_interval_crosses_zero_above_small_harm_budget"]
    cost = row.expected_compute_seconds + (
        row.extra_matcher_trajectories * config.trajectory_cost_seconds
    )
    objective = (
        alpha * robust_mean * row.pixel_fraction
        - config.cost_penalty_per_second * cost
    )
    if objective <= 0.0:
        return None, ["nonpositive_cost_adjusted_objective"]
    return _Choice(
        row=row, trust_alpha=alpha, robust_mean=robust_mean,
        robust_lower=robust_lower, robust_upper=robust_upper,
        harm_upper=harm_upper, objective=objective,
    ), []


def _conflict(left: RegionActionHypothesis,
              right: RegionActionHypothesis) -> bool:
    return bool(
        left.region_key == right.region_key
        or set(left.conflict_keys) & set(right.conflict_keys)
    )


def _best_subset(choices: Sequence[_Choice], config: SelectorV3Config
                 ) -> tuple[_Choice, ...]:
    if len(choices) > config.maximum_exact_hypotheses:
        return ()
    best: tuple[_Choice, ...] = ()
    best_key = (0.0, 0.0, ())
    for size in range(1, len(choices) + 1):
        for subset in combinations(choices, size):
            if any(_conflict(left.row, right.row)
                   for left, right in combinations(subset, 2)):
                continue
            pixels = sum(item.row.pixel_fraction for item in subset)
            harm = sum(
                item.trust_alpha * item.harm_upper * item.row.pixel_fraction
                for item in subset
            )
            trajectories = sum(item.row.extra_matcher_trajectories for item in subset)
            compute = sum(
                item.row.expected_compute_seconds
                + item.row.extra_matcher_trajectories * config.trajectory_cost_seconds
                for item in subset
            )
            if (pixels > config.maximum_total_pixel_fraction
                    or harm > config.maximum_total_weighted_harm
                    or trajectories > config.maximum_matcher_trajectories
                    or compute > config.maximum_compute_seconds):
                continue
            objective = sum(item.objective for item in subset)
            lower = sum(
                item.trust_alpha * item.robust_lower * item.row.pixel_fraction
                for item in subset
            )
            names = tuple(sorted(item.row.key for item in subset))
            key = (objective, lower, tuple(reversed(names)))
            if key > best_key:
                best, best_key = tuple(subset), key
    return best


def select_region_actions_v3(
    hypotheses: Sequence[RegionActionHypothesis],
    *,
    config: SelectorV3Config = SelectorV3Config(),
) -> SelectorV3Decision:
    """Select a non-conflicting regional action set under harm and cost budgets."""
    rows = tuple(hypotheses)
    if len({row.key for row in rows}) != len(rows):
        raise ValueError("hypothesis keys must be unique")
    rejected: dict[str, tuple[str, ...]] = {}
    admitted: list[RegionActionHypothesis] = []
    for row in rows:
        reasons = _physical_reasons(row, config)
        if reasons:
            rejected[row.key] = tuple(reasons)
        else:
            admitted.append(row)
    choices: list[_Choice] = []
    probes: list[RegionActionHypothesis] = []
    intervals: dict[str, tuple[float, float]] = {}
    for row in admitted:
        choice, reasons = _task_choice(row, config)
        if reasons == ["task_witness_missing"]:
            probes.append(row)
        elif reasons:
            rejected[row.key] = tuple(reasons)
        else:
            assert choice is not None
            choices.append(choice)
            intervals[row.key] = (choice.robust_lower, choice.robust_upper)

    ambiguous: set[str] = set()
    for left, right in combinations(choices, 2):
        if not _conflict(left.row, right.row):
            continue
        separated = bool(
            left.robust_lower > right.robust_upper + config.separation_margin
            or right.robust_lower > left.robust_upper + config.separation_margin
        )
        if not separated:
            ambiguous.update((left.row.key, right.row.key))
    usable = [choice for choice in choices if choice.row.key not in ambiguous]
    if len(usable) > config.maximum_exact_hypotheses:
        return SelectorV3Decision(
            state="native", admitted_hypotheses=tuple(sorted(row.key for row in admitted)),
            selected_hypotheses=(), probe_hypotheses=(),
            ambiguous_hypotheses=tuple(sorted(ambiguous)),
            rejected_hypotheses=rejected, trust_scales={},
            robust_utility_intervals=intervals, total_objective=0.0,
            total_weighted_harm=0.0, total_compute_seconds=0.0,
            total_matcher_trajectories=0,
            stop_reason="exact_regional_selection_budget_exceeded",
        )
    selected = _best_subset(usable, config)
    selected_objective = sum(choice.objective for choice in selected)
    # VOI is regional and net of probe cost.  A missing witness in one region
    # must not suppress an already safe action in a disjoint region, and an
    # unaffordable probe is never requested merely because its gross upper
    # value is large.
    eligible_probes: list[tuple[RegionActionHypothesis, float, float]] = []
    for row in probes:
        cost = row.expected_compute_seconds + (
            row.extra_matcher_trajectories * config.trajectory_cost_seconds
        )
        if (cost > config.maximum_compute_seconds
                or row.extra_matcher_trajectories
                > config.maximum_matcher_trajectories):
            rejected[row.key] = ("probe_budget_exceeded",)
            continue
        displaced_objective = sum(
            choice.objective for choice in selected if _conflict(choice.row, row)
        )
        net_upper = (
            row.probe_value_upper - displaced_objective
            - config.cost_penalty_per_second * cost
        )
        if net_upper > 0.0:
            eligible_probes.append((row, net_upper, cost))
    probe_order = tuple(
        row.key for row, _, _ in sorted(
            eligible_probes,
            key=lambda item: (
                item[1] / max(item[2], 1e-12), item[1], item[0].key,
            ),
            reverse=True,
        )
    )
    if not selected and ambiguous:
        return SelectorV3Decision(
            state="ambiguous", admitted_hypotheses=tuple(sorted(row.key for row in admitted)),
            selected_hypotheses=(), probe_hypotheses=probe_order,
            ambiguous_hypotheses=tuple(sorted(ambiguous)), rejected_hypotheses=rejected,
            trust_scales={}, robust_utility_intervals=intervals,
            total_objective=0.0, total_weighted_harm=0.0,
            total_compute_seconds=0.0, total_matcher_trajectories=0,
            stop_reason="conflicting_task_utility_intervals_overlap",
        )
    if not selected:
        if probe_order:
            return SelectorV3Decision(
                state="probe",
                admitted_hypotheses=tuple(sorted(row.key for row in admitted)),
                selected_hypotheses=(), probe_hypotheses=probe_order,
                ambiguous_hypotheses=(), rejected_hypotheses=rejected,
                trust_scales={}, robust_utility_intervals=intervals,
                total_objective=0.0, total_weighted_harm=0.0,
                total_compute_seconds=0.0, total_matcher_trajectories=0,
                stop_reason="positive_net_regional_value_of_information",
            )
        return SelectorV3Decision(
            state="native", admitted_hypotheses=tuple(sorted(row.key for row in admitted)),
            selected_hypotheses=(), probe_hypotheses=probe_order,
            ambiguous_hypotheses=(), rejected_hypotheses=rejected,
            trust_scales={}, robust_utility_intervals=intervals,
            total_objective=0.0, total_weighted_harm=0.0,
            total_compute_seconds=0.0, total_matcher_trajectories=0,
            stop_reason=("no_physically_admitted_action" if not admitted
                         else "no_safe_positive_budget_feasible_action"),
        )
    harm = sum(
        choice.trust_alpha * choice.harm_upper * choice.row.pixel_fraction
        for choice in selected
    )
    compute = sum(
        choice.row.expected_compute_seconds
        + choice.row.extra_matcher_trajectories * config.trajectory_cost_seconds
        for choice in selected
    )
    trajectories = sum(choice.row.extra_matcher_trajectories for choice in selected)
    return SelectorV3Decision(
        state="selected", admitted_hypotheses=tuple(sorted(row.key for row in admitted)),
        selected_hypotheses=tuple(sorted(choice.row.key for choice in selected)),
        probe_hypotheses=probe_order,
        ambiguous_hypotheses=tuple(sorted(ambiguous)),
        rejected_hypotheses=rejected,
        trust_scales={choice.row.key: choice.trust_alpha for choice in selected},
        robust_utility_intervals=intervals,
        total_objective=float(selected_objective),
        total_weighted_harm=float(harm), total_compute_seconds=float(compute),
        total_matcher_trajectories=int(trajectories),
        stop_reason=(
            "safe_regions_selected_with_pending_local_decisions"
            if ambiguous or probe_order
            else "profile_constrained_regional_utility_optimum"
        ),
    )
