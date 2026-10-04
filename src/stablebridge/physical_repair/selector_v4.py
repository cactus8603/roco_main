"""Action-system selector with certified intervention paths.

Selector v4 does not infer a renderer corruption label.  It accepts an action
only when the *executed direction* has positive held-out physical and task
lower bounds, its real footprint is contained, every declared confound is
rejected, and an action-conditional risk envelope permits a nonzero trust
radius.  Region composition explicitly charges measured interaction bounds;
native remains the zero-cost feasible decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Mapping, Sequence

import numpy as np

from .action_profiles import ACTION_PHYSICS_PROFILES
from .contracts import ActionSpec, COORDINATE_FRAMES


V4_STATES = frozenset({"native", "selected"})


def _finite(value: float, name: str) -> float:
    item = float(value)
    if not np.isfinite(item):
        raise ValueError(f"{name} must be finite")
    return item


def _nonnegative(value: float, name: str) -> float:
    item = _finite(value, name)
    if item < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return item


def _fraction(value: float, name: str) -> float:
    item = _finite(value, name)
    if not 0.0 <= item <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return item


@dataclass(frozen=True)
class FootprintEvidence:
    """Hashes and containment checks for the action's observed influence."""

    source_input_hash: str
    physical_support_hash: str
    declared_change_support_hash: str
    observed_change_support_hash: str
    influence_support_hash: str
    output_support_hash: str
    coordinate_frame: str
    changed_fraction: float
    influence_fraction: float
    output_fraction: float
    changed_outside_declared_fraction: float
    output_outside_influence_fraction: float
    halo_radius_px: float

    def __post_init__(self) -> None:
        hashes = (
            self.source_input_hash, self.physical_support_hash,
            self.declared_change_support_hash, self.observed_change_support_hash,
            self.influence_support_hash, self.output_support_hash,
        )
        if any(not value for value in hashes):
            raise ValueError("footprint hashes must be nonempty")
        if self.coordinate_frame not in COORDINATE_FRAMES:
            raise ValueError("invalid footprint coordinate frame")
        for value, name in (
            (self.changed_fraction, "changed fraction"),
            (self.influence_fraction, "influence fraction"),
            (self.output_fraction, "output fraction"),
            (self.changed_outside_declared_fraction, "changed outside declared"),
            (self.output_outside_influence_fraction, "output outside influence"),
        ):
            _fraction(value, name)
        _nonnegative(self.halo_radius_px, "halo radius")
        if self.changed_fraction > self.influence_fraction + 1e-12:
            raise ValueError("influence support cannot be smaller than changed support")
        if self.output_fraction > self.influence_fraction + 1e-12:
            raise ValueError("output support cannot exceed influence support")


@dataclass(frozen=True)
class DirectionalParameterBound:
    """Worst-case quadratic lower bounds along one real action path."""

    key: str
    physical_support_hash: str
    task_support_hash: str
    physical_directional_gain_lower: float
    physical_curvature_upper: float
    task_directional_gain_lower: float
    task_curvature_upper: float
    action_norm_upper: float
    path_remainder_quadratic_upper: float
    maximum_strength: float = 1.0

    def __post_init__(self) -> None:
        if not self.key or not self.physical_support_hash or not self.task_support_hash:
            raise ValueError("directional bound identity and supports are required")
        _finite(self.physical_directional_gain_lower, "physical directional gain")
        _finite(self.task_directional_gain_lower, "task directional gain")
        for value, name in (
            (self.physical_curvature_upper, "physical curvature"),
            (self.task_curvature_upper, "task curvature"),
            (self.action_norm_upper, "action norm"),
            (self.path_remainder_quadratic_upper, "path remainder"),
        ):
            _nonnegative(value, name)
        _fraction(self.maximum_strength, "maximum strength")
        if self.maximum_strength <= 0.0:
            raise ValueError("maximum strength must be positive")

    @property
    def physical_quadratic_penalty(self) -> float:
        return float(0.5 * self.physical_curvature_upper * self.action_norm_upper ** 2)

    @property
    def task_quadratic_penalty(self) -> float:
        return float(0.5 * (
            self.task_curvature_upper * self.action_norm_upper ** 2
            + self.path_remainder_quadratic_upper
        ))

    def task_utility_lower(self, alpha: float) -> float:
        alpha = _fraction(alpha, "trust alpha")
        return float(
            alpha * self.task_directional_gain_lower
            - alpha * alpha * self.task_quadratic_penalty
        )


@dataclass(frozen=True)
class ActionConditionalRiskBound:
    """Selection-aware quadratic harm envelope for one action/endpoint."""

    action_identity: str
    calibration_support_hash: str
    calibration_version: str
    effective_n: float
    selection_aware: bool
    harm_linear_upper: float
    harm_quadratic_upper: float
    severe_linear_upper: float
    severe_quadratic_upper: float

    def __post_init__(self) -> None:
        if (not self.action_identity or not self.calibration_support_hash
                or not self.calibration_version):
            raise ValueError("risk calibration identity is required")
        for value, name in (
            (self.effective_n, "risk effective n"),
            (self.harm_linear_upper, "harm linear upper"),
            (self.harm_quadratic_upper, "harm quadratic upper"),
            (self.severe_linear_upper, "severe linear upper"),
            (self.severe_quadratic_upper, "severe quadratic upper"),
        ):
            _nonnegative(value, name)

    def harm_upper(self, alpha: float) -> float:
        alpha = _fraction(alpha, "trust alpha")
        return float(
            alpha * self.harm_linear_upper
            + 0.5 * alpha * alpha * self.harm_quadratic_upper
        )

    def severe_upper(self, alpha: float) -> float:
        alpha = _fraction(alpha, "trust alpha")
        return float(
            alpha * self.severe_linear_upper
            + 0.5 * alpha * alpha * self.severe_quadratic_upper
        )


@dataclass(frozen=True)
class EProcessEvidence:
    """Actual anytime-valid evidence for adaptive parameter/action probes."""

    attempts_seen: int = 1
    e_value: float = 1.0
    null_name: str = "fixed_once"
    support_hash: str = "fixed_once"
    increments_hash: str = "fixed_once"

    def __post_init__(self) -> None:
        if self.attempts_seen < 1:
            raise ValueError("attempts seen must be positive")
        _nonnegative(self.e_value, "e-value")
        if not self.null_name or not self.support_hash or not self.increments_hash:
            raise ValueError("e-process identity and provenance are required")


@dataclass(frozen=True)
class ActionSystemHypothesis:
    key: str
    action: ActionSpec
    region_key: str
    physical_status: str
    source_input_hash: str
    directional_bounds: tuple[DirectionalParameterBound, ...]
    footprint: FootprintEvidence
    information_retention_lower: float
    visibility_certified: bool
    collision_fraction: float
    collision_policy: str
    forward_closure_margin_lower: float
    inverse_closure_margin_lower: float
    confound_rejection_margins: Mapping[str, float]
    risk: ActionConditionalRiskBound
    pixel_fraction: float
    expected_compute_seconds: float
    extra_matcher_trajectories: int
    sequential: EProcessEvidence = EProcessEvidence()
    overlap_keys: tuple[str, ...] = ()
    interaction_quadratic_upper: Mapping[str, float] = field(default_factory=dict)
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if not self.key or not self.region_key or not self.source_input_hash:
            raise ValueError("hypothesis identity, region, and source hash are required")
        if self.action.operator_id not in ACTION_PHYSICS_PROFILES:
            raise ValueError("action lacks a physics profile")
        endpoint = self.action.hypothesized_degraded_endpoint
        if not self.key.startswith(f"{self.action.operator_id}@{endpoint}#"):
            raise ValueError("hypothesis key does not match action identity")
        if not self.directional_bounds:
            raise ValueError("action needs at least one directional parameter bound")
        keys = [bound.key for bound in self.directional_bounds]
        if len(set(keys)) != len(keys):
            raise ValueError("directional parameter keys must be unique")
        profile = ACTION_PHYSICS_PROFILES[self.action.operator_id]
        if profile.parameter_geometry == "fixed" and len(keys) != 1:
            raise ValueError("fixed actions need one directional bound")
        if self.source_input_hash != self.footprint.source_input_hash:
            raise ValueError("footprint source hash changed")
        if self.action.coordinate_frame != self.footprint.coordinate_frame:
            raise ValueError("action and footprint coordinates differ")
        for bound in self.directional_bounds:
            if bound.physical_support_hash != self.footprint.physical_support_hash:
                raise ValueError("physical directional support changed")
            if bound.task_support_hash != self.footprint.output_support_hash:
                raise ValueError("task directional support changed")
        expected_identity = f"{self.action.operator_id}@{endpoint}"
        if self.risk.action_identity != expected_identity:
            raise ValueError("risk calibration is not action/endpoint conditional")
        _fraction(self.information_retention_lower, "information retention")
        _fraction(self.collision_fraction, "collision fraction")
        _finite(self.forward_closure_margin_lower, "forward closure margin")
        _finite(self.inverse_closure_margin_lower, "inverse closure margin")
        _fraction(self.pixel_fraction, "pixel fraction")
        _nonnegative(self.expected_compute_seconds, "compute seconds")
        if self.extra_matcher_trajectories < 0:
            raise ValueError("matcher trajectory count must be nonnegative")
        if len(set(self.overlap_keys)) != len(self.overlap_keys):
            raise ValueError("overlap keys must be unique")
        if self.key in self.overlap_keys:
            raise ValueError("hypothesis cannot overlap itself")
        if any(not key for key in self.confound_rejection_margins):
            raise ValueError("confound names must be nonempty")
        for value in self.confound_rejection_margins.values():
            _finite(value, "confound rejection margin")
        for key, value in self.interaction_quadratic_upper.items():
            if not key:
                raise ValueError("interaction key must be nonempty")
            _nonnegative(value, "interaction upper")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("runtime hypothesis cannot read labels, truth, or outcomes")


@dataclass(frozen=True)
class SelectorV4Config:
    maximum_familywise_error_rate: float = 0.05
    minimum_effective_samples: float = 20.0
    minimum_information_retention: float = 0.05
    minimum_trust_alpha: float = 0.01
    maximum_action_harm: float = 0.05
    severe_harm_threshold: float = 0.25
    maximum_total_weighted_harm: float = 0.05
    maximum_total_pixel_fraction: float = 1.0
    maximum_matcher_trajectories: int = 3
    maximum_compute_seconds: float = 30.0
    trajectory_cost_seconds: float = 0.0
    cost_penalty_per_second: float = 0.0
    maximum_exact_hypotheses: int = 20

    def __post_init__(self) -> None:
        for value, name in (
            (self.maximum_familywise_error_rate, "familywise error"),
            (self.minimum_information_retention, "minimum information retention"),
            (self.minimum_trust_alpha, "minimum trust alpha"),
            (self.maximum_total_pixel_fraction, "maximum pixel fraction"),
        ):
            _fraction(value, name)
        for value, name in (
            (self.minimum_effective_samples, "minimum effective samples"),
            (self.maximum_action_harm, "maximum action harm"),
            (self.severe_harm_threshold, "severe harm threshold"),
            (self.maximum_total_weighted_harm, "total weighted harm"),
            (self.maximum_compute_seconds, "maximum compute seconds"),
            (self.trajectory_cost_seconds, "trajectory cost"),
            (self.cost_penalty_per_second, "cost penalty"),
        ):
            _nonnegative(value, name)
        if self.maximum_matcher_trajectories < 0 or self.maximum_exact_hypotheses < 1:
            raise ValueError("invalid search budget")
        if self.maximum_action_harm > self.severe_harm_threshold:
            raise ValueError("action harm budget cannot exceed severe threshold")


@dataclass(frozen=True)
class _Choice:
    row: ActionSystemHypothesis
    alpha: float
    utility_lower: float
    harm_upper: float
    severe_upper: float
    cost_seconds: float


@dataclass(frozen=True)
class SelectorV4Decision:
    state: str
    selected_hypotheses: tuple[str, ...]
    rejected_hypotheses: Mapping[str, tuple[str, ...]]
    trust_scales: Mapping[str, float]
    certified_utility_lowers: Mapping[str, float]
    total_certified_utility_lower: float
    total_interaction_upper: float
    total_weighted_harm: float
    total_compute_seconds: float
    total_matcher_trajectories: int
    stop_reason: str

    def __post_init__(self) -> None:
        if self.state not in V4_STATES or not self.stop_reason:
            raise ValueError("invalid Selector-v4 decision")
        if self.state == "selected" and not self.selected_hypotheses:
            raise ValueError("selected decision needs a hypothesis")
        if self.state == "native" and self.selected_hypotheses:
            raise ValueError("native decision cannot select actions")
        _finite(self.total_certified_utility_lower, "total utility lower")
        for value, name in (
            (self.total_interaction_upper, "interaction upper"),
            (self.total_weighted_harm, "weighted harm"),
            (self.total_compute_seconds, "compute seconds"),
        ):
            _nonnegative(value, name)
        if self.total_matcher_trajectories < 0:
            raise ValueError("negative matcher trajectories")


def _physical_reasons(row: ActionSystemHypothesis,
                      config: SelectorV4Config) -> list[str]:
    reasons: list[str] = []
    profile = ACTION_PHYSICS_PROFILES[row.action.operator_id]
    if row.physical_status != "supported":
        reasons.append(f"physical_{row.physical_status}")
    if any(bound.physical_directional_gain_lower <= 0.0
           for bound in row.directional_bounds):
        reasons.append("physical_action_direction_not_positive")
    if any(bound.task_directional_gain_lower <= 0.0
           for bound in row.directional_bounds):
        reasons.append("task_action_direction_not_positive")
    if row.forward_closure_margin_lower <= 0.0:
        reasons.append("forward_closure_not_positive")
    if profile.reversibility == "conditional" and row.inverse_closure_margin_lower <= 0.0:
        reasons.append("conditional_inverse_closure_not_positive")
    if row.information_retention_lower <= config.minimum_information_retention:
        reasons.append("task_relevant_information_insufficient")
    if not row.visibility_certified:
        reasons.append("visibility_uncertified")
    if row.collision_fraction > 0.0 and row.collision_policy == "unresolved":
        reasons.append("collision_unresolved")
    if row.footprint.changed_outside_declared_fraction > 0.0:
        reasons.append("action_changed_outside_declared_support")
    if row.footprint.output_outside_influence_fraction > 0.0:
        reasons.append("output_escaped_influence_support")
    if row.footprint.output_fraction <= 0.0 or row.pixel_fraction <= 0.0:
        reasons.append("empty_output_support")
    missing_confounds = sorted(
        set(profile.dangerous_confounds) - set(row.confound_rejection_margins)
    )
    reasons.extend(f"missing_confound_veto:{name}" for name in missing_confounds)
    reasons.extend(
        f"confound_not_rejected:{name}"
        for name in profile.dangerous_confounds
        if row.confound_rejection_margins.get(name, 0.0) <= 0.0
    )
    if not row.risk.selection_aware:
        reasons.append("risk_not_selection_aware")
    if row.risk.effective_n < config.minimum_effective_samples:
        reasons.append("risk_effective_n_insufficient")
    if row.sequential.attempts_seen > 1:
        threshold = 1.0 / config.maximum_familywise_error_rate
        if row.sequential.e_value < threshold:
            reasons.append("adaptive_probe_e_process_insufficient")
        if row.sequential.support_hash != row.footprint.physical_support_hash:
            reasons.append("adaptive_probe_support_changed")
    return reasons


def _quadratic_cap(linear: float, quadratic: float, budget: float) -> float:
    """Largest nonnegative alpha satisfying linear*a + .5*q*a^2 <= budget."""
    if budget < 0.0:
        return 0.0
    if quadratic == 0.0:
        return math.inf if linear == 0.0 else budget / linear
    discriminant = linear * linear + 2.0 * quadratic * budget
    return max(0.0, (-linear + math.sqrt(discriminant)) / quadratic)


def _choice(row: ActionSystemHypothesis,
            config: SelectorV4Config) -> tuple[_Choice | None, list[str]]:
    maximum = min(bound.maximum_strength for bound in row.directional_bounds)
    for bound in row.directional_bounds:
        penalty = bound.physical_quadratic_penalty
        if penalty > 0.0:
            # Physical descent must stay strictly positive along the delivered
            # path, not merely at its infinitesimal origin.
            root = bound.physical_directional_gain_lower / penalty
            maximum = min(maximum, float(np.nextafter(root, 0.0)))
    maximum = min(
        maximum,
        _quadratic_cap(
            row.risk.harm_linear_upper, row.risk.harm_quadratic_upper,
            config.maximum_action_harm,
        ),
        _quadratic_cap(
            row.risk.severe_linear_upper, row.risk.severe_quadratic_upper,
            config.severe_harm_threshold,
        ),
        1.0,
    )
    if maximum < config.minimum_trust_alpha:
        return None, ["certified_trust_radius_below_minimum"]
    candidates = {float(config.minimum_trust_alpha), float(maximum)}
    for bound in row.directional_bounds:
        penalty = bound.task_quadratic_penalty
        if penalty > 0.0:
            candidates.add(float(bound.task_directional_gain_lower / (2.0 * penalty)))
    for left, right in combinations(row.directional_bounds, 2):
        numerator = left.task_directional_gain_lower - right.task_directional_gain_lower
        denominator = left.task_quadratic_penalty - right.task_quadratic_penalty
        if abs(denominator) > 1e-15:
            candidates.add(float(numerator / denominator))
    candidates = {
        value for value in candidates
        if config.minimum_trust_alpha <= value <= maximum and np.isfinite(value)
    }
    if not candidates:
        return None, ["no_feasible_certified_trust_radius"]
    evaluated = []
    for alpha in candidates:
        lower = min(bound.task_utility_lower(alpha) for bound in row.directional_bounds)
        evaluated.append((lower, -alpha, alpha))
    utility, _, alpha = max(evaluated)
    if utility <= 0.0:
        return None, ["nonpositive_directional_utility_lower"]
    harm = row.risk.harm_upper(alpha)
    severe = row.risk.severe_upper(alpha)
    if harm > config.maximum_action_harm + 1e-12:
        return None, ["action_harm_budget_exceeded"]
    if severe > config.severe_harm_threshold + 1e-12:
        return None, ["severe_harm_guard"]
    cost = row.expected_compute_seconds + (
        row.extra_matcher_trajectories * config.trajectory_cost_seconds
    )
    if utility * row.pixel_fraction - config.cost_penalty_per_second * cost <= 0.0:
        return None, ["nonpositive_cost_adjusted_certified_utility"]
    return _Choice(
        row=row, alpha=float(alpha), utility_lower=float(utility),
        harm_upper=float(harm), severe_upper=float(severe), cost_seconds=float(cost),
    ), []


def _pair_interaction(left: _Choice, right: _Choice) -> float | None:
    overlaps = bool(
        left.row.region_key == right.row.region_key
        or right.row.key in left.row.overlap_keys
        or left.row.key in right.row.overlap_keys
    )
    if not overlaps:
        return 0.0
    left_value = left.row.interaction_quadratic_upper.get(right.row.key)
    right_value = right.row.interaction_quadratic_upper.get(left.row.key)
    if left_value is None or right_value is None:
        return None
    if not math.isclose(float(left_value), float(right_value), rel_tol=1e-9, abs_tol=1e-12):
        return None
    return float(left_value) * left.alpha * right.alpha


def _best_subset(choices: Sequence[_Choice], config: SelectorV4Config
                 ) -> tuple[tuple[_Choice, ...], float, float]:
    if len(choices) > config.maximum_exact_hypotheses:
        return (), 0.0, 0.0
    best: tuple[_Choice, ...] = ()
    best_interaction = 0.0
    best_utility = 0.0
    best_key = (0.0, 0.0, ())
    for size in range(1, len(choices) + 1):
        for subset in combinations(choices, size):
            pair_values = [_pair_interaction(left, right)
                           for left, right in combinations(subset, 2)]
            if any(value is None for value in pair_values):
                continue
            interaction = float(sum(value for value in pair_values if value is not None))
            pixels = sum(item.row.pixel_fraction for item in subset)
            harm = sum(
                item.harm_upper * item.row.pixel_fraction for item in subset
            )
            trajectories = sum(item.row.extra_matcher_trajectories for item in subset)
            compute = sum(item.cost_seconds for item in subset)
            if (pixels > config.maximum_total_pixel_fraction
                    or harm > config.maximum_total_weighted_harm
                    or trajectories > config.maximum_matcher_trajectories
                    or compute > config.maximum_compute_seconds):
                continue
            utility = sum(
                item.utility_lower * item.row.pixel_fraction for item in subset
            ) - interaction
            objective = utility - config.cost_penalty_per_second * compute
            names = tuple(sorted(item.row.key for item in subset))
            key = (objective, utility, tuple(reversed(names)))
            if objective > 0.0 and key > best_key:
                best = tuple(subset)
                best_interaction = interaction
                best_utility = utility
                best_key = key
    return best, best_interaction, best_utility


def select_action_systems_v4(
    hypotheses: Sequence[ActionSystemHypothesis],
    *,
    config: SelectorV4Config = SelectorV4Config(),
) -> SelectorV4Decision:
    """Choose certified action paths under footprint, harm, and interaction bounds."""
    rows = tuple(hypotheses)
    if len({row.key for row in rows}) != len(rows):
        raise ValueError("hypothesis keys must be unique")
    rejected: dict[str, tuple[str, ...]] = {}
    choices = []
    for row in rows:
        reasons = _physical_reasons(row, config)
        if reasons:
            rejected[row.key] = tuple(reasons)
            continue
        choice, reasons = _choice(row, config)
        if reasons:
            rejected[row.key] = tuple(reasons)
        else:
            assert choice is not None
            choices.append(choice)
    selected, interaction, utility = _best_subset(choices, config)
    if not selected:
        return SelectorV4Decision(
            state="native", selected_hypotheses=(), rejected_hypotheses=rejected,
            trust_scales={}, certified_utility_lowers={},
            total_certified_utility_lower=0.0, total_interaction_upper=0.0,
            total_weighted_harm=0.0, total_compute_seconds=0.0,
            total_matcher_trajectories=0,
            stop_reason=(
                "no_action_system_passed_hard_gates" if not choices
                else "no_positive_budget_feasible_composition"
            ),
        )
    harm = sum(item.harm_upper * item.row.pixel_fraction for item in selected)
    compute = sum(item.cost_seconds for item in selected)
    trajectories = sum(item.row.extra_matcher_trajectories for item in selected)
    return SelectorV4Decision(
        state="selected",
        selected_hypotheses=tuple(sorted(item.row.key for item in selected)),
        rejected_hypotheses=rejected,
        trust_scales={item.row.key: item.alpha for item in selected},
        certified_utility_lowers={
            item.row.key: item.utility_lower for item in selected
        },
        total_certified_utility_lower=float(utility),
        total_interaction_upper=float(interaction),
        total_weighted_harm=float(harm), total_compute_seconds=float(compute),
        total_matcher_trajectories=int(trajectories),
        stop_reason="certified_action_system_optimum",
    )
