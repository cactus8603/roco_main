"""Action-path selector with separate physics, control, and delivery variables.

Selector v5 fixes two ambiguities in the v4 contract.

``theta`` denotes the latent degradation parameters that remain compatible
with the observations.  ``u`` denotes one *executed* repair control, including
its actual input-space strength.  ``beta`` is a final output-space trust blend
between native and the matcher output obtained after executing ``u``.  These
objects are deliberately distinct: running a matcher on a half-strength input
repair is not equivalent to halving the output of a full-strength repair.

Every selectable control therefore binds a repaired-input hash, a candidate
output hash, the complete latent parameter set it covers, immutable physical
and task supports, and a control-specific risk calibration.  The decision
returns the exact control key and both input strength and output trust.  It
never reads a corruption label or task truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Mapping, Sequence

import numpy as np

from .action_profiles import ACTION_PHYSICS_PROFILES
from .action_path_geometry import validate_action_path
from .contracts import ActionSpec
from .selector_v4 import FootprintEvidence


V5_STATES = frozenset({"native", "selected"})


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
class ActionInvariantBound:
    """One action-specific inequality on frozen physical evidence.

    A name in a registry is not evidence.  The bound must state a signed
    conservative margin and cover every still-identifiable latent parameter.
    Deterministic constraints (for example a JPEG DCT-cell inclusion) do not
    need a statistical effective sample size; statistical constraints do.
    """

    name: str
    margin_lower: float
    support_hash: str
    latent_parameter_keys: tuple[str, ...]
    deterministic: bool = False
    effective_n: float = 0.0
    method: str = "fixed_holdout"

    def __post_init__(self) -> None:
        if not self.name or not self.support_hash or not self.method:
            raise ValueError("invariant identity, support, and method are required")
        _finite(self.margin_lower, "invariant margin")
        _nonnegative(self.effective_n, "invariant effective n")
        if (not self.latent_parameter_keys
                or len(set(self.latent_parameter_keys))
                != len(self.latent_parameter_keys)):
            raise ValueError("invariant needs unique latent parameter keys")


@dataclass(frozen=True)
class ControlConditionalRiskBound:
    """Selection-aware harm envelope for one exact executed control path."""

    action_identity: str
    control_key: str
    control_policy_hash: str
    support_policy_hash: str
    calibration_data_hash: str
    calibration_version: str
    effective_n: float
    selection_aware: bool
    harm_linear_upper: float
    harm_quadratic_upper: float
    severe_linear_upper: float
    severe_quadratic_upper: float

    def __post_init__(self) -> None:
        values = (
            self.action_identity, self.control_key, self.control_policy_hash,
            self.support_policy_hash, self.calibration_data_hash,
            self.calibration_version,
        )
        if any(not value for value in values):
            raise ValueError("control-risk identity and provenance are required")
        for value, name in (
            (self.effective_n, "risk effective n"),
            (self.harm_linear_upper, "harm linear upper"),
            (self.harm_quadratic_upper, "harm quadratic upper"),
            (self.severe_linear_upper, "severe linear upper"),
            (self.severe_quadratic_upper, "severe quadratic upper"),
        ):
            _nonnegative(value, name)

    def harm_upper(self, beta: float) -> float:
        beta = _fraction(beta, "output trust beta")
        return float(
            beta * self.harm_linear_upper
            + 0.5 * beta * beta * self.harm_quadratic_upper
        )

    def severe_upper(self, beta: float) -> float:
        beta = _fraction(beta, "output trust beta")
        return float(
            beta * self.severe_linear_upper
            + 0.5 * beta * beta * self.severe_quadratic_upper
        )


@dataclass(frozen=True)
class ExecutedControlPath:
    """Bounds for one exact input intervention and its actual matcher rerun.

    ``input_strength`` is already executed before the matcher.  The task bound
    is then a bound along the output chord from native to that exact rerun;
    ``beta`` may shrink delivery but cannot masquerade as a different input
    intervention.
    """

    key: str
    input_strength: float
    input_path_id: str
    control_parameter_hash: str
    repaired_input_hash: str
    candidate_output_hash: str
    output_support_policy_hash: str
    input_bound_support_hash: str
    delivery_bound_support_hash: str
    latent_parameter_keys: tuple[str, ...]
    footprint: FootprintEvidence
    input_directional_gain_lower: float
    input_curvature_upper: float
    information_retention_lower: float
    delivery_directional_gain_lower: float
    delivery_curvature_upper: float
    output_direction_norm_upper: float
    output_path_remainder_quadratic_upper: float
    maximum_output_trust: float
    risk: ControlConditionalRiskBound
    control_selection_adjusted: bool
    expected_compute_seconds: float
    extra_matcher_trajectories: int
    disjoint_path_keys: tuple[str, ...] = ()
    overlap_path_keys: tuple[str, ...] = ()
    interaction_quadratic_upper: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identities = (
            self.key, self.input_path_id, self.control_parameter_hash,
            self.repaired_input_hash,
            self.candidate_output_hash, self.output_support_policy_hash,
            self.input_bound_support_hash, self.delivery_bound_support_hash,
        )
        if any(not value for value in identities):
            raise ValueError("executed control identity and hashes are required")
        strength = _fraction(self.input_strength, "input strength")
        if strength <= 0.0:
            raise ValueError("input strength must be positive")
        if (not self.latent_parameter_keys
                or len(set(self.latent_parameter_keys))
                != len(self.latent_parameter_keys)):
            raise ValueError("control needs unique latent parameter coverage")
        _finite(self.input_directional_gain_lower, "input directional gain")
        _finite(self.delivery_directional_gain_lower, "delivery directional gain")
        for value, name in (
            (self.input_curvature_upper, "input curvature"),
            (self.delivery_curvature_upper, "delivery curvature"),
            (self.output_direction_norm_upper, "output direction norm"),
            (self.output_path_remainder_quadratic_upper, "output path remainder"),
            (self.expected_compute_seconds, "compute seconds"),
        ):
            _nonnegative(value, name)
        _fraction(self.information_retention_lower, "information retention")
        maximum = _fraction(self.maximum_output_trust, "maximum output trust")
        if maximum <= 0.0:
            raise ValueError("maximum output trust must be positive")
        if self.extra_matcher_trajectories < 0:
            raise ValueError("matcher trajectory count must be nonnegative")
        if self.risk.control_key != self.key:
            raise ValueError("risk calibration does not match executed control")
        if self.risk.support_policy_hash != self.output_support_policy_hash:
            raise ValueError("risk calibration and output support policy differ")
        if self.input_bound_support_hash != self.footprint.physical_support_hash:
            raise ValueError(
                "input directional bound and physical support content differ"
            )
        if self.delivery_bound_support_hash != self.footprint.output_support_hash:
            raise ValueError(
                "delivery utility bound and output support content differ"
            )
        if (len(set(self.disjoint_path_keys)) != len(self.disjoint_path_keys)
                or len(set(self.overlap_path_keys)) != len(self.overlap_path_keys)):
            raise ValueError("interaction-relation path keys must be unique")
        if set(self.disjoint_path_keys) & set(self.overlap_path_keys):
            raise ValueError("a path cannot be both disjoint and overlapping")
        if any(not key for key in self.interaction_quadratic_upper):
            raise ValueError("interaction path keys must be nonempty")
        if set(self.interaction_quadratic_upper) != set(self.overlap_path_keys):
            raise ValueError("every declared overlap needs exactly one interaction bound")
        for value in self.interaction_quadratic_upper.values():
            _nonnegative(value, "interaction upper")

    @property
    def input_physical_gain_lower(self) -> float:
        strength = float(self.input_strength)
        return float(
            strength * self.input_directional_gain_lower
            - 0.5 * strength * strength * self.input_curvature_upper
        )

    @property
    def delivery_quadratic_penalty(self) -> float:
        return float(0.5 * (
            self.delivery_curvature_upper * self.output_direction_norm_upper ** 2
            + self.output_path_remainder_quadratic_upper
        ))

    def utility_lower(self, beta: float) -> float:
        beta = _fraction(beta, "output trust beta")
        if beta > self.maximum_output_trust:
            raise ValueError("output trust exceeds the certified chord")
        return float(
            beta * self.delivery_directional_gain_lower
            - beta * beta * self.delivery_quadratic_penalty
        )


@dataclass(frozen=True)
class ActionPathHypothesis:
    key: str
    action: ActionSpec
    region_key: str
    source_input_hash: str
    admission_support_hash: str
    physical_status: str
    latent_parameter_keys: tuple[str, ...]
    invariants: tuple[ActionInvariantBound, ...]
    confound_rejection_margins: Mapping[str, float]
    controls: tuple[ExecutedControlPath, ...]
    pixel_fraction: float
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if (not self.key or not self.region_key or not self.source_input_hash
                or not self.admission_support_hash):
            raise ValueError(
                "hypothesis identity, region, source, and admission support are required"
            )
        if self.action.operator_id not in ACTION_PHYSICS_PROFILES:
            raise ValueError("action lacks a physics profile")
        endpoint = self.action.hypothesized_degraded_endpoint
        if not self.key.startswith(f"{self.action.operator_id}@{endpoint}#"):
            raise ValueError("hypothesis key does not match action identity")
        if (not self.latent_parameter_keys
                or len(set(self.latent_parameter_keys))
                != len(self.latent_parameter_keys)):
            raise ValueError("hypothesis needs a nonempty latent parameter set")
        if not self.controls or len({row.key for row in self.controls}) != len(
            self.controls
        ):
            raise ValueError("hypothesis needs unique executed controls")
        if len({row.name for row in self.invariants}) != len(self.invariants):
            raise ValueError("action invariants must be unique")
        latent = set(self.latent_parameter_keys)
        for bound in self.invariants:
            if set(bound.latent_parameter_keys) != latent:
                raise ValueError("every invariant must cover the latent parameter set")
            if bound.support_hash != self.admission_support_hash:
                raise ValueError(
                    "every invariant must be recomputed on the admission support"
                )
        action_identity = f"{self.action.operator_id}@{endpoint}"
        for control in self.controls:
            validate_action_path(self.action.operator_id, control.input_path_id)
            if set(control.latent_parameter_keys) != latent:
                raise ValueError("every control must cover the latent parameter set")
            if control.footprint.source_input_hash != self.source_input_hash:
                raise ValueError("control footprint source changed")
            if (control.footprint.physical_support_hash
                    != self.admission_support_hash):
                raise ValueError(
                    "control physical support does not match hypothesis admission support"
                )
            if control.footprint.coordinate_frame != self.action.coordinate_frame:
                raise ValueError("control footprint coordinate frame changed")
            if control.risk.action_identity != action_identity:
                raise ValueError("control risk is not action/endpoint conditional")
        _fraction(self.pixel_fraction, "pixel fraction")
        if any(not key for key in self.confound_rejection_margins):
            raise ValueError("confound names must be nonempty")
        for value in self.confound_rejection_margins.values():
            _finite(value, "confound rejection margin")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("runtime hypothesis cannot read labels, truth, or outcomes")


@dataclass(frozen=True)
class SelectorV5Config:
    minimum_effective_samples: float = 20.0
    minimum_information_retention: float = 0.05
    minimum_output_trust: float = 0.01
    maximum_control_harm: float = 0.05
    severe_harm_threshold: float = 0.25
    maximum_total_weighted_harm: float = 0.05
    maximum_total_pixel_fraction: float = 1.0
    maximum_matcher_trajectories: int = 3
    maximum_compute_seconds: float = 30.0
    trajectory_cost_seconds: float = 0.0
    cost_penalty_per_second: float = 0.0
    maximum_exact_choices: int = 20

    def __post_init__(self) -> None:
        for value, name in (
            (self.minimum_information_retention, "minimum information retention"),
            (self.minimum_output_trust, "minimum output trust"),
            (self.maximum_total_pixel_fraction, "maximum pixel fraction"),
        ):
            _fraction(value, name)
        for value, name in (
            (self.minimum_effective_samples, "minimum effective samples"),
            (self.maximum_control_harm, "maximum control harm"),
            (self.severe_harm_threshold, "severe harm threshold"),
            (self.maximum_total_weighted_harm, "total weighted harm"),
            (self.maximum_compute_seconds, "maximum compute seconds"),
            (self.trajectory_cost_seconds, "trajectory cost"),
            (self.cost_penalty_per_second, "cost penalty"),
        ):
            _nonnegative(value, name)
        if self.maximum_control_harm > self.severe_harm_threshold:
            raise ValueError("control harm budget cannot exceed severe threshold")
        if self.maximum_matcher_trajectories < 0 or self.maximum_exact_choices < 1:
            raise ValueError("invalid exact-search budget")


@dataclass(frozen=True)
class _Choice:
    hypothesis: ActionPathHypothesis
    control: ExecutedControlPath
    beta: float
    utility_lower: float
    harm_upper: float
    severe_upper: float
    cost_seconds: float

    @property
    def path_key(self) -> str:
        return f"{self.hypothesis.key}/{self.control.key}"


@dataclass(frozen=True)
class SelectorV5Decision:
    state: str
    selected_hypotheses: tuple[str, ...]
    selected_controls: Mapping[str, str]
    input_strengths: Mapping[str, float]
    output_trusts: Mapping[str, float]
    certified_utility_lowers: Mapping[str, float]
    rejected_hypotheses: Mapping[str, tuple[str, ...]]
    rejected_controls: Mapping[str, tuple[str, ...]]
    total_certified_utility_lower: float
    total_interaction_upper: float
    total_weighted_harm: float
    total_compute_seconds: float
    total_matcher_trajectories: int
    stop_reason: str

    def __post_init__(self) -> None:
        if self.state not in V5_STATES or not self.stop_reason:
            raise ValueError("invalid Selector-v5 decision")
        if self.state == "selected" and not self.selected_hypotheses:
            raise ValueError("selected decision needs a hypothesis")
        if self.state == "native" and self.selected_hypotheses:
            raise ValueError("native decision cannot select actions")
        selected = set(self.selected_hypotheses)
        if any(set(values) != selected for values in (
            self.selected_controls, self.input_strengths, self.output_trusts,
            self.certified_utility_lowers,
        )):
            raise ValueError("selected decision maps must share hypothesis keys")


def _required_invariant_names(operator_id: str) -> set[str]:
    profile = ACTION_PHYSICS_PROFILES[operator_id]
    return set(profile.decision_witnesses) | {
        name for name in profile.closure_witnesses
        if name != "task_signed_utility"
    }


def _hypothesis_reasons(row: ActionPathHypothesis,
                        config: SelectorV5Config) -> list[str]:
    reasons: list[str] = []
    profile = ACTION_PHYSICS_PROFILES[row.action.operator_id]
    if row.physical_status != "supported":
        reasons.append(f"physical_{row.physical_status}")
    observed = {bound.name: bound for bound in row.invariants}
    for name in sorted(_required_invariant_names(row.action.operator_id)):
        bound = observed.get(name)
        if bound is None:
            reasons.append(f"missing_action_invariant:{name}")
        elif bound.margin_lower <= 0.0:
            reasons.append(f"action_invariant_not_positive:{name}")
        elif (not bound.deterministic
              and bound.effective_n < config.minimum_effective_samples):
            reasons.append(f"action_invariant_effective_n_insufficient:{name}")
    for name in profile.dangerous_confounds:
        if name not in row.confound_rejection_margins:
            reasons.append(f"missing_confound_veto:{name}")
        elif row.confound_rejection_margins[name] <= 0.0:
            reasons.append(f"confound_not_rejected:{name}")
    if len(row.controls) > 1 and any(
        not control.control_selection_adjusted for control in row.controls
    ):
        reasons.append("control_search_not_selection_adjusted")
    if row.pixel_fraction <= 0.0:
        reasons.append("empty_region")
    return reasons


def _quadratic_cap(linear: float, quadratic: float, budget: float) -> float:
    if quadratic == 0.0:
        return math.inf if linear == 0.0 else budget / linear
    discriminant = linear * linear + 2.0 * quadratic * budget
    return max(0.0, (-linear + math.sqrt(discriminant)) / quadratic)


def _control_choice(row: ActionPathHypothesis, control: ExecutedControlPath,
                    config: SelectorV5Config) -> tuple[_Choice | None, list[str]]:
    reasons: list[str] = []
    footprint = control.footprint
    if control.input_physical_gain_lower <= 0.0:
        reasons.append("executed_input_path_not_physically_positive")
    if control.information_retention_lower <= config.minimum_information_retention:
        reasons.append("task_relevant_information_insufficient")
    if footprint.changed_outside_declared_fraction > 0.0:
        reasons.append("action_changed_outside_declared_support")
    if footprint.output_outside_influence_fraction > 0.0:
        reasons.append("output_escaped_influence_support")
    if footprint.output_fraction <= 0.0:
        reasons.append("empty_output_support")
    if control.delivery_directional_gain_lower <= 0.0:
        reasons.append("exact_rerun_direction_not_positive")
    if not control.risk.selection_aware:
        reasons.append("control_risk_not_selection_aware")
    if control.risk.effective_n < config.minimum_effective_samples:
        reasons.append("control_risk_effective_n_insufficient")
    if reasons:
        return None, reasons
    maximum = min(
        control.maximum_output_trust,
        _quadratic_cap(
            control.risk.harm_linear_upper, control.risk.harm_quadratic_upper,
            config.maximum_control_harm,
        ),
        _quadratic_cap(
            control.risk.severe_linear_upper,
            control.risk.severe_quadratic_upper,
            config.severe_harm_threshold,
        ),
        1.0,
    )
    if maximum < config.minimum_output_trust:
        return None, ["certified_output_trust_below_minimum"]
    candidates = {float(config.minimum_output_trust), float(maximum)}
    penalty = control.delivery_quadratic_penalty
    if penalty > 0.0:
        candidates.add(float(control.delivery_directional_gain_lower / (2.0 * penalty)))
    candidates = {
        beta for beta in candidates
        if config.minimum_output_trust <= beta <= maximum and np.isfinite(beta)
    }
    scored = [(control.utility_lower(beta), -beta, beta) for beta in candidates]
    utility, _, beta = max(scored)
    if utility <= 0.0:
        return None, ["nonpositive_exact_control_utility_lower"]
    harm = control.risk.harm_upper(beta)
    severe = control.risk.severe_upper(beta)
    cost = control.expected_compute_seconds + (
        control.extra_matcher_trajectories * config.trajectory_cost_seconds
    )
    if utility * row.pixel_fraction - config.cost_penalty_per_second * cost <= 0.0:
        return None, ["nonpositive_cost_adjusted_control_utility"]
    return _Choice(
        hypothesis=row, control=control, beta=float(beta),
        utility_lower=float(utility), harm_upper=float(harm),
        severe_upper=float(severe), cost_seconds=float(cost),
    ), []


def _pair_interaction(left: _Choice, right: _Choice) -> float | None:
    # Two alternatives for the same region are never silently composed.  A
    # future ordered-composition object must make non-commutativity explicit.
    if left.hypothesis.region_key == right.hypothesis.region_key:
        return None
    left_disjoint = right.path_key in left.control.disjoint_path_keys
    right_disjoint = left.path_key in right.control.disjoint_path_keys
    if left_disjoint and right_disjoint:
        return 0.0
    left_overlap = right.path_key in left.control.overlap_path_keys
    right_overlap = left.path_key in right.control.overlap_path_keys
    # Unknown, asymmetric, or contradictory spatial relations cannot be
    # silently interpreted as zero matcher interaction.
    if not (left_overlap and right_overlap):
        return None
    left_value = left.control.interaction_quadratic_upper.get(right.path_key)
    right_value = right.control.interaction_quadratic_upper.get(left.path_key)
    if left_value is None or right_value is None:
        return None
    if not math.isclose(float(left_value), float(right_value),
                        rel_tol=1e-9, abs_tol=1e-12):
        return None
    return float(left_value) * left.beta * right.beta


def _best_subset(choices: Sequence[_Choice], config: SelectorV5Config
                 ) -> tuple[tuple[_Choice, ...], float, float]:
    if len(choices) > config.maximum_exact_choices:
        return (), 0.0, 0.0
    best: tuple[_Choice, ...] = ()
    best_interaction = 0.0
    best_utility = 0.0
    best_key = (0.0, 0.0, ())
    for size in range(1, len(choices) + 1):
        for subset in combinations(choices, size):
            if len({item.hypothesis.key for item in subset}) != len(subset):
                continue
            pair_values = [
                _pair_interaction(left, right)
                for left, right in combinations(subset, 2)
            ]
            if any(value is None for value in pair_values):
                continue
            interaction = float(sum(value for value in pair_values if value is not None))
            pixels = sum(item.hypothesis.pixel_fraction for item in subset)
            harm = sum(
                item.harm_upper * item.hypothesis.pixel_fraction for item in subset
            )
            trajectories = sum(
                item.control.extra_matcher_trajectories for item in subset
            )
            compute = sum(item.cost_seconds for item in subset)
            if (pixels > config.maximum_total_pixel_fraction
                    or harm > config.maximum_total_weighted_harm
                    or trajectories > config.maximum_matcher_trajectories
                    or compute > config.maximum_compute_seconds):
                continue
            utility = sum(
                item.utility_lower * item.hypothesis.pixel_fraction for item in subset
            ) - interaction
            objective = utility - config.cost_penalty_per_second * compute
            names = tuple(sorted(item.path_key for item in subset))
            key = (objective, utility, tuple(reversed(names)))
            if objective > 0.0 and key > best_key:
                best = tuple(subset)
                best_interaction = interaction
                best_utility = utility
                best_key = key
    return best, best_interaction, best_utility


def select_action_paths_v5(
    hypotheses: Sequence[ActionPathHypothesis],
    *,
    config: SelectorV5Config = SelectorV5Config(),
) -> SelectorV5Decision:
    """Select exact executed controls, not corruption labels or abstract actions."""
    rows = tuple(hypotheses)
    if len({row.key for row in rows}) != len(rows):
        raise ValueError("hypothesis keys must be unique")
    rejected_hypotheses: dict[str, tuple[str, ...]] = {}
    rejected_controls: dict[str, tuple[str, ...]] = {}
    choices: list[_Choice] = []
    for row in rows:
        reasons = _hypothesis_reasons(row, config)
        if reasons:
            rejected_hypotheses[row.key] = tuple(reasons)
            continue
        for control in row.controls:
            choice, control_reasons = _control_choice(row, control, config)
            path_key = f"{row.key}/{control.key}"
            if control_reasons:
                rejected_controls[path_key] = tuple(control_reasons)
            else:
                assert choice is not None
                choices.append(choice)
    selected, interaction, utility = _best_subset(choices, config)
    if not selected:
        return SelectorV5Decision(
            state="native", selected_hypotheses=(), selected_controls={},
            input_strengths={}, output_trusts={}, certified_utility_lowers={},
            rejected_hypotheses=rejected_hypotheses,
            rejected_controls=rejected_controls,
            total_certified_utility_lower=0.0, total_interaction_upper=0.0,
            total_weighted_harm=0.0, total_compute_seconds=0.0,
            total_matcher_trajectories=0,
            stop_reason=(
                "no_exact_control_passed_hard_gates" if not choices
                else "no_positive_budget_feasible_composition"
            ),
        )
    ordered = tuple(sorted(selected, key=lambda item: item.hypothesis.key))
    hypotheses_selected = tuple(item.hypothesis.key for item in ordered)
    harm = sum(item.harm_upper * item.hypothesis.pixel_fraction for item in ordered)
    compute = sum(item.cost_seconds for item in ordered)
    trajectories = sum(item.control.extra_matcher_trajectories for item in ordered)
    return SelectorV5Decision(
        state="selected", selected_hypotheses=hypotheses_selected,
        selected_controls={
            item.hypothesis.key: item.control.key for item in ordered
        },
        input_strengths={
            item.hypothesis.key: item.control.input_strength for item in ordered
        },
        output_trusts={item.hypothesis.key: item.beta for item in ordered},
        certified_utility_lowers={
            item.hypothesis.key: item.utility_lower for item in ordered
        },
        rejected_hypotheses=rejected_hypotheses,
        rejected_controls=rejected_controls,
        total_certified_utility_lower=float(utility),
        total_interaction_upper=float(interaction),
        total_weighted_harm=float(harm), total_compute_seconds=float(compute),
        total_matcher_trajectories=int(trajectories),
        stop_reason="certified_exact_control_optimum",
    )
