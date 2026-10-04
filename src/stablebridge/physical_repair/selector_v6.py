"""Metric-aligned exact-control selector with typed physical receipts.

Selector v6 consumes the evidence contracts in ``selector_v6_evidence``.  It
does not classify a corruption family and it does not manufacture utility from
uncertainty.  A selectable control must have:

* every action-specific law and confound veto on the same admission support;
* a complete latent-set receipt and an anytime-valid/familywise probe receipt;
* concrete physical/evidence/visibility/response/risk/utility support maps;
* an exact actual-rerun EPE path bound on the certified delivery support; and
* selection-aware EPE tail calibration at the exact delivered beta.

Native is always the zero-cost feasible decision.  Small calibrated harm is
allowed within explicit budgets; severe-tail probability and CVaR are hard
vetoes.  Same-region controls remain alternatives.  Cross-region composition
requires symmetric disjoint proof or a symmetric measured interaction bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Mapping, Sequence

import numpy as np

from .action_path_geometry import validate_action_path
from .action_profiles import ACTION_PHYSICS_PROFILES
from .contracts import ActionSpec
from .epe_verifier_task_bounds import (
    EPE_METRIC_ID,
    IndependentVerifierEPEPathBound,
)
from .selector_v6_evidence import (
    ActionSupportDecomposition,
    SequentialProbeReceipt,
    TypedPhysicalLawReceipt,
)


V6_STATES = frozenset({"native", "selected"})

# These witnesses compare an already physically supported action with another
# family/endpoint hypothesis.  They may prune or rank competitors, but their
# absence must not veto an action that independently passes its own-null laws,
# exact-control signed utility, and tail-risk gates.  Requiring them here would
# reintroduce a corruption-family/endpoint classifier through the back door.
OPTIONAL_DOMINANCE_LAWS_V6: Mapping[str, frozenset[str]] = {
    "common_disk": frozenset({"same_parameter_endpoint"}),
    "common_gaussian": frozenset({"same_parameter_endpoint"}),
    "common_motion": frozenset({
        "directional_otf_competitor", "same_parameter_endpoint",
    }),
}

# Motion still needs a directional forward-model own-null test.  The shared
# historical profile named only its pairwise competitor witness, so v6 adds the
# action-vs-identity OTF law explicitly instead of treating family recognition
# as admission evidence.
ADMISSION_LAW_ADDITIONS_V6: Mapping[str, frozenset[str]] = {
    "common_motion": frozenset({"otf_magnitude"}),
}

# Historical profiles called both evidence-invalidating nuisances and ordinary
# alternative mechanisms "dangerous confounds".  V6 requires only the former
# as admission vetoes.  A rival family may still be a useful action on the
# observed input and is handled by optional dominance plus signed task utility.
OPTIONAL_RIVAL_MECHANISMS_V6: Mapping[str, frozenset[str]] = {
    "common_disk": frozenset({"gaussian_blur", "motion_blur"}),
    "common_gaussian": frozenset({"additive_noise", "defocus_disk"}),
    "common_motion": frozenset({"defocus_disk"}),
}

# Warp/occlusion is spatially varying evidence about *where* an otherwise
# plausible blur action may be applied.  It is not a global alternative blur
# mechanism.  Blur own-null receipts already include a held-out translation
# nuisance law, while exact delivery separately requires visibility,
# collision, fold, uncovered and influence maps.  Requiring ``warp_error`` a
# second time as an action-admission confound collapses physical applicability
# into support localization.  Deferral never authorizes delivery: a control
# without the support-stage maps still has empty certified delivery support.
SUPPORT_STAGE_CONFOUNDS_V6: Mapping[str, frozenset[str]] = {
    "common_disk": frozenset({"warp_error"}),
    "common_gaussian": frozenset({"warp_error"}),
    "common_motion": frozenset({"warp_error"}),
}


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
class EPETailRiskPoint:
    """Selection-aware harm bounds at one exact delivered output trust."""

    beta: float
    mean_harm_upper_px: float
    cvar_harm_upper_px: float
    severe_probability_upper: float

    def __post_init__(self) -> None:
        beta = _fraction(self.beta, "tail-risk beta")
        if beta <= 0.0:
            raise ValueError("tail-risk beta must be positive")
        _nonnegative(self.mean_harm_upper_px, "mean harm upper")
        _nonnegative(self.cvar_harm_upper_px, "CVaR harm upper")
        _fraction(self.severe_probability_upper, "severe probability upper")


@dataclass(frozen=True)
class SelectionAwareEPETailRiskBound:
    """No-interpolation tail receipt for one action/control/support policy."""

    action_identity: str
    hypothesis_key: str
    control_key: str
    support_policy_hash: str
    support_content_hash: str
    calibration_version: str
    calibration_data_hash: str
    calibration_policy_hash: str
    selection_family_hash: str
    covered_path_keys: tuple[str, ...]
    selection_method: str
    familywise_error_rate: float
    e_value: float
    effective_n: float
    confidence_level: float
    severe_threshold_px: float
    cvar_level: float
    frozen_before_selection: bool
    points: tuple[EPETailRiskPoint, ...]
    metric_id: str = EPE_METRIC_ID

    def __post_init__(self) -> None:
        identities = (
            self.action_identity, self.hypothesis_key, self.control_key,
            self.support_policy_hash,
            self.support_content_hash, self.calibration_version,
            self.calibration_data_hash, self.calibration_policy_hash,
            self.selection_family_hash, self.selection_method,
            self.metric_id,
        )
        if any(not value for value in identities):
            raise ValueError("tail-risk identity and provenance are required")
        if _nonnegative(self.effective_n, "tail-risk effective n") <= 0.0:
            raise ValueError("tail-risk effective n must be positive")
        confidence = _fraction(self.confidence_level, "tail-risk confidence")
        if not 0.0 < confidence < 1.0:
            raise ValueError("tail-risk confidence must lie in (0,1)")
        if _nonnegative(self.severe_threshold_px, "severe threshold") <= 0.0:
            raise ValueError("severe threshold must be positive")
        cvar_level = _fraction(self.cvar_level, "CVaR level")
        if not 0.0 < cvar_level < 1.0:
            raise ValueError("CVaR level must lie in (0,1)")
        if self.metric_id != EPE_METRIC_ID:
            raise ValueError("tail risk must use the final EPE metric")
        if (not self.covered_path_keys
                or len(set(self.covered_path_keys)) != len(self.covered_path_keys)):
            raise ValueError("tail calibration needs unique covered path keys")
        if self.selection_method not in {
            "fixed_single", "simultaneous_bound", "e_process",
        }:
            raise ValueError("unknown tail selection correction")
        alpha = _fraction(self.familywise_error_rate, "tail familywise error rate")
        if not 0.0 < alpha < 1.0:
            raise ValueError("tail familywise error rate must lie in (0,1)")
        _nonnegative(self.e_value, "tail e-value")
        if len(self.covered_path_keys) > 1 and self.selection_method == "fixed_single":
            raise ValueError("multi-control tail calibration lacks selection correction")
        if self.selection_method == "e_process" and self.e_value < 1.0 / alpha:
            raise ValueError("tail e-process did not cross its evidence threshold")
        own_path = f"{self.hypothesis_key}/{self.control_key}"
        if own_path not in self.covered_path_keys:
            raise ValueError("tail calibration does not cover its own exact path")
        if not self.points:
            raise ValueError("tail risk needs at least one exact beta point")
        betas = tuple(point.beta for point in self.points)
        if len(set(betas)) != len(betas):
            raise ValueError("tail-risk beta points must be unique")

    def point(self, beta: float) -> EPETailRiskPoint:
        value = _finite(beta, "output trust beta")
        matches = [
            point for point in self.points
            if math.isclose(point.beta, value, rel_tol=0.0, abs_tol=1e-12)
        ]
        if len(matches) != 1:
            raise ValueError("no exact tail calibration exists at this beta")
        return matches[0]


@dataclass(frozen=True)
class ExecutedControlV6:
    """One exact input repair and its exact actual matcher rerun."""

    key: str
    input_strength: float
    input_path_id: str
    control_parameter_hash: str
    repaired_input_hash: str
    native_output_hash: str
    candidate_output_hash: str
    physical_bound_support_hash: str
    latent_set_hash: str
    input_directional_gain_lower: float
    input_curvature_upper: float
    information_retention_lower: float
    support: ActionSupportDecomposition
    task_bound: IndependentVerifierEPEPathBound
    tail_risk: SelectionAwareEPETailRiskBound
    sequential: SequentialProbeReceipt
    expected_compute_seconds: float
    extra_matcher_trajectories: int
    disjoint_path_keys: tuple[str, ...] = ()
    overlap_path_keys: tuple[str, ...] = ()
    interaction_epe_upper_px: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identities = (
            self.key, self.input_path_id, self.control_parameter_hash,
            self.repaired_input_hash, self.native_output_hash,
            self.candidate_output_hash, self.physical_bound_support_hash,
            self.latent_set_hash,
        )
        if any(not value for value in identities):
            raise ValueError("executed v6 control identity is incomplete")
        strength = _fraction(self.input_strength, "input strength")
        if strength <= 0.0:
            raise ValueError("input strength must be positive")
        _finite(self.input_directional_gain_lower, "input directional gain")
        _nonnegative(self.input_curvature_upper, "input curvature")
        _fraction(self.information_retention_lower, "information retention")
        _nonnegative(self.expected_compute_seconds, "compute seconds")
        if self.extra_matcher_trajectories < 0:
            raise ValueError("matcher trajectory count must be nonnegative")
        if self.task_bound.control_key != self.key:
            raise ValueError("EPE task bound belongs to another exact control")
        if self.task_bound.native_output_hash != self.native_output_hash:
            raise ValueError("EPE task bound native output changed")
        if self.task_bound.candidate_output_hash != self.candidate_output_hash:
            raise ValueError("EPE task bound candidate output changed")
        if self.task_bound.source_input_hash != self.support.source_input_hash:
            raise ValueError("EPE verifier and regional supports use different sources")
        if self.task_bound.task_support_hash != self.support.delivery_support_hash:
            raise ValueError("EPE task bound and delivery support differ")
        if self.task_bound.task_support_policy_hash != self.support.support_policy_hash:
            raise ValueError("EPE task bound and support policies differ")
        if self.tail_risk.control_key != self.key:
            raise ValueError("tail calibration belongs to another exact control")
        if self.tail_risk.support_content_hash != self.support.delivery_support_hash:
            raise ValueError("tail calibration and delivery support differ")
        if self.tail_risk.support_policy_hash != self.support.support_policy_hash:
            raise ValueError("tail calibration and support policies differ")
        if self.sequential.selected_control_key != self.key:
            raise ValueError("probe receipt selected another exact control")
        if self.sequential.support_policy_hash != self.support.support_policy_hash:
            raise ValueError("probe receipt and support policies differ")
        if self.sequential.support_content_hash != self.support.candidate_support_hash:
            raise ValueError("probe receipt and candidate support differ")
        if (len(set(self.disjoint_path_keys)) != len(self.disjoint_path_keys)
                or len(set(self.overlap_path_keys)) != len(self.overlap_path_keys)):
            raise ValueError("interaction path keys must be unique")
        if set(self.disjoint_path_keys) & set(self.overlap_path_keys):
            raise ValueError("a path cannot be both disjoint and overlapping")
        if set(self.interaction_epe_upper_px) != set(self.overlap_path_keys):
            raise ValueError("every overlap needs one EPE interaction bound")
        for value in self.interaction_epe_upper_px.values():
            _nonnegative(value, "EPE interaction upper")

    @property
    def input_physical_gain_lower(self) -> float:
        strength = float(self.input_strength)
        return float(
            strength * self.input_directional_gain_lower
            - 0.5 * strength * strength * self.input_curvature_upper
        )


@dataclass(frozen=True)
class ActionHypothesisV6:
    key: str
    action: ActionSpec
    region_key: str
    source_input_hash: str
    admission_support_hash: str
    latent_set_hash: str
    latent_parameter_keys: tuple[str, ...]
    physical_status: str
    law_receipts: tuple[TypedPhysicalLawReceipt, ...]
    controls: tuple[ExecutedControlV6, ...]
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        identities = (
            self.key, self.region_key, self.source_input_hash,
            self.admission_support_hash, self.latent_set_hash,
        )
        if any(not value for value in identities):
            raise ValueError("v6 hypothesis identity is incomplete")
        if self.action.operator_id not in ACTION_PHYSICS_PROFILES:
            raise ValueError("action lacks a physics profile")
        endpoint = self.action.hypothesized_degraded_endpoint
        action_identity = f"{self.action.operator_id}@{endpoint}"
        if not self.key.startswith(f"{action_identity}#"):
            raise ValueError("hypothesis key does not match action identity")
        if (not self.latent_parameter_keys
                or len(set(self.latent_parameter_keys))
                != len(self.latent_parameter_keys)):
            raise ValueError("hypothesis needs unique latent parameter keys")
        if not self.law_receipts:
            raise ValueError("hypothesis needs typed physical laws")
        law_keys = tuple((row.role, row.law_id) for row in self.law_receipts)
        if len(set(law_keys)) != len(law_keys):
            raise ValueError("typed physical laws must be unique by role and id")
        for receipt in self.law_receipts:
            if receipt.action_identity != action_identity:
                raise ValueError("physical law belongs to another action/endpoint")
            if receipt.certificate_support_hash != self.admission_support_hash:
                raise ValueError("physical law and admission supports differ")
            if receipt.latent_set_hash != self.latent_set_hash:
                raise ValueError("physical law and hypothesis latent sets differ")
            if set(receipt.latent_parameter_keys) != set(self.latent_parameter_keys):
                raise ValueError("physical law latent parameters are incomplete")
        if not self.controls or len({row.key for row in self.controls}) != len(
            self.controls
        ):
            raise ValueError("hypothesis needs unique exact controls")
        for control in self.controls:
            validate_action_path(self.action.operator_id, control.input_path_id)
            if control.latent_set_hash != self.latent_set_hash:
                raise ValueError("control and hypothesis latent sets differ")
            if control.physical_bound_support_hash != self.admission_support_hash:
                raise ValueError("control physical bound and admission supports differ")
            if control.support.source_input_hash != self.source_input_hash:
                raise ValueError("control regional evidence source changed")
            if control.task_bound.action_key != self.key:
                raise ValueError("EPE task bound belongs to another hypothesis")
            if control.tail_risk.hypothesis_key != self.key:
                raise ValueError("tail calibration belongs to another hypothesis")
            if control.tail_risk.action_identity != action_identity:
                raise ValueError("tail calibration belongs to another action/endpoint")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("runtime hypothesis cannot read labels, truth, or outcomes")


@dataclass(frozen=True)
class SelectorV6Config:
    minimum_effective_samples: float = 20.0
    minimum_information_retention: float = 0.05
    maximum_control_mean_harm_px: float = 0.05
    maximum_severe_probability: float = 0.05
    severe_harm_threshold_px: float = 0.25
    maximum_cvar_harm_px: float = 0.25
    required_cvar_level: float = 0.95
    minimum_tail_confidence: float = 0.95
    maximum_total_weighted_harm_px: float = 0.05
    maximum_total_delivery_fraction: float = 1.0
    maximum_matcher_trajectories: int = 3
    maximum_compute_seconds: float = 30.0
    trajectory_cost_seconds: float = 0.0
    cost_penalty_px_per_second: float = 0.0
    maximum_exact_choices: int = 20

    def __post_init__(self) -> None:
        _nonnegative(self.minimum_effective_samples, "minimum effective samples")
        _fraction(self.minimum_information_retention, "minimum information retention")
        _nonnegative(self.maximum_control_mean_harm_px, "control mean harm")
        _fraction(self.maximum_severe_probability, "maximum severe probability")
        if _nonnegative(self.severe_harm_threshold_px, "severe threshold") <= 0.0:
            raise ValueError("severe threshold must be positive")
        _nonnegative(self.maximum_cvar_harm_px, "maximum CVaR harm")
        cvar_level = _fraction(self.required_cvar_level, "required CVaR level")
        if not 0.0 < cvar_level < 1.0:
            raise ValueError("required CVaR level must lie in (0,1)")
        tail_confidence = _fraction(
            self.minimum_tail_confidence, "minimum tail confidence",
        )
        if not 0.0 < tail_confidence < 1.0:
            raise ValueError("minimum tail confidence must lie in (0,1)")
        _nonnegative(self.maximum_total_weighted_harm_px, "total weighted harm")
        _fraction(self.maximum_total_delivery_fraction, "delivery fraction")
        _nonnegative(self.maximum_compute_seconds, "compute seconds")
        _nonnegative(self.trajectory_cost_seconds, "trajectory cost")
        _nonnegative(self.cost_penalty_px_per_second, "cost penalty")
        if self.maximum_matcher_trajectories < 0 or self.maximum_exact_choices < 1:
            raise ValueError("invalid exact-search budget")


@dataclass(frozen=True)
class CertifiedControlOptionV6:
    """One fully verified exact control/trust option.

    This is the public bridge for a bounded high-level planner.  It exposes
    only options that already passed the same physical, support, metric and
    selection-aware tail gates used by :func:`select_action_paths_v6`.
    ``utility_upper_px`` is the matching triangle-inequality upper endpoint;
    it is used only to decide whether another paid probe could resolve an
    ambiguity, never to authorize delivery.
    """

    hypothesis: ActionHypothesisV6
    control: ExecutedControlV6
    beta: float
    utility_lower_px: float
    utility_upper_px: float
    mean_harm_upper_px: float
    cvar_harm_upper_px: float
    severe_probability_upper: float
    cost_seconds: float

    @property
    def path_key(self) -> str:
        return f"{self.hypothesis.key}/{self.control.key}"

    @property
    def delivery_fraction(self) -> float:
        return self.control.support.delivery_fraction


@dataclass(frozen=True)
class SelectorV6Decision:
    state: str
    selected_hypotheses: tuple[str, ...]
    selected_controls: Mapping[str, str]
    input_strengths: Mapping[str, float]
    output_trusts: Mapping[str, float]
    certified_epe_gain_lowers: Mapping[str, float]
    rejected_hypotheses: Mapping[str, tuple[str, ...]]
    rejected_controls: Mapping[str, tuple[str, ...]]
    total_certified_epe_gain_lower: float
    total_interaction_upper_px: float
    total_weighted_mean_harm_upper_px: float
    total_compute_seconds: float
    total_matcher_trajectories: int
    stop_reason: str

    def __post_init__(self) -> None:
        if self.state not in V6_STATES or not self.stop_reason:
            raise ValueError("invalid Selector-v6 decision")
        if self.state == "selected" and not self.selected_hypotheses:
            raise ValueError("selected decision needs a hypothesis")
        if self.state == "native" and self.selected_hypotheses:
            raise ValueError("native decision cannot contain actions")
        selected = set(self.selected_hypotheses)
        maps = (
            self.selected_controls, self.input_strengths, self.output_trusts,
            self.certified_epe_gain_lowers,
        )
        if any(set(value) != selected for value in maps):
            raise ValueError("selected decision maps must share hypothesis keys")


def _required_laws(operator_id: str) -> tuple[set[str], set[str]]:
    profile = ACTION_PHYSICS_PROFILES[operator_id]
    laws = (
        set(profile.decision_witnesses)
        - set(OPTIONAL_DOMINANCE_LAWS_V6.get(operator_id, ()))
        | set(ADMISSION_LAW_ADDITIONS_V6.get(operator_id, ()))
    ) | {
        name for name in profile.closure_witnesses
        if name != "task_signed_utility"
    }
    confounds = (
        set(profile.dangerous_confounds)
        - set(OPTIONAL_RIVAL_MECHANISMS_V6.get(operator_id, ()))
        - set(SUPPORT_STAGE_CONFOUNDS_V6.get(operator_id, ()))
    )
    return laws, confounds


def physical_receipt_set_rejection_reasons_v6(
    *,
    operator_id: str,
    action_identity: str,
    admission_support_hash: str,
    latent_set_hash: str,
    latent_parameter_keys: Sequence[str],
    physical_status: str,
    law_receipts: Sequence[TypedPhysicalLawReceipt],
    config: SelectorV6Config = SelectorV6Config(),
) -> tuple[str, ...]:
    """Validate an outcome-blind physical prefix before a control is run.

    Real receipt materializers need to check the physical half of Selector-v6
    before native/candidate outputs, dense EPE bounds, or tail calibration are
    available.  This public function deliberately applies the exact same
    admission laws and confound vetoes as the full selector; it cannot create
    a selectable control or authorize delivery.

    Malformed or cross-bound receipt sets raise ``ValueError``.  A well-formed
    but scientifically insufficient set returns explicit fail-closed reasons.
    """

    if operator_id not in ACTION_PHYSICS_PROFILES:
        raise ValueError("physical receipt set names an unknown operator")
    if not action_identity or not admission_support_hash or not latent_set_hash:
        raise ValueError("physical receipt-set identity is incomplete")
    if not action_identity.startswith(f"{operator_id}@"):
        raise ValueError("physical receipt set action and operator differ")
    parameters = tuple(latent_parameter_keys)
    if not parameters or len(set(parameters)) != len(parameters):
        raise ValueError("physical receipt set needs unique latent parameters")
    receipts = tuple(law_receipts)
    if not receipts:
        raise ValueError("physical receipt set is empty")
    identities = tuple((item.role, item.law_id) for item in receipts)
    if len(set(identities)) != len(identities):
        raise ValueError("physical receipt set has duplicate role/law identities")
    for receipt in receipts:
        if receipt.action_identity != action_identity:
            raise ValueError("physical receipt belongs to another action/endpoint")
        if receipt.certificate_support_hash != admission_support_hash:
            raise ValueError("physical receipt and admission supports differ")
        if receipt.latent_set_hash != latent_set_hash:
            raise ValueError("physical receipt and latent sets differ")
        if set(receipt.latent_parameter_keys) != set(parameters):
            raise ValueError("physical receipt latent parameters are incomplete")

    reasons: list[str] = []
    if physical_status != "supported":
        reasons.append(f"physical_{physical_status}")
    required, confounds = _required_laws(operator_id)
    observed_laws = {
        receipt.law_id: receipt for receipt in receipts
        if receipt.role in {"invariant", "closure"}
    }
    observed_confounds = {
        receipt.law_id: receipt for receipt in receipts
        if receipt.role == "confound_veto"
    }
    for law_id in sorted(required):
        receipt = observed_laws.get(law_id)
        if receipt is None:
            reasons.append(f"missing_typed_law:{law_id}")
        elif not receipt.passed:
            reasons.append(f"typed_law_failed:{law_id}")
        elif (not receipt.deterministic
              and receipt.effective_n < config.minimum_effective_samples):
            reasons.append(f"typed_law_effective_n_insufficient:{law_id}")
    for confound in sorted(confounds):
        receipt = observed_confounds.get(confound)
        if receipt is None:
            reasons.append(f"missing_confound_veto:{confound}")
        elif not receipt.passed:
            reasons.append(f"confound_not_rejected:{confound}")
    return tuple(reasons)


def _hypothesis_reasons(row: ActionHypothesisV6,
                        config: SelectorV6Config) -> list[str]:
    return list(physical_receipt_set_rejection_reasons_v6(
        operator_id=row.action.operator_id,
        action_identity=(
            f"{row.action.operator_id}@"
            f"{row.action.hypothesized_degraded_endpoint}"
        ),
        admission_support_hash=row.admission_support_hash,
        latent_set_hash=row.latent_set_hash,
        latent_parameter_keys=row.latent_parameter_keys,
        physical_status=row.physical_status,
        law_receipts=row.law_receipts,
        config=config,
    ))


def _control_choices(row: ActionHypothesisV6, control: ExecutedControlV6,
                     config: SelectorV6Config,
                     predeclared_path_keys: frozenset[str],
                     ) -> tuple[list[CertifiedControlOptionV6], list[str]]:
    reasons: list[str] = []
    if control.input_physical_gain_lower <= 0.0:
        reasons.append("executed_input_path_not_physically_positive")
    if control.information_retention_lower <= config.minimum_information_retention:
        reasons.append("task_relevant_information_insufficient")
    if control.support.delivery_fraction <= 0.0:
        reasons.append("empty_certified_delivery_support")
    if not control.tail_risk.frozen_before_selection:
        reasons.append("tail_calibration_not_frozen")
    if not predeclared_path_keys.issubset(control.tail_risk.covered_path_keys):
        reasons.append("tail_calibration_selection_family_incomplete")
    if control.tail_risk.effective_n < config.minimum_effective_samples:
        reasons.append("tail_calibration_effective_n_insufficient")
    if control.tail_risk.confidence_level < config.minimum_tail_confidence:
        reasons.append("tail_calibration_confidence_insufficient")
    if not math.isclose(
        control.tail_risk.severe_threshold_px,
        config.severe_harm_threshold_px,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        reasons.append("tail_severe_threshold_mismatch")
    if not math.isclose(
        control.tail_risk.cvar_level,
        config.required_cvar_level,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        reasons.append("tail_cvar_level_mismatch")
    if reasons:
        return [], reasons
    choices: list[CertifiedControlOptionV6] = []
    beta_reasons: list[str] = []
    for point in control.tail_risk.points:
        beta = point.beta
        if beta > control.task_bound.maximum_output_trust:
            beta_reasons.append(f"beta_outside_epe_path:{beta:g}")
            continue
        utility = control.task_bound.utility_lower(beta)
        if utility <= 0.0:
            beta_reasons.append(f"nonpositive_epe_utility:{beta:g}")
            continue
        mean_harm = max(
            control.task_bound.harm_budget_upper(beta),
            point.mean_harm_upper_px,
        )
        if mean_harm > config.maximum_control_mean_harm_px:
            beta_reasons.append(f"mean_harm_budget_exceeded:{beta:g}")
            continue
        if point.severe_probability_upper > config.maximum_severe_probability:
            beta_reasons.append(f"severe_probability_exceeded:{beta:g}")
            continue
        if point.cvar_harm_upper_px > config.maximum_cvar_harm_px:
            beta_reasons.append(f"cvar_harm_exceeded:{beta:g}")
            continue
        cost = control.expected_compute_seconds + (
            control.extra_matcher_trajectories * config.trajectory_cost_seconds
        )
        weighted_utility = utility * control.support.delivery_fraction
        if weighted_utility - config.cost_penalty_px_per_second * cost <= 0.0:
            beta_reasons.append(f"nonpositive_cost_adjusted_utility:{beta:g}")
            continue
        utility_upper = (
            control.task_bound.nominal_verifier_gain(beta)
            + 2.0 * control.task_bound.mean_epe_radius_upper
        )
        choices.append(CertifiedControlOptionV6(
            hypothesis=row,
            control=control,
            beta=beta,
            utility_lower_px=utility,
            utility_upper_px=utility_upper,
            mean_harm_upper_px=mean_harm,
            cvar_harm_upper_px=point.cvar_harm_upper_px,
            severe_probability_upper=point.severe_probability_upper,
            cost_seconds=cost,
        ))
    if not choices:
        reasons.extend(beta_reasons or ["no_exact_tail_calibration_point"])
    return choices, reasons


def _pair_interaction(
    left: CertifiedControlOptionV6,
    right: CertifiedControlOptionV6,
) -> float | None:
    if left.hypothesis.region_key == right.hypothesis.region_key:
        return None
    left_disjoint = right.path_key in left.control.disjoint_path_keys
    right_disjoint = left.path_key in right.control.disjoint_path_keys
    if left_disjoint and right_disjoint:
        return 0.0
    left_overlap = right.path_key in left.control.overlap_path_keys
    right_overlap = left.path_key in right.control.overlap_path_keys
    if not (left_overlap and right_overlap):
        return None
    left_value = left.control.interaction_epe_upper_px.get(right.path_key)
    right_value = right.control.interaction_epe_upper_px.get(left.path_key)
    if left_value is None or right_value is None:
        return None
    if not math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-12):
        return None
    return float(left_value) * left.beta * right.beta


def _has_unbounded_higher_order_overlap(
    subset: Sequence[CertifiedControlOptionV6],
) -> bool:
    """Reject overlap components larger than a calibrated pair.

    Pairwise interaction bounds do not in general upper-bound a three-way or
    higher-order composition term.  Until a typed joint-interaction receipt is
    available, mutually declared disjoint paths may compose freely, while each
    connected overlap component is limited to two exact controls.
    """
    adjacency: dict[str, set[str]] = {
        item.path_key: set() for item in subset
    }
    by_key = {item.path_key: item for item in subset}
    for left, right in combinations(subset, 2):
        if (right.path_key in left.control.overlap_path_keys
                and left.path_key in right.control.overlap_path_keys):
            adjacency[left.path_key].add(right.path_key)
            adjacency[right.path_key].add(left.path_key)
    remaining = set(by_key)
    while remaining:
        root = remaining.pop()
        component = {root}
        frontier = [root]
        while frontier:
            current = frontier.pop()
            unseen = adjacency[current] - component
            component.update(unseen)
            remaining.difference_update(unseen)
            frontier.extend(unseen)
        if len(component) > 2:
            return True
    return False


def _best_subset(
    choices: Sequence[CertifiedControlOptionV6],
    config: SelectorV6Config,
) -> tuple[tuple[CertifiedControlOptionV6, ...], float, float]:
    if len(choices) > config.maximum_exact_choices:
        return (), 0.0, 0.0
    best: tuple[CertifiedControlOptionV6, ...] = ()
    best_interaction = 0.0
    best_utility = 0.0
    best_key = (0.0, 0.0, ())
    for size in range(1, len(choices) + 1):
        for subset in combinations(choices, size):
            if len({item.hypothesis.key for item in subset}) != len(subset):
                continue
            if _has_unbounded_higher_order_overlap(subset):
                continue
            pair_values = [
                _pair_interaction(left, right)
                for left, right in combinations(subset, 2)
            ]
            if any(value is None for value in pair_values):
                continue
            interaction = float(sum(
                value for value in pair_values if value is not None
            ))
            delivery = sum(item.delivery_fraction for item in subset)
            harm = sum(
                item.mean_harm_upper_px * item.delivery_fraction
                for item in subset
            )
            trajectories = sum(
                item.control.extra_matcher_trajectories for item in subset
            )
            compute = sum(item.cost_seconds for item in subset)
            if (delivery > config.maximum_total_delivery_fraction
                    or harm > config.maximum_total_weighted_harm_px
                    or trajectories > config.maximum_matcher_trajectories
                    or compute > config.maximum_compute_seconds):
                continue
            utility = sum(
                item.utility_lower_px * item.delivery_fraction
                for item in subset
            ) - interaction
            objective = utility - config.cost_penalty_px_per_second * compute
            names = tuple(sorted(item.path_key for item in subset))
            key = (objective, utility, tuple(reversed(names)))
            if objective > 0.0 and key > best_key:
                best = tuple(subset)
                best_interaction = interaction
                best_utility = utility
                best_key = key
    return best, best_interaction, best_utility


@dataclass(frozen=True)
class CertifiedControlEnumerationV6:
    """All exact options admitted by the frozen Selector-v6 contracts."""

    options: tuple[CertifiedControlOptionV6, ...]
    rejected_hypotheses: Mapping[str, tuple[str, ...]]
    rejected_controls: Mapping[str, tuple[str, ...]]
    predeclared_path_keys: frozenset[str]

    def __post_init__(self) -> None:
        option_keys = tuple((item.path_key, item.beta) for item in self.options)
        if len(option_keys) != len(set(option_keys)):
            raise ValueError("certified control options are duplicated")
        if any(path not in self.predeclared_path_keys for path, _ in option_keys):
            raise ValueError("certified option lies outside the declared family")


def enumerate_certified_control_options_v6(
    hypotheses: Sequence[ActionHypothesisV6],
    *,
    config: SelectorV6Config = SelectorV6Config(),
    predeclared_path_keys: frozenset[str] | None = None,
) -> CertifiedControlEnumerationV6:
    """Enumerate verified controls without selecting or composing them.

    A sequential planner may declare a larger choice family before any probe
    is executed and pass it through ``predeclared_path_keys``.  Every accepted
    control's tail receipt must cover that entire family, including optional
    controls that were not eventually run.  This closes the common loophole
    of recalibrating the family after observing early probe responses.
    """

    rows = tuple(hypotheses)
    if len({row.key for row in rows}) != len(rows):
        raise ValueError("hypothesis keys must be unique")
    observed_paths = frozenset(
        f"{row.key}/{control.key}"
        for row in rows for control in row.controls
    )
    declared = observed_paths if predeclared_path_keys is None else frozenset(
        predeclared_path_keys
    )
    if not observed_paths.issubset(declared) or any(not key for key in declared):
        raise ValueError("observed controls must belong to the predeclared family")
    rejected_hypotheses: dict[str, tuple[str, ...]] = {}
    rejected_controls: dict[str, tuple[str, ...]] = {}
    choices: list[CertifiedControlOptionV6] = []
    for row in rows:
        hypothesis_reasons = _hypothesis_reasons(row, config)
        if hypothesis_reasons:
            rejected_hypotheses[row.key] = tuple(hypothesis_reasons)
            continue
        for control in row.controls:
            control_choices, control_reasons = _control_choices(
                row, control, config, declared,
            )
            path_key = f"{row.key}/{control.key}"
            if control_reasons:
                rejected_controls[path_key] = tuple(control_reasons)
            choices.extend(control_choices)
    choices.sort(key=lambda item: (item.path_key, item.beta))
    return CertifiedControlEnumerationV6(
        options=tuple(choices),
        rejected_hypotheses=rejected_hypotheses,
        rejected_controls=rejected_controls,
        predeclared_path_keys=declared,
    )


def select_action_paths_v6(
    hypotheses: Sequence[ActionHypothesisV6],
    *,
    config: SelectorV6Config = SelectorV6Config(),
) -> SelectorV6Decision:
    """Select exact controls and exact calibrated betas, otherwise native."""
    enumeration = enumerate_certified_control_options_v6(
        hypotheses, config=config,
    )
    choices = enumeration.options
    rejected_hypotheses = dict(enumeration.rejected_hypotheses)
    rejected_controls = dict(enumeration.rejected_controls)
    selected, interaction, utility = _best_subset(choices, config)
    if not selected:
        return SelectorV6Decision(
            state="native",
            selected_hypotheses=(),
            selected_controls={},
            input_strengths={},
            output_trusts={},
            certified_epe_gain_lowers={},
            rejected_hypotheses=rejected_hypotheses,
            rejected_controls=rejected_controls,
            total_certified_epe_gain_lower=0.0,
            total_interaction_upper_px=0.0,
            total_weighted_mean_harm_upper_px=0.0,
            total_compute_seconds=0.0,
            total_matcher_trajectories=0,
            stop_reason=(
                "no_exact_control_passed_typed_evidence" if not choices
                else "no_positive_budget_feasible_composition"
            ),
        )
    ordered = tuple(sorted(selected, key=lambda item: item.hypothesis.key))
    hypotheses_selected = tuple(item.hypothesis.key for item in ordered)
    harm = sum(
        item.mean_harm_upper_px * item.delivery_fraction for item in ordered
    )
    compute = sum(item.cost_seconds for item in ordered)
    trajectories = sum(
        item.control.extra_matcher_trajectories for item in ordered
    )
    return SelectorV6Decision(
        state="selected",
        selected_hypotheses=hypotheses_selected,
        selected_controls={
            item.hypothesis.key: item.control.key for item in ordered
        },
        input_strengths={
            item.hypothesis.key: item.control.input_strength for item in ordered
        },
        output_trusts={item.hypothesis.key: item.beta for item in ordered},
        certified_epe_gain_lowers={
            item.hypothesis.key: item.utility_lower_px for item in ordered
        },
        rejected_hypotheses=rejected_hypotheses,
        rejected_controls=rejected_controls,
        total_certified_epe_gain_lower=float(utility),
        total_interaction_upper_px=float(interaction),
        total_weighted_mean_harm_upper_px=float(harm),
        total_compute_seconds=float(compute),
        total_matcher_trajectories=int(trajectories),
        stop_reason="metric_aligned_exact_control_optimum",
    )
