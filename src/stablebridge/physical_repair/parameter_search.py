"""Fail-closed, bounded search for regional action parameters.

The policy in this module never observes corruption labels or ground truth.
Physical evidence proposes a connected parameter bracket; matcher-derived
evidence may then accept, shrink, refine once, or return the region to the
immutable native prediction.  It deliberately does not execute a matcher so
the decision contract can be frozen and replayed independently of a backbone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .contracts import ActionSpec


PARAMETER_CERTIFICATE_STATES = frozenset(
    {"supported", "ambiguous", "rejected", "unsupported"}
)
SEARCH_DECISIONS = frozenset({"probe", "accepted", "shrunk", "native"})


def _finite(value: float, name: str) -> float:
    item = float(value)
    if not np.isfinite(item):
        raise ValueError(f"{name} must be finite")
    return item


def _finite_nonnegative(value: float, name: str) -> float:
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
class ParameterCertificate:
    """Physical, region-level justification for a finite parameter bracket."""

    action: ActionSpec
    parameter_name: str
    status: str
    parameter_domain: tuple[float, float]
    initial_bracket: tuple[float, float]
    identifiable_set: tuple[float, ...]
    physical_score_curve: Mapping[float, float]
    null_score_curve: Mapping[float, float]
    fit_support_fraction: float
    check_support_fraction: float
    spatial_stability: float
    family_margin: float
    region_ids: tuple[str, ...]
    comparison_support_hash: str
    refinement_parameter: float | None = None
    ambiguity_reasons: tuple[str, ...] = ()
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parameter_name:
            raise ValueError("parameter name must be nonempty")
        if self.status not in PARAMETER_CERTIFICATE_STATES:
            raise ValueError("invalid parameter-certificate status")
        domain = tuple(_finite(value, "parameter domain")
                       for value in self.parameter_domain)
        bracket = tuple(_finite(value, "initial bracket")
                        for value in self.initial_bracket)
        if len(domain) != 2 or domain[0] >= domain[1]:
            raise ValueError("parameter domain must be an increasing pair")
        if (len(bracket) != 2 or bracket[0] >= bracket[1]
                or bracket[0] < domain[0] or bracket[1] > domain[1]):
            raise ValueError("initial bracket must be increasing and inside the domain")
        identifiable = tuple(_finite(value, "identifiable parameter")
                             for value in self.identifiable_set)
        if any(value < bracket[0] or value > bracket[1] for value in identifiable):
            raise ValueError("identifiable parameters must lie inside the bracket")
        if self.status == "supported" and not identifiable:
            raise ValueError("supported certificate needs an identifiable set")
        if not self.region_ids or any(not item for item in self.region_ids):
            raise ValueError("parameter certificate needs named regions")
        if len(set(self.region_ids)) != len(self.region_ids):
            raise ValueError("region ids must be unique")
        if not self.comparison_support_hash:
            raise ValueError("a frozen comparison-support hash is required")
        _fraction(self.fit_support_fraction, "fit support fraction")
        _fraction(self.check_support_fraction, "check support fraction")
        _fraction(self.spatial_stability, "spatial stability")
        _finite(self.family_margin, "family margin")
        for curve_name, curve in (("physical", self.physical_score_curve),
                                  ("null", self.null_score_curve)):
            if not curve:
                raise ValueError(f"{curve_name} score curve must be nonempty")
            for parameter, score in curve.items():
                parameter = _finite(parameter, f"{curve_name} parameter")
                _finite(score, f"{curve_name} score")
                if parameter < domain[0] or parameter > domain[1]:
                    raise ValueError(f"{curve_name} parameter lies outside the domain")
        if self.refinement_parameter is not None:
            refinement = _finite(self.refinement_parameter, "refinement parameter")
            if refinement <= bracket[0] or refinement >= bracket[1]:
                raise ValueError("refinement parameter must be strictly inside the bracket")
        if not all(np.isfinite(float(value)) for value in self.diagnostics.values()):
            raise ValueError("parameter diagnostics must be finite")
        if self.status == "supported" and self.ambiguity_reasons:
            raise ValueError("supported certificate cannot have ambiguity reasons")
        if self.status != "supported" and not self.ambiguity_reasons:
            raise ValueError("non-supported certificate needs a reason")

    @property
    def bracket_endpoints(self) -> tuple[float, float]:
        return (float(self.initial_bracket[0]), float(self.initial_bracket[1]))

    @property
    def refinement(self) -> float:
        if self.refinement_parameter is not None:
            return float(self.refinement_parameter)
        lower, upper = self.bracket_endpoints
        return float((lower + upper) / 2.0)


@dataclass(frozen=True)
class ProbeObservation:
    """Observable evidence from one repaired-CC parameter probe.

    ``predicted_utility_mean`` is signed: positive means predicted EPE
    reduction.  ``utility_radius`` defines its conservative interval.  The
    harm fields are separately calibrated so a positive mean cannot hide a
    severe-tail warning.
    """

    parameter: float
    repaired_input_hash: str
    output_hashes: Mapping[str, str]
    comparison_support_hash: str
    predicted_utility_mean: float
    utility_radius: float
    predicted_harm_upper: float
    severe_harm_score: float
    heldout_physical_gain: float
    support_retention: float
    native_uncertainty: float
    candidate_uncertainty: float
    matcher_risk_reduction: float
    quartet_dispersion_reduction: float | None = None
    spent_trajectories: int = 1
    evidence_conflict: bool = False
    family_harm_guard: bool = False
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite(self.parameter, "probe parameter")
        if not self.repaired_input_hash:
            raise ValueError("repaired input hash must be nonempty")
        if not self.comparison_support_hash:
            raise ValueError("comparison support hash must be nonempty")
        if "CC" not in self.output_hashes or not self.output_hashes["CC"]:
            raise ValueError("every deployment probe requires a repaired-CC output")
        if any(branch not in {"CC", "RC", "CR", "RR"}
               for branch in self.output_hashes):
            raise ValueError("probe output hashes use an unknown branch")
        if any(not value for value in self.output_hashes.values()):
            raise ValueError("probe output hashes must be nonempty")
        _finite(self.predicted_utility_mean, "predicted utility mean")
        _finite_nonnegative(self.utility_radius, "utility radius")
        _finite_nonnegative(self.predicted_harm_upper, "predicted harm upper bound")
        _finite_nonnegative(self.severe_harm_score, "severe harm score")
        _finite(self.heldout_physical_gain, "held-out physical gain")
        _fraction(self.support_retention, "support retention")
        _finite_nonnegative(self.native_uncertainty, "native uncertainty")
        _finite_nonnegative(self.candidate_uncertainty, "candidate uncertainty")
        _finite(self.matcher_risk_reduction, "matcher risk reduction")
        if self.quartet_dispersion_reduction is not None:
            _finite(self.quartet_dispersion_reduction,
                    "quartet dispersion reduction")
            if set(self.output_hashes) != {"CC", "RC", "CR", "RR"}:
                raise ValueError(
                    "quartet dispersion requires all CC/RC/CR/RR output hashes"
                )
        if self.spent_trajectories < 1:
            raise ValueError("a completed probe must spend at least one trajectory")
        if not all(np.isfinite(float(value)) for value in self.diagnostics.values()):
            raise ValueError("probe diagnostics must be finite")

    @property
    def utility_lower(self) -> float:
        return float(self.predicted_utility_mean - self.utility_radius)

    @property
    def utility_upper(self) -> float:
        return float(self.predicted_utility_mean + self.utility_radius)


@dataclass(frozen=True)
class ParameterSearchConfig:
    """Frozen thresholds for the first ``K=2+1`` deployment policy."""

    minimum_support_retention: float = 0.5
    minimum_family_margin: float = 0.0
    minimum_utility_lower: float = 0.0
    separation_margin: float = 0.0
    small_harm_budget: float = 0.05
    severe_harm_threshold: float = 0.25
    base_trust_alpha: float = 1.0
    minimum_shrunk_alpha: float = 0.05
    parameter_tolerance: float = 1e-8
    max_parameter_probes: int = 3

    def __post_init__(self) -> None:
        _fraction(self.minimum_support_retention, "minimum support retention")
        _finite(self.minimum_family_margin, "minimum family margin")
        _finite(self.minimum_utility_lower, "minimum utility lower bound")
        _finite_nonnegative(self.separation_margin, "separation margin")
        _finite_nonnegative(self.small_harm_budget, "small harm budget")
        _finite_nonnegative(self.severe_harm_threshold, "severe harm threshold")
        if self.small_harm_budget > self.severe_harm_threshold:
            raise ValueError("small harm budget cannot exceed the severe threshold")
        _fraction(self.base_trust_alpha, "base trust alpha")
        _fraction(self.minimum_shrunk_alpha, "minimum shrunk alpha")
        if self.minimum_shrunk_alpha > self.base_trust_alpha:
            raise ValueError("minimum shrunk alpha cannot exceed base alpha")
        if _finite_nonnegative(self.parameter_tolerance, "parameter tolerance") == 0.0:
            raise ValueError("parameter tolerance must be positive")
        if self.max_parameter_probes != 3:
            raise ValueError("the first bounded policy is fixed to K=2+1")


@dataclass(frozen=True)
class RegionSearchDecision:
    """One replayable transition result for one certificate/region group."""

    decision: str
    requested_parameters: tuple[float, ...]
    selected_parameter: float | None
    trust_alpha: float
    probe_count: int
    spent_trajectories: int
    stop_reason: str
    state_trace: tuple[str, ...]
    rejected_parameters: Mapping[float, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in SEARCH_DECISIONS:
            raise ValueError("invalid parameter-search decision")
        if self.probe_count < 0 or self.probe_count > 3:
            raise ValueError("probe count lies outside K=2+1")
        if self.spent_trajectories < 0:
            raise ValueError("spent trajectory count must be nonnegative")
        _fraction(self.trust_alpha, "decision trust alpha")
        if not self.stop_reason or not self.state_trace:
            raise ValueError("decision needs a reason and state trace")
        requested = tuple(_finite(value, "requested parameter")
                          for value in self.requested_parameters)
        if len(set(requested)) != len(requested):
            raise ValueError("requested parameters must be unique")
        if self.decision == "probe":
            if not requested or self.selected_parameter is not None or self.trust_alpha != 0.0:
                raise ValueError("probe decision can only request new parameters")
        else:
            if requested:
                raise ValueError("terminal decision cannot request parameters")
        if self.decision in {"accepted", "shrunk"}:
            if self.selected_parameter is None or self.trust_alpha <= 0.0:
                raise ValueError("accepted decision needs a parameter and positive trust")
        elif self.selected_parameter is not None or self.trust_alpha != 0.0:
            raise ValueError("native/probe decision cannot select an update")


def _same_parameter(left: float, right: float, tolerance: float) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def _find_probe(probes: Sequence[ProbeObservation], parameter: float,
                tolerance: float) -> ProbeObservation | None:
    matches = [probe for probe in probes
               if _same_parameter(probe.parameter, parameter, tolerance)]
    if len(matches) > 1:
        raise ValueError("duplicate probe parameter")
    return matches[0] if matches else None


def _native(probes: Sequence[ProbeObservation], reason: str,
            trace: tuple[str, ...], rejected: Mapping[float, str]) -> RegionSearchDecision:
    return RegionSearchDecision(
        decision="native", requested_parameters=(), selected_parameter=None,
        trust_alpha=0.0, probe_count=len(probes),
        spent_trajectories=sum(probe.spent_trajectories for probe in probes),
        stop_reason=reason, state_trace=trace + ("native",),
        rejected_parameters=dict(rejected),
    )


def _probe(probes: Sequence[ProbeObservation], requested: Sequence[float],
           reason: str, trace: tuple[str, ...],
           rejected: Mapping[float, str]) -> RegionSearchDecision:
    return RegionSearchDecision(
        decision="probe", requested_parameters=tuple(float(value) for value in requested),
        selected_parameter=None, trust_alpha=0.0, probe_count=len(probes),
        spent_trajectories=sum(probe.spent_trajectories for probe in probes),
        stop_reason=reason, state_trace=trace,
        rejected_parameters=dict(rejected),
    )


def _finish(probes: Sequence[ProbeObservation], selected: ProbeObservation,
            decision: str, alpha: float, reason: str,
            trace: tuple[str, ...], rejected: Mapping[float, str]) -> RegionSearchDecision:
    return RegionSearchDecision(
        decision=decision, requested_parameters=(),
        selected_parameter=float(selected.parameter), trust_alpha=float(alpha),
        probe_count=len(probes),
        spent_trajectories=sum(probe.spent_trajectories for probe in probes),
        stop_reason=reason, state_trace=trace + (decision,),
        rejected_parameters=dict(rejected),
    )


def bounded_parameter_search(
    certificate: ParameterCertificate,
    probes: Sequence[ProbeObservation] = (),
    *,
    config: ParameterSearchConfig = ParameterSearchConfig(),
) -> RegionSearchDecision:
    """Advance a physical-bracket search without exceeding ``K=2+1``.

    The caller executes every requested parameter as a real repaired-CC
    matcher rerun, then calls this function again with the accumulated
    observations.  A terminal result selects at most one parameter; composing
    that result against native remains the responsibility of ``bounded_output``.
    """
    probes = tuple(probes)
    if len(probes) > config.max_parameter_probes:
        raise ValueError("parameter search exceeded K=2+1")
    trace = ("proposed",)
    if certificate.status != "supported":
        return _native(probes, f"family_{certificate.status}", trace, {})
    if certificate.family_margin <= config.minimum_family_margin:
        return _native(probes, "family_does_not_beat_null", trace, {})
    trace += ("family-supported", "bracketed")
    lower, upper = certificate.bracket_endpoints
    allowed = (lower, upper, certificate.refinement)
    for probe in probes:
        if not any(_same_parameter(probe.parameter, value, config.parameter_tolerance)
                   for value in allowed):
            raise ValueError("probe parameter is outside the frozen K=2+1 set")
    missing_endpoints = [value for value in (lower, upper)
                         if _find_probe(probes, value, config.parameter_tolerance) is None]
    if missing_endpoints:
        return _probe(probes, missing_endpoints, "probe_bracket_endpoints", trace, {})

    trace += ("probed",)
    rejected: dict[float, str] = {}
    eligible: list[ProbeObservation] = []
    if any(probe.comparison_support_hash != certificate.comparison_support_hash
           for probe in probes):
        return _native(probes, "comparison_support_changed", trace, rejected)
    if any(probe.family_harm_guard for probe in probes):
        return _native(probes, "family_harm_guard", trace, rejected)
    for probe in probes:
        if probe.evidence_conflict:
            rejected[float(probe.parameter)] = "evidence_conflict"
        elif probe.support_retention < config.minimum_support_retention:
            rejected[float(probe.parameter)] = "support_retention"
        elif (probe.severe_harm_score >= config.severe_harm_threshold
              or probe.predicted_harm_upper >= config.severe_harm_threshold):
            rejected[float(probe.parameter)] = "severe_harm_guard"
        elif probe.heldout_physical_gain <= 0.0:
            rejected[float(probe.parameter)] = "physical_check_failed"
        else:
            eligible.append(probe)
    if not eligible:
        return _native(probes, "no_safe_supported_probe", trace, rejected)

    ranked = sorted(
        eligible,
        key=lambda item: (item.predicted_utility_mean, item.utility_lower,
                          item.heldout_physical_gain, -item.parameter),
        reverse=True,
    )
    best = ranked[0]
    alternatives = ranked[1:]
    separated = not alternatives or all(
        best.utility_lower > other.utility_upper + config.separation_margin
        for other in alternatives
    )
    if not separated:
        refinement = certificate.refinement
        if (_find_probe(probes, refinement, config.parameter_tolerance) is None
                and len(probes) < config.max_parameter_probes
                and any(item.utility_upper > config.minimum_utility_lower
                        for item in eligible)):
            return _probe(
                probes, (refinement,), "endpoint_utility_intervals_overlap",
                trace, rejected,
            )
        return _native(probes, "utility_not_separated_after_refinement", trace, rejected)

    if best.utility_lower > config.minimum_utility_lower:
        return _finish(
            probes, best, "accepted", config.base_trust_alpha,
            "positive_separated_utility", trace, rejected,
        )
    if best.predicted_utility_mean <= 0.0:
        return _native(probes, "nonpositive_predicted_utility", trace, rejected)

    # The mean is beneficial but its uncertainty interval includes small harm.
    # Spend only a confidence- and harm-scaled fraction of the trust budget.
    confidence_scale = best.predicted_utility_mean / (
        best.predicted_utility_mean + best.utility_radius + 1e-12
    )
    if best.predicted_harm_upper <= 0.0:
        harm_scale = 1.0
    else:
        harm_scale = min(1.0, config.small_harm_budget / best.predicted_harm_upper)
    alpha = config.base_trust_alpha * confidence_scale * harm_scale
    if alpha < config.minimum_shrunk_alpha:
        return _native(probes, "shrunk_trust_below_minimum", trace, rejected)
    return _finish(
        probes, best, "shrunk", alpha,
        "positive_mean_with_bounded_small_harm", trace, rejected,
    )
