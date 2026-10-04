"""Typed, action-specific selector without a generic sole winner.

The selector deliberately separates three questions:

* is an action supported by evidence against its own physical null;
* is there recoverable, visible support on which that action may operate; and
* does an independent matcher-side witness predict useful, bounded intervention.

It never receives a corruption label or ground-truth outcome.  More than one
action may remain physically supported.  A unique action is returned only when
its signed-utility interval is separated from every surviving alternative;
otherwise the caller must acquire the requested witnesses or use native output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .contracts import ActionSpec, PhysicalCertificate


CORRECTION_METHODS = frozenset({
    "none", "bonferroni", "holm", "simultaneous_bound", "fixed_sequence",
    "e_value",
})
COLLISION_POLICIES = frozenset({"none", "exclude", "downweight", "unresolved"})
SELECTOR_STATES = frozenset({"native", "probe", "ambiguous", "selected"})


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
class ActionMechanism:
    """Auditable physical factors expected for one operator family."""

    family: str
    own_null: str
    physical_features: tuple[str, ...]
    recoverability_factors: tuple[str, ...]
    parameter_name: str | None = None

    def __post_init__(self) -> None:
        if not self.family or not self.own_null:
            raise ValueError("mechanism family and own null must be named")
        if not self.physical_features or not self.recoverability_factors:
            raise ValueError("mechanism needs physical and recoverability factors")
        if any(not value for value in (*self.physical_features,
                                       *self.recoverability_factors)):
            raise ValueError("mechanism feature names must be nonempty")


# This registry is explanatory rather than a learned family classifier.  It
# states what must be measured before the corresponding action may enter the
# supported set.  Aliases preserve the operator names used by frozen studies.
ACTION_MECHANISMS: Mapping[str, ActionMechanism] = {
    "impulse_exact_median3": ActionMechanism(
        "impulse", "persistent-or-corresponding-extremum",
        ("sparse_local_extremum", "paired_correspondence_disagreement"),
        ("non_impulse_neighbour_support", "action_footprint"),
        "impulse_density",
    ),
    "impulse_median3": ActionMechanism(
        "impulse", "persistent-or-corresponding-extremum",
        ("sparse_local_extremum", "paired_correspondence_disagreement"),
        ("non_impulse_neighbour_support", "action_footprint"),
        "impulse_density",
    ),
    "wiener3": ActionMechanism(
        "additive_noise", "shared_scene-spectrum-without-independent-noise",
        ("paired_auto_cross_psd", "j_invariant_or_sure_risk", "spatial_stationarity"),
        ("signal_band_retention", "estimated_noise_psd"),
        "noise_psd",
    ),
    "common_disk": ActionMechanism(
        "defocus_disk", "identity-otf-or-competing-psf",
        ("bessel_zero_pattern", "heldout_redegradation", "phase_consistency"),
        ("common_passband", "fisher_information", "translation_crlb"),
        "radius",
    ),
    "common_gaussian": ActionMechanism(
        "gaussian_blur", "identity-otf-or-competing-psf",
        ("log_otf_quadratic", "heldout_redegradation", "phase_consistency"),
        ("common_passband", "fisher_information", "translation_crlb"),
        "sigma",
    ),
    "common_motion": ActionMechanism(
        "motion_blur", "identity-otf-or-competing-psf",
        ("directional_sinc", "cepstral_periodicity", "spatial_psf_stability"),
        ("common_passband", "directional_information", "translation_crlb"),
        "length_angle",
    ),
    "jpeg_deblock": ActionMechanism(
        "jpeg", "unaligned-or-nonquantized-dct",
        ("dct_quantization_lattice", "grid_phase", "reencode_consistency"),
        ("ringing_reduction", "texture_preservation", "dct_interval_support"),
        "quality_table",
    ),
    "pixelate": ActionMechanism(
        "pixelate", "no-sampling-lattice",
        ("sampling_lattice", "spectral_replicas", "grid_phase"),
        ("alias_free_band", "sampling_support"),
        "sampling_factor",
    ),
    "rank3_pair": ActionMechanism(
        "radiometry", "nonmonotone-or-noninvertible-radiometry",
        ("rank_order_preservation", "monotone_derivative", "clipping_test"),
        ("invertible_intensity_support", "texture_preservation"),
        "monotone_map",
    ),
}


@dataclass(frozen=True)
class EvidenceSplit:
    """Provenance for parameter fit (A), certificate (B), and task witness (C)."""

    fit_support_hash: str
    certificate_support_hash: str
    verifier_support_hash: str | None
    hypotheses_tested: int
    correction_method: str
    familywise_error_rate: float
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if not self.fit_support_hash or not self.certificate_support_hash:
            raise ValueError("A/B support hashes are required")
        if self.fit_support_hash == self.certificate_support_hash:
            raise ValueError("parameter fit and certificate supports must differ")
        if self.verifier_support_hash is not None:
            if not self.verifier_support_hash:
                raise ValueError("C support hash must be nonempty when present")
            if self.verifier_support_hash in {
                self.fit_support_hash, self.certificate_support_hash,
            }:
                raise ValueError("task verifier support must be separate from A/B")
        if self.hypotheses_tested < 1:
            raise ValueError("at least one physical hypothesis is required")
        if self.correction_method not in CORRECTION_METHODS:
            raise ValueError("unknown multiplicity correction")
        _fraction(self.familywise_error_rate, "familywise error rate")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("selector evidence cannot read labels, truth, or outcomes")

    @property
    def multiplicity_controlled(self) -> bool:
        return bool(
            self.hypotheses_tested == 1
            or self.correction_method != "none"
        )


@dataclass(frozen=True)
class RecoverabilityEvidence:
    """Action support after visibility, transport, and information accounting."""

    status: str
    input_support_hash: str
    output_support_hash: str
    input_support_fraction: float
    output_support_fraction: float
    information_retention_lower: float
    translation_crlb_upper_px: float
    uncovered_fraction: float
    collision_fraction: float
    collision_policy: str
    visibility_certified: bool
    action_footprint_radius_px: float
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"supported", "ambiguous", "rejected", "unsupported"}:
            raise ValueError("invalid recoverability status")
        if not self.input_support_hash or not self.output_support_hash:
            raise ValueError("input and output support hashes are required")
        _fraction(self.input_support_fraction, "input support fraction")
        _fraction(self.output_support_fraction, "output support fraction")
        _fraction(self.information_retention_lower, "information retention lower bound")
        _nonnegative(self.translation_crlb_upper_px, "translation CRLB upper bound")
        _fraction(self.uncovered_fraction, "uncovered fraction")
        _fraction(self.collision_fraction, "collision fraction")
        _nonnegative(self.action_footprint_radius_px, "action footprint radius")
        if self.collision_policy not in COLLISION_POLICIES:
            raise ValueError("invalid collision policy")
        if self.status == "supported" and self.rejection_reasons:
            raise ValueError("supported recoverability cannot have rejection reasons")
        if self.status != "supported" and not self.rejection_reasons:
            raise ValueError("non-supported recoverability needs a reason")


@dataclass(frozen=True)
class TaskWitness:
    """GT-free signed-utility and harm prediction on a fixed output support."""

    action_key: str
    verifier_support_hash: str
    comparison_support_hash: str
    predicted_utility_mean: float
    utility_radius: float
    predicted_harm_upper: float
    severe_harm_score: float
    support_retention: float
    extra_matcher_trajectories: int
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if not self.action_key or not self.verifier_support_hash:
            raise ValueError("task witness identity and verifier hash are required")
        if not self.comparison_support_hash:
            raise ValueError("task comparison support hash is required")
        _finite(self.predicted_utility_mean, "predicted utility mean")
        _nonnegative(self.utility_radius, "utility radius")
        _nonnegative(self.predicted_harm_upper, "predicted harm upper bound")
        _nonnegative(self.severe_harm_score, "severe harm score")
        _fraction(self.support_retention, "support retention")
        if self.extra_matcher_trajectories < 1:
            raise ValueError("task witness must account for a matcher trajectory")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("task witness cannot read labels, truth, or outcomes")

    @property
    def utility_lower(self) -> float:
        return float(self.predicted_utility_mean - self.utility_radius)

    @property
    def utility_upper(self) -> float:
        return float(self.predicted_utility_mean + self.utility_radius)


@dataclass(frozen=True)
class ActionCandidate:
    key: str
    certificate: PhysicalCertificate
    own_null_name: str
    own_null_margin_lower: float
    evidence_split: EvidenceSplit
    recoverability: RecoverabilityEvidence
    task_witness: TaskWitness | None = None

    def __post_init__(self) -> None:
        if not self.key or not self.own_null_name:
            raise ValueError("candidate and own-null names are required")
        if self.certificate.action.operator_id not in ACTION_MECHANISMS:
            raise ValueError("candidate operator has no physical mechanism registry entry")
        _finite(self.own_null_margin_lower, "own-null margin lower bound")
        if self.task_witness is not None and self.task_witness.action_key != self.key:
            raise ValueError("task witness belongs to a different action")


@dataclass(frozen=True)
class PairwiseDominance:
    """A direct witness may prune a loser; it never creates support for a winner."""

    winner_key: str
    loser_key: str
    margin_lower: float
    comparison_support_hash: str
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if (not self.winner_key or not self.loser_key
                or self.winner_key == self.loser_key):
            raise ValueError("pairwise witness needs distinct named actions")
        if not self.comparison_support_hash:
            raise ValueError("pairwise comparison support hash is required")
        _finite(self.margin_lower, "pairwise margin lower bound")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("pairwise witness cannot read labels, truth, or outcomes")


@dataclass(frozen=True)
class SelectorV2Config:
    maximum_familywise_error_rate: float = 0.05
    minimum_information_retention: float = 0.0
    minimum_support_retention: float = 0.5
    minimum_utility_lower: float = 0.0
    separation_margin: float = 0.0
    small_harm_budget: float = 0.05
    severe_harm_threshold: float = 0.25
    minimum_shrunk_alpha: float = 0.05

    def __post_init__(self) -> None:
        _fraction(self.maximum_familywise_error_rate,
                  "maximum familywise error rate")
        _fraction(self.minimum_information_retention,
                  "minimum information retention")
        _fraction(self.minimum_support_retention, "minimum support retention")
        _finite(self.minimum_utility_lower, "minimum utility lower bound")
        _nonnegative(self.separation_margin, "separation margin")
        _nonnegative(self.small_harm_budget, "small harm budget")
        _nonnegative(self.severe_harm_threshold, "severe harm threshold")
        _fraction(self.minimum_shrunk_alpha, "minimum shrunk alpha")
        if self.small_harm_budget > self.severe_harm_threshold:
            raise ValueError("small harm budget cannot exceed severe threshold")


@dataclass(frozen=True)
class SelectorV2Decision:
    state: str
    supported_actions: tuple[str, ...]
    pruned_actions: Mapping[str, str]
    probe_actions: tuple[str, ...]
    selected_action: str | None
    trust_alpha: float
    rejected_actions: Mapping[str, tuple[str, ...]]
    utility_intervals: Mapping[str, tuple[float, float]]
    stop_reason: str

    def __post_init__(self) -> None:
        if self.state not in SELECTOR_STATES or not self.stop_reason:
            raise ValueError("invalid selector decision")
        _fraction(self.trust_alpha, "trust alpha")
        if self.state == "selected":
            if self.selected_action is None or self.trust_alpha <= 0.0:
                raise ValueError("selected decision needs an action and trust")
        elif self.selected_action is not None or self.trust_alpha != 0.0:
            raise ValueError("non-selected decision cannot modify native output")


def _physical_reasons(candidate: ActionCandidate,
                      config: SelectorV2Config) -> list[str]:
    reasons: list[str] = []
    if candidate.certificate.status != "supported":
        reasons.append(f"physical_{candidate.certificate.status}")
    if candidate.own_null_margin_lower <= 0.0:
        reasons.append("own_null_not_rejected")
    split = candidate.evidence_split
    if not split.multiplicity_controlled:
        reasons.append("multiplicity_uncontrolled")
    if split.familywise_error_rate > config.maximum_familywise_error_rate:
        reasons.append("familywise_error_too_large")
    recoverability = candidate.recoverability
    if recoverability.status != "supported":
        reasons.append(f"recoverability_{recoverability.status}")
    if not recoverability.visibility_certified:
        reasons.append("visibility_uncertified")
    if (recoverability.collision_fraction > 0.0
            and recoverability.collision_policy == "unresolved"):
        reasons.append("collision_unresolved")
    if recoverability.information_retention_lower <= config.minimum_information_retention:
        reasons.append("insufficient_information_retention")
    if recoverability.output_support_fraction <= 0.0:
        reasons.append("empty_output_support")
    return reasons


def _witness_reasons(candidate: ActionCandidate,
                     config: SelectorV2Config) -> list[str]:
    witness = candidate.task_witness
    if witness is None:
        return ["task_witness_missing"]
    reasons: list[str] = []
    split = candidate.evidence_split
    if split.verifier_support_hash is None:
        reasons.append("verifier_partition_missing")
    elif witness.verifier_support_hash != split.verifier_support_hash:
        reasons.append("verifier_partition_changed")
    if witness.comparison_support_hash != candidate.recoverability.output_support_hash:
        reasons.append("comparison_support_changed")
    if witness.support_retention < config.minimum_support_retention:
        reasons.append("support_retention")
    if (witness.severe_harm_score >= config.severe_harm_threshold
            or witness.predicted_harm_upper >= config.severe_harm_threshold):
        reasons.append("severe_harm_guard")
    if witness.predicted_utility_mean <= 0.0:
        reasons.append("nonpositive_predicted_utility")
    return reasons


def select_action_set(
    candidates: Sequence[ActionCandidate],
    pairwise: Sequence[PairwiseDominance] = (),
    *,
    config: SelectorV2Config = SelectorV2Config(),
) -> SelectorV2Decision:
    """Return a supported set, probes, or one interval-separated action.

    Physical certificates can only admit their own action.  Pairwise witnesses
    can prune an already-supported loser but cannot admit an unsupported
    winner.  Missing or overlapping task evidence therefore results in a
    bounded probe request or native fallback, never a generic top-1 guess.
    """
    rows = tuple(candidates)
    by_key = {row.key: row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("candidate keys must be unique")

    rejected: dict[str, tuple[str, ...]] = {}
    supported: set[str] = set()
    for row in rows:
        reasons = _physical_reasons(row, config)
        if reasons:
            rejected[row.key] = tuple(reasons)
        else:
            supported.add(row.key)

    pruned: dict[str, str] = {}
    for witness in pairwise:
        if witness.margin_lower <= 0.0:
            continue
        if witness.winner_key in supported and witness.loser_key in supported:
            supported.remove(witness.loser_key)
            pruned[witness.loser_key] = witness.winner_key

    if not supported:
        return SelectorV2Decision(
            state="native", supported_actions=(), pruned_actions=pruned,
            probe_actions=(), selected_action=None, trust_alpha=0.0,
            rejected_actions=rejected, utility_intervals={},
            stop_reason="no_physically_supported_recoverable_action",
        )

    probes: list[str] = []
    eligible: list[ActionCandidate] = []
    intervals: dict[str, tuple[float, float]] = {}
    for key in sorted(supported):
        row = by_key[key]
        reasons = _witness_reasons(row, config)
        if reasons == ["task_witness_missing"]:
            probes.append(key)
            continue
        if reasons:
            rejected[key] = tuple(reasons)
            continue
        assert row.task_witness is not None
        eligible.append(row)
        intervals[key] = (
            row.task_witness.utility_lower, row.task_witness.utility_upper,
        )

    surviving = tuple(sorted(supported))
    if probes:
        return SelectorV2Decision(
            state="probe", supported_actions=surviving, pruned_actions=pruned,
            probe_actions=tuple(probes), selected_action=None, trust_alpha=0.0,
            rejected_actions=rejected, utility_intervals=intervals,
            stop_reason="independent_task_witness_required",
        )
    if not eligible:
        return SelectorV2Decision(
            state="native", supported_actions=surviving, pruned_actions=pruned,
            probe_actions=(), selected_action=None, trust_alpha=0.0,
            rejected_actions=rejected, utility_intervals=intervals,
            stop_reason="no_safe_positive_task_witness",
        )

    ranked = sorted(
        eligible,
        key=lambda row: (
            row.task_witness.predicted_utility_mean,
            row.task_witness.utility_lower,
            row.key,
        ),
        reverse=True,
    )
    best = ranked[0]
    assert best.task_witness is not None
    separated = all(
        best.task_witness.utility_lower
        > other.task_witness.utility_upper + config.separation_margin
        for other in ranked[1:]
        if other.task_witness is not None
    )
    if not separated:
        return SelectorV2Decision(
            state="ambiguous", supported_actions=surviving,
            pruned_actions=pruned, probe_actions=(), selected_action=None,
            trust_alpha=0.0, rejected_actions=rejected,
            utility_intervals=intervals,
            stop_reason="task_utility_intervals_overlap",
        )

    witness = best.task_witness
    if witness.utility_lower > config.minimum_utility_lower:
        alpha = 1.0
        reason = "positive_separated_utility_lower_bound"
    else:
        confidence_scale = witness.predicted_utility_mean / (
            witness.predicted_utility_mean + witness.utility_radius + 1e-12
        )
        harm_scale = 1.0 if witness.predicted_harm_upper <= 0.0 else min(
            1.0, config.small_harm_budget / witness.predicted_harm_upper,
        )
        alpha = float(confidence_scale * harm_scale)
        if alpha < config.minimum_shrunk_alpha:
            rejected[best.key] = ("shrunk_trust_below_minimum",)
            return SelectorV2Decision(
                state="native", supported_actions=surviving,
                pruned_actions=pruned, probe_actions=(), selected_action=None,
                trust_alpha=0.0, rejected_actions=rejected,
                utility_intervals=intervals,
                stop_reason="small_harm_trust_too_small",
            )
        reason = "positive_mean_with_bounded_small_harm"
    return SelectorV2Decision(
        state="selected", supported_actions=surviving, pruned_actions=pruned,
        probe_actions=(), selected_action=best.key, trust_alpha=alpha,
        rejected_actions=rejected, utility_intervals=intervals,
        stop_reason=reason,
    )
