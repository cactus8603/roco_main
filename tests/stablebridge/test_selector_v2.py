import pytest

from stablebridge.physical_repair.contracts import ActionSpec, PhysicalCertificate
from stablebridge.physical_repair.selector_v2 import (
    ACTION_MECHANISMS,
    ActionCandidate,
    EvidenceSplit,
    PairwiseDominance,
    RecoverabilityEvidence,
    SelectorV2Config,
    TaskWitness,
    select_action_set,
)


def certificate(operator="wiener3", endpoint="first", status="supported"):
    action = ActionSpec(
        operator, "selector-v2-test", "image", endpoint, endpoint,
        f"{endpoint}_native",
    )
    return PhysicalCertificate(
        action=action, status=status, observation_support_fraction=0.8,
        identifiable_support_fraction=0.5, null_score=0.8,
        action_score=0.4, spatial_holdout_gain=0.2,
        parameter_uncertainty=0.1, fit_stability=0.9,
        rejection_reasons=() if status == "supported" else ("test_rejection",),
    )


def split(prefix="x", hypotheses=4, correction="bonferroni", verifier=True):
    return EvidenceSplit(
        fit_support_hash=f"{prefix}-fit",
        certificate_support_hash=f"{prefix}-cert",
        verifier_support_hash=f"{prefix}-verify" if verifier else None,
        hypotheses_tested=hypotheses, correction_method=correction,
        familywise_error_rate=0.05,
    )


def recoverability(prefix="x", *, status="supported", collision=0.0,
                   collision_policy="none", visibility=True):
    return RecoverabilityEvidence(
        status=status, input_support_hash=f"{prefix}-input",
        output_support_hash=f"{prefix}-output", input_support_fraction=0.4,
        output_support_fraction=0.3, information_retention_lower=0.6,
        translation_crlb_upper_px=0.2, uncovered_fraction=0.0,
        collision_fraction=collision, collision_policy=collision_policy,
        visibility_certified=visibility, action_footprint_radius_px=2.0,
        rejection_reasons=() if status == "supported" else ("not_recoverable",),
    )


def witness(key, prefix="x", mean=0.3, radius=0.05, harm=0.02,
            severe=0.0, retention=0.9):
    return TaskWitness(
        action_key=key, verifier_support_hash=f"{prefix}-verify",
        comparison_support_hash=f"{prefix}-output",
        predicted_utility_mean=mean, utility_radius=radius,
        predicted_harm_upper=harm, severe_harm_score=severe,
        support_retention=retention, extra_matcher_trajectories=1,
    )


def candidate(key="wiener3@first", operator="wiener3", endpoint="first",
              prefix="x", task=None, evidence=None, recovery=None,
              own_null_margin=0.2):
    return ActionCandidate(
        key=key, certificate=certificate(operator, endpoint),
        own_null_name=ACTION_MECHANISMS[operator].own_null,
        own_null_margin_lower=own_null_margin,
        evidence_split=evidence or split(prefix),
        recoverability=recovery or recoverability(prefix),
        task_witness=task,
    )


def test_multiple_physical_actions_remain_supported_and_request_witnesses():
    rows = (
        candidate(prefix="noise"),
        candidate("common_motion@first", "common_motion", "first", "motion"),
    )
    result = select_action_set(rows)
    assert result.state == "probe"
    assert set(result.supported_actions) == {"wiener3@first", "common_motion@first"}
    assert set(result.probe_actions) == set(result.supported_actions)
    assert result.selected_action is None


def test_direct_pairwise_witness_can_prune_but_not_admit_an_action():
    rows = (
        candidate(prefix="noise"),
        candidate("common_motion@first", "common_motion", "first", "motion"),
    )
    result = select_action_set(rows, (
        PairwiseDominance(
            "wiener3@first", "common_motion@first", 0.1, "pair-support",
        ),
    ))
    assert result.state == "probe"
    assert result.supported_actions == ("wiener3@first",)
    assert result.pruned_actions == {"common_motion@first": "wiener3@first"}


def test_uncorrected_multi_hypothesis_search_cannot_enter_supported_set():
    row = candidate(evidence=split("noise", hypotheses=8, correction="none"))
    result = select_action_set((row,))
    assert result.state == "native"
    assert "multiplicity_uncontrolled" in result.rejected_actions[row.key]


def test_unresolved_transport_collision_rejects_action():
    row = candidate(recovery=recoverability(
        "noise", collision=0.2, collision_policy="unresolved",
    ))
    result = select_action_set((row,))
    assert result.state == "native"
    assert "collision_unresolved" in result.rejected_actions[row.key]


def test_severe_harm_guard_returns_native():
    key = "wiener3@first"
    row = candidate(prefix="noise", task=witness(
        key, "noise", mean=0.5, radius=0.02, harm=0.3,
    ))
    result = select_action_set((row,))
    assert result.state == "native"
    assert "severe_harm_guard" in result.rejected_actions[key]


def test_small_nonzero_harm_shrinks_trust_instead_of_requiring_zero_harm():
    key = "wiener3@first"
    row = candidate(prefix="noise", task=witness(
        key, "noise", mean=0.08, radius=0.10, harm=0.10,
    ))
    result = select_action_set((row,))
    assert result.state == "selected"
    assert result.selected_action == key
    assert result.stop_reason == "positive_mean_with_bounded_small_harm"
    assert 0.05 <= result.trust_alpha < 1.0


def test_overlapping_task_intervals_abstain_as_ambiguous():
    first_key = "wiener3@first"
    second_key = "common_motion@first"
    rows = (
        candidate(first_key, "wiener3", "first", "noise",
                  task=witness(first_key, "noise", mean=0.30, radius=0.15)),
        candidate(second_key, "common_motion", "first", "motion",
                  task=witness(second_key, "motion", mean=0.25, radius=0.15)),
    )
    result = select_action_set(rows)
    assert result.state == "ambiguous"
    assert result.selected_action is None


def test_separated_task_interval_selects_unique_action():
    first_key = "wiener3@first"
    second_key = "common_motion@first"
    rows = (
        candidate(first_key, "wiener3", "first", "noise",
                  task=witness(first_key, "noise", mean=0.50, radius=0.05)),
        candidate(second_key, "common_motion", "first", "motion",
                  task=witness(second_key, "motion", mean=0.10, radius=0.05)),
    )
    result = select_action_set(rows)
    assert result.state == "selected"
    assert result.selected_action == first_key
    assert result.trust_alpha == pytest.approx(1.0)


def test_task_witness_must_use_frozen_comparison_support():
    key = "wiener3@first"
    bad = TaskWitness(
        action_key=key, verifier_support_hash="noise-verify",
        comparison_support_hash="changed-output",
        predicted_utility_mean=0.5, utility_radius=0.05,
        predicted_harm_upper=0.02, severe_harm_score=0.0,
        support_retention=0.9, extra_matcher_trajectories=1,
    )
    result = select_action_set((candidate(prefix="noise", task=bad),))
    assert result.state == "native"
    assert "comparison_support_changed" in result.rejected_actions[key]


def test_evidence_split_rejects_label_or_outcome_leakage():
    with pytest.raises(ValueError, match="labels, truth, or outcomes"):
        EvidenceSplit(
            "fit", "certificate", "verify", 1, "none", 0.05,
            corruption_label_read=True,
        )


def test_registry_names_action_specific_physics_and_information_factors():
    assert "paired_auto_cross_psd" in ACTION_MECHANISMS["wiener3"].physical_features
    assert "directional_sinc" in ACTION_MECHANISMS["common_motion"].physical_features
    assert "bessel_zero_pattern" in ACTION_MECHANISMS["common_disk"].physical_features
    assert "dct_quantization_lattice" in ACTION_MECHANISMS["jpeg_deblock"].physical_features
    assert "alias_free_band" in ACTION_MECHANISMS["pixelate"].recoverability_factors
    assert "invertible_intensity_support" in ACTION_MECHANISMS["rank3_pair"].recoverability_factors
