import pytest

from stablebridge.physical_repair import (
    ActionSpec,
    ParameterCertificate,
    ParameterSearchConfig,
    ProbeObservation,
    bounded_parameter_search,
)


def certificate(*, status="supported", family_margin=0.2, refinement=None):
    return ParameterCertificate(
        action=ActionSpec(
            "defocus_common_passband", "v1", "image", "first", "first",
            "first_native", branch_pair="CC",
        ),
        parameter_name="radius",
        status=status,
        parameter_domain=(0.5, 8.0),
        initial_bracket=(1.0, 3.0),
        identifiable_set=(1.0, 2.0, 3.0),
        physical_score_curve={1.0: 0.7, 2.0: 0.4, 3.0: 0.6},
        null_score_curve={1.0: 0.9, 2.0: 0.9, 3.0: 0.9},
        fit_support_fraction=0.4,
        check_support_fraction=0.3,
        spatial_stability=0.8,
        family_margin=family_margin,
        region_ids=("region-0",),
        comparison_support_hash="support-v1",
        refinement_parameter=refinement,
        ambiguity_reasons=() if status == "supported" else ("flat_curve",),
    )


def probe(parameter, mean, radius, *, harm=0.02, severe=0.0,
          retention=0.9, physical_gain=0.1, support_hash="support-v1",
          conflict=False, family_guard=False, trajectories=1):
    return ProbeObservation(
        parameter=parameter,
        repaired_input_hash=f"input-{parameter}",
        output_hashes={"CC": f"cc-{parameter}"},
        comparison_support_hash=support_hash,
        predicted_utility_mean=mean,
        utility_radius=radius,
        predicted_harm_upper=harm,
        severe_harm_score=severe,
        heldout_physical_gain=physical_gain,
        support_retention=retention,
        native_uncertainty=0.3,
        candidate_uncertainty=0.2,
        matcher_risk_reduction=0.1,
        spent_trajectories=trajectories,
        evidence_conflict=conflict,
        family_harm_guard=family_guard,
    )


def test_unsupported_family_falls_back_without_spending_a_probe():
    result = bounded_parameter_search(certificate(status="ambiguous"))
    assert result.decision == "native"
    assert result.probe_count == 0
    assert result.spent_trajectories == 0
    assert result.stop_reason == "family_ambiguous"


def test_supported_family_first_requests_both_bracket_endpoints():
    result = bounded_parameter_search(certificate())
    assert result.decision == "probe"
    assert result.requested_parameters == (1.0, 3.0)
    assert result.state_trace == ("proposed", "family-supported", "bracketed")


def test_separated_positive_lower_bound_selects_endpoint():
    result = bounded_parameter_search(
        certificate(),
        (probe(1.0, 0.35, 0.05), probe(3.0, 0.05, 0.04)),
    )
    assert result.decision == "accepted"
    assert result.selected_parameter == pytest.approx(1.0)
    assert result.trust_alpha == pytest.approx(1.0)
    assert result.probe_count == 2


def test_overlapping_endpoints_request_exactly_one_midpoint():
    first = bounded_parameter_search(
        certificate(),
        (probe(1.0, 0.20, 0.10), probe(3.0, 0.18, 0.10)),
    )
    assert first.decision == "probe"
    assert first.requested_parameters == (2.0,)
    final = bounded_parameter_search(
        certificate(),
        (probe(1.0, 0.20, 0.10), probe(3.0, 0.18, 0.10),
         probe(2.0, 0.19, 0.10)),
    )
    assert final.decision == "native"
    assert final.probe_count == 3
    assert final.stop_reason == "utility_not_separated_after_refinement"


def test_refinement_can_identify_a_separated_midpoint():
    result = bounded_parameter_search(
        certificate(),
        (probe(1.0, 0.10, 0.08), probe(3.0, 0.11, 0.08),
         probe(2.0, 0.40, 0.04)),
    )
    assert result.decision == "accepted"
    assert result.selected_parameter == pytest.approx(2.0)


def test_small_uncertain_harm_shrinks_instead_of_forcing_zero_harm():
    config = ParameterSearchConfig(small_harm_budget=0.05)
    result = bounded_parameter_search(
        certificate(),
        (probe(1.0, 0.08, 0.10, harm=0.10),
         probe(3.0, -0.20, 0.05, harm=0.04)),
        config=config,
    )
    assert result.decision == "shrunk"
    assert result.selected_parameter == pytest.approx(1.0)
    assert 0.05 <= result.trust_alpha < 1.0


def test_severe_probe_is_rejected_but_safe_separated_probe_can_survive():
    result = bounded_parameter_search(
        certificate(),
        (probe(1.0, 0.5, 0.02, harm=0.3),
         probe(3.0, 0.2, 0.02, harm=0.03)),
    )
    assert result.decision == "accepted"
    assert result.selected_parameter == pytest.approx(3.0)
    assert result.rejected_parameters[1.0] == "severe_harm_guard"


def test_family_harm_guard_forces_native():
    result = bounded_parameter_search(
        certificate(),
        (probe(1.0, 0.5, 0.02, family_guard=True),
         probe(3.0, 0.4, 0.02)),
    )
    assert result.decision == "native"
    assert result.stop_reason == "family_harm_guard"


def test_support_must_remain_fixed_across_parameter_probes():
    result = bounded_parameter_search(
        certificate(),
        (probe(1.0, 0.5, 0.02),
         probe(3.0, 0.4, 0.02, support_hash="changed")),
    )
    assert result.decision == "native"
    assert result.stop_reason == "comparison_support_changed"


def test_quartet_signal_requires_all_four_counterfactual_outputs():
    with pytest.raises(ValueError, match="all CC/RC/CR/RR"):
        ProbeObservation(
            parameter=1.0,
            repaired_input_hash="input",
            output_hashes={"CC": "cc"},
            comparison_support_hash="support-v1",
            predicted_utility_mean=0.2,
            utility_radius=0.1,
            predicted_harm_upper=0.02,
            severe_harm_score=0.0,
            heldout_physical_gain=0.1,
            support_retention=0.9,
            native_uncertainty=0.3,
            candidate_uncertainty=0.2,
            matcher_risk_reduction=0.1,
            quartet_dispersion_reduction=0.1,
        )


def test_fourth_parameter_probe_is_impossible():
    with pytest.raises(ValueError, match=r"K=2\+1"):
        bounded_parameter_search(
            certificate(),
            (probe(1.0, 0.1, 0.1), probe(3.0, 0.1, 0.1),
             probe(2.0, 0.1, 0.1), probe(2.5, 0.1, 0.1)),
        )
