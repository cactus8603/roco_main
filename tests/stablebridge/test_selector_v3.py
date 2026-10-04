from dataclasses import replace

import pytest

from stablebridge.physical_repair.action_profiles import ACTION_PHYSICS_PROFILES
from stablebridge.physical_repair.contracts import ActionSpec, PhysicalCertificate
from stablebridge.physical_repair.selector_v2 import EvidenceSplit, RecoverabilityEvidence
from stablebridge.physical_repair.selector_v3 import (
    ParameterState,
    RegionActionHypothesis,
    SelectorV3Config,
    SequentialEvidence,
    WitnessBound,
    select_region_actions_v3,
)


def physical(operator="jpeg_deblock", endpoint="first"):
    return PhysicalCertificate(
        action=ActionSpec(
            operator, "test", "image", endpoint, endpoint, f"{endpoint}_native",
        ),
        status="supported", observation_support_fraction=0.5,
        identifiable_support_fraction=0.5, null_score=0.2,
        action_score=0.1, spatial_holdout_gain=0.1,
        parameter_uncertainty=0.1, fit_stability=1.0,
    )


def split():
    return EvidenceSplit(
        fit_support_hash="fit", certificate_support_hash="check",
        verifier_support_hash="task", hypotheses_tested=4,
        correction_method="bonferroni", familywise_error_rate=0.05,
    )


def recover(region="r"):
    return RecoverabilityEvidence(
        status="supported", input_support_hash=f"in-{region}",
        output_support_hash=f"out-{region}", input_support_fraction=0.2,
        output_support_fraction=0.2, information_retention_lower=0.8,
        translation_crlb_upper_px=0.2, uncovered_fraction=0.0,
        collision_fraction=0.0, collision_policy="none",
        visibility_certified=True, action_footprint_radius_px=2.0,
    )


def parameter(key="fixed", mean=0.4, radius=0.1, harm=0.02,
              physical_margin=0.2, information=0.8, task=True):
    return ParameterState(
        key=key, physical_margin_lower=physical_margin,
        information_retention_lower=information,
        predicted_utility_mean=mean if task else None,
        utility_radius=radius if task else None,
        predicted_harm_upper=harm if task else None,
        severe_harm_score=harm if task else None,
    )


def hypothesis(name="a", region="r1", parameters=None, operator="jpeg_deblock",
               endpoint="first", missing_decision=(), missing_closure=(),
               conflicts=(), probe_value=0.0, sequential=SequentialEvidence(),
               pixels=0.2, compute=1.0, trajectories=1):
    profile = ACTION_PHYSICS_PROFILES[operator]
    states = tuple(parameters or (parameter(),))
    parameter_keys = tuple(item.key for item in states)
    decision = tuple(
        WitnessBound(
            name=value, margin_lower=0.1, support_hash="check",
            parameter_keys=parameter_keys, effective_n=8.0,
        )
        for value in profile.decision_witnesses if value not in missing_decision
    )
    closure = tuple(
        WitnessBound(
            name=value, margin_lower=0.1, support_hash="check",
            parameter_keys=parameter_keys, effective_n=8.0,
        )
        for value in profile.closure_witnesses
        if value != "task_signed_utility" and value not in missing_closure
    )
    has_task = states[0].has_task_witness
    return RegionActionHypothesis(
        key=f"{operator}@{endpoint}#{name}",
        certificate=physical(operator, endpoint), evidence_split=split(),
        recoverability=recover(region), region_key=region,
        parameters=states,
        decision_witnesses=decision, closure_witnesses=closure,
        fit_effective_n=8.0, check_effective_n=8.0,
        pixel_fraction=pixels, task_support_retention=0.9,
        task_verifier_support_hash="task" if has_task else None,
        task_comparison_support_hash=f"out-{region}" if has_task else None,
        expected_compute_seconds=compute,
        extra_matcher_trajectories=trajectories,
        sequential=sequential, conflict_keys=tuple(conflicts),
        probe_value_upper=probe_value,
    )


def test_missing_profile_witness_fails_closed():
    row = hypothesis(missing_closure=("codec_reencode_noninferiority",))
    result = select_region_actions_v3((row,))
    assert result.state == "native"
    assert any(reason.startswith("missing_closure_witness")
               for reason in result.rejected_hypotheses[row.key])


def test_parameter_set_uses_worst_physical_and_information_member():
    row = hypothesis(parameters=(
        parameter("good"), parameter("bad", physical_margin=-0.01),
    ))
    result = select_region_actions_v3((row,))
    assert result.state == "native"
    assert "parameter_set_physical_lower_not_positive" in result.rejected_hypotheses[row.key]


def test_witness_name_without_positive_bound_is_not_evidence():
    row = hypothesis()
    invalid = replace(row.decision_witnesses[0], margin_lower=0.0)
    row = replace(row, decision_witnesses=(invalid, *row.decision_witnesses[1:]))
    result = select_region_actions_v3((row,))
    assert result.state == "native"
    assert any("witness_bound_not_positive" in reason
               for reason in result.rejected_hypotheses[row.key])


def test_physical_witness_must_remain_on_frozen_b_support():
    row = hypothesis()
    changed = replace(row.closure_witnesses[0], support_hash="other-check")
    row = replace(row, closure_witnesses=(changed, *row.closure_witnesses[1:]))
    result = select_region_actions_v3((row,))
    assert result.state == "native"
    assert any("witness_support_changed" in reason
               for reason in result.rejected_hypotheses[row.key])


def test_optional_stopping_requires_anytime_valid_evidence():
    row = hypothesis(sequential=SequentialEvidence(
        attempts_seen=3, anytime_valid=False, method="fixed_ci_reused", alpha_spent=0.05,
    ))
    result = select_region_actions_v3((row,))
    assert "optional_stopping_uncontrolled" in result.rejected_hypotheses[row.key]


def test_missing_task_witness_is_probed_without_suppressing_safe_disjoint_region():
    incumbent = hypothesis(name="incumbent", parameters=(parameter(mean=0.5),))
    unknown = hypothesis(
        name="unknown", region="r2", parameters=(parameter(task=False),),
        probe_value=0.2,
    )
    result = select_region_actions_v3((incumbent, unknown))
    assert result.state == "selected"
    assert result.selected_hypotheses == (incumbent.key,)
    assert result.probe_hypotheses == (unknown.key,)


def test_overlapping_intervals_on_same_region_remain_ambiguous():
    first = hypothesis(name="first", parameters=(parameter(mean=0.5, radius=0.2),))
    second = hypothesis(name="second", parameters=(parameter(mean=0.45, radius=0.2),))
    result = select_region_actions_v3((first, second))
    assert result.state == "ambiguous"
    assert set(result.ambiguous_hypotheses) == {first.key, second.key}


def test_local_ambiguity_does_not_suppress_safe_disjoint_region():
    first = hypothesis(name="first", region="r1",
                       parameters=(parameter(mean=0.5, radius=0.2),))
    second = hypothesis(name="second", region="r1",
                        parameters=(parameter(mean=0.45, radius=0.2),))
    safe = hypothesis(name="safe", region="r2",
                      parameters=(parameter(mean=0.6, radius=0.05),))
    result = select_region_actions_v3((first, second, safe))
    assert result.state == "selected"
    assert result.selected_hypotheses == (safe.key,)
    assert set(result.ambiguous_hypotheses) == {first.key, second.key}


def test_probe_value_is_net_of_cost_and_respects_budget():
    unknown = hypothesis(
        name="unknown", parameters=(parameter(task=False),),
        probe_value=10.0, compute=50.0,
    )
    result = select_region_actions_v3(
        (unknown,), config=SelectorV3Config(maximum_compute_seconds=5.0),
    )
    assert result.state == "native"
    assert result.probe_hypotheses == ()
    assert result.rejected_hypotheses[unknown.key] == ("probe_budget_exceeded",)


def test_exact_regional_optimizer_respects_harm_and_conflicts():
    first = hypothesis(name="first", region="r1", pixels=0.2,
                       parameters=(parameter(mean=0.6, harm=0.02),))
    second = hypothesis(name="second", region="r2", pixels=0.2,
                        parameters=(parameter(mean=0.5, harm=0.02),))
    expensive = hypothesis(name="expensive", region="r3", pixels=0.2,
                           parameters=(parameter(mean=0.9, harm=0.02),),
                           compute=50.0)
    result = select_region_actions_v3(
        (first, second, expensive),
        config=SelectorV3Config(maximum_compute_seconds=5.0),
    )
    assert result.state == "selected"
    assert set(result.selected_hypotheses) == {first.key, second.key}
    assert expensive.key not in result.selected_hypotheses


def test_small_harm_policy_shrinks_trust_instead_of_requiring_zero_harm():
    row = hypothesis(parameters=(
        parameter(mean=0.05, radius=0.10, harm=0.02),
    ))
    result = select_region_actions_v3((row,))
    assert result.state == "selected"
    assert 0.05 <= result.trust_scales[row.key] < 1.0
    assert result.total_weighted_harm > 0.0


def test_task_witness_must_use_the_frozen_output_support():
    row = replace(hypothesis(), task_comparison_support_hash="other-support")
    result = select_region_actions_v3((row,))
    assert result.state == "native"
    assert "comparison_support_changed" in result.rejected_hypotheses[row.key]


def test_severe_harm_is_rejected_even_when_mean_utility_is_positive():
    row = hypothesis(parameters=(parameter(mean=1.0, radius=0.1, harm=0.3),))
    result = select_region_actions_v3((row,))
    assert result.state == "native"
    assert "severe_harm_guard" in result.rejected_hypotheses[row.key]


def test_fixed_profile_rejects_multiple_parameter_states_at_construction():
    with pytest.raises(ValueError, match="fixed actions"):
        hypothesis(
            operator="impulse_exact_median3",
            parameters=(parameter("a"), parameter("b")),
        )
