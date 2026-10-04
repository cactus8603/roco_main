from dataclasses import replace

import pytest

from stablebridge.physical_repair.action_profiles import ACTION_PHYSICS_PROFILES
from stablebridge.physical_repair.contracts import ActionSpec
from stablebridge.physical_repair.selector_v4 import (
    ActionConditionalRiskBound,
    ActionSystemHypothesis,
    DirectionalParameterBound,
    EProcessEvidence,
    FootprintEvidence,
    SelectorV4Config,
    select_action_systems_v4,
)


def footprint(region="r1", spill=0.0, output_spill=0.0):
    return FootprintEvidence(
        source_input_hash="source", physical_support_hash=f"phys-{region}",
        declared_change_support_hash=f"declared-{region}",
        observed_change_support_hash=f"changed-{region}",
        influence_support_hash=f"influence-{region}",
        output_support_hash=f"output-{region}", coordinate_frame="first_native",
        changed_fraction=0.1, influence_fraction=0.2, output_fraction=0.1,
        changed_outside_declared_fraction=spill,
        output_outside_influence_fraction=output_spill, halo_radius_px=4.0,
    )


def bound(region="r1", key="theta", physical=0.2, task=0.4,
          task_curvature=0.0, norm=1.0, remainder=0.0, strength=1.0):
    return DirectionalParameterBound(
        key=key, physical_support_hash=f"phys-{region}",
        task_support_hash=f"output-{region}",
        physical_directional_gain_lower=physical,
        physical_curvature_upper=0.1,
        task_directional_gain_lower=task,
        task_curvature_upper=task_curvature,
        action_norm_upper=norm,
        path_remainder_quadratic_upper=remainder,
        maximum_strength=strength,
    )


def risk(linear=0.02, quadratic=0.0, severe_linear=0.02,
         severe_quadratic=0.0, aware=True, n=50.0):
    return ActionConditionalRiskBound(
        action_identity="jpeg_deblock@first", calibration_support_hash="cal",
        calibration_version="test", effective_n=n, selection_aware=aware,
        harm_linear_upper=linear, harm_quadratic_upper=quadratic,
        severe_linear_upper=severe_linear,
        severe_quadratic_upper=severe_quadratic,
    )


def hypothesis(name="a", region="r1", bounds=None, footprint_row=None,
               confounds=True, inverse=0.2, risk_row=None,
               sequential=EProcessEvidence(), overlaps=(), interactions=None,
               pixels=0.1):
    action = ActionSpec(
        "jpeg_deblock", "test", "image", "first", "first", "first_native",
    )
    profile = ACTION_PHYSICS_PROFILES[action.operator_id]
    return ActionSystemHypothesis(
        key=f"jpeg_deblock@first#{name}", action=action, region_key=region,
        physical_status="supported", source_input_hash="source",
        directional_bounds=tuple(bounds or (bound(region),)),
        footprint=footprint_row or footprint(region),
        information_retention_lower=0.8, visibility_certified=True,
        collision_fraction=0.0, collision_policy="none",
        forward_closure_margin_lower=0.2,
        inverse_closure_margin_lower=inverse,
        confound_rejection_margins=(
            {key: 0.1 for key in profile.dangerous_confounds} if confounds else {}
        ),
        risk=risk_row or risk(), pixel_fraction=pixels,
        expected_compute_seconds=0.1, extra_matcher_trajectories=1,
        sequential=sequential, overlap_keys=tuple(overlaps),
        interaction_quadratic_upper=dict(interactions or {}),
    )


def test_mechanism_presence_without_task_action_direction_falls_back():
    row = hypothesis(bounds=(bound(task=0.0),))
    decision = select_action_systems_v4((row,))
    assert decision.state == "native"
    assert "task_action_direction_not_positive" in decision.rejected_hypotheses[row.key]


def test_trust_radius_is_quadratic_optimum_not_confidence_ratio():
    row = hypothesis(bounds=(bound(task=0.4, task_curvature=2.0),))
    decision = select_action_systems_v4((row,))
    assert decision.state == "selected"
    assert decision.trust_scales[row.key] == pytest.approx(0.2)
    assert decision.certified_utility_lowers[row.key] == pytest.approx(0.04)


def test_physical_curvature_prevents_overrepair_along_action_path():
    directional = replace(
        bound(task=0.4), physical_directional_gain_lower=0.2,
        physical_curvature_upper=2.0,
    )
    row = hypothesis(bounds=(directional,))
    decision = select_action_systems_v4((row,))
    assert decision.state == "selected"
    assert decision.trust_scales[row.key] == pytest.approx(0.2)


def test_lower_feasible_boundary_is_considered_when_vertex_is_too_small():
    row = hypothesis(bounds=(bound(task=0.03, task_curvature=2.0),))
    decision = select_action_systems_v4(
        (row,), config=SelectorV4Config(minimum_trust_alpha=0.02),
    )
    assert decision.state == "selected"
    assert decision.trust_scales[row.key] == pytest.approx(0.02)


def test_harm_envelope_caps_radius_but_allows_small_nonzero_harm():
    row = hypothesis(risk_row=risk(linear=1.0, severe_linear=0.1))
    decision = select_action_systems_v4((row,))
    assert decision.state == "selected"
    assert decision.trust_scales[row.key] == pytest.approx(0.05)
    assert 0.0 < decision.total_weighted_harm <= 0.05


def test_every_declared_dangerous_confound_needs_positive_veto():
    row = hypothesis(confounds=False)
    decision = select_action_systems_v4((row,))
    assert decision.state == "native"
    reasons = decision.rejected_hypotheses[row.key]
    assert any(reason.startswith("missing_confound_veto") for reason in reasons)
    assert any(reason.startswith("confound_not_rejected") for reason in reasons)


def test_real_footprint_spill_fails_closed():
    row = hypothesis(footprint_row=footprint(spill=0.001))
    decision = select_action_systems_v4((row,))
    assert "action_changed_outside_declared_support" in decision.rejected_hypotheses[row.key]


def test_action_conditional_risk_must_be_selection_aware():
    row = hypothesis(risk_row=risk(aware=False))
    decision = select_action_systems_v4((row,))
    assert "risk_not_selection_aware" in decision.rejected_hypotheses[row.key]


def test_adaptive_retry_uses_actual_e_process_threshold():
    weak = hypothesis(sequential=EProcessEvidence(
        attempts_seen=3, e_value=19.9, null_name="no_gain",
        support_hash="phys-r1", increments_hash="increments",
    ))
    rejected = select_action_systems_v4((weak,))
    assert "adaptive_probe_e_process_insufficient" in rejected.rejected_hypotheses[weak.key]
    strong = replace(weak, sequential=replace(weak.sequential, e_value=20.0))
    accepted = select_action_systems_v4((strong,))
    assert accepted.state == "selected"


def test_parameter_set_uses_worst_directional_member():
    row = hypothesis(bounds=(
        bound(key="lo", task=0.4), bound(key="hi", task=-0.01),
    ))
    decision = select_action_systems_v4((row,))
    assert "task_action_direction_not_positive" in decision.rejected_hypotheses[row.key]


def test_overlap_without_measured_interaction_cannot_compose():
    left_key = "jpeg_deblock@first#left"
    right_key = "jpeg_deblock@first#right"
    left = hypothesis(name="left", region="r1", overlaps=(right_key,))
    right = hypothesis(name="right", region="r2", overlaps=(left_key,))
    decision = select_action_systems_v4((left, right))
    assert decision.state == "selected"
    assert len(decision.selected_hypotheses) == 1


def test_bounded_pair_interaction_is_charged_to_composition():
    left_key = "jpeg_deblock@first#left"
    right_key = "jpeg_deblock@first#right"
    left = hypothesis(
        name="left", region="r1", overlaps=(right_key,),
        interactions={right_key: 0.01},
    )
    right = hypothesis(
        name="right", region="r2", overlaps=(left_key,),
        interactions={left_key: 0.01},
    )
    decision = select_action_systems_v4((left, right))
    assert set(decision.selected_hypotheses) == {left_key, right_key}
    assert decision.total_interaction_upper == pytest.approx(0.01)
    assert decision.total_certified_utility_lower == pytest.approx(0.07)


def test_conditional_reversibility_requires_inverse_closure():
    row = hypothesis(inverse=0.0)
    decision = select_action_systems_v4((row,))
    assert "conditional_inverse_closure_not_positive" in decision.rejected_hypotheses[row.key]


def test_risk_calibration_identity_must_match_action_endpoint():
    wrong = replace(risk(), action_identity="jpeg_deblock@second")
    with pytest.raises(ValueError, match="action/endpoint conditional"):
        hypothesis(risk_row=wrong)
