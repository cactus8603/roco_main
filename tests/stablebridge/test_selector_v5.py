from dataclasses import replace

import pytest

from stablebridge.physical_repair.contracts import ActionSpec
from stablebridge.physical_repair.selector_v4 import FootprintEvidence
from stablebridge.physical_repair.selector_v5 import (
    ActionInvariantBound,
    ActionPathHypothesis,
    ControlConditionalRiskBound,
    ExecutedControlPath,
    SelectorV5Config,
    select_action_paths_v5,
)


LATENT = ("impulse-density=[0.01,0.03]",)
REQUIRED = (
    "sparse_local_extremum",
    "paired_correspondence_disagreement",
    "spatial_distribution",
    "heldout_center_prediction",
    "residual_impulse_rate",
)


def _action(endpoint: str = "first") -> ActionSpec:
    return ActionSpec(
        operator_id="impulse_exact_median3", operator_version="v5-test",
        domain="image", hypothesized_degraded_endpoint=endpoint,
        modified_endpoint=endpoint, coordinate_frame=f"{endpoint}_native",
    )


def _footprint(source: str = "source", suffix: str = "half") -> FootprintEvidence:
    return FootprintEvidence(
        source_input_hash=source, physical_support_hash="admission-support",
        declared_change_support_hash=f"declared-{suffix}",
        observed_change_support_hash=f"changed-{suffix}",
        influence_support_hash=f"influence-{suffix}",
        output_support_hash=f"output-{suffix}", coordinate_frame="first_native",
        changed_fraction=0.1, influence_fraction=0.2, output_fraction=0.15,
        changed_outside_declared_fraction=0.0,
        output_outside_influence_fraction=0.0, halo_radius_px=1.0,
    )


def _risk(control: str, support: str, *, harm: float = 0.1
          ) -> ControlConditionalRiskBound:
    return ControlConditionalRiskBound(
        action_identity="impulse_exact_median3@first", control_key=control,
        control_policy_hash=f"policy-{control}", support_policy_hash=support,
        calibration_data_hash="calibration-data", calibration_version="v1",
        effective_n=40.0, selection_aware=True,
        harm_linear_upper=harm, harm_quadratic_upper=0.0,
        severe_linear_upper=0.0, severe_quadratic_upper=0.0,
    )


def _control(key: str, strength: float, *, task_gain: float = 1.0,
             selected_adjusted: bool = True, source: str = "source",
             suffix: str | None = None) -> ExecutedControlPath:
    suffix = suffix or key
    footprint = _footprint(source, suffix)
    return ExecutedControlPath(
        key=key, input_strength=strength,
        input_path_id="observed_to_proposal_rgb_chord_v1",
        control_parameter_hash=f"parameters-{key}",
        repaired_input_hash=f"input-{key}", candidate_output_hash=f"flow-{key}",
        output_support_policy_hash=f"support-policy-{suffix}",
        input_bound_support_hash="admission-support",
        delivery_bound_support_hash=f"output-{suffix}",
        latent_parameter_keys=LATENT, footprint=footprint,
        input_directional_gain_lower=1.0, input_curvature_upper=0.0,
        information_retention_lower=0.8,
        delivery_directional_gain_lower=task_gain,
        delivery_curvature_upper=0.0, output_direction_norm_upper=1.0,
        output_path_remainder_quadratic_upper=0.0,
        maximum_output_trust=1.0,
        risk=_risk(key, f"support-policy-{suffix}"),
        control_selection_adjusted=selected_adjusted,
        expected_compute_seconds=1.0, extra_matcher_trajectories=1,
    )


def _invariants() -> tuple[ActionInvariantBound, ...]:
    return tuple(
        ActionInvariantBound(
            name=name, margin_lower=1.0, support_hash="admission-support",
            latent_parameter_keys=LATENT, effective_n=40.0,
        )
        for name in REQUIRED
    )


def _hypothesis(*controls: ExecutedControlPath, key: str = "h",
                region: str = "r", source: str = "source",
                invariants=None) -> ActionPathHypothesis:
    return ActionPathHypothesis(
        key=f"impulse_exact_median3@first#{key}", action=_action(),
        region_key=region, source_input_hash=source,
        admission_support_hash="admission-support", physical_status="supported",
        latent_parameter_keys=LATENT,
        invariants=_invariants() if invariants is None else invariants,
        confound_rejection_margins={
            "specular_highlight": 1.0, "traffic_light": 1.0,
            "thin_structure": 1.0,
        },
        controls=controls, pixel_fraction=0.1,
    )


def test_selects_exact_input_control_and_separate_output_trust():
    half = _control("input-half", 0.5, task_gain=1.0)
    full = _control("input-full", 1.0, task_gain=-1.0)
    decision = select_action_paths_v5([_hypothesis(half, full)])
    key = "impulse_exact_median3@first#h"
    assert decision.state == "selected"
    assert decision.selected_controls[key] == "input-half"
    assert decision.input_strengths[key] == pytest.approx(0.5)
    # Harm 0.1 * beta <= 0.05: output trust is independently capped at 0.5.
    assert decision.output_trusts[key] == pytest.approx(0.5)
    assert f"{key}/input-full" in decision.rejected_controls


def test_missing_action_specific_invariant_fails_closed():
    controls = (_control("fixed", 1.0),)
    decision = select_action_paths_v5([
        _hypothesis(*controls, invariants=_invariants()[:-1]),
    ])
    assert decision.state == "native"
    reasons = decision.rejected_hypotheses["impulse_exact_median3@first#h"]
    assert "missing_action_invariant:residual_impulse_rate" in reasons


def test_multi_control_search_requires_selection_adjustment():
    decision = select_action_paths_v5([
        _hypothesis(
            _control("left", 0.25, selected_adjusted=False),
            _control("right", 0.75, selected_adjusted=True),
        ),
    ])
    assert decision.state == "native"
    assert "control_search_not_selection_adjusted" in decision.rejected_hypotheses[
        "impulse_exact_median3@first#h"
    ]


def test_risk_must_be_calibrated_on_exact_output_support():
    footprint = _footprint()
    with pytest.raises(ValueError, match="support policy differ"):
        ExecutedControlPath(
            key="half", input_strength=0.5,
            input_path_id="observed_to_proposal_rgb_chord_v1",
            control_parameter_hash="p",
            repaired_input_hash="i", candidate_output_hash="o",
            output_support_policy_hash="expected-support-policy",
            input_bound_support_hash="admission-support",
            delivery_bound_support_hash="output-half",
            latent_parameter_keys=LATENT, footprint=footprint,
            input_directional_gain_lower=1.0, input_curvature_upper=0.0,
            information_retention_lower=0.8,
            delivery_directional_gain_lower=1.0,
            delivery_curvature_upper=0.0, output_direction_norm_upper=1.0,
            output_path_remainder_quadratic_upper=0.0,
            maximum_output_trust=1.0,
            risk=_risk("half", "different-support-policy"),
            control_selection_adjusted=True,
            expected_compute_seconds=1.0, extra_matcher_trajectories=1,
        )


def test_control_must_cover_complete_latent_parameter_set():
    control = replace(_control("fixed", 1.0), latent_parameter_keys=("other",))
    with pytest.raises(ValueError, match="cover the latent parameter set"):
        _hypothesis(control)


def test_every_invariant_must_use_exact_admission_support():
    wrong = replace(_invariants()[0], support_hash="different-region")
    with pytest.raises(ValueError, match="recomputed on the admission support"):
        _hypothesis(
            _control("fixed", 1.0),
            invariants=(wrong, *_invariants()[1:]),
        )


def test_control_physical_support_must_match_admission_support():
    control = _control("fixed", 1.0)
    control = replace(
        control,
        input_bound_support_hash="different-region",
        footprint=replace(
            control.footprint, physical_support_hash="different-region",
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        _hypothesis(control)


@pytest.mark.parametrize(
    "field,expected",
    [
        ("input_bound_support_hash", "input directional bound"),
        ("delivery_bound_support_hash", "delivery utility bound"),
    ],
)
def test_exact_control_bounds_cannot_borrow_another_support(field, expected):
    with pytest.raises(ValueError, match=expected):
        replace(_control("fixed", 1.0), **{field: "different-content"})


def test_same_region_actions_are_alternatives_not_implicit_composition():
    weaker = _hypothesis(_control("weak", 0.5, task_gain=0.5), key="weak")
    stronger = _hypothesis(_control("strong", 0.5, task_gain=1.0), key="strong")
    decision = select_action_paths_v5([weaker, stronger])
    assert decision.state == "selected"
    assert decision.selected_hypotheses == (
        "impulse_exact_median3@first#strong",
    )


def test_control_specific_harm_can_reject_one_path_without_rejecting_action():
    safe = _control("safe", 0.5, task_gain=0.8)
    unsafe = replace(
        _control("unsafe", 1.0, task_gain=2.0),
        risk=_risk("unsafe", "support-policy-unsafe", harm=100.0),
    )
    decision = select_action_paths_v5([
        _hypothesis(safe, unsafe),
    ], config=SelectorV5Config(minimum_output_trust=0.01))
    key = "impulse_exact_median3@first#h"
    assert decision.state == "selected"
    assert decision.selected_controls[key] == "safe"
    assert f"{key}/unsafe" in decision.rejected_controls


def test_unknown_cross_region_relation_cannot_be_composed_as_disjoint():
    left = _hypothesis(
        _control("left-control", 0.5), key="left", region="left-region",
    )
    right = _hypothesis(
        _control("right-control", 0.5), key="right", region="right-region",
    )
    decision = select_action_paths_v5([left, right])
    assert decision.state == "selected"
    assert len(decision.selected_hypotheses) == 1


def test_two_regions_compose_only_with_symmetric_disjointness_proof():
    left_path = "impulse_exact_median3@first#left/left-control"
    right_path = "impulse_exact_median3@first#right/right-control"
    left_control = replace(
        _control("left-control", 0.5), disjoint_path_keys=(right_path,),
    )
    right_control = replace(
        _control("right-control", 0.5), disjoint_path_keys=(left_path,),
    )
    left = _hypothesis(
        left_control, key="left", region="left-region",
    )
    right = _hypothesis(
        right_control, key="right", region="right-region",
    )
    decision = select_action_paths_v5([left, right])
    assert decision.state == "selected"
    assert set(decision.selected_hypotheses) == {
        "impulse_exact_median3@first#left",
        "impulse_exact_median3@first#right",
    }


def test_declared_overlap_requires_an_interaction_bound():
    with pytest.raises(ValueError, match="exactly one interaction bound"):
        replace(
            _control("left-control", 0.5),
            overlap_path_keys=("some/other-path",),
        )


def test_runtime_labels_or_outcomes_are_forbidden():
    with pytest.raises(ValueError, match="cannot read labels"):
        replace(
            _hypothesis(_control("fixed", 1.0)),
            corruption_label_read=True,
        )


def test_nonpositive_input_physics_returns_native():
    control = replace(
        _control("overrepair", 1.0),
        input_directional_gain_lower=1.0,
        input_curvature_upper=3.0,
    )
    decision = select_action_paths_v5([_hypothesis(control)])
    assert decision.state == "native"
    assert "executed_input_path_not_physically_positive" in decision.rejected_controls[
        "impulse_exact_median3@first#h/overrepair"
    ]
