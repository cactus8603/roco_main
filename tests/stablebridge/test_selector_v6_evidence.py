from dataclasses import replace

import numpy as np
import pytest

from stablebridge.physical_repair.selector_v6_evidence import (
    ACTION_OPERATOR_SIGNATURES,
    ActionSupportDecomposition,
    SequentialProbeReceipt,
    TypedPhysicalLawReceipt,
    validate_action_operator_signatures,
)


def law(**kwargs) -> TypedPhysicalLawReceipt:
    values = dict(
        action_identity="common_gaussian@second",
        law_id="log_otf_quadratic",
        role="invariant",
        null_id="identity_or_non_gaussian_otf",
        statistic_id="heldout_log_otf_curvature_lcb",
        unit="log_amplitude_per_cycles2",
        direction="greater_is_better",
        margin_lower=0.2,
        certificate_support_hash="b-fold-support",
        parameter_fit_support_hash="a-fold-support",
        latent_set_hash="theta-set",
        evaluated_latent_set_hash="theta-set",
        latent_parameter_keys=("sigma_interval", "endpoint"),
        worst_case_member_key="sigma_upper@endpoint_second",
        optimization_receipt_hash="interval-enumeration-receipt",
        deterministic=False,
        effective_n=48.0,
        hypotheses_tested=6,
        correction_method="simultaneous_bound",
        familywise_error_rate=0.05,
        assumption_veto_margins={"texture_support": 0.1, "warp_error": 0.02},
    )
    values.update(kwargs)
    return TypedPhysicalLawReceipt(**values)


def sequential(**kwargs) -> SequentialProbeReceipt:
    values = dict(
        policy_hash="two-plus-one-policy",
        trace_hash="actual-control-trace",
        support_policy_hash="fixed-support-policy",
        support_content_hash="fixed-support-content",
        method="simultaneous_bound",
        attempted_control_keys=("r1", "r2", "mid"),
        selected_control_key="mid",
        maximum_attempts=3,
        alpha_budget=0.05,
        alpha_spent=0.05,
        e_value=1.0,
        simultaneous_family_size=3,
        stopping_reason="physical_closure_and_information_dominance",
        frozen_before_probes=True,
    )
    values.update(kwargs)
    return SequentialProbeReceipt(**values)


def support_decomposition(*, uncertainty: np.ndarray | None = None,
                          utility: np.ndarray | None = None,
                          delivery: np.ndarray | None = None,
                          eligible: np.ndarray | None = None
                          ) -> ActionSupportDecomposition:
    shape = (3, 4)
    full = np.ones(shape, dtype=bool)
    physical = full.copy(); physical[0, 0] = False
    evidence = full.copy(); evidence[0, 1] = False
    visibility = full.copy(); visibility[0, 2] = False
    response = full.copy(); response[0, 3] = False
    risk = full.copy(); risk[1, 0] = False
    collision = np.zeros(shape, dtype=bool); collision[1, 1] = True
    fold = np.zeros(shape, dtype=bool); fold[1, 2] = True
    uncovered = np.zeros(shape, dtype=bool); uncovered[1, 3] = True
    influence = full.copy(); influence[2, 0] = False
    expected_eligible = (
        physical & evidence & visibility & response & risk
        & ~collision & ~fold & ~uncovered & influence
    )
    if utility is None:
        utility = np.ones(shape, dtype=np.float32)
        utility[2, 1] = -0.1
    expected_delivery = expected_eligible & (utility > 0.0)
    return ActionSupportDecomposition(
        source_input_hash="native-input",
        support_policy_hash="response-risk-visibility-v1",
        coordinate_frame="flow_native",
        transport_receipt_hash="transport-replay",
        uncertainty=(
            np.linspace(0.0, 1.0, num=12, dtype=np.float32).reshape(shape)
            if uncertainty is None else uncertainty
        ),
        physical_applicability=physical,
        observable_evidence=evidence,
        visibility=visibility,
        candidate_response=response,
        candidate_risk_nonincrease=risk,
        collision=collision,
        local_fold=fold,
        uncovered=uncovered,
        influence_support=influence,
        candidate_eligible_support=(
            expected_eligible if eligible is None else eligible
        ),
        signed_utility_lower=utility,
        certified_delivery_support=(
            expected_delivery if delivery is None else delivery
        ),
    )


def test_operator_signatures_cover_every_profile_and_preserve_path_laws():
    validate_action_operator_signatures()
    assert set(ACTION_OPERATOR_SIGNATURES) == {
        "impulse_exact_median3", "impulse_median3", "wiener3",
        "common_disk", "common_gaussian", "common_motion", "jpeg_deblock",
        "pixelate", "rank3_pair",
    }
    assert ACTION_OPERATOR_SIGNATURES["common_gaussian"].algebraic_structure == (
        "convolution_semigroup"
    )
    assert ACTION_OPERATOR_SIGNATURES["common_disk"].algebraic_structure.endswith(
        "nonsemigroup"
    )
    assert ACTION_OPERATOR_SIGNATURES["common_motion"].influence_geometry.startswith(
        "anisotropic"
    )


def test_typed_law_keeps_null_units_theta_and_assumption_vetoes():
    receipt = law()
    assert receipt.passed
    assert receipt.unit == "log_amplitude_per_cycles2"
    assert receipt.worst_case_member_key == "sigma_upper@endpoint_second"
    assert len(receipt.assumption_receipt_hash) == 64


def test_typed_law_rejects_partial_theta_and_same_fold_reuse():
    with pytest.raises(ValueError, match="complete latent set"):
        law(evaluated_latent_set_hash="only-best-sigma")
    with pytest.raises(ValueError, match="fit and certificate"):
        law(parameter_fit_support_hash="b-fold-support")


def test_typed_law_requires_multiplicity_and_never_reads_outcomes():
    with pytest.raises(ValueError, match="simultaneous correction"):
        law(correction_method="fixed_single")
    with pytest.raises(ValueError, match="truth"):
        law(ground_truth_or_outcome_read=True)


def test_failed_assumption_is_a_real_failed_receipt_not_a_missing_feature():
    receipt = law(assumption_veto_margins={"warp_error": -0.01})
    assert not receipt.passed


def test_deterministic_law_has_no_fake_fit_fold():
    receipt = law(
        law_id="dct_cell_inclusion",
        role="closure",
        direction="inside_set",
        parameter_fit_support_hash=None,
        deterministic=True,
        effective_n=0.0,
    )
    assert receipt.passed
    with pytest.raises(ValueError, match="statistical fit fold"):
        replace(receipt, parameter_fit_support_hash="invented-fold")


def test_simultaneous_probe_receipt_covers_every_attempted_control():
    receipt = sequential()
    assert receipt.selected_control_key == "mid"
    with pytest.raises(ValueError, match="does not cover"):
        sequential(simultaneous_family_size=2)
    with pytest.raises(ValueError, match="probe budget"):
        sequential(maximum_attempts=2)


def test_fixed_once_and_eprocess_have_non_boolean_validity_rules():
    fixed = sequential(
        method="fixed_once", attempted_control_keys=("full",),
        selected_control_key="full", maximum_attempts=1,
        alpha_spent=0.0, simultaneous_family_size=1,
    )
    assert fixed.method == "fixed_once"
    with pytest.raises(ValueError, match="fixed-once"):
        replace(fixed, attempted_control_keys=("full", "half"), maximum_attempts=2)
    accepted = sequential(
        method="e_process", e_value=20.0, simultaneous_family_size=1,
    )
    assert accepted.e_value == 20.0
    with pytest.raises(ValueError, match="did not cross"):
        sequential(method="e_process", e_value=19.99, simultaneous_family_size=1)


def test_probe_policy_must_be_frozen_and_cannot_read_task_outcomes():
    with pytest.raises(ValueError, match="not frozen"):
        sequential(frozen_before_probes=False)
    with pytest.raises(ValueError, match="truth"):
        sequential(ground_truth_or_outcome_read=True)


def test_support_maps_have_exact_distinct_semantics():
    support = support_decomposition()
    assert support.candidate_fraction > support.delivery_fraction
    assert not support.physical_applicability[0, 0]
    assert support.uncertainty[0, 0] == 0.0
    assert not support.certified_delivery_support[2, 1]
    assert support.candidate_support_hash != support.delivery_support_hash


def test_uncertainty_never_changes_eligibility_or_delivery_algebra():
    low = support_decomposition(uncertainty=np.zeros((3, 4), dtype=np.float32))
    high = support_decomposition(uncertainty=np.ones((3, 4), dtype=np.float32))
    assert low.uncertainty_hash != high.uncertainty_hash
    assert np.array_equal(
        low.candidate_eligible_support, high.candidate_eligible_support,
    )
    assert np.array_equal(
        low.certified_delivery_support, high.certified_delivery_support,
    )


def test_candidate_eligibility_cannot_silently_include_collision_or_missing_evidence():
    valid = support_decomposition()
    bad = valid.candidate_eligible_support.copy()
    bad[1, 1] = True
    with pytest.raises(ValueError, match="exact evidence intersection"):
        support_decomposition(eligible=bad)


def test_delivery_requires_positive_signed_utility_on_the_same_map():
    valid = support_decomposition()
    bad = valid.certified_delivery_support.copy()
    bad[2, 1] = True
    with pytest.raises(ValueError, match="intersected with utility"):
        support_decomposition(delivery=bad)


def test_support_maps_are_immutable_after_hashing():
    support = support_decomposition()
    with pytest.raises(ValueError):
        support.visibility[0, 0] = False
    with pytest.raises(ValueError):
        support.signed_utility_lower[0, 0] = -1.0
