from dataclasses import replace

import numpy as np
import pytest

from stablebridge.physical_repair.epe_verifier_task_bounds import (
    EPEVerifierCalibration,
    independent_verifier_epe_path_bound,
)
from stablebridge.physical_repair.verifier_task_bounds import support_mask_sha256


def calibration(radius: float = 0.1) -> EPEVerifierCalibration:
    return EPEVerifierCalibration(
        verifier_id="native-cycle-epe-v1",
        representation_id="flow-px",
        calibration_version="sealed-dense-same-support-v1",
        calibration_data_hash="calibration-data",
        support_policy_hash="runtime-visible-response-v1",
        mean_epe_radius_upper=radius,
        effective_n=100.0,
        confidence_level=0.95,
        selection_aware=True,
        action_blind=True,
        frozen_before_selection=True,
    )


def outputs():
    native = np.zeros((3, 4, 2), dtype=np.float32)
    verifier = np.ones_like(native)
    candidate = np.full_like(native, 0.5)
    support = np.ones((3, 4), dtype=bool)
    return native, candidate, verifier, support


def make_bound(**kwargs):
    native, candidate, verifier, support = outputs()
    return independent_verifier_epe_path_bound(
        native,
        candidate,
        verifier,
        support,
        action_key="common_gaussian@second#region7",
        control_key="variance-midpoint-s0.5",
        control_selection_receipt_hash="fixed-two-plus-one-eprocess",
        source_input_hash="native-pair",
        verifier_input_hash="native-pair",
        task_support_hash=support_mask_sha256(support),
        task_support_policy_hash="runtime-visible-response-v1",
        calibration=calibration(),
        **kwargs,
    )


def mean_epe(value: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(value - target, axis=-1)))


def test_metric_aligned_bound_is_positive_when_candidate_moves_toward_verifier():
    bound = make_bound()
    assert bound.nominal_verifier_gain(1.0) == pytest.approx(np.sqrt(0.5))
    assert bound.utility_lower(1.0) == pytest.approx(np.sqrt(0.5) - 0.2)
    assert bound.signed_harm_upper(1.0) == pytest.approx(-np.sqrt(0.5) + 0.2)
    assert bound.harm_budget_upper(1.0) == 0.0


def test_epe_lower_and_harm_upper_hold_for_targets_inside_mean_radius():
    native, candidate, verifier, _ = outputs()
    bound = make_bound()
    rng = np.random.default_rng(19)
    for _ in range(100):
        error = rng.normal(size=verifier.shape)
        scale = 0.1 / np.mean(np.linalg.norm(error, axis=-1))
        target = verifier + scale * error
        for beta in (0.05, 0.25, 0.7, 1.0):
            delivered = native + beta * (candidate - native)
            actual_gain = mean_epe(native, target) - mean_epe(delivered, target)
            actual_harm = -actual_gain
            assert bound.utility_lower(beta) <= actual_gain + 1e-12
            assert actual_harm <= bound.signed_harm_upper(beta) + 1e-12


def test_zero_trust_is_exact_native_not_a_radius_penalty():
    bound = make_bound()
    assert bound.utility_lower(0.0) == 0.0
    assert bound.signed_harm_upper(0.0) == 0.0
    assert bound.harm_budget_upper(0.0) == 0.0


def test_large_radius_refuses_to_certify_gain():
    native, candidate, verifier, support = outputs()
    bound = independent_verifier_epe_path_bound(
        native,
        candidate,
        verifier,
        support,
        action_key="common_gaussian@second#region7",
        control_key="full",
        control_selection_receipt_hash="fixed-once",
        source_input_hash="native-pair",
        verifier_input_hash="native-pair",
        task_support_hash=support_mask_sha256(support),
        task_support_policy_hash="runtime-visible-response-v1",
        calibration=calibration(radius=1.0),
    )
    assert bound.utility_lower(1.0) < 0.0
    assert bound.harm_budget_upper(1.0) > 0.0


@pytest.mark.parametrize(
    ("changed", "message"),
    (
        ({"action_blind": False}, "action-blind"),
        ({"selection_aware": False}, "selection-aware"),
        ({"frozen_before_selection": False}, "frozen"),
    ),
)
def test_postselected_or_candidate_aware_calibration_is_rejected(changed, message):
    native, candidate, verifier, support = outputs()
    with pytest.raises(ValueError, match=message):
        independent_verifier_epe_path_bound(
            native,
            candidate,
            verifier,
            support,
            action_key="common_gaussian@second#region7",
            control_key="full",
            control_selection_receipt_hash="fixed-once",
            source_input_hash="native-pair",
            verifier_input_hash="native-pair",
            task_support_hash=support_mask_sha256(support),
            task_support_policy_hash="runtime-visible-response-v1",
            calibration=replace(calibration(), **changed),
        )


def test_runtime_and_calibration_support_policies_must_match():
    native, candidate, verifier, support = outputs()
    with pytest.raises(ValueError, match="support policies differ"):
        independent_verifier_epe_path_bound(
            native,
            candidate,
            verifier,
            support,
            action_key="common_gaussian@second#region7",
            control_key="full",
            control_selection_receipt_hash="fixed-once",
            source_input_hash="native-pair",
            verifier_input_hash="native-pair",
            task_support_hash=support_mask_sha256(support),
            task_support_policy_hash="different-policy",
            calibration=calibration(),
        )


def test_support_content_and_native_source_are_immutable_contracts():
    native, candidate, verifier, support = outputs()
    with pytest.raises(ValueError, match="content hash changed"):
        independent_verifier_epe_path_bound(
            native,
            candidate,
            verifier,
            support,
            action_key="common_gaussian@second#region7",
            control_key="full",
            control_selection_receipt_hash="fixed-once",
            source_input_hash="native-pair",
            verifier_input_hash="native-pair",
            task_support_hash="stale-support",
            task_support_policy_hash="runtime-visible-response-v1",
            calibration=calibration(),
        )
    with pytest.raises(ValueError, match="proposal-independent"):
        independent_verifier_epe_path_bound(
            native,
            candidate,
            verifier,
            support,
            action_key="common_gaussian@second#region7",
            control_key="full",
            control_selection_receipt_hash="fixed-once",
            source_input_hash="native-pair",
            verifier_input_hash="candidate-pair",
            task_support_hash=support_mask_sha256(support),
            task_support_policy_hash="runtime-visible-response-v1",
            calibration=calibration(),
        )


def test_outside_support_changes_do_not_change_the_bound_and_vectors_are_read_only():
    native, candidate, verifier, support = outputs()
    support[:] = False
    support[0, 0] = True
    candidate[2, 3] = -1000.0
    first = independent_verifier_epe_path_bound(
        native,
        candidate,
        verifier,
        support,
        action_key="common_gaussian@second#region7",
        control_key="full",
        control_selection_receipt_hash="fixed-once",
        source_input_hash="native-pair",
        verifier_input_hash="native-pair",
        task_support_hash=support_mask_sha256(support),
        task_support_policy_hash="runtime-visible-response-v1",
        calibration=calibration(),
    )
    candidate[2, 3] = 1000.0
    second = independent_verifier_epe_path_bound(
        native,
        candidate,
        verifier,
        support,
        action_key="common_gaussian@second#region7",
        control_key="full",
        control_selection_receipt_hash="fixed-once",
        source_input_hash="native-pair",
        verifier_input_hash="native-pair",
        task_support_hash=support_mask_sha256(support),
        task_support_policy_hash="runtime-visible-response-v1",
        calibration=calibration(),
    )
    assert first.utility_lower(1.0) == pytest.approx(second.utility_lower(1.0))
    with pytest.raises(ValueError):
        first.candidate_direction[0, 0] = 4.0


def test_output_trust_cannot_leave_the_certified_chord():
    bound = make_bound(maximum_output_trust=0.5)
    assert np.isfinite(bound.utility_lower(0.5))
    with pytest.raises(ValueError, match="outside"):
        bound.utility_lower(0.5001)
