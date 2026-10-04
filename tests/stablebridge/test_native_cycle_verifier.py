import numpy as np
import pytest

from stablebridge.physical_repair.native_cycle_verifier import (
    native_cycle_verifier,
)
from stablebridge.physical_repair.verifier_task_bounds import (
    VerifierCalibration,
    independent_verifier_squared_l2_bound,
    support_mask_sha256,
)


def test_constant_translation_cycle_recovers_forward_verifier():
    forward = np.zeros((5, 7, 2), np.float32)
    reverse = np.zeros_like(forward)
    forward[..., 0] = 1.0
    reverse[..., 0] = -1.0
    evidence = native_cycle_verifier(forward, reverse)
    assert np.all(evidence.valid_support[:, :-1])
    assert not np.any(evidence.valid_support[:, -1])
    assert np.array_equal(
        evidence.verifier_flow[evidence.valid_support],
        forward[evidence.valid_support],
    )
    assert evidence.action_blind


def test_nonfinite_reverse_neighbour_invalidates_bilinear_sample():
    forward = np.zeros((4, 5, 2), np.float32)
    forward[..., 0] = 0.5
    reverse = -forward
    reverse[1, 2] = np.nan
    evidence = native_cycle_verifier(forward, reverse)
    assert not evidence.valid_support[1, 1]
    assert not evidence.valid_support[1, 2]
    assert np.all(evidence.verifier_flow[~evidence.valid_support] == 0.0)


def test_shape_mismatch_fails_closed():
    with pytest.raises(ValueError, match="share HxWx2"):
        native_cycle_verifier(
            np.zeros((3, 4, 2), np.float32),
            np.zeros((3, 5, 2), np.float32),
        )


def test_cycle_verifier_can_feed_action_specific_bound_on_fixed_intersection():
    truth = np.zeros((4, 6, 2), np.float32)
    truth[..., 0] = 1.0
    native = truth.copy()
    native[..., 0] = 0.0
    reverse = np.zeros_like(native)
    reverse[..., 0] = -1.0
    cycle = native_cycle_verifier(native, reverse)
    declared = np.ones(native.shape[:2], bool)
    support = declared & cycle.valid_support
    candidate = native.copy()
    candidate[..., 0] = 0.75
    calibration = VerifierCalibration(
        verifier_id="native-cycle-v1", representation_id="flow-px",
        calibration_version="test", calibration_data_hash="cal",
        support_policy_hash="fixed-intersection", error_radius_upper=0.1,
        effective_n=50.0, confidence_level=0.9, selection_aware=True,
        action_blind=True, frozen_before_selection=True,
    )
    witness = independent_verifier_squared_l2_bound(
        native, candidate, cycle.verifier_flow, support,
        action_key="common_motion@first#length17", parameter_key="length17",
        source_input_hash="native-pair", verifier_input_hash="native-pair",
        task_support_hash=support_mask_sha256(support),
        calibration=calibration,
    )
    assert witness.task_directional_gain_lower > 0.0


def test_cycle_closure_is_not_claimed_as_self_calibrating():
    forward = np.zeros((3, 4, 2), np.float32)
    reverse = np.zeros_like(forward)
    evidence = native_cycle_verifier(forward, reverse)
    assert not hasattr(evidence, "error_radius_upper")
