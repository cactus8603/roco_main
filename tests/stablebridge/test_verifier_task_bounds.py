from dataclasses import replace

import numpy as np
import pytest

from stablebridge.physical_repair.verifier_task_bounds import (
    VerifierCalibration,
    independent_verifier_squared_l2_bound,
    support_mask_sha256,
)


def calibration(radius=0.1):
    return VerifierCalibration(
        verifier_id="independent-flow-v1",
        representation_id="flow-px",
        calibration_version="sealed-cal-v1",
        calibration_data_hash="cal-data",
        support_policy_hash="support-policy",
        error_radius_upper=radius,
        effective_n=100.0,
        confidence_level=0.95,
        selection_aware=True,
        action_blind=True,
        frozen_before_selection=True,
    )


def outputs():
    native = np.zeros((2, 3, 2), dtype=np.float32)
    verifier = np.ones_like(native)
    candidate = np.full_like(native, 0.5)
    support = np.ones((2, 3), dtype=bool)
    return native, candidate, verifier, support


def make_bound(**kwargs):
    native, candidate, verifier, support = outputs()
    return independent_verifier_squared_l2_bound(
        native, candidate, verifier, support,
        action_key="rank3_pair@first#r1",
        parameter_key="rank3",
        source_input_hash="source",
        verifier_input_hash="source",
        task_support_hash=support_mask_sha256(support),
        calibration=calibration(),
        **kwargs,
    )


def test_specific_candidate_direction_has_positive_robust_lower_bound():
    witness = make_bound()
    assert witness.nominal_directional_gain == pytest.approx(2.0)
    assert witness.verifier_radius_penalty == pytest.approx(np.sqrt(0.02))
    assert witness.task_directional_gain_lower > 0.0
    assert witness.utility_lower(1.0) > 0.0


def test_bound_is_below_actual_gain_for_every_target_inside_calibrated_ball():
    native, candidate, verifier, support = outputs()
    rng = np.random.default_rng(8)
    for _ in range(100):
        error = rng.normal(size=verifier.shape)
        error *= 0.1 / np.sqrt(np.mean(np.sum(error * error, axis=-1)))
        target = verifier + error
        witness = independent_verifier_squared_l2_bound(
            native, candidate, verifier, support,
            action_key="rank3_pair@first#r1", parameter_key="rank3",
            source_input_hash="source", verifier_input_hash="source",
            task_support_hash=support_mask_sha256(support), calibration=calibration(),
        )
        for alpha in (0.1, 0.35, 1.0):
            delivered = native + alpha * (candidate - native)
            actual = np.mean(np.sum(
                (native - target) ** 2 - (delivered - target) ** 2,
                axis=-1,
            ))
            assert witness.utility_lower(alpha) <= actual + 1e-12


def test_large_verifier_radius_refuses_to_invent_positive_direction():
    native, candidate, verifier, support = outputs()
    witness = independent_verifier_squared_l2_bound(
        native, candidate, verifier, support,
        action_key="rank3_pair@first#r1", parameter_key="rank3",
        source_input_hash="source", verifier_input_hash="source",
        task_support_hash=support_mask_sha256(support), calibration=calibration(radius=2.0),
    )
    assert witness.task_directional_gain_lower < 0.0
    assert witness.utility_lower(0.1) < 0.0


@pytest.mark.parametrize(
    ("changed", "message"),
    (
        ({"action_blind": False}, "action-blind"),
        ({"selection_aware": False}, "selection-aware"),
        ({"frozen_before_selection": False}, "frozen"),
    ),
)
def test_nonindependent_or_postselected_verifier_is_rejected(changed, message):
    native, candidate, verifier, support = outputs()
    with pytest.raises(ValueError, match=message):
        independent_verifier_squared_l2_bound(
            native, candidate, verifier, support,
            action_key="rank3_pair@first#r1", parameter_key="rank3",
            source_input_hash="source", verifier_input_hash="source",
            task_support_hash=support_mask_sha256(support),
            calibration=replace(calibration(), **changed),
        )


def test_verifier_must_use_the_native_source_provenance():
    native, candidate, verifier, support = outputs()
    with pytest.raises(ValueError, match="proposal-independent"):
        independent_verifier_squared_l2_bound(
            native, candidate, verifier, support,
            action_key="rank3_pair@first#r1", parameter_key="rank3",
            source_input_hash="source", verifier_input_hash="candidate",
            task_support_hash=support_mask_sha256(support), calibration=calibration(),
        )


def test_task_bound_attaches_to_v4_without_reusing_physical_score():
    witness = make_bound(maximum_strength=0.5)
    directional = witness.to_directional_parameter_bound(
        physical_support_hash="physical-support",
        physical_directional_gain_lower=0.3,
        physical_curvature_upper=0.2,
    )
    assert directional.physical_directional_gain_lower == pytest.approx(0.3)
    assert directional.task_directional_gain_lower == pytest.approx(
        witness.task_directional_gain_lower
    )
    assert directional.task_support_hash == support_mask_sha256(outputs()[3])
    assert directional.maximum_strength == pytest.approx(0.5)


def test_support_scope_excludes_changes_outside_the_delivered_region():
    native, candidate, verifier, support = outputs()
    support[:] = False
    support[0, 0] = True
    candidate[1, 2] = -1000.0
    local = independent_verifier_squared_l2_bound(
        native, candidate, verifier, support,
        action_key="rank3_pair@first#r1", parameter_key="rank3",
        source_input_hash="source", verifier_input_hash="source",
        task_support_hash=support_mask_sha256(support), calibration=calibration(),
    )
    candidate[1, 2] = 1000.0
    changed_outside = independent_verifier_squared_l2_bound(
        native, candidate, verifier, support,
        action_key="rank3_pair@first#r1", parameter_key="rank3",
        source_input_hash="source", verifier_input_hash="source",
        task_support_hash=support_mask_sha256(support), calibration=calibration(),
    )
    assert local.task_directional_gain_lower == pytest.approx(
        changed_outside.task_directional_gain_lower
    )


def test_declared_task_support_hash_must_match_actual_mask():
    native, candidate, verifier, support = outputs()
    with pytest.raises(ValueError, match="content hash changed"):
        independent_verifier_squared_l2_bound(
            native, candidate, verifier, support,
            action_key="rank3_pair@first#r1", parameter_key="rank3",
            source_input_hash="source", verifier_input_hash="source",
            task_support_hash="stale-mask", calibration=calibration(),
        )
