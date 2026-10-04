from dataclasses import replace

import numpy as np
import pytest

from stablebridge.physical_repair.candidate_action_bank import (
    FrozenActionReceipt,
    OPTICAL_FLOW_ACTION_BANK,
    OPTICAL_FLOW_ACTION_BANK_HASH,
    OPTICAL_FLOW_MINIMAL_PILOT_ACTION_IDS,
    OPTICAL_FLOW_MINIMAL_PILOT_BANK,
    OPTICAL_FLOW_MINIMAL_PILOT_BANK_HASH,
    OPTICAL_FLOW_OPENED_CANDIDATE_ACTION_IDS,
    OPTICAL_FLOW_OPENED_CANDIDATE_FAMILIES,
    OPTICAL_FLOW_OPENED_CANDIDATE_BANK,
    OPTICAL_FLOW_OPENED_CANDIDATE_BANK_HASH,
    OPTICAL_NATIVE_ACTION_ID,
    prepare_optical_pair_action,
    validate_frozen_action_bank,
)


EXPECTED_ACTION_IDS = {
    "CSB/OF/SEA-RAFT/action/V3-gaussian-s0p5",
    "CSB/OF/SEA-RAFT/action/P1-gaussian-s1",
    "CSB/OF/SEA-RAFT/action/R1-gaussian-s1p5",
    "CSB/OF/SEA-RAFT/action/R2-gaussian-s2",
    "CSB/OF/SEA-RAFT/action/P2-unsharp-s1",
    "CSB/OF/SEA-RAFT/action/R3-unsharp-a1",
    "CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5",
    "CSB/OF/SEA-RAFT/action/V3-radiometric-blend-l0p25",
    "CSB/OF/SEA-RAFT/action/V3-radiometric-blend-l0p5",
    "CSB/OF/SEA-RAFT/action/V3-radiometric-blend-l0p75",
    "CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1",
    "CSB/OF/SEA-RAFT/action/R5-joint-channel-percentile-s1",
    "CSB/OF/SEA-RAFT/action/R6-joint-percentile-s5",
    "CSB/OF/SEA-RAFT/action/P4-iters8",
    "CSB/OF/SEA-RAFT/action/R7-iters12",
}


def canonical_pair() -> np.ndarray:
    rng = np.random.default_rng(20261003)
    return np.ascontiguousarray(
        rng.integers(0, 256, size=(2, 3, 37, 53), dtype=np.uint8)
        .astype(np.float32) / 255.0,
    )


def test_optical_bank_is_exactly_frozen_and_nonauthoritative() -> None:
    validate_frozen_action_bank()
    assert set(OPTICAL_FLOW_ACTION_BANK) == EXPECTED_ACTION_IDS
    assert OPTICAL_FLOW_ACTION_BANK_HASH == (
        "2dfc67effd4f8d06910c1217d717da45436c778084b8be8bfb95aeae212bb9f2"
    )
    for action_id, arm in OPTICAL_FLOW_ACTION_BANK.items():
        payload = arm.selector_payload()
        assert set(payload) == {
            "action_id", "operator", "strength", "endpoint", "support",
            "cost", "availability", "receipt",
        }
        assert payload["action_id"] == action_id
        assert arm.endpoint == "both"
        assert arm.cost.observed_rows == 1200
        assert arm.cost.extra_matcher_trajectories == 1
        assert arm.receipt.production_authority is False
        assert arm.availability.endswith("TEST_ONLY")


def test_minimal_pilot_is_four_nonnative_mechanisms_plus_external_native() -> None:
    assert OPTICAL_FLOW_MINIMAL_PILOT_ACTION_IDS == (
        "CSB/OF/SEA-RAFT/action/P1-gaussian-s1",
        "CSB/OF/SEA-RAFT/action/P2-unsharp-s1",
        "CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1",
        "CSB/OF/SEA-RAFT/action/P4-iters8",
    )
    assert tuple(OPTICAL_FLOW_MINIMAL_PILOT_BANK) == (
        OPTICAL_FLOW_MINIMAL_PILOT_ACTION_IDS
    )
    assert OPTICAL_NATIVE_ACTION_ID not in OPTICAL_FLOW_MINIMAL_PILOT_BANK
    assert OPTICAL_FLOW_MINIMAL_PILOT_BANK_HASH == (
        "c65fd16c8607c7868998806b78f935c617a445d7e4257d997d81fb2123e4c2a1"
    )


def test_opened_candidate_bank_is_e278_three_family_four_control_set() -> None:
    assert OPTICAL_FLOW_OPENED_CANDIDATE_ACTION_IDS == (
        "CSB/OF/SEA-RAFT/action/P1-gaussian-s1",
        "CSB/OF/SEA-RAFT/action/R2-gaussian-s2",
        "CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5",
        "CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1",
    )
    assert tuple(OPTICAL_FLOW_OPENED_CANDIDATE_BANK) == (
        OPTICAL_FLOW_OPENED_CANDIDATE_ACTION_IDS
    )
    assert tuple(OPTICAL_FLOW_OPENED_CANDIDATE_FAMILIES) == (
        "optical.lowpass_hf_suppression.v1",
        "optical.detail_recovery_unsharp.v1",
        "optical.joint_radiometry.v1",
    )
    assert tuple(
        action
        for members in OPTICAL_FLOW_OPENED_CANDIDATE_FAMILIES.values()
        for action in members
    ) == OPTICAL_FLOW_OPENED_CANDIDATE_ACTION_IDS
    assert OPTICAL_NATIVE_ACTION_ID not in OPTICAL_FLOW_OPENED_CANDIDATE_BANK
    assert all(
        arm.availability.endswith("TEST_ONLY")
        and arm.receipt.production_authority is False
        for arm in OPTICAL_FLOW_OPENED_CANDIDATE_BANK.values()
    )
    assert OPTICAL_FLOW_OPENED_CANDIDATE_BANK_HASH == (
        "f112ec81c24cc9c5aac70d0ab5fc919c9dd127d2396a9dbb85c46501cada8070"
    )


def test_candidate_bank_is_exposed_by_the_main_physical_repair_api() -> None:
    from stablebridge import physical_repair

    assert physical_repair.OPTICAL_FLOW_ACTION_BANK is OPTICAL_FLOW_ACTION_BANK
    assert physical_repair.prepare_optical_pair_action is prepare_optical_pair_action


def test_every_image_operator_is_finite_and_compute_actions_are_exact_identity() -> None:
    pair = canonical_pair()
    for action_id, arm in OPTICAL_FLOW_ACTION_BANK.items():
        result = prepare_optical_pair_action(pair, action_id)
        assert result.action_id == action_id
        assert result.images.shape == pair.shape
        assert result.images.dtype == np.float32
        assert np.isfinite(result.images).all()
        assert result.extra_matcher_trajectories == 1
        if arm.operator == "matcher_iteration_override":
            assert result.images is pair
            assert result.matcher_iterations_override == int(arm.strength)
        else:
            assert result.matcher_iterations_override is None
            assert float(result.images.min()) >= 0.0
            assert float(result.images.max()) <= 1.0


def test_native_is_exact_zero_cost_fallback_but_not_a_duplicate_candidate() -> None:
    pair = canonical_pair()
    result = prepare_optical_pair_action(pair, OPTICAL_NATIVE_ACTION_ID)
    assert result.images is pair
    assert result.matcher_iterations_override is None
    assert result.extra_matcher_trajectories == 0
    assert OPTICAL_NATIVE_ACTION_ID not in OPTICAL_FLOW_ACTION_BANK


def test_operator_semantics_are_distinct() -> None:
    pair = canonical_pair()
    smooth = prepare_optical_pair_action(
        pair, "CSB/OF/SEA-RAFT/action/P1-gaussian-s1",
    ).images
    sharpen = prepare_optical_pair_action(
        pair, "CSB/OF/SEA-RAFT/action/P2-unsharp-s1",
    ).images
    radiometry = prepare_optical_pair_action(
        pair, "CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1",
    ).images
    assert not np.array_equal(smooth, pair)
    assert not np.array_equal(sharpen, pair)
    assert not np.array_equal(radiometry, pair)
    assert float(np.var(smooth)) < float(np.var(pair))
    assert not np.array_equal(smooth, sharpen)


def test_quantized_actions_fail_closed_on_non_uint8_derived_input() -> None:
    for action_id in (
        "CSB/OF/SEA-RAFT/action/R5-joint-channel-percentile-s1",
        "CSB/OF/SEA-RAFT/action/R6-joint-percentile-s5",
    ):
        assert OPTICAL_FLOW_ACTION_BANK[action_id].support == (
            "whole_pair_uint8_derived_float32_v1"
        )
    pair = canonical_pair().copy()
    pair[0, 0, 0, 0] += np.float32(1e-3)
    with pytest.raises(ValueError, match="uint8-derived"):
        prepare_optical_pair_action(
            pair, "CSB/OF/SEA-RAFT/action/R6-joint-percentile-s5",
        )


def test_unknown_action_bad_tensor_hash_drift_and_authority_fail_closed() -> None:
    pair = canonical_pair()
    with pytest.raises(ValueError, match="unknown frozen Optical action"):
        prepare_optical_pair_action(pair, "missing")
    with pytest.raises(ValueError, match="expected finite float32 pair"):
        prepare_optical_pair_action(pair.astype(np.float64), OPTICAL_NATIVE_ACTION_ID)
    arm = next(iter(OPTICAL_FLOW_ACTION_BANK.values()))
    with pytest.raises(ValueError, match="arm hash drift"):
        replace(arm, arm_hash="0" * 64)
    with pytest.raises(ValueError, match="cannot grant production authority"):
        replace(
            arm.receipt,
            production_authority=True,
        )
    with pytest.raises(ValueError, match="production authority"):
        FrozenActionReceipt(
            receipt_id="forged",
            operator_source_sha256="0" * 64,
            evidence_sha256="1" * 64,
            evidence_scope="forged",
            production_authority=True,
        )
