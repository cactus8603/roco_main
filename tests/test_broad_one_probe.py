import numpy as np

from stablebridge.physical_repair.broad_one_probe import (
    build_blur_one_probe_candidate,
    compose_one_probe_delivery,
    flow_support_from_regions,
    route_action_receipts,
)


def test_cross_family_receipts_fail_closed_including_noise_to_motion():
    decision = route_action_receipts(
        blur_selected={
            "action": "common_motion", "hypothesis": "first",
            "parameter": {"length": 5, "angle": 0.0}, "regions": [[0, 0]],
        },
        impulse_supported_endpoints=["first"],
    )
    assert decision.accepted is False
    assert decision.action_id == "native"
    assert decision.reason == "ambiguous_cross_family_physical_receipts"
    assert decision.competing_families == ("blur", "noise_or_impulse")


def test_multi_endpoint_same_family_is_one_candidate():
    decision = route_action_receipts(
        jpeg_supported_endpoints=["second", "first", "second"],
    )
    assert decision.accepted is True
    assert decision.action_family == "jpeg_quality"
    assert decision.action_id == "jpeg_deblock@first+second"


def test_flow_regions_are_explicit_not_uncertainty():
    support = flow_support_from_regions((130, 130), [[0, 0], [64, 64]])
    assert support.dtype == np.float32
    assert int(support.sum()) == 2 * 64 * 64
    assert support[129, 129] == 0.0


def test_blur_candidate_changes_only_authorized_endpoint_and_region():
    rng = np.random.default_rng(17)
    first = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    second = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    flow = np.zeros((128, 128, 2), dtype=np.float32)
    selected = {
        "action": "common_gaussian", "hypothesis": "second",
        "parameter": 1.0, "regions": [[0, 0]],
    }
    candidate = build_blur_one_probe_candidate(first, second, flow, selected)
    assert candidate.modified_endpoints == ("first",)
    assert np.array_equal(candidate.second, second)
    assert np.any(candidate.first[:64, :64] != first[:64, :64])
    assert np.array_equal(candidate.first[64:, :], first[64:, :])
    assert np.array_equal(candidate.first[:, 64:], first[:, 64:])
    assert np.array_equal(candidate.first[candidate.first_support <= 0],
                          first[candidate.first_support <= 0])


def test_one_probe_delivery_separates_response_risk_and_influence_and_caps():
    native = np.zeros((32, 40, 2), dtype=np.float32)
    candidate = np.full_like(native, 10.0)
    native_risk = np.full((32, 40), 2.0, dtype=np.float32)
    candidate_risk = np.full((32, 40), 1.0, dtype=np.float32)
    seed = np.zeros((32, 40), dtype=np.float32)
    seed[10:12, 15:17] = 1.0
    result = compose_one_probe_delivery(
        native, candidate, native_risk, candidate_risk, seed,
        influence_halo_px=2, absolute_update_cap_px=0.25,
    )
    assert np.any(result.response_support)
    assert np.all(result.risk_support)
    assert result.delivery_support.sum() == 36
    assert result.max_update_norm <= 0.25 + 1e-6
    assert result.exact_native_outside is True
    assert np.array_equal(result.flow[~result.delivery_support],
                          native[~result.delivery_support])


def test_risk_veto_reverts_candidate_even_with_response_and_influence():
    native = np.zeros((8, 8, 2), dtype=np.float32)
    candidate = np.ones_like(native)
    native_risk = np.ones((8, 8), dtype=np.float32)
    candidate_risk = np.full((8, 8), 2.0, dtype=np.float32)
    result = compose_one_probe_delivery(
        native, candidate, native_risk, candidate_risk,
        np.ones((8, 8), dtype=np.float32), influence_halo_px=0,
    )
    assert np.any(result.response_support)
    assert not np.any(result.risk_support)
    assert not np.any(result.delivery_support)
    assert np.array_equal(result.flow, native)
