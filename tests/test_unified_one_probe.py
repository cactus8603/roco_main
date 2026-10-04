import numpy as np
import pytest

from stablebridge.physical_repair.unified_one_probe import (
    build_impulse_one_probe_candidate,
    route_action_receipts,
)


def _pair(height=48, width=64):
    yy, xx = np.mgrid[:height, :width]
    first = np.clip(np.rint(np.stack((
        20 + 210 * xx / (width - 1),
        30 + 190 * yy / (height - 1),
        90 + 50 * np.sin((xx + yy) / 9.0),
    ), axis=2)), 0, 255).astype(np.uint8)
    second = first.copy()
    flow = np.zeros((height, width, 2), dtype=np.float32)
    return first, second, flow


def test_impulse_candidate_changes_only_certified_endpoint_support():
    first, second, flow = _pair()
    first[10, 12] = 0
    first[20, 22] = 255
    support = np.zeros(first.shape[:2], dtype=np.float32)
    support[10, 12] = support[20, 22] = 1.0
    candidate = build_impulse_one_probe_candidate(
        first, second, flow, {"first": support},
    )
    changed = np.any(candidate.first != first, axis=2)
    assert candidate.action_family == "noise_or_impulse"
    assert candidate.modified_endpoints == ("first",)
    assert np.all(changed <= (support > 0.0))
    assert changed.sum() == 2
    assert np.array_equal(candidate.second, second)
    assert np.array_equal(candidate.flow_influence_seed > 0.0, changed)


def test_second_endpoint_change_is_pulled_to_flow_coordinates():
    first, second, flow = _pair()
    second[15, 18] = 255
    support = np.zeros(first.shape[:2], dtype=np.float32)
    support[15, 18] = 1.0
    candidate = build_impulse_one_probe_candidate(
        first, second, flow, {"second": support},
    )
    assert candidate.modified_endpoints == ("second",)
    assert candidate.flow_influence_seed[15, 18] == 1.0
    assert not np.any(candidate.first != first)


def test_unactionable_or_invalid_impulse_support_fails_closed():
    first, second, flow = _pair()
    empty = np.zeros(first.shape[:2], dtype=np.float32)
    with pytest.raises(ValueError, match="not actionable"):
        build_impulse_one_probe_candidate(first, second, flow, {"first": empty})
    with pytest.raises(ValueError, match="first and/or second"):
        build_impulse_one_probe_candidate(first, second, flow, {"third": empty})


def test_cross_family_receipts_abstain_in_unified_controller():
    decision = route_action_receipts(
        jpeg_supported_endpoints=("first",),
        impulse_supported_endpoints=("second",),
    )
    assert not decision.accepted
    assert decision.action_id == "native"
    assert decision.competing_families == ("jpeg_quality", "noise_or_impulse")
