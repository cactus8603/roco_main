import numpy as np
import pytest

from stablebridge.physical_repair.noise_visibility_receipts import (
    BidirectionalVisibilityReceipt,
    bidirectional_visibility_receipt,
    dis_bidirectional_flow,
    visibility_qualified_median_responses,
)


def _pair(height=128, width=160):
    yy, xx = np.mgrid[:height, :width]
    base = np.stack((
        20.0 + 210.0 * xx / (width - 1),
        30.0 + 190.0 * yy / (height - 1),
        50.0 + 80.0 * np.sin((xx + yy) / 17.0),
    ), axis=2)
    image = np.clip(np.rint(base), 0, 255).astype(np.uint8)
    flow = np.zeros((height, width, 2), np.float32)
    return image.copy(), image.copy(), flow.copy(), flow.copy()


def test_zero_flow_pair_has_bidirectional_visibility():
    first, _, forward, backward = _pair()
    receipt = bidirectional_visibility_receipt(forward, backward)
    assert receipt.first_visible_pixels == receipt.first_valid_pixels
    assert receipt.second_visible_pixels == receipt.second_valid_pixels
    assert receipt.first_visible_fraction == receipt.second_visible_fraction == 1.0
    assert receipt.first_visible_mask.shape == first.shape[:2]


def test_inconsistent_reverse_flow_is_rejected_by_visibility_proxy():
    _, _, forward, backward = _pair()
    backward[..., 0] = 6.0
    receipt = bidirectional_visibility_receipt(forward, backward)
    assert receipt.first_visible_fraction == 0.0
    assert receipt.first_fb_error_p50_px == pytest.approx(6.0)


def test_sparse_impulse_has_positive_visible_exact_response():
    first, second, forward, backward = _pair()
    rng = np.random.default_rng(17)
    locations = rng.choice(first.shape[0] * first.shape[1], 500, replace=False)
    flat = first.reshape(-1, 3)
    flat[locations[:250]] = 0
    flat[locations[250:]] = 255
    visibility, responses = visibility_qualified_median_responses(
        first, second, forward, backward,
    )
    assert visibility.first_visible_fraction == 1.0
    assert responses["first"].response_supported
    assert responses["first"].fit_gain_lower_255 > 0.0
    assert responses["first"].check_gain_lower_255 > 0.0
    assert not responses["second"].response_supported


def test_persistent_thin_structure_is_not_a_positive_action_response():
    first, second, forward, backward = _pair()
    first[:, 50] = 0
    second[:, 50] = 0
    _, responses = visibility_qualified_median_responses(
        first, second, forward, backward,
    )
    assert not responses["first"].response_supported
    assert not responses["second"].response_supported


def test_visibility_removes_changed_support_under_inconsistent_geometry():
    first, second, forward, backward = _pair()
    first[30:90:3, 30:120:3] = 0
    backward[..., 0] = 8.0
    _, responses = visibility_qualified_median_responses(
        first, second, forward, backward,
    )
    assert responses["first"].pre_visibility_support_pixels > 0
    assert responses["first"].visible_support_pixels == 0
    assert not responses["first"].response_supported


def test_visibility_receipt_rejects_hash_and_outcome_tampering():
    _, _, forward, backward = _pair()
    receipt = bidirectional_visibility_receipt(forward, backward)
    payload = dict(receipt.__dict__)
    payload["first_visible_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        BidirectionalVisibilityReceipt(**payload)
    payload = dict(receipt.__dict__)
    payload["ground_truth_or_outcome_read"] = True
    with pytest.raises(ValueError, match="labels or outcomes"):
        BidirectionalVisibilityReceipt(**payload)


def test_invalid_inputs_fail_closed():
    first, second, forward, backward = _pair()
    with pytest.raises(ValueError, match="uint8 RGB"):
        visibility_qualified_median_responses(
            first.astype(np.float32), second, forward, backward,
        )
    with pytest.raises(ValueError, match="finite HxWx2"):
        bidirectional_visibility_receipt(forward[..., :1], backward)
    with pytest.raises(ValueError, match="at least 16"):
        visibility_qualified_median_responses(
            first, second, forward, backward, tile_size=8,
        )


def test_fixed_dis_alignment_is_finite_and_deterministic():
    first, second, _, _ = _pair(64, 80)
    second = np.roll(second, 1, axis=1)
    forward_a, backward_a = dis_bidirectional_flow(first, second)
    forward_b, backward_b = dis_bidirectional_flow(first, second)
    assert forward_a.shape == backward_a.shape == (64, 80, 2)
    assert np.isfinite(forward_a).all() and np.isfinite(backward_a).all()
    assert np.array_equal(forward_a, forward_b)
    assert np.array_equal(backward_a, backward_b)
    with pytest.raises(ValueError, match="unknown DIS"):
        dis_bidirectional_flow(first, second, preset="unknown")
