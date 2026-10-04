import cv2
import numpy as np
import pytest

from stablebridge.physical_repair.noise_action_efficiency import (
    MedianActionEfficiencyReceipt,
    support_aligned_clipping_null_receipt,
    visibility_qualified_median_efficiency,
)


def _pair(height=160, width=192):
    yy, xx = np.mgrid[:height, :width]
    base = np.stack((
        20.0 + 210.0 * xx / (width - 1),
        25.0 + 200.0 * yy / (height - 1),
        50.0 + 70.0 * np.sin((xx + yy) / 21.0),
    ), axis=2)
    image = np.clip(np.rint(base), 0, 255).astype(np.uint8)
    flow = np.zeros((height, width, 2), np.float32)
    return image.copy(), image.copy(), flow.copy(), flow.copy()


def test_sparse_impulse_passes_aligned_null_and_efficiency():
    first, second, forward, backward = _pair()
    rng = np.random.default_rng(7)
    locations = rng.choice(first.shape[0] * first.shape[1], 1600, replace=False)
    flat = first.reshape(-1, 3)
    flat[locations[:800]] = 0
    flat[locations[800:]] = 255
    clipping = support_aligned_clipping_null_receipt(first, endpoint="first")
    _, receipts = visibility_qualified_median_efficiency(
        first, second, forward, backward,
    )
    assert clipping.supported
    assert receipts["first"].supported
    assert receipts["first"].fit_efficiency_lower > 0.5
    assert receipts["first"].check_efficiency_lower > 0.5
    assert not receipts["second"].supported


def test_saturated_plateau_interior_is_not_counted_as_isolated_extrema():
    first, _, _, _ = _pair()
    first[30:100, 40:130] = 0
    receipt = support_aligned_clipping_null_receipt(first, endpoint="first")
    # Only four removable corners enter the exact action event; the plateau
    # interior and straight saturated boundary do not masquerade as isolated
    # impulse support.  Whether those corners are useful is decided by the
    # paired action-efficiency receipt rather than by a morphology veto.
    assert receipt.observed_isolated_extrema == 4


def test_persistent_thin_structure_fails_action_efficiency():
    first, second, forward, backward = _pair()
    first[:, 70] = 0
    second[:, 70] = 0
    _, receipts = visibility_qualified_median_efficiency(
        first, second, forward, backward,
    )
    assert not receipts["first"].supported
    assert not receipts["second"].supported


def test_inconsistent_geometry_removes_action_support():
    first, second, forward, backward = _pair()
    first[40:120:2, 40:150:2] = 255
    backward[..., 0] = 10.0
    _, receipts = visibility_qualified_median_efficiency(
        first, second, forward, backward,
    )
    assert receipts["first"].visible_support_pixels == 0
    assert not receipts["first"].supported


def test_efficiency_receipt_rejects_hash_and_outcome_tampering():
    first, second, forward, backward = _pair()
    _, receipts = visibility_qualified_median_efficiency(
        first, second, forward, backward,
    )
    payload = dict(receipts["first"].__dict__)
    payload["support_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        MedianActionEfficiencyReceipt(**payload)
    payload = dict(receipts["first"].__dict__)
    payload["ground_truth_or_outcome_read"] = True
    with pytest.raises(ValueError, match="labels or outcomes"):
        MedianActionEfficiencyReceipt(**payload)


def test_diffuse_noise_does_not_form_a_large_isolated_extrema_excess():
    first, _, _, _ = _pair()
    rng = np.random.default_rng(19)
    noisy = np.clip(np.rint(
        first.astype(np.float64) + rng.normal(0.0, 15.0, first.shape)
    ), 0, 255).astype(np.uint8)
    receipt = support_aligned_clipping_null_receipt(noisy, endpoint="first")
    # Exact support alignment is less conservative than E169; this synthetic
    # check ensures ordinary diffuse noise still stays below the count bound.
    assert not receipt.supported


def test_invalid_configuration_fails_closed():
    first, second, forward, backward = _pair()
    with pytest.raises(ValueError, match="concrete endpoint"):
        support_aligned_clipping_null_receipt(first, endpoint="both")
    with pytest.raises(ValueError, match="at least 16"):
        visibility_qualified_median_efficiency(
            first, second, forward, backward, tile_size=8,
        )
