import cv2
import numpy as np
import pytest

from stablebridge.physical_repair.noise_law_receipts import (
    DiffuseClippingNullReceipt,
    diffuse_clipping_null_receipt,
    paired_noise_law_receipts,
)


def _ramp_pair(height=256, width=320):
    yy, xx = np.mgrid[:height, :width]
    ramp = 20.0 + 215.0 * xx / max(width - 1, 1)
    base = np.stack((
        ramp + 7.0 * np.sin(yy / 19.0),
        ramp + 5.0 * np.cos((xx + yy) / 23.0),
        ramp + 6.0 * np.sin((2 * xx - yy) / 31.0),
    ), axis=2)
    image = np.clip(np.rint(base), 0, 255).astype(np.uint8)
    return image.copy(), image.copy(), np.zeros((height, width, 2), np.float32)


def _add_noise(image, scale, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, scale, image.shape)
    return np.clip(np.rint(image.astype(np.float64) + noise), 0, 255).astype(np.uint8)


def _shot_noise(image, strength, seed):
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.maximum(image.astype(np.float64), 1.0) * strength)
    noise = rng.normal(0.0, sigma)
    return np.clip(np.rint(image.astype(np.float64) + noise), 0, 255).astype(np.uint8)


def _multiplicative_noise(image, strength, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, strength, image.shape) * image.astype(np.float64)
    return np.clip(np.rint(image.astype(np.float64) + noise), 0, 255).astype(np.uint8)


def test_sparse_impulse_exceeds_diffuse_null_and_exact_median_responds():
    first, second, flow = _ramp_pair()
    rng = np.random.default_rng(3)
    locations = rng.choice(first.shape[0] * first.shape[1], 900, replace=False)
    corrupt = first.copy().reshape(-1, 3)
    corrupt[locations[:450]] = 0
    corrupt[locations[450:]] = 255
    corrupt = corrupt.reshape(first.shape)
    rows = paired_noise_law_receipts(corrupt, second, flow)
    first_row = rows["first"]
    assert first_row.clipping_null.supported
    assert first_row.clipping_null.excess_count_lower > 0.0
    assert first_row.median_response.response_supported
    assert first_row.median_response.fit_gain_lower_255 > 0.0
    assert first_row.median_response.check_gain_lower_255 > 0.0
    assert first_row.median_feasible_before_visibility
    assert not first_row.median_response.visibility_assumption_resolved
    assert not rows["second"].median_feasible_before_visibility


def test_diffuse_clipping_null_is_conservative_for_clipped_gaussian_noise():
    first, _, _ = _ramp_pair()
    corrupt = _add_noise(first, 28.0, 11)
    receipt = diffuse_clipping_null_receipt(corrupt, endpoint="first")
    # The Frechet RGB-correlation upper bound should explain diffuse clipping;
    # this is intentionally a conservative applicability test.
    assert receipt.observed_isolated_extrema > 0
    assert receipt.excess_count_lower <= 0.0
    assert not receipt.supported


def test_persistent_thin_structure_fails_exact_median_response():
    first, second, flow = _ramp_pair()
    first[:, 79:80] = 0
    second[:, 79:80] = 0
    rows = paired_noise_law_receipts(first, second, flow)
    assert rows["first"].clipping_null.observed_isolated_extrema > 0
    assert not rows["first"].median_response.response_supported
    assert not rows["first"].median_feasible_before_visibility


@pytest.mark.parametrize(
    "generator,expected_hint",
    [
        (lambda image: _add_noise(image, 13.0, 21), "wiener3"),
        (lambda image: _shot_noise(image, 2.4, 22), "anscombe_wiener3"),
        (lambda image: _multiplicative_noise(image, 0.16, 23), "log_wiener3"),
    ],
)
def test_variance_law_generates_action_specific_hint(generator, expected_hint):
    first, second, flow = _ramp_pair()
    receipt = paired_noise_law_receipts(generator(first), second, flow)["first"]
    assert receipt.variance_law.law_hint == expected_hint


def test_receipts_reject_hash_tampering_and_outcome_access():
    first, _, _ = _ramp_pair(96, 128)
    receipt = diffuse_clipping_null_receipt(first, endpoint="first")
    payload = dict(receipt.__dict__)
    tampered = receipt.support_mask.copy()
    tampered[5, 5] = ~tampered[5, 5]
    payload["support_mask"] = tampered
    with pytest.raises(ValueError, match="hash"):
        DiffuseClippingNullReceipt(**payload)
    payload = dict(receipt.__dict__)
    payload["ground_truth_or_outcome_read"] = True
    with pytest.raises(ValueError, match="labels or outcomes"):
        DiffuseClippingNullReceipt(**payload)


def test_invalid_shapes_and_small_tile_fail_closed():
    first, second, flow = _ramp_pair(96, 128)
    with pytest.raises(ValueError, match="uint8 RGB"):
        paired_noise_law_receipts(first.astype(np.float32), second, flow)
    with pytest.raises(ValueError, match="finite HxWx2"):
        paired_noise_law_receipts(first, second, flow[..., :1])
    with pytest.raises(ValueError, match="at least 16"):
        paired_noise_law_receipts(first, second, flow, tile_size=8)
