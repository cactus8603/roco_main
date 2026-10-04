import numpy as np
import pytest

from stablebridge.physical_repair.spatial_impulse_receipts import (
    SpatialImpulseReceipt,
    spatial_impulse_receipt,
)


def _image(height=192, width=256):
    yy, xx = np.mgrid[:height, :width]
    return np.clip(np.rint(np.stack((
        20.0 + 210.0 * xx / (width - 1),
        30.0 + 190.0 * yy / (height - 1),
        90.0 + 50.0 * np.sin((xx + yy) / 19.0),
    ), axis=2)), 0, 255).astype(np.uint8)


def test_diffuse_balanced_impulses_pass_spatial_law():
    image = _image()
    rng = np.random.default_rng(101)
    locations = rng.choice(image.shape[0] * image.shape[1], 5000, replace=False)
    flat = image.reshape(-1, 3)
    flat[locations[:2500]] = 0
    flat[locations[2500:]] = 255
    receipt = spatial_impulse_receipt(image, endpoint="first")
    assert receipt.supported
    assert receipt.black_scales[0].required_cells == 16
    assert receipt.white_scales[0].required_cells == 16
    assert all(scale.passed for scale in receipt.black_scales)
    assert all(scale.passed for scale in receipt.white_scales)


def test_clustered_two_tail_extrema_fail_diffusion_veto():
    image = _image()
    rng = np.random.default_rng(103)
    locations = rng.choice(96 * 128, 3500, replace=False)
    yy, xx = np.divmod(locations, 128)
    image[yy[:1750], xx[:1750]] = 0
    image[yy[1750:], xx[1750:]] = 255
    receipt = spatial_impulse_receipt(image, endpoint="second")
    assert not receipt.supported
    assert any(not scale.passed for scale in receipt.black_scales)
    assert any(not scale.passed for scale in receipt.white_scales)


def test_low_count_two_tail_events_are_unidentifiable_not_supported():
    image = _image()
    rng = np.random.default_rng(107)
    locations = rng.choice(image.shape[0] * image.shape[1], 80, replace=False)
    flat = image.reshape(-1, 3)
    flat[locations[:40]] = 0
    flat[locations[40:]] = 255
    receipt = spatial_impulse_receipt(image, endpoint="first")
    assert not receipt.supported
    assert not receipt.black_scales[0].identifiable
    assert not receipt.white_scales[0].identifiable


def test_diffuse_gaussian_noise_has_no_spatial_impulse_authority():
    image = _image()
    rng = np.random.default_rng(109)
    noisy = np.clip(np.rint(
        image.astype(np.float64) + rng.normal(0.0, 15.0, image.shape)
    ), 0, 255).astype(np.uint8)
    assert not spatial_impulse_receipt(noisy, endpoint="first").supported


def test_spatial_receipt_rejects_decision_and_outcome_tampering():
    receipt = spatial_impulse_receipt(_image(), endpoint="first")
    payload = dict(receipt.__dict__)
    payload["supported"] = not receipt.supported
    with pytest.raises(ValueError, match="decision drift"):
        SpatialImpulseReceipt(**payload)
    payload = dict(receipt.__dict__)
    payload["ground_truth_or_outcome_read"] = True
    with pytest.raises(ValueError, match="cannot read labels or outcomes"):
        SpatialImpulseReceipt(**payload)


def test_invalid_grid_or_endpoint_fails_closed():
    with pytest.raises(ValueError, match="concrete endpoint"):
        spatial_impulse_receipt(_image(), endpoint="both")
    with pytest.raises(ValueError, match="unique increasing"):
        spatial_impulse_receipt(_image(), endpoint="first", grid_sizes=(8, 4))
