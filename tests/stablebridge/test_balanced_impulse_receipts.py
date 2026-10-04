import numpy as np
import pytest

from stablebridge.physical_repair.balanced_impulse_receipts import (
    BalancedImpulseReceipt,
    balanced_impulse_receipt,
)


def _image(height=192, width=256):
    yy, xx = np.mgrid[:height, :width]
    return np.clip(np.rint(np.stack((
        20.0 + 210.0 * xx / (width - 1),
        30.0 + 190.0 * yy / (height - 1),
        90.0 + 50.0 * np.sin((xx + yy) / 19.0),
    ), axis=2)), 0, 255).astype(np.uint8)


def test_balanced_random_impulse_passes_both_tail_nulls():
    image = _image()
    rng = np.random.default_rng(19)
    locations = rng.choice(image.shape[0] * image.shape[1], 3500, replace=False)
    flat = image.reshape(-1, 3)
    flat[locations[:1750]] = 0
    flat[locations[1750:]] = 255
    receipt = balanced_impulse_receipt(image, endpoint="first")
    assert receipt.supported
    assert receipt.black_excess_lower > 0.0
    assert receipt.white_excess_lower > 0.0
    assert 0.45 < receipt.black_fraction_of_events < 0.55


def test_one_sided_white_clipping_cannot_borrow_absent_black_evidence():
    image = _image()
    rng = np.random.default_rng(23)
    locations = rng.choice(image.shape[0] * image.shape[1], 3000, replace=False)
    image.reshape(-1, 3)[locations] = 255
    receipt = balanced_impulse_receipt(image, endpoint="second")
    assert receipt.white_excess_lower > 0.0
    assert receipt.observed_black_events == 0
    assert not receipt.supported


def test_saturated_plateau_corners_do_not_form_two_tail_contamination():
    image = _image()
    image[30:130, 40:180] = 255
    receipt = balanced_impulse_receipt(image, endpoint="first")
    assert not receipt.supported
    assert receipt.observed_black_events == 0


def test_diffuse_gaussian_noise_fails_two_tail_law():
    image = _image()
    rng = np.random.default_rng(29)
    noisy = np.clip(np.rint(
        image.astype(np.float64) + rng.normal(0.0, 15.0, image.shape)
    ), 0, 255).astype(np.uint8)
    assert not balanced_impulse_receipt(noisy, endpoint="first").supported


def test_receipt_rejects_hash_and_outcome_tampering():
    receipt = balanced_impulse_receipt(_image(), endpoint="first")
    payload = dict(receipt.__dict__)
    payload["combined_support_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        BalancedImpulseReceipt(**payload)
    payload = dict(receipt.__dict__)
    payload["ground_truth_or_outcome_read"] = True
    with pytest.raises(ValueError, match="labels or outcomes"):
        BalancedImpulseReceipt(**payload)


def test_invalid_endpoint_and_multiplicity_fail_closed():
    with pytest.raises(ValueError, match="concrete endpoint"):
        balanced_impulse_receipt(_image(), endpoint="both")
    with pytest.raises(ValueError, match="multiplicity"):
        balanced_impulse_receipt(
            _image(), endpoint="first", hypotheses_tested=1,
        )
