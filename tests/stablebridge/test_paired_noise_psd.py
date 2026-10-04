import numpy as np
import pytest

from stablebridge.physical_repair.paired_noise_psd import (
    PairedWienerPSDReceipt,
    paired_wiener_psd_receipt,
    paired_wiener_psd_receipts,
)


def _textured_pair(size: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    base = np.stack((
        90.0 + 45.0 * np.sin(xx / 9.0) + 20.0 * np.cos(yy / 17.0),
        110.0 + 35.0 * np.sin((xx + yy) / 13.0),
        120.0 + 30.0 * np.cos((2 * xx - yy) / 19.0),
    ), axis=2)
    image = np.clip(np.rint(base), 0, 255).astype(np.uint8)
    return image.copy(), image.copy(), np.zeros((size, size, 2), np.float32)


def _noise(image: np.ndarray, seed: int, sigma: float = 18.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def test_whole_frame_first_noise_has_no_within_image_baseline_requirement():
    first, second, flow = _textured_pair()
    first = _noise(first, 7)
    rows = paired_wiener_psd_receipts(first, second, flow)
    assert rows["first"].status == "supported"
    assert rows["first"].fit_gain_lower > 0.0
    assert rows["first"].check_gain_lower > 0.0
    assert rows["first"].endpoint_exclusive_power_fraction > 0.02
    assert rows["second"].status == "rejected"


def test_whole_frame_both_noise_can_support_both_endpoints():
    first, second, flow = _textured_pair()
    rows = paired_wiener_psd_receipts(
        _noise(first, 11), _noise(second, 13), flow,
    )
    assert rows["first"].status == "supported"
    assert rows["second"].status == "supported"


def test_clean_pair_rejects_wiener_action():
    first, second, flow = _textured_pair()
    rows = paired_wiener_psd_receipts(first, second, flow)
    assert rows["first"].status == "rejected"
    assert rows["second"].status == "rejected"


def test_receipt_rejects_support_tampering_and_outcome_access():
    first, second, flow = _textured_pair()
    row = paired_wiener_psd_receipt(_noise(first, 17), second, flow, endpoint="first")
    payload = dict(row.__dict__)
    payload["support_mask"] = np.zeros_like(row.support_mask)
    with pytest.raises(ValueError, match="hash"):
        PairedWienerPSDReceipt(**payload)
    payload = dict(row.__dict__)
    payload["ground_truth_or_outcome_read"] = True
    with pytest.raises(ValueError, match="labels or outcomes"):
        PairedWienerPSDReceipt(**payload)


def test_invalid_endpoint_and_shapes_fail_closed():
    first, second, flow = _textured_pair()
    with pytest.raises(ValueError, match="endpoint"):
        paired_wiener_psd_receipt(first, second, flow, endpoint="both")
    with pytest.raises(ValueError, match="paired uint8"):
        paired_wiener_psd_receipt(first.astype(np.float32), second, flow, endpoint="first")
