import numpy as np

from stablebridge.physical_repair.blur_parameter_certificates_v3_1 import (
    blur_parameter_certificate_bank_v3_1,
)


def texture(size=256):
    yy, xx = np.mgrid[:size, :size]
    value = (
        110 + 35 * np.sin(xx / 3.7) + 29 * np.cos(yy / 5.3)
        + 18 * np.sin((xx + yy) / 8.1)
        + 12 * ((xx // 11 + yy // 13) % 2)
    )
    return np.stack((value, np.roll(value, 5, 1), np.roll(value, 7, 0)), 2).clip(
        0, 255
    ).astype(np.uint8)


def flow(image):
    return np.zeros((*image.shape[:2], 2), np.float32)


def test_low_texture_pair_is_an_explicit_native_fallback():
    flat = np.full((256, 256, 3), 127, np.uint8)
    bank = blur_parameter_certificate_bank_v3_1(flat, flat, flow(flat))
    assert bank.status == "unsupported"
    assert bank.evidence == {}
    assert bank.rejection_reasons
    assert any(value < 4 for value in bank.fold_block_counts.values())


def test_textured_pair_preserves_the_frozen_v3_bank():
    clean = texture()
    bank = blur_parameter_certificate_bank_v3_1(clean, clean, flow(clean))
    assert bank.status == "evaluated"
    assert len(bank.evidence) == 6
    assert not bank.rejection_reasons
    assert all(value >= 4 for value in bank.fold_block_counts.values())

