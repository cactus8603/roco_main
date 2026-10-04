import time

import cv2
import numpy as np

from stablebridge.physical_repair.blur_parameter_certificates_v2 import (
    FIT_SAMPLE_STRIDE,
    blur_parameter_certificates_v2,
)


def texture(size=256):
    yy, xx = np.mgrid[:size, :size]
    value = (
        110 + 35 * np.sin(xx / 3.7) + 29 * np.cos(yy / 5.3)
        + 18 * np.sin((xx + yy) / 8.1) + 12 * ((xx // 11 + yy // 13) % 2)
    )
    return np.stack((value, np.roll(value, 5, 1), np.roll(value, 7, 0)), 2).clip(
        0, 255
    ).astype(np.uint8)


def flow(image):
    return np.zeros((*image.shape[:2], 2), np.float32)


def motion(image, length=11):
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2] = 1.0 / length
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def test_v2_keeps_exact_b_fold_and_recovers_gaussian_parameter():
    clean = texture()
    result = blur_parameter_certificates_v2(
        cv2.GaussianBlur(clean, (0, 0), 2.0), clean, flow(clean),
    )
    evidence = result["common_gaussian@first"]
    assert evidence.certificate.status == "supported"
    assert evidence.direct_pairwise_winner
    assert evidence.selected_parameter == (2.0,)
    assert evidence.certificate.diagnostics["fit_sample_stride"] == FIT_SAMPLE_STRIDE
    assert evidence.certificate.diagnostics["check_sample_stride"] == 1.0


def test_v2_motion_direct_and_noise_never_motion_direct():
    clean = texture()
    motion_result = blur_parameter_certificates_v2(motion(clean), clean, flow(clean))
    assert motion_result["common_motion@first"].direct_pairwise_winner
    rng = np.random.default_rng(23)
    noisy = np.clip(
        clean.astype(np.float32) + rng.normal(0, 18, clean.shape), 0, 255,
    ).astype(np.uint8)
    noise_result = blur_parameter_certificates_v2(noisy, clean, flow(clean))
    assert not any(
        value.direct_pairwise_winner for key, value in noise_result.items()
        if key.startswith("common_motion@")
    )


def test_v2_clean_abstains_from_direct_action():
    clean = texture()
    result = blur_parameter_certificates_v2(clean, clean, flow(clean))
    assert not any(value.direct_pairwise_winner for value in result.values())
