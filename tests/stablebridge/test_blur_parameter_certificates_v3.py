import cv2
import numpy as np

from stablebridge.physical_repair.blur_parameter_certificates_v3 import (
    action_specific_direct_winners,
    blur_parameter_certificates_v3,
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


def test_clean_abstains_under_same_parameter_endpoint_witness():
    clean = texture()
    result = blur_parameter_certificates_v3(clean, clean, flow(clean))
    assert action_specific_direct_winners(result) == ()
    assert all(
        value.certificate.diagnostics["endpoint_same_parameter"] == 1.0
        for value in result.values()
    )


def test_gaussian_and_motion_keep_correct_direct_winners():
    clean = texture()
    gaussian = blur_parameter_certificates_v3(
        cv2.GaussianBlur(clean, (0, 0), 2.0), clean, flow(clean),
    )
    assert action_specific_direct_winners(gaussian) == ("common_gaussian@first",)
    moved = blur_parameter_certificates_v3(motion(clean), clean, flow(clean))
    assert action_specific_direct_winners(moved) == ("common_motion@first",)


def test_noise_never_has_motion_direct_winner():
    clean = texture()
    rng = np.random.default_rng(29)
    noisy = np.clip(
        clean.astype(np.float32) + rng.normal(0, 18, clean.shape), 0, 255,
    ).astype(np.uint8)
    result = blur_parameter_certificates_v3(noisy, clean, flow(clean))
    assert not any(
        key.startswith("common_motion@") for key in action_specific_direct_winners(result)
    )
