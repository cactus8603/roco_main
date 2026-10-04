import cv2
import numpy as np

from stablebridge.physical_repair.blur_parameter_certificates_v4 import (
    action_specific_direct_winners_v4,
    blur_parameter_certificates_v4,
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


def test_clean_abstains_and_records_shared_geometry_witness():
    clean = texture()
    result = blur_parameter_certificates_v4(clean, clean, flow(clean))
    assert action_specific_direct_winners_v4(result) == ()
    assert all(
        value.certificate.diagnostics["identity_selected_registration"] == 1.0
        for value in result.values()
    )


def test_true_gaussian_and_motion_survive_shared_geometry_gate():
    clean = texture()
    gaussian = blur_parameter_certificates_v4(
        cv2.GaussianBlur(clean, (0, 0), 2.0), clean, flow(clean),
    )
    assert action_specific_direct_winners_v4(gaussian) == (
        "common_gaussian@first",
    )
    moved = blur_parameter_certificates_v4(motion(clean), clean, flow(clean))
    assert action_specific_direct_winners_v4(moved) == ("common_motion@first",)


def test_pure_registration_error_does_not_become_blur_direct():
    clean = texture()
    shifted = np.roll(clean, 2, axis=1)
    result = blur_parameter_certificates_v4(shifted, clean, flow(clean))
    assert action_specific_direct_winners_v4(result) == ()

