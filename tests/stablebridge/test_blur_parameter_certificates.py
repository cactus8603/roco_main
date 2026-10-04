import cv2
import numpy as np

from stablebridge.physical_repair.blur_parameter_certificates import (
    blur_parameter_certificates,
    simultaneous_z,
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


def zero_flow(image):
    return np.zeros((*image.shape[:2], 2), np.float32)


def motion(image, length=11, angle=0):
    kernel = np.zeros((length, length), np.float32)
    center = (length - 1) / 2
    theta = np.deg2rad(angle)
    dx, dy = np.cos(theta) * center, np.sin(theta) * center
    cv2.line(
        kernel,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        1.0, 1,
    )
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def test_simultaneous_bound_is_stricter_than_pointwise_95_percent():
    assert simultaneous_z() > 1.96


def test_gaussian_parameter_is_fit_on_a_and_supported_on_disjoint_b():
    sharp = texture()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2.0)
    result = blur_parameter_certificates(blurred, sharp, zero_flow(sharp))
    gaussian = result["common_gaussian@first"]
    assert gaussian.certificate.status == "supported", gaussian.certificate.rejection_reasons
    assert gaussian.recoverability_status == "supported"
    assert 1.0 <= gaussian.selected_parameter[0] <= 3.0
    assert gaussian.fit_support_hash != gaussian.check_support_hash
    assert gaussian.certificate.action.modified_endpoint == "second"
    assert gaussian.direct_pairwise_winner


def test_motion_parameter_has_directional_support_on_independent_tiles():
    sharp = texture()
    blurred = motion(sharp, length=11, angle=0)
    result = blur_parameter_certificates(blurred, sharp, zero_flow(sharp))
    evidence = result["common_motion@first"]
    assert evidence.certificate.status == "supported", evidence.certificate.rejection_reasons
    assert 7 <= evidence.selected_parameter[0] <= 15
    angle = evidence.selected_parameter[1] % 180
    assert min(angle, 180 - angle) <= 30
    assert evidence.direct_pairwise_winner


def test_additive_noise_does_not_support_motion_repair():
    sharp = texture()
    rng = np.random.default_rng(17)
    noisy = np.clip(
        sharp.astype(np.float32) + rng.normal(0, 18, sharp.shape), 0, 255,
    ).astype(np.uint8)
    result = blur_parameter_certificates(noisy, sharp, zero_flow(sharp))
    assert result["common_motion@first"].certificate.status != "supported"
    assert not any(
        item.direct_pairwise_winner
        for key, item in result.items() if key.startswith("common_motion@")
    )


def test_clean_pair_does_not_create_a_supported_blur_action():
    clean = texture()
    result = blur_parameter_certificates(clean, clean, zero_flow(clean))
    assert not any(item.certificate.status == "supported" for item in result.values())
