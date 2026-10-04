from dataclasses import replace

import cv2
import numpy as np

from stablebridge.physical_repair.blur_parameter_certificates_v3 import (
    action_specific_direct_winners,
    blur_parameter_certificates_v3,
)
from stablebridge.physical_repair.blur_parameter_certificates_v5 import (
    action_specific_direct_winners_v5,
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


def disk(image, radius=6):
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def test_negative_disk_spectral_own_null_revokes_direct_authority():
    clean = texture()
    evidence = blur_parameter_certificates_v3(disk(clean), clean, flow(clean))
    winners = action_specific_direct_winners(evidence)
    assert winners == ("common_disk@first",)
    key = winners[0]
    evidence[key] = replace(
        evidence[key], own_null_spectral_improvement_lcb=-1.0,
    )
    assert action_specific_direct_winners_v5(evidence) == ()


def test_disk_spectral_gate_is_rejected_by_a_true_synthetic_disk():
    clean = texture()
    disk_evidence = blur_parameter_certificates_v3(
        disk(clean), clean, flow(clean),
    )
    assert action_specific_direct_winners(disk_evidence) == (
        "common_disk@first",
    )
    assert disk_evidence["common_disk@first"].own_null_spectral_improvement_lcb < 0.0
    assert action_specific_direct_winners_v5(disk_evidence) == ()


def test_rejected_disk_rule_does_not_change_gaussian_semantics():
    clean = texture()
    gaussian = blur_parameter_certificates_v3(
        cv2.GaussianBlur(clean, (0, 0), 2.0), clean, flow(clean),
    )
    assert action_specific_direct_winners_v5(gaussian) == (
        "common_gaussian@first",
    )
