from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import cv2
import numpy as np
import pytest

from stablebridge.physical_repair.blur_parameter_certificates_v3 import (
    blur_parameter_certificates_v3,
)
from stablebridge.physical_repair.blur_parameter_certificates_v6 import (
    SPATIAL_NONINFERIORITY_MARGIN_V6,
    action_specific_direct_winners_v6,
    blur_arm_evidence_v6,
    blur_parameter_certificates_v6,
)


def _texture(size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    value = (
        110 + 35 * np.sin(xx / 3.7) + 29 * np.cos(yy / 5.3)
        + 18 * np.sin((xx + yy) / 8.1)
        + 12 * ((xx // 11 + yy // 13) % 2)
    )
    return np.stack(
        (value, np.roll(value, 5, 1), np.roll(value, 7, 0)), axis=2,
    ).clip(0, 255).astype(np.uint8)


def _disk(image: np.ndarray, radius: int = 6) -> np.ndarray:
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _motion(image: np.ndarray, length: int = 11) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2] = 1.0 / length
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


@lru_cache(maxsize=None)
def _case(name: str):
    clean = _texture()
    if name == "disk":
        first = _disk(clean)
    elif name == "gaussian":
        first = cv2.GaussianBlur(clean, (0, 0), 2.0)
    elif name == "motion":
        first = _motion(clean)
    elif name == "clean":
        first = clean
    else:  # pragma: no cover - test construction guard
        raise ValueError(name)
    flow = np.zeros((*clean.shape[:2], 2), dtype=np.float32)
    return clean, first, flow


@pytest.mark.parametrize(
    ("family", "expected_arm", "expected_mode"),
    (
        ("disk", "common_isotropic@first", "disk"),
        ("gaussian", "common_isotropic@first", "gaussian"),
        ("motion", "common_motion@first", "motion"),
    ),
)
def test_three_physical_families_map_to_two_public_arms(
    family, expected_arm, expected_mode,
):
    clean, first, flow = _case(family)
    evidence = blur_parameter_certificates_v6(first, clean, flow)
    assert action_specific_direct_winners_v6(evidence) == (expected_arm,)
    assert evidence[expected_arm].internal_family == expected_mode
    assert evidence[expected_arm].certificate.action.operator_id == (
        expected_arm.split("@", 1)[0]
    )
    assert evidence[expected_arm].recoverable_regions


def test_clean_pair_abstains():
    clean, first, flow = _case("clean")
    evidence = blur_parameter_certificates_v6(first, clean, flow)
    assert action_specific_direct_winners_v6(evidence) == ()


def test_global_information_median_is_not_a_second_local_gate():
    clean, first, flow = _case("disk")
    source = blur_parameter_certificates_v3(first, clean, flow)
    disk = source["common_disk@first"]
    assert len(disk.recoverable_regions) >= 4
    source["common_disk@first"] = replace(
        disk,
        information_retention_median=0.0,
        recoverability_status="unsupported",
    )
    collapsed = blur_arm_evidence_v6(source)
    assert collapsed["common_isotropic@first"].status == "supported"


def test_motion_spatial_margin_is_one_third_lsb_and_strict():
    clean, first, flow = _case("motion")
    source = blur_parameter_certificates_v3(first, clean, flow)
    motion = source["common_motion@first"]
    inside = dict(motion.pairwise_improvement_lcb)
    inside["spatial_vs_disk"] = -0.5 * SPATIAL_NONINFERIORITY_MARGIN_V6
    inside["spatial_vs_gaussian"] = -0.5 * SPATIAL_NONINFERIORITY_MARGIN_V6
    source["common_motion@first"] = replace(
        motion, pairwise_improvement_lcb=inside,
    )
    collapsed = blur_arm_evidence_v6(source)
    assert collapsed["common_motion@first"].status == "supported"

    outside = dict(inside)
    outside["spatial_vs_disk"] = -1.01 * SPATIAL_NONINFERIORITY_MARGIN_V6
    source["common_motion@first"] = replace(
        motion, pairwise_improvement_lcb=outside,
    )
    rejected = blur_arm_evidence_v6(source)["common_motion@first"]
    assert rejected.status == "rejected"
    assert "SPATIAL_VS_DISK_BELOW_QUANTIZATION_NONINFERIORITY" in (
        rejected.rejection_reasons
    )
