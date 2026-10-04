from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import cv2
import numpy as np
import pytest

from stablebridge.physical_repair.blur_mechanism_evaluators_v2 import (
    evaluate_blur_mechanism_v2,
)
from stablebridge.physical_repair.local_blur_successor_v3 import (
    build_local_blur_successor_v3,
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
def _case(family: str):
    clean = _texture()
    if family == "disk":
        first = _disk(clean)
    elif family == "gaussian":
        first = cv2.GaussianBlur(clean, (0, 0), 2.0)
    elif family == "motion":
        first = _motion(clean)
    else:  # pragma: no cover
        raise ValueError(family)
    flow = np.zeros((*clean.shape[:2], 2), dtype=np.float32)
    successor = build_local_blur_successor_v3(first, clean, flow)
    return clean, first, flow, successor


@pytest.mark.parametrize("family", ("disk", "gaussian", "motion"))
def test_public_arm_mechanism_prefers_active_operator(family):
    clean, first, flow, successor = _case(family)
    receipt = evaluate_blur_mechanism_v2(
        physical_pair_sha256="a" * 64,
        before_pair=(first, clean),
        successor=successor,
        observed_native_flow=flow,
    )
    endpoint = receipt.endpoint_records[0]
    assert endpoint["active_relative_reduction"] > 0.0
    assert endpoint["public_arm_specificity_difference"] > 0.0
    assert endpoint["outside_support_byte_identity"] is True
    assert endpoint["read_halo_pixels"] >= endpoint["support_pixels"]
    assert receipt.internal_family == family
    assert receipt.fit_support_sha256 != receipt.check_support_sha256
    roles = {row["role"] for row in endpoint["control_records"]}
    if family in {"disk", "gaussian"}:
        assert roles == {
            "SAME_PUBLIC_ARM_ISOTROPIC_MODE_DIAGNOSTIC",
            "WRONG_PUBLIC_ARM_MOTION",
        }
    else:
        assert roles == {
            "SAME_PUBLIC_ARM_ORTHOGONAL_MOTION_DIAGNOSTIC",
            "WRONG_PUBLIC_ARM_ISOTROPIC",
        }


def test_tampered_write_support_fails_closed():
    clean, first, flow, successor = _case("gaussian")
    bad = successor.second_support.copy()
    bad[0, 0] = ~bad[0, 0]
    tampered = replace(successor, second_support=bad)
    with pytest.raises(RuntimeError, match="support drift"):
        evaluate_blur_mechanism_v2(
            physical_pair_sha256="a" * 64,
            before_pair=(first, clean),
            successor=tampered,
            observed_native_flow=flow,
        )


def test_tampered_read_halo_fails_closed():
    clean, first, flow, successor = _case("motion")
    bad = successor.second_read_support.copy()
    bad[0, 0] = ~bad[0, 0]
    tampered = replace(successor, second_read_support=bad)
    with pytest.raises(RuntimeError, match="read-halo drift"):
        evaluate_blur_mechanism_v2(
            physical_pair_sha256="a" * 64,
            before_pair=(first, clean),
            successor=tampered,
            observed_native_flow=flow,
        )
