from __future__ import annotations

import cv2
import numpy as np
import pytest

from stablebridge.physical_repair.local_blur_successor_v3 import (
    LOCAL_BLUR_ACTION_IDS_V3,
    LOCAL_BLUR_OPERATOR_IDS_V3,
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


def _flow(image: np.ndarray) -> np.ndarray:
    return np.zeros((*image.shape[:2], 2), dtype=np.float32)


@pytest.mark.parametrize(
    ("physical_family", "degrade", "public_arm"),
    (
        ("disk", _disk, "common_isotropic"),
        (
            "gaussian",
            lambda image: cv2.GaussianBlur(image, (0, 0), 2.0),
            "common_isotropic",
        ),
        ("motion", _motion, "common_motion"),
    ),
)
def test_three_blur_operators_materialize_as_two_local_arms(
    physical_family, degrade, public_arm,
):
    clean = _texture()
    first = degrade(clean)
    result = build_local_blur_successor_v3(first, clean, _flow(clean))
    assert result.status == "EXECUTED_LOCAL_RECOVERABLE_V3"
    assert result.public_arm == public_arm
    assert result.winner_id == f"{public_arm}@first"
    assert result.internal_family == physical_family
    assert result.identified_endpoint == "first"
    assert result.exact_endpoint == "second"
    assert result.action_id == LOCAL_BLUR_ACTION_IDS_V3[public_arm]
    assert result.operator_id == LOCAL_BLUR_OPERATOR_IDS_V3[public_arm]
    assert not result.first_support.any()
    assert int(result.second_support.sum()) == 8 * 64 * 64
    assert np.array_equal(result.first_rgb, first)
    assert np.any(result.second_rgb[result.second_support] != clean[result.second_support])
    assert np.array_equal(
        result.second_rgb[~result.second_support], clean[~result.second_support],
    )
    assert result.receipt["outside_support_byte_identity"] is True
    assert np.all(result.second_support <= result.second_read_support)
    assert result.second_read_support.sum() >= result.second_support.sum()
    assert result.receipt["endpoint_read_support_sha256s"]["second"]
    assert result.receipt["scientific_qualification"] is False
    assert result.receipt["selector_admission"] is False


def test_blurred_second_routes_local_isotropic_action_to_first():
    clean = _texture()
    second = cv2.GaussianBlur(clean, (0, 0), 2.0)
    result = build_local_blur_successor_v3(clean, second, _flow(clean))
    assert result.status == "EXECUTED_LOCAL_RECOVERABLE_V3"
    assert result.winner_id == "common_isotropic@second"
    assert result.internal_family == "gaussian"
    assert result.exact_endpoint == "first"
    assert result.first_support.any()
    assert not result.second_support.any()
    assert np.array_equal(result.second_rgb, second)
    assert np.array_equal(result.first_rgb[~result.first_support], clean[~result.first_support])


def test_clean_pair_abstains_with_deterministic_receipt():
    clean = _texture()
    first = build_local_blur_successor_v3(clean, clean, _flow(clean))
    second = build_local_blur_successor_v3(clean, clean, _flow(clean))
    assert first.status == "UNSUPPORTED_DIRECT_WINNER_COUNT"
    assert first.action_id is None and first.operator_id is None
    assert first.exact_endpoint is None
    assert not first.first_support.any() and not first.second_support.any()
    assert not first.first_read_support.any() and not first.second_read_support.any()
    assert np.array_equal(first.first_rgb, clean)
    assert first.receipt["receipt_sha256"] == second.receipt["receipt_sha256"]
