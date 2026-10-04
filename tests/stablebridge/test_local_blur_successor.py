from __future__ import annotations

import cv2
import numpy as np
import pytest

from stablebridge.physical_repair.local_blur_successor import (
    LOCAL_BLUR_ACTION_ID_V2,
    LOCAL_BLUR_OPERATOR_ID_V2,
    build_local_blur_successor_v2,
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
    ("family", "degrade"),
    (
        ("disk", _disk),
        ("gaussian", lambda image: cv2.GaussianBlur(image, (0, 0), 2.0)),
        ("motion", _motion),
    ),
)
def test_three_blur_branches_become_local_opposite_endpoint_successors(
    family, degrade,
):
    clean = _texture()
    first = degrade(clean)
    result = build_local_blur_successor_v2(first, clean, _flow(clean))
    assert result.status == "EXECUTED_LOCAL_RECOVERABLE_V2"
    assert result.winner_id == f"common_{family}@first"
    assert result.family == family
    assert result.identified_endpoint == "first"
    assert result.exact_endpoint == "second"
    assert not result.first_support.any()
    assert int(result.second_support.sum()) == 8 * 64 * 64
    assert np.array_equal(result.first_rgb, first)
    assert np.any(result.second_rgb[result.second_support] != clean[result.second_support])
    assert np.array_equal(
        result.second_rgb[~result.second_support], clean[~result.second_support],
    )
    assert result.receipt["action_id"] == LOCAL_BLUR_ACTION_ID_V2
    assert result.receipt["operator_id"] == LOCAL_BLUR_OPERATOR_ID_V2
    assert result.receipt["outside_support_byte_identity"] is True


def test_blurred_second_routes_local_action_to_first():
    clean = _texture()
    second = cv2.GaussianBlur(clean, (0, 0), 2.0)
    result = build_local_blur_successor_v2(clean, second, _flow(clean))
    assert result.status == "EXECUTED_LOCAL_RECOVERABLE_V2"
    assert result.winner_id == "common_gaussian@second"
    assert result.exact_endpoint == "first"
    assert result.first_support.any()
    assert not result.second_support.any()
    assert np.array_equal(result.second_rgb, second)
    assert np.array_equal(result.first_rgb[~result.first_support], clean[~result.first_support])


def test_clean_pair_abstains_and_receipt_is_deterministic():
    clean = _texture()
    first = build_local_blur_successor_v2(clean, clean, _flow(clean))
    second = build_local_blur_successor_v2(clean, clean, _flow(clean))
    assert first.status == "UNSUPPORTED_DIRECT_WINNER_COUNT"
    assert first.exact_endpoint is None
    assert not first.first_support.any() and not first.second_support.any()
    assert np.array_equal(first.first_rgb, clean)
    assert first.receipt["receipt_sha256"] == second.receipt["receipt_sha256"]
