from __future__ import annotations

import cv2
import numpy as np
import pytest

from stablebridge.physical_repair.blur_mechanism_evaluators import (
    evaluate_blur_mechanism_v1,
)
from stablebridge.physical_repair.local_blur_successor import (
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


@pytest.mark.parametrize(
    ("family", "degrade"),
    (
        ("disk", _disk),
        ("gaussian", lambda image: cv2.GaussianBlur(image, (0, 0), 2.0)),
        ("motion", _motion),
    ),
)
def test_branch_specific_otf_metric_prefers_selected_blur(family, degrade):
    clean = _texture()
    first = degrade(clean)
    flow = np.zeros((*clean.shape[:2], 2), dtype=np.float32)
    successor = build_local_blur_successor_v2(first, clean, flow)
    receipt = evaluate_blur_mechanism_v1(
        physical_pair_sha256="a" * 64,
        before_pair=(first, clean),
        after_pair=(successor.first_rgb, successor.second_rgb),
        endpoint_supports=(successor.first_support, successor.second_support),
        observed_native_flow=flow,
        winner_id=successor.winner_id,
        family=successor.family,
        identified_endpoint=successor.identified_endpoint,
        selected_parameter=successor.selected_parameter,
        recoverable_regions=successor.recoverable_regions,
    )
    row = receipt.endpoint_records[0]
    assert receipt.family_id == family
    assert row["active_relative_reduction"] > 0.05
    assert row["specificity_difference"] > 0.05
    assert row["outside_support_byte_identity"] is True
    assert len(row["region_records"]) == 8


def test_support_drift_is_rejected_before_metric_execution():
    clean = _texture()
    first = cv2.GaussianBlur(clean, (0, 0), 2.0)
    flow = np.zeros((*clean.shape[:2], 2), dtype=np.float32)
    successor = build_local_blur_successor_v2(first, clean, flow)
    bad = successor.second_support.copy()
    bad[0, 0] = ~bad[0, 0]
    with pytest.raises(RuntimeError, match="support drift"):
        evaluate_blur_mechanism_v1(
            physical_pair_sha256="a" * 64,
            before_pair=(first, clean),
            after_pair=(successor.first_rgb, successor.second_rgb),
            endpoint_supports=(successor.first_support, bad),
            observed_native_flow=flow,
            winner_id=successor.winner_id,
            family=successor.family,
            identified_endpoint=successor.identified_endpoint,
            selected_parameter=successor.selected_parameter,
            recoverable_regions=successor.recoverable_regions,
        )
