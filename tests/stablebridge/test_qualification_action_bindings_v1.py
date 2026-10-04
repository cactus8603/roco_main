from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stablebridge.physical_repair.qualification_action_bindings_v1 import (
    build_native_action_binding_v1,
    build_qualification_action_binding_v1,
)
from stablebridge.physical_repair.qualification_matrix_v2 import (
    ISOTROPIC_ACTION_ID,
    build_uncapped_qualification_matrix_v2,
)


ROOT = Path(__file__).resolve().parents[2]
BANK = json.loads((
    ROOT / "experiments/E262_action_bank_v2_foundation_freeze/ACTION_BANK_V2.json"
).read_text())


def texture(size=256):
    yy, xx = np.mgrid[:size, :size]
    value = (
        110 + 35 * np.sin(xx / 3.7) + 29 * np.cos(yy / 5.3)
        + 18 * np.sin((xx + yy) / 8.1)
        + 12 * ((xx // 11 + yy // 13) % 2)
    )
    return np.stack((value, np.roll(value, 4, 1), np.roll(value, 7, 0)), 2).clip(
        0, 255,
    ).astype(np.uint8)


def disk(image, radius=6):
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def test_native_binding_is_exact_zero_action_and_bank_bound():
    action, execution = build_native_action_binding_v1(
        bank=BANK,
        case_id="case-1",
        physical_pair_sha256="1" * 64,
    )
    assert action.action_id == "native"
    assert action.strength == 0.0
    assert action.operator_id == "identity"
    assert action.bank_sha256 == BANK["bank_sha256"]
    assert execution["receipt_sha256"]
    assert execution["outcome_read"] is False


def test_blur_binding_carries_internal_mechanism_and_exact_local_lineage():
    clean = texture()
    first = disk(clean)
    flow = np.zeros((*clean.shape[:2], 2), dtype=np.float32)
    result = build_uncapped_qualification_matrix_v2(first, clean, flow)
    row = next(
        value for value in result.receipt["rows"]
        if value["qualification_action_id"] == ISOTROPIC_ACTION_ID
    )
    local = next(
        value for value in result.local_case_receipts
        if value.qualification_action_id == ISOTROPIC_ACTION_ID
    )
    bundle = build_qualification_action_binding_v1(
        bank=BANK,
        qualification_row=row,
        local_receipt=local,
    )
    assert bundle.action.action_id == ISOTROPIC_ACTION_ID
    assert bundle.action.mechanism_id == "common_disk"
    assert bundle.action.operator_id == local.operator_id
    assert bundle.action.exact_parameter_sha256 == local.exact_parameter_sha256
    assert bundle.action_execution_receipt_sha256 == local.receipt_sha256
    assert bundle.action.support_sha256 == bundle.local_support_realization_sha256


def test_binding_rejects_row_parameter_or_bank_drift():
    clean = texture()
    first = disk(clean)
    flow = np.zeros((*clean.shape[:2], 2), dtype=np.float32)
    result = build_uncapped_qualification_matrix_v2(first, clean, flow)
    row = next(
        value for value in result.receipt["rows"]
        if value["qualification_action_id"] == ISOTROPIC_ACTION_ID
    )
    local = next(
        value for value in result.local_case_receipts
        if value.qualification_action_id == ISOTROPIC_ACTION_ID
    )
    drifted = dict(row)
    drifted["parameter_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact parameter drift"):
        build_qualification_action_binding_v1(
            bank=BANK,
            qualification_row=drifted,
            local_receipt=local,
        )
    bad_bank = dict(BANK)
    bad_bank["status"] = "DRIFTED"
    with pytest.raises(ValueError, match="semantic SHA-256 drift"):
        build_native_action_binding_v1(
            bank=bad_bank,
            case_id="case-1",
            physical_pair_sha256="1" * 64,
        )
