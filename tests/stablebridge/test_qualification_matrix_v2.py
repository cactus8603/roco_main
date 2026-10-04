from __future__ import annotations

import inspect
import io
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

from stablebridge.physical_repair import qualification_matrix as v1
from stablebridge.physical_repair.qualification_matrix_v2 import (
    ISOTROPIC_ACTION_ID,
    MOTION_ACTION_ID,
    QUALIFICATION_MATRIX_SCHEMA_V2,
    _blur_policy_rows,
    build_uncapped_qualification_matrix_v2,
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
    ("family", "degrade", "expected"),
    (
        ("disk", _disk, ISOTROPIC_ACTION_ID),
        (
            "gaussian",
            lambda image: cv2.GaussianBlur(image, (0, 0), 2.0),
            ISOTROPIC_ACTION_ID,
        ),
        ("motion", _motion, MOTION_ACTION_ID),
    ),
)
def test_matrix_has_six_rows_and_only_one_blur_arm_executes(
    family, degrade, expected,
):
    clean = _texture()
    first = degrade(clean)
    flow = np.zeros((*clean.shape[:2], 2), dtype=np.float32)
    result = build_uncapped_qualification_matrix_v2(first, clean, flow)
    receipt = result.receipt
    assert receipt["schema"] == QUALIFICATION_MATRIX_SCHEMA_V2
    assert receipt["counts"]["policies"] == 6
    assert receipt["blur_certificate_call_count"] == 1
    assert receipt["blur_public_arm_executed_count"] == 1
    blur = receipt["rows"][2:4]
    executed = [
        row for row in blur if row["qualification_status"] == "ELIGIBLE_EXECUTED"
    ]
    assert len(executed) == 1
    assert executed[0]["qualification_action_id"] == expected
    assert executed[0]["blur_branch"]["internal_family"] == family
    assert executed[0]["mechanism_evaluator_status"] == (
        "IMPLEMENTED_DEVELOPMENT_ESTIMAND_READY"
    )
    for row in receipt["rows"]:
        assert row["prospective_cost_status"] == "UNBOUND_TYPED_MISSING"
        assert row["child_observable_status"] == "NOT_RUN_TYPED_MISSING"
        assert row["scientific_qualification"] is False
        assert row["selector_admission"] is False
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == v1.canonical_sha256(payload)


def test_blur_v6_does_not_depend_on_legacy_blur_eligibility():
    clean = _texture()
    first = _disk(clean)
    flow = np.zeros((*clean.shape[:2], 2), dtype=np.float32)
    pair_hash = v1._physical_pair_sha256(first, clean, flow)
    legacy_abstention = SimpleNamespace(
        control_id="blur_v3_cross_endpoint",
        family="blur",
        operator="legacy",
        authorization="eligible_action",
        execution_status="UNSUPPORTED",
        applicability="LEGACY_GLOBAL_INFORMATION_GATE_FAILED",
        modified_endpoints=(),
        parameters={},
        receipts=(),
        observable_score=None,
        diagnostics={},
        eligible=False,
    )
    rows, arms, local, mechanism = _blur_policy_rows(
        first,
        clean,
        flow,
        pair_sha256=pair_hash,
        legacy_control=legacy_abstention,
    )
    assert rows[0]["qualification_status"] == "ELIGIBLE_EXECUTED"
    assert rows[0]["qualification_action_id"] == ISOTROPIC_ACTION_ID
    assert rows[1]["qualification_status"] == "TYPED_MISSING"
    assert len(arms) == len(local) == len(mechanism) == 1


def test_jpeg_v3_rows_use_block_exact_successors_and_mechanisms():
    clean = _texture(384)
    stream = io.BytesIO()
    Image.fromarray(clean).save(
        stream, format="JPEG", quality=6, subsampling=2,
    )
    stream.seek(0)
    compressed = np.asarray(Image.open(stream).convert("RGB"), dtype=np.uint8)
    first = clean.copy()
    first[:, :192] = compressed[:, :192]
    flow = np.zeros((*clean.shape[:2], 2), dtype=np.float32)
    result = build_uncapped_qualification_matrix_v2(first, clean, flow)
    jpeg_rows = result.receipt["rows"][4:]
    assert len(jpeg_rows) == 2
    assert all(
        row["qualification_status"] == "ELIGIBLE_EXECUTED"
        for row in jpeg_rows
    )
    assert all(
        row["mechanism_evaluator_status"]
        == "IMPLEMENTED_DEVELOPMENT_ESTIMAND_READY"
        for row in jpeg_rows
    )
    assert all(
        row["local_case_materialization_status"]
        == "LOCAL_OUTPUT_CONTRACT_READY_COST_AND_CHILDREN_PENDING"
        for row in jpeg_rows
    )


def test_public_surface_is_observable_only():
    assert list(
        inspect.signature(build_uncapped_qualification_matrix_v2).parameters
    ) == [
        "observed_first_rgb",
        "observed_second_rgb",
        "observed_native_flow",
    ]
