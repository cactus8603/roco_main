from __future__ import annotations

import inspect
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
WORK_E = ROOT / "research/region_aware_action_learning_20261001/work_package_E"
sys.path.insert(0, str(WORK_E))

from choice_registry import ELIGIBLE_AUTHORIZATION, ExactControl  # noqa: E402
from stablebridge.physical_repair.qualification_matrix import (  # noqa: E402
    _build_uncapped_matrix,
    _physical_pair_sha256,
    _replace_legacy_blur_with_v2,
    build_uncapped_qualification_matrix,
)
from stablebridge.physical_repair.local_blur_successor import (  # noqa: E402
    LOCAL_BLUR_ACTION_ID_V2,
    LOCAL_BLUR_OPERATOR_ID_V2,
)
from stablebridge.physical_repair.local_jpeg_successors import (  # noqa: E402
    LOCAL_JPEG_CODEC_ACTION_ID_V2,
    LOCAL_JPEG_CODEC_OPERATOR_ID_V2,
    LOCAL_JPEG_QCELL_ACTION_ID_V2,
    LOCAL_JPEG_QCELL_OPERATOR_ID_V2,
)


TARGETS = (
    "paired_impulse_median3",
    "paired_additive_wiener3.local_tile_sigma_v2",
    LOCAL_BLUR_ACTION_ID_V2,
    LOCAL_JPEG_QCELL_ACTION_ID_V2,
    LOCAL_JPEG_CODEC_ACTION_ID_V2,
)


def _receipt():
    return {
        "status": "supported",
        "observation_support_fraction": 1.0,
        "identifiable_support_fraction": 1.0,
        "spatial_holdout_gain": 1.0,
        "parameter_uncertainty": 0.0,
        "fit_stability": 1.0,
    }


def _control(control_id: str, first: np.ndarray, second: np.ndarray) -> ExactControl:
    target = first.copy()
    target[2:6, 2:6] = 31
    support = np.zeros(first.shape[:2], dtype=bool)
    support[2:6, 2:6] = True
    operator = "synthetic"
    parameters = {"synthetic": True}
    endpoints = ("first",)
    if control_id == "paired_impulse_median3":
        operator = "impulse_exact_median3"
    elif control_id == "paired_additive_wiener3.local_tile_sigma_v2":
        operator = "wiener3_local_tile_sigma_v2"
    if control_id == LOCAL_BLUR_ACTION_ID_V2:
        operator = LOCAL_BLUR_OPERATOR_ID_V2
        parameters = {
            "winner": "common_gaussian@second",
            "family": "gaussian",
            "selected_parameter": [1.0],
            "identified_endpoint": "second",
            "modified_endpoint": "first",
            "recoverable_regions": [[0, 0]],
        }
    elif control_id == LOCAL_JPEG_QCELL_ACTION_ID_V2:
        operator = LOCAL_JPEG_QCELL_OPERATOR_ID_V2
        parameters = {
            "first": {
                "estimated_ijg_quality": 6,
                "selected_strength": 1.0,
            },
        }
    elif control_id == LOCAL_JPEG_CODEC_ACTION_ID_V2:
        operator = LOCAL_JPEG_CODEC_OPERATOR_ID_V2
        parameters = {
            "first": {
                "estimated_ijg_quality": 6,
                "selected_strength": 1.0,
            },
        }
    return ExactControl(
        control_id=control_id,
        family="synthetic",
        operator=operator,
        authorization=ELIGIBLE_AUTHORIZATION,
        execution_status="EXECUTED",
        applicability="SYNTHETIC_BEFORE_ONLY",
        modified_endpoints=endpoints,
        parameters=parameters,
        receipts=(_receipt(),),
        proposal=(target, second.copy()),
        supports=(support, np.zeros_like(support)),
        observable_score=1.0,
    )


def _controls(first: np.ndarray, second: np.ndarray):
    controls = [_control(control_id, first, second) for control_id in TARGETS]
    controls.append(ExactControl(
        control_id="radiometry_rank3_diagnostic",
        family="radiometry",
        operator="E62_rank3_pair",
        authorization="diagnostic_only",
        execution_status="NOT_REACHED",
        applicability="SYNTHETIC_DIAGNOSTIC",
        receipts=(_receipt(),),
    ))
    return controls


def _inputs():
    first = np.zeros((8, 8, 3), dtype=np.uint8)
    second = np.zeros_like(first)
    flow = np.zeros((8, 8, 2), dtype=np.float32)
    return first, second, flow


def _materialize(control, first, second):
    return {
        0.5: (first.copy(), second.copy()),
        1.0: tuple(value.copy() for value in control.proposal),
    }


def test_uncapped_matrix_executes_all_five_and_retains_only_strength_one():
    first, second, flow = _inputs()
    controls = _controls(first, second)
    result = _build_uncapped_matrix(
        first,
        second,
        flow,
        controls=controls,
        physical_pair_sha256=_physical_pair_sha256(first, second, flow),
        materialize=_materialize,
    )
    assert result.receipt["counts"] == {
        "policies": 5,
        "eligible_executed": 5,
        "typed_missing": 0,
        "materialized_arms": 5,
        "local_output_contracts_ready": 5,
        "mechanism_evaluations_ready": 0,
    }
    assert len(result.arms) == 5
    assert len(result.local_case_receipts) == 5
    assert all(row["input_strength"] == 1.0 for row in result.receipt["rows"])
    assert all(not row["capacity_selection_used"] for row in result.receipt["rows"])
    blur = result.receipt["rows"][2]
    assert blur["exact_endpoint"] == "first"
    assert blur["blur_branch"]["winner_id"] == "common_gaussian@second"
    assert blur["blur_branch"]["modified_endpoint"] == "first"


def test_ineligible_action_is_typed_missing_and_not_materialized():
    first, second, flow = _inputs()
    controls = _controls(first, second)
    controls[1] = ExactControl(
        control_id="paired_additive_wiener3.local_tile_sigma_v2",
        family="noise",
        operator="wiener3_local_tile_sigma_v2",
        authorization=ELIGIBLE_AUTHORIZATION,
        execution_status="UNSUPPORTED",
        applicability="NO_SUPPORTED_PAIRED_ADDITIVE",
        receipts=(_receipt(),),
    )
    result = _build_uncapped_matrix(
        first,
        second,
        flow,
        controls=controls,
        physical_pair_sha256=_physical_pair_sha256(first, second, flow),
        materialize=_materialize,
    )
    assert result.receipt["counts"]["typed_missing"] == 1
    assert result.receipt["rows"][1]["qualification_status"] == "TYPED_MISSING"
    assert len(result.arms) == 4


def test_legacy_blur_detection_without_local_recoverability_retains_matrix(
    monkeypatch,
):
    first, second, flow = _inputs()
    controls = _controls(first, second)
    legacy = _control("blur_v3_cross_endpoint", first, second)
    legacy.parameters = {
        "winner": "common_gaussian@second",
        "family": "gaussian",
        "selected_parameter": [1.0],
        "identified_endpoint": "second",
        "modified_endpoint": "first",
    }
    controls[2] = legacy
    successor = SimpleNamespace(
        status="UNSUPPORTED_RECOVERABLE_REGIONS",
        receipt={"receipt_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        "stablebridge.physical_repair.qualification_matrix."
        "build_local_blur_successor_v2",
        lambda *_args: successor,
    )

    replaced = _replace_legacy_blur_with_v2(controls, first, second, flow)
    result = _build_uncapped_matrix(
        first,
        second,
        flow,
        controls=replaced,
        physical_pair_sha256=_physical_pair_sha256(first, second, flow),
        materialize=_materialize,
    )

    assert result.receipt["counts"] == {
        "policies": 5,
        "eligible_executed": 4,
        "typed_missing": 1,
        "materialized_arms": 4,
        "local_output_contracts_ready": 4,
        "mechanism_evaluations_ready": 0,
    }
    blur = result.receipt["rows"][2]
    assert blur["qualification_action_id"] == LOCAL_BLUR_ACTION_ID_V2
    assert blur["qualification_status"] == "TYPED_MISSING"
    assert blur["source_execution_status"] == "UNSUPPORTED"
    assert blur["source_applicability"] == "UNSUPPORTED_RECOVERABLE_REGIONS"
    assert len(result.arms) == 4


def test_capacity_mutation_and_missing_policy_fail_closed():
    first, second, flow = _inputs()
    controls = _controls(first, second)
    controls[0].selection_status = "NOT_SELECTED_CAPACITY"
    with pytest.raises(RuntimeError, match="forbids ranked/capacity"):
        _build_uncapped_matrix(
            first,
            second,
            flow,
            controls=controls,
            physical_pair_sha256=_physical_pair_sha256(first, second, flow),
            materialize=_materialize,
        )
    with pytest.raises(RuntimeError, match="control universe drift"):
        _build_uncapped_matrix(
            first,
            second,
            flow,
            controls=[
                item for item in controls
                if item.control_id != LOCAL_JPEG_CODEC_ACTION_ID_V2
            ],
            physical_pair_sha256=_physical_pair_sha256(first, second, flow),
            materialize=_materialize,
        )


def test_support_escape_and_blur_branch_drift_fail_closed():
    first, second, flow = _inputs()
    controls = _controls(first, second)

    def escaped(control, observed_first, observed_second):
        outputs = list(value.copy() for value in control.proposal)
        outputs[0][0, 0] = 99
        return {
            0.5: (observed_first.copy(), observed_second.copy()),
            1.0: tuple(outputs),
        }

    with pytest.raises(RuntimeError, match="exact proposal"):
        _build_uncapped_matrix(
            first,
            second,
            flow,
            controls=controls,
            physical_pair_sha256=_physical_pair_sha256(first, second, flow),
            materialize=escaped,
        )
    controls = _controls(first, second)
    controls[2].parameters["modified_endpoint"] = "second"
    with pytest.raises(RuntimeError, match="opposite endpoint"):
        _build_uncapped_matrix(
            first,
            second,
            flow,
            controls=controls,
            physical_pair_sha256=_physical_pair_sha256(first, second, flow),
            materialize=_materialize,
        )


def test_public_surface_is_observable_only():
    assert list(inspect.signature(build_uncapped_qualification_matrix).parameters) == [
        "observed_first_rgb",
        "observed_second_rgb",
        "observed_native_flow",
    ]


def test_public_runner_materializes_real_impulse_local_receipt():
    first = np.full((256, 256, 3), 128, dtype=np.uint8)
    second = first.copy()
    for y0, x0 in ((16, 16), (16, 144), (144, 16), (144, 144)):
        for index in range(12):
            y = y0 + (index // 4) * 5
            x = x0 + (index % 4) * 5
            first[y, x] = 0 if index % 2 == 0 else 255
    flow = np.zeros((256, 256, 2), dtype=np.float32)
    result = build_uncapped_qualification_matrix(first, second, flow)
    row = result.receipt["rows"][0]
    assert row["qualification_status"] == "ELIGIBLE_EXECUTED"
    assert row["local_case_materialization_status"].startswith("LOCAL_OUTPUT")
    assert row["mechanism_evaluator_status"] == (
        "IMPLEMENTED_DEVELOPMENT_ESTIMAND_READY"
    )
    assert len(result.mechanism_receipts) == 1
    local = result.local_case_receipts[0].as_dict()
    assert local["qualification_action_id"] == (
        "paired_impulse_median3.endpoint_supported_v1"
    )
    assert local["endpoint_records"][0]["support_pixels"] == 48
    assert local["endpoint_records"][0]["changed_pixels"] == 48


def test_public_runner_replaces_identity_v1_wiener_with_local_v2():
    rng = np.random.default_rng(251)
    first = np.full((256, 256, 3), 128, dtype=np.uint8)
    second = first.copy()
    first[64:96, 64:128] = np.clip(
        np.rint(128 + rng.normal(0, 25, (32, 64, 3))), 16, 240,
    ).astype(np.uint8)
    flow = np.zeros((256, 256, 2), dtype=np.float32)
    result = build_uncapped_qualification_matrix(first, second, flow)
    row = result.receipt["rows"][1]
    assert row["qualification_action_id"] == (
        "paired_additive_wiener3.local_tile_sigma_v2"
    )
    assert row["qualification_status"] == "ELIGIBLE_EXECUTED"
    assert row["local_case_materialization_status"].startswith("LOCAL_OUTPUT")
    assert row["mechanism_evaluator_status"] == (
        "IMPLEMENTED_DEVELOPMENT_ESTIMAND_READY"
    )
    assert len(result.mechanism_receipts) == 1
    local = result.local_case_receipts[0].as_dict()
    assert local["endpoint_records"][0]["support_pixels"] == 2 * 32 * 32
    assert local["endpoint_records"][0]["changed_pixels"] == 2 * 32 * 32


def test_public_runner_replaces_whole_endpoint_blur_with_local_v2():
    yy, xx = np.mgrid[:256, :256]
    value = (
        110 + 35 * np.sin(xx / 3.7) + 29 * np.cos(yy / 5.3)
        + 18 * np.sin((xx + yy) / 8.1)
        + 12 * ((xx // 11 + yy // 13) % 2)
    )
    clean = np.stack(
        (value, np.roll(value, 5, 1), np.roll(value, 7, 0)), axis=2,
    ).clip(0, 255).astype(np.uint8)
    first = cv2.GaussianBlur(clean, (0, 0), 2.0)
    flow = np.zeros((256, 256, 2), dtype=np.float32)
    result = build_uncapped_qualification_matrix(first, clean, flow)
    row = result.receipt["rows"][2]
    assert row["qualification_action_id"] == LOCAL_BLUR_ACTION_ID_V2
    assert row["qualification_status"] == "ELIGIBLE_EXECUTED"
    assert row["exact_endpoint"] == "second"
    assert row["endpoints"][1]["support_pixels"] == 8 * 64 * 64
    assert row["endpoints"][1]["support_pixels"] < 256 * 256
    assert row["local_case_materialization_status"].startswith("LOCAL_OUTPUT")
    assert row["mechanism_evaluator_status"] == (
        "IMPLEMENTED_DEVELOPMENT_ESTIMAND_READY"
    )
    assert len(result.local_case_receipts) == 1
    assert len(result.mechanism_receipts) == 1
    mechanism = result.mechanism_receipts[0]
    assert mechanism.family_id == "gaussian"
    assert mechanism.endpoint_records[0]["active_relative_reduction"] > 0.05
    assert mechanism.endpoint_records[0]["specificity_difference"] > 0.05


def test_public_runner_versions_both_jpeg_policies_onto_local_support():
    yy, xx = np.mgrid[:384, :384]
    value = (
        110 + 35 * np.sin(xx / 3.7) + 29 * np.cos(yy / 5.3)
        + 18 * np.sin((xx + yy) / 8.1)
        + 12 * ((xx // 11 + yy // 13) % 2)
    )
    clean = np.stack(
        (value, np.roll(value, 5, 1), np.roll(value, 7, 0)), axis=2,
    ).clip(0, 255).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(clean).save(
        stream, format="JPEG", quality=6, subsampling=2,
    )
    stream.seek(0)
    jpeg = np.asarray(Image.open(stream).convert("RGB"), dtype=np.uint8)
    first = clean.copy()
    first[:, :192] = jpeg[:, :192]
    flow = np.zeros((384, 384, 2), dtype=np.float32)
    result = build_uncapped_qualification_matrix(first, clean, flow)

    rows = {
        row["qualification_action_id"]: row
        for row in result.receipt["rows"]
    }
    jpeg_receipts = {
        receipt.action_id: receipt
        for receipt in result.mechanism_receipts
        if receipt.action_id in {
            LOCAL_JPEG_QCELL_ACTION_ID_V2,
            LOCAL_JPEG_CODEC_ACTION_ID_V2,
        }
    }
    jpeg_locals = {
        receipt.qualification_action_id: receipt
        for receipt in result.local_case_receipts
        if receipt.qualification_action_id in {
            LOCAL_JPEG_QCELL_ACTION_ID_V2,
            LOCAL_JPEG_CODEC_ACTION_ID_V2,
        }
    }
    assert set(jpeg_receipts) == {
        LOCAL_JPEG_QCELL_ACTION_ID_V2,
        LOCAL_JPEG_CODEC_ACTION_ID_V2,
    }
    assert set(jpeg_locals) == set(jpeg_receipts)
    for action_id, receipt in jpeg_receipts.items():
        row = rows[action_id]
        endpoint = receipt.endpoint_records[0]
        assert row["qualification_status"] == "ELIGIBLE_EXECUTED"
        assert row["exact_endpoint"] == "first"
        assert row["endpoints"][0]["support_pixels"] == 18 * 64 * 64
        assert row["local_case_materialization_status"].startswith("LOCAL_OUTPUT")
        assert row["mechanism_evaluator_status"] == (
            "IMPLEMENTED_DEVELOPMENT_ESTIMAND_READY"
        )
        assert endpoint["active_relative_reduction"] > 0.05
        assert endpoint["specificity_difference"] > 0.05
        assert endpoint["outside_support_byte_identity"]
