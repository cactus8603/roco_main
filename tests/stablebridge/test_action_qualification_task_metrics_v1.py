from __future__ import annotations

import numpy as np
import pytest

from stablebridge.physical_repair.action_qualification_task_metrics_v1 import (
    score_action_case_v1,
)


def _inputs():
    gt = np.zeros((4, 5, 2), dtype=np.float32)
    native = np.zeros_like(gt)
    native[..., 0] = 1.0
    action = np.zeros_like(gt)
    action[..., 0] = 0.5
    valid = np.ones((4, 5), dtype=np.bool_)
    labels = np.zeros((4, 5), dtype=np.int32)
    return native, action, gt, valid, labels


def _score(**changes):
    native, action, gt, valid, labels = _inputs()
    values = {
        "case_id": "case", "component_id": "component",
        "source_dataset": "spring", "outer_fold": 0,
        "mechanism_stratum": "common_disk", "action_id": "action",
        "action_binding_sha256": "a" * 64,
        "native_flow": native, "action_flow": action,
        "ground_truth_flow": gt, "gt_valid_mask": valid,
        "region_labels": labels, "rgb_diagonal_px": 10.0,
    }
    values.update(changes)
    return score_action_case_v1(**values)


def test_positive_gain_and_normalization_are_exact():
    sidecars, receipt = _score()
    assert np.allclose(sidecars["gain_raw_px"], 0.5)
    assert np.allclose(sidecars["harm_raw_px"], 0.0)
    case = receipt["case_metrics"]
    assert case["mean_native_EPE_raw_px"] == 1.0
    assert case["mean_action_EPE_raw_px"] == 0.5
    assert case["mean_gain_raw_px"] == 0.5
    assert case["mean_gain_normalized_px"] == 50.0
    assert not case["case_any_severe"]


def test_cvar_includes_zeros_and_uses_ceil_top_five_percent():
    native, action, gt, valid, labels = _inputs()
    action[:] = native
    action[0, 0, 0] = 3.0  # harm=2 at one of 20 pixels; top ceil(1)=2.
    _, receipt = _score(
        native_flow=native, action_flow=action, ground_truth_flow=gt,
        gt_valid_mask=valid, region_labels=labels,
    )
    region = receipt["regions"][0]
    assert region["harmed_pixel_fraction"] == 0.05
    assert region["pixel_harm_cvar95_raw_px"] == 2.0
    assert region["tail_violation"]


def test_fewer_than_16_pixels_is_unlabeled_not_zero_or_case_missing():
    native, action, gt, valid, labels = _inputs()
    labels[:, 4] = 1  # region 1 has four valid pixels and is unlabeled.
    _, receipt = _score(
        native_flow=native, action_flow=action, ground_truth_flow=gt,
        gt_valid_mask=valid, region_labels=labels,
    )
    assert receipt["regions"][1]["status"] == "UNLABELED_GT_SUPPORT_LT16"
    assert receipt["regions"][1]["mean_net_gain_raw_px"] is None
    assert receipt["case_metrics"]["labeled_region_count"] == 1
    assert receipt["case_metrics"]["unlabeled_region_count"] == 1


def test_action_nonfinite_on_frozen_support_cannot_shrink_denominator():
    native, action, gt, valid, labels = _inputs()
    action[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="denominator shrinking is forbidden"):
        _score(
            native_flow=native, action_flow=action, ground_truth_flow=gt,
            gt_valid_mask=valid, region_labels=labels,
        )


def test_receipt_and_sidecars_are_hash_bound():
    sidecars, receipt = _score()
    assert receipt["observation_sha256"]
    assert receipt["pixel_sidecars"]["gain_raw_px"]["array_sha256"]
    drifted = sidecars["gain_raw_px"].copy()
    drifted[0, 0] += 1.0
    assert receipt["pixel_sidecars"]["gain_raw_px"]["array_sha256"] != (
        __import__(
            "stablebridge.physical_repair.action_qualification_task_metrics_v1",
            fromlist=["array_sha256"],
        ).array_sha256(drifted)
    )


def test_spring_min_of_four_uses_complete_vectors_for_each_prediction():
    native, action, _, valid, labels = _inputs()
    native[:] = (1.0, 9.0)
    action[:] = (9.0, 1.0)
    truth = np.empty((4, 4, 5, 2), dtype=np.float32)
    truth[0] = (1.0, 10.0)   # nearest native: EPE 1
    truth[1] = (10.0, 1.0)   # nearest action: EPE 1
    truth[2] = (0.0, 0.0)
    truth[3] = (20.0, 20.0)
    sidecars, receipt = _score(
        native_flow=native, action_flow=action, ground_truth_flow=truth,
        gt_valid_mask=valid, region_labels=labels,
    )
    assert np.all(sidecars["nearest_GT_branch_native"] == 0)
    assert np.all(sidecars["nearest_GT_branch_action"] == 1)
    assert np.allclose(sidecars["epe_native_raw_px"], 1.0)
    assert np.allclose(sidecars["epe_action_raw_px"], 1.0)
    assert receipt["GT_matching"] == {
        "branch_count": 4,
        "rule": "MINIMUM_EPE_OVER_COMPLETE_GT_VECTORS_PER_PREDICTION",
        "native_and_action_may_select_different_branches": True,
        "per_coordinate_branch_mixing": False,
        "flow_units": "NATIVE_RGB_PIXELS_NO_RESOLUTION_DIVISION",
    }


def test_spring_min_of_four_never_mixes_x_and_y_from_different_branches():
    native, action, _, valid, labels = _inputs()
    native[:] = (0.0, 0.0)
    action[:] = (1.0, 1.0)
    truth = np.empty((2, 4, 5, 2), dtype=np.float32)
    truth[0] = (1.0, 100.0)
    truth[1] = (100.0, 1.0)
    sidecars, _ = _score(
        native_flow=native, action_flow=action, ground_truth_flow=truth,
        gt_valid_mask=valid, region_labels=labels,
    )
    # Per-coordinate mixing would make action EPE zero. Whole-vector matching
    # correctly leaves a large error.
    assert np.all(sidecars["epe_action_raw_px"] > 90.0)


def test_branch_specific_invalidity_uses_any_valid_complete_vector_support():
    native, action, _, _, labels = _inputs()
    truth = np.zeros((2, 4, 5, 2), dtype=np.float32)
    truth[0, 0, 0] = np.nan
    branch_valid = np.ones((2, 4, 5), dtype=np.bool_)
    branch_valid[0, 0, 0] = False
    sidecars, receipt = _score(
        native_flow=native, action_flow=action, ground_truth_flow=truth,
        gt_valid_mask=branch_valid, region_labels=labels,
    )
    assert sidecars["gt_valid"][0, 0]
    assert sidecars["gt_valid_branch_count"][0, 0] == 1
    assert receipt["fixed_GT_support"]["support_rule"] == (
        "AT_LEAST_ONE_VALID_COMPLETE_GT_VECTOR"
    )
