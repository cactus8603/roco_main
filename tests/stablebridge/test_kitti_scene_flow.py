from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from stablebridge.kitti_scene_flow import (
    discover_training_pairs,
    evaluate_disparity,
    evaluate_flow,
    official_outlier_mask,
    read_disparity,
    read_flow,
)


def test_official_outlier_uses_absolute_and_relative_thresholds() -> None:
    error = np.array([2.9, 4.0, 4.0, 3.0, 5.1])
    magnitude = np.array([10.0, 100.0, 10.0, 10.0, 100.0])
    assert official_outlier_mask(error, magnitude).tolist() == [False, False, True, False, True]


def test_kitti_png_decode_and_metrics(tmp_path: Path) -> None:
    disparity_encoded = np.array([[0, 640], [2560, 25600]], dtype=np.uint16)
    disparity_path = tmp_path / "disp.png"
    assert cv2.imwrite(str(disparity_path), disparity_encoded)
    disparity, disparity_valid = read_disparity(disparity_path)
    np.testing.assert_allclose(disparity, [[0.0, 2.5], [10.0, 100.0]])
    assert disparity_valid.tolist() == [[False, True], [True, True]]
    disparity_prediction = disparity.copy()
    disparity_prediction[1, 0] += 4.0
    disparity_prediction[1, 1] += 4.0
    disparity_metric = evaluate_disparity(disparity_prediction, disparity, disparity_valid)
    assert disparity_metric.valid_pixels == 3
    assert disparity_metric.endpoint_error_sum == pytest.approx(8.0)
    assert disparity_metric.outlier_pixels == 1
    assert disparity_metric.outlier_percent == pytest.approx(100.0 / 3.0)

    u = np.array([[0.0, 4.0], [-2.0, 100.0]], dtype=np.float32)
    v = np.array([[0.0, -3.0], [2.0, 0.0]], dtype=np.float32)
    valid = np.array([[0, 1], [1, 1]], dtype=np.uint16)
    flow_encoded = np.stack(
        (
            valid,
            np.rint(v * 64.0 + 32768.0).astype(np.uint16),
            np.rint(u * 64.0 + 32768.0).astype(np.uint16),
        ),
        axis=-1,
    )
    flow_path = tmp_path / "flow.png"
    assert cv2.imwrite(str(flow_path), flow_encoded)
    flow, flow_valid = read_flow(flow_path)
    np.testing.assert_allclose(flow[..., 0], u)
    np.testing.assert_allclose(flow[..., 1], v)
    assert flow_valid.tolist() == [[False, True], [True, True]]
    prediction = flow.copy()
    prediction[0, 1, 0] += 4.0  # magnitude 5: both thresholds exceeded
    prediction[1, 1, 0] += 4.0  # magnitude 100: relative threshold not exceeded
    flow_metric = evaluate_flow(prediction, flow, flow_valid)
    assert flow_metric.valid_pixels == 3
    assert flow_metric.endpoint_error_sum == pytest.approx(8.0)
    assert flow_metric.outlier_pixels == 1
    assert flow_metric.outlier_percent == pytest.approx(100.0 / 3.0)


def test_nonfinite_dense_prediction_is_rejected_instead_of_reducing_density() -> None:
    disparity = np.ones((2, 2), np.float32)
    valid = np.ones((2, 2), bool)
    disparity_prediction = disparity.copy()
    disparity_prediction[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite disparity prediction"):
        evaluate_disparity(disparity_prediction, disparity, valid)

    flow = np.zeros((2, 2, 2), np.float32)
    flow_prediction = flow.copy()
    flow_prediction[1, 1, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite flow prediction"):
        evaluate_flow(flow_prediction, flow, valid)


def test_discover_binds_same_scene_stereo_and_flow(tmp_path: Path) -> None:
    training = tmp_path / "training"
    for directory in ("image_2", "image_3", "disp_occ_0", "flow_occ"):
        (training / directory).mkdir(parents=True)
    for relative in (
        "image_2/000000_10.png",
        "image_2/000000_11.png",
        "image_3/000000_10.png",
        "disp_occ_0/000000_10.png",
        "flow_occ/000000_10.png",
    ):
        (training / relative).write_bytes(b"exists")
    pairs = discover_training_pairs(tmp_path, require_count=1)
    assert len(pairs) == 1
    assert pairs[0].scene_id == "000000"
    assert pairs[0].stereo_left == pairs[0].flow_source
    assert pairs[0].flow_target.name == "000000_11.png"
