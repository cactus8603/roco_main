"""Independent integer PNG and little-endian FLO conversion examples."""
import struct

import cv2
import numpy as np
import pytest

from waft_repair.backend import load_ground_truth
from waft_repair.ground_truth import convert_ground_truth


def kitti(path, labels):
    assert cv2.imwrite(str(path), labels)
    return path


def flo(path, flow):
    height, width = flow.shape[:2]
    path.write_bytes(struct.pack("<fii", 202021.25, width, height) + flow.astype("<f4").tobytes())
    return path


def test_kitti_integer_values_channels_and_separate_visibility(tmp_path):
    labels = np.array([[[1, 32704, 32800], [1, 32896, 32640], [0, 65535, 0]]], dtype=np.uint16)
    # Noc vector values deliberately differ: only its validity marker is used.
    noc = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.uint16)
    result = convert_ground_truth("kitti", kitti(tmp_path / "occ.png", labels), tmp_path / "converted.npz",
                                  kitti(tmp_path / "noc.png", noc))
    with np.load(tmp_path / "converted.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["flow"], np.array([[[.5, -1.], [-2., 2.], [-512., 511.984375]]], dtype=np.float32))
        np.testing.assert_array_equal(saved["valid"], [[True, True, False]])
        np.testing.assert_array_equal(saved["visible"], [[True, False, False]])
        assert saved["flow"].dtype == np.float32 and saved["valid"].dtype == saved["visible"].dtype == np.bool_
    # Visibility must not shrink evaluation validity.
    gt, valid, visible = load_ground_truth(tmp_path / "converted.npz")
    assert gt.shape == (1, 2, 1, 3) and valid.sum() == 2 and visible.sum() == 1
    assert result["valid_pixels"] == 2 and result["visible_pixels"] == 1


def test_kitti_rejects_wrong_png_dtype_and_visibility_size(tmp_path):
    bad = kitti(tmp_path / "bad.png", np.ones((2, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint16"):
        convert_ground_truth("kitti", bad, tmp_path / "out.npz")
    good = kitti(tmp_path / "good.png", np.ones((2, 3, 3), dtype=np.uint16))
    mismatch = kitti(tmp_path / "mismatch.png", np.ones((2, 4, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="dimensions differ"):
        convert_ground_truth("kitti", good, tmp_path / "out.npz", mismatch)
    assert not (tmp_path / "out.npz").exists()


def test_sintel_little_endian_values_validity_threshold_and_visibility(tmp_path):
    flow = np.array([[[.25, -1.5], [999.5, -999.5], [1000., 0.]],
                     [[-1000., 0.], [np.nan, 2.], [1., np.inf]]], dtype=np.float32)
    path = flo(tmp_path / "flow.flo", flow)
    mask = np.array([[0, 255, 0], [255, 0, 0]], np.uint8)
    assert cv2.imwrite(str(tmp_path / "occlusions.png"), mask)
    result = convert_ground_truth("sintel", path, tmp_path / "converted.npz", tmp_path / "occlusions.png")
    with np.load(tmp_path / "converted.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["flow"], flow)
        np.testing.assert_array_equal(saved["valid"], [[True, True, False], [False, False, False]])
        np.testing.assert_array_equal(saved["visible"], [[True, False, True], [False, True, True]])
    assert result["native_hw"] == [2, 3] and result["valid_pixels"] == 2


@pytest.mark.parametrize("payload", [b"short", struct.pack("<fii", 1., 2, 2), struct.pack("<fii", 202021.25, -2, 2),
                                     struct.pack("<fii", 202021.25, 2, 2) + b"\0" * 28,
                                     struct.pack("<fii", 202021.25, 2, 2) + b"\0" * 36])
def test_sintel_rejects_incomplete_header_bad_magic_and_nonexact_payload(tmp_path, payload):
    (tmp_path / "bad.flo").write_bytes(payload)
    with pytest.raises(ValueError):
        convert_ground_truth("sintel", tmp_path / "bad.flo", tmp_path / "out.npz")
    assert not (tmp_path / "out.npz").exists()


def test_optional_visibility_and_no_overwrite(tmp_path):
    path = flo(tmp_path / "flow.flo", np.zeros((2, 3, 2), dtype=np.float32))
    output = tmp_path / "converted.npz"
    result = convert_ground_truth("sintel", path, output)
    before = output.read_bytes()
    with np.load(output, allow_pickle=False) as saved:
        assert set(saved.files) == {"flow", "valid"}
    assert result["visibility_supplied"] is False
    with pytest.raises(ValueError, match="already exists"):
        convert_ground_truth("sintel", path, output)
    assert output.read_bytes() == before


def test_sintel_rejects_wrong_occlusion_channels_and_empty_validity(tmp_path):
    path = flo(tmp_path / "flow.flo", np.zeros((2, 3, 2), dtype=np.float32))
    mask = kitti(tmp_path / "mask.png", np.zeros((2, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="single-channel"):
        convert_ground_truth("sintel", path, tmp_path / "out.npz", mask)
    path = flo(tmp_path / "invalid.flo", np.full((2, 3, 2), 1000., dtype=np.float32))
    with pytest.raises(ValueError, match="valid original GT"):
        convert_ground_truth("sintel", path, tmp_path / "out.npz")
