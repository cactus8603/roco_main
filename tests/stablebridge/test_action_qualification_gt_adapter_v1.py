from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from stablebridge.physical_repair.action_qualification_gt_adapter_v1 import (
    decode_bound_flow_GT_v1,
)


def _binding(path: Path):
    data = path.read_bytes()
    return {
        "absolute_external_read_only": True,
        "modality": "flow_ground_truth",
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_spring_flo5_decodes_four_complete_native_pixel_branches(tmp_path):
    raw = np.zeros((4, 6, 2), dtype=np.float32)
    for y in range(4):
        for x in range(6):
            raw[y, x] = (10 * y + x, -(10 * y + x))
    path = tmp_path / "flow.flo5"
    with h5py.File(path, "w") as stream:
        stream.create_dataset("flow", data=raw)
    flows, valid, receipt = decode_bound_flow_GT_v1(
        source_dataset="spring", binding=_binding(path), lattice_hw=(2, 3),
        component_id="spring-component",
    )
    assert flows.shape == (4, 2, 3, 2)
    assert valid.shape == (4, 2, 3)
    assert np.array_equal(flows[0], raw[0::2, 0::2])
    assert np.array_equal(flows[1], raw[1::2, 0::2])
    assert receipt["flow_units"] == "NATIVE_RGB_PIXELS_NO_RESOLUTION_DIVISION"
    assert not receipt["spatial_resize"]


def test_spring_branch_invalidity_is_preserved_not_pixel_dropped(tmp_path):
    raw = np.zeros((4, 4, 2), dtype=np.float32)
    raw[0, 0] = np.nan
    path = tmp_path / "flow.flo5"
    with h5py.File(path, "w") as stream:
        stream.create_dataset("flow", data=raw)
    _, valid, receipt = decode_bound_flow_GT_v1(
        source_dataset="spring", binding=_binding(path), lattice_hw=(2, 2),
        component_id="spring-component",
    )
    assert not valid[0, 0, 0]
    assert valid[1:, 0, 0].all()
    assert receipt["fixed_pixel_support_count"] == 4


def test_kitti_uint16_bgr_decode_preserves_sparse_validity(tmp_path):
    encoded = np.zeros((2, 3, 3), dtype=np.uint16)
    encoded[0, 1] = (1, 32768 + 64, 32768 + 128)  # valid, v=1, u=2
    path = tmp_path / "flow.png"
    assert cv2.imwrite(str(path), encoded)
    flows, valid, receipt = decode_bound_flow_GT_v1(
        source_dataset="kitti", binding=_binding(path), lattice_hw=(2, 3),
        component_id="kitti-component",
    )
    assert flows.shape == (1, 2, 3, 2)
    assert valid.shape == (1, 2, 3)
    assert valid[0, 0, 1]
    assert np.array_equal(flows[0, 0, 1], np.array([2.0, 1.0], np.float32))
    assert receipt["fixed_pixel_support_count"] == 1


def test_hash_or_geometry_drift_fails_closed(tmp_path):
    raw = np.zeros((4, 4, 2), dtype=np.float32)
    path = tmp_path / "flow.flo5"
    with h5py.File(path, "w") as stream:
        stream.create_dataset("flow", data=raw)
    binding = _binding(path)
    binding["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash drift"):
        decode_bound_flow_GT_v1(
            source_dataset="spring", binding=binding, lattice_hw=(2, 2),
            component_id="component",
        )
    with pytest.raises(ValueError, match="exactly 2x"):
        decode_bound_flow_GT_v1(
            source_dataset="spring", binding=_binding(path), lattice_hw=(3, 3),
            component_id="component",
        )
