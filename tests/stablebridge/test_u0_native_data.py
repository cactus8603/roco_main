from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from stablebridge.physical_repair.u0_native_data import (
    U0NativeDataPlanV1,
    U0ProbeSpecV1,
    apply_u0_probe_v1,
    build_u0_native_data_plan_v1,
    evenly_spaced_frames_v1,
    restore_u0_probe_flow_v1,
    verify_u0_native_data_plan_file_v1,
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fixture(tmp_path: Path) -> Path:
    rgb = tmp_path / "rgb"
    flow = tmp_path / "flow"
    disp = tmp_path / "disp"
    scenes = ("0001", "0002", "0003", "0004")
    for scene_index, scene in enumerate(scenes):
        for frame in (1, 2, 3):
            image = np.full((64, 96, 3), scene_index * 20 + frame, np.uint8)
            path = rgb / "train" / scene / "frame_left" / f"frame_left_{frame:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(path)
        for frame in (1, 2):
            path = flow / "train" / scene / "flow_FW_left" / f"flow_FW_left_{frame:04d}.flo5"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"label-never-opened-" + scene.encode() + bytes([frame]))
    checkpoint = tmp_path / "matcher.pth"
    checkpoint.write_bytes(b"frozen-matcher")
    registry = {
        "datasets": {"spring": {
            "rgb_root": str(rgb), "flow_root": str(flow), "disp_root": str(disp),
            "dataset_version": "synthetic-v1",
        }},
        "models": {"main": {"flow": {
            "local_path": str(checkpoint), "sha256": digest(checkpoint.read_bytes()),
            "metadata": {"size_bytes": checkpoint.stat().st_size},
        }}},
    }
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    config = {
        "contract": "CSB-U0-NATIVE-FLOW-DATA-v1-20261005",
        "plan_id": "synthetic-u0-plan-v1",
        "registry": "registry.json",
        "dataset_key": "spring",
        "model_profile": "main",
        "matcher_provider_id": "synthetic-native-flow",
        "task": "flow",
        "dataset_split": "train",
        "view": "left",
        "context_hw": [32, 64],
        "crop_policy": "central_native_no_resize",
        "frames_per_scene": 2,
        "split_id": "synthetic-scene-split-v1",
        "splits": {
            "fit": ["0001"], "validation": ["0002"],
            "calibration": ["0003"], "evaluation": ["0004"],
        },
        "probes": [
            {"probe_id": "roll-x8", "transform": "shared_translation", "shift_xy": [8, 0], "scalar": 1.0},
            {"probe_id": "exposure-080", "transform": "shared_exposure", "shift_xy": [0, 0], "scalar": 0.8},
        ],
        "scientific_scope": "synthetic test only",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_build_plan_is_deterministic_scene_disjoint_and_label_unopened(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    first = build_u0_native_data_plan_v1(config, root=tmp_path)
    second = build_u0_native_data_plan_v1(config, root=tmp_path)
    assert first == second
    assert first.plan_hash == second.plan_hash
    assert len(first.rows) == 8
    assert first.as_dict()["labels_opened"] is False
    assert first.as_dict()["matcher_outputs_materialized"] is False
    assert first.as_dict()["role_group_counts"] == {
        "fit": 1, "validation": 1, "calibration": 1, "evaluation": 1,
    }
    assert all(row.label_present for row in first.rows)
    assert all(row.roi_xyhw == (16, 16, 32, 64) for row in first.rows)
    assert all(row.source_frame + 1 == row.target_frame for row in first.rows)
    serialized = tmp_path / "plan.json"
    serialized.write_text(json.dumps(first.as_dict()))
    assert verify_u0_native_data_plan_file_v1(serialized) == first


def test_input_content_change_changes_plan_identity(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    first = build_u0_native_data_plan_v1(config, root=tmp_path)
    image_path = Path(first.rows[0].input_paths[0])
    image = np.array(Image.open(image_path).convert("RGB"), copy=True)
    image[0, 0, 0] += 1
    Image.fromarray(image).save(image_path)
    second = build_u0_native_data_plan_v1(config, root=tmp_path)
    assert first.plan_hash != second.plan_hash
    assert first.rows[0].input_hashes != second.rows[0].input_hashes


def test_missing_endpoint_and_scene_leakage_fail_closed(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    payload = json.loads(config.read_text())
    payload["splits"]["evaluation"] = ["0001"]
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="more than one split"):
        build_u0_native_data_plan_v1(config, root=tmp_path)

    config = fixture(tmp_path / "missing")
    missing = tmp_path / "missing" / "rgb/train/0001/frame_left/frame_left_0003.png"
    missing.unlink()
    with pytest.raises(ValueError, match="fewer valid frames"):
        payload = json.loads(config.read_text())
        payload["frames_per_scene"] = 2
        config.write_text(json.dumps(payload))
        build_u0_native_data_plan_v1(config, root=tmp_path / "missing")


def test_forbidden_metadata_and_plan_drift_are_rejected(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    payload = json.loads(config.read_text())
    payload["selected_control"] = "forbidden"
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="forbids action/control field"):
        build_u0_native_data_plan_v1(config, root=tmp_path)

    config = fixture(tmp_path / "drift")
    plan = build_u0_native_data_plan_v1(config, root=tmp_path / "drift")
    with pytest.raises(ValueError, match="probe recipe drift"):
        rows = list(plan.rows)
        rows[0] = replace(rows[0], probe_hashes=(digest(b"different"),), row_hash="")
        replace(plan, rows=tuple(rows), plan_hash="")

    serialized = plan.as_dict()
    serialized["labels_opened"] = True
    path = tmp_path / "drift" / "opened.json"
    path.write_text(json.dumps(serialized))
    with pytest.raises(ValueError, match="labels_opened must remain false"):
        verify_u0_native_data_plan_file_v1(path)


def test_probe_transforms_and_inverse_flow_are_deterministic() -> None:
    image = np.arange(32 * 64 * 3, dtype=np.uint8).reshape(32, 64, 3)
    flow = np.arange(2 * 32 * 64, dtype=np.float32).reshape(2, 32, 64)
    translation = U0ProbeSpecV1("roll-x8", "shared_translation", (8, 0), 1.0)
    first, second, support = apply_u0_probe_v1(image, image, translation)
    np.testing.assert_array_equal(first, second)
    assert not support[:, :8].any() and not support[:, -8:].any()
    shifted = np.roll(flow, (0, 8), axis=(1, 2))
    np.testing.assert_array_equal(restore_u0_probe_flow_v1(shifted, translation), flow)

    exposure = U0ProbeSpecV1("exposure-080", "shared_exposure", (0, 0), 0.8)
    low_a, _, full = apply_u0_probe_v1(image, image, exposure)
    low_b, _, _ = apply_u0_probe_v1(image, image, exposure)
    np.testing.assert_array_equal(low_a, low_b)
    assert full.all()
    np.testing.assert_array_equal(restore_u0_probe_flow_v1(flow, exposure), flow)


def test_even_frame_selection_is_endpoint_inclusive_and_unique() -> None:
    assert evenly_spaced_frames_v1(range(1, 13), 8) == (1, 3, 4, 6, 7, 9, 10, 12)
    assert evenly_spaced_frames_v1(range(1, 13), 1) == (7,)
    with pytest.raises(ValueError, match="fewer valid frames"):
        evenly_spaced_frames_v1((1, 2), 3)
