from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from stablebridge.bridge_runtime import stablebridge_s0_decision
from stablebridge.e29_runtime import (
    BridgePrediction,
    RuntimeMeasurement,
    assert_panel_access,
    run_pair_panel,
)
from stablebridge.kitti_scene_flow import KittiScenePair


ROOT = Path(__file__).resolve().parents[2]
RENDERER_PATH = ROOT / "experiments/E29_external_joint_panel/render_conditions.py"
SPEC = importlib.util.spec_from_file_location("e29_runtime_test_renderer", RENDERER_PATH)
assert SPEC is not None and SPEC.loader is not None
RENDERER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RENDERER
SPEC.loader.exec_module(RENDERER)


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _pair(tmp_path: Path) -> KittiScenePair:
    height, width = 24, 32
    yy, xx = np.mgrid[:height, :width]
    first = np.stack((xx * 7, yy * 9, (xx + yy) * 5), axis=-1).clip(0, 255).astype(np.uint8)
    second = np.flip(first, axis=1).copy()
    left = tmp_path / "image_2/000000_10.png"
    right = tmp_path / "image_3/000000_10.png"
    target = tmp_path / "image_2/000000_11.png"
    _write_rgb(left, first)
    _write_rgb(right, second)
    _write_rgb(target, np.roll(first, 1, axis=1))

    disparity_path = tmp_path / "disp_occ_0/000000_10.png"
    disparity_path.parent.mkdir(parents=True)
    disparity = np.full((height, width), 10 * 256, np.uint16)
    assert cv2.imwrite(str(disparity_path), disparity)

    flow_path = tmp_path / "flow_occ/000000_10.png"
    flow_path.parent.mkdir(parents=True)
    valid = np.ones((height, width), np.uint16)
    encoded = np.stack((valid, np.full_like(valid, 32768), np.full_like(valid, 32768)), -1)
    assert cv2.imwrite(str(flow_path), encoded)
    return KittiScenePair("000000", left, right, disparity_path, left, target, flow_path)


def test_streaming_panel_is_complete_resumable_and_discards_dense_output(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    calls = []

    def predictor(first: np.ndarray, second: np.ndarray, row_key: str) -> BridgePrediction:
        calls.append(row_key)
        return BridgePrediction(
            prediction=np.full(first.shape[:2], 10.0, np.float32),
            decision=stablebridge_s0_decision(row_key, 0),
            runtime=RuntimeMeasurement(1.0, .05, .8, 1, 100, 200),
            execution_mode="direct",
        )

    rows_path = tmp_path / "rows.jsonl"
    first = run_pair_panel(
        pair=pair,
        task="stereo",
        role="C1",
        system_id="task_fallback",
        execution_mode="direct",
        predictor=predictor,
        render_condition_variants=RENDERER.render_condition_variants,
        conditions=RENDERER.CONDITIONS,
        output_path=rows_path,
    )
    assert first == {"added": 13, "skipped": 0, "total": 13}
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert len(rows) == 13
    assert sum(row["is_clean"] for row in rows) == 1
    assert all(row["metric"]["mean_endpoint_error"] == 0 for row in rows)
    assert all(row["metric"]["endpoint_error_sum"] == 0 for row in rows)
    assert all(row["metric"]["outlier_pixels"] == 0 for row in rows)
    assert all(row["runtime"]["expert_forwards"] == 1 for row in rows)
    assert all(len(row["prediction_sha256"]) == 64 for row in rows)
    assert not list(tmp_path.rglob("*.npy"))

    second = run_pair_panel(
        pair=pair,
        task="stereo",
        role="C1",
        system_id="task_fallback",
        execution_mode="direct",
        predictor=predictor,
        render_condition_variants=RENDERER.render_condition_variants,
        conditions=RENDERER.CONDITIONS,
        output_path=rows_path,
    )
    assert second == {"added": 0, "skipped": 13, "total": 13}
    assert len(calls) == 13  # Resume skips before an expensive predictor call.


def test_h2_is_checked_before_any_dataset_access(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    pair = KittiScenePair("000001", missing, missing, missing, missing, missing, missing)
    with pytest.raises(PermissionError, match="sealed"):
        run_pair_panel(
            pair=pair,
            task="stereo",
            role="H2",
            system_id="task_fallback",
            execution_mode="direct",
            predictor=lambda *_: None,  # type: ignore[arg-type]
            render_condition_variants=RENDERER.render_condition_variants,
            conditions=RENDERER.CONDITIONS,
            output_path=tmp_path / "rows.jsonl",
        )

    analysis = tmp_path / "h1_analysis.json"
    analysis.write_text('{"status":"PASS"}\n')
    gate = tmp_path / "h1_gate.json"
    gate.write_text(json.dumps({
        "schema": "bridge-e29-h1-gate/v1",
        "advance_to_h2": True,
        "transfer_setting": "T1_calibrated",
        "primary_h2_gate_analysis": True,
        "analysis_path": str(analysis),
        "analysis_sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
    }))
    assert_panel_access("H2", gate)

    value = json.loads(gate.read_text())
    value["transfer_setting"] = "T0_frozen"
    gate.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="primary T1"):
        assert_panel_access("H2", gate)
