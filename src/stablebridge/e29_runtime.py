"""Streaming, resumable execution support for the E29 joint KITTI panel.

The model-specific adapter supplies a frozen predictor.  This module owns the
parts that must be identical for StableBridge and PairBridge: pair exposure,
ground-truth binding, metric computation, output hashing, one-row audit writes,
resume semantics, and the H2 seal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from .bridge_runtime import (
    ActionSpec,
    BridgeDecision,
    FLOW_PAIR2,
    STEREO_CORE3,
    plan_direct_execution,
    validate_registry,
)
from .kitti_scene_flow import (
    KittiScenePair,
    evaluate_disparity,
    evaluate_flow,
    read_disparity,
    read_flow,
)


E29_ROW_SCHEMA = "bridge-e29-stream-row/v1"
EXECUTION_MODES = {"direct", "always_compute"}


@dataclass(frozen=True)
class RuntimeMeasurement:
    total_seconds: float
    observer_seconds: float
    expert_seconds: float
    expert_forwards: int
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None

    def __post_init__(self) -> None:
        values = (self.total_seconds, self.observer_seconds, self.expert_seconds)
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("runtime measurements must be finite and nonnegative")
        if self.expert_forwards < 1:
            raise ValueError("each completed row must execute at least one expert")
        if self.observer_seconds + self.expert_seconds > self.total_seconds + 1e-6:
            raise ValueError("observer plus expert time cannot exceed total time")
        for value in (self.peak_allocated_bytes, self.peak_reserved_bytes):
            if value is not None and value < 0:
                raise ValueError("memory measurements must be nonnegative")


@dataclass(frozen=True)
class BridgePrediction:
    prediction: np.ndarray
    decision: BridgeDecision
    runtime: RuntimeMeasurement
    execution_mode: str = "direct"
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.execution_mode not in EXECUTION_MODES:
            raise ValueError(f"execution_mode must be one of {sorted(EXECUTION_MODES)}")


Predictor = Callable[[np.ndarray, np.ndarray, str], BridgePrediction]
Renderer = Callable[..., Mapping[str, object]]


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        item = np.ascontiguousarray(value)
        digest.update(str(item.dtype).encode("utf-8"))
        digest.update(np.asarray(item.shape, dtype=np.int64).tobytes())
        digest.update(item.tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def assert_panel_access(role: str, h1_gate_path: str | Path | None = None) -> None:
    """Fail before image access when H2 has not been explicitly unlocked."""
    if role not in {"C1", "H1", "H2"}:
        raise ValueError("role must be C1, H1, or H2")
    if role != "H2":
        return
    if h1_gate_path is None:
        raise PermissionError("H2 is sealed until a passing H1 gate is supplied")
    gate_path = Path(h1_gate_path)
    if not gate_path.is_file():
        raise PermissionError(f"H2 gate does not exist: {gate_path}")
    gate = json.loads(gate_path.read_text())
    if gate.get("schema") != "bridge-e29-h1-gate/v1" or gate.get("advance_to_h2") is not True:
        raise PermissionError("H2 remains sealed because the H1 gate did not pass")
    if (gate.get("transfer_setting") != "T1_calibrated"
            or gate.get("primary_h2_gate_analysis") is not True):
        raise PermissionError("H2 requires the predeclared primary T1 H1 gate")
    analysis_path = Path(gate.get("analysis_path", ""))
    if not analysis_path.is_file() or file_sha256(analysis_path) != gate.get("analysis_sha256"):
        raise PermissionError("H2 gate analysis artifact is missing or has changed")


def _completed_keys(path: Path) -> set[tuple[str, str, str, str, str, str]]:
    keys: set[tuple[str, str, str, str, str, str]] = set()
    if not path.exists():
        return keys
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        row = json.loads(line)
        if row.get("schema") != E29_ROW_SCHEMA:
            raise ValueError(f"unexpected row schema at {path}:{line_number}")
        key = tuple(row[name] for name in (
            "system_id", "execution_mode", "task", "scene_id", "condition", "exposure"
        ))
        if key in keys:
            raise ValueError(f"duplicate completed row at {path}:{line_number}: {key}")
        keys.add(key)
    return keys


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _task_material(pair: KittiScenePair, task: str):
    if task == "stereo":
        first, second = _read_rgb(pair.stereo_left), _read_rgb(pair.stereo_right)
        truth, valid = read_disparity(pair.stereo_disparity)
        label_path = pair.stereo_disparity
    elif task == "flow":
        first, second = _read_rgb(pair.flow_source), _read_rgb(pair.flow_target)
        truth, valid = read_flow(pair.flow_ground_truth)
        label_path = pair.flow_ground_truth
    else:
        raise ValueError("task must be stereo or flow")
    if first.shape != second.shape or first.shape[:2] != valid.shape:
        raise ValueError(f"KITTI {task} input/label shape mismatch for scene {pair.scene_id}")
    return first, second, truth, valid, label_path


def _variants(first: np.ndarray, second: np.ndarray, *, sample_key: str,
              render_condition_variants: Renderer,
              conditions: Sequence[str]):
    yield "clean", "clean", first, second, {
        "renderer": "identity",
        "sample_key": sample_key,
        "condition": "clean",
        "exposure": "clean",
        "changes_geometry": False,
    }
    for condition in conditions:
        rendered = render_condition_variants(
            first, second, sample_key=sample_key, condition=condition
        )
        if set(rendered) != {"first", "second", "both"}:
            raise ValueError(f"renderer returned unexpected exposure set for {condition}")
        for exposure in ("first", "second", "both"):
            item = rendered[exposure]
            metadata = dict(item.metadata)
            if metadata.get("changes_geometry") is not False:
                raise ValueError("E29 core renderer may not change geometry")
            yield condition, exposure, item.first, item.second, metadata


def run_pair_panel(
    *,
    pair: KittiScenePair,
    task: str,
    role: str,
    system_id: str,
    execution_mode: str,
    predictor: Predictor,
    render_condition_variants: Renderer,
    conditions: Sequence[str],
    output_path: str | Path,
    h1_gate_path: str | Path | None = None,
    registry: Sequence[ActionSpec] | None = None,
) -> dict[str, int]:
    """Run all 13 frozen conditions for one task/scene and append audit rows.

    The function is safe to rerun. Completed rows are validated and skipped;
    dense predictions are hashed and released after their scalar metrics have
    been written.
    """
    assert_panel_access(role, h1_gate_path)
    if not system_id:
        raise ValueError("system_id must be nonempty")
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(f"execution_mode must be one of {sorted(EXECUTION_MODES)}")
    registry = tuple(registry or (STEREO_CORE3 if task == "stereo" else FLOW_PAIR2))
    validate_registry(registry)
    output_path = Path(output_path)
    completed = _completed_keys(output_path)
    first, second, truth, valid, label_path = _task_material(pair, task)
    base_sample_key = f"kitti2015:{task}:{pair.scene_id}"
    added = skipped = 0
    for condition, exposure, observed_first, observed_second, renderer_metadata in _variants(
        first,
        second,
        sample_key=base_sample_key,
        render_condition_variants=render_condition_variants,
        conditions=conditions,
    ):
        row_key = f"{base_sample_key}:{condition}:{exposure}"
        key = (system_id, execution_mode, task, pair.scene_id, condition, exposure)
        if key in completed:
            skipped += 1
            continue
        started = time.perf_counter()
        result = predictor(observed_first, observed_second, row_key)
        call_wall_seconds = time.perf_counter() - started
        if result.execution_mode != execution_mode:
            raise ValueError("predictor execution mode does not match the requested shard")
        if result.decision.task != task or result.decision.sample_key != row_key:
            raise ValueError("predictor decision does not identify the current task/row")
        plan = plan_direct_execution(result.decision, registry)
        if result.execution_mode == "direct" and result.runtime.expert_forwards != 1:
            raise ValueError("direct E29 execution must run exactly one expert")
        prediction = np.asarray(result.prediction)
        metric = (
            evaluate_disparity(prediction, truth, valid)
            if task == "stereo"
            else evaluate_flow(prediction, truth, valid)
        )
        input_digest = array_sha256(observed_first, observed_second)
        row = {
            "schema": E29_ROW_SCHEMA,
            "system_id": system_id,
            "execution_mode": result.execution_mode,
            "task": task,
            "role": role,
            "scene_id": pair.scene_id,
            "condition": condition,
            "exposure": exposure,
            "is_clean": condition == "clean",
            "row_key": row_key,
            "input_sha256": input_digest,
            "prediction_sha256": array_sha256(prediction),
            "ground_truth_sha256": file_sha256(label_path),
            "selected_action_id": plan.selected_action_id,
            "fallback_action_id": plan.fallback_action_id,
            "intervened": plan.selected_action_id != plan.fallback_action_id,
            "executor_id": plan.executor_id,
            "policy_id": plan.policy_id,
            "decision_source": plan.decision_source,
            "observer_ids": list(plan.observer_ids),
            "calibration_id": plan.calibration_id,
            "metric": asdict(metric),
            "runtime": {**asdict(result.runtime), "predictor_call_wall_seconds": call_wall_seconds},
            "renderer": renderer_metadata,
            "metadata": dict(result.metadata or {}),
        }
        _append_jsonl(output_path, row)
        completed.add(key)
        added += 1
        del prediction, result
    return {"added": added, "skipped": skipped, "total": added + skipped}
