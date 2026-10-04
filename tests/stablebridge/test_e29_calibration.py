from __future__ import annotations

from copy import deepcopy

import pytest

from stablebridge.e29_calibration import calibrate_c1_rows


def _metric(epe: float, outlier: float) -> dict:
    return {
        "valid_pixels": 100,
        "endpoint_error_sum": epe * 100,
        "outlier_pixels": int(outlier),
        "mean_endpoint_error": epe,
        "outlier_percent": outlier,
    }


def _rows() -> list[dict]:
    rows = []
    for task in ("stereo", "flow"):
        fallback = "identity" if task == "stereo" else "cc"
        candidate_actions = ("identity", "denoise", "sharpen") if task == "stereo" else ("cc", "rr")
        native_mode = "direct" if task == "stereo" else "always_compute"
        for scene in ("000000", "000001"):
            conditions = [("clean", "clean")]
            conditions.extend(
                (f"condition_{index // 3}", ("first", "second", "both")[index % 3])
                for index in range(12)
            )
            for condition, exposure in conditions:
                clean = condition == "clean"
                key = (scene, condition, exposure)
                candidate_metric = {}
                for action in candidate_actions:
                    if clean:
                        candidate_metric[action] = _metric(1.0 if action == fallback else 1.005,
                                                           2.0 if action == fallback else 2.1)
                    elif task == "stereo":
                        candidate_metric[action] = {
                            "identity": _metric(10.0, 20.0),
                            "denoise": _metric(8.0, 16.0),
                            "sharpen": _metric(9.0, 18.0),
                        }[action]
                    else:
                        candidate_metric[action] = {
                            "cc": _metric(10.0, 20.0), "rr": _metric(7.0, 14.0)
                        }[action]
                    rows.append({
                        "schema": "bridge-e29-stream-row/v1", "role": "C1", "task": task,
                        "system_id": f"candidate:{action}", "execution_mode": "direct",
                        "scene_id": scene, "condition": condition, "exposure": exposure,
                        "selected_action_id": action, "fallback_action_id": fallback,
                        "prediction_sha256": f"{task}:{key}:{action}",
                        "metric": candidate_metric[action], "metadata": {},
                    })
                rows.append({
                    "schema": "bridge-e29-stream-row/v1", "role": "C1", "task": task,
                    "system_id": "task_fallback", "execution_mode": "direct",
                    "scene_id": scene, "condition": condition, "exposure": exposure,
                    "selected_action_id": fallback, "fallback_action_id": fallback,
                    "prediction_sha256": f"{task}:{key}:{fallback}",
                    "metric": candidate_metric[fallback], "metadata": {},
                })
                native_action = fallback if clean else ("denoise" if task == "stereo" else "rr")
                rows.append({
                    "schema": "bridge-e29-stream-row/v1", "role": "C1", "task": task,
                    "system_id": "native_bridge", "execution_mode": native_mode,
                    "scene_id": scene, "condition": condition, "exposure": exposure,
                    "selected_action_id": native_action, "fallback_action_id": fallback,
                    "prediction_sha256": f"{task}:{key}:{native_action}",
                    "metric": candidate_metric[native_action], "metadata": {},
                })
                if task == "stereo":
                    final_action = fallback if clean else "denoise"
                    metadata = {"route": {
                        "threshold": 0.699835896,
                        "selected_treatment_indices": [0, 0, 0, 0, 0],
                        "selected_score": [0.1] * 5 if clean else [0.8] * 5,
                    }}
                else:
                    final_action = fallback if clean else "rr"
                    probability = 0.1 if clean else 0.9
                    metadata = {"policy": {
                        "action_probability": [[probability] * 6 for _ in range(3)],
                        "firewall_probability": [[probability] * 6 for _ in range(3)],
                        "action_threshold": [[0.5] * 6 for _ in range(3)],
                        "firewall_threshold": [[0.5] * 6 for _ in range(3)],
                    }}
                rows.append({
                    "schema": "bridge-e29-stream-row/v1", "role": "C1", "task": task,
                    "system_id": "final_asymmetric_bridge_first", "execution_mode": "direct",
                    "scene_id": scene, "condition": condition, "exposure": exposure,
                    "selected_action_id": final_action, "fallback_action_id": fallback,
                    "prediction_sha256": f"{task}:{key}:{final_action}",
                    "metric": candidate_metric[final_action], "metadata": metadata,
                })
    return rows


def test_c1_calibration_replays_t0_and_prefers_no_scalar_change_on_tie() -> None:
    result = calibrate_c1_rows(_rows(), require_scenes=2)
    assert result["status"] == "FROZEN"
    assert result["tasks"]["stereo"]["score_threshold"] == 0.699835896
    assert result["tasks"]["flow"]["action_logit_offset"] == 0.0
    assert result["tasks"]["flow"]["firewall_logit_offset"] == 0.0
    assert result["tasks"]["stereo"]["selected_metrics"]["feasible"] is True
    assert result["tasks"]["flow"]["selected_metrics"]["feasible"] is True


def test_c1_calibration_rejects_non_c1_rows() -> None:
    rows = _rows()
    rows[0]["role"] = "H1"
    with pytest.raises(ValueError, match="C1 rows only"):
        calibrate_c1_rows(rows, require_scenes=2)


def test_c1_calibration_requires_exact_t0_candidate_hash() -> None:
    rows = deepcopy(_rows())
    for row in rows:
        if (row["task"] == "flow" and row["system_id"] == "final_asymmetric_bridge_first"
                and row["condition"] != "clean"):
            row["prediction_sha256"] = "wrong"
            break
    with pytest.raises(ValueError, match="selected candidate hash differs"):
        calibrate_c1_rows(rows, require_scenes=2)
