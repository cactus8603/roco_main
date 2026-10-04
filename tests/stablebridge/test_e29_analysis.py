from __future__ import annotations

from copy import deepcopy

from stablebridge.e29_analysis import analyze_h1_rows


def _rows() -> list[dict]:
    rows = []
    for task in ("stereo", "flow"):
        systems = (
            ("task_fallback", "direct"),
            ("native_bridge", "direct" if task == "stereo" else "always_compute"),
            ("final_asymmetric_bridge_first", "direct"),
            ("final_asymmetric_bridge_first", "always_compute"),
        )
        fallback_action = "identity" if task == "stereo" else "cc"
        active_action = "denoise" if task == "stereo" else "rr"
        for scene in ("000000", "000001"):
            conditions = [("clean", "clean")]
            conditions.extend((f"condition_{index // 3}", ("first", "second", "both")[index % 3])
                              for index in range(12))
            for condition, exposure in conditions:
                clean = condition == "clean"
                for system, mode in systems:
                    if system == "task_fallback":
                        epe, outlier, action = (1.0, 2.0, fallback_action) if clean else (10.0, 20.0, fallback_action)
                        total, observer, expert_forwards = 10.0, 0.0, 1
                    elif system == "native_bridge":
                        epe, outlier, action = (1.0, 2.0, fallback_action) if clean else (9.0, 18.0, active_action)
                        total, observer = 10.0, .5
                        expert_forwards = 1 if task == "stereo" else 2
                    else:
                        epe, outlier, action = (1.005, 2.1, fallback_action) if clean else (8.0, 16.0, active_action)
                        if mode == "direct":
                            total, observer, expert_forwards = 10.0, .5, 1
                        else:
                            total, observer, expert_forwards = 18.0, .5, 2
                    row_key = f"kitti2015:{task}:{scene}:{condition}:{exposure}"
                    selected_hash = f"selected:{task}:{scene}:{condition}:{exposure}"
                    rows.append({
                        "schema": "bridge-e29-stream-row/v1",
                        "system_id": system,
                        "execution_mode": mode,
                        "task": task,
                        "role": "H1",
                        "scene_id": scene,
                        "condition": condition,
                        "exposure": exposure,
                        "is_clean": clean,
                        "row_key": row_key,
                        "prediction_sha256": selected_hash if system == "final_asymmetric_bridge_first" else f"{system}:{row_key}",
                        "selected_action_id": action,
                        "fallback_action_id": fallback_action,
                        "intervened": action != fallback_action,
                        "metric": {"valid_pixels": 100, "endpoint_error_sum": epe * 100,
                                   "outlier_pixels": int(outlier),
                                   "mean_endpoint_error": epe, "outlier_percent": outlier},
                        "runtime": {"total_seconds": total, "observer_seconds": observer,
                                    "expert_seconds": total - observer,
                                    "expert_forwards": expert_forwards,
                                    "peak_allocated_bytes": 100,
                                    "peak_reserved_bytes": 200},
                    })
    return rows


def test_h1_gate_passes_only_with_joint_quality_cost_and_exactness() -> None:
    rows = _rows()
    result = analyze_h1_rows(rows, require_scenes=2, bootstrap_samples=200)
    assert result["status"] == "PASS"
    assert result["advance_to_h2"] is True
    assert all(result["gate_checks"].values())
    assert result["expert_forward_reduction"] == {"stereo": .5, "flow": .5}
    for task in ("stereo", "flow"):
        final = result["panels"][task]["final_asymmetric_bridge_first"]
        assert final["robust_gain_epe_percent_scene_macro"] == 20.0
        assert final["clean_epe_relative_cost_scene_macro"] < .01
        assert final["pixel_pooled"]["clean"]["valid_pixels"] == 200
        assert final["pixel_pooled"]["corrupt_panel"]["valid_pixels"] == 2400
        assert final["pixel_pooled"]["corrupt_panel"]["outlier_percent"] == 16.0
        assert final["runtime"]["observer_fraction_of_median"] == .05
        assert final["runtime"]["observer_median_seconds"] == .5
        assert result["runtime_efficiency"][task]["paired_mean_saved_seconds"] == 8.0
        assert result["runtime_efficiency"][task]["median_latency_reduction"] == 4 / 9

    broken = deepcopy(rows)
    for row in broken:
        if (row["task"] == "stereo"
                and row["system_id"] == "final_asymmetric_bridge_first"
                and row["execution_mode"] == "always_compute"):
            row["prediction_sha256"] = "different"
            break
    failed = analyze_h1_rows(broken, require_scenes=2, bootstrap_samples=200)
    assert failed["status"] == "FAIL"
    assert failed["advance_to_h2"] is False
    assert failed["gate_checks"]["direct_output_hash_exact_every_row"] is False

    slower = deepcopy(rows)
    for row in slower:
        if (row["system_id"] == "final_asymmetric_bridge_first"
                and row["execution_mode"] == "direct"):
            row["runtime"]["total_seconds"] = 20.0
    failed = analyze_h1_rows(slower, require_scenes=2, bootstrap_samples=200)
    assert failed["status"] == "FAIL"
    assert failed["gate_checks"][
        "each_task_direct_mean_and_median_latency_lt_always_compute"
    ] is False
