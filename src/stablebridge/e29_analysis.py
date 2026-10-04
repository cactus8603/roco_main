"""Frozen aggregation and H1 gate logic for the E29 joint panel."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Iterable

import numpy as np


REQUIRED_BY_TASK = {
    "stereo": (
        ("task_fallback", "direct"),
        ("native_bridge", "direct"),
        ("final_asymmetric_bridge_first", "direct"),
        ("final_asymmetric_bridge_first", "always_compute"),
    ),
    "flow": (
        ("task_fallback", "direct"),
        # The original PairBridge selector needs both CC and RR predictions to
        # construct its consistency evidence.  Calling it a direct one-expert
        # method would silently change the native baseline.
        ("native_bridge", "always_compute"),
        ("final_asymmetric_bridge_first", "direct"),
        ("final_asymmetric_bridge_first", "always_compute"),
    ),
}

PRIMARY_MODE = {
    "task_fallback": "direct",
    "native_bridge": {"stereo": "direct", "flow": "always_compute"},
    "final_asymmetric_bridge_first": "direct",
}


def _q(values, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _mean(values) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _row_id(row: dict) -> tuple[str, str, str]:
    return row["scene_id"], row["condition"], row["exposure"]


def _pixel_pooled(rows: Iterable[dict]) -> dict[str, float | int]:
    rows = list(rows)
    valid = sum(int(row["metric"]["valid_pixels"]) for row in rows)
    if valid <= 0:
        raise ValueError("pixel-pooled metric has no valid pixels")
    error_sum = sum(float(row["metric"]["endpoint_error_sum"]) for row in rows)
    outliers = sum(int(row["metric"]["outlier_pixels"]) for row in rows)
    if not np.isfinite(error_sum) or error_sum < 0 or not 0 <= outliers <= valid:
        raise ValueError("invalid exact metric counts in E29 row")
    return {
        "valid_pixels": valid,
        "endpoint_error_sum": error_sum,
        "outlier_pixels": outliers,
        "mean_endpoint_error": error_sum / valid,
        "outlier_percent": 100.0 * outliers / valid,
    }


def _index(rows: Iterable[dict], *, role: str):
    result: dict[tuple[str, str, str], dict[tuple[str, str, str], dict]] = defaultdict(dict)
    for row in rows:
        if row.get("role") != role:
            continue
        if row.get("schema") != "bridge-e29-stream-row/v1":
            raise ValueError("unexpected E29 row schema")
        group = (row["task"], row["system_id"], row["execution_mode"])
        key = _row_id(row)
        if key in result[group]:
            raise ValueError(f"duplicate E29 result row: {group} {key}")
        result[group][key] = row
    return result


def _validate_panel(index, *, require_scenes: int) -> dict[str, set[tuple[str, str, str]]]:
    expected: dict[str, set[tuple[str, str, str]]] = {}
    for task in ("stereo", "flow"):
        for system, mode in REQUIRED_BY_TASK[task]:
            group = (task, system, mode)
            if group not in index:
                raise ValueError(f"missing required E29 group: {group}")
            keys = set(index[group])
            if task not in expected:
                expected[task] = keys
            elif keys != expected[task]:
                raise ValueError(f"incomplete or misaligned E29 rows for {group}")
        scenes = sorted({key[0] for key in expected[task]})
        if len(scenes) != require_scenes:
            raise ValueError(f"expected {require_scenes} {task} scenes, found {len(scenes)}")
        for scene in scenes:
            scene_keys = [key for key in expected[task] if key[0] == scene]
            clean = [key for key in scene_keys if key[1] == "clean"]
            corrupt = [key for key in scene_keys if key[1] != "clean"]
            if len(clean) != 1 or len(corrupt) != 12:
                raise ValueError(f"scene {scene} must have one clean and 12 corrupt {task} rows")
    if {key[0] for key in expected["stereo"]} != {key[0] for key in expected["flow"]}:
        raise ValueError("Stereo and Flow must use the same H1 scene identities")
    return expected


def _panel(method: dict, fallback: dict) -> dict:
    keys = sorted(method)
    scenes = sorted({key[0] for key in keys})
    corrupt_scene_gain, corrupt_scene_outlier_gain = [], []
    clean_epe_cost, clean_outlier_cost = [], []
    harms, intervention_gain = [], []
    for scene in scenes:
        corrupt = [key for key in keys if key[0] == scene and key[1] != "clean"]
        clean = [key for key in keys if key[0] == scene and key[1] == "clean"]
        base = np.asarray([fallback[key]["metric"]["mean_endpoint_error"] for key in corrupt])
        value = np.asarray([method[key]["metric"]["mean_endpoint_error"] for key in corrupt])
        base_out = np.asarray([fallback[key]["metric"]["outlier_percent"] for key in corrupt])
        value_out = np.asarray([method[key]["metric"]["outlier_percent"] for key in corrupt])
        corrupt_scene_gain.append(100.0 * float(base.mean() - value.mean()) / max(float(base.mean()), 1e-12))
        corrupt_scene_outlier_gain.append(float(base_out.mean() - value_out.mean()))
        key = clean[0]
        base_clean = float(fallback[key]["metric"]["mean_endpoint_error"])
        value_clean = float(method[key]["metric"]["mean_endpoint_error"])
        clean_epe_cost.append((value_clean - base_clean) / max(base_clean, 1e-12))
        clean_outlier_cost.append(float(method[key]["metric"]["outlier_percent"])
                                  - float(fallback[key]["metric"]["outlier_percent"]))
    corrupt_keys = [key for key in keys if key[1] != "clean"]
    for key in corrupt_keys:
        delta = (float(method[key]["metric"]["mean_endpoint_error"])
                 - float(fallback[key]["metric"]["mean_endpoint_error"]))
        harms.append(max(delta, 0.0))
        if method[key]["intervened"]:
            intervention_gain.append(-delta)
    active = [key for key in corrupt_keys if method[key]["intervened"]]
    clean_keys = [key for key in keys if key[1] == "clean"]
    latency = np.asarray([float(method[key]["runtime"]["total_seconds"]) for key in keys])
    observer = np.asarray([float(method[key]["runtime"]["observer_seconds"]) for key in keys])
    allocated = [method[key]["runtime"].get("peak_allocated_bytes") for key in keys]
    reserved = [method[key]["runtime"].get("peak_reserved_bytes") for key in keys]
    return {
        "scenes": len(scenes),
        "rows": len(keys),
        "robust_gain_epe_percent_scene_macro": _mean(corrupt_scene_gain),
        "robust_gain_outlier_percentage_points_scene_macro": _mean(corrupt_scene_outlier_gain),
        "clean_epe_relative_cost_scene_macro": _mean(clean_epe_cost),
        "clean_outlier_cost_percentage_points_scene_macro": _mean(clean_outlier_cost),
        "corrupt_coverage": len(active) / len(corrupt_keys),
        "beneficial_interventions": int(sum(value > 0 for value in intervention_gain)),
        "harmful_interventions": int(sum(value < 0 for value in intervention_gain)),
        "p95_harm_epe": _q(harms, .95),
        "max_harm_epe": float(max(harms)),
        "scene_robust_gain_epe_percent": dict(zip(scenes, map(float, corrupt_scene_gain))),
        "pixel_pooled": {
            "clean": _pixel_pooled(method[key] for key in clean_keys),
            "corrupt_panel": _pixel_pooled(method[key] for key in corrupt_keys),
        },
        "runtime": {
            "mean_seconds": float(latency.mean()),
            "median_seconds": float(np.median(latency)),
            "p95_seconds": _q(latency, .95),
            "observer_median_seconds": float(np.median(observer)),
            "observer_fraction_of_median": float(np.median(observer) / max(np.median(latency), 1e-12)),
            "expert_forwards": int(sum(int(method[key]["runtime"]["expert_forwards"]) for key in keys)),
            "peak_allocated_bytes": max((int(value) for value in allocated if value is not None), default=None),
            "peak_reserved_bytes": max((int(value) for value in reserved if value is not None), default=None),
        },
    }


def _candidate_oracle(index, task: str, expected_keys):
    candidates = [group for group in index if group[0] == task and group[1].startswith("candidate:")
                  and group[2] == "direct"]
    if len(candidates) < 2:
        return None
    if any(set(index[group]) != expected_keys for group in candidates):
        raise ValueError(f"candidate oracle rows are incomplete for {task}")
    output = {}
    for key in expected_keys:
        winner = min((index[group][key] for group in candidates),
                     key=lambda row: float(row["metric"]["mean_endpoint_error"]))
        row = deepcopy(winner)
        row["system_id"] = "candidate_oracle"
        row["intervened"] = row["selected_action_id"] != row["fallback_action_id"]
        output[key] = row
    return output


def analyze_h1_rows(rows: Iterable[dict], *, require_scenes: int = 70,
                    bootstrap_samples: int = 10000,
                    bootstrap_seed: int = 290927) -> dict:
    index = _index(rows, role="H1")
    expected = _validate_panel(index, require_scenes=require_scenes)
    panels, final_scene_gain = {}, {}
    for task in ("stereo", "flow"):
        fallback = index[(task, "task_fallback", "direct")]
        panels[task] = {}
        systems = sorted({group[1] for group in index if group[0] == task})
        for system in systems:
            mode_value = PRIMARY_MODE.get(system, "direct")
            mode = mode_value[task] if isinstance(mode_value, dict) else mode_value
            group = (task, system, mode)
            if group not in index or set(index[group]) != expected[task]:
                continue
            panels[task][system] = _panel(index[group], fallback)
        oracle = _candidate_oracle(index, task, expected[task])
        if oracle is not None:
            panels[task]["candidate_oracle"] = _panel(oracle, fallback)
        final_scene_gain[task] = panels[task]["final_asymmetric_bridge_first"]["scene_robust_gain_epe_percent"]

    scene_ids = sorted(final_scene_gain["stereo"])
    rng = np.random.default_rng(bootstrap_seed)
    task_samples = {task: [] for task in ("stereo", "flow")}
    macro_samples = []
    for _ in range(bootstrap_samples):
        sample = rng.choice(scene_ids, len(scene_ids), replace=True)
        values = {}
        for task in ("stereo", "flow"):
            values[task] = _mean([final_scene_gain[task][scene] for scene in sample])
            task_samples[task].append(values[task])
        macro_samples.append((values["stereo"] + values["flow"]) / 2.0)

    exactness, reductions, runtime_efficiency = {}, {}, {}
    for task in ("stereo", "flow"):
        direct = index[(task, "final_asymmetric_bridge_first", "direct")]
        always = index[(task, "final_asymmetric_bridge_first", "always_compute")]
        exactness[task] = all(direct[key]["prediction_sha256"] == always[key]["prediction_sha256"]
                              for key in expected[task])
        direct_forwards = sum(int(direct[key]["runtime"]["expert_forwards"]) for key in expected[task])
        always_forwards = sum(int(always[key]["runtime"]["expert_forwards"]) for key in expected[task])
        reductions[task] = 1.0 - direct_forwards / always_forwards
        direct_seconds = np.asarray([
            float(direct[key]["runtime"]["total_seconds"]) for key in sorted(expected[task])
        ], dtype=np.float64)
        always_seconds = np.asarray([
            float(always[key]["runtime"]["total_seconds"]) for key in sorted(expected[task])
        ], dtype=np.float64)
        saved_seconds = always_seconds - direct_seconds
        runtime_efficiency[task] = {
            "direct_mean_seconds": float(direct_seconds.mean()),
            "always_compute_mean_seconds": float(always_seconds.mean()),
            "direct_median_seconds": float(np.median(direct_seconds)),
            "always_compute_median_seconds": float(np.median(always_seconds)),
            "direct_p95_seconds": _q(direct_seconds, .95),
            "always_compute_p95_seconds": _q(always_seconds, .95),
            "paired_mean_saved_seconds": float(saved_seconds.mean()),
            "paired_median_saved_seconds": float(np.median(saved_seconds)),
            "mean_latency_reduction": float(1.0 - direct_seconds.mean() / always_seconds.mean()),
            "median_latency_reduction": float(
                1.0 - np.median(direct_seconds) / np.median(always_seconds)
            ),
            "direct_observer_median_seconds": float(np.median([
                float(direct[key]["runtime"]["observer_seconds"]) for key in expected[task]
            ])),
        }

    final = {task: panels[task]["final_asymmetric_bridge_first"] for task in ("stereo", "flow")}
    native = {task: panels[task]["native_bridge"] for task in ("stereo", "flow")}
    checks = {
        "each_task_mean_robust_gain_positive": all(
            final[task]["robust_gain_epe_percent_scene_macro"] > 0 for task in final),
        "equal_task_macro_bootstrap_ci95_lower_positive": _q(macro_samples, .025) > 0,
        "each_task_clean_outlier_cost_lte_0p2pp": all(
            final[task]["clean_outlier_cost_percentage_points_scene_macro"] <= .2 for task in final),
        "each_task_clean_epe_relative_cost_lte_1pct": all(
            final[task]["clean_epe_relative_cost_scene_macro"] <= .01 for task in final),
        "each_task_p95_harm_lte_native": all(
            final[task]["p95_harm_epe"] <= native[task]["p95_harm_epe"] + 1e-12 for task in final),
        "direct_output_hash_exact_every_row": all(exactness.values()),
        "each_task_expert_forward_reduction_gte_25pct": all(value >= .25 for value in reductions.values()),
        "each_task_direct_mean_and_median_latency_lt_always_compute": all(
            value["paired_mean_saved_seconds"] > 0
            and value["paired_median_saved_seconds"] > 0
            for value in runtime_efficiency.values()
        ),
    }
    return {
        "schema": "bridge-e29-h1-analysis/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scene_count": require_scenes,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "task_ci95": {task: [_q(values, .025), _q(values, .975)]
                          for task, values in task_samples.items()},
            "equal_task_macro_ci95": [_q(macro_samples, .025), _q(macro_samples, .975)],
        },
        "panels": panels,
        "direct_exactness": exactness,
        "expert_forward_reduction": reductions,
        "runtime_efficiency": runtime_efficiency,
        "gate_checks": checks,
        "advance_to_h2": bool(all(checks.values())),
    }
