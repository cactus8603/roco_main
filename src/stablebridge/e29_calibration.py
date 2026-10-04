"""Predeclared C1-only scalar calibration for the E29 T1 setting."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


SCHEMA = "bridge-e29-t1-calibration/v1"
STEREO_THRESHOLDS = (
    0.299835896,
    0.499835896,
    0.599835896,
    0.649835896,
    0.699835896,
    0.724835896,
    0.749835896,
    0.799835896,
    0.899835896,
    2.0,
)
FLOW_LOGIT_OFFSETS = (-20.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
FALLBACK_ACTION = {"stereo": "identity", "flow": "cc"}
CANDIDATES = {
    "stereo": ("identity", "denoise", "sharpen"),
    "flow": ("cc", "rr"),
}
NATIVE_MODE = {"stereo": "direct", "flow": "always_compute"}


def _key(row: dict) -> tuple[str, str, str]:
    return row["scene_id"], row["condition"], row["exposure"]


def _groups(rows: Iterable[dict]) -> dict[tuple[str, str, str], dict]:
    output: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        if row.get("schema") != "bridge-e29-stream-row/v1":
            raise ValueError("unexpected E29 row schema")
        if row.get("role") != "C1":
            raise ValueError("T1 calibration may consume C1 rows only")
        group = (row["task"], row["system_id"], row["execution_mode"])
        item = (group, _key(row))
        if item in output:
            raise ValueError(f"duplicate C1 row: {item}")
        output[item] = row
    return output


def _index(groups: dict, group: tuple[str, str, str]) -> dict[tuple[str, str, str], dict]:
    value = {key: row for (row_group, key), row in groups.items() if row_group == group}
    if not value:
        raise ValueError(f"missing C1 group: {group}")
    return value


def _validate_keys(keys: set[tuple[str, str, str]], *, require_scenes: int) -> None:
    scenes = sorted({key[0] for key in keys})
    if len(scenes) != require_scenes:
        raise ValueError(f"expected {require_scenes} C1 scenes, found {len(scenes)}")
    for scene in scenes:
        scene_keys = [key for key in keys if key[0] == scene]
        if sum(key[1] == "clean" for key in scene_keys) != 1:
            raise ValueError(f"scene {scene} must have exactly one clean row")
        if sum(key[1] != "clean" for key in scene_keys) != 12:
            raise ValueError(f"scene {scene} must have exactly 12 corrupt rows")


def _shift_probability(value: np.ndarray, offset: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if float(offset) == 0.0:
        return value.copy()
    epsilon = np.finfo(np.float32).eps
    clipped = np.clip(value, epsilon, 1.0 - epsilon)
    logit = np.log(clipped) - np.log1p(-clipped)
    return 1.0 / (1.0 + np.exp(-(logit + offset)))


def _stereo_action(row: dict, threshold: float) -> str:
    route = row.get("metadata", {}).get("route", {})
    selected = np.asarray(route.get("selected_treatment_indices"), dtype=np.int64)
    scores = np.asarray(route.get("selected_score"), dtype=np.float64)
    if selected.ndim != 1 or scores.shape != selected.shape or len(selected) == 0:
        raise ValueError("Stereo C1 route metadata is incomplete")
    if np.any((selected < 0) | (selected > 1)) or not np.isfinite(scores).all():
        raise ValueError("Stereo C1 route metadata is invalid")
    tile_actions = np.where(scores >= threshold, selected + 1, 0)
    counts = np.bincount(tile_actions, minlength=3)
    treatment = int(np.argmax(counts[1:]) + 1)
    action = treatment if counts[treatment] > len(tile_actions) // 2 else 0
    return CANDIDATES["stereo"][action]


def _flow_action(row: dict, action_offset: float, firewall_offset: float) -> str:
    policy = row.get("metadata", {}).get("policy", {})
    probability = np.asarray(policy.get("action_probability"), dtype=np.float64)
    firewall = np.asarray(policy.get("firewall_probability"), dtype=np.float64)
    action_threshold = np.asarray(policy.get("action_threshold"), dtype=np.float64)
    firewall_threshold = np.asarray(policy.get("firewall_threshold"), dtype=np.float64)
    if any(value.shape != (3, 6) for value in
           (probability, firewall, action_threshold, firewall_threshold)):
        raise ValueError("Flow C1 policy metadata must contain four 3x6 matrices")
    if not all(np.isfinite(value).all() for value in
               (probability, firewall, action_threshold, firewall_threshold)):
        raise ValueError("Flow C1 policy metadata is non-finite")
    fold = ((_shift_probability(probability, action_offset) > action_threshold)
            & (_shift_probability(firewall, firewall_offset) > firewall_threshold))
    seed = fold.sum(axis=1) > 3
    return "rr" if seed.sum() > 1 else "cc"


def _metrics(row: dict) -> tuple[float, float]:
    metric = row["metric"]
    return float(metric["mean_endpoint_error"]), float(metric["outlier_percent"])


def _evaluate(keys, actions, candidates, fallback, native_p95: float) -> dict[str, object]:
    scenes = sorted({key[0] for key in keys})
    task = next(iter(fallback.values()))["task"]
    scene_gain, clean_epe_cost, clean_outlier_cost = [], [], []
    harms, active_gain = [], []
    for scene in scenes:
        corrupt = [key for key in keys if key[0] == scene and key[1] != "clean"]
        clean = next(key for key in keys if key[0] == scene and key[1] == "clean")
        base = np.asarray([_metrics(fallback[key])[0] for key in corrupt])
        chosen = np.asarray([_metrics(candidates[actions[key]][key])[0] for key in corrupt])
        scene_gain.append(100.0 * float(base.mean() - chosen.mean()) / max(float(base.mean()), 1e-12))
        base_clean = _metrics(fallback[clean])
        chosen_clean = _metrics(candidates[actions[clean]][clean])
        clean_epe_cost.append((chosen_clean[0] - base_clean[0]) / max(base_clean[0], 1e-12))
        clean_outlier_cost.append(chosen_clean[1] - base_clean[1])
    corrupt_keys = [key for key in keys if key[1] != "clean"]
    for key in corrupt_keys:
        delta = _metrics(candidates[actions[key]][key])[0] - _metrics(fallback[key])[0]
        harms.append(max(delta, 0.0))
        if actions[key] != FALLBACK_ACTION[task]:
            active_gain.append(-delta)
    active = [key for key in corrupt_keys if actions[key] != FALLBACK_ACTION[task]]
    result = {
        "robust_gain_epe_percent_scene_macro": float(np.mean(scene_gain)),
        "clean_epe_relative_cost_scene_macro": float(np.mean(clean_epe_cost)),
        "clean_outlier_cost_percentage_points_scene_macro": float(np.mean(clean_outlier_cost)),
        "corrupt_coverage": len(active) / len(corrupt_keys),
        "beneficial_interventions": int(sum(value > 0 for value in active_gain)),
        "harmful_interventions": int(sum(value < 0 for value in active_gain)),
        "p95_harm_epe": float(np.quantile(harms, .95)),
        "max_harm_epe": float(max(harms)),
        "native_p95_harm_epe": float(native_p95),
    }
    result["feasible"] = bool(
        result["clean_outlier_cost_percentage_points_scene_macro"] <= .2
        and result["clean_epe_relative_cost_scene_macro"] <= .01
        and result["p95_harm_epe"] <= native_p95 + 1e-12
    )
    return result


def _native_p95(keys, native, fallback) -> float:
    harms = [
        max(_metrics(native[key])[0] - _metrics(fallback[key])[0], 0.0)
        for key in keys if key[1] != "clean"
    ]
    return float(np.quantile(harms, .95))


def _task_inputs(groups, task: str, require_scenes: int):
    fallback = _index(groups, (task, "task_fallback", "direct"))
    native = _index(groups, (task, "native_bridge", NATIVE_MODE[task]))
    final = _index(groups, (task, "final_asymmetric_bridge_first", "direct"))
    candidates = {
        action: _index(groups, (task, f"candidate:{action}", "direct"))
        for action in CANDIDATES[task]
    }
    keys = set(fallback)
    _validate_keys(keys, require_scenes=require_scenes)
    for name, value in {"native": native, "final": final, **candidates}.items():
        if set(value) != keys:
            raise ValueError(f"misaligned C1 rows for {task}/{name}")
    fallback_candidate = candidates[FALLBACK_ACTION[task]]
    for key in keys:
        if fallback[key]["prediction_sha256"] != fallback_candidate[key]["prediction_sha256"]:
            raise ValueError(f"fallback/candidate hash differs for {task}/{key}")
    return sorted(keys), fallback, native, final, candidates


def _verify_t0(keys, task, final, candidates, action_function) -> None:
    for key in keys:
        action = action_function(final[key])
        if action != final[key]["selected_action_id"]:
            raise ValueError(f"T0 metadata action replay differs for {task}/{key}")
        if candidates[action][key]["prediction_sha256"] != final[key]["prediction_sha256"]:
            raise ValueError(f"T0 selected candidate hash differs for {task}/{key}")


def calibrate_c1_rows(rows: Iterable[dict], *, require_scenes: int = 60) -> dict[str, object]:
    groups = _groups(rows)
    output: dict[str, object] = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "selection_rule": (
            "maximize C1 scene-macro corrupt EPE gain subject to clean outlier cost <=0.2pp, "
            "clean relative EPE cost <=1%, and p95 harm no worse than the native Bridge; "
            "ties prefer lower p95 harm then the smallest scalar change"
        ),
        "tasks": {},
    }

    keys, fallback, native, final, candidates = _task_inputs(groups, "stereo", require_scenes)
    base_thresholds = {float(row["metadata"]["route"]["threshold"]) for row in final.values()}
    if len(base_thresholds) != 1 or abs(next(iter(base_thresholds)) - 0.699835896) > 1e-9:
        raise ValueError("Stereo T0 threshold differs from the frozen S0 threshold")
    _verify_t0(keys, "stereo", final, candidates,
               lambda row: _stereo_action(row, 0.699835896))
    native_p95 = _native_p95(keys, native, fallback)
    trials = []
    for threshold in STEREO_THRESHOLDS:
        actions = {key: _stereo_action(final[key], threshold) for key in keys}
        metrics = _evaluate(keys, actions, candidates, fallback, native_p95)
        trials.append({"score_threshold": threshold, **metrics})
    feasible = [row for row in trials if row["feasible"]]
    winner = max(feasible, key=lambda row: (
        row["robust_gain_epe_percent_scene_macro"], -row["p95_harm_epe"],
        -abs(row["score_threshold"] - 0.699835896), -row["corrupt_coverage"],
    ))
    output["tasks"]["stereo"] = {
        "score_threshold": winner["score_threshold"],
        "selected_metrics": winner,
        "grid": list(STEREO_THRESHOLDS),
        "trials": trials,
    }

    keys, fallback, native, final, candidates = _task_inputs(groups, "flow", require_scenes)
    _verify_t0(keys, "flow", final, candidates,
               lambda row: _flow_action(row, 0.0, 0.0))
    native_p95 = _native_p95(keys, native, fallback)
    trials = []
    for action_offset in FLOW_LOGIT_OFFSETS:
        for firewall_offset in FLOW_LOGIT_OFFSETS:
            actions = {
                key: _flow_action(final[key], action_offset, firewall_offset)
                for key in keys
            }
            metrics = _evaluate(keys, actions, candidates, fallback, native_p95)
            trials.append({
                "action_logit_offset": action_offset,
                "firewall_logit_offset": firewall_offset,
                **metrics,
            })
    feasible = [row for row in trials if row["feasible"]]
    winner = max(feasible, key=lambda row: (
        row["robust_gain_epe_percent_scene_macro"], -row["p95_harm_epe"],
        -(abs(row["action_logit_offset"]) + abs(row["firewall_logit_offset"])),
        -row["corrupt_coverage"],
    ))
    output["tasks"]["flow"] = {
        "action_logit_offset": winner["action_logit_offset"],
        "firewall_logit_offset": winner["firewall_logit_offset"],
        "selected_metrics": winner,
        "grid": list(FLOW_LOGIT_OFFSETS),
        "trials": trials,
    }
    return output
