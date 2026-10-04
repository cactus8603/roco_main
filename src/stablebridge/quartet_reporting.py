"""Registered scene-macro S02 capacity gates; no deployment or significance claim."""
from __future__ import annotations

from collections import defaultdict
import math
import numbers


DEFAULT_GATE = {
    "minimum_queries": 1000,
    "minimum_computable_coverage_pct": 30.,
    "minimum_repair_rate_pct": 5.,
    "minimum_positive_scenes": 3,
    "required_scenes": 4,
    "wrong_max_fraction_of_primary": .5,
    "minimum_primary_minus_wrong_common_eligible_pp": 0.,
    "minimum_paired_delta_px": 0.,
    "cost_ratio_bounds": [.8, 1.25],
}


def _get(value, path):
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _finite(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values):
    values = list(values)
    return sum(float(value) for value in values) / len(values) if values and all(_finite(value) for value in values) else None


def _profile_name(value):
    return str(value["name"] if isinstance(value, dict) else value)


def _identity(row):
    case = row["case"]
    return str(case["scene"]), int(case["frame"]), _profile_name(case["profile"])


METRIC_PATHS = {
    "paired_delta_px": "primary.paired_delta_px",
    "path_gain_over_c0_px": "primary.path_gain_over_c0_px",
    "rematch_gain_over_c0_px": "primary.rematch_gain_over_c0_px",
    "wrong_gain_over_c0_px": "primary.wrong_gain_over_c0_px",
    "path_repair_rate_pct": "primary.newly_repairable_E1_oracle_pct",
    "rematch_repair_rate_pct": "primary.rematch_repair_rate_pct",
    "wrong_repair_rate_pct": "primary.wrong_repair_rate_pct",
    "primary_minus_wrong_repair_rate_pp": "primary.primary_minus_wrong_repair_rate_pp",
    "primary_minus_rematch_repair_rate_pp": "primary.primary_minus_rematch_repair_rate_pp",
    "path_computable_coverage_pct": "path.computable_coverage_pct",
    "path_eligible_coverage_pct": "path.eligible_coverage_pct",
    "wrong_computable_coverage_pct": "wrong.computable_coverage_pct",
    "wrong_eligible_coverage_pct": "wrong.eligible_coverage_pct",
    "rematch_valid_coverage_pct": "rematch_valid_coverage_pct",
    "common_eligible_coverage_pct": "common_eligible.coverage_pct_of_fixed_Q",
    "common_eligible_primary_repair_rate_pct": "common_eligible.primary_repair_rate_pct",
    "common_eligible_wrong_repair_rate_pct": "common_eligible.wrong_repair_rate_pct",
    "common_eligible_primary_minus_wrong_repair_rate_pp": "common_eligible.primary_minus_wrong_repair_rate_pp",
    "unresolved_E0_oracle_pct_of_gt_valid": "unresolved_E0_oracle_pct_of_gt_valid",
}
RAW_METRICS = ("baseline_error_px", "error_px", "signed_gain_px", "repair_rate_pct",
               "harm_over_delta_pct", "tail_harm_over_delta_pct")
for _family in ("path", "wrong"):
    for _representation in ("raw2d", "scored_candidate", "computable_raw2d_diagnostic",
                             "raw2d_eligible_only", "scored_candidate_eligible_only"):
        for _metric in RAW_METRICS:
            METRIC_PATHS[f"raw_{_family}_{_representation}_{_metric}"] = (
                f"{_family}.fixed_candidate_macro.{_representation}.{_metric}")
for _oracle in ("C0", "CP", "CR", "CW"):
    for _metric in RAW_METRICS:
        METRIC_PATHS[f"oracle_{_oracle}_{_metric}"] = f"oracles.{_oracle}.{_metric}"
for _state in ("newly_repairable_E1_oracle", "still_unresolved_partial_improvement",
               "still_unresolved_no_improvement", "still_unresolved_no_usable_path"):
    METRIC_PATHS[f"state_{_state}_pct"] = f"states.{_state}.pct_of_fixed_Q"

REQUIRED_CAPACITY_METRICS = (
    "paired_delta_px", "path_computable_coverage_pct", "path_repair_rate_pct",
    "wrong_repair_rate_pct", "common_eligible_primary_minus_wrong_repair_rate_pp",
)


def _status(scientific, complete, resource_complete, capacity_pass, resource_matched):
    if not scientific:
        return "FUNCTIONAL_ONLY"
    if not complete or not resource_complete:
        return "INCOMPLETE"
    if not resource_matched:
        return "INCONCLUSIVE_RESOURCE_MISMATCH"
    return "PASS" if capacity_pass else "FAIL"


def aggregate_capacity(results, config):
    """Aggregate registered profiles within scene, then scenes within each task.

    ``results`` rows contain ``case={scene,frame,profile}``, ``task``, an
    evaluate_capacity ``capacity`` summary, and a ``cost`` ledger.  Both tasks
    are always reported separately. Missing rows, duplicate identities and null
    profile metrics never disappear from an average. Additional case identities
    make registration incomplete rather than expanding the registered panel.

    Capacity and resource checks remain separate. A measured runtime mismatch
    yields INCONCLUSIVE_RESOURCE_MISMATCH even when numerical capacity checks
    pass. ``scientific_gate=False`` always labels the run FUNCTIONAL_ONLY and
    reports no scientific pass. It still reports functional completeness.
    """
    gate = {**DEFAULT_GATE, **config.get("gate", {})}
    unknown = set(gate) - set(DEFAULT_GATE)
    if unknown:
        raise ValueError(f"Unknown capacity gate keys: {sorted(unknown)}")
    bounds = gate["cost_ratio_bounds"]
    if len(bounds) != 2 or not all(_finite(x) for x in bounds) or not 0 < bounds[0] <= bounds[1]:
        raise ValueError("cost_ratio_bounds must be a positive ordered pair")
    for name, value in gate.items():
        if name != "cost_ratio_bounds" and (not _finite(value) or value < 0):
            raise ValueError(f"Invalid gate threshold {name}")
    if gate["minimum_positive_scenes"] > gate["required_scenes"]:
        raise ValueError("Positive scene count cannot exceed required scene count")
    scientific = config.get("scientific_gate", True)
    if not isinstance(scientific, bool):
        raise ValueError("scientific_gate must be boolean")
    scene_frames = [(str(scene), int(frame)) for scene, frame in config["scene_frames"]]
    profiles = [_profile_name(profile) for profile in config["profiles"]]
    if len(set(scene_frames)) != len(scene_frames) or len(set(profiles)) != len(profiles):
        raise ValueError("Registered scene/frame and profile identities must be unique")
    if not scene_frames or not profiles:
        raise ValueError("Registered scene/frame and profile lists must be nonempty")
    expected = {(scene, frame, profile) for scene, frame in scene_frames for profile in profiles}
    scenes = sorted({scene for scene, _ in scene_frames})
    grouped = {task: {} for task in ("stereo", "flow")}
    for row in results:
        task = row["task"]
        if task not in grouped:
            raise ValueError(f"Unknown task {task}")
        key = _identity(row)
        if key in grouped[task]:
            raise ValueError(f"Duplicate registered query row {(task,) + key}")
        count = _get(row, "capacity.counts.unresolved_E0_oracle")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("Each row must record a nonnegative integer fixed-Q count")
        grouped[task][key] = row
    cost_keys = sorted({key for row in results for key, value in row.get("cost", {}).items()
                        if value is None or (_finite(value) and not isinstance(value, bool))}
                      | {"path_to_rematch_seconds_ratio"})
    tasks = {}
    for task, rows in grouped.items():
        missing, extra = sorted(expected - set(rows)), sorted(set(rows) - expected)
        scene_results = {}
        for scene in scenes:
            profile_results = {}
            for profile in profiles:
                keys = sorted(key for key in expected if key[0] == scene and key[2] == profile)
                cases = [rows.get(key) for key in keys]
                active = [case for case in cases if case is not None]
                metrics = {name: _mean(_get(case, f"capacity.{path}") if case else None for case in cases)
                           for name, path in METRIC_PATHS.items()}
                costs = {key: _mean(_get(case, f"cost.{key}") if case else None for case in cases)
                         for key in cost_keys}
                bad_statuses = [case["capacity"].get("status", "missing") for case in active
                                if case["capacity"].get("status") != "ok"]
                profile_results[profile] = {
                    "expected_cases": len(keys), "recorded_cases": len(active),
                    "missing_case_ids": [list(key) for key in keys if key not in rows],
                    "non_ok_capacity_statuses": bad_statuses,
                    "total_unresolved_E0_oracle": sum(case["capacity"]["counts"]["unresolved_E0_oracle"] for case in active),
                    "capacity_complete": len(active) == len(keys) and not bad_statuses
                    and all(metrics[name] is not None for name in REQUIRED_CAPACITY_METRICS),
                    "resource_complete": costs["path_to_rematch_seconds_ratio"] is not None,
                    "metrics": metrics, "cost": costs,
                }
            scene_metrics = {name: _mean(value["metrics"][name] for value in profile_results.values())
                             for name in METRIC_PATHS}
            scene_cost = {name: _mean(value["cost"][name] for value in profile_results.values()) for name in cost_keys}
            scene_results[scene] = {
                "profiles": profile_results,
                "total_unresolved_E0_oracle": sum(value["total_unresolved_E0_oracle"] for value in profile_results.values()),
                "capacity_complete": all(value["capacity_complete"] for value in profile_results.values()),
                "resource_complete": all(value["resource_complete"] for value in profile_results.values()),
                "metrics": scene_metrics, "cost": scene_cost,
            }
        macro = {name: _mean(value["metrics"][name] for value in scene_results.values()) for name in METRIC_PATHS}
        cost_macro = {name: _mean(value["cost"][name] for value in scene_results.values()) for name in cost_keys}
        registered_scene_count = len(scenes) == gate["required_scenes"]
        panel_complete = not missing and not extra
        capacity_complete = panel_complete and all(value["capacity_complete"] for value in scene_results.values())
        resource_complete = panel_complete and all(value["resource_complete"] for value in scene_results.values())
        total_q = sum(value["total_unresolved_E0_oracle"] for value in scene_results.values())
        positive_scenes = sum(value["metrics"]["paired_delta_px"] is not None
                              and value["metrics"]["paired_delta_px"] > gate["minimum_paired_delta_px"]
                              for value in scene_results.values())
        p, w = macro["path_repair_rate_pct"], macro["wrong_repair_rate_pct"]
        checks = {
            "registered_panel_complete": panel_complete,
            "capacity_metrics_complete": capacity_complete,
            "required_scenes": registered_scene_count,
            "minimum_queries": total_q >= gate["minimum_queries"],
            "minimum_computable_coverage_pct": macro["path_computable_coverage_pct"] is not None
                and macro["path_computable_coverage_pct"] >= gate["minimum_computable_coverage_pct"],
            "minimum_repair_rate_pct": p is not None and p >= gate["minimum_repair_rate_pct"],
            "paired_delta_above_minimum": macro["paired_delta_px"] is not None
                and macro["paired_delta_px"] > gate["minimum_paired_delta_px"],
            "minimum_positive_scenes": positive_scenes >= gate["minimum_positive_scenes"],
            "wrong_below_fraction_of_primary": p is not None and w is not None
                and w < gate["wrong_max_fraction_of_primary"] * p,
            "common_eligible_gap_above_minimum": macro["common_eligible_primary_minus_wrong_repair_rate_pp"] is not None
                and macro["common_eligible_primary_minus_wrong_repair_rate_pp"] > gate["minimum_primary_minus_wrong_common_eligible_pp"],
        }
        capacity_pass = all(checks.values()) if capacity_complete else None
        ratio = cost_macro["path_to_rematch_seconds_ratio"]
        matched = bounds[0] <= ratio <= bounds[1] if ratio is not None and resource_complete else None
        tasks[task] = {
            "status": _status(scientific, capacity_complete, resource_complete, capacity_pass, matched),
            "capacity_pass": capacity_pass if scientific else None,
            "capacity_checks_would_pass": capacity_pass,
            "resource_matched": matched,
            "functional_complete": panel_complete and all(
                case["capacity"].get("status") == "ok" for key, case in rows.items() if key in expected),
            "total_unresolved_E0_oracle": total_q, "positive_scenes": positive_scenes,
            "wrong_to_primary_repair_ratio": w / p if p is not None and w is not None and p > 0 else None,
            "checks": checks,
            "completeness": {"capacity_complete": capacity_complete, "resource_complete": resource_complete,
                             "expected_rows": len(expected), "recorded_registered_rows": len(expected & set(rows)),
                             "missing_case_ids": [list(key) for key in missing], "extra_case_ids": [list(key) for key in extra]},
            "scene_macro": macro, "cost_scene_macro": cost_macro, "scenes": scene_results,
        }
    complete = all(value["completeness"]["capacity_complete"] for value in tasks.values())
    resource_complete = all(value["completeness"]["resource_complete"] for value in tasks.values())
    capacity_pass = all(value["capacity_pass"] for value in tasks.values()) if complete and scientific else None
    resource_matched = all(value["resource_matched"] for value in tasks.values()) if resource_complete else None
    return {
        "report_version": "s02_registered_scene_macro_gate_v1", "scientific_gate": scientific,
        "status": _status(scientific, complete, resource_complete, capacity_pass, resource_matched),
        "capacity_pass": capacity_pass, "resource_matched": resource_matched,
        "functional_complete": all(value["functional_complete"] for value in tasks.values()),
        "gate": gate, "tasks": tasks,
        "aggregation": {
            "within_profile": "equal_registered_frames",
            "within_scene": "equal_registered_profiles",
            "between_scenes": "equal_registered_scenes",
            "raw_path_candidate_macro": "equal_registered_candidates_before_profile_scene_averaging",
            "null_policy": "null_at_any_registered_frame_propagates_to_profile_scene_task; never_silently_drop",
            "query_counts": "sum_fixed_Q_counts_over_registered_cases; correlated_queries_not_independent_samples",
            "resource_ratio": "mean_case_ratios_with_same_profile_then_scene_weighting; not_ratio_of_pooled_times",
            "physical_cost": "joint_cache_times_are_reported_per_task; do_not_sum_task_macros_as_physical_runtime",
        },
        "interpretation": (
            "Development mechanism screening only; no statistical significance, safe-selection, or calibrated trust claim. "
            "A failed capacity gate concerns these frozen predictors, composer, candidate banks and budget; "
            "it does not prove quartet observations contain no useful information. "
            "Passing permits study of observable selection and calibration, not deployment."
        ),
    }
