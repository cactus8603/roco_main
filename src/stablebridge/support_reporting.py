"""Fixed-cohort actual-output evaluation and registered S04 confirmation gates.

Ground truth is used here only for offline evaluation. Candidate-oracle errors
are reported separately and never substitute for an arm's emitted prediction.
"""
from __future__ import annotations

import numpy as np

from .evaluation import error_map, gt_valid_mask, region_metrics


CONTRASTS = {
    "processed_vs_no_support": ("processed", "no_support"),
    "processed_vs_raw": ("processed", "raw"),
    "processed_vs_wrong": ("processed", "wrong"),
    "processed_vs_propagation_processed": ("processed", "propagation_processed"),
    "gt_diagnostic_vs_no_support": ("gt_diagnostic", "no_support"),
}
REGIONS = ("all", "initial_good", "hard")
METRICS = (
    "baseline_error_px", "error_px", "gain_px", "repair_rate_pct", "repair_delta_pp",
    "partial_improvement_pct", "initial_good_damage_pct", "harm_over_3px_pct",
    "tail_harm_over_5px_pct",
)
COUNT_METRICS = ("pixels", "nonfinite_output_pixels", "nonfinite_baseline_pixels")


def _profile(value):
    return str(value["name"] if isinstance(value, dict) else value)


def _scene(value):
    value = str(value)
    return value.zfill(4) if value.isdigit() else value


def _strict_mean(values):
    values = list(values)
    return (float(np.mean(values, dtype=np.float64)) if values
            and all(value is not None and np.isfinite(value) for value in values) else None)


def _relative_gain(metrics):
    gain, initial = metrics.get("gain_px"), metrics.get("baseline_error_px")
    return gain / initial if gain is not None and initial is not None and initial > 0 else None


def _region(baseline_error, current_error, mask, threshold, severe_harm, tail_harm):
    existing = region_metrics(baseline_error, current_error, mask,
                              threshold=threshold, severe_harm_px=severe_harm)
    base, current = baseline_error[mask], current_error[mask]
    count = int(mask.sum())
    with np.errstate(invalid="ignore", over="ignore"):
        gain = base - current
    def pct(values):
        return 100.0 * float(np.mean(values)) if count else None
    undefined_delta = np.isnan(gain).any()
    result = {
        "pixels": count, "status": existing["status"],
        "baseline_error_px": existing["baseline_error_px"], "error_px": existing["error_px"],
        "gain_px": existing["gain_px"],
        "repair_rate_pct": pct(current <= threshold),
        "repair_delta_pp": pct((current <= threshold).astype(float) - (base <= threshold)),
        "partial_improvement_pct": pct((current > threshold) & (current < base)
                                        & np.isfinite(current) & np.isfinite(base)),
        "initial_good_damage_pct": pct((base <= threshold) & (current > threshold)),
        "harm_over_3px_pct": None if undefined_delta else pct(gain < -severe_harm),
        "tail_harm_over_5px_pct": None if undefined_delta else pct(gain < -tail_harm),
        "nonfinite_output_pixels": existing["nonfinite_prediction_pixels"],
        "nonfinite_baseline_pixels": existing["nonfinite_baseline_pixels"],
    }
    result["relative_gain_fraction"] = _relative_gain(result)
    return result


def evaluate_support_case(initial, outputs, gt4, fixed_Q, oracle_errors=None, metadata=None,
                          *, extra_regions=None, threshold=1.0, severe_harm=3.0, tail_harm=5.0):
    """Return a JSON-safe case report using common masks for every actual arm.

    ``initial`` and every output are [2,H,W]; GT is [4,2,H,W]. ``fixed_Q`` is
    the already frozen strong-E0 oracle cohort and must be GT-valid and initially
    bad. Additional diagnostic masks are intersected with GT validity once,
    identically for all arms. ``metadata`` should contain task, scene, frame,
    profile and split for exact registered-panel aggregation.

    Contrasts use gain = reference error minus proposed-arm error. Partial
    improvement still has error > threshold. All percentages use their stated
    region's complete denominator, including failed output pixels. Therefore
    initial-good damage is conditional on initial-good only in that region.
    """
    if not (np.isfinite([threshold, severe_harm, tail_harm]).all()
            and threshold > 0 and 0 <= severe_harm <= tail_harm):
        raise ValueError("Require threshold > 0 and 0 <= severe_harm <= tail_harm")
    initial_error = error_map(initial, gt4)
    valid = gt_valid_mask(gt4)
    q = np.asarray(fixed_Q)
    if q.shape != valid.shape or q.dtype != np.bool_:
        raise ValueError("fixed_Q must be a boolean HxW mask")
    if (q & ~valid).any() or (q & (initial_error <= threshold)).any():
        raise ValueError("fixed_Q must be GT-valid and initially above the error threshold")
    regions = {"all": valid, "initial_good": valid & (initial_error <= threshold), "hard": q.copy()}
    for name, mask in (extra_regions or {}).items():
        if name in regions:
            raise ValueError("Extra diagnostic regions cannot replace registered primary regions")
        mask = np.asarray(mask)
        if mask.shape != valid.shape or mask.dtype != np.bool_:
            raise ValueError("Extra regions must be boolean HxW masks")
        regions[str(name)] = valid & mask
    output_fields = dict(outputs)
    if "identity" in output_fields and not np.array_equal(output_fields["identity"], initial, equal_nan=True):
        raise ValueError("identity output must equal the original prediction")
    output_fields["identity"] = np.asarray(initial)
    errors = {name: error_map(value, gt4) for name, value in output_fields.items()}
    arms = {name: {region: _region(initial_error, error, mask, threshold, severe_harm, tail_harm)
                   for region, mask in regions.items()} for name, error in errors.items()}
    contrasts = {
        name: {region: _region(errors[reference], errors[proposed], mask,
                               threshold, severe_harm, tail_harm)
               for region, mask in regions.items()}
        for name, (proposed, reference) in CONTRASTS.items()
        if proposed in errors and reference in errors
    }
    oracles = {}
    for name, values in (oracle_errors or {}).items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != valid.shape or (values[np.isfinite(values)] < 0).any():
            raise ValueError("Oracle error must be nonnegative HxW native-pixel error")
        values = np.where(np.isfinite(values), values, np.inf)
        oracles[name] = {region: _region(initial_error, values, mask,
                                        threshold, severe_harm, tail_harm)
                         for region, mask in regions.items()}
    failures = {name: int((valid & ~np.isfinite(np.asarray(value)).all(axis=0)).sum())
                for name, value in output_fields.items()}
    return {
        "version": "s04_actual_output_fixed_query_v1", "metadata": dict(metadata or {}),
        "status": "failed_prediction" if any(failures.values()) else ("ok" if valid.any() else "empty_GT"),
        "threshold_px": float(threshold), "severe_harm_px": float(severe_harm),
        "tail_harm_px": float(tail_harm),
        "counts": {"dense": int(valid.size), **{name: int(mask.sum()) for name, mask in regions.items()}},
        "arms": arms, "contrasts": contrasts, "candidate_oracles": oracles,
        "nonfinite_outputs_on_valid_GT": failures,
        "cohort": "one_fixed_strong_E0_Q_for_all_arms_no_success_or_availability_filtering",
        "interpretation": "actual_outputs_primary; candidate_oracles_diagnostic_only; no_calibrated_trust_claim",
    }


def _combine(regions, pooled=False, exclude_empty=True):
    """Exclude disclosed empty cases, never missing/failed nonempty cases.

    Scene aggregation sets exclude_empty=False: an entirely empty scene must
    invalidate that macro rather than disappear from the registered panel.
    """
    regions = list(regions)
    present = [region for region in regions if region is not None]
    missing = len(regions) - len(present)
    active = [region for region in present if region["pixels"] > 0]
    result = {key: sum(region.get(key, 0) for region in present) for key in COUNT_METRICS}
    result.update(cases=len(regions), missing_cases=missing,
                  empty_cases=sum(region["pixels"] == 0 for region in present),
                  nonempty_cases=len(active))
    for metric in METRICS:
        if pooled:
            good = not missing and active and all(region.get(metric) is not None
                    and np.isfinite(region[metric]) for region in active)
            result[metric] = (sum(region[metric] * region["pixels"] for region in active)
                              / sum(region["pixels"] for region in active)) if good else None
        else:
            contributors = active if exclude_empty else present
            result[metric] = (_strict_mean(region.get(metric) for region in contributors)
                              if not missing else None)
    result["relative_gain_fraction"] = _relative_gain(result)
    return result


def _nested_summary(rows, arm_names, contrast_names, region_names, pooled=False, oracle_names=()):
    return {kind: {name: {region: _combine(
        [row.get(kind, {}).get(name, {}).get(region) if row is not None else None for row in rows],
        pooled=pooled) for region in region_names} for name in names}
        for kind, names in (("arms", arm_names), ("contrasts", contrast_names),
                            ("candidate_oracles", oracle_names))}


def _macro_from_scenes(scenes, arm_names, contrast_names, region_names, oracle_names=()):
    return {kind: {name: {region: _combine(
        [scene[kind][name][region] for scene in scenes.values()], exclude_empty=False)
        for region in region_names} for name in names}
        for kind, names in (("arms", arm_names), ("contrasts", contrast_names),
                            ("candidate_oracles", oracle_names))}


def _get_metric(summary, kind, name, region, metric):
    return summary.get(kind, {}).get(name, {}).get(region, {}).get(metric)


def _ge(value, threshold):
    return value is not None and np.isfinite(value) and value >= threshold


def _le(value, threshold):
    return value is not None and np.isfinite(value) and value <= threshold


def _confirmation_gate(confirmation, gate):
    if confirmation is None:
        return {"status": "INCONCLUSIVE_INCOMPLETE_PANEL", "passed": False,
                "GT_diagnostic": {"status": "NOT_EVALUATED", "passed": False}, "checks": {}}
    macro = confirmation["scene_macro"]
    scene_rows = confirmation["scenes"]
    def contrast(name, metric):
        return _get_metric(macro, "contrasts", name, "hard", metric)
    def positive_scenes(name):
        return sum((_get_metric(scene, "contrasts", name, "hard", "gain_px") or 0) > 0
                   for scene in scene_rows.values())
    supported_scenes = sum(scene["counts"]["hard"] >= gate["minimum_Q_per_scene"]
                           for scene in scene_rows.values())
    q_checks = {
        "required_scenes": len(scene_rows) == gate["required_scenes"],
        "minimum_Q_per_task": confirmation["counts"]["hard"] >= gate["minimum_Q_per_task"],
        "minimum_supported_scenes": supported_scenes >= gate["minimum_supported_scenes"],
    }
    checks = {
        "vs_no_support_gain_px": _ge(contrast("processed_vs_no_support", "gain_px"), gate["vs_no_support_Q_gain_px"]),
        "vs_no_support_gain_fraction": _ge(contrast("processed_vs_no_support", "relative_gain_fraction"), gate["vs_no_support_Q_gain_fraction"]),
        "vs_no_support_repair_gain_pp": _ge(contrast("processed_vs_no_support", "repair_delta_pp"), gate["vs_no_support_repair_gain_pp"]),
        "vs_no_support_positive_scenes": positive_scenes("processed_vs_no_support") >= gate["minimum_positive_scenes"],
        "vs_raw_gain_px": _ge(contrast("processed_vs_raw", "gain_px"), gate["processed_vs_raw_Q_gain_px"]),
        "vs_raw_positive_scenes": positive_scenes("processed_vs_raw") >= gate["minimum_positive_scenes"],
        "vs_wrong_gain_px": _ge(contrast("processed_vs_wrong", "gain_px"), gate["processed_vs_wrong_Q_gain_px"]),
        "vs_wrong_repair_gain_pp": _ge(contrast("processed_vs_wrong", "repair_delta_pp"), gate["processed_vs_wrong_repair_gain_pp"]),
        "vs_wrong_positive_scenes": positive_scenes("processed_vs_wrong") >= gate["minimum_positive_scenes"],
        "vs_propagation_gain_px": _ge(contrast("processed_vs_propagation_processed", "gain_px"), gate["processed_vs_propagation_Q_gain_px"]),
        "vs_propagation_repair_gain_pp": _ge(contrast("processed_vs_propagation_processed", "repair_delta_pp"), gate["processed_vs_propagation_repair_gain_pp"]),
        "all_valid_nonregression": _ge(_get_metric(macro, "arms", "processed", "all", "gain_px"), -gate["maximum_all_error_increase_px"]),
        "initial_good_damage": _le(_get_metric(macro, "arms", "processed", "initial_good", "initial_good_damage_pct"), gate["maximum_initial_good_damage_pct"]),
        "severe_harm": _le(_get_metric(macro, "arms", "processed", "all", "harm_over_3px_pct"), gate["maximum_severe_harm_pct"]),
    }
    gt_checks = {
        "vs_no_support_gain_px": _ge(contrast("gt_diagnostic_vs_no_support", "gain_px"), gate["vs_no_support_Q_gain_px"]),
        "vs_no_support_gain_fraction": _ge(contrast("gt_diagnostic_vs_no_support", "relative_gain_fraction"), gate["vs_no_support_Q_gain_fraction"]),
        "vs_no_support_repair_gain_pp": _ge(contrast("gt_diagnostic_vs_no_support", "repair_delta_pp"), gate["vs_no_support_repair_gain_pp"]),
    }
    required_metrics = [contrast(name, metric) for name in CONTRASTS if name != "gt_diagnostic_vs_no_support"
                        for metric in ("gain_px", "repair_delta_pp")]
    required_metrics += [_get_metric(macro, "arms", "processed", "initial_good", "initial_good_damage_pct"),
                         _get_metric(macro, "arms", "processed", "all", "gain_px")]
    def status(checks_to_use, metrics):
        if not confirmation["complete"]:
            return "INCONCLUSIVE_INCOMPLETE_PANEL"
        if confirmation["failed_prediction_cases"]:
            return "FAILED_NONFINITE_PREDICTION"
        if not all(q_checks.values()):
            return "INCONCLUSIVE_INSUFFICIENT_Q"
        if any(value is None or not np.isfinite(value) for value in metrics):
            return "INCONCLUSIVE_UNDEFINED_METRIC"
        return "PASS" if all(checks_to_use.values()) else "FAIL"
    verdict = status(checks, required_metrics)
    gt_status = status(gt_checks, [contrast("gt_diagnostic_vs_no_support", metric)
                                  for metric in ("gain_px", "relative_gain_fraction", "repair_delta_pp")])
    return {"status": verdict, "passed": verdict == "PASS", "checks": checks,
            "Q_checks": q_checks, "total_Q": confirmation["counts"]["hard"],
            "supported_scenes": supported_scenes,
            "positive_scenes": {name: positive_scenes(name) for name in ("processed_vs_no_support", "processed_vs_raw", "processed_vs_wrong")},
            "GT_diagnostic": {"status": gt_status, "passed": gt_status == "PASS", "checks": gt_checks,
                              "positive_scenes": positive_scenes("gt_diagnostic_vs_no_support")},
            "scope": "one_seed_mechanism_screening_only_not_calibrated_trust_or_significance"}


def _cost_summary(rows):
    present = [row for row in rows if row is not None]
    if not any("cost" in row for row in present):
        return {"status": "not_supplied"}
    scenes = sorted({_scene(row["metadata"]["scene"]) for row in present})
    names = sorted({name for row in present for name in row.get("cost", {})})
    metrics = ("repair_wall_seconds", "standalone_backbone_calls", "standalone_backbone_seconds",
               "support_processing_seconds", "candidate_slots", "target_candidate_reads",
               "peak_allocated_bytes", "observation_wait_frames")
    macro = {name: {metric: (_strict_mean(_strict_mean(
        row.get("cost", {}).get(name, {}).get(metric) for row in present
        if _scene(row["metadata"]["scene"]) == scene) for scene in scenes)
        if len(present) == len(rows) else None)
        for metric in metrics} for name in names}
    physical = {}
    for metric in ("forward_calls", "charged_forward_seconds", "cache_build_wall_seconds",
                   "support_processing_seconds"):
        values = [row.get("physical_cache_cost", {}).get(metric) for row in rows if row is not None]
        physical[metric] = sum(values) if len(values) == len(rows) and values and all(value is not None for value in values) else None
    return {"status": "reported" if len(present) == len(rows) else "incomplete_panel", "per_arm_scene_macro": macro, "physical_cache_totals_once_per_case": physical,
            "repair_timer_includes_offline_oracle_scoring": any(
                item.get("includes_offline_oracle_scoring", False) for row in present for item in row.get("cost", {}).values()),
            "interpretation": "standalone_arm_costs_are_alternatives_not_additive; cache_total_charged_once_per_task_case"}


def aggregate_support_results(rows, config):
    """Equal registered cases within scene, then equal scenes; gate confirmation only.

    Pooled metrics use pixel counts and remain a secondary description. Missing
    rows/arms are explicit and invalidate confirmation. Empty case-regions are
    disclosed and excluded identically for all arms within each scene. Missing
    rows and failed nonempty regions invalidate the macro. An entirely empty
    registered scene is not excluded: its task macro remains undefined.
    The 5% improvement is macro gain divided by macro reference error.
    """
    rows = list(rows)
    profiles = [_profile(value) for value in config["profiles"]]
    if len(profiles) != len(set(profiles)):
        raise ValueError("Profiles must be unique")
    tasks = tuple(config.get("tasks", ("stereo", "flow")))
    arms = tuple(config["evaluation"]["arms"])
    oracle_names = sorted({name for row in rows for name in row.get("candidate_oracles", {})})
    regions = sorted(set(REGIONS) | {region for row in rows for values in row["arms"].values() for region in values})
    lookup = {}
    scene_split = {}
    registered = {}
    for split, scenes in config["splits"].items():
        registered[split] = {}
        for scene, frames in scenes.items():
            scene = _scene(scene)
            if scene in scene_split and scene_split[scene] != split:
                raise ValueError(f"Registered scene appears in multiple splits: {scene}")
            scene_split[scene] = split
            if len(frames) != len(set(frames)):
                raise ValueError("Registered frames must be unique")
            registered[split][scene] = [(int(frame), profile) for frame in frames for profile in profiles]
    for row in rows:
        metadata = row["metadata"]
        task, split = str(metadata["task"]), str(metadata["split"])
        scene, frame, profile = _scene(metadata["scene"]), int(metadata["frame"]), _profile(metadata["profile"])
        if (task not in tasks or split not in registered or scene not in registered[split]
                or (frame, profile) not in registered[split][scene]):
            raise ValueError(f"Unregistered result identity: {(task, split, scene, frame, profile)}")
        key = (task, split, scene, frame, profile)
        if key in lookup:
            raise ValueError(f"Duplicate result identity: {key}")
        lookup[key] = row
    report = {"version": "s04_registered_actual_output_gate_v1", "tasks": {}, "gate": dict(config["gate"]),
              "averaging": {"scene_macro": "equal_nonempty_registered_cases_within_scene_then_equal_all_registered_scenes",
                            "pooled": "secondary_pixel_count_weighted_query_instances_not_independent_samples",
                            "null_policy": "explicit_empty_cases_excluded_identically_within_scene; missing_or_failed_nonempty_case_or_entire_empty_scene_nulls_macro",
                            "relative_gain": "macro_gain_divided_by_macro_reference_error"},
              "primary_split": "confirmation", "calibrated_trust": False}
    for task in tasks:
        split_results = {}
        for split, scene_cases in registered.items():
            scene_results, split_rows, missing, missing_arms = {}, [], [], []
            for scene, cases in scene_cases.items():
                scene_rows = []
                for frame, profile in cases:
                    row = lookup.get((task, split, scene, frame, profile))
                    scene_rows.append(row)
                    if row is None:
                        missing.append({"scene": scene, "frame": frame, "profile": profile})
                    else:
                        absent = [name for name in arms if name not in row["arms"]]
                        if absent:
                            missing_arms.append({"scene": scene, "frame": frame, "profile": profile, "arms": absent})
                split_rows.extend(scene_rows)
                counts = {region: sum(row["counts"][region] for row in scene_rows if row is not None)
                          for region in REGIONS}
                scene_results[scene] = {**_nested_summary(scene_rows, arms, CONTRASTS, regions, oracle_names=oracle_names),
                                        "counts": counts, "expected_cases": len(cases),
                                        "recorded_cases": sum(row is not None for row in scene_rows),
                                        "case_denominators": [{"frame": frame, "profile": profile,
                                            "hard": row["counts"]["hard"] if row is not None else None,
                                            "all": row["counts"]["all"] if row is not None else None}
                                           for (frame, profile), row in zip(cases, scene_rows)]}
            split_results[split] = {
                "scene_macro": _macro_from_scenes(scene_results, arms, CONTRASTS, regions, oracle_names=oracle_names),
                "pooled": _nested_summary(split_rows, arms, CONTRASTS, regions, pooled=True, oracle_names=oracle_names),
                "scenes": scene_results, "counts": {region: sum(scene["counts"][region] for scene in scene_results.values()) for region in REGIONS},
                "expected_cases": len(split_rows), "recorded_cases": sum(row is not None for row in split_rows),
                "missing_cases": missing, "missing_arms": missing_arms,
                "empty_Q_cases": sum(row is not None and row["counts"]["hard"] == 0 for row in split_rows),
                "failed_prediction_cases": sum(row is not None and row["status"] == "failed_prediction" for row in split_rows),
                "complete": not missing and not missing_arms,
                "cost": _cost_summary(split_rows),
            }
        report["tasks"][task] = {"splits": split_results,
            "confirmation_gate": _confirmation_gate(split_results.get("confirmation"), config["gate"])}
    report["status"] = "PASS" if all(value["confirmation_gate"]["passed"] for value in report["tasks"].values()) else "NOT_PASSED"
    report["interpretation"] = "both_tasks_required; privileged_GT_success_does_not_pass_actual_support_gate"
    return report


def render_report(report):
    """Render a concise Chinese result without upgrading failed/partial findings."""
    def number(value, digits=4):
        return "未定義" if value is None else f"{value:.{digits}f}"
    if report.get("smoke") or not any("confirmation" in value["splits"] for value in report["tasks"].values()):
        lines = ["# S04 工程 smoke", "", "本次只驗證資料、訓練、輸出與報表的串接；尚未執行 confirmation 科學 gate。", "",
                 "| Task | 已評估 cases | Q 合計 | 非有限輸出 cases |", "|---|---:|---:|---:|"]
        for task, value in report["tasks"].items():
            splits = list(value["splits"].values())
            lines.append(f"| {task} | {sum(item['recorded_cases'] for item in splits)} | {sum(item['counts']['hard'] for item in splits)} | {sum(item['failed_prediction_cases'] for item in splits)} |")
        return "\n".join(lines + ["", "少步數 smoke 的數值不作方法有效性結論，也不觸發調參或重訓。", ""])
    lines = ["# S04 支援條件化重新匹配", "", f"共同機制 gate：**{report['status']}**。", "",
             "正式判定只使用固定 confirmation、最後 checkpoint 的實際輸出。GT 支援及候選 oracle 均為診斷，不能代替實際 arm 通過。", "",
             "| Task | 判定 | Q | Processed vs none：Q gain | 修復率差 pp | Processed vs raw：Q gain | vs wrong：Q gain | vs propagation：Q gain | GT 診斷 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for task, result in report["tasks"].items():
        gate = result["confirmation_gate"]
        confirmation = result["splits"].get("confirmation", {})
        macro = confirmation.get("scene_macro", {})
        def value(name, metric="gain_px"):
            return number(_get_metric(macro, "contrasts", name, "hard", metric))
        lines.append(f"| {task} | {gate['status']} | {gate.get('total_Q', 0):,} | {value('processed_vs_no_support')} | {value('processed_vs_no_support', 'repair_delta_pp')} | {value('processed_vs_raw')} | {value('processed_vs_wrong')} | {value('processed_vs_propagation_processed')} | {gate['GT_diagnostic']['status']} |")
    lines += ["", "Macro 先對 scene 內非空固定 cases 等權，再對全部固定 scenes 等權；空 case 明列且對所有 arms 使用相同排除。pooled 是另一種描述，不能互換分母。", "",
              "| Task | Scene | Q 合計 | 各固定 case 的 Q |", "|---|---|---:|---|"]
    for task, result in report["tasks"].items():
        for scene, data in result["splits"].get("confirmation", {}).get("scenes", {}).items():
            counts = "; ".join(f"{row['frame']}/{row['profile']}={row['hard']}" for row in data["case_denominators"])
            lines.append(f"| {task} | {scene} | {data['counts']['hard']:,} | {counts} |")
    lines += ["", "空 Q case 明列；缺列、非空失敗案例及整個空 scene 都不會被刪除以取得更好的 macro。完整 JSON 同時保留全有效區、initial-good 傷害、>3 px／>5 px 誤修、部分改善及各對照的 pooled 結果。", ""]
    if report["status"] == "PASS":
        lines.append("兩任務通過本次預設機制篩選；單 seed、受控 crop 結果仍不代表統計顯著、可信度已校準或可安全部署。")
    else:
        lines.append("本版尚未通過共同機制 gate。候選存在、單 arm 改善或特權 GT 診斷通過，都不能改寫此判定；下一版需另立設定與確認協議。")
    return "\n".join(lines) + "\n"
