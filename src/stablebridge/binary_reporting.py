"""Native-error evaluation and harm-constrained S05 binary calibration.

There are only two outputs: preserve E0, or use the one frozen candidate A.
Scores and eligibility decide which output to use. GT errors are consumed only
by evaluation and by the explicitly separate calibration procedure below.
Fractions are in [0, 1], never percentages. No native error/gain is clipped.
"""
from collections import defaultdict
import numpy as np


QUANTILES = (0., .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99, .995, .999, 1.)
DEFAULT_CONFIG = {
    "max_damage_good_fraction": .005,
    "max_severe_harm_fraction": .001,
    "min_update_fraction": .001,
    "min_gain_px": .01,
    "min_positive_scenes": 3,
    "expected_scenes": 4,
}
DENOMINATORS = {
    "baseline_error_px": "valid_pixels", "error_px": "valid_pixels",
    "gain_px": "valid_pixels", "update_fraction": "valid_pixels",
    "eligible_fraction": "valid_pixels", "benefit_mass": "valid_pixels",
    "harm_mass": "valid_pixels", "severe_harm_fraction": "valid_pixels",
    "damage_good_fraction": "original_good_pixels", "repair_rate": "original_bad_pixels",
    "error_p95_px": "valid_pixels", "error_p99_px": "valid_pixels",
    "harm_p95_px": "valid_pixels", "harm_p99_px": "valid_pixels",
    "worst_harm_px": "valid_pixels",
}


def _mask(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean mask with shape {shape}")
    return value


def evaluate_case(e0, ea, valid, eligible, accept, metadata=None):
    """Evaluate one emitted binary output with the complete valid denominator.

    All arrays have the same arbitrary shape. e0/ea are native, nonnegative
    per-pixel errors. ``valid`` is the fixed evaluation mask, never a decision
    input. ``accept`` must be a subset of observable ``eligible``. Unaccepted
    NaN/Inf candidate errors do not contaminate the identity output; accepted
    nonfinite errors are explicit failures, not silently removed or repaired.

    B=benefit_mass and H=harm_mass are mean positive and negative native gains,
    normalized by ALL valid pixels, so gain=B-H. Their unnormalized sums are
    also returned. Good damage is conditional on original e0<=1; repair is
    conditional on original e0>1. Empty conditional cohorts return None.
    Error and harm tails both use all valid pixels, including untouched ones.
    """
    e0, ea = np.asarray(e0, dtype=np.float64), np.asarray(ea, dtype=np.float64)
    if e0.shape != ea.shape:
        raise ValueError("e0 and ea must have the same shape")
    valid = _mask(valid, e0.shape, "valid")
    eligible = _mask(eligible, e0.shape, "eligible")
    accept = _mask(accept, e0.shape, "accept")
    if np.any(accept & ~eligible):
        raise ValueError("accept must be a subset of observable eligible")
    if np.any(e0[np.isfinite(e0)] < 0) or np.any(ea[np.isfinite(ea)] < 0):
        raise ValueError("Native errors must be nonnegative")
    metadata = dict(metadata or {})
    reserved = {"counts", "metrics", "metric_status", "sums", "denominators", "status", "metadata"}
    if reserved.intersection(metadata):
        raise ValueError("metadata collides with report fields")
    base, current = e0[valid], np.where(accept, ea, e0)[valid]
    accepted = accept[valid]
    good, bad = base <= 1., base > 1.
    base_finite, current_finite = np.isfinite(base).all(), np.isfinite(current).all()
    with np.errstate(invalid="ignore", over="ignore"):
        gain = base-current
    comparable = base_finite and current_finite and np.isfinite(gain).all()
    harm = np.maximum(-gain, 0.)
    benefit = np.maximum(gain, 0.)
    n = len(base)
    counts = {
        "total_pixels": int(e0.size), "valid_pixels": n, "invalid_pixels": int(e0.size-n),
        "eligible_pixels": int((eligible & valid).sum()), "accepted_pixels": int(accepted.sum()),
        "original_good_pixels": int(good.sum()), "original_bad_pixels": int(bad.sum()),
        "nonfinite_baseline_pixels": int((~np.isfinite(base)).sum()),
        "nonfinite_output_pixels": int((~np.isfinite(current)).sum()),
        "nonfinite_eligible_candidate_pixels": int((valid & eligible & ~np.isfinite(ea)).sum()),
        "severe_harm_pixels": int((accepted & (harm > 3.)).sum()) if comparable else None,
        "damaged_good_pixels": int((accepted & good & (current > 1.)).sum()) if comparable else None,
        "repaired_bad_pixels": int((accepted & bad & (current <= 1.)).sum()) if comparable else None,
    }
    metrics, statuses = {}, {}
    def put(name, value, *, finite=True):
        denominator = counts[DENOMINATORS[name]]
        if not finite:
            metrics[name], statuses[name] = None, "failed_prediction"
        elif not denominator:
            metrics[name], statuses[name] = None, "empty_denominator"
        else:
            metrics[name], statuses[name] = float(value), "ok"
    safe_n = max(n, 1)
    put("baseline_error_px", base.sum()/safe_n, finite=base_finite)
    put("error_px", current.sum()/safe_n, finite=current_finite)
    put("gain_px", gain.sum()/safe_n, finite=comparable)
    put("benefit_mass", benefit.sum()/safe_n, finite=comparable)
    put("harm_mass", harm.sum()/safe_n, finite=comparable)
    put("update_fraction", counts["accepted_pixels"]/safe_n)
    put("eligible_fraction", counts["eligible_pixels"]/safe_n)
    put("severe_harm_fraction", (counts["severe_harm_pixels"] or 0)/safe_n, finite=comparable)
    put("damage_good_fraction", (counts["damaged_good_pixels"] or 0)/max(int(good.sum()), 1), finite=comparable)
    put("repair_rate", (counts["repaired_bad_pixels"] or 0)/max(int(bad.sum()), 1), finite=comparable)
    for percentile in (95, 99):
        put(f"error_p{percentile}_px", np.percentile(current, percentile) if n and current_finite else 0., finite=current_finite)
        put(f"harm_p{percentile}_px", np.percentile(harm, percentile) if n and comparable else 0., finite=comparable)
    put("worst_harm_px", harm.max() if n and comparable else 0., finite=comparable)
    sums = {"gain_px": float(gain.sum()) if comparable else None,
            "benefit_px": float(benefit.sum()) if comparable else None,
            "harm_px": float(harm.sum()) if comparable else None}
    return {**metadata, "metadata": metadata, "counts": counts, "metrics": metrics,
            "metric_status": statuses, "sums": sums, "denominators": dict(DENOMINATORS),
            "status": "empty" if not n else ("ok" if comparable else "failed_prediction")}


def _one_cohort(items, *, calibration=False):
    """Refuse accidental averaging/calibration across tasks, splits or arms."""
    for key in ("task", "split", "arm"):
        values = {str(item.get("metadata", item).get(key)) for item in items
                  if item.get("metadata", item).get(key) is not None}
        if len(values) > 1:
            raise ValueError(f"Group {key} separately; got mixed {key}: {sorted(values)}")
        if calibration and key == "split" and values and values != {"calibration"}:
            raise ValueError("choose_threshold accepts only the calibration split")


def aggregate(rows):
    """Equal cases (conditions/frames) within scene, then equal scene means.

    Empty conditional rates may be omitted, with their counts and denominators
    reported. Nonempty failed cases are never omitted: they invalidate that
    scene and the overall metric. Quantile metrics are scene macros of case
    quantiles, not a pooled-pixel quantile. Group tasks/splits/arms beforehand.
    """
    rows = list(rows)
    _one_cohort(rows)
    grouped = defaultdict(list)
    for row in rows:
        scene = row.get("scene", row.get("metadata", {}).get("scene"))
        if scene is None:
            raise ValueError("Every case needs metadata.scene for scene-macro evaluation")
        grouped[str(scene)].append(row)
    result = {"cases": len(rows), "scenes": len(grouped), "scene_ids": sorted(grouped),
              "within_scene_weighting": "equal_conditions_and_frames",
              "between_scene_weighting": "equal_scenes", "metrics": {}}
    for metric, denominator_key in DENOMINATORS.items():
        per_scene = {}
        contributing_cases = omitted_cases = failed_cases = denominator_pixels = 0
        for scene, cases in sorted(grouped.items()):
            active, failed, empty, denominator = [], 0, 0, 0
            for row in cases:
                denominator += row["counts"][denominator_key]
                status = row["metric_status"][metric]
                value = row["metrics"][metric]
                if status == "empty_denominator":
                    empty += 1
                elif status != "ok" or value is None or not np.isfinite(value):
                    failed += 1
                else:
                    active.append(value)
            per_scene[scene] = {"value": float(np.mean(active)) if active and not failed else None,
                                "cases": len(cases), "contributing_cases": len(active),
                                "omitted_empty_cases": empty, "failed_cases": failed,
                                "denominator_pixels": denominator}
            contributing_cases += len(active)
            omitted_cases += empty
            failed_cases += failed
            denominator_pixels += denominator
        values = [s["value"] for s in per_scene.values() if s["value"] is not None]
        result["metrics"][metric] = {
            "scene_macro": float(np.mean(values)) if values and not failed_cases else None,
            "per_scene": per_scene, "contributing_cases": contributing_cases,
            "omitted_empty_cases": omitted_cases, "failed_cases": failed_cases,
            "contributing_scenes": len(values),
            "omitted_empty_scenes": sum(s["omitted_empty_cases"] == s["cases"] for s in per_scene.values()),
            "failed_scenes": sum(s["failed_cases"] > 0 for s in per_scene.values()),
            "denominator": denominator_key, "denominator_pixels": denominator_pixels,
        }
    return result


def _config(config):
    result = {**DEFAULT_CONFIG, **(config or {})}
    for key in ("max_damage_good_fraction", "max_severe_harm_fraction", "min_update_fraction"):
        if not np.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f"{key} must be a fraction in [0,1]")
    if not np.isfinite(result["min_gain_px"]) or result["min_gain_px"] < 0:
        raise ValueError("min_gain_px must be finite and nonnegative")
    for key in ("min_positive_scenes", "expected_scenes"):
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if result["min_positive_scenes"] > result["expected_scenes"]:
        raise ValueError("min_positive_scenes cannot exceed expected_scenes")
    return result


def apply_policy(score, eligible, policy):
    """GT-free binary selection, suitable for a frozen confirmation policy."""
    score = np.asarray(score)
    eligible = _mask(eligible, score.shape, "eligible")
    if policy.get("keep_identity"):
        if policy.get("threshold") is not None:
            raise ValueError("Identity policy must have null threshold")
        return np.zeros(score.shape, bool)
    threshold = policy.get("threshold")
    if threshold is None or not np.isfinite(threshold):
        raise ValueError("Active policy needs a finite threshold")
    return eligible & np.isfinite(score) & (score > threshold)


def _feasibility(summary, config):
    values = {key: summary["metrics"][key]["scene_macro"] for key in (
        "damage_good_fraction", "severe_harm_fraction", "update_fraction", "gain_px")}
    checks = {
        "damage_good_bound": values["damage_good_fraction"] is not None and values["damage_good_fraction"] <= config["max_damage_good_fraction"],
        "severe_harm_bound": values["severe_harm_fraction"] is not None and values["severe_harm_fraction"] <= config["max_severe_harm_fraction"],
        "minimum_update": values["update_fraction"] is not None and values["update_fraction"] >= config["min_update_fraction"],
        "minimum_gain": values["gain_px"] is not None and values["gain_px"] >= config["min_gain_px"],
    }
    return {"harm_feasible": checks["damage_good_bound"] and checks["severe_harm_bound"],
            "active_feasible": all(checks.values()), "checks": checks}


def choose_threshold(cases, config=None):
    """Calibrate once on a separate, single-task calibration cohort.

    Each case has score/e0/ea/valid/eligible arrays plus metadata (including
    scene; split, when present, must be 'calibration'). The grid is the fixed
    QUANTILES of pooled *eligible finite scores*, with no GT-valid filter.
    Accept iff eligible & finite(score) & score > threshold. +Inf (keep every
    original output) is represented in JSON as threshold=None, keep_identity.
    Baseline allcandidate accepts every eligible pixel, independent of score.

    Maximize native scene-macro gain subject to both harm constraints and both
    activity floors. Exact ties prefer fewer updates, then larger threshold.
    If no active threshold qualifies, return an explicit identity fallback.
    """
    cases = list(cases)
    _one_cohort(cases, calibration=True)
    cfg = _config(config)
    pooled = []
    for case in cases:
        score = np.asarray(case["score"])
        eligible = _mask(case["eligible"], score.shape, "eligible")
        pooled.append(score[eligible & np.isfinite(score)])
    pool = np.concatenate(pooled) if pooled else np.empty(0)
    quantile_thresholds = list(zip(QUANTILES, np.quantile(pool, QUANTILES).tolist())) if len(pool) else []
    identity_policy = {"threshold": None, "keep_identity": True, "comparison": "strict_greater"}

    def evaluate(policy=None, *, allcandidate=False):
        rows = [evaluate_case(case["e0"], case["ea"], case["valid"], case["eligible"],
                              np.asarray(case["eligible"]) if allcandidate else apply_policy(case["score"], case["eligible"], policy),
                              case.get("metadata")) for case in cases]
        summary = aggregate(rows)
        return {"rows": rows, "summary": summary, **_feasibility(summary, cfg)}

    identity = evaluate(identity_policy)
    allcandidate = evaluate(allcandidate=True)
    grid, best, evaluated_thresholds = [], None, {}
    for quantile, threshold in quantile_thresholds:
        policy = {"threshold": float(threshold), "keep_identity": False, "comparison": "strict_greater"}
        if threshold not in evaluated_thresholds:
            evaluated_thresholds[threshold] = evaluate(policy)
        result = evaluated_thresholds[threshold]
        grid.append({"quantile": quantile, "policy": policy, "summary": result["summary"],
                     "harm_feasible": result["harm_feasible"], "active_feasible": result["active_feasible"], "checks": result["checks"]})
        if result["active_feasible"]:
            metrics = result["summary"]["metrics"]
            updates = sum(row["counts"]["accepted_pixels"] for row in result["rows"])
            key = (metrics["gain_px"]["scene_macro"], -updates, threshold)
            if best is None or key > best[0]:
                best = key, policy, result
    grid.append({"quantile": None, "policy": identity_policy, "represents_threshold": "+infinity", "summary": identity["summary"],
                 "harm_feasible": identity["harm_feasible"], "active_feasible": False, "checks": identity["checks"]})
    if best is None:
        policy = {**identity_policy, "reason": "no_feasible_active_threshold"}
        selected = identity
    else:
        _, policy, selected = best
        policy = {**policy, "reason": "maximum_native_gain_under_harm_constraints"}
    return {"policy": policy, "rows": selected["rows"], "summary": selected["summary"],
            "grid_evaluations": grid, "score_quantiles": list(QUANTILES), "pooled_eligible_finite_scores": len(pool),
            "config": cfg, "baselines": {"identity": identity, "allcandidate": allcandidate},
            "selection_scope": "calibration_only; frozen threshold must be reused on confirmation",
            "native_errors_clipped": False}


def confirmation_gate(summary, config=None):
    """Descriptive registered gate, not a statistical significance claim."""
    cfg = _config(config)
    feasibility = _feasibility(summary, cfg)
    scene_values = summary["metrics"]["gain_px"]["per_scene"]
    positives = [scene for scene, item in scene_values.items() if item["value"] is not None and item["value"] > 0]
    checks = {**feasibility["checks"], "expected_scene_count": summary["scenes"] == cfg["expected_scenes"],
              "minimum_positive_scenes": len(positives) >= cfg["min_positive_scenes"]}
    return {"passed": all(checks.values()), "checks": checks, "positive_scenes": len(positives),
            "positive_scene_ids": positives, "scenes": summary["scenes"], "config": cfg,
            "statistical_significance_claim": False}
