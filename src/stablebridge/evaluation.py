"""Fixed-denominator Spring min4 scoring; labels never enter inference code."""
from collections import defaultdict
import numpy as np


SCORER_VERSION = "spring_min4_native_v1"


def _validate(prediction, gt4):
    prediction, gt4 = np.asarray(prediction), np.asarray(gt4)
    if prediction.ndim != 3 or prediction.shape[0] != 2:
        raise ValueError("Prediction must be 2xHxW native displacement")
    if gt4.shape != (4,)+prediction.shape:
        raise ValueError("Ground truth must be 4x2xHxW")
    return prediction, gt4


def gt_valid_mask(gt4):
    """Vendor semantics: any NaN invalidates min4; a finite branch beats Inf."""
    gt4 = np.asarray(gt4)
    if gt4.ndim != 4 or gt4.shape[:2] != (4,2):
        raise ValueError("Ground truth must be 4x2xHxW")
    return ~np.isnan(gt4).any(axis=(0,1)) & np.isfinite(gt4).all(axis=1).any(axis=0)


def error_map(prediction, gt4):
    """Minimum WHOLE-vector error across four GT samples, without rescaling.

    np.min propagates NaN like the official torch.minimum implementation.
    Nonfinite results become Inf; validity is a separate GT-only mask so a
    failed prediction never silently removes a position from the denominator.
    """
    prediction, gt4 = _validate(prediction, gt4)
    with np.errstate(invalid="ignore", over="ignore"):
        errors = np.linalg.norm(gt4-prediction[None], axis=1).min(axis=0)
    return np.where(np.isfinite(errors), errors, np.inf)


def fixed_hard_mask(reference_predictions, gt4, threshold=1.):
    """Freeze H* from a registered A_ref bank, including its identity U0."""
    predictions = np.asarray(reference_predictions)
    if predictions.ndim != 4 or predictions.shape[0] < 1:
        raise ValueError("A_ref must contain at least one Kx2xHxW prediction")
    errors = np.stack([error_map(prediction, gt4) for prediction in predictions])
    return gt_valid_mask(gt4) & (errors.min(axis=0) > threshold)


def region_metrics(initial_error, current_error, region, *, threshold=1., severe_harm_px=3.):
    """Use a pre-fixed region, including every failed current prediction.

    Nonfinite output yields error_px=None plus explicit failure count; this
    JSON-safe null is a failure flag, never a reason to average fewer cases.
    """
    mask = np.asarray(region, dtype=bool)
    if mask.shape != np.shape(initial_error) or mask.shape != np.shape(current_error):
        raise ValueError("Error and region shapes must match")
    initial, current = np.asarray(initial_error)[mask], np.asarray(current_error)[mask]
    count = len(initial)
    failures = int((~np.isfinite(current)).sum())
    base_failures = int((~np.isfinite(initial)).sum())
    def mean(values):
        return float(np.mean(values, dtype=np.float64)) if count and np.isfinite(values).all() else None
    with np.errstate(invalid="ignore"):
        gain = initial-current
    return {"pixels": count, "baseline_error_px": mean(initial), "error_px": mean(current),
            "gain_px": mean(gain), "benefit_px": mean(np.maximum(gain, 0)),
            "harm_px": mean(np.maximum(-gain, 0)),
            "over_1px_pct": 100*float(np.mean(current > threshold)) if count else None,
            "severe_harm_pct": 100*float(np.mean(gain < -severe_harm_px)) if count else None,
            "repaired_to_threshold_pct": 100*float(np.mean(current <= threshold)) if count else None,
            "nonfinite_prediction_pixels": failures,
            "nonfinite_baseline_pixels": base_failures,
            "status": "empty" if not count else ("failed_prediction" if failures or base_failures else "ok")}


def evaluate_pair(baseline, current, gt4, hard_mask=None, *, threshold=1., severe_harm_px=3., selected_mask=None):
    """Score all/initial-good/fixed-hard cohorts with identical GT denominators."""
    initial, updated = error_map(baseline, gt4), error_map(current, gt4)
    valid = gt_valid_mask(gt4)
    if threshold <= 0 or severe_harm_px < 0:
        raise ValueError("Threshold must be positive and harm tolerance nonnegative")
    regions = {"all": valid, "initial_good": valid & (initial <= threshold),
               "initial_bad": valid & (initial > threshold)}
    if hard_mask is not None:
        if np.shape(hard_mask) != valid.shape:
            raise ValueError("hard_mask shape does not match image")
        regions["hard"] = valid & np.asarray(hard_mask, dtype=bool)
    if selected_mask is not None:
        if np.shape(selected_mask) != valid.shape:
            raise ValueError("selected_mask shape does not match image")
        regions["selected"] = valid & np.asarray(selected_mask, dtype=bool)
    return {"scorer_version": SCORER_VERSION, "threshold_px": threshold,
            "severe_harm_px": severe_harm_px,
            "regions": {name: region_metrics(initial, updated, region, threshold=threshold,
                                             severe_harm_px=severe_harm_px)
                        for name, region in regions.items()}}


def aggregate_scene_macro(rows, *, metric="error_px", region="all", value_key="evaluation"):
    """Equal cases within scene, then equal scenes, separately per task.

    Each row has task, scene, and value_key (an evaluate_pair result). Arm or
    condition comparisons must call this separately on their registered rows.
    Empty regions are counted and excluded explicitly. Failed nonempty regions
    invalidate that scene and aggregate instead of disappearing from the mean.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        result = row[value_key]["regions"][region]
        groups[str(row["task"])][str(row["scene"])].append(result)
    output = {}
    for task, scenes in sorted(groups.items()):
        scene_results = {}
        for scene, cases in sorted(scenes.items()):
            active = [case for case in cases if case["pixels"] > 0]
            missing = sum(case.get(metric) is None for case in active)
            value = float(np.mean([case[metric] for case in active])) if active and not missing else None
            scene_results[scene] = {"value": value, "cases": len(cases),
                                    "nonempty_cases": len(active), "failed_cases": missing,
                                    "empty_cases": len(cases)-len(active)}
        active_scenes = [result for result in scene_results.values() if result["nonempty_cases"]]
        failures = sum(result["value"] is None for result in active_scenes)
        macro = float(np.mean([result["value"] for result in active_scenes])) if active_scenes and not failures else None
        output[task] = {"scene_macro": macro, "scenes": scene_results,
                        "nonempty_scenes": len(active_scenes), "failed_scenes": failures,
                        "within_scene_weighting": "equal_registered_rows",
                        "between_scene_weighting": "equal_scenes", "region": region, "metric": metric}
    return output
