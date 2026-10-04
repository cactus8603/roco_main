"""Offline, fixed-query candidate-capacity scoring for S02 quartet paths.

This module creates oracle labels only.  It does not infer actionable updates or
calibrated trust.  All errors are native-pixel whole-vector Spring min4 errors.
The caller must put identity first in C0 and supply the separately recorded raw
2-D path and the candidate admitted by its registered geometry adapter.
"""
import numpy as np


SCORER_VERSION = "s02_fixed_query_capacity_min4_v1"
STATE_NAMES = (
    "outside_unresolved_E0_oracle",
    "newly_repairable_E1_oracle",
    "still_unresolved_partial_improvement",
    "still_unresolved_no_improvement",
    "still_unresolved_no_usable_path",
)


def _bank(value, name, n=None, j=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] != 2 or value.shape[0] < 1:
        raise ValueError(f"{name} must be nonempty KxNx2")
    if n is not None and value.shape[1] != n:
        raise ValueError(f"{name} has a different query count")
    if j is not None and value.shape[0] != j:
        raise ValueError(f"{name} has a different registered candidate count")
    return value


def _mask(value, name, shape):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array of shape {shape}")
    return value


def _errors(predictions, gt4):
    # Any NaN invalidates the query under Spring semantics.  Validity is kept
    # separately: an infinite prediction never removes a query from Q.
    with np.errstate(invalid="ignore", over="ignore"):
        errors = np.linalg.norm(predictions[:, None] - gt4[None], axis=-1).min(axis=1)
    return np.where(np.isfinite(errors), errors, np.inf)


def _mean(values):
    values = np.asarray(values)
    return float(np.mean(values, dtype=np.float64)) if values.size and np.isfinite(values).all() else None


def _pct(values):
    values = np.asarray(values)
    return 100. * float(np.mean(values)) if values.size else None


def _difference(left, right):
    with np.errstate(invalid="ignore", over="ignore"):
        return left - right


def _paired(initial, current, region, threshold, harm_delta, tail_delta):
    initial, current = initial[region], current[region]
    gain = _difference(initial, current)
    undefined = np.isnan(gain).any()
    return {
        "pixels": int(initial.size),
        "baseline_error_px": _mean(initial),
        "error_px": _mean(current),
        "signed_gain_px": _mean(gain),
        "repair_rate_pct": _pct(current <= threshold),
        "harm_over_delta_pct": None if undefined else _pct(gain < -harm_delta),
        "tail_harm_over_delta_pct": None if undefined else _pct(gain < -tail_delta),
        "nonfinite_baseline_pixels": int((~np.isfinite(initial)).sum()),
        "nonfinite_candidate_pixels": int((~np.isfinite(current)).sum()),
    }


def _candidate_family(raw, candidate, computable, eligible, gt4, direct_error,
                      q, threshold, harm_delta, tail_delta):
    raw_errors, candidate_errors = _errors(raw, gt4), _errors(candidate, gt4)
    raw_fallback = np.where(eligible, raw_errors, direct_error[None])
    candidate_fallback = np.where(eligible, candidate_errors, direct_error[None])
    computable_raw = np.where(computable, raw_errors, direct_error[None])
    per_candidate = []
    for j in range(len(raw)):
        usable = q & computable[j]
        row = {
            "candidate_index": j,
            "computable_coverage_pct": _pct(computable[j, q]),
            "eligible_coverage_pct": _pct(eligible[j, q]),
            "raw2d": _paired(direct_error, raw_fallback[j], q, threshold, harm_delta, tail_delta),
            "scored_candidate": _paired(direct_error, candidate_fallback[j], q, threshold, harm_delta, tail_delta),
            "raw2d_eligible_only": _paired(direct_error, raw_errors[j], q & eligible[j], threshold, harm_delta, tail_delta),
            "scored_candidate_eligible_only": _paired(direct_error, candidate_errors[j], q & eligible[j], threshold, harm_delta, tail_delta),
            "computable_raw2d_diagnostic": _paired(direct_error, computable_raw[j], q, threshold, harm_delta, tail_delta),
            "vertical_residual_computable_pixels": int(usable.sum()),
            "absolute_vertical_residual_px": _mean(np.abs(raw[j, usable, 1])),
            "candidate_adapter_shift_px": _mean(np.linalg.norm(candidate[j, usable] - raw[j, usable], axis=-1)),
        }
        per_candidate.append(row)
    macro = {}
    for representation in ("raw2d", "scored_candidate", "computable_raw2d_diagnostic",
                           "raw2d_eligible_only", "scored_candidate_eligible_only"):
        macro[representation] = {
            metric: _mean([row[representation][metric] for row in per_candidate])
            if all(row[representation][metric] is not None for row in per_candidate) else None
            for metric in ("baseline_error_px", "error_px", "signed_gain_px", "repair_rate_pct",
                           "harm_over_delta_pct", "tail_harm_over_delta_pct")
        }
    return {
        "computable_coverage_pct": _pct(computable[:, q].any(axis=0)),
        "eligible_coverage_pct": _pct(eligible[:, q].any(axis=0)),
        "per_candidate": per_candidate,
        "fixed_candidate_macro": macro,
        "eligible_only_denominators": [row["raw2d_eligible_only"]["pixels"] for row in per_candidate],
        "macro_weighting": "equal_all_registered_candidates; any_empty_candidate_nulls_eligible_only_macro",
        "fallback": "identity_for_every_ineligible_or_unavailable_candidate",
    }, raw_errors, candidate_errors, raw_fallback, candidate_fallback


def evaluate_capacity(c0, path_raw, path_candidates, path_computable, path_eligible,
                      rematch, rematch_valid, wrong, wrong_computable, wrong_eligible,
                      gt4, threshold=1., harm_delta=1., tail_delta=5., *, wrong_raw=None):
    """Return ``(JSON-safe summary, per-query NumPy sidecar)``.

    Bank shapes are C0=(K,N,2), other banks=(J,N,2), masks=(J,N), and
    GT=(4,N,2). C0[0] is identity.  All banks use the same dense query order.
    The fixed cohort is GT-valid AND min(C0 error)>threshold.  CP, CR and CW
    each retain ALL C0 candidates, so they are comparable oracle capacities.

    Masked unavailable paths can contain nonfinite sentinels. Nonfinite C0
    predictions at GT-valid sites, or predictions declared computable/valid,
    explicitly fail the evaluation; they are never dropped from Q.  Eligibility
    must imply computability. No visibility or same-surface claim is inferred.
    ``path_candidates`` may be a registered stereo horizontal projection;
    ``path_raw`` is always evaluated and retained independently in 2-D.
    ``wrong`` is the admitted wrong-path candidate. ``wrong_raw`` preserves its
    unprojected displacement; omitting it explicitly assumes wrong is raw too.
    """
    if not all(np.isfinite(x) for x in (threshold, harm_delta, tail_delta)):
        raise ValueError("Thresholds must be finite")
    if threshold <= 0 or harm_delta < 0 or tail_delta < harm_delta:
        raise ValueError("Require threshold > 0 and 0 <= harm_delta <= tail_delta")
    c0 = _bank(c0, "c0")
    n = c0.shape[1]
    path_raw = _bank(path_raw, "path_raw", n)
    j = path_raw.shape[0]
    path_candidates = _bank(path_candidates, "path_candidates", n, j)
    rematch = _bank(rematch, "rematch", n, j)
    wrong = _bank(wrong, "wrong", n, j)
    wrong_raw = wrong if wrong_raw is None else _bank(wrong_raw, "wrong_raw", n, j)
    gt4 = np.asarray(gt4, dtype=np.float64)
    if gt4.shape != (4, n, 2):
        raise ValueError("gt4 must be 4xNx2")
    shape = (j, n)
    path_computable = _mask(path_computable, "path_computable", shape)
    path_eligible = _mask(path_eligible, "path_eligible", shape)
    rematch_valid = _mask(rematch_valid, "rematch_valid", shape)
    wrong_computable = _mask(wrong_computable, "wrong_computable", shape)
    wrong_eligible = _mask(wrong_eligible, "wrong_eligible", shape)
    if (path_eligible & ~path_computable).any() or (wrong_eligible & ~wrong_computable).any():
        raise ValueError("Eligible paths must also be computable")

    valid_gt = ~np.isnan(gt4).any(axis=(0, 2)) & np.isfinite(gt4).all(axis=2).any(axis=0)
    c0_errors = _errors(c0, gt4)
    direct_error, c0_min = c0_errors[0], c0_errors.min(axis=0)
    q = valid_gt & (c0_min > threshold)
    path_summary, path_raw_errors, path_errors, path_raw_fallback, path_fallback = _candidate_family(
        path_raw, path_candidates, path_computable, path_eligible, gt4, direct_error,
        q, threshold, harm_delta, tail_delta)
    wrong_summary, wrong_raw_errors, wrong_errors, wrong_raw_fallback, wrong_fallback = _candidate_family(
        wrong_raw, wrong, wrong_computable, wrong_eligible, gt4, direct_error,
        q, threshold, harm_delta, tail_delta)
    rematch_errors = _errors(rematch, gt4)
    cp_min = np.minimum(c0_min, np.where(path_eligible, path_errors, np.inf).min(axis=0))
    cr_min = np.minimum(c0_min, np.where(rematch_valid, rematch_errors, np.inf).min(axis=0))
    cw_min = np.minimum(c0_min, np.where(wrong_eligible, wrong_errors, np.inf).min(axis=0))
    any_path, any_wrong = path_eligible.any(axis=0), wrong_eligible.any(axis=0)
    common = q & any_path & any_wrong
    repair_p, repair_r, repair_w = cp_min <= threshold, cr_min <= threshold, cw_min <= threshold
    paired_delta = _difference(cr_min, cp_min)
    path_gain = _difference(c0_min, cp_min)
    state = np.zeros(n, dtype=np.int8)
    state[q & repair_p] = 1
    state[q & ~repair_p & any_path & (cp_min < c0_min)] = 2
    state[q & ~repair_p & any_path & ~(cp_min < c0_min)] = 3
    state[q & ~any_path] = 4
    assert np.count_nonzero(state) == np.count_nonzero(q)

    nonfinite = {
        "c0": ~np.isfinite(c0).all(axis=-1) & valid_gt[None],
        "path_raw_computable": ~np.isfinite(path_raw).all(axis=-1) & path_computable & valid_gt[None],
        "path_candidate_eligible": ~np.isfinite(path_candidates).all(axis=-1) & path_eligible & valid_gt[None],
        "rematch_valid": ~np.isfinite(rematch).all(axis=-1) & rematch_valid & valid_gt[None],
        "wrong_computable": ~np.isfinite(wrong_raw).all(axis=-1) & wrong_computable & valid_gt[None],
        "wrong_candidate_eligible": ~np.isfinite(wrong).all(axis=-1) & wrong_eligible & valid_gt[None],
    }
    failures = {name: int(mask.sum()) for name, mask in nonfinite.items()}
    status = "failed_prediction" if any(failures.values()) else ("ok" if q.any() else "empty")
    summary = {
        "scorer_version": SCORER_VERSION,
        "status": status,
        "threshold_px": float(threshold),
        "harm_delta_px": float(harm_delta),
        "tail_harm_delta_px": float(tail_delta),
        "cohort": "gt_valid_and_min_C0_error_above_threshold",
        "counts": {
            "dense_queries": n,
            "gt_valid": int(valid_gt.sum()),
            "error_E0_gt": int((valid_gt & (direct_error > threshold)).sum()),
            "unresolved_E0_oracle": int(q.sum()),
            "c0_candidates": int(c0.shape[0]),
            "new_candidates_per_arm": j,
        },
        "unresolved_E0_oracle_pct_of_gt_valid": _pct(q[valid_gt]),
        "primary": {
            "paired_delta_px": _mean(paired_delta[q]),
            "path_gain_over_c0_px": _mean(path_gain[q]),
            "rematch_gain_over_c0_px": _mean(_difference(c0_min, cr_min)[q]),
            "wrong_gain_over_c0_px": _mean(_difference(c0_min, cw_min)[q]),
            "newly_repairable_E1_oracle_pct": _pct(repair_p[q]),
            "rematch_repair_rate_pct": _pct(repair_r[q]),
            "wrong_repair_rate_pct": _pct(repair_w[q]),
            "primary_minus_wrong_repair_rate_pp": _mean(100. * (repair_p[q].astype(float) - repair_w[q])),
            "primary_minus_rematch_repair_rate_pp": _mean(100. * (repair_p[q].astype(float) - repair_r[q])),
        },
        "oracles": {
            name: _paired(c0_min, minimum, q, threshold, harm_delta, tail_delta)
            for name, minimum in (("C0", c0_min), ("CP", cp_min), ("CR", cr_min), ("CW", cw_min))
        },
        "path": path_summary,
        "wrong": wrong_summary,
        "raw_path_raw2d": {
            "all_Q": path_summary["fixed_candidate_macro"]["raw2d"],
            "eligible_only": path_summary["fixed_candidate_macro"]["raw2d_eligible_only"],
            "eligible_only_candidate_pixels": path_summary["eligible_only_denominators"],
        },
        "raw_path_projected": {
            "all_Q": path_summary["fixed_candidate_macro"]["scored_candidate"],
            "eligible_only": path_summary["fixed_candidate_macro"]["scored_candidate_eligible_only"],
            "eligible_only_candidate_pixels": path_summary["eligible_only_denominators"],
            "semantics": "caller_supplied_geometry_candidate; flow_may_be_unprojected",
        },
        "rematch_valid_coverage_pct": _pct(rematch_valid[:, q].any(axis=0)),
        "common_eligible": {
            "pixels": int(common.sum()),
            "coverage_pct_of_fixed_Q": _pct(common[q]),
            "primary_repair_rate_pct": _pct(repair_p[common]),
            "wrong_repair_rate_pct": _pct(repair_w[common]),
            "primary_minus_wrong_repair_rate_pp": _mean(100. * (repair_p[common].astype(float) - repair_w[common])),
        },
        "states": {name: {"pixels": int((state == code).sum()), "pct_of_fixed_Q": _pct(state[q] == code)}
                   for code, name in enumerate(STATE_NAMES) if code},
        "nonfinite_predictions_on_gt_valid": failures,
        "interpretation": "oracle_candidate_capacity_only; actionable_and_trusted_are_not_inferred",
    }
    arrays = {
        "gt_valid": valid_gt, "Q": q, "unresolved_E0_oracle": q,
        "error_E0_gt": valid_gt & (direct_error > threshold),
        "c0_error": c0_errors, "identity_error": direct_error,
        "path_raw2d_error": path_raw_errors, "path_candidate_error": path_errors,
        "path_raw2d_error_with_identity_fallback": path_raw_fallback,
        "path_candidate_error_with_identity_fallback": path_fallback,
        "rematch_error": rematch_errors, "wrong_error": wrong_errors,
        "wrong_raw2d_error": wrong_raw_errors,
        "wrong_raw2d_error_with_identity_fallback": wrong_raw_fallback,
        "wrong_error_with_identity_fallback": wrong_fallback,
        "C0_min_error": c0_min, "CP_min_error": cp_min,
        "CR_min_error": cr_min, "CW_min_error": cw_min,
        "paired_delta": paired_delta, "path_gain_over_c0": path_gain,
        "path_computable": path_computable, "path_eligible": path_eligible,
        "rematch_valid": rematch_valid, "wrong_computable": wrong_computable,
        "wrong_eligible": wrong_eligible, "common_eligible_Q": common,
        "path_vertical_residual": path_raw[..., 1],
        "path_candidate_adapter_shift": path_candidates - path_raw,
        "wrong_vertical_residual": wrong_raw[..., 1],
        "wrong_candidate_adapter_shift": wrong - wrong_raw,
        "state_code": state,
    }
    arrays.update({f"nonfinite_{name}": mask for name, mask in nonfinite.items()})
    return summary, arrays
