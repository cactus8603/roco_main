"""Outcome scorer for exact action-vs-native optical-flow qualification.

The scorer is deliberately independent of dataset I/O.  Callers must supply a
pre-frozen GT-valid mask and fixed region labels.  Action-dependent finite or
support masks are never allowed to shrink the denominator.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


SCHEMA = "stablebridge-action-qualification-task-observation/v1"
MIN_VALID_PIXELS_PER_REGION = 16
HARMED_PIXEL_THRESHOLD_RAW_PX = 1.0
HARMED_PIXEL_FRACTION_UPPER = 0.05
CVAR95_HARM_UPPER_RAW_PX = 0.25
SEVERE_REGION_GAIN_BELOW_RAW_PX = -0.25


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _flow(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape HxWx2")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be floating point")
    return np.ascontiguousarray(array.astype(np.float64, copy=False))


def _ground_truth_branches(value: np.ndarray) -> np.ndarray:
    """Return KxHxWx2 GT without ever mixing vector components."""
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 2:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[-1] != 2:
        raise ValueError("ground_truth_flow must have shape HxWx2 or KxHxWx2")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("ground_truth_flow must be floating point")
    return np.ascontiguousarray(array.astype(np.float64, copy=False))


def _mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.bool_:
        raise ValueError(f"{name} must be a bool HxW array")
    return np.ascontiguousarray(array)


def _GT_masks(
    value: np.ndarray, branches: int, shape: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise ValueError("gt_valid_mask must be bool HxW or KxHxW")
    if array.shape == shape:
        array = np.broadcast_to(array, (branches, *shape))
    if array.shape != (branches, *shape):
        raise ValueError("gt_valid_mask must be bool HxW or KxHxW")
    return np.ascontiguousarray(array)


def _labels(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("region_labels must be an integer HxW array")
    if np.any(array < 0):
        raise ValueError("region labels must be nonnegative")
    return np.ascontiguousarray(array.astype(np.int64, copy=False))


def _finite_number(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values, dtype=np.float64))


def _cvar95_including_zeros(harm: np.ndarray) -> float:
    if harm.ndim != 1 or harm.size == 0:
        raise ValueError("CVaR95 needs a nonempty one-dimensional sample")
    count = max(1, math.ceil(0.05 * int(harm.size)))
    # partition avoids a full sort while retaining the largest count values.
    tail = np.partition(harm, harm.size - count)[-count:]
    return _mean(tail)


def score_action_case_v1(
    *,
    case_id: str,
    component_id: str,
    source_dataset: str,
    outer_fold: int,
    mechanism_stratum: str,
    action_id: str,
    action_binding_sha256: str,
    native_flow: np.ndarray,
    action_flow: np.ndarray,
    ground_truth_flow: np.ndarray,
    gt_valid_mask: np.ndarray,
    region_labels: np.ndarray,
    rgb_diagonal_px: float,
    write_support: np.ndarray | None = None,
    read_support: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return pixel sidecars and a hash-bound case/region observation."""
    native = _flow(native_flow, "native_flow")
    action = _flow(action_flow, "action_flow")
    truth = _ground_truth_branches(ground_truth_flow)
    if native.shape != action.shape or native.shape != truth.shape[1:]:
        raise ValueError("native/action/GT flow shapes differ")
    shape = native.shape[:2]
    branch_valid = _GT_masks(gt_valid_mask, int(truth.shape[0]), shape)
    valid = np.any(branch_valid, axis=0)
    labels = _labels(region_labels, shape)
    if not np.any(valid):
        raise ValueError("fixed GT support is empty")
    if not np.isfinite(truth[branch_valid]).all():
        raise ValueError("GT has nonfinite values on frozen valid support")
    if not np.isfinite(native[valid]).all():
        raise ValueError("native flow has nonfinite values on frozen GT support")
    if not np.isfinite(action[valid]).all():
        raise ValueError(
            "action flow has nonfinite values on frozen GT support; denominator "
            "shrinking is forbidden"
        )
    diagonal = _finite_number(rgb_diagonal_px, "rgb_diagonal_px")
    if diagonal <= 0.0:
        raise ValueError("rgb_diagonal_px must be positive")
    write = (
        np.zeros(shape, dtype=np.bool_)
        if write_support is None else _mask(write_support, shape, "write_support")
    )
    read = (
        np.zeros(shape, dtype=np.bool_)
        if read_support is None else _mask(read_support, shape, "read_support")
    )

    native_branch_epe = np.linalg.norm(native[None, ...] - truth, axis=-1)
    action_branch_epe = np.linalg.norm(action[None, ...] - truth, axis=-1)
    native_branch_epe = np.where(branch_valid, native_branch_epe, np.inf)
    action_branch_epe = np.where(branch_valid, action_branch_epe, np.inf)
    native_epe = np.min(native_branch_epe, axis=0)
    action_epe = np.min(action_branch_epe, axis=0)
    native_gt_branch = np.argmin(native_branch_epe, axis=0).astype(np.int16)
    action_gt_branch = np.argmin(action_branch_epe, axis=0).astype(np.int16)
    gain = native_epe - action_epe
    harm = np.maximum(-gain, 0.0)
    benefit = np.maximum(gain, 0.0)
    sidecars = {
        "epe_native_raw_px": np.ascontiguousarray(native_epe.astype(np.float32)),
        "epe_action_raw_px": np.ascontiguousarray(action_epe.astype(np.float32)),
        "gain_raw_px": np.ascontiguousarray(gain.astype(np.float32)),
        "harm_raw_px": np.ascontiguousarray(harm.astype(np.float32)),
        "gt_valid": valid,
        "gt_valid_branch_count": np.ascontiguousarray(
            np.sum(branch_valid, axis=0, dtype=np.int16)
        ),
        "region_id": labels,
        "nearest_GT_branch_native": np.ascontiguousarray(native_gt_branch),
        "nearest_GT_branch_action": np.ascontiguousarray(action_gt_branch),
    }

    regions = []
    for region_id in sorted(int(value) for value in np.unique(labels)):
        region_all = labels == region_id
        region_valid = region_all & valid
        count = int(np.count_nonzero(region_valid))
        common = {
            "region_id": region_id,
            "region_pixels": int(np.count_nonzero(region_all)),
            "valid_pixels": count,
            "write_support_valid_pixels": int(np.count_nonzero(region_valid & write)),
            "read_support_valid_pixels": int(np.count_nonzero(region_valid & read)),
        }
        if count < MIN_VALID_PIXELS_PER_REGION:
            regions.append({
                **common,
                "status": "UNLABELED_GT_SUPPORT_LT16",
                "mean_benefit_mass_raw_px": None,
                "mean_harm_mass_raw_px": None,
                "mean_net_gain_raw_px": None,
                "harmed_pixel_fraction": None,
                "pixel_harm_cvar95_raw_px": None,
                "worst_pixel_harm_raw_px": None,
                "worst_pixel_flat_index": None,
                "severe_region": None,
                "tail_violation": None,
            })
            continue
        region_gain = gain[region_valid]
        region_harm = harm[region_valid]
        flat_indices = np.flatnonzero(region_valid)
        worst_local = int(np.argmax(region_harm))
        harmed_fraction = _mean(
            region_harm > HARMED_PIXEL_THRESHOLD_RAW_PX
        )
        cvar95 = _cvar95_including_zeros(region_harm)
        mean_gain = _mean(region_gain)
        regions.append({
            **common,
            "status": "LABELED",
            "mean_benefit_mass_raw_px": _mean(benefit[region_valid]),
            "mean_harm_mass_raw_px": _mean(region_harm),
            "mean_net_gain_raw_px": mean_gain,
            "harmed_pixel_fraction": harmed_fraction,
            "pixel_harm_cvar95_raw_px": cvar95,
            "worst_pixel_harm_raw_px": float(region_harm[worst_local]),
            "worst_pixel_flat_index": int(flat_indices[worst_local]),
            "severe_region": mean_gain < SEVERE_REGION_GAIN_BELOW_RAW_PX,
            "tail_violation": (
                harmed_fraction > HARMED_PIXEL_FRACTION_UPPER
                or cvar95 > CVAR95_HARM_UPPER_RAW_PX
            ),
        })
    labeled = [row for row in regions if row["status"] == "LABELED"]
    if not labeled:
        raise ValueError("case has no region with at least 16 frozen GT-valid pixels")
    valid_gain = gain[valid]
    valid_harm = harm[valid]
    pixel_harmed_fraction = _mean(
        valid_harm > HARMED_PIXEL_THRESHOLD_RAW_PX
    )
    payload = {
        "schema": SCHEMA,
        "case_id": case_id,
        "component_id": component_id,
        "source_dataset": source_dataset,
        "outer_fold": int(outer_fold),
        "mechanism_stratum": mechanism_stratum,
        "action_id": action_id,
        "action_binding_sha256": action_binding_sha256,
        "fixed_GT_support": {
            "valid_pixels": int(np.count_nonzero(valid)),
            "mask_sha256": array_sha256(valid),
            "branch_valid_mask_sha256": array_sha256(branch_valid),
            "support_rule": "AT_LEAST_ONE_VALID_COMPLETE_GT_VECTOR",
            "action_dependent_denominator_shrinking": False,
        },
        "GT_matching": {
            "branch_count": int(truth.shape[0]),
            "rule": "MINIMUM_EPE_OVER_COMPLETE_GT_VECTORS_PER_PREDICTION",
            "native_and_action_may_select_different_branches": True,
            "per_coordinate_branch_mixing": False,
            "flow_units": "NATIVE_RGB_PIXELS_NO_RESOLUTION_DIVISION",
        },
        "input_sha256": {
            "native_flow": array_sha256(np.asarray(native_flow)),
            "action_flow": array_sha256(np.asarray(action_flow)),
            "ground_truth_flow": array_sha256(np.asarray(ground_truth_flow)),
            "ground_truth_valid_mask": array_sha256(np.asarray(gt_valid_mask)),
            "region_labels": array_sha256(labels),
            "write_support": array_sha256(write),
            "read_support": array_sha256(read),
        },
        "pixel_sidecars": {
            name: {
                "dtype": str(value.dtype), "shape": list(value.shape),
                "array_sha256": array_sha256(value),
            }
            for name, value in sidecars.items()
        },
        "case_metrics": {
            "mean_native_EPE_raw_px": _mean(native_epe[valid]),
            "mean_action_EPE_raw_px": _mean(action_epe[valid]),
            "mean_gain_raw_px": _mean(valid_gain),
            "mean_gain_normalized_px": 1000.0 * _mean(valid_gain) / diagonal,
            "mean_harm_mass_raw_px": _mean(valid_harm),
            "pixel_weighted_harmed_pixel_fraction": pixel_harmed_fraction,
            "maximum_region_mean_harm_mass_raw_px": max(
                row["mean_harm_mass_raw_px"] for row in labeled
            ),
            "maximum_within_region_pixel_CVaR95_harm_raw_px": max(
                row["pixel_harm_cvar95_raw_px"] for row in labeled
            ),
            "maximum_worst_pixel_harm_raw_px": max(
                row["worst_pixel_harm_raw_px"] for row in labeled
            ),
            "severe_region_count": sum(row["severe_region"] for row in labeled),
            "tail_violation_region_count": sum(
                row["tail_violation"] for row in labeled
            ),
            "case_any_severe": any(row["severe_region"] for row in labeled),
            "case_any_tail_violation": any(
                row["tail_violation"] for row in labeled
            ),
            "labeled_region_count": len(labeled),
            "unlabeled_region_count": len(regions) - len(labeled),
        },
        "regions": regions,
        "thresholds": {
            "minimum_valid_pixels_per_region": MIN_VALID_PIXELS_PER_REGION,
            "harmed_pixel_threshold_raw_px": HARMED_PIXEL_THRESHOLD_RAW_PX,
            "harmed_pixel_fraction_upper": HARMED_PIXEL_FRACTION_UPPER,
            "pixel_harm_cvar95_raw_px_upper": CVAR95_HARM_UPPER_RAW_PX,
            "severe_region_gain_below_raw_px": SEVERE_REGION_GAIN_BELOW_RAW_PX,
        },
        "GT_read": True,
        "scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
    }
    return sidecars, {**payload, "observation_sha256": canonical_sha256(payload)}


__all__ = [
    "CVAR95_HARM_UPPER_RAW_PX",
    "HARMED_PIXEL_FRACTION_UPPER",
    "HARMED_PIXEL_THRESHOLD_RAW_PX",
    "MIN_VALID_PIXELS_PER_REGION",
    "SCHEMA",
    "SEVERE_REGION_GAIN_BELOW_RAW_PX",
    "array_sha256",
    "canonical_sha256",
    "score_action_case_v1",
]
