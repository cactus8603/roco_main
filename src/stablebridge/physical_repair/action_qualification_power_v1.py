"""Joint development power simulation for the frozen seven-entry action bank.

This is a planning engine, not a scientific qualification engine.  It uses the
40-component Spring/KITTI development joint ledger to simulate the fixed E263
allocation while preserving whole-component correlation and typed missingness.
The six task and fourteen mechanism tests share one Holm family.  Fourteen
pixel-tail bounds use a shared empirical max-T critical value.
"""
from __future__ import annotations

import hashlib
import json
import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from .action_qualification_joint_ledger_v1 import (
    ACTION_ORDER,
    ACTION_TO_STRATA,
    EXPECTED_SOURCE_COMPONENTS,
    OBSERVED,
    SCHEMA as JOINT_ROW_SCHEMA,
    STRATUM_ORDER,
    canonical_sha256,
)


SCHEMA = "stablebridge-action-qualification-joint-power/v1"
REPEATS = 10_000
SEED = 20_261_004
FAMILYWISE_ALPHA = 0.05
TARGET_POWER = 0.80
TASK_POINT_GATE = 0.01
MECHANISM_POINT_GATE = 0.05
HARMED_FRACTION_UPPER = 0.05
CVAR95_UPPER_RAW_PX = 0.25
PRIMARY_TASK_ALTERNATIVE = 0.02
MECHANISM_PRIMARY_ALTERNATIVE = 0.10
MECHANISM_SPECIFICITY_ALTERNATIVE = 0.05
HARMED_FRACTION_ALTERNATIVE = 0.025
CVAR95_ALTERNATIVE_RAW_PX = 0.125

# The fixed fresh roster has Spring 4/4/4/3/3 and KITTI one per fold.
TARGET_FOLDS_BY_SOURCE = {
    "spring": (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4),
    "kitti": (0, 1, 2, 3, 4),
}


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_component_rows(joint: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(joint, Mapping):
        raise ValueError("joint ledger must be a mapping")
    manifest = joint.get("manifest")
    rows = joint.get("components")
    if not isinstance(manifest, Mapping) or not isinstance(rows, Sequence):
        raise ValueError("joint ledger lacks manifest or components")
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != canonical_sha256(manifest_payload):
        raise ValueError("joint-ledger manifest semantic hash drift")
    if (
        manifest.get("component_count") != 40
        or manifest.get("case_count") != 280
        or manifest.get("scientific_n") != 40
        or manifest.get("source_component_counts") != EXPECTED_SOURCE_COMPONENTS
    ):
        raise ValueError("joint-ledger frozen cohort cardinality drift")

    result = []
    identities: set[tuple[str, str]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping) or raw.get("schema") != JOINT_ROW_SCHEMA:
            raise ValueError("invalid component joint row")
        payload = {
            key: value for key, value in raw.items()
            if key != "component_observation_sha256"
        }
        if raw.get("component_observation_sha256") != canonical_sha256(payload):
            raise ValueError("component observation semantic hash drift")
        source = raw.get("source_dataset")
        component = raw.get("component_id")
        if source not in EXPECTED_SOURCE_COMPONENTS or not isinstance(component, str):
            raise ValueError("component identity drift")
        identity = (source, component)
        if identity in identities:
            raise ValueError("duplicate component row")
        identities.add(identity)
        if set(raw.get("task", {})) != set(ACTION_ORDER):
            raise ValueError("component task vector drift")
        if set(raw.get("action_gates", {})) != set(ACTION_ORDER):
            raise ValueError("component action-gate vector drift")
        for field in ("availability", "mechanism", "tails"):
            if set(raw.get(field, {})) != set(STRATUM_ORDER):
                raise ValueError(f"component {field} vector drift")
        result.append(dict(raw))
    counts = {
        source: sum(row["source_dataset"] == source for row in result)
        for source in EXPECTED_SOURCE_COMPONENTS
    }
    if counts != EXPECTED_SOURCE_COMPONENTS:
        raise ValueError("component source counts drift")
    hashes = [row["component_observation_sha256"] for row in result]
    if manifest.get("component_observation_sha256") != hashes:
        raise ValueError("manifest component ordering/hash binding drift")
    return result


def _finite_or_nan(value: object, name: str) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric or null")
    return result


def _metric_layout() -> tuple[list[str], np.ndarray]:
    names = [f"task:{action}" for action in ACTION_ORDER]
    alternatives = [PRIMARY_TASK_ALTERNATIVE] * len(ACTION_ORDER)
    for stratum in STRATUM_ORDER:
        names.extend((
            f"mechanism:{stratum}:primary",
            f"mechanism:{stratum}:specificity",
        ))
        alternatives.extend((
            MECHANISM_PRIMARY_ALTERNATIVE,
            MECHANISM_SPECIFICITY_ALTERNATIVE,
        ))
    for stratum in STRATUM_ORDER:
        names.extend((
            f"tail:{stratum}:harmed_fraction",
            f"tail:{stratum}:cvar95",
        ))
        alternatives.extend((
            HARMED_FRACTION_ALTERNATIVE,
            CVAR95_ALTERNATIVE_RAW_PX,
        ))
    return names, np.asarray(alternatives, dtype=np.float64)


METRIC_NAMES, PLANNING_ALTERNATIVES = _metric_layout()
METRIC_INDEX = {name: index for index, name in enumerate(METRIC_NAMES)}
HYPOTHESIS_NAMES = (
    tuple(f"task:{action}" for action in ACTION_ORDER)
    + tuple(
        f"mechanism:{stratum}:{kind}"
        for stratum in STRATUM_ORDER
        for kind in ("primary", "specificity")
    )
)
HYPOTHESIS_INDEX = tuple(METRIC_INDEX[name] for name in HYPOTHESIS_NAMES)
TAIL_NAMES = tuple(
    f"tail:{stratum}:{kind}"
    for stratum in STRATUM_ORDER
    for kind in ("harmed_fraction", "cvar95")
)
TAIL_INDEX = tuple(METRIC_INDEX[name] for name in TAIL_NAMES)


def _extract(rows: Sequence[Mapping[str, Any]], source: str) -> dict[str, Any]:
    selected = [row for row in rows if row["source_dataset"] == source]
    metrics = np.full((len(selected), len(METRIC_NAMES)), np.nan, dtype=np.float64)
    availability = np.zeros((len(selected), len(STRATUM_ORDER)), dtype=np.bool_)
    severe = np.zeros((len(selected), len(STRATUM_ORDER)), dtype=np.bool_)
    cost = np.zeros((len(selected), len(ACTION_ORDER)), dtype=np.bool_)
    children = np.zeros_like(cost)
    for row_index, row in enumerate(selected):
        for action_index, action in enumerate(ACTION_ORDER):
            task = row["task"][action]
            if task.get("status") == OBSERVED:
                metrics[row_index, METRIC_INDEX[f"task:{action}"]] = (
                    _finite_or_nan(task.get("normalized_gain"), "normalized_gain")
                )
            gates = row["action_gates"][action]
            if not all(isinstance(gates.get(key), bool) for key in (
                "all_executed", "cost_gate_pass", "child_observables_complete",
            )):
                raise ValueError("action gate must be boolean")
            cost[row_index, action_index] = gates["cost_gate_pass"]
            children[row_index, action_index] = gates["child_observables_complete"]
        for stratum_index, stratum in enumerate(STRATUM_ORDER):
            availability[row_index, stratum_index] = (
                row["availability"][stratum].get("status") == OBSERVED
            )
            mechanism = row["mechanism"][stratum]
            if mechanism.get("status") == OBSERVED:
                for kind in ("primary", "specificity"):
                    metrics[
                        row_index, METRIC_INDEX[f"mechanism:{stratum}:{kind}"]
                    ] = _finite_or_nan(mechanism.get(kind), kind)
            tail = row["tails"][stratum]
            severe_value = tail.get("severe_event")
            if severe_value is not None and not isinstance(severe_value, bool):
                raise ValueError("tail severe_event must be boolean or null")
            if severe_value is True:
                severe[row_index, stratum_index] = True
            if tail.get("status") == OBSERVED:
                metrics[
                    row_index, METRIC_INDEX[f"tail:{stratum}:harmed_fraction"]
                ] = _finite_or_nan(
                    tail.get("harmed_pixel_fraction"), "harmed_pixel_fraction",
                )
                metrics[
                    row_index, METRIC_INDEX[f"tail:{stratum}:cvar95"]
                ] = _finite_or_nan(
                    tail.get("pixel_harm_cvar95_raw_px"),
                    "pixel_harm_cvar95_raw_px",
                )
                if not isinstance(severe_value, bool):
                    raise ValueError("observed tail needs boolean severe_event")
            elif severe_value is not None and severe_value is not True:
                raise ValueError(
                    "missing tail may carry only a nonfinite-action severe event"
                )
    return {
        "metrics": metrics,
        "availability": availability,
        "severe": severe,
        "cost": cost,
        "children": children,
    }


def _shift_to_alternatives(values: np.ndarray) -> np.ndarray:
    shifted = values.copy()
    for column, alternative in enumerate(PLANNING_ALTERNATIVES):
        observed = np.isfinite(values[:, column])
        if not np.any(observed):
            continue
        residual = values[observed, column] - np.mean(values[observed, column])
        shifted[observed, column] = alternative + residual
    for name in TAIL_NAMES:
        index = METRIC_INDEX[name]
        if name.endswith("harmed_fraction"):
            shifted[:, index] = np.clip(shifted[:, index], 0.0, 1.0)
        else:
            shifted[:, index] = np.maximum(shifted[:, index], 0.0)
    return shifted


def _source_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.isfinite(values)
    count = np.sum(observed, axis=1)
    total = np.nansum(values, axis=1)
    mean = np.divide(
        total, count, out=np.full_like(total, np.nan, dtype=np.float64),
        where=count > 0,
    )
    centered = np.where(observed, values - mean[:, None, :], 0.0)
    sumsq = np.sum(centered * centered, axis=1)
    variance = np.divide(
        sumsq, count - 1,
        out=np.full_like(sumsq, np.nan, dtype=np.float64),
        where=count > 1,
    )
    return mean, variance, count


def _holm(p_values: np.ndarray) -> np.ndarray:
    repeats, hypotheses = p_values.shape
    result = np.zeros_like(p_values, dtype=np.bool_)
    for repeat in range(repeats):
        order = np.argsort(p_values[repeat], kind="stable")
        rejecting = True
        for rank, index in enumerate(order):
            threshold = FAMILYWISE_ALPHA / (hypotheses - rank)
            passed = rejecting and p_values[repeat, index] <= threshold
            result[repeat, index] = passed
            if not passed:
                rejecting = False
    return result


def _availability_gate(
    selected: Mapping[str, np.ndarray],
) -> np.ndarray:
    repeats = next(iter(selected.values())).shape[0]
    combined = np.concatenate([selected["spring"], selected["kitti"]], axis=1)
    enough_total = np.sum(combined, axis=1) >= 20
    enough_source = np.ones((repeats, len(STRATUM_ORDER)), dtype=np.bool_)
    for source in ("spring", "kitti"):
        enough_source &= np.sum(selected[source], axis=1) >= 2
    fold_arrays = []
    folds = TARGET_FOLDS_BY_SOURCE["spring"] + TARGET_FOLDS_BY_SOURCE["kitti"]
    for fold in range(5):
        mask = np.asarray([value == fold for value in folds], dtype=np.bool_)
        fold_arrays.append(np.sum(combined[:, mask, :], axis=1) >= 4)
    enough_fold = np.logical_and.reduce(fold_arrays)
    return enough_total & enough_source & enough_fold


def evaluate_joint_power_v1(
    joint_ledger: Mapping[str, Any], *, protocol_sha256: str,
    repeats: int = REPEATS, seed: int = SEED,
) -> dict[str, Any]:
    """Run the frozen primary joint power simulation.

    The development residuals are centered within source and shifted to the
    pre-frozen planning alternatives.  Whole rows are resampled into the exact
    Spring18/KITTI5 target slots.  This grants only power-admission evidence;
    fresh action outcomes remain required for every scientific decision.
    """
    protocol_sha256 = _sha(protocol_sha256, "protocol_sha256")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < REPEATS:
        raise ValueError("joint power simulation requires at least 10,000 repeats")
    if seed != SEED:
        raise ValueError("joint power simulation seed is frozen at 20261004")
    rows = _validate_component_rows(joint_ledger)
    extracted = {source: _extract(rows, source) for source in ("spring", "kitti")}
    shifted = {
        source: _shift_to_alternatives(extracted[source]["metrics"])
        for source in extracted
    }
    generator = np.random.default_rng(seed)
    indices = {
        source: generator.integers(
            0, shifted[source].shape[0],
            size=(repeats, len(TARGET_FOLDS_BY_SOURCE[source])),
        )
        for source in ("spring", "kitti")
    }
    selected_metrics = {
        source: shifted[source][indices[source]] for source in shifted
    }
    source_stats = {
        source: _source_statistics(selected_metrics[source])
        for source in selected_metrics
    }
    spring_mean, spring_var, spring_n = source_stats["spring"]
    kitti_mean, kitti_var, kitti_n = source_stats["kitti"]
    estimable = (spring_n >= 2) & (kitti_n >= 2)
    point = 0.5 * (spring_mean + kitti_mean)
    standard_error = np.sqrt(0.25 * (
        spring_var / spring_n + kitti_var / kitti_n
    ))
    estimable &= np.isfinite(point) & np.isfinite(standard_error)
    z = np.divide(
        point, standard_error,
        out=np.full_like(point, -math.inf),
        where=estimable & (standard_error > 0.0),
    )
    zero_se_positive = estimable & (standard_error == 0.0) & (point > 0.0)
    p_values_all = np.vectorize(
        lambda value: 1.0 - NormalDist().cdf(float(value)), otypes=[float],
    )(z)
    p_values_all[zero_se_positive] = 1.0 / (repeats + 1.0)
    p_values_all[~estimable] = 1.0
    lower = point - NormalDist().inv_cdf(0.975) * standard_error
    lower[~estimable] = np.nan

    hypothesis_p = p_values_all[:, HYPOTHESIS_INDEX]
    holm = _holm(hypothesis_p)
    hypothesis_position = {
        name: index for index, name in enumerate(HYPOTHESIS_NAMES)
    }

    # Shared empirical max-T upper critical value for all fourteen tail metrics.
    tail_point = point[:, TAIL_INDEX]
    tail_se = standard_error[:, TAIL_INDEX]
    tail_alt = PLANNING_ALTERNATIVES[np.asarray(TAIL_INDEX)]
    standardized = np.divide(
        tail_point - tail_alt,
        tail_se,
        out=np.zeros_like(tail_point),
        where=np.isfinite(tail_se) & (tail_se > 0.0),
    )
    # A missing tail fails its own action below; it must not make the shared
    # critical value infinite for every other fully observed action.
    standardized[~estimable[:, TAIL_INDEX]] = math.nan
    with np.errstate(all="ignore"):
        max_t = np.nanmax(standardized, axis=1)
    finite_max_t = max_t[np.isfinite(max_t)]
    if finite_max_t.size == 0:
        raise ValueError("no joint tail replicate is estimable")
    max_t_critical = float(np.quantile(
        finite_max_t, 0.95, method="higher",
    ))
    tail_ucb = tail_point + max_t_critical * tail_se
    tail_estimable = estimable[:, TAIL_INDEX]

    selected_availability = {
        source: extracted[source]["availability"][indices[source]]
        for source in extracted
    }
    availability_pass = _availability_gate(selected_availability)
    selected_severe = {
        source: extracted[source]["severe"][indices[source]]
        for source in extracted
    }
    severe_any = np.any(np.concatenate(
        [selected_severe["spring"], selected_severe["kitti"]], axis=1,
    ), axis=1)
    selected_cost = {
        source: extracted[source]["cost"][indices[source]]
        for source in extracted
    }
    selected_children = {
        source: extracted[source]["children"][indices[source]]
        for source in extracted
    }
    cost_pass = np.all(np.concatenate(
        [selected_cost["spring"], selected_cost["kitti"]], axis=1,
    ), axis=1)
    children_pass = np.all(np.concatenate(
        [selected_children["spring"], selected_children["kitti"]], axis=1,
    ), axis=1)

    action_repeat_pass: dict[str, np.ndarray] = {}
    gate_rates: dict[str, Any] = {}
    for action_index, action in enumerate(ACTION_ORDER):
        task_name = f"task:{action}"
        task_metric_index = METRIC_INDEX[task_name]
        task_holm_index = hypothesis_position[task_name]
        task_pass = (
            holm[:, task_holm_index]
            & estimable[:, task_metric_index]
            & (point[:, task_metric_index] >= TASK_POINT_GATE)
            & (lower[:, task_metric_index] > 0.0)
        )
        mechanism_pass = np.ones(repeats, dtype=np.bool_)
        tail_pass = np.ones(repeats, dtype=np.bool_)
        availability_action = np.ones(repeats, dtype=np.bool_)
        severe_pass = np.ones(repeats, dtype=np.bool_)
        for stratum in ACTION_TO_STRATA[action]:
            stratum_index = STRATUM_ORDER.index(stratum)
            availability_action &= availability_pass[:, stratum_index]
            severe_pass &= ~severe_any[:, stratum_index]
            for kind in ("primary", "specificity"):
                name = f"mechanism:{stratum}:{kind}"
                metric_index = METRIC_INDEX[name]
                mechanism_pass &= (
                    holm[:, hypothesis_position[name]]
                    & estimable[:, metric_index]
                    & (lower[:, metric_index] > 0.0)
                )
                if kind == "primary":
                    mechanism_pass &= point[:, metric_index] >= MECHANISM_POINT_GATE
            harm_position = TAIL_NAMES.index(f"tail:{stratum}:harmed_fraction")
            cvar_position = TAIL_NAMES.index(f"tail:{stratum}:cvar95")
            tail_pass &= (
                tail_estimable[:, harm_position]
                & tail_estimable[:, cvar_position]
                & (tail_ucb[:, harm_position] <= HARMED_FRACTION_UPPER)
                & (tail_ucb[:, cvar_position] <= CVAR95_UPPER_RAW_PX)
            )
        gates = {
            "task": task_pass,
            "mechanism": mechanism_pass,
            "tail": tail_pass,
            "availability": availability_action,
            "severe_zero": severe_pass,
            "cost": cost_pass[:, action_index],
            "children": children_pass[:, action_index],
        }
        action_repeat_pass[action] = np.logical_and.reduce(tuple(gates.values()))
        gate_rates[action] = {
            name: float(np.mean(value)) for name, value in gates.items()
        }

    probabilities = {
        action: float(np.mean(action_repeat_pass[action]))
        for action in ACTION_ORDER
    }
    admitted = [
        action for action in ACTION_ORDER if probabilities[action] >= TARGET_POWER
    ]
    all_six = np.logical_and.reduce(tuple(action_repeat_pass.values()))
    input_hash = canonical_sha256({
        "manifest_sha256": joint_ledger["manifest"]["manifest_sha256"],
        "component_observation_sha256": joint_ledger["manifest"][
            "component_observation_sha256"
        ],
    })
    payload = {
        "schema": SCHEMA,
        "status": "COMPLETE_DEVELOPMENT_POWER_SIMULATION_NOT_FRESH_SCIENCE",
        "input_joint_ledger_sha256": input_hash,
        "protocol_sha256": protocol_sha256,
        "simulation": {
            "repeats": repeats,
            "seed": seed,
            "resampling_unit": "whole_component_joint_row",
            "source_stratified_shared_stream": True,
            "target_source_counts": {"spring": 18, "kitti": 5},
            "target_fold_counts": {str(fold): count for fold, count in enumerate((5, 5, 5, 4, 4))},
            "development_residual_centering": "within_source",
            "planning_alternatives": {
                "task_normalized_gain": PRIMARY_TASK_ALTERNATIVE,
                "mechanism_primary": MECHANISM_PRIMARY_ALTERNATIVE,
                "mechanism_specificity": MECHANISM_SPECIFICITY_ALTERNATIVE,
                "harmed_fraction": HARMED_FRACTION_ALTERNATIVE,
                "CVaR95_raw_px": CVAR95_ALTERNATIVE_RAW_PX,
                "severe_event": 0,
            },
            "test_approximation": "source_equal_normal_Wald_inside_joint_Monte_Carlo",
            "global_family": "six_task_plus_fourteen_mechanism_Holm_FWER_0p05",
            "tail_family": "fourteen_metric_shared_empirical_maxT_UCB",
            "tail_maxT_critical": max_t_critical,
        },
        "per_action": {
            action: {
                "adoption_probability": probabilities[action],
                "meets_0p80_power_admission": probabilities[action] >= TARGET_POWER,
                "gate_pass_rates": gate_rates[action],
            }
            for action in ACTION_ORDER
        },
        "admitted_action_ids_for_fixed_E263_cohort": admitted,
        "all_six_pass_probability_report_only": float(np.mean(all_six)),
        "all_six_probability_is_not_an_all_or_nothing_gate": True,
        "power_admission_complete": True,
        "fresh_scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
    }
    return {**payload, "power_result_sha256": canonical_sha256(payload)}


__all__ = [
    "FAMILYWISE_ALPHA", "REPEATS", "SCHEMA", "SEED", "TARGET_POWER",
    "evaluate_joint_power_v1",
]
