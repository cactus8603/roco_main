"""Pre-outcome final science decision for the frozen seven-entry action bank.

This evaluator consumes only a hash-bound Spring18/KITTI5 fresh component
ledger plus the independently sealed development power-admission result.  It
implements the pre-registered six task + fourteen mechanism Holm family,
source-equal shared bootstrap, fourteen-tail max-T upper bounds, availability,
cost, child-observable, severe-harm, chronology, and replay gates.

The native entry remains a reference rather than a hypothesis.  A successful
action is qualified only for a later selector build; this module grants no
selector or production authority.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .action_qualification_fresh_joint_ledger_v1 import (
    EXPECTED_FOLDS_BY_SOURCE,
    EXPECTED_SOURCE_COMPONENTS,
    MANIFEST_SCHEMA,
    SCHEMA as FRESH_ROW_SCHEMA,
)
from .action_qualification_joint_ledger_v1 import (
    ACTION_ORDER,
    ACTION_TO_STRATA,
    OBSERVED,
    STRATUM_ORDER,
    canonical_sha256,
)
from .action_qualification_power_v1 import SCHEMA as POWER_SCHEMA


SCHEMA = "stablebridge-action-qualification-fresh-science-decision/v1"
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20_261_004
FAMILYWISE_ALPHA = 0.05
TASK_POINT_GATE = 0.01
MECHANISM_PRIMARY_POINT_GATE = 0.05
HARMED_FRACTION_UPPER = 0.05
CVAR95_UPPER_RAW_PX = 0.25
MINIMUM_COMPONENTS = 20
MINIMUM_PER_SOURCE = 2
MINIMUM_PER_FOLD = 4


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_or_nan(value: object, name: str) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric or null")
    return result


def _validate_fresh_ledger(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("fresh joint ledger must be a mapping")
    manifest = value.get("manifest")
    rows = value.get("components")
    if not isinstance(manifest, Mapping) or not isinstance(rows, Sequence):
        raise ValueError("fresh joint ledger lacks manifest or components")
    manifest_payload = {
        key: item for key, item in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("manifest_sha256") != canonical_sha256(manifest_payload)
        or manifest.get("component_count") != 23
        or manifest.get("case_count") != 161
        or manifest.get("scientific_n") != 23
        or manifest.get("source_component_counts") != EXPECTED_SOURCE_COMPONENTS
        or manifest.get("fold_counts_by_source") != EXPECTED_FOLDS_BY_SOURCE
    ):
        raise ValueError("fresh joint-ledger manifest drift")

    result: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    binding = manifest.get("evidence_bindings")
    for raw in rows:
        if not isinstance(raw, Mapping) or raw.get("schema") != FRESH_ROW_SCHEMA:
            raise ValueError("invalid fresh component row")
        payload = {
            key: item for key, item in raw.items()
            if key != "component_observation_sha256"
        }
        if raw.get("component_observation_sha256") != canonical_sha256(payload):
            raise ValueError("fresh component semantic hash drift")
        source = raw.get("source_dataset")
        component = raw.get("component_id")
        fold = raw.get("outer_fold")
        if (
            source not in EXPECTED_SOURCE_COMPONENTS
            or not isinstance(component, str)
            or not isinstance(fold, int)
            or fold not in range(5)
            or raw.get("evidence_bindings") != binding
        ):
            raise ValueError("fresh component identity/binding drift")
        identity = (source, component)
        if identity in identities:
            raise ValueError("duplicate fresh component")
        identities.add(identity)
        if set(raw.get("task", {})) != set(ACTION_ORDER):
            raise ValueError("fresh task vector drift")
        if set(raw.get("action_gates", {})) != set(ACTION_ORDER):
            raise ValueError("fresh action-gate vector drift")
        for field in ("availability", "mechanism", "tails"):
            if set(raw.get(field, {})) != set(STRATUM_ORDER):
                raise ValueError(f"fresh {field} vector drift")
        result.append(dict(raw))
    if len(result) != 23:
        raise ValueError("fresh ledger must contain 23 component rows")
    if manifest.get("component_observation_sha256") != [
        row["component_observation_sha256"] for row in result
    ]:
        raise ValueError("fresh manifest component ordering/hash drift")
    return result


def _validate_power(value: Mapping[str, Any]) -> dict[str, bool]:
    if not isinstance(value, Mapping) or value.get("schema") != POWER_SCHEMA:
        raise ValueError("development power-admission schema drift")
    payload = {
        key: item for key, item in value.items() if key != "power_result_sha256"
    }
    if value.get("power_result_sha256") != canonical_sha256(payload):
        raise ValueError("development power-admission semantic hash drift")
    if value.get("power_admission_complete") is not True:
        raise ValueError("development power admission is incomplete")
    per_action = value.get("per_action")
    if not isinstance(per_action, Mapping) or set(per_action) != set(ACTION_ORDER):
        raise ValueError("development power per-action vector drift")
    result = {}
    for action in ACTION_ORDER:
        admitted = per_action[action].get("meets_0p80_power_admission")
        probability = per_action[action].get("adoption_probability")
        if (
            not isinstance(admitted, bool)
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
            or admitted != (float(probability) >= 0.8)
        ):
            raise ValueError("development power action decision drift")
        result[action] = admitted
    return result


def _metric_layout() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    task = tuple(f"task:{action}" for action in ACTION_ORDER)
    mechanism = tuple(
        f"mechanism:{stratum}:{kind}"
        for stratum in STRATUM_ORDER for kind in ("primary", "specificity")
    )
    tails = tuple(
        f"tail:{stratum}:{kind}"
        for stratum in STRATUM_ORDER for kind in ("harmed_fraction", "cvar95")
    )
    return task, mechanism, tails


TASK_NAMES, MECHANISM_NAMES, TAIL_NAMES = _metric_layout()
HYPOTHESIS_NAMES = TASK_NAMES + MECHANISM_NAMES
METRIC_NAMES = HYPOTHESIS_NAMES + TAIL_NAMES
METRIC_INDEX = {name: index for index, name in enumerate(METRIC_NAMES)}


def _extract(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    metrics = np.full((len(rows), len(METRIC_NAMES)), np.nan, dtype=np.float64)
    availability = np.zeros((len(rows), len(STRATUM_ORDER)), dtype=np.bool_)
    for row_index, row in enumerate(rows):
        for action in ACTION_ORDER:
            value = row["task"][action]
            if value.get("status") == OBSERVED:
                metrics[row_index, METRIC_INDEX[f"task:{action}"]] = (
                    _finite_or_nan(value.get("normalized_gain"), "normalized_gain")
                )
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
                raise ValueError("fresh tail severe_event must be boolean or null")
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
                    raise ValueError("observed fresh tail needs severe_event")
            elif severe_value is not None and severe_value is not True:
                raise ValueError(
                    "missing fresh tail may carry only a nonfinite-action severe event"
                )
    return metrics, availability


def _coverage(
    observed: np.ndarray, rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_source = {
        source: int(sum(
            bool(observed[index]) and row["source_dataset"] == source
            for index, row in enumerate(rows)
        ))
        for source in EXPECTED_SOURCE_COMPONENTS
    }
    by_fold = {
        str(fold): int(sum(
            bool(observed[index]) and row["outer_fold"] == fold
            for index, row in enumerate(rows)
        ))
        for fold in range(5)
    }
    checks = {
        "minimum_components": int(np.count_nonzero(observed)) >= MINIMUM_COMPONENTS,
        "minimum_per_source": min(by_source.values()) >= MINIMUM_PER_SOURCE,
        "minimum_per_fold": min(by_fold.values()) >= MINIMUM_PER_FOLD,
    }
    return {
        "observed_components": int(np.count_nonzero(observed)),
        "components_by_source": by_source,
        "components_by_fold": by_fold,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _bootstrap_metric(
    values: np.ndarray, rows: Sequence[Mapping[str, Any]],
    uniforms: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    source_points = []
    source_boot = []
    source_counts = {}
    for source in ("spring", "kitti"):
        selected = np.asarray([
            values[index] for index, row in enumerate(rows)
            if row["source_dataset"] == source and math.isfinite(values[index])
        ], dtype=np.float64)
        source_counts[source] = int(selected.size)
        if selected.size < MINIMUM_PER_SOURCE:
            return {
                "estimable": False,
                "source_observed_components": source_counts,
                "point": None, "ci95_lower": None, "ci95_upper": None,
                "p_value": 1.0, "bootstrap_standard_error": None,
            }
        indices = np.minimum(
            (uniforms[source][:, :selected.size] * selected.size).astype(np.int64),
            selected.size - 1,
        )
        source_points.append(float(np.mean(selected, dtype=np.float64)))
        source_boot.append(np.mean(selected[indices], axis=1, dtype=np.float64))
    point = float(np.mean(source_points, dtype=np.float64))
    bootstrap = np.mean(np.stack(source_boot, axis=1), axis=1, dtype=np.float64)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975), method="linear")
    centered = bootstrap - point
    p_value = float(
        (1 + np.count_nonzero(centered <= -point))
        / (BOOTSTRAP_REPEATS + 1)
    )
    standard_error = float(np.std(bootstrap, ddof=1))
    return {
        "estimable": True,
        "source_observed_components": source_counts,
        "point": point,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "p_value": p_value,
        "bootstrap_standard_error": standard_error,
        "_bootstrap": bootstrap,
    }


def _holm(results: dict[str, dict[str, Any]]) -> None:
    order = sorted(HYPOTHESIS_NAMES, key=lambda name: (results[name]["p_value"], name))
    rejecting = True
    total = len(order)
    for rank, name in enumerate(order, start=1):
        threshold = FAMILYWISE_ALPHA / (total - rank + 1)
        passed = bool(rejecting and results[name]["p_value"] <= threshold)
        if not passed:
            rejecting = False
        results[name]["holm_rank"] = rank
        results[name]["holm_threshold"] = threshold
        results[name]["holm_reject"] = passed


def evaluate_fresh_action_science_v1(
    fresh_joint_ledger: Mapping[str, Any],
    *, development_power_admission: Mapping[str, Any],
    science_protocol_sha256: str,
) -> dict[str, Any]:
    """Issue per-action fresh decisions under the frozen fail-closed rules."""
    science_protocol_sha256 = _sha(
        science_protocol_sha256, "science_protocol_sha256",
    )
    rows = _validate_fresh_ledger(fresh_joint_ledger)
    power = _validate_power(development_power_admission)
    metrics, availability = _extract(rows)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    uniforms = {
        source: generator.random((BOOTSTRAP_REPEATS, count))
        for source, count in EXPECTED_SOURCE_COMPONENTS.items()
    }

    inference: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    for name, index in METRIC_INDEX.items():
        observed = np.isfinite(metrics[:, index])
        coverage[name] = _coverage(observed, rows)
        inference[name] = _bootstrap_metric(
            metrics[:, index], rows, uniforms,
        )
    _holm(inference)

    tail_bootstrap = []
    for name in TAIL_NAMES:
        row = inference[name]
        if not row["estimable"]:
            tail_bootstrap.append(np.full(BOOTSTRAP_REPEATS, np.nan))
            continue
        standard_error = row["bootstrap_standard_error"]
        if standard_error == 0.0:
            tail_bootstrap.append(np.zeros(BOOTSTRAP_REPEATS, dtype=np.float64))
        else:
            tail_bootstrap.append(
                (row["_bootstrap"] - row["point"]) / standard_error
            )
    standardized = np.stack(tail_bootstrap, axis=1)
    # An unestimable tail fails only its owning action through the coverage
    # gate.  It must not make the shared critical value infinite or prevent
    # fully observed actions from receiving a decision.
    with np.errstate(all="ignore"):
        max_t = np.nanmax(standardized, axis=1)
    if not np.any(np.isfinite(max_t)):
        raise ValueError("no fresh tail metric is estimable")
    max_t = max_t[np.isfinite(max_t)]
    max_t_critical = float(np.quantile(max_t, 0.95, method="higher"))
    for name in TAIL_NAMES:
        row = inference[name]
        row["simultaneous_upper"] = (
            float(row["point"] + max_t_critical * row["bootstrap_standard_error"])
            if row["estimable"] else None
        )

    # Internal bootstrap vectors are never serialized into the receipt.
    for row in inference.values():
        row.pop("_bootstrap", None)

    availability_coverage = {
        stratum: _coverage(availability[:, index], rows)
        for index, stratum in enumerate(STRATUM_ORDER)
    }
    severe_counts = {
        stratum: sum(
            row["tails"][stratum].get("severe_event") is True for row in rows
        )
        for stratum in STRATUM_ORDER
    }

    decisions: dict[str, Any] = {}
    for action in ACTION_ORDER:
        task_name = f"task:{action}"
        task = inference[task_name]
        task_gate = bool(
            task["estimable"]
            and coverage[task_name]["pass"]
            and task["holm_reject"]
            and task["point"] >= TASK_POINT_GATE
            and task["ci95_lower"] > 0.0
        )
        mechanism_gate = True
        tail_gate = True
        availability_gate = True
        severe_count = 0
        mechanism_coverage_gate = True
        tail_coverage_gate = True
        for stratum in ACTION_TO_STRATA[action]:
            availability_gate &= availability_coverage[stratum]["pass"]
            severe_count += severe_counts[stratum]
            for kind in ("primary", "specificity"):
                name = f"mechanism:{stratum}:{kind}"
                result = inference[name]
                mechanism_coverage_gate &= coverage[name]["pass"]
                mechanism_gate &= bool(
                    result["estimable"]
                    and coverage[name]["pass"]
                    and result["holm_reject"]
                    and result["ci95_lower"] > 0.0
                    and (kind != "primary" or result["point"] >= MECHANISM_PRIMARY_POINT_GATE)
                )
            harmed = f"tail:{stratum}:harmed_fraction"
            cvar = f"tail:{stratum}:cvar95"
            tail_coverage_gate &= coverage[harmed]["pass"] and coverage[cvar]["pass"]
            tail_gate &= bool(
                inference[harmed]["estimable"]
                and inference[cvar]["estimable"]
                and coverage[harmed]["pass"]
                and coverage[cvar]["pass"]
                and inference[harmed]["simultaneous_upper"] <= HARMED_FRACTION_UPPER
                and inference[cvar]["simultaneous_upper"] <= CVAR95_UPPER_RAW_PX
            )
        # The component reducer has already treated a named nonexecution as
        # N/A while intersecting cost/child evidence over every *observed*
        # owned stratum.  Do not bypass a whole component merely because one
        # isotropic internal stratum is missing: that would hide a real cost
        # or child failure on the other, executed stratum.
        cost_gate = all(
            row["action_gates"][action]["cost_gate_pass"]
            for row in rows
        )
        child_gate = all(
            row["action_gates"][action]["child_observables_complete"]
            for row in rows
        )
        execution_gate = all(
            row["action_gates"][action]["all_executed"] for row in rows
        )
        gates = {
            "development_power_admitted": power[action],
            "complete_available_or_named_typed_missing_accounting": True,
            "availability": availability_gate,
            "task_coverage": coverage[task_name]["pass"],
            "mechanism_coverage": mechanism_coverage_gate,
            "tail_coverage": tail_coverage_gate,
            "global_Holm_task": task_gate,
            "global_Holm_mechanism": mechanism_gate,
            "simultaneous_pixel_tail": tail_gate,
            "zero_severe_events": severe_count == 0,
            "prospective_cost": cost_gate,
            "local_child_observables": child_gate,
        }
        qualifies = all(gates.values())
        # Negative decisions have their own evidence requirements.  A named
        # nonfinite action is an observed discrete severe event even though
        # its numeric task/tail values are unestimable; several such failures
        # must not paradoxically turn a disable decision into INCONCLUSIVE by
        # lowering numeric task coverage.  Likewise a conclusive efficacy
        # upper bound does not depend on an unrelated mechanism field being
        # estimable.
        adequate_for_severe_decision = bool(
            power[action] and availability_gate
        )
        adequate_for_efficacy_decision = bool(
            power[action]
            and availability_gate
            and coverage[task_name]["pass"]
            and task["estimable"]
        )
        adequate_for_redesign_decision = bool(
            adequate_for_efficacy_decision
            and mechanism_coverage_gate
            and tail_coverage_gate
        )
        disable_reason = None
        if adequate_for_severe_decision and severe_count > 0:
            decision = "DISABLE_EXACT_ACTION_VERSION"
            disable_reason = "ADEQUATELY_POWERED_FRESH_UNRESOLVED_SEVERE_HARM"
        elif (
            adequate_for_efficacy_decision
            and task["ci95_upper"] < TASK_POINT_GATE
        ):
            decision = "DISABLE_EXACT_ACTION_VERSION"
            disable_reason = "ADEQUATELY_POWERED_EFFICACY_UPPER_BELOW_PRACTICAL_MINIMUM"
        elif qualifies:
            decision = "ADOPT_EXACT_ACTION_FOR_LATER_SELECTOR_BUILD"
        elif not adequate_for_redesign_decision:
            decision = "INCONCLUSIVE"
        else:
            decision = "RETAIN_AND_VERSION_REDESIGN"
        decisions[action] = {
            "decision": decision,
            "disable_reason": disable_reason,
            "scientifically_qualified_exact_action": qualifies,
            "severe_event_count": severe_count,
            "all_targets_executed_diagnostic": execution_gate,
            "gates": gates,
        }

    ledger_manifest = fresh_joint_ledger["manifest"]
    payload = {
        "schema": SCHEMA,
        "status": "COMPLETE_FRESH_ACTION_LEVEL_SCIENCE_DECISION",
        "science_protocol_sha256": science_protocol_sha256,
        "fresh_joint_ledger_manifest_sha256": ledger_manifest["manifest_sha256"],
        "development_power_result_sha256": development_power_admission[
            "power_result_sha256"
        ],
        "protocol": {
            "independent_unit": "connected_source_scene_family_component",
            "scientific_n": 23,
            "source_equal_estimator": True,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "global_family": "six_task_plus_fourteen_mechanism_Holm_FWER_0p05",
            "tail_family": "fourteen_metric_shared_bootstrap_maxT_UCB",
            "tail_maxT_critical": max_t_critical,
            "common_isotropic_disk_gaussian_mean_within_component": True,
            "native_is_reference_not_hypothesis": True,
        },
        "thresholds": {
            "task_point_normalized_px_at_least": TASK_POINT_GATE,
            "task_and_mechanism_CI95_lower_strictly_above": 0.0,
            "mechanism_primary_point_at_least": MECHANISM_PRIMARY_POINT_GATE,
            "harmed_pixel_fraction_simultaneous_upper": HARMED_FRACTION_UPPER,
            "pixel_harm_CVaR95_simultaneous_upper_raw_px": CVAR95_UPPER_RAW_PX,
            "severe_event_count": 0,
            "minimum_components": MINIMUM_COMPONENTS,
            "minimum_per_source": MINIMUM_PER_SOURCE,
            "minimum_per_fold": MINIMUM_PER_FOLD,
        },
        "hypothesis_and_tail_results": inference,
        "coverage": {
            "metrics": coverage,
            "availability_by_stratum": availability_coverage,
        },
        "per_action": decisions,
        "native": {
            "action_id": "native",
            "decision": "REFERENCE_RETAIN_NOT_A_QUALIFICATION_HYPOTHESIS",
        },
        "adopted_action_ids": [
            action for action, result in decisions.items()
            if result["decision"] == "ADOPT_EXACT_ACTION_FOR_LATER_SELECTOR_BUILD"
        ],
        "all_six_adopted_is_report_only": True,
        "selector_admission": False,
        "production_authority": False,
    }
    return {**payload, "science_result_sha256": canonical_sha256(payload)}


__all__ = [
    "BOOTSTRAP_REPEATS", "BOOTSTRAP_SEED", "SCHEMA",
    "evaluate_fresh_action_science_v1",
]
