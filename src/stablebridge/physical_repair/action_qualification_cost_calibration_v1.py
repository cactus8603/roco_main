"""Pre-GT stress selection and prospective cost ceilings for action qualification."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .action_qualification_joint_ledger_v1 import ACTION_ORDER, canonical_sha256


SELECTION_SCHEMA = "stablebridge-action-qualification-cost-stress-selection/v1"
PROFILE_SCHEMA = "stablebridge-action-qualification-isolated-cost-profile/v1"
CEILING_SCHEMA = "stablebridge-action-qualification-cost-ceilings/v1"
SOURCES = ("spring", "kitti")
REPEATS = 3
FORWARDS_PER_PROFILE = 8
TIME_MULTIPLIER = 1.25
TIME_FIXED_SLACK_SECONDS = 0.05
BYTES_MULTIPLIER = 1.10
BYTES_FIXED_SLACK = 16 * (1 << 20)
PEAK_FIXED_SLACK = 256 * (1 << 20)
PEAK_QUANTUM = 256 * (1 << 20)
DEVICE_GUARD_PEAK_RESERVED = 9 * (1 << 30)


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _integer_nonnegative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _terminal_hash_valid(value: Mapping[str, Any]) -> bool:
    payload = {
        key: item for key, item in value.items()
        if key not in {"semantic_receipt_sha256", "telemetry"}
    }
    return value.get("semantic_receipt_sha256") == canonical_sha256(payload)


def _stress_tuple(value: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    expected_order = [
        "image_pixels", "read_halo_pixels", "support_pixels",
        "modified_endpoint_count", "case_id",
    ]
    payload = {
        key: item for key, item in value.items() if key != "stress_key_sha256"
    }
    if (
        value.get("stress_key_sha256") != canonical_sha256(payload)
        or value.get("selection_order") != expected_order
        or value.get("task_GT_read") is not False
        or value.get("outcome_read") is not False
    ):
        raise ValueError("cost stress key semantic drift")
    numbers = tuple(
        _integer_nonnegative(value.get(key), key)
        for key in expected_order[:-1]
    )
    case_id = value.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("cost stress case_id missing")
    if numbers[1] < numbers[2] or numbers[3] not in {1, 2}:
        raise ValueError("cost stress support/endpoint geometry invalid")
    return (*numbers, case_id)


def select_cost_stress_cases_v1(
    terminals: Sequence[Mapping[str, Any]], *, protocol_sha256: str,
) -> dict[str, Any]:
    """Select one outcome-blind maximum-stress case per source and action."""
    protocol_sha256 = _sha(protocol_sha256, "protocol_sha256")
    if len(terminals) != 280:
        raise ValueError("cost selection requires all 280 prediction terminals")
    case_ids: set[str] = set()
    candidates: dict[tuple[str, str], list[tuple[tuple[Any, ...], Mapping[str, Any]]]] = {}
    for terminal in terminals:
        if (
            terminal.get("schema") != "e269-development-prediction-terminal/v1"
            or terminal.get("terminal") is not True
            or terminal.get("status") not in {"AVAILABLE", "TYPED_MISSING"}
            or terminal.get("task_GT_read") is not False
            or terminal.get("outcome_read") is not False
            or not _terminal_hash_valid(terminal)
        ):
            raise ValueError("invalid prediction terminal in cost selection")
        case_id = terminal.get("case_id")
        if not isinstance(case_id, str) or case_id in case_ids:
            raise ValueError("duplicate or invalid prediction case_id")
        case_ids.add(case_id)
        source = terminal.get("source_dataset")
        action = terminal.get("target_public_action_id")
        if source not in SOURCES or action not in ACTION_ORDER:
            raise ValueError("prediction source/action drift")
        if terminal["status"] == "AVAILABLE":
            stress = terminal.get("stress_selection_metrics")
            if not isinstance(stress, Mapping):
                raise ValueError("available terminal lacks cost stress metrics")
            key = _stress_tuple(stress)
            if key[-1] != case_id:
                raise ValueError("stress key case identity drift")
            candidates.setdefault((source, action), []).append((key, terminal))

    selections: dict[str, dict[str, Any]] = {}
    all_bound = True
    for source in SOURCES:
        selections[source] = {}
        for action in ACTION_ORDER:
            rows = candidates.get((source, action), [])
            if not rows:
                all_bound = False
                selections[source][action] = {
                    "status": "UNBOUND_NO_AVAILABLE_ACTION",
                    "case_id": None,
                    "terminal_semantic_receipt_sha256": None,
                    "stress_key": None,
                }
                continue
            key, selected = max(rows, key=lambda item: item[0])
            selections[source][action] = {
                "status": "SELECTED_PRE_GT_MAXIMUM_STRESS",
                "case_id": selected["case_id"],
                "component_id": selected["component_id"],
                "mechanism_stratum": selected["mechanism_stratum"],
                "terminal_semantic_receipt_sha256": selected[
                    "semantic_receipt_sha256"
                ],
                "action_binding_sha256": selected["action_binding_sha256"],
                "stress_key": list(key),
            }
    payload = {
        "schema": SELECTION_SCHEMA,
        "status": (
            "COMPLETE_TWELVE_PRE_GT_STRESS_CASES" if all_bound
            else "INCOMPLETE_ONE_OR_MORE_SOURCE_ACTIONS_UNAVAILABLE"
        ),
        "protocol_sha256": protocol_sha256,
        "terminal_count": len(terminals),
        "selection_rule": (
            "lexicographic_max(image_pixels,read_halo_pixels,support_pixels,"
            "modified_endpoint_count,case_id)_within_source_action"
        ),
        "selections": selections,
        "selected_count": sum(
            row["status"] == "SELECTED_PRE_GT_MAXIMUM_STRESS"
            for source_rows in selections.values() for row in source_rows.values()
        ),
        "all_source_actions_bound": all_bound,
        "task_GT_read": False,
        "outcome_read": False,
        "scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
    }
    return {**payload, "selection_sha256": canonical_sha256(payload)}


def _ceil_seconds(value: float) -> float:
    return math.ceil(value * 1000.0) / 1000.0


def _ceil_bytes(value: float, quantum: int) -> int:
    return int(math.ceil(value / quantum) * quantum)


def build_prospective_cost_ceilings_v1(
    profiles: Sequence[Mapping[str, Any]], *, selection_sha256: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Bind 36 isolated cold profiles into six prospective hard ceilings."""
    selection_sha256 = _sha(selection_sha256, "selection_sha256")
    protocol_sha256 = _sha(protocol_sha256, "protocol_sha256")
    if len(profiles) != len(SOURCES) * len(ACTION_ORDER) * REPEATS:
        raise ValueError("cost calibration requires exactly 36 profiles")
    expected = {
        (source, action, repeat)
        for source in SOURCES for action in ACTION_ORDER for repeat in range(REPEATS)
    }
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in profiles:
        if row.get("schema") != PROFILE_SCHEMA:
            raise ValueError("cost profile schema drift")
        payload = {
            key: item for key, item in row.items() if key != "profile_sha256"
        }
        if row.get("profile_sha256") != canonical_sha256(payload):
            raise ValueError("cost profile semantic hash drift")
        key = (row.get("source_dataset"), row.get("action_id"), row.get("repeat"))
        if key not in expected or key in indexed:
            raise ValueError("cost profile source/action/repeat drift")
        if (
            row.get("selection_sha256") != selection_sha256
            or row.get("protocol_sha256") != protocol_sha256
            or row.get("cache_state") != "cold"
            or row.get("forward_count") != FORWARDS_PER_PROFILE
            or row.get("retry_count") != 0
            or row.get("OOM_count") != 0
            or row.get("task_GT_read") is not False
            or row.get("outcome_read") is not False
        ):
            raise ValueError("cost profile hard execution gate failed")
        for field in (
            "operator_cpu_seconds", "operator_wall_seconds",
            "bundle_cpu_seconds", "bundle_wall_seconds",
        ):
            _finite_nonnegative(row.get(field), field)
        for field in (
            "bytes_moved", "peak_allocated_bytes", "peak_reserved_bytes",
        ):
            _integer_nonnegative(row.get(field), field)
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("cost profile factorial is incomplete")

    ceilings = {}
    for action in ACTION_ORDER:
        rows = [
            indexed[(source, action, repeat)]
            for source in SOURCES for repeat in range(REPEATS)
        ]
        maximum = {
            field: max(row[field] for row in rows)
            for field in (
                "operator_cpu_seconds", "operator_wall_seconds",
                "bundle_cpu_seconds", "bundle_wall_seconds", "bytes_moved",
                "peak_allocated_bytes", "peak_reserved_bytes",
            )
        }
        peak_allocated = _ceil_bytes(
            maximum["peak_allocated_bytes"] + PEAK_FIXED_SLACK, PEAK_QUANTUM,
        )
        peak_reserved = _ceil_bytes(
            maximum["peak_reserved_bytes"] + PEAK_FIXED_SLACK, PEAK_QUANTUM,
        )
        ceilings[action] = {
            "development_max": maximum,
            "operator_cpu_seconds_upper": _ceil_seconds(
                TIME_MULTIPLIER * maximum["operator_cpu_seconds"]
                + TIME_FIXED_SLACK_SECONDS
            ),
            "operator_wall_seconds_upper": _ceil_seconds(
                TIME_MULTIPLIER * maximum["operator_wall_seconds"]
                + TIME_FIXED_SLACK_SECONDS
            ),
            "bundle_cpu_seconds_upper": _ceil_seconds(
                TIME_MULTIPLIER * maximum["bundle_cpu_seconds"]
                + TIME_FIXED_SLACK_SECONDS
            ),
            "bundle_wall_seconds_upper": _ceil_seconds(
                TIME_MULTIPLIER * maximum["bundle_wall_seconds"]
                + TIME_FIXED_SLACK_SECONDS
            ),
            "bytes_moved_upper": _ceil_bytes(
                BYTES_MULTIPLIER * maximum["bytes_moved"] + BYTES_FIXED_SLACK,
                1 << 20,
            ),
            "peak_allocated_bytes_upper": peak_allocated,
            "peak_reserved_bytes_upper": peak_reserved,
            "device_guard_pass": peak_reserved <= DEVICE_GUARD_PEAK_RESERVED,
            "forward_count_upper": FORWARDS_PER_PROFILE,
            "retry_count_upper": 0,
            "OOM_count_upper": 0,
        }
    payload = {
        "schema": CEILING_SCHEMA,
        "status": (
            "COMPLETE_PROSPECTIVE_COST_CEILINGS"
            if all(row["device_guard_pass"] for row in ceilings.values())
            else "DEVICE_GUARD_FAILED"
        ),
        "selection_sha256": selection_sha256,
        "protocol_sha256": protocol_sha256,
        "profile_count": len(profiles),
        "timing_repeats_add_to_scientific_n": False,
        "formula": {
            "time_multiplier": TIME_MULTIPLIER,
            "time_fixed_slack_seconds": TIME_FIXED_SLACK_SECONDS,
            "bytes_multiplier": BYTES_MULTIPLIER,
            "bytes_fixed_slack": BYTES_FIXED_SLACK,
            "peak_fixed_slack": PEAK_FIXED_SLACK,
            "peak_quantum": PEAK_QUANTUM,
            "device_guard_peak_reserved_bytes": DEVICE_GUARD_PEAK_RESERVED,
        },
        "ceilings": ceilings,
        "task_GT_read": False,
        "outcome_read": False,
        "fresh_scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
    }
    return {**payload, "ceilings_sha256": canonical_sha256(payload)}


__all__ = [
    "ACTION_ORDER", "CEILING_SCHEMA", "FORWARDS_PER_PROFILE", "PROFILE_SCHEMA",
    "REPEATS", "SELECTION_SCHEMA", "SOURCES",
    "build_prospective_cost_ceilings_v1", "select_cost_stress_cases_v1",
]
