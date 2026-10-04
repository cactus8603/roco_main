"""Build the component-level joint ledger used by action-bank power planning.

The development cohort contains seven controlled-corruption cases per
independent component.  Those cases, their regions, and the disk/Gaussian
sub-strata must never be counted as additional independent observations.  This
module performs that reduction without inspecting arrays or ground truth.

Task and mechanism availability are deliberately separate.  A valid action
child can have an unestimable mechanism metric and still contribute task and
tail evidence; conversely no missing field is ever imputed as zero.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA = "stablebridge-action-qualification-component-joint-ledger/v1"
CASE_SCHEMA = "stablebridge-action-qualification-joint-case/v1"
OBSERVED = "OBSERVED"
TYPED_MISSING = "TYPED_MISSING"

ACTION_TO_STRATA = {
    "paired_impulse_median3.endpoint_supported_v1": (
        "paired_impulse_median3",
    ),
    "paired_additive_wiener3.local_tile_sigma_v2": (
        "paired_additive_wiener3",
    ),
    "common_isotropic.local_recoverable_v3": (
        "common_disk", "common_gaussian",
    ),
    "common_motion.local_recoverable_v3": ("common_motion",),
    "jpeg_qcell_v3.local_macroblock_v3": ("jpeg_qcell_v3",),
    "jpeg_codec_path_v4.local_macroblock_v3": ("jpeg_codec_path_v4",),
}
ACTION_ORDER = tuple(ACTION_TO_STRATA)
STRATUM_TO_ACTION = {
    stratum: action
    for action, strata in ACTION_TO_STRATA.items()
    for stratum in strata
}
STRATUM_ORDER = tuple(STRATUM_TO_ACTION)
SOURCES = ("spring", "kitti")
EXPECTED_SOURCE_COMPONENTS = {"spring": 18, "kitti": 22}
_SHA256_LENGTH = 64


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric")
    return result


def _status(value: object, name: str) -> str:
    if value not in (OBSERVED, TYPED_MISSING):
        raise ValueError(f"{name} must be OBSERVED or TYPED_MISSING")
    return str(value)


def _validate_metric_pair(
    row: Mapping[str, Any], *, status_key: str, value_keys: Sequence[str],
) -> None:
    status = _status(row.get(status_key), status_key)
    reason_key = status_key.replace("_status", "_missing_reason")
    if status == OBSERVED:
        for key in value_keys:
            _finite(row.get(key), key)
        if row.get(reason_key) is not None:
            raise ValueError(f"{reason_key} must be null when observed")
    else:
        if any(row.get(key) is not None for key in value_keys):
            raise ValueError(f"typed-missing {status_key} cannot carry numbers")
        _text(row.get(reason_key), reason_key)


def validate_case_row_v1(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize one frozen component/stratum case row."""
    if row.get("schema") != CASE_SCHEMA:
        raise ValueError("joint case schema drift")
    case_id = _text(row.get("case_id"), "case_id")
    component_id = _text(row.get("component_id"), "component_id")
    source = _text(row.get("source_dataset"), "source_dataset")
    if source not in SOURCES:
        raise ValueError("source_dataset must be spring or kitti")
    fold = row.get("outer_fold")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
        raise ValueError("outer_fold must be an integer in [0,4]")
    stratum = _text(row.get("mechanism_stratum"), "mechanism_stratum")
    if stratum not in STRATUM_TO_ACTION:
        raise ValueError("unknown mechanism_stratum")
    action = _text(row.get("action_id"), "action_id")
    if action != STRATUM_TO_ACTION[stratum]:
        raise ValueError("action does not own mechanism_stratum")

    execution_status = _status(row.get("execution_status"), "execution_status")
    execution_reason = row.get("execution_missing_reason")
    if execution_status == OBSERVED:
        if execution_reason is not None:
            raise ValueError("observed execution cannot have a missing reason")
    else:
        _text(execution_reason, "execution_missing_reason")

    _validate_metric_pair(
        row,
        status_key="task_status",
        value_keys=(
            "task_gain_normalized_px", "harmed_pixel_fraction",
            "pixel_harm_cvar95_raw_px",
        ),
    )
    _validate_metric_pair(
        row,
        status_key="mechanism_status",
        value_keys=("mechanism_primary", "mechanism_specificity"),
    )
    if execution_status == TYPED_MISSING and (
        row["task_status"] == OBSERVED or row["mechanism_status"] == OBSERVED
    ):
        raise ValueError("a nonexecuted action cannot have observed outcomes")

    severe = row.get("severe_event")
    if row["task_status"] == OBSERVED:
        if not isinstance(severe, bool):
            raise ValueError("observed task row needs boolean severe_event")
    elif severe is not None:
        # A nonfinite action on the pre-frozen GT support must not disappear
        # as ordinary missingness. Its numeric task/tail estimands remain
        # unestimable, while the discrete severe event stays observed.
        if not (
            severe is True
            and row.get("task_missing_reason")
            == "NONFINITE_ACTION_ON_FIXED_GT_SUPPORT"
        ):
            raise ValueError(
                "typed-missing task row may carry only a nonfinite-action severe event"
            )

    for key in ("cost_gate_pass", "child_observables_complete"):
        if not isinstance(row.get(key), bool):
            raise ValueError(f"{key} must be boolean")
    receipt = _sha(row.get("case_receipt_sha256"), "case_receipt_sha256")

    result = {
        "schema": CASE_SCHEMA,
        "case_id": case_id,
        "component_id": component_id,
        "source_dataset": source,
        "outer_fold": fold,
        "mechanism_stratum": stratum,
        "action_id": action,
        "execution_status": execution_status,
        "execution_missing_reason": execution_reason,
        "task_status": row["task_status"],
        "task_missing_reason": row.get("task_missing_reason"),
        "task_gain_normalized_px": row.get("task_gain_normalized_px"),
        "harmed_pixel_fraction": row.get("harmed_pixel_fraction"),
        "pixel_harm_cvar95_raw_px": row.get("pixel_harm_cvar95_raw_px"),
        "severe_event": severe,
        "mechanism_status": row["mechanism_status"],
        "mechanism_missing_reason": row.get("mechanism_missing_reason"),
        "mechanism_primary": row.get("mechanism_primary"),
        "mechanism_specificity": row.get("mechanism_specificity"),
        "cost_gate_pass": row["cost_gate_pass"],
        "child_observables_complete": row["child_observables_complete"],
        "case_receipt_sha256": receipt,
    }
    return result


def _mean_or_missing(
    rows: Sequence[Mapping[str, Any]], status_key: str, value_key: str,
) -> float | None:
    if any(row[status_key] != OBSERVED for row in rows):
        return None
    return float(sum(float(row[value_key]) for row in rows) / len(rows))


def build_component_joint_ledger_v1(
    rows: Sequence[Mapping[str, Any]], *, protocol_sha256: str,
    component_roster_sha256: str,
    expected_source_components: Mapping[str, int] = EXPECTED_SOURCE_COMPONENTS,
) -> dict[str, Any]:
    """Reduce exactly seven cases per component to one joint planning row."""
    protocol_sha256 = _sha(protocol_sha256, "protocol_sha256")
    component_roster_sha256 = _sha(
        component_roster_sha256, "component_roster_sha256",
    )
    if dict(expected_source_components) != EXPECTED_SOURCE_COMPONENTS:
        raise ValueError("E269 development source counts are frozen at 18/22")
    canonical = [validate_case_row_v1(row) for row in rows]
    expected_cases = sum(EXPECTED_SOURCE_COMPONENTS.values()) * len(STRATUM_ORDER)
    if len(canonical) != expected_cases:
        raise ValueError(f"joint ledger requires exactly {expected_cases} cases")

    keys: set[tuple[str, str]] = set()
    grouped: dict[str, list[dict[str, Any]]] = {}
    identity: dict[str, tuple[str, int]] = {}
    for row in canonical:
        key = (row["component_id"], row["mechanism_stratum"])
        if key in keys:
            raise ValueError("duplicate component/stratum case")
        keys.add(key)
        current = (row["source_dataset"], row["outer_fold"])
        previous = identity.setdefault(row["component_id"], current)
        if previous != current:
            raise ValueError("component source/fold identity drift")
        grouped.setdefault(row["component_id"], []).append(row)

    source_counts = {source: 0 for source in SOURCES}
    component_rows = []
    for component_id, selected in sorted(grouped.items()):
        if {row["mechanism_stratum"] for row in selected} != set(STRATUM_ORDER):
            raise ValueError("every component must contain all seven strata")
        source, fold = identity[component_id]
        source_counts[source] += 1
        by_stratum = {row["mechanism_stratum"]: row for row in selected}

        task = {}
        action_gates = {}
        for action in ACTION_ORDER:
            owned = [by_stratum[stratum] for stratum in ACTION_TO_STRATA[action]]
            value = _mean_or_missing(owned, "task_status", "task_gain_normalized_px")
            task[action] = {
                "status": OBSERVED if value is not None else TYPED_MISSING,
                "normalized_gain": value,
                "owned_strata": list(ACTION_TO_STRATA[action]),
            }
            action_gates[action] = {
                "all_executed": all(
                    row["execution_status"] == OBSERVED for row in owned
                ),
                # Cost and child gates apply to every observed execution.  A
                # named nonexecution has no numeric cost/child to test and is
                # handled by the separate availability gate.  Expressing the
                # N/A rule here is essential for the two-stratum isotropic
                # action: one missing stratum must never hide a failure on the
                # other, actually executed stratum.
                "cost_gate_pass": all(
                    row["execution_status"] != OBSERVED
                    or row["cost_gate_pass"]
                    for row in owned
                ),
                "child_observables_complete": all(
                    row["execution_status"] != OBSERVED
                    or row["child_observables_complete"]
                    for row in owned
                ),
            }

        mechanism = {}
        tails = {}
        availability = {}
        for stratum in STRATUM_ORDER:
            row = by_stratum[stratum]
            availability[stratum] = {
                "status": row["execution_status"],
                "typed_missing_reason": row["execution_missing_reason"],
            }
            mechanism[stratum] = {
                "status": row["mechanism_status"],
                "primary": row["mechanism_primary"],
                "specificity": row["mechanism_specificity"],
            }
            tails[stratum] = {
                "status": row["task_status"],
                "harmed_pixel_fraction": row["harmed_pixel_fraction"],
                "pixel_harm_cvar95_raw_px": row["pixel_harm_cvar95_raw_px"],
                "severe_event": row["severe_event"],
            }

        payload = {
            "schema": SCHEMA,
            "component_id": component_id,
            "source_dataset": source,
            "outer_fold": fold,
            "task": task,
            "availability": availability,
            "mechanism": mechanism,
            "tails": tails,
            "action_gates": action_gates,
            "case_receipt_sha256_by_stratum": {
                stratum: by_stratum[stratum]["case_receipt_sha256"]
                for stratum in STRATUM_ORDER
            },
            "protocol_sha256": protocol_sha256,
            "component_roster_sha256": component_roster_sha256,
        }
        component_rows.append({
            **payload, "component_observation_sha256": canonical_sha256(payload),
        })

    if source_counts != EXPECTED_SOURCE_COMPONENTS:
        raise ValueError("component source counts must be Spring18/KITTI22")
    input_payload = sorted(
        (row["case_id"], row["case_receipt_sha256"]) for row in canonical
    )
    manifest_payload = {
        "schema": "stablebridge-action-qualification-joint-ledger-manifest/v1",
        "component_count": len(component_rows),
        "case_count": len(canonical),
        "scientific_n": len(component_rows),
        "source_component_counts": source_counts,
        "case_region_stratum_and_endpoint_never_increase_n": True,
        "common_isotropic_disk_gaussian_averaged_within_component": True,
        "task_and_mechanism_missingness_are_separate": True,
        "protocol_sha256": protocol_sha256,
        "component_roster_sha256": component_roster_sha256,
        "input_case_receipts_sha256": canonical_sha256(input_payload),
        "component_observation_sha256": [
            row["component_observation_sha256"] for row in component_rows
        ],
    }
    return {
        "components": component_rows,
        "manifest": {
            **manifest_payload,
            "manifest_sha256": canonical_sha256(manifest_payload),
        },
    }


__all__ = [
    "ACTION_ORDER", "ACTION_TO_STRATA", "CASE_SCHEMA",
    "EXPECTED_SOURCE_COMPONENTS", "OBSERVED", "SCHEMA", "STRATUM_ORDER",
    "TYPED_MISSING", "build_component_joint_ledger_v1", "canonical_sha256",
    "validate_case_row_v1",
]
