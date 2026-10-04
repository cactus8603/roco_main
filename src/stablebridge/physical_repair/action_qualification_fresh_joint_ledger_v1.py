"""Fresh Spring18/KITTI5 component ledger for action-level qualification.

The fresh cohort has seven controlled-corruption cases per connected scene
component.  This reducer keeps the component as the only independent unit and
binds every upstream evidence package needed by the final science evaluator.
It deliberately shares the validated case-row contract with the development
ledger while emitting a distinct schema that cannot be mistaken for power
calibration data.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .action_qualification_joint_ledger_v1 import (
    ACTION_ORDER,
    ACTION_TO_STRATA,
    CASE_SCHEMA,
    OBSERVED,
    STRATUM_ORDER,
    TYPED_MISSING,
    canonical_sha256,
    validate_case_row_v1,
)


SCHEMA = "stablebridge-action-qualification-fresh-component-joint-ledger/v1"
MANIFEST_SCHEMA = (
    "stablebridge-action-qualification-fresh-joint-ledger-manifest/v1"
)
EXPECTED_SOURCE_COMPONENTS = {"spring": 18, "kitti": 5}
EXPECTED_FOLDS_BY_SOURCE = {
    "spring": {"0": 4, "1": 4, "2": 4, "3": 3, "4": 3},
    "kitti": {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1},
}
REQUIRED_EVIDENCE_BINDINGS = (
    "execution_freeze_sha256",
    "science_protocol_sha256",
    "development_power_admission_sha256",
    "prospective_cost_ceilings_sha256",
    "fresh_prediction_replay_sha256",
    "chronology_sha256",
    "task_observation_manifest_sha256",
    "mechanism_observation_manifest_sha256",
    "child_observation_manifest_sha256",
)


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _mean_or_missing(
    rows: Sequence[Mapping[str, Any]], status_key: str, value_key: str,
) -> float | None:
    if any(row[status_key] != OBSERVED for row in rows):
        return None
    return float(sum(float(row[value_key]) for row in rows) / len(rows))


def build_fresh_component_joint_ledger_v1(
    rows: Sequence[Mapping[str, Any]], *, roster_sha256: str,
    evidence_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Reduce exactly 161 fresh cases to 23 hash-bound component vectors."""
    roster_sha256 = _sha(roster_sha256, "roster_sha256")
    if set(evidence_bindings) != set(REQUIRED_EVIDENCE_BINDINGS):
        raise ValueError("fresh ledger evidence bindings are incomplete or extra")
    bindings = {
        name: _sha(evidence_bindings[name], name)
        for name in REQUIRED_EVIDENCE_BINDINGS
    }
    canonical = [validate_case_row_v1(row) for row in rows]
    expected_cases = sum(EXPECTED_SOURCE_COMPONENTS.values()) * len(STRATUM_ORDER)
    if len(canonical) != expected_cases:
        raise ValueError(f"fresh joint ledger requires exactly {expected_cases} cases")

    keys: set[tuple[str, str]] = set()
    grouped: dict[str, list[dict[str, Any]]] = {}
    identity: dict[str, tuple[str, int]] = {}
    for row in canonical:
        key = (row["component_id"], row["mechanism_stratum"])
        if key in keys:
            raise ValueError("duplicate fresh component/stratum case")
        keys.add(key)
        current = (row["source_dataset"], row["outer_fold"])
        previous = identity.setdefault(row["component_id"], current)
        if previous != current:
            raise ValueError("fresh component source/fold identity drift")
        grouped.setdefault(row["component_id"], []).append(row)

    source_counts = {source: 0 for source in EXPECTED_SOURCE_COMPONENTS}
    fold_counts = {
        source: {str(fold): 0 for fold in range(5)}
        for source in EXPECTED_SOURCE_COMPONENTS
    }
    components = []
    for component_id, selected in sorted(grouped.items()):
        if {row["mechanism_stratum"] for row in selected} != set(STRATUM_ORDER):
            raise ValueError("every fresh component must contain all seven strata")
        source, fold = identity[component_id]
        source_counts[source] += 1
        fold_counts[source][str(fold)] += 1
        by_stratum = {row["mechanism_stratum"]: row for row in selected}

        task: dict[str, Any] = {}
        action_gates: dict[str, Any] = {}
        for action in ACTION_ORDER:
            owned = [by_stratum[stratum] for stratum in ACTION_TO_STRATA[action]]
            task_gain = _mean_or_missing(
                owned, "task_status", "task_gain_normalized_px",
            )
            task[action] = {
                "status": OBSERVED if task_gain is not None else TYPED_MISSING,
                "normalized_gain": task_gain,
                "owned_strata": list(ACTION_TO_STRATA[action]),
            }
            action_gates[action] = {
                "all_executed": all(
                    row["execution_status"] == OBSERVED for row in owned
                ),
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

        availability: dict[str, Any] = {}
        mechanism: dict[str, Any] = {}
        tails: dict[str, Any] = {}
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
            "roster_sha256": roster_sha256,
            "evidence_bindings": bindings,
        }
        components.append({
            **payload,
            "component_observation_sha256": canonical_sha256(payload),
        })

    if source_counts != EXPECTED_SOURCE_COMPONENTS:
        raise ValueError("fresh source counts must be Spring18/KITTI5")
    if fold_counts != EXPECTED_FOLDS_BY_SOURCE:
        raise ValueError("fresh source/fold allocation drift")
    case_inputs = sorted(
        (row["case_id"], row["case_receipt_sha256"]) for row in canonical
    )
    manifest_payload = {
        "schema": MANIFEST_SCHEMA,
        "component_count": len(components),
        "case_count": len(canonical),
        "scientific_n": len(components),
        "source_component_counts": source_counts,
        "fold_counts_by_source": fold_counts,
        "case_region_stratum_endpoint_and_action_never_increase_n": True,
        "common_isotropic_disk_gaussian_averaged_within_component": True,
        "task_and_mechanism_missingness_are_separate": True,
        "roster_sha256": roster_sha256,
        "evidence_bindings": bindings,
        "input_case_receipts_sha256": canonical_sha256(case_inputs),
        "component_observation_sha256": [
            row["component_observation_sha256"] for row in components
        ],
    }
    return {
        "components": components,
        "manifest": {
            **manifest_payload,
            "manifest_sha256": canonical_sha256(manifest_payload),
        },
    }


__all__ = [
    "EXPECTED_FOLDS_BY_SOURCE", "EXPECTED_SOURCE_COMPONENTS", "MANIFEST_SCHEMA",
    "REQUIRED_EVIDENCE_BINDINGS", "SCHEMA",
    "build_fresh_component_joint_ledger_v1",
]
