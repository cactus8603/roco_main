#!/usr/bin/env python3
"""Build the exhaustive 29-action admission ledger from hash-bound evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/E243_restoration_candidate_integration_v1/FROZEN_ACTION_BANK.json"
REDUCED = HERE / "REDUCED_CANDIDATE_BANK.json"
OPENED = HERE / "OPENED_CANDIDATE_ACTION_BANK.json"
E279 = HERE / "E279_SELECTION_AWARE_LOO_RESULT.json"
E281 = HERE / "E281_SELECTION_AWARE_FAMILY_LOO_RESULT.json"
E282 = HERE / "E282_STRENGTH_PROPOSAL_RESULT.json"
OUTPUT = HERE / "ACTION_ADMISSION_MATRIX.json"
AUDITS = (
    ROOT / "experiments/E275_reduced_bank_observable_routing_v1/ACTION_ROUTE_AUDIT.json",
    ROOT / "experiments/E276_observable_positive_bank_v1/ACTION_ROUTE_AUDIT.json",
    ROOT / "experiments/E277_stable_observable_bank_v1/ACTION_ROUTE_AUDIT.json",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    source = _load(SOURCE)
    reduced = _load(REDUCED)
    opened = _load(OPENED)
    e279 = _load(E279)
    e281 = _load(E281)
    e282 = _load(E282)

    candidate_family = {
        control["action_id"]: family["family_id"]
        for family in opened["families"]
        for control in family["exact_controls"]
    }
    initial_family = {
        action: family["family_id"]
        for family in reduced["candidate_families"]
        for action in family["exact_controls"]
    }
    overlap = {
        row["action_id"]: row["reason"]
        for row in reduced["overlap_pruned_controls"]
    }
    shadow = {
        action: (group["family_id"], group["disposition"])
        for group in reduced["shadow_mechanism_groups"]
        for action in group["actions"]
    }
    routed_rejections = {}
    audit_hashes = []
    for path in AUDITS:
        audit = _load(path)
        audit_hashes.append({"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)})
        for action in audit["failing_actions"]:
            routed_rejections[action] = {
                "experiment": path.parent.name,
                **audit["actions"][action],
            }

    rows = []
    for action in source["actions"]:
        action_id = action["action_id"]
        availability = action["availability"]
        availability_status = (
            availability.get("status")
            or availability.get("catalog_label")
            or availability.get("execution")
        )
        row = {
            "action_id": action_id,
            "operator": action["operator"],
            "availability_status": availability_status,
            "production_authority": False,
            "fresh_component_disjoint_validation": False,
            "final_validated": False,
        }
        if action_id in candidate_family:
            comparison = e279["comparisons"][action_id]
            family_comparison = e281["comparisons"][candidate_family[action_id]]
            row.update({
                "family_id": candidate_family[action_id],
                "disposition": "FROZEN_OPENED_CANDIDATE_NOT_FINAL",
                "theory_overlap_gate": "PASS_DISTINCT_EXACT_CONTROL_WITHIN_FAMILY",
                "oracle_capacity_gate": (
                    "PASS_SIMULTANEOUS_LOWER_BOUND_POSITIVE"
                    if comparison["oracle_capacity_nonredundancy_passes"]
                    else "FAIL"
                ),
                "observable_routed_contribution_gate": "PASS_E278_INDIVIDUAL_95CI",
                "selection_aware_loo_gate": (
                    "PASS_SIMULTANEOUS_LOWER_BOUND_POSITIVE"
                    if comparison["selection_aware_nonredundancy_passes"]
                    else "FAIL_SIMULTANEOUS_LOWER_BOUND_NOT_POSITIVE"
                ),
                "oracle_incremental_corrupt_macro_epe_px": comparison[
                    "oracle_corrupt_macro_incremental_mean_epe_px"
                ],
                "oracle_simultaneous_lower_px": comparison[
                    "oracle_corrupt_macro_bonferroni_one_sided_95pct_lower_px"
                ],
                "selection_aware_loo_delta_px": comparison[
                    "loo_minus_full_corrupt_macro_mean_epe_px"
                ],
                "selection_aware_simultaneous_lower_px": comparison[
                    "corrupt_macro_bonferroni_one_sided_95pct_lower_px"
                ],
                "selection_aware_family_loo_gate": (
                    "PASS_SIMULTANEOUS_LOWER_BOUND_POSITIVE"
                    if family_comparison["selection_aware_nonredundancy_passes"]
                    else "FAIL_SIMULTANEOUS_LOWER_BOUND_NOT_POSITIVE"
                ),
                "selection_aware_family_loo_delta_px": family_comparison[
                    "loo_minus_full_corrupt_macro_mean_epe_px"
                ],
                "selection_aware_family_simultaneous_lower_px": family_comparison[
                    "corrupt_macro_bonferroni_one_sided_95pct_lower_px"
                ],
                "strength_proposal": {
                    "mode": "DISCRETE_EXACT_CONTROL",
                    "operator_parameter": action["operator"].get("parameters", {}),
                    "normalized_input_strength": 1.0,
                    "output_beta": 1.0,
                    "continuous_interpolation_authorized": False,
                    "opened_e3_validation": (
                        e282["status"]
                        if candidate_family[action_id]
                        == "optical.lowpass_hf_suppression.v1"
                        else "BINARY_NATIVE_VS_EXACT_CONTROL_ONLY"
                    ),
                },
                "blocking_gate": "FRESH_VALIDATION_MISSING",
            })
        elif action_id in routed_rejections:
            failure = routed_rejections[action_id]
            row.update({
                "family_id": initial_family[action_id],
                "disposition": "REJECT_OPENED_OBSERVABLE_ROUTING",
                "theory_overlap_gate": "PASS_INITIAL_CAPACITY_SHORTLIST",
                "oracle_capacity_gate": "PASS_INITIAL_99P5_98_RETENTION_SUBSET",
                "observable_routed_contribution_gate": failure["disposition"],
                "selection_aware_loo_gate": "NOT_RUN_AFTER_EARLIER_REJECTION",
                "blocking_gate": "OBSERVABLE_ROUTED_CONTRIBUTION",
                "evidence_experiment": failure["experiment"],
                "routed_contribution_95ci_px": failure[
                    "scene_cluster_bootstrap_population_contribution_95_interval_raw_px"
                ],
            })
        elif action_id in overlap:
            row.update({
                "family_id": "optical_capacity_overlap_pruned",
                "disposition": "REJECT_THEORY_CAPACITY_OVERLAP",
                "theory_overlap_gate": "FAIL_JOINT_SUBSET_NECESSITY",
                "oracle_capacity_gate": "NOT_NEEDED_FOR_FROZEN_RETENTION_GATE",
                "observable_routed_contribution_gate": "NOT_RUN_AFTER_EARLIER_REJECTION",
                "selection_aware_loo_gate": "NOT_RUN_AFTER_EARLIER_REJECTION",
                "blocking_gate": "THEORY_CAPACITY_OVERLAP",
                "reason": overlap[action_id],
            })
        elif action_id in shadow:
            family_id, disposition = shadow[action_id]
            row.update({
                "family_id": family_id,
                "disposition": disposition,
                "theory_overlap_gate": "SHADOW_FAMILY_COMPARATOR",
                "oracle_capacity_gate": "MISSING_TASK_ALIGNED_FULL_ACTION_EVIDENCE",
                "observable_routed_contribution_gate": "NOT_RUN",
                "selection_aware_loo_gate": "NOT_RUN",
                "blocking_gate": "FULL_ACTION_OR_FRESH_TASK_EVIDENCE_MISSING",
            })
        else:
            raise ValueError(f"unpartitioned action: {action_id}")
        rows.append(row)

    ids = [row["action_id"] for row in rows]
    if len(ids) != 29 or len(set(ids)) != 29:
        raise ValueError("admission matrix must contain 29 unique actions")
    counts = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    output = {
        "schema": "action-bank-exhaustive-admission-matrix/v1",
        "status": "COMPLETE_OPENED_DEVELOPMENT_FINAL_VALIDATED_EMPTY",
        "authority": {
            "production": False,
            "scientific_qualification": False,
            "selector_admission": False,
        },
        "action_count": len(rows),
        "final_validated_action_ids": [],
        "frozen_opened_candidate_action_ids": list(candidate_family),
        "counts_by_disposition": counts,
        "actions": rows,
        "sources": {
            "frozen_29": {"path": str(SOURCE.relative_to(ROOT)), "sha256": _sha256(SOURCE)},
            "capacity_reduction": {"path": str(REDUCED.relative_to(ROOT)), "sha256": _sha256(REDUCED)},
            "opened_candidate": {"path": str(OPENED.relative_to(ROOT)), "sha256": _sha256(OPENED)},
            "selection_aware_loo": {"path": str(E279.relative_to(ROOT)), "sha256": _sha256(E279)},
            "selection_aware_family_loo": {"path": str(E281.relative_to(ROOT)), "sha256": _sha256(E281)},
            "strength_proposal": {"path": str(E282.relative_to(ROOT)), "sha256": _sha256(E282)},
            "route_audits": audit_hashes,
        },
    }
    _atomic_json(OUTPUT, output)
    print(json.dumps({
        "status": output["status"],
        "actions": len(rows),
        "final": 0,
        "opened_candidates": len(candidate_family),
        "output": str(OUTPUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
