#!/usr/bin/env python3
"""Build a pruned successor experiment from an action-route audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--expected-actions", type=int, required=True)
    parser.add_argument("--expected-families", type=int, required=True)
    args = parser.parse_args()
    source = args.source_experiment.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / name for name in ("CANDIDATE_BANK.json", "TEACHER.jsonl", "TEACHER.json", "PROTOCOL.json")]
    if any(path.exists() for path in outputs):
        raise SystemExit("immutable successor input already exists")

    audit_path = source / "ACTION_ROUTE_AUDIT.json"
    audit = load(audit_path)
    passing = list(audit["passing_actions_for_retrained_reduced_bank"])
    if len(passing) != args.expected_actions or len(set(passing)) != len(passing):
        raise ValueError("unexpected passing-action count")
    source_bank_path = source / "CANDIDATE_BANK.json"
    source_bank = load(source_bank_path)
    families = []
    for family in source_bank["candidate_families"]:
        kept = [action for action in family["exact_controls"] if action in passing]
        if kept:
            families.append({**family, "exact_controls": kept})
    flattened = [action for family in families for action in family["exact_controls"]]
    if flattened != passing or len(families) != args.expected_families:
        raise ValueError("family filtering/order drift")

    candidate_bank = {
        "schema": "observable-positive-successor-bank/v1",
        "status": "OPENED_DEVELOPMENT_RETRAINING_CANDIDATE",
        "authority": {"production": False, "scientific_qualification": False, "selector_admission": False},
        "family_count": len(families),
        "exact_control_count": len(passing),
        "candidate_families": families,
        "pruned_after_source_audit": audit["failing_actions"],
        "source_bank": {"path": str(source_bank_path), "sha256": sha256(source_bank_path)},
        "source_audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
    }
    candidate_path = output_dir / "CANDIDATE_BANK.json"
    atomic_json(candidate_path, candidate_bank)

    source_teacher_summary_path = source / "TEACHER.json"
    source_teacher = load(source_teacher_summary_path)
    source_records_path = Path(source_teacher["records"])
    if sha256(source_records_path) != source_teacher["records_sha256"]:
        raise ValueError("source teacher record hash drift")
    allowed = {"CSB/OF/SEA-RAFT/action/P0", *passing}
    rows = []
    for line in source_records_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        row["schema"] = "observable-positive-successor-teacher/v1"
        row["action_targets"] = {
            action: values for action, values in row["action_targets"].items()
            if action in allowed
        }
        if set(row["action_targets"]) != allowed:
            raise ValueError(f"teacher action filtering failed: {row['row_id']}")
        rows.append(row)
    if len(rows) != 1200:
        raise ValueError("teacher row count drift")
    records_path = output_dir / "TEACHER.jsonl"
    atomic_text(records_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    teacher = {
        "schema": "observable-positive-successor-teacher-summary/v1",
        "status": "COMPLETE_OPENED_DEVELOPMENT",
        "diagnostic_only": True,
        "record_count": len(rows),
        "action_ids": ["CSB/OF/SEA-RAFT/action/P0", *passing],
        "records": str(records_path),
        "records_sha256": sha256(records_path),
        "source_teacher": {"path": str(source_teacher_summary_path), "sha256": sha256(source_teacher_summary_path)},
        "source_teacher_records_sha256": sha256(source_records_path),
    }
    teacher_path = output_dir / "TEACHER.json"
    atomic_json(teacher_path, teacher)

    protocol = load(source / "PROTOCOL.json")
    protocol["controller_id"] = args.controller_id
    protocol["claim_scope"] = (
        "opened-development nested scene-OOF retraining screen for an "
        f"observable-positive {len(families)}-family {len(passing)}-anchor bank"
    )
    protocol["action_ids"] = teacher["action_ids"]
    protocol["action_families"] = families
    protocol["sources"]["teacher"] = str(teacher_path)
    protocol["sources"]["teacher_sha256"] = sha256(teacher_path)
    protocol["sources"]["teacher_records_sha256"] = sha256(records_path)
    for key in (
        "candidate_bank", "candidate_bank_sha256", "e275_action_route_audit",
        "e275_action_route_audit_sha256",
    ):
        protocol["sources"].pop(key, None)
    protocol["sources"]["candidate_bank"] = str(candidate_path)
    protocol["sources"]["candidate_bank_sha256"] = sha256(candidate_path)
    protocol["sources"]["predecessor_action_route_audit"] = str(audit_path)
    protocol["sources"]["predecessor_action_route_audit_sha256"] = sha256(audit_path)
    atomic_json(output_dir / "PROTOCOL.json", protocol)
    print(json.dumps({
        "status": "FROZEN_OPENED_DEVELOPMENT_INPUTS",
        "families": len(families),
        "actions_including_native": len(passing) + 1,
        "rows": len(rows),
        "protocol_sha256": sha256(output_dir / "PROTOCOL.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
