#!/usr/bin/env python3
"""Freeze one E278 leave-one-action-out diagnostic retraining input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/E278_minimal_stable_observable_bank_v1"
NATIVE = "CSB/OF/SEA-RAFT/action/P0"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-action", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [
        output / name
        for name in ("CANDIDATE_BANK.json", "TEACHER.jsonl", "TEACHER.json", "PROTOCOL.json")
    ]
    if any(path.exists() for path in targets):
        raise SystemExit("immutable LOO inputs already exist")

    source_bank_path = SOURCE / "CANDIDATE_BANK.json"
    source_bank = _load(source_bank_path)
    source_ids = [
        action
        for family in source_bank["candidate_families"]
        for action in family["exact_controls"]
    ]
    if args.exclude_action not in source_ids:
        raise ValueError("excluded action is not in the frozen E278 bank")
    retained = [action for action in source_ids if action != args.exclude_action]
    families = []
    for family in source_bank["candidate_families"]:
        members = [
            action for action in family["exact_controls"]
            if action != args.exclude_action
        ]
        if members:
            families.append({**family, "exact_controls": members})
    if [action for family in families for action in family["exact_controls"]] != retained:
        raise ValueError("family/action order drift")

    bank = {
        "schema": "selection-aware-loo-candidate-bank/v1",
        "status": "OPENED_DEVELOPMENT_METHOD_DIAGNOSTIC_ONLY",
        "authority": {
            "production": False,
            "scientific_qualification": False,
            "selector_admission": False,
        },
        "excluded_action_id": args.exclude_action,
        "family_count": len(families),
        "exact_control_count": len(retained),
        "candidate_families": families,
        "source_full_bank": {
            "path": str(source_bank_path),
            "sha256": _sha256(source_bank_path),
        },
    }
    bank_path = output / "CANDIDATE_BANK.json"
    _atomic_json(bank_path, bank)

    source_teacher_path = SOURCE / "TEACHER.json"
    source_teacher = _load(source_teacher_path)
    source_records_path = Path(source_teacher["records"])
    if _sha256(source_records_path) != source_teacher["records_sha256"]:
        raise ValueError("source teacher record hash drift")
    allowed = {NATIVE, *retained}
    rows = []
    for line in source_records_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        row["schema"] = "selection-aware-loo-teacher/v1"
        row["action_targets"] = {
            action: target
            for action, target in row["action_targets"].items()
            if action in allowed
        }
        if set(row["action_targets"]) != allowed:
            raise ValueError(f"teacher filtering failed for {row['row_id']}")
        rows.append(row)
    if len(rows) != 1200:
        raise ValueError("teacher row count drift")
    records_path = output / "TEACHER.jsonl"
    _atomic_text(
        records_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    teacher = {
        "schema": "selection-aware-loo-teacher-summary/v1",
        "status": "COMPLETE_OPENED_DEVELOPMENT_METHOD_DIAGNOSTIC",
        "diagnostic_only": True,
        "record_count": len(rows),
        "action_ids": [NATIVE, *retained],
        "records": str(records_path),
        "records_sha256": _sha256(records_path),
        "source_teacher": {
            "path": str(source_teacher_path),
            "sha256": _sha256(source_teacher_path),
        },
        "source_teacher_records_sha256": _sha256(source_records_path),
    }
    teacher_path = output / "TEACHER.json"
    _atomic_json(teacher_path, teacher)

    protocol = _load(SOURCE / "PROTOCOL.json")
    short = args.exclude_action.rsplit("/", 1)[-1]
    protocol["controller_id"] = f"CSB/OF/SEA-RAFT/controller/e278-loo-{short}/v1"
    protocol["claim_scope"] = (
        "opened-development selection-aware leave-one-action-out method "
        "diagnostic; not fresh qualification"
    )
    protocol["action_ids"] = teacher["action_ids"]
    protocol["action_families"] = families
    protocol["diagnostic_only"] = True
    protocol["exposure_boundary"] = {
        "fresh_or_deployment_authority": False,
        "opened_development": True,
        "researcher_has_seen_source_outcomes": True,
        "adaptive_e3_reuse": True,
    }
    protocol["sources"].pop("predecessor_action_route_audit", None)
    protocol["sources"].pop("predecessor_action_route_audit_sha256", None)
    protocol["sources"]["candidate_bank"] = str(bank_path)
    protocol["sources"]["candidate_bank_sha256"] = _sha256(bank_path)
    protocol["sources"]["teacher"] = str(teacher_path)
    protocol["sources"]["teacher_sha256"] = _sha256(teacher_path)
    protocol["sources"]["teacher_records_sha256"] = _sha256(records_path)
    protocol["sources"]["full_bank_result"] = str(SOURCE / "RESULT.json")
    protocol["sources"]["full_bank_result_sha256"] = _sha256(SOURCE / "RESULT.json")
    protocol["loo_design"] = {
        "excluded_action_id": args.exclude_action,
        "comparison": "paired outer-OOF CTRL-FACT loss versus the full E278 bank",
        "interpretation": "method diagnostic only because E3 was adaptively reused",
    }
    protocol_path = output / "PROTOCOL.json"
    _atomic_json(protocol_path, protocol)
    print(json.dumps({
        "status": "FROZEN_OPENED_DEVELOPMENT_LOO_INPUT",
        "excluded_action_id": args.exclude_action,
        "retained_actions": len(retained),
        "families": len(families),
        "rows": len(rows),
        "protocol_sha256": _sha256(protocol_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
