#!/usr/bin/env python3
"""Freeze one E278 leave-one-family-out diagnostic retraining input."""

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


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-family", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [
        output / name
        for name in (
            "CANDIDATE_BANK.json",
            "TEACHER.jsonl",
            "TEACHER.json",
            "PROTOCOL.json",
        )
    ]
    if any(path.exists() for path in targets):
        raise SystemExit("immutable family-LOO inputs already exist")

    source_bank_path = SOURCE / "CANDIDATE_BANK.json"
    source_bank = _load(source_bank_path)
    families = list(source_bank["candidate_families"])
    matching = [
        family for family in families if family["family_id"] == args.exclude_family
    ]
    if len(matching) != 1:
        raise ValueError("excluded family must occur exactly once")
    excluded_actions = list(matching[0]["exact_controls"])
    retained_families = [
        family for family in families if family["family_id"] != args.exclude_family
    ]
    retained_actions = [
        action
        for family in retained_families
        for action in family["exact_controls"]
    ]
    if not excluded_actions or not retained_actions:
        raise ValueError("family LOO must retain and exclude at least one control")

    bank = {
        "schema": "selection-aware-family-loo-candidate-bank/v1",
        "status": "OPENED_DEVELOPMENT_METHOD_DIAGNOSTIC_ONLY",
        "authority": {
            "production": False,
            "scientific_qualification": False,
            "selector_admission": False,
        },
        "excluded_family_id": args.exclude_family,
        "excluded_action_ids": excluded_actions,
        "family_count": len(retained_families),
        "exact_control_count": len(retained_actions),
        "candidate_families": retained_families,
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
    allowed = {NATIVE, *retained_actions}
    rows = []
    for line in source_records_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        row["schema"] = "selection-aware-family-loo-teacher/v1"
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
        "schema": "selection-aware-family-loo-teacher-summary/v1",
        "status": "COMPLETE_OPENED_DEVELOPMENT_METHOD_DIAGNOSTIC",
        "diagnostic_only": True,
        "record_count": len(rows),
        "action_ids": [NATIVE, *retained_actions],
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
    short = args.exclude_family.rsplit(".", 2)[-2]
    protocol["controller_id"] = (
        f"CSB/OF/SEA-RAFT/controller/e278-family-loo-{short}/v1"
    )
    protocol["claim_scope"] = (
        "opened-development selection-aware leave-one-family-out method "
        "diagnostic; not fresh qualification"
    )
    protocol["action_ids"] = teacher["action_ids"]
    protocol["action_families"] = retained_families
    protocol["diagnostic_only"] = True
    nonnative = len(retained_actions)
    for key in (
        "utility_head_outputs",
        "harm_head_outputs",
        "catastrophe_head_outputs",
    ):
        protocol["factorized_model"][key] = nonnative
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
    protocol["sources"]["full_bank_result_sha256"] = _sha256(
        SOURCE / "RESULT.json"
    )
    protocol["family_loo_design"] = {
        "excluded_family_id": args.exclude_family,
        "excluded_action_ids": excluded_actions,
        "comparison": "paired outer-OOF CTRL-FACT loss versus the full E278 bank",
        "interpretation": "method diagnostic only because E3 was adaptively reused",
    }
    protocol_path = output / "PROTOCOL.json"
    _atomic_json(protocol_path, protocol)
    print(
        json.dumps(
            {
                "status": "FROZEN_OPENED_DEVELOPMENT_FAMILY_LOO_INPUT",
                "excluded_family_id": args.exclude_family,
                "excluded_actions": excluded_actions,
                "retained_actions": len(retained_actions),
                "families": len(retained_families),
                "rows": len(rows),
                "protocol_sha256": _sha256(protocol_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
