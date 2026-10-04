#!/usr/bin/env python3
"""Fail-closed verifier for the exhaustive 29-action admission matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = Path(__file__).with_name("ACTION_ADMISSION_MATRIX.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    matrix = _load(MATRIX)
    rows = matrix["actions"]
    ids = [row["action_id"] for row in rows]
    assert matrix["status"] == "COMPLETE_OPENED_DEVELOPMENT_FINAL_VALIDATED_EMPTY"
    assert matrix["action_count"] == len(ids) == len(set(ids)) == 29
    assert matrix["final_validated_action_ids"] == []
    assert not any(matrix["authority"].values())
    assert all(
        row["final_validated"] is False
        and row["fresh_component_disjoint_validation"] is False
        and row["production_authority"] is False
        for row in rows
    )

    for source in (
        matrix["sources"]["frozen_29"],
        matrix["sources"]["capacity_reduction"],
        matrix["sources"]["opened_candidate"],
        matrix["sources"]["selection_aware_loo"],
        matrix["sources"]["selection_aware_family_loo"],
        matrix["sources"]["strength_proposal"],
        *matrix["sources"]["route_audits"],
    ):
        path = ROOT / source["path"]
        assert path.is_file(), path
        assert _sha256(path) == source["sha256"], path

    dispositions = [row["disposition"] for row in rows]
    assert dispositions.count("FROZEN_OPENED_CANDIDATE_NOT_FINAL") == 4
    assert dispositions.count("REJECT_OPENED_OBSERVABLE_ROUTING") == 6
    assert dispositions.count("REJECT_THEORY_CAPACITY_OVERLAP") == 5
    assert sum(value for key, value in matrix["counts_by_disposition"].items() if key.startswith("SHADOW_")) == 14
    assert set(matrix["frozen_opened_candidate_action_ids"]) == {
        row["action_id"]
        for row in rows
        if row["disposition"] == "FROZEN_OPENED_CANDIDATE_NOT_FINAL"
    }
    selection_passes = [
        row["action_id"]
        for row in rows
        if row.get("selection_aware_loo_gate") == "PASS_SIMULTANEOUS_LOWER_BOUND_POSITIVE"
    ]
    assert selection_passes == ["CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5"]
    for row in rows:
        if row["disposition"] == "FROZEN_OPENED_CANDIDATE_NOT_FINAL":
            assert row["oracle_capacity_gate"] == "PASS_SIMULTANEOUS_LOWER_BOUND_POSITIVE"
            assert row["oracle_simultaneous_lower_px"] > 0.0
            assert row["selection_aware_family_loo_gate"] in {
                "PASS_SIMULTANEOUS_LOWER_BOUND_POSITIVE",
                "FAIL_SIMULTANEOUS_LOWER_BOUND_NOT_POSITIVE",
            }
            strength = row["strength_proposal"]
            assert strength["mode"] == "DISCRETE_EXACT_CONTROL"
            assert strength["normalized_input_strength"] == 1.0
            assert strength["output_beta"] == 1.0
            assert strength["continuous_interpolation_authorized"] is False
            if row["family_id"] == "optical.lowpass_hf_suppression.v1":
                assert strength["opened_e3_validation"] == (
                    "REJECT_ROUTED_STRENGTH_PROPOSAL_ON_OPENED_E3"
                )
            else:
                assert strength["opened_e3_validation"] == (
                    "BINARY_NATIVE_VS_EXACT_CONTROL_ONLY"
                )

    print(
        "PASS: all 29 actions have exactly one evidence-backed disposition; "
        "4 remain opened candidates and 0 are final validated"
    )


if __name__ == "__main__":
    main()
