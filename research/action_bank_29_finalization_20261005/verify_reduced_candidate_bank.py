#!/usr/bin/env python3
"""Verify the 29-to-10 reduction ledger and its authority boundary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    source = load(ROOT / "experiments/E243_restoration_candidate_integration_v1/FROZEN_ACTION_BANK.json")
    result = load(HERE / "OPTICAL_REDUCTION_RESULT.json")
    reduced = load(HERE / "REDUCED_CANDIDATE_BANK.json")

    source_ids = {row["action_id"] for row in source["actions"]}
    candidate_ids = {
        action
        for family in reduced["candidate_families"]
        for action in family["exact_controls"]
    }
    pruned_ids = {row["action_id"] for row in reduced["overlap_pruned_controls"]}
    shadow_ids = {
        action
        for family in reduced["shadow_mechanism_groups"]
        for action in family["actions"]
    }
    expected = set(
        result["global_exact_control_reduction"]
        ["minimum_passing_subset_with_all_families"]["selected"]
    )

    assert len(source_ids) == 29
    assert len(candidate_ids) == 10
    assert len(pruned_ids) == 5
    assert len(shadow_ids) == 14
    assert candidate_ids == expected
    assert not (candidate_ids & pruned_ids)
    assert not (candidate_ids & shadow_ids)
    assert not (pruned_ids & shadow_ids)
    assert candidate_ids | pruned_ids | shadow_ids == source_ids
    assert len(reduced["candidate_families"]) == 4
    assert len(reduced["shadow_mechanism_groups"]) == 10
    assert reduced["counts"]["final_qualified_exact_controls"] == 0
    assert all(value is False for value in reduced["authority"].values())
    measured = result["global_exact_control_reduction"]["minimum_passing_subset"]
    necessity = result["selected_subset_necessity"]
    gates = reduced["reduction_gate"]
    assert measured["overall_retention_percent"] >= gates[
        "overall_oracle_capacity_retention_percent_minimum"
    ]
    assert measured["minimum_condition_retention_percent"] >= gates[
        "each_corruption_oracle_capacity_retention_percent_minimum"
    ]
    assert necessity[
        "all_selected_anchors_individually_required_for_frozen_gates"
    ] is True
    assert reduced["opened_development_result"][
        "all_selected_anchors_individually_required_for_frozen_gates"
    ] is True
    print("PASS: 29 controls partition into 10 reduced candidates, 5 overlap-pruned, and 14 shadow controls; authority remains false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
