#!/usr/bin/env python3
"""Verify the E282 low-pass discrete-strength proposal diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = Path(__file__).with_name("E282_STRENGTH_PROPOSAL_RESULT.json")
RUNS = {
    "two_strengths": (
        ROOT / "experiments/E282a_lowpass_two_strength_router_v1",
        (
            "CSB/OF/SEA-RAFT/action/P1-gaussian-s1",
            "CSB/OF/SEA-RAFT/action/R2-gaussian-s2",
        ),
    ),
    "sigma1_only": (
        ROOT / "experiments/E282b_lowpass_sigma1_router_v1",
        ("CSB/OF/SEA-RAFT/action/P1-gaussian-s1",),
    ),
    "sigma2_only": (
        ROOT / "experiments/E282c_lowpass_sigma2_router_v1",
        ("CSB/OF/SEA-RAFT/action/R2-gaussian-s2",),
    ),
}
NATIVE = "CSB/OF/SEA-RAFT/action/P0"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    analysis = _load(ANALYSIS)
    assert analysis["status"] == "REJECT_ROUTED_STRENGTH_PROPOSAL_ON_OPENED_E3"
    assert not any(analysis["authority"].values())
    assert analysis["bootstrap"]["unit"] == "physical_scene"
    assert analysis["bootstrap"]["scene_count"] == 30
    assert analysis["bootstrap"]["draws"] == 10_000
    assert analysis["two_strength_policy_value_passes_for_both_anchors"] is False
    assert analysis["two_strength_oracle_value_passes_for_both_anchors"] is True

    reference_ids = None
    reference_folds = None
    for name, (directory, actions) in RUNS.items():
        protocol = _load(directory / "PROTOCOL.json")
        result = _load(directory / "RESULT.json")
        rows = _rows(directory / "PREDICTIONS.jsonl")
        evidence = analysis["runs"][name]
        assert tuple(protocol["action_ids"]) == (NATIVE, *actions)
        assert tuple(protocol["subset_design"]["included_action_ids"]) == actions
        assert protocol["exposure_boundary"]["adaptive_e3_reuse"] is True
        assert result["status"] == "complete" and result["diagnostic_only"] is True
        assert tuple(result["action_ids"]) == (NATIVE, *actions)
        assert len(rows) == 1200
        row_ids = [row["row_id"] for row in rows]
        folds = [row["outer_fold"] for row in rows]
        if reference_ids is None:
            reference_ids, reference_folds = row_ids, folds
        assert row_ids == reference_ids and folds == reference_folds
        assert evidence["protocol_sha256"] == _sha256(directory / "PROTOCOL.json")
        assert evidence["predictions_sha256"] == _sha256(
            directory / "PREDICTIONS.jsonl"
        )

    assert analysis["runs"]["two_strengths"]["interventions"] == 0
    assert analysis["runs"]["sigma1_only"]["interventions"] == 0
    assert analysis["runs"]["sigma2_only"]["interventions"] == 17
    for comparison in analysis["comparisons"].values():
        assert comparison["two_strength_policy_value_passes"] is False
        assert comparison["two_strength_oracle_value_passes"] is True
        assert (
            comparison[
                "single_minus_two_strength_oracle_bonferroni_one_sided_95pct_lower_px"
            ]
            > 0.0
        )

    print(
        "PASS: E282 shows complementary Gaussian oracle capacity but rejects "
        "the current routed two-strength proposal on opened E3"
    )


if __name__ == "__main__":
    main()
