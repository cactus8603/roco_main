#!/usr/bin/env python3
"""Verify E279 LOO sources, pairing, authority, and frozen decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = Path(__file__).with_name("E279_SELECTION_AWARE_LOO_RESULT.json")
FULL = ROOT / "experiments/E278_minimal_stable_observable_bank_v1"
LOO = {
    "CSB/OF/SEA-RAFT/action/P1-gaussian-s1": ROOT / "experiments/E279a_selection_aware_loo_p1_v1",
    "CSB/OF/SEA-RAFT/action/R2-gaussian-s2": ROOT / "experiments/E279b_selection_aware_loo_r2_v1",
    "CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5": ROOT / "experiments/E279c_selection_aware_loo_r4_v1",
    "CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1": ROOT / "experiments/E279d_selection_aware_loo_p3_v1",
}
NATIVE = "CSB/OF/SEA-RAFT/action/P0"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    analysis = _load(ANALYSIS)
    assert analysis["status"] == "COMPLETE_OPENED_DEVELOPMENT_METHOD_DIAGNOSTIC"
    assert not any(analysis["authority"].values())
    assert analysis["bootstrap"]["unit"] == "physical_scene"
    assert analysis["bootstrap"]["scene_count"] == 30
    assert analysis["bootstrap"]["draws"] == 10_000

    full_result = _load(FULL / "RESULT.json")
    full_ids = tuple(full_result["action_ids"])
    full_rows = _rows(FULL / "PREDICTIONS.jsonl")
    full_row_ids = [row["row_id"] for row in full_rows]
    full_folds = [row["outer_fold"] for row in full_rows]
    assert full_ids[0] == NATIVE and len(full_ids) == 5
    assert analysis["full_bank"]["predictions_sha256"] == _sha256(
        FULL / "PREDICTIONS.jsonl"
    )

    for excluded, directory in LOO.items():
        bank = _load(directory / "CANDIDATE_BANK.json")
        protocol = _load(directory / "PROTOCOL.json")
        result = _load(directory / "RESULT.json")
        rows = _rows(directory / "PREDICTIONS.jsonl")
        expected = tuple(action for action in full_ids if action != excluded)
        comparison = analysis["comparisons"][excluded]
        assert bank["excluded_action_id"] == excluded
        assert bank["status"] == "OPENED_DEVELOPMENT_METHOD_DIAGNOSTIC_ONLY"
        assert not any(bank["authority"].values())
        assert protocol["loo_design"]["excluded_action_id"] == excluded
        assert protocol["exposure_boundary"]["adaptive_e3_reuse"] is True
        assert tuple(protocol["action_ids"]) == expected
        assert result["status"] == "complete" and result["diagnostic_only"] is True
        assert tuple(result["action_ids"]) == expected
        assert [row["row_id"] for row in rows] == full_row_ids
        assert [row["outer_fold"] for row in rows] == full_folds
        assert comparison["loo_predictions_sha256"] == _sha256(
            directory / "PREDICTIONS.jsonl"
        )

    assert analysis["selection_aware_nonredundant_actions_on_opened_e3"] == [
        "CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5"
    ]
    assert set(analysis["oracle_capacity_nonredundant_actions_on_opened_e3"]) == set(LOO)
    for comparison in analysis["comparisons"].values():
        assert comparison["oracle_corrupt_macro_bonferroni_one_sided_95pct_lower_px"] > 0

    print(
        "PASS: E279 pairs four nested grouped LOO retrainings; only R4 has "
        "selection-aware E3 support, while all four add oracle capacity"
    )


if __name__ == "__main__":
    main()
