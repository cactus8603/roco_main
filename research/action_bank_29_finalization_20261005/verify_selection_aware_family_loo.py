#!/usr/bin/env python3
"""Verify family-level LOO sources, pairing, authority, and decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = Path(__file__).with_name("E281_SELECTION_AWARE_FAMILY_LOO_RESULT.json")
FULL = ROOT / "experiments/E278_minimal_stable_observable_bank_v1"
LOWPASS = ROOT / "experiments/E281_selection_aware_family_loo_lowpass_v1"
FAMILY_ACTIONS = {
    "optical.lowpass_hf_suppression.v1": (
        "CSB/OF/SEA-RAFT/action/P1-gaussian-s1",
        "CSB/OF/SEA-RAFT/action/R2-gaussian-s2",
    ),
    "optical.detail_recovery_unsharp.v1": (
        "CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5",
    ),
    "optical.joint_radiometry.v1": (
        "CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1",
    ),
}
LOO = {
    "optical.lowpass_hf_suppression.v1": LOWPASS,
    "optical.detail_recovery_unsharp.v1": (
        ROOT / "experiments/E279c_selection_aware_loo_r4_v1"
    ),
    "optical.joint_radiometry.v1": (
        ROOT / "experiments/E279d_selection_aware_loo_p3_v1"
    ),
}


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
    assert analysis["full_bank"]["predictions_sha256"] == _sha256(
        FULL / "PREDICTIONS.jsonl"
    )

    for family_id, directory in LOO.items():
        excluded = FAMILY_ACTIONS[family_id]
        expected = tuple(action for action in full_ids if action not in excluded)
        protocol = _load(directory / "PROTOCOL.json")
        result = _load(directory / "RESULT.json")
        rows = _rows(directory / "PREDICTIONS.jsonl")
        comparison = analysis["comparisons"][family_id]
        assert tuple(comparison["excluded_action_ids"]) == excluded
        assert tuple(protocol["action_ids"]) == expected
        assert protocol["exposure_boundary"]["adaptive_e3_reuse"] is True
        if family_id == "optical.lowpass_hf_suppression.v1":
            bank = _load(directory / "CANDIDATE_BANK.json")
            assert bank["excluded_family_id"] == family_id
            assert tuple(bank["excluded_action_ids"]) == excluded
            assert protocol["family_loo_design"]["excluded_family_id"] == family_id
            assert tuple(protocol["family_loo_design"]["excluded_action_ids"]) == excluded
        else:
            assert protocol["loo_design"]["excluded_action_id"] == excluded[0]
        assert result["status"] == "complete" and result["diagnostic_only"] is True
        assert tuple(result["action_ids"]) == expected
        assert [row["row_id"] for row in rows] == full_row_ids
        assert [row["outer_fold"] for row in rows] == full_folds
        assert comparison["loo_predictions_sha256"] == _sha256(
            directory / "PREDICTIONS.jsonl"
        )

    expected_policy = {
        "optical.lowpass_hf_suppression.v1",
        "optical.detail_recovery_unsharp.v1",
    }
    assert set(analysis["selection_aware_nonredundant_families_on_opened_e3"]) == (
        expected_policy
    )
    assert set(analysis["oracle_capacity_nonredundant_families_on_opened_e3"]) == set(
        FAMILY_ACTIONS
    )
    for family_id, comparison in analysis["comparisons"].items():
        assert comparison["oracle_corrupt_macro_bonferroni_one_sided_95pct_lower_px"] > 0
        assert comparison["selection_aware_nonredundancy_passes"] == (
            family_id in expected_policy
        )

    print(
        "PASS: paired nested family-LOO supports low-pass and unsharp routing "
        "value on opened E3; all three families add oracle capacity"
    )


if __name__ == "__main__":
    main()
