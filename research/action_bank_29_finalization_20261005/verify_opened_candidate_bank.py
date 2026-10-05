#!/usr/bin/env python3
"""Fail-closed verification for the frozen opened-development candidate bank."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name("OPENED_CANDIDATE_ACTION_BANK.json")
FINAL_PATH = Path(__file__).with_name("FINAL_VALIDATED_ACTION_BANK.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)


def main() -> None:
    from stablebridge.physical_repair.candidate_action_bank import (
        OPTICAL_FLOW_CAPACITY_ACTION_IDS,
        OPTICAL_FLOW_CAPACITY_BANK,
        OPTICAL_FLOW_CAPACITY_BANK_HASH,
        OPTICAL_FLOW_CAPACITY_FAMILIES,
        OPTICAL_FLOW_CAPACITY_SOURCE_MANIFEST_SHA256,
        OPTICAL_FLOW_OPENED_CANDIDATE_ACTION_IDS,
        OPTICAL_FLOW_OPENED_CANDIDATE_BANK,
        OPTICAL_FLOW_OPENED_CANDIDATE_BANK_HASH,
        OPTICAL_FLOW_OPENED_CANDIDATE_FAMILIES,
        OPTICAL_NATIVE_ACTION_ID,
    )

    manifest = _load(MANIFEST_PATH)
    final = _load(FINAL_PATH)
    assert manifest["status"] == (
        "FROZEN_OPENED_DEVELOPMENT_CANDIDATE_REQUIRES_FRESH_CONFIRMATION"
    )
    assert not any(manifest["authority"].values())
    assert final["status"] == "EMPTY_PENDING_FRESH_SELECTION_AWARE_VALIDATION"
    assert final["validated_action_ids"] == []
    assert final["family_count"] == final["exact_control_count"] == 0
    assert not any(final["authority"].values())
    assert final["frozen_candidate_source"]["sha256"] == _sha256(MANIFEST_PATH)
    capacity = final["cross_component_capacity_bank"]
    capacity_path = ROOT / capacity["path"]
    assert capacity["sha256"] == _sha256(capacity_path)
    runtime_capacity = _load(capacity_path)
    assert capacity["source_manifest_sha256"] == (
        runtime_capacity["source_manifest_sha256"]
    )
    assert capacity["status"] == "FINAL_OPENED_CROSS_COMPONENT_CAPACITY_BANK"
    assert capacity["authority"] == "capacity_only_not_selector_or_production"
    assert runtime_capacity["exact_anchor_count"] == 9
    assert runtime_capacity["family_count"] == 4
    assert runtime_capacity["authority"]["production_authority"] is False
    assert runtime_capacity["bank_hash"] == OPTICAL_FLOW_CAPACITY_BANK_HASH
    assert runtime_capacity["source_manifest_sha256"] == (
        OPTICAL_FLOW_CAPACITY_SOURCE_MANIFEST_SHA256
    )
    assert runtime_capacity["families"] == {
        family: list(action_ids)
        for family, action_ids in OPTICAL_FLOW_CAPACITY_FAMILIES.items()
    }
    assert tuple(
        action_id
        for action_ids in OPTICAL_FLOW_CAPACITY_FAMILIES.values()
        for action_id in action_ids
    ) == OPTICAL_FLOW_CAPACITY_ACTION_IDS
    assert set(OPTICAL_FLOW_CAPACITY_BANK) == set(OPTICAL_FLOW_CAPACITY_ACTION_IDS)
    assert manifest["native_fallback_action_id"] == OPTICAL_NATIVE_ACTION_ID
    assert manifest["bank_hash"] == OPTICAL_FLOW_OPENED_CANDIDATE_BANK_HASH

    families = manifest["families"]
    manifest_family_map = {
        family["family_id"]: tuple(
            control["action_id"] for control in family["exact_controls"]
        )
        for family in families
    }
    manifest_ids = tuple(
        action_id
        for members in manifest_family_map.values()
        for action_id in members
    )
    assert manifest["family_count"] == len(families) == 3
    assert manifest["exact_control_count"] == len(manifest_ids) == 4
    assert manifest_family_map == dict(OPTICAL_FLOW_OPENED_CANDIDATE_FAMILIES)
    assert manifest_ids == OPTICAL_FLOW_OPENED_CANDIDATE_ACTION_IDS
    assert set(manifest_ids) == set(OPTICAL_FLOW_OPENED_CANDIDATE_BANK)

    for family in families:
        for control in family["exact_controls"]:
            arm = OPTICAL_FLOW_OPENED_CANDIDATE_BANK[control["action_id"]]
            payload = arm.selector_payload(include_hash=True)
            assert control["operator"] == payload["operator"]["id"]
            payload_parameters = json.loads(
                json.dumps(payload["operator"]["parameters"], sort_keys=True)
            )
            assert control["operator_parameters"] == payload_parameters
            assert control["endpoint"] == payload["endpoint"]
            assert control["support"] == payload["support"]
            assert control["availability"] == payload["availability"]
            assert not payload["receipt"]["production_authority"]

    for artifact in manifest["artifacts"].values():
        path = ROOT / artifact["path"]
        assert path.is_file(), path
        assert _sha256(path) == artifact["sha256"], path

    e278 = ROOT / "experiments/E278_minimal_stable_observable_bank_v1"
    candidate = _load(e278 / "CANDIDATE_BANK.json")
    protocol = _load(e278 / "PROTOCOL.json")
    result = _load(e278 / "RESULT.json")
    audit = _load(e278 / "ACTION_ROUTE_AUDIT.json")
    expected_with_native = (OPTICAL_NATIVE_ACTION_ID, *manifest_ids)

    assert not any(candidate["authority"].values())
    assert tuple(protocol["action_ids"]) == expected_with_native
    assert tuple(result["action_ids"]) == expected_with_native
    assert result["status"] == "complete"
    assert result["diagnostic_only"] is True
    assert result["target_access_for_controller"] == "none"
    assert result["condition_label_model_visible"] is False

    summary = result["variant_summaries"]["CTRL-FACT"]
    interval = result["bootstrap"]["intervals"]["CTRL-FACT"]
    evidence = manifest["opened_development_validation"]
    assert _close(
        summary["corrupt_relative_gain_vs_p0"],
        evidence["corrupt_relative_gain_vs_native"],
    )
    assert _close(
        summary["clean_relative_degradation"], evidence["clean_relative_degradation"]
    )
    assert summary["corrupt_conditions_improved"] == summary["corrupt_condition_count"] == 4
    assert interval["corrupt_relative_gain_vs_p0_95ci"][0] > 0.0
    assert interval["clean_relative_degradation_95ci"][1] <= 0.02
    assert summary["catastrophe_per_intervention"] <= 0.005

    assert audit["failing_actions"] == []
    assert tuple(audit["passing_actions_for_retrained_reduced_bank"]) == manifest_ids
    assert not any(audit["authority"].values())
    for action_id in manifest_ids:
        action = audit["actions"][action_id]
        assert action["interventions"] > 0
        assert action["net_gain_sum_raw_px"] > 0.0
        assert action[
            "scene_cluster_bootstrap_population_contribution_95_interval_raw_px"
        ][0] > 0.0

    assert manifest["strength_contract"]["mode"] == "discrete_exact_controls_only"
    assert manifest["strength_contract"]["continuous_interpolation_authorized"] is False
    assert manifest["strength_contract"]["opened_e3_strength_proposal_validation"] == (
        "REJECT_ROUTED_TWO_ANCHOR_PROPOSAL"
    )
    assert evidence["normative_twelve_of_twenty_corruption_gate_adjudicable"] is False
    loo = manifest["selection_aware_loo_diagnostic"]
    assert tuple(loo["oracle_capacity_nonredundant_action_ids"]) == manifest_ids
    assert loo["selection_aware_nonredundant_action_ids"] == [
        "CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5"
    ]
    family_loo = manifest["selection_aware_family_loo_diagnostic"]
    assert tuple(family_loo["oracle_capacity_nonredundant_family_ids"]) == tuple(
        manifest_family_map
    )
    assert family_loo["selection_aware_nonredundant_family_ids"] == [
        "optical.lowpass_hf_suppression.v1",
        "optical.detail_recovery_unsharp.v1",
    ]
    strength = manifest["strength_proposal_diagnostic"]
    assert strength["two_strength_oracle_value_passes_for_both_anchors"] is True
    assert strength["two_strength_policy_value_passes_for_both_anchors"] is False
    assert strength["decision"] == "REJECT_ROUTED_STRENGTH_PROPOSAL_ON_OPENED_E3"
    print(
        "PASS: 4-control admission candidate and 9-anchor capacity bank are "
        "hash-bound; final validated bank is empty"
    )


if __name__ == "__main__":
    main()
