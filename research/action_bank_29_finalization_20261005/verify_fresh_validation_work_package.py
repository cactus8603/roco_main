#!/usr/bin/env python3
"""Verify the frozen action-bank fresh-validation work package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).with_name("FRESH_VALIDATION_WORK_PACKAGE.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    package = _load(PACKAGE)
    assert package["status"] == (
        "FROZEN_METHOD_WITH_REJECT_ONLY_FRESH_PANEL_WAITING_FOR_"
        "POWERED_EXTENSION_AND_OUTCOMES"
    )
    assert not any(package["authority"].values())
    candidates = tuple(package["candidate_action_ids"])
    assert len(candidates) == len(set(candidates)) == 4
    assert package["native_action_id"] not in candidates
    assert package["primary_confirmation_action_ids"] == [
        "CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5"
    ]
    assert set(package["shadow_hypothesis_action_ids"]) == set(candidates) - set(
        package["primary_confirmation_action_ids"]
    )
    family_members = {
        action
        for members in package["candidate_families"].values()
        for action in members
    }
    assert family_members == set(candidates)

    run_ids = [run["run_id"] for run in package["unique_router_runs"]]
    assert len(run_ids) == len(set(run_ids)) == 9
    assert set(run_ids) == {
        "FULL4",
        "LOO_P1",
        "LOO_R2",
        "LOO_R4",
        "LOO_P3",
        "LOO_FAMILY_LOWPASS",
        "LOWPASS_TWO_STRENGTHS",
        "LOWPASS_SIGMA1_ONLY",
        "LOWPASS_SIGMA2_ONLY",
    }
    for run in package["unique_router_runs"]:
        assert run["action_ids"]
        assert set(run["action_ids"]).issubset(candidates)
        assert len(run["action_ids"]) == len(set(run["action_ids"]))
    primary_run = package["primary_confirmation_run"]
    assert primary_run["run_id"] == "PRIMARY_R4"
    assert primary_run["action_ids"] == package["primary_confirmation_action_ids"]
    assert package["comprehensive_diagnostic_run_matrix_authority"] == (
        "optional mechanism diagnostics only; cannot admit shadow hypotheses"
    )

    cv = package["cross_validation"]
    assert cv["default_outer_fold_count"] == 5
    assert cv["default_inner_fold_count"] == 4
    assert cv["minimum_fold_count_supported_by_trainer"] == 3
    gates = package["global_full_bank_gates"]
    assert gates["corrupt_macro_relative_gain_vs_native_min"] == 0.05
    assert gates["clean_relative_degradation_max"] == 0.02
    assert gates["catastrophe_per_intervention_max"] == 0.005
    assert gates["normative_corruptions_improved_min"] == 12
    assert gates["normative_corruption_count"] == 20

    for source in package["frozen_sources"].values():
        path = ROOT / source["path"]
        assert path.is_file(), path
        assert _sha256(path) == source["sha256"], path

    preflight_source = package["frozen_sources"]["gpu_action_preflight"]
    preflight = _load(ROOT / preflight_source["path"])
    assert preflight["status"] == "complete"
    assert preflight["evidence_level"] == "E3-execution-only"
    assert preflight["action_ids"] == [package["native_action_id"], *candidates]
    assert preflight["checks"]["cuda_visible"] is True
    assert preflight["checks"]["all_actions_shape_dtype_finite"] is True
    assert preflight["checks"]["actions_and_inference_completed_before_gt_load"] is True
    assert preflight["checks"]["fresh_panel_used"] is False
    assert preflight["checks"]["authorizes_final_admission"] is False

    panel_source = package["frozen_sources"]["external_panel_phase1"]
    panel = _load(ROOT / panel_source["path"])
    assert panel["status"] == "FROZEN_PRE_OUTCOME_DECODE_PHASE1_REJECT_ONLY"
    assert panel["panel"]["independent_group_count"] == 20
    assert panel["panel"]["physical_pair_count"] == 160
    assert panel["panel"]["expanded_row_count"] == 9760
    assert panel["power_audit"]["powered_for_final_admission"] is False
    assert panel["authority"]["decode_official_flow"] is False
    assert panel["exposure_audit"]["current_members_overlap_with_previously_decoded_members"] == 0

    amendment_source = package["frozen_sources"]["external_panel_phase1_protocol_amendment"]
    amendment = _load(ROOT / amendment_source["path"])
    assert amendment["status"] == "FROZEN_PRE_OUTCOME_CLARIFICATION_NO_NEW_AUTHORITY"
    assert amendment["unchanged_requirements"]["all_action_predictions_generated_before_any_official_flow_decode"] is True
    assert amendment["storage_interpretation"]["dense_predictions_must_be_retained"] is False
    assert not any(amendment["authority"].values())

    options_source = package["frozen_sources"]["powered_panel_options_audit"]
    options = _load(ROOT / options_source["path"])
    assert options["status"] == "AUDITED_WAITING_FOR_AUTHORITY"
    by_id = {item["option_id"]: item for item in options["options"]}
    assert by_id["E284_TARTANAIR_PHASE1"]["independent_groups"] == 20
    assert by_id["KITTI_H2"]["independent_groups"] == 70
    assert by_id["KITTI_H2"]["currently_frozen_corruption_count"] == 5
    assert by_id["KITTI_H2"]["required_normative_corruption_count"] == 20
    assert by_id["KITTI_H2"]["normative_corruption_coverage_satisfied"] is False
    assert by_id["KITTI_H2"]["current_h2_execution_authorized"] is False
    assert by_id["KITTI_H2"]["current_access_counts"] == {
        "images_decoded": 0,
        "model_forwards": 0,
        "truth_reads": 0,
    }

    shortlist_source = package["frozen_sources"]["primary_confirmation_shortlist"]
    shortlist = _load(ROOT / shortlist_source["path"])
    assert shortlist["status"] == "FROZEN_PRE_OUTCOME_R4_ONLY_NOT_FINAL"
    assert shortlist["primary_confirmation_action_ids"] == package["primary_confirmation_action_ids"]
    assert not any(shortlist["authority"].values())

    receipt_source = package["frozen_sources"]["external_panel_r4_preoutcome_receipt"]
    receipt = _load(ROOT / receipt_source["path"])
    assert receipt["status"] == "COMPLETE_PRE_OUTCOME_R4_PANEL_RECEIPTS"
    assert receipt["row_count"] == 9760
    assert receipt["independent_group_count"] == 20
    assert receipt["action_ids"] == [
        package["native_action_id"], *package["primary_confirmation_action_ids"]
    ]
    assert receipt["feature_count"] == 60
    assert receipt["checks"] == {
        "all_frozen_rows_exactly_once": True,
        "all_rows_target_free": True,
        "before_only_features_complete_and_finite": True,
        "dense_predictions_retained": False,
        "model_visible_payload_whitelist": True,
        "native_and_r4_prediction_content_hashes_complete": True,
        "official_flow_paths_absent_from_receipts": True,
    }
    assert not any(receipt["authority"].values())

    reproducibility_source = package["frozen_sources"]["external_panel_r4_reproducibility_audit"]
    reproducibility = _load(ROOT / reproducibility_source["path"])
    assert reproducibility["status"] == "PASS_EXACT_CONTENT_REPRODUCTION"
    assert all(reproducibility["exact_matches"].values())
    assert not any(reproducibility["authority"].values())

    readiness = package["execution_readiness"]
    assert readiness["method_and_run_matrix_frozen"] is True
    assert readiness["primary_r4_confirmation_plan_frozen"] is True
    assert readiness["dynamic_trainer_parity_verified"] is True
    assert readiness["gpu_action_preflight_verified"] is True
    assert readiness["fresh_panel_manifest_available"] is True
    assert readiness["fresh_panel_independent_group_count"] == 20
    assert readiness["fresh_panel_powered_for_final_admission"] is False
    assert readiness["fresh_panel_permitted_authority"] == "reject_only"
    assert readiness["sealed_h2_independent_group_count"] == 70
    assert readiness["sealed_h2_meets_smallest_finite_planning_count"] is True
    assert readiness["sealed_h2_current_corruption_count"] == 5
    assert readiness["sealed_h2_normative_corruption_coverage_satisfied"] is False
    assert readiness["sealed_h2_execution_authorized"] is False
    assert readiness["powered_authorized_panel_available"] is False
    assert readiness["fresh_before_only_feature_records_available"] is True
    assert readiness["fresh_native_r4_prediction_receipts_available"] is True
    assert readiness["fresh_preoutcome_row_count"] == 9760
    assert readiness["fresh_exact_action_outcomes_available"] is False
    assert readiness["gpu_visible_in_current_environment"] is True
    assert readiness["gpu_access_boundary"] == (
        "CUDA is visible only outside the filesystem sandbox; GPU execution "
        "requires an escalated command"
    )
    assert readiness["ready_to_execute"] is False
    print(
        "PASS: R4-only primary confirmation, optional nine-run diagnostics, "
        "GPU preflight, and the 20-group reject-only panel are hash-bound; "
        "sealed H2 has 70 groups but only five frozen corruptions and no "
        "execution authority; final admission remains closed"
    )


if __name__ == "__main__":
    main()
