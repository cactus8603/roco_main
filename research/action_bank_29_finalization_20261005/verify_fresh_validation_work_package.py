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
        "FROZEN_METHOD_WITH_REJECT_ONLY_PHASE1_AND_POWERED_H2_FINAL_"
        "PROTOCOL_WAITING_FOR_AUTHORITY_AND_OUTCOMES"
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
    assert options["status"] == (
        "AUDITED_POWERED_R4_PROTOCOL_FROZEN_WAITING_FOR_AUTHORITY_AND_PHASE1"
    )
    by_id = {item["option_id"]: item for item in options["options"]}
    assert by_id["E284_TARTANAIR_PHASE1"]["independent_groups"] == 20
    assert by_id["KITTI_H2"]["independent_groups"] == 70
    assert by_id["KITTI_H2"]["currently_frozen_corruption_count"] == 5
    assert by_id["KITTI_H2"]["required_normative_corruption_count"] == 20
    assert by_id["KITTI_H2"]["normative_corruption_coverage_satisfied"] is False
    assert by_id["KITTI_H2"]["action_bank_final_protocol_frozen"] is True
    assert by_id["KITTI_H2"]["action_bank_final_protocol_corruption_count"] == 20
    assert by_id["KITTI_H2"]["action_bank_final_protocol_normative_coverage_satisfied"] is True
    assert by_id["KITTI_H2"]["action_bank_final_protocol_execution_authorized"] is False
    assert by_id["KITTI_H2"]["action_bank_exact_panel_frozen"] is True
    assert by_id["KITTI_H2"]["action_bank_exact_panel_row_count"] == 4270
    assert by_id["KITTI_H2"]["action_bank_exact_panel_payloads_opened"] is False
    assert by_id["KITTI_H2"]["action_bank_preoutcome_runner_frozen"] is True
    assert by_id["KITTI_H2"]["action_bank_preoutcome_fail_closed_tests_passed"] is True
    assert by_id["KITTI_H2"]["action_bank_preoutcome_execution_authorized"] is False
    assert by_id["KITTI_H2"]["current_h2_execution_authorized"] is False
    assert by_id["KITTI_H2"]["current_access_counts"] == {
        "images_decoded": 0,
        "model_forwards": 0,
        "truth_reads": 0,
    }

    h2_protocol_source = package["frozen_sources"]["powered_h2_r4_final_protocol"]
    h2_protocol = _load(ROOT / h2_protocol_source["path"])
    assert h2_protocol["status"] == "FROZEN_METADATA_ONLY_H2_ACCESS_NOT_AUTHORIZED"
    assert h2_protocol["sealed_panel"]["scene_count"] == 70
    assert h2_protocol["sealed_panel"]["corruption_count"] == 20
    assert h2_protocol["sealed_panel"]["expected_row_count"] == 4270
    assert h2_protocol["power_contract"]["powered_for_r4_positive_admission"] is True
    assert h2_protocol["power_contract"]["powered_for_shadow_action_admission"] is False
    assert not any(h2_protocol["authority"].values())

    h2_panel_source = package["frozen_sources"]["powered_h2_r4_exact_panel"]
    h2_panel = _load(ROOT / h2_panel_source["path"])
    assert h2_panel["status"] == "FROZEN_METADATA_ONLY_H2_ACCESS_NOT_AUTHORIZED"
    assert h2_panel["panel"]["independent_group_count"] == 70
    assert h2_panel["panel"]["expanded_row_count"] == 4270
    assert h2_panel["panel"]["groups_per_outer_fold"] == 14
    assert h2_panel["exposure_audit"]["h2_image_payloads_opened_during_freeze"] is False
    assert h2_panel["exposure_audit"]["h2_truth_payloads_opened_during_freeze"] is False
    assert h2_panel["exposure_audit"]["h2_model_forwards_during_freeze"] == 0
    assert not any(h2_panel["authority"].values())

    h2_preoutcome_source = package["frozen_sources"]["powered_h2_r4_preoutcome_protocol"]
    h2_preoutcome = _load(ROOT / h2_preoutcome_source["path"])
    assert h2_preoutcome["status"] == "FROZEN_PREOUTCOME_NOT_AUTHORIZED"
    assert h2_preoutcome["execution"]["total_rows"] == 4270
    assert h2_preoutcome["execution"]["expected_total_model_forwards"] == 17080
    assert h2_preoutcome["authorization_contract"]["current_authorized"] is False
    assert not any(h2_preoutcome["authority"].values())
    h2_preoutcome_tests_source = package["frozen_sources"]["powered_h2_r4_preoutcome_tests"]
    h2_preoutcome_tests = _load(ROOT / h2_preoutcome_tests_source["path"])
    assert h2_preoutcome_tests["status"] == "PASS"
    assert h2_preoutcome_tests["tests_run"] == 4
    assert h2_preoutcome_tests["failures"] == h2_preoutcome_tests["errors"] == 0
    assert h2_preoutcome_tests["h2_image_payloads_opened_by_tests"] == 0
    assert h2_preoutcome_tests["h2_truth_payloads_opened_by_tests"] == 0
    assert h2_preoutcome_tests["h2_model_forwards_by_tests"] == 0
    assert not any(h2_preoutcome_tests["authority"].values())

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

    leakage_source = package["frozen_sources"]["dynamic_router_outer_label_invariance"]
    leakage = _load(ROOT / leakage_source["path"])
    assert leakage["status"] == "PASS_NO_OUTER_TARGET_LEAKAGE_DETECTED"
    assert leakage["held_out_rows_perturbed"] == leakage["held_out_rows_compared"] == 120
    assert leakage["held_out_evaluation_values_changed"] == 120
    assert leakage["held_out_model_outputs_exactly_equal"] is True
    assert leakage["held_out_action_indices_exactly_equal"] is True
    assert leakage["held_out_model_sha256_exactly_equal"] is True
    assert leakage["held_out_thresholds_calibration_and_nested_training_exactly_equal"] is True
    assert leakage["other_fold_models_changed"] == 5
    assert not any(leakage["authority"].values())
    for source in leakage["sources"].values():
        source_path = Path(source["path"])
        assert source_path.is_file(), source_path
        assert _sha256(source_path) == source["sha256"], source_path

    scoring_source = package["frozen_sources"]["external_panel_r4_reject_only_scoring_protocol"]
    scoring = _load(ROOT / scoring_source["path"])
    assert scoring["status"] == "FROZEN_PRE_OUTCOME_NOT_AUTHORIZED"
    assert scoring["outcome_stage"]["current_authorized"] is False
    assert scoring["outcome_stage"]["current_official_flow_payloads_opened_by_e286"] == 0
    assert scoring["phase1_decision"]["positive_admission_from_e286"] is False
    assert not any(scoring["authority"].values())

    scoring_tests_source = package["frozen_sources"]["external_panel_r4_scoring_tests"]
    scoring_tests = _load(ROOT / scoring_tests_source["path"])
    assert scoring_tests["status"] == "PASS"
    assert scoring_tests["tests_run"] == 7
    assert scoring_tests["failures"] == scoring_tests["errors"] == 0
    assert scoring_tests["current_sealed_panel_members_opened_by_tests"] == 0
    assert not any(scoring_tests["authority"].values())

    readiness = package["execution_readiness"]
    assert readiness["method_and_run_matrix_frozen"] is True
    assert readiness["primary_r4_confirmation_plan_frozen"] is True
    assert readiness["dynamic_trainer_parity_verified"] is True
    assert readiness["dynamic_trainer_outer_target_isolation_verified"] is True
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
    assert readiness["powered_r4_final_protocol_frozen"] is True
    assert readiness["powered_r4_final_protocol_independent_group_count"] == 70
    assert readiness["powered_r4_final_protocol_normative_corruption_count"] == 20
    assert readiness["powered_r4_final_protocol_normative_coverage_satisfied"] is True
    assert readiness["powered_r4_final_protocol_execution_authorized"] is False
    assert readiness["powered_r4_exact_panel_frozen"] is True
    assert readiness["powered_r4_exact_panel_row_count"] == 4270
    assert readiness["powered_r4_exact_panel_payloads_opened"] is False
    assert readiness["powered_r4_preoutcome_runner_frozen"] is True
    assert readiness["powered_r4_preoutcome_fail_closed_tests_passed"] is True
    assert readiness["powered_r4_preoutcome_execution_authorized"] is False
    assert readiness["powered_authorized_panel_available"] is False
    assert readiness["fresh_before_only_feature_records_available"] is True
    assert readiness["fresh_native_r4_prediction_receipts_available"] is True
    assert readiness["fresh_preoutcome_row_count"] == 9760
    assert readiness["fresh_outcome_scoring_protocol_frozen"] is True
    assert readiness["fresh_outcome_decode_authorized"] is False
    assert readiness["fresh_exact_action_outcomes_available"] is False
    assert readiness["gpu_visible_in_current_environment"] is True
    assert readiness["gpu_access_boundary"] == (
        "CUDA is visible only outside the filesystem sandbox; GPU execution "
        "requires an escalated command"
    )
    assert readiness["ready_to_execute"] is False
    print(
        "PASS: R4-only primary confirmation, optional nine-run diagnostics, "
        "GPU preflight, outer-target isolation, and the 20-group reject-only "
        "panel are hash-bound; E288-E290 freeze a powered 70-scene, "
        "20-corruption, 4,270-row R4 final panel and target-free runner but "
        "H2 has no execution authority; final admission remains closed"
    )


if __name__ == "__main__":
    main()
