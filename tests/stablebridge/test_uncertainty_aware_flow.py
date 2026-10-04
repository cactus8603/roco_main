from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from stablebridge.physical_repair.uncertainty_aware_flow import (
    BoundedUncertaintyFlowRefinerV1,
    DecoupledTrainingPolicyV1,
    FlowChangeAuthorizationReceiptV1,
    UncertaintyAvailabilityV1,
    UncertaintyClaimSemanticsV1,
    UncertaintyEvidenceRoleV1,
    UncertaintyFlowModeV1,
    UncertaintyMapSemanticsV1,
    UncertaintyReceiptV1,
    UncertaintyRuntimePolicyV1,
    UncertaintyTrainingStageV1,
    augmentation_consistency_laplace_loss_v1,
    bounded_uncertainty_guided_flow_v1,
    summarize_uncertainty_v1,
    uncertainty_reliability_v1,
    uncertainty_update_gate_v1,
)
from stablebridge.physical_repair.uncertainty_family_competition import (
    UncertaintyBoundPlannerScoreV1,
    postaction_uncertainty_delta_v1,
    preaction_uncertainty_artifact_v1,
)
from stablebridge.physical_repair.selector_v7 import (
    PlannerCandidateV7,
    PredictionKindV7,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def uncertainty_receipt(role: str, marker: str = "native") -> UncertaintyReceiptV1:
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    summary = summarize_uncertainty_v1(
        values, np.ones((4, 4), bool), high_uncertainty_threshold=12.0,
    )
    return UncertaintyReceiptV1(
        task="flow",
        observation_role=role,
        provider_id="work-b-head-v1",
        provider_checkpoint_hash=digest("checkpoint"),
        input_hashes=(digest(f"first:{marker}"), digest(f"second:{marker}")),
        flow_hash=digest(f"flow:{marker}"),
        map_hash=digest(f"map:{marker}"),
        map_shape=(4, 4),
        semantics=UncertaintyMapSemanticsV1.LAPLACE_SCALE,
        iteration=None,
        availability=UncertaintyAvailabilityV1.AVAILABLE,
        availability_reason="available",
        summary=summary,
        spatial_embedding_hash=digest(f"embedding:{marker}"),
    )


def test_uncertainty_summary_and_receipt_are_sealed():
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    valid = np.ones((4, 4), bool)
    summary = summarize_uncertainty_v1(
        values, valid, high_uncertainty_threshold=12.0,
    )
    assert summary.valid_count == 16
    assert summary.p50 <= summary.p90 <= summary.p99 <= summary.maximum
    assert summary.high_uncertainty_fraction == pytest.approx(0.25)
    receipt = UncertaintyReceiptV1(
        task="flow",
        observation_role="native",
        provider_id="work-b-head-v1",
        provider_checkpoint_hash=digest("checkpoint"),
        input_hashes=(digest("first"), digest("second")),
        flow_hash=digest("flow"),
        map_hash=digest("map"),
        map_shape=(4, 4),
        semantics=UncertaintyMapSemanticsV1.LAPLACE_SCALE,
        iteration=None,
        availability=UncertaintyAvailabilityV1.AVAILABLE,
        availability_reason="available",
        summary=summary,
        spatial_embedding_hash=digest("embedding"),
    )
    assert len(receipt.receipt_hash) == 64
    with pytest.raises(ValueError, match="hash drift"):
        replace(receipt, map_hash=digest("changed"))


def test_planner_score_must_bind_native_uncertainty_artifact():
    native = uncertainty_receipt("native")
    artifact = preaction_uncertainty_artifact_v1(
        candidate_id="candidate-a",
        planned_arm_id="arm-a",
        base_before_feature_hash=digest("base-feature"),
        native_uncertainty=native,
        uncertainty_spatial_feature_hash=digest("spatial-feature"),
        feature_schema_hash=digest("schema"),
    )
    score = PlannerCandidateV7(
        candidate_id="candidate-a",
        planned_arm_id="arm-a",
        cost_hash="cost-hash",
        planner_allowlist_hash="planner-allowlist",
        before_feature_hash=artifact.artifact_hash,
        prediction_kind=PredictionKindV7.PLANNER_MODEL_PREDICTION_PREDECISION,
        planner_predicted_gain_raw_px=0.2,
        before_features_complete=True,
    )
    UncertaintyBoundPlannerScoreV1(score, artifact, artifact.artifact_hash)
    with pytest.raises(ValueError, match="did not bind"):
        UncertaintyBoundPlannerScoreV1(score, artifact, digest("wrong"))


def test_postaction_delta_requires_native_and_executed_child_roles():
    native = uncertainty_receipt("native")
    child = uncertainty_receipt("executed_child", "child")
    evidence = postaction_uncertainty_delta_v1(
        candidate_id="candidate-a",
        observation_hash=digest("observation"),
        native_uncertainty=native,
        child_uncertainty=child,
        mean_delta=-0.2,
        p90_delta=-0.1,
        p99_delta=0.3,
        resolved_fraction=0.4,
        newly_uncertain_fraction=0.1,
        delta_spatial_feature_hash=digest("delta-feature"),
    )
    assert evidence.native_uncertainty_receipt_hash == native.receipt_hash
    with pytest.raises(ValueError, match="executed-child"):
        postaction_uncertainty_delta_v1(
            candidate_id="candidate-a",
            observation_hash=digest("observation"),
            native_uncertainty=native,
            child_uncertainty=native,
            mean_delta=0,
            p90_delta=0,
            p99_delta=0,
            resolved_fraction=0,
            newly_uncertain_fraction=0,
            delta_spatial_feature_hash=digest("delta-feature"),
        )


def test_typed_missing_receipt_cannot_carry_values():
    with pytest.raises(ValueError, match="cannot carry"):
        UncertaintyReceiptV1(
            task="flow",
            observation_role="native",
            provider_id="observer",
            provider_checkpoint_hash=digest("checkpoint"),
            input_hashes=(digest("first"),),
            flow_hash=digest("flow"),
            map_hash=digest("map"),
            map_shape=None,
            semantics=UncertaintyMapSemanticsV1.LOG_VARIANCE,
            iteration=None,
            availability=UncertaintyAvailabilityV1.TYPED_MISSING,
            availability_reason="not_materialized",
            summary=None,
            spatial_embedding_hash=None,
        )


def test_observer_only_is_default_and_cannot_change_flow():
    policy = UncertaintyRuntimePolicyV1()
    assert policy.mode is UncertaintyFlowModeV1.OBSERVER_ONLY
    flow = np.zeros((2, 3, 2), np.float32)
    with pytest.raises(PermissionError, match="observer-only"):
        bounded_uncertainty_guided_flow_v1(
            flow,
            flow + 1,
            np.zeros((2, 3), np.float32),
            np.ones((2, 3), bool),
            semantics=UncertaintyMapSemanticsV1.LOG_VARIANCE,
            policy=policy,
        )


def test_bounded_refinement_obeys_norm_cap_and_base_uncertainty_budget():
    base = np.zeros((1, 2, 2), np.float32)
    proposed = np.full((1, 2, 2), 10.0, np.float32)
    valid = np.ones((1, 2), bool)
    policy = UncertaintyRuntimePolicyV1(
        mode=UncertaintyFlowModeV1.BOUNDED_REFINEMENT,
        flow_update_enabled=True,
        maximum_update_px=0.5,
        require_calibrated_provider=False,
        require_calibrated_flow_change=False,
    )
    low_u = bounded_uncertainty_guided_flow_v1(
        base, proposed, np.full((1, 2), -10.0), valid,
        semantics=UncertaintyMapSemanticsV1.LOG_VARIANCE, policy=policy,
    )
    high_u = bounded_uncertainty_guided_flow_v1(
        base, proposed, np.full((1, 2), 10.0), valid,
        semantics=UncertaintyMapSemanticsV1.LOG_VARIANCE, policy=policy,
    )
    assert np.max(np.linalg.norm(low_u - base, axis=2)) <= 0.5 + 1e-6
    assert np.linalg.norm(high_u) > np.linalg.norm(low_u)


def test_numpy_refinement_preserves_invalid_base_values_exactly():
    base = np.array([[[-0.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    proposed = base + 10.0
    valid = np.array([[False, True]])
    policy = UncertaintyRuntimePolicyV1(
        mode=UncertaintyFlowModeV1.BOUNDED_REFINEMENT,
        flow_update_enabled=True,
        maximum_update_px=0.5,
        require_calibrated_provider=False,
        require_calibrated_flow_change=False,
    )
    result = bounded_uncertainty_guided_flow_v1(
        base,
        proposed,
        np.full((1, 2), 10.0, dtype=np.float32),
        valid,
        semantics=UncertaintyMapSemanticsV1.LOG_VARIANCE,
        policy=policy,
    )
    assert result[0, 0].tobytes() == base[0, 0].tobytes()


def test_update_gate_reverses_only_when_uncertainty_describes_proposal():
    reliability = np.array([[0.1, 0.9]], np.float32)
    base_gate = uncertainty_update_gate_v1(
        reliability, UncertaintyEvidenceRoleV1.BASE_FLOW,
    )
    proposal_gate = uncertainty_update_gate_v1(
        reliability, UncertaintyEvidenceRoleV1.PROPOSED_FLOW,
    )
    condition_gate = uncertainty_update_gate_v1(
        reliability, UncertaintyEvidenceRoleV1.CONDITION_ONLY,
    )
    np.testing.assert_allclose(base_gate, 1.0 - reliability)
    np.testing.assert_allclose(proposal_gate, reliability)
    np.testing.assert_allclose(condition_gate, 1.0)


def test_flow_change_rejects_surrogate_when_calibration_is_required():
    flow = np.zeros((2, 2, 2), np.float32)
    policy = UncertaintyRuntimePolicyV1(
        mode=UncertaintyFlowModeV1.BOUNDED_REFINEMENT,
        flow_update_enabled=True,
        maximum_update_px=0.5,
    )
    with pytest.raises(PermissionError, match="not calibrated"):
        bounded_uncertainty_guided_flow_v1(
            flow,
            flow + 1,
            np.zeros((2, 2), np.float32),
            np.ones((2, 2), bool),
            semantics=UncertaintyMapSemanticsV1.LOG_VARIANCE,
            policy=policy,
            receipt=uncertainty_receipt("native"),
        )
    calibrated = replace(
        uncertainty_receipt("native"),
        claim_semantics=UncertaintyClaimSemanticsV1.CALIBRATED_ENDPOINT_ERROR,
        calibration_receipt_hash=digest("calibration"),
        receipt_hash="",
    )
    with pytest.raises(PermissionError, match="own calibrated authorization"):
        bounded_uncertainty_guided_flow_v1(
            flow,
            flow + 1,
            np.zeros((2, 2), np.float32),
            np.ones((2, 2), bool),
            semantics=UncertaintyMapSemanticsV1.LOG_VARIANCE,
            policy=policy,
            receipt=calibrated,
        )
    authorization = FlowChangeAuthorizationReceiptV1(
        proposal_provider_id="bounded-refiner-u1",
        proposal_provider_checkpoint_hash=digest("refiner-checkpoint"),
        uncertainty_provider_checkpoint_hash=calibrated.provider_checkpoint_hash,
        uncertainty_calibration_receipt_hash=calibrated.calibration_receipt_hash,
        independent_validation_split_hash=digest("validation-split"),
        flow_change_calibration_receipt_hash=digest("flow-change-calibration"),
        authorized_mode=UncertaintyFlowModeV1.BOUNDED_REFINEMENT,
        evidence_role=UncertaintyEvidenceRoleV1.BASE_FLOW,
        calibrated_maximum_update_px=0.5,
        qualified=True,
    )
    result = bounded_uncertainty_guided_flow_v1(
        flow,
        flow + 1,
        np.zeros((2, 2), np.float32),
        np.ones((2, 2), bool),
        semantics=UncertaintyMapSemanticsV1.LOG_VARIANCE,
        policy=policy,
        receipt=calibrated,
        flow_change_authorization=authorization,
    )
    assert result.shape == flow.shape


def test_flow_change_authorization_is_bound_to_policy_and_u0_provider():
    calibrated = replace(
        uncertainty_receipt("native"),
        claim_semantics=UncertaintyClaimSemanticsV1.CALIBRATED_ENDPOINT_ERROR,
        calibration_receipt_hash=digest("calibration"),
        receipt_hash="",
    )
    policy = UncertaintyRuntimePolicyV1(
        mode=UncertaintyFlowModeV1.BOUNDED_REFINEMENT,
        flow_update_enabled=True,
        maximum_update_px=0.5,
    )
    wrong_provider = FlowChangeAuthorizationReceiptV1(
        proposal_provider_id="bounded-refiner-u1",
        proposal_provider_checkpoint_hash=digest("refiner-checkpoint"),
        uncertainty_provider_checkpoint_hash=digest("other-u0"),
        uncertainty_calibration_receipt_hash=calibrated.calibration_receipt_hash,
        independent_validation_split_hash=digest("validation-split"),
        flow_change_calibration_receipt_hash=digest("flow-change-calibration"),
        authorized_mode=UncertaintyFlowModeV1.BOUNDED_REFINEMENT,
        evidence_role=UncertaintyEvidenceRoleV1.BASE_FLOW,
        calibrated_maximum_update_px=0.5,
        qualified=True,
    )
    with pytest.raises(PermissionError, match="different uncertainty"):
        bounded_uncertainty_guided_flow_v1(
            np.zeros((1, 1, 2), np.float32),
            np.ones((1, 1, 2), np.float32),
            np.zeros((1, 1), np.float32),
            np.ones((1, 1), bool),
            semantics=UncertaintyMapSemanticsV1.LOG_VARIANCE,
            policy=policy,
            receipt=calibrated,
            flow_change_authorization=wrong_provider,
        )


def test_semantics_are_converted_before_reliability():
    scale = np.array([[2.0]], np.float32)
    from_scale = uncertainty_reliability_v1(
        scale, UncertaintyMapSemanticsV1.LAPLACE_SCALE,
    )
    from_variance = uncertainty_reliability_v1(
        scale ** 2, UncertaintyMapSemanticsV1.VARIANCE,
    )
    np.testing.assert_allclose(from_scale, from_variance, rtol=0, atol=1e-7)


def test_training_stages_enforce_gradient_firewall_contract():
    DecoupledTrainingPolicyV1(
        stage=UncertaintyTrainingStageV1.UNCERTAINTY_HEAD_ONLY,
    )
    DecoupledTrainingPolicyV1(
        stage=UncertaintyTrainingStageV1.FROZEN_UNCERTAINTY_REFINER,
        uncertainty_head_trainable=False,
        refiner_trainable=True,
    )
    with pytest.raises(ValueError, match="coupled"):
        DecoupledTrainingPolicyV1(
            stage=UncertaintyTrainingStageV1.UNCERTAINTY_HEAD_ONLY,
            detach_consistency_teacher=False,
        )


def test_laplace_teacher_and_refiner_detach_uncertainty():
    torch = pytest.importorskip("torch")
    log_scale = torch.zeros((1, 1, 3, 3), requires_grad=True)
    reference = torch.zeros((1, 2, 3, 3), requires_grad=True)
    augmented = torch.ones((1, 2, 3, 3), requires_grad=True)
    valid = torch.ones((1, 1, 3, 3), dtype=torch.bool)
    loss = augmentation_consistency_laplace_loss_v1(
        log_scale, reference, augmented, valid,
    )
    loss.backward()
    assert log_scale.grad is not None
    assert reference.grad is None
    assert augmented.grad is None

    refiner = BoundedUncertaintyFlowRefinerV1(4, maximum_update_px=0.25)
    features = torch.zeros((1, 4, 3, 3), requires_grad=True)
    base = torch.zeros((1, 2, 3, 3), requires_grad=True)
    uncertainty = torch.zeros((1, 1, 3, 3), requires_grad=True)
    refiner(features, base, uncertainty).sum().backward()
    assert features.grad is not None
    assert base.grad is not None
    assert uncertainty.grad is None


def test_laplace_teacher_rejects_nonboolean_or_nonfinite_valid_evidence():
    torch = pytest.importorskip("torch")
    log_scale = torch.zeros((1, 1, 1, 1))
    reference = torch.zeros((1, 2, 1, 1))
    augmented = torch.ones((1, 2, 1, 1))
    with pytest.raises(ValueError, match="boolean"):
        augmentation_consistency_laplace_loss_v1(
            log_scale, reference, augmented, torch.ones((1, 1, 1, 1)),
        )
    with pytest.raises(ValueError, match="finite"):
        augmentation_consistency_laplace_loss_v1(
            torch.full_like(log_scale, float("nan")),
            reference,
            augmented,
            torch.ones((1, 1, 1, 1), dtype=torch.bool),
        )
