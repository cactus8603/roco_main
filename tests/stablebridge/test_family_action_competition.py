from __future__ import annotations

from dataclasses import replace

import pytest

from stablebridge.physical_repair.family_action_competition import (
    ActionFamilyManifestV1,
    ExactControlFamilyBindingV1,
    FamilyCommitGateV1,
    FrozenActionFamilyDefinitionV1,
    propose_family_competition_v1,
    select_family_portfolio_v1,
    validate_action_family_manifest_v1,
)
from stablebridge.physical_repair.selector_v7 import (
    ASSESSOR_FEATURE_PROFILE_V7,
    PRIMARY_ASSESSOR_HEADS_V7,
    ArmRealizationReceiptV7,
    ArmRealizationStatusV7,
    AvailabilityV7,
    CalibrationMethodV7,
    CalibrationReceiptV7,
    CandidateArmV7,
    CandidateManifestV7,
    CandidateScopeV7,
    CostAccountV7,
    CostCountersV7,
    FoldRoleV7,
    PlannerCandidateV7,
    PredictionKindV7,
    PortfolioStateV7,
    SELECTOR_V7_SCHEMA_VERSION,
    SevereEstimandV7,
    make_native_arm_v7,
)


def counters(count: int = 1, *, wall: float = 1.0) -> CostCountersV7:
    return CostCountersV7(
        natural_forwards=0,
        reused_forwards=0,
        candidate_forwards=count,
        reverse_forwards=0,
        cpu_seconds=0.1 * count,
        gpu_seconds=0.9 * count,
        wall_seconds=wall,
        bytes_moved=1024 * count,
        peak_allocated_bytes=1024,
        peak_reserved_bytes=2048,
        matcher_trajectories=count,
        cache_state="cold",
    )


def arm(
    index: int,
    name: str,
    *,
    strength: float,
    beta: float = 1.0,
    cost: float = 1.0,
) -> CandidateArmV7:
    return CandidateArmV7(
        candidate_id=name,
        action_index=index,
        action_identity=name.split(":", 1)[0],
        mechanism_id=f"mechanism:{name}",
        operator_version="v1",
        exact_control_id=f"control:{name}",
        endpoint_id="second",
        input_strength=strength,
        output_beta=beta,
        action_hash=f"action-hash:{name}",
        control_hash=f"control-hash:{name}",
        endpoint_hash="endpoint-hash",
        support_hash=f"support-hash:{name}",
        support_policy_hash="support-policy",
        rollback_hash=f"rollback-hash:{name}",
        source_input_hashes=("source-hash",),
        materialization_recipe_id=f"recipe:{name}",
        materialization_recipe_version="v1",
        materialization_recipe_hash=f"recipe-hash:{name}",
        provenance_hash=f"provenance:{name}",
        scope=CandidateScopeV7.CASE_ATOMIC,
        unit_ids=("case-1",),
        interaction_receipts=(),
        offline_bank_acquisition_cost_ceiling=cost,
        offline_bank_acquisition_cost_counter_ceiling=counters(
            wall=cost,
        ),
        prospective_runtime_cost_ceiling=cost,
        prospective_runtime_cost_counter_ceiling=counters(wall=cost),
        hard_legal=True,
    )


def manifest(*rows: CandidateArmV7) -> CandidateManifestV7:
    native = make_native_arm_v7(
        execution_unit_id="case-1",
        endpoint_id="second",
        support_hash="native-support",
        support_policy_hash="support-policy",
        source_hash="source-hash",
        provenance_hash="native-provenance",
        source_materialization_hash="native-bytes",
    )
    return CandidateManifestV7(
        case_id="case-1",
        execution_unit_id="case-1",
        fold_role=FoldRoleV7.EVAL,
        source_hash="source-hash",
        support_policy_hash="support-policy",
        planner_allowlist_hash="planner-allowlist",
        assessor_allowlist_hash="assessor-allowlist",
        schema_version=SELECTOR_V7_SCHEMA_VERSION,
        provenance_hash="manifest-provenance",
        frozen_before_outcome=True,
        candidates=(native, *rows),
    )


def binding(
    row: CandidateArmV7,
    family: str,
    strength_id: str,
) -> ExactControlFamilyBindingV1:
    return ExactControlFamilyBindingV1(
        candidate_id=row.candidate_id,
        planned_arm_id=row.planned_arm_id,
        action_identity=row.action_identity,
        exact_control_id=row.exact_control_id,
        action_family_id=family,
        strength_id=strength_id,
        variant_id="operator-default",
        beta_id=f"beta-{row.output_beta:g}",
        input_strength=row.input_strength,
        output_beta=row.output_beta,
    )


def family_manifest(
    bank: CandidateManifestV7,
    *rows: ExactControlFamilyBindingV1,
) -> ActionFamilyManifestV1:
    family_ids = sorted({row.action_family_id for row in rows})
    return ActionFamilyManifestV1(
        selector_manifest_hash=bank.manifest_hash,
        selector_candidate_family_hash=bank.candidate_family_hash,
        frozen_before_outcome=True,
        definitions=tuple(
            FrozenActionFamilyDefinitionV1(
                action_family_id=family_id,
                member_action_identities=(family_id,),
                registry_source_hash="test-family-registry-v1",
            )
            for family_id in family_ids
        ),
        bindings=rows,
    )


def score(row: CandidateArmV7, value: float) -> PlannerCandidateV7:
    return PlannerCandidateV7(
        candidate_id=row.candidate_id,
        planned_arm_id=row.planned_arm_id,
        cost_hash=row.cost_hash,
        planner_allowlist_hash="planner-allowlist",
        before_feature_hash=f"before:{row.candidate_id}",
        prediction_kind=(
            PredictionKindV7.PLANNER_MODEL_PREDICTION_PREDECISION
        ),
        planner_predicted_gain_raw_px=value,
        before_features_complete=True,
    )


def realization(
    bank: CandidateManifestV7,
    row: CandidateArmV7,
) -> ArmRealizationReceiptV7:
    return ArmRealizationReceiptV7(
        manifest_hash=bank.manifest_hash,
        planned_arm_id=row.planned_arm_id,
        source_hash=bank.source_hash,
        status=ArmRealizationStatusV7.COMPLETE,
        status_reason="complete",
        realized_input_hash=f"input:{row.candidate_id}",
        realized_output_hash=f"output:{row.candidate_id}",
        realized_support_hash=f"support:{row.candidate_id}",
        realized_operator_hash=row.action_hash,
        tensor_shape=(2, 3, 2),
        tensor_dtype="float32",
        realized_bytes=48,
        offline_bank_acquisition_cost=(
            row.offline_bank_acquisition_cost_ceiling
        ),
        offline_bank_acquisition_cost_counters=(
            row.offline_bank_acquisition_cost_counter_ceiling
        ),
        cost_account=CostAccountV7.OFFLINE_BANK_ACQUISITION,
        provenance_hash=f"realization:{row.candidate_id}",
    )


def calibration(
    bank: CandidateManifestV7,
    *covered: CandidateArmV7,
) -> CalibrationReceiptV7:
    return CalibrationReceiptV7(
        manifest_hash=bank.manifest_hash,
        candidate_family_hash=bank.candidate_family_hash,
        target_fold_role=bank.fold_role,
        calibration_fold_id="fold-cal",
        model_hash="model-hash",
        feature_schema_hash="feature-schema",
        calibration_data_hash="calibration-data",
        calibration_policy_hash="calibration-policy",
        calibration_provenance_hash="calibration-provenance",
        planner_allowlist_hash=bank.planner_allowlist_hash,
        assessor_allowlist_hash=bank.assessor_allowlist_hash,
        assessor_feature_profile=ASSESSOR_FEATURE_PROFILE_V7,
        head_names=PRIMARY_ASSESSOR_HEADS_V7,
        severe_estimand=SevereEstimandV7.EXECUTION_UNIT_ANY_ROW_SEVERE,
        selection_method=CalibrationMethodV7.COMPONENT_MAX_RESIDUAL,
        covered_planned_arm_ids=tuple(row.planned_arm_id for row in covered),
        frozen_before_decision=True,
        availability=AvailabilityV7.AVAILABLE,
        availability_reason="available",
        effective_components=100,
        minimum_effective_components=9,
        confidence_level=0.95,
        familywise_error_rate=0.10,
        calibration_quantile_index=91,
        px_head_scale=0.25,
        probability_head_scale=1.0,
        minimum_fit_components_per_cell=3,
        minimum_calibration_components_per_cell=1,
        observed_fit_components_per_cell=10,
        observed_calibration_components_per_cell=2,
        row_severe_cutoff_g_raw_px=-0.25,
        cvar_level=0.95,
        rho_any_row_severe=0.05,
        maximum_harm=0.05,
        maximum_harmed_fraction=0.05,
        maximum_cvar95=0.25,
        cost_penalty=0.0,
        maximum_prospective_runtime_cost=4.0,
        maximum_prospective_runtime_cost_counters=counters(4, wall=4.0),
    )


def propose(
    bank: CandidateManifestV7,
    families: ActionFamilyManifestV1,
    scores: tuple[PlannerCandidateV7, ...],
    *,
    top_k: int = 2,
    budget: float = 4.0,
):
    rows = tuple(row for row in bank.candidates if row.action_index != 0)
    return propose_family_competition_v1(
        bank,
        families,
        scores,
        tuple(realization(bank, row) for row in rows),
        (),
        top_k=top_k,
        maximum_prospective_runtime_cost=budget,
        maximum_prospective_runtime_cost_counters=counters(
            4, wall=budget,
        ),
        cost_penalty_raw_px_per_cost_unit=0.0,
        planner_policy_hash="base-planner-policy",
        planner_provenance_hash="planner-provenance",
    )


def test_family_manifest_exactly_binds_every_non_native_arm() -> None:
    weak = arm(1, "blur:weak", strength=0.25)
    strong = arm(2, "blur:strong", strength=0.75)
    bank = manifest(weak, strong)
    families = family_manifest(
        bank,
        binding(weak, "blur", "weak"),
        binding(strong, "blur", "strong"),
    )

    validate_action_family_manifest_v1(bank, families)
    assert families.family_manifest_hash

    incomplete = family_manifest(bank, binding(weak, "blur", "weak"))
    with pytest.raises(ValueError, match="cover every non-native"):
        validate_action_family_manifest_v1(bank, incomplete)


def test_family_manifest_rejects_strength_drift() -> None:
    weak = arm(1, "blur:weak", strength=0.25)
    bank = manifest(weak)
    wrong = replace(binding(weak, "blur", "weak"), input_strength=0.5,
                    binding_hash="")
    families = family_manifest(bank, wrong)

    with pytest.raises(ValueError, match="input strength drifted"):
        validate_action_family_manifest_v1(bank, families)


def test_top_k_contains_distinct_action_families() -> None:
    blur_weak = arm(1, "blur:weak", strength=0.25)
    blur_strong = arm(2, "blur:strong", strength=0.75)
    jpeg = arm(3, "jpeg:medium", strength=0.5)
    bank = manifest(blur_weak, blur_strong, jpeg)
    families = family_manifest(
        bank,
        binding(blur_weak, "blur", "weak"),
        binding(blur_strong, "blur", "strong"),
        binding(jpeg, "jpeg", "medium"),
    )

    result = propose(
        bank,
        families,
        (score(blur_weak, 10.0), score(blur_strong, 9.0), score(jpeg, 8.0)),
    )

    assert [row.candidate_id for row in result.proposal.candidates] == [
        "blur:weak",
        "jpeg:medium",
    ]
    assert {row.action_family_id for row in result.family_winners} == {
        "blur",
        "jpeg",
    }
    assert result.suppressed_candidate_ids == ("blur:strong",)
    assert result.proposal.planner_policy_hash == (
        result.effective_planner_policy_hash
    )


def test_family_uses_best_feasible_strength_not_unaffordable_winner() -> None:
    expensive = arm(1, "blur:expensive", strength=0.75, cost=3.0)
    affordable = arm(2, "blur:affordable", strength=0.25, cost=1.0)
    jpeg = arm(3, "jpeg:medium", strength=0.5, cost=1.0)
    bank = manifest(expensive, affordable, jpeg)
    families = family_manifest(
        bank,
        binding(expensive, "blur", "expensive"),
        binding(affordable, "blur", "affordable"),
        binding(jpeg, "jpeg", "medium"),
    )

    result = propose(
        bank,
        families,
        (score(expensive, 10.0), score(affordable, 7.0), score(jpeg, 6.0)),
        budget=2.0,
    )

    assert [row.candidate_id for row in result.proposal.candidates] == [
        "blur:affordable",
        "jpeg:medium",
    ]
    assert "blur:expensive" in result.suppressed_candidate_ids


def test_nonpositive_strength_heads_produce_empty_native_safe_proposal() -> None:
    blur = arm(1, "blur:weak", strength=0.25)
    jpeg = arm(2, "jpeg:medium", strength=0.5)
    bank = manifest(blur, jpeg)
    families = family_manifest(
        bank,
        binding(blur, "blur", "weak"),
        binding(jpeg, "jpeg", "medium"),
    )

    result = propose(
        bank,
        families,
        (score(blur, 0.0), score(jpeg, -1.0)),
    )

    assert result.proposal.candidates == ()
    assert result.family_winners == ()
    assert result.suppressed_candidate_ids == ("blur:weak", "jpeg:medium")


def test_missing_strength_score_fails_closed() -> None:
    weak = arm(1, "blur:weak", strength=0.25)
    strong = arm(2, "blur:strong", strength=0.75)
    bank = manifest(weak, strong)
    families = family_manifest(
        bank,
        binding(weak, "blur", "weak"),
        binding(strong, "blur", "strong"),
    )

    with pytest.raises(ValueError, match="exactly one score"):
        propose(bank, families, (score(weak, 1.0),))


def test_commit_gate_requires_calibration_for_full_hard_legal_family() -> None:
    blur = arm(1, "blur:weak", strength=0.25)
    jpeg = arm(2, "jpeg:medium", strength=0.5)
    bank = manifest(blur, jpeg)
    families = family_manifest(
        bank,
        binding(blur, "blur", "weak"),
        binding(jpeg, "jpeg", "medium"),
    )
    family_proposal = propose(
        bank,
        families,
        (score(blur, 2.0), score(jpeg, 1.0)),
        top_k=1,
    )
    receipts = tuple(realization(bank, row) for row in (blur, jpeg))

    result = select_family_portfolio_v1(
        bank,
        families,
        family_proposal,
        (),
        (),
        calibration(bank, blur),
        receipts,
        (),
    )

    assert result.gate is (
        FamilyCommitGateV1.CALIBRATION_CHOICE_FAMILY_INCOMPLETE
    )
    assert set(result.required_calibration_planned_arm_ids) == {
        blur.planned_arm_id,
        jpeg.planned_arm_id,
    }
    assert result.decision.state is PortfolioStateV7.NATIVE
