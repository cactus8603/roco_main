from __future__ import annotations

from dataclasses import replace
import math

import pytest

from stablebridge.physical_repair.selector_v7 import (
    ActionRiskVectorV7,
    ArmAliasReceiptV7,
    ArmRealizationReceiptV7,
    ArmRealizationStatusV7,
    AuditEligibilityReceiptV7,
    ASSESSOR_FEATURE_PROFILE_V7,
    ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7,
    ASSESSOR_POST_ACTION_FIELDS_V7,
    AvailabilityV7,
    CalibrationMethodV7,
    CalibrationReceiptV7,
    CapacityIntervalStatusV7,
    CapacityIntervalV7,
    CapacityReasonV7,
    CapacityReportV7,
    CapacityRouteV7,
    CandidateArmV7,
    CandidateExclusionStateV7,
    CandidateManifestV7,
    CandidateObservationV7,
    CandidateScopeV7,
    ConfirmatoryInferenceContractV7,
    CostAccountV7,
    CostCountersV7,
    DecisionReasonV7,
    FoldRoleV7,
    FeatureAllowlistsV7,
    FeatureDependencyKindV7,
    FeatureFieldV7,
    FeatureStageV7,
    HardRejectReasonV7,
    InteractionKindV7,
    InteractionReceiptV7,
    PLANNER_BEFORE_ONLY_FIELDS_V7,
    PLANNER_EXTENSION_FIELDS_V7,
    PLANNER_FEATURE_PROFILE_V7,
    PREACTION_SHARED_FIELDS_V7,
    PRIMARY_ASSESSOR_HEADS_V7,
    NUMERIC_PROVENANCE_V7,
    PlannerCandidateV7,
    PlannerProfileV7,
    PlannerProposalV7,
    PredictionKindV7,
    PortfolioStateV7,
    RuntimeFeatureV7,
    EvidenceBlockReceiptV7,
    EvidenceBlockV7,
    SELECTOR_V7_SCHEMA_VERSION,
    SevereEstimandV7,
    selector_v7_integration_fingerprint,
    selector_v7_schema_fingerprint,
    make_native_arm_v7,
    propose_candidates_v7,
    select_portfolio_v7,
)


def feature_allowlists() -> FeatureAllowlistsV7:
    shared = tuple(
        FeatureFieldV7(
            canonical_field_id=f"field:{name}",
            name=name,
            stage=FeatureStageV7.PREACTION_SHARED,
            available_at_stage=FeatureStageV7.PREACTION_SHARED,
            source_hash=f"source:{name}",
            dependency_kind=FeatureDependencyKindV7.PREEXECUTION_RECEIPT,
            dependency_hash=f"dependency:{name}",
        )
        for name in sorted(PREACTION_SHARED_FIELDS_V7)
    )
    assessor_extension = tuple(
        FeatureFieldV7(
            canonical_field_id=f"field:{name}",
            name=name,
            stage=FeatureStageV7.ASSESSOR_POSTACTION_EXTENSION,
            available_at_stage=FeatureStageV7.ASSESSOR_POSTACTION_EXTENSION,
            source_hash=f"source:{name}",
            dependency_kind=FeatureDependencyKindV7.POST_ACTION_OBSERVATION,
            dependency_hash=f"dependency:{name}",
        )
        for name in sorted(ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7)
    )
    return FeatureAllowlistsV7(shared, (), assessor_extension)


def cost_counters(*, candidate_forwards: int = 1,
                  wall_seconds: float = 1.0) -> CostCountersV7:
    return CostCountersV7(
        natural_forwards=0,
        reused_forwards=0,
        candidate_forwards=candidate_forwards,
        reverse_forwards=0,
        cpu_seconds=0.2,
        gpu_seconds=0.8,
        wall_seconds=wall_seconds,
        bytes_moved=1024,
        peak_allocated_bytes=1024,
        peak_reserved_bytes=2048,
        matcher_trajectories=candidate_forwards,
        cache_state="cold",
    )


def budget_counters(count: int = 4) -> CostCountersV7:
    return CostCountersV7(
        natural_forwards=0, reused_forwards=0, candidate_forwards=count,
        reverse_forwards=0, cpu_seconds=0.2 * count,
        gpu_seconds=0.8 * count, wall_seconds=float(count),
        bytes_moved=1024 * count, peak_allocated_bytes=1024,
        peak_reserved_bytes=2048, matcher_trajectories=count,
        cache_state="cold",
    )


def arm(
    index: int,
    *,
    name: str | None = None,
    action_identity: str | None = None,
    scope: CandidateScopeV7 = CandidateScopeV7.CASE_ATOMIC,
    units: tuple[str, ...] = ("case-atomic",),
    interactions: tuple[InteractionReceiptV7, ...] = (),
    cost: float = 1.0,
    hard_legal: bool = True,
    reasons: tuple[HardRejectReasonV7, ...] = (),
) -> CandidateArmV7:
    key = name or f"candidate-{index}"
    return CandidateArmV7(
        candidate_id=key,
        action_index=index,
        action_identity=action_identity or f"action-{index}",
        mechanism_id=f"mechanism-{index}",
        operator_version="v1",
        exact_control_id=f"control-{index}",
        endpoint_id="second",
        input_strength=1.0,
        output_beta=1.0,
        action_hash=f"action-hash-{index}",
        control_hash=f"control-hash-{index}",
        endpoint_hash="endpoint-hash",
        support_hash=f"support-hash-{index}",
        support_policy_hash="support-policy",
        rollback_hash=f"rollback-hash-{index}",
        source_input_hashes=("source-hash",),
        materialization_recipe_id=f"recipe-{index}",
        materialization_recipe_version="v1",
        materialization_recipe_hash=f"materialization-recipe-hash-{index}",
        provenance_hash=f"provenance-hash-{index}",
        scope=scope,
        unit_ids=units,
        interaction_receipts=interactions,
        offline_bank_acquisition_cost_ceiling=cost,
        offline_bank_acquisition_cost_counter_ceiling=cost_counters(),
        prospective_runtime_cost_ceiling=cost,
        prospective_runtime_cost_counter_ceiling=cost_counters(),
        hard_legal=hard_legal,
        hard_reject_reasons=reasons,
    )


def manifest(*arms: CandidateArmV7, frozen: bool = True) -> CandidateManifestV7:
    allowlists = feature_allowlists()
    native = make_native_arm_v7(
        execution_unit_id="case-atomic",
        endpoint_id="second",
        support_hash="native-support",
        support_policy_hash="support-policy",
        source_hash="source-hash",
        provenance_hash="native-provenance",
        source_materialization_hash="native-bytes",
    )
    return CandidateManifestV7(
        case_id="case-1",
        execution_unit_id="case-atomic",
        fold_role=FoldRoleV7.EVAL,
        source_hash="source-hash",
        support_policy_hash="support-policy",
        planner_allowlist_hash=allowlists.planner_allowlist_hash,
        assessor_allowlist_hash=allowlists.assessor_allowlist_hash,
        schema_version=SELECTOR_V7_SCHEMA_VERSION,
        provenance_hash="manifest-provenance",
        frozen_before_outcome=frozen,
        candidates=(native, *arms),
    )


def score(row: CandidateArmV7, utility: float = 1.0,
          complete: bool = True) -> PlannerCandidateV7:
    return PlannerCandidateV7(
        candidate_id=row.candidate_id,
        planned_arm_id=row.planned_arm_id,
        cost_hash=row.cost_hash,
        planner_allowlist_hash=feature_allowlists().planner_allowlist_hash,
        before_feature_hash=f"before:{row.candidate_id}",
        prediction_kind=PredictionKindV7.PLANNER_MODEL_PREDICTION_PREDECISION,
        planner_predicted_gain_raw_px=utility,
        before_features_complete=complete,
    )


def realization(bank: CandidateManifestV7, row: CandidateArmV7, *,
                output_hash: str | None = None,
                status: ArmRealizationStatusV7 = ArmRealizationStatusV7.COMPLETE,
                ) -> ArmRealizationReceiptV7:
    complete = status is ArmRealizationStatusV7.COMPLETE
    return ArmRealizationReceiptV7(
        manifest_hash=bank.manifest_hash,
        planned_arm_id=row.planned_arm_id,
        source_hash=bank.source_hash,
        status=status,
        status_reason=status.value,
        realized_input_hash=f"input:{row.candidate_id}" if complete else None,
        realized_output_hash=(
            output_hash or f"output:{row.candidate_id}" if complete else None
        ),
        realized_support_hash=(
            f"realized-support:{row.candidate_id}" if complete else None
        ),
        realized_operator_hash=row.action_hash if complete else None,
        tensor_shape=(2, 3, 2) if complete else (),
        tensor_dtype="float32" if complete else None,
        realized_bytes=48 if complete else None,
        offline_bank_acquisition_cost=(
            row.offline_bank_acquisition_cost_ceiling if complete else None
        ),
        offline_bank_acquisition_cost_counters=(
            row.offline_bank_acquisition_cost_counter_ceiling
            if complete else None
        ),
        cost_account=CostAccountV7.OFFLINE_BANK_ACQUISITION,
        provenance_hash=f"realization-provenance:{row.candidate_id}",
    )


_RECEIPTS: dict[str, tuple[tuple[ArmRealizationReceiptV7, ...],
                           tuple[ArmAliasReceiptV7, ...]]] = {}


def proposal(bank: CandidateManifestV7, *rows: CandidateArmV7,
             top_k: int = 2,
             realizations: tuple[ArmRealizationReceiptV7, ...] | None = None,
             aliases: tuple[ArmAliasReceiptV7, ...] = ()) -> PlannerProposalV7:
    receipts = (
        tuple(realization(bank, row) for row in rows)
        if realizations is None else realizations
    )
    result = propose_candidates_v7(
        bank,
        tuple(score(row) for row in rows),
        receipts,
        aliases,
        top_k=top_k,
        maximum_prospective_runtime_cost=4.0,
        maximum_prospective_runtime_cost_counters=budget_counters(),
        planner_policy_hash="planner-policy",
        planner_provenance_hash="planner-provenance",
    )
    _RECEIPTS[result.proposal_hash] = (receipts, aliases)
    return result


def decide(bank, plan, observations, risks, cal, *, realizations=None,
           aliases=None, **kwargs):
    stored_realizations, stored_aliases = _RECEIPTS.get(
        plan.proposal_hash,
        (tuple(realization(bank, row) for row in bank.candidates
               if row.action_index != 0), ()),
    )
    return select_portfolio_v7(
        bank, plan, observations, risks, cal,
        stored_realizations if realizations is None else realizations,
        stored_aliases if aliases is None else aliases,
        **kwargs,
    )


def calibration(bank: CandidateManifestV7, *rows: CandidateArmV7,
                max_cost: float = 4.0) -> CalibrationReceiptV7:
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
        covered_planned_arm_ids=tuple(
            row.planned_arm_id for row in rows
        ),
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
        maximum_prospective_runtime_cost=max_cost,
        maximum_prospective_runtime_cost_counters=budget_counters(),
    )


def observation(
    bank: CandidateManifestV7,
    plan: PlannerProposalV7,
    row: CandidateArmV7,
    *,
    availability: AvailabilityV7 = AvailabilityV7.AVAILABLE,
    units: tuple[str, ...] | None = None,
    atomic: bool = True,
    cost: float | None = None,
    action_hash: str | None = None,
    receipt: ArmRealizationReceiptV7 | None = None,
) -> CandidateObservationV7:
    if receipt is None:
        receipt = next(
            item for item in _RECEIPTS[plan.proposal_hash][0]
            if item.planned_arm_id == row.planned_arm_id
        )
    return CandidateObservationV7(
        manifest_hash=bank.manifest_hash,
        proposal_hash=plan.proposal_hash,
        candidate_id=row.candidate_id,
        planned_arm_id=row.planned_arm_id,
        realization_receipt_hash=receipt.receipt_hash,
        action_hash=row.action_hash if action_hash is None else action_hash,
        control_hash=row.control_hash,
        endpoint_hash=row.endpoint_hash,
        support_hash=row.support_hash,
        realized_support_hash=str(receipt.realized_support_hash),
        interaction_hash=row.interaction_hash,
        cost_hash=row.cost_hash,
        source_hash=bank.source_hash,
        planner_feature_profile=PLANNER_FEATURE_PROFILE_V7,
        assessor_feature_profile=ASSESSOR_FEATURE_PROFILE_V7,
        planner_allowlist_hash=bank.planner_allowlist_hash,
        assessor_allowlist_hash=bank.assessor_allowlist_hash,
        preaction_shared_feature_hash=f"before:{row.candidate_id}",
        feature_schema_hash="feature-schema",
        provenance_hash=f"observation-provenance:{row.candidate_id}",
        availability=availability,
        availability_reason=availability.value,
        runtime_features=tuple(
            RuntimeFeatureV7(name, 0.2)
            for name in sorted(ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7)
        ),
        evidence_blocks=tuple(
            EvidenceBlockReceiptV7(
                block=block,
                availability=AvailabilityV7.AVAILABLE,
                source_hash=f"source:{block.value}",
                receipt_hash=f"receipt:{block.value}",
            )
            for block in EvidenceBlockV7
        ),
        evidence_source_hashes=("cc-receipt", "response-receipt"),
        observed_unit_ids=row.unit_ids if units is None else units,
        atomically_observed=atomic,
        observed_prospective_runtime_cost=(
            row.prospective_runtime_cost_ceiling if cost is None else cost
        ),
        observed_prospective_runtime_cost_counters=(
            row.prospective_runtime_cost_counter_ceiling
        ),
        cost_account=CostAccountV7.PROSPECTIVE_RUNTIME,
        cost_receipt_hash=f"cost-receipt:{row.candidate_id}",
    )


def risk(
    bank: CandidateManifestV7,
    plan: PlannerProposalV7,
    seen: CandidateObservationV7,
    row: CandidateArmV7,
    cal: CalibrationReceiptV7,
    *,
    benefit: float = 0.8,
    harm: float = 0.02,
    severe: float = 0.01,
    harmed_fraction: float = 0.05,
    cvar: float = 0.2,
    availability: AvailabilityV7 = AvailabilityV7.AVAILABLE,
    feature_eligible: bool = True,
    calibration_valid: bool = True,
) -> ActionRiskVectorV7:
    available = availability is AvailabilityV7.AVAILABLE
    return ActionRiskVectorV7(
        manifest_hash=bank.manifest_hash,
        proposal_hash=plan.proposal_hash,
        observation_hash=seen.observation_hash,
        candidate_id=row.candidate_id,
        planned_arm_id=row.planned_arm_id,
        action_hash=row.action_hash,
        control_hash=row.control_hash,
        endpoint_hash=row.endpoint_hash,
        support_hash=row.support_hash,
        interaction_hash=row.interaction_hash,
        cost_hash=row.cost_hash,
        assessor_hash="assessor-hash",
        assessor_provenance_hash="assessor-provenance",
        assessor_feature_profile=ASSESSOR_FEATURE_PROFILE_V7,
        assessor_allowlist_hash=bank.assessor_allowlist_hash,
        prediction_kind=PredictionKindV7.ASSESSOR_MODEL_PREDICTION_PREDECISION,
        calibration_receipt_hash=cal.receipt_hash,
        availability=availability,
        availability_reason=availability.value,
        feature_eligible=feature_eligible if available else False,
        calibration_valid=calibration_valid if available else False,
        bound_bookkeeping_valid=available,
        bookkeeping_receipt_hash="target-g-equals-b-minus-h:1e-12",
        predicted_benefit=benefit if available else None,
        benefit_lower=benefit if available else None,
        predicted_harm=harm if available else None,
        harm_upper=harm if available else None,
        predicted_any_row_severe_probability=(severe / 2) if available else None,
        any_row_severe_probability_upper=severe if available else None,
        predicted_harmed_pixel_fraction=(
            harmed_fraction / 2 if available else None
        ),
        harmed_pixel_fraction_upper=harmed_fraction if available else None,
        predicted_pixel_harm_cvar95=(cvar / 2) if available else None,
        pixel_harm_cvar95_upper=cvar if available else None,
    )


def bundle(*rows: CandidateArmV7):
    bank = manifest(*rows)
    plan = proposal(bank, *rows)
    cal = calibration(bank, *rows)
    observations = tuple(observation(bank, plan, row) for row in rows)
    risks = tuple(risk(bank, plan, seen, row, cal)
                  for row, seen in zip(rows, observations))
    return bank, plan, cal, observations, risks


def test_native_is_mandatory_feasible_zero_cost_action0():
    row = arm(1)
    bank = manifest(row)
    assert bank.native.action_index == 0
    assert bank.native.scope is CandidateScopeV7.NATIVE
    assert bank.native.hard_legal
    assert bank.native.prospective_runtime_cost_ceiling == 0.0
    with pytest.raises(ValueError, match="exactly one action-0 native"):
        allowlists = feature_allowlists()
        CandidateManifestV7(
            case_id="case", execution_unit_id="unit",
            fold_role=FoldRoleV7.EVAL, source_hash="source",
            support_policy_hash="support",
            planner_allowlist_hash=allowlists.planner_allowlist_hash,
            assessor_allowlist_hash=allowlists.assessor_allowlist_hash,
            schema_version=SELECTOR_V7_SCHEMA_VERSION,
            provenance_hash="provenance",
            frozen_before_outcome=True, candidates=(row,),
        )


def test_planner_topk_is_deterministic_and_only_uses_frozen_manifest():
    alpha = arm(1, action_identity="zeta", cost=1.0)
    beta = arm(2, action_identity="alpha", cost=1.0)
    gamma = arm(3, action_identity="mu", cost=2.0)
    bank = manifest(alpha, beta, gamma)
    scores = (score(alpha), score(gamma), score(beta))
    receipts = tuple(realization(bank, row) for row in (alpha, beta, gamma))
    p1 = propose_candidates_v7(
        bank, scores, receipts, (), top_k=1, planner_policy_hash="p",
        planner_provenance_hash="prov",
        maximum_prospective_runtime_cost=4.0,
        maximum_prospective_runtime_cost_counters=budget_counters(),
    )
    p2 = propose_candidates_v7(
        bank, reversed(scores), receipts, (), top_k=2, planner_policy_hash="p",
        planner_provenance_hash="prov",
        maximum_prospective_runtime_cost=4.0,
        maximum_prospective_runtime_cost_counters=budget_counters(),
    )
    expected = tuple(
        row.candidate_id for row in sorted(
            (alpha, beta, gamma), key=lambda item: (
                item.prospective_runtime_cost_ceiling, item.planned_arm_id,
            ),
        )
    )
    assert tuple(row.candidate_id for row in p1.candidates) == expected[:1]
    assert tuple(row.candidate_id for row in p2.candidates) == expected[:2]
    with pytest.raises(ValueError, match="frozen"):
        frozen_false = manifest(alpha, frozen=False)
        propose_candidates_v7(
            frozen_false, (score(alpha),),
            (realization(frozen_false, alpha),), (), top_k=1,
            planner_policy_hash="p", planner_provenance_hash="prov",
            maximum_prospective_runtime_cost=4.0,
            maximum_prospective_runtime_cost_counters=budget_counters(),
        )


def test_k2_replaces_risky_first_proposal_with_safer_second_candidate():
    first = arm(1, action_identity="first")
    second = arm(2, action_identity="second")
    bank, plan, cal, observations, _ = bundle(first, second)
    risky = risk(
        bank, plan, observations[0], first, cal,
        benefit=3.0, harm=0.01, severe=0.40,
    )
    safer = risk(
        bank, plan, observations[1], second, cal,
        benefit=0.7, harm=0.02, severe=0.01,
    )
    decision = decide(
        bank, plan, observations, (risky, safer), cal,
    )
    assert decision.state is PortfolioStateV7.COMMIT
    assert decision.selected_candidate_id == second.candidate_id
    assert decision.safe_candidate_ids == (second.candidate_id,)


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"severe": 0.20}, DecisionReasonV7.ANY_ROW_SEVERE_BUDGET_EXCEEDED),
        ({"harm": 0.40}, DecisionReasonV7.RISK_BUDGET_EXCEEDED),
        ({"harmed_fraction": 0.40}, DecisionReasonV7.RISK_BUDGET_EXCEEDED),
        ({"cvar": 0.80}, DecisionReasonV7.RISK_BUDGET_EXCEEDED),
    ],
)
def test_risk_heads_are_separate_hard_gates_not_one_scalar(
    overrides, expected_reason,
):
    row = arm(1)
    bank, plan, cal, observations, _ = bundle(row)
    unsafe = risk(
        bank, plan, observations[0], row, cal,
        benefit=100.0, **overrides,
    )
    decision = decide(bank, plan, observations, (unsafe,), cal)
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (expected_reason,)


def test_nonpositive_lb_minus_uh_minus_cost_falls_back_to_native():
    row = arm(1, cost=2.0)
    bank, plan, cal, observations, _ = bundle(row)
    unprofitable = risk(
        bank, plan, observations[0], row, cal,
        benefit=0.01, harm=0.02,
    )
    decision = decide(
        bank, plan, observations, (unprofitable,), cal,
    )
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.selected_candidate_id == bank.native.candidate_id
    assert decision.reasons == (
        DecisionReasonV7.NONPOSITIVE_CONSERVATIVE_UTILITY,
    )


def test_manifest_drift_and_candidate_injection_fail_closed():
    row = arm(1)
    bank, plan, cal, observations, risks = bundle(row)
    drifted = replace(plan, manifest_hash="another-manifest", proposal_hash="")
    decision = decide(bank, drifted, observations, risks, cal)
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (DecisionReasonV7.MANIFEST_DRIFT,)

    ghost = replace(
        plan.candidates[0], candidate_id="ghost", planned_arm_id="ghost",
    )
    injected = replace(plan, candidates=(ghost,), proposal_hash="")
    decision = decide(bank, injected, observations, risks, cal)
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (DecisionReasonV7.CANDIDATE_INJECTION,)


def test_hard_reject_cannot_be_reopened_by_a_good_score():
    rejected = arm(
        1, hard_legal=False,
        reasons=(HardRejectReasonV7.ILLEGAL_LINEAGE,),
    )
    bank = manifest(rejected)
    reopened = PlannerProposalV7(
        manifest_hash=bank.manifest_hash,
        candidate_family_hash=bank.candidate_family_hash,
        source_hash=bank.source_hash,
        fold_role=bank.fold_role,
        planner_allowlist_hash=bank.planner_allowlist_hash,
        planner_policy_hash="planner",
        planner_provenance_hash="provenance",
        profile=PlannerProfileV7.K1_ABLATION,
        top_k=1,
        maximum_prospective_runtime_cost=4.0,
        reserved_prospective_runtime_cost=(
            rejected.prospective_runtime_cost_ceiling
        ),
        maximum_prospective_runtime_cost_counters=budget_counters(),
        reserved_prospective_runtime_cost_counters=(
            rejected.prospective_runtime_cost_counter_ceiling
        ),
        frozen_before_observation=True,
        candidates=(score(rejected, utility=1000.0),),
    )
    cal = calibration(bank, rejected)
    decision = decide(bank, reopened, (), (), cal)
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (
        DecisionReasonV7.HARD_REJECT_REOPEN_ATTEMPT,
    )


def test_k_overflow_fails_closed_without_executing_evidence():
    first, second = arm(1), arm(2)
    bank = manifest(first, second)
    overflow = PlannerProposalV7(
        manifest_hash=bank.manifest_hash,
        candidate_family_hash=bank.candidate_family_hash,
        source_hash=bank.source_hash,
        fold_role=bank.fold_role,
        planner_allowlist_hash=bank.planner_allowlist_hash,
        planner_policy_hash="planner",
        planner_provenance_hash="provenance",
        profile=PlannerProfileV7.K1_ABLATION,
        top_k=1,
        maximum_prospective_runtime_cost=4.0,
        reserved_prospective_runtime_cost=2.0,
        maximum_prospective_runtime_cost_counters=budget_counters(),
        reserved_prospective_runtime_cost_counters=(
            first.prospective_runtime_cost_counter_ceiling.plus(
                second.prospective_runtime_cost_counter_ceiling
            )
        ),
        frozen_before_observation=True,
        candidates=(score(first), score(second)),
    )
    cal = calibration(bank, first, second)
    decision = decide(bank, overflow, (), (), cal)
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (DecisionReasonV7.PROPOSAL_OVERFLOW,)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nan_and_inf_are_rejected_and_corrupt_decoded_objects_fail_closed(bad):
    with pytest.raises(ValueError, match="finite"):
        PlannerCandidateV7(
            candidate_id="c", planned_arm_id="identity",
            cost_hash="cost", planner_allowlist_hash="planner",
            before_feature_hash="features",
            prediction_kind=(
                PredictionKindV7.PLANNER_MODEL_PREDICTION_PREDECISION
            ),
            planner_predicted_gain_raw_px=bad,
            before_features_complete=True,
        )
    row = arm(1)
    bank, plan, cal, observations, risks = bundle(row)
    object.__setattr__(risks[0], "harm_upper", bad)
    decision = decide(bank, plan, observations, risks, cal)
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (DecisionReasonV7.MALFORMED_INPUT,)


def test_action_or_control_binding_mismatch_fails_closed():
    row = arm(1)
    bank = manifest(row)
    plan = proposal(bank, row, top_k=1)
    cal = calibration(bank, row)
    mismatched = observation(
        bank, plan, row, action_hash="different-action-hash",
    )
    assessed = risk(bank, plan, mismatched, row, cal)
    decision = decide(
        bank, plan, (mismatched,), (assessed,), cal,
    )
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (DecisionReasonV7.ACTION_BINDING_MISMATCH,)


def test_unknown_or_asymmetric_local_interaction_is_permanently_ineligible():
    one_way = InteractionReceiptV7(
        source_unit_id="r1", target_unit_id="r2",
        kind=InteractionKindV7.DISJOINT, interaction_upper=0.0,
        receipt_hash="pair-proof",
    )
    unknown = arm(
        1, scope=CandidateScopeV7.CERTIFIED_LOCAL, units=("r1", "r2"),
        interactions=(one_way,), hard_legal=False,
        reasons=(HardRejectReasonV7.UNKNOWN_INTERACTION,),
    )
    bank = manifest(unknown)
    reopened = PlannerProposalV7(
        manifest_hash=bank.manifest_hash,
        candidate_family_hash=bank.candidate_family_hash,
        source_hash=bank.source_hash,
        fold_role=bank.fold_role,
        planner_allowlist_hash=bank.planner_allowlist_hash,
        planner_policy_hash="planner",
        planner_provenance_hash="provenance",
        profile=PlannerProfileV7.K1_ABLATION,
        top_k=1,
        maximum_prospective_runtime_cost=4.0,
        reserved_prospective_runtime_cost=(
            unknown.prospective_runtime_cost_ceiling
        ),
        maximum_prospective_runtime_cost_counters=budget_counters(),
        reserved_prospective_runtime_cost_counters=(
            unknown.prospective_runtime_cost_counter_ceiling
        ),
        frozen_before_observation=True,
        candidates=(score(unknown),),
    )
    cal = calibration(bank, unknown)
    decision = decide(bank, reopened, (), (), cal)
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (DecisionReasonV7.UNKNOWN_INTERACTION,)


def test_v439_primary_forbids_even_symmetric_multi_local_composition():
    pair = (
        InteractionReceiptV7(
            "r1", "r2", InteractionKindV7.DISJOINT, 0.0, "pair-proof",
        ),
        InteractionReceiptV7(
            "r2", "r1", InteractionKindV7.DISJOINT, 0.0, "pair-proof",
        ),
    )
    with pytest.raises(ValueError, match="forbids multi-local"):
        arm(
            1, scope=CandidateScopeV7.CERTIFIED_LOCAL,
            units=("r1", "r2"), interactions=pair,
        )


def test_global_action_partial_observation_or_commit_attempt_fails_closed():
    row = arm(1, scope=CandidateScopeV7.CASE_ATOMIC, units=("case-atomic",))
    bank = manifest(row)
    plan = proposal(bank, row, top_k=1)
    cal = calibration(bank, row)
    partial = observation(
        bank, plan, row, units=("case-atomic",), atomic=False,
    )
    assessed = risk(bank, plan, partial, row, cal)
    decision = decide(bank, plan, (partial,), (assessed,), cal)
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (DecisionReasonV7.GLOBAL_PARTIAL_COMMIT,)


def test_typed_missing_ood_and_low_support_are_explicit_native_states():
    row = arm(1)
    bank = manifest(row)
    plan = proposal(bank, row, top_k=1)
    cal = calibration(bank, row)
    for status, reason in (
        (AvailabilityV7.TYPED_MISSING, DecisionReasonV7.TYPED_MISSING),
        (AvailabilityV7.OOD, DecisionReasonV7.OOD),
        (AvailabilityV7.LOW_SUPPORT, DecisionReasonV7.LOW_SUPPORT),
    ):
        seen = observation(bank, plan, row, availability=status)
        assessed = risk(bank, plan, seen, row, cal)
        decision = decide(
            bank, plan, (seen,), (assessed,), cal,
        )
        assert decision.state is PortfolioStateV7.NATIVE
        assert decision.reasons == (reason,)


@pytest.mark.parametrize(
    "name", ["corruption_family", "dataset_id", "scene", "image_path", "gt", "outcome"],
)
def test_forbidden_runtime_identity_or_target_features_are_rejected(name):
    with pytest.raises(ValueError, match="forbidden"):
        RuntimeFeatureV7(name, 1.0)


def test_observed_or_reserved_cost_overflow_returns_native():
    row = arm(1, cost=1.0)
    bank = manifest(row)
    plan = proposal(bank, row, top_k=1)
    cal = calibration(bank, row)
    seen = observation(bank, plan, row, cost=1.1)
    assessed = risk(bank, plan, seen, row, cal)
    decision = decide(bank, plan, (seen,), (assessed,), cal)
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (DecisionReasonV7.COST_BUDGET_EXCEEDED,)


def test_one_audit_only_and_child_counter_is_bounded():
    row = arm(1)
    bank, plan, cal, observations, _ = bundle(row)
    ambiguous = replace(risk(
        bank, plan, observations[0], row, cal,
        benefit=0.03, harm=0.04,
    ), predicted_benefit=0.10, risk_vector_hash="")
    audit = AuditEligibilityReceiptV7(
        manifest_hash=bank.manifest_hash,
        proposal_hash=plan.proposal_hash,
        observation_hash=observations[0].observation_hash,
        candidate_id=row.candidate_id,
        planned_arm_id=row.planned_arm_id,
        independent_evidence_id="independent-evidence",
        independent_evidence_policy_hash="audit-policy",
        interval_crosses_commit_boundary=True,
        value_of_information_lower=0.20,
        prospective_runtime_evidence_cost=0.10,
        prospective_runtime_evidence_cost_counters=CostCountersV7(
            natural_forwards=0, reused_forwards=0, candidate_forwards=0,
            reverse_forwards=0, cpu_seconds=0.01, gpu_seconds=0.0,
            wall_seconds=0.01, bytes_moved=8, peak_allocated_bytes=0,
            peak_reserved_bytes=0, matcher_trajectories=0,
            cache_state="cold",
        ),
        cost_account=CostAccountV7.PROSPECTIVE_RUNTIME,
        frozen_before_decision=True,
        provenance_hash="audit-provenance",
    )
    first = decide(
        bank, plan, observations, (ambiguous,), cal, audit_count=0,
        audit_receipts=(audit,),
    )
    assert first.state is PortfolioStateV7.AUDIT
    assert first.audit_count == 1
    second = decide(
        bank, plan, observations, (ambiguous,), cal, audit_count=1,
        audit_receipts=(audit,),
    )
    assert second.state is PortfolioStateV7.NATIVE
    assert second.reasons == (DecisionReasonV7.AUDIT_BUDGET_EXHAUSTED,)
    child_overflow = decide(
        bank, plan, observations, (ambiguous,), cal, child_count=2,
    )
    assert child_overflow.state is PortfolioStateV7.NATIVE
    assert child_overflow.reasons == (DecisionReasonV7.CHILD_BUDGET_EXHAUSTED,)


def test_audit_cannot_reopen_a_severe_tail_veto():
    row = arm(1)
    bank, plan, cal, observations, _ = bundle(row)
    unsafe = replace(
        risk(
            bank, plan, observations[0], row, cal,
            benefit=0.03, harm=0.04, severe=0.20,
        ),
        predicted_benefit=0.10,
        risk_vector_hash="",
    )
    audit = AuditEligibilityReceiptV7(
        manifest_hash=bank.manifest_hash,
        proposal_hash=plan.proposal_hash,
        observation_hash=observations[0].observation_hash,
        candidate_id=row.candidate_id,
        planned_arm_id=row.planned_arm_id,
        independent_evidence_id="independent-evidence",
        independent_evidence_policy_hash="audit-policy",
        interval_crosses_commit_boundary=True,
        value_of_information_lower=0.20,
        prospective_runtime_evidence_cost=0.10,
        prospective_runtime_evidence_cost_counters=CostCountersV7(
            natural_forwards=0, reused_forwards=0, candidate_forwards=0,
            reverse_forwards=0, cpu_seconds=0.01, gpu_seconds=0.0,
            wall_seconds=0.01, bytes_moved=8, peak_allocated_bytes=0,
            peak_reserved_bytes=0, matcher_trajectories=0,
            cache_state="cold",
        ),
        cost_account=CostAccountV7.PROSPECTIVE_RUNTIME,
        frozen_before_decision=True,
        provenance_hash="audit-provenance",
    )
    decision = decide(
        bank, plan, observations, (unsafe,), cal, audit_receipts=(audit,),
    )
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (
        DecisionReasonV7.ANY_ROW_SEVERE_BUDGET_EXCEEDED,
    )


def test_commit_tie_break_is_deterministic_risk_then_cost_then_identity():
    zeta = arm(1, action_identity="zeta", cost=1.0)
    alpha = arm(2, action_identity="alpha", cost=1.0)
    bank, plan, cal, observations, risks = bundle(zeta, alpha)
    forward = decide(bank, plan, observations, risks, cal)
    reverse = decide(
        bank, plan, tuple(reversed(observations)), tuple(reversed(risks)), cal,
    )
    assert forward.state is reverse.state is PortfolioStateV7.COMMIT
    assert forward.selected_candidate_id == reverse.selected_candidate_id
    assert forward.selected_candidate_id == alpha.candidate_id


def test_objects_are_frozen_and_hashes_bind_exact_candidate_content():
    row = arm(1)
    with pytest.raises(Exception):
        row.action_hash = "changed"  # type: ignore[misc]
    bank = manifest(row)
    first = realization(bank, row, output_hash="output:first")
    second = realization(bank, row, output_hash="output:second")
    assert first.planned_arm_id == second.planned_arm_id == row.planned_arm_id
    assert first.receipt_hash != second.receipt_hash
    with pytest.raises(ValueError, match="planned arm id"):
        replace(
            row, materialization_recipe_hash="changed-recipe",
            planned_arm_id=row.planned_arm_id,
        )


def test_v439_calibration_constants_are_not_runtime_tunable():
    row = arm(1)
    bank = manifest(row)
    cal = calibration(bank, row)
    for field, value in (
        ("rho_any_row_severe", 0.051),
        ("maximum_harm", 0.051),
        ("maximum_harmed_fraction", 0.051),
        ("maximum_cvar95", 0.251),
        ("cost_penalty", 0.001),
        ("familywise_error_rate", 0.05),
        ("row_severe_cutoff_g_raw_px", -0.20),
    ):
        with pytest.raises(ValueError):
            replace(cal, **{field: value, "receipt_hash": ""})


def test_planner_skips_cost_overflow_by_frozen_rank_without_knapsack():
    expensive = arm(1, cost=3.0)
    cheap_a = arm(2, cost=1.0)
    cheap_b = arm(3, cost=1.0)
    bank = manifest(expensive, cheap_a, cheap_b)
    plan = propose_candidates_v7(
        bank,
        (score(expensive, 3.0), score(cheap_a, 2.0), score(cheap_b, 1.0)),
        tuple(realization(bank, row) for row in (expensive, cheap_a, cheap_b)),
        (),
        top_k=2,
        maximum_prospective_runtime_cost=2.0,
        maximum_prospective_runtime_cost_counters=budget_counters(2),
        planner_policy_hash="ridge-l2-1",
        planner_provenance_hash="fit-only",
    )
    assert tuple(row.candidate_id for row in plan.candidates) == (
        cheap_a.candidate_id, cheap_b.candidate_id,
    )
    assert plan.reserved_prospective_runtime_cost == pytest.approx(2.0)


def test_unknown_unobserved_arm_remains_typed_and_is_never_proposed():
    observed = arm(1)
    unknown = arm(2)
    bank = manifest(observed, unknown)
    receipts = (
        realization(bank, observed),
        realization(
            bank, unknown,
            status=ArmRealizationStatusV7.UNKNOWN_UNOBSERVED_ARM,
        ),
    )
    plan = propose_candidates_v7(
        bank, (score(unknown, 10.0), score(observed, 1.0)), receipts, (),
        top_k=2, maximum_prospective_runtime_cost=4.0,
        maximum_prospective_runtime_cost_counters=budget_counters(),
        planner_policy_hash="ridge-l2-1",
        planner_provenance_hash="fit-only",
    )
    assert tuple(row.candidate_id for row in plan.candidates) == (
        observed.candidate_id,
    )
    assert bank.exclusion_state(
        unknown.candidate_id,
        tuple(row.candidate_id for row in plan.candidates),
    ) is CandidateExclusionStateV7.SOFT_NOT_PROPOSED


def test_canonical_realized_arm_identity_deduplicates_aliases():
    original = arm(1)
    alias_arm = replace(
        arm(2, action_identity="byte-equivalent-alias"),
        action_hash=original.action_hash,
    )
    bank = manifest(original, alias_arm)
    original_receipt = realization(bank, original, output_hash="same-output")
    alias_receipt = ArmRealizationReceiptV7(
        manifest_hash=bank.manifest_hash,
        planned_arm_id=alias_arm.planned_arm_id,
        source_hash=bank.source_hash,
        status=ArmRealizationStatusV7.COMPLETE,
        status_reason="complete",
        realized_input_hash=original_receipt.realized_input_hash,
        realized_output_hash=original_receipt.realized_output_hash,
        realized_support_hash=original_receipt.realized_support_hash,
        realized_operator_hash=original_receipt.realized_operator_hash,
        tensor_shape=original_receipt.tensor_shape,
        tensor_dtype=original_receipt.tensor_dtype,
        realized_bytes=original_receipt.realized_bytes,
        offline_bank_acquisition_cost=1.0,
        offline_bank_acquisition_cost_counters=cost_counters(),
        cost_account=CostAccountV7.OFFLINE_BANK_ACQUISITION,
        provenance_hash="alias-realization-provenance",
    )
    representative, alias_member = sorted(
        (original, alias_arm), key=lambda row: row.planned_arm_id,
    )
    receipts_by_id = {
        original.planned_arm_id: original_receipt,
        alias_arm.planned_arm_id: alias_receipt,
    }
    representative_receipt = receipts_by_id[representative.planned_arm_id]
    alias_member_receipt = receipts_by_id[alias_member.planned_arm_id]
    alias = ArmAliasReceiptV7(
        manifest_hash=bank.manifest_hash,
        representative_planned_arm_id=representative.planned_arm_id,
        alias_planned_arm_id=alias_member.planned_arm_id,
        representative_realization_receipt_hash=representative_receipt.receipt_hash,
        alias_realization_receipt_hash=alias_member_receipt.receipt_hash,
        executable_bytes_hash=str(representative_receipt.executable_bytes_hash),
        realized_operator_hash=str(representative_receipt.realized_operator_hash),
        prospective_runtime_verification_cost=0.1,
        prospective_runtime_verification_cost_counters=CostCountersV7(
            natural_forwards=0, reused_forwards=0, candidate_forwards=0,
            reverse_forwards=0, cpu_seconds=0.01, gpu_seconds=0.0,
            wall_seconds=0.01, bytes_moved=48, peak_allocated_bytes=0,
            peak_reserved_bytes=0, matcher_trajectories=0,
            cache_state="cold",
        ),
        cost_account=CostAccountV7.PROSPECTIVE_RUNTIME,
        provenance_hash="alias-proof",
    )
    plan = propose_candidates_v7(
        bank, (score(original, 1.0), score(alias_arm, 1.0)),
        (original_receipt, alias_receipt), (alias,), top_k=2,
        maximum_prospective_runtime_cost=4.0,
        maximum_prospective_runtime_cost_counters=budget_counters(),
        planner_policy_hash="planner", planner_provenance_hash="provenance",
    )
    assert len(plan.candidates) == 1
    assert plan.candidates[0].planned_arm_id == representative.planned_arm_id
    assert plan.reserved_prospective_runtime_cost == pytest.approx(1.1)
    assert original_receipt.offline_bank_acquisition_cost == pytest.approx(1.0)
    assert alias_receipt.offline_bank_acquisition_cost == pytest.approx(1.0)
    cal = calibration(bank, representative)
    seen = observation(
        bank, plan, representative, receipt=representative_receipt, cost=2.0,
    )
    assessed = risk(bank, plan, seen, representative, cal)
    double_execution = decide(
        bank, plan, (seen,), (assessed,), cal,
        realizations=(original_receipt, alias_receipt), aliases=(alias,),
    )
    assert double_execution.state is PortfolioStateV7.NATIVE
    assert double_execution.reasons == (
        DecisionReasonV7.COST_BUDGET_EXCEEDED,
    )
    with pytest.raises(ValueError, match="canonical aliases"):
        propose_candidates_v7(
            bank, (score(original), score(alias_arm)),
            (original_receipt, alias_receipt), (), top_k=2,
            maximum_prospective_runtime_cost=4.0,
            maximum_prospective_runtime_cost_counters=budget_counters(),
            planner_policy_hash="planner", planner_provenance_hash="provenance",
        )


def test_single_certified_local_unit_is_primary_legal_and_committable():
    row = arm(
        1, scope=CandidateScopeV7.CERTIFIED_LOCAL, units=("region-7",),
    )
    bank, plan, cal, observations, risks = bundle(row)
    decision = decide(bank, plan, observations, risks, cal)
    assert decision.state is PortfolioStateV7.COMMIT
    assert decision.committed_unit_ids == ("region-7",)


def test_required_block_missing_marks_observed_feature_ineligible():
    row = arm(1)
    bank, plan, cal, observations, _ = bundle(row)
    blocks = tuple(
        replace(
            block,
            availability=AvailabilityV7.TYPED_MISSING,
        ) if block.block is EvidenceBlockV7.E233_CYCLE_G0 else block
        for block in observations[0].evidence_blocks
    )
    seen = replace(
        observations[0], evidence_blocks=blocks, observation_hash="",
    )
    assessed = risk(bank, plan, seen, row, cal)
    decision = decide(bank, plan, (seen,), (assessed,), cal)
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (DecisionReasonV7.TYPED_MISSING,)
    assert (row.candidate_id,
            CandidateExclusionStateV7.OBSERVED_BUT_FEATURE_INELIGIBLE) in (
                decision.exclusion_states
            )


def test_g_equals_b_minus_h_bookkeeping_receipt_is_a_feature_gate():
    row = arm(1)
    bank, plan, cal, observations, risks = bundle(row)
    inconsistent = replace(
        risks[0], bound_bookkeeping_valid=False, risk_vector_hash="",
    )
    decision = decide(
        bank, plan, observations, (inconsistent,), cal,
    )
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (DecisionReasonV7.FEATURE_INELIGIBLE,)
    assert decision.exclusion_states == ((
        row.candidate_id,
        CandidateExclusionStateV7.OBSERVED_BUT_FEATURE_INELIGIBLE,
    ),)


def test_exact_three_stage_exclusion_ledger_distinguishes_hard_and_soft():
    selected = arm(1)
    soft = arm(2)
    hard = arm(
        3, hard_legal=False,
        reasons=(HardRejectReasonV7.ILLEGAL_LINEAGE,),
    )
    bank = manifest(selected, soft, hard)
    plan = proposal(bank, selected, top_k=1)
    cal = calibration(bank, selected)
    seen = observation(bank, plan, selected)
    assessed = risk(bank, plan, seen, selected, cal)
    decision = decide(bank, plan, (seen,), (assessed,), cal)
    assert set(decision.exclusion_states) == {
        (soft.candidate_id, CandidateExclusionStateV7.SOFT_NOT_PROPOSED),
        (hard.candidate_id,
         CandidateExclusionStateV7.HARD_ILLEGAL_PREEXECUTION),
    }


def test_cost_counter_overflow_fails_closed_even_when_scalar_cost_matches():
    row = arm(1)
    bank, plan, cal, observations, _ = bundle(row)
    excessive = replace(
        row.prospective_runtime_cost_counter_ceiling,
        matcher_trajectories=(
            row.prospective_runtime_cost_counter_ceiling.matcher_trajectories + 1
        ),
    )
    seen = replace(
        observations[0], observed_prospective_runtime_cost_counters=excessive,
        observation_hash="",
    )
    assessed = risk(bank, plan, seen, row, cal)
    decision = decide(bank, plan, (seen,), (assessed,), cal)
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (DecisionReasonV7.COST_BUDGET_EXCEEDED,)


def test_v439_allowlists_share_preacton_rows_and_only_extensions_are_disjoint():
    allowlists = feature_allowlists()
    assert PLANNER_EXTENSION_FIELDS_V7 == frozenset()
    assert {row.name for row in allowlists.planner_fields} == (
        PREACTION_SHARED_FIELDS_V7
    )
    assert {row.name for row in allowlists.assessor_fields} == (
        ASSESSOR_POST_ACTION_FIELDS_V7
    )
    assert all(
        planner is assessor
        for planner, assessor in zip(
            allowlists.planner_fields,
            allowlists.assessor_fields[:len(allowlists.planner_fields)],
        )
    )
    assert allowlists.planner_allowlist_hash != allowlists.assessor_allowlist_hash


def test_postaction_field_cannot_enter_shared_base_or_alias_its_field_id():
    allowlists = feature_allowlists()
    injected = replace(
        allowlists.assessor_postaction_extension_fields[0],
        name=next(iter(PREACTION_SHARED_FIELDS_V7)),
    )
    with pytest.raises(ValueError, match="extension"):
        FeatureAllowlistsV7(
            allowlists.preaction_shared_fields,
            allowlists.planner_extension_fields,
            (injected, *allowlists.assessor_postaction_extension_fields[1:]),
        )
    with pytest.raises(ValueError, match="forbidden"):
        FeatureFieldV7(
            canonical_field_id="held-target",
            name="held_target_value",
            stage=FeatureStageV7.PREACTION_SHARED,
            available_at_stage=FeatureStageV7.PREACTION_SHARED,
            source_hash="held-source",
            dependency_kind=FeatureDependencyKindV7.PREEXECUTION_RECEIPT,
            dependency_hash="held-dependency",
        )
    duplicate_id = replace(
        allowlists.preaction_shared_fields[1],
        canonical_field_id=allowlists.preaction_shared_fields[0].canonical_field_id,
    )
    with pytest.raises(ValueError, match="unique"):
        FeatureAllowlistsV7(
            (
                allowlists.preaction_shared_fields[0], duplicate_id,
                *allowlists.preaction_shared_fields[2:],
            ),
            (), allowlists.assessor_postaction_extension_fields,
        )
    bad_dependency = replace(
        allowlists.assessor_postaction_extension_fields[0],
        dependency_kind=FeatureDependencyKindV7.PREEXECUTION_RECEIPT,
    )
    with pytest.raises(ValueError, match="post-action dependency"):
        FeatureAllowlistsV7(
            allowlists.preaction_shared_fields, (),
            (bad_dependency, *allowlists.assessor_postaction_extension_fields[1:]),
        )


def test_assessor_must_reference_the_exact_planner_shared_feature_values():
    row = arm(1)
    bank, plan, cal, observations, risks = bundle(row)
    drifted = replace(
        observations[0], preaction_shared_feature_hash="different-values",
        observation_hash="",
    )
    assessed = risk(bank, plan, drifted, row, cal)
    decision = decide(bank, plan, (drifted,), (assessed,), cal)
    assert decision.state is PortfolioStateV7.NATIVE
    assert decision.reasons == (DecisionReasonV7.ACTION_BINDING_MISMATCH,)


def test_primary_assessor_has_exactly_five_heads_and_no_independent_g_bound():
    row = arm(1)
    bank, plan, cal, observations, _ = bundle(row)
    assessed = risk(bank, plan, observations[0], row, cal,
                    benefit=0.8, harm=0.2)
    assert assessed.head_count == len(PRIMARY_ASSESSOR_HEADS_V7) == 5
    assert assessed.G_hat == pytest.approx(0.6)
    assert assessed.net_gain_lower == pytest.approx(0.6)
    assert not hasattr(assessed, "L_G")
    assert not hasattr(assessed, "U_G")
    assert not hasattr(assessed, "net_gain_upper")
    with pytest.raises(ValueError, match="benefit lower"):
        replace(
            assessed, predicted_benefit=0.1, benefit_lower=0.2,
            risk_vector_hash="",
        )


def _capacity_interval(
    estimate: float, lower: float, upper: float,
) -> CapacityIntervalV7:
    return CapacityIntervalV7(
        status=CapacityIntervalStatusV7.ESTIMABLE,
        estimate=estimate, lower=lower, upper=upper,
        valid_resamples=10_000,
    )


@pytest.mark.parametrize(
    ("confirmed_n", "possible_n", "confirmed", "possible", "route", "reason"),
    (
        (
            1, 1,
            _capacity_interval(0.10, 0.04, 0.17),
            _capacity_interval(0.10, 0.05, 0.19),
            CapacityRouteV7.ACTION_BANK_BOTTLENECK,
            CapacityReasonV7.POSSIBLE_UPPER_BELOW_BANK_THRESHOLD,
        ),
        (
            6, 7,
            _capacity_interval(0.60, 0.50, 0.70),
            _capacity_interval(0.70, 0.55, 0.85),
            CapacityRouteV7.CAPACITY_SUFFICIENT_FOR_SELECTOR_TEST,
            CapacityReasonV7.CONFIRMED_LOWER_AT_LEAST_SELECTOR_THRESHOLD,
        ),
        (
            4, 6,
            _capacity_interval(0.40, 0.30, 0.50),
            _capacity_interval(0.60, 0.40, 0.70),
            CapacityRouteV7.INCONCLUSIVE_CAPACITY_OR_MISSING_ARMS,
            CapacityReasonV7.BOUNDS_INCONCLUSIVE,
        ),
    ),
)
def test_capacity_routes_use_confirmed_lower_and_possible_upper(
    confirmed_n, possible_n, confirmed, possible, route, reason,
):
    report = CapacityReportV7(
        fixed_current_any_row_severe_denominator=10,
        eligible_independent_components=5,
        confirmed_safe_capacity_numerator=confirmed_n,
        possible_safe_capacity_numerator=possible_n,
        confirmed_interval=confirmed,
        possible_interval=possible,
        current_action_membership_complete=True,
        lineage_complete=True,
        valid_masks_nonzero=True,
        route=route,
        reason_code=reason,
    )
    assert report.route is route


def test_capacity_unknown_current_membership_is_typed_missing_and_no_ci_runs():
    missing = CapacityIntervalV7(
        status=CapacityIntervalStatusV7.CURRENT_ACTION_MEMBERSHIP_INCOMPLETE,
        estimate=None, lower=None, upper=None, valid_resamples=0,
    )
    report = CapacityReportV7(
        fixed_current_any_row_severe_denominator=None,
        eligible_independent_components=3,
        confirmed_safe_capacity_numerator=None,
        possible_safe_capacity_numerator=None,
        confirmed_interval=missing,
        possible_interval=missing,
        current_action_membership_complete=False,
        lineage_complete=False,
        valid_masks_nonzero=False,
        route=CapacityRouteV7.INCONCLUSIVE_CURRENT_ACTION_MEMBERSHIP,
        reason_code=(
            CapacityReasonV7.CURRENT_ACTION_OUTCOME_MASK_OR_LINEAGE_UNKNOWN
        ),
    )
    assert report.confirmed_interval.valid_resamples == 0


def test_capacity_empty_and_underfilled_bootstrap_states_fail_closed():
    zero = CapacityIntervalV7(
        status=CapacityIntervalStatusV7.NO_CURRENT_SEVERE_DENOMINATOR,
        estimate=None, lower=None, upper=None, valid_resamples=0,
    )
    no_denominator = CapacityReportV7(
        fixed_current_any_row_severe_denominator=0,
        eligible_independent_components=2,
        confirmed_safe_capacity_numerator=0,
        possible_safe_capacity_numerator=0,
        confirmed_interval=zero, possible_interval=zero,
        current_action_membership_complete=True,
        lineage_complete=True, valid_masks_nonzero=True,
        route=(
            CapacityRouteV7.NO_CURRENT_SEVERE_CAPACITY_ROUTE_NOT_APPLICABLE
        ),
        reason_code=CapacityReasonV7.NO_CURRENT_SEVERE_DENOMINATOR,
    )
    assert no_denominator.fixed_current_any_row_severe_denominator == 0

    underfilled = CapacityIntervalV7(
        status=CapacityIntervalStatusV7.NOT_ESTIMABLE,
        estimate=None, lower=None, upper=None, valid_resamples=9_499,
    )
    report = CapacityReportV7(
        fixed_current_any_row_severe_denominator=10,
        eligible_independent_components=2,
        confirmed_safe_capacity_numerator=3,
        possible_safe_capacity_numerator=6,
        confirmed_interval=underfilled, possible_interval=underfilled,
        current_action_membership_complete=True,
        lineage_complete=True, valid_masks_nonzero=True,
        route=CapacityRouteV7.INCONCLUSIVE_CAPACITY_OR_MISSING_ARMS,
        reason_code=CapacityReasonV7.INSUFFICIENT_VALID_BOOTSTRAPS,
    )
    assert report.route is CapacityRouteV7.INCONCLUSIVE_CAPACITY_OR_MISSING_ARMS


def test_capacity_schema_rejects_interpolation_and_impossible_membership_bounds():
    with pytest.raises(ValueError, match="inverted-CDF"):
        replace(_capacity_interval(0.5, 0.4, 0.6), quantile_method="linear")
    with pytest.raises(ValueError, match="bounds"):
        CapacityReportV7(
            fixed_current_any_row_severe_denominator=10,
            eligible_independent_components=2,
            confirmed_safe_capacity_numerator=7,
            possible_safe_capacity_numerator=6,
            confirmed_interval=_capacity_interval(0.7, 0.6, 0.8),
            possible_interval=_capacity_interval(0.6, 0.5, 0.7),
            current_action_membership_complete=True,
            lineage_complete=True, valid_masks_nonzero=True,
            route=CapacityRouteV7.INCONCLUSIVE_CAPACITY_OR_MISSING_ARMS,
            reason_code=CapacityReasonV7.BOUNDS_INCONCLUSIVE,
        )


def test_v439_inference_contract_is_exact_and_does_not_execute_statistics():
    contract = ConfirmatoryInferenceContractV7()
    assert contract.resamples == 10_000
    assert contract.seed == 20_261_003
    assert contract.tie_order == ("H1", "H2")
    assert contract.holm_first_threshold == pytest.approx(0.025)
    assert contract.holm_second_threshold == pytest.approx(0.05)
    with pytest.raises(ValueError, match="drifted"):
        ConfirmatoryInferenceContractV7(seed=1)


def test_schema_and_sr0a_sr1_integration_fingerprints_are_deterministic():
    first = selector_v7_schema_fingerprint()
    second = selector_v7_schema_fingerprint()
    integration = selector_v7_integration_fingerprint()
    assert first == second == integration.schema_fingerprint
    assert integration.schema_version == SELECTOR_V7_SCHEMA_VERSION
    assert integration.workplan_schema == "selector-v7-redesign-workplan/v4"
    assert integration.canonicalization_version.endswith("/v3")
    row = arm(1)
    with pytest.raises(ValueError, match="schema version"):
        replace(manifest(row), schema_version="selector-v7-contract/v4.38-draft",
                manifest_hash="")


def test_numeric_provenance_does_not_claim_v6_authority_for_new_estimands():
    provenance = {row.constant_name: row.provenance
                  for row in NUMERIC_PROVENANCE_V7}
    assert "no_v6_authority" in provenance["rho_any_row_severe"]
    assert provenance["tau_harmed_fraction"] == "new_v7_preregistration"
    assert provenance["row_severe_cutoff_G_raw_px"] == (
        "new_v7_preregistered_net_gain_semantics"
    )
    assert provenance["pixel_harm_CVaR_level"] == (
        "numeric_reference_selector_v6"
    )


def test_offline_realization_cost_does_not_consume_prospective_runtime_budget():
    row = replace(
        arm(1, cost=2.0),
        offline_bank_acquisition_cost_ceiling=3.0,
        offline_bank_acquisition_cost_counter_ceiling=budget_counters(3),
        prospective_runtime_cost_ceiling=1.0,
        prospective_runtime_cost_counter_ceiling=cost_counters(),
        cost_hash="",
    )
    bank = manifest(row)
    receipt = realization(bank, row)
    assert receipt.offline_bank_acquisition_cost == pytest.approx(3.0)
    plan = propose_candidates_v7(
        bank, (score(row),), (receipt,), (), top_k=1,
        maximum_prospective_runtime_cost=1.0,
        maximum_prospective_runtime_cost_counters=cost_counters(),
        planner_policy_hash="planner", planner_provenance_hash="provenance",
    )
    assert plan.reserved_prospective_runtime_cost == pytest.approx(1.0)


def test_realization_operator_drift_is_rejected_before_planning():
    row = arm(1)
    bank = manifest(row)
    forged = replace(
        realization(bank, row),
        realized_operator_hash="wrong-operator",
        receipt_hash="",
    )
    with pytest.raises(ValueError, match="binding"):
        propose_candidates_v7(
            bank, (score(row),), (forged,), (), top_k=1,
            maximum_prospective_runtime_cost=4.0,
            maximum_prospective_runtime_cost_counters=budget_counters(),
            planner_policy_hash="planner", planner_provenance_hash="provenance",
        )
