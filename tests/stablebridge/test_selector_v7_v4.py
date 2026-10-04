from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import runpy

import pytest

import stablebridge.physical_repair as package
from stablebridge.physical_repair import selector_v7 as v1
from stablebridge.physical_repair import selector_v7_v4 as v3
from stablebridge.physical_repair.selector_v7_v4 import (
    ActionRiskVectorV7V4,
    AliasChronologyModeV7V4,
    AliasRunPlanV7V4,
    CalibrationChoiceFamilyReceiptV7V4,
    CalibrationReceiptV7V4,
    CandidateManifestV7V4,
    CandidateObservationV7V4,
    DecisionReasonV7V4,
    E235_OUTPUT_REPLAY_SCHEMA_HASH_V7V4,
    ExternalInferenceReplayAuthorityV7V4,
    ExternalInferenceReplayVerificationV7V4,
    ExternalReplayAuthorityModeV7V4,
    ImmutableNativeReceiptV7V4,
    ManifestFreezeReceiptV7V4,
    ObservationSourceV7V4,
    PlannerProposalV7V4,
    PlannerFitReceiptV7V4,
    PlannerFeatureSchemaReceiptV7V4,
    PlannerModelReceiptV7V4,
    PlannerScoreBankV7V4,
    PlannerScoreReceiptV7V4,
    PreRunAliasSealV7V4,
    RuntimeCostLedgerV7V4,
    RuntimeExecutionReceiptV7V4,
    RuntimeOperationV7V4,
    SplitAuthorityReceiptV7V4,
    PLANNER_ALGORITHM_ID_V7V4,
    PLANNER_ALGORITHM_VERSION_V7V4,
    PLANNER_CONFIG_HASH_V7V4,
    PLANNER_MODEL_SCHEMA_HASH_V7V4,
    PLANNER_SCHEMA_CONFIG_HASH_V7V4,
    PLANNER_SCHEMA_PRODUCER_CODE_HASH_V7V4,
    PLANNER_TARGET_ID_V7V4,
    SELECTOR_V7_V1_FAILED_REVIEW_SHA256,
    SELECTOR_V7_V2_REVIEWED_SHA256,
    SELECTOR_V7_V3_REVIEWED_SHA256,
    SELECTOR_V7_V4_SCHEMA_VERSION,
    calibration_choice_family_digest_v7_v4,
    make_planner_score_receipt_v7_v4,
    make_test_only_replay_verification_v7_v4,
    planner_model_identity_hash_v7_v4,
    planner_score_output_bytes_hash_v7_v4,
    propose_candidates_v7_v4 as _production_propose_candidates_v7_v4,
    propose_candidates_v7_v4_test_only,
    select_portfolio_v7_v4 as _production_select_portfolio_v7_v4,
    select_portfolio_v7_v4_test_only,
    selector_v7_v4_schema_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]
FIX = runpy.run_path(str(ROOT / "tests/stablebridge/test_selector_v7.py"))
_AUTHORITY_BY_MANIFEST: dict[str, ExternalInferenceReplayAuthorityV7V4] = {}
_VERIFICATION_BY_SCORE_BANK: dict[
    str, ExternalInferenceReplayVerificationV7V4
] = {}


def native_receipt(bank: v1.CandidateManifestV7) -> ImmutableNativeReceiptV7V4:
    native = bank.native
    return ImmutableNativeReceiptV7V4(
        source_hash=bank.source_hash,
        source_materialization_hash=native.rollback_hash,
        endpoint_id=native.endpoint_id,
        execution_unit_id=bank.execution_unit_id,
        support_hash=native.support_hash,
        support_policy_hash=bank.support_policy_hash,
        producer_hash="synthetic-native-producer",
        frozen_before_manifest=True,
    )


def manifest_v2(*arms: v1.CandidateArmV7, frozen: bool = True):
    bank = FIX["manifest"](*arms, frozen=frozen)
    freeze = ManifestFreezeReceiptV7V4(
        base_manifest_hash=bank.manifest_hash,
        candidate_family_hash=bank.candidate_family_hash,
        source_hash=bank.source_hash,
        target_fold_role=bank.fold_role,
        sr0a_manifest_receipt_hash="synthetic-sr0a-manifest-receipt",
        freeze_sequence=1,
        outcome_capability_absent=True,
        provenance_hash="synthetic-freeze-provenance",
    )
    registry = FIX["feature_allowlists"]()
    split = SplitAuthorityReceiptV7V4(
        outer_split_id="outer-split-synthetic",
        target_fold_id="fold-eval",
        calibration_fold_id="fold-cal",
        fit_fold_ids=("outer-fit-1", "outer-fit-2", "outer-fit-3"),
        calibration_fold_ids=("fold-cal",),
        evaluation_fold_ids=("fold-eval",),
        sr0a_manifest_receipt_hash="synthetic-sr0a-manifest-receipt",
        partition_root_hash="synthetic-partition-root",
        assignment_hash="synthetic-split-assignment",
        split_policy_hash="synthetic-split-policy",
        frozen_before_fit=True,
        provenance_hash="synthetic-split-provenance",
    )
    fit = PlannerFitReceiptV7V4(
        fit_fold_role=v1.FoldRoleV7.FIT,
        target_fold_role=bank.fold_role,
        target_fold_id="fold-eval",
        fit_fold_ids=("outer-fit-1", "outer-fit-2", "outer-fit-3"),
        split_authority=split,
        fit_data_hash="synthetic-fit-data",
        fit_target_hash="synthetic-fit-target-G",
        group_weighting_hash="synthetic-group-weights",
        standardization_artifact_hash="synthetic-fit-mean-std-bytes",
        standardization_receipt_hash="synthetic-fit-mean-std-receipt",
        availability=v1.AvailabilityV7.AVAILABLE,
        availability_reason="available",
        frozen_before_target_fold=True,
        provenance_hash="synthetic-fit-provenance",
    )
    feature_schema = PlannerFeatureSchemaReceiptV7V4(
        feature_registry=registry,
        schema_producer_code_hash=PLANNER_SCHEMA_PRODUCER_CODE_HASH_V7V4,
        schema_config_hash=PLANNER_SCHEMA_CONFIG_HASH_V7V4,
        frozen_before_fit=True,
        provenance_hash="synthetic-schema-provenance",
    )
    model_bytes_hash = "synthetic-planner-model-bytes"
    external_authority = ExternalInferenceReplayAuthorityV7V4(
        integration_package_id="sr0a-e235-synthetic-fixture",
        integration_package_version="TEST_ONLY-v1",
        integration_package_hash="synthetic-e235-package",
        package_freeze_receipt_hash="synthetic-e235-package-freeze",
        integration_allowlist_hash="synthetic-e235-allowlist",
        root_review_seal_hash="TEST_ONLY_NO_ROOT_SEAL",
        root_review_status="TEST_ONLY",
        executor_code_hash="synthetic-feature-producer-code",
        producer_code_hash="synthetic-inference-producer-code",
        producer_config_hash=PLANNER_CONFIG_HASH_V7V4,
        runtime_hash="synthetic-inference-runtime",
        planner_model_bytes_hash=model_bytes_hash,
        planner_feature_schema_hash=feature_schema.schema_hash,
        planner_model_schema_hash=PLANNER_MODEL_SCHEMA_HASH_V7V4,
        output_replay_schema_hash=E235_OUTPUT_REPLAY_SCHEMA_HASH_V7V4,
        frozen_before_score=True,
        mode=ExternalReplayAuthorityModeV7V4.TEST_ONLY,
    )
    model_hash = planner_model_identity_hash_v7_v4(
        fit, feature_schema, model_bytes_hash, external_authority.authority_hash,
    )
    model = PlannerModelReceiptV7V4(
        fit_receipt=fit,
        feature_registry=registry,
        feature_schema_receipt=feature_schema,
        model_bytes_hash=model_bytes_hash,
        required_e235_replay_authority_hash=external_authority.authority_hash,
        model_hash=model_hash,
        model_provenance_hash="fit-only-planner",
        feature_schema_hash=feature_schema.schema_hash,
        algorithm_id=PLANNER_ALGORITHM_ID_V7V4,
        algorithm_version=PLANNER_ALGORITHM_VERSION_V7V4,
        config_hash=PLANNER_CONFIG_HASH_V7V4,
        target_id=PLANNER_TARGET_ID_V7V4,
        frozen_before_scoring=True,
    )
    manifest = CandidateManifestV7V4(
        bank, native_receipt(bank), freeze, model,
    )
    _AUTHORITY_BY_MANIFEST[manifest.manifest_v4_hash] = external_authority
    return bank, manifest


def explicit_score_receipt(
    manifest: CandidateManifestV7V4,
    row: v1.CandidateArmV7,
    score: v1.PlannerCandidateV7,
) -> PlannerScoreReceiptV7V4:
    authority = _AUTHORITY_BY_MANIFEST[manifest.manifest_v4_hash]
    return make_planner_score_receipt_v7_v4(
        manifest, row, score,
        feature_artifact_bytes_hash=f"feature-bytes:{row.planned_arm_id}",
        feature_producer_code_hash=authority.executor_code_hash,
        feature_producer_config_hash=authority.producer_config_hash,
        inference_producer_code_hash=authority.producer_code_hash,
        inference_runtime_hash=authority.runtime_hash,
        inference_run_id=f"inference:{row.planned_arm_id}",
        external_replay_receipt_hash=f"e235-replay:{row.planned_arm_id}",
        provenance_hash=f"synthetic-score:{row.planned_arm_id}",
    )


def direct_plan(
    manifest: CandidateManifestV7V4,
    row: v1.CandidateArmV7,
    *,
    run_id: str | None = None,
) -> AliasRunPlanV7V4:
    return AliasRunPlanV7V4(
        manifest_v4_hash=manifest.manifest_v4_hash,
        runtime_run_id=run_id or f"run:{row.candidate_id}",
        representative_planned_arm_id=row.planned_arm_id,
        alias_planned_arm_ids=(),
        alias_receipt_bindings=(),
        mode=AliasChronologyModeV7V4.DIRECT_NO_ALIAS,
        earliest_run_start_ns=100,
        pre_run_alias_seal=None,
        runtime_policy_hash="runtime-policy",
        provenance_hash=f"run-plan:{row.candidate_id}",
    )


def proposal_v2(
    manifest: CandidateManifestV7V4,
    rows: tuple[v1.CandidateArmV7, ...],
    receipts: tuple[v1.ArmRealizationReceiptV7, ...],
    *,
    scores: tuple[v1.PlannerCandidateV7, ...] | None = None,
    aliases: tuple[v1.ArmAliasReceiptV7, ...] = (),
    run_plans: tuple[AliasRunPlanV7V4, ...] | None = None,
    top_k: int = 2,
    maximum_cost: float = 4.0,
) -> PlannerProposalV7V4:
    plans = run_plans or tuple(direct_plan(manifest, row) for row in rows)
    base_scores = scores or tuple(FIX["score"](row) for row in rows)
    by_planned_id = {row.planned_arm_id: row for row in rows}
    try:
        score_receipts = tuple(
            explicit_score_receipt(
                manifest, by_planned_id[score.planned_arm_id], score,
            )
            for score in base_scores
        )
    except KeyError as error:
        raise ValueError("complete planner score bank contains injected row") from error
    authority = _AUTHORITY_BY_MANIFEST[manifest.manifest_v4_hash]
    verification = make_test_only_replay_verification_v7_v4(
        manifest, authority, score_receipts,
        provenance_hash="synthetic-external-replay-verification",
    )
    proposal = propose_candidates_v7_v4_test_only(
        manifest,
        score_receipts,
        receipts, aliases, plans,
        top_k=top_k,
        maximum_prospective_runtime_cost=maximum_cost,
        maximum_prospective_runtime_cost_counters=FIX["budget_counters"](),
        planner_algorithm_id=manifest.planner_model_receipt.algorithm_id,
        planner_algorithm_version=manifest.planner_model_receipt.algorithm_version,
        planner_config_hash=manifest.planner_model_receipt.config_hash,
        planner_provenance_hash=(
            manifest.planner_model_receipt.model_provenance_hash
        ),
        external_replay_authority=authority,
        external_replay_verification=verification,
    )
    _VERIFICATION_BY_SCORE_BANK[
        proposal.score_bank.score_bank_hash
    ] = verification
    return proposal


def select_portfolio_v7_v4(
    manifest: CandidateManifestV7V4,
    proposal: PlannerProposalV7V4,
    observations,
    risks,
    calibration,
    realizations,
    aliases,
    ledgers,
    **kwargs,
):
    authority = _AUTHORITY_BY_MANIFEST.get(
        getattr(manifest, "manifest_v4_hash", ""),
    )
    score_bank = getattr(proposal, "score_bank", None)
    verification = _VERIFICATION_BY_SCORE_BANK.get(
        getattr(score_bank, "score_bank_hash", ""),
    )
    if authority is not None and verification is not None:
        return select_portfolio_v7_v4_test_only(
            manifest, proposal, observations, risks, calibration,
            realizations, aliases, ledgers,
            external_replay_authority=authority,
            external_replay_verification=verification,
            **kwargs,
        )
    return _production_select_portfolio_v7_v4(
        manifest, proposal, observations, risks, calibration,
        realizations, aliases, ledgers,
        external_replay_authority=authority,
        external_replay_verification=verification,
        **kwargs,
    )


def execution_receipt(
    manifest: CandidateManifestV7V4,
    proposal: PlannerProposalV7V4,
    run_plan: AliasRunPlanV7V4,
    row: v1.CandidateArmV7,
    realization: v1.ArmRealizationReceiptV7,
    *,
    operation: RuntimeOperationV7V4 = RuntimeOperationV7V4.EXECUTE_ARM,
    reused_from: str | None = None,
    cost: float | None = None,
    counters: v1.CostCountersV7 | None = None,
    start: int = 110,
    finish: int = 120,
) -> RuntimeExecutionReceiptV7V4:
    return RuntimeExecutionReceiptV7V4(
        manifest_v4_hash=manifest.manifest_v4_hash,
        proposal_hash=proposal.proposal_hash,
        alias_run_plan_hash=run_plan.plan_hash,
        runtime_run_id=run_plan.runtime_run_id,
        planned_arm_id=row.planned_arm_id,
        realization_receipt_hash=realization.receipt_hash,
        executable_bytes_hash=str(realization.executable_bytes_hash),
        operation=operation,
        reused_from_planned_arm_id=reused_from,
        started_at_ns=start,
        finished_at_ns=finish,
        actual_prospective_runtime_cost=(
            row.prospective_runtime_cost_ceiling if cost is None else cost
        ),
        actual_prospective_runtime_cost_counters=(
            row.prospective_runtime_cost_counter_ceiling
            if counters is None else counters
        ),
        cost_account=v1.CostAccountV7.PROSPECTIVE_RUNTIME,
        provenance_hash=f"runtime:{row.candidate_id}:{operation.value}",
    )


def ledger(
    manifest: CandidateManifestV7V4,
    proposal: PlannerProposalV7V4,
    run_plan: AliasRunPlanV7V4,
    execution_receipts: tuple[RuntimeExecutionReceiptV7V4, ...],
    *,
    discovered_at: int | None = None,
) -> RuntimeCostLedgerV7V4:
    counters = v1.CostCountersV7.zero(
        cache_state=execution_receipts[0].actual_prospective_runtime_cost_counters.cache_state,
    )
    for receipt in execution_receipts:
        counters = counters.plus(receipt.actual_prospective_runtime_cost_counters)
    return RuntimeCostLedgerV7V4(
        manifest_v4_hash=manifest.manifest_v4_hash,
        proposal_hash=proposal.proposal_hash,
        alias_run_plan_hash=run_plan.plan_hash,
        runtime_run_id=run_plan.runtime_run_id,
        representative_planned_arm_id=run_plan.representative_planned_arm_id,
        alias_planned_arm_ids=run_plan.alias_planned_arm_ids,
        mode=run_plan.mode,
        execution_receipts=execution_receipts,
        post_hoc_discovered_at_ns=discovered_at,
        charged_prospective_runtime_cost=sum(
            row.actual_prospective_runtime_cost for row in execution_receipts
        ),
        charged_prospective_runtime_cost_counters=counters,
        cost_account=v1.CostAccountV7.PROSPECTIVE_RUNTIME,
        provenance_hash=f"ledger:{run_plan.runtime_run_id}",
    )


def observation(
    manifest: CandidateManifestV7V4,
    proposal: PlannerProposalV7V4,
    row: v1.CandidateArmV7,
    realization: v1.ArmRealizationReceiptV7,
    *,
    runtime_ledger: RuntimeCostLedgerV7V4 | None,
) -> CandidateObservationV7V4:
    return CandidateObservationV7V4(
        manifest_v4_hash=manifest.manifest_v4_hash,
        proposal_hash=proposal.proposal_hash,
        candidate_id=row.candidate_id,
        planned_arm_id=row.planned_arm_id,
        realization_receipt_hash=realization.receipt_hash,
        action_hash=row.action_hash,
        control_hash=row.control_hash,
        endpoint_hash=row.endpoint_hash,
        support_hash=row.support_hash,
        realized_support_hash=str(realization.realized_support_hash),
        interaction_hash=row.interaction_hash,
        cost_hash=row.cost_hash,
        source_hash=manifest.base_manifest.source_hash,
        planner_allowlist_hash=manifest.base_manifest.planner_allowlist_hash,
        assessor_allowlist_hash=manifest.base_manifest.assessor_allowlist_hash,
        preaction_shared_feature_hash=f"before:{row.candidate_id}",
        feature_schema_hash="feature-schema",
        provenance_hash=f"observation:{row.candidate_id}",
        availability=v1.AvailabilityV7.AVAILABLE,
        availability_reason="available",
        runtime_features=tuple(
            v1.RuntimeFeatureV7(name, 0.2)
            for name in sorted(v1.ASSESSOR_POSTACTION_EXTENSION_FIELDS_V7)
        ),
        evidence_blocks=tuple(
            v1.EvidenceBlockReceiptV7(
                block=block,
                availability=v1.AvailabilityV7.AVAILABLE,
                source_hash=f"source:{block.value}",
                receipt_hash=f"receipt:{block.value}",
            )
            for block in v1.EvidenceBlockV7
        ),
        evidence_source_hashes=("cc-source", "response-source"),
        observed_unit_ids=row.unit_ids,
        atomically_observed=True,
        observation_source=(
            ObservationSourceV7V4.PROSPECTIVE_RUNTIME_OBSERVATION
            if runtime_ledger is not None
            else ObservationSourceV7V4.FROZEN_FAMILY_OBSERVATION
        ),
        runtime_cost_ledger_hash=(
            runtime_ledger.ledger_hash if runtime_ledger is not None else None
        ),
    )


def calibration_v2(
    manifest: CandidateManifestV7V4,
    proposal: PlannerProposalV7V4,
    rows: tuple[v1.CandidateArmV7, ...],
    observations: tuple[CandidateObservationV7V4, ...],
    *,
    covered: tuple[v1.CandidateArmV7, ...] | None = None,
) -> CalibrationReceiptV7V4:
    covered_rows = covered or rows
    base = FIX["calibration"](
        manifest.base_manifest, *covered_rows,
    )
    family = CalibrationChoiceFamilyReceiptV7V4(
        manifest_v4_hash=manifest.manifest_v4_hash,
        choice_family_digest=calibration_choice_family_digest_v7_v4(
            manifest, proposal,
        ),
        choice_family_cardinality=len(covered_rows),
        covered_planned_arm_ids=tuple(
            row.planned_arm_id for row in covered_rows
        ),
        score_bank_universe_digest=proposal.score_bank.universe_digest,
        calibration_fold_id=base.calibration_fold_id,
        calibration_data_hash=base.calibration_data_hash,
        assessor_model_hash=base.model_hash,
        simultaneous_correction_id="component-max-residual-five-heads",
        frozen_before_evaluation_runtime=True,
        provenance_hash="synthetic-CAL-family-provenance",
    )
    return CalibrationReceiptV7V4(
        manifest_v4_hash=manifest.manifest_v4_hash,
        base_receipt=base,
        choice_family_receipt=family,
        assessor_model_provenance_hash="assessor-model-provenance",
        assessor_algorithm_id="five-head-assessor",
        assessor_algorithm_version="v1",
        assessor_feature_schema_hash=base.feature_schema_hash,
        assessor_config_hash="five-head-config",
        simultaneous_correction_id="component-max-residual-five-heads",
        frozen_before_decision=True,
    )


def risk_v2(
    manifest: CandidateManifestV7V4,
    proposal: PlannerProposalV7V4,
    observation: CandidateObservationV7V4,
    row: v1.CandidateArmV7,
    calibration: CalibrationReceiptV7V4,
    *,
    benefit: float = 0.8,
    harm: float = 0.02,
    severe: float = 0.01,
) -> ActionRiskVectorV7V4:
    base = v1.ActionRiskVectorV7(
        manifest_hash=manifest.base_manifest.manifest_hash,
        proposal_hash=proposal.proposal_hash,
        observation_hash=observation.observation_hash,
        candidate_id=row.candidate_id,
        planned_arm_id=row.planned_arm_id,
        action_hash=row.action_hash,
        control_hash=row.control_hash,
        endpoint_hash=row.endpoint_hash,
        support_hash=row.support_hash,
        interaction_hash=row.interaction_hash,
        cost_hash=row.cost_hash,
        assessor_hash=calibration.assessor_model_hash,
        assessor_provenance_hash=(
            calibration.assessor_model_provenance_hash
        ),
        assessor_feature_profile=v1.ASSESSOR_FEATURE_PROFILE_V7,
        assessor_allowlist_hash=manifest.base_manifest.assessor_allowlist_hash,
        prediction_kind=(
            v1.PredictionKindV7.ASSESSOR_MODEL_PREDICTION_PREDECISION
        ),
        calibration_receipt_hash=calibration.calibration_v4_hash,
        availability=v1.AvailabilityV7.AVAILABLE,
        availability_reason="available",
        feature_eligible=True,
        calibration_valid=True,
        bound_bookkeeping_valid=True,
        bookkeeping_receipt_hash="G-hat-equals-B-hat-minus-H-hat",
        predicted_benefit=benefit,
        benefit_lower=benefit,
        predicted_harm=harm,
        harm_upper=harm,
        predicted_any_row_severe_probability=severe / 2,
        any_row_severe_probability_upper=severe,
        predicted_harmed_pixel_fraction=0.01,
        harmed_pixel_fraction_upper=0.02,
        predicted_pixel_harm_cvar95=0.05,
        pixel_harm_cvar95_upper=0.10,
    )
    return ActionRiskVectorV7V4(
        base_vector=base,
        assessor_model_hash=calibration.assessor_model_hash,
        assessor_model_provenance_hash=(
            calibration.assessor_model_provenance_hash
        ),
        assessor_algorithm_id=calibration.assessor_algorithm_id,
        assessor_algorithm_version=calibration.assessor_algorithm_version,
        assessor_feature_schema_hash=(
            calibration.assessor_feature_schema_hash
        ),
        assessor_config_hash=calibration.assessor_config_hash,
        calibration_v4_hash=calibration.calibration_v4_hash,
    )


def bundle_v2(
    *rows: v1.CandidateArmV7,
    top_k: int = 2,
    utilities: tuple[float, ...] | None = None,
):
    base, manifest = manifest_v2(*rows)
    receipts = tuple(FIX["realization"](base, row) for row in rows)
    plans = tuple(direct_plan(manifest, row) for row in rows)
    scores = tuple(
        FIX["score"](row, utility=(utilities or ((1.0,) * len(rows)))[index])
        for index, row in enumerate(rows)
    )
    proposal = proposal_v2(
        manifest, tuple(rows), receipts, scores=scores,
        run_plans=plans, top_k=top_k,
    )
    by_id = {row.planned_arm_id: row for row in rows}
    realization_by_id = {
        row.planned_arm_id: receipt for row, receipt in zip(rows, receipts)
    }
    plan_by_id = {row.representative_planned_arm_id: row for row in plans}
    ledgers = []
    for planned_id in proposal.selected_planned_arm_ids:
        row = by_id[planned_id]
        run_plan = plan_by_id[planned_id]
        execution = execution_receipt(
            manifest, proposal, run_plan, row, realization_by_id[planned_id],
        )
        ledgers.append(ledger(manifest, proposal, run_plan, (execution,)))
    ledger_by_id = {row.representative_planned_arm_id: row for row in ledgers}
    observations = tuple(
        observation(
            manifest, proposal, row, realization_by_id[row.planned_arm_id],
            runtime_ledger=ledger_by_id.get(row.planned_arm_id),
        )
        for row in rows
    )
    calibration = calibration_v2(
        manifest, proposal, tuple(rows), observations,
    )
    risks = tuple(
        risk_v2(manifest, proposal, seen, row, calibration)
        for row, seen in zip(rows, observations)
    )
    return (
        base, manifest, receipts, plans, proposal, tuple(ledgers),
        observations, calibration, risks,
    )


def test_v4_happy_path_commits_and_fingerprint_records_failed_predecessor():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    decision = select_portfolio_v7_v4(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.COMMIT
    assert decision.selected_candidate_id == row.candidate_id
    assert manifest.schema_version == SELECTOR_V7_V4_SCHEMA_VERSION
    assert manifest.predecessor_failed_review_sha256 == (
        SELECTOR_V7_V3_REVIEWED_SHA256
    )
    assert len(selector_v7_v4_schema_fingerprint()) == 64


def test_v4_public_exports_are_complete():
    assert all(hasattr(package, name) for name in v3.__all__)
    assert set(v3.__all__).issubset(package.__all__)


def test_r01_destructive_action0_impostor_rejects_at_v2_manifest_boundary():
    row = FIX["arm"](1)
    base, authority = manifest_v2(row)
    forged_native = replace(
        base.native,
        mechanism_id="destructive-restoration",
        operator_version="mutable-v99",
        action_hash="not-native-action",
        control_hash="not-native-control",
        rollback_hash="not-native-rollback",
        planned_arm_id="",
    )
    forged = replace(
        base, candidates=(forged_native, row),
        candidate_family_hash="", manifest_hash="",
    )
    with pytest.raises(ValueError, match="canonical and immutable"):
        CandidateManifestV7V4(
            forged, native_receipt(base), authority.freeze_receipt,
            authority.planner_model_receipt,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mechanism_id", "other"),
        ("operator_version", "mutable-v2"),
        ("endpoint_id", "other-endpoint"),
        ("support_hash", "other-support"),
        ("rollback_hash", "other-rollback"),
        ("action_hash", "other-action-hash"),
        ("control_hash", "other-control-hash"),
    ),
)
def test_r01_each_canonical_native_binding_is_locked(field, value):
    row = FIX["arm"](1)
    base, authority = manifest_v2(row)
    forged_native = replace(base.native, **{field: value, "planned_arm_id": ""})
    forged = replace(
        base, candidates=(forged_native, row),
        candidate_family_hash="", manifest_hash="",
    )
    with pytest.raises(ValueError, match="canonical and immutable"):
        CandidateManifestV7V4(
            forged, native_receipt(base), authority.freeze_receipt,
            authority.planner_model_receipt,
        )


def test_r01_commit_boundary_revalidates_native_after_object_tampering():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    object.__setattr__(
        manifest.base_manifest.native, "mechanism_id", "destructive-restoration",
    )
    decision = select_portfolio_v7_v4(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason in {
        DecisionReasonV7V4.MALFORMED_INPUT,
        DecisionReasonV7V4.NATIVE_IDENTITY_INVALID,
    }


def test_r02_builder_requires_complete_exact_score_bank():
    high = FIX["arm"](1, action_identity="high")
    low = FIX["arm"](2, action_identity="low")
    base, manifest = manifest_v2(high, low)
    receipts = (
        FIX["realization"](base, high), FIX["realization"](base, low),
    )
    plans = (direct_plan(manifest, high), direct_plan(manifest, low))
    with pytest.raises(ValueError, match="score bank"):
        proposal_v2(
            manifest, (high, low), receipts,
            scores=(FIX["score"](low, utility=-5.0),),
            run_plans=plans, top_k=1,
        )


def test_r02_commit_replays_before_feature_completeness_and_topk():
    high = FIX["arm"](1, action_identity="high")
    low = FIX["arm"](2, action_identity="low")
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(
        high, low, top_k=1, utilities=(10.0, 1.0),
    )
    assert proposal.selected_planned_arm_ids == (high.planned_arm_id,)
    incomplete_score = replace(
        proposal.score_bank.scores[0], before_features_complete=False,
    )
    score_rows = tuple(
        incomplete_score if row.planned_arm_id == high.planned_arm_id else row
        for row in proposal.score_bank.scores
    )
    with pytest.raises(ValueError, match="model/FIT authority"):
        PlannerScoreBankV7V4(
            manifest_v4_hash=proposal.score_bank.manifest_v4_hash,
            candidate_family_hash=proposal.score_bank.candidate_family_hash,
            planner_allowlist_hash=proposal.score_bank.planner_allowlist_hash,
            planner_model_receipt=proposal.score_bank.planner_model_receipt,
            universe_digest=proposal.score_bank.universe_digest,
            universe_cardinality=proposal.score_bank.universe_cardinality,
            scores=score_rows,
            frozen_before_observation=True,
            score_receipts=proposal.score_bank.score_receipts,
        )


def test_r02_handcrafted_partial_score_bank_fails_closed_at_commit():
    first = FIX["arm"](1)
    second = FIX["arm"](2)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(first, second, top_k=1)
    kept = proposal.score_bank.scores[:1]
    partial = PlannerScoreBankV7V4(
        manifest_v4_hash=proposal.score_bank.manifest_v4_hash,
        candidate_family_hash=proposal.score_bank.candidate_family_hash,
        planner_allowlist_hash=proposal.score_bank.planner_allowlist_hash,
        planner_model_receipt=proposal.score_bank.planner_model_receipt,
        universe_digest="forged-partial-universe",
        universe_cardinality=1,
        scores=kept,
        frozen_before_observation=True,
        score_receipts=proposal.score_bank.score_receipts[:1],
    )
    forged = replace(
        proposal, score_bank=partial,
        deterministic_ranked_planned_arm_ids=(kept[0].planned_arm_id,),
        selected_planned_arm_ids=(kept[0].planned_arm_id,),
        proposal_hash="",
    )
    decision = select_portfolio_v7_v4(
        manifest, forged, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.PLANNER_AUTHORITY_MISMATCH


def test_r02_full_bank_handcrafted_lower_top1_is_replayed_and_rejected():
    high = FIX["arm"](1, action_identity="high")
    low = FIX["arm"](2, action_identity="low")
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(
        high, low, top_k=1, utilities=(100.0, -100.0),
    )
    assert proposal.selected_planned_arm_ids == (high.planned_arm_id,)
    forged = replace(
        proposal,
        selected_planned_arm_ids=(low.planned_arm_id,),
        proposal_hash="",
    )
    decision = select_portfolio_v7_v4(
        manifest, forged, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.PROPOSAL_REPLAY_MISMATCH


def test_r02_score_candidate_injection_is_not_a_complete_bank():
    row = FIX["arm"](1)
    base, manifest = manifest_v2(row)
    receipt = FIX["realization"](base, row)
    plan = direct_plan(manifest, row)
    injected_arm = FIX["arm"](99)
    with pytest.raises(ValueError, match="score bank"):
        proposal_v2(
            manifest, (row,), (receipt,),
            scores=(FIX["score"](row), FIX["score"](injected_arm)),
            run_plans=(plan,), top_k=1,
        )


def test_r03_uncalibrated_assessor_model_cannot_commit():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    forged_base = replace(
        risks[0].base_vector,
        assessor_hash="different-uncalibrated-model",
        assessor_provenance_hash="different-model-provenance",
        risk_vector_hash="",
    )
    forged = replace(
        risks[0], base_vector=forged_base,
        assessor_model_hash="different-uncalibrated-model",
        assessor_model_provenance_hash="different-model-provenance",
        risk_vector_v4_hash="",
    )
    decision = select_portfolio_v7_v4(
        manifest, proposal, observations, (forged,), calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.ASSESSOR_CALIBRATION_MISMATCH


def test_r03_algorithm_schema_and_config_are_exactly_calibration_bound():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    for field in (
        "assessor_algorithm_id", "assessor_algorithm_version",
        "assessor_feature_schema_hash", "assessor_config_hash",
    ):
        forged = replace(
            risks[0], **{field: f"different:{field}", "risk_vector_v4_hash": ""},
        )
        decision = select_portfolio_v7_v4(
            manifest, proposal, observations, (forged,), calibration,
            receipts, (), ledgers,
        )
        assert decision.reason is (
            DecisionReasonV7V4.ASSESSOR_CALIBRATION_MISMATCH
        )


def test_nested_base_risk_and_calibration_hash_drift_fail_closed():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    object.__setattr__(risks[0].base_vector, "harm_upper", float("nan"))
    decision = select_portfolio_v7_v4(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.NO_SAFE_CANDIDATE
    assert decision.arm_reasons == ((
        row.planned_arm_id, DecisionReasonV7V4.FEATURE_INELIGIBLE,
    ),)

    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    object.__setattr__(calibration.base_receipt, "maximum_harm", 1e9)
    decision = select_portfolio_v7_v4(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.MALFORMED_INPUT


def test_malformed_top_level_proposal_returns_typed_native_decision():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    decision = select_portfolio_v7_v4(
        manifest, None, observations, risks, calibration,  # type: ignore[arg-type]
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.MALFORMED_INPUT


def test_r04_calibration_must_cover_full_observed_choice_family():
    first = FIX["arm"](1)
    selected = FIX["arm"](2)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(first, selected, top_k=1)
    selected_row = next(
        row for row in (first, selected)
        if row.planned_arm_id in proposal.selected_planned_arm_ids
    )
    truncated = calibration_v2(
        manifest, proposal, (first, selected), observations,
        covered=(selected_row,),
    )
    truncated_risks = tuple(
        risk_v2(manifest, proposal, seen, row, truncated)
        for row, seen in zip((first, selected), observations)
    )
    decision = select_portfolio_v7_v4(
        manifest, proposal, observations, truncated_risks, truncated,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.CALIBRATION_FAMILY_INCOMPLETE


def test_selected_runtime_observation_and_risk_are_required_independently_of_cal():
    first = FIX["arm"](1)
    second = FIX["arm"](2)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(first, second, top_k=1)
    selected = set(proposal.selected_planned_arm_ids)
    selected_observations = tuple(
        row for row in observations if row.planned_arm_id in selected
    )
    selected_risks = tuple(
        row for row in risks if row.base_vector.planned_arm_id in selected
    )
    for observed, assessed in (
        ((), selected_risks),
        (selected_observations, ()),
    ):
        decision = select_portfolio_v7_v4(
            manifest, proposal, observed, assessed, calibration,
            receipts, (), ledgers,
        )
        assert decision.state is v1.PortfolioStateV7.NATIVE
        assert decision.reason is DecisionReasonV7V4.NO_SAFE_CANDIDATE
        assert decision.arm_reasons


def alias_fixture(mode: AliasChronologyModeV7V4):
    original = FIX["arm"](1)
    equivalent = replace(
        FIX["arm"](2, action_identity="byte-equivalent-alias"),
        action_hash=original.action_hash,
    )
    base, manifest = manifest_v2(original, equivalent)
    original_receipt = FIX["realization"](
        base, original, output_hash="same-output",
    )
    equivalent_receipt = v1.ArmRealizationReceiptV7(
        manifest_hash=base.manifest_hash,
        planned_arm_id=equivalent.planned_arm_id,
        source_hash=base.source_hash,
        status=v1.ArmRealizationStatusV7.COMPLETE,
        status_reason="complete",
        realized_input_hash=original_receipt.realized_input_hash,
        realized_output_hash=original_receipt.realized_output_hash,
        realized_support_hash=original_receipt.realized_support_hash,
        realized_operator_hash=original_receipt.realized_operator_hash,
        tensor_shape=original_receipt.tensor_shape,
        tensor_dtype=original_receipt.tensor_dtype,
        realized_bytes=original_receipt.realized_bytes,
        offline_bank_acquisition_cost=(
            equivalent.offline_bank_acquisition_cost_ceiling
        ),
        offline_bank_acquisition_cost_counters=(
            equivalent.offline_bank_acquisition_cost_counter_ceiling
        ),
        cost_account=v1.CostAccountV7.OFFLINE_BANK_ACQUISITION,
        provenance_hash="alias-realization-provenance",
    )
    representative, alias_member = sorted(
        (original, equivalent), key=lambda row: row.planned_arm_id,
    )
    receipt_by_id = {
        original.planned_arm_id: original_receipt,
        equivalent.planned_arm_id: equivalent_receipt,
    }
    representative_receipt = receipt_by_id[representative.planned_arm_id]
    alias_member_receipt = receipt_by_id[alias_member.planned_arm_id]
    verify_counters = v1.CostCountersV7(
        natural_forwards=0, reused_forwards=0, candidate_forwards=0,
        reverse_forwards=0, cpu_seconds=0.01, gpu_seconds=0.0,
        wall_seconds=0.01, bytes_moved=48, peak_allocated_bytes=0,
        peak_reserved_bytes=0, matcher_trajectories=0,
        cache_state="cold",
    )
    alias = v1.ArmAliasReceiptV7(
        manifest_hash=base.manifest_hash,
        representative_planned_arm_id=representative.planned_arm_id,
        alias_planned_arm_id=alias_member.planned_arm_id,
        representative_realization_receipt_hash=(
            representative_receipt.receipt_hash
        ),
        alias_realization_receipt_hash=alias_member_receipt.receipt_hash,
        executable_bytes_hash=str(
            representative_receipt.executable_bytes_hash
        ),
        realized_operator_hash=str(
            representative_receipt.realized_operator_hash
        ),
        prospective_runtime_verification_cost=0.1,
        prospective_runtime_verification_cost_counters=verify_counters,
        cost_account=v1.CostAccountV7.PROSPECTIVE_RUNTIME,
        provenance_hash="alias-proof",
    )
    bindings = ((alias_member.planned_arm_id, alias.receipt_hash),)
    seal = None
    if mode is AliasChronologyModeV7V4.PRE_RUN_FROZEN_REUSE:
        seal = PreRunAliasSealV7V4(
            manifest_v4_hash=manifest.manifest_v4_hash,
            representative_planned_arm_id=representative.planned_arm_id,
            alias_planned_arm_ids=(alias_member.planned_arm_id,),
            alias_receipt_bindings=bindings,
            sealed_at_ns=50,
            reuse_policy_hash="exact-byte-reuse-policy",
            provenance_hash="pre-run-alias-seal",
        )
    plan = AliasRunPlanV7V4(
        manifest_v4_hash=manifest.manifest_v4_hash,
        runtime_run_id=f"runtime:{mode.value}",
        representative_planned_arm_id=representative.planned_arm_id,
        alias_planned_arm_ids=(alias_member.planned_arm_id,),
        alias_receipt_bindings=bindings,
        mode=mode,
        earliest_run_start_ns=100,
        pre_run_alias_seal=seal,
        runtime_policy_hash="runtime-policy",
        provenance_hash=f"run-plan:{mode.value}",
    )
    receipts = (original_receipt, equivalent_receipt)
    proposal = proposal_v2(
        manifest, (representative,), receipts,
        scores=(FIX["score"](representative),), aliases=(alias,),
        run_plans=(plan,), top_k=1,
    )
    return {
        "base": base,
        "manifest": manifest,
        "representative": representative,
        "alias_member": alias_member,
        "receipts": receipts,
        "representative_receipt": representative_receipt,
        "alias_member_receipt": alias_member_receipt,
        "alias": alias,
        "verify_counters": verify_counters,
        "plan": plan,
        "proposal": proposal,
    }


def finish_alias_bundle(case, executions, *, discovered_at=None):
    runtime_ledger = ledger(
        case["manifest"], case["proposal"], case["plan"],
        tuple(executions), discovered_at=discovered_at,
    )
    seen = observation(
        case["manifest"], case["proposal"], case["representative"],
        case["representative_receipt"], runtime_ledger=runtime_ledger,
    )
    calibration = calibration_v2(
        case["manifest"], case["proposal"], (case["representative"],),
        (seen,),
    )
    assessed = risk_v2(
        case["manifest"], case["proposal"], seen,
        case["representative"], calibration,
    )
    decision = select_portfolio_v7_v4(
        case["manifest"], case["proposal"], (seen,), (assessed,),
        calibration, case["receipts"], (case["alias"],),
        (runtime_ledger,),
    )
    return decision, runtime_ledger


def test_r05_pre_run_frozen_reuse_charges_one_execution_and_each_verification():
    case = alias_fixture(AliasChronologyModeV7V4.PRE_RUN_FROZEN_REUSE)
    representative_execution = execution_receipt(
        case["manifest"], case["proposal"], case["plan"],
        case["representative"], case["representative_receipt"],
    )
    alias_verification = execution_receipt(
        case["manifest"], case["proposal"], case["plan"],
        case["alias_member"], case["alias_member_receipt"],
        operation=RuntimeOperationV7V4.VERIFY_ALIAS_REUSE,
        reused_from=case["representative"].planned_arm_id,
        cost=case["alias"].prospective_runtime_verification_cost,
        counters=case["verify_counters"], start=121, finish=122,
    )
    decision, runtime_ledger = finish_alias_bundle(
        case, (representative_execution, alias_verification),
    )
    assert decision.state is v1.PortfolioStateV7.COMMIT
    assert case["proposal"].reserved_prospective_runtime_cost == pytest.approx(1.1)
    assert runtime_ledger.charged_prospective_runtime_cost == pytest.approx(1.1)
    assert all(
        row.offline_bank_acquisition_cost == pytest.approx(1.0)
        for row in case["receipts"]
    )


@pytest.mark.parametrize(
    "mode", (
        AliasChronologyModeV7V4.POST_HOC_DISCOVERY,
        AliasChronologyModeV7V4.SAME_RUN_DUPLICATE,
    ),
)
def test_r05_late_or_same_run_alias_charges_every_executed_arm(mode):
    case = alias_fixture(mode)
    executions = tuple(
        execution_receipt(
            case["manifest"], case["proposal"], case["plan"], arm, receipt,
            start=110 + 20 * index, finish=120 + 20 * index,
        )
        for index, (arm, receipt) in enumerate((
            (case["representative"], case["representative_receipt"]),
            (case["alias_member"], case["alias_member_receipt"]),
        ))
    )
    decision, runtime_ledger = finish_alias_bundle(
        case, executions,
        discovered_at=(150 if mode is AliasChronologyModeV7V4.POST_HOC_DISCOVERY
                       else None),
    )
    assert decision.state is v1.PortfolioStateV7.COMMIT
    assert case["proposal"].reserved_prospective_runtime_cost == pytest.approx(2.0)
    assert runtime_ledger.charged_prospective_runtime_cost == pytest.approx(2.0)


def test_r05_pre_run_reuse_cannot_hide_same_run_double_execution():
    case = alias_fixture(AliasChronologyModeV7V4.PRE_RUN_FROZEN_REUSE)
    executions = tuple(
        execution_receipt(
            case["manifest"], case["proposal"], case["plan"], arm, receipt,
            start=110 + 20 * index, finish=120 + 20 * index,
        )
        for index, (arm, receipt) in enumerate((
            (case["representative"], case["representative_receipt"]),
            (case["alias_member"], case["alias_member_receipt"]),
        ))
    )
    decision, _ = finish_alias_bundle(case, executions)
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.ALIAS_CHRONOLOGY_INVALID


def test_r05_post_hoc_discovery_cannot_claim_cheap_presealed_reuse():
    case = alias_fixture(AliasChronologyModeV7V4.POST_HOC_DISCOVERY)
    representative_execution = execution_receipt(
        case["manifest"], case["proposal"], case["plan"],
        case["representative"], case["representative_receipt"],
    )
    alias_verification = execution_receipt(
        case["manifest"], case["proposal"], case["plan"],
        case["alias_member"], case["alias_member_receipt"],
        operation=RuntimeOperationV7V4.VERIFY_ALIAS_REUSE,
        reused_from=case["representative"].planned_arm_id,
        cost=case["alias"].prospective_runtime_verification_cost,
        counters=case["verify_counters"], start=121, finish=122,
    )
    decision, _ = finish_alias_bundle(
        case, (representative_execution, alias_verification),
        discovered_at=150,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.ALIAS_CHRONOLOGY_INVALID


def test_r05_run_ids_timestamps_and_cost_accounts_are_structural():
    case = alias_fixture(AliasChronologyModeV7V4.PRE_RUN_FROZEN_REUSE)
    late_seal = replace(
        case["plan"].pre_run_alias_seal,
        sealed_at_ns=case["plan"].earliest_run_start_ns,
        seal_hash="",
    )
    with pytest.raises(ValueError, match="not frozen before runtime"):
        replace(
            case["plan"], pre_run_alias_seal=late_seal, plan_hash="",
        )
    with pytest.raises(ValueError, match="wrong cost account"):
        RuntimeExecutionReceiptV7V4(
            manifest_v4_hash=case["manifest"].manifest_v4_hash,
            proposal_hash=case["proposal"].proposal_hash,
            alias_run_plan_hash=case["plan"].plan_hash,
            runtime_run_id=case["plan"].runtime_run_id,
            planned_arm_id=case["representative"].planned_arm_id,
            realization_receipt_hash=case["representative_receipt"].receipt_hash,
            executable_bytes_hash=str(
                case["representative_receipt"].executable_bytes_hash
            ),
            operation=RuntimeOperationV7V4.EXECUTE_ARM,
            reused_from_planned_arm_id=None,
            started_at_ns=110,
            finished_at_ns=120,
            actual_prospective_runtime_cost=1.0,
            actual_prospective_runtime_cost_counters=(
                case["representative"].prospective_runtime_cost_counter_ceiling
            ),
            cost_account=v1.CostAccountV7.OFFLINE_BANK_ACQUISITION,
            provenance_hash="wrong-account",
        )


def test_reviewed_failed_v1_bytes_remain_unchanged():
    assert hashlib.sha256(
        (ROOT / "src/stablebridge/physical_repair/selector_v7.py").read_bytes()
    ).hexdigest() == SELECTOR_V7_V1_FAILED_REVIEW_SHA256
    assert hashlib.sha256(
        (ROOT / "tests/stablebridge/test_selector_v7.py").read_bytes()
    ).hexdigest() == "e48df3e376d8a934f408ea7d83f2d777af9266412251f86c5ffb607d1216e144"


def test_r06_outcome_unfrozen_manifest_rejects_with_matching_freeze_metadata():
    row = FIX["arm"](1)
    base, authority = manifest_v2(row)
    late = replace(
        base, frozen_before_outcome=False,
        candidate_family_hash="", manifest_hash="",
    )
    late_freeze = replace(
        authority.freeze_receipt,
        base_manifest_hash=late.manifest_hash,
        candidate_family_hash=late.candidate_family_hash,
        source_hash=late.source_hash,
        target_fold_role=late.fold_role,
        receipt_hash="",
    )
    with pytest.raises(ValueError, match="not frozen outcome-blind"):
        CandidateManifestV7V4(
            late, native_receipt(late), late_freeze,
            authority.planner_model_receipt,
        )


def test_r06_commit_rechecks_manifest_freeze_after_recursive_decode():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    object.__setattr__(manifest.base_manifest, "frozen_before_outcome", False)
    decision = select_portfolio_v7_v4(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.MALFORMED_INPUT


def test_r07_risk_assessor_allowlist_drift_is_fail_closed_after_resigning():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    selected_id = proposal.selected_planned_arm_ids[0]
    selected_risk = next(
        item for item in risks
        if item.base_vector.planned_arm_id == selected_id
    )
    leaked_base = replace(
        selected_risk.base_vector,
        assessor_allowlist_hash="different-held-outcome-allowlist",
        risk_vector_hash="",
    )
    leaked = replace(
        selected_risk, base_vector=leaked_base, risk_vector_v4_hash="",
    )
    decision = select_portfolio_v7_v4(
        manifest, proposal, observations, (leaked,), calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.ASSESSOR_CALIBRATION_MISMATCH


def test_r07_calibration_fold_and_both_allowlists_are_revalidated():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    wrong_base = replace(
        calibration.base_receipt,
        target_fold_role=v1.FoldRoleV7.FIT,
        planner_allowlist_hash="wrong-planner-allowlist",
        assessor_allowlist_hash="wrong-assessor-allowlist",
        receipt_hash="",
    )
    wrong_calibration = replace(
        calibration, base_receipt=wrong_base, calibration_v4_hash="",
    )
    selected_id = proposal.selected_planned_arm_ids[0]
    selected_row = next(
        item for item in (row,) if item.planned_arm_id == selected_id
    )
    selected_seen = next(
        item for item in observations if item.planned_arm_id == selected_id
    )
    wrong_risk = risk_v2(
        manifest, proposal, selected_seen, selected_row, wrong_calibration,
    )
    decision = select_portfolio_v7_v4(
        manifest, proposal, (selected_seen,), (wrong_risk,),
        wrong_calibration, receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.CALIBRATION_FAMILY_INCOMPLETE


def test_r08_arbitrary_planner_algorithm_and_config_cannot_be_resigned():
    row = FIX["arm"](1)
    base, manifest = manifest_v2(row)
    receipt = FIX["realization"](base, row)
    plan = direct_plan(manifest, row)
    score_receipts = (explicit_score_receipt(
        manifest, row, FIX["score"](row),
    ),)
    authority = _AUTHORITY_BY_MANIFEST[manifest.manifest_v4_hash]
    verification = make_test_only_replay_verification_v7_v4(
        manifest, authority, score_receipts,
        provenance_hash="test-r08-verification",
    )
    with pytest.raises(ValueError, match="manifest-authorized"):
        propose_candidates_v7_v4_test_only(
            manifest, score_receipts, (receipt,), (), (plan,),
            top_k=1,
            maximum_prospective_runtime_cost=4.0,
            maximum_prospective_runtime_cost_counters=FIX["budget_counters"](),
            planner_algorithm_id="held-outcome-oracle",
            planner_algorithm_version="v666",
            planner_config_hash="lambda-zero-no-fit-standardization",
            planner_provenance_hash="unreviewed",
            external_replay_authority=authority,
            external_replay_verification=verification,
        )
    with pytest.raises(ValueError, match="algorithm/config/target authority"):
        replace(
            manifest.planner_model_receipt,
            algorithm_id="held-outcome-oracle", receipt_hash="",
        )


def test_r08_score_bank_model_receipt_must_equal_manifest_authority():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    with pytest.raises(ValueError, match="exact model bytes"):
        replace(
            manifest.planner_model_receipt,
            model_hash="different-unreviewed-model", receipt_hash="",
        )


def test_r09_k1_requires_only_selected_runtime_state_but_full_cal_authority():
    first = FIX["arm"](1)
    second = FIX["arm"](2)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(
        first, second, top_k=1, utilities=(10.0, 1.0),
    )
    selected = set(proposal.selected_planned_arm_ids)
    selected_observations = tuple(
        item for item in observations if item.planned_arm_id in selected
    )
    selected_risks = tuple(
        item for item in risks if item.base_vector.planned_arm_id in selected
    )
    decision = select_portfolio_v7_v4(
        manifest, proposal, selected_observations, selected_risks,
        calibration, receipts, (), ledgers,
    )
    assert len(selected) == len(ledgers) == len(selected_observations) == 1
    assert calibration.choice_family_cardinality == 2
    assert set(calibration.covered_planned_arm_ids) == {
        first.planned_arm_id, second.planned_arm_id,
    }
    assert decision.state is v1.PortfolioStateV7.COMMIT


def test_r09_unselected_typed_missing_and_nonfinite_state_cannot_force_native():
    first = FIX["arm"](1)
    second = FIX["arm"](2)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(
        first, second, top_k=1, utilities=(10.0, 1.0),
    )
    selected = set(proposal.selected_planned_arm_ids)
    chosen_observation = next(
        item for item in observations if item.planned_arm_id in selected
    )
    chosen_risk = next(
        item for item in risks if item.base_vector.planned_arm_id in selected
    )
    unselected_observation = next(
        item for item in observations if item.planned_arm_id not in selected
    )
    unselected_risk = next(
        item for item in risks
        if item.base_vector.planned_arm_id not in selected
    )
    missing_observation = replace(
        unselected_observation,
        availability=v1.AvailabilityV7.TYPED_MISSING,
        availability_reason="typed-missing-unselected-postaction",
        observation_hash="",
    )
    missing_base = replace(
        unselected_risk.base_vector,
        observation_hash=missing_observation.observation_hash,
        availability=v1.AvailabilityV7.TYPED_MISSING,
        availability_reason="typed-missing-unselected-postaction",
        feature_eligible=False,
        calibration_valid=False,
        bound_bookkeeping_valid=False,
        predicted_benefit=None,
        benefit_lower=None,
        predicted_harm=None,
        harm_upper=None,
        predicted_any_row_severe_probability=None,
        any_row_severe_probability_upper=None,
        predicted_harmed_pixel_fraction=None,
        harmed_pixel_fraction_upper=None,
        predicted_pixel_harm_cvar95=None,
        pixel_harm_cvar95_upper=None,
        risk_vector_hash="",
    )
    missing_risk = replace(
        unselected_risk, base_vector=missing_base, risk_vector_v4_hash="",
    )
    object.__setattr__(missing_observation.runtime_features[0], "value", float("nan"))
    decision = select_portfolio_v7_v4(
        manifest, proposal,
        (chosen_observation, missing_observation),
        (chosen_risk, missing_risk), calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.COMMIT


def test_v4_preserves_reviewed_v2_bytes():
    assert hashlib.sha256(
        (ROOT / "src/stablebridge/physical_repair/selector_v7_v2.py").read_bytes()
    ).hexdigest() == SELECTOR_V7_V2_REVIEWED_SHA256


@pytest.mark.parametrize(
    "fit_ids",
    (
        (),
        ("fit-a", "fit-b"),
        ("fit-a", "fit-b", "fold-cal"),
    ),
)
def test_r10_empty_wrong_cardinality_or_cal_overlap_fit_split_rejects(fit_ids):
    row = FIX["arm"](1)
    _, manifest = manifest_v2(row)
    split = manifest.planner_model_receipt.fit_receipt.split_authority
    with pytest.raises(ValueError):
        replace(split, fit_fold_ids=fit_ids, receipt_hash="")


def test_r10_canonical_schema_rejects_held_outcome_resigning():
    row = FIX["arm"](1)
    _, manifest = manifest_v2(row)
    schema = manifest.planner_model_receipt.feature_schema_receipt
    with pytest.raises(ValueError, match="producer/config"):
        replace(
            schema,
            schema_producer_code_hash="held-outcome-schema-producer",
            schema_config_hash="held-outcome-schema-config",
            schema_hash="",
        )
    with pytest.raises(ValueError, match="canonical/registry-bound"):
        replace(
            manifest.planner_model_receipt,
            feature_schema_hash="held-outcome-feature-schema",
            model_hash=manifest.planner_model_receipt.model_hash,
            receipt_hash="",
        )


def test_r10_exact_disposition_binds_v3_and_v2_review_statuses():
    disposition = v3.SelectorV7V4Disposition()
    assert disposition.predecessor_review_status == (
        "FAIL_SR1_V3_PLANNER_AUTHORITY_AND_BENEFIT_PRESERVATION_BYPASSES_NO_SEAL"
    )
    assert disposition.v2_review_status == (
        "FAIL_SR1_V2_NEW_CONTRACT_BYPASSES_NO_SEAL"
    )
    assert len(disposition.predecessor_review_sha256) == 64
    assert len(disposition.v2_review_sha256) == 64


def test_r11_score_bank_cannot_auto_generate_inference_receipts():
    row = FIX["arm"](1)
    base, manifest = manifest_v2(row)
    realization = FIX["realization"](base, row)
    plan = direct_plan(manifest, row)
    explicit = (explicit_score_receipt(manifest, row, FIX["score"](row)),)
    authority = _AUTHORITY_BY_MANIFEST[manifest.manifest_v4_hash]
    verification = make_test_only_replay_verification_v7_v4(
        manifest, authority, explicit, provenance_hash="r11-explicit-only",
    )
    with pytest.raises(ValueError, match="explicit typed inference"):
        propose_candidates_v7_v4_test_only(
            manifest, (FIX["score"](row),), (realization,), (), (plan,),
            top_k=1,
            maximum_prospective_runtime_cost=4.0,
            maximum_prospective_runtime_cost_counters=FIX["budget_counters"](),
            planner_algorithm_id=manifest.planner_model_receipt.algorithm_id,
            planner_algorithm_version=(
                manifest.planner_model_receipt.algorithm_version
            ),
            planner_config_hash=manifest.planner_model_receipt.config_hash,
            planner_provenance_hash=(
                manifest.planner_model_receipt.model_provenance_hash
            ),
            external_replay_authority=authority,
            external_replay_verification=verification,
        )


def test_r11_resigned_score_fails_without_matching_external_replay():
    row = FIX["arm"](1)
    base, manifest = manifest_v2(row)
    realization = FIX["realization"](base, row)
    plan = direct_plan(manifest, row)
    original = explicit_score_receipt(manifest, row, FIX["score"](row, 1.0))
    authority = _AUTHORITY_BY_MANIFEST[manifest.manifest_v4_hash]
    verification = make_test_only_replay_verification_v7_v4(
        manifest, authority, (original,), provenance_hash="r11-old-replay",
    )
    hostile_score = replace(
        original.base_score, planner_predicted_gain_raw_px=999.0,
    )
    hostile_execution = replace(
        original.inference_execution,
        predicted_gain_raw_px=999.0,
        output_score_bytes_hash=planner_score_output_bytes_hash_v7_v4(999.0),
        receipt_hash="",
    )
    hostile_receipt = PlannerScoreReceiptV7V4(
        hostile_score, original.before_feature_artifact, hostile_execution,
    )
    with pytest.raises(ValueError, match="exact score bank"):
        propose_candidates_v7_v4_test_only(
            manifest, (hostile_receipt,), (realization,), (), (plan,),
            top_k=1,
            maximum_prospective_runtime_cost=4.0,
            maximum_prospective_runtime_cost_counters=FIX["budget_counters"](),
            planner_algorithm_id=manifest.planner_model_receipt.algorithm_id,
            planner_algorithm_version=(
                manifest.planner_model_receipt.algorithm_version
            ),
            planner_config_hash=manifest.planner_model_receipt.config_hash,
            planner_provenance_hash=(
                manifest.planner_model_receipt.model_provenance_hash
            ),
            external_replay_authority=authority,
            external_replay_verification=verification,
        )


def test_r11_production_commit_requires_external_pass_trust_and_rejects_test_only():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    absent = _production_select_portfolio_v7_v4(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert absent.state is v1.PortfolioStateV7.NATIVE
    assert absent.reason is DecisionReasonV7V4.PLANNER_AUTHORITY_MISMATCH
    authority = _AUTHORITY_BY_MANIFEST[manifest.manifest_v4_hash]
    verification = _VERIFICATION_BY_SCORE_BANK[proposal.score_bank.score_bank_hash]
    test_only_on_production = _production_select_portfolio_v7_v4(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
        external_replay_authority=authority,
        external_replay_verification=verification,
        trusted_root_review_seal_hashes=frozenset({
            authority.root_review_seal_hash,
        }),
        trusted_verification_hashes=frozenset({
            verification.verification_hash,
        }),
    )
    assert test_only_on_production.state is v1.PortfolioStateV7.NATIVE
    assert test_only_on_production.reason is (
        DecisionReasonV7V4.PLANNER_AUTHORITY_MISMATCH
    )


def test_r12_k2_local_missing_peer_preserves_safe_commit_and_one_audit():
    first = FIX["arm"](1, action_identity="first")
    second = FIX["arm"](2, action_identity="second")
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(
        first, second, top_k=2, utilities=(10.0, 9.0),
    )
    missing_base = replace(
        risks[1].base_vector,
        availability=v1.AvailabilityV7.TYPED_MISSING,
        availability_reason="typed-missing-selected-peer",
        feature_eligible=False,
        calibration_valid=False,
        bound_bookkeeping_valid=False,
        predicted_benefit=None,
        benefit_lower=None,
        predicted_harm=None,
        harm_upper=None,
        predicted_any_row_severe_probability=None,
        any_row_severe_probability_upper=None,
        predicted_harmed_pixel_fraction=None,
        harmed_pixel_fraction_upper=None,
        predicted_pixel_harm_cvar95=None,
        pixel_harm_cvar95_upper=None,
        risk_vector_hash="",
    )
    missing = replace(
        risks[1], base_vector=missing_base, risk_vector_v4_hash="",
    )
    decision = select_portfolio_v7_v4(
        manifest, proposal, observations, (risks[0], missing), calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.COMMIT
    assert decision.selected_candidate_id == first.candidate_id
    assert decision.arm_reasons == ((
        second.planned_arm_id, DecisionReasonV7V4.FEATURE_INELIGIBLE,
    ),)
    assert sum(value for _, value in decision.arm_audit_states) == 1


def test_r12_k2_local_nonfinite_peer_preserves_safe_commit():
    first = FIX["arm"](1, action_identity="first")
    second = FIX["arm"](2, action_identity="second")
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(
        first, second, top_k=2, utilities=(10.0, 9.0),
    )
    object.__setattr__(risks[1].base_vector, "harm_upper", float("nan"))
    decision = select_portfolio_v7_v4(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.COMMIT
    assert decision.selected_candidate_id == first.candidate_id
    assert dict(decision.arm_reasons)[second.planned_arm_id] is (
        DecisionReasonV7V4.FEATURE_INELIGIBLE
    )


def test_r12_shared_calibration_drift_remains_whole_decision_native():
    first = FIX["arm"](1)
    second = FIX["arm"](2)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(first, second, top_k=2)
    object.__setattr__(calibration.base_receipt, "maximum_harm", float("nan"))
    decision = select_portfolio_v7_v4(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V4.MALFORMED_INPUT
