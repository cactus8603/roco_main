from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import runpy

import pytest

import stablebridge.physical_repair as package
from stablebridge.physical_repair import selector_v7 as v1
from stablebridge.physical_repair import selector_v7_v2 as v2
from stablebridge.physical_repair.selector_v7_v2 import (
    ActionRiskVectorV7V2,
    AliasChronologyModeV7V2,
    AliasRunPlanV7V2,
    CalibrationReceiptV7V2,
    CandidateManifestV7V2,
    CandidateObservationV7V2,
    DecisionReasonV7V2,
    ImmutableNativeReceiptV7V2,
    ObservationSourceV7V2,
    PlannerProposalV7V2,
    PlannerScoreBankV7V2,
    PreRunAliasSealV7V2,
    RuntimeCostLedgerV7V2,
    RuntimeExecutionReceiptV7V2,
    RuntimeOperationV7V2,
    SELECTOR_V7_V1_FAILED_REVIEW_SHA256,
    SELECTOR_V7_V2_SCHEMA_VERSION,
    calibration_choice_family_digest_v7_v2,
    propose_candidates_v7_v2,
    select_portfolio_v7_v2,
    selector_v7_v2_schema_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]
FIX = runpy.run_path(str(ROOT / "tests/stablebridge/test_selector_v7.py"))


def native_receipt(bank: v1.CandidateManifestV7) -> ImmutableNativeReceiptV7V2:
    native = bank.native
    return ImmutableNativeReceiptV7V2(
        source_hash=bank.source_hash,
        source_materialization_hash=native.rollback_hash,
        endpoint_id=native.endpoint_id,
        execution_unit_id=bank.execution_unit_id,
        support_hash=native.support_hash,
        support_policy_hash=bank.support_policy_hash,
        producer_hash="synthetic-native-producer",
        frozen_before_manifest=True,
    )


def manifest_v2(*arms: v1.CandidateArmV7):
    bank = FIX["manifest"](*arms)
    return bank, CandidateManifestV7V2(bank, native_receipt(bank))


def direct_plan(
    manifest: CandidateManifestV7V2,
    row: v1.CandidateArmV7,
    *,
    run_id: str | None = None,
) -> AliasRunPlanV7V2:
    return AliasRunPlanV7V2(
        manifest_v2_hash=manifest.manifest_v2_hash,
        runtime_run_id=run_id or f"run:{row.candidate_id}",
        representative_planned_arm_id=row.planned_arm_id,
        alias_planned_arm_ids=(),
        alias_receipt_bindings=(),
        mode=AliasChronologyModeV7V2.DIRECT_NO_ALIAS,
        earliest_run_start_ns=100,
        pre_run_alias_seal=None,
        runtime_policy_hash="runtime-policy",
        provenance_hash=f"run-plan:{row.candidate_id}",
    )


def proposal_v2(
    manifest: CandidateManifestV7V2,
    rows: tuple[v1.CandidateArmV7, ...],
    receipts: tuple[v1.ArmRealizationReceiptV7, ...],
    *,
    scores: tuple[v1.PlannerCandidateV7, ...] | None = None,
    aliases: tuple[v1.ArmAliasReceiptV7, ...] = (),
    run_plans: tuple[AliasRunPlanV7V2, ...] | None = None,
    top_k: int = 2,
    maximum_cost: float = 4.0,
) -> PlannerProposalV7V2:
    plans = run_plans or tuple(direct_plan(manifest, row) for row in rows)
    return propose_candidates_v7_v2(
        manifest,
        scores or tuple(FIX["score"](row) for row in rows),
        receipts, aliases, plans,
        top_k=top_k,
        maximum_prospective_runtime_cost=maximum_cost,
        maximum_prospective_runtime_cost_counters=FIX["budget_counters"](),
        planner_algorithm_id="group-weighted-ridge",
        planner_algorithm_version="v1",
        planner_config_hash="ridge-l2-1",
        planner_provenance_hash="fit-only-planner",
    )


def execution_receipt(
    manifest: CandidateManifestV7V2,
    proposal: PlannerProposalV7V2,
    run_plan: AliasRunPlanV7V2,
    row: v1.CandidateArmV7,
    realization: v1.ArmRealizationReceiptV7,
    *,
    operation: RuntimeOperationV7V2 = RuntimeOperationV7V2.EXECUTE_ARM,
    reused_from: str | None = None,
    cost: float | None = None,
    counters: v1.CostCountersV7 | None = None,
    start: int = 110,
    finish: int = 120,
) -> RuntimeExecutionReceiptV7V2:
    return RuntimeExecutionReceiptV7V2(
        manifest_v2_hash=manifest.manifest_v2_hash,
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
    manifest: CandidateManifestV7V2,
    proposal: PlannerProposalV7V2,
    run_plan: AliasRunPlanV7V2,
    execution_receipts: tuple[RuntimeExecutionReceiptV7V2, ...],
    *,
    discovered_at: int | None = None,
) -> RuntimeCostLedgerV7V2:
    counters = v1.CostCountersV7.zero(
        cache_state=execution_receipts[0].actual_prospective_runtime_cost_counters.cache_state,
    )
    for receipt in execution_receipts:
        counters = counters.plus(receipt.actual_prospective_runtime_cost_counters)
    return RuntimeCostLedgerV7V2(
        manifest_v2_hash=manifest.manifest_v2_hash,
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
    manifest: CandidateManifestV7V2,
    proposal: PlannerProposalV7V2,
    row: v1.CandidateArmV7,
    realization: v1.ArmRealizationReceiptV7,
    *,
    runtime_ledger: RuntimeCostLedgerV7V2 | None,
) -> CandidateObservationV7V2:
    return CandidateObservationV7V2(
        manifest_v2_hash=manifest.manifest_v2_hash,
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
            ObservationSourceV7V2.PROSPECTIVE_RUNTIME_OBSERVATION
            if runtime_ledger is not None
            else ObservationSourceV7V2.FROZEN_FAMILY_OBSERVATION
        ),
        runtime_cost_ledger_hash=(
            runtime_ledger.ledger_hash if runtime_ledger is not None else None
        ),
    )


def calibration_v2(
    manifest: CandidateManifestV7V2,
    proposal: PlannerProposalV7V2,
    rows: tuple[v1.CandidateArmV7, ...],
    observations: tuple[CandidateObservationV7V2, ...],
    *,
    covered: tuple[v1.CandidateArmV7, ...] | None = None,
) -> CalibrationReceiptV7V2:
    covered_rows = covered or rows
    base = FIX["calibration"](
        manifest.base_manifest, *covered_rows,
    )
    return CalibrationReceiptV7V2(
        manifest_v2_hash=manifest.manifest_v2_hash,
        base_receipt=base,
        choice_family_digest=calibration_choice_family_digest_v7_v2(
            manifest, proposal, observations,
        ),
        choice_family_cardinality=len(covered_rows),
        covered_planned_arm_ids=tuple(
            row.planned_arm_id for row in covered_rows
        ),
        assessor_model_hash=base.model_hash,
        assessor_model_provenance_hash="assessor-model-provenance",
        assessor_algorithm_id="five-head-assessor",
        assessor_algorithm_version="v1",
        assessor_feature_schema_hash=base.feature_schema_hash,
        assessor_config_hash="five-head-config",
        simultaneous_correction_id="component-max-residual-five-heads",
        frozen_before_decision=True,
    )


def risk_v2(
    manifest: CandidateManifestV7V2,
    proposal: PlannerProposalV7V2,
    observation: CandidateObservationV7V2,
    row: v1.CandidateArmV7,
    calibration: CalibrationReceiptV7V2,
    *,
    benefit: float = 0.8,
    harm: float = 0.02,
    severe: float = 0.01,
) -> ActionRiskVectorV7V2:
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
        calibration_receipt_hash=calibration.calibration_v2_hash,
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
    return ActionRiskVectorV7V2(
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
        calibration_v2_hash=calibration.calibration_v2_hash,
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


def test_v2_happy_path_commits_and_fingerprint_records_failed_predecessor():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    decision = select_portfolio_v7_v2(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.COMMIT
    assert decision.selected_candidate_id == row.candidate_id
    assert manifest.schema_version == SELECTOR_V7_V2_SCHEMA_VERSION
    assert manifest.predecessor_failed_review_sha256 == (
        SELECTOR_V7_V1_FAILED_REVIEW_SHA256
    )
    assert len(selector_v7_v2_schema_fingerprint()) == 64


def test_v2_public_exports_are_complete():
    assert all(hasattr(package, name) for name in v2.__all__)
    assert set(v2.__all__).issubset(package.__all__)


def test_r01_destructive_action0_impostor_rejects_at_v2_manifest_boundary():
    row = FIX["arm"](1)
    base = FIX["manifest"](row)
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
        CandidateManifestV7V2(forged, native_receipt(base))


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
    base = FIX["manifest"](row)
    forged_native = replace(base.native, **{field: value, "planned_arm_id": ""})
    forged = replace(
        base, candidates=(forged_native, row),
        candidate_family_hash="", manifest_hash="",
    )
    with pytest.raises(ValueError, match="canonical and immutable"):
        CandidateManifestV7V2(forged, native_receipt(base))


def test_r01_commit_boundary_revalidates_native_after_object_tampering():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    object.__setattr__(
        manifest.base_manifest.native, "mechanism_id", "destructive-restoration",
    )
    decision = select_portfolio_v7_v2(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason in {
        DecisionReasonV7V2.MALFORMED_INPUT,
        DecisionReasonV7V2.NATIVE_IDENTITY_INVALID,
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
    forged_bank = PlannerScoreBankV7V2(
        manifest_v2_hash=proposal.score_bank.manifest_v2_hash,
        candidate_family_hash=proposal.score_bank.candidate_family_hash,
        planner_allowlist_hash=proposal.score_bank.planner_allowlist_hash,
        universe_digest=proposal.score_bank.universe_digest,
        universe_cardinality=proposal.score_bank.universe_cardinality,
        scores=score_rows,
        frozen_before_observation=True,
    )
    forged = replace(
        proposal, score_bank=forged_bank, proposal_hash="",
    )
    decision = select_portfolio_v7_v2(
        manifest, forged, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V2.PROPOSAL_REPLAY_MISMATCH


def test_r02_handcrafted_partial_score_bank_fails_closed_at_commit():
    first = FIX["arm"](1)
    second = FIX["arm"](2)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(first, second, top_k=1)
    kept = proposal.score_bank.scores[:1]
    partial = PlannerScoreBankV7V2(
        manifest_v2_hash=proposal.score_bank.manifest_v2_hash,
        candidate_family_hash=proposal.score_bank.candidate_family_hash,
        planner_allowlist_hash=proposal.score_bank.planner_allowlist_hash,
        universe_digest="forged-partial-universe",
        universe_cardinality=1,
        scores=kept,
        frozen_before_observation=True,
    )
    forged = replace(
        proposal, score_bank=partial,
        deterministic_ranked_planned_arm_ids=(kept[0].planned_arm_id,),
        selected_planned_arm_ids=(kept[0].planned_arm_id,),
        proposal_hash="",
    )
    decision = select_portfolio_v7_v2(
        manifest, forged, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V2.SCORE_BANK_INCOMPLETE


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
    decision = select_portfolio_v7_v2(
        manifest, forged, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V2.PROPOSAL_REPLAY_MISMATCH


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
        risk_vector_v2_hash="",
    )
    decision = select_portfolio_v7_v2(
        manifest, proposal, observations, (forged,), calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V2.ASSESSOR_CALIBRATION_MISMATCH


def test_r03_algorithm_schema_and_config_are_exactly_calibration_bound():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    for field in (
        "assessor_algorithm_id", "assessor_algorithm_version",
        "assessor_feature_schema_hash", "assessor_config_hash",
    ):
        forged = replace(
            risks[0], **{field: f"different:{field}", "risk_vector_v2_hash": ""},
        )
        decision = select_portfolio_v7_v2(
            manifest, proposal, observations, (forged,), calibration,
            receipts, (), ledgers,
        )
        assert decision.reason is (
            DecisionReasonV7V2.ASSESSOR_CALIBRATION_MISMATCH
        )


def test_nested_base_risk_and_calibration_hash_drift_fail_closed():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    object.__setattr__(risks[0].base_vector, "harm_upper", float("nan"))
    decision = select_portfolio_v7_v2(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V2.MALFORMED_INPUT

    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    object.__setattr__(calibration.base_receipt, "maximum_harm", 1e9)
    decision = select_portfolio_v7_v2(
        manifest, proposal, observations, risks, calibration,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V2.MALFORMED_INPUT


def test_malformed_top_level_proposal_returns_typed_native_decision():
    row = FIX["arm"](1)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(row, top_k=1)
    decision = select_portfolio_v7_v2(
        manifest, None, observations, risks, calibration,  # type: ignore[arg-type]
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V2.MALFORMED_INPUT


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
    decision = select_portfolio_v7_v2(
        manifest, proposal, observations, truncated_risks, truncated,
        receipts, (), ledgers,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V2.CALIBRATION_FAMILY_INCOMPLETE


def test_r04_missing_or_extra_observation_or_risk_cannot_truncate_family():
    first = FIX["arm"](1)
    second = FIX["arm"](2)
    (base, manifest, receipts, plans, proposal, ledgers,
     observations, calibration, risks) = bundle_v2(first, second, top_k=1)
    for observed, assessed in (
        (observations[:1], risks),
        (observations, risks[:1]),
    ):
        decision = select_portfolio_v7_v2(
            manifest, proposal, observed, assessed, calibration,
            receipts, (), ledgers,
        )
        assert decision.reason is (
            DecisionReasonV7V2.CALIBRATION_FAMILY_INCOMPLETE
        )


def alias_fixture(mode: AliasChronologyModeV7V2):
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
    if mode is AliasChronologyModeV7V2.PRE_RUN_FROZEN_REUSE:
        seal = PreRunAliasSealV7V2(
            manifest_v2_hash=manifest.manifest_v2_hash,
            representative_planned_arm_id=representative.planned_arm_id,
            alias_planned_arm_ids=(alias_member.planned_arm_id,),
            alias_receipt_bindings=bindings,
            sealed_at_ns=50,
            reuse_policy_hash="exact-byte-reuse-policy",
            provenance_hash="pre-run-alias-seal",
        )
    plan = AliasRunPlanV7V2(
        manifest_v2_hash=manifest.manifest_v2_hash,
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
    decision = select_portfolio_v7_v2(
        case["manifest"], case["proposal"], (seen,), (assessed,),
        calibration, case["receipts"], (case["alias"],),
        (runtime_ledger,),
    )
    return decision, runtime_ledger


def test_r05_pre_run_frozen_reuse_charges_one_execution_and_each_verification():
    case = alias_fixture(AliasChronologyModeV7V2.PRE_RUN_FROZEN_REUSE)
    representative_execution = execution_receipt(
        case["manifest"], case["proposal"], case["plan"],
        case["representative"], case["representative_receipt"],
    )
    alias_verification = execution_receipt(
        case["manifest"], case["proposal"], case["plan"],
        case["alias_member"], case["alias_member_receipt"],
        operation=RuntimeOperationV7V2.VERIFY_ALIAS_REUSE,
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
        AliasChronologyModeV7V2.POST_HOC_DISCOVERY,
        AliasChronologyModeV7V2.SAME_RUN_DUPLICATE,
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
        discovered_at=(150 if mode is AliasChronologyModeV7V2.POST_HOC_DISCOVERY
                       else None),
    )
    assert decision.state is v1.PortfolioStateV7.COMMIT
    assert case["proposal"].reserved_prospective_runtime_cost == pytest.approx(2.0)
    assert runtime_ledger.charged_prospective_runtime_cost == pytest.approx(2.0)


def test_r05_pre_run_reuse_cannot_hide_same_run_double_execution():
    case = alias_fixture(AliasChronologyModeV7V2.PRE_RUN_FROZEN_REUSE)
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
    assert decision.reason is DecisionReasonV7V2.ALIAS_CHRONOLOGY_INVALID


def test_r05_post_hoc_discovery_cannot_claim_cheap_presealed_reuse():
    case = alias_fixture(AliasChronologyModeV7V2.POST_HOC_DISCOVERY)
    representative_execution = execution_receipt(
        case["manifest"], case["proposal"], case["plan"],
        case["representative"], case["representative_receipt"],
    )
    alias_verification = execution_receipt(
        case["manifest"], case["proposal"], case["plan"],
        case["alias_member"], case["alias_member_receipt"],
        operation=RuntimeOperationV7V2.VERIFY_ALIAS_REUSE,
        reused_from=case["representative"].planned_arm_id,
        cost=case["alias"].prospective_runtime_verification_cost,
        counters=case["verify_counters"], start=121, finish=122,
    )
    decision, _ = finish_alias_bundle(
        case, (representative_execution, alias_verification),
        discovered_at=150,
    )
    assert decision.state is v1.PortfolioStateV7.NATIVE
    assert decision.reason is DecisionReasonV7V2.ALIAS_CHRONOLOGY_INVALID


def test_r05_run_ids_timestamps_and_cost_accounts_are_structural():
    case = alias_fixture(AliasChronologyModeV7V2.PRE_RUN_FROZEN_REUSE)
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
        RuntimeExecutionReceiptV7V2(
            manifest_v2_hash=case["manifest"].manifest_v2_hash,
            proposal_hash=case["proposal"].proposal_hash,
            alias_run_plan_hash=case["plan"].plan_hash,
            runtime_run_id=case["plan"].runtime_run_id,
            planned_arm_id=case["representative"].planned_arm_id,
            realization_receipt_hash=case["representative_receipt"].receipt_hash,
            executable_bytes_hash=str(
                case["representative_receipt"].executable_bytes_hash
            ),
            operation=RuntimeOperationV7V2.EXECUTE_ARM,
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
