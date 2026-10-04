from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy

import pytest

import stablebridge.physical_repair as package
from stablebridge.physical_repair import selector_v7 as v1
from stablebridge.physical_repair import selector_v7_v5 as v5
from stablebridge.physical_repair import selector_v7_v6 as v6
from stablebridge.physical_repair import selector_v7_v7 as v7


ROOT = Path(__file__).resolve().parents[2]
V6_FIX = runpy.run_path(str(ROOT / "tests/stablebridge/test_selector_v7_v6.py"))
V5_FIX = V6_FIX["V5_FIX"]
FIX = V6_FIX["FIX"]


def _kwargs(case):
    return {
        "production_registry": case["production_registry"],
        "trusted_production_registry_hashes": case["production_trust"],
        "trusted_split_registry_hashes": case["split_trust"],
    }


def _select(case, *, audit_count=0, child_count=0):
    return v7.select_portfolio_v7_v7(
        case["proposal"],
        tuple(case["observations"]),
        tuple(case["risks"]),
        case["calibration"],
        tuple(case["realizations"]),
        (),
        tuple(case["ledgers"]),
        audit_count=audit_count,
        child_count=child_count,
        **_kwargs(case),
    )


def _one_arm_case():
    case = V6_FIX["_production_fixture"]()
    return {
        **case,
        "observations": (case["observation"],),
        "risks": (case["risk"],),
        "realizations": (case["realization"],),
        "ledgers": (case["ledger"],),
    }


def _two_arm_case():
    rows = (FIX["arm"](1), FIX["arm"](2))
    base, test_manifest = V5_FIX["manifest_v2"](*rows)
    test_authority = V5_FIX["_AUTHORITY_BY_MANIFEST"][
        test_manifest.manifest_v5_hash
    ]
    authority = replace(
        test_authority,
        integration_package_version="PRODUCTION-v1-two-arm",
        root_review_seal_hash="trusted-production-root-review-seal-two-arm",
        root_review_status="PASS",
        mode=v5.ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED,
        authority_hash="",
    )
    old_model = test_manifest.planner_model_receipt
    model_hash = v5.planner_model_identity_hash_v7_v5(
        old_model.fit_receipt,
        old_model.feature_schema_receipt,
        old_model.model_bytes_hash,
        authority.authority_hash,
    )
    model = replace(
        old_model,
        required_e235_replay_authority_hash=authority.authority_hash,
        model_hash=model_hash,
        receipt_hash="",
    )
    manifest_v5 = v5.CandidateManifestV7V5(
        base,
        test_manifest.immutable_native_receipt,
        test_manifest.freeze_receipt,
        model,
    )
    V5_FIX["_AUTHORITY_BY_MANIFEST"][manifest_v5.manifest_v5_hash] = authority
    realizations = tuple(FIX["realization"](base, row) for row in rows)
    plans = tuple(V5_FIX["direct_plan"](manifest_v5, row) for row in rows)
    scores = tuple(
        V5_FIX["explicit_score_receipt"](
            manifest_v5,
            row,
            FIX["score"](row, utility=1.0 - 0.2 * index),
        )
        for index, row in enumerate(rows)
    )
    bindings = tuple(sorted(
        (
            score.base_score.planned_arm_id,
            score.inference_execution.receipt_hash,
            score.inference_execution.external_replay_receipt_hash,
            score.inference_execution.output_score_bytes_hash,
        )
        for score in scores
    ))
    verification = v5.ExternalInferenceReplayVerificationV7V5(
        authority_hash=authority.authority_hash,
        base_manifest_hash=base.manifest_hash,
        planner_model_hash=model.model_hash,
        planner_model_bytes_hash=model.model_bytes_hash,
        planner_feature_schema_hash=model.feature_schema_hash,
        integration_allowlist_hash=authority.integration_allowlist_hash,
        root_review_seal_hash=authority.root_review_seal_hash,
        root_review_status="PASS",
        output_replay_schema_hash=authority.output_replay_schema_hash,
        execution_bindings=bindings,
        verified_before_proposal=True,
        mode=v5.ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED,
        provenance_hash="trusted-production-verification-two-arm",
    )
    entry = v6.ProductionReplayAuthorityEntryV7V6(authority, verification)
    production_registry = v6.TrustedProductionAuthorityRegistryV7V6(
        registry_id="trusted-production-registry-two-arm",
        registry_version="v1",
        entries=(entry,),
        frozen_before_production=True,
    )
    production_trust = frozenset({production_registry.registry_hash})
    split_registry, split_trust = V6_FIX["_split_registry"](
        manifest_v5, suffix="two-arm",
    )
    manifest = v6.bind_candidate_manifest_v7_v6(
        manifest_v5,
        split_registry,
        trusted_split_registry_hashes=split_trust,
    )
    proposal = v6.propose_candidates_v7_v6(
        manifest,
        scores,
        realizations,
        (),
        plans,
        production_registry=production_registry,
        trusted_production_registry_hashes=production_trust,
        trusted_split_registry_hashes=split_trust,
        top_k=2,
        maximum_prospective_runtime_cost=4.0,
        maximum_prospective_runtime_cost_counters=FIX["budget_counters"](),
        planner_algorithm_id=model.algorithm_id,
        planner_algorithm_version=model.algorithm_version,
        planner_config_hash=model.config_hash,
        planner_provenance_hash=model.model_provenance_hash,
    )
    executions = tuple(
        V5_FIX["execution_receipt"](
            manifest_v5, proposal.base_proposal, plan, row, realization,
        )
        for row, realization, plan in zip(rows, realizations, plans)
    )
    ledgers = tuple(
        V5_FIX["ledger"](
            manifest_v5, proposal.base_proposal, plan, (execution,),
        )
        for plan, execution in zip(plans, executions)
    )
    observations = tuple(
        V5_FIX["observation"](
            manifest_v5,
            proposal.base_proposal,
            row,
            realization,
            runtime_ledger=ledger,
        )
        for row, realization, ledger in zip(rows, realizations, ledgers)
    )
    calibration = V5_FIX["calibration_v2"](
        manifest_v5, proposal.base_proposal, rows, observations,
    )
    risks = tuple(
        V5_FIX["risk_v2"](
            manifest_v5,
            proposal.base_proposal,
            observation,
            row,
            calibration,
            benefit=0.9 - 0.1 * index,
        )
        for index, (row, observation) in enumerate(zip(rows, observations))
    )
    return {
        "rows": rows,
        "manifest": manifest,
        "proposal": proposal,
        "observations": observations,
        "risks": risks,
        "calibration": calibration,
        "realizations": realizations,
        "ledgers": ledgers,
        "production_registry": production_registry,
        "production_trust": production_trust,
        "split_registry": split_registry,
        "split_trust": split_trust,
    }


def _unsafe_replace(instance, **changes):
    forged = object.__new__(type(instance))
    for name in instance.__dataclass_fields__:
        object.__setattr__(
            forged, name, changes.get(name, getattr(instance, name)),
        )
    return forged


def _forged_v6_decision(genuine, **changes):
    forged_base = replace(
        genuine.base_decision.base_decision,
        decision_hash="",
        **changes,
    )
    return v6.PortfolioDecisionV7V6(
        forged_base, genuine.base_decision.proposal,
    )


def test_v7_binds_exact_v6_failed_review_and_preserves_predecessor_bytes():
    assert hashlib.sha256(
        (ROOT / "src/stablebridge/physical_repair/selector_v7_v6.py").read_bytes()
    ).hexdigest() == v7.SELECTOR_V7_V6_REVIEWED_SHA256
    assert hashlib.sha256(
        (ROOT / "tests/stablebridge/test_selector_v7_v6.py").read_bytes()
    ).hexdigest() == v7.SELECTOR_V7_V6_REVIEWED_TEST_SHA256
    assert hashlib.sha256(
        (
            ROOT
            / "research/selector_v7_redesign_20261003/"
            "sr1_root_review_v6/REVIEW.json"
        ).read_bytes()
    ).hexdigest() == v7.SR1_ROOT_REVIEW_V6_SHA256
    disposition = v7.SelectorV7V7Disposition()
    assert disposition.predecessor_review_status == v7.SR1_ROOT_REVIEW_V6_STATUS
    assert disposition.scope == "S5_ONLY"
    assert "DEFERRED" in disposition.audit_status
    assert len(v7.selector_v7_v7_schema_fingerprint()) == 64


def test_r01_r14_v6_contract_is_inherited_without_mutating_v6():
    assert v7.CandidateManifestV7V7 is v6.CandidateManifestV7V6
    assert v7.PlannerProposalV7V7 is v6.PlannerProposalV7V6
    assert v7.ActionRiskVectorV7V7 is v6.ActionRiskVectorV7V6
    assert v7.CalibrationReceiptV7V7 is v6.CalibrationReceiptV7V6


def test_r15_positive_path_replays_and_serializes_complete_decision():
    case = _one_arm_case()
    decision = _select(case)
    base = decision.base_decision.base_decision
    assert base.state is v5.DecisionStateV7V5.COMMIT
    assert decision.replay_receipt.replay_performed is True
    assert v7.verify_production_decision_v7_v7(decision, **_kwargs(case))
    payload = json.loads(v7.serialize_production_decision_v7_v7(
        decision, **_kwargs(case),
    ))
    assert payload["selected_candidate_id"] == base.selected_candidate_id
    assert payload["safe_candidate_ids"] == list(base.safe_candidate_ids)
    assert payload["arm_reasons"] == [
        [key, reason.value] for key, reason in base.arm_reasons
    ]
    assert payload["replay_bundle_hash"] == decision.replay_bundle.bundle_hash
    assert payload["scope"] == "S5_ONLY"


def test_r15_injected_candidate_cannot_construct_receipt_or_decision():
    case = _one_arm_case()
    genuine = _select(case)
    forged = _forged_v6_decision(
        genuine,
        selected_candidate_id="attacker-injected-candidate",
        safe_candidate_ids=("attacker-injected-candidate",),
        arm_reasons=(),
        arm_audit_states=(),
    )
    with pytest.raises(ValueError, match="deterministic selector replay"):
        v7.DecisionReplayReceiptV7V7(genuine.replay_bundle, forged)
    with pytest.raises(ValueError):
        v7.PortfolioDecisionV7V7(
            forged, genuine.replay_bundle, genuine.replay_receipt,
        )


def test_r15_wrong_in_bank_candidate_is_rejected_by_exact_replay():
    case = _two_arm_case()
    genuine = _select(case)
    score_ids = {
        row.candidate_id
        for row in case["proposal"].base_proposal.score_bank.scores
    }
    selected = genuine.base_decision.base_decision.selected_candidate_id
    wrong = next(candidate for candidate in score_ids if candidate != selected)
    forged = _forged_v6_decision(
        genuine,
        selected_candidate_id=wrong,
        safe_candidate_ids=(wrong,),
    )
    assert wrong in score_ids
    with pytest.raises(ValueError, match="deterministic selector replay"):
        v7.DecisionReplayReceiptV7V7(genuine.replay_bundle, forged)


@pytest.mark.parametrize(
    "changes",
    [
        {"safe_candidate_ids": ()},
        {"arm_reasons": (("invented-arm", v5.DecisionReasonV7V5.NO_SAFE_CANDIDATE),)},
        {"reason": v5.DecisionReasonV7V5.NO_SAFE_CANDIDATE},
    ],
)
def test_r15_wrong_safe_set_or_reasons_cannot_be_resigned(changes):
    case = _one_arm_case()
    genuine = _select(case)
    forged = _forged_v6_decision(genuine, **changes)
    tampered = _unsafe_replace(genuine, base_decision=forged)
    assert not v7.verify_production_decision_v7_v7(tampered, **_kwargs(case))
    with pytest.raises(ValueError, match="exact trusted S5 selector replay"):
        v7.serialize_production_decision_v7_v7(tampered, **_kwargs(case))


def test_r15_missing_replay_input_changes_output_and_cannot_bind_genuine_decision():
    case = _one_arm_case()
    genuine = _select(case)
    incomplete = v7.make_decision_replay_bundle_v7_v7(
        case["proposal"],
        (),
        case["risks"],
        case["calibration"],
        case["realizations"],
        (),
        case["ledgers"],
        production_registry=case["production_registry"],
    )
    with pytest.raises(ValueError, match="deterministic selector replay"):
        v7.DecisionReplayReceiptV7V7(incomplete, genuine.base_decision)


def test_r15_tampered_bundle_hash_and_wrong_external_registry_fail_closed():
    case = _one_arm_case()
    genuine = _select(case)
    with pytest.raises(ValueError, match="bundle hash"):
        replace(genuine.replay_bundle, bundle_hash="0" * 64)
    assert not v7.verify_production_decision_v7_v7(
        genuine,
        production_registry=case["production_registry"],
        trusted_production_registry_hashes=frozenset(),
        trusted_split_registry_hashes=case["split_trust"],
    )


@pytest.mark.parametrize("audit_count", [0, 1])
def test_r16_audit_is_rejected_at_construction_verification_and_serialization(
    audit_count,
):
    case = _one_arm_case()
    genuine = _select(case, audit_count=audit_count)
    forged = _forged_v6_decision(
        genuine,
        state=v5.DecisionStateV7V5.AUDIT,
        selected_candidate_id=None,
        safe_candidate_ids=(),
        arm_reasons=(),
        arm_audit_states=(),
        reason=v5.DecisionReasonV7V5.NO_SAFE_CANDIDATE,
        audit_count=audit_count,
    )
    with pytest.raises(ValueError, match="AUDIT is categorically rejected"):
        v7.PortfolioDecisionV7V7(
            forged, genuine.replay_bundle, genuine.replay_receipt,
        )
    tampered = _unsafe_replace(genuine, base_decision=forged)
    assert not v7.verify_production_decision_v7_v7(tampered, **_kwargs(case))
    with pytest.raises(ValueError, match="exact trusted S5 selector replay"):
        v7.serialize_production_decision_v7_v7(tampered, **_kwargs(case))


def test_v7_public_exports_are_complete_and_no_test_only_decision_entrypoint():
    assert set(v7.__all__).issubset(package.__all__)
    assert all(hasattr(package, name) for name in v7.__all__)
    assert "select_portfolio_v7_v7_test_only" not in v7.__all__
    assert "DecisionStateV7V7" in v7.__all__
