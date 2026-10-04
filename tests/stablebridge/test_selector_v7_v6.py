from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import runpy

import pytest

import stablebridge.physical_repair as package
from stablebridge.physical_repair import selector_v7 as v1
from stablebridge.physical_repair import selector_v7_v5 as v5
from stablebridge.physical_repair import selector_v7_v6 as v6


ROOT = Path(__file__).resolve().parents[2]
V5_FIX = runpy.run_path(str(ROOT / "tests/stablebridge/test_selector_v7_v5.py"))
REVIEW_FIX = runpy.run_path(str(
    ROOT / "research/selector_v7_redesign_20261003/sr1_root_review_v5/"
    "tests/test_adversarial.py"
))
FIX = V5_FIX["FIX"]


def _split_registry(manifest: v5.CandidateManifestV7V5, *, suffix="main"):
    root = manifest.planner_model_receipt.fit_receipt.split_authority.authority_root_receipt
    authority = v6.SR0ASplitRootAuthorityV7V6(
        sr0a_manifest_receipt_hash=root.sr0a_manifest_receipt_hash,
        root_receipt=root,
        authority_provenance_hash=f"trusted-sr0a-root:{suffix}",
        frozen_before_sr1=True,
    )
    registry = v6.TrustedSR0ASplitRegistryV7V6(
        registry_id=f"trusted-sr0a-split-registry:{suffix}",
        registry_version="v1",
        authorities=(authority,),
        frozen_before_sr1=True,
    )
    return registry, frozenset({registry.registry_hash})


def _production_fixture():
    row = FIX["arm"](1)
    base, test_manifest = V5_FIX["manifest_v2"](row)
    test_authority = V5_FIX["_AUTHORITY_BY_MANIFEST"][
        test_manifest.manifest_v5_hash
    ]
    production_authority = replace(
        test_authority,
        integration_package_version="PRODUCTION-v1",
        root_review_seal_hash="trusted-production-root-review-seal",
        root_review_status="PASS",
        mode=v5.ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED,
        authority_hash="",
    )
    old_model = test_manifest.planner_model_receipt
    model_hash = v5.planner_model_identity_hash_v7_v5(
        old_model.fit_receipt,
        old_model.feature_schema_receipt,
        old_model.model_bytes_hash,
        production_authority.authority_hash,
    )
    production_model = replace(
        old_model,
        required_e235_replay_authority_hash=production_authority.authority_hash,
        model_hash=model_hash,
        receipt_hash="",
    )
    production_manifest_v5 = v5.CandidateManifestV7V5(
        base,
        test_manifest.immutable_native_receipt,
        test_manifest.freeze_receipt,
        production_model,
    )
    V5_FIX["_AUTHORITY_BY_MANIFEST"][
        production_manifest_v5.manifest_v5_hash
    ] = production_authority
    realization = FIX["realization"](base, row)
    plan = V5_FIX["direct_plan"](production_manifest_v5, row)
    score_receipt = V5_FIX["explicit_score_receipt"](
        production_manifest_v5, row, FIX["score"](row, utility=1.0),
    )
    execution = score_receipt.inference_execution
    production_verification = v5.ExternalInferenceReplayVerificationV7V5(
        authority_hash=production_authority.authority_hash,
        base_manifest_hash=base.manifest_hash,
        planner_model_hash=production_model.model_hash,
        planner_model_bytes_hash=production_model.model_bytes_hash,
        planner_feature_schema_hash=production_model.feature_schema_hash,
        integration_allowlist_hash=(
            production_authority.integration_allowlist_hash
        ),
        root_review_seal_hash=production_authority.root_review_seal_hash,
        root_review_status="PASS",
        output_replay_schema_hash=production_authority.output_replay_schema_hash,
        execution_bindings=((
            row.planned_arm_id,
            execution.receipt_hash,
            execution.external_replay_receipt_hash,
            execution.output_score_bytes_hash,
        ),),
        verified_before_proposal=True,
        mode=v5.ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED,
        provenance_hash="trusted-production-verification",
    )
    entry = v6.ProductionReplayAuthorityEntryV7V6(
        production_authority, production_verification,
    )
    production_registry = v6.TrustedProductionAuthorityRegistryV7V6(
        registry_id="trusted-production-registry",
        registry_version="v1",
        entries=(entry,),
        frozen_before_production=True,
    )
    production_trust = frozenset({production_registry.registry_hash})
    split_registry, split_trust = _split_registry(production_manifest_v5)
    manifest = v6.bind_candidate_manifest_v7_v6(
        production_manifest_v5,
        split_registry,
        trusted_split_registry_hashes=split_trust,
    )
    proposal = v6.propose_candidates_v7_v6(
        manifest,
        (score_receipt,),
        (realization,),
        (),
        (plan,),
        production_registry=production_registry,
        trusted_production_registry_hashes=production_trust,
        trusted_split_registry_hashes=split_trust,
        top_k=1,
        maximum_prospective_runtime_cost=4.0,
        maximum_prospective_runtime_cost_counters=FIX["budget_counters"](),
        planner_algorithm_id=production_model.algorithm_id,
        planner_algorithm_version=production_model.algorithm_version,
        planner_config_hash=production_model.config_hash,
        planner_provenance_hash=production_model.model_provenance_hash,
    )
    execution_receipt = V5_FIX["execution_receipt"](
        production_manifest_v5,
        proposal.base_proposal,
        plan,
        row,
        realization,
    )
    ledger = V5_FIX["ledger"](
        production_manifest_v5,
        proposal.base_proposal,
        plan,
        (execution_receipt,),
    )
    observation = V5_FIX["observation"](
        production_manifest_v5,
        proposal.base_proposal,
        row,
        realization,
        runtime_ledger=ledger,
    )
    calibration = V5_FIX["calibration_v2"](
        production_manifest_v5,
        proposal.base_proposal,
        (row,),
        (observation,),
    )
    risk = V5_FIX["risk_v2"](
        production_manifest_v5,
        proposal.base_proposal,
        observation,
        row,
        calibration,
    )
    return {
        "row": row,
        "manifest": manifest,
        "proposal": proposal,
        "realization": realization,
        "ledger": ledger,
        "observation": observation,
        "calibration": calibration,
        "risk": risk,
        "production_registry": production_registry,
        "production_trust": production_trust,
        "split_registry": split_registry,
        "split_trust": split_trust,
    }


def test_v6_binds_exact_v5_fail_review_and_preserves_v5_bytes():
    assert hashlib.sha256(
        (ROOT / "research/selector_v7_redesign_20261003/sr1_root_review_v5/REVIEW.json").read_bytes()
    ).hexdigest() == v6.SR1_ROOT_REVIEW_V5_SHA256
    assert hashlib.sha256(
        (ROOT / "src/stablebridge/physical_repair/selector_v7_v5.py").read_bytes()
    ).hexdigest() == v6.SELECTOR_V7_V5_REVIEWED_SHA256
    assert hashlib.sha256(
        (ROOT / "tests/stablebridge/test_selector_v7_v5.py").read_bytes()
    ).hexdigest() == v6.SELECTOR_V7_V5_REVIEWED_TEST_SHA256
    disposition = v6.SelectorV7V6Disposition()
    assert disposition.predecessor_review_status == v6.SR1_ROOT_REVIEW_V5_STATUS
    assert len(v6.selector_v7_v6_schema_fingerprint()) == 64
    assert set(v6.__all__).issubset(package.__all__)
    assert all(hasattr(package, name) for name in v6.__all__)


def test_r01_r12_v5_semantics_are_inherited_without_duplication():
    assert v6.ActionRiskVectorV7V6 is v5.ActionRiskVectorV7V5
    assert v6.CalibrationReceiptV7V6 is v5.CalibrationReceiptV7V5
    assert v6.CandidateObservationV7V6 is v5.CandidateObservationV7V5
    assert v6.RuntimeCostLedgerV7V6 is v5.RuntimeCostLedgerV7V5
    row = FIX["arm"](1)
    bundle = V5_FIX["bundle_v2"](row, top_k=1)
    decision = V5_FIX["select_portfolio_v7_v5"](
        bundle[1], bundle[4], bundle[-3], bundle[-1], bundle[-2],
        bundle[2], (), bundle[5],
    )
    assert decision.state is v5.DecisionStateV7V5.TEST_ONLY_WOULD_COMMIT


def test_r13_typed_production_registry_positive_proposal_and_decision_path():
    case = _production_fixture()
    assert v6.verify_production_proposal_v7_v6(
        case["proposal"],
        production_registry=case["production_registry"],
        trusted_production_registry_hashes=case["production_trust"],
        trusted_split_registry_hashes=case["split_trust"],
    )
    assert v6.serialize_production_proposal_v7_v6(
        case["proposal"],
        production_registry=case["production_registry"],
        trusted_production_registry_hashes=case["production_trust"],
        trusted_split_registry_hashes=case["split_trust"],
    )
    decision = v6.select_portfolio_v7_v6(
        case["proposal"],
        (case["observation"],),
        (case["risk"],),
        case["calibration"],
        (case["realization"],),
        (),
        (case["ledger"],),
        production_registry=case["production_registry"],
        trusted_production_registry_hashes=case["production_trust"],
        trusted_split_registry_hashes=case["split_trust"],
    )
    assert decision.base_decision.state is v5.DecisionStateV7V5.COMMIT
    assert v6.verify_production_decision_v7_v6(
        decision,
        production_registry=case["production_registry"],
        trusted_production_registry_hashes=case["production_trust"],
        trusted_split_registry_hashes=case["split_trust"],
    )
    assert v6.serialize_production_decision_v7_v6(
        decision,
        production_registry=case["production_registry"],
        trusted_production_registry_hashes=case["production_trust"],
        trusted_split_registry_hashes=case["split_trust"],
    )


def test_r13_registry_constructor_categorically_rejects_test_only_preimages():
    row = FIX["arm"](1)
    bundle = V5_FIX["bundle_v2"](row, top_k=1)
    authority = V5_FIX["_AUTHORITY_BY_MANIFEST"][bundle[1].manifest_v5_hash]
    verification = V5_FIX["_VERIFICATION_BY_SCORE_BANK"][
        bundle[4].score_bank.score_bank_hash
    ]
    with pytest.raises(ValueError, match="categorically rejects TEST_ONLY"):
        v6.ProductionReplayAuthorityEntryV7V6(authority, verification)


def test_r13_coordinated_retag_cannot_pass_typed_v6_consumers():
    production = _production_fixture()
    row = FIX["arm"](1)
    bundle = V5_FIX["bundle_v2"](row, top_k=1)
    split_registry, split_trust = _split_registry(bundle[1], suffix="test-only")
    manifest = v6.bind_candidate_manifest_v7_v6(
        bundle[1], split_registry,
        trusted_split_registry_hashes=split_trust,
    )
    proposal = v6.PlannerProposalV7V6(
        bundle[4], manifest, v6.TEST_ONLY_PRODUCTION_REGISTRY_MARKER,
    )
    test_decision = V5_FIX["select_portfolio_v7_v5"](
        bundle[1], bundle[4], bundle[-3], bundle[-1], bundle[-2],
        bundle[2], (), bundle[5],
    )
    forged_base = replace(
        proposal.base_proposal,
        external_authority_mode=v5.ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED,
        production_eligibility=v5.ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE,
        proposal_hash="",
    )
    forged = v6.PlannerProposalV7V6(
        forged_base,
        manifest,
        production["production_registry"].registry_hash,
    )
    assert not v6.verify_production_proposal_v7_v6(
        forged,
        production_registry=production["production_registry"],
        trusted_production_registry_hashes=production["production_trust"],
        trusted_split_registry_hashes=split_trust,
    )
    with pytest.raises(ValueError, match="typed trusted production"):
        v6.serialize_production_proposal_v7_v6(
            forged,
            production_registry=production["production_registry"],
            trusted_production_registry_hashes=production["production_trust"],
            trusted_split_registry_hashes=split_trust,
        )
    forged_base_with_production_hashes = replace(
        forged_base,
        external_authority_hash=(
            production["production_registry"].entries[0].authority.authority_hash
        ),
        external_verification_hash=(
            production["production_registry"].entries[0].verification.verification_hash
        ),
        proposal_hash="",
    )
    forged_with_prod_hashes = v6.PlannerProposalV7V6(
        forged_base_with_production_hashes,
        manifest,
        production["production_registry"].registry_hash,
    )
    assert not v6.verify_production_proposal_v7_v6(
        forged_with_prod_hashes,
        production_registry=production["production_registry"],
        trusted_production_registry_hashes=production["production_trust"],
        trusted_split_registry_hashes=split_trust,
    )
    forged_decision_base = replace(
        test_decision,
        state=v5.DecisionStateV7V5.COMMIT,
        proposal_hash=forged_base.proposal_hash,
        external_authority_mode=v5.ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED,
        production_eligibility=v5.ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE,
        decision_hash="",
    )
    forged_decision = v6.PortfolioDecisionV7V6(
        forged_decision_base, forged,
    )
    assert not v6.verify_production_decision_v7_v6(
        forged_decision,
        production_registry=production["production_registry"],
        trusted_production_registry_hashes=production["production_trust"],
        trusted_split_registry_hashes=split_trust,
    )


def test_r14_original_externally_trusted_split_registry_positive_path():
    row = FIX["arm"](1)
    _, manifest_v5 = V5_FIX["manifest_v2"](row)
    registry, trusted = _split_registry(manifest_v5)
    manifest = v6.bind_candidate_manifest_v7_v6(
        manifest_v5, registry, trusted_split_registry_hashes=trusted,
    )
    assert v6.verify_manifest_split_authority_v7_v6(
        manifest, trusted_split_registry_hashes=trusted,
    )
    split = manifest_v5.planner_model_receipt.fit_receipt.split_authority
    assert manifest.split_authority_hash == registry.authorities[0].authority_hash
    assert split.authority_root_receipt == registry.authorities[0].root_receipt


@pytest.mark.parametrize("alien", [False, True])
def test_r14_coordinated_assignment_or_universe_reanchor_lacks_external_trust(alien):
    row = FIX["arm"](1)
    _, original = V5_FIX["manifest_v2"](row)
    original_registry, original_trust = _split_registry(original, suffix="original")
    universe = (
        ("alien-cal", "alien-eval", "alien-fit-1", "alien-fit-2", "alien-fit-3")
        if alien else None
    )
    hostile, _, _ = REVIEW_FIX["_coordinated_split_reanchor"](
        original, universe=universe, rotation=2 if not alien else 1,
    )
    with pytest.raises(ValueError, match="absent from typed SR0A registry"):
        v6.bind_candidate_manifest_v7_v6(
            hostile,
            original_registry,
            trusted_split_registry_hashes=original_trust,
        )
    hostile_registry, _ = _split_registry(hostile, suffix="hostile")
    with pytest.raises(ValueError, match="lacks external trust"):
        v6.bind_candidate_manifest_v7_v6(
            hostile,
            hostile_registry,
            trusted_split_registry_hashes=original_trust,
        )
    hostile_manifest = v6.CandidateManifestV7V6(
        hostile,
        hostile_registry,
        hostile_registry.authorities[0].authority_hash,
    )
    assert not v6.verify_manifest_split_authority_v7_v6(
        hostile_manifest,
        trusted_split_registry_hashes=original_trust,
    )


def test_test_only_helpers_are_not_public_v6_exports():
    assert "propose_candidates_v7_v6_test_only" not in v6.__all__
    assert "select_portfolio_v7_v6_test_only" not in v6.__all__
