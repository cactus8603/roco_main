"""Append-only SelectorV7 SR1-v7 deterministic-decision successor.

V7 preserves the v6 R01--R14 authority chain and closes the two material
v6 review findings at the S5 production-decision boundary:

* every accepted decision carries a complete, typed, frozen replay bundle;
  construction, verification, and serialization invoke the deterministic v6
  selector and require equality of the entire decision output;
* ``AUDIT`` is outside this S5-only contract and is rejected at every v7
  decision boundary.  S6 remains deferred until a typed audit eligibility,
  boundary, independent-evidence, VOI-lower-bound, and cost receipt exists.

This module grants no authority seal.  Its bytes are a candidate for a fresh
independent root review.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

from . import selector_v7 as v1
from . import selector_v7_v5 as v5
from . import selector_v7_v6 as v6


SELECTOR_V7_V7_SCHEMA_VERSION = "selector-v7-contract/v4.39-sr1-v7-candidate"
SELECTOR_V7_V7_CANONICALIZATION_VERSION = "selector-v7-v7-canonical-json/v1"
SELECTOR_V7_V6_REVIEWED_SHA256 = (
    "8bca6471c1889f36be698cd80c47c614aeafdc5f4a69e2506c75e28bc75c503b"
)
SELECTOR_V7_V6_REVIEWED_TEST_SHA256 = (
    "3eb2b6b3e4c6b6aa6b1e99145e2b716e4a9199e886a65a0c298f0eabea3fbd9b"
)
SELECTOR_V7_V6_REVIEWED_PACKAGE_SHA256 = (
    "4897a41dbdcfc7e7f21529f9730dd351bfea3471af713f84b1251f1756730a83"
)
SELECTOR_V7_V6_SCHEMA_FINGERPRINT = (
    "daa818c6d84a74ca6c71e0f52e57e511b1859a7232404214c96e6af8b038866e"
)
SR1_ROOT_REVIEW_V6_SHA256 = (
    "a3a969e4a8415be55bd79d849462b6b9c6de24ed7c1d7f012eaf631f07250fd1"
)
SR1_ROOT_REVIEW_V6_STATUS = (
    "FAIL_SR1_V6_UNREPLAYED_DECISION_AND_UNDERCONSTRAINED_AUDIT_BYPASSES_NO_SEAL"
)
SELECTOR_V7_V7_SCOPE = "S5_ONLY"
S6_AUDIT_STATUS = "DEFERRED_UNTIL_TYPED_AUDIT_ELIGIBILITY_VOI_AND_COST_RECEIPT"


def _sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sealed(provided: str, payload: object, name: str) -> str:
    expected = _sha256(payload)
    if provided and provided != expected:
        raise ValueError(f"{name} does not match canonical content")
    return expected


def _rehash_valid(value: object) -> bool:
    return bool(v5._rehash_valid(value))


def _bounded_counter(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise ValueError(f"{name} must be the integer 0 or 1")
    return value


def _require_frozen_tuple(
    rows: object,
    expected_type: type,
    name: str,
) -> tuple[object, ...]:
    if not isinstance(rows, tuple):
        raise ValueError(f"{name} must be an immutable tuple")
    if any(not isinstance(row, expected_type) or not _rehash_valid(row) for row in rows):
        raise ValueError(f"{name} contains an invalid typed receipt")
    return rows


def _decision_payload(decision: v6.PortfolioDecisionV7V6) -> dict[str, object]:
    """Canonical full S5 output, including every safety/audit explanation."""
    base = decision.base_decision
    return {
        "decision_v6_hash": decision.decision_v6_hash,
        "base_decision_hash": base.decision_hash,
        "proposal_v6_hash": decision.proposal.proposal_v6_hash,
        "state": base.state.value,
        "manifest_v5_hash": base.manifest_v5_hash,
        "proposal_hash": base.proposal_hash,
        "calibration_v5_hash": base.calibration_v5_hash,
        "native_candidate_id": base.native_candidate_id,
        "selected_candidate_id": base.selected_candidate_id,
        "safe_candidate_ids": list(base.safe_candidate_ids),
        "arm_reasons": [[key, reason.value] for key, reason in base.arm_reasons],
        "arm_audit_states": [list(row) for row in base.arm_audit_states],
        "reason": base.reason.value,
        "audit_count": base.audit_count,
        "child_count": base.child_count,
        "external_authority_mode": base.external_authority_mode.value,
        "external_authority_hash": base.external_authority_hash,
        "external_verification_hash": base.external_verification_hash,
        "production_eligibility": base.production_eligibility.value,
    }


def _require_s5_decision(decision: v6.PortfolioDecisionV7V6) -> None:
    state = decision.base_decision.state
    if state is v5.DecisionStateV7V5.AUDIT:
        raise ValueError(
            "AUDIT is categorically rejected by the S5-only v7 contract; S6 is deferred"
        )
    if state not in {v5.DecisionStateV7V5.COMMIT, v5.DecisionStateV7V5.NATIVE}:
        raise ValueError("v7 production decision must be S5 COMMIT or NATIVE")


@dataclass(frozen=True)
class SelectorV7V7Disposition:
    predecessor_review_status: str = SR1_ROOT_REVIEW_V6_STATUS
    predecessor_review_sha256: str = SR1_ROOT_REVIEW_V6_SHA256
    predecessor_source_sha256: str = SELECTOR_V7_V6_REVIEWED_SHA256
    predecessor_test_sha256: str = SELECTOR_V7_V6_REVIEWED_TEST_SHA256
    predecessor_package_sha256: str = SELECTOR_V7_V6_REVIEWED_PACKAGE_SHA256
    scope: str = SELECTOR_V7_V7_SCOPE
    audit_status: str = S6_AUDIT_STATUS
    successor_status: str = "CANDIDATE_PENDING_INDEPENDENT_ROOT_REVIEW"

    def __post_init__(self) -> None:
        if (
            self.predecessor_review_status != SR1_ROOT_REVIEW_V6_STATUS
            or self.predecessor_review_sha256 != SR1_ROOT_REVIEW_V6_SHA256
            or self.predecessor_source_sha256 != SELECTOR_V7_V6_REVIEWED_SHA256
            or self.predecessor_test_sha256 != SELECTOR_V7_V6_REVIEWED_TEST_SHA256
            or self.predecessor_package_sha256
            != SELECTOR_V7_V6_REVIEWED_PACKAGE_SHA256
            or self.scope != SELECTOR_V7_V7_SCOPE
            or self.audit_status != S6_AUDIT_STATUS
            or self.successor_status
            != "CANDIDATE_PENDING_INDEPENDENT_ROOT_REVIEW"
        ):
            raise ValueError("SelectorV7-v7 predecessor disposition drifted")


@dataclass(frozen=True)
class DecisionReplayBundleV7V7:
    """Complete immutable preimage for one deterministic S5 decision replay."""

    proposal: v6.PlannerProposalV7V6
    observations: tuple[v5.CandidateObservationV7V5, ...]
    risks: tuple[v5.ActionRiskVectorV7V5, ...]
    calibration: v5.CalibrationReceiptV7V5
    realizations: tuple[v1.ArmRealizationReceiptV7, ...]
    aliases: tuple[v1.ArmAliasReceiptV7, ...]
    runtime_cost_ledgers: tuple[v5.RuntimeCostLedgerV7V5, ...]
    audit_count: int
    child_count: int
    production_registry: v6.TrustedProductionAuthorityRegistryV7V6
    selector_implementation_sha256: str = SELECTOR_V7_V6_REVIEWED_SHA256
    selector_schema_fingerprint: str = SELECTOR_V7_V6_SCHEMA_FINGERPRINT
    scope: str = SELECTOR_V7_V7_SCOPE
    frozen_before_decision: bool = True
    bundle_hash: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.proposal, v6.PlannerProposalV7V6)
            or not _rehash_valid(self.proposal)
            or not isinstance(
                self.production_registry,
                v6.TrustedProductionAuthorityRegistryV7V6,
            )
            or not _rehash_valid(self.production_registry)
        ):
            raise ValueError("decision replay bundle needs typed proposal/authority")
        _require_frozen_tuple(
            self.observations, v5.CandidateObservationV7V5, "observations",
        )
        _require_frozen_tuple(self.risks, v5.ActionRiskVectorV7V5, "risk vectors")
        _require_frozen_tuple(
            self.realizations, v1.ArmRealizationReceiptV7, "realizations",
        )
        _require_frozen_tuple(self.aliases, v1.ArmAliasReceiptV7, "aliases")
        _require_frozen_tuple(
            self.runtime_cost_ledgers,
            v5.RuntimeCostLedgerV7V5,
            "runtime cost ledgers",
        )
        if (
            not isinstance(self.calibration, v5.CalibrationReceiptV7V5)
            or not _rehash_valid(self.calibration)
        ):
            raise ValueError("decision replay calibration is invalid")
        _bounded_counter(self.audit_count, "audit_count")
        _bounded_counter(self.child_count, "child_count")
        if self.selector_implementation_sha256 != SELECTOR_V7_V6_REVIEWED_SHA256:
            raise ValueError("decision replay selector implementation drifted")
        if (
            self.selector_schema_fingerprint != SELECTOR_V7_V6_SCHEMA_FINGERPRINT
            or v6.selector_v7_v6_schema_fingerprint()
            != SELECTOR_V7_V6_SCHEMA_FINGERPRINT
        ):
            raise ValueError("decision replay selector schema drifted")
        if self.scope != SELECTOR_V7_V7_SCOPE or self.frozen_before_decision is not True:
            raise ValueError("decision replay bundle must be frozen for S5")
        if (
            self.proposal.production_registry_hash
            != self.production_registry.registry_hash
        ):
            raise ValueError("decision replay registry differs from proposal authority")
        if not v6.verify_production_proposal_v7_v6(
            self.proposal,
            production_registry=self.production_registry,
            trusted_production_registry_hashes=frozenset({
                self.production_registry.registry_hash,
            }),
            trusted_split_registry_hashes=frozenset({
                self.proposal.manifest.split_registry.registry_hash,
            }),
        ):
            raise ValueError("decision replay proposal does not reproduce its authority")
        payload = {
            "schema_version": SELECTOR_V7_V7_SCHEMA_VERSION,
            "proposal_v6_hash": self.proposal.proposal_v6_hash,
            "manifest_v6_hash": self.proposal.manifest.manifest_v6_hash,
            "observation_hashes": [row.observation_hash for row in self.observations],
            "risk_vector_hashes": [row.risk_vector_v5_hash for row in self.risks],
            "calibration_v5_hash": self.calibration.calibration_v5_hash,
            "realization_receipt_hashes": [row.receipt_hash for row in self.realizations],
            "alias_receipt_hashes": [row.receipt_hash for row in self.aliases],
            "runtime_cost_ledger_hashes": [row.ledger_hash for row in self.runtime_cost_ledgers],
            "audit_count": self.audit_count,
            "child_count": self.child_count,
            "production_registry_hash": self.production_registry.registry_hash,
            "split_registry_hash": self.proposal.manifest.split_registry.registry_hash,
            "selector_implementation_sha256": self.selector_implementation_sha256,
            "selector_schema_fingerprint": self.selector_schema_fingerprint,
            "scope": self.scope,
            "frozen_before_decision": self.frozen_before_decision,
        }
        object.__setattr__(self, "bundle_hash", _sealed(
            self.bundle_hash, payload, "decision replay bundle hash",
        ))


def _replay_bundle(bundle: DecisionReplayBundleV7V7) -> v6.PortfolioDecisionV7V6:
    """Invoke the exact predecessor selector over the complete typed preimage."""
    if not isinstance(bundle, DecisionReplayBundleV7V7):
        raise ValueError("decision replay requires a typed bundle")
    return v6.select_portfolio_v7_v6(
        bundle.proposal,
        bundle.observations,
        bundle.risks,
        bundle.calibration,
        bundle.realizations,
        bundle.aliases,
        bundle.runtime_cost_ledgers,
        production_registry=bundle.production_registry,
        trusted_production_registry_hashes=frozenset({
            bundle.production_registry.registry_hash,
        }),
        trusted_split_registry_hashes=frozenset({
            bundle.proposal.manifest.split_registry.registry_hash,
        }),
        audit_count=bundle.audit_count,
        child_count=bundle.child_count,
    )


@dataclass(frozen=True)
class DecisionReplayReceiptV7V7:
    """Replay evidence whose validity is established by executing the selector."""

    bundle: DecisionReplayBundleV7V7
    replayed_decision: v6.PortfolioDecisionV7V6
    selector_implementation_sha256: str = SELECTOR_V7_V6_REVIEWED_SHA256
    selector_schema_fingerprint: str = SELECTOR_V7_V6_SCHEMA_FINGERPRINT
    replay_performed: bool = True
    decision_output_payload_hash: str = ""
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.bundle, DecisionReplayBundleV7V7)
            or not _rehash_valid(self.bundle)
            or not isinstance(self.replayed_decision, v6.PortfolioDecisionV7V6)
            or not _rehash_valid(self.replayed_decision)
        ):
            raise ValueError("decision replay receipt needs valid typed preimages")
        _require_s5_decision(self.replayed_decision)
        if (
            self.selector_implementation_sha256 != SELECTOR_V7_V6_REVIEWED_SHA256
            or self.selector_implementation_sha256
            != self.bundle.selector_implementation_sha256
            or self.selector_schema_fingerprint != SELECTOR_V7_V6_SCHEMA_FINGERPRINT
            or self.selector_schema_fingerprint
            != self.bundle.selector_schema_fingerprint
            or self.replay_performed is not True
        ):
            raise ValueError("decision replay execution identity drifted")
        expected = _replay_bundle(self.bundle)
        _require_s5_decision(expected)
        if expected != self.replayed_decision:
            raise ValueError("decision does not equal deterministic selector replay")
        output_hash = _sha256(_decision_payload(expected))
        if self.decision_output_payload_hash not in {"", output_hash}:
            raise ValueError("decision replay output payload hash drifted")
        object.__setattr__(self, "decision_output_payload_hash", output_hash)
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash,
            {
                "bundle_hash": self.bundle.bundle_hash,
                "replayed_decision_v6_hash": expected.decision_v6_hash,
                "replayed_base_decision_hash": expected.base_decision.decision_hash,
                "decision_output_payload_hash": output_hash,
                "proposal_v6_hash": self.bundle.proposal.proposal_v6_hash,
                "manifest_v6_hash": self.bundle.proposal.manifest.manifest_v6_hash,
                "production_registry_hash": self.bundle.production_registry.registry_hash,
                "split_registry_hash": (
                    self.bundle.proposal.manifest.split_registry.registry_hash
                ),
                "selector_implementation_sha256": self.selector_implementation_sha256,
                "selector_schema_fingerprint": self.selector_schema_fingerprint,
                "replay_performed": self.replay_performed,
                "scope": SELECTOR_V7_V7_SCOPE,
            },
            "decision replay receipt hash",
        ))


@dataclass(frozen=True)
class PortfolioDecisionV7V7:
    base_decision: v6.PortfolioDecisionV7V6
    replay_bundle: DecisionReplayBundleV7V7
    replay_receipt: DecisionReplayReceiptV7V7
    scope: str = SELECTOR_V7_V7_SCOPE
    audit_status: str = S6_AUDIT_STATUS
    decision_v7_hash: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.base_decision, v6.PortfolioDecisionV7V6)
            or not _rehash_valid(self.base_decision)
            or not isinstance(self.replay_bundle, DecisionReplayBundleV7V7)
            or not _rehash_valid(self.replay_bundle)
            or not isinstance(self.replay_receipt, DecisionReplayReceiptV7V7)
            or not _rehash_valid(self.replay_receipt)
        ):
            raise ValueError("v7 decision needs valid typed replay artifacts")
        _require_s5_decision(self.base_decision)
        if self.scope != SELECTOR_V7_V7_SCOPE or self.audit_status != S6_AUDIT_STATUS:
            raise ValueError("v7 decision scope drifted")
        if (
            self.base_decision.proposal != self.replay_bundle.proposal
            or self.replay_receipt.bundle != self.replay_bundle
            or self.replay_receipt.replayed_decision != self.base_decision
        ):
            raise ValueError("v7 decision differs from its replay bundle/receipt")
        expected = _replay_bundle(self.replay_bundle)
        _require_s5_decision(expected)
        if expected != self.base_decision:
            raise ValueError("v7 decision does not equal deterministic selector replay")
        output_hash = _sha256(_decision_payload(expected))
        if output_hash != self.replay_receipt.decision_output_payload_hash:
            raise ValueError("v7 decision full output differs from replay receipt")
        object.__setattr__(self, "decision_v7_hash", _sealed(
            self.decision_v7_hash,
            {
                "schema_version": SELECTOR_V7_V7_SCHEMA_VERSION,
                "base_decision_v6_hash": self.base_decision.decision_v6_hash,
                "base_decision_hash": self.base_decision.base_decision.decision_hash,
                "replay_bundle_hash": self.replay_bundle.bundle_hash,
                "replay_receipt_hash": self.replay_receipt.receipt_hash,
                "decision_output_payload_hash": output_hash,
                "proposal_v6_hash": self.replay_bundle.proposal.proposal_v6_hash,
                "manifest_v6_hash": (
                    self.replay_bundle.proposal.manifest.manifest_v6_hash
                ),
                "production_registry_hash": (
                    self.replay_bundle.production_registry.registry_hash
                ),
                "split_registry_hash": (
                    self.replay_bundle.proposal.manifest.split_registry.registry_hash
                ),
                "state": self.base_decision.base_decision.state.value,
                "scope": self.scope,
                "audit_status": self.audit_status,
            },
            "SelectorV7-v7 decision hash",
        ))


def make_decision_replay_bundle_v7_v7(
    proposal: v6.PlannerProposalV7V6,
    observations: Sequence[v5.CandidateObservationV7V5],
    risks: Sequence[v5.ActionRiskVectorV7V5],
    calibration: v5.CalibrationReceiptV7V5,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    runtime_cost_ledgers: Sequence[v5.RuntimeCostLedgerV7V5],
    *,
    production_registry: v6.TrustedProductionAuthorityRegistryV7V6,
    audit_count: int = 0,
    child_count: int = 0,
) -> DecisionReplayBundleV7V7:
    return DecisionReplayBundleV7V7(
        proposal=proposal,
        observations=tuple(observations),
        risks=tuple(risks),
        calibration=calibration,
        realizations=tuple(realizations),
        aliases=tuple(aliases),
        runtime_cost_ledgers=tuple(runtime_cost_ledgers),
        audit_count=audit_count,
        child_count=child_count,
        production_registry=production_registry,
    )


def select_portfolio_v7_v7(
    proposal: v6.PlannerProposalV7V6,
    observations: Sequence[v5.CandidateObservationV7V5],
    risks: Sequence[v5.ActionRiskVectorV7V5],
    calibration: v5.CalibrationReceiptV7V5,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    runtime_cost_ledgers: Sequence[v5.RuntimeCostLedgerV7V5],
    *,
    production_registry: v6.TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
    audit_count: int = 0,
    child_count: int = 0,
) -> PortfolioDecisionV7V7:
    if (
        production_registry.registry_hash
        not in trusted_production_registry_hashes
        or proposal.manifest.split_registry.registry_hash
        not in trusted_split_registry_hashes
    ):
        raise ValueError("v7 selector lacks external production/split trust")
    bundle = make_decision_replay_bundle_v7_v7(
        proposal,
        observations,
        risks,
        calibration,
        realizations,
        aliases,
        runtime_cost_ledgers,
        production_registry=production_registry,
        audit_count=audit_count,
        child_count=child_count,
    )
    replayed = _replay_bundle(bundle)
    _require_s5_decision(replayed)
    receipt = DecisionReplayReceiptV7V7(bundle, replayed)
    decision = PortfolioDecisionV7V7(replayed, bundle, receipt)
    if not verify_production_decision_v7_v7(
        decision,
        production_registry=production_registry,
        trusted_production_registry_hashes=trusted_production_registry_hashes,
        trusted_split_registry_hashes=trusted_split_registry_hashes,
    ):
        raise ValueError("new v7 decision failed exact deterministic replay")
    return decision


def verify_production_decision_v7_v7(
    decision: object,
    *,
    production_registry: v6.TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
) -> bool:
    """Production gate: replay, then compare the complete decision exactly."""
    try:
        if not isinstance(decision, PortfolioDecisionV7V7):
            return False
        if decision.base_decision.base_decision.state is v5.DecisionStateV7V5.AUDIT:
            return False
        if (
            not _rehash_valid(decision)
            or decision.scope != SELECTOR_V7_V7_SCOPE
            or decision.audit_status != S6_AUDIT_STATUS
            or decision.replay_bundle.production_registry != production_registry
            or production_registry.registry_hash
            not in trusted_production_registry_hashes
            or decision.replay_bundle.proposal.manifest.split_registry.registry_hash
            not in trusted_split_registry_hashes
        ):
            return False
        if not v6.verify_production_proposal_v7_v6(
            decision.replay_bundle.proposal,
            production_registry=production_registry,
            trusted_production_registry_hashes=trusted_production_registry_hashes,
            trusted_split_registry_hashes=trusted_split_registry_hashes,
        ):
            return False
        replayed = _replay_bundle(decision.replay_bundle)
        _require_s5_decision(replayed)
        if replayed != decision.base_decision:
            return False
        expected_receipt = DecisionReplayReceiptV7V7(
            decision.replay_bundle, replayed,
        )
        if expected_receipt != decision.replay_receipt:
            return False
        return bool(v6.verify_production_decision_v7_v6(
            replayed,
            production_registry=production_registry,
            trusted_production_registry_hashes=trusted_production_registry_hashes,
            trusted_split_registry_hashes=trusted_split_registry_hashes,
        ))
    except Exception:
        return False


def serialize_production_decision_v7_v7(
    decision: object,
    *,
    production_registry: v6.TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
) -> bytes:
    if not verify_production_decision_v7_v7(
        decision,
        production_registry=production_registry,
        trusted_production_registry_hashes=trusted_production_registry_hashes,
        trusted_split_registry_hashes=trusted_split_registry_hashes,
    ):
        raise ValueError("decision is not the exact trusted S5 selector replay")
    assert isinstance(decision, PortfolioDecisionV7V7)
    payload = _decision_payload(decision.base_decision)
    payload.update({
        "schema_version": SELECTOR_V7_V7_SCHEMA_VERSION,
        "decision_v7_hash": decision.decision_v7_hash,
        "replay_bundle_hash": decision.replay_bundle.bundle_hash,
        "replay_receipt_hash": decision.replay_receipt.receipt_hash,
        "decision_output_payload_hash": (
            decision.replay_receipt.decision_output_payload_hash
        ),
        "selector_implementation_sha256": (
            decision.replay_bundle.selector_implementation_sha256
        ),
        "selector_schema_fingerprint": (
            decision.replay_bundle.selector_schema_fingerprint
        ),
        "production_registry_hash": (
            decision.replay_bundle.production_registry.registry_hash
        ),
        "split_registry_hash": (
            decision.replay_bundle.proposal.manifest.split_registry.registry_hash
        ),
        "scope": decision.scope,
        "audit_status": decision.audit_status,
    })
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


# R01--R14 are inherited byte-for-byte through the reviewed v6 candidate.
ActionRiskVectorV7V7 = v6.ActionRiskVectorV7V6
AliasChronologyModeV7V7 = v6.AliasChronologyModeV7V6
AliasRunPlanV7V7 = v6.AliasRunPlanV7V6
CalibrationReceiptV7V7 = v6.CalibrationReceiptV7V6
CandidateManifestV7V7 = v6.CandidateManifestV7V6
CandidateObservationV7V7 = v6.CandidateObservationV7V6
DecisionReasonV7V7 = v6.DecisionReasonV7V6
DecisionStateV7V7 = v6.DecisionStateV7V6
ExternalInferenceReplayAuthorityV7V7 = v6.ExternalInferenceReplayAuthorityV7V6
ExternalInferenceReplayVerificationV7V7 = v6.ExternalInferenceReplayVerificationV7V6
ExternalReplayAuthorityModeV7V7 = v6.ExternalReplayAuthorityModeV7V6
FiveFoldPartitionV7V7 = v6.FiveFoldPartitionV7V6
FiveFoldSplitAuthorityRootV7V7 = v6.FiveFoldSplitAuthorityRootV7V6
PlannerProposalV7V7 = v6.PlannerProposalV7V6
PlannerScoreReceiptV7V7 = v6.PlannerScoreReceiptV7V6
ProductionEligibilityV7V7 = v6.ProductionEligibilityV7V6
ProductionReplayAuthorityEntryV7V7 = v6.ProductionReplayAuthorityEntryV7V6
RuntimeCostLedgerV7V7 = v6.RuntimeCostLedgerV7V6
SplitAuthorityReceiptV7V7 = v6.SplitAuthorityReceiptV7V6
SR0ASplitRootAuthorityV7V7 = v6.SR0ASplitRootAuthorityV7V6
TrustedProductionAuthorityRegistryV7V7 = v6.TrustedProductionAuthorityRegistryV7V6
TrustedSR0ASplitRegistryV7V7 = v6.TrustedSR0ASplitRegistryV7V6
bind_candidate_manifest_v7_v7 = v6.bind_candidate_manifest_v7_v6
propose_candidates_v7_v7 = v6.propose_candidates_v7_v6
serialize_production_proposal_v7_v7 = v6.serialize_production_proposal_v7_v6
verify_manifest_split_authority_v7_v7 = v6.verify_manifest_split_authority_v7_v6
verify_production_proposal_v7_v7 = v6.verify_production_proposal_v7_v6


def selector_v7_v7_schema_fingerprint() -> str:
    classes = (
        SelectorV7V7Disposition,
        DecisionReplayBundleV7V7,
        DecisionReplayReceiptV7V7,
        PortfolioDecisionV7V7,
    )
    return _sha256({
        "schema_version": SELECTOR_V7_V7_SCHEMA_VERSION,
        "canonicalization_version": SELECTOR_V7_V7_CANONICALIZATION_VERSION,
        "v6_source_sha256": SELECTOR_V7_V6_REVIEWED_SHA256,
        "v6_test_sha256": SELECTOR_V7_V6_REVIEWED_TEST_SHA256,
        "v6_package_sha256": SELECTOR_V7_V6_REVIEWED_PACKAGE_SHA256,
        "v6_fail_review_sha256": SR1_ROOT_REVIEW_V6_SHA256,
        "v6_fail_review_status": SR1_ROOT_REVIEW_V6_STATUS,
        "v6_schema_fingerprint": SELECTOR_V7_V6_SCHEMA_FINGERPRINT,
        "scope": SELECTOR_V7_V7_SCOPE,
        "audit_status": S6_AUDIT_STATUS,
        "classes": {
            cls.__name__: list(cls.__dataclass_fields__) for cls in classes
        },
    })


__all__ = [
    "ActionRiskVectorV7V7",
    "AliasChronologyModeV7V7",
    "AliasRunPlanV7V7",
    "CalibrationReceiptV7V7",
    "CandidateManifestV7V7",
    "CandidateObservationV7V7",
    "DecisionReasonV7V7",
    "DecisionReplayBundleV7V7",
    "DecisionReplayReceiptV7V7",
    "DecisionStateV7V7",
    "ExternalInferenceReplayAuthorityV7V7",
    "ExternalInferenceReplayVerificationV7V7",
    "ExternalReplayAuthorityModeV7V7",
    "FiveFoldPartitionV7V7",
    "FiveFoldSplitAuthorityRootV7V7",
    "PlannerProposalV7V7",
    "PlannerScoreReceiptV7V7",
    "PortfolioDecisionV7V7",
    "ProductionEligibilityV7V7",
    "ProductionReplayAuthorityEntryV7V7",
    "RuntimeCostLedgerV7V7",
    "S6_AUDIT_STATUS",
    "SELECTOR_V7_V6_REVIEWED_PACKAGE_SHA256",
    "SELECTOR_V7_V6_REVIEWED_SHA256",
    "SELECTOR_V7_V6_REVIEWED_TEST_SHA256",
    "SELECTOR_V7_V6_SCHEMA_FINGERPRINT",
    "SELECTOR_V7_V7_CANONICALIZATION_VERSION",
    "SELECTOR_V7_V7_SCHEMA_VERSION",
    "SELECTOR_V7_V7_SCOPE",
    "SR0ASplitRootAuthorityV7V7",
    "SR1_ROOT_REVIEW_V6_SHA256",
    "SR1_ROOT_REVIEW_V6_STATUS",
    "SelectorV7V7Disposition",
    "SplitAuthorityReceiptV7V7",
    "TrustedProductionAuthorityRegistryV7V7",
    "TrustedSR0ASplitRegistryV7V7",
    "bind_candidate_manifest_v7_v7",
    "make_decision_replay_bundle_v7_v7",
    "propose_candidates_v7_v7",
    "select_portfolio_v7_v7",
    "serialize_production_decision_v7_v7",
    "serialize_production_proposal_v7_v7",
    "selector_v7_v7_schema_fingerprint",
    "verify_manifest_split_authority_v7_v7",
    "verify_production_decision_v7_v7",
    "verify_production_proposal_v7_v7",
]
