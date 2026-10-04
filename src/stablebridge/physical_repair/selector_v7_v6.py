"""Append-only SelectorV7 SR1-v6 authority successor.

V6 preserves the reviewed v5 scientific/selection semantics and adds external,
typed trust anchors at the two v5 root-review gaps:

* production consumers recursively replay typed E235 authority/verification
  receipt preimages through an externally trusted registry;
* the exact SR0A five-fold universe and all five assignments are carried by a
  typed split-root registry whose hash must be trusted outside the manifest.

This module grants no authority seal.  A later independent root review must
adjudicate these candidate bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Sequence

from . import selector_v7 as v1
from . import selector_v7_v5 as v5


SELECTOR_V7_V6_SCHEMA_VERSION = "selector-v7-contract/v4.39-sr1-v6-candidate"
SELECTOR_V7_V6_CANONICALIZATION_VERSION = "selector-v7-v6-canonical-json/v1"
SELECTOR_V7_V5_REVIEWED_SHA256 = (
    "3dc3b5dc5b52243216f451f8d45edf428312765e36e4827a272d73ac2c8efbb7"
)
SELECTOR_V7_V5_REVIEWED_TEST_SHA256 = (
    "035b59508b7dcd8633b5c62c9dd0397d486af8f726a16c15634fd30d96874821"
)
SR1_ROOT_REVIEW_V5_SHA256 = (
    "005f63494c094dff78925f6f59c3036cc779fb38758539a283fff641d8404103"
)
SR1_ROOT_REVIEW_V5_STATUS = (
    "FAIL_SR1_V5_COORDINATED_RETAG_AND_SPLIT_ROOT_REANCHOR_BYPASSES_NO_SEAL"
)
TEST_ONLY_PRODUCTION_REGISTRY_MARKER = "TEST_ONLY_NO_PRODUCTION_REGISTRY"


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


@dataclass(frozen=True)
class SelectorV7V6Disposition:
    predecessor_review_status: str = SR1_ROOT_REVIEW_V5_STATUS
    predecessor_review_sha256: str = SR1_ROOT_REVIEW_V5_SHA256
    predecessor_source_sha256: str = SELECTOR_V7_V5_REVIEWED_SHA256
    predecessor_test_sha256: str = SELECTOR_V7_V5_REVIEWED_TEST_SHA256
    successor_status: str = "CANDIDATE_PENDING_INDEPENDENT_ROOT_REVIEW"

    def __post_init__(self) -> None:
        if (
            self.predecessor_review_status != SR1_ROOT_REVIEW_V5_STATUS
            or self.predecessor_review_sha256 != SR1_ROOT_REVIEW_V5_SHA256
            or self.predecessor_source_sha256 != SELECTOR_V7_V5_REVIEWED_SHA256
            or self.predecessor_test_sha256 != SELECTOR_V7_V5_REVIEWED_TEST_SHA256
            or self.successor_status
            != "CANDIDATE_PENDING_INDEPENDENT_ROOT_REVIEW"
        ):
            raise ValueError("SelectorV7-v6 predecessor disposition drifted")


@dataclass(frozen=True)
class SR0ASplitRootAuthorityV7V6:
    """Typed preimage for one externally trusted SR0A split root."""

    sr0a_manifest_receipt_hash: str
    root_receipt: v5.FiveFoldSplitAuthorityRootV7V5
    authority_provenance_hash: str
    frozen_before_sr1: bool
    authority_hash: str = ""

    def __post_init__(self) -> None:
        _text(self.sr0a_manifest_receipt_hash, "SR0A manifest receipt hash")
        _text(self.authority_provenance_hash, "split authority provenance hash")
        if not isinstance(
            self.root_receipt, v5.FiveFoldSplitAuthorityRootV7V5,
        ) or not _rehash_valid(self.root_receipt):
            raise ValueError("split authority needs a valid typed root receipt")
        root = self.root_receipt
        if root.sr0a_manifest_receipt_hash != self.sr0a_manifest_receipt_hash:
            raise ValueError("typed split root differs from SR0A manifest receipt")
        if self.frozen_before_sr1 is not True or root.frozen_before_fit is not True:
            raise ValueError("typed split root must be frozen before SR1")
        assignments = [
            {
                "target": row.target_fold_id,
                "calibration": row.calibration_fold_id,
                "fit": list(row.fit_fold_ids),
                "evaluation": row.evaluation_fold_id,
                "partition_hash": row.partition_hash,
            }
            for row in root.partitions
        ]
        payload = {
            "sr0a_manifest_receipt_hash": self.sr0a_manifest_receipt_hash,
            "root_receipt_hash": root.root_hash,
            "fold_universe_hash": root.fold_universe_hash,
            "fold_universe_ids": list(root.fold_universe_ids),
            "assignments": assignments,
            "authority_provenance_hash": self.authority_provenance_hash,
            "frozen_before_sr1": self.frozen_before_sr1,
        }
        object.__setattr__(self, "authority_hash", _sealed(
            self.authority_hash, payload, "typed SR0A split-root authority hash",
        ))


@dataclass(frozen=True)
class TrustedSR0ASplitRegistryV7V6:
    registry_id: str
    registry_version: str
    authorities: tuple[SR0ASplitRootAuthorityV7V6, ...]
    frozen_before_sr1: bool
    registry_hash: str = ""

    def __post_init__(self) -> None:
        _text(self.registry_id, "split registry id")
        _text(self.registry_version, "split registry version")
        rows = tuple(sorted(
            self.authorities,
            key=lambda row: row.sr0a_manifest_receipt_hash,
        ))
        if not rows or any(
            not isinstance(row, SR0ASplitRootAuthorityV7V6)
            or not _rehash_valid(row)
            for row in rows
        ):
            raise ValueError("split registry needs valid typed authorities")
        if len({row.sr0a_manifest_receipt_hash for row in rows}) != len(rows):
            raise ValueError("one exact split root is allowed per SR0A receipt")
        if len({row.authority_hash for row in rows}) != len(rows):
            raise ValueError("split registry authority hashes must be unique")
        if self.frozen_before_sr1 is not True:
            raise ValueError("split registry must be frozen before SR1")
        payload = {
            "registry_id": self.registry_id,
            "registry_version": self.registry_version,
            "authority_hashes": [row.authority_hash for row in rows],
            "frozen_before_sr1": self.frozen_before_sr1,
        }
        object.__setattr__(self, "authorities", rows)
        object.__setattr__(self, "registry_hash", _sealed(
            self.registry_hash, payload, "trusted SR0A split registry hash",
        ))


@dataclass(frozen=True)
class ProductionReplayAuthorityEntryV7V6:
    """Typed production E235 authority plus its exact verification preimage."""

    authority: v5.ExternalInferenceReplayAuthorityV7V5
    verification: v5.ExternalInferenceReplayVerificationV7V5
    entry_hash: str = ""

    def __post_init__(self) -> None:
        authority = self.authority
        verification = self.verification
        if (
            not isinstance(authority, v5.ExternalInferenceReplayAuthorityV7V5)
            or not isinstance(
                verification, v5.ExternalInferenceReplayVerificationV7V5,
            )
            or not _rehash_valid(authority)
            or not _rehash_valid(verification)
        ):
            raise ValueError("production registry needs valid typed E235 receipts")
        if (
            authority.mode is not v5.ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED
            or verification.mode
            is not v5.ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED
            or authority.root_review_status != "PASS"
            or verification.root_review_status != "PASS"
        ):
            raise ValueError("production registry categorically rejects TEST_ONLY preimages")
        exact = (
            verification.authority_hash == authority.authority_hash
            and verification.planner_model_bytes_hash
            == authority.planner_model_bytes_hash
            and verification.planner_feature_schema_hash
            == authority.planner_feature_schema_hash
            and verification.integration_allowlist_hash
            == authority.integration_allowlist_hash
            and verification.root_review_seal_hash
            == authority.root_review_seal_hash
            and verification.output_replay_schema_hash
            == authority.output_replay_schema_hash
        )
        if not exact:
            raise ValueError("production verification differs from typed authority")
        object.__setattr__(self, "entry_hash", _sealed(
            self.entry_hash,
            {
                "authority_hash": authority.authority_hash,
                "verification_hash": verification.verification_hash,
                "mode": authority.mode.value,
                "root_review_seal_hash": authority.root_review_seal_hash,
            },
            "production replay authority entry hash",
        ))


@dataclass(frozen=True)
class TrustedProductionAuthorityRegistryV7V6:
    registry_id: str
    registry_version: str
    entries: tuple[ProductionReplayAuthorityEntryV7V6, ...]
    frozen_before_production: bool
    registry_hash: str = ""

    def __post_init__(self) -> None:
        _text(self.registry_id, "production registry id")
        _text(self.registry_version, "production registry version")
        rows = tuple(sorted(
            self.entries, key=lambda row: row.authority.authority_hash,
        ))
        if not rows or any(
            not isinstance(row, ProductionReplayAuthorityEntryV7V6)
            or not _rehash_valid(row)
            for row in rows
        ):
            raise ValueError("production registry needs typed authority entries")
        if len({row.authority.authority_hash for row in rows}) != len(rows):
            raise ValueError("production authority hashes must be unique")
        if len({row.verification.verification_hash for row in rows}) != len(rows):
            raise ValueError("production verification hashes must be unique")
        if self.frozen_before_production is not True:
            raise ValueError("production registry must be externally frozen")
        object.__setattr__(self, "entries", rows)
        object.__setattr__(self, "registry_hash", _sealed(
            self.registry_hash,
            {
                "registry_id": self.registry_id,
                "registry_version": self.registry_version,
                "entry_hashes": [row.entry_hash for row in rows],
                "frozen_before_production": self.frozen_before_production,
            },
            "trusted production authority registry hash",
        ))


def _split_authority_for_manifest(
    base: v5.CandidateManifestV7V5,
    registry: TrustedSR0ASplitRegistryV7V6,
) -> SR0ASplitRootAuthorityV7V6:
    split = base.planner_model_receipt.fit_receipt.split_authority
    matches = tuple(
        row for row in registry.authorities
        if row.sr0a_manifest_receipt_hash == split.sr0a_manifest_receipt_hash
        and row.root_receipt.root_hash == split.split_authority_root_hash
    )
    if len(matches) != 1:
        raise ValueError("manifest split root is absent from typed SR0A registry")
    return matches[0]


@dataclass(frozen=True)
class CandidateManifestV7V6:
    base_manifest: v5.CandidateManifestV7V5
    split_registry: TrustedSR0ASplitRegistryV7V6
    split_authority_hash: str
    schema_version: str = SELECTOR_V7_V6_SCHEMA_VERSION
    predecessor_fail_review_sha256: str = SR1_ROOT_REVIEW_V5_SHA256
    manifest_v6_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SELECTOR_V7_V6_SCHEMA_VERSION:
            raise ValueError("SelectorV7-v6 schema version drifted")
        if self.predecessor_fail_review_sha256 != SR1_ROOT_REVIEW_V5_SHA256:
            raise ValueError("SelectorV7-v6 predecessor review drifted")
        if (
            not isinstance(self.base_manifest, v5.CandidateManifestV7V5)
            or not _rehash_valid(self.base_manifest)
            or not isinstance(self.split_registry, TrustedSR0ASplitRegistryV7V6)
            or not _rehash_valid(self.split_registry)
        ):
            raise ValueError("v6 manifest needs valid v5 manifest and split registry")
        authority = _split_authority_for_manifest(
            self.base_manifest, self.split_registry,
        )
        if authority.authority_hash != self.split_authority_hash:
            raise ValueError("v6 manifest split authority hash drifted")
        model_split = self.base_manifest.planner_model_receipt.fit_receipt.split_authority
        root = authority.root_receipt
        partition = next(
            (row for row in root.partitions
             if row.target_fold_id == model_split.target_fold_id),
            None,
        )
        freeze = self.base_manifest.freeze_receipt
        exact = (
            model_split.authority_root_receipt == root
            and model_split.split_authority_root_hash == root.root_hash
            and model_split.fold_universe_ids == root.fold_universe_ids
            and model_split.fold_universe_hash == root.fold_universe_hash
            and model_split.sr0a_manifest_receipt_hash
            == authority.sr0a_manifest_receipt_hash
            and freeze.sr0a_manifest_receipt_hash
            == authority.sr0a_manifest_receipt_hash
            and freeze.split_authority_root_hash == root.root_hash
            and freeze.fold_universe_hash == root.fold_universe_hash
            and partition is not None
            and model_split.partition_root_hash == partition.partition_hash
            and model_split.assignment_hash == partition.partition_hash
            and model_split.fit_fold_ids == partition.fit_fold_ids
            and model_split.calibration_fold_ids
            == (partition.calibration_fold_id,)
            and model_split.evaluation_fold_ids
            == (partition.evaluation_fold_id,)
        )
        if not exact:
            raise ValueError("manifest differs from externally registered split assignment")
        object.__setattr__(self, "manifest_v6_hash", _sealed(
            self.manifest_v6_hash,
            {
                "schema_version": self.schema_version,
                "predecessor_fail_review_sha256": self.predecessor_fail_review_sha256,
                "base_manifest_v5_hash": self.base_manifest.manifest_v5_hash,
                "split_registry_hash": self.split_registry.registry_hash,
                "split_authority_hash": authority.authority_hash,
            },
            "SelectorV7-v6 manifest hash",
        ))


def bind_candidate_manifest_v7_v6(
    base_manifest: v5.CandidateManifestV7V5,
    split_registry: TrustedSR0ASplitRegistryV7V6,
    *,
    trusted_split_registry_hashes: frozenset[str],
) -> CandidateManifestV7V6:
    if split_registry.registry_hash not in trusted_split_registry_hashes:
        raise ValueError("SR0A split registry lacks external trust")
    authority = _split_authority_for_manifest(base_manifest, split_registry)
    return CandidateManifestV7V6(
        base_manifest=base_manifest,
        split_registry=split_registry,
        split_authority_hash=authority.authority_hash,
    )


def verify_manifest_split_authority_v7_v6(
    manifest: object,
    *,
    trusted_split_registry_hashes: frozenset[str],
) -> bool:
    return bool(
        isinstance(manifest, CandidateManifestV7V6)
        and _rehash_valid(manifest)
        and manifest.split_registry.registry_hash
        in trusted_split_registry_hashes
    )


@dataclass(frozen=True)
class PlannerProposalV7V6:
    base_proposal: v5.PlannerProposalV7V5
    manifest: CandidateManifestV7V6
    production_registry_hash: str
    proposal_v6_hash: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.base_proposal, v5.PlannerProposalV7V5)
            or not _rehash_valid(self.base_proposal)
            or not isinstance(self.manifest, CandidateManifestV7V6)
            or not _rehash_valid(self.manifest)
        ):
            raise ValueError("v6 proposal needs valid typed proposal and manifest")
        _text(self.production_registry_hash, "proposal production registry hash")
        base = self.base_proposal
        if base.manifest_v5_hash != self.manifest.base_manifest.manifest_v5_hash:
            raise ValueError("v6 proposal differs from v6 manifest")
        if base.score_bank.planner_model_receipt != (
            self.manifest.base_manifest.planner_model_receipt
        ):
            raise ValueError("v6 proposal model differs from manifest authority")
        if base.external_authority_mode is v5.ExternalReplayAuthorityModeV7V5.TEST_ONLY:
            if self.production_registry_hash != TEST_ONLY_PRODUCTION_REGISTRY_MARKER:
                raise ValueError("TEST_ONLY proposal cannot claim production registry")
        elif (
            base.external_authority_mode
            is v5.ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED
            and self.production_registry_hash == TEST_ONLY_PRODUCTION_REGISTRY_MARKER
        ):
            raise ValueError("production proposal needs typed production registry")
        object.__setattr__(self, "proposal_v6_hash", _sealed(
            self.proposal_v6_hash,
            {
                "base_proposal_hash": base.proposal_hash,
                "manifest_v6_hash": self.manifest.manifest_v6_hash,
                "split_registry_hash": self.manifest.split_registry.registry_hash,
                "production_registry_hash": self.production_registry_hash,
                "external_authority_mode": base.external_authority_mode.value,
                "external_authority_hash": base.external_authority_hash,
                "external_verification_hash": base.external_verification_hash,
                "production_eligibility": base.production_eligibility.value,
            },
            "SelectorV7-v6 proposal hash",
        ))


@dataclass(frozen=True)
class PortfolioDecisionV7V6:
    base_decision: v5.PortfolioDecisionV7V5
    proposal: PlannerProposalV7V6
    decision_v6_hash: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.base_decision, v5.PortfolioDecisionV7V5)
            or not _rehash_valid(self.base_decision)
            or not isinstance(self.proposal, PlannerProposalV7V6)
            or not _rehash_valid(self.proposal)
        ):
            raise ValueError("v6 decision needs valid typed decision and proposal")
        decision = self.base_decision
        base_proposal = self.proposal.base_proposal
        if (
            decision.proposal_hash != base_proposal.proposal_hash
            or decision.manifest_v5_hash
            != self.proposal.manifest.base_manifest.manifest_v5_hash
            or decision.external_authority_mode
            is not base_proposal.external_authority_mode
            or decision.external_authority_hash
            != base_proposal.external_authority_hash
            or decision.external_verification_hash
            != base_proposal.external_verification_hash
            or decision.production_eligibility
            is not base_proposal.production_eligibility
        ):
            raise ValueError("v6 decision differs from its typed proposal artifact")
        object.__setattr__(self, "decision_v6_hash", _sealed(
            self.decision_v6_hash,
            {
                "base_decision_hash": decision.decision_hash,
                "proposal_v6_hash": self.proposal.proposal_v6_hash,
                "manifest_v6_hash": self.proposal.manifest.manifest_v6_hash,
                "production_registry_hash": self.proposal.production_registry_hash,
                "state": decision.state.value,
                "external_authority_mode": decision.external_authority_mode.value,
                "external_authority_hash": decision.external_authority_hash,
                "external_verification_hash": decision.external_verification_hash,
                "production_eligibility": decision.production_eligibility.value,
            },
            "SelectorV7-v6 decision hash",
        ))


def _entry_for_hashes(
    registry: TrustedProductionAuthorityRegistryV7V6,
    authority_hash: str,
    verification_hash: str,
) -> ProductionReplayAuthorityEntryV7V6:
    matches = tuple(
        row for row in registry.entries
        if row.authority.authority_hash == authority_hash
        and row.verification.verification_hash == verification_hash
    )
    if len(matches) != 1:
        raise ValueError("proposal hashes do not resolve to one typed production entry")
    return matches[0]


def _expected_execution_bindings(
    score_receipts: Sequence[v5.PlannerScoreReceiptV7V5],
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(sorted(
        (
            row.base_score.planned_arm_id,
            row.inference_execution.receipt_hash,
            row.inference_execution.external_replay_receipt_hash,
            row.inference_execution.output_score_bytes_hash,
        )
        for row in score_receipts
    ))


def _entry_matches_score_artifacts(
    manifest: v5.CandidateManifestV7V5,
    score_receipts: Sequence[v5.PlannerScoreReceiptV7V5],
    entry: ProductionReplayAuthorityEntryV7V6,
) -> bool:
    authority = entry.authority
    verification = entry.verification
    model = manifest.planner_model_receipt
    receipts = tuple(score_receipts)
    if not receipts or any(
        not isinstance(row, v5.PlannerScoreReceiptV7V5)
        or not _rehash_valid(row)
        for row in receipts
    ):
        return False
    exact = (
        model.required_e235_replay_authority_hash == authority.authority_hash
        and authority.planner_model_bytes_hash == model.model_bytes_hash
        and authority.planner_feature_schema_hash == model.feature_schema_hash
        and authority.planner_model_schema_hash == v5.PLANNER_MODEL_SCHEMA_HASH_V7V5
        and verification.authority_hash == authority.authority_hash
        and verification.base_manifest_hash == manifest.base_manifest.manifest_hash
        and verification.planner_model_hash == model.model_hash
        and verification.planner_model_bytes_hash == model.model_bytes_hash
        and verification.planner_feature_schema_hash == model.feature_schema_hash
        and verification.execution_bindings == _expected_execution_bindings(receipts)
    )
    if not exact:
        return False
    for receipt in receipts:
        artifact = receipt.before_feature_artifact
        execution = receipt.inference_execution
        if (
            artifact.producer_code_hash != authority.executor_code_hash
            or artifact.producer_config_hash != authority.producer_config_hash
            or artifact.runtime_hash != authority.runtime_hash
            or execution.producer_code_hash != authority.producer_code_hash
            or execution.producer_config_hash != authority.producer_config_hash
            or execution.runtime_hash != authority.runtime_hash
            or execution.required_e235_replay_authority_hash
            != authority.authority_hash
        ):
            return False
    return True


def verify_production_proposal_v7_v6(
    proposal: object,
    *,
    production_registry: TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
) -> bool:
    try:
        if (
            not isinstance(proposal, PlannerProposalV7V6)
            or not _rehash_valid(proposal)
            or not isinstance(
                production_registry, TrustedProductionAuthorityRegistryV7V6,
            )
            or not _rehash_valid(production_registry)
            or production_registry.registry_hash
            not in trusted_production_registry_hashes
            or proposal.production_registry_hash
            != production_registry.registry_hash
            or not verify_manifest_split_authority_v7_v6(
                proposal.manifest,
                trusted_split_registry_hashes=trusted_split_registry_hashes,
            )
        ):
            return False
        base = proposal.base_proposal
        if (
            base.external_authority_mode
            is not v5.ExternalReplayAuthorityModeV7V5.PRODUCTION_TRUSTED
            or base.production_eligibility
            is not v5.ProductionEligibilityV7V5.PRODUCTION_ELIGIBLE
        ):
            return False
        entry = _entry_for_hashes(
            production_registry,
            base.external_authority_hash,
            base.external_verification_hash,
        )
        if not _entry_matches_score_artifacts(
            proposal.manifest.base_manifest,
            base.score_bank.score_receipts,
            entry,
        ):
            return False
        return v5.verify_production_proposal_v7_v5(
            base,
            trusted_authority_hashes=frozenset({entry.authority.authority_hash}),
            trusted_verification_hashes=frozenset({
                entry.verification.verification_hash,
            }),
        )
    except Exception:
        return False


def serialize_production_proposal_v7_v6(
    proposal: object,
    *,
    production_registry: TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
) -> bytes:
    if not verify_production_proposal_v7_v6(
        proposal,
        production_registry=production_registry,
        trusted_production_registry_hashes=trusted_production_registry_hashes,
        trusted_split_registry_hashes=trusted_split_registry_hashes,
    ):
        raise ValueError("proposal lacks typed trusted production authority")
    assert isinstance(proposal, PlannerProposalV7V6)
    base = proposal.base_proposal
    return json.dumps({
        "schema_version": SELECTOR_V7_V6_SCHEMA_VERSION,
        "proposal_v6_hash": proposal.proposal_v6_hash,
        "base_proposal_hash": base.proposal_hash,
        "manifest_v6_hash": proposal.manifest.manifest_v6_hash,
        "split_registry_hash": proposal.manifest.split_registry.registry_hash,
        "production_registry_hash": proposal.production_registry_hash,
        "external_authority_mode": base.external_authority_mode.value,
        "external_authority_hash": base.external_authority_hash,
        "external_verification_hash": base.external_verification_hash,
        "production_eligibility": base.production_eligibility.value,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def verify_production_decision_v7_v6(
    decision: object,
    *,
    production_registry: TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
) -> bool:
    try:
        if (
            not isinstance(decision, PortfolioDecisionV7V6)
            or not _rehash_valid(decision)
            or not verify_production_proposal_v7_v6(
                decision.proposal,
                production_registry=production_registry,
                trusted_production_registry_hashes=(
                    trusted_production_registry_hashes
                ),
                trusted_split_registry_hashes=trusted_split_registry_hashes,
            )
        ):
            return False
        base = decision.base_decision
        proposal = decision.proposal.base_proposal
        if (
            base.proposal_hash != proposal.proposal_hash
            or base.external_authority_hash != proposal.external_authority_hash
            or base.external_verification_hash
            != proposal.external_verification_hash
        ):
            return False
        return v5.verify_production_decision_v7_v5(
            base,
            trusted_proposal_hashes=frozenset({proposal.proposal_hash}),
            trusted_authority_hashes=frozenset({proposal.external_authority_hash}),
            trusted_verification_hashes=frozenset({
                proposal.external_verification_hash,
            }),
        )
    except Exception:
        return False


def serialize_production_decision_v7_v6(
    decision: object,
    *,
    production_registry: TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
) -> bytes:
    if not verify_production_decision_v7_v6(
        decision,
        production_registry=production_registry,
        trusted_production_registry_hashes=trusted_production_registry_hashes,
        trusted_split_registry_hashes=trusted_split_registry_hashes,
    ):
        raise ValueError("decision lacks typed trusted proposal/authority chain")
    assert isinstance(decision, PortfolioDecisionV7V6)
    base = decision.base_decision
    return json.dumps({
        "schema_version": SELECTOR_V7_V6_SCHEMA_VERSION,
        "decision_v6_hash": decision.decision_v6_hash,
        "base_decision_hash": base.decision_hash,
        "proposal_v6_hash": decision.proposal.proposal_v6_hash,
        "manifest_v6_hash": decision.proposal.manifest.manifest_v6_hash,
        "state": base.state.value,
        "split_registry_hash": (
            decision.proposal.manifest.split_registry.registry_hash
        ),
        "production_registry_hash": decision.proposal.production_registry_hash,
        "external_authority_mode": base.external_authority_mode.value,
        "external_authority_hash": base.external_authority_hash,
        "external_verification_hash": base.external_verification_hash,
        "production_eligibility": base.production_eligibility.value,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _entry_for_score_receipts(
    manifest: CandidateManifestV7V6,
    score_receipts: Sequence[v5.PlannerScoreReceiptV7V5],
    registry: TrustedProductionAuthorityRegistryV7V6,
) -> ProductionReplayAuthorityEntryV7V6:
    matches = tuple(
        row for row in registry.entries
        if _entry_matches_score_artifacts(
            manifest.base_manifest, score_receipts, row,
        )
    )
    if len(matches) != 1:
        raise ValueError("score family does not resolve to one production entry")
    return matches[0]


def propose_candidates_v7_v6(
    manifest: CandidateManifestV7V6,
    score_receipts: Sequence[v5.PlannerScoreReceiptV7V5],
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    alias_run_plans: Sequence[v5.AliasRunPlanV7V5],
    *,
    production_registry: TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
    top_k: int,
    maximum_prospective_runtime_cost: float,
    maximum_prospective_runtime_cost_counters: v1.CostCountersV7,
    planner_algorithm_id: str,
    planner_algorithm_version: str,
    planner_config_hash: str,
    planner_provenance_hash: str,
) -> PlannerProposalV7V6:
    if (
        production_registry.registry_hash
        not in trusted_production_registry_hashes
        or not verify_manifest_split_authority_v7_v6(
            manifest,
            trusted_split_registry_hashes=trusted_split_registry_hashes,
        )
    ):
        raise ValueError("v6 planner lacks external registry trust")
    entry = _entry_for_score_receipts(manifest, score_receipts, production_registry)
    base = v5.propose_candidates_v7_v5(
        manifest.base_manifest,
        score_receipts,
        realizations,
        aliases,
        alias_run_plans,
        top_k=top_k,
        maximum_prospective_runtime_cost=maximum_prospective_runtime_cost,
        maximum_prospective_runtime_cost_counters=(
            maximum_prospective_runtime_cost_counters
        ),
        planner_algorithm_id=planner_algorithm_id,
        planner_algorithm_version=planner_algorithm_version,
        planner_config_hash=planner_config_hash,
        planner_provenance_hash=planner_provenance_hash,
        external_replay_authority=entry.authority,
        external_replay_verification=entry.verification,
        trusted_root_review_seal_hashes=frozenset({
            entry.authority.root_review_seal_hash,
        }),
        trusted_verification_hashes=frozenset({
            entry.verification.verification_hash,
        }),
    )
    proposal = PlannerProposalV7V6(
        base, manifest, production_registry.registry_hash,
    )
    if not verify_production_proposal_v7_v6(
        proposal,
        production_registry=production_registry,
        trusted_production_registry_hashes=trusted_production_registry_hashes,
        trusted_split_registry_hashes=trusted_split_registry_hashes,
    ):
        raise ValueError("new v6 proposal failed typed production replay")
    return proposal


def propose_candidates_v7_v6_test_only(
    manifest: CandidateManifestV7V6,
    score_receipts: Sequence[v5.PlannerScoreReceiptV7V5],
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    alias_run_plans: Sequence[v5.AliasRunPlanV7V5],
    **kwargs: object,
) -> PlannerProposalV7V6:
    base = v5.propose_candidates_v7_v5_test_only(
        manifest.base_manifest,
        score_receipts,
        realizations,
        aliases,
        alias_run_plans,
        **kwargs,
    )
    return PlannerProposalV7V6(
        base, manifest, TEST_ONLY_PRODUCTION_REGISTRY_MARKER,
    )


def select_portfolio_v7_v6(
    proposal: PlannerProposalV7V6,
    observations: Sequence[v5.CandidateObservationV7V5],
    risks: Sequence[v5.ActionRiskVectorV7V5],
    calibration: v5.CalibrationReceiptV7V5,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    runtime_cost_ledgers: Sequence[v5.RuntimeCostLedgerV7V5],
    *,
    production_registry: TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
    audit_count: int = 0,
    child_count: int = 0,
) -> PortfolioDecisionV7V6:
    if not verify_production_proposal_v7_v6(
        proposal,
        production_registry=production_registry,
        trusted_production_registry_hashes=trusted_production_registry_hashes,
        trusted_split_registry_hashes=trusted_split_registry_hashes,
    ):
        raise ValueError("v6 selector rejects untrusted proposal artifact")
    entry = _entry_for_hashes(
        production_registry,
        proposal.base_proposal.external_authority_hash,
        proposal.base_proposal.external_verification_hash,
    )
    base = v5.select_portfolio_v7_v5(
        proposal.manifest.base_manifest,
        proposal.base_proposal,
        observations,
        risks,
        calibration,
        realizations,
        aliases,
        runtime_cost_ledgers,
        audit_count=audit_count,
        child_count=child_count,
        external_replay_authority=entry.authority,
        external_replay_verification=entry.verification,
        trusted_root_review_seal_hashes=frozenset({
            entry.authority.root_review_seal_hash,
        }),
        trusted_verification_hashes=frozenset({
            entry.verification.verification_hash,
        }),
    )
    decision = PortfolioDecisionV7V6(base, proposal)
    if not verify_production_decision_v7_v6(
        decision,
        production_registry=production_registry,
        trusted_production_registry_hashes=trusted_production_registry_hashes,
        trusted_split_registry_hashes=trusted_split_registry_hashes,
    ):
        raise ValueError("new v6 decision failed typed production replay")
    return decision


def select_portfolio_v7_v6_test_only(
    proposal: PlannerProposalV7V6,
    observations: Sequence[v5.CandidateObservationV7V5],
    risks: Sequence[v5.ActionRiskVectorV7V5],
    calibration: v5.CalibrationReceiptV7V5,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    runtime_cost_ledgers: Sequence[v5.RuntimeCostLedgerV7V5],
    *,
    external_replay_authority: v5.ExternalInferenceReplayAuthorityV7V5,
    external_replay_verification: v5.ExternalInferenceReplayVerificationV7V5,
    audit_count: int = 0,
    child_count: int = 0,
) -> PortfolioDecisionV7V6:
    if proposal.production_registry_hash != TEST_ONLY_PRODUCTION_REGISTRY_MARKER:
        raise ValueError("test-only selector needs test-only v6 proposal")
    base = v5.select_portfolio_v7_v5_test_only(
        proposal.manifest.base_manifest,
        proposal.base_proposal,
        observations,
        risks,
        calibration,
        realizations,
        aliases,
        runtime_cost_ledgers,
        external_replay_authority=external_replay_authority,
        external_replay_verification=external_replay_verification,
        audit_count=audit_count,
        child_count=child_count,
    )
    return PortfolioDecisionV7V6(base, proposal)


# V6 retains the v5 scientific receipt semantics; only authority transport is
# new.  Aliases make that inheritance explicit without duplicating frozen code.
ActionRiskVectorV7V6 = v5.ActionRiskVectorV7V5
AliasChronologyModeV7V6 = v5.AliasChronologyModeV7V5
AliasRunPlanV7V6 = v5.AliasRunPlanV7V5
CalibrationReceiptV7V6 = v5.CalibrationReceiptV7V5
CandidateObservationV7V6 = v5.CandidateObservationV7V5
DecisionReasonV7V6 = v5.DecisionReasonV7V5
DecisionStateV7V6 = v5.DecisionStateV7V5
ExternalInferenceReplayAuthorityV7V6 = v5.ExternalInferenceReplayAuthorityV7V5
ExternalInferenceReplayVerificationV7V6 = v5.ExternalInferenceReplayVerificationV7V5
ExternalReplayAuthorityModeV7V6 = v5.ExternalReplayAuthorityModeV7V5
FiveFoldPartitionV7V6 = v5.FiveFoldPartitionV7V5
FiveFoldSplitAuthorityRootV7V6 = v5.FiveFoldSplitAuthorityRootV7V5
PlannerScoreReceiptV7V6 = v5.PlannerScoreReceiptV7V5
ProductionEligibilityV7V6 = v5.ProductionEligibilityV7V5
RuntimeCostLedgerV7V6 = v5.RuntimeCostLedgerV7V5
SplitAuthorityReceiptV7V6 = v5.SplitAuthorityReceiptV7V5


def selector_v7_v6_schema_fingerprint() -> str:
    classes = (
        SelectorV7V6Disposition,
        SR0ASplitRootAuthorityV7V6,
        TrustedSR0ASplitRegistryV7V6,
        ProductionReplayAuthorityEntryV7V6,
        TrustedProductionAuthorityRegistryV7V6,
        CandidateManifestV7V6,
        PlannerProposalV7V6,
        PortfolioDecisionV7V6,
    )
    return _sha256({
        "schema_version": SELECTOR_V7_V6_SCHEMA_VERSION,
        "canonicalization_version": SELECTOR_V7_V6_CANONICALIZATION_VERSION,
        "v5_source_sha256": SELECTOR_V7_V5_REVIEWED_SHA256,
        "v5_test_sha256": SELECTOR_V7_V5_REVIEWED_TEST_SHA256,
        "v5_fail_review_sha256": SR1_ROOT_REVIEW_V5_SHA256,
        "v5_fail_review_status": SR1_ROOT_REVIEW_V5_STATUS,
        "classes": {
            cls.__name__: list(cls.__dataclass_fields__) for cls in classes
        },
        "base_v5_schema_fingerprint": v5.selector_v7_v5_schema_fingerprint(),
    })


__all__ = [
    "ActionRiskVectorV7V6",
    "AliasChronologyModeV7V6",
    "AliasRunPlanV7V6",
    "CalibrationReceiptV7V6",
    "CandidateManifestV7V6",
    "CandidateObservationV7V6",
    "DecisionReasonV7V6",
    "DecisionStateV7V6",
    "ExternalInferenceReplayAuthorityV7V6",
    "ExternalInferenceReplayVerificationV7V6",
    "ExternalReplayAuthorityModeV7V6",
    "FiveFoldPartitionV7V6",
    "FiveFoldSplitAuthorityRootV7V6",
    "PlannerProposalV7V6",
    "PlannerScoreReceiptV7V6",
    "PortfolioDecisionV7V6",
    "ProductionEligibilityV7V6",
    "ProductionReplayAuthorityEntryV7V6",
    "RuntimeCostLedgerV7V6",
    "SELECTOR_V7_V5_REVIEWED_SHA256",
    "SELECTOR_V7_V5_REVIEWED_TEST_SHA256",
    "SELECTOR_V7_V6_CANONICALIZATION_VERSION",
    "SELECTOR_V7_V6_SCHEMA_VERSION",
    "SR0ASplitRootAuthorityV7V6",
    "SR1_ROOT_REVIEW_V5_SHA256",
    "SR1_ROOT_REVIEW_V5_STATUS",
    "SelectorV7V6Disposition",
    "SplitAuthorityReceiptV7V6",
    "TrustedProductionAuthorityRegistryV7V6",
    "TrustedSR0ASplitRegistryV7V6",
    "bind_candidate_manifest_v7_v6",
    "propose_candidates_v7_v6",
    "select_portfolio_v7_v6",
    "serialize_production_decision_v7_v6",
    "serialize_production_proposal_v7_v6",
    "selector_v7_v6_schema_fingerprint",
    "verify_manifest_split_authority_v7_v6",
    "verify_production_decision_v7_v6",
    "verify_production_proposal_v7_v6",
]
