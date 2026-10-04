"""Append-only SelectorV7 SR1-v8 fail-closed execution-boundary successor.

The v7 independent review proved that an in-process replay can execute a
substituted selector while declaring the reviewed source hash.  Closing that
finding requires an isolated runner with a complete implementation, runtime,
import and native-library closure, plus an execution receipt authenticated by
an authority outside the caller process.  No such independently reviewed
runner authority exists yet.

V8 therefore exposes no production decision positive path.  Selection and
serialization fail closed and verification always returns false.  A later
append-only successor may open the gate only after it binds the external
runner authority described by :class:`SelectorExecutionContractV7V8`.
This module does not claim R17 closed and grants no authority seal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import NoReturn

from . import selector_v7_v7 as v7


SELECTOR_V7_V8_SCHEMA_VERSION = "selector-v7-contract/v4.39-sr1-v8-preexecution"
SELECTOR_V7_V8_CANONICALIZATION_VERSION = "selector-v7-v8-canonical-json/v1"
SELECTOR_V7_V8_SCOPE = "S5_ONLY_PRODUCTION_BLOCKED"
SELECTOR_V7_V8_AUTHORITY_STATUS = (
    "BLOCKED_MISSING_EXTERNALLY_AUTHENTICATED_ISOLATED_RUNNER"
)
SELECTOR_V7_V8_R17_DISPOSITION = "OPEN_FAIL_CLOSED_NO_PRODUCTION_POSITIVE_PATH"
S6_AUDIT_STATUS_V7V8 = (
    "DEFERRED_UNTIL_TYPED_AUDIT_ELIGIBILITY_VOI_AND_COST_RECEIPT"
)

SELECTOR_V7_V7_REVIEWED_SHA256 = (
    "945772c77b3302477a046971d7bf6532fdb9dd7ffcb56819a931eac027b9448c"
)
SELECTOR_V7_V7_REVIEWED_TEST_SHA256 = (
    "b4ab812a87fba19ea6f3eaa42af539aea1c03b8fc80add332df3f12cb531dfae"
)
SELECTOR_V7_V7_REVIEWED_PACKAGE_SHA256 = (
    "48fc5d8552bc474fac0b72417898762fcc04c12ad9fd04aa311d73d55dccf36c"
)
SR1_ROOT_REVIEW_V7_SHA256 = (
    "960c6f7131d55f9725b6380e73685e0f7a09e5474400423d68d94ad880f10723"
)
SR1_ROOT_REVIEW_V7_STATUS = (
    "FAIL_SR1_V7_RUNTIME_SELECTOR_SUBSTITUTION_BYPASS_NO_SEAL"
)

# Planning bindings only.  A future runner must rehash these before importing
# anything, then also require exact set equality for its Python/stdlib/
# third-party/extension/native closure.  This tuple is not an attestation.
SELECTOR_V7_V1_V7_SOURCE_CLOSURE = (
    ("src/stablebridge/physical_repair/selector_v7.py", "3299aa0c36f1b765291b894608f6b65945dd2df81f604634ae8224e4f757aeb5"),
    ("src/stablebridge/physical_repair/selector_v7_v2.py", "8fbc8e3751ccb2febfaf7b8a4291be4a65ced4cfe56238c55bcf56657a1526c6"),
    ("src/stablebridge/physical_repair/selector_v7_v3.py", "abe48316abcfb4f1a5f3a1386d4b435323adc36198a167beb7b0043455caf6db"),
    ("src/stablebridge/physical_repair/selector_v7_v4.py", "fb8bb892f07f9e735646d45b489c895c31c2fbb200f1650369fd7f5ee94a155e"),
    ("src/stablebridge/physical_repair/selector_v7_v5.py", "3dc3b5dc5b52243216f451f8d45edf428312765e36e4827a272d73ac2c8efbb7"),
    ("src/stablebridge/physical_repair/selector_v7_v6.py", "8bca6471c1889f36be698cd80c47c614aeafdc5f4a69e2506c75e28bc75c503b"),
    ("src/stablebridge/physical_repair/selector_v7_v7.py", SELECTOR_V7_V7_REVIEWED_SHA256),
)


def _sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def selector_v7_v8_source_closure_digest() -> str:
    """Return the planning closure digest, not execution authority."""
    return _sha256([list(row) for row in SELECTOR_V7_V1_V7_SOURCE_CLOSURE])


class SelectorExecutionAuthorityUnavailableV7V8(RuntimeError):
    """The required external execution authority does not yet exist."""


@dataclass(frozen=True)
class SelectorExecutionContractV7V8:
    """Minimum acceptance contract for a future external replay runner."""

    process_mode: str = "ISOLATED_EXTERNAL_PROCESS"
    python_flags: tuple[str, ...] = ("-I", "-S")
    environment_mode: str = "ALLOWLIST_ONLY_NO_CALLER_ENV_COPY"
    implementation_scope: str = "COMPLETE_TRANSITIVE_SELECTOR_V1_V6"
    runtime_scope: str = "PYTHON_STDLIB_THIRD_PARTY_EXTENSION_NATIVE_LIBRARIES"
    import_policy: str = "EXACT_PATH_BYTES_SHA256_SET_EQUALITY_BEFORE_LOAD"
    request_codec: str = "TYPED_CANONICAL_NON_EXECUTABLE_NO_PICKLE"
    mutable_import_policy: str = "REJECT"
    foreign_import_policy: str = "REJECT"
    execution_receipt_anchor: str = "EXTERNAL_TO_CALLER_PROCESS"
    external_registry_policy: str = "INDEPENDENTLY_REVIEWED_ROOT_NOT_CALLER_MINTED"
    production_positive_path: bool = False
    authority_status: str = SELECTOR_V7_V8_AUTHORITY_STATUS
    contract_hash: str = ""

    def __post_init__(self) -> None:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "contract_hash"
        }
        expected = _sha256(payload)
        if (
            self.process_mode != "ISOLATED_EXTERNAL_PROCESS"
            or self.python_flags != ("-I", "-S")
            or self.environment_mode != "ALLOWLIST_ONLY_NO_CALLER_ENV_COPY"
            or self.implementation_scope != "COMPLETE_TRANSITIVE_SELECTOR_V1_V6"
            or self.runtime_scope
            != "PYTHON_STDLIB_THIRD_PARTY_EXTENSION_NATIVE_LIBRARIES"
            or self.import_policy
            != "EXACT_PATH_BYTES_SHA256_SET_EQUALITY_BEFORE_LOAD"
            or self.request_codec != "TYPED_CANONICAL_NON_EXECUTABLE_NO_PICKLE"
            or self.mutable_import_policy != "REJECT"
            or self.foreign_import_policy != "REJECT"
            or self.execution_receipt_anchor != "EXTERNAL_TO_CALLER_PROCESS"
            or self.external_registry_policy
            != "INDEPENDENTLY_REVIEWED_ROOT_NOT_CALLER_MINTED"
            or self.production_positive_path is not False
            or self.authority_status != SELECTOR_V7_V8_AUTHORITY_STATUS
            or self.contract_hash not in {"", expected}
        ):
            raise ValueError("SelectorV7-v8 execution contract drifted")
        object.__setattr__(self, "contract_hash", expected)


@dataclass(frozen=True)
class SelectorV7V8Disposition:
    predecessor_review_status: str = SR1_ROOT_REVIEW_V7_STATUS
    predecessor_review_sha256: str = SR1_ROOT_REVIEW_V7_SHA256
    predecessor_source_sha256: str = SELECTOR_V7_V7_REVIEWED_SHA256
    predecessor_test_sha256: str = SELECTOR_V7_V7_REVIEWED_TEST_SHA256
    predecessor_package_sha256: str = SELECTOR_V7_V7_REVIEWED_PACKAGE_SHA256
    scope: str = SELECTOR_V7_V8_SCOPE
    audit_status: str = S6_AUDIT_STATUS_V7V8
    r17_disposition: str = SELECTOR_V7_V8_R17_DISPOSITION
    authority_status: str = SELECTOR_V7_V8_AUTHORITY_STATUS
    successor_status: str = "PREEXECUTION_BLOCKED_WAIT_EXTERNAL_RUNNER_AUTHORITY"

    def __post_init__(self) -> None:
        if (
            self.predecessor_review_status != SR1_ROOT_REVIEW_V7_STATUS
            or self.predecessor_review_sha256 != SR1_ROOT_REVIEW_V7_SHA256
            or self.predecessor_source_sha256 != SELECTOR_V7_V7_REVIEWED_SHA256
            or self.predecessor_test_sha256 != SELECTOR_V7_V7_REVIEWED_TEST_SHA256
            or self.predecessor_package_sha256
            != SELECTOR_V7_V7_REVIEWED_PACKAGE_SHA256
            or self.scope != SELECTOR_V7_V8_SCOPE
            or self.audit_status != S6_AUDIT_STATUS_V7V8
            or self.r17_disposition != SELECTOR_V7_V8_R17_DISPOSITION
            or self.authority_status != SELECTOR_V7_V8_AUTHORITY_STATUS
            or self.successor_status
            != "PREEXECUTION_BLOCKED_WAIT_EXTERNAL_RUNNER_AUTHORITY"
        ):
            raise ValueError("SelectorV7-v8 predecessor disposition drifted")


@dataclass(frozen=True)
class PortfolioDecisionV7V8:
    """Reserved future type; construction is impossible in this successor."""

    predecessor_decision: v7.PortfolioDecisionV7V7
    external_execution_receipt_hash: str
    decision_v8_hash: str = ""

    def __post_init__(self) -> NoReturn:
        raise SelectorExecutionAuthorityUnavailableV7V8(
            SELECTOR_V7_V8_AUTHORITY_STATUS
        )


def select_portfolio_v7_v8(*args: object, **kwargs: object) -> NoReturn:
    """Reject before invoking any mutable in-process selector callable."""
    del args, kwargs
    raise SelectorExecutionAuthorityUnavailableV7V8(
        SELECTOR_V7_V8_AUTHORITY_STATUS
    )


def verify_production_decision_v7_v8(
    decision: object,
    **authority_inputs: object,
) -> bool:
    """No v8 artifact is production-verifiable without external authority."""
    del decision, authority_inputs
    return False


def serialize_production_decision_v7_v8(
    decision: object,
    **authority_inputs: object,
) -> NoReturn:
    """Never serialize a decision from an unauthenticated execution boundary."""
    del decision, authority_inputs
    raise SelectorExecutionAuthorityUnavailableV7V8(
        SELECTOR_V7_V8_AUTHORITY_STATUS
    )


# Proposal/manifest construction remains the reviewed predecessor surface.  It
# cannot cross the closed decision gate and grants no production authority.
ActionRiskVectorV7V8 = v7.ActionRiskVectorV7V7
AliasChronologyModeV7V8 = v7.AliasChronologyModeV7V7
AliasRunPlanV7V8 = v7.AliasRunPlanV7V7
CalibrationReceiptV7V8 = v7.CalibrationReceiptV7V7
CandidateManifestV7V8 = v7.CandidateManifestV7V7
CandidateObservationV7V8 = v7.CandidateObservationV7V7
DecisionReasonV7V8 = v7.DecisionReasonV7V7
DecisionStateV7V8 = v7.DecisionStateV7V7
ExternalInferenceReplayAuthorityV7V8 = v7.ExternalInferenceReplayAuthorityV7V7
ExternalInferenceReplayVerificationV7V8 = (
    v7.ExternalInferenceReplayVerificationV7V7
)
ExternalReplayAuthorityModeV7V8 = v7.ExternalReplayAuthorityModeV7V7
FiveFoldPartitionV7V8 = v7.FiveFoldPartitionV7V7
FiveFoldSplitAuthorityRootV7V8 = v7.FiveFoldSplitAuthorityRootV7V7
PlannerProposalV7V8 = v7.PlannerProposalV7V7
PlannerScoreReceiptV7V8 = v7.PlannerScoreReceiptV7V7
ProductionEligibilityV7V8 = v7.ProductionEligibilityV7V7
ProductionReplayAuthorityEntryV7V8 = v7.ProductionReplayAuthorityEntryV7V7
RuntimeCostLedgerV7V8 = v7.RuntimeCostLedgerV7V7
SplitAuthorityReceiptV7V8 = v7.SplitAuthorityReceiptV7V7
SR0ASplitRootAuthorityV7V8 = v7.SR0ASplitRootAuthorityV7V7
TrustedProductionAuthorityRegistryV7V8 = v7.TrustedProductionAuthorityRegistryV7V7
TrustedSR0ASplitRegistryV7V8 = v7.TrustedSR0ASplitRegistryV7V7
bind_candidate_manifest_v7_v8 = v7.bind_candidate_manifest_v7_v7
propose_candidates_v7_v8 = v7.propose_candidates_v7_v7
serialize_production_proposal_v7_v8 = v7.serialize_production_proposal_v7_v7
verify_manifest_split_authority_v7_v8 = v7.verify_manifest_split_authority_v7_v7
verify_production_proposal_v7_v8 = v7.verify_production_proposal_v7_v7


def selector_v7_v8_schema_fingerprint() -> str:
    classes = (
        SelectorExecutionContractV7V8,
        SelectorV7V8Disposition,
        PortfolioDecisionV7V8,
    )
    return _sha256({
        "schema_version": SELECTOR_V7_V8_SCHEMA_VERSION,
        "canonicalization_version": SELECTOR_V7_V8_CANONICALIZATION_VERSION,
        "v7_source_sha256": SELECTOR_V7_V7_REVIEWED_SHA256,
        "v7_test_sha256": SELECTOR_V7_V7_REVIEWED_TEST_SHA256,
        "v7_package_sha256": SELECTOR_V7_V7_REVIEWED_PACKAGE_SHA256,
        "v7_fail_review_sha256": SR1_ROOT_REVIEW_V7_SHA256,
        "v7_fail_review_status": SR1_ROOT_REVIEW_V7_STATUS,
        "source_closure_digest": selector_v7_v8_source_closure_digest(),
        "scope": SELECTOR_V7_V8_SCOPE,
        "audit_status": S6_AUDIT_STATUS_V7V8,
        "r17_disposition": SELECTOR_V7_V8_R17_DISPOSITION,
        "authority_status": SELECTOR_V7_V8_AUTHORITY_STATUS,
        "classes": {
            cls.__name__: list(cls.__dataclass_fields__) for cls in classes
        },
    })


__all__ = [
    "ActionRiskVectorV7V8",
    "AliasChronologyModeV7V8",
    "AliasRunPlanV7V8",
    "CalibrationReceiptV7V8",
    "CandidateManifestV7V8",
    "CandidateObservationV7V8",
    "DecisionReasonV7V8",
    "DecisionStateV7V8",
    "ExternalInferenceReplayAuthorityV7V8",
    "ExternalInferenceReplayVerificationV7V8",
    "ExternalReplayAuthorityModeV7V8",
    "FiveFoldPartitionV7V8",
    "FiveFoldSplitAuthorityRootV7V8",
    "PlannerProposalV7V8",
    "PlannerScoreReceiptV7V8",
    "PortfolioDecisionV7V8",
    "ProductionEligibilityV7V8",
    "ProductionReplayAuthorityEntryV7V8",
    "RuntimeCostLedgerV7V8",
    "S6_AUDIT_STATUS_V7V8",
    "SELECTOR_V7_V1_V7_SOURCE_CLOSURE",
    "SELECTOR_V7_V7_REVIEWED_PACKAGE_SHA256",
    "SELECTOR_V7_V7_REVIEWED_SHA256",
    "SELECTOR_V7_V7_REVIEWED_TEST_SHA256",
    "SELECTOR_V7_V8_AUTHORITY_STATUS",
    "SELECTOR_V7_V8_CANONICALIZATION_VERSION",
    "SELECTOR_V7_V8_R17_DISPOSITION",
    "SELECTOR_V7_V8_SCHEMA_VERSION",
    "SELECTOR_V7_V8_SCOPE",
    "SR0ASplitRootAuthorityV7V8",
    "SR1_ROOT_REVIEW_V7_SHA256",
    "SR1_ROOT_REVIEW_V7_STATUS",
    "SelectorExecutionAuthorityUnavailableV7V8",
    "SelectorExecutionContractV7V8",
    "SelectorV7V8Disposition",
    "SplitAuthorityReceiptV7V8",
    "TrustedProductionAuthorityRegistryV7V8",
    "TrustedSR0ASplitRegistryV7V8",
    "bind_candidate_manifest_v7_v8",
    "propose_candidates_v7_v8",
    "select_portfolio_v7_v8",
    "selector_v7_v8_schema_fingerprint",
    "selector_v7_v8_source_closure_digest",
    "serialize_production_decision_v7_v8",
    "serialize_production_proposal_v7_v8",
    "verify_manifest_split_authority_v7_v8",
    "verify_production_decision_v7_v8",
    "verify_production_proposal_v7_v8",
]
