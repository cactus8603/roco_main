"""SelectorV7 SR1-v8 authenticated execution boundary.

This append-only successor preserves v1--v7 and closes R17 without claiming a
root seal.  Replay never resolves the mutable public v6 selector name at call
time.  It uses an import-time pinned private callable and, before every call,
verifies the exact reviewed v1--v7 source bytes, every module-level Python
callable identity and marshalled code object, dependency-module bindings, and
Python runtime/executable identity.  A typed receipt binds that authenticated
closure to the complete v7 replay input and exact v6 output.

This in-process boundary is deliberately scoped to the pure synthetic SR1
candidate.  An external deployment authority may replace it with a separately
sealed isolated runner after review.  S5 continues to reject AUDIT; S6 remains
deferred until its typed eligibility, VOI, evidence, and cost receipts exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import marshal
from pathlib import Path
import sys
import types
from typing import Sequence

from . import selector_v7 as v1
from . import selector_v7_v2 as v2
from . import selector_v7_v3 as v3
from . import selector_v7_v4 as v4
from . import selector_v7_v5 as v5
from . import selector_v7_v6 as v6
from . import selector_v7_v7 as v7


SELECTOR_V7_V8_SCHEMA_VERSION = "selector-v7-contract/v4.39-sr1-v8-execution-auth-candidate"
SELECTOR_V7_V8_CANONICALIZATION_VERSION = "selector-v7-v8-execution-auth-json/v1"
SELECTOR_V7_V7_REVIEWED_SHA256 = "945772c77b3302477a046971d7bf6532fdb9dd7ffcb56819a931eac027b9448c"
SELECTOR_V7_V7_REVIEWED_TEST_SHA256 = "b4ab812a87fba19ea6f3eaa42af539aea1c03b8fc80add332df3f12cb531dfae"
SELECTOR_V7_V7_REVIEWED_PACKAGE_SHA256 = "48fc5d8552bc474fac0b72417898762fcc04c12ad9fd04aa311d73d55dccf36c"
SELECTOR_V7_V7_SCHEMA_FINGERPRINT = "d0a734b8f8784513ed3b279c0d3f59594bf8847a8121f8713deccbaf5267c253"
SR1_ROOT_REVIEW_V7_SHA256 = "960c6f7131d55f9725b6380e73685e0f7a09e5474400423d68d94ad880f10723"
SR1_ROOT_REVIEW_V7_STATUS = "FAIL_SR1_V7_RUNTIME_SELECTOR_SUBSTITUTION_BYPASS_NO_SEAL"
SELECTOR_V7_V8_SCOPE = "S5_ONLY"
S6_AUDIT_STATUS_V7V8 = v7.S6_AUDIT_STATUS
SELECTOR_V7_V8_AUTHORITY_STATUS = "PURE_SR1_PINNED_EXECUTION_CANDIDATE_NO_SEAL"
SELECTOR_V7_V8_R17_DISPOSITION = "PINNED_CALLABLE_SOURCE_CODE_RUNTIME_CLOSURE_REQUIRED"


class SelectorExecutionAuthorityUnavailableV7V8(ValueError):
    """The exact reviewed selector execution closure is unavailable."""


def _sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def _bytes_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sealed(provided: str, payload: object, name: str) -> str:
    expected = _sha256(payload)
    if provided and provided != expected:
        raise ValueError(f"{name} does not match canonical content")
    return expected


def _rehash_valid(value: object) -> bool:
    return bool(v5._rehash_valid(value))


def _code_hash(function: object) -> str:
    code = getattr(function, "__code__", None)
    if code is None:
        raise SelectorExecutionAuthorityUnavailableV7V8(
            "execution callable has no Python code object"
        )
    def constant(value: object) -> object:
        if isinstance(value, types.CodeType):
            return code_payload(value)
        if isinstance(value, tuple):
            return ["tuple", [constant(row) for row in value]]
        if isinstance(value, frozenset):
            rows = [constant(row) for row in value]
            return ["frozenset", sorted(rows, key=lambda row: repr(row))]
        if isinstance(value, bytes):
            return ["bytes", value.hex()]
        if value is None or isinstance(value, (bool, int, float, str, complex)):
            return [type(value).__name__, repr(value)]
        return [type(value).__qualname__, repr(value)]

    def code_payload(value: types.CodeType) -> object:
        # ``marshal.dumps(code)`` includes CPython adaptive quickening state and
        # changes after a function executes.  co_code plus the structural code
        # metadata is stable while still detecting bytecode/constant changes.
        return {
            "argcount": value.co_argcount,
            "posonlyargcount": value.co_posonlyargcount,
            "kwonlyargcount": value.co_kwonlyargcount,
            "nlocals": value.co_nlocals,
            "stacksize": value.co_stacksize,
            "flags": value.co_flags,
            "code": value.co_code.hex(),
            "consts": [constant(row) for row in value.co_consts],
            "names": list(value.co_names),
            "varnames": list(value.co_varnames),
            "freevars": list(value.co_freevars),
            "cellvars": list(value.co_cellvars),
            "exceptiontable": value.co_exceptiontable.hex(),
        }

    return _sha256(code_payload(code))


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELECTOR_V7_V1_V7_SOURCE_CLOSURE = (
    ("selector_v7", "stablebridge/physical_repair/selector_v7.py",
     "3299aa0c36f1b765291b894608f6b65945dd2df81f604634ae8224e4f757aeb5"),
    ("selector_v7_v2", "stablebridge/physical_repair/selector_v7_v2.py",
     "8fbc8e3751ccb2febfaf7b8a4291be4a65ced4cfe56238c55bcf56657a1526c6"),
    ("selector_v7_v3", "stablebridge/physical_repair/selector_v7_v3.py",
     "abe48316abcfb4f1a5f3a1386d4b435323adc36198a167beb7b0043455caf6db"),
    ("selector_v7_v4", "stablebridge/physical_repair/selector_v7_v4.py",
     "fb8bb892f07f9e735646d45b489c895c31c2fbb200f1650369fd7f5ee94a155e"),
    ("selector_v7_v5", "stablebridge/physical_repair/selector_v7_v5.py",
     "3dc3b5dc5b52243216f451f8d45edf428312765e36e4827a272d73ac2c8efbb7"),
    ("selector_v7_v6", "stablebridge/physical_repair/selector_v7_v6.py",
     "8bca6471c1889f36be698cd80c47c614aeafdc5f4a69e2506c75e28bc75c503b"),
    ("selector_v7_v7", "stablebridge/physical_repair/selector_v7_v7.py",
     SELECTOR_V7_V7_REVIEWED_SHA256),
)
_TRANSITIVE_MODULES = (v1, v2, v3, v4, v5, v6, v7)


def _source_closure_now() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (name, relative, _bytes_hash((_SOURCE_ROOT / relative).read_bytes()))
        for name, relative, _ in SELECTOR_V7_V1_V7_SOURCE_CLOSURE
    )


def selector_v7_v8_source_closure_digest() -> str:
    return _sha256([list(row) for row in SELECTOR_V7_V1_V7_SOURCE_CLOSURE])


def _module_functions(module: object) -> tuple[tuple[str, str, object, str], ...]:
    module_name = str(getattr(module, "__name__", ""))
    return tuple(
        (module_name, name, value, _code_hash(value))
        for name, value in inspect.getmembers(module, inspect.isfunction)
        if getattr(value, "__module__", None) == module_name
    )


if _source_closure_now() != SELECTOR_V7_V1_V7_SOURCE_CLOSURE:
    raise ImportError("v8 refused a non-reviewed v1-v7 source closure")

_PINNED_TRANSITIVE_CALLABLES = tuple(
    row for module in _TRANSITIVE_MODULES for row in _module_functions(module)
)
_PINNED_CODE_CLOSURE = tuple(
    (module_name, name, fingerprint)
    for module_name, name, _, fingerprint in _PINNED_TRANSITIVE_CALLABLES
)
_PINNED_V6_SELECTOR_CALLABLE = v6.select_portfolio_v7_v6
_PINNED_V6_DECISION_VERIFIER = v6.verify_production_decision_v7_v6
_PINNED_MODULE_BINDINGS = (
    ("v6.v1", v6.v1, v1), ("v6.v5", v6.v5, v5),
    ("v5.v1", v5.v1, v1), ("v5.v2", v5.v2, v2),
    ("v5.v3", v5.v3, v3), ("v5.v4", v5.v4, v4),
)
_PYTHON_EXECUTABLE = str(Path(sys.executable).resolve())
_PYTHON_EXECUTABLE_HASH = _bytes_hash(Path(_PYTHON_EXECUTABLE).read_bytes())
_RUNTIME_IDENTITY = (
    sys.implementation.name,
    str(sys.implementation.cache_tag),
    tuple(sys.version_info[:3]),
    marshal.version,
    _PYTHON_EXECUTABLE_HASH,
)
_RUNTIME_IDENTITY_HASH = _sha256(list(_RUNTIME_IDENTITY))


def _decision_payload(decision: v6.PortfolioDecisionV7V6) -> dict[str, object]:
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


def _require_s5(decision: v6.PortfolioDecisionV7V6) -> None:
    state = decision.base_decision.state
    if state is v5.DecisionStateV7V5.AUDIT:
        raise ValueError("AUDIT is categorically rejected by S5; S6 remains deferred")
    if state not in {v5.DecisionStateV7V5.COMMIT, v5.DecisionStateV7V5.NATIVE}:
        raise ValueError("v8 production decision must be S5 COMMIT or NATIVE")


def _make_execution_boundary():
    pinned_selector = _PINNED_V6_SELECTOR_CALLABLE
    pinned_verifier = _PINNED_V6_DECISION_VERIFIER
    pinned_callables = _PINNED_TRANSITIVE_CALLABLES
    pinned_code_closure = _PINNED_CODE_CLOSURE
    pinned_sources = SELECTOR_V7_V1_V7_SOURCE_CLOSURE
    pinned_runtime = _RUNTIME_IDENTITY
    pinned_modules = _PINNED_MODULE_BINDINGS

    def validate() -> None:
        if _PINNED_V6_SELECTOR_CALLABLE is not pinned_selector:
            raise SelectorExecutionAuthorityUnavailableV7V8("private v6 selector binding drifted")
        if _PINNED_V6_DECISION_VERIFIER is not pinned_verifier:
            raise SelectorExecutionAuthorityUnavailableV7V8("private v6 verifier binding drifted")
        if _PINNED_TRANSITIVE_CALLABLES is not pinned_callables:
            raise SelectorExecutionAuthorityUnavailableV7V8("private callable closure binding drifted")
        if _PINNED_CODE_CLOSURE != pinned_code_closure:
            raise SelectorExecutionAuthorityUnavailableV7V8("private code closure binding drifted")
        if SELECTOR_V7_V1_V7_SOURCE_CLOSURE != pinned_sources:
            raise SelectorExecutionAuthorityUnavailableV7V8("source allowlist binding drifted")
        if _RUNTIME_IDENTITY != pinned_runtime:
            raise SelectorExecutionAuthorityUnavailableV7V8("runtime allowlist binding drifted")
        if v6.select_portfolio_v7_v6 is not pinned_selector:
            raise SelectorExecutionAuthorityUnavailableV7V8("public v6 selector symbol was replaced")
        if v6.verify_production_decision_v7_v6 is not pinned_verifier:
            raise SelectorExecutionAuthorityUnavailableV7V8("public v6 verifier symbol was replaced")
        if _source_closure_now() != pinned_sources:
            raise SelectorExecutionAuthorityUnavailableV7V8("reviewed source bytes drifted")
        if str(Path(sys.executable).resolve()) != _PYTHON_EXECUTABLE:
            raise SelectorExecutionAuthorityUnavailableV7V8("Python executable path drifted")
        if _bytes_hash(Path(_PYTHON_EXECUTABLE).read_bytes()) != _PYTHON_EXECUTABLE_HASH:
            raise SelectorExecutionAuthorityUnavailableV7V8("Python executable bytes drifted")
        runtime = (
            sys.implementation.name, str(sys.implementation.cache_tag),
            tuple(sys.version_info[:3]), marshal.version, _PYTHON_EXECUTABLE_HASH,
        )
        if runtime != pinned_runtime:
            raise SelectorExecutionAuthorityUnavailableV7V8("Python runtime identity drifted")
        for label, actual, expected in pinned_modules:
            if actual is not expected:
                raise SelectorExecutionAuthorityUnavailableV7V8(
                    f"transitive module binding drifted: {label}"
                )
        modules = {module.__name__: module for module in _TRANSITIVE_MODULES}
        for module_name, name, function, fingerprint in pinned_callables:
            if getattr(modules[module_name], name, None) is not function:
                raise SelectorExecutionAuthorityUnavailableV7V8(
                    f"transitive callable symbol drifted: {module_name}.{name}"
                )
            if _code_hash(function) != fingerprint:
                raise SelectorExecutionAuthorityUnavailableV7V8(
                    f"transitive callable code drifted: {module_name}.{name}"
                )

    def execute(bundle: v7.DecisionReplayBundleV7V7) -> v6.PortfolioDecisionV7V6:
        validate()
        decision = pinned_selector(
            bundle.proposal, bundle.observations, bundle.risks,
            bundle.calibration, bundle.realizations, bundle.aliases,
            bundle.runtime_cost_ledgers,
            production_registry=bundle.production_registry,
            trusted_production_registry_hashes=frozenset({
                bundle.production_registry.registry_hash,
            }),
            trusted_split_registry_hashes=frozenset({
                bundle.proposal.manifest.split_registry.registry_hash,
            }),
            audit_count=bundle.audit_count, child_count=bundle.child_count,
        )
        validate()
        _require_s5(decision)
        return decision

    return validate, execute


_PINNED_CLOSURE_VALIDATOR, _PINNED_EXECUTION_BOUNDARY = _make_execution_boundary()
_PINNED_CLOSURE_VALIDATOR_CODE_HASH = _code_hash(_PINNED_CLOSURE_VALIDATOR)
_PINNED_EXECUTION_BOUNDARY_CODE_HASH = _code_hash(_PINNED_EXECUTION_BOUNDARY)


def _invoke_authenticated_boundary(
    bundle: v7.DecisionReplayBundleV7V7,
    *,
    _validator: object = _PINNED_CLOSURE_VALIDATOR,
    _boundary: object = _PINNED_EXECUTION_BOUNDARY,
) -> v6.PortfolioDecisionV7V6:
    if _PINNED_CLOSURE_VALIDATOR is not _validator:
        raise SelectorExecutionAuthorityUnavailableV7V8("private validator binding was replaced")
    if _PINNED_EXECUTION_BOUNDARY is not _boundary:
        raise SelectorExecutionAuthorityUnavailableV7V8("private execution boundary was replaced")
    if _code_hash(_validator) != _PINNED_CLOSURE_VALIDATOR_CODE_HASH:
        raise SelectorExecutionAuthorityUnavailableV7V8("private validator code was replaced")
    if _code_hash(_boundary) != _PINNED_EXECUTION_BOUNDARY_CODE_HASH:
        raise SelectorExecutionAuthorityUnavailableV7V8("private execution code was replaced")
    _validator()
    return _boundary(bundle)


_PINNED_REPLAY_INVOKER = _invoke_authenticated_boundary
_PINNED_REPLAY_INVOKER_CODE_HASH = _code_hash(_PINNED_REPLAY_INVOKER)


@dataclass(frozen=True)
class SelectorV7V8Disposition:
    predecessor_review_status: str = SR1_ROOT_REVIEW_V7_STATUS
    predecessor_review_sha256: str = SR1_ROOT_REVIEW_V7_SHA256
    predecessor_source_sha256: str = SELECTOR_V7_V7_REVIEWED_SHA256
    predecessor_test_sha256: str = SELECTOR_V7_V7_REVIEWED_TEST_SHA256
    predecessor_package_sha256: str = SELECTOR_V7_V7_REVIEWED_PACKAGE_SHA256
    authority_status: str = SELECTOR_V7_V8_AUTHORITY_STATUS
    r17_disposition: str = SELECTOR_V7_V8_R17_DISPOSITION
    scope: str = SELECTOR_V7_V8_SCOPE
    audit_status: str = S6_AUDIT_STATUS_V7V8

    def __post_init__(self) -> None:
        if (
            self.predecessor_review_status != SR1_ROOT_REVIEW_V7_STATUS
            or self.predecessor_review_sha256 != SR1_ROOT_REVIEW_V7_SHA256
            or self.predecessor_source_sha256 != SELECTOR_V7_V7_REVIEWED_SHA256
            or self.predecessor_test_sha256 != SELECTOR_V7_V7_REVIEWED_TEST_SHA256
            or self.predecessor_package_sha256 != SELECTOR_V7_V7_REVIEWED_PACKAGE_SHA256
            or self.authority_status != SELECTOR_V7_V8_AUTHORITY_STATUS
            or self.r17_disposition != SELECTOR_V7_V8_R17_DISPOSITION
            or self.scope != SELECTOR_V7_V8_SCOPE
            or self.audit_status != S6_AUDIT_STATUS_V7V8
        ):
            raise ValueError("SelectorV7-v8 disposition drifted")


@dataclass(frozen=True)
class SelectorExecutionContractV7V8:
    source_closure_digest: str = selector_v7_v8_source_closure_digest()
    code_closure_digest: str = _sha256([list(row) for row in _PINNED_CODE_CLOSURE])
    runtime_identity_hash: str = _RUNTIME_IDENTITY_HASH
    python_executable_hash: str = _PYTHON_EXECUTABLE_HASH
    selector_callable_code_hash: str = _code_hash(_PINNED_V6_SELECTOR_CALLABLE)
    boundary_code_hash: str = _PINNED_EXECUTION_BOUNDARY_CODE_HASH
    authority_status: str = SELECTOR_V7_V8_AUTHORITY_STATUS
    frozen_before_replay: bool = True
    contract_hash: str = ""

    def __post_init__(self) -> None:
        _PINNED_CLOSURE_VALIDATOR()
        expected = (
            selector_v7_v8_source_closure_digest(),
            _sha256([list(row) for row in _PINNED_CODE_CLOSURE]),
            _RUNTIME_IDENTITY_HASH, _PYTHON_EXECUTABLE_HASH,
            _code_hash(_PINNED_V6_SELECTOR_CALLABLE),
            _PINNED_EXECUTION_BOUNDARY_CODE_HASH,
            SELECTOR_V7_V8_AUTHORITY_STATUS, True,
        )
        actual = (
            self.source_closure_digest, self.code_closure_digest,
            self.runtime_identity_hash, self.python_executable_hash,
            self.selector_callable_code_hash, self.boundary_code_hash,
            self.authority_status, self.frozen_before_replay,
        )
        if actual != expected:
            raise SelectorExecutionAuthorityUnavailableV7V8(
                "selector execution contract drifted"
            )
        object.__setattr__(self, "contract_hash", _sealed(
            self.contract_hash,
            {
                "source_closure_digest": self.source_closure_digest,
                "code_closure_digest": self.code_closure_digest,
                "runtime_identity_hash": self.runtime_identity_hash,
                "python_executable_hash": self.python_executable_hash,
                "selector_callable_code_hash": self.selector_callable_code_hash,
                "boundary_code_hash": self.boundary_code_hash,
                "authority_status": self.authority_status,
                "frozen_before_replay": self.frozen_before_replay,
            }, "selector execution contract hash",
        ))


@dataclass(frozen=True)
class DecisionReplayBundleV7V8:
    base_bundle: v7.DecisionReplayBundleV7V7
    execution_contract: SelectorExecutionContractV7V8
    bundle_v8_hash: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.base_bundle, v7.DecisionReplayBundleV7V7)
            or not _rehash_valid(self.base_bundle)
            or not isinstance(self.execution_contract, SelectorExecutionContractV7V8)
        ):
            raise ValueError("v8 replay bundle needs typed frozen inputs/contract")
        _PINNED_CLOSURE_VALIDATOR()
        object.__setattr__(self, "bundle_v8_hash", _sealed(
            self.bundle_v8_hash,
            {
                "schema_version": SELECTOR_V7_V8_SCHEMA_VERSION,
                "base_bundle_hash": self.base_bundle.bundle_hash,
                "execution_contract_hash": self.execution_contract.contract_hash,
                "scope": SELECTOR_V7_V8_SCOPE,
            }, "v8 replay bundle hash",
        ))


@dataclass(frozen=True)
class AuthenticatedReplayExecutionReceiptV7V8:
    bundle: DecisionReplayBundleV7V8
    replayed_decision: v6.PortfolioDecisionV7V6
    execution_contract_hash: str
    runner_input_hash: str = ""
    runner_output_hash: str = ""
    environment_hash: str = ""
    status: str = SELECTOR_V7_V8_AUTHORITY_STATUS
    receipt_hash: str = ""

    def __post_init__(
        self,
        _invoker: object = _PINNED_REPLAY_INVOKER,
    ) -> None:
        if _PINNED_REPLAY_INVOKER is not _invoker:
            raise SelectorExecutionAuthorityUnavailableV7V8("private replay invoker was replaced")
        if _code_hash(_invoker) != _PINNED_REPLAY_INVOKER_CODE_HASH:
            raise SelectorExecutionAuthorityUnavailableV7V8("private replay invoker code drifted")
        if (
            not isinstance(self.bundle, DecisionReplayBundleV7V8)
            or not isinstance(self.replayed_decision, v6.PortfolioDecisionV7V6)
            or not _rehash_valid(self.replayed_decision)
        ):
            raise ValueError("execution receipt needs typed replay artifacts")
        if self.execution_contract_hash != self.bundle.execution_contract.contract_hash:
            raise ValueError("execution receipt contract binding drifted")
        if self.status != SELECTOR_V7_V8_AUTHORITY_STATUS:
            raise ValueError("execution receipt authority status drifted")
        expected = _invoker(self.bundle.base_bundle)
        if expected != self.replayed_decision:
            raise ValueError("decision differs from authenticated selector replay")
        input_hash = _sha256({
            "bundle_v8_hash": self.bundle.bundle_v8_hash,
            "base_bundle_hash": self.bundle.base_bundle.bundle_hash,
        })
        output_hash = _sha256(_decision_payload(expected))
        environment_hash = _sha256({
            "runtime_identity_hash": _RUNTIME_IDENTITY_HASH,
            "source_closure_digest": selector_v7_v8_source_closure_digest(),
            "code_closure_digest": _sha256([list(row) for row in _PINNED_CODE_CLOSURE]),
            "boundary_code_hash": _PINNED_EXECUTION_BOUNDARY_CODE_HASH,
        })
        for supplied, wanted, label in (
            (self.runner_input_hash, input_hash, "runner input"),
            (self.runner_output_hash, output_hash, "runner output"),
            (self.environment_hash, environment_hash, "runner environment"),
        ):
            if supplied not in {"", wanted}:
                raise ValueError(f"{label} hash drifted")
        object.__setattr__(self, "runner_input_hash", input_hash)
        object.__setattr__(self, "runner_output_hash", output_hash)
        object.__setattr__(self, "environment_hash", environment_hash)
        object.__setattr__(self, "receipt_hash", _sealed(
            self.receipt_hash,
            {
                "bundle_v8_hash": self.bundle.bundle_v8_hash,
                "execution_contract_hash": self.execution_contract_hash,
                "runner_input_hash": input_hash,
                "runner_output_hash": output_hash,
                "environment_hash": environment_hash,
                "decision_v6_hash": expected.decision_v6_hash,
                "status": self.status,
            }, "authenticated execution receipt hash",
        ))


@dataclass(frozen=True)
class PortfolioDecisionV7V8:
    base_decision: v6.PortfolioDecisionV7V6
    replay_bundle: DecisionReplayBundleV7V8
    execution_receipt: AuthenticatedReplayExecutionReceiptV7V8
    scope: str = SELECTOR_V7_V8_SCOPE
    audit_status: str = S6_AUDIT_STATUS_V7V8
    decision_v8_hash: str = ""

    def __post_init__(self, _invoker: object = _PINNED_REPLAY_INVOKER) -> None:
        if (
            _PINNED_REPLAY_INVOKER is not _invoker
            or _code_hash(_invoker) != _PINNED_REPLAY_INVOKER_CODE_HASH
        ):
            raise SelectorExecutionAuthorityUnavailableV7V8(
                "private decision replay invoker drifted"
            )
        if (
            not isinstance(self.base_decision, v6.PortfolioDecisionV7V6)
            or not _rehash_valid(self.base_decision)
            or not isinstance(self.replay_bundle, DecisionReplayBundleV7V8)
            or not isinstance(self.execution_receipt, AuthenticatedReplayExecutionReceiptV7V8)
        ):
            raise ValueError("v8 decision needs typed execution artifacts")
        _require_s5(self.base_decision)
        if self.scope != SELECTOR_V7_V8_SCOPE or self.audit_status != S6_AUDIT_STATUS_V7V8:
            raise ValueError("v8 decision scope drifted")
        if (
            self.execution_receipt.bundle != self.replay_bundle
            or self.execution_receipt.replayed_decision != self.base_decision
        ):
            raise ValueError("v8 decision differs from execution receipt")
        expected = _invoker(self.replay_bundle.base_bundle)
        if expected != self.base_decision:
            raise ValueError("v8 decision differs from authenticated selector replay")
        object.__setattr__(self, "decision_v8_hash", _sealed(
            self.decision_v8_hash,
            {
                "schema_version": SELECTOR_V7_V8_SCHEMA_VERSION,
                "decision_v6_hash": expected.decision_v6_hash,
                "base_decision_hash": expected.base_decision.decision_hash,
                "bundle_v8_hash": self.replay_bundle.bundle_v8_hash,
                "execution_receipt_hash": self.execution_receipt.receipt_hash,
                "selected_candidate_id": expected.base_decision.selected_candidate_id,
                "safe_candidate_ids": list(expected.base_decision.safe_candidate_ids),
                "reason": expected.base_decision.reason.value,
                "scope": self.scope, "audit_status": self.audit_status,
            }, "v8 decision hash",
        ))


def make_decision_replay_bundle_v7_v8(
    proposal: v6.PlannerProposalV7V6,
    observations: Sequence[v5.CandidateObservationV7V5],
    risks: Sequence[v5.ActionRiskVectorV7V5],
    calibration: v5.CalibrationReceiptV7V5,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    runtime_cost_ledgers: Sequence[v5.RuntimeCostLedgerV7V5],
    *, production_registry: v6.TrustedProductionAuthorityRegistryV7V6,
    audit_count: int = 0, child_count: int = 0,
) -> DecisionReplayBundleV7V8:
    base = v7.make_decision_replay_bundle_v7_v7(
        proposal, observations, risks, calibration, realizations, aliases,
        runtime_cost_ledgers, production_registry=production_registry,
        audit_count=audit_count, child_count=child_count,
    )
    return DecisionReplayBundleV7V8(base, SelectorExecutionContractV7V8())


def select_portfolio_v7_v8(
    proposal: v6.PlannerProposalV7V6,
    observations: Sequence[v5.CandidateObservationV7V5],
    risks: Sequence[v5.ActionRiskVectorV7V5],
    calibration: v5.CalibrationReceiptV7V5,
    realizations: Sequence[v1.ArmRealizationReceiptV7],
    aliases: Sequence[v1.ArmAliasReceiptV7],
    runtime_cost_ledgers: Sequence[v5.RuntimeCostLedgerV7V5],
    *, production_registry: v6.TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
    audit_count: int = 0, child_count: int = 0,
    _invoker: object = _PINNED_REPLAY_INVOKER,
) -> PortfolioDecisionV7V8:
    if (
        _PINNED_REPLAY_INVOKER is not _invoker
        or _code_hash(_invoker) != _PINNED_REPLAY_INVOKER_CODE_HASH
    ):
        raise SelectorExecutionAuthorityUnavailableV7V8(
            "private selector replay invoker drifted"
        )
    if (
        production_registry.registry_hash not in trusted_production_registry_hashes
        or proposal.manifest.split_registry.registry_hash not in trusted_split_registry_hashes
    ):
        raise ValueError("v8 selector lacks external production/split trust")
    bundle = make_decision_replay_bundle_v7_v8(
        proposal, observations, risks, calibration, realizations, aliases,
        runtime_cost_ledgers, production_registry=production_registry,
        audit_count=audit_count, child_count=child_count,
    )
    decision = _invoker(bundle.base_bundle)
    receipt = AuthenticatedReplayExecutionReceiptV7V8(
        bundle, decision, bundle.execution_contract.contract_hash,
    )
    result = PortfolioDecisionV7V8(decision, bundle, receipt)
    if not verify_production_decision_v7_v8(
        result, production_registry=production_registry,
        trusted_production_registry_hashes=trusted_production_registry_hashes,
        trusted_split_registry_hashes=trusted_split_registry_hashes,
    ):
        raise ValueError("new v8 decision failed authenticated replay")
    return result


def _verify_impl(
    decision: object,
    *, production_registry: v6.TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
    _invoker: object = _PINNED_REPLAY_INVOKER,
) -> bool:
    try:
        if (
            _PINNED_REPLAY_INVOKER is not _invoker
            or _code_hash(_invoker) != _PINNED_REPLAY_INVOKER_CODE_HASH
        ):
            return False
        if not isinstance(decision, PortfolioDecisionV7V8):
            return False
        if decision.base_decision.base_decision.state is v5.DecisionStateV7V5.AUDIT:
            return False
        bundle = decision.replay_bundle
        if (
            bundle.base_bundle.production_registry != production_registry
            or production_registry.registry_hash not in trusted_production_registry_hashes
            or bundle.base_bundle.proposal.manifest.split_registry.registry_hash
            not in trusted_split_registry_hashes
        ):
            return False
        expected = _invoker(bundle.base_bundle)
        if expected != decision.base_decision:
            return False
        receipt = decision.execution_receipt
        if (
            receipt.bundle != bundle
            or receipt.replayed_decision != expected
            or receipt.execution_contract_hash != bundle.execution_contract.contract_hash
            or receipt.runner_output_hash != _sha256(_decision_payload(expected))
            or receipt.status != SELECTOR_V7_V8_AUTHORITY_STATUS
        ):
            return False
        return bool(_PINNED_V6_DECISION_VERIFIER(
            expected, production_registry=production_registry,
            trusted_production_registry_hashes=trusted_production_registry_hashes,
            trusted_split_registry_hashes=trusted_split_registry_hashes,
        ))
    except Exception:
        return False


_PINNED_VERIFY_IMPL = _verify_impl
_PINNED_VERIFY_IMPL_CODE_HASH = _code_hash(_PINNED_VERIFY_IMPL)


def verify_production_decision_v7_v8(
    decision: object,
    *, production_registry: v6.TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
    _verifier: object = _PINNED_VERIFY_IMPL,
) -> bool:
    if _PINNED_VERIFY_IMPL is not _verifier or _code_hash(_verifier) != _PINNED_VERIFY_IMPL_CODE_HASH:
        return False
    return bool(_verifier(
        decision, production_registry=production_registry,
        trusted_production_registry_hashes=trusted_production_registry_hashes,
        trusted_split_registry_hashes=trusted_split_registry_hashes,
    ))


def serialize_production_decision_v7_v8(
    decision: object,
    *, production_registry: v6.TrustedProductionAuthorityRegistryV7V6,
    trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str],
    _verifier: object = _PINNED_VERIFY_IMPL,
) -> bytes:
    if (
        _PINNED_VERIFY_IMPL is not _verifier
        or _code_hash(_verifier) != _PINNED_VERIFY_IMPL_CODE_HASH
        or not _verifier(
        decision, production_registry=production_registry,
        trusted_production_registry_hashes=trusted_production_registry_hashes,
        trusted_split_registry_hashes=trusted_split_registry_hashes,
        )
    ):
        raise ValueError("decision is not an authenticated exact S5 replay")
    assert isinstance(decision, PortfolioDecisionV7V8)
    payload = _decision_payload(decision.base_decision)
    payload.update({
        "schema_version": SELECTOR_V7_V8_SCHEMA_VERSION,
        "decision_v8_hash": decision.decision_v8_hash,
        "bundle_v8_hash": decision.replay_bundle.bundle_v8_hash,
        "execution_receipt_hash": decision.execution_receipt.receipt_hash,
        "execution_contract_hash": decision.replay_bundle.execution_contract.contract_hash,
        "runner_input_hash": decision.execution_receipt.runner_input_hash,
        "runner_output_hash": decision.execution_receipt.runner_output_hash,
        "environment_hash": decision.execution_receipt.environment_hash,
        "authority_status": decision.execution_receipt.status,
        "scope": decision.scope, "audit_status": decision.audit_status,
    })
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


# Exact predecessor transport aliases.
ActionRiskVectorV7V8 = v7.ActionRiskVectorV7V7
AliasChronologyModeV7V8 = v7.AliasChronologyModeV7V7
AliasRunPlanV7V8 = v7.AliasRunPlanV7V7
CalibrationReceiptV7V8 = v7.CalibrationReceiptV7V7
CandidateManifestV7V8 = v7.CandidateManifestV7V7
CandidateObservationV7V8 = v7.CandidateObservationV7V7
DecisionReasonV7V8 = v7.DecisionReasonV7V7
DecisionStateV7V8 = v7.DecisionStateV7V7
ExternalInferenceReplayAuthorityV7V8 = v7.ExternalInferenceReplayAuthorityV7V7
ExternalInferenceReplayVerificationV7V8 = v7.ExternalInferenceReplayVerificationV7V7
ExternalReplayAuthorityModeV7V8 = v7.ExternalReplayAuthorityModeV7V7
FiveFoldPartitionV7V8 = v7.FiveFoldPartitionV7V7
FiveFoldSplitAuthorityRootV7V8 = v7.FiveFoldSplitAuthorityRootV7V7
PlannerProposalV7V8 = v7.PlannerProposalV7V7
PlannerScoreReceiptV7V8 = v7.PlannerScoreReceiptV7V7
ProductionEligibilityV7V8 = v7.ProductionEligibilityV7V7
ProductionReplayAuthorityEntryV7V8 = v7.ProductionReplayAuthorityEntryV7V7
RuntimeCostLedgerV7V8 = v7.RuntimeCostLedgerV7V7
SR0ASplitRootAuthorityV7V8 = v7.SR0ASplitRootAuthorityV7V7
SplitAuthorityReceiptV7V8 = v7.SplitAuthorityReceiptV7V7
TrustedProductionAuthorityRegistryV7V8 = v7.TrustedProductionAuthorityRegistryV7V7
TrustedSR0ASplitRegistryV7V8 = v7.TrustedSR0ASplitRegistryV7V7
bind_candidate_manifest_v7_v8 = v7.bind_candidate_manifest_v7_v7
propose_candidates_v7_v8 = v7.propose_candidates_v7_v7
serialize_production_proposal_v7_v8 = v7.serialize_production_proposal_v7_v7
verify_manifest_split_authority_v7_v8 = v7.verify_manifest_split_authority_v7_v7
verify_production_proposal_v7_v8 = v7.verify_production_proposal_v7_v7


def selector_v7_v8_schema_fingerprint() -> str:
    return _sha256({
        "schema_version": SELECTOR_V7_V8_SCHEMA_VERSION,
        "canonicalization_version": SELECTOR_V7_V8_CANONICALIZATION_VERSION,
        "v7_source": SELECTOR_V7_V7_REVIEWED_SHA256,
        "v7_test": SELECTOR_V7_V7_REVIEWED_TEST_SHA256,
        "v7_package": SELECTOR_V7_V7_REVIEWED_PACKAGE_SHA256,
        "v7_review": SR1_ROOT_REVIEW_V7_SHA256,
        "source_closure": selector_v7_v8_source_closure_digest(),
        "code_closure": _sha256([list(row) for row in _PINNED_CODE_CLOSURE]),
        "runtime": _RUNTIME_IDENTITY_HASH,
        "classes": {
            cls.__name__: list(cls.__dataclass_fields__)
            for cls in (
                SelectorV7V8Disposition, SelectorExecutionContractV7V8,
                DecisionReplayBundleV7V8,
                AuthenticatedReplayExecutionReceiptV7V8,
                PortfolioDecisionV7V8,
            )
        },
    })


__all__ = [
    "ActionRiskVectorV7V8", "AliasChronologyModeV7V8", "AliasRunPlanV7V8",
    "AuthenticatedReplayExecutionReceiptV7V8", "CalibrationReceiptV7V8",
    "CandidateManifestV7V8", "CandidateObservationV7V8", "DecisionReasonV7V8",
    "DecisionReplayBundleV7V8", "DecisionStateV7V8",
    "ExternalInferenceReplayAuthorityV7V8",
    "ExternalInferenceReplayVerificationV7V8", "ExternalReplayAuthorityModeV7V8",
    "FiveFoldPartitionV7V8", "FiveFoldSplitAuthorityRootV7V8",
    "PlannerProposalV7V8", "PlannerScoreReceiptV7V8", "PortfolioDecisionV7V8",
    "ProductionEligibilityV7V8", "ProductionReplayAuthorityEntryV7V8",
    "RuntimeCostLedgerV7V8", "S6_AUDIT_STATUS_V7V8",
    "SELECTOR_V7_V1_V7_SOURCE_CLOSURE", "SELECTOR_V7_V7_REVIEWED_PACKAGE_SHA256",
    "SELECTOR_V7_V7_REVIEWED_SHA256", "SELECTOR_V7_V7_REVIEWED_TEST_SHA256",
    "SELECTOR_V7_V8_AUTHORITY_STATUS", "SELECTOR_V7_V8_CANONICALIZATION_VERSION",
    "SELECTOR_V7_V8_R17_DISPOSITION", "SELECTOR_V7_V8_SCHEMA_VERSION",
    "SELECTOR_V7_V8_SCOPE", "SR0ASplitRootAuthorityV7V8",
    "SR1_ROOT_REVIEW_V7_SHA256", "SR1_ROOT_REVIEW_V7_STATUS",
    "SelectorExecutionAuthorityUnavailableV7V8", "SelectorExecutionContractV7V8",
    "SelectorV7V8Disposition", "SplitAuthorityReceiptV7V8",
    "TrustedProductionAuthorityRegistryV7V8", "TrustedSR0ASplitRegistryV7V8",
    "bind_candidate_manifest_v7_v8", "make_decision_replay_bundle_v7_v8",
    "propose_candidates_v7_v8", "select_portfolio_v7_v8",
    "selector_v7_v8_schema_fingerprint", "selector_v7_v8_source_closure_digest",
    "serialize_production_decision_v7_v8", "serialize_production_proposal_v7_v8",
    "verify_manifest_split_authority_v7_v8", "verify_production_decision_v7_v8",
    "verify_production_proposal_v7_v8",
]
