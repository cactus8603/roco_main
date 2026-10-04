"""Append-only SelectorV7-v9 isolated execution candidate.

This module is intentionally not exported by ``physical_repair.__init__``.
The public v8 production entrypoint remains fail closed until these exact v9
bytes receive a fresh independent root review.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types
from typing import Sequence


ROOT = Path("/ssd1/cactus8603/roco_main")
WORKER_PATH = ROOT / "src/stablebridge/physical_repair/selector_v7_v9_worker.py"
AUTHORITY_PATH = ROOT / "research/selector_v7_redesign_20261003/sr1_v9_external_runtime_authority_candidate_v1/AUTHORITY.json"
MODULE_ALLOWLIST_PATH = ROOT / "research/selector_v7_redesign_20261003/sr1_v9_external_runtime_authority_candidate_v1/MODULE_ALLOWLIST.json"
PREDECESSOR_REVIEW_PATH = ROOT / "research/selector_v7_redesign_20261003/sr1_execution_auth_root_review_v1/REVIEW.json"
PREDECESSOR_REVIEW_SHA256 = "c8762d145ee868e1f199f439709eb6baddbc97a8a46aec10be2268a6b93daf54"
RUNTIME_AUTHORITY_SHA256 = "7353384d044d7baaebe3cb8c54da9c9f6ae8034766dd80a721676fed4aa54d1f"
MODULE_ALLOWLIST_SHA256 = "e5940969cdf430b5c12d5243875e0151cc33ddba4ec0514f4c395159f043b310"
WORKER_SHA256 = "11ffc47f7fad5f411e411936a528cbf8b28cbca3b3e7fadcb4103dc345d7953a"
EXECUTABLE = "/ssd7/cactus8603/roco_spring/.conda-stereo/bin/python"
EXECUTABLE_SHA256 = "5d891882c099d02caa0eab0ce66fe74a5de4dc91ab8b7731a61a5702836b3b0a"
COMMAND_FLAGS = ("-I", "-S", "-B")
CLEAN_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
SELECTOR_V7_V9_SCHEMA_VERSION = "selector-v7-contract/v4.39-sr1-v9-isolated-candidate"
SELECTOR_V7_V9_SCOPE = "S5_ONLY"
S6_AUDIT_STATUS_V7V9 = "DEFERRED_UNTIL_TYPED_AUDIT_ELIGIBILITY_VOI_AND_COST_RECEIPT"
SELECTOR_V7_V9_STATUS = "PURE_SR1_ISOLATED_EXECUTION_CANDIDATE_NO_SEAL"
ALLOWED_TYPE_MODULES = frozenset(f"stablebridge.physical_repair.{name}" for name in (
    "selector_v7", "selector_v7_v2", "selector_v7_v3", "selector_v7_v4",
    "selector_v7_v5", "selector_v7_v6", "selector_v7_v7",
))
SOURCE_CLOSURE = {
    "selector_v7": "3299aa0c36f1b765291b894608f6b65945dd2df81f604634ae8224e4f757aeb5",
    "selector_v7_v2": "8fbc8e3751ccb2febfaf7b8a4291be4a65ced4cfe56238c55bcf56657a1526c6",
    "selector_v7_v3": "abe48316abcfb4f1a5f3a1386d4b435323adc36198a167beb7b0043455caf6db",
    "selector_v7_v4": "fb8bb892f07f9e735646d45b489c895c31c2fbb200f1650369fd7f5ee94a155e",
    "selector_v7_v5": "3dc3b5dc5b52243216f451f8d45edf428312765e36e4827a272d73ac2c8efbb7",
    "selector_v7_v6": "8bca6471c1889f36be698cd80c47c614aeafdc5f4a69e2506c75e28bc75c503b",
    "selector_v7_v7": "945772c77b3302477a046971d7bf6532fdb9dd7ffcb56819a931eac027b9448c",
    "selector_v7_v8": "df12cb11bb53069d750ed0e6dcbe09c4298093aeb1d653877bad16fd77074354",
    "selector_v7_v8_execution_auth": "59ccee43ed8e4093c35df0f590f316d545280ea00ba7c45507ecd2400a7b9fca",
}


class SelectorV7V9IsolationError(ValueError):
    pass


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": _sha_bytes(raw)}


def _function_fingerprint(function: object) -> str:
    code = getattr(function, "__code__", None)
    if not isinstance(code, types.CodeType):
        raise SelectorV7V9IsolationError("launcher callable lacks Python code")

    def constant(value: object) -> object:
        if isinstance(value, types.CodeType):
            return code_value(value)
        if isinstance(value, tuple):
            return ["tuple", [constant(row) for row in value]]
        if isinstance(value, frozenset):
            return ["frozenset", sorted((constant(row) for row in value), key=repr)]
        if isinstance(value, bytes):
            return ["bytes", value.hex()]
        if value is None or isinstance(value, (bool, int, float, str, complex)):
            return [type(value).__name__, repr(value)]
        return [type(value).__qualname__, repr(value)]

    def code_value(value: types.CodeType) -> object:
        return {
            "argcount": value.co_argcount, "posonly": value.co_posonlyargcount,
            "kwonly": value.co_kwonlyargcount, "nlocals": value.co_nlocals,
            "stacksize": value.co_stacksize, "flags": value.co_flags,
            "code": value.co_code.hex(), "consts": [constant(row) for row in value.co_consts],
            "names": list(value.co_names), "varnames": list(value.co_varnames),
            "freevars": list(value.co_freevars), "cellvars": list(value.co_cellvars),
            "exceptiontable": value.co_exceptiontable.hex(),
        }

    closure = getattr(function, "__closure__", None) or ()
    payload = {
        "code": code_value(code),
        "defaults": constant(getattr(function, "__defaults__", None)),
        "kwdefaults": [[key, constant(item)] for key, item in sorted((getattr(function, "__kwdefaults__", None) or {}).items())],
        "closure": [constant(cell.cell_contents) for cell in closure],
    }
    return _sha_bytes(_canonical(payload))


def _encode(value: object) -> object:
    if isinstance(value, Enum):
        cls = type(value)
        if cls.__module__ not in ALLOWED_TYPE_MODULES:
            raise ValueError("enum module is outside selector allowlist")
        return {"type": "enum", "value": {"module": cls.__module__, "qualname": cls.__qualname__, "value": object.__getattribute__(value, "_value_")}}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite protocol float")
        return value
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [_encode(row) for row in value]}
    if isinstance(value, list):
        return {"type": "list", "value": [_encode(row) for row in value]}
    if isinstance(value, frozenset):
        rows = [_encode(row) for row in value]
        return {"type": "frozenset", "value": sorted(rows, key=lambda row: _canonical(row))}
    if isinstance(value, dict):
        rows = [[_encode(key), _encode(item)] for key, item in value.items()]
        return {"type": "dict", "value": sorted(rows, key=lambda row: _canonical(row[0]))}
    cls = type(value)
    if cls.__module__ not in ALLOWED_TYPE_MODULES:
        raise ValueError(f"typed input class outside selector allowlist: {cls.__module__}.{cls.__qualname__}")
    try:
        fields = object.__getattribute__(value, "__dict__")
    except AttributeError as exc:
        raise ValueError("typed input must expose frozen dataclass fields") from exc
    if not isinstance(fields, dict) or not fields:
        raise ValueError("typed input dataclass fields absent")
    return {"type": "dataclass", "value": {"module": cls.__module__, "qualname": cls.__qualname__, "fields": [[name, _encode(item)] for name, item in sorted(fields.items())]}}


def _verify_static_closure() -> tuple[dict[str, object], dict[str, object]]:
    if _sha_bytes(PREDECESSOR_REVIEW_PATH.read_bytes()) != PREDECESSOR_REVIEW_SHA256:
        raise SelectorV7V9IsolationError("predecessor FAIL review drift")
    for path, expected, label in (
        (AUTHORITY_PATH, RUNTIME_AUTHORITY_SHA256, "runtime authority"),
        (MODULE_ALLOWLIST_PATH, MODULE_ALLOWLIST_SHA256, "module allowlist"),
        (WORKER_PATH, WORKER_SHA256, "worker"),
    ):
        if _sha_bytes(path.read_bytes()) != expected:
            raise SelectorV7V9IsolationError(f"{label} drift")
    executable = Path(EXECUTABLE).resolve(strict=True)
    if _sha_bytes(executable.read_bytes()) != EXECUTABLE_SHA256:
        raise SelectorV7V9IsolationError("reviewed executable drift")
    source_dir = ROOT / "src/stablebridge/physical_repair"
    for name, expected in SOURCE_CLOSURE.items():
        if _sha_bytes((source_dir / f"{name}.py").read_bytes()) != expected:
            raise SelectorV7V9IsolationError(f"reviewed selector source drift: {name}")
    authority = json.loads(AUTHORITY_PATH.read_text())
    allowlist = json.loads(MODULE_ALLOWLIST_PATH.read_text())
    if authority.get("authority") is not False or authority.get("scope") != "SYNTHETIC_SR1_S5_ONLY":
        raise SelectorV7V9IsolationError("runtime authority scope drift")
    return authority, allowlist


def _request_bytes(payload: dict[str, object]) -> bytes:
    encoded = _encode(payload)
    envelope = {"schema": "selector-v7-v9-isolated-request/v1", "payload": encoded, "request_hash": _sha_bytes(_canonical(encoded))}
    return _canonical(envelope) + b"\n"


def _validate_module_closure(records: object, allowlist: dict[str, object]) -> None:
    if records != allowlist.get("modules"):
        raise SelectorV7V9IsolationError("worker module closure differs from frozen allowlist")
    for record in records:
        origin = record.get("origin")
        if origin in {"BUILTIN_OR_FROZEN", "SYNTHETIC_PACKAGE"}:
            if set(record) != {"module", "origin"}:
                raise SelectorV7V9IsolationError("non-file module record drift")
            continue
        path = Path(record["path"])
        if _binding(path) != {"path": str(path), "bytes": record["bytes"], "sha256": record["sha256"]}:
            raise SelectorV7V9IsolationError("worker imported module bytes drift")


def _launch(request_bytes: bytes) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    authority, allowlist = _verify_static_closure()
    if (
        subprocess.run is not _PINNED_SUBPROCESS_RUN
        or _function_fingerprint(_PINNED_SUBPROCESS_RUN) != _PINNED_SUBPROCESS_RUN_FINGERPRINT
        or subprocess.Popen is not _PINNED_POPEN
        or any(getattr(subprocess.Popen, name, None) is not function or _function_fingerprint(function) != fingerprint for name, function, fingerprint in _PINNED_POPEN_METHODS)
        or (subprocess.PIPE, subprocess.STDOUT, subprocess.DEVNULL) != (-1, -2, -3)
    ):
        raise SelectorV7V9IsolationError("launcher transport callable drift")
    command = [EXECUTABLE, *COMMAND_FLAGS, str(WORKER_PATH)]
    probe = subprocess.run(command, input=request_bytes, capture_output=True, cwd=ROOT, env=dict(CLEAN_ENVIRONMENT), check=False)
    if probe.returncode != 0 or probe.stderr or len(probe.stdout) > 2_000_000:
        raise SelectorV7V9IsolationError(f"isolated worker failed closed: {probe.stderr[-500:]!r}")
    response = json.loads(probe.stdout)
    required = {"schema", "status", "request_hash", "output", "output_hash", "source_closure", "module_closure", "runtime", "read_boundary"}
    request = json.loads(request_bytes)
    if set(response) != required or response["schema"] != "selector-v7-v9-isolated-response/v1" or response["status"] != "PASS_ISOLATED_EXACT_S5_REPLAY":
        raise SelectorV7V9IsolationError("isolated response schema/status drift")
    if response["request_hash"] != request["request_hash"] or response["output_hash"] != _sha_bytes(_canonical(response["output"])):
        raise SelectorV7V9IsolationError("isolated response input/output hash drift")
    if response["source_closure"] != SOURCE_CLOSURE or response["runtime"] != authority["runtime"]:
        raise SelectorV7V9IsolationError("isolated source/runtime receipt drift")
    if response["read_boundary"] != {"protected_outcome": 0, "GT": 0, "H2": 0, "real_payload": 0, "GPU": False}:
        raise SelectorV7V9IsolationError("isolated read boundary drift")
    _validate_module_closure(response["module_closure"], allowlist)
    return response, authority, allowlist


_PINNED_SUBPROCESS_RUN = subprocess.run
_PINNED_SUBPROCESS_RUN_FINGERPRINT = _function_fingerprint(_PINNED_SUBPROCESS_RUN)
_PINNED_POPEN = subprocess.Popen
_PINNED_POPEN_METHODS = tuple(
    (name, getattr(_PINNED_POPEN, name), _function_fingerprint(getattr(_PINNED_POPEN, name)))
    for name in ("__init__", "__enter__", "__exit__", "communicate", "poll", "wait")
)
_PINNED_LAUNCH = _launch
_PINNED_LAUNCH_FINGERPRINT = _function_fingerprint(_PINNED_LAUNCH)
_EXPECTED_TRANSPORT_FINGERPRINTS = {
    "run": "8df62b9f587667164d0c7ce8bb985e25bca0f53c5f3ed08c6f3a44b898ad2a68",
    "__init__": "4feecb62f9cd981b7426934b218b74beeb717e80db0120c541f1abab3cc94bad",
    "__enter__": "1ff7e16b7e393b03781acd30080178ad5bf63aa4295b81bd00cd78aed6c0c7f6",
    "__exit__": "ef5892dbb7b7c5b1b8ef4be66a5bd254c5ec5952a4e3fc5a843659c30bf4e98d",
    "communicate": "08b206f78aab0d82c5af6248911dcb8f3724b1b6b2c6857a4ecdcc946be70181",
    "poll": "4f1671a4de1870ca8f6249a9098408e58c7b0da1d80b13109f1358bdb5221f30",
    "wait": "4e72f81303a525e03f6484c98ef444acf09bb74dced1c447ef78e02057b81dc0",
    "launch": "54f321d80c5e99929f2d73fac2f6fdc437308d2cff7f89b2fb7b28632a7637ac",
}
if (
    Path(sys.executable).resolve() != Path("/ssd7/cactus8603/roco_spring/.conda-stereo/bin/python3.12")
    or _sha_bytes(Path(sys.executable).resolve().read_bytes()) != EXECUTABLE_SHA256
    or _binding(Path(subprocess.__file__).resolve())["sha256"] != "85d29b2bf0249f5436838298c9a60ee93508b1102e9ac43b001f8a7e7ae8f375"
    or _PINNED_SUBPROCESS_RUN_FINGERPRINT != _EXPECTED_TRANSPORT_FINGERPRINTS["run"]
    or {name: fingerprint for name, _, fingerprint in _PINNED_POPEN_METHODS}
       != {name: value for name, value in _EXPECTED_TRANSPORT_FINGERPRINTS.items() if name not in {"run", "launch"}}
    or _PINNED_LAUNCH_FINGERPRINT != _EXPECTED_TRANSPORT_FINGERPRINTS["launch"]
):
    raise ImportError("v9 launcher refused a non-reviewed parent runtime/transport closure")


@dataclass(frozen=True)
class IsolatedReplayExecutionReceiptV7V9:
    request_hash: str
    output_hash: str
    response_hash: str
    worker_binding: dict[str, object]
    executable_binding: dict[str, object]
    runtime_authority_binding: dict[str, object]
    module_allowlist_binding: dict[str, object]
    module_closure_digest: str
    source_closure_digest: str
    status: str = "PASS_ISOLATED_EXACT_S5_REPLAY"
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        payload = {name: value for name, value in object.__getattribute__(self, "__dict__").items() if name != "receipt_hash"}
        wanted = _sha_bytes(_canonical(payload))
        if self.receipt_hash not in {"", wanted}:
            raise ValueError("isolated execution receipt hash drift")
        object.__setattr__(self, "receipt_hash", wanted)


@dataclass(frozen=True)
class PortfolioDecisionV7V9:
    request_bytes: bytes
    decision_payload: dict[str, object]
    execution_receipt: IsolatedReplayExecutionReceiptV7V9
    scope: str = SELECTOR_V7_V9_SCOPE
    audit_status: str = S6_AUDIT_STATUS_V7V9
    decision_v9_hash: str = ""

    def __post_init__(self) -> None:
        if self.scope != SELECTOR_V7_V9_SCOPE or self.audit_status != S6_AUDIT_STATUS_V7V9:
            raise ValueError("v9 decision scope drift")
        if self.decision_payload.get("state") == "audit":
            raise ValueError("AUDIT is categorically rejected by S5")
        payload = {"request_hash": self.execution_receipt.request_hash, "output_hash": self.execution_receipt.output_hash, "receipt_hash": self.execution_receipt.receipt_hash, "scope": self.scope, "audit_status": self.audit_status}
        wanted = _sha_bytes(_canonical(payload))
        if self.decision_v9_hash not in {"", wanted}:
            raise ValueError("v9 decision hash drift")
        object.__setattr__(self, "decision_v9_hash", wanted)


def _receipt(request_bytes: bytes, response: dict[str, object], authority: dict[str, object], allowlist: dict[str, object]) -> IsolatedReplayExecutionReceiptV7V9:
    return IsolatedReplayExecutionReceiptV7V9(
        request_hash=response["request_hash"], output_hash=response["output_hash"], response_hash=_sha_bytes(_canonical(response)),
        worker_binding=_binding(WORKER_PATH), executable_binding=_binding(Path(EXECUTABLE).resolve()),
        runtime_authority_binding=_binding(AUTHORITY_PATH), module_allowlist_binding=_binding(MODULE_ALLOWLIST_PATH),
        module_closure_digest=_sha_bytes(_canonical(response["module_closure"])), source_closure_digest=_sha_bytes(_canonical(SOURCE_CLOSURE)),
    )


def select_portfolio_v7_v9(
    proposal: object, observations: Sequence[object], risks: Sequence[object], calibration: object,
    realizations: Sequence[object], aliases: Sequence[object], runtime_cost_ledgers: Sequence[object],
    *, production_registry: object, trusted_production_registry_hashes: frozenset[str],
    trusted_split_registry_hashes: frozenset[str], audit_count: int = 0, child_count: int = 0,
    _launcher: object = _PINNED_LAUNCH,
) -> PortfolioDecisionV7V9:
    if _launch is not _PINNED_LAUNCH or _launcher is not _PINNED_LAUNCH or _function_fingerprint(_launcher) != _PINNED_LAUNCH_FINGERPRINT:
        raise SelectorV7V9IsolationError("isolated launcher binding drift")
    payload = {"proposal": proposal, "observations": tuple(observations), "risks": tuple(risks), "calibration": calibration, "realizations": tuple(realizations), "aliases": tuple(aliases), "runtime_cost_ledgers": tuple(runtime_cost_ledgers), "production_registry": production_registry, "trusted_production_registry_hashes": trusted_production_registry_hashes, "trusted_split_registry_hashes": trusted_split_registry_hashes, "audit_count": audit_count, "child_count": child_count}
    request_bytes = _request_bytes(payload)
    response, authority, allowlist = _launcher(request_bytes)
    receipt = _receipt(request_bytes, response, authority, allowlist)
    decision = PortfolioDecisionV7V9(request_bytes, response["output"], receipt)
    if not verify_production_decision_v7_v9(decision):
        raise SelectorV7V9IsolationError("new v9 decision failed isolated exact replay")
    return decision


def verify_production_decision_v7_v9(decision: object, *, _launcher: object = _PINNED_LAUNCH) -> bool:
    try:
        if _launch is not _PINNED_LAUNCH or _launcher is not _PINNED_LAUNCH or _function_fingerprint(_launcher) != _PINNED_LAUNCH_FINGERPRINT:
            return False
        if not isinstance(decision, PortfolioDecisionV7V9) or decision.decision_payload.get("state") == "audit":
            return False
        response, authority, allowlist = _launcher(decision.request_bytes)
        receipt = _receipt(decision.request_bytes, response, authority, allowlist)
        return response["output"] == decision.decision_payload and receipt == decision.execution_receipt and PortfolioDecisionV7V9(decision.request_bytes, response["output"], receipt) == decision
    except Exception:
        return False


def serialize_production_decision_v7_v9(decision: object) -> bytes:
    if not verify_production_decision_v7_v9(decision):
        raise ValueError("decision is not an isolated exact S5 replay")
    assert isinstance(decision, PortfolioDecisionV7V9)
    payload = dict(decision.decision_payload)
    payload.update({"schema_version": SELECTOR_V7_V9_SCHEMA_VERSION, "decision_v9_hash": decision.decision_v9_hash, "execution_receipt_hash": decision.execution_receipt.receipt_hash, "authority_status": SELECTOR_V7_V9_STATUS, "scope": decision.scope, "audit_status": decision.audit_status})
    return _canonical(payload)


__all__ = [
    "IsolatedReplayExecutionReceiptV7V9", "PortfolioDecisionV7V9",
    "S6_AUDIT_STATUS_V7V9", "SELECTOR_V7_V9_SCHEMA_VERSION",
    "SELECTOR_V7_V9_SCOPE", "SELECTOR_V7_V9_STATUS",
    "SelectorV7V9IsolationError", "select_portfolio_v7_v9",
    "serialize_production_decision_v7_v9", "verify_production_decision_v7_v9",
]
