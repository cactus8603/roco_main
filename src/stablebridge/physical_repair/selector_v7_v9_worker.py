"""Isolated SelectorV7-v9 worker; invoked only with ``-I -S -B``.

The worker reads and authenticates source bytes before compiling any selector
module.  It creates a fresh lightweight package namespace, so caller-preloaded
objects and mutations cannot enter the execution process.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types


ROOT = Path("/ssd1/cactus8603/roco_main")
SOURCE_DIR = ROOT / "src/stablebridge/physical_repair"
AUTHORITY_PATH = ROOT / "research/selector_v7_redesign_20261003/sr1_v9_external_runtime_authority_candidate_v1/AUTHORITY.json"
MODULE_ALLOWLIST_PATH = ROOT / "research/selector_v7_redesign_20261003/sr1_v9_external_runtime_authority_candidate_v1/MODULE_ALLOWLIST.json"
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
EXECUTED_MODULES = tuple(SOURCE_CLOSURE)[:7]
EXPECTED_ENV = {"LANG": "C", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
EXPECTED_EXECUTABLE = Path("/ssd7/cactus8603/roco_spring/.conda-stereo/bin/python3.12")
EXPECTED_EXECUTABLE_SHA256 = "5d891882c099d02caa0eab0ce66fe74a5de4dc91ab8b7731a61a5702836b3b0a"
EXPECTED_REVIEW_SHA256 = "c8762d145ee868e1f199f439709eb6baddbc97a8a46aec10be2268a6b93daf54"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": sha256_bytes(raw)}


def fail(message: str) -> None:
    raise RuntimeError(message)


def verify_preload() -> tuple[dict[str, bytes], dict[str, object], dict[str, object]]:
    if os.environ != EXPECTED_ENV:
        fail("worker environment drift")
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode and sys.flags.ignore_environment):
        fail("worker isolation flags drift")
    executable = Path(sys.executable).resolve(strict=True)
    if executable != EXPECTED_EXECUTABLE or sha256_bytes(executable.read_bytes()) != EXPECTED_EXECUTABLE_SHA256:
        fail("worker executable drift")
    runtime = {
        "implementation": sys.implementation.name,
        "cache_tag": str(sys.implementation.cache_tag),
        "version": list(sys.version_info[:3]),
        "marshal_version": __import__("marshal").version,
    }
    if runtime != {"implementation": "cpython", "cache_tag": "cpython-312", "version": [3, 12, 14], "marshal_version": 4}:
        fail("worker runtime drift")
    authority = json.loads(AUTHORITY_PATH.read_text())
    if authority.get("predecessor_fail_review", {}).get("sha256") != EXPECTED_REVIEW_SHA256:
        fail("runtime authority predecessor drift")
    for record in (authority["external_reviewed_runtime"]["recipe_amendment"], authority["external_reviewed_runtime"]["root_seal"], authority["predecessor_fail_review"]):
        observed = binding(ROOT / str(record["path"]))
        if observed["bytes"] != record["bytes"] or observed["sha256"] != record["sha256"]:
            fail("external runtime authority binding drift")
    worker_record = authority.get("worker")
    current_worker = binding(Path(__file__).resolve())
    if not isinstance(worker_record, dict) or current_worker["bytes"] != worker_record.get("bytes") or current_worker["sha256"] != worker_record.get("sha256") or worker_record.get("path") != "src/stablebridge/physical_repair/selector_v7_v9_worker.py":
        fail("worker self-binding drift")
    if authority["executable"] != {"bytes": executable.stat().st_size, "invoked_path": "/ssd7/cactus8603/roco_spring/.conda-stereo/bin/python", "resolved_path": str(executable), "sha256": EXPECTED_EXECUTABLE_SHA256}:
        fail("external executable allowlist drift")
    sources: dict[str, bytes] = {}
    for name, expected in SOURCE_CLOSURE.items():
        raw = (SOURCE_DIR / f"{name}.py").read_bytes()
        if sha256_bytes(raw) != expected:
            fail(f"selector source drift before load: {name}")
        sources[name] = raw
    allowlist = json.loads(MODULE_ALLOWLIST_PATH.read_text()) if MODULE_ALLOWLIST_PATH.exists() else {"bootstrap": True}
    if allowlist.get("schema") == "selector-v7-v9-module-allowlist/v1":
        for record in allowlist.get("modules", []):
            if "path" not in record:
                if record.get("origin") not in {"BUILTIN_OR_FROZEN", "SYNTHETIC_PACKAGE"}:
                    fail("preload module allowlist schema drift")
                continue
            path = Path(record["path"]).resolve(strict=True)
            observed = binding(path)
            if observed["bytes"] != record.get("bytes") or observed["sha256"] != record.get("sha256"):
                fail(f"preload module binding drift: {record.get('module')}")
            if not (path == SOURCE_DIR or SOURCE_DIR in path.parents or Path("/ssd7/cactus8603/roco_spring/.conda-stereo/lib/python3.12") in path.parents):
                fail("preload module origin outside allowlist")
    return sources, authority, allowlist


def load_fresh(sources: dict[str, bytes]) -> dict[str, types.ModuleType]:
    stablebridge = types.ModuleType("stablebridge")
    stablebridge.__path__ = [str(ROOT / "src/stablebridge")]
    physical = types.ModuleType("stablebridge.physical_repair")
    physical.__path__ = [str(SOURCE_DIR)]
    physical.__package__ = "stablebridge.physical_repair"
    stablebridge.physical_repair = physical
    sys.modules["stablebridge"] = stablebridge
    sys.modules["stablebridge.physical_repair"] = physical
    loaded = {}
    for short in EXECUTED_MODULES:
        full = f"stablebridge.physical_repair.{short}"
        module = types.ModuleType(full)
        module.__file__ = str(SOURCE_DIR / f"{short}.py")
        module.__package__ = "stablebridge.physical_repair"
        sys.modules[full] = module
        setattr(physical, short, module)
        exec(compile(sources[short], module.__file__, "exec", dont_inherit=True), module.__dict__)
        loaded[short] = module
    return loaded


def resolve_type(modules: dict[str, types.ModuleType], module_name: str, qualname: str) -> object:
    prefix = "stablebridge.physical_repair."
    if not module_name.startswith(prefix):
        fail("typed input module is not allowlisted")
    short = module_name[len(prefix):]
    if short not in modules:
        fail("typed input selector version is not loaded")
    value: object = modules[short]
    for part in qualname.split("."):
        if part == "<locals>":
            fail("local typed input class rejected")
        value = getattr(value, part)
    return value


def decode(value: object, modules: dict[str, types.ModuleType]) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            fail("nonfinite protocol float")
        return value
    if not isinstance(value, dict) or set(value) != {"type", "value"}:
        fail("invalid typed protocol node")
    kind, payload = value["type"], value["value"]
    if kind in {"tuple", "list", "frozenset"}:
        rows = [decode(row, modules) for row in payload]
        return tuple(rows) if kind == "tuple" else (frozenset(rows) if kind == "frozenset" else rows)
    if kind == "dict":
        return {decode(row[0], modules): decode(row[1], modules) for row in payload}
    if kind in {"enum", "dataclass"}:
        if set(payload) != ({"module", "qualname", "value"} if kind == "enum" else {"module", "qualname", "fields"}):
            fail("typed object schema drift")
        cls = resolve_type(modules, payload["module"], payload["qualname"])
        if kind == "enum":
            return cls(payload["value"])
        fields = payload["fields"]
        if not isinstance(fields, list) or any(not isinstance(row, list) or len(row) != 2 for row in fields):
            fail("dataclass field schema drift")
        return cls(**{name: decode(item, modules) for name, item in fields})
    fail("unknown typed protocol node")


def module_closure() -> list[dict[str, object]]:
    result = []
    for name, module in sorted(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if raw is None:
            origin = "SYNTHETIC_PACKAGE" if name in {"stablebridge", "stablebridge.physical_repair"} else "BUILTIN_OR_FROZEN"
            result.append({"module": name, "origin": origin})
            continue
        path = Path(raw).resolve(strict=True)
        allowed = path == SOURCE_DIR or SOURCE_DIR in path.parents or Path("/ssd7/cactus8603/roco_spring/.conda-stereo/lib/python3.12") in path.parents
        if not allowed:
            fail(f"module origin outside allowlist: {name}:{path}")
        result.append({"module": name, **binding(path)})
    return result


def decision_payload(decision: object, v5: types.ModuleType) -> dict[str, object]:
    base = decision.base_decision
    if base.state is v5.DecisionStateV7V5.AUDIT:
        fail("AUDIT is categorically rejected by S5")
    if base.state not in {v5.DecisionStateV7V5.COMMIT, v5.DecisionStateV7V5.NATIVE}:
        fail("non-S5 decision rejected")
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


def main() -> None:
    sources, authority, expected_modules = verify_preload()
    request_raw = sys.stdin.buffer.read()
    request = json.loads(request_raw)
    if set(request) != {"schema", "payload", "request_hash"} or request["schema"] != "selector-v7-v9-isolated-request/v1":
        fail("request envelope drift")
    if request["request_hash"] != sha256_bytes(canonical_bytes(request["payload"])):
        fail("request hash drift")
    modules = load_fresh(sources)
    payload = decode(request["payload"], modules)
    if not isinstance(payload, dict) or set(payload) != {"proposal", "observations", "risks", "calibration", "realizations", "aliases", "runtime_cost_ledgers", "production_registry", "trusted_production_registry_hashes", "trusted_split_registry_hashes", "audit_count", "child_count"}:
        fail("decoded request field drift")
    v5, v6 = modules["selector_v7_v5"], modules["selector_v7_v6"]
    decision = v6.select_portfolio_v7_v6(
        payload["proposal"], payload["observations"], payload["risks"], payload["calibration"],
        payload["realizations"], payload["aliases"], payload["runtime_cost_ledgers"],
        production_registry=payload["production_registry"],
        trusted_production_registry_hashes=payload["trusted_production_registry_hashes"],
        trusted_split_registry_hashes=payload["trusted_split_registry_hashes"],
        audit_count=payload["audit_count"], child_count=payload["child_count"],
    )
    output = decision_payload(decision, v5)
    for name, expected in SOURCE_CLOSURE.items():
        if sha256_bytes((SOURCE_DIR / f"{name}.py").read_bytes()) != expected:
            fail(f"selector source drift after execution: {name}")
    closure = module_closure()
    if expected_modules.get("schema") == "selector-v7-v9-module-allowlist/v1" and closure != expected_modules["modules"]:
        fail("exact imported-module closure drift")
    response = {
        "schema": "selector-v7-v9-isolated-response/v1",
        "status": "PASS_ISOLATED_EXACT_S5_REPLAY",
        "request_hash": request["request_hash"],
        "output": output,
        "output_hash": sha256_bytes(canonical_bytes(output)),
        "source_closure": SOURCE_CLOSURE,
        "module_closure": closure,
        "runtime": authority["runtime"],
        "read_boundary": {"protected_outcome": 0, "GT": 0, "H2": 0, "real_payload": 0, "GPU": False},
    }
    sys.stdout.buffer.write(canonical_bytes(response) + b"\n")


if __name__ == "__main__":
    main()
