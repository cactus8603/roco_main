"""External SelectorV7-v9 consumer and replay verifier candidate.

This file is deliberately not imported by the stablebridge package.  A trusted
scheduler invokes these exact bytes with the reviewed Python executable and
``-I -S -B``.  The JSON emitted by another process is never sufficient for
acceptance: ``verify`` independently reruns the exact worker and compares the
complete claimed envelope.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/ssd1/cactus8603/roco_main")
CLI_PATH = ROOT / "src/stablebridge/physical_repair/selector_v7_v9_cli_v2.py"
WORKER_PATH = ROOT / "src/stablebridge/physical_repair/selector_v7_v9_worker.py"
AUTHORITY_PATH = ROOT / "research/selector_v7_redesign_20261003/sr1_v9_external_runtime_authority_candidate_v2/AUTHORITY.json"
V1_AUTHORITY_PATH = ROOT / "research/selector_v7_redesign_20261003/sr1_v9_external_runtime_authority_candidate_v1/AUTHORITY.json"
MODULE_ALLOWLIST_PATH = ROOT / "research/selector_v7_redesign_20261003/sr1_v9_external_runtime_authority_candidate_v1/MODULE_ALLOWLIST.json"
PREDECESSOR_REVIEW_PATH = ROOT / "research/selector_v7_redesign_20261003/sr1_execution_auth_root_review_v1/REVIEW.json"
OUTER_CONTRACT_PATH = ROOT / "docs/research/isolated_execution_outer_boundary_contract_20261003.md"
EXECUTABLE = Path("/ssd7/cactus8603/roco_spring/.conda-stereo/bin/python3.12")
EXECUTABLE_INVOKED = "/ssd7/cactus8603/roco_spring/.conda-stereo/bin/python"
EXECUTABLE_SHA256 = "5d891882c099d02caa0eab0ce66fe74a5de4dc91ab8b7731a61a5702836b3b0a"
WORKER_SHA256 = "11ffc47f7fad5f411e411936a528cbf8b28cbca3b3e7fadcb4103dc345d7953a"
V1_AUTHORITY_SHA256 = "7353384d044d7baaebe3cb8c54da9c9f6ae8034766dd80a721676fed4aa54d1f"
MODULE_ALLOWLIST_SHA256 = "e5940969cdf430b5c12d5243875e0151cc33ddba4ec0514f4c395159f043b310"
PREDECESSOR_REVIEW_SHA256 = "c8762d145ee868e1f199f439709eb6baddbc97a8a46aec10be2268a6b93daf54"
OUTER_CONTRACT_SHA256 = "d4f3807ddad748d738cbd41b382df784bde9373512684a32d233e689490979f5"
EXPECTED_SYS_PATH = [
    "/ssd7/cactus8603/roco_spring/.conda-stereo/lib/python312.zip",
    "/ssd7/cactus8603/roco_spring/.conda-stereo/lib/python3.12",
    "/ssd7/cactus8603/roco_spring/.conda-stereo/lib/python3.12/lib-dynload",
]
CLEAN_ENV = {"LANG": "C", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
READ_BOUNDARY = {"protected_outcome": 0, "GT": 0, "H2": 0, "real_payload": 0, "GPU": False}
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


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": sha(raw)}


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_json(value: bytes) -> object:
    return json.loads(value, parse_constant=lambda token: fail(f"nonfinite JSON token: {token}"))


def verify_static() -> tuple[dict[str, object], dict[str, object]]:
    if os.environ != CLEAN_ENV:
        fail("consumer environment drift")
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode and sys.flags.ignore_environment):
        fail("consumer isolation flags drift")
    if sys.path != EXPECTED_SYS_PATH:
        fail("consumer import path drift")
    if Path.cwd().resolve(strict=True) != ROOT or ROOT.is_symlink() or CLI_PATH.is_symlink() or CLI_PATH.resolve(strict=True) != CLI_PATH:
        fail("consumer working directory/canonical path drift")
    executable = Path(sys.executable).resolve(strict=True)
    if executable != EXECUTABLE or sha(executable.read_bytes()) != EXECUTABLE_SHA256:
        fail("consumer executable drift")
    authority = parse_json(AUTHORITY_PATH.read_bytes())
    if authority.get("schema") != "selector-v7-v9-external-runtime-authority-candidate/v2" or authority.get("authority") is not False:
        fail("consumer authority schema drift")
    exact = {
        "cli": (CLI_PATH, authority.get("cli")),
        "worker": (WORKER_PATH, authority.get("worker")),
        "v1_authority": (V1_AUTHORITY_PATH, authority.get("v1_authority")),
        "module_allowlist": (MODULE_ALLOWLIST_PATH, authority.get("module_allowlist")),
        "predecessor_fail_review": (PREDECESSOR_REVIEW_PATH, authority.get("predecessor_fail_review")),
        "outer_boundary_contract": (OUTER_CONTRACT_PATH, authority.get("outer_boundary_contract")),
    }
    for label, (path, record) in exact.items():
        observed = binding(path)
        if not isinstance(record, dict) or record != {"path": str(path.relative_to(ROOT)), "bytes": observed["bytes"], "sha256": observed["sha256"]}:
            fail(f"{label} external binding drift")
    expected_hashes = {
        "worker": WORKER_SHA256,
        "v1_authority": V1_AUTHORITY_SHA256,
        "module_allowlist": MODULE_ALLOWLIST_SHA256,
        "predecessor_fail_review": PREDECESSOR_REVIEW_SHA256,
        "outer_boundary_contract": OUTER_CONTRACT_SHA256,
    }
    for label, expected in expected_hashes.items():
        if exact[label][1]["sha256"] != expected:
            fail(f"{label} reviewed hash drift")
    if authority.get("sys_path") != EXPECTED_SYS_PATH or authority.get("clean_environment") != CLEAN_ENV:
        fail("consumer runtime boundary drift")
    if authority.get("executable") != {
        "invoked_path": EXECUTABLE_INVOKED,
        "resolved_path": str(EXECUTABLE),
        "bytes": EXECUTABLE.stat().st_size,
        "sha256": EXECUTABLE_SHA256,
    }:
        fail("consumer executable authority drift")
    allowlist = parse_json(MODULE_ALLOWLIST_PATH.read_bytes())
    return authority, allowlist


def validate_request(request: object) -> dict[str, object]:
    if not isinstance(request, dict) or set(request) != {"schema", "payload", "request_hash"}:
        fail("request envelope drift")
    if request["schema"] != "selector-v7-v9-isolated-request/v1":
        fail("request schema drift")
    if request["request_hash"] != sha(canonical(request["payload"])):
        fail("request payload hash drift")
    return request


def validate_worker_response(response: object, request: dict[str, object], authority: dict[str, object], allowlist: dict[str, object]) -> dict[str, object]:
    required = {"schema", "status", "request_hash", "output", "output_hash", "source_closure", "module_closure", "runtime", "read_boundary"}
    if not isinstance(response, dict) or set(response) != required:
        fail("worker response fields drift")
    if response["schema"] != "selector-v7-v9-isolated-response/v1" or response["status"] != "PASS_ISOLATED_EXACT_S5_REPLAY":
        fail("worker response status drift")
    if response["request_hash"] != request["request_hash"] or response["output_hash"] != sha(canonical(response["output"])):
        fail("worker request/output binding drift")
    if response["source_closure"] != SOURCE_CLOSURE or response["runtime"] != authority["runtime"]:
        fail("worker source/runtime closure drift")
    if response["module_closure"] != allowlist.get("modules") or response["read_boundary"] != READ_BOUNDARY:
        fail("worker module/read closure drift")
    return response


def invoke_worker(request: dict[str, object], authority: dict[str, object], allowlist: dict[str, object]) -> tuple[dict[str, object], bytes]:
    request_bytes = canonical(request) + b"\n"
    try:
        probe = subprocess.run(
            [EXECUTABLE_INVOKED, "-I", "-S", "-B", str(WORKER_PATH)],
            input=request_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=ROOT, env=dict(CLEAN_ENV), check=False, timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        fail(f"exact isolated worker timeout: {exc.timeout}")
    if probe.returncode != 0 or probe.stderr or len(probe.stdout) > 2_000_000:
        fail("exact isolated worker execution failed")
    response = validate_worker_response(parse_json(probe.stdout), request, authority, allowlist)
    if probe.stdout != canonical(response) + b"\n":
        fail("worker output is not one canonical JSON record")
    return response, probe.stdout


def make_envelope(request: dict[str, object], response: dict[str, object], worker_stdout: bytes, authority: dict[str, object]) -> dict[str, object]:
    request_bytes = canonical(request) + b"\n"
    response_hash = sha(canonical(response))
    receipt = {
        "schema": "selector-v7-v9-external-execution-receipt/v2",
        "status": "PASS_EXACT_CLI_AND_WORKER_REPLAY_CANDIDATE",
        "request_bytes_hash": sha(request_bytes),
        "request_hash": request["request_hash"],
        "output_hash": response["output_hash"],
        "worker_response_hash": response_hash,
        "worker_stdout_bytes": len(worker_stdout),
        "worker_stdout_hash": sha(worker_stdout),
        "worker_exit_status": 0,
        "worker_invocation_count": 1,
        "cli": authority["cli"],
        "worker": authority["worker"],
        "executable": authority["executable"],
        "v1_authority": authority["v1_authority"],
        "module_allowlist": authority["module_allowlist"],
        "source_closure_hash": sha(canonical(SOURCE_CLOSURE)),
        "module_closure_hash": sha(canonical(response["module_closure"])),
        "read_boundary": READ_BOUNDARY,
    }
    receipt["receipt_hash"] = sha(canonical(receipt))
    envelope = {
        "schema": "selector-v7-v9-external-accepted-envelope/v2",
        "status": "EXACT_EXTERNAL_REPLAY_CANDIDATE_WAIT_ROOT_REVIEW",
        "authority": False,
        "request_hash": request["request_hash"],
        "output": response["output"],
        "output_hash": response["output_hash"],
        "execution_receipt": receipt,
    }
    envelope["envelope_hash"] = sha(canonical(envelope))
    return envelope


def execute(request: dict[str, object]) -> dict[str, object]:
    authority, allowlist = verify_static()
    request = validate_request(request)
    response, worker_stdout = invoke_worker(request, authority, allowlist)
    envelope = make_envelope(request, response, worker_stdout, authority)
    # Close source/runtime TOCTOU around worker execution.
    verify_static()
    return envelope


def main() -> None:
    command = parse_json(sys.stdin.buffer.read())
    if not isinstance(command, dict) or command.get("schema") != "selector-v7-v9-scheduler-command/v2":
        fail("scheduler command schema drift")
    mode = command.get("mode")
    if mode == "execute" and set(command) == {"schema", "mode", "request"}:
        output = execute(command["request"])
    elif mode == "verify" and set(command) == {"schema", "mode", "request", "claimed_envelope"}:
        expected = execute(command["request"])
        accepted = canonical(command["claimed_envelope"]) == canonical(expected)
        output = {
            "schema": "selector-v7-v9-external-verification-result/v2",
            "status": "PASS_EXACT_EXTERNAL_REPLAY" if accepted else "REJECT_CLAIM_DIFFERS_FROM_EXACT_REPLAY",
            "accepted": accepted,
            "request_hash": expected["request_hash"],
            "expected_envelope_hash": expected["envelope_hash"],
            "claimed_envelope_hash": sha(canonical(command["claimed_envelope"])),
        }
    else:
        fail("scheduler command mode/fields drift")
    sys.stdout.buffer.write(canonical(output) + b"\n")


if __name__ == "__main__":
    main()
