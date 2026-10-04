"""Pinned outer acceptance gate for the SelectorV7-v9 CLI candidate.

The integration boundary is this separately launched program, never an
imported Python wrapper or a caller-supplied JSON dictionary.  ``execute``
launches the exact CLI once to produce an envelope and again in verification
mode to replay it.  ``verify`` reconstructs the complete scheduler result and
requires canonical byte equality.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/ssd1/cactus8603/roco_main")
SELF = ROOT / "src/stablebridge/physical_repair/selector_v7_v9_scheduler_v2.py"
CLI = ROOT / "src/stablebridge/physical_repair/selector_v7_v9_cli_v2.py"
AUTHORITY = ROOT / "research/selector_v7_redesign_20261003/sr1_v9_external_runtime_authority_candidate_v2/AUTHORITY.json"
EXECUTABLE = Path("/ssd7/cactus8603/roco_spring/.conda-stereo/bin/python3.12")
EXECUTABLE_INVOKED = "/ssd7/cactus8603/roco_spring/.conda-stereo/bin/python"
EXECUTABLE_SHA256 = "5d891882c099d02caa0eab0ce66fe74a5de4dc91ab8b7731a61a5702836b3b0a"
EXPECTED_SYS_PATH = [
    "/ssd7/cactus8603/roco_spring/.conda-stereo/lib/python312.zip",
    "/ssd7/cactus8603/roco_spring/.conda-stereo/lib/python3.12",
    "/ssd7/cactus8603/roco_spring/.conda-stereo/lib/python3.12/lib-dynload",
]
CLEAN_ENV = {"LANG": "C", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": sha(raw)}


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_json(value: bytes) -> object:
    return json.loads(value, parse_constant=lambda token: fail(f"nonfinite JSON token: {token}"))


def verify_boundary() -> dict[str, object]:
    if os.environ != CLEAN_ENV:
        fail("scheduler environment drift")
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode and sys.flags.ignore_environment):
        fail("scheduler isolation flags drift")
    if sys.path != EXPECTED_SYS_PATH:
        fail("scheduler import path drift")
    if Path.cwd().resolve(strict=True) != ROOT or ROOT.is_symlink() or SELF.is_symlink() or SELF.resolve(strict=True) != SELF:
        fail("scheduler working directory/canonical path drift")
    executable = Path(sys.executable).resolve(strict=True)
    if executable != EXECUTABLE or sha(executable.read_bytes()) != EXECUTABLE_SHA256:
        fail("scheduler executable drift")
    authority = parse_json(AUTHORITY.read_bytes())
    if authority.get("schema") != "selector-v7-v9-external-runtime-authority-candidate/v2" or authority.get("authority") is not False:
        fail("scheduler authority drift")
    for label, path in (("scheduler", SELF), ("cli", CLI)):
        if authority.get(label) != binding(path):
            fail(f"scheduler {label} binding drift")
    return authority


def invoke_cli(command: dict[str, object]) -> tuple[dict[str, object], bytes]:
    try:
        probe = subprocess.run(
            [EXECUTABLE_INVOKED, "-I", "-S", "-B", str(CLI)], input=canonical(command),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT, env=dict(CLEAN_ENV), check=False, timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        fail(f"pinned CLI timeout: {exc.timeout}")
    if probe.returncode != 0 or probe.stderr or len(probe.stdout) > 2_000_000:
        fail("pinned CLI execution failed")
    value = parse_json(probe.stdout)
    if not isinstance(value, dict):
        fail("pinned CLI returned non-object")
    if probe.stdout != canonical(value) + b"\n":
        fail("pinned CLI output is not one canonical JSON record")
    return value, probe.stdout


def execute(request: object, authority: dict[str, object]) -> dict[str, object]:
    command = {"schema": "selector-v7-v9-scheduler-command/v2", "mode": "execute", "request": request}
    envelope, execute_stdout = invoke_cli(command)
    verifier_command = {
        "schema": "selector-v7-v9-scheduler-command/v2", "mode": "verify",
        "request": request, "claimed_envelope": envelope,
    }
    verification, verify_stdout = invoke_cli(verifier_command)
    if verification.get("accepted") is not True or verification.get("status") != "PASS_EXACT_EXTERNAL_REPLAY":
        fail("second pinned CLI replay rejected first CLI output")
    receipt = {
        "schema": "selector-v7-v9-scheduler-acceptance-receipt/v2",
        "status": "PASS_TWO_PROCESS_EXACT_REPLAY_CANDIDATE",
        "scheduler": authority["scheduler"], "cli": authority["cli"],
        "executable": authority["executable"], "authority": False,
        "request_hash": envelope["request_hash"],
        "envelope_hash": envelope["envelope_hash"],
        "cli_verification_hash": sha(canonical(verification)),
        "execute_cli_stdout_bytes": len(execute_stdout),
        "execute_cli_stdout_hash": sha(execute_stdout),
        "verify_cli_stdout_bytes": len(verify_stdout),
        "verify_cli_stdout_hash": sha(verify_stdout),
        "cli_exit_statuses": [0, 0],
        "isolated_cli_invocation_count": 2,
        "isolated_worker_invocation_count": 2,
        "replay_semantics": "FRESH_DETERMINISTIC_EXACT_REQUEST_REPLAY_NO_CACHE",
    }
    receipt["receipt_hash"] = sha(canonical(receipt))
    result = {
        "schema": "selector-v7-v9-scheduler-result/v2",
        "status": "ACCEPTED_EXACT_REPLAY_CANDIDATE_WAIT_ROOT_REVIEW",
        "authority": False,
        "accepted": True,
        "candidate_envelope": envelope,
        "cli_verification": verification,
        "scheduler_receipt": receipt,
    }
    result["result_hash"] = sha(canonical(result))
    return result


def main() -> None:
    authority = verify_boundary()
    command = parse_json(sys.stdin.buffer.read())
    if not isinstance(command, dict) or command.get("schema") != "selector-v7-v9-outer-scheduler-command/v2":
        fail("outer scheduler command schema drift")
    mode = command.get("mode")
    if mode == "execute" and set(command) == {"schema", "mode", "request"}:
        output = execute(command["request"], authority)
    elif mode == "verify" and set(command) == {"schema", "mode", "request", "claimed_result"}:
        expected = execute(command["request"], authority)
        accepted = canonical(command["claimed_result"]) == canonical(expected)
        output = {
            "schema": "selector-v7-v9-outer-verification-result/v2",
            "status": "PASS_EXACT_SCHEDULER_REPLAY" if accepted else "REJECT_CLAIM_DIFFERS_FROM_EXACT_SCHEDULER_REPLAY",
            "authority": False, "accepted": accepted,
            "expected_result_hash": expected["result_hash"],
            "claimed_result_hash": sha(canonical(command["claimed_result"])),
        }
    else:
        fail("outer scheduler command mode/fields drift")
    verify_boundary()
    sys.stdout.buffer.write(canonical(output) + b"\n")


if __name__ == "__main__":
    main()
