from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types

import pytest

from stablebridge import physical_repair as package
from stablebridge.physical_repair import selector_v7_v9_isolated as v1_wrapper


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "src/stablebridge/physical_repair/selector_v7_v9_cli_v2.py"
SCHEDULER = ROOT / "src/stablebridge/physical_repair/selector_v7_v9_scheduler_v2.py"
PYTHON = "/ssd7/cactus8603/roco_spring/.conda-stereo/bin/python"
CLEAN_ENV = {"LANG": "C", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
V7_FIX = runpy.run_path(str(ROOT / "tests/stablebridge/test_selector_v7_v7.py"))


def request() -> dict[str, object]:
    case = V7_FIX["_one_arm_case"]()
    kwargs = V7_FIX["_kwargs"](case)
    payload = {
        "proposal": case["proposal"], "observations": tuple(case["observations"]),
        "risks": tuple(case["risks"]), "calibration": case["calibration"],
        "realizations": tuple(case["realizations"]), "aliases": (),
        "runtime_cost_ledgers": tuple(case["ledgers"]),
        "production_registry": kwargs["production_registry"],
        "trusted_production_registry_hashes": kwargs["trusted_production_registry_hashes"],
        "trusted_split_registry_hashes": kwargs["trusted_split_registry_hashes"],
        "audit_count": 0, "child_count": 0,
    }
    return json.loads(v1_wrapper._request_bytes(payload))


def command(mode: str, request_value: dict[str, object], claim: object = None) -> dict[str, object]:
    value = {"schema": "selector-v7-v9-scheduler-command/v2", "mode": mode, "request": request_value}
    if mode == "verify":
        value["claimed_envelope"] = claim
    return value


def exact_external_program(program: Path, value: object) -> dict[str, object]:
    """Invoke the reviewed boundary without parent subprocess/hashlib objects."""
    stdin_read, stdin_write = os.pipe()
    stdout_read, stdout_write = os.pipe()
    stderr_read, stderr_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.dup2(stdin_read, 0); os.dup2(stdout_write, 1); os.dup2(stderr_write, 2)
            for fd in (stdin_read, stdin_write, stdout_read, stdout_write, stderr_read, stderr_write):
                if fd > 2:
                    os.close(fd)
            os.execve(PYTHON, [PYTHON, "-I", "-S", "-B", str(program)], CLEAN_ENV)
        finally:
            os._exit(127)
    os.close(stdin_read); os.close(stdout_write); os.close(stderr_write)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    os.write(stdin_write, raw); os.close(stdin_write)
    stdout = bytearray(); stderr = bytearray()
    while True:
        chunk = os.read(stdout_read, 65536)
        if not chunk:
            break
        stdout.extend(chunk)
    while True:
        chunk = os.read(stderr_read, 65536)
        if not chunk:
            break
        stderr.extend(chunk)
    os.close(stdout_read); os.close(stderr_read)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, bytes(stderr).decode(errors="replace")
    assert not stderr
    return json.loads(stdout)


def exact_external_cli(value: object) -> dict[str, object]:
    return exact_external_program(CLI, value)


def outer_command(mode: str, request_value: dict[str, object], claim: object = None) -> dict[str, object]:
    value = {"schema": "selector-v7-v9-outer-scheduler-command/v2", "mode": mode, "request": request_value}
    if mode == "verify":
        value["claimed_result"] = claim
    return value


def exact_external_scheduler(value: object) -> dict[str, object]:
    return exact_external_program(SCHEDULER, value)


def test_exact_external_execute_and_independent_verify():
    req = request()
    envelope = exact_external_cli(command("execute", req))
    assert envelope["output"]["state"] == "commit"
    assert envelope["output"]["selected_candidate_id"] == "candidate-1"
    assert envelope["authority"] is False
    verified = exact_external_cli(command("verify", req, envelope))
    assert verified["accepted"] is True
    assert verified["status"] == "PASS_EXACT_EXTERNAL_REPLAY"


def test_whole_wrapper_parent_subprocess_and_hash_poison_cannot_cross_boundary(monkeypatch):
    req = request()
    monkeypatch.setattr(v1_wrapper, "select_portfolio_v7_v9", lambda *a, **k: {"forged": True})
    monkeypatch.setattr(v1_wrapper, "verify_production_decision_v7_v9", lambda *a, **k: True)
    monkeypatch.setattr(v1_wrapper, "json", object())
    monkeypatch.setattr(v1_wrapper, "Path", lambda *a, **k: Path("/attacker"))
    monkeypatch.setitem(sys.modules, "stablebridge.physical_repair.selector_v7_v6", types.SimpleNamespace(attacker=True))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("poison")))
    monkeypatch.setattr(hashlib, "sha256", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("poison")))
    result = exact_external_scheduler(outer_command("execute", req))
    assert result["candidate_envelope"]["output"]["selected_candidate_id"] == "candidate-1"
    assert result["accepted"] is True and result["authority"] is False
    assert exact_external_scheduler(outer_command("verify", req, result))["accepted"] is True


def test_fabricated_child_stdout_receipt_and_verifier_dict_are_rejected(monkeypatch):
    req = request()
    genuine = exact_external_cli(command("execute", req))
    forged = json.loads(json.dumps(genuine))
    forged["output"]["selected_candidate_id"] = "attacker-candidate"
    forged["output_hash"] = "0" * 64
    forged["execution_receipt"]["output_hash"] = "0" * 64
    forged["execution_receipt"]["receipt_hash"] = "0" * 64
    forged["envelope_hash"] = "0" * 64
    monkeypatch.setattr(v1_wrapper, "verify_production_decision_v7_v9", lambda *a, **k: True)
    fabricated_verifier_stdout = {"accepted": True, "status": "PASS_EXACT_EXTERNAL_REPLAY"}
    assert fabricated_verifier_stdout["accepted"] is True  # untrusted bytes alone prove nothing
    verified = exact_external_cli(command("verify", req, forged))
    assert verified["accepted"] is False
    assert verified["status"] == "REJECT_CLAIM_DIFFERS_FROM_EXACT_REPLAY"
    assert verified["claimed_envelope_hash"] != verified["expected_envelope_hash"]


def test_fake_scheduler_stdout_or_nested_cli_receipt_is_rejected_by_exact_outer_replay():
    req = request()
    genuine = exact_external_scheduler(outer_command("execute", req))
    forged = json.loads(json.dumps(genuine))
    forged["candidate_envelope"]["output"]["selected_candidate_id"] = "attacker-candidate"
    forged["cli_verification"] = {"accepted": True, "status": "PASS_EXACT_EXTERNAL_REPLAY"}
    forged["scheduler_receipt"]["receipt_hash"] = "0" * 64
    forged["result_hash"] = "0" * 64
    verification = exact_external_scheduler(outer_command("verify", req, forged))
    assert verification["accepted"] is False
    assert verification["status"] == "REJECT_CLAIM_DIFFERS_FROM_EXACT_SCHEDULER_REPLAY"


def test_old_result_is_rejected_for_a_different_exact_request():
    req = request()
    old = exact_external_scheduler(outer_command("execute", req))
    changed = json.loads(json.dumps(req))
    fields = {row[0]: row for row in changed["payload"]["value"]}
    fields["child_count"][1] = 1
    changed["request_hash"] = hashlib.new(
        "sha256", json.dumps(changed["payload"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    verification = exact_external_scheduler(outer_command("verify", changed, old))
    assert verification["accepted"] is False
    assert verification["expected_result_hash"] != old["result_hash"]


@pytest.mark.parametrize("audit_count", [0, 1])
def test_r16_audit_output_cannot_be_accepted(audit_count):
    req = request()
    genuine = exact_external_cli(command("execute", req))
    forged = json.loads(json.dumps(genuine))
    forged["output"]["state"] = "audit"
    forged["output"]["audit_count"] = audit_count
    assert exact_external_cli(command("verify", req, forged))["accepted"] is False


def test_append_only_public_boundary_and_v1_freeze_unchanged():
    assert not hasattr(package, "select_portfolio_v7_v9")
    assert "select_portfolio_v7_v9" not in package.__all__
    frozen = ROOT / "research/selector_v7_redesign_20261003/sr1_v9_isolated_execution_candidate_v1"
    assert hashlib.new("sha256", (frozen / "PREEXECUTION_FREEZE_CANDIDATE.json").read_bytes()).hexdigest() == "8402339ebeb7191fc1d82204fa17d85edadd88bc22bec472aaa9f0b40c6abd18"
    assert hashlib.new("sha256", (frozen / "CANDIDATE.json").read_bytes()).hexdigest() == "bebe0c3fc3ac58ebbf4d07a4a801150963a4720883db8e3fef70a989b49e1688"
