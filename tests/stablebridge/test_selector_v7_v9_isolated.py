from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import runpy
import subprocess
import textwrap

import pytest

from stablebridge import physical_repair as package
from stablebridge.physical_repair import selector_v7_v8 as public_v8
from stablebridge.physical_repair import selector_v7_v9_isolated as v9


ROOT = Path(__file__).resolve().parents[2]
V7_FIX = runpy.run_path(str(ROOT / "tests/stablebridge/test_selector_v7_v7.py"))


def case_and_kwargs():
    case = V7_FIX["_one_arm_case"]()
    return case, V7_FIX["_kwargs"](case)


def select(case, kwargs, *, audit_count=0):
    return v9.select_portfolio_v7_v9(
        case["proposal"], case["observations"], case["risks"],
        case["calibration"], case["realizations"], (), case["ledgers"],
        audit_count=audit_count, **kwargs,
    )


def test_authentic_isolated_replay_receipt_verify_and_serialize():
    case, kwargs = case_and_kwargs()
    decision = select(case, kwargs)
    assert decision.decision_payload["state"] == "commit"
    assert decision.decision_payload["selected_candidate_id"] == "candidate-1"
    assert decision.execution_receipt.status == "PASS_ISOLATED_EXACT_S5_REPLAY"
    assert v9.verify_production_decision_v7_v9(decision)
    assert json.loads(v9.serialize_production_decision_v7_v9(decision))["selected_candidate_id"] == "candidate-1"


def test_fresh_child_ignores_preimport_callable_alias_class_and_global_poison():
    script = r'''
from pathlib import Path
import json, runpy
from stablebridge.physical_repair import selector_v7_v5 as v5
from stablebridge.physical_repair import selector_v7_v6 as v6
from stablebridge.physical_repair import selector_v7_v8_execution_auth as old_auth
casefix=runpy.run_path(str(Path.cwd()/"tests/stablebridge/test_selector_v7_v7.py"))
case=casefix["_one_arm_case"](); kwargs=casefix["_kwargs"](case)
v5.PortfolioDecisionV7V5.__post_init__=lambda self: None
v5.SELECTOR_V7_V5_SCHEMA_VERSION="attacker"
v6.select_portfolio_v7_v6.__kwdefaults__["audit_count"]=1
old_auth._PINNED_EXECUTION_BOUNDARY.__closure__[0].cell_contents=lambda *a,**k: None
v6.select_portfolio_v7_v6=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("caller poison executed"))
v6.v5=object()
from stablebridge.physical_repair import selector_v7_v9_isolated as v9
d=v9.select_portfolio_v7_v9(case["proposal"],case["observations"],case["risks"],case["calibration"],case["realizations"],(),case["ledgers"],**kwargs)
print(json.dumps({"selected":d.decision_payload["selected_candidate_id"],"verified":v9.verify_production_decision_v7_v9(d)}))
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    probe = subprocess.run(
        ["/ssd7/cactus8603/roco_spring/.conda-stereo/bin/python3.12", "-B", "-c", textwrap.dedent(script)],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    assert json.loads(probe.stdout) == {"selected": "candidate-1", "verified": True}


def test_caller_environment_is_not_inherited(monkeypatch):
    case, kwargs = case_and_kwargs()
    monkeypatch.setenv("PYTHONPATH", "/attacker")
    monkeypatch.setenv("PYTHONHOME", "/attacker")
    monkeypatch.setenv("LD_PRELOAD", "/attacker.so")
    decision = select(case, kwargs)
    assert v9.verify_production_decision_v7_v9(decision)


def test_runner_and_invoker_substitution_fail_closed(monkeypatch):
    case, kwargs = case_and_kwargs()
    monkeypatch.setattr(v9.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(v9.SelectorV7V9IsolationError, match="transport callable drift"):
        select(case, kwargs)
    monkeypatch.undo()
    monkeypatch.setattr(v9.subprocess.Popen, "communicate", lambda *a, **k: (b"", b""))
    with pytest.raises(v9.SelectorV7V9IsolationError, match="transport callable drift"):
        select(case, kwargs)
    monkeypatch.undo()
    monkeypatch.setattr(v9._PINNED_LAUNCH, "__defaults__", (None,))
    with pytest.raises(v9.SelectorV7V9IsolationError, match="launcher binding drift"):
        select(case, kwargs)
    monkeypatch.undo()
    monkeypatch.setattr(v9, "_launch", lambda request: ({}, {}, {}))
    with pytest.raises(v9.SelectorV7V9IsolationError, match="launcher binding drift"):
        select(case, kwargs)


def test_worker_runtime_source_and_allowlist_substitution_fail_closed(monkeypatch):
    case, kwargs = case_and_kwargs()
    monkeypatch.setattr(v9, "EXECUTABLE", "/usr/bin/python3")
    with pytest.raises(v9.SelectorV7V9IsolationError):
        select(case, kwargs)
    monkeypatch.undo()
    monkeypatch.setitem(v9.SOURCE_CLOSURE, "selector_v7_v6", "0" * 64)
    with pytest.raises(v9.SelectorV7V9IsolationError, match="source drift"):
        select(case, kwargs)
    monkeypatch.undo()
    monkeypatch.setattr(v9, "MODULE_ALLOWLIST_SHA256", "0" * 64)
    with pytest.raises(v9.SelectorV7V9IsolationError, match="module allowlist drift"):
        select(case, kwargs)
    monkeypatch.undo()
    monkeypatch.setattr(v9, "WORKER_SHA256", "0" * 64)
    with pytest.raises(v9.SelectorV7V9IsolationError, match="worker drift"):
        select(case, kwargs)


def test_input_output_and_receipt_mutation_rejected():
    case, kwargs = case_and_kwargs()
    genuine = select(case, kwargs)
    payload = dict(genuine.decision_payload)
    payload["selected_candidate_id"] = "attacker-injected-candidate"
    forged = replace(genuine, decision_payload=payload, decision_v9_hash="")
    assert not v9.verify_production_decision_v7_v9(forged)
    with pytest.raises(ValueError, match="isolated exact S5 replay"):
        v9.serialize_production_decision_v7_v9(forged)
    receipt = replace(genuine.execution_receipt, output_hash="0" * 64, receipt_hash="")
    forged_receipt = replace(genuine, execution_receipt=receipt, decision_v9_hash="")
    assert not v9.verify_production_decision_v7_v9(forged_receipt)


@pytest.mark.parametrize("audit_count", [0, 1])
def test_r16_audit_construction_verification_and_serialization_rejected(audit_count):
    case, kwargs = case_and_kwargs()
    genuine = select(case, kwargs, audit_count=audit_count)
    payload = dict(genuine.decision_payload)
    payload.update({"state": "audit", "audit_count": audit_count})
    with pytest.raises(ValueError, match="AUDIT is categorically rejected"):
        v9.PortfolioDecisionV7V9(genuine.request_bytes, payload, genuine.execution_receipt)


def test_public_package_remains_fail_closed_and_v9_is_not_exported():
    assert not hasattr(package, "select_portfolio_v7_v9")
    assert "select_portfolio_v7_v9" not in package.__all__
    with pytest.raises(public_v8.SelectorExecutionAuthorityUnavailableV7V8):
        public_v8.select_portfolio_v7_v8(object())
