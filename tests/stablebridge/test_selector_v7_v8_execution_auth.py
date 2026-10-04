from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from stablebridge.physical_repair import selector_v7_v5 as v5
from stablebridge.physical_repair import selector_v7_v6 as v6
from stablebridge.physical_repair import selector_v7_v8_execution_auth as v8


ROOT = Path(__file__).resolve().parents[2]
V7_FIX = runpy.run_path(str(ROOT / "tests/stablebridge/test_selector_v7_v7.py"))


def _case():
    return V7_FIX["_one_arm_case"]()


def _kwargs(case):
    return V7_FIX["_kwargs"](case)


def _select(case, *, audit_count=0):
    return v8.select_portfolio_v7_v8(
        case["proposal"], case["observations"], case["risks"],
        case["calibration"], case["realizations"], (), case["ledgers"],
        audit_count=audit_count, **_kwargs(case),
    )


def _forged_v6(genuine, **changes):
    base = replace(
        genuine.base_decision.base_decision,
        decision_hash="",
        **changes,
    )
    return v6.PortfolioDecisionV7V6(base, genuine.base_decision.proposal)


def _unsafe_replace(instance, **changes):
    forged = object.__new__(type(instance))
    for name in instance.__dataclass_fields__:
        object.__setattr__(
            forged, name, changes.get(name, getattr(instance, name)),
        )
    return forged


def test_v8_binds_exact_v7_failure_and_preserves_v1_v7_sources():
    assert hashlib.sha256(
        (ROOT / "src/stablebridge/physical_repair/selector_v7_v7.py").read_bytes()
    ).hexdigest() == v8.SELECTOR_V7_V7_REVIEWED_SHA256
    assert hashlib.sha256(
        (ROOT / "tests/stablebridge/test_selector_v7_v7.py").read_bytes()
    ).hexdigest() == v8.SELECTOR_V7_V7_REVIEWED_TEST_SHA256
    assert hashlib.sha256(
        (ROOT / "research/selector_v7_redesign_20261003/sr1_root_review_v7/REVIEW.json").read_bytes()
    ).hexdigest() == v8.SR1_ROOT_REVIEW_V7_SHA256
    assert v8._source_closure_now() == v8.SELECTOR_V7_V1_V7_SOURCE_CLOSURE
    assert len(v8._PINNED_CODE_CLOSURE) >= 100
    assert len(v8.selector_v7_v8_schema_fingerprint()) == 64


def test_authentic_path_binds_full_input_output_closure_and_environment():
    case = _case()
    decision = _select(case)
    receipt = decision.execution_receipt
    assert decision.base_decision.base_decision.state is v5.DecisionStateV7V5.COMMIT
    assert receipt.runner_input_hash
    assert receipt.runner_output_hash == v8._sha256(
        v8._decision_payload(decision.base_decision)
    )
    assert receipt.environment_hash
    assert receipt.execution_contract_hash == (
        decision.replay_bundle.execution_contract.contract_hash
    )
    assert v8.verify_production_decision_v7_v8(decision, **_kwargs(case))
    payload = json.loads(v8.serialize_production_decision_v7_v8(
        decision, **_kwargs(case),
    ))
    assert payload["selected_candidate_id"] == (
        decision.base_decision.base_decision.selected_candidate_id
    )
    assert payload["safe_candidate_ids"] == list(
        decision.base_decision.base_decision.safe_candidate_ids
    )
    assert payload["authority_status"] == v8.SELECTOR_V7_V8_AUTHORITY_STATUS


def test_r17_public_v6_selector_substitution_rejects_every_boundary(monkeypatch):
    case = _case()
    genuine = _select(case)
    forged = _forged_v6(
        genuine,
        selected_candidate_id="attacker-injected-candidate",
        safe_candidate_ids=("attacker-injected-candidate",),
        arm_reasons=(), arm_audit_states=(),
    )
    monkeypatch.setattr(v6, "select_portfolio_v7_v6", lambda *a, **k: forged)
    with pytest.raises(v8.SelectorExecutionAuthorityUnavailableV7V8):
        v8.AuthenticatedReplayExecutionReceiptV7V8(
            genuine.replay_bundle,
            forged,
            genuine.replay_bundle.execution_contract.contract_hash,
        )
    with pytest.raises(v8.SelectorExecutionAuthorityUnavailableV7V8):
        _select(case)
    assert not v8.verify_production_decision_v7_v8(genuine, **_kwargs(case))
    with pytest.raises(ValueError, match="authenticated exact S5 replay"):
        v8.serialize_production_decision_v7_v8(genuine, **_kwargs(case))


def test_r17_transitive_v5_symbol_substitution_is_rejected(monkeypatch):
    case = _case()
    genuine = _select(case)
    original = v6.v5.select_portfolio_v7_v5
    monkeypatch.setattr(
        v6.v5, "select_portfolio_v7_v5", lambda *a, **k: original(*a, **k),
    )
    with pytest.raises(v8.SelectorExecutionAuthorityUnavailableV7V8):
        v8.AuthenticatedReplayExecutionReceiptV7V8(
            genuine.replay_bundle,
            genuine.base_decision,
            genuine.replay_bundle.execution_contract.contract_hash,
        )
    assert not v8.verify_production_decision_v7_v8(genuine, **_kwargs(case))


@pytest.mark.parametrize(
    "binding",
    [
        "_PINNED_EXECUTION_BOUNDARY",
        "_PINNED_REPLAY_INVOKER",
        "_PINNED_V6_SELECTOR_CALLABLE",
        "_PINNED_TRANSITIVE_CALLABLES",
    ],
)
def test_r17_private_binding_substitution_is_rejected(monkeypatch, binding):
    case = _case()
    genuine = _select(case)
    monkeypatch.setattr(v8, binding, lambda *a, **k: genuine.base_decision)
    with pytest.raises(v8.SelectorExecutionAuthorityUnavailableV7V8):
        v8.AuthenticatedReplayExecutionReceiptV7V8(
            genuine.replay_bundle,
            genuine.base_decision,
            genuine.replay_bundle.execution_contract.contract_hash,
        )
    assert not v8.verify_production_decision_v7_v8(genuine, **_kwargs(case))


def test_r17_pinned_callable_code_object_substitution_is_rejected(monkeypatch):
    case = _case()
    genuine = _select(case)
    monkeypatch.setattr(
        v6.select_portfolio_v7_v6,
        "__code__",
        (lambda *args, **kwargs: None).__code__,
    )
    with pytest.raises(v8.SelectorExecutionAuthorityUnavailableV7V8):
        v8.SelectorExecutionContractV7V8()
    assert not v8.verify_production_decision_v7_v8(genuine, **_kwargs(case))


def test_r17_source_allowlist_or_runtime_binding_substitution_rejects(monkeypatch):
    case = _case()
    genuine = _select(case)
    hostile = tuple(v8.SELECTOR_V7_V1_V7_SOURCE_CLOSURE[:-1])
    monkeypatch.setattr(v8, "SELECTOR_V7_V1_V7_SOURCE_CLOSURE", hostile)
    with pytest.raises(v8.SelectorExecutionAuthorityUnavailableV7V8):
        v8.SelectorExecutionContractV7V8()
    assert not v8.verify_production_decision_v7_v8(genuine, **_kwargs(case))


def test_r15_exact_output_still_rejects_resigned_candidate():
    case = _case()
    genuine = _select(case)
    forged = _forged_v6(
        genuine,
        selected_candidate_id="attacker-injected-candidate",
        safe_candidate_ids=("attacker-injected-candidate",),
        arm_reasons=(), arm_audit_states=(),
    )
    with pytest.raises(ValueError, match="authenticated selector replay"):
        v8.AuthenticatedReplayExecutionReceiptV7V8(
            genuine.replay_bundle,
            forged,
            genuine.replay_bundle.execution_contract.contract_hash,
        )
    tampered = _unsafe_replace(genuine, base_decision=forged)
    assert not v8.verify_production_decision_v7_v8(tampered, **_kwargs(case))


@pytest.mark.parametrize("audit_count", [0, 1])
def test_r16_audit_remains_rejected_for_both_counter_states(audit_count):
    case = _case()
    genuine = _select(case, audit_count=audit_count)
    forged = _forged_v6(
        genuine,
        state=v5.DecisionStateV7V5.AUDIT,
        selected_candidate_id=None,
        safe_candidate_ids=(), arm_reasons=(), arm_audit_states=(),
        reason=v5.DecisionReasonV7V5.NO_SAFE_CANDIDATE,
        audit_count=audit_count,
    )
    with pytest.raises(ValueError, match="AUDIT is categorically rejected"):
        v8.PortfolioDecisionV7V8(
            forged, genuine.replay_bundle, genuine.execution_receipt,
        )
    tampered = _unsafe_replace(genuine, base_decision=forged)
    assert not v8.verify_production_decision_v7_v8(tampered, **_kwargs(case))
    with pytest.raises(ValueError, match="authenticated exact S5 replay"):
        v8.serialize_production_decision_v7_v8(tampered, **_kwargs(case))


def test_v8_has_no_test_only_selector_entrypoint():
    assert "select_portfolio_v7_v8_test_only" not in v8.__all__
    assert not hasattr(v8, "select_portfolio_v7_v8_test_only")
