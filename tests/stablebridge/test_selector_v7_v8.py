from __future__ import annotations

import hashlib
from pathlib import Path
import runpy

import pytest

import stablebridge.physical_repair as package
from stablebridge.physical_repair import selector_v7_v6 as v6
from stablebridge.physical_repair import selector_v7_v7 as v7
from stablebridge.physical_repair import selector_v7_v8 as v8


ROOT = Path(__file__).resolve().parents[2]
V7_FIX = runpy.run_path(str(ROOT / "tests/stablebridge/test_selector_v7_v7.py"))


def _genuine_v7_decision():
    case = V7_FIX["_one_arm_case"]()
    return case, V7_FIX["_select"](case)


def test_v8_binds_exact_v7_fail_review_and_predecessor_bytes():
    assert hashlib.sha256(
        (ROOT / "src/stablebridge/physical_repair/selector_v7_v7.py").read_bytes()
    ).hexdigest() == v8.SELECTOR_V7_V7_REVIEWED_SHA256
    assert hashlib.sha256(
        (ROOT / "tests/stablebridge/test_selector_v7_v7.py").read_bytes()
    ).hexdigest() == v8.SELECTOR_V7_V7_REVIEWED_TEST_SHA256
    assert hashlib.sha256(
        (
            ROOT
            / "research/selector_v7_redesign_20261003/"
            "sr1_root_review_v7/REVIEW.json"
        ).read_bytes()
    ).hexdigest() == v8.SR1_ROOT_REVIEW_V7_SHA256
    disposition = v8.SelectorV7V8Disposition()
    assert disposition.predecessor_review_status == v8.SR1_ROOT_REVIEW_V7_STATUS
    assert disposition.r17_disposition.startswith("OPEN_FAIL_CLOSED")
    assert "BLOCKED" in disposition.authority_status


def test_v8_planning_closure_binds_all_v1_through_v7_source_bytes():
    assert len(v8.SELECTOR_V7_V1_V7_SOURCE_CLOSURE) == 7
    assert len({path for path, _ in v8.SELECTOR_V7_V1_V7_SOURCE_CLOSURE}) == 7
    for path, expected in v8.SELECTOR_V7_V1_V7_SOURCE_CLOSURE:
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert len(v8.selector_v7_v8_source_closure_digest()) == 64
    assert len(v8.selector_v7_v8_schema_fingerprint()) == 64


def test_v8_requires_isolated_external_runner_and_has_no_positive_path():
    contract = v8.SelectorExecutionContractV7V8()
    assert contract.python_flags == ("-I", "-S")
    assert contract.process_mode == "ISOLATED_EXTERNAL_PROCESS"
    assert contract.environment_mode == "ALLOWLIST_ONLY_NO_CALLER_ENV_COPY"
    assert contract.import_policy == "EXACT_PATH_BYTES_SHA256_SET_EQUALITY_BEFORE_LOAD"
    assert contract.execution_receipt_anchor == "EXTERNAL_TO_CALLER_PROCESS"
    assert contract.production_positive_path is False
    assert len(contract.contract_hash) == 64


def test_r17_direct_v6_runtime_substitution_is_never_invoked(monkeypatch):
    calls: list[str] = []

    def substituted(*args, **kwargs):
        del args, kwargs
        calls.append("substituted-v6-runtime-executed")
        raise AssertionError("v8 must reject before invoking v6")

    monkeypatch.setattr(v6, "select_portfolio_v7_v6", substituted)
    with pytest.raises(
        v8.SelectorExecutionAuthorityUnavailableV7V8,
        match="BLOCKED_MISSING_EXTERNALLY_AUTHENTICATED_ISOLATED_RUNNER",
    ):
        v8.select_portfolio_v7_v8(object(), locally_minted_registry=object())
    assert calls == []


def test_r17_transitive_v5_runtime_substitution_is_never_invoked(monkeypatch):
    calls: list[str] = []

    def substituted(*args, **kwargs):
        del args, kwargs
        calls.append("substituted-v5-runtime-executed")
        raise AssertionError("v8 must reject before invoking v5")

    monkeypatch.setattr(v6.v5, "select_portfolio_v7_v5", substituted)
    with pytest.raises(v8.SelectorExecutionAuthorityUnavailableV7V8):
        v8.select_portfolio_v7_v8(
            object(),
            external_execution_receipt={"self_signed": True},
            trusted_execution_registry_hashes=frozenset({"caller-minted"}),
        )
    assert calls == []


def test_caller_minted_registry_and_coordinated_receipt_cannot_enable_v8():
    fake = {
        "status": "PASS",
        "runner_hash": "0" * 64,
        "closure_hash": "1" * 64,
        "output_hash": "2" * 64,
        "receipt_hash": "3" * 64,
    }
    with pytest.raises(v8.SelectorExecutionAuthorityUnavailableV7V8):
        v8.select_portfolio_v7_v8(
            fake,
            external_execution_receipt=fake,
            trusted_execution_registry_hashes=frozenset({"3" * 64}),
        )
    assert not v8.verify_production_decision_v7_v8(
        fake,
        external_execution_receipt=fake,
        trusted_execution_registry_hashes=frozenset({"3" * 64}),
    )


def test_even_genuine_v7_decision_cannot_cross_unavailable_execution_boundary():
    _, genuine = _genuine_v7_decision()
    assert not v8.verify_production_decision_v7_v8(
        genuine,
        trusted_execution_registry_hashes=frozenset(),
    )
    with pytest.raises(v8.SelectorExecutionAuthorityUnavailableV7V8):
        v8.serialize_production_decision_v7_v8(
            genuine,
            trusted_execution_registry_hashes=frozenset(),
        )
    with pytest.raises(v8.SelectorExecutionAuthorityUnavailableV7V8):
        v8.PortfolioDecisionV7V8(genuine, "future-external-receipt")


def test_v8_preserves_proposal_surface_but_not_decision_authority():
    assert v8.CandidateManifestV7V8 is v7.CandidateManifestV7V7
    assert v8.PlannerProposalV7V8 is v7.PlannerProposalV7V7
    assert v8.propose_candidates_v7_v8 is v7.propose_candidates_v7_v7
    assert v8.verify_production_proposal_v7_v8 is v7.verify_production_proposal_v7_v7
    assert v8.SELECTOR_V7_V8_SCOPE == "S5_ONLY_PRODUCTION_BLOCKED"
    assert "DEFERRED" in v8.S6_AUDIT_STATUS_V7V8


def test_v8_exports_are_complete_and_no_test_only_or_local_enable_switch():
    assert set(v8.__all__).issubset(package.__all__)
    assert all(hasattr(package, name) for name in v8.__all__)
    assert "select_portfolio_v7_v8_test_only" not in v8.__all__
    assert not hasattr(v8, "select_portfolio_v7_v8_test_only")
    assert not hasattr(v8, "enable_production_v7_v8")
    assert not hasattr(v8, "register_trusted_runner_v7_v8")
