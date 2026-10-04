from __future__ import annotations

import copy
import hashlib

import pytest

from stablebridge.physical_repair.action_qualification_fresh_joint_ledger_v1 import (
    REQUIRED_EVIDENCE_BINDINGS,
    build_fresh_component_joint_ledger_v1,
)
from stablebridge.physical_repair.action_qualification_joint_ledger_v1 import (
    ACTION_ORDER,
    CASE_SCHEMA,
    STRATUM_TO_ACTION,
    STRATUM_ORDER,
    canonical_sha256,
)
from stablebridge.physical_repair.action_qualification_power_v1 import SCHEMA as POWER_SCHEMA
from stablebridge.physical_repair.action_qualification_science_v1 import (
    BOOTSTRAP_REPEATS,
    evaluate_fresh_action_science_v1,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _bindings():
    return {name: _sha(name) for name in REQUIRED_EVIDENCE_BINDINGS}


def _rows():
    rows = []
    allocations = {
        "spring": (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4),
        "kitti": (0, 1, 2, 3, 4),
    }
    for source, folds in allocations.items():
        for index, fold in enumerate(folds):
            component = f"{source}-{index:02d}"
            residual = ((index % 5) - 2) * 0.0001
            for stratum in STRATUM_ORDER:
                case = f"{component}-{stratum}"
                rows.append({
                    "schema": CASE_SCHEMA,
                    "case_id": case,
                    "component_id": component,
                    "source_dataset": source,
                    "outer_fold": fold,
                    "mechanism_stratum": stratum,
                    "action_id": STRATUM_TO_ACTION[stratum],
                    "execution_status": "OBSERVED",
                    "execution_missing_reason": None,
                    "task_status": "OBSERVED",
                    "task_missing_reason": None,
                    "task_gain_normalized_px": 0.03 + residual,
                    "harmed_pixel_fraction": 0.01 + residual,
                    "pixel_harm_cvar95_raw_px": 0.05 + residual,
                    "severe_event": False,
                    "mechanism_status": "OBSERVED",
                    "mechanism_missing_reason": None,
                    "mechanism_primary": 0.10 + residual,
                    "mechanism_specificity": 0.05 + residual,
                    "cost_gate_pass": True,
                    "child_observables_complete": True,
                    "case_receipt_sha256": _sha(case),
                })
    return rows


def _ledger(rows=None):
    return build_fresh_component_joint_ledger_v1(
        _rows() if rows is None else rows,
        roster_sha256=_sha("roster"), evidence_bindings=_bindings(),
    )


def _power(*, denied_action=None):
    per_action = {}
    for action in ACTION_ORDER:
        probability = 0.79 if action == denied_action else 0.95
        per_action[action] = {
            "adoption_probability": probability,
            "meets_0p80_power_admission": probability >= 0.8,
        }
    payload = {
        "schema": POWER_SCHEMA,
        "power_admission_complete": True,
        "per_action": per_action,
    }
    return {**payload, "power_result_sha256": canonical_sha256(payload)}


def _evaluate(rows=None, *, denied_action=None):
    return evaluate_fresh_action_science_v1(
        _ledger(rows), development_power_admission=_power(denied_action=denied_action),
        science_protocol_sha256=_sha("science-protocol"),
    )


def test_all_six_positive_safe_complete_actions_are_adopted_individually():
    result = _evaluate()
    assert result["protocol"]["bootstrap_repeats"] == BOOTSTRAP_REPEATS
    assert len(result["adopted_action_ids"]) == 6
    assert all(
        row["decision"] == "ADOPT_EXACT_ACTION_FOR_LATER_SELECTOR_BUILD"
        for row in result["per_action"].values()
    )
    assert result["native"]["decision"].startswith("REFERENCE_RETAIN")
    assert not result["selector_admission"]
    payload = {k: v for k, v in result.items() if k != "science_result_sha256"}
    assert result["science_result_sha256"] == canonical_sha256(payload)


def test_power_denial_is_inconclusive_only_for_that_action():
    denied = ACTION_ORDER[0]
    result = _evaluate(denied_action=denied)
    assert result["per_action"][denied]["decision"] == "INCONCLUSIVE"
    assert all(
        row["decision"] == "ADOPT_EXACT_ACTION_FOR_LATER_SELECTOR_BUILD"
        for action, row in result["per_action"].items() if action != denied
    )


def test_adequately_powered_futility_disables_exact_version_not_family():
    rows = _rows()
    target = ACTION_ORDER[0]
    for row in rows:
        if row["action_id"] == target:
            row["task_gain_normalized_px"] = -0.03
    result = _evaluate(rows)
    decision = result["per_action"][target]
    assert decision["decision"] == "DISABLE_EXACT_ACTION_VERSION"
    assert decision["disable_reason"].endswith("EFFICACY_UPPER_BELOW_PRACTICAL_MINIMUM")


def test_fresh_unresolved_severe_event_disables_only_exact_version():
    rows = _rows()
    target = ACTION_ORDER[3]
    row = next(value for value in rows if value["action_id"] == target)
    row["severe_event"] = True
    result = _evaluate(rows)
    decision = result["per_action"][target]
    assert decision["decision"] == "DISABLE_EXACT_ACTION_VERSION"
    assert decision["disable_reason"].endswith("UNRESOLVED_SEVERE_HARM")


def test_nonfinite_action_is_task_missing_but_still_severe_and_disabling():
    rows = _rows()
    target = ACTION_ORDER[2]
    row = next(value for value in rows if value["action_id"] == target)
    row.update({
        "task_status": "TYPED_MISSING",
        "task_missing_reason": "NONFINITE_ACTION_ON_FIXED_GT_SUPPORT",
        "task_gain_normalized_px": None,
        "harmed_pixel_fraction": None,
        "pixel_harm_cvar95_raw_px": None,
        "severe_event": True,
    })
    result = _evaluate(rows)
    decision = result["per_action"][target]
    assert decision["decision"] == "DISABLE_EXACT_ACTION_VERSION"
    assert decision["severe_event_count"] == 1


def test_many_nonfinite_severe_events_do_not_paradoxically_become_inconclusive():
    rows = _rows()
    target = ACTION_ORDER[2]
    changed = 0
    for row in rows:
        if row["action_id"] == target and changed < 5:
            row.update({
                "task_status": "TYPED_MISSING",
                "task_missing_reason": "NONFINITE_ACTION_ON_FIXED_GT_SUPPORT",
                "task_gain_normalized_px": None,
                "harmed_pixel_fraction": None,
                "pixel_harm_cvar95_raw_px": None,
                "severe_event": True,
            })
            changed += 1
    result = _evaluate(rows)
    decision = result["per_action"][target]
    assert decision["decision"] == "DISABLE_EXACT_ACTION_VERSION"
    assert decision["severe_event_count"] == 5


def test_conclusive_task_futility_is_not_hidden_by_mechanism_missingness():
    rows = _rows()
    target = ACTION_ORDER[0]
    changed = 0
    for row in rows:
        if row["action_id"] == target:
            row["task_gain_normalized_px"] = -0.03
            if changed < 4:
                row.update({
                    "mechanism_status": "TYPED_MISSING",
                    "mechanism_missing_reason": "MECHANISM_NOT_ESTIMABLE",
                    "mechanism_primary": None,
                    "mechanism_specificity": None,
                })
                changed += 1
    result = _evaluate(rows)
    assert result["per_action"][target]["decision"] == (
        "DISABLE_EXACT_ACTION_VERSION"
    )


def test_failed_mechanism_without_strong_negative_evidence_requests_redesign():
    rows = _rows()
    target = ACTION_ORDER[1]
    for row in rows:
        if row["action_id"] == target:
            row["mechanism_primary"] = -0.01
            row["mechanism_specificity"] = -0.01
    result = _evaluate(rows)
    assert result["per_action"][target]["decision"] == "RETAIN_AND_VERSION_REDESIGN"


def test_missing_coverage_is_inconclusive_and_never_imputed_zero():
    rows = _rows()
    target = ACTION_ORDER[0]
    changed = 0
    for row in rows:
        if row["action_id"] == target and changed < 4:
            row.update({
                "execution_status": "TYPED_MISSING",
                "execution_missing_reason": "UNSUPPORTED",
                "task_status": "TYPED_MISSING",
                "task_missing_reason": "UNSUPPORTED",
                "task_gain_normalized_px": None,
                "harmed_pixel_fraction": None,
                "pixel_harm_cvar95_raw_px": None,
                "severe_event": None,
                "mechanism_status": "TYPED_MISSING",
                "mechanism_missing_reason": "UNSUPPORTED",
                "mechanism_primary": None,
                "mechanism_specificity": None,
                "cost_gate_pass": False,
            })
            changed += 1
    result = _evaluate(rows)
    assert result["per_action"][target]["decision"] == "INCONCLUSIVE"
    task = result["coverage"]["metrics"][f"task:{target}"]
    assert task["observed_components"] == 19
    assert not task["pass"]


def test_one_unestimable_action_tail_does_not_block_other_action_decisions():
    rows = _rows()
    target = ACTION_ORDER[0]
    for row in rows:
        if row["action_id"] == target and row["source_dataset"] == "kitti":
            row.update({
                "execution_status": "TYPED_MISSING",
                "execution_missing_reason": "UNSUPPORTED",
                "task_status": "TYPED_MISSING",
                "task_missing_reason": "UNSUPPORTED",
                "task_gain_normalized_px": None,
                "harmed_pixel_fraction": None,
                "pixel_harm_cvar95_raw_px": None,
                "severe_event": None,
                "mechanism_status": "TYPED_MISSING",
                "mechanism_missing_reason": "UNSUPPORTED",
                "mechanism_primary": None,
                "mechanism_specificity": None,
                "cost_gate_pass": False,
                "child_observables_complete": False,
            })
    result = _evaluate(rows)
    assert result["per_action"][target]["decision"] == "INCONCLUSIVE"
    assert all(
        row["decision"] == "ADOPT_EXACT_ACTION_FOR_LATER_SELECTOR_BUILD"
        for action, row in result["per_action"].items() if action != target
    )


@pytest.mark.parametrize("failed_gate", ("cost_gate_pass", "child_observables_complete"))
def test_missing_isotropic_stratum_cannot_hide_executed_sibling_gate_failure(
    failed_gate,
):
    rows = _rows()
    target = "common_isotropic.local_recoverable_v3"
    component = "spring-00"
    missing = next(
        row for row in rows
        if row["component_id"] == component
        and row["mechanism_stratum"] == "common_disk"
    )
    missing.update({
        "execution_status": "TYPED_MISSING",
        "execution_missing_reason": "UNSUPPORTED",
        "task_status": "TYPED_MISSING",
        "task_missing_reason": "UNSUPPORTED",
        "task_gain_normalized_px": None,
        "harmed_pixel_fraction": None,
        "pixel_harm_cvar95_raw_px": None,
        "severe_event": None,
        "mechanism_status": "TYPED_MISSING",
        "mechanism_missing_reason": "UNSUPPORTED",
        "mechanism_primary": None,
        "mechanism_specificity": None,
    })
    executed = next(
        row for row in rows
        if row["component_id"] == component
        and row["mechanism_stratum"] == "common_gaussian"
    )
    executed[failed_gate] = False
    result = _evaluate(rows)
    assert result["per_action"][target]["gates"][
        "prospective_cost" if failed_gate == "cost_gate_pass"
        else "local_child_observables"
    ] is False
    assert result["per_action"][target]["decision"] != (
        "ADOPT_EXACT_ACTION_FOR_LATER_SELECTOR_BUILD"
    )


def test_tampered_ledger_or_power_receipt_fails_closed():
    ledger = _ledger()
    broken = copy.deepcopy(ledger)
    broken["components"][0]["task"][ACTION_ORDER[0]]["normalized_gain"] = 99.0
    with pytest.raises(ValueError, match="semantic hash drift"):
        evaluate_fresh_action_science_v1(
            broken, development_power_admission=_power(),
            science_protocol_sha256=_sha("science-protocol"),
        )
    power = _power()
    power["per_action"][ACTION_ORDER[0]]["adoption_probability"] = 0.0
    with pytest.raises(ValueError, match="semantic hash drift"):
        evaluate_fresh_action_science_v1(
            ledger, development_power_admission=power,
            science_protocol_sha256=_sha("science-protocol"),
        )
