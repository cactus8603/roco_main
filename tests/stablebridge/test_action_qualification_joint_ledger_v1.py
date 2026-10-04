from __future__ import annotations

import copy
import hashlib

import pytest

from stablebridge.physical_repair.action_qualification_joint_ledger_v1 import (
    ACTION_TO_STRATA,
    CASE_SCHEMA,
    OBSERVED,
    STRATUM_ORDER,
    TYPED_MISSING,
    build_component_joint_ledger_v1,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _rows():
    rows = []
    inverse = {
        stratum: action
        for action, strata in ACTION_TO_STRATA.items()
        for stratum in strata
    }
    for source, count in (("spring", 18), ("kitti", 22)):
        for index in range(count):
            component = f"{source}-{index:02d}"
            for stratum_index, stratum in enumerate(STRATUM_ORDER):
                case_id = f"{component}-{stratum}"
                rows.append({
                    "schema": CASE_SCHEMA,
                    "case_id": case_id,
                    "component_id": component,
                    "source_dataset": source,
                    "outer_fold": index % 5,
                    "mechanism_stratum": stratum,
                    "action_id": inverse[stratum],
                    "execution_status": OBSERVED,
                    "execution_missing_reason": None,
                    "task_status": OBSERVED,
                    "task_missing_reason": None,
                    "task_gain_normalized_px": float(stratum_index + 1),
                    "harmed_pixel_fraction": 0.01,
                    "pixel_harm_cvar95_raw_px": 0.1,
                    "severe_event": False,
                    "mechanism_status": OBSERVED,
                    "mechanism_missing_reason": None,
                    "mechanism_primary": 0.2,
                    "mechanism_specificity": 0.1,
                    "cost_gate_pass": True,
                    "child_observables_complete": True,
                    "case_receipt_sha256": _sha(case_id),
                })
    return rows


def _build(rows):
    return build_component_joint_ledger_v1(
        rows,
        protocol_sha256=_sha("protocol"),
        component_roster_sha256=_sha("roster"),
    )


def _component(result, component_id):
    return next(
        row for row in result["components"]
        if row["component_id"] == component_id
    )


def test_280_cases_reduce_to_40_independent_components():
    result = _build(_rows())
    assert len(result["components"]) == 40
    assert result["manifest"]["case_count"] == 280
    assert result["manifest"]["scientific_n"] == 40
    assert result["manifest"]["source_component_counts"] == {
        "spring": 18, "kitti": 22,
    }
    assert set(result["components"][0]["availability"]) == set(STRATUM_ORDER)


def test_isotropic_disk_and_gaussian_are_averaged_inside_component():
    rows = _rows()
    first = rows[0]["component_id"]
    for row in rows:
        if row["component_id"] == first and row["mechanism_stratum"] == "common_disk":
            row["task_gain_normalized_px"] = 2.0
        if row["component_id"] == first and row["mechanism_stratum"] == "common_gaussian":
            row["task_gain_normalized_px"] = 6.0
    component = _component(_build(rows), first)
    value = component["task"]["common_isotropic.local_recoverable_v3"]
    assert value["normalized_gain"] == 4.0
    assert value["owned_strata"] == ["common_disk", "common_gaussian"]


def test_mechanism_missing_does_not_delete_valid_task_evidence():
    rows = _rows()
    target = rows[0]
    target["mechanism_status"] = TYPED_MISSING
    target["mechanism_missing_reason"] = "MECHANISM_METRIC_UNESTIMABLE"
    target["mechanism_primary"] = None
    target["mechanism_specificity"] = None
    component = _component(_build(rows), target["component_id"])
    action = target["action_id"]
    stratum = target["mechanism_stratum"]
    assert component["task"][action]["status"] == OBSERVED
    assert component["mechanism"][stratum]["status"] == TYPED_MISSING


def test_missing_one_isotropic_task_stratum_makes_action_task_missing_not_zero():
    rows = _rows()
    target = next(row for row in rows if row["mechanism_stratum"] == "common_disk")
    target["task_status"] = TYPED_MISSING
    target["task_missing_reason"] = "NONFINITE_ACTION_ON_FIXED_GT_SUPPORT"
    for key in (
        "task_gain_normalized_px", "harmed_pixel_fraction",
        "pixel_harm_cvar95_raw_px", "severe_event",
    ):
        target[key] = None
    component = _component(_build(rows), target["component_id"])
    value = component["task"]["common_isotropic.local_recoverable_v3"]
    assert value["status"] == TYPED_MISSING
    assert value["normalized_gain"] is None


def test_nonfinite_action_keeps_discrete_severe_event_while_task_is_unestimable():
    rows = _rows()
    target = rows[0]
    target["task_status"] = TYPED_MISSING
    target["task_missing_reason"] = "NONFINITE_ACTION_ON_FIXED_GT_SUPPORT"
    target["task_gain_normalized_px"] = None
    target["harmed_pixel_fraction"] = None
    target["pixel_harm_cvar95_raw_px"] = None
    target["severe_event"] = True
    component = _component(_build(rows), target["component_id"])
    tail = component["tails"][target["mechanism_stratum"]]
    assert tail["status"] == TYPED_MISSING
    assert tail["severe_event"] is True


def test_ordinary_typed_missing_cannot_smuggle_a_severe_event():
    rows = _rows()
    target = rows[0]
    target.update({
        "task_status": TYPED_MISSING,
        "task_missing_reason": "UNSUPPORTED",
        "task_gain_normalized_px": None,
        "harmed_pixel_fraction": None,
        "pixel_harm_cvar95_raw_px": None,
        "severe_event": True,
    })
    with pytest.raises(ValueError, match="nonfinite-action severe"):
        _build(rows)


@pytest.mark.parametrize("mutation", ("duplicate", "wrong_source_count", "bad_action"))
def test_invalid_independence_or_action_identity_fails_closed(mutation):
    rows = _rows()
    if mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "wrong_source_count":
        for row in rows:
            if row["component_id"] == "kitti-21":
                row["source_dataset"] = "spring"
    else:
        rows[0]["action_id"] = "common_motion.local_recoverable_v3"
    with pytest.raises(ValueError):
        _build(rows)


def test_nonexecuted_action_cannot_smuggle_observed_outcomes():
    rows = _rows()
    rows[0]["execution_status"] = TYPED_MISSING
    rows[0]["execution_missing_reason"] = "UNSUPPORTED"
    with pytest.raises(ValueError, match="cannot have observed outcomes"):
        _build(rows)
