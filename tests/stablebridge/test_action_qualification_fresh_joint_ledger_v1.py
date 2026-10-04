from __future__ import annotations

import copy
import hashlib

import pytest

from stablebridge.physical_repair.action_qualification_fresh_joint_ledger_v1 import (
    EXPECTED_FOLDS_BY_SOURCE,
    REQUIRED_EVIDENCE_BINDINGS,
    build_fresh_component_joint_ledger_v1,
)
from stablebridge.physical_repair.action_qualification_joint_ledger_v1 import (
    CASE_SCHEMA,
    STRATUM_TO_ACTION,
    STRATUM_ORDER,
    canonical_sha256,
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
                    "task_gain_normalized_px": 0.02,
                    "harmed_pixel_fraction": 0.01,
                    "pixel_harm_cvar95_raw_px": 0.05,
                    "severe_event": False,
                    "mechanism_status": "OBSERVED",
                    "mechanism_missing_reason": None,
                    "mechanism_primary": 0.10,
                    "mechanism_specificity": 0.05,
                    "cost_gate_pass": True,
                    "child_observables_complete": True,
                    "case_receipt_sha256": _sha(case),
                })
    return rows


def test_fresh_161_cases_reduce_to_23_components_and_bind_every_package():
    result = build_fresh_component_joint_ledger_v1(
        _rows(), roster_sha256=_sha("roster"), evidence_bindings=_bindings(),
    )
    manifest = result["manifest"]
    assert manifest["case_count"] == 161
    assert manifest["component_count"] == manifest["scientific_n"] == 23
    assert manifest["source_component_counts"] == {"spring": 18, "kitti": 5}
    assert manifest["fold_counts_by_source"] == EXPECTED_FOLDS_BY_SOURCE
    assert manifest["evidence_bindings"] == _bindings()
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    assert manifest["manifest_sha256"] == canonical_sha256(payload)


def test_common_isotropic_is_one_action_mean_not_two_independent_samples():
    rows = _rows()
    for row in rows:
        if row["component_id"] == "spring-00" and row["mechanism_stratum"] == "common_disk":
            row["task_gain_normalized_px"] = 0.01
        if row["component_id"] == "spring-00" and row["mechanism_stratum"] == "common_gaussian":
            row["task_gain_normalized_px"] = 0.03
    result = build_fresh_component_joint_ledger_v1(
        rows, roster_sha256=_sha("roster"), evidence_bindings=_bindings(),
    )
    component = next(row for row in result["components"] if row["component_id"] == "spring-00")
    iso = component["task"]["common_isotropic.local_recoverable_v3"]
    assert iso["normalized_gain"] == pytest.approx(0.02)
    assert iso["owned_strata"] == ["common_disk", "common_gaussian"]


def test_missing_one_isotropic_internal_stratum_makes_public_task_missing():
    rows = _rows()
    row = next(
        value for value in rows
        if value["component_id"] == "spring-00"
        and value["mechanism_stratum"] == "common_disk"
    )
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
    result = build_fresh_component_joint_ledger_v1(
        rows, roster_sha256=_sha("roster"), evidence_bindings=_bindings(),
    )
    component = next(value for value in result["components"] if value["component_id"] == "spring-00")
    assert component["task"]["common_isotropic.local_recoverable_v3"]["status"] == "TYPED_MISSING"
    assert not component["action_gates"]["common_isotropic.local_recoverable_v3"]["all_executed"]


def test_wrong_fold_allocation_and_binding_set_fail_closed():
    rows = _rows()
    rows[0]["outer_fold"] = 1
    with pytest.raises(ValueError, match="source/fold identity drift"):
        build_fresh_component_joint_ledger_v1(
            rows, roster_sha256=_sha("roster"), evidence_bindings=_bindings(),
        )
    bindings = _bindings()
    bindings.pop("chronology_sha256")
    with pytest.raises(ValueError, match="bindings"):
        build_fresh_component_joint_ledger_v1(
            _rows(), roster_sha256=_sha("roster"), evidence_bindings=bindings,
        )


def test_duplicate_and_partial_cases_fail_closed():
    rows = _rows()
    with pytest.raises(ValueError, match="exactly 161"):
        build_fresh_component_joint_ledger_v1(
            rows[:-1], roster_sha256=_sha("roster"), evidence_bindings=_bindings(),
        )
    broken = copy.deepcopy(rows)
    broken[-1] = copy.deepcopy(broken[0])
    with pytest.raises(ValueError, match="duplicate"):
        build_fresh_component_joint_ledger_v1(
            broken, roster_sha256=_sha("roster"), evidence_bindings=_bindings(),
        )
