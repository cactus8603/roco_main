from __future__ import annotations

import copy
import hashlib

import pytest

from stablebridge.physical_repair.action_qualification_cost_calibration_v1 import (
    ACTION_ORDER,
    PROFILE_SCHEMA,
    SOURCES,
    build_prospective_cost_ceilings_v1,
    select_cost_stress_cases_v1,
)
from stablebridge.physical_repair.action_qualification_joint_ledger_v1 import (
    canonical_sha256,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _terminal(index: int, source: str, action: str, *, available: bool = True):
    case_id = f"{source}-{action}-{index:03d}"
    stress_payload = {
        "image_pixels": 100 + index,
        "read_halo_pixels": 30 + index,
        "support_pixels": 20 + index,
        "modified_endpoint_count": 1 + index % 2,
        "case_id": case_id,
        "selection_order": [
            "image_pixels", "read_halo_pixels", "support_pixels",
            "modified_endpoint_count", "case_id",
        ],
        "task_GT_read": False,
        "outcome_read": False,
    }
    payload = {
        "schema": "e269-development-prediction-terminal/v1",
        "status": "AVAILABLE" if available else "TYPED_MISSING",
        "terminal": True,
        "case_id": case_id,
        "component_id": f"component-{source}-{index:03d}",
        "source_dataset": source,
        "mechanism_stratum": "synthetic",
        "target_public_action_id": action,
        "action_binding_sha256": _sha(case_id) if available else None,
        "stress_selection_metrics": ({
            **stress_payload,
            "stress_key_sha256": canonical_sha256(stress_payload),
        } if available else None),
        "task_GT_read": False,
        "outcome_read": False,
    }
    return {**payload, "semantic_receipt_sha256": canonical_sha256(payload)}


def _terminals():
    rows = []
    # Exact total 280 is required; cycling actions intentionally supplies many
    # candidates so selection must use the key, not input order.
    for source, count in (("spring", 126), ("kitti", 154)):
        for index in range(count):
            rows.append(_terminal(index, source, ACTION_ORDER[index % 6]))
    return rows


def test_pre_gt_stress_selection_returns_twelve_lexicographic_maxima():
    value = select_cost_stress_cases_v1(
        list(reversed(_terminals())), protocol_sha256=_sha("protocol"),
    )
    assert value["selected_count"] == 12
    assert value["all_source_actions_bound"]
    for source in SOURCES:
        for action in ACTION_ORDER:
            row = value["selections"][source][action]
            assert row["status"] == "SELECTED_PRE_GT_MAXIMUM_STRESS"
            assert row["stress_key"][0] == max(
                terminal["stress_selection_metrics"]["image_pixels"]
                for terminal in _terminals()
                if terminal["source_dataset"] == source
                and terminal["target_public_action_id"] == action
            )


def _profiles(selection_sha: str):
    rows = []
    for source in SOURCES:
        for action_index, action in enumerate(ACTION_ORDER):
            for repeat in range(3):
                payload = {
                    "schema": PROFILE_SCHEMA,
                    "source_dataset": source,
                    "action_id": action,
                    "repeat": repeat,
                    "case_id": f"selected-{source}-{action}",
                    "selection_sha256": selection_sha,
                    "protocol_sha256": _sha("protocol"),
                    "cache_state": "cold",
                    "forward_count": 8,
                    "retry_count": 0,
                    "OOM_count": 0,
                    "operator_cpu_seconds": 0.1 + action_index * 0.01,
                    "operator_wall_seconds": 0.2 + repeat * 0.01,
                    "bundle_cpu_seconds": 1.0 + repeat * 0.1,
                    "bundle_wall_seconds": 2.0 + action_index * 0.1,
                    "bytes_moved": 10_000_000 + repeat,
                    "peak_allocated_bytes": 1 << 30,
                    "peak_reserved_bytes": 2 << 30,
                    "task_GT_read": False,
                    "outcome_read": False,
                }
                rows.append({
                    **payload, "profile_sha256": canonical_sha256(payload),
                })
    return rows


def test_36_cold_profiles_produce_six_prospective_ceiling_receipts():
    selection = _sha("selection")
    value = build_prospective_cost_ceilings_v1(
        _profiles(selection), selection_sha256=selection,
        protocol_sha256=_sha("protocol"),
    )
    assert value["profile_count"] == 36
    assert len(value["ceilings"]) == 6
    assert value["status"] == "COMPLETE_PROSPECTIVE_COST_CEILINGS"
    assert all(row["forward_count_upper"] == 8 for row in value["ceilings"].values())
    assert all(row["device_guard_pass"] for row in value["ceilings"].values())


@pytest.mark.parametrize("field,value", (
    ("cache_state", "warm"), ("forward_count", 7),
    ("retry_count", 1), ("OOM_count", 1),
))
def test_profile_hard_gates_fail_closed(field, value):
    selection = _sha("selection")
    rows = _profiles(selection)
    rows[0][field] = value
    payload = {key: item for key, item in rows[0].items() if key != "profile_sha256"}
    rows[0]["profile_sha256"] = canonical_sha256(payload)
    with pytest.raises(ValueError, match="hard execution gate"):
        build_prospective_cost_ceilings_v1(
            rows, selection_sha256=selection, protocol_sha256=_sha("protocol"),
        )


def test_missing_source_action_is_explicitly_unbound_not_imputed():
    rows = _terminals()
    target = ACTION_ORDER[0]
    for row in rows:
        if row["source_dataset"] == "kitti" and row["target_public_action_id"] == target:
            replacement = _terminal(
                int(row["case_id"].rsplit("-", 1)[1]), "kitti", target,
                available=False,
            )
            row.clear()
            row.update(replacement)
    value = select_cost_stress_cases_v1(rows, protocol_sha256=_sha("protocol"))
    assert not value["all_source_actions_bound"]
    assert value["selections"]["kitti"][target]["status"].startswith("UNBOUND")


def test_tampered_terminal_hash_is_rejected():
    rows = _terminals()
    rows[0] = copy.deepcopy(rows[0])
    rows[0]["component_id"] = "tampered"
    with pytest.raises(ValueError, match="invalid prediction terminal"):
        select_cost_stress_cases_v1(rows, protocol_sha256=_sha("protocol"))
