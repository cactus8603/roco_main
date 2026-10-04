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
    canonical_sha256,
)
from stablebridge.physical_repair.action_qualification_power_v1 import (
    REPEATS,
    SEED,
    evaluate_joint_power_v1,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _joint(*, impulse_missing_fraction: float = 0.0, one_nonfinite: bool = False):
    rows = []
    inverse = {
        stratum: action
        for action, strata in ACTION_TO_STRATA.items()
        for stratum in strata
    }
    for source, count in (("spring", 18), ("kitti", 22)):
        for index in range(count):
            component = f"{source}-{index:02d}"
            residual = ((index % 5) - 2) * 0.0001
            for stratum in STRATUM_ORDER:
                case_id = f"{component}-{stratum}"
                missing = (
                    stratum == "paired_impulse_median3"
                    and index < round(count * impulse_missing_fraction)
                )
                nonfinite = (
                    one_nonfinite and source == "spring" and index == 0
                    and stratum == "paired_impulse_median3"
                )
                rows.append({
                    "schema": CASE_SCHEMA,
                    "case_id": case_id,
                    "component_id": component,
                    "source_dataset": source,
                    "outer_fold": index % 5,
                    "mechanism_stratum": stratum,
                    "action_id": inverse[stratum],
                    "execution_status": TYPED_MISSING if missing else OBSERVED,
                    "execution_missing_reason": "UNSUPPORTED" if missing else None,
                    "task_status": TYPED_MISSING if (missing or nonfinite) else OBSERVED,
                    "task_missing_reason": (
                        "UNSUPPORTED" if missing else (
                            "NONFINITE_ACTION_ON_FIXED_GT_SUPPORT" if nonfinite else None
                        )
                    ),
                    "task_gain_normalized_px": None if (missing or nonfinite) else 0.02 + residual,
                    "harmed_pixel_fraction": None if (missing or nonfinite) else 0.025 + residual,
                    "pixel_harm_cvar95_raw_px": None if (missing or nonfinite) else 0.125 + residual,
                    "severe_event": True if nonfinite else (None if missing else False),
                    "mechanism_status": TYPED_MISSING if missing else OBSERVED,
                    "mechanism_missing_reason": "UNSUPPORTED" if missing else None,
                    "mechanism_primary": None if missing else 0.10 + residual,
                    "mechanism_specificity": None if missing else 0.05 + residual,
                    "cost_gate_pass": not missing,
                    "child_observables_complete": True,
                    "case_receipt_sha256": _sha(case_id),
                })
    return build_component_joint_ledger_v1(
        rows,
        protocol_sha256=_sha("protocol"),
        component_roster_sha256=_sha("roster"),
    )


def test_joint_primary_simulation_is_deterministic_hash_bound_and_non_authoritative():
    joint = _joint()
    first = evaluate_joint_power_v1(joint, protocol_sha256=_sha("protocol"))
    second = evaluate_joint_power_v1(joint, protocol_sha256=_sha("protocol"))
    assert first == second
    payload = {key: value for key, value in first.items() if key != "power_result_sha256"}
    assert first["power_result_sha256"] == canonical_sha256(payload)
    assert first["simulation"]["repeats"] == REPEATS
    assert first["simulation"]["seed"] == SEED
    assert len(first["admitted_action_ids_for_fixed_E263_cohort"]) == 6
    assert not first["fresh_scientific_qualification"]
    assert not first["selector_admission"]


def test_missingness_is_resampled_and_can_deny_only_the_affected_action():
    result = evaluate_joint_power_v1(
        _joint(impulse_missing_fraction=0.8), protocol_sha256=_sha("protocol"),
    )
    impulse = "paired_impulse_median3.endpoint_supported_v1"
    assert result["per_action"][impulse]["adoption_probability"] < 0.8
    assert impulse not in result["admitted_action_ids_for_fixed_E263_cohort"]
    assert any(
        row["meets_0p80_power_admission"]
        for action, row in result["per_action"].items() if action != impulse
    )


def test_all_six_probability_is_report_only_not_the_per_action_rule():
    result = evaluate_joint_power_v1(_joint(), protocol_sha256=_sha("protocol"))
    assert result["all_six_probability_is_not_an_all_or_nothing_gate"]
    assert all(
        row["meets_0p80_power_admission"]
        == (row["adoption_probability"] >= 0.8)
        for row in result["per_action"].values()
    )


def test_nonfinite_task_missingness_remains_a_severe_gate_event():
    result = evaluate_joint_power_v1(
        _joint(one_nonfinite=True), protocol_sha256=_sha("protocol"),
    )
    impulse = "paired_impulse_median3.endpoint_supported_v1"
    assert result["per_action"][impulse]["gate_pass_rates"]["severe_zero"] < 1.0
    assert result["per_action"][impulse]["adoption_probability"] < 0.8
    assert all(
        row["meets_0p80_power_admission"]
        for action, row in result["per_action"].items() if action != impulse
    )


def test_tampered_component_or_manifest_fails_closed():
    joint = _joint()
    broken = copy.deepcopy(joint)
    broken["components"][0]["source_dataset"] = "spring"
    with pytest.raises(ValueError, match="semantic hash drift"):
        evaluate_joint_power_v1(broken, protocol_sha256=_sha("protocol"))
    broken = copy.deepcopy(joint)
    broken["manifest"]["scientific_n"] = 280
    with pytest.raises(ValueError, match="manifest semantic hash drift"):
        evaluate_joint_power_v1(broken, protocol_sha256=_sha("protocol"))


def test_repeats_and_seed_are_frozen():
    joint = _joint()
    with pytest.raises(ValueError, match="10,000"):
        evaluate_joint_power_v1(
            joint, protocol_sha256=_sha("protocol"), repeats=9999,
        )
    with pytest.raises(ValueError, match="seed"):
        evaluate_joint_power_v1(
            joint, protocol_sha256=_sha("protocol"), seed=1,
        )
