from __future__ import annotations

import hashlib
import json

import pytest

from stablebridge.physical_repair.action_qualification_inference import (
    MECHANISM_STRATA_V1,
    MechanismObservationV1,
    evaluate_wave1_mechanism_inference_v1,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def observations(
    *,
    primary: float = 0.20,
    specificity: float = 0.10,
) -> list[MechanismObservationV1]:
    rows = []
    for stratum in MECHANISM_STRATA_V1:
        for index in range(20):
            source = "spring" if index < 10 else "sintel"
            rows.append(MechanismObservationV1(
                case_id=f"{stratum}:case:{index}",
                component_id=f"scene:{index}",
                source_dataset=source,
                outer_fold=index % 5,
                mechanism_stratum=stratum,
                active_relative_reduction=primary + (index % 3) * 0.001,
                specificity_difference=specificity + (index % 2) * 0.001,
                mechanism_receipt_sha256=sha(f"receipt:{stratum}:{index}"),
            ))
    return rows


def evaluate(rows):
    return evaluate_wave1_mechanism_inference_v1(
        rows,
        required_components_by_stratum={name: 20 for name in MECHANISM_STRATA_V1},
        power_plan_sha256=sha("power plan"),
        cohort_manifest_sha256=sha("cohort"),
        complete_matrix_receipt_sha256=sha("matrix"),
        chronology_receipt_sha256=sha("chronology"),
    )


def test_all_seven_strata_and_five_policies_qualify_deterministically():
    rows = observations()
    first = evaluate(rows)
    second = evaluate(list(reversed(rows)))

    assert first == second
    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert len(first["stratum_results"]) == 7
    assert len(first["policy_results"]) == 5
    assert len(first["holm_tests"]) == 14
    assert first["all_mechanisms_qualified"] is True
    assert all(row["mechanism_qualified"] for row in first["stratum_results"])
    assert first["scientific_qualification"] is False
    assert first["selector_admission"] is False
    json.dumps(first, allow_nan=False)


def test_case_density_cannot_reweight_a_scene_component():
    rows = observations()
    base = evaluate(rows)
    target = rows[0]
    rows.extend([
        MechanismObservationV1(
            case_id=f"extra:{index}",
            component_id=target.component_id,
            source_dataset=target.source_dataset,
            outer_fold=target.outer_fold,
            mechanism_stratum=target.mechanism_stratum,
            active_relative_reduction=target.active_relative_reduction,
            specificity_difference=target.specificity_difference,
            mechanism_receipt_sha256=sha(f"extra:{index}"),
        )
        for index in range(10)
    ])
    dense = evaluate(rows)
    assert (
        base["stratum_results"][0]["primary"]["point"]
        == dense["stratum_results"][0]["primary"]["point"]
    )


def test_insufficient_stratum_fails_closed_without_nan_json():
    rows = [
        row for row in observations()
        if not (
            row.mechanism_stratum == "common_motion"
            and row.component_id == "scene:19"
        )
    ]
    result = evaluate(rows)
    motion = next(
        row for row in result["stratum_results"]
        if row["mechanism_stratum"] == "common_motion"
    )
    blur = next(
        row for row in result["policy_results"]
        if row["policy_id"] == "blur_v3_cross_endpoint.local_recoverable_v2"
    )
    assert motion["sample_audit"]["sample_ready"] is False
    assert motion["primary"]["point"] is None
    assert motion["mechanism_qualified"] is False
    assert blur["mechanism_qualified"] is False
    assert result["all_mechanisms_qualified"] is False
    json.dumps(result, allow_nan=False)


def test_weak_specificity_fails_holm_and_only_its_policy_gate():
    rows = observations()
    rows = [
        MechanismObservationV1(
            **{
                **row.__dict__,
                "specificity_difference": (
                    -0.01 if row.mechanism_stratum == "jpeg_qcell_v3"
                    else row.specificity_difference
                ),
            }
        )
        for row in rows
    ]
    result = evaluate(rows)
    qcell = next(
        row for row in result["stratum_results"]
        if row["mechanism_stratum"] == "jpeg_qcell_v3"
    )
    codec = next(
        row for row in result["stratum_results"]
        if row["mechanism_stratum"] == "jpeg_codec_path_v4"
    )
    assert qcell["mechanism_qualified"] is False
    assert qcell["specificity"]["holm"]["holm_reject"] is False
    assert codec["mechanism_qualified"] is True


def test_component_and_case_identity_drift_are_rejected():
    rows = observations()
    with pytest.raises(ValueError, match="duplicate case"):
        evaluate([*rows, rows[0]])

    drifted = list(rows)
    row = drifted[-1]
    drifted[-1] = MechanismObservationV1(
        **{**row.__dict__, "component_id": "scene:0", "outer_fold": 4}
    )
    with pytest.raises(ValueError, match="component identity drifted"):
        evaluate(drifted)


def test_power_plan_is_complete_and_cannot_drop_below_twenty():
    kwargs = dict(
        observations=observations(),
        power_plan_sha256=sha("power plan"),
        cohort_manifest_sha256=sha("cohort"),
        complete_matrix_receipt_sha256=sha("matrix"),
        chronology_receipt_sha256=sha("chronology"),
    )
    with pytest.raises(ValueError, match="exactly the seven"):
        evaluate_wave1_mechanism_inference_v1(
            required_components_by_stratum={"common_disk": 20}, **kwargs,
        )
    plan = {name: 20 for name in MECHANISM_STRATA_V1}
    plan["common_disk"] = 19
    with pytest.raises(ValueError, match=">= 20"):
        evaluate_wave1_mechanism_inference_v1(
            required_components_by_stratum=plan, **kwargs,
        )
