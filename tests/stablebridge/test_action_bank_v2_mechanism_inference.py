from __future__ import annotations

import hashlib

from stablebridge.physical_repair.action_bank_v2_mechanism_inference import (
    ACTION_TO_STRATA_V2,
    MECHANISM_METRIC_BY_STRATUM_V2,
    evaluate_action_bank_v2_mechanisms,
)
from stablebridge.physical_repair.action_qualification_inference import (
    MECHANISM_STRATA_V1,
    MechanismObservationV1,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def observations(*, weak: str | None = None):
    rows = []
    for stratum in MECHANISM_STRATA_V1:
        for index in range(20):
            rows.append(MechanismObservationV1(
                case_id=f"{stratum}:{index}",
                component_id=f"component:{index}",
                source_dataset="spring" if index < 10 else "kitti",
                outer_fold=index % 5,
                mechanism_stratum=stratum,
                active_relative_reduction=(
                    -0.1 if stratum == weak else 0.2 + index / 10000
                ),
                specificity_difference=(
                    -0.1 if stratum == weak else 0.1 + index / 10000
                ),
                mechanism_receipt_sha256=sha(f"receipt:{stratum}:{index}"),
            ))
    return rows


def evaluate(rows):
    return evaluate_action_bank_v2_mechanisms(
        rows,
        required_components_by_stratum={name: 20 for name in MECHANISM_STRATA_V1},
        power_plan_sha256=sha("power"),
        cohort_manifest_sha256=sha("cohort"),
        complete_matrix_receipt_sha256=sha("matrix"),
        chronology_receipt_sha256=sha("chronology"),
        action_bank_sha256=sha("bank"),
    )


def test_exact_e262_six_action_mapping_is_deterministic():
    first = evaluate(observations())
    second = evaluate(list(reversed(observations())))
    assert first == second
    assert len(first["action_results"]) == 6
    assert {row["action_id"] for row in first["action_results"]} == set(
        ACTION_TO_STRATA_V2
    )
    assert "blur_v3_cross_endpoint.local_recoverable_v2" not in ACTION_TO_STRATA_V2
    assert first["all_public_action_mechanisms_qualified"] is True
    metrics = {
        row["mechanism_stratum"]: row["metric_id"]
        for row in first["stratum_results"]
    }
    assert metrics == MECHANISM_METRIC_BY_STRATUM_V2
    assert metrics["common_disk"] == metrics["common_gaussian"]
    assert metrics["jpeg_qcell_v3"].endswith("_v3")
    assert first["scientific_qualification"] is False
    assert first["selector_admission"] is False


def test_gaussian_failure_blocks_isotropic_but_not_motion():
    value = evaluate(observations(weak="common_gaussian"))
    actions = {row["action_id"]: row for row in value["action_results"]}
    assert actions["common_isotropic.local_recoverable_v3"][
        "mechanism_qualified"
    ] is False
    assert actions["common_motion.local_recoverable_v3"][
        "mechanism_qualified"
    ] is True
    assert value["all_public_action_mechanisms_qualified"] is False


def test_motion_failure_does_not_block_isotropic():
    value = evaluate(observations(weak="common_motion"))
    actions = {row["action_id"]: row for row in value["action_results"]}
    assert actions["common_isotropic.local_recoverable_v3"][
        "mechanism_qualified"
    ] is True
    assert actions["common_motion.local_recoverable_v3"][
        "mechanism_qualified"
    ] is False
