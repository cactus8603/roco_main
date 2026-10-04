from __future__ import annotations

import pytest

from stablebridge.physical_repair.action_qualification_mechanism_estimand_v2 import (
    reduce_mechanism_endpoints_v2,
)


def _record(endpoint: str, active: float, specificity: float) -> dict:
    return {
        "endpoint": endpoint,
        "outside_support_byte_identity": True,
        "active_relative_reduction": active,
        "specificity_difference": specificity,
    }


@pytest.mark.parametrize(
    "stratum", ("common_disk", "common_gaussian", "common_motion"),
)
def test_cross_endpoint_blur_scores_executed_equalization_endpoint(stratum):
    value = reduce_mechanism_endpoints_v2(
        mechanism_stratum=stratum,
        generator_expected_endpoints=("first",),
        endpoint_records=(_record("second", 0.5, 0.4),),
        identity_proof=lambda endpoint: endpoint == "first",
    )
    assert value["mechanism_estimand_endpoint_basis"] == (
        "EXECUTED_CROSS_ENDPOINT_EQUALIZATION"
    )
    assert value["generator_expected_endpoints"] == ["first"]
    assert value["executed_endpoints"] == ["second"]
    assert value["evaluated_endpoints"] == ["second"]
    assert value["structural_zero_target_endpoints"] == []
    assert value["off_generator_endpoint_writes"] == ["second"]
    assert value["primary"] == 0.5
    assert value["specificity"] == 0.4


def test_generator_targeted_action_keeps_hash_proven_structural_zero():
    value = reduce_mechanism_endpoints_v2(
        mechanism_stratum="paired_impulse_median3",
        generator_expected_endpoints=("first",),
        endpoint_records=(_record("second", 0.5, 0.4),),
        identity_proof=lambda endpoint: endpoint == "first",
    )
    assert value["mechanism_estimand_endpoint_basis"] == "GENERATOR_TARGETED_REPAIR"
    assert value["evaluated_endpoints"] == ["first"]
    assert value["structural_zero_target_endpoints"] == ["first"]
    assert value["primary"] == 0.0
    assert value["specificity"] == 0.0
    assert value["executed_diagnostic_primary"] == 0.5


def test_generator_targeted_missing_endpoint_requires_identity_proof():
    with pytest.raises(ValueError, match="byte-identity proof"):
        reduce_mechanism_endpoints_v2(
            mechanism_stratum="jpeg_qcell_v3",
            generator_expected_endpoints=("first",),
            endpoint_records=(_record("second", 0.5, 0.4),),
            identity_proof=lambda endpoint: False,
        )


def test_cross_endpoint_blur_rejects_more_than_one_executed_endpoint():
    with pytest.raises(ValueError, match="one executed endpoint"):
        reduce_mechanism_endpoints_v2(
            mechanism_stratum="common_motion",
            generator_expected_endpoints=("first",),
            endpoint_records=(
                _record("first", 0.2, 0.1),
                _record("second", 0.5, 0.4),
            ),
            identity_proof=lambda endpoint: True,
        )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), True))
def test_nonfinite_or_boolean_mechanism_value_is_rejected(value):
    with pytest.raises(ValueError, match="must be finite"):
        reduce_mechanism_endpoints_v2(
            mechanism_stratum="common_disk",
            generator_expected_endpoints=("first",),
            endpoint_records=(_record("second", value, 0.4),),
            identity_proof=lambda endpoint: True,
        )
