from __future__ import annotations

import numpy as np
import pytest

from stablebridge.physical_repair.local_case_materialization import (
    LOCAL_CASE_MATERIALIZATION_SCHEMA_V1,
    build_local_case_materialization_v1,
)


def _inputs():
    first = np.zeros((8, 8, 3), dtype=np.uint8)
    second = np.zeros_like(first)
    flow = np.zeros((8, 8, 2), dtype=np.float32)
    support = np.zeros((8, 8), dtype=bool)
    support[2:6, 2:6] = True
    output = first.copy()
    output[2:6, 2:6] = 31
    return first, second, flow, support, output


@pytest.mark.parametrize(
    ("action_id", "control_id", "operator_id"),
    (
        (
            "paired_impulse_median3.endpoint_supported_v1",
            "paired_impulse_median3",
            "impulse_exact_median3",
        ),
        (
            "paired_additive_wiener3.endpoint_supported_v1",
            "paired_additive_wiener3",
            "wiener3",
        ),
    ),
)
def test_local_receipt_proves_outside_identity_and_closes_case_prerequisites(
    action_id, control_id, operator_id,
):
    first, second, flow, support, output = _inputs()
    receipt = build_local_case_materialization_v1(
        qualification_action_id=action_id,
        exact_control_id=control_id,
        operator_id=operator_id,
        observed_first_rgb=first,
        observed_second_rgb=second,
        observed_native_flow=flow,
        output_first_rgb=output,
        output_second_rgb=second,
        endpoint_supports=(support, np.zeros_like(support)),
        modified_endpoints=("first",),
        physical_pair_sha256="a" * 64,
        endpoint_policy_sha256="b" * 64,
    )
    payload = receipt.as_dict()
    assert payload["schema"] == LOCAL_CASE_MATERIALIZATION_SCHEMA_V1
    assert payload["outside_support_byte_identity"] is True
    assert payload["case_binding_missing"] == ["prospective_cost_ceiling_receipt"]
    assert payload["endpoint_records"][0]["changed_pixels"] == 16
    assert payload["endpoint_records"][1]["changed_pixels"] == 0
    prerequisites = receipt.to_case_binding_prerequisites_v2(
        cost_ceiling_receipt_sha256="c" * 64,
    )
    assert prerequisites.case_id == "physical-pair:" + "a" * 64
    assert prerequisites.support_realization_sha256 == receipt.support_realization_sha256


def test_local_receipt_rejects_escape_empty_change_and_wrong_operator():
    first, second, flow, support, output = _inputs()
    escaped = output.copy()
    escaped[0, 0] = 1
    kwargs = dict(
        qualification_action_id="paired_impulse_median3.endpoint_supported_v1",
        exact_control_id="paired_impulse_median3",
        operator_id="impulse_exact_median3",
        observed_first_rgb=first,
        observed_second_rgb=second,
        observed_native_flow=flow,
        output_first_rgb=escaped,
        output_second_rgb=second,
        endpoint_supports=(support, np.zeros_like(support)),
        modified_endpoints=("first",),
        physical_pair_sha256="a" * 64,
        endpoint_policy_sha256="b" * 64,
    )
    with pytest.raises(RuntimeError, match="escaped"):
        build_local_case_materialization_v1(**kwargs)
    kwargs["output_first_rgb"] = first
    with pytest.raises(RuntimeError, match="no acting"):
        build_local_case_materialization_v1(**kwargs)
    kwargs["output_first_rgb"] = output
    kwargs["operator_id"] = "wiener3"
    with pytest.raises(ValueError, match="identity drift"):
        build_local_case_materialization_v1(**kwargs)


def test_local_receipt_is_deterministic_and_role_hashes_allow_identical_images():
    first, second, flow, support, output = _inputs()
    kwargs = dict(
        qualification_action_id="paired_additive_wiener3.endpoint_supported_v1",
        exact_control_id="paired_additive_wiener3",
        operator_id="wiener3",
        observed_first_rgb=first,
        observed_second_rgb=second,
        observed_native_flow=flow,
        output_first_rgb=output,
        output_second_rgb=second,
        endpoint_supports=(support, np.zeros_like(support)),
        modified_endpoints=("first",),
        physical_pair_sha256="a" * 64,
        endpoint_policy_sha256="b" * 64,
    )
    first_receipt = build_local_case_materialization_v1(**kwargs)
    second_receipt = build_local_case_materialization_v1(**kwargs)
    assert first_receipt == second_receipt
    assert len(set(first_receipt.source_input_sha256s)) == 3
