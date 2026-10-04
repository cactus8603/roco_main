from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import cv2
import numpy as np
import pytest

from stablebridge.physical_repair.action_bank_contracts import ActionEndpointV2
from stablebridge.physical_repair.local_blur_successor_v3 import (
    build_local_blur_successor_v3,
)
from stablebridge.physical_repair.local_case_materialization_v2 import (
    LOCAL_CASE_MATERIALIZATION_SCHEMA_V2,
    build_local_case_materialization_v2,
)


def _texture(size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    value = (
        110 + 35 * np.sin(xx / 3.7) + 29 * np.cos(yy / 5.3)
        + 18 * np.sin((xx + yy) / 8.1)
        + 12 * ((xx // 11 + yy // 13) % 2)
    )
    return np.stack(
        (value, np.roll(value, 5, 1), np.roll(value, 7, 0)), axis=2,
    ).clip(0, 255).astype(np.uint8)


def _disk(image: np.ndarray, radius: int = 6) -> np.ndarray:
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _motion(image: np.ndarray, length: int = 11) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2] = 1.0 / length
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


@lru_cache(maxsize=None)
def _case(family: str):
    clean = _texture()
    if family == "disk":
        first = _disk(clean)
    elif family == "gaussian":
        first = cv2.GaussianBlur(clean, (0, 0), 2.0)
    elif family == "motion":
        first = _motion(clean)
    else:  # pragma: no cover
        raise ValueError(family)
    flow = np.zeros((*clean.shape[:2], 2), dtype=np.float32)
    successor = build_local_blur_successor_v3(first, clean, flow)
    return clean, first, flow, successor


@pytest.mark.parametrize("family", ("disk", "gaussian", "motion"))
def test_receipt_binds_parameter_write_read_transport_and_rollback(family):
    clean, first, flow, successor = _case(family)
    receipt = build_local_case_materialization_v2(
        case_id=f"case-{family}",
        physical_pair_sha256="a" * 64,
        endpoint_policy_sha256="b" * 64,
        observed_first_rgb=first,
        observed_second_rgb=clean,
        observed_native_flow=flow,
        successor=successor,
    )
    payload = receipt.as_dict()
    assert payload["schema"] == LOCAL_CASE_MATERIALIZATION_SCHEMA_V2
    assert payload["outside_write_support_byte_identity"] is True
    assert payload["case_binding_missing"] == [
        "prospective_cost_ceiling_receipt"
    ]
    assert payload["child_observables_missing"] == [
        "region_projected_CC_RR_CR_RC_receipt"
    ]
    modified = [row for row in payload["endpoint_records"] if row["modified"]]
    assert len(modified) == 1
    assert modified[0]["changed_pixels"] > 0
    assert modified[0]["read_support_pixels"] >= modified[0]["write_support_pixels"]
    assert receipt.exact_parameter_sha256
    prerequisites = receipt.to_case_binding_prerequisites_v2(
        cost_ceiling_receipt_sha256="c" * 64,
    )
    assert prerequisites.case_id == f"case-{family}"
    assert prerequisites.exact_endpoint is ActionEndpointV2.SECOND


def test_receipt_is_deterministic():
    clean, first, flow, successor = _case("gaussian")
    kwargs = dict(
        case_id="case-gaussian",
        physical_pair_sha256="a" * 64,
        endpoint_policy_sha256="b" * 64,
        observed_first_rgb=first,
        observed_second_rgb=clean,
        observed_native_flow=flow,
        successor=successor,
    )
    assert build_local_case_materialization_v2(
        **kwargs
    ) == build_local_case_materialization_v2(**kwargs)


def test_receipt_rejects_source_and_read_halo_drift():
    clean, first, flow, successor = _case("motion")
    changed_first = first.copy()
    changed_first[0, 0, 0] ^= 1
    with pytest.raises(RuntimeError, match="source input drift"):
        build_local_case_materialization_v2(
            case_id="case-motion",
            physical_pair_sha256="a" * 64,
            endpoint_policy_sha256="b" * 64,
            observed_first_rgb=changed_first,
            observed_second_rgb=clean,
            observed_native_flow=flow,
            successor=successor,
        )

    bad_read = successor.second_read_support.copy()
    bad_read[successor.second_support] = False
    tampered = replace(successor, second_read_support=bad_read)
    with pytest.raises(RuntimeError, match="write escaped read halo"):
        build_local_case_materialization_v2(
            case_id="case-motion",
            physical_pair_sha256="a" * 64,
            endpoint_policy_sha256="b" * 64,
            observed_first_rgb=first,
            observed_second_rgb=clean,
            observed_native_flow=flow,
            successor=tampered,
        )
