from __future__ import annotations

import io

import numpy as np
from PIL import Image
import pytest

from stablebridge.physical_repair.action_bank_contracts import ActionEndpointV2
from stablebridge.physical_repair.local_jpeg_case_materialization_v3 import (
    LOCAL_JPEG_CASE_SCHEMA_V3,
    build_local_jpeg_case_materialization_v3,
)
from stablebridge.physical_repair.local_jpeg_successors_v3 import (
    LOCAL_JPEG_CODEC_ACTION_ID_V3,
    LOCAL_JPEG_QCELL_ACTION_ID_V3,
    build_local_jpeg_successor_v3,
)


def _texture(size: int = 384) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    value = (
        110 + 35 * np.sin(xx / 3.7) + 29 * np.cos(yy / 5.3)
        + 18 * np.sin((xx + yy) / 8.1)
        + 12 * ((xx // 11 + yy // 13) % 2)
    )
    return np.stack(
        (value, np.roll(value, 5, 1), np.roll(value, 7, 0)), axis=2,
    ).clip(0, 255).astype(np.uint8)


def _jpeg(image: np.ndarray, quality: int = 6) -> np.ndarray:
    stream = io.BytesIO()
    Image.fromarray(image).save(
        stream, format="JPEG", quality=quality, subsampling=2,
    )
    stream.seek(0)
    return np.asarray(Image.open(stream).convert("RGB"), dtype=np.uint8)


def _partially_compressed() -> np.ndarray:
    clean = _texture()
    observed = clean.copy()
    observed[:, :192] = _jpeg(clean)[:, :192]
    return observed


@pytest.mark.parametrize(
    ("policy", "action_id"),
    (
        ("qcell", LOCAL_JPEG_QCELL_ACTION_ID_V3),
        ("codec", LOCAL_JPEG_CODEC_ACTION_ID_V3),
    ),
)
def test_jpeg_v3_case_binds_local_write_and_full_proposal_read(policy, action_id):
    first = _partially_compressed()
    second = _texture()
    flow = np.zeros((*first.shape[:2], 2), dtype=np.float32)
    successor = build_local_jpeg_successor_v3(
        first, policy=policy, endpoint="first", estimated_quality=6,
    )
    receipt = build_local_jpeg_case_materialization_v3(
        action_id=action_id,
        case_id=f"case-{policy}",
        physical_pair_sha256="a" * 64,
        endpoint_policy_sha256="b" * 64,
        observed_first_rgb=first,
        observed_second_rgb=second,
        observed_native_flow=flow,
        successors={"first": successor},
    )
    payload = receipt.as_dict()
    assert payload["schema"] == LOCAL_JPEG_CASE_SCHEMA_V3
    assert receipt.exact_endpoint == "first"
    first_row, second_row = payload["endpoint_records"]
    assert first_row["modified"] is True
    assert first_row["write_support_pixels"] < first_row["read_support_pixels"]
    assert first_row["read_support_pixels"] == first.shape[0] * first.shape[1]
    assert second_row["modified"] is False
    assert second_row["write_support_pixels"] == 0
    assert payload["child_observables_ready"] is False
    prerequisites = receipt.to_case_binding_prerequisites_v2(
        cost_ceiling_receipt_sha256="c" * 64,
    )
    assert prerequisites.exact_endpoint is ActionEndpointV2.FIRST


def test_jpeg_v3_case_can_bind_both_endpoints_and_is_deterministic():
    first = _partially_compressed()
    second = _partially_compressed()
    flow = np.zeros((*first.shape[:2], 2), dtype=np.float32)
    successors = {
        endpoint: build_local_jpeg_successor_v3(
            image, policy="qcell", endpoint=endpoint, estimated_quality=6,
        )
        for endpoint, image in (("first", first), ("second", second))
    }
    kwargs = dict(
        action_id=LOCAL_JPEG_QCELL_ACTION_ID_V3,
        case_id="case-both",
        physical_pair_sha256="a" * 64,
        endpoint_policy_sha256="b" * 64,
        observed_first_rgb=first,
        observed_second_rgb=second,
        observed_native_flow=flow,
        successors=successors,
    )
    left = build_local_jpeg_case_materialization_v3(**kwargs)
    right = build_local_jpeg_case_materialization_v3(**kwargs)
    assert left == right
    assert left.exact_endpoint == "both"


def test_jpeg_v3_case_rejects_endpoint_drift():
    first = _partially_compressed()
    second = _texture()
    flow = np.zeros((*first.shape[:2], 2), dtype=np.float32)
    successor = build_local_jpeg_successor_v3(
        first, policy="qcell", endpoint="first", estimated_quality=6,
    )
    with pytest.raises(ValueError, match="endpoint drift"):
        build_local_jpeg_case_materialization_v3(
            action_id=LOCAL_JPEG_QCELL_ACTION_ID_V3,
            case_id="case-drift",
            physical_pair_sha256="a" * 64,
            endpoint_policy_sha256="b" * 64,
            observed_first_rgb=first,
            observed_second_rgb=second,
            observed_native_flow=flow,
            successors={"second": successor},
        )
