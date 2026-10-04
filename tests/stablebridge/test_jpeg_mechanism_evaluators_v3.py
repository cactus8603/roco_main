from __future__ import annotations

import io

import numpy as np
from PIL import Image
import pytest

from stablebridge.physical_repair.jpeg_mechanism_evaluators_v3 import (
    evaluate_jpeg_mechanism_v3,
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
def test_v3_mechanism_is_specific_safe_and_local(policy, action_id):
    observed = _partially_compressed()
    successor = build_local_jpeg_successor_v3(
        observed, policy=policy, endpoint="first", estimated_quality=6,
    )
    receipt = evaluate_jpeg_mechanism_v3(
        action_id=action_id,
        physical_pair_sha256="a" * 64,
        endpoint="first",
        before_endpoint=observed,
        after_endpoint=successor.output_rgb,
        endpoint_support=successor.support,
        estimated_quality=6,
        selected_strength=successor.selected_strength,
    )
    row = receipt.endpoint_records[0]
    assert row["active_relative_reduction"] > 0.05
    assert row["specificity_difference"] > 0.05
    assert row["quantization_violation_max_dct"] <= 4.0
    assert row["outside_support_byte_identity"] is True
    assert receipt.as_dict()["composition_policy"] == (
        "ALIGNED_8PX_BLOCK_EXACT_SELECTION_V1"
    )
    if policy == "codec":
        assert row["codec_closure_excess_max_255"] <= 1.0


def test_v3_evaluator_rejects_post_action_support():
    observed = _partially_compressed()
    successor = build_local_jpeg_successor_v3(
        observed, policy="qcell", endpoint="first", estimated_quality=6,
    )
    changed = np.any(successor.output_rgb != observed, axis=2)
    assert not np.array_equal(changed, successor.support)
    with pytest.raises(RuntimeError, match="support drift"):
        evaluate_jpeg_mechanism_v3(
            action_id=LOCAL_JPEG_QCELL_ACTION_ID_V3,
            physical_pair_sha256="a" * 64,
            endpoint="first",
            before_endpoint=observed,
            after_endpoint=successor.output_rgb,
            endpoint_support=changed,
            estimated_quality=6,
            selected_strength=1.0,
        )
