from __future__ import annotations

import io

import numpy as np
from PIL import Image
import pytest

from stablebridge.physical_repair.local_jpeg_successors import (
    LOCAL_JPEG_CODEC_ACTION_ID_V2,
    LOCAL_JPEG_QCELL_ACTION_ID_V2,
    build_local_jpeg_successor_v2,
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


def _partially_compressed() -> tuple[np.ndarray, np.ndarray]:
    clean = _texture()
    observed = clean.copy()
    observed[:, :192] = _jpeg(clean)[:, :192]
    return observed, clean


@pytest.mark.parametrize(
    ("policy", "action_id"),
    (
        ("qcell", LOCAL_JPEG_QCELL_ACTION_ID_V2),
        ("codec", LOCAL_JPEG_CODEC_ACTION_ID_V2),
    ),
)
def test_before_only_macroblock_support_localizes_both_jpeg_policies(
    policy, action_id,
):
    observed, clean = _partially_compressed()
    result = build_local_jpeg_successor_v2(
        observed, policy=policy, endpoint="first", estimated_quality=6,
    )
    assert result.status == "EXECUTED_LOCAL_MACROBLOCK_V2"
    assert result.action_id == action_id
    assert len(result.active_macro_tiles) == 18
    assert int(result.support.sum()) == 18 * 64 * 64
    assert int(result.support.sum()) < observed.shape[0] * observed.shape[1]
    assert np.any(result.output_rgb[result.support] != observed[result.support])
    assert np.array_equal(
        result.output_rgb[~result.support], observed[~result.support],
    )
    assert result.receipt["quantization_violation_max_dct"] <= 4.0
    assert result.receipt["support_policy"]["pre_action_only"] is True


def test_clean_image_has_no_authorized_macroblock_action():
    clean = _texture()
    result = build_local_jpeg_successor_v2(
        clean, policy="qcell", endpoint="second", estimated_quality=6,
    )
    assert result.status == "UNSUPPORTED_INSUFFICIENT_ACTIVE_MACRO_TILES"
    assert not result.support.any()
    assert np.array_equal(result.output_rgb, clean)
