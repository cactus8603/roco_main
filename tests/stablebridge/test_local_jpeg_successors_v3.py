from __future__ import annotations

import io

import numpy as np
from PIL import Image
import pytest

from stablebridge.physical_repair.jpeg_codec_path_actions import (
    _quantization_violation,
    jpeg_codec_path_action,
)
from stablebridge.physical_repair.jpeg_quantization_actions import (
    DCT_ROUNDING_BOUND,
    jpeg_quantization_projected_deblock,
)
from stablebridge.physical_repair.local_jpeg_successors_v3 import (
    LOCAL_JPEG_CODEC_ACTION_ID_V3,
    LOCAL_JPEG_QCELL_ACTION_ID_V3,
    _block_exact_composite,
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
def test_v3_is_local_block_exact_and_quantization_safe(policy, action_id):
    observed = _partially_compressed()
    result = build_local_jpeg_successor_v3(
        observed, policy=policy, endpoint="first", estimated_quality=6,
    )
    assert result.status == "EXECUTED_LOCAL_BLOCK_EXACT_V3"
    assert result.action_id == action_id
    assert result.receipt["composition_policy"]["pixel_feather_sigma_px"] == 0.0
    assert result.receipt["outside_support_byte_identity"] is True
    assert np.any(result.output_rgb[result.support] != observed[result.support])
    assert np.array_equal(result.output_rgb[~result.support], observed[~result.support])
    assert (
        _quantization_violation(observed, result.output_rgb, 6)
        <= DCT_ROUNDING_BOUND
    )

    proposal = (
        jpeg_quantization_projected_deblock(
            observed, estimated_quality=6,
        ).image
        if policy == "qcell"
        else jpeg_codec_path_action(
            observed, estimated_quality=6,
        ).image
    )
    height = (observed.shape[0] // 8) * 8
    width = (observed.shape[1] // 8) * 8
    for y0 in range(0, height, 8):
        for x0 in range(0, width, 8):
            output_block = result.output_rgb[y0:y0 + 8, x0:x0 + 8]
            observed_block = observed[y0:y0 + 8, x0:x0 + 8]
            proposal_block = proposal[y0:y0 + 8, x0:x0 + 8]
            assert (
                np.array_equal(output_block, observed_block)
                or np.array_equal(output_block, proposal_block)
            )


def test_v3_rejects_support_that_splits_a_dct_block():
    observed = _texture(128)
    proposal = observed.copy()
    proposal[0, 0] = 0
    support = np.zeros(observed.shape[:2], dtype=bool)
    support[0, 0] = True
    with pytest.raises(RuntimeError, match="split an 8x8 DCT block"):
        _block_exact_composite(observed, proposal, support)


def test_v3_clean_input_abstains_without_acting_support():
    clean = _texture()
    result = build_local_jpeg_successor_v3(
        clean, policy="qcell", endpoint="second", estimated_quality=6,
    )
    assert result.status == "UNSUPPORTED_INSUFFICIENT_ACTIVE_MACRO_TILES"
    assert not result.support.any()
    assert np.array_equal(result.output_rgb, clean)
