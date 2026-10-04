import io

import numpy as np
from PIL import Image

from stablebridge.physical_repair.jpeg_quantization_actions import (
    DCT_ROUNDING_BOUND,
    jpeg_quantization_action_certificate,
    jpeg_quantization_projected_deblock,
)


def texture(size=384):
    yy, xx = np.mgrid[:size, :size]
    value = (
        110 + 35 * np.sin(xx / 3.7) + 29 * np.cos(yy / 5.3)
        + 18 * np.sin((xx + yy) / 8.1) + 12 * ((xx // 11 + yy // 13) % 2)
    )
    return np.stack((value, np.roll(value, 5, 1), np.roll(value, 7, 0)), 2).clip(
        0, 255
    ).astype(np.uint8)


def jpeg(image, quality=6):
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="JPEG", quality=quality, subsampling=2)
    stream.seek(0)
    return np.asarray(Image.open(stream).convert("RGB"), dtype=np.uint8)


def test_projected_action_stays_within_decoder_rounding_allowance():
    observed = jpeg(texture(), 6)
    action = jpeg_quantization_projected_deblock(observed, estimated_quality=6)
    assert action.quantization_violation_max <= DCT_ROUNDING_BOUND
    assert np.any(action.image != observed)
    assert action.iterations >= 1


def test_projected_action_certificate_is_supported_on_strong_jpeg():
    observed = jpeg(texture(), 6)
    certificate = jpeg_quantization_action_certificate(
        observed, endpoint="first", estimated_quality=6, presence_lcb=0.1,
    )
    assert certificate.status == "supported", certificate.rejection_reasons
    assert certificate.competing_model_scores["boundary_reduction_lcb_255"] > 0.0
    assert certificate.competing_model_scores["codec_closure_excess_ub_255"] <= 1.0


def test_negative_presence_still_rejects_projected_action():
    clean = texture()
    certificate = jpeg_quantization_action_certificate(
        clean, endpoint="second", estimated_quality=6, presence_lcb=-0.1,
    )
    assert certificate.status == "rejected"
    assert "jpeg_presence_not_supported" in certificate.rejection_reasons

