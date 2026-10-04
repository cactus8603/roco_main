import cv2
import numpy as np

from stablebridge.physical_repair.jpeg_action_certificates import (
    jpeg_block_linear,
    jpeg_deblock_action_certificate,
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
    encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )[1]
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def test_supported_jpeg_action_has_boundary_gain_codec_closure_and_exact_footprint():
    observed = jpeg(texture(), 6)
    certificate = jpeg_deblock_action_certificate(
        observed, endpoint="first", estimated_quality=6, presence_lcb=0.1,
    )
    assert certificate.status == "supported"
    assert certificate.competing_model_scores["boundary_reduction_lcb_255"] > 0.0
    assert certificate.competing_model_scores["codec_closure_excess_ub_255"] <= 1.0
    assert certificate.diagnostics["outside_footprint_exact"] == 1.0


def test_presence_failure_rejects_action_even_if_boundary_smoothing_changes_pixels():
    clean = texture()
    certificate = jpeg_deblock_action_certificate(
        clean, endpoint="second", estimated_quality=6, presence_lcb=-0.1,
    )
    assert certificate.status == "rejected"
    assert "jpeg_presence_not_supported" in certificate.rejection_reasons


def test_candidate_that_changes_interior_pixels_fails_declared_footprint():
    observed = jpeg(texture(), 6)
    candidate = jpeg_block_linear(observed)
    candidate[4, 4] = 255 - candidate[4, 4]
    certificate = jpeg_deblock_action_certificate(
        observed, endpoint="first", estimated_quality=6, presence_lcb=0.1,
        candidate=candidate,
    )
    assert certificate.status == "rejected"
    assert "action_changed_outside_declared_footprint" in certificate.rejection_reasons

