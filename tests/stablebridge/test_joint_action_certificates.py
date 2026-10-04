import cv2
import numpy as np

from stablebridge.physical_repair.joint_action_certificates import (
    JOINT_ACTIONS,
    joint_action_certificate,
    joint_action_certificates,
)


def texture(height=96, width=128):
    yy, xx = np.mgrid[:height, :width]
    base = 110 + 45 * np.sin(xx / 4.0) + 30 * np.cos(yy / 7.0)
    return np.stack((base, np.roll(base, 3, 1), np.roll(base, 5, 0)), axis=2).clip(
        0, 255
    ).astype(np.uint8)


def test_joint_certificates_return_six_observable_action_records():
    first = texture()
    second = first.copy()
    flow = np.zeros((*first.shape[:2], 2), np.float32)
    records = joint_action_certificates(first, second, flow)
    assert tuple(records) == tuple(f"{operator}@{endpoint}"
                                   for operator, endpoint in JOINT_ACTIONS)
    assert all(value.action.operator_version == "v3-joint-response"
               for value in records.values())
    assert all(value.status != "supported" for value in records.values())


def test_impulse_action_has_positive_response_on_isolated_endpoint_outliers():
    clean = texture(128, 160)
    corrupt = clean.copy()
    rng = np.random.default_rng(20260930)
    mask = rng.random(clean.shape[:2]) < 0.075
    salt = rng.random(clean.shape[:2]) < 0.5
    corrupt[mask & salt] = 255
    corrupt[mask & ~salt] = 0
    flow = np.zeros((*clean.shape[:2], 2), np.float32)
    certificate = joint_action_certificate(
        corrupt, clean, flow,
        operator_id="impulse_exact_median3", endpoint="first",
    )
    assert certificate.estimated_parameters["combined_residual_gain"] > 0.0
    assert certificate.estimated_parameters["rgb_residual_gain"] > 0.0
    assert certificate.diagnostics["effective_pixels"] >= 32


def test_joint_certificate_reports_warp_competitor_for_misalignment():
    first = texture()
    matrix = np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], np.float32)
    second = cv2.warpAffine(first, matrix, (first.shape[1], first.shape[0]),
                            borderMode=cv2.BORDER_REFLECT_101)
    flow = np.zeros((*first.shape[:2], 2), np.float32)
    certificate = joint_action_certificate(
        first, second, flow, operator_id="wiener3", endpoint="first",
    )
    assert certificate.competing_model_scores["local_warp_offset_gain"] >= 0.0
    assert "affine_radiometry_gain" in certificate.competing_model_scores

