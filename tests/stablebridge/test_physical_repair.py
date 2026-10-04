import cv2
import numpy as np
import pytest

from stablebridge.physical_repair import (
    ActionSpec,
    CommonPassbandConfig,
    SupportMaps,
    apply_local_image_action,
    blend_local_proposal,
    bounded_output,
    common_passband_certificate,
    additive_noise_certificate,
    impulse_noise_certificate,
    paired_impulse_certificates,
    observable_endpoint_applicability,
    observable_support_maps,
    transport_flow_mask_to_second,
    transport_second_mask_to_flow,
    zoom_common_passband_certificate,
)


def texture(height=96, width=128):
    yy, xx = np.mgrid[:height, :width]
    value = 110 + 45 * np.sin(xx / 4.0) + 30 * np.cos(yy / 7.0)
    return np.stack((value, np.roll(value, 3, 1), np.roll(value, 5, 0)), axis=2).clip(0, 255).astype(np.uint8)


def motion(image, length=13):
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2] = 1.0 / length
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def zoom_blur(image, maximum=1.24):
    height, width = image.shape[:2]
    total = image.astype(np.float32)
    count = 1
    for factor in np.arange(1.02, maximum + 1e-6, 0.02):
        resized = cv2.resize(image, None, fx=float(factor), fy=float(factor))
        y0 = (resized.shape[0] - height) // 2
        x0 = (resized.shape[1] - width) // 2
        total += resized[y0:y0 + height, x0:x0 + width]
        count += 1
    return np.clip(np.rint(total / count), 0, 255).astype(np.uint8)


def test_same_state_action_requires_state_identity():
    with pytest.raises(ValueError):
        ActionSpec("x", "v1", "feature", "first", "output", "flow_native",
                   execution_semantics="same_state", intervention_step=1)


def test_trust_region_is_exact_outside_support_and_caps_nonfinite_candidate():
    native = np.zeros((5, 7, 2), np.float32)
    candidate = np.full_like(native, 20.0)
    candidate[0, 0] = np.nan
    support = np.zeros((5, 7), np.float32)
    support[1:4, 2:6] = 1.0
    result = bounded_output(native, candidate, support, alpha=1.0, epsilon=0.25)
    assert result.max_update_norm <= 0.250002
    assert np.array_equal(result.flow[support == 0], native[support == 0])
    assert np.array_equal(result.flow[0, 0], native[0, 0])


def test_support_maps_keep_four_meanings_separate():
    shape = (4, 6)
    maps = SupportMaps(
        "flow_native", np.ones(shape), np.full(shape, 0.5),
        np.full(shape, 0.25), np.pad(np.ones((2, 2)), ((1, 1), (2, 2))),
    )
    assert np.allclose(maps.executable_support[1:3, 2:4], 0.125)
    assert np.count_nonzero(maps.executable_support) == 4


def test_second_mask_transport_uses_flow_coordinates_and_zero_border():
    mask = np.zeros((4, 6), np.float32)
    mask[:, 3] = 1.0
    flow = np.zeros((4, 6, 2), np.float32)
    flow[..., 0] = 1.0
    transported, valid = transport_second_mask_to_flow(mask, flow)
    assert np.allclose(transported[:, 2], 1.0)
    assert np.all(transported[:, 3:] == 0.0)
    assert np.all(~valid[:, -1])


def test_flow_mask_transport_splats_to_second_and_exposes_holes():
    mask = np.zeros((4, 6), np.float32)
    mask[:, 2] = 1.0
    flow = np.zeros((4, 6, 2), np.float32)
    flow[..., 0] = 1.0
    transported, weight = transport_flow_mask_to_second(mask, flow)
    assert np.allclose(transported[:, 3], 1.0)
    assert np.all(transported[:, :3] == 0.0)
    assert np.all(weight[:, 0] == 0.0)
    assert np.allclose(weight[:, 1:], 1.0)


def test_flow_mask_transport_reports_collision_weight():
    mask = np.ones((3, 4), np.float32)
    flow = np.zeros((3, 4, 2), np.float32)
    flow[:, 0, 0] = 1.0
    _, weight = transport_flow_mask_to_second(mask, flow)
    assert np.all(weight[:, 1] == pytest.approx(2.0))


def test_observable_second_endpoint_support_is_returned_in_flow_frame():
    first = texture(32, 48)
    second = first.copy()
    second[10, 20] = 255
    flow = np.zeros((32, 48, 2), np.float32)
    maps = observable_support_maps(
        first, second, flow, operator_id="impulse_exact_median3", endpoint="second",
    )
    assert maps.coordinate_frame == "flow_native"
    assert maps.physical_applicability[10, 20] == pytest.approx(1.0)
    assert np.count_nonzero(maps.predicted_signed_utility) == 0


def test_endpoint_support_keeps_second_input_in_its_own_coordinates():
    first = texture(32, 48)
    second = first.copy()
    second[10, 20] = 255
    flow = np.zeros((32, 48, 2), np.float32)
    flow[..., 0] = 1.0
    support = observable_endpoint_applicability(
        first, second, flow, operator_id="impulse_exact_median3", endpoint="second",
    )
    assert support.input_second[10, 20] == pytest.approx(1.0)
    assert support.output_flow[10, 19] == pytest.approx(1.0)
    assert np.count_nonzero(support.input_first) == 0


def test_impulse_certificate_accepts_sparse_outliers_and_rejects_clean_texture():
    clean = texture(96, 128)
    corrupt = clean.copy()
    corrupt.reshape(-1, 3)[::97] = 255
    accepted = impulse_noise_certificate(corrupt, endpoint="first")
    rejected = impulse_noise_certificate(clean, endpoint="first")
    assert accepted.status == "supported"
    assert rejected.status != "supported"


def test_additive_noise_certificate_requires_spatially_coherent_excess():
    clean = texture(128, 160)
    corrupt = clean.copy().astype(np.float32)
    rng = np.random.default_rng(17)
    corrupt[32:96, 48:112] += rng.normal(0, 25, (64, 64, 3))
    corrupt = np.clip(corrupt, 0, 255).astype(np.uint8)
    accepted = additive_noise_certificate(corrupt, endpoint="first")
    rejected = additive_noise_certificate(clean, endpoint="first")
    assert accepted.status == "supported", accepted.rejection_reasons
    assert rejected.status != "supported"


def test_paired_impulse_certificate_uses_correspondence_null():
    clean = texture(96, 128)
    corrupt = clean.copy()
    corrupt.reshape(-1, 3)[::97] = 255
    flow = np.zeros((96, 128, 2), np.float32)
    accepted, other = paired_impulse_certificates(corrupt, clean, flow)
    clean_first, clean_second = paired_impulse_certificates(clean, clean, flow)
    assert accepted.status == "supported", accepted.rejection_reasons
    assert other.status != "supported"
    assert clean_first.status != "supported"
    assert clean_second.status != "supported"


def test_local_action_reapplies_hard_support_after_feathering():
    image = texture(64, 80)
    support = np.zeros(image.shape[:2], np.float32)
    support[20:44, 25:55] = 1.0
    output, record = apply_local_image_action(
        image, support, operator_id="wiener3", endpoint="first",
    )
    assert np.array_equal(output[support == 0], image[support == 0])
    assert record.support_fraction == pytest.approx(float((support > 0).mean()))


def test_local_precomputed_proposal_is_exact_outside_support():
    image = texture(64, 80)
    proposal = cv2.GaussianBlur(image, (0, 0), 2.0)
    support = np.zeros(image.shape[:2], np.float32)
    support[16:48, 20:60] = 1.0
    output, record = blend_local_proposal(
        image, proposal, support, operator_id="common_gaussian",
        endpoint="second", feather_sigma=1.0,
    )
    assert np.array_equal(output[support == 0], image[support == 0])
    assert np.any(output[support > 0] != image[support > 0])
    assert record.operator_id == "common_gaussian"
    assert record.endpoint == "second"


def test_motion_certificate_supports_one_sided_motion_and_rejects_symmetric_noise():
    clean = texture()
    flow = np.zeros((*clean.shape[:2], 2), np.float32)
    blurred = motion(clean)
    permissive = CommonPassbandConfig(
        minimum_gradient_deficit=0.01, minimum_spectral_deficit=0.005,
        minimum_holdout_relative_gain=0.005,
        minimum_motion_specificity=0.0, minimum_fit_stability=0.25,
    )
    supported = common_passband_certificate(
        blurred, clean, flow, degraded_endpoint="first", config=permissive,
    )
    assert supported.status == "supported", supported.rejection_reasons
    rng = np.random.default_rng(9)
    noise = rng.normal(0, 18, clean.shape)
    noisy_first = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    noisy_second = np.clip(clean.astype(np.float32) + np.roll(noise, 1, 1), 0, 255).astype(np.uint8)
    rejected = common_passband_certificate(
        noisy_first, noisy_second, flow, degraded_endpoint="first", config=permissive,
    )
    assert rejected.status != "supported"
    assert any(reason.startswith("no_one_sided") for reason in rejected.rejection_reasons)


def test_zoom_certificate_supports_radial_blur_and_rejects_symmetric_contrast():
    clean = texture(128, 160)
    flow = np.zeros((*clean.shape[:2], 2), np.float32)
    permissive = CommonPassbandConfig(
        minimum_gradient_deficit=0.005, minimum_spectral_deficit=0.005,
        minimum_holdout_relative_gain=0.0, minimum_motion_specificity=0.0,
        minimum_fit_stability=0.0,
    )
    supported = zoom_common_passband_certificate(
        zoom_blur(clean), clean, flow, degraded_endpoint="first", config=permissive,
    )
    assert supported.status == "supported", supported.rejection_reasons
    contrast = np.clip((clean.astype(np.float32) - 128) * 1.4 + 128, 0, 255).astype(np.uint8)
    rejected = zoom_common_passband_certificate(
        contrast, contrast, flow, degraded_endpoint="first", config=permissive,
    )
    assert rejected.status != "supported"
    assert "no_one_sided_bandwidth_loss" in rejected.rejection_reasons
