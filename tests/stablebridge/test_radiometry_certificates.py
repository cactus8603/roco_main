import numpy as np

from stablebridge.physical_repair.radiometry_certificates import (
    affine_radiometry_certificate,
    rank_action_invariance_evidence,
)


def texture(size=384):
    yy, xx = np.mgrid[:size, :size]
    base = 90 + 42 * np.sin(xx / 7.0) + 31 * np.cos(yy / 9.0)
    return np.stack((base, np.roll(base, 11, 1), np.roll(base, 17, 0)), 2).clip(
        5, 245
    ).astype(np.uint8)


def zero_flow(image):
    return np.zeros((*image.shape[:2], 2), dtype=np.float32)


def test_positive_affine_radiometry_has_invertible_heldout_closure():
    first = texture()
    second = np.clip(np.rint(first.astype(np.float32) * 0.55 + 28.0), 0, 255).astype(np.uint8)
    evidence = affine_radiometry_certificate(first, second, zero_flow(first))
    assert evidence.certificate.status == "supported", evidence.certificate.rejection_reasons
    assert evidence.certificate.competing_model_scores["monotone_derivative_min_lcb"] > 0.0
    assert evidence.certificate.competing_model_scores["heldout_inverse_gain_lcb_255"] > 0.0
    assert evidence.rgb_gain_intervals is not None
    assert evidence.rgb_offset_intervals is not None
    assert all(lower > 0.0 for lower, _ in evidence.rgb_gain_intervals)
    assert np.any(evidence.independent_check_support)
    assert np.all(evidence.independent_check_support <= evidence.recoverable_support)


def test_clean_identity_does_not_claim_radiometric_action_applicability():
    first = texture()
    evidence = affine_radiometry_certificate(first, first.copy(), zero_flow(first))
    assert evidence.certificate.status == "rejected"
    assert "no_heldout_forward_closure_gain" in evidence.certificate.rejection_reasons


def test_nonmonotone_intensity_inversion_is_rejected():
    first = texture()
    second = 255 - first
    evidence = affine_radiometry_certificate(first, second, zero_flow(first))
    assert evidence.certificate.status == "rejected"
    assert "monotone_derivative_lower_not_positive" in evidence.certificate.rejection_reasons


def test_clipped_pixels_are_explicitly_removed_from_recoverable_support():
    first = texture()
    second = np.clip(first.astype(np.int16) + 100, 0, 255).astype(np.uint8)
    evidence = affine_radiometry_certificate(first, second, zero_flow(first))
    assert np.any(evidence.clipping_censored)
    assert not np.any(evidence.recoverable_support & evidence.clipping_censored)
    assert np.all(evidence.recoverable_support <= evidence.correspondence_support)


def test_additive_noise_does_not_gain_affine_inverse_closure():
    first = texture()
    rng = np.random.default_rng(7)
    second = np.clip(
        first.astype(np.float32) + rng.normal(0.0, 18.0, first.shape), 0, 255,
    ).astype(np.uint8)
    evidence = affine_radiometry_certificate(first, second, zero_flow(first))
    assert evidence.certificate.status == "rejected"
    assert any("closure_gain" in reason for reason in evidence.certificate.rejection_reasons)


def test_common_positive_affine_map_commutes_with_rank_at_zero_flow():
    image = texture(96)[16:80, 16:80]
    evidence = rank_action_invariance_evidence(
        image, zero_flow(image), (1.0, 1.0, 1.0), (4.0, 4.0, 4.0),
    )
    assert evidence.exact_continuous_grayscale_invariance
    assert evidence.radiometry_max_abs == 0.0
    assert evidence.warp_max_abs == 0.0
    assert evidence.total_max_abs == 0.0


def test_channel_specific_gains_are_not_rank_invariant():
    rng = np.random.default_rng(19)
    image = rng.integers(40, 141, size=(64, 64, 3), dtype=np.uint8)
    evidence = rank_action_invariance_evidence(
        image, zero_flow(image), (1.35, 0.65, 1.0), (0.0, 20.0, 5.0),
    )
    assert not evidence.exact_continuous_grayscale_invariance
    assert evidence.continuous_gain_residual_linf > 0.0
    assert evidence.radiometry_disagreement_fraction > 0.0
    assert evidence.total_disagreement_fraction > 0.0


def test_rank_support_requires_complete_unclipped_neighbourhood():
    image = np.full((9, 9, 3), 80, dtype=np.uint8)
    support = np.ones((9, 9), dtype=bool)
    support[4, 4] = False
    evidence = rank_action_invariance_evidence(
        image, zero_flow(image), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),
        evaluation_support=support,
    )
    # Border erosion removes 32 centers; the unsupported center removes its
    # complete 3x3 set of possible rank centers from the remaining 7x7 area.
    assert int(evidence.neighbourhood_support.sum()) == 40
    assert not np.any(evidence.neighbourhood_support & ~evidence.evaluation_support)


def test_rank_commutator_excludes_action_induced_clipping():
    image = np.full((32, 32, 3), 180, dtype=np.uint8)
    evidence = rank_action_invariance_evidence(
        image, zero_flow(image), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0),
    )
    assert np.all(evidence.clipping_censored)
    assert not np.any(evidence.neighbourhood_support)
    assert evidence.total_disagreement_fraction == 1.0
