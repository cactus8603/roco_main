from dataclasses import replace

import numpy as np
import pytest

from stablebridge.physical_repair.radiometry_rank_invariants import (
    AffineRadiometryParameterBox,
    rank_action_parameter_set_evidence,
    rank_action_parameter_set_from_radiometry,
)
from stablebridge.physical_repair.radiometry_certificates import (
    affine_radiometry_certificate,
)


def _image(size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(47)
    return rng.integers(30, 181, size=(size, size, 3), dtype=np.uint8)


def _zero_flow(image: np.ndarray) -> np.ndarray:
    return np.zeros((*image.shape[:2], 2), dtype=np.float32)


def _identity_box() -> AffineRadiometryParameterBox:
    return AffineRadiometryParameterBox(
        gain_intervals=((1.0, 1.0),) * 3,
        offset_intervals=((0.0, 0.0),) * 3,
        calibration_version="identity-point-v1",
    )


def test_point_identity_box_certifies_complete_interior_and_selector_bound():
    image = _image()
    evidence = rank_action_parameter_set_evidence(
        image, _zero_flow(image), _identity_box(),
    )
    assert int(evidence.total_invariant_support.sum()) == (64 - 2) ** 2
    bound = evidence.to_selector_bound(("rgb-affine-box",))
    assert bound.name == "rank_action_commutator"
    assert bound.margin_lower > 0.0
    assert bound.deterministic
    assert _identity_box().sha256 in bound.method


def test_channel_uncertainty_reduces_support_without_task_threshold():
    image = _image()
    identity = rank_action_parameter_set_evidence(
        image, _zero_flow(image), _identity_box(),
    )
    uncertain = rank_action_parameter_set_evidence(
        image, _zero_flow(image),
        AffineRadiometryParameterBox(
            gain_intervals=((0.65, 1.35), (0.65, 1.35), (0.65, 1.35)),
            offset_intervals=((-8.0, 8.0),) * 3,
            calibration_version="wide-channel-box-v1",
        ),
    )
    assert uncertain.total_invariant_support.sum() < identity.total_invariant_support.sum()
    assert uncertain.support_fraction < identity.support_fraction


def test_clipping_anywhere_in_parameter_box_removes_full_neighbourhood():
    image = np.full((16, 16, 3), 180, dtype=np.uint8)
    evidence = rank_action_parameter_set_evidence(
        image, _zero_flow(image),
        AffineRadiometryParameterBox(
            gain_intervals=((1.0, 2.0),) * 3,
            offset_intervals=((0.0, 0.0),) * 3,
            calibration_version="possible-clipping-v1",
        ),
    )
    assert not np.any(evidence.clipping_safe_support)
    assert not np.any(evidence.total_invariant_support)
    assert evidence.to_selector_bound(("rgb-affine-box",)).margin_lower == 0.0


def test_extreme_parameter_box_cannot_wrap_into_valid_uint8_support():
    image = np.full((16, 16, 3), 80, dtype=np.uint8)
    evidence = rank_action_parameter_set_evidence(
        image, _zero_flow(image),
        AffineRadiometryParameterBox(
            gain_intervals=((1.0, 1e100),) * 3,
            offset_intervals=((-1e100, 1e100),) * 3,
            calibration_version="extreme-rejected-v1",
        ),
    )
    assert not np.any(evidence.clipping_safe_support)
    assert not np.any(evidence.total_invariant_support)


def test_evaluation_hole_is_eroded_from_all_affected_rank_centers():
    image = _image(9)
    support = np.ones((9, 9), dtype=bool)
    support[4, 4] = False
    evidence = rank_action_parameter_set_evidence(
        image, _zero_flow(image), _identity_box(), evaluation_support=support,
    )
    assert int(evidence.full_neighbourhood_support.sum()) == 40
    assert np.array_equal(
        evidence.total_invariant_support, evidence.full_neighbourhood_support,
    )


def test_fractional_warp_commutator_reduces_exact_rank_support():
    image = _image()
    flow = _zero_flow(image)
    flow[..., 0] = 0.5
    evidence = rank_action_parameter_set_evidence(
        image, flow, _identity_box(),
    )
    assert evidence.warp_commutator_exact_support.sum() < (
        evidence.full_neighbourhood_support.sum()
    )
    assert np.array_equal(
        evidence.total_invariant_support,
        evidence.radiometry_order_invariant_support
        & evidence.warp_commutator_exact_support,
    )


def test_parameter_box_validation_is_fail_closed():
    with pytest.raises(ValueError, match="strictly positive"):
        replace(_identity_box(), gain_intervals=((-1.0, 1.0),) * 3)
    with pytest.raises(ValueError, match="aligned boolean"):
        rank_action_parameter_set_evidence(
            _image(), _zero_flow(_image()), _identity_box(),
            evaluation_support=np.ones((64, 64), dtype=np.uint8),
        )


def test_affine_certificate_box_is_evaluated_only_on_independent_fold():
    image = _image(384)
    second = np.clip(
        np.rint(image.astype(np.float32) * 0.70 + 18.0), 0, 255,
    ).astype(np.uint8)
    flow = _zero_flow(image)
    radiometry = affine_radiometry_certificate(image, second, flow)
    assert radiometry.certificate.status == "supported"
    evidence = rank_action_parameter_set_from_radiometry(second, flow, radiometry)
    assert np.all(evidence.evaluation_support <= radiometry.independent_check_support)
    bound = evidence.to_selector_bound(
        ("simultaneous-rgb-affine-box",),
    )
    assert evidence.parameter_box.sha256 in bound.method
    assert bound.margin_lower > 0.0
    assert evidence.support_fraction > 0.0


def test_affine_parameter_box_cannot_be_reused_on_another_input():
    image = _image(384)
    second = np.clip(
        np.rint(image.astype(np.float32) * 0.70 + 18.0), 0, 255,
    ).astype(np.uint8)
    flow = _zero_flow(image)
    radiometry = affine_radiometry_certificate(image, second, flow)
    changed = second.copy()
    changed[100, 100, 0] ^= np.uint8(1)
    with pytest.raises(ValueError, match="does not match"):
        rank_action_parameter_set_from_radiometry(changed, flow, radiometry)
    shifted_flow = flow.copy()
    shifted_flow[..., 0] = 0.25
    with pytest.raises(ValueError, match="does not match"):
        rank_action_parameter_set_from_radiometry(second, shifted_flow, radiometry)
