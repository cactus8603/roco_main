import pytest


torch = pytest.importorskip("torch")

from stablebridge.physical_repair.uncertainty_features import (
    FEATURE_CHANNELS_V1,
    build_searaft_observable_features_v1,
    warp_second_to_first_torch_v1,
)


def test_identity_warp_and_feature_schema():
    image = torch.linspace(0, 1, 3 * 5 * 7).reshape(1, 3, 5, 7)
    flow = torch.zeros(1, 2, 5, 7)
    risk = torch.full((1, 1, 5, 7), 0.25)
    warped, valid = warp_second_to_first_torch_v1(image, flow)
    assert torch.allclose(warped, image)
    assert bool(valid.all())
    features, feature_valid = build_searaft_observable_features_v1(
        image, image, flow, risk,
    )
    assert features.shape == (1, 12, 5, 7)
    assert len(FEATURE_CHANNELS_V1) == 12
    assert bool(feature_valid.all())
    assert not features.requires_grad


def test_out_of_bounds_warp_is_explicitly_invalid():
    image = torch.ones(1, 3, 4, 5)
    flow = torch.zeros(1, 2, 4, 5)
    flow[:, 0] = 100
    warped, valid = warp_second_to_first_torch_v1(image, flow)
    assert not bool(valid.any())
    assert torch.count_nonzero(warped) == 0


def test_feature_builder_rejects_non_runtime_rgb_range():
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        build_searaft_observable_features_v1(
            torch.full((1, 3, 2, 2), 255.0),
            torch.zeros((1, 3, 2, 2)),
            torch.zeros((1, 2, 2, 2)),
            torch.zeros((1, 1, 2, 2)),
        )
