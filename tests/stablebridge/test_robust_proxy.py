import numpy as np

from stablebridge.robust_proxy import (
    apply_observation,
    expand_profile_spec,
    observation_maps,
    parse_profile,
    robust20_profiles,
    transform_vector_ground_truth,
)


def test_profile_matrix_is_twenty_by_three_without_clean() -> None:
    profiles = expand_profile_spec({"kind": "robustspring20_proxy", "severities": [1, 2, 3]})
    assert len(profiles) == 60
    assert len({profile["name"] for profile in profiles}) == 60
    assert all(parse_profile(profile["name"]) is not None for profile in profiles)
    assert all("clean" not in profile["name"] for profile in profiles)


def test_all_profiles_render_uint8() -> None:
    image = np.full((24, 32, 3), 127, np.uint8)
    for profile in robust20_profiles():
        result, metadata = apply_observation(
            image, profile=profile["name"], seed=7, scene="0001", frame=5, view="left"
        )
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        assert metadata["condition"] == profile["condition"]
        assert metadata["severity"] == profile["severity"]


def test_elastic_vector_transport_changes_coordinates_and_masks_boundaries() -> None:
    profile = "robust20__elastic_transform__s2"
    source = observation_maps((32, 48), profile=profile, seed=7, scene="0001", frame=1, view="left")
    target = observation_maps((32, 48), profile=profile, seed=7, scene="0001", frame=1, view="right")
    gt = np.zeros((4, 2, 32, 48), np.float32)
    gt[:, 0] = -3.0
    transformed = transform_vector_ground_truth(gt, source, target)
    valid = np.isfinite(transformed).all(axis=1)
    assert 0.25 < valid.mean() < 1.0
    assert not np.allclose(np.nan_to_num(transformed[:, 0]), np.nan_to_num(gt[:, 0]))
