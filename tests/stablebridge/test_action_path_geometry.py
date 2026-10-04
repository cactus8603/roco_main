import numpy as np
import pytest

from stablebridge.physical_repair.action_path_geometry import (
    GAUSSIAN_VARIANCE_PATH,
    gaussian_added_variance_set,
    gaussian_information_retention_lower,
    gaussian_otf_magnitude,
    gaussian_semigroup_sigma,
    validate_action_path,
)


def test_gaussian_strength_is_linear_in_variance_not_sigma():
    assert gaussian_semigroup_sigma(4.0, 0.25) == pytest.approx(2.0)
    assert gaussian_semigroup_sigma(4.0, 0.5) == pytest.approx(np.sqrt(8.0))


def test_gaussian_otf_composes_exactly_along_semigroup_time():
    frequencies = np.linspace(0.0, 0.5, 51)
    full_sigma = 2.5
    left = gaussian_otf_magnitude(
        frequencies, gaussian_semigroup_sigma(full_sigma, 0.3),
    )
    right = gaussian_otf_magnitude(
        frequencies, gaussian_semigroup_sigma(full_sigma, 0.7),
    )
    assert np.allclose(
        left * right,
        gaussian_otf_magnitude(frequencies, full_sigma),
        rtol=1e-13, atol=1e-15,
    )


def test_endpoint_intervals_produce_robust_added_variance_set():
    result = gaussian_added_variance_set((1.0, 1.2), (2.0, 2.2))
    assert result.added_variance_interval == pytest.approx((2.56, 3.84))
    assert result.minimax_added_variance == pytest.approx(3.2)
    assert result.worst_variance_mismatch == pytest.approx(0.64)
    assert result.minimax_sigma == pytest.approx(np.sqrt(3.2))


def test_overlapping_endpoint_order_fails_closed():
    with pytest.raises(ValueError, match="not uniformly identifiable"):
        gaussian_added_variance_set((1.0, 2.0), (1.5, 2.5))


def test_disk_and_motion_cannot_borrow_gaussian_semigroup_law():
    validate_action_path("common_gaussian", GAUSSIAN_VARIANCE_PATH)
    with pytest.raises(ValueError, match="cannot use"):
        validate_action_path("common_disk", GAUSSIAN_VARIANCE_PATH)
    with pytest.raises(ValueError, match="cannot use"):
        validate_action_path("common_motion", GAUSSIAN_VARIANCE_PATH)


def test_information_retention_is_worst_band_otf():
    value = gaussian_information_retention_lower(1.0, 0.25)
    assert value == pytest.approx(np.exp(-2.0 * np.pi ** 2 * 0.25 ** 2))
    assert gaussian_information_retention_lower(1.0, 0.0) == 1.0


@pytest.mark.parametrize("sigma,strength", [(-1.0, 0.5), (1.0, -0.1), (1.0, 1.1)])
def test_invalid_gaussian_path_inputs_rejected(sigma, strength):
    with pytest.raises(ValueError):
        gaussian_semigroup_sigma(sigma, strength)
