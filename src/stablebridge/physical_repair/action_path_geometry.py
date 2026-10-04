"""Action-specific parameter paths used by exact repair controls.

A scalar called ``strength`` has no physical meaning without a path.  For a
Gaussian convolution the natural path is heat-semigroup time (variance), so
``sigma(s) = sqrt(s) * sigma_full``.  Scaling sigma linearly would instead
apply only ``s**2`` of the intended semigroup time.  Disk and line-motion
kernels are not closed under convolution and must not inherit this law.

The registry is descriptive and fail-closed.  A path name never authorizes an
action; Selector v5 still requires measured physical and task bounds for the
exact repaired input and matcher output.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


FIXED_PATH = "fixed_control_v1"
RGB_CHORD_PATH = "observed_to_proposal_rgb_chord_v1"
GAUSSIAN_VARIANCE_PATH = "gaussian_variance_semigroup_v1"
JPEG_CODEC_PATH = "jpeg_codec_feasible_projection_v1"
AFFINE_RADIOMETRY_PATH = "positive_affine_group_path_v1"


@dataclass(frozen=True)
class ActionPathLaw:
    path_id: str
    parameter_coordinate: str
    closed_under_composition: bool
    reversible: bool
    exact_law: str

    def __post_init__(self) -> None:
        if not self.path_id or not self.parameter_coordinate or not self.exact_law:
            raise ValueError("action path law must be fully named")


ACTION_PATH_LAWS: Mapping[str, ActionPathLaw] = {
    FIXED_PATH: ActionPathLaw(
        FIXED_PATH, "none", False, False, "one exact discrete control",
    ),
    RGB_CHORD_PATH: ActionPathLaw(
        RGB_CHORD_PATH, "rgb amplitude", False, False,
        "x_s=(1-s)*observed+s*proposal",
    ),
    GAUSSIAN_VARIANCE_PATH: ActionPathLaw(
        GAUSSIAN_VARIANCE_PATH, "variance sigma^2", True, False,
        "sigma(s)^2=s*sigma_full^2",
    ),
    JPEG_CODEC_PATH: ActionPathLaw(
        JPEG_CODEC_PATH, "codec-feasible projection amplitude", False, False,
        "every realized point remains inside measured DCT/RGB constraints",
    ),
    AFFINE_RADIOMETRY_PATH: ActionPathLaw(
        AFFINE_RADIOMETRY_PATH, "log-positive-gain and additive offset", True, True,
        "gain(s)=exp(s*log(gain)); offset follows the composed affine path",
    ),
}


OPERATOR_ALLOWED_PATHS: Mapping[str, frozenset[str]] = {
    "impulse_exact_median3": frozenset({FIXED_PATH, RGB_CHORD_PATH}),
    "impulse_median3": frozenset({FIXED_PATH, RGB_CHORD_PATH}),
    "wiener3": frozenset({FIXED_PATH, RGB_CHORD_PATH}),
    "common_disk": frozenset({FIXED_PATH, RGB_CHORD_PATH}),
    "common_gaussian": frozenset({
        FIXED_PATH, RGB_CHORD_PATH, GAUSSIAN_VARIANCE_PATH,
    }),
    "common_motion": frozenset({FIXED_PATH, RGB_CHORD_PATH}),
    "jpeg_deblock": frozenset({FIXED_PATH, RGB_CHORD_PATH, JPEG_CODEC_PATH}),
    "pixelate": frozenset({FIXED_PATH, RGB_CHORD_PATH}),
    "rank3_pair": frozenset({FIXED_PATH, RGB_CHORD_PATH}),
}


def validate_action_path(operator_id: str, path_id: str) -> None:
    """Reject unnamed laws and invalid cross-family parameter semantics."""
    if path_id not in ACTION_PATH_LAWS:
        raise ValueError(f"unknown action path law: {path_id}")
    if operator_id not in OPERATOR_ALLOWED_PATHS:
        raise ValueError(f"operator has no path registry: {operator_id}")
    if path_id not in OPERATOR_ALLOWED_PATHS[operator_id]:
        raise ValueError(f"{operator_id} cannot use {path_id}")


def gaussian_semigroup_sigma(full_sigma: float, strength: float) -> float:
    """Return sigma at a fraction of Gaussian semigroup time."""
    sigma = float(full_sigma)
    amount = float(strength)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("full Gaussian sigma must be finite and nonnegative")
    if not np.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise ValueError("Gaussian path strength must lie in [0,1]")
    return float(math.sqrt(amount) * sigma)


def gaussian_otf_magnitude(spatial_frequency: np.ndarray | float,
                           sigma: float) -> np.ndarray:
    """Continuous isotropic Gaussian OTF magnitude in cycles per pixel."""
    frequency = np.asarray(spatial_frequency, dtype=np.float64)
    sigma = float(sigma)
    if not np.all(np.isfinite(frequency)) or np.any(frequency < 0.0):
        raise ValueError("spatial frequency must be finite and nonnegative")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("Gaussian sigma must be finite and nonnegative")
    return np.exp(-2.0 * math.pi * math.pi * sigma * sigma * frequency * frequency)


@dataclass(frozen=True)
class GaussianAddedVarianceSet:
    """Partially identified Gaussian action from endpoint sigma intervals."""

    source_sigma_interval: tuple[float, float]
    target_sigma_interval: tuple[float, float]
    added_variance_interval: tuple[float, float]
    minimax_added_variance: float
    minimax_sigma: float
    worst_variance_mismatch: float

    def __post_init__(self) -> None:
        for interval, name in (
            (self.source_sigma_interval, "source sigma interval"),
            (self.target_sigma_interval, "target sigma interval"),
            (self.added_variance_interval, "added variance interval"),
        ):
            if (len(interval) != 2 or not all(np.isfinite(interval))
                    or interval[0] < 0.0 or interval[0] > interval[1]):
                raise ValueError(f"invalid {name}")
        for value, name in (
            (self.minimax_added_variance, "minimax added variance"),
            (self.minimax_sigma, "minimax sigma"),
            (self.worst_variance_mismatch, "worst variance mismatch"),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid {name}")


def gaussian_added_variance_set(
    source_sigma_interval: tuple[float, float],
    target_sigma_interval: tuple[float, float],
) -> GaussianAddedVarianceSet:
    """Propagate endpoint uncertainty through the Gaussian semigroup law.

    A one-sided action is identifiable only when the target is more blurred
    than the source for *every* member of both intervals.  The returned
    midpoint is the minimax control for absolute error in added variance; it
    is a proposal, not delivery authority.
    """
    source = tuple(float(value) for value in source_sigma_interval)
    target = tuple(float(value) for value in target_sigma_interval)
    for interval, name in ((source, "source"), (target, "target")):
        if (len(interval) != 2 or not all(np.isfinite(interval))
                or interval[0] < 0.0 or interval[0] > interval[1]):
            raise ValueError(f"invalid {name} sigma interval")
    lower = target[0] ** 2 - source[1] ** 2
    upper = target[1] ** 2 - source[0] ** 2
    if lower <= 0.0 or upper < lower:
        raise ValueError("Gaussian endpoint order is not uniformly identifiable")
    midpoint = 0.5 * (lower + upper)
    mismatch = 0.5 * (upper - lower)
    return GaussianAddedVarianceSet(
        source_sigma_interval=source,
        target_sigma_interval=target,
        added_variance_interval=(float(lower), float(upper)),
        minimax_added_variance=float(midpoint),
        minimax_sigma=float(math.sqrt(midpoint)),
        worst_variance_mismatch=float(mismatch),
    )


def gaussian_information_retention_lower(sigma: float,
                                         maximum_frequency: float) -> float:
    """Worst amplitude retention over [0, maximum_frequency]."""
    frequency = float(maximum_frequency)
    if not np.isfinite(frequency) or not 0.0 <= frequency <= 0.5:
        raise ValueError("maximum frequency must lie in [0,0.5]")
    return float(gaussian_otf_magnitude(frequency, sigma))
