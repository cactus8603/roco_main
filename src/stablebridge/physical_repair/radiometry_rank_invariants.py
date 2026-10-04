"""Parameter-set robust invariants for the local-rank radiometry action.

Point-estimate commutators are useful diagnostics but they do not cover the
latent affine parameters that remain compatible with the observations.  This
module certifies a stricter support: for every RGB gain/offset in a supplied
box, no pixel in the complete 3x3 neighbourhood clips and every center versus
neighbour gray-order relation is unchanged.  The support is then intersected
with the exact observed rank/warp commutator support.

The construction is deliberately outcome-free.  It uses interval arithmetic
and exact descriptor relations instead of fitting a task threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import cv2
import numpy as np

from .radiometry_certificates import (
    RadiometryEvidence,
    _array_sha256,
    _erode_full_neighbourhood,
    _local_rank,
    _warp_second,
)
from .selector_v5 import ActionInvariantBound
from .verifier_task_bounds import support_mask_sha256


def _interval(value: tuple[float, float], name: str, *, positive: bool) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two endpoints")
    lower, upper = float(value[0]), float(value[1])
    if not np.isfinite((lower, upper)).all() or lower > upper:
        raise ValueError(f"invalid {name}")
    if positive and lower <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return lower, upper


@dataclass(frozen=True)
class AffineRadiometryParameterBox:
    """A rectangular identified set for a global positive RGB affine map."""

    gain_intervals: tuple[tuple[float, float], ...]
    offset_intervals: tuple[tuple[float, float], ...]
    calibration_version: str

    def __post_init__(self) -> None:
        if len(self.gain_intervals) != 3 or len(self.offset_intervals) != 3:
            raise ValueError("radiometry parameter box needs three RGB intervals")
        gains = tuple(
            _interval(value, f"gain interval {index}", positive=True)
            for index, value in enumerate(self.gain_intervals)
        )
        offsets = tuple(
            _interval(value, f"offset interval {index}", positive=False)
            for index, value in enumerate(self.offset_intervals)
        )
        if not self.calibration_version:
            raise ValueError("parameter box calibration version is required")
        object.__setattr__(self, "gain_intervals", gains)
        object.__setattr__(self, "offset_intervals", offsets)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.calibration_version.encode("utf-8"))
        digest.update(np.asarray(self.gain_intervals, dtype=np.float64).tobytes())
        digest.update(np.asarray(self.offset_intervals, dtype=np.float64).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class RankActionParameterSetEvidence:
    """Exact support on which every affine-box member preserves local rank."""

    parameter_box: AffineRadiometryParameterBox
    evaluation_support: np.ndarray
    full_neighbourhood_support: np.ndarray
    clipping_safe_support: np.ndarray
    radiometry_order_invariant_support: np.ndarray
    warp_commutator_exact_support: np.ndarray
    total_invariant_support: np.ndarray
    minimum_order_margin_255: np.ndarray

    def __post_init__(self) -> None:
        shape = self.evaluation_support.shape
        masks = (
            self.evaluation_support, self.full_neighbourhood_support,
            self.clipping_safe_support,
            self.radiometry_order_invariant_support,
            self.warp_commutator_exact_support,
            self.total_invariant_support,
        )
        if any(mask.dtype != bool or mask.shape != shape for mask in masks):
            raise ValueError("rank parameter-set supports must be aligned boolean maps")
        if (self.minimum_order_margin_255.shape != shape
                or not np.isfinite(self.minimum_order_margin_255).all()):
            raise ValueError("rank order margin must be a finite aligned map")
        if np.any(self.full_neighbourhood_support & ~self.evaluation_support):
            raise ValueError("full neighbourhood support escaped evaluation support")
        if np.any(self.clipping_safe_support & ~self.full_neighbourhood_support):
            raise ValueError("clipping-safe support escaped full neighbourhood support")
        if np.any(
            self.radiometry_order_invariant_support & ~self.clipping_safe_support
        ):
            raise ValueError("rank-order invariant support escaped clipping-safe support")
        expected = (
            self.radiometry_order_invariant_support
            & self.warp_commutator_exact_support
        )
        if not np.array_equal(self.total_invariant_support, expected):
            raise ValueError("total rank invariant support is not the exact intersection")
        if np.any(
            self.minimum_order_margin_255[self.total_invariant_support] <= 0.0
        ):
            raise ValueError("certified rank support needs a positive discrete margin")

    @property
    def support_fraction(self) -> float:
        denominator = max(int(self.evaluation_support.sum()), 1)
        return float(self.total_invariant_support.sum() / denominator)

    def to_selector_bound(
        self,
        latent_parameter_keys: tuple[str, ...],
    ) -> ActionInvariantBound:
        """Bind the analytic parameter-set proof to Selector v5."""
        if not latent_parameter_keys:
            raise ValueError("latent parameter keys are required")
        support = self.total_invariant_support
        margin = (
            float(np.min(self.minimum_order_margin_255[support]) / 255.0)
            if np.any(support) else 0.0
        )
        return ActionInvariantBound(
            name="rank_action_commutator",
            margin_lower=margin,
            support_hash=support_mask_sha256(support),
            latent_parameter_keys=latent_parameter_keys,
            deterministic=True,
            effective_n=float(support.sum()),
            method=f"opencv_affine_box_rank_order_v1:{self.parameter_box.sha256}",
        )


def _gray_interval(channel_lower: np.ndarray, channel_upper: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    # Use the exact runtime backend rather than reproducing nominal BT.601
    # coefficients: optimized OpenCV builds differ by one gray level on rare
    # rounding boundaries.  RGB-to-gray is coordinate-wise monotone, so the
    # all-lower and all-upper corners are exact extrema of the integer box.
    lower = cv2.cvtColor(
        channel_lower.astype(np.uint8), cv2.COLOR_RGB2GRAY,
    ).astype(np.int16)
    upper = cv2.cvtColor(
        channel_upper.astype(np.uint8), cv2.COLOR_RGB2GRAY,
    ).astype(np.int16)
    return lower, upper


def rank_action_parameter_set_evidence(
    second: np.ndarray,
    native_flow: np.ndarray,
    parameter_box: AffineRadiometryParameterBox,
    *,
    evaluation_support: np.ndarray | None = None,
) -> RankActionParameterSetEvidence:
    """Certify local-rank invariance uniformly over an affine parameter box.

    RGB rounding is conservatively intervalized before the fixed-point gray
    conversion.  Treating the global parameters as if they could vary by
    pixel only enlarges the interval, so an accepted order relation remains a
    valid (possibly conservative) statement for the actual global map.
    """
    image = np.asarray(second)
    flow = np.asarray(native_flow, dtype=np.float32)
    if (image.ndim != 3 or image.shape[2] != 3
            or not np.issubdtype(image.dtype, np.number)
            or not np.isfinite(image).all()
            or np.any(image < 0.0) or np.any(image > 255.0)
            or flow.shape != (*image.shape[:2], 2)
            or not np.isfinite(flow).all()):
        raise ValueError("expected finite RGB image and aligned finite flow")
    if evaluation_support is None:
        requested = np.ones(image.shape[:2], dtype=bool)
    else:
        requested = np.asarray(evaluation_support)
        if requested.dtype != bool or requested.shape != image.shape[:2]:
            raise ValueError("evaluation support must be an aligned boolean map")

    warped, valid = _warp_second(image.astype(np.float32), flow)
    evaluation = np.ascontiguousarray(requested & valid)
    full_neighbourhood = _erode_full_neighbourhood(evaluation)

    gains = np.asarray(parameter_box.gain_intervals, dtype=np.float64)
    offsets = np.asarray(parameter_box.offset_intervals, dtype=np.float64)
    raw_lower = (
        warped.astype(np.float64) * gains[None, None, :, 0]
        + offsets[None, None, :, 0]
    )
    raw_upper = (
        warped.astype(np.float64) * gains[None, None, :, 1]
        + offsets[None, None, :, 1]
    )
    # This over-approximates np.rint at half-integers, which is conservative.
    # Keep the unbounded values in floating point until after clipping so an
    # extreme rejected parameter box cannot wrap around an integer dtype.
    channel_lower_unbounded = np.ceil(raw_lower - 0.5)
    channel_upper_unbounded = np.floor(raw_upper + 0.5)
    pixel_unclipped = np.all(
        (channel_lower_unbounded >= 1) & (channel_upper_unbounded <= 254),
        axis=2,
    )
    clipping_safe = _erode_full_neighbourhood(evaluation & pixel_unclipped)
    bounded_lower = np.clip(channel_lower_unbounded, 0, 255).astype(np.uint8)
    bounded_upper = np.clip(channel_upper_unbounded, 0, 255).astype(np.uint8)
    gray_lower, gray_upper = _gray_interval(bounded_lower, bounded_upper)

    warped_uint8 = np.clip(np.rint(warped), 0, 255).astype(np.uint8)
    reference_gray = cv2.cvtColor(warped_uint8, cv2.COLOR_RGB2GRAY)
    height, width = reference_gray.shape
    padded_reference = np.pad(reference_gray, 1, mode="edge")
    padded_lower = np.pad(gray_lower, 1, mode="edge")
    padded_upper = np.pad(gray_upper, 1, mode="edge")
    center_reference = reference_gray
    center_lower, center_upper = gray_lower, gray_upper
    invariant = clipping_safe.copy()
    minimum_margin = np.full((height, width), 255.0, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbour_reference = padded_reference[
                1 + dy:1 + dy + height, 1 + dx:1 + dx + width,
            ]
            neighbour_lower = padded_lower[
                1 + dy:1 + dy + height, 1 + dx:1 + dx + width,
            ]
            neighbour_upper = padded_upper[
                1 + dy:1 + dy + height, 1 + dx:1 + dx + width,
            ]
            greater = center_reference > neighbour_reference
            less = center_reference < neighbour_reference
            equal = ~(greater | less)
            greater_margin = center_lower.astype(np.int32) - neighbour_upper
            less_margin = neighbour_lower.astype(np.int32) - center_upper
            exact_tie = (
                (center_lower == center_upper)
                & (neighbour_lower == neighbour_upper)
                & (center_lower == neighbour_lower)
            )
            relation_ok = (
                (greater & (greater_margin > 0))
                | (less & (less_margin > 0))
                | (equal & exact_tie)
            )
            relation_margin = np.where(
                greater, greater_margin,
                np.where(less, less_margin, np.where(exact_tie, 1, 0)),
            ).astype(np.float32)
            invariant &= relation_ok
            minimum_margin = np.minimum(minimum_margin, relation_margin)
    order_invariant = np.ascontiguousarray(invariant)

    rank_warped = _local_rank(warped)
    rank_source = _local_rank(image)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    warped_rank = cv2.remap(
        rank_source, xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    warp_exact = np.ascontiguousarray(
        full_neighbourhood & (rank_warped == warped_rank)
    )
    total = np.ascontiguousarray(order_invariant & warp_exact)
    return RankActionParameterSetEvidence(
        parameter_box=parameter_box,
        evaluation_support=evaluation,
        full_neighbourhood_support=full_neighbourhood,
        clipping_safe_support=clipping_safe,
        radiometry_order_invariant_support=order_invariant,
        warp_commutator_exact_support=warp_exact,
        total_invariant_support=total,
        minimum_order_margin_255=np.ascontiguousarray(minimum_margin),
    )


def rank_action_parameter_set_from_radiometry(
    second: np.ndarray,
    native_flow: np.ndarray,
    radiometry: RadiometryEvidence,
) -> RankActionParameterSetEvidence:
    """Apply the A-fold affine box only on the certificate's held-out B fold."""
    image = np.asarray(second)
    flow = np.asarray(native_flow, dtype=np.float32)
    if (_array_sha256(image) != radiometry.second_input_hash
            or _array_sha256(flow) != radiometry.native_flow_hash):
        raise ValueError("radiometry parameter box does not match supplied native inputs")
    if (radiometry.rgb_gain_intervals is None
            or radiometry.rgb_offset_intervals is None):
        raise ValueError("radiometry certificate has no positive simultaneous parameter box")
    box = AffineRadiometryParameterBox(
        gain_intervals=radiometry.rgb_gain_intervals,
        offset_intervals=radiometry.rgb_offset_intervals,
        calibration_version=(
            f"{radiometry.certificate.calibration_version}"
            "-ab-simultaneous-affine-box-v1"
        ),
    )
    return rank_action_parameter_set_evidence(
        image, flow, box,
        evaluation_support=radiometry.independent_check_support,
    )
