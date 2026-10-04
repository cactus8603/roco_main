"""Action-specific certificate for invertible affine radiometry.

The certificate is intentionally narrower than a renderer family label.  It
asks whether a positive, spatially stable RGB affine transfer explains paired
intensities on held-out correspondences, whether the inverse closes on the
same held-out support, and which pixels remain uncensored by clipping.

The radiometry model and the proposed rank action have different invariance
groups.  In particular, three positive channel-wise gains do not in general
preserve the ordering of RGB-to-gray intensities.  The certificate therefore
also measures the actual rank/radiometry and rank/warp commutators.  These are
action witnesses, not task-utility claims; no task outcome is used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from statistics import NormalDist
import time

import cv2
import numpy as np

from .contracts import ActionSpec, PhysicalCertificate


TILE = 64
MIN_TILES = 4
FAMILYWISE_ALPHA = 0.05


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _bounds(values: np.ndarray, z_value: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        return float("-inf"), float("inf")
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean
    radius = z_value * float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean - radius, mean + radius


def _warp_second(second: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
    valid = (
        (map_x >= 0.0) & (map_x <= width - 1.0)
        & (map_y >= 0.0) & (map_y <= height - 1.0)
    )
    warped = cv2.remap(
        second.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    return np.ascontiguousarray(warped), np.ascontiguousarray(valid)


def _tile_slices(shape: tuple[int, int], parity: int):
    height, width = shape
    for y0 in range(0, height - TILE + 1, TILE):
        for x0 in range(0, width - TILE + 1, TILE):
            if (y0 // TILE + x0 // TILE) % 2 == parity:
                yield slice(y0, y0 + TILE), slice(x0, x0 + TILE)


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 32:
        return None
    centered = x - float(x.mean())
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return None
    slope = float(np.dot(centered, y - float(y.mean())) / denominator)
    offset = float(y.mean() - slope * x.mean())
    if not np.isfinite(slope) or not np.isfinite(offset):
        return None
    return slope, offset


def _local_rank(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.clip(np.rint(image), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    center = padded[1:-1, 1:-1]
    less = np.zeros_like(center, dtype=np.float32)
    equal = np.zeros_like(center, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbour = padded[1 + dy:1 + dy + gray.shape[0],
                               1 + dx:1 + dx + gray.shape[1]]
            less += neighbour < center
            equal += neighbour == center
    return np.ascontiguousarray((less + 0.5 * equal) / 8.0, dtype=np.float32)


def _erode_full_neighbourhood(mask: np.ndarray) -> np.ndarray:
    """Keep centers whose complete 3x3 rank neighbourhood is admissible."""
    eroded = cv2.erode(
        np.asarray(mask, dtype=np.uint8), np.ones((3, 3), dtype=np.uint8),
        iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return np.ascontiguousarray(eroded.astype(bool))


def _rank_commutator_statistics(
    left: np.ndarray,
    right: np.ndarray,
    support: np.ndarray,
) -> tuple[float, float, float]:
    absolute = np.abs(
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    )
    if not np.any(support):
        return 1.0, 1.0, 1.0
    selected = absolute[support]
    return (
        float(selected.mean()), float(selected.max()),
        float(np.mean(selected > 0.0)),
    )


@dataclass(frozen=True)
class RankActionInvarianceEvidence:
    """Outcome-free closure evidence for the exact local-rank action.

    If ``g`` is the fitted RGB radiometric map, ``W`` is the native warp and
    ``R`` is the 3x3 local-rank transform, the three reported discrepancies
    are ``R(g(WI))-R(WI)``, ``R(WI)-W(RI)``, and ``R(g(WI))-W(RI)``.  The
    decomposition prevents invertibility of ``g`` from being mistaken for
    invariance of ``R``.
    """

    evaluation_support: np.ndarray
    neighbourhood_support: np.ndarray
    clipping_censored: np.ndarray
    rgb_gain: tuple[float, float, float]
    rgb_offset: tuple[float, float, float]
    continuous_common_gain: float
    continuous_gain_residual_linf: float
    exact_continuous_grayscale_invariance: bool
    radiometry_mean_abs: float
    radiometry_max_abs: float
    radiometry_disagreement_fraction: float
    warp_mean_abs: float
    warp_max_abs: float
    warp_disagreement_fraction: float
    total_mean_abs: float
    total_max_abs: float
    total_disagreement_fraction: float

    def __post_init__(self) -> None:
        shape = self.evaluation_support.shape
        maps = (
            self.evaluation_support, self.neighbourhood_support,
            self.clipping_censored,
        )
        if any(value.dtype != bool or value.shape != shape for value in maps):
            raise ValueError("rank action support maps must be aligned boolean maps")
        if np.any(self.neighbourhood_support & ~self.evaluation_support):
            raise ValueError("rank neighbourhood support must lie inside evaluation support")
        if np.any(self.neighbourhood_support & self.clipping_censored):
            raise ValueError("rank neighbourhood support cannot contain clipped centers")
        scalars = (
            *self.rgb_gain, *self.rgb_offset, self.continuous_common_gain,
            self.continuous_gain_residual_linf, self.radiometry_mean_abs,
            self.radiometry_max_abs, self.radiometry_disagreement_fraction,
            self.warp_mean_abs, self.warp_max_abs,
            self.warp_disagreement_fraction, self.total_mean_abs,
            self.total_max_abs, self.total_disagreement_fraction,
        )
        if not np.isfinite(scalars).all():
            raise ValueError("rank action evidence must be finite")
        if any(value < 0.0 for value in scalars[6:]):
            raise ValueError("rank action discrepancies must be nonnegative")
        for value in (
            self.radiometry_disagreement_fraction,
            self.warp_disagreement_fraction,
            self.total_disagreement_fraction,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("rank disagreement fractions must lie in [0,1]")


def rank_action_invariance_evidence(
    second: np.ndarray,
    native_flow: np.ndarray,
    rgb_gain: tuple[float, float, float] | np.ndarray,
    rgb_offset: tuple[float, float, float] | np.ndarray,
    *,
    evaluation_support: np.ndarray | None = None,
) -> RankActionInvarianceEvidence:
    """Measure action closure without reading a corruption label or outcome.

    ``rgb_gain`` and ``rgb_offset`` map the native-flow-warped second endpoint
    toward the first endpoint.  An optional support should be independent of
    the fit used to obtain those parameters (the affine certificate uses its
    checkerboard-B tiles).
    """
    image = np.asarray(second)
    flow = np.asarray(native_flow, dtype=np.float32)
    gain = np.asarray(rgb_gain, dtype=np.float64)
    offset = np.asarray(rgb_offset, dtype=np.float64)
    if (image.ndim != 3 or image.shape[2] != 3
            or not np.issubdtype(image.dtype, np.number)
            or not np.isfinite(image).all()
            or np.any(image < 0.0) or np.any(image > 255.0)
            or flow.shape != (*image.shape[:2], 2) or not np.isfinite(flow).all()
            or gain.shape != (3,) or offset.shape != (3,)
            or not np.isfinite(gain).all() or not np.isfinite(offset).all()
            or np.any(gain <= 0.0)):
        raise ValueError("expected finite RGB image, flow, and positive affine map")
    if evaluation_support is None:
        requested = np.ones(image.shape[:2], dtype=bool)
    else:
        requested = np.asarray(evaluation_support)
        if requested.dtype != bool or requested.shape != image.shape[:2]:
            raise ValueError("evaluation support must be an aligned boolean map")

    warped, valid = _warp_second(image.astype(np.float32), flow)
    height, width = flow.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
    rank_source = _local_rank(image)
    warped_rank = cv2.remap(
        rank_source, map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    raw_transformed = (
        warped.astype(np.float64) * gain[None, None, :]
        + offset[None, None, :]
    )
    clipped = np.any(
        (warped <= 0.0) | (warped >= 255.0)
        | (raw_transformed <= 0.0) | (raw_transformed >= 255.0),
        axis=2,
    )
    base_support = requested & valid & ~clipped
    neighbourhood = _erode_full_neighbourhood(base_support)
    rank_warped = _local_rank(warped)
    rank_transformed = _local_rank(raw_transformed)
    radiometry = _rank_commutator_statistics(
        rank_transformed, rank_warped, neighbourhood,
    )
    warp = _rank_commutator_statistics(rank_warped, warped_rank, neighbourhood)
    total = _rank_commutator_statistics(
        rank_transformed, warped_rank, neighbourhood,
    )

    common_gain = float(gain.mean())
    gain_residual = float(np.max(np.abs(gain - common_gain)))
    exact_tolerance = 16.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(gain))),
    )
    return RankActionInvarianceEvidence(
        evaluation_support=np.ascontiguousarray(requested & valid),
        neighbourhood_support=neighbourhood,
        clipping_censored=np.ascontiguousarray((requested & valid) & clipped),
        rgb_gain=tuple(float(value) for value in gain),
        rgb_offset=tuple(float(value) for value in offset),
        continuous_common_gain=common_gain,
        continuous_gain_residual_linf=gain_residual,
        exact_continuous_grayscale_invariance=gain_residual <= exact_tolerance,
        radiometry_mean_abs=radiometry[0],
        radiometry_max_abs=radiometry[1],
        radiometry_disagreement_fraction=radiometry[2],
        warp_mean_abs=warp[0], warp_max_abs=warp[1],
        warp_disagreement_fraction=warp[2],
        total_mean_abs=total[0], total_max_abs=total[1],
        total_disagreement_fraction=total[2],
    )


@dataclass(frozen=True)
class RadiometryEvidence:
    certificate: PhysicalCertificate
    first_input_hash: str
    second_input_hash: str
    native_flow_hash: str
    correspondence_support: np.ndarray
    recoverable_support: np.ndarray
    clipping_censored: np.ndarray
    independent_check_support: np.ndarray
    rgb_gain: tuple[float, float, float]
    rgb_offset: tuple[float, float, float]
    rgb_gain_intervals: tuple[tuple[float, float], ...] | None
    rgb_offset_intervals: tuple[tuple[float, float], ...] | None
    rank_action_invariance: RankActionInvarianceEvidence | None

    def __post_init__(self) -> None:
        if not self.first_input_hash or not self.second_input_hash or not self.native_flow_hash:
            raise ValueError("radiometry input provenance hashes are required")
        shape = self.correspondence_support.shape
        if (self.correspondence_support.dtype != bool
                or self.recoverable_support.dtype != bool
                or self.clipping_censored.dtype != bool
                or self.independent_check_support.dtype != bool
                or self.recoverable_support.shape != shape
                or self.clipping_censored.shape != shape
                or self.independent_check_support.shape != shape):
            raise ValueError("radiometry support maps must be aligned boolean maps")
        if np.any(self.recoverable_support & ~self.correspondence_support):
            raise ValueError("recoverable support must lie inside correspondence support")
        if np.any(self.recoverable_support & self.clipping_censored):
            raise ValueError("clipping-censored pixels cannot be recoverable")
        if np.any(self.independent_check_support & ~self.recoverable_support):
            raise ValueError("independent check support must be recoverable")
        if (self.rgb_gain_intervals is None) != (self.rgb_offset_intervals is None):
            raise ValueError("gain and offset intervals must be present together")
        if self.rgb_gain_intervals is not None:
            if len(self.rgb_gain_intervals) != 3 or len(self.rgb_offset_intervals) != 3:
                raise ValueError("radiometry intervals need three RGB channels")
            for interval in (*self.rgb_gain_intervals, *self.rgb_offset_intervals):
                if (len(interval) != 2 or not np.isfinite(interval).all()
                        or interval[0] > interval[1]):
                    raise ValueError("invalid radiometry parameter interval")
            if any(interval[0] <= 0.0 for interval in self.rgb_gain_intervals):
                raise ValueError("radiometry gain intervals must remain positive")


def affine_radiometry_certificate(
    first: np.ndarray,
    second: np.ndarray,
    native_flow: np.ndarray,
    *,
    alpha: float = FAMILYWISE_ALPHA,
) -> RadiometryEvidence:
    """Fit on checkerboard A tiles and certify monotone inverse closure on B."""
    started = time.perf_counter()
    first = np.asarray(first)
    second = np.asarray(second)
    flow = np.asarray(native_flow, dtype=np.float32)
    if (first.dtype != np.uint8 or second.dtype != np.uint8
            or first.ndim != 3 or first.shape != second.shape or first.shape[2] != 3
            or flow.shape != (*first.shape[:2], 2) or not np.isfinite(flow).all()
            or min(first.shape[:2]) < 2 * TILE):
        raise ValueError("expected aligned uint8 RGB pair and finite HxWx2 native flow")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    warped, valid = _warp_second(second, flow)
    first_f = first.astype(np.float32)
    # Exact extrema are censored observations: their latent irradiance is only
    # interval-valued, so they cannot sign an invertibility claim.
    clipped = np.any(
        (first_f <= 0.0) | (first_f >= 255.0)
        | (warped <= 0.0) | (warped >= 255.0),
        axis=2,
    )
    recoverable = valid & ~clipped
    per_channel_slopes: list[list[float]] = [[], [], []]
    per_channel_offsets: list[list[float]] = [[], [], []]
    fit_tiles = 0
    for ys, xs in _tile_slices(valid.shape, parity=0):
        chosen = recoverable[ys, xs]
        tile_rows = []
        for channel in range(3):
            fit = _fit_affine(
                warped[ys, xs, channel][chosen], first_f[ys, xs, channel][chosen],
            )
            if fit is None:
                break
            tile_rows.append(fit)
        if len(tile_rows) != 3:
            continue
        fit_tiles += 1
        for channel, (slope, offset) in enumerate(tile_rows):
            per_channel_slopes[channel].append(slope)
            per_channel_offsets[channel].append(offset)
    gains = np.asarray([
        np.median(values) if values else float("nan")
        for values in per_channel_slopes
    ], dtype=np.float64)
    offsets = np.asarray([
        np.median(values) if values else float("nan")
        for values in per_channel_offsets
    ], dtype=np.float64)
    parameter_z = float(NormalDist().inv_cdf(1.0 - alpha / (2.0 * 6.0)))
    gain_intervals_raw = tuple(
        _bounds(np.asarray(values), parameter_z) for values in per_channel_slopes
    )
    offset_intervals_raw = tuple(
        _bounds(np.asarray(values), parameter_z) for values in per_channel_offsets
    )
    parameter_intervals_valid = (
        all(np.isfinite(interval).all() for interval in gain_intervals_raw)
        and all(np.isfinite(interval).all() for interval in offset_intervals_raw)
        and all(interval[0] > 0.0 for interval in gain_intervals_raw)
    )
    gain_intervals = gain_intervals_raw if parameter_intervals_valid else None
    offset_intervals = offset_intervals_raw if parameter_intervals_valid else None
    z_value = float(NormalDist().inv_cdf(1.0 - alpha / (2.0 * 5.0)))
    slope_lcbs = np.asarray([
        _bounds(np.asarray(values), z_value)[0] for values in per_channel_slopes
    ], dtype=np.float64)
    forward_gains, inverse_gains = [], []
    identity_values, action_values = [], []
    check_tiles = 0
    if np.isfinite(gains).all() and np.isfinite(offsets).all() and np.all(gains != 0.0):
        prediction = warped * gains[None, None, :] + offsets[None, None, :]
        inverse = (first_f - offsets[None, None, :]) / gains[None, None, :]
        identity_error = np.mean(np.abs(first_f - warped), axis=2)
        forward_error = np.mean(np.abs(first_f - prediction), axis=2)
        inverse_error = np.mean(np.abs(warped - inverse), axis=2)
        for ys, xs in _tile_slices(valid.shape, parity=1):
            chosen = recoverable[ys, xs]
            if int(chosen.sum()) < 32:
                continue
            check_tiles += 1
            identity = float(identity_error[ys, xs][chosen].mean())
            forward = float(forward_error[ys, xs][chosen].mean())
            inverse_value = float(inverse_error[ys, xs][chosen].mean())
            identity_values.append(identity)
            action_values.append(forward)
            forward_gains.append(identity - forward)
            inverse_gains.append(identity - inverse_value)
    forward_lcb, _ = _bounds(np.asarray(forward_gains), z_value)
    inverse_lcb, _ = _bounds(np.asarray(inverse_gains), z_value)
    rank_first = _local_rank(first_f)
    rank_second = _local_rank(warped)
    rank_residual = float(np.mean(np.abs(rank_first[recoverable] - rank_second[recoverable]))) \
        if np.any(recoverable) else 1.0
    rank_check_support = np.zeros_like(recoverable)
    for ys, xs in _tile_slices(valid.shape, parity=1):
        rank_check_support[ys, xs] = recoverable[ys, xs]
    rank_invariance = None
    if (np.isfinite(gains).all() and np.isfinite(offsets).all()
            and np.all(gains > 0.0)):
        rank_invariance = rank_action_invariance_evidence(
            second, flow, gains, offsets,
            evaluation_support=rank_check_support,
        )
    reasons = []
    if fit_tiles < MIN_TILES:
        reasons.append("insufficient_fit_tiles")
    if check_tiles < MIN_TILES:
        reasons.append("insufficient_independent_check_tiles")
    if not np.isfinite(slope_lcbs).all() or float(np.min(slope_lcbs)) <= 0.0:
        reasons.append("monotone_derivative_lower_not_positive")
    if forward_lcb <= 0.0:
        reasons.append("no_heldout_forward_closure_gain")
    if inverse_lcb <= 0.0:
        reasons.append("no_heldout_inverse_closure_gain")
    if not np.any(recoverable):
        reasons.append("no_unclipped_recoverable_support")
    identity_median = float(np.median(identity_values)) if identity_values else 255.0
    action_median = float(np.median(action_values)) if action_values else 255.0
    fit_stability = float(np.mean([
        slope > 0.0 for values in per_channel_slopes for slope in values
    ])) if fit_tiles else 0.0
    certificate = PhysicalCertificate(
        action=ActionSpec(
            operator_id="rank3_pair", operator_version="v2-affine-invertible-ab",
            domain="feature", hypothesized_degraded_endpoint="both",
            modified_endpoint="both", coordinate_frame="flow_native",
        ),
        status="supported" if not reasons else "rejected",
        observation_support_fraction=float(valid.mean()),
        identifiable_support_fraction=float(recoverable.mean()),
        null_score=identity_median / 255.0,
        action_score=action_median / 255.0,
        spatial_holdout_gain=float(min(forward_lcb, inverse_lcb) / 255.0),
        parameter_uncertainty=float(max(
            [np.std(values, ddof=1) if len(values) > 1 else 0.0
             for values in per_channel_slopes] or [0.0]
        )),
        fit_stability=fit_stability,
        competing_model_scores={
            "monotone_derivative_min_lcb": float(np.min(slope_lcbs))
            if np.isfinite(slope_lcbs).all() else -1.0,
            "heldout_forward_gain_lcb_255": float(forward_lcb),
            "heldout_inverse_gain_lcb_255": float(inverse_lcb),
            "rank_residual_fraction": rank_residual,
            "rank_radiometry_commutator_mean": (
                rank_invariance.radiometry_mean_abs
                if rank_invariance is not None else 1.0
            ),
            "rank_warp_commutator_mean": (
                rank_invariance.warp_mean_abs
                if rank_invariance is not None else 1.0
            ),
            "rank_total_commutator_mean": (
                rank_invariance.total_mean_abs
                if rank_invariance is not None else 1.0
            ),
            "rank_total_commutator_disagreement_fraction": (
                rank_invariance.total_disagreement_fraction
                if rank_invariance is not None else 1.0
            ),
            "rank_continuous_gain_residual_linf": (
                rank_invariance.continuous_gain_residual_linf
                if rank_invariance is not None else 1.0
            ),
            "clipping_censored_fraction": float((valid & clipped).mean()),
        },
        estimated_parameters={
            "red_gain": float(gains[0]) if np.isfinite(gains[0]) else 0.0,
            "green_gain": float(gains[1]) if np.isfinite(gains[1]) else 0.0,
            "blue_gain": float(gains[2]) if np.isfinite(gains[2]) else 0.0,
            "red_offset": float(offsets[0]) if np.isfinite(offsets[0]) else 0.0,
            "green_offset": float(offsets[1]) if np.isfinite(offsets[1]) else 0.0,
            "blue_offset": float(offsets[2]) if np.isfinite(offsets[2]) else 0.0,
        },
        diagnostics={
            "fit_tiles": float(fit_tiles), "check_tiles": float(check_tiles),
            "simultaneous_z": z_value,
            "affine_parameter_simultaneous_z": parameter_z,
            "affine_parameter_box_available": float(parameter_intervals_valid),
            "recoverable_fraction_of_valid": float(recoverable.sum() / max(valid.sum(), 1)),
            "gain_channel_span": float(np.ptp(gains)) if np.isfinite(gains).all() else 0.0,
        },
        rejection_reasons=tuple(reasons),
        calibration_version="radiometry-affine-invertible-ab-v2",
        measured_probe_seconds=time.perf_counter() - started,
    )
    return RadiometryEvidence(
        certificate=certificate,
        first_input_hash=_array_sha256(first),
        second_input_hash=_array_sha256(second),
        native_flow_hash=_array_sha256(flow),
        correspondence_support=np.ascontiguousarray(valid),
        recoverable_support=np.ascontiguousarray(recoverable),
        clipping_censored=np.ascontiguousarray(valid & clipped),
        independent_check_support=np.ascontiguousarray(rank_check_support),
        rgb_gain=tuple(float(value) for value in gains),
        rgb_offset=tuple(float(value) for value in offsets),
        rgb_gain_intervals=gain_intervals,
        rgb_offset_intervals=offset_intervals,
        rank_action_invariance=rank_invariance,
    )
