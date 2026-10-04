"""Paired physical evidence for a Wiener repair.

The older additive-noise certificate needs a quieter region in the same image,
so it necessarily abstains when additive noise covers the whole frame.  This
module instead uses the two video endpoints as repeated, approximately aligned
measurements.  It keeps three logically different questions separate:

* does one endpoint contain endpoint-exclusive, approximately white power;
* does the exact Wiener action reduce held-out paired residual energy; and
* does the action retain the observable signal band.

None of these quantities is a task-outcome or corruption-label oracle.  A
positive result is an action-own physical/effectiveness receipt; task delivery
still needs the matcher-risk, signed-utility, tail and visibility receipts used
by Selector-v6.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import cv2
import numpy as np

from .operators import apply_local_image_action


@dataclass(frozen=True)
class PairedWienerPSDReceipt:
    """Observable evidence for one concrete endpoint and exact Wiener action."""

    endpoint: str
    status: str
    tile_size: int
    valid_tiles: int
    fit_tiles: int
    check_tiles: int
    fit_gain_lower: float
    check_gain_lower: float
    endpoint_exclusive_power_fraction: float
    residual_lag_correlation_abs: float
    signal_band_retention: float
    estimated_noise_sigma_255: float
    support_mask: np.ndarray
    support_hash: str
    rejection_reasons: tuple[str, ...]
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if self.endpoint not in {"first", "second"}:
            raise ValueError("paired Wiener receipt needs a concrete endpoint")
        if self.status not in {"supported", "rejected"}:
            raise ValueError("unknown paired Wiener receipt status")
        if self.tile_size < 16 or min(
            self.valid_tiles, self.fit_tiles, self.check_tiles,
        ) < 0:
            raise ValueError("invalid paired Wiener tile accounting")
        values = (
            self.fit_gain_lower,
            self.check_gain_lower,
            self.endpoint_exclusive_power_fraction,
            self.residual_lag_correlation_abs,
            self.signal_band_retention,
            self.estimated_noise_sigma_255,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("paired Wiener evidence must be finite")
        mask = np.asarray(self.support_mask)
        if mask.dtype != np.bool_ or mask.ndim != 2:
            raise ValueError("paired Wiener support must be a boolean HxW mask")
        mask = np.ascontiguousarray(mask)
        expected_hash = hashlib.sha256(mask.tobytes()).hexdigest()
        if self.support_hash != expected_hash:
            raise ValueError("paired Wiener support hash mismatch")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("paired Wiener evidence cannot read labels or outcomes")
        object.__setattr__(self, "support_mask", mask)


def _validate(first: np.ndarray, second: np.ndarray, flow: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(first)
    second = np.asarray(second)
    flow = np.asarray(flow, dtype=np.float32)
    if (first.dtype != np.uint8 or second.dtype != np.uint8
            or first.ndim != 3 or first.shape != second.shape
            or first.shape[2] != 3 or flow.shape != (*first.shape[:2], 2)
            or not np.isfinite(flow).all()):
        raise ValueError("expected paired uint8 RGB and finite HxWx2 flow")
    return (
        np.ascontiguousarray(first),
        np.ascontiguousarray(second),
        np.ascontiguousarray(flow),
    )


def _warp_second(value: np.ndarray, flow: np.ndarray,
                 interpolation: int = cv2.INTER_LINEAR
                 ) -> tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx + flow[..., 0]
    map_y = yy + flow[..., 1]
    valid = ((map_x >= 1.0) & (map_x <= width - 2.0)
             & (map_y >= 1.0) & (map_y <= height - 2.0))
    warped = cv2.remap(
        np.asarray(value), map_x, map_y, interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    return warped, valid


def _highpass(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    low = cv2.GaussianBlur(
        source, (0, 0), 1.2, borderType=cv2.BORDER_REFLECT_101,
    )
    return np.ascontiguousarray(source - low)


def _band_masks(size: int) -> tuple[np.ndarray, np.ndarray]:
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    high = (radius >= 0.22) & (radius <= 0.48)
    signal = (radius >= 0.04) & (radius <= 0.22)
    return high, signal


def _lag_correlation(residual: np.ndarray, valid: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if residual.ndim != 3 or valid.shape != residual.shape[:2] or not np.any(valid):
        return 1.0
    center = np.mean(residual[valid], axis=0, keepdims=True)
    centered = residual - center.reshape(1, 1, -1)

    def correlation(left: np.ndarray, right: np.ndarray,
                    pair_valid: np.ndarray) -> float:
        if not np.any(pair_valid):
            return 1.0
        x = left[pair_valid]
        y = right[pair_valid]
        denominator = math.sqrt(max(
            float(np.sum(x * x) * np.sum(y * y)), 1e-24,
        ))
        return float(np.sum(x * y) / denominator)

    horizontal = correlation(
        centered[:, 1:], centered[:, :-1], valid[:, 1:] & valid[:, :-1],
    )
    vertical = correlation(
        centered[1:], centered[:-1], valid[1:] & valid[:-1],
    )
    return float(max(abs(horizontal), abs(vertical)))


def _lower_mean(values: np.ndarray) -> float:
    """A deterministic one-standard-error lower summary, not a p-value."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    mean = float(values.mean())
    if values.size == 1:
        return mean
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    return mean - standard_error


def _receipt_from_precomputed(
    *,
    endpoint: str,
    first_hp: np.ndarray,
    warped_second: np.ndarray,
    repaired_first_hp: np.ndarray,
    warped_repaired_second: np.ndarray,
    valid: np.ndarray,
    tile_size: int,
    minimum_tiles_per_fold: int,
    minimum_valid_fraction: float,
    minimum_exclusive_power_fraction: float,
    maximum_lag_correlation_abs: float,
    minimum_signal_band_retention: float,
) -> PairedWienerPSDReceipt:
    height, width = valid.shape
    high_band, signal_band = _band_masks(tile_size)
    window_1d = np.hanning(tile_size).astype(np.float32)
    window = window_1d[:, None] * window_1d[None, :]
    gains: list[list[float]] = [[], []]
    exclusive_values: list[float] = []
    lag_values: list[float] = []
    retention_values: list[float] = []
    sigma_values: list[float] = []
    accepted_tiles: list[tuple[int, int]] = []

    for top in range(0, height - tile_size + 1, tile_size):
        for left in range(0, width - tile_size + 1, tile_size):
            ys = slice(top, top + tile_size)
            xs = slice(left, left + tile_size)
            tile_valid = valid[ys, xs]
            if float(tile_valid.mean()) < minimum_valid_fraction:
                continue
            native_endpoint = first_hp[ys, xs] if endpoint == "first" else warped_second[ys, xs]
            paired_endpoint = warped_second[ys, xs] if endpoint == "first" else first_hp[ys, xs]
            repaired_endpoint = (
                repaired_first_hp[ys, xs]
                if endpoint == "first" else warped_repaired_second[ys, xs]
            )
            before = first_hp[ys, xs] - warped_second[ys, xs]
            after = (
                repaired_first_hp[ys, xs] - warped_second[ys, xs]
                if endpoint == "first"
                else first_hp[ys, xs] - warped_repaired_second[ys, xs]
            )
            choose = tile_valid[..., None]
            before_energy = float(np.sum((before * before) * choose) / max(choose.sum() * 3, 1))
            after_energy = float(np.sum((after * after) * choose) / max(choose.sum() * 3, 1))
            gain = before_energy - after_energy

            endpoint_fft = np.fft.rfft2(native_endpoint * window[..., None], axes=(0, 1))
            paired_fft = np.fft.rfft2(paired_endpoint * window[..., None], axes=(0, 1))
            repaired_fft = np.fft.rfft2(repaired_endpoint * window[..., None], axes=(0, 1))
            endpoint_auto = np.abs(endpoint_fft) ** 2
            paired_auto = np.abs(paired_fft) ** 2
            cross = np.abs(endpoint_fft * np.conj(paired_fft))
            high_auto = float(endpoint_auto[high_band].mean())
            high_other = float(paired_auto[high_band].mean())
            high_cross = float(cross[high_band].mean())
            exclusive = max(high_auto - high_cross, 0.0)
            exclusive_fraction = exclusive / max(high_auto, high_other, 1e-12)
            # The endpoint auto-spectrum contains the very noise that the
            # action is supposed to remove, so auto/auto would call successful
            # denoising a loss of signal.  Cross-power with the untouched paired
            # measurement estimates the shared observable signal instead.
            signal_native = float(cross[signal_band].mean())
            repaired_cross = np.abs(repaired_fft * np.conj(paired_fft))
            signal_repaired = float(repaired_cross[signal_band].mean())
            retention = signal_repaired / max(signal_native, 1e-12)
            residual = native_endpoint - paired_endpoint
            lag = _lag_correlation(residual, tile_valid)
            sigma = float(np.median(np.abs(residual[tile_valid])))
            sigma *= 255.0 / 0.6744897501960817

            parity = ((top // tile_size) + (left // tile_size)) % 2
            gains[parity].append(gain)
            exclusive_values.append(exclusive_fraction)
            lag_values.append(lag)
            retention_values.append(retention)
            sigma_values.append(sigma)
            accepted_tiles.append((top, left))

    fit_gain = _lower_mean(np.asarray(gains[0]))
    check_gain = _lower_mean(np.asarray(gains[1]))
    exclusive_power = float(np.median(exclusive_values)) if exclusive_values else 0.0
    lag_correlation = float(np.median(lag_values)) if lag_values else 1.0
    signal_retention = float(np.median(retention_values)) if retention_values else 0.0
    noise_sigma = float(np.median(sigma_values)) if sigma_values else 0.0
    reasons: list[str] = []
    if len(gains[0]) < minimum_tiles_per_fold:
        reasons.append("insufficient_fit_tiles")
    if len(gains[1]) < minimum_tiles_per_fold:
        reasons.append("insufficient_check_tiles")
    if fit_gain <= 0.0:
        reasons.append("nonpositive_fit_action_gain")
    if check_gain <= 0.0:
        reasons.append("nonpositive_check_action_gain")
    if exclusive_power < minimum_exclusive_power_fraction:
        reasons.append("insufficient_endpoint_exclusive_power")
    if lag_correlation > maximum_lag_correlation_abs:
        reasons.append("residual_not_white_enough")
    if signal_retention < minimum_signal_band_retention:
        reasons.append("signal_band_retention_too_low")

    support = np.zeros(valid.shape, dtype=bool)
    for top, left in accepted_tiles:
        support[top:top + tile_size, left:left + tile_size] = valid[
            top:top + tile_size, left:left + tile_size
        ]
    support = np.ascontiguousarray(support)
    support_hash = hashlib.sha256(support.tobytes()).hexdigest()
    return PairedWienerPSDReceipt(
        endpoint=endpoint,
        status="supported" if not reasons else "rejected",
        tile_size=tile_size,
        valid_tiles=len(accepted_tiles),
        fit_tiles=len(gains[0]),
        check_tiles=len(gains[1]),
        fit_gain_lower=fit_gain,
        check_gain_lower=check_gain,
        endpoint_exclusive_power_fraction=exclusive_power,
        residual_lag_correlation_abs=lag_correlation,
        signal_band_retention=signal_retention,
        estimated_noise_sigma_255=noise_sigma,
        support_mask=support,
        support_hash=support_hash,
        rejection_reasons=tuple(reasons),
    )


def _paired_receipts(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    *,
    tile_size: int,
    minimum_tiles_per_fold: int,
    minimum_valid_fraction: float,
    minimum_exclusive_power_fraction: float,
    maximum_lag_correlation_abs: float,
    minimum_signal_band_retention: float,
) -> dict[str, PairedWienerPSDReceipt]:
    first, second, flow = _validate(first, second, flow)
    if tile_size < 16 or minimum_tiles_per_fold < 1:
        raise ValueError("invalid tile configuration")
    for value, name in (
        (minimum_valid_fraction, "minimum valid fraction"),
        (minimum_exclusive_power_fraction, "minimum exclusive power"),
        (maximum_lag_correlation_abs, "maximum lag correlation"),
        (minimum_signal_band_retention, "minimum signal retention"),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0,1]")

    full = np.ones(first.shape[:2], dtype=np.float32)
    repaired_first, _ = apply_local_image_action(
        first, full, operator_id="wiener3", endpoint="first",
    )
    repaired_second, _ = apply_local_image_action(
        second, full, operator_id="wiener3", endpoint="second",
    )
    first_hp = _highpass(first)
    second_hp = _highpass(second)
    repaired_first_hp = _highpass(repaired_first)
    repaired_second_hp = _highpass(repaired_second)
    warped_second, valid = _warp_second(second_hp, flow)
    warped_repaired_second, repaired_valid = _warp_second(repaired_second_hp, flow)
    valid &= repaired_valid
    common = {
        "first_hp": first_hp,
        "warped_second": warped_second,
        "repaired_first_hp": repaired_first_hp,
        "warped_repaired_second": warped_repaired_second,
        "valid": valid,
        "tile_size": tile_size,
        "minimum_tiles_per_fold": minimum_tiles_per_fold,
        "minimum_valid_fraction": minimum_valid_fraction,
        "minimum_exclusive_power_fraction": minimum_exclusive_power_fraction,
        "maximum_lag_correlation_abs": maximum_lag_correlation_abs,
        "minimum_signal_band_retention": minimum_signal_band_retention,
    }
    return {
        endpoint: _receipt_from_precomputed(endpoint=endpoint, **common)
        for endpoint in ("first", "second")
    }


def paired_wiener_psd_receipt(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    *,
    endpoint: str,
    tile_size: int = 64,
    minimum_tiles_per_fold: int = 4,
    minimum_valid_fraction: float = 0.85,
    minimum_exclusive_power_fraction: float = 0.02,
    maximum_lag_correlation_abs: float = 0.35,
    minimum_signal_band_retention: float = 0.50,
) -> PairedWienerPSDReceipt:
    """Build one endpoint receipt while sharing the paired computation path."""
    if endpoint not in {"first", "second"}:
        raise ValueError("endpoint must be first or second")
    return _paired_receipts(
        first, second, flow,
        tile_size=tile_size,
        minimum_tiles_per_fold=minimum_tiles_per_fold,
        minimum_valid_fraction=minimum_valid_fraction,
        minimum_exclusive_power_fraction=minimum_exclusive_power_fraction,
        maximum_lag_correlation_abs=maximum_lag_correlation_abs,
        minimum_signal_band_retention=minimum_signal_band_retention,
    )[endpoint]


def paired_wiener_psd_receipts(first: np.ndarray, second: np.ndarray,
                               flow: np.ndarray
                               ) -> dict[str, PairedWienerPSDReceipt]:
    """Evaluate both endpoints independently with shared, exact computation."""
    return _paired_receipts(
        first, second, flow,
        tile_size=64,
        minimum_tiles_per_fold=4,
        minimum_valid_fraction=0.85,
        minimum_exclusive_power_fraction=0.02,
        maximum_lag_correlation_abs=0.35,
        minimum_signal_band_retention=0.50,
    )
