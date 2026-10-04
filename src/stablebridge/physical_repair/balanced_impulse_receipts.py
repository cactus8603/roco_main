"""Two-tail physical receipts for random salt-and-pepper contamination.

The receipt does not classify a renderer family.  It asks whether *each* tail
of the exact median action event exceeds its own diffuse-clipping null.  This
prevents a large one-sided collection of natural white or black extrema from
borrowing evidence from the absent opposite contamination tail.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import cv2
import numpy as np
from scipy.special import ndtr

from .operators import impulse_support


def _hash(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask, dtype=bool).tobytes()
    ).hexdigest()


def _image(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.uint8 or result.ndim != 3 or result.shape[2] != 3:
        raise ValueError("expected uint8 RGB")
    return np.ascontiguousarray(result)


def _noise_sigma_rgb(image: np.ndarray) -> tuple[float, float, float]:
    source = image.astype(np.float32)
    output = []
    for channel in range(3):
        laplacian = cv2.Laplacian(
            source[..., channel], cv2.CV_32F, ksize=1,
            borderType=cv2.BORDER_REFLECT_101,
        )
        center = float(np.median(laplacian))
        sigma = float(
            np.median(np.abs(laplacian - center))
            / 0.6744897501960817 / math.sqrt(20.0)
        )
        output.append(max(sigma, 0.5))
    return tuple(output)  # type: ignore[return-value]


def _upper(expected: float, valid_pixels: int, alpha: float,
           hypotheses_tested: int) -> float:
    log_term = math.log(float(hypotheses_tested) / float(alpha))
    return min(
        float(valid_pixels),
        expected + math.sqrt(2.0 * expected * log_term)
        + 2.0 * log_term / 3.0,
    )


@dataclass(frozen=True)
class BalancedImpulseReceipt:
    endpoint: str
    alpha: float
    hypotheses_tested: int
    valid_pixels: int
    observed_black_events: int
    observed_white_events: int
    expected_black_upper_mean: float
    expected_white_upper_mean: float
    simultaneous_black_count_upper: float
    simultaneous_white_count_upper: float
    black_excess_lower: float
    white_excess_lower: float
    black_fraction_of_events: float
    sigma_rgb_255: tuple[float, float, float]
    combined_support_mask: np.ndarray
    black_support_hash: str
    white_support_hash: str
    combined_support_hash: str
    supported: bool
    law_id: str = "separate_black_white_exact_median_event_excess_v1"
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if self.endpoint not in {"first", "second"}:
            raise ValueError("balanced impulse receipt needs a concrete endpoint")
        if not 0.0 < self.alpha < 1.0 or self.hypotheses_tested < 2:
            raise ValueError("invalid two-tail multiplicity")
        if min(self.observed_black_events, self.observed_white_events) < 0:
            raise ValueError("invalid event count")
        mask = np.ascontiguousarray(self.combined_support_mask)
        if mask.dtype != np.bool_ or mask.ndim != 2:
            raise ValueError("combined support must be boolean HxW")
        if int(mask.sum()) != self.observed_black_events + self.observed_white_events:
            raise ValueError("combined event count drift")
        if _hash(mask) != self.combined_support_hash:
            raise ValueError("combined support hash mismatch")
        expected = self.black_excess_lower > 0.0 and self.white_excess_lower > 0.0
        if self.supported != expected:
            raise ValueError("two-tail decision drift")
        if not all(np.isfinite((
            self.expected_black_upper_mean, self.expected_white_upper_mean,
            self.simultaneous_black_count_upper,
            self.simultaneous_white_count_upper, self.black_excess_lower,
            self.white_excess_lower, self.black_fraction_of_events,
            *self.sigma_rgb_255,
        ))):
            raise ValueError("two-tail values must be finite")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("two-tail receipt cannot read labels or outcomes")
        object.__setattr__(self, "combined_support_mask", mask)


def balanced_impulse_receipt(
    image: np.ndarray,
    *,
    endpoint: str,
    alpha: float = 0.05,
    hypotheses_tested: int = 16,
) -> BalancedImpulseReceipt:
    """Test black and white exact-median events against separate nulls."""
    image = _image(image)
    if endpoint not in {"first", "second"}:
        raise ValueError("balanced impulse receipt needs a concrete endpoint")
    height, width = image.shape[:2]
    valid = np.zeros((height, width), dtype=bool)
    valid[1:-1, 1:-1] = True
    combined = (impulse_support(image) > 0.0) & valid
    black = combined & np.all(image <= 1, axis=2)
    white = combined & np.all(image >= 254, axis=2)
    median = cv2.medianBlur(image, 3).astype(np.float64)
    median_black = np.all(median <= 1.0, axis=2)
    median_white = np.all(median >= 254.0, axis=2)
    sigma = np.asarray(_noise_sigma_rgb(image), dtype=np.float64)
    black_probability = (
        np.min(ndtr((1.5 - median) / sigma.reshape(1, 1, 3)), axis=2)
        * (~median_black)
    )
    white_probability = (
        np.min(ndtr((median - 253.5) / sigma.reshape(1, 1, 3)), axis=2)
        * (~median_white)
    )
    expected_black = float(black_probability[valid].sum())
    expected_white = float(white_probability[valid].sum())
    black_upper = _upper(
        expected_black, int(valid.sum()), alpha, hypotheses_tested
    )
    white_upper = _upper(
        expected_white, int(valid.sum()), alpha, hypotheses_tested
    )
    black_count, white_count = int(black.sum()), int(white.sum())
    total = black_count + white_count
    return BalancedImpulseReceipt(
        endpoint=endpoint, alpha=float(alpha),
        hypotheses_tested=int(hypotheses_tested),
        valid_pixels=int(valid.sum()),
        observed_black_events=black_count,
        observed_white_events=white_count,
        expected_black_upper_mean=expected_black,
        expected_white_upper_mean=expected_white,
        simultaneous_black_count_upper=black_upper,
        simultaneous_white_count_upper=white_upper,
        black_excess_lower=float(black_count - black_upper),
        white_excess_lower=float(white_count - white_upper),
        black_fraction_of_events=float(black_count / total) if total else 0.0,
        sigma_rgb_255=tuple(float(value) for value in sigma),
        combined_support_mask=np.ascontiguousarray(combined),
        black_support_hash=_hash(black), white_support_hash=_hash(white),
        combined_support_hash=_hash(combined),
        supported=black_count > black_upper and white_count > white_upper,
    )
