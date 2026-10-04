"""Proposal-independent flow verifier from a native forward/reverse cycle.

For native forward flow ``f`` and a reverse flow ``r`` computed on the same
unmodified image pair, a reverse prediction sampled at ``x + f(x)`` gives the
flow-valued verifier ``v(x) = -r(x + f(x))``.  It never consumes a repaired
candidate.  Occluded, out-of-frame, nonfinite, or interpolation-mixed reverse
samples are invalid and must be removed before a task bound is formed.

Cycle closure by itself is not a correctness certificate.  The error of this
verifier still needs a frozen, selection-aware calibration radius before it can
be passed to ``independent_verifier_squared_l2_bound``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import cv2
import numpy as np

from .verifier_task_bounds import support_mask_sha256


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class NativeCycleVerifierEvidence:
    verifier_flow: np.ndarray
    valid_support: np.ndarray
    native_forward_hash: str
    native_reverse_hash: str
    verifier_output_hash: str
    valid_support_hash: str
    valid_fraction: float
    construction: str = "negative_native_reverse_at_native_forward_endpoint"
    action_blind: bool = True

    def __post_init__(self) -> None:
        verifier = np.asarray(self.verifier_flow)
        valid = np.asarray(self.valid_support)
        if verifier.ndim != 3 or verifier.shape[-1] != 2:
            raise ValueError("cycle verifier must be HxWx2")
        if valid.shape != verifier.shape[:2] or valid.dtype != np.bool_:
            raise ValueError("cycle verifier support must be a boolean HxW mask")
        if not np.all(np.isfinite(verifier[valid])):
            raise ValueError("valid verifier values must be finite")
        if any(not value for value in (
            self.native_forward_hash, self.native_reverse_hash,
            self.verifier_output_hash, self.valid_support_hash,
            self.construction,
        )):
            raise ValueError("cycle verifier provenance is required")
        if self.valid_support_hash != support_mask_sha256(valid):
            raise ValueError("cycle verifier support hash changed")
        if not 0.0 <= float(self.valid_fraction) <= 1.0:
            raise ValueError("valid fraction must lie in [0,1]")
        if not self.action_blind:
            raise ValueError("native cycle verifier must be action-blind")


def native_cycle_verifier(
    native_forward: np.ndarray,
    native_reverse: np.ndarray,
) -> NativeCycleVerifierEvidence:
    """Construct ``-reverse(x + forward(x))`` with strict interpolation validity."""
    forward_input = np.asarray(native_forward)
    reverse_input = np.asarray(native_reverse)
    forward = np.asarray(native_forward, dtype=np.float32)
    reverse = np.asarray(native_reverse, dtype=np.float32)
    if (forward.shape != reverse.shape or forward.ndim != 3
            or forward.shape[-1] != 2):
        raise ValueError("native forward/reverse flows must share HxWx2 shape")
    height, width = forward.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    target_x = xx + forward[..., 0]
    target_y = yy + forward[..., 1]
    reverse_finite = np.isfinite(reverse).all(axis=2)
    sampled_reverse = cv2.remap(
        np.nan_to_num(reverse, nan=0.0, posinf=0.0, neginf=0.0),
        target_x, target_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    # Requiring an interpolated finite-indicator of exactly one prevents a
    # single invalid reverse neighbour from entering a bilinear sample.
    finite_sample = cv2.remap(
        reverse_finite.astype(np.float32), target_x, target_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    ) >= 1.0
    valid = (
        np.isfinite(forward).all(axis=2)
        & np.isfinite(target_x) & np.isfinite(target_y)
        & (target_x >= 0.0) & (target_x <= width - 1.0)
        & (target_y >= 0.0) & (target_y <= height - 1.0)
        & finite_sample
    )
    verifier = np.ascontiguousarray(-sampled_reverse, dtype=np.float32)
    verifier[~valid] = 0.0
    valid = np.ascontiguousarray(valid, dtype=bool)
    return NativeCycleVerifierEvidence(
        verifier_flow=verifier,
        valid_support=valid,
        native_forward_hash=_array_hash(forward_input),
        native_reverse_hash=_array_hash(reverse_input),
        verifier_output_hash=_array_hash(verifier),
        valid_support_hash=support_mask_sha256(valid),
        valid_fraction=float(valid.mean()),
    )
