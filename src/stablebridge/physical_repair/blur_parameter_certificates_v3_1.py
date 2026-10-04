"""Fail-closed observable-support wrapper around the frozen blur v3 bank.

The v3 family certificates require enough textured, visible 64x64 blocks on
both A and B folds.  A broad H1 audit exposed a low-texture JPEG pair for which
the internal simultaneous bound is correctly undefined (``-inf``), but v3
attempted to serialize that value as a certificate diagnostic.  This successor
does not manufacture a finite score.  It checks identifiability support first
and returns an empty action bank with explicit reasons when a fold is too small.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .blur_parameter_certificates import (
    BlurFamilyEvidence,
    ENDPOINTS,
    FAMILYWISE_ALPHA,
    MIN_BLOCKS_PER_SPLIT,
    _blocks,
    _validate,
    _warp_second,
)
from .blur_parameter_certificates_v3 import blur_parameter_certificates_v3


@dataclass(frozen=True)
class BlurCertificateBank:
    """Evidence plus the observable-support decision for a whole blur bank."""

    status: str
    evidence: Mapping[str, BlurFamilyEvidence]
    fold_block_counts: Mapping[str, int]
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"evaluated", "unsupported"}:
            raise ValueError("invalid blur certificate bank status")
        expected = {
            f"{endpoint}_{fold}"
            for endpoint in ENDPOINTS for fold in ("fit", "check")
        }
        if set(self.fold_block_counts) != expected:
            raise ValueError("blur bank needs every endpoint/fold block count")
        if any(type(value) is not int or value < 0
               for value in self.fold_block_counts.values()):
            raise ValueError("blur bank block counts must be nonnegative integers")
        if self.status == "evaluated":
            if self.rejection_reasons or not self.evidence:
                raise ValueError("evaluated blur bank needs evidence and no rejection")
        elif self.evidence or not self.rejection_reasons:
            raise ValueError("unsupported blur bank must fail closed with a reason")


def blur_parameter_certificate_bank_v3_1(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    *,
    alpha: float = FAMILYWISE_ALPHA,
) -> BlurCertificateBank:
    """Evaluate v3 only when both endpoints have identifiable A/B support."""
    first, second, flow = _validate(first, second, flow)
    counts = observable_blur_fold_counts(first, second, flow)
    reasons = [
        f"{name}_observable_blocks"
        for name, count in counts.items()
        if count < MIN_BLOCKS_PER_SPLIT
    ]
    if reasons:
        return BlurCertificateBank(
            status="unsupported", evidence={}, fold_block_counts=counts,
            rejection_reasons=tuple(reasons),
        )
    try:
        evidence = blur_parameter_certificates_v3(
            first, second, flow, alpha=alpha,
        )
    except ValueError as error:
        if str(error) != "certificate diagnostics must be finite":
            raise
        return BlurCertificateBank(
            status="unsupported", evidence={}, fold_block_counts=counts,
            rejection_reasons=("nonfinite_simultaneous_bound",),
        )
    return BlurCertificateBank(
        status="evaluated", evidence=evidence, fold_block_counts=counts,
    )


def observable_blur_fold_counts(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
) -> dict[str, int]:
    """Count independent textured blocks without fitting any blur model."""
    first, second, flow = _validate(first, second, flow)
    warped_second, valid = _warp_second(second, flow)
    counts: dict[str, int] = {}
    for endpoint in ENDPOINTS:
        sharp, target = (
            (warped_second, first.astype(np.float32)) if endpoint == "first"
            else (first.astype(np.float32), warped_second)
        )
        blocks = _blocks(valid, sharp, target)
        for parity, fold in ((0, "fit"), (1, "check")):
            count = sum(block[2] == parity for block in blocks)
            counts[f"{endpoint}_{fold}"] = count
    return counts
