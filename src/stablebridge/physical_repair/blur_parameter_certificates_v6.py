"""Outcome-blind isotropic/motion blur-arm certificate proposed by E261.

The historical v3 certificate fits disk, Gaussian, and motion independently.
E261 showed that disk and Gaussian occupy one practically overlapping
isotropic intervention family: exposing them as separate selector arms loses
availability without changing the local operator required at execution.  This
version therefore selects disk or Gaussian as an internal A-fold nuisance mode
and exposes one ``common_isotropic`` arm plus one ``common_motion`` arm.

Every retained 64 px region already has positive held-out spatial gain and
information retention >= 0.05.  The v2 global information-median gate is thus
not repeated.  Motion retains directional spectral separation and accepts
spatial family differences only within a one-third-LSB non-inferiority margin.

This module is a development/mechanism primitive.  It reads no task GT or
action outcome and grants no scientific, selector, or production authority.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

import numpy as np

from .blur_parameter_certificates import (
    BlurFamilyEvidence,
    ENDPOINTS,
    FAMILYWISE_ALPHA,
    MIN_BLOCKS_PER_SPLIT,
    MIN_ENDPOINT_GAIN,
)
from .blur_parameter_certificates_v3 import blur_parameter_certificates_v3
from .contracts import ActionSpec, PhysicalCertificate


ISOTROPIC_ARM_V6 = "common_isotropic"
MOTION_ARM_V6 = "common_motion"
BLUR_ARMS_V6 = (ISOTROPIC_ARM_V6, MOTION_ARM_V6)
ISOTROPIC_INTERNAL_FAMILIES_V6 = ("disk", "gaussian")
SPATIAL_NONINFERIORITY_MARGIN_V6 = 1.0 / (3.0 * 255.0)
CALIBRATION_VERSION_V6 = "blur-v6-e261-isotropic-motion-development"


def _fit_score(value: BlurFamilyEvidence) -> float:
    key = ",".join(f"{float(item):g}" for item in value.selected_parameter)
    score = float(value.fit_score_curve[key])
    if not math.isfinite(score):
        raise RuntimeError("selected blur fit score must be finite")
    return score


@dataclass(frozen=True)
class BlurArmEvidenceV6:
    """One public blur arm backed by a selected internal physical operator."""

    arm: str
    endpoint: str
    internal_family: str
    selected_parameter: tuple[float, ...]
    source_action_key: str
    fit_support_sha256: str
    check_support_sha256: str
    fit_score: float
    check_blocks: int
    own_null_improvement_lcb: float
    endpoint_improvement_lcb: float
    pairwise_improvement_lcb: Mapping[str, float]
    information_retention_median: float
    recoverable_regions: tuple[tuple[int, int], ...]
    status: str
    rejection_reasons: tuple[str, ...]
    certificate: PhysicalCertificate

    def __post_init__(self) -> None:
        if self.arm not in BLUR_ARMS_V6 or self.endpoint not in ENDPOINTS:
            raise ValueError("invalid v6 blur arm or endpoint")
        if self.internal_family not in {
            *ISOTROPIC_INTERNAL_FAMILIES_V6, "motion",
        }:
            raise ValueError("invalid v6 internal blur family")
        if self.arm == ISOTROPIC_ARM_V6 and self.internal_family == "motion":
            raise ValueError("isotropic arm cannot use the motion operator")
        if self.arm == MOTION_ARM_V6 and self.internal_family != "motion":
            raise ValueError("motion arm must use the motion operator")
        if not self.selected_parameter or not self.source_action_key:
            raise ValueError("v6 blur evidence needs source parameter evidence")
        if self.source_action_key != (
            f"common_{self.internal_family}@{self.endpoint}"
        ):
            raise ValueError("v6 blur source-action lineage mismatch")
        if (
            len(self.fit_support_sha256) != 64
            or len(self.check_support_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in self.fit_support_sha256)
            or any(character not in "0123456789abcdef"
                   for character in self.check_support_sha256)
            or self.fit_support_sha256 == self.check_support_sha256
        ):
            raise ValueError("v6 blur evidence needs distinct A/B support hashes")
        if not math.isfinite(float(self.fit_score)) or self.check_blocks < 0:
            raise ValueError("invalid v6 blur fit evidence")
        if self.status not in {"supported", "rejected"}:
            raise ValueError("invalid v6 blur-arm status")
        if (self.status == "supported") == bool(self.rejection_reasons):
            raise ValueError("v6 blur status and rejection reasons disagree")
        if self.certificate.status != self.status:
            raise ValueError("v6 blur evidence and physical certificate disagree")

    @property
    def action_key(self) -> str:
        return f"{self.arm}@{self.endpoint}"


def _certificate(
    selected: BlurFamilyEvidence,
    *,
    arm: str,
    reasons: list[str],
    fit_score: float,
) -> PhysicalCertificate:
    status = "supported" if not reasons else "rejected"
    modified = "second" if selected.endpoint == "first" else "first"
    action = ActionSpec(
        arm,
        "v6-e261-isotropic-motion-development",
        "image",
        selected.endpoint,
        modified,
        "flow_native",
    )
    diagnostics = dict(selected.certificate.diagnostics)
    diagnostics.update({
        "v6_selected_fit_score": fit_score,
        "v6_recoverable_regions": float(len(selected.recoverable_regions)),
        "v6_global_information_median_is_diagnostic_only": 1.0,
        "v6_spatial_noninferiority_margin": (
            SPATIAL_NONINFERIORITY_MARGIN_V6 if arm == MOTION_ARM_V6 else 0.0
        ),
    })
    competing = dict(selected.certificate.competing_model_scores)
    competing.update({
        f"v6_{name}_lcb": float(value)
        for name, value in selected.pairwise_improvement_lcb.items()
    })
    competing["v6_endpoint_same_parameter_lcb"] = float(
        selected.endpoint_improvement_lcb
    )
    return replace(
        selected.certificate,
        action=action,
        status=status,
        identifiable_support_fraction=float(
            len(selected.recoverable_regions) / max(selected.check_blocks, 1)
        ),
        competing_model_scores=competing,
        diagnostics=diagnostics,
        rejection_reasons=tuple(reasons),
        calibration_version=CALIBRATION_VERSION_V6,
    )


def _isotropic(
    evidence: Mapping[str, BlurFamilyEvidence], endpoint: str,
) -> BlurArmEvidenceV6:
    selected = min(
        (evidence[f"common_{family}@{endpoint}"]
         for family in ISOTROPIC_INTERNAL_FAMILIES_V6),
        key=lambda value: (
            _fit_score(value), value.family, tuple(value.selected_parameter),
        ),
    )
    reasons: list[str] = []
    if selected.check_blocks < MIN_BLOCKS_PER_SPLIT:
        reasons.append("INSUFFICIENT_CHECK_BLOCKS")
    if selected.own_null_improvement_lcb <= 0.0:
        reasons.append("OWN_NULL_LCB_NOT_POSITIVE")
    if selected.endpoint_improvement_lcb <= MIN_ENDPOINT_GAIN:
        reasons.append("ENDPOINT_MARGIN_NOT_ABOVE_ONE_LSB")
    if selected.pairwise_improvement_lcb["spatial_vs_motion"] <= 0.0:
        reasons.append("NOT_SPATIALLY_SEPARATED_FROM_MOTION")
    if len(selected.recoverable_regions) < MIN_BLOCKS_PER_SPLIT:
        reasons.append("FEWER_THAN_FOUR_LOCAL_RECOVERABLE_REGIONS")
    score = _fit_score(selected)
    certificate = _certificate(
        selected, arm=ISOTROPIC_ARM_V6, reasons=reasons, fit_score=score,
    )
    return BlurArmEvidenceV6(
        arm=ISOTROPIC_ARM_V6,
        endpoint=endpoint,
        internal_family=selected.family,
        selected_parameter=tuple(map(float, selected.selected_parameter)),
        source_action_key=selected.action_key,
        fit_support_sha256=selected.fit_support_hash,
        check_support_sha256=selected.check_support_hash,
        fit_score=score,
        check_blocks=int(selected.check_blocks),
        own_null_improvement_lcb=float(selected.own_null_improvement_lcb),
        endpoint_improvement_lcb=float(selected.endpoint_improvement_lcb),
        pairwise_improvement_lcb=dict(selected.pairwise_improvement_lcb),
        information_retention_median=float(
            selected.information_retention_median
        ),
        recoverable_regions=tuple(selected.recoverable_regions),
        status=certificate.status,
        rejection_reasons=tuple(reasons),
        certificate=certificate,
    )


def _motion(
    evidence: Mapping[str, BlurFamilyEvidence], endpoint: str,
) -> BlurArmEvidenceV6:
    selected = evidence[f"common_motion@{endpoint}"]
    pairwise = selected.pairwise_improvement_lcb
    reasons: list[str] = []
    if selected.check_blocks < MIN_BLOCKS_PER_SPLIT:
        reasons.append("INSUFFICIENT_CHECK_BLOCKS")
    if selected.own_null_improvement_lcb <= 0.0:
        reasons.append("OWN_NULL_LCB_NOT_POSITIVE")
    if selected.endpoint_improvement_lcb <= MIN_ENDPOINT_GAIN:
        reasons.append("ENDPOINT_MARGIN_NOT_ABOVE_ONE_LSB")
    if pairwise["spectral_vs_disk"] <= 0.0:
        reasons.append("SPECTRAL_DIRECTIONALITY_VS_DISK_NOT_POSITIVE")
    if pairwise["spectral_vs_gaussian"] <= 0.0:
        reasons.append("SPECTRAL_DIRECTIONALITY_VS_GAUSSIAN_NOT_POSITIVE")
    if pairwise["spatial_vs_disk"] <= -SPATIAL_NONINFERIORITY_MARGIN_V6:
        reasons.append("SPATIAL_VS_DISK_BELOW_QUANTIZATION_NONINFERIORITY")
    if pairwise["spatial_vs_gaussian"] <= -SPATIAL_NONINFERIORITY_MARGIN_V6:
        reasons.append("SPATIAL_VS_GAUSSIAN_BELOW_QUANTIZATION_NONINFERIORITY")
    if len(selected.recoverable_regions) < MIN_BLOCKS_PER_SPLIT:
        reasons.append("FEWER_THAN_FOUR_LOCAL_RECOVERABLE_REGIONS")
    score = _fit_score(selected)
    certificate = _certificate(
        selected, arm=MOTION_ARM_V6, reasons=reasons, fit_score=score,
    )
    return BlurArmEvidenceV6(
        arm=MOTION_ARM_V6,
        endpoint=endpoint,
        internal_family="motion",
        selected_parameter=tuple(map(float, selected.selected_parameter)),
        source_action_key=selected.action_key,
        fit_support_sha256=selected.fit_support_hash,
        check_support_sha256=selected.check_support_hash,
        fit_score=score,
        check_blocks=int(selected.check_blocks),
        own_null_improvement_lcb=float(selected.own_null_improvement_lcb),
        endpoint_improvement_lcb=float(selected.endpoint_improvement_lcb),
        pairwise_improvement_lcb=dict(pairwise),
        information_retention_median=float(
            selected.information_retention_median
        ),
        recoverable_regions=tuple(selected.recoverable_regions),
        status=certificate.status,
        rejection_reasons=tuple(reasons),
        certificate=certificate,
    )


def blur_arm_evidence_v6(
    evidence: Mapping[str, BlurFamilyEvidence],
) -> dict[str, BlurArmEvidenceV6]:
    """Collapse frozen family evidence into two public arm hypotheses."""
    expected = {
        f"common_{family}@{endpoint}"
        for family in ("disk", "gaussian", "motion")
        for endpoint in ENDPOINTS
    }
    if set(evidence) != expected:
        raise ValueError("v6 needs the complete six-row v3 blur evidence map")
    output: dict[str, BlurArmEvidenceV6] = {}
    for endpoint in ("first", "second"):
        for item in (_isotropic(evidence, endpoint), _motion(evidence, endpoint)):
            output[item.action_key] = item
    return output


def blur_parameter_certificates_v6(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    *,
    alpha: float = FAMILYWISE_ALPHA,
) -> dict[str, BlurArmEvidenceV6]:
    """Build v3 family evidence and collapse it into the E261 v6 arms."""
    return blur_arm_evidence_v6(
        blur_parameter_certificates_v3(first, second, flow, alpha=alpha)
    )


def action_specific_direct_winners_v6(
    evidence: Mapping[str, BlurArmEvidenceV6],
) -> tuple[str, ...]:
    """Return all supported arm/endpoint candidates in canonical order.

    Callers that materialize an action must still require exactly one result;
    zero and multiple candidates are fail-closed abstentions.
    """
    return tuple(
        key for key, value in sorted(evidence.items())
        if value.status == "supported"
    )


__all__ = [
    "BLUR_ARMS_V6",
    "CALIBRATION_VERSION_V6",
    "ISOTROPIC_ARM_V6",
    "ISOTROPIC_INTERNAL_FAMILIES_V6",
    "MOTION_ARM_V6",
    "SPATIAL_NONINFERIORITY_MARGIN_V6",
    "BlurArmEvidenceV6",
    "action_specific_direct_winners_v6",
    "blur_arm_evidence_v6",
    "blur_parameter_certificates_v6",
]
