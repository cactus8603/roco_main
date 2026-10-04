"""Metric-aligned EPE bounds from an action-blind verifier.

The existing squared-L2 certificate is useful when the task metric is MSE.
It must not be relabelled as an endpoint-error certificate.  This module uses
the triangle inequality directly in the final endpoint L2 metric.

Let ``n`` be the native output, ``c = n + d`` an exact matcher rerun, ``v`` an
action-blind verifier, and ``y`` the unknown task target.  On one immutable
support, assume a frozen, selection-aware calibration guarantees

    mean(||v - y||_2) <= r.

Then for every output trust ``beta`` the endpoint-error gain obeys

    EPE(n, y) - EPE(n + beta*d, y)
      >= mean(||n-v||_2 - ||n + beta*d-v||_2) - 2*r.

The corresponding signed harm has the reverse upper bound.  The result is a
mean-on-the-declared-support certificate; it is not a pixel-maximum or severe
tail certificate.  Those require a separately calibrated tail object.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .verifier_task_bounds import support_mask_sha256


EPE_METRIC_ID = "mean_endpoint_l2_px"


def _finite(value: float, name: str) -> float:
    item = float(value)
    if not np.isfinite(item):
        raise ValueError(f"{name} must be finite")
    return item


def _nonnegative(value: float, name: str) -> float:
    item = _finite(value, name)
    if item < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return item


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EPEVerifierCalibration:
    """Frozen verifier radius in the same mean endpoint metric and support policy."""

    verifier_id: str
    representation_id: str
    calibration_version: str
    calibration_data_hash: str
    support_policy_hash: str
    mean_epe_radius_upper: float
    effective_n: float
    confidence_level: float
    selection_aware: bool
    action_blind: bool
    frozen_before_selection: bool
    metric_id: str = EPE_METRIC_ID
    estimand: str = "row_mean_on_exact_runtime_support"

    def __post_init__(self) -> None:
        identities = (
            self.verifier_id, self.representation_id,
            self.calibration_version, self.calibration_data_hash,
            self.support_policy_hash, self.metric_id, self.estimand,
        )
        if any(not value for value in identities):
            raise ValueError("EPE verifier calibration identity is required")
        _nonnegative(self.mean_epe_radius_upper, "mean EPE radius")
        if _nonnegative(self.effective_n, "effective n") <= 0.0:
            raise ValueError("effective n must be positive")
        confidence = _finite(self.confidence_level, "confidence level")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence level must lie in (0,1)")
        if self.metric_id != EPE_METRIC_ID:
            raise ValueError("calibration metric is not mean endpoint L2")
        if self.estimand != "row_mean_on_exact_runtime_support":
            raise ValueError("unsupported EPE calibration estimand")


@dataclass(frozen=True)
class IndependentVerifierEPEPathBound:
    """A nonlinear, metric-aligned trust path for one exact control rerun."""

    action_key: str
    control_key: str
    control_selection_receipt_hash: str
    source_input_hash: str
    verifier_input_hash: str
    task_support_hash: str
    task_support_content_hash: str
    task_support_policy_hash: str
    native_output_hash: str
    candidate_output_hash: str
    verifier_output_hash: str
    verifier_id: str
    calibration_version: str
    calibration_data_hash: str
    support_count: int
    mean_epe_radius_upper: float
    maximum_output_trust: float
    native_verifier_residual: np.ndarray
    candidate_direction: np.ndarray
    metric_id: str = EPE_METRIC_ID

    def __post_init__(self) -> None:
        identities = (
            self.action_key, self.control_key,
            self.control_selection_receipt_hash, self.source_input_hash,
            self.verifier_input_hash, self.task_support_hash,
            self.task_support_content_hash, self.task_support_policy_hash,
            self.native_output_hash, self.candidate_output_hash,
            self.verifier_output_hash, self.verifier_id,
            self.calibration_version, self.calibration_data_hash,
            self.metric_id,
        )
        if any(not value for value in identities):
            raise ValueError("EPE task-bound identity and provenance are required")
        if self.source_input_hash != self.verifier_input_hash:
            raise ValueError("verifier must be computed from the native source")
        if self.support_count < 1:
            raise ValueError("task support must be nonempty")
        _nonnegative(self.mean_epe_radius_upper, "mean EPE radius")
        maximum = _finite(self.maximum_output_trust, "maximum output trust")
        if not 0.0 < maximum <= 1.0:
            raise ValueError("maximum output trust must lie in (0,1]")
        if self.metric_id != EPE_METRIC_ID:
            raise ValueError("task bound metric is not mean endpoint L2")
        residual = np.ascontiguousarray(
            self.native_verifier_residual, dtype=np.float64,
        )
        direction = np.ascontiguousarray(self.candidate_direction, dtype=np.float64)
        if (residual.shape != direction.shape or residual.ndim != 2
                or residual.shape[0] != self.support_count
                or residual.shape[1] < 1
                or not np.isfinite(residual).all()
                or not np.isfinite(direction).all()):
            raise ValueError("EPE path vectors must be finite aligned NxD arrays")
        if not np.any(direction != 0.0):
            raise ValueError("candidate has no effect on task support")
        residual.setflags(write=False)
        direction.setflags(write=False)
        object.__setattr__(self, "native_verifier_residual", residual)
        object.__setattr__(self, "candidate_direction", direction)

    def _beta(self, beta: float) -> float:
        value = _finite(beta, "output trust beta")
        if not 0.0 <= value <= self.maximum_output_trust:
            raise ValueError("output trust lies outside the certified path")
        return value

    def nominal_verifier_gain(self, beta: float) -> float:
        """Observable EPE reduction toward the action-blind verifier."""
        beta = self._beta(beta)
        before = np.linalg.norm(self.native_verifier_residual, axis=1)
        after = np.linalg.norm(
            self.native_verifier_residual + beta * self.candidate_direction,
            axis=1,
        )
        return float(np.mean(before - after))

    def utility_lower(self, beta: float) -> float:
        """Lower bound on true mean EPE gain on the exact support."""
        beta = self._beta(beta)
        if beta == 0.0:
            return 0.0
        return float(
            self.nominal_verifier_gain(beta) - 2.0 * self.mean_epe_radius_upper
        )

    def signed_harm_upper(self, beta: float) -> float:
        """Upper bound on candidate-minus-native mean EPE (may be negative)."""
        beta = self._beta(beta)
        if beta == 0.0:
            return 0.0
        return float(
            -self.nominal_verifier_gain(beta) + 2.0 * self.mean_epe_radius_upper
        )

    def harm_budget_upper(self, beta: float) -> float:
        """Nonnegative version suitable for a harm-budget constraint."""
        return max(0.0, self.signed_harm_upper(beta))


def independent_verifier_epe_path_bound(
    native_output: np.ndarray,
    candidate_output: np.ndarray,
    verifier_output: np.ndarray,
    support: np.ndarray,
    *,
    action_key: str,
    control_key: str,
    control_selection_receipt_hash: str,
    source_input_hash: str,
    verifier_input_hash: str,
    task_support_hash: str,
    task_support_policy_hash: str,
    calibration: EPEVerifierCalibration,
    minimum_effective_n: float = 20.0,
    maximum_output_trust: float = 1.0,
) -> IndependentVerifierEPEPathBound:
    """Build an EPE-aligned bound without reading task truth or corruption labels."""
    required = (
        action_key, control_key, control_selection_receipt_hash,
        source_input_hash, verifier_input_hash, task_support_hash,
        task_support_policy_hash,
    )
    if any(not value for value in required):
        raise ValueError("exact control and support provenance are required")
    if source_input_hash != verifier_input_hash:
        raise ValueError("verifier provenance is not proposal-independent")
    if not calibration.action_blind:
        raise ValueError("verifier calibration is not action-blind")
    if not calibration.selection_aware:
        raise ValueError("verifier calibration is not selection-aware")
    if not calibration.frozen_before_selection:
        raise ValueError("verifier calibration was not frozen before selection")
    if calibration.effective_n < _nonnegative(
        minimum_effective_n, "minimum effective n",
    ):
        raise ValueError("verifier calibration effective n is insufficient")
    if task_support_policy_hash != calibration.support_policy_hash:
        raise ValueError("runtime and calibration support policies differ")

    native_input = np.asarray(native_output)
    candidate_input = np.asarray(candidate_output)
    verifier_input = np.asarray(verifier_output)
    native = np.asarray(native_output, dtype=np.float64)
    candidate = np.asarray(candidate_output, dtype=np.float64)
    verifier = np.asarray(verifier_output, dtype=np.float64)
    mask = np.asarray(support)
    if (native.shape != candidate.shape or native.shape != verifier.shape
            or native.ndim < 2 or native.shape[-1] < 1):
        raise ValueError("native, candidate, and verifier outputs must align as ...xD")
    if mask.dtype != np.bool_ or mask.shape != native.shape[:-1]:
        raise ValueError("support must be a boolean map over all non-vector dimensions")
    if (not np.isfinite(native).all() or not np.isfinite(candidate).all()
            or not np.isfinite(verifier).all()):
        raise ValueError("task outputs must be finite")
    support_count = int(mask.sum())
    if support_count < 1:
        raise ValueError("task support is empty")
    support_content_hash = support_mask_sha256(mask)
    if task_support_hash != support_content_hash:
        raise ValueError("task support content hash changed")

    residual = native[mask] - verifier[mask]
    direction = candidate[mask] - native[mask]
    return IndependentVerifierEPEPathBound(
        action_key=action_key,
        control_key=control_key,
        control_selection_receipt_hash=control_selection_receipt_hash,
        source_input_hash=source_input_hash,
        verifier_input_hash=verifier_input_hash,
        task_support_hash=task_support_hash,
        task_support_content_hash=support_content_hash,
        task_support_policy_hash=task_support_policy_hash,
        native_output_hash=_array_hash(native_input),
        candidate_output_hash=_array_hash(candidate_input),
        verifier_output_hash=_array_hash(verifier_input),
        verifier_id=calibration.verifier_id,
        calibration_version=calibration.calibration_version,
        calibration_data_hash=calibration.calibration_data_hash,
        support_count=support_count,
        mean_epe_radius_upper=calibration.mean_epe_radius_upper,
        maximum_output_trust=maximum_output_trust,
        native_verifier_residual=residual,
        candidate_direction=direction,
    )
