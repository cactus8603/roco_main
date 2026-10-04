"""Action-specific task bounds from an independent verifier.

The physical certificate answers whether an intervention is plausible.  This
module answers a different question: whether the *executed output direction*
is guaranteed to move toward a proposal-independent task verifier.

Let ``n`` be the native output, ``c = n + d`` the candidate output, ``v`` an
independent verifier, and assume the unknown task target ``y`` satisfies

    sqrt(mean(||v - y||^2)) <= r.

For the delivered path ``n + alpha d``, Cauchy--Schwarz gives the exact robust
lower bound

    MSE(n, y) - MSE(n + alpha d, y)
      >= alpha * [-2 <n-v,d> - 2 r ||d||] - alpha^2 ||d||^2.

Unlike a candidate-response score, the verifier is not allowed to consume the
candidate.  The radius must be frozen and selection-aware; otherwise the
bound is rejected rather than silently treated as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .selector_v4 import DirectionalParameterBound


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


def support_mask_sha256(support: np.ndarray) -> str:
    """Canonical content hash used by the verifier and Selector-v4 footprint."""
    mask = np.asarray(support, dtype=bool)
    return _array_hash(mask)


@dataclass(frozen=True)
class VerifierCalibration:
    """Frozen high-probability error radius for a verifier/support policy."""

    verifier_id: str
    representation_id: str
    calibration_version: str
    calibration_data_hash: str
    support_policy_hash: str
    error_radius_upper: float
    effective_n: float
    confidence_level: float
    selection_aware: bool
    action_blind: bool
    frozen_before_selection: bool
    metric: str = "mean_squared_l2"

    def __post_init__(self) -> None:
        values = (
            self.verifier_id, self.representation_id, self.calibration_version,
            self.calibration_data_hash, self.support_policy_hash,
        )
        if any(not value for value in values):
            raise ValueError("verifier calibration identity is required")
        _nonnegative(self.error_radius_upper, "verifier error radius")
        if _nonnegative(self.effective_n, "verifier effective n") <= 0.0:
            raise ValueError("verifier effective n must be positive")
        confidence = _finite(self.confidence_level, "confidence level")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence level must lie in (0,1)")
        if self.metric != "mean_squared_l2":
            raise ValueError("only the proved mean-squared L2 bound is supported")


@dataclass(frozen=True)
class IndependentVerifierTaskBound:
    """Auditable robust bound for one actual candidate output direction."""

    action_key: str
    parameter_key: str
    source_input_hash: str
    verifier_input_hash: str
    task_support_hash: str
    task_support_content_hash: str
    native_output_hash: str
    candidate_output_hash: str
    verifier_output_hash: str
    verifier_id: str
    calibration_version: str
    calibration_data_hash: str
    support_policy_hash: str
    support_count: int
    nominal_directional_gain: float
    verifier_radius_penalty: float
    task_directional_gain_lower: float
    action_norm_upper: float
    task_curvature_upper: float
    maximum_strength: float

    def __post_init__(self) -> None:
        identities = (
            self.action_key, self.parameter_key, self.source_input_hash,
            self.verifier_input_hash, self.task_support_hash,
            self.task_support_content_hash, self.native_output_hash,
            self.candidate_output_hash, self.verifier_output_hash,
            self.verifier_id, self.calibration_version,
            self.calibration_data_hash, self.support_policy_hash,
        )
        if any(not value for value in identities):
            raise ValueError("task-bound identity and provenance are required")
        if self.source_input_hash != self.verifier_input_hash:
            raise ValueError("verifier must be computed from the native source")
        if self.support_count < 1:
            raise ValueError("task support must be nonempty")
        for value, name in (
            (self.nominal_directional_gain, "nominal directional gain"),
            (self.task_directional_gain_lower, "task directional gain"),
        ):
            _finite(value, name)
        for value, name in (
            (self.verifier_radius_penalty, "verifier radius penalty"),
            (self.action_norm_upper, "action norm"),
            (self.task_curvature_upper, "task curvature"),
        ):
            _nonnegative(value, name)
        strength = _finite(self.maximum_strength, "maximum strength")
        if not 0.0 < strength <= 1.0:
            raise ValueError("maximum strength must lie in (0,1]")

    def utility_lower(self, alpha: float) -> float:
        alpha = _finite(alpha, "trust alpha")
        if not 0.0 <= alpha <= self.maximum_strength:
            raise ValueError("trust alpha lies outside the certified path")
        penalty = 0.5 * self.task_curvature_upper * self.action_norm_upper ** 2
        return float(alpha * self.task_directional_gain_lower - alpha * alpha * penalty)

    def to_directional_parameter_bound(
        self,
        *,
        physical_support_hash: str,
        physical_directional_gain_lower: float,
        physical_curvature_upper: float,
        path_remainder_quadratic_upper: float = 0.0,
    ) -> DirectionalParameterBound:
        """Attach an independent task proof to a separately proved physical path."""
        return DirectionalParameterBound(
            key=self.parameter_key,
            physical_support_hash=physical_support_hash,
            task_support_hash=self.task_support_hash,
            physical_directional_gain_lower=physical_directional_gain_lower,
            physical_curvature_upper=physical_curvature_upper,
            task_directional_gain_lower=self.task_directional_gain_lower,
            task_curvature_upper=self.task_curvature_upper,
            action_norm_upper=self.action_norm_upper,
            path_remainder_quadratic_upper=path_remainder_quadratic_upper,
            maximum_strength=self.maximum_strength,
        )


def independent_verifier_squared_l2_bound(
    native_output: np.ndarray,
    candidate_output: np.ndarray,
    verifier_output: np.ndarray,
    support: np.ndarray,
    *,
    action_key: str,
    parameter_key: str,
    source_input_hash: str,
    verifier_input_hash: str,
    task_support_hash: str,
    calibration: VerifierCalibration,
    minimum_effective_n: float = 20.0,
    maximum_strength: float = 1.0,
) -> IndependentVerifierTaskBound:
    """Prove a candidate-specific MSE-gain parabola on a fixed output support.

    The final output path must be a convex blend of the two supplied task
    outputs.  No task truth or corruption label is accepted by this function.
    """
    if not action_key or not parameter_key or not task_support_hash:
        raise ValueError("action, parameter, and task support identities are required")
    if not source_input_hash or source_input_hash != verifier_input_hash:
        raise ValueError("verifier provenance is not proposal-independent")
    if not calibration.action_blind:
        raise ValueError("verifier calibration is not action-blind")
    if not calibration.selection_aware:
        raise ValueError("verifier radius is not selection-aware")
    if not calibration.frozen_before_selection:
        raise ValueError("verifier radius was not frozen before selection")
    if calibration.effective_n < _nonnegative(minimum_effective_n, "minimum effective n"):
        raise ValueError("verifier calibration effective n is insufficient")
    maximum_strength = _finite(maximum_strength, "maximum strength")
    if not 0.0 < maximum_strength <= 1.0:
        raise ValueError("maximum strength must lie in (0,1]")

    native = np.asarray(native_output, dtype=np.float64)
    candidate = np.asarray(candidate_output, dtype=np.float64)
    verifier = np.asarray(verifier_output, dtype=np.float64)
    mask = np.asarray(support, dtype=bool)
    if native.shape != candidate.shape or native.shape != verifier.shape:
        raise ValueError("native, candidate, and verifier outputs must align")
    if native.ndim < 2 or mask.shape != native.shape[:-1]:
        raise ValueError("support must index all output dimensions except channels")
    if not np.all(np.isfinite(native)) or not np.all(np.isfinite(candidate)):
        raise ValueError("task outputs must be finite")
    if not np.all(np.isfinite(verifier)):
        raise ValueError("verifier output must be finite")
    support_count = int(mask.sum())
    if support_count < 1:
        raise ValueError("task support is empty")
    support_content_hash = support_mask_sha256(mask)
    if task_support_hash != support_content_hash:
        raise ValueError("task support content hash changed")

    native_selected = native[mask]
    delta = candidate[mask] - native_selected
    verifier_residual = native_selected - verifier[mask]
    squared_delta = np.sum(delta * delta, axis=-1)
    action_norm = float(np.sqrt(np.mean(squared_delta)))
    if action_norm <= 0.0:
        raise ValueError("candidate has no effect on task support")
    nominal = float(np.mean(-2.0 * np.sum(verifier_residual * delta, axis=-1)))
    radius_penalty = float(2.0 * calibration.error_radius_upper * action_norm)
    directional_lower = float(nominal - radius_penalty)

    return IndependentVerifierTaskBound(
        action_key=action_key,
        parameter_key=parameter_key,
        source_input_hash=source_input_hash,
        verifier_input_hash=verifier_input_hash,
        task_support_hash=task_support_hash,
        task_support_content_hash=support_content_hash,
        native_output_hash=_array_hash(native_output),
        candidate_output_hash=_array_hash(candidate_output),
        verifier_output_hash=_array_hash(verifier_output),
        verifier_id=calibration.verifier_id,
        calibration_version=calibration.calibration_version,
        calibration_data_hash=calibration.calibration_data_hash,
        support_policy_hash=calibration.support_policy_hash,
        support_count=support_count,
        nominal_directional_gain=nominal,
        verifier_radius_penalty=radius_penalty,
        task_directional_gain_lower=directional_lower,
        action_norm_upper=action_norm,
        task_curvature_upper=2.0,
        maximum_strength=maximum_strength,
    )
