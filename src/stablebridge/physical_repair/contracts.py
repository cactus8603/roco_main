"""Typed, fail-closed contracts for regional repair.

Every mask carries a named coordinate frame.  Ground-truth values and
corruption labels deliberately have no place in the runtime records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


ENDPOINTS = frozenset({"first", "second", "both", "unknown", "output"})
DOMAINS = frozenset({"image", "feature", "output"})
BRANCHES = frozenset({"CC", "RC", "CR", "RR", "not_applicable"})
SEMANTICS = frozenset({"from_zero", "same_state", "output_only"})
CERTIFICATE_STATES = frozenset({"supported", "ambiguous", "rejected", "unsupported"})
COORDINATE_FRAMES = frozenset({"first_native", "second_native", "flow_native"})


def _finite_nonnegative(value: float, name: str) -> float:
    item = float(value)
    if not np.isfinite(item) or item < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return item


def _mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    item = np.asarray(value, dtype=np.float32)
    if item.shape != shape or not np.isfinite(item).all():
        raise ValueError(f"{name} must be a finite HxW map")
    if np.any(item < 0.0) or np.any(item > 1.0):
        raise ValueError(f"{name} must lie in [0,1]")
    return np.ascontiguousarray(item)


@dataclass(frozen=True)
class ActionSpec:
    operator_id: str
    operator_version: str
    domain: str
    hypothesized_degraded_endpoint: str
    modified_endpoint: str
    coordinate_frame: str
    branch_pair: str = "not_applicable"
    execution_semantics: str = "from_zero"
    intervention_step: int | None = None
    source_state_hash: str | None = None
    expected_compute_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.operator_id or not self.operator_version:
            raise ValueError("operator identity must be nonempty")
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {sorted(DOMAINS)}")
        if self.hypothesized_degraded_endpoint not in ENDPOINTS - {"output"}:
            raise ValueError("invalid hypothesized degraded endpoint")
        if self.modified_endpoint not in ENDPOINTS - {"unknown"}:
            raise ValueError("invalid modified endpoint")
        if self.coordinate_frame not in COORDINATE_FRAMES:
            raise ValueError("invalid action coordinate frame")
        if self.branch_pair not in BRANCHES:
            raise ValueError("invalid C/R branch pair")
        if self.execution_semantics not in SEMANTICS:
            raise ValueError("invalid execution semantics")
        if self.execution_semantics == "same_state":
            if self.intervention_step is None or self.intervention_step < 0:
                raise ValueError("same-state action requires an intervention step")
            if not self.source_state_hash:
                raise ValueError("same-state action requires a source state hash")
        _finite_nonnegative(self.expected_compute_seconds, "expected compute")


@dataclass(frozen=True)
class PhysicalCertificate:
    action: ActionSpec
    status: str
    observation_support_fraction: float
    identifiable_support_fraction: float
    null_score: float
    action_score: float
    spatial_holdout_gain: float
    parameter_uncertainty: float
    fit_stability: float
    competing_model_scores: Mapping[str, float] = field(default_factory=dict)
    estimated_parameters: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()
    calibration_version: str = "uncalibrated"
    measured_probe_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.status not in CERTIFICATE_STATES:
            raise ValueError("invalid certificate status")
        for name in ("observation_support_fraction", "identifiable_support_fraction"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        for name in ("null_score", "action_score", "parameter_uncertainty",
                     "measured_probe_seconds"):
            _finite_nonnegative(getattr(self, name), name)
        if not np.isfinite(float(self.spatial_holdout_gain)):
            raise ValueError("spatial holdout gain must be finite")
        if not np.isfinite(float(self.fit_stability)) or not 0.0 <= self.fit_stability <= 1.0:
            raise ValueError("fit stability must lie in [0,1]")
        values = (*self.competing_model_scores.values(),
                  *self.estimated_parameters.values(), *self.diagnostics.values())
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("certificate diagnostics must be finite")
        if self.status == "supported" and self.rejection_reasons:
            raise ValueError("supported certificate cannot have rejection reasons")
        if self.status != "supported" and not self.rejection_reasons:
            raise ValueError("non-supported certificate needs a reason")


@dataclass(frozen=True)
class SupportMaps:
    coordinate_frame: str
    native_error_risk: np.ndarray
    physical_applicability: np.ndarray
    evidence_support: np.ndarray
    predicted_signed_utility: np.ndarray

    def __post_init__(self) -> None:
        if self.coordinate_frame not in COORDINATE_FRAMES:
            raise ValueError("invalid support coordinate frame")
        shape = np.asarray(self.native_error_risk).shape
        if len(shape) != 2:
            raise ValueError("support maps must be HxW")
        for name in ("native_error_risk", "physical_applicability", "evidence_support"):
            object.__setattr__(self, name, _mask(getattr(self, name), shape, name))
        utility = np.asarray(self.predicted_signed_utility, dtype=np.float32)
        if utility.shape != shape or not np.isfinite(utility).all():
            raise ValueError("predicted utility must be a finite HxW map")
        object.__setattr__(self, "predicted_signed_utility", np.ascontiguousarray(utility))

    @property
    def executable_support(self) -> np.ndarray:
        positive = (self.predicted_signed_utility > 0.0).astype(np.float32)
        return np.ascontiguousarray(
            self.physical_applicability * self.evidence_support * positive,
            dtype=np.float32,
        )


@dataclass(frozen=True)
class CostLedger:
    native_seconds: float
    evidence_seconds: float = 0.0
    physical_fit_seconds: float = 0.0
    action_seconds: float = 0.0
    extra_matcher_seconds: float = 0.0
    postcheck_seconds: float = 0.0
    compose_seconds: float = 0.0
    transfer_seconds: float = 0.0
    extra_matcher_trajectories: int = 0

    def __post_init__(self) -> None:
        for name in ("native_seconds", "evidence_seconds", "physical_fit_seconds",
                     "action_seconds", "extra_matcher_seconds", "postcheck_seconds",
                     "compose_seconds", "transfer_seconds"):
            _finite_nonnegative(getattr(self, name), name)
        if self.extra_matcher_trajectories < 0:
            raise ValueError("extra trajectory count must be nonnegative")

    @property
    def total_seconds(self) -> float:
        return float(sum((self.native_seconds, self.evidence_seconds,
                          self.physical_fit_seconds, self.action_seconds,
                          self.extra_matcher_seconds, self.postcheck_seconds,
                          self.compose_seconds, self.transfer_seconds)))


@dataclass(frozen=True)
class BoundedOutput:
    flow: np.ndarray
    update_norm: np.ndarray
    exact_fallback: np.ndarray
    alpha: float
    epsilon: float
    max_update_norm: float

    def __post_init__(self) -> None:
        flow = np.asarray(self.flow, dtype=np.float32)
        norm = np.asarray(self.update_norm, dtype=np.float32)
        fallback = np.asarray(self.exact_fallback, dtype=bool)
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise ValueError("bounded flow must be HxWx2")
        if norm.shape != flow.shape[:2] or fallback.shape != norm.shape:
            raise ValueError("bounded-output maps do not match flow")
        if not np.isfinite(flow).all() or not np.isfinite(norm).all():
            raise ValueError("bounded output must be finite")
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError("alpha must lie in [0,1]")
        _finite_nonnegative(self.epsilon, "epsilon")
        _finite_nonnegative(self.max_update_norm, "max update norm")
        object.__setattr__(self, "flow", np.ascontiguousarray(flow))
        object.__setattr__(self, "update_norm", np.ascontiguousarray(norm))
        object.__setattr__(self, "exact_fallback", np.ascontiguousarray(fallback))
