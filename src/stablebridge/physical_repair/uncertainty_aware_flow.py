"""Provider-neutral uncertainty training and bounded flow refinement primitives.

The production matcher is deliberately not changed by importing this module.
It provides the pieces needed to train an uncertainty observer first, then a
bounded flow refiner while keeping the two gradient paths decoupled.  The
runtime policy defaults to ``observer_only``; changing flow is an explicit,
hashed opt-in and therefore requires new outcomes and calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from typing import Sequence

import numpy as np

try:  # Contract and NumPy tests remain usable in dependency-light readers.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


UNCERTAINTY_AWARE_FLOW_SCHEMA_V1 = "stablebridge-uncertainty-aware-flow/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty canonical text")
    return value


def _digest(value: object, name: str) -> str:
    result = _text(value, name)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class UncertaintyAvailabilityV1(str, Enum):
    AVAILABLE = "available"
    TYPED_MISSING = "typed_missing"
    OOD = "ood"


class UncertaintyMapSemanticsV1(str, Enum):
    LOG_LAPLACE_SCALE = "log_laplace_scale"
    LAPLACE_SCALE = "laplace_scale"
    LOG_VARIANCE = "log_variance"
    VARIANCE = "variance"


class UncertaintyClaimSemanticsV1(str, Enum):
    SELF_CONSISTENCY_SURROGATE = "self_consistency_surrogate"
    CALIBRATED_ENDPOINT_ERROR = "calibrated_endpoint_error"


class UncertaintyFlowModeV1(str, Enum):
    OBSERVER_ONLY = "observer_only"
    BOUNDED_REFINEMENT = "bounded_refinement"
    BIDIRECTIONAL_FUSION = "bidirectional_fusion"


class UncertaintyEvidenceRoleV1(str, Enum):
    """Which flow the uncertainty map describes.

    A base-flow uncertainty map must not be mistaken for confidence in the
    proposed correction.  The two imply opposite blend directions.
    """

    BASE_FLOW = "base_flow"
    PROPOSED_FLOW = "proposed_flow"
    CONDITION_ONLY = "condition_only"


class UncertaintyTrainingStageV1(str, Enum):
    UNCERTAINTY_HEAD_ONLY = "uncertainty_head_only"
    FROZEN_UNCERTAINTY_REFINER = "frozen_uncertainty_refiner"
    ALTERNATING_DECOUPLED = "alternating_decoupled"


@dataclass(frozen=True)
class UncertaintySummaryV1:
    valid_count: int
    mean: float
    p50: float
    p90: float
    p99: float
    maximum: float
    high_uncertainty_fraction: float
    threshold: float

    def __post_init__(self) -> None:
        if isinstance(self.valid_count, bool) or self.valid_count <= 0:
            raise ValueError("uncertainty summary needs positive valid count")
        values = tuple(
            _finite(value, name) for name, value in (
                ("mean", self.mean), ("p50", self.p50), ("p90", self.p90),
                ("p99", self.p99), ("maximum", self.maximum),
                ("high uncertainty fraction", self.high_uncertainty_fraction),
                ("threshold", self.threshold),
            )
        )
        if not 0.0 <= values[5] <= 1.0:
            raise ValueError("high uncertainty fraction must lie in [0,1]")
        if not values[1] <= values[2] <= values[3] <= values[4]:
            raise ValueError("uncertainty quantiles are not ordered")


def summarize_uncertainty_v1(
    uncertainty: np.ndarray,
    valid_mask: np.ndarray,
    *,
    high_uncertainty_threshold: float,
) -> UncertaintySummaryV1:
    values = np.asarray(uncertainty, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    threshold = _finite(high_uncertainty_threshold, "high uncertainty threshold")
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("uncertainty and valid mask must be matching HxW arrays")
    valid = valid & np.isfinite(values)
    selected = values[valid]
    if selected.size == 0:
        raise ValueError("uncertainty summary has empty finite support")
    quantiles = np.quantile(selected, [0.5, 0.9, 0.99], method="linear")
    return UncertaintySummaryV1(
        valid_count=int(selected.size),
        mean=float(selected.mean()),
        p50=float(quantiles[0]),
        p90=float(quantiles[1]),
        p99=float(quantiles[2]),
        maximum=float(selected.max()),
        high_uncertainty_fraction=float(np.mean(selected >= threshold)),
        threshold=threshold,
    )


@dataclass(frozen=True)
class UncertaintyReceiptV1:
    """Sealed provenance for one native or executed-child uncertainty map."""

    task: str
    observation_role: str
    provider_id: str
    provider_checkpoint_hash: str
    input_hashes: tuple[str, ...]
    flow_hash: str
    map_hash: str | None
    map_shape: tuple[int, int] | None
    semantics: UncertaintyMapSemanticsV1
    iteration: int | None
    availability: UncertaintyAvailabilityV1
    availability_reason: str
    summary: UncertaintySummaryV1 | None
    spatial_embedding_hash: str | None
    claim_semantics: UncertaintyClaimSemanticsV1 = (
        UncertaintyClaimSemanticsV1.SELF_CONSISTENCY_SURROGATE
    )
    calibration_receipt_hash: str | None = None
    schema: str = UNCERTAINTY_AWARE_FLOW_SCHEMA_V1
    receipt_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if self.schema != UNCERTAINTY_AWARE_FLOW_SCHEMA_V1:
            raise ValueError("uncertainty receipt schema drift")
        if self.task not in {"flow", "stereo"}:
            raise ValueError("uncertainty task must be flow or stereo")
        if self.observation_role not in {"native", "executed_child"}:
            raise ValueError("uncertainty role must be native or executed_child")
        _text(self.provider_id, "uncertainty provider id")
        _digest(self.provider_checkpoint_hash, "provider checkpoint hash")
        inputs = tuple(self.input_hashes)
        if not inputs:
            raise ValueError("uncertainty receipt needs input hashes")
        for value in inputs:
            _digest(value, "uncertainty input hash")
        _digest(self.flow_hash, "uncertainty flow hash")
        if not isinstance(self.semantics, UncertaintyMapSemanticsV1):
            raise ValueError("uncertainty semantics must be typed")
        if not isinstance(self.claim_semantics, UncertaintyClaimSemanticsV1):
            raise ValueError("uncertainty claim semantics must be typed")
        if self.claim_semantics is UncertaintyClaimSemanticsV1.CALIBRATED_ENDPOINT_ERROR:
            if self.calibration_receipt_hash is None:
                raise ValueError("calibrated uncertainty needs a calibration receipt")
            _digest(self.calibration_receipt_hash, "uncertainty calibration receipt hash")
        elif self.calibration_receipt_hash is not None:
            raise ValueError("surrogate uncertainty cannot claim a calibration receipt")
        if not isinstance(self.availability, UncertaintyAvailabilityV1):
            raise ValueError("uncertainty availability must be typed")
        _text(self.availability_reason, "uncertainty availability reason")
        if self.iteration is not None and (
            isinstance(self.iteration, bool) or self.iteration <= 0
        ):
            raise ValueError("uncertainty iteration must be positive or absent")
        if self.availability is UncertaintyAvailabilityV1.AVAILABLE:
            if self.map_hash is None or self.map_shape is None or self.summary is None:
                raise ValueError("available uncertainty receipt is incomplete")
            _digest(self.map_hash, "uncertainty map hash")
            if (
                len(self.map_shape) != 2
                or any(isinstance(v, bool) or v <= 0 for v in self.map_shape)
            ):
                raise ValueError("uncertainty map shape must be positive HxW")
            if self.spatial_embedding_hash is not None:
                _digest(self.spatial_embedding_hash, "spatial embedding hash")
        elif any(
            value is not None for value in (
                self.map_hash, self.map_shape, self.summary,
                self.spatial_embedding_hash,
            )
        ):
            raise ValueError("unavailable uncertainty receipt cannot carry values")
        payload = {
            "schema": self.schema,
            "task": self.task,
            "observation_role": self.observation_role,
            "provider_id": self.provider_id,
            "provider_checkpoint_hash": self.provider_checkpoint_hash,
            "input_hashes": list(inputs),
            "flow_hash": self.flow_hash,
            "map_hash": self.map_hash,
            "map_shape": list(self.map_shape) if self.map_shape is not None else None,
            "semantics": self.semantics.value,
            "iteration": self.iteration,
            "availability": self.availability.value,
            "availability_reason": self.availability_reason,
            "summary": None if self.summary is None else {
                name: getattr(self.summary, name)
                for name in self.summary.__dataclass_fields__
            },
            "spatial_embedding_hash": self.spatial_embedding_hash,
            "claim_semantics": self.claim_semantics.value,
            "calibration_receipt_hash": self.calibration_receipt_hash,
        }
        expected = _hash(payload)
        if self.receipt_hash and self.receipt_hash != expected:
            raise ValueError("uncertainty receipt hash drift")
        object.__setattr__(self, "input_hashes", inputs)
        object.__setattr__(self, "receipt_hash", expected)


@dataclass(frozen=True)
class UncertaintyRuntimePolicyV1:
    """Runtime authority.  The safe default observes but never changes flow."""

    mode: UncertaintyFlowModeV1 = UncertaintyFlowModeV1.OBSERVER_ONLY
    evidence_role: UncertaintyEvidenceRoleV1 = UncertaintyEvidenceRoleV1.BASE_FLOW
    flow_update_enabled: bool = False
    maximum_update_px: float = 0.0
    minimum_update_fraction: float = 0.0
    stop_gradient_through_uncertainty: bool = True
    require_calibrated_provider: bool = True
    require_calibrated_flow_change: bool = True
    policy_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, UncertaintyFlowModeV1):
            raise ValueError("uncertainty flow mode must be typed")
        if not isinstance(self.evidence_role, UncertaintyEvidenceRoleV1):
            raise ValueError("uncertainty evidence role must be typed")
        maximum = _finite(self.maximum_update_px, "maximum uncertainty flow update")
        minimum_fraction = _finite(
            self.minimum_update_fraction, "minimum uncertainty update fraction"
        )
        if maximum < 0.0:
            raise ValueError("maximum uncertainty flow update must be nonnegative")
        if not 0.0 <= minimum_fraction <= 1.0:
            raise ValueError("minimum uncertainty update fraction must lie in [0,1]")
        if self.mode is UncertaintyFlowModeV1.OBSERVER_ONLY:
            if self.flow_update_enabled or maximum != 0.0 or minimum_fraction != 0.0:
                raise ValueError("observer-only uncertainty cannot modify flow")
        elif not self.flow_update_enabled or maximum <= 0.0:
            raise ValueError("flow-changing uncertainty needs an explicit positive bound")
        if self.stop_gradient_through_uncertainty is not True:
            raise ValueError("v1 requires detached uncertainty in the flow refiner")
        expected = _hash({
            "mode": self.mode.value,
            "evidence_role": self.evidence_role.value,
            "flow_update_enabled": self.flow_update_enabled,
            "maximum_update_px": maximum,
            "minimum_update_fraction": minimum_fraction,
            "stop_gradient_through_uncertainty": self.stop_gradient_through_uncertainty,
            "require_calibrated_provider": self.require_calibrated_provider,
            "require_calibrated_flow_change": self.require_calibrated_flow_change,
        })
        if self.policy_hash and self.policy_hash != expected:
            raise ValueError("uncertainty runtime policy hash drift")
        object.__setattr__(self, "maximum_update_px", maximum)
        object.__setattr__(self, "minimum_update_fraction", minimum_fraction)
        object.__setattr__(self, "policy_hash", expected)


@dataclass(frozen=True)
class FlowChangeAuthorizationReceiptV1:
    """Independent authority for a proposal provider to modify flow.

    Calibrating U0 does not calibrate a U1 refiner.  This receipt keeps those
    claims separate and binds the allowed update geometry to one frozen
    proposal-provider checkpoint and uncertainty provider.
    """

    proposal_provider_id: str
    proposal_provider_checkpoint_hash: str
    uncertainty_provider_checkpoint_hash: str
    uncertainty_calibration_receipt_hash: str
    independent_validation_split_hash: str
    flow_change_calibration_receipt_hash: str
    authorized_mode: UncertaintyFlowModeV1
    evidence_role: UncertaintyEvidenceRoleV1
    calibrated_maximum_update_px: float
    qualified: bool
    receipt_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        _text(self.proposal_provider_id, "proposal provider id")
        for name in (
            "proposal_provider_checkpoint_hash",
            "uncertainty_provider_checkpoint_hash",
            "uncertainty_calibration_receipt_hash",
            "independent_validation_split_hash",
            "flow_change_calibration_receipt_hash",
        ):
            _digest(getattr(self, name), name.replace("_", " "))
        if self.authorized_mode is UncertaintyFlowModeV1.OBSERVER_ONLY:
            raise ValueError("flow-change authorization cannot target observer-only mode")
        if not isinstance(self.authorized_mode, UncertaintyFlowModeV1):
            raise ValueError("authorized flow-change mode must be typed")
        if not isinstance(self.evidence_role, UncertaintyEvidenceRoleV1):
            raise ValueError("authorized uncertainty role must be typed")
        maximum = _finite(
            self.calibrated_maximum_update_px, "calibrated maximum flow update"
        )
        if maximum <= 0.0:
            raise ValueError("calibrated maximum flow update must be positive")
        if not isinstance(self.qualified, bool):
            raise ValueError("flow-change qualification must be boolean")
        expected = _hash({
            "proposal_provider_id": self.proposal_provider_id,
            "proposal_provider_checkpoint_hash": self.proposal_provider_checkpoint_hash,
            "uncertainty_provider_checkpoint_hash": self.uncertainty_provider_checkpoint_hash,
            "uncertainty_calibration_receipt_hash": self.uncertainty_calibration_receipt_hash,
            "independent_validation_split_hash": self.independent_validation_split_hash,
            "flow_change_calibration_receipt_hash": self.flow_change_calibration_receipt_hash,
            "authorized_mode": self.authorized_mode.value,
            "evidence_role": self.evidence_role.value,
            "calibrated_maximum_update_px": maximum,
            "qualified": self.qualified,
        })
        if self.receipt_hash and self.receipt_hash != expected:
            raise ValueError("flow-change authorization receipt hash drift")
        object.__setattr__(self, "calibrated_maximum_update_px", maximum)
        object.__setattr__(self, "receipt_hash", expected)


@dataclass(frozen=True)
class DecoupledTrainingPolicyV1:
    """Three-stage schedule adapted from U2Flow's decoupling result."""

    stage: UncertaintyTrainingStageV1
    detach_consistency_teacher: bool = True
    detach_uncertainty_in_refiner: bool = True
    base_matcher_trainable: bool = False
    uncertainty_head_trainable: bool = True
    refiner_trainable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stage, UncertaintyTrainingStageV1):
            raise ValueError("uncertainty training stage must be typed")
        if not self.detach_consistency_teacher or not self.detach_uncertainty_in_refiner:
            raise ValueError("v1 forbids coupled uncertainty/flow gradients")
        expected = {
            UncertaintyTrainingStageV1.UNCERTAINTY_HEAD_ONLY: (False, True, False),
            UncertaintyTrainingStageV1.FROZEN_UNCERTAINTY_REFINER: (False, False, True),
            UncertaintyTrainingStageV1.ALTERNATING_DECOUPLED: (True, True, True),
        }[self.stage]
        observed = (
            self.base_matcher_trainable,
            self.uncertainty_head_trainable,
            self.refiner_trainable,
        )
        if observed != expected:
            raise ValueError("trainable modules do not match the decoupled stage")


def _log_variance_numpy(
    uncertainty: np.ndarray,
    semantics: UncertaintyMapSemanticsV1,
) -> np.ndarray:
    values = np.asarray(uncertainty, dtype=np.float64)
    if semantics is UncertaintyMapSemanticsV1.LOG_VARIANCE:
        result = values
    elif semantics is UncertaintyMapSemanticsV1.VARIANCE:
        result = np.log(np.maximum(values, 1e-12))
    elif semantics is UncertaintyMapSemanticsV1.LOG_LAPLACE_SCALE:
        result = 2.0 * values
    elif semantics is UncertaintyMapSemanticsV1.LAPLACE_SCALE:
        result = 2.0 * np.log(np.maximum(values, 1e-12))
    else:  # pragma: no cover - enum exhaustiveness
        raise ValueError("unsupported uncertainty semantics")
    if not np.isfinite(result).all():
        raise ValueError("uncertainty contains nonfinite values")
    return result


def uncertainty_reliability_v1(
    uncertainty: np.ndarray,
    semantics: UncertaintyMapSemanticsV1,
) -> np.ndarray:
    """Return sigmoid(-log variance), matching U2Flow's reliability role."""

    log_variance = np.clip(_log_variance_numpy(uncertainty, semantics), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(log_variance))).astype(np.float32)


def uncertainty_update_gate_v1(
    reliability: np.ndarray,
    role: UncertaintyEvidenceRoleV1,
    *,
    minimum_fraction: float = 0.0,
) -> np.ndarray:
    """Convert reliability to an update budget without reversing its meaning."""

    value = np.asarray(reliability, dtype=np.float32)
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("reliability must be finite and lie in [0,1]")
    floor = _finite(minimum_fraction, "minimum uncertainty update fraction")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("minimum uncertainty update fraction must lie in [0,1]")
    if role is UncertaintyEvidenceRoleV1.BASE_FLOW:
        raw = 1.0 - value
    elif role is UncertaintyEvidenceRoleV1.PROPOSED_FLOW:
        raw = value
    elif role is UncertaintyEvidenceRoleV1.CONDITION_ONLY:
        raw = np.ones_like(value)
    else:  # pragma: no cover - enum exhaustiveness
        raise ValueError("unsupported uncertainty evidence role")
    return (floor + (1.0 - floor) * raw).astype(np.float32)


def authorize_uncertainty_for_policy_v1(
    receipt: UncertaintyReceiptV1 | None,
    policy: UncertaintyRuntimePolicyV1,
    flow_change_authorization: FlowChangeAuthorizationReceiptV1 | None = None,
) -> None:
    """Enforce separate U0 and flow-change calibration claims."""

    if policy.mode is UncertaintyFlowModeV1.OBSERVER_ONLY:
        return
    if receipt is None:
        if policy.require_calibrated_provider:
            raise PermissionError("flow-changing uncertainty needs a calibrated receipt")
        return
    if receipt.availability is not UncertaintyAvailabilityV1.AVAILABLE:
        raise PermissionError("unavailable uncertainty cannot modify flow")
    if policy.require_calibrated_provider and (
        receipt.claim_semantics
        is not UncertaintyClaimSemanticsV1.CALIBRATED_ENDPOINT_ERROR
        or receipt.calibration_receipt_hash is None
    ):
        raise PermissionError("self-consistency uncertainty is not calibrated task risk")
    if not policy.require_calibrated_flow_change:
        return
    if flow_change_authorization is None:
        raise PermissionError("flow change needs its own calibrated authorization")
    if not flow_change_authorization.qualified:
        raise PermissionError("flow-change provider did not pass independent qualification")
    if (
        flow_change_authorization.authorized_mode is not policy.mode
        or flow_change_authorization.evidence_role is not policy.evidence_role
        or policy.maximum_update_px
        > flow_change_authorization.calibrated_maximum_update_px + 1e-12
    ):
        raise PermissionError("flow-change authorization does not cover runtime policy")
    if receipt is None:
        raise PermissionError("flow-change authorization cannot bind missing uncertainty")
    if (
        flow_change_authorization.uncertainty_provider_checkpoint_hash
        != receipt.provider_checkpoint_hash
        or flow_change_authorization.uncertainty_calibration_receipt_hash
        != receipt.calibration_receipt_hash
    ):
        raise PermissionError("flow-change authorization binds different uncertainty")


def bounded_uncertainty_guided_flow_v1(
    base_flow: np.ndarray,
    proposed_flow: np.ndarray,
    uncertainty: np.ndarray,
    valid_mask: np.ndarray,
    *,
    semantics: UncertaintyMapSemanticsV1,
    policy: UncertaintyRuntimePolicyV1,
    receipt: UncertaintyReceiptV1 | None = None,
    flow_change_authorization: FlowChangeAuthorizationReceiptV1 | None = None,
) -> np.ndarray:
    """Reliability-weight a proposed residual and enforce a pointwise norm cap.

    This is a runtime primitive, not a trained or calibrated policy.  It refuses
    to run under the default observer-only policy.
    """

    if policy.mode is UncertaintyFlowModeV1.OBSERVER_ONLY:
        raise PermissionError("observer-only uncertainty is not authorized to change flow")
    authorize_uncertainty_for_policy_v1(
        receipt, policy, flow_change_authorization,
    )
    base = np.asarray(base_flow, dtype=np.float32)
    proposed = np.asarray(proposed_flow, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    u = np.asarray(uncertainty)
    if base.ndim != 3 or base.shape[2] != 2 or proposed.shape != base.shape:
        raise ValueError("base and proposed flow must be matching HxWx2 arrays")
    if u.shape != base.shape[:2] or valid.shape != base.shape[:2]:
        raise ValueError("uncertainty and validity must match the flow lattice")
    if not np.isfinite(base).all() or not np.isfinite(proposed).all():
        raise ValueError("flow inputs must be finite")
    reliability = uncertainty_reliability_v1(u, semantics)
    update_gate = uncertainty_update_gate_v1(
        reliability,
        policy.evidence_role,
        minimum_fraction=policy.minimum_update_fraction,
    )
    delta = proposed - base
    norm = np.linalg.norm(delta.astype(np.float64), axis=2)
    scale = np.minimum(1.0, policy.maximum_update_px / np.maximum(norm, 1e-12))
    weighted = delta * (update_gate * scale * valid.astype(np.float32))[..., None]
    result = np.where(valid[..., None], base + weighted, base)
    if not np.isfinite(result).all():
        raise FloatingPointError("uncertainty-guided flow produced nonfinite values")
    return result.astype(np.float32)


if nn is not None:

    class DecoupledUncertaintyHeadV1(nn.Module):
        """Small observer compatible with the existing 12-channel Work B bank."""

        def __init__(self, feature_channels: int = 12, hidden_channels: int = 32):
            super().__init__()
            self.feature_channels = int(feature_channels)
            self.network = nn.Sequential(
                nn.Conv2d(self.feature_channels, hidden_channels, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(hidden_channels, 1, 3, padding=1),
            )

        def forward(self, features):
            if features.ndim != 4 or features.shape[1] != self.feature_channels:
                raise ValueError("uncertainty features have the wrong NCHW shape")
            return self.network(features)


    class BoundedUncertaintyFlowRefinerV1(nn.Module):
        """Predict a bounded residual while treating uncertainty as fixed evidence."""

        def __init__(
            self,
            feature_channels: int,
            hidden_channels: int = 32,
            maximum_update_px: float = 1.0,
            evidence_role: UncertaintyEvidenceRoleV1 = (
                UncertaintyEvidenceRoleV1.BASE_FLOW
            ),
            minimum_update_fraction: float = 0.0,
        ):
            super().__init__()
            if maximum_update_px <= 0:
                raise ValueError("flow refiner needs a positive update bound")
            if not isinstance(evidence_role, UncertaintyEvidenceRoleV1):
                raise ValueError("flow refiner uncertainty role must be typed")
            if not 0.0 <= minimum_update_fraction <= 1.0:
                raise ValueError("minimum update fraction must lie in [0,1]")
            self.feature_channels = int(feature_channels)
            self.maximum_update_px = float(maximum_update_px)
            self.evidence_role = evidence_role
            self.minimum_update_fraction = float(minimum_update_fraction)
            self.network = nn.Sequential(
                nn.Conv2d(self.feature_channels + 3, hidden_channels, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(hidden_channels, 2, 3, padding=1),
            )

        def forward(self, features, base_flow, log_variance, valid_mask=None):
            if features.ndim != 4 or features.shape[1] != self.feature_channels:
                raise ValueError("refiner features have the wrong NCHW shape")
            if base_flow.ndim != 4 or base_flow.shape[1] != 2:
                raise ValueError("base flow must have shape N2HW")
            if features.shape[0] != base_flow.shape[0] or features.shape[2:] != base_flow.shape[2:]:
                raise ValueError("refiner features and base flow must share N,H,W")
            if log_variance.shape != base_flow[:, :1].shape:
                raise ValueError("log variance must have shape N1HW")
            if not bool(torch.isfinite(features).all()):
                raise ValueError("refiner features must be finite")
            if not bool(torch.isfinite(base_flow).all()):
                raise ValueError("base flow must be finite")
            if not bool(torch.isfinite(log_variance).all()):
                raise ValueError("log variance must be finite")
            if valid_mask is None:
                valid = torch.ones_like(log_variance, dtype=torch.bool)
            else:
                if valid_mask.shape == base_flow[:, 0].shape:
                    valid = valid_mask[:, None]
                elif valid_mask.shape == log_variance.shape:
                    valid = valid_mask
                else:
                    raise ValueError("refiner valid mask must have shape NHW or N1HW")
                if valid.dtype != torch.bool:
                    raise ValueError("refiner valid mask must be boolean")
            reliability = torch.sigmoid(-log_variance.detach())
            if self.evidence_role is UncertaintyEvidenceRoleV1.BASE_FLOW:
                raw_gate = 1.0 - reliability
            elif self.evidence_role is UncertaintyEvidenceRoleV1.PROPOSED_FLOW:
                raw_gate = reliability
            else:
                raw_gate = torch.ones_like(reliability)
            update_gate = self.minimum_update_fraction + (
                1.0 - self.minimum_update_fraction
            ) * raw_gate
            raw = self.network(torch.cat((features, base_flow, reliability), dim=1))
            direction = torch.tanh(raw)
            norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1.0)
            delta = direction / norm * update_gate * self.maximum_update_px
            # Invalid pixels are a preservation boundary, not merely zero loss.
            # ``where`` makes the output exactly equal to the supplied base flow.
            return torch.where(valid.expand_as(base_flow), base_flow + delta, base_flow)

else:

    class DecoupledUncertaintyHeadV1:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("DecoupledUncertaintyHeadV1 requires torch")

    class BoundedUncertaintyFlowRefinerV1:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("BoundedUncertaintyFlowRefinerV1 requires torch")


def augmentation_consistency_laplace_loss_v1(
    log_scale,
    reference_flow,
    restored_augmented_flow,
    valid_mask,
):
    """Laplace NLL with a detached augmentation-consistency teacher."""

    if torch is None:
        raise ImportError("augmentation consistency loss requires torch")
    if reference_flow.shape != restored_augmented_flow.shape:
        raise ValueError("consistency flows must have identical shapes")
    if reference_flow.ndim != 4 or reference_flow.shape[1] != 2:
        raise ValueError("consistency flows must have shape N2HW")
    if log_scale.shape != reference_flow[:, :1].shape:
        raise ValueError("log scale must have shape N1HW")
    if valid_mask.shape not in {log_scale.shape, log_scale[:, 0].shape}:
        raise ValueError("valid mask has the wrong shape")
    if valid_mask.dtype != torch.bool:
        raise ValueError("consistency valid mask must be boolean")
    valid = valid_mask
    if valid.ndim == 3:
        valid = valid[:, None]
    if not bool(valid.any()):
        raise ValueError("augmentation consistency has empty valid support")
    discrepancy = torch.linalg.vector_norm(
        restored_augmented_flow.detach() - reference_flow.detach(),
        dim=1,
        keepdim=True,
    )
    selected_discrepancy = discrepancy[valid]
    selected_log_scale = log_scale[valid]
    if not bool(torch.isfinite(selected_discrepancy).all()) or not bool(
        torch.isfinite(selected_log_scale).all()
    ):
        raise ValueError("consistency evidence must be finite on valid support")
    # Keep the heteroscedastic NLL numerically stable without changing the
    # receipt-level semantics of the predicted log scale.
    bounded_log_scale = selected_log_scale.clamp(-30.0, 30.0)
    return (
        selected_discrepancy * torch.exp(-bounded_log_scale) + bounded_log_scale
    ).mean()


__all__ = [
    "UNCERTAINTY_AWARE_FLOW_SCHEMA_V1",
    "BoundedUncertaintyFlowRefinerV1",
    "DecoupledTrainingPolicyV1",
    "DecoupledUncertaintyHeadV1",
    "FlowChangeAuthorizationReceiptV1",
    "UncertaintyAvailabilityV1",
    "UncertaintyClaimSemanticsV1",
    "UncertaintyEvidenceRoleV1",
    "UncertaintyFlowModeV1",
    "UncertaintyMapSemanticsV1",
    "UncertaintyReceiptV1",
    "UncertaintyRuntimePolicyV1",
    "UncertaintySummaryV1",
    "UncertaintyTrainingStageV1",
    "augmentation_consistency_laplace_loss_v1",
    "authorize_uncertainty_for_policy_v1",
    "bounded_uncertainty_guided_flow_v1",
    "summarize_uncertainty_v1",
    "uncertainty_reliability_v1",
    "uncertainty_update_gate_v1",
]
