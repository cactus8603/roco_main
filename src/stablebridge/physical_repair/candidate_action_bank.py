"""Frozen, production-neutral action templates for selector candidate banks.

The selector already accepts arbitrary exact arms.  This module therefore does
not add another selector or grant execution authority.  It freezes the eight
pieces needed to materialize a new arm -- action id, operator, strength,
endpoint, support, cost, availability, and receipt -- and ports the small
deterministic Optical Flow action grid into the shared package.

All records below are development/test-only.  Their source evidence used a
Spring-trained SEA-RAFT checkpoint on exposed Spring scenes; callers must still
create case-bound source/support/rollback hashes and pass the normal selector
calibration and harm gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ACTION_ENDPOINTS = frozenset({"first", "second", "both"})
ACTION_AVAILABILITY = frozenset({
    "AVAILABLE_TEST_ONLY",
    "AVAILABLE_BACKEND_BOUND_TEST_ONLY",
    "BLOCKED",
    "REJECTED",
})

OPTICAL_NATIVE_ACTION_ID = "CSB/OF/SEA-RAFT/action/P0"


def _finite_nonnegative(value: float, name: str) -> float:
    item = float(value)
    if not math.isfinite(item) or item < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return item


def _sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class FrozenActionCost:
    """Observed cost receipt; it is evidence, not a future runtime promise."""

    extra_matcher_trajectories: int
    observed_rows: int
    mean_wall_seconds: float
    p95_wall_seconds: float
    maximum_wall_seconds: float
    peak_vram_bytes: int
    hardware_profile: str
    source_sha256: str

    def __post_init__(self) -> None:
        if (isinstance(self.extra_matcher_trajectories, bool)
                or self.extra_matcher_trajectories < 0):
            raise ValueError("extra matcher trajectories must be nonnegative")
        if isinstance(self.observed_rows, bool) or self.observed_rows < 1:
            raise ValueError("an observed cost needs at least one row")
        mean = _finite_nonnegative(self.mean_wall_seconds, "mean wall time")
        p95 = _finite_nonnegative(self.p95_wall_seconds, "p95 wall time")
        maximum = _finite_nonnegative(self.maximum_wall_seconds, "maximum wall time")
        if p95 > maximum or mean > maximum:
            raise ValueError("cost summaries exceed the observed maximum")
        if isinstance(self.peak_vram_bytes, bool) or self.peak_vram_bytes < 0:
            raise ValueError("peak VRAM must be a nonnegative integer")
        if not self.hardware_profile:
            raise ValueError("cost hardware profile is required")
        if SHA256_RE.fullmatch(self.source_sha256) is None:
            raise ValueError("cost source needs a SHA-256 binding")

    def as_dict(self) -> dict[str, object]:
        return {
            "extra_matcher_trajectories": self.extra_matcher_trajectories,
            "observed_rows": self.observed_rows,
            "mean_wall_seconds": self.mean_wall_seconds,
            "p95_wall_seconds": self.p95_wall_seconds,
            "maximum_wall_seconds": self.maximum_wall_seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
            "hardware_profile": self.hardware_profile,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class FrozenActionReceipt:
    """Immutable provenance and claim boundary for one action template."""

    receipt_id: str
    operator_source_sha256: str
    evidence_sha256: str
    evidence_scope: str
    production_authority: bool = False

    def __post_init__(self) -> None:
        if not self.receipt_id or not self.evidence_scope:
            raise ValueError("action receipt identity and evidence scope are required")
        for value in (self.operator_source_sha256, self.evidence_sha256):
            if SHA256_RE.fullmatch(value) is None:
                raise ValueError("action receipt sources need SHA-256 bindings")
        if self.production_authority is not False:
            raise ValueError("candidate-bank receipts cannot grant production authority")

    def as_dict(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "operator_source_sha256": self.operator_source_sha256,
            "evidence_sha256": self.evidence_sha256,
            "evidence_scope": self.evidence_scope,
            "production_authority": self.production_authority,
        }


@dataclass(frozen=True)
class FrozenActionArm:
    """The minimal frozen template from which a Selector-v7 arm is bound."""

    action_id: str
    operator: str
    strength: float
    endpoint: str
    support: str
    cost: FrozenActionCost
    availability: str
    receipt: FrozenActionReceipt
    operator_parameters: tuple[tuple[str, object], ...] = ()
    arm_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if not self.action_id or not self.operator or not self.support:
            raise ValueError("action id, operator, and support are required")
        _finite_nonnegative(self.strength, "action strength")
        if self.endpoint not in ACTION_ENDPOINTS:
            raise ValueError("action endpoint must be first, second, or both")
        if self.availability not in ACTION_AVAILABILITY:
            raise ValueError("unknown action availability")
        if not isinstance(self.cost, FrozenActionCost):
            raise ValueError("action cost must be typed")
        if not isinstance(self.receipt, FrozenActionReceipt):
            raise ValueError("action receipt must be typed")
        keys = tuple(key for key, _ in self.operator_parameters)
        if (any(not isinstance(key, str) or not key for key in keys)
                or len(keys) != len(set(keys)) or keys != tuple(sorted(keys))):
            raise ValueError("operator parameters must have sorted unique string keys")
        payload = self.selector_payload(include_hash=False)
        expected = _sha256(payload)
        if self.arm_hash and self.arm_hash != expected:
            raise ValueError("frozen action arm hash drift")
        object.__setattr__(self, "arm_hash", expected)

    def selector_payload(self, *, include_hash: bool = False) -> dict[str, object]:
        """Return exactly the user-facing eight-field insertion contract."""

        result: dict[str, object] = {
            "action_id": self.action_id,
            "operator": {
                "id": self.operator,
                "parameters": dict(self.operator_parameters),
            },
            "strength": self.strength,
            "endpoint": self.endpoint,
            "support": self.support,
            "cost": self.cost.as_dict(),
            "availability": self.availability,
            "receipt": self.receipt.as_dict(),
        }
        if include_hash:
            result["arm_hash"] = self.arm_hash
        return result


@dataclass(frozen=True)
class PreparedPairAction:
    action_id: str
    images: np.ndarray
    matcher_iterations_override: int | None
    extra_matcher_trajectories: int


_RECOVERY_CODE_SHA256 = (
    "9a04afea980ab3bf3b36c0fb686d84a03a63e4b169f609c26fe91afd89a7e6fa"
)
_PILOT_CODE_SHA256 = (
    "fd14c78cdbdb87757acd8a9b5bd1d014703bd2471b97a9519fa22b4bd72cac6d"
)
_V3_CODE_SHA256 = (
    "5846fb6fc59ae3bfc04138f54ff0f7322d95e245d34a16daae8a77ba0e88fbc9"
)
_RECOVERY_EVIDENCE_SHA256 = (
    "55eb21b9adab7166dd81b613148ddd6fe5ec12f52b630fe39d66209de2968974"
)
_V3_EVIDENCE_SHA256 = (
    "e6e7e980c8a17f8335dba9deb9a020285ea592e746cf356ba8592d81dd7b6733"
)
_EVIDENCE_SCOPE = "E3_EXPOSED_SPRING_DEVELOPMENT_DIAGNOSTIC"
_SUPPORT = "whole_pair_finite_float32_v1"
_QUANTIZED_SUPPORT = "whole_pair_uint8_derived_float32_v1"
_RECOVERY_HARDWARE = "RTX3090_SPRING_1080x1920_BATCH1_RECOVERY_V2B"
_V3_HARDWARE = "RTX3090_SPRING_1080x1920_BATCH1_ACTION_FAMILY_V3"


def _cost(mean: float, p95: float, maximum: float, *, v3: bool = False) -> FrozenActionCost:
    return FrozenActionCost(
        extra_matcher_trajectories=1,
        observed_rows=1200,
        mean_wall_seconds=mean,
        p95_wall_seconds=p95,
        maximum_wall_seconds=maximum,
        peak_vram_bytes=758_295_040 if v3 else 1_224_415_744,
        hardware_profile=_V3_HARDWARE if v3 else _RECOVERY_HARDWARE,
        source_sha256=_V3_EVIDENCE_SHA256 if v3 else _RECOVERY_EVIDENCE_SHA256,
    )


def _receipt(action_id: str, *, source: str) -> FrozenActionReceipt:
    if source == "pilot":
        code, evidence = _PILOT_CODE_SHA256, _RECOVERY_EVIDENCE_SHA256
    elif source == "recovery":
        code, evidence = _RECOVERY_CODE_SHA256, _RECOVERY_EVIDENCE_SHA256
    elif source == "v3":
        code, evidence = _V3_CODE_SHA256, _V3_EVIDENCE_SHA256
    else:  # pragma: no cover - construction-time guard
        raise ValueError("unknown receipt source")
    return FrozenActionReceipt(
        receipt_id=f"optical-flow-track:{action_id}:frozen-evidence-v1",
        operator_source_sha256=code,
        evidence_sha256=evidence,
        evidence_scope=_EVIDENCE_SCOPE,
    )


def _arm(
    action_id: str,
    operator: str,
    strength: float,
    parameters: Mapping[str, object],
    cost: FrozenActionCost,
    *,
    source: str,
    support: str = _SUPPORT,
    availability: str = "AVAILABLE_TEST_ONLY",
) -> FrozenActionArm:
    return FrozenActionArm(
        action_id=action_id,
        operator=operator,
        strength=strength,
        endpoint="both",
        support=support,
        cost=cost,
        availability=availability,
        receipt=_receipt(action_id, source=source),
        operator_parameters=tuple(sorted(parameters.items())),
    )


_OPTICAL_ARMS = (
    _arm(
        "CSB/OF/SEA-RAFT/action/V3-gaussian-s0p5", "gaussian_blur_pair", 0.5,
        {"border": "BORDER_REFLECT_101", "sigma": 0.5},
        _cost(0.362343, 0.429387, 10.241477, v3=True), source="v3",
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/P1-gaussian-s1", "gaussian_blur_pair", 1.0,
        {"border": "BORDER_REFLECT_101", "sigma": 1.0},
        _cost(0.245109, 0.335293, 1.261549), source="pilot",
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/R1-gaussian-s1p5", "gaussian_blur_pair", 1.5,
        {"border": "BORDER_REFLECT_101", "sigma": 1.5},
        _cost(0.181532, 0.206029, 4.998507), source="recovery",
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/R2-gaussian-s2", "gaussian_blur_pair", 2.0,
        {"border": "BORDER_REFLECT_101", "sigma": 2.0},
        _cost(0.181924, 0.199584, 4.957938), source="recovery",
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/P2-unsharp-s1", "unsharp_mask_pair", 0.5,
        {"amount": 0.5, "clip": (0.0, 1.0), "sigma": 1.0},
        _cost(0.344811, 0.473038, 0.643602), source="pilot",
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/R3-unsharp-a1", "unsharp_mask_pair", 1.0,
        {"amount": 1.0, "clip": (0.0, 1.0), "sigma": 1.0},
        _cost(0.176747, 0.193681, 4.950263), source="recovery",
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5", "unsharp_mask_pair", 1.5,
        {"amount": 1.5, "clip": (0.0, 1.0), "sigma": 1.0},
        _cost(0.179939, 0.199244, 4.958334), source="recovery",
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/V3-radiometric-blend-l0p25",
        "joint_pair_percentile_blend", 0.25,
        {"lambda": 0.25, "lower": 1.0, "upper": 99.0},
        _cost(0.573799, 0.756338, 18.119995, v3=True), source="v3",
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/V3-radiometric-blend-l0p5",
        "joint_pair_percentile_blend", 0.5,
        {"lambda": 0.5, "lower": 1.0, "upper": 99.0},
        _cost(0.538783, 0.750080, 11.270877, v3=True), source="v3",
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/V3-radiometric-blend-l0p75",
        "joint_pair_percentile_blend", 0.75,
        {"lambda": 0.75, "lower": 1.0, "upper": 99.0},
        _cost(0.535525, 0.719383, 11.070053, v3=True), source="v3",
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1",
        "joint_pair_percentile_normalization", 1.0,
        {"lower": 1.0, "upper": 99.0},
        _cost(0.302952, 0.395301, 0.524167), source="pilot",
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/R5-joint-channel-percentile-s1",
        "joint_pair_channel_percentile_normalization", 1.0,
        {"lower": 1.0, "upper": 99.0},
        _cost(0.175814, 0.191086, 4.950071), source="recovery",
        support=_QUANTIZED_SUPPORT,
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/R6-joint-percentile-s5",
        "joint_pair_percentile_normalization", 5.0,
        {"lower": 5.0, "upper": 95.0},
        _cost(0.179094, 0.197916, 4.957791), source="recovery",
        support=_QUANTIZED_SUPPORT,
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/P4-iters8", "matcher_iteration_override", 8.0,
        {"iterations": 8, "matcher": "SEA-RAFT"},
        _cost(0.140936, 0.164664, 0.192868), source="pilot",
        availability="AVAILABLE_BACKEND_BOUND_TEST_ONLY",
    ),
    _arm(
        "CSB/OF/SEA-RAFT/action/R7-iters12", "matcher_iteration_override", 12.0,
        {"iterations": 12, "matcher": "SEA-RAFT"},
        _cost(0.152001, 0.168472, 3.186374), source="recovery",
        availability="AVAILABLE_BACKEND_BOUND_TEST_ONLY",
    ),
)

OPTICAL_FLOW_ACTION_BANK: Mapping[str, FrozenActionArm] = MappingProxyType({
    arm.action_id: arm for arm in _OPTICAL_ARMS
})
OPTICAL_FLOW_ACTION_BANK_HASH = _sha256([
    OPTICAL_FLOW_ACTION_BANK[key].selector_payload(include_hash=True)
    for key in sorted(OPTICAL_FLOW_ACTION_BANK)
])

# P0 is the mandatory native fallback and is deliberately not duplicated here.
# These four nonnative mechanisms are the smallest family-balanced pilot from
# the upstream P0--P4 study.  The other eleven frozen variants remain available
# for explicit strength-ablation manifests, but should not silently enlarge a
# selector's multiplicity burden.
OPTICAL_FLOW_MINIMAL_PILOT_ACTION_IDS = (
    "CSB/OF/SEA-RAFT/action/P1-gaussian-s1",
    "CSB/OF/SEA-RAFT/action/P2-unsharp-s1",
    "CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1",
    "CSB/OF/SEA-RAFT/action/P4-iters8",
)
OPTICAL_FLOW_MINIMAL_PILOT_BANK: Mapping[str, FrozenActionArm] = MappingProxyType({
    action_id: OPTICAL_FLOW_ACTION_BANK[action_id]
    for action_id in OPTICAL_FLOW_MINIMAL_PILOT_ACTION_IDS
})
OPTICAL_FLOW_MINIMAL_PILOT_BANK_HASH = _sha256([
    OPTICAL_FLOW_MINIMAL_PILOT_BANK[key].selector_payload(include_hash=True)
    for key in sorted(OPTICAL_FLOW_MINIMAL_PILOT_BANK)
])

# E278 is the smallest opened-development candidate bank that simultaneously
# retained three distinct repair mechanisms, passed its nested scene-OOF
# routing gate, and gave every retained exact control a positive scene-cluster
# bootstrap lower bound for routed contribution.  The same E3 panel was used
# adaptively in E275--E278, so this is a frozen candidate for fresh validation,
# not a validated/final bank.  Every arm remains TEST_ONLY and native remains
# the mandatory fallback.
OPTICAL_FLOW_OPENED_CANDIDATE_ACTION_IDS = (
    "CSB/OF/SEA-RAFT/action/P1-gaussian-s1",
    "CSB/OF/SEA-RAFT/action/R2-gaussian-s2",
    "CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5",
    "CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1",
)
OPTICAL_FLOW_OPENED_CANDIDATE_FAMILIES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "optical.lowpass_hf_suppression.v1": (
        "CSB/OF/SEA-RAFT/action/P1-gaussian-s1",
        "CSB/OF/SEA-RAFT/action/R2-gaussian-s2",
    ),
    "optical.detail_recovery_unsharp.v1": (
        "CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5",
    ),
    "optical.joint_radiometry.v1": (
        "CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1",
    ),
})
OPTICAL_FLOW_OPENED_CANDIDATE_BANK: Mapping[str, FrozenActionArm] = MappingProxyType({
    action_id: OPTICAL_FLOW_ACTION_BANK[action_id]
    for action_id in OPTICAL_FLOW_OPENED_CANDIDATE_ACTION_IDS
})
OPTICAL_FLOW_OPENED_CANDIDATE_BANK_HASH = _sha256([
    OPTICAL_FLOW_OPENED_CANDIDATE_BANK[key].selector_payload(include_hash=True)
    for key in sorted(OPTICAL_FLOW_OPENED_CANDIDATE_BANK)
])


def validate_frozen_action_bank(
    bank: Mapping[str, FrozenActionArm] = OPTICAL_FLOW_ACTION_BANK,
) -> None:
    if not bank or len(bank) != len(set(bank)):
        raise ValueError("action bank must be nonempty with unique ids")
    for key, arm in bank.items():
        if key != arm.action_id:
            raise ValueError("action-bank key and action id differ")
        if arm.receipt.production_authority:
            raise ValueError("candidate action bank cannot grant production authority")


def _validate_pair(images: np.ndarray) -> np.ndarray:
    item = np.asarray(images)
    if (item.dtype != np.float32 or item.ndim != 4
            or item.shape[0] != 2 or item.shape[1] != 3
            or not np.isfinite(item).all()
            or float(item.min()) < 0.0 or float(item.max()) > 1.0):
        raise ValueError("expected finite float32 pair [2,3,H,W] in [0,1]")
    return item


def _gaussian(images: np.ndarray, sigma: float) -> np.ndarray:
    frames = np.transpose(images, (0, 2, 3, 1))
    result = [
        cv2.GaussianBlur(
            frame, (0, 0), sigmaX=sigma, sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        for frame in frames
    ]
    return np.ascontiguousarray(
        np.transpose(np.stack(result), (0, 3, 1, 2)), dtype=np.float32,
    )


def _percentile_normalize(
    images: np.ndarray, lower: float, upper: float,
) -> np.ndarray:
    low, high = np.percentile(images, [lower, upper])
    denominator = max(float(high - low), 1e-6)
    return np.ascontiguousarray(
        np.clip((images - np.float32(low)) / np.float32(denominator), 0.0, 1.0),
        dtype=np.float32,
    )


def _quantized_percentile(
    images: np.ndarray, percentile: float, *, per_channel: bool,
) -> np.ndarray:
    scaled = images * np.float32(255.0)
    rounded = np.rint(scaled)
    if float(np.max(np.abs(scaled - rounded))) > 2.6e-5:
        raise ValueError("quantized percentile actions require uint8-derived inputs")
    quantized = rounded.astype(np.uint8)
    channels = np.stack([
        np.bincount(quantized[:, index].ravel(), minlength=256)
        for index in range(3)
    ]).astype(np.int64, copy=False)
    histograms = channels if per_channel else channels.sum(axis=0, keepdims=True)
    values: list[float] = []
    for histogram in histograms:
        count = int(histogram.sum())
        position = (count - 1) * percentile / 100.0
        lower_rank, upper_rank = int(np.floor(position)), int(np.ceil(position))
        fraction = position - lower_rank
        cumulative = np.cumsum(histogram)
        lower_value = int(np.searchsorted(cumulative, lower_rank + 1))
        upper_value = int(np.searchsorted(cumulative, upper_rank + 1))
        values.append((lower_value + fraction * (upper_value - lower_value)) / 255.0)
    result = np.asarray(values, dtype=np.float32)
    return result.reshape(1, 3, 1, 1) if per_channel else result[0]


def prepare_optical_pair_action(
    images: np.ndarray, action_id: str,
) -> PreparedPairAction:
    """Materialize an Optical action without reading labels, truth, or outcomes."""

    source = _validate_pair(images)
    if action_id == OPTICAL_NATIVE_ACTION_ID:
        return PreparedPairAction(action_id, source, None, 0)
    try:
        arm = OPTICAL_FLOW_ACTION_BANK[action_id]
    except KeyError as exc:
        raise ValueError(f"unknown frozen Optical action: {action_id}") from exc
    parameters = dict(arm.operator_parameters)
    if arm.operator == "gaussian_blur_pair":
        output = _gaussian(source, float(parameters["sigma"]))
        iterations = None
    elif arm.operator == "unsharp_mask_pair":
        blur = _gaussian(source, float(parameters["sigma"]))
        amount = np.float32(parameters["amount"])
        output = np.ascontiguousarray(
            np.clip(source + amount * (source - blur), 0.0, 1.0),
            dtype=np.float32,
        )
        iterations = None
    elif arm.operator == "joint_pair_percentile_blend":
        normalized = _percentile_normalize(
            source, float(parameters["lower"]), float(parameters["upper"]),
        )
        weight = np.float32(parameters["lambda"])
        output = np.ascontiguousarray(
            np.clip(source + weight * (normalized - source), 0.0, 1.0),
            dtype=np.float32,
        )
        iterations = None
    elif arm.operator == "joint_pair_percentile_normalization":
        lower, upper = float(parameters["lower"]), float(parameters["upper"])
        if (lower, upper) == (5.0, 95.0):
            low = _quantized_percentile(source, lower, per_channel=False)
            high = _quantized_percentile(source, upper, per_channel=False)
            output = np.ascontiguousarray(
                np.clip((source - low) / max(float(high - low), 1e-6), 0.0, 1.0),
                dtype=np.float32,
            )
        else:
            output = _percentile_normalize(source, lower, upper)
        iterations = None
    elif arm.operator == "joint_pair_channel_percentile_normalization":
        low = _quantized_percentile(source, float(parameters["lower"]), per_channel=True)
        high = _quantized_percentile(source, float(parameters["upper"]), per_channel=True)
        output = np.ascontiguousarray(
            np.clip((source - low) / np.maximum(high - low, 1e-6), 0.0, 1.0),
            dtype=np.float32,
        )
        iterations = None
    elif arm.operator == "matcher_iteration_override":
        output = source
        iterations = int(parameters["iterations"])
    else:  # pragma: no cover - registry validation guard
        raise RuntimeError(f"unimplemented frozen operator: {arm.operator}")
    if output.shape != source.shape or output.dtype != np.float32 or not np.isfinite(output).all():
        raise RuntimeError("frozen Optical action violated the pair tensor contract")
    return PreparedPairAction(
        action_id=action_id,
        images=output,
        matcher_iterations_override=iterations,
        extra_matcher_trajectories=arm.cost.extra_matcher_trajectories,
    )


validate_frozen_action_bank()


__all__ = [
    "ACTION_AVAILABILITY",
    "ACTION_ENDPOINTS",
    "FrozenActionArm",
    "FrozenActionCost",
    "FrozenActionReceipt",
    "OPTICAL_FLOW_ACTION_BANK",
    "OPTICAL_FLOW_ACTION_BANK_HASH",
    "OPTICAL_FLOW_MINIMAL_PILOT_ACTION_IDS",
    "OPTICAL_FLOW_MINIMAL_PILOT_BANK",
    "OPTICAL_FLOW_MINIMAL_PILOT_BANK_HASH",
    "OPTICAL_FLOW_OPENED_CANDIDATE_ACTION_IDS",
    "OPTICAL_FLOW_OPENED_CANDIDATE_FAMILIES",
    "OPTICAL_FLOW_OPENED_CANDIDATE_BANK",
    "OPTICAL_FLOW_OPENED_CANDIDATE_BANK_HASH",
    "OPTICAL_NATIVE_ACTION_ID",
    "PreparedPairAction",
    "prepare_optical_pair_action",
    "validate_frozen_action_bank",
]
