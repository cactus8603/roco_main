"""Action-bank-independent uncertainty quality and association metrics.

The uncertainty head is trained against a self-consistency distance, so proxy
NLL alone cannot establish task calibration.  This module evaluates a frozen
map against held-out endpoint error and, separately, tests whether a frozen
uncertainty summary is associated with an exact control's observed outcome.
It never converts association into a causal or safety claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def _vectors(*values: object) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value).reshape(-1) for value in values)
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("evaluation vectors must have identical shapes")
    return arrays


def _finite_selected(
    log_scale: object,
    endpoint_error: object,
    valid_mask: object,
) -> tuple[np.ndarray, np.ndarray]:
    uncertainty, error, valid = _vectors(log_scale, endpoint_error, valid_mask)
    valid = valid.astype(bool, copy=False)
    valid &= np.isfinite(uncertainty) & np.isfinite(error) & (error >= 0)
    if not valid.any():
        raise ValueError("uncertainty evaluation has empty finite support")
    return uncertainty[valid].astype(np.float64), error[valid].astype(np.float64)


def _average_ranks(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    sorted_value = value[order]
    ranks = np.empty(value.size, dtype=np.float64)
    # Vectorized tie groups avoid a Python loop per pixel for dense maps.
    starts_mask = np.empty(value.size, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_value[1:] != sorted_value[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.concatenate((starts[1:], np.asarray((value.size,), dtype=starts.dtype)))
    average = 0.5 * (starts + ends - 1) + 1.0
    ranks[order] = np.repeat(average, ends - starts)
    return ranks


def spearman_v1(first: object, second: object) -> float:
    left, right = _vectors(first, second)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return math.nan
    x, y = _average_ranks(left[valid]), _average_ranks(right[valid])
    x -= x.mean()
    y -= y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return math.nan if denominator == 0 else float(np.dot(x, y) / denominator)


def binary_auroc_v1(score: object, positive: object) -> float:
    scores, labels = _vectors(score, positive)
    valid = np.isfinite(scores)
    scores = scores[valid].astype(np.float64)
    labels = labels[valid].astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = _average_ranks(scores)
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _risk_curve(error: np.ndarray, ordering: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = error[ordering]
    cumulative = np.cumsum(ordered, dtype=np.float64)
    counts = np.arange(1, error.size + 1, dtype=np.float64)
    return counts / error.size, cumulative / counts


@dataclass(frozen=True)
class UncertaintyEvaluationV1:
    valid_count: int
    mean_endpoint_error: float
    exponential_nll: float
    spearman_error: float
    aurc: float
    oracle_aurc: float
    ause: float
    severe_threshold: float
    severe_auroc: float
    coverage_calibration_mae: float
    evaluated_coverages: tuple[float, ...]
    empirical_coverages: tuple[float, ...]


def evaluate_log_scale_uncertainty_v1(
    log_scale: object,
    endpoint_error: object,
    valid_mask: object,
    *,
    severe_threshold: float,
    calibration_coverages: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> UncertaintyEvaluationV1:
    """Evaluate ``u=log(b)`` for the NLL ``error / b + log(b)``.

    That NLL corresponds to an exponential model for the nonnegative endpoint
    error magnitude, whose q-coverage bound is ``-b log(1-q)``.
    """

    threshold = float(severe_threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("severe threshold must be finite and nonnegative")
    uncertainty, error = _finite_selected(log_scale, endpoint_error, valid_mask)
    clipped = np.clip(uncertainty, -30.0, 30.0)
    scale = np.exp(clipped)
    nll = float(np.mean(error / scale + clipped))
    coverage, risk = _risk_curve(error, np.argsort(uncertainty, kind="mergesort"))
    _, oracle_risk = _risk_curve(error, np.argsort(error, kind="mergesort"))
    if error.size == 1:
        aurc = oracle_aurc = float(error[0])
    else:
        aurc = float(np.trapezoid(risk, coverage))
        oracle_aurc = float(np.trapezoid(oracle_risk, coverage))
    requested = tuple(float(value) for value in calibration_coverages)
    if not requested or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in requested
    ):
        raise ValueError("calibration coverages must lie strictly in (0,1)")
    empirical = tuple(
        float(np.mean(error <= (-scale * math.log1p(-value))))
        for value in requested
    )
    calibration_mae = float(np.mean(np.abs(np.asarray(empirical) - requested)))
    return UncertaintyEvaluationV1(
        valid_count=int(error.size),
        mean_endpoint_error=float(error.mean()),
        exponential_nll=nll,
        spearman_error=spearman_v1(uncertainty, error),
        aurc=aurc,
        oracle_aurc=oracle_aurc,
        ause=max(0.0, aurc - oracle_aurc),
        severe_threshold=threshold,
        severe_auroc=binary_auroc_v1(uncertainty, error >= threshold),
        coverage_calibration_mae=calibration_mae,
        evaluated_coverages=requested,
        empirical_coverages=empirical,
    )


@dataclass(frozen=True)
class GroupBootstrapIntervalV1:
    point: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    group_count: int


def grouped_bootstrap_mean_v1(
    values: object,
    groups: object,
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> GroupBootstrapIntervalV1:
    observations, group_values = _vectors(values, groups)
    observations = observations.astype(np.float64)
    valid = np.isfinite(observations)
    observations, group_values = observations[valid], group_values[valid]
    if not observations.size:
        raise ValueError("group bootstrap has no finite observations")
    if isinstance(resamples, bool) or resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie in (0,1)")
    unique = np.unique(group_values)
    if unique.size < 2:
        raise ValueError("group bootstrap needs at least two groups")
    by_group = {group: observations[group_values == group] for group in unique}
    generator = np.random.default_rng(seed)
    sampled_means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draw = generator.choice(unique, size=unique.size, replace=True)
        sampled_means[index] = np.concatenate([by_group[group] for group in draw]).mean()
    alpha = 1.0 - confidence
    return GroupBootstrapIntervalV1(
        point=float(observations.mean()),
        lower=float(np.quantile(sampled_means, alpha / 2)),
        upper=float(np.quantile(sampled_means, 1.0 - alpha / 2)),
        confidence=float(confidence),
        resamples=int(resamples),
        group_count=int(unique.size),
    )


@dataclass(frozen=True)
class ActionUncertaintyAssociationV1:
    exact_control_id: str
    count: int
    uncertainty_gain_spearman: float
    uncertainty_harm_spearman: float
    positive_gain_auroc: float
    severe_harm_auroc: float


def action_uncertainty_associations_v1(
    uncertainty_score: object,
    gain: object,
    harm: object,
    exact_control_ids: Sequence[str],
    *,
    severe_harm_threshold: float,
) -> tuple[ActionUncertaintyAssociationV1, ...]:
    uncertainty, gains, harms = _vectors(uncertainty_score, gain, harm)
    controls = np.asarray(tuple(exact_control_ids), dtype=object)
    if controls.shape != uncertainty.shape:
        raise ValueError("exact control ids must align with outcomes")
    threshold = float(severe_harm_threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("severe harm threshold must be finite and nonnegative")
    rows = []
    for control in sorted(set(controls.tolist())):
        if not isinstance(control, str) or not control:
            raise ValueError("exact control ids must be nonempty strings")
        selected = controls == control
        valid = selected & np.isfinite(uncertainty) & np.isfinite(gains) & np.isfinite(harms)
        u, g, h = uncertainty[valid], gains[valid], harms[valid]
        rows.append(ActionUncertaintyAssociationV1(
            exact_control_id=control,
            count=int(valid.sum()),
            uncertainty_gain_spearman=spearman_v1(u, g),
            uncertainty_harm_spearman=spearman_v1(u, h),
            positive_gain_auroc=binary_auroc_v1(u, g > 0),
            severe_harm_auroc=binary_auroc_v1(u, h >= threshold),
        ))
    return tuple(rows)


__all__ = [
    "ActionUncertaintyAssociationV1",
    "GroupBootstrapIntervalV1",
    "UncertaintyEvaluationV1",
    "action_uncertainty_associations_v1",
    "binary_auroc_v1",
    "evaluate_log_scale_uncertainty_v1",
    "grouped_bootstrap_mean_v1",
    "spearman_v1",
]
