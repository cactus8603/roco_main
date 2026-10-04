"""Shared candidate utility learning with scene-separated, post-selection calibration.

The model predicts signed, *unclipped* error reduction and P(harm > 1 pixel).
Only fit labels train model weights. Calibration reproduces the matcher policy:
eligible candidates -> gain argmax with identity fixed to zero -> acceptance
thresholds. Empirical calibration is not a statistical or OOD safety guarantee.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Mapping

import numpy as np
import torch
from torch import nn


FEATURE_SCHEMA = "candidate_features_v1_12"
FEATURE_NAMES = (
    "query_risk", "paired_cost_clipped4", "encoder_cost_clipped2",
    "image_descriptor_cost_clipped2", "baseline_cost_clipped4",
    "update_length_div32_clipped32", "candidate_dx_div128",
    "candidate_dy_div128", "candidate_supported", "baseline_supported",
    "is_stereo", "is_history",
)
_SPLIT_NAMES = {"fit": "fit", "calibration": "calibration", "calib": "calibration",
                "test": "evaluation", "eval": "evaluation", "dev_eval": "evaluation",
                "development_evaluation": "evaluation", "evaluation": "evaluation"}


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 60
    hidden_dim: int = 32
    batch_size: int = 4096
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    max_fit_rows: int = 131072
    seed: int = 20260916
    harm_loss_weight: float = 1.0
    max_update_px: float = 32.0
    require_baseline_support: bool = True
    severe_harm_px: float = 1.0
    max_calibration_severe_harm_rate: float = 0.05
    max_calibration_harm_benefit_ratio: float = 0.25
    min_calibration_scenes: int = 2
    min_accepted_scenes: int = 2
    min_accepted_queries: int = 32
    threshold_quantiles: int = 9
    min_scene_gain_px: float = 0.0

    def __post_init__(self):
        for name in ("epochs", "hidden_dim", "batch_size", "max_fit_rows", "min_calibration_scenes",
                     "min_accepted_scenes", "min_accepted_queries", "threshold_quantiles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "weight_decay", "harm_loss_weight", "max_update_px",
                     "severe_harm_px", "max_calibration_harm_benefit_ratio", "min_scene_gain_px"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0 or self.max_update_px == 0 or self.harm_loss_weight == 0:
            raise ValueError("learning_rate, max_update_px and harm_loss_weight must be positive")
        if self.severe_harm_px != 1.0:
            raise ValueError("The v1 harm head is defined specifically as harm > 1 native pixel")
        if not 0 <= self.max_calibration_severe_harm_rate <= 1:
            raise ValueError("max_calibration_severe_harm_rate must be in [0,1]")
        if self.max_update_px >= 1024:
            raise ValueError("12D features clip update length at 1024px; larger policies cannot be reconstructed")


class _UtilityNet(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(12, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, 2))

    def forward(self, x):
        return self.network(x)


def _strings(value, name):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "USiu":
        raise ValueError(f"{name} must be a one-dimensional string/integer array, without pickled objects")
    result = value.astype(str)
    if np.any(result == ""):
        raise ValueError(f"{name} cannot contain empty identifiers")
    return result


def validate_records(records: Mapping) -> dict:
    """Validate layout and scene/query split integrity, returning canonical arrays.

    Optional candidate_index explicitly identifies the identity as index zero.
    Without it, each query's first encountered row is identity; its zero update
    and baseline-equal error are checked. Row ordering otherwise is unrestricted.
    """
    required = ("features", "baseline_error", "candidate_error", "query_id", "scene", "split")
    missing = [name for name in required if name not in records]
    if missing:
        raise ValueError(f"Missing training arrays: {missing}")
    feature = np.asarray(records["features"], np.float32)
    if feature.ndim != 2 or feature.shape[1] != 12 or len(feature) == 0 or not np.isfinite(feature).all():
        raise ValueError("features must be nonempty, finite [M,12]")
    count = len(feature)
    data = {"features": feature}
    for name in ("baseline_error", "candidate_error"):
        array = np.asarray(records[name], np.float64)
        if array.shape != (count,) or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"{name} must contain M finite nonnegative native-pixel errors")
        data[name] = array
    for name in ("query_id", "scene", "split"):
        data[name] = _strings(records[name], name)
        if len(data[name]) != count:
            raise ValueError(f"{name} must contain M entries")
    unknown = set(data["split"]) - set(_SPLIT_NAMES)
    if unknown:
        raise ValueError(f"Unknown split names: {sorted(unknown)}")
    data["split"] = np.asarray([_SPLIT_NAMES[v] for v in data["split"]])
    if "feature_schema" in records and str(np.asarray(records["feature_schema"]).item()) != FEATURE_SCHEMA:
        raise ValueError("Candidate feature schema mismatch")
    for index in (8, 9, 10, 11):
        if not np.all(np.isin(feature[:, index], [0, 1])):
            raise ValueError(f"Feature {FEATURE_NAMES[index]} must be binary")
    if np.any(feature[:, 5] < 0):
        raise ValueError("Update lengths must be nonnegative")
    for scene in np.unique(data["scene"]):
        if len(np.unique(data["split"][data["scene"] == scene])) != 1:
            raise ValueError(f"Scene leakage across splits: {scene}")
    unique_queries, group = np.unique(data["query_id"], return_inverse=True)
    data["group"] = group
    data["unique_queries"] = unique_queries
    if "candidate_index" in records:
        candidate_index = np.asarray(records["candidate_index"])
        if candidate_index.shape != (count,) or candidate_index.dtype.kind not in "iu" or np.any(candidate_index < 0):
            raise ValueError("candidate_index must be nonnegative integer [M]")
        candidate_index = candidate_index.astype(np.int64)
    else:
        candidate_index = np.zeros(count, np.int64)
        counters = np.zeros(len(unique_queries), np.int64)
        for row, query in enumerate(group):
            candidate_index[row] = counters[query]
            counters[query] += 1
    data["candidate_index"] = candidate_index
    order = np.argsort(group, kind="stable")
    segments = np.split(order, np.flatnonzero(np.diff(group[order])) + 1)
    identity = np.empty(len(segments), np.int64)
    for query, rows in enumerate(segments):
        if len(np.unique(data["scene"][rows])) != 1 or len(np.unique(data["split"][rows])) != 1:
            raise ValueError(f"Query identity reused across scene/split boundaries: {unique_queries[query]}")
        if len(np.unique(candidate_index[rows])) != len(rows):
            raise ValueError(f"Duplicate candidate indices for query {unique_queries[query]}")
        roots = rows[candidate_index[rows] == 0]
        if len(roots) != 1:
            raise ValueError(f"Query {unique_queries[query]} needs exactly one identity candidate")
        first = int(roots[0])
        identity[query] = first
        if not np.allclose(data["baseline_error"][rows], data["baseline_error"][first], rtol=0, atol=1e-6):
            raise ValueError("Candidates in one query disagree on baseline error")
        if (abs(data["candidate_error"][first] - data["baseline_error"][first]) > 1e-6
                or abs(float(feature[first, 5])) > 1e-6):
            raise ValueError("Identity candidate must have zero update and baseline-equal error")
        if np.any(feature[rows, 9] != feature[first, 8]):
            raise ValueError("baseline_supported must agree with the identity's candidate_supported")
        if np.any(feature[rows, 10] != feature[first, 10]):
            raise ValueError("Candidates in one query disagree on task")
    data["identity_rows"] = identity
    data["groups_rows"] = segments
    data["gain"] = data["baseline_error"] - data["candidate_error"]  # deliberately no clipping
    data["severe_harm"] = (data["gain"] < -1.0).astype(np.float32)
    if not np.any(data["split"] == "fit"):
        raise ValueError("A nonempty fit split is required")
    return data


class CandidateAcceptor:
    """Frozen shared stereo/flow acceptor matching FeatureRematcher's interface."""
    feature_schema = FEATURE_SCHEMA

    def __init__(self, network, mean, std, gain_scale, config, *, gain_threshold=math.inf,
                 harm_threshold=0.0, report=None):
        self.network = network.eval().requires_grad_(False).cpu()
        self.mean = torch.as_tensor(np.array(mean, copy=True), dtype=torch.float32)
        self.std = torch.as_tensor(np.array(std, copy=True), dtype=torch.float32)
        self.gain_scale = float(gain_scale)
        self.config = config
        self.max_update_px = config.max_update_px
        self.require_baseline_support = config.require_baseline_support
        self.gain_threshold = float(gain_threshold)
        self.harm_threshold = float(harm_threshold)
        self.report = report or {}
        if self.mean.shape != (12,) or self.std.shape != (12,) or not torch.isfinite(self.mean).all() \
                or not torch.isfinite(self.std).all() or not torch.all(self.std > 0):
            raise ValueError("Invalid fit-only normalization")
        if not math.isfinite(self.gain_scale) or self.gain_scale <= 0:
            raise ValueError("Invalid gain scale")

    def predict(self, features):
        array = np.asarray(features, np.float32)
        if array.ndim < 2 or array.shape[-1] != 12 or not np.isfinite(array).all():
            raise ValueError("Expected finite candidate features [...,12]")
        shape = array.shape[:-1]
        rows = torch.from_numpy(np.array(array.reshape(-1, 12), copy=True))
        outputs = []
        with torch.inference_mode():
            for chunk in rows.split(self.config.batch_size):
                outputs.append(self.network((chunk - self.mean) / self.std).cpu())
        if not outputs:
            return np.empty(shape, np.float32), np.empty(shape, np.float32)
        output = torch.cat(outputs)
        gain = (output[:, 0] * self.gain_scale).numpy().reshape(shape)
        harm = output[:, 1].sigmoid().numpy().reshape(shape)
        if not np.isfinite(gain).all() or not np.isfinite(harm).all():
            raise FloatingPointError("Utility model produced non-finite predictions; reject this run")
        return gain, harm

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        policy = {"state": "reject_all" if not math.isfinite(self.gain_threshold) else "empirically_calibrated",
                  "gain_threshold": self.gain_threshold if math.isfinite(self.gain_threshold) else None,
                  "harm_threshold": self.harm_threshold}
        payload = {"schema_version": 1, "feature_schema": FEATURE_SCHEMA,
                   "feature_names": list(FEATURE_NAMES), "state_dict": self.network.state_dict(),
                   "mean": self.mean, "std": self.std, "gain_scale": self.gain_scale,
                   "config": asdict(self.config), "policy": policy, "report": self.report}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        metadata = {key: value for key, value in payload.items() if key not in ("state_dict", "mean", "std")}
        metadata.update({"path": str(path.resolve()), "sha256": _digest(path),
                         "mean": self.mean.tolist(), "std": self.std.tolist()})
        sidecar = path.with_suffix(path.suffix + ".json")
        temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
        temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(sidecar)


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_acceptor(path: str | Path) -> CandidateAcceptor:
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if metadata["sha256"] != _digest(path):
        raise ValueError("Utility checkpoint digest differs from its provenance")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("Unsupported candidate utility checkpoint schema")
    config = TrainingConfig(**payload["config"])
    network = _UtilityNet(config.hidden_dim)
    network.load_state_dict(payload["state_dict"], strict=True)
    policy = payload["policy"]
    threshold = math.inf if policy["state"] == "reject_all" else float(policy["gain_threshold"])
    return CandidateAcceptor(network, payload["mean"].numpy(), payload["std"].numpy(), payload["gain_scale"],
                             config, gain_threshold=threshold, harm_threshold=policy["harm_threshold"], report=payload["report"])


def select_queries(data: dict, gain: np.ndarray, harm: np.ndarray, config: TrainingConfig,
                   split="calibration") -> dict:
    """Reproduce deployment selection *before* acceptance, including tie breaks.

    Candidate order is candidate_index; identity is first and fixed to gain 0.
    High predicted harm does not remove a candidate before argmax, matching the
    current matcher. All query rows remain in the evaluation denominator.
    """
    gain, harm = np.asarray(gain), np.asarray(harm)
    if gain.shape != (len(data["features"]),) or harm.shape != gain.shape:
        raise ValueError("Predictions must align with all candidate rows")
    if not np.isfinite(gain).all() or not np.isfinite(harm).all() or np.any((harm < 0) | (harm > 1)):
        raise ValueError("Candidate predictions must be finite, with harm probabilities in [0,1]")
    feature = data["features"]
    eligible = (feature[:, 8] > 0.5) & (feature[:, 5] * 32 <= config.max_update_px)
    if config.require_baseline_support:
        eligible &= feature[:, 9] > 0.5
    winners, identities = [], []
    for rows, root in zip(data["groups_rows"], data["identity_rows"]):
        if data["split"][root] != split:
            continue
        rows = rows[np.argsort(data["candidate_index"][rows], kind="stable")]
        scores = np.where(eligible[rows], gain[rows], -np.inf).copy()
        scores[data["candidate_index"][rows] == 0] = 0.0
        winner = int(rows[int(np.argmax(scores))])
        winners.append(winner)
        identities.append(root)
    selected = np.asarray(winners, np.int64)
    roots = np.asarray(identities, np.int64)
    return {"row": selected, "identity_row": roots,
            "query_id": data["query_id"][roots], "scene": data["scene"][roots],
            "task": np.where(feature[roots, 10] > .5, "stereo", "flow"),
            "predicted_gain": np.where(selected == roots, 0, gain[selected]),
            "predicted_harm": harm[selected], "actual_gain": data["gain"][selected],
            "eligible_nonidentity": (selected != roots) & eligible[selected]}


def calibrate_policy(selected: dict, config: TrainingConfig) -> dict:
    """Choose empirical thresholds on already selected calibration proposals.

    Requirements constrain observed calibration harms and per-scene gains. The
    same labels select thresholds, so these are calibration diagnostics, not an
    independent safety certificate or unbiased evaluation estimate.
    """
    scenes = np.unique(selected["scene"])
    tasks = np.unique(selected["task"])
    total = len(selected["scene"])
    common = {"calibration_queries": total, "calibration_scenes": scenes.tolist(),
              "selection_before_thresholds": True,
              "safety_claim": "empirical_calibration_only_no_finite_sample_or_OOD_guarantee",
              "denominator": "all_calibration_queries_in_each_task_scene_including_rejected",
              "calibration_tasks": tasks.tolist(),
              "aggregation": "equal_scene_mean_per_task_then_equal_task_mean; each_task_scene_gated_separately"}
    reject = {**common, "state": "reject_all", "gain_threshold": None, "harm_threshold": 0.0,
              "accepted_queries": 0, "scene_macro_gain_px": 0.0}
    if len(scenes) < config.min_calibration_scenes:
        return {**reject, "reason": "insufficient_scene_separated_calibration_data"}
    if set(tasks) != {"flow", "stereo"}:
        return {**reject, "reason": "both_tasks_required_to_calibrate_shared_policy"}
    for task in tasks:
        if len(np.unique(selected["scene"][selected["task"] == task])) < config.min_calibration_scenes:
            return {**reject, "reason": "insufficient_calibration_scenes_for_one_task"}
    eligible = selected["eligible_nonidentity"]
    possible = selected["predicted_gain"][eligible & (selected["predicted_gain"] > 0)]
    if len(possible) == 0:
        return {**reject, "reason": "no_positive_eligible_proposal_after_argmax"}
    thresholds = np.unique(np.r_[0.0, np.quantile(possible, np.linspace(0, 1, config.threshold_quantiles))])
    harm_thresholds = (0.0, .01, .025, .05, .1, .2, .5, 1.0)
    best = None
    considered = 0
    actual = selected["actual_gain"]
    for gain_threshold in thresholds:
        for harm_threshold in harm_thresholds:
            accepted = eligible & (selected["predicted_gain"] > gain_threshold) & (selected["predicted_harm"] <= harm_threshold)
            if int(accepted.sum()) < config.min_accepted_queries:
                continue
            if len(np.unique(selected["scene"][accepted])) < config.min_accepted_scenes:
                continue
            considered += 1
            severe_rate = float(np.mean(actual[accepted] < -config.severe_harm_px))
            benefit = float(np.maximum(actual[accepted], 0).sum())
            damage = float(np.maximum(-actual[accepted], 0).sum())
            ratio = damage / benefit if benefit else math.inf
            if severe_rate > config.max_calibration_severe_harm_rate or ratio > config.max_calibration_harm_benefit_ratio:
                continue
            task_metrics, scene_task_gains, scene_task_rates = {}, {}, {}
            satisfies_all_tasks = True
            for task in tasks:
                task_rows = selected["task"] == task
                task_accepted = task_rows & accepted
                task_scenes = np.unique(selected["scene"][task_rows])
                accepted_scenes = np.unique(selected["scene"][task_accepted])
                if (int(task_accepted.sum()) < config.min_accepted_queries
                        or len(accepted_scenes) < config.min_accepted_scenes):
                    satisfies_all_tasks = False
                    break
                task_benefit = float(np.maximum(actual[task_accepted], 0).sum())
                task_damage = float(np.maximum(-actual[task_accepted], 0).sum())
                task_ratio = task_damage / task_benefit if task_benefit else math.inf
                task_severe_rate = float(np.mean(actual[task_accepted] < -config.severe_harm_px))
                task_scene_gains = []
                for scene in task_scenes:
                    rows = task_rows & (selected["scene"] == scene)
                    scene_accepted = rows & accepted
                    key = f"{task}/{scene}"
                    scene_task_gains[key] = float(np.where(accepted[rows], actual[rows], 0).mean())
                    scene_task_rates[key] = (float(np.mean(actual[scene_accepted] < -config.severe_harm_px))
                                             if scene_accepted.any() else 0.0)
                    task_scene_gains.append(scene_task_gains[key])
                task_score = float(np.mean(task_scene_gains))
                if (task_score <= 0 or task_ratio > config.max_calibration_harm_benefit_ratio
                        or task_severe_rate > config.max_calibration_severe_harm_rate):
                    satisfies_all_tasks = False
                    break
                task_metrics[str(task)] = {"scene_macro_gain_px": task_score,
                                      "accepted_queries": int(task_accepted.sum()),
                                      "accepted_scenes": accepted_scenes.tolist(),
                                      "observed_harm_benefit_ratio": task_ratio,
                                      "observed_severe_harm_rate": task_severe_rate}
            if (not satisfies_all_tasks or min(scene_task_gains.values(), default=-math.inf) < config.min_scene_gain_px
                    or max(scene_task_rates.values(), default=math.inf) > config.max_calibration_severe_harm_rate):
                continue
            score = float(np.mean([metrics["scene_macro_gain_px"] for metrics in task_metrics.values()]))
            candidate = {**common, "state": "empirically_calibrated", "reason": "passed_observed_calibration_constraints",
                         "gain_threshold": float(gain_threshold), "harm_threshold": float(harm_threshold),
                         "accepted_queries": int(accepted.sum()), "scene_macro_gain_px": score,
                         "observed_severe_harm_rate": severe_rate, "observed_harm_benefit_ratio": ratio,
                         "task_metrics": task_metrics, "scene_task_gain_px": scene_task_gains,
                         "scene_task_severe_harm_rate": scene_task_rates}
            # Stable tie break: lower empirical harm, lower harm threshold,
            # then stricter gain threshold for identical scene-macro gain.
            key = (score, -severe_rate, -harm_threshold, gain_threshold)
            if best is None or key > best[0]:
                best = key, candidate
    return {**(best[1] if best else {**reject, "reason": "no_threshold_satisfies_empirical_constraints"}),
            "evaluated_threshold_pairs_with_minimum_support": considered,
            "threshold_grid_gain": thresholds.tolist(), "threshold_grid_harm": list(harm_thresholds)}


def _fit_indices(data, config, rng):
    fit = np.flatnonzero(data["split"] == "fit")
    if len(fit) <= config.max_fit_rows:
        return fit
    # Round-robin scene/task strata stop one dense scene or task from consuming
    # the fitting budget. Selection is independent of errors and gain labels.
    labels = np.char.add(data["scene"][fit], np.where(data["features"][fit, 10] > 0.5, "/stereo", "/flow"))
    strata = [rng.permutation(fit[labels == label]) for label in np.unique(labels)]
    quota = max(1, config.max_fit_rows // len(strata))
    selected = np.concatenate([rows[:quota] for rows in strata])[:config.max_fit_rows]
    if len(selected) < config.max_fit_rows:
        remainder = np.setdiff1d(fit, selected, assume_unique=True)
        selected = np.r_[selected, rng.permutation(remainder)[:config.max_fit_rows - len(selected)]]
    return np.sort(selected)


def train_acceptor(records: Mapping, output_path: str | Path | None = None,
                   config: TrainingConfig | None = None, *, dataset_provenance=None) -> CandidateAcceptor:
    """Train on fit only, calibrate on disjoint scenes, never score evaluation rows."""
    config = config or TrainingConfig()
    data = validate_records(records)
    start = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    fit = _fit_indices(data, config, rng)
    x = data["features"][fit]
    mean, std = x.mean(axis=0, dtype=np.float64), x.std(axis=0, dtype=np.float64)
    std = np.maximum(std, 1e-3)
    gains = data["gain"][fit]
    # Scaling changes conditioning, not the signed MSE target or its tails.
    gain_scale = max(1.0, float(np.std(gains)))
    target = torch.from_numpy(np.stack((gains / gain_scale, data["severe_harm"][fit]), axis=1).astype(np.float32))
    if not torch.isfinite(target).all():
        raise ValueError("Gain magnitude cannot be represented by the float32 training model")
    inputs = torch.from_numpy(((x - mean) / std).astype(np.float32))
    network = _UtilityNet(config.hidden_dim)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # Equal total weight per scene/task stratum, then per query and candidate.
    strata = np.char.add(data["scene"][fit], np.where(x[:, 10] > 0.5, "/stereo", "/flow"))
    weights = np.zeros(len(fit), np.float32)
    for label in np.unique(strata):
        rows = np.flatnonzero(strata == label)
        _, inverse, counts = np.unique(data["group"][fit[rows]], return_inverse=True, return_counts=True)
        weights[rows] = 1 / (len(counts) * counts[inverse])
    weights /= weights.mean()
    weights = torch.from_numpy(weights)
    history = []
    network.train()
    for epoch in range(config.epochs):
        permutation = rng.permutation(len(inputs))
        accumulated, mse_sum, bce_sum = 0.0, 0.0, 0.0
        for offset in range(0, len(inputs), config.batch_size):
            indices = torch.from_numpy(permutation[offset:offset + config.batch_size])
            prediction = network(inputs[indices])
            mse = (prediction[:, 0] - target[indices, 0]).square()
            bce = nn.functional.binary_cross_entropy_with_logits(prediction[:, 1], target[indices, 1], reduction="none")
            loss = ((mse + config.harm_loss_weight * bce) * weights[indices]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite utility training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = len(indices)
            accumulated += float(loss.detach()) * count
            mse_sum += float((mse * weights[indices]).mean().detach()) * count
            bce_sum += float((bce * weights[indices]).mean().detach()) * count
        history.append({"epoch": epoch + 1, "weighted_loss": accumulated / len(inputs),
                        "weighted_scaled_gain_mse": mse_sum / len(inputs), "weighted_harm_bce": bce_sum / len(inputs)})
    acceptor = CandidateAcceptor(network, mean, std, gain_scale, config)
    # Evaluation rows are not passed to predict or considered for thresholds.
    cal_rows = np.flatnonzero(data["split"] == "calibration")
    gain = np.zeros(len(data["features"]), np.float32)
    harm = np.ones(len(data["features"]), np.float32)
    if len(cal_rows):
        gain[cal_rows], harm[cal_rows] = acceptor.predict(data["features"][cal_rows])
    selected = select_queries(data, gain, harm, config)
    calibration = calibrate_policy(selected, config)
    acceptor.gain_threshold = math.inf if calibration["state"] == "reject_all" else calibration["gain_threshold"]
    acceptor.harm_threshold = calibration["harm_threshold"]
    splits = {}
    for split in ("fit", "calibration", "evaluation"):
        rows = data["split"] == split
        splits[split] = {"scenes": np.unique(data["scene"][rows]).tolist(), "rows": int(rows.sum()),
                         "queries": int(len(np.unique(data["query_id"][rows])))}
    acceptor.report = {"trained_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - start,
                       "feature_schema": FEATURE_SCHEMA, "normalization_source": "sampled_fit_rows_only",
                       "fit_rows_used": len(fit), "fit_rows_available": int((data["split"] == "fit").sum()),
                       "gain_target": "baseline_error_minus_candidate_error_signed_unclipped_native_pixels",
                       "gain_loss": "squared_error_scaled_by_fit_gain_std_no_target_clipping",
                       "harm_target": "candidate_error_minus_baseline_error_strictly_greater_than_1px",
                       "fit_gain_min": float(gains.min()), "fit_gain_max": float(gains.max()),
                       "fit_negative_gain_rows": int((gains < 0).sum()),
                       "fit_gain_below_minus20_rows": int((gains < -20).sum()),
                       "splits": splits, "scene_overlap": False,
                       "weights_selected_by": "fixed_epoch_count_no_calibration_model_selection",
                       "evaluation_rows_used_for_training_or_thresholds": 0,
                       "candidate_selection": "eligibility_then_argmax_with_identity_gain_zero_then_thresholds",
                       "calibration": calibration, "training_history": history,
                       "dataset_provenance": dataset_provenance or {},
                       "no_guarantee": "Calibration constrains observed scenes only; evaluate untouched scenes separately."}
    if output_path is not None:
        acceptor.save(output_path)
    return acceptor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-fit-rows", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--max-update-px", type=float, default=32.0)
    parser.add_argument("--allow-unsupported-baseline", action="store_true")
    parser.add_argument("--min-accepted-queries", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    dataset = Path(args.dataset).resolve()
    with np.load(dataset, allow_pickle=False) as archive:
        records = {name: archive[name] for name in archive.files}
    config = TrainingConfig(epochs=args.epochs, hidden_dim=args.hidden_dim, batch_size=args.batch_size,
                            max_fit_rows=args.max_fit_rows, seed=args.seed, max_update_px=args.max_update_px,
                            require_baseline_support=not args.allow_unsupported_baseline,
                            min_accepted_queries=args.min_accepted_queries)
    acceptor = train_acceptor(records, args.output, config,
                              dataset_provenance={"path": str(dataset), "sha256": _digest(dataset)})
    print(json.dumps({"output": str(Path(args.output).resolve()), "calibration": acceptor.report["calibration"],
                      "fit_rows_used": acceptor.report["fit_rows_used"]}, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
