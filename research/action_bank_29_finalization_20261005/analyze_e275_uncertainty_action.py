#!/usr/bin/env python3
"""Opened-development E275 native-uncertainty/action-outcome diagnostic.

The analyzer joins the hash-bound E275 teacher rows to the frozen, target-free
feature rows.  It never treats condition labels, scene ids, fold ids, paths, or
outcomes as model-visible inputs.  Results are descriptive development evidence
only: E275 has already been opened and reused to reduce the candidate bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "experiments/E275_reduced_bank_observable_routing_v1/PROTOCOL.json"
DEFAULT_OUTPUT = HERE / "E275_UNCERTAINTY_ACTION_DIAGNOSTIC.json"
ALLOWED_EXTERNAL_ROOT = Path("/ssd7").resolve()
FORBIDDEN_ROOTS = (Path("/tmp").resolve(), Path("/ssd8").resolve())
SEVERE_HARM_CUTOFF_RAW_PX = -0.25
UNCERTAINTY_BUNDLE_ID = "B0_native"


@dataclass(frozen=True)
class DiagnosticInputs:
    protocol_path: Path
    teacher_summary_path: Path
    teacher_records_path: Path
    feature_summary_path: Path
    feature_records_path: Path
    feature_schema_path: Path
    action_ids: tuple[str, ...]
    family_by_action: Mapping[str, str]
    uncertainty_names: tuple[str, ...]
    row_ids: tuple[str, ...]
    scenes: np.ndarray
    uncertainty: np.ndarray
    gains: np.ndarray
    source_bindings: Mapping[str, Mapping[str, object]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the immutable JSON report; otherwise only print a summary",
    )
    return parser


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _checked_path(path: Path, *, role: str, output: bool = False) -> Path:
    # Reject forbidden lexical roots before any realpath/symlink traversal.
    # This analyzer must not even consult those filesystems.
    expanded = os.path.expanduser(str(path))
    lexical = Path(os.path.abspath(os.path.normpath(expanded)))
    if any(_is_within(lexical, root) for root in FORBIDDEN_ROOTS):
        raise ValueError(f"{role} may not use /tmp or /ssd8: {lexical}")
    if output:
        if not _is_within(lexical, ROOT):
            raise ValueError(f"{role} must remain inside the repository: {lexical}")
    elif not (
        _is_within(lexical, ROOT) or _is_within(lexical, ALLOWED_EXTERNAL_ROOT)
    ):
        raise ValueError(f"{role} must be inside the repository or /ssd7: {lexical}")
    return lexical


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"expected JSON objects in {path}")
    return rows


def _source_path(value: object, *, relative_to: Path, role: str) -> Path:
    raw = Path(str(value))
    return _checked_path(raw if raw.is_absolute() else relative_to / raw, role=role)


def _verify_hash(path: Path, expected: object, *, role: str) -> None:
    actual = _sha256(path)
    if actual != str(expected):
        raise ValueError(f"{role} SHA-256 mismatch: {actual} != {expected}")


def load_inputs(protocol_path: Path = DEFAULT_PROTOCOL) -> DiagnosticInputs:
    """Load and validate the exact hash-bound E275 analysis population."""

    protocol_path = _checked_path(protocol_path, role="protocol")
    protocol = _load_json(protocol_path)
    if protocol.get("status") != "frozen_pre_training":
        raise ValueError("E275 protocol is not frozen_pre_training")
    if protocol.get("diagnostic_only") is not True:
        raise ValueError("E275 protocol must remain diagnostic-only")
    exposure = protocol.get("exposure_boundary", {})
    if not isinstance(exposure, dict) or exposure.get("opened_development") is not True:
        raise ValueError("E275 exposure boundary is not opened development")

    sources = protocol.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("protocol sources are missing")
    teacher_summary_path = _source_path(
        sources["teacher"], relative_to=ROOT, role="teacher summary"
    )
    feature_summary_path = _source_path(
        sources["features"], relative_to=ROOT, role="feature summary"
    )
    feature_schema_path = _source_path(
        sources["feature_schema"], relative_to=ROOT, role="feature schema"
    )
    _verify_hash(teacher_summary_path, sources["teacher_sha256"], role="teacher summary")
    _verify_hash(feature_summary_path, sources["features_sha256"], role="feature summary")
    _verify_hash(feature_schema_path, sources["feature_schema_sha256"], role="feature schema")

    teacher_summary = _load_json(teacher_summary_path)
    feature_summary = _load_json(feature_summary_path)
    feature_schema = _load_json(feature_schema_path)
    teacher_records_path = _source_path(
        teacher_summary["records"],
        relative_to=teacher_summary_path.parent,
        role="teacher records",
    )
    feature_records_path = _source_path(
        feature_summary["records"],
        relative_to=feature_summary_path.parent,
        role="feature records",
    )
    _verify_hash(
        teacher_records_path,
        sources["teacher_records_sha256"],
        role="teacher records",
    )
    if "records_sha256" in teacher_summary:
        _verify_hash(
            teacher_records_path,
            teacher_summary["records_sha256"],
            role="teacher records summary binding",
        )
    _verify_hash(
        feature_records_path,
        sources["feature_records_sha256"],
        role="feature records",
    )
    if "records_sha256" in feature_summary:
        _verify_hash(
            feature_records_path,
            feature_summary["records_sha256"],
            role="feature records summary binding",
        )

    teacher_rows = _load_jsonl(teacher_records_path)
    feature_rows = _load_jsonl(feature_records_path)
    expected_count = int(teacher_summary["record_count"])
    if expected_count != 1_200 or len(teacher_rows) != expected_count:
        raise ValueError("E275 teacher population must contain exactly 1,200 rows")
    if len(feature_rows) != expected_count:
        raise ValueError("feature and teacher populations differ")
    teacher_ids = tuple(str(row["row_id"]) for row in teacher_rows)
    feature_ids = tuple(str(row["row_id"]) for row in feature_rows)
    if teacher_ids != feature_ids or len(set(teacher_ids)) != expected_count:
        raise ValueError("feature/teacher row identity or order drift")

    action_ids = tuple(str(value) for value in protocol["action_ids"])
    if tuple(str(value) for value in teacher_summary["action_ids"]) != action_ids:
        raise ValueError("protocol and teacher action order differ")
    if len(action_ids) != 11 or len(set(action_ids)) != len(action_ids):
        raise ValueError("E275 must contain native plus ten unique exact controls")
    native = action_ids[0]
    if not native.endswith("/P0"):
        raise ValueError("first E275 action is not native P0")

    family_by_action: dict[str, str] = {}
    for family in protocol["action_families"]:
        family_id = str(family["family_id"])
        for action_id in family["exact_controls"]:
            action = str(action_id)
            if action in family_by_action:
                raise ValueError(f"exact control appears in multiple families: {action}")
            family_by_action[action] = family_id
    if set(family_by_action) != set(action_ids[1:]):
        raise ValueError("family mapping does not cover the exact nonnative bank")

    feature_names = tuple(str(value) for value in feature_summary["feature_names"])
    if int(feature_summary["feature_count"]) != len(feature_names):
        raise ValueError("feature summary cardinality drift")
    bundles = {
        str(bundle["cue_id"]): tuple(str(name) for name in bundle["feature_names"])
        for bundle in feature_rows[0]["evidence_bundles"]
    }
    uncertainty_names = bundles.get(UNCERTAINTY_BUNDLE_ID, ())
    if len(uncertainty_names) != 14 or not set(uncertainty_names).issubset(feature_names):
        raise ValueError("B0_native uncertainty bundle must contain exactly 14 features")
    schema_groups = {
        str(group["id"]): group for group in feature_schema.get("feature_groups", [])
    }
    if UNCERTAINTY_BUNDLE_ID not in schema_groups:
        raise ValueError("feature schema does not define B0_native")
    if schema_groups[UNCERTAINTY_BUNDLE_ID].get("target_access") != "none":
        raise ValueError("B0_native is not target-free")

    uncertainty = np.asarray(
        [
            [float(row["model_visible_payload"]["features"][name]) for name in uncertainty_names]
            for row in feature_rows
        ],
        dtype=np.float64,
    )
    losses = np.asarray(
        [
            [float(row["action_targets"][action]["raw_epe"]) for action in action_ids]
            for row in teacher_rows
        ],
        dtype=np.float64,
    )
    scenes = np.asarray([str(row["scene"]) for row in teacher_rows], dtype=object)
    feature_scenes = np.asarray([str(row["scene"]) for row in feature_rows], dtype=object)
    if not np.array_equal(scenes, feature_scenes):
        raise ValueError("feature/teacher scene binding drift")
    if not np.isfinite(uncertainty).all() or not np.isfinite(losses).all():
        raise ValueError("E275 diagnostic inputs contain NaN or Inf")
    if np.any(losses < 0.0):
        raise ValueError("E275 raw EPE must be nonnegative")
    unique_scenes, counts = np.unique(scenes, return_counts=True)
    if unique_scenes.size != 30 or set(counts.tolist()) != {40}:
        raise ValueError("E275 must contain 30 equally weighted 40-row scenes")
    gains = losses[:, [0]] - losses[:, 1:]

    return DiagnosticInputs(
        protocol_path=protocol_path,
        teacher_summary_path=teacher_summary_path,
        teacher_records_path=teacher_records_path,
        feature_summary_path=feature_summary_path,
        feature_records_path=feature_records_path,
        feature_schema_path=feature_schema_path,
        action_ids=action_ids,
        family_by_action=family_by_action,
        uncertainty_names=uncertainty_names,
        row_ids=teacher_ids,
        scenes=scenes,
        uncertainty=uncertainty,
        gains=gains,
        source_bindings={
            "implementation": _binding(Path(__file__).resolve()),
            "protocol": _binding(protocol_path),
            "teacher_summary": _binding(teacher_summary_path),
            "teacher_records": _binding(teacher_records_path),
            "feature_summary": _binding(feature_summary_path),
            "feature_records": _binding(feature_records_path),
            "feature_schema": _binding(feature_schema_path),
        },
    )


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks with deterministic tie handling."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("rankdata requires one finite vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _rank_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("rank matrix must be finite and two-dimensional")
    return np.column_stack([rankdata_average(array[:, index]) for index in range(array.shape[1])])


def _correlation_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    numerator = left_centered.T @ right_centered
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=0)[:, None]
        * np.sum(right_centered * right_centered, axis=0)[None, :]
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUROC; high scores are the positive orientation."""

    score = np.asarray(scores, dtype=np.float64)
    label = np.asarray(labels, dtype=bool)
    if score.ndim != 1 or label.shape != score.shape or not np.isfinite(score).all():
        raise ValueError("AUROC inputs must be matching finite vectors")
    positives = int(label.sum())
    negatives = int(label.size - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata_average(score)
    statistic = float(ranks[label].sum() - positives * (positives - 1) / 2.0)
    return statistic / (positives * negatives)


def _auroc_matrix(score_ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    feature_count = score_ranks.shape[1]
    action_count = labels.shape[1]
    result = np.full((feature_count, action_count), np.nan, dtype=np.float64)
    for action_index in range(action_count):
        positive = labels[:, action_index]
        n_positive = int(positive.sum())
        n_negative = int(positive.size - n_positive)
        if n_positive == 0 or n_negative == 0:
            continue
        statistic = (
            score_ranks[positive].sum(axis=0)
            - n_positive * (n_positive - 1) / 2.0
        )
        result[:, action_index] = statistic / (n_positive * n_negative)
    return result


def metric_matrices(
    uncertainty: np.ndarray,
    gains: np.ndarray,
    *,
    severe_harm_cutoff_raw_px: float,
) -> dict[str, np.ndarray]:
    if uncertainty.ndim != 2 or gains.ndim != 2 or uncertainty.shape[0] != gains.shape[0]:
        raise ValueError("uncertainty and gain matrices must have matching rows")
    if not np.isfinite(uncertainty).all() or not np.isfinite(gains).all():
        raise ValueError("diagnostic matrices must be finite")
    if not math.isfinite(severe_harm_cutoff_raw_px) or severe_harm_cutoff_raw_px >= 0.0:
        raise ValueError("severe-harm cutoff must be finite and negative")
    uncertainty_ranks = _rank_matrix(uncertainty)
    gain_ranks = _rank_matrix(gains)
    harm_ranks = _rank_matrix(np.maximum(-gains, 0.0))
    return {
        "spearman_signed_gain": _correlation_matrix(uncertainty_ranks, gain_ranks),
        "spearman_harm_magnitude": _correlation_matrix(uncertainty_ranks, harm_ranks),
        "auroc_benefit": _auroc_matrix(uncertainty_ranks, gains > 0.0),
        "auroc_severe_harm": _auroc_matrix(
            uncertainty_ranks, gains < severe_harm_cutoff_raw_px
        ),
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _interval(values: np.ndarray, draws: int) -> dict[str, object]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "scene_bootstrap_95ci": (
            [float(value) for value in np.percentile(finite, [2.5, 97.5])]
            if finite.size
            else None
        ),
        "valid_bootstrap_draws": int(finite.size),
        "requested_bootstrap_draws": draws,
    }


def build_report(
    inputs: DiagnosticInputs,
    *,
    bootstrap_draws: int = 2_000,
    seed: int = 20261005,
    severe_harm_cutoff_raw_px: float = SEVERE_HARM_CUTOFF_RAW_PX,
) -> dict[str, object]:
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int) or bootstrap_draws < 1:
        raise ValueError("bootstrap draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    point = metric_matrices(
        inputs.uncertainty,
        inputs.gains,
        severe_harm_cutoff_raw_px=severe_harm_cutoff_raw_px,
    )
    metric_names = tuple(point)
    feature_count = inputs.uncertainty.shape[1]
    action_count = inputs.gains.shape[1]
    samples = {
        name: np.full(
            (bootstrap_draws, feature_count, action_count), np.nan, dtype=np.float64
        )
        for name in metric_names
    }
    outcome_names = (
        "mean_signed_gain_raw_px",
        "benefit_rate",
        "harm_rate",
        "severe_harm_rate",
    )
    outcome_samples = np.full(
        (bootstrap_draws, len(outcome_names), action_count), np.nan, dtype=np.float64
    )
    unique_scenes = np.asarray(sorted(set(inputs.scenes.tolist())), dtype=object)
    indices_by_scene = {
        scene: np.flatnonzero(inputs.scenes == scene) for scene in unique_scenes
    }
    rng = np.random.default_rng(seed)
    for draw_index in range(bootstrap_draws):
        chosen = rng.integers(0, unique_scenes.size, size=unique_scenes.size)
        row_indices = np.concatenate(
            [indices_by_scene[unique_scenes[index]] for index in chosen]
        )
        draw_gains = inputs.gains[row_indices]
        draw_metrics = metric_matrices(
            inputs.uncertainty[row_indices],
            draw_gains,
            severe_harm_cutoff_raw_px=severe_harm_cutoff_raw_px,
        )
        for name, values in draw_metrics.items():
            samples[name][draw_index] = values
        outcome_samples[draw_index, 0] = draw_gains.mean(axis=0)
        outcome_samples[draw_index, 1] = (draw_gains > 0.0).mean(axis=0)
        outcome_samples[draw_index, 2] = (draw_gains < 0.0).mean(axis=0)
        outcome_samples[draw_index, 3] = (
            draw_gains < severe_harm_cutoff_raw_px
        ).mean(axis=0)

    actions: dict[str, object] = {}
    for action_index, action_id in enumerate(inputs.action_ids[1:]):
        gains = inputs.gains[:, action_index]
        outcome_estimates = {
            "mean_signed_gain_raw_px": float(gains.mean()),
            "benefit_rate": float((gains > 0.0).mean()),
            "harm_rate": float((gains < 0.0).mean()),
            "severe_harm_rate": float((gains < severe_harm_cutoff_raw_px).mean()),
        }
        outcomes = {}
        for outcome_index, name in enumerate(outcome_names):
            outcomes[name] = {
                "estimate": outcome_estimates[name],
                **_interval(
                    outcome_samples[:, outcome_index, action_index], bootstrap_draws
                ),
            }
        feature_metrics: dict[str, object] = {}
        for feature_index, feature_name in enumerate(inputs.uncertainty_names):
            feature_metrics[feature_name] = {
                name: {
                    "estimate": _finite_or_none(values[feature_index, action_index]),
                    **_interval(
                        samples[name][:, feature_index, action_index], bootstrap_draws
                    ),
                }
                for name, values in point.items()
            }
        actions[action_id] = {
            "family_id": inputs.family_by_action[action_id],
            "outcome_counts": {
                "rows": int(gains.size),
                "beneficial": int((gains > 0.0).sum()),
                "harmful": int((gains < 0.0).sum()),
                "zero_gain": int((gains == 0.0).sum()),
                "severe_harm": int((gains < severe_harm_cutoff_raw_px).sum()),
            },
            "outcomes": outcomes,
            "uncertainty_features": feature_metrics,
        }

    return {
        "schema": "e275-uncertainty-action-diagnostic/v1",
        "status": "COMPLETE_OPENED_DEVELOPMENT_DIAGNOSTIC",
        "development_only": True,
        "authority": {
            "final_action_bank": False,
            "scientific_qualification": False,
            "selector_admission": False,
            "calibration": False,
            "safety": False,
            "production": False,
        },
        "claim_boundary": (
            "Descriptive, scene-clustered analysis of adaptively reused opened E3/E275 "
            "scalar uncertainty and exact-control raw-EPE outcomes. It is not fresh "
            "validation, post-selection calibration, a pixel-tail analysis, or a final-bank claim."
        ),
        "population": {
            "rows": len(inputs.row_ids),
            "physical_scenes": int(unique_scenes.size),
            "rows_per_scene": 40,
            "exact_controls": action_count,
            "uncertainty_features": feature_count,
        },
        "targets": {
            "signed_gain_raw_px": "native_raw_epe - action_raw_epe",
            "benefit": "signed_gain_raw_px > 0",
            "harm_magnitude_raw_px": "max(-signed_gain_raw_px, 0)",
            "severe_harm": f"signed_gain_raw_px < {severe_harm_cutoff_raw_px}",
            "pixel_harm_targets_available": False,
        },
        "method": {
            "uncertainty_bundle": UNCERTAINTY_BUNDLE_ID,
            "uncertainty_source": "target-free native SEA-RAFT endpoint_rms and mixture_entropy scalars",
            "rank_association": "Spearman with average ranks for ties",
            "auroc": "Mann-Whitney AUROC; higher uncertainty is the positive-score orientation",
            "bootstrap_unit": "physical_scene",
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": seed,
            "bootstrap_interval": "percentile 2.5/97.5",
            "condition_labels_model_visible": False,
            "multiplicity_adjusted": False,
        },
        "uncertainty_feature_names": list(inputs.uncertainty_names),
        "actions": actions,
        "sources": dict(inputs.source_bindings),
    }


def _atomic_json(path: Path, value: object) -> None:
    output = _checked_path(path, role="output", output=True)
    if output.exists():
        raise FileExistsError(f"immutable diagnostic output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent does not exist: {output.parent}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = load_inputs(args.protocol)
    report = build_report(
        inputs,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        severe_harm_cutoff_raw_px=SEVERE_HARM_CUTOFF_RAW_PX,
    )
    if args.write:
        _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "development_only": report["development_only"],
                "rows": report["population"]["rows"],
                "physical_scenes": report["population"]["physical_scenes"],
                "exact_controls": report["population"]["exact_controls"],
                "uncertainty_features": report["population"]["uncertainty_features"],
                "bootstrap_draws": args.bootstrap_draws,
                "output": str(args.output.resolve()) if args.write else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
