#!/usr/bin/env python3
"""Train the primary FlowBridge router with dynamic grouped nested K-fold.

This is a narrow, parity-tested generalization of the frozen L1 trainer.  It
keeps the same factorized model, targets, calibration grid, and metrics while
removing the hard-coded 1,200-row/six-fold assumptions.  It intentionally omits
diagnostic classifier baselines that are irrelevant to final bank admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from roco_optical_flow.cli import train_flowbridge_l1 as frozen
from roco_optical_flow.controller import (
    derive_targets,
    fit_target_thresholds,
    select_factorized_routes,
    summarize_policy,
)


EXTERNAL_ROOT = Path(frozen.__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else EXTERNAL_ROOT / path


def _verified_source(manifest: dict, path_key: str, hash_key: str) -> Path:
    sources = manifest["sources"]
    path = _resolve(str(sources[path_key]))
    if not path.is_file() or _sha256(path) != sources[hash_key]:
        raise SystemExit(f"frozen source mismatch: {path_key}")
    return path


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:  # noqa: C901, PLR0915 - explicit evidence lifecycle
    args = _parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    records_output = args.records.expanduser().resolve()
    models_output = args.models.expanduser().resolve()
    if output.exists() or records_output.exists() or models_output.exists():
        raise SystemExit("immutable output, predictions, or model directory already exists")
    manifest = _load(manifest_path)
    if manifest.get("status") != "frozen_pre_training":
        raise SystemExit("protocol must be frozen_pre_training")

    feature_schema_path = _verified_source(manifest, "feature_schema", "feature_schema_sha256")
    feature_summary_path = _verified_source(manifest, "features", "features_sha256")
    teacher_summary_path = _verified_source(manifest, "teacher", "teacher_sha256")
    _verified_source(manifest, "sample_manifest", "sample_manifest_sha256")
    _verified_source(manifest, "diagnostic_atlas", "diagnostic_atlas_sha256")
    feature_schema = _load(feature_schema_path)
    feature_summary = _load(feature_summary_path)
    teacher_summary = _load(teacher_summary_path)
    feature_records_path = _resolve(str(feature_summary["records"]))
    teacher_records_path = _resolve(str(teacher_summary["records"]))
    sources = manifest["sources"]
    if _sha256(feature_records_path) != sources["feature_records_sha256"]:
        raise SystemExit("feature-record hash mismatch")
    if _sha256(teacher_records_path) != sources["teacher_records_sha256"]:
        raise SystemExit("teacher-record hash mismatch")
    feature_rows = _read_jsonl(feature_records_path)
    teacher_rows = _read_jsonl(teacher_records_path)
    if not feature_rows or len(feature_rows) != len(teacher_rows):
        raise SystemExit("feature and teacher populations differ")
    row_ids = [str(row["row_id"]) for row in feature_rows]
    if row_ids != [str(row["row_id"]) for row in teacher_rows]:
        raise SystemExit("feature and teacher row order differs")

    action_ids = [str(value) for value in manifest["action_ids"]]
    if len(action_ids) < 2 or len(set(action_ids)) != len(action_ids):
        raise SystemExit("action family must contain native and unique non-native controls")
    nonnative = len(action_ids) - 1
    feature_names = [str(value) for value in feature_summary["feature_names"]]
    raw_features = np.asarray([
        [float(row["model_visible_payload"]["features"][name]) for name in feature_names]
        for row in feature_rows
    ], dtype=np.float64)
    action_losses = np.asarray([
        [float(row["action_targets"][action]["raw_epe"]) for action in action_ids]
        for row in teacher_rows
    ], dtype=np.float64)
    folds = np.asarray([int(row["outer_fold"]) for row in teacher_rows], dtype=np.int64)
    scenes = np.asarray([str(row["scene"]) for row in feature_rows], dtype=object)
    conditions = np.asarray(
        [str(row["condition_provenance"]) for row in feature_rows], dtype=object
    )
    if not np.isfinite(raw_features).all() or not np.isfinite(action_losses).all():
        raise SystemExit("features or targets contain NaN/Inf")
    fold_ids = sorted(set(folds.tolist()))
    if len(fold_ids) < 3 or fold_ids != list(range(len(fold_ids))):
        raise SystemExit("outer folds must be contiguous integers with K >= 3")
    scene_fold = {}
    for scene, fold in zip(scenes.tolist(), folds.tolist()):
        previous = scene_fold.setdefault(scene, fold)
        if previous != fold:
            raise SystemExit(f"scene leakage across outer folds: {scene}")
    condition_names = sorted(set(conditions.tolist()))
    if "clean" not in condition_names or len(condition_names) < 2:
        raise SystemExit("panel must contain clean and at least one corruption")

    group_features = {
        str(bundle["cue_id"]): [str(name) for name in bundle["feature_names"]]
        for bundle in feature_rows[0]["evidence_bundles"]
    }
    full_groups = feature_schema["ablation_sets"]["full_scalar"]
    selected_names = sorted(name for group in full_groups for name in group_features[group])
    if selected_names != feature_names:
        raise SystemExit("full_scalar feature membership differs from feature artifact")
    columns = np.arange(len(feature_names), dtype=np.int64)

    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = manifest["factorized_model"]
    seeds = [int(value) for value in config["ensemble_seeds"]]
    calibration_policy = manifest["policy_calibration"]
    utility_grid = [float(value) for value in calibration_policy["utility_min_grid"]]
    harm_grid = [float(value) for value in calibration_policy["max_harm_probability_grid"]]
    catastrophe_grid = [
        float(value) for value in calibration_policy["max_catastrophe_probability_grid"]
    ]
    feasibility = calibration_policy["primary_feasibility"]

    selections = {
        "CTRL-P0": np.zeros(len(feature_rows), dtype=np.int64),
        "CTRL-FACT": np.zeros(len(feature_rows), dtype=np.int64),
        "CTRL-ORACLE": np.argmin(action_losses, axis=1),
    }
    outputs = {
        name: np.full((len(feature_rows), nonnative), np.nan, dtype=np.float32)
        for name in ("utility", "harm", "catastrophe")
    }
    outer_catastrophe = np.zeros((len(feature_rows), nonnative), dtype=bool)
    outer_runs = []
    model_temp = models_output.with_name(f".{models_output.name}.tmp-{os.getpid()}")
    model_temp.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    for outer_fold in fold_ids:
        outer_indices = np.flatnonzero(folds == outer_fold)
        inner_indices = np.flatnonzero(folds != outer_fold)
        inner_fold_ids = [fold for fold in fold_ids if fold != outer_fold]
        inner_oof = {
            name: np.full((len(feature_rows), nonnative), np.nan, dtype=np.float32)
            for name in ("utility", "harm", "catastrophe")
        }
        inner_catastrophe = np.zeros((len(feature_rows), nonnative), dtype=bool)
        nested = []
        for validation_fold in inner_fold_ids:
            validation = np.flatnonzero(folds == validation_fold)
            training = np.flatnonzero((folds != outer_fold) & (folds != validation_fold))
            if not training.size or not validation.size:
                raise RuntimeError("empty nested fold")
            thresholds = fit_target_thresholds(action_losses[training])
            inner_catastrophe[validation] = derive_targets(
                action_losses[validation], thresholds
            )[2].astype(bool)
            prediction, _, _, _, losses = frozen._train_factorized_ensemble(
                torch,
                raw_features,
                action_losses,
                training,
                validation,
                columns,
                config,
                seeds,
            )
            for name in prediction:
                inner_oof[name][validation] = prediction[name]
            nested.append({
                "validation_fold": validation_fold,
                "training_rows": int(training.size),
                "validation_rows": int(validation.size),
                "thresholds": thresholds,
                "final_training_losses": losses,
            })
        if any(not np.isfinite(value[inner_indices]).all() for value in inner_oof.values()):
            raise RuntimeError(f"incomplete inner OOF predictions for outer fold {outer_fold}")

        builders = [
            (
                {
                    "utility_min": utility_min,
                    "max_harm_probability": harm_max,
                    "max_catastrophe_probability": catastrophe_max,
                },
                lambda um=utility_min, hm=harm_max, cm=catastrophe_max: select_factorized_routes(
                    inner_oof["utility"][inner_indices],
                    inner_oof["harm"][inner_indices],
                    inner_oof["catastrophe"][inner_indices],
                    utility_min=um,
                    max_harm_probability=hm,
                    max_catastrophe_probability=cm,
                ),
            )
            for utility_min in utility_grid
            for harm_max in harm_grid
            for catastrophe_max in catastrophe_grid
        ]
        calibrated, calibration = frozen._calibrate(
            builders,
            action_losses[inner_indices],
            conditions[inner_indices].tolist(),
            scenes[inner_indices].tolist(),
            inner_catastrophe[inner_indices],
            action_ids,
            feasibility,
        )
        final_thresholds = fit_target_thresholds(action_losses[inner_indices])
        outer_catastrophe[outer_indices] = derive_targets(
            action_losses[outer_indices], final_thresholds
        )[2].astype(bool)
        prediction, _, models, normalizer, final_losses = frozen._train_factorized_ensemble(
            torch,
            raw_features,
            action_losses,
            inner_indices,
            outer_indices,
            columns,
            config,
            seeds,
            return_models=True,
        )
        for name in prediction:
            outputs[name][outer_indices] = prediction[name]
        selections["CTRL-FACT"][outer_indices] = frozen._apply_calibrated(
            calibrated,
            lambda cfg: select_factorized_routes(
                prediction["utility"],
                prediction["harm"],
                prediction["catastrophe"],
                utility_min=float(cfg["utility_min"]),
                max_harm_probability=float(cfg["max_harm_probability"]),
                max_catastrophe_probability=float(cfg["max_catastrophe_probability"]),
            ),
            outer_indices.size,
        )
        filename = f"outer{outer_fold}-factorized.npz"
        model_path = model_temp / filename
        frozen._save_model_bundle(
            model_path,
            models,
            normalizer,
            feature_names,
            {"thresholds": final_thresholds, "policy": calibrated},
        )
        outer_runs.append({
            "outer_fold": outer_fold,
            "inner_rows": int(inner_indices.size),
            "outer_rows": int(outer_indices.size),
            "inner_scene_count": len(set(scenes[inner_indices].tolist())),
            "outer_scene_count": len(set(scenes[outer_indices].tolist())),
            "target_thresholds": final_thresholds,
            "calibrated_config": calibrated,
            "calibration": calibration,
            "nested_training": nested,
            "final_training_losses": final_losses,
            "model": {
                "path": str(models_output / filename),
                "sha256": _sha256(model_path),
                "ensemble_size": len(models),
            },
        })
        print(json.dumps({
            "status": "running",
            "outer_fold": outer_fold,
            "complete_outer_folds": len(outer_runs),
            "outer_fold_count": len(fold_ids),
            "primary_config": calibrated,
        }, sort_keys=True), flush=True)

    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise RuntimeError("incomplete outer predictions")
    summaries = {
        name: summarize_policy(
            selection,
            action_losses,
            conditions.tolist(),
            scenes.tolist(),
            outer_catastrophe,
            action_ids,
        )
        for name, selection in selections.items()
    }
    p0_corrupt = float(summaries["CTRL-P0"]["policy_corrupt_macro_epe"])
    oracle_corrupt = float(summaries["CTRL-ORACLE"]["policy_corrupt_macro_epe"])
    for summary in summaries.values():
        current = float(summary["policy_corrupt_macro_epe"])
        denominator = p0_corrupt - oracle_corrupt
        summary["oracle_gap_capture"] = (
            (p0_corrupt - current) / denominator if denominator > 0.0 else None
        )
    bootstrap = frozen._bootstrap_intervals(
        selections,
        action_losses,
        conditions,
        scenes,
        selections["CTRL-ORACLE"],
        replicates=5000,
        seed=20260905,
    )

    records_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_records = records_output.with_name(f".{records_output.name}.tmp-{os.getpid()}")
    with temporary_records.open("x", encoding="utf-8", buffering=1) as handle:
        for index, (feature, teacher) in enumerate(zip(feature_rows, teacher_rows)):
            selected = int(selections["CTRL-FACT"][index])
            gain = float(action_losses[index, 0] - action_losses[index, selected])
            record = {
                "schema": "dynamic-group-router-prediction/v1",
                "row_id": feature["row_id"],
                "canonical_scene_key": feature["canonical_scene_key"],
                "scene": feature["scene"],
                "condition_provenance": feature["condition_provenance"],
                "condition_label_model_visible": False,
                "outer_fold": int(teacher["outer_fold"]),
                "target_access_for_controller": "none",
                "target_access_for_evaluation": "analysis_gt",
                "primary_model_outputs": {
                    name: outputs[name][index].astype(float).tolist() for name in outputs
                },
                "policies": {
                    "CTRL-FACT": {
                        "action_id": action_ids[selected],
                        "action_index": selected,
                        "selected_epe": float(action_losses[index, selected]),
                        "raw_gain": gain,
                        "harm": gain < 0.0,
                        "catastrophe": bool(
                            selected > 0 and outer_catastrophe[index, selected - 1]
                        ),
                    }
                },
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary_records.replace(records_output)
    model_temp.replace(models_output)
    primary = summaries["CTRL-FACT"]
    result = {
        "schema": "dynamic-group-router-result/v1",
        "status": "complete",
        "diagnostic_only": True,
        "controller_id": manifest["controller_id"],
        "claim_scope": manifest["claim_scope"],
        "target_access_for_controller": "none",
        "condition_label_model_visible": False,
        "row_count": len(feature_rows),
        "scene_count": len(scene_fold),
        "outer_fold_count": len(fold_ids),
        "inner_fold_count_per_outer": len(fold_ids) - 1,
        "action_ids": action_ids,
        "variant_summaries": summaries,
        "outer_runs": outer_runs,
        "bootstrap": bootstrap,
        "records": str(records_output),
        "records_sha256": _sha256(records_output),
        "models": str(models_output),
        "model_file_count": len(list(models_output.glob("*.npz"))),
        "sources": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "features": str(feature_summary_path),
            "features_sha256": _sha256(feature_summary_path),
            "teacher": str(teacher_summary_path),
            "teacher_sha256": _sha256(teacher_summary_path),
            "implementation": str(Path(__file__).resolve()),
            "implementation_sha256": _sha256(Path(__file__).resolve()),
            "frozen_trainer_dependency": str(Path(frozen.__file__).resolve()),
            "frozen_trainer_dependency_sha256": _sha256(Path(frozen.__file__).resolve()),
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "device": "cpu",
            "gpu_used": False,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "primary": {
            "corrupt_relative_gain_vs_p0": primary["corrupt_relative_gain_vs_p0"],
            "clean_relative_degradation": primary["clean_relative_degradation"],
            "catastrophe_per_intervention": primary["catastrophe_per_intervention"],
            "corrupt_conditions_improved": primary["corrupt_conditions_improved"],
        },
    }
    _atomic_json(output, result)
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "records": str(records_output),
        "primary": result["primary"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
