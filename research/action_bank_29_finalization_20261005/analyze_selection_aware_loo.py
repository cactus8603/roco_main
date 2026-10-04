#!/usr/bin/env python3
"""Paired scene-cluster analysis of E278 versus four retrained LOO banks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("E279_SELECTION_AWARE_LOO_RESULT.json")
FULL = ROOT / "experiments/E278_minimal_stable_observable_bank_v1"
LOO = {
    "CSB/OF/SEA-RAFT/action/P1-gaussian-s1": ROOT / "experiments/E279a_selection_aware_loo_p1_v1",
    "CSB/OF/SEA-RAFT/action/R2-gaussian-s2": ROOT / "experiments/E279b_selection_aware_loo_r2_v1",
    "CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5": ROOT / "experiments/E279c_selection_aware_loo_r4_v1",
    "CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1": ROOT / "experiments/E279d_selection_aware_loo_p3_v1",
}
CONDITIONS = (
    "brightness-s2",
    "gaussian-blur-s2",
    "gaussian-noise-s2",
    "jpeg-s2",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _scene_matrix(rows: list[dict], values: np.ndarray, conditions: tuple[str, ...]) -> np.ndarray:
    scenes = sorted({str(row["scene"]) for row in rows})
    matrix = np.empty((len(scenes), len(conditions)), dtype=np.float64)
    for i, scene in enumerate(scenes):
        for j, condition in enumerate(conditions):
            indices = [
                n for n, row in enumerate(rows)
                if str(row["scene"]) == scene
                and str(row["condition_provenance"]) == condition
            ]
            if not indices:
                raise ValueError(f"missing scene/condition cell: {scene}/{condition}")
            matrix[i, j] = float(np.mean(values[indices]))
    return matrix


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    full_path = FULL / "PREDICTIONS.jsonl"
    full_rows = _read(full_path)
    row_ids = [row["row_id"] for row in full_rows]
    scenes = sorted({str(row["scene"]) for row in full_rows})
    if len(full_rows) != 1200 or len(scenes) != 30:
        raise ValueError("unexpected full-bank population")
    full_loss = np.asarray(
        [row["policies"]["CTRL-FACT"]["selected_epe"] for row in full_rows],
        dtype=np.float64,
    )
    native_loss = np.asarray(
        [row["policies"]["CTRL-P0"]["selected_epe"] for row in full_rows],
        dtype=np.float64,
    )
    full_corrupt = _scene_matrix(full_rows, full_loss, CONDITIONS).mean(axis=1)
    native_corrupt = _scene_matrix(full_rows, native_loss, CONDITIONS).mean(axis=1)
    full_clean = _scene_matrix(full_rows, full_loss, ("clean",))[:, 0]
    native_clean = _scene_matrix(full_rows, native_loss, ("clean",))[:, 0]
    teacher_summary = json.loads((FULL / "TEACHER.json").read_text(encoding="utf-8"))
    teacher_rows = _read(Path(teacher_summary["records"]))
    if [row["row_id"] for row in teacher_rows] != row_ids:
        raise ValueError("teacher/prediction row alignment drift")
    action_ids = list(teacher_summary["action_ids"])
    action_losses = np.asarray(
        [
            [row["action_targets"][action]["raw_epe"] for action in action_ids]
            for row in teacher_rows
        ],
        dtype=np.float64,
    )
    full_oracle_loss = action_losses.min(axis=1)
    full_oracle_corrupt = _scene_matrix(
        full_rows, full_oracle_loss, CONDITIONS
    ).mean(axis=1)

    seed = 20261005
    draws = 10_000
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(scenes), size=(draws, len(scenes)))
    comparisons = {}
    corrupt_bootstraps = []
    oracle_bootstraps = []
    for action_id, directory in LOO.items():
        path = directory / "PREDICTIONS.jsonl"
        rows = _read(path)
        if [row["row_id"] for row in rows] != row_ids:
            raise ValueError(f"row alignment drift: {directory}")
        loss = np.asarray(
            [row["policies"]["CTRL-FACT"]["selected_epe"] for row in rows],
            dtype=np.float64,
        )
        loo_corrupt = _scene_matrix(rows, loss, CONDITIONS).mean(axis=1)
        loo_clean = _scene_matrix(rows, loss, ("clean",))[:, 0]
        corrupt_delta = loo_corrupt - full_corrupt
        clean_delta = loo_clean - full_clean
        corrupt_boot = corrupt_delta[sampled].mean(axis=1)
        clean_boot = clean_delta[sampled].mean(axis=1)
        corrupt_bootstraps.append(corrupt_boot)
        excluded_index = action_ids.index(action_id)
        retained_indices = [
            index for index in range(len(action_ids)) if index != excluded_index
        ]
        loo_oracle_loss = action_losses[:, retained_indices].min(axis=1)
        loo_oracle_corrupt = _scene_matrix(
            full_rows, loo_oracle_loss, CONDITIONS
        ).mean(axis=1)
        oracle_delta = loo_oracle_corrupt - full_oracle_corrupt
        oracle_boot = oracle_delta[sampled].mean(axis=1)
        oracle_bootstraps.append(oracle_boot)
        strict_winner = (
            (action_losses[:, excluded_index] < action_losses[:, retained_indices].min(axis=1))
            & (action_losses[:, excluded_index] < action_losses[:, 0])
        )
        per_condition = {}
        for condition in CONDITIONS:
            full_cell = _scene_matrix(full_rows, full_loss, (condition,))[:, 0]
            loo_cell = _scene_matrix(rows, loss, (condition,))[:, 0]
            delta = loo_cell - full_cell
            boot = delta[sampled].mean(axis=1)
            per_condition[condition] = {
                "loo_minus_full_mean_epe_px": float(delta.mean()),
                "scene_bootstrap_95ci_px": [
                    float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5)),
                ],
            }
        result = json.loads((directory / "RESULT.json").read_text(encoding="utf-8"))
        summary = result["variant_summaries"]["CTRL-FACT"]
        corrupt_mean = float(corrupt_delta.mean())
        corrupt_sd = float(corrupt_delta.std(ddof=1))
        planning_n = None
        if corrupt_mean > 0.0:
            z_alpha = NormalDist().inv_cdf(1.0 - 0.05 / len(LOO))
            z_power = NormalDist().inv_cdf(0.80)
            planning_n = int(np.ceil(((z_alpha + z_power) * corrupt_sd / corrupt_mean) ** 2))
        comparisons[action_id] = {
            "loo_experiment": directory.name,
            "loo_predictions_sha256": _sha256(path),
            "loo_minus_full_corrupt_macro_mean_epe_px": corrupt_mean,
            "loo_minus_full_corrupt_macro_scene_sd_px": corrupt_sd,
            "approximate_independent_scenes_for_80pct_power": planning_n,
            "corrupt_macro_scene_bootstrap_95ci_px": [
                float(np.percentile(corrupt_boot, 2.5)),
                float(np.percentile(corrupt_boot, 97.5)),
            ],
            "oracle_corrupt_macro_incremental_mean_epe_px": float(oracle_delta.mean()),
            "oracle_corrupt_macro_scene_bootstrap_95ci_px": [
                float(np.percentile(oracle_boot, 2.5)),
                float(np.percentile(oracle_boot, 97.5)),
            ],
            "oracle_unique_strict_positive_winner_rows": int(strict_winner.sum()),
            "loo_minus_full_clean_mean_epe_px": float(clean_delta.mean()),
            "clean_scene_bootstrap_95ci_px": [
                float(np.percentile(clean_boot, 2.5)),
                float(np.percentile(clean_boot, 97.5)),
            ],
            "per_condition": per_condition,
            "loo_global_gates": {
                "corrupt_relative_gain_vs_native": summary["corrupt_relative_gain_vs_p0"],
                "clean_relative_degradation": summary["clean_relative_degradation"],
                "corrupt_conditions_improved": summary["corrupt_conditions_improved"],
                "catastrophe_per_intervention": summary["catastrophe_per_intervention"],
            },
        }

    # Bonferroni one-sided 95% familywise lower bounds across the four actions.
    # This is conservative and avoids treating four individual 95% intervals as
    # a simultaneous non-redundancy claim.
    lower_percentile = 100.0 * (0.05 / len(LOO))
    for (action_id, comparison), bootstrap in zip(comparisons.items(), corrupt_bootstraps):
        lower = float(np.percentile(bootstrap, lower_percentile))
        comparison["corrupt_macro_bonferroni_one_sided_95pct_lower_px"] = lower
        comparison["selection_aware_nonredundancy_passes"] = lower > 0.0
    for (action_id, comparison), bootstrap in zip(comparisons.items(), oracle_bootstraps):
        lower = float(np.percentile(bootstrap, lower_percentile))
        comparison["oracle_corrupt_macro_bonferroni_one_sided_95pct_lower_px"] = lower
        comparison["oracle_capacity_nonredundancy_passes"] = lower > 0.0

    full_result = json.loads((FULL / "RESULT.json").read_text(encoding="utf-8"))
    full_summary = full_result["variant_summaries"]["CTRL-FACT"]
    passing = [
        action for action, value in comparisons.items()
        if value["selection_aware_nonredundancy_passes"]
    ]
    output = {
        "schema": "selection-aware-loo-opened-development-analysis/v1",
        "status": "COMPLETE_OPENED_DEVELOPMENT_METHOD_DIAGNOSTIC",
        "authority": {
            "production": False,
            "scientific_qualification": False,
            "selector_admission": False,
        },
        "interpretation": (
            "Paired retraining estimates substitution-aware incremental value, but E3 was "
            "adaptively reused and cannot provide final admission evidence."
        ),
        "estimand": "LOO CTRL-FACT loss minus full-bank CTRL-FACT loss on paired outer-OOF rows",
        "positive_delta_meaning": "the full bank has lower EPE, so the excluded action adds value",
        "bootstrap": {
            "unit": "physical_scene",
            "scene_count": len(scenes),
            "draws": draws,
            "seed": seed,
            "multiplicity": "Bonferroni one-sided familywise 95% lower bound across four actions",
            "lower_percentile": lower_percentile,
        },
        "power_planning_caveat": (
            "Normal-approximation counts use opened E3 scene variance, one-sided alpha "
            "0.05/4, and 80% power. They are planning diagnostics, not guaranteed sample sizes."
        ),
        "full_bank": {
            "experiment": FULL.name,
            "predictions_sha256": _sha256(full_path),
            "corrupt_relative_gain_vs_native": full_summary["corrupt_relative_gain_vs_p0"],
            "clean_relative_degradation": full_summary["clean_relative_degradation"],
            "catastrophe_per_intervention": full_summary["catastrophe_per_intervention"],
            "corrupt_macro_mean_epe": float(full_corrupt.mean()),
            "native_corrupt_macro_mean_epe": float(native_corrupt.mean()),
            "clean_mean_epe": float(full_clean.mean()),
            "native_clean_mean_epe": float(native_clean.mean()),
        },
        "comparisons": comparisons,
        "selection_aware_nonredundant_actions_on_opened_e3": passing,
        "selection_aware_rejected_actions_on_opened_e3": [
            action for action in LOO if action not in passing
        ],
        "oracle_capacity_nonredundant_actions_on_opened_e3": [
            action for action, value in comparisons.items()
            if value["oracle_capacity_nonredundancy_passes"]
        ],
    }
    _atomic_json(OUTPUT, output)
    print(json.dumps({
        "status": output["status"],
        "passing": passing,
        "rejected": output["selection_aware_rejected_actions_on_opened_e3"],
        "output": str(OUTPUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
