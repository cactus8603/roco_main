#!/usr/bin/env python3
"""Paired scene-cluster analysis of E278 versus retrained family-LOO banks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("E281_SELECTION_AWARE_FAMILY_LOO_RESULT.json")
FULL = ROOT / "experiments/E278_minimal_stable_observable_bank_v1"
FAMILIES = {
    "optical.lowpass_hf_suppression.v1": {
        "actions": (
            "CSB/OF/SEA-RAFT/action/P1-gaussian-s1",
            "CSB/OF/SEA-RAFT/action/R2-gaussian-s2",
        ),
        "loo": ROOT / "experiments/E281_selection_aware_family_loo_lowpass_v1",
    },
    "optical.detail_recovery_unsharp.v1": {
        "actions": ("CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5",),
        "loo": ROOT / "experiments/E279c_selection_aware_loo_r4_v1",
    },
    "optical.joint_radiometry.v1": {
        "actions": ("CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1",),
        "loo": ROOT / "experiments/E279d_selection_aware_loo_p3_v1",
    },
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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _scene_matrix(
    rows: list[dict], values: np.ndarray, conditions: tuple[str, ...]
) -> np.ndarray:
    scenes = sorted({str(row["scene"]) for row in rows})
    matrix = np.empty((len(scenes), len(conditions)), dtype=np.float64)
    for scene_index, scene in enumerate(scenes):
        for condition_index, condition in enumerate(conditions):
            indices = [
                index
                for index, row in enumerate(rows)
                if str(row["scene"]) == scene
                and str(row["condition_provenance"]) == condition
            ]
            if not indices:
                raise ValueError(f"missing scene/condition cell: {scene}/{condition}")
            matrix[scene_index, condition_index] = float(np.mean(values[indices]))
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
    full_corrupt = _scene_matrix(full_rows, full_loss, CONDITIONS).mean(axis=1)
    full_clean = _scene_matrix(full_rows, full_loss, ("clean",))[:, 0]

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
    full_oracle = _scene_matrix(
        full_rows, action_losses.min(axis=1), CONDITIONS
    ).mean(axis=1)

    seed = 20261005
    draws = 10_000
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(scenes), size=(draws, len(scenes)))
    comparisons = {}
    policy_bootstraps = []
    oracle_bootstraps = []
    for family_id, definition in FAMILIES.items():
        directory = definition["loo"]
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
        policy_delta = loo_corrupt - full_corrupt
        clean_delta = loo_clean - full_clean
        policy_boot = policy_delta[sampled].mean(axis=1)
        clean_boot = clean_delta[sampled].mean(axis=1)
        policy_bootstraps.append(policy_boot)

        excluded_indices = [action_ids.index(action) for action in definition["actions"]]
        retained_indices = [
            index for index in range(len(action_ids)) if index not in excluded_indices
        ]
        loo_oracle_loss = action_losses[:, retained_indices].min(axis=1)
        loo_oracle = _scene_matrix(full_rows, loo_oracle_loss, CONDITIONS).mean(axis=1)
        oracle_delta = loo_oracle - full_oracle
        oracle_boot = oracle_delta[sampled].mean(axis=1)
        oracle_bootstraps.append(oracle_boot)

        per_condition = {}
        for condition in CONDITIONS:
            full_cell = _scene_matrix(full_rows, full_loss, (condition,))[:, 0]
            loo_cell = _scene_matrix(rows, loss, (condition,))[:, 0]
            delta = loo_cell - full_cell
            bootstrap = delta[sampled].mean(axis=1)
            per_condition[condition] = {
                "loo_minus_full_mean_epe_px": float(delta.mean()),
                "scene_bootstrap_95ci_px": [
                    float(np.percentile(bootstrap, 2.5)),
                    float(np.percentile(bootstrap, 97.5)),
                ],
            }

        result = json.loads((directory / "RESULT.json").read_text(encoding="utf-8"))
        summary = result["variant_summaries"]["CTRL-FACT"]
        effect = float(policy_delta.mean())
        effect_sd = float(policy_delta.std(ddof=1))
        planning_n = None
        if effect > 0.0:
            z_alpha = NormalDist().inv_cdf(1.0 - 0.05 / len(FAMILIES))
            z_power = NormalDist().inv_cdf(0.80)
            planning_n = int(np.ceil(((z_alpha + z_power) * effect_sd / effect) ** 2))
        comparisons[family_id] = {
            "excluded_action_ids": list(definition["actions"]),
            "loo_experiment": directory.name,
            "loo_predictions_sha256": _sha256(path),
            "loo_minus_full_corrupt_macro_mean_epe_px": effect,
            "loo_minus_full_corrupt_macro_scene_sd_px": effect_sd,
            "approximate_independent_scenes_for_80pct_power": planning_n,
            "corrupt_macro_scene_bootstrap_95ci_px": [
                float(np.percentile(policy_boot, 2.5)),
                float(np.percentile(policy_boot, 97.5)),
            ],
            "loo_minus_full_clean_mean_epe_px": float(clean_delta.mean()),
            "clean_scene_bootstrap_95ci_px": [
                float(np.percentile(clean_boot, 2.5)),
                float(np.percentile(clean_boot, 97.5)),
            ],
            "oracle_corrupt_macro_incremental_mean_epe_px": float(oracle_delta.mean()),
            "oracle_corrupt_macro_scene_bootstrap_95ci_px": [
                float(np.percentile(oracle_boot, 2.5)),
                float(np.percentile(oracle_boot, 97.5)),
            ],
            "per_condition": per_condition,
            "loo_global_gates": {
                "corrupt_relative_gain_vs_native": summary["corrupt_relative_gain_vs_p0"],
                "clean_relative_degradation": summary["clean_relative_degradation"],
                "corrupt_conditions_improved": summary["corrupt_conditions_improved"],
                "catastrophe_per_intervention": summary["catastrophe_per_intervention"],
            },
        }

    lower_percentile = 100.0 * (0.05 / len(FAMILIES))
    for comparison, bootstrap in zip(comparisons.values(), policy_bootstraps):
        lower = float(np.percentile(bootstrap, lower_percentile))
        comparison["corrupt_macro_bonferroni_one_sided_95pct_lower_px"] = lower
        comparison["selection_aware_nonredundancy_passes"] = lower > 0.0
    for comparison, bootstrap in zip(comparisons.values(), oracle_bootstraps):
        lower = float(np.percentile(bootstrap, lower_percentile))
        comparison["oracle_corrupt_macro_bonferroni_one_sided_95pct_lower_px"] = lower
        comparison["oracle_capacity_nonredundancy_passes"] = lower > 0.0

    output = {
        "schema": "selection-aware-family-loo-opened-development-analysis/v1",
        "status": "COMPLETE_OPENED_DEVELOPMENT_METHOD_DIAGNOSTIC",
        "authority": {
            "production": False,
            "scientific_qualification": False,
            "selector_admission": False,
        },
        "interpretation": (
            "Family-level retraining measures substitution-aware incremental value "
            "after all controls in one repair mechanism are removed. E3 was adaptively "
            "reused, so these results cannot provide final admission authority."
        ),
        "estimand": "family-LOO CTRL-FACT loss minus full-bank CTRL-FACT loss on paired outer-OOF rows",
        "positive_delta_meaning": "the full bank has lower EPE, so the excluded family adds value",
        "bootstrap": {
            "unit": "physical_scene",
            "scene_count": len(scenes),
            "draws": draws,
            "seed": seed,
            "multiplicity": "Bonferroni one-sided familywise 95% lower bound across three families",
            "lower_percentile": lower_percentile,
        },
        "power_planning_caveat": (
            "Normal-approximation counts use opened E3 scene variance, one-sided alpha "
            "0.05/3, and 80% power. They are planning diagnostics, not guaranteed sample sizes."
        ),
        "full_bank": {
            "experiment": FULL.name,
            "predictions_sha256": _sha256(full_path),
        },
        "comparisons": comparisons,
        "selection_aware_nonredundant_families_on_opened_e3": [
            family
            for family, value in comparisons.items()
            if value["selection_aware_nonredundancy_passes"]
        ],
        "oracle_capacity_nonredundant_families_on_opened_e3": [
            family
            for family, value in comparisons.items()
            if value["oracle_capacity_nonredundancy_passes"]
        ],
    }
    _atomic_json(OUTPUT, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "selection_aware_passing": output[
                    "selection_aware_nonredundant_families_on_opened_e3"
                ],
                "oracle_capacity_passing": output[
                    "oracle_capacity_nonredundant_families_on_opened_e3"
                ],
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
