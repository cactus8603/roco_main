#!/usr/bin/env python3
"""Analyze whether the opened E3 router uses two low-pass strengths well."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("E282_STRENGTH_PROPOSAL_RESULT.json")
RUNS = {
    "two_strengths": ROOT / "experiments/E282a_lowpass_two_strength_router_v1",
    "sigma1_only": ROOT / "experiments/E282b_lowpass_sigma1_router_v1",
    "sigma2_only": ROOT / "experiments/E282c_lowpass_sigma2_router_v1",
}
P1 = "CSB/OF/SEA-RAFT/action/P1-gaussian-s1"
R2 = "CSB/OF/SEA-RAFT/action/R2-gaussian-s2"
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


def _scene_matrix(rows: list[dict], values: np.ndarray) -> np.ndarray:
    scenes = sorted({str(row["scene"]) for row in rows})
    matrix = np.empty((len(scenes), len(CONDITIONS)), dtype=np.float64)
    for scene_index, scene in enumerate(scenes):
        for condition_index, condition in enumerate(CONDITIONS):
            indices = [
                index
                for index, row in enumerate(rows)
                if row["scene"] == scene and row["condition_provenance"] == condition
            ]
            if not indices:
                raise ValueError(f"missing cell: {scene}/{condition}")
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
    rows = {name: _read(directory / "PREDICTIONS.jsonl") for name, directory in RUNS.items()}
    reference_ids = [row["row_id"] for row in rows["two_strengths"]]
    reference_folds = [row["outer_fold"] for row in rows["two_strengths"]]
    for name, values in rows.items():
        if len(values) != 1200 or [row["row_id"] for row in values] != reference_ids:
            raise ValueError(f"row alignment drift: {name}")
        if [row["outer_fold"] for row in values] != reference_folds:
            raise ValueError(f"fold alignment drift: {name}")

    results = {
        name: json.loads((directory / "RESULT.json").read_text(encoding="utf-8"))
        for name, directory in RUNS.items()
    }
    summaries = {}
    policy_scene = {}
    oracle_scene = {}
    for name, values in rows.items():
        result = results[name]
        policy_loss = np.asarray(
            [row["policies"]["CTRL-FACT"]["selected_epe"] for row in values],
            dtype=np.float64,
        )
        policy_scene[name] = _scene_matrix(values, policy_loss).mean(axis=1)
        teacher = json.loads((RUNS[name] / "TEACHER.json").read_text(encoding="utf-8"))
        teacher_rows = _read(Path(teacher["records"]))
        if [row["row_id"] for row in teacher_rows] != reference_ids:
            raise ValueError(f"teacher alignment drift: {name}")
        action_ids = list(teacher["action_ids"])
        action_losses = np.asarray(
            [
                [row["action_targets"][action]["raw_epe"] for action in action_ids]
                for row in teacher_rows
            ],
            dtype=np.float64,
        )
        oracle_scene[name] = _scene_matrix(values, action_losses.min(axis=1)).mean(axis=1)
        summary = result["variant_summaries"]["CTRL-FACT"]
        route_counts = Counter(
            row["policies"]["CTRL-FACT"]["action_id"] for row in values
        )
        per_condition = {}
        for condition in ("clean", *CONDITIONS):
            per_condition[condition] = dict(
                sorted(
                    Counter(
                        row["policies"]["CTRL-FACT"]["action_id"]
                        for row in values
                        if row["condition_provenance"] == condition
                    ).items()
                )
            )
        summaries[name] = {
            "experiment": RUNS[name].name,
            "protocol_sha256": _sha256(RUNS[name] / "PROTOCOL.json"),
            "predictions_sha256": _sha256(RUNS[name] / "PREDICTIONS.jsonl"),
            "action_ids": result["action_ids"],
            "corrupt_relative_gain_vs_native": summary["corrupt_relative_gain_vs_p0"],
            "clean_relative_degradation": summary["clean_relative_degradation"],
            "corrupt_conditions_improved": summary["corrupt_conditions_improved"],
            "interventions": summary["interventions"],
            "harmful_per_intervention": summary["harmful_per_intervention"],
            "catastrophe_per_intervention": summary["catastrophe_per_intervention"],
            "oracle_corrupt_relative_gain_vs_native": result["variant_summaries"][
                "CTRL-ORACLE"
            ]["corrupt_relative_gain_vs_p0"],
            "outer_folds_calibrated_all_native": sum(
                run["calibrated_config"].get("all_p0", False)
                for run in result["outer_runs"]
            ),
            "route_counts": dict(sorted(route_counts.items())),
            "route_counts_by_condition": per_condition,
        }

    seed = 20261005
    draws = 10_000
    scene_count = len(policy_scene["two_strengths"])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, scene_count, size=(draws, scene_count))
    comparisons = {}
    lower_percentile = 100.0 * (0.05 / 2)
    for name, removed_action in (("sigma1_only", R2), ("sigma2_only", P1)):
        policy_delta = policy_scene[name] - policy_scene["two_strengths"]
        oracle_delta = oracle_scene[name] - oracle_scene["two_strengths"]
        policy_boot = policy_delta[sampled].mean(axis=1)
        oracle_boot = oracle_delta[sampled].mean(axis=1)
        policy_lower = float(np.percentile(policy_boot, lower_percentile))
        oracle_lower = float(np.percentile(oracle_boot, lower_percentile))
        comparisons[name] = {
            "removed_action_id": removed_action,
            "single_minus_two_strength_policy_mean_epe_px": float(policy_delta.mean()),
            "single_minus_two_strength_policy_scene_bootstrap_95ci_px": [
                float(np.percentile(policy_boot, 2.5)),
                float(np.percentile(policy_boot, 97.5)),
            ],
            "single_minus_two_strength_policy_bonferroni_one_sided_95pct_lower_px": policy_lower,
            "two_strength_policy_value_passes": policy_lower > 0.0,
            "single_minus_two_strength_oracle_mean_epe_px": float(oracle_delta.mean()),
            "single_minus_two_strength_oracle_scene_bootstrap_95ci_px": [
                float(np.percentile(oracle_boot, 2.5)),
                float(np.percentile(oracle_boot, 97.5)),
            ],
            "single_minus_two_strength_oracle_bonferroni_one_sided_95pct_lower_px": oracle_lower,
            "two_strength_oracle_value_passes": oracle_lower > 0.0,
        }

    output = {
        "schema": "lowpass-discrete-strength-proposal-opened-development/v1",
        "status": "REJECT_ROUTED_STRENGTH_PROPOSAL_ON_OPENED_E3",
        "authority": {
            "production": False,
            "scientific_qualification": False,
            "selector_admission": False,
            "continuous_strength": False,
        },
        "interpretation": (
            "The two Gaussian anchors add oracle capacity, but the independently "
            "trained two-strength router calibrates to native in all six folds and "
            "does not outperform the single-strength routers. The current evidence "
            "supports retaining the anchors as a frozen capacity hypothesis, not a "
            "validated strength-proposal claim."
        ),
        "bootstrap": {
            "unit": "physical_scene",
            "scene_count": scene_count,
            "draws": draws,
            "seed": seed,
            "multiplicity": "Bonferroni one-sided familywise 95% lower bound over two removed strengths",
            "lower_percentile": lower_percentile,
        },
        "runs": summaries,
        "comparisons": comparisons,
        "two_strength_policy_value_passes_for_both_anchors": all(
            comparison["two_strength_policy_value_passes"]
            for comparison in comparisons.values()
        ),
        "two_strength_oracle_value_passes_for_both_anchors": all(
            comparison["two_strength_oracle_value_passes"]
            for comparison in comparisons.values()
        ),
    }
    _atomic_json(OUTPUT, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "policy_pass": output["two_strength_policy_value_passes_for_both_anchors"],
                "oracle_pass": output["two_strength_oracle_value_passes_for_both_anchors"],
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
