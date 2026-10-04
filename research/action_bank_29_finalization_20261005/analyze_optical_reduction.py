#!/usr/bin/env python3
"""Recompute the architecture-aligned reduction of the 15 optical arms.

This is an opened-development capacity audit.  It never grants selector or
production authority.  The four families below are mathematical action
families; their members are exact strength/operator variants competing with
native on the same 1,200-row E3 panel.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CATALOG = ROOT / "experiments/E243_restoration_candidate_integration_v1/FROZEN_ACTION_BANK.json"
INVENTORY = Path(
    "/ssd7/cactus8603/roco_spring/optical-flow-track/outputs/"
    "action-family-v3/sea-raft-action-inventory-v3.2.json"
)
NATIVE = "CSB/OF/SEA-RAFT/action/P0"

FAMILIES = {
    "lowpass_hf_suppression": (
        "CSB/OF/SEA-RAFT/action/V3-gaussian-s0p5",
        "CSB/OF/SEA-RAFT/action/P1-gaussian-s1",
        "CSB/OF/SEA-RAFT/action/R1-gaussian-s1p5",
        "CSB/OF/SEA-RAFT/action/R2-gaussian-s2",
    ),
    "detail_recovery_unsharp": (
        "CSB/OF/SEA-RAFT/action/P2-unsharp-s1",
        "CSB/OF/SEA-RAFT/action/R3-unsharp-a1",
        "CSB/OF/SEA-RAFT/action/R4-unsharp-a1p5",
    ),
    "joint_radiometry": (
        "CSB/OF/SEA-RAFT/action/V3-radiometric-blend-l0p25",
        "CSB/OF/SEA-RAFT/action/V3-radiometric-blend-l0p5",
        "CSB/OF/SEA-RAFT/action/V3-radiometric-blend-l0p75",
        "CSB/OF/SEA-RAFT/action/P3-joint-percentile-s1",
        "CSB/OF/SEA-RAFT/action/R5-joint-channel-percentile-s1",
        "CSB/OF/SEA-RAFT/action/R6-joint-percentile-s5",
    ),
    "matcher_compute": (
        "CSB/OF/SEA-RAFT/action/P4-iters8",
        "CSB/OF/SEA-RAFT/action/R7-iters12",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def load_rows() -> tuple[list[dict], list[dict]]:
    inventory = read_json(INVENTORY)
    rows_by_id: dict[str, dict] = {}
    bindings: list[dict] = []
    for source_name in ("recovery", "v3"):
        for item in inventory["source_folds"][source_name]:
            record = item["records"]
            path = Path(record["path"])
            if sha256(path) != record["sha256"]:
                raise ValueError(f"source hash drift: {path}")
            bindings.append({
                "source": source_name,
                "path": str(path),
                "sha256": record["sha256"],
            })
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    incoming = json.loads(line)
                    row_id = incoming["row_id"]
                    if row_id not in rows_by_id:
                        rows_by_id[row_id] = incoming
                    else:
                        current = rows_by_id[row_id]
                        if (
                            current["condition"] != incoming["condition"]
                            or current["scene"] != incoming["scene"]
                        ):
                            raise ValueError(f"cross-source row drift: {row_id}")
                        current["actions"].update(incoming["actions"])
    rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    if len(rows) != 1200 or len({row["row_id"] for row in rows}) != 1200:
        raise ValueError("expected 1,200 unique E3 rows")
    return rows, bindings


def capacity(
    matrix: np.ndarray,
    native: np.ndarray,
    indices: Iterable[int],
    mask: np.ndarray,
) -> float:
    chosen = tuple(indices)
    if not chosen:
        return 0.0
    best = np.min(matrix[mask][:, chosen], axis=1)
    return float(np.maximum(native[mask] - best, 0.0).mean())


def pct(value: float, denominator: float) -> float:
    return 100.0 * value / denominator if denominator > 0 else 100.0


def summarize_subset(
    action_ids: tuple[str, ...],
    selected: tuple[str, ...],
    index: dict[str, int],
    matrix: np.ndarray,
    native: np.ndarray,
    conditions: np.ndarray,
    condition_names: tuple[str, ...],
) -> dict:
    all_indices = tuple(index[action] for action in action_ids)
    selected_indices = tuple(index[action] for action in selected)
    corrupt = conditions != "clean"
    full = capacity(matrix, native, all_indices, corrupt)
    observed = capacity(matrix, native, selected_indices, corrupt)
    by_condition = {}
    min_retention = 100.0
    for condition in condition_names:
        mask = conditions == condition
        full_condition = capacity(matrix, native, all_indices, mask)
        selected_condition = capacity(matrix, native, selected_indices, mask)
        retention = pct(selected_condition, full_condition)
        min_retention = min(min_retention, retention)
        by_condition[condition] = {
            "full_positive_oracle_mean_gain_raw_px": full_condition,
            "selected_positive_oracle_mean_gain_raw_px": selected_condition,
            "retention_percent": retention,
        }
    return {
        "selected": list(selected),
        "selected_count": len(selected),
        "full_positive_oracle_mean_gain_raw_px": full,
        "selected_positive_oracle_mean_gain_raw_px": observed,
        "overall_retention_percent": pct(observed, full),
        "minimum_condition_retention_percent": min_retention,
        "by_condition": by_condition,
    }


def best_small_subset(
    action_ids: tuple[str, ...],
    index: dict[str, int],
    matrix: np.ndarray,
    native: np.ndarray,
    conditions: np.ndarray,
    condition_names: tuple[str, ...],
) -> dict:
    rows = []
    for size in range(1, len(action_ids) + 1):
        for selected in itertools.combinations(action_ids, size):
            row = summarize_subset(
                action_ids, selected, index, matrix, native, conditions,
                condition_names,
            )
            rows.append(row)
        passing = [
            row for row in rows
            if row["selected_count"] == size
            and row["overall_retention_percent"] >= 99.5
            and row["minimum_condition_retention_percent"] >= 98.0
        ]
        if passing:
            return max(
                passing,
                key=lambda row: (
                    row["minimum_condition_retention_percent"],
                    row["overall_retention_percent"],
                    tuple(row["selected"]),
                ),
            )
    return summarize_subset(
        action_ids, action_ids, index, matrix, native, conditions,
        condition_names,
    )


def global_subset_search(
    action_ids: tuple[str, ...],
    index: dict[str, int],
    matrix: np.ndarray,
    native: np.ndarray,
    conditions: np.ndarray,
    condition_names: tuple[str, ...],
) -> dict:
    """Find compact joint subsets against the full 15-arm oracle.

    Per-family searches can over-retain controls because another family may
    recover the same rows.  This exhaustive 2^15-scale search measures the
    actual cross-family redundancy.  We report both an unconstrained optimum
    and an optimum that keeps at least one exact control from every family.
    """
    family_by_action = {
        action: family
        for family, members in FAMILIES.items()
        for action in members
    }
    frontier = []
    passing_any = None
    passing_all_families = None
    family_names = set(FAMILIES)
    for size in range(1, len(action_ids) + 1):
        candidates = []
        for selected in itertools.combinations(action_ids, size):
            row = summarize_subset(
                action_ids, selected, index, matrix, native, conditions,
                condition_names,
            )
            represented = sorted({family_by_action[action] for action in selected})
            row["represented_families"] = represented
            row["all_families_represented"] = set(represented) == family_names
            candidates.append(row)
        best = max(
            candidates,
            key=lambda row: (
                row["minimum_condition_retention_percent"],
                row["overall_retention_percent"],
                tuple(row["selected"]),
            ),
        )
        frontier.append({
            "selected_count": size,
            "selected": best["selected"],
            "represented_families": best["represented_families"],
            "overall_retention_percent": best["overall_retention_percent"],
            "minimum_condition_retention_percent": best[
                "minimum_condition_retention_percent"
            ],
            "condition_retention_percent": {
                condition: values["retention_percent"]
                for condition, values in best["by_condition"].items()
            },
        })
        pass_rows = [
            row for row in candidates
            if row["overall_retention_percent"] >= 99.5
            and row["minimum_condition_retention_percent"] >= 98.0
        ]
        if passing_any is None and pass_rows:
            passing_any = max(
                pass_rows,
                key=lambda row: (
                    row["minimum_condition_retention_percent"],
                    row["overall_retention_percent"],
                    tuple(row["selected"]),
                ),
            )
        family_pass_rows = [row for row in pass_rows if row["all_families_represented"]]
        if passing_all_families is None and family_pass_rows:
            passing_all_families = max(
                family_pass_rows,
                key=lambda row: (
                    row["minimum_condition_retention_percent"],
                    row["overall_retention_percent"],
                    tuple(row["selected"]),
                ),
            )
    if passing_any is None or passing_all_families is None:
        raise ValueError("full action set unexpectedly failed its own retention gates")
    return {
        "minimum_passing_subset": passing_any,
        "minimum_passing_subset_with_all_families": passing_all_families,
        "best_minimum_condition_retention_by_control_count": frontier,
    }


def scene_cluster_bootstrap(
    action_ids: tuple[str, ...],
    selected: tuple[str, ...],
    index: dict[str, int],
    matrix: np.ndarray,
    native: np.ndarray,
    conditions: np.ndarray,
    scenes: np.ndarray,
    condition_names: tuple[str, ...],
    draws: int = 2000,
) -> dict:
    """Estimate reduction stability while preserving scene-level dependence."""
    rng = np.random.default_rng(20261005)
    unique_scenes = np.asarray(sorted(set(scenes)))
    full_indices = tuple(index[action] for action in action_ids)
    selected_indices = tuple(index[action] for action in selected)
    overall = []
    per_condition = {condition: [] for condition in condition_names}
    for _ in range(draws):
        sampled = rng.choice(unique_scenes, size=len(unique_scenes), replace=True)
        multiplicity = {scene: int(np.sum(sampled == scene)) for scene in set(sampled)}
        row_weights = np.asarray([multiplicity.get(scene, 0) for scene in scenes])
        corrupt = conditions != "clean"
        full_best = np.min(matrix[:, full_indices], axis=1)
        selected_best = np.min(matrix[:, selected_indices], axis=1)
        full_gain = np.maximum(native - full_best, 0.0)
        selected_gain = np.maximum(native - selected_best, 0.0)
        mask = corrupt & (row_weights > 0)
        overall.append(
            pct(
                float(np.average(selected_gain[mask], weights=row_weights[mask])),
                float(np.average(full_gain[mask], weights=row_weights[mask])),
            )
        )
        for condition in condition_names:
            mask = (conditions == condition) & (row_weights > 0)
            per_condition[condition].append(
                pct(
                    float(np.average(selected_gain[mask], weights=row_weights[mask])),
                    float(np.average(full_gain[mask], weights=row_weights[mask])),
                )
            )

    def interval(values: list[float]) -> dict:
        array = np.asarray(values)
        return {
            "p2_5": float(np.percentile(array, 2.5)),
            "median": float(np.percentile(array, 50.0)),
            "p97_5": float(np.percentile(array, 97.5)),
        }

    return {
        "unit": "scene",
        "draws": draws,
        "seed": 20261005,
        "selection_refit_inside_draw": False,
        "overall_retention_percent_interval": interval(overall),
        "condition_retention_percent_intervals": {
            condition: interval(values)
            for condition, values in per_condition.items()
        },
        "interpretation": (
            "Fixed-subset stability diagnostic on opened development data; it is "
            "not fresh qualification and does not correct subset-selection optimism."
        ),
    }


def selected_subset_necessity(
    action_ids: tuple[str, ...],
    selected: tuple[str, ...],
    index: dict[str, int],
    matrix: np.ndarray,
    native: np.ndarray,
    conditions: np.ndarray,
    condition_names: tuple[str, ...],
) -> dict:
    """Audit whether each selected anchor is dispensable from the frozen set."""
    corrupt = conditions != "clean"
    selected_indices = tuple(index[action] for action in selected)
    selected_values = matrix[:, selected_indices]
    best = np.min(selected_values, axis=1)
    positive = best < native
    tied = np.isclose(selected_values, best[:, None], rtol=0.0, atol=1e-12)
    unique = np.sum(tied, axis=1) == 1
    rows = {}
    for position, action in enumerate(selected):
        without = tuple(item for item in selected if item != action)
        summary = summarize_subset(
            action_ids, without, index, matrix, native, conditions,
            condition_names,
        )
        winner_mask = corrupt & positive & unique & tied[:, position]
        summary["passes_reduction_gates"] = (
            summary["overall_retention_percent"] >= 99.5
            and summary["minimum_condition_retention_percent"] >= 98.0
        )
        summary["unique_positive_winner_rows"] = int(np.sum(winner_mask))
        summary["unique_positive_winner_mean_gain_raw_px"] = (
            float(np.mean(native[winner_mask] - matrix[winner_mask, index[action]]))
            if np.any(winner_mask) else 0.0
        )
        rows[action] = summary
    return {
        "all_selected_anchors_individually_required_for_frozen_gates": all(
            not row["passes_reduction_gates"] for row in rows.values()
        ),
        "leave_one_selected_anchor_out": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    catalog = read_json(CATALOG)
    catalog_ids = {row["action_id"] for row in catalog["actions"]}
    optical_ids = tuple(action for values in FAMILIES.values() for action in values)
    if len(catalog_ids) != 29 or len(optical_ids) != 15 or len(set(optical_ids)) != 15:
        raise ValueError("catalog/family cardinality drift")
    if not set(optical_ids) <= catalog_ids:
        raise ValueError("family references an action outside the frozen 29")

    rows, bindings = load_rows()
    actions_in_rows = set(rows[0]["actions"])
    if actions_in_rows != set(optical_ids) | {NATIVE}:
        raise ValueError("E3 action set drift")
    if any(set(row["actions"]) != actions_in_rows for row in rows):
        raise ValueError("incomplete action row")

    ordered = tuple(sorted(optical_ids))
    index = {action: i for i, action in enumerate(ordered)}
    matrix = np.asarray([
        [row["actions"][action]["metrics"]["epe"] for action in ordered]
        for row in rows
    ], dtype=np.float64)
    native = np.asarray([
        row["actions"][NATIVE]["metrics"]["epe"] for row in rows
    ], dtype=np.float64)
    conditions = np.asarray([row["condition"] for row in rows])
    scenes = np.asarray([row["scene"] for row in rows])
    condition_names = tuple(sorted(set(conditions) - {"clean"}))
    corrupt = conditions != "clean"

    family_results = {}
    for family, action_ids in FAMILIES.items():
        selected = best_small_subset(
            action_ids, index, matrix, native, conditions, condition_names,
        )
        full_indices = tuple(index[action] for action in action_ids)
        full = capacity(matrix, native, full_indices, corrupt)
        loao = {}
        for action in action_ids:
            kept = tuple(index[item] for item in action_ids if item != action)
            without = capacity(matrix, native, kept, corrupt)
            loao[action] = {
                "absolute_positive_oracle_loss_raw_px": full - without,
                "relative_family_capacity_loss_percent": pct(full - without, full),
            }
        family_results[family] = {
            "members": list(action_ids),
            "recommended_strength_subset": selected,
            "leave_one_action_out": loao,
        }

    family_names = tuple(FAMILIES)
    family_frontier = []
    full_indices = tuple(index[action] for action in optical_ids)
    full_capacity = capacity(matrix, native, full_indices, corrupt)
    full_by_condition = {
        condition: capacity(
            matrix, native, full_indices, conditions == condition,
        )
        for condition in condition_names
    }
    for size in range(1, len(family_names) + 1):
        candidates = []
        for selected_families in itertools.combinations(family_names, size):
            selected_actions = tuple(
                action
                for family in selected_families
                for action in FAMILIES[family]
            )
            selected_indices = tuple(index[action] for action in selected_actions)
            observed = capacity(matrix, native, selected_indices, corrupt)
            per_condition = {
                condition: pct(
                    capacity(
                        matrix, native, selected_indices,
                        conditions == condition,
                    ),
                    full_by_condition[condition],
                )
                for condition in condition_names
            }
            candidates.append({
                "families": list(selected_families),
                "overall_retention_percent": pct(observed, full_capacity),
                "minimum_condition_retention_percent": min(per_condition.values()),
                "condition_retention_percent": per_condition,
            })
        family_frontier.append(max(
            candidates,
            key=lambda row: (
                row["overall_retention_percent"],
                row["minimum_condition_retention_percent"],
                tuple(row["families"]),
            ),
        ))

    clean_mask = conditions == "clean"
    fixed_condition_response = {}
    for action in ordered:
        action_values = matrix[:, index[action]]
        clean_delta = float((native[clean_mask] - action_values[clean_mask]).mean())
        fixed_condition_response[action] = {
            "clean_mean_gain_raw_px": clean_delta,
            "clean_relative_gain_percent": 100.0 * clean_delta / float(native[clean_mask].mean()),
            "target_condition_mean_gain_raw_px": {
                condition: float(
                    (native[conditions == condition] - action_values[conditions == condition]).mean()
                )
                for condition in condition_names
            },
        }

    global_subsets = global_subset_search(
        optical_ids, index, matrix, native, conditions, condition_names,
    )
    reduced_selected = tuple(
        global_subsets["minimum_passing_subset_with_all_families"]["selected"]
    )
    bootstrap = scene_cluster_bootstrap(
        optical_ids, reduced_selected, index, matrix, native, conditions, scenes,
        condition_names,
    )
    necessity = selected_subset_necessity(
        optical_ids, reduced_selected, index, matrix, native, conditions,
        condition_names,
    )

    result = {
        "schema": "roco-action-bank-29-optical-reduction/v2",
        "status": "PASS_OPENED_DEVELOPMENT_REDUCTION_NO_SELECTOR_AUTHORITY",
        "authority": {
            "scientific_qualification": False,
            "selector_admission": False,
            "production": False,
        },
        "source": {
            "catalog": {"path": str(CATALOG.relative_to(ROOT)), "sha256": sha256(CATALOG)},
            "inventory": {"path": str(INVENTORY), "sha256": sha256(INVENTORY)},
            "records": bindings,
        },
        "population": {
            "rows": len(rows),
            "scenes": len({row["scene"] for row in rows}),
            "conditions": {name: int(np.sum(conditions == name)) for name in sorted(set(conditions))},
            "outcome_status": "OPENED_E3_DEVELOPMENT_DIAGNOSTIC",
        },
        "gates": {
            "strength_subset_overall_capacity_retention_percent": 99.5,
            "strength_subset_each_condition_retention_percent": 98.0,
            "native_positive_only_rollback": True,
        },
        "full_15_arm_positive_oracle_mean_gain_raw_px": full_capacity,
        "families": family_results,
        "family_count_frontier": family_frontier,
        "global_exact_control_reduction": global_subsets,
        "selected_subset_scene_cluster_bootstrap": bootstrap,
        "selected_subset_necessity": necessity,
        "fixed_condition_response": fixed_condition_response,
        "interpretation": (
            "Capacity and overlap evidence for reducing exact arms into family/strength "
            "grids. Fresh component-disjoint safety and observable routing remain required."
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.write:
        (HERE / "OPTICAL_REDUCTION_RESULT.json").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
