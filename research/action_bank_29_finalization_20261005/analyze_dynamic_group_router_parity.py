#!/usr/bin/env python3
"""Verify that the dynamic grouped router exactly reproduces frozen E278."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "experiments/E278_minimal_stable_observable_bank_v1"
DYNAMIC = ROOT / "experiments/E280_dynamic_group_router_parity_v1"
OUTPUT = DYNAMIC / "PARITY.json"
OUTPUT_NAMES = ("utility", "harm", "catastrophe")


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _maximum_output_error(reference: list[dict], dynamic: list[dict], name: str) -> float:
    differences = (
        abs(float(expected) - float(observed))
        for left, right in zip(reference, dynamic)
        for expected, observed in zip(
            left["primary_model_outputs"][name],
            right["primary_model_outputs"][name],
        )
    )
    return max(differences, default=0.0)


def _build() -> dict:
    reference_result_path = REFERENCE / "RESULT.json"
    dynamic_result_path = DYNAMIC / "RESULT.json"
    reference_records_path = REFERENCE / "PREDICTIONS.jsonl"
    dynamic_records_path = DYNAMIC / "PREDICTIONS.jsonl"
    reference_result = _load(reference_result_path)
    dynamic_result = _load(dynamic_result_path)
    reference_rows = _rows(reference_records_path)
    dynamic_rows = _rows(dynamic_records_path)

    if len(reference_rows) != len(dynamic_rows):
        raise AssertionError("prediction row count differs")
    if reference_result["action_ids"] != dynamic_result["action_ids"]:
        raise AssertionError("action IDs differ")
    if [row["row_id"] for row in reference_rows] != [
        row["row_id"] for row in dynamic_rows
    ]:
        raise AssertionError("row identity/order differs")
    if [row["outer_fold"] for row in reference_rows] != [
        row["outer_fold"] for row in dynamic_rows
    ]:
        raise AssertionError("outer fold assignment differs")

    policy_keys = (
        "action_id",
        "action_index",
        "selected_epe",
        "raw_gain",
        "harm",
        "catastrophe",
    )
    policy_mismatch_count = sum(
        any(
            left["policies"]["CTRL-FACT"][key]
            != right["policies"]["CTRL-FACT"][key]
            for key in policy_keys
        )
        for left, right in zip(reference_rows, dynamic_rows)
    )
    output_max_abs_difference = {
        name: _maximum_output_error(reference_rows, dynamic_rows, name)
        for name in OUTPUT_NAMES
    }

    if policy_mismatch_count:
        raise AssertionError(f"{policy_mismatch_count} CTRL-FACT policy rows differ")
    if any(value != 0.0 for value in output_max_abs_difference.values()):
        raise AssertionError(f"model outputs differ: {output_max_abs_difference}")
    if (
        reference_result["variant_summaries"]["CTRL-FACT"]
        != dynamic_result["variant_summaries"]["CTRL-FACT"]
    ):
        raise AssertionError("CTRL-FACT summary differs")
    if (
        reference_result["bootstrap"]["intervals"]["CTRL-FACT"]
        != dynamic_result["bootstrap"]["intervals"]["CTRL-FACT"]
    ):
        raise AssertionError("CTRL-FACT bootstrap interval differs")

    reference_outer = reference_result["outer_runs"]
    dynamic_outer = dynamic_result["outer_runs"]
    if len(reference_outer) != len(dynamic_outer):
        raise AssertionError("outer fold count differs")
    for expected, observed in zip(reference_outer, dynamic_outer):
        comparisons = {
            "outer_fold": (expected["outer_fold"], observed["outer_fold"]),
            "target_thresholds": (
                expected["target_thresholds"],
                observed["target_thresholds"],
            ),
            "calibrated_config": (
                expected["calibrated_configs"]["CTRL-FACT"],
                observed["calibrated_config"],
            ),
            "calibration": (
                expected["calibration"]["CTRL-FACT"],
                observed["calibration"],
            ),
            "final_training_losses": (
                expected["final_training_losses"]["factorized"],
                {"full_scalar": observed["final_training_losses"]},
            ),
        }
        for label, (left, right) in comparisons.items():
            if left != right:
                raise AssertionError(
                    f"outer {expected['outer_fold']} {label} differs"
                )

    reference_models = sorted((REFERENCE / "MODELS").glob("outer*-factorized.npz"))
    dynamic_models = sorted((DYNAMIC / "MODELS").glob("outer*-factorized.npz"))
    reference_model_hashes = {path.name: _sha256(path) for path in reference_models}
    dynamic_model_hashes = {path.name: _sha256(path) for path in dynamic_models}
    if reference_model_hashes != dynamic_model_hashes:
        raise AssertionError("saved factorized models differ")

    return {
        "schema": "dynamic-group-router-parity/v1",
        "status": "PASS_EXACT_REPRODUCTION",
        "diagnostic_only": True,
        "authority": {
            "fresh_validation": False,
            "action_admission": False,
            "production_promotion": False,
        },
        "reference_experiment": str(REFERENCE.relative_to(ROOT)),
        "dynamic_experiment": str(DYNAMIC.relative_to(ROOT)),
        "row_count": len(reference_rows),
        "outer_fold_count": len(reference_outer),
        "checks": {
            "row_ids_and_order_exact": True,
            "fold_assignments_exact": True,
            "action_ids_exact": True,
            "ctrl_fact_policy_mismatch_count": policy_mismatch_count,
            "primary_model_output_max_abs_difference": output_max_abs_difference,
            "ctrl_fact_summary_exact": True,
            "ctrl_fact_bootstrap_interval_exact": True,
            "outer_thresholds_calibration_and_losses_exact": True,
            "saved_factorized_model_hashes_exact": True,
        },
        "source_hashes": {
            "reference_result": _sha256(reference_result_path),
            "reference_predictions": _sha256(reference_records_path),
            "dynamic_result": _sha256(dynamic_result_path),
            "dynamic_predictions": _sha256(dynamic_records_path),
            "dynamic_trainer": _sha256(Path(__file__).with_name("train_dynamic_group_router.py")),
        },
        "factorized_model_sha256": reference_model_hashes,
        "interpretation": (
            "The configurable trainer reproduces the frozen six-fold CTRL-FACT "
            "implementation exactly. This validates implementation parity only; "
            "E3 remains opened development data and cannot admit actions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    value = _build()
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.write:
        if OUTPUT.exists():
            if OUTPUT.read_text(encoding="utf-8") != serialized:
                raise SystemExit(f"refusing to replace nonidentical evidence: {OUTPUT}")
        else:
            temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(OUTPUT)
    print(
        "PASS: dynamic GroupKFold trainer exactly reproduces 1,200 E278 "
        "CTRL-FACT rows, outputs, calibration, intervals, and model hashes"
    )


if __name__ == "__main__":
    main()
