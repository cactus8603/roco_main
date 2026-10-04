"""Detached sequential S01-R..S05-R Robust20 x severity workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .util import ROOT, sha256
from .workflow import Stage, execute_plan


CONFIGS = {
    "s01r": "e01_s01r_robust20_v1.json",
    "s02r": "e01_s02r_robust20_v1.json",
    "s03r": "e01_s03r_robust20_v1.json",
    "s04r": "e01_s04r_robust20_v1.json",
    "s05r": "e01_s05r_robust20_v1.json",
}


def build_plan(output_root: str | Path, *, device: str = "cuda:6") -> dict:
    output_root = Path(output_root).resolve()
    workflow_dir = output_root / "_workflow"
    config_root = ROOT / "configs/stablebridge/robust20"
    configs = {key: config_root / name for key, name in CONFIGS.items()}
    inputs = set((ROOT / "src/stablebridge").glob("*.py"))
    inputs.update(configs.values())
    inputs.update({
        ROOT / "configs/stablebridge/data_models_v1.json",
        ROOT / "experiments/E01_evidence_mechanism/manifests/development_clips_v1.json",
        ROOT / "docs/specs/s01r_robust20_revalidation_contract_v1_20260917.md",
    })
    records = [
        {"path": str(path.resolve()), "relative_path": str(path.resolve().relative_to(ROOT)),
         "sha256": sha256(path)} for path in sorted(inputs)
    ]
    runs = {key: output_root / {
        "s01r": "s01r_signal_v1", "s02r": "s02r_temporal_v1",
        "s03r": "s03r_source_control_v1", "s04r": "s04r_support_rematch_v1",
        "s05r": "s05r_binary_revision_v1",
    }[key] for key in CONFIGS}
    python = sys.executable
    stages = [
        Stage("s01r", (python, "-m", "stablebridge.cli", "run", "--config", str(configs["s01r"]),
                       "--run-dir", str(runs["s01r"]), "--device", device),
              (str(runs["s01r"] / "report.json"), str(runs["s01r"] / "status.json")),
              str(runs["s01r"] / "status.json")),
        Stage("s02r", (python, "-m", "stablebridge.quartet_experiment", "--config", str(configs["s02r"]),
                       "--run-dir", str(runs["s02r"]), "--device", device),
              (str(runs["s02r"] / "metrics/summary.json"), str(runs["s02r"] / "complete.json"))),
        Stage("s03r", (python, "-m", "stablebridge.cli", "run", "--config", str(configs["s03r"]),
                       "--run-dir", str(runs["s03r"]), "--device", device),
              (str(runs["s03r"] / "report.json"), str(runs["s03r"] / "status.json")),
              str(runs["s03r"] / "status.json")),
        Stage("s04r", (python, "-m", "stablebridge.support_experiment", "--config", str(configs["s04r"]),
                       "--output", str(runs["s04r"]), "--device", device),
              (str(runs["s04r"] / "report.json"), str(runs["s04r"] / "status.json")),
              str(runs["s04r"] / "status.json")),
        Stage("s05r", (python, "-m", "stablebridge.binary_experiment", "--config", str(configs["s05r"]),
                       "--output", str(runs["s05r"]), "--device", device),
              (str(runs["s05r"] / "report.json"), str(runs["s05r"] / "status.json")),
              str(runs["s05r"] / "status.json")),
    ]
    return {
        "schema_version": 1,
        "workflow_name": "robust20_e01_s01r_to_s05r_v2",
        "root": str(ROOT),
        "output_dir": str(workflow_dir),
        "output_root": str(output_root),
        "device": device,
        "conditions": 20,
        "severities": [1, 2, 3],
        "clean_included": False,
        "python_executable": python,
        "inputs": records,
        "stages": [stage.__dict__ for stage in stages],
        "runs": {key: str(value) for key, value in runs.items()},
        "stop_policy": "stop_on_engineering_failure; preserve scientific negative results",
        "automatic_hyperparameter_retry": False,
        "scientific_success_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.output_root, device=args.device)
    if args.dry_run:
        import json
        print(json.dumps(plan, indent=2, ensure_ascii=False))
    else:
        execute_plan(plan)


if __name__ == "__main__":
    main()
