from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "experiments/E29_external_joint_panel/make_launch_plan.py"
SPEC = importlib.util.spec_from_file_location("e29_launch_plan", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_launch_plan_has_complete_nonoverlapping_primary_matrix() -> None:
    plan = MODULE.build_launch_plan()
    runs = plan["runs"]
    run_ids = [run["run_id"] for run in runs]
    outputs = [run["output"] for run in runs]
    assert len(run_ids) == len(set(run_ids)) == 41
    assert len(outputs) == len(set(outputs)) == 41
    assert plan["primary_transfer_setting"] == "T1_calibrated"
    assert plan["analysis_recipes"]["H1_T0_report_only"]["publishes_h2_gate"] is False
    assert plan["analysis_recipes"]["H1_T1_primary"]["publishes_h2_gate"] is True

    assert plan["phase_summary"] == {
        "c1_calibration_inputs": {"runs": 11, "rows": 8580, "expert_forwards": 9360},
        "h1_common_and_t0": {"runs": 13, "rows": 11830, "expert_forwards": 15470},
        "h1_t1_final": {"runs": 4, "rows": 3640, "expert_forwards": 6370},
        "h2_primary_t1": {"runs": 13, "rows": 11830, "expert_forwards": 15470},
    }


def test_t1_and_h2_guards_are_explicit() -> None:
    plan = MODULE.build_launch_plan()
    for run in plan["runs"]:
        environment = run["environment"]
        if run["transfer_setting"] == "T1_calibrated" and run["system_id"] == "final_asymmetric_bridge_first":
            assert environment["E29_T1_CONFIG"].endswith("/t1_config.json")
        else:
            assert "E29_T1_CONFIG" not in environment
        if run["task"] == "flow":
            assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        if run["role"] == "H2":
            assert run["conditional_on_h1_gate"] is True
            assert "--h1-gate" in run["argv"]
        else:
            assert run["conditional_on_h1_gate"] is False
