"""CPU integration of cached inference, geometry, fixed-Q scoring and ledgers."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from stablebridge.quartet_experiment import DIRECT, EDGES, evaluate_case
from stablebridge.util import save_json, save_npz, sha256


class QuartetRunnerIntegrationTests(unittest.TestCase):
    def test_cached_quartet_evaluates_both_tasks_with_fixed_cohort_and_shared_cost(self):
        h, w = 32, 32
        origin = [768, 380]
        case = {"id": "synthetic_0001_clean", "scene": "0001", "frame": 1,
                "profile": {"name": "clean"}}
        config = {
            "artifact_compression": "none",
            "context_hw": [h, w], "origin_xy": origin,
            "operators": ["identity", "median3", "gaussian1"],
            "path_operator_tuples": [[a, b, (a + b) % 3] for a in range(3) for b in range(3)],
            "wrong_middle_shift_xy": [5, 3], "epipolar_tolerance_px": 1.0,
            "threshold_px": 1.0, "harm_delta_px": 1.0, "tail_delta_px": 5.0,
        }
        # Both direct answers are wrong by 3 px. Their three-edge alternatives
        # are exact where the complete path fits: stereo 5-7-2=-4, flow -7+2+7=2.
        field_dx = {"D0": -7, "D1": -7, "D1reverse": 7,
                    "FL": 5, "FR": 2, "FRbw": -2}
        true_dx = {"stereo": -4, "flow": 2}
        gt_fields = {}
        expected_q = np.ones((h, w), bool)
        expected_q[0, 0] = False       # GT-invalid query.
        expected_q[10, 10] = False     # Already repaired by a C0 alternative.

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            provider = Mock()
            gt_paths = {}
            for task, unit_seconds in (("stereo", 1.0), ("flow", 2.0)):
                arrays, ledger = {}, []
                for edge_id, (edge_task, source, target, _) in EDGES.items():
                    if edge_task != task:
                        continue
                    for op_index in range(3):
                        field = np.zeros((2, h, w), np.float32)
                        field[0] = field_dx[edge_id]
                        if edge_id == DIRECT[task] and op_index == 1:
                            field[:, 10, 10] = (true_dx[task], 0)
                        key = f"{edge_id}_{op_index}"
                        arrays[key] = field
                        arrays[f"{key}_uncertainty"] = np.zeros((h, w), np.float32)
                        ledger.append({
                            "id": key, "kind": "edge", "edge": edge_id,
                            "source": source, "target": target, "task": task,
                            "operator": config["operators"][op_index],
                            "charged_seconds": unit_seconds, "forward_calls": 1,
                            "peak_allocated_bytes": int(unit_seconds * 1000),
                            "peak_reserved_bytes": int(unit_seconds * 2000),
                        })
                for index in range(9):
                    key = f"rematch_{index}"
                    arrays[key] = arrays[f"{DIRECT[task]}_0"].copy()
                    valid = np.ones((h, w), bool)
                    valid[:, 0] = False
                    arrays[f"{key}_valid"] = valid
                    ledger.append({"id": key, "kind": "rematch", "task": task,
                                   "charged_seconds": unit_seconds, "forward_calls": 1,
                                   "peak_allocated_bytes": int(unit_seconds * 1000),
                                   "peak_reserved_bytes": int(unit_seconds * 2000)})
                self.assertEqual(len(ledger), 18)
                artifact = directory / "predictions" / f"{case['id']}_{task}.npz"
                save_npz(artifact, **arrays)
                save_json(artifact.with_suffix(".json"), {"ledger": ledger})
                gt = np.zeros((4, 2, h, w), np.float32)
                gt[:, 0] = true_dx[task]
                gt[:, :, 0, 0] = np.nan
                gt_fields[task] = gt
                gt_paths[task] = directory / f"synthetic_{task}_gt.bin"
                gt_paths[task].write_bytes(gt.tobytes())

            def read_gt(task, scene, frame, *, roi_xyhw):
                self.assertEqual((scene, frame), ("0001", 1))
                self.assertEqual(roi_xyhw, (768, 380, h, w))
                # This task's fixed candidates must be materialized before its
                # evaluation labels become available.
                candidate_file = directory / "predictions" / f"{case['id']}_{task}_candidates.npz"
                self.assertTrue(candidate_file.is_file())
                return gt_fields[task]

            provider.read_gt.side_effect = read_gt
            provider.gt_path.side_effect = lambda task, scene, frame: gt_paths[task]
            with patch("stablebridge.quartet_experiment.gt_composed_diagnostic",
                       return_value=({"status": "test_omitted"}, {})) as diagnostic:
                results = evaluate_case(provider, case, config, directory)

            self.assertEqual([result["task"] for result in results], ["stereo", "flow"])
            self.assertEqual(provider.read_gt.call_count, 2)
            self.assertEqual(diagnostic.call_count, 2)
            for call in diagnostic.call_args_list:
                np.testing.assert_array_equal(call.args[-1], expected_q.ravel())

            for result in results:
                task = result["task"]
                with self.subTest(task=task):
                    capacity, cost = result["capacity"], result["cost"]
                    self.assertEqual(capacity["status"], "ok")
                    self.assertEqual(capacity["counts"]["dense_queries"], h * w)
                    self.assertEqual(capacity["counts"]["gt_valid"], h * w - 1)
                    self.assertEqual(capacity["counts"]["unresolved_E0_oracle"], h * w - 2)
                    self.assertEqual(capacity["counts"]["new_candidates_per_arm"], 9)
                    self.assertLess(capacity["path"]["computable_coverage_pct"], 100)
                    self.assertLess(capacity["rematch_valid_coverage_pct"], 100)
                    self.assertGreater(capacity["primary"]["newly_repairable_E1_oracle_pct"], 0)
                    self.assertLess(capacity["primary"]["newly_repairable_E1_oracle_pct"], 100)
                    self.assertEqual(capacity["primary"]["rematch_repair_rate_pct"], 0)
                    self.assertGreater(capacity["primary"]["paired_delta_px"], 0)
                    self.assertEqual(sum(row["pixels"] for row in capacity["states"].values()), h * w - 2)

                    self.assertEqual(cost["path_forward_calls"], 9)
                    self.assertEqual(cost["rematch_forward_calls"], 9)
                    self.assertEqual(cost["physical_joint_inference_forward_calls"], 36)
                    self.assertEqual(cost["physical_joint_inference_seconds"], 54)
                    self.assertAlmostEqual(cost["standalone_path_incremental_seconds"] - cost["compose_seconds"],
                                           15 if task == "stereo" else 12)
                    self.assertEqual(cost["standalone_rematch_incremental_seconds"],
                                     9 if task == "stereo" else 18)
                    self.assertEqual(cost["C0_seconds"], 3 if task == "stereo" else 6)
                    self.assertEqual(cost["peak_allocated_bytes"], 2000)
                    self.assertEqual(cost["peak_reserved_bytes"], 4000)
                    self.assertEqual(cost["additional_observation_wait_frames"], int(task == "stereo"))
                    self.assertEqual(result["actionable_E1"], "not_estimated")
                    self.assertEqual(result["trusted_E1"], "not_calibrated")

                    metrics_path = directory / "metrics" / f"{case['id']}_{task}.json"
                    self.assertEqual(json.loads(metrics_path.read_text())["capacity"], capacity)
                    with np.load(metrics_path.with_suffix(".npz")) as metrics:
                        np.testing.assert_array_equal(metrics["Q"], expected_q.ravel())
                        np.testing.assert_array_equal(metrics["unresolved_E0_oracle"], expected_q.ravel())
                        self.assertEqual(metrics["path_candidate_error"].shape, (9, h * w))
                        self.assertEqual(metrics["path_vertical_residual"].shape, (9, h * w))
                        self.assertEqual(metrics["CR_min_error"].shape, (h * w,))
                        # Paths leaving the crop still belong to the fixed Q
                        # cohort and retain the C0 oracle capacity.
                        unavailable = metrics["Q"] & ~metrics["path_eligible"].any(axis=0)
                        self.assertTrue(unavailable.any())
                        np.testing.assert_array_equal(metrics["CP_min_error"][unavailable],
                                                      metrics["C0_min_error"][unavailable])

                    candidate_path = directory / "predictions" / f"{case['id']}_{task}_candidates.npz"
                    self.assertEqual(result["candidate_artifact_sha256"], sha256(candidate_path))
                    self.assertEqual(result["gt_file_sha256"], sha256(gt_paths[task]))
                    with np.load(candidate_path) as candidates:
                        self.assertEqual(candidates["path_coordinates"].shape, (9, 4, h * w, 2))
                        self.assertEqual(candidates["path_direct_disagreement_px"].shape, (9, h * w))
                        self.assertEqual(candidates["wrong_eligible"].shape, (9, h * w))


if __name__ == "__main__":
    unittest.main()
