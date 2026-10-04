"""Scene weighting, fixed registration and resource qualification contracts."""
import copy
import json
import unittest

import numpy as np

from stablebridge.path_evaluation import evaluate_capacity
from stablebridge.quartet_reporting import aggregate_capacity


def fixture():
    n = 200
    def bank(x):
        value = np.zeros((1, n, 2))
        value[..., 0] = x
        return value
    valid = np.ones((1, n), bool)
    capacity, _ = evaluate_capacity(bank(3), bank(0), bank(0), valid, valid,
                                    bank(2), valid, bank(3), valid, valid, np.zeros((4, n, 2)))
    config = {"scientific_gate": True,
              "scene_frames": [[f"{i:04d}", 1] for i in range(4)],
              "profiles": [{"name": name} for name in ("clean", "noise")], "gate": {}}
    rows = [{"case": {"scene": scene, "frame": frame, "profile": profile}, "task": task,
             "capacity": copy.deepcopy(capacity),
             "cost": {"path_to_rematch_seconds_ratio": 1., "wait_seconds": None,
                      "additional_observation_wait_frames": 1 if task == "stereo" else 0}}
            for task in ("stereo", "flow") for scene, frame in config["scene_frames"]
            for profile in config["profiles"]]
    return rows, config


class QuartetReportingTests(unittest.TestCase):
    def test_complete_panel_passes_and_tasks_keep_independent_metrics(self):
        rows, config = fixture()
        result = aggregate_capacity(rows, config)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["capacity_pass"])
        self.assertEqual(result["tasks"]["stereo"]["total_unresolved_E0_oracle"], 1600)
        self.assertEqual(result["tasks"]["flow"]["scene_macro"]["paired_delta_px"], 2.)
        self.assertEqual(result["tasks"]["stereo"]["cost_scene_macro"]["additional_observation_wait_frames"], 1.)
        self.assertEqual(result["tasks"]["flow"]["cost_scene_macro"]["additional_observation_wait_frames"], 0.)
        self.assertIsNone(result["tasks"]["flow"]["cost_scene_macro"]["wait_seconds"])
        self.assertIn("raw_path_computable_raw2d_diagnostic_tail_harm_over_delta_pct", result["tasks"]["flow"]["scene_macro"])
        json.dumps(result, allow_nan=False)

    def test_scene_macro_equal_profiles_not_query_weighted(self):
        rows, config = fixture()
        for row in rows:
            if row["task"] == "stereo" and row["case"]["scene"] == "0000":
                clean = row["case"]["profile"]["name"] == "clean"
                row["capacity"]["counts"]["unresolved_E0_oracle"] = 100000 if clean else 1
                row["capacity"]["primary"]["paired_delta_px"] = 10. if clean else 0.
        result = aggregate_capacity(rows, config)
        self.assertEqual(result["tasks"]["stereo"]["scenes"]["0000"]["metrics"]["paired_delta_px"], 5.)
        self.assertEqual(result["tasks"]["stereo"]["scene_macro"]["paired_delta_px"], 2.75)
        self.assertEqual(result["tasks"]["flow"]["scene_macro"]["paired_delta_px"], 2.)

    def test_null_profile_is_incomplete_and_never_dropped(self):
        rows, config = fixture()
        rows[0]["capacity"]["primary"]["paired_delta_px"] = None
        result = aggregate_capacity(rows, config)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIsNone(result["capacity_pass"])
        self.assertIsNone(result["tasks"]["stereo"]["scene_macro"]["paired_delta_px"])
        self.assertIsNone(result["tasks"]["stereo"]["scenes"]["0000"]["metrics"]["paired_delta_px"])

    def test_missing_registration_and_duplicates_are_not_ignored(self):
        rows, config = fixture()
        result = aggregate_capacity(rows[1:], config)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(len(result["tasks"]["stereo"]["completeness"]["missing_case_ids"]), 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_capacity(rows + [rows[0]], config)

    def test_wrong_control_exactly_half_primary_fails_strict_gate(self):
        rows, config = fixture()
        for row in rows:
            row["capacity"]["primary"]["wrong_repair_rate_pct"] = 50.
        result = aggregate_capacity(rows, config)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["tasks"]["stereo"]["checks"]["wrong_below_fraction_of_primary"])
        self.assertEqual(result["tasks"]["stereo"]["wrong_to_primary_repair_ratio"], .5)

    def test_zero_paired_delta_or_common_gap_fails_strict_gate(self):
        for metric in ("delta", "gap"):
            with self.subTest(metric=metric):
                rows, config = fixture()
                for row in rows:
                    if metric == "delta":
                        row["capacity"]["primary"]["paired_delta_px"] = 0.
                    else:
                        row["capacity"]["common_eligible"]["primary_minus_wrong_repair_rate_pp"] = 0.
                self.assertEqual(aggregate_capacity(rows, config)["status"], "FAIL")

    def test_capacity_pass_is_preserved_with_resource_mismatch(self):
        rows, config = fixture()
        for row in rows:
            row["cost"]["path_to_rematch_seconds_ratio"] = 1.26
        result = aggregate_capacity(rows, config)
        self.assertEqual(result["status"], "INCONCLUSIVE_RESOURCE_MISMATCH")
        self.assertTrue(result["capacity_pass"])
        self.assertFalse(result["resource_matched"])

    def test_cost_bounds_are_inclusive_and_missing_cost_incomplete(self):
        for boundary in (.8, 1.25):
            rows, config = fixture()
            for row in rows:
                row["cost"]["path_to_rematch_seconds_ratio"] = boundary
            self.assertEqual(aggregate_capacity(rows, config)["status"], "PASS")
        rows[0]["cost"]["path_to_rematch_seconds_ratio"] = None
        result = aggregate_capacity(rows, config)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(result["capacity_pass"])
        self.assertIsNone(result["resource_matched"])

    def test_smoke_cannot_emit_scientific_pass(self):
        rows, config = fixture()
        config["scientific_gate"] = False
        result = aggregate_capacity(rows, config)
        self.assertEqual(result["status"], "FUNCTIONAL_ONLY")
        self.assertIsNone(result["capacity_pass"])
        self.assertIsNone(result["tasks"]["stereo"]["capacity_pass"])
        self.assertTrue(result["functional_complete"])

    def test_failed_prediction_status_blocks_otherwise_passing_oracle(self):
        rows, config = fixture()
        rows[0]["capacity"]["status"] = "failed_prediction"
        result = aggregate_capacity(rows, config)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertFalse(result["functional_complete"])


if __name__ == "__main__":
    unittest.main()
