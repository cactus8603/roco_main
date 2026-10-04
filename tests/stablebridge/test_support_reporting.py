"""Actual-output scoring, noninterchangeable denominators and S04 fixed gates."""
import json
import unittest

import numpy as np

from stablebridge.support_reporting import (aggregate_support_results,
    evaluate_support_case, render_report)


ARMS = ["identity", "no_support", "raw", "processed", "wrong", "gt_diagnostic",
        "propagation_processed"]


def field(x):
    x = np.asarray(x, dtype=float).reshape(1, -1)
    return np.stack((x, np.zeros_like(x)))


def configuration(scenes=("0006", "0008"), profiles=("clean",)):
    return {
        "tasks": ["stereo"], "profiles": [{"name": name} for name in profiles],
        "splits": {"confirmation": {scene: [1] for scene in scenes}},
        "evaluation": {"arms": ARMS},
        "gate": {
            "minimum_Q_per_task": 1, "minimum_Q_per_scene": 1,
            "minimum_supported_scenes": len(scenes), "required_scenes": len(scenes),
            "minimum_positive_scenes": len(scenes),
            "vs_no_support_Q_gain_px": 0.05, "vs_no_support_Q_gain_fraction": 0.05,
            "vs_no_support_repair_gain_pp": 2.0, "processed_vs_raw_Q_gain_px": 0.02,
            "processed_vs_wrong_Q_gain_px": 0.02, "processed_vs_wrong_repair_gain_pp": 1.0,
            "processed_vs_propagation_Q_gain_px": 0.02, "processed_vs_propagation_repair_gain_pp": 1.0,
            "maximum_all_error_increase_px": 0.01, "maximum_initial_good_damage_pct": 1.0,
            "maximum_severe_harm_pct": 0.5,
        },
    }


def case(scene, profile="clean", hard_count=2, gain=1.5, initial_error=2.0, split="confirmation"):
    initial = field([initial_error] * hard_count + [0.0])
    current = field([initial_error - gain] * hard_count + [0.0])
    outputs = {name: initial.copy() for name in ARMS if name != "identity"}
    outputs["processed"] = current
    outputs["gt_diagnostic"] = field([0.25] * hard_count + [0.0])
    q = np.array([[True] * hard_count + [False]])
    metadata = {"task": "stereo", "scene": scene, "frame": 1, "profile": profile, "split": split}
    return evaluate_support_case(initial, outputs, np.zeros((4,) + initial.shape), q, metadata=metadata)


class SupportReportingTests(unittest.TestCase):
    def test_fixed_denominators_partial_improvement_and_harm_sign(self):
        initial = field([10, 2, 0.5])
        outputs = {"processed": field([3, 1, 7]), "no_support": initial}
        q = np.array([[True, True, False]])
        result = evaluate_support_case(initial, outputs, np.zeros((4,) + initial.shape), q)
        metrics = result["arms"]["processed"]
        self.assertEqual(metrics["hard"]["pixels"], 2)
        self.assertEqual(metrics["hard"]["repair_rate_pct"], 50)
        self.assertEqual(metrics["hard"]["partial_improvement_pct"], 50)
        self.assertEqual(metrics["hard"]["gain_px"], 4)
        self.assertEqual(metrics["initial_good"]["initial_good_damage_pct"], 100)
        self.assertAlmostEqual(metrics["all"]["tail_harm_over_5px_pct"], 100 / 3)
        self.assertAlmostEqual(metrics["all"]["harm_over_3px_pct"], 100 / 3)
        self.assertEqual(result["contrasts"]["processed_vs_no_support"]["hard"]["repair_delta_pp"], 50)

    def test_nonfinite_output_does_not_shrink_Q_or_claim_full_repair(self):
        initial = field([2, 2, 0])
        outputs = {"processed": field([np.nan, 0.5, 0]), "no_support": initial}
        result = evaluate_support_case(initial, outputs, np.zeros((4,) + initial.shape),
                                       np.array([[True, True, False]]))
        metrics = result["arms"]["processed"]["hard"]
        self.assertEqual(result["status"], "failed_prediction")
        self.assertEqual(metrics["pixels"], 2)
        self.assertEqual(metrics["nonfinite_output_pixels"], 1)
        self.assertEqual(metrics["repair_rate_pct"], 50)
        self.assertIsNone(metrics["error_px"])
        self.assertIsNone(metrics["gain_px"])
        json.dumps(result, allow_nan=False)

    def test_min4_uses_whole_vector_and_rejects_invalid_fixed_Q(self):
        initial = np.array([[[10.0]], [[0.0]]])
        gt = np.array([[0, 3], [4, 0], [0, 3], [4, 0]], dtype=float).reshape(4, 2, 1, 1)
        result = evaluate_support_case(initial, {"processed": np.zeros_like(initial)}, gt,
                                       np.ones((1, 1), bool))
        self.assertEqual(result["arms"]["processed"]["hard"]["error_px"], 3)
        gt[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            evaluate_support_case(initial, {}, gt, np.ones((1, 1), bool))

    def test_scene_macro_is_not_pooled_and_relative_gain_uses_macro_baseline(self):
        rows = [case("0006", "clean", 1, 1, 4), case("0006", "noise", 1, 3, 4),
                case("0008", "clean", 100, 0, 4), case("0008", "noise", 100, 0, 4)]
        report = aggregate_support_results(rows, configuration(profiles=("clean", "noise")))
        result = report["tasks"]["stereo"]["splits"]["confirmation"]
        macro = result["scene_macro"]["contrasts"]["processed_vs_no_support"]["hard"]
        pooled = result["pooled"]["contrasts"]["processed_vs_no_support"]["hard"]
        self.assertEqual(macro["gain_px"], 1)
        self.assertEqual(macro["relative_gain_fraction"], 0.25)
        self.assertAlmostEqual(pooled["gain_px"], 4 / 202)
        self.assertEqual(result["counts"]["hard"], 202)
        self.assertEqual(result["scenes"]["0006"]["counts"]["hard"], 2)

    def test_empty_registered_Q_case_is_explicit_and_identically_excluded(self):
        rows = [case("0006", "clean", 2), case("0006", "noise", 0)]
        report = aggregate_support_results(rows, configuration(scenes=("0006",), profiles=("clean", "noise")))
        result = report["tasks"]["stereo"]
        confirmation = result["splits"]["confirmation"]
        self.assertEqual(confirmation["empty_Q_cases"], 1)
        self.assertEqual(confirmation["scene_macro"]["arms"]["processed"]["hard"]["error_px"], 0.5)
        self.assertEqual(confirmation["pooled"]["arms"]["processed"]["hard"]["error_px"], 0.5)
        self.assertEqual(confirmation["scenes"]["0006"]["arms"]["processed"]["hard"]["empty_cases"], 1)
        self.assertEqual(result["confirmation_gate"]["status"], "PASS")

    def test_entire_empty_scene_is_never_silently_removed(self):
        rows = [case("0006", hard_count=2), case("0008", hard_count=0)]
        config = configuration()
        config["gate"]["minimum_supported_scenes"] = 1
        report = aggregate_support_results(rows, config)
        result = report["tasks"]["stereo"]
        self.assertIsNone(result["splits"]["confirmation"]["scene_macro"]["arms"]["processed"]["hard"]["error_px"])
        self.assertEqual(result["confirmation_gate"]["status"], "INCONCLUSIVE_UNDEFINED_METRIC")

    def test_actual_outputs_can_pass_and_privileged_oracle_cannot_override_failure(self):
        rows = [case("0006"), case("0008")]
        report = aggregate_support_results(rows, configuration())
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["tasks"]["stereo"]["confirmation_gate"]["GT_diagnostic"]["passed"])
        failed_rows = [case("0006", gain=0), case("0008", gain=0)]
        for row in failed_rows:
            oracle_metrics = {region: dict(value) for region, value in row["arms"]["gt_diagnostic"].items()}
            row["candidate_oracles"] = {"processed": oracle_metrics}
        failed = aggregate_support_results(failed_rows, configuration())
        self.assertEqual(failed["tasks"]["stereo"]["confirmation_gate"]["status"], "FAIL")
        self.assertTrue(failed["tasks"]["stereo"]["confirmation_gate"]["GT_diagnostic"]["passed"])
        diagnostic = failed["tasks"]["stereo"]["splits"]["confirmation"]["scene_macro"]["candidate_oracles"]
        self.assertEqual(diagnostic["processed"]["hard"]["repair_rate_pct"], 100)
        self.assertIn("尚未通過", render_report(failed))

    def test_failed_nonempty_case_is_never_removed_from_macro_or_gate(self):
        rows = [case("0006"), case("0008")]
        initial = field([2, 2, 0])
        outputs = {name: initial.copy() for name in ARMS if name != "identity"}
        outputs["processed"][0, 0, 0] = np.inf
        rows[1] = evaluate_support_case(initial, outputs, np.zeros((4,) + initial.shape),
            np.array([[True, True, False]]), metadata=rows[1]["metadata"])
        report = aggregate_support_results(rows, configuration())
        result = report["tasks"]["stereo"]
        self.assertIsNone(result["splits"]["confirmation"]["scene_macro"]["arms"]["processed"]["hard"]["error_px"])
        self.assertEqual(result["confirmation_gate"]["status"], "FAILED_NONFINITE_PREDICTION")

    def test_missing_confirmation_smoke_is_inconclusive_and_missing_case_not_dropped(self):
        config = configuration(scenes=("0006",))
        config["splits"] = {"train": {"0006": [1]}}
        smoke = aggregate_support_results([case("0006", split="train")], config)
        self.assertEqual(smoke["tasks"]["stereo"]["confirmation_gate"]["status"], "INCONCLUSIVE_INCOMPLETE_PANEL")
        json.dumps(smoke, allow_nan=False)
        self.assertIn("工程 smoke", render_report(smoke))
        self.assertNotIn("本版尚未通過", render_report(smoke))
        missing = aggregate_support_results([case("0006")], configuration())
        result = missing["tasks"]["stereo"]
        self.assertEqual(len(result["splits"]["confirmation"]["missing_cases"]), 1)
        self.assertEqual(result["confirmation_gate"]["status"], "INCONCLUSIVE_INCOMPLETE_PANEL")
        with self.assertRaises(ValueError):
            aggregate_support_results([case("0006"), case("0006")], configuration())

    def test_extra_region_cannot_replace_primary_and_all_contrasts_use_same_mask(self):
        initial = field([3, 3, 0])
        gt = np.zeros((4,) + initial.shape)
        q = np.array([[True, True, False]])
        result = evaluate_support_case(initial, {"processed": field([0, 2, 0]), "no_support": initial}, gt, q,
                                       extra_regions={"hard_outside_guard": np.array([[False, True, False]])})
        self.assertEqual(result["arms"]["processed"]["hard"]["pixels"], 2)
        self.assertEqual(result["contrasts"]["processed_vs_no_support"]["hard_outside_guard"]["pixels"], 1)
        with self.assertRaises(ValueError):
            evaluate_support_case(initial, {}, gt, q, extra_regions={"hard": q})


if __name__ == "__main__":
    unittest.main()
