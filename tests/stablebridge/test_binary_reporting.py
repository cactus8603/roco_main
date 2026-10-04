"""Native metric denominators, scene weighting and fixed binary calibration."""
import json
import unittest

import numpy as np

from stablebridge.binary_reporting import (
    aggregate, apply_policy, choose_threshold, confirmation_gate, evaluate_case,
)


def case(e0, ea, score=None, *, scene="a", split="calibration", eligible=None, valid=None):
    e0, ea = np.asarray(e0, float), np.asarray(ea, float)
    return {"e0": e0, "ea": ea, "score": np.asarray(score if score is not None else np.arange(e0.size), float),
            "eligible": np.ones(e0.shape, bool) if eligible is None else np.asarray(eligible, bool),
            "valid": np.ones(e0.shape, bool) if valid is None else np.asarray(valid, bool),
            "metadata": {"task": "flow", "scene": scene, "split": split}}


def evaluated(c, accept=None):
    return evaluate_case(c["e0"], c["ea"], c["valid"], c["eligible"],
                         c["eligible"] if accept is None else np.asarray(accept, bool), c["metadata"])


class BinaryMetricsTests(unittest.TestCase):
    def test_full_valid_severe_denominator_and_conditional_damage(self):
        c = case([.5, .5, 2, 4, 99], [4.5, .5, .5, 4, 0], valid=[1, 1, 1, 1, 0])
        r = evaluated(c, [1, 0, 1, 0, 1])
        m = r["metrics"]
        self.assertEqual(r["counts"]["valid_pixels"], 4)
        self.assertEqual(m["update_fraction"], .5)
        self.assertEqual(m["severe_harm_fraction"], .25)  # 1/4, not 1/2 accepted.
        self.assertEqual(m["damage_good_fraction"], .5)  # 1/2 original good.
        self.assertEqual(m["repair_rate"], .5)  # 1/2 original bad.
        self.assertEqual(m["benefit_mass"], .375)
        self.assertEqual(m["harm_mass"], 1.)
        self.assertEqual(m["gain_px"], -.625)
        self.assertEqual(m["gain_px"], m["benefit_mass"]-m["harm_mass"])
        self.assertEqual(r["sums"]["harm_px"], 4.)
        self.assertEqual(r["denominators"]["damage_good_fraction"], "original_good_pixels")

    def test_empty_good_is_null_and_scene_macro_omission_is_reported(self):
        a = evaluated(case([2, 3], [1, 2], scene="a"))
        b = evaluated(case([.5, .5], [2, .5], scene="b"))
        self.assertIsNone(a["metrics"]["damage_good_fraction"])
        summary = aggregate([a, b])
        d = summary["metrics"]["damage_good_fraction"]
        self.assertEqual(d["scene_macro"], .5)
        self.assertEqual(d["omitted_empty_cases"], 1)
        self.assertEqual(d["omitted_empty_scenes"], 1)
        self.assertEqual(d["contributing_scenes"], 1)
        self.assertEqual(d["denominator_pixels"], 2)
        self.assertIsNone(b["metrics"]["repair_rate"])

    def test_tails_are_native_and_include_untouched_pixels(self):
        c = case([.5]*99+[2], [.5]*99+[1002])
        accept = np.zeros(100, bool); accept[-1] = True
        r = evaluated(c, accept)
        self.assertEqual(r["metrics"]["worst_harm_px"], 1000.)
        self.assertEqual(r["metrics"]["harm_p95_px"], 0.)
        self.assertAlmostEqual(r["metrics"]["harm_p99_px"], 10.)
        self.assertEqual(r["metrics"]["harm_mass"], 10.)
        self.assertEqual(r["metrics"]["gain_px"], -10.)
        self.assertAlmostEqual(r["metrics"]["error_p99_px"], np.percentile([.5]*99+[1002], 99))

    def test_unaccepted_invalid_candidate_preserves_identity_and_failure_is_not_dropped(self):
        c = case([.5, 2], [np.nan, 1], [np.nan, 2])
        mask = apply_policy(c["score"], c["eligible"], {"keep_identity": False, "threshold": 1.})
        good = evaluated(c, mask)
        self.assertEqual(good["metrics"]["error_px"], .75)
        bad = evaluated(c)
        self.assertEqual(bad["counts"]["nonfinite_output_pixels"], 1)
        self.assertIsNone(bad["metrics"]["gain_px"])
        report = aggregate([good, bad])
        self.assertIsNone(report["metrics"]["gain_px"]["scene_macro"])
        self.assertEqual(report["metrics"]["gain_px"]["failed_cases"], 1)
        json.dumps(report, allow_nan=False)

    def test_scene_first_not_case_weighted_and_mixed_cohorts_rejected(self):
        rows = [evaluated(case([2], [1], scene="a")), evaluated(case([4], [1], scene="a")),
                evaluated(case([10], [1], scene="b"))]
        self.assertEqual(aggregate(rows)["metrics"]["gain_px"]["scene_macro"], 5.5)
        rows[-1]["metadata"]["task"] = "stereo"
        with self.assertRaisesRegex(ValueError, "mixed task"):
            aggregate(rows)
        with self.assertRaisesRegex(ValueError, "subset"):
            evaluate_case([1], [2], np.array([True]), np.array([False]), np.array([True]))


class BinaryCalibrationTests(unittest.TestCase):
    def test_feasible_high_score_subset_and_identity_allcandidate_baselines(self):
        # Allcandidate damages an original-good pixel; higher score keeps that
        # pixel unchanged and repairs 1/4 of the image with no harm.
        c = case([.5, .5, 4, 4], [4.5, .5, 1, 4], [0, 0, 1, 0])
        result = choose_threshold([c])
        self.assertFalse(result["policy"]["keep_identity"])
        self.assertGreaterEqual(result["policy"]["threshold"], 0.)
        self.assertEqual(result["summary"]["metrics"]["gain_px"]["scene_macro"], .75)
        self.assertEqual(result["summary"]["metrics"]["damage_good_fraction"]["scene_macro"], 0.)
        self.assertFalse(result["baselines"]["allcandidate"]["harm_feasible"])
        self.assertTrue(result["baselines"]["identity"]["harm_feasible"])
        self.assertEqual(len(result["grid_evaluations"]), 17)
        # Multiple quantiles produce identical outputs: larger threshold wins.
        feasible = [r["policy"]["threshold"] for r in result["grid_evaluations"] if r["active_feasible"]]
        self.assertEqual(result["policy"]["threshold"], max(feasible))
        json.dumps(result, allow_nan=False)

    def test_grid_uses_eligible_scores_without_gt_validity_and_strict_comparison(self):
        c = case([.5, 3, 100], [.5, 1, 0], [0, 1, 100], valid=[1, 1, 0])
        result = choose_threshold([c])
        self.assertEqual(result["pooled_eligible_finite_scores"], 3)
        thresholds = [g["policy"]["threshold"] for g in result["grid_evaluations"] if not g["policy"]["keep_identity"]]
        self.assertEqual(max(thresholds), 100.)
        self.assertTrue(any(1 < t < 100 for t in thresholds))
        self.assertEqual(apply_policy([1, 2, np.nan, np.inf], np.ones(4, bool),
                                     {"threshold": 1., "keep_identity": False}).tolist(), [False, True, False, False])

    def test_inactive_or_harmful_calibration_returns_json_safe_identity(self):
        harmful = choose_threshold([case([.5, .5], [10, 20], [1, 2])])
        self.assertTrue(harmful["policy"]["keep_identity"])
        self.assertIsNone(harmful["policy"]["threshold"])
        self.assertEqual(harmful["summary"]["metrics"]["update_fraction"]["scene_macro"], 0.)
        no_scores = choose_threshold([case([.5, 2], [1, 1], [np.nan, np.inf])])
        self.assertTrue(no_scores["policy"]["keep_identity"])
        self.assertEqual(no_scores["pooled_eligible_finite_scores"], 0)
        # Positive but under the registered 0.01 native-pixel gain floor.
        tiny = choose_threshold([case([.5, 2], [.5, 1.99], [0, 1])])
        self.assertTrue(tiny["policy"]["keep_identity"])
        json.dumps([harmful, no_scores, tiny], allow_nan=False)

    def test_update_floor_and_unassessable_good_damage_do_not_pass(self):
        e0 = np.full(2000, .5); e0[-1] = 101.
        ea = e0.copy(); ea[-1] = 1.
        scores = np.zeros(2000); scores[-1] = 1.
        sparse = choose_threshold([case(e0, ea, scores)])
        self.assertTrue(sparse["policy"]["keep_identity"])
        proposal = sparse["grid_evaluations"][0]
        self.assertTrue(proposal["checks"]["minimum_gain"])
        self.assertFalse(proposal["checks"]["minimum_update"])
        no_good = choose_threshold([case([4, 4], [4, 1], [0, 1])])
        self.assertTrue(no_good["policy"]["keep_identity"])
        self.assertIsNone(no_good["summary"]["metrics"]["damage_good_fraction"]["scene_macro"])

    def test_severe_harm_at_three_is_not_over_three(self):
        r = evaluated(case([.5, .5], [3.5, 3.5001]))
        self.assertEqual(r["metrics"]["severe_harm_fraction"], .5)

    def test_calibration_refuses_confirmation_or_mixed_split(self):
        with self.assertRaisesRegex(ValueError, "calibration split"):
            choose_threshold([case([.5, 4], [.5, 1], [0, 1], split="confirmation")])
        with self.assertRaisesRegex(ValueError, "mixed split"):
            choose_threshold([case([.5], [.5]), case([.5], [.5], split="confirmation")])

    def test_confirmation_requires_three_positive_scenes_and_harm_bounds(self):
        rows = [evaluated(case([.5, 4], [.5, 1], scene=str(i), split="confirmation"), [0, 1])
                for i in range(3)]
        rows.append(evaluated(case([.5, 4], [.5, 4], scene="3", split="confirmation"), [0, 0]))
        gate = confirmation_gate(aggregate(rows))
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["positive_scenes"], 3)
        self.assertFalse(gate["statistical_significance_claim"])
        self.assertFalse(confirmation_gate(aggregate(rows[:3]))["passed"])


if __name__ == "__main__":
    unittest.main()
