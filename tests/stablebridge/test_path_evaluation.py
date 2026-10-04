"""Analytical fixed-query capacity tests; no model inference or dataset access."""
import json
import unittest

import numpy as np

from stablebridge.path_evaluation import evaluate_capacity, STATE_NAMES


def bank(x, y=0.):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return np.stack((x, np.broadcast_to(y, x.shape)), axis=-1)


def inputs(c0, path, rematch=None, wrong=None):
    path = np.asarray(path)
    j, n = path.shape[:2]
    return dict(c0=c0, path_raw=path.copy(), path_candidates=path.copy(),
                path_computable=np.ones((j, n), bool), path_eligible=np.ones((j, n), bool),
                rematch=path.copy() if rematch is None else rematch,
                rematch_valid=np.ones((j, n), bool),
                wrong=bank(np.full((j, n), 10.)) if wrong is None else wrong,
                wrong_computable=np.ones((j, n), bool), wrong_eligible=np.ones((j, n), bool),
                gt4=np.zeros((4, n, 2)))


class CapacityTests(unittest.TestCase):
    def test_primary_compares_oracle_unions_not_identity_or_plain_rematch(self):
        args = inputs(bank([[10., 10.], [2., 2.]]), bank([5., .5]), bank([7., 1.5]))
        summary, arrays = evaluate_capacity(**args)
        np.testing.assert_array_equal(arrays["C0_min_error"], [2., 2.])
        np.testing.assert_array_equal(arrays["CP_min_error"], [2., .5])
        np.testing.assert_array_equal(arrays["CR_min_error"], [2., 1.5])
        self.assertEqual(summary["primary"]["paired_delta_px"], .5)
        self.assertEqual(summary["primary"]["path_gain_over_c0_px"], .75)
        self.assertEqual(summary["path"]["per_candidate"][0]["raw2d"]["signed_gain_px"], 7.25)
        self.assertEqual(summary["states"]["still_unresolved_no_improvement"]["pixels"], 1)

    def test_unavailable_queries_remain_in_denominator_and_raw_identity_fallback(self):
        args = inputs(bank([5., 5., 5., 5.]), bank([0., 0., np.nan, 0.]))
        args["path_computable"][0, 2] = False
        args["path_eligible"][0, 2:] = False
        summary, arrays = evaluate_capacity(**args)
        self.assertEqual(summary["counts"]["unresolved_E0_oracle"], 4)
        self.assertEqual(summary["primary"]["newly_repairable_E1_oracle_pct"], 50.)
        self.assertEqual(summary["path"]["computable_coverage_pct"], 75.)
        self.assertEqual(summary["path"]["eligible_coverage_pct"], 50.)
        self.assertEqual(summary["path"]["per_candidate"][0]["raw2d"]["signed_gain_px"], 2.5)
        self.assertEqual(summary["raw_path_raw2d"]["all_Q"]["signed_gain_px"], 2.5)
        self.assertEqual(summary["raw_path_raw2d"]["eligible_only"]["signed_gain_px"], 5.)
        self.assertEqual(summary["raw_path_raw2d"]["eligible_only_candidate_pixels"], [2])
        np.testing.assert_array_equal(arrays["CP_min_error"], [0., 0., 5., 5.])

    def test_states_are_exhaustive_and_partial_improvement_is_unresolved(self):
        args = inputs(bank([[8., 8., 8., 8., 1.], [6., 6., 6., 6., 1.]]), bank([1., 2., 7., 0., 0.]))
        args["path_computable"][0, 3] = False
        args["path_eligible"][0, 3] = False
        summary, arrays = evaluate_capacity(**args)
        np.testing.assert_array_equal(arrays["state_code"], [1, 2, 3, 4, 0])
        self.assertEqual(sum(item["pixels"] for item in summary["states"].values()), 4)
        for name in STATE_NAMES[1:]:
            self.assertEqual(summary["states"][name], {"pixels": 1, "pct_of_fixed_Q": 25.})

    def test_raw_vertical_error_is_not_silently_projected_away(self):
        args = inputs(bank([10., 10.]), bank([0., 0.], y=8.))
        args["path_candidates"] = bank([0., 0.])
        args["path_eligible"][0, 1] = False
        summary, arrays = evaluate_capacity(**args)
        row = summary["path"]["per_candidate"][0]
        self.assertEqual(row["raw2d"]["error_px"], 9.)
        self.assertEqual(row["scored_candidate"]["error_px"], 5.)
        self.assertEqual(row["computable_raw2d_diagnostic"]["error_px"], 8.)
        self.assertEqual(row["absolute_vertical_residual_px"], 8.)
        np.testing.assert_array_equal(arrays["path_vertical_residual"], [[8., 8.]])

    def test_min4_whole_vectors_ties_nan_and_infinite_gt(self):
        args = inputs(bank([3., 3., 3., 3., 1.]), bank([1., 0., 0., 0., 1.]))
        args["gt4"][:, 0] = [[0., 2.], [2., 0.], [8., 8.], [9., 9.]]
        args["gt4"][0, 1, 0] = np.nan
        args["gt4"][0, 2, 0] = np.inf
        args["gt4"][:, 3, 0] = np.inf
        summary, arrays = evaluate_capacity(**args)
        np.testing.assert_array_equal(arrays["gt_valid"], [True, False, True, False, True])
        # Query 0 C0 exactly meets 1px; query 4 does too. Only query 2 is unresolved.
        np.testing.assert_array_equal(arrays["Q"], [False, False, True, False, False])
        self.assertEqual(arrays["path_candidate_error"][0, 0], 1.)
        self.assertEqual(summary["counts"]["unresolved_E0_oracle"], 1)
        json.dumps(summary, allow_nan=False)

    def test_wrong_raw_vertical_diagnostic_is_distinct_from_projection(self):
        args = inputs(bank([10., 10.]), bank([0., 0.]), wrong=bank([0., 0.]))
        args["wrong_raw"] = bank([0., 0.], y=[.5, 8.])
        args["wrong_eligible"][0, 1] = False
        summary, arrays = evaluate_capacity(**args)
        row = summary["wrong"]["per_candidate"][0]
        self.assertEqual(row["raw2d"]["error_px"], 5.25)
        self.assertEqual(row["scored_candidate"]["error_px"], 5.)
        self.assertEqual(row["computable_raw2d_diagnostic"]["error_px"], 4.25)
        self.assertEqual(summary["primary"]["wrong_repair_rate_pct"], 50.)
        np.testing.assert_array_equal(arrays["wrong_raw2d_error"], [[.5, 8.]])
        np.testing.assert_array_equal(arrays["wrong_error"], [[0., 0.]])
        np.testing.assert_array_equal(arrays["wrong_vertical_residual"], [[.5, 8.]])
        np.testing.assert_array_equal(arrays["wrong_candidate_adapter_shift"][..., 1], [[-.5, -8.]])

    def test_wrong_nonfinite_raw_fails_even_if_projected_candidate_is_finite(self):
        args = inputs(bank([3., 3.]), bank([0., 0.]), wrong=bank([0., 0.]))
        args["wrong_raw"] = bank([0., 0.], y=[0., np.nan])
        summary, arrays = evaluate_capacity(**args)
        self.assertEqual(summary["status"], "failed_prediction")
        self.assertEqual(summary["nonfinite_predictions_on_gt_valid"]["wrong_computable"], 1)
        self.assertEqual(summary["nonfinite_predictions_on_gt_valid"]["wrong_candidate_eligible"], 0)
        self.assertEqual(summary["counts"]["unresolved_E0_oracle"], 2)
        self.assertEqual(summary["primary"]["wrong_repair_rate_pct"], 100.)
        self.assertTrue(np.isinf(arrays["wrong_raw2d_error"][0, 1]))
        json.dumps(summary, allow_nan=False)

    def test_nonfinite_claimed_predictions_fail_without_reducing_cohort(self):
        args = inputs(bank([3., 3.]), bank([0., np.nan]))
        args["rematch"] = bank([2., 2.])
        summary, arrays = evaluate_capacity(**args)
        self.assertEqual(summary["status"], "failed_prediction")
        self.assertEqual(summary["counts"]["unresolved_E0_oracle"], 2)
        self.assertEqual(summary["primary"]["newly_repairable_E1_oracle_pct"], 50.)
        self.assertEqual(summary["nonfinite_predictions_on_gt_valid"]["path_raw_computable"], 1)
        self.assertIsNone(summary["path"]["per_candidate"][0]["raw2d"]["error_px"])
        np.testing.assert_array_equal(arrays["CP_min_error"], [0., 3.])
        json.dumps(summary, allow_nan=False)

    def test_nonfinite_c0_is_explicit_failure_even_if_other_c0_is_finite(self):
        args = inputs(bank([[3., np.nan], [2., 2.]]), bank([0., 0.]))
        summary, arrays = evaluate_capacity(**args)
        self.assertEqual(summary["status"], "failed_prediction")
        self.assertEqual(summary["counts"]["unresolved_E0_oracle"], 2)
        np.testing.assert_array_equal(arrays["C0_min_error"], [2., 2.])
        self.assertIsNone(summary["path"]["per_candidate"][0]["raw2d"]["signed_gain_px"])
        json.dumps(summary, allow_nan=False)

    def test_empty_cohort_is_null_not_zero_performance(self):
        summary, arrays = evaluate_capacity(**inputs(bank([1., 0.]), bank([0., 0.])))
        self.assertEqual(summary["status"], "empty")
        self.assertEqual(summary["counts"]["unresolved_E0_oracle"], 0)
        self.assertIsNone(summary["primary"]["paired_delta_px"])
        self.assertIsNone(summary["primary"]["newly_repairable_E1_oracle_pct"])
        self.assertIsNone(summary["path"]["computable_coverage_pct"])
        self.assertIsNone(summary["states"]["newly_repairable_E1_oracle"]["pct_of_fixed_Q"])
        self.assertFalse(arrays["Q"].any())
        json.dumps(summary, allow_nan=False)

    def test_common_eligible_reports_both_rates_and_fixed_query_coverage(self):
        args = inputs(bank([3., 3., 3., 3.]), bank([0., 0., 0., 0.]), wrong=bank([0., 2., 0., 0.]))
        args["path_eligible"][0, 3] = False
        args["wrong_eligible"][0, 2:] = False
        summary, arrays = evaluate_capacity(**args)
        self.assertEqual(summary["primary"]["primary_minus_wrong_repair_rate_pp"], 50.)
        self.assertEqual(summary["common_eligible"], {
            "pixels": 2, "coverage_pct_of_fixed_Q": 50., "primary_repair_rate_pct": 100.,
            "wrong_repair_rate_pct": 50., "primary_minus_wrong_repair_rate_pp": 50.})
        np.testing.assert_array_equal(arrays["common_eligible_Q"], [True, True, False, False])

    def test_raw_candidate_macro_is_equal_candidates_not_oracle_selection(self):
        args = inputs(bank([4., 4.]), bank([[0., 8.], [2., 2.]]))
        summary, _ = evaluate_capacity(**args)
        self.assertEqual(summary["primary"]["path_gain_over_c0_px"], 3.)
        self.assertEqual(summary["path"]["fixed_candidate_macro"]["raw2d"]["signed_gain_px"], 1.)
        self.assertEqual(summary["path"]["fixed_candidate_macro"]["raw2d"]["harm_over_delta_pct"], 25.)

    def test_invalid_masks_and_bank_shapes_are_rejected(self):
        args = inputs(bank([3., 3.]), bank([0., 0.]))
        args["path_computable"][0, 0] = False
        with self.assertRaisesRegex(ValueError, "computable"):
            evaluate_capacity(**args)
        args = inputs(bank([3., 3.]), bank([0., 0.]))
        args["rematch"] = bank([[0., 0.], [0., 0.]])
        with self.assertRaisesRegex(ValueError, "candidate count"):
            evaluate_capacity(**args)


if __name__ == "__main__":
    unittest.main()
