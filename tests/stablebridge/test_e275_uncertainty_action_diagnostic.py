from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "research/action_bank_29_finalization_20261005/analyze_e275_uncertainty_action.py"
)
MODULE_NAME = "e275_uncertainty_action_diagnostic"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MODULE
SPEC.loader.exec_module(MODULE)


class E275UncertaintyActionUnitTests(unittest.TestCase):
    def test_average_rank_and_auroc_ties(self) -> None:
        ranks = MODULE.rankdata_average(np.asarray([2.0, 1.0, 2.0, 4.0]))
        np.testing.assert_allclose(ranks, [1.5, 0.0, 1.5, 3.0])
        self.assertEqual(
            MODULE.binary_auroc(
                np.asarray([0.0, 1.0, 2.0, 3.0]),
                np.asarray([False, False, True, True]),
            ),
            1.0,
        )
        self.assertIsNone(
            MODULE.binary_auroc(
                np.asarray([0.0, 1.0]), np.asarray([True, True])
            )
        )

    def test_metric_orientation_is_explicit(self) -> None:
        uncertainty = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        gains = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        metrics = MODULE.metric_matrices(
            uncertainty, gains, severe_harm_cutoff_raw_px=-1.5
        )
        self.assertAlmostEqual(metrics["spearman_signed_gain"][0, 0], 1.0)
        self.assertLess(metrics["spearman_harm_magnitude"][0, 0], 0.0)
        self.assertAlmostEqual(metrics["auroc_benefit"][0, 0], 1.0)
        self.assertAlmostEqual(metrics["auroc_severe_harm"][0, 0], 0.0)

    def test_forbidden_and_external_paths_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE._checked_path(Path("/tmp/e275.json"), role="input")
        with self.assertRaises(ValueError):
            MODULE._checked_path(Path("/ssd8/e275.json"), role="input")
        with self.assertRaises(ValueError):
            MODULE._checked_path(Path("/ssd7/e275.json"), role="output", output=True)

    @unittest.skipUnless(MODULE.DEFAULT_PROTOCOL.is_file(), "E275 protocol unavailable")
    def test_real_e275_schema_and_development_boundary(self) -> None:
        inputs = MODULE.load_inputs()
        self.assertEqual(inputs.uncertainty.shape, (1200, 14))
        self.assertEqual(inputs.gains.shape, (1200, 10))
        report = MODULE.build_report(inputs, bootstrap_draws=3, seed=7)
        self.assertEqual(report["schema"], "e275-uncertainty-action-diagnostic/v1")
        self.assertTrue(report["development_only"])
        self.assertFalse(any(report["authority"].values()))
        self.assertFalse(report["targets"]["pixel_harm_targets_available"])
        self.assertFalse(report["method"]["multiplicity_adjusted"])
        self.assertIn("implementation", report["sources"])
        self.assertEqual(len(report["actions"]), 10)
        for action in report["actions"].values():
            self.assertEqual(len(action["uncertainty_features"]), 14)
            self.assertGreater(action["outcome_counts"]["severe_harm"], 0)


if __name__ == "__main__":
    unittest.main()
