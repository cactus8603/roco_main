"""Small synthetic tests of labels, split isolation and deployed policy alignment."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from stablebridge.learning import (CandidateAcceptor, TrainingConfig, _UtilityNet,
                                   calibrate_policy, load_acceptor, select_queries,
                                   train_acceptor, validate_records)


def records():
    data = {name: [] for name in ("features", "baseline_error", "candidate_error", "query_id", "scene", "split", "candidate_index")}
    for scene, split in (("fit0", "fit"), ("fit1", "fit"), ("cal0", "calibration"),
                         ("cal1", "calibration"), ("eval0", "eval")):
        for query in range(2):
            for candidate in range(3):
                f = np.zeros(12, np.float32)
                f[0] = query / 2
                f[1:5] = [0.3, 0.4, 0.2, 0.5]
                f[5] = candidate / 32
                f[6] = candidate / 128
                f[8:10] = 1
                f[10] = query % 2
                data["features"].append(f)
                data["baseline_error"].append(10.0)
                data["candidate_error"].append((10.0, 7.0, 110.0)[candidate])
                data["query_id"].append(f"{scene}/q{query}")
                data["scene"].append(scene)
                data["split"].append(split)
                data["candidate_index"].append(candidate)
    return {name: np.asarray(values) for name, values in data.items()}


def calibration_config():
    return TrainingConfig(epochs=1, hidden_dim=4, batch_size=64, min_accepted_queries=2,
                          min_calibration_scenes=2, min_accepted_scenes=2)


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_signed_unclipped_targets_and_strict_harm_threshold(self):
        source = records()
        source["candidate_error"][1] = 11.0  # exactly +1 is not severe
        data = validate_records(source)
        self.assertEqual(data["gain"].min(), -100)
        self.assertEqual(data["gain"][1], -1)
        self.assertEqual(data["severe_harm"][1], 0)
        self.assertEqual(data["severe_harm"][2], 1)
        self.assertEqual(data["gain"].max(), 3)

    def test_scene_and_query_leakage_rejected(self):
        source = records()
        source["scene"][source["split"] == "calibration"] = "fit0"
        with self.assertRaisesRegex(ValueError, "Scene leakage"):
            validate_records(source)
        source = records()
        source["query_id"][6] = source["query_id"][0]
        with self.assertRaisesRegex(ValueError, "Query identity reused"):
            validate_records(source)

    def test_identity_and_fallback_layout_validation(self):
        source = records()
        no_indices = {key: value for key, value in source.items() if key != "candidate_index"}
        data = validate_records(no_indices)
        np.testing.assert_array_equal(data["candidate_index"], source["candidate_index"])
        source["candidate_error"][0] = 9.0
        with self.assertRaisesRegex(ValueError, "Identity candidate"):
            validate_records(source)

    def test_calibration_after_argmax_does_not_choose_safe_runner_up(self):
        data = validate_records(records())
        gain = np.tile([7, 3, 4], len(data["features"]) // 3).astype(np.float32)
        harm = np.tile([0, .01, .99], len(data["features"]) // 3).astype(np.float32)
        selected = select_queries(data, gain, harm, calibration_config())
        self.assertTrue(np.all(data["candidate_index"][selected["row"]] == 2))
        policy = calibrate_policy(selected, calibration_config())
        self.assertEqual(policy["state"], "reject_all")
        self.assertEqual(policy["accepted_queries"], 0)
        # A policy that screened by harm before argmax would wrongly calibrate
        # candidate 1, unlike the actual deployment policy.

    def test_positive_calibration_preserves_complete_query_denominator(self):
        data = validate_records(records())
        gain = np.tile([100, 3, -4], len(data["features"]) // 3).astype(np.float32)
        harm = np.tile([0, .02, .99], len(data["features"]) // 3).astype(np.float32)
        selected = select_queries(data, gain, harm, calibration_config())
        self.assertTrue(np.all(data["candidate_index"][selected["row"]] == 1))
        selected["predicted_gain"][0] = 0  # rejected query still counts
        policy = calibrate_policy(selected, replace(calibration_config(), min_accepted_queries=1, min_accepted_scenes=1))
        self.assertEqual(policy["state"], "empirically_calibrated")
        self.assertEqual(policy["accepted_queries"], 3)
        self.assertEqual(policy["scene_macro_gain_px"], 2.25)
        self.assertEqual(policy["calibration_queries"], 4)
        self.assertEqual(policy["task_metrics"]["flow"]["scene_macro_gain_px"], 1.5)
        self.assertEqual(policy["task_metrics"]["stereo"]["scene_macro_gain_px"], 3.0)

    def test_task_regression_cannot_hide_in_pooled_gain(self):
        data = validate_records(records())
        gain = np.tile([0, 3, -4], len(data["features"]) // 3).astype(np.float32)
        harm = np.zeros(len(gain), np.float32)
        selected = select_queries(data, gain, harm, calibration_config())
        selected["actual_gain"][selected["task"] == "flow"] = -0.1
        selected["actual_gain"][selected["task"] == "stereo"] = 100.0
        self.assertGreater(selected["actual_gain"].mean(), 0)
        policy = calibrate_policy(selected, calibration_config())
        self.assertEqual(policy["state"], "reject_all")

    def test_development_evaluation_alias(self):
        source = records()
        source["split"] = np.asarray(["development_evaluation" if s == "eval" else s for s in source["split"]])
        data = validate_records(source)
        self.assertTrue(np.all(data["split"][source["scene"] == "eval0"] == "evaluation"))

    def test_eligibility_identity_ties_and_fail_closed(self):
        data = validate_records(records())
        gain = np.tile([100, 0, -4], len(data["features"]) // 3).astype(np.float32)
        harm = np.zeros(len(gain), np.float32)
        selected = select_queries(data, gain, harm, calibration_config())
        self.assertTrue(np.all(selected["row"] == selected["identity_row"]))
        self.assertEqual(calibrate_policy(selected, calibration_config())["state"], "reject_all")
        gain[data["candidate_index"] == 1] = 1000
        data["features"][data["candidate_index"] == 1, 5] = 2  # 64 px > 32px budget
        selected = select_queries(data, gain, harm, calibration_config())
        self.assertTrue(np.all(selected["row"] == selected["identity_row"]))
        selected = {name: value[:1] for name, value in selected.items()}
        self.assertEqual(calibrate_policy(selected, calibration_config())["reason"],
                         "insufficient_scene_separated_calibration_data")

    def test_frozen_predict_roundtrip_and_reject_all_serialization(self):
        config = calibration_config()
        network = _UtilityNet(config.hidden_dim)
        for parameter in network.parameters():
            parameter.data.zero_()
        network.network[-1].bias.data[:] = torch.tensor([-2.0, 0.0])
        model = CandidateAcceptor(network, np.zeros(12), np.ones(12), 100.0, config)
        features = np.zeros((3, 5, 12), np.float32)
        gain, harm = model.predict(features)
        np.testing.assert_array_equal(gain, -200)
        np.testing.assert_array_equal(harm, .5)
        self.assertEqual(gain.shape, (3, 5))
        self.assertTrue(all(not p.requires_grad for p in model.network.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "utility.pt"
            model.save(path)
            metadata = json.loads(path.with_suffix(".pt.json").read_text())
            self.assertIsNone(metadata["policy"]["gain_threshold"])
            self.assertNotIn("Infinity", path.with_suffix(".pt.json").read_text())
            loaded = load_acceptor(path)
            self.assertTrue(math.isinf(loaded.gain_threshold))
            np.testing.assert_array_equal(loaded.predict(features)[0], gain)

    def test_positive_calibration_report_loads_with_weights_only(self):
        data = validate_records(records())
        gain = np.tile([0, 3, -4], len(data["features"]) // 3).astype(np.float32)
        harm = np.zeros_like(gain)
        policy = calibrate_policy(select_queries(data, gain, harm, calibration_config()), calibration_config())
        self.assertEqual(policy["state"], "empirically_calibrated")
        model = CandidateAcceptor(_UtilityNet(4), np.zeros(12), np.ones(12), 1, calibration_config(),
                                  gain_threshold=policy["gain_threshold"], harm_threshold=policy["harm_threshold"],
                                  report={"calibration": policy})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "positive.pt"
            model.save(path)
            loaded = load_acceptor(path)
            self.assertEqual(loaded.report, model.report)

    def test_tiny_synthetic_fit_ignores_calibration_and_eval_feature_statistics(self):
        # Unit fixture only (12 fit rows, one CPU optimizer step); no research
        # data is loaded or training process launched by the tests.
        first = records()
        second = {key: value.copy() for key, value in first.items()}
        other = second["split"] != "fit"
        second["features"][other, 0] = 5000
        second["candidate_error"][other & (second["candidate_index"] != 0)] = 10000
        a = train_acceptor(first, config=calibration_config())
        b = train_acceptor(second, config=calibration_config())
        np.testing.assert_array_equal(a.mean.numpy(), b.mean.numpy())
        np.testing.assert_array_equal(a.std.numpy(), b.std.numpy())
        for key, value in a.network.state_dict().items():
            torch.testing.assert_close(value, b.network.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(a.report["fit_gain_min"], -100)
        self.assertGreater(a.report["fit_gain_below_minus20_rows"], 0)
        self.assertEqual(a.report["evaluation_rows_used_for_training_or_thresholds"], 0)

    def test_no_calibration_rows_remain_reject_all(self):
        source = records()
        keep = source["split"] != "calibration"
        source = {name: value[keep] for name, value in source.items()}
        result = train_acceptor(source, config=calibration_config())
        self.assertTrue(math.isinf(result.gain_threshold))
        self.assertEqual(result.report["calibration"]["reason"], "insufficient_scene_separated_calibration_data")


if __name__ == "__main__":
    unittest.main()
