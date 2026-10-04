"""Replay decisions remain independent of GT, including invalid/mutated labels."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from stablebridge.learning import CandidateAcceptor, TrainingConfig, _UtilityNet
from stablebridge.replay import decide_candidates, replay_run, score_decisions, summarize


def fixture():
    features = np.zeros((6, 12), np.float32)
    features[:, 8:10] = 1
    features[:, 10] = 1
    features[:, 5] = np.tile([0, 1 / 32, 2 / 32], 2)
    features[:, 0] = np.tile([0, 3, 2], 2)
    return {"features": features, "candidate_index": np.tile([0, 1, 2], 2),
            "query_id": np.repeat(["stereo_0001_0050_correlated_noise/solved/10/20", "stereo_0001_0050_correlated_noise/solved/11/20"], 3),
            "scene": np.full(6, "0001"), "split": np.full(6, "dev_eval"),
            "baseline_error": np.full(6, 10.0), "candidate_error": np.tile([10, 7, 20], 2)}


class FixedAcceptor:
    config = TrainingConfig()
    gain_threshold = 0.0
    harm_threshold = 0.05

    def predict(self, features):
        return features[:, 0], np.zeros(len(features), np.float32)


class ReplayTests(unittest.TestCase):
    def test_permuting_labels_does_not_change_decisions(self):
        a = fixture()
        b = {key: value.copy() for key, value in a.items()}
        b["candidate_error"][[1, 2, 4, 5]] = [999, 0, 0, 999]
        b["baseline_error"][:] = np.nan
        first = decide_candidates(a, FixedAcceptor())
        second = decide_candidates(b, FixedAcceptor())
        self.assertEqual(first, second)
        self.assertTrue(all(row["accepted"] for row in first))
        self.assertTrue(all(row["selected_candidate_index"] == 1 for row in first))
        scored = score_decisions(first, a["baseline_error"], a["candidate_error"])
        self.assertEqual(scored[0]["evaluation"]["signed_gain_px"], 3)
        altered = score_decisions(second, b["baseline_error"], b["candidate_error"])
        self.assertTrue(all(not row["evaluation"]["valid_gt"] for row in altered))

    def test_gt_arrays_not_accessed_by_decision_function(self):
        class Guard(dict):
            def __getitem__(self, key):
                if "error" in key:
                    raise AssertionError("GT was consulted for selection")
                return super().__getitem__(key)
        self.assertEqual(len(decide_candidates(Guard(fixture()), FixedAcceptor())), 2)

    def test_rejection_keeps_denominator_and_zero_gain(self):
        data = fixture()
        data["split"] = np.full(len(data["features"]), "development_evaluation")
        model = FixedAcceptor()
        model.gain_threshold = 100
        decisions = decide_candidates(data, model)
        scored = score_decisions(decisions, data["baseline_error"], data["candidate_error"])
        result = summarize(scored)[0]
        self.assertEqual(result["query_pooled"]["queries"], 2)
        self.assertEqual(result["query_pooled"]["accepted_queries"], 0)
        self.assertEqual(result["scene_macro"]["signed_gain_px"], 0)
        self.assertEqual(result["task"], "stereo")
        self.assertEqual(result["split"], "evaluation")
        self.assertEqual(scored[0]["profile"], "correlated_noise")

    def test_invalid_gt_still_has_decision_and_acceptance_denominator(self):
        data = fixture()
        decisions = decide_candidates(data, FixedAcceptor())
        data["candidate_error"] = data["candidate_error"].astype(float)
        data["candidate_error"][3:] = np.nan
        data["baseline_error"][3:] = np.nan
        scored = score_decisions(decisions, data["baseline_error"], data["candidate_error"])
        pooled = summarize(scored)[0]["query_pooled"]
        self.assertEqual(pooled["queries"], 2)
        self.assertEqual(pooled["valid_gt_queries"], 1)
        self.assertEqual(pooled["accepted_queries"], 2)
        self.assertEqual(pooled["signed_gain_px"], 3)

    def test_file_replay_and_immutable_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, output = root / "run", root / "replay"
            (run / "supervision").mkdir(parents=True)
            source = run / "supervision/case.npz"
            np.savez(source, **fixture())
            (run / "manifest.json").write_text(json.dumps({"matching": {"max_update_px": 32.0, "require_baseline_support": True}}))
            config = TrainingConfig(hidden_dim=4)
            network = _UtilityNet(config.hidden_dim)
            for parameter in network.parameters():
                parameter.data.zero_()
            network.network[-1].bias.data[:] = torch.tensor([1.0, -10.0])
            model = CandidateAcceptor(network, np.zeros(12), np.ones(12), 1, config,
                                      gain_threshold=0, harm_threshold=.05)
            checkpoint = root / "model.pt"
            model.save(checkpoint)
            result = replay_run(run, checkpoint, output)
            self.assertEqual(result["query_count"], 2)
            self.assertFalse(result["whole_image_result"])
            self.assertFalse(result["closed_loop_result"])
            self.assertTrue((output / "results.md").exists())
            decisions = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["accepted"] for row in decisions))
            replay_run(run, checkpoint, output)  # same immutable inputs may replay
            changed = fixture()
            changed["features"][1, 0] += 1
            np.savez(source, **changed)
            with self.assertRaisesRegex(ValueError, "changed"):
                replay_run(run, checkpoint, output)


if __name__ == "__main__":
    unittest.main()
