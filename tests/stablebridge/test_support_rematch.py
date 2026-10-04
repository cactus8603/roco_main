"""Synthetic interface checks; these do not claim empirical repair capacity."""
from __future__ import annotations

import unittest

import torch

from stablebridge.support_rematch import SupportConditionedRematcher, sample_native_map


class SupportRematchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(84)
        self.model = SupportConditionedRematcher(feature_channels=4, hidden_dim=16, projected_dim=8)
        initial = torch.zeros(1, 2, 64, 64)
        initial[:, 0], initial[:, 1] = 2, -1
        self.inputs = {
            "source_features": torch.randn(1, 4, 4, 4),
            "target_features": torch.randn(1, 4, 4, 4),
            "initial_uv": initial,
            "query_xy": torch.tensor([[[32., 32.], [18., 32.], [46., 32.]]]),
            "supports_xy": torch.tensor([[[4., 4.], [59., 4.], [4., 59.], [59., 59.]]]),
            "supports_uv": torch.tensor([[[10., 3.], [-10., 4.], [5., -8.], [-6., 9.]]]),
            "support_valid": torch.ones(1, 4, dtype=torch.bool),
            "support_source": torch.full((1, 4), 2, dtype=torch.long),
            "source_rgb": torch.rand(1, 3, 64, 64),
            "target_rgb": torch.rand(1, 3, 64, 64),
            "task": "flow",
        }

    def test_native_feature_centers_and_bilinear_interpolation(self):
        feature = torch.tensor([[[[0., 2.], [4., 6.]]]])
        xy = torch.tensor([[[7.5, 7.5], [23.5, 23.5], [15.5, 15.5], [0., 0.]]])
        actual = sample_native_map(feature, xy, stride=16, center_offset=7.5)
        torch.testing.assert_close(actual[..., 0], torch.tensor([[0., 6., 3., 0.]]))

    def test_budget_identity_and_strict_stereo_geometry(self):
        self.inputs["task"] = "stereo"
        self.inputs["supports_uv"][..., 1] = 99
        output = self.model(**self.inputs)
        self.assertEqual(output["candidate_uv"].shape, (1, 3, 30, 2))
        for name in ("candidate_uv", "refined_candidate_uv", "output_uv", "selected_uv"):
            self.assertTrue(bool((output[name][..., 1] == 0).all()), name)
        torch.testing.assert_close(output["candidate_uv"][:, :, 0, 0], torch.full((1, 3), 2.))
        torch.testing.assert_close(output["refined_candidate_uv"][:, :, 0], output["candidate_uv"][:, :, 0])
        torch.testing.assert_close(output["probabilities"].sum(-1), torch.ones(1, 3))
        self.assertEqual(self.model.contract()["candidate_count"], 30)
        self.assertEqual(self.model.contract()["parameters"], sum(p.numel() for p in self.model.parameters()))

    def test_target_feature_and_rgb_evidence_each_affect_rematching(self):
        original = self.model(**self.inputs)
        changed_features = self.model(**{**self.inputs, "target_features": self.inputs["target_features"] + 2})
        changed_rgb = self.model(**{**self.inputs, "target_rgb": 1 - self.inputs["target_rgb"]})
        self.assertFalse(torch.allclose(original["logits"], changed_features["logits"]))
        self.assertFalse(torch.allclose(original["output_uv"], changed_features["output_uv"]))
        self.assertFalse(torch.allclose(original["logits"], changed_rgb["logits"]))
        self.assertFalse(torch.allclose(original["output_uv"], changed_rgb["output_uv"]))

    def test_direct_propagation_cannot_read_target_features_or_rgb(self):
        original = self.model(**self.inputs, rematch=False)
        changed = self.model(**{**self.inputs,
                                "target_features": torch.full_like(self.inputs["target_features"], float("nan")),
                                "target_rgb": torch.full_like(self.inputs["target_rgb"], float("nan"))}, rematch=False)
        for key in original:
            torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)

    def test_support_association_changes_hypotheses_but_not_independent_search(self):
        original = self.model(**self.inputs)
        wrong = self.model(**{**self.inputs, "supports_uv": self.inputs["supports_uv"].roll(1, dims=1)})
        torch.testing.assert_close(original["candidate_uv"][:, :, :10], wrong["candidate_uv"][:, :, :10])
        self.assertFalse(torch.equal(original["candidate_uv"][:, :, 10:], wrong["candidate_uv"][:, :, 10:]))
        self.assertFalse(torch.allclose(original["output_uv"], wrong["output_uv"]))
        torch.testing.assert_close(original["support_neighbor_indices"], wrong["support_neighbor_indices"])

    def test_no_support_ignores_values_provenance_and_uses_equal_budget(self):
        original = self.model(**self.inputs, support_mode="none")
        changed = self.model(**{**self.inputs, "supports_uv": self.inputs["supports_uv"] * 999,
                                "support_source": torch.full_like(self.inputs["support_source"], 3)}, support_mode="none")
        for key in original:
            torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)
        self.assertEqual(original["candidate_uv"].shape[2], 30)
        self.assertFalse(bool(original["candidate_has_support"].any()))
        empty = {**self.inputs, "supports_xy": torch.zeros(1, 0, 2), "supports_uv": torch.zeros(1, 0, 2),
                 "support_source": torch.zeros(1, 0, dtype=torch.long),
                 "support_valid": torch.zeros(1, 0, dtype=torch.bool)}
        absent = self.model(**empty)
        torch.testing.assert_close(original["output_uv"], absent["output_uv"], rtol=0, atol=0)

    def test_query_exclusion_happens_before_any_support_aggregation(self):
        # Both the exact query and a point on the 8px exclusion boundary carry
        # arbitrary privileged values. Neither may affect this query's result.
        query = torch.tensor([[[32., 32.]]])
        inputs = {**self.inputs, "query_xy": query,
                  "supports_xy": torch.cat((torch.tensor([[[32., 32.], [40., 32.]]]), self.inputs["supports_xy"]), dim=1),
                  "supports_uv": torch.cat((torch.tensor([[[123., 456.], [-123., -456.]]]), self.inputs["supports_uv"]), dim=1),
                  "support_source": torch.full((1, 6), 3, dtype=torch.long),
                  "support_valid": torch.ones(1, 6, dtype=torch.bool)}
        original = self.model(**inputs)
        alternate_values = inputs["supports_uv"].clone()
        alternate_values[:, :2] *= -300
        alternate = self.model(**{**inputs, "supports_uv": alternate_values})
        self.assertTrue(bool((original["support_neighbor_indices"] >= 2).all()))
        for key in original:
            torch.testing.assert_close(original[key], alternate[key], rtol=0, atol=0)

    def test_out_of_crop_search_retains_exact_identity(self):
        inputs = {**self.inputs, "initial_uv": torch.full_like(self.inputs["initial_uv"], 1000.)}
        output = self.model(**inputs, support_mode="none")
        self.assertTrue(bool(output["candidate_valid"][:, :, 0].all()))
        self.assertFalse(bool(output["candidate_valid"][:, :, 1:].any()))
        torch.testing.assert_close(output["output_uv"], torch.full((1, 3, 2), 1000.), rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(output["output_uv"]).all()))

    def test_gradients_train_only_module_not_backbone_or_support_values(self):
        for name in ("source_features", "target_features", "initial_uv", "supports_uv", "source_rgb", "target_rgb"):
            self.inputs[name].requires_grad_(True)
        output = self.model(**self.inputs)
        output["output_uv"].square().mean().backward()
        for name in ("source_features", "target_features", "initial_uv", "supports_uv", "source_rgb", "target_rgb"):
            self.assertIsNone(self.inputs[name].grad, name)
        for layer in (self.model.feature_projection, self.model.score_head, self.model.residual_head):
            self.assertIsNotNone(layer.weight.grad)
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_chunking_preserves_native_query_results(self):
        original = self.model(**self.inputs)
        self.model.query_chunk_size = 1
        chunked = self.model(**self.inputs)
        for key in original:
            torch.testing.assert_close(original[key], chunked[key], rtol=1e-5, atol=1e-6)

    def test_invalid_supports_fall_back_without_nan(self):
        broken = {**self.inputs, "supports_uv": torch.full_like(self.inputs["supports_uv"], float("nan"))}
        output = self.model(**broken)
        absent = self.model(**self.inputs, support_mode="none")
        torch.testing.assert_close(output["output_uv"], absent["output_uv"], rtol=0, atol=0)
        self.assertTrue(bool((output["support_neighbor_indices"] == -1).all()))


if __name__ == "__main__":
    unittest.main()
