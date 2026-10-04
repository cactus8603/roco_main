"""CPU checks for descriptor geometry, missing evidence and equal-capacity masks."""
import unittest

import numpy as np
import torch

from stablebridge.binary_revision import (BinaryRevisionMLP, DESCRIPTOR_DIM, FEATURE_NAMES,
                                         GEOMETRY_DIM, MATCHING_DIM, build_descriptors)


def inputs(height=32, width=48):
    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    features = rng.normal(size=(4, height // 16, width // 16)).astype(np.float32)
    initial = np.zeros((2, height, width), np.float32)
    return {"image0": image, "image1": image.copy(), "source_features": features,
            "target_features": features.copy(), "initial": initial,
            "candidate": initial.copy(), "candidate_valid": np.ones((height, width), bool)}


class BinaryRevisionDescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_shape_identity_costs_and_exact_feature_schema(self):
        result, meta = build_descriptors(inputs(), chunk_size=113)
        self.assertEqual(result.shape, (32, 48, 64))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual((GEOMETRY_DIM, MATCHING_DIM, DESCRIPTOR_DIM), (32, 32, 64))
        self.assertEqual(meta["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(len(set(meta["feature_names"])), 64)
        self.assertFalse(meta["gt_used"])
        for name in ("initial_center_rgb_l1", "candidate_patch_rgb_rmse", "initial_encoder_cosine_cost",
                     "candidate_minus_initial_center_rgb_l1", "candidate_minus_initial_encoder_unit_vector_l1"):
            np.testing.assert_allclose(result[12, 20, FEATURE_NAMES.index(name)], 0, atol=2e-7)

    def test_rgb_inbounds_does_not_make_outer_feature_border_valid(self):
        source = inputs()
        source["candidate"][0, 12, 12] = -12  # endpoint x=0: valid RGB, invalid encoder evidence
        result, _ = build_descriptors(source)
        get = lambda name: result[12, 12, FEATURE_NAMES.index(name)]
        self.assertEqual(get("candidate_endpoint_in_image"), 1)
        self.assertEqual(get("candidate_center_rgb_valid"), 1)
        self.assertEqual(get("candidate_feature_pair_geometrically_valid"), 0)
        self.assertEqual(get("candidate_encoder_pair_valid"), 0)
        self.assertEqual(get("candidate_encoder_cosine_cost"), 0)
        self.assertEqual(get("candidate_encoder_unit_vector_l1"), 0)
        self.assertEqual(get("candidate_patch_pair_valid"), 0)
        self.assertEqual(get("candidate_patch_rgb_rmse"), 0)
        self.assertEqual(get("candidate_minus_initial_encoder_cosine_cost"), 0)

    def test_oob_and_nonfinite_candidate_never_become_matching_evidence(self):
        source = inputs()
        source["candidate"][0, 12, 12] = 1000
        source["candidate"][0, 12, 13] = np.nan
        source["candidate_valid"][12, 14] = False
        result, _ = build_descriptors(source)
        self.assertTrue(np.isfinite(result).all())
        for x in (12, 13, 14):
            for name in ("candidate_center_rgb_valid", "candidate_patch_pair_valid", "candidate_encoder_pair_valid",
                         "candidate_center_rgb_l1", "candidate_patch_rgb_rmse", "candidate_encoder_cosine_cost"):
                self.assertEqual(result[12, x, FEATURE_NAMES.index(name)], 0)
        self.assertEqual(result[12, 13, FEATURE_NAMES.index("candidate_finite")], 0)
        self.assertEqual(result[12, 12, FEATURE_NAMES.index("candidate_finite")], 1)
        # A fallback vector can remain geometrically in bounds even when its
        # supplied candidate availability is false; these are separate facts.
        self.assertEqual(result[12, 14, FEATURE_NAMES.index("candidate_endpoint_in_image")], 1)
        self.assertEqual(result[12, 14, FEATURE_NAMES.index("candidate_valid")], 0)

    def test_geometry_excludes_appearance_and_uncertainty(self):
        a = inputs()
        first, _ = build_descriptors(a)
        b = {name: value.copy() for name, value in a.items()}
        b["image0"][:] = 0
        b["image1"][:] = 255
        b["source_features"] *= -5
        b["target_features"] *= 3
        class Unreadable:
            def __array__(self, *args, **kwargs):
                raise AssertionError("uncertainty must not be inspected")
        b["raw_uncertainty"] = Unreadable()
        second, _ = build_descriptors(b)
        np.testing.assert_array_equal(first[..., :32], second[..., :32])
        self.assertFalse(np.array_equal(first[..., 32:], second[..., 32:]))
        with self.assertRaisesRegex(ValueError, "allowlist"):
            build_descriptors({**a, "Q": np.ones((32, 48), bool)})
        with self.assertRaisesRegex(ValueError, "allowlist"):
            build_descriptors({**a, "gt": np.zeros((2, 32, 48))})

    def test_chunk_size_preserves_values_and_frozen_maps_are_not_mutated(self):
        data = inputs()
        before = {name: value.copy() for name, value in data.items()}
        a, _ = build_descriptors(data, chunk_size=37)
        b, _ = build_descriptors(data, chunk_size=2048)
        np.testing.assert_allclose(a, b, atol=2e-7, rtol=1e-6)
        for name in before:
            np.testing.assert_array_equal(data[name], before[name])


class BinaryRevisionMLPTests(unittest.TestCase):
    def test_geometry_mask_is_invariant_to_matching_and_has_zero_matching_gradient(self):
        torch.manual_seed(2)
        model = BinaryRevisionMLP(64, 32, "geometry", "gain")
        features = torch.randn(5, 64, requires_grad=True)
        changed = features.detach().clone()
        changed[:, 32:] = torch.nan
        torch.testing.assert_close(model(features), model(changed), rtol=0, atol=0)
        model(features).sum().backward()
        self.assertTrue(torch.equal(features.grad[:, 32:], torch.zeros_like(features.grad[:, 32:])))

    def test_equal_information_capacity_and_objective_scores(self):
        for objective in ("gain", "errors"):
            geometry = BinaryRevisionMLP(64, 32, "geometry", objective)
            full = BinaryRevisionMLP(64, 32, "full", objective)
            self.assertEqual(geometry.contract()["parameters"], full.contract()["parameters"])
            self.assertEqual(geometry.contract()["trunk_parameters"], 24832)
            output = full(torch.randn(4, 64))
            self.assertEqual(output.shape, (4, 1 if objective == "gain" else 2))
            self.assertTrue(torch.isfinite(output).all())
            expected = output[:, 0] if objective == "gain" else output[:, 0] - output[:, 1]
            torch.testing.assert_close(full.selection_scores(output), expected)
            with torch.no_grad():
                for parameter in full.parameters():
                    parameter.zero_()
                full.head.bias.fill_(-3)
            self.assertTrue(torch.all(full(torch.zeros(2, 64)) == -3))

    def test_shared_initial_weights_do_not_overwrite_information_contract(self):
        torch.manual_seed(9)
        geometry = BinaryRevisionMLP(64, 32, "geometry", "gain")
        full = BinaryRevisionMLP(64, 32, "full", "gain")
        full.load_state_dict(geometry.state_dict(), strict=True)
        self.assertFalse(torch.any(geometry.information_mask[32:]))
        self.assertTrue(torch.all(full.information_mask))
        first = torch.zeros(2, 64)
        second = first.clone()
        second[:, 32:] = 1
        torch.testing.assert_close(geometry(first), geometry(second), rtol=0, atol=0)
        self.assertFalse(torch.equal(full(first), full(second)))

    def test_visible_nonfinite_and_invalid_shapes_fail(self):
        model = BinaryRevisionMLP(64, 32, "full", "errors")
        with self.assertRaises(ValueError):
            model(torch.zeros(4, 63))
        bad = torch.zeros(4, 64)
        bad[0, 33] = torch.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            model(bad)


if __name__ == "__main__":
    unittest.main()
