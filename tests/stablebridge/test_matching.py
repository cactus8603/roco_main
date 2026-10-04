"""Synthetic correspondence checks exercise re-matching, not neural accuracy."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from stablebridge.contracts import FrameEvidence
from stablebridge.matching import (FeatureRematcher, MatchConfig, _rank01,
                                  compress_features, query_and_anchor_masks)


def fixture(dx=2, h=6, w=12):
    """Each source pixel has a distinct descriptor; matching target is x+dx."""
    features = np.eye(h*w, dtype=np.float32).reshape(h*w, h, w)
    target = np.roll(features, dx, axis=2)
    image = np.zeros((h, w, 3), np.uint8)
    prediction = SimpleNamespace(displacement=np.zeros((2, h, w), np.float32),
                                 raw_uncertainty=np.zeros((h, w), np.float32),
                                 source_features=features, target_features=target,
                                 metadata={"origin_xy": [0, 0]})
    config = MatchConfig(feature_dim=h*w, feature_stride=1, source_neighbors=(),
                         max_queries=10, feature_weight=1, movement_penalty=0,
                         min_improvement=.01, max_candidates=12)
    mask = np.zeros((h, w), bool)
    mask[3, 4] = True
    return prediction, image, config, mask


def oversized_history_fixture():
    """A legal local proposal costs .2; an illegal 3px history proposal costs 0."""
    prediction, image, config, mask = fixture()
    y, x = np.argwhere(mask)[0]
    correct = prediction.source_features[:, y, x]
    distractor = prediction.source_features[:, 0, 0]
    prediction.target_features[:] = distractor[:, None, None]
    prediction.target_features[:, y, x+1] = .8 * correct + .6 * distractor
    prediction.target_features[:, y, x+3] = correct
    local = np.zeros_like(prediction.displacement)
    local[0] = 1
    history = FrameEvidence('scene', 1, 'left', prediction.source_features, source_groups=('history-root',))
    return prediction, image, replace(config, max_update_px=1.), mask, local, history


class MatchingTest(unittest.TestCase):
    def test_empty_query_keeps_a_complete_identity_bank_for_fixed_reporting(self):
        prediction,image,config,_=fixture()
        result=FeatureRematcher(config).refine(
            prediction,image,image,'flow',mask=np.zeros(image.shape[:2],bool))
        self.assertEqual(result['candidates'].shape,(1,0,2))
        self.assertEqual(result['candidate_features'].shape,(1,0,12))
        self.assertEqual(result['candidate_names'],['identity'])
        np.testing.assert_array_equal(result['output'],prediction.displacement)

    def test_oversized_history_cannot_cancel_an_admissible_spatial_update(self):
        prediction, image, config, mask, local, history = oversized_history_fixture()
        matcher = FeatureRematcher(config)
        spatial = matcher.refine(prediction, image, image, 'flow', mask=mask, local_candidates=(local,))
        temporal = matcher.refine(prediction, image, image, 'flow', mask=mask,
                                  local_candidates=(local,), temporal_records=(history,))
        historical_index = temporal['candidate_names'].index('history_0')
        np.testing.assert_array_equal(temporal['candidates'][historical_index, 0], [3, 0])
        self.assertTrue(temporal['candidate_valid'][historical_index, 0])
        self.assertLess(temporal['candidate_costs'][historical_index, 0],
                        temporal['candidate_costs'][temporal['candidate_names'].index('local_0'), 0])
        np.testing.assert_array_equal(spatial['output'][:, 3, 4], [1, 0])
        np.testing.assert_array_equal(temporal['output'], spatial['output'])
        self.assertTrue(temporal['accepted'][3, 4])
        self.assertEqual(temporal['candidate_names'][temporal['best_indices'][0]], 'local_0')

    def test_identity_is_selected_when_only_ineligible_history_can_beat_it(self):
        prediction, image, config, mask, _, history = oversized_history_fixture()
        result = FeatureRematcher(config).refine(prediction, image, image, 'flow', mask=mask,
                                                temporal_records=(history,))
        self.assertEqual(result['best_indices'][0], 0)
        self.assertFalse(result['accepted'].any())
        np.testing.assert_array_equal(result['output'], prediction.displacement)

    def test_learned_eligibility_already_rejects_oversized_history_before_selection(self):
        prediction, image, config, mask, local, history = oversized_history_fixture()
        class Acceptor:
            feature_schema = 'candidate_features_v1_12'
            max_update_px = config.max_update_px
            require_baseline_support = config.require_baseline_support
            gain_threshold = 0.
            harm_threshold = .1
            def predict(self, features):
                return features[..., 4]-features[..., 1], np.zeros(features.shape[:-1], np.float32)
        result = FeatureRematcher(config).refine(prediction, image, image, 'flow', mask=mask,
                    local_candidates=(local,), temporal_records=(history,), acceptor=Acceptor())
        np.testing.assert_array_equal(result['output'][:, 3, 4], [1, 0])
        self.assertEqual(result['candidate_names'][result['best_indices'][0]], 'local_0')

    def test_explicit_unsupported_baseline_policy_still_allows_observed_candidate(self):
        prediction, image, config, mask = fixture()
        prediction.displacement[0, 3, 4] = 100
        local = np.zeros_like(prediction.displacement)
        local[0] = 2
        result = FeatureRematcher(replace(config, max_update_px=200, require_baseline_support=False)).refine(
            prediction, image, image, 'flow', mask=mask, local_candidates=(local,))
        self.assertFalse(result['candidate_valid'][0, 0])
        self.assertTrue(result['accepted'][3, 4])
        np.testing.assert_array_equal(result['output'][:, 3, 4], [2, 0])

    def test_equal_observations_have_equal_risk_and_fraction_limits_query_count(self):
        np.testing.assert_array_equal(_rank01(np.ones((2, 3))), .5)
        np.testing.assert_array_equal(_rank01(np.array([0, 0, 2, 4, 4])), [.125, .125, .5, .875, .875])
        prediction, image, config, _ = fixture()
        config = replace(config, query_fraction=.1, max_queries=100)
        mask, anchors, risk = query_and_anchor_masks(prediction.displacement,
                                                    prediction.raw_uncertainty, image, image, config)
        np.testing.assert_array_equal(risk, .5)
        self.assertEqual(mask.sum(), int(np.ceil(.1 * mask.size)))
        self.assertTrue(np.all(~(mask & anchors)))
        self.assertGreater(np.flatnonzero(mask)[-1] - np.flatnonzero(mask)[0], mask.size // 2)

    def test_flow_rematches_current_target_and_only_updates_query(self):
        prediction, image, config, mask = fixture(dx=2)
        result = FeatureRematcher(config).refine(prediction, image, image, "flow", mask=mask)
        self.assertTrue(result["accepted"][3, 4])
        np.testing.assert_array_equal(result["output"][:, 3, 4], [2, 0])
        np.testing.assert_array_equal(result["output"][:, ~mask], prediction.displacement[:, ~mask])
        best = result["best_indices"][0]
        self.assertLess(result["candidate_costs"][best, 0], result["candidate_costs"][0, 0])
        self.assertTrue(result["candidate_valid"][best, 0])

    def test_stereo_repair_keeps_vertical_zero(self):
        prediction, image, config, mask = fixture(dx=-2)
        result = FeatureRematcher(config).refine(prediction, image, image, "stereo", mask=mask)
        np.testing.assert_array_equal(result["output"][:, 3, 4], [-2, 0])
        np.testing.assert_array_equal(result["output"][1], 0)
        np.testing.assert_array_equal(result["output"][:, ~mask], prediction.displacement[:, ~mask])
        prediction.displacement[1, 0, 0] = 1
        with self.assertRaises(ValueError):
            FeatureRematcher(config).refine(prediction, image, image, "stereo", mask=mask)

    def test_temporal_budget_always_retains_identity_and_counts_association(self):
        prediction, image, config, mask = fixture()
        config = replace(config, max_candidates=2)
        history = [FrameEvidence("scene", i, "left", prediction.source_features,
                                  source_groups=(f"frame-{i}",)) for i in range(5)]
        result = FeatureRematcher(config).refine(prediction, image, image, "flow", mask=mask,
                                                 temporal_records=history)
        self.assertEqual(result["candidate_names"][0], "identity")
        np.testing.assert_array_equal(result["candidates"][0, 0], [0, 0])
        self.assertEqual(len(result["candidate_names"]), 2)
        self.assertEqual(len(result["diagnostics"]["association"]), 1)
        only_identity = FeatureRematcher(replace(config, max_candidates=1)).refine(
            prediction, image, image, "flow", mask=mask, temporal_records=history)
        self.assertEqual(only_identity["candidate_names"], ["identity"])
        self.assertEqual(only_identity["diagnostics"]["association"], [])
        np.testing.assert_array_equal(only_identity["output"], prediction.displacement)

    def test_repeated_source_does_not_change_candidates_or_crowd_distinct_history(self):
        prediction, image, config, mask = fixture()
        config = replace(config, max_candidates=3)
        a = FrameEvidence("scene", 1, "left", prediction.source_features, source_groups=("root-a",))
        b = replace(a, frame=2, source_groups=("root-b",))
        matcher = FeatureRematcher(config)
        ordinary = matcher.refine(prediction, image, image, "flow", mask=mask, temporal_records=(a, b))
        copied = matcher.refine(prediction, image, image, "flow", mask=mask, temporal_records=(a, a, a, b))
        self.assertEqual(ordinary["candidate_groups"], copied["candidate_groups"])
        np.testing.assert_array_equal(ordinary["candidates"], copied["candidates"])
        np.testing.assert_array_equal(ordinary["output"], copied["output"])
        self.assertEqual(len(copied["diagnostics"]["association"]), 2)

    def test_historical_descriptor_bridge_uses_current_target_not_old_motion(self):
        prediction, image, config, mask = fixture(dx=3)
        matcher = FeatureRematcher(config)
        history = FrameEvidence("scene", 1, "left", np.roll(prediction.source_features, -1, axis=2),
                                origin_xy=(100, 200), source_groups=("root-a",),
                                metadata={"historical_displacement": [999, -777]})
        xy = np.array([[4, 3]], np.float32)
        prepared = matcher.prepare(prediction, image, image)
        candidates, _, diagnostics = matcher.temporal_candidates(xy, prepared, (history,), (10, 20), "flow")
        np.testing.assert_array_equal(candidates[0][0], [3, 0])
        self.assertEqual(diagnostics[0]["example_native_endpoints"],
                         {"source": [14, 23], "anchor": [103, 203], "current_target": [17, 23]})
        self.assertFalse(diagnostics[0]["historical_displacement_used"])
        self.assertFalse(diagnostics[0]["cycle_constraint_verified"])
        prediction.target_features = np.roll(prediction.source_features, -2, axis=2)
        prepared = matcher.prepare(prediction, image, image)
        changed, _, _ = matcher.temporal_candidates(xy, prepared, (history,), (10, 20), "flow")
        np.testing.assert_array_equal(changed[0][0], [-2, 0])

    def test_empty_query_preserves_complete_output_without_preparing_features(self):
        prediction, image, config, mask = fixture()
        matcher = FeatureRematcher(config)
        def forbidden(*args):
            raise AssertionError("An empty query must not generate or read features")
        matcher.prepare = forbidden
        result = matcher.refine(prediction, image, image, "flow", mask=np.zeros_like(mask))
        np.testing.assert_array_equal(result["output"], prediction.displacement)
        self.assertEqual(result["diagnostics"]["candidate_evaluations"], 0)
        self.assertFalse(result["accepted"].any())

    def test_no_ground_truth_field_is_read(self):
        prediction, image, config, mask = fixture()
        class PredictionWithoutLabels:
            def __getattr__(self, name):
                if name in {"gt", "ground_truth", "gt4", "error", "hard_gt"}:
                    raise AssertionError("Deployment matching accessed ground truth")
                return getattr(prediction, name)
        result = FeatureRematcher(config).refine(PredictionWithoutLabels(), image, image, "flow", mask=mask)
        self.assertTrue(result["accepted"].any())
        self.assertTrue(np.isfinite(result["candidate_features"]).all())

    def test_unsupported_original_match_keeps_dense_fallback(self):
        prediction, image, config, mask = fixture()
        prediction.displacement[0, 3, 4] = 100
        local = np.zeros_like(prediction.displacement)
        local[0] = 2
        result = FeatureRematcher(replace(config, max_update_px=200)).refine(
            prediction, image, image, "flow", mask=mask, local_candidates=(local,))
        self.assertFalse(result["candidate_valid"][0, 0])
        self.assertFalse(result["accepted"].any())
        np.testing.assert_array_equal(result["output"], prediction.displacement)
        self.assertTrue(np.isfinite(result["candidate_features"]).all())

    def test_shared_projection_is_deterministic_and_rejects_mixed_versions(self):
        rng = np.random.default_rng(10)
        value = rng.normal(size=(16, 3, 4)).astype(np.float32)
        a, b = compress_features(value, 5, 19), compress_features(value.copy(), 5, 19)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(np.linalg.norm(a, axis=0), 1, atol=1e-6)
        self.assertFalse(np.allclose(a, compress_features(value, 5, 20)))
        prediction, image, config, _ = fixture()
        matcher = FeatureRematcher(config)
        prepared = matcher.prepare(prediction, image, image)
        history = FrameEvidence("scene", 1, "left", prepared["source"],
                                metadata={"feature_spec": {**prepared["feature_spec"], "projection_seed": 1}})
        with self.assertRaises(ValueError):
            matcher.temporal_candidates(np.array([[4, 3]], np.float32), prepared, (history,), (0, 0), "flow")
        prediction.target_features = prediction.target_features[:3]
        with self.assertRaises(ValueError):
            matcher.prepare(prediction, image, image)

    def test_explicit_masks_and_candidate_fields_obey_shape_and_work_limits(self):
        prediction, image, config, mask = fixture()
        matcher = FeatureRematcher(config)
        with self.assertRaises(ValueError):
            matcher.refine(prediction, image, image, "flow", mask=np.ones_like(mask))
        with self.assertRaises(ValueError):
            matcher.refine(prediction, image, image, "flow", mask=mask, donor=np.zeros((2, 2, 2)))
        invalid = prediction.displacement.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            matcher.refine(prediction, image, image, "flow", mask=mask, local_candidates=(invalid,))

    def test_acceptor_must_share_feature_schema_and_calibrated_policy(self):
        prediction, image, config, mask = fixture()
        matcher = FeatureRematcher(config)
        class Acceptor:
            feature_schema = 'candidate_features_v1_12'
            max_update_px = config.max_update_px
            require_baseline_support = config.require_baseline_support
            gain_threshold = 0
            harm_threshold = .1
            def predict(self, features):
                # Toy observational score for interface verification only.
                return (features[..., 4] - features[..., 1], np.zeros(features.shape[:-1], np.float32))
        acceptor = Acceptor()
        result = matcher.refine(prediction, image, image, 'flow', mask=mask, acceptor=acceptor)
        self.assertEqual(result['feature_schema'], 'candidate_features_v1_12')
        self.assertEqual(result['candidate_features'].shape[-1], 12)
        self.assertTrue(result['accepted'].any())
        acceptor.max_update_px = config.max_update_px + 1
        with self.assertRaises(ValueError):
            matcher.refine(prediction, image, image, 'flow', mask=mask, acceptor=acceptor)
        acceptor.max_update_px = config.max_update_px
        acceptor.require_baseline_support = not config.require_baseline_support
        with self.assertRaises(ValueError):
            matcher.refine(prediction, image, image, 'flow', mask=mask, acceptor=acceptor)
        acceptor.require_baseline_support = config.require_baseline_support
        acceptor.feature_schema = 'wrong_schema'
        with self.assertRaises(ValueError):
            matcher.refine(prediction, image, image, 'flow', mask=mask, acceptor=acceptor)


if __name__ == "__main__":
    unittest.main()
