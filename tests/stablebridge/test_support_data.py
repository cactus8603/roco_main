"""S04 cache/label contracts on synthetic arrays, with no dataset/model reads."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from stablebridge.support_data import (
    REMATCH_SHIFTS_XY, fixed_support_grid, support_guard_mask,
    processed_support_median, infer_support_case, save_support_cache,
    load_support_cache, offline_labels, sparse_gt_support, save_label_cache,
)
from stablebridge.quartet_experiment import restore_translation


class SyntheticProvider:
    def read_pair(self, task, scene, frame, context_hw, **kwargs):
        h, w = context_hw
        rgb = np.arange(h*w*3, dtype=np.uint8).reshape(h, w, 3)
        return {"image0": rgb, "image1": rgb.copy(),
                "roi_xyhw": (*kwargs["origin_xy"], h, w),
                "metadata": {"gt_read": False, "inputs": []}}

    def read_gt(self, *args, **kwargs):
        raise AssertionError("Inference attempted to read GT")


class SyntheticAdapter:
    task = "flow"
    device = "cpu"
    context_hw = (32, 64)

    def __init__(self):
        self.calls = []

    def predict(self, image0, image1, origin_xy):
        self.calls.append((image0.copy(), image1.copy(), origin_xy))
        h, w = self.context_hw
        field = np.zeros((2, h, w), np.float32)
        field[0] = 2
        return SimpleNamespace(displacement=field, raw_uncertainty=np.zeros((h, w), np.float32),
            source_features=np.ones((3, h//16, w//16), np.float32),
            target_features=np.ones((3, h//16, w//16), np.float32),
            token_centers_xy=np.zeros((h//16, w//16, 2), np.float32),
            metadata={"forward_calls": 1, "frozen": True})


class SupportDataTests(unittest.TestCase):
    def test_fixed_grid_and_inclusive_euclidean_guard_are_gt_independent(self):
        xy = fixed_support_grid()
        self.assertEqual(xy.shape, (120, 2))
        np.testing.assert_array_equal(xy[0], (15, 15))
        np.testing.assert_array_equal(xy[-1], (367, 303))
        guard = support_guard_mask((32, 32), [[15, 15]], radius_px=8)
        self.assertTrue(guard[15, 23])
        self.assertFalse(guard[15, 24])
        self.assertFalse(guard[23, 23])

    def test_median_excludes_invalid_whole_vectors_and_has_identity_fallback(self):
        bank = np.zeros((3, 2, 1, 3), np.float32)
        bank[0] = 1
        bank[1] = 3
        bank[2] = 100
        bank[2, 0, 0, 0] = np.nan
        valid = np.ones((3, 1, 3), bool)
        valid[2, 0, 1] = False
        valid[:, 0, 2] = False
        np.testing.assert_array_equal(processed_support_median(bank, valid),
                                      np.array([[[2, 2, 1]], [[2, 2, 1]]], np.float32))

    def test_twelve_forward_cache_preserves_s02_ops_and_wrap_validity_without_gt(self):
        adapter = SyntheticAdapter()
        result = infer_support_case(adapter, SyntheticProvider(),
                   {"task": "flow", "scene": "0001", "frame": 1, "profile": {"name": "clean"}},
                   context_hw=(32, 64), origin_xy=(0, 0))
        arrays, meta = result["arrays"], result["metadata"]
        self.assertEqual(len(adapter.calls), 12)
        self.assertEqual(meta["cost"]["forward_calls"], 12)
        self.assertEqual(meta["cost"]["incremental_E0_processing_forward_calls"], 11)
        self.assertFalse(meta["gt_read"])
        self.assertNotIn("gt4", arrays)
        self.assertNotIn("fixed_q", arrays)
        self.assertEqual(arrays["e0_bank"].shape, (12, 2, 32, 64))
        self.assertTrue(arrays["e0_valid"][:3].all())
        raw = arrays["image0"]
        for index, (dx, dy) in enumerate(REMATCH_SHIFTS_XY, 3):
            np.testing.assert_array_equal(adapter.calls[index][0], np.roll(raw, (dy, dx), axis=(0, 1)))
            _, expected_valid = restore_translation(arrays["initial"], (dx, dy))
            np.testing.assert_array_equal(arrays["e0_valid"][index], expected_valid)

    def test_cache_schema_and_hash_reject_gt_contamination_or_mutation(self):
        result = infer_support_case(SyntheticAdapter(), SyntheticProvider(),
                   {"task": "flow", "scene": "0001", "frame": 1},
                   context_hw=(32, 64), origin_xy=(0, 0))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"inference.npz"
            save_support_cache(path, result)
            loaded = load_support_cache(path)
            np.testing.assert_array_equal(loaded["arrays"]["e0_bank"], result["arrays"]["e0_bank"])
            contaminated = {"arrays": {**result["arrays"], "fixed_q": np.ones((32, 64), bool)},
                            "metadata": result["metadata"]}
            with self.assertRaises(ValueError):
                save_support_cache(Path(temporary)/"bad.npz", contaminated)
            with path.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_support_cache(path)

    def test_strong_q_uses_valid_extra_candidates_and_keeps_guard_in_primary(self):
        initial = np.full((2, 3, 4), 3., np.float32)
        gt4 = np.zeros((4, 2, 3, 4), np.float32)
        bank = np.repeat(initial[None], 12, axis=0)
        valid = np.ones((12, 3, 4), bool)
        bank[3, :, 1, 1] = 0
        bank[4, :, 2, 2] = 0
        valid[4, 2, 2] = False  # Wrapped perfect answers must not resolve Q.
        labels = offline_labels(initial, bank, valid, gt4, [[0, 0]], guard_px=0)
        self.assertFalse(labels["fixed_q"][1, 1])
        self.assertTrue(labels["fixed_q"][2, 2])
        self.assertTrue(labels["fixed_q"][0, 0])  # Primary never shrinks by support geometry.
        self.assertFalse(labels["fixed_q_outside_support_guard"][0, 0])
        self.assertEqual(int(labels["fixed_q"].sum()), 11)

    def test_sparse_gt_uses_fixed_branch_and_only_exact_support_values(self):
        gt4 = np.zeros((4, 2, 3, 4), np.float32)
        gt4[0, :, 1, 2] = (10, 20)
        gt4[1, :, 1, 2] = (1, 2)
        gt4[0, 0, 2, 3] = np.nan
        values, valid = sparse_gt_support(gt4, [[2, 1], [3, 2]])
        np.testing.assert_array_equal(values[0], (10, 20))
        np.testing.assert_array_equal(values[1], (0, 0))
        np.testing.assert_array_equal(valid, (True, False))
        with self.assertRaises(ValueError):
            sparse_gt_support(gt4, [[2.25, 1]])

    def test_labels_are_physically_distinct_and_cannot_load_as_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"labels.npz"
            metadata = {"gt_read": True, "role": "training_targets_and_offline_evaluation_only"}
            saved = save_label_cache(path, {"gt4": np.zeros((4, 2, 3, 4))}, metadata)
            self.assertEqual(len(saved["artifact"]["sha256"]), 64)
            with self.assertRaises(ValueError):
                load_support_cache(path)
            self.assertTrue(json.loads(path.with_suffix(".json").read_text())["gt_read"])


if __name__ == "__main__":
    unittest.main()
