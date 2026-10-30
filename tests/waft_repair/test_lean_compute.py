"""Portable CPU checks for LiteR math, final readout, and bounded geometry."""
import copy
import unittest
from unittest.mock import patch

import torch

from waft_repair.lean import runtime as R
from waft_repair.lean import LiteFeatureRepair
from waft_repair.motion.geometry import resample_grid, sample_at_xy
from waft_repair.motion.native import native_task_loss
from lean_helpers import reference_step, systems


class LeanComputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.pair = (torch.linspace(0, 255, 3*16*24).reshape(1, 3, 16, 24),
                     torch.linspace(255, 0, 3*16*24).reshape(1, 3, 16, 24))
        self.gt = torch.ones(1, 2, 16, 24)
        self.valid = torch.ones(1, 16, 24, dtype=torch.bool)

    def assert_bytes_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertEqual(expected.shape, actual.shape)
            a = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            b = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            self.assertTrue(torch.equal(a, b))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_bytes_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for a, b in zip(expected, actual):
                self.assert_bytes_equal(a, b)
        else:
            self.assertEqual(expected, actual)

    def test_all5_and_two_updates_preserve_reference_R_AdamW_RNG_bytes(self):
        reference, lean = systems()
        reference.eval(); lean.eval()
        with torch.no_grad():
            expected, actual = reference(*self.pair), lean(*self.pair)
        for key in ('flow', 'info'):
            self.assert_bytes_equal(expected[key], actual[key])
        self.assertEqual(float(native_task_loss(expected, self.gt, self.valid)),
                         float(R.native_loss(actual, self.gt, self.valid)))
        reference_optimizer, lean_optimizer = R.make_optimizer(reference), R.make_optimizer(lean)
        for step in range(2):
            learning_rate = (step+1)*1e-4
            before_rng = torch.get_rng_state().clone()
            expected = reference_step(reference, reference_optimizer, self.pair, self.gt,
                self.valid, learning_rate=learning_rate)
            reference_rng = torch.get_rng_state().clone()
            torch.set_rng_state(before_rng)
            actual = R.train_step(lean, lean_optimizer, self.pair, self.gt, self.valid,
                learning_rate=learning_rate, activation_offload=False)
            self.assert_bytes_equal(reference_rng, torch.get_rng_state())
            self.assertEqual(expected, actual)
            self.assert_bytes_equal(reference.restorer.state_dict(), lean.restorer.state_dict())
            self.assert_bytes_equal(reference_optimizer.state_dict(), lean_optimizer.state_dict())
            reference.eval(); lean.eval()
            with torch.no_grad():
                expected_output, actual_output = reference(*self.pair), lean(*self.pair)
            for key in ('flow', 'info'):
                self.assert_bytes_equal(expected_output[key], actual_output[key])
        self.assertFalse(torch.cuda.is_initialized())

    def test_final_only_matches_last_flow_and_skips_four_readouts(self):
        _, system = systems()
        system.eval()
        with torch.no_grad():
            prepared = system.prepare_pair(*self.pair)
            repaired = system.repair_features(prepared)
            expected = system.solve_features(repaired, native_hw=prepared.native_hw,
                padding=prepared.padding)
            with patch.object(system.core.refine_net, 'forward', wraps=system.core.refine_net.forward) as refine, \
                    patch.object(system.core, 'upsample_data', wraps=system.core.upsample_data) as readout, \
                    patch.object(system, '_coords_grid_fn', wraps=system._coords_grid_fn) as coordinates:
                actual = system.solve_features(repaired, native_hw=prepared.native_hw,
                    padding=prepared.padding, final_only=True)
            self.assertEqual(refine.call_count, 5)
            self.assertEqual(readout.call_count, 1)
            self.assertEqual(coordinates.call_count, 1)
        self.assertEqual(len(actual['flow']), 1)
        self.assertEqual(len(actual['info']), 1)
        self.assert_bytes_equal(expected['flow'][-1], actual['flow'][0])
        self.assert_bytes_equal(expected['info'][-1], actual['info'][0])

    def test_EPE_only_is_exact_and_never_computes_MoL(self):
        _, system = systems()
        expected = R.evaluate(system, self.pair, self.gt, self.valid)
        with patch.object(R, 'native_loss', side_effect=AssertionError('Unneeded MoL')), \
                patch.object(system.core.refine_net, 'forward', wraps=system.core.refine_net.forward) as refine, \
                patch.object(system.core, 'upsample_data', wraps=system.core.upsample_data) as readout:
            actual = R.evaluate(system, self.pair, self.gt, self.valid, compute_mol=False)
        self.assertEqual(actual, {'EPE': expected['EPE']})
        self.assertEqual(refine.call_count, 5)
        self.assertEqual(readout.call_count, 1)

    def test_CPU_activation_offload_preserves_complete_update(self):
        _, ordinary = systems()
        offloaded = copy.deepcopy(ordinary)
        first, second = R.make_optimizer(ordinary), R.make_optimizer(offloaded)
        a = R.train_step(ordinary, first, self.pair, self.gt, self.valid,
            learning_rate=1e-4, activation_offload=False)
        b = R.train_step(offloaded, second, self.pair, self.gt, self.valid,
            learning_rate=1e-4, activation_offload=True)
        self.assertEqual(a, b)
        self.assert_bytes_equal(ordinary.restorer.state_dict(), offloaded.restorer.state_dict())
        self.assert_bytes_equal(first.state_dict(), second.state_dict())
        self.assertFalse(torch.cuda.is_initialized())

    def test_geometry_matches_original_resampling_and_is_bounded_unserialized(self):
        repair = LiteFeatureRepair()
        before = copy.deepcopy(repair.state_dict())
        first = repair._sampling_geometry(112, 112, device='cpu', dtype=torch.float32)
        second = repair._sampling_geometry(112, 112, device='cpu', dtype=torch.float32)
        self.assertTrue(all(a is b for a, b in zip(first, second)))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23)
            semantics = torch.randn(2, 16, 8, 8)
            context = torch.randn(2, 16, 28, 28)
        self.assert_bytes_equal(resample_grid(semantics, 14, 6.5, 4, 0., (28, 28)),
            sample_at_xy(semantics, first[0][None].expand(2, -1, -1, -1)))
        self.assert_bytes_equal(resample_grid(context, 4, 0., 2, 0., (56, 56)),
            sample_at_xy(context, first[1][None].expand(2, -1, -1, -1)))
        for width in range(2, repair.GEOMETRY_CACHE_LIMIT+3):
            repair._sampling_geometry(112, 112*width, device='cpu', dtype=torch.float32)
            self.assertLessEqual(len(repair._geometry_cache), repair.GEOMETRY_CACHE_LIMIT)
        self.assertEqual(len(repair._geometry_cache), repair.GEOMETRY_CACHE_LIMIT)
        self.assertNotIn((112, 112, torch.device('cpu'), torch.float32), repair._geometry_cache)
        self.assertEqual(list(repair.named_buffers()), [])
        self.assert_bytes_equal(before, repair.state_dict())

    def test_module_moves_clear_geometry_and_refresh_cached_device(self):
        _, system = systems()
        R.evaluate(system, self.pair, self.gt, self.valid, compute_mol=False)
        self.assertTrue(system.restorer._geometry_cache)
        self.assertTrue(system._pair_checked)
        system.double()
        self.assertFalse(system.restorer._geometry_cache)
        self.assertFalse(system._pair_checked)
        self.assertEqual(system.validate(), next(system.core.parameters()).device)
        system.float().cpu()
        R.evaluate(system, self.pair, self.gt, self.valid, compute_mol=False)
        self.assertTrue(system.restorer._geometry_cache)
        system.to('meta')
        self.assertEqual(system.validate(), torch.device('meta'))
        self.assertFalse(system.restorer._geometry_cache)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
