"""Native model checkpoints are portable weights, separate from exact resume."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from waft_repair.motion.checkpoint import (
    SCHEMA, export_model_checkpoint, import_checkpoint, load_model_checkpoint, repair_state_digest,
)
from waft_repair.motion.io import atomic_torch, sha256
from waft_repair.motion.model import MotionFeatureRepair, MotionRepairConfig
from waft_repair.motion.native import NativeRepairSystem
from waft_repair.motion.training import NativeTrainer, TrainConfig, expected_cursor


class NoForwardConsumer:
    def validate(self):
        return torch.device('cpu')


class MotionCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.model = MotionFeatureRepair()
        cls.portable = export_model_checkpoint(cls.model, 0, metadata={'purpose': 'CPU test'})

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, state, name='model.pt'):
        path = self.directory / name; atomic_torch(path, state)
        return path

    def original_checkpoint(self):
        model = MotionFeatureRepair()
        trainer = NativeTrainer(model, NoForwardConsumer(), TrainConfig(updates=6, warmup_updates=2),
                                model_config=asdict(model.config),
                                binding={'fixed_cases': ['a', 'b', 'c'], 'epochs': 2, 'shuffle_seed': 20261028},
                                core_binding={'state_digest': NativeRepairSystem.SOURCE_ONLY_STATE},
                                observation_binding={'source': '/private/old/taps.py'},
                                source_binding={'/private/old/training.py': '0' * 64})
        return trainer.state_dict(cursor=expected_cursor(trainer.binding, 0, 'init'))

    def test_portable_round_trip_exact_161_tensors_and_explicit_identity(self):
        source = self.save(self.portable)
        loaded, receipt = load_model_checkpoint(source, expected_sha256=sha256(source))
        self.assertFalse(loaded.training)
        self.assertEqual(receipt['schema'], SCHEMA)
        self.assertEqual(receipt['repair_tensor_count'], 161)
        self.assertEqual(receipt['parameter_count'], 3000576)
        self.assertEqual(receipt['completed_updates'], 0)
        self.assertEqual(receipt['repair_state_digest'], self.portable['repair_state_digest'])
        for key, value in loaded.state_dict().items():
            self.assertTrue(torch.equal(value, self.portable['repair'][key]), key)
        self.assertFalse(receipt['optimizer_loaded']); self.assertFalse(receipt['rng_restored'])

    def test_export_owns_tensors_and_validates_count_and_plain_finite_metadata(self):
        model = MotionFeatureRepair()
        packet = export_model_checkpoint(model, 12, metadata={'labels': ['native'], 'loss': .5})
        key = next(iter(packet['repair']))
        with torch.no_grad(): model.state_dict()[key].add_(1)
        self.assertFalse(torch.equal(model.state_dict()[key], packet['repair'][key]))
        for count in (-1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                export_model_checkpoint(self.model, count)
        for metadata in ({'value': float('nan')}, {'value': torch.zeros(1)}, {'value': (1, 2)}, {1: 'value'}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                export_model_checkpoint(self.model, 0, metadata=metadata)
        with self.assertRaises(ValueError):
            export_model_checkpoint(MotionFeatureRepair(MotionRepairConfig(profile='test')), 0)

    def test_signed_zero_bytes_are_preserved_and_authenticated(self):
        model = MotionFeatureRepair()
        positive = export_model_checkpoint(model, 0)['repair_state_digest']
        with torch.no_grad(): model.global_head.bias[0] = -0.0
        packet = export_model_checkpoint(model, 1)
        self.assertNotEqual(positive, packet['repair_state_digest'])
        loaded, _ = load_model_checkpoint(self.save(packet))
        self.assertTrue(bool(torch.signbit(loaded.global_head.bias[0])))
        self.assertEqual(repair_state_digest(loaded.state_dict()), packet['repair_state_digest'])

    def test_original_native_weights_import_without_optimizer_rng_or_host_paths(self):
        state = self.original_checkpoint()
        source = self.save(state, 'native-training.pt')
        torch.manual_seed(901); random.seed(901); np.random.seed(901)
        torch_before = torch.get_rng_state().clone()
        python_before = random.getstate(); numpy_before = np.random.get_state()
        model, receipt = load_model_checkpoint(source)
        self.assertEqual(receipt['schema'], state['schema'])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertEqual(python_before, random.getstate())
        self.assertTrue(np.array_equal(numpy_before[1], np.random.get_state()[1]))
        self.assertEqual(repair_state_digest(model.state_dict()), repair_state_digest(state['repair']))
        output = self.directory / 'portable.pt'
        converted = import_checkpoint(source, output, expected_sha256=sha256(source))
        packet = torch.load(output, map_location='cpu', weights_only=True)
        self.assertEqual(packet['schema'], SCHEMA)
        self.assertNotIn('/private/', json.dumps(packet['metadata']))
        self.assertEqual(packet['metadata']['origin_checkpoint_sha256'], sha256(source))
        self.assertNotIn('optimizer', packet); self.assertNotIn('torch_rng', packet)
        self.assertEqual(converted['checkpoint_sha256'], sha256(output))
        self.assertFalse(converted['optimizer_imported']); self.assertFalse(converted['rng_imported'])
        _, reloaded = load_model_checkpoint(output)
        self.assertEqual(reloaded['repair_state_digest'], receipt['repair_state_digest'])
        with self.assertRaises(ValueError): import_checkpoint(source, output)
        with self.assertRaises(ValueError): import_checkpoint(source, source)

    def test_portable_rejects_wrong_shapes_dtypes_nonfinite_trace_and_identity(self):
        def repair_change(name, function):
            return lambda state: function(state['repair'], name)
        first = next(iter(self.portable['repair']))
        corruptions = [
            lambda state: state.update(completed_updates=True),
            lambda state: state.update(core_state_digest='0' * 64),
            lambda state: state.update(waft_source_commit='0' * 40),
            lambda state: state.update(waft_checkpoint_sha256='0' * 64),
            lambda state: state.update(repair_state_digest='0' * 64),
            lambda state: state['repair'].update(trace_head=torch.zeros(1)),
            repair_change(first, lambda values, key: values.update({key: values[key].half()})),
            repair_change(first, lambda values, key: values.update({key: values[key][0]})),
            repair_change(first, lambda values, key: values[key].fill_(float('inf'))),
            lambda state: state['model_config'].update(seed=True),
        ]
        for index, corrupt in enumerate(corruptions):
            state = copy.deepcopy(self.portable); corrupt(state)
            with self.subTest(index=index), self.assertRaises(ValueError):
                load_model_checkpoint(self.save(state))

    def test_original_native_rejects_wrong_stage_auxiliary_cursor_core_and_schema(self):
        original = self.original_checkpoint()
        corruptions = [lambda state: state['config'].update(stage='feature'),
                       lambda state: state['config'].update(auxiliary_distillation_weight=.1),
                       lambda state: state['config'].update(epsilon=True),
                       lambda state: state['verified_core_binding'].update(state_digest='0' * 64),
                       lambda state: state.update(completed_updates=7),
                       lambda state: state['cursor'].update(next_case_id='wrong'),
                       lambda state: state.pop('torch_rng'),
                       lambda state: state['model_config'].update(trace_dim=16),
                       lambda state: state.update(schema='waft-repair-checkpoint')]
        for index, corrupt in enumerate(corruptions):
            state = copy.deepcopy(original); corrupt(state)
            with self.subTest(index=index), self.assertRaises(ValueError):
                load_model_checkpoint(self.save(state))

    def test_file_sha_mismatch_and_unsupported_device_rejected(self):
        source = self.save(self.portable)
        with self.assertRaises(ValueError):
            load_model_checkpoint(source, expected_sha256='0' * 64)
        with self.assertRaises(ValueError):
            load_model_checkpoint(source, device='meta')


if __name__ == '__main__':
    unittest.main()
