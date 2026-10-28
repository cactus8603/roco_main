"""CPU numerical contracts for the native-flow trainer; no WAFT weights needed."""
from __future__ import annotations

import copy
import io
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from waft_repair.motion.native import native_task_loss
from waft_repair.motion.training import (
    NativePilotTrainer, NativeTrainer, TrainConfig, cpu_tree, epoch_order,
    expected_cursor, full_valid_epe, learning_rate, model_inputs,
)


class TinyRepair(nn.Module):
    """A small differentiable repair fixture; not an optical-flow baseline."""
    def __init__(self):
        super().__init__()
        self.body = nn.Conv2d(64, 8, 1)
        self.head = nn.Conv2d(8, 64, 1)
        self.dropout = nn.Dropout2d(.2)

    def forward(self, original, sigma, observations, *baseline,
                native_hw, padding, return_aux=False, motion_prior=None):
        b, frames, channels, height, width = original.shape
        hidden = torch.tanh(self.body(original.reshape(b * frames, channels, height, width)))
        delta = self.head(self.dropout(hidden)).reshape_as(original)
        return original + sigma.reshape(b, 1, 64, 1, 1) * delta


class FrozenConsumer:
    def __init__(self):
        self.core = nn.Conv2d(64, 6, 1)
        self.core.requires_grad_(False).eval()
        self.before = cpu_tree(self.core.state_dict())

    def solve_features(self, features, *, native_hw, padding):
        value = self.core(features[:, 0] + .2 * features[:, 1])
        value = F.interpolate(value, size=native_hw, mode='bilinear', align_corners=False)
        return {'flow': [value[:, :2] * (1.0 + .05 * i) for i in range(5)],
                'info': [value[:, 2:] + .01 * i for i in range(5)]}

    def validate(self):
        if any(not torch.equal(value, self.before[key]) or value.requires_grad
               for key, value in self.core.state_dict().items()):
            raise ValueError('Frozen fixture changed')
        if any(parameter.grad is not None for parameter in self.core.parameters()):
            raise ValueError('Frozen fixture received parameter gradients')


def fixture():
    generator = torch.Generator().manual_seed(987)
    inputs = {
        'original': torch.randn((1, 2, 64, 3, 4), generator=generator),
        'sigma': torch.rand((1, 64), generator=generator) + .5,
        'observations': {key: torch.randn((1, 2, channels, 2, 2), generator=generator)
                         for key, channels in (('cnn_pre_s2', 64), ('cnn_pre_s4', 128),
                                               ('dav2_block6', 384), ('dav2_block12', 384))},
        'baseline_flow': torch.zeros((1, 2, 6, 8)),
        'baseline_info': torch.zeros((1, 4, 6, 8)),
        'native_hw': (6, 8), 'padding': (0, 0, 0, 0),
    }
    return inputs, torch.randn((1, 2, 6, 8), generator=generator), torch.ones((1, 6, 8), dtype=torch.bool)


def make_trainer(*, updates=6):
    torch.manual_seed(123); np.random.seed(123); random.seed(123)
    model = TinyRepair()
    consumer = FrozenConsumer()
    config = TrainConfig(updates=updates, warmup_updates=2)
    binding = {'fixed_cases': ['a', 'b', 'c'], 'epochs': updates // 3, 'shuffle_seed': 20261028}
    return NativeTrainer(model, consumer, config, model_config={'fixture': 'tiny-v1'},
                         binding=binding, core_binding={'fixture': 'frozen'},
                         observation_binding={'fixture': 'original-four-taps'},
                         source_binding={'packaged-fixture': 'v1'})


def assert_tree_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor)
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            assert_tree_equal(test, first, second)
    else:
        test.assertEqual(left, right)


class NativeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_full_run_defaults_and_lr_endpoints(self):
        config = TrainConfig()
        self.assertEqual((config.updates, config.warmup_updates), (5120, 160))
        self.assertEqual(learning_rate(config, 1), 3e-4 / 160)
        self.assertEqual(learning_rate(config, 160), 3e-4)
        self.assertEqual(learning_rate(config, 5121), config.peak_lr * .1)
        self.assertIs(NativeTrainer, NativePilotTrainer)
        for kwargs in ({'auxiliary_distillation_weight': 1}, {'warmup_updates': 5120},
                       {'stage': 'feature'}, {'epsilon': 0}, {'peak_lr': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainConfig(**kwargs)

    def test_shuffle_and_exact_next_input(self):
        before = random.getstate()
        ids = list('abcdef')
        expected = list(range(6)); random.Random(20261029).shuffle(expected)
        self.assertEqual(epoch_order(ids, 20261028, 1), expected)
        self.assertEqual(random.getstate(), before)
        binding = {'fixed_cases': ids, 'epochs': 2, 'shuffle_seed': 20261028}
        for n in range(13):
            cursor = expected_cursor(binding, n, 'fixture')
            self.assertEqual(cursor['next_offset'], n % 6)
            self.assertEqual(cursor['next_case_id'], None if n == 12 else
                             ids[epoch_order(ids, 20261028, n // 6 + 1)[n % 6]])
        with self.assertRaises(ValueError):
            epoch_order(['a', 'a'], 123, 1)

    def test_labels_and_gradients_cannot_enter_model_inputs(self):
        prepared, gt, _ = fixture()
        self.assertEqual(model_inputs(prepared).keys(), prepared.keys())
        for key in ('gt', 'target', 'teacher', 'quality', 'baseline_output'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                model_inputs({**prepared, key: gt})
        contaminated = {**prepared, 'original': prepared['original'].clone().requires_grad_()}
        with self.assertRaises(ValueError):
            model_inputs(contaminated)

    def test_first_step_updates_body_and_head_and_keeps_consumer_frozen(self):
        trainer = make_trainer(); prepared, gt, valid = fixture()
        before = cpu_tree(trainer.model.state_dict())
        metrics = trainer.step(prepared, gt, valid)
        self.assertEqual(metrics['completed_updates'], 1)
        self.assertGreater(metrics['gradient_norm_before_clip'], 0)
        for key, value in trainer.model.state_dict().items():
            self.assertFalse(torch.equal(value, before[key]), key)
        self.assertEqual(len(trainer.optimizer.state), len(list(trainer.model.parameters())))
        trainer.consumer.validate()

    def test_cpu_resume_reproduces_weights_adam_cursor_and_all_rng(self):
        trainer = make_trainer(); prepared, gt, valid = fixture()
        trainer.step(prepared, gt, valid); trainer.step(prepared, gt, valid)
        saved = trainer.state_dict(cursor=expected_cursor(trainer.binding, 2, 'two'))
        for _ in range(4):
            trainer.step(prepared, gt, valid)
        expected = trainer.state_dict(cursor=expected_cursor(trainer.binding, 6, 'final'))
        expected_draws = (torch.rand(3), random.random(), np.random.rand())
        resumed = make_trainer(); resumed.load_state_dict(saved)
        self.assertEqual(resumed.completed_updates, 2)
        for _ in range(4):
            resumed.step(prepared, gt, valid)
        actual = resumed.state_dict(cursor=expected_cursor(resumed.binding, 6, 'final'))
        assert_tree_equal(self, expected, actual)
        actual_draws = (torch.rand(3), random.random(), np.random.rand())
        assert_tree_equal(self, expected_draws, actual_draws)
        with self.assertRaises(ValueError):
            resumed.step(prepared, gt, valid)

    def test_checkpoint_copies_do_not_alias_live_model_or_adam(self):
        trainer = make_trainer(); prepared, gt, valid = fixture()
        trainer.step(prepared, gt, valid)
        saved = trainer.state_dict(cursor=expected_cursor(trainer.binding, 1, 'one'))
        first = next(iter(saved['repair']))
        saved['repair'][first].add_(100)
        self.assertFalse(torch.equal(saved['repair'][first], trainer.model.state_dict()[first]))
        saved['optimizer']['state'][0]['exp_avg'].add_(100)
        self.assertFalse(torch.equal(saved['optimizer']['state'][0]['exp_avg'],
                                     trainer.optimizer.state_dict()['state'][0]['exp_avg']))

    def test_full_checkpoint_round_trips_weights_only_without_pickle_extensions(self):
        trainer = make_trainer(); prepared, gt, valid = fixture()
        trainer.step(prepared, gt, valid)
        saved = trainer.state_dict(cursor=expected_cursor(trainer.binding, 1, 'portable'))
        stream = io.BytesIO(); torch.save(saved, stream); stream.seek(0)
        loaded = torch.load(stream, map_location='cpu', weights_only=True)
        trainer.validate_checkpoint(loaded)
        assert_tree_equal(self, saved, loaded)

    def test_corrupt_checkpoints_rejected_before_any_mutation(self):
        trainer = make_trainer(); prepared, gt, valid = fixture()
        trainer.step(prepared, gt, valid)
        saved = trainer.state_dict(cursor=expected_cursor(trainer.binding, 1, 'before'))
        corruptions = [lambda s: s.update(completed_updates=3),
                       lambda s: s['verified_core_binding'].update(fixture='other'),
                       lambda s: s['cursor'].update(next_case_id='wrong'),
                       lambda s: s['optimizer']['state'][0]['exp_avg_sq'].fill_(-1),
                       lambda s: s['repair'][next(iter(s['repair']))].fill_(float('nan')),
                       lambda s: s.update(torch_rng=torch.zeros(2, dtype=torch.uint8))]
        for corrupt in corruptions:
            state = copy.deepcopy(saved); corrupt(state)
            with self.subTest(corrupt=corrupt), self.assertRaises((ValueError, RuntimeError)):
                trainer.load_state_dict(state)
            actual = trainer.state_dict(cursor=expected_cursor(trainer.binding, 1, 'before'))
            assert_tree_equal(self, saved, actual)

    def test_checkpoint_commit_failure_rolls_back_weights_optimizer_rng_and_cursor(self):
        trainer = make_trainer(); prepared, gt, valid = fixture()
        trainer.step(prepared, gt, valid)
        older = trainer.state_dict(cursor=expected_cursor(trainer.binding, 1, 'older'))
        trainer.step(prepared, gt, valid)
        current = trainer.state_dict(cursor=expected_cursor(trainer.binding, 2, 'current'))
        original_load = trainer.optimizer.load_state_dict
        failures = []
        def fail_once(state):
            if not failures:
                failures.append(True)
                raise RuntimeError('simulated optimizer installation failure')
            return original_load(state)
        trainer.optimizer.load_state_dict = fail_once
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            trainer.load_state_dict(older)
        actual = trainer.state_dict(cursor=expected_cursor(trainer.binding, 2, 'current'))
        assert_tree_equal(self, current, actual)

    def test_native_loss_uses_all_five_and_original_label_population(self):
        gt = torch.zeros(1, 2, 2, 2)
        gt[:, :, 0, 0] = float('nan')
        valid = torch.ones(1, 2, 2, dtype=torch.bool); valid[:, 0, 0] = False
        flow = torch.ones(1, 2, 2, 2, requires_grad=True)
        info = torch.zeros(1, 4, 2, 2, requires_grad=True)
        output = {'flow': [flow] * 5, 'info': [info] * 5}
        loss = native_task_loss(output, gt, valid)
        # At zero mixture logits and zero log-scales: log(2) + |residual|.
        expected = (torch.log(torch.tensor(2.0)) + 1) * sum(.85 ** i for i in range(5))
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward(); self.assertTrue(torch.isfinite(flow.grad).all())
        self.assertEqual(float(full_valid_epe(flow.detach(), gt, valid)), 2 ** .5)
        with self.assertRaises(ValueError):
            native_task_loss({'flow': [flow] * 4, 'info': [info] * 4}, gt, valid)


if __name__ == '__main__':
    unittest.main()
