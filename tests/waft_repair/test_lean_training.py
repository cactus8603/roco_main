"""CPU regression tests for checkpoint binding, cumulative clocks and driver."""
import copy
import csv
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from waft_repair.lean import common as C, train as T, runtime as R
from lean_helpers import systems


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from waft_repair.lean import clock
        cls.timing = clock

    def parameters(self, **extra):
        return dict(dataset='sintel', epochs=4, total_updates=4, warmup_updates=1,
            peak_lr=3e-4, minimum_lr_fraction=.1, shuffle_seed=1,
            cases=[dict(case_id='toy', RGB_first={'path':'/raw/a','sha256':'a'*64,'bytes':1})],
            expected_core_digest='a'*64, **extra)

    def checkpoint(self, out, p):
        _, system = systems()
        optimizer = R.make_optimizer(system)
        with patch.object(torch.cuda, 'get_rng_state_all', return_value=[]):
            T.save_latest(out, system, optimizer, p, 0, 0)
        return system, optimizer

    def test_same_case_ID_cannot_resume_with_changed_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); p = self.parameters()
            system, optimizer = self.checkpoint(out, p)
            changed = copy.deepcopy(p)
            changed['cases'][0]['RGB_first']['sha256'] = 'b'*64
            with self.assertRaisesRegex(ValueError, 'contract'):
                T.restore(out/'latest.pt', system, optimizer, changed, restore_random=False)

    def test_model_configuration_and_implementation_binding_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); p = self.parameters()
            system, optimizer = self.checkpoint(out, p)
            state = C.load_torch(out/'latest.pt')
            for field in ('model_config', 'implementation'):
                bad = copy.deepcopy(state)
                if field == 'model_config':
                    bad[field]['seed'] += 1
                else:
                    bad['training_contract'][field]['train.py'] = 'b'*64
                with self.assertRaisesRegex(ValueError, 'configuration|contract'):
                    T.restore(None, system, optimizer, p, state=bad, restore_random=False)

    def test_CPU_restore_validation_does_not_restore_any_RNG_or_touch_CUDA(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); p = self.parameters()
            system, optimizer = self.checkpoint(out, p)
            before = torch.get_rng_state().clone()
            with patch.object(T, 'restore_rng', side_effect=AssertionError('Early RNG restore')):
                self.assertEqual(T.restore(out/'latest.pt', system, optimizer, p,
                                           restore_random=False), 0)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertFalse(torch.cuda.is_initialized())

    def test_unbound_checkpoint_is_rejected_without_legacy_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); p = self.parameters()
            system, optimizer = self.checkpoint(out, p)
            state = C.load_torch(out/'latest.pt'); state.pop('training_contract')
            with self.assertRaisesRegex(ValueError, 'Unbound legacy'):
                T.restore(None, system, optimizer, p, state=state, restore_random=False)

    def test_CPU_preflight_rejects_changed_AdamW_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory); p=self.parameters()
            system, optimizer=self.checkpoint(out, p)
            state=C.load_torch(out/'latest.pt')
            for flag, value in [('maximize',True),('amsgrad',True),('foreach',True),('fused',True)]:
                bad=copy.deepcopy(state)
                bad['optimizer_state']['param_groups'][0][flag]=value
                with self.assertRaisesRegex(ValueError, 'AdamW settings'):
                    T.restore(None, system, optimizer, p, state=bad, restore_random=False)

    def test_CPU_preflight_rejects_broken_RNG_without_changing_global_state(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory); p=self.parameters()
            system, optimizer=self.checkpoint(out, p)
            state=C.load_torch(out/'latest.pt')
            before=torch.get_rng_state().clone()
            for field in ('torch','python','numpy','CUDA'):
                bad=copy.deepcopy(state)
                bad['rng'].pop(field)
                with self.assertRaisesRegex(ValueError, 'RNG'):
                    T.restore(None, system, optimizer, p, state=bad, restore_random=False)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertFalse(torch.cuda.is_initialized())

    def make_clock_trainer(self):
        tick = [0.]
        budgets = self.timing.RuntimeBudgets()
        saved_clock = self.timing.ActiveClock(budgets, clock=lambda: tick[0])
        saved_clock.elapsed.update(active=4., pressure_wait=2., before_context_wait=3., overhead=6.)
        saved = saved_clock.snapshot()
        trainer = T.Trainer.__new__(T.Trainer)
        trainer.clock = self.timing.ActiveClock(budgets, clock=lambda: tick[0])
        tick[0] = 7.; trainer.clock.switch('before_context_wait')
        tick[0] = 16.; trainer.clock.switch('overhead')
        trainer.timing = SimpleNamespace(ActiveClock=SimpleNamespace(
            from_snapshot=lambda snapshot: self.timing.ActiveClock.from_snapshot(snapshot, clock=lambda: tick[0])))
        trainer.admission = {'carried_pressure_wait_seconds':25.}
        trainer.p = {}
        return trainer, saved

    def test_resume_keeps_setup_time_terminal_time_and_carried_wait(self):
        trainer, saved = self.make_clock_trainer()
        previous = copy.deepcopy(saved)
        previous['elapsed_seconds'].update(overhead=8., before_context_wait=5., active=5.)
        trainer.admission['previous_clock'] = previous
        trainer.resume_clock({'clock':saved})
        self.assertEqual(trainer.clock.snapshot()['elapsed_seconds'],
            dict(active=5., pressure_wait=25., before_context_wait=14., overhead=15.))

    def test_resume_cannot_change_or_reset_time_budgets(self):
        trainer, saved = self.make_clock_trainer()
        saved['budgets']['neural_active_seconds'] += 1
        with self.assertRaisesRegex(ValueError, 'budgets differ'):
            trainer.resume_clock({'clock':saved})
        trainer, saved = self.make_clock_trainer()
        saved['elapsed_seconds']['active'] = saved['budgets']['neural_active_seconds']+1
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            trainer.resume_clock({'clock':saved})

    def test_nonzero_cleanup_is_FAILED_with_this_invocation_token(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = SimpleNamespace(out=Path(directory), completed=0, system=None,
                clock=SimpleNamespace(snapshot=lambda:{}), run=lambda:{},
                cleanup=lambda:dict(references_cleared=True, allocated_bytes=1, reserved_bytes=0))
            argv = ['train.py','--dataset','sintel','--config','unused-config',
                    '--admission','unused','--run-token','current']
            with patch('sys.argv', argv), patch.object(T, 'Trainer', return_value=trainer):
                with self.assertRaisesRegex(RuntimeError, 'Training failed'):
                    T.main()
            result = C.read(Path(directory)/'result.json')
            self.assertEqual(result['status'], 'FAILED')
            self.assertEqual(result['run_token'], 'current')
            self.assertEqual(result['error']['type'], 'CleanupFailure')

    def test_driver_evaluates_only_once_at_final_epoch(self):
        _, system = systems(); optimizer = R.make_optimizer(system)
        pair = (torch.zeros(1,3,16,24), torch.ones(1,3,16,24))
        gt = torch.ones(1,2,16,24); valid = torch.ones(1,16,24,dtype=torch.bool)
        p = self.parameters(eval_every_epochs=1, backbone_checkpoint={'sha256':'b'*64})
        p['cases'] = [dict(case_id='toy')]
        with tempfile.TemporaryDirectory() as directory:
            trainer = T.Trainer.__new__(T.Trainer)
            trainer.out=Path(directory); trainer.p=p; trainer.system=system; trainer.optimizer=optimizer
            trainer.completed=0; trainer.initial={}; trainer.inputs=None
            trainer.setup=lambda:None; trainer.check_resources=lambda **_:None
            trainer.boundary=lambda _:optimizer.zero_grad(set_to_none=True)
            trainer.clock=SimpleNamespace(measure=lambda *_:nullcontext(), check=lambda:{})
            trainer.evaluate=Mock(return_value={'toy':{'EPE':.25}})
            with patch.object(T.data, 'load', return_value=(pair,gt,valid)), \
                    patch.object(torch.cuda, 'get_rng_state_all', return_value=[]):
                trainer.run()
            trainer.evaluate.assert_called_once_with()
            with (trainer.out/'metrics.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([row['mean_EPE'] for row in rows], ['', '', '', '0.25'])
            self.assertTrue(all(row['native_MoL']=='' for row in rows))


if __name__ == '__main__':
    unittest.main()
