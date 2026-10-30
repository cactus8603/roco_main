"""CPU-only launcher fixtures: no subprocess, service, signal or CUDA calls."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from waft_repair.lean import common as C
from waft_repair.lean import launch as P
from waft_repair.lean.clock import ActiveClock, RuntimeBudgets


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)/'training'
        self.control = self.out/'runtime'
        self.config_path = Path(self.temporary.name)/'config.json'
        self.protocol = dict(artifact_root=str(self.control), allowed_gpu_uuids=['GPU-toy'],
                             runtime=dict(running_pressure_wait_seconds=21600))
        C.write(self.config_path, dict(resource_control='mock-only', datasets=dict(kitti=dict(
            output_dir=str(self.out), resource_protocol=self.protocol, child_timeout_seconds=10))))
        self.args = SimpleNamespace(config=self.config_path, dataset='kitti', resume=None)
        self.owner = dict(pid=111, start_ticks=100, uid=1000)
        self.rc = SimpleNamespace(admit=Mock(side_effect=self.admit), release_own_slot=Mock())
        self.child = Mock(pid=321)
        self.child.wait.return_value = 0
        self.cleanup = dict(references_cleared=True, allocated_bytes=0, reserved_bytes=0)
        self.result_token = None
        self.result_status = 'COMPLETE'
        self.write_result = True
        self.final_clock = dict(schema='mock-clock', elapsed_seconds=dict(pressure_wait=22.5))

    def admit(self, protocol):
        return dict(owner=copy.deepcopy(self.owner), selected_gpu_uuid='GPU-toy'), object()

    def popen(self, argv, **kwargs):
        if self.write_result:
            token = argv[argv.index('--run-token')+1]
            C.write(self.out/'result.json', dict(status=self.result_status,
                run_token=self.result_token or token, cleanup=self.cleanup,
                clock=self.final_clock, training_contract_digest='contract-toy'))
        return self.child

    def run_parent(self, *, popen=None, pidfd_error=None):
        with patch.dict(P.os.environ, {'CUDA_VISIBLE_DEVICES':''}), \
             patch.object(P, 'load_site_adapter', return_value=self.rc), \
             patch.object(P.subprocess, 'Popen', side_effect=popen or self.popen), \
             patch.object(P.os, 'pidfd_open', side_effect=pidfd_error, return_value=77), \
             patch.object(P.os, 'close'), \
             patch.object(P.signal, 'pidfd_send_signal') as sent:
            result = P.run(self.args)
        return result, sent

    def run_failure(self, **kwargs):
        with self.assertRaises(RuntimeError):
            self.run_parent(**kwargs)
        return C.read(self.control/'run.json')

    def make_completed_previous(self):
        self.run_parent()
        previous = C.read(self.control/'run.json')
        C.write(self.control/'running_resource_wait.json', dict(
            status='RUNNING_HOST_PRESSURE_RECOVERED', trial_owner=copy.deepcopy(self.owner),
            selected_gpu_uuid='GPU-toy', budget_seconds=21600, cumulative_wait_seconds=22.5))
        self.args.resume = Path(self.temporary.name)/'latest.pt'
        self.owner = dict(pid=222, start_ticks=200, uid=1000)
        self.rc.admit.reset_mock()
        self.rc.release_own_slot.reset_mock()
        return previous

    def test_complete_requires_matching_child_result_and_zero_cleanup(self):
        run, sent = self.run_parent()
        self.assertEqual(run['status'], 'COMPLETE')
        self.assertEqual(run['run_token'], C.read(self.out/'result.json')['run_token'])
        self.assertEqual(run['final_clock'], self.final_clock)
        self.assertEqual(run['training_contract_digest'], 'contract-toy')
        self.assertTrue(run['actual_wait']['actual_wait_completed'])
        self.assertTrue(run['own_lease_released'])
        self.rc.release_own_slot.assert_called_once()
        sent.assert_not_called()

    def test_second_timeout_persists_unresolved_wait_and_keeps_reservation(self):
        self.child.wait.side_effect = [subprocess.TimeoutExpired('toy',10), subprocess.TimeoutExpired('toy',300)]
        run = self.run_failure()
        self.assertEqual(run['status'], 'WAIT_UNRESOLVED')
        self.assertFalse(run['actual_wait']['actual_wait_completed'])
        self.assertIsNone(run['actual_wait']['returncode'])
        self.assertFalse(run['own_lease_released'])
        self.assertEqual(self.child.wait.call_count,2)
        self.rc.release_own_slot.assert_not_called()

    def test_interrupted_zero_exit_is_failed_after_real_wait_and_cleanup(self):
        self.child.wait.side_effect = [KeyboardInterrupt(),0]
        run = self.run_failure()
        self.assertEqual(run['status'],'FAILED')
        self.assertTrue(run['actual_wait']['actual_wait_completed'])
        self.assertTrue(run['actual_wait']['interrupted'])
        self.assertTrue(run['own_lease_released'])

    def test_result_from_another_invocation_cannot_release_reservation(self):
        self.result_token = 'previous-invocation'
        run = self.run_failure()
        self.assertEqual(run['status'],'FAILED')
        self.assertTrue(run['actual_wait']['actual_wait_completed'])
        self.assertFalse(run['own_lease_released'])
        self.rc.release_own_slot.assert_not_called()

    def test_nonzero_cleanup_cannot_release_reservation(self):
        self.cleanup['allocated_bytes'] = 1
        run = self.run_failure()
        self.assertEqual(run['status'],'FAILED')
        self.assertEqual(run['CUDA_cleanup']['allocated_bytes'],1)
        self.rc.release_own_slot.assert_not_called()

    def test_spawn_failure_records_no_claimed_child_wait(self):
        def fail(*args, **kwargs):
            raise OSError('fixture spawn failure')
        run = self.run_failure(popen=fail)
        self.assertEqual(run['status'],'FAILED')
        self.assertIsNone(run['actual_wait'])
        self.assertFalse(run['own_lease_released'])

    def test_pidfd_failure_still_waits_and_records_the_failure(self):
        run = self.run_failure(pidfd_error=OSError('fixture pidfd failure'))
        self.assertEqual(run['status'],'FAILED')
        self.assertTrue(run['actual_wait']['actual_wait_completed'])
        self.assertEqual(run['errors'][0]['stage'],'pidfd_open')
        self.child.wait.assert_called_once_with(timeout=10)

    def test_missing_result_does_not_reuse_previous_cleanup(self):
        self.write_result = False
        run = self.run_failure()
        self.assertEqual(run['status'],'FAILED')
        self.assertTrue(run['actual_wait']['actual_wait_completed'])
        self.assertIsNone(run['CUDA_cleanup'])
        self.rc.release_own_slot.assert_not_called()

    def test_resume_rebinds_same_output_pressure_time_without_reset(self):
        previous = self.make_completed_previous()
        run, _ = self.run_parent()
        admission = C.read(self.control/'admission.json')
        ledger = C.read(self.control/'running_resource_wait.json')
        self.assertEqual(run['status'],'COMPLETE')
        self.assertNotEqual(run['run_token'],previous['run_token'])
        self.assertEqual(admission['owner'],self.owner)
        self.assertEqual(admission['carried_pressure_wait_seconds'],22.5)
        self.assertEqual(admission['previous_clock'],self.final_clock)
        self.assertEqual(ledger['trial_owner'],self.owner)
        self.assertEqual(ledger['cumulative_wait_seconds'],22.5)
        self.assertEqual(ledger['budget_seconds'],21600)
        self.assertEqual(ledger['remaining_wait_seconds'],21600-22.5)
        self.assertEqual(C.read(self.control/'previous_run.json')['run'],previous)

    def test_resume_rejects_wrong_ledger_owner_before_admission(self):
        previous = self.make_completed_previous()
        ledger = C.read(self.control/'running_resource_wait.json')
        ledger['trial_owner'] = dict(pid=999)
        C.write(self.control/'running_resource_wait.json',ledger)
        with self.assertRaisesRegex(ValueError,'ledger owner'):
            self.run_parent()
        self.rc.admit.assert_not_called()
        self.assertEqual(C.read(self.control/'run.json'),previous)

    def test_resume_rejects_missing_cleanup_release_before_admission(self):
        self.make_completed_previous()
        previous = C.read(self.control/'run.json')
        previous['own_lease_released'] = False
        C.write(self.control/'run.json',previous)
        with self.assertRaisesRegex(ValueError,'not completely reaped'):
            self.run_parent()
        self.rc.admit.assert_not_called()

    def test_resume_rejects_previous_result_token_mismatch(self):
        self.make_completed_previous()
        result = C.read(self.out/'result.json')
        result['run_token'] = 'different-previous-run'
        C.write(self.out/'result.json',result)
        with self.assertRaisesRegex(ValueError,'not completely reaped'):
            self.run_parent()
        self.rc.admit.assert_not_called()

    def test_managed_launcher_keeps_resource_limits_and_uses_package_modules(self):
        with patch.object(P,'load_site_adapter',return_value=self.rc), \
             patch.object(P.subprocess,'run') as managed:
            P.managed(self.args)
        argv = managed.call_args.args[0]
        for field in ('MemoryHigh=14G','MemoryMax=16G','MemorySwapMax=0','TasksMax=512',
                      'RuntimeMaxSec=50000','Restart=no','KillSignal=SIGINT',
                      'TimeoutStopSec=300','SendSIGKILL=no'):
            self.assertIn('--property='+field,argv)
        self.assertIn('waft_repair.lean.launch',argv)
        self.assertEqual(argv[argv.index('-m')+1],'waft_repair.lean.launch')
        self.assertIn('--managed-child',argv)
        self.assertEqual(argv[argv.index('--config')+1],str(self.config_path.resolve()))
        self.assertEqual(C.PYTHON,Path(sys.executable))

    def test_missing_site_adapter_fails_before_service_or_child_launch(self):
        config = C.read(self.config_path)
        del config['resource_control']
        C.write(self.config_path,config)
        with patch.object(P.subprocess,'run') as service, patch.object(P.subprocess,'Popen') as child:
            with self.assertRaisesRegex(ValueError,'site-adapter'):
                P.managed(self.args)
            with self.assertRaisesRegex(ValueError,'site-adapter'):
                P.run(self.args)
        service.assert_not_called()
        child.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_incomplete_site_adapter_cannot_omit_running_or_cuda_gates(self):
        path = Path(self.temporary.name)/'incomplete_adapter.py'
        path.write_text('def admit(protocol): return None\n')
        with self.assertRaisesRegex(ValueError,'all admission, CUDA, running and release gates'):
            C.load_site_adapter(dict(resource_control=str(path)))

    def test_clock_resume_keeps_cumulative_durations_and_enforces_original_budget(self):
        now = [0.]
        budgets = RuntimeBudgets(neural_active_seconds=5, before_context_wait_seconds=5,
            running_pressure_wait_seconds=5, overhead_seconds=5, shutdown_margin_seconds=1)
        clock = ActiveClock(budgets,clock=lambda:now[0])
        with clock.measure('active'):
            now[0] = 3.
        saved = clock.check()
        now[0] = 1000.
        resumed = ActiveClock.from_snapshot(saved,clock=lambda:now[0])
        self.assertEqual(resumed.check()['elapsed_seconds']['active'],3.)
        self.assertEqual(resumed.check()['budgets'],saved['budgets'])
        with resumed.measure('active'):
            now[0] += 3.
        with self.assertRaisesRegex(ValueError,'active runtime budget exhausted'):
            resumed.check()


if __name__ == '__main__':
    unittest.main()
