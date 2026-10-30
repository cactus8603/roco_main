"""One CPU parent, one owned GPU child, one compact lifecycle record."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import traceback
import uuid

from .common import ROOT, PYTHON, read, write, load_site_adapter


def managed(args):
    """Create the bounded CPU-parent service only on an explicit CLI launch."""
    load_site_adapter(read(args.config))
    name = 'waft-repair-lean-'+args.dataset+'.service'
    argv = ['systemd-run','--user','--unit='+name,'--property=Type=exec',
        '--property=WorkingDirectory='+str(ROOT),'--property=MemoryHigh=14G',
        '--property=MemoryMax=16G','--property=MemorySwapMax=0','--property=TasksMax=512',
        '--property=RuntimeMaxSec=50000','--property=Restart=no','--property=KillSignal=SIGINT',
        '--property=TimeoutStopSec=300','--property=SendSIGKILL=no',
        '--setenv=CUDA_VISIBLE_DEVICES=','--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=PYTHONPATH='+str(ROOT/'src'),'--setenv=MALLOC_ARENA_MAX=2',
        str(PYTHON),'-B','-m','waft_repair.lean.launch','--managed-child','--dataset',args.dataset,
        '--config',str(args.config.resolve())]
    if args.resume:
        argv += ['--resume',str(args.resume.resolve())]
    subprocess.run(argv, check=True)


def wait_child(child, pidfd, timeout):
    interrupted = False
    signal_error = None
    try:
        code = child.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as first:
        interrupted = True
        if pidfd is not None:
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGINT)
            except ProcessLookupError:
                pass  # Already exited: the actual wait remains mandatory.
            except BaseException as exc:
                signal_error = dict(type=type(exc).__name__, message=str(exc))
        try:
            code = child.wait(timeout=300)
        except BaseException as exc:
            return dict(actual_wait_completed=False, returncode=None, interrupted=True,
                child_pid=child.pid, error=dict(type=type(exc).__name__, message=str(exc)),
                first_interruption=type(first).__name__, signal_error=signal_error)
    except BaseException as exc:
        return dict(actual_wait_completed=False, returncode=None, interrupted=False,
            child_pid=child.pid, error=dict(type=type(exc).__name__, message=str(exc)))
    return dict(actual_wait_completed=True, returncode=code, interrupted=interrupted,
                child_pid=child.pid, signal_error=signal_error)


def clean_cleanup(cleanup):
    return (isinstance(cleanup, dict) and cleanup.get('references_cleared') is True
        and type(cleanup.get('allocated_bytes')) is int and cleanup['allocated_bytes'] == 0
        and type(cleanup.get('reserved_bytes')) is int and cleanup['reserved_bytes'] == 0)


def resume_evidence(out, p):
    """Carry bounded pressure time only from one completely reaped prior run."""
    control = out/'runtime'
    paths = dict(admission=control/'admission.json', run=control/'run.json',
                 result=out/'result.json', ledger=Path(p['resource_protocol']['artifact_root'])/'running_resource_wait.json')
    if not any(path.exists() for path in paths.values()):
        return None
    previous = {key: read(path) for key, path in paths.items() if path.exists()}
    if not {'admission', 'run', 'result'} <= previous.keys():
        raise ValueError('Resume requires the previous admission, actual wait and cleanup records')
    admission, run, result = (previous[key] for key in ('admission','run','result'))
    waited = run.get('actual_wait') or {}
    if (run.get('status') not in ('COMPLETE','FAILED')
            or waited.get('actual_wait_completed') is not True
            or type(waited.get('returncode')) is not int
            or type(waited.get('child_pid')) is not int or waited['child_pid'] <= 0
            or run.get('child_pid') != waited['child_pid']
            or not clean_cleanup(run.get('CUDA_cleanup'))
            or run.get('own_lease_released') is not True
            or run.get('admission_owner') != admission.get('owner')
            or run.get('selected_gpu_uuid') != admission.get('selected_gpu_uuid')
            or not isinstance(run.get('run_token'), str) or not run['run_token']
            or admission.get('run_token') != run['run_token']
            or result.get('run_token') != run['run_token']
            or result.get('cleanup') != run.get('CUDA_cleanup')
            or result.get('clock') != run.get('final_clock')
            or result.get('training_contract_digest') != run.get('training_contract_digest')):
        raise ValueError('Previous run is not completely reaped, cleaned and bound; preserve its reservation')
    budget = p['resource_protocol']['runtime']['running_pressure_wait_seconds']
    if budget != 21600:
        raise ValueError('Original 21600-second pressure budget required')
    carried = admission.get('carried_pressure_wait_seconds', 0.)
    if type(carried) not in (int, float) or not math.isfinite(carried) or not 0 <= carried <= budget:
        raise ValueError('Invalid previous carried pressure time')
    cumulative = 0.
    if 'ledger' in previous:
        ledger = previous['ledger']
        cumulative = ledger.get('cumulative_wait_seconds')
        if (ledger.get('trial_owner') != admission.get('owner')
                or ledger.get('selected_gpu_uuid') != admission.get('selected_gpu_uuid')
                or ledger.get('budget_seconds') != budget
                or type(cumulative) not in (int,float) or not math.isfinite(cumulative)
                or not carried <= cumulative <= budget):
            raise ValueError('Previous pressure ledger owner, GPU or cumulative budget differs')
    elif carried:
        raise ValueError('Missing previous cumulative pressure ledger')
    previous['carried_pressure_wait_seconds'] = float(cumulative)
    return previous


def carry_pressure(previous, admission, p):
    """Rebind the verified ledger to the new owner without resetting its budget."""
    cumulative = previous['carried_pressure_wait_seconds'] if previous else 0.
    admission['carried_pressure_wait_seconds'] = cumulative
    if previous:
        if admission['selected_gpu_uuid'] != previous['admission']['selected_gpu_uuid']:
            raise ValueError('Resume reservation must keep its verified GPU binding')
        clock = previous['run'].get('final_clock')
        if clock is not None:
            admission['previous_clock'] = clock
        if 'ledger' in previous:
            ledger = dict(previous['ledger'], trial_owner=admission['owner'],
                selected_gpu_uuid=admission['selected_gpu_uuid'], status='RESUMED_PRESSURE_BUDGET',
                cumulative_wait_seconds=cumulative, call_wait_seconds=0.,
                remaining_wait_seconds=21600-cumulative, carried_from_run_token=previous['run']['run_token'])
            write(Path(p['resource_protocol']['artifact_root'])/'running_resource_wait.json', ledger)


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        raise RuntimeError('Launch parent must be CPU-only')
    config = read(args.config); p = config['datasets'][args.dataset]
    rc = load_site_adapter(config)
    out = Path(p['output_dir']); control = out/'runtime'
    control.mkdir(parents=True, exist_ok=True)
    if args.resume is None and any(path.exists() for path in (
            out/'result.json', control/'run.json', control/'admission.json')):
        raise ValueError('Existing run requires an explicit --resume')
    previous = resume_evidence(out, p) if args.resume else None
    if previous:
        write(control/'previous_run.json', previous)
    state = dict(status='STARTING', run_token=uuid.uuid4().hex, parent_pid=os.getpid(),
                 actual_wait=None, CUDA_cleanup=None, own_lease_released=False, errors=[])
    child = pidfd = None
    write(control/'run.json', state)
    try:
        admission, handle = rc.admit(p['resource_protocol'])
        state.update(admission_owner=admission['owner'], selected_gpu_uuid=admission['selected_gpu_uuid'])
        carry_pressure(previous, admission, p)
        admission['run_token'] = state['run_token']
        write(control/'admission.json', admission)
        argv = [str(PYTHON), '-B', '-m', 'waft_repair.lean.train', '--dataset', args.dataset,
                '--config', str(args.config.resolve()), '--admission', str(control/'admission.json'),
                '--run-token', state['run_token']]
        if args.resume:
            argv += ['--resume', str(args.resume.resolve())]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=admission['selected_gpu_uuid'],
            PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MALLOC_ARENA_MAX='2', TMPDIR=str(control))
        state.update(status='RUNNING', argv=argv)
        write(control/'run.json', state)
        with (out/'train.log').open('a') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            write(control/'run.json', state)
            try:
                pidfd = os.pidfd_open(child.pid)
            except BaseException as exc:
                state['errors'].append(dict(stage='pidfd_open', type=type(exc).__name__, message=str(exc)))
            state['actual_wait'] = wait_child(child, pidfd, p['child_timeout_seconds'])
        waited = state['actual_wait']
        if waited['actual_wait_completed'] is not True:
            raise RuntimeError('Actual child wait unresolved; preserve own lease')
        result = read(out/'result.json')
        if result.get('run_token') != state['run_token']:
            raise ValueError('Result belongs to a different child invocation; preserve own lease')
        cleanup = result.get('cleanup')
        state.update(CUDA_cleanup=cleanup, result_status=result.get('status'),
                     final_clock=result.get('clock'), training_contract_digest=result.get('training_contract_digest'))
        if not clean_cleanup(cleanup):
            raise RuntimeError('CUDA cleanup unresolved; preserve own lease')
        rc.release_own_slot(handle, admission, p['resource_protocol'], child_waited=True, cleanup=cleanup)
        state['own_lease_released'] = True
        if waited['returncode'] != 0 or waited['interrupted'] or result.get('status') != 'COMPLETE':
            raise RuntimeError('Training failed or interrupted; see train.log and result.json')
    except BaseException as exc:
        state['errors'].append(dict(type=type(exc).__name__, message=str(exc)))
    finally:
        if pidfd is not None:
            try:
                os.close(pidfd)
            except BaseException as exc:
                state['errors'].append(dict(stage='pidfd_close', type=type(exc).__name__, message=str(exc)))
        unresolved = child is not None and not (state.get('actual_wait') or {}).get('actual_wait_completed')
        state['status'] = 'WAIT_UNRESOLVED' if unresolved else 'FAILED' if state['errors'] else 'COMPLETE'
        write(control/'run.json', state)
    if state['status'] != 'COMPLETE':
        raise RuntimeError('Training lifecycle failed; see runtime/run.json')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('kitti','sintel'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--managed-child', action='store_true', help=argparse.SUPPRESS)
    options = parser.parse_args()
    try:
        if options.managed_child:
            run(options)
        else:
            managed(options)
    except BaseException:
        traceback.print_exc()
        raise
