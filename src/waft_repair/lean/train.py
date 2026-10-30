"""Compact trainer: scalar epoch CSV, one resumable checkpoint, final weights."""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from .common import HERE, ROOT, read, write, save_torch, load_torch, load_site_adapter
from .lite_repair import LiteFeatureRepair, LiteRepairConfig
from . import data, runtime, loader, clock as timing, memory


def rate(step, p):
    warm, total, peak = p['warmup_updates'], p['total_updates'], p['peak_lr']
    if step <= warm:
        return peak*step/warm
    x = min(max((step-warm)/(total-warm), 0.), 1.)
    return peak*(p['minimum_lr_fraction']+(1-p['minimum_lr_fraction'])*.5*(1+math.cos(math.pi*x)))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def rng_state():
    n = np.random.get_state()
    return dict(torch=torch.get_rng_state(), CUDA=[x.cpu() for x in torch.cuda.get_rng_state_all()],
        python=random.getstate(), numpy=dict(kind=n[0], keys=n[1].tolist(), position=int(n[2]),
        has_gauss=int(n[3]), cached_gaussian=float(n[4])))


def restore_rng(rng):
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state_all(rng['CUDA'])
    random.setstate(rng['python'])
    n = rng['numpy']
    np.random.set_state((n['kind'], np.asarray(n['keys'], np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))


def validate_rng(rng, *, official=False):
    """Validate CPU-restorable RNG structure without changing any global RNG."""
    if not isinstance(rng, dict) or set(rng) != {'torch','CUDA','python','numpy'}:
        raise ValueError('Incomplete saved RNG state')
    def byte_state(value):
        return (isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0)
    if (not byte_state(rng['torch']) or not isinstance(rng['CUDA'], list)
            or (official and len(rng['CUDA']) != 1)
            or any(not byte_state(value) for value in rng['CUDA'])):
        raise ValueError('Invalid saved Torch/CUDA RNG structure')
    try:
        torch.Generator(device='cpu').set_state(rng['torch'])
        random.Random().setstate(rng['python'])
        n = rng['numpy']
        if (n['kind'] != 'MT19937' or len(n['keys']) != 624
                or any(type(value) is not int or not 0 <= value < 2**32 for value in n['keys'])):
            raise ValueError('Invalid NumPy RNG keys')
        np.random.RandomState().set_state((n['kind'], np.asarray(n['keys'], np.uint32),
            n['position'], n['has_gauss'], n['cached_gaussian']))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError('Invalid saved RNG state') from exc


def schedule(p):
    return {key: p[key] for key in ('epochs','total_updates','warmup_updates','peak_lr','minimum_lr_fraction','shuffle_seed')}


def training_contract(system, p):
    """Bind data and execution once at startup, never in the step loop."""
    sources = [HERE/name for name in ('train.py','runtime.py','lite_repair.py','data.py','common.py')]
    package = HERE.parent
    sources += [package/'motion'/name for name in ('consumer.py','native.py','geometry.py','finite.py')]
    sources += [HERE/name for name in ('activation.py','activation_async.py','loader.py','clock.py','memory.py')]
    return dict(schema='waft-liteR-training-contract/v1', schedule=schedule(p),
        core_digest=p['expected_core_digest'], cases=p['cases'],
        model_config=system.restorer.configuration(),
        resources={key: p.get('resource_protocol', {}).get(key) for key in
                   ('resource_profile','resources','runtime','allowed_gpu_uuids')},
        clock_budgets=p.get('clock_budgets'),
        implementation={str(path.relative_to(HERE.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sources})


def contract_digest(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def legacy_protocol(state, p):
    from .common import verify
    ref = p.get('resume_protocol')
    if not ref or state.get('protocol_sha256') != ref['sha256']:
        raise ValueError('Unbound legacy checkpoint; explicit verified protocol required')
    old = read(verify(ref))
    previous = old['datasets'][p['dataset']]
    if schedule(previous) != schedule(p) or previous['cases'] != p['cases']:
        raise ValueError('Legacy resume schedule/data contract differs')
    return old


def restore(path, system, optimizer, p, *, state=None, restore_random=True):
    state = load_torch(path) if state is None else state
    if state['core_digest'] != p['expected_core_digest'] or state['fixed_cases'] != [r['case_id'] for r in p['cases']]:
        raise ValueError('Resume backbone or data population differs')
    if state.get('schedule') != schedule(p):
        if state.get('training_contract') is not None:
            raise ValueError('Cannot change the resume schedule')
        # Old sanity checkpoints bind their complete immutable schedule through protocol SHA.
        legacy_protocol(state, p)
    expected_contract = training_contract(system, p)
    if state.get('training_contract') is None:
        old = legacy_protocol(state, p)
        if old['architecture'] != vars(system.restorer.config):
            raise ValueError('Legacy repair configuration differs')
    elif (state['training_contract'] != expected_contract or
          state.get('training_contract_digest') != contract_digest(expected_contract)):
        raise ValueError('Resume training contract differs')
    if state.get('model_config') != system.restorer.configuration():
        raise ValueError('Resume model configuration differs')
    completed = state['completed_updates']
    if type(completed) is not int or completed < 0 or completed > p['total_updates'] or completed % len(p['cases']):
        raise ValueError('Only complete epoch checkpoints are resumable')
    saved_r = state['repair_state']
    current_r = system.restorer.state_dict()
    if saved_r.keys() != current_r.keys():
        raise ValueError('Incomplete saved R state')
    for key, value in saved_r.items():
        if value.shape != current_r[key].shape or value.dtype != current_r[key].dtype:
            raise ValueError('Saved R shape/dtype differs')
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite saved R')
    saved_opt = state['optimizer_state']
    if len(saved_opt['param_groups']) != 1:
        raise ValueError('One AdamW group required')
    group = saved_opt['param_groups'][0]
    expected_group = optimizer.state_dict()['param_groups'][0]
    if (group.keys() != expected_group.keys() or
            any(group[key] != value for key, value in expected_group.items() if key not in ('params','lr'))):
        raise ValueError('Saved AdamW settings differ')
    if group['params'] != list(range(len(saved_r))) or set(saved_opt['state']) != (set(group['params']) if completed else set()):
        raise ValueError('Incomplete saved AdamW moments')
    for index, item in saved_opt['state'].items():
        shape = list(saved_r.values())[index].shape
        if (set(item) != {'step','exp_avg','exp_avg_sq'} or
                item['exp_avg'].shape != shape or item['exp_avg_sq'].shape != shape or
                item['exp_avg'].dtype != list(saved_r.values())[index].dtype or
                item['exp_avg_sq'].dtype != list(saved_r.values())[index].dtype):
            raise ValueError('Saved AdamW moment shape differs')
        if float(item['step']) != completed or any(not bool(torch.isfinite(v).all()) for v in item.values()):
            raise ValueError('Invalid saved AdamW state/cursor')
    validate_rng(state.get('rng'), official=system.strict_official)
    system.restorer.load_state_dict(state['repair_state'], strict=True)
    optimizer.load_state_dict(state['optimizer_state'])
    if restore_random:
        restore_rng(state['rng'])
    system._restored_initial_epe = state.get('initial_EPE')
    system._restored_clock = state.get('clock')
    system._training_contract = expected_contract
    return completed


def save_latest(out, system, optimizer, p, completed, epoch, initial_epe=None, clock=None):
    binding = getattr(system, '_training_contract', None)
    if binding is None:
        binding = system._training_contract = training_contract(system, p)
    repair_state = cpu_tree(system.restorer.state_dict())
    optimizer_state = cpu_tree(optimizer.state_dict())
    rng = rng_state()
    save_torch(out/'latest.pt', dict(schema='waft-liteR-lean-training/v2',
        repair_state=repair_state, optimizer_state=optimizer_state,
        rng=rng, completed_updates=completed, epoch=epoch, schedule=schedule(p),
        fixed_cases=[r['case_id'] for r in p['cases']], core_digest=p['expected_core_digest'],
        model_config=system.restorer.configuration(), initial_EPE=initial_epe,
        training_contract=binding, training_contract_digest=contract_digest(binding),
        clock=clock() if callable(clock) else clock))


def append_metrics(path, row):
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('epoch','updates','mean_loss','mean_EPE','native_MoL','lr','seconds'))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.config = read(args.config)
        self.p = dict(self.config['datasets'][args.dataset], dataset=args.dataset)
        self.out = Path(self.p['output_dir'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.system = self.optimizer = self.core = self.inputs = None
        self.completed = 0
        self.last_check = 0.
        self.rc = load_site_adapter(self.config)
        self.memory = memory
        self.timing = timing
        budget = self.timing.RuntimeBudgets(**self.p['clock_budgets'])
        budget.validate_child_timeout(self.p['child_timeout_seconds'])
        self.clock = self.timing.ActiveClock(budget)
        self.admission = read(args.admission)
        self.initial = self.p.get('initial_epe_by_case', {})

    def resume_clock(self, state):
        saved = state.get('clock')
        if saved is None:
            from .common import verify
            legacy_protocol(state, self.p)
            if not self.p.get('resume_clock'):
                raise ValueError('Legacy resume requires its existing cumulative clock')
            saved = read(verify(self.p['resume_clock']))['clock']
        current = self.clock.snapshot()
        if saved['budgets'] != current['budgets']:
            raise ValueError('Resume clock budgets differ')
        restored = self.timing.ActiveClock.from_snapshot(saved)
        prior = self.admission.get('previous_clock')
        if prior is not None:
            if prior['budgets'] != current['budgets']:
                raise ValueError('Previous invocation clock budgets differ')
            validated = self.timing.ActiveClock.from_snapshot(prior).snapshot()
            for phase in restored.PHASES:
                restored.elapsed[phase] = max(restored.elapsed[phase], validated['elapsed_seconds'][phase])
        for phase, elapsed in current['elapsed_seconds'].items():
            restored.elapsed[phase] += elapsed
        restored.elapsed['pressure_wait'] = max(restored.elapsed['pressure_wait'],
            self.admission.get('carried_pressure_wait_seconds', 0.))
        restored.check()
        self.clock = restored

    def check_resources(self, force=False):
        if force or time.monotonic()-self.last_check >= self.p['resource_check_seconds']:
            self.clock.check()
            def sleep(seconds):
                with self.clock.measure('pressure_wait'):
                    time.sleep(seconds)
            self.rc.check(self.admission, self.p['resource_protocol'], _sleep=sleep)
            self.last_check = time.monotonic()

    def boundary(self, row):
        self.optimizer.zero_grad(set_to_none=True)
        if not (self.inputs is not None and self.inputs.cached):
            for key in ('RGB_first','RGB_second','raw_gt','gt'):
                if key in row:
                    self.memory.advise_readonly(row[key]['path'])
        self.memory.release_unused(torch)

    def load_case(self, row):
        inputs = getattr(self, 'inputs', None)
        return inputs.get(row) if inputs is not None else data.load(row)

    def evaluate(self):
        values = {}
        for row in self.p['cases']:
            self.check_resources()
            pair, gt, valid = self.load_case(row)
            with self.clock.measure('active'):
                values[row['case_id']] = runtime.evaluate(self.system, pair, gt, valid,
                    compute_mol=getattr(self, 'config', {}).get('evaluation_native_mol', False))
            del pair, gt, valid
            self.boundary(row)
        return values

    def setup(self):
        p = self.p
        resume_state = load_torch(self.args.resume) if self.args.resume else None
        if resume_state is not None:
            self.resume_clock(resume_state)
        if p['total_updates'] != p['epochs']*len(p['cases']) or not 0 < p['warmup_updates'] < p['total_updates']:
            raise ValueError('Explicit complete schedule required')
        if not self.args.resume and ((self.out/'latest.pt').exists() or (self.out/'metrics.csv').exists()):
            raise ValueError('Existing run requires explicit --resume; do not overwrite it')
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU-only official model loading required')
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.manual_seed(self.config['seed']); random.seed(self.config['seed']); np.random.seed(self.config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.inputs = data.RawCaseCache(p['cases'], self.config.get('raw_cache_max_bytes', 128 << 20))
        self.inputs.release_files(self.memory)
        item = p['backbone_checkpoint']
        self.core, receipt = loader.load_model(p['WAFT_source'], dict(checkpoint=item['path'], sha256=item['sha256']))
        if receipt['all_saved_tensor_count'] != 508 or receipt['all_loaded_tensor_bytes_digest'] != p['expected_core_digest']:
            raise ValueError('Wrong official backbone')
        repair = LiteFeatureRepair(LiteRepairConfig(**self.config['architecture']))
        # Complete source/state binding on CPU; do not copy the frozen core
        # back from CUDA just to recompute its startup digest.
        self.system = runtime.make_system(self.core, repair, expected_core_digest=p['expected_core_digest'],
            **self.core._lean_geometry)
        self.system._training_contract = training_contract(self.system, p)
        self.contract = self.system._training_contract
        self.optimizer = runtime.make_optimizer(self.system, learning_rate=p['peak_lr'])
        if resume_state is not None:
            self.completed = restore(self.args.resume, self.system, self.optimizer, p,
                                     state=resume_state, restore_random=False)
            self.initial = self.system._restored_initial_epe or self.initial
        with self.clock.measure('before_context_wait'):
            self.rc.wait_before_context(p['resource_protocol'], self.admission, existing_model_loads=1)
        self.rc.setup_cuda(p['resource_protocol'], self.admission)
        self.system.cuda()
        # Recreate the still-empty optimizer on the final device; restoring
        # AdamW there preserves the original CUDA optimizer-state placement.
        self.optimizer = runtime.make_optimizer(self.system, learning_rate=p['peak_lr'])
        if resume_state is not None:
            self.optimizer.load_state_dict(resume_state['optimizer_state'])
            restore_rng(resume_state['rng'])
        del resume_state
        write(self.out/'config.json', dict(self.config, selected_dataset=self.args.dataset))
        self.last_check = 0.

    def run(self):
        self.setup()
        p = self.p; final_eval = None
        for epoch in range(self.completed//len(p['cases'])+1, p['epochs']+1):
            started = time.monotonic()
            order = list(p['cases']); random.Random(p['shuffle_seed']+epoch).shuffle(order)
            losses = []
            for row in order:
                self.check_resources()
                pair, gt, valid = self.load_case(row)
                with self.clock.measure('active'):
                    metrics = runtime.train_step(self.system, self.optimizer, pair, gt, valid,
                        learning_rate=rate(self.completed+1, p))
                self.completed += 1
                losses.append(metrics['loss'])
                del pair, gt, valid, metrics
                self.boundary(row)
            self.check_resources(force=True)
            final_eval = self.evaluate() if epoch == p['epochs'] else None
            save_latest(self.out, self.system, self.optimizer, p, self.completed, epoch,
                        self.initial, self.clock.check)
            means = dict(mean_EPE='', native_MoL='')
            if final_eval is not None:
                means['mean_EPE'] = sum(x['EPE'] for x in final_eval.values())/len(final_eval)
                if all('native_MoL' in x for x in final_eval.values()):
                    means['native_MoL'] = sum(x['native_MoL'] for x in final_eval.values())/len(final_eval)
            record = dict(epoch=epoch, updates=self.completed, mean_loss=sum(losses)/len(losses),
                lr=rate(self.completed, p), seconds=time.monotonic()-started, **means)
            append_metrics(self.out/'metrics.csv', record)
            write(self.out/'progress.json', dict(status='TRAINING', **record))
            print('epoch=%d updates=%d loss=%.6f EPE=%s' % (epoch,self.completed,record['mean_loss'],record['mean_EPE']), flush=True)
        if final_eval is None:
            final_eval = self.evaluate()
        # Final weights are small and bound to the selected official domain profile.
        save_torch(self.out/'final.pt', dict(repair_state=cpu_tree(self.system.restorer.state_dict()),
            model_config=self.system.restorer.configuration(), core_digest=p['expected_core_digest'],
            backbone_checkpoint=p['backbone_checkpoint'], completed_updates=self.completed))
        return dict(completed_updates=self.completed, initial_EPE=self.initial,
                    final_EPE={key: value['EPE'] for key,value in final_eval.items()})

    def cleanup(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        self.optimizer = None
        if self.system is not None:
            self.system.cpu()
        elif self.core is not None:
            self.core.cpu()
        self.system = self.core = self.inputs = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
            self.memory.release_unused(torch)
            return dict(references_cleared=True, allocated_bytes=torch.cuda.memory_allocated(),
                        reserved_bytes=torch.cuda.memory_reserved())
        return dict(references_cleared=True, allocated_bytes=0, reserved_bytes=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('kitti','sintel'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--run-token')
    args = parser.parse_args()
    trainer = None; result = {}; error = None
    try:
        trainer = Trainer(args)
        result = trainer.run()
    except BaseException as exc:
        traceback.print_exc()
        error = dict(type=type(exc).__name__, message=str(exc))
        traceback.clear_frames(exc.__traceback__)
    finally:
        try:
            cleanup = trainer.cleanup() if trainer is not None else dict(
                references_cleared=True, allocated_bytes=0, reserved_bytes=0)
        except BaseException as exc:
            cleanup = dict(references_cleared=False, allocated_bytes=None, reserved_bytes=None)
            error = error or dict(type='CleanupFailure', message=str(exc))
        if not (cleanup['references_cleared'] and cleanup['allocated_bytes'] == cleanup['reserved_bytes'] == 0):
            error = error or dict(type='CleanupFailure', message='CUDA cleanup did not reach zero')
        out = trainer.out if trainer is not None else Path(read(args.config)['datasets'][args.dataset]['output_dir'])
        result.setdefault('completed_updates', trainer.completed if trainer is not None else 0)
        result['run_token'] = args.run_token
        if trainer is not None:
            result['clock'] = trainer.clock.snapshot()
            binding = getattr(trainer.system, '_training_contract', None)
            # cleanup has released the model; the setup contract is small CPU metadata.
            binding = binding or getattr(trainer, 'contract', None)
            if binding is not None:
                result['training_contract_digest'] = contract_digest(binding)
        write(out/'result.json', dict(status='COMPLETE' if error is None else 'FAILED', error=error,
            cleanup=cleanup, **result))
    if error or cleanup['allocated_bytes'] != 0 or cleanup['reserved_bytes'] != 0:
        raise RuntimeError('Training failed; see result.json')


if __name__ == '__main__':
    main()
