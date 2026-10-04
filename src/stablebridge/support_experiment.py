"""S04 locked train/evaluate worker, intended for an independent user service.

GT-derived Q never enters model kwargs or query sampling. Confirmation files
are opened only after every task/model final checkpoint has been committed.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from .support_data import (infer_support_case, save_support_cache, load_support_cache,
                           read_case_labels, offline_labels, save_label_cache, sparse_gt_support,
                           OPERATORS, REMATCH_SHIFTS_XY)
from .support_rematch import SupportConditionedRematcher
from .evaluation import error_map
from .util import ROOT, event, initialize_run, save_json, save_npz, sha256
from .robust_proxy import expand_profile_spec


def validate_config(config):
    if config['contract'] != 'CSB-S04-SUPPORT-REMATCH-v1-20260917':
        raise ValueError('Unexpected S04 contract')
    if config['operators'] != list(OPERATORS) or config['rematch_shifts_xy'] != [list(x) for x in REMATCH_SHIFTS_XY]:
        raise ValueError('Strong E0 must match the locked operations')
    seen = set()
    for split, scenes in config['splits'].items():
        overlap = seen.intersection(scenes)
        if overlap:
            raise ValueError(f'Scene leakage into {split}: {overlap}')
        seen.update(scenes)
    for split in ('calibration', 'confirmation'):
        if set(config['splits'][split]) & set(config['exposure']['prior_project_design_scenes']):
            raise ValueError('Fresh split contains an exposed scene')
    if config['learning']['seeds'] != [config['seed']]:
        raise ValueError('This worker supports exactly the registered one-seed screen')
    if config['learning']['models'] != ['no_support', 'actual', 'gt_diagnostic', 'propagation']:
        raise ValueError('All four trained controls are required')
    if config['support']['count'] != 120 or config['model']['nearest_supports'] != 4:
        raise ValueError('Unexpected support count')
    if config['search']['candidates_per_query'] != 30:
        raise ValueError('Unexpected search budget')


def cases_for(config, split, task=None):
    rows = []
    for scene, frames in config['splits'][split].items():
        for frame in frames:
            for profile in config['profiles']:
                for current_task in ('stereo', 'flow') if task is None else (task,):
                    identifier = f"{current_task}_{scene}_{frame:04d}_{profile['name']}"
                    rows.append({'case_id': identifier, 'task': current_task, 'scene': scene,
                                 'frame': frame, 'profile': profile, 'split': 'train', 'study_split': split})
    return rows


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def paths_for(directory, case):
    stem = directory / 'cache' / case['study_split'] / case['case_id']
    return stem.with_suffix('.npz'), stem.with_name(stem.name + '_labels.npz')


def load_case(directory, case):
    inference, labels = paths_for(directory, case)
    item = load_support_cache(inference)
    record = json.loads(labels.with_suffix('.json').read_text())
    if sha256(labels) != record['artifact']['sha256']:
        raise ValueError('Label hash mismatch')
    with np.load(labels, allow_pickle=False) as archive:
        label_arrays = {key: archive[key].copy() for key in archive.files}
    return item['arrays'], label_arrays, item['metadata']


def prepare_model_inputs(arrays, query_xy, arm, device, *, gt_support=None):
    """Closed whitelist. No GT/Q arrays accepted, except explicit sparse GT arm."""
    support = arrays['support_raw'] if arm == 'raw' else arrays['support_processed']
    valid = arrays['support_valid'].copy()
    # All predicted supports come from the same original pair. Keep provenance
    # identical across value interventions; do not leak the processing arm.
    source_id = 1
    if arm == 'wrong':
        support = np.roll(support, len(support)//2, axis=0)
        valid = np.roll(valid, len(valid)//2)
    elif arm == 'gt_diagnostic':
        if gt_support is None:
            raise ValueError('Sparse GT diagnostic requires explicit sparse labels')
        support, valid = gt_support
        source_id = 3
    elif arm == 'no_support':
        valid[:] = False
        support = np.zeros_like(support)
        source_id = 0
    elif arm not in ('raw', 'processed'):
        raise ValueError(f'Unknown support arm {arm}')
    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(value), dtype=dtype, device=device)[None]
    kwargs = {
        'source_features': tensor(arrays['source_features']), 'target_features': tensor(arrays['target_features']),
        'initial_uv': tensor(arrays['initial']), 'query_xy': tensor(query_xy),
        'supports_xy': tensor(arrays['support_xy']), 'supports_uv': tensor(support),
        'support_valid': tensor(valid, torch.bool),
        'support_source': tensor(np.full(len(support), source_id), torch.long),
        'source_rgb': tensor(arrays['image0'].transpose(2, 0, 1).astype(np.float32)/255),
        'target_rgb': tensor(arrays['image1'].transpose(2, 0, 1).astype(np.float32)/255),
        'support_mode': 'none' if arm == 'no_support' else 'provided',
    }
    return kwargs


def target_tensors(labels, query_xy, device):
    x, y = query_xy.astype(np.int64).T
    # NumPy advanced indexing produces [N,4,2] only after explicit transpose.
    values = labels['gt4'][:, :, y, x].transpose(2, 0, 1).copy()
    branch_valid = np.isfinite(values).all(axis=-1)
    valid = labels['gt_valid'][y, x]
    values = np.nan_to_num(values, nan=0., posinf=0., neginf=0.)
    return (torch.as_tensor(values, device=device)[None],
            torch.as_tensor(branch_valid, device=device)[None],
            torch.as_tensor(valid, device=device)[None])


def native_min4_loss(result, targets, branch_valid, query_valid, ce_weight):
    difference = result['output_uv'][:, :, None] - targets
    errors = torch.sqrt(difference.square().sum(-1) + 1e-6).masked_fill(~branch_valid, torch.inf).min(-1).values
    if not bool(query_valid.any()):
        raise ValueError('Uniform sampled batch contains no valid training labels')
    output_loss = errors[query_valid].mean()
    with torch.no_grad():
        candidate_difference = result['refined_candidate_uv'][:, :, :, None] - targets[:, :, None]
        candidate_error = torch.linalg.vector_norm(candidate_difference, dim=-1)
        candidate_error = candidate_error.masked_fill(~branch_valid[:, :, None], torch.inf).min(-1).values
        candidate_error = candidate_error.masked_fill(~result['candidate_valid'], torch.inf)
        best = candidate_error.argmin(-1)
    ce = F.cross_entropy(result['logits'][query_valid], best[query_valid])
    return output_loss + ce_weight*ce, {'error_loss': float(output_loss.detach()), 'candidate_ce': float(ce.detach())}


def _synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def build_cache(config, directory, split, device, deadline):
    from .backbones import CroCoAdapter
    from .data import SpringProvider
    registry = json.loads((ROOT / config['registry']).read_text())
    provider = SpringProvider.from_registry(ROOT / config['registry'])
    for task in ('stereo', 'flow'):
        model = registry['models']['main'][task]
        adapter = CroCoAdapter(task, model['local_path'], device=device,
                               context_hw=config['context_hw'], expected_sha256=model['sha256'])
        for index, case in enumerate(cases_for(config, split, task)):
            check_deadline(deadline)
            inference, label_path = paths_for(directory, case)
            if inference.exists() or label_path.exists():
                raise FileExistsError('No silent resume of partial or overwritten caches')
            result = infer_support_case(adapter, provider, case, seed=config['seed'],
                                         context_hw=tuple(config['context_hw']), origin_xy=tuple(config['origin_xy']))
            save_support_cache(inference, result)
            label = read_case_labels(provider, case, roi_xyhw=result['metadata']['pair']['roi_xyhw'],
                                     seed=config['seed'])
            a = result['arrays']
            labels = offline_labels(a['initial'], a['e0_bank'], a['e0_valid'], label['gt4'], a['support_xy'])
            save_label_cache(label_path, labels, label['metadata'])
            event(directory, 'cache_case_complete', split=split, case=case['case_id'], index=index,
                  Q=int(labels['fixed_q'].sum()), calls=result['metadata']['cost']['forward_calls'])
        del adapter
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()


def train_models(config, directory, device, deadline):
    learning = config['learning']
    for task in ('stereo', 'flow'):
        cases = cases_for(config, 'train', task)
        loaded = []
        for case in cases:
            arrays, labels, _ = load_case(directory, case)
            # Keep labels physically separate and drop scoring-only bank/Q before training.
            arrays = {key: value for key, value in arrays.items() if key not in ('e0_bank', 'e0_valid', 'raw_uncertainty', 'token_centers_xy', 'processed_support')}
            labels = {key: labels[key] for key in ('gt4', 'gt_valid')}
            loaded.append((arrays, labels))
        for model_name in learning['models']:
            seed_all(config['seed'])
            rng = np.random.default_rng(config['seed'])
            model = SupportConditionedRematcher(**config['model']).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning['learning_rate'], weight_decay=learning['weight_decay'])
            history, order = [], []
            _synchronize(device)
            start = time.perf_counter()
            if torch.device(device).type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for step in range(learning['steps_per_model']):
                check_deadline(deadline)
                if step % len(cases) == 0:
                    order = rng.permutation(len(cases))
                arrays, labels = loaded[int(order[step % len(cases)])]
                h, w = arrays['initial'].shape[-2:]
                pixel_ids = rng.integers(0, h*w, size=learning['queries_per_step'])
                queries = np.stack((pixel_ids % w, pixel_ids // w), axis=-1).astype(np.float32)
                arm = (model_name if model_name in ('no_support', 'gt_diagnostic') else
                       learning['actual_support_cycle'][step % len(learning['actual_support_cycle'])])
                gt_support = sparse_gt_support(labels['gt4'], arrays['support_xy']) if arm == 'gt_diagnostic' else None
                inputs = prepare_model_inputs(arrays, queries, arm, device, gt_support=gt_support)
                result = model(**inputs, task=task, rematch=model_name != 'propagation')
                loss, values = native_min4_loss(result, *target_tensors(labels, queries, device), learning['candidate_ce_weight'])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), learning['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                if step == 0 or (step+1) % 100 == 0 or step+1 == learning['steps_per_model']:
                    entry = {'step': step+1, 'arm': arm, 'loss': float(loss.detach()), 'gradient_norm': float(gradient), **values}
                    history.append(entry)
                    event(directory, 'train_progress', task=task, model=model_name, **entry)
                del inputs, result, loss
            _synchronize(device)
            checkpoint = directory / 'checkpoints' / f'{task}_{model_name}.pth'
            record = {'task': task, 'model_name': model_name, 'model_kwargs': config['model'],
                      'state_dict': model.state_dict(), 'steps': learning['steps_per_model'], 'seed': config['seed'],
                      'config_sha256': sha256(directory / 'config.json'), 'contract': model.contract()}
            torch.save(record, checkpoint)
            save_json(checkpoint.with_suffix('.json'), {'task': task, 'model': model_name, 'sha256': sha256(checkpoint),
                'steps': learning['steps_per_model'], 'history': history, 'training_wall_seconds': time.perf_counter()-start,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else 0,
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if torch.device(device).type == 'cuda' else 0,
                'contract': model.contract(), 'train_cases': [case['case_id'] for case in cases], 'GT_support': model_name == 'gt_diagnostic'})
            del model, optimizer
            gc.collect()
            if torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
        del loaded


ARM_MODELS = {'no_support': ('no_support', 'no_support', True), 'raw': ('actual', 'raw', True),
              'processed': ('actual', 'processed', True), 'wrong': ('actual', 'wrong', True),
              'gt_diagnostic': ('gt_diagnostic', 'gt_diagnostic', True),
              'propagation_raw': ('propagation', 'raw', False),
              'propagation_processed': ('propagation', 'processed', False),
              'propagation_wrong': ('propagation', 'wrong', False)}


def evaluate_split(config, directory, split, device, deadline):
    from .support_reporting import evaluate_support_case
    rows = []
    for task in ('stereo', 'flow'):
        models = {}
        for name in config['learning']['models']:
            path = directory / 'checkpoints' / f'{task}_{name}.pth'
            expected = json.loads(path.with_suffix('.json').read_text())['sha256']
            if sha256(path) != expected:
                raise ValueError('Final checkpoint changed before evaluation')
            state = torch.load(path, map_location='cpu', weights_only=True)
            model = SupportConditionedRematcher(**state['model_kwargs']).to(device).eval()
            model.load_state_dict(state['state_dict'], strict=True)
            models[name] = model
        for case in cases_for(config, split, task):
            check_deadline(deadline)
            arrays, labels, metadata = load_case(directory, case)
            h, w = arrays['initial'].shape[-2:]
            yy, xx = np.mgrid[:h, :w]
            queries = np.stack((xx, yy), axis=-1).reshape(-1, 2).astype(np.float32)
            outputs = {'identity': arrays['initial'], 'e0_processed_dense': arrays['processed_support']}
            oracles, ledger, coverage, support_coverage = {}, {}, {}, {}
            sparse_gt = sparse_gt_support(labels['gt4'], arrays['support_xy'])
            for arm, (model_name, support_arm, rematch) in ARM_MODELS.items():
                model = models[model_name]
                prediction = np.empty((h*w, 2), dtype=np.float32)
                oracle = np.empty(h*w, dtype=np.float32)
                computable = np.empty(h*w, dtype=np.float32)
                support_computable = np.empty(h*w, dtype=np.float32)
                inference_seconds, oracle_seconds = 0., 0.
                _synchronize(device)
                start = time.perf_counter()
                if torch.device(device).type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                with torch.no_grad():
                    for chunk in range(0, len(queries), config['model']['query_chunk_size']):
                        check_deadline(deadline)
                        _synchronize(device)
                        chunk_start = time.perf_counter()
                        q = queries[chunk:chunk+config['model']['query_chunk_size']]
                        inputs = prepare_model_inputs(arrays, q, support_arm, device,
                                                      gt_support=sparse_gt if support_arm == 'gt_diagnostic' else None)
                        result = model(**inputs, task=task, rematch=rematch)
                        prediction[chunk:chunk+len(q)] = result['output_uv'][0].cpu().numpy()
                        _synchronize(device)
                        inference_seconds += time.perf_counter()-chunk_start
                        diagnostic_start = time.perf_counter()
                        gt, branches, _ = target_tensors(labels, q, device)
                        difference = result['refined_candidate_uv'][:, :, :, None] - gt[:, :, None]
                        errors = torch.linalg.vector_norm(difference, dim=-1)
                        errors = errors.masked_fill(~branches[:, :, None], torch.inf).min(-1).values
                        errors = errors.masked_fill(~result['candidate_valid'], torch.inf).min(-1).values
                        oracle[chunk:chunk+len(q)] = errors[0].cpu().numpy()
                        computable[chunk:chunk+len(q)] = result['candidate_valid'][0, :, 1:].float().mean(-1).cpu().numpy()
                        support_computable[chunk:chunk+len(q)] = (result['candidate_valid'][0, :, 10:] &
                            result['candidate_has_support'][0, :, 10:]).float().mean(-1).cpu().numpy()
                        _synchronize(device)
                        oracle_seconds += time.perf_counter()-diagnostic_start
                _synchronize(device)
                outputs[arm] = prediction.T.reshape(2, h, w)
                oracles[arm] = oracle.reshape(h, w)
                coverage[arm] = computable.reshape(h, w)
                support_coverage[arm] = support_computable.reshape(h, w)
                base_calls = 12 if support_arm in ('processed', 'wrong') else 1
                base_seconds = sum(item['charged_seconds'] for item in metadata['ledger'][:base_calls])
                seconds = time.perf_counter()-start
                ledger[arm] = {'repair_wall_seconds': inference_seconds, 'evaluation_wall_seconds': seconds,
                               'offline_oracle_seconds': oracle_seconds, 'standalone_backbone_calls': base_calls,
                               'standalone_backbone_seconds': base_seconds,
                               'support_processing_seconds': metadata['cost']['support_processing_seconds'] if base_calls == 12 else 0.,
                               'candidate_slots': len(queries)*model.candidate_count,
                               'target_candidate_reads': len(queries)*model.candidate_count if rematch else 0,
                               'target_RGB_sample_locations': len(queries)*model.candidate_count*9 if rematch else 0,
                               'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else 0,
                               'privileged_GT': support_arm == 'gt_diagnostic', 'observation_wait_frames': 0,
                               'repair_wall_includes_offline_oracle_scoring': False,
                               'peak_memory_includes_offline_oracle_scoring': True}
            meta = {'task': task, 'scene': case['scene'], 'frame': case['frame'], 'profile': case['profile']['name'],
                    'split': split, 'case_id': case['case_id']}
            row = evaluate_support_case(arrays['initial'], outputs, labels['gt4'], labels['fixed_q'],
                                        oracle_errors=oracles, metadata=meta)
            # Support quality is diagnosed only AFTER inference. It is not a selector input.
            sx, sy = arrays['support_xy'].astype(int).T
            row['support_quality'] = {}
            for name, field in [('raw', arrays['initial']), ('processed', arrays['processed_support'])]:
                errors = error_map(field, labels['gt4'])[sy, sx]
                valid = labels['gt_valid'][sy, sx]
                row['support_quality'][name] = {'valid_count': int(valid.sum()),
                    'mean_error_px': float(errors[valid].mean()) if valid.any() else None,
                    'within_1px_pct': float(100*(errors[valid] <= 1).mean()) if valid.any() else None}
            row['cost'] = ledger
            row['physical_cache_cost'] = metadata['cost']
            row['coverage'] = {arm: {'mean_nonidentity_candidate_fraction': float(value.mean()),
                                    'Q_mean_nonidentity_candidate_fraction': float(value[labels['fixed_q']].mean()) if labels['fixed_q'].any() else None,
                                    'mean_support_candidate_fraction': float(support_coverage[arm].mean()),
                                    'Q_mean_support_candidate_fraction': float(support_coverage[arm][labels['fixed_q']].mean()) if labels['fixed_q'].any() else None}
                               for arm, value in coverage.items()}
            common = labels['fixed_q'] & (coverage['processed'] > 0) & (coverage['wrong'] > 0)
            row['common_computable_PW'] = {'pixels': int(common.sum()), 'criterion': 'at least one nonidentity eligible candidate each; not visibility',
                'gain_px': float((error_map(outputs['wrong'], labels['gt4'])-error_map(outputs['processed'], labels['gt4']))[common].mean()) if common.any() else None}
            common_support = labels['fixed_q'] & (support_coverage['processed'] > 0) & (support_coverage['wrong'] > 0)
            row['common_support_computable_PW'] = {'pixels': int(common_support.sum()),
                'criterion': 'at least one actual support-derived eligible slot10:30 each; no general-search fallback or visibility claim',
                'gain_px': float((error_map(outputs['wrong'], labels['gt4'])-error_map(outputs['processed'], labels['gt4']))[common_support].mean()) if common_support.any() else None}
            row['outside_support_guard'] = evaluate_support_case(arrays['initial'], outputs, labels['gt4'],
                                        labels['fixed_q_outside_support_guard'], metadata=meta)['contrasts']
            save_json(directory / 'metrics' / split / f"{case['case_id']}.json", row)
            save_npz(directory / 'predictions' / split / f"{case['case_id']}.npz",
                     **{f'output_{k}': v for k, v in outputs.items()},
                     **{f'oracle_error_{k}': v for k, v in oracles.items()},
                     **{f'coverage_{k}': v for k, v in coverage.items()},
                     **{f'support_coverage_{k}': v for k, v in support_coverage.items()}, fixed_q=labels['fixed_q'])
            rows.append(row)
            event(directory, 'evaluation_case_complete', split=split, case=case['case_id'])
        del models
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
    save_json(directory / 'metrics' / f'{split}_rows.json', rows)
    return rows


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('Predeclared worker time budget reached; no automatic continuation')


def verify_frozen(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name, expected in manifest['source_hashes'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Source changed during locked run: {name}')
    for row in manifest['registered_inputs']:
        if sha256(Path(row['path'])) != row['sha256']:
            raise ValueError(f'Contract/config changed during locked run: {row["path"]}')


def run(config_path, directory, device, smoke=False):
    from .support_reporting import aggregate_support_results, render_report
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    config['profiles'] = expand_profile_spec(config['profiles'])
    validate_config(config)
    if smoke:
        config = copy.deepcopy(config)
        config['learning']['steps_per_model'] = 5
        first = next(iter(config['splits']['train'].items()))
        config['splits'] = {'train': {first[0]: [first[1][0]]}}
        config['profiles'] = [{'name': 'clean'}]
        config['scientific_scope'] = 'engineering_smoke_only_same_train_case_evaluation'
    if (directory / 'manifest.json').exists():
        raise FileExistsError('Use a new run directory; this worker does not silently resume')
    if shutil.disk_usage(directory.parent).free < config['limits']['minimum_free_disk_gb']*10**9:
        raise RuntimeError('Insufficient free disk under the registered limit')
    registered = [config_path, ROOT / config['spec'], ROOT / config['registry']]
    initialize_run(directory, {'study': config['study'], 'config': config, 'smoke': smoke,
                               'registered_inputs': [{'path': str(p), 'sha256': sha256(p)} for p in registered],
                               'device': device, 'training_detached_expected': True})
    save_json(directory / 'config.json', config)
    shutil.copyfile(ROOT / config['spec'], directory / 'contract.md')
    torch.set_num_threads(config['cpu_threads'])
    if torch.device(device).type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no silent CPU fallback for scientific job')
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'], device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    seed_all(config['seed'])
    deadline = time.monotonic()+config['limits']['maximum_run_hours']*3600
    start = time.perf_counter()
    stage = 'initializing'
    def status(value):
        nonlocal stage
        stage = value
        verify_frozen(directory)
        save_json(directory / 'status.json', {'state': 'running', 'stage': stage, 'pid': os.getpid(),
                                             'smoke': smoke, 'updated_unix': time.time()})
    try:
        status('train_cache')
        build_cache(config, directory, 'train', device, deadline)
        status('train_four_models_per_task')
        train_models(config, directory, device, deadline)
        rows = []
        if smoke:
            status('engineering_dense_evaluation')
            rows.extend(evaluate_split(config, directory, 'train', device, deadline))
        else:
            for split in ('development', 'calibration', 'confirmation'):
                status(f'{split}_cache')
                build_cache(config, directory, split, device, deadline)
                status(f'{split}_dense_evaluation')
                rows.extend(evaluate_split(config, directory, split, device, deadline))
        status('reporting')
        report = aggregate_support_results(rows, config)
        report['scope'] = config['scientific_scope']
        report['smoke'] = smoke
        report['total_wall_seconds'] = time.perf_counter()-start
        report['model_contract'] = SupportConditionedRematcher(**config['model']).contract()
        save_json(directory / 'report.json', report)
        (directory / 'report.md').write_text(render_report(report))
        save_json(directory / 'status.json', {'state': 'completed', 'stage': 'completed', 'smoke': smoke,
                                             'report': str(directory / 'report.json'), 'updated_unix': time.time()})
        event(directory, 'completed', smoke=smoke, wall_seconds=report['total_wall_seconds'])
    except Exception as error:
        save_json(directory / 'status.json', {'state': 'failed', 'stage': stage, 'error': repr(error),
                    'traceback': traceback.format_exc(), 'smoke': smoke, 'updated_unix': time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:6')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args.config, args.output, args.device, args.smoke)
    except Exception as error:
        # Startup validation/CUDA errors also need a visible service artifact.
        if not (args.output / 'status.json').exists():
            save_json(args.output / 'status.json', {'state': 'failed', 'stage': 'startup',
                'error': repr(error), 'traceback': traceback.format_exc(), 'smoke': args.smoke,
                'updated_unix': time.time()})
        raise


if __name__ == '__main__':
    main()
