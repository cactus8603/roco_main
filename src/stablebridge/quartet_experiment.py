"""S02-v2 fixed-query capacity experiment. Inference never reads ground truth.

Two frozen checkpoints share a native-pixel field/composer interface. Models
are loaded sequentially; all network predictions are committed before labels
are opened. Composition is a pure function of those predictions and the locked
config. This is an offline capacity diagnostic, not an update controller.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import time

import h5py
import numpy as np
import torch

from .backbones import CroCoAdapter
from .data import SpringProvider, apply_corruption
from .robust_proxy import (apply_observation as apply_robust20_observation,
                           expand_profile_spec, parse_profile)
from .geometry import in_bounds
from .path_geometry import DenseEdge, compose_path, project_stereo_path
from .pipeline import local_operator, _orient_reverse_displacement
from .util import ROOT, event, initialize_run, save_json, save_npz, sha256


# Each reverse edge is inferred from its own image pair, never negated in place.
EDGES = {
    'D0': ('stereo', 'L0', 'R0', False),
    'D1': ('stereo', 'L1', 'R1', False),
    'D1reverse': ('stereo', 'R1', 'L1', True),
    'FL': ('flow', 'L0', 'L1', False),
    'FR': ('flow', 'R0', 'R1', False),
    'FRbw': ('flow', 'R1', 'R0', False),
}
PATHS = {'stereo': ('FL', 'D1', 'FRbw'), 'flow': ('D0', 'FR', 'D1reverse')}
DIRECT = {'stereo': 'D0', 'flow': 'FL'}


def save_archive(path, config, **arrays):
    """Atomic diagnostic storage; compression is excluded from method cost."""
    if config.get('artifact_compression', 'deflate') == 'deflate':
        save_npz(path, **arrays)
        return
    if config['artifact_compression'] != 'none':
        raise ValueError('artifact_compression must be none or deflate')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def restore_translation(field, shift_xy):
    """Undo a joint image roll; reject wrapped source and endpoint support."""
    dx, dy = map(int, shift_xy)
    restored = np.roll(np.asarray(field), (-dy, -dx), axis=(1, 2)).copy()
    h, w = restored.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    q = np.stack((xx, yy), axis=-1)
    endpoint = q + restored.transpose(1, 2, 0)
    valid = (in_bounds(q + (dx, dy), (h, w)) & in_bounds(endpoint, (h, w))
             & in_bounds(endpoint + (dx, dy), (h, w)))
    return restored, valid


def load_quartet(provider, case, config):
    start = time.perf_counter()
    images, records = {}, {}
    h, w = config['context_hw']
    x, y = config['origin_xy']
    profile = case['profile']
    for node, offset, view in [('L0', 0, 'left'), ('R0', 0, 'right'),
                                ('L1', 1, 'left'), ('R1', 1, 'right')]:
        frame = case['frame'] + offset
        raw = provider.read_image(case['scene'], frame, view)
        provider._validate_roi((x, y, h, w), raw.shape[:2])
        if parse_profile(profile['name']) is not None:
            if profile.get('options'):
                raise ValueError('Robust20 profiles do not accept ad-hoc options')
            corrupted, record = apply_robust20_observation(
                raw, profile=profile['name'], seed=config['seed'], scene=case['scene'],
                frame=frame, view=view)
        else:
            corrupted, record = apply_corruption(
                raw, profile['name'], seed=config['seed'], scene=case['scene'],
                frame=frame, view=view, options=profile.get('options'))
        path = provider.image_path(case['scene'], frame, view)
        images[node] = np.ascontiguousarray(corrupted[y:y+h, x:x+w])
        records[node] = {**record, 'file': str(path), 'file_sha256': sha256(path),
                         'frame': frame, 'view': view, 'origin_xy': [x, y]}
    return images, records, time.perf_counter() - start


def infer_task(adapter, images, config, task):
    """Return current-pair and quartet edge candidates with a measured ledger."""
    arrays, ledger = {}, []
    device = adapter.device
    for edge_id, (edge_task, source, target, flipped) in EDGES.items():
        if edge_task != task:
            continue
        for op_index, op in enumerate(config['operators']):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            im0, im1 = images[source], images[target]
            if op != 'identity':
                im0, im1 = local_operator(im0, op), local_operator(im1, op)
            if flipped:
                im0, im1 = np.ascontiguousarray(im0[:, ::-1]), np.ascontiguousarray(im1[:, ::-1])
            prediction = adapter.predict(im0, im1, config['origin_xy'])
            field, uncertainty = prediction.displacement, prediction.raw_uncertainty
            if flipped:
                field = _orient_reverse_displacement('stereo', field)
                uncertainty = uncertainty[:, ::-1].copy()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            key = f'{edge_id}_{op_index}'
            arrays[key] = field
            arrays[f'{key}_uncertainty'] = uncertainty
            ledger.append({'id': key, 'kind': 'edge', 'edge': edge_id, 'operator': op,
                           'source': source, 'target': target, 'task': task,
                           'charged_seconds': elapsed, 'forward_calls': 1,
                           'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                           'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                           'prediction': prediction.metadata})
            del prediction
    _, source, target, _ = EDGES[DIRECT[task]]
    for index, (dx, dy) in enumerate(config['rematch_shifts_xy']):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        im0 = np.roll(images[source], (dy, dx), axis=(0, 1))
        im1 = np.roll(images[target], (dy, dx), axis=(0, 1))
        prediction = adapter.predict(im0, im1, config['origin_xy'])
        field, valid = restore_translation(prediction.displacement, (dx, dy))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        key = f'rematch_{index}'
        arrays[key], arrays[f'{key}_valid'] = field, valid
        ledger.append({'id': key, 'kind': 'rematch', 'task': task, 'shift_xy': [dx, dy],
                       'charged_seconds': elapsed, 'forward_calls': 1,
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                       'prediction': prediction.metadata})
        del prediction
    return arrays, ledger


def build_paths(fields, task, config, wrong=False):
    h, w = config['context_hw']
    origin = tuple(config['origin_xy'])
    yy, xx = np.mgrid[:h, :w]
    queries = np.stack((xx, yy), axis=-1).reshape(-1, 2) + origin
    results = []
    start = time.perf_counter()
    for combo in config['path_operator_tuples']:
        chain = []
        for edge_index, (edge_id, op_index) in enumerate(zip(PATHS[task], combo)):
            _, source, target, _ = EDGES[edge_id]
            field = fields[f'{edge_id}_{op_index}']
            if wrong and edge_index == 1:
                dx, dy = config['wrong_middle_shift_xy']
                field = np.roll(field, (dy, dx), axis=(1, 2))
            chain.append(DenseEdge(field, source, target, origin, origin, (h, w)))
        result = compose_path(queries, chain)
        if task == 'stereo':
            result.update(project_stereo_path(result['raw_displacement'], result['computable'],
                                               config['epipolar_tolerance_px']))
        else:
            result['candidate'] = result['raw_displacement'].copy()
            result['eligible'] = result['computable'].copy()
        results.append(result)
    elapsed = time.perf_counter() - start
    return {key: np.stack([result[key] for result in results])
            for key in ('raw_displacement', 'candidate', 'computable', 'eligible',
                        'coordinates', 'edge_computable')}, elapsed


def gt_composed_diagnostic(provider, case, task, config, gt4, query_mask):
    """Fixed GT branch zero per edge: diagnostic, never a physical upper bound."""
    parsed = parse_profile(case['profile']['name'])
    if parsed is not None and parsed[0] == 'elastic_transform':
        return {'status': 'not_computed_for_elastic_transformed_multiedge_GT'}, {}
    fields, provenance = {}, []
    h, w = config['context_hw']
    x, y = config['origin_xy']
    for edge_id in PATHS[task]:
        edge_task, source, _, _ = EDGES[edge_id]
        view = 'left' if source[0] == 'L' else 'right'
        frame = case['frame'] + int(source[1])
        if edge_id == 'FRbw':
            modality = 'flow_BW_right'
            path = provider.flow_root / 'train' / case['scene'] / modality / f'{modality}_{frame:04d}.flo5'
            if not path.is_file():
                return {'status': 'missing_GT_edge', 'file': str(path)}, {}
            with h5py.File(path, 'r') as stream:
                raw = np.asarray(stream['flow'][2*y:2*(y+h):2, 2*x:2*(x+w):2], np.float32)
            field = raw.transpose(2, 0, 1)
        else:
            path = provider.gt_path(edge_task, case['scene'], frame, view)
            if not path.is_file():
                return {'status': 'missing_GT_edge', 'file': str(path)}, {}
            if parse_profile(case['profile']['name']) is None:
                field = provider.read_gt(edge_task, case['scene'], frame, view,
                                         roi_xyhw=(x, y, h, w))[0]
            else:
                field = provider.read_corrupted_gt(
                    edge_task, case['scene'], frame, view=view, roi_xyhw=(x, y, h, w),
                    profile=case['profile']['name'], seed=config.get('seed', 0))[0]
        fields[f'{edge_id}_0'] = field
        provenance.append({'edge': edge_id, 'path': str(path), 'sha256': sha256(path), 'branch': 0})
    diagnostic_config = {**config, 'path_operator_tuples': [[0, 0, 0]]}
    result, seconds = build_paths(fields, task, diagnostic_config)
    raw = result['raw_displacement'][0]
    with np.errstate(invalid='ignore'):
        error = np.linalg.norm(raw[None] - gt4, axis=-1).min(axis=0)
    valid = query_mask & result['computable'][0]
    return {'status': 'GT_composed_diagnostic_not_upper_bound', 'sources': provenance,
            'computable_queries': int(valid.sum()), 'fixed_query_count': int(query_mask.sum()),
            'raw_error_computable_mean_px': float(error[valid].mean()) if valid.any() else None,
            'raw_repair_rate_fixed_Q': float((valid & (error <= config['threshold_px'])).sum()/query_mask.sum())
            if query_mask.any() else None, 'composition_seconds': seconds,
            'surface_identity_certified': False}, result


def evaluate_case(provider, case, config, directory):
    from .path_evaluation import evaluate_capacity
    fields, metadata = {}, {}
    for task in ('stereo', 'flow'):
        with np.load(directory / 'predictions' / f"{case['id']}_{task}.npz") as data:
            for key in data.files:
                fields[f'{task}_{key}' if key.startswith('rematch') else key] = data[key]
        metadata[task] = json.loads((directory / 'predictions' / f"{case['id']}_{task}.json").read_text())
    results = []
    for task in ('stereo', 'flow'):
        evaluation_start = time.perf_counter()
        path, compose_seconds = build_paths(fields, task, config)
        wrong, wrong_seconds = build_paths(fields, task, config, wrong=True)
        c0 = np.stack([fields[f'{DIRECT[task]}_{i}'].reshape(2, -1).T for i in range(3)])
        rematch = np.stack([fields[f'{task}_rematch_{i}'].reshape(2, -1).T for i in range(9)])
        rematch_valid = np.stack([fields[f'{task}_rematch_{i}_valid'].ravel() for i in range(9)])
        # This task's endpoints/masks are fixed before its labels are read.
        # Composition has no GT argument and cannot change from earlier scores.
        candidate_file = directory / 'predictions' / f"{case['id']}_{task}_candidates.npz"
        save_archive(candidate_file, config, c0=c0, rematch=rematch, rematch_valid=rematch_valid,
                 path_direct_disagreement_px=np.linalg.norm(path['raw_displacement'] - c0[0], axis=-1),
                 **{f'path_{k}': v for k, v in path.items()},
                 **{f'wrong_{k}': v for k, v in wrong.items()})
        x, y = config['origin_xy']; h, w = config['context_hw']
        if parse_profile(case['profile']['name']) is None:
            gt_source = provider.read_gt(task, case['scene'], case['frame'],
                                         roi_xyhw=(x, y, h, w))
        else:
            gt_source = provider.read_corrupted_gt(
                task, case['scene'], case['frame'], roi_xyhw=(x, y, h, w),
                profile=case['profile']['name'], seed=config.get('seed', 0))
        gt4 = gt_source.transpose(0, 2, 3, 1).reshape(4, -1, 2)
        summary, arrays = evaluate_capacity(
            c0=c0, path_raw=path['raw_displacement'], path_candidates=path['candidate'],
            path_computable=path['computable'], path_eligible=path['eligible'],
            rematch=rematch, rematch_valid=rematch_valid,
            wrong=wrong['candidate'], wrong_computable=wrong['computable'],
            wrong_eligible=wrong['eligible'], wrong_raw=wrong['raw_displacement'],
            gt4=gt4, threshold=config['threshold_px'],
            harm_delta=config['harm_delta_px'], tail_delta=config['tail_delta_px'])
        all_ledgers = [record for data in metadata.values() for record in data['ledger']]
        path_ledger = [r for r in all_ledgers if r['kind'] == 'edge' and r['edge'] in PATHS[task]]
        rematch_ledger = [r for r in metadata[task]['ledger'] if r['kind'] == 'rematch']
        base_ledger = [r for r in metadata[task]['ledger'] if r.get('edge') == DIRECT[task]]
        path_cost = sum(r['charged_seconds'] for r in path_ledger) + compose_seconds
        rematch_cost = sum(r['charged_seconds'] for r in rematch_ledger)
        cost = {'standalone_path_incremental_seconds': path_cost,
                'standalone_rematch_incremental_seconds': rematch_cost,
                'path_to_rematch_seconds_ratio': path_cost / rematch_cost,
                'C0_seconds': sum(r['charged_seconds'] for r in base_ledger),
                'path_forward_calls': sum(r['forward_calls'] for r in path_ledger),
                'rematch_forward_calls': len(rematch_ledger), 'compose_seconds': compose_seconds,
                'wrong_control_composition_seconds_separate': wrong_seconds,
                'additional_observation_wait_frames': 1 if task == 'stereo' else 0,
                'additional_observations': ['L1', 'R1'] if task == 'stereo' else ['R0', 'R1'],
                'wait_seconds': None, 'physical_fps_verified': False,
                'joint_cache_policy': 'unique edge/action once; standalone fully charges all required edges',
                'physical_joint_inference_forward_calls': len(all_ledgers),
                'physical_joint_inference_seconds': sum(r['charged_seconds'] for r in all_ledgers),
                'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in all_ledgers),
                'peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in all_ledgers)}
        # evaluator exposes the fixed offline Q explicitly; no selection uses this mask.
        query_mask = arrays['unresolved_E0_oracle']
        diagnostic, diagnostic_arrays = gt_composed_diagnostic(provider, case, task, config, gt4, query_mask)
        result = {'case': case, 'task': task, 'capacity': summary, 'cost': cost,
                  'GT_composed_diagnostic': diagnostic,
                  'candidate_artifact_sha256': sha256(candidate_file),
                  'gt_file': str(provider.gt_path(task, case['scene'], case['frame'])),
                  'gt_file_sha256': sha256(provider.gt_path(task, case['scene'], case['frame'])),
                  'actionable_E1': 'not_estimated', 'trusted_E1': 'not_calibrated',
                  'evaluation_including_composition_io_seconds': time.perf_counter() - evaluation_start}
        save_json(directory / 'metrics' / f"{case['id']}_{task}.json", result)
        save_archive(directory / 'metrics' / f"{case['id']}_{task}.npz", config, **arrays,
                 **{f'gt_diagnostic_{k}': v for k, v in diagnostic_arrays.items()})
        results.append(result)
        event(directory, 'case_evaluated', case=case['id'], task=task)
    return results


def run(config_path, directory, device):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    config['profiles'] = expand_profile_spec(config['profiles'])
    if config['model_profile'] != 'main' or config['operators'] != ['identity', 'median3', 'gaussian1']:
        raise ValueError('S02-v2 requires the registered main checkpoints and three ordered operators')
    if (len(config['path_operator_tuples']) != 9 or len(config['rematch_shifts_xy']) != 9
            or any(len(c) != 3 or any(i not in (0, 1, 2) for i in c)
                   for c in config['path_operator_tuples'])):
        raise ValueError('S02-v2 requires nine three-edge paths and nine current-pair re-estimates')
    registry_path = ROOT / config['registry']
    registry = json.loads(registry_path.read_text())
    cases = [{'id': f"{scene}_{frame:04d}_{profile['name']}", 'scene': scene,
              'frame': frame, 'profile': profile}
             for scene, frame in config['scene_frames'] for profile in config['profiles']]
    directory = initialize_run(directory, {'study': 'S02-v2', 'config': config,
        'config_file': str(config_path), 'config_sha256': sha256(config_path),
        'registry_sha256': sha256(registry_path), 'device': device,
        'spec_sha256': sha256(ROOT / config['spec']), 'cases': cases})
    save_json(directory / 'config.json', config)
    (directory / 'contract.md').write_text((ROOT / config['spec']).read_text())
    torch.set_num_threads(config['cpu_threads'])
    if not str(device).startswith('cuda:'):
        raise ValueError('Registered timing experiment requires an explicit CUDA device')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'], device)
    provider = SpringProvider.from_registry(registry_path)
    start_all = time.perf_counter()
    stages = []
    event(directory, 'inference_started', cases=len(cases), label_access=False)
    for task in ('stereo', 'flow'):
        start_load = time.perf_counter()
        model = registry['models']['main'][task]
        adapter = CroCoAdapter(task, model['local_path'], device, config['context_hw'], model['sha256'])
        stage = {'task': task, 'loading_seconds': time.perf_counter() - start_load,
                 'model': adapter.metadata}
        images, _, _ = load_quartet(provider, cases[0], config)
        _, source, target, _ = EDGES[DIRECT[task]]
        parity = adapter.native_forward_parity(images[source], images[target])
        stage['validation_warmup'] = parity
        if not parity['passed']:
            raise RuntimeError(f'{task} native forward parity failed')
        event(directory, 'model_ready', task=task, parity=parity)
        for case in cases:
            images, observations, input_seconds = load_quartet(provider, case, config)
            arrays, ledger = infer_task(adapter, images, config, task)
            save_start = time.perf_counter()
            prediction_path = directory / 'predictions' / f"{case['id']}_{task}.npz"
            save_archive(prediction_path, config, **arrays)
            save_json(prediction_path.with_suffix('.json'), {
                'case': case, 'task': task, 'observations': observations, 'ledger': ledger,
                'input_io_and_corruption_seconds': input_seconds,
                'artifact_write_seconds': time.perf_counter() - save_start,
                'artifact_sha256': sha256(prediction_path), 'GT_read': False})
            event(directory, 'inference_case', case=case['id'], task=task, calls=len(ledger))
        stages.append(stage)
        del adapter, arrays, images
        gc.collect()
        torch.cuda.empty_cache()
    save_json(directory / 'inference_complete.json', {
        'status': 'complete_all_candidates_predicted_without_GT', 'stages': stages,
        'inference_wall_seconds': time.perf_counter() - start_all,
        'device_name': torch.cuda.get_device_name(device)})
    event(directory, 'evaluation_started', all_inference_complete=True)
    results = [result for case in cases for result in evaluate_case(provider, case, config, directory)]
    from .quartet_reporting import aggregate_capacity
    report = aggregate_capacity(results, config)
    report['total_wall_seconds'] = time.perf_counter() - start_all
    report['process_peak_host_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    save_json(directory / 'metrics' / 'summary.json', report)
    save_json(directory / 'complete.json', {'status': 'complete', 'cases': len(results),
                                           'summary': 'metrics/summary.json'})
    event(directory, 'completed', cases=len(results))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:6')
    args = parser.parse_args()
    try:
        run(args.config, args.run_dir, args.device)
    except Exception as error:
        save_json(args.run_dir / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise


if __name__ == '__main__':
    main()
