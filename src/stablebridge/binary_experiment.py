"""Bounded, detached S05 binary learnability experiment; fixed final checkpoints."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch

from .binary_data import cases_for, load_prepared, prepare_split
from .binary_revision import BinaryRevisionMLP, DESCRIPTOR_DIM, GEOMETRY_DIM
from .binary_reporting import evaluate_case, aggregate, choose_threshold, apply_policy, confirmation_gate
from .util import ROOT, event, initialize_run, save_json, save_npz, sha256
from .robust_proxy import expand_profile_spec


def validate_config(config):
    if config['contract'] != 'CSB-S05-BINARY-REVISION-v1-20260917':
        raise ValueError('Unknown S05 contract')
    if config['candidate_shift_xy'] != [-8, 0] or config['context_hw'] != [320, 384]:
        raise ValueError('S05 fixed operation/context violated')
    used = set()
    for split, scenes in config['splits'].items():
        if used.intersection(scenes):
            raise ValueError('Scene overlap across splits')
        used.update(scenes)
    if set(config['splits']['confirmation']) & set(config['exposure']['previously_exposed']):
        raise ValueError('Confirmation scene already exposed')
    if config['learning']['target_clipping'] is not None:
        raise ValueError('This experiment uses unclipped native targets')


def model_names(config):
    return [(f'{info}_{objective}_s{seed}', info, objective, seed)
            for seed in config['learning']['seeds']
            for info in ('geometry', 'full') for objective in ('gain', 'errors')]


def make_model(config, info, objective, device):
    return BinaryRevisionMLP(DESCRIPTOR_DIM, GEOMETRY_DIM, information=info,
                             objective=objective, hidden_dim=config['model']['hidden_dim']).to(device)


def train_models(config, directory, device, check_deadline):
    learning = config['learning']
    data = []
    for case in cases_for(config, 'train'):
        features, _, labels = load_prepared(directory, case)
        valid_indices = np.flatnonzero(labels['valid'])
        if not len(valid_indices):
            raise ValueError('Empty training scene case, no silent reweighting')
        data.append((features.reshape(-1, DESCRIPTOR_DIM), labels['e0'].reshape(-1),
                     labels['ea'].reshape(-1), valid_indices))
    for name, info, objective, seed in model_names(config):
        check_deadline()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        model = make_model(config, info, objective, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning['learning_rate'],
                                     weight_decay=learning['weight_decay'])
        start = time.perf_counter()
        losses = []
        for step in range(learning['steps_per_model']):
            check_deadline()
            # All models of one seed receive the exact same case/query draws.
            features, e0, ea, valid_indices = data[int(rng.integers(len(data)))]
            indices = rng.choice(valid_indices, size=learning['queries_per_step'], replace=True)
            inputs = torch.as_tensor(np.array(features[indices]), device=device)
            if objective == 'gain':
                targets = (e0[indices]-ea[indices])[:, None]
            else:
                targets = np.stack([e0[indices], ea[indices]], axis=-1)
            targets = torch.as_tensor(targets / learning['target_scale_px'], device=device)
            outputs = model(inputs)
            loss = (outputs-targets).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite loss; no automatic objective change')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), learning['gradient_clip'])
            optimizer.step()
            losses.append(float(loss.detach()))
            if (step+1) % 250 == 0 or step+1 == learning['steps_per_model']:
                event(directory, 'train_progress', model=name, step=step+1,
                      loss=float(np.mean(losses[-250:])), seconds=time.perf_counter()-start)
        torch.cuda.synchronize(device)
        path = directory / 'checkpoints' / (name + '.pth')
        torch.save({'state_dict': model.state_dict(), 'information': info, 'objective': objective,
                    'seed': seed, 'steps': learning['steps_per_model']}, path)
        save_json(path.with_suffix('.json'), {'name': name, 'sha256': sha256(path),
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'hidden_dim': config['model']['hidden_dim'], 'information': info,
                  'objective': objective, 'wall_seconds': time.perf_counter()-start,
                  'final_loss': losses[-1], 'initial_loss': losses[0], 'seed': seed,
                  'final_step_only': True})
        event(directory, 'model_complete', model=name, seconds=time.perf_counter()-start)
        del model, optimizer


def baseline_scores(features, observable, schema):
    names = schema['feature_names']
    # Matching score name is asserted explicitly; never infer orientation from GT.
    match_index = names.index('candidate_minus_initial_patch_rgb_l1')
    return {'uncertainty': observable['uncertainty'],
            'small_change': -observable['delta_norm'],
            'rgb_match': -features[..., match_index]}


def score_split(config, directory, split, device, check_deadline):
    models = {}
    for name, info, objective, seed in model_names(config):
        model = make_model(config, info, objective, device)
        state = torch.load(directory / 'checkpoints' / (name+'.pth'), map_location=device, weights_only=True)
        model.load_state_dict(state['state_dict'], strict=True)
        models[name] = (model.eval(), objective)
    output = {}
    for case in cases_for(config, split):
        check_deadline()
        features, obs, _ = load_prepared(directory, case, with_labels=False)
        schema = json.loads((directory / 'data' / split / (case['case_id']+'.json')).read_text())['descriptor']
        scores = baseline_scores(features, obs, schema)
        flat = features.reshape(-1, DESCRIPTOR_DIM)
        timings = {}
        for name, (model, objective) in models.items():
            start = time.perf_counter()
            chunks = []
            with torch.inference_mode():
                for offset in range(0, len(flat), config['score_chunk']):
                    x = torch.as_tensor(np.array(flat[offset:offset+config['score_chunk']]), device=device)
                    out = model(x)
                    gain = out[:, 0] if objective == 'gain' else out[:, 0]-out[:, 1]
                    chunks.append(gain.cpu().numpy()*config['learning']['target_scale_px'])
            scores[name] = np.concatenate(chunks).reshape(features.shape[:2])
            timings[name] = time.perf_counter()-start
        meta = {'case_id': case['case_id'], 'task': 'stereo', 'scene': case['scene'],
                'frame': case['frame'], 'profile': case['profile']['name'], 'split': split}
        # Ground-truth errors are attached ONLY after all inference scores exist.
        label_path = directory / 'data' / split / (case['case_id']+'_labels.npz')
        with np.load(label_path) as archive:
            labels = {k: archive[k] for k in archive.files}
        for name, score in scores.items():
            output.setdefault(name, []).append({'score': score, 'e0': labels['e0'], 'ea': labels['ea'],
                 'valid': labels['valid'], 'eligible': obs['eligible'], 'metadata': meta})
        save_npz(directory / 'scores' / split / (case['case_id']+'.npz'), **scores)
        save_json(directory / 'scores' / split / (case['case_id']+'.json'), {'metadata': meta, 'timings': timings,
                  'runtime_forward_calls': 2, 'observation_wait_frames': 0})
        event(directory, 'scores_complete', split=split, case=case['case_id'])
    return output


def metric(summary, name):
    return summary['metrics'][name]['scene_macro']


def verify_calibration_lock(directory):
    record = json.loads((directory / 'calibration_complete.json').read_text())
    if sha256(directory / 'policies.json') != record['policy_sha256']:
        raise ValueError('Policy changed after calibration')
    for name, digest in record['checkpoints'].items():
        if sha256(directory / 'checkpoints' / (name+'.pth')) != digest:
            raise ValueError(f'Checkpoint changed after calibration: {name}')


def screen(summary, config):
    return confirmation_gate(summary, config['acceptance'])


def harm_feasible(summary, config):
    return all(metric(summary, key) is not None and metric(summary, key) <= config['acceptance'][limit]
               for key, limit in [('damage_good_fraction', 'max_damage_good_fraction'),
                                 ('severe_harm_fraction', 'max_severe_harm_fraction')])


def evaluate_policies(cases_by_method, policies, directory, split, config):
    results = {}
    source_cases = next(iter(cases_by_method.values()))
    for name, cases in cases_by_method.items():
        rows = [evaluate_case(c['e0'], c['ea'], c['valid'], c['eligible'],
                              apply_policy(c['score'], c['eligible'], policies[name]), c['metadata']) for c in cases]
        summary = aggregate(rows)
        results[name] = {'policy': policies[name], 'rows': rows, 'summary': summary, 'screen': screen(summary, config)}
    for name in ('identity', 'allcandidate'):
        rows = [evaluate_case(c['e0'], c['ea'], c['valid'], c['eligible'],
                    np.zeros_like(c['eligible']) if name == 'identity' else c['eligible'], c['metadata']) for c in source_cases]
        results[name] = {'rows': rows, 'summary': aggregate(rows)}
    save_json(directory / 'metrics' / (split+'.json'), results)
    return results


def scientific_decision(results, config):
    baselines = ['identity', 'allcandidate', 'uncertainty', 'small_change', 'rgb_match']
    feasible = [n for n in baselines if harm_feasible(results[n]['summary'], config)]
    feasible = [n for n in feasible if metric(results[n]['summary'], 'gain_px') is not None]
    best = max(feasible, key=lambda n: metric(results[n]['summary'], 'gain_px')) if feasible else None
    margin = config['acceptance']['increment_over_baseline_px']
    def difference(left, right):
        a = metric(results[left]['summary'], 'gain_px') if left is not None else None
        b = metric(results[right]['summary'], 'gain_px') if right is not None else None
        return a-b if a is not None and b is not None else None
    entries = {}
    for info in ('geometry', 'full'):
        for objective in ('gain', 'errors'):
            names = [f'{info}_{objective}_s{s}' for s in config['learning']['seeds']]
            entries[f'{info}_{objective}'] = {
                'passed_both_seeds': all(results[n]['screen']['passed'] and difference(n, best) is not None and
                    difference(n, best) >= margin for n in names),
                'increment_over_best_feasible_baseline_px': {
                    n: difference(n, best) for n in names}}
    evidence = {}
    for objective in ('gain', 'errors'):
        deltas = {str(seed): difference(f'full_{objective}_s{seed}', f'geometry_{objective}_s{seed}')
                  for seed in config['learning']['seeds']}
        evidence[objective] = {'full_minus_geometry_px': deltas,
            'increment_supported_both_seeds': all(v is not None and v >= margin for v in deltas.values()) and
                all(results[f'full_{objective}_s{s}']['screen']['passed'] for s in config['learning']['seeds'])}
    return {'primary_full_gain_passed': entries['full_gain']['passed_both_seeds'],
            'any_registered_learned_arm_passed': any(v['passed_both_seeds'] for v in entries.values()),
            'best_harm_feasible_simple_baseline': best, 'simple_feasible': feasible,
            'arms': entries, 'matching_evidence_increment': evidence,
            'interpretation': 'bounded mechanism screen; no significance, calibrated per-pixel trust, roll causality, flow or novelty claim'}


def render_report(report):
    lines = ['# S05 binary revision', '', f"Scope: {report['scope']}", '',
             f"Wall time: {report['wall_seconds']/60:.1f} min.", '',
             '| Method | gain px | update % | original-good damage % | >3 px harm % | screen |',
             '|---|---:|---:|---:|---:|---|']
    for name, result in report['results'].items():
        def fmt(k, scale=1):
            v = metric(result['summary'], k)
            return 'undefined' if v is None else f'{v*scale:.6f}'
        lines.append(f"| {name} | {fmt('gain_px')} | {fmt('update_fraction',100)} | {fmt('damage_good_fraction',100)} | {fmt('severe_harm_fraction',100)} | {result.get('screen',{}).get('passed','reference')} |")
    lines += ['', '## Decision', '', '```json', json.dumps(report.get('decision', {}), ensure_ascii=False, indent=2), '```', '',
              'Native errors are unclipped. Scenes receive equal weight; pixels are correlated. '+
              'These harm limits are an empirical screen, not a deployment safety guarantee. '+
              'The same fixed crop is used throughout. Old training caches were reused; fresh evaluation executes two frozen forwards per pair.']
    return '\n'.join(lines)+'\n'


def run(config_path, directory, device, smoke=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    config['profiles'] = expand_profile_spec(config['profiles'])
    validate_config(config)
    if smoke:
        config = copy.deepcopy(config)
        scene = next(iter(config['splits']['train']))
        config['splits'] = {'train': {scene: [config['splits']['train'][scene][0]]}}
        config['profiles'] = [{'name': 'clean'}]
        config['learning']['steps_per_model'] = 5
        config['learning']['seeds'] = config['learning']['seeds'][:1]
        config['acceptance'].update(expected_scenes=1, min_positive_scenes=1)
    if (directory / 'manifest.json').exists():
        raise FileExistsError('No silent resume; use a new run directory')
    if shutil.disk_usage(ROOT).free < 6*10**9:
        raise RuntimeError('At least 6 GB free workspace space required')
    inputs = [config_path, ROOT / config['spec'], ROOT / config['registry'], ROOT / config['exposure']['audit_path']]
    initialize_run(directory, {'study': config['study'], 'config': config, 'smoke': smoke,
         'registered_inputs': [{'path': str(p), 'sha256': sha256(p)} for p in inputs],
         'device': device, 'training_detached_expected': True})
    save_json(directory / 'config.json', config)
    (directory / 'contract.md').write_text((ROOT / config['spec']).read_text())
    torch.set_num_threads(config['cpu_threads'])
    if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
        raise RuntimeError('GPU scientific worker requires CUDA; no silent fallback')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'], device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    deadline = start + config['limits']['maximum_run_minutes']*60
    def check_deadline():
        if time.perf_counter() > deadline:
            raise TimeoutError('Fixed experiment budget exhausted; no automatic extension')
    stage = 'startup'
    def status(value):
        nonlocal stage
        from .support_experiment import verify_frozen
        stage = value
        verify_frozen(directory)
        save_json(directory / 'status.json', {'state': 'running', 'stage': stage, 'pid': os.getpid(), 'updated_unix': time.time(), 'smoke': smoke})
    try:
        status('prepare_existing_training_data')
        prepare_split(config, directory, 'train', device, check_deadline)
        status('train_fixed_budget_models')
        train_models(config, directory, device, check_deadline)
        split = 'train' if smoke else 'calibration'
        if not smoke:
            status('prepare_fresh_calibration')
            prepare_split(config, directory, split, device, check_deadline)
        status('calibration_scores')
        scores = score_split(config, directory, split, device, check_deadline)
        if smoke:
            for cases in scores.values():
                for case in cases:
                    case['metadata'] = {**case['metadata'], 'split': 'calibration',
                        'engineering_source_split': 'train', 'engineering_smoke': True}
        status('calibrate_and_freeze_thresholds')
        policies = {}
        for name, cases in scores.items():
            calibrated = choose_threshold(cases, config['acceptance'])
            policies[name] = calibrated['policy']
            save_json(directory / 'calibration' / (name+'.json'), calibrated)
        save_json(directory / 'policies.json', policies)
        policy_hash = sha256(directory / 'policies.json')
        save_json(directory / 'calibration_complete.json', {'policy_sha256': policy_hash, 'unix_time': time.time(),
              'checkpoints': {n: sha256(directory / 'checkpoints' / (n+'.pth')) for n,_,_,_ in model_names(config)}})
        if not smoke:
            del scores
            verify_calibration_lock(directory)
            status('prepare_fresh_confirmation_after_policy_lock')
            prepare_split(config, directory, 'confirmation', device, check_deadline)
            verify_calibration_lock(directory)
            status('confirmation_scores')
            scores = score_split(config, directory, 'confirmation', device, check_deadline)
            verify_calibration_lock(directory)
        status('evaluate_fixed_policies')
        results = evaluate_policies(scores, policies, directory, 'train' if smoke else 'confirmation', config)
        report = {'scope': 'engineering smoke only' if smoke else config['scientific_scope'],
                  'smoke': smoke, 'wall_seconds': time.perf_counter()-start, 'results': results,
                  'decision': {} if smoke else scientific_decision(results, config)}
        save_json(directory / 'report.json', report)
        (directory / 'report.md').write_text(render_report(report))
        save_json(directory / 'status.json', {'state': 'completed', 'stage': 'completed', 'smoke': smoke,
                  'wall_seconds': report['wall_seconds'], 'updated_unix': time.time()})
        event(directory, 'completed', wall_seconds=report['wall_seconds'], smoke=smoke)
    except Exception as error:
        save_json(directory / 'status.json', {'state': 'failed', 'stage': stage, 'error': repr(error),
                  'traceback': traceback.format_exc(), 'updated_unix': time.time(), 'smoke': smoke})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        run(args.config, args.output, args.device, args.smoke)
    except Exception as error:
        if not (args.output / 'status.json').exists():
            save_json(args.output / 'status.json', {'state': 'failed', 'stage': 'startup',
                      'error': repr(error), 'traceback': traceback.format_exc(), 'smoke': args.smoke})
        raise


if __name__ == '__main__':
    main()
