"""S05 two-answer inputs; GT is opened separately after observable features."""
from pathlib import Path
import gc
import json
import time

import numpy as np
import torch

from .evaluation import error_map, gt_valid_mask
from .quartet_experiment import restore_translation
from .support_data import load_support_cache, read_case_labels
from .util import ROOT, save_json, save_npz, sha256


def cases_for(config, split):
    return [{"case_id": f"stereo_{scene}_{frame:04d}_{profile['name']}",
             "task": "stereo", "scene": scene, "frame": frame,
             "profile": profile, "split": "train", "study_split": split}
            for scene, frames in config['splits'][split].items()
            for frame in frames for profile in config['profiles']]


def project_two_answers(arrays, candidate_index):
    """Discard every other candidate and support/GT channel before features."""
    initial = arrays['initial'].copy()
    candidate = arrays['e0_bank'][candidate_index].copy()
    eligible = arrays['e0_valid'][candidate_index].copy() & np.isfinite(candidate).all(0)
    candidate[:, ~eligible] = initial[:, ~eligible]
    return {key: arrays[key].copy() for key in
            ('image0', 'image1', 'source_features', 'target_features', 'raw_uncertainty')} | {
                'initial': initial, 'candidate': candidate, 'candidate_valid': eligible}


def read_old_training(config, case):
    source = ROOT / config['training_cache'] / 'train' / (case['case_id'] + '.npz')
    loaded = load_support_cache(source)
    meta, old = loaded['metadata'], loaded['arrays']
    if meta['context_hw'] != config['context_hw'] or meta['origin_xy'] != config['origin_xy']:
        raise ValueError('Training cache geometry mismatch')
    if meta['case']['scene'] != case['scene'] or meta['case']['frame'] != case['frame'] or meta['seed'] != config['seed']:
        raise ValueError('Training cache provenance mismatch')
    indices = [i for i, entry in enumerate(meta['bank_order'])
               if entry['kind'] == 'shift' and entry['operation'] == config['candidate_shift_xy']]
    if len(indices) != 1:
        raise ValueError('Exact fixed operation missing from old training cache')
    idx = indices[0]
    arrays = project_two_answers(old, idx)
    selected = [meta['ledger'][0], meta['ledger'][idx]]
    return arrays, {'source_path': str(source), 'source_sha256': meta['artifact']['sha256'],
                    'candidate_bank_index': idx, 'gt_read': False,
                    'runtime_forward_calls': 2, 'historical_cache_forward_calls': 12,
                    'charged_forward_seconds': sum(row['charged_seconds'] for row in selected),
                    'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in selected)}, source


def read_old_labels(source, arrays):
    path = source.with_name(source.stem + '_labels.npz')
    meta = json.loads(path.with_suffix('.json').read_text())
    if sha256(path) != meta['artifact']['sha256'] or not meta['gt_read']:
        raise ValueError('Label cache integrity failure')
    with np.load(path, allow_pickle=False) as archive:
        gt4 = archive['gt4']
    return score_labels(arrays, gt4), {'path': str(path), 'sha256': sha256(path),
                                     'role': 'training_targets_only'}


def score_labels(arrays, gt4):
    return {'e0': error_map(arrays['initial'], gt4),
            'ea': error_map(arrays['candidate'], gt4), 'valid': gt_valid_mask(gt4)}


def infer_pair(config, case, provider, adapter):
    start = time.perf_counter()
    profile = case['profile']
    pair = provider.read_pair('stereo', case['scene'], case['frame'], config['context_hw'],
                              split='train', profile=profile['name'], seed=config['seed'],
                              corruption_options=profile.get('options'), origin_xy=config['origin_xy'])
    if pair['metadata'].get('gt_read'):
        raise ValueError('GT entered pair construction')
    torch.cuda.reset_peak_memory_stats(adapter.device)
    initial = adapter.predict(pair['image0'], pair['image1'], origin_xy=config['origin_xy'])
    dx, dy = config['candidate_shift_xy']
    shifted = adapter.predict(np.roll(pair['image0'], (dy, dx), axis=(0, 1)),
                              np.roll(pair['image1'], (dy, dx), axis=(0, 1)),
                              origin_xy=config['origin_xy'])
    candidate, eligible = restore_translation(shifted.displacement, (dx, dy))
    eligible &= np.isfinite(candidate).all(0)
    candidate[:, ~eligible] = initial.displacement[:, ~eligible]
    arrays = {'image0': pair['image0'], 'image1': pair['image1'],
              'source_features': initial.source_features, 'target_features': initial.target_features,
              'initial': initial.displacement, 'candidate': candidate,
              'candidate_valid': eligible, 'raw_uncertainty': initial.raw_uncertainty}
    torch.cuda.synchronize(adapter.device)
    metadata = {'pair': pair['metadata'], 'gt_read': False, 'runtime_forward_calls': 2,
                'historical_cache_forward_calls': 2, 'charged_forward_seconds': time.perf_counter()-start,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(adapter.device),
                'checkpoint_sha256': adapter.metadata['checkpoint'], 'observation_wait_frames': 0}
    return arrays, metadata


def prepare_split(config, directory, split, device, check_deadline):
    from .binary_revision import build_descriptors
    from .backbones import CroCoAdapter
    from .data import SpringProvider
    from .util import event
    adapter = provider = None
    if split != 'train':
        registry = json.loads((ROOT / config['registry']).read_text())
        checkpoint = registry['models']['main']['stereo']
        provider = SpringProvider.from_registry(ROOT / config['registry'])
        adapter = CroCoAdapter('stereo', checkpoint['local_path'], device=device,
                              context_hw=config['context_hw'], expected_sha256=checkpoint['sha256'])
    for index, case in enumerate(cases_for(config, split)):
        check_deadline()
        stem = directory / 'data' / split / case['case_id']
        stem.parent.mkdir(parents=True, exist_ok=True)
        if stem.with_suffix('.npy').exists():
            raise FileExistsError('No silent reuse of partial new S05 cache')
        start = time.perf_counter()
        if split == 'train':
            arrays, metadata, old_source = read_old_training(config, case)
        else:
            arrays, metadata = infer_pair(config, case, provider, adapter)
            save_npz(stem.with_name(stem.name + '_inputs.npz'), **arrays)
        descriptor_start = time.perf_counter()
        descriptors, schema = build_descriptors(arrays, device=device, chunk_size=config['descriptor_chunk'])
        metadata['descriptor_wall_seconds'] = time.perf_counter()-descriptor_start
        # Commit inference-only inputs before opening separate labels.
        np.save(stem.with_suffix('.npy'), descriptors, allow_pickle=False)
        save_npz(stem.with_name(stem.name + '_observable.npz'),
                 eligible=arrays['candidate_valid'], uncertainty=arrays['raw_uncertainty'],
                 delta_norm=np.linalg.norm(arrays['candidate']-arrays['initial'], axis=0))
        if split == 'train':
            labels, label_meta = read_old_labels(old_source, arrays)
        else:
            record = read_case_labels(provider, case, roi_xyhw=metadata['pair']['roi_xyhw'],
                                      seed=config['seed'])
            labels, label_meta = score_labels(arrays, record['gt4']), record['metadata']
        save_npz(stem.with_name(stem.name + '_labels.npz'), **labels)
        save_json(stem.with_suffix('.json'), {'case': case, 'input': metadata, 'descriptor': schema,
                  'descriptor_sha256': sha256(stem.with_suffix('.npy')), 'labels': label_meta,
                  'prepare_wall_seconds': time.perf_counter()-start})
        event(directory, 'data_case_complete', split=split, case=case['case_id'], index=index,
              seconds=time.perf_counter()-start)
        del arrays, descriptors, labels
    del adapter, provider
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()


def load_prepared(directory, case, *, with_labels=True):
    stem = directory / 'data' / case['study_split'] / case['case_id']
    descriptors = np.load(stem.with_suffix('.npy'), mmap_mode='r')
    with np.load(stem.with_name(stem.name + '_observable.npz')) as z:
        observable = {k: z[k] for k in z.files}
    labels = None
    if with_labels:
        with np.load(stem.with_name(stem.name + '_labels.npz')) as z:
            labels = {k: z[k] for k in z.files}
    return descriptors, observable, labels
