"""Bind a fixed few-sample lean configuration without loading or starting a model."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re

from PIL import Image

from .common import descriptor, _fsync_parent


OFFICIAL = {
    'kitti': dict(checkpoint_sha256='656bcbeb7c6b8627eae577d5564f2849f32e1d53f85dcac7324fbeddc96bc9da',
                  core_digest='2c5e8002cb19e0f8d05ee2bfbb1cec6b0507e7690defeebc2b938c7670bf377b'),
    'sintel': dict(checkpoint_sha256='aad14eea3d68a70a4009332d941a863a1e5f513af252f6d9136725e654808322',
                   core_digest='040e5864cde10395466f9102b3ea72f629263cb1ce1068d6afcc2400ba2b97c6'),
}
ARCHITECTURE = dict(blocks_s2=1, blocks_s4=1, expansion=2,
                    profile='encoder_only_lite_v1', seed=20261030, semantic_width=8, width=16)
CLOCK_BUDGETS = dict(before_context_wait_seconds=1800, neural_active_seconds=7200,
                     overhead_seconds=7200, running_pressure_wait_seconds=21600,
                     shutdown_margin_seconds=300)
RESOURCES = dict(child_context_reserve_mib=512, device='cuda:0', gpu_uuid='auto',
                 gpu_wait_seconds=20, maximum_total_users=3, maximum_utilization_percent=100,
                 memory_fraction=.19, minimum_free_mib=5700, minimum_headroom_mib=512)
OUTPUT_POLICY = dict(epoch_summary='metrics.csv',
    evaluation='once after final epoch; last-flow EPE only; native5 MoL training unchanged',
    final_summary='result.json', final_weights='final.pt', per_step_EPE=False,
    per_step_source_hash=False, per_step_tensor_finite_scan=False,
    progress='progress.json overwritten each epoch',
    required_checks='startup binding; finite loss/gradient norm; periodic resource protection',
    resume_checkpoint='latest.pt overwritten every epoch', tensor_diagnostics=False)
SCENES = ('alley_1', 'ambush_7', 'market_5')


def _existing(path, *, directory=False):
    result = Path(path).expanduser().resolve(strict=True)
    if not (result.is_dir() if directory else result.is_file()):
        raise ValueError('Existing directory required' if directory else 'Existing regular file required')
    return result


def _training_root(root, dataset):
    root = _existing(root, directory=True)
    child = 'image_2' if dataset == 'kitti' else 'clean'
    for candidate in (root, root/'training'):
        if (candidate/child).is_dir():
            return candidate
    raise ValueError('Data root must contain the dataset training directories, or a training subdirectory')


def _rgb_geometry(one, two):
    result = None
    for path in (one, two):
        with Image.open(path) as image:
            if image.mode != 'RGB' or min(image.size) <= 0:
                raise ValueError('RGB image pair required')
            current = (image.height, image.width)
        if result is not None and current != result:
            raise ValueError('RGB pair geometry differs')
        result = current
    return list(result)


def _cases(root, dataset, official):
    bound = {}
    def bind(path):
        path = Path(path)
        if path not in bound:
            bound[path] = descriptor(path)
        return bound[path]
    rows = []
    selections = ((f'{number:06d}', None) for number in range(3)) if dataset == 'kitti' else (
        (scene, domain) for scene in SCENES for domain in ('clean', 'final'))
    for name, domain in selections:
        if dataset == 'kitti':
            one, two = (root/'image_2'/(name+suffix) for suffix in ('_10.png', '_11.png'))
            label, label_key = root/'flow_occ'/(name+'_10.png'), 'raw_gt'
            case_id = 'kitti2015/'+name
        else:
            one, two = (root/domain/name/('frame_'+number+'.png') for number in ('0001', '0002'))
            label, label_key = root/'flow'/name/'frame_0001.flo', 'gt'
            case_id = 'sintel/'+domain+'/'+name+'/frame_0001'
        row = dict(case_id=case_id, dataset=dataset, native_hw=_rgb_geometry(one, two),
                   RGB_first=bind(one), RGB_second=bind(two), partition='train',
                   official_checkpoint_sha256=official['checkpoint_sha256'],
                   backbone_state_digest=official['core_digest'])
        row[label_key] = bind(label)
        if domain is not None:
            row.update(domain=domain, scene=name)
        rows.append(row)
    return rows


def build_config(*, dataset, data_root, waft_source, checkpoint, output, resource_control, gpu_uuid):
    """Hash raw files and the official checkpoint; never import the site adapter."""
    if dataset not in OFFICIAL:
        raise ValueError('Dataset must be kitti or sintel')
    if not isinstance(gpu_uuid, str) or not re.fullmatch(
            r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', gpu_uuid):
        raise ValueError('One explicit physical GPU UUID is required')
    adapter = _existing(resource_control)
    source = _existing(waft_source, directory=True)
    output = Path(output).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ValueError('Output must be a directory path')
    official = OFFICIAL[dataset]
    saved = descriptor(_existing(checkpoint))
    if saved['sha256'] != official['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 differs from the selected official dataset profile')
    root = _training_root(data_root, dataset)
    rows = _cases(root, dataset, official)
    control = str(output/'runtime')
    profile = dict(dataset=dataset, WAFT_source=str(source), backbone_checkpoint=saved,
        expected_core_digest=official['core_digest'], cases=rows, epochs=32,
        total_updates=32*len(rows), warmup_updates=len(rows), peak_lr=.0003,
        minimum_lr_fraction=.1, shuffle_seed=20261028, eval_every_epochs=32,
        checkpoint_every_epochs=1, child_timeout_seconds=43200,
        clock_budgets=dict(CLOCK_BUDGETS), output_dir=str(output), resource_check_seconds=30,
        resource_protocol=dict(allowed_gpu_uuids=[gpu_uuid], artifact_root=control,
            resource_profile='f1_epe_full5_019', resources=dict(RESOURCES),
            runtime=dict(CPU_admission_wait_seconds=1800, running_pressure_wait_seconds=21600),
            temp_root=control))
    return dict(schema='waft-liteR-lean/v2', seed=20261030, architecture=dict(ARCHITECTURE),
        datasets={dataset: profile}, resource_control=str(adapter), raw_cache_max_bytes=128 << 20,
        automatic_full_training_promotion=False, evaluation_native_mol=False,
        evaluation_policy='end_of_training_only', output_policy=copy.deepcopy(OUTPUT_POLICY),
        default_entry='python -m waft_repair.lean.launch --dataset '+dataset+' --config <config.json>',
        resume_entry='python -m waft_repair.lean.launch --dataset '+dataset+
                     ' --config <config.json> --resume <latest.pt>')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(OFFICIAL), required=True)
    for name in ('data-root', 'waft-source', 'checkpoint', 'output', 'resource-control', 'config'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = vars(parser.parse_args(argv))
    destination = args.pop('config').expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError('Config already exists; choose a new path')
    result = build_config(**args)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n'
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor_fd, 'w', encoding='utf-8') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_parent(destination)
    print('Wrote configuration: '+str(destination.resolve()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
