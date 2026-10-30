"""Portable CPU-only configuration fixtures; no adapter execution or model load."""
import json
from pathlib import Path
import struct
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from waft_repair.lean import common, configure


GPU = 'GPU-00000000-0000-0000-0000-000000000001'


def fixture(tmp_path, dataset):
    root = tmp_path/'data'/'training'
    source = tmp_path/'official-source'
    source.mkdir()
    adapter = tmp_path/'site_adapter.py'
    adapter.write_text("raise AssertionError('Configuration must not import the adapter')\n")
    checkpoint = tmp_path/(dataset+'.pth')
    checkpoint.write_bytes(b'Fixture checkpoint: never torch.load')
    pixels = np.arange(4*5*3, dtype=np.uint8).reshape(4, 5, 3)
    if dataset == 'kitti':
        for number in range(3):
            for suffix in ('_10.png', '_11.png'):
                path = root/'image_2'/(f'{number:06d}'+suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(path)
            path = root/'flow_occ'/(f'{number:06d}_10.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'GT validated by the training startup, not a model call')
    else:
        for scene in configure.SCENES:
            for domain in ('clean', 'final'):
                for number in ('0001', '0002'):
                    path = root/domain/scene/('frame_'+number+'.png')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(pixels).save(path)
            path = root/'flow'/scene/'frame_0001.flo'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(struct.pack('<fii', 202021.25, 5, 4)+np.zeros((4,5,2), np.float32).tobytes())
    return dict(dataset=dataset, data_root=root.parent, waft_source=source, checkpoint=checkpoint,
                output=tmp_path/'results', resource_control=adapter, gpu_uuid=GPU)


def accepted_descriptor(checkpoint, dataset):
    def bound(path):
        result = common.descriptor(path)
        if Path(path).resolve() == checkpoint.resolve():
            result['sha256'] = configure.OFFICIAL[dataset]['checkpoint_sha256']
        return result
    return bound


@pytest.mark.parametrize('dataset,count,warmup', [('kitti',3,3), ('sintel',6,6)])
@pytest.mark.parametrize('training_root', [False, True])
def test_fixed_paths_order_raw_binding_and_original_schedule_resources(tmp_path, dataset, count, warmup, training_root):
    arguments = fixture(tmp_path, dataset)
    if training_root:
        arguments['data_root'] /= 'training'
    with patch.object(configure, 'descriptor', side_effect=accepted_descriptor(arguments['checkpoint'], dataset)) as bind, \
         patch.object(torch, 'load', side_effect=AssertionError('Checkpoint tensor loading forbidden')):
        result = configure.build_config(**arguments)
    profile = result['datasets'][dataset]
    assert list(result['datasets']) == [dataset]
    assert len(profile['cases']) == count
    if dataset == 'kitti':
        expected = ['kitti2015/'+f'{number:06d}' for number in range(3)]
    else:
        expected = ['sintel/'+domain+'/'+scene+'/frame_0001'
                    for scene in configure.SCENES for domain in ('clean', 'final')]
    assert [row['case_id'] for row in profile['cases']] == expected
    for row in profile['cases']:
        assert row['native_hw'] == [4,5]
        for key in ('RGB_first', 'RGB_second', 'raw_gt' if dataset == 'kitti' else 'gt'):
            assert row[key] == common.descriptor(row[key]['path'])
    assert bind.call_count == (10 if dataset == 'kitti' else 16)  # One checkpoint, unique raw files.
    assert (profile['epochs'],profile['total_updates'],profile['warmup_updates']) == (32,32*count,warmup)
    assert profile['peak_lr'] == .0003 and profile['minimum_lr_fraction'] == .1
    assert profile['shuffle_seed'] == 20261028 and profile['eval_every_epochs'] == 32
    assert profile['clock_budgets'] == dict(before_context_wait_seconds=1800, neural_active_seconds=7200,
        overhead_seconds=7200, running_pressure_wait_seconds=21600, shutdown_margin_seconds=300)
    assert profile['resource_protocol']['resources'] == dict(child_context_reserve_mib=512,
        device='cuda:0',gpu_uuid='auto',gpu_wait_seconds=20,maximum_total_users=3,
        maximum_utilization_percent=100,memory_fraction=.19,minimum_free_mib=5700,minimum_headroom_mib=512)
    assert profile['resource_protocol']['allowed_gpu_uuids'] == [GPU]
    assert profile['resource_protocol']['runtime'] == dict(CPU_admission_wait_seconds=1800,
                                                          running_pressure_wait_seconds=21600)
    control = str((tmp_path/'results'/'runtime').resolve())
    assert profile['resource_protocol']['artifact_root'] == profile['resource_protocol']['temp_root'] == control
    assert profile['WAFT_source'] == str(arguments['waft_source'].resolve()) and 'WAFT_config' not in profile
    assert profile['expected_core_digest'] == configure.OFFICIAL[dataset]['core_digest']
    assert result['raw_cache_max_bytes'] == 128 << 20
    assert result['evaluation_policy'] == 'end_of_training_only' and result['evaluation_native_mol'] is False
    assert result['output_policy'] == configure.OUTPUT_POLICY
    assert not {'strict_loader','runtime_source','memory_hygiene'} & result.keys()
    assert not {'initial_epe_by_case','resume_protocol','resume_clock'} & profile.keys()
    assert not arguments['output'].exists() and not torch.cuda.is_initialized()


def test_reject_wrong_checkpoint_missing_adapter_or_invalid_gpu_without_config_creation(tmp_path):
    arguments = fixture(tmp_path, 'kitti')
    with pytest.raises(ValueError, match='Checkpoint SHA256'):
        configure.build_config(**arguments)
    with patch.object(configure, 'descriptor', side_effect=accepted_descriptor(arguments['checkpoint'], 'kitti')):
        missing = dict(arguments, resource_control=tmp_path/'missing.py')
        with pytest.raises(FileNotFoundError):
            configure.build_config(**missing)
        with pytest.raises(ValueError, match='GPU UUID'):
            configure.build_config(**dict(arguments, gpu_uuid='0'))
    assert not arguments['output'].exists()


def test_cli_writes_only_private_new_config_and_refuses_existing_or_symlink(tmp_path):
    arguments = fixture(tmp_path, 'sintel')
    destination = tmp_path/'private'/'run.json'
    argv = [item for key,value in arguments.items() for item in ('--'+key.replace('_','-'),str(value))]
    argv += ['--config',str(destination)]
    with patch.object(configure, 'descriptor', side_effect=accepted_descriptor(arguments['checkpoint'], 'sintel')), \
         patch.object(torch, 'load', side_effect=AssertionError('Model loading forbidden')):
        assert configure.main(argv) == 0
    result = json.loads(destination.read_text())
    assert result['datasets']['sintel']['total_updates'] == 192
    assert destination.stat().st_mode & 0o777 == 0o600
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        configure.main(argv)
    assert destination.read_bytes() == before
    link = tmp_path/'linked.json'
    link.symlink_to(tmp_path/'not-yet-created.json')
    argv[-1] = str(link)
    with pytest.raises(FileExistsError):
        configure.main(argv)
    assert link.is_symlink() and not link.exists()
    assert not arguments['output'].exists() and not torch.cuda.is_initialized()
