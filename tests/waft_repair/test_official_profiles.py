"""Inference weights retain their full official backbone identity."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from waft_repair.backend import CHECKPOINT_PROFILES
from waft_repair.motion.checkpoint import export_model_checkpoint,load_model_checkpoint,import_checkpoint
from waft_repair.motion.io import atomic_torch
from waft_repair.motion.model import MotionFeatureRepair
from waft_repair import native_pipeline as P


@pytest.mark.parametrize('digest,profile',CHECKPOINT_PROFILES.items())
def test_export_load_import_preserve_the_selected_official_profile(tmp_path,digest,profile):
    model=MotionFeatureRepair();packet=export_model_checkpoint(model,7,waft_checkpoint_sha256=digest)
    assert packet['core_state_digest']==profile['state_sha256']
    path=tmp_path/'model.pt';atomic_torch(path,packet)
    _,receipt=load_model_checkpoint(path)
    assert receipt['waft_checkpoint_sha256']==digest and receipt['core_state_digest']==profile['state_sha256']
    imported=tmp_path/'imported.pt';import_checkpoint(path,imported)
    assert torch.load(imported,weights_only=True)['waft_checkpoint_sha256']==digest
    packet['core_state_digest']='0'*64;atomic_torch(path,packet)
    with pytest.raises(ValueError,match='WAFT identity'):load_model_checkpoint(path)


def test_inference_rejects_repair_backbone_mismatch_before_image_or_NN_work(tmp_path):
    source,sintel=list(CHECKPOINT_PROFILES)[::2]
    with patch('waft_repair.motion.checkpoint.load_model_checkpoint',return_value=(object(),dict(waft_checkpoint_sha256=sintel))), \
         patch.object(P,'_system',return_value=(SimpleNamespace(),dict(waft_checkpoint_sha256=source))), \
         patch.object(P,'load_rgb_pair',side_effect=AssertionError('RGB read before identity check')):
        with pytest.raises(ValueError,match='profiles differ'):
            P.infer('repair','one','two',tmp_path/'prediction.npz','source','weights')


def test_system_uses_the_selected_core_digest_and_releases_the_checkpoint_cache(tmp_path):
    path=tmp_path/'sintel.pth';path.write_bytes(b'fixture')
    backend=SimpleNamespace(core=object(),metadata=dict(state_sha256='bound-digest'),
        padder_factory=object(),coords_grid=object(),bilinear_sampler=object())
    with patch.object(P,'load_backend',return_value=backend), \
         patch.object(P,'MotionNativeRepairSystem',return_value=object()) as system, \
         patch.object(P,'case_boundary') as boundary:
        _,metadata=P._system('source',path,'cpu')
    assert metadata==backend.metadata
    assert system.call_args.kwargs['expected_core_digest']=='bound-digest'
    assert boundary.call_args.kwargs['paths']==[path]
