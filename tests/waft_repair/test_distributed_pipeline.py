"""CPU-only portable driver stop/resume; synthetic consumer, full repair model."""
from datetime import timedelta
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.multiprocessing import spawn

from waft_repair import distributed_pipeline as P
from waft_repair.backend import CHECKPOINT_PROFILES
from waft_repair.cli import parser
from waft_repair.motion import distributed as D
from waft_repair.motion.data import load_manifest
from waft_repair.motion.io import atomic_json
from waft_repair.motion.model import MotionFeatureRepair
from test_distributed import free_port
from test_motion_model import fixture
from test_motion_training import FrozenConsumer


class FixtureSystem(FrozenConsumer):
    def __init__(self):
        super().__init__();self.restorer=MotionFeatureRepair()
    def eval(self):self.restorer.eval();return self
    def verify_core_binding(self):self.validate()


def _driver_worker(rank,first_port,second_port,spec,output):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(first_port),WORLD_SIZE='2',RANK=str(rank),LOCAL_RANK=str(rank))
    init=P.dist.init_process_group
    def bounded(*args,**kwargs):kwargs['timeout']=timedelta(seconds=60);return init(*args,**kwargs)
    P.dist.init_process_group=bounded
    D.registered_consumer_contract=lambda *args:True
    def system(*args,**kwargs):
        digest=next(iter(CHECKPOINT_PROFILES))
        return FixtureSystem(),dict(state_sha256=CHECKPOINT_PROFILES[digest]['state_sha256'],waft_checkpoint_sha256=digest)
    def raw(system,row,device):
        original,sigma,taps,prior,native,padding=fixture()
        prepared=dict(original=original,sigma=sigma,observations=taps,motion_prior=prior,
            native_hw=native,padding=padding,baseline_flow=torch.zeros(1,2,*native),baseline_info=torch.zeros(1,4,*native))
        gt,valid=P.read_labels(row['ground_truth']['path'],'npz')
        with torch.no_grad():baseline=system.solve_features(original,native_hw=native,padding=padding)
        with (Path(output)/f'fixture-visits-{rank}.jsonl').open('a') as stream:stream.write(json.dumps(row['id'])+'\n')
        return prepared,gt,valid,baseline
    P._system=system;P._raw_case=raw
    args=dict(epochs=3,warmup_updates=1,device='cpu',reserve_gib=0,checkpoint_every=1,activation_offload='none')
    first=P.train(spec,output,'fixture-source','fixture-weights',stop_after_updates=1,**args)
    assert first['completed_updates']==1 and first['status']=='STOPPED_AT_REQUESTED_UPDATE'
    os.environ['MASTER_PORT']=str(second_port)
    result=P.train(spec,output,'fixture-source','fixture-weights',resume=str(Path(output)/'checkpoint.pt'),**args)
    assert result['status']=='COMPLETE' and result['completed_updates']==3
    assert not torch.cuda.is_initialized()


def test_two_rank_driver_stops_resumes_without_repeating_initial_evaluation(tmp_path):
    rows=[]
    for index in range(2):
        for frame in (1,2):Image.fromarray(np.full((57,61,3),30+index*20+frame,np.uint8)).save(tmp_path/f'{index}-{frame}.png')
        np.savez(tmp_path/f'{index}.npz',flow=np.full((57,61,2),.1+index*.2,np.float32),valid=np.ones((57,61),bool))
        rows.append(dict(id=f'synthetic/{index}',split='train',image1=f'{index}-1.png',image2=f'{index}-2.png',
            ground_truth=f'{index}.npz',ground_truth_format='npz'))
    spec=tmp_path/'images.json';atomic_json(spec,dict(schema='waft-native-images/v1',samples=rows))
    output=tmp_path/'run'
    spawn(_driver_worker,args=(free_port(),free_port(),str(spec),str(output)),nprocs=2,join=True)
    state=torch.load(output/'checkpoint.pt',weights_only=True)
    assert state['completed_updates']==3 and state['cursor']['next_case_ids']==[]
    assert len(state['repair'])==len(state['optimizer']['state'])==161
    assert all(int(row['step'])==3 for row in state['optimizer']['state'].values())
    for rank in (0,1):
        history=[json.loads(line) for line in (output/('history.jsonl' if rank==0 else 'history-rank1.jsonl')).read_text().splitlines()]
        assert [row['completed_updates'] for row in history]==[1,2,3]
        # One initial eval, three TRAIN visits, one final eval on each rank.
        assert len((output/f'fixture-visits-{rank}.jsonl').read_text().splitlines())==5
    assert len(list((output/'evaluations/init').glob('*.pt')))==2
    assert len(list((output/'evaluations/E003').glob('*.pt')))==2


@pytest.mark.parametrize('digest,profile',CHECKPOINT_PROFILES.items())
def test_manifest_binds_checkpoint_and_all508_digest_together(tmp_path,digest,profile):
    value=dict(schema='waft-native-manifest/v1',model_abi='paired-feature-repair/motion-aligned-v1',
        waft=dict(waft_checkpoint_sha256=digest,state_sha256=profile['state_sha256']),samples=[dict(id='x',split='train')])
    path=tmp_path/'manifest.json';atomic_json(path,value);load_manifest(path)
    value['waft']['state_sha256']='0'*64;atomic_json(path,value)
    with pytest.raises(ValueError,match='declared official'):load_manifest(path)


def test_sintel_ddp_cli_is_explicit_two_rank_rgb_training():
    args=parser().parse_args(['native-train-ddp','--spec','images.json','--output','run',
        '--waft-source','source','--waft-checkpoint','sintel.pth'])
    assert args.epochs==80 and args.warmup_updates==796 and args.checkpoint_every==256 and args.device=='cuda'
    with pytest.raises(SystemExit):parser().parse_args(['native-train-ddp','--teacher','features'])
