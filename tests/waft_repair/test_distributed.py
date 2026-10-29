"""Two real Gloo CPU processes: gradient mean, AdamW and both rank RNG replay."""
import copy
from datetime import timedelta
import os
import random
import socket

import numpy as np
import pytest
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import spawn

from waft_repair.motion import distributed as D
from waft_repair.motion.training import repair_forward,native_task_loss
from test_motion_training import fixture,FrozenConsumer


class Scalar161(nn.Module):
    """161 tiny trainable tensors exercise complete optimizer-state checks."""
    def __init__(self):
        super().__init__();self.values=nn.ParameterList([nn.Parameter(torch.tensor(.001)) for _ in range(161)])
        self.dropout=False

    def forward(self,original,sigma,observations,*baseline,native_hw,padding,**kw):
        delta=sum(self.values)/161
        if self.dropout:delta=delta*torch.rand(())
        return original+delta


def free_port():
    with socket.socket() as stream:
        stream.bind(('127.0.0.1',0));return stream.getsockname()[1]


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def _worker(rank,port,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2,timeout=timedelta(seconds=45))
    try:
        # Fixture-only substitution: neither test pretends to load official WAFT.
        D.registered_model_contract=lambda *args:True;D.registered_consumer_contract=lambda *args:True
        torch.manual_seed(123);np.random.seed(123+rank);random.seed(123+rank)
        model=Scalar161();consumer=FrozenConsumer()
        initial=copy.deepcopy(model.state_dict());config=D.TrainConfig(updates=4,warmup_updates=1)
        trainer=D.DistributedNativeTrainer(model,DistributedDataParallel(model),consumer,config,
            model_config={'fixture':'161-scalars'},binding=dict(fixed_cases=list('abcd'),epochs=2,
                shuffle_seed=20261028,world_size=2,native_objective=D.NATIVE_OBJECTIVE),
            core_binding={'fixture':'frozen'},observation_binding={},source_binding={'fixture':'v1'})
        prepared,gt,valid=fixture();gt=gt+.2*rank
        averaged={}
        trainer.step(prepared,gt,valid,raw_gradient_callback=lambda values:averaged.update(values))
        reference=Scalar161();reference.load_state_dict(initial)
        optimizer=torch.optim.AdamW(reference.parameters(),lr=D.learning_rate(config,1),eps=config.epsilon,weight_decay=0,foreach=False)
        for r in (0,1):
            source,truth,mask=fixture();output_values=consumer.solve_features(repair_forward(reference,source),native_hw=source['native_hw'],padding=source['padding'])
            (native_task_loss(output_values,truth+.2*r,mask)/2).backward()
        norm=torch.stack([p.grad.double().square().sum() for p in reference.parameters()]).sum().sqrt()
        scale=min(1.,config.clip_norm/(float(norm)+1e-12))
        for name,p in reference.named_parameters():
            torch.testing.assert_close(averaged[name],p.grad,rtol=2e-6,atol=1e-9)
            p.grad.mul_(scale)
        optimizer.step()
        for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=1e-9)
        torch.rand(rank+1);random.random();np.random.rand(rank+1)
        state=trainer.checkpoint_state(cursor=D.expected_cursor(trainer.binding,1,'one'))
        box=[state];dist.broadcast_object_list(box,src=0);state=box[0]
        trainer.validate_checkpoint(state)
        model.dropout=True
        trainer.step(prepared,gt,valid)
        expected=D.saved_state_digests(model.state_dict(),trainer.optimizer.state_dict(),2)
        trainer.load_state_dict(state)
        assert torch.equal(torch.get_rng_state(),state['ddp']['rank_rng'][str(rank)]['torch_rng'])
        trainer.step(prepared,gt,valid)
        assert D.saved_state_digests(model.state_dict(),trainer.optimizer.state_dict(),2)==expected
        damaged=copy.deepcopy(state);damaged['optimizer']['state'][0]['step']+=1
        before=D.saved_state_digests(model.state_dict(),trainer.optimizer.state_dict(),2)
        with pytest.raises(ValueError):trainer.load_state_dict(damaged)
        assert D.saved_state_digests(model.state_dict(),trainer.optimizer.state_dict(),2)==before
        if rank==0:torch.save(state,output)
        assert not torch.cuda.is_initialized()
    finally:dist.destroy_process_group()


def test_two_process_gradient_mean_full_adamw_and_rank_rng_resume(tmp_path):
    path=tmp_path/'state.pt';spawn(_worker,args=(free_port(),str(path)),nprocs=2,join=True)
    state=torch.load(path,weights_only=True)
    assert len(state['repair'])==len(state['optimizer']['state'])==161
    assert state['ddp']['replica_state']['0']==state['ddp']['replica_state']['1']
    assert not torch.equal(state['ddp']['rank_rng']['0']['torch_rng'],state['ddp']['rank_rng']['1']['torch_rng'])


def test_ddp_cursor_visits_each_case_once_per_epoch():
    binding=dict(fixed_cases=[f'case{i}' for i in range(8)],epochs=80,shuffle_seed=20261028)
    for epoch in range(80):
        pairs=[D.expected_cursor(binding,epoch*4+step,'fixed')['next_case_ids'] for step in range(4)]
        assert sorted(sum(pairs,[]))==binding['fixed_cases']
    assert D.expected_cursor(binding,320,'E080')['next_case_ids']==[]
