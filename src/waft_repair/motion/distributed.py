"""Original KITTI native-five-step supervision with two synchronized R replicas.

Only R is optimized. The complete WAFT is frozen but its five-step consumer
remains differentiable with respect to the repaired features. Each rank uses
one case's original native MoL; DDP averages gradients once before rank0 FP64
norm clipping and the unchanged AdamW update. No teacher features are used.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel



from . import training as BASE
from .training import (TrainConfig, learning_rate, epoch_order, model_inputs,
                       repair_forward, native_task_loss, full_valid_epe, require)
from .native import MotionNativeRepairSystem
from .finite import all_tensors_finite
from .model import MotionFeatureRepair

SCHEMA = 'waft-repair-ddp-native-training/v1'
STATE_KEYS = BASE.STATE_KEYS | {'ddp'}
GRADIENT_REDUCTION = 'mean-of-local-case-mean-gradients'
LOSS_REDUCTION = 'arithmetic-mean-of-2-local-native-five-step-MoL'
CLIP_POLICY = 'rank0-FP64-global-parameter-norm-of-DDP-mean-gradients-broadcast/v1'

NATIVE_OBJECTIVE = dict(kind='unchanged-KITTI-full-five-step-native-MoL/v1',gamma=.85,
    reduction='original sequence_mol_loss pooled native valid pixel/components',auxiliary_distillation_weight=0.)


def registered_model_contract(model,model_config):
    import inspect
    expected=dict(profile='motion_aligned_v1',fine_dim=32,s4_dim=96,s8_dim=128,
        s16_dim=192,s4_blocks=2,s8_blocks=3,s16_blocks=6,decoder_blocks=2,seed=20261028)
    require(type(model) is MotionFeatureRepair,'Registered full MotionFeatureRepair required')
    require(model_config==expected and asdict(model.config)==expected,'Exact FULL original architecture configuration')
    require(len(model.state_dict())==161 and len(list(model.parameters()))==161
        and sum(p.numel() for p in model.parameters())==3000576,'Registered FULL 3,000,576/161 ABI')
    require(model.configuration()['model_abi']=='paired-feature-repair/motion-aligned-v1'
        and getattr(model,'descriptor_transport_mode',None) is None,'Original registered transport required')
    return True


def expected_cursor(binding, completed_updates, label):
    ids=binding['fixed_cases'];size=len(ids);epochs=binding['epochs'];seed=binding['shuffle_seed']
    require(size>0 and size%2==0 and len(set(ids))==size,'Complete even DDP population')
    steps=size//2
    require(type(completed_updates) is int and 0<=completed_updates<=epochs*steps,'Bounded global DDP cursor')
    require(isinstance(label,str) and bool(label),'Checkpoint label')
    epoch,offset=divmod(completed_updates,steps);finished=completed_updates==epochs*steps
    order=[] if finished else epoch_order(ids,seed,epoch+1)
    next_ids=[] if finished else [ids[order[2*offset+rank]] for rank in range(2)]
    return dict(label=label,fixed_order=copy.deepcopy(ids),order_schema='seeded-epoch-python-random-ddp-pairs/v1',
        shuffle_seed=seed,next_epoch=epoch+1,next_offset=offset,next_case_ids=next_ids,
        next_case_index=2*offset,global_optimizer_updates=completed_updates,
        global_examples_seen=2*completed_updates,world_size=2)


def _cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(_cpu(v) for v in value)
    return copy.deepcopy(value)


def _same(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


def _rng(device):
    value=np.random.get_state()
    return dict(torch_rng=torch.get_rng_state().clone(),python_rng=random.getstate(),
        numpy_rng=[value[0],value[1].tolist(),*value[2:]],
        device_rng=torch.cuda.get_rng_state(device).cpu() if device.type=='cuda' else None)


def _tensor_digest_update(digest,name,value):
    value=value.detach().cpu().contiguous()
    digest.update(str(name).encode()+b'\0')
    digest.update(str((tuple(value.shape),value.dtype)).encode()+b'\0')
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def saved_state_digests(repair_state,optimizer_state,completed):
    """Canonical CPU/GPU-independent replica digest, also usable by an auditor."""
    repair=hashlib.sha256()
    for name,value in sorted(repair_state.items()):_tensor_digest_update(repair,name,value)
    opt=hashlib.sha256()
    opt.update(json.dumps(optimizer_state['param_groups'],sort_keys=True,separators=(',',':')).encode())
    for index,row in sorted(optimizer_state['state'].items()):
        for key,value in sorted(row.items()):_tensor_digest_update(opt,str(index)+':'+key,value)
    return dict(completed_updates=completed,repair_sha256=repair.hexdigest(),optimizer_sha256=opt.hexdigest())


def _state_digests(model,optimizer,completed):
    return saved_state_digests(model.state_dict(),optimizer.state_dict(),completed)


def registered_consumer_contract(consumer,model,core_binding):
    require(isinstance(consumer,MotionNativeRepairSystem) and consumer.strict_official
        and consumer.restorer is model,'Original strict MotionNativeRepairSystem with the same raw R required')
    require(consumer.checkpoint_refinement is True,'Original KITTI checkpoint_refinement=True required')
    consumer.validate()
    require(consumer.core_binding()['state_digest']==core_binding['state_digest'],
            'Consumer must match the bound full508 frozen WAFT')



class DistributedNativeTrainer(BASE.NativePilotTrainer):
    """Only R enters DDP/AdamW; the supplied complete WAFT remains frozen.

    ``checkpoint_state`` is collective: both ranks call it at the same cursor;
    only group rank0 receives the CPU checkpoint. No prepared input/GT is retained.
    """
    def __init__(self,model,ddp_model,consumer,config,*,model_config,binding,
                 core_binding,observation_binding,source_binding,process_group=None):
        require(dist.is_available() and dist.is_initialized(),'An explicitly initialized DDP process group required')
        require(isinstance(ddp_model,DistributedDataParallel) and ddp_model.module is model,
                'DDP must wrap exactly the original registered R, not the frozen core or another adapter')
        self.process_group=ddp_model.process_group if process_group is None else process_group
        require(self.process_group is ddp_model.process_group,'Optimizer and DDP process groups must agree')
        self.rank=dist.get_rank(self.process_group);self.world_size=dist.get_world_size(self.process_group)
        require(self.world_size==2 and self.rank in (0,1),'Exactly two synchronized replicas required')
        self.backend=dist.get_backend(self.process_group)
        require(self.backend in ('gloo','nccl'),'Explicit Gloo CPU validation or NCCL GPU execution required')
        self.ddp_model=ddp_model
        registered_consumer_contract(consumer,model,core_binding)
        registered_model_contract(model,model_config)
        require(config.stage=='native' and config.auxiliary_distillation_weight==0.,'Original KITTI native objective with auxiliary disabled')
        require(binding.get('native_objective')==NATIVE_OBJECTIVE
            and not {'teacher_index','teacher_method','feature_objective'}.intersection(binding),
            'Explicit original native objective; teacher/feature-fitting bindings are forbidden')
        require(binding.get('world_size')==2 and len(binding['fixed_cases'])%2==0 and config.updates==binding['epochs']*(len(binding['fixed_cases'])//2),
                'DDP budget must cover every case once per epoch, two cases per optimizer step')
        super().__init__(model,consumer,config,model_config=model_config,binding=binding,
            core_binding=core_binding,observation_binding=observation_binding,source_binding=source_binding)
        self.device=self.parameters[0].device
        require(all(p.device==self.device and p.dtype==torch.float32 for p in self.parameters),'One FP32 R device per rank')
        require((self.backend=='nccl')==(self.device.type=='cuda'),'NCCL GPU or Gloo CPU validation device contract')

    def _collective_require(self,condition,message):
        flag=torch.tensor(int(bool(condition)),device=self.device,dtype=torch.int64)
        dist.all_reduce(flag,op=dist.ReduceOp.MIN,group=self.process_group)
        require(int(flag)==1,message)

    def step(self,prepared,gt,valid,*,raw_gradient_callback=None):
        require(self.completed_updates<self.config.updates,'Fixed DDP update budget exhausted')
        require(isinstance(gt,torch.Tensor) and isinstance(valid,torch.Tensor)
            and not gt.requires_grad and gt.grad_fn is None and not valid.requires_grad
            and valid.dtype==torch.bool and gt.dtype==torch.float32
            and gt.device==valid.device==self.device and gt.shape==(1,2,*prepared['native_hw'])
            and valid.shape==(1,*prepared['native_hw']),'Detached local native GT/mask, separate from R inputs, required')
        require(prepared['original'].shape[0]==1,'Exactly one complete paired case per rank required')
        started=time.monotonic();self.ddp_model.train();self.optimizer.zero_grad(set_to_none=True)
        features=repair_forward(self.ddp_model,prepared)
        output=self.consumer.solve_features(features,native_hw=prepared['native_hw'],padding=prepared['padding'])
        loss=native_task_loss(output,gt,valid)
        epe=full_valid_epe(output['flow'][-1].detach(),gt,valid)
        self._collective_require(bool(torch.isfinite(loss)),'Finite original native-five-step MoL required on both ranks')
        forward=time.monotonic()
        # DDP performs the one and only mean across the two local gradients.
        loss.backward()
        grads=[p.grad for p in self.parameters]
        self._collective_require(all(g is not None for g in grads) and all_tensors_finite(g for g in grads if g is not None),
                                 'All161 finite synchronized averaged gradients required on both ranks')
        norm=(torch.stack([g.detach().double().square().sum() for g in grads]).sum().sqrt()
              if self.rank==0 else torch.zeros((),device=self.device,dtype=torch.float64))
        source=dist.get_global_rank(self.process_group,0)
        dist.broadcast(norm,src=source,group=self.process_group)
        self._collective_require(bool(torch.isfinite(norm)) and float(norm)>0,'Finite positive FP64 DDP mean-gradient norm, exactly as original KITTI')
        factor=min(1.,self.config.clip_norm/(float(norm)+1e-12))
        callback_error=None
        if raw_gradient_callback is not None:
            try:
                # Param hooks observe pre-DDP local gradients; this callback is
                # deliberately after reduction and before clipping instead.
                raw_gradient_callback({name:p.grad.detach().clone() for name,p in self.model.named_parameters()})
            except Exception as error:callback_error=error
        self._collective_require(callback_error is None,'A raw averaged-gradient witness callback failed on a rank')
        if factor<1:
            for value in grads:value.mul_(factor)
        backward=time.monotonic();rate=self.optimizer.param_groups[0]['lr']
        self.optimizer.step();self.completed_updates+=1
        self.optimizer.param_groups[0]['lr']=learning_rate(self.config,self.completed_updates+1)
        self._collective_require(all_tensors_finite(self.parameters) and all_tensors_finite(
            value for row in self.optimizer.state.values() for value in row.values() if isinstance(value,torch.Tensor)),
            'Finite complete synchronized post-AdamW state')
        self.consumer.validate()
        scale=prepared['sigma'].reshape(-1,1,64,1,1)
        correction=float(((features.detach()-prepared['original'])/scale).square().mean().sqrt())
        local=torch.tensor([float(loss.detach()),correction,float(epe)],device=self.device,dtype=torch.float64)
        rank_stats=[torch.empty_like(local) for _ in range(2)]
        dist.all_gather(rank_stats,local,group=self.process_group)
        values=[row.cpu().tolist() for row in rank_stats];rank_losses=[v[0] for v in values]
        metrics=dict(stage='native_ddp',completed_updates=self.completed_updates,
            loss=math.fsum(rank_losses)/2,local_loss=float(loss.detach()),raw_loss=float(loss.detach()),
            backward_loss=float(loss.detach()),rank_losses=rank_losses,rank=self.rank,world_size=2,
            global_batch_size=2,local_batch_size=1,global_examples_seen=2*self.completed_updates,
            gradient_reduction=GRADIENT_REDUCTION,clip_policy=CLIP_POLICY,
            learning_rate_used=rate,gradient_norm_before_clip=float(norm),gradient_clip_scale=factor,
            normalized_correction_RMS=math.sqrt(math.fsum(v[1]**2 for v in values)/2),
            local_normalized_correction_RMS=correction,rank_normalized_correction_RMS=[v[1] for v in values],
            full_valid_final_EPE=math.fsum(v[2] for v in values)/2,
            local_full_valid_final_EPE=float(epe),rank_full_valid_final_EPE=[v[2] for v in values],
            training_native_G5=1,global_training_native_G5=2,training_teacher_feature_reads=0,
            measurement_phase='Original five-step native MoL/EPE before optimizer.step; reported loss/EPE are two local case means. Backward uses unscaled local native loss; DDP averages gradients once. GT never enters R inputs.',
            timing_seconds=dict(forward_loss=forward-started,backward_and_grad_check=backward-forward,
                optimizer_and_checks=time.monotonic()-backward))
        del features,output,loss,epe,grads,norm,rank_stats,local
        return metrics

    def checkpoint_state(self,*,cursor):
        """Collect rank RNG + replica hashes; return a complete CPU state on r0."""
        require(cursor==expected_cursor(self.binding,self.completed_updates,cursor.get('label')),'Exact DDP checkpoint cursor')
        local=dict(rank=self.rank,rng=_rng(self.device),replica=_state_digests(self.model,self.optimizer,self.completed_updates))
        gathered=[None,None]
        dist.all_gather_object(gathered,local,group=self.process_group)
        require([row['rank'] for row in gathered]==[0,1],'All rank checkpoint evidence')
        require(gathered[0]['replica']==gathered[1]['replica'],'Both repair and complete AdamW replica states must be bit-identical')
        if self.rank!=0:return None
        rank_rng={str(row['rank']):row['rng'] for row in gathered}
        replicas={str(row['rank']):row['replica'] for row in gathered}
        return self.state_dict(cursor=cursor,rank_rng_states=rank_rng,replica_states=replicas)

    def state_dict(self,*,cursor,rank_rng_states,replica_states):
        """Noncollective builder, for rank0 after collective checkpoint_state."""
        require(self.rank==0,'Only group rank0 builds a persistent checkpoint')
        require(set(rank_rng_states)==set(replica_states)=={'0','1'},'Every rank RNG and replica state required')
        require(replica_states['0']==replica_states['1'],'Bit-identical averaged-gradient AdamW replicas')
        require(cursor==expected_cursor(self.binding,self.completed_updates,cursor.get('label')),'Bound DDP cursor')
        rng=rank_rng_states['0']
        return dict(schema=SCHEMA,config=asdict(self.config),model_config=copy.deepcopy(self.model_config),
            repair=_cpu(self.model.state_dict()),optimizer=_cpu(self.optimizer.state_dict()),
            scheduler=dict(schema='native-linear-warmup-cosine/v1',last_epoch=self.completed_updates,
                next_learning_rate=learning_rate(self.config,self.completed_updates+1)),
            completed_updates=self.completed_updates,binding=copy.deepcopy(self.binding),cursor=copy.deepcopy(cursor),
            verified_implementation_binding=copy.deepcopy(self.source_binding),
            verified_core_binding=copy.deepcopy(self.core_binding),verified_observation_binding=copy.deepcopy(self.observation_binding),
            **_cpu(rng),ddp=dict(world_size=2,backend=self.backend,gradient_reduction=GRADIENT_REDUCTION,
                loss_reduction=LOSS_REDUCTION,clip_policy=CLIP_POLICY,rank_rng=_cpu(rank_rng_states),
                replica_state=copy.deepcopy(replica_states),replica_state_bit_exact=True))

    def validate_checkpoint(self,state):
        require(isinstance(state,dict) and set(state)==STATE_KEYS and state['schema']==SCHEMA,'Complete DDP checkpoint schema')
        require(state['config']==asdict(self.config) and state['model_config']==self.model_config
            and state['binding']==self.binding and state['verified_implementation_binding']==self.source_binding
            and state['verified_core_binding']==self.core_binding
            and state['verified_observation_binding']==self.observation_binding,'Complete DDP source/data/model bindings')
        n=state['completed_updates'];require(type(n) is int and 0<=n<=self.config.updates,'Global DDP optimizer cursor')
        require(state['cursor']==expected_cursor(self.binding,n,state['cursor'].get('label')),'DDP epoch/case-pair cursor')
        require(state['scheduler']==dict(schema='native-linear-warmup-cosine/v1',last_epoch=n,
            next_learning_rate=learning_rate(self.config,n+1)),'DDP schedule cursor')
        current=self.model.state_dict();saved=state['repair']
        require(saved.keys()==current.keys() and len(saved)==161,'All161 original raw repair state names')
        for name,ref in current.items():
            value=saved[name]
            require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.shape==ref.shape
                and value.dtype==ref.dtype and not value.requires_grad and value.grad_fn is None
                and bool(torch.isfinite(value).all()),'Complete finite saved raw repair tensor '+name)
        optimizer=state['optimizer']
        require(set(optimizer)=={'state','param_groups'} and len(optimizer['param_groups'])==1,'Complete one-group AdamW')
        expected=copy.deepcopy(self.optimizer.state_dict()['param_groups'][0]);expected['lr']=learning_rate(self.config,n+1)
        require(optimizer['param_groups'][0]==expected,'DDP AdamW settings and parameter order')
        require(set(optimizer['state'])==(set(range(161)) if n else set()),'Every replica AdamW moment cursor')
        for index,row in optimizer['state'].items():
            require(set(row)=={'step','exp_avg','exp_avg_sq'} and float(row['step'])==n,'Complete global AdamW steps')
            for key in ('exp_avg','exp_avg_sq'):
                value=row[key]
                require(value.device.type=='cpu' and value.shape==self.parameters[index].shape
                    and value.dtype==self.parameters[index].dtype and not value.requires_grad
                    and bool(torch.isfinite(value).all()),'Complete finite DDP AdamW moment')
            require(bool((row['exp_avg_sq']>=0).all()),'Nonnegative DDP AdamW variance')
        info=state['ddp']
        require(set(info)=={'world_size','backend','gradient_reduction','loss_reduction','clip_policy','rank_rng','replica_state','replica_state_bit_exact'}
            and info['world_size']==2 and info['backend']==self.backend and info['gradient_reduction']==GRADIENT_REDUCTION
            and info['loss_reduction']==LOSS_REDUCTION and info['clip_policy']==CLIP_POLICY,'Exact DDP numerical protocol')
        require(set(info['rank_rng'])==set(info['replica_state'])=={'0','1'} and info['replica_state_bit_exact'] is True
            and info['replica_state']['0']==info['replica_state']['1'],'Complete bit-identical replica proof')
        require(info['replica_state']['0']==saved_state_digests(saved,optimizer,n),
                'Replica proof hashes must match the actual saved repair/AdamW tensors')
        for rank,rng in info['rank_rng'].items():
            require(set(rng)=={'torch_rng','python_rng','numpy_rng','device_rng'},'Complete per-rank RNG keys')
            torch.Generator(device='cpu').set_state(rng['torch_rng'])
            random.Random().setstate(rng['python_rng'])
            value=rng['numpy_rng'];np.random.RandomState(0).set_state((value[0],np.asarray(value[1],dtype=np.uint32),*value[2:]))
            require((rng['device_rng'] is None)==(self.device.type=='cpu'),'Per-rank RNG device contract')
            if rng['device_rng'] is not None:
                require(rng['device_rng'].device.type=='cpu' and rng['device_rng'].dtype==torch.uint8
                    and rng['device_rng'].ndim==1 and rng['device_rng'].numel()>0,'Raw per-rank CUDA RNG bytes')
        require(_same(state['torch_rng'],info['rank_rng']['0']['torch_rng'])
            and state['python_rng']==info['rank_rng']['0']['python_rng']
            and state['numpy_rng']==info['rank_rng']['0']['numpy_rng'],'Top RNG fields mirror rank0 only')
        top_device=state['device_rng'];rank0_device=info['rank_rng']['0']['device_rng']
        require((top_device is None and rank0_device is None)
            or (isinstance(top_device,torch.Tensor) and isinstance(rank0_device,torch.Tensor)
                and _same(top_device,rank0_device)),'Top device RNG mirrors rank0')
        return True

    def load_state_dict(self,state):
        """Same-plan recovery/reset installs the selected rank's own RNG."""
        self.validate_checkpoint(state)
        self.model.load_state_dict(state['repair'],strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        self.completed_updates=state['completed_updates']
        rng=state['ddp']['rank_rng'][str(self.rank)]
        torch.set_rng_state(rng['torch_rng']);random.setstate(rng['python_rng'])
        value=rng['numpy_rng'];np.random.set_state((value[0],np.asarray(value[1],dtype=np.uint32),*value[2:]))
        if rng['device_rng'] is not None:torch.cuda.set_rng_state(rng['device_rng'],self.device)


__all__=['DistributedNativeTrainer','TrainConfig','SCHEMA','STATE_KEYS','expected_cursor',
         'epoch_order','learning_rate','model_inputs','repair_forward','native_task_loss','full_valid_epe','NATIVE_OBJECTIVE',
         'GRADIENT_REDUCTION','LOSS_REDUCTION','CLIP_POLICY','saved_state_digests']
