"""The native five-update optical-flow trainer used by the current KITTI run.

WAFT is a frozen, differentiable consumer: GT enters only ``step`` and the
original five-term mixture-of-Laplace objective. The repair's inputs are an
explicit label-free allowlist. This packaged trainer preserves the operational
update arithmetic; CPU checkpoint copies and prevalidated transactional resume
are packaging safeguards, not a new training method.

A source binding is intentionally strict. Moving an operational checkpoint into
a new checkout is a weights-import operation, not an exact operational resume.
No GPU, model construction, filesystem write, or weight download occurs here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .native import native_task_loss
from .finite import all_tensors_finite


SCHEMA = 'waft-repair-redesign-training/v1'
STATE_KEYS = {'schema','config','model_config','repair','optimizer','scheduler',
    'completed_updates','binding','cursor','verified_implementation_binding',
    'verified_core_binding','verified_observation_binding','torch_rng','python_rng',
    'numpy_rng','device_rng'}


def require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class TrainConfig:
    updates: int = 5120
    peak_lr: float = 3e-4
    weight_decay: float = 0.
    epsilon: float = 1e-8
    clip_norm: float = 1.
    warmup_updates: int = 160
    minimum_lr_fraction: float = .1
    stage: str = 'native'
    auxiliary_distillation_weight: float = 0.

    def __post_init__(self):
        require(type(self.updates) is int and self.updates>0 and self.stage=='native', 'Native update budget required')
        require(type(self.warmup_updates) is int and 0<self.warmup_updates<self.updates, 'Explicit finite warmup')
        require(all(math.isfinite(v) for v in (self.peak_lr,self.weight_decay,self.epsilon,
            self.clip_norm,self.minimum_lr_fraction,self.auxiliary_distillation_weight)), 'Finite optimizer contract')
        require(self.peak_lr>0 and self.epsilon>0 and self.clip_norm>0 and self.weight_decay>=0
            and 0<self.minimum_lr_fraction<=1 and self.auxiliary_distillation_weight==0.,
            'Explicit AdamW settings; teacher auxiliary disabled for full training')


def learning_rate(config, next_update):
    require(type(next_update) is int and 1<=next_update<=config.updates+1, 'One-based update rate cursor')
    if next_update<=config.warmup_updates:
        return config.peak_lr*next_update/config.warmup_updates
    fraction=min(1.,(next_update-config.warmup_updates)/(config.updates-config.warmup_updates))
    return config.peak_lr*(config.minimum_lr_fraction+(1-config.minimum_lr_fraction)*.5*(1+math.cos(math.pi*fraction)))


def epoch_order(case_ids, seed, epoch):
    """One-based independently seeded permutations; never consumes global RNG."""
    require(type(seed) is int and type(epoch) is int and epoch >= 1, 'Explicit shuffle seed/epoch')
    require(len(case_ids)>0 and len(set(case_ids))==len(case_ids),'Unique training population')
    result=list(range(len(case_ids)))
    random.Random(seed+epoch).shuffle(result)
    return result


def expected_cursor(binding, completed_updates, label):
    ids=binding['fixed_cases'];size=len(ids);epochs=binding['epochs'];seed=binding['shuffle_seed']
    require(type(completed_updates) is int and 0<=completed_updates<=epochs*size,'Bounded epoch cursor')
    require(isinstance(label,str) and bool(label),'Checkpoint label')
    epoch,offset=divmod(completed_updates,size);finished=completed_updates==epochs*size
    order=[] if finished else epoch_order(ids,seed,epoch+1)
    return dict(label=label,fixed_order=ids,order_schema='seeded-epoch-python-random/v1',
        shuffle_seed=seed,next_epoch=epoch+1,next_offset=offset,
        next_case_id=None if finished else ids[order[offset]],
        next_case_index=offset,global_optimizer_updates=completed_updates)


def implementation_binding(paths):
    return {str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths)}


def model_inputs(prepared):
    """An allowlist copy, so packet/GT/teacher/quality metadata cannot reach R."""
    keys={'original','sigma','observations','baseline_flow','baseline_info','native_hw','padding'}
    require(set(prepared)==keys or set(prepared)==keys|{'motion_prior'}, 'Exact GT-free model-input keys')
    result={k:prepared[k] for k in keys}
    if 'motion_prior' in prepared:result['motion_prior']=prepared['motion_prior']
    for k in ('original','sigma','baseline_flow','baseline_info'):
        v=result[k]
        require(isinstance(v,torch.Tensor) and not v.requires_grad and v.grad_fn is None, 'Detached input '+k)
    require(set(result['observations'])=={'cnn_pre_s2','cnn_pre_s4','dav2_block6','dav2_block12'}, 'Original four taps only')
    require(all(not v.requires_grad and v.grad_fn is None for v in result['observations'].values()), 'Detached frozen taps')
    return result


def repair_forward(model, prepared, *, return_aux=False):
    values=model_inputs(prepared)
    prior=values.pop('motion_prior',None)
    kwargs=dict(native_hw=values['native_hw'],padding=values['padding'],return_aux=return_aux)
    if prior is not None:kwargs['motion_prior']=prior
    if prior is not None:
        return model(values['original'],values['sigma'],values['observations'],**kwargs)
    return model(values['original'],values['sigma'],values['observations'],
        values['baseline_flow'],values['baseline_info'],**kwargs)


def full_valid_epe(flow, gt, valid):
    require(valid.dtype==torch.bool and valid.shape==(flow.shape[0],*flow.shape[-2:])
        and bool(valid.any()) and flow.shape==gt.shape, 'Original full valid population')
    error=(flow-gt).permute(0,2,3,1)[valid]
    return torch.linalg.vector_norm(error.double(),dim=-1).mean()


class NativePilotTrainer:
    def __init__(self, model, consumer, config, *, model_config, binding,
                 core_binding, observation_binding, source_binding):
        self.model,self.consumer,self.config=model,consumer,config
        self.model_config=copy.deepcopy(model_config);self.binding=copy.deepcopy(binding)
        self.core_binding=copy.deepcopy(core_binding);self.observation_binding=copy.deepcopy(observation_binding)
        self.source_binding=copy.deepcopy(source_binding)
        self.parameters=list(model.parameters());require(self.parameters,'Trainable repair required')
        require(isinstance(config,TrainConfig), 'Explicit native training configuration')
        require(isinstance(self.binding,dict) and {'fixed_cases','epochs','shuffle_seed'} <= self.binding.keys(),
            'Fixed population, epoch count, and shuffle seed binding required')
        require(type(self.binding['epochs']) is int and self.binding['epochs'] > 0
            and len(self.binding['fixed_cases'])*self.binding['epochs']==config.updates,
            'Training budget must equal complete fixed epochs')
        epoch_order(self.binding['fixed_cases'],self.binding['shuffle_seed'],1)
        require(all(p.requires_grad for p in self.parameters),'Only trainable repair optimizer parameters')
        self.optimizer=torch.optim.AdamW(self.parameters,lr=learning_rate(config,1),
            eps=config.epsilon,weight_decay=config.weight_decay,foreach=False)
        self.completed_updates=0

    def step(self, prepared, gt, valid):
        require(self.completed_updates<self.config.updates,'Fixed budget exhausted; no automatic extension')
        require(not gt.requires_grad and gt.grad_fn is None and not valid.requires_grad,'Detached labels')
        started=time.monotonic();self.model.train();self.optimizer.zero_grad(set_to_none=True)
        features=repair_forward(self.model,prepared)
        output=self.consumer.solve_features(features,native_hw=prepared['native_hw'],padding=prepared['padding'])
        loss=native_task_loss(output,gt,valid)
        epe=full_valid_epe(output['flow'][-1].detach(),gt,valid)
        require(bool(torch.isfinite(loss)),'Finite original native5 MoL')
        forward=time.monotonic();loss.backward()
        grads=[p.grad for p in self.parameters if p.grad is not None]
        require(bool(grads) and all_tensors_finite(grads),'Finite nonempty repair gradients')
        norm=torch.stack([v.detach().double().square().sum() for v in grads]).sum().sqrt()
        require(bool(torch.isfinite(norm)) and float(norm)>0,'Finite positive native repair gradient')
        factor=min(1.,self.config.clip_norm/(float(norm)+1e-12))
        if factor<1:
            for value in grads:value.mul_(factor)
        backward=time.monotonic();rate=self.optimizer.param_groups[0]['lr']
        self.optimizer.step();self.completed_updates+=1
        self.optimizer.param_groups[0]['lr']=learning_rate(self.config,self.completed_updates+1)
        require(all_tensors_finite(self.parameters),'Finite post-update repair parameters')
        require(all_tensors_finite(v for row in self.optimizer.state.values() for v in row.values()
            if isinstance(v,torch.Tensor)),'Finite AdamW state')
        self.consumer.validate()
        scale=prepared['sigma'].reshape(-1,1,64,1,1)
        metrics=dict(stage='native',completed_updates=self.completed_updates,loss=float(loss.detach()),
            raw_loss=float(loss.detach()),backward_loss=float(loss.detach()),full_valid_final_EPE=float(epe),
            learning_rate_used=rate,gradient_norm_before_clip=float(norm),gradient_clip_scale=factor,
            normalized_correction_RMS=float(((features.detach()-prepared['original'])/scale).square().mean().sqrt()),
            measurement_phase='Loss/EPE/delta from forward before this optimizer.step; completed_updates is after step.',
            timing_seconds=dict(forward_loss=forward-started,backward_and_grad_check=backward-forward,
                optimizer_and_checks=time.monotonic()-backward))
        del features,output,loss,grads,norm,epe
        return metrics

    def state_dict(self, *, cursor):
        require(cursor==expected_cursor(self.binding,self.completed_updates,cursor.get('label')),
            'Save cursor must match the next shuffled input')
        r=np.random.get_state();device=self.parameters[0].device
        return dict(schema=SCHEMA,config=asdict(self.config),model_config=copy.deepcopy(self.model_config),
            repair=cpu_tree(self.model.state_dict()),optimizer=cpu_tree(self.optimizer.state_dict()),
            scheduler=dict(schema='native-linear-warmup-cosine/v1',last_epoch=self.completed_updates,
                next_learning_rate=learning_rate(self.config,self.completed_updates+1)),
            completed_updates=self.completed_updates,binding=copy.deepcopy(self.binding),cursor=copy.deepcopy(cursor),
            verified_implementation_binding=copy.deepcopy(self.source_binding),
            verified_core_binding=copy.deepcopy(self.core_binding),verified_observation_binding=copy.deepcopy(self.observation_binding),
            torch_rng=torch.get_rng_state().clone(),python_rng=random.getstate(),
            numpy_rng=[r[0],r[1].tolist(),*r[2:]],
            device_rng=torch.cuda.get_rng_state(device).cpu() if device.type=='cuda' else None)

    def validate_checkpoint(self,state):
        require(isinstance(state,dict) and set(state)==STATE_KEYS and state['schema']==SCHEMA,'Complete redesign state')
        require(state['config']==asdict(self.config) and state['model_config']==self.model_config
            and state['binding']==self.binding and state['verified_implementation_binding']==self.source_binding
            and state['verified_core_binding']==self.core_binding
            and state['verified_observation_binding']==self.observation_binding,'State implementation/data/model binding')
        n=state['completed_updates'];require(type(n) is int and 0<=n<=self.config.updates,'Checkpoint update cursor')
        require(state['scheduler']==dict(schema='native-linear-warmup-cosine/v1',last_epoch=n,
            next_learning_rate=learning_rate(self.config,n+1)),'Checkpoint schedule cursor')
        cursor=state['cursor']
        require(cursor==expected_cursor(self.binding,n,cursor.get('label')), 'Checkpoint shuffled epoch input cursor')
        current=self.model.state_dict();saved=state['repair']
        require(saved.keys()==current.keys(),'Repair state names')
        for key,ref in current.items():
            value=saved[key];require(isinstance(value,torch.Tensor) and value.device.type=='cpu'
                and value.shape==ref.shape and value.dtype==ref.dtype and not value.requires_grad
                and bool(torch.isfinite(value).all()),'Saved repair tensor '+key)
        opt=state['optimizer'];require(set(opt)=={'state','param_groups'} and len(opt['param_groups'])==1,'Full AdamW state')
        group=opt['param_groups'][0]
        expected=copy.deepcopy(self.optimizer.state_dict()['param_groups'][0]);expected['lr']=learning_rate(self.config,n+1)
        require(group==expected,'AdamW settings/schedule mismatch')
        require(set(opt['state'])==set(range(len(self.parameters))) if n else not opt['state'],'All repair AdamW moments/cursor')
        for i,row in opt['state'].items():
            require(set(row)=={'step','exp_avg','exp_avg_sq'},'AdamW full moment keys')
            for name in ('exp_avg','exp_avg_sq'):
                value=row[name];require(value.device.type=='cpu' and value.shape==self.parameters[i].shape
                    and value.dtype==self.parameters[i].dtype and not value.requires_grad
                    and bool(torch.isfinite(value).all()),'Finite complete AdamW moment')
            require(bool((row['exp_avg_sq']>=0).all()) and float(row['step'])==n,'AdamW variance/update cursor')
        value=state['torch_rng']
        require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.uint8
            and value.ndim==1 and value.numel()>0 and not value.requires_grad, 'Complete saved CPU RNG')
        torch.Generator(device='cpu').set_state(value)
        random.Random().setstate(state['python_rng'])
        r=state['numpy_rng'];np.random.RandomState(0).set_state((r[0],np.asarray(r[1],dtype=np.uint32),*r[2:]))
        require((state['device_rng'] is None)==(self.parameters[0].device.type=='cpu'),'Checkpoint RNG device contract')
        if state['device_rng'] is not None:
            value=state['device_rng']
            require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.uint8
                and value.ndim==1 and value.numel()>0 and not value.requires_grad,'Complete saved CUDA RNG')
            torch.Generator(device=self.parameters[0].device).set_state(value)
        return True

    def _apply_state(self,state):
        self.model.load_state_dict(state['repair'],strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        self.completed_updates=state['completed_updates']
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng']);r=state['numpy_rng']
        np.random.set_state((r[0],np.asarray(r[1],dtype=np.uint32),*r[2:]))
        if state['device_rng'] is not None:
            torch.cuda.set_rng_state(state['device_rng'],self.parameters[0].device)

    def load_state_dict(self,state):
        """Validate without mutations, then commit; roll back ordinary load errors.

        Exact numerical continuation is tested for single-thread CPU execution.
        Restoring GPU RNG does not guarantee deterministic CUDA kernel replay.
        """
        self.validate_checkpoint(state)
        previous=self.state_dict(cursor=expected_cursor(self.binding,self.completed_updates,'rollback'))
        try:
            self._apply_state(state)
        except Exception:
            self._apply_state(previous)
            raise


# The historical name is retained so existing native-run recipes remain clear.
NativeTrainer = NativePilotTrainer


def cpu_tree(value):
    """Independent CPU-owned checkpoint tensors, including AdamW moments."""
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {key:cpu_tree(item) for key,item in value.items()}
    if isinstance(value,list):
        return [cpu_tree(item) for item in value]
    if isinstance(value,tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)
