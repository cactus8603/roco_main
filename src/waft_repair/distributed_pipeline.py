"""Two-rank native training from RGB, with complete collective checkpoints.

Launch explicitly with torchrun. This module creates no background services and
does not import operational checkpoints into a different source/data binding.
"""
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .backend import SOURCE_COMMIT,load_rgb_pair
from .memory_hygiene import case_boundary,advise_readonly
from .motion.data import read_spec,read_labels
from .motion.distributed import DistributedNativeTrainer,TrainConfig,epoch_order,expected_cursor,NATIVE_OBJECTIVE
from .motion.io import atomic_json,atomic_torch,check_descriptor,descriptor,load_tensor,output_lock,require,sha256
from .motion.checkpoint import export_model_checkpoint
from .native_pipeline import _package_sources,_reconcile_history,_room,_system,_flow_metrics


def _sources():
    from .motion import distributed,sintel
    result=_package_sources()
    result.update({module.__name__:sha256(module.__file__) for module in (distributed,sintel)})
    result[__name__]=sha256(__file__)
    return result


def _raw_case(system,row,device):
    """Rebuild original observations at each visit; no feature/teacher files."""
    for key in ('image1','image2','ground_truth'):check_descriptor(row[key],Path('/'))
    pair=load_rgb_pair(row['image1']['path'],row['image2']['path']).to(device)
    system.eval()
    with torch.no_grad():observed=system.prepare_pair(pair[:,0],pair[:,1])
    prepared={key:getattr(observed,key) for key in ('original','sigma','observations',
        'baseline_flow','baseline_info','native_hw','padding','motion_prior')}
    baseline=observed.baseline_output
    # Labels are read after the full RGB-only model inputs have been constructed.
    gt,valid=read_labels(row['ground_truth']['path'],row['ground_truth_format'])
    require(tuple(gt.shape[-2:])==prepared['native_hw'],'Original RGB/GT geometry differs')
    del observed,pair
    return prepared,gt.to(device),valid.to(device),baseline


def _boundary(output,rank,row,completed):
    return case_boundary(torch,paths=[row[k]['path'] for k in ('image1','image2','ground_truth')],
        log_path=output/f'memory-rank{rank}.jsonl',completed_updates=completed)


def _evaluate(system,rows,output,rank,device,label,completed,*,retain=False):
    """Both ranks process disjoint eval rows; write bounded per-case records."""
    from .motion.training import repair_forward,native_task_loss
    root=output/'evaluations'/label;root.mkdir(parents=True,exist_ok=True)
    with (root/f'rank{rank}.jsonl').open('x') as stream:
        for index in range(rank,len(rows),2):
            row=rows[index];prepared,gt,valid,baseline=_raw_case(system,row,device)
            with torch.no_grad():
                features=repair_forward(system.restorer,prepared)
                predicted=system.solve_features(features,native_hw=prepared['native_hw'],padding=prepared['padding'])
                nll=float(native_task_loss(predicted,gt,valid))
            record=dict(id=row['id'],split=row['split'],native5_MoL=nll,
                all5=[_flow_metrics(flow,gt,valid) for flow in predicted['flow']],
                baseline_all5=[_flow_metrics(flow,gt,valid) for flow in baseline['flow']])
            if retain:
                target=root/f'{index:06d}.pt'
                atomic_torch(target,dict(id=row['id'],flow=predicted['flow'],info=predicted['info']))
                record['artifact']=descriptor(target,relative_to=root)
                advise_readonly(target)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            del prepared,gt,valid,baseline,features,predicted
            _boundary(output,rank,row,completed)
    dist.barrier()


def train(spec,output,waft_source,waft_checkpoint,*,epochs=80,seed=20261028,
          device='cuda',warmup_updates=796,resume=None,checkpoint_every=256,
          stop_after_updates=None,activation_offload='cpu',reserve_gib=20,memory_fraction=None):
    """One original case/rank; one DDP gradient mean; AdamW and full rank RNG."""
    require(int(os.environ.get('WORLD_SIZE','0'))==2,'Launch with torchrun --nproc-per-node=2')
    require(device in ('cpu','cuda'),'DDP device is cpu or cuda; local rank selects the GPU')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    require(rank in (0,1) and local in (0,1),'One node with exactly two ranks required')
    require(type(epochs) is int and epochs>0 and type(checkpoint_every) is int and checkpoint_every>0,
            'Positive fixed epoch/checkpoint counts required')
    require(activation_offload in ('cpu','none'),'Activation offload must be cpu or none')
    if device=='cuda':torch.cuda.set_device(local)
    target=torch.device('cuda',local) if device=='cuda' else torch.device('cpu')
    initialized=False
    try:
        dist.init_process_group('nccl' if device=='cuda' else 'gloo',timeout=timedelta(hours=2))
        initialized=True
        output=Path(output).resolve()
        if rank==0:
            if resume:require(output.is_dir() and (output/'run.json').is_file(),'Resume existing run directory')
            else:output.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        # Rank0 owns the output lock; rank1 never opens/unlinks it.
        with output_lock(output) if rank==0 else nullcontext():
            _room(output,reserve_gib)
            samples=read_spec(spec);rows=[row for row in samples if row['split']=='train']
            ids=[row['id'] for row in rows]
            require(bool(ids) and len(ids)%2==0,'A nonempty even TRAIN population is required')
            budget=epochs*(len(rows)//2);config=TrainConfig(updates=budget,warmup_updates=warmup_updates)
            limit=budget if stop_after_updates is None else stop_after_updates
            require(type(limit) is int and 0<limit<=budget,'Stop cursor is within the unchanged full schedule')
            torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
            system,metadata=_system(waft_source,waft_checkpoint,target,memory_fraction=memory_fraction)
            model=system.restorer
            require(model.config.seed==seed,'Use the registered initialization seed20261028')
            ddp=DistributedDataParallel(model,device_ids=[local] if device=='cuda' else None,
                broadcast_buffers=False,find_unused_parameters=False)
            binding=dict(fixed_cases=ids,epochs=epochs,shuffle_seed=seed,world_size=2,
                native_objective=NATIVE_OBJECTIVE,spec_sha256=sha256(spec),
                raw_sources={row['id']:{k:row[k] for k in ('image1','image2','ground_truth')} for row in samples},
                schema='native-rgb-ddp-epochs/v1')
            core=dict(state_digest=metadata['state_sha256'],source_commit=SOURCE_COMMIT,torch_version=str(torch.__version__))
            trainer=DistributedNativeTrainer(model,ddp,system,config,model_config=asdict(model.config),
                binding=binding,core_binding=core,observation_binding=dict(RGB_each_visit=True),source_binding=_sources())
            run=dict(schema='waft-native-ddp-run/v1',binding=binding,training_config=asdict(config),
                source_binding=_sources(),core_binding=core,activation_offload=activation_offload,
                checkpoint_every=checkpoint_every,CPU_threads=torch.get_num_threads(),
                CPU_interop_threads=torch.get_num_interop_threads(),fixed_final_epoch=epochs,
                automatic_checkpoint_selection=False)
            if resume:
                require(json.loads((output/'run.json').read_text())==run,'Run metadata/settings changed during resume')
                trainer.load_state_dict(load_tensor(resume))
                _reconcile_history(output,trainer.completed_updates,
                    'history.jsonl' if rank==0 else 'history-rank1.jsonl')
            elif rank==0:atomic_json(output/'run.json',run)
            require(limit>=trainer.completed_updates,'Stop cursor precedes the committed checkpoint')
            dist.barrier()

            def save(label):
                state=trainer.checkpoint_state(cursor=expected_cursor(binding,trainer.completed_updates,label))
                if rank==0:
                    trainer.validate_checkpoint(state)
                    atomic_torch(output/'checkpoint.pt',state)
                    atomic_torch(output/'model.pt',export_model_checkpoint(model,trainer.completed_updates,
                        metadata=dict(training_method='native5-MoL-DDP',spec_sha256=binding['spec_sha256']),
                        waft_checkpoint_sha256=metadata['waft_checkpoint_sha256']))
                    if label.startswith('E'):atomic_torch(output/'checkpoints'/(label+'.pt'),state)
                    atomic_json(output/'progress.json',dict(completed_updates=trainer.completed_updates,
                        total_updates=budget,cursor=state['cursor'],status='COMPLETE' if trainer.completed_updates==budget else 'TRAINING'))
                dist.barrier()
                case_boundary(torch)

            if not resume:
                _evaluate(system,samples,output,rank,target,'init',0,retain=True)
                save('initial')
            with (output/('history.jsonl' if rank==0 else 'history-rank1.jsonl')).open('a') as log:
                while trainer.completed_updates<limit:
                    _room(output,reserve_gib)
                    epoch,offset=divmod(trainer.completed_updates,len(rows)//2)
                    order=epoch_order(ids,seed,epoch+1);row=rows[order[2*offset+rank]]
                    prepared,gt,valid,baseline=_raw_case(system,row,target)
                    del baseline
                    context=torch.autograd.graph.save_on_cpu(pin_memory=False) if device=='cuda' and activation_offload=='cpu' else nullcontext()
                    with context:metrics=trainer.step(prepared,gt,valid)
                    metrics.update(id=row['id'],epoch=epoch+1,offset_in_epoch=offset)
                    log.write(json.dumps(metrics,allow_nan=False)+'\n');log.flush()
                    del prepared,gt,valid
                    n=trainer.completed_updates;_boundary(output,rank,row,n)
                    if n%(len(rows)//2)==0 and (n//(len(rows)//2) in (1,4,8,16,32) or n==budget):
                        label=f'E{n//(len(rows)//2):03d}'
                        selected=samples if n==budget or n//(len(rows)//2)==32 else [r for r in samples if r['split']=='dev']
                        _evaluate(system,selected,output,rank,target,label,n,retain=n==budget)
                        save(label)
                    elif n%checkpoint_every==0 or n==limit:save('latest')
            save('final' if trainer.completed_updates==budget else 'stopped')
            system.verify_core_binding()
            result=dict(status='COMPLETE' if trainer.completed_updates==budget else 'STOPPED_AT_REQUESTED_UPDATE',
                completed_updates=trainer.completed_updates,total_updates=budget,world_size=2,global_batch_size=2)
            if rank==0:atomic_json(output/'result.json',result)
            return result
    finally:
        if initialized:dist.destroy_process_group()
