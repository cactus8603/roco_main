"""Portable current-model prepare/train/resume/evaluate/RGB inference pipeline.

Only this adapter is optimized; all original WAFT weights stay frozen. Dataset
labels and GT-free cached observations are separate. Importing starts no work.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .backend import load_backend,load_rgb_pair,STATE_SHA256,SOURCE_COMMIT
from .memory_hygiene import case_boundary, advise_readonly
from .motion.io import (atomic_json,atomic_torch,atomic_text,check_descriptor,cpu_tree,descriptor,
                        load_tensor,output_lock,require,sha256)
from .motion.model import MotionFeatureRepair,MotionRepairConfig
from .motion.native import MotionNativeRepairSystem,native_task_loss
from .motion.data import (INPUT_SCHEMA,MANIFEST_SCHEMA,input_packet,load_manifest,
                         load_sample,read_labels,read_spec,validate_packet)
from .motion.training import (TrainConfig,NativeTrainer,epoch_order,expected_cursor,repair_forward)


def _system(source,checkpoint,device,*,repair=None,checkpoint_refinement=True,memory_fraction=None):
    device=torch.device(device)
    if memory_fraction is not None:
        require(device.type=='cuda' and 0<memory_fraction<=1,'Memory fraction requires explicit CUDA and a value in (0,1]')
        torch.cuda.set_per_process_memory_fraction(memory_fraction,device)
    backend=load_backend(source,checkpoint,device=device)
    model=MotionFeatureRepair() if repair is None else repair
    system=MotionNativeRepairSystem(backend.core,model.to(device),checkpoint_refinement=checkpoint_refinement,
        expected_core_digest=backend.metadata['state_sha256'],
        padder_factory=backend.padder_factory,coords_grid_fn=backend.coords_grid,
        bilinear_sampler_fn=backend.bilinear_sampler)
    case_boundary(torch,paths=[checkpoint])
    return system,backend.metadata


def _room(output,reserve_gib):
    require(reserve_gib>=0 and shutil.disk_usage(output).free>=reserve_gib*1024**3,
            'Output disk has less free space than the explicit reserve')


def _package_sources():
    # Relative module identities keep CPU continuation portable across installs.
    from .motion import model,native,loss,consumer,training,data,geometry,finite,io
    from . import backend,targets,ground_truth,data as legacy_data,native_cli,memory_hygiene
    from .motion import checkpoint
    modules=[model,native,loss,consumer,training,data,geometry,finite,io,backend,targets,
             ground_truth,legacy_data,native_cli,checkpoint,memory_hygiene]
    result={m.__name__:sha256(m.__file__) for m in modules}
    result[__name__]=sha256(__file__)
    return result


def _reconcile_history(output,completed_updates,history_name='history.jsonl'):
    """Preserve uncommitted post-checkpoint lines and resume the committed prefix."""
    latest=load_tensor(output/'checkpoint.pt')
    require(latest.get('completed_updates')==completed_updates,
        'Resume the latest committed checkpoint.pt; rewinding an existing run is not supported')
    history=output/history_name
    if not history.exists():
        require(completed_updates==0,'Committed training history is missing')
        return
    lines=history.read_text().splitlines(keepends=True)
    require(len(lines)>=completed_updates,'History is shorter than the committed checkpoint')
    for index,line in enumerate(lines[:completed_updates],1):
        value=json.loads(line)
        require(value['completed_updates']==index,'Committed history update order differs')
    if len(lines)>completed_updates:
        backup=output/f'{history.stem}-uncommitted-{time.time_ns()}.jsonl'
        atomic_text(backup,''.join(lines[completed_updates:]))
        atomic_text(history,''.join(lines[:completed_updates]))


def prepare(spec,output,waft_source,waft_checkpoint,*,device='cpu',reserve_gib=20,memory_fraction=None):
    """Cache one case at a time: F0/taps/fifth prior; separate GT; original all5."""
    samples=read_spec(spec);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    system,metadata=_system(waft_source,waft_checkpoint,device,memory_fraction=memory_fraction)
    system.eval();rows=[];started=time.monotonic()
    for index,sample in enumerate(samples):
        _room(output,reserve_gib)
        pair=load_rgb_pair(sample['image1']['path'],sample['image2']['path']).to(device)
        with torch.no_grad():prepared=system.prepare_pair(pair[:,0],pair[:,1])
        packet=cpu_tree(input_packet(prepared,sample['id']));validate_packet(packet,expected_id=sample['id'])
        prefix=f'packets/{index:06d}'
        atomic_torch(output/(prefix+'-input.pt'),packet)
        baseline=cpu_tree({key:prepared.baseline_output[key] for key in ('flow','info')})
        atomic_torch(output/(prefix+'-baseline.pt'),baseline)
        # Labels are first consumed here, after observation generation/storage.
        gt,valid=read_labels(sample['ground_truth']['path'],sample['ground_truth_format'])
        require(tuple(gt.shape[-2:])==prepared.native_hw,'RGB and GT native dimensions differ')
        atomic_torch(output/(prefix+'-labels.pt'),dict(schema='waft-native-labels/v1',id=sample['id'],gt=gt,valid=valid))
        row=dict(id=sample['id'],split=sample['split'],native_hw=list(prepared.native_hw),padding=list(prepared.padding),
            valid_pixels=int(valid.sum()),input=descriptor(output/(prefix+'-input.pt'),relative_to=output),
            labels=descriptor(output/(prefix+'-labels.pt'),relative_to=output),
            baseline=descriptor(output/(prefix+'-baseline.pt'),relative_to=output),
            sources={key:sample[key] for key in ('image1','image2','ground_truth')})
        if 'group' in sample:row['group']=sample['group']
        rows.append(row)
        atomic_json(output/'progress.json',dict(status='PREPARING',completed_pairs=len(rows),total_pairs=len(samples)))
        del pair,prepared,packet,baseline,gt,valid
        case_boundary(torch,paths=[sample[k]['path'] for k in ('image1','image2','ground_truth')])
    system.verify_core_binding()
    manifest=dict(schema=MANIFEST_SCHEMA,model_abi='paired-feature-repair/motion-aligned-v1',
        waft=metadata,samples=rows,supervision='Separate original native GT; no teacher features',
        preparation=dict(source_spec_sha256=sha256(spec),elapsed_seconds=time.monotonic()-started))
    atomic_json(output/'manifest.json',manifest)
    atomic_json(output/'progress.json',dict(status='COMPLETE',completed_pairs=len(rows),total_pairs=len(rows)))
    return dict(manifest=str(output/'manifest.json'),pairs=len(rows),
        train_pairs=sum(r['split']=='train' for r in rows),dev_pairs=sum(r['split']=='dev' for r in rows),
        teacher_features_generated=False,WAFT_frozen=True)


def _flow_metrics(flow,gt,valid):
    mask=valid.detach().cpu().numpy()[0]
    truth=gt.detach().cpu().numpy()[0].astype(np.float64)
    prediction=flow.detach().cpu().numpy()[0].astype(np.float64)
    require(bool(mask.any()) and np.isfinite(truth[:,mask]).all() and np.isfinite(prediction[:,mask]).all(),
            'Finite nonempty original valid population required')
    error=np.linalg.norm(prediction[:,mask]-truth[:,mask],axis=0)
    magnitude=np.linalg.norm(truth[:,mask],axis=0)
    bad=(error>3)&(error>.05*magnitude)
    return dict(EPE=float(error.mean()),EPE_sum=float(error.sum()),Fl_bad=int(bad.sum()),
        Fl_percent=float(100*bad.mean()),valid_pixels=int(mask.sum()))


def _evaluate(system,manifest_path,manifest,rows,output,*,save_outputs=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);records=[];system.eval()
    device=next(system.restorer.parameters()).device
    for index,row in enumerate(rows):
        prepared,gt,valid=load_sample(manifest_path,row,device)
        with torch.no_grad():
            features=repair_forward(system.restorer,prepared)
            predicted=system.solve_features(features,native_hw=prepared['native_hw'],padding=prepared['padding'])
            nll=float(native_task_loss(predicted,gt,valid))
        all5=[_flow_metrics(flow,gt,valid) for flow in predicted['flow']]
        baseline=load_tensor(check_descriptor(row['baseline'],manifest_path.parent))
        baseline5=[_flow_metrics(flow,gt,valid) for flow in baseline['flow']]
        artifact=None
        if save_outputs:
            target=output/f'{index:06d}.pt'
            atomic_torch(target,dict(schema='waft-native-saved-flow/v1',id=row['id'],split=row['split'],
                native_hw=prepared['native_hw'],padding=prepared['padding'],
                flow=predicted['flow'],info=predicted['info']))
            artifact=descriptor(target,relative_to=output)
            advise_readonly(target)
        records.append(dict(id=row['id'],split=row['split'],native5_MoL=nll,
                            all5=all5,baseline_all5=baseline5,artifact=artifact))
        del features,predicted,baseline,prepared,gt,valid
        case_boundary(torch,paths=[manifest_path.parent/row[k]['path'] for k in ('input','labels','baseline')])
    summaries={}
    for split in sorted({r['split'] for r in records}):
        selected=[r for r in records if r['split']==split];count=sum(r['all5'][-1]['valid_pixels'] for r in selected)
        summaries[split]=dict(cases=len(selected),valid_pixels=count,
            native5_MoL=float(np.mean([r['native5_MoL'] for r in selected])),
            EPE_case_equal=float(np.mean([r['all5'][-1]['EPE'] for r in selected])),
            EPE_pooled=sum(r['all5'][-1]['EPE_sum'] for r in selected)/count,
            Fl_percent_case_equal=float(np.mean([r['all5'][-1]['Fl_percent'] for r in selected])),
            Fl_percent_pooled=100*sum(r['all5'][-1]['Fl_bad'] for r in selected)/count,
            baseline_EPE_case_equal=float(np.mean([r['baseline_all5'][-1]['EPE'] for r in selected])),
            improved_cases=sum(r['all5'][-1]['EPE']<r['baseline_all5'][-1]['EPE'] for r in selected))
    result=dict(schema='waft-native-evaluation/v1',splits=summaries,cases=records,
                aggregation='Same original native GT-valid support, all five readouts, case-equal and pooled final metrics')
    atomic_json(output/'summary.json',result)
    return result


def train(manifest,output,waft_source,waft_checkpoint,*,epochs=32,seed=20261028,device='cpu',
          warmup_updates=160,resume=None,stop_after_updates=None,checkpoint_every=32,
          activation_offload='cpu',reserve_gib=20,memory_fraction=None):
    """Fixed seeded natural epochs; dev monitoring never selects checkpoints."""
    from .motion.checkpoint import export_model_checkpoint
    path,manifest=load_manifest(manifest);rows=[r for r in manifest['samples'] if r['split']=='train']
    require(bool(rows) and type(epochs) is int and epochs>0,'Positive epochs and TRAIN samples required')
    require(type(checkpoint_every) is int and checkpoint_every>0,'Positive checkpoint interval required')
    require(activation_offload in ('cpu','none'),'Activation offload must be cpu or none')
    budget=epochs*len(rows)
    config=TrainConfig(updates=budget,warmup_updates=warmup_updates)
    if stop_after_updates is not None:
        require(type(stop_after_updates) is int and 0<stop_after_updates<=budget,'Stop cursor must be within the unchanged total budget')
    output=Path(output).resolve()
    if resume is None:output.mkdir(parents=True,exist_ok=False)
    else:require(output.is_dir() and (output/'run.json').is_file(),'Resume needs its existing output directory')
    with output_lock(output):
        _room(output,reserve_gib)
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        seeded_model=MotionFeatureRepair(MotionRepairConfig(seed=seed))
        system,metadata=_system(waft_source,waft_checkpoint,device,repair=seeded_model,memory_fraction=memory_fraction)
        require(metadata==manifest['waft'],'Prepared inputs and actual WAFT binding differ')
        binding=dict(fixed_cases=[r['id'] for r in rows],epochs=epochs,shuffle_seed=seed,
                     manifest_sha256=sha256(path),schema='native-epoch-data/v1')
        source_binding=_package_sources()
        core_binding=dict(state_digest=metadata['state_sha256'],source_commit=SOURCE_COMMIT,torch_version=str(torch.__version__))
        trainer=NativeTrainer(system.restorer,system,config,model_config=asdict(system.restorer.config),
            binding=binding,core_binding=core_binding,
            observation_binding=dict(model_abi=manifest['model_abi'],four_original_taps=True),source_binding=source_binding)
        run=dict(schema='waft-native-run/v1',training_config=asdict(config),binding=binding,
            model=system.restorer.configuration(),source_binding=source_binding,core_binding=core_binding,
            activation_offload=activation_offload,checkpoint_refinement=True,
            CPU_threads=torch.get_num_threads(),CPU_interop_threads=torch.get_num_interop_threads(),
            fixed_final_epoch=epochs,automatic_checkpoint_selection=False,automatic_extension=False)
        if resume:
            require(json.loads((output/'run.json').read_text())==run,'Run metadata/settings changed during resume')
            trainer.load_state_dict(load_tensor(resume))
            _reconcile_history(output,trainer.completed_updates)
        else:atomic_json(output/'run.json',run)
        def save(label):
            cursor=expected_cursor(binding,trainer.completed_updates,label)
            state=trainer.state_dict(cursor=cursor);trainer.validate_checkpoint(state)
            atomic_torch(output/'checkpoint.pt',state)
            portable=export_model_checkpoint(system.restorer,trainer.completed_updates,
                metadata=dict(training_method='native5-MoL-gamma0.85',manifest_sha256=binding['manifest_sha256']),
                waft_checkpoint_sha256=metadata['waft_checkpoint_sha256'])
            atomic_torch(output/'model.pt',portable)
            if label.startswith('E'):atomic_torch(output/'checkpoints'/(label+'.pt'),state)
            atomic_json(output/'progress.json',dict(status='COMPLETE' if trainer.completed_updates==budget else 'TRAINING',
                completed_updates=trainer.completed_updates,total_updates=budget,cursor=cursor))
        if not resume:
            _evaluate(system,path,manifest,manifest['samples'],output/'evaluations/init',save_outputs=False)
            save('initial')
        current=trainer.completed_updates;limit=budget if stop_after_updates is None else stop_after_updates
        require(limit>=current,'Stop cursor precedes loaded checkpoint')
        with (output/'history.jsonl').open('a',encoding='utf-8') as log:
            while trainer.completed_updates<limit:
                _room(output,reserve_gib)
                epoch,offset=divmod(trainer.completed_updates,len(rows))
                order=epoch_order(binding['fixed_cases'],seed,epoch+1);row=rows[order[offset]]
                prepared,gt,valid=load_sample(path,row,device)
                context=torch.autograd.graph.save_on_cpu(pin_memory=False) if (
                    torch.device(device).type=='cuda' and activation_offload=='cpu') else nullcontext()
                with context:metrics=trainer.step(prepared,gt,valid)
                metrics.update(id=row['id'],epoch=epoch+1,offset_in_epoch=offset)
                log.write(json.dumps(metrics,allow_nan=False)+'\n');log.flush()
                del prepared,gt,valid
                case_boundary(torch,paths=[path.parent/row[k]['path'] for k in ('input','labels')],
                              log_path=output/'memory-boundaries.jsonl',completed_updates=trainer.completed_updates)
                n=trainer.completed_updates
                if n%len(rows)==0 and (n//len(rows) in (1,4,8,16,32) or n==budget):
                    label=f'E{n//len(rows):03d}'
                    _evaluate(system,path,manifest,manifest['samples'],output/'evaluations'/label,save_outputs=n==budget)
                    save(label)
                elif n%checkpoint_every==0 or n==limit:save('latest')
        save('final' if trainer.completed_updates==budget else 'stopped')
        system.verify_core_binding()
        result=dict(status='COMPLETE' if trainer.completed_updates==budget else 'STOPPED_AT_REQUESTED_UPDATE',
            completed_updates=trainer.completed_updates,total_updates=budget,model=str(output/'model.pt'),
            checkpoint=str(output/'checkpoint.pt'),WAFT_frozen=True,automatic_checkpoint_selection=False)
        atomic_json(output/'result.json',result)
        return result


def evaluate(manifest,repair_checkpoint,output,waft_source,waft_checkpoint,*,split='dev',device='cpu',expected_sha256=None):
    from .motion.checkpoint import load_model_checkpoint
    require(split in ('train','dev','all'),'Select train, dev or all')
    path,manifest=load_manifest(manifest)
    rows=[r for r in manifest['samples'] if split=='all' or r['split']==split]
    require(bool(rows),'Requested evaluation split is empty')
    model,receipt=load_model_checkpoint(repair_checkpoint,device=device,expected_sha256=expected_sha256)
    system,metadata=_system(waft_source,waft_checkpoint,device,repair=model)
    require(metadata==manifest['waft'],'Evaluation inputs bind a different WAFT')
    require(metadata['waft_checkpoint_sha256']==receipt['waft_checkpoint_sha256'],
            'Repair checkpoint and evaluation WAFT profiles differ')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    result=_evaluate(system,path,manifest,rows,output)
    system.verify_core_binding()
    return dict(output=str(output/'summary.json'),splits=result['splits'],checkpoint=receipt,optimizer_updates=0)


def infer(repair_checkpoint,image1,image2,output,waft_source,waft_checkpoint,*,device='cpu',expected_sha256=None):
    """GT-free two-image deployment. It never calls label readers."""
    from .motion.checkpoint import load_model_checkpoint
    output=Path(output).resolve();require(not output.exists(),'Prediction output already exists')
    model,receipt=load_model_checkpoint(repair_checkpoint,device=device,expected_sha256=expected_sha256)
    system,metadata=_system(waft_source,waft_checkpoint,device,repair=model)
    require(metadata['waft_checkpoint_sha256']==receipt['waft_checkpoint_sha256'],
            'Repair checkpoint and inference WAFT profiles differ')
    system.eval();pair=load_rgb_pair(image1,image2).to(device)
    with torch.no_grad():predicted=system.predict(pair[:,0],pair[:,1])
    flow_steps=np.stack([flow[0].permute(1,2,0).cpu().numpy() for flow in predicted['flow']])
    info_steps=np.stack([info[0].permute(1,2,0).cpu().numpy() for info in predicted['info']])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:np.savez(stream,flow=flow_steps[-1],flow_steps=flow_steps,info_steps=info_steps)
    system.verify_core_binding()
    return dict(output=str(output),native_hw=list(flow_steps.shape[1:3]),flow_steps_shape=list(flow_steps.shape),
                GT_inputs=False,teacher_inputs=False,optimizer_updates=0,checkpoint=receipt)

