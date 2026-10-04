"""Reproducible, resumable E01 driver. Labels are read after inference only."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch

from .artifacts import OFFICIAL_FILES, verify_checkpoint_provenance
from .backbones import CroCoAdapter
from .data import SpringProvider
from .evaluation import SCORER_VERSION, error_map, evaluate_pair, fixed_hard_mask, gt_valid_mask
from .geometry import sample_numpy
from .matching import MatchConfig
from .pipeline import PipelineConfig, StableBridge, _augmentation_shift
from .util import ROOT, event, initialize_run, jsonable, save_json, save_npz, sha256, stable_seed
from .robust_proxy import expand_profile_spec, parse_profile


S01_GATE_ARMS = (
    'hstar_generic', 'hstar_raw', 'hstar_solved', 'hstar_solved_no_rematch',
    'hstar_wrong_support', 'hstar_privileged_location',
    'hstar_privileged_value', 'hstar_privileged_location_value',
)


def _fixed_mask_sample(mask, budget, *identity):
    """Deterministically sample a fixed GT-only diagnostic mask."""
    mask=np.asarray(mask,bool)
    if isinstance(budget,bool) or not isinstance(budget,int) or budget < 1:
        raise ValueError('Diagnostic query budget must be a positive integer')
    indices=np.flatnonzero(mask.ravel())
    if len(indices) > budget:
        rng=np.random.default_rng(stable_seed('s01_hstar_sample_v1',*identity))
        indices=np.sort(rng.choice(indices,budget,replace=False))
    sampled=np.zeros_like(mask);sampled.ravel()[indices]=True
    return sampled


def _privileged_gt_donor(baseline, gt):
    """Choose the finite min4 branch nearest U0, for offline support diagnosis."""
    baseline=np.asarray(baseline,np.float32);gt=np.asarray(gt,np.float32)
    if gt.shape != (4,)+baseline.shape:
        raise ValueError('Privileged donor requires 4x2xHxW GT matching baseline')
    finite=np.isfinite(gt).all(axis=1)
    with np.errstate(invalid='ignore',over='ignore'):
        errors=np.linalg.norm(gt-baseline[None],axis=1)
    errors=np.where(finite,errors,np.inf)
    best=errors.argmin(axis=0)
    gather=np.broadcast_to(best[None,None],(1,2,*best.shape))
    selected=np.take_along_axis(gt,gather,axis=0)[0]
    valid=np.isfinite(errors).any(axis=0)
    return np.where(valid[None],selected,baseline).astype(np.float32)


def _privileged_repairable_support_mask(reference_predictions, gt, query, threshold):
    """Offline upper bound: a non-query support is eligible if any A_ref solves it."""
    if not np.isfinite(float(threshold)) or threshold <= 0:
        raise ValueError('Support threshold must be finite and positive')
    predictions=np.asarray(reference_predictions)
    if predictions.ndim != 4 or predictions.shape[1] != 2:
        raise ValueError('Reference predictions must be Kx2xHxW')
    query=np.asarray(query,bool)
    if query.shape != predictions.shape[2:]:
        raise ValueError('Query mask and reference predictions differ in shape')
    errors=np.stack([error_map(field,gt) for field in predictions])
    return gt_valid_mask(gt)&(errors.min(axis=0)<=threshold)&~query


def _support_donor_record(bridge, prepared, baseline, local_fields, anchors, query, solved,
                          reverse_fields=(), augmentation_fields=None, *, task=None,
                          action_predictions=(), action_prepared=()):
    """Record the exact local-operator decision at support sites used by H*.

    The record contains observations and frozen predictions only.  GT labeling
    remains an offline report step, after the run has completed.
    """
    baseline=np.asarray(baseline,np.float32);anchors=np.asarray(anchors,bool)
    query=np.asarray(query,bool);solved=np.asarray(solved,np.float32)
    if baseline.shape != solved.shape or baseline.shape[1:] != anchors.shape or query.shape != anchors.shape:
        raise ValueError('Donor diagnostic arrays must share one native grid')
    needed=np.zeros_like(anchors);qy,qx=np.nonzero(query);h,w=anchors.shape
    for dx,dy in bridge.rematcher.config.source_neighbors:
        xx,yy=qx+dx,qy+dy
        inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        needed[yy[inside],xx[inside]]=True
    y,x=np.nonzero(anchors&needed);xy=np.stack((x,y),-1).astype(np.float32)
    names=['identity']+[f'local_{index}' for index in range(len(local_fields))]
    candidates=np.stack([field[:,y,x].T for field in (baseline,*local_fields)]).astype(np.float32)
    k,n,_=candidates.shape
    if reverse_fields and len(reverse_fields) != k:
        raise ValueError('Reverse donor fields must align with identity and local operators')
    cycle_cost=np.empty((k,n),np.float32);cycle_valid=np.empty((k,n),bool)
    for index in range(k):
        if not reverse_fields:
            cycle_cost[index]=np.inf;cycle_valid[index]=False
            continue
        endpoint=xy+candidates[index]
        reverse,valid_cycle=sample_numpy(reverse_fields[index],endpoint)
        cycle_valid[index]=valid_cycle
        cycle_cost[index]=np.where(valid_cycle,np.linalg.norm(candidates[index]+reverse.T,axis=-1),np.inf)
    augmentation_fields=augmentation_fields or {}
    augmentation_names=tuple(augmentation_fields)
    stability_cost=np.empty((len(augmentation_names),k,n),np.float32)
    stability_valid=np.empty((len(augmentation_names),k,n),bool)
    for augmentation_index,name in enumerate(augmentation_names):
        fields=augmentation_fields[name]
        if len(fields) != k:
            raise ValueError('Augmentation donor fields must align with identity and local operators')
        dx,dy=_augmentation_shift(name)
        shifted_source=xy+np.array((dx,dy),np.float32)
        source_valid=((shifted_source[:,0]>=0)&(shifted_source[:,0]<=w-1)&
                      (shifted_source[:,1]>=0)&(shifted_source[:,1]<=h-1))
        for index in range(k):
            shifted_endpoint=xy+candidates[index]+np.array((dx,dy),np.float32)
            endpoint_valid=((shifted_endpoint[:,0]>=0)&(shifted_endpoint[:,0]<=w-1)&
                            (shifted_endpoint[:,1]>=0)&(shifted_endpoint[:,1]<=h-1))
            augmented=fields[index][:,y,x].T
            valid_aug=source_valid&endpoint_valid&np.isfinite(augmented).all(axis=-1)
            stability_valid[augmentation_index,index]=valid_aug
            stability_cost[augmentation_index,index]=np.where(
                valid_aug,np.linalg.norm(candidates[index]-augmented,axis=-1),np.inf)
    native_cost=np.empty((k,n),np.float32);native_feature_cost=np.empty((k,n),np.float32)
    native_photo_cost=np.empty((k,n),np.float32);native_uncertainty=np.empty((k,n),np.float32)
    native_valid=np.empty((k,n),bool);native_margin=np.empty((k,n),np.float32)
    native_margin_valid=np.empty((k,n),bool);native_alternative_valid_fraction=np.empty((k,n),np.float32)
    # Predictions are always present in diagnostic state because cycle and
    # augmentation studies also need the action fields.  Native evidence is
    # enabled only when each action has its own prepared descriptor bundle.
    action_native_available=bool(action_prepared)
    if action_native_available:
        if task not in ('stereo','flow') or len(action_predictions)!=k or len(action_prepared)!=k:
            raise ValueError('Action-native evidence must align with task and local operators')
        offsets=([(0,0),(-2,0),(2,0),(-4,0),(4,0),(-8,0),(8,0)] if task=='stereo' else
                 [(0,0),(-2,0),(2,0),(0,-2),(0,2),(-4,0),(4,0),(0,-4),(0,4),
                  (-8,0),(8,0),(0,-8),(0,8)])
        offsets=np.asarray(offsets,np.float32)
        for index,(prediction,own_prepared) in enumerate(zip(action_predictions,action_prepared)):
            center=candidates[index]
            center_cost,center_valid,center_feature,center_photo,_=bridge.rematcher.score(
                own_prepared,xy,center[None],center)
            native_cost[index]=center_cost[0];native_feature_cost[index]=center_feature[0]
            native_photo_cost[index]=center_photo[0];native_valid[index]=center_valid[0]
            native_uncertainty[index]=np.asarray(prediction.raw_uncertainty,np.float32)[y,x]
            hypotheses=center[None]+offsets[:,None]
            if task=='stereo':hypotheses[...,1]=0
            local_cost,local_valid,_,_,_=bridge.rematcher.score(own_prepared,xy,hypotheses,center)
            alternatives=np.where(local_valid[1:],local_cost[1:],np.inf)
            alternative_min=alternatives.min(axis=0)
            valid_alternative=np.isfinite(alternative_min)
            native_margin_valid[index]=center_valid[0]&valid_alternative
            native_margin[index]=0.
            np.subtract(alternative_min,center_cost[0],out=native_margin[index],
                        where=native_margin_valid[index])
            native_alternative_valid_fraction[index]=local_valid[1:].mean(axis=0)
    else:
        for array in (native_cost,native_feature_cost,native_photo_cost,native_uncertainty,
                      native_margin,native_alternative_valid_fraction):array.fill(0.)
        native_valid.fill(False);native_margin_valid.fill(False)
    if not n:
        return {'support_xy':xy,'candidates':candidates,
            'candidate_valid':np.empty((k,0),bool),'combined_cost':np.empty((k,0),np.float32),
            'feature_cost':np.empty((k,0),np.float32),'photo_cost':np.empty((k,0),np.float32),
            'delta':np.empty((k,0),np.float32),'best_indices':np.empty((0,),np.int64),
            'changed':np.empty((0,),bool),'candidate_names':names,
            'cycle_cost':cycle_cost,'cycle_valid':cycle_valid,
            'stability_cost':stability_cost,'stability_valid':stability_valid,
            'native_cost':native_cost,'native_feature_cost':native_feature_cost,
            'native_photo_cost':native_photo_cost,'native_uncertainty':native_uncertainty,
            'native_valid':native_valid,'native_margin':native_margin,
            'native_margin_valid':native_margin_valid,
            'native_alternative_valid_fraction':native_alternative_valid_fraction,
            'schema':'s01_support_donor_observations_v4','unique_support_sites':0,
            'cycle_available':bool(reverse_fields),'augmentation_names':list(augmentation_names),
            'augmentation_available':bool(augmentation_names),
            'action_native_available':action_native_available}
    raw,valid,feature_cost,photo_cost,delta=bridge.rematcher.score(
        prepared,xy,candidates,candidates[0])
    scores=raw+bridge.rematcher.config.movement_penalty*delta
    eligible=valid&(delta<=bridge.rematcher.config.max_update_px)
    scores=np.where(eligible,scores,np.inf)
    best=scores.argmin(0);ids=np.arange(n)
    changed=(best!=0)&valid[0]&(scores[best,ids]+bridge.rematcher.config.min_improvement<scores[0])
    selected=candidates[0].copy();selected[changed]=candidates[best[changed],ids[changed]]
    if not np.array_equal(selected,solved[:,y,x].T):
        raise ValueError('Recorded donor selection differs from solved donor field')
    return {'support_xy':xy,'candidates':candidates,'candidate_valid':valid,
        'combined_cost':raw,'feature_cost':feature_cost,'photo_cost':photo_cost,'delta':delta,
        'best_indices':best,'changed':changed,'candidate_names':names,
        'cycle_cost':cycle_cost,'cycle_valid':cycle_valid,
        'stability_cost':stability_cost,'stability_valid':stability_valid,
        'native_cost':native_cost,'native_feature_cost':native_feature_cost,
        'native_photo_cost':native_photo_cost,'native_uncertainty':native_uncertainty,
        'native_valid':native_valid,'native_margin':native_margin,
        'native_margin_valid':native_margin_valid,
        'native_alternative_valid_fraction':native_alternative_valid_fraction,
        'schema':'s01_support_donor_observations_v4','unique_support_sites':int(n),
        'cycle_available':bool(reverse_fields),'augmentation_names':list(augmentation_names),
        'augmentation_available':bool(augmentation_names),
        'action_native_available':action_native_available}


def _without_rematch(result, baseline):
    """Directly propagate the first legal support proposal; do not rank by target cost."""
    copied={key:value for key,value in result.items()}
    xy=result['query_xy'].astype(int);n=len(xy);candidates=result['candidates']
    names=result['candidate_names'];cfg=result['diagnostics']['config']
    base=baseline[:,xy[:,1],xy[:,0]].T if n else np.empty((0,2),np.float32)
    delta=np.linalg.norm(candidates-base[None],axis=-1)
    legal=result['candidate_valid']&(delta<=cfg['max_update_px'])
    if cfg.get('require_baseline_support',True) and n:
        legal &= result['candidate_valid'][0][None]
    best=np.zeros(n,np.int64);available=np.zeros(n,bool)
    for index,name in enumerate(names):
        if not name.startswith('support_'):continue
        choose=(~available)&legal[index]&(delta[index]>1e-6)
        best[choose]=index;available|=choose
    output=baseline.copy();accepted=np.zeros(baseline.shape[1:],bool)
    if n:
        output[:,xy[available,1],xy[available,0]]=candidates[best[available],np.flatnonzero(available)].T
        accepted[xy[available,1],xy[available,0]]=True
    copied.update(output=output,accepted=accepted,best_indices=best)
    copied['diagnostics']={**result['diagnostics'],'accepted_count':int(available.sum()),
        'selection_rule':'first_legal_support_without_current_target_cost_v1',
        'target_cost_used_for_selection':False}
    return copied


def _s01_spatial_gate_results(bridge, state, gt, hard, task, case_id, gate):
    """Run P1 diagnostics after inference; GT is confined to named privileged arms."""
    query=_fixed_mask_sample(hard,int(gate['query_budget']),case_id,gate['sample_seed'])
    pred=state['prediction'];prepared=state['prepared'];local=state['local_fields']
    im0,im1=state['image0'],state['image1'];origin=state['origin_xy']
    u0=pred.displacement;runtime_anchors=np.asarray(state['runtime_anchors'],bool)&~query
    privileged_anchors=_privileged_repairable_support_mask(
        (u0,*local),gt,query,float(gate['support_error_threshold_px']))
    gt_donor=_privileged_gt_donor(u0,gt)
    solved,solved_changes=bridge._solved_donor(prepared,u0,local,runtime_anchors,query)
    donor_record=_support_donor_record(
        bridge,prepared,u0,local,runtime_anchors,query,solved,state.get('reverse_fields',()),
        state.get('augmentation_fields',{}),task=task,
        action_predictions=state.get('action_predictions',()),
        action_prepared=state.get('action_prepared',()))
    privileged_location_donor,_=bridge._solved_donor(
        prepared,u0,local,privileged_anchors,query)
    wrong=np.roll(solved,int(gate['wrong_support_shift_px']),axis=2)
    definitions={
        'hstar_generic':(runtime_anchors,u0,'generic','none'),
        'hstar_raw':(runtime_anchors,u0,'raw','none'),
        'hstar_solved':(runtime_anchors,solved,'solved','none'),
        'hstar_wrong_support':(runtime_anchors,wrong,'solved','synthetic_wrong_value'),
        'hstar_privileged_location':(privileged_anchors,privileged_location_donor,'solved','GT_repairable_support_location'),
        'hstar_privileged_value':(runtime_anchors,gt_donor,'solved','GT_support_value'),
        'hstar_privileged_location_value':(privileged_anchors,gt_donor,'solved','GT_support_location_and_value'),
    }
    results={}
    for name,(anchors,donor,matcher_arm,privilege) in definitions.items():
        item=bridge.rematcher.refine(pred,im0,im1,task,prepared=prepared,mask=query,
            anchors=anchors,donor=donor,local_candidates=local,origin_xy=origin,arm=matcher_arm)
        item['diagnostics'].update({'s01_gate_arm':name,'query_source':'sampled_fixed_H_star',
            'privileged_input':privilege,'deployable':privilege=='none',
            'fixed_query_pixels':int(query.sum()),'full_H_star_pixels':int(hard.sum()),
            'support_pixels':int(anchors.sum()),
            'gate_donor_unique_support_sites':donor_record['unique_support_sites'],
            'gate_donor_changed_sites':int(solved_changes)})
        results[name]=item
    results['hstar_solved_no_rematch']=_without_rematch(results['hstar_solved'],u0)
    results['hstar_solved_no_rematch']['diagnostics'].update(
        {'s01_gate_arm':'hstar_solved_no_rematch','deployable':True,'privileged_input':'none'})
    return results,query,donor_record


def setup_device(device, memory_fraction=.60):
    torch.set_num_threads(4)
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; do not silently run a 437M model on CPU')
        torch.cuda.set_device(torch.device(device))
        torch.cuda.set_per_process_memory_fraction(memory_fraction, torch.device(device))


def resolve_models(profile, registry):
    output = {}
    for task in ('stereo', 'flow'):
        if profile == 'main':
            path = ROOT / 'artifacts/models/croco_v2/main' / OFFICIAL_FILES[task]
        elif profile == 'legacy_spring':
            path = Path(registry['models']['legacy_spring'][task]['path'])
        else:
            raise ValueError('Unknown model profile')
        output[task] = verify_checkpoint_provenance(path, task)
    return output


def geometry_run(directory, *, device='cuda:0', profile='main', contexts=((512,768),)):
    registry = json.loads((ROOT/'configs/stablebridge/data_models_v1.json').read_text())
    models = resolve_models(profile, registry)
    manifest = {'study': 'E01/S00_input_geometry', 'kind': 'real_model_geometry_validation',
                'model_profile': profile, 'models': models, 'scene': '0001', 'frame': 50,
                'contexts': ['native']+[list(hw) for hw in contexts], 'device': device,
                'precision': 'float32', 'scorer': SCORER_VERSION, 'gt_for_evaluation_only': True,
                'dataset_exposure': 'development_scene_no_unseen_claim'}
    directory = initialize_run(directory, manifest)
    setup_device(device)
    provider = SpringProvider.from_registry(ROOT/'configs/stablebridge/data_models_v1.json')
    rows = []
    save_json(directory/'status.json', {'state': 'running'})
    for task in ('stereo', 'flow'):
        for context in (None,)+tuple(contexts):
            adapter = CroCoAdapter(task, models[task]['path'], device=device, context_hw=context,
                                   expected_sha256=models[task]['sha256'])
            pair = provider.read_pair(task, '0001', 50, adapter.context_hw)
            parity = adapter.native_forward_parity(pair['image0'], pair['image1'])
            if not parity['passed']:
                raise RuntimeError(f'{task} failed native forward parity')
            if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
            pred = adapter.predict(pair['image0'],pair['image1'],pair['origin_xy'])
            gt = provider.read_gt(task,'0001',50,roi_xyhw=pair['roi_xyhw'])
            ch,cw=adapter.native_hw
            ch,cw=min(ch,adapter.context_hw[0]),min(cw,adapter.context_hw[1])
            sy,sx=(adapter.context_hw[0]-ch)//2,(adapter.context_hw[1]-cw)//2
            core=pred.displacement[:,sy:sy+ch,sx:sx+cw]
            core_gt=gt[:,:,sy:sy+ch,sx:sx+cw]
            yy,xx=np.indices(pred.displacement.shape[1:]);endpoint=np.stack((xx,yy),0)+pred.displacement
            h,w=pred.displacement.shape[1:]
            outside=(endpoint[0]<0)|(endpoint[0]>w-1)|(endpoint[1]<0)|(endpoint[1]>h-1)
            gt_endpoint=gt+np.stack((xx,yy),0)[None]
            gt_inside=(gt_endpoint[:,0]>=0)&(gt_endpoint[:,0]<=w-1)&(gt_endpoint[:,1]>=0)&(gt_endpoint[:,1]<=h-1)
            row={'task':task,'context_hw':list(adapter.context_hw),'parity':parity,
                 'all_output_finite':bool(np.isfinite(pred.displacement).all()),
                 'source_features_shape':list(pred.source_features.shape),
                 'target_features_shape':list(pred.target_features.shape),
                 'feature_centers_first_native_xy':pred.token_centers_xy[0,0].tolist(),
                 'predicted_endpoint_outside_context_pct':float(100*outside.mean()),
                 'gt_endpoint_outside_context_pct':float(100*(~gt_inside.any(0)&gt_valid_mask(gt)).sum()/max(1,gt_valid_mask(gt).sum())),
                 'evaluation':evaluate_pair(pred.displacement,pred.displacement,gt),
                 'common_native_roi_evaluation':evaluate_pair(core,core,core_gt),
                 'common_native_roi_xyhw':[pair['origin_xy'][0]+sx,pair['origin_xy'][1]+sy,ch,cw],
                 'costs':pred.metadata,'input_metadata':pair['metadata'],
                 'peak_allocated_cuda_bytes':int(torch.cuda.max_memory_allocated()) if device.startswith('cuda') else 0}
            rows.append(row)
            save_json(directory/'geometry.json',rows)
            event(directory,'geometry_case_complete',task=task,context=adapter.context_hw,
                  parity=parity['passed'],error=row['evaluation']['regions']['all']['error_px'])
            del adapter,pred,gt;gc.collect()
            if device.startswith('cuda'):torch.cuda.empty_cache()
    save_json(directory/'status.json',{'state':'completed','cases':len(rows)})
    return rows


def _candidate_labels(result, gt, baseline):
    """Post-inference labels. They are never returned to the running pipeline."""
    xy=result['query_xy'].astype(int);x,y=xy[:,0],xy[:,1]
    candidates=result['candidates']
    local_gt=gt[:,:,y,x].transpose(0,2,1)
    errors=np.linalg.norm(local_gt[None]-candidates[:,None],axis=-1).min(axis=1)
    valid=gt_valid_mask(gt)[y,x]
    errors[:,~valid]=np.nan
    base_error=error_map(baseline,gt)[y,x]
    return errors,base_error


def _case_rows(identity, inferred, gt, *, export_candidates=True):
    baseline=inferred['baseline'];hard=fixed_hard_mask(inferred['reference_predictions'],gt)
    rows=[];candidate_records=[];outputs={'U0':baseline}
    for arm,result in inferred['arms'].items():
        outputs[arm]=result['output']
        if 'candidates' not in result:continue
        errors,base_error=_candidate_labels(result,gt,baseline)
        xy=result['query_xy'].astype(int);x,y=xy[:,0],xy[:,1];k,n=errors.shape
        # Per-query oracle over precisely the generated bank, including identity.
        cfg=result.get('diagnostics',{}).get('config',{})
        features=result['candidate_features']
        eligible=features[...,8].astype(bool)&(features[...,5]*32 <= cfg.get('max_update_px',32.))
        if cfg.get('require_baseline_support',True):eligible &= features[...,9].astype(bool)
        eligible[0]=True
        best=np.where(eligible,np.nan_to_num(errors,nan=np.inf),np.inf).argmin(0)
        oracle=baseline.copy();oracle[:,y,x]=result['candidates'][best,np.arange(n)].T
        outputs[arm+'_candidate_oracle']=oracle
        if export_candidates and arm in ('solved','temporal'):
            candidate_records.append({'features':result['candidate_features'].reshape(-1,12),
                'baseline_error':np.broadcast_to(base_error,(k,n)).ravel(),
                'candidate_error':errors.ravel(),
                'candidate_index':np.repeat(np.arange(k),n),
                'query_id':np.tile(np.array([identity['case_id']+'/'+arm+f'/{xx}/{yy}' for xx,yy in xy]),k),
                'scene':np.full(k*n,identity['scene']), 'split':np.full(k*n,identity['split'])})
    for arm,output in outputs.items():
        actual_arm=arm.removesuffix('_candidate_oracle')
        result=inferred['arms'].get(actual_arm,{})
        selected=result.get('query_mask',inferred['query_mask'])
        gt_valid=gt_valid_mask(gt)
        rows.append({**identity,'arm':arm,'diagnostic':arm.endswith('_candidate_oracle'),
            'oracle_scope':'decision_candidates_at_snapshot' if arm.endswith('_candidate_oracle') else None,
            'metrics':evaluate_pair(baseline,output,gt,hard,selected_mask=selected),
            'selected_pixels':int((selected & gt_valid).sum()),
            'accepted_pixels':int((result.get('accepted',np.zeros_like(selected))&gt_valid).sum()) if not arm.endswith('_candidate_oracle') else None,
            'costs':{**inferred['costs'],'arm':result.get('diagnostics',{})}})
    return rows,outputs,candidate_records,hard


def verify_saved_case(case_file, directory):
    """Resume verifies the exact RGB, GT and output bytes previously scored."""
    saved=json.loads(Path(case_file).read_text())
    inputs=saved['input_metadata']['inputs']
    for item in inputs:
        if sha256(item['path']) != item['file_sha256']:
            raise ValueError('Saved case RGB bytes changed during resume')
    gt=saved['evaluation_gt']
    if sha256(gt['path']) != gt['sha256']:
        raise ValueError('Saved case evaluation GT bytes changed during resume')
    for relative,digest in saved.get('artifact_hashes',{}).items():
        if sha256(Path(directory)/relative) != digest:
            raise ValueError('Saved case output artifact changed during resume')
    return saved


def run_experiment(config_path, directory, *, device='cuda:0', acceptor_path=None):
    config_path=Path(config_path)
    config=json.loads(config_path.read_text())
    config['profiles'] = expand_profile_spec(config['profiles'])
    clips_path=ROOT/config['clips_manifest']
    clips=json.loads(clips_path.read_text())['clips']
    if config.get('scenes'):clips=[c for c in clips if c['scene'] in config['scenes']]
    if not clips or not config.get('profiles') or not config.get('arms'):
        raise ValueError('Experiment must specify nonempty scenes, profiles and arms')
    if config.get('frames','center') not in ('center','three','all'):
        raise ValueError('frames must be center, three or all')
    expected_cases=sum((1 if config.get('frames','center')=='center' else
                        3 if config['frames']=='three' else
                        len(c['stereo_query_frames']) if task=='stereo' else len(c['flow_query_pairs']))
                       for c in clips for task in config.get('tasks',['stereo','flow']))*len(config['profiles'])
    gate=config.get('spatial_gate')
    if gate is not None:
        required={'query_budget','sample_seed','support_error_threshold_px','wrong_support_shift_px',
                  'privileged_location_definition'}
        if set(gate) != required:
            raise ValueError(f'spatial_gate must contain exactly {sorted(required)}')
        if config.get('frames','center') != 'center' or config.get('pipeline',{}).get('mode','M0') != 'M0':
            raise ValueError('S01 spatial gate is a center-frame M0 diagnostic')
        if int(gate['query_budget']) > int(config.get('matching',{}).get('max_queries',4096)):
            raise ValueError('spatial_gate query budget exceeds matcher max_queries')
        if not np.isfinite(float(gate['support_error_threshold_px'])) or float(gate['support_error_threshold_px']) <= 0:
            raise ValueError('spatial_gate support threshold must be finite and positive')
        if gate['privileged_location_definition'] != 'min_A_ref_error_le_1px_outside_query':
            raise ValueError('Unknown privileged support-location definition')
    action_gate=config.get('action_response_gate')
    if action_gate is not None:
        required={'split','repair_threshold_native_px','harm_delta_native_px',
                  'tail_harm_delta_native_px','aggregation','missing_evidence',
                  'minimum_valid_candidate_slots_pct','minimum_oracle_scene_macro_gain_px',
                  'minimum_rank_aggregate_scene_macro_gain_px',
                  'minimum_gain_over_combined_movement_px',
                  'minimum_pair_ordering_improvement_percentage_points',
                  'minimum_positive_scenes_out_of_four',
                  'require_harmful_selection_rate_not_above_combined','on_failure'}
        if set(action_gate) != required:
            raise ValueError(f'action_response_gate must contain exactly {sorted(required)}')
        pipeline_options=config.get('pipeline',{})
        if gate is None or not pipeline_options.get('diagnostic_reverse_pairs'):
            raise ValueError('Action-response gate requires the S01 spatial gate and reverse pairs')
        if tuple(pipeline_options.get('diagnostic_augmentations',())) != ('roll_x8','roll_y8'):
            raise ValueError('Action-response v5 requires the locked x/y translation probes')
        thresholds=[float(action_gate[key]) for key in (
            'repair_threshold_native_px','harm_delta_native_px','tail_harm_delta_native_px')]
        if any(not np.isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError('Action-response error and harm thresholds must be positive')
        if thresholds[2] <= thresholds[1]:
            raise ValueError('Tail harm threshold must exceed the harm threshold')
        if action_gate['missing_evidence'] != 'retain_identity':
            raise ValueError('Action-response v5 must fail closed to identity')
    native_gate=config.get('action_native_gate')
    if native_gate is not None:
        required={'split','primary_score','repair_threshold_native_px','harm_delta_native_px',
                  'tail_harm_delta_native_px','missing_evidence','minimum_valid_candidate_slots_pct',
                  'minimum_oracle_scene_macro_gain_px','minimum_native_margin_scene_macro_gain_px',
                  'minimum_gain_over_combined_movement_px',
                  'minimum_pair_ordering_improvement_percentage_points',
                  'minimum_positive_scenes_out_of_four',
                  'require_harmful_selection_rate_not_above_combined','on_failure'}
        if set(native_gate) != required:
            raise ValueError(f'action_native_gate must contain exactly {sorted(required)}')
        pipeline_options=config.get('pipeline',{})
        if gate is None or not pipeline_options.get('diagnostic_action_native_evidence'):
            raise ValueError('Action-native gate requires the S01 spatial gate and native evidence')
        if pipeline_options.get('diagnostic_augmentations'):
            raise ValueError('Action-native v6 isolates native evidence from augmentation probes')
        if native_gate['primary_score']!='negative_native_local_margin':
            raise ValueError('Unknown action-native primary score')
        if native_gate['missing_evidence']!='retain_identity':
            raise ValueError('Action-native v6 must fail closed to identity')
    applicability_gate=config.get('evidence_applicability_gate')
    if applicability_gate is not None:
        required={'split','discovery_scenes','stereo_primary','flow_primary',
                  'repair_threshold_native_px','harm_delta_native_px','tail_harm_delta_native_px',
                  'missing_evidence','minimum_valid_candidate_slots_pct',
                  'minimum_oracle_scene_macro_gain_px','minimum_primary_scene_macro_gain_px',
                  'minimum_gain_over_combined_movement_px','minimum_absolute_pair_ordering_pct',
                  'minimum_positive_scenes_out_of_four',
                  'require_harmful_selection_rate_not_above_combined','on_failure'}
        if set(applicability_gate)!=required:
            raise ValueError(f'evidence_applicability_gate must contain exactly {sorted(required)}')
        pipeline_options=config.get('pipeline',{})
        if gate is None or not pipeline_options.get('diagnostic_reverse_pairs'):
            raise ValueError('Evidence-applicability gate requires spatial gate and reverse pairs')
        if tuple(pipeline_options.get('diagnostic_augmentations',()))!=('roll_x8','roll_y8'):
            raise ValueError('Evidence-applicability v7 requires locked x/y translation probes')
        expected_rules={
            'stereo_primary':'equal_rank_combined_movement_reverse_cycle_roll_y8_stability',
            'flow_primary':'mean_roll_x8_roll_y8_stability'}
        if any(applicability_gate[key]!=value for key,value in expected_rules.items()):
            raise ValueError('Unknown evidence-applicability primary rule')
        if applicability_gate['missing_evidence']!='retain_identity':
            raise ValueError('Evidence-applicability v7 must fail closed to identity')
        if set(config.get('scenes',()))&set(applicability_gate['discovery_scenes']):
            raise ValueError('Evidence-applicability confirmation scenes overlap discovery scenes')
    registry=json.loads((ROOT/'configs/stablebridge/data_models_v1.json').read_text())
    models=resolve_models(config.get('model_profile','main'),registry)
    match=MatchConfig(**config.get('matching',{}));pipe_config=PipelineConfig(**config.get('pipeline',{}))
    context=config.get('context_hw')
    acceptor=None
    if acceptor_path:
        from .learning import load_acceptor
        acceptor=load_acceptor(acceptor_path)
    effective_arms=len(config['arms'])+(len(S01_GATE_ARMS) if gate is not None else 0)
    manifest={**config,'run_schema':1,'config_path':str(config_path.resolve()),'config_sha256':sha256(config_path),
              'clip_manifest_sha256':sha256(clips_path),'model_artifacts':models,'device':device,
              'effective_clips':clips,'spring_provider':registry['datasets']['spring'],
              'matching':asdict(match),'pipeline':asdict(pipe_config),'scorer_version':SCORER_VERSION,
              'acceptor':None if not acceptor_path else {'path':str(Path(acceptor_path).resolve()),'sha256':sha256(acceptor_path)},
              'hard_mask':'GT-only diagnostic: min over fixed identity/median3/gaussian1 bank >1 native px',
              'runtime_query_mask':'GT-free fixed observational risk; same mask across arms',
              'oracle':'best generated candidate at current information cutoff, diagnostic only',
              'expected_cases':expected_cases,'expected_metric_rows':expected_cases*(1+2*effective_arms),
              'gt_status':'evaluation_and_supervision_only; never passed into StableBridge.predict',
              'exposure':'all listed scenes previously used for development; no final unseen claim',
              'corruption_claim':('RobustSpring-20 local proxy with three severities; weather/motion are not '
                                  'official-renderer equivalence' if all(
                                      str(p.get('name','')).startswith('robust20__') for p in config['profiles'])
                                  else 'controlled synthetic profiles; not official RobustSpring corruption performance')}
    directory=initialize_run(directory,manifest)
    setup_device(device,config.get('cuda_memory_fraction',.60))
    provider=SpringProvider.from_registry(ROOT/'configs/stablebridge/data_models_v1.json')
    save_json(directory/'status.json',{'state':'running','started_unix':time.time()})
    # Cases are atomic. Resume replays the prefix to rebuild causal state and
    # compares saved source/input hashes; it never skips memory warmup.
    for task in config.get('tasks',['stereo','flow']):
        adapter=CroCoAdapter(task,models[task]['path'],device=device,context_hw=context,
                             expected_sha256=models[task]['sha256'])
        bridge=StableBridge(adapter,pipe_config,match,acceptor)
        for clip in clips:
            scene=clip['scene'];center=clip['legacy_center_frame']
            frame_mode=config.get('frames','center')
            frames=([center] if frame_mode=='center' else
                    [center-1,center,center+1] if frame_mode=='three' else
                    clip['stereo_query_frames'] if task=='stereo' else [p[0] for p in clip['flow_query_pairs']])
            for condition in config['profiles']:
                profile=condition['name'];options=dict(condition.get('options',{}))
                if profile=='recovery':options['recovery_frame']=center
                namespace=f'{scene}/{task}/{profile}/{models[task]["sha256"]}'
                bridge.reset(namespace)
                for frame in frames:
                    case_id=f'{task}_{scene}_{frame:04d}_{profile}'
                    case_file=directory/'metrics'/f'{case_id}.json'
                    old=verify_saved_case(case_file,directory) if case_file.exists() else None
                    if old is not None and pipe_config.mode=='M0':continue
                    event(directory,'case_started',case_id=case_id)
                    pair=provider.read_pair(task,scene,frame,adapter.context_hw,profile=profile,
                                            seed=config.get('seed',20260916),corruption_options=options)
                    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
                    inferred=(bridge.predict(pair,arms=config['arms'],return_diagnostic_state=True)
                              if gate is not None else bridge.predict(pair,arms=config['arms']))
                    inferred['costs']['peak_allocated_cuda_bytes']=int(torch.cuda.max_memory_allocated()) if device.startswith('cuda') else 0
                    if old is not None:
                        if old['input_metadata'] != pair['metadata']:
                            raise ValueError('Input bytes changed during resume')
                        continue
                    if parse_profile(profile) is None:
                        gt=provider.read_gt(task,scene,frame,roi_xyhw=pair['roi_xyhw'])
                    else:
                        gt=provider.read_corrupted_gt(
                            task,scene,frame,roi_xyhw=pair['roi_xyhw'],profile=profile,
                            seed=config.get('seed',20260916))
                    if gate is not None:
                        # Labels become available only after deployment inference
                        # has returned. All resulting arms are explicit offline
                        # mechanism diagnostics, never deployment outputs.
                        state=inferred.pop('_diagnostic_state')
                        hard=fixed_hard_mask(inferred['reference_predictions'],gt)
                        gate_started=time.perf_counter()
                        gate_results,gate_mask,donor_record=_s01_spatial_gate_results(
                            bridge,state,gt,hard,task,case_id,gate)
                        inferred['arms'].update(gate_results)
                        inferred['diagnostic_query_mask']=gate_mask
                        inferred['donor_diagnostic']=donor_record
                        inferred['costs']['spatial_gate_seconds']=time.perf_counter()-gate_started
                    identity={'task':task,'scene':scene,'frame':frame,'profile':profile,'case_id':case_id,
                              'split':clip['development_role'],'warmup':frame==frames[0]}
                    rows,outputs,candidates,hard=_case_rows(identity,inferred,gt)
                    # Original dense estimates and every actual arm are retained.
                    save_npz(directory/'predictions'/f'{case_id}.npz',
                             **{k:v for k,v in outputs.items() if not k.endswith('_candidate_oracle')},
                             hard_mask=hard,query_mask=inferred['query_mask'],
                             **({'diagnostic_query_mask':inferred['diagnostic_query_mask']} if gate is not None else {}))
                    decisions={k:{key:r[key] for key in ('diagnostics','candidate_names','candidate_groups','feature_schema') if key in r}
                               for k,r in inferred['arms'].items()}
                    decision_payload={'query':inferred['query'],'snapshot':inferred['snapshot'],'arms':decisions,
                                      'accepted_outputs_are_not_memory_evidence':True}
                    if gate is not None:
                        donor=inferred['donor_diagnostic']
                        decision_payload['donor_diagnostic']={
                            key:donor[key] for key in ('schema','candidate_names','unique_support_sites',
                                'cycle_available','augmentation_names','augmentation_available')}
                        decision_payload['donor_diagnostic']['action_native_available']=donor['action_native_available']
                        save_npz(directory/'decisions'/f'{case_id}_donor.npz',
                                 **{key:value for key,value in donor.items()
                                    if isinstance(value,np.ndarray)})
                    save_json(directory/'decisions'/f'{case_id}.json',decision_payload)
                    for arm,result in inferred['arms'].items():
                        if 'query_xy' not in result:continue
                        xy=result['query_xy'].astype(int)
                        save_npz(directory/'decisions'/f'{case_id}_{arm}.npz',
                                 query_xy=result['query_xy'],candidates=result['candidates'],
                                 candidate_valid=result['candidate_valid'],
                                 candidate_costs=result['candidate_costs'],
                                 candidate_features=result['candidate_features'],
                                 best_indices=result['best_indices'],
                                 accepted=result['accepted'][xy[:,1],xy[:,0]])
                    for index,record in enumerate(candidates):
                        save_npz(directory/'supervision'/f'{case_id}_{index}.npz',**record)
                    # GT provenance is evaluation-side only.
                    gt_path=provider.gt_path(task,scene,frame)
                    artifacts=[p for folder in ('predictions','decisions','supervision')
                               for p in (directory/folder).glob(case_id+'*') if p.is_file()]
                    save_json(case_file,{'rows':rows,'input_metadata':pair['metadata'],
                                         'evaluation_gt':{'path':str(gt_path),'sha256':sha256(gt_path)},
                                         'artifact_hashes':{str(p.relative_to(directory)):sha256(p) for p in artifacts}})
                    event(directory,'case_complete',case_id=case_id,seconds=inferred['costs']['total_seconds'],
                          baseline_error=rows[0]['metrics']['regions']['all']['error_px'],
                          memory_records=len(bridge.memory))
                    del inferred,outputs,candidates,gt
        del bridge,adapter;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
    save_json(directory/'status.json',{'state':'completed','finished_unix':time.time()})
    materialize_results(directory)
    event(directory,'run_complete')
    return directory


def materialize_results(directory):
    directory=Path(directory)
    rows=[]
    for path in sorted((directory/'metrics').glob('*.json')):
        rows.extend(json.loads(path.read_text())['rows'])
    temporary=directory/'metrics.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(jsonable(row),allow_nan=False)+'\n' for row in rows))
    temporary.replace(directory/'metrics.jsonl')
    from .reporting import write_report
    write_report(directory)


def collect_supervision(run_directories, destination):
    directories=[Path(p).resolve() for p in run_directories]
    if len(set(directories)) != len(directories):
        raise ValueError('Duplicate run directories would duplicate supervised query groups')
    records=[]
    for directory in directories:
        for path in sorted((Path(directory)/'supervision').glob('*.npz')):
            with np.load(path,allow_pickle=False) as data:
                record={key:data[key] for key in data.files}
                from hashlib import sha256 as hash_bytes
                prefix=hash_bytes(str(directory).encode()).hexdigest()[:16]+'/'
                record['query_id']=np.char.add(prefix,record['query_id'])
                records.append(record)
    if not records:raise ValueError('No generated candidate supervision found')
    keys=records[0].keys()
    arrays={key:np.concatenate([record[key] for record in records]) for key in keys}
    finite=np.isfinite(arrays['baseline_error'])&np.isfinite(arrays['candidate_error'])&np.isfinite(arrays['features']).all(1)
    # Remove whole query groups with invalid GT rather than shifting identity indices.
    bad_queries=np.unique(arrays['query_id'][~finite])
    keep=~np.isin(arrays['query_id'],bad_queries)
    save_npz(destination,**{key:arr[keep] for key,arr in arrays.items()})
    save_json(str(destination)+'.json',{'runs':[str(Path(p).resolve()) for p in run_directories],
              'rows':int(keep.sum()),'invalid_gt_queries_removed':len(bad_queries),
              'label_scope':'post-inference supervised learning only',
              'signed_gain_target':'baseline_error-candidate_error; never truncated'})
