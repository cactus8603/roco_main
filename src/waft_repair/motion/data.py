"""GT-free cached observations and separately bound native flow supervision."""
from __future__ import annotations

from dataclasses import asdict, fields
import json
from pathlib import Path
import re

import cv2
import numpy as np
import torch

from .io import atomic_json, atomic_torch, check_descriptor, descriptor, load_tensor, require, sha256
from .model import MotionPrior, prepare_motion_prior

INPUT_SCHEMA='waft-native-input/v1'
MANIFEST_SCHEMA='waft-native-manifest/v1'
INPUT_KEYS={'schema','id','original','sigma','observations','native_hw','padding',
            'baseline_flow','baseline_info','motion_prior'}
SAMPLE_KEYS={'id','split','image1','image2','ground_truth','ground_truth_format','group'}


def read_spec(path):
    path=Path(path).resolve(strict=True);value=json.loads(path.read_text())
    require(isinstance(value,dict) and set(value)=={'schema','samples'}
        and value['schema']=='waft-native-images/v1' and isinstance(value['samples'],list)
        and bool(value['samples']),'An explicit waft-native-images/v1 samples list is required')
    samples=[];ids=set();split_images={};groups={}
    for row in value['samples']:
        require(isinstance(row,dict) and not set(row)-SAMPLE_KEYS
            and {'id','split','image1','image2','ground_truth','ground_truth_format'}<=set(row),
            'Native sample keys differ; no target/teacher is accepted')
        require(isinstance(row['id'],str) and re.fullmatch(r'[A-Za-z0-9_./-]+',row['id'])
                and '..' not in row['id'].split('/') and row['id'] not in ids,'Unique safe sample IDs required')
        require(row['split'] in ('train','dev'),'Sample split must be train or dev')
        require(row['ground_truth_format'] in ('kitti','npz','sintel'),'Select kitti, npz or sintel GT')
        item=dict(row);ids.add(row['id'])
        for key in ('image1','image2','ground_truth'):
            file=(path.parent/row[key]).resolve(strict=True);item[key]=descriptor(file)
            from ..memory_hygiene import advise_readonly
            advise_readonly(file)
        for key in ('image1','image2'):
            digest=item[key]['sha256'];previous=split_images.setdefault(digest,row['split'])
            require(previous==row['split'],'Identical RGB bytes occur in both train and dev')
        group=row.get('group')
        if group is not None:
            require(isinstance(group,str) and bool(group),'Group must be a nonempty string')
            require(groups.setdefault(group,row['split'])==row['split'],'Group appears in both train and dev')
        samples.append(item)
    return samples


def read_labels(path, format):
    """Offline TRAIN/evaluation labels; never call this from image inference."""
    if format=='npz':
        from ..backend import load_ground_truth
        gt,valid,_=load_ground_truth(path)
    elif format=='kitti':
        raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        require(raw is not None and raw.dtype==np.uint16 and raw.ndim==3 and raw.shape[2]==3,
                'Original KITTI flow_occ must be a three-channel uint16 PNG')
        flow=(raw[...,[2,1]].astype(np.float32)-np.float32(32768))/np.float32(64)
        gt=torch.from_numpy(np.ascontiguousarray(flow.transpose(2,0,1)[None]))
        valid=torch.from_numpy(np.ascontiguousarray(((raw[...,0]>0)&np.isfinite(flow).all(-1))[None]))
    elif format=='sintel':
        from ..ground_truth import _sintel_flow
        flow=_sintel_flow(Path(path));gt=torch.from_numpy(np.ascontiguousarray(flow.transpose(2,0,1)[None]))
        valid=torch.from_numpy(np.ascontiguousarray((np.isfinite(flow).all(-1)&(np.abs(flow)<1000).all(-1))[None]))
    else:raise ValueError('Unknown original GT format')
    require(gt.dtype==torch.float32 and valid.dtype==torch.bool
        and gt.shape==(1,2,*valid.shape[-2:]) and bool(valid.any()),'Nonempty native FP32 GT/boolean support required')
    require(bool(torch.isfinite(gt.permute(0,2,3,1)[valid]).all()),'Nonfinite valid GT')
    return gt,valid


def input_packet(prepared, id):
    """Allowlist only original model observations. Labels live in another file."""
    return dict(schema=INPUT_SCHEMA,id=id,original=prepared.original,sigma=prepared.sigma.reshape(64),
        observations=dict(prepared.observations),native_hw=prepared.native_hw,padding=prepared.padding,
        baseline_flow=prepared.baseline_flow,baseline_info=prepared.baseline_info,
        motion_prior=asdict(prepared.motion_prior))


def validate_packet(value, *, expected_id=None):
    require(isinstance(value,dict) and set(value)==INPUT_KEYS and value['schema']==INPUT_SCHEMA,
            'Exact GT-free native input packet required')
    require(expected_id is None or value['id']==expected_id,'Input packet ID differs')
    h,w=value['native_hw'];left,right,top,bottom=value['padding'];hp=h+top+bottom;wp=w+left+right
    require(h>0 and w>0 and all(type(x) is int and x>=0 for x in (left,right,top,bottom))
            and hp%112==0 and wp%112==0,'Original WAFT padding geometry required')
    shape=(1,2,64,hp//2,wp//2)
    require(isinstance(value['original'],torch.Tensor) and tuple(value['original'].shape)==shape,'F0 geometry differs')
    require(tuple(value['sigma'].shape)==(64,) and bool((value['sigma']>0).all()),'Original positive sigma64 required')
    observations=value['observations'];channels={'cnn_pre_s2':(64,2),'cnn_pre_s4':(128,4),
        'dav2_block6':(384,14),'dav2_block12':(384,14)}
    require(isinstance(observations,dict) and observations.keys()==channels.keys(),'Exactly original four encoder observations required')
    for key,(count,stride) in channels.items():
        require(tuple(observations[key].shape)==(1,2,count,hp//stride,wp//stride),'Tap geometry differs: '+key)
    require(tuple(value['baseline_flow'].shape)==(1,2,h,w) and tuple(value['baseline_info'].shape)==(1,4,h,w),
            'Native baseline flow/info geometry differs')
    tensors=[value[k] for k in ('original','sigma','baseline_flow','baseline_info')]+list(observations.values())
    require(all(isinstance(v,torch.Tensor) and v.device.type=='cpu' and v.dtype==torch.float32
        and not v.requires_grad and v.grad_fn is None and bool(torch.isfinite(v).all()) for v in tensors),
        'Finite detached CPU FP32 original observations required')
    prior_dict=value['motion_prior']
    require(isinstance(prior_dict,dict) and set(prior_dict)=={f.name for f in fields(MotionPrior)},'Complete plain motion prior required')
    prior=MotionPrior(**prior_dict)
    require(prior.native_hw==tuple(value['native_hw']) and prior.padding==tuple(value['padding'])
            and prior.padded_hw==(hp,wp),'Motion prior geometry differs')
    # Strict creation/reload binding checks the actual canonical derived values.
    expected=prepare_motion_prior(value['baseline_flow'],value['baseline_info'],
                                 native_hw=value['native_hw'],padding=value['padding'])
    for field in fields(expected):
        actual,wanted=getattr(prior,field.name),getattr(expected,field.name)
        require(isinstance(actual,torch.Tensor) and torch.equal(actual,wanted) and actual.dtype==wanted.dtype
                and actual.device.type=='cpu' and not actual.requires_grad and actual.grad_fn is None
                if isinstance(wanted,torch.Tensor) else actual==wanted,'Motion prior differs: '+field.name)
    return value


def load_manifest(path):
    path=Path(path).resolve(strict=True);value=json.loads(path.read_text())
    require(value.get('schema')==MANIFEST_SCHEMA and isinstance(value.get('samples'),list)
            and bool(value['samples']),'A nonempty native manifest is required')
    require(value.get('model_abi')=='paired-feature-repair/motion-aligned-v1','Manifest selects a different model')
    from ..backend import CHECKPOINT_PROFILES
    core=value.get('waft',{})
    profile=CHECKPOINT_PROFILES.get(core.get('waft_checkpoint_sha256'))
    require(profile is not None and core.get('state_sha256')==profile['state_sha256'],
            'Manifest must bind a complete declared official WAFT checkpoint')
    ids=[row['id'] for row in value['samples']]
    require(len(set(ids))==len(ids) and all(row['split'] in ('train','dev') for row in value['samples']),
            'Unique IDs and explicit train/dev split required')
    return path,value


def load_sample(manifest_path,row,device='cpu'):
    root=Path(manifest_path).resolve().parent
    packet=load_tensor(check_descriptor(row['input'],root));validate_packet(packet,expected_id=row['id'])
    labels=load_tensor(check_descriptor(row['labels'],root))
    require(isinstance(labels,dict) and set(labels)=={'schema','id','gt','valid'}
        and labels['schema']=='waft-native-labels/v1' and labels['id']==row['id'],'Separate checked native labels required')
    gt,valid=labels['gt'],labels['valid'];h,w=packet['native_hw']
    require(gt.dtype==torch.float32 and gt.shape==(1,2,h,w) and valid.dtype==torch.bool
        and valid.shape==(1,h,w) and bool(valid.any()) and not gt.requires_grad,
        'Original native label geometry/support differs')
    require(int(valid.sum())==row['valid_pixels'],'Original valid denominator changed')
    require(bool(torch.isfinite(gt.permute(0,2,3,1)[valid]).all())
        and gt.grad_fn is None and not valid.requires_grad,'Detached finite GT on the original valid support required')
    prepared={key:packet[key].to(device) for key in ('original','sigma','baseline_flow','baseline_info')}
    prepared.update(observations={k:v.to(device) for k,v in packet['observations'].items()},
        native_hw=tuple(packet['native_hw']),padding=tuple(packet['padding']),
        motion_prior=MotionPrior(**packet['motion_prior']).to(device))
    return prepared,gt.to(device),valid.to(device)


def make_kitti_spec(kitti_root,output,split_file=None):
    """Create only an explicit image/GT list, preserving the published160/40 IDs."""
    if split_file is None:split_file=Path(__file__).with_name('kitti_split.json')
    split=json.loads(Path(split_file).read_text());require(split['schema']=='waft-native-kitti-split/v1','Checked KITTI split required')
    train,dev=split['train_ids'],split['dev_ids']
    require(len(train)==160 and len(dev)==40 and not set(train)&set(dev)
            and set(train+dev)=={f'{i:06d}' for i in range(200)},'Exact canonical KITTI160/40 required')
    root=Path(kitti_root).resolve(strict=True)
    if (root/'training').is_dir():root=root/'training'
    samples=[]
    for kind,ids in (('train',train),('dev',dev)):
        for id in ids:
            row=dict(id='kitti2015/'+id,split=kind,
                image1=str((root/'image_2'/f'{id}_10.png').resolve(strict=True)),
                image2=str((root/'image_2'/f'{id}_11.png').resolve(strict=True)),
                ground_truth=str((root/'flow_occ'/f'{id}_10.png').resolve(strict=True)),ground_truth_format='kitti')
            samples.append(row)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'images.json',dict(schema='waft-native-images/v1',samples=samples))
    atomic_json(output/'split.json',split)
    return dict(spec=str(output/'images.json'),train_pairs=160,dev_pairs=40,
        scope='Canonical historically exposed dev40; not official KITTI hidden test')

