"""CPU boundaries for native inputs, separate labels, portable paths and KITTI IDs."""
from copy import deepcopy
from dataclasses import fields
from importlib.resources import files
import json
from pathlib import Path
import shutil
import struct

import cv2
import numpy as np
import pytest
import torch

from waft_repair.motion import MotionFeatureRepair, MotionRepairConfig, MotionNativeRepairSystem
from waft_repair.motion.data import (
    INPUT_KEYS, INPUT_SCHEMA, MANIFEST_SCHEMA, input_packet, load_manifest,
    load_sample, make_kitti_spec, read_labels, read_spec, validate_packet,
)
from waft_repair.motion.io import atomic_json, atomic_torch, descriptor, sha256
from motion_helpers import TinyCore, TapEncoder, TapCNN, ZeroPadder, coordinates, sampler


@pytest.fixture(scope='module', autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def prepared():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(442);core=TinyCore();core.encoder,core.fnet=TapEncoder(),TapCNN()
    cfg=MotionRepairConfig(profile='test',fine_dim=4,s4_dim=8,s8_dim=8,s16_dim=12,
        s4_blocks=1,s8_blocks=1,s16_blocks=1,decoder_blocks=1)
    system=MotionNativeRepairSystem(core,MotionFeatureRepair(cfg),strict_official=False,
        checkpoint_refinement=False,padder_factory=ZeroPadder,
        coords_grid_fn=coordinates,bilinear_sampler_fn=sampler)
    one=torch.arange(3*57*61).reshape(1,3,57,61).remainder(256).float()
    return system.prepare_pair(one,one.roll(2,-1))


@pytest.fixture
def packet(prepared):
    return deepcopy(input_packet(prepared,'synthetic/0'))


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if hasattr(a,'__dataclass_fields__'):return all(same(getattr(a,f.name),getattr(b,f.name)) for f in fields(a))
    return a==b


def write_spec(directory,rows):
    path=directory/'images.json';atomic_json(path,{'schema':'waft-native-images/v1','samples':rows});return path


def source_files(directory,prefix):
    # Specification parsing binds bytes without decoding or opening labels.
    result={}
    for name in ('image1','image2','ground_truth'):
        p=directory/(prefix+'-'+name+'.bin');p.write_bytes((prefix+'-'+name).encode());result[name]=p.name
    return result


def row(directory,prefix='a',split='train',**extra):
    return dict(id='synthetic/'+prefix,split=split,ground_truth_format='npz',**source_files(directory,prefix),**extra)


def cached_case(directory,packet):
    input_path=directory/'inputs'/'case.pt';label_path=directory/'labels'/'case.pt'
    h,w=packet['native_hw'];gt=torch.zeros(1,2,h,w);valid=torch.ones(1,h,w,dtype=torch.bool);valid[...,0,0]=False
    labels=dict(schema='waft-native-labels/v1',id=packet['id'],gt=gt,valid=valid)
    atomic_torch(input_path,packet);atomic_torch(label_path,labels)
    sample=dict(id=packet['id'],split='train',input=descriptor(input_path,relative_to=directory),
        labels=descriptor(label_path,relative_to=directory),valid_pixels=int(valid.sum()))
    path=directory/'manifest.json'
    atomic_json(path,dict(schema=MANIFEST_SCHEMA,model_abi='paired-feature-repair/motion-aligned-v1',
        waft={'state_sha256':MotionNativeRepairSystem.SOURCE_ONLY_STATE},samples=[sample]))
    return path,sample,labels


def test_read_spec_resolves_relative_files_and_binds_original_bytes(tmp_path):
    expected=row(tmp_path);path=write_spec(tmp_path,[expected]);samples=read_spec(path)
    assert len(samples)==1 and samples[0]['id']==expected['id']
    for key in ('image1','image2','ground_truth'):
        file=tmp_path/expected[key]
        assert samples[0][key]=={'path':str(file.resolve()),'bytes':file.stat().st_size,'sha256':sha256(file)}


@pytest.mark.parametrize('extra', ['teacher','target','gt','teacher_features'])
def test_read_spec_rejects_teacher_or_label_sidechannels(tmp_path,extra):
    value=row(tmp_path);value[extra]='unused'
    with pytest.raises(ValueError,match='no target/teacher'):
        read_spec(write_spec(tmp_path,[value]))


def test_read_spec_rejects_same_rgb_content_across_splits_even_at_different_paths(tmp_path):
    train=row(tmp_path,'a');dev=row(tmp_path,'b','dev')
    (tmp_path/dev['image2']).write_bytes((tmp_path/train['image1']).read_bytes())
    with pytest.raises(ValueError,match='RGB bytes'):
        read_spec(write_spec(tmp_path,[train,dev]))


def test_read_spec_group_overlap_duplicate_id_unsafe_id_and_unknown_GT_rejected(tmp_path):
    a=row(tmp_path,'a',group='component');b=row(tmp_path,'b','dev',group='component')
    with pytest.raises(ValueError,match='Group'):
        read_spec(write_spec(tmp_path,[a,b]))
    b.pop('group');b['id']=a['id']
    with pytest.raises(ValueError,match='IDs'):
        read_spec(write_spec(tmp_path,[a,b]))
    for key,bad in [('id','a/../b'),('ground_truth_format','png-guessed')]:
        badrow=dict(a);badrow[key]=bad
        with pytest.raises(ValueError):read_spec(write_spec(tmp_path,[badrow]))


def test_kitti_labels_decode_BGR_integer_units_without_visibility_filter(tmp_path):
    raw=np.array([[[1,32704,32800],[1,32896,32640],[0,65535,0]]],dtype=np.uint16)
    path=tmp_path/'flow.png';assert cv2.imwrite(str(path),raw)
    gt,valid=read_labels(path,'kitti')
    torch.testing.assert_close(gt,torch.tensor([[[[.5,-2.,-512.]],[[-1.,2.,511.984375]]]]),rtol=0,atol=0)
    assert valid.tolist()==[[[True,True,False]]]
    wrong=tmp_path/'uint8.png';assert cv2.imwrite(str(wrong),raw.astype(np.uint8))
    with pytest.raises(ValueError,match='uint16'):read_labels(wrong,'kitti')


def test_sintel_labels_use_original_little_endian_flow_and_finite_threshold(tmp_path):
    flow=np.array([[[.25,-1.5],[999.5,-999.5],[1000.,0.],[np.nan,1.]]],dtype=np.float32)
    path=tmp_path/'flow.flo';path.write_bytes(struct.pack('<fii',202021.25,4,1)+flow.astype('<f4').tobytes())
    gt,valid=read_labels(path,'sintel')
    assert valid.tolist()==[[[True,True,False,False]]]
    torch.testing.assert_close(gt[:,:,:,:3],torch.from_numpy(flow[:,:3].transpose(2,0,1)[None]),rtol=0,atol=0)


def test_npz_labels_require_native_dtype_geometry_and_all_valid_finite(tmp_path):
    flow=np.zeros((3,5,2),dtype=np.float32);valid=np.ones((3,5),dtype=np.bool_)
    path=tmp_path/'labels.npz';np.savez(path,flow=flow,valid=valid)
    gt,mask=read_labels(path,'npz');assert gt.shape==(1,2,3,5) and mask.shape==(1,3,5)
    for badflow,badvalid in [(flow.astype(np.float64),valid),(flow,valid[:,:4]),(np.full_like(flow,np.nan),valid)]:
        np.savez(path,flow=badflow,valid=badvalid)
        with pytest.raises(ValueError):read_labels(path,'npz')
    with pytest.raises(ValueError,match='Unknown'):read_labels(path,'guess')


def test_packet_odd_padding_and_exact_canonical_prior_binding(packet,prepared):
    assert set(packet)==INPUT_KEYS and packet['schema']==INPUT_SCHEMA
    assert packet['padding']==(25,26,27,28) and packet['native_hw']==(57,61)
    assert validate_packet(packet,expected_id='synthetic/0') is packet
    assert same(packet['original'],prepared.original)
    bad=deepcopy(packet);bad['motion_prior']['gather_xy_s4'][0,8,8,0]+=.25
    with pytest.raises(ValueError,match='Motion prior differs'):
        validate_packet(bad)
    bad=deepcopy(packet);bad['padding']=(26,25,27,28)
    with pytest.raises(ValueError,match='Motion prior geometry'):
        validate_packet(bad)


@pytest.mark.parametrize('extra',['gt','valid','teacher_features','target','baseline_output'])
def test_packet_strict_allowlist_prevents_labels_and_extra_state(packet,extra):
    packet[extra]=torch.zeros(1)
    with pytest.raises(ValueError,match='GT-free'):
        validate_packet(packet)


def test_packet_changed_taps_sigma_precision_graph_and_prior_graph_rejected(packet):
    mutations=[lambda p:p['observations'].__setitem__('cnn_pre_s2',p['observations']['cnn_pre_s2'][...,:-1]),
        lambda p:p.__setitem__('sigma',torch.zeros(64)),lambda p:p.__setitem__('original',p['original'].double()),
        lambda p:p.__setitem__('original',p['original'].requires_grad_()),
        lambda p:p['motion_prior'].__setitem__('evidence_s4',p['motion_prior']['evidence_s4'].requires_grad_())]
    for mutate in mutations:
        bad=deepcopy(packet);mutate(bad)
        with pytest.raises(ValueError):validate_packet(bad)


def test_relative_cache_can_move_and_labels_remain_outside_model_inputs(tmp_path,packet):
    source=tmp_path/'source';source.mkdir();path,row,labels=cached_case(source,packet)
    moved=tmp_path/'moved';shutil.copytree(source,moved)
    prepared,gt,valid=load_sample(moved/'manifest.json',row)
    assert set(prepared)=={'original','sigma','observations','baseline_flow','baseline_info','native_hw','padding','motion_prior'}
    assert 'gt' not in prepared and 'valid' not in prepared and same(gt,labels['gt']) and same(valid,labels['valid'])
    newlabels=deepcopy(labels);newlabels['gt'].fill_(37)
    atomic_torch(moved/'labels/case.pt',newlabels);changed=dict(row);changed['labels']=descriptor(moved/'labels/case.pt',relative_to=moved)
    again,newgt,_=load_sample(moved/'manifest.json',changed)
    assert same(prepared,again) and not same(gt,newgt)


def test_changed_cache_bytes_rejected_before_loading(tmp_path,packet):
    path,row,_=cached_case(tmp_path,packet)
    cache=tmp_path/row['input']['path'];cache.write_bytes(cache.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='bytes changed'):load_sample(path,row)


def test_label_extra_keys_wrong_shape_count_ID_and_valid_NaN_rejected(tmp_path,packet):
    path,row,labels=cached_case(tmp_path,packet)
    modifications=[lambda p:p.__setitem__('teacher',torch.zeros(1)),lambda p:p.__setitem__('id','other'),
        lambda p:p.__setitem__('gt',p['gt'][...,:-1]),lambda p:p.__setitem__('valid',p['valid'] & False),
        lambda p:p['gt'].__setitem__((0,0,1,1),float('nan'))]
    for modify in modifications:
        bad=deepcopy(labels);modify(bad);atomic_torch(tmp_path/'labels/case.pt',bad)
        checked=dict(row);checked['labels']=descriptor(tmp_path/'labels/case.pt',relative_to=tmp_path)
        with pytest.raises(ValueError):load_sample(path,checked)
    atomic_torch(tmp_path/'labels/case.pt',labels);checked=dict(row,valid_pixels=row['valid_pixels']-1)
    with pytest.raises(ValueError,match='denominator'):load_sample(path,checked)


def test_manifest_binds_exact_model_core_unique_ids_and_split(tmp_path,packet):
    path,row,_=cached_case(tmp_path,packet);_,good=load_manifest(path)
    alterations=[lambda p:p.__setitem__('model_abi','other'),lambda p:p['waft'].__setitem__('state_sha256','0'*64),
        lambda p:p['samples'].append(dict(p['samples'][0])),lambda p:p['samples'][0].__setitem__('split','test')]
    for alter in alterations:
        bad=deepcopy(good);alter(bad);atomic_json(path,bad)
        with pytest.raises(ValueError):load_manifest(path)


def test_packaged_canonical_KITTI_160_40_snapshot_and_example_are_identical():
    resource=files('waft_repair.motion').joinpath('kitti_split.json');value=json.loads(resource.read_text())
    dev='3 40 41 42 65 66 67 68 70 72 73 74 89 90 99 100 101 104 105 107 108 115 116 117 118 119 128 141 142 144 151 153 165 171 172 173 174 175 183 196'
    expected_dev=[f'{int(i):06d}' for i in dev.split()]
    assert value['dev_ids']==expected_dev
    assert value['train_ids']==[f'{i:06d}' for i in range(200) if f'{i:06d}' not in expected_dev]
    assert len(value['train_ids'])==160 and len(value['dev_ids'])==40
    example=Path(__file__).resolve().parents[2]/'examples/native/kitti-split.json'
    assert json.loads(example.read_text())==value


def test_make_kitti_spec_uses_resource_exact_IDs_without_loading_images_or_models(tmp_path):
    root=tmp_path/'KITTI'/'training';(root/'image_2').mkdir(parents=True);(root/'flow_occ').mkdir()
    for i in range(200):
        for suffix in ('10','11'):(root/'image_2'/f'{i:06d}_{suffix}.png').write_bytes(f'{i}-{suffix}'.encode())
        (root/'flow_occ'/f'{i:06d}_10.png').write_bytes(f'GT-{i}'.encode())
    output=tmp_path/'spec';receipt=make_kitti_spec(root.parent,output)
    value=json.loads((output/'images.json').read_text());split=json.loads((output/'split.json').read_text())
    assert receipt['train_pairs']==160 and receipt['dev_pairs']==40
    assert [v['id'].split('/')[-1] for v in value['samples'] if v['split']=='train']==split['train_ids']
    assert [v['id'].split('/')[-1] for v in value['samples'] if v['split']=='dev']==split['dev_ids']
    assert len(read_spec(output/'images.json'))==200
    with pytest.raises(FileExistsError):make_kitti_spec(root.parent,output)
