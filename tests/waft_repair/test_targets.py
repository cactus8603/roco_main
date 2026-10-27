"""Portable CPU tests; these do not claim pretrained-WAFT EPE validation."""
from copy import deepcopy
from pathlib import Path
import csv
import json

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from waft_repair import targets as T


@pytest.fixture(autouse=True)
def small_cpu_threads():
    torch.set_num_threads(2)


def rgb_fixture(h=6, w=8):
    image = torch.arange(h*w).reshape(1, 1, h, w).float().expand(1, 3, h, w)
    pair = torch.stack((image, image+20), 1)
    return pair, torch.zeros(1, 2, h, w), torch.ones(1, h, w, dtype=torch.bool)


def test_fractional_forward_GT_pull_and_visibility_exact():
    pair, gt, valid = rgb_fixture()
    visible = valid.clone(); visible[:, 2, 2] = False
    gt[:, 0] = .5
    actual, mask = T.reconstruct_visible_rgb(pair, gt, valid, visible)
    # Linear intensity ramp admits an independent analytic interpolation.
    expected = pair.clone()
    expected[:, 0, :, :, :-1] = pair[:, 1, :, :, :-1]+.5
    expected[:, 0, :, 2, 2] = pair[:, 0, :, 2, 2]
    assert torch.equal(actual, expected)
    assert not mask[0, 2, 2] and not mask[:, :, -1].any()
    assert torch.equal(pair[:, 1], actual[:, 1])


def test_invalid_nonfinite_OOB_and_invisible_retain_original_bytes():
    pair, gt, valid = rgb_fixture()
    pair[:, 0, :, 0, 0] = -0.
    visible = valid.clone()
    valid[0, 1, 1] = False; visible[0, 2, 2] = False
    gt[0, 0, 3, 3] = float('nan'); gt[0, 1, 4, 4] = 99.
    gt[0, 0, 0, 0] = -1.
    actual, mask = T.reconstruct_visible_rgb(pair, gt, valid, visible)
    expected = pair[0, 0, :, ~mask[0]].numpy().view(np.uint32)
    assert np.array_equal(actual[0, 0, :, ~mask[0]].numpy().view(np.uint32), expected)
    assert torch.equal(actual[:, 1], pair[:, 1])


def test_visibility_is_required_and_zero_flow_is_valid():
    pair, gt, valid = rgb_fixture()
    with pytest.raises(ValueError, match='annotation'):
        T.reconstruct_visible_rgb(pair, gt, valid, None)
    actual, mask = T.reconstruct_visible_rgb(pair, gt, valid, valid)
    assert mask.all() and torch.equal(actual[:, 0], pair[:, 1])


def test_scoring_keeps_occluded_and_OOB_valid_GT():
    pair, gt, valid = rgb_fixture()
    gt[:, 0] = 1000.
    _, generation = T.reconstruct_visible_rgb(pair, gt, valid, torch.zeros_like(valid))
    assert not generation.any()
    metrics = T.flow_metrics(gt.clone(), gt, valid)
    assert metrics['EPE'] == 0 and metrics['valid_pixels'] == 48


class Padder:
    def __init__(self, shape, factor):
        h, w = shape[-2:]; dh, dw = (-h) % factor, (-w) % factor
        self._pad = (dw//2, dw-dw//2, dh//2, dh-dh//2)
    def pad(self, value):return F.pad(value, self._pad, mode='replicate')


def coordinates(n, h, w, device):
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    return torch.stack((x, y), 0).float()[None].repeat(n, 1, 1, 1)


def sampler(value, coords):
    h, w = value.shape[-2:]
    scale = coords.new_tensor((w-1, h-1))
    return F.grid_sample(value, 2*coords/scale-1, align_corners=True)


class ListConv(nn.Conv2d):
    def forward(self, value):return [super().forward(value)]


class DictConv(nn.Conv2d):
    def forward(self, value):return {'out':super().forward(value)}


class TinyCore(nn.Module):
    factor = 4
    def __init__(self):
        super().__init__();torch.manual_seed(19)
        self.encoder=nn.Conv2d(3,4,1,stride=2)
        self.fnet=ListConv(3,4,1,stride=2)
        self.fmap_conv=nn.Conv2d(8,64,1)
        self.hidden_conv=nn.Conv2d(128,4,1)
        self.warp_linear=nn.Conv2d(134,4,1)
        self.refine_net=DictConv(4,4,1)
        self.refine_transform=nn.Conv2d(8,4,1)
        self.flow_head=nn.Conv2d(4,6,1)
        self.upsample_weight=nn.Conv2d(4,36,1)
        self.eval().requires_grad_(False)
    def normalize_image(self, value):return value/255
    def upsample_data(self, flow, info, mask):
        return (F.interpolate(2*flow,scale_factor=2,mode='bilinear',align_corners=True),
                F.interpolate(info,scale_factor=2,mode='bilinear',align_corners=True))


def backend(core=None, **kw):
    return T.FrozenWAFTBackend(core or TinyCore(), Padder, coordinates, sampler, strict=False, **kw)


def test_cheap_cached_second_and_encoder_call_budget():
    b=backend();pair,gt,valid=rgb_fixture();original=b.encode_pair(pair)
    calls=[];handle=b.core.encoder.register_forward_hook(lambda *unused:calls.append(True))
    target=T.cheap_target(b,pair,gt,valid,valid,original=original);handle.remove()
    assert len(calls)==1 and target['optimizer_updates']==0
    assert torch.equal(target['teacher_features'][:,1],original.features[:,1])
    assert target['feature_projection'] is None
    wrong=T.FeaturePair(original.features,original.native_hw,original.padding,'another-render')
    with pytest.raises(ValueError,match='another RGB'):
        T.cheap_target(b,pair,gt,valid,valid,original=wrong)


def test_strict_backend_rejects_a_fixture_and_unfrozen_core():
    with pytest.raises(ValueError,match='full508'):
        T.FrozenWAFTBackend(TinyCore(),Padder,coordinates,sampler)
    core=TinyCore();core.train()
    with pytest.raises(ValueError,match='eval/frozen'):
        backend(core)


def test_full5_checkpoint_parity_live_coordinates_and_frozen_state():
    b=backend(checkpoint_refinement=False);pair,_,_=rgb_fixture(5,7);original=b.encode_pair(pair)
    states={k:v.clone()for k,v in b.core.state_dict().items()}
    calls=[]
    def spy(value,coords):calls.append(coords.requires_grad);return sampler(value,coords)
    b.bilinear_sampler=spy
    features=original.features.clone().requires_grad_()
    direct=b.solve(features,original.native_hw,original.padding)
    grad=torch.autograd.grad(direct['flow'][-1].square().mean(),features)[0]
    assert calls==[False,True,True,True,True]
    b.checkpoint_refinement=True;calls.clear()
    replay=b.solve(features,original.native_hw,original.padding)
    checked=torch.autograd.grad(replay['flow'][-1].square().mean(),features)[0]
    assert all(torch.equal(a,c)for key in ('flow','info')for a,c in zip(direct[key],replay[key]))
    assert torch.equal(grad,checked) and grad.abs().sum()>0
    assert len(replay['flow'])==len(replay['info'])==5
    assert replay['flow'][-1].shape==(1,2,5,7)
    assert all(torch.equal(states[k],v)for k,v in b.core.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in b.core.parameters())


def test_native_mask_and_original_sigma_use_original_native_centers_only():
    original=torch.ones(1,2,64,4,4)
    mask=T.native_mask(original,(5,5),(1,2,1,2))
    original.masked_fill_(~mask.expand_as(original),99.)
    original[:,:,0,1:3,1:3]=2.
    sigma,actual=T.original_sigma(original,(5,5),(1,2,1,2))
    assert torch.equal(mask,actual) and sigma[0]==2 and torch.equal(sigma[1:],torch.ones(63))


class ScalarBackend:
    """Analytic final output; earlier4 readouts deliberately disagree with GT."""
    def __init__(self,constant=False):self.calls=[];self.constant=constant
    def validate(self):return torch.device('cpu')
    def solve(self,features,native_hw,padding):
        self.calls.append(features.detach().clone())
        value=features[0,0,0,1,1]
        if self.constant:value=value*0+1
        flow=torch.stack((value,value*0)).reshape(1,2,1,1).expand(1,2,*native_hw)
        return dict(flow=[flow*0-100]*4+[flow],info=[flow*0]*5)


@pytest.mark.parametrize('constant',[False,True])
@pytest.mark.parametrize('steps',[32,256,512])
def test_dense_no_early_stop_exact_first_last_update_bound_and_first_best(constant,steps):
    b=ScalarBackend(constant);base=torch.ones(1,2,64,4,4)
    mask=T.native_mask(base,(5,5),(1,2,1,2));base.masked_fill_(~mask.expand_as(base),-0.)
    original=T.FeaturePair(base,(5,5),(1,2,1,2),'fixture')
    gt=torch.zeros(1,2,5,5);gt[:,0]=2.
    valid=torch.ones(1,5,5,dtype=torch.bool)
    result=T.expensive_target(b,original,gt,valid,steps=steps)
    assert result['optimizer_updates']==steps and len(result['history'])==steps+1
    assert result['method']==f'waft-dense-final-epe-{steps}-v1'
    assert len(b.calls)==steps+3  # initial no-grad +budget+1 search +selected no-grad
    assert result['selected_step']==min(result['history'],key=lambda r:r['EPE'])['step']
    outside=~mask.expand_as(base)
    assert np.array_equal(base[outside].numpy().view(np.uint32),result['teacher_features'][outside].numpy().view(np.uint32))
    assert result['selected_delta'].abs().max()<=.25
    if constant:
        assert result['selected_step']==0 and not result['qualified']
        assert torch.equal(result['teacher_features'],base)
    else:
        # Constant-sign gradient gives Adam's analytic first step lr*g/(|g|+eps).
        g=1/(1+1e-6);expected=1+.001*g/(g+1e-8)
        assert abs(float(b.calls[2][0,0,0,1,1])-expected)<1e-7
        if steps>=256:
            assert b.calls[-2][0,0,0,1,1]==1.25
            assert result['target_metrics']['EPE']==.75 and result['qualified']
        else:
            assert float(b.calls[-2][0,0,0,1,1])==pytest.approx(1.032,abs=2e-6)
            assert result['target_metrics']['EPE']==pytest.approx(.968,abs=2e-6)
            assert not result['qualified']


@pytest.mark.parametrize('steps',[0,31,64,128,513,True,32.0,'32',None])
def test_invalid_budget_is_rejected_before_backend_or_generation(steps):
    with pytest.raises(ValueError,match='32, 256, 512'):
        T.expensive_target(None,None,None,None,steps=steps)


def test_default512_and_short_budget_prefixes_preserve_feature_bytes():
    base=torch.ones(1,2,64,4,4)
    original=T.FeaturePair(base,(5,5),(1,2,1,2),'fixture')
    gt=torch.zeros(1,2,5,5);gt[:,0]=2.
    valid=torch.ones(1,5,5,dtype=torch.bool)
    full_backend=ScalarBackend()
    full=T.expensive_target(full_backend,original,gt,valid)
    for steps in (32,256,512):
        short_backend=ScalarBackend()
        short=T.expensive_target(short_backend,original,gt,valid,steps=steps)
        assert short['history']==full['history'][:steps+1]
        assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8))
                   for a,b in zip(short_backend.calls[:-1],full_backend.calls[:steps+2]))
        if steps==512:
            assert short['method']==T.EXPENSIVE_METHOD
            for key in ('teacher_features','selected_delta','sigma','native_mask'):
                assert torch.equal(short[key].view(torch.uint8),full[key].view(torch.uint8))
            assert short['selected_step']==full['selected_step']


def test_dense32_runs_complete_five_update_backend_with_frozen_weights():
    b=backend()
    pair,gt,valid=rgb_fixture(5,7)
    original=b.encode_pair(pair)
    before={key:value.clone() for key,value in b.core.state_dict().items()}
    calls=[]
    hook=b.core.flow_head.register_forward_hook(lambda *unused:calls.append(True))
    result=T.expensive_target(b,original,gt,valid,steps=32)
    hook.remove()
    # 35 solves plus the checkpoint recomputation for 32 backwards.
    assert len(calls)==5*(35+32)
    assert len(result['history'])==33 and 0<=result['selected_step']<=32
    assert all(torch.equal(before[key],value) for key,value in b.core.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in b.core.parameters())


def test_final_loss_is_vector_EPE_not_component_MSE():
    gt=torch.zeros(1,2,1,1);flow=torch.tensor([[[[3.]],[[4.]]]],requires_grad=True)
    loss=T.final_epe_loss(flow,gt,torch.ones(1,1,1,dtype=torch.bool))
    assert loss.item()==5.
    loss.backward();torch.testing.assert_close(flow.grad,torch.tensor([[[[.6]],[[.8]]]]))

