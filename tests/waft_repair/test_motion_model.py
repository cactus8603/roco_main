"""CPU motion geometry, information boundary and live-gradient checks."""
from dataclasses import replace
import math
import unittest

import torch

from waft_repair.motion.model import (
    MotionFeatureRepair,MotionPrior,MotionRepairConfig,prepare_motion_prior,
    gather_frame2_to_frame1,splat_frame1_to_frame2,_deterministic_scatter_add)


def config():
    return MotionRepairConfig(profile='test',fine_dim=8,s4_dim=12,s8_dim=16,s16_dim=24,
                              s4_blocks=1,s8_blocks=1,s16_blocks=1,decoder_blocks=1)


def fixture(batch=1):
    # Odd offsets expose native/padded and stride2/stride4 half-cell mistakes.
    native=(57,61); padding=(25,26,27,28); hp=wp=112
    generator=torch.Generator().manual_seed(811)
    original=torch.randn((batch,2,64,56,56),generator=generator)
    sigma=torch.linspace(.5,1.5,64)
    taps={key:torch.randn((batch,2,channels,hp//stride,wp//stride),generator=generator)
          for key,channels,stride in [('cnn_pre_s2',64,2),('cnn_pre_s4',128,4),
                                       ('dav2_block6',384,14),('dav2_block12',384,14)]}
    flow=torch.zeros((batch,2,*native)); info=torch.zeros((batch,4,*native))
    prior=prepare_motion_prior(flow,info,native_hw=native,padding=padding)
    return original,sigma,taps,prior,native,padding


class MotionGeometryTests(unittest.TestCase):
    def test_forward_translation_with_odd_padding_both_directions(self):
        _,_,_,_,native,pad=fixture()
        flow=torch.zeros((1,2,*native)); flow[:,0]=4.
        prior=prepare_motion_prior(flow,torch.zeros((1,4,*native)),native_hw=native,padding=pad)
        x=torch.arange(28,dtype=torch.float32)[None,None,None,:].expand(1,1,28,28).contiguous()
        gathered=gather_frame2_to_frame1(x,prior)
        active=prior.gather_valid_s4
        expected=(x+1)*active
        # grid_sample's normalized-coordinate conversion has FP32 roundoff;
        # the physical endpoint itself is checked exactly below.
        torch.testing.assert_close(gathered,expected,rtol=0,atol=2e-6)
        torch.testing.assert_close(prior.gather_xy_s4[0,8,8],torch.tensor((9.,8.)),rtol=0,atol=0)
        splatted=splat_frame1_to_frame2(x,prior)
        covered=prior.collision_mass_s4>0
        torch.testing.assert_close(splatted,(x-1)*covered,rtol=0,atol=0)
        self.assertFalse(bool(covered[0,0,8,7]))  # frame2 first native lattice column is a hole.
        self.assertTrue(bool(covered[0,0,8,8]))
        self.assertLess(float(prior.evidence_s4[0,1,0,8,8]),0.)  # transported -f12, not f12.

    def test_zero_motion_uses_native_support_and_no_padding_evidence(self):
        _,_,_,prior,_,_=fixture()
        source=torch.ones((1,3,28,28))
        out=gather_frame2_to_frame1(source,prior)
        torch.testing.assert_close(out,prior.native_s4.expand(-1,3,-1,-1).float(),rtol=0,atol=0)
        self.assertEqual(float(out[:,:,0].sum()),0.)
        self.assertEqual(float(prior.collision_mass_s4.max()),1.)
        self.assertTrue(torch.equal(prior.gather_valid_s4,prior.native_s4))

    def test_subpixel_splat_collision_holes_and_source_gradients(self):
        _,_,_,_,native,pad=fixture()
        flow=torch.full((1,2,*native),10000.)
        # Two supported source lattice points land at exactly the same target.
        sources=((8,8),(8,9)); target=(9,10)
        for sy,sx in sources:
            ny,nx=4*sy-pad[2],4*sx-pad[0]
            flow[0,:,ny,nx]=torch.tensor((4*(target[1]-sx),4*(target[0]-sy)))
        prior=prepare_motion_prior(flow,torch.zeros((1,4,*native)),native_hw=native,padding=pad)
        self.assertEqual(float(prior.collision_mass_s4[0,0,target[0],target[1]]),2.)
        self.assertEqual(int((prior.collision_mass_s4>0).sum()),1)
        values=torch.zeros((1,1,28,28)); values[0,0,sources[0][0],sources[0][1]]=2.; values[0,0,sources[1][0],sources[1][1]]=4.
        values.requires_grad_(); result=splat_frame1_to_frame2(values,prior)
        self.assertEqual(float(result[0,0,target[0],target[1]]),3.)
        result.sum().backward()
        self.assertEqual(float(values.grad[0,0,sources[0][0],sources[0][1]]),.5)
        self.assertEqual(float(values.grad[0,0,sources[1][0],sources[1][1]]),.5)
        self.assertEqual(float(result[0,0,0,0]),0.)

    def test_fractional_translation_preserves_weight_mass(self):
        _,_,_,_,native,pad=fixture()
        flow=torch.zeros((1,2,*native)); flow[:,0]=2.; flow[:,1]=1.
        prior=prepare_motion_prior(flow,torch.zeros((1,4,*native)),native_hw=native,padding=pad)
        torch.testing.assert_close(prior.splat_weights.sum(1),prior.gather_valid_s4.flatten(2)[:,0].float(),rtol=0,atol=0)
        self.assertAlmostEqual(float(prior.collision_mass_s4.sum()),float(prior.gather_valid_s4.sum()),places=5)
        result=splat_frame1_to_frame2(torch.ones((1,2,28,28)),prior)
        torch.testing.assert_close(result,(prior.collision_mass_s4>0).expand(-1,2,-1,-1).float(),rtol=0,atol=0)

    def test_baseline_labels_or_live_graph_cannot_enter_prior(self):
        _,_,_,_,native,pad=fixture()
        flow=torch.zeros((1,2,*native),requires_grad=True); info=torch.zeros((1,4,*native))
        with self.assertRaises(ValueError): prepare_motion_prior(flow,info,native_hw=native,padding=pad)
        with self.assertRaises(TypeError): prepare_motion_prior(flow.detach(),info,native_hw=native,padding=pad,gt=flow)
        info[0,0,0,0]=math.nan
        with self.assertRaises(ValueError): prepare_motion_prior(flow.detach(),info,native_hw=native,padding=pad)

    def test_scatter_local_determinism_flags_restore_on_success_and_failure(self):
        previous=torch.are_deterministic_algorithms_enabled()
        warn=torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            for mode,warn_only in ((False,False),(False,True),(True,True)):
                torch.use_deterministic_algorithms(mode,warn_only=warn_only)
                _deterministic_scatter_add(torch.zeros(1,2),1,torch.tensor([[0,0]]),torch.ones(1,2))
                self.assertEqual(torch.are_deterministic_algorithms_enabled(),mode)
                self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(),warn_only)
                with self.assertRaises(RuntimeError):
                    _deterministic_scatter_add(torch.zeros(1,2),1,torch.tensor([[9,9]]),torch.ones(1,2))
                self.assertEqual(torch.are_deterministic_algorithms_enabled(),mode)
                self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(),warn_only)
        finally:
            torch.use_deterministic_algorithms(previous,warn_only=warn)


class MotionModelTests(unittest.TestCase):
    def test_zero_heads_preserve_identity_including_signed_zero(self):
        original,sigma,taps,prior,native,pad=fixture()
        original[...,0,0]=0.; original[...,0,1]=-0.
        model=MotionFeatureRepair(config())
        output,aux=model(original,sigma,taps,motion_prior=prior,native_hw=native,padding=pad,return_aux=True)
        self.assertTrue(torch.equal(output.contiguous().view(torch.uint8),original.contiguous().view(torch.uint8)))
        self.assertEqual(float(aux['normalized_residual'].abs().max()),0.)
        for head in (model.spatial_head,model.raw_own_head,model.raw_match_head): self.assertIsNone(head.bias)
        self.assertFalse(torch.cuda.is_initialized())

    def test_raw_full64_linear_bypass_preserves_amplitude(self):
        original,sigma,taps,prior,native,pad=fixture()
        model=MotionFeatureRepair(config())
        with torch.no_grad(): model.raw_own_head.weight[63,63,0,0]=.25
        output,aux=model(original,sigma,taps,motion_prior=prior,native_hw=native,padding=pad,return_aux=True)
        expected=.25*(original[:,:,63]-original[:,:,63].mean((-2,-1),keepdim=True))
        torch.testing.assert_close(output[:,:,63]-original[:,:,63],expected,rtol=2e-5,atol=3e-7)
        self.assertLess(float(aux['spatial_normalized_residual'].mean((-2,-1)).abs().max()),1e-7)
        self.assertEqual(float(aux['global_normalized_residual'].abs().max()),0.)

    def test_both_heads_and_body_receive_real_gradients_by_second_step(self):
        original,sigma,taps,prior,native,pad=fixture()
        model=MotionFeatureRepair(config()); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.)
        scale=sigma[None,None,:,None,None]
        target=original+.03*scale*(original.roll(1,-1)+.5)
        body_names=('fine_stem.weight','encoder4.0.mlp.0.weight','deep.0.mlp.0.weight')
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output=model(original,sigma,taps,motion_prior=prior,native_hw=native,padding=pad)
            loss=((output-target)/scale).square().mean(); loss.backward()
            self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
            self.assertGreater(float(model.spatial_head.weight.grad.abs().sum()),0.)
            self.assertGreater(float(model.global_head.weight.grad.abs().sum()),0.)
            for name in body_names:
                gradient=dict(model.named_parameters())[name].grad
                self.assertIsNotNone(gradient)
                if step==0: self.assertEqual(float(gradient.abs().max()),0.)
                else: self.assertGreater(float(gradient.abs().max()),0.)
            optimizer.step()

    def test_unsupported_raw_differences_are_zero_and_context_is_explicit(self):
        original,sigma,taps,prior,native,pad=fixture()
        model=MotionFeatureRepair(config())
        _,aux=model(original,sigma,taps,motion_prior=prior,native_hw=native,padding=pad,return_aux=True)
        support=aux['matched_support_s4'].expand_as(aux['aligned_raw_difference_s4'])
        self.assertEqual(float(aux['aligned_raw_difference_s4'][support==0].abs().sum()),0.)
        self.assertIn('not true backward',aux['frame2_context'])

    def test_batch2_and_prior_device_transfer_keep_shape(self):
        original,sigma,taps,prior,native,pad=fixture(batch=2)
        model=MotionFeatureRepair(config())
        output=model(original,sigma.repeat(2,1),taps,motion_prior=prior.to('cpu'),native_hw=native,padding=pad)
        self.assertEqual(output.shape,original.shape)
        self.assertTrue(torch.equal(output,original))

    def test_extra_teacher_caseid_fields_and_changed_geometry_are_rejected(self):
        original,sigma,taps,prior,native,pad=fixture(); model=MotionFeatureRepair(config())
        for extra in ('gt','teacher_features','case_id'):
            wrong=dict(taps); wrong[extra]=original
            with self.assertRaises(ValueError): model(original,sigma,wrong,motion_prior=prior,native_hw=native,padding=pad)
        with self.assertRaises(TypeError): model(original,sigma,taps,motion_prior=prior,native_hw=native,padding=pad,gt=original)
        with self.assertRaises(ValueError): model(original,sigma,taps,motion_prior=replace(prior,padding=(26,25,27,28)),native_hw=native,padding=pad)
        with self.assertRaises(ValueError): model(original.requires_grad_(),sigma,taps,motion_prior=prior,native_hw=native,padding=pad)

    def test_seed_preserves_callers_rng_and_default_capacity_is_bounded(self):
        torch.manual_seed(762); before=torch.get_rng_state().clone()
        model=MotionFeatureRepair()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        count=sum(p.numel() for p in model.parameters())
        self.assertGreater(count,2_000_000); self.assertLess(count,4_000_000)
        cost=model.cost_estimate((448,1344))
        self.assertEqual(cost['parameter_count'],count)
        self.assertEqual(cost['conv_linear_FLOPs'],2*cost['conv_linear_MACs'])
        self.assertEqual(cost['conv_linear_MACs'],sum(cost['per_layer_MACs'].values()))

    def test_static_mac_census_matches_actual_cpu_forward_shapes(self):
        original,sigma,taps,prior,native,pad=fixture(); model=MotionFeatureRepair(config())
        actual=[]; handles=[]
        def count(module,inputs,output):
            if isinstance(module,torch.nn.Conv2d):
                actual.append(output.numel()*(module.in_channels//module.groups)*math.prod(module.kernel_size))
            else:
                actual.append(output.numel()*module.in_features)
        for module in model.modules():
            if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear)):
                handles.append(module.register_forward_hook(count))
        try:
            model(original,sigma,taps,motion_prior=prior,native_hw=native,padding=pad)
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(sum(actual),model.cost_estimate((112,112))['conv_linear_MACs'])


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    unittest.main()
