"""Synthetic CPU RGB integration and repair-only deployment loading."""
from copy import deepcopy
from dataclasses import asdict, fields, replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from waft_repair.motion.model import (
    MotionFeatureRepair, MotionRepairConfig, prepare_motion_prior)
from waft_repair.motion.native import (
    MotionNativeRepairSystem, PreparedMotion, load_legacy_motion_repair_checkpoint)
from waft_repair.motion.native import PreparedRepair
from motion_helpers import TinyCore, coordinates, sampler, TapEncoder, TapCNN, ZeroPadder


def fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(442)
        core = TinyCore()
        core.encoder, core.fnet = TapEncoder(), TapCNN()
    config = MotionRepairConfig(profile='test', fine_dim=4, s4_dim=8, s8_dim=8,
        s16_dim=12, s4_blocks=1, s8_blocks=1, s16_blocks=1, decoder_blocks=1)
    system = MotionNativeRepairSystem(core, MotionFeatureRepair(config), strict_official=False,
        checkpoint_refinement=False, padder_factory=ZeroPadder,
        coords_grid_fn=coordinates, bilinear_sampler_fn=sampler)
    one = torch.arange(3*57*61).reshape(1,3,57,61).remainder(256).float()
    return system, one, one.roll(2,-1)


def bytes_equal(a,b):
    return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def checkpoint(system):
    return dict(schema='waft-repair-redesign-training/v1',
        config=dict(stage='native',updates=128,auxiliary_distillation_weight=0.),
        model_config=asdict(system.restorer.config),repair=deepcopy(system.restorer.state_dict()),
        optimizer={'sentinel':'do not install'},scheduler={'sentinel':'do not restore'},
        completed_updates=128,binding={'fixed_cases':['toy/0']},
        cursor=dict(label='P128',fixed_order=['toy/0'],next_case_index=0,global_optimizer_updates=128),
        verified_implementation_binding={'test_source':'synthetic'},
        verified_core_binding={'state_digest':system.core_binding()['state_digest']},
        verified_observation_binding={'source':'synthetic'},
        torch_rng=torch.get_rng_state().clone(),python_rng=None,numpy_rng=None,device_rng=None)


class MotionRGBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_one_prefix_baseline5_and_reusable_repaired5(self):
        system,one,two=fixture();counts=dict(encoder=0,CNN=0,hidden=0,refine=0,R=0)
        def hook(key):
            def count(*_):counts[key]+=1
            return count
        handles=[m.register_forward_hook(hook(k)) for m,k in (
            (system.core.encoder,'encoder'),(system.core.fnet,'CNN'),
            (system.core.hidden_conv,'hidden'),(system.core.refine_net,'refine'),(system.restorer,'R'))]
        calls=[]
        def canonical(flow,info,**kwargs):
            calls.append((flow.device.type,info.device.type,flow.is_contiguous(),info.is_contiguous()))
            return prepare_motion_prior(flow,info,**kwargs)
        try:
            with patch('waft_repair.motion.native.prepare_motion_prior',side_effect=canonical):
                prepared=system.prepare_pair(one,two)
            self.assertIsInstance(prepared,PreparedMotion)
            self.assertEqual(calls,[('cpu','cpu',True,True)])
            self.assertEqual(counts,dict(encoder=2,CNN=2,hidden=1,refine=5,R=0))
            self.assertTrue(bytes_equal(prepared.baseline_flow,prepared.baseline_output['flow'][-1]))
            self.assertTrue(bytes_equal(prepared.baseline_info,prepared.baseline_output['info'][-1]))
            expected=prepare_motion_prior(prepared.baseline_flow,prepared.baseline_info,
                native_hw=prepared.native_hw,padding=prepared.padding)
            for field in fields(expected):
                a,b=getattr(prepared.motion_prior,field.name),getattr(expected,field.name)
                self.assertTrue(bytes_equal(a,b) if isinstance(a,torch.Tensor) else a==b)
            with torch.no_grad():output=system.forward_prepared(prepared,return_aux=True)
            self.assertEqual(counts,dict(encoder=2,CNN=2,hidden=2,refine=10,R=1))
            self.assertTrue(bytes_equal(output['features'],prepared.original))
            for key in ('flow','info'):
                self.assertEqual(len(output[key]),5)
                for a,b in zip(output[key],prepared.baseline_output[key]):self.assertTrue(bytes_equal(a,b))
            self.assertIn('normalized_residual',output['repair_aux'])
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            for handle in handles:handle.remove()

    def test_live_repair_gradient_while_baseline_and_all_core_stay_detached(self):
        system,one,two=fixture();prepared=system.prepare_pair(one.requires_grad_(),two)
        self.assertFalse(prepared.original.requires_grad)
        self.assertTrue(all(not x.requires_grad and x.grad_fn is None
            for x in (prepared.sigma,prepared.baseline_flow,prepared.baseline_info,*prepared.observations.values())))
        before={k:v.clone() for k,v in system.core.state_dict().items()}
        output=system.forward_prepared(prepared);output['flow'][-1].square().mean().backward()
        self.assertGreater(float(system.restorer.raw_own_head.weight.grad.abs().sum()),0.)
        self.assertGreater(float(system.restorer.spatial_head.weight.grad.abs().sum()),0.)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in system.core.parameters()))
        self.assertTrue(all(bytes_equal(before[k],v) for k,v in system.core.state_dict().items()))
        self.assertTrue(all(not m.training for m in system.core.modules()))
        self.assertIsNone(one.grad)

    def test_outer_autocast_keeps_observations_prior_and_outputs_exact_FP32(self):
        system,one,two=fixture();plain=system.prepare_pair(one,two)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            amp=system.prepare_pair(one,two)
            with torch.no_grad():result=system.forward_prepared(amp)
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        for a,b in ((plain.original,amp.original),(plain.sigma,amp.sigma),
                    (plain.baseline_flow,amp.baseline_flow),(plain.baseline_info,amp.baseline_info)):
            self.assertEqual(b.dtype,torch.float32);self.assertTrue(bytes_equal(a,b))
        self.assertTrue(bytes_equal(result['features'],plain.original))
        for key in ('flow','info'):
            for a,b in zip(result[key],plain.baseline_output[key]):self.assertTrue(bytes_equal(a,b))

    def test_predict_nonzero_repair_replays_prepared_and_requires_eval(self):
        system,one,two=fixture()
        with self.assertRaises(ValueError):system.predict(one,two)
        with torch.no_grad():
            system.restorer.raw_own_head.weight[0,0,0,0]=.02
            system.restorer.global_head.bias[1]=.003
        system.eval();prepared=system.prepare_pair(one,two)
        with torch.no_grad():expected=system.forward_prepared(prepared)
        actual=system.predict(one,two)
        self.assertTrue(bytes_equal(actual['features'],expected['features']))
        self.assertFalse(bytes_equal(actual['features'],prepared.original))
        for key in ('flow','info'):
            for a,b in zip(actual[key],expected[key]):self.assertTrue(bytes_equal(a,b))

    def test_plain_prepared_and_GT_keyword_and_live_baseline_are_rejected(self):
        system,one,two=fixture();prepared=system.prepare_pair(one,two)
        plain=PreparedRepair(prepared.original,prepared.sigma,prepared.observations,
            prepared.native_hw,prepared.padding)
        with self.assertRaises(TypeError):system.repair_features(plain)
        with self.assertRaises(TypeError):system.prepare_pair(one,two,gt=one)
        with self.assertRaises(ValueError):system.repair_features(replace(prepared,
            baseline_flow=prepared.baseline_flow.clone().requires_grad_()))


class MotionCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_safe_file_SHA_loads_only_repair_and_restores_no_RNG_optimizer_or_core(self):
        system,_,_=fixture();state=checkpoint(system)
        state['repair']['global_head.bias'].fill_(.05)
        core={k:v.clone() for k,v in system.core.state_dict().items()};rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'P128.pt';torch.save(state,path);sha=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('torch.optim.AdamW',side_effect=AssertionError('optimizer forbidden')), \
                 patch('torch.set_rng_state',side_effect=AssertionError('RNG restore forbidden')):
                receipt=load_legacy_motion_repair_checkpoint(system,path,expected_sha256=sha,
                    expected_binding=state['binding'])
        self.assertTrue(receipt['loaded_repair_only']);self.assertFalse(receipt['restored_core'])
        self.assertFalse(receipt['restored_optimizer']);self.assertFalse(receipt['restored_rng'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(torch.equal(system.restorer.global_head.bias,torch.full((64,),.05)))
        self.assertTrue(all(bytes_equal(core[k],v) for k,v in system.core.state_dict().items()))
        self.assertFalse(system.training);system.validate()

    def test_invalid_schema_config_cursor_core_or_parameter_cannot_partially_mutate_R(self):
        system,_,_=fixture();good=checkpoint(system);before=deepcopy(system.restorer.state_dict())
        def bad_schema(s):s['schema']='flow-feature-repair-training/v4'
        def bad_config(s):s['model_config']['fine_dim']+=1
        def bad_count(s):s['completed_updates']=127
        def bad_cursor(s):s['cursor']['global_optimizer_updates']=127
        def bad_core(s):s['verified_core_binding']['state_digest']='0'*64
        def bad_shape(s):s['repair']['global_head.bias']=torch.zeros(63)
        def bad_dtype(s):s['repair']['global_head.bias']=torch.zeros(64,dtype=torch.float64)
        def bad_finite(s):s['repair']['global_head.bias'][0]=float('nan')
        for mutate in (bad_schema,bad_config,bad_count,bad_cursor,bad_core,bad_shape,bad_dtype,bad_finite):
            state=deepcopy(good);state['repair']['raw_own_head.weight'].fill_(.01);mutate(state)
            with self.subTest(mutate=mutate.__name__),self.assertRaises(ValueError):
                load_legacy_motion_repair_checkpoint(system,state)
            self.assertTrue(all(bytes_equal(v,system.restorer.state_dict()[k]) for k,v in before.items()))

    def test_SHA_failure_and_mapping_SHA_contract_reject_before_install(self):
        system,_,_=fixture();state=checkpoint(system)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'P128.pt';torch.save(state,path)
            with self.assertRaises(ValueError):load_legacy_motion_repair_checkpoint(system,path,expected_sha256='0'*64)
        with self.assertRaises(ValueError):load_legacy_motion_repair_checkpoint(system,state,expected_sha256='0'*64)
        with self.assertRaises(ValueError):load_legacy_motion_repair_checkpoint(system,state,expected_sha256='invalid')
        with self.assertRaises(ValueError):load_legacy_motion_repair_checkpoint(system,state,expected_binding={'different':True})

    def test_unexpected_load_hook_rolls_back_every_repair_tensor_and_mode(self):
        system,_,_=fixture();state=checkpoint(system);state['repair']['global_head.bias'].fill_(.3)
        before=deepcopy(system.restorer.state_dict());load=system.restorer.load_state_dict
        def fail(*args,**kwargs):
            load(*args,**kwargs)
            raise RuntimeError('synthetic post-copy hook failure')
        with patch.object(system.restorer,'load_state_dict',side_effect=fail):
            with self.assertRaises(RuntimeError):load_legacy_motion_repair_checkpoint(system,state)
        self.assertTrue(system.training);self.assertTrue(system.restorer.training)
        self.assertTrue(all(bytes_equal(v,system.restorer.state_dict()[k]) for k,v in before.items()))
        system.validate()


if __name__=='__main__':unittest.main()
