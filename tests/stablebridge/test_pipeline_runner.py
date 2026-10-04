"""CPU integration contracts: no model download, real data, GPU or training."""
from contextlib import ExitStack,contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from stablebridge import runner
from stablebridge.learning import validate_records
from stablebridge.pipeline import (StableBridge,PipelineConfig,_augment_pair,
                                   _orient_reverse_displacement,
                                   _restore_augmented_displacement)
from stablebridge.util import save_json,save_npz,sha256


def identity_inference(pair):
    h,w=pair['image0'].shape[:2]
    baseline=np.zeros((2,h,w),np.float32)
    mask=np.zeros((h,w),bool);mask[0,1]=True
    features=np.zeros((1,1,12),np.float32);features[...,8:10]=1
    arm={'output':baseline.copy(),'query_mask':mask,'accepted':np.zeros_like(mask),
         'query_xy':np.array([[1,0]],np.float32),'candidates':np.zeros((1,1,2),np.float32),
         'candidate_features':features,'candidate_valid':np.ones((1,1),bool),
         'candidate_costs':np.zeros((1,1),np.float32),'best_indices':np.zeros(1,np.int64),
         'candidate_names':['identity'],'candidate_groups':['current_pair'],
         'diagnostics':{'config':{'max_update_px':32,'require_baseline_support':True}}}
    return {'baseline':baseline,'reference_predictions':[baseline]*3,'arms':{'solved':arm},
            'query_mask':mask,'costs':{'total_seconds':0},'query':{},'snapshot':{}}


@contextmanager
def fake_runtime(root,mode='M0',frames='center'):
    """Exercise real runner IO/metrics/report, replacing only data/model inference."""
    root=Path(root)
    (root/'configs/stablebridge').mkdir(parents=True)
    save_json(root/'configs/stablebridge/data_models_v1.json',{'datasets':{'spring':{
        'rgb_root':str(root),'flow_root':str(root),'disp_root':str(root)}}})
    image_file=root/'rgb.bin';image_file.write_bytes(b'original-rgb')
    gt_file=root/'gt.bin';gt_file.write_bytes(b'original-gt')
    trace=[]
    clips={'clips':[{'scene':'0001','legacy_center_frame':2,'development_role':'fit',
                     'stereo_query_frames':[1,2,3], 'flow_query_pairs':[[1,2],[2,3]]}]}
    save_json(root/'clips.json',clips)
    config={'study':'E01/test','clips_manifest':'clips.json','tasks':['flow'],'frames':frames,
            'pipeline':{'mode':mode},'profiles':[{'name':'clean'}],'arms':['solved']}
    config_path=root/'config.json';save_json(config_path,config)
    class Provider:
        def read_pair(self,task,scene,frame,context_hw,**kwargs):
            trace.append(('input',frame))
            return {'image0':np.zeros((2,3,3),np.uint8),'image1':np.zeros((2,3,3),np.uint8),
                    'origin_xy':(10,20),'roi_xyhw':(10,20,2,3),
                    'metadata':{'task':task,'scene':scene,'source_frame':frame,'target_frame':frame+1,
                                'source_view':'left','target_view':'left','gt_read':False,
                                'inputs':[{'path':str(image_file),'file_sha256':sha256(image_file)}]}}
        def read_gt(self,task,scene,frame,**kwargs):
            trace.append(('gt',frame))
            return np.zeros((4,2,2,3),np.float32)
        def gt_path(self,*args,**kwargs):return gt_file
    class Adapter:
        def __init__(self,task,*args,**kwargs):self.task=task;self.context_hw=(2,3)
    class Bridge:
        def __init__(self,backbone,config,*args):self.config=config;self.memory=[]
        def reset(self,namespace):self.memory=[]
        def predict(self,pair,arms):
            if 'gt' in pair or pair['metadata'].get('gt_read'):raise AssertionError('GT leaked into inference')
            frame=pair['metadata']['source_frame']
            trace.append(('predict',frame,tuple(self.memory)))
            result=identity_inference(pair)
            if self.config.mode=='M1':self.memory.append(frame)
            return result
    models={task:{'path':str(root/(task+'.pth')),'sha256':'a'*64} for task in ('stereo','flow')}
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner,'ROOT',root))
        stack.enter_context(patch.object(runner,'resolve_models',return_value=models))
        stack.enter_context(patch.object(runner,'setup_device'))
        stack.enter_context(patch.object(runner.SpringProvider,'from_registry',return_value=Provider()))
        stack.enter_context(patch.object(runner,'CroCoAdapter',Adapter))
        stack.enter_context(patch.object(runner,'StableBridge',Bridge))
        stack.enter_context(patch.object(runner,'event'))
        yield SimpleNamespace(config=config_path,run=root/'run',trace=trace,image=image_file,gt=gt_file)


class PipelineRunnerTests(unittest.TestCase):
    def test_fixed_hstar_sample_is_deterministic_bounded_and_subset(self):
        mask=np.ones((8,9),bool);mask[0,0]=False
        first=runner._fixed_mask_sample(mask,11,'case',7)
        second=runner._fixed_mask_sample(mask,11,'case',7)
        self.assertEqual(first.sum(),11)
        np.testing.assert_array_equal(first,second)
        self.assertTrue(np.all(~first|mask))

    def test_privileged_donor_uses_nearest_finite_whole_gt_vector(self):
        baseline=np.array([[[2.,9.]],[[1.,9.]]],np.float32)
        gt=np.full((4,2,1,2),np.nan,np.float32)
        gt[0,:,0,0]=[8,8];gt[1,:,0,0]=[3,1]
        donor=runner._privileged_gt_donor(baseline,gt)
        np.testing.assert_array_equal(donor[:,0,0],[3,1])
        np.testing.assert_array_equal(donor[:,0,1],baseline[:,0,1])

    def test_privileged_location_uses_all_reference_operators_and_excludes_query(self):
        u0=np.full((2,2,3),4.,np.float32);local=u0.copy();local[:,0,1]=0
        gt=np.zeros((4,2,2,3),np.float32);query=np.zeros((2,3),bool);query[1,2]=True
        # U0 is wrong everywhere; the local operator repairs only (x=1,y=0).
        mask=runner._privileged_repairable_support_mask((u0,local),gt,query,1.)
        expected=np.zeros((2,3),bool);expected[0,1]=True
        np.testing.assert_array_equal(mask,expected)

    def test_support_donor_record_replays_only_used_anchor_decisions(self):
        baseline=np.zeros((2,3,5),np.float32)
        local=baseline.copy();local[0,1,1]=1;local[0,1,3]=1
        query=np.zeros((3,5),bool);query[1,2]=True
        anchors=np.zeros_like(query);anchors[1,1]=True;anchors[1,3]=True;anchors[0,0]=True
        config=SimpleNamespace(source_neighbors=((-1,0),(1,0)),movement_penalty=0.,
                               max_update_px=32.,min_improvement=.015)
        class Rematcher:
            def __init__(self):self.config=config
            def score(self,prepared,xy,candidates,identity):
                shape=candidates.shape[:2]
                raw=np.full(shape,.5,np.float32);raw[1]=.1
                valid=np.ones(shape,bool);delta=np.linalg.norm(candidates-identity[None],axis=-1)
                return raw,valid,raw.copy(),raw.copy(),delta
        bridge=SimpleNamespace(rematcher=Rematcher())
        solved=baseline.copy();solved[0,1,1]=1;solved[0,1,3]=1
        reverse_identity=np.zeros_like(baseline);reverse_local=np.zeros_like(baseline)
        reverse_local[0,1,2]=-1;reverse_local[0,1,4]=-1
        record=runner._support_donor_record(
            bridge,{},baseline,(local,),anchors,query,solved,(reverse_identity,reverse_local))
        np.testing.assert_array_equal(record['support_xy'],[[1,1],[3,1]])
        np.testing.assert_array_equal(record['best_indices'],[1,1])
        np.testing.assert_array_equal(record['changed'],[True,True])
        self.assertEqual(record['candidate_names'],['identity','local_0'])
        self.assertEqual(record['unique_support_sites'],2)
        np.testing.assert_array_equal(record['cycle_valid'],True)
        np.testing.assert_allclose(record['cycle_cost'],0)

    def test_support_donor_record_contains_action_specific_stability(self):
        baseline=np.zeros((2,3,5),np.float32)
        local=baseline.copy();local[0,1,1]=1;local[0,1,3]=1
        query=np.zeros((3,5),bool);query[1,2]=True
        anchors=np.zeros_like(query);anchors[1,1]=True;anchors[1,3]=True
        config=SimpleNamespace(source_neighbors=((-1,0),(1,0)),movement_penalty=0.,
                               max_update_px=32.,min_improvement=.015)
        class Rematcher:
            def __init__(self):self.config=config
            def score(self,prepared,xy,candidates,identity):
                shape=candidates.shape[:2];raw=np.full(shape,.5,np.float32);raw[1]=.1
                valid=np.ones(shape,bool);delta=np.linalg.norm(candidates-identity[None],axis=-1)
                return raw,valid,raw.copy(),raw.copy(),delta
        # roll_x8 is out of range for this tiny synthetic grid, so validity is
        # expected to fail closed while the observations remain well-shaped.
        bridge=SimpleNamespace(rematcher=Rematcher());solved=local.copy()
        record=runner._support_donor_record(
            bridge,{},baseline,(local,),anchors,query,solved,
            augmentation_fields={'roll_x8':(baseline.copy(),local.copy())})
        self.assertEqual(record['augmentation_names'],['roll_x8'])
        self.assertEqual(record['stability_cost'].shape,(1,2,2))
        self.assertFalse(record['stability_valid'].any())

    def test_support_donor_record_contains_action_native_margin(self):
        baseline=np.zeros((2,3,5),np.float32);local=baseline.copy();local[0,1,1]=1;local[0,1,3]=1
        query=np.zeros((3,5),bool);query[1,2]=True
        anchors=np.zeros_like(query);anchors[1,1]=True;anchors[1,3]=True
        config=SimpleNamespace(source_neighbors=((-1,0),(1,0)),movement_penalty=0.,
                               max_update_px=32.,min_improvement=.015)
        class Rematcher:
            def __init__(self):self.config=config
            def score(self,prepared,xy,candidates,identity):
                raw=np.linalg.norm(candidates,axis=-1).astype(np.float32)
                valid=np.ones(raw.shape,bool);delta=np.linalg.norm(candidates-identity[None],axis=-1)
                return raw,valid,raw.copy(),raw.copy(),delta
        predictions=(SimpleNamespace(raw_uncertainty=np.ones((3,5),np.float32)),
                     SimpleNamespace(raw_uncertainty=np.full((3,5),2.,np.float32)))
        record=runner._support_donor_record(
            SimpleNamespace(rematcher=Rematcher()),{},baseline,(local,),anchors,query,baseline.copy(),
            task='stereo',action_predictions=predictions,action_prepared=({},{}))
        self.assertTrue(record['action_native_available'])
        self.assertEqual(record['native_margin'].shape,(2,2))
        np.testing.assert_array_equal(record['native_margin_valid'],True)
        np.testing.assert_array_equal(record['native_uncertainty'][0],1.)
        np.testing.assert_array_equal(record['native_uncertainty'][1],2.)

    def test_action_predictions_without_prepared_features_do_not_enable_native_evidence(self):
        baseline=np.zeros((2,3,5),np.float32);local=baseline.copy()
        query=np.zeros((3,5),bool);query[1,2]=True
        anchors=np.zeros_like(query);anchors[1,1]=True
        config=SimpleNamespace(source_neighbors=((-1,0),),movement_penalty=0.,
                               max_update_px=32.,min_improvement=.015)
        class Rematcher:
            def __init__(self):self.config=config
            def score(self,prepared,xy,candidates,identity):
                raw=np.zeros(candidates.shape[:2],np.float32);valid=np.ones(raw.shape,bool)
                delta=np.linalg.norm(candidates-identity[None],axis=-1)
                return raw,valid,raw.copy(),raw.copy(),delta
        predictions=(SimpleNamespace(raw_uncertainty=np.ones((3,5),np.float32)),
                     SimpleNamespace(raw_uncertainty=np.ones((3,5),np.float32)))
        record=runner._support_donor_record(
            SimpleNamespace(rematcher=Rematcher()),{},baseline,(local,),anchors,query,baseline,
            task='stereo',action_predictions=predictions,action_prepared=())
        self.assertFalse(record['action_native_available'])
        self.assertFalse(record['native_valid'].any())

    def test_stereo_reverse_orientation_unflips_and_negates_horizontal_motion(self):
        flipped=np.array([[[[-2,-3,-4]]],[[[0,0,0]]]],np.float32).reshape(2,1,3)
        native=_orient_reverse_displacement('stereo',flipped)
        np.testing.assert_array_equal(native[0,0],[4,3,2])
        np.testing.assert_array_equal(native[1,0],[0,0,0])
        np.testing.assert_array_equal(_orient_reverse_displacement('flow',flipped),flipped)
        with self.assertRaises(ValueError):PipelineConfig(diagnostic_reverse_pairs=1)

    def test_shared_translation_augmentation_round_trips_field_and_images(self):
        image=np.arange(10*12*3,dtype=np.uint16).reshape(10,12,3).astype(np.uint8)
        first,second=_augment_pair(image,image.copy(),'roll_x8')
        np.testing.assert_array_equal(first,second)
        field=np.arange(2*10*12,dtype=np.float32).reshape(2,10,12)
        augmented=np.roll(field,(0,8),axis=(1,2))
        np.testing.assert_array_equal(_restore_augmented_displacement(augmented,'roll_x8'),field)
        with self.assertRaises(ValueError):PipelineConfig(diagnostic_augmentations=('unknown',))

    def test_without_rematch_uses_first_legal_support_only(self):
        baseline=np.zeros((2,1,3),np.float32);mask=np.zeros((1,3),bool);mask[0,1]=True
        candidates=np.zeros((4,1,2),np.float32)
        candidates[1,0]=[40,0];candidates[2,0]=[2,0];candidates[3,0]=[1,0]
        result={'output':baseline.copy(),'query_mask':mask,'accepted':np.zeros_like(mask),
            'query_xy':np.array([[1,0]],np.float32),'candidates':candidates,
            'candidate_valid':np.ones((4,1),bool),'best_indices':np.zeros(1,np.int64),
            'candidate_names':['identity','support_bad','support_first','support_second'],
            'diagnostics':{'config':{'max_update_px':32.,'require_baseline_support':True}}}
        direct=runner._without_rematch(result,baseline)
        np.testing.assert_array_equal(direct['output'][:,0,1],[2,0])
        self.assertEqual(direct['best_indices'][0],2)
        self.assertTrue(direct['accepted'][0,1])
        self.assertFalse(direct['diagnostics']['target_cost_used_for_selection'])

    def test_pipeline_rejects_labels_before_any_forward(self):
        backbone=SimpleNamespace(task='flow',predict=lambda *a,**kw:self.fail('Forward called after label injection'))
        bridge=StableBridge(backbone,PipelineConfig())
        bridge.reset('scene/flow/clean')
        for pair in ({'gt':np.zeros(1)},{'metadata':{'gt_read':True}}):
            with self.assertRaisesRegex(ValueError,'labels'):
                bridge.predict(pair)

    def test_candidate_oracle_respects_support_and_step_and_does_not_mutate_actual(self):
        pair={'image0':np.zeros((2,3,3),np.uint8)}
        inferred=identity_inference(pair)
        base=inferred['baseline'];base[0]=8
        mask=np.zeros((2,3),bool);mask[0]=True
        candidate=np.zeros((4,3,2),np.float32)
        candidate[...,0]=np.array([8,4,0,-9])[:,None]
        features=np.zeros((4,3,12),np.float32)
        features[...,5]=np.abs(candidate[...,0]-8)/32
        features[...,8:10]=1
        features[2,:,8]=0  # Perfect GT candidate lacks image evidence.
        features[:,1,9]=0  # No baseline support: preserve identity at query 2.
        arm=inferred['arms']['solved']
        arm.update({'output':base.copy(),'query_mask':mask,'accepted':np.zeros_like(mask),
                    'query_xy':np.array([[0,0],[1,0],[2,0]],np.float32),
                    'candidates':candidate,'candidate_features':features,
                    'diagnostics':{'config':{'max_update_px':4,'require_baseline_support':True}}})
        inferred['query_mask']=mask
        gt=np.zeros((4,2,2,3),np.float32);gt[:,0,0,2]=np.nan
        original=arm['output'].copy()
        identity={'task':'flow','scene':'0001','frame':1,'profile':'clean','split':'fit','case_id':'x'}
        rows,outputs,records,hard=runner._case_rows(identity,inferred,gt)
        np.testing.assert_array_equal(outputs['solved_candidate_oracle'][0,0,:2],[4,8])
        np.testing.assert_array_equal(outputs['solved'],original)
        np.testing.assert_array_equal(arm['output'],original)
        self.assertEqual(rows[1]['selected_pixels'],2)
        self.assertEqual(rows[2]['oracle_scope'],'decision_candidates_at_snapshot')
        self.assertEqual(len(records),1)

    def test_gt_is_read_after_inference_and_final_report_counts_match(self):
        with tempfile.TemporaryDirectory() as temporary,fake_runtime(temporary) as env:
            runner.run_experiment(env.config,env.run,device='cpu')
            steps=[item[0] for item in env.trace]
            self.assertLess(steps.index('predict'),steps.index('gt'))
            manifest=json.loads((env.run/'manifest.json').read_text())
            report=json.loads((env.run/'report.json').read_text())
            self.assertEqual(manifest['expected_cases'],1)
            self.assertEqual(manifest['expected_metric_rows'],3)
            self.assertEqual(report['metric_rows'],3)
            self.assertEqual(report['completeness'],'complete_by_row_count')
            self.assertEqual(report['run_status']['state'],'completed')

    def test_m1_resume_replays_prefix_before_missing_case(self):
        with tempfile.TemporaryDirectory() as temporary,fake_runtime(temporary,'M1','three') as env:
            runner.run_experiment(env.config,env.run,device='cpu')
            (env.run/'metrics/flow_0001_0003_clean.json').unlink()
            env.trace.clear()
            runner.run_experiment(env.config,env.run,device='cpu')
            predicted=[item for item in env.trace if item[0]=='predict']
            self.assertEqual(predicted,[('predict',1,()),('predict',2,(1,)),('predict',3,(1,2))])
            self.assertEqual([item for item in env.trace if item[0]=='gt'],[('gt',3)])
            report=json.loads((env.run/'report.json').read_text())
            self.assertEqual(report['metric_rows'],9)
            self.assertEqual(report['expected_metric_rows'],9)

    def test_m0_resume_rejects_changed_rgb(self):
        with tempfile.TemporaryDirectory() as temporary,fake_runtime(temporary) as env:
            runner.run_experiment(env.config,env.run,device='cpu')
            env.image.write_bytes(b'changed-rgb')
            with self.assertRaises(ValueError):
                runner.run_experiment(env.config,env.run,device='cpu')

    def test_m1_resume_rejects_changed_evaluation_gt(self):
        with tempfile.TemporaryDirectory() as temporary,fake_runtime(temporary,'M1') as env:
            runner.run_experiment(env.config,env.run,device='cpu')
            env.gt.write_bytes(b'changed-gt')
            with self.assertRaises(ValueError):
                runner.run_experiment(env.config,env.run,device='cpu')

    def test_collect_different_studies_with_same_run_basename_keeps_query_ids_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            runs=[root/'S01/runs/pilot_v1',root/'S02/runs/pilot_v1']
            features=np.zeros((2,12),np.float32);features[:,8:10]=1;features[1,5]=1/32
            for path in runs:
                save_npz(path/'supervision/a.npz',features=features,baseline_error=np.array([2.,2.]),
                         candidate_error=np.array([2.,1.]),candidate_index=np.array([0,1]),
                         query_id=np.array(['same-query','same-query']),scene=np.array(['0001','0001']),
                         split=np.array(['fit','fit']))
            destination=root/'combined.npz'
            runner.collect_supervision(runs,destination)
            with np.load(destination,allow_pickle=False) as file:records=dict(file)
            self.assertEqual(len(np.unique(records['query_id'])),2)
            validate_records(records)

    def test_collect_rejects_repeated_run_argument(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'one-run'
            features=np.zeros((1,12),np.float32);features[:,8:10]=1
            save_npz(run/'supervision/a.npz',features=features,baseline_error=np.array([2.]),
                     candidate_error=np.array([2.]),candidate_index=np.array([0]),query_id=np.array(['query']),
                     scene=np.array(['0001']),split=np.array(['fit']))
            with self.assertRaises(ValueError):
                runner.collect_supervision([run,run],root/'out.npz')


if __name__=='__main__':unittest.main(verbosity=2)
