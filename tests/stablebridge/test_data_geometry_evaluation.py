"""CPU contract tests with synthetic Spring files; no unseen scene pixels."""
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
import torch

from stablebridge.data import SpringProvider, apply_corruption, GT_OFFSETS
from stablebridge.geometry import (PixelTransform, pixel_grid, sample_numpy, sample_torch,
                                   native_displacement_from_local)
from stablebridge.evaluation import (error_map, gt_valid_mask, evaluate_pair,
                                     aggregate_scene_macro, fixed_hard_mask)


class SpringProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.provider = SpringProvider(self.root/"rgb", self.root/"flow", self.root/"disp")
        self.image = np.arange(6*8*3, dtype=np.uint8).reshape(6,8,3)
        for frame in (1,2):
            for view in ("left","right"):
                path = self.provider.image_path("0001", frame, view)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(self.image).save(path)
        self.disparity = np.arange(12*16, dtype=np.float32).reshape(12,16)+8
        for task in ("stereo","flow"):
            for view in ("left","right"):
                path = self.provider.gt_path(task, "0001", 1, view)
                path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(path, "w") as file:
                    if task == "stereo":
                        file.create_dataset("disparity", data=self.disparity)
                    else:
                        file.create_dataset("flow", data=np.stack((self.disparity, -self.disparity), axis=-1))

    def tearDown(self):
        self.temp.cleanup()

    def test_provider_registry_paths(self):
        registry = self.root/"registry.json"
        registry.write_text(json.dumps({"datasets":{"spring": {
            "rgb_root":str(self.root/"rgb"), "flow_root":str(self.root/"flow"),
            "disp_root":str(self.root/"disp")}}}))
        provider = SpringProvider.from_registry(registry)
        np.testing.assert_array_equal(provider.read_image(1,1), self.image)
        with self.assertRaises(ValueError):
            provider.gt_path("stereo", 1,1, split="test")

    def test_min4_crop_order_sign_and_native_units(self):
        roi = (2,1,3,4)
        for task in ("stereo","flow"):
            actual = self.provider.read_gt(task, 1,1,roi_xyhw=roi)
            raw = self.disparity[2:8,4:12]
            expected = np.stack([np.stack((-raw[dy::2,dx::2], np.zeros((3,4))))
                                 if task == "stereo" else
                                 np.stack((raw[dy::2,dx::2], -raw[dy::2,dx::2]))
                                 for dy,dx in GT_OFFSETS]).astype(np.float32)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.shape, (4,2,3,4))
        right = self.provider.read_gt("stereo",1,1,view="right",roi_xyhw=roi)
        self.assertEqual(right[0,0,0,0], self.disparity[2,4])

    def test_pair_never_reads_gt_and_corrupts_full_image_before_crop(self):
        with patch.object(SpringProvider,"read_gt",side_effect=AssertionError("GT leak")):
            pair = self.provider.read_pair("flow",1,1,(2,4), profile="gaussian_noise",seed=72)
        self.assertNotIn("gt",pair)
        self.assertFalse(pair["metadata"]["gt_read"])
        self.assertEqual(pair["metadata"]["information_cutoff_event"],2)
        expected,_ = apply_corruption(self.image,"gaussian_noise",seed=72,scene=1,frame=1,view="left")
        np.testing.assert_array_equal(pair["image0"], expected[2:4,2:6])
        np.testing.assert_array_equal(self.provider.read_image(1,1),self.image)
        self.assertEqual(len(pair["metadata"]["inputs"][0]["file_sha256"]),64)
        with self.assertRaises(ValueError):
            self.provider.read_pair("stereo",1,1,(7,8))

    def test_determinism_and_explicit_correlations(self):
        base = dict(seed=12,scene=1,frame=1,view="left")
        a,_ = apply_corruption(self.image,"gaussian_noise",**base)
        b,_ = apply_corruption(self.image,"gaussian_noise",**base)
        np.testing.assert_array_equal(a,b)
        independent,_ = apply_corruption(self.image,"gaussian_noise",**{**base,"frame":2})
        self.assertFalse(np.array_equal(a,independent))
        persistent = {"temporal_correlation":"persistent","view_correlation":"shared"}
        a,_ = apply_corruption(self.image,"gaussian_noise",options=persistent,**base)
        b,_ = apply_corruption(self.image,"gaussian_noise",options=persistent,
                              **{**base,"view":"right","frame":2})
        np.testing.assert_array_equal(a,b)

    def test_persistent_overlay_and_recovery_event(self):
        a, am = apply_corruption(self.image,"persistent_overlay",seed=19,frame=1)
        b, bm = apply_corruption(self.image,"persistent_overlay",seed=19,frame=9)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(am["rectangle_xywh"],bm["rectangle_xywh"])
        self.assertEqual(am["attachment"],"image_plane_not_scene_surface")
        with self.assertRaises(ValueError):
            apply_corruption(self.image,"recovery")
        noisy,_ = apply_corruption(self.image,"recovery",frame=2,options={"recovery_frame":3})
        clean,meta = apply_corruption(self.image,"recovery",frame=3,options={"recovery_frame":3})
        self.assertFalse(np.array_equal(noisy,self.image))
        np.testing.assert_array_equal(clean,self.image)
        self.assertEqual(meta["sigma_01"],0.)


class GeometryTests(unittest.TestCase):
    def test_pixel_centres_and_distinct_endpoint_transforms(self):
        transform = PixelTransform(origin_xy=(100,200),scale_xy=(2,3),pad_xy=(4,5),native_hw=(32,64))
        xy = np.array([[100,200],[110.25,204.75]])
        np.testing.assert_allclose(transform.local_to_native(transform.native_to_local(xy)),xy)
        np.testing.assert_allclose(transform.patch_to_native(transform.native_to_patch(xy,16),16),xy)
        native = PixelTransform()
        np.testing.assert_allclose(native.native_to_patch([[7.5,7.5]],16),[[0,0]])
        grid = pixel_grid(2,3,origin_xy=(12,14))
        np.testing.assert_array_equal(grid[1,2],[14,15])
        source = PixelTransform(origin_xy=(100,0))
        target = PixelTransform(origin_xy=(80,0),scale_xy=(2,2))
        flow = native_displacement_from_local([[10,3]],[[20.5,6.5]],source,target)
        np.testing.assert_allclose(flow,[[-20,0]])

    def test_numpy_sampling_never_counts_padding_or_oob_as_observation(self):
        field = np.arange(12,dtype=np.float32).reshape(3,4)
        mask = np.ones((3,4),bool)
        mask[1,2] = False
        xy = np.array([[0,0],[3,2],[1,1],[1.5,1],[-.001,0],[3.001,2],[np.nan,1]])
        result,valid = sample_numpy(field,xy,mask)
        np.testing.assert_array_equal(valid,[True,True,True,False,False,False,False])
        np.testing.assert_allclose(result,[0,11,5,0,0,0,0])

    def test_torch_numpy_sampling_parity_including_nan_features(self):
        torch.set_num_threads(1)
        field = np.arange(24,dtype=np.float32).reshape(2,3,4)
        field[:,2,1] = np.nan
        xy = np.array([[[0,0],[3,2],[1.25,.75]],[[0,2],[1,2],[-1,0]]],dtype=np.float32)
        n_values,n_valid = sample_numpy(field,xy)
        t_values,t_valid = sample_torch(torch.from_numpy(field[None]),torch.from_numpy(xy[None]))
        np.testing.assert_array_equal(t_valid[0].numpy(),n_valid)
        np.testing.assert_allclose(t_values[0].numpy(),n_values,atol=2e-6)
        single = torch.tensor([[[[3.]]]])
        out,valid = sample_torch(single,torch.tensor([[[0.,0.],[.1,0.]]]))
        np.testing.assert_allclose(out.numpy(),[[[3,0]]])
        np.testing.assert_array_equal(valid.numpy(),[[True,False]])


class EvaluationTests(unittest.TestCase):
    def test_direct_parity_with_released_croco_spring_scorer(self):
        path = Path(__file__).resolve().parents[2]/"research/croco_trusted_first/vendor/croco/stereoflow/criterion.py"
        spec = importlib.util.spec_from_file_location("stablebridge_vendor_criterion_parity",path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(1729)
        for task in ("stereo","flow"):
            gt = rng.normal(size=(4,2,3,5)).astype(np.float32)*10
            pred = rng.normal(size=(2,3,5)).astype(np.float32)*5
            if task == "stereo":
                gt[:,1] = 0
                pred[1] = 0
            gt[2,0,0,0] = np.nan
            gt[2,0,0,1] = np.inf
            channels = 1 if task == "stereo" else 2
            raw = np.empty((channels,6,10),np.float32)
            for branch,(dy,dx) in zip(gt,GT_OFFSETS):
                raw[:,dy::2,dx::2] = -branch[:1] if task == "stereo" else branch
            output = -pred[:1] if task == "stereo" else pred
            official = module.StereoDatasetMetrics() if task == "stereo" else module.FlowDatasetMetrics()
            official.reset()
            official.add_batch(torch.from_numpy(output.copy())[None],torch.from_numpy(raw)[None])
            expected = official.get_results()
            actual = evaluate_pair(pred,pred,gt)["regions"]["all"]
            self.assertEqual(actual["pixels"],int(official.agg_N))
            key = "L1err" if task == "stereo" else "EPE"
            np.testing.assert_allclose(actual["error_px"],expected[key],rtol=2e-6,atol=2e-6)
            np.testing.assert_allclose(actual["over_1px_pct"],expected["bad@1.0"],rtol=2e-6,atol=2e-6)

    def test_nan_propagation_infinity_and_vector_minimum(self):
        gt = np.zeros((4,2,1,5),np.float32)
        gt[:,0,0,0] = [-8,-9,-10,-11]
        gt[1,0,0,1] = np.nan
        gt[1,0,0,2] = np.inf
        gt[:,0,0,3] = np.inf
        gt[:,:,0,4] = [[0,2],[2,0],[3,3],[4,4]]
        pred = np.zeros((2,1,5),np.float32)
        pred[0,0,0] = -8.25
        np.testing.assert_array_equal(error_map(pred,gt),[[.25,np.inf,0,np.inf,2]])
        np.testing.assert_array_equal(gt_valid_mask(gt),[[True,False,True,False,True]])

    def test_fixed_hard_denominator_threshold_and_failures(self):
        gt = np.zeros((4,2,1,4),np.float32)
        initial = np.zeros((2,1,4),np.float32)
        initial[0,0] = [1,2,3,4]
        local = initial.copy();local[0,0,1] = 0
        hard = fixed_hard_mask(np.stack((initial,local)),gt)
        np.testing.assert_array_equal(hard,[[False,False,True,True]])
        current = initial.copy();current[0,0,2] = 0;current[0,0,3] = np.nan
        result = evaluate_pair(initial,current,gt,hard)
        self.assertEqual(result["regions"]["all"]["pixels"],4)
        self.assertEqual(result["regions"]["hard"]["pixels"],2)
        self.assertEqual(result["regions"]["initial_good"]["pixels"],1)
        self.assertEqual(result["regions"]["all"]["nonfinite_prediction_pixels"],1)
        self.assertIsNone(result["regions"]["all"]["error_px"])
        self.assertEqual(result["regions"]["hard"]["over_1px_pct"],50)
        json.dumps(result,allow_nan=False)

    def test_scene_macro_is_not_case_or_pixel_weighted_and_keeps_failures(self):
        def row(scene,error,n=1):
            gt = np.zeros((4,2,1,n),np.float32)
            pred = np.zeros((2,1,n),np.float32);pred[0] = error
            return {"task":"flow","scene":scene,"evaluation":evaluate_pair(pred,pred,gt)}
        rows = [row("a",2,30),row("a",4,1),row("b",9,1)]
        out = aggregate_scene_macro(rows)
        self.assertEqual(out["flow"]["scene_macro"],6)
        rows.append(row("b",np.nan,1))
        out = aggregate_scene_macro(rows)
        self.assertIsNone(out["flow"]["scene_macro"])
        self.assertEqual(out["flow"]["failed_scenes"],1)
        json.dumps(out,allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
