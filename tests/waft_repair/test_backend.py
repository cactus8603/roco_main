"""Image/GT boundaries and the portable frozen-WAFT integration contract."""
import json
from pathlib import Path
import socket
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from waft_repair import backend as B
from waft_repair import data
from waft_repair.targets import CHECKPOINT_SHA256, CHEAP_METHOD, FeaturePair


def image(path, value, hw=(6, 6)):
    Image.fromarray(np.full((*hw, 3), value, dtype=np.uint8), mode="RGB").save(path)
    return path


def ground_truth(path, *, visible=True):
    arrays = dict(flow=np.zeros((6, 6, 2), np.float32), valid=np.ones((6, 6), bool))
    if visible:
        arrays["visible"] = np.ones((6, 6), bool)
    np.savez(path, **arrays)
    return path


def test_rgb_io_retains_original_units_order_and_native_geometry(tmp_path):
    array = np.zeros((6, 7, 3), np.uint8)
    array[..., 0], array[..., 1], array[..., 2] = 10, 20, 30
    Image.fromarray(array).save(tmp_path / "first.png")
    pair = B.load_rgb_pair(tmp_path / "first.png", image(tmp_path / "second.png", 40, (6, 7)))
    assert pair.shape == (1, 2, 3, 6, 7)
    assert pair.device.type == "cpu" and pair.dtype == torch.float32
    assert torch.equal(pair[0, 0, :, 0, 0], torch.tensor([10., 20., 30.]))
    with pytest.raises(ValueError, match="equal native"):
        B.load_rgb_pair(tmp_path / "first.png", image(tmp_path / "different.png", 5))


def test_gt_io_checks_dtype_masks_and_retains_full_valid_support(tmp_path):
    path = ground_truth(tmp_path / "flow.npz")
    gt, valid, visible = B.load_ground_truth(path)
    assert gt.shape == (1, 2, 6, 6) and gt.dtype == torch.float32
    assert valid.shape == visible.shape == (1, 6, 6)
    np.savez(path, flow=np.zeros((6, 6, 2), np.float32), valid=np.ones((6, 6), np.uint8))
    with pytest.raises(ValueError, match="boolean"):
        B.load_ground_truth(path)
    np.savez(path, flow=np.zeros((6, 6, 2), np.float64), valid=np.ones((6, 6), bool))
    with pytest.raises(ValueError, match="float32"):
        B.load_ground_truth(path)


def test_constructor_forces_offline_initializers_and_restores_every_patch_on_error():
    calls = []
    def create(name, **kwargs):
        calls.append((name, kwargs))
    def depth(**kwargs):
        calls.append(("depth", kwargs))
    timm = SimpleNamespace(create_model=create)
    module = SimpleNamespace(DepthAnythingFeature=depth)
    connect, connection = socket.socket.connect, socket.create_connection
    with pytest.raises(RuntimeError, match="offline"):
        with B._offline_constructor(module, timm):
            timm.create_model("vit_small_patch16_224", pretrained=True)
            module.DepthAnythingFeature(pretrained=True)
            socket.create_connection(("example.com", 80))
    assert calls == [("vit_small_patch16_224", {"pretrained": False}), ("depth", {"pretrained": False})]
    assert timm.create_model is create and module.DepthAnythingFeature is depth
    assert socket.socket.connect is connect and socket.create_connection is connection


def test_wrong_checkpoint_is_rejected_before_constructor_or_source_import(tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"not the official checkpoint")
    with patch.object(B, "_official_modules", side_effect=AssertionError("imported source")):
        with pytest.raises(ValueError, match="checkpoint SHA256 differs"):
            B.load_backend(tmp_path, weights)


def test_wrong_source_commit_is_rejected_before_model_construction(tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"fixture")
    with patch.object(B.subprocess, "check_output", return_value="wrong-commit\n") as command:
        with pytest.raises(ValueError, match="pinned official WAFT"):
            B.load_backend(tmp_path, weights, expected_checkpoint_sha=data.sha256(weights))
    assert command.call_args.args[0] == ["git", "-C", str(tmp_path), "rev-parse", "HEAD"]


class FakeBackend:
    metadata = {"waft_checkpoint_sha256": CHECKPOINT_SHA256}
    def encode_pair(self, pair):
        values = pair.mean((2, 3, 4))[:, :, None, None, None]
        features = values.expand(1, 2, 64, 4, 4).clone()
        return FeaturePair(features, (6, 6), (1, 1, 1, 1), "unused")

    def solve(self, features, native_hw, padding):
        assert features.shape == (1, 2, 64, 4, 4)
        return {key: [torch.full((1, channels, *native_hw), float(step)) for step in range(5)]
                for key, channels in (("flow", 2), ("info", 4))}


def fake_cheap(backend, pair, gt, valid, visible, *, original=None):
    original = original or backend.encode_pair(pair)
    target = original.features.clone()
    target[:, 0] += 1
    return dict(method=CHEAP_METHOD, teacher_features=target)


def test_prepare_final_uses_both_clean_frames_and_checked_manifest(tmp_path):
    for name, value in (("final1", 10), ("final2", 20), ("clean1", 30), ("clean2", 40)):
        image(tmp_path / (name + ".png"), value)
    ground_truth(tmp_path / "flow.npz")
    row = dict(pair_id="scene/frame", domain="sintel-final", image1="final1.png", image2="final2.png",
               teacher_image1="clean1.png", teacher_image2="clean2.png", target_source_pair_id="clean/frame", flow="flow.npz")
    data.write_json(tmp_path / "raw.json", {"samples": [row]})
    with patch.object(B, "load_backend", return_value=FakeBackend()), patch.object(B, "cheap_target", side_effect=fake_cheap):
        manifest = B.prepare_dataset(tmp_path / "raw.json", tmp_path / "prepared", "source", "weights")
    sample = data.load_sample(manifest, 0)
    assert torch.all(sample["original"][:, 0] == 10) and torch.all(sample["original"][:, 1] == 20)
    assert torch.all(sample["target"][:, 0] == 31) and torch.all(sample["target"][:, 1] == 40)
    with patch.object(B, "load_backend", side_effect=AssertionError("loaded before freshness check")):
        with pytest.raises(ValueError, match="already exists"):
            B.prepare_dataset(tmp_path / "raw.json", tmp_path / "prepared", "source", "weights")


def test_prepare_cheap_rejects_missing_visibility(tmp_path):
    image(tmp_path / "first.png", 10)
    image(tmp_path / "second.png", 20)
    ground_truth(tmp_path / "flow.npz", visible=False)
    data.write_json(tmp_path / "raw.json", {"samples": [dict(pair_id="pair", domain="kitti", image1="first.png", image2="second.png", flow="flow.npz")]})
    with patch.object(B, "load_backend", return_value=FakeBackend()):
        with pytest.raises(ValueError, match="visible mask"):
            B.prepare_dataset(tmp_path / "raw.json", tmp_path / "prepared", "source", "weights")


def test_inference_uses_no_gt_and_saves_every_native_step(tmp_path):
    image(tmp_path / "first.png", 10)
    image(tmp_path / "second.png", 20)
    weights = tmp_path / "repair.pt"
    weights.write_bytes(b"fixture")
    loader = ModuleType("waft_repair.checkpoint")
    loader.load_model = lambda *args, **kwargs: (lambda features, sigma, mask: features + 1,
                                              dict(waft_checkpoint_sha256=CHECKPOINT_SHA256, target_kind="cheap"))
    with patch.dict(sys.modules, {"waft_repair.checkpoint": loader}), patch.object(B, "load_backend", return_value=FakeBackend()), patch.object(B, "load_ground_truth", side_effect=AssertionError("GT read")):
        metadata = B.infer_images(weights, tmp_path / "first.png", tmp_path / "second.png", tmp_path / "flow.npz", "source", "weights")
    with np.load(tmp_path / "flow.npz", allow_pickle=False) as saved:
        assert saved["flow"].shape == (6, 6, 2)
        assert saved["flow_steps"].shape == (5, 6, 6, 2)
        assert saved["info_steps"].shape == (5, 6, 6, 4)
        assert np.all(saved["flow"] == 4)
        assert json.loads(str(saved["metadata"]))["ground_truth_used"] is False
    assert metadata["ground_truth_used"] is False
