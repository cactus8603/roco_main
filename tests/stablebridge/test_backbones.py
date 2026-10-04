"""CPU contract tests; no network, checkpoint download or full CroCo allocation."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import FrozenInstanceError
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from stablebridge import artifacts, backbones


class FakeHead(torch.nn.Module):
    return_all_blocks = True

    def __init__(self):
        super().__init__()
        self.num_channels = 3

    def forward(self, blocks, info):
        tokens = blocks[-1] if isinstance(blocks, list) else blocks
        height, width = info["height"], info["width"]
        grid = tokens.transpose(1, 2).reshape(1, 3, height // 16, width // 16)
        return torch.nn.functional.interpolate(grid[:, :self.num_channels], size=(height, width), mode="nearest")


class FakeCroCo(torch.nn.Module):
    def __init__(self, head, **kwargs):
        super().__init__()
        self.head = head
        self.patch_embed = Namespace(patch_size=(16, 16))
        self.dummy = torch.nn.Parameter(torch.tensor(1.0))
        self.architecture = kwargs
        self.loaded_strict = None

    def load_state_dict(self, state, strict=True):
        self.loaded_strict = strict
        if state != {"mock": "verified"}:
            raise ValueError("unexpected mock state")

    def encode_image_pairs(self, source, target, return_all_blocks):
        def tokens(image):
            return torch.nn.functional.avg_pool2d(image, 16).flatten(2).transpose(1, 2)
        a, b = tokens(source), tokens(target)
        height, width = source.shape[-2:]
        pos = torch.cartesian_prod(torch.arange(height // 16), torch.arange(width // 16))[None]
        return ([a * 0.5, a] if return_all_blocks else a), b, pos, pos

    def _decoder(self, source, pos, masks, target, pos2, return_all_blocks):
        return [source - target, source + 2 * target] if return_all_blocks else source + 2 * target

    def forward(self, source, target):
        # An independent simple analytic native path provides parity evidence
        # without repeating the adapter's encoder/decoder control flow.
        combined = source + 2 * target
        grid = torch.nn.functional.avg_pool2d(combined, 16)
        return torch.nn.functional.interpolate(grid[:, :self.head.num_channels], size=source.shape[-2:], mode="nearest")


def fake_state(task="flow", position="RoPE100"):
    return {"args": Namespace(task=task, crop=(32, 64), criterion="LaplacianLossBounded(a=1,b=3)",
                              croco_args={"img_size": (32, 64), "pos_embed": position}),
            "model": {"mock": "verified"}}


class CroCoAdapterTests(unittest.TestCase):
    def build(self, task="flow", context_hw=None, position="RoPE100"):
        with patch.object(backbones, "load_official_checkpoint", return_value=(fake_state(task, position), {"sha256": "test"})), \
                patch.object(backbones, "_vendor_classes", return_value=(FakeCroCo, FakeHead)):
            return backbones.CroCoAdapter(task, "/not-loaded.pth", context_hw=context_hw)

    def test_native_geometry_features_and_sign(self):
        adapter = self.build("stereo")
        image0 = np.zeros((32, 64, 3), np.uint8)
        image1 = np.full_like(image0, 255)
        prediction = adapter.predict(image0, image1, origin_xy=(101, 203))
        self.assertEqual(prediction.displacement.shape, (2, 32, 64))
        self.assertEqual(prediction.displacement.dtype, np.float32)
        np.testing.assert_array_equal(prediction.displacement[1], 0)
        self.assertEqual(prediction.source_features.shape, (3, 2, 4))
        np.testing.assert_allclose(prediction.source_features[:, 0, 0],
                                   -np.asarray(backbones.IMAGENET_MEAN) / backbones.IMAGENET_STD, rtol=1e-5)
        np.testing.assert_allclose(prediction.target_features[:, 0, 0],
                                   (1 - np.asarray(backbones.IMAGENET_MEAN)) / backbones.IMAGENET_STD, rtol=1e-5)
        np.testing.assert_array_equal(prediction.token_centers_xy[0, 0], [108.5, 210.5])
        np.testing.assert_array_equal(prediction.token_centers_xy[-1, -1], [156.5, 226.5])
        expected_x = -(-0.485 + 2 * (1 - 0.485)) / 0.229
        np.testing.assert_allclose(prediction.displacement[0], expected_x, rtol=1e-5)
        self.assertTrue(adapter.model.loaded_strict)
        self.assertFalse(adapter.model.training)
        self.assertFalse(adapter.model.dummy.requires_grad)
        with self.assertRaises(FrozenInstanceError):
            prediction.displacement = None

    def test_parity_both_head_paths(self):
        for all_blocks in (True, False):
            adapter = self.build()
            adapter.model.head.return_all_blocks = all_blocks
            rng = np.random.default_rng(3)
            a = rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)
            b = rng.integers(0, 256, a.shape, dtype=np.uint8)
            result = adapter.native_forward_parity(a, b)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["forward_calls"], 2)

    def test_expansion_preserves_pixels_and_is_explicit(self):
        adapter = self.build(context_hw=(64, 96))
        self.assertEqual(adapter.native_hw, (32, 64))
        self.assertEqual(adapter.model.architecture["img_size"], (64, 96))
        image = np.zeros((64, 96, 3), np.uint8)
        self.assertEqual(adapter.predict(image, image).source_features.shape, (3, 4, 6))
        with self.assertRaisesRegex(ValueError, "never resized"):
            adapter.predict(image[:32], image[:32])
        with self.assertRaisesRegex(ValueError, "RoPE"):
            self.build(context_hw=(64, 96), position="cosine")
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            self.build(context_hw=(48, 64))

    def test_invalid_input_and_tf32_restoration(self):
        adapter = self.build()
        image = np.zeros((32, 64, 3), np.uint8)
        with self.assertRaises(TypeError):
            adapter.predict(image.astype(np.float32), image)
        with self.assertRaises(ValueError):
            adapter.predict(image, image, origin_xy=(np.nan, 0))
        old = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            adapter.predict(image, image)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old

    def test_bad_token_order_fails_closed(self):
        adapter = self.build()
        original = adapter.model.encode_image_pairs
        def wrong(*args, **kwargs):
            a, b, pos, pos2 = original(*args, **kwargs)
            return a, b, pos.flip(1), pos2
        image = np.zeros((32, 64, 3), np.uint8)
        with patch.object(adapter.model, "encode_image_pairs", wrong), self.assertRaisesRegex(RuntimeError, "patch order"):
            adapter.predict(image, image)


class FakeResponse(io.BytesIO):
    def __init__(self, data, url, status=200, headers=None):
        super().__init__(data)
        self.url, self.status = url, status
        self.headers = headers or {}

    def geturl(self):
        return self.url


class ArtifactTests(unittest.TestCase):
    def test_atomic_download_metadata_and_existing_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            data = b"official mock bytes"
            url = artifacts.OFFICIAL_BASE + artifacts.OFFICIAL_FILES["flow"]
            response = FakeResponse(data, url, headers={"Content-Length": str(len(data)), "ETag": '"abc"'})
            with patch.object(artifacts, "urlopen", return_value=response), \
                    patch.object(torch, "load", return_value=fake_state()) as loader:
                meta = artifacts.ensure_official_checkpoint("flow", directory)
            path = Path(meta["path"])
            self.assertEqual(path.read_bytes(), data)
            self.assertTrue(path.with_suffix(".pth.json").exists())
            self.assertFalse(path.with_suffix(".pth.part").exists())
            self.assertFalse(meta["spring_supervised_finetuning"])
            self.assertEqual(loader.call_args.kwargs["weights_only"], False)
            with patch.object(artifacts, "urlopen", side_effect=AssertionError("must reuse verified file")):
                self.assertEqual(artifacts.ensure_official_checkpoint("flow", directory)["sha256"], meta["sha256"])
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "do not match"):
                artifacts.ensure_official_checkpoint("flow", directory)

    def test_resume_ranges_and_restart_if_ignored(self):
        for range_supported in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                part = Path(directory) / "weights.part"
                state = Path(directory) / "weights.part.json"
                part.write_bytes(b"abcd")
                url = artifacts.OFFICIAL_BASE + "crocoflow.pth"
                state.write_text(json.dumps({"source_url": url, "etag": '"same"', "total_bytes": 8}))
                response = FakeResponse(b"efgh" if range_supported else b"abcdefgh", url,
                                        status=206 if range_supported else 200,
                                        headers={"ETag": '"same"', "Content-Range": "bytes 4-7/8", "Content-Length": "8"})
                with patch.object(artifacts, "urlopen", return_value=response) as opener:
                    artifacts._download(url, part, state)
                self.assertEqual(part.read_bytes(), b"abcdefgh")
                self.assertEqual(opener.call_args.args[0].get_header("Range"), "bytes=4-")
                self.assertEqual(opener.call_args.args[0].get_header("If-range"), '"same"')

    def test_reject_unknown_or_mismatched_before_unpickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crocoflow.pth"
            path.write_bytes(b"unknown untrusted checkpoint")
            with patch.object(torch, "load") as loader:
                with self.assertRaisesRegex(ValueError, "Unregistered"):
                    artifacts.load_official_checkpoint(path, "flow")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    artifacts.load_official_checkpoint(path, "flow", "0" * 64)
                loader.assert_not_called()

    def test_failed_digest_never_publishes_or_unpickles(self):
        with tempfile.TemporaryDirectory() as directory:
            url = artifacts.OFFICIAL_BASE + artifacts.OFFICIAL_FILES["flow"]
            response = FakeResponse(b"bad", url, headers={"Content-Length": "3"})
            with patch.object(artifacts, "urlopen", return_value=response), patch.object(torch, "load") as loader:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    artifacts.ensure_official_checkpoint("flow", directory, "0" * 64)
                loader.assert_not_called()
            self.assertFalse((Path(directory) / "crocoflow.pth").exists())
            self.assertTrue((Path(directory) / "crocoflow.pth.part").exists())

    def test_truncated_response_preserves_partial_and_wrong_range_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            part, state = Path(directory) / "p", Path(directory) / "state"
            url = artifacts.OFFICIAL_BASE + "crocoflow.pth"
            response = FakeResponse(b"abc", url, headers={"Content-Length": "8", "ETag": "v1"})
            with patch.object(artifacts, "urlopen", return_value=response), self.assertRaises(IOError):
                artifacts._download(url, part, state)
            self.assertEqual(part.read_bytes(), b"abc")
            response = FakeResponse(b"defgh", url, status=206, headers={"Content-Range": "bytes 4-7/8"})
            with patch.object(artifacts, "urlopen", return_value=response), self.assertRaisesRegex(ValueError, "resume range"):
                artifacts._download(url, part, state)
            self.assertEqual(part.read_bytes(), b"abc")


if __name__ == "__main__":
    unittest.main()
