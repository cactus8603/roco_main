"""Frozen native-coordinate CroCo v2 adapter; no implicit resizing or fallback.

Encoder features are the last encoder block **after enc_norm**, before the
cross-image decoder; they are not L2-normalized. Patch centers use integer pixel
centers (0..W-1), so a 16-pixel patch has center 7.5, not 8.0.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

from .artifacts import load_official_checkpoint, sha256_file


VENDOR_ROOT = Path(__file__).resolve().parents[2] / "research/croco_trusted_first/vendor/croco"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CroCoPrediction:
    displacement: np.ndarray  # float32 [2,H,W], source -> target, original pixels
    raw_uncertainty: np.ndarray  # float32 [H,W], uncalibrated head output
    source_features: np.ndarray  # float32 [C,H/16,W/16], post enc_norm
    target_features: np.ndarray
    token_centers_xy: np.ndarray  # float32 [H/16,W/16,2], global original-image coordinates
    metadata: dict[str, Any]


def _vendor_classes():
    existing = sys.modules.get("models")
    if existing is not None:
        location = Path(getattr(existing, "__file__", "") or "/")
        locations = [Path(p) for p in getattr(existing, "__path__", [])]
        if not (location.is_relative_to(VENDOR_ROOT) or any(p.is_relative_to(VENDOR_ROOT) for p in locations)):
            raise ImportError("A different 'models' module is already imported; use an isolated CroCo process")
    sys.path.insert(0, str(VENDOR_ROOT)) if str(VENDOR_ROOT) not in sys.path else None
    # Upstream import sets TF32 globally. Preserve the caller's state here;
    # actual predictions explicitly use float32 with TF32 disabled below.
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        from models.croco_downstream import CroCoDownstreamBinocular
        from models.head_downstream import PixelwiseTaskWithDPT
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return CroCoDownstreamBinocular, PixelwiseTaskWithDPT


def _hw(value, name="context_hw") -> tuple[int, int]:
    if isinstance(value, int):
        value = (value, value)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain height and width")
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x <= 0 or x % 32 for x in value):
        raise ValueError(f"{name} must contain positive multiples of 32")
    return tuple(map(int, value))


@contextmanager
def _precision(device: torch.device):
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class CroCoAdapter:
    """One frozen official task checkpoint at one explicitly declared context size.

    Changing ``context_hw`` changes the model's patch-grid declaration while
    preserving checkpoint parameters via strict loading. Only RoPE checkpoints
    may change context. Both images have the same size and crop origin; separate
    origins or asymmetric views require a different explicit geometry adapter.
    """

    def __init__(self, task: str, checkpoint: str | Path, device="cpu", context_hw=None,
                 expected_sha256: str | None = None):
        if task not in ("stereo", "flow"):
            raise ValueError("task must be stereo or flow")
        self.task, self.device = task, torch.device(device)
        state, provenance = load_official_checkpoint(checkpoint, task, expected_sha256)
        args = state["args"]
        self.native_hw = _hw(getattr(args, "crop", args.croco_args.get("img_size")), "native_hw")
        self.context_hw = self.native_hw if context_hw is None else _hw(context_hw)
        architecture = dict(args.croco_args)
        if self.context_hw != self.native_hw and not str(architecture.get("pos_embed", "")).startswith("RoPE"):
            raise ValueError("Expanded context requires RoPE; positional interpolation is not implicit")
        architecture["img_size"] = self.context_hw
        CroCoDownstreamBinocular, PixelwiseTaskWithDPT = _vendor_classes()
        head = PixelwiseTaskWithDPT()
        head.num_channels = 2 if task == "stereo" else 3
        self.model = CroCoDownstreamBinocular(head, **architecture)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval().requires_grad_(False).to(device=self.device, dtype=torch.float32)
        self.patch_hw = tuple(map(int, self.model.patch_embed.patch_size))
        if self.patch_hw != (16, 16):
            raise ValueError("The StableBridge CroCo feature interface requires 16x16 patches")
        self._provenance = provenance
        self.metadata = {
            "adapter": "croco_v2_native_v1", "task": task, "checkpoint": provenance,
            "native_hw": list(self.native_hw), "context_hw": list(self.context_hw),
            "criterion": str(args.criterion), "croco_args": architecture,
            "frozen": True, "parameters": sum(p.numel() for p in self.model.parameters()),
            "feature_layer": "last_encoder_block_after_enc_norm_before_decoder",
            "feature_l2_normalized": False, "feature_stride_xy": [16, 16],
            "feature_center_offset_xy": [7.5, 7.5],
            "uncertainty": "raw_Laplacian_head_parameter_not_calibrated_probability",
            "normalization": {"rgb_range": "uint8_0_255", "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "precision": "float32_no_autocast_TF32_disabled", "device": str(self.device),
            "vendor_revision": (VENDOR_ROOT / "REVISION").read_text().strip(),
            "vendor_forward_sha256": sha256_file(VENDOR_ROOT / "models/croco_downstream.py"),
        }
        del state

    def _inputs(self, image0, image1):
        tensors = []
        mean = torch.tensor(IMAGENET_MEAN, device=self.device)[None, :, None, None]
        std = torch.tensor(IMAGENET_STD, device=self.device)[None, :, None, None]
        for index, image in enumerate((image0, image1)):
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                raise TypeError(f"image{index} must be an RGB uint8 numpy array")
            if image.shape != (*self.context_hw, 3):
                raise ValueError(f"image{index} must have shape {(*self.context_hw, 3)}; images are never resized")
            tensor = torch.from_numpy(np.array(image, copy=True, order="C")).permute(2, 0, 1)[None]
            tensor = tensor.to(device=self.device, dtype=torch.float32) / 255.0
            tensors.append((tensor - mean) / std)
        return tensors

    def _forward_features(self, source, target):
        # This follows CroCoDownstreamBinocular.forward exactly, exposing the
        # already computed encoder activations (no extra encoder pass).
        all_blocks = bool(getattr(self.model.head, "return_all_blocks", False))
        out, out2, pos, pos2 = self.model.encode_image_pairs(source, target, return_all_blocks=all_blocks)
        final_source = out[-1] if all_blocks else out
        decoded = self.model._decoder(final_source, pos, None, out2, pos2, return_all_blocks=all_blocks)
        if all_blocks:
            decoded = out + decoded
        h, w = self.context_hw
        prediction = self.model.head(decoded, {"height": h, "width": w})
        return prediction, final_source, out2, pos, pos2

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict(self, image0, image1, origin_xy=(0, 0)) -> CroCoPrediction:
        call_start = time.perf_counter()
        origin = np.asarray(origin_xy, dtype=np.float64)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("origin_xy must contain two finite native pixel coordinates")
        source, target = self._inputs(image0, image1)
        self._synchronize()
        start = time.perf_counter()
        preprocessing_seconds = start - call_start
        with _precision(self.device):
            raw, features0, features1, pos0, pos1 = self._forward_features(source, target)
        self._synchronize()
        seconds = time.perf_counter() - start
        h, w = self.context_hw
        expected = (1, 2 if self.task == "stereo" else 3, h, w)
        if tuple(raw.shape) != expected:
            raise RuntimeError(f"Unexpected CroCo output {tuple(raw.shape)}; expected {expected}")
        values = raw[0].float().cpu().numpy().copy()
        if not np.isfinite(values).all():
            raise FloatingPointError("CroCo produced non-finite predictions")
        displacement = (np.stack((-values[0], np.zeros_like(values[0])))
                        if self.task == "stereo" else values[:2].copy())
        # Vendor PositionGetter returns (row, column), while the public geometry
        # uses (x, y). Check row-major order before reshaping token features.
        yy, xx = np.mgrid[:h // 16, :w // 16]
        expected_pos = np.stack((yy, xx), axis=-1).reshape(-1, 2)
        for positions in (pos0, pos1):
            if not np.array_equal(positions[0].cpu().numpy(), expected_pos):
                raise RuntimeError("Unexpected CroCo patch order; cannot assign native coordinates")
        def spatial(tokens):
            if tokens.ndim != 3 or tuple(tokens.shape[:2]) != (1, (h // 16) * (w // 16)):
                raise RuntimeError("Unexpected encoder token shape")
            result = tokens[0].transpose(0, 1).reshape(-1, h // 16, w // 16).float().cpu().numpy().copy()
            if not np.isfinite(result).all():
                raise FloatingPointError("CroCo produced non-finite features")
            return result
        centers = (np.stack((xx, yy), axis=-1) * 16 + 7.5 + origin).astype(np.float32)
        source_features, target_features = spatial(features0), spatial(features1)
        metadata = {**self.metadata, "origin_xy": origin.tolist(), "forward_seconds": seconds,
                    "preprocessing_seconds": preprocessing_seconds,
                    "total_predict_seconds": time.perf_counter() - call_start,
                    "forward_calls": 1, "image_resize": False,
                    "displacement_convention": "source_to_target_xy_native_pixels"}
        return CroCoPrediction(displacement, values[-1].copy(), source_features, target_features, centers, metadata)

    def native_forward_parity(self, image0, image1, atol=1e-5, rtol=1e-5) -> dict:
        """Compare extraction path against unmodified upstream forward, all heads.

        This intentionally costs two full forwards and must be charged as a
        validation operation, not hidden in the reported method runtime.
        """
        source, target = self._inputs(image0, image1)
        self._synchronize()
        start = time.perf_counter()
        with _precision(self.device):
            native = self.model(source, target)
            extracted, *_ = self._forward_features(source, target)
        self._synchronize()
        finite = bool(torch.isfinite(native).all() and torch.isfinite(extracted).all())
        return {"passed": finite and bool(torch.allclose(native, extracted, atol=atol, rtol=rtol)),
                "max_abs_difference": float((native - extracted).abs().max().cpu()),
                "atol": atol, "rtol": rtol, "forward_calls": 2,
                "seconds": time.perf_counter() - start,
                "context_hw": list(self.context_hw), "all_output_channels_compared": True}
