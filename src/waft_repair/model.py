"""Trainable feature repair between the two parts of a frozen WAFT model.

The model receives only original features and fixed normalization metadata.
It predicts an unbounded residual over both frames' complete padded grids.
Targets, ground-truth flow, WAFT, and optimizers stay outside this module.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .context import PairedContext


def _autocast_enabled(device_type: str) -> bool:
    """Support the current PyTorch API and PyTorch 2.1."""
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:
        if device_type == "cpu":
            return torch.is_autocast_cpu_enabled()
        if device_type == "cuda":
            return torch.is_autocast_enabled()
        raise ValueError("repair supports only CPU or CUDA execution")


class _FixedFeatureResidual(torch.autograd.Function):
    """Preserve signed-zero input bytes while retaining the residual gradient."""

    @staticmethod
    def forward(ctx, original: Tensor, displacement: Tensor) -> Tensor:
        return torch.where(displacement == 0, original, original + displacement)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return None, gradient


class _ChannelLayerNorm(nn.LayerNorm):
    def forward(self, value: Tensor) -> Tensor:
        return super().forward(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _LocalResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64, dtype=torch.float32)
        self.norm = _ChannelLayerNorm(64, eps=1e-6, dtype=torch.float32)
        self.expand = nn.Conv2d(64, 128, 1, dtype=torch.float32)
        self.activation = nn.GELU()
        self.project = nn.Conv2d(128, 64, 1, dtype=torch.float32)

    def forward(self, value: Tensor) -> Tensor:
        update = self.depthwise(value)
        update = self.project(self.activation(self.expand(self.norm(update))))
        return value + update


class FeatureRepair(nn.Module):
    """Local repair plus paired SLiM context, with 3,923,200 parameters.

    ``original`` is detached FP32 ``[B,2,64,H,W]`` fused WAFT features.
    ``sigma`` is the fixed, native-domain original per-channel RMS, either
    ``[64]`` or ``[B,64]``. ``native_mask`` is boolean geometry metadata with
    shape ``[1|B,1,1,H,W]``. It does not mask inputs, residuals, or padding.

    Both readouts start at zero: the initial output preserves the original
    bytes, including negative zero. The output is ``original + raw_delta``;
    sigma normalizes inputs and never scales or bounds the output.

    Construction creates CPU FP32 parameters and preserves the caller's RNG.
    ``reference`` uses stock PyTorch for small grids. ``native`` explicitly
    requires the documented Mamba CUDA dependencies and CUDA feature inputs.
    Neither backend downloads weights, loads WAFT, or starts training.
    """

    def __init__(
        self, *, backend: str = "reference", seed: int = 20261026,
        max_reference_tokens: int = 4096,
    ) -> None:
        super().__init__()
        if backend not in ("reference", "native"):
            raise ValueError("backend must be 'reference' or 'native'")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.backend, self.seed = backend, seed
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.input_projection = nn.Conv2d(64, 192, 1, dtype=torch.float32)
            self.body = PairedContext(
                width=192, depth=2, aggregation=4, backend=backend,
                max_reference_tokens=max_reference_tokens,
            )
            self.readout = nn.Conv2d(192, 64, 1, dtype=torch.float32)
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed((seed + 1) % (2 ** 64))
            self.local_body = nn.Sequential(_LocalResidualBlock(), _LocalResidualBlock())
            self.local_readout = nn.Conv2d(64, 64, 1, dtype=torch.float32)
            nn.init.zeros_(self.local_readout.weight)
            nn.init.zeros_(self.local_readout.bias)

    def _validate(self, original: Tensor, sigma: Tensor, native_mask: Tensor) -> Tensor:
        if (original.dtype != torch.float32 or original.ndim != 5
                or tuple(original.shape[1:3]) != (2, 64) or original.shape[0] < 1
                or original.requires_grad or original.grad_fn is not None
                or not bool(torch.isfinite(original).all())):
            raise ValueError("original must be detached finite FP32 [B,2,64,H,W]")
        batch, _, _, height, width = original.shape
        if original.device.type not in ("cpu", "cuda"):
            raise ValueError("repair supports only CPU or CUDA execution")
        if _autocast_enabled(original.device.type):
            raise ValueError("repair requires FP32 execution with autocast disabled")
        if any(p.dtype != torch.float32 or p.device != original.device for p in self.parameters()):
            raise ValueError("all repair parameters must be FP32 on the feature input device")
        if min(height, width) < 4:
            raise ValueError("paired context requires H,W >= 4")
        if (sigma.dtype != original.dtype or sigma.device != original.device
                or tuple(sigma.shape) not in ((64,), (batch, 64))
                or sigma.requires_grad or sigma.grad_fn is not None
                or not bool(torch.isfinite(sigma).all()) or not bool((sigma > 0).all())):
            raise ValueError("sigma must be fixed positive finite FP32 [64] or [B,64]")
        if (native_mask.dtype != torch.bool or native_mask.device != original.device
                or tuple(native_mask.shape) not in ((1, 1, 1, height, width),
                                                     (batch, 1, 1, height, width))
                or not bool(native_mask.flatten(1).any(1).all())):
            raise ValueError("native_mask must be bool [1|B,1,1,H,W], nonempty per example")
        tokens = 4 * (height // 4) * (width // 4)
        if self.backend == "reference" and tokens > self.body.max_reference_tokens:
            raise RuntimeError(
                f"reference scan length {tokens} exceeds max_reference_tokens="
                f"{self.body.max_reference_tokens}; use backend='native' explicitly"
            )
        if self.backend == "native" and original.device.type != "cuda":
            raise RuntimeError("native backend requires CUDA features; no fallback")
        return sigma.reshape(1 if sigma.ndim == 1 else batch, 1, 64, 1, 1)

    def forward(self, original: Tensor, sigma: Tensor, native_mask: Tensor) -> Tensor:
        scale = self._validate(original, sigma, native_mask)
        normalized = original / scale
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("normalizing full-grid features by sigma produced nonfinite values")
        batch, _, _, height, width = normalized.shape
        flat = normalized.reshape(batch * 2, 64, height, width)
        contextual = self.input_projection(flat)
        contextual = self.body(contextual.reshape(batch, 2, 192, height, width))
        contextual = contextual.reshape(batch * 2, 192, height, width)
        raw_delta = self.readout(contextual) + self.local_readout(self.local_body(flat))
        raw_delta = raw_delta.reshape(batch, 2, 64, height, width)
        if raw_delta.dtype != torch.float32 or not bool(torch.isfinite(raw_delta).all()):
            raise ValueError("repair raw delta must remain finite FP32; mixed precision is not enabled")
        result = _FixedFeatureResidual.apply(original, raw_delta)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("adding the unbounded full-grid correction overflowed FP32 features")
        return result

    def configuration(self) -> dict:
        """Return reconstructable architecture settings and its data contract."""
        return {
            "architecture": "paired SLiM context plus per-frame local residual repair",
            "backend": self.backend, "seed": self.seed,
            "max_reference_tokens": self.body.max_reference_tokens,
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "state_tensors": len(self.state_dict()), "channels": 64,
            "context_width": 192, "context_blocks": 2, "context_aggregation": 4,
            "local_width": 64, "local_blocks": 2, "local_receptive_field": [5, 5],
            "local_normalization": "per-position channel LayerNorm, eps=1e-6",
            "local_expansion": 2, "local_shared_between_frames": True,
            "local_spatial_downsampling": False,
            "local_parameter_count": sum(p.numel() for p in self.local_body.parameters())
            + sum(p.numel() for p in self.local_readout.parameters()),
            "local_initialization_seed": (self.seed + 1) % (2 ** 64),
            "cross_frame_interaction": True,
            "feature_domain": "complete padded feature grid",
            "input_domain": "all original features divided by fixed original sigma",
            "output": "F0 + raw_delta, full padded grid",
            "output_units": "original WAFT feature units",
            "output_scaling": "none; raw readout is added directly to F0",
            "correction_bound": None, "output_nonlinearity": "none",
            "sigma_role": "input and external loss normalization only",
            "sigma_domain": "fixed original native per-channel RMS supplied by caller",
            "native_mask_role": "validated geometry metadata; no input or output masking",
            "padding": "visible to model and eligible for unbounded correction",
            "delta_fusion": "SLiM raw readout + local raw readout",
            "initial_output": "F0 byte-exact, including signed zero; both readouts zero",
            "WAFT_registered": False, "pretrained": False,
        }


def features(model: FeatureRepair, original: Tensor, sigma: Tensor, mask: Tensor) -> Tensor:
    """Apply non-reentrant activation checkpointing when gradients are enabled."""
    if torch.is_grad_enabled():
        return checkpoint(model, original, sigma, mask, use_reentrant=False, preserve_rng_state=True)
    return model(original, sigma, mask)
