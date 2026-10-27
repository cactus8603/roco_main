"""SLiM's aggregated Inception/Mamba interaction, isolated from its RGB matcher.

Adapted from Band-127/SLiM commit 8b34762c836e80f142d013d7e83e63d05efca1e3,
``src/backbone/convmamba.py`` and ``src/backbone/mamba/MambaEncoder.py``.
Copyright (c) 2026 CSW33(Choo Sin Wai), MIT; see
``third_party/feature_repair/SLIM_LICENSE``.

The CPU reference below adapts the parameterization, initialization and scalar
state recurrence of state-spaces/mamba v2.2.2, copyright (c) 2024 Tri Dao,
Albert Gu, Apache-2.0; see ``third_party/feature_repair/MAMBA_LICENSE``. Changes:
pure PyTorch recurrence and RMS normalization, bounded reference sequence
length, no generation cache, no distributed/fused execution. This is a
mathematical reference, not an assertion of CUDA numerical or speed parity.

The released SLiM encoder builds both token lists in row-major order, although
it reconstructs the second list in column-major order. This implementation
deliberately retains that released behavior, including on rectangular grids.
There is no warp, correspondence readout, RGB encoder, positional encoding,
pretrained weight download, or frozen WAFT registration here.
"""

from __future__ import annotations

from importlib import metadata
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SLIM_UPSTREAM_COMMIT = "8b34762c836e80f142d013d7e83e63d05efca1e3"
MAMBA_NATIVE_VERSION = "2.2.2"


class _LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape,
            self.weight, self.bias, self.eps,
        ).permute(0, 3, 1, 2)


class _GatedRMSNorm(nn.Module):
    """Mamba2's default norm_before_gate=False, one RMS group."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.float32))
        self.eps = 1e-5

    def forward(self, y: Tensor, z: Tensor) -> Tensor:
        dtype = y.dtype
        work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        gated = y.to(work_dtype) * F.silu(z.to(work_dtype))
        normalized = gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.to(work_dtype)).to(dtype)


class Mamba2Reference(nn.Module):
    """Non-fused Mamba2 subset used by released SLiM, with native state keys.

    expand=2, d_state=4, d_conv=3, ngroups=1, no projection biases,
    D_has_hdim=False, gated RMS normalization. The recurrent update is
    s[t] = exp(dt[t] A) s[t-1] + dt[t] x[t] B[t],
    y[t] = C[t] s[t] + D x[t]. The state begins at zero on every call.
    Native CUDA kernels are neither imported nor emulated by a placeholder.
    """

    def __init__(self, d_model: int, max_reference_tokens: int = 4096) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = 4
        self.d_conv = 3
        self.expand = 2
        self.d_inner = self.expand * d_model
        self.headdim = min(64, d_model // 4)
        if self.headdim < 1 or self.d_inner % self.headdim:
            raise ValueError("SLiM width must give an integral Mamba2 head count")
        self.nheads = self.d_inner // self.headdim
        self.max_reference_tokens = max_reference_tokens
        projection_width = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, projection_width, bias=False, dtype=torch.float32)
        conv_width = self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(
            conv_width, conv_width, self.d_conv,
            padding=self.d_conv - 1, groups=conv_width, bias=True, dtype=torch.float32,
        )
        dt = torch.exp(torch.rand(self.nheads, dtype=torch.float32) * (math.log(.1) - math.log(.001)) + math.log(.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.A_log = nn.Parameter(torch.empty(self.nheads, dtype=torch.float32).uniform_(1, 16).log())
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads, dtype=torch.float32))
        self.D._no_weight_decay = True
        self.norm = _GatedRMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, dtype=torch.float32)

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.d_model:
            raise ValueError(f"expected [B,L,{self.d_model}] sequence")
        batch, length, _ = sequence.shape
        if not sequence.is_floating_point() or length < 1:
            raise ValueError("reference sequence must be nonempty and floating point")
        if length > self.max_reference_tokens:
            raise RuntimeError(
                f"Mamba2 reference length {length} exceeds max_reference_tokens="
                f"{self.max_reference_tokens}; use the explicit native backend for large grids"
            )
        z, xbc, raw_dt = self.in_proj(sequence).split(
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1,
        )
        # Causal convolution: padding is on both sides internally, retain only
        # the first L outputs. This is the fused native causal-conv semantics.
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2))[..., :length].transpose(1, 2))
        x, b_coeff, c_coeff = xbc.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        work_dtype = torch.float64 if sequence.dtype == torch.float64 else torch.float32
        x = x.to(work_dtype).reshape(batch, length, self.nheads, self.headdim)
        b_coeff, c_coeff = b_coeff.to(work_dtype), c_coeff.to(work_dtype)
        dt = F.softplus(raw_dt.to(work_dtype) + self.dt_bias.to(work_dtype))
        a = -self.A_log.to(work_dtype).exp()
        state = torch.zeros(
            batch, self.nheads, self.headdim, self.d_state,
            device=sequence.device, dtype=work_dtype,
        )
        ys = []
        for index in range(length):
            decay = torch.exp(dt[:, index] * a)[..., None, None]
            drive = (
                dt[:, index, :, None, None]
                * x[:, index, :, :, None]
                * b_coeff[:, index, None, None, :]
            )
            state = decay * state + drive
            y = (state * c_coeff[:, index, None, None, :]).sum(-1)
            ys.append(y + self.D.to(work_dtype)[None, :, None] * x[:, index])
        y = torch.stack(ys, dim=1).reshape(batch, length, self.d_inner).to(sequence.dtype)
        return self.out_proj(self.norm(y, z))


def _make_mamba(width: int, backend: str, max_reference_tokens: int) -> nn.Module:
    if backend == "reference":
        return Mamba2Reference(width, max_reference_tokens=max_reference_tokens)
    if backend != "native":
        raise ValueError("backend must be 'reference' or 'native'")
    try:
        installed = metadata.version("mamba-ssm")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "native SLiM backend requires mamba-ssm==2.2.2 and its CUDA dependencies; "
            "no reference fallback is performed"
        ) from error
    if installed != MAMBA_NATIVE_VERSION:
        raise RuntimeError(f"native SLiM backend requires mamba-ssm==2.2.2, found {installed}")
    try:
        from mamba_ssm import Mamba2
        from mamba_ssm.ops.triton import ssd_combined
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError("mamba-ssm native CUDA extension import failed; no fallback is performed") from error
    # Mamba2 tolerates a missing causal-conv import, but its default fused path
    # later dereferences these extension methods. Fail before constructing a
    # model that cannot run. This inspection never executes a CUDA kernel.
    extension = getattr(ssd_combined, "causal_conv1d_cuda", None)
    if not all(callable(getattr(extension, name, None))
               for name in ("causal_conv1d_fwd", "causal_conv1d_bwd")):
        raise RuntimeError(
            "native SLiM requires the compiled causal-conv1d forward/backward extension; "
            "no reference fallback is performed"
        )
    # v2.2.2's D initializer omits dtype even when other factory kwargs specify
    # FP32. Normalize that constant parameter too, without changing RNG draws.
    return Mamba2(
        d_model=width, d_state=4, d_conv=3, expand=2,
        headdim=min(64, width // 4), device="cpu", dtype=torch.float32,
    ).float()


class _MambaLayer(nn.Module):
    def __init__(self, width: int, backend: str, max_reference_tokens: int) -> None:
        super().__init__()
        self.mamba_layer = _make_mamba(width, backend, max_reference_tokens)

    def forward(self, sequence: Tensor) -> Tensor:
        return self.mamba_layer(sequence)


class _QuadConcatMamba(nn.Module):
    def __init__(self, width: int, backend: str, max_reference_tokens: int) -> None:
        super().__init__()
        self.mamba_forward = _MambaLayer(width, backend, max_reference_tokens)
        self.mamba_backward = _MambaLayer(width, backend, max_reference_tokens)

    def forward(self, *sequences: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        sequence = torch.cat(sequences, dim=1)
        fused = .5 * (
            self.mamba_forward(sequence)
            + self.mamba_backward(sequence.flip(1)).flip(1)
        )
        return fused.chunk(4, dim=1)


class _MambaEncoder(nn.Module):
    def __init__(self, width: int, aggregation: int, backend: str, max_reference_tokens: int) -> None:
        super().__init__()
        self.mamba = _QuadConcatMamba(width, backend, max_reference_tokens)
        self.layer_norm = _LayerNorm2d(width, dtype=torch.float32)
        self.downsample_conv = nn.Conv2d(width, width, aggregation, stride=aggregation, dtype=torch.float32)

    def forward(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        original_size = x0.shape[-2:]
        x0 = self.layer_norm(self.downsample_conv(x0))
        x1 = self.layer_norm(self.downsample_conv(x1))
        height, width = x0.shape[-2:]
        x0_hw = x0.flatten(2).transpose(1, 2)
        x1_hw = x1.flatten(2).transpose(1, 2)
        # Released SLiM semantics: this is row-major again, despite the name.
        # Preserve it rather than silently replacing the published operation.
        x0_wh = x0.permute(0, 2, 3, 1).flatten(1, 2)
        x1_wh = x1.permute(0, 2, 3, 1).flatten(1, 2)
        x0_hw, x1_hw, x0_wh, x1_wh = self.mamba(x0_hw, x1_hw, x0_wh, x1_wh)
        x0 = (
            x0_hw.reshape(x0.shape[0], height, width, -1).permute(0, 3, 1, 2)
            + x0_wh.reshape(x0.shape[0], width, height, -1).permute(0, 3, 2, 1)
        )
        x1 = (
            x1_hw.reshape(x1.shape[0], height, width, -1).permute(0, 3, 1, 2)
            + x1_wh.reshape(x1.shape[0], width, height, -1).permute(0, 3, 2, 1)
        )
        return tuple(F.interpolate(x, size=original_size, mode="bilinear", align_corners=False) for x in (x0, x1))


class _InceptionNeXt(nn.Module):
    def __init__(self, width: int, aggregation: int) -> None:
        super().__init__()
        split = width // 8
        self.split_indices = (split, split, split, width - 3 * split)
        self.branch_1 = nn.Sequential(nn.Conv2d(split, split, 3, padding=1, groups=split, dtype=torch.float32))
        self.branch_2 = nn.Sequential(nn.Conv2d(split, split, (7, 1), padding=(3, 0), groups=split, dtype=torch.float32))
        self.branch_3 = nn.Sequential(nn.Conv2d(split, split, (1, 7), padding=(0, 3), groups=split, dtype=torch.float32))
        self.mlp = nn.Sequential(
            _LayerNorm2d(width, dtype=torch.float32), nn.Conv2d(width, 4 * width, 1, dtype=torch.float32),
            nn.GELU(), nn.Conv2d(4 * width, width, 1, dtype=torch.float32),
        )
        self.downsample_conv = nn.Conv2d(width, width, aggregation, stride=aggregation, dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        x_hw, x_h, x_w, x_id = self.downsample_conv(x.contiguous()).split(self.split_indices, dim=1)
        x = self.mlp(torch.cat((self.branch_1(x_hw), self.branch_2(x_h), self.branch_3(x_w), x_id), dim=1))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class PairedContext(nn.Module):
    """Released aggregated SLiM interaction on [B,2,width,H,W] native grids.

    The default core has 3,859,520 trainable parameters. Its input and output
    retain each frame's grid; it performs information exchange, not warping.
    Full-resolution 64D projections and the raw repair readout belong to
    the caller. Spatial aggregation uses floor division exactly as upstream.

    ``reference`` is explicitly limited to engineering checks on small grids.
    ``native`` requires the exact upstream mamba-ssm version and CUDA tensors.
    Native execution/gradients need separate GPU validation before training.
    """

    def __init__(
        self, width: int = 192, depth: int = 2, aggregation: int = 4,
        backend: str = "reference", max_reference_tokens: int = 4096,
    ) -> None:
        super().__init__()
        if not isinstance(width, int) or width < 8 or width % 8:
            raise ValueError("width must be a positive multiple of 8 and at least 8")
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        if not isinstance(aggregation, int) or aggregation < 1:
            raise ValueError("aggregation must be a positive integer")
        if not isinstance(max_reference_tokens, int) or max_reference_tokens < 1:
            raise ValueError("max_reference_tokens must be a positive integer")
        if backend not in ("reference", "native"):
            raise ValueError("backend must be 'reference' or 'native'")
        self.width, self.depth, self.aggregation = width, depth, aggregation
        self.backend = backend
        self.max_reference_tokens = max_reference_tokens
        self.convs = nn.ModuleList(_InceptionNeXt(width, aggregation) for _ in range(depth))
        self.mambas = nn.ModuleList(
            _MambaEncoder(width, aggregation, backend, max_reference_tokens)
            for _ in range(depth)
        )
        # Upstream initializes only the convolution branches with trunc_normal,
        # preserving the Mamba library's own projection/SSM initialization.
        with torch.no_grad():
            for module in self.convs.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.trunc_normal_(module.weight, std=.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def validate_input(self, features: Tensor) -> None:
        """Reject native-size CPU references before allocating branch activations."""
        if features.ndim != 5 or features.shape[1:3] != (2, self.width):
            raise ValueError(f"expected [B,2,{self.width},H,W] paired features")
        if not features.is_floating_point() or features.shape[0] < 1:
            raise ValueError("features must be floating point with nonempty batch")
        height, width = features.shape[-2:]
        if min(height, width) < self.aggregation:
            raise ValueError("each spatial dimension must be at least aggregation")
        tokens = 4 * (height // self.aggregation) * (width // self.aggregation)
        if self.backend == "reference" and tokens > self.max_reference_tokens:
            raise RuntimeError(
                f"SLiM reference scan length {tokens} exceeds max_reference_tokens="
                f"{self.max_reference_tokens}; select backend='native' explicitly for large grids"
            )
        if self.backend == "native" and features.device.type != "cuda":
            raise RuntimeError("SLiM native backend requires CUDA tensors; no CPU fallback is performed")

    def forward(self, features: Tensor) -> Tensor:
        self.validate_input(features)
        x0, x1 = features.unbind(dim=1)
        for convolution, mamba in zip(self.convs, self.mambas):
            x0 = x0 + convolution(x0)
            x1 = x1 + convolution(x1)
            update0, update1 = mamba(x0, x1)
            x0, x1 = x0 + update0, x1 + update1
        return torch.stack((x0, x1), dim=1)
