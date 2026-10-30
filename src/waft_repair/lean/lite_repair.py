"""Tiny encoder-only feature residual, evaluated before any flow refinement.

Each direction uses its own encoder features/taps with shared repair weights.
There is no flow, decoder info, motion prior, label or sample ID in this API.
The full original 64-channel feature and its fixed sigma have an exact skip.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..motion.finite import all_tensors_finite
from ..motion.geometry import sample_at_xy, sampling_grid


def padded_geometry(native_hw, padding):
    if (not isinstance(native_hw, (tuple, list)) or len(native_hw) != 2
            or any(type(v) is not int or v <= 0 for v in native_hw)
            or not isinstance(padding, (tuple, list)) or len(padding) != 4
            or any(type(v) is not int or v < 0 for v in padding)):
        raise ValueError('Positive integer native H,W and nonnegative L,R,T,B required')
    h, w = native_hw
    left, right, top, bottom = padding
    hp, wp = h + top + bottom, w + left + right
    if hp % 112 or wp % 112:
        raise ValueError('Original WAFT padded geometry must be divisible by 112')
    return hp, wp


@dataclass(frozen=True)
class LiteRepairConfig:
    profile: str = 'encoder_only_lite_v1'
    width: int = 16
    semantic_width: int = 8
    blocks_s4: int = 1
    blocks_s2: int = 1
    expansion: int = 2
    seed: int = 20261030

    def __post_init__(self):
        if self.profile != 'encoder_only_lite_v1':
            raise ValueError('Fixed encoder_only_lite_v1 profile required')
        if (self.width, self.semantic_width, self.blocks_s4,
                self.blocks_s2, self.expansion) != (16, 8, 1, 1, 2):
            raise ValueError('Scientific lite shape is fixed at 16/8/1/1/2')
        if type(self.seed) is not int:
            raise ValueError('Integer initialization seed required')


class _LayerNorm2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, value):
        return self.norm(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class _DepthwiseResidual(nn.Module):
    def __init__(self, width, expansion):
        super().__init__()
        self.norm = _LayerNorm2D(width)
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.up = nn.Conv2d(width, width * expansion, 1)
        self.down = nn.Conv2d(width * expansion, width, 1)

    def forward(self, value):
        return value + self.down(F.gelu(self.up(self.depthwise(self.norm(value)))))


class _ExactIdentityResidual(torch.autograd.Function):
    """Preserve even signed-zero bits at init while retaining residual gradient."""
    @staticmethod
    def forward(ctx, original, displacement):
        return torch.where(displacement == 0, original, original + displacement)

    @staticmethod
    def backward(ctx, gradient):
        # The encoder feature is detached by contract. All learned gradients
        # enter only the displacement branch, including at an exactly zero head.
        return None, gradient


class LiteFeatureRepair(nn.Module):
    observation_channels = dict(cnn_pre_s2=64, cnn_pre_s4=128,
                                dav2_block6=384, dav2_block12=384)
    GEOMETRY_CACHE_LIMIT = 4

    def __init__(self, config: LiteRepairConfig | None = None):
        super().__init__()
        self.config = LiteRepairConfig() if config is None else config
        if not isinstance(self.config, LiteRepairConfig):
            raise TypeError('LiteRepairConfig required')
        c = self.config
        self.validate_inputs = False
        # Derived coordinates only: no feature, label, or learned state is kept.
        # An ordinary bounded cache deliberately stays outside state_dict.
        self._geometry_cache = OrderedDict()
        # Initialization is reproducible without advancing caller CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(c.seed)
            self.fine_stem = nn.Conv2d(128, c.width, 1)
            self.cnn4_stem = nn.Conv2d(128, c.width, 1)
            self.dav6_stem = nn.Conv2d(384, c.semantic_width, 1)
            self.dav12_stem = nn.Conv2d(384, c.semantic_width, 1)
            self.context_mix = nn.Conv2d(c.width + 2*c.semantic_width, c.width, 1)
            self.context_block = _DepthwiseResidual(c.width, c.expansion)
            self.fine_mix = nn.Conv2d(2*c.width, c.width, 1)
            self.fine_block = _DepthwiseResidual(c.width, c.expansion)
            self.residual_head = nn.Conv2d(c.width, 64, 1)
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

    def _apply(self, fn, recurse=True):
        self._geometry_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def _sampling_geometry(self, hp, wp, *, device, dtype):
        key = (hp, wp, torch.device(device), dtype)
        existing = self._geometry_cache.get(key)
        if existing is not None:
            self._geometry_cache.move_to_end(key)
            return existing
        semantic_xy = sampling_grid((hp//14, wp//14), 14, 6.5,
            (hp//4, wp//4), 4, 0., device=device, dtype=dtype)
        context_xy = sampling_grid((hp//4, wp//4), 4, 0.,
            (hp//2, wp//2), 2, 0., device=device, dtype=dtype)
        ys = torch.arange(hp//4, device=device)*4
        xs = torch.arange(wp//4, device=device)*4
        support = ((ys >= 6.5) & (ys <= hp-7.5))[:, None] & ((xs >= 6.5) & (xs <= wp-7.5))[None]
        result = (semantic_xy, context_xy, support)
        self._geometry_cache[key] = result
        if len(self._geometry_cache) > self.GEOMETRY_CACHE_LIMIT:
            self._geometry_cache.popitem(last=False)
        return result

    def configuration(self):
        return dict(**asdict(self.config),
            model_abi='paired-feature-repair/encoder-only-lite-v1',
            observation_channels=dict(self.observation_channels),
            output='F0+original_sigma*encoder_only_residual64',
            feature_inputs='same-pass own F0/sigma/CNN S2/S4 and DAV2 block6/12; shared per-frame weights',
            flow_inputs=False, decoder_info_inputs=False, motion_prior_inputs=False,
            label_inputs=False, pair_alignment='none',
            intended_consumer='One original WAFT five-update consumer after repair; no baseline solve',
            parameter_count=sum(v.numel() for v in self.parameters()))

    def cost_estimate(self, padded_hw, *, batch=1):
        """Exact static Conv/Linear MAC census, including both frames.

        One MAC is a multiply-add. No latency or whole-WAFT speed claim is made;
        normalization, GELU, interpolation and reduction operations are excluded.
        """
        if (not isinstance(padded_hw, (tuple, list)) or len(padded_hw) != 2
                or any(type(v) is not int or v <= 0 or v % 112 for v in padded_hw)
                or type(batch) is not int or batch <= 0):
            raise ValueError('Positive batch and divisible-by-112 padded H,W required')
        hp, wp = padded_hw
        strides = dict(fine_stem=2, cnn4_stem=4, dav6_stem=14,
            dav12_stem=14, context_mix=4, context_block=4,
            fine_mix=2, fine_block=2, residual_head=2)
        details = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                stride = strides[name.split('.', 1)[0]]
                locations = 2 * batch * (hp//stride) * (wp//stride)
                details[name] = int(locations * module.out_channels *
                    (module.in_channels//module.groups) * module.kernel_size[0] * module.kernel_size[1])
            elif isinstance(module, nn.Linear):
                raise AssertionError('Unexpected linear operation needs an explicit census rule')
        total = sum(details.values())
        return dict(parameter_count=sum(v.numel() for v in self.parameters()),
            padded_hw=list(padded_hw), paired_batch=batch,
            conv_linear_MACs=total, conv_linear_FLOPs=2*total,
            per_layer_MACs=details,
            excluded='Encoder/WAFT consumer, norm/GELU, grid resampling, scaling/reductions; not latency')

    @staticmethod
    def _cnn_scaled(value):
        # Same detached paired/channel RMS convention as the existing repair:
        # no label, flow or new learned encoder is involved.
        with torch.no_grad():
            rms = value.square().mean((1, 3, 4), keepdim=True).sqrt()
            floor = .01 * rms.flatten(1).median(1).values.reshape(-1, 1, 1, 1, 1) + 1e-6
            scale = torch.maximum(rms, floor)
        return value / scale

    def _validate(self, original, sigma, observations, native_hw, padding):
        hp, wp = padded_geometry(native_hw, padding)
        if (not isinstance(original, Tensor) or original.ndim != 5 or original.shape[0] < 1
                or tuple(original.shape[1:]) != (2, 64, hp//2, wp//2)):
            raise ValueError('Original F0 must be [B,2,64,Hp/2,Wp/2]')
        batch = original.shape[0]
        if not isinstance(sigma, Tensor) or tuple(sigma.shape) not in ((64,), (batch, 64)):
            raise ValueError('Original sigma [64] or [B,64] required')
        if not isinstance(observations, Mapping) or set(observations) != set(self.observation_channels):
            raise ValueError('Exactly the four declared same-pass encoder taps required')
        tensors = [original, sigma]
        for key, channels in self.observation_channels.items():
            stride = 2 if key == 'cnn_pre_s2' else 4 if key == 'cnn_pre_s4' else 14
            value = observations[key]
            if not isinstance(value, Tensor) or tuple(value.shape) != (batch, 2, channels, hp//stride, wp//stride):
                raise ValueError('Original encoder tap shape mismatch: '+key)
            tensors.append(value)
        if (original.device.type not in ('cpu', 'cuda')
                or torch.is_autocast_enabled(original.device.type)
                or any(v.dtype != torch.float32 or v.device != original.device
                       or v.requires_grad or v.grad_fn is not None for v in tensors)
                or not all_tensors_finite(tensors) or not bool((sigma > 0).all())):
            raise ValueError('Detached finite same-device FP32 inputs/sigma, positive sigma and no autocast required')
        if any(v.dtype != torch.float32 or v.device != original.device for v in self.parameters()):
            raise ValueError('Same-device FP32 repair parameters required')
        return batch, hp, wp

    def forward(self, original: Tensor, sigma: Tensor, observations: Mapping[str, Tensor], *,
                native_hw, padding, return_aux=False):
        if self.validate_inputs:
            batch, hp, wp = self._validate(original, sigma, observations, native_hw, padding)
        else:
            batch = original.shape[0]
            hp, wp = padded_geometry(native_hw, padding)
        scale = sigma.reshape(1 if sigma.ndim == 1 else batch, 1, 64, 1, 1)
        raw = original / scale

        def flat(value):
            return value.reshape(2*batch, value.shape[2], *value.shape[-2:])

        fine = F.gelu(self.fine_stem(torch.cat((flat(raw),
            flat(self._cnn_scaled(observations['cnn_pre_s2']))), 1)))
        cnn4 = F.gelu(self.cnn4_stem(flat(self._cnn_scaled(observations['cnn_pre_s4']))))
        semantics = torch.cat((self.dav6_stem(flat(observations['dav2_block6'])),
                               self.dav12_stem(flat(observations['dav2_block12']))), 1)
        # Original DAV2 patch14 centers are 14j+6.5; original CNN/F0 centers
        # are stride*j. Declared physical grids prevent half-pixel drift.
        semantic_xy, context_xy, support = self._sampling_geometry(hp, wp,
            device=raw.device, dtype=raw.dtype)
        semantics = sample_at_xy(semantics, semantic_xy[None].expand(2*batch, -1, -1, -1))
        semantics = semantics * support[None, None]
        context = self.context_block(F.gelu(self.context_mix(torch.cat((cnn4, semantics), 1))))
        context2 = sample_at_xy(context, context_xy[None].expand(2*batch, -1, -1, -1))
        detail = self.fine_block(F.gelu(self.fine_mix(torch.cat((fine, context2), 1))))
        residual = self.residual_head(detail).reshape(batch, 2, 64, hp//2, wp//2)
        repaired = _ExactIdentityResidual.apply(original, scale*residual)
        if self.validate_inputs and not all_tensors_finite((residual, repaired)):
            raise FloatingPointError('Nonfinite lightweight feature repair output')
        if return_aux:
            return repaired, dict(normalized_residual=residual,
                condition_source='encoder features/taps only; before the first WAFT update')
        return repaired
