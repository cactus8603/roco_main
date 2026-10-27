"""Two fixed, GT-supervised WAFT target-feature recipes.

No model, weights, datasets, device, workers, or files are acquired on import.
The caller supplies an already loaded official WAFT and its geometry helpers.
Historical native results come from the audited experiments documented in
docs/waft_targets_v1, not from an additional run of this portable module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


CHEAP_METHOD = "waft-visible-pull-rgb-v1"
EXPENSIVE_METHOD = "waft-dense-final-epe-512-v1"
CHECKPOINT_SHA256 = "1b49d1fc4513c6fa34a8de8f791dd8735d366d4b25d8dea47626a79056edd402"
STEPS = 5


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def tensor_sha256(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    prefix = str((str(value.dtype), tuple(value.shape))).encode()
    return hashlib.sha256(prefix + value.numpy().tobytes()).hexdigest()


def _rgb(pair):
    require(isinstance(pair, Tensor) and pair.dtype == torch.float32
            and pair.ndim == 5 and tuple(pair.shape[:3]) == (1, 2, 3),
            "One FP32 RGB pair [1,2,3,H,W] required")
    require(torch.isfinite(pair).all() and pair.min() >= 0 and pair.max() <= 255,
            "RGB must be finite in original 0..255 units")


def _support(gt, valid):
    require(gt.dtype == torch.float32 and gt.ndim == 4 and tuple(gt.shape[:2]) == (1, 2),
            "Native FP32 GT [1,2,H,W] required")
    require(not gt.requires_grad and valid.dtype == torch.bool
            and valid.device == gt.device and valid.shape == (1, *gt.shape[-2:]),
            "Fixed GT and native bool valid [1,H,W] on the same device required")
    support = valid & torch.isfinite(gt).all(1)
    require(support.any(), "At least one finite valid GT pixel required")
    return support


@torch.no_grad()
def reconstruct_visible_rgb(pair: Tensor, gt: Tensor, valid: Tensor,
                            visible: Tensor) -> tuple[Tensor, Tensor]:
    """Exact selected CPU-FP64 pull, cast once to FP32; alpha is fixed at1.

    KITTI visible = flow_noc valid; Sintel visible = occlusions ==0. Neither
    visibility nor inbounds changes the final flow evaluation support.
    """
    _rgb(pair)
    h, w = pair.shape[-2:]
    require(pair.device.type == gt.device.type == valid.device.type == "cpu",
            "Selected RGB reconstruction uses CPU interpolation")
    require(gt.dtype == torch.float32 and gt.shape == (1, 2, h, w)
            and valid.dtype == torch.bool and valid.shape == (1, h, w),
            "Native forward GT and valid mask required")
    require(isinstance(visible, Tensor) and visible.device.type == "cpu"
            and visible.dtype == torch.bool and visible.shape == valid.shape,
            "Explicit nonocclusion annotation required; no geometry-only fallback")
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                            torch.arange(w, dtype=torch.float64), indexing="ij")
    finite = torch.isfinite(gt).all(1)
    safe = torch.where(finite[:, None], gt, 0).double()
    x, y = xx[None] + safe[:, 0], yy[None] + safe[:, 1]
    mask = valid & finite & visible & (x >= 0) & (x <= w-1) & (y >= 0) & (y <= h-1)
    x, y = x[0, mask[0]], y[0, mask[0]]
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0+1).clamp_max(w-1), (y0+1).clamp_max(h-1)
    wx, wy = (x-x0)[None], (y-y0)[None]
    donor = pair[0, 1].double()
    pixels = ((1-wx)*(1-wy)*donor[:, y0, x0]
              + wx*(1-wy)*donor[:, y0, x1]
              + (1-wx)*wy*donor[:, y1, x0]
              + wx*wy*donor[:, y1, x1]).float()
    result = pair.clone()
    result[0, 0, :, mask[0]] = pixels
    return result, mask


@dataclass(frozen=True)
class FeaturePair:
    features: Tensor
    native_hw: tuple[int, int]
    padding: tuple[int, int, int, int]
    rgb_sha256: str


def native_mask(original: Tensor, native_hw, padding) -> Tensor:
    require(original.dtype == torch.float32 and original.ndim == 5
            and tuple(original.shape[:3]) == (1, 2, 64), "Fixed paired64 FP32 field required")
    h, w = tuple(native_hw)
    left, right, top, bottom = tuple(padding)
    require(all(type(v) is int and v >= 0 for v in padding) and h > 0 and w > 0,
            "Positive native geometry and nonnegative integer padding required")
    fh, fw = original.shape[-2:]
    require((2*fh, 2*fw) == (h+top+bottom, w+left+right), "Exact stride2 padded geometry required")
    yy, xx = torch.meshgrid(torch.arange(fh, device=original.device),
                            torch.arange(fw, device=original.device), indexing="ij")
    mask = ((2*xx-left >= 0) & (2*xx-left < w)
            & (2*yy-top >= 0) & (2*yy-top < h))[None, None, None]
    require(mask.any(), "Nonempty native feature support required")
    return mask


@torch.no_grad()
def original_sigma(original: Tensor, native_hw, padding) -> tuple[Tensor, Tensor]:
    """Original-only channel RMS, both frames/native centers, FP64 reduction."""
    mask = native_mask(original, native_hw, padding)
    values = original[0, :, :, mask[0, 0, 0]].double()
    rms = values.square().mean(dim=(0, 2)).sqrt()
    floor = .01*rms.median()+1e-8  # lower median: channel order statistic31/64
    sigma = rms.clamp_min(floor).float()
    require(torch.isfinite(sigma).all() and (sigma > 0).all(), "Positive finite original sigma")
    return sigma, mask


class FrozenWAFTBackend:
    """Official prefix and complete5 feature consumer, including live geometry.

    Pass the official Padder, coords_grid and bilinear_sampler explicitly.
    strict=False permits small CPU fixtures; it is not the evaluated recipe.
    """
    def __init__(self, core: nn.Module, padder_factory: Callable,
                 coords_grid: Callable, bilinear_sampler: Callable, *,
                 strict: bool = True, checkpoint_refinement: bool = True):
        self.core = core
        self.padder_factory = padder_factory
        self.coords_grid = coords_grid
        self.bilinear_sampler = bilinear_sampler
        self.strict = strict
        self.checkpoint_refinement = checkpoint_refinement
        self.validate()

    def validate(self):
        core = self.core
        require(not any(m.training for m in core.modules())
                and all(not p.requires_grad and p.grad is None for p in core.parameters()),
                "Every original WAFT module must stay eval/frozen/gradNone")
        parameters = list(core.parameters())
        require(parameters and all(p.dtype == torch.float32 and p.device == parameters[0].device
                                   for p in parameters), "Populated FP32 core on one device required")
        require(type(core.factor) is int and core.factor > 0 and core.factor % 2 == 0,
                "Even original padding factor required")
        for name in ("encoder", "fnet", "fmap_conv", "hidden_conv", "warp_linear", "refine_net",
                     "refine_transform", "flow_head", "upsample_weight"):
            require(isinstance(getattr(core, name, None), nn.Module), "Missing official module: "+name)
        if self.strict:
            require(core.factor == 112 and len(parameters) == len(core.state_dict()) == 508
                    and sum(p.numel() for p in parameters) == 56_932_619
                    and not list(core.buffers()), "Exact DAv2-a2 full508 WAFT required")
        return parameters[0].device

    def _encode(self, image, padder):
        padded = padder.pad(self.core.normalize_image(image))
        semantic = self.core.encoder(padded)
        rgb = self.core.fnet(padded)[0]
        return self.core.fmap_conv(torch.cat((semantic, rgb), dim=1))

    @torch.no_grad()
    def encode_pair(self, pair: Tensor) -> FeaturePair:
        _rgb(pair)
        device = self.validate()
        padder = self.padder_factory(pair[:, 0].shape, factor=self.core.factor)
        features = torch.stack([self._encode(frame.to(device), padder)
                                for frame in pair.unbind(1)], 1)
        geometry = FeaturePair(features, tuple(pair.shape[-2:]), tuple(padder._pad), tensor_sha256(pair))
        native_mask(features, geometry.native_hw, geometry.padding)
        return geometry

    def _refine(self, source, warped, hidden, flow):
        value = self.core.warp_linear(torch.cat((source, warped, hidden, flow), 1))
        value = self.core.refine_net(value)
        hidden = self.core.refine_transform(torch.cat((value["out"], hidden), 1))
        update = self.core.flow_head(hidden)
        mask = .25*self.core.upsample_weight(hidden)
        return hidden, update, mask

    def solve(self, features: Tensor, native_hw, padding) -> dict:
        device = self.validate()
        require(features.device == device and torch.isfinite(features).all(), "Finite features on core device")
        native_mask(features, native_hw, padding)
        source, target = features[:, 0], features[:, 1]
        hidden = self.core.hidden_conv(torch.cat((source, target), 1))
        flow = source.new_zeros((source.shape[0], 2, *source.shape[-2:]))
        left, right, top, bottom = padding
        outputs, infos = [], []
        for _ in range(STEPS):
            coordinates = self.coords_grid(flow.shape[0], *flow.shape[-2:], device=device)+flow
            warped = self.bilinear_sampler(target, coordinates.permute(0, 2, 3, 1))
            if self.checkpoint_refinement and torch.is_grad_enabled():
                hidden, update, mask = checkpoint(self._refine, source, warped, hidden, flow,
                                                  use_reentrant=False, preserve_rng_state=True)
            else:
                hidden, update, mask = self._refine(source, warped, hidden, flow)
            coarse = flow+update[:, :2]
            full, info = self.core.upsample_data(coarse, update[:, 2:], mask)
            full = full[..., top:full.shape[-2]-bottom, left:full.shape[-1]-right]
            info = info[..., top:info.shape[-2]-bottom, left:info.shape[-1]-right]
            require(tuple(full.shape) == (1, 2, *native_hw), "Native readout geometry changed")
            outputs.append(full)
            infos.append(info)
            flow = coarse  # No recurrent-flow or sampling-coordinate detach.
        self.validate()
        return dict(flow=outputs, info=infos)


@torch.no_grad()
def cheap_target(backend: FrozenWAFTBackend, pair: Tensor, gt: Tensor, valid: Tensor,
                 visible: Tensor, *, original: FeaturePair | None = None) -> dict:
    """Selected visible_pull_1. For Final, pass its corresponding Clean pair.

    A cached original must come from this same teacher RGB pair, including
    its second frame. A Final original cannot supply the Clean second field.
    """
    changed, support = reconstruct_visible_rgb(pair, gt, valid, visible)
    original = backend.encode_pair(pair) if original is None else original
    require(original.rgb_sha256 == tensor_sha256(pair), "Cached original belongs to another RGB pair")
    require(original.native_hw == tuple(pair.shape[-2:]), "Original native geometry mismatch")
    padder = backend.padder_factory(pair[:, 0].shape, factor=backend.core.factor)
    require(tuple(padder._pad) == original.padding, "Same complete padded geometry required")
    first = backend._encode(changed[:, 0].to(backend.validate()), padder)
    target = torch.stack((first, original.features[:, 1]), 1)
    return dict(method=CHEAP_METHOD, teacher_features=target.detach(),
                native_hw=original.native_hw, padding=original.padding,
                generation_mask=support, teacher_rgb=changed,
                teacher_original_rgb_sha256=original.rgb_sha256,
                modified_first_encoder_calls=1, optimizer_updates=0,
                feature_projection=None)


def final_epe_loss(flow: Tensor, gt: Tensor, valid: Tensor) -> Tensor:
    """Original FP32 vector-norm objective; no MSE/epsilon surrogate."""
    support = _support(gt, valid)
    require(flow.dtype == torch.float32 and flow.shape == gt.shape and flow.device == gt.device,
            "FP32 native flow must match GT")
    error = (flow-gt).permute(0, 2, 3, 1)[support]
    loss = torch.linalg.vector_norm(error, dim=-1).mean()
    require(torch.isfinite(loss), "Nonfinite final EPE")
    return loss


@torch.no_grad()
def flow_metrics(flow: Tensor, gt: Tensor, valid: Tensor) -> dict:
    support = _support(gt, valid)
    error = (flow.permute(0, 2, 3, 1)[support].double()
             - gt.permute(0, 2, 3, 1)[support].double())
    require(torch.isfinite(error).all(), "Nonfinite scored flow")
    return dict(EPE=float(error.square().sum(-1).sqrt().mean()),
                GT_MSE=float(error.square().mean()), valid_pixels=int(support.sum()))


class _IdentityAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original, displacement):
        return torch.where(displacement == 0, original, original+displacement)

    @staticmethod
    def backward(ctx, gradient):
        return None, gradient


def _field(original, sigma, delta, mask):
    displacement = sigma.reshape(1, 1, 64, 1, 1)*delta
    return torch.where(mask, _IdentityAdd.apply(original, displacement), original)


def expensive_target(backend: FrozenWAFTBackend, original: FeaturePair,
                     gt: Tensor, valid: Tensor) -> dict:
    """Exactly512 per-pair dense AdamW updates, earliest strict best of0..512.

    GT directly supervises only the fifth native flow. All5 recurrent updates
    remain connected, while every original WAFT parameter stays frozen.
    The20% quality flag is retained from the expensive cache contract; it
    never skips a case or terminates optimization early.
    """
    require(torch.is_grad_enabled(), "Dense512 generation needs autograd")
    device = backend.validate()
    base = original.features.detach().clone()
    require(base.device == device and tuple(gt.shape[-2:]) == original.native_hw
            and gt.device == valid.device == device, "Matching original/GT geometry and device")
    _support(gt, valid)
    sigma, mask = original_sigma(base, original.native_hw, original.padding)
    delta = nn.Parameter(torch.zeros_like(base))
    optimizer = torch.optim.AdamW([delta], lr=.001, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., foreach=False, fused=False)
    with torch.no_grad():
        initial = backend.solve(base, original.native_hw, original.padding)
        baseline = flow_metrics(initial['flow'][-1], gt, valid)
        identity = {k: [v.detach().cpu().clone() for v in initial[k]] for k in ('flow', 'info')}
    del initial
    best_delta, best_epe, best_step = delta.detach().clone(), baseline['EPE'], 0
    history, best_search_metrics = [], baseline
    for step in range(513):
        optimizer.zero_grad(set_to_none=True)
        output = backend.solve(_field(base, sigma, delta, mask), original.native_hw, original.padding)
        require(len(output['flow']) == len(output['info']) == 5, "Exactly5 native updates required")
        measured = flow_metrics(output['flow'][-1], gt, valid)
        if step == 0:
            require(all(torch.allclose(value.detach().cpu(), expected, atol=1e-4, rtol=1e-5)
                        for key in ('flow', 'info') for value, expected in zip(output[key], identity[key])),
                    "Original F0 gradmode parity failed")
        if measured['EPE'] < best_epe:
            best_epe, best_step, best_search_metrics = measured['EPE'], step, measured
            best_delta.copy_(delta.detach())
        history.append(dict(step=step, **measured))
        if step == 512:
            break
        loss = final_epe_loss(output['flow'][-1], gt, valid)
        loss.backward()
        backend.validate()
        require(delta.grad is not None and torch.isfinite(delta.grad).all(), "Finite dense gradient required")
        norm = float(torch.linalg.vector_norm(delta.grad.double()))
        coefficient = min(1., 1./(norm+1e-6))
        with torch.no_grad():
            delta.grad.copy_((delta.grad.double()*coefficient).float())
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-.25, .25)
            delta.masked_fill_(~mask.expand_as(delta), 0)
        del output, loss
    del output, identity
    selected = _field(base, sigma, best_delta, mask).detach()
    with torch.no_grad():
        selected_output = backend.solve(selected, original.native_hw, original.padding)
        best_metrics = flow_metrics(selected_output['flow'][-1], gt, valid)
    feature_mse = float((((base.double()-selected.double())/sigma.double().reshape(1, 1, 64, 1, 1))
                         .masked_select(mask.expand_as(base))).square().mean())
    qualified = (baseline['EPE'] > 0 and baseline['GT_MSE'] > 0
                 and 100*(1-best_metrics['EPE']/baseline['EPE']) >= 20.
                 and best_metrics['GT_MSE'] < baseline['GT_MSE'] and feature_mse > 0.)
    return dict(method=EXPENSIVE_METHOD, original=base, teacher_features=selected,
                sigma=sigma, native_mask=mask, selected_delta=best_delta.detach(),
                selected_step=best_step, optimizer_updates=512,
                native_hw=original.native_hw, padding=original.padding,
                baseline_metrics=baseline, target_metrics=best_metrics,
                selected_search_metrics=best_search_metrics, normalized_feature_MSE=feature_mse,
                qualified=qualified, history=history)
