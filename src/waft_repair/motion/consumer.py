"""GT-driven feature repair through an entirely frozen native WAFT consumer.

No weights, data, optimizer or checkpoint are loaded here. The caller supplies
the original loaded WAFT and a separate pair-feature restorer. Clean and Final
are ordinary input pairs to the same model; neither is an inference sidecar.
Only the original encoder path uses no_grad. The five-update consumer remains
differentiable with respect to its repaired feature inputs, with the original
flow/query detach operations preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def require(condition, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _field(value: Tensor, name: str, channels: int, device=None, *, finite: bool = True) -> None:
    require(isinstance(value, Tensor) and value.device.type in ("cpu", "cuda"),
            name + " must be a materialized CPU/CUDA tensor")
    require(value.dtype == torch.float32 and value.layout == torch.strided
            and value.ndim == 4 and value.shape[0] > 0
            and value.shape[1] == channels and min(value.shape[-2:]) > 0,
            name + " must be a nonempty dense FP32 B,C,H,W field")
    require(device is None or value.device == device, name + " device differs")
    if finite:
        require(torch.isfinite(value).all(), name + " must be finite")


def _storage_keys(module: nn.Module) -> set:
    return {(str(p.device), p.untyped_storage().data_ptr())
            for p in module.parameters() if p.numel()}


@dataclass(frozen=True)
class PreparedFlowFeatures:
    normalized_rgb: Tensor       # B,2,3,Hpad,Wpad
    native_features: Tensor      # B,2,64,Hpad/2,Wpad/2; frozen anchor
    padding: tuple[int, int, int, int]
    native_hw: tuple[int, int]


class FrozenWAFTFeaturePipelineV1(nn.Module):
    """One trainable restorer; the original complete WAFT stays eval/frozen.

    The restorer accepts normalized padded RGB and detached fused64 maps, with
    explicit native_hw/padding, and returns repaired maps in each endpoint's
    own coordinate grid. It must not register or alias this WAFT instance.
    ``strict_official=False`` is exclusively for injected synthetic fixtures.
    """
    STEPS = 5

    def __init__(self, core: nn.Module, restorer: nn.Module, *,
                 strict_official: bool = True, checkpoint_refinement: bool = True,
                 padder_factory: Callable | None = None,
                 coords_grid_fn: Callable | None = None,
                 bilinear_sampler_fn: Callable | None = None):
        super().__init__()
        require(isinstance(core, nn.Module) and isinstance(restorer, nn.Module),
                "Explicit loaded core and independent restorer required")
        require(type(strict_official) is bool and type(checkpoint_refinement) is bool,
                "Explicit strict/checkpoint booleans required")
        require(all(p.grad is None for p in core.parameters()),
                "Supplied WAFT must have no pending parameter gradients")
        self.core = core.eval().requires_grad_(False)
        self.restorer = restorer
        self.strict_official = strict_official
        self.checkpoint_refinement = checkpoint_refinement
        self._padder_factory = padder_factory
        self._coords_grid_fn = coords_grid_fn
        self._bilinear_sampler_fn = bilinear_sampler_fn
        require(not (_storage_keys(core) & _storage_keys(restorer)),
                "The new restorer must not alias any original WAFT parameter")
        self.validate()

    def train(self, mode: bool = True):
        require(type(mode) is bool, "Training mode must be boolean")
        self.training = mode
        self.restorer.train(mode)
        self.core.eval()
        return self

    def validate(self) -> torch.device:
        require(not any(m.training for m in self.core.modules())
                and all(not p.requires_grad and p.grad is None for p in self.core.parameters()),
                "Every original WAFT module must stay eval/frozen without gradients")
        parameters = list(self.core.parameters())
        require(bool(parameters), "Populated frozen WAFT required")
        device = parameters[0].device
        require(device.type in ("cpu", "cuda")
                and all(p.device == device and p.dtype == torch.float32 for p in parameters),
                "Original WAFT must share a real FP32 device")
        require(type(getattr(self.core, "factor", None)) is int
                and self.core.factor > 0 and self.core.factor % 2 == 0,
                "An even native padding factor is required")
        for name in ("encoder", "fnet", "fmap_conv", "hidden_conv", "warp_linear",
                     "refine_net", "refine_transform", "flow_head", "upsample_weight"):
            require(isinstance(getattr(self.core, name, None), nn.Module), "Missing original module: " + name)
        require(callable(getattr(self.core, "normalize_image", None))
                and callable(getattr(self.core, "upsample_data", None)),
                "Original preprocessing and readout required")
        if self.strict_official:
            require(self.core.factor == 112 and len(parameters) == 508
                    and len(self.core.state_dict()) == 508
                    and sum(p.numel() for p in parameters) == 56_932_619
                    and not list(self.core.buffers()), "Exact complete DAV2 A2 WAFT required")
        repair_parameters = list(self.restorer.parameters())
        require(bool(repair_parameters)
                and all(p.device == device and p.dtype == torch.float32 and p.requires_grad
                        for p in repair_parameters),
                "Only an independent trainable FP32 restorer is supported")
        require(not (_storage_keys(self.core) & _storage_keys(self.restorer)),
                "Original WAFT and restorer storage must remain disjoint")
        return device

    def restoration_parameters(self):
        return self.restorer.parameters()

    def trainable_named_parameters(self):
        return (("restorer." + name, p) for name, p in self.restorer.named_parameters())

    def _geometry(self):
        if any(fn is None for fn in (self._padder_factory, self._coords_grid_fn, self._bilinear_sampler_fn)):
            from utils.utils import Padder, coords_grid, bilinear_sampler
            return (self._padder_factory or Padder, self._coords_grid_fn or coords_grid,
                    self._bilinear_sampler_fn or bilinear_sampler)
        return self._padder_factory, self._coords_grid_fn, self._bilinear_sampler_fn

    @torch.no_grad()
    def prepare(self, image1: Tensor, image2: Tensor) -> PreparedFlowFeatures:
        device = self.validate()
        for image in (image1, image2):
            _field(image, "Observed RGB", 3, device)
            require(((image >= 0) & (image <= 255)).all(), "Observed RGB must lie in [0,255]")
        require(image1.shape == image2.shape, "The two observed frames must share geometry")
        factory, _, _ = self._geometry()
        one, two = self.core.normalize_image(image1), self.core.normalize_image(image2)
        padder = factory(one.shape, factor=self.core.factor)
        one, two = padder.pad(one), padder.pad(two)
        # Preserve the exact official extractor operation order.
        semantic1, semantic2 = self.core.encoder(one), self.core.encoder(two)
        rgb1, rgb2 = self.core.fnet(one)[0], self.core.fnet(two)[0]
        source = self.core.fmap_conv(torch.cat((semantic1, rgb1), dim=1))
        target = self.core.fmap_conv(torch.cat((semantic2, rgb2), dim=1))
        return PreparedFlowFeatures(torch.stack((one, two), 1).detach(),
            torch.stack((source, target), 1).detach(), tuple(padder._pad), tuple(image1.shape[-2:]))

    def _refine(self, source, warped, hidden, flow):
        value = self.core.warp_linear(torch.cat((source, warped, hidden, flow), 1))
        value = self.core.refine_net(value)
        hidden = self.core.refine_transform(torch.cat((value["out"], hidden), 1))
        update = self.core.flow_head(hidden)
        mask = .25 * self.core.upsample_weight(hidden)
        return hidden, update, mask

    def solve_features(self, features: Tensor, *, padding: tuple[int, int, int, int],
                       native_hw: tuple[int, int], trace: bool = False) -> dict:
        device = self.validate()
        require(type(trace) is bool, "Trace must be boolean")
        require(isinstance(features, Tensor) and features.ndim == 5
                and features.shape[1:3] == (2, 64), "Own-grid B,2,64,Hf,Wf features required")
        source, target = features[:, 0], features[:, 1]
        _field(source, "Repaired source", 64, device)
        _field(target, "Repaired target", 64, device)
        require(isinstance(padding, tuple) and len(padding) == 4
                and all(type(v) is int and v >= 0 for v in padding), "Nonnegative native padding tuple required")
        require(isinstance(native_hw, tuple) and len(native_hw) == 2
                and all(type(v) is int and v > 0 for v in native_hw), "Positive native H,W required")
        b, _, h, w = source.shape
        left, right, top, bottom = padding
        require((2*h, 2*w) == (native_hw[0]+top+bottom, native_hw[1]+left+right)
                and h > 1 and w > 1
                and (2*h) % self.core.factor == (2*w) % self.core.factor == 0,
                "Stride-two geometry must match original native padding")
        _, coords_grid, sampler = self._geometry()
        hidden = self.core.hidden_conv(torch.cat((source, target), 1))
        flow = source.new_zeros((b, 2, h, w))
        outputs, infos, records = [], [], []
        for step in range(self.STEPS):
            flow = flow.detach()
            coordinates = (coords_grid(b, h, w, device=device) + flow).detach()
            warped = sampler(target, coordinates.permute(0, 2, 3, 1))
            if self.checkpoint_refinement and torch.is_grad_enabled():
                hidden, update, mask = checkpoint(self._refine, source, warped, hidden, flow,
                                                  use_reentrant=False, preserve_rng_state=True)
            else:
                hidden, update, mask = self._refine(source, warped, hidden, flow)
            coarse = flow + update[:, :2]
            full, info = self.core.upsample_data(coarse, update[:, 2:], mask)
            full = full[..., top:full.shape[-2]-bottom, left:full.shape[-1]-right]
            info = info[..., top:info.shape[-2]-bottom, left:info.shape[-1]-right]
            _field(full, "Native flow", 2, device)
            _field(info, "Native info", 4, device)
            require(tuple(full.shape) == (b, 2, *native_hw)
                    and tuple(info.shape) == (b, 4, *native_hw), "Native output geometry differs")
            outputs.append(full)
            infos.append(info)
            if trace:
                records.append(dict(step=step+1, flow=full.detach().clone(),
                    coarse=coarse.detach().clone(), coordinates=coordinates.detach().clone()))
            flow = coarse
        return dict(flow=outputs, info=infos, refinement_trace=records)

    def forward(self, image1: Tensor, image2: Tensor, *, diagnostics: bool = False) -> dict:
        require(type(diagnostics) is bool, "Diagnostics must be boolean")
        prepared = self.prepare(image1, image2)
        device = prepared.native_features.device
        # Frozen weights still provide a live consumer-to-feature graph.
        with torch.autocast(device_type=device.type, enabled=False):
            repaired = self.restorer(prepared.normalized_rgb, prepared.native_features,
                native_hw=prepared.native_hw, padding=prepared.padding,
                return_diagnostics=diagnostics)
            alignment = {}
            if diagnostics:
                repaired, alignment = repaired
            result = self.solve_features(repaired, padding=prepared.padding,
                                         native_hw=prepared.native_hw, trace=diagnostics)
        result.update(features=repaired, base_features=prepared.native_features,
            padding=prepared.padding, native_hw=prepared.native_hw,
            restoration_diagnostics=alignment)
        return result

    @torch.no_grad()
    def predict(self, image1: Tensor, image2: Tensor, *, diagnostics: bool = False) -> dict:
        require(not self.training, "Call eval() before prediction")
        return self(image1, image2, diagnostics=diagnostics)


@dataclass(frozen=True)
class FinalEPEObjective:
    loss: Tensor
    numerator: Tensor
    valid_pixels: int
    excluded_nonfinite_GT: int


def final_native_epe(predictions: Sequence[Tensor], gt: Tensor, valid: Tensor) -> FinalEPEObjective:
    """Exact final-update vector EPE on finite valid GT; no feature target.

    Valid occluded and image-OOB endpoints remain in the optical-flow metric.
    GT is detached; no per-component mean, magnitude cap, epsilon surrogate or
    intermediate-update training objective is substituted for final EPE.
    """
    require(isinstance(predictions, (tuple, list)) and len(predictions) == 5,
            "All five original native readouts are required")
    _field(gt, "Native GT flow", 2, finite=False)
    require(isinstance(valid, Tensor) and valid.device == gt.device
            and valid.dtype == torch.bool and tuple(valid.shape) == (gt.shape[0], *gt.shape[-2:]),
            "Valid GT mask must be Boolean B,H,W on the native device")
    finite = torch.isfinite(gt.detach()).all(1)
    support = valid & finite
    count = int(support.sum())
    require(count > 0, "No finite valid GT pixels")
    final = predictions[-1]
    require(isinstance(final, Tensor) and final.shape == gt.shape and final.device == gt.device
            and final.dtype == torch.float32, "Final native flow must match GT geometry/device")
    values = final.permute(0, 2, 3, 1)[support]
    require(torch.isfinite(values).all(), "Nonfinite predicted flow on valid GT support")
    target = gt.detach().permute(0, 2, 3, 1)[support]
    numerator = torch.linalg.vector_norm(values-target, dim=1).sum(dtype=torch.float32)
    require(torch.isfinite(numerator), "Final EPE sum is nonfinite")
    return FinalEPEObjective(numerator/count, numerator, count, int((valid & ~finite).sum()))


def clean_final_task_loss(clean_predictions: Sequence[Tensor], final_predictions: Sequence[Tensor],
                          gt: Tensor, valid: Tensor) -> Tensor:
    """Equal-render mean of actual final EPE, with one shared repair model."""
    clean = final_native_epe(clean_predictions, gt, valid)
    final = final_native_epe(final_predictions, gt, valid)
    return .5*(clean.loss + final.loss)
