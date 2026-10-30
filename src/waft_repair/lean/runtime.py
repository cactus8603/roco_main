"""Lean training math: encoder-only R before one G5; scalar monitoring only."""
import math

import torch
from torch.utils.checkpoint import checkpoint

from ..motion.native import NativeRepairSystem, PreparedRepair
from .activation_async import AsyncDeduplicatedSaveOnCPU


@torch.no_grad()
def sigma_native(original, native_hw, padding):
    h, w = native_hw
    left, right, top, bottom = padding
    fh, fw = original.shape[-2:]
    y, x = torch.meshgrid(torch.arange(fh, device=original.device),
                          torch.arange(fw, device=original.device), indexing='ij')
    support = (2*x >= left) & (2*x < left+w) & (2*y >= top) & (2*y < top+h)
    scales = []
    for pair in original:
        rms = pair[:, :, support].double().square().mean((0, 2)).sqrt()
        scales.append(rms.clamp_min(.01*rms.median()+1e-8).float())
    return torch.stack(scales).detach()


class LeanSystem(NativeRepairSystem):
    """Startup validation is strict; the training loop avoids diagnostic scans."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device = next(self.core.parameters()).device
        self._lean_ready = True
        self._pair_checked = False

    def validate(self):
        if getattr(self, '_lean_ready', False):
            return self._device
        return super().validate()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Construction/binding may happen on CPU before the complete system is
        # moved. Never keep the construction device in the lean fast path.
        self._device = next(self.core.parameters()).device
        self._pair_checked = False
        return result

    @torch.no_grad()
    def prepare_pair(self, image1, image2):
        with torch.autocast(device_type=image1.device.type, enabled=False):
            return self._prepare_pair_fp32(image1, image2)

    def _prepare_pair_fp32(self, image1, image2):
        # Match the original prefix operation order, including normalized taps.
        captures = {key: [] for key in self.restorer.observation_channels}
        vit = self.core.encoder.encoder
        norm_calls = 0
        def cnn(name):
            def capture(module, args, value):
                captures[name].append(value.detach())
            return capture
        def semantic(module, args, value):
            nonlocal norm_calls
            if norm_calls % 4 in (1, 3):
                key = 'dav2_block6' if norm_calls % 4 == 1 else 'dav2_block12'
                captures[key].append(value.detach())
            norm_calls += 1
        handles = [self.core.fnet.conv1.register_forward_hook(cnn('cnn_pre_s2')),
                   self.core.fnet.conv2.register_forward_hook(cnn('cnn_pre_s4')),
                   vit.norm.register_forward_hook(semantic)]
        try:
            one, two = self.core.normalize_image(image1), self.core.normalize_image(image2)
            padder = self._geometry()[0](one.shape, factor=self.core.factor)
            one, two = padder.pad(one), padder.pad(two)
            semantic1, semantic2 = self.core.encoder(one), self.core.encoder(two)
            rgb1, rgb2 = self.core.fnet(one)[0], self.core.fnet(two)[0]
            source = self.core.fmap_conv(torch.cat((semantic1, rgb1), 1))
            target = self.core.fmap_conv(torch.cat((semantic2, rgb2), 1))
        finally:
            for handle in handles:
                handle.remove()
        hp, wp = one.shape[-2:]
        observations = {}
        registers = vit.num_register_tokens
        for key, values in captures.items():
            if key.startswith('dav2'):
                values = [v[:, 1+registers:].transpose(1, 2).reshape(-1, 384, hp//14, wp//14) for v in values]
            observations[key] = torch.stack(values, 1).detach()
        original = torch.stack((source, target), 1).detach()
        native_hw, padding = tuple(image1.shape[-2:]), tuple(padder._pad)
        scale = sigma_native(original, native_hw, padding)
        if not self._pair_checked:
            if norm_calls != 8 or any(len(values) != 2 for values in captures.values()):
                raise ValueError('Original same-pass four taps required')
            self.restorer._validate(original, scale, observations, native_hw, padding)
            self._pair_checked = True
        return PreparedRepair(original, scale, observations, native_hw, padding)

    def repair_features(self, prepared, *, return_aux=False):
        return self.restorer(prepared.original, prepared.sigma, prepared.observations,
                             native_hw=prepared.native_hw, padding=prepared.padding, return_aux=return_aux)

    def solve_features(self, features, *, native_hw, padding, trace=False, final_only=False):
        source, target = features[:, 0], features[:, 1]
        b, _, h, w = source.shape
        left, right, top, bottom = padding
        _, coords_grid, sampler = self._geometry()
        hidden = self.core.hidden_conv(torch.cat((source, target), 1))
        flow = source.new_zeros((b, 2, h, w))
        base_coordinates = coords_grid(b, h, w, device=source.device)
        flows, infos = [], []
        for step in range(5):
            flow = flow.detach()
            coordinates = (base_coordinates+flow).detach()
            warped = sampler(target, coordinates.permute(0, 2, 3, 1))
            if self.checkpoint_refinement and torch.is_grad_enabled():
                hidden, update, mask = checkpoint(self._refine, source, warped, hidden, flow,
                    use_reentrant=False, preserve_rng_state=True)
            else:
                hidden, update, mask = self._refine(source, warped, hidden, flow)
            coarse = flow+update[:, :2]
            if not final_only or step == 4:
                full, info = self.core.upsample_data(coarse, update[:, 2:], mask)
                flows.append(full[..., top:full.shape[-2]-bottom, left:full.shape[-1]-right])
                infos.append(info[..., top:info.shape[-2]-bottom, left:info.shape[-1]-right])
            flow = coarse
        return dict(flow=flows, info=infos)


def make_system(core, repair, *, expected_core_digest=None, strict_official=True, checkpoint_refinement=True, **geometry):
    if strict_official and (not isinstance(expected_core_digest, str) or len(expected_core_digest) != 64):
        raise ValueError('Official selected checkpoint binding required')
    repair.validate_inputs = False
    return LeanSystem(core, repair, expected_core_digest=expected_core_digest,
        strict_official=strict_official, checkpoint_refinement=checkpoint_refinement, **geometry)


def make_optimizer(system, *, learning_rate=3e-4):
    return torch.optim.AdamW(system.restorer.parameters(), lr=learning_rate, eps=1e-8,
                             weight_decay=0., foreach=False)


def native_loss(output, gt, valid):
    """Original finite native5 MoL arithmetic, without per-tensor sync scans."""
    if len(output['flow']) != 5 or len(output['info']) != 5:
        raise ValueError('Exactly five native outputs required')
    with torch.no_grad():
        support = valid & (torch.linalg.vector_norm(gt.detach().double(), dim=1) < 40000.)
    target = gt.detach().permute(0, 2, 3, 1)[support]
    count = target.numel()
    if not count:
        raise ValueError('No supervised pixels')
    numerator = None
    for i, (flow, info) in enumerate(zip(output['flow'], output['info'])):
        prediction = flow.permute(0, 2, 3, 1)[support]
        parameters = info.permute(0, 2, 3, 1)[support]
        weights = parameters[:, :2]
        raw_scale = parameters[:, 2:]
        scale = torch.stack((raw_scale[:, 0].clamp(0, 10), raw_scale[:, 1].clamp(0, 0)), dim=1)
        distance = (target.to(prediction.dtype)-prediction).abs()
        components = weights-math.log(2.)-scale
        components = components[:, None, :]-distance[:, :, None]*torch.exp(-scale)[:, None, :]
        nll = torch.logsumexp(weights, dim=1)[:, None]-torch.logsumexp(components, dim=2)
        term = .85**(4-i)*nll.sum()
        numerator = term if numerator is None else numerator+term
    return numerator/count


def _inputs(system, image_pair, gt, valid):
    device = system.validate()
    return tuple(x.to(device) for x in image_pair), gt.to(device), valid.to(device)


def train_step(system, optimizer, image_pair, gt, valid, *, learning_rate, activation_offload=True):
    if not system.training:
        system.train()
    pair, gt, valid = _inputs(system, image_pair, gt, valid)
    optimizer.param_groups[0]['lr'] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    from contextlib import nullcontext
    context = AsyncDeduplicatedSaveOnCPU(max_resident_cpu_bytes=8 << 30, async_d2h=True) if activation_offload else nullcontext()
    with context:
        output = system(*pair)
        loss = native_loss(output, gt, valid)
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        gradients = [p.grad for p in system.restorer.parameters()]
        if any(value is None for value in gradients):
            raise RuntimeError('Missing R gradient')
        norm = torch.stack([g.detach().double().square().sum() for g in gradients]).sum().sqrt()
        norm_value = float(norm)
        if not math.isfinite(norm_value):
            raise FloatingPointError('Nonfinite gradient norm')
        scale = min(1., 1./(norm_value+1e-12))
        if scale < 1:
            for gradient in gradients:
                gradient.mul_(scale)
        optimizer.step()
    del output, loss, gradients, norm, pair, gt, valid
    return dict(loss=loss_value, gradient_norm=norm_value)


@torch.no_grad()
def evaluate(system, image_pair, gt, valid, *, compute_mol=True):
    if system.training:
        system.eval()
    pair, gt, valid = _inputs(system, image_pair, gt, valid)
    if compute_mol:
        output = system(*pair)
    else:
        prepared = system.prepare_pair(*pair)
        with torch.autocast(device_type=prepared.original.device.type, enabled=False):
            repaired = system.repair_features(prepared)
            output = system.solve_features(repaired, native_hw=prepared.native_hw,
                padding=prepared.padding, final_only=True)
    distance = (output['flow'][-1].double()-gt.double()).permute(0, 2, 3, 1)[valid]
    epe = float(torch.linalg.vector_norm(distance, dim=1).mean())
    if not math.isfinite(epe):
        raise FloatingPointError('Nonfinite evaluation')
    result = dict(EPE=epe)
    if compute_mol:
        loss = float(native_loss(output, gt, valid))
        if not math.isfinite(loss):
            raise FloatingPointError('Nonfinite evaluation')
        result['native_MoL'] = loss
    return result
