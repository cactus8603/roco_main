"""Tiny CPU arithmetic fixtures; no official model, data, or CUDA is used."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from waft_repair.lean import LiteFeatureRepair, make_system
from waft_repair.motion.native import NativeRepairSystem, native_task_loss
from motion_helpers import TapEncoder, TapCNN, coordinates, sampler


class Padder:
    def __init__(self, shape, factor):
        h, w = shape[-2:]
        ph, pw = ((h+8)//factor+1)*factor-h, ((w+8)//factor+1)*factor-w
        self._pad = (pw//2, pw-pw//2, ph//2, ph-ph//2)

    def pad(self, value):
        return F.pad(value, self._pad, mode='replicate')


class Refine(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 1)

    def forward(self, value):
        return {'out': torch.tanh(self.conv(value))}


class Core(nn.Module):
    """Synthetic complete consumer ABI; never an official WAFT approximation."""
    factor = 112

    def __init__(self):
        super().__init__()
        self.encoder = TapEncoder()
        self.fnet = TapCNN()
        self.fmap_conv = nn.Conv2d(64, 64, 1)
        self.hidden_conv = nn.Conv2d(128, 8, 1)
        self.warp_linear = nn.Conv2d(64+64+8+2, 8, 1)
        self.refine_net = Refine()
        self.refine_transform = nn.Conv2d(16, 8, 1)
        self.flow_head = nn.Conv2d(8, 6, 1)
        self.upsample_weight = nn.Conv2d(8, 36, 1)

    @staticmethod
    def normalize_image(rgb):
        return rgb/127.5-1

    @staticmethod
    def upsample_data(flow, info, mask):
        return (F.interpolate(2*flow+.01*mask[:, :2], scale_factor=2, mode='bilinear', align_corners=True),
                F.interpolate(info, scale_factor=2, mode='bilinear', align_corners=True))


def systems():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        core = Core()
    geometry = dict(padder_factory=Padder, coords_grid_fn=coordinates, bilinear_sampler_fn=sampler)
    reference = NativeRepairSystem(core, LiteFeatureRepair(), strict_official=False, **geometry)
    lean = make_system(copy.deepcopy(core), LiteFeatureRepair(), strict_official=False, **geometry)
    return reference, lean


def reference_step(system, optimizer, image_pair, gt, valid, *, learning_rate):
    """Original all-five native loss and FP64 clipping on the reference graph."""
    system.train()
    optimizer.param_groups[0]['lr'] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    output = system(*image_pair)
    loss = native_task_loss(output, gt, valid)
    loss_value = float(loss.detach())
    loss.backward()
    gradients = [p.grad for p in system.restorer.parameters()]
    norm = torch.stack([g.detach().double().square().sum() for g in gradients]).sum().sqrt()
    norm_value = float(norm)
    scale = min(1., 1./(norm_value+1e-12))
    if scale < 1:
        for gradient in gradients:
            gradient.mul_(scale)
    optimizer.step()
    return dict(loss=loss_value, gradient_norm=norm_value)
