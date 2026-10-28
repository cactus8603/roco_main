"""Synthetic CPU fixtures; these implement no scientific WAFT approximation."""
import torch
from torch import nn
from torch.nn import functional as F

class _TinyRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1)

    def forward(self, value):
        return {"out": torch.tanh(self.conv(value))}


class TinyCore(nn.Module):
    """Explicit synthetic factor112 consumer; encoder is never executed here."""

    def __init__(self):
        super().__init__()
        self.factor = 112
        self.encoder = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.fnet = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.fmap_conv = nn.Conv2d(64, 64, 1)
        self.hidden_conv = nn.Conv2d(128, 4, 1)
        self.warp_linear = nn.Conv2d(64 + 64 + 4 + 2, 4, 1)
        self.refine_net = _TinyRefine()
        self.refine_transform = nn.Conv2d(8, 4, 1)
        self.flow_head = nn.Conv2d(4, 6, 1)
        self.upsample_weight = nn.Conv2d(4, 36, 1)

    @staticmethod
    def normalize_image(value):
        return value / 127.5 - 1

    @staticmethod
    def upsample_data(flow, info, mask):
        full = F.interpolate(2 * flow + .01 * mask[:, :2], scale_factor=2,
                             mode="bilinear", align_corners=True)
        return full, F.interpolate(info, scale_factor=2, mode="bilinear", align_corners=True)


def coordinates(batch, height, width, device):
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device),
                            indexing="ij")
    return torch.stack((xx, yy), 0).float()[None].expand(batch, -1, -1, -1)


def sampler(value, xy):
    height, width = value.shape[-2:]
    normalized = torch.stack((2 * xy[..., 0] / (width - 1) - 1,
                              2 * xy[..., 1] / (height - 1) - 1), -1)
    return F.grid_sample(value, normalized, align_corners=True)



class TapViT(nn.Module):
    patch_size = 14
    embed_dim = 384
    num_register_tokens = 0

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 384, 14, stride=14)
        self.norm = nn.LayerNorm(384, eps=1e-6)


class TapEncoder(nn.Module):
    intermediate_layer_idx = {'vits': [2, 5, 8, 11]}

    def __init__(self):
        super().__init__()
        self.encoder = TapViT()
        self.output = nn.Conv2d(384, 32, 1)

    def forward(self, rgb):
        patches = self.encoder.patch(rgb)
        b, c, h, w = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = torch.cat((tokens.mean(1, keepdim=True), tokens), 1)
        for depth in range(4):
            normalized = self.encoder.norm(tokens + depth)
        value = normalized[:, 1:].transpose(1, 2).reshape(b, c, h, w)
        return F.interpolate(self.output(value), size=(rgb.shape[-2]//2, rgb.shape[-1]//2),
                             mode='bilinear', align_corners=True)


class TapCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.output = nn.Conv2d(64, 32, 1)

    def forward(self, rgb):
        fine = self.conv1(rgb)
        self.conv2(fine)
        return [self.output(fine)]


class ZeroPadder:
    def __init__(self, dims, factor):
        h, w = dims[-2:]
        ph, pw = ((h+8)//factor+1)*factor-h, ((w+8)//factor+1)*factor-w
        self._pad = (pw//2, pw-pw//2, ph//2, ph-ph//2)

    def pad(self, value):
        return F.pad(value, self._pad, mode='constant', value=0)


