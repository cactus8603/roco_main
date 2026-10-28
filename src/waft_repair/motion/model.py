"""Motion-aligned own-grid repair, conditioned only on frozen observations.

The forward baseline lives on frame1. Frame2 context is a normalized forward
splat with explicit holes/collision mass, never a claimed backward-flow solve.
No labels, teacher features, sample identifiers or learned position tables are
accepted. The original full-padded F0/sigma amplitudes have direct linear paths.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .finite import all_tensors_finite
from .geometry import native_support, pixel_grid, resample_grid, sample_at_xy, sample_support


def _geometry(native_hw, padding):
    if (not isinstance(native_hw, (tuple, list)) or len(native_hw) != 2
            or any(type(v) is not int or v <= 0 for v in native_hw)
            or not isinstance(padding, (tuple, list)) or len(padding) != 4
            or any(type(v) is not int or v < 0 for v in padding)):
        raise ValueError('Positive native H,W and nonnegative integer L,R,T,B required')
    h, w = native_hw; left, right, top, bottom = padding
    hp, wp = h+top+bottom, w+left+right
    if hp % 112 or wp % 112:
        raise ValueError('Original WAFT padded geometry must be divisible by112')
    return hp, wp


@dataclass(frozen=True)
class MotionRepairConfig:
    profile: str = 'motion_aligned_v1'
    fine_dim: int = 32
    s4_dim: int = 96
    s8_dim: int = 128
    s16_dim: int = 192
    s4_blocks: int = 2
    s8_blocks: int = 3
    s16_blocks: int = 6
    decoder_blocks: int = 2
    seed: int = 20261028

    def __post_init__(self):
        if self.profile not in ('motion_aligned_v1', 'test'):
            raise ValueError('Unknown motion repair profile')
        names = ('fine_dim','s4_dim','s8_dim','s16_dim','s4_blocks','s8_blocks','s16_blocks','decoder_blocks')
        if any(type(getattr(self,name)) is not int or getattr(self,name) <= 0 for name in names):
            raise ValueError('Positive integer dimensions and block counts required')
        if type(self.seed) is not int:
            raise ValueError('Integer seed required')
        if self.profile == 'motion_aligned_v1' and tuple(getattr(self,n) for n in names) != (32,96,128,192,2,3,6,2):
            raise ValueError('Scientific motion_aligned_v1 shape is fixed; custom CPU fixtures use test')


@dataclass(frozen=True)
class MotionPrior:
    """Detached derived inputs; no model/GT/case state or native labels."""
    native_hw: tuple[int, int]
    padding: tuple[int, int, int, int]
    padded_hw: tuple[int, int]
    gather_xy_s4: Tensor
    gather_valid_s4: Tensor
    splat_indices: Tensor
    splat_weights: Tensor
    collision_mass_s4: Tensor
    evidence_s4: Tensor
    native_s2: Tensor
    native_s4: Tensor
    native_s16: Tensor

    def to(self, device):
        return replace(self, **{field.name:getattr(self,field.name).to(device=device)
                                for field in fields(self) if isinstance(getattr(self,field.name),Tensor)})


def _deterministic_scatter_add(output,dimension,indices,source):
    """Select Torch2.5's deterministic CUDA scatter_add_ only for this call.

    Global deterministic mode would reject the separate grid_sample backward.
    Dispatch selects the deterministic scatter implementation before this flag
    is restored. Its source gradient is a gather; the four source-corner
    contributions are reduced by the ordinary fixed-size broadcast backward.
    """
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True,warn_only=False)
        return output.scatter_add_(dimension,indices,source)
    finally:
        torch.use_deterministic_algorithms(previous,warn_only=warn_only)


def _splat(value, indices, weights, mass):
    """One vectorized scatter, four bilinear corners, normalized by actual mass."""
    batch, channels, height, width = value.shape
    count = height*width
    source = (value.reshape(batch,channels,1,count)*weights[:,None]).reshape(batch,channels,4*count)
    index = indices.reshape(batch,1,4*count).expand(-1,channels,-1)
    numerator = _deterministic_scatter_add(value.new_zeros(batch,channels,count),2,index,source)
    result = numerator.reshape(batch,channels,height,width)/mass.clamp_min(1e-12)
    return result * (mass > 0).to(value.dtype)


def gather_frame2_to_frame1(value: Tensor, prior: MotionPrior):
    """Read frame2 at p+f12(p), leaving unsupported frame1 sites exactly zero."""
    return sample_at_xy(value,prior.gather_xy_s4)*prior.gather_valid_s4.to(value.dtype)


def splat_frame1_to_frame2(value: Tensor, prior: MotionPrior):
    """Transport frame1 values to frame2; holes are zero and collisions average."""
    return _splat(value,prior.splat_indices,prior.splat_weights,prior.collision_mass_s4)


@torch.no_grad()
def prepare_motion_prior(baseline_flow: Tensor, baseline_info: Tensor, *, native_hw, padding):
    """Cache GT-free alignment once per original baseline/geometry.

    flow/info are the unchanged original WAFT fifth native readout. Info on
    frame2 is transported forward uncertainty, not backward uncertainty.
    Directional frame2 displacement is transported -f12, not true inverse flow.
    """
    hp, wp = _geometry(native_hw,padding)
    h, w = native_hw
    if (not isinstance(baseline_flow,Tensor) or baseline_flow.ndim != 4
            or tuple(baseline_flow.shape[1:]) != (2,h,w) or baseline_flow.shape[0] < 1):
        raise ValueError('Native baseline fifth flow must be [B,2,H,W]')
    batch = baseline_flow.shape[0]
    if (not isinstance(baseline_info,Tensor) or tuple(baseline_info.shape) != (batch,4,h,w)
            or any(v.dtype != torch.float32 or v.device != baseline_flow.device
                   or v.requires_grad or v.grad_fn is not None for v in (baseline_flow,baseline_info))
            or baseline_flow.device.type not in ('cpu','cuda')
            or not all_tensors_finite((baseline_flow,baseline_info))):
        raise ValueError('Finite fixed same-device FP32 baseline flow/info required')
    device = baseline_flow.device; hw4 = (hp//4,wp//4); n = hw4[0]*hw4[1]
    masks = {stride:native_support(native_hw,padding,(hp//stride,wp//stride),stride,0.,device=device)
             for stride in (2,4,16)}
    physical = 4*pixel_grid(hw4,device=device)
    native_xy = physical - physical.new_tensor((padding[0],padding[2]))
    own_support = masks[4].expand(batch,-1,-1,-1)
    # S4 centers and integer L/T put every query at an integer native pixel.
    # Integer gather avoids grid_sample's normalized-coordinate roundoff,
    # which can mix a sharply different neighboring flow into an exact query.
    native_index = (native_xy[...,1].long().clamp(0,h-1)*w
                    +native_xy[...,0].long().clamp(0,w-1)).reshape(1,1,n)
    observed = torch.cat((baseline_flow,baseline_info),1).flatten(2).gather(
        2,native_index.expand(batch,6,n)).reshape(batch,6,*hw4)*own_support
    flow = observed[:,:2]*own_support
    # Remove the meaningless common logit offset; retain both native scale
    # settings explicitly (component2 has fixed log_scale0 in original WAFT).
    info = torch.cat((observed[:,2:4]-observed[:,2:4].mean(1,keepdim=True),
                      observed[:,4:5].clamp(0.,10.),observed[:,5:6].clamp(0.,0.)),1)*own_support
    endpoint = (physical[None]+flow.permute(0,2,3,1))/4
    valid = own_support & sample_support(masks[4],endpoint)[:,None]
    flat = endpoint.reshape(batch,n,2); low = torch.floor(flat).long(); fraction = flat-low
    offsets = torch.tensor(((0,0),(1,0),(0,1),(1,1)),device=device)
    corners = low[:,None]+offsets[None,:,None]
    x,y = corners[...,0],corners[...,1]
    weights = torch.stack(((1-fraction[...,0])*(1-fraction[...,1]),
                           fraction[...,0]*(1-fraction[...,1]),
                           (1-fraction[...,0])*fraction[...,1],
                           fraction[...,0]*fraction[...,1]),1)
    inside = (x >= 0)&(x < hw4[1])&(y >= 0)&(y < hw4[0])
    weights = weights*inside*valid.reshape(batch,1,n)
    indices = y.clamp(0,hw4[0]-1)*hw4[1]+x.clamp(0,hw4[1]-1)
    mass = _deterministic_scatter_add(flow.new_zeros(batch,1,n),2,
                                     indices.reshape(batch,1,4*n),weights.reshape(batch,1,4*n))
    mass = mass.reshape(batch,1,*hw4)
    transported = _splat(torch.cat((-flow,info),1),indices,weights,mass)
    scale = flow.new_tensor((w,h)).view(1,2,1,1)
    forward_displacement, inverse_proposal = flow/scale,transported[:,:2]/scale
    target_density = sample_at_xy(mass,endpoint)*valid
    def evidence(displacement, uncertainty, support, available, density):
        magnitude = displacement.square().sum(1,keepdim=True).sqrt()
        return torch.cat((displacement,magnitude,uncertainty,support.float(),available.float(),torch.log1p(density)),1)
    e0 = evidence(forward_displacement,info,own_support,valid,target_density)
    e1 = evidence(inverse_proposal,transported[:,2:],own_support,mass>0,mass)
    result = MotionPrior(tuple(native_hw),tuple(padding),(hp,wp),endpoint.detach(),valid.detach(),
        indices.detach(),weights.detach(),mass.detach(),torch.stack((e0,e1),1).detach(),
        masks[2],masks[4],masks[16])
    if not all_tensors_finite(v for v in (result.gather_xy_s4,result.splat_weights,result.evidence_s4,result.collision_mass_s4)):
        raise FloatingPointError('Nonfinite derived alignment prior')
    return result


class _LayerNorm2D(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.norm = nn.LayerNorm(channels,eps=1e-6)

    def forward(self,value):
        return self.norm(value.permute(0,2,3,1)).permute(0,3,1,2).contiguous()


class _Block(nn.Module):
    def __init__(self,channels,expansion):
        super().__init__()
        self.norm = _LayerNorm2D(channels)
        self.depthwise = nn.Conv2d(channels,channels,3,padding=1,groups=channels)
        self.mlp = nn.Sequential(nn.Conv2d(channels,channels*expansion,1),nn.GELU(),
                                 nn.Conv2d(channels*expansion,channels,1))

    def forward(self,value):
        return value+self.mlp(self.depthwise(self.norm(value)))


class _IdentityResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx,original,displacement):
        return torch.where(displacement == 0,original,original+displacement)

    @staticmethod
    def backward(ctx,gradient):
        return None,gradient


def _blocks(channels,count,expansion):
    return nn.Sequential(*(_Block(channels,expansion) for _ in range(count)))


class MotionFeatureRepair(nn.Module):
    """Shared directional convolutional U-shape plus full64 raw-amplitude paths."""
    observation_channels = dict(cnn_pre_s2=64,cnn_pre_s4=128,dav2_block6=384,dav2_block12=384)

    def __init__(self,config: MotionRepairConfig | None = None):
        super().__init__(); self.config = MotionRepairConfig() if config is None else config
        if not isinstance(self.config,MotionRepairConfig):
            raise TypeError('MotionRepairConfig required')
        c = self.config
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(c.seed)
            self.fine_stem = nn.Conv2d(128,c.fine_dim,1)
            self.fine_down = nn.Conv2d(c.fine_dim,64,3,stride=2,padding=1)
            self.cnn4_stem = nn.Conv2d(128,64,1)
            self.dav6_stem = nn.Conv2d(384,c.s4_dim//2,1)
            self.dav12_stem = nn.Conv2d(384,c.s4_dim-c.s4_dim//2,1)
            self.semantic_mix = nn.Conv2d(c.s4_dim,c.s4_dim,1)
            self.own_mix = nn.Conv2d(128+c.s4_dim,c.s4_dim,1)
            self.match_mix = nn.Conv2d(2*c.s4_dim+128+10,c.s4_dim,1)
            self.encoder4 = _blocks(c.s4_dim,c.s4_blocks,2)
            self.down4 = nn.Conv2d(c.s4_dim,c.s8_dim,3,stride=2,padding=1)
            self.encoder8 = _blocks(c.s8_dim,c.s8_blocks,3)
            self.down8 = nn.Conv2d(c.s8_dim,c.s16_dim,3,stride=2,padding=1)
            self.deep = _blocks(c.s16_dim,c.s16_blocks,4)
            self.up8 = nn.Conv2d(c.s16_dim+c.s8_dim,c.s8_dim,1)
            self.decoder8 = _blocks(c.s8_dim,c.decoder_blocks,2)
            self.up4 = nn.Conv2d(c.s8_dim+c.s4_dim,c.s4_dim,1)
            self.decoder4 = _blocks(c.s4_dim,c.decoder_blocks,2)
            self.fine_mix = nn.Conv2d(c.s4_dim+c.fine_dim,c.fine_dim,1)
            self.fine_block = _Block(c.fine_dim,2)  # Exactly one S2 local block.
            self.spatial_head = nn.Conv2d(c.fine_dim,64,1,bias=False)
            self.raw_own_head = nn.Conv2d(64,64,1,bias=False)
            self.raw_match_head = nn.Conv2d(128,64,1,bias=False)
            self.global_hidden = nn.Sequential(nn.Linear(c.s16_dim+128,c.s16_dim),nn.GELU())
            self.global_head = nn.Linear(c.s16_dim,64)
            for head in (self.spatial_head,self.raw_own_head,self.raw_match_head,self.global_head):
                nn.init.zeros_(head.weight)
                if head.bias is not None: nn.init.zeros_(head.bias)

    def configuration(self):
        return dict(**asdict(self.config),model_abi='paired-feature-repair/motion-aligned-v1',
            observation_channels=dict(self.observation_channels),
            output='F0+original_sigma*(global64+spatial64-mean_fullpad_HW(spatial64))',
            alignment='F2→F1 gather by detached f12; F1→F2 normalized bilinear forward splat with holes/collision mass',
            frame2_flow='Transported -f12 proposal, not a solved backward flow',
            frame2_info='Transported forward mixture uncertainty, not backward uncertainty',
            raw_amplitude_paths='Full64 F0/sigma linear head plus full64 S4 aligned signed/absolute raw residual head; no LN/projection bottleneck',
            parameter_count=sum(v.numel() for v in self.parameters()))

    def cost_estimate(self,padded_hw,*,batch=1):
        """Static Conv2d/Linear MAC census, not a latency benchmark.

        A multiply-add counts as one MAC (two FLOPs). This excludes baseline
        G5, the original encoder/consumer, norm/GELU, transport and reductions.
        """
        if (len(padded_hw) != 2 or any(type(v) is not int or v <= 0 or v % 112 for v in padded_hw)
                or type(batch) is not int or batch <= 0):
            raise ValueError('Positive batch and original padded geometry required')
        hp,wp = padded_hw
        prefixes = dict(fine_stem=2,fine_down=4,cnn4_stem=4,dav6_stem=14,dav12_stem=14,
            semantic_mix=14,own_mix=4,match_mix=4,encoder4=4,down4=8,encoder8=8,down8=16,
            deep=16,up8=8,decoder8=8,up4=4,decoder4=4,fine_mix=2,fine_block=2,
            spatial_head=2,raw_own_head=2,raw_match_head=2)
        details = {}
        for name,module in self.named_modules():
            if isinstance(module,nn.Conv2d):
                stride = prefixes[name.split('.',1)[0]]
                locations = 2*batch*(hp//stride)*(wp//stride)
                details[name] = int(locations*module.out_channels*(module.in_channels//module.groups)
                                    *module.kernel_size[0]*module.kernel_size[1])
            elif isinstance(module,nn.Linear):
                details[name] = int(2*batch*module.in_features*module.out_features)
        total = sum(details.values())
        return dict(parameter_count=sum(v.numel() for v in self.parameters()),
            padded_hw=list(padded_hw),paired_batch=batch,conv_linear_MACs=total,
            conv_linear_FLOPs=2*total,per_layer_MACs=details,
            excluded='Encoder, baselineG5, repairedG5, norms/GELU, transport, reductions; no measured latency claim')

    def _validate(self,original,sigma,observations,prior,native_hw,padding):
        hp,wp = _geometry(native_hw,padding)
        if (not isinstance(original,Tensor) or original.ndim != 5 or original.shape[0] < 1
                or tuple(original.shape[1:]) != (2,64,hp//2,wp//2)):
            raise ValueError('F0 must be original own-grid [B,2,64,Hp/2,Wp/2]')
        batch = original.shape[0]
        if not isinstance(sigma,Tensor) or tuple(sigma.shape) not in ((64,),(batch,64)):
            raise ValueError('Original fixed sigma [64] or [B,64] required')
        if not isinstance(observations,Mapping) or set(observations) != set(self.observation_channels):
            raise ValueError('Only four declared original same-pass encoder observations accepted')
        tensors = [original,sigma]
        for key,channels in self.observation_channels.items():
            stride = 2 if key == 'cnn_pre_s2' else 4 if key == 'cnn_pre_s4' else 14
            value = observations[key]
            if not isinstance(value,Tensor) or tuple(value.shape) != (batch,2,channels,hp//stride,wp//stride):
                raise ValueError('Original tap shape mismatch: '+key)
            tensors.append(value)
        if (original.device.type not in ('cpu','cuda') or torch.is_autocast_enabled(original.device.type)
                or any(v.dtype != torch.float32 or v.device != original.device or v.requires_grad
                       or v.grad_fn is not None for v in tensors)
                or not all_tensors_finite(tensors) or not bool((sigma > 0).all())):
            raise ValueError('Detached finite same-device FP32 observations/sigma; autocast disabled')
        if (not isinstance(prior,MotionPrior) or prior.native_hw != tuple(native_hw)
                or prior.padding != tuple(padding) or prior.padded_hw != (hp,wp)):
            raise ValueError('Cached MotionPrior must bind original native/padded geometry')
        expected = dict(gather_xy_s4=(batch,hp//4,wp//4,2),gather_valid_s4=(batch,1,hp//4,wp//4),
            splat_indices=(batch,4,hp*wp//16),splat_weights=(batch,4,hp*wp//16),
            collision_mass_s4=(batch,1,hp//4,wp//4),evidence_s4=(batch,2,10,hp//4,wp//4),
            native_s2=(1,1,hp//2,wp//2),native_s4=(1,1,hp//4,wp//4),native_s16=(1,1,hp//16,wp//16))
        for key,shape in expected.items():
            value = getattr(prior,key)
            dtype = torch.long if key == 'splat_indices' else torch.bool if key in (
                'gather_valid_s4','native_s2','native_s4','native_s16') else torch.float32
            if (not isinstance(value,Tensor) or tuple(value.shape) != shape or value.dtype != dtype
                    or value.device != original.device or value.requires_grad or value.grad_fn is not None):
                raise ValueError('Cached prior shape/type/device mismatch: '+key)
        if any(v.dtype != torch.float32 or v.device != original.device for v in self.parameters()):
            raise ValueError('Repair parameters must be same-device FP32')
        return batch,hp,wp

    @staticmethod
    def _cnn_scaled(value):
        # Detached paired/channel RMS over the original full padded observation;
        # unlike per-pixel LN this retains within-channel spatial amplitude.
        with torch.no_grad():
            rms = value.square().mean((1,3,4),keepdim=True).sqrt()
            floor = .01*rms.flatten(1).median(1).values.reshape(-1,1,1,1,1)+1e-6
            scale = torch.maximum(rms,floor)
        return value/scale

    def forward(self,original: Tensor,sigma: Tensor,observations: Mapping[str,Tensor],*,
                motion_prior: MotionPrior,native_hw,padding,return_aux=False):
        batch,hp,wp = self._validate(original,sigma,observations,motion_prior,native_hw,padding)
        prior = motion_prior
        scale = sigma.reshape(1 if sigma.ndim == 1 else batch,1,64,1,1)
        raw = original/scale
        def flat(value):
            return value.reshape(2*batch,value.shape[2],*value.shape[-2:])
        fine = F.gelu(self.fine_stem(torch.cat((flat(raw),flat(self._cnn_scaled(observations['cnn_pre_s2']))),1)))
        cnn4 = F.gelu(self.cnn4_stem(flat(self._cnn_scaled(observations['cnn_pre_s4']))))
        semantics = self.semantic_mix(torch.cat((self.dav6_stem(flat(observations['dav2_block6'])),
                                                 self.dav12_stem(flat(observations['dav2_block12']))),1))
        # Patch14 centers are 14j+6.5; CNN/F0 centers are stride*j.
        semantics = resample_grid(semantics,14,6.5,4,0.,(hp//4,wp//4))
        sem_y = torch.arange(hp//4,device=raw.device)*4
        sem_x = torch.arange(wp//4,device=raw.device)*4
        semantic_support = ((sem_y >= 6.5)&(sem_y <= hp-7.5))[:,None] & ((sem_x >= 6.5)&(sem_x <= wp-7.5))[None]
        semantics = semantics*semantic_support[None,None]
        own = F.gelu(self.own_mix(torch.cat((self.fine_down(fine),cnn4,semantics),1)))
        own_pair = own.reshape(batch,2,-1,hp//4,wp//4)
        raw4 = raw[...,::2,::2]  # S2→S4 stride2 selection preserves physical origin0.
        # Combine raw and projected fields so each directional transfer uses one
        # batched gather/scatter, never per-token/candidate/chunk Python loops.
        transfer = torch.cat((own_pair,raw4),2)
        cross = torch.stack((gather_frame2_to_frame1(transfer[:,1],prior),
                             splat_frame1_to_frame2(transfer[:,0],prior)),1)
        cross_projected,cross_raw = cross[:,:,:own.shape[1]],cross[:,:,own.shape[1]:]
        available = torch.stack((prior.gather_valid_s4,prior.collision_mass_s4 > 0),1).to(raw.dtype)
        difference = (raw4-cross_raw)*available
        raw_match = torch.cat((difference,difference.abs()),2)
        encoded4 = self.encoder4(F.gelu(self.match_mix(torch.cat((own,flat(cross_projected),
                                                     flat(raw_match),flat(prior.evidence_s4)),1))))
        encoded8 = self.encoder8(self.down4(encoded4))
        deep = self.deep(self.down8(encoded8))
        decoded8 = self.decoder8(F.gelu(self.up8(torch.cat((encoded8,
            resample_grid(deep,16,0.,8,0.,(hp//8,wp//8))),1))))
        decoded4 = self.decoder4(F.gelu(self.up4(torch.cat((encoded4,
            resample_grid(decoded8,8,0.,4,0.,(hp//4,wp//4))),1))))
        detail = self.fine_block(F.gelu(self.fine_mix(torch.cat((fine,
            resample_grid(decoded4,4,0.,2,0.,(hp//2,wp//2))),1))))
        spatial_raw = self.spatial_head(detail)+self.raw_own_head(flat(raw))
        spatial_raw = spatial_raw+self.raw_match_head(resample_grid(flat(raw_match),4,0.,2,0.,(hp//2,wp//2)))
        spatial_raw = spatial_raw.reshape(batch,2,64,hp//2,wp//2)
        spatial_mean = spatial_raw.mean((-2,-1),keepdim=True)
        spatial = spatial_raw-spatial_mean
        native2 = prior.native_s2[:,None].to(raw.dtype)
        count2 = native2.sum((-2,-1)).clamp_min(1)
        raw_mean = (raw*native2).sum((-2,-1))/count2
        raw_rms = ((raw.square()*native2).sum((-2,-1))/count2).sqrt()
        native16 = prior.native_s16.to(deep.dtype)
        pooled = (deep*native16).sum((-2,-1))/native16.sum((-2,-1)).clamp_min(1)
        global_input = torch.cat((pooled,raw_mean.reshape(2*batch,64),raw_rms.reshape(2*batch,64)),1)
        global_delta = self.global_head(self.global_hidden(global_input)).reshape(batch,2,64,1,1)
        residual = global_delta+spatial
        repaired = _IdentityResidual.apply(original,scale*residual)
        if not all_tensors_finite((repaired,)):
            raise FloatingPointError('Nonfinite motion repair output')
        if not return_aux:
            return repaired
        return repaired,dict(normalized_residual=residual,global_normalized_residual=global_delta,
            spatial_normalized_residual=spatial,spatial_raw_mean=spatial_mean,
            aligned_raw_difference_s4=difference,matched_support_s4=available,
            forward_collision_mass_s4=prior.collision_mass_s4,
            frame2_context='Normalized forward splat; transported -f12 and forward info, not true backward solve')
