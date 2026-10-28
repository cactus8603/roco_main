"""Same-pass observations and original frozen WAFT five-step task graph.

The caller supplies a loaded official source-only WAFT. Nothing downloads or
loads weights on import. The inherited consumer preserves original flow and
coordinate detaches; the full-feedback teacher backend is deliberately unused.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import inspect
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from .consumer import FrozenWAFTFeaturePipelineV1
from .loss import LossSettings, sequence_mol_loss, supervision_mask


@dataclass(frozen=True)
class PreparedRepair:
    original: Tensor
    sigma: Tensor
    observations: Mapping[str, Tensor]
    native_hw: tuple[int, int]
    padding: tuple[int, int, int, int]


@torch.no_grad()
def original_sigma(original: Tensor, native_hw, padding) -> Tensor:
    """Detached original-only paired/native FP64 RMS, original lower median."""
    if original.ndim != 5 or original.shape[1:3] != (2, 64) or original.dtype != torch.float32:
        raise ValueError('Original FP32 B,2,64,Hf,Wf required')
    h, w = native_hw
    left, right, top, bottom = padding
    fh, fw = original.shape[-2:]
    if (2*fh, 2*fw) != (h+top+bottom, w+left+right):
        raise ValueError('Native geometry does not match stride2 features')
    y, x = torch.meshgrid(torch.arange(fh, device=original.device),
                          torch.arange(fw, device=original.device), indexing='ij')
    mask = (2*x >= left) & (2*x < left+w) & (2*y >= top) & (2*y < top+h)
    if not bool(mask.any()) or not bool(torch.isfinite(original).all()):
        raise ValueError('Nonempty native support and finite original required')
    result = []
    for pair in original:
        values = pair[:, :, mask].double()
        rms = values.square().mean(dim=(0, 2)).sqrt()
        result.append(rms.clamp_min(.01*rms.median()+1e-8).float())
    return torch.stack(result).detach()


def teacher_feature_loss(repaired: Tensor, teacher: Tensor, sigma: Tensor) -> Tensor:
    """Full padded paired normalized MSE; teacher and sigma are labels only."""
    if (repaired.ndim != 5 or repaired.shape[1:3] != (2, 64)
            or teacher.shape != repaired.shape or teacher.requires_grad
            or teacher.grad_fn is not None or sigma.requires_grad or sigma.grad_fn is not None):
        raise ValueError('Paired prediction and detached same-shape teacher/sigma required')
    if any(t.dtype != torch.float32 or t.device != repaired.device
           or not bool(torch.isfinite(t).all()) for t in (repaired, teacher, sigma)):
        raise ValueError('Teacher operands must be finite same-device FP32')
    b = repaired.shape[0]
    if sigma.shape == (64,):
        scale = sigma.view(1, 1, 64, 1, 1)
    elif sigma.shape == (b, 64):
        scale = sigma.view(b, 1, 64, 1, 1)
    else:
        raise ValueError('Sigma must have64 or B,64 elements')
    if not bool((scale > 0).all()):
        raise ValueError('Positive sigma required')
    loss = ((repaired-teacher)/scale).square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite distillation loss')
    return loss


def native_task_loss(output: dict, gt: Tensor, valid: Tensor) -> Tensor:
    """Five native MoL terms, gamma=.85, pooled pixel/component reduction.

At admitted finite values this is the official objective. Nonfinite outputs
are errors rather than silently excluded predictions. GT invalid positions
are indexed out before arithmetic by the reviewed native loss helper.
"""
    flows, infos = output.get('flow', ()), output.get('info', ())
    if len(flows) != 5 or len(infos) != 5 or gt.requires_grad or gt.grad_fn is not None:
        raise ValueError('Exactly five native flow/info outputs and detached GT required')
    if any(not bool(torch.isfinite(t).all()) for t in (*flows, *infos)):
        raise FloatingPointError('Nonfinite native flow/info')
    if not bool(supervision_mask(gt, valid, 40000.).any()):
        raise ValueError('Native task batch has no supervised pixels')
    # Reuse the original finite-input MoL expression, not a final-EPE surrogate.
    loss = sequence_mol_loss(output, gt, valid, LossSettings(gamma=.85))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite native NF/scalar loss')
    return loss


class NativeRepairSystem(FrozenWAFTFeaturePipelineV1):
    """One independent repair module; original complete WAFT stays frozen.

``prepare_pair`` only observes C2/C4 and normalized ViT6/12 from the original
prefix. ``forward_prepared`` also accepts separately cached observations,
which is useful for feature distillation without another encoder call.
"""

    SOURCE_ONLY_STATE = '994e1164487d20779ec14a72d828a8d20433d00ac553ecaeb5d09aab554227ee'

    def __init__(self, *args, expected_core_digest=None, dino_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        if dino_provider is not None:
            raise ValueError('Motion repair uses original WAFT observations; DINOv3 is unsupported')
        self.dino_provider = None
        self.validate()
        if self.strict_official:
            expected_settings = dict(algorithm='waft-a2', feature_encoder='dav2',
                                     iterative_module='vits', iters=5, scale=0,
                                     use_var=True, var_min=0, var_max=10)
            actual_args = getattr(self.core, 'args', None)
            if any(getattr(actual_args, key, None) != value for key, value in expected_settings.items()):
                raise ValueError('Original source-only WAFT architecture/loss settings differ')
        self._core_binding = self._measure_core_binding()
        expected = expected_core_digest
        if expected is None and self.strict_official:
            expected = self.SOURCE_ONLY_STATE
        if expected is not None and self._core_binding['state_digest'] != expected:
            raise ValueError('Loaded frozen WAFT state differs from the declared checkpoint')

    def observation_binding(self):
        """No additional encoder: observations come from the supplied WAFT."""
        return None

    def verify_observation_binding(self):
        self.validate()
        return None

    def _measure_core_binding(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.core.state_dict().items()):
            raw = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode()+b'\0')
            digest.update(str((tuple(value.shape), value.dtype)).encode()+b'\0')
            digest.update(memoryview(raw).cast('B'))
        paths = set()
        for module in self.core.modules():
            try:
                source = inspect.getsourcefile(type(module))
            except TypeError:
                source = None
            if source and Path(source).is_file():
                paths.add(Path(source).resolve())
        # Bind the helpers that will actually run, including the inherited
        # utils.utils fallback when callers omit one or more explicit helpers.
        # Callable identity also distinguishes alternatives in the same file.
        geometry_helpers = {}
        for name, function in zip(('padder', 'coords_grid', 'bilinear_sampler'), self._geometry()):
            try:
                source = inspect.getsourcefile(function)
            except TypeError:
                source = None
            resolved = str(Path(source).resolve()) if source and Path(source).is_file() else None
            geometry_helpers[name] = {'module': getattr(function, '__module__', None),
                                      'qualname': getattr(function, '__qualname__', None),
                                      'source': resolved}
            if resolved is not None:
                paths.add(Path(resolved))
        args = getattr(self.core, 'args', None)
        settings = {key: getattr(args, key) for key in ('algorithm', 'feature_encoder',
                    'iterative_module', 'iters', 'scale', 'use_var', 'var_min', 'var_max')
                    if hasattr(args, key)}
        return {'state_digest': digest.hexdigest(), 'torch_version': str(torch.__version__),
                'architecture_settings': settings,
                'geometry_helpers': geometry_helpers,
                'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}}

    def core_binding(self):
        return copy.deepcopy(self._core_binding)

    def verify_core_binding(self):
        """Checkpoint boundary check, intentionally not a per-forward rehash."""
        self.validate()
        if self._measure_core_binding() != self._core_binding:
            raise ValueError('Frozen WAFT state/source changed since construction')
        return self.core_binding()

    @torch.no_grad()
    def prepare_pair(self, image1: Tensor, image2: Tensor) -> PreparedRepair:
        # Frozen cached F0/taps use an FP32 observation contract even when a
        # caller enables autocast around the enclosing training/inference step.
        if not isinstance(image1, Tensor):
            raise ValueError('Observed RGB must be a tensor')
        with torch.autocast(device_type=image1.device.type, enabled=False):
            return self._prepare_pair_fp32(image1, image2)

    def _prepare_pair_fp32(self, image1: Tensor, image2: Tensor) -> PreparedRepair:
        if getattr(self, '_capturing_repair_taps', False):
            raise RuntimeError('Same-pass tap collection is not reentrant')
        encoder = self.core.encoder
        vit = getattr(encoder, 'encoder', None)
        if (vit is None or getattr(vit, 'patch_size', None) != 14
                or getattr(vit, 'embed_dim', None) != 384
                or not hasattr(vit, 'norm') or not hasattr(self.core.fnet, 'conv1')
                or not hasattr(self.core.fnet, 'conv2')):
            raise ValueError('Original DAV2-vits14 and predecoder CNN observations required')
        if tuple(encoder.intermediate_layer_idx.get('vits', ())) != (2, 5, 8, 11):
            raise ValueError('Original DAV2 observation order differs')
        captures = {'cnn_pre_s2': [], 'cnn_pre_s4': [], 'dav2_block6': [], 'dav2_block12': []}
        norm_calls = 0
        handles = []

        def cnn_hook(name):
            def hook(_module, _args, value):
                captures[name].append(value.detach())
            return hook

        def token_hook(_module, _args, value):
            nonlocal norm_calls
            if norm_calls % 4 in (1, 3):
                captures['dav2_block6' if norm_calls % 4 == 1 else 'dav2_block12'].append(value.detach())
            norm_calls += 1

        self._capturing_repair_taps = True
        try:
            handles.append(self.core.fnet.conv1.register_forward_hook(cnn_hook('cnn_pre_s2')))
            handles.append(self.core.fnet.conv2.register_forward_hook(cnn_hook('cnn_pre_s4')))
            handles.append(vit.norm.register_forward_hook(token_hook))
            prepared = self.prepare(image1, image2)
        finally:
            for handle in handles:
                handle.remove()
            self._capturing_repair_taps = False
        if norm_calls != 8 or any(len(v) != 2 for v in captures.values()):
            raise ValueError('Expected one exact two-frame encoder prefix')
        hp, wp = prepared.normalized_rgb.shape[-2:]
        ph, pw = hp//14, wp//14
        registers = vit.num_register_tokens
        observations = {}
        for key, values in captures.items():
            if key.startswith('dav2'):
                if any(v.shape != (image1.shape[0], 1+registers+ph*pw, 384) for v in values):
                    raise ValueError('Original normalized patch token geometry differs')
                values = [v[:, 1+registers:].transpose(1, 2).reshape(-1, 384, ph, pw) for v in values]
            observations[key] = torch.stack(values, 1).detach()
        return PreparedRepair(prepared.native_features,
                              original_sigma(prepared.native_features, prepared.native_hw, prepared.padding),
                              observations, prepared.native_hw, prepared.padding)

    def repair_features(self, prepared: PreparedRepair, *, return_aux=False):
        self.validate()
        return self.restorer(prepared.original, prepared.sigma, prepared.observations,
                             native_hw=prepared.native_hw, padding=prepared.padding,
                             return_aux=return_aux)

    def forward_prepared(self, prepared: PreparedRepair, *, return_aux=False):
        with torch.autocast(device_type=prepared.original.device.type, enabled=False):
            value = self.repair_features(prepared, return_aux=return_aux)
            repaired, aux = value if return_aux else (value, {})
            output = self.solve_features(repaired, native_hw=prepared.native_hw, padding=prepared.padding)
        output.update(features=repaired, base_features=prepared.original,
                      padding=prepared.padding, native_hw=prepared.native_hw, repair_aux=aux)
        return output

    def forward(self, image1: Tensor, image2: Tensor, *, return_aux=False):
        return self.forward_prepared(self.prepare_pair(image1, image2), return_aux=return_aux)

    @torch.no_grad()
    def predict(self, image1: Tensor, image2: Tensor, *, return_aux=False):
        if self.training:
            raise ValueError('Call eval before prediction')
        return self(image1, image2, return_aux=return_aux)


# Motion-specific RGB integration and the optional legacy pilot loader.
from collections.abc import Mapping
from dataclasses import asdict
import os

from .finite import all_tensors_finite
from .model import MotionFeatureRepair, MotionPrior, prepare_motion_prior

@dataclass(frozen=True)
class PreparedMotion(PreparedRepair):
    """GT-free original observations and the original baseline's motion prior.

    ``baseline_output`` retains all five original flow/info readouts for callers
    that need identity controls. Its final tensors are ``baseline_flow/info``.
    It is never passed to the trainable module as labels or a hidden state.
    """
    baseline_flow: Tensor
    baseline_info: Tensor
    motion_prior: MotionPrior
    baseline_output: Mapping[str, object]


class MotionNativeRepairSystem(NativeRepairSystem):
    """One RGB prefix, one baseline G5, R, then one repaired-feature G5.

    ``prepare_pair`` is detached and reusable. ``forward_prepared`` performs
    no encoder or baseline solve, and remains differentiable through R and the
    unchanged frozen consumer. ``predict`` is the inherited no-grad RGB entry.
    """

    def __init__(self, core, restorer: MotionFeatureRepair, **kwargs):
        if not isinstance(restorer, MotionFeatureRepair):
            raise TypeError('A separately constructed MotionFeatureRepair is required')
        if kwargs.get('dino_provider') is not None:
            raise ValueError('Motion repair uses the original four WAFT observations only')
        super().__init__(core, restorer, **kwargs)

    @torch.no_grad()
    def prepare_pair(self, image1: Tensor, image2: Tensor) -> PreparedMotion:
        if not isinstance(image1, Tensor):
            raise ValueError('Observed RGB must be a tensor')
        with torch.autocast(device_type=image1.device.type, enabled=False):
            observed = super().prepare_pair(image1, image2)
            baseline = self.solve_features(observed.original,
                native_hw=observed.native_hw, padding=observed.padding)
            if len(baseline.get('flow', ())) != 5 or len(baseline.get('info', ())) != 5:
                raise ValueError('Exactly five original baseline flow/info readouts required')
            flow, info = baseline['flow'][-1].detach(), baseline['info'][-1].detach()
            # The pilot cached the native baseline on CPU before deriving prior.
            # Do not replace this with a GPU scatter reduction during deployment.
            with torch.autocast(device_type='cpu', enabled=False):
                prior = prepare_motion_prior(flow.to(device='cpu').contiguous(),
                    info.to(device='cpu').contiguous(), native_hw=observed.native_hw,
                    padding=observed.padding)
            prior = prior.to(observed.original.device)
            self.validate()
            return PreparedMotion(observed.original, observed.sigma,
                observed.observations, observed.native_hw, observed.padding,
                flow, info, prior, baseline)

    def repair_features(self, prepared: PreparedMotion, *, return_aux=False):
        self.validate()
        if not isinstance(prepared, PreparedMotion):
            raise TypeError('PreparedMotion with the canonical CPU-derived prior is required')
        batch = prepared.original.shape[0]
        for value, channels in ((prepared.baseline_flow, 2), (prepared.baseline_info, 4)):
            if (not isinstance(value, Tensor)
                    or tuple(value.shape) != (batch, channels, *prepared.native_hw)
                    or value.dtype != torch.float32 or value.device != prepared.original.device
                    or value.requires_grad or value.grad_fn is not None):
                raise ValueError('Detached same-device native baseline flow/info required')
        if not all_tensors_finite((prepared.baseline_flow, prepared.baseline_info)):
            raise ValueError('Finite baseline flow/info required')
        return self.restorer(prepared.original, prepared.sigma, prepared.observations,
            motion_prior=prepared.motion_prior, native_hw=prepared.native_hw,
            padding=prepared.padding, return_aux=return_aux)


_CHECKPOINT_SCHEMA = 'waft-repair-redesign-training/v1'
_CHECKPOINT_KEYS = {'schema', 'config', 'model_config', 'repair', 'optimizer',
    'scheduler', 'completed_updates', 'binding', 'cursor',
    'verified_implementation_binding', 'verified_core_binding',
    'verified_observation_binding', 'torch_rng', 'python_rng', 'numpy_rng', 'device_rng'}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load_legacy_motion_repair_checkpoint(system: MotionNativeRepairSystem, checkpoint, *,
        expected_sha256=None, expected_completed_updates=128, expected_binding=None):
    """Install R from a legacy native128 pilot checkpoint; load no core/RNG.

    Current full-training checkpoints use the loader in training.py; this
    helper is provided only for explicitly selected legacy pilot artifacts.

    Paths use safe CPU ``weights_only`` loading. A mapping must have an equally
    trusted loading origin and cannot be combined with a file SHA assertion.
    This is a deployment loader, not optimizer/scheduler resume validation.
    Schema, architecture, completed-update cursor, core digest and every repair
    tensor are checked before mutation. Optional expected binding pins the
    original training provenance without requiring inference images to be TRAIN.
    """
    if not isinstance(system, MotionNativeRepairSystem):
        raise TypeError('MotionNativeRepairSystem required')
    _require(type(expected_completed_updates) is int and expected_completed_updates > 0,
             'Explicit positive completed-update count required')
    if expected_sha256 is not None:
        _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
                 and all(c in '0123456789abcdef' for c in expected_sha256),
                 'Expected SHA256 must be lowercase hexadecimal')
    source, digest = None, None
    if isinstance(checkpoint, (str, os.PathLike)):
        path = Path(checkpoint).resolve()
        digest = _file_sha256(path)
        _require(expected_sha256 is None or digest == expected_sha256,
                 'Motion checkpoint file SHA256 differs')
        state = torch.load(path, map_location='cpu', weights_only=True)
        source = str(path)
    elif isinstance(checkpoint, Mapping):
        _require(expected_sha256 is None, 'A file path is required for expected SHA256')
        state = checkpoint
    else:
        raise TypeError('Checkpoint path or safely loaded mapping required')
    system.validate()
    _require(isinstance(state, Mapping) and set(state) == _CHECKPOINT_KEYS
             and state['schema'] == _CHECKPOINT_SCHEMA, 'Complete redesign-v1 checkpoint required')
    _require(state['model_config'] == asdict(system.restorer.config),
             'Checkpoint motion architecture/config differs')
    n = state['completed_updates']
    _require(type(n) is int and n == expected_completed_updates,
             'Checkpoint completed-update count differs')
    config = state['config']
    _require(isinstance(config, Mapping) and config.get('stage') == 'native'
             and type(config.get('updates')) is int and config['updates'] >= n
             and config.get('auxiliary_distillation_weight') == 0.,
             'Original native-only training provenance required')
    for field in ('binding', 'verified_implementation_binding',
                  'verified_core_binding', 'verified_observation_binding'):
        _require(isinstance(state[field], Mapping), 'Checkpoint binding missing: ' + field)
    if expected_binding is not None:
        _require(state['binding'] == expected_binding, 'Checkpoint training binding differs')
    verified_core = system.verify_core_binding()
    _require(state['verified_core_binding'].get('state_digest') == verified_core['state_digest'],
             'Checkpoint frozen WAFT digest differs from supplied core')
    cursor = state['cursor']
    _require(isinstance(cursor, Mapping), 'Checkpoint input cursor required')
    order = cursor.get('fixed_order')
    _require(isinstance(order, list) and bool(order)
             and all(isinstance(v, str) and v for v in order) and len(set(order)) == len(order)
             and type(cursor.get('next_case_index')) is int
             and cursor['next_case_index'] == n % len(order)
             and type(cursor.get('global_optimizer_updates')) is int
             and cursor['global_optimizer_updates'] == n,
             'Checkpoint input/update cursor differs')
    saved, current = state['repair'], system.restorer.state_dict()
    _require(isinstance(saved, Mapping) and saved.keys() == current.keys(),
             'Exact repair parameter names required')
    for name, value in saved.items():
        ref = current[name]
        _require(isinstance(value, Tensor) and value.device.type == 'cpu'
                 and value.layout == torch.strided and value.dtype == ref.dtype
                 and value.shape == ref.shape and not value.requires_grad and value.grad_fn is None,
                 'Checkpoint repair tensor shape/dtype/device differs: ' + name)
    _require(all_tensors_finite(saved.values()), 'Nonfinite checkpoint repair tensor')
    # Only R is copied. Unexpected hooks/eval errors must not leave partial R.
    backup = {name: value.detach().clone() for name, value in current.items()}
    modes = [(module, module.training) for module in system.modules()]
    try:
        system.restorer.load_state_dict(saved, strict=True)
        system.eval()
        system.validate()
    except BaseException:
        with torch.no_grad():
            for name, value in system.restorer.state_dict().items():
                value.copy_(backup[name])
        for module, mode in modes:
            module.training = mode
        raise
    return dict(schema='waft-motion-repair-inference-load/v1', checkpoint_path=source,
        checkpoint_sha256=digest, completed_updates=n,
        model_config=copy.deepcopy(state['model_config']), binding=copy.deepcopy(state['binding']),
        loaded_repair_only=True, restored_core=False, restored_optimizer=False, restored_rng=False,
        motion_prior_contract='Original native fifth flow/info -> contiguous FP32 CPU prepare_motion_prior -> feature device')
