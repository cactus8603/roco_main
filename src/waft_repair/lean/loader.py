"""Strict offline CPU loader with the original lean initializer RNG behavior."""
import argparse
import importlib
import importlib.metadata
from pathlib import Path
import subprocess

import torch

from ..backend import (CHECKPOINT_PROFILES, MODEL_CONFIG, SOURCE_COMMIT,
                       _official_modules, _offline_constructor, _state_digest)
from .common import descriptor


def load_model(source, item):
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only official model loading required')
    source = Path(source).resolve(strict=True)
    head = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(source),'status','--porcelain',
                                    '--untracked-files=no'], text=True).strip()
    if head != SOURCE_COMMIT or dirty:
        raise ValueError('Unmodified pinned official WAFT source required')
    checkpoint = descriptor(item['checkpoint'])
    digest = checkpoint['sha256']
    if digest != item['sha256'] or digest not in CHECKPOINT_PROFILES:
        raise ValueError('Known complete official checkpoint SHA256 required')
    if importlib.metadata.version('timm') != '1.0.11':
        raise ValueError('Reviewed timm==1.0.11 required')
    module, helpers = _official_modules(source)
    # No RNG fork: the reviewed operation consumes CPU initializer RNG before
    # overwriting every core tensor. Keep that behavior in this new entry.
    with torch.device('cpu'), _offline_constructor(module, importlib.import_module('timm')) as changes:
        core = module.WAFTv2(argparse.Namespace(**MODEL_CONFIG))
    if changes != dict(timm=1, depth=1, network_attempts=0):
        raise ValueError('Unexpected separate initializer or network attempt')
    state = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
    if isinstance(state, dict) and set(state) == {'state_dict'}:
        state = state['state_dict']
    if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key,value in state.items()):
        raise ValueError('Complete tensor-only WAFT state required')
    if all(key.startswith('module.') for key in state):
        state = {key.removeprefix('module.'): value for key,value in state.items()}
    expected = core.state_dict()
    if len(state) != 508 or state.keys() != expected.keys():
        raise ValueError('All 508 official tensors required')
    for key,value in state.items():
        if (value.device.type != 'cpu' or value.dtype != expected[key].dtype
                or value.shape != expected[key].shape or not bool(torch.isfinite(value).all())):
            raise ValueError('Official tensor shape/dtype/finite value differs: '+key)
    state_digest = _state_digest(state)
    if state_digest != CHECKPOINT_PROFILES[digest]['state_sha256']:
        raise ValueError('Full official state digest differs')
    core.load_state_dict(state, strict=True)
    if _state_digest(core.state_dict()) != state_digest:
        raise ValueError('Loaded complete official tensor bytes differ')
    core.eval().requires_grad_(False)
    core._lean_geometry = dict(padder_factory=helpers.Padder, coords_grid_fn=helpers.coords_grid,
                               bilinear_sampler_fn=helpers.bilinear_sampler)
    return core, dict(all_saved_tensor_count=508, all_loaded_tensor_bytes_digest=state_digest,
                      strict_all_tensor_bytes_equal=True, forward_calls=0,
                      CUDA_initialized=False, checkpoint=checkpoint)
