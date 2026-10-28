"""Portable inference weights for the native, motion-aligned 3M repair.

The original native trainer's full checkpoints can be imported for inference.
Their optimizer and RNG are never installed. Exact continuation is a separate
operation handled by ``training.NativeTrainer`` with unchanged bindings.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path

import torch

from ..backend import SOURCE_COMMIT
from ..targets import CHECKPOINT_SHA256
from .io import atomic_torch, cpu_tree, load_tensor, require, sha256
from .model import MotionFeatureRepair, MotionRepairConfig
from .native import NativeRepairSystem
from .training import SCHEMA as TRAINING_SCHEMA, STATE_KEYS, TrainConfig, expected_cursor, learning_rate


SCHEMA = 'waft-native-repair-model/v1'
KEYS = {'schema', 'model_config', 'repair', 'repair_state_digest', 'core_state_digest',
        'waft_checkpoint_sha256', 'waft_source_commit', 'completed_updates', 'metadata'}
REPAIR_TENSORS = 161
REPAIR_PARAMETERS = 3000576


def _metadata(value):
    """Only finite plain JSON values are portable metadata."""
    value = {} if value is None else value
    require(isinstance(value, dict), 'Checkpoint metadata must be a JSON object')
    def check(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            require(math.isfinite(item), 'Metadata floats must be finite')
            return
        if isinstance(item, dict):
            require(all(type(key) is str for key in item), 'Metadata keys must be strings')
            for nested in item.values(): check(nested)
            return
        if type(item) is list:
            for nested in item: check(nested)
            return
        raise ValueError('Checkpoint metadata must contain only plain JSON values')
    check(value)
    return json.loads(json.dumps(value, allow_nan=False))


def repair_state_digest(state):
    """Hash names, shapes, dtypes and every byte, including signed zeros."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b'\0')
        digest.update(str((tuple(value.shape), value.dtype)).encode() + b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _config(value):
    require(isinstance(value, dict) and set(value) == {field.name for field in fields(MotionRepairConfig)},
            'Complete motion-aligned architecture configuration required')
    config = MotionRepairConfig(**value)
    require(config.profile == 'motion_aligned_v1', 'Only the production 3M motion-aligned repair is supported')
    return config


def _weights(state, model):
    reference = model.state_dict()
    require(isinstance(state, dict) and state.keys() == reference.keys()
            and len(state) == REPAIR_TENSORS, 'Exact original 161-tensor repair state required')
    require(sum(parameter.numel() for parameter in model.parameters()) == REPAIR_PARAMETERS,
            'Production motion-aligned parameter count differs')
    for name, ref in reference.items():
        value = state[name]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and value.dtype == ref.dtype == torch.float32 and value.shape == ref.shape
                and value.layout == torch.strided and not value.requires_grad
                and bool(torch.isfinite(value).all()), 'Invalid saved repair tensor: ' + name)


def export_model_checkpoint(model, completed_updates, metadata=None):
    """Create CPU-owned inference weights; this contains no optimizer or RNG."""
    require(isinstance(model, MotionFeatureRepair), 'MotionFeatureRepair model required')
    require(type(completed_updates) is int and completed_updates >= 0, 'Nonnegative completed update count required')
    config = _config(asdict(model.config))
    state = cpu_tree(model.state_dict())
    _weights(state, MotionFeatureRepair(config))
    return dict(schema=SCHEMA, model_config=asdict(config), repair=state,
                repair_state_digest=repair_state_digest(state),
                core_state_digest=NativeRepairSystem.SOURCE_ONLY_STATE,
                waft_checkpoint_sha256=CHECKPOINT_SHA256, waft_source_commit=SOURCE_COMMIT,
                completed_updates=completed_updates, metadata=_metadata(metadata))


def _native_metadata(state):
    require(set(state) == STATE_KEYS, 'Complete original native training checkpoint required')
    value = state['config']
    require(isinstance(value, dict) and set(value) == {field.name for field in fields(TrainConfig)},
            'Complete original native training configuration required')
    config = TrainConfig(**value)
    require(value == asdict(config) and config.stage == 'native'
            and config.auxiliary_distillation_weight == 0., 'Native five-step training with no teacher auxiliary required')
    for key in ('peak_lr', 'weight_decay', 'epsilon', 'clip_norm', 'minimum_lr_fraction',
                'auxiliary_distillation_weight'):
        require(type(value[key]) in (float, int), 'Explicit numeric native optimizer settings required')
    n = state['completed_updates']
    require(type(n) is int and 0 <= n <= config.updates, 'Bounded native training update count required')
    core = state['verified_core_binding']
    require(isinstance(core, dict) and core.get('state_digest') == NativeRepairSystem.SOURCE_ONLY_STATE,
            'Original complete frozen source-only WAFT state is required')
    binding = state['binding']; cursor = state['cursor']
    require(isinstance(binding, dict) and {'fixed_cases', 'epochs', 'shuffle_seed'} <= binding.keys()
            and type(binding['epochs']) is int and binding['epochs'] > 0
            and isinstance(binding['fixed_cases'], list) and len(binding['fixed_cases']) > 0
            and config.updates == len(binding['fixed_cases']) * binding['epochs'],
            'Complete native fixed-population epoch binding required')
    require(isinstance(cursor, dict) and cursor == expected_cursor(binding, n, cursor.get('label')),
            'Original native next-input cursor differs from its update count')
    require(state['scheduler'] == dict(schema='native-linear-warmup-cosine/v1', last_epoch=n,
                                      next_learning_rate=learning_rate(config, n + 1)),
            'Original native schedule differs from its update count')
    return n


def load_model_checkpoint(path, device='cpu', expected_sha256=None):
    """Load strictly validated native repair weights into a new evaluation model.

    ``weights_only=True`` is mandatory. A full native training checkpoint is
    accepted only as a weights source: its optimizer, cursor and RNG are never
    restored to the caller. Model validation and loading happen on CPU before
    the explicitly selected destination device is used.
    """
    selected = torch.device(device)
    require(selected.type in ('cpu', 'cuda'), 'Select an explicit CPU or CUDA device')
    path = Path(path).resolve(strict=True)
    source_sha = sha256(path)
    if expected_sha256 is not None:
        require(source_sha == expected_sha256, 'Checkpoint/file SHA256 differs')
    state = load_tensor(path, expected_sha256=source_sha)
    require(sha256(path) == source_sha, 'Checkpoint changed during safe loading')
    require(isinstance(state, dict), 'Native repair checkpoint must be a mapping')
    schema = state.get('schema')
    if schema == SCHEMA:
        require(set(state) == KEYS, 'Complete portable native repair checkpoint required')
        require(state['core_state_digest'] == NativeRepairSystem.SOURCE_ONLY_STATE
                and state['waft_checkpoint_sha256'] == CHECKPOINT_SHA256
                and state['waft_source_commit'] == SOURCE_COMMIT, 'Pinned frozen WAFT identity differs')
        n = state['completed_updates']
        require(type(n) is int and n >= 0, 'Nonnegative completed update count required')
        metadata = _metadata(state['metadata'])
    elif schema == TRAINING_SCHEMA:
        n = _native_metadata(state)
        metadata = {'source_format': 'original-native-training',
                    'import_method': 'repair-weights-only; optimizer and RNG are not restored'}
    else:
        raise ValueError('Unsupported checkpoint: use native motion-aligned repair weights, not feature-MSE or trace models')
    config = _config(state['model_config'])
    model = MotionFeatureRepair(config)
    _weights(state['repair'], model)
    digest = repair_state_digest(state['repair'])
    if schema == SCHEMA:
        require(state['repair_state_digest'] == digest, 'Saved repair tensor bytes differ from their digest')
    model.load_state_dict(state['repair'], strict=True)
    require(repair_state_digest(model.state_dict()) == digest, 'Loaded repair tensor bytes differ')
    if selected.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA was selected but is unavailable')
    model.to(selected).eval()
    receipt = dict(schema=schema, checkpoint_sha256=source_sha, repair_state_digest=digest,
                   model_config=asdict(config), core_state_digest=NativeRepairSystem.SOURCE_ONLY_STATE,
                   waft_checkpoint_sha256=CHECKPOINT_SHA256, waft_source_commit=SOURCE_COMMIT,
                   completed_updates=n, repair_tensor_count=REPAIR_TENSORS,
                   parameter_count=REPAIR_PARAMETERS, device=str(selected),
                   optimizer_loaded=False, rng_restored=False, metadata=metadata)
    return model, receipt


def import_checkpoint(checkpoint, output, expected_sha256=None):
    """Convert validated native weights to a portable inference-only checkpoint.

    The conversion records source bytes and format, rather than copying the
    operational run's machine paths, source bindings, optimizer or RNG.
    """
    source = Path(checkpoint).resolve(strict=True)
    destination = Path(output).resolve()
    require(source != destination and not destination.exists(), 'Import requires a new output checkpoint path')
    model, receipt = load_model_checkpoint(source, expected_sha256=expected_sha256)
    metadata = dict(origin_checkpoint_sha256=receipt['checkpoint_sha256'], origin_schema=receipt['schema'],
                    origin_repair_tensor_count=receipt['repair_tensor_count'],
                    import_method='validated-native-repair-weights-only')
    packet = export_model_checkpoint(model, receipt['completed_updates'], metadata=metadata)
    atomic_torch(destination, packet)
    return dict(schema=SCHEMA, checkpoint_sha256=sha256(destination),
                repair_state_digest=packet['repair_state_digest'], completed_updates=receipt['completed_updates'],
                repair_tensor_count=REPAIR_TENSORS, origin_checkpoint_sha256=receipt['checkpoint_sha256'],
                optimizer_imported=False, rng_imported=False)
