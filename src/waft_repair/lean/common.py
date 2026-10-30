"""Portable lean I/O helpers; importing creates no CUDA context or work."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd().resolve()
PYTHON = Path(sys.executable)


def _reject_constant(value):
    raise ValueError('Nonfinite JSON constant rejected: '+value)


def read(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream, parse_constant=_reject_constant)


def _fsync_parent(path):
    directory = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_output(path, producer, *, binary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=str(path.parent))
    try:
        mode = 'wb' if binary else 'w'
        kwargs = {} if binary else dict(encoding='utf-8', newline='\n')
        with os.fdopen(fd, mode, **kwargs) as stream:
            producer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return path


def write(path, obj):
    # Encode before creating/replacing any file: NaN or unserializable metadata
    # must leave the previous committed JSON untouched.
    encoded = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n'
    return _atomic_output(path, lambda stream: stream.write(encoded), binary=False)


def descriptor(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError('Regular file descriptor required: '+str(path))
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('File changed during SHA verification: '+str(path))
    return dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest())


def verify(desc):
    if (not isinstance(desc, dict) or not {'path', 'bytes', 'sha256'} <= set(desc)
            or not isinstance(desc['path'], str) or not Path(desc['path']).is_absolute()
            or type(desc['bytes']) is not int or desc['bytes'] < 0
            or not isinstance(desc['sha256'], str) or len(desc['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in desc['sha256'])):
        raise ValueError('Absolute path/bytes/lowercase SHA256 descriptor required')
    actual = descriptor(desc['path'])
    if any(actual[key] != desc[key] for key in ('bytes', 'sha256')):
        raise ValueError('Descriptor bytes/SHA mismatch: '+desc['path'])
    return Path(actual['path'])


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(Path(path)))
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load module: '+str(path))
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module  # dataclasses inspect their registered module.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def save_torch(path, obj):
    import torch
    def validate(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cpu':
                raise ValueError('save_torch requires already detached CPU state tensors')
            if value.requires_grad or value.grad_fn is not None:
                raise ValueError('save_torch refuses a live autograd tensor')
        elif isinstance(value, dict):
            for item in value.values(): validate(item)
        elif isinstance(value, (tuple, list)):
            for item in value: validate(item)
    validate(obj)
    return _atomic_output(path, lambda stream: torch.save(obj, stream), binary=True)


def load_torch(path):
    import torch
    return torch.load(Path(path), map_location='cpu', weights_only=True)


def load_site_adapter(config):
    """Require an explicit local resource adapter; no unprotected fallback."""
    path = config.get('resource_control')
    if not isinstance(path, str) or not path.strip():
        raise ValueError('An explicit resource_control site-adapter path is required')
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('resource_control must be a local Python module file')
    adapter = load_module('_lean_site_resources', path)
    required = ('admit', 'release_own_slot', 'wait_before_context', 'setup_cuda', 'check')
    if any(not callable(getattr(adapter, name, None)) for name in required):
        raise ValueError('Site adapter must provide all admission, CUDA, running and release gates')
    return adapter
