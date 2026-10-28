"""Small atomic files and explicit safe tensor loading for the native workflow."""
from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def descriptor(path, *, relative_to=None):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path.relative_to(relative_to)) if relative_to else str(path),
                bytes=path.stat().st_size,sha256=sha256(path))


def check_descriptor(item, root):
    require(isinstance(item,dict) and set(item)=={'path','bytes','sha256'},'Complete file descriptor required')
    path=(Path(root)/item['path']).resolve(strict=True)
    require(path.stat().st_size==item['bytes'] and sha256(path)==item['sha256'],'File bytes changed: '+str(path))
    return path


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().to('cpu',copy=True).contiguous()
    if dataclasses.is_dataclass(value):return cpu_tree(dataclasses.asdict(value))
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu_tree(v) for v in value)
    return value


def _atomic(path, writer):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            writer(stream);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_json(path,value):
    payload=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    _atomic(path,lambda stream:stream.write(payload))


def atomic_torch(path,value):
    _atomic(path,lambda stream:torch.save(cpu_tree(value),stream))


def atomic_text(path,value):
    payload=value.encode('utf-8')
    _atomic(path,lambda stream:stream.write(payload))


def load_tensor(path, *, expected_sha256=None):
    path=Path(path).resolve(strict=True)
    if expected_sha256 is not None:require(sha256(path)==expected_sha256,'Checkpoint/file SHA256 differs')
    return torch.load(path,map_location='cpu',weights_only=True,mmap=True)


@contextmanager
def output_lock(directory):
    """Prevent two foreground trainers from mutating the same output directory."""
    import fcntl
    with (Path(directory)/'run.lock').open('a+b') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError('Another trainer owns this output directory') from error
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)
