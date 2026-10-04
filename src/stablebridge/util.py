"""Artifact IO and immutable run manifests; no experiment runs on import."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import platform
from importlib.metadata import version, PackageNotFoundError

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def event(directory, kind, **fields):
    entry = {'unix_time': time.time(), 'event': kind, **fields}
    path = Path(directory) / 'logs/events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(jsonable(entry), ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(jsonable(entry), ensure_ascii=False, allow_nan=False), flush=True)


def initialize_run(directory, manifest):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    code = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'src/stablebridge').glob('*.py'))}
    packages={}
    for name in ('torch','numpy','scipy','opencv-python','h5py','Pillow'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]='not_in_package_metadata'
    effective = {**manifest, 'source_hashes': code,
                 'environment':{'python':platform.python_version(),'packages':packages}}
    path = directory / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != jsonable(effective):
            raise ValueError('Run configuration or source changed; create a new run directory.')
    else:
        save_json(path, effective)
        for name in ['source', 'logs', 'checkpoints', 'predictions', 'decisions', 'metrics']:
            (directory / name).mkdir(exist_ok=True)
        for relative in code:
            destination = directory / 'source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
    return directory
