"""Portable, fail-closed loader for the pinned SEA-RAFT Spring-M model."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch


SEA_RAFT_VENDOR_COMMIT = "9137517ba24e628442aec097d3afe71d03503b75"
SEA_RAFT_CHECKPOINT_SHA256 = (
    "cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102"
)
SEA_RAFT_CHECKPOINT_BYTES = 78_778_760
SEA_RAFT_CRITICAL_SOURCE_SHA256 = {
    "config/eval/spring-M.json": (
        "32e1413055eecf506c92fe6cdcffcd5179694e22e66f558fa93a766d1a95d1d8"
    ),
    "core/corr.py": "e89bc770f1e6712447d90b8a8c75fd136c98816726d4338972f8ea5d7d6d24fb",
    "core/extractor.py": "684cfe479ef6cbe5796397749939676c79626f0c3fcbc033c8fb5cab9475bc7a",
    "core/layer.py": "580f4cced2b5f3551667b4b4b24b8ff7cd024ef4e3f8723bf3c5fbb7baf74563",
    "core/raft.py": "4bb02c831f04b16a27fd016c8bb4ecfcf5834ea88f40e84fac9d2555864d5979",
    "core/update.py": "978dc580e4ebd394b29894f5d5460153a2fe5a7d4c50e5e83823330298133f69",
    "core/utils/utils.py": (
        "04d88de0f161db56f856776e05d4337cecaadabdc1decc104b22c6ea642f11ae"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> Mapping[str, object]:
    def reject_duplicates(
        pairs: Sequence[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return value


def verify_pinned_sea_raft_identity(
    *,
    vendor_root: Path,
    config_path: Path,
    checkpoint: Path,
) -> dict[str, object]:
    """Verify the exact clean source, config, and weight identities."""

    vendor_root = vendor_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    expected_config_path = (vendor_root / "config/eval/spring-M.json").resolve()
    if not vendor_root.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("SEA-RAFT vendor source or checkpoint is missing")
    if config_path != expected_config_path or not config_path.is_file():
        raise ValueError("SEA-RAFT config must be the pinned vendor spring-M.json")
    commit = subprocess.run(
        ["git", "-C", str(vendor_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != SEA_RAFT_VENDOR_COMMIT:
        raise RuntimeError(f"SEA-RAFT commit mismatch: {commit}")
    dirty = subprocess.run(
        [
            "git", "-C", str(vendor_root), "status", "--porcelain",
            "--untracked-files=all",
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    if dirty:
        raise RuntimeError("SEA-RAFT vendor checkout is not clean")

    source_hashes: dict[str, str] = {}
    for relative, expected in SEA_RAFT_CRITICAL_SOURCE_SHA256.items():
        source = vendor_root / relative
        actual = sha256_file(source)
        if actual != expected:
            raise RuntimeError(f"SEA-RAFT source hash mismatch: {relative}")
        source_hashes[relative] = actual
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != SEA_RAFT_CHECKPOINT_SHA256:
        raise RuntimeError("SEA-RAFT Spring-M checkpoint hash mismatch")
    if checkpoint.stat().st_size != SEA_RAFT_CHECKPOINT_BYTES:
        raise RuntimeError("SEA-RAFT Spring-M checkpoint size mismatch")

    config = _strict_json(config_path)
    expected_config = {
        "name": "spring-M",
        "dataset": "spring",
        "pretrain": "resnet34",
        "initial_dim": 64,
        "block_dims": [64, 128, 256],
        "dim": 128,
        "radius": 4,
        "num_blocks": 2,
        "iters": 4,
        "scale": -1,
        "use_var": True,
        "var_min": 0,
        "var_max": 10,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"unexpected SEA-RAFT Spring-M config {key}={config.get(key)!r}"
            )
    return {
        "vendor_root": str(vendor_root),
        "vendor_commit": commit,
        "vendor_clean": True,
        "critical_source_sha256": source_hashes,
        "config_path": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "official_eval_iters": int(config["iters"]),
        "official_prediction_count": int(config["iters"]) + 1,
    }


def _import_official(vendor_root: Path) -> tuple[Any, Any, Any]:
    core = (vendor_root / "core").resolve()
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    raft = importlib.import_module("raft")
    corr = importlib.import_module("corr")
    utils = importlib.import_module("utils.utils")
    for module in (raft, corr, utils):
        source = Path(module.__file__).resolve()
        if core not in source.parents:
            raise RuntimeError(f"flat-module collision outside SEA-RAFT: {source}")
    return raft, corr, utils


def load_pinned_sea_raft_official_model(
    *,
    vendor_root: Path,
    config_path: Path,
    checkpoint: Path,
    device: str,
) -> tuple[Any, Any, Any, dict[str, object]]:
    """Load the pinned official model, resolving only declared tensor aliases."""

    identity = verify_pinned_sea_raft_identity(
        vendor_root=vendor_root,
        config_path=config_path,
        checkpoint=checkpoint,
    )
    raft_module, corr_module, utils_module = _import_official(vendor_root)
    config = _strict_json(config_path)
    model = raft_module.RAFT(argparse.Namespace(**config))
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "SEA-RAFT training requires the 'safetensors' package; "
            "install this repository with the training extra"
        ) from error

    state = load_file(str(checkpoint), device="cpu")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as stream:
        aliases = stream.metadata() or {}
    incompatible = model.load_state_dict(state, strict=False)
    unresolved = [
        key for key in incompatible.missing_keys
        if not (key in aliases and aliases[key] in state)
    ]
    if unresolved or incompatible.unexpected_keys:
        raise RuntimeError(
            "SEA-RAFT checkpoint/model mismatch: "
            f"unresolved={sorted(unresolved)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    model = model.to(torch.device(device)).eval()
    report = {
        **identity,
        "torch_home": os.environ.get("TORCH_HOME"),
        "checkpoint_tensor_count": len(state),
        "model_state_key_count": len(model.state_dict()),
        "alias_resolved_missing_keys": sorted(incompatible.missing_keys),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, corr_module.CorrBlock, utils_module, report


__all__ = [
    "SEA_RAFT_CHECKPOINT_BYTES",
    "SEA_RAFT_CHECKPOINT_SHA256",
    "SEA_RAFT_CRITICAL_SOURCE_SHA256",
    "SEA_RAFT_VENDOR_COMMIT",
    "load_pinned_sea_raft_official_model",
    "sha256_file",
    "verify_pinned_sea_raft_identity",
]
