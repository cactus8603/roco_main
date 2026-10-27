"""Safe, explicit repair checkpoints; no pretrained model downloads."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from .data import require, valid_sha
from .model import FeatureRepair

SCHEMA = "waft-repair-checkpoint"


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str((tuple(value.shape), value.dtype)).encode()
                      + b"\0" + value.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path: str | Path, value: dict, *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with path.open("xb") as stream:
            torch.save(cpu_tree(value), stream)
        return
    temporary = path.with_name(path.name + ".writing-" + str(os.getpid()))
    try:
        save_checkpoint(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint(path: str | Path) -> dict:
    value = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    require(isinstance(value, dict) and value.get("schema") == SCHEMA, "Unsupported repair checkpoint")
    require(valid_sha(value.get("waft_checkpoint_sha256")), "Checkpoint lacks its frozen WAFT identity")
    require(value.get("target_kind") in ("cheap", "expensive"), "Checkpoint lacks its target kind")
    require(isinstance(value.get("model_state"), dict) and bool(value["model_state"]), "Checkpoint has no model weights")
    require(state_digest(value["model_state"]) == value.get("model_state_sha256"), "Repair checkpoint weights changed")
    require(all(isinstance(t, torch.Tensor) and t.dtype == torch.float32 and torch.isfinite(t).all()
                for t in value["model_state"].values()), "Repair checkpoint requires finite FP32 model weights")
    return value


def load_model(checkpoint_path: str | Path, *, device: str = "cpu", backend: str | None = None):
    """Rebuild architecture and load weights strictly; backend can be changed."""
    checkpoint = read_checkpoint(checkpoint_path)
    config = checkpoint["configuration"]
    selected_backend = backend if backend is not None else config["backend"]
    model = FeatureRepair(seed=int(config["seed"]), backend=selected_backend,
                          max_reference_tokens=int(config["max_reference_tokens"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    require(state_digest(model.state_dict()) == checkpoint["model_state_sha256"], "Loaded model weights differ from the checkpoint")
    selected_device = torch.device(device)
    require(selected_device.type in ("cpu", "cuda"), "Select an explicit cpu or cuda device")
    if selected_device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was selected but is unavailable")
    model.to(selected_device).eval()
    return model, checkpoint
