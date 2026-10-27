"""Feature-only fitting and prediction. Neither path calls WAFT or loads GT."""
from __future__ import annotations

import json
import math
import random
from functools import wraps
from pathlib import Path

import torch

from .checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from .checkpoint import cpu_tree, load_model, read_checkpoint, save_checkpoint, state_digest
from .data import (FEATURE_SCHEMA, compute_sigma, load_manifest, load_packet, load_sample,
                   packet_checkpoint_sha, packet_pair_id, require, sha256, validate_features, write_json)
from .model import FeatureRepair, features


def epoch_order(count: int, epoch: int, seed: int) -> list[int]:
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def _cpu_reference_threads(function):
    """Scope single-thread CPU reductions so save/resume is reproducible."""
    @wraps(function)
    def run(*args, **kwargs):
        if torch.device(kwargs.get("device", "cpu")).type != "cpu":
            return function(*args, **kwargs)
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            return function(*args, **kwargs)
        finally:
            torch.set_num_threads(previous)
    return run


def feature_mse(prediction, target, sigma, native_mask):
    """Mean of ((prediction-target)/sigma)^2 over both complete padded fields."""
    require(prediction.shape == target.shape and prediction.ndim == 5
            and tuple(prediction.shape[1:3]) == (2, 64), "Matching paired 64-channel fields required")
    require(prediction.dtype == target.dtype == sigma.dtype == torch.float32, "FP32 loss operands required")
    require(tuple(sigma.shape) == (64,) and torch.isfinite(sigma).all() and (sigma > 0).all(), "Positive original channel RMS required")
    require(native_mask.dtype == torch.bool and tuple(native_mask.shape) == (1, 1, 1, *prediction.shape[-2:])
            and native_mask.any(), "Native geometry metadata required")
    require(len({v.device for v in (prediction, target, sigma, native_mask)}) == 1, "Loss operand devices must match")
    return ((prediction - target) / sigma.reshape(1, 1, 64, 1, 1)).square().mean()


def clipped_step(model, optimizer, *, max_norm: float = 1.0) -> dict:
    """FP64 global norm and scaling, then ordinary FP32 AdamW update."""
    parameters = list(model.parameters())
    require(all(p.grad is not None for p in parameters), "All repair parameters must have gradient tensors")
    require(all(torch.isfinite(p.grad).all() for p in parameters), "Nonfinite repair gradient")
    total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    for parameter in parameters:
        total.add_(parameter.grad.detach().double().square().sum())
    norm = float(total.sqrt())
    require(math.isfinite(norm), "Nonfinite FP64 global gradient norm")
    coefficient = min(1.0, max_norm / (norm + 1e-6))
    for parameter in parameters:
        parameter.grad.copy_((parameter.grad.double() * coefficient).float())
    optimizer.step()
    return {"gradient_norm": norm, "clip_coefficient": coefficient}


def _operands(sample, device):
    return [sample[key].detach().to(device) for key in ("original", "target", "sigma", "native_mask")]


def _rng_state(device):
    result = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device)
    return result


def _restore_rng(value, device):
    torch.set_rng_state(value["torch"])
    random.setstate(value["python"])
    if device.type == "cuda":
        require("cuda" in value, "Resuming on CUDA requires a saved CUDA RNG state")
        torch.cuda.set_rng_state(value["cuda"], device)


@torch.no_grad()
def evaluate_features(model, manifest, device) -> dict:
    """Fixed-checkpoint MSE; online training losses are kept separately."""
    model.eval()
    records = []
    for index, row in enumerate(manifest["samples"]):
        sample = load_sample(manifest, row)
        original, target, sigma, mask = _operands(sample, device)
        prediction = model(original, sigma, mask)
        require(torch.isfinite(prediction).all(), "Nonfinite repaired feature")
        scale = sigma.double().reshape(1, 1, 64, 1, 1)
        identity = ((original.double() - target.double()) / scale).square()
        error = ((prediction.double() - target.double()) / scale).square()
        records.append({"pair_id": row["pair_id"], "identity_mse": float(identity.mean()),
                        "repaired_mse": float(error.mean()),
                        "frame_identity_mse": [float(identity[:, frame].mean()) for frame in range(2)],
                        "frame_repaired_mse": [float(error[:, frame].mean()) for frame in range(2)]})
    return {"samples": records, "identity_mse": math.fsum(r["identity_mse"] for r in records) / len(records),
            "repaired_mse": math.fsum(r["repaired_mse"] for r in records) / len(records),
            "weighting": "equal mean over sample pairs; both frames and every padded coordinate",
            "interpretation": "Feature fitting on these samples; this does not establish optical-flow EPE or held-out improvement"}


@_cpu_reference_threads
def train(manifest_path: str | Path, output_dir: str | Path, *, device: str = "cpu",
          backend: str = "reference", epochs: int = 10, seed: int = 20261026,
          resume: str | Path | None = None, max_updates: int | None = None,
          checkpoint_every: int | None = None, max_reference_tokens: int = 4096) -> dict:
    """Train one pair per update with natural, seeded per-epoch permutations.

    Defaults reproduce the feature learner's optimizer recipe. The portable CPU
    reference backend is for small tensors; full native grids require the CUDA
    backend. A resume restores weights, AdamW moments, next sample cursor and
    RNG. CPU reference fitting scopes one intra-op thread and restores the
    caller's setting. Exact CPU resume is tested; CUDA kernel nondeterminism
    can still affect numerical replay. max_updates is a total update limit
    for a controlled interruption.
    """
    require(type(epochs) is int and epochs > 0, "epochs must be a positive integer")
    require(type(seed) is int, "seed must be an integer")
    manifest = load_manifest(manifest_path)
    count = len(manifest["samples"])
    total_updates = count * epochs
    limit = total_updates if max_updates is None else max_updates
    require(type(limit) is int and 0 < limit <= total_updates, "max_updates must be in 1..epochs*sample_count")
    interval = count if checkpoint_every is None else checkpoint_every
    require(type(interval) is int and interval > 0, "checkpoint_every must be positive")
    selected_device = torch.device(device)
    require(selected_device.type in ("cpu", "cuda"), "Select an explicit cpu or cuda device")
    if selected_device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was selected but is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    # Validate every packet before any optimizer update or output is created.
    for row in manifest["samples"]:
        load_sample(manifest, row)
    output = Path(output_dir).resolve()
    config = {"epochs": epochs, "seed": seed, "learning_rate": 0.001, "betas": [0.9, 0.999],
              "epsilon": 1e-12, "weight_decay": 0.0, "gradient_clip_norm": 1.0,
              "gradient_norm_precision": "float64", "warmup": False,
              "pairs_per_update": 1, "epoch_shuffle": "Python Random(seed + zero_based_epoch)",
              "cpu_intraop_threads": 1 if selected_device.type == "cpu" else None,
              "torch_version": str(torch.__version__), "autocast": False, "tf32": False}
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(seed)
        random.seed(seed)
        model = FeatureRepair(backend=backend, seed=seed, max_reference_tokens=max_reference_tokens).to(selected_device).train()
        history, step = [], 0
    else:
        checkpoint = read_checkpoint(resume)
        require(checkpoint.get("manifest_sha256") == manifest["_sha256"], "Resume dataset differs from the checkpoint")
        require(checkpoint.get("training") == config, "Resume training settings differ from the checkpoint")
        require(checkpoint.get("device_type") == selected_device.type, "Exact resume requires the original device type")
        require(checkpoint["configuration"]["backend"] == backend, "Exact resume requires the original backend")
        require(checkpoint["configuration"]["max_reference_tokens"] == max_reference_tokens, "Resume model configuration differs")
        if output.exists():
            # Only this run's own directory can be continued. Other output is never overwritten.
            require(Path(resume).resolve().parent == output, "Existing output directory is not the resumed checkpoint directory")
        else:
            output.mkdir(parents=True)
        model, _ = load_model(resume, device=device, backend=backend)
        model.train()
        history, step = checkpoint["history"], checkpoint["updates_completed"]
        require(len(history) == step and step < limit, "Resume must advance beyond its saved update cursor")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-12,
                                  weight_decay=0.0, foreach=False, fused=False)
    if resume is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        _restore_rng(checkpoint["rng_state"], selected_device)
    last_checkpoint = output / "checkpoint.pt"

    def save():
        optimizer.zero_grad(set_to_none=True)
        packet = {"schema": CHECKPOINT_SCHEMA, "model_state": model.state_dict(),
                  "model_state_sha256": state_digest(model.state_dict()), "configuration": model.configuration(),
                  "optimizer_state": optimizer.state_dict(), "rng_state": _rng_state(selected_device),
                  "training": config, "history": history, "updates_completed": step,
                  "next_epoch": step // count, "next_position": step % count,
                  "manifest_sha256": manifest["_sha256"], "waft_checkpoint_sha256": manifest["waft_checkpoint_sha256"],
                  "target_kind": manifest["target_kind"], "device_type": selected_device.type,
                  "sample_ids": [r["pair_id"] for r in manifest["samples"]]}
        save_checkpoint(last_checkpoint, packet, overwrite=last_checkpoint.exists())
        write_json(output / "history.json", history, overwrite=(output / "history.json").exists())

    while step < limit:
        epoch, position = divmod(step, count)
        index = epoch_order(count, epoch, seed)[position]
        row = manifest["samples"][index]
        original, target, sigma, mask = _operands(load_sample(manifest, row), selected_device)
        optimizer.zero_grad(set_to_none=True)
        prediction = features(model, original, sigma, mask)
        if step == 0:
            require(torch.equal(prediction.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8)), "Zero-readout model must start at exact identity")
        loss = feature_mse(prediction, target, sigma, mask)
        require(torch.isfinite(loss), "Nonfinite feature loss")
        loss.backward()
        report = clipped_step(model, optimizer)
        step += 1
        history.append({"update": step, "epoch": epoch + 1, "position": position + 1, "pair_id": row["pair_id"],
                        "online_mse_before_update": float(loss.detach()), **report})
        write_json(output / "progress.json", {"updates_completed": step, "total_updates": total_updates,
                                              "status": "training", "pair_id": row["pair_id"]},
                   overwrite=(output / "progress.json").exists())
        if step % interval == 0 or step == limit:
            save()
    metrics = evaluate_features(model, manifest, selected_device)
    result = {"status": "completed" if step == total_updates else "paused_at_update_limit",
              "updates_completed": step, "total_updates": total_updates, "epochs": epochs,
              "sample_count": count, "target_kind": manifest["target_kind"],
              "checkpoint": str(last_checkpoint), "checkpoint_sha256": sha256(last_checkpoint),
              "feature_metrics": metrics, "waft_forward_calls": 0, "waft_backward_calls": 0, "direct_gt_flow_loss": False}
    write_json(output / "result.json", result, overwrite=(output / "result.json").exists())
    write_json(output / "progress.json", {"updates_completed": step, "total_updates": total_updates, "status": result["status"]}, overwrite=True)
    return result


@torch.no_grad()
def infer_features(checkpoint_path: str | Path, input_packet_path: str | Path, output_path: str | Path,
                   *, device: str = "cpu", backend: str | None = None) -> dict:
    """Repair features without using targets, RGB or GT in the forward pass.

    New portable input packets contain features and metadata only. A legacy
    packet may also contain other tensors: the safe loader opens its mapping,
    but those extra values are never passed to the repair model.
    """
    output_path = Path(output_path)
    require(not output_path.exists(), "Prediction output already exists")
    packet = load_packet(input_packet_path)
    original = packet.get("features", packet.get("original"))
    require(original is not None, "Input packet needs features or the compact cache's original field")
    validate_features(original)
    native_hw = packet.get("native_hw")
    if native_hw is None and isinstance(packet.get("rgb"), torch.Tensor):
        native_hw = list(packet["rgb"].shape[-2:])
    require(native_hw is not None and "padding" in packet, "Prediction packet requires native geometry")
    pair_id = packet_pair_id(packet)
    require(isinstance(pair_id, str) and bool(pair_id), "Prediction packet requires pair identity")
    sigma, mask = compute_sigma(original, native_hw, packet["padding"])
    model, checkpoint = load_model(checkpoint_path, device=device, backend=backend)
    require(packet_checkpoint_sha(packet) == checkpoint["waft_checkpoint_sha256"], "Input features and repair checkpoint use different frozen WAFT weights")
    prediction = model(original.to(device), sigma.to(device), mask.to(device)).cpu()
    validate_features(prediction)
    result = {"schema": FEATURE_SCHEMA, "pair_id": pair_id, "domain": packet.get("domain", "unspecified"),
              "native_hw": list(native_hw), "padding": list(packet["padding"]), "features": prediction,
              "waft_checkpoint_sha256": checkpoint["waft_checkpoint_sha256"],
              "repair_checkpoint_sha256": sha256(checkpoint_path), "input_packet_sha256": sha256(input_packet_path),
              "target_kind_used_for_training": checkpoint["target_kind"]}
    save_checkpoint(output_path, result)
    return {"output": str(output_path.resolve()), "pair_id": pair_id, "shape": list(prediction.shape),
            "sha256": sha256(output_path), "targets_or_gt_used": False}
