"""Offline official WAFT loading and the image-to-feature-to-flow workflow.

WAFT is always fully frozen. Ground truth is read only during offline target
preparation; image inference accepts just an image pair and repair weights.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from . import data
from .targets import (CHECKPOINT_SHA256, EXPENSIVE_STEPS, FrozenWAFTBackend, cheap_target,
                      expensive_target)


SOURCE_COMMIT = "b152ff1cad1af8c185ee7b141997c48ff3334c87"
STATE_SHA256 = "994e1164487d20779ec14a72d828a8d20433d00ac553ecaeb5d09aab554227ee"
CHECKPOINT_PROFILES = {
    CHECKPOINT_SHA256: dict(name='source-only', state_sha256=STATE_SHA256),
    '656bcbeb7c6b8627eae577d5564f2849f32e1d53f85dcac7324fbeddc96bc9da':
        dict(name='kitti', state_sha256='2c5e8002cb19e0f8d05ee2bfbb1cec6b0507e7690defeebc2b938c7670bf377b'),
    'aad14eea3d68a70a4009332d941a863a1e5f513af252f6d9136725e654808322':
        dict(name='sintel', state_sha256='040e5864cde10395466f9102b3ea72f629263cb1ce1068d6afcc2400ba2b97c6'),
}
MODEL_CONFIG = dict(feature_encoder="dav2", iterative_module="vits", algorithm="waft-a2",
                    iters=5, scale=0, use_var=True, var_min=0, var_max=10)


def _state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str((tuple(value.shape), value.dtype)).encode() + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _offline_constructor(module, timm):
    """Discard separate pretrained initializers; the full WAFT state replaces them."""
    create_model, depth = timm.create_model, module.DepthAnythingFeature
    connect, connection = socket.socket.connect, socket.create_connection
    changes = dict(timm=0, depth=0, network_attempts=0)

    def offline_timm(name, *args, **kwargs):
        data.require(name == "vit_small_patch16_224", "Unexpected WAFT initializer architecture")
        changes["timm"] += 1
        kwargs["pretrained"] = False
        return create_model(name, *args, **kwargs)

    def offline_depth(*args, **kwargs):
        changes["depth"] += 1
        kwargs["pretrained"] = False
        return depth(*args, **kwargs)

    def no_network(*args, **kwargs):
        changes["network_attempts"] += 1
        raise RuntimeError("WAFT construction is offline; network access is disabled")

    try:
        timm.create_model, module.DepthAnythingFeature = offline_timm, offline_depth
        socket.socket.connect, socket.create_connection = no_network, no_network
        yield changes
    finally:
        timm.create_model, module.DepthAnythingFeature = create_model, depth
        socket.socket.connect, socket.create_connection = connect, connection


def _official_modules(source: Path):
    """Reject collisions with another checkout's generic model/utils namespaces."""
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in ("model", "utils", "thirdparty"):
            location = getattr(module, "__file__", None)
            data.require(location is None or Path(location).resolve().is_relative_to(source),
                         "A different model/utils/thirdparty package is already loaded; use a fresh process")
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("model.waft_a2")
        helpers = importlib.import_module("utils.utils")
    finally:
        sys.path.remove(str(source))
    data.require(Path(module.__file__).resolve() == source / "model/waft_a2.py",
                 "WAFT model came from a different source tree")
    data.require(Path(helpers.__file__).resolve() == source / "utils/utils.py",
                 "WAFT geometry helpers came from a different source tree")
    return module, helpers


def load_backend(source, checkpoint, device="cpu", *,
                 expected_checkpoint_sha=None, verify_source=True):
    """Strictly load a known complete official checkpoint on CPU, then move it.

    The caller supplies an official WAFT checkout and its complete checkpoint.
    No separate DAv2/ViT weights are downloaded or read. Source checking may be
    disabled only for an already verified source archive without Git metadata.
    """
    source, checkpoint = Path(source).resolve(strict=True), Path(checkpoint).resolve(strict=True)
    device = torch.device(device)
    data.require(device.type in ("cpu", "cuda"), "WAFT supports explicit CPU or CUDA execution")
    actual_sha = data.sha256(checkpoint)
    if expected_checkpoint_sha is None:
        data.require(actual_sha in CHECKPOINT_PROFILES, 'WAFT checkpoint SHA256 differs from all known official checkpoints')
        expected_checkpoint_sha = actual_sha
    data.require(data.valid_sha(expected_checkpoint_sha), "Expected checkpoint SHA256 is required")
    data.require(actual_sha == expected_checkpoint_sha, "WAFT checkpoint SHA256 differs")
    if verify_source:
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        data.require(head == SOURCE_COMMIT, "Use the pinned official WAFT source commit: " + SOURCE_COMMIT)
        changes = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
        data.require(not changes.strip(), "The pinned official WAFT tracked source files must be unchanged")
    data.require(expected_checkpoint_sha in CHECKPOINT_PROFILES, 'Unknown official checkpoint profile')
    profile = CHECKPOINT_PROFILES[expected_checkpoint_sha]
    data.require(importlib.metadata.version("timm") == "1.0.11", "WAFT requires timm==1.0.11")
    module, helpers = _official_modules(source)
    with torch.device("cpu"), torch.random.fork_rng(devices=[]), _offline_constructor(module, importlib.import_module("timm")) as initialized:
        core = module.WAFTv2(argparse.Namespace(**MODEL_CONFIG))
    data.require(initialized == dict(timm=1, depth=1, network_attempts=0), "Unexpected WAFT initializer or network attempt")
    data.require(all(p.device.type == "cpu" for p in core.parameters()), "WAFT must be initialized on CPU")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and set(state) == {"state_dict"}:
        state = state["state_dict"]
    data.require(isinstance(state, dict) and state and all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                                        for k, v in state.items()), "Complete tensor-only WAFT state required")
    if all(name.startswith("module.") for name in state):
        state = {name.removeprefix("module."): value for name, value in state.items()}
    expected = core.state_dict()
    data.require(len(state) == 508 and set(state) == set(expected), "Checkpoint must contain all 508 exact WAFT tensors")
    for name, value in state.items():
        data.require(value.dtype == expected[name].dtype and value.shape == expected[name].shape
                     and bool(torch.isfinite(value).all()), "Checkpoint shape/dtype/finite value differs: " + name)
    digest = _state_digest(state)
    data.require(digest == profile['state_sha256'], "Loaded state differs from the complete declared official WAFT checkpoint")
    core.load_state_dict(state, strict=True)
    data.require(_state_digest(core.state_dict()) == digest, "Strictly loaded WAFT tensor bytes differ")
    core.eval().requires_grad_(False)
    del state, expected
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    core = core.to(device)
    backend = FrozenWAFTBackend(core, helpers.Padder, helpers.coords_grid, helpers.bilinear_sampler)
    backend.metadata = dict(waft_checkpoint_sha256=expected_checkpoint_sha, source_commit=SOURCE_COMMIT,
                            state_sha256=digest, parameter_tensors=508, parameters=56_932_619,
                            frozen=True, updates=5, source_verified=bool(verify_source))
    return backend


def load_rgb_pair(image1, image2) -> torch.Tensor:
    """Read two equally sized RGB images as CPU FP32 [1,2,3,H,W], in 0..255."""
    frames = []
    for path in (image1, image2):
        with Image.open(path) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        frames.append(torch.from_numpy(array).permute(2, 0, 1).float())
    data.require(frames[0].shape == frames[1].shape, "The RGB pair must have equal native dimensions")
    return torch.stack(frames, 0).unsqueeze(0)


def load_ground_truth(path):
    """Read a canonical NPZ containing FP32 flow, bool valid, optional visible.

    Flow is [H,W,2] or [1,2,H,W]; masks are [H,W] or [1,H,W]. Visibility
    means known nonoccluded support and is mandatory for cheap targets.
    """
    with np.load(path, allow_pickle=False) as saved:
        data.require("flow" in saved and "valid" in saved, "GT NPZ needs flow and valid arrays")
        flow = saved["flow"].copy()
        valid = saved["valid"].copy()
        visible = saved["visible"].copy() if "visible" in saved else None
    data.require(flow.dtype == np.float32, "GT flow must be float32")
    if flow.ndim == 3 and flow.shape[-1] == 2:
        flow = flow.transpose(2, 0, 1)[None]
    data.require(flow.ndim == 4 and flow.shape[:2] == (1, 2), "GT flow must be [H,W,2] or [1,2,H,W]")
    shape = (1, *flow.shape[-2:])
    masks = []
    for name, mask in (("valid", valid), ("visible", visible)):
        if mask is None:
            masks.append(None)
            continue
        data.require(mask.dtype == np.bool_, "GT " + name + " mask must be boolean")
        mask = mask[None] if mask.ndim == 2 else mask
        data.require(mask.shape == shape, "GT " + name + " mask native geometry differs")
        masks.append(torch.from_numpy(np.ascontiguousarray(mask)))
    tensor = torch.from_numpy(np.ascontiguousarray(flow))
    data.require(bool((masks[0] & torch.isfinite(tensor).all(1)).any()), "At least one finite valid GT pixel is required")
    return tensor, masks[0], masks[1]


def _save_tensor_packet(path: Path, packet: dict) -> None:
    value = {key: item.detach().cpu().contiguous() if isinstance(item, torch.Tensor) else item
             for key, item in packet.items()}
    with path.open("xb") as stream:
        torch.save(value, stream)


def prepare_dataset(spec_path, output_dir, source, checkpoint, device="cpu", target_kind="cheap",
                    *, target_steps: int | None = None):
    """Generate full paired input/target features from an explicit RGB/GT spec."""
    data.require(target_kind in ("cheap", "expensive"), "target_kind must be cheap or expensive")
    data.require(target_steps is None or (type(target_steps) is int and target_steps in EXPENSIVE_STEPS),
                 "target_steps must be one of 32, 256, 512")
    data.require(target_steps is None or target_kind == "expensive",
                 "target_steps is only valid for expensive targets")
    steps = 512 if target_steps is None else target_steps
    spec_path = Path(spec_path).resolve(strict=True)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    rows = spec.get("samples")
    data.require(isinstance(rows, list) and rows, "Raw sample specification needs nonempty samples")
    ids = [row.get("pair_id") for row in rows]
    data.require(all(isinstance(x, str) and x for x in ids) and len(ids) == len(set(ids)), "Unique nonempty pair_id values required")
    output = Path(output_dir).resolve()
    data.require(not output.exists(), "Output directory already exists; choose a new directory")
    backend = load_backend(source, checkpoint, device)
    output.mkdir(parents=True, exist_ok=False)
    portable = dict(schema=data.SCHEMA, waft_checkpoint_sha256=backend.metadata["waft_checkpoint_sha256"],
                    target_kind=target_kind, samples=[])
    if target_kind == "expensive":
        portable["target_steps"] = steps
    for index, row in enumerate(rows):
        domain = row.get("domain")
        data.require(domain in ("kitti", "sintel-clean", "sintel-final", "synthetic"), "Unknown domain")
        resolve = lambda key: (spec_path.parent / row[key]).resolve(strict=True)
        pair = load_rgb_pair(resolve("image1"), resolve("image2"))
        gt, valid, visible = load_ground_truth(resolve("flow"))
        data.require(tuple(gt.shape[-2:]) == tuple(pair.shape[-2:]), "GT and RGB native geometry differ")
        original = backend.encode_pair(pair)
        teacher_id = row["pair_id"]
        teacher_domain = domain
        if target_kind == "cheap":
            data.require(visible is not None, "Cheap targets require a known-nonoccluded visible mask")
            teacher_pair, cached = pair, original
            if domain == "sintel-final":
                teacher_id = row.get("target_source_pair_id")
                data.require(isinstance(teacher_id, str) and bool(teacher_id), "Cheap Final requires target_source_pair_id for the paired Clean RGB")
                data.require("teacher_image1" in row and "teacher_image2" in row, "Cheap Final requires both paired Clean teacher images")
                teacher_pair = load_rgb_pair(resolve("teacher_image1"), resolve("teacher_image2"))
                data.require(teacher_pair.shape == pair.shape, "Paired Clean/Final RGB must share native geometry and the specified flow GT")
                cached = None
                teacher_domain = "sintel-clean"
            target = cheap_target(backend, teacher_pair, gt, valid, visible, original=cached)
        else:
            target = expensive_target(backend, original, gt.to(device), valid.to(device), steps=steps)
        metadata = dict(schema=data.FEATURE_SCHEMA, pair_id=row["pair_id"], domain=domain,
                        native_hw=list(original.native_hw), padding=list(original.padding),
                        waft_checkpoint_sha256=backend.metadata["waft_checkpoint_sha256"])
        input_name, target_name = f"pair-{index:06d}-input.pt", f"pair-{index:06d}-target.pt"
        _save_tensor_packet(output / input_name, dict(metadata, features=original.features))
        target_metadata = dict(metadata, target_kind=target_kind, target_method=target["method"],
                               teacher_domain=teacher_domain, target_source_pair_id=teacher_id,
                               teacher_features=target["teacher_features"])
        target_metadata.update(pair_id=teacher_id, domain=teacher_domain)
        if target_kind == "expensive":
            for key in ("qualified", "selected_step", "optimizer_updates", "baseline_metrics", "target_metrics"):
                target_metadata[key] = target[key]
        _save_tensor_packet(output / target_name, target_metadata)
        entry = dict(pair_id=row["pair_id"], domain=domain, native_hw=metadata["native_hw"],
                     padding=metadata["padding"], input_file=input_name, target_file=target_name)
        if target_kind == "expensive":
            entry["teacher_updates"] = target["optimizer_updates"]
        if domain == "sintel-final" and target_kind == "cheap":
            entry["target_source_pair_id"] = teacher_id
        portable["samples"].append(entry)
    data.write_json(output / "samples.json", portable)
    data.prepare_manifest(output / "samples.json", output / "manifest.json")
    return output / "manifest.json"


def infer_images(checkpoint, image1, image2, output_npz, source, waft_checkpoint,
                 device="cpu", backend="reference"):
    """Run GT-free RGB inference and save every native flow plus final flow H,W,2."""
    from .checkpoint import load_model
    output = Path(output_npz)
    data.require(not output.exists(), "Inference output already exists; choose a new path")
    frozen = load_backend(source, waft_checkpoint, device)
    repair, saved = load_model(checkpoint, device=device, backend=backend)
    data.require(saved["waft_checkpoint_sha256"] == frozen.metadata["waft_checkpoint_sha256"],
                 "Repair checkpoint was trained with another WAFT checkpoint")
    pair = load_rgb_pair(image1, image2)
    original = frozen.encode_pair(pair)
    sigma, mask = data.compute_sigma(original.features.detach().cpu(), original.native_hw, original.padding)
    with torch.no_grad():
        repaired = repair(original.features, sigma.to(device), mask.to(device))
        predictions = frozen.solve(repaired, original.native_hw, original.padding)
    arrays = {"flow": predictions["flow"][-1][0].permute(1, 2, 0).cpu().numpy()}
    for key in ("flow", "info"):
        arrays[key + "_steps"] = np.stack([value[0].permute(1, 2, 0).cpu().numpy() for value in predictions[key]])
    metadata = dict(frozen.metadata, target_kind=saved["target_kind"], native_hw=list(original.native_hw),
                    padding=list(original.padding), repair_checkpoint_sha256=data.sha256(checkpoint),
                    ground_truth_used=False)
    arrays["metadata"] = np.array(json.dumps(metadata))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez(stream, **arrays)
    return metadata

