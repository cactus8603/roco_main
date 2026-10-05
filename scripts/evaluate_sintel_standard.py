#!/usr/bin/env python3
"""Evaluate pinned SEA-RAFT and an integrated checkpoint on standard Sintel train.

MPI-Sintel test ground truth is hidden.  This tool therefore reproduces the
local protocol used by upstream SEA-RAFT's ``validate_sintel``: evaluate the
complete public training split, report Clean and Final separately, average
pixel metrics within each image and then average across images, and feed RGB
to SEA-RAFT in its official float32 0--255 range.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import struct
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from stablebridge.physical_repair.integrated_uncertainty_trainer import (
    FLOW_OUTPUT_CONTRACT_V1,
    INTEGRATED_TRAINER_SCHEMA_V2,
    IntegratedTrainerConfigV2,
    validate_sea_raft_input_v1,
)
from stablebridge.physical_repair.sea_raft_uncertainty_adapter import (
    load_pinned_sea_raft_model,
)


SCHEMA = "stablebridge-sintel-standard-evaluation/v1"


@dataclass(frozen=True)
class SintelPair:
    render_pass: str
    scene: str
    frame: str
    first: Path
    second: Path
    flow: Path


@dataclass(frozen=True)
class PassMetrics:
    pairs: int
    epe: float
    outlier_1px_percent: float
    outlier_3px_percent: float
    outlier_5px_percent: float
    seconds: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_flo(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        if stream.read(4) != b"PIEH":
            raise ValueError(f"invalid Sintel .flo magic: {path}")
        width, height = struct.unpack("<ii", stream.read(8))
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid Sintel .flo dimensions: {path}")
        payload = np.frombuffer(stream.read(), dtype="<f4")
    expected = height * width * 2
    if payload.size != expected:
        raise ValueError(f"truncated Sintel .flo payload: {path}")
    flow = payload.reshape(height, width, 2).astype(np.float32, copy=False)
    if not np.isfinite(flow).all():
        raise ValueError(f"non-finite Sintel flow: {path}")
    return flow


def discover_sintel_training_pairs(root: Path) -> dict[str, tuple[SintelPair, ...]]:
    root = root.expanduser().resolve()
    flow_root = root / "training" / "flow"
    result: dict[str, tuple[SintelPair, ...]] = {}
    for render_pass in ("clean", "final"):
        image_root = root / "training" / render_pass
        if not image_root.is_dir() or not flow_root.is_dir():
            raise FileNotFoundError(f"incomplete Sintel training tree under {root}")
        pairs: list[SintelPair] = []
        for scene_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            images = sorted(scene_dir.glob("*.png"))
            if len(images) < 2:
                raise ValueError(f"Sintel scene has fewer than two frames: {scene_dir}")
            for first, second in zip(images, images[1:]):
                flow = flow_root / scene_dir.name / f"{first.stem}.flo"
                if not flow.is_file():
                    raise FileNotFoundError(flow)
                pairs.append(SintelPair(
                    render_pass=render_pass,
                    scene=scene_dir.name,
                    frame=first.stem,
                    first=first,
                    second=second,
                    flow=flow,
                ))
        if not pairs:
            raise ValueError(f"Sintel {render_pass} inventory is empty")
        result[render_pass] = tuple(pairs)
    clean_identity = tuple((pair.scene, pair.frame) for pair in result["clean"])
    final_identity = tuple((pair.scene, pair.frame) for pair in result["final"])
    if clean_identity != final_identity:
        raise ValueError("Sintel Clean/Final pair identities differ")
    return result


def _image(path: Path) -> torch.Tensor:
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


def _final_flow(output: object) -> torch.Tensor:
    if not isinstance(output, Mapping):
        raise TypeError("SEA-RAFT output must be a mapping")
    flows = output.get("flow")
    if not isinstance(flows, (tuple, list)) or not flows:
        raise ValueError("SEA-RAFT output has no flow sequence")
    flow = flows[-1]
    if not isinstance(flow, torch.Tensor) or flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("SEA-RAFT final flow must be a Bx2xHxW tensor")
    return flow


def _predict(
    model: torch.nn.Module,
    image1: torch.Tensor,
    image2: torch.Tensor,
    *,
    iters: int,
    scale: int,
) -> torch.Tensor:
    validate_sea_raft_input_v1(image1)
    validate_sea_raft_input_v1(image2)
    factor = 2.0 ** scale
    if scale:
        image1 = F.interpolate(image1, scale_factor=factor, mode="bilinear", align_corners=False)
        image2 = F.interpolate(image2, scale_factor=factor, mode="bilinear", align_corners=False)
    output = model(image1, image2, iters=iters, test_mode=True)
    flow = _final_flow(output)
    if scale:
        flow = F.interpolate(flow, scale_factor=1.0 / factor, mode="bilinear", align_corners=False)
        flow = flow * (1.0 / factor)
    return flow


@torch.inference_mode()
def evaluate_pass(
    model: torch.nn.Module,
    pairs: Sequence[SintelPair],
    *,
    device: torch.device,
    iters: int,
    scale: int,
    progress_every: int,
) -> PassMetrics:
    model.eval()
    epe_means: list[float] = []
    below_1: list[float] = []
    below_3: list[float] = []
    below_5: list[float] = []
    started = time.monotonic()
    for index, pair in enumerate(pairs, start=1):
        first = _image(pair.first)[None].to(device, non_blocking=True)
        second = _image(pair.second)[None].to(device, non_blocking=True)
        truth_np = _read_flo(pair.flow)
        truth = torch.from_numpy(truth_np).permute(2, 0, 1)[None].to(device)
        prediction = _predict(model, first, second, iters=iters, scale=scale)
        if prediction.shape != truth.shape:
            raise ValueError(
                f"prediction/GT shape mismatch for {pair.render_pass}/{pair.scene}/{pair.frame}: "
                f"{tuple(prediction.shape)} != {tuple(truth.shape)}"
            )
        error = torch.linalg.vector_norm(prediction.float() - truth.float(), dim=1)
        epe_means.append(float(error.mean().cpu()))
        below_1.append(float((error < 1.0).float().mean().cpu()))
        below_3.append(float((error < 3.0).float().mean().cpu()))
        below_5.append(float((error < 5.0).float().mean().cpu()))
        if progress_every and (index % progress_every == 0 or index == len(pairs)):
            print(
                json.dumps({
                    "event": "progress",
                    "render_pass": pair.render_pass,
                    "completed_pairs": index,
                    "total_pairs": len(pairs),
                    "running_epe": float(np.mean(epe_means)),
                    "utc": _utc_now(),
                }, sort_keys=True),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    return PassMetrics(
        pairs=len(pairs),
        epe=float(np.mean(epe_means)),
        outlier_1px_percent=100.0 * (1.0 - float(np.mean(below_1))),
        outlier_3px_percent=100.0 * (1.0 - float(np.mean(below_3))),
        outlier_5px_percent=100.0 * (1.0 - float(np.mean(below_5))),
        seconds=elapsed,
    )


def _factory(path: str, kwargs: Mapping[str, Any], *, device: torch.device) -> torch.nn.Module:
    module_name, separator, symbol_name = path.partition(":")
    if not separator:
        raise ValueError("model factory must use module:symbol syntax")
    symbol = getattr(importlib.import_module(module_name), symbol_name)
    model = symbol(device=str(device), **dict(kwargs))
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model factory did not return torch.nn.Module")
    return model


def _load_integrated_model(
    config: IntegratedTrainerConfigV2,
    checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("integrated checkpoint must be a mapping")
    if payload.get("schema") != INTEGRATED_TRAINER_SCHEMA_V2:
        raise ValueError("integrated checkpoint schema mismatch")
    if payload.get("config_digest") != config.digest:
        raise ValueError("integrated checkpoint config digest mismatch")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("integrated checkpoint has no model state")
    model = _factory(
        str(config.model["factory"]),
        config.model["kwargs"],
        device=device,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {
        "path": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "saved_utc": payload.get("saved_utc"),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "config_digest": payload.get("config_digest"),
    }


def _load_base_model(
    config: IntegratedTrainerConfigV2,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    kwargs = config.model["kwargs"]
    model, report = load_pinned_sea_raft_model(
        device=str(device),
        vendor_root=Path(str(kwargs["vendor_root"])),
        config_path=Path(str(kwargs["config_path"])),
        checkpoint=Path(str(kwargs["checkpoint"])),
    )
    return model.eval(), dict(report)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sintel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument(
        "--scale", type=int, default=0,
        help="SEA-RAFT pre/post scale exponent; standard Sintel evaluation uses 0",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.iters < 1:
        raise ValueError("iters must be positive")
    if args.progress_every < 0:
        raise ValueError("progress-every must be nonnegative")
    inventory = discover_sintel_training_pairs(args.sintel_root)
    inventory_summary = {
        render_pass: {
            "pairs": len(pairs),
            "scenes": len({pair.scene for pair in pairs}),
        }
        for render_pass, pairs in inventory.items()
    }
    if args.inventory_only:
        print(json.dumps(inventory_summary, indent=2, sort_keys=True))
        return 0
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = IntegratedTrainerConfigV2.from_json(args.config)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": _utc_now(),
        "protocol": {
            "dataset": "MPI-Sintel public training split",
            "render_passes": ["clean", "final"],
            "dataset_rgb_range": "uint8_0_255",
            "matcher_rgb_range": "float32_0_255",
            "backbone_normalization": "2*(rgb/255)-1_inside_SEA_RAFT",
            "flow_units": FLOW_OUTPUT_CONTRACT_V1,
            "metric_reduction": "mean_pixels_per_image_then_mean_images",
            "iters": args.iters,
            "scale": args.scale,
            "full_resolution": args.scale == 0,
            "hidden_test_ground_truth_available": False,
        },
        "inventory": inventory_summary,
        "models": {},
    }

    def run_model(name: str, model: torch.nn.Module, lineage: Mapping[str, object]) -> None:
        passes = {
            render_pass: asdict(evaluate_pass(
                model,
                pairs,
                device=device,
                iters=args.iters,
                scale=args.scale,
                progress_every=args.progress_every,
            ))
            for render_pass, pairs in inventory.items()
        }
        result["models"][name] = {
            "lineage": dict(lineage),
            "passes": passes,
            "clean_final_mean_epe": float(np.mean([
                passes["clean"]["epe"], passes["final"]["epe"],
            ])),
        }
        _atomic_json(args.output.expanduser().resolve(), result)
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_base:
        base_model, base_lineage = _load_base_model(config, device)
        run_model("sea_raft_base", base_model, base_lineage)
        del base_model
    integrated, checkpoint_lineage = _load_integrated_model(config, checkpoint, device)
    run_model("u2_best", integrated, checkpoint_lineage)
    print(json.dumps({
        "event": "completed",
        "output": str(args.output.expanduser().resolve()),
        "models": sorted(result["models"]),
        "utc": _utc_now(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
