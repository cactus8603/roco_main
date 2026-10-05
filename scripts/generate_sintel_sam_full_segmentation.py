#!/usr/bin/env python3
"""Generate traced SAM/UnSAMFlow-style full segmentation for Sintel frames."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

from stablebridge.physical_repair.cuda_memory_guard import (
    CudaMemoryReservationPolicyV1,
    CudaMemoryReservationV1,
)
from stablebridge.physical_repair.sam_semantic_smoothness import (
    SAM_FULL_SEGMENTATION_SCHEMA_V1,
    compose_sam_full_segmentation_v1,
)
from stablebridge.physical_repair.u2flow_training_data import (
    build_u2flow_training_manifest_v1,
)


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


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", choices=("vit_h", "vit_l", "vit_b"), default="vit_h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--roles", nargs="+", default=("fit", "validation"),
        choices=("fit", "validation", "calibration", "evaluation"),
    )
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--maximum-masks", type=int, default=255)
    parser.add_argument("--reserve-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-vram-fraction", type=float, default=0.88)
    parser.add_argument("--headroom-mib", type=int, default=3072)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    data_config = args.data_config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sam_source = args.sam_source.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    for path in (data_config, checkpoint, sam_source / "segment_anything" / "__init__.py"):
        if not path.exists():
            raise FileNotFoundError(path)
    payload = json.loads(data_config.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Sintel data config must contain an object")
    manifest = build_u2flow_training_manifest_v1(payload)
    selected_roles = frozenset(args.roles)
    root = Path(manifest.root).resolve()
    frames: dict[str, Path] = {}
    for row in manifest.rows:
        if row.split_role.value not in selected_roles:
            continue
        for identity in row.base_inputs:
            source = Path(identity.path).resolve()
            relative = str(source.relative_to(root))
            frames[relative] = source
    if not frames:
        raise ValueError("selected roles contain no Sintel frames")

    sys.path.insert(0, str(sam_source))
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    finally:
        sys.path.remove(str(sam_source))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for SAM generation but is unavailable")
    sam = sam_model_registry[args.model_type](checkpoint=str(checkpoint)).to(device)
    sam.eval()
    guard = CudaMemoryReservationV1(
        CudaMemoryReservationPolicyV1(
            enabled=bool(args.reserve_vram),
            target_fraction=float(args.target_vram_fraction),
            minimum_headroom_mib=int(args.headroom_mib),
            chunk_mib=256,
            retry_fraction_decrement=0.05,
            minimum_target_fraction=0.65,
        ),
        device,
    )
    memory_report = guard.prime()
    generator_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "output_mode": "binary_mask",
    }
    generator = SamAutomaticMaskGenerator(sam, **generator_kwargs)
    checkpoint_sha256 = _sha256(checkpoint)
    sam_commit = "unknown"
    git_head = sam_source / ".git" / "refs" / "heads" / "main"
    packed_head = sam_source / ".git" / "HEAD"
    if git_head.is_file():
        sam_commit = git_head.read_text(encoding="utf-8").strip()
    elif packed_head.is_file():
        head = packed_head.read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            sam_commit = head

    output_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    ordered = sorted(frames.items())
    for index, (relative, source) in enumerate(ordered, start=1):
        destination = output_root / relative
        if destination.is_file():
            with Image.open(destination) as existing:
                existing_size = existing.size
            with Image.open(source) as source_image:
                source_size = source_image.size
            if existing_size == source_size:
                records[relative] = {
                    "sha256": _sha256(destination),
                    "source_sha256": _sha256(source),
                }
                continue
        with Image.open(source) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        masks = generator.generate(rgb)
        labels = compose_sam_full_segmentation_v1(
            masks, rgb.shape[:2], maximum_masks=args.maximum_masks,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            Image.fromarray(labels).save(temporary, format="PNG")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        records[relative] = {
            "sha256": _sha256(destination),
            "source_sha256": _sha256(source),
            "region_count": int(labels.max()),
        }
        _atomic_json(output_root / "progress.json", {
            "schema": SAM_FULL_SEGMENTATION_SCHEMA_V1,
            "complete": False,
            "completed_frames": index,
            "total_frames": len(ordered),
            "last_relative_path": relative,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "cuda_memory": memory_report,
        })

    _atomic_json(output_root / "manifest.json", {
        "schema": SAM_FULL_SEGMENTATION_SCHEMA_V1,
        "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "source_data_config": str(data_config),
        "source_data_config_sha256": _sha256(data_config),
        "roles": sorted(selected_roles),
        "sam_source": str(sam_source),
        "sam_source_commit": sam_commit,
        "sam_checkpoint": str(checkpoint),
        "sam_checkpoint_sha256": checkpoint_sha256,
        "sam_model_type": args.model_type,
        "generator": generator_kwargs,
        "maximum_masks": args.maximum_masks,
        "frame_count": len(records),
        "records": records,
    })
    (output_root / "progress.json").unlink(missing_ok=True)
    print(json.dumps({
        "output_root": str(output_root),
        "frame_count": len(records),
        "sam_checkpoint_sha256": checkpoint_sha256,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
