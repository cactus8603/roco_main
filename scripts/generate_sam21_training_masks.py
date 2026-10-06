#!/usr/bin/env python3
"""Generate resumable SAM 2.1 full segmentations and packed key objects.

The Sintel inventory is derived from the frozen StableBridge scene-role
manifest.  The KITTI inventory is restricted to authorized training RGB
directories in the frozen KITTI data manifest; benchmark/multiview testing
images are not touched by default.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

from stablebridge.physical_repair.sam_semantic_smoothness import (
    SAM_FULL_SEGMENTATION_SCHEMA_V1,
    compose_sam_full_segmentation_v1,
)
from stablebridge.physical_repair.u2flow_training_data import (
    build_u2flow_training_manifest_v1,
)


SAM21_TRAINING_MASK_SCHEMA_V1 = "stablebridge-sam21-training-masks/v1"


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
    parser.add_argument("--dataset", required=True, choices=("sintel", "kitti"))
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sam2-source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sintel-roles", nargs="+",
        default=("fit", "validation", "calibration", "evaluation"),
        choices=("fit", "validation", "calibration", "evaluation"),
    )
    parser.add_argument(
        "--kitti-splits", nargs="+", default=("training",),
        choices=("training", "testing"),
    )
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--stability-score-offset", type=float)
    parser.add_argument("--box-nms-thresh", type=float)
    parser.add_argument("--crop-n-layers", type=int)
    parser.add_argument("--crop-nms-thresh", type=float)
    parser.add_argument("--crop-overlap-ratio", type=float)
    parser.add_argument("--crop-n-points-downscale-factor", type=int)
    parser.add_argument("--min-mask-region-area", type=int)
    parser.add_argument("--maximum-masks", type=int, default=255)
    parser.add_argument(
        "--maximum-frames", type=int, default=0,
        help="debug-only inventory prefix; zero processes the complete inventory",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--use-m2m", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _sintel_inventory(
    config_path: Path, roles: frozenset[str],
) -> tuple[dict[str, Path], dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Sintel data config must contain an object")
    manifest = build_u2flow_training_manifest_v1(payload)
    root = Path(manifest.root).resolve()
    targets: dict[str, Path] = {}
    for row in manifest.rows:
        if row.split_role.value not in roles:
            continue
        for identity in row.base_inputs:
            source = Path(identity.path).resolve()
            targets[str(source.relative_to(root))] = source
    return targets, {
        "source_root": str(root),
        "roles": sorted(roles),
    }


def _kitti_inventory(
    config_path: Path, splits: frozenset[str],
) -> tuple[dict[str, Path], dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != "stablebridge-kitti-u2flow-data/v1":
        raise ValueError("unsupported KITTI data config")
    storage_root = Path(str(payload["storage_root"])).resolve()
    targets: dict[str, Path] = {}
    source_roots: dict[str, str] = {}
    for raw_source in payload.get("sources", ()):
        if not isinstance(raw_source, Mapping):
            raise ValueError("KITTI source entry must be an object")
        source_id = str(raw_source["source_id"])
        extracted = (storage_root / str(raw_source["extracted_relative_path"])).resolve()
        if not extracted.is_dir() or storage_root not in extracted.parents:
            raise ValueError(f"KITTI extraction root is invalid: {source_id}")
        source_roots[source_id] = str(extracted)
        for relative_directory in raw_source.get("required_relative_paths", ()):
            relative = Path(str(relative_directory))
            if relative.parts[:1] not in {(split,) for split in splits}:
                continue
            if relative.name not in {"image_2", "image_3", "colored_0", "colored_1"}:
                continue
            directory = extracted / relative
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for source in sorted(directory.glob("*.png")):
                key = str(Path(source_id) / source.relative_to(extracted))
                targets[key] = source.resolve()
    return targets, {
        "storage_root": str(storage_root),
        "source_roots": source_roots,
        "splits": sorted(splits),
        "testing_authorized": "testing" in splits,
    }


def _key_object_masks(masks: list[dict[str, Any]], image_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_hw
    if not masks:
        return np.zeros((0, height, width), dtype=bool)
    stack = np.stack([
        np.asarray(item["segmentation"], dtype=bool) for item in masks
    ])
    accepted: list[np.ndarray] = []
    for index, item in enumerate(masks):
        mask = stack[index]
        box = item.get("bbox")
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            continue
        box_width, box_height = float(box[2]), float(box[3])
        area = int(item.get("area", int(mask.sum())))
        if not (50 <= box_height <= 200 and 50 <= box_width <= 300):
            continue
        if box_height <= 0 or box_width <= 0 or area / (box_height * box_width) < 0.5:
            continue
        overlap_count = int(np.any(stack[:, mask], axis=1).sum())
        if overlap_count >= 6:
            accepted.append(mask)
    return (
        np.stack(accepted)
        if accepted
        else np.zeros((0, height, width), dtype=bool)
    )


def _write_key_objects(path: Path, masks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(
            temporary,
            packed=np.packbits(masks, axis=-1, bitorder="little"),
            height=np.int32(masks.shape[1]),
            width=np.int32(masks.shape[2]),
            count=np.int32(masks.shape[0]),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = _arguments()
    config_path = args.data_config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sam2_source = args.sam2_source.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    for path in (config_path, checkpoint, sam2_source / "sam2" / "__init__.py"):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.dataset == "sintel":
        targets, inventory = _sintel_inventory(
            config_path, frozenset(args.sintel_roles),
        )
    else:
        targets, inventory = _kitti_inventory(
            config_path, frozenset(args.kitti_splits),
        )
    if not targets:
        raise ValueError("selected dataset inventory is empty")
    if args.maximum_frames < 0:
        raise ValueError("maximum frames must be nonnegative")
    if args.shard_count < 1:
        raise ValueError("shard count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must lie in [0, shard count)")

    sys.path.insert(0, str(sam2_source))
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
    finally:
        sys.path.remove(str(sam2_source))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for SAM2.1 generation but is unavailable")
    model = build_sam2(
        args.model_config, str(checkpoint), device=str(device),
        apply_postprocessing=False,
    )
    model.eval()
    generator_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "output_mode": "binary_mask",
        "use_m2m": bool(args.use_m2m),
    }
    # Keep these absent unless explicitly requested so the current production
    # shards retain identical lineage, while remote quality sweeps can exercise
    # SAM2's denser crop/NMS settings without editing this script.
    for name in (
        "stability_score_offset", "box_nms_thresh", "crop_n_layers",
        "crop_nms_thresh", "crop_overlap_ratio",
        "crop_n_points_downscale_factor", "min_mask_region_area",
    ):
        value = getattr(args, name)
        if value is not None:
            generator_kwargs[name] = value
    generator = SAM2AutomaticMaskGenerator(model, **generator_kwargs)
    checkpoint_sha256 = _sha256(checkpoint)
    commit_result = subprocess.run(
        ["git", "-C", str(sam2_source), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    sam2_commit = (
        commit_result.stdout.strip()
        if commit_result.returncode == 0 else "unknown"
    )
    full_root = output_root / "full_seg"
    key_root = output_root / "key_objects"
    full_root.mkdir(parents=True, exist_ok=True)
    key_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    ordered = sorted(targets.items())
    inventory_frame_count = len(ordered)
    if args.maximum_frames:
        ordered = ordered[:args.maximum_frames]
    selected_frame_count = len(ordered)
    ordered = ordered[args.shard_index::args.shard_count]
    if not ordered:
        raise ValueError("selected SAM2.1 shard is empty")
    shard_root = output_root / "shards"
    shard_stem = f"{args.shard_index:03d}-of-{args.shard_count:03d}"
    for index, (key, source) in enumerate(ordered, start=1):
        full_path = full_root / key
        key_path = key_root / Path(key).with_suffix(".npz")
        if full_path.is_file() and key_path.is_file():
            with Image.open(source) as source_image:
                expected_size = source_image.size
            with Image.open(full_path) as existing:
                full_size = existing.size
            if full_size == expected_size:
                with np.load(key_path) as packed:
                    key_count = int(packed["count"])
                records[key] = {
                    "source_path": str(source),
                    "source_sha256": _sha256(source),
                    "sha256": _sha256(full_path),
                    "full_seg_sha256": _sha256(full_path),
                    "key_objects_sha256": _sha256(key_path),
                    "key_object_count": key_count,
                }
                continue
        with Image.open(source) as image:
            # Torchvision wraps this array with ``torch.from_numpy``; keep it
            # writable so the conversion has defined behavior.
            rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        masks = generator.generate(rgb)
        labels = compose_sam_full_segmentation_v1(
            masks, rgb.shape[:2], maximum_masks=args.maximum_masks,
        )
        objects = _key_object_masks(masks, rgb.shape[:2])
        full_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = full_path.with_name(f".{full_path.name}.tmp-{os.getpid()}")
        try:
            Image.fromarray(labels).save(temporary, format="PNG")
            os.replace(temporary, full_path)
        finally:
            temporary.unlink(missing_ok=True)
        _write_key_objects(key_path, objects)
        records[key] = {
            "source_path": str(source),
            "source_sha256": _sha256(source),
            "sha256": _sha256(full_path),
            "full_seg_sha256": _sha256(full_path),
            "key_objects_sha256": _sha256(key_path),
            "region_count": int(labels.max()),
            "key_object_count": int(objects.shape[0]),
        }
        _atomic_json(shard_root / f"{shard_stem}.progress.json", {
            "schema": SAM21_TRAINING_MASK_SCHEMA_V1,
            "complete": False,
            "dataset": args.dataset,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "completed_frames": index,
            "total_frames": len(ordered),
            "last_key": key,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })

    manifest = {
        "schema": SAM_FULL_SEGMENTATION_SCHEMA_V1,
        "extended_schema": SAM21_TRAINING_MASK_SCHEMA_V1,
        "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "source_data_config": str(config_path),
        "source_data_config_sha256": _sha256(config_path),
        **inventory,
        "sam_source": str(sam2_source),
        "sam_source_commit": sam2_commit,
        "sam_checkpoint": str(checkpoint),
        "sam_checkpoint_sha256": checkpoint_sha256,
        "sam_model_type": "sam2.1_hiera_large",
        "generator": generator_kwargs,
        "maximum_masks": args.maximum_masks,
        "inventory_frame_count": inventory_frame_count,
        "inventory_truncated": selected_frame_count != inventory_frame_count,
        "selected_frame_count": selected_frame_count,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "frame_count": len(records),
        "key_object_encoding": "npz_packbits_little_width_axis",
        "key_object_root": str(key_root),
        "records": records,
    }
    shard_manifest_path = shard_root / f"{shard_stem}.manifest.json"
    _atomic_json(shard_manifest_path, manifest)
    (shard_root / f"{shard_stem}.progress.json").unlink(missing_ok=True)

    # The last completed shard publishes the combined manifest.  A file lock
    # makes this safe when multiple GPU jobs finish at nearly the same time.
    lock_path = shard_root / ".finalize.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        shard_payloads: list[Mapping[str, Any]] = []
        for shard_index in range(args.shard_count):
            candidate = shard_root / (
                f"{shard_index:03d}-of-{args.shard_count:03d}.manifest.json"
            )
            if not candidate.is_file():
                break
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError(f"invalid SAM2.1 shard manifest: {candidate}")
            shard_payloads.append(payload)
        if len(shard_payloads) == args.shard_count:
            lineage_fields = (
                "schema", "extended_schema", "dataset", "source_data_config",
                "source_data_config_sha256", "sam_source_commit",
                "sam_checkpoint_sha256", "sam_model_type", "generator",
                "maximum_masks", "inventory_frame_count",
                "inventory_truncated", "selected_frame_count", "shard_count",
            )
            baseline = shard_payloads[0]
            for payload in shard_payloads[1:]:
                for field in lineage_fields:
                    if payload.get(field) != baseline.get(field):
                        raise ValueError(
                            f"SAM2.1 shard lineage drift for field {field}"
                        )
            combined_records: dict[str, Any] = {}
            for payload in shard_payloads:
                for key, record in dict(payload["records"]).items():
                    if key in combined_records:
                        raise ValueError(f"duplicate SAM2.1 frame across shards: {key}")
                    combined_records[key] = record
            if len(combined_records) != selected_frame_count:
                raise ValueError(
                    "completed SAM2.1 shard manifests do not cover the selected inventory"
                )
            combined = dict(baseline)
            combined.pop("shard_index", None)
            combined["created_utc"] = datetime.now(timezone.utc).isoformat()
            combined["complete"] = True
            combined["frame_count"] = len(combined_records)
            combined["shards"] = [
                str(shard_root / f"{index:03d}-of-{args.shard_count:03d}.manifest.json")
                for index in range(args.shard_count)
            ]
            combined["records"] = combined_records
            _atomic_json(full_root / "manifest.json", combined)
            _atomic_json(output_root / "manifest.json", combined)
    print(json.dumps({
        "output_root": str(output_root),
        "frame_count": len(records),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "sam_checkpoint_sha256": checkpoint_sha256,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
