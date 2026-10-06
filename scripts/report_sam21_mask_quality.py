#!/usr/bin/env python3
"""Summarize an incomplete or completed SAM2.1 preprocessing output tree."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _storage(paths: list[Path]) -> dict[str, int]:
    return {
        "apparent_bytes": sum(path.stat().st_size for path in paths),
        "allocated_bytes": sum(path.stat().st_blocks * 512 for path in paths),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    full_root = root / "full_seg"
    key_root = root / "key_objects"
    full_paths = sorted(full_root.rglob("*.png")) if full_root.is_dir() else []
    key_paths = sorted(key_root.rglob("*.npz")) if key_root.is_dir() else []
    full_keys = {str(path.relative_to(full_root)) for path in full_paths}
    key_keys = {
        str(path.relative_to(key_root).with_suffix(".png")) for path in key_paths
    }

    key_counts: list[int] = []
    key_pixels: list[int] = []
    invalid_key_files: list[str] = []
    for path in key_paths:
        try:
            with np.load(path, allow_pickle=False) as payload:
                count = int(payload["count"])
                height = int(payload["height"])
                width = int(payload["width"])
                packed = payload["packed"]
            if packed.shape != (count, height, (width + 7) // 8):
                raise ValueError("packed shape drift")
            key_counts.append(count)
            if count:
                unpacked = np.unpackbits(
                    packed, axis=-1, count=width, bitorder="little",
                )
                key_pixels.extend(
                    int(value) for value in unpacked.sum(axis=(1, 2)).tolist()
                )
        except Exception as exc:  # report every damaged artifact together
            invalid_key_files.append(f"{path}: {type(exc).__name__}: {exc}")

    region_counts: list[int] = []
    invalid_full_files: list[str] = []
    for path in full_paths:
        try:
            with Image.open(path) as image:
                labels = np.asarray(image)
            if labels.ndim != 2:
                raise ValueError("segmentation is not single-channel")
            region_counts.append(int(labels.max()))
        except Exception as exc:
            invalid_full_files.append(f"{path}: {type(exc).__name__}: {exc}")

    nonempty = sum(value > 0 for value in key_counts)
    result: dict[str, Any] = {
        "schema": "stablebridge-sam21-mask-quality-report/v1",
        "output_root": str(root),
        "expected_frames": args.expected_frames,
        "full_segmentation_files": len(full_paths),
        "key_object_files": len(key_paths),
        "paired_files": len(full_keys & key_keys),
        "missing_key_object_files": sorted(full_keys - key_keys),
        "missing_full_segmentation_files": sorted(key_keys - full_keys),
        "invalid_full_segmentation_files": invalid_full_files,
        "invalid_key_object_files": invalid_key_files,
        "full_segmentation_storage": _storage(full_paths),
        "key_object_storage": _storage(key_paths),
        "region_count": {
            "mean": None if not region_counts else sum(region_counts) / len(region_counts),
            "median": _percentile(region_counts, 0.5),
            "p95": _percentile(region_counts, 0.95),
            "maximum": max(region_counts, default=None),
        },
        "key_objects": {
            "total": sum(key_counts),
            "nonempty_frames": nonempty,
            "nonempty_frame_fraction": (
                None if not key_counts else nonempty / len(key_counts)
            ),
            "count_histogram": {
                str(key): value for key, value in sorted(Counter(key_counts).items())
            },
            "object_pixel_median": _percentile(key_pixels, 0.5),
            "object_pixel_p95": _percentile(key_pixels, 0.95),
        },
        "semantic_cache_capacity_reference": 100,
        "has_at_least_100_distinct_key_objects": sum(key_counts) >= 100,
        "complete_manifest_present": (full_root / "manifest.json").is_file(),
    }
    if args.expected_frames is not None:
        result["completion_fraction"] = len(full_paths) / args.expected_frames
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_output is not None:
        target = args.json_output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
