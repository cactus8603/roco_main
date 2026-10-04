"""Exact native-lattice GT adapters for Spring and KITTI action qualification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import h5py
import numpy as np


SCHEMA = "stablebridge-action-qualification-bound-flow-GT/v1"
SPRING_OFFSETS = ((0, 0), (1, 0), (0, 1), (1, 1))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_binding(binding: Mapping[str, Any]) -> Path:
    if (
        binding.get("absolute_external_read_only") is not True
        or binding.get("modality") != "flow_ground_truth"
    ):
        raise ValueError("GT binding lost external read-only flow authority")
    path = Path(str(binding.get("path"))).resolve(strict=True)
    if (
        path.stat().st_size != int(binding.get("bytes", -1))
        or _file_sha256(path) != binding.get("sha256")
    ):
        raise ValueError("GT binding file hash drift")
    return path


def _spring(path: Path, lattice_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as stream:
        if "flow" not in stream:
            raise ValueError("Spring flo5 lacks flow dataset")
        dataset = stream["flow"]
        height, width = lattice_hw
        if dataset.shape != (2 * height, 2 * width, 2):
            raise ValueError("Spring GT must be exactly 2x the RGB lattice")
        raw = np.asarray(dataset, dtype=np.float32)
    branches = np.stack([
        raw[dy::2, dx::2, :] for dy, dx in SPRING_OFFSETS
    ]).astype(np.float32, copy=False)
    valid = np.isfinite(branches).all(axis=-1)
    if not np.any(valid):
        raise ValueError("Spring GT has no valid complete flow vector")
    return np.ascontiguousarray(branches), np.ascontiguousarray(valid)


def _kitti(path: Path, lattice_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        encoded is None or encoded.dtype != np.uint16
        or encoded.ndim != 3 or encoded.shape[2] != 3
        or encoded.shape[:2] != lattice_hw
    ):
        raise ValueError("KITTI GT must be uint16 BGR flow on the RGB lattice")
    valid = encoded[..., 0] > 0
    if not np.any(valid):
        raise ValueError("KITTI GT has no valid pixels")
    u = (encoded[..., 2].astype(np.float32) - 32768.0) / 64.0
    v = (encoded[..., 1].astype(np.float32) - 32768.0) / 64.0
    flow = np.stack((u, v), axis=-1)[None, ...]
    return np.ascontiguousarray(flow), np.ascontiguousarray(valid[None, ...])


def decode_bound_flow_GT_v1(
    *, source_dataset: str, binding: Mapping[str, Any],
    lattice_hw: tuple[int, int], component_id: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode a hash-bound official GT payload without resizing or unit changes."""
    if (
        not isinstance(lattice_hw, tuple) or len(lattice_hw) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in lattice_hw)
    ):
        raise ValueError("lattice_hw must be a positive integer pair")
    if not isinstance(component_id, str) or not component_id:
        raise ValueError("component_id must be nonempty")
    path = _validate_binding(binding)
    if source_dataset == "spring":
        flows, valid = _spring(path, lattice_hw)
        rule = "OFFICIAL_MINIMUM_EPE_OVER_FOUR_COMPLETE_SUBPIXEL_VECTORS"
    elif source_dataset == "kitti":
        flows, valid = _kitti(path, lattice_hw)
        rule = "OFFICIAL_SPARSE_VALID_UINT16_FLOW"
    else:
        raise ValueError("source_dataset must be spring or kitti")
    payload = {
        "schema": SCHEMA,
        "component_id": component_id,
        "source_dataset": source_dataset,
        "GT_file_sha256": binding["sha256"],
        "lattice_hw": list(lattice_hw),
        "branch_count": int(flows.shape[0]),
        "scoring_rule": rule,
        "flow_units": "NATIVE_RGB_PIXELS_NO_RESOLUTION_DIVISION",
        "spatial_resize": False,
        "flow_branches_array_sha256": array_sha256(flows),
        "branch_valid_mask_array_sha256": array_sha256(valid),
        "fixed_pixel_support_count": int(np.count_nonzero(np.any(valid, axis=0))),
        "GT_read": True,
        "scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
    }
    return flows, valid, {**payload, "receipt_sha256": canonical_sha256(payload)}


__all__ = [
    "SCHEMA", "SPRING_OFFSETS", "array_sha256", "canonical_sha256",
    "decode_bound_flow_GT_v1",
]
