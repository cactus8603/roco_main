"""Convert original KITTI PNG or Sintel FLO annotations into portable NPZ.

This is an offline CPU data conversion. Flow supervision retains every valid
GT pixel; a separate visibility mask is used only when constructing targets.
"""
from __future__ import annotations

from pathlib import Path
import struct

import cv2
import numpy as np

from .data import require, sha256


def _kitti_png(path: Path) -> np.ndarray:
    labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    require(labels is not None and labels.dtype == np.uint16
            and labels.ndim == 3 and labels.shape[2] == 3,
            "KITTI flow must be an unchanged three-channel uint16 PNG")
    require(labels.shape[0] > 0 and labels.shape[1] > 0, "KITTI flow must have positive dimensions")
    return labels


def _sintel_flow(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(12)
        require(len(header) == 12, "Sintel FLO needs a complete 12-byte little-endian header")
        magic, width, height = struct.unpack("<fii", header)
        require(magic == 202021.25 and width > 0 and height > 0,
                "Invalid Sintel FLO magic or dimensions")
        expected = 12 + height * width * 2 * 4
        require(path.stat().st_size == expected, "Sintel FLO payload size differs from its native dimensions")
        payload = stream.read()
    return np.frombuffer(payload, dtype="<f4").reshape(height, width, 2).astype(np.float32, copy=True)


def convert_ground_truth(format, flow_path, output_path, visibility_path=None) -> dict:
    """Save canonical ``flow``, ``valid`` and optional ``visible`` NPZ arrays.

    ``format='kitti'`` reads the original flow_occ PNG; optional visibility
    is its corresponding flow_noc PNG. Values always come from flow_occ.
    ``format='sintel'`` reads a little-endian FLO; optional visibility is the
    corresponding occlusions PNG, where zero means known nonoccluded.

    Output flow is FP32 [H,W,2], with bool [H,W] masks. Existing output is
    never overwritten. Omitting visibility is suitable for expensive targets;
    cheap target preparation rejects an annotation without visibility.
    """
    require(format in ("kitti", "sintel"), "Select format='kitti' or format='sintel'")
    flow_path = Path(flow_path).resolve(strict=True)
    output_path = Path(output_path).resolve()
    require(not output_path.exists(), "Converted GT output already exists; choose a new path")
    visibility_path = Path(visibility_path).resolve(strict=True) if visibility_path is not None else None
    visible = None
    if format == "kitti":
        labels = _kitti_png(flow_path)
        u = (labels[..., 2].astype(np.float32) - np.float32(32768)) / np.float32(64)
        v = (labels[..., 1].astype(np.float32) - np.float32(32768)) / np.float32(64)
        flow = np.stack((u, v), axis=-1)
        valid = (labels[..., 0] != 0) & np.isfinite(flow).all(axis=-1)
        if visibility_path is not None:
            nonoccluded = _kitti_png(visibility_path)
            require(nonoccluded.shape == labels.shape, "KITTI flow_occ and flow_noc native dimensions differ")
            visible = nonoccluded[..., 0] != 0
    else:
        flow = _sintel_flow(flow_path)
        valid = np.isfinite(flow).all(axis=-1) & (np.abs(flow) < np.float32(1000)).all(axis=-1)
        if visibility_path is not None:
            occlusions = cv2.imread(str(visibility_path), cv2.IMREAD_UNCHANGED)
            require(occlusions is not None and occlusions.dtype == np.uint8
                    and occlusions.ndim == 2 and occlusions.shape == flow.shape[:2],
                    "Sintel occlusions must be a native-size single-channel uint8 PNG")
            visible = occlusions == 0
    require(bool(valid.any()), "At least one finite valid original GT pixel is required")
    arrays = dict(flow=np.ascontiguousarray(flow, dtype=np.float32),
                  valid=np.ascontiguousarray(valid, dtype=np.bool_))
    if visible is not None:
        arrays["visible"] = np.ascontiguousarray(visible, dtype=np.bool_)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        np.savez(stream, **arrays)
    return dict(format=format, output=str(output_path), native_hw=list(flow.shape[:2]),
                flow_dtype="float32", valid_pixels=int(valid.sum()),
                visible_pixels=int(visible.sum()) if visible is not None else None,
                visibility_supplied=visible is not None, flow_source_sha256=sha256(flow_path),
                visibility_source_sha256=sha256(visibility_path) if visibility_path is not None else None,
                output_sha256=sha256(output_path))
