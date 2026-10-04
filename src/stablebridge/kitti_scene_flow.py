"""KITTI Scene Flow 2015 input binding and official-style task metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class KittiScenePair:
    scene_id: str
    stereo_left: Path
    stereo_right: Path
    stereo_disparity: Path
    flow_source: Path
    flow_target: Path
    flow_ground_truth: Path

    def to_record(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class KittiMetric:
    valid_pixels: int
    endpoint_error_sum: float
    outlier_pixels: int
    mean_endpoint_error: float
    outlier_percent: float


def discover_training_pairs(root: str | Path, *, require_count: int | None = 200) -> list[KittiScenePair]:
    """Bind D1 and Fl inputs/labels from an extracted KITTI training directory."""
    root = Path(root)
    training = root / "training" if (root / "training").is_dir() else root
    scene_ids = sorted(path.name.removesuffix("_10.png") for path in (training / "image_2").glob("*_10.png"))
    pairs: list[KittiScenePair] = []
    for scene_id in scene_ids:
        pair = KittiScenePair(
            scene_id=scene_id,
            stereo_left=training / "image_2" / f"{scene_id}_10.png",
            stereo_right=training / "image_3" / f"{scene_id}_10.png",
            stereo_disparity=training / "disp_occ_0" / f"{scene_id}_10.png",
            flow_source=training / "image_2" / f"{scene_id}_10.png",
            flow_target=training / "image_2" / f"{scene_id}_11.png",
            flow_ground_truth=training / "flow_occ" / f"{scene_id}_10.png",
        )
        missing = [path for key, path in asdict(pair).items() if key != "scene_id" and not path.is_file()]
        if missing:
            raise FileNotFoundError(f"KITTI scene {scene_id} is incomplete: {missing}")
        pairs.append(pair)
    if require_count is not None and len(pairs) != require_count:
        raise ValueError(f"expected {require_count} KITTI training scenes, found {len(pairs)}")
    return pairs


def read_disparity(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode KITTI uint16 disparity PNG (1/256 pixel; zero is invalid)."""
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise FileNotFoundError(path)
    if encoded.dtype != np.uint16 or encoded.ndim != 2:
        raise ValueError(f"expected uint16 single-channel disparity PNG: {path}")
    valid = encoded > 0
    disparity = encoded.astype(np.float32) / 256.0
    return disparity, valid


def read_flow(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode KITTI uint16 BGR flow PNG into (u, v) pixels and valid mask."""
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise FileNotFoundError(path)
    if encoded.dtype != np.uint16 or encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError(f"expected uint16 three-channel flow PNG: {path}")
    valid = encoded[..., 0] > 0
    u = (encoded[..., 2].astype(np.float32) - 32768.0) / 64.0
    v = (encoded[..., 1].astype(np.float32) - 32768.0) / 64.0
    return np.stack((u, v), axis=-1), valid


def official_outlier_mask(error: np.ndarray, ground_truth_magnitude: np.ndarray) -> np.ndarray:
    """KITTI bad-pixel rule: absolute error >3 px and relative error >5%."""
    error = np.asarray(error, dtype=np.float64)
    magnitude = np.asarray(ground_truth_magnitude, dtype=np.float64)
    relative = np.divide(
        error,
        magnitude,
        out=np.full_like(error, np.inf),
        where=magnitude > 0,
    )
    return (error > 3.0) & (relative > 0.05)


def evaluate_disparity(prediction: np.ndarray, ground_truth: np.ndarray,
                       valid: np.ndarray) -> KittiMetric:
    prediction = np.asarray(prediction, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if prediction.shape != ground_truth.shape or valid.shape != ground_truth.shape:
        raise ValueError("disparity prediction, ground truth, and valid mask must share shape")
    if not np.isfinite(ground_truth[valid]).all():
        raise ValueError("non-finite disparity ground truth on a valid pixel")
    if not np.isfinite(prediction[valid]).all():
        raise ValueError("non-finite disparity prediction on a valid pixel")
    if not valid.any():
        raise ValueError("no valid disparity pixels")
    error = np.abs(prediction - ground_truth)
    outlier = official_outlier_mask(error, np.abs(ground_truth))
    valid_error = error[valid]
    outlier_pixels = int(outlier[valid].sum())
    valid_pixels = int(valid.sum())
    return KittiMetric(
        valid_pixels=valid_pixels,
        endpoint_error_sum=float(valid_error.sum(dtype=np.float64)),
        outlier_pixels=outlier_pixels,
        mean_endpoint_error=float(valid_error.mean()),
        outlier_percent=float(100.0 * outlier_pixels / valid_pixels),
    )


def evaluate_flow(prediction: np.ndarray, ground_truth: np.ndarray,
                  valid: np.ndarray) -> KittiMetric:
    prediction = np.asarray(prediction, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if prediction.shape != ground_truth.shape or prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError("flow prediction and ground truth must have shape HxWx2")
    if valid.shape != ground_truth.shape[:2]:
        raise ValueError("flow valid mask must have shape HxW")
    if not np.isfinite(ground_truth[valid]).all():
        raise ValueError("non-finite flow ground truth on a valid pixel")
    if not np.isfinite(prediction[valid]).all():
        raise ValueError("non-finite flow prediction on a valid pixel")
    if not valid.any():
        raise ValueError("no valid flow pixels")
    error = np.linalg.norm(prediction - ground_truth, axis=-1)
    magnitude = np.linalg.norm(ground_truth, axis=-1)
    outlier = official_outlier_mask(error, magnitude)
    valid_error = error[valid]
    outlier_pixels = int(outlier[valid].sum())
    valid_pixels = int(valid.sum())
    return KittiMetric(
        valid_pixels=valid_pixels,
        endpoint_error_sum=float(valid_error.sum(dtype=np.float64)),
        outlier_pixels=outlier_pixels,
        mean_endpoint_error=float(valid_error.mean()),
        outlier_percent=float(100.0 * outlier_pixels / valid_pixels),
    )
