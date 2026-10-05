"""SAM full-segmentation conversion and U²Flow-style homography loss."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch


SAM_FULL_SEGMENTATION_SCHEMA_V1 = "stablebridge-sam-full-segmentation/v1"


@dataclass(frozen=True)
class SamHomographySmoothnessPolicyV1:
    """Conservative subset of UnSAMFlow/U²Flow's SAM homography objective."""

    enabled: bool = False
    weight: float = 0.1
    maximum_regions: int = 6
    uncertainty_variance_threshold: float = 2.0
    minimum_reliable_points: int = 4
    minimum_reliable_fraction: float = 0.2
    ransac_reprojection_threshold: float = 3.0
    minimum_inlier_fraction: float = 0.5
    per_region_loss_cap: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("SAM homography enabled must be boolean")
        for name in ("maximum_regions", "minimum_reliable_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "weight", "uncertainty_variance_threshold",
            "minimum_reliable_fraction", "ransac_reprojection_threshold",
            "minimum_inlier_fraction", "per_region_loss_cap",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.enabled and self.weight <= 0.0:
            raise ValueError("enabled SAM homography loss needs a positive weight")
        if self.uncertainty_variance_threshold <= 0.0:
            raise ValueError("uncertainty variance threshold must be positive")
        if not 0.0 <= self.minimum_reliable_fraction <= 1.0:
            raise ValueError("minimum reliable fraction must lie in [0,1]")
        if not 0.0 <= self.minimum_inlier_fraction <= 1.0:
            raise ValueError("minimum inlier fraction must lie in [0,1]")
        if self.ransac_reprojection_threshold <= 0.0 or self.per_region_loss_cap <= 0.0:
            raise ValueError("RANSAC threshold and region loss cap must be positive")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None,
    ) -> "SamHomographySmoothnessPolicyV1":
        if value is None:
            return cls()
        if set(value) != set(asdict(cls())):
            raise ValueError("SAM homography smoothness policy fields drift")
        return cls(**dict(value))


@dataclass(frozen=True)
class SamHomographySmoothnessResultV1:
    loss: torch.Tensor
    candidate_regions: int
    fitted_regions: int


def compose_sam_full_segmentation_v1(
    masks: Sequence[Mapping[str, Any]],
    image_hw: tuple[int, int],
    *,
    maximum_masks: int = 255,
) -> np.ndarray:
    """Compose overlapping SAM masks with UnSAMFlow's smallest-mask priority.

    Label zero is background.  Full-frame masks are discarded, masks are
    ordered from large to small, and smaller masks overwrite larger masks.
    The result is compactly relabelled and stored as uint16 PNG-compatible
    data (rather than silently wrapping when an image contains >255 regions).
    """

    height, width = image_hw
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if isinstance(maximum_masks, bool) or maximum_masks < 1 or maximum_masks > 65535:
        raise ValueError("maximum_masks must lie in [1,65535]")
    accepted: list[tuple[int, np.ndarray]] = []
    for item in masks:
        segmentation = np.asarray(item.get("segmentation"), dtype=bool)
        if segmentation.shape != (height, width):
            raise ValueError("SAM mask geometry drift")
        measured_area = int(segmentation.sum())
        declared_area = int(item.get("area", measured_area))
        if declared_area != measured_area:
            raise ValueError("SAM mask area metadata drift")
        if 0 < measured_area < height * width:
            accepted.append((measured_area, segmentation))
    accepted.sort(key=lambda item: item[0], reverse=True)
    accepted = accepted[:maximum_masks]
    labels = np.zeros((height, width), dtype=np.uint16)
    for label, (_area, segmentation) in enumerate(accepted, start=1):
        labels[segmentation] = label
    unique, inverse = np.unique(labels, return_inverse=True)
    if unique.size - 1 > maximum_masks:
        raise RuntimeError("SAM full segmentation exceeded configured label bound")
    return inverse.reshape(height, width).astype(np.uint16, copy=False)


def sam_homography_smoothness_loss_v1(
    flow: torch.Tensor,
    segment_ids: torch.Tensor,
    log_variance: torch.Tensor,
    *,
    policy: SamHomographySmoothnessPolicyV1,
) -> SamHomographySmoothnessResultV1:
    """Fit detached regional homographies and penalize differentiable flow.

    Region selection and reliable-point fitting follow U²Flow's public loss:
    rank non-background SAM regions by uncertain-pixel count, keep at most six
    by default, require 20% reliable support and a 50% RANSAC inlier rate.
    OpenCV estimates ``H`` from detached points; gradients flow only through
    the current flow field in the residual.
    """

    if flow.ndim != 4 or flow.shape[1] != 2 or not flow.is_floating_point():
        raise ValueError("flow must be floating N2HW")
    if segment_ids.ndim == 3:
        segment_ids = segment_ids[:, None]
    if segment_ids.shape != flow[:, :1].shape:
        raise ValueError("SAM segment ids must match the flow grid")
    if segment_ids.dtype not in {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    }:
        raise TypeError("SAM segment ids must use an integer dtype")
    if log_variance.shape != flow[:, :1].shape or not log_variance.is_floating_point():
        raise ValueError("uncertainty log variance must match the flow grid")
    if not bool(torch.isfinite(flow).all()) or not bool(torch.isfinite(log_variance).all()):
        raise ValueError("SAM homography inputs must be finite")
    if not policy.enabled:
        return SamHomographySmoothnessResultV1(flow.sum() * 0.0, 0, 0)

    batch, _, height, width = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    coordinates = torch.stack((xx, yy), dim=-1)
    variance = torch.exp(log_variance.detach().float().clamp(-30.0, 30.0))
    reliable = variance <= policy.uncertainty_variance_threshold
    loss = flow.sum() * 0.0
    candidate_regions = 0
    fitted_regions = 0

    for batch_index in range(batch):
        labels = segment_ids[batch_index, 0].to(dtype=torch.int64)
        uncertain_labels = labels[~reliable[batch_index, 0]]
        uncertain_labels = uncertain_labels[uncertain_labels > 0]
        if uncertain_labels.numel() == 0:
            continue
        counts = torch.bincount(uncertain_labels)
        order = torch.argsort(counts, descending=True)
        selected = [
            int(item) for item in order.tolist()
            if item > 0 and int(counts[item]) > 0
        ][: policy.maximum_regions]
        candidate_regions += len(selected)
        displaced = coordinates + flow[batch_index].permute(1, 2, 0)
        for label in selected:
            region = labels == label
            region_reliable = reliable[batch_index, 0, region]
            reliable_count = int(region_reliable.sum().item())
            region_count = int(region.sum().item())
            if (
                reliable_count < policy.minimum_reliable_points
                or reliable_count / max(region_count, 1) < policy.minimum_reliable_fraction
            ):
                continue
            source = coordinates[region]
            target = displaced[region]
            fit_source = source[region_reliable].detach().float().cpu().numpy()
            fit_target = target[region_reliable].detach().float().cpu().numpy()
            homography, inliers = cv2.findHomography(
                fit_source,
                fit_target,
                cv2.RANSAC,
                policy.ransac_reprojection_threshold,
            )
            if homography is None or inliers is None or not np.isfinite(homography).all():
                continue
            if float(inliers.mean()) < policy.minimum_inlier_fraction:
                continue
            matrix = torch.as_tensor(homography, device=flow.device, dtype=flow.dtype)
            homogeneous = torch.cat((source, torch.ones_like(source[:, :1])), dim=1)
            projected_h = homogeneous @ matrix.T
            denominator = projected_h[:, 2:3]
            safe = denominator.abs() > 1e-8
            safe_denominator = torch.where(
                safe,
                denominator,
                torch.where(
                    denominator < 0,
                    torch.full_like(denominator, -1e-8),
                    torch.full_like(denominator, 1e-8),
                ),
            )
            projected = projected_h[:, :2] / safe_denominator
            residual = torch.where(safe, projected - target, torch.zeros_like(target))
            per_region = residual.abs().sum() / float(height * width)
            loss = loss + torch.nan_to_num(
                per_region, nan=0.0, posinf=policy.per_region_loss_cap,
                neginf=0.0,
            ).clamp(0.0, policy.per_region_loss_cap)
            fitted_regions += 1
    return SamHomographySmoothnessResultV1(
        loss=loss / float(batch),
        candidate_regions=candidate_regions,
        fitted_regions=fitted_regions,
    )


__all__ = [
    "SAM_FULL_SEGMENTATION_SCHEMA_V1",
    "SamHomographySmoothnessPolicyV1",
    "SamHomographySmoothnessResultV1",
    "compose_sam_full_segmentation_v1",
    "sam_homography_smoothness_loss_v1",
]
