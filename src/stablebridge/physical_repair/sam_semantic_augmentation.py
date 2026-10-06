"""U²Flow-style semantic object augmentation from traced SAM regions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SamSemanticAugmentationPolicyV1:
    """Configuration matching U²Flow's delayed SAM object-copy objective.

    U²Flow stores separately generated key-object masks.  StableBridge keeps
    the smaller, lineage-traced full segmentation, so candidate objects are
    the visible regions that satisfy the public key-object size/density gates.
    """

    enabled: bool = False
    weight: float = 0.05
    activation_epoch: int = 10
    cache_size: int = 100
    objects_per_batch: int = 3
    minimum_region_fraction: float = 0.005
    minimum_box_height: int = 50
    maximum_box_height: int = 200
    minimum_box_width: int = 50
    maximum_box_width: int = 300
    minimum_box_fill_fraction: float = 0.5
    minimum_motion_scale: float = 0.8
    maximum_motion_scale: float = 1.5
    reverse_motion_probability: float = 0.5
    horizontal_flip_probability: float = 0.5
    fallback_to_full_segmentation: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("SAM semantic augmentation enabled must be boolean")
        if not isinstance(self.fallback_to_full_segmentation, bool):
            raise ValueError("SAM full-segmentation fallback must be boolean")
        for name in (
            "activation_epoch", "cache_size", "objects_per_batch",
            "minimum_box_height", "maximum_box_height",
            "minimum_box_width", "maximum_box_width",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.cache_size < 1 or self.objects_per_batch < 1:
            raise ValueError("SAM semantic cache and object counts must be positive")
        if (
            self.minimum_box_height < 1
            or self.minimum_box_width < 1
            or self.maximum_box_height < self.minimum_box_height
            or self.maximum_box_width < self.minimum_box_width
        ):
            raise ValueError("SAM semantic object box bounds are invalid")
        for name in (
            "weight", "minimum_region_fraction", "minimum_box_fill_fraction",
            "minimum_motion_scale", "maximum_motion_scale",
            "reverse_motion_probability", "horizontal_flip_probability",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.enabled and self.weight <= 0.0:
            raise ValueError("enabled SAM semantic augmentation needs a positive weight")
        if not 0.0 < self.minimum_region_fraction <= 1.0:
            raise ValueError("minimum SAM region fraction must lie in (0,1]")
        if not 0.0 <= self.minimum_box_fill_fraction <= 1.0:
            raise ValueError("minimum SAM box fill fraction must lie in [0,1]")
        if self.minimum_motion_scale <= 0.0:
            raise ValueError("minimum SAM motion scale must be positive")
        if self.maximum_motion_scale < self.minimum_motion_scale:
            raise ValueError("SAM motion scale bounds are invalid")
        for name in (
            "reverse_motion_probability", "horizontal_flip_probability",
        ):
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"{name} must lie in [0,1]")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None,
    ) -> "SamSemanticAugmentationPolicyV1":
        if value is None:
            return cls()
        normalized = dict(value)
        normalized.setdefault("fallback_to_full_segmentation", False)
        if set(normalized) != set(asdict(cls())):
            raise ValueError("SAM semantic augmentation policy fields drift")
        return cls(**normalized)


@dataclass(frozen=True)
class SamSemanticAugmentationBatchV1:
    frames: torch.Tensor
    target_flow: torch.Tensor
    valid: torch.Tensor
    object_count: int
    object_pixel_count: int


def select_sam_object_masks_v1(
    segment_ids: torch.Tensor,
    *,
    policy: SamSemanticAugmentationPolicyV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly select one visible key-object candidate per batch item."""

    if segment_ids.ndim == 3:
        segment_ids = segment_ids[:, None]
    if segment_ids.ndim != 4 or segment_ids.shape[1] != 1:
        raise ValueError("SAM segment ids must have shape B1HW or BHW")
    if segment_ids.dtype not in {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    }:
        raise TypeError("SAM segment ids must use an integer dtype")
    batch, _, height, width = segment_ids.shape
    selected = torch.zeros_like(segment_ids, dtype=torch.bool)
    present = torch.zeros(batch, device=segment_ids.device, dtype=torch.bool)
    minimum_area = policy.minimum_region_fraction * float(height * width)
    for batch_index in range(batch):
        labels = segment_ids[batch_index, 0]
        candidates: list[int] = []
        for raw_label in torch.unique(labels).tolist():
            label = int(raw_label)
            if label <= 0:
                continue
            region = labels == label
            area = int(region.sum().item())
            if area < minimum_area:
                continue
            coordinates = torch.nonzero(region, as_tuple=False)
            box_height = int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1)
            box_width = int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1)
            if not (
                policy.minimum_box_height <= box_height <= policy.maximum_box_height
                and policy.minimum_box_width <= box_width <= policy.maximum_box_width
            ):
                continue
            if area / float(box_height * box_width) < policy.minimum_box_fill_fraction:
                continue
            candidates.append(label)
        if candidates:
            chosen = candidates[
                int(torch.randint(len(candidates), (), device=segment_ids.device).item())
            ]
            selected[batch_index, 0] = labels == chosen
            present[batch_index] = True
    return selected, present


class SamSemanticObjectCacheV1:
    """Fixed-size CPU cache with compact uint8 image/mask storage."""

    def __init__(self, cache_size: int) -> None:
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("SAM semantic cache size must be positive")
        self.cache_size = cache_size
        self.count = 0
        self._masks: torch.Tensor | None = None
        self._images: torch.Tensor | None = None
        self._motions: torch.Tensor | None = None

    @property
    def ready(self) -> bool:
        return self.count == self.cache_size

    def _initialize(self, image_hw: tuple[int, int]) -> None:
        self._masks = torch.zeros(
            self.cache_size, 1, *image_hw, dtype=torch.uint8,
        )
        self._images = torch.zeros(
            self.cache_size, 3, *image_hw, dtype=torch.uint8,
        )
        self._motions = torch.zeros(self.cache_size, 2, dtype=torch.float32)

    def push(
        self,
        masks: torch.Tensor,
        images: torch.Tensor,
        motions: torch.Tensor,
    ) -> int:
        if masks.ndim != 4 or masks.shape[1] != 1 or masks.dtype is not torch.bool:
            raise ValueError("cached SAM masks must be boolean B1HW")
        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("cached SAM object images must be floating B3HW")
        if motions.shape != (masks.shape[0], 2) or not motions.is_floating_point():
            raise ValueError("cached SAM object motions must be floating B2")
        if images.shape[0] != masks.shape[0] or images.shape[-2:] != masks.shape[-2:]:
            raise ValueError("cached SAM object tensors must share batch/geometry")
        if not bool(torch.isfinite(images).all()) or not bool(torch.isfinite(motions).all()):
            raise ValueError("cached SAM object tensors must be finite")
        if masks.shape[0] == 0:
            return 0
        if self._masks is None:
            self._initialize(tuple(int(item) for item in masks.shape[-2:]))
        assert self._masks is not None and self._images is not None and self._motions is not None
        if tuple(self._masks.shape[-2:]) != tuple(masks.shape[-2:]):
            raise ValueError("SAM semantic cache geometry drift")
        compact_masks = masks.detach().to(device="cpu", dtype=torch.uint8)
        compact_images = images.detach().to(device="cpu").clamp(0.0, 255.0).round().to(torch.uint8)
        compact_motions = motions.detach().to(device="cpu", dtype=torch.float32)
        pushed = int(masks.shape[0])
        for index in range(pushed):
            if self.count < self.cache_size:
                destination = self.count
                self.count += 1
            else:
                destination = int(torch.randint(self.cache_size, ()).item())
            self._masks[destination].copy_(compact_masks[index])
            self._images[destination].copy_(compact_images[index])
            self._motions[destination].copy_(compact_motions[index])
        return pushed

    def sample(
        self,
        batch_size: int,
        *,
        objects_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        policy: SamSemanticAugmentationPolicyV1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total = batch_size * objects_per_batch
        if not self.ready or total > self.count:
            raise ValueError("SAM semantic cache is not ready for this sample")
        assert self._masks is not None and self._images is not None and self._motions is not None
        indices = torch.randperm(self.count)[:total]
        masks = self._masks[indices].to(device=device, dtype=dtype)
        images = self._images[indices].to(device=device, dtype=dtype)
        motions = self._motions[indices].to(device=device, dtype=dtype)
        scale = torch.empty(total, device=device, dtype=dtype).uniform_(
            policy.minimum_motion_scale, policy.maximum_motion_scale,
        )
        reverse = torch.rand(total, device=device) < policy.reverse_motion_probability
        scale = torch.where(reverse, -scale, scale)
        motions = motions * scale[:, None]
        flip = torch.rand(total, device=device) < policy.horizontal_flip_probability
        if bool(flip.any()):
            masks[flip] = masks[flip].flip(-1)
            images[flip] = images[flip].flip(-1)
            motions[flip, 0] *= -1.0
        shape = (objects_per_batch, batch_size)
        return (
            masks.reshape(*shape, 1, *masks.shape[-2:]),
            images.reshape(*shape, 3, *images.shape[-2:]),
            motions.reshape(*shape, 2),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "cache_size": self.cache_size,
            "count": self.count,
            "masks": self._masks,
            "images": self._images,
            "motions": self._motions,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if set(value) != {"cache_size", "count", "masks", "images", "motions"}:
            raise ValueError("SAM semantic cache checkpoint fields drift")
        if int(value["cache_size"]) != self.cache_size:
            raise ValueError("SAM semantic cache checkpoint size drift")
        count = int(value["count"])
        masks, images, motions = value["masks"], value["images"], value["motions"]
        if count == 0:
            if any(item is not None for item in (masks, images, motions)):
                raise ValueError("empty SAM semantic cache has tensor payload")
            self.count = 0
            return
        if not all(isinstance(item, torch.Tensor) for item in (masks, images, motions)):
            raise ValueError("SAM semantic cache checkpoint tensors are invalid")
        assert isinstance(masks, torch.Tensor)
        assert isinstance(images, torch.Tensor)
        assert isinstance(motions, torch.Tensor)
        if (
            masks.shape[0] != self.cache_size or images.shape[0] != self.cache_size
            or motions.shape != (self.cache_size, 2)
            or masks.shape[1] != 1 or images.shape[1] != 3
            or masks.shape[-2:] != images.shape[-2:]
            or masks.dtype is not torch.uint8 or images.dtype is not torch.uint8
        ):
            raise ValueError("SAM semantic cache checkpoint geometry/dtype drift")
        if not 0 < count <= self.cache_size:
            raise ValueError("SAM semantic cache checkpoint count is invalid")
        self._masks = masks.detach().to(device="cpu").clone()
        self._images = images.detach().to(device="cpu").clone()
        self._motions = motions.detach().to(device="cpu", dtype=torch.float32).clone()
        self.count = count


def _translate_v1(
    value: torch.Tensor,
    motion: torch.Tensor,
    *,
    padding_mode: str,
) -> torch.Tensor:
    batch, _, height, width = value.shape
    if motion.shape != (batch, 2):
        raise ValueError("translation motion must have shape B2")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=value.device, dtype=value.dtype),
        torch.arange(width, device=value.device, dtype=value.dtype),
        indexing="ij",
    )
    source_x = xx[None] - motion[:, 0, None, None]
    source_y = yy[None] - motion[:, 1, None, None]
    normalized_x = 2.0 * source_x / max(width - 1, 1) - 1.0
    normalized_y = 2.0 * source_y / max(height - 1, 1) - 1.0
    grid = torch.stack((normalized_x, normalized_y), dim=-1)
    return F.grid_sample(
        value, grid, mode="bilinear", padding_mode=padding_mode,
        align_corners=True,
    )


def compose_sam_semantic_augmentation_v1(
    frames: torch.Tensor,
    teacher_flow: torch.Tensor,
    valid: torch.Tensor,
    cached_masks: torch.Tensor,
    cached_images: torch.Tensor,
    cached_motions: torch.Tensor,
) -> SamSemanticAugmentationBatchV1:
    """Paste cached SAM objects and materialize their detached flow target."""

    if frames.ndim != 5 or frames.shape[1:3] != (2, 3):
        raise ValueError("semantic target frames must have shape B23HW")
    if teacher_flow.shape != (frames.shape[0], 2, *frames.shape[-2:]):
        raise ValueError("semantic teacher flow must match target frames")
    if valid.ndim == 3:
        valid = valid[:, None]
    if valid.shape != teacher_flow[:, :1].shape or valid.dtype is not torch.bool:
        raise ValueError("semantic validity must be boolean B1HW")
    if cached_masks.ndim != 5 or cached_masks.shape[1] != frames.shape[0]:
        raise ValueError("cached semantic masks must have shape RB1HW")
    if cached_images.shape != (
        cached_masks.shape[0], frames.shape[0], 3, *frames.shape[-2:],
    ):
        raise ValueError("cached semantic images must have shape RB3HW")
    if cached_motions.shape != (cached_masks.shape[0], frames.shape[0], 2):
        raise ValueError("cached semantic motions must have shape RB2")
    first, second = frames[:, 0].clone(), frames[:, 1].clone()
    target = teacher_flow.detach().clone()
    support = valid.clone()
    object_pixels = 0
    for round_index in range(cached_masks.shape[0]):
        mask = cached_masks[round_index].clamp(0.0, 1.0)
        image = cached_images[round_index]
        motion = cached_motions[round_index]
        first = mask * image + (1.0 - mask) * first
        moved_mask = _translate_v1(mask, motion, padding_mode="zeros").clamp(0.0, 1.0)
        moved_image = _translate_v1(image, motion, padding_mode="border")
        second = moved_mask * moved_image + (1.0 - moved_mask) * second
        target = mask * motion[:, :, None, None] + (1.0 - mask) * target
        hard_mask = mask >= 0.5
        support |= hard_mask
        object_pixels += int(hard_mask.sum().item())
    return SamSemanticAugmentationBatchV1(
        frames=torch.stack((first, second), dim=1),
        target_flow=target,
        valid=support,
        object_count=int(cached_masks.shape[0] * cached_masks.shape[1]),
        object_pixel_count=object_pixels,
    )


def sam_semantic_sequence_loss_v1(
    predictions: Sequence[torch.Tensor],
    target_flow: torch.Tensor,
    valid: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Public U²Flow/UnSAMFlow SmoothL1 sequence objective."""

    sequence = tuple(predictions)
    if not sequence:
        raise ValueError("SAM semantic sequence loss needs predictions")
    if valid.ndim == 3:
        valid = valid[:, None]
    if valid.shape != target_flow[:, :1].shape or valid.dtype is not torch.bool:
        raise ValueError("SAM semantic valid support must be boolean B1HW")
    if not bool(valid.any()):
        raise ValueError("SAM semantic loss has empty valid support")
    total = target_flow.sum() * 0.0
    expanded = valid.expand_as(target_flow)
    denominator = valid.sum().to(dtype=target_flow.dtype).clamp_min(1.0)
    count = len(sequence)
    for index, prediction in enumerate(sequence):
        if prediction.shape != target_flow.shape:
            raise ValueError("SAM semantic prediction/target geometry drift")
        difference = (prediction - target_flow.detach()).abs()
        component = F.smooth_l1_loss(
            difference, torch.zeros_like(difference), reduction="none",
        )
        total = total + gamma ** (count - index - 1) * component[expanded].sum() / denominator
    return total


__all__ = [
    "SamSemanticAugmentationBatchV1",
    "SamSemanticAugmentationPolicyV1",
    "SamSemanticObjectCacheV1",
    "compose_sam_semantic_augmentation_v1",
    "sam_semantic_sequence_loss_v1",
    "select_sam_object_masks_v1",
]
