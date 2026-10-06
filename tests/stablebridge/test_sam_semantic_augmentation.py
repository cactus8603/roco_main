from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stablebridge.physical_repair.sam_semantic_augmentation import (
    SamSemanticAugmentationPolicyV1,
    SamSemanticObjectCacheV1,
    compose_sam_semantic_augmentation_v1,
    sam_semantic_sequence_loss_v1,
    select_sam_object_masks_v1,
)


def _policy(**overrides):
    value = {
        "enabled": True,
        "weight": 0.05,
        "activation_epoch": 1,
        "cache_size": 1,
        "objects_per_batch": 1,
        "minimum_region_fraction": 0.01,
        "minimum_box_height": 1,
        "maximum_box_height": 20,
        "minimum_box_width": 1,
        "maximum_box_width": 20,
        "minimum_box_fill_fraction": 0.5,
        "minimum_motion_scale": 1.0,
        "maximum_motion_scale": 1.0,
        "reverse_motion_probability": 0.0,
        "horizontal_flip_probability": 0.0,
    }
    value.update(overrides)
    return SamSemanticAugmentationPolicyV1(**value)


def test_selects_visible_sam_object_and_rejects_background():
    segments = torch.zeros(1, 1, 8, 10, dtype=torch.int64)
    segments[:, :, 2:6, 3:8] = 7
    masks, present = select_sam_object_masks_v1(segments, policy=_policy())
    assert present.tolist() == [True]
    assert masks.dtype is torch.bool
    assert int(masks.sum()) == 20


def test_cache_composition_sequence_loss_and_checkpoint_round_trip():
    policy = _policy()
    cache = SamSemanticObjectCacheV1(policy.cache_size)
    mask = torch.zeros(1, 1, 8, 10, dtype=torch.bool)
    mask[:, :, 2:6, 3:8] = True
    source = torch.full((1, 3, 8, 10), 128.0)
    motion = torch.tensor([[1.0, 0.0]])
    assert cache.push(mask, source, motion) == 1
    assert cache.ready
    cached = cache.sample(
        1, objects_per_batch=1, device=torch.device("cpu"),
        dtype=torch.float32, policy=policy,
    )
    frames = torch.zeros(1, 2, 3, 8, 10)
    teacher = torch.zeros(1, 2, 8, 10)
    valid = torch.ones(1, 1, 8, 10, dtype=torch.bool)
    augmented = compose_sam_semantic_augmentation_v1(
        frames, teacher, valid, *cached,
    )
    assert augmented.object_count == 1
    assert augmented.object_pixel_count == 20
    assert torch.allclose(augmented.target_flow[:, 0][mask[:, 0]], torch.ones(20))
    prediction = (augmented.target_flow + 0.5).requires_grad_()
    loss = sam_semantic_sequence_loss_v1(
        (prediction,), augmented.target_flow, augmented.valid, gamma=0.8,
    )
    assert float(loss) > 0.0
    loss.backward()
    assert prediction.grad is not None

    restored = SamSemanticObjectCacheV1(policy.cache_size)
    restored.load_state_dict(cache.state_dict())
    assert restored.ready
    assert restored.count == cache.count
