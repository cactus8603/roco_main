from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stablebridge.physical_repair.sam_semantic_smoothness import (
    SamHomographySmoothnessPolicyV1,
    compose_sam_full_segmentation_v1,
    sam_homography_smoothness_loss_v1,
)


def _policy() -> SamHomographySmoothnessPolicyV1:
    return SamHomographySmoothnessPolicyV1(
        enabled=True,
        weight=0.1,
        maximum_regions=6,
        uncertainty_variance_threshold=2.0,
        minimum_reliable_points=4,
        minimum_reliable_fraction=0.2,
        ransac_reprojection_threshold=0.1,
        minimum_inlier_fraction=0.5,
        per_region_loss_cap=0.5,
    )


def test_full_segmentation_uses_smallest_mask_priority_and_background_zero():
    large = np.zeros((5, 6), dtype=bool)
    large[1:5, 1:6] = True
    small = np.zeros((5, 6), dtype=bool)
    small[2:4, 3:5] = True
    full = np.ones((5, 6), dtype=bool)
    result = compose_sam_full_segmentation_v1([
        {"segmentation": small, "area": int(small.sum())},
        {"segmentation": full, "area": int(full.sum())},
        {"segmentation": large, "area": int(large.sum())},
    ], (5, 6))
    assert result.dtype == np.uint16
    assert result[0, 0] == 0
    assert result[1, 1] != 0
    assert result[2, 3] not in {0, result[1, 1]}


def test_homography_loss_fits_from_reliable_pixels_and_backpropagates():
    flow = torch.zeros(1, 2, 7, 8, requires_grad=True)
    with torch.no_grad():
        flow[:, 0] = 1.25
        flow[:, 1] = -0.5
        flow[:, :, 3, 4] += 1.0
    segments = torch.ones(1, 1, 7, 8, dtype=torch.int64)
    alpha = torch.zeros(1, 1, 7, 8)
    alpha[:, :, 3, 4] = np.log(10.0)
    result = sam_homography_smoothness_loss_v1(
        flow, segments, alpha, policy=_policy(),
    )
    assert result.candidate_regions == 1
    assert result.fitted_regions == 1
    assert float(result.loss) > 0.0
    result.loss.backward()
    assert flow.grad is not None
    assert float(flow.grad.abs().sum()) > 0.0


def test_disabled_homography_is_an_exact_zero():
    flow = torch.randn(2, 2, 3, 4, requires_grad=True)
    result = sam_homography_smoothness_loss_v1(
        flow,
        torch.zeros(2, 1, 3, 4, dtype=torch.int64),
        torch.zeros(2, 1, 3, 4),
        policy=SamHomographySmoothnessPolicyV1(),
    )
    assert float(result.loss) == 0.0
    assert result.candidate_regions == result.fitted_regions == 0
