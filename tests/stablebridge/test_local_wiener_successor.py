from __future__ import annotations

import numpy as np

from stablebridge.physical_repair.local_wiener_successor import (
    LOCAL_WIENER_ACTION_ID_V2,
    build_local_wiener_successor_v2,
)


def _localized_additive_case():
    rng = np.random.default_rng(251)
    first = np.full((256, 256, 3), 128, dtype=np.uint8)
    second = first.copy()
    first[64:96, 64:128] = np.clip(
        np.rint(128 + rng.normal(0, 25, (32, 64, 3))), 16, 240,
    ).astype(np.uint8)
    flow = np.zeros((256, 256, 2), dtype=np.float32)
    return first, second, flow


def test_local_tile_sigma_successor_repairs_the_v1_identity_case():
    first, second, flow = _localized_additive_case()
    result = build_local_wiener_successor_v2(first, second, flow)
    assert result.status == "EXECUTED_LOCAL_TILE_SIGMA_V2"
    assert result.exact_endpoint == "first"
    assert result.receipt["action_id"] == LOCAL_WIENER_ACTION_ID_V2
    assert result.first_support.sum() == 2 * 32 * 32
    changed = np.any(result.first_rgb != first, axis=2)
    assert changed.any()
    assert not np.any(changed & ~result.first_support)
    assert np.array_equal(result.first_rgb[~result.first_support], first[~result.first_support])
    assert np.array_equal(result.second_rgb, second)


def test_successor_is_deterministic_and_abstains_without_additive_authority():
    first, second, flow = _localized_additive_case()
    one = build_local_wiener_successor_v2(first, second, flow)
    two = build_local_wiener_successor_v2(first, second, flow)
    assert one.receipt == two.receipt
    assert np.array_equal(one.first_rgb, two.first_rgb)

    clean = np.full_like(first, 128)
    abstained = build_local_wiener_successor_v2(clean, clean.copy(), flow)
    assert abstained.status.startswith("TYPED_MISSING")
    assert abstained.exact_endpoint is None
    assert np.array_equal(abstained.first_rgb, clean)
    assert np.array_equal(abstained.second_rgb, clean)
