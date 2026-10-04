from __future__ import annotations

import numpy as np
import pytest

from stablebridge.physical_repair.local_wiener_successor import (
    build_local_wiener_successor_v2,
)
from stablebridge.physical_repair.noise_mechanism_evaluators import (
    evaluate_noise_mechanism_v1,
)
from stablebridge.physical_repair.qualification_matrix import (
    build_uncapped_qualification_matrix,
)


def _impulse_case():
    first = np.full((256, 256, 3), 128, dtype=np.uint8)
    second = first.copy()
    for y0, x0 in ((16, 16), (16, 144), (144, 16), (144, 144)):
        for index in range(12):
            first[y0 + (index // 4) * 5, x0 + (index % 4) * 5] = (
                0 if index % 2 == 0 else 255
            )
    return first, second, np.zeros((256, 256, 2), dtype=np.float32)


def _wiener_case():
    rng = np.random.default_rng(251)
    first = np.full((256, 256, 3), 128, dtype=np.uint8)
    second = first.copy()
    first[64:96, 64:128] = np.clip(
        np.rint(128 + rng.normal(0, 25, (32, 64, 3))), 16, 240,
    ).astype(np.uint8)
    return first, second, np.zeros((256, 256, 2), dtype=np.float32)


def test_impulse_mechanism_matches_frozen_anchor_and_reduces_signal():
    first, second, flow = _impulse_case()
    result = build_uncapped_qualification_matrix(first, second, flow)
    arm = result.arms[0]
    receipt = evaluate_noise_mechanism_v1(
        action_id=arm.qualification_action_id,
        physical_pair_sha256=result.receipt["physical_pair_sha256"],
        before_pair=(first, second),
        after_pair=(arm.first_rgb, arm.second_rgb),
        endpoint_supports=(arm.first_support, arm.second_support),
    )
    row = receipt.endpoint_records[0]
    assert row["active_relative_reduction"] > 0.05
    assert row["specificity_difference"] > 0.05
    assert row["outside_support_byte_identity"] is True


def test_wiener_mechanism_reduces_high_band_excess_with_zero_inactive_change():
    first, second, flow = _wiener_case()
    successor = build_local_wiener_successor_v2(first, second, flow)
    receipt = evaluate_noise_mechanism_v1(
        action_id="paired_additive_wiener3.local_tile_sigma_v2",
        physical_pair_sha256="a" * 64,
        before_pair=(first, second),
        after_pair=(successor.first_rgb, successor.second_rgb),
        endpoint_supports=(successor.first_support, successor.second_support),
    )
    row = receipt.endpoint_records[0]
    assert row["active_relative_reduction"] > 0.05
    assert row["negative_control_relative_reduction"] == pytest.approx(0.0)
    assert row["specificity_difference"] > 0.05
    assert row["outside_support_byte_identity"] is True


def test_mechanism_evaluator_fails_closed_on_escape_and_partial_wiener_tile():
    first, second, flow = _wiener_case()
    successor = build_local_wiener_successor_v2(first, second, flow)
    escaped = successor.first_rgb.copy()
    escaped[0, 0] = 1
    with pytest.raises(RuntimeError, match="escaped"):
        evaluate_noise_mechanism_v1(
            action_id="paired_additive_wiener3.local_tile_sigma_v2",
            physical_pair_sha256="a" * 64,
            before_pair=(first, second),
            after_pair=(escaped, second),
            endpoint_supports=(successor.first_support, successor.second_support),
        )
    partial = successor.first_support.copy()
    partial[64, 64] = False
    with pytest.raises(RuntimeError, match="whole-tile"):
        evaluate_noise_mechanism_v1(
            action_id="paired_additive_wiener3.local_tile_sigma_v2",
            physical_pair_sha256="a" * 64,
            before_pair=(first, second),
            after_pair=(successor.first_rgb, successor.second_rgb),
            endpoint_supports=(partial, successor.second_support),
        )
