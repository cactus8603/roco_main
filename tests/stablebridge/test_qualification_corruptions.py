from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from stablebridge.physical_repair.qualification_corruptions import (
    CONTROLLED_CORRUPTION_STRATA_V1,
    materialize_controlled_corruption_v1,
)


def pair() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:384, :512]
    value = (
        110 + 35 * np.sin(xx / 3.7) + 29 * np.cos(yy / 5.3)
        + 18 * np.sin((xx + yy) / 8.1)
        + 12 * ((xx // 11 + yy // 13) % 2)
    )
    first = np.stack(
        (value, np.roll(value, 5, 1), np.roll(value, 7, 0)), axis=2,
    ).clip(0, 255).astype(np.uint8)
    return first, np.roll(first, 2, axis=1).copy()


@pytest.mark.parametrize("stratum", CONTROLLED_CORRUPTION_STRATA_V1)
def test_corruptions_are_deterministic_local_and_outcome_blind(stratum):
    first, second = pair()
    before_first = first.copy()
    before_second = second.copy()
    one = materialize_controlled_corruption_v1(
        first=first, second=second, component_id="scene-a",
        mechanism_stratum=stratum,
    )
    two = materialize_controlled_corruption_v1(
        first=first, second=second, component_id="scene-a",
        mechanism_stratum=stratum,
    )
    assert np.array_equal(first, before_first)
    assert np.array_equal(second, before_second)
    assert np.array_equal(one.first, two.first)
    assert np.array_equal(one.second, second)
    assert np.array_equal(one.supports[0], np.any(one.first != first, axis=2))
    assert not one.supports[1].any()
    assert np.array_equal(one.first[~one.supports[0]], first[~one.supports[0]])
    assert one.receipt == two.receipt
    record = one.receipt.as_dict()
    assert record["ground_truth_read"] is False
    assert record["native_flow_read"] is False
    assert record["matcher_output_read"] is False
    assert record["action_outcome_read"] is False


def test_component_identity_changes_seeded_noise_and_receipt():
    first, second = pair()
    one = materialize_controlled_corruption_v1(
        first=first, second=second, component_id="scene-a",
        mechanism_stratum="paired_additive_wiener3",
    )
    two = materialize_controlled_corruption_v1(
        first=first, second=second, component_id="scene-b",
        mechanism_stratum="paired_additive_wiener3",
    )
    assert not np.array_equal(one.first, two.first)
    assert one.receipt.receipt_sha256 != two.receipt.receipt_sha256


def test_invalid_geometry_and_receipt_drift_fail_closed():
    first, second = pair()
    with pytest.raises(ValueError, match=">=256px"):
        materialize_controlled_corruption_v1(
            first=first[:128], second=second[:128], component_id="scene-a",
            mechanism_stratum="common_disk",
        )
    result = materialize_controlled_corruption_v1(
        first=first, second=second, component_id="scene-a",
        mechanism_stratum="common_disk",
    )
    with pytest.raises(ValueError, match="receipt hash drift"):
        replace(result.receipt, receipt_sha256=hashlib.sha256(b"bad").hexdigest())
