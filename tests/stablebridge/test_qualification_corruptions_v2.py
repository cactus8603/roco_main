from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from stablebridge.physical_repair.qualification_corruptions_v2 import (
    CONTROLLED_CORRUPTION_STRATA_V2,
    RECIPE_PARAMETERS_V2,
    materialize_controlled_corruption_v2,
)


def pair() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:384, :512]
    value = (
        112 + 39 * np.sin(xx / 3.7) + 31 * np.cos(yy / 5.3)
        + 21 * np.sin((xx + yy) / 8.1)
        + 13 * ((xx // 11 + yy // 13) % 2)
    )
    first = np.stack(
        (value, np.roll(value, 5, 1), np.roll(value, 7, 0)), axis=2,
    ).clip(0, 255).astype(np.uint8)
    return first, np.roll(first, 2, axis=1).copy()


def materialize(stratum: str, *, pair_id: str = "pair-001"):
    first, second = pair()
    return materialize_controlled_corruption_v2(
        first=first,
        second=second,
        component_id="scene-a",
        physical_pair_id=pair_id,
        mechanism_stratum=stratum,
    )


@pytest.mark.parametrize("stratum", CONTROLLED_CORRUPTION_STRATA_V2)
def test_v2_corruptions_are_deterministic_support_exact_and_outcome_blind(stratum):
    first, second = pair()
    before_first, before_second = first.copy(), second.copy()
    one = materialize_controlled_corruption_v2(
        first=first, second=second, component_id="scene-a",
        physical_pair_id="pair-001", mechanism_stratum=stratum,
    )
    two = materialize_controlled_corruption_v2(
        first=first, second=second, component_id="scene-a",
        physical_pair_id="pair-001", mechanism_stratum=stratum,
    )
    assert np.array_equal(first, before_first)
    assert np.array_equal(second, before_second)
    assert np.array_equal(one.first, two.first)
    assert np.array_equal(one.second, two.second)
    assert one.receipt == two.receipt
    for source, output, support in zip(
        (first, second), (one.first, one.second), one.supports,
    ):
        assert np.array_equal(support, np.any(output != source, axis=2))
        assert np.array_equal(output[~support], source[~support])
    record = one.receipt.as_dict()
    assert record["ground_truth_read"] is False
    assert record["native_flow_read"] is False
    assert record["matcher_output_read"] is False
    assert record["action_eligibility_read"] is False
    assert record["action_outcome_read"] is False
    assert record["consumer_firewall"][
        "only_corrupted_rgb_crosses_native_matcher_boundary"
    ] is True


def test_impulse_v2_has_four_tiles_256_balanced_requested_points():
    result = materialize("paired_impulse_median3")
    recipe = result.receipt.recipe
    assert recipe["samples_per_active_tile"] == 64
    assert recipe["requested_impulse_pixels"] == 256
    assert recipe["polarity_balance_requested"] == [128, 128]
    assert len(recipe["tile_origins_y_x"]) == 4
    assert len({tuple(value) for value in recipe["sample_coordinates_y_x"]}) == 256
    assert 0 < result.receipt.changed_pixels[0] <= 256
    assert result.receipt.changed_pixels[1] == 0


def test_additive_v2_corrupts_both_endpoints_and_disables_exact_impulse_priority():
    result = materialize("paired_additive_wiener3")
    assert all(value > 0 for value in result.receipt.changed_pixels)
    for image in (result.first, result.second):
        raw_extreme = np.all(image <= 1, axis=2) | np.all(image >= 254, axis=2)
        assert not raw_extreme.any()
    recipe = result.receipt.recipe
    assert recipe["sigma_uint8"] == 60.0
    assert recipe["active_tile_shape"] == [2, 2]
    for endpoint in ("first", "second"):
        details = recipe["endpoint_details"][endpoint]
        assert len(details["tile_origins_y_x"]) == 4
        assert details["noise_tile_pixels"] == 4 * 32 * 32


@pytest.mark.parametrize(
    "stratum,family,parameter",
    [
        ("common_disk", "disk", [8.0]),
        ("common_gaussian", "gaussian", [3.0]),
        ("common_motion", "motion", [17.0, 30.0]),
    ],
)
def test_blur_v2_covers_every_complete_tile_and_records_moments(
    stratum, family, parameter,
):
    result = materialize(stratum)
    recipe = result.receipt.recipe
    assert recipe["family"] == family
    assert recipe["parameter"] == parameter
    assert recipe["complete_tile_count"] == 6 * 8
    assert recipe["complete_tile_extent_xyxy"] == [0, 0, 512, 384]
    assert recipe["kernel_second_moment_x_px2"] > 0.0
    assert recipe["kernel_second_moment_y_px2"] >= 0.0
    assert recipe["exact_family_must_be_unique_direct_winner"] is True
    assert result.receipt.changed_pixels[1] == 0


@pytest.mark.parametrize("stratum", ["jpeg_qcell_v3", "jpeg_codec_path_v4"])
def test_jpeg_v2_uses_same_full_endpoint_q6_input(stratum):
    result = materialize(stratum)
    recipe = result.receipt.recipe
    assert recipe["full_endpoint"] is True
    assert recipe["ijg_quality"] == 6
    assert recipe["subsampling"] == 2
    assert recipe["region_xyxy"] == [0, 0, 512, 384]
    assert result.receipt.changed_pixels[0] > 0
    assert result.receipt.changed_pixels[1] == 0
    other = materialize(
        "jpeg_codec_path_v4" if stratum == "jpeg_qcell_v3" else "jpeg_qcell_v3"
    )
    assert np.array_equal(result.first, other.first)


def test_physical_pair_identity_participates_in_seed_and_receipt():
    one = materialize("paired_additive_wiener3", pair_id="pair-001")
    two = materialize("paired_additive_wiener3", pair_id="pair-002")
    assert not np.array_equal(one.first, two.first)
    assert one.receipt.receipt_sha256 != two.receipt.receipt_sha256


def test_v2_recipe_table_is_exactly_seven_strata():
    assert tuple(RECIPE_PARAMETERS_V2) == CONTROLLED_CORRUPTION_STRATA_V2


def test_invalid_geometry_identity_stratum_and_receipt_hash_fail_closed():
    first, second = pair()
    with pytest.raises(ValueError, match=">=256px"):
        materialize_controlled_corruption_v2(
            first=first[:128], second=second[:128], component_id="scene-a",
            physical_pair_id="pair-001", mechanism_stratum="common_disk",
        )
    with pytest.raises(ValueError, match="physical_pair_id"):
        materialize_controlled_corruption_v2(
            first=first, second=second, component_id="scene-a",
            physical_pair_id="", mechanism_stratum="common_disk",
        )
    with pytest.raises(ValueError, match="unknown"):
        materialize_controlled_corruption_v2(
            first=first, second=second, component_id="scene-a",
            physical_pair_id="pair-001", mechanism_stratum="unknown",
        )
    result = materialize("common_disk")
    with pytest.raises(ValueError, match="receipt hash drift"):
        replace(result.receipt, receipt_sha256=hashlib.sha256(b"bad").hexdigest())
