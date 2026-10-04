from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from stablebridge.physical_repair.u2flow_augmentations import (
    SINTEL_OFFICIAL_CODE_CROP_PROVENANCE,
    SINTEL_PAPER_REPORTED_CROP_HW,
    SINTEL_U2FLOW_AUGMENTATION_PROFILE_V1,
    U2FlowAugmentationProfileV1,
    apply_u2flow_recipe_v1,
    augment_pair_v1,
    derive_u2flow_view_seed_v1,
    held_out_u2flow_profile_v1,
    sample_u2flow_role_recipe_v1,
    sample_u2flow_recipe_v1,
    transform_points_v1,
    validate_u2flow_recipe_v1,
)


def _profile(**changes: object) -> U2FlowAugmentationProfileV1:
    values: dict[str, object] = {
        "profile_id": "test-u2flow-profile-v1",
        "crop_hw": (6, 8),
        "crop_provenance": "test_fixture",
        "paper_reported_crop_hw": (448, 1024),
        "horizontal_flip_probability": 0.5,
        "vertical_flip_probability": 0.0,
        "swap_probability": 0.5,
        "affine_probability": 0.0,
        "rotation_degrees_range": (0.0, 0.0),
        "scale_range": (1.0, 1.0),
        "translate_x_fraction_range": (0.0, 0.0),
        "translate_y_fraction_range": (0.0, 0.0),
        "brightness_range": (1.0, 1.0),
        "contrast_range": (1.0, 1.0),
        "saturation_range": (1.0, 1.0),
        "gaussian_blur_probability": 0.0,
        "gaussian_blur_sigma_range": (0.5, 0.5),
        "last_frame_erasing_probability": 0.0,
        "erase_area_fraction_range": (0.1, 0.1),
        "erase_aspect_ratio_range": (1.0, 1.0),
    }
    values.update(changes)
    return U2FlowAugmentationProfileV1(**values)


def _gradient_pair(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((height, width))
    base = (yy * width + xx).astype(np.uint8)
    first = np.stack((base, base, base), axis=-1)
    second = first + np.uint8(20)
    return first, second


def test_sintel_default_distinguishes_official_code_crop_from_paper_crop() -> None:
    profile = SINTEL_U2FLOW_AUGMENTATION_PROFILE_V1

    assert profile.crop_hw == (384, 832)
    assert profile.crop_provenance == SINTEL_OFFICIAL_CODE_CROP_PROVENANCE
    assert "official_code" in profile.crop_provenance
    assert "not_paper" in profile.crop_provenance
    assert profile.paper_reported_crop_hw == SINTEL_PAPER_REPORTED_CROP_HW
    assert profile.paper_reported_crop_hw == (448, 1024)
    assert len(profile.profile_hash) == 64


def test_sampling_is_reproducible_and_hash_bound() -> None:
    profile = _profile()

    first = sample_u2flow_recipe_v1((9, 12), seed=1234, profile=profile)
    replay = sample_u2flow_recipe_v1((9, 12), seed=1234, profile=profile)
    different = sample_u2flow_recipe_v1((9, 12), seed=1235, profile=profile)

    assert first == replay
    assert first.recipe_hash == replay.recipe_hash
    assert first.recipe_hash != different.recipe_hash
    assert first.profile_hash == profile.profile_hash


def test_role_recipe_makes_fit_epoch_vary_and_held_out_epoch_invariant() -> None:
    profile = _profile(crop_hw=(6, 8))
    fit_zero, fit_profile = sample_u2flow_role_recipe_v1(
        (10, 14),
        row_id="finetune:clean:alley_1:0001-0002",
        split_role="fit",
        epoch=0,
        master_seed=8603,
        fit_profile=profile,
    )
    fit_one, _ = sample_u2flow_role_recipe_v1(
        (10, 14),
        row_id="finetune:clean:alley_1:0001-0002",
        split_role="fit",
        epoch=1,
        master_seed=8603,
        fit_profile=profile,
    )
    held_zero, held_profile = sample_u2flow_role_recipe_v1(
        (10, 14),
        row_id="finetune:clean:ambush_2:0001-0002",
        split_role="validation",
        epoch=0,
        master_seed=8603,
        fit_profile=profile,
    )
    held_later, held_profile_later = sample_u2flow_role_recipe_v1(
        (10, 14),
        row_id="finetune:clean:ambush_2:0001-0002",
        split_role="validation",
        epoch=99,
        master_seed=8603,
        fit_profile=profile,
    )

    assert fit_zero.recipe_hash != fit_one.recipe_hash
    assert fit_profile == profile
    assert held_zero == held_later
    assert held_profile == held_profile_later == held_out_u2flow_profile_v1(profile)
    assert held_zero.spatial.crop_top == 2
    assert held_zero.spatial.crop_left == 3
    assert held_zero.spatial.horizontal_flip is False
    assert held_zero.spatial.vertical_flip is False
    assert held_zero.spatial.swap_endpoints is False
    assert held_zero.spatial.affine_enabled is False
    assert held_zero.first_appearance.brightness == 1.0
    assert held_zero.last_frame_erasing.enabled is False
    validate_u2flow_recipe_v1(held_zero, held_profile)


def test_row_epoch_seed_schedule_is_stable_and_role_builder_rejects_unknown() -> None:
    seed = derive_u2flow_view_seed_v1("row:1", epoch=2, master_seed=3)
    assert seed == derive_u2flow_view_seed_v1("row:1", epoch=2, master_seed=3)
    assert seed != derive_u2flow_view_seed_v1("row:1", epoch=3, master_seed=3)
    with pytest.raises(ValueError, match="unknown U2Flow split role"):
        sample_u2flow_role_recipe_v1(
            (9, 12),
            row_id="row:1",
            split_role="test",
            epoch=0,
            master_seed=3,
            fit_profile=_profile(),
        )


def test_pair_uses_one_shared_crop_and_flip_geometry() -> None:
    profile = _profile(
        horizontal_flip_probability=1.0,
        swap_probability=0.0,
    )
    first, second = _gradient_pair(9, 12)

    result = augment_pair_v1(first, second, seed=7, profile=profile)

    assert result.first.shape == (6, 8, 3)
    assert result.second.shape == (6, 8, 3)
    assert result.recipe.spatial.horizontal_flip is True
    assert result.recipe.spatial.swap_endpoints is False
    np.testing.assert_array_equal(
        result.second.astype(np.int16) - result.first.astype(np.int16),
        np.full((6, 8, 3), 20, dtype=np.int16),
    )
    assert result.valid_support.all()


def test_vertical_flip_is_shared_by_both_endpoints() -> None:
    profile = _profile(
        horizontal_flip_probability=0.0,
        vertical_flip_probability=1.0,
        swap_probability=0.0,
    )
    first, second = _gradient_pair(6, 8)

    result = augment_pair_v1(first, second, seed=11, profile=profile)

    assert result.recipe.spatial.vertical_flip is True
    np.testing.assert_array_equal(result.first, np.flipud(first))
    np.testing.assert_array_equal(result.second, np.flipud(second))


def test_swap_is_common_endpoint_order_operation() -> None:
    profile = _profile(
        horizontal_flip_probability=0.0,
        swap_probability=1.0,
    )
    first = np.full((6, 8, 3), 11, dtype=np.uint8)
    second = np.full((6, 8, 3), 29, dtype=np.uint8)

    result = augment_pair_v1(first, second, seed=2, profile=profile)

    assert result.recipe.spatial.swap_endpoints is True
    np.testing.assert_array_equal(result.first, second)
    np.testing.assert_array_equal(result.second, first)


def test_last_frame_erasing_happens_after_swap_and_never_erases_first() -> None:
    profile = _profile(
        horizontal_flip_probability=0.0,
        swap_probability=1.0,
        last_frame_erasing_probability=1.0,
        erase_area_fraction_range=(0.25, 0.25),
    )
    original_first = np.full((6, 8, 3), 17, dtype=np.uint8)
    original_second = np.full((6, 8, 3), 91, dtype=np.uint8)

    result = augment_pair_v1(original_first, original_second, seed=10, profile=profile)
    erase = result.recipe.last_frame_erasing

    assert erase.enabled is True
    np.testing.assert_array_equal(result.first, original_second)
    expected_second = original_first.copy()
    expected_second[
        erase.top : erase.top + erase.height,
        erase.left : erase.left + erase.width,
    ] = np.asarray(erase.fill_rgb, dtype=np.uint8)
    np.testing.assert_array_equal(result.second, expected_second)


def test_appearance_supports_color_jitter_and_gaussian_blur() -> None:
    profile = _profile(
        brightness_range=(1.25, 1.25),
        contrast_range=(1.1, 1.1),
        saturation_range=(0.5, 0.5),
        gaussian_blur_probability=1.0,
        gaussian_blur_sigma_range=(1.0, 1.0),
        horizontal_flip_probability=0.0,
        swap_probability=0.0,
    )
    first = np.zeros((6, 8, 3), dtype=np.uint8)
    first[3, 4] = (255, 40, 10)
    second = first.copy()

    result = augment_pair_v1(first, second, seed=4, profile=profile)

    assert result.recipe.first_appearance.brightness == 1.25
    assert result.recipe.first_appearance.contrast == 1.1
    assert result.recipe.first_appearance.saturation == 0.5
    assert result.recipe.first_appearance.gaussian_blur is True
    assert result.recipe.first_appearance.gaussian_blur_sigma == 1.0
    assert np.count_nonzero(result.first) > np.count_nonzero(first)
    assert not np.array_equal(result.first, first)


def test_affine_returns_inverse_parameters_and_strict_valid_support() -> None:
    profile = _profile(
        crop_hw=(8, 10),
        horizontal_flip_probability=1.0,
        swap_probability=0.0,
        affine_probability=1.0,
        rotation_degrees_range=(12.0, 12.0),
        scale_range=(1.0, 1.0),
        translate_x_fraction_range=(0.2, 0.2),
        translate_y_fraction_range=(-0.1, -0.1),
    )
    recipe = sample_u2flow_recipe_v1((8, 10), seed=3, profile=profile)

    forward = np.asarray(recipe.spatial.input_to_output).reshape(3, 3)
    inverse = np.asarray(recipe.spatial.output_to_input).reshape(3, 3)
    np.testing.assert_allclose(forward @ inverse, np.eye(3), atol=1e-9)
    points = np.asarray([[0.0, 0.0], [4.0, 3.0], [9.0, 7.0]])
    moved = transform_points_v1(points, recipe.spatial.input_to_output)
    restored = transform_points_v1(moved, recipe.spatial.output_to_input)
    np.testing.assert_allclose(restored, points, atol=1e-9)

    support = validate_u2flow_recipe_v1(recipe, profile)
    assert support.dtype == np.bool_
    assert support.shape == (8, 10)
    assert support.any()
    assert not support.all()


def test_profile_and_recipe_tampering_fail_closed() -> None:
    profile = _profile()
    recipe = sample_u2flow_recipe_v1((9, 12), seed=22, profile=profile)

    with pytest.raises(ValueError, match="profile hash drift"):
        replace(profile, crop_hw=(5, 8))
    with pytest.raises(ValueError, match="recipe hash drift"):
        replace(recipe, seed=23)
    with pytest.raises(ValueError, match="recipe hash drift"):
        replace(recipe, valid_support_hash="f" * 64)

    drifted_profile = replace(
        profile,
        brightness_range=(0.9, 1.1),
        profile_hash="",
    )
    with pytest.raises(ValueError, match="binding drift"):
        validate_u2flow_recipe_v1(recipe, drifted_profile)

    object.__setattr__(profile, "crop_provenance", "tampered_source")
    with pytest.raises(ValueError, match="profile hash drift"):
        validate_u2flow_recipe_v1(recipe, profile)


def test_matrix_and_nested_recipe_tampering_fail_closed() -> None:
    profile = _profile()
    recipe = sample_u2flow_recipe_v1((9, 12), seed=8, profile=profile)

    # Frozen dataclasses prevent ordinary mutation.  Bypass that protection to
    # model corrupted in-memory/deserialized state and exercise runtime checks.
    object.__setattr__(recipe.spatial, "crop_left", recipe.spatial.crop_left + 1)
    with pytest.raises(ValueError, match="parameters/matrix drift"):
        validate_u2flow_recipe_v1(recipe, profile)


def test_support_hash_drift_is_detected_even_if_outer_hash_is_resealed() -> None:
    profile = _profile()
    recipe = sample_u2flow_recipe_v1((9, 12), seed=18, profile=profile)
    object.__setattr__(recipe, "valid_support_hash", "a" * 64)

    with pytest.raises(ValueError, match="valid support hash drift"):
        validate_u2flow_recipe_v1(recipe, profile)


def test_invalid_inputs_and_undersized_crop_fail_closed() -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="crop exceeds"):
        sample_u2flow_recipe_v1((5, 8), seed=1, profile=profile)
    with pytest.raises(ValueError, match="uint64"):
        sample_u2flow_recipe_v1((6, 8), seed=True, profile=profile)

    recipe = sample_u2flow_recipe_v1((6, 8), seed=1, profile=profile)
    uint8_frame = np.zeros((6, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="uint8"):
        apply_u2flow_recipe_v1(
            uint8_frame.astype(np.float32), uint8_frame, recipe, profile=profile
        )
    with pytest.raises(ValueError, match="shape"):
        apply_u2flow_recipe_v1(
            uint8_frame[:, :, :1], uint8_frame, recipe, profile=profile
        )


def test_apply_replay_is_byte_reproducible() -> None:
    profile = _profile(
        affine_probability=1.0,
        rotation_degrees_range=(-5.0, 5.0),
        scale_range=(0.95, 1.05),
        translate_x_fraction_range=(-0.05, 0.05),
        translate_y_fraction_range=(-0.05, 0.05),
        brightness_range=(0.9, 1.1),
        contrast_range=(0.9, 1.1),
        saturation_range=(0.9, 1.1),
        gaussian_blur_probability=0.5,
        last_frame_erasing_probability=0.5,
    )
    first, second = _gradient_pair(9, 12)
    recipe = sample_u2flow_recipe_v1((9, 12), seed=998, profile=profile)

    first_result = apply_u2flow_recipe_v1(first, second, recipe, profile=profile)
    replay = apply_u2flow_recipe_v1(first, second, recipe, profile=profile)

    np.testing.assert_array_equal(first_result.first, replay.first)
    np.testing.assert_array_equal(first_result.second, replay.second)
    np.testing.assert_array_equal(first_result.valid_support, replay.valid_support)
    assert first_result.recipe.recipe_hash == replay.recipe.recipe_hash
