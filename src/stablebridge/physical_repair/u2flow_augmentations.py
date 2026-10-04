"""Deterministic, in-memory U²Flow-style pair augmentations.

This module intentionally does not read or write files and does not depend on
``torchvision``.  A sampled recipe is a frozen, hash-bound record: the same
spatial transform is used for both frames, while appearance transforms are
recorded per output frame.  Endpoint swapping happens before appearance and
erasing, so ``last_frame_erasing`` always refers to the returned second frame.

The default Sintel crop is the 384x832 setting used by the public official-code
profile.  The 448x1024 value reported in the paper is retained as explicit
metadata so the two settings cannot be silently conflated.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import re
from typing import Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


U2FLOW_PROFILE_SCHEMA_V1 = "stablebridge-u2flow-augmentation-profile/v1"
U2FLOW_RECIPE_SCHEMA_V1 = "stablebridge-u2flow-augmentation-recipe/v1"
SINTEL_OFFICIAL_CODE_CROP_PROVENANCE = (
    "u2flow_official_code_profile_not_paper_table"
)
SINTEL_PAPER_REPORTED_CROP_HW = (448, 1024)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UINT64_MAX = (1 << 64) - 1

try:  # Pillow >= 9.1
    _PIL_AFFINE = Image.Transform.AFFINE
    _PIL_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - compatibility with older Pillow
    _PIL_AFFINE = Image.AFFINE
    _PIL_BILINEAR = Image.BILINEAR


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty canonical text")
    return value


def _digest(value: object, name: str) -> str:
    result = _canonical_text(value, name)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _positive_pair(value: Sequence[int], name: str) -> tuple[int, int]:
    result = tuple(value)
    if (
        len(result) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in result)
        or any(item <= 0 for item in result)
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return result[0], result[1]


def _range_pair(
    value: Sequence[float],
    name: str,
    *,
    positive: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float]:
    result = tuple(value)
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    low = _finite(result[0], f"{name} lower bound")
    high = _finite(result[1], f"{name} upper bound")
    if low > high:
        raise ValueError(f"{name} lower bound exceeds upper bound")
    if positive and low <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    if lower_bound is not None and low < lower_bound:
        raise ValueError(f"{name} is below its allowed lower bound")
    if upper_bound is not None and high > upper_bound:
        raise ValueError(f"{name} exceeds its allowed upper bound")
    return low, high


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _matrix_tuple(value: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(_finite(item, name) for item in value)
    if len(result) != 9:
        raise ValueError(f"{name} must contain nine finite values")
    return result


def _matrix_array(value: Sequence[float]) -> np.ndarray:
    return np.asarray(tuple(value), dtype=np.float64).reshape(3, 3)


@dataclass(frozen=True)
class U2FlowAugmentationProfileV1:
    """Allowed sampling ranges for one augmentation family."""

    profile_id: str = "u2flow-sintel-official-code-crop-384x832-v1"
    crop_hw: tuple[int, int] = (384, 832)
    crop_provenance: str = SINTEL_OFFICIAL_CODE_CROP_PROVENANCE
    paper_reported_crop_hw: tuple[int, int] = SINTEL_PAPER_REPORTED_CROP_HW
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.1
    swap_probability: float = 0.5
    affine_probability: float = 0.5
    rotation_degrees_range: tuple[float, float] = (-17.0, 17.0)
    scale_range: tuple[float, float] = (0.9, 1.1)
    translate_x_fraction_range: tuple[float, float] = (-0.05, 0.05)
    translate_y_fraction_range: tuple[float, float] = (-0.05, 0.05)
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    saturation_range: tuple[float, float] = (0.8, 1.2)
    gaussian_blur_probability: float = 0.2
    gaussian_blur_sigma_range: tuple[float, float] = (0.1, 2.0)
    last_frame_erasing_probability: float = 0.5
    erase_area_fraction_range: tuple[float, float] = (0.02, 0.2)
    erase_aspect_ratio_range: tuple[float, float] = (0.3, 3.3)
    schema: str = U2FLOW_PROFILE_SCHEMA_V1
    profile_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        _canonical_text(self.profile_id, "profile id")
        if self.schema != U2FLOW_PROFILE_SCHEMA_V1:
            raise ValueError("unsupported U2Flow augmentation profile schema")
        object.__setattr__(self, "crop_hw", _positive_pair(self.crop_hw, "crop_hw"))
        object.__setattr__(
            self,
            "paper_reported_crop_hw",
            _positive_pair(self.paper_reported_crop_hw, "paper_reported_crop_hw"),
        )
        _canonical_text(self.crop_provenance, "crop provenance")
        for attribute in (
            "horizontal_flip_probability",
            "vertical_flip_probability",
            "swap_probability",
            "affine_probability",
            "gaussian_blur_probability",
            "last_frame_erasing_probability",
        ):
            object.__setattr__(
                self, attribute, _probability(getattr(self, attribute), attribute)
            )
        range_specs = (
            ("rotation_degrees_range", False, None, None),
            ("scale_range", True, None, None),
            ("translate_x_fraction_range", False, -1.0, 1.0),
            ("translate_y_fraction_range", False, -1.0, 1.0),
            ("brightness_range", True, None, None),
            ("contrast_range", True, None, None),
            ("saturation_range", True, None, None),
            ("gaussian_blur_sigma_range", True, None, None),
            ("erase_area_fraction_range", True, None, 1.0),
            ("erase_aspect_ratio_range", True, None, None),
        )
        for attribute, positive, lower, upper in range_specs:
            object.__setattr__(
                self,
                attribute,
                _range_pair(
                    getattr(self, attribute),
                    attribute,
                    positive=positive,
                    lower_bound=lower,
                    upper_bound=upper,
                ),
            )
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.profile_hash and self.profile_hash != expected:
            raise ValueError("U2Flow augmentation profile hash drift")
        object.__setattr__(self, "profile_hash", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "crop_hw": list(self.crop_hw),
            "crop_provenance": self.crop_provenance,
            "paper_reported_crop_hw": list(self.paper_reported_crop_hw),
            "horizontal_flip_probability": self.horizontal_flip_probability,
            "vertical_flip_probability": self.vertical_flip_probability,
            "swap_probability": self.swap_probability,
            "affine_probability": self.affine_probability,
            "rotation_degrees_range": list(self.rotation_degrees_range),
            "scale_range": list(self.scale_range),
            "translate_x_fraction_range": list(self.translate_x_fraction_range),
            "translate_y_fraction_range": list(self.translate_y_fraction_range),
            "brightness_range": list(self.brightness_range),
            "contrast_range": list(self.contrast_range),
            "saturation_range": list(self.saturation_range),
            "gaussian_blur_probability": self.gaussian_blur_probability,
            "gaussian_blur_sigma_range": list(self.gaussian_blur_sigma_range),
            "last_frame_erasing_probability": self.last_frame_erasing_probability,
            "erase_area_fraction_range": list(self.erase_area_fraction_range),
            "erase_aspect_ratio_range": list(self.erase_aspect_ratio_range),
        }
        if include_hash:
            result["profile_hash"] = self.profile_hash
        return result


@dataclass(frozen=True)
class SpatialAffineRecipeV1:
    crop_top: int
    crop_left: int
    crop_hw: tuple[int, int]
    horizontal_flip: bool
    vertical_flip: bool
    swap_endpoints: bool
    affine_enabled: bool
    rotation_degrees: float
    scale: float
    translate_xy_px: tuple[float, float]
    input_to_output: tuple[float, ...]
    output_to_input: tuple[float, ...]
    linear_determinant: float

    def __post_init__(self) -> None:
        for value, name in (
            (self.crop_top, "crop top"),
            (self.crop_left, "crop left"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        object.__setattr__(self, "crop_hw", _positive_pair(self.crop_hw, "crop_hw"))
        _strict_bool(self.horizontal_flip, "horizontal flip")
        _strict_bool(self.vertical_flip, "vertical flip")
        _strict_bool(self.swap_endpoints, "endpoint swap")
        _strict_bool(self.affine_enabled, "affine enabled")
        object.__setattr__(
            self, "rotation_degrees", _finite(self.rotation_degrees, "rotation")
        )
        scale = _finite(self.scale, "scale")
        if scale <= 0.0:
            raise ValueError("scale must be strictly positive")
        object.__setattr__(self, "scale", scale)
        translation = tuple(self.translate_xy_px)
        if len(translation) != 2:
            raise ValueError("translation must contain two values")
        object.__setattr__(
            self,
            "translate_xy_px",
            (
                _finite(translation[0], "x translation"),
                _finite(translation[1], "y translation"),
            ),
        )
        input_to_output = _matrix_tuple(self.input_to_output, "input_to_output")
        output_to_input = _matrix_tuple(self.output_to_input, "output_to_input")
        object.__setattr__(self, "input_to_output", input_to_output)
        object.__setattr__(self, "output_to_input", output_to_input)
        forward = _matrix_array(input_to_output)
        inverse = _matrix_array(output_to_input)
        if not np.allclose(forward[2], (0.0, 0.0, 1.0), atol=1e-12, rtol=0.0):
            raise ValueError("input_to_output must be affine")
        if not np.allclose(inverse[2], (0.0, 0.0, 1.0), atol=1e-12, rtol=0.0):
            raise ValueError("output_to_input must be affine")
        if not np.allclose(forward @ inverse, np.eye(3), atol=1e-9, rtol=1e-9):
            raise ValueError("spatial affine matrices are not mutual inverses")
        determinant = _finite(self.linear_determinant, "linear determinant")
        observed = float(np.linalg.det(forward[:2, :2]))
        if abs(observed) <= 1e-12 or not math.isclose(
            determinant, observed, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("spatial affine determinant drift")
        object.__setattr__(self, "linear_determinant", determinant)
        if not self.affine_enabled and (
            self.rotation_degrees != 0.0
            or self.scale != 1.0
            or self.translate_xy_px != (0.0, 0.0)
        ):
            raise ValueError("disabled affine must use identity parameters")

    def as_dict(self) -> dict[str, object]:
        return {
            "crop_top": self.crop_top,
            "crop_left": self.crop_left,
            "crop_hw": list(self.crop_hw),
            "horizontal_flip": self.horizontal_flip,
            "vertical_flip": self.vertical_flip,
            "swap_endpoints": self.swap_endpoints,
            "affine_enabled": self.affine_enabled,
            "rotation_degrees": self.rotation_degrees,
            "scale": self.scale,
            "translate_xy_px": list(self.translate_xy_px),
            "input_to_output": list(self.input_to_output),
            "output_to_input": list(self.output_to_input),
            "linear_determinant": self.linear_determinant,
        }


@dataclass(frozen=True)
class AppearanceRecipeV1:
    brightness: float
    contrast: float
    saturation: float
    gaussian_blur: bool
    gaussian_blur_sigma: float

    def __post_init__(self) -> None:
        for attribute in ("brightness", "contrast", "saturation"):
            value = _finite(getattr(self, attribute), attribute)
            if value <= 0.0:
                raise ValueError(f"{attribute} must be strictly positive")
            object.__setattr__(self, attribute, value)
        _strict_bool(self.gaussian_blur, "gaussian blur")
        sigma = _finite(self.gaussian_blur_sigma, "gaussian blur sigma")
        if sigma < 0.0 or (self.gaussian_blur and sigma <= 0.0):
            raise ValueError("enabled Gaussian blur needs positive sigma")
        if not self.gaussian_blur and sigma != 0.0:
            raise ValueError("disabled Gaussian blur must have zero sigma")
        object.__setattr__(self, "gaussian_blur_sigma", sigma)

    def as_dict(self) -> dict[str, object]:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "gaussian_blur": self.gaussian_blur,
            "gaussian_blur_sigma": self.gaussian_blur_sigma,
        }


@dataclass(frozen=True)
class LastFrameErasingRecipeV1:
    enabled: bool
    top: int
    left: int
    height: int
    width: int
    fill_rgb: tuple[int, int, int]

    def __post_init__(self) -> None:
        _strict_bool(self.enabled, "last-frame erasing enabled")
        for attribute in ("top", "left", "height", "width"):
            value = getattr(self, attribute)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"erase {attribute} must be a nonnegative integer")
        fill = tuple(self.fill_rgb)
        if len(fill) != 3 or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > 255
            for item in fill
        ):
            raise ValueError("erase fill_rgb must contain three uint8 values")
        object.__setattr__(self, "fill_rgb", fill)
        if self.enabled:
            if self.height == 0 or self.width == 0:
                raise ValueError("enabled last-frame erasing needs positive area")
        elif (
            self.top != 0
            or self.left != 0
            or self.height != 0
            or self.width != 0
            or self.fill_rgb != (0, 0, 0)
        ):
            raise ValueError("disabled last-frame erasing must use zero parameters")

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "top": self.top,
            "left": self.left,
            "height": self.height,
            "width": self.width,
            "fill_rgb": list(self.fill_rgb),
        }


@dataclass(frozen=True)
class U2FlowAugmentationRecipeV1:
    profile_hash: str
    seed: int
    input_hw: tuple[int, int]
    spatial: SpatialAffineRecipeV1
    first_appearance: AppearanceRecipeV1
    second_appearance: AppearanceRecipeV1
    last_frame_erasing: LastFrameErasingRecipeV1
    valid_support_hash: str
    schema: str = U2FLOW_RECIPE_SCHEMA_V1
    recipe_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if self.schema != U2FLOW_RECIPE_SCHEMA_V1:
            raise ValueError("unsupported U2Flow augmentation recipe schema")
        _digest(self.profile_hash, "profile hash")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or self.seed > _UINT64_MAX
        ):
            raise ValueError("seed must be a uint64 integer")
        object.__setattr__(self, "input_hw", _positive_pair(self.input_hw, "input_hw"))
        if not isinstance(self.spatial, SpatialAffineRecipeV1):
            raise ValueError("spatial recipe must be typed")
        if self.spatial.crop_hw[0] > self.input_hw[0] or self.spatial.crop_hw[1] > self.input_hw[1]:
            raise ValueError("crop exceeds input dimensions")
        if self.spatial.crop_top + self.spatial.crop_hw[0] > self.input_hw[0]:
            raise ValueError("crop bottom exceeds input dimensions")
        if self.spatial.crop_left + self.spatial.crop_hw[1] > self.input_hw[1]:
            raise ValueError("crop right exceeds input dimensions")
        for value in (self.first_appearance, self.second_appearance):
            if not isinstance(value, AppearanceRecipeV1):
                raise ValueError("appearance recipe must be typed")
        if not isinstance(self.last_frame_erasing, LastFrameErasingRecipeV1):
            raise ValueError("last-frame erasing recipe must be typed")
        erasing = self.last_frame_erasing
        if erasing.enabled and (
            erasing.top + erasing.height > self.spatial.crop_hw[0]
            or erasing.left + erasing.width > self.spatial.crop_hw[1]
        ):
            raise ValueError("last-frame erase box exceeds crop")
        _digest(self.valid_support_hash, "valid support hash")
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.recipe_hash and self.recipe_hash != expected:
            raise ValueError("U2Flow augmentation recipe hash drift")
        object.__setattr__(self, "recipe_hash", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": self.schema,
            "profile_hash": self.profile_hash,
            "seed": self.seed,
            "input_hw": list(self.input_hw),
            "spatial": self.spatial.as_dict(),
            "first_appearance": self.first_appearance.as_dict(),
            "second_appearance": self.second_appearance.as_dict(),
            "last_frame_erasing": self.last_frame_erasing.as_dict(),
            "valid_support_hash": self.valid_support_hash,
        }
        if include_hash:
            result["recipe_hash"] = self.recipe_hash
        return result


@dataclass(frozen=True)
class AugmentedPairV1:
    first: np.ndarray = field(repr=False, compare=False)
    second: np.ndarray = field(repr=False, compare=False)
    valid_support: np.ndarray = field(repr=False, compare=False)
    recipe: U2FlowAugmentationRecipeV1


def _native_affine_matrix(
    input_hw: tuple[int, int],
    rotation_degrees: float,
    scale: float,
    translate_xy_px: tuple[float, float],
) -> np.ndarray:
    height, width = input_hw
    radians = math.radians(rotation_degrees)
    cosine = math.cos(radians) * scale
    sine = math.sin(radians) * scale
    linear = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    to_origin = np.array(
        [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    from_origin = np.array(
        [
            [1.0, 0.0, center_x + translate_xy_px[0]],
            [0.0, 1.0, center_y + translate_xy_px[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return from_origin @ linear @ to_origin


def _compose_spatial_matrix(
    input_hw: tuple[int, int],
    crop_hw: tuple[int, int],
    crop_top: int,
    crop_left: int,
    horizontal_flip: bool,
    vertical_flip: bool,
    rotation_degrees: float,
    scale: float,
    translate_xy_px: tuple[float, float],
) -> np.ndarray:
    native_affine = _native_affine_matrix(
        input_hw, rotation_degrees, scale, translate_xy_px
    )
    crop = np.array(
        [[1.0, 0.0, -crop_left], [0.0, 1.0, -crop_top], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if horizontal_flip:
        flip = np.array(
            [[-1.0, 0.0, crop_hw[1] - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    else:
        flip = np.eye(3, dtype=np.float64)
    if vertical_flip:
        vertical = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, crop_hw[0] - 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    else:
        vertical = np.eye(3, dtype=np.float64)
    return vertical @ flip @ crop @ native_affine


def valid_support_from_recipe_v1(recipe: U2FlowAugmentationRecipeV1) -> np.ndarray:
    """Return output pixels whose inverse-mapped centers lie in the input."""

    if not isinstance(recipe, U2FlowAugmentationRecipeV1):
        raise ValueError("recipe must be a typed U2Flow recipe")
    output_height, output_width = recipe.spatial.crop_hw
    yy, xx = np.indices((output_height, output_width), dtype=np.float64)
    homogeneous = np.stack((xx.ravel(), yy.ravel(), np.ones(xx.size)), axis=0)
    source = _matrix_array(recipe.spatial.output_to_input) @ homogeneous
    source_x = source[0].reshape(output_height, output_width)
    source_y = source[1].reshape(output_height, output_width)
    input_height, input_width = recipe.input_hw
    return (
        np.isfinite(source_x)
        & np.isfinite(source_y)
        & (source_x >= 0.0)
        & (source_x <= input_width - 1.0)
        & (source_y >= 0.0)
        & (source_y <= input_height - 1.0)
    )


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    if bounds[0] == bounds[1]:
        return bounds[0]
    return float(rng.uniform(bounds[0], bounds[1]))


def _bernoulli(rng: np.random.Generator, probability: float) -> bool:
    if probability == 0.0:
        return False
    if probability == 1.0:
        return True
    return bool(rng.random() < probability)


def _sample_appearance(
    rng: np.random.Generator, profile: U2FlowAugmentationProfileV1
) -> AppearanceRecipeV1:
    blurred = _bernoulli(rng, profile.gaussian_blur_probability)
    return AppearanceRecipeV1(
        brightness=_uniform(rng, profile.brightness_range),
        contrast=_uniform(rng, profile.contrast_range),
        saturation=_uniform(rng, profile.saturation_range),
        gaussian_blur=blurred,
        gaussian_blur_sigma=(
            _uniform(rng, profile.gaussian_blur_sigma_range) if blurred else 0.0
        ),
    )


def _sample_erasing(
    rng: np.random.Generator,
    profile: U2FlowAugmentationProfileV1,
) -> LastFrameErasingRecipeV1:
    if not _bernoulli(rng, profile.last_frame_erasing_probability):
        return LastFrameErasingRecipeV1(False, 0, 0, 0, 0, (0, 0, 0))
    height, width = profile.crop_hw
    area_fraction = _uniform(rng, profile.erase_area_fraction_range)
    log_aspect = _uniform(
        rng,
        (
            math.log(profile.erase_aspect_ratio_range[0]),
            math.log(profile.erase_aspect_ratio_range[1]),
        ),
    )
    aspect = math.exp(log_aspect)
    area = float(height * width) * area_fraction
    erase_height = min(height, max(1, int(round(math.sqrt(area / aspect)))))
    erase_width = min(width, max(1, int(round(math.sqrt(area * aspect)))))
    top = int(rng.integers(0, height - erase_height + 1))
    left = int(rng.integers(0, width - erase_width + 1))
    fill = tuple(int(value) for value in rng.integers(0, 256, size=3))
    return LastFrameErasingRecipeV1(
        True, top, left, erase_height, erase_width, fill
    )


def sample_u2flow_recipe_v1(
    input_hw: Sequence[int],
    *,
    seed: int,
    profile: U2FlowAugmentationProfileV1 | None = None,
    crop_mode: str = "random",
) -> U2FlowAugmentationRecipeV1:
    """Sample a reproducible, fully materialized pair-augmentation recipe."""

    selected = SINTEL_U2FLOW_AUGMENTATION_PROFILE_V1 if profile is None else profile
    if not isinstance(selected, U2FlowAugmentationProfileV1):
        raise ValueError("profile must be a typed U2Flow augmentation profile")
    native_hw = _positive_pair(input_hw, "input_hw")
    if selected.crop_hw[0] > native_hw[0] or selected.crop_hw[1] > native_hw[1]:
        raise ValueError("profile crop exceeds input dimensions")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > _UINT64_MAX:
        raise ValueError("seed must be a uint64 integer")
    rng = np.random.default_rng(seed)
    maximum_top = native_hw[0] - selected.crop_hw[0]
    maximum_left = native_hw[1] - selected.crop_hw[1]
    if crop_mode == "random":
        crop_top = int(rng.integers(0, maximum_top + 1))
        crop_left = int(rng.integers(0, maximum_left + 1))
    elif crop_mode == "center":
        crop_top = maximum_top // 2
        crop_left = maximum_left // 2
    else:
        raise ValueError("crop_mode must be random or center")
    horizontal_flip = _bernoulli(rng, selected.horizontal_flip_probability)
    vertical_flip = _bernoulli(rng, selected.vertical_flip_probability)
    swap_endpoints = _bernoulli(rng, selected.swap_probability)
    affine_enabled = _bernoulli(rng, selected.affine_probability)
    if affine_enabled:
        rotation_degrees = _uniform(rng, selected.rotation_degrees_range)
        scale = _uniform(rng, selected.scale_range)
        translate_xy_px = (
            _uniform(rng, selected.translate_x_fraction_range) * native_hw[1],
            _uniform(rng, selected.translate_y_fraction_range) * native_hw[0],
        )
    else:
        rotation_degrees = 0.0
        scale = 1.0
        translate_xy_px = (0.0, 0.0)
    forward = _compose_spatial_matrix(
        native_hw,
        selected.crop_hw,
        crop_top,
        crop_left,
        horizontal_flip,
        vertical_flip,
        rotation_degrees,
        scale,
        translate_xy_px,
    )
    inverse = np.linalg.inv(forward)
    spatial = SpatialAffineRecipeV1(
        crop_top=crop_top,
        crop_left=crop_left,
        crop_hw=selected.crop_hw,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        swap_endpoints=swap_endpoints,
        affine_enabled=affine_enabled,
        rotation_degrees=rotation_degrees,
        scale=scale,
        translate_xy_px=translate_xy_px,
        input_to_output=tuple(float(value) for value in forward.ravel()),
        output_to_input=tuple(float(value) for value in inverse.ravel()),
        linear_determinant=float(np.linalg.det(forward[:2, :2])),
    )
    first_appearance = _sample_appearance(rng, selected)
    second_appearance = _sample_appearance(rng, selected)
    erasing = _sample_erasing(rng, selected)

    # A temporary valid digest lets the typed recipe compute the same support
    # through its public function without weakening constructor validation.
    provisional = U2FlowAugmentationRecipeV1(
        profile_hash=selected.profile_hash,
        seed=seed,
        input_hw=native_hw,
        spatial=spatial,
        first_appearance=first_appearance,
        second_appearance=second_appearance,
        last_frame_erasing=erasing,
        valid_support_hash="0" * 64,
    )
    support_hash = _array_sha256(valid_support_from_recipe_v1(provisional))
    return U2FlowAugmentationRecipeV1(
        profile_hash=selected.profile_hash,
        seed=seed,
        input_hw=native_hw,
        spatial=spatial,
        first_appearance=first_appearance,
        second_appearance=second_appearance,
        last_frame_erasing=erasing,
        valid_support_hash=support_hash,
    )


def derive_u2flow_view_seed_v1(
    row_id: str,
    *,
    epoch: int,
    master_seed: int,
) -> int:
    """Derive a stable uint64 seed from a row, epoch, and experiment seed."""

    _canonical_text(row_id, "row id")
    for value, name in ((epoch, "epoch"), (master_seed, "master seed")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    payload = f"stablebridge-u2flow-view/v1\0{master_seed}\0{epoch}\0{row_id}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def held_out_u2flow_profile_v1(
    fit_profile: U2FlowAugmentationProfileV1,
) -> U2FlowAugmentationProfileV1:
    """Return the identity-appearance profile used by held-out scene roles."""

    if not isinstance(fit_profile, U2FlowAugmentationProfileV1):
        raise ValueError("fit_profile must be a typed U2Flow profile")
    return replace(
        fit_profile,
        profile_id=f"{fit_profile.profile_id}-held-out-center-identity",
        crop_provenance=f"{fit_profile.crop_provenance}-held-out-center-identity",
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        swap_probability=0.0,
        affine_probability=0.0,
        rotation_degrees_range=(0.0, 0.0),
        scale_range=(1.0, 1.0),
        translate_x_fraction_range=(0.0, 0.0),
        translate_y_fraction_range=(0.0, 0.0),
        brightness_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
        gaussian_blur_probability=0.0,
        gaussian_blur_sigma_range=(1.0, 1.0),
        last_frame_erasing_probability=0.0,
        profile_hash="",
    )


def sample_u2flow_role_recipe_v1(
    input_hw: Sequence[int],
    *,
    row_id: str,
    split_role: str,
    epoch: int,
    master_seed: int,
    fit_profile: U2FlowAugmentationProfileV1 | None = None,
) -> tuple[U2FlowAugmentationRecipeV1, U2FlowAugmentationProfileV1]:
    """Build a replayable fit view or deterministic held-out center view.

    Only ``fit`` views vary by epoch.  Validation, calibration, and evaluation
    use one identity-appearance center crop and deliberately ignore the epoch.
    Returning the selected profile with the recipe makes validation fail closed.
    """

    selected_fit = (
        SINTEL_U2FLOW_AUGMENTATION_PROFILE_V1
        if fit_profile is None
        else fit_profile
    )
    if split_role not in {"fit", "validation", "calibration", "evaluation"}:
        raise ValueError("unknown U2Flow split role")
    if split_role == "fit":
        seed = derive_u2flow_view_seed_v1(
            row_id, epoch=epoch, master_seed=master_seed
        )
        profile = selected_fit
        crop_mode = "random"
    else:
        seed = derive_u2flow_view_seed_v1(
            row_id, epoch=0, master_seed=master_seed
        )
        profile = held_out_u2flow_profile_v1(selected_fit)
        crop_mode = "center"
    return (
        sample_u2flow_recipe_v1(
            input_hw,
            seed=seed,
            profile=profile,
            crop_mode=crop_mode,
        ),
        profile,
    )


def _in_closed_range(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _validate_probability_outcome(value: bool, probability: float, name: str) -> None:
    if probability == 0.0 and value:
        raise ValueError(f"{name} impossible under zero probability")
    if probability == 1.0 and not value:
        raise ValueError(f"{name} impossible under unit probability")


def validate_u2flow_recipe_v1(
    recipe: U2FlowAugmentationRecipeV1,
    profile: U2FlowAugmentationProfileV1,
) -> np.ndarray:
    """Fail closed on provenance, parameter, matrix, and support drift."""

    if not isinstance(recipe, U2FlowAugmentationRecipeV1):
        raise ValueError("recipe must be a typed U2Flow recipe")
    if not isinstance(profile, U2FlowAugmentationProfileV1):
        raise ValueError("profile must be a typed U2Flow profile")
    expected_profile_hash = _canonical_sha256(profile.as_dict(include_hash=False))
    if expected_profile_hash != profile.profile_hash:
        raise ValueError("U2Flow augmentation profile hash drift")
    if recipe.profile_hash != profile.profile_hash:
        raise ValueError("recipe/profile binding drift")
    if recipe.spatial.crop_hw != profile.crop_hw:
        raise ValueError("recipe crop drifted from profile")
    spatial = recipe.spatial
    _validate_probability_outcome(
        spatial.horizontal_flip,
        profile.horizontal_flip_probability,
        "horizontal flip",
    )
    _validate_probability_outcome(
        spatial.vertical_flip,
        profile.vertical_flip_probability,
        "vertical flip",
    )
    _validate_probability_outcome(spatial.swap_endpoints, profile.swap_probability, "swap")
    _validate_probability_outcome(
        spatial.affine_enabled, profile.affine_probability, "affine"
    )
    if spatial.affine_enabled:
        if not _in_closed_range(spatial.rotation_degrees, profile.rotation_degrees_range):
            raise ValueError("rotation lies outside profile")
        if not _in_closed_range(spatial.scale, profile.scale_range):
            raise ValueError("scale lies outside profile")
        tx_fraction = spatial.translate_xy_px[0] / recipe.input_hw[1]
        ty_fraction = spatial.translate_xy_px[1] / recipe.input_hw[0]
        if not _in_closed_range(tx_fraction, profile.translate_x_fraction_range):
            raise ValueError("x translation lies outside profile")
        if not _in_closed_range(ty_fraction, profile.translate_y_fraction_range):
            raise ValueError("y translation lies outside profile")
    expected_forward = _compose_spatial_matrix(
        recipe.input_hw,
        spatial.crop_hw,
        spatial.crop_top,
        spatial.crop_left,
        spatial.horizontal_flip,
        spatial.vertical_flip,
        spatial.rotation_degrees,
        spatial.scale,
        spatial.translate_xy_px,
    )
    if not np.allclose(
        _matrix_array(spatial.input_to_output),
        expected_forward,
        atol=1e-9,
        rtol=1e-9,
    ):
        raise ValueError("spatial affine parameters/matrix drift")
    for appearance in (recipe.first_appearance, recipe.second_appearance):
        for value, bounds, name in (
            (appearance.brightness, profile.brightness_range, "brightness"),
            (appearance.contrast, profile.contrast_range, "contrast"),
            (appearance.saturation, profile.saturation_range, "saturation"),
        ):
            if not _in_closed_range(value, bounds):
                raise ValueError(f"{name} lies outside profile")
        _validate_probability_outcome(
            appearance.gaussian_blur,
            profile.gaussian_blur_probability,
            "Gaussian blur",
        )
        if appearance.gaussian_blur and not _in_closed_range(
            appearance.gaussian_blur_sigma, profile.gaussian_blur_sigma_range
        ):
            raise ValueError("Gaussian blur sigma lies outside profile")
    erasing = recipe.last_frame_erasing
    _validate_probability_outcome(
        erasing.enabled,
        profile.last_frame_erasing_probability,
        "last-frame erasing",
    )
    if erasing.enabled:
        area_fraction = (erasing.height * erasing.width) / float(
            profile.crop_hw[0] * profile.crop_hw[1]
        )
        # Pixel rounding and clipping can move the realized fraction slightly.
        tolerance = 2.0 / min(profile.crop_hw)
        if not (
            profile.erase_area_fraction_range[0] - tolerance
            <= area_fraction
            <= profile.erase_area_fraction_range[1] + tolerance
        ):
            raise ValueError("erase area lies outside profile")
    support = valid_support_from_recipe_v1(recipe)
    if _array_sha256(support) != recipe.valid_support_hash:
        raise ValueError("valid support hash drift")
    expected_hash = _canonical_sha256(recipe.as_dict(include_hash=False))
    if expected_hash != recipe.recipe_hash:
        raise ValueError("recipe hash drift")
    return support


def _validate_image(value: np.ndarray, input_hw: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8")
    if array.ndim != 3 or array.shape != (input_hw[0], input_hw[1], 3):
        raise ValueError(f"{name} must have shape HxWx3 matching recipe input_hw")
    return array


def _apply_spatial(image: np.ndarray, spatial: SpatialAffineRecipeV1) -> np.ndarray:
    output_height, output_width = spatial.crop_hw
    if not spatial.affine_enabled:
        result = image[
            spatial.crop_top : spatial.crop_top + output_height,
            spatial.crop_left : spatial.crop_left + output_width,
        ]
        if spatial.horizontal_flip:
            result = np.fliplr(result)
        if spatial.vertical_flip:
            result = np.flipud(result)
        return np.ascontiguousarray(result, dtype=np.uint8)
    inverse = _matrix_array(spatial.output_to_input)
    coefficients = tuple(float(value) for value in inverse[:2].ravel())
    transformed = Image.fromarray(image, mode="RGB").transform(
        (output_width, output_height),
        _PIL_AFFINE,
        coefficients,
        resample=_PIL_BILINEAR,
        fillcolor=(0, 0, 0),
    )
    return np.asarray(transformed, dtype=np.uint8).copy()


def _apply_appearance(image: np.ndarray, recipe: AppearanceRecipeV1) -> np.ndarray:
    result = Image.fromarray(image, mode="RGB")
    result = ImageEnhance.Brightness(result).enhance(recipe.brightness)
    result = ImageEnhance.Contrast(result).enhance(recipe.contrast)
    result = ImageEnhance.Color(result).enhance(recipe.saturation)
    if recipe.gaussian_blur:
        result = result.filter(ImageFilter.GaussianBlur(radius=recipe.gaussian_blur_sigma))
    return np.asarray(result, dtype=np.uint8).copy()


def apply_u2flow_recipe_v1(
    first: np.ndarray,
    second: np.ndarray,
    recipe: U2FlowAugmentationRecipeV1,
    *,
    profile: U2FlowAugmentationProfileV1 | None = None,
) -> AugmentedPairV1:
    """Apply a validated recipe to an in-memory uint8 RGB frame pair."""

    selected = SINTEL_U2FLOW_AUGMENTATION_PROFILE_V1 if profile is None else profile
    support = validate_u2flow_recipe_v1(recipe, selected)
    first_array = _validate_image(first, recipe.input_hw, "first frame")
    second_array = _validate_image(second, recipe.input_hw, "second frame")
    spatial_first = _apply_spatial(first_array, recipe.spatial)
    spatial_second = _apply_spatial(second_array, recipe.spatial)
    if recipe.spatial.swap_endpoints:
        spatial_first, spatial_second = spatial_second, spatial_first
    output_first = _apply_appearance(spatial_first, recipe.first_appearance)
    output_second = _apply_appearance(spatial_second, recipe.second_appearance)
    erasing = recipe.last_frame_erasing
    if erasing.enabled:
        output_second = output_second.copy()
        output_second[
            erasing.top : erasing.top + erasing.height,
            erasing.left : erasing.left + erasing.width,
        ] = np.asarray(erasing.fill_rgb, dtype=np.uint8)
    return AugmentedPairV1(output_first, output_second, support.copy(), recipe)


def augment_pair_v1(
    first: np.ndarray,
    second: np.ndarray,
    *,
    seed: int,
    profile: U2FlowAugmentationProfileV1 | None = None,
) -> AugmentedPairV1:
    """Sample and apply one deterministic U²Flow-style recipe."""

    first_array = np.asarray(first)
    if first_array.ndim != 3:
        raise ValueError("first frame must be HxWx3")
    selected = SINTEL_U2FLOW_AUGMENTATION_PROFILE_V1 if profile is None else profile
    recipe = sample_u2flow_recipe_v1(
        first_array.shape[:2], seed=seed, profile=selected
    )
    return apply_u2flow_recipe_v1(first, second, recipe, profile=selected)


def transform_points_v1(
    points_xy: np.ndarray, matrix: Sequence[float]
) -> np.ndarray:
    """Apply a stored affine matrix to Nx2 points (use its inverse to undo)."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx2 array")
    transform = _matrix_array(_matrix_tuple(matrix, "point transform matrix"))
    if not np.allclose(transform[2], (0.0, 0.0, 1.0), atol=1e-12, rtol=0.0):
        raise ValueError("point transform matrix must be affine")
    homogeneous = np.concatenate(
        (points, np.ones((points.shape[0], 1), dtype=np.float64)), axis=1
    )
    return (homogeneous @ transform.T)[:, :2]


SINTEL_U2FLOW_AUGMENTATION_PROFILE_V1 = U2FlowAugmentationProfileV1()


__all__ = [
    "AppearanceRecipeV1",
    "AugmentedPairV1",
    "LastFrameErasingRecipeV1",
    "SINTEL_OFFICIAL_CODE_CROP_PROVENANCE",
    "SINTEL_PAPER_REPORTED_CROP_HW",
    "SINTEL_U2FLOW_AUGMENTATION_PROFILE_V1",
    "SpatialAffineRecipeV1",
    "U2FLOW_PROFILE_SCHEMA_V1",
    "U2FLOW_RECIPE_SCHEMA_V1",
    "U2FlowAugmentationProfileV1",
    "U2FlowAugmentationRecipeV1",
    "apply_u2flow_recipe_v1",
    "augment_pair_v1",
    "derive_u2flow_view_seed_v1",
    "held_out_u2flow_profile_v1",
    "sample_u2flow_role_recipe_v1",
    "sample_u2flow_recipe_v1",
    "transform_points_v1",
    "valid_support_from_recipe_v1",
    "validate_u2flow_recipe_v1",
]
