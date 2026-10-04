"""Versioned seven-stratum controlled corruptions for qualification.

V1 was structurally deterministic but too weak for the frozen eligibility
gates: sparse blur tiles could not populate the held-out fold, the impulse
count sat below its detector threshold, partial JPEG diluted phase evidence,
and additive noise did not reliably isolate the Wiener branch.  V2 changes
only the controlled intervention.  Runtime action identities, operators,
supports, strengths, and selector semantics are untouched.

The generator may read the source RGB pair and frozen component/pair identity.
It never reads flow, task GT, matcher output, action eligibility, or outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .blur_parameter_certificates import _kernel


CONTROLLED_CORRUPTION_SCHEMA_V2 = "stablebridge-controlled-corruption/v2"
CONTROLLED_CORRUPTION_STRATA_V2 = (
    "paired_impulse_median3",
    "paired_additive_wiener3",
    "common_disk",
    "common_gaussian",
    "common_motion",
    "jpeg_qcell_v3",
    "jpeg_codec_path_v4",
)

RECIPE_PARAMETERS_V2: dict[str, dict[str, Any]] = {
    "paired_impulse_median3": {
        "generator": "balanced_local_salt_pepper_v2",
        "endpoint": "first",
        "tile_size_px": 64,
        "active_tiles": 4,
        "samples_per_active_tile": 64,
        "values_uint8": [0, 255],
    },
    "paired_additive_wiener3": {
        "generator": "paired_connected_additive_gaussian_v2",
        "endpoint": "both",
        "tile_size_px": 32,
        "active_tile_shape": [2, 2],
        "active_tiles_per_endpoint": 4,
        "sigma_uint8": 60.0,
        "anti_impulse_preconditioner_clip_uint8": [2, 253],
        "independent_endpoint_seeds": True,
    },
    "common_disk": {
        "generator": "full_complete_tile_blur_v2",
        "endpoint": "first",
        "family": "disk",
        "parameter": [8.0],
        "tile_size_px": 64,
    },
    "common_gaussian": {
        "generator": "full_complete_tile_blur_v2",
        "endpoint": "first",
        "family": "gaussian",
        "parameter": [3.0],
        "tile_size_px": 64,
    },
    "common_motion": {
        "generator": "full_complete_tile_blur_v2",
        "endpoint": "first",
        "family": "motion",
        "parameter": [17.0, 30.0],
        "tile_size_px": 64,
    },
    "jpeg_qcell_v3": {
        "generator": "full_endpoint_ijg_jpeg_v2",
        "endpoint": "first",
        "ijg_quality": 6,
        "subsampling": 2,
    },
    "jpeg_codec_path_v4": {
        "generator": "full_endpoint_ijg_jpeg_v2",
        "endpoint": "first",
        "ijg_quality": 6,
        "subsampling": 2,
        "same_degraded_input_as": "jpeg_qcell_v3",
    },
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_pair(
    first: np.ndarray, second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first)
    second = np.asarray(second)
    if (
        first.dtype != np.uint8
        or first.ndim != 3
        or first.shape[2] != 3
        or second.dtype != np.uint8
        or second.shape != first.shape
        or min(first.shape[:2]) < 256
    ):
        raise ValueError("controlled corruption v2 needs equal HxWx3 uint8 endpoints >=256px")
    return np.ascontiguousarray(first), np.ascontiguousarray(second)


def _identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{name} must be a bounded nonempty string")
    return value


def _seed(
    component_id: str,
    physical_pair_id: str,
    stratum: str,
    endpoint: str,
) -> int:
    fields = (
        _identity(component_id, "component_id"),
        _identity(physical_pair_id, "physical_pair_id"),
        stratum,
        endpoint,
        "controlled-corruption-v2",
    )
    digest = hashlib.sha256("\0".join(fields).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _tile_origins(
    shape: tuple[int, int], tile: int, count: int, seed: int,
) -> tuple[tuple[int, int], ...]:
    rows, columns = shape[0] // tile, shape[1] // tile
    candidates = [
        (row * tile, column * tile)
        for row in range(rows) for column in range(columns)
    ]
    if len(candidates) < count:
        raise ValueError("image has too few full tiles for corruption v2")
    generator = np.random.default_rng(seed)
    selected = generator.choice(len(candidates), size=count, replace=False)
    return tuple(sorted(candidates[int(index)] for index in selected))


def _impulse(
    image: np.ndarray,
    component_id: str,
    physical_pair_id: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    recipe = dict(RECIPE_PARAMETERS_V2["paired_impulse_median3"])
    seed = _seed(
        component_id, physical_pair_id, "paired_impulse_median3", "first",
    )
    origins = _tile_origins(image.shape[:2], 64, 4, seed)
    output = image.copy()
    declared = np.zeros(image.shape[:2], dtype=bool)
    generator = np.random.default_rng(seed ^ 0xA17E2D5B)
    coordinates = []
    for tile_index, (y0, x0) in enumerate(origins):
        flat = generator.choice(64 * 64, size=64, replace=False)
        for sample_index, value in enumerate(flat):
            y, x = y0 + int(value // 64), x0 + int(value % 64)
            output[y, x] = (
                0 if (tile_index + sample_index) % 2 == 0 else 255
            )
            declared[y, x] = True
            coordinates.append([y, x])
    recipe.update({
        "tile_origins_y_x": [list(value) for value in origins],
        "sample_coordinates_y_x": coordinates,
        "requested_impulse_pixels": 256,
        "polarity_balance_requested": [128, 128],
        "eligibility_design_minimum_detected_pixels": 32,
        "eligibility_design_minimum_occupied_tiles": 4,
    })
    return output, declared, recipe


def _connected_block_origin(
    shape: tuple[int, int], seed: int,
) -> tuple[int, int]:
    rows, columns = shape[0] // 32, shape[1] // 32
    candidates = [
        (row, column)
        for row in range(2, rows - 3)
        for column in range(2, columns - 3)
    ]
    if not candidates:
        raise ValueError("image has no guarded 2x2 additive tile block")
    return candidates[seed % len(candidates)]


def _additive_endpoint(
    image: np.ndarray,
    component_id: str,
    physical_pair_id: str,
    endpoint: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    seed = _seed(
        component_id, physical_pair_id, "paired_additive_wiener3", endpoint,
    )
    row, column = _connected_block_origin(image.shape[:2], seed)
    origins = tuple(
        ((row + dy) * 32, (column + dx) * 32)
        for dy in range(2) for dx in range(2)
    )
    noise_support = np.zeros(image.shape[:2], dtype=bool)
    for y0, x0 in origins:
        noise_support[y0:y0 + 32, x0:x0 + 32] = True

    # Eliminate exact 0/255 extrema before adding diffuse noise so the frozen
    # impulse-priority dispatcher cannot mask the intended additive branch.
    base = np.clip(image, 2, 253).astype(np.uint8)
    preconditioner_support = np.any(base != image, axis=2)
    generator = np.random.default_rng(seed ^ 0x60ADD17E)
    noise = generator.normal(0.0, 60.0, (int(noise_support.sum()), 3))
    output = base.copy()
    output[noise_support] = np.clip(
        np.rint(base[noise_support].astype(np.float64) + noise), 2, 253,
    ).astype(np.uint8)
    declared = preconditioner_support | noise_support
    return output, declared, {
        "endpoint": endpoint,
        "tile_origins_y_x": [list(value) for value in origins],
        "noise_tile_pixels": int(noise_support.sum()),
        "preconditioner_changed_pixels": int(preconditioner_support.sum()),
        "noise_support_sha256": _array_sha256(noise_support),
        "preconditioner_support_sha256": _array_sha256(preconditioner_support),
    }


def _additive(
    first: np.ndarray,
    second: np.ndarray,
    component_id: str,
    physical_pair_id: str,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any],
]:
    output_first, declared_first, first_details = _additive_endpoint(
        first, component_id, physical_pair_id, "first",
    )
    output_second, declared_second, second_details = _additive_endpoint(
        second, component_id, physical_pair_id, "second",
    )
    recipe = dict(RECIPE_PARAMETERS_V2["paired_additive_wiener3"])
    recipe["endpoint_details"] = {
        "first": first_details,
        "second": second_details,
    }
    recipe["eligibility_design"] = {
        "impulse_endpoints_required_rejected": ["first", "second"],
        "minimum_connected_tiles": 2,
        "minimum_robust_z": 4.0,
        "maximum_active_tile_fraction": 0.8,
    }
    return (
        output_first, output_second,
        declared_first, declared_second, recipe,
    )


def _kernel_second_moments(kernel: np.ndarray) -> tuple[float, float]:
    kernel = np.asarray(kernel, dtype=np.float64)
    yy, xx = np.mgrid[:kernel.shape[0], :kernel.shape[1]]
    cy = float(np.sum(yy * kernel))
    cx = float(np.sum(xx * kernel))
    return (
        float(np.sum((yy - cy) ** 2 * kernel)),
        float(np.sum((xx - cx) ** 2 * kernel)),
    )


def _blur(
    image: np.ndarray, stratum: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    recipe = dict(RECIPE_PARAMETERS_V2[stratum])
    family = str(recipe["family"])
    parameter = tuple(float(value) for value in recipe["parameter"])
    kernel = _kernel(family, parameter)
    proposal = cv2.filter2D(
        image, -1, kernel, borderType=cv2.BORDER_REFLECT_101,
    )
    complete_height = (image.shape[0] // 64) * 64
    complete_width = (image.shape[1] // 64) * 64
    declared = np.zeros(image.shape[:2], dtype=bool)
    declared[:complete_height, :complete_width] = True
    output = image.copy()
    output[declared] = proposal[declared]
    moment_y, moment_x = _kernel_second_moments(kernel)
    recipe.update({
        "complete_tile_extent_xyxy": [0, 0, complete_width, complete_height],
        "complete_tile_rows": complete_height // 64,
        "complete_tile_columns": complete_width // 64,
        "complete_tile_count": (complete_height // 64) * (complete_width // 64),
        "kernel_shape": list(kernel.shape),
        "kernel_second_moment_y_px2": moment_y,
        "kernel_second_moment_x_px2": moment_x,
        "equivalent_gaussian_sigma_y_px": float(np.sqrt(moment_y)),
        "equivalent_gaussian_sigma_x_px": float(np.sqrt(moment_x)),
        "exact_family_must_be_unique_direct_winner": True,
        "minimum_recoverable_regions": 4,
        "minimum_information_median": 0.05,
    })
    return output, declared, recipe


def _jpeg(
    image: np.ndarray, stratum: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    recipe = dict(RECIPE_PARAMETERS_V2[stratum])
    stream = io.BytesIO()
    Image.fromarray(image).save(
        stream, format="JPEG", quality=6, subsampling=2,
    )
    stream.seek(0)
    output = np.ascontiguousarray(
        np.asarray(Image.open(stream).convert("RGB"), dtype=np.uint8)
    )
    declared = np.ones(image.shape[:2], dtype=bool)
    recipe.update({
        "region_xyxy": [0, 0, image.shape[1], image.shape[0]],
        "full_endpoint": True,
        "eligibility_design_minimum_validation_parity_tiles": 4,
        "eligibility_design_minimum_active_macro_tiles": 4,
        "qualification_requires_exact_target_action_executed": True,
    })
    return output, declared, recipe


@dataclass(frozen=True)
class ControlledCorruptionReceiptV2:
    component_id: str
    physical_pair_id: str
    mechanism_stratum: str
    input_pair_sha256s: tuple[str, str]
    output_pair_sha256s: tuple[str, str]
    changed_support_sha256s: tuple[str, str]
    changed_pixels: tuple[int, int]
    recipe: dict[str, Any]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _identity(self.component_id, "component_id")
        _identity(self.physical_pair_id, "physical_pair_id")
        if self.mechanism_stratum not in CONTROLLED_CORRUPTION_STRATA_V2:
            raise ValueError("unknown controlled-corruption v2 stratum")
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("controlled-corruption v2 receipt hash drift")
        object.__setattr__(self, "receipt_sha256", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema": CONTROLLED_CORRUPTION_SCHEMA_V2,
            "component_id": self.component_id,
            "physical_pair_id": self.physical_pair_id,
            "mechanism_stratum": self.mechanism_stratum,
            "input_pair_sha256s": list(self.input_pair_sha256s),
            "output_pair_sha256s": list(self.output_pair_sha256s),
            "changed_support_sha256s": list(self.changed_support_sha256s),
            "changed_pixels": list(self.changed_pixels),
            "recipe": self.recipe,
            "consumer_firewall": {
                "corruption_receipt_forbidden_consumers": [
                    "native_matcher", "action_discovery", "action_matcher",
                ],
                "only_corrupted_rgb_crosses_native_matcher_boundary": True,
            },
            "ground_truth_read": False,
            "native_flow_read": False,
            "matcher_output_read": False,
            "action_eligibility_read": False,
            "action_outcome_read": False,
            "scientific_qualification": False,
        }
        if include_hash:
            result["receipt_sha256"] = self.receipt_sha256
        return result


@dataclass(frozen=True)
class ControlledCorruptionResultV2:
    first: np.ndarray
    second: np.ndarray
    supports: tuple[np.ndarray, np.ndarray]
    receipt: ControlledCorruptionReceiptV2


def materialize_controlled_corruption_v2(
    *,
    first: np.ndarray,
    second: np.ndarray,
    component_id: str,
    physical_pair_id: str,
    mechanism_stratum: str,
) -> ControlledCorruptionResultV2:
    """Materialize one frozen v2 intervention without reading outcomes."""

    if mechanism_stratum not in CONTROLLED_CORRUPTION_STRATA_V2:
        raise ValueError("unknown controlled-corruption v2 stratum")
    first, second = _validate_pair(first, second)
    component_id = _identity(component_id, "component_id")
    physical_pair_id = _identity(physical_pair_id, "physical_pair_id")
    input_hashes = (_array_sha256(first), _array_sha256(second))

    if mechanism_stratum == "paired_impulse_median3":
        output_first, declared_first, recipe = _impulse(
            first, component_id, physical_pair_id,
        )
        output_second, declared_second = second.copy(), np.zeros(
            second.shape[:2], dtype=bool,
        )
    elif mechanism_stratum == "paired_additive_wiener3":
        (
            output_first, output_second, declared_first, declared_second, recipe,
        ) = _additive(first, second, component_id, physical_pair_id)
    elif mechanism_stratum in {"common_disk", "common_gaussian", "common_motion"}:
        output_first, declared_first, recipe = _blur(first, mechanism_stratum)
        output_second, declared_second = second.copy(), np.zeros(
            second.shape[:2], dtype=bool,
        )
    else:
        output_first, declared_first, recipe = _jpeg(first, mechanism_stratum)
        output_second, declared_second = second.copy(), np.zeros(
            second.shape[:2], dtype=bool,
        )

    outputs = (
        np.ascontiguousarray(output_first), np.ascontiguousarray(output_second),
    )
    actual = (
        np.ascontiguousarray(np.any(outputs[0] != first, axis=2)),
        np.ascontiguousarray(np.any(outputs[1] != second, axis=2)),
    )
    declared = (
        np.ascontiguousarray(declared_first, dtype=bool),
        np.ascontiguousarray(declared_second, dtype=bool),
    )
    for endpoint, changed, allowed in zip(("first", "second"), actual, declared):
        if bool(np.any(changed & ~allowed)):
            raise RuntimeError(f"controlled corruption v2 escaped {endpoint} support")
    if not any(bool(mask.any()) for mask in actual):
        raise RuntimeError("controlled corruption v2 made no change")

    receipt = ControlledCorruptionReceiptV2(
        component_id=component_id,
        physical_pair_id=physical_pair_id,
        mechanism_stratum=mechanism_stratum,
        input_pair_sha256s=input_hashes,
        output_pair_sha256s=(
            _array_sha256(outputs[0]), _array_sha256(outputs[1]),
        ),
        changed_support_sha256s=(
            _array_sha256(actual[0]), _array_sha256(actual[1]),
        ),
        changed_pixels=(int(actual[0].sum()), int(actual[1].sum())),
        recipe=recipe,
    )
    return ControlledCorruptionResultV2(
        first=outputs[0], second=outputs[1], supports=actual, receipt=receipt,
    )


__all__ = [
    "CONTROLLED_CORRUPTION_SCHEMA_V2",
    "CONTROLLED_CORRUPTION_STRATA_V2",
    "RECIPE_PARAMETERS_V2",
    "ControlledCorruptionReceiptV2",
    "ControlledCorruptionResultV2",
    "materialize_controlled_corruption_v2",
]
