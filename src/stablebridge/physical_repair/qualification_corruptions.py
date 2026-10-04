"""Deterministic controlled corruptions for Wave-1 qualification cohorts.

The generators use only a frozen component identity and source RGB arrays.
They do not inspect native flow, task ground truth, matcher predictions, action
eligibility, or outcomes.  The corruption receipt is generation metadata and
must never be supplied to action discovery or to the matcher.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from .blur_parameter_certificates import _kernel


CONTROLLED_CORRUPTION_SCHEMA_V1 = "stablebridge-controlled-corruption/v1"
CONTROLLED_CORRUPTION_STRATA_V1 = (
    "paired_impulse_median3",
    "paired_additive_wiener3",
    "common_disk",
    "common_gaussian",
    "common_motion",
    "jpeg_qcell_v3",
    "jpeg_codec_path_v4",
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
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
        raise ValueError("controlled corruption needs equal HxWx3 uint8 endpoints >=256px")
    return np.ascontiguousarray(first), np.ascontiguousarray(second)


def _seed(component_id: str, stratum: str) -> int:
    if not isinstance(component_id, str) or not component_id.strip():
        raise ValueError("component_id must be a nonempty string")
    digest = hashlib.sha256(
        f"{component_id}\0{stratum}\0controlled-corruption-v1".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _tile_origins(
    shape: tuple[int, int], tile: int, count: int, seed: int, *, margin: int = 0,
) -> tuple[tuple[int, int], ...]:
    rows = shape[0] // tile
    columns = shape[1] // tile
    candidates = [
        (row * tile, column * tile)
        for row in range(margin, rows - margin)
        for column in range(margin, columns - margin)
    ]
    if len(candidates) < count:
        raise ValueError("image has too few full tiles for frozen corruption recipe")
    generator = np.random.default_rng(seed)
    selected = generator.choice(len(candidates), size=count, replace=False)
    return tuple(sorted(candidates[int(index)] for index in selected))


def _impulse(
    image: np.ndarray, component_id: str, stratum: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    seed = _seed(component_id, stratum)
    origins = _tile_origins(image.shape[:2], 64, 4, seed)
    output = image.copy()
    support = np.zeros(image.shape[:2], dtype=bool)
    generator = np.random.default_rng(seed ^ 0x1A2B3C4D)
    coordinates = []
    for tile_index, (y0, x0) in enumerate(origins):
        flat = generator.choice(64 * 64, size=12, replace=False)
        for sample_index, value in enumerate(flat):
            y, x = y0 + int(value // 64), x0 + int(value % 64)
            output[y, x] = 0 if (tile_index + sample_index) % 2 == 0 else 255
            support[y, x] = True
            coordinates.append([y, x])
    return output, support, {
        "generator": "content_independent_local_salt_pepper_v1",
        "tile_size_px": 64,
        "tile_origins_y_x": [list(value) for value in origins],
        "sample_coordinates_y_x": coordinates,
        "samples_per_tile": 12,
        "values_uint8": [0, 255],
    }


def _additive(
    image: np.ndarray, component_id: str, stratum: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    seed = _seed(component_id, stratum)
    rows = image.shape[0] // 32
    columns = image.shape[1] // 32
    candidates = [
        (row, column)
        for row in range(2, rows - 2)
        for column in range(2, columns - 3)
    ]
    if not candidates:
        raise ValueError("image has no guarded adjacent 32px tile pair")
    row, column = candidates[seed % len(candidates)]
    origins = ((row * 32, column * 32), (row * 32, (column + 1) * 32))
    support = np.zeros(image.shape[:2], dtype=bool)
    for y0, x0 in origins:
        support[y0:y0 + 32, x0:x0 + 32] = True
    generator = np.random.default_rng(seed ^ 0x5EEDADD)
    noise = generator.normal(0.0, 25.0, (int(support.sum()), 3))
    output = image.copy()
    output[support] = np.clip(
        np.rint(image[support].astype(np.float64) + noise), 0, 255,
    ).astype(np.uint8)
    return output, support, {
        "generator": "content_independent_local_additive_gaussian_v1",
        "tile_size_px": 32,
        "tile_origins_y_x": [list(value) for value in origins],
        "sigma_uint8": 25.0,
        "clip_uint8": True,
    }


def _blur(
    image: np.ndarray, component_id: str, stratum: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    family_parameters: dict[str, tuple[str, tuple[float, ...]]] = {
        "common_disk": ("disk", (6.0,)),
        "common_gaussian": ("gaussian", (2.0,)),
        "common_motion": ("motion", (11.0, 0.0)),
    }
    family, parameter = family_parameters[stratum]
    seed = _seed(component_id, stratum)
    origins = _tile_origins(image.shape[:2], 64, 4, seed, margin=1)
    kernel = _kernel(family, parameter)
    proposal = cv2.filter2D(
        image, -1, kernel, borderType=cv2.BORDER_REFLECT_101,
    )
    support = np.zeros(image.shape[:2], dtype=bool)
    for y0, x0 in origins:
        support[y0:y0 + 64, x0:x0 + 64] = True
    output = image.copy()
    output[support] = proposal[support]
    return output, support, {
        "generator": "local_blur_fixture_v1",
        "family": family,
        "parameter": list(parameter),
        "tile_size_px": 64,
        "tile_origins_y_x": [list(value) for value in origins],
    }


def _jpeg(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stream = io.BytesIO()
    Image.fromarray(image).save(
        stream, format="JPEG", quality=6, subsampling=2,
    )
    stream.seek(0)
    proposal = np.asarray(Image.open(stream).convert("RGB"), dtype=np.uint8)
    width = max(64, (image.shape[1] // 2 // 64) * 64)
    if width >= image.shape[1]:
        raise ValueError("image too narrow for partial aligned JPEG corruption")
    support = np.zeros(image.shape[:2], dtype=bool)
    support[:, :width] = True
    output = image.copy()
    output[:, :width] = proposal[:, :width]
    return np.ascontiguousarray(output), support, {
        "generator": "partial_aligned_ijg_jpeg_v1",
        "ijg_quality": 6,
        "subsampling": 2,
        "region_xyxy": [0, 0, width, image.shape[0]],
    }


@dataclass(frozen=True)
class ControlledCorruptionReceiptV1:
    component_id: str
    mechanism_stratum: str
    input_pair_sha256s: tuple[str, str]
    output_pair_sha256s: tuple[str, str]
    changed_support_sha256s: tuple[str, str]
    changed_pixels: tuple[int, int]
    recipe: dict[str, Any]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.mechanism_stratum not in CONTROLLED_CORRUPTION_STRATA_V1:
            raise ValueError("unknown controlled-corruption stratum")
        payload = self.as_dict(include_hash=False)
        expected = _canonical_sha256(payload)
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("controlled-corruption receipt hash drift")
        object.__setattr__(self, "receipt_sha256", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema": CONTROLLED_CORRUPTION_SCHEMA_V1,
            "component_id": self.component_id,
            "mechanism_stratum": self.mechanism_stratum,
            "input_pair_sha256s": list(self.input_pair_sha256s),
            "output_pair_sha256s": list(self.output_pair_sha256s),
            "changed_support_sha256s": list(self.changed_support_sha256s),
            "changed_pixels": list(self.changed_pixels),
            "recipe": self.recipe,
            "forbidden_consumer": ["action_discovery", "matcher"],
            "ground_truth_read": False,
            "native_flow_read": False,
            "matcher_output_read": False,
            "action_outcome_read": False,
            "scientific_qualification": False,
        }
        if include_hash:
            result["receipt_sha256"] = self.receipt_sha256
        return result


@dataclass(frozen=True)
class ControlledCorruptionResultV1:
    first: np.ndarray
    second: np.ndarray
    supports: tuple[np.ndarray, np.ndarray]
    receipt: ControlledCorruptionReceiptV1


def materialize_controlled_corruption_v1(
    *,
    first: np.ndarray,
    second: np.ndarray,
    component_id: str,
    mechanism_stratum: str,
) -> ControlledCorruptionResultV1:
    if mechanism_stratum not in CONTROLLED_CORRUPTION_STRATA_V1:
        raise ValueError("unknown controlled-corruption stratum")
    first, second = _validate_pair(first, second)
    input_hashes = (_array_sha256(first), _array_sha256(second))
    if mechanism_stratum == "paired_impulse_median3":
        changed, declared, recipe = _impulse(first, component_id, mechanism_stratum)
    elif mechanism_stratum == "paired_additive_wiener3":
        changed, declared, recipe = _additive(first, component_id, mechanism_stratum)
    elif mechanism_stratum in {"common_disk", "common_gaussian", "common_motion"}:
        changed, declared, recipe = _blur(first, component_id, mechanism_stratum)
    else:
        changed, declared, recipe = _jpeg(first)
    actual = np.any(changed != first, axis=2)
    if not actual.any() or bool(np.any(actual & ~declared)):
        raise RuntimeError("controlled corruption escaped support or made no change")
    output_first = np.ascontiguousarray(changed)
    output_second = second.copy()
    supports = (np.ascontiguousarray(actual), np.zeros(actual.shape, dtype=bool))
    receipt = ControlledCorruptionReceiptV1(
        component_id=component_id,
        mechanism_stratum=mechanism_stratum,
        input_pair_sha256s=input_hashes,
        output_pair_sha256s=(
            _array_sha256(output_first), _array_sha256(output_second),
        ),
        changed_support_sha256s=(
            _array_sha256(supports[0]), _array_sha256(supports[1]),
        ),
        changed_pixels=(int(supports[0].sum()), 0),
        recipe=recipe,
    )
    return ControlledCorruptionResultV1(
        first=output_first,
        second=output_second,
        supports=supports,
        receipt=receipt,
    )


__all__ = [
    "CONTROLLED_CORRUPTION_SCHEMA_V1",
    "CONTROLLED_CORRUPTION_STRATA_V1",
    "ControlledCorruptionReceiptV1",
    "ControlledCorruptionResultV1",
    "materialize_controlled_corruption_v1",
]
