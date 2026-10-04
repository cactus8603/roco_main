"""Block-aligned local successors for the two frozen JPEG proposal paths.

E259 exposed a real contract error in v2: Gaussian pixel feathering is not a
closed operation in an aligned 8x8 DCT quantization cell.  V3 keeps the same
before-only 64px support rule, but composes at whole 8x8 blocks.  Every coded
block is therefore either the observed block or the corresponding block from
the already-certified full proposal.  No safety threshold is relaxed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .jpeg_codec_path_actions import (
    DYADIC_ACTION_STRENGTHS,
    _quantization_violation,
    jpeg_codec_path_action,
)
from .jpeg_quantization_actions import (
    BLOCK,
    DCT_ROUNDING_BOUND,
    jpeg_quantization_projected_deblock,
)
from .local_jpeg_successors import (
    MIN_ACTIVE_MACRO_TILES,
    jpeg_macroblock_support_v2,
)


LOCAL_JPEG_QCELL_ACTION_ID_V3 = "jpeg_qcell_v3.local_macroblock_v3"
LOCAL_JPEG_CODEC_ACTION_ID_V3 = "jpeg_codec_path_v4.local_macroblock_v3"
LOCAL_JPEG_QCELL_OPERATOR_ID_V3 = "jpeg_qcell_local_block_exact_v3"
LOCAL_JPEG_CODEC_OPERATOR_ID_V3 = "jpeg_codec_path_local_block_exact_v3"
LOCAL_JPEG_SCHEMA_V3 = "stablebridge-local-jpeg-successor/v3"
_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_BINDINGS = {
    "qcell": (
        _ROOT / "src/stablebridge/physical_repair/jpeg_quantization_actions.py",
        "0c953576e52160447098d99e148a787b49eaf7cbe1f4af6c89375ec147fdb575",
    ),
    "codec": (
        _ROOT / "src/stablebridge/physical_repair/jpeg_codec_path_actions.py",
        "be59a59ad5b292df4eafe17f6205739c22e31232012a75b5bef98d8c6e4acd53",
    ),
    "v2_before_only_support": (
        _ROOT / "src/stablebridge/physical_repair/local_jpeg_successors.py",
        "ba675f0e5e557d3104c41f86c22616c0113f88e0837ba11a35193eb8eb3c9505",
    ),
}


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _verify_sources() -> dict[str, str]:
    result = {}
    root = _ROOT.resolve()
    for name, (path, expected) in _SOURCE_BINDINGS.items():
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"local JPEG v3 source escaped workspace: {name}")
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"local JPEG v3 source drift: {name}; expected {expected}, got {actual}"
            )
        result[name] = actual
    return result


def _validate_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if (
        image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
        or min(image.shape[:2]) < 128
    ):
        raise ValueError("expected sufficiently large uint8 RGB image")
    return np.ascontiguousarray(image)


def _block_exact_composite(
    image: np.ndarray,
    proposal: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Choose an entire aligned 8x8 RGB block from one feasible endpoint."""

    image = _validate_image(image)
    proposal = np.asarray(proposal)
    support = np.asarray(support)
    if proposal.dtype != np.uint8 or proposal.shape != image.shape:
        raise ValueError("invalid JPEG v3 proposal")
    if support.dtype != np.bool_ or support.shape != image.shape[:2]:
        raise ValueError("invalid JPEG v3 support")
    height = (support.shape[0] // BLOCK) * BLOCK
    width = (support.shape[1] // BLOCK) * BLOCK
    core = support[:height, :width]
    blocks = core.reshape(
        height // BLOCK, BLOCK, width // BLOCK, BLOCK,
    ).transpose(0, 2, 1, 3)
    all_true = np.all(blocks, axis=(2, 3))
    all_false = ~np.any(blocks, axis=(2, 3))
    if not np.all(all_true | all_false):
        raise RuntimeError("JPEG v3 support split an 8x8 DCT block")
    if np.any(support[height:, :]) or np.any(support[:, width:]):
        raise RuntimeError("JPEG v3 support entered an incomplete DCT block")
    output = image.copy()
    output[support] = proposal[support]
    if not np.array_equal(output[~support], image[~support]):
        raise RuntimeError("JPEG v3 output escaped support")
    return np.ascontiguousarray(output)


@dataclass(frozen=True)
class LocalJPEGSuccessorV3:
    status: str
    action_id: str
    operator_id: str
    estimated_quality: int
    selected_strength: float
    output_rgb: np.ndarray
    support: np.ndarray
    active_macro_tiles: tuple[tuple[int, int], ...]
    receipt: Mapping[str, Any]


def build_local_jpeg_successor_v3(
    observed_rgb: np.ndarray,
    *,
    policy: str,
    endpoint: str,
    estimated_quality: int,
) -> LocalJPEGSuccessorV3:
    """Materialize qcell or codec path on whole authorized DCT blocks only."""

    source_hashes = _verify_sources()
    image = _validate_image(observed_rgb)
    if policy not in {"qcell", "codec"} or endpoint not in {"first", "second"}:
        raise ValueError("invalid local JPEG v3 policy or endpoint")
    if type(estimated_quality) is not int or not 1 <= estimated_quality <= 95:
        raise ValueError("estimated IJG quality must be an integer in [1,95]")
    if policy == "qcell":
        action_id = LOCAL_JPEG_QCELL_ACTION_ID_V3
        operator_id = LOCAL_JPEG_QCELL_OPERATOR_ID_V3
        proposal_record = jpeg_quantization_projected_deblock(
            image, estimated_quality=estimated_quality,
        )
        proposal = proposal_record.image
        selected_strength = 1.0
        attempted_strengths = (1.0,)
    else:
        action_id = LOCAL_JPEG_CODEC_ACTION_ID_V3
        operator_id = LOCAL_JPEG_CODEC_OPERATOR_ID_V3
        proposal_record = jpeg_codec_path_action(
            image, estimated_quality=estimated_quality,
        )
        proposal = proposal_record.image
        selected_strength = float(proposal_record.strength)
        attempted_strengths = tuple(map(float, proposal_record.attempted_strengths))

    support, tile_records = jpeg_macroblock_support_v2(image)
    support = np.ascontiguousarray(support, dtype=bool)
    active_tiles = tuple(
        tuple(map(int, row["origin_y_x"]))
        for row in tile_records if bool(row["active"])
    )
    base = {
        "schema": LOCAL_JPEG_SCHEMA_V3,
        "action_id": action_id,
        "operator_id": operator_id,
        "endpoint": endpoint,
        "estimated_ijg_quality": estimated_quality,
        "selected_strength": selected_strength,
        "attempted_strengths": list(attempted_strengths),
        "input_sha256": _array_sha256(image),
        "full_proposal_sha256": _array_sha256(proposal),
        "support_policy": {
            "inherited_before_only_support": "jpeg_macroblock_support_v2",
            "macro_tile_px": 64,
            "jpeg_block_px": BLOCK,
            "minimum_active_macro_tiles": MIN_ACTIVE_MACRO_TILES,
            "pre_action_only": True,
        },
        "composition_policy": {
            "id": "ALIGNED_8PX_BLOCK_EXACT_SELECTION_V1",
            "pixel_feather_sigma_px": 0.0,
            "closure_argument": (
                "each complete DCT block is byte-exactly observed or byte-exactly "
                "the frozen full proposal"
            ),
        },
        "active_macro_tiles": [list(value) for value in active_tiles],
        "tile_evidence_sha256": _canonical_sha256(tile_records),
        "source_manifest_sha256": _canonical_sha256(source_hashes),
        "runtime_inputs": ["observed_endpoint_rgb", "receipt_bound_quality"],
        "ground_truth_read": False,
        "task_outcome_read": False,
        "scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
    }

    def abstain(status: str) -> LocalJPEGSuccessorV3:
        value = {
            **base,
            "status": status,
            "support_sha256": _array_sha256(support),
            "support_pixels": int(support.sum()),
        }
        value["receipt_sha256"] = _canonical_sha256(value)
        return LocalJPEGSuccessorV3(
            status=status, action_id=action_id, operator_id=operator_id,
            estimated_quality=estimated_quality,
            selected_strength=selected_strength, output_rgb=image.copy(),
            support=support, active_macro_tiles=active_tiles, receipt=value,
        )

    if len(active_tiles) < MIN_ACTIVE_MACRO_TILES:
        return abstain("UNSUPPORTED_INSUFFICIENT_ACTIVE_MACRO_TILES")
    if policy == "codec" and selected_strength not in DYADIC_ACTION_STRENGTHS:
        return abstain("UNSUPPORTED_CODEC_NATIVE_FALLBACK")

    output = _block_exact_composite(image, proposal, support)
    changed = np.any(output != image, axis=2)
    violation = _quantization_violation(image, output, estimated_quality)
    if violation > DCT_ROUNDING_BOUND:
        raise RuntimeError("block-exact JPEG v3 left the frozen qcell allowance")
    if np.any(changed & ~support):
        raise RuntimeError("block-exact JPEG v3 escaped support")
    status = (
        "EXECUTED_LOCAL_BLOCK_EXACT_V3"
        if bool(changed.any()) else "IDENTITY_ONLY_LOCAL_BLOCK_EXACT_V3"
    )
    value = {
        **base,
        "status": status,
        "support_sha256": _array_sha256(support),
        "support_pixels": int(support.sum()),
        "output_sha256": _array_sha256(output),
        "changed_mask_sha256": _array_sha256(changed),
        "changed_pixels": int(changed.sum()),
        "changed_subset_of_support": True,
        "outside_support_byte_identity": bool(np.array_equal(
            output[~support], image[~support],
        )),
        "quantization_violation_max_dct": violation,
        "quantization_rounding_bound_dct": DCT_ROUNDING_BOUND,
        "proposal_receipt": {
            "selected_strength": selected_strength,
            "attempted_strengths": list(attempted_strengths),
            "proposal_quantization_violation_max": float(
                proposal_record.quantization_violation_max
            ),
        },
    }
    value["receipt_sha256"] = _canonical_sha256(value)
    return LocalJPEGSuccessorV3(
        status=status, action_id=action_id, operator_id=operator_id,
        estimated_quality=estimated_quality, selected_strength=selected_strength,
        output_rgb=output, support=support, active_macro_tiles=active_tiles,
        receipt=value,
    )


__all__ = [
    "LOCAL_JPEG_CODEC_ACTION_ID_V3",
    "LOCAL_JPEG_CODEC_OPERATOR_ID_V3",
    "LOCAL_JPEG_QCELL_ACTION_ID_V3",
    "LOCAL_JPEG_QCELL_OPERATOR_ID_V3",
    "LOCAL_JPEG_SCHEMA_V3",
    "LocalJPEGSuccessorV3",
    "build_local_jpeg_successor_v3",
]
