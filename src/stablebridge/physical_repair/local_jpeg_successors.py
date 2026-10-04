"""Versioned before-only local successors for Wave-1 JPEG policies.

The endpoint/quality certificate remains E132 plus the existing action-specific
held-out certificate.  This module adds the missing local authorization mask:
aligned 64 px macro-tiles whose observed, pre-action 8 px boundary excess is
strictly greater than one raw luma level.  Proposal changes cannot authorize
support.  The exact qcell or adaptive-codec proposal is hard-remasked to those
macro-tiles with the common inward feather compositor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .jpeg_action_certificates import MACRO_TILE, _boundary_excess
from .jpeg_codec_path_actions import (
    DYADIC_ACTION_STRENGTHS,
    _quantization_violation,
    jpeg_codec_path_action,
)
from .jpeg_quantization_actions import (
    DCT_ROUNDING_BOUND,
    jpeg_quantization_projected_deblock,
)
from .operators import blend_local_proposal


LOCAL_JPEG_QCELL_ACTION_ID_V2 = "jpeg_qcell_v3.local_macroblock_v2"
LOCAL_JPEG_CODEC_ACTION_ID_V2 = "jpeg_codec_path_v4.local_macroblock_v2"
LOCAL_JPEG_QCELL_OPERATOR_ID_V2 = "jpeg_qcell_local_macroblock_v2"
LOCAL_JPEG_CODEC_OPERATOR_ID_V2 = "jpeg_codec_path_local_macroblock_v2"
LOCAL_JPEG_SCHEMA_V2 = "stablebridge-local-jpeg-successor/v2"
BOUNDARY_EXCESS_THRESHOLD_255 = 1.0
MIN_ACTIVE_MACRO_TILES = 4
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
    "jpeg_certificate": (
        _ROOT / "src/stablebridge/physical_repair/jpeg_action_certificates.py",
        "d4eba4a15eef5a0ad4bc81f54a5c445c8720847ebdcc4314c01615b52be467fe",
    ),
    "presence_certificate": (
        _ROOT / "experiments/E132_jpeg_lattice_phase_certificate/certificate.py",
        "6920ae04d4143cc53c010b34208fc32651ec2febc572a60585edba4f277c19b1",
    ),
    "local_compositor": (
        _ROOT / "src/stablebridge/physical_repair/operators.py",
        "d638c3c0d7ba7f497b96749b4c81895a35055c16a2f1d85becd0a8b00f766c2e",
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
            raise RuntimeError(f"local JPEG source escaped workspace: {name}")
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"local JPEG source drift: {name}; expected {expected}, got {actual}"
            )
        result[name] = actual
    return result


def _validate_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if (
        image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
        or min(image.shape[:2]) < 2 * MACRO_TILE
    ):
        raise ValueError("expected sufficiently large uint8 RGB image")
    return np.ascontiguousarray(image)


def jpeg_macroblock_support_v2(
    image: np.ndarray,
) -> tuple[np.ndarray, tuple[Mapping[str, Any], ...]]:
    """Return an aligned support from pre-action boundary evidence only."""

    image = _validate_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    support = np.zeros(image.shape[:2], dtype=bool)
    records = []
    height, width = image.shape[:2]
    for y0 in range(0, height - MACRO_TILE + 1, MACRO_TILE):
        for x0 in range(0, width - MACRO_TILE + 1, MACRO_TILE):
            value = float(_boundary_excess(
                gray[y0:y0 + MACRO_TILE, x0:x0 + MACRO_TILE],
            ))
            active = value > BOUNDARY_EXCESS_THRESHOLD_255
            if active:
                support[y0:y0 + MACRO_TILE, x0:x0 + MACRO_TILE] = True
            records.append({
                "origin_y_x": [y0, x0],
                "boundary_excess_255": value,
                "active": active,
            })
    return np.ascontiguousarray(support), tuple(records)


@dataclass(frozen=True)
class LocalJPEGSuccessorV2:
    status: str
    action_id: str
    operator_id: str
    estimated_quality: int
    selected_strength: float
    output_rgb: np.ndarray
    support: np.ndarray
    active_macro_tiles: tuple[tuple[int, int], ...]
    receipt: Mapping[str, Any]


def build_local_jpeg_successor_v2(
    observed_rgb: np.ndarray,
    *,
    policy: str,
    endpoint: str,
    estimated_quality: int,
) -> LocalJPEGSuccessorV2:
    """Materialize qcell or adaptive codec only in authorized macro-tiles."""

    source_hashes = _verify_sources()
    image = _validate_image(observed_rgb)
    if policy not in {"qcell", "codec"} or endpoint not in {"first", "second"}:
        raise ValueError("invalid local JPEG policy or endpoint")
    if type(estimated_quality) is not int or not 1 <= estimated_quality <= 95:
        raise ValueError("estimated IJG quality must be an integer in [1,95]")
    if policy == "qcell":
        action_id = LOCAL_JPEG_QCELL_ACTION_ID_V2
        operator_id = LOCAL_JPEG_QCELL_OPERATOR_ID_V2
        proposal_record = jpeg_quantization_projected_deblock(
            image, estimated_quality=estimated_quality,
        )
        proposal = proposal_record.image
        selected_strength = 1.0
        attempted_strengths = (1.0,)
    else:
        action_id = LOCAL_JPEG_CODEC_ACTION_ID_V2
        operator_id = LOCAL_JPEG_CODEC_OPERATOR_ID_V2
        proposal_record = jpeg_codec_path_action(
            image, estimated_quality=estimated_quality,
        )
        proposal = proposal_record.image
        selected_strength = float(proposal_record.strength)
        attempted_strengths = tuple(map(float, proposal_record.attempted_strengths))
    support, tile_records = jpeg_macroblock_support_v2(image)
    active_tiles = tuple(
        tuple(map(int, row["origin_y_x"]))
        for row in tile_records if bool(row["active"])
    )
    base = {
        "schema": LOCAL_JPEG_SCHEMA_V2,
        "action_id": action_id,
        "operator_id": operator_id,
        "endpoint": endpoint,
        "estimated_ijg_quality": estimated_quality,
        "selected_strength": selected_strength,
        "attempted_strengths": list(attempted_strengths),
        "input_sha256": _array_sha256(image),
        "full_proposal_sha256": _array_sha256(proposal),
        "support_policy": {
            "macro_tile_px": MACRO_TILE,
            "jpeg_block_px": 8,
            "boundary_excess_threshold_255_strictly_above": (
                BOUNDARY_EXCESS_THRESHOLD_255
            ),
            "minimum_active_macro_tiles": MIN_ACTIVE_MACRO_TILES,
            "pre_action_only": True,
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
    if len(active_tiles) < MIN_ACTIVE_MACRO_TILES:
        base.update({
            "status": "UNSUPPORTED_INSUFFICIENT_ACTIVE_MACRO_TILES",
            "support_sha256": _array_sha256(support),
            "support_pixels": int(support.sum()),
        })
        base["receipt_sha256"] = _canonical_sha256(base)
        return LocalJPEGSuccessorV2(
            status=str(base["status"]), action_id=action_id,
            operator_id=operator_id, estimated_quality=estimated_quality,
            selected_strength=selected_strength, output_rgb=image.copy(),
            support=support, active_macro_tiles=active_tiles, receipt=base,
        )
    if policy == "codec" and selected_strength not in DYADIC_ACTION_STRENGTHS:
        base.update({
            "status": "UNSUPPORTED_CODEC_NATIVE_FALLBACK",
            "support_sha256": _array_sha256(support),
            "support_pixels": int(support.sum()),
        })
        base["receipt_sha256"] = _canonical_sha256(base)
        return LocalJPEGSuccessorV2(
            status=str(base["status"]), action_id=action_id,
            operator_id=operator_id, estimated_quality=estimated_quality,
            selected_strength=selected_strength, output_rgb=image.copy(),
            support=support, active_macro_tiles=active_tiles, receipt=base,
        )
    output, local_record = blend_local_proposal(
        image, proposal, support.astype(np.float32),
        operator_id=operator_id, endpoint=endpoint, feather_sigma=1.0,
    )
    changed = np.any(output != image, axis=2)
    violation = _quantization_violation(image, output, estimated_quality)
    if violation > DCT_ROUNDING_BOUND:
        raise RuntimeError("localized JPEG successor left the frozen qcell allowance")
    if np.any(changed & ~support):
        raise RuntimeError("localized JPEG successor escaped support")
    status = (
        "EXECUTED_LOCAL_MACROBLOCK_V2"
        if bool(changed.any()) else "IDENTITY_ONLY_LOCAL_MACROBLOCK_V2"
    )
    base.update({
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
        "local_action_record": asdict(local_record),
    })
    base["receipt_sha256"] = _canonical_sha256(base)
    return LocalJPEGSuccessorV2(
        status=status, action_id=action_id, operator_id=operator_id,
        estimated_quality=estimated_quality,
        selected_strength=selected_strength,
        output_rgb=np.ascontiguousarray(output), support=support,
        active_macro_tiles=active_tiles, receipt=base,
    )


__all__ = [
    "BOUNDARY_EXCESS_THRESHOLD_255",
    "LOCAL_JPEG_CODEC_ACTION_ID_V2",
    "LOCAL_JPEG_CODEC_OPERATOR_ID_V2",
    "LOCAL_JPEG_QCELL_ACTION_ID_V2",
    "LOCAL_JPEG_QCELL_OPERATOR_ID_V2",
    "LOCAL_JPEG_SCHEMA_V2",
    "LocalJPEGSuccessorV2",
    "build_local_jpeg_successor_v2",
    "jpeg_macroblock_support_v2",
]
