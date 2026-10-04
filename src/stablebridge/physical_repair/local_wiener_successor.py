"""Versioned local-noise Wiener successor for the Wave-1 additive policy.

The v1 operator detects localized noisy 32px tiles, then estimates one noise
sigma from the full endpoint.  On the canonical sparse-noisy-tile case that
global median is zero, making the Wiener gain one and the action identity-only.
This successor retains the before-only paired selector and exact active-tile
support, but uses each active tile's already-observed robust Laplacian sigma in
the Wiener shrinkage formula.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .noise_certificates import _tile_noise
from .operators import blend_local_proposal
from .paired_noise_certificates import select_paired_noise_action


LOCAL_WIENER_ACTION_ID_V2 = "paired_additive_wiener3.local_tile_sigma_v2"
LOCAL_WIENER_OPERATOR_ID_V2 = "wiener3_local_tile_sigma_v2"
LOCAL_WIENER_SCHEMA_V2 = "stablebridge-local-wiener-successor/v2"
_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_BINDINGS = {
    "noise_certificates": (
        _ROOT / "src/stablebridge/physical_repair/noise_certificates.py",
        "459d1e97b7b11cbb9bc58eed3ea9052f9d2f53d5b0437007a0cb99daab2bba55",
    ),
    "paired_noise_certificates": (
        _ROOT / "src/stablebridge/physical_repair/paired_noise_certificates.py",
        "6c7031197f81862d1f6959561f5b0d2d600bc235172ab2b38b8db054a0237848",
    ),
    "local_operators": (
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
    hashes = {}
    for name, (path, expected) in _SOURCE_BINDINGS.items():
        actual = hashlib.sha256(path.resolve(strict=True).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"local Wiener source drift: {name}; expected {expected}, got {actual}"
            )
        hashes[name] = actual
    return hashes


def _validate(
    first: np.ndarray, second: np.ndarray, flow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(first)
    second = np.asarray(second)
    flow = np.asarray(flow)
    if (
        first.dtype != np.uint8 or second.dtype != np.uint8
        or first.ndim != 3 or first.shape != second.shape or first.shape[2] != 3
        or flow.dtype != np.float32 or flow.shape != (*first.shape[:2], 2)
        or not np.isfinite(flow).all()
    ):
        raise ValueError("expected matching uint8 RGB pair and finite float32 flow")
    return (
        np.ascontiguousarray(first),
        np.ascontiguousarray(second),
        np.ascontiguousarray(flow),
    )


def _active_tile_support_and_sigma(
    image: np.ndarray, *, tile_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    tile_sigma, shape = _tile_noise(image, tile_size)
    baseline = float(np.median(tile_sigma))
    mad = float(np.median(np.abs(tile_sigma - baseline)))
    robust_scale = max(1.482602218505602 * mad, 0.5)
    active = ((tile_sigma - baseline) / robust_scale >= 4.0) & (tile_sigma >= 2.0)
    support = np.repeat(
        np.repeat(active, tile_size, axis=0), tile_size, axis=1,
    )[:shape[0], :shape[1]]
    sigma_map = np.repeat(
        np.repeat(tile_sigma, tile_size, axis=0), tile_size, axis=1,
    )[:shape[0], :shape[1]]
    sigma_map = np.where(support, sigma_map, 0.0).astype(np.float32)
    return (
        np.ascontiguousarray(support, dtype=bool),
        np.ascontiguousarray(sigma_map),
        {
            "tile_size_px": float(tile_size),
            "baseline_sigma_255": baseline,
            "robust_scale_sigma_255": robust_scale,
            "active_tile_count": float(active.sum()),
            "active_tile_fraction": float(active.mean()),
            "active_sigma_min_255": float(tile_sigma[active].min()) if np.any(active) else 0.0,
            "active_sigma_max_255": float(tile_sigma[active].max()) if np.any(active) else 0.0,
        },
    )


def _local_tile_wiener_proposal(
    image: np.ndarray, sigma_map: np.ndarray,
) -> np.ndarray:
    source = image.astype(np.float32)
    output = np.empty_like(source)
    noise = np.square(sigma_map.astype(np.float32))
    for channel in range(3):
        plane = source[..., channel]
        mean = cv2.blur(plane, (3, 3), borderType=cv2.BORDER_REFLECT_101)
        square = cv2.blur(
            plane * plane, (3, 3), borderType=cv2.BORDER_REFLECT_101,
        )
        variance = np.maximum(square - mean * mean, 0.0)
        gain = np.maximum(variance - noise, 0.0) / np.maximum(
            variance, np.maximum(noise, np.float32(1e-12)),
        )
        output[..., channel] = mean + gain * (plane - mean)
    return np.ascontiguousarray(np.clip(np.rint(output), 0, 255).astype(np.uint8))


@dataclass(frozen=True)
class LocalWienerSuccessorV2:
    status: str
    exact_endpoint: str | None
    first_rgb: np.ndarray
    second_rgb: np.ndarray
    first_support: np.ndarray
    second_support: np.ndarray
    receipt: dict[str, Any]


def build_local_wiener_successor_v2(
    observed_first_rgb: np.ndarray,
    observed_second_rgb: np.ndarray,
    observed_native_flow: np.ndarray,
) -> LocalWienerSuccessorV2:
    """Build the v2 child using before-only paired-noise evidence."""

    sources = _verify_sources()
    first, second, flow = _validate(
        observed_first_rgb, observed_second_rgb, observed_native_flow,
    )
    decision = select_paired_noise_action(first, second, flow)
    base = {
        "schema": LOCAL_WIENER_SCHEMA_V2,
        "action_id": LOCAL_WIENER_ACTION_ID_V2,
        "operator_id": LOCAL_WIENER_OPERATOR_ID_V2,
        "paired_decision": {
            "action": str(decision["action"]),
            "endpoint": str(decision["endpoint"]),
            "family": str(decision["family"]),
        },
        "source_hashes": sources,
        "input_hashes": {
            "first": _array_sha256(first),
            "second": _array_sha256(second),
            "native_flow": _array_sha256(flow),
        },
        "runtime_inputs": [
            "observed_first_rgb", "observed_second_rgb", "observed_native_flow",
        ],
        "ground_truth_read": False,
        "outcome_read": False,
        "production_authority": False,
    }
    empty = np.zeros(first.shape[:2], dtype=bool)
    if decision["action"] != "wiener3":
        receipt = {
            **base,
            "status": "TYPED_MISSING_PAIRED_SELECTOR_DID_NOT_AUTHORIZE_ADDITIVE",
            "exact_endpoint": None,
            "endpoint_records": [],
            "scientific_qualification": False,
            "selector_admission": False,
        }
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        return LocalWienerSuccessorV2(
            status=receipt["status"], exact_endpoint=None,
            first_rgb=first.copy(), second_rgb=second.copy(),
            first_support=empty.copy(), second_support=empty.copy(), receipt=receipt,
        )

    endpoint = str(decision["endpoint"])
    endpoints = ("first", "second") if endpoint == "both" else (endpoint,)
    observed = (first, second)
    outputs = [first.copy(), second.copy()]
    supports = [empty.copy(), empty.copy()]
    endpoint_records = []
    for index, name in enumerate(("first", "second")):
        if name not in endpoints:
            endpoint_records.append({
                "endpoint": name, "modified": False,
                "support_pixels": 0, "changed_pixels": 0,
            })
            continue
        support, sigma_map, diagnostics = _active_tile_support_and_sigma(observed[index])
        if not bool(support.any()):
            raise RuntimeError("paired additive authorization produced empty active support")
        proposal = _local_tile_wiener_proposal(observed[index], sigma_map)
        output, _ = blend_local_proposal(
            observed[index], proposal, support.astype(np.float32),
            operator_id=LOCAL_WIENER_OPERATOR_ID_V2, endpoint=name,
            feather_sigma=1.0, effective_support=support.astype(np.float32),
            diagnostics=diagnostics,
        )
        changed = np.any(output != observed[index], axis=2)
        if not bool(changed.any()):
            raise RuntimeError("local Wiener successor remained identity-only")
        if np.any(changed & ~support) or not np.array_equal(
            output[~support], observed[index][~support],
        ):
            raise RuntimeError("local Wiener successor escaped support")
        outputs[index] = output
        supports[index] = support
        endpoint_records.append({
            "endpoint": name,
            "modified": True,
            "support_sha256": _array_sha256(support),
            "support_pixels": int(support.sum()),
            "sigma_map_sha256": _array_sha256(sigma_map),
            "changed_mask_sha256": _array_sha256(changed),
            "changed_pixels": int(changed.sum()),
            "output_sha256": _array_sha256(output),
            "outside_support_byte_identity": True,
            "diagnostics": diagnostics,
        })
    receipt = {
        **base,
        "status": "EXECUTED_LOCAL_TILE_SIGMA_V2",
        "exact_endpoint": endpoint,
        "endpoint_records": endpoint_records,
        "formula": (
            "gain=max(local_variance-tile_sigma^2,0)/"
            "max(local_variance,tile_sigma^2,1e-12)"
        ),
        "composition": "HARD_SUPPORT_INWARD_FEATHER_SIGMA_1_OUTSIDE_BYTE_EXACT",
        "scientific_qualification": False,
        "selector_admission": False,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return LocalWienerSuccessorV2(
        status=receipt["status"], exact_endpoint=endpoint,
        first_rgb=outputs[0], second_rgb=outputs[1],
        first_support=supports[0], second_support=supports[1], receipt=receipt,
    )


__all__ = [
    "LOCAL_WIENER_ACTION_ID_V2",
    "LOCAL_WIENER_OPERATOR_ID_V2",
    "LOCAL_WIENER_SCHEMA_V2",
    "LocalWienerSuccessorV2",
    "build_local_wiener_successor_v2",
]
