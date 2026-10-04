"""Unified typed-certificate controller for impulse, blur, and JPEG actions.

This module extends the frozen blur/JPEG one-probe executor without changing
that executor's source hash.  A caller must first obtain an action-specific
physical certificate.  The controller then admits exactly one family, builds
one regional candidate, and uses the same response/risk/trust-region delivery
law for every family.

The impulse path intentionally consumes explicit endpoint-coordinate support
from the clipping/paired-response/two-tail/spatial certificate stack.  It does
not infer an impulse action from uncertainty, a corruption label, or task
outcome.  Cross-family evidence is an abstention.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

import cv2
import numpy as np

from .broad_one_probe import (
    OneProbeDelivery,
    RegionalInputCandidate,
    RouteDecision,
    build_blur_one_probe_candidate,
    build_jpeg_one_probe_candidate,
    compose_one_probe_delivery,
    route_action_receipts,
)
from .operators import blend_local_proposal, impulse_support
from .support import transport_second_mask_to_flow


def _pair(
    first: np.ndarray, second: np.ndarray, native_flow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(first)
    second = np.asarray(second)
    flow = np.asarray(native_flow, dtype=np.float32)
    if (first.dtype != np.uint8 or second.dtype != np.uint8
            or first.ndim != 3 or first.shape != second.shape
            or first.shape[2] != 3 or flow.shape != (*first.shape[:2], 2)
            or not np.isfinite(flow).all()):
        raise ValueError("expected paired uint8 RGB and finite HxWx2 native flow")
    return (
        np.ascontiguousarray(first), np.ascontiguousarray(second),
        np.ascontiguousarray(flow),
    )


def _support(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if (result.shape != shape or not np.isfinite(result).all()
            or np.any(result < 0.0) or np.any(result > 1.0)):
        raise ValueError("impulse endpoint support must be finite HxW in [0,1]")
    return np.ascontiguousarray(result)


def build_impulse_one_probe_candidate(
    first: np.ndarray,
    second: np.ndarray,
    native_flow: np.ndarray,
    endpoint_supports: Mapping[str, np.ndarray],
) -> RegionalInputCandidate:
    """Build the exact full-strength median candidate on certified supports.

    The physical support is re-intersected with the observable exact-extremum
    support.  This fails closed if a purported certificate has no actionable
    pixel.  The input intervention is bit-exact outside the supplied endpoint
    support, and the flow-lattice influence seed is derived from pixels that
    actually changed rather than from a whole-image family flag.
    """
    first, second, native_flow = _pair(first, second, native_flow)
    keys = tuple(sorted(map(str, endpoint_supports)))
    if not keys or any(key not in {"first", "second"} for key in keys):
        raise ValueError("impulse candidate needs first and/or second support")
    images = {"first": first, "second": second}
    outputs = {"first": first.copy(), "second": second.copy()}
    supports = {
        "first": np.zeros(first.shape[:2], dtype=np.float32),
        "second": np.zeros(first.shape[:2], dtype=np.float32),
    }
    changes = {
        "first": np.zeros(first.shape[:2], dtype=np.float32),
        "second": np.zeros(first.shape[:2], dtype=np.float32),
    }
    records: dict[str, object] = {}
    for endpoint in keys:
        certified = _support(endpoint_supports[endpoint], first.shape[:2])
        effective = certified * impulse_support(images[endpoint])
        if not np.any(effective > 0.0):
            raise ValueError(f"certified impulse support is not actionable: {endpoint}")
        proposal = cv2.medianBlur(images[endpoint], 3)
        output, record = blend_local_proposal(
            images[endpoint], proposal, certified,
            operator_id="impulse_exact_median3", endpoint=endpoint,
            feather_sigma=0.0, effective_support=effective,
            diagnostics={"certificate_support_pixels": float(np.sum(certified > 0.0))},
        )
        changed = np.any(output != images[endpoint], axis=2).astype(np.float32)
        if not np.any(changed):
            raise ValueError(f"certified impulse action produced no change: {endpoint}")
        if np.any((changed > 0.0) & ~(certified > 0.0)):
            raise RuntimeError("impulse action changed outside certified support")
        outputs[endpoint] = output
        supports[endpoint] = certified
        changes[endpoint] = np.ascontiguousarray(changed)
        records[endpoint] = asdict(record)
    second_flow, second_valid = transport_second_mask_to_flow(
        changes["second"], native_flow,
    )
    seed = np.maximum(changes["first"], second_flow)
    if not np.any(seed > 0.0):
        raise RuntimeError("impulse action has no flow-native influence seed")
    return RegionalInputCandidate(
        first=np.ascontiguousarray(outputs["first"]),
        second=np.ascontiguousarray(outputs["second"]),
        action_family="noise_or_impulse",
        action_id="impulse_exact_median3@" + "+".join(keys),
        modified_endpoints=keys,
        flow_influence_seed=np.ascontiguousarray(seed, dtype=np.float32),
        first_support=supports["first"], second_support=supports["second"],
        diagnostics={
            "endpoint_actions": records,
            "first_changed_pixels": int(changes["first"].sum()),
            "second_changed_pixels": int(changes["second"].sum()),
            "second_transport_valid_fraction": float(second_valid.mean()),
            "flow_influence_seed_pixels": int(np.sum(seed > 0.0)),
            "full_strength": True,
        },
    )


__all__ = [
    "OneProbeDelivery", "RegionalInputCandidate", "RouteDecision",
    "build_blur_one_probe_candidate", "build_impulse_one_probe_candidate",
    "build_jpeg_one_probe_candidate", "compose_one_probe_delivery",
    "route_action_receipts",
]
