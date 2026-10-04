"""Action-specific regional repair with one matcher candidate trajectory.

The module deliberately keeps four decisions separate:

1. an action-specific physical receipt admits one action family;
2. the receipt supplies an input-coordinate region where that action is
   observable and recoverable;
3. one actual local matcher rerun supplies response and candidate risk; and
4. a deterministic update cap plus native fallback controls delivery.

No corruption label, task outcome, or generic uncertainty scalar is accepted
by this interface.  Uncertainty may schedule the single probe, but cannot
authorize an action or output pixel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import cv2
import numpy as np

from .jpeg_codec_path_actions import jpeg_codec_path_action
from .operators import blend_local_proposal
from .support import (
    transport_flow_mask_to_second,
    transport_second_mask_to_flow,
)


@dataclass(frozen=True)
class RouteDecision:
    action_family: str
    action_id: str
    reason: str
    accepted: bool
    competing_families: tuple[str, ...]


@dataclass(frozen=True)
class RegionalInputCandidate:
    first: np.ndarray
    second: np.ndarray
    action_family: str
    action_id: str
    modified_endpoints: tuple[str, ...]
    flow_influence_seed: np.ndarray
    first_support: np.ndarray
    second_support: np.ndarray
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class OneProbeDelivery:
    flow: np.ndarray
    delivery_support: np.ndarray
    response_support: np.ndarray
    risk_support: np.ndarray
    influence_envelope: np.ndarray
    update_norm: np.ndarray
    max_update_norm: float
    exact_native_outside: bool


def route_action_receipts(
    *,
    blur_selected: Mapping[str, object] | None = None,
    jpeg_supported_endpoints: Sequence[str] = (),
    impulse_supported_endpoints: Sequence[str] = (),
) -> RouteDecision:
    """Admit exactly one physically supported action family or abstain.

    Multiple endpoints of one family form one paired candidate.  Evidence for
    two different families is an unresolved model conflict, not permission to
    choose the larger arbitrary score.  This is the explicit noise-to-motion
    guard: an impulse/noise receipt and a motion-blur receipt cannot silently
    collapse into a motion action.
    """
    families: list[str] = []
    if blur_selected is not None:
        action = str(blur_selected.get("action", ""))
        if action not in {"common_disk", "common_gaussian", "common_motion"}:
            return RouteDecision(
                "native", "native", "invalid_blur_action", False, (),
            )
        families.append("blur")
    jpeg_endpoints = tuple(sorted(set(map(str, jpeg_supported_endpoints))))
    impulse_endpoints = tuple(sorted(set(map(str, impulse_supported_endpoints))))
    if any(endpoint not in {"first", "second"} for endpoint in jpeg_endpoints):
        return RouteDecision("native", "native", "invalid_jpeg_endpoint", False, ())
    if any(endpoint not in {"first", "second"} for endpoint in impulse_endpoints):
        return RouteDecision("native", "native", "invalid_impulse_endpoint", False, ())
    if jpeg_endpoints:
        families.append("jpeg_quality")
    if impulse_endpoints:
        families.append("noise_or_impulse")
    if not families:
        return RouteDecision("native", "native", "no_supported_action", False, ())
    if len(families) != 1:
        return RouteDecision(
            "native", "native", "ambiguous_cross_family_physical_receipts",
            False, tuple(sorted(families)),
        )
    family = families[0]
    if family == "blur":
        action_id = str(blur_selected["action"])
    elif family == "jpeg_quality":
        action_id = "jpeg_deblock@" + "+".join(jpeg_endpoints)
    else:
        action_id = "impulse_exact_median3@" + "+".join(impulse_endpoints)
    return RouteDecision(
        family, action_id, "unique_action_specific_physical_receipt", True,
        tuple(families),
    )


def flow_support_from_regions(
    shape: tuple[int, int], regions: Sequence[Sequence[int]], *, tile_px: int = 64,
) -> np.ndarray:
    if (len(shape) != 2 or min(shape) <= 0 or tile_px <= 0
            or not isinstance(regions, Sequence)):
        raise ValueError("invalid regional support geometry")
    output = np.zeros(shape, dtype=np.float32)
    for region in regions:
        if len(region) != 2:
            raise ValueError("region must contain y0,x0")
        y0, x0 = int(region[0]), int(region[1])
        if y0 < 0 or x0 < 0 or y0 >= shape[0] or x0 >= shape[1]:
            raise ValueError("region origin outside image")
        output[y0:min(y0 + tile_px, shape[0]),
               x0:min(x0 + tile_px, shape[1])] = 1.0
    return np.ascontiguousarray(output)


def _validate_pair(first: np.ndarray, second: np.ndarray,
                   flow: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(first)
    second = np.asarray(second)
    flow = np.asarray(flow, dtype=np.float32)
    if (first.dtype != np.uint8 or second.dtype != np.uint8
            or first.ndim != 3 or first.shape != second.shape or first.shape[2] != 3
            or flow.shape != (*first.shape[:2], 2) or not np.isfinite(flow).all()):
        raise ValueError("expected paired uint8 RGB and finite HxWx2 native flow")
    return np.ascontiguousarray(first), np.ascontiguousarray(second), np.ascontiguousarray(flow)


def _disk_kernel(radius: int) -> np.ndarray:
    if radius < 1:
        raise ValueError("disk radius must be positive")
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    return kernel / float(kernel.sum())


def _motion_kernel(length: int, angle: float) -> np.ndarray:
    if length < 3 or length % 2 == 0 or not np.isfinite(angle):
        raise ValueError("motion length must be odd >=3 and angle finite")
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    theta = math.radians(float(angle))
    dx, dy = math.cos(theta) * center, math.sin(theta) * center
    cv2.line(
        kernel,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        1.0, 1,
    )
    return kernel / float(max(kernel.sum(), 1.0))


def _blur_proposal(image: np.ndarray, action: str, parameter: object) -> np.ndarray:
    if action == "common_disk":
        kernel = _disk_kernel(int(round(float(parameter))))
    elif action == "common_gaussian":
        sigma = float(parameter)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("Gaussian sigma must be positive")
        radius = max(1, int(math.ceil(3.0 * sigma)))
        vector = cv2.getGaussianKernel(2 * radius + 1, sigma, cv2.CV_32F)
        kernel = vector @ vector.T
        kernel /= float(kernel.sum())
    elif action == "common_motion" and isinstance(parameter, Mapping):
        kernel = _motion_kernel(
            int(round(float(parameter["length"]))), float(parameter["angle"]),
        )
    else:
        raise ValueError("unsupported common-passband action")
    value = cv2.filter2D(
        image.astype(np.float32), -1, kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return np.ascontiguousarray(np.clip(np.rint(value), 0, 255).astype(np.uint8))


def build_blur_one_probe_candidate(
    first: np.ndarray, second: np.ndarray, native_flow: np.ndarray,
    selected: Mapping[str, object], *, tile_px: int = 64,
    feather_sigma_px: float = 1.0,
) -> RegionalInputCandidate:
    """Localize one E134-style blur action to its recoverable check regions."""
    first, second, native_flow = _validate_pair(first, second, native_flow)
    action = str(selected.get("action", ""))
    hypothesis = str(selected.get("hypothesis", ""))
    regions = selected.get("regions", ())
    if action not in {"common_disk", "common_gaussian", "common_motion"}:
        raise ValueError("invalid blur action")
    if hypothesis not in {"first", "second"}:
        raise ValueError("blur action needs a one-sided hypothesis")
    flow_support = flow_support_from_regions(
        first.shape[:2], regions, tile_px=tile_px,
    )
    if not np.any(flow_support):
        raise ValueError("blur receipt has no recoverable region")
    first_support = np.zeros(first.shape[:2], dtype=np.float32)
    second_support = np.zeros(first.shape[:2], dtype=np.float32)
    output_first, output_second = first.copy(), second.copy()
    parameter = selected.get("parameter")
    if hypothesis == "first":
        # The first endpoint is inferred degraded; common-passband matching
        # modifies the sharper second endpoint in its own coordinates.
        second_support, splat_weight = transport_flow_mask_to_second(
            flow_support, native_flow,
        )
        proposal = _blur_proposal(second, action, parameter)
        output_second, record = blend_local_proposal(
            second, proposal, second_support, operator_id=action,
            endpoint="second", feather_sigma=feather_sigma_px,
        )
        modified = ("second",)
        collision_fraction = float(np.mean(splat_weight > 1.00001))
        uncovered_fraction = float(np.mean(splat_weight <= 0.0))
    else:
        first_support = flow_support.copy()
        proposal = _blur_proposal(first, action, parameter)
        output_first, record = blend_local_proposal(
            first, proposal, first_support, operator_id=action,
            endpoint="first", feather_sigma=feather_sigma_px,
        )
        modified = ("first",)
        collision_fraction = uncovered_fraction = 0.0
    if np.array_equal(output_first, first) and np.array_equal(output_second, second):
        raise ValueError("physically admitted blur action produced no input change")
    return RegionalInputCandidate(
        first=output_first, second=output_second,
        action_family="blur", action_id=action,
        modified_endpoints=modified,
        flow_influence_seed=np.ascontiguousarray(flow_support),
        first_support=np.ascontiguousarray(first_support),
        second_support=np.ascontiguousarray(second_support),
        diagnostics={
            "hypothesis": hypothesis,
            "parameter": parameter,
            "recoverable_regions": len(regions),
            "flow_support_fraction": float(np.mean(flow_support > 0.0)),
            "second_transport_collision_fraction": collision_fraction,
            "second_transport_uncovered_fraction": uncovered_fraction,
            "local_action": asdict(record),
        },
    )


def build_jpeg_one_probe_candidate(
    first: np.ndarray, second: np.ndarray, native_flow: np.ndarray,
    endpoint_certificates: Mapping[str, Mapping[str, object]],
) -> RegionalInputCandidate:
    """Regenerate exact E147 codec-path actions and their changed support."""
    first, second, native_flow = _validate_pair(first, second, native_flow)
    outputs = {"first": first.copy(), "second": second.copy()}
    supports = {
        "first": np.zeros(first.shape[:2], dtype=np.float32),
        "second": np.zeros(first.shape[:2], dtype=np.float32),
    }
    parameters: dict[str, object] = {}
    active: list[str] = []
    for endpoint in ("first", "second"):
        receipt = endpoint_certificates.get(endpoint)
        if receipt is None or receipt.get("status") != "supported":
            continue
        estimated = receipt.get("estimated_parameters", {})
        quality = int(round(float(estimated["ijg_quality"])))
        result = jpeg_codec_path_action(
            first if endpoint == "first" else second,
            estimated_quality=quality,
        )
        expected_strength = float(estimated["action_strength"])
        if result.strength != expected_strength:
            raise RuntimeError("JPEG codec-path strength drift")
        source = first if endpoint == "first" else second
        changed = np.any(result.image != source, axis=2)
        if not np.any(changed):
            raise RuntimeError("supported JPEG action produced no change")
        if not np.array_equal(result.image[~changed], source[~changed]):
            raise RuntimeError("JPEG action changed outside exact change support")
        outputs[endpoint] = result.image
        supports[endpoint] = np.ascontiguousarray(changed.astype(np.float32))
        parameters[endpoint] = {
            "ijg_quality": quality, "strength": result.strength,
            "quantization_violation_max": result.quantization_violation_max,
        }
        active.append(endpoint)
    if not active:
        raise ValueError("no supported JPEG endpoint")
    second_flow, second_valid = transport_second_mask_to_flow(
        supports["second"], native_flow,
    )
    flow_seed = np.maximum(supports["first"], second_flow)
    if not np.any(flow_seed):
        raise RuntimeError("JPEG action has no flow-native influence seed")
    return RegionalInputCandidate(
        first=np.ascontiguousarray(outputs["first"]),
        second=np.ascontiguousarray(outputs["second"]),
        action_family="jpeg_quality",
        action_id="jpeg_deblock@" + "+".join(active),
        modified_endpoints=tuple(active),
        flow_influence_seed=np.ascontiguousarray(flow_seed, dtype=np.float32),
        first_support=supports["first"], second_support=supports["second"],
        diagnostics={
            "parameters": parameters,
            "first_changed_fraction": float(supports["first"].mean()),
            "second_changed_fraction": float(supports["second"].mean()),
            "second_transport_valid_fraction": float(second_valid.mean()),
            "flow_influence_seed_fraction": float(np.mean(flow_seed > 0.0)),
        },
    )


def compose_one_probe_delivery(
    native_flow: np.ndarray, candidate_flow: np.ndarray,
    native_risk: np.ndarray, candidate_risk: np.ndarray,
    flow_influence_seed: np.ndarray, *, response_threshold_px: float = 0.01,
    influence_halo_px: int = 16, absolute_update_cap_px: float = 0.25,
    numerical_guard_px: float = 1e-4,
) -> OneProbeDelivery:
    """Use actual local response/risk, then cap once relative to native.

    The absolute per-pixel update cap gives a deterministic metric-aligned
    bound: by the reverse triangle inequality, no valid pixel's endpoint error
    can increase by more than the update norm.  It is a bounded trust region,
    not a claim that the accepted direction is always beneficial.
    """
    native = np.asarray(native_flow, dtype=np.float32)
    candidate = np.asarray(candidate_flow, dtype=np.float32)
    native_risk = np.asarray(native_risk, dtype=np.float32)
    candidate_risk = np.asarray(candidate_risk, dtype=np.float32)
    seed = np.asarray(flow_influence_seed, dtype=np.float32)
    if (native.ndim != 3 or native.shape[2] != 2 or candidate.shape != native.shape
            or native_risk.shape != native.shape[:2]
            or candidate_risk.shape != native.shape[:2]
            or seed.shape != native.shape[:2]
            or not np.isfinite(native).all() or not np.isfinite(candidate).all()
            or not np.isfinite(native_risk).all() or not np.isfinite(candidate_risk).all()
            or not np.isfinite(seed).all() or np.any(native_risk < 0.0)
            or np.any(candidate_risk < 0.0) or np.any(seed < 0.0)
            or np.any(seed > 1.0)):
        raise ValueError("invalid one-probe flow, risk, or influence inputs")
    if (not np.isfinite(response_threshold_px) or response_threshold_px < 0.0
            or influence_halo_px < 0
            or not np.isfinite(absolute_update_cap_px)
            or absolute_update_cap_px <= numerical_guard_px
            or numerical_guard_px < 0.0):
        raise ValueError("invalid one-probe delivery constants")
    active_seed = (seed > 0.0).astype(np.uint8)
    if influence_halo_px:
        size = 2 * int(influence_halo_px) + 1
        influence = cv2.dilate(active_seed, np.ones((size, size), np.uint8)) > 0
    else:
        influence = active_seed > 0
    delta = candidate.astype(np.float64) - native.astype(np.float64)
    norm = np.linalg.norm(delta, axis=2)
    response = norm > float(response_threshold_px)
    risk = candidate_risk <= native_risk
    support = influence & response & risk
    radius = np.minimum(
        native_risk.astype(np.float64) + candidate_risk.astype(np.float64),
        float(absolute_update_cap_px - numerical_guard_px),
    )
    scale = np.minimum(1.0, radius / np.maximum(norm, 1e-12))
    output = native.astype(np.float64) + delta * (support * scale)[..., None]
    output = np.ascontiguousarray(output, dtype=np.float32)
    update = np.linalg.norm(
        output.astype(np.float64) - native.astype(np.float64), axis=2,
    ).astype(np.float32)
    maximum = float(update.max(initial=0.0))
    outside = ~support
    exact = bool(np.array_equal(output[outside], native[outside]))
    if not exact:
        raise RuntimeError("one-probe native fallback is not bit exact")
    if maximum > absolute_update_cap_px + numerical_guard_px:
        raise RuntimeError("one-probe update exceeded absolute trust cap")
    return OneProbeDelivery(
        flow=output, delivery_support=np.ascontiguousarray(support),
        response_support=np.ascontiguousarray(response),
        risk_support=np.ascontiguousarray(risk),
        influence_envelope=np.ascontiguousarray(influence),
        update_norm=update, max_update_norm=maximum,
        exact_native_outside=exact,
    )
