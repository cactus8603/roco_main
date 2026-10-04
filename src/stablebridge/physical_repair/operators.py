"""Observable, mask-respecting local image actions."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class LocalImageAction:
    operator_id: str
    endpoint: str
    changed_fraction: float
    support_fraction: float
    diagnostics: dict[str, float]


def _validate(image: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image)
    support = np.asarray(support, dtype=np.float32)
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or support.shape != image.shape[:2] or not np.isfinite(support).all()
            or np.any(support < 0.0) or np.any(support > 1.0)):
        raise ValueError("expected uint8 RGB and matching support in [0,1]")
    return np.ascontiguousarray(image), np.ascontiguousarray(support)


def blend_local_proposal(image: np.ndarray, proposal: np.ndarray,
                         support: np.ndarray, *, operator_id: str,
                         endpoint: str, feather_sigma: float = 1.0,
                         effective_support: np.ndarray | None = None,
                         diagnostics: dict[str, float] | None = None
                         ) -> tuple[np.ndarray, LocalImageAction]:
    """Blend an audited full-image proposal only inside endpoint support.

    Filtering may read context outside the authorized region when constructing
    ``proposal``.  The intervention itself is nevertheless hard re-masked after
    inward feathering, so a zero-support input pixel is bit-exact.  This lets an
    experiment reuse the exact global blur/JPEG implementation while testing a
    genuinely local input intervention rather than an output-flow splice.
    """
    image, hard_support = _validate(image, support)
    proposal = np.asarray(proposal)
    if (proposal.dtype != np.uint8 or proposal.shape != image.shape
            or endpoint not in {"first", "second"}
            or not operator_id or not np.isfinite(float(feather_sigma))
            or float(feather_sigma) < 0.0):
        raise ValueError("invalid local proposal, endpoint, operator, or feather")
    proposal = np.ascontiguousarray(proposal)
    if effective_support is None:
        effective = hard_support
    else:
        effective = np.asarray(effective_support, dtype=np.float32)
        if (effective.shape != hard_support.shape or not np.isfinite(effective).all()
                or np.any(effective < 0.0) or np.any(effective > 1.0)):
            raise ValueError("effective support must be a finite HxW map in [0,1]")
        effective = np.ascontiguousarray(effective * hard_support)
    if float(feather_sigma) == 0.0:
        feather = effective.copy()
    else:
        feather = cv2.GaussianBlur(
            effective, (0, 0), float(feather_sigma),
            borderType=cv2.BORDER_REFLECT_101,
        )
        feather = np.clip(feather, 0.0, 1.0) * hard_support
    result = image.astype(np.float32) * (1.0 - feather[..., None])
    result += proposal.astype(np.float32) * feather[..., None]
    output = np.ascontiguousarray(np.clip(np.rint(result), 0, 255).astype(np.uint8))
    outside = hard_support <= 0.0
    if not np.array_equal(output[outside], image[outside]):
        raise RuntimeError("local proposal changed pixels outside support")
    changed = np.any(output != image, axis=2)
    details = dict(diagnostics or {})
    selected = hard_support > 0.0
    details.update({
        "feather_sigma_px": float(feather_sigma),
        "proposal_delta_mean_255": (
            float(np.mean(np.abs(
                proposal[selected].astype(np.float32)
                - image[selected].astype(np.float32)
            ))) if np.any(selected) else 0.0
        ),
        "blend_weight_mean": float(feather.mean()),
    })
    return output, LocalImageAction(
        operator_id=operator_id, endpoint=endpoint,
        changed_fraction=float(changed.mean()),
        support_fraction=float(selected.mean()),
        diagnostics=details,
    )


def _wiener3(image: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    source = image.astype(np.float32)
    output = np.empty_like(source)
    estimates = []
    for channel in range(3):
        plane = source[..., channel]
        lap = cv2.Laplacian(plane, cv2.CV_32F, ksize=1,
                            borderType=cv2.BORDER_REFLECT_101)
        center = float(np.median(lap))
        sigma = float(np.median(np.abs(lap - center))
                      / 0.6744897501960817 / math.sqrt(20.0))
        mean = cv2.blur(plane, (3, 3), borderType=cv2.BORDER_REFLECT_101)
        square = cv2.blur(plane * plane, (3, 3), borderType=cv2.BORDER_REFLECT_101)
        variance = np.maximum(square - mean * mean, 0.0)
        noise = sigma * sigma
        gain = np.maximum(variance - noise, 0.0) / np.maximum(
            variance, np.float32(max(noise, 1e-12)),
        )
        output[..., channel] = mean + gain * (plane - mean)
        estimates.append(sigma)
    return np.clip(np.rint(output), 0, 255).astype(np.uint8), tuple(estimates)


def impulse_support(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected uint8 RGB")
    median = cv2.medianBlur(image, 3)
    black = np.all(image <= 1, axis=2)
    white = np.all(image >= 254, axis=2)
    median_black = np.all(median <= 1, axis=2)
    median_white = np.all(median >= 254, axis=2)
    return np.ascontiguousarray(
        ((black & ~median_black) | (white & ~median_white)).astype(np.float32)
    )


def apply_local_image_action(image: np.ndarray, support: np.ndarray, *,
                             operator_id: str, endpoint: str) -> tuple[np.ndarray, LocalImageAction]:
    """Apply an action only inside its own endpoint-coordinate support.

    The operator may read a small halo, but blending is re-masked after every
    smoothing operation so invalid support cannot leak into the output.
    """
    image, hard_support = _validate(image, support)
    if endpoint not in {"first", "second"}:
        raise ValueError("local image action needs a concrete endpoint")
    if operator_id == "impulse_exact_median3":
        applicability = impulse_support(image)
        effective = hard_support * applicability
        proposal = cv2.medianBlur(image, 3)
        diagnostics = {"detected_fraction": float(applicability.mean())}
    elif operator_id == "wiener3":
        effective = hard_support.copy()
        proposal, sigma = _wiener3(image)
        diagnostics = {f"sigma_{name}_255": float(value)
                       for name, value in zip(("r", "g", "b"), sigma)}
    else:
        raise ValueError(f"unsupported local operator: {operator_id}")
    return blend_local_proposal(
        image, proposal, hard_support, operator_id=operator_id,
        endpoint=endpoint, feather_sigma=1.0,
        effective_support=effective, diagnostics=diagnostics,
    )
