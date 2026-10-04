"""Keep risk, physical applicability, support, and utility as separate maps."""
from __future__ import annotations

import cv2
import numpy as np

from .contracts import SupportMaps
from .operators import impulse_support


def _normalize(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros(value.shape, dtype=np.float32)
    selected = np.asarray(value, dtype=np.float32)[valid]
    if not selected.size:
        return output
    low, high = np.quantile(selected, (0.50, 0.95))
    output[valid] = np.clip((value[valid] - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return output


def _warp_second(second: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = second.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx + flow[..., 0].astype(np.float32)
    map_y = yy + flow[..., 1].astype(np.float32)
    valid = ((map_x >= 0.0) & (map_x <= width - 1.0)
             & (map_y >= 0.0) & (map_y <= height - 1.0))
    warped = cv2.remap(second.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return warped, valid


def transport_second_mask_to_flow(mask: np.ndarray, flow: np.ndarray
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Sample a second-native mask at the native first-to-second correspondence.

    This is a pull into the first/native-flow lattice.  Out-of-frame samples are
    zero and returned as invalid; callers must not silently treat them as
    evidence of absence.
    """
    mask = np.asarray(mask, dtype=np.float32)
    flow = np.asarray(flow, dtype=np.float32)
    if (mask.ndim != 2 or flow.shape != (*mask.shape, 2)
            or not np.isfinite(mask).all() or not np.isfinite(flow).all()
            or np.any(mask < 0.0) or np.any(mask > 1.0)):
        raise ValueError("expected a finite second-native mask in [0,1] and HxWx2 flow")
    height, width = mask.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx + flow[..., 0]
    map_y = yy + flow[..., 1]
    valid = ((map_x >= 0.0) & (map_x <= width - 1.0)
             & (map_y >= 0.0) & (map_y <= height - 1.0))
    transported = cv2.remap(
        mask, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    transported[~valid] = 0.0
    return np.ascontiguousarray(transported, dtype=np.float32), valid


def transport_flow_mask_to_second(mask: np.ndarray, flow: np.ndarray
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly splat a flow-native mask into the second-image lattice.

    The inverse of :func:`transport_second_mask_to_flow` is not a simple remap:
    first-image samples may collide or leave holes in the second view.  This
    routine accumulates bilinear value and weight, returns their normalized
    ratio, and exposes the accumulated weight separately.  ``weight == 0`` is
    unknown/uncovered, while ``weight > 1`` explicitly records collisions.
    Consumers must not interpret uncovered pixels as negative evidence.
    """
    mask = np.asarray(mask, dtype=np.float32)
    flow = np.asarray(flow, dtype=np.float32)
    if (mask.ndim != 2 or flow.shape != (*mask.shape, 2)
            or not np.isfinite(mask).all() or not np.isfinite(flow).all()
            or np.any(mask < 0.0) or np.any(mask > 1.0)):
        raise ValueError("expected a finite flow-native mask in [0,1] and HxWx2 flow")
    height, width = mask.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    target_x = xx + flow[..., 0]
    target_y = yy + flow[..., 1]
    source_valid = ((target_x >= 0.0) & (target_x <= width - 1.0)
                    & (target_y >= 0.0) & (target_y <= height - 1.0))
    x0 = np.floor(target_x).astype(np.int64)
    y0 = np.floor(target_y).astype(np.int64)
    dx = target_x - x0
    dy = target_y - y0
    accumulated = np.zeros((height, width), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    for offset_x, offset_y, contribution in (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    ):
        target_ix = x0 + offset_x
        target_iy = y0 + offset_y
        chosen = (source_valid & (target_ix >= 0) & (target_ix < width)
                  & (target_iy >= 0) & (target_iy < height)
                  & (contribution > 0.0))
        np.add.at(weight, (target_iy[chosen], target_ix[chosen]), contribution[chosen])
        np.add.at(
            accumulated, (target_iy[chosen], target_ix[chosen]),
            contribution[chosen] * mask[chosen],
        )
    transported = np.zeros((height, width), dtype=np.float32)
    covered = weight > 1e-12
    transported[covered] = (accumulated[covered] / weight[covered]).astype(np.float32)
    return (np.ascontiguousarray(np.clip(transported, 0.0, 1.0)),
            np.ascontiguousarray(weight, dtype=np.float32))


def _applicability(image: np.ndarray, operator_id: str) -> np.ndarray:
    if operator_id == "impulse_exact_median3":
        return impulse_support(image)
    if operator_id == "wiener3":
        local = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)
        high = np.mean(np.abs(image.astype(np.float32) - local), axis=2)
        return _normalize(high, np.ones(high.shape, dtype=bool))
    raise ValueError(f"unsupported support operator: {operator_id}")


def observable_support_maps(first: np.ndarray, second: np.ndarray, flow: np.ndarray,
                            *, operator_id: str, endpoint: str,
                            predicted_signed_utility: np.ndarray | None = None) -> SupportMaps:
    """Construct four distinct observable maps in native flow coordinates."""
    first = np.asarray(first); second = np.asarray(second); flow = np.asarray(flow)
    if (first.dtype != np.uint8 or second.dtype != np.uint8
            or first.ndim != 3 or first.shape != second.shape or first.shape[2] != 3
            or flow.shape != (*first.shape[:2], 2) or not np.isfinite(flow).all()):
        raise ValueError("invalid observed pair or flow")
    warped, valid = _warp_second(second, flow)
    residual = np.mean(np.abs(first.astype(np.float32) - warped), axis=2)
    risk = _normalize(residual, valid)
    gray_first = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_second = cv2.cvtColor(
        np.clip(np.rint(warped), 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY,
    ).astype(np.float32)
    texture = np.zeros(first.shape[:2], dtype=np.float32)
    for gray in (gray_first, gray_second):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        texture = np.maximum(texture, np.sqrt(gx * gx + gy * gy))
    evidence = _normalize(texture, valid)
    evidence = np.clip(evidence + 0.25, 0.0, 1.0) * valid.astype(np.float32)
    if endpoint not in {"first", "second", "both"}:
        raise ValueError("support endpoint must be first, second, or both")
    first_applicability = (_applicability(first, operator_id)
                           if endpoint in {"first", "both"}
                           else np.zeros(first.shape[:2], dtype=np.float32))
    second_applicability = (_applicability(second, operator_id)
                            if endpoint in {"second", "both"}
                            else np.zeros(second.shape[:2], dtype=np.float32))
    transported_second, second_valid = transport_second_mask_to_flow(
        second_applicability, flow,
    )
    applicability = np.maximum(first_applicability, transported_second)
    # A second-endpoint hypothesis with no in-frame correspondence is unknown,
    # not positive support.  The evidence map retains the explicit valid mask.
    if endpoint == "second":
        applicability *= second_valid.astype(np.float32)
    utility = (np.zeros(first.shape[:2], dtype=np.float32)
               if predicted_signed_utility is None
               else np.asarray(predicted_signed_utility, dtype=np.float32))
    return SupportMaps(
        coordinate_frame="flow_native", native_error_risk=risk,
        physical_applicability=applicability,
        evidence_support=evidence,
        predicted_signed_utility=utility,
    )
