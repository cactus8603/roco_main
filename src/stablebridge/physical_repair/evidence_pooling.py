"""Reusable, outcome-blind region pooling for cycle/G0 evidence.

The E233 evidence package is sealed and must remain replayable.  This module is
an append-only successor to its ``pool_region`` helper.  It preserves the
E233/v2 field semantics while moving the three whole-flow float64 conversions
to case scope and evaluating any number of fold-specific fit scales from one
numeric region pass.

This helper only summarizes arrays supplied by its caller.  It does not read
ground truth, outcomes, labels, experiment directories, or authority records.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


_CYCLE_MASK_FIELDS = (
    "cycle_valid",
    "forward_nonfinite",
    "coordinate_invalid",
    "interpolation_incomplete",
)
_G0_MASK_FIELDS = (
    "jacobian_finite",
    "fold",
    "near_singular",
    "ambiguity",
)


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    """Match the sealed E233/v2 finite-value summary exactly."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.quantile(values, 0.5, method="linear")),
        "p90": float(np.quantile(values, 0.9, method="linear")),
        "p95": float(np.quantile(values, 0.95, method="linear")),
        "max": float(values.max()),
    }


def _owned_array(
    value: np.ndarray,
    *,
    dtype: np.dtype[Any] | type[Any],
) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class RegionPoolBatch:
    """One unscaled pool and exact E233-shaped pools for named fit scales."""

    unscaled: dict[str, Any]
    by_fit_scale: dict[str, dict[str, Any]]


class CaseEvidencePooler:
    """Case-scoped cache for exact E233-compatible region summaries.

    Construction owns immutable copies of all inputs.  The three flow arrays
    are converted to float64 once.  ``pool_region_fit_scales`` then computes
    region norms, statistics, tiles, parity, and G0 fields once and derives all
    threshold-only fields from the same delta vector.
    """

    def __init__(
        self,
        native_forward: np.ndarray,
        full_forward: np.ndarray,
        cycle: Mapping[str, np.ndarray],
        g0: Mapping[str, np.ndarray],
    ) -> None:
        native = np.asarray(native_forward)
        full = np.asarray(full_forward)
        if native.ndim != 3 or native.shape[-1] != 2:
            raise ValueError("native forward must be HxWx2")
        if full.shape != native.shape:
            raise ValueError("full forward must share the native HxWx2 shape")
        self._shape = native.shape[:2]

        try:
            verifier = np.asarray(cycle["verifier"])
        except KeyError as exc:
            raise ValueError("cycle is missing verifier") from exc
        if verifier.shape != native.shape:
            raise ValueError("cycle verifier must share the forward HxWx2 shape")

        self._native_forward = _owned_array(native, dtype=np.float64)
        self._full_forward = _owned_array(full, dtype=np.float64)
        self._verifier = _owned_array(verifier, dtype=np.float64)

        cycle_arrays: dict[str, np.ndarray] = {}
        for field in _CYCLE_MASK_FIELDS:
            try:
                value = np.asarray(cycle[field])
            except KeyError as exc:
                raise ValueError(f"cycle is missing {field}") from exc
            if value.shape != self._shape or value.dtype != np.bool_:
                raise ValueError(f"cycle {field} must be a boolean HxW array")
            cycle_arrays[field] = _owned_array(value, dtype=bool)
        self._cycle = cycle_arrays

        g0_arrays: dict[str, np.ndarray] = {}
        try:
            jacobian_det = np.asarray(g0["jacobian_det"])
        except KeyError as exc:
            raise ValueError("g0 is missing jacobian_det") from exc
        if jacobian_det.shape != self._shape:
            raise ValueError("g0 jacobian_det must be an HxW array")
        g0_arrays["jacobian_det"] = _owned_array(
            jacobian_det, dtype=np.float64,
        )
        for field in _G0_MASK_FIELDS:
            try:
                value = np.asarray(g0[field])
            except KeyError as exc:
                raise ValueError(f"g0 is missing {field}") from exc
            if value.shape != self._shape or value.dtype != np.bool_:
                raise ValueError(f"g0 {field} must be a boolean HxW array")
            g0_arrays[field] = _owned_array(value, dtype=bool)
        self._g0 = g0_arrays

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def float64_flow_cache_nbytes(self) -> int:
        """Resident bytes for native, full, and verifier float64 flow caches."""
        return int(
            self._native_forward.nbytes
            + self._full_forward.nbytes
            + self._verifier.nbytes
        )

    @property
    def total_cache_nbytes(self) -> int:
        """Resident bytes including owned masks and G0 arrays."""
        structural = sum(value.nbytes for value in self._cycle.values())
        structural += sum(value.nbytes for value in self._g0.values())
        return self.float64_flow_cache_nbytes + int(structural)

    def _region(
        self,
        bbox: Sequence[int],
    ) -> tuple[slice, slice, int, int, int, int]:
        if len(bbox) != 4:
            raise ValueError("bbox must be [x0, y0, x1, y1]")
        if any(isinstance(value, (bool, np.bool_)) for value in bbox):
            raise ValueError("bbox coordinates must be integers")
        try:
            x0, y0, x1, y1 = (int(value) for value in bbox)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("bbox coordinates must be integers") from exc
        if any(float(value) != integer for value, integer in zip(
            bbox, (x0, y0, x1, y1), strict=True,
        )):
            raise ValueError("bbox coordinates must be integers")
        height, width = self._shape
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("bbox must be a nonempty in-bounds region")
        return slice(y0, y1), slice(x0, x1), x0, y0, x1, y1

    def _pool_once(
        self,
        bbox: Sequence[int],
    ) -> tuple[dict[str, Any], np.ndarray]:
        ys, xs, x0, y0, x1, y1 = self._region(bbox)
        valid = self._cycle["cycle_valid"][ys, xs]
        area = int((y1 - y0) * (x1 - x0))
        before = np.linalg.norm(
            self._native_forward[ys, xs] - self._verifier[ys, xs], axis=2,
        )
        full = np.linalg.norm(
            self._full_forward[ys, xs] - self._verifier[ys, xs], axis=2,
        )
        delta = full - before
        before_values = before[valid]
        full_values = full[valid]
        delta_values = delta[valid]

        tile_means: list[float] = []
        parity: dict[str, list[float]] = {"0": [], "1": []}
        for ty0 in range((y0 // 32) * 32, y1, 32):
            for tx0 in range((x0 // 32) * 32, x1, 32):
                iy0, iy1 = max(y0, ty0), min(y1, ty0 + 32)
                ix0, ix1 = max(x0, tx0), min(x1, tx0 + 32)
                local = delta[
                    iy0 - y0:iy1 - y0,
                    ix0 - x0:ix1 - x0,
                ]
                local_valid = valid[
                    iy0 - y0:iy1 - y0,
                    ix0 - x0:ix1 - x0,
                ]
                values = local[local_valid]
                values = values[np.isfinite(values)]
                if values.size:
                    mean = float(values.mean())
                    tile_means.append(mean)
                    parity[str(((ty0 // 32) + (tx0 // 32)) % 2)].append(mean)
        tile = np.asarray(tile_means, dtype=np.float64)

        cycle = self._cycle
        g0 = self._g0
        pooled: dict[str, Any] = {
            "availability": (
                "AVAILABLE_EXACT"
                if delta_values.size
                else "TYPED_MISSING_ABSTAIN_NATIVE"
            ),
            "area_px": area,
            "cycle_valid_count": int(valid.sum()),
            "cycle_valid_fraction": float(valid.mean()),
            "e_before": _stats(before_values),
            "e_full": _stats(full_values),
            "delta": {
                key: value
                for key, value in _stats(delta_values).items()
                if key in ("count", "mean", "median", "p90", "p95")
            },
            "delta_positive_fraction": (
                float((delta_values > 0).mean())
                if delta_values.size else None
            ),
            "delta_above_fit_scale_fraction": None,
            "fit_scale_availability": "TYPED_MISSING_PRE_FIT_SCALE",
            "parity32_mean_delta": {
                key: (float(np.mean(values)) if values else None)
                for key, values in parity.items()
            },
            "tile32_mean_delta_quantiles": {
                "count": int(tile.size),
                "q10": (
                    float(np.quantile(tile, 0.1, method="linear"))
                    if tile.size else None
                ),
                "q50": (
                    float(np.quantile(tile, 0.5, method="linear"))
                    if tile.size else None
                ),
                "q90": (
                    float(np.quantile(tile, 0.9, method="linear"))
                    if tile.size else None
                ),
            },
            "invalid_fractions": {
                key: float(cycle[key][ys, xs].mean())
                for key in (
                    "forward_nonfinite",
                    "coordinate_invalid",
                    "interpolation_incomplete",
                )
            },
            "g0": {
                "coordinate_valid_fraction": float((
                    ~cycle["coordinate_invalid"][ys, xs]
                    & ~cycle["forward_nonfinite"][ys, xs]
                ).mean()),
                "transport_complete_fraction": float(valid.mean()),
                "jacobian_finite_fraction": float(
                    g0["jacobian_finite"][ys, xs].mean()
                ),
                "jacobian_det": _stats(
                    g0["jacobian_det"][ys, xs][
                        g0["jacobian_finite"][ys, xs]
                    ]
                ),
                "local_fold_fraction": float(g0["fold"][ys, xs].mean()),
                "near_singular_fraction_epsilon_0_05": float(
                    g0["near_singular"][ys, xs].mean()
                ),
                "ambiguity_fraction": float(
                    g0["ambiguity"][ys, xs].mean()
                ),
            },
        }
        return pooled, delta_values

    @staticmethod
    def _apply_fit_scale(
        unscaled: dict[str, Any],
        delta_values: np.ndarray,
        fit_scale: float | None,
    ) -> dict[str, Any]:
        pooled = deepcopy(unscaled)
        scale_available = (
            fit_scale is not None and math.isfinite(fit_scale)
        )
        pooled["delta_above_fit_scale_fraction"] = (
            float((delta_values > fit_scale).mean())
            if delta_values.size and scale_available else None
        )
        pooled["fit_scale_availability"] = (
            "AVAILABLE_FIT_ONLY"
            if scale_available else "TYPED_MISSING_PRE_FIT_SCALE"
        )
        return pooled

    def pool_region(
        self,
        bbox: Sequence[int],
        fit_scale: float | None = None,
    ) -> dict[str, Any]:
        """Return one field-for-field E233/v2-compatible region summary."""
        unscaled, delta_values = self._pool_once(bbox)
        return self._apply_fit_scale(unscaled, delta_values, fit_scale)

    def pool_region_fit_scales(
        self,
        bbox: Sequence[int],
        fit_scales: Mapping[str, float | None],
    ) -> RegionPoolBatch:
        """Pool once and derive exact summaries for every named fit scale."""
        unscaled, delta_values = self._pool_once(bbox)
        by_fit_scale = {
            str(key): self._apply_fit_scale(unscaled, delta_values, fit_scale)
            for key, fit_scale in fit_scales.items()
        }
        return RegionPoolBatch(
            unscaled=deepcopy(unscaled),
            by_fit_scale=by_fit_scale,
        )


__all__ = ["CaseEvidencePooler", "RegionPoolBatch"]
