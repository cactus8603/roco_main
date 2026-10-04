"""Spatial-diffusion receipts for random salt-and-pepper contamination.

The receipt is an input-only no-action rival.  Conditional on a homogeneous
random impulse process and the exact local-median opportunity mask, events of
each polarity should cover every spatial cell whose probability of remaining
empty is already below a simultaneous family-wise bound.  Native palette
extrema are commonly clustered on objects, edges, or renderer-specific
regions, so they fail this coverage law even when both black and white tails
are present.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import cv2
import numpy as np

from .operators import impulse_support


def _hash(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask, dtype=bool).tobytes()
    ).hexdigest()


def _image(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.uint8 or result.ndim != 3 or result.shape[2] != 3:
        raise ValueError("expected uint8 RGB")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class SpatialScaleReceipt:
    grid_size: int
    event_count: int
    opportunity_count: int
    required_cells: int
    occupied_required_cells: int
    identifiable: bool
    passed: bool
    event_counts: tuple[int, ...]
    opportunity_counts: tuple[int, ...]
    required_cell_flags: tuple[bool, ...]

    def __post_init__(self) -> None:
        cells = self.grid_size * self.grid_size
        if self.grid_size < 2:
            raise ValueError("spatial grid must contain multiple cells")
        if not (len(self.event_counts) == len(self.opportunity_counts)
                == len(self.required_cell_flags) == cells):
            raise ValueError("spatial grid cardinality drift")
        if self.event_count != sum(self.event_counts):
            raise ValueError("spatial event count drift")
        if self.opportunity_count != sum(self.opportunity_counts):
            raise ValueError("spatial opportunity count drift")
        required = sum(self.required_cell_flags)
        occupied = sum(
            flag and count > 0
            for flag, count in zip(self.required_cell_flags, self.event_counts)
        )
        if self.required_cells != required or self.occupied_required_cells != occupied:
            raise ValueError("spatial required-cell drift")
        if self.identifiable != (required > 0):
            raise ValueError("spatial identifiability drift")
        if self.passed != (occupied == required):
            raise ValueError("spatial pass decision drift")


@dataclass(frozen=True)
class SpatialImpulseReceipt:
    endpoint: str
    alpha: float
    hypotheses_tested: int
    grid_sizes: tuple[int, ...]
    per_cell_alpha: float
    black_event_count: int
    white_event_count: int
    black_event_hash: str
    white_event_hash: str
    black_scales: tuple[SpatialScaleReceipt, ...]
    white_scales: tuple[SpatialScaleReceipt, ...]
    supported: bool
    law_id: str = "conditional_spatial_diffusion_empty_cell_fwer_v1"
    corruption_label_read: bool = False
    ground_truth_or_outcome_read: bool = False

    def __post_init__(self) -> None:
        if self.endpoint not in {"first", "second"}:
            raise ValueError("spatial impulse receipt needs a concrete endpoint")
        if not 0.0 < self.alpha < 1.0 or self.hypotheses_tested < 1:
            raise ValueError("invalid spatial multiplicity")
        if tuple(scale.grid_size for scale in self.black_scales) != self.grid_sizes:
            raise ValueError("black spatial grid drift")
        if tuple(scale.grid_size for scale in self.white_scales) != self.grid_sizes:
            raise ValueError("white spatial grid drift")
        expected_alpha = self.alpha / (
            self.hypotheses_tested * sum(grid * grid for grid in self.grid_sizes)
        )
        if not math.isclose(self.per_cell_alpha, expected_alpha,
                            rel_tol=0.0, abs_tol=1e-18):
            raise ValueError("spatial per-cell alpha drift")
        expected = bool(
            self.black_scales[0].identifiable
            and self.white_scales[0].identifiable
            and all(scale.passed for scale in self.black_scales)
            and all(scale.passed for scale in self.white_scales)
        )
        if self.supported != expected:
            raise ValueError("spatial diffusion decision drift")
        if self.corruption_label_read or self.ground_truth_or_outcome_read:
            raise ValueError("spatial receipt cannot read labels or outcomes")


def _scale_receipt(
    events: np.ndarray,
    opportunities: np.ndarray,
    *,
    grid_size: int,
    per_cell_alpha: float,
) -> SpatialScaleReceipt:
    height, width = events.shape
    y_edges = np.linspace(0, height, grid_size + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, grid_size + 1, dtype=np.int64)
    event_counts = []
    opportunity_counts = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            ys = slice(int(y_edges[gy]), int(y_edges[gy + 1]))
            xs = slice(int(x_edges[gx]), int(x_edges[gx + 1]))
            event_counts.append(int(events[ys, xs].sum()))
            opportunity_counts.append(int(opportunities[ys, xs].sum()))
    total_events = int(events.sum())
    total_opportunities = int(opportunities.sum())
    required = []
    for count in opportunity_counts:
        if total_events <= 0 or total_opportunities <= 0 or count <= 0:
            required.append(False)
            continue
        probability = float(count / total_opportunities)
        empty_probability = (
            0.0 if probability >= 1.0
            else math.exp(total_events * math.log1p(-probability))
        )
        required.append(empty_probability <= per_cell_alpha)
    required_count = sum(required)
    occupied_required = sum(
        flag and count > 0 for flag, count in zip(required, event_counts)
    )
    return SpatialScaleReceipt(
        grid_size=grid_size,
        event_count=total_events,
        opportunity_count=total_opportunities,
        required_cells=required_count,
        occupied_required_cells=occupied_required,
        identifiable=required_count > 0,
        passed=occupied_required == required_count,
        event_counts=tuple(event_counts),
        opportunity_counts=tuple(opportunity_counts),
        required_cell_flags=tuple(required),
    )


def spatial_impulse_receipt(
    image: np.ndarray,
    *,
    endpoint: str,
    alpha: float = 0.05,
    hypotheses_tested: int = 16,
    grid_sizes: tuple[int, ...] = (4, 8, 16),
) -> SpatialImpulseReceipt:
    """Certify diffuse support for each salt-and-pepper polarity.

    Coarse-grid identifiability is required for both polarities.  At every
    scale, any cell that should be occupied with simultaneous confidence must
    actually contain an event; unpowered finer cells do not create a veto.
    """
    image = _image(image)
    if endpoint not in {"first", "second"}:
        raise ValueError("spatial impulse receipt needs a concrete endpoint")
    grids = tuple(int(value) for value in grid_sizes)
    if not grids or grids[0] < 2 or tuple(sorted(set(grids))) != grids:
        raise ValueError("grid sizes must be unique increasing integers")
    if not 0.0 < alpha < 1.0 or hypotheses_tested < 1:
        raise ValueError("invalid spatial multiplicity")
    height, width = image.shape[:2]
    valid = np.zeros((height, width), dtype=bool)
    valid[1:-1, 1:-1] = True
    action = (impulse_support(image) > 0.0) & valid
    median = cv2.medianBlur(image, 3)
    image_black = np.all(image <= 1, axis=2)
    image_white = np.all(image >= 254, axis=2)
    median_black = np.all(median <= 1, axis=2)
    median_white = np.all(median >= 254, axis=2)
    black_events = action & image_black & ~median_black
    white_events = action & image_white & ~median_white
    black_opportunities = valid & ~median_black
    white_opportunities = valid & ~median_white
    per_cell_alpha = alpha / (
        hypotheses_tested * sum(grid * grid for grid in grids)
    )
    black_scales = tuple(
        _scale_receipt(
            black_events, black_opportunities,
            grid_size=grid, per_cell_alpha=per_cell_alpha,
        ) for grid in grids
    )
    white_scales = tuple(
        _scale_receipt(
            white_events, white_opportunities,
            grid_size=grid, per_cell_alpha=per_cell_alpha,
        ) for grid in grids
    )
    supported = bool(
        black_scales[0].identifiable and white_scales[0].identifiable
        and all(scale.passed for scale in black_scales)
        and all(scale.passed for scale in white_scales)
    )
    return SpatialImpulseReceipt(
        endpoint=endpoint, alpha=float(alpha),
        hypotheses_tested=int(hypotheses_tested), grid_sizes=grids,
        per_cell_alpha=float(per_cell_alpha),
        black_event_count=int(black_events.sum()),
        white_event_count=int(white_events.sum()),
        black_event_hash=_hash(black_events),
        white_event_hash=_hash(white_events),
        black_scales=black_scales, white_scales=white_scales,
        supported=supported,
    )
