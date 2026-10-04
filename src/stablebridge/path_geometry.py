"""Compose native-pixel correspondence paths without using labels.

Computability means that all required samples and destinations are inside
observed image support. It says nothing about visibility, surface identity,
or prediction correctness. Every displacement is already expressed in global
native pixel units; this module performs no resize or crop-coordinate scaling.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .geometry import in_bounds, sample_numpy


@dataclass(frozen=True)
class DenseEdge:
    """A directed field sampled over the source crop's native pixel centres.

    ``field`` has shape [2,H,W], with (dx,dy) in global native coordinates.
    ``target_hw`` defaults to the source field's height and width. The optional
    ``observed_mask`` declares available source samples, not an occlusion mask
    or a certificate that a correspondence follows the same surface.
    """

    field: np.ndarray
    source_node: str
    target_node: str
    source_origin_xy: tuple = (0.0, 0.0)
    target_origin_xy: tuple = (0.0, 0.0)
    target_hw: Optional[tuple] = None
    observed_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        field = np.asarray(self.field)
        if (field.ndim != 3 or field.shape[0] != 2 or min(field.shape[1:]) < 1
                or not np.issubdtype(field.dtype, np.number)
                or np.issubdtype(field.dtype, np.complexfloating)):
            raise ValueError("field must be a real numeric array with shape [2,H,W]")
        object.__setattr__(self, "field", field)
        for name in ("source_node", "target_node"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("source_origin_xy", "target_origin_xy"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain two finite values")
            object.__setattr__(self, name, tuple(value))
        hw = field.shape[1:] if self.target_hw is None else self.target_hw
        if (len(hw) != 2 or not np.isfinite(hw).all() or min(hw) < 1
                or any(int(value) != value for value in hw)):
            raise ValueError("target_hw must contain positive integer height and width")
        object.__setattr__(self, "target_hw", tuple(int(value) for value in hw))
        if self.observed_mask is not None:
            if np.shape(self.observed_mask) != field.shape[1:]:
                raise ValueError("observed_mask must match the field's HxW source support")
            object.__setattr__(self, "observed_mask", np.asarray(self.observed_mask, dtype=bool))


def compose_path(query_xy, edges: Sequence[DenseEdge]):
    """Follow each directed edge at the coordinate reached by its predecessor.

    Inputs are [N,2] global native query coordinates and a nonempty edge chain.
    Bilinear sampling requires every nonzero-weight corner to be finite and
    observed. An edge is computable only if its source sample and its resulting
    destination are in their declared crops. A failed path cannot become valid
    again at a later edge. No out-of-bounds coordinate is clipped into support.

    Returns ``coordinates`` [K+1,N,2], cumulative ``edge_computable`` [K,N],
    ``computable`` [N], and ``raw_displacement`` [N,2]. Failed paths have zero
    displacement placeholders and freeze at their last computable coordinate;
    their masks must be used when forming a candidate bank. Nonfinite initial
    queries use zero coordinate placeholders and remain invalid throughout.
    """
    query = np.asarray(query_xy, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("query_xy must have shape [N,2]")
    edges = tuple(edges)
    if not edges:
        raise ValueError("A path must contain at least one edge")
    if not all(isinstance(edge, DenseEdge) for edge in edges):
        raise TypeError("Every path edge must be a DenseEdge")
    for previous, current in zip(edges, edges[1:]):
        if previous.target_node != current.source_node:
            raise ValueError("Adjacent path edges must share target/source node identity")

    alive = np.isfinite(query).all(axis=1)
    initial = np.where(alive[:, None], query, 0.0)
    position = initial.copy()
    coordinates = [position.copy()]
    edge_computable = []
    for edge in edges:
        sampled, source_valid = sample_numpy(
            edge.field, position - edge.source_origin_xy, edge.observed_mask)
        with np.errstate(over="ignore", invalid="ignore"):
            proposed = position + sampled.T
        destination_valid = in_bounds(proposed - edge.target_origin_xy, edge.target_hw)
        alive = alive & source_valid & destination_valid
        position = np.where(alive[:, None], proposed, position)
        coordinates.append(position.copy())
        edge_computable.append(alive.copy())

    displacement = np.where(alive[:, None], position - initial, 0.0)
    return {
        "raw_displacement": displacement,
        "coordinates": np.stack(coordinates),
        "edge_computable": np.stack(edge_computable),
        "computable": alive.copy(),
    }


def project_stereo_path(raw_displacement, computable, tolerance=1.0):
    """Offer a horizontal candidate only under an explicit epipolar tolerance.

    ``candidate`` retains the horizontal displacement sign (left-to-right is
    normally negative); it is not positive disparity magnitude. Eligibility
    requires computability, finite raw components, and abs(dy) <= tolerance in
    native pixels. The raw vector and signed vertical residual are returned
    separately: projection never certifies that the full path is correct.
    Ineligible entries must not be inserted as usable candidates. This test
    has no dependence on agreement with an existing direct prediction.
    """
    raw = np.asarray(raw_displacement, dtype=np.float64)
    mask = np.asarray(computable, dtype=bool)
    if raw.ndim != 2 or raw.shape[1] != 2 or mask.shape != raw.shape[:1]:
        raise ValueError("Expected raw_displacement [N,2] and computable [N]")
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative native-pixel value")
    finite = np.isfinite(raw).all(axis=1)
    eligible = mask & finite & (np.abs(raw[:, 1]) <= tolerance)
    candidate = np.zeros_like(raw)
    candidate[:, 0] = np.where(finite, raw[:, 0], 0.0)
    return {
        "candidate": candidate,
        "eligible": eligible,
        "vertical_residual": raw[:, 1].copy(),
        "raw_displacement": raw.copy(),
    }
