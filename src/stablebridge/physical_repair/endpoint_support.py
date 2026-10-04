"""Own-coordinate action supports plus their transported output support."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .support import _applicability, transport_second_mask_to_flow


@dataclass(frozen=True)
class EndpointApplicability:
    operator_id: str
    endpoint: str
    input_first: np.ndarray
    input_second: np.ndarray
    output_flow: np.ndarray
    second_transport_valid: np.ndarray

    def __post_init__(self) -> None:
        shape = np.asarray(self.input_first).shape
        if len(shape) != 2 or np.asarray(self.input_second).shape != shape:
            raise ValueError("endpoint applicability maps must share one HxW shape")
        for name in ("input_first", "input_second", "output_flow"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if (value.shape != shape or not np.isfinite(value).all()
                    or np.any(value < 0.0) or np.any(value > 1.0)):
                raise ValueError(f"{name} must be finite and lie in [0,1]")
            object.__setattr__(self, name, np.ascontiguousarray(value))
        valid = np.asarray(self.second_transport_valid, dtype=bool)
        if valid.shape != shape:
            raise ValueError("second transport validity must match supports")
        object.__setattr__(self, "second_transport_valid", np.ascontiguousarray(valid))


def observable_endpoint_applicability(first: np.ndarray, second: np.ndarray,
                                      flow: np.ndarray, *, operator_id: str,
                                      endpoint: str) -> EndpointApplicability:
    """Return input masks in each endpoint's lattice and output support in flow lattice."""
    first = np.asarray(first)
    second = np.asarray(second)
    flow = np.asarray(flow, dtype=np.float32)
    if (first.dtype != np.uint8 or second.dtype != np.uint8
            or first.ndim != 3 or first.shape != second.shape or first.shape[2] != 3
            or flow.shape != (*first.shape[:2], 2) or not np.isfinite(flow).all()
            or endpoint not in {"first", "second", "both"}):
        raise ValueError("invalid observed pair, flow, or endpoint")
    zeros = np.zeros(first.shape[:2], dtype=np.float32)
    first_map = (_applicability(first, operator_id)
                 if endpoint in {"first", "both"} else zeros.copy())
    second_map = (_applicability(second, operator_id)
                  if endpoint in {"second", "both"} else zeros.copy())
    transported, valid = transport_second_mask_to_flow(second_map, flow)
    output = np.maximum(first_map, transported)
    return EndpointApplicability(
        operator_id=operator_id, endpoint=endpoint,
        input_first=first_map, input_second=second_map,
        output_flow=output, second_transport_valid=valid,
    )

