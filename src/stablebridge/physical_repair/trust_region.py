"""Final output composition relative to an immutable native prediction."""
from __future__ import annotations

import numpy as np

from .contracts import BoundedOutput


def bounded_output(native: np.ndarray, candidate: np.ndarray, support: np.ndarray,
                   *, alpha: float = 0.5, epsilon: float = 4.0,
                   tolerance: float = 1e-4) -> BoundedOutput:
    """Compose one candidate and cap the final update in native flow units.

    Unsupported pixels are assigned directly from ``native``.  A non-finite
    candidate therefore cannot contaminate the fallback region.
    """
    native = np.asarray(native, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    support = np.asarray(support, dtype=np.float32)
    if (native.ndim != 3 or native.shape[2] != 2 or candidate.shape != native.shape
            or support.shape != native.shape[:2] or not np.isfinite(native).all()
            or not np.isfinite(support).all() or np.any(support < 0.0)
            or np.any(support > 1.0)):
        raise ValueError("invalid native, candidate, or support")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and nonnegative")
    active = support > 0.0
    finite_candidate = np.isfinite(candidate).all(axis=2)
    active &= finite_candidate
    delta = np.zeros_like(native, dtype=np.float32)
    delta[active] = ((candidate[active] - native[active])
                     * support[active, None] * np.float32(alpha))
    norm = np.linalg.norm(delta.astype(np.float64), axis=2)
    # Project slightly inside the mathematical boundary so float32 addition to
    # a large native displacement does not round the reconstructed update past
    # epsilon.  The guard is part of the explicit numerical contract.
    inner_epsilon = max(float(epsilon) - float(tolerance), 0.0)
    scale = np.minimum(1.0, inner_epsilon / np.maximum(norm, 1e-30))
    delta *= scale[..., None].astype(np.float32)
    output = native.copy()
    output[active] = native[active] + delta[active]
    update_norm = np.linalg.norm(
        output.astype(np.float64) - native.astype(np.float64), axis=2,
    ).astype(np.float32)
    maximum = float(update_norm.max(initial=0.0))
    if maximum > epsilon + tolerance:
        raise RuntimeError("final update exceeded the requested trust region")
    fallback = ~active
    if not np.array_equal(output[fallback], native[fallback]):
        raise RuntimeError("native fallback is not bit exact")
    return BoundedOutput(
        flow=output, update_norm=update_norm, exact_fallback=fallback,
        alpha=float(alpha), epsilon=float(epsilon), max_update_norm=maximum,
    )
