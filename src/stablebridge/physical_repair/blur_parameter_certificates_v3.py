"""Same-parameter endpoint witness for the blur-certificate successor.

E141 showed that comparing independently optimized endpoint parameters can
manufacture a direction on a clean pair.  This version first runs the frozen
cost-reduced A/B certificate, then recomputes endpoint dominance with the same
family and the same selected parameter on both endpoint hypotheses.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .blur_parameter_certificates import (
    ENDPOINTS,
    FAMILYWISE_ALPHA,
    FAMILIES,
    _apply,
    _blocks,
    _fit_mask,
    _lcb,
    _spatial_scores,
    _validate,
    _warp_second,
    simultaneous_z,
)
from .blur_parameter_certificates_v2 import blur_parameter_certificates_v2


def action_specific_direct_winners(evidence) -> tuple[str, ...]:
    """Return an order-independent direct set with family-specific evidence.

    Motion requires its directional spectrum to beat disk and Gaussian in
    addition to the shared spatial forward tests.  If independently fitted
    candidates still create more than one winner at an endpoint, none of those
    candidates receives direct-delivery authority.
    """
    provisional = []
    for key, value in sorted(evidence.items()):
        if not value.direct_pairwise_winner:
            continue
        if value.family == "motion":
            spectral = [
                margin for name, margin in value.pairwise_improvement_lcb.items()
                if name.startswith("spectral_vs_")
            ]
            if not spectral or not all(margin > 0.0 for margin in spectral):
                continue
        provisional.append(key)
    counts = {
        endpoint: sum(key.endswith(f"@{endpoint}") for key in provisional)
        for endpoint in ENDPOINTS
    }
    return tuple(
        key for key in provisional
        if counts[key.rsplit("@", 1)[1]] == 1
    )


def blur_parameter_certificates_v3(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    *,
    alpha: float = FAMILYWISE_ALPHA,
):
    """Return v2 evidence with a paired same-parameter endpoint margin."""
    first, second, flow = _validate(first, second, flow)
    evidence = blur_parameter_certificates_v2(first, second, flow, alpha=alpha)
    warped_second, valid = _warp_second(second, flow)
    contexts = {}
    for endpoint in ENDPOINTS:
        sharp, target = (
            (warped_second, first.astype(np.float32)) if endpoint == "first"
            else (first.astype(np.float32), warped_second)
        )
        blocks = _blocks(valid, sharp, target)
        fit_mask = _fit_mask(valid.shape, valid, blocks)
        contexts[endpoint] = (sharp, target, blocks, fit_mask)

    pairwise_z = simultaneous_z(alpha, 9)
    output = {}
    for family in FAMILIES:
        for endpoint in ENDPOINTS:
            key = f"common_{family}@{endpoint}"
            current = evidence[key]
            parameter = current.selected_parameter
            scores = {}
            for hypothesis in ENDPOINTS:
                sharp, target, blocks, fit_mask = contexts[hypothesis]
                values = _spatial_scores(
                    _apply(sharp, family, parameter), target, fit_mask, blocks,
                )
                scores[hypothesis] = {
                    (y0, x0): float(values[index])
                    for index, (y0, x0, parity) in enumerate(blocks)
                    if parity == 1
                }
            other = "second" if endpoint == "first" else "first"
            common = sorted(set(scores[endpoint]) & set(scores[other]))
            delta = np.asarray([
                scores[other][coordinate] - scores[endpoint][coordinate]
                for coordinate in common
            ], dtype=np.float64)
            endpoint_lcb = _lcb(delta, pairwise_z)
            competing = dict(current.certificate.competing_model_scores)
            competing["endpoint_margin_lcb"] = endpoint_lcb
            diagnostics = dict(current.certificate.diagnostics)
            diagnostics.update({
                "endpoint_same_parameter": 1.0,
                "endpoint_common_check_blocks": float(len(common)),
            })
            certificate = replace(
                current.certificate,
                action=replace(
                    current.certificate.action,
                    operator_version="v3-same-parameter-endpoint",
                ),
                competing_model_scores=competing,
                diagnostics=diagnostics,
                calibration_version="blur-v3-same-parameter-endpoint-bonferroni",
            )
            output[key] = replace(
                current,
                endpoint_improvement_lcb=endpoint_lcb,
                certificate=certificate,
            )
    return output
