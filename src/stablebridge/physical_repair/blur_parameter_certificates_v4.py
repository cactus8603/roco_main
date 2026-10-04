"""Rejected shared-registration-nuisance diagnostic for the blur certificate.

E143 exposed clean false-direct rows even after endpoint parameters were
coupled.  In v3, identity and a smoothing action could each choose a different
best local translation.  Smoothing is less sensitive to misregistration, so
that comparison can mistake geometry tolerance for blur evidence.  This
diagnostic lets identity choose the nuisance offset and evaluates the action at
that exact offset on held-out blocks.  It did not change either clean false
direct exposed during E143, so it is retained as a negative mechanism test and
must not be treated as a delivery-ready selector successor.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .blur_parameter_certificates import (
    ENDPOINTS,
    FAMILYWISE_ALPHA,
    FAMILIES,
    _affine,
    _apply,
    _blocks,
    _fit_mask,
    _lcb,
    _validate,
    _warp_second,
    simultaneous_z,
)
from .blur_parameter_certificates_v3 import (
    action_specific_direct_winners,
    blur_parameter_certificates_v3,
)


OFFSETS = tuple((dy, dx) for dy in (-2, 0, 2) for dx in (-2, 0, 2))
MECHANISM_STATUS = "REJECTED_ON_E143_CLEAN_COUNTEREXAMPLES"


def _spatial_offset_scores(
    candidate: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    blocks: list[tuple[int, int, int]],
) -> np.ndarray:
    """Return one residual per block and fixed registration offset."""
    gain, offset = _affine(candidate, target, support)
    adjusted = candidate.astype(np.float32) * gain.astype(np.float32)
    adjusted += offset.astype(np.float32)
    target = target.astype(np.float32)
    radius = 2
    scores = np.empty((len(blocks), len(OFFSETS)), dtype=np.float64)
    for index, (y0, x0, _) in enumerate(blocks):
        target_patch = target[
            y0 + radius:y0 + 64 - radius,
            x0 + radius:x0 + 64 - radius,
        ]
        for column, (dy, dx) in enumerate(OFFSETS):
            candidate_patch = adjusted[
                y0 + radius + dy:y0 + 64 - radius + dy,
                x0 + radius + dx:x0 + 64 - radius + dx,
            ]
            scores[index, column] = float(np.median(np.mean(
                np.abs(target_patch - candidate_patch), axis=2,
            ))) / 255.0
    return scores


def action_specific_direct_winners_v4(evidence) -> tuple[str, ...]:
    """Keep a v3 direct action only with positive shared-geometry evidence."""
    return tuple(
        key for key in action_specific_direct_winners(evidence)
        if evidence[key].certificate.competing_model_scores[
            "shared_geometry_own_lcb"
        ] > 0.0
    )


def blur_parameter_certificates_v4(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    *,
    alpha: float = FAMILYWISE_ALPHA,
):
    """Return v3 evidence plus a simultaneous same-offset own-null margin."""
    first, second, flow = _validate(first, second, flow)
    evidence = blur_parameter_certificates_v3(first, second, flow, alpha=alpha)
    warped_second, valid = _warp_second(second, flow)
    z_value = simultaneous_z(alpha, len(FAMILIES) * len(ENDPOINTS))
    contexts = {}
    for endpoint in ENDPOINTS:
        sharp, target = (
            (warped_second, first.astype(np.float32)) if endpoint == "first"
            else (first.astype(np.float32), warped_second)
        )
        blocks = _blocks(valid, sharp, target)
        fit_mask = _fit_mask(valid.shape, valid, blocks)
        identity = _spatial_offset_scores(sharp, target, fit_mask, blocks)
        contexts[endpoint] = (sharp, target, blocks, fit_mask, identity)

    output = {}
    for family in FAMILIES:
        for endpoint in ENDPOINTS:
            key = f"common_{family}@{endpoint}"
            current = evidence[key]
            sharp, target, blocks, fit_mask, identity = contexts[endpoint]
            action = _spatial_offset_scores(
                _apply(sharp, family, current.selected_parameter),
                target, fit_mask, blocks,
            )
            check = np.asarray([parity == 1 for _, _, parity in blocks], dtype=bool)
            identity_check = identity[check]
            action_check = action[check]
            if len(identity_check):
                chosen = np.argmin(identity_check, axis=1)
                rows = np.arange(len(chosen))
                delta = identity_check[rows, chosen] - action_check[rows, chosen]
            else:
                delta = np.asarray([], dtype=np.float64)
            shared_lcb = _lcb(delta, z_value)
            competing = dict(current.certificate.competing_model_scores)
            competing["shared_geometry_own_lcb"] = shared_lcb
            diagnostics = dict(current.certificate.diagnostics)
            diagnostics.update({
                "shared_geometry_offsets": float(len(OFFSETS)),
                "shared_geometry_check_blocks": float(len(delta)),
                "shared_geometry_simultaneous_z": z_value,
                "identity_selected_registration": 1.0,
            })
            certificate = replace(
                current.certificate,
                action=replace(
                    current.certificate.action,
                    operator_version="v4-shared-registration-nuisance",
                ),
                competing_model_scores=competing,
                diagnostics=diagnostics,
                calibration_version="blur-v4-shared-registration-bonferroni",
            )
            output[key] = replace(current, certificate=certificate)
    return output
