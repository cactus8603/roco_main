"""Cost-reduced successor to the frozen E140 blur certificate.

Parameter search uses a deterministic 4-pixel subgrid on A-fold tiles.  After
one parameter per family has been selected, all B-fold own-null, family, and
endpoint bounds are recomputed with the exact full-resolution block median.
Thus the deployment certificate is unchanged in resolution; only nuisance
parameter fitting is cheaper.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .blur_parameter_certificates import (
    BlurFamilyEvidence,
    ENDPOINTS,
    FAMILYWISE_ALPHA,
    FAMILIES,
    MIN_BLOCKS_PER_SPLIT,
    MIN_ENDPOINT_GAIN,
    MIN_INFORMATION_RETENTION,
    SIMULTANEOUS_HYPOTHESES,
    TILE,
    _apply,
    _blocks,
    _estimated_parameters,
    _fit_mask,
    _gray,
    _identity_scores,
    _lcb,
    _model_scores,
    _observation,
    _parameter_key,
    _robust_se,
    _spatial_scores,
    _support_hash,
    _validate,
    _warp_second,
    parameter_grid,
    simultaneous_z,
)
from .contracts import ActionSpec, PhysicalCertificate


FIT_SAMPLE_STRIDE = 4


def _fit_spatial_scores(candidate: np.ndarray, target: np.ndarray,
                        support: np.ndarray,
                        blocks: Sequence[tuple[int, int, int]]) -> np.ndarray:
    # Importing the frozen affine helper keeps the nuisance model identical to
    # E140 while this successor changes only the fixed A-fold sampling lattice.
    from .blur_parameter_certificates import _affine

    gain, offset = _affine(candidate, target, support)
    adjusted = candidate.astype(np.float32) * gain.astype(np.float32)
    adjusted += offset.astype(np.float32)
    target = target.astype(np.float32)
    radius = 2
    scores = []
    for y0, x0, _ in blocks:
        target_patch = target[
            y0 + radius:y0 + TILE - radius:FIT_SAMPLE_STRIDE,
            x0 + radius:x0 + TILE - radius:FIT_SAMPLE_STRIDE,
        ]
        alternatives = []
        for dy in (-radius, 0, radius):
            for dx in (-radius, 0, radius):
                candidate_patch = adjusted[
                    y0 + radius + dy:y0 + TILE - radius + dy:FIT_SAMPLE_STRIDE,
                    x0 + radius + dx:x0 + TILE - radius + dx:FIT_SAMPLE_STRIDE,
                ]
                alternatives.append(np.mean(
                    np.abs(target_patch - candidate_patch), axis=2,
                ))
        # One partition call for all nine offsets avoids thousands of tiny
        # per-offset median dispatches while preserving the exact sampled data.
        medians = np.median(np.stack(alternatives, axis=0), axis=(1, 2)) / 255.0
        scores.append(float(np.min(medians)))
    return np.asarray(scores, dtype=np.float64)


def blur_parameter_certificates_v2(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    *,
    alpha: float = FAMILYWISE_ALPHA,
) -> dict[str, BlurFamilyEvidence]:
    """Use subsampled A-fold search and exact full-resolution B-fold checks."""
    first, second, flow = _validate(first, second, flow)
    warped_second, valid = _warp_second(second, flow)
    own_z = simultaneous_z(alpha, SIMULTANEOUS_HYPOTHESES)
    pairwise_z = simultaneous_z(alpha, 9)
    intermediates: dict[tuple[str, str], dict[str, object]] = {}

    for endpoint in ENDPOINTS:
        sharp, target = (
            (warped_second, first.astype(np.float32)) if endpoint == "first"
            else (first.astype(np.float32), warped_second)
        )
        blocks = _blocks(valid, sharp, target)
        gray_sharp, gray_target = _gray(sharp), _gray(target)
        observations = [
            (y0, x0, parity, _observation(
                gray_sharp[y0:y0 + TILE, x0:x0 + TILE],
                gray_target[y0:y0 + TILE, x0:x0 + TILE],
            ))
            for y0, x0, parity in blocks
        ]
        fit_indices = [index for index, row in enumerate(observations) if row[2] == 0]
        check_indices = [index for index, row in enumerate(observations) if row[2] == 1]
        fit_blocks = [blocks[index] for index in fit_indices]
        observed = [row[3] for row in observations]
        identity_spectral = _identity_scores(observed)
        fit_mask = _fit_mask(valid.shape, valid, blocks)
        identity_spatial = _spatial_scores(sharp, target, fit_mask, blocks)

        for family in FAMILIES:
            curve: dict[tuple[float, ...], float] = {}
            spectral_cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
            fit_cache: dict[tuple[float, ...], np.ndarray] = {}

            def evaluate(parameter: tuple[float, ...]) -> None:
                if parameter in spectral_cache:
                    return
                spectral_cache[parameter] = _model_scores(observed, family, parameter)
                fit_cache[parameter] = _fit_spatial_scores(
                    _apply(sharp, family, parameter), target, fit_mask, fit_blocks,
                )
                curve[parameter] = (
                    float(np.median(fit_cache[parameter]))
                    if fit_indices else math.inf
                )

            for parameter in parameter_grid(family):
                evaluate(parameter)
            if family == "motion" and curve:
                coarse = min(curve, key=lambda parameter: (curve[parameter], parameter))
                length, angle = int(round(coarse[0])), int(round(coarse[1]))
                refinements = {
                    (float(candidate_length), float(candidate_angle % 180))
                    for candidate_length in range(
                        max(3, length - 4), min(17, length + 4) + 1, 2,
                    )
                    for candidate_angle in (angle - 15, angle, angle + 15)
                }
                for parameter in sorted(refinements):
                    evaluate(parameter)
            selected = min(curve, key=lambda parameter: (
                curve[parameter],
                float(np.median(spectral_cache[parameter][0][fit_indices]))
                if fit_indices else math.inf,
                parameter,
            ))
            selected_fit = fit_cache[selected]
            identifiable = []
            for parameter in sorted(curve):
                difference = fit_cache[parameter] - selected_fit
                if (parameter == selected or not fit_indices
                        or float(np.median(difference))
                        <= own_z * _robust_se(difference)):
                    identifiable.append(parameter)

            selected_spectral, information = spectral_cache[selected]
            # Exact B-fold evidence is deliberately recomputed after selection.
            selected_spatial = _spatial_scores(
                _apply(sharp, family, selected), target, fit_mask, blocks,
            )
            spectral_delta = (
                identity_spectral[check_indices] - selected_spectral[check_indices]
            )
            spatial_delta = identity_spatial[check_indices] - selected_spatial[check_indices]
            check_information = information[check_indices]
            regions = tuple(
                (observations[index][0], observations[index][1])
                for local, index in enumerate(check_indices)
                if spatial_delta[local] > 0.0
                and check_information[local] >= MIN_INFORMATION_RETENTION
            )
            intermediates[(family, endpoint)] = {
                "blocks": blocks,
                "fit_indices": fit_indices,
                "check_indices": check_indices,
                "curve": curve,
                "selected": selected,
                "identifiable": tuple(identifiable),
                "spectral_scores": selected_spectral,
                "spatial_scores": selected_spatial,
                "information": information,
                "identity_spectral": identity_spectral,
                "identity_spatial": identity_spatial,
                "spectral_delta": spectral_delta,
                "spatial_delta": spatial_delta,
                "spectral_lcb": _lcb(spectral_delta, own_z),
                "spatial_lcb": _lcb(spatial_delta, own_z),
                "regions": regions,
            }

    output: dict[str, BlurFamilyEvidence] = {}
    for family in FAMILIES:
        for endpoint in ENDPOINTS:
            item = intermediates[(family, endpoint)]
            check = item["check_indices"]
            own_spectral = item["spectral_scores"][check]
            own_spatial = item["spatial_scores"][check]
            pairwise: dict[str, float] = {}
            for competitor in FAMILIES:
                if competitor == family:
                    continue
                other = intermediates[(competitor, endpoint)]
                pairwise[f"spectral_vs_{competitor}"] = _lcb(
                    other["spectral_scores"][check] - own_spectral, pairwise_z,
                )
                pairwise[f"spatial_vs_{competitor}"] = _lcb(
                    other["spatial_scores"][check] - own_spatial, pairwise_z,
                )
            other_endpoint = "second" if endpoint == "first" else "first"
            endpoint_lcb = _lcb(
                intermediates[(family, other_endpoint)]["spatial_scores"][check]
                - own_spatial,
                pairwise_z,
            )
            spatial_lcb = float(item["spatial_lcb"])
            spectral_lcb = float(item["spectral_lcb"])
            enough = len(check) >= MIN_BLOCKS_PER_SPLIT
            status = "supported" if enough and spatial_lcb > 0.0 else "rejected"
            information_median = (
                float(np.median(item["information"][check])) if len(check) else 0.0
            )
            recoverability = (
                "supported" if len(item["regions"]) >= MIN_BLOCKS_PER_SPLIT
                and information_median >= MIN_INFORMATION_RETENTION
                else "unsupported"
            )
            reasons = []
            if not enough:
                reasons.append("insufficient_independent_check_blocks")
            if spatial_lcb <= 0.0:
                reasons.append("simultaneous_spatial_own_null_not_rejected")
            selected = item["selected"]
            uncertainty = max(
                np.linalg.norm(np.asarray(parameter) - np.asarray(selected))
                for parameter in item["identifiable"]
            )
            modified = "second" if endpoint == "first" else "first"
            action = ActionSpec(
                f"common_{family}", "v2-subsampled-a-exact-b", "image",
                endpoint, modified, "flow_native",
            )
            certificate = PhysicalCertificate(
                action=action, status=status,
                observation_support_fraction=float(
                    len(item["blocks"]) * TILE * TILE
                    / max(first.shape[0] * first.shape[1], 1)
                ),
                identifiable_support_fraction=float(
                    len(item["regions"]) / max(len(check), 1)
                ),
                null_score=float(np.median(item["identity_spatial"][check]))
                if len(check) else 0.0,
                action_score=float(np.median(own_spatial)) if len(check) else 0.0,
                spatial_holdout_gain=float(np.median(item["spatial_delta"]))
                if len(check) else 0.0,
                parameter_uncertainty=float(uncertainty),
                fit_stability=float(np.mean(item["spatial_delta"] > 0.0))
                if len(check) else 0.0,
                competing_model_scores={
                    f"pairwise_lcb_vs_{name}": float(value)
                    for name, value in pairwise.items()
                } | {"endpoint_margin_lcb": endpoint_lcb},
                estimated_parameters=_estimated_parameters(family, selected),
                diagnostics={
                    "fit_blocks": float(len(item["fit_indices"])),
                    "check_blocks": float(len(check)),
                    "fit_sample_stride": float(FIT_SAMPLE_STRIDE),
                    "check_sample_stride": 1.0,
                    "simultaneous_z": own_z,
                    "pairwise_simultaneous_z": pairwise_z,
                    "minimum_endpoint_gain": MIN_ENDPOINT_GAIN,
                    "information_retention_median": information_median,
                    "recoverable_regions": float(len(item["regions"])),
                    "spectral_own_null_lcb": spectral_lcb,
                },
                rejection_reasons=tuple(reasons),
                calibration_version="blur-v2-subsampled-a-exact-b-bonferroni",
            )
            evidence = BlurFamilyEvidence(
                family=family, endpoint=endpoint,
                selected_parameter=selected,
                identifiable_parameters=item["identifiable"],
                fit_score_curve={
                    _parameter_key(parameter): float(score)
                    for parameter, score in item["curve"].items()
                },
                fit_support_hash=_support_hash(endpoint, 0, item["blocks"]),
                check_support_hash=_support_hash(endpoint, 1, item["blocks"]),
                check_blocks=len(check),
                own_null_improvement_median=float(np.median(item["spatial_delta"]))
                if len(check) else 0.0,
                own_null_improvement_lcb=spatial_lcb,
                own_null_spectral_improvement_lcb=spectral_lcb,
                pairwise_improvement_lcb=pairwise,
                endpoint_improvement_lcb=endpoint_lcb,
                information_retention_median=information_median,
                recoverable_regions=item["regions"],
                recoverability_status=recoverability,
                certificate=certificate,
            )
            output[evidence.action_key] = evidence
    return output
