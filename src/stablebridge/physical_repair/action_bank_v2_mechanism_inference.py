"""Versioned mechanism inference adapter for the frozen E262 action bank.

The numerical seven-stratum inference remains the frozen Wave-1 procedure.
This adapter changes only the public-action ownership implied by E262:
disk/Gaussian are two internal mechanism strata of ``common_isotropic`` while
motion is an independent public arm.  Keeping this in a new module prevents
the sealed five-policy E256 result from silently changing meaning.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .action_qualification_inference import (
    MECHANISM_STRATA_V1,
    MechanismObservationV1,
    evaluate_wave1_mechanism_inference_v1,
)


ACTION_BANK_V2_MECHANISM_INFERENCE_SCHEMA = (
    "stablebridge-action-bank-v2-scene-disjoint-mechanism-inference/v1"
)

ACTION_TO_STRATA_V2 = {
    "paired_impulse_median3.endpoint_supported_v1": (
        "paired_impulse_median3",
    ),
    "paired_additive_wiener3.local_tile_sigma_v2": (
        "paired_additive_wiener3",
    ),
    "common_isotropic.local_recoverable_v3": (
        "common_disk",
        "common_gaussian",
    ),
    "common_motion.local_recoverable_v3": ("common_motion",),
    "jpeg_qcell_v3.local_macroblock_v3": ("jpeg_qcell_v3",),
    "jpeg_codec_path_v4.local_macroblock_v3": ("jpeg_codec_path_v4",),
}

MECHANISM_METRIC_BY_STRATUM_V2 = {
    "paired_impulse_median3": "impulse_anchor_deviation_reduction_v1",
    "paired_additive_wiener3": "active_tile_excess_noise_power_reduction_v1",
    "common_disk": "isotropic_psf_endpoint_discrepancy_reduction_v2",
    "common_gaussian": "isotropic_psf_endpoint_discrepancy_reduction_v2",
    "common_motion": "motion_psf_endpoint_discrepancy_reduction_v2",
    "jpeg_qcell_v3": "jpeg_qcell_block_exact_boundary_excess_reduction_v3",
    "jpeg_codec_path_v4": (
        "jpeg_codec_path_block_exact_boundary_excess_reduction_v3"
    ),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def evaluate_action_bank_v2_mechanisms(
    observations: Sequence[MechanismObservationV1],
    *,
    required_components_by_stratum: Mapping[str, int],
    power_plan_sha256: str,
    cohort_manifest_sha256: str,
    complete_matrix_receipt_sha256: str,
    chronology_receipt_sha256: str,
    action_bank_sha256: str,
) -> dict[str, Any]:
    """Evaluate seven mechanisms and map them to the six E262 nonnative arms.

    A public action passes this necessary mechanism gate only when every
    mechanism stratum it owns passes.  The result deliberately grants neither
    full scientific qualification nor selector admission.
    """

    bank_sha = _sha(action_bank_sha256, "action_bank_sha256")
    base = evaluate_wave1_mechanism_inference_v1(
        observations,
        required_components_by_stratum=required_components_by_stratum,
        power_plan_sha256=power_plan_sha256,
        cohort_manifest_sha256=cohort_manifest_sha256,
        complete_matrix_receipt_sha256=complete_matrix_receipt_sha256,
        chronology_receipt_sha256=chronology_receipt_sha256,
    )
    stratum_results = {
        row["mechanism_stratum"]: row for row in base["stratum_results"]
    }
    if set(stratum_results) != set(MECHANISM_STRATA_V1):
        raise RuntimeError("frozen seven-stratum inference output drifted")
    for stratum, result in stratum_results.items():
        # The bootstrap engine is version-invariant, but E262's blur and JPEG
        # mechanism estimands are not the legacy E256 metric versions.
        result["metric_id"] = MECHANISM_METRIC_BY_STRATUM_V2[stratum]
    action_results = [
        {
            "action_id": action_id,
            "required_mechanism_strata": list(strata),
            "mechanism_qualified": all(
                stratum_results[stratum]["mechanism_qualified"]
                for stratum in strata
            ),
        }
        for action_id, strata in ACTION_TO_STRATA_V2.items()
    ]
    payload = {
        key: value for key, value in base.items()
        if key not in {
            "schema", "receipt_sha256", "policy_results",
            "all_mechanisms_qualified",
        }
    }
    payload["schema"] = ACTION_BANK_V2_MECHANISM_INFERENCE_SCHEMA
    payload["protocol"] = {
        **payload["protocol"],
        "mechanism_metric_by_stratum": MECHANISM_METRIC_BY_STRATUM_V2,
        "public_action_to_strata": {
            action_id: list(strata)
            for action_id, strata in ACTION_TO_STRATA_V2.items()
        },
        "mapping_rule": (
            "common_disk and common_gaussian are internal strata of "
            "common_isotropic.local_recoverable_v3; common_motion is owned "
            "only by common_motion.local_recoverable_v3"
        ),
        "public_nonnative_action_count": 6,
        "native_control_action_id": "native",
    }
    payload["evidence_bindings"] = {
        **payload["evidence_bindings"],
        "action_bank_sha256": bank_sha,
        "wave1_numeric_inference_receipt_sha256": base["receipt_sha256"],
    }
    payload["action_results"] = action_results
    payload["all_public_action_mechanisms_qualified"] = all(
        row["mechanism_qualified"] for row in action_results
    )
    payload["scientific_qualification"] = False
    payload["selector_admission"] = False
    payload["production_authority"] = False
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


__all__ = [
    "ACTION_BANK_V2_MECHANISM_INFERENCE_SCHEMA",
    "ACTION_TO_STRATA_V2",
    "MECHANISM_METRIC_BY_STRATUM_V2",
    "evaluate_action_bank_v2_mechanisms",
]
