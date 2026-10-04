"""Endpoint-aware mechanism estimands for action-bank qualification.

The blur-v3 actions are explicitly cross-endpoint equalizers: a certificate
identifies the degraded endpoint and the action writes the selected PSF to the
opposite endpoint.  Their mechanism receipt therefore exists on the executed
endpoint.  Treating the unchanged generator endpoint as the primary estimand
would make the advertised mechanism algebraically zero by construction.

Noise and JPEG actions remain generator-targeted repairs.  For those actions,
an expected endpoint omitted from a receipt contributes a structural zero only
when the caller supplies an exact byte-identity proof.

This module performs no I/O and reads no task outcome or ground truth.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence


ENDPOINTS = ("first", "second")
BLUR_CROSS_ENDPOINT_STRATA = frozenset({
    "common_disk",
    "common_gaussian",
    "common_motion",
})


def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def reduce_mechanism_endpoints_v2(
    *,
    mechanism_stratum: str,
    generator_expected_endpoints: Sequence[str],
    endpoint_records: Sequence[Mapping[str, Any]],
    identity_proof: Callable[[str], bool],
) -> dict[str, Any]:
    """Reduce endpoint receipts under the action's frozen mechanism semantics.

    Cross-endpoint blur equalization is evaluated where the descriptor writes.
    Every other action is evaluated on the controlled generator endpoint(s).
    The returned executed diagnostic is always based on the receipt records and
    is kept separate from the primary estimand.
    """

    expected = tuple(generator_expected_endpoints)
    if (
        not expected
        or len(set(expected)) != len(expected)
        or any(endpoint not in ENDPOINTS for endpoint in expected)
    ):
        raise ValueError("generator expected endpoints must be unique first/second")
    if not endpoint_records:
        raise ValueError("mechanism endpoint records are missing")

    indexed: dict[str, Mapping[str, Any]] = {}
    executed_active: list[float] = []
    executed_specificity: list[float] = []
    for record in endpoint_records:
        endpoint = record.get("endpoint")
        if endpoint not in ENDPOINTS or endpoint in indexed:
            raise ValueError("mechanism endpoint records must be unique first/second")
        if record.get("outside_support_byte_identity") is not True:
            raise ValueError("mechanism record lacks outside-support identity")
        active = _finite(
            record.get("active_relative_reduction"),
            "active_relative_reduction",
        )
        specificity = _finite(
            record.get(
                "public_arm_specificity_difference",
                record.get("specificity_difference"),
            ),
            "specificity_difference",
        )
        indexed[str(endpoint)] = record
        executed_active.append(active)
        executed_specificity.append(specificity)

    executed_endpoints = tuple(indexed)
    structural_zeros: list[str] = []
    if mechanism_stratum in BLUR_CROSS_ENDPOINT_STRATA:
        if len(executed_endpoints) != 1:
            raise ValueError("cross-endpoint blur mechanism needs one executed endpoint")
        evaluated = executed_endpoints
        primary_active = list(executed_active)
        primary_specificity = list(executed_specificity)
        endpoint_basis = "EXECUTED_CROSS_ENDPOINT_EQUALIZATION"
    else:
        evaluated = expected
        primary_active = []
        primary_specificity = []
        for endpoint in expected:
            record = indexed.get(endpoint)
            if record is None:
                if not identity_proof(endpoint):
                    raise ValueError(
                        "unmodified generator endpoint lacks byte-identity proof"
                    )
                primary_active.append(0.0)
                primary_specificity.append(0.0)
                structural_zeros.append(endpoint)
                continue
            primary_active.append(_finite(
                record.get("active_relative_reduction"),
                "active_relative_reduction",
            ))
            primary_specificity.append(_finite(
                record.get(
                    "public_arm_specificity_difference",
                    record.get("specificity_difference"),
                ),
                "specificity_difference",
            ))
        endpoint_basis = "GENERATOR_TARGETED_REPAIR"

    return {
        "mechanism_estimand_endpoint_basis": endpoint_basis,
        "generator_expected_endpoints": list(expected),
        "executed_endpoints": list(executed_endpoints),
        "evaluated_endpoints": list(evaluated),
        "structural_zero_target_endpoints": structural_zeros,
        "off_generator_endpoint_writes": [
            endpoint for endpoint in executed_endpoints if endpoint not in expected
        ],
        "primary": sum(primary_active) / len(primary_active),
        "specificity": sum(primary_specificity) / len(primary_specificity),
        "executed_diagnostic_primary": (
            sum(executed_active) / len(executed_active)
        ),
        "executed_diagnostic_specificity": (
            sum(executed_specificity) / len(executed_specificity)
        ),
    }


__all__ = [
    "BLUR_CROSS_ENDPOINT_STRATA",
    "ENDPOINTS",
    "reduce_mechanism_endpoints_v2",
]
