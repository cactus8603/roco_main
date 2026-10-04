"""Scene-disjoint mechanism inference for the Wave-1 action bank.

This module deliberately answers only the physical-mechanism question.  It
does not inspect optical-flow ground truth, task utility, pixel harm, cost, or
child observables, and therefore cannot grant scientific qualification or
selector authority.  The input rows are expected to be derived from sealed
mechanism receipts after the complete action-by-eligible-case matrix has been
executed.

The seven physical strata map to five runtime policies.  Disk, Gaussian, and
motion remain separate non-poolable gates for the one routed blur policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


MECHANISM_INFERENCE_SCHEMA_V1 = (
    "stablebridge-wave1-scene-disjoint-mechanism-inference/v1"
)
BOOTSTRAP_REPEATS_V1 = 10_000
BOOTSTRAP_SEED_V1 = 20_261_004
FAMILYWISE_ALPHA_V1 = 0.05
MINIMUM_POINT_REDUCTION_V1 = 0.05
OUTER_FOLDS_V1 = (0, 1, 2, 3, 4)

MECHANISM_STRATA_V1 = (
    "paired_impulse_median3",
    "paired_additive_wiener3",
    "common_disk",
    "common_gaussian",
    "common_motion",
    "jpeg_qcell_v3",
    "jpeg_codec_path_v4",
)

MECHANISM_METRIC_BY_STRATUM_V1 = {
    "paired_impulse_median3": "impulse_anchor_deviation_reduction_v1",
    "paired_additive_wiener3": "active_tile_excess_noise_power_reduction_v1",
    "common_disk": "disk_otf_endpoint_discrepancy_reduction_v1",
    "common_gaussian": "gaussian_otf_endpoint_discrepancy_reduction_v1",
    "common_motion": "motion_otf_endpoint_discrepancy_reduction_v1",
    "jpeg_qcell_v3": "jpeg_qcell_aligned_boundary_excess_reduction_v2",
    "jpeg_codec_path_v4": (
        "jpeg_codec_path_aligned_boundary_excess_reduction_v2"
    ),
}

POLICY_TO_STRATA_V1 = {
    "paired_impulse_median3.endpoint_supported_v1": (
        "paired_impulse_median3",
    ),
    "paired_additive_wiener3.local_tile_sigma_v2": (
        "paired_additive_wiener3",
    ),
    "blur_v3_cross_endpoint.local_recoverable_v2": (
        "common_disk", "common_gaussian", "common_motion",
    ),
    "jpeg_qcell_v3.local_macroblock_v2": ("jpeg_qcell_v3",),
    "jpeg_codec_path_v4.local_macroblock_v2": ("jpeg_codec_path_v4",),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value: object, name: str) -> str:
    value = _text(value, name)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class MechanismObservationV1:
    """One available case-level mechanism observation.

    Multiple cases may belong to one connected component.  They are averaged
    within that component before any inferential weighting, so case density
    cannot change the estimand.
    """

    case_id: str
    component_id: str
    source_dataset: str
    outer_fold: int
    mechanism_stratum: str
    active_relative_reduction: float
    specificity_difference: float
    mechanism_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("case_id", "component_id", "source_dataset"):
            _text(getattr(self, name), name)
        if self.mechanism_stratum not in MECHANISM_STRATA_V1:
            raise ValueError("unknown Wave-1 mechanism stratum")
        if (
            isinstance(self.outer_fold, bool)
            or not isinstance(self.outer_fold, int)
            or self.outer_fold not in OUTER_FOLDS_V1
        ):
            raise ValueError("outer_fold must be one of the frozen five folds")
        object.__setattr__(
            self,
            "active_relative_reduction",
            _finite(self.active_relative_reduction, "active_relative_reduction"),
        )
        object.__setattr__(
            self,
            "specificity_difference",
            _finite(self.specificity_difference, "specificity_difference"),
        )
        _sha(self.mechanism_receipt_sha256, "mechanism_receipt_sha256")


def _validate_power_plan(
    required_components_by_stratum: Mapping[str, int],
) -> dict[str, int]:
    if set(required_components_by_stratum) != set(MECHANISM_STRATA_V1):
        raise ValueError("power plan must cover exactly the seven mechanism strata")
    result: dict[str, int] = {}
    for stratum in MECHANISM_STRATA_V1:
        value = required_components_by_stratum[stratum]
        if isinstance(value, bool) or not isinstance(value, int) or value < 20:
            raise ValueError("each frozen power-plan minimum must be an integer >= 20")
        result[stratum] = value
    return result


def _component_values(
    rows: Sequence[MechanismObservationV1],
    stratum: str,
    field: str,
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.mechanism_stratum != stratum:
            continue
        values.setdefault(row.source_dataset, {}).setdefault(
            row.component_id, [],
        ).append(float(getattr(row, field)))
    return {
        source: {
            component: float(np.mean(case_values, dtype=np.float64))
            for component, case_values in sorted(components.items())
        }
        for source, components in sorted(values.items())
    }


def _dataset_equal_point(values: Mapping[str, Mapping[str, float]]) -> float:
    return float(np.mean([
        np.mean(tuple(components.values()), dtype=np.float64)
        for components in values.values()
    ], dtype=np.float64))


def _shared_bootstrap_uniforms(
    rows: Sequence[MechanismObservationV1],
) -> dict[str, np.ndarray]:
    maximum_by_source: dict[str, int] = {}
    for stratum in MECHANISM_STRATA_V1:
        grouped = _component_values(rows, stratum, "active_relative_reduction")
        for source, components in grouped.items():
            maximum_by_source[source] = max(
                maximum_by_source.get(source, 0), len(components),
            )
    generator = np.random.default_rng(BOOTSTRAP_SEED_V1)
    return {
        source: generator.random((BOOTSTRAP_REPEATS_V1, maximum_by_source[source]))
        for source in sorted(maximum_by_source)
    }


def _bootstrap(
    values: Mapping[str, Mapping[str, float]],
    uniforms: Mapping[str, np.ndarray],
) -> tuple[float, float, float, float]:
    point = _dataset_equal_point(values)
    source_samples = []
    for source, components in values.items():
        array = np.asarray(tuple(components.values()), dtype=np.float64)
        n = len(array)
        indices = np.minimum((uniforms[source][:, :n] * n).astype(np.int64), n - 1)
        source_samples.append(np.mean(array[indices], axis=1, dtype=np.float64))
    samples = np.mean(np.stack(source_samples, axis=1), axis=1, dtype=np.float64)
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    # One-sided basic-bootstrap p-value for H0: theta <= 0.  The +1 rule
    # prevents an impossible zero p-value and makes the finite resampling
    # experiment explicit.
    centered = samples - point
    p_value = float(
        (1 + np.count_nonzero(centered <= -point))
        / (BOOTSTRAP_REPEATS_V1 + 1)
    )
    return point, float(lower), float(upper), p_value


def _sample_audit(
    rows: Sequence[MechanismObservationV1],
    stratum: str,
    required_components: int,
) -> dict[str, Any]:
    selected = [row for row in rows if row.mechanism_stratum == stratum]
    components: dict[str, tuple[str, int]] = {}
    cases_per_component: dict[str, int] = {}
    for row in selected:
        identity = (row.source_dataset, row.outer_fold)
        previous = components.setdefault(row.component_id, identity)
        if previous != identity:
            raise ValueError("a connected component crossed source or outer-fold identity")
        cases_per_component[row.component_id] = cases_per_component.get(row.component_id, 0) + 1
    by_source: dict[str, int] = {}
    by_fold = {str(fold): 0 for fold in OUTER_FOLDS_V1}
    for source, fold in components.values():
        by_source[source] = by_source.get(source, 0) + 1
        by_fold[str(fold)] += 1
    checks = {
        "minimum_components": len(components) >= required_components,
        "minimum_source_datasets": len(by_source) >= 2,
        "minimum_components_per_source": (
            bool(by_source) and min(by_source.values()) >= 2
        ),
        "minimum_components_per_outer_fold": min(by_fold.values()) >= 4,
    }
    return {
        "case_count": len(selected),
        "component_count": len(components),
        "required_component_count": required_components,
        "source_dataset_count": len(by_source),
        "components_by_source_dataset": dict(sorted(by_source.items())),
        "components_by_outer_fold": by_fold,
        "maximum_cases_per_component": max(cases_per_component.values(), default=0),
        "checks": checks,
        "sample_ready": all(checks.values()),
    }


def _holm_step_down(tests: list[dict[str, Any]]) -> None:
    order = sorted(
        range(len(tests)),
        key=lambda index: (tests[index]["p_value"], index),
    )
    still_rejecting = True
    m = len(tests)
    for rank, index in enumerate(order, start=1):
        threshold = FAMILYWISE_ALPHA_V1 / (m - rank + 1)
        reject = bool(still_rejecting and tests[index]["p_value"] <= threshold)
        if not reject:
            still_rejecting = False
        tests[index]["holm_rank"] = rank
        tests[index]["holm_threshold"] = threshold
        tests[index]["holm_reject"] = reject


def evaluate_wave1_mechanism_inference_v1(
    observations: Sequence[MechanismObservationV1],
    *,
    required_components_by_stratum: Mapping[str, int],
    power_plan_sha256: str,
    cohort_manifest_sha256: str,
    complete_matrix_receipt_sha256: str,
    chronology_receipt_sha256: str,
) -> dict[str, Any]:
    """Evaluate all seven mechanism strata under the frozen Wave-1 design.

    The returned ``all_mechanisms_qualified`` flag is necessary but never
    sufficient for scientific qualification.  Task efficacy, tail/region
    safety, coverage, prospective cost, child observables, and independent
    replay remain separate mandatory gates.
    """

    rows = tuple(observations)
    if any(not isinstance(row, MechanismObservationV1) for row in rows):
        raise ValueError("observations must be typed MechanismObservationV1 rows")
    requirements = _validate_power_plan(required_components_by_stratum)
    for value, name in (
        (power_plan_sha256, "power_plan_sha256"),
        (cohort_manifest_sha256, "cohort_manifest_sha256"),
        (complete_matrix_receipt_sha256, "complete_matrix_receipt_sha256"),
        (chronology_receipt_sha256, "chronology_receipt_sha256"),
    ):
        _sha(value, name)

    case_keys: set[tuple[str, str]] = set()
    global_component_identity: dict[str, tuple[str, int]] = {}
    for row in rows:
        key = (row.mechanism_stratum, row.case_id)
        if key in case_keys:
            raise ValueError("duplicate case within mechanism stratum")
        case_keys.add(key)
        identity = (row.source_dataset, row.outer_fold)
        previous = global_component_identity.setdefault(row.component_id, identity)
        if previous != identity:
            raise ValueError("component identity drifted across mechanism strata")

    rows = tuple(sorted(rows, key=lambda row: (
        row.mechanism_stratum,
        row.source_dataset,
        row.component_id,
        row.case_id,
    )))
    uniforms = _shared_bootstrap_uniforms(rows)
    stratum_results: dict[str, dict[str, Any]] = {}
    ordered_tests: list[dict[str, Any]] = []

    for stratum in MECHANISM_STRATA_V1:
        audit = _sample_audit(rows, stratum, requirements[stratum])
        if audit["sample_ready"]:
            primary = _bootstrap(
                _component_values(rows, stratum, "active_relative_reduction"),
                uniforms,
            )
            specificity = _bootstrap(
                _component_values(rows, stratum, "specificity_difference"),
                uniforms,
            )
        else:
            primary = specificity = (math.nan, math.nan, math.nan, 1.0)

        result = {
            "mechanism_stratum": stratum,
            "metric_id": MECHANISM_METRIC_BY_STRATUM_V1[stratum],
            "sample_audit": audit,
            "primary": {
                "estimand": "dataset_equal_component_equal_active_relative_reduction",
                "point": None if not audit["sample_ready"] else primary[0],
                "ci95_lower": None if not audit["sample_ready"] else primary[1],
                "ci95_upper": None if not audit["sample_ready"] else primary[2],
                "p_value": primary[3],
                "minimum_point": MINIMUM_POINT_REDUCTION_V1,
            },
            "specificity": {
                "estimand": "dataset_equal_component_equal_specificity_difference",
                "point": None if not audit["sample_ready"] else specificity[0],
                "ci95_lower": None if not audit["sample_ready"] else specificity[1],
                "ci95_upper": None if not audit["sample_ready"] else specificity[2],
                "p_value": specificity[3],
            },
        }
        stratum_results[stratum] = result
        for test_name in ("primary", "specificity"):
            ordered_tests.append({
                "mechanism_stratum": stratum,
                "test": test_name,
                "p_value": result[test_name]["p_value"],
            })

    _holm_step_down(ordered_tests)
    tests_by_key = {
        (row["mechanism_stratum"], row["test"]): row
        for row in ordered_tests
    }
    for stratum, result in stratum_results.items():
        for test_name in ("primary", "specificity"):
            result[test_name]["holm"] = {
                name: tests_by_key[(stratum, test_name)][name]
                for name in ("holm_rank", "holm_threshold", "holm_reject")
            }
        result["mechanism_qualified"] = bool(
            result["sample_audit"]["sample_ready"]
            and result["primary"]["point"] >= MINIMUM_POINT_REDUCTION_V1
            and result["primary"]["ci95_lower"] > 0.0
            and result["specificity"]["ci95_lower"] > 0.0
            and result["primary"]["holm"]["holm_reject"]
            and result["specificity"]["holm"]["holm_reject"]
        )

    policy_results = {
        policy: {
            "required_mechanism_strata": list(strata),
            "mechanism_qualified": all(
                stratum_results[stratum]["mechanism_qualified"]
                for stratum in strata
            ),
        }
        for policy, strata in POLICY_TO_STRATA_V1.items()
    }
    all_qualified = all(
        row["mechanism_qualified"] for row in policy_results.values()
    )
    payload = {
        "schema": MECHANISM_INFERENCE_SCHEMA_V1,
        "protocol": {
            "mechanism_strata": list(MECHANISM_STRATA_V1),
            "mechanism_metric_by_stratum": MECHANISM_METRIC_BY_STRATUM_V1,
            "policy_to_strata": {
                key: list(value) for key, value in POLICY_TO_STRATA_V1.items()
            },
            "unit": "connected_component_of_group_id_through_physical_pair_id",
            "aggregation": "case_equal_then_component_equal_then_source_dataset_equal",
            "bootstrap": {
                "repeats": BOOTSTRAP_REPEATS_V1,
                "seed": BOOTSTRAP_SEED_V1,
                "source_stratified": True,
                "shared_uniform_stream": True,
                "ci": "two_sided_percentile_95",
                "p_value": "one_sided_basic_bootstrap_plus_one_H0_theta_le_0",
            },
            "multiplicity": {
                "method": "HOLM_STEP_DOWN",
                "familywise_alpha": FAMILYWISE_ALPHA_V1,
                "ordered_test_count": 14,
            },
            "minimum_point_active_relative_reduction": (
                MINIMUM_POINT_REDUCTION_V1
            ),
            "outer_folds": list(OUTER_FOLDS_V1),
        },
        "evidence_bindings": {
            "power_plan_sha256": power_plan_sha256,
            "cohort_manifest_sha256": cohort_manifest_sha256,
            "complete_matrix_receipt_sha256": complete_matrix_receipt_sha256,
            "chronology_receipt_sha256": chronology_receipt_sha256,
        },
        "required_components_by_stratum": requirements,
        "stratum_results": [
            stratum_results[stratum] for stratum in MECHANISM_STRATA_V1
        ],
        "holm_tests": ordered_tests,
        "policy_results": [
            {"policy_id": policy, **result}
            for policy, result in policy_results.items()
        ],
        "all_mechanisms_qualified": all_qualified,
        "scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
        "remaining_mandatory_gates": [
            "task_efficacy",
            "pixel_and_region_tail_safety",
            "coverage_and_typed_missing",
            "prospective_cost_ceiling",
            "region_projected_CC_RR_CR_RC_child_observables",
            "independent_replay",
        ],
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


__all__ = [
    "BOOTSTRAP_REPEATS_V1",
    "BOOTSTRAP_SEED_V1",
    "MECHANISM_INFERENCE_SCHEMA_V1",
    "MECHANISM_METRIC_BY_STRATUM_V1",
    "MECHANISM_STRATA_V1",
    "MechanismObservationV1",
    "POLICY_TO_STRATA_V1",
    "evaluate_wave1_mechanism_inference_v1",
]
