"""Outcome-blind E234 feature builder and deterministic five-head engine.

This module implements the recipe sealed by
``E234_p05_b1_parent_recipe_preexecution_v2``.  It contains no source opener
and grants no payload authority.  Callers are responsible for enforcing the
stage/read boundary; all numerical and schema violations here fail closed.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


RAW_WIDTH = 19
DESIGN_WIDTH = 39
HEAD_ORDER = ("B", "H", "harmed_fraction", "CVaR95", "severe")
HURDLE_HEADS = ("B", "H", "harmed_fraction", "CVaR95")
SUPPORT_CONDITIONED_HEADS = ("B", "H", "CVaR95")
CALIBRATION_HEAD_ORDER = ("net", "H", "harmed_fraction", "CVaR95")
RISK_HEADS = ("H", "harmed_fraction", "CVaR95")
ACTION_FAMILIES = (
    "common_disk",
    "common_gaussian",
    "common_motion",
    "jpeg_deblock",
    "p05_noise_impulse_median3",
    "p05_noise_additive_wiener3",
)
FIXED_NONCONFORMITY_SCALES = {
    "net": 0.25,
    "H": 0.25,
    "harmed_fraction": 1.0,
    "CVaR95": 0.25,
}
FALLBACK_LEVELS = (
    ("family", "endpoint", "strength_bits"),
    ("family", "strength_bits"),
    ("strength_bits",),
    ("GLOBAL",),
)

RIDGE_LAMBDA = 1.0
SEVERE_CUTOFF = -0.25
LOGISTIC_MAX_ITERATIONS = 200
LOGISTIC_GRADIENT_INFINITY_TOLERANCE = 1e-10
LOGISTIC_RELATIVE_OBJECTIVE_TOLERANCE = 1e-12
LOGISTIC_ARMIJO_C1 = 1e-4
LOGISTIC_BACKTRACKING = 0.5
LOGISTIC_MINIMUM_STEP = 2.0**-30
RETENTION_MINIMUM = 0.90
RETENTION_SOLE_TOLERANCE = 1e-12
INCIDENT_CASE_ID = "x05p01-38373743ba091f2ff7dc7b4e"
INCIDENT_SHA256 = "8cde55b44a964ba02d734db9b29077969c34adf88bee1e3feeb3788ac5970db3"
INCIDENT_VERIFICATION_SHA256 = (
    "1c29459da004031b62259c8071f650fd081e578bc69f9febf34aa756a0ca8cb9"
)
ZERO_READ_BOUNDARY = {
    "GPU_used": False,
    "GT_arrays_or_values_read": 0,
    "H2_rows_or_values_read": 0,
    "outcome_rows_read": 0,
    "target_payloads_decoded": 0,
    "target_values_read": 0,
}


class E234EngineError(RuntimeError):
    """A contract or numerical failure that makes the outer fold native."""


def _normalize_zero(value: float) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise E234EngineError("FINITE_BINARY64_REQUIRED")
    return 0.0 if value == 0.0 else value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise E234EngineError(f"NUMBER_REQUIRED:{name}")
    result = float(value)
    if not math.isfinite(result):
        raise E234EngineError(f"FINITE_REQUIRED:{name}")
    return _normalize_zero(result)


def _binary(value: Any, name: str) -> float:
    result = _number(value, name)
    if result not in (0.0, 1.0):
        raise E234EngineError(f"BINARY_REQUIRED:{name}")
    return result


def _normalize_tree(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return _normalize_zero(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_tree(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise E234EngineError("CANONICAL_JSON_STRING_KEYS_REQUIRED")
        return {key: _normalize_tree(item) for key, item in value.items()}
    raise E234EngineError(f"NON_CANONICAL_JSON_TYPE:{type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the frozen compact CPython JSON form plus one LF byte."""
    normalized = _normalize_tree(value)
    text = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_recursive_synthetic_boundary(value: Any) -> None:
    """Reject real-source and incident metadata at any nesting depth."""
    forbidden_keys = {
        "payload",
        "payload_path",
        "payload_bytes",
        "payload_metadata",
        "source",
        "source_path",
        "source_shard",
        "source_shard_path",
        "shard_path",
        "outcome",
        "outcome_path",
        "evaluation_outcome",
        "gt",
        "gt_array",
        "gt_arrays",
        "h2",
        "h2_rows",
        "real_payload",
        "real_source",
    }

    def walk(item: Any, location: str) -> None:
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise E234EngineError(f"SYNTHETIC_BOUNDARY_NONSTRING_KEY:{location}")
            lowered = {key.casefold(): key for key in item}
            collision = forbidden_keys.intersection(lowered)
            if collision:
                raise E234EngineError(
                    f"SYNTHETIC_BOUNDARY_FORBIDDEN_FIELD:{location}:{sorted(collision)[0]}"
                )
            if "observation_id" in item and item.get("fixture_scope") != "SYNTHETIC_ADVERSARIAL_ONLY":
                raise E234EngineError(f"SYNTHETIC_FIXTURE_SCOPE_REQUIRED:{location}")
            if "fixture_scope" in item and item.get("fixture_scope") != "SYNTHETIC_ADVERSARIAL_ONLY":
                raise E234EngineError(f"NON_SYNTHETIC_FIXTURE_SCOPE:{location}")
            if item.get("case_id") == INCIDENT_CASE_ID:
                raise E234EngineError(f"INCIDENT_CASE_FORBIDDEN:{location}")
            if "incident_case_excluded" in item and item.get("incident_case_excluded") is not True:
                raise E234EngineError(f"INCIDENT_EXCLUSION_FALSE:{location}")
            if "zero_excluded_row_reads" in item and item.get("zero_excluded_row_reads") is not True:
                raise E234EngineError(f"EXCLUDED_ROW_READ_DRIFT:{location}")
            if "incident_hash" in item and item.get("incident_hash") != INCIDENT_SHA256:
                raise E234EngineError(f"INCIDENT_HASH_DRIFT:{location}")
            if (
                "verification_hash" in item
                and item.get("verification_hash") != INCIDENT_VERIFICATION_SHA256
            ):
                raise E234EngineError(f"INCIDENT_VERIFICATION_HASH_DRIFT:{location}")
            for key, nested in item.items():
                walk(nested, f"{location}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                walk(nested, f"{location}[{index}]")
            return
        _normalize_tree(item)

    walk(value, "$synthetic")


def _synthetic_case_set_digest(*values: Any) -> str:
    case_ids: set[str] = set()

    def collect(item: Any) -> None:
        if isinstance(item, Mapping):
            case_id = item.get("case_id")
            if case_id is not None:
                if not isinstance(case_id, str) or not case_id:
                    raise E234EngineError("SYNTHETIC_CASE_ID_DRIFT")
                case_ids.add(case_id)
            for nested in item.values():
                collect(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                collect(nested)

    for value in values:
        collect(value)
    return canonical_sha256(sorted(case_ids))


def make_synthetic_receipt(
    *,
    stage: str,
    request: Any,
    output: Any,
    implementation_sha256: str,
) -> dict[str, Any]:
    validate_recursive_synthetic_boundary(request)
    validate_recursive_synthetic_boundary(output)
    if not isinstance(stage, str) or not stage:
        raise E234EngineError("SYNTHETIC_RECEIPT_STAGE_REQUIRED")
    if (
        not isinstance(implementation_sha256, str)
        or len(implementation_sha256) != 64
        or any(character not in "0123456789abcdef" for character in implementation_sha256)
    ):
        raise E234EngineError("IMPLEMENTATION_SHA256_REQUIRED")
    receipt = {
        "schema": "e234-b1-synthetic-core-engine-receipt/v3",
        "stage": stage,
        "scope": "SYNTHETIC_ADVERSARIAL_ONLY",
        "request_sha256": canonical_sha256(request),
        "output_sha256": canonical_sha256(output),
        "implementation_sha256": implementation_sha256,
        "incident_case_excluded": True,
        "incident_hash": INCIDENT_SHA256,
        "verification_hash": INCIDENT_VERIFICATION_SHA256,
        "selected_case_set_digest": _synthetic_case_set_digest(request, output),
        "zero_excluded_row_reads": True,
        "read_boundary": dict(ZERO_READ_BOUNDARY),
    }
    validate_synthetic_receipt(
        receipt,
        request=request,
        output=output,
        implementation_sha256=implementation_sha256,
    )
    return receipt


def validate_synthetic_receipt(
    receipt: Mapping[str, Any],
    *,
    request: Any,
    output: Any,
    implementation_sha256: str,
) -> None:
    validate_recursive_synthetic_boundary(request)
    validate_recursive_synthetic_boundary(output)
    expected_keys = {
        "schema",
        "stage",
        "scope",
        "request_sha256",
        "output_sha256",
        "implementation_sha256",
        "incident_case_excluded",
        "incident_hash",
        "verification_hash",
        "selected_case_set_digest",
        "zero_excluded_row_reads",
        "read_boundary",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        raise E234EngineError("SYNTHETIC_RECEIPT_KEY_SET_DRIFT")
    if (
        receipt.get("schema") != "e234-b1-synthetic-core-engine-receipt/v3"
        or receipt.get("scope") != "SYNTHETIC_ADVERSARIAL_ONLY"
        or not isinstance(receipt.get("stage"), str)
        or not receipt.get("stage")
    ):
        raise E234EngineError("SYNTHETIC_RECEIPT_TYPED_STATE_DRIFT")
    if (
        receipt.get("request_sha256") != canonical_sha256(request)
        or receipt.get("output_sha256") != canonical_sha256(output)
        or receipt.get("implementation_sha256") != implementation_sha256
    ):
        raise E234EngineError("SYNTHETIC_RECEIPT_SUBJECT_DRIFT")
    if (
        receipt.get("incident_case_excluded") is not True
        or receipt.get("incident_hash") != INCIDENT_SHA256
        or receipt.get("verification_hash") != INCIDENT_VERIFICATION_SHA256
        or receipt.get("selected_case_set_digest")
        != _synthetic_case_set_digest(request, output)
        or receipt.get("zero_excluded_row_reads") is not True
        or receipt.get("read_boundary") != ZERO_READ_BOUNDARY
    ):
        raise E234EngineError("SYNTHETIC_RECEIPT_INCIDENT_OR_BOUNDARY_DRIFT")
    validate_recursive_synthetic_boundary(receipt)


def _nested(row: Mapping[str, Any], key: str, subkey: str) -> Any:
    value = row.get(key)
    if not isinstance(value, Mapping) or subkey not in value:
        raise E234EngineError(f"MISSING_FIELD:{key}.{subkey}")
    return value[subkey]


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    observation_id = row.get("observation_id")
    component_id = row.get("component_id")
    if not isinstance(observation_id, str) or not observation_id:
        raise E234EngineError("OBSERVATION_ID_REQUIRED")
    if not isinstance(component_id, str) or not component_id:
        raise E234EngineError("COMPONENT_ID_REQUIRED")
    return observation_id, component_id


def build_raw_features(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact 19 raw columns using outcome-blind receipt fields only."""
    observation_id, component_id = _identity(row)
    family = _nested(row, "typed_action_signature", "family")
    if family not in ACTION_FAMILIES:
        raise E234EngineError("UNKNOWN_ACTION_FAMILY")

    raw: list[float | None] = [
        1.0 if family == known else 0.0 for known in ACTION_FAMILIES
    ]
    raw.append(
        _binary(
            _nested(row, "typed_action_signature", "jpeg_variant"),
            "jpeg_variant",
        )
    )

    endpoint = _nested(row, "action", "endpoint")
    if endpoint not in ("first", "second", "both"):
        raise E234EngineError("UNKNOWN_ENDPOINT")
    raw.extend(
        [1.0 if endpoint == "second" else 0.0, 1.0 if endpoint == "both" else 0.0]
    )

    stage = _nested(row, "action", "stage")
    if stage not in ("full", "half"):
        raise E234EngineError("UNKNOWN_STAGE")
    raw.append(1.0 if stage == "half" else 0.0)
    raw.append(
        _normalize_zero(
            _number(_nested(row, "action", "input_strength"), "input_strength") - 1.0
        )
    )

    alpha_availability = row.get("output_alpha_availability")
    if alpha_availability == "AVAILABLE_EXACT":
        raw.append(
            _normalize_zero(
                _number(row.get("output_alpha_mean"), "output_alpha_mean") - 1.0
            )
        )
        alpha_missing = 0.0
    elif alpha_availability == "TYPED_MISSING":
        if row.get("output_alpha_mean") is not None:
            raise E234EngineError("TYPED_MISSING_ALPHA_HAS_VALUE")
        raw.append(None)
        alpha_missing = 1.0
    else:
        raise E234EngineError("UNKNOWN_ALPHA_AVAILABILITY")

    response_availability = row.get("bounded_response_availability")
    if response_availability == "AVAILABLE_EXACT":
        response = _number(row.get("bounded_response_mean_px"), "bounded_response_mean_px")
        if response < 0.0:
            raise E234EngineError("NEGATIVE_BOUNDED_RESPONSE")
        raw.extend([response, _normalize_zero(math.log1p(response))])
        response_missing = 0.0
    elif response_availability == "TYPED_MISSING":
        if row.get("bounded_response_mean_px") is not None:
            raise E234EngineError("TYPED_MISSING_RESPONSE_HAS_VALUE")
        raw.extend([None, None])
        response_missing = 1.0
    else:
        raise E234EngineError("UNKNOWN_RESPONSE_AVAILABILITY")
    raw.extend([alpha_missing, response_missing])

    area = _number(row.get("area_px"), "area_px")
    if area < 0.0:
        raise E234EngineError("NEGATIVE_AREA")
    support_fraction = _number(
        _nested(row, "support", "support_fraction"), "support_fraction"
    )
    supported_px = _number(_nested(row, "support", "supported_px"), "supported_px")
    if not 0.0 <= support_fraction <= 1.0:
        raise E234EngineError("SUPPORT_FRACTION_OUT_OF_RANGE")
    if supported_px < 0.0 or not supported_px.is_integer():
        raise E234EngineError("INVALID_SUPPORTED_PX")
    raw.extend(
        [
            _normalize_zero(math.log1p(area)),
            support_fraction,
            1.0 if supported_px > 0.0 else 0.0,
        ]
    )
    if len(raw) != RAW_WIDTH:
        raise AssertionError("internal raw width drift")

    missing = [1.0 if value is None else 0.0 for value in raw]
    expected_missing = {11: alpha_missing, 12: response_missing, 13: response_missing}
    if any(mask != expected_missing.get(index, 0.0) for index, mask in enumerate(missing)):
        raise E234EngineError("UNAUTHORIZED_RAW_MISSING")
    feature_valid = not any(missing)
    return {
        "observation_id": observation_id,
        "component_id": component_id,
        "raw": raw,
        "transform_missing": missing,
        "feature_valid": feature_valid,
        "disposition": (
            "ELIGIBLE" if feature_valid else "ABSTAIN_NATIVE_FEATURE_MISSING"
        ),
    }


def _median(
    records: Sequence[tuple[float, str]],
) -> tuple[float, list[str], list[float]]:
    if not records:
        raise E234EngineError("EMPTY_MEDIAN")
    ordered = sorted(
        ((_normalize_zero(float(value)), observation_id) for value, observation_id in records),
        key=lambda item: (item[0], item[1]),
    )
    if any(not isinstance(observation_id, str) for _, observation_id in ordered):
        raise E234EngineError("MEDIAN_OBSERVATION_ID_REQUIRED")
    n = len(ordered)
    if n % 2:
        value, observation_id = ordered[n // 2]
        return value, [observation_id], [value]
    lower = ordered[n // 2 - 1]
    upper = ordered[n // 2]
    value = _normalize_zero((0.5 * lower[0]) + (0.5 * upper[0]))
    return value, [lower[1], upper[1]], [lower[0], upper[0]]


def _require_unique_observations(rows: Sequence[Mapping[str, Any]]) -> None:
    ids = [_identity(row)[0] for row in rows]
    if len(ids) != len(set(ids)):
        raise E234EngineError("DUPLICATE_OBSERVATION_ID")


def fit_feature_transform(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit row-unweighted median/MAD statistics on exact MODEL rows."""
    if not rows:
        raise E234EngineError("EMPTY_MODEL_TRANSFORM")
    ordered_rows = sorted(rows, key=lambda row: _identity(row)[0])
    _require_unique_observations(ordered_rows)
    if any(row.get("role") != "MODEL" for row in ordered_rows):
        raise E234EngineError("MODEL_ROLE_IDENTITY_DRIFT")
    if any(row.get("availability") != "AVAILABLE_EXACT" for row in ordered_rows):
        raise E234EngineError("MODEL_ROW_NOT_AVAILABLE_EXACT")
    built = [build_raw_features(row) for row in ordered_rows]
    if any(not record["feature_valid"] for record in built):
        raise E234EngineError("MODEL_FEATURE_MISSING")

    centers: list[float] = []
    scales: list[float] = []
    raw_constant: list[bool] = []
    statistics: list[dict[str, Any]] = []
    for index in range(RAW_WIDTH):
        values = [
            (float(record["raw"][index]), record["observation_id"]) for record in built
        ]
        center, center_ids, center_values = _median(values)
        deviations = [
            (_normalize_zero(abs(value - center)), observation_id)
            for value, observation_id in values
        ]
        mad, mad_ids, mad_values = _median(deviations)
        scale = 1.0 if mad <= 1e-6 else mad
        minimum = min(value for value, _ in values)
        maximum = max(value for value, _ in values)
        constant = _normalize_zero(minimum) == _normalize_zero(maximum)
        centers.append(center)
        scales.append(scale)
        raw_constant.append(constant)
        statistics.append(
            {
                "raw_index": index,
                "center": center,
                "center_observation_ids": center_ids,
                "center_values": center_values,
                "mad": mad,
                "mad_observation_ids": mad_ids,
                "mad_values": mad_values,
                "scale": scale,
                "minimum": minimum,
                "maximum": maximum,
                "constant_raw": constant,
            }
        )
    fixed_zero_columns = [
        index + 1 for index, constant in enumerate(raw_constant) if constant
    ] + list(range(20, 39))
    artifact = {
        "schema": "e234-b1-19-to-39-transform/v2",
        "raw_width": RAW_WIDTH,
        "encoded_width": DESIGN_WIDTH,
        "population_observation_ids": [record["observation_id"] for record in built],
        "population_component_ids": sorted({record["component_id"] for record in built}),
        "centers": centers,
        "scales": scales,
        "statistics": statistics,
        "fixed_zero_columns": sorted(fixed_zero_columns),
        "intercept_fixed_zero": False,
        "row_weighting": "ONE_PER_ROW_NO_COMPONENT_OR_HEAD_WEIGHTING",
    }
    artifact["membership_digest"] = canonical_sha256(
        {
            "observation_ids": artifact["population_observation_ids"],
            "component_ids": artifact["population_component_ids"],
        }
    )
    return artifact


def apply_feature_transform(
    row: Mapping[str, Any], transform: Mapping[str, Any]
) -> dict[str, Any]:
    if transform.get("schema") != "e234-b1-19-to-39-transform/v2":
        raise E234EngineError("TRANSFORM_SCHEMA_DRIFT")
    if transform.get("raw_width") != RAW_WIDTH or transform.get("encoded_width") != DESIGN_WIDTH:
        raise E234EngineError("TRANSFORM_WIDTH_DRIFT")
    centers = transform.get("centers")
    scales = transform.get("scales")
    if not isinstance(centers, list) or not isinstance(scales, list):
        raise E234EngineError("TRANSFORM_ARRAYS_REQUIRED")
    if len(centers) != RAW_WIDTH or len(scales) != RAW_WIDTH:
        raise E234EngineError("TRANSFORM_SHAPE_DRIFT")

    built = build_raw_features(row)
    design = [1.0]
    for index, raw in enumerate(built["raw"]):
        center = _number(centers[index], f"center_{index}")
        scale = _number(scales[index], f"scale_{index}")
        if scale <= 0.0:
            raise E234EngineError("NONPOSITIVE_TRANSFORM_SCALE")
        filled = center if raw is None else float(raw)
        design.append(_normalize_zero((filled - center) / scale))
    design.extend(built["transform_missing"])
    if len(design) != DESIGN_WIDTH or not all(math.isfinite(value) for value in design):
        raise E234EngineError("ENCODED_DESIGN_INVALID")
    return {**built, "design": design}


def component_weights(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    if not rows:
        raise E234EngineError("EMPTY_MODEL_POPULATION")
    _require_unique_observations(rows)
    counts: dict[str, int] = {}
    for row in rows:
        _, component_id = _identity(row)
        counts[component_id] = counts.get(component_id, 0) + 1
    preliminary = np.asarray(
        [1.0 / counts[_identity(row)[1]] for row in rows], dtype=np.float64
    )
    preliminary_mean = float(np.mean(preliminary))
    weights = preliminary / preliminary_mean
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise E234EngineError("INVALID_COMPONENT_WEIGHTS")
    records = [
        {
            "observation_id": _identity(row)[0],
            "component_id": _identity(row)[1],
            "weight": _normalize_zero(float(weight)),
        }
        for row, weight in zip(rows, weights, strict=True)
    ]
    artifact = {
        "component_counts": dict(sorted(counts.items())),
        "preliminary_mean": preliminary_mean,
        "weight_records": records,
        "weights": [_normalize_zero(float(value)) for value in weights],
        "weight_sum": _normalize_zero(float(np.sum(weights))),
        "weight_mean": _normalize_zero(float(np.mean(weights))),
    }
    return weights, artifact


def _finite_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise E234EngineError(f"NUMERIC_ARRAY_REQUIRED:{name}")
    array = raw.astype(np.float64, copy=False)
    if ndim is not None and array.ndim != ndim:
        raise E234EngineError(f"DIMENSION_DRIFT:{name}")
    if not np.all(np.isfinite(array)):
        raise E234EngineError(f"NONFINITE:{name}")
    return array


def _active_columns(fixed_zero_columns: Sequence[int]) -> np.ndarray:
    if any(isinstance(index, bool) or not isinstance(index, int) for index in fixed_zero_columns):
        raise E234EngineError("FIXED_ZERO_COLUMN_TYPE_DRIFT")
    fixed = set(fixed_zero_columns)
    if len(fixed) != len(fixed_zero_columns):
        raise E234EngineError("DUPLICATE_FIXED_ZERO_COLUMN")
    if 0 in fixed or any(index < 0 or index >= DESIGN_WIDTH for index in fixed):
        raise E234EngineError("FIXED_ZERO_COLUMN_DRIFT")
    return np.asarray(
        [index for index in range(DESIGN_WIDTH) if index not in fixed],
        dtype=np.int64,
    )


def _penalty(active: np.ndarray) -> np.ndarray:
    return np.diag(
        np.asarray([0.0 if int(index) == 0 else 1.0 for index in active], dtype=np.float64)
    )


def _sigmoid(predictor: np.ndarray) -> np.ndarray:
    clipped = np.clip(predictor, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logistic_objective(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    beta: np.ndarray,
    penalty: np.ndarray,
) -> float:
    predictor = design @ beta
    losses = (
        np.maximum(predictor, 0.0)
        - target * predictor
        + np.log1p(np.exp(-np.abs(predictor)))
    )
    value = float(
        np.sum(weights * losses) / np.sum(weights)
        + 0.5 * RIDGE_LAMBDA * (beta @ penalty @ beta)
    )
    if not math.isfinite(value):
        raise E234EngineError("NONFINITE_LOGISTIC_OBJECTIVE")
    return value


def fit_logistic(
    design: Any,
    target: Any,
    weights: Any,
    fixed_zero_columns: Sequence[int],
    *,
    max_iterations: int = LOGISTIC_MAX_ITERATIONS,
) -> dict[str, Any]:
    design = _finite_array(design, "LOGISTIC_DESIGN", ndim=2)
    target = _finite_array(target, "LOGISTIC_TARGET", ndim=1)
    weights = _finite_array(weights, "LOGISTIC_WEIGHTS", ndim=1)
    if (
        design.shape != (len(target), DESIGN_WIDTH)
        or weights.shape != target.shape
        or len(target) == 0
        or np.any((target != 0.0) & (target != 1.0))
        or np.any(weights <= 0.0)
    ):
        raise E234EngineError("INVALID_LOGISTIC_INPUT")
    _active_columns(fixed_zero_columns)
    if np.all(target == target[0]):
        probability = _normalize_zero(
            float((np.sum(weights * target) + 0.5) / (np.sum(weights) + 1.0))
        )
        return {
            "state": "CONSTANT_DEGENERATE_CLASS",
            "coefficients": [0.0] * DESIGN_WIDTH,
            "constant_probability": probability,
            "iterations": 0,
            "optimization_executed": False,
        }
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise E234EngineError("LOGISTIC_NONCONVERGENCE")

    active = _active_columns(fixed_zero_columns)
    active_design = design[:, active]
    penalty = _penalty(active)
    beta = np.zeros(len(active), dtype=np.float64)
    weight_sum = float(np.sum(weights))
    objective = _logistic_objective(active_design, target, weights, beta, penalty)
    for iteration in range(1, max_iterations + 1):
        probability = _sigmoid(active_design @ beta)
        gradient = (
            active_design.T @ (weights * (probability - target)) / weight_sum
            + RIDGE_LAMBDA * (penalty @ beta)
        )
        if not np.all(np.isfinite(gradient)):
            raise E234EngineError("LOGISTIC_NONFINITE_GRADIENT")
        if float(np.max(np.abs(gradient))) <= LOGISTIC_GRADIENT_INFINITY_TOLERANCE:
            return _expanded_logistic_fit(beta, active, iteration - 1)
        curvature = weights * probability * (1.0 - probability)
        hessian = (
            (active_design.T * curvature) @ active_design / weight_sum
            + RIDGE_LAMBDA * penalty
        )
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise E234EngineError("LOGISTIC_SINGULAR_SOLVE") from exc
        descent = float(gradient @ delta)
        if not np.all(np.isfinite(delta)) or not math.isfinite(descent) or descent <= 0.0:
            raise E234EngineError("LOGISTIC_INVALID_DESCENT")
        step = 1.0
        accepted = False
        while step >= LOGISTIC_MINIMUM_STEP:
            proposed = beta - step * delta
            proposed_objective = _logistic_objective(
                active_design, target, weights, proposed, penalty
            )
            if proposed_objective <= objective - LOGISTIC_ARMIJO_C1 * step * descent:
                accepted = True
                break
            step *= LOGISTIC_BACKTRACKING
        if not accepted:
            raise E234EngineError("LOGISTIC_LINE_SEARCH_FAILURE")
        relative = abs(objective - proposed_objective) / max(1.0, abs(objective))
        beta = proposed
        objective = proposed_objective
        if relative <= LOGISTIC_RELATIVE_OBJECTIVE_TOLERANCE:
            return _expanded_logistic_fit(beta, active, iteration)
    raise E234EngineError("LOGISTIC_NONCONVERGENCE")


def _expanded_logistic_fit(
    beta: np.ndarray, active: np.ndarray, iterations: int
) -> dict[str, Any]:
    full = np.zeros(DESIGN_WIDTH, dtype=np.float64)
    full[active] = beta
    return {
        "state": "FITTED",
        "coefficients": [_normalize_zero(float(value)) for value in full],
        "constant_probability": None,
        "iterations": iterations,
        "optimization_executed": True,
    }


def fit_ridge(
    design: Any,
    target: Any,
    weights: Any,
    fixed_zero_columns: Sequence[int],
) -> dict[str, Any]:
    design = _finite_array(design, "RIDGE_DESIGN", ndim=2)
    target = _finite_array(target, "RIDGE_TARGET", ndim=1)
    weights = _finite_array(weights, "RIDGE_WEIGHTS", ndim=1)
    if (
        design.shape != (len(target), DESIGN_WIDTH)
        or target.shape != weights.shape
        or len(target) == 0
        or np.any(weights <= 0.0)
    ):
        raise E234EngineError("INVALID_RIDGE_INPUT")
    active = _active_columns(fixed_zero_columns)
    active_design = design[:, active]
    penalty = _penalty(active)
    weight_sum = float(np.sum(weights))
    matrix = (
        (active_design.T * weights) @ active_design / weight_sum
        + RIDGE_LAMBDA * penalty
    )
    vector = active_design.T @ (weights * target) / weight_sum
    try:
        beta = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError as exc:
        raise E234EngineError("RIDGE_SINGULAR_SOLVE") from exc
    if not np.all(np.isfinite(beta)):
        raise E234EngineError("RIDGE_NONFINITE_COEFFICIENT")
    full = np.zeros(DESIGN_WIDTH, dtype=np.float64)
    full[active] = beta
    return {
        "state": "FITTED",
        "coefficients": [_normalize_zero(float(value)) for value in full],
        "optimization_executed": True,
    }


def _target(row: Mapping[str, Any], name: str) -> float:
    if row.get("target_availability") != "AVAILABLE_EXACT":
        raise E234EngineError("TARGET_NOT_AVAILABLE_EXACT")
    targets = row.get("targets")
    if not isinstance(targets, Mapping) or name not in targets:
        raise E234EngineError(f"TARGET_MISSING:{name}")
    value = _number(targets[name], f"target_{name}")
    if name != "net_gain" and value < 0.0:
        raise E234EngineError(f"NEGATIVE_TARGET:{name}")
    if name == "harmed_fraction" and value > 1.0:
        raise E234EngineError("FRACTION_TARGET_OUT_OF_RANGE")
    return value


def fit_hurdle(
    design: np.ndarray,
    target: np.ndarray,
    supports: np.ndarray,
    weights: np.ndarray,
    fixed_zero_columns: Sequence[int],
    head: str,
) -> dict[str, Any]:
    if head not in HURDLE_HEADS:
        raise E234EngineError("UNKNOWN_HURDLE_HEAD")
    occurrence_target = (target > 0.0).astype(np.float64)
    occurrence = fit_logistic(
        design, occurrence_target, weights, fixed_zero_columns
    )
    positive = target > 0.0
    if not np.any(positive):
        magnitude = {
            "state": "CONSTANT_ZERO_NO_POSITIVE_TARGET",
            "coefficients": [0.0] * DESIGN_WIDTH,
            "optimization_executed": False,
        }
    else:
        conditioned = target[positive]
        if head in SUPPORT_CONDITIONED_HEADS:
            conditioned = conditioned / np.maximum(0.01, supports[positive])
        magnitude_target = np.log1p(conditioned)
        if not np.all(np.isfinite(magnitude_target)):
            raise E234EngineError("NONFINITE_MAGNITUDE_TARGET")
        magnitude = fit_ridge(
            design[positive], magnitude_target, weights[positive], fixed_zero_columns
        )
    return {
        "kind": "HURDLE",
        "occurrence": occurrence,
        "magnitude": magnitude,
        "support_conditioned": head in SUPPORT_CONDITIONED_HEADS,
    }


def fit_five_head_model(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit all five heads; any invalid state raises a fold-fatal error."""
    if not rows:
        raise E234EngineError("EMPTY_MODEL_POPULATION")
    ordered = sorted(rows, key=lambda row: _identity(row)[0])
    _require_unique_observations(ordered)
    if any(row.get("role") != "MODEL" for row in ordered):
        raise E234EngineError("MODEL_ROLE_IDENTITY_DRIFT")
    if any(row.get("availability") != "AVAILABLE_EXACT" for row in ordered):
        raise E234EngineError("MODEL_ROW_NOT_AVAILABLE_EXACT")
    transform = fit_feature_transform(ordered)
    encoded = [apply_feature_transform(row, transform) for row in ordered]
    if any(not record["feature_valid"] for record in encoded):
        raise E234EngineError("MODEL_FEATURE_INVALID")
    design = _finite_array(
        [record["design"] for record in encoded], "MODEL_DESIGN", ndim=2
    )
    weights, weight_artifact = component_weights(encoded)
    supports = _finite_array(
        [record["raw"][17] for record in encoded], "MODEL_SUPPORT", ndim=1
    )
    targets = {
        "B": _finite_array([_target(row, "benefit") for row in ordered], "B_TARGET", ndim=1),
        "H": _finite_array([_target(row, "harm") for row in ordered], "H_TARGET", ndim=1),
        "harmed_fraction": _finite_array(
            [_target(row, "harmed_fraction") for row in ordered],
            "FRACTION_TARGET",
            ndim=1,
        ),
        "CVaR95": _finite_array(
            [_target(row, "cvar95") for row in ordered], "CVAR_TARGET", ndim=1
        ),
        "net_gain": _finite_array(
            [_target(row, "net_gain") for row in ordered], "NET_TARGET", ndim=1
        ),
    }
    heads = {
        head: fit_hurdle(
            design,
            targets[head],
            supports,
            weights,
            transform["fixed_zero_columns"],
            head,
        )
        for head in HURDLE_HEADS
    }
    severe_target = (targets["net_gain"] < SEVERE_CUTOFF).astype(np.float64)
    heads["severe"] = {
        "kind": "LOGISTIC",
        **fit_logistic(
            design,
            severe_target,
            weights,
            transform["fixed_zero_columns"],
        ),
    }
    artifact = {
        "schema": "e234-b1-five-head-model/v2",
        "status": "FITTED",
        "design_width": DESIGN_WIDTH,
        "head_order": list(HEAD_ORDER),
        "transform": transform,
        "weights": weight_artifact,
        "population_observation_ids": [record["observation_id"] for record in encoded],
        "population_component_ids": sorted({record["component_id"] for record in encoded}),
        "heads": heads,
        "solver": {
            "lambda": RIDGE_LAMBDA,
            "max_iterations": LOGISTIC_MAX_ITERATIONS,
            "gradient_infinity_tolerance": LOGISTIC_GRADIENT_INFINITY_TOLERANCE,
            "relative_objective_tolerance": LOGISTIC_RELATIVE_OBJECTIVE_TOLERANCE,
            "armijo_c1": LOGISTIC_ARMIJO_C1,
            "backtracking": LOGISTIC_BACKTRACKING,
            "minimum_step": LOGISTIC_MINIMUM_STEP,
            "sigmoid_predictor_clip": [-40.0, 40.0],
            "intercept_unpenalized": True,
        },
    }
    artifact["model_digest"] = canonical_sha256(artifact)
    validate_model(artifact)
    return artifact


def fit_outer_fold_or_native(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the immutable whole-fold fallback for every engine failure."""
    try:
        model = fit_five_head_model(rows)
    except (E234EngineError, FloatingPointError, OverflowError, ValueError) as exc:
        return {
            "schema": "e234-b1-five-head-fold-state/v2",
            "status": "FOLD_NATIVE_MODEL_FAILURE",
            "reason": f"{type(exc).__name__}:{exc}",
            "model": None,
            "immutable_native": True,
            "retry_authorized": False,
            "alternate_solver_authorized": False,
        }
    return {
        "schema": "e234-b1-five-head-fold-state/v2",
        "status": "PASS_FIVE_HEAD_MODEL",
        "reason": None,
        "model": model,
        "immutable_native": False,
        "retry_authorized": False,
        "alternate_solver_authorized": False,
    }


def _validate_coefficients(
    record: Mapping[str, Any], fixed_zero_columns: Sequence[int], name: str
) -> None:
    coefficients = record.get("coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != DESIGN_WIDTH:
        raise E234EngineError(f"COEFFICIENT_WIDTH_DRIFT:{name}")
    values = [_number(value, f"{name}_coefficient") for value in coefficients]
    if any(values[index] != 0.0 for index in fixed_zero_columns):
        raise E234EngineError(f"FIXED_ZERO_COEFFICIENT_DRIFT:{name}")


def validate_model(model: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "status",
        "design_width",
        "head_order",
        "transform",
        "weights",
        "population_observation_ids",
        "population_component_ids",
        "heads",
        "solver",
        "model_digest",
    }
    if set(model) != expected_keys:
        raise E234EngineError("MODEL_KEY_SET_DRIFT")
    if model.get("schema") != "e234-b1-five-head-model/v2":
        raise E234EngineError("MODEL_SCHEMA_DRIFT")
    if model.get("status") != "FITTED":
        raise E234EngineError("MODEL_STATUS_DRIFT")
    if model.get("design_width") != DESIGN_WIDTH or model.get("head_order") != list(HEAD_ORDER):
        raise E234EngineError("MODEL_LAYOUT_DRIFT")
    transform = model.get("transform")
    heads = model.get("heads")
    if not isinstance(transform, Mapping) or not isinstance(heads, Mapping):
        raise E234EngineError("MODEL_STRUCTURE_DRIFT")
    _validate_transform(transform)
    _validate_solver(model.get("solver"))
    _validate_weight_artifact(
        model.get("weights"),
        model.get("population_observation_ids"),
        model.get("population_component_ids"),
    )
    if model.get("population_observation_ids") != transform.get("population_observation_ids"):
        raise E234EngineError("MODEL_TRANSFORM_POPULATION_DRIFT")
    if model.get("population_component_ids") != transform.get("population_component_ids"):
        raise E234EngineError("MODEL_TRANSFORM_COMPONENT_DRIFT")
    fixed = transform.get("fixed_zero_columns")
    if not isinstance(fixed, list):
        raise E234EngineError("FIXED_ZERO_COLUMNS_REQUIRED")
    _active_columns(fixed)
    if set(heads) != set(HEAD_ORDER):
        raise E234EngineError("MODEL_HEAD_SET_DRIFT")
    for name in HURDLE_HEADS:
        head = heads[name]
        if not isinstance(head, Mapping) or head.get("kind") != "HURDLE":
            raise E234EngineError(f"HURDLE_HEAD_DRIFT:{name}")
        if head.get("support_conditioned") is not (name in SUPPORT_CONDITIONED_HEADS):
            raise E234EngineError(f"SUPPORT_CONDITIONING_DRIFT:{name}")
        occurrence = head.get("occurrence")
        magnitude = head.get("magnitude")
        if not isinstance(occurrence, Mapping) or not isinstance(magnitude, Mapping):
            raise E234EngineError(f"HURDLE_STATE_MISSING:{name}")
        _validate_logistic_state(occurrence, fixed, f"{name}.occurrence")
        _validate_magnitude_state(magnitude, fixed, f"{name}.magnitude")
    severe = heads["severe"]
    if not isinstance(severe, Mapping) or severe.get("kind") != "LOGISTIC":
        raise E234EngineError("SEVERE_HEAD_DRIFT")
    _validate_logistic_state(severe, fixed, "severe")
    without_digest = {key: value for key, value in model.items() if key != "model_digest"}
    if model.get("model_digest") != canonical_sha256(without_digest):
        raise E234EngineError("MODEL_DIGEST_DRIFT")


def _validate_transform(transform: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "raw_width",
        "encoded_width",
        "population_observation_ids",
        "population_component_ids",
        "centers",
        "scales",
        "statistics",
        "fixed_zero_columns",
        "intercept_fixed_zero",
        "row_weighting",
        "membership_digest",
    }
    if set(transform) != expected_keys:
        raise E234EngineError("TRANSFORM_KEY_SET_DRIFT")
    if (
        transform.get("schema") != "e234-b1-19-to-39-transform/v2"
        or transform.get("raw_width") != RAW_WIDTH
        or transform.get("encoded_width") != DESIGN_WIDTH
        or transform.get("intercept_fixed_zero") is not False
        or transform.get("row_weighting")
        != "ONE_PER_ROW_NO_COMPONENT_OR_HEAD_WEIGHTING"
    ):
        raise E234EngineError("TRANSFORM_CONTRACT_DRIFT")
    observation_ids = transform.get("population_observation_ids")
    component_ids = transform.get("population_component_ids")
    if (
        not isinstance(observation_ids, list)
        or not observation_ids
        or observation_ids != sorted(observation_ids)
        or len(observation_ids) != len(set(observation_ids))
        or not all(isinstance(value, str) and value for value in observation_ids)
    ):
        raise E234EngineError("TRANSFORM_OBSERVATION_POPULATION_DRIFT")
    if (
        not isinstance(component_ids, list)
        or not component_ids
        or component_ids != sorted(set(component_ids))
        or not all(isinstance(value, str) and value for value in component_ids)
    ):
        raise E234EngineError("TRANSFORM_COMPONENT_POPULATION_DRIFT")
    if transform.get("membership_digest") != canonical_sha256(
        {"observation_ids": observation_ids, "component_ids": component_ids}
    ):
        raise E234EngineError("TRANSFORM_MEMBERSHIP_DIGEST_DRIFT")
    centers = transform.get("centers")
    scales = transform.get("scales")
    statistics = transform.get("statistics")
    if (
        not isinstance(centers, list)
        or not isinstance(scales, list)
        or not isinstance(statistics, list)
        or len(centers) != RAW_WIDTH
        or len(scales) != RAW_WIDTH
        or len(statistics) != RAW_WIDTH
    ):
        raise E234EngineError("TRANSFORM_STATISTIC_SHAPE_DRIFT")
    expected_fixed = list(range(20, 39))
    statistic_keys = {
        "raw_index",
        "center",
        "center_observation_ids",
        "center_values",
        "mad",
        "mad_observation_ids",
        "mad_values",
        "scale",
        "minimum",
        "maximum",
        "constant_raw",
    }
    for index, statistic in enumerate(statistics):
        if not isinstance(statistic, Mapping) or set(statistic) != statistic_keys:
            raise E234EngineError("TRANSFORM_STATISTIC_KEY_DRIFT")
        if statistic.get("raw_index") != index:
            raise E234EngineError("TRANSFORM_STATISTIC_INDEX_DRIFT")
        center = _number(centers[index], f"transform_center_{index}")
        scale = _number(scales[index], f"transform_scale_{index}")
        stat_center = _number(statistic.get("center"), f"stat_center_{index}")
        mad = _number(statistic.get("mad"), f"stat_mad_{index}")
        stat_scale = _number(statistic.get("scale"), f"stat_scale_{index}")
        minimum = _number(statistic.get("minimum"), f"stat_minimum_{index}")
        maximum = _number(statistic.get("maximum"), f"stat_maximum_{index}")
        if minimum > maximum or mad < 0.0 or scale <= 0.0:
            raise E234EngineError("TRANSFORM_STATISTIC_VALUE_DRIFT")
        expected_scale = 1.0 if mad <= 1e-6 else mad
        constant = _normalize_zero(minimum) == _normalize_zero(maximum)
        if (
            center != stat_center
            or scale != stat_scale
            or scale != expected_scale
            or statistic.get("constant_raw") is not constant
        ):
            raise E234EngineError("TRANSFORM_STATISTIC_RELATION_DRIFT")
        central_ids = statistic.get("center_observation_ids")
        central_values = statistic.get("center_values")
        mad_ids = statistic.get("mad_observation_ids")
        mad_values = statistic.get("mad_values")
        expected_central_count = 1 if len(observation_ids) % 2 else 2
        if (
            not isinstance(central_ids, list)
            or not isinstance(central_values, list)
            or not isinstance(mad_ids, list)
            or not isinstance(mad_values, list)
            or len(central_ids) != expected_central_count
            or len(central_values) != expected_central_count
            or len(mad_ids) != expected_central_count
            or len(mad_values) != expected_central_count
            or any(value not in observation_ids for value in central_ids + mad_ids)
        ):
            raise E234EngineError("TRANSFORM_CENTRAL_RECORD_DRIFT")
        central_numbers = [_number(value, "center_value") for value in central_values]
        mad_numbers = [_number(value, "mad_value") for value in mad_values]
        expected_center = (
            central_numbers[0]
            if len(central_numbers) == 1
            else _normalize_zero((0.5 * central_numbers[0]) + (0.5 * central_numbers[1]))
        )
        expected_mad = (
            mad_numbers[0]
            if len(mad_numbers) == 1
            else _normalize_zero((0.5 * mad_numbers[0]) + (0.5 * mad_numbers[1]))
        )
        if center != expected_center or mad != expected_mad:
            raise E234EngineError("TRANSFORM_CENTRAL_VALUE_DRIFT")
        if constant:
            expected_fixed.append(index + 1)
    fixed = transform.get("fixed_zero_columns")
    if fixed != sorted(expected_fixed):
        raise E234EngineError("TRANSFORM_FIXED_ZERO_SET_DRIFT")
    _active_columns(fixed)


def _validate_solver(solver: Any) -> None:
    expected = {
        "lambda": RIDGE_LAMBDA,
        "max_iterations": LOGISTIC_MAX_ITERATIONS,
        "gradient_infinity_tolerance": LOGISTIC_GRADIENT_INFINITY_TOLERANCE,
        "relative_objective_tolerance": LOGISTIC_RELATIVE_OBJECTIVE_TOLERANCE,
        "armijo_c1": LOGISTIC_ARMIJO_C1,
        "backtracking": LOGISTIC_BACKTRACKING,
        "minimum_step": LOGISTIC_MINIMUM_STEP,
        "sigmoid_predictor_clip": [-40.0, 40.0],
        "intercept_unpenalized": True,
    }
    if solver != expected:
        raise E234EngineError("MODEL_SOLVER_CONTRACT_DRIFT")


def _validate_weight_artifact(
    artifact: Any, observation_ids: Any, component_ids: Any
) -> None:
    expected_keys = {
        "component_counts",
        "preliminary_mean",
        "weight_records",
        "weights",
        "weight_sum",
        "weight_mean",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected_keys:
        raise E234EngineError("WEIGHT_ARTIFACT_KEY_DRIFT")
    records = artifact.get("weight_records")
    weights = artifact.get("weights")
    counts = artifact.get("component_counts")
    if (
        not isinstance(records, list)
        or not isinstance(weights, list)
        or len(records) != len(observation_ids)
        or len(weights) != len(observation_ids)
        or not isinstance(counts, Mapping)
    ):
        raise E234EngineError("WEIGHT_ARTIFACT_SHAPE_DRIFT")
    if [record.get("observation_id") for record in records] != observation_ids:
        raise E234EngineError("WEIGHT_OBSERVATION_ORDER_DRIFT")
    observed_counts: dict[str, int] = {}
    for record in records:
        if set(record) != {"observation_id", "component_id", "weight"}:
            raise E234EngineError("WEIGHT_RECORD_KEY_DRIFT")
        component_id = record.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            raise E234EngineError("WEIGHT_COMPONENT_ID_DRIFT")
        observed_counts[component_id] = observed_counts.get(component_id, 0) + 1
    if dict(sorted(observed_counts.items())) != dict(counts):
        raise E234EngineError("WEIGHT_COMPONENT_COUNTS_DRIFT")
    if sorted(observed_counts) != component_ids:
        raise E234EngineError("WEIGHT_COMPONENT_POPULATION_DRIFT")
    preliminary = np.asarray(
        [1.0 / observed_counts[record["component_id"]] for record in records],
        dtype=np.float64,
    )
    preliminary_mean = float(np.mean(preliminary))
    expected_weights = preliminary / preliminary_mean
    actual_weights = _finite_array(weights, "MODEL_WEIGHT_ARTIFACT", ndim=1)
    record_weights = _finite_array(
        [record["weight"] for record in records], "MODEL_WEIGHT_RECORDS", ndim=1
    )
    if (
        _number(artifact.get("preliminary_mean"), "preliminary_mean")
        != preliminary_mean
        or not np.array_equal(actual_weights, expected_weights)
        or not np.array_equal(record_weights, expected_weights)
        or _number(artifact.get("weight_sum"), "weight_sum")
        != float(np.sum(expected_weights))
        or _number(artifact.get("weight_mean"), "weight_mean")
        != float(np.mean(expected_weights))
    ):
        raise E234EngineError("WEIGHT_FORMULA_DRIFT")


def _validate_logistic_state(
    record: Mapping[str, Any], fixed: Sequence[int], name: str
) -> None:
    state = record.get("state")
    _validate_coefficients(record, fixed, name)
    if state == "CONSTANT_DEGENERATE_CLASS":
        if record.get("optimization_executed") is not False or record.get("iterations") != 0:
            raise E234EngineError(f"DEGENERATE_OPTIMIZATION_DRIFT:{name}")
        if any(value != 0.0 for value in record["coefficients"]):
            raise E234EngineError(f"DEGENERATE_COEFFICIENT_DRIFT:{name}")
        probability = _number(record.get("constant_probability"), f"{name}_probability")
        if not 0.0 < probability < 1.0:
            raise E234EngineError(f"DEGENERATE_PROBABILITY_DRIFT:{name}")
    elif state == "FITTED":
        if record.get("optimization_executed") is not True or record.get("constant_probability") is not None:
            raise E234EngineError(f"FITTED_LOGISTIC_STATE_DRIFT:{name}")
    else:
        raise E234EngineError(f"UNKNOWN_LOGISTIC_STATE:{name}")


def _validate_magnitude_state(
    record: Mapping[str, Any], fixed: Sequence[int], name: str
) -> None:
    _validate_coefficients(record, fixed, name)
    state = record.get("state")
    if state == "CONSTANT_ZERO_NO_POSITIVE_TARGET":
        if record.get("optimization_executed") is not False:
            raise E234EngineError(f"ZERO_MAGNITUDE_OPTIMIZATION_DRIFT:{name}")
        if any(value != 0.0 for value in record["coefficients"]):
            raise E234EngineError(f"ZERO_MAGNITUDE_COEFFICIENT_DRIFT:{name}")
    elif state == "FITTED":
        if record.get("optimization_executed") is not True:
            raise E234EngineError(f"FITTED_MAGNITUDE_STATE_DRIFT:{name}")
    else:
        raise E234EngineError(f"UNKNOWN_MAGNITUDE_STATE:{name}")


def _predict_logistic(record: Mapping[str, Any], design: np.ndarray) -> np.ndarray:
    state = record.get("state")
    if state == "CONSTANT_DEGENERATE_CLASS":
        probability = _number(record.get("constant_probability"), "constant_probability")
        return np.full(design.shape[0], probability, dtype=np.float64)
    if state == "FITTED":
        coefficients = _finite_array(record.get("coefficients"), "LOGISTIC_COEFFICIENTS", ndim=1)
        if coefficients.shape != (DESIGN_WIDTH,):
            raise E234EngineError("LOGISTIC_COEFFICIENT_WIDTH")
        return _sigmoid(design @ coefficients)
    raise E234EngineError("UNKNOWN_LOGISTIC_STATE")


def predict_five_heads(
    model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validate_model(model)
    transformed = [apply_feature_transform(row, model["transform"]) for row in rows]
    if any(not record["feature_valid"] for record in transformed):
        raise E234EngineError("SCORE_FEATURE_MISSING_ABSTAIN_NATIVE")
    design = _finite_array(
        [record["design"] for record in transformed], "SCORE_DESIGN", ndim=2
    )
    supports = _finite_array(
        [record["raw"][17] for record in transformed], "SCORE_SUPPORT", ndim=1
    )
    predicted: dict[str, np.ndarray] = {}
    for name in HURDLE_HEADS:
        head = model["heads"][name]
        occurrence = _predict_logistic(head["occurrence"], design)
        magnitude = head["magnitude"]
        if magnitude.get("state") == "CONSTANT_ZERO_NO_POSITIVE_TARGET":
            conditional = np.zeros(design.shape[0], dtype=np.float64)
        elif magnitude.get("state") == "FITTED":
            coefficients = _finite_array(
                magnitude.get("coefficients"), "MAGNITUDE_COEFFICIENTS", ndim=1
            )
            if coefficients.shape != (DESIGN_WIDTH,):
                raise E234EngineError("MAGNITUDE_COEFFICIENT_WIDTH")
            conditional = np.expm1(np.clip(design @ coefficients, 0.0, 20.0))
        else:
            raise E234EngineError("UNKNOWN_MAGNITUDE_STATE")
        value = occurrence * conditional
        if name in SUPPORT_CONDITIONED_HEADS:
            value = value * supports
        if name == "harmed_fraction":
            value = np.clip(value, 0.0, 1.0)
        predicted[name] = value
    predicted["severe"] = _predict_logistic(model["heads"]["severe"], design)

    results = []
    for index, record in enumerate(transformed):
        values = {
            name: _normalize_zero(float(predicted[name][index])) for name in HEAD_ORDER
        }
        if (
            any(not math.isfinite(value) or value < 0.0 for value in values.values())
            or values["harmed_fraction"] > 1.0
            or values["severe"] > 1.0
        ):
            raise E234EngineError("INVALID_PREDICTION")
        results.append(
            {
                "observation_id": record["observation_id"],
                "component_id": record["component_id"],
                "model_digest": model["model_digest"],
                "predictions": values,
                "expected_net": _normalize_zero(values["B"] - values["H"]),
            }
        )
    return results


def strength_bits(value: Any) -> str:
    return struct.pack(">d", _normalize_zero(_number(value, "input_strength"))).hex()


def identity_key(row: Mapping[str, Any]) -> dict[str, str]:
    build_raw_features(row)
    return {
        "family": str(_nested(row, "typed_action_signature", "family")),
        "endpoint": str(_nested(row, "action", "endpoint")),
        "strength_bits": strength_bits(_nested(row, "action", "input_strength")),
    }


def _level_key(identity: Mapping[str, str], level: tuple[str, ...]) -> dict[str, Any]:
    if level == ("GLOBAL",):
        return {"level": ["GLOBAL"], "values": ["GLOBAL"]}
    return {"level": list(level), "values": [identity[name] for name in level]}


def _matches(identity: Mapping[str, str], key: Mapping[str, Any]) -> bool:
    if key.get("level") == ["GLOBAL"]:
        return key.get("values") == ["GLOBAL"]
    levels = key.get("level")
    values = key.get("values")
    if not isinstance(levels, list) or not isinstance(values, list) or len(levels) != len(values):
        raise E234EngineError("STRATUM_KEY_DRIFT")
    return all(
        identity.get(name) == value
        for name, value in zip(levels, values, strict=True)
    )


def _eligible(row: Mapping[str, Any], *, require_target: bool) -> bool:
    try:
        built = build_raw_features(row)
    except E234EngineError:
        return False
    if not built["feature_valid"] or row.get("availability") != "AVAILABLE_EXACT":
        return False
    if require_target and row.get("target_availability") != "AVAILABLE_EXACT":
        return False
    return True


def _validate_role_populations(
    model_rows: Sequence[Mapping[str, Any]],
    scale_rows: Sequence[Mapping[str, Any]],
) -> None:
    _require_unique_observations(model_rows)
    _require_unique_observations(scale_rows)
    if any(row.get("role") != "MODEL" for row in model_rows):
        raise E234EngineError("MODEL_ROLE_IDENTITY_DRIFT")
    if any(row.get("role") != "SCALE" for row in scale_rows):
        raise E234EngineError("SCALE_ROLE_IDENTITY_DRIFT")
    model_observations = {_identity(row)[0] for row in model_rows}
    scale_observations = {_identity(row)[0] for row in scale_rows}
    if model_observations.intersection(scale_observations):
        raise E234EngineError("MODEL_SCALE_OBSERVATION_OVERLAP")
    model_components = {_identity(row)[1] for row in model_rows}
    scale_components = {_identity(row)[1] for row in scale_rows}
    if model_components.intersection(scale_components):
        raise E234EngineError("MODEL_SCALE_COMPONENT_OVERLAP")


def _membership(
    rows: Sequence[Mapping[str, Any]],
    key: Mapping[str, Any],
    *,
    require_target: bool,
) -> list[Mapping[str, Any]]:
    _require_unique_observations(rows)
    result = [
        row
        for row in rows
        if _eligible(row, require_target=require_target)
        and _matches(identity_key(row), key)
    ]
    return sorted(result, key=lambda row: _identity(row)[0])


def _prediction(row: Mapping[str, Any], name: str) -> float:
    predictions = row.get("predictions")
    if not isinstance(predictions, Mapping):
        raise E234EngineError("PREDICTIONS_REQUIRED")
    return _number(predictions.get(name), f"prediction_{name}")


def _calibration_target(row: Mapping[str, Any], name: str) -> float:
    mapping = {
        "H": "harm",
        "CVaR95": "cvar95",
        "harmed_fraction": "harmed_fraction",
        "net": "net_gain",
    }
    return _target(row, mapping[name])


def row_residuals(row: Mapping[str, Any]) -> dict[str, float]:
    b_hat = _prediction(row, "B")
    h_hat = _prediction(row, "H")
    return {
        "net": _normalize_zero(abs(_calibration_target(row, "net") - (b_hat - h_hat))),
        "H": _normalize_zero(max(_calibration_target(row, "H") - h_hat, 0.0)),
        "harmed_fraction": _normalize_zero(
            max(
                _calibration_target(row, "harmed_fraction")
                - _prediction(row, "harmed_fraction"),
                0.0,
            )
        ),
        "CVaR95": _normalize_zero(
            max(
                _calibration_target(row, "CVaR95") - _prediction(row, "CVaR95"),
                0.0,
            )
        ),
    }


def component_max_scores(scale_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not scale_rows:
        raise E234EngineError("EMPTY_SCALE_MEMBERSHIP")
    _require_unique_observations(scale_rows)
    if any(row.get("role") != "SCALE" for row in scale_rows):
        raise E234EngineError("SCALE_ROLE_IDENTITY_DRIFT")
    if any(not _eligible(row, require_target=True) for row in scale_rows):
        raise E234EngineError("INELIGIBLE_SCALE_ROW")
    prediction_model_digests = {row.get("prediction_model_digest") for row in scale_rows}
    if len(prediction_model_digests) != 1:
        raise E234EngineError("SCALE_PREDICTION_MODEL_DIGEST_DRIFT")
    prediction_model_digest = next(iter(prediction_model_digests))
    if (
        not isinstance(prediction_model_digest, str)
        or len(prediction_model_digest) != 64
        or any(character not in "0123456789abcdef" for character in prediction_model_digest)
    ):
        raise E234EngineError("SCALE_PREDICTION_MODEL_DIGEST_REQUIRED")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in scale_rows:
        groups.setdefault(_identity(row)[1], []).append(row)
    fixed_scale_hash = canonical_sha256(
        {
            "head_order": list(CALIBRATION_HEAD_ORDER),
            "fixed_scales": FIXED_NONCONFORMITY_SCALES,
        }
    )
    scores: list[dict[str, Any]] = []
    for component_id in sorted(groups):
        component_rows = sorted(groups[component_id], key=lambda row: _identity(row)[0])
        candidates: list[tuple[float, str, int, str]] = []
        residual_hashes = []
        for row in component_rows:
            observation_id = _identity(row)[0]
            residuals = row_residuals(row)
            residual_hashes.append(
                {
                    "observation_id": observation_id,
                    "sha256_by_head": {
                        head: canonical_sha256(
                            {
                                "observation_id": observation_id,
                                "head": head,
                                "residual": residuals[head],
                            }
                        )
                        for head in CALIBRATION_HEAD_ORDER
                    },
                    "combined_sha256": canonical_sha256(
                        {"observation_id": observation_id, "residuals": residuals}
                    ),
                }
            )
            for head_index, head in enumerate(CALIBRATION_HEAD_ORDER):
                standardized = _normalize_zero(
                    residuals[head] / FIXED_NONCONFORMITY_SCALES[head]
                )
                if standardized < 0.0:
                    raise E234EngineError("NEGATIVE_STANDARDIZED_RESIDUAL")
                candidates.append((standardized, observation_id, head_index, head))
        maximum = max(candidate[0] for candidate in candidates)
        witness = min(
            (candidate for candidate in candidates if candidate[0] == maximum),
            key=lambda item: (item[1], item[2]),
        )
        observation_ids = [_identity(row)[0] for row in component_rows]
        membership_digest = canonical_sha256(
            {"component_id": component_id, "observation_ids": observation_ids}
        )
        scores.append(
            {
                "component_id": component_id,
                "observation_ids": observation_ids,
                "row_residual_hashes": residual_hashes,
                "R_c": maximum,
                "witness_observation_id": witness[1],
                "witness_head": witness[3],
                "witness_head_index": witness[2],
                "fixed_scale_table_sha256": fixed_scale_hash,
                "membership_digest": membership_digest,
            }
        )
    scores.sort(
        key=lambda row: (
            row["R_c"],
            row["component_id"],
            row["witness_observation_id"],
            row["witness_head_index"],
        )
    )
    return {
        "prediction_model_digest": prediction_model_digest,
        "fixed_scales": dict(FIXED_NONCONFORMITY_SCALES),
        "fixed_scale_table_sha256": fixed_scale_hash,
        "component_scores": scores,
        "component_population_digest": canonical_sha256(
            [
                {
                    "component_id": row["component_id"],
                    "membership_digest": row["membership_digest"],
                }
                for row in scores
            ]
        ),
    }


def conformal_rank(component_count: int) -> int:
    if isinstance(component_count, bool) or not isinstance(component_count, int) or component_count < 0:
        raise E234EngineError("INVALID_COMPONENT_COUNT")
    return (9 * (component_count + 1) + 9) // 10


def stratum_qualification(
    model_rows: Sequence[Mapping[str, Any]],
    scale_rows: Sequence[Mapping[str, Any]],
    key: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_role_populations(model_rows, scale_rows)
    model_members = _membership(model_rows, key, require_target=False)
    scale_members = _membership(scale_rows, key, require_target=True)
    model_components = sorted({_identity(row)[1] for row in model_members})
    scale_components = sorted({_identity(row)[1] for row in scale_members})
    component_count = len(scale_components)
    rank = conformal_rank(component_count)
    rank_available = 1 <= rank <= component_count
    artifact = {
        "key": dict(key),
        "model_observation_ids": [_identity(row)[0] for row in model_members],
        "scale_observation_ids": [_identity(row)[0] for row in scale_members],
        "model_component_ids": model_components,
        "scale_component_ids": scale_components,
        "model_component_count": len(model_components),
        "scale_component_count": component_count,
        "nominal_rank": rank,
        "rank_available": rank_available,
        "qualifies": (
            len(model_components) >= 8
            and component_count >= 9
            and rank_available
        ),
    }
    artifact["membership_digest"] = canonical_sha256(
        {
            name: artifact[name]
            for name in (
                "key",
                "model_observation_ids",
                "scale_observation_ids",
                "model_component_ids",
                "scale_component_ids",
            )
        }
    )
    return artifact


def _risk_scale(head: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if head not in RISK_HEADS:
        raise E234EngineError("UNKNOWN_RISK_HEAD")
    ordered_rows = sorted(rows, key=lambda row: _identity(row)[0])
    positives = [
        (_calibration_target(row, head), _identity(row)[0])
        for row in ordered_rows
        if _calibration_target(row, head) > 0.0
    ]
    if positives:
        positive_median, median_ids, median_values = _median(positives)
    else:
        positive_median, median_ids, median_values = 0.0, [], []
    residual_records = [
        (abs(row_residuals(row)[head]), _identity(row)[0]) for row in ordered_rows
    ]
    residual_records.sort(key=lambda item: (item[0], item[1]))
    if not residual_records:
        raise E234EngineError("EMPTY_RISK_SCALE_POPULATION")
    q75_rank = (3 * len(residual_records) + 3) // 4
    q75_value, q75_id = residual_records[q75_rank - 1]
    value = _normalize_zero(max(positive_median, q75_value, 0.01))
    observation_ids = [_identity(row)[0] for row in ordered_rows]
    return {
        "head": head,
        "population_observation_ids": observation_ids,
        "population_digest": canonical_sha256(observation_ids),
        "positive_target_observation_ids": [
            observation_id
            for _, observation_id in sorted(positives, key=lambda item: (item[0], item[1]))
        ],
        "positive_target_median": positive_median,
        "median_central_observation_ids": median_ids,
        "median_central_values": median_values,
        "q75_rank": q75_rank,
        "q75_selected_observation_id": q75_id,
        "q75_abs_residual": q75_value,
        "floor": 0.01,
        "risk_scale": value,
    }


def calibrate_component_max_for_query(
    model_rows: Sequence[Mapping[str, Any]],
    scale_rows: Sequence[Mapping[str, Any]],
    query_row: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_role_populations(model_rows, scale_rows)
    query_identity = identity_key(query_row)
    attempts: list[dict[str, Any]] = []
    for level_index, level in enumerate(FALLBACK_LEVELS):
        key = _level_key(query_identity, level)
        qualified = stratum_qualification(model_rows, scale_rows, key)
        attempts.append(qualified)
        if not qualified["qualifies"]:
            continue
        selected_rows = _membership(scale_rows, key, require_target=True)
        score_artifact = component_max_scores(selected_rows)
        rank = qualified["nominal_rank"]
        selected_score = score_artifact["component_scores"][rank - 1]
        q_value = selected_score["R_c"]
        radii = {
            head: _normalize_zero(q_value * FIXED_NONCONFORMITY_SCALES[head])
            for head in CALIBRATION_HEAD_ORDER
        }
        risk_scales = {head: _risk_scale(head, selected_rows) for head in RISK_HEADS}
        artifact = {
            "schema": "e234-b1-component-max-calibration/v2",
            "status": "QUALIFIED",
            "selected_key": key,
            "fallback_level_index": level_index,
            "qualification_attempts": attempts,
            "component_score_artifact": score_artifact,
            "model_digest": score_artifact["prediction_model_digest"],
            "Q": q_value,
            "selected_component_score": selected_score,
            "radii": radii,
            "risk_scales": risk_scales,
            "population_hashes": {
                "fixed_nonconformity_scale_table": score_artifact[
                    "fixed_scale_table_sha256"
                ],
                "one_score_per_component": score_artifact[
                    "component_population_digest"
                ],
                "row_level_risk_scale": {
                    head: risk_scales[head]["population_digest"] for head in RISK_HEADS
                },
            },
            "claim_scope": "FOUR_RESIDUAL_BOUNDS_JOINTLY_FOR_ALL_ELIGIBLE_ROWS_OF_ONE_EXCHANGEABLE_FUTURE_COMPONENT_ONLY",
            "severe_probability_calibrated": False,
        }
        artifact["calibration_digest"] = canonical_sha256(artifact)
        validate_calibration(artifact)
        return artifact
    return {
        "schema": "e234-b1-component-max-calibration/v2",
        "status": "FOLD_NATIVE_GLOBAL_INSUFFICIENT",
        "qualification_attempts": attempts,
        "Q_serialized": False,
        "native_fallback": True,
        "retry_authorized": False,
    }


def validate_calibration(calibration: Mapping[str, Any]) -> None:
    """Validate a complete typed calibration artifact, not its status alone."""
    expected_keys = {
        "schema",
        "status",
        "selected_key",
        "fallback_level_index",
        "qualification_attempts",
        "component_score_artifact",
        "model_digest",
        "Q",
        "selected_component_score",
        "radii",
        "risk_scales",
        "population_hashes",
        "claim_scope",
        "severe_probability_calibrated",
        "calibration_digest",
    }
    if set(calibration) != expected_keys:
        raise E234EngineError("CALIBRATION_KEY_SET_DRIFT")
    if (
        calibration.get("schema") != "e234-b1-component-max-calibration/v2"
        or calibration.get("status") != "QUALIFIED"
    ):
        raise E234EngineError("CALIBRATION_TYPED_STATE_DRIFT")
    level_index = calibration.get("fallback_level_index")
    attempts = calibration.get("qualification_attempts")
    if (
        isinstance(level_index, bool)
        or not isinstance(level_index, int)
        or not 0 <= level_index < len(FALLBACK_LEVELS)
        or not isinstance(attempts, list)
        or len(attempts) != level_index + 1
    ):
        raise E234EngineError("CALIBRATION_FALLBACK_TRACE_DRIFT")
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            raise E234EngineError("CALIBRATION_QUALIFICATION_TYPE_DRIFT")
        attempt_keys = {
            "key",
            "model_observation_ids",
            "scale_observation_ids",
            "model_component_ids",
            "scale_component_ids",
            "model_component_count",
            "scale_component_count",
            "nominal_rank",
            "rank_available",
            "qualifies",
            "membership_digest",
        }
        if set(attempt) != attempt_keys:
            raise E234EngineError("CALIBRATION_QUALIFICATION_KEY_DRIFT")
        expected_key_level = FALLBACK_LEVELS[index]
        key = attempt.get("key")
        if not isinstance(key, Mapping):
            raise E234EngineError("CALIBRATION_QUALIFICATION_KEY_TYPE_DRIFT")
        expected_level = ["GLOBAL"] if expected_key_level == ("GLOBAL",) else list(expected_key_level)
        if key.get("level") != expected_level:
            raise E234EngineError("CALIBRATION_FALLBACK_ORDER_DRIFT")
        model_ids = attempt.get("model_observation_ids")
        scale_ids = attempt.get("scale_observation_ids")
        model_components = attempt.get("model_component_ids")
        scale_components = attempt.get("scale_component_ids")
        if any(
            not isinstance(values, list)
            or values != sorted(values)
            or len(values) != len(set(values))
            or not all(isinstance(value, str) and value for value in values)
            for values in (model_ids, scale_ids, model_components, scale_components)
        ):
            raise E234EngineError("CALIBRATION_MEMBERSHIP_ARRAY_DRIFT")
        if set(model_ids).intersection(scale_ids) or set(model_components).intersection(scale_components):
            raise E234EngineError("CALIBRATION_ROLE_MEMBERSHIP_OVERLAP")
        model_count = attempt.get("model_component_count")
        scale_count = attempt.get("scale_component_count")
        rank = attempt.get("nominal_rank")
        if (
            model_count != len(model_components)
            or scale_count != len(scale_components)
            or rank != conformal_rank(scale_count)
            or attempt.get("rank_available") is not (1 <= rank <= scale_count)
        ):
            raise E234EngineError("CALIBRATION_QUALIFICATION_REDUCTION_DRIFT")
        qualifies = model_count >= 8 and scale_count >= 9 and 1 <= rank <= scale_count
        if attempt.get("qualifies") is not qualifies or qualifies is not (index == level_index):
            raise E234EngineError("CALIBRATION_QUALIFICATION_STATE_DRIFT")
        expected_digest = canonical_sha256(
            {
                name: attempt[name]
                for name in (
                    "key",
                    "model_observation_ids",
                    "scale_observation_ids",
                    "model_component_ids",
                    "scale_component_ids",
                )
            }
        )
        if attempt.get("membership_digest") != expected_digest:
            raise E234EngineError("CALIBRATION_MEMBERSHIP_DIGEST_DRIFT")
    selected_attempt = attempts[level_index]
    if calibration.get("selected_key") != selected_attempt["key"]:
        raise E234EngineError("CALIBRATION_SELECTED_KEY_DRIFT")

    score_artifact = calibration.get("component_score_artifact")
    if not isinstance(score_artifact, Mapping) or set(score_artifact) != {
        "prediction_model_digest",
        "fixed_scales",
        "fixed_scale_table_sha256",
        "component_scores",
        "component_population_digest",
    }:
        raise E234EngineError("COMPONENT_SCORE_ARTIFACT_DRIFT")
    model_digest = calibration.get("model_digest")
    if (
        not isinstance(model_digest, str)
        or len(model_digest) != 64
        or any(character not in "0123456789abcdef" for character in model_digest)
        or score_artifact.get("prediction_model_digest") != model_digest
    ):
        raise E234EngineError("CALIBRATION_MODEL_PROVENANCE_DRIFT")
    if score_artifact.get("fixed_scales") != FIXED_NONCONFORMITY_SCALES:
        raise E234EngineError("FIXED_NONCONFORMITY_SCALE_DRIFT")
    fixed_hash = canonical_sha256(
        {
            "head_order": list(CALIBRATION_HEAD_ORDER),
            "fixed_scales": FIXED_NONCONFORMITY_SCALES,
        }
    )
    if score_artifact.get("fixed_scale_table_sha256") != fixed_hash:
        raise E234EngineError("FIXED_NONCONFORMITY_SCALE_HASH_DRIFT")
    scores = score_artifact.get("component_scores")
    if not isinstance(scores, list) or len(scores) != selected_attempt["scale_component_count"]:
        raise E234EngineError("COMPONENT_SCORE_COUNT_DRIFT")
    if {score.get("component_id") for score in scores} != set(selected_attempt["scale_component_ids"]):
        raise E234EngineError("COMPONENT_SCORE_MEMBERSHIP_DRIFT")
    score_keys = {
        "component_id",
        "observation_ids",
        "row_residual_hashes",
        "R_c",
        "witness_observation_id",
        "witness_head",
        "witness_head_index",
        "fixed_scale_table_sha256",
        "membership_digest",
    }
    for score in scores:
        if not isinstance(score, Mapping) or set(score) != score_keys:
            raise E234EngineError("COMPONENT_SCORE_KEY_DRIFT")
        component_id = score.get("component_id")
        observation_ids = score.get("observation_ids")
        if (
            not isinstance(component_id, str)
            or not isinstance(observation_ids, list)
            or not observation_ids
            or observation_ids != sorted(observation_ids)
            or len(observation_ids) != len(set(observation_ids))
            or not set(observation_ids).issubset(selected_attempt["scale_observation_ids"])
        ):
            raise E234EngineError("COMPONENT_SCORE_OBSERVATION_DRIFT")
        value = _number(score.get("R_c"), "component_score")
        if value < 0.0:
            raise E234EngineError("NEGATIVE_COMPONENT_SCORE")
        witness_id = score.get("witness_observation_id")
        witness_head = score.get("witness_head")
        witness_index = score.get("witness_head_index")
        if (
            witness_id not in observation_ids
            or witness_head not in CALIBRATION_HEAD_ORDER
            or witness_index != CALIBRATION_HEAD_ORDER.index(witness_head)
            or score.get("fixed_scale_table_sha256") != fixed_hash
            or score.get("membership_digest")
            != canonical_sha256(
                {"component_id": component_id, "observation_ids": observation_ids}
            )
        ):
            raise E234EngineError("COMPONENT_SCORE_PROVENANCE_DRIFT")
        residual_hashes = score.get("row_residual_hashes")
        if (
            not isinstance(residual_hashes, list)
            or [row.get("observation_id") for row in residual_hashes] != observation_ids
            or any(
                set(row) != {"observation_id", "sha256_by_head", "combined_sha256"}
                or set(row.get("sha256_by_head", {})) != set(CALIBRATION_HEAD_ORDER)
                or any(
                    not isinstance(digest, str) or len(digest) != 64
                    for digest in [row.get("combined_sha256"), *row.get("sha256_by_head", {}).values()]
                )
                for row in residual_hashes
            )
        ):
            raise E234EngineError("COMPONENT_SCORE_RESIDUAL_HASH_DRIFT")
    expected_sorted = sorted(
        scores,
        key=lambda row: (
            row["R_c"],
            row["component_id"],
            row["witness_observation_id"],
            row["witness_head_index"],
        ),
    )
    if scores != expected_sorted:
        raise E234EngineError("COMPONENT_SCORE_SORT_DRIFT")
    population_digest = canonical_sha256(
        [
            {
                "component_id": row["component_id"],
                "membership_digest": row["membership_digest"],
            }
            for row in scores
        ]
    )
    if score_artifact.get("component_population_digest") != population_digest:
        raise E234EngineError("COMPONENT_POPULATION_DIGEST_DRIFT")
    selected = scores[selected_attempt["nominal_rank"] - 1]
    q_value = _number(calibration.get("Q"), "Q")
    if calibration.get("selected_component_score") != selected or q_value != selected["R_c"]:
        raise E234EngineError("CALIBRATION_Q_SELECTION_DRIFT")
    expected_radii = {
        head: _normalize_zero(q_value * FIXED_NONCONFORMITY_SCALES[head])
        for head in CALIBRATION_HEAD_ORDER
    }
    if calibration.get("radii") != expected_radii:
        raise E234EngineError("CALIBRATION_RADII_DRIFT")

    risk_scales = calibration.get("risk_scales")
    if not isinstance(risk_scales, Mapping) or set(risk_scales) != set(RISK_HEADS):
        raise E234EngineError("RISK_SCALE_HEAD_SET_DRIFT")
    for head in RISK_HEADS:
        record = risk_scales[head]
        risk_keys = {
            "head",
            "population_observation_ids",
            "population_digest",
            "positive_target_observation_ids",
            "positive_target_median",
            "median_central_observation_ids",
            "median_central_values",
            "q75_rank",
            "q75_selected_observation_id",
            "q75_abs_residual",
            "floor",
            "risk_scale",
        }
        if not isinstance(record, Mapping) or set(record) != risk_keys or record.get("head") != head:
            raise E234EngineError("RISK_SCALE_RECORD_DRIFT")
        population = record.get("population_observation_ids")
        if population != selected_attempt["scale_observation_ids"]:
            raise E234EngineError("RISK_SCALE_POPULATION_DRIFT")
        if record.get("population_digest") != canonical_sha256(population):
            raise E234EngineError("RISK_SCALE_POPULATION_HASH_DRIFT")
        if record.get("q75_rank") != (3 * len(population) + 3) // 4:
            raise E234EngineError("RISK_SCALE_Q75_RANK_DRIFT")
        if record.get("q75_selected_observation_id") not in population:
            raise E234EngineError("RISK_SCALE_Q75_ID_DRIFT")
        if (
            _number(record.get("floor"), "risk_scale_floor") != 0.01
            or _number(record.get("risk_scale"), "risk_scale") < 0.01
            or _number(record.get("positive_target_median"), "positive_target_median") < 0.0
            or _number(record.get("q75_abs_residual"), "q75_abs_residual") < 0.0
        ):
            raise E234EngineError("RISK_SCALE_VALUE_DRIFT")
    population_hashes = calibration.get("population_hashes")
    expected_population_hashes = {
        "fixed_nonconformity_scale_table": fixed_hash,
        "one_score_per_component": population_digest,
        "row_level_risk_scale": {
            head: risk_scales[head]["population_digest"] for head in RISK_HEADS
        },
    }
    if population_hashes != expected_population_hashes:
        raise E234EngineError("CALIBRATION_POPULATION_HASH_DRIFT")
    if (
        calibration.get("claim_scope")
        != "FOUR_RESIDUAL_BOUNDS_JOINTLY_FOR_ALL_ELIGIBLE_ROWS_OF_ONE_EXCHANGEABLE_FUTURE_COMPONENT_ONLY"
        or calibration.get("severe_probability_calibrated") is not False
    ):
        raise E234EngineError("CALIBRATION_CLAIM_DRIFT")
    without_digest = {
        key: value for key, value in calibration.items() if key != "calibration_digest"
    }
    if calibration.get("calibration_digest") != canonical_sha256(without_digest):
        raise E234EngineError("CALIBRATION_DIGEST_DRIFT")


def score_with_calibration(
    row: Mapping[str, Any],
    prediction: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    validate_calibration(calibration)
    values = prediction.get("predictions")
    if not isinstance(values, Mapping):
        raise E234EngineError("PREDICTION_VECTOR_REQUIRED")
    if prediction.get("model_digest") != calibration.get("model_digest"):
        raise E234EngineError("PREDICTION_MODEL_PROVENANCE_DRIFT")
    b_hat = _number(values.get("B"), "B_hat")
    h_hat = _number(values.get("H"), "H_hat")
    radii = calibration.get("radii")
    scale_records = calibration.get("risk_scales")
    if not isinstance(radii, Mapping) or not isinstance(scale_records, Mapping):
        raise E234EngineError("CALIBRATION_SCHEMA_DRIFT")
    risk_scales = {
        head: _number(scale_records[head]["risk_scale"], f"risk_scale_{head}")
        for head in RISK_HEADS
    }
    if any(value <= 0.0 for value in risk_scales.values()):
        raise E234EngineError("NONPOSITIVE_RISK_SCALE")
    upper = {
        "H": _normalize_zero(h_hat + _number(radii.get("H"), "radius_H")),
        "harmed_fraction": _normalize_zero(
            _number(values.get("harmed_fraction"), "fraction_hat")
            + _number(radii.get("harmed_fraction"), "radius_fraction")
        ),
        "CVaR95": _normalize_zero(
            _number(values.get("CVaR95"), "CVaR95_hat")
            + _number(radii.get("CVaR95"), "radius_CVaR95")
        ),
    }
    terms = {
        "H": upper["H"] / risk_scales["H"],
        "harmed_fraction": upper["harmed_fraction"] / risk_scales["harmed_fraction"],
        "CVaR95": upper["CVaR95"] / risk_scales["CVaR95"],
        "severe": _number(values.get("severe"), "severe_probability"),
    }
    if any(not math.isfinite(value) for value in terms.values()):
        raise E234EngineError("NONFINITE_JOINT_RISK_TERM")
    expected_net = _normalize_zero(b_hat - h_hat)
    return {
        "observation_id": _identity(row)[0],
        "typed_admission": row.get("typed_admission") is True,
        "predictions": {head: _number(values[head], head) for head in HEAD_ORDER},
        "predicted_expected_net": expected_net,
        "expected_net_lower_diagnostic": _normalize_zero(
            expected_net - _number(radii.get("net"), "radius_net")
        ),
        "upper_bounds": upper,
        "joint_risk_terms": terms,
        "joint_risk": _normalize_zero(max(terms.values())),
    }


def finite_cutoff(value: Any) -> dict[str, Any]:
    return {"kind": "FINITE", "value": _number(value, "cutoff")}


def positive_infinity_cutoff() -> dict[str, str]:
    return {"kind": "POSITIVE_INFINITY"}


def validate_cutoff(cutoff: Mapping[str, Any]) -> None:
    if not isinstance(cutoff, Mapping):
        raise E234EngineError("CUTOFF_OBJECT_REQUIRED")
    kind = cutoff.get("kind")
    if kind == "FINITE":
        if set(cutoff) != {"kind", "value"}:
            raise E234EngineError("FINITE_CUTOFF_KEY_DRIFT")
        _number(cutoff["value"], "cutoff")
    elif kind == "POSITIVE_INFINITY":
        if set(cutoff) != {"kind"}:
            raise E234EngineError("SENTINEL_KEY_DRIFT")
    else:
        raise E234EngineError("UNKNOWN_CUTOFF_TAG")
    canonical_json_bytes(dict(cutoff))


def accepts_joint_risk(joint_risk: Any, cutoff: Mapping[str, Any]) -> bool:
    risk = _number(joint_risk, "joint_risk")
    validate_cutoff(cutoff)
    return cutoff["kind"] == "POSITIVE_INFINITY" or risk < cutoff["value"]


def select_threshold(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: _identity(row)[0])
    if not ordered:
        raise E234EngineError("EMPTY_THRESHOLD_POPULATION")
    _require_unique_observations(ordered)
    finite_values = sorted(
        {_normalize_zero(_number(row.get("joint_risk"), "joint_risk")) for row in ordered}
    )
    candidates = [finite_cutoff(value) for value in finite_values]
    candidates.append(positive_infinity_cutoff())
    base_positive = [
        row
        for row in ordered
        if row.get("typed_admission") is True
        and _number(row.get("predicted_expected_net"), "predicted_expected_net") > 0.0
    ]
    denominator = math.fsum(
        max(_target(row, "net_gain"), 0.0) for row in base_positive
    )
    if denominator == 0.0 or not math.isfinite(denominator):
        raise E234EngineError("ZERO_POSITIVE_MASS_DENOMINATOR")
    records = []
    for candidate_rank, cutoff in enumerate(candidates):
        accepted = [
            row
            for row in base_positive
            if accepts_joint_risk(row["joint_risk"], cutoff)
        ]
        numerator = math.fsum(max(_target(row, "net_gain"), 0.0) for row in accepted)
        retention = numerator / denominator
        accepted_ids = {_identity(row)[0] for row in accepted}
        severe_rejected = sum(
            1
            for row in base_positive
            if _target(row, "net_gain") < SEVERE_CUTOFF
            and _identity(row)[0] not in accepted_ids
        )
        delivered = math.fsum(
            _target(row, "net_gain") if _identity(row)[0] in accepted_ids else 0.0
            for row in ordered
        )
        records.append(
            {
                "candidate": cutoff,
                "candidate_rank": candidate_rank,
                "severe_rejected": severe_rejected,
                "delivered_net_gain_fsum": _normalize_zero(delivered),
                "accepted_rows": len(accepted),
                "positive_mass_numerator": _normalize_zero(numerator),
                "positive_mass_denominator": _normalize_zero(denominator),
                "positive_mass_retention": _normalize_zero(retention),
                "feasible": retention + RETENTION_SOLE_TOLERANCE >= RETENTION_MINIMUM,
            }
        )
    feasible = [record for record in records if record["feasible"]]
    if not feasible:
        raise E234EngineError("NO_FEASIBLE_THRESHOLD")
    selected = max(
        feasible,
        key=lambda row: (
            row["severe_rejected"],
            row["delivered_net_gain_fsum"],
            row["accepted_rows"],
            -row["candidate_rank"],
        ),
    )
    return {
        "schema": "e234-b1-tagged-threshold-selection/v2",
        "status": "SELECTED",
        "ordered_observation_ids": [_identity(row)[0] for row in ordered],
        "candidates": records,
        "selected_candidate_rank": selected["candidate_rank"],
        "selected_cutoff": selected["candidate"],
        "selected_metrics": selected,
        "acceptance": "TYPED_ADMISSION_AND_PREDICTED_EXPECTED_NET_GT_ZERO_AND_STRICT_JOINT_RISK_LT_CUTOFF",
        "retention_sole_tolerance": RETENTION_SOLE_TOLERANCE,
    }


def formal_acceptance(
    typed_admission: bool,
    benefit_hat: Any,
    harm_hat: Any,
    joint_risk: Any,
    cutoff: Mapping[str, Any],
) -> bool:
    if not isinstance(typed_admission, bool):
        raise E234EngineError("TYPED_ADMISSION_BOOLEAN_REQUIRED")
    expected_net = _number(benefit_hat, "benefit_hat") - _number(harm_hat, "harm_hat")
    return typed_admission and expected_net > 0.0 and accepts_joint_risk(joint_risk, cutoff)


__all__ = [
    "ACTION_FAMILIES",
    "CALIBRATION_HEAD_ORDER",
    "DESIGN_WIDTH",
    "E234EngineError",
    "FALLBACK_LEVELS",
    "FIXED_NONCONFORMITY_SCALES",
    "HEAD_ORDER",
    "INCIDENT_CASE_ID",
    "INCIDENT_SHA256",
    "INCIDENT_VERIFICATION_SHA256",
    "RAW_WIDTH",
    "ZERO_READ_BOUNDARY",
    "accepts_joint_risk",
    "apply_feature_transform",
    "build_raw_features",
    "calibrate_component_max_for_query",
    "canonical_json_bytes",
    "canonical_sha256",
    "component_max_scores",
    "component_weights",
    "conformal_rank",
    "finite_cutoff",
    "fit_feature_transform",
    "fit_five_head_model",
    "fit_logistic",
    "fit_outer_fold_or_native",
    "fit_ridge",
    "formal_acceptance",
    "identity_key",
    "positive_infinity_cutoff",
    "predict_five_heads",
    "row_residuals",
    "score_with_calibration",
    "select_threshold",
    "strength_bits",
    "stratum_qualification",
    "validate_cutoff",
    "validate_calibration",
    "validate_model",
    "validate_recursive_synthetic_boundary",
    "make_synthetic_receipt",
    "validate_synthetic_receipt",
]
