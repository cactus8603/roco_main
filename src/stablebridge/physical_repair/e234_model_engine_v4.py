"""Append-only structural successor for the outcome-blind E234 core engine.

The frozen v3 implementation remains the numerical authority for the 19→39
transform, five-head fit, conformal reduction, and threshold policy.  This
module adds the locally decidable validation and prediction bindings required
by the independent v3 FAIL review.  It does not authenticate fitted
coefficients and therefore grants no activation, payload, or execution
authority.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from . import e234_model_engine as v3


# Immutable transitive binding.  The candidate package rehashes the file too;
# this literal makes the dependency visible to consumers of this module.
E234_CORE_V3_SOURCE_SHA256 = (
    "fdc5e63fe400445e8d573fd30c2376f79dbfa07f7a1afc42b59e3d7c2aa8fb6b"
)
E234_CORE_V3_FAIL_REVIEW_SHA256 = (
    "c46a69d247b7ab77d4bbaddbf423e3ce1aa9b88fd2f135da1cdbcc97f30ad74d"
)
E234_CORE_V4_VALIDATOR = "e234-core-structural-validator/v4"
PREDICTION_SCHEMA = "e234-b1-five-head-prediction/v4"

# Public constants and unchanged primitives deliberately preserve v3 names.
ACTION_FAMILIES = v3.ACTION_FAMILIES
CALIBRATION_HEAD_ORDER = v3.CALIBRATION_HEAD_ORDER
DESIGN_WIDTH = v3.DESIGN_WIDTH
E234EngineError = v3.E234EngineError
FALLBACK_LEVELS = v3.FALLBACK_LEVELS
FIXED_NONCONFORMITY_SCALES = v3.FIXED_NONCONFORMITY_SCALES
HEAD_ORDER = v3.HEAD_ORDER
HURDLE_HEADS = v3.HURDLE_HEADS
INCIDENT_CASE_ID = v3.INCIDENT_CASE_ID
INCIDENT_SHA256 = v3.INCIDENT_SHA256
INCIDENT_VERIFICATION_SHA256 = v3.INCIDENT_VERIFICATION_SHA256
LOGISTIC_MAX_ITERATIONS = v3.LOGISTIC_MAX_ITERATIONS
RAW_WIDTH = v3.RAW_WIDTH
RISK_HEADS = v3.RISK_HEADS
SUPPORT_CONDITIONED_HEADS = v3.SUPPORT_CONDITIONED_HEADS
ZERO_READ_BOUNDARY = v3.ZERO_READ_BOUNDARY

accepts_joint_risk = v3.accepts_joint_risk
apply_feature_transform = v3.apply_feature_transform
build_raw_features = v3.build_raw_features
canonical_json_bytes = v3.canonical_json_bytes
canonical_sha256 = v3.canonical_sha256
component_weights = v3.component_weights
conformal_rank = v3.conformal_rank
finite_cutoff = v3.finite_cutoff
fit_feature_transform = v3.fit_feature_transform
fit_logistic = v3.fit_logistic
fit_ridge = v3.fit_ridge
formal_acceptance = v3.formal_acceptance
identity_key = v3.identity_key
make_synthetic_receipt = v3.make_synthetic_receipt
positive_infinity_cutoff = v3.positive_infinity_cutoff
select_threshold = v3.select_threshold
strength_bits = v3.strength_bits
stratum_qualification = v3.stratum_qualification
validate_cutoff = v3.validate_cutoff
validate_recursive_synthetic_boundary = v3.validate_recursive_synthetic_boundary
validate_synthetic_receipt = v3.validate_synthetic_receipt


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    observation_id = row.get("observation_id")
    component_id = row.get("component_id")
    if not isinstance(observation_id, str) or not observation_id:
        raise E234EngineError("OBSERVATION_ID_REQUIRED")
    if not isinstance(component_id, str) or not component_id:
        raise E234EngineError("COMPONENT_ID_REQUIRED")
    return observation_id, component_id


def _finite_number(value: Any, name: str) -> float:
    return v3._number(value, name)


def _strict_logistic_state(
    record: Mapping[str, Any], *, severe: bool, name: str
) -> None:
    expected = {
        "state",
        "coefficients",
        "constant_probability",
        "iterations",
        "optimization_executed",
    }
    if severe:
        expected.add("kind")
    if not isinstance(record, Mapping) or set(record) != expected:
        raise E234EngineError(f"LOGISTIC_STATE_KEY_SET_DRIFT:{name}")
    iterations = record.get("iterations")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not 0 <= iterations <= LOGISTIC_MAX_ITERATIONS
    ):
        raise E234EngineError(f"LOGISTIC_ITERATION_RANGE_DRIFT:{name}")
    if record.get("state") == "CONSTANT_DEGENERATE_CLASS" and iterations != 0:
        raise E234EngineError(f"DEGENERATE_ITERATION_STATE_DRIFT:{name}")
    if record.get("state") == "FITTED" and record.get("optimization_executed") is not True:
        raise E234EngineError(f"FITTED_ITERATION_STATE_DRIFT:{name}")


def _strict_model_structure(model: Mapping[str, Any]) -> None:
    heads = model["heads"]
    for name in HURDLE_HEADS:
        head = heads[name]
        if set(head) != {"kind", "occurrence", "magnitude", "support_conditioned"}:
            raise E234EngineError(f"HURDLE_HEAD_KEY_SET_DRIFT:{name}")
        _strict_logistic_state(head["occurrence"], severe=False, name=f"{name}.occurrence")
        magnitude = head["magnitude"]
        if not isinstance(magnitude, Mapping) or set(magnitude) != {
            "state",
            "coefficients",
            "optimization_executed",
        }:
            raise E234EngineError(f"MAGNITUDE_STATE_KEY_SET_DRIFT:{name}")
        if magnitude.get("state") == "CONSTANT_ZERO_NO_POSITIVE_TARGET":
            if magnitude.get("optimization_executed") is not False:
                raise E234EngineError(f"MAGNITUDE_STATE_INVARIANT_DRIFT:{name}")
        elif magnitude.get("state") == "FITTED":
            if magnitude.get("optimization_executed") is not True:
                raise E234EngineError(f"MAGNITUDE_STATE_INVARIANT_DRIFT:{name}")
        else:
            raise E234EngineError(f"UNKNOWN_MAGNITUDE_STATE:{name}")
    severe = heads["severe"]
    if severe.get("kind") != "LOGISTIC":
        raise E234EngineError("SEVERE_HEAD_DRIFT")
    _strict_logistic_state(severe, severe=True, name="severe")


def validate_model(model: Mapping[str, Any]) -> None:
    """Run the v3 checks plus exact nested schemas and iteration semantics."""
    v3.validate_model(model)
    _strict_model_structure(model)


def fit_five_head_model(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model = v3.fit_five_head_model(rows)
    validate_model(model)
    return model


def fit_outer_fold_or_native(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Retain whole-fold fail-closed behavior while applying v4 validation."""
    try:
        model = fit_five_head_model(rows)
    except (E234EngineError, FloatingPointError, OverflowError, ValueError) as exc:
        return {
            "schema": "e234-b1-five-head-fold-state/v4",
            "status": "FOLD_NATIVE_MODEL_FAILURE",
            "reason": f"{type(exc).__name__}:{exc}",
            "model": None,
            "immutable_native": True,
            "retry_authorized": False,
            "alternate_solver_authorized": False,
        }
    return {
        "schema": "e234-b1-five-head-fold-state/v4",
        "status": "PASS_FIVE_HEAD_MODEL",
        "reason": None,
        "model": model,
        "immutable_native": False,
        "retry_authorized": False,
        "alternate_solver_authorized": False,
    }


def _prediction_payload(
    *,
    observation_id: str,
    component_id: str,
    model_digest: str,
    predictions: Mapping[str, Any],
    expected_net: Any,
) -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "observation_id": observation_id,
        "component_id": component_id,
        "model_digest": model_digest,
        "predictions": dict(predictions),
        "expected_net": expected_net,
    }


def validate_prediction_receipt(
    receipt: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    model: Mapping[str, Any] | None = None,
    expected_model_digest: str | None = None,
) -> None:
    """Deeply validate a typed prediction and optionally replay it from model."""
    expected_keys = {
        "schema",
        "observation_id",
        "component_id",
        "model_digest",
        "predictions",
        "expected_net",
        "prediction_digest",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        raise E234EngineError("PREDICTION_RECEIPT_KEY_SET_DRIFT")
    observation_id, component_id = _identity(row)
    if (
        receipt.get("schema") != PREDICTION_SCHEMA
        or receipt.get("observation_id") != observation_id
        or receipt.get("component_id") != component_id
    ):
        raise E234EngineError("PREDICTION_ROW_IDENTITY_DRIFT")
    model_digest = receipt.get("model_digest")
    if (
        not isinstance(model_digest, str)
        or len(model_digest) != 64
        or any(character not in "0123456789abcdef" for character in model_digest)
        or (expected_model_digest is not None and model_digest != expected_model_digest)
    ):
        raise E234EngineError("PREDICTION_MODEL_PROVENANCE_DRIFT")
    predictions = receipt.get("predictions")
    if not isinstance(predictions, Mapping) or list(predictions) != list(HEAD_ORDER):
        raise E234EngineError("PREDICTION_HEAD_SCHEMA_DRIFT")
    values = {name: _finite_number(predictions[name], f"prediction_{name}") for name in HEAD_ORDER}
    if (
        any(value < 0.0 for value in values.values())
        or values["harmed_fraction"] > 1.0
        or values["severe"] > 1.0
    ):
        raise E234EngineError("PREDICTION_VALUE_DRIFT")
    expected_net = v3._normalize_zero(values["B"] - values["H"])
    if _finite_number(receipt.get("expected_net"), "prediction_expected_net") != expected_net:
        raise E234EngineError("PREDICTION_EXPECTED_NET_DRIFT")
    payload = _prediction_payload(
        observation_id=observation_id,
        component_id=component_id,
        model_digest=model_digest,
        predictions=predictions,
        expected_net=expected_net,
    )
    if receipt.get("prediction_digest") != canonical_sha256(payload):
        raise E234EngineError("PREDICTION_DIGEST_DRIFT")
    if model is not None:
        validate_model(model)
        if model.get("model_digest") != model_digest:
            raise E234EngineError("PREDICTION_MODEL_PROVENANCE_DRIFT")
        recomputed = predict_five_heads(model, [row])[0]
        if dict(receipt) != recomputed:
            raise E234EngineError("PREDICTION_REPLAY_DRIFT")


def predict_five_heads(
    model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return exact-key, row-bound and self-digesting prediction receipts."""
    validate_model(model)
    result: list[dict[str, Any]] = []
    # Replay one row at a time.  BLAS reductions over a batch can differ by a
    # few ulps from a later one-row verification, which is incompatible with
    # exact receipt equality even though both computations are deterministic.
    for row in rows:
        prediction = v3.predict_five_heads(model, [row])[0]
        observation_id, component_id = _identity(row)
        payload = _prediction_payload(
            observation_id=observation_id,
            component_id=component_id,
            model_digest=model["model_digest"],
            predictions=prediction["predictions"],
            expected_net=prediction["expected_net"],
        )
        receipt = {**payload, "prediction_digest": canonical_sha256(payload)}
        validate_prediction_receipt(receipt, row, expected_model_digest=model["model_digest"])
        result.append(receipt)
    return result


def attach_prediction_receipt(
    row: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy a SCALE row and attach the exact typed receipt plus v3 fields."""
    validate_prediction_receipt(receipt, row)
    attached = deepcopy(dict(row))
    attached["prediction_receipt"] = deepcopy(dict(receipt))
    attached["prediction_model_digest"] = receipt["model_digest"]
    attached["predictions"] = deepcopy(dict(receipt["predictions"]))
    return attached


def _validate_scale_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        receipt = row.get("prediction_receipt")
        if not isinstance(receipt, Mapping):
            raise E234EngineError("TYPED_PREDICTION_RECEIPT_REQUIRED")
        validate_prediction_receipt(
            receipt,
            row,
            expected_model_digest=row.get("prediction_model_digest"),
        )
        if (
            row.get("prediction_model_digest") != receipt.get("model_digest")
            or row.get("predictions") != receipt.get("predictions")
        ):
            raise E234EngineError("SCALE_PREDICTION_RECEIPT_DRIFT")


def row_residuals(row: Mapping[str, Any]) -> dict[str, float]:
    _validate_scale_prediction_rows([row])
    return v3.row_residuals(row)


def component_max_scores(scale_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _validate_scale_prediction_rows(scale_rows)
    return v3.component_max_scores(scale_rows)


def _validate_fallback_key(key: Any, level: tuple[str, ...]) -> None:
    if not isinstance(key, Mapping) or set(key) != {"level", "values"}:
        raise E234EngineError("CALIBRATION_FALLBACK_KEY_SCHEMA_DRIFT")
    expected_level = ["GLOBAL"] if level == ("GLOBAL",) else list(level)
    if key.get("level") != expected_level:
        raise E234EngineError("CALIBRATION_FALLBACK_ORDER_DRIFT")
    values = key.get("values")
    if level == ("GLOBAL",):
        if values != ["GLOBAL"]:
            raise E234EngineError("CALIBRATION_GLOBAL_KEY_VALUE_DRIFT")
        return
    if (
        not isinstance(values, list)
        or len(values) != len(level)
        or not all(isinstance(value, str) and value for value in values)
    ):
        raise E234EngineError("CALIBRATION_FALLBACK_KEY_VALUE_DRIFT")
    identities = dict(zip(level, values, strict=True))
    if "family" in identities and identities["family"] not in ACTION_FAMILIES:
        raise E234EngineError("CALIBRATION_FALLBACK_FAMILY_DRIFT")
    if "endpoint" in identities and identities["endpoint"] not in {"first", "second", "both"}:
        raise E234EngineError("CALIBRATION_FALLBACK_ENDPOINT_DRIFT")
    if "strength_bits" in identities:
        strength = identities["strength_bits"]
        if len(strength) != 16 or any(character not in "0123456789abcdef" for character in strength):
            raise E234EngineError("CALIBRATION_FALLBACK_STRENGTH_BITS_DRIFT")


def _validate_component_partition(calibration: Mapping[str, Any]) -> None:
    selected = calibration["qualification_attempts"][calibration["fallback_level_index"]]
    scores = calibration["component_score_artifact"]["component_scores"]
    component_ids = [score["component_id"] for score in scores]
    if sorted(component_ids) != selected["scale_component_ids"] or len(component_ids) != len(set(component_ids)):
        raise E234EngineError("COMPONENT_SCORE_COMPONENT_PARTITION_DRIFT")
    memberships = [observation for score in scores for observation in score["observation_ids"]]
    if (
        len(memberships) != len(set(memberships))
        or sorted(memberships) != selected["scale_observation_ids"]
    ):
        raise E234EngineError("COMPONENT_SCORE_OBSERVATION_PARTITION_DRIFT")


def _validate_risk_scale_record(record: Mapping[str, Any], population: list[str]) -> None:
    positive_ids = record["positive_target_observation_ids"]
    central_ids = record["median_central_observation_ids"]
    central_values = record["median_central_values"]
    if (
        not isinstance(positive_ids, list)
        or len(positive_ids) != len(set(positive_ids))
        or not set(positive_ids).issubset(population)
        or not isinstance(central_ids, list)
        or not isinstance(central_values, list)
    ):
        raise E234EngineError("RISK_SCALE_MEDIAN_RECORD_DRIFT")
    median = _finite_number(record["positive_target_median"], "positive_target_median")
    if not positive_ids:
        if central_ids != [] or central_values != [] or median != 0.0:
            raise E234EngineError("RISK_SCALE_EMPTY_POSITIVE_MEDIAN_DRIFT")
    else:
        middle = len(positive_ids) // 2
        expected_ids = (
            [positive_ids[middle]]
            if len(positive_ids) % 2
            else [positive_ids[middle - 1], positive_ids[middle]]
        )
        values = [_finite_number(value, "median_central_value") for value in central_values]
        if (
            central_ids != expected_ids
            or len(values) != len(expected_ids)
            or any(value <= 0.0 for value in values)
            or values != sorted(values)
        ):
            raise E234EngineError("RISK_SCALE_MEDIAN_RECORD_DRIFT")
        expected_median = (
            values[0]
            if len(values) == 1
            else v3._normalize_zero(0.5 * values[0] + 0.5 * values[1])
        )
        if median != expected_median:
            raise E234EngineError("RISK_SCALE_MEDIAN_VALUE_DRIFT")
    q75 = _finite_number(record["q75_abs_residual"], "q75_abs_residual")
    floor = _finite_number(record["floor"], "risk_scale_floor")
    scale = _finite_number(record["risk_scale"], "risk_scale")
    if q75 < 0.0 or floor != 0.01:
        raise E234EngineError("RISK_SCALE_REDUCTION_RECORD_DRIFT")
    expected_scale = v3._normalize_zero(max(median, q75, floor))
    if scale != expected_scale:
        raise E234EngineError("RISK_SCALE_FORMULA_DRIFT")


def validate_calibration(calibration: Mapping[str, Any]) -> None:
    """Run v3 validation plus exact fallback, partition, and scale checks."""
    v3.validate_calibration(calibration)
    attempts = calibration["qualification_attempts"]
    for index, attempt in enumerate(attempts):
        _validate_fallback_key(attempt["key"], FALLBACK_LEVELS[index])
    selected = attempts[calibration["fallback_level_index"]]
    if calibration["selected_key"] != selected["key"]:
        raise E234EngineError("CALIBRATION_SELECTED_KEY_DRIFT")
    _validate_component_partition(calibration)
    population = selected["scale_observation_ids"]
    for head in RISK_HEADS:
        _validate_risk_scale_record(calibration["risk_scales"][head], population)


def calibrate_component_max_for_query(
    model_rows: Sequence[Mapping[str, Any]],
    scale_rows: Sequence[Mapping[str, Any]],
    query_row: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_scale_prediction_rows(scale_rows)
    calibration = v3.calibrate_component_max_for_query(model_rows, scale_rows, query_row)
    if calibration.get("status") == "QUALIFIED":
        validate_calibration(calibration)
    return calibration


def score_with_calibration(
    row: Mapping[str, Any],
    model: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Score only after replaying the exact prediction from full model+row."""
    validate_model(model)
    validate_calibration(calibration)
    if model.get("model_digest") != calibration.get("model_digest"):
        raise E234EngineError("PREDICTION_MODEL_PROVENANCE_DRIFT")
    receipt = predict_five_heads(model, [row])[0]
    validate_prediction_receipt(receipt, row, model=model)
    legacy_prediction = {
        "observation_id": receipt["observation_id"],
        "component_id": receipt["component_id"],
        "model_digest": receipt["model_digest"],
        "predictions": receipt["predictions"],
        "expected_net": receipt["expected_net"],
    }
    scored = v3.score_with_calibration(row, legacy_prediction, calibration)
    return {
        "schema": "e234-b1-calibrated-score/v4",
        "component_id": receipt["component_id"],
        "prediction_digest": receipt["prediction_digest"],
        **scored,
    }


__all__ = [
    "ACTION_FAMILIES",
    "CALIBRATION_HEAD_ORDER",
    "DESIGN_WIDTH",
    "E234EngineError",
    "E234_CORE_V3_FAIL_REVIEW_SHA256",
    "E234_CORE_V3_SOURCE_SHA256",
    "E234_CORE_V4_VALIDATOR",
    "FALLBACK_LEVELS",
    "FIXED_NONCONFORMITY_SCALES",
    "HEAD_ORDER",
    "HURDLE_HEADS",
    "INCIDENT_CASE_ID",
    "INCIDENT_SHA256",
    "INCIDENT_VERIFICATION_SHA256",
    "PREDICTION_SCHEMA",
    "RAW_WIDTH",
    "ZERO_READ_BOUNDARY",
    "accepts_joint_risk",
    "apply_feature_transform",
    "attach_prediction_receipt",
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
    "make_synthetic_receipt",
    "positive_infinity_cutoff",
    "predict_five_heads",
    "row_residuals",
    "score_with_calibration",
    "select_threshold",
    "strength_bits",
    "stratum_qualification",
    "validate_calibration",
    "validate_cutoff",
    "validate_model",
    "validate_prediction_receipt",
    "validate_recursive_synthetic_boundary",
    "validate_synthetic_receipt",
]
