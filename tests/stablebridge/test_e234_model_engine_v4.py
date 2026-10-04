from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import runpy

import pytest

from stablebridge.physical_repair import e234_model_engine_v4 as engine


ROOT = Path(__file__).resolve().parents[2]
V3_TESTS = runpy.run_path(str(ROOT / "tests/stablebridge/test_e234_model_engine.py"))
synthetic_row = V3_TESTS["synthetic_row"]
populations = V3_TESTS["populations"]


def _resign_model(model: dict) -> None:
    model["model_digest"] = engine.canonical_sha256(
        {key: value for key, value in model.items() if key != "model_digest"}
    )


def _resign_calibration(calibration: dict) -> None:
    calibration["calibration_digest"] = engine.canonical_sha256(
        {
            key: value
            for key, value in calibration.items()
            if key != "calibration_digest"
        }
    )


@pytest.fixture(scope="module")
def artifacts() -> dict:
    model_rows, scale_rows = populations(12)
    model = engine.fit_five_head_model(model_rows)
    receipts = engine.predict_five_heads(model, scale_rows)
    bound_scale = [
        engine.attach_prediction_receipt(row, receipt)
        for row, receipt in zip(scale_rows, receipts, strict=True)
    ]
    query = synthetic_row(999, "query", strength=9.5)
    calibration = engine.calibrate_component_max_for_query(
        model_rows, bound_scale, query,
    )
    return {
        "model_rows": model_rows,
        "scale_rows": bound_scale,
        "model": model,
        "receipts": receipts,
        "query": query,
        "calibration": calibration,
    }


def test_append_only_parent_and_fail_review_bindings_are_exact() -> None:
    assert hashlib.sha256(
        (ROOT / "src/stablebridge/physical_repair/e234_model_engine.py").read_bytes()
    ).hexdigest() == engine.E234_CORE_V3_SOURCE_SHA256
    assert hashlib.sha256(
        (
            ROOT
            / "research/e234_model_engine_root_review_20261003_v2/REVIEW.json"
        ).read_bytes()
    ).hexdigest() == engine.E234_CORE_V3_FAIL_REVIEW_SHA256


def test_v4_fit_predict_calibrate_and_score_positive_path(artifacts: dict) -> None:
    engine.validate_model(artifacts["model"])
    engine.validate_calibration(artifacts["calibration"])
    receipt = artifacts["receipts"][0]
    engine.validate_prediction_receipt(
        receipt,
        artifacts["scale_rows"][0],
        model=artifacts["model"],
    )
    score = engine.score_with_calibration(
        artifacts["query"], artifacts["model"], artifacts["calibration"]
    )
    assert score["schema"] == "e234-b1-calibrated-score/v4"
    assert score["observation_id"] == artifacts["query"]["observation_id"]
    assert score["component_id"] == artifacts["query"]["component_id"]
    assert len(score["prediction_digest"]) == 64


def test_c01_exact_logistic_and_hurdle_nested_key_sets(artifacts: dict) -> None:
    attacked = deepcopy(artifacts["model"])
    attacked["heads"]["B"]["occurrence"]["attacker_extra"] = "accepted-by-v3"
    _resign_model(attacked)
    with pytest.raises(engine.E234EngineError, match="LOGISTIC_STATE_KEY_SET_DRIFT"):
        engine.validate_model(attacked)

    attacked = deepcopy(artifacts["model"])
    attacked["heads"]["B"]["attacker_extra"] = "accepted-by-v3"
    _resign_model(attacked)
    with pytest.raises(engine.E234EngineError, match="HURDLE_HEAD_KEY_SET_DRIFT"):
        engine.validate_model(attacked)


@pytest.mark.parametrize("iterations", [-999, 201, True, 1.5])
def test_c01_logistic_iteration_type_and_range(artifacts: dict, iterations) -> None:
    attacked = deepcopy(artifacts["model"])
    fitted = next(
        candidate
        for name in (*engine.HURDLE_HEADS, "severe")
        for candidate in [
            attacked["heads"][name]
            if name == "severe"
            else attacked["heads"][name]["occurrence"]
        ]
        if candidate["state"] == "FITTED"
    )
    fitted["iterations"] = iterations
    _resign_model(attacked)
    with pytest.raises(engine.E234EngineError, match="LOGISTIC_ITERATION_RANGE_DRIFT"):
        engine.validate_model(attacked)


def test_c02_risk_scale_formula_is_exact(artifacts: dict) -> None:
    attacked = deepcopy(artifacts["calibration"])
    record = attacked["risk_scales"]["H"]
    expected = max(
        record["positive_target_median"],
        record["q75_abs_residual"],
        record["floor"],
    )
    record["risk_scale"] = expected + 10.0
    _resign_calibration(attacked)
    with pytest.raises(engine.E234EngineError, match="RISK_SCALE_FORMULA_DRIFT"):
        engine.validate_calibration(attacked)


def test_c02_median_and_q75_record_relations_are_deep(artifacts: dict) -> None:
    attacked = deepcopy(artifacts["calibration"])
    record = attacked["risk_scales"]["H"]
    record["median_central_observation_ids"] = ["not-in-positive-population"]
    _resign_calibration(attacked)
    with pytest.raises(engine.E234EngineError, match="RISK_SCALE_MEDIAN_RECORD_DRIFT"):
        engine.validate_calibration(attacked)

    attacked = deepcopy(artifacts["calibration"])
    record = attacked["risk_scales"]["H"]
    record["q75_abs_residual"] = -0.5
    record["risk_scale"] = max(
        record["positive_target_median"], record["q75_abs_residual"], 0.01
    )
    _resign_calibration(attacked)
    with pytest.raises(
        engine.E234EngineError,
        match="RISK_SCALE_(VALUE|REDUCTION_RECORD)_DRIFT",
    ):
        engine.validate_calibration(attacked)


def test_c03_global_key_values_and_exact_key_schema(artifacts: dict) -> None:
    attacked = deepcopy(artifacts["calibration"])
    selected = attacked["qualification_attempts"][attacked["fallback_level_index"]]
    assert selected["key"] == {"level": ["GLOBAL"], "values": ["GLOBAL"]}
    selected["key"]["values"] = ["ATTACK"]
    selected["membership_digest"] = engine.canonical_sha256(
        {
            name: selected[name]
            for name in (
                "key",
                "model_observation_ids",
                "scale_observation_ids",
                "model_component_ids",
                "scale_component_ids",
            )
        }
    )
    attacked["selected_key"] = deepcopy(selected["key"])
    _resign_calibration(attacked)
    with pytest.raises(engine.E234EngineError, match="CALIBRATION_GLOBAL_KEY_VALUE_DRIFT"):
        engine.validate_calibration(attacked)


def test_c03_component_scores_partition_scale_observations_exactly_once(
    artifacts: dict,
) -> None:
    attacked = deepcopy(artifacts["calibration"])
    scores = attacked["component_score_artifact"]["component_scores"]
    selected = attacked["selected_component_score"]
    target = next(score for score in scores if score != selected)
    donor = next(score for score in scores if score is not target)
    target["observation_ids"] = deepcopy(donor["observation_ids"])
    target["row_residual_hashes"] = deepcopy(donor["row_residual_hashes"])
    target["witness_observation_id"] = target["observation_ids"][0]
    target["membership_digest"] = engine.canonical_sha256(
        {
            "component_id": target["component_id"],
            "observation_ids": target["observation_ids"],
        }
    )
    population_digest = engine.canonical_sha256(
        [
            {
                "component_id": score["component_id"],
                "membership_digest": score["membership_digest"],
            }
            for score in scores
        ]
    )
    attacked["component_score_artifact"]["component_population_digest"] = population_digest
    attacked["population_hashes"]["one_score_per_component"] = population_digest
    _resign_calibration(attacked)
    with pytest.raises(
        engine.E234EngineError,
        match="COMPONENT_SCORE_OBSERVATION_PARTITION_DRIFT",
    ):
        engine.validate_calibration(attacked)


def test_c04_prediction_receipt_binds_observation_component_and_exact_schema(
    artifacts: dict,
) -> None:
    receipt = artifacts["receipts"][0]
    other_row = artifacts["scale_rows"][1]
    with pytest.raises(engine.E234EngineError, match="PREDICTION_ROW_IDENTITY_DRIFT"):
        engine.validate_prediction_receipt(receipt, other_row)

    attacked = deepcopy(receipt)
    attacked["extra"] = True
    with pytest.raises(engine.E234EngineError, match="PREDICTION_RECEIPT_KEY_SET_DRIFT"):
        engine.validate_prediction_receipt(attacked, artifacts["scale_rows"][0])


def test_c04_scoring_replays_full_model_and_rejects_self_signed_prediction(
    artifacts: dict,
) -> None:
    row = artifacts["scale_rows"][0]
    attacked = deepcopy(artifacts["receipts"][0])
    attacked["predictions"]["B"] = 999.0
    attacked["expected_net"] = attacked["predictions"]["B"] - attacked["predictions"]["H"]
    attacked["prediction_digest"] = engine.canonical_sha256(
        {key: value for key, value in attacked.items() if key != "prediction_digest"}
    )
    # A typed digest provides structural integrity, not model authority.
    engine.validate_prediction_receipt(attacked, row)
    with pytest.raises(engine.E234EngineError, match="PREDICTION_REPLAY_DRIFT"):
        engine.validate_prediction_receipt(attacked, row, model=artifacts["model"])

    score = engine.score_with_calibration(
        row, artifacts["model"], artifacts["calibration"]
    )
    assert score["predictions"] == artifacts["receipts"][0]["predictions"]
    assert score["predicted_expected_net"] != 999.0


def test_scale_calibration_requires_typed_prediction_receipts(artifacts: dict) -> None:
    attacked = deepcopy(artifacts["scale_rows"])
    attacked[0].pop("prediction_receipt")
    with pytest.raises(engine.E234EngineError, match="TYPED_PREDICTION_RECEIPT_REQUIRED"):
        engine.calibrate_component_max_for_query(
            artifacts["model_rows"], attacked, artifacts["query"]
        )

    attacked = deepcopy(artifacts["scale_rows"])
    attacked[0]["predictions"]["B"] += 1.0
    with pytest.raises(engine.E234EngineError, match="SCALE_PREDICTION_RECEIPT_DRIFT"):
        engine.calibrate_component_max_for_query(
            artifacts["model_rows"], attacked, artifacts["query"]
        )


def test_active_coefficient_resign_remains_external_authority_open(
    artifacts: dict,
) -> None:
    attacked = deepcopy(artifacts["model"])
    record = next(
        candidate
        for name in (*engine.HURDLE_HEADS, "severe")
        for candidate in [
            attacked["heads"][name]
            if name == "severe"
            else attacked["heads"][name]["occurrence"]
        ]
        if candidate["state"] == "FITTED"
    )
    active = next(
        index
        for index in range(engine.DESIGN_WIDTH)
        if index not in attacked["transform"]["fixed_zero_columns"]
    )
    record["coefficients"][active] += 100.0
    _resign_model(attacked)
    engine.validate_model(attacked)
    original = engine.predict_five_heads(
        artifacts["model"], [artifacts["scale_rows"][0]]
    )[0]
    changed = engine.predict_five_heads(
        attacked, [artifacts["scale_rows"][0]]
    )[0]
    assert changed["predictions"] != original["predictions"]


def test_area_px_follows_sealed_continuous_nonnegative_recipe() -> None:
    row = synthetic_row(0, "area")
    row["area_px"] = 64.5
    built = engine.build_raw_features(row)
    assert built["raw"][16] > 0.0
