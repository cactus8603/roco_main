from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from stablebridge.physical_repair.e234_model_engine import (
    CALIBRATION_HEAD_ORDER,
    DESIGN_WIDTH,
    E234EngineError,
    FIXED_NONCONFORMITY_SCALES,
    HEAD_ORDER,
    INCIDENT_CASE_ID,
    accepts_joint_risk,
    apply_feature_transform,
    build_raw_features,
    calibrate_component_max_for_query,
    canonical_json_bytes,
    canonical_sha256,
    component_max_scores,
    component_weights,
    conformal_rank,
    fit_feature_transform,
    fit_five_head_model,
    fit_logistic,
    fit_outer_fold_or_native,
    fit_ridge,
    formal_acceptance,
    make_synthetic_receipt,
    positive_infinity_cutoff,
    predict_five_heads,
    score_with_calibration,
    select_threshold,
    strength_bits,
    validate_cutoff,
    validate_calibration,
    validate_model,
    validate_recursive_synthetic_boundary,
    validate_synthetic_receipt,
)


FAMILIES = (
    "common_disk",
    "common_gaussian",
    "common_motion",
    "jpeg_deblock",
    "p05_noise_impulse_median3",
    "p05_noise_additive_wiener3",
)


def synthetic_row(
    index: int,
    component: str,
    *,
    observation_id: str | None = None,
    family: str | None = None,
    endpoint: str | None = None,
    strength: float | None = None,
    target_shift: float = 0.0,
) -> dict:
    benefit = 0.0 if index % 4 == 0 else 0.08 + 0.015 * (index % 7)
    harm = 0.0 if index % 5 == 0 else 0.03 + 0.012 * (index % 9)
    if index % 11 == 0:
        harm += 0.38
    fraction = 0.0 if index % 3 == 0 else 0.01 * (1 + index % 8)
    cvar = 0.0 if index % 6 == 0 else 0.11 + 0.02 * (index % 6)
    benefit += target_shift
    harm += target_shift * 0.3
    return {
        "fixture_scope": "SYNTHETIC_ADVERSARIAL_ONLY",
        "case_id": f"synthetic-case-{component}",
        "observation_id": observation_id or f"o-{index:03d}",
        "component_id": component,
        "role": "MODEL",
        "availability": "AVAILABLE_EXACT",
        "typed_action_signature": {
            "family": family or FAMILIES[index % len(FAMILIES)],
            "jpeg_variant": 1 if index % 4 == 0 else 0,
        },
        "action": {
            "endpoint": endpoint or ("first", "second", "both")[index % 3],
            "stage": "half" if index % 2 else "full",
            "input_strength": strength if strength is not None else 0.75 + 0.25 * (index % 4),
        },
        "output_alpha_availability": "AVAILABLE_EXACT",
        "output_alpha_mean": 0.8 + 0.04 * (index % 7),
        "bounded_response_availability": "AVAILABLE_EXACT",
        "bounded_response_mean_px": 0.1 + 0.03 * (index % 9),
        "area_px": 64 + 3 * index,
        "support": {
            "support_fraction": 0.35 + 0.05 * (index % 11),
            "supported_px": 10 + index,
        },
        "target_availability": "AVAILABLE_EXACT",
        "targets": {
            "benefit": benefit,
            "harm": harm,
            "harmed_fraction": fraction,
            "cvar95": cvar,
            "net_gain": benefit - harm,
        },
        "typed_admission": True,
    }


def populations(component_count: int = 12) -> tuple[list[dict], list[dict]]:
    model = []
    scale = []
    for component_index in range(component_count):
        for local_index in range(2):
            index = 2 * component_index + local_index
            model.append(
                synthetic_row(
                    index,
                    f"model-c{component_index:02d}",
                    observation_id=f"model-o{index:03d}",
                )
            )
            scale_row = synthetic_row(
                index + 37,
                f"scale-c{component_index:02d}",
                observation_id=f"scale-o{index:03d}",
                target_shift=0.015,
            )
            scale_row["role"] = "SCALE"
            scale_row["prediction_model_digest"] = "a" * 64
            scale_row["predictions"] = {
                "B": max(scale_row["targets"]["benefit"] - 0.025, 0.0),
                "H": max(scale_row["targets"]["harm"] - 0.02, 0.0),
                "harmed_fraction": max(
                    scale_row["targets"]["harmed_fraction"] - 0.005, 0.0
                ),
                "CVaR95": max(scale_row["targets"]["cvar95"] - 0.03, 0.0),
                "severe": 0.1 + 0.01 * (index % 4),
            }
            scale.append(scale_row)
    return model, scale


def test_exact_raw_19_to_design_39_and_missing_abstention() -> None:
    rows = [synthetic_row(index, f"c{index}") for index in range(6)]
    raw = build_raw_features(rows[0])
    assert len(raw["raw"]) == 19
    assert raw["raw"][:6] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    transform = fit_feature_transform(rows)
    encoded = apply_feature_transform(rows[0], transform)
    assert len(encoded["design"]) == DESIGN_WIDTH
    assert encoded["design"][0] == 1.0
    assert encoded["design"][20:] == [0.0] * 19

    missing = deepcopy(rows[0])
    missing["output_alpha_availability"] = "TYPED_MISSING"
    missing["output_alpha_mean"] = None
    missing["bounded_response_availability"] = "TYPED_MISSING"
    missing["bounded_response_mean_px"] = None
    diagnostic = apply_feature_transform(missing, transform)
    assert diagnostic["disposition"] == "ABSTAIN_NATIVE_FEATURE_MISSING"
    assert diagnostic["design"][12:15] == [0.0, 0.0, 0.0]
    assert diagnostic["design"][31:34] == [1.0, 1.0, 1.0]
    with pytest.raises(E234EngineError, match="MODEL_FEATURE_MISSING"):
        fit_feature_transform([rows[1], missing])


def test_row_unweighted_even_median_mad_and_zero_mad_not_constant() -> None:
    even = [
        synthetic_row(0, "c0", strength=1.0),
        synthetic_row(1, "c1", strength=3.0),
    ]
    transform = fit_feature_transform(even)
    stat = transform["statistics"][10]
    assert stat["center"] == 1.0
    assert stat["mad"] == 1.0
    assert stat["scale"] == 1.0

    zero_mad = [
        synthetic_row(0, "z0", strength=1.0),
        synthetic_row(1, "z1", strength=1.0),
        synthetic_row(2, "z2", strength=1.0 + 1e-7),
    ]
    transform = fit_feature_transform(zero_mad)
    stat = transform["statistics"][10]
    assert stat["mad"] == 0.0
    assert stat["scale"] == 1.0
    assert stat["constant_raw"] is False
    assert 11 not in transform["fixed_zero_columns"]
    assert 0 not in transform["fixed_zero_columns"]
    assert canonical_json_bytes(-0.0) == b"0.0\n"


def test_component_weights_equalize_components_and_have_mean_one() -> None:
    rows = [
        {"observation_id": "a", "component_id": "one"},
        {"observation_id": "b", "component_id": "two"},
        {"observation_id": "c", "component_id": "two"},
        {"observation_id": "d", "component_id": "two"},
    ]
    weights, artifact = component_weights(rows)
    assert float(np.mean(weights)) == pytest.approx(1.0, abs=3e-16)
    assert sum(weights[:1]) == pytest.approx(sum(weights[1:]))
    assert artifact["component_counts"] == {"one": 1, "two": 3}
    assert [row["observation_id"] for row in artifact["weight_records"]] == [
        "a",
        "b",
        "c",
        "d",
    ]


@pytest.mark.parametrize("target_value", [0.0, 1.0])
def test_degenerate_logistic_uses_weighted_jeffreys_and_zero_coefficients(
    target_value: float,
) -> None:
    design = np.ones((3, DESIGN_WIDTH), dtype=np.float64)
    target = np.full(3, target_value, dtype=np.float64)
    weights = np.asarray([0.5, 1.0, 1.5], dtype=np.float64)
    fit = fit_logistic(design, target, weights, list(range(1, DESIGN_WIDTH)))
    expected = (3.0 * target_value + 0.5) / 4.0
    assert fit["state"] == "CONSTANT_DEGENERATE_CLASS"
    assert fit["constant_probability"] == expected
    assert fit["coefficients"] == [0.0] * DESIGN_WIDTH
    assert fit["optimization_executed"] is False


def test_ridge_equation_has_unpenalized_intercept_and_fixed_zero_restore() -> None:
    design = np.zeros((3, DESIGN_WIDTH), dtype=np.float64)
    design[:, 0] = 1.0
    target = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    weights = np.asarray([1.0, 1.0, 2.0], dtype=np.float64)
    fit = fit_ridge(design, target, weights, list(range(1, DESIGN_WIDTH)))
    assert fit["coefficients"][0] == 2.25
    assert fit["coefficients"][1:] == [0.0] * (DESIGN_WIDTH - 1)


def test_logistic_constants_and_failure_are_fail_closed() -> None:
    design = np.zeros((2, DESIGN_WIDTH), dtype=np.float64)
    design[:, 0] = 1.0
    with pytest.raises(E234EngineError, match="LOGISTIC_NONCONVERGENCE"):
        fit_logistic(
            design,
            np.asarray([0.0, 1.0]),
            np.ones(2),
            list(range(1, DESIGN_WIDTH)),
            max_iterations=0,
        )
    bad = synthetic_row(0, "bad")
    bad["availability"] = "IDENTITY_ONLY"
    state = fit_outer_fold_or_native([bad])
    assert state["status"] == "FOLD_NATIVE_MODEL_FAILURE"
    assert state["immutable_native"] is True
    assert state["retry_authorized"] is False
    assert state["alternate_solver_authorized"] is False


def test_five_head_fit_predict_states_support_and_model_digest() -> None:
    model_rows, _ = populations()
    model = fit_five_head_model(model_rows)
    assert model["head_order"] == list(HEAD_ORDER)
    assert model["design_width"] == DESIGN_WIDTH
    assert model["solver"] == {
        "lambda": 1.0,
        "max_iterations": 200,
        "gradient_infinity_tolerance": 1e-10,
        "relative_objective_tolerance": 1e-12,
        "armijo_c1": 1e-4,
        "backtracking": 0.5,
        "minimum_step": 2.0**-30,
        "sigmoid_predictor_clip": [-40.0, 40.0],
        "intercept_unpenalized": True,
    }
    assert model["heads"]["B"]["support_conditioned"] is True
    assert model["heads"]["H"]["support_conditioned"] is True
    assert model["heads"]["CVaR95"]["support_conditioned"] is True
    assert model["heads"]["harmed_fraction"]["support_conditioned"] is False
    prediction = predict_five_heads(model, [model_rows[0]])[0]
    assert list(prediction["predictions"]) == list(HEAD_ORDER)
    assert 0.0 <= prediction["predictions"]["harmed_fraction"] <= 1.0
    assert 0.0 <= prediction["predictions"]["severe"] <= 1.0

    mutated = deepcopy(model)
    fixed_column = mutated["transform"]["fixed_zero_columns"][0]
    mutated["heads"]["B"]["occurrence"]["coefficients"][fixed_column] = 1.0
    mutated["model_digest"] = canonical_sha256(
        {key: value for key, value in mutated.items() if key != "model_digest"}
    )
    with pytest.raises(E234EngineError, match="FIXED_ZERO_COEFFICIENT_DRIFT"):
        validate_model(mutated)

    solver_mutation = deepcopy(model)
    solver_mutation["solver"]["lambda"] = 2.0
    solver_mutation["model_digest"] = canonical_sha256(
        {key: value for key, value in solver_mutation.items() if key != "model_digest"}
    )
    with pytest.raises(E234EngineError, match="MODEL_SOLVER_CONTRACT_DRIFT"):
        validate_model(solver_mutation)

    transform_mutation = deepcopy(model)
    transform_mutation["transform"]["centers"][0] += 0.25
    transform_mutation["model_digest"] = canonical_sha256(
        {key: value for key, value in transform_mutation.items() if key != "model_digest"}
    )
    with pytest.raises(E234EngineError, match="TRANSFORM_STATISTIC_RELATION_DRIFT"):
        validate_model(transform_mutation)


def test_zero_positive_hurdles_enter_exact_zero_magnitude_state() -> None:
    rows = [synthetic_row(index, f"zero-c{index}") for index in range(8)]
    for row in rows:
        row["targets"] = {
            "benefit": 0.0,
            "harm": 0.0,
            "harmed_fraction": 0.0,
            "cvar95": 0.0,
            "net_gain": 0.0,
        }
    model = fit_five_head_model(rows)
    for head in ("B", "H", "harmed_fraction", "CVaR95"):
        assert model["heads"][head]["magnitude"] == {
            "state": "CONSTANT_ZERO_NO_POSITIVE_TARGET",
            "coefficients": [0.0] * DESIGN_WIDTH,
            "optimization_executed": False,
        }
        assert model["heads"][head]["occurrence"]["state"] == (
            "CONSTANT_DEGENERATE_CLASS"
        )


def test_component_max_is_one_per_component_across_rows_heads_and_ties() -> None:
    row_a = synthetic_row(0, "tie", observation_id="a")
    row_b = synthetic_row(1, "tie", observation_id="b")
    for row in (row_a, row_b):
        row["role"] = "SCALE"
        row["prediction_model_digest"] = "a" * 64
        row["targets"] = {
            "benefit": 0.25,
            "harm": 0.25,
            "harmed_fraction": 1.0,
            "cvar95": 0.25,
            "net_gain": 0.25,
        }
        row["predictions"] = {
            "B": 0.0,
            "H": 0.0,
            "harmed_fraction": 0.0,
            "CVaR95": 0.0,
            "severe": 0.0,
        }
    artifact = component_max_scores([row_b, row_a])
    assert artifact["fixed_scales"] == FIXED_NONCONFORMITY_SCALES
    assert len(artifact["component_scores"]) == 1
    score = artifact["component_scores"][0]
    assert score["R_c"] == 1.0
    assert score["witness_observation_id"] == "a"
    assert score["witness_head"] == "net"
    assert score["witness_head_index"] == 0
    assert len(score["row_residual_hashes"][0]["sha256_by_head"]) == 4


def test_conformal_rank_has_no_clipping_and_m9_is_first_available() -> None:
    for component_count in range(1, 9):
        assert conformal_rank(component_count) > component_count
    assert conformal_rank(9) == 9
    assert conformal_rank(9) / 10 == 0.9
    for component_count in (10, 19, 100, 1000):
        rank = conformal_rank(component_count)
        assert rank <= component_count
        assert rank / (component_count + 1) >= 0.9


def test_fallback_to_global_one_q_four_radii_and_separate_risk_scales() -> None:
    model_rows, scale_rows = populations(12)
    query = synthetic_row(
        999,
        "query",
        family="common_disk",
        endpoint="first",
        strength=9.5,
    )
    calibration = calibrate_component_max_for_query(model_rows, scale_rows, query)
    assert calibration["status"] == "QUALIFIED"
    assert calibration["fallback_level_index"] == 3
    assert calibration["selected_key"] == {
        "level": ["GLOBAL"],
        "values": ["GLOBAL"],
    }
    assert len(calibration["component_score_artifact"]["component_scores"]) == 12
    assert calibration["radii"] == {
        head: calibration["Q"] * FIXED_NONCONFORMITY_SCALES[head]
        for head in CALIBRATION_HEAD_ORDER
    }
    assert set(calibration["risk_scales"]) == {"H", "harmed_fraction", "CVaR95"}
    assert all(
        record["risk_scale"] >= 0.01
        for record in calibration["risk_scales"].values()
    )
    assert set(calibration["population_hashes"]) == {
        "fixed_nonconformity_scale_table",
        "one_score_per_component",
        "row_level_risk_scale",
    }
    assert calibration["severe_probability_calibrated"] is False


def test_global_insufficiency_has_no_q_and_is_immutable_native() -> None:
    model_rows, scale_rows = populations(8)
    result = calibrate_component_max_for_query(model_rows, scale_rows, model_rows[0])
    assert result["status"] == "FOLD_NATIVE_GLOBAL_INSUFFICIENT"
    assert result["Q_serialized"] is False
    assert "Q" not in result
    assert result["native_fallback"] is True
    assert all(not attempt["rank_available"] for attempt in result["qualification_attempts"])


def test_model_scale_role_identity_and_component_disjointness_fail_closed() -> None:
    model_rows, scale_rows = populations(9)
    wrong_role = deepcopy(scale_rows)
    wrong_role[0]["role"] = "MODEL"
    with pytest.raises(E234EngineError, match="SCALE_ROLE_IDENTITY_DRIFT"):
        calibrate_component_max_for_query(model_rows, wrong_role, model_rows[0])

    overlap = deepcopy(scale_rows)
    overlap[0]["component_id"] = model_rows[0]["component_id"]
    with pytest.raises(E234EngineError, match="MODEL_SCALE_COMPONENT_OVERLAP"):
        calibrate_component_max_for_query(model_rows, overlap, model_rows[0])

    observation_overlap = deepcopy(scale_rows)
    observation_overlap[0]["observation_id"] = model_rows[0]["observation_id"]
    with pytest.raises(E234EngineError, match="MODEL_SCALE_OBSERVATION_OVERLAP"):
        calibrate_component_max_for_query(model_rows, observation_overlap, model_rows[0])


def test_calibration_is_deeply_typed_and_status_or_resigning_is_insufficient() -> None:
    model_rows, scale_rows = populations(12)
    query = synthetic_row(999, "query", strength=9.5)
    calibration = calibrate_component_max_for_query(model_rows, scale_rows, query)
    validate_calibration(calibration)
    with pytest.raises(E234EngineError, match="CALIBRATION_KEY_SET_DRIFT"):
        validate_calibration({"status": "QUALIFIED"})

    q_mutation = deepcopy(calibration)
    q_mutation["Q"] += 1.0
    q_mutation["calibration_digest"] = canonical_sha256(
        {key: value for key, value in q_mutation.items() if key != "calibration_digest"}
    )
    with pytest.raises(E234EngineError, match="CALIBRATION_Q_SELECTION_DRIFT"):
        validate_calibration(q_mutation)

    overlap_mutation = deepcopy(calibration)
    selected = overlap_mutation["qualification_attempts"][-1]
    selected["scale_component_ids"][0] = selected["model_component_ids"][0]
    selected["scale_component_ids"].sort()
    selected["membership_digest"] = canonical_sha256(
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
    overlap_mutation["calibration_digest"] = canonical_sha256(
        {key: value for key, value in overlap_mutation.items() if key != "calibration_digest"}
    )
    with pytest.raises(E234EngineError, match="CALIBRATION_ROLE_MEMBERSHIP_OVERLAP"):
        validate_calibration(overlap_mutation)

    prediction = {
        "model_digest": "b" * 64,
        "predictions": {
            "B": 0.2,
            "H": 0.1,
            "harmed_fraction": 0.01,
            "CVaR95": 0.1,
            "severe": 0.05,
        },
    }
    with pytest.raises(E234EngineError, match="PREDICTION_MODEL_PROVENANCE_DRIFT"):
        score_with_calibration(query, prediction, calibration)


def test_strength_bits_normalizes_signed_zero() -> None:
    assert strength_bits(-0.0) == strength_bits(0.0) == "0000000000000000"


def test_tagged_sentinel_canonical_strict_accept_and_numeric_rejections() -> None:
    sentinel = positive_infinity_cutoff()
    assert canonical_json_bytes(sentinel) == b'{"kind":"POSITIVE_INFINITY"}\n'
    assert accepts_joint_risk(1e300, sentinel) is True
    assert accepts_joint_risk(0.5, {"kind": "FINITE", "value": 0.5}) is False
    assert accepts_joint_risk(0.499, {"kind": "FINITE", "value": 0.5}) is True
    assert formal_acceptance(True, 1.0, 0.5, 0.5, sentinel) is True
    assert formal_acceptance(True, 0.5, 0.5, 0.1, sentinel) is False
    with pytest.raises(E234EngineError):
        validate_cutoff({"kind": "POSITIVE_INFINITY", "value": 1.0})
    with pytest.raises(E234EngineError):
        validate_cutoff({"kind": "FINITE", "value": math.inf})
    with pytest.raises(E234EngineError):
        validate_cutoff({"kind": "FINITE", "value": True})
    with pytest.raises(E234EngineError):
        validate_cutoff({"kind": "OTHER"})


def test_threshold_candidates_ordered_fsum_and_tie_rule() -> None:
    rows = []
    for index, (risk, gain) in enumerate(
        [(0.2, 0.45), (0.1, 0.45), (0.3, 0.10), (0.3, -0.4)]
    ):
        row = synthetic_row(index + 30, f"threshold-c{index}")
        row["observation_id"] = f"threshold-o{index}"
        row["joint_risk"] = risk
        row["predicted_expected_net"] = 0.2
        row["targets"]["net_gain"] = gain
        rows.append(row)
    artifact = select_threshold(list(reversed(rows)))
    candidates = [record["candidate"] for record in artifact["candidates"]]
    assert candidates == [
        {"kind": "FINITE", "value": 0.1},
        {"kind": "FINITE", "value": 0.2},
        {"kind": "FINITE", "value": 0.3},
        {"kind": "POSITIVE_INFINITY"},
    ]
    assert artifact["ordered_observation_ids"] == sorted(
        row["observation_id"] for row in rows
    )
    assert artifact["retention_sole_tolerance"] == 1e-12
    assert artifact["selected_cutoff"]["kind"] in ("FINITE", "POSITIVE_INFINITY")


def test_recursive_synthetic_boundary_blocks_nested_sources_and_incident() -> None:
    row = synthetic_row(0, "synthetic")
    validate_recursive_synthetic_boundary({"rows": [row]})
    with pytest.raises(E234EngineError, match="FORBIDDEN_FIELD"):
        validate_recursive_synthetic_boundary(
            {"rows": [row], "metadata": {"nested": {"payload_path": "/real/data"}}}
        )
    incident = deepcopy(row)
    incident["case_id"] = INCIDENT_CASE_ID
    with pytest.raises(E234EngineError, match="INCIDENT_CASE_FORBIDDEN"):
        validate_recursive_synthetic_boundary({"nested": [{"deeper": incident}]})
    wrong_scope = deepcopy(row)
    wrong_scope["fixture_scope"] = "REAL"
    with pytest.raises(E234EngineError, match="SYNTHETIC_FIXTURE_SCOPE_REQUIRED"):
        validate_recursive_synthetic_boundary(wrong_scope)


def test_synthetic_receipt_requires_incident_case_set_and_recursive_boundary() -> None:
    request = {"rows": [synthetic_row(0, "r0"), synthetic_row(1, "r1")]}
    output = {"summary": {"finite": True}}
    implementation_sha = "a" * 64
    receipt = make_synthetic_receipt(
        stage="SYNTHETIC_MODEL",
        request=request,
        output=output,
        implementation_sha256=implementation_sha,
    )
    validate_synthetic_receipt(
        receipt,
        request=request,
        output=output,
        implementation_sha256=implementation_sha,
    )
    assert receipt["incident_hash"]
    assert receipt["verification_hash"]
    assert receipt["selected_case_set_digest"]
    mutated = deepcopy(receipt)
    mutated.pop("incident_hash")
    with pytest.raises(E234EngineError, match="RECEIPT_KEY_SET_DRIFT"):
        validate_synthetic_receipt(
            mutated,
            request=request,
            output=output,
            implementation_sha256=implementation_sha,
        )
    with pytest.raises(E234EngineError, match="FORBIDDEN_FIELD"):
        make_synthetic_receipt(
            stage="SYNTHETIC_MODEL",
            request={"wrapper": {"source": {"path": "/real"}}},
            output=output,
            implementation_sha256=implementation_sha,
        )
