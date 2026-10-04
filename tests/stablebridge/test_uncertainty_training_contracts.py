from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from stablebridge.physical_repair.uncertainty_training_contracts import (
    U0CalibrationMetricV1,
    U0CalibrationReceiptV1,
    U0CalibrationThresholdV1,
    U0CheckpointReceiptV1,
    U0ClaimSemanticsV1,
    U0NativeManifestV1,
    U0NativeStateV1,
    U0SplitRoleV1,
    U0TeacherMetricV1,
    reject_action_control_fields_v1,
    verify_u0_calibration_v1,
    verify_u0_native_checkpoint_v1,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def state(index: int, role: U0SplitRoleV1, *, group: str | None = None) -> U0NativeStateV1:
    return U0NativeStateV1(
        state_id=f"native-{index}",
        group_id=group or f"group-{index}",
        split_role=role,
        matcher_provider_id="waft-native-v1",
        matcher_checkpoint_hash=digest("matcher"),
        source_receipt_schema="x05-p01-native-matcher-receipt/v1",
        matcher_receipt_hash=digest(f"matcher-receipt-{index}"),
        input_hashes=(digest(f"first-{index}"), digest(f"second-{index}")),
        base_flow_hash=digest(f"flow-{index}"),
        base_risk_hash=digest(f"risk-{index}"),
        probe_receipt_hashes=tuple(digest(f"probe-{index}-{probe}") for probe in range(3)),
    )


def manifest() -> U0NativeManifestV1:
    return U0NativeManifestV1(
        manifest_id="u0-native-fold-0",
        split_id="component-split-v1-fold-0",
        matcher_provider_id="waft-native-v1",
        matcher_checkpoint_hash=digest("matcher"),
        teacher_recipe_hash=digest("flip-translation-brightness-v1"),
        teacher_metric=U0TeacherMetricV1.L1_VECTOR_INCONSISTENCY,
        probes_per_state=3,
        states=(
            state(0, U0SplitRoleV1.FIT),
            state(1, U0SplitRoleV1.VALIDATION),
            state(2, U0SplitRoleV1.CALIBRATION),
            state(3, U0SplitRoleV1.EVALUATION),
        ),
    )


def checkpoint(value: U0NativeManifestV1) -> U0CheckpointReceiptV1:
    return U0CheckpointReceiptV1.for_manifest(
        value,
        checkpoint_id="u0-native-fold-0-seed-11",
        feature_schema_hash=digest("workb-observable-12ch-v1"),
        normalizer_hash=digest("fit-only-normalizer"),
        model_architecture_hash=digest("conv-12-32-32-1"),
        model_state_hash=digest("model-state"),
        training_code_hash=digest("trainer"),
    )


def calibration(
    value: U0NativeManifestV1,
    model: U0CheckpointReceiptV1,
) -> U0CalibrationReceiptV1:
    return U0CalibrationReceiptV1.for_checkpoint(
        value,
        model,
        calibration_id="u0-native-fold-0-calibration",
        calibration_data_hash=digest("calibration-labels"),
        calibration_method_hash=digest("isotonic-v1"),
        support_definition_hash=digest("finite-gt-common-support"),
        target_semantics="endpoint_error_native_px",
        metrics=(
            U0CalibrationMetricV1("nll", 0.4, 128),
            U0CalibrationMetricV1("ause", 0.2, 128),
        ),
        thresholds=(
            U0CalibrationThresholdV1("severe_error", 3.0, "native_px"),
            U0CalibrationThresholdV1("high_uncertainty", 1.5, "laplace_scale"),
        ),
    )


def test_native_manifest_round_trips_and_hashes_derived_state_and_split() -> None:
    value = manifest()
    restored = U0NativeManifestV1.from_dict(value.as_dict())
    assert restored == value
    assert restored.manifest_hash == value.manifest_hash
    assert restored.native_state_set_hash == value.native_state_set_hash
    assert restored.split_hash == value.split_hash
    assert set(row.split_role for row in restored.states) == set(U0SplitRoleV1)


@pytest.mark.parametrize(
    "field",
    ["action_id", "selected_action", "control_id", "arm_name", "input_strength", "family_id"],
)
def test_manifest_mapping_strictly_rejects_action_and_control_fields(field: str) -> None:
    payload = manifest().as_dict()
    payload["states"][0][field] = "forbidden"
    with pytest.raises(ValueError, match="forbids action/control field"):
        U0NativeManifestV1.from_dict(payload)


def test_recursive_guard_rejects_deep_mixed_action_metadata() -> None:
    with pytest.raises(ValueError, match="forbids action/control field"):
        reject_action_control_fields_v1({
            "safe": {"provenance": [{"optimizer": "adamw"}, {"arm": "full"}]},
        })


def test_manifest_rejects_non_native_scope_unknown_fields_and_role_leakage() -> None:
    payload = manifest().as_dict()
    payload["source_scope"] = "mixed"
    with pytest.raises(ValueError, match="native_only"):
        U0NativeManifestV1.from_dict(payload)

    payload = manifest().as_dict()
    payload["unreviewed"] = True
    with pytest.raises(ValueError, match="fields drifted"):
        U0NativeManifestV1.from_dict(payload)

    rows = list(manifest().states)
    rows[1] = state(1, U0SplitRoleV1.VALIDATION, group=rows[0].group_id)
    with pytest.raises(ValueError, match="leaks across split"):
        replace(manifest(), states=tuple(rows), manifest_hash="")


def test_manifest_rejects_probe_reuse_and_matcher_drift() -> None:
    value = manifest()
    rows = list(value.states)
    rows[1] = replace(
        rows[1],
        probe_receipt_hashes=rows[0].probe_receipt_hashes,
        state_hash="",
    )
    with pytest.raises(ValueError, match="reuses a probe"):
        replace(value, states=tuple(rows), manifest_hash="")

    rows = list(value.states)
    rows[1] = replace(rows[1], matcher_checkpoint_hash=digest("other"), state_hash="")
    with pytest.raises(ValueError, match="matcher binding drift"):
        replace(value, states=tuple(rows), manifest_hash="")

    rows = list(value.states)
    with pytest.raises(ValueError, match="native-only producer"):
        rows[1] = replace(
            rows[1],
            source_receipt_schema="x05-p01-action-materialization-receipt/v1",
            state_hash="",
        )


def test_checkpoint_receipt_binds_all_training_identities() -> None:
    value = manifest()
    receipt = checkpoint(value)
    verify_u0_native_checkpoint_v1(value, receipt, receipt.checkpoint_binding())
    assert receipt.claim_semantics is U0ClaimSemanticsV1.SELF_CONSISTENCY_SURROGATE
    serialized = receipt.as_dict()
    assert serialized["feature_schema_hash"] == digest("workb-observable-12ch-v1")
    assert serialized["training_state_count"] == len(
        value.role_states(U0SplitRoleV1.FIT)
    )
    assert serialized["normalizer_hash"] == digest("fit-only-normalizer")
    assert serialized["teacher_recipe_hash"] == value.teacher_recipe_hash
    assert serialized["teacher_metric"] == "l1_vector_inconsistency"
    assert serialized["split_hash"] == value.split_hash
    assert serialized["model_architecture_hash"] == digest("conv-12-32-32-1")
    assert serialized["model_state_hash"] == digest("model-state")
    assert U0CheckpointReceiptV1.from_dict(serialized) == receipt


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("feature_schema_hash", digest("other-feature-schema")),
        ("normalizer_hash", digest("other-normalizer")),
        ("teacher_recipe_hash", digest("other-teacher")),
        ("split_hash", digest("other-split")),
        ("model_state_hash", digest("other-model")),
    ],
)
def test_checkpoint_binding_drift_fails_closed(field: str, replacement: str) -> None:
    value = manifest()
    receipt = checkpoint(value)
    binding = receipt.checkpoint_binding()
    binding[field] = replacement
    with pytest.raises(ValueError, match="does not match"):
        verify_u0_native_checkpoint_v1(value, receipt, binding)


def test_mixed_action_checkpoint_cannot_borrow_native_receipt() -> None:
    value = manifest()
    receipt = checkpoint(value)
    binding = receipt.checkpoint_binding()
    binding["training_data_scope"] = "mixed_native_and_action"
    binding["selected_action_count"] = 340
    with pytest.raises(ValueError, match="forbids action/control field"):
        verify_u0_native_checkpoint_v1(value, receipt, binding)

    other_manifest = replace(
        value,
        manifest_id="mixed-checkpoint-cover",
        states=tuple(reversed(value.states)),
        manifest_hash="",
    )
    # Canonical ordering makes mere input order irrelevant, but an actual state
    # substitution changes the native set and cannot borrow the receipt.
    rows = list(other_manifest.states)
    rows[0] = state(9, rows[0].split_role, group=rows[0].group_id)
    other_manifest = replace(other_manifest, states=tuple(rows), manifest_hash="")
    with pytest.raises(ValueError, match="not bound"):
        verify_u0_native_checkpoint_v1(
            other_manifest, receipt, receipt.checkpoint_binding(),
        )


def test_uncalibrated_checkpoint_cannot_claim_task_error_calibration() -> None:
    value = manifest()
    receipt = checkpoint(value)
    with pytest.raises(ValueError, match="only a consistency surrogate"):
        replace(
            receipt,
            claim_semantics=U0ClaimSemanticsV1.CALIBRATED_ENDPOINT_ERROR,
            receipt_hash="",
        )


def test_calibration_receipt_binds_checkpoint_calibration_split_metrics_and_thresholds() -> None:
    value = manifest()
    model = checkpoint(value)
    receipt = calibration(value, model)
    verify_u0_calibration_v1(value, model, receipt)
    assert receipt.claim_semantics is U0ClaimSemanticsV1.CALIBRATED_ENDPOINT_ERROR
    assert receipt.checkpoint_receipt_hash == model.receipt_hash
    assert receipt.model_state_hash == model.model_state_hash
    assert receipt.calibration_split_hash == value.role_hash(U0SplitRoleV1.CALIBRATION)
    assert receipt.calibration_state_count == 1
    assert receipt.calibration_group_count == 1
    assert [row.name for row in receipt.metrics] == ["ause", "nll"]
    assert {row.name for row in receipt.thresholds} == {
        "severe_error", "high_uncertainty",
    }
    assert U0CalibrationReceiptV1.from_dict(receipt.as_dict()) == receipt


def test_calibration_checkpoint_or_split_drift_fails_closed() -> None:
    value = manifest()
    model = checkpoint(value)
    receipt = calibration(value, model)
    with pytest.raises(ValueError, match="not bound"):
        verify_u0_calibration_v1(
            value,
            model,
            replace(receipt, calibration_split_hash=digest("wrong-split"), receipt_hash=""),
        )
    with pytest.raises(ValueError, match="not bound"):
        verify_u0_calibration_v1(
            value,
            model,
            replace(receipt, checkpoint_receipt_hash=digest("wrong-checkpoint"), receipt_hash=""),
        )


def test_calibration_requires_unique_finite_metrics_and_thresholds() -> None:
    value = manifest()
    model = checkpoint(value)
    with pytest.raises(ValueError, match="metric names must be unique"):
        U0CalibrationReceiptV1.for_checkpoint(
            value,
            model,
            calibration_id="duplicate-metrics",
            calibration_data_hash=digest("data"),
            calibration_method_hash=digest("method"),
            support_definition_hash=digest("support"),
            target_semantics="endpoint_error_native_px",
            metrics=(
                U0CalibrationMetricV1("nll", 0.4, 10),
                U0CalibrationMetricV1("nll", 0.5, 10),
            ),
            thresholds=(U0CalibrationThresholdV1("severe", 3.0, "native_px"),),
        )
    with pytest.raises(ValueError, match="must be finite"):
        U0CalibrationMetricV1("nll", float("nan"), 10)


def test_canonical_hashes_ignore_input_order_but_not_content() -> None:
    value = manifest()
    reordered = replace(value, states=tuple(reversed(value.states)), manifest_hash="")
    assert reordered.manifest_hash == value.manifest_hash
    assert reordered.native_state_set_hash == value.native_state_set_hash
    changed = replace(
        value,
        teacher_metric=U0TeacherMetricV1.L2_VECTOR_INCONSISTENCY,
        manifest_hash="",
    )
    assert changed.manifest_hash != value.manifest_hash
