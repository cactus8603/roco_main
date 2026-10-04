"""Native-only U0 uncertainty training and calibration contracts.

These contracts deliberately sit outside :mod:`uncertainty_aware_flow` so the
runtime primitives remain independent of any particular training bank.  U0 is
defined here as an observer trained only from native matcher states.  Action,
arm, control, strength, and family fields are therefore rejected rather than
silently ignored.

The receipts are provenance contracts, not scientific guarantees.  In
particular, an augmentation-consistency checkpoint remains a consistency
surrogate until a separate calibration receipt binds it to a declared task
error target on the calibration split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from typing import Mapping, Sequence


U0_NATIVE_TRAINING_SCHEMA_V1 = "stablebridge-u0-native-training/v1"
U0_NATIVE_CHECKPOINT_SCHEMA_V1 = "stablebridge-u0-native-checkpoint/v1"
U0_CALIBRATION_SCHEMA_V1 = "stablebridge-u0-calibration/v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")
_FORBIDDEN_FIELD_TOKENS = frozenset({
    "action", "actions", "arm", "arms", "beta", "candidate", "candidates",
    "control", "controls", "family", "families", "strength", "strengths",
})


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty canonical text")
    return value


def _name(value: object, name: str) -> str:
    result = _text(value, name)
    if _CANONICAL_NAME.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase canonical name")
    return result


def _digest(value: object, name: str) -> str:
    result = _text(value, name)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _field_tokens(value: str) -> set[str]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return {
        token for token in re.split(r"[^a-z0-9]+", value.lower()) if token
    }


def reject_action_control_fields_v1(value: object, *, location: str = "root") -> None:
    """Reject action-bank vocabulary anywhere in a JSON-like mapping.

    Exact typed constructors already exclude unknown fields.  This recursive
    guard additionally protects mapping ingestion and checkpoint sidecars,
    including nested metadata that a shallow allowlist could miss.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} contains a non-string field name")
            forbidden = _field_tokens(key) & _FORBIDDEN_FIELD_TOKENS
            if forbidden:
                raise ValueError(
                    f"native-only U0 forbids action/control field {location}.{key}"
                )
            reject_action_control_fields_v1(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_action_control_fields_v1(child, location=f"{location}[{index}]")


def _require_exact_fields(
    value: Mapping[str, object], expected: set[str], name: str,
) -> None:
    reject_action_control_fields_v1(value, location=name)
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{name} fields drifted; missing={missing}, extra={extra}")


class U0SplitRoleV1(str, Enum):
    FIT = "fit"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    EVALUATION = "evaluation"


class U0TeacherMetricV1(str, Enum):
    L1_VECTOR_INCONSISTENCY = "l1_vector_inconsistency"
    L2_VECTOR_INCONSISTENCY = "l2_vector_inconsistency"


class U0ClaimSemanticsV1(str, Enum):
    SELF_CONSISTENCY_SURROGATE = "self_consistency_surrogate"
    CALIBRATED_ENDPOINT_ERROR = "calibrated_endpoint_error"


@dataclass(frozen=True)
class U0NativeStateV1:
    state_id: str
    group_id: str
    split_role: U0SplitRoleV1
    matcher_provider_id: str
    matcher_checkpoint_hash: str
    source_receipt_schema: str
    matcher_receipt_hash: str
    input_hashes: tuple[str, str]
    base_flow_hash: str
    base_risk_hash: str
    probe_receipt_hashes: tuple[str, ...]
    state_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        _text(self.state_id, "U0 state id")
        _text(self.group_id, "U0 group id")
        if not isinstance(self.split_role, U0SplitRoleV1):
            raise ValueError("U0 split role must be typed")
        _text(self.matcher_provider_id, "matcher provider id")
        source_schema = _text(self.source_receipt_schema, "native source receipt schema")
        source_tokens = _field_tokens(source_schema)
        if "native" not in source_tokens or source_tokens & _FORBIDDEN_FIELD_TOKENS:
            raise ValueError("U0 source receipt schema must attest a native-only producer")
        for value, name in (
            (self.matcher_checkpoint_hash, "matcher checkpoint hash"),
            (self.matcher_receipt_hash, "matcher receipt hash"),
            (self.base_flow_hash, "base flow hash"),
            (self.base_risk_hash, "base risk hash"),
        ):
            _digest(value, name)
        inputs = tuple(self.input_hashes)
        if len(inputs) != 2:
            raise ValueError("native U0 state needs exactly two input hashes")
        for value in inputs:
            _digest(value, "U0 input hash")
        probes = tuple(self.probe_receipt_hashes)
        if not probes or len(probes) != len(set(probes)):
            raise ValueError("native U0 state needs unique probe receipts")
        for value in probes:
            _digest(value, "U0 probe receipt hash")
        object.__setattr__(self, "input_hashes", inputs)
        object.__setattr__(self, "probe_receipt_hashes", probes)
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.state_hash and self.state_hash != expected:
            raise ValueError("native U0 state hash drift")
        object.__setattr__(self, "state_hash", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "state_id": self.state_id,
            "group_id": self.group_id,
            "split_role": self.split_role.value,
            "source_scope": "native_only",
            "matcher_provider_id": self.matcher_provider_id,
            "matcher_checkpoint_hash": self.matcher_checkpoint_hash,
            "source_receipt_schema": self.source_receipt_schema,
            "matcher_receipt_hash": self.matcher_receipt_hash,
            "input_hashes": list(self.input_hashes),
            "base_flow_hash": self.base_flow_hash,
            "base_risk_hash": self.base_risk_hash,
            "probe_receipt_hashes": list(self.probe_receipt_hashes),
        }
        if include_hash:
            result["state_hash"] = self.state_hash
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "U0NativeStateV1":
        expected = {
            "state_id", "group_id", "split_role", "source_scope",
            "matcher_provider_id", "matcher_checkpoint_hash",
            "source_receipt_schema", "matcher_receipt_hash", "input_hashes", "base_flow_hash",
            "base_risk_hash", "probe_receipt_hashes", "state_hash",
        }
        _require_exact_fields(value, expected, "U0 native state")
        if value["source_scope"] != "native_only":
            raise ValueError("U0 state source scope must be native_only")
        return cls(
            state_id=value["state_id"],
            group_id=value["group_id"],
            split_role=U0SplitRoleV1(value["split_role"]),
            matcher_provider_id=value["matcher_provider_id"],
            matcher_checkpoint_hash=value["matcher_checkpoint_hash"],
            source_receipt_schema=value["source_receipt_schema"],
            matcher_receipt_hash=value["matcher_receipt_hash"],
            input_hashes=tuple(value["input_hashes"]),
            base_flow_hash=value["base_flow_hash"],
            base_risk_hash=value["base_risk_hash"],
            probe_receipt_hashes=tuple(value["probe_receipt_hashes"]),
            state_hash=value["state_hash"],
        )


def _role_projection(
    states: Sequence[U0NativeStateV1], role: U0SplitRoleV1,
) -> list[dict[str, str]]:
    return [
        {"state_id": row.state_id, "group_id": row.group_id, "state_hash": row.state_hash}
        for row in sorted(states, key=lambda item: item.state_id)
        if row.split_role is role
    ]


@dataclass(frozen=True)
class U0NativeManifestV1:
    manifest_id: str
    split_id: str
    matcher_provider_id: str
    matcher_checkpoint_hash: str
    teacher_recipe_hash: str
    teacher_metric: U0TeacherMetricV1
    probes_per_state: int
    states: tuple[U0NativeStateV1, ...]
    schema: str = U0_NATIVE_TRAINING_SCHEMA_V1
    manifest_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if self.schema != U0_NATIVE_TRAINING_SCHEMA_V1:
            raise ValueError("U0 native manifest schema drift")
        _text(self.manifest_id, "U0 manifest id")
        _text(self.split_id, "U0 split id")
        _text(self.matcher_provider_id, "matcher provider id")
        _digest(self.matcher_checkpoint_hash, "matcher checkpoint hash")
        _digest(self.teacher_recipe_hash, "teacher recipe hash")
        if not isinstance(self.teacher_metric, U0TeacherMetricV1):
            raise ValueError("U0 teacher metric must be typed")
        probe_count = _positive_integer(self.probes_per_state, "probes per state")
        states = tuple(self.states)
        if not states or any(not isinstance(row, U0NativeStateV1) for row in states):
            raise ValueError("U0 native manifest needs typed states")
        if len({row.state_id for row in states}) != len(states):
            raise ValueError("U0 native manifest has duplicate state ids")
        if len({row.state_hash for row in states}) != len(states):
            raise ValueError("U0 native manifest has duplicate state identities")
        all_probes = [value for row in states for value in row.probe_receipt_hashes]
        if len(all_probes) != len(set(all_probes)):
            raise ValueError("U0 native manifest reuses a probe receipt")
        for row in states:
            if (
                row.matcher_provider_id != self.matcher_provider_id
                or row.matcher_checkpoint_hash != self.matcher_checkpoint_hash
            ):
                raise ValueError("U0 state matcher binding drift")
            if len(row.probe_receipt_hashes) != probe_count:
                raise ValueError("U0 state probe cardinality drift")
        role_by_group: dict[str, U0SplitRoleV1] = {}
        for row in states:
            previous = role_by_group.setdefault(row.group_id, row.split_role)
            if previous is not row.split_role:
                raise ValueError("U0 group leaks across split roles")
        roles = {row.split_role for row in states}
        if roles != set(U0SplitRoleV1):
            raise ValueError("U0 native manifest needs all four split roles")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "probes_per_state", probe_count)
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.manifest_hash and self.manifest_hash != expected:
            raise ValueError("U0 native manifest hash drift")
        object.__setattr__(self, "manifest_hash", expected)

    @property
    def native_state_set_hash(self) -> str:
        return _canonical_sha256([
            row.state_hash for row in sorted(self.states, key=lambda item: item.state_id)
        ])

    @property
    def split_hash(self) -> str:
        return _canonical_sha256({
            "split_id": self.split_id,
            "roles": {
                role.value: _role_projection(self.states, role)
                for role in U0SplitRoleV1
            },
        })

    def role_hash(self, role: U0SplitRoleV1) -> str:
        if not isinstance(role, U0SplitRoleV1):
            raise ValueError("U0 role hash needs a typed split role")
        return _canonical_sha256({
            "split_id": self.split_id,
            "role": role.value,
            "states": _role_projection(self.states, role),
        })

    def role_states(self, role: U0SplitRoleV1) -> tuple[U0NativeStateV1, ...]:
        if not isinstance(role, U0SplitRoleV1):
            raise ValueError("U0 role selection needs a typed split role")
        return tuple(row for row in self.states if row.split_role is role)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": self.schema,
            "manifest_id": self.manifest_id,
            "source_scope": "native_only",
            "split_id": self.split_id,
            "split_hash": self.split_hash,
            "native_state_set_hash": self.native_state_set_hash,
            "matcher_provider_id": self.matcher_provider_id,
            "matcher_checkpoint_hash": self.matcher_checkpoint_hash,
            "teacher_recipe_hash": self.teacher_recipe_hash,
            "teacher_metric": self.teacher_metric.value,
            "probes_per_state": self.probes_per_state,
            "states": [
                row.as_dict() for row in sorted(self.states, key=lambda item: item.state_id)
            ],
        }
        if include_hash:
            result["manifest_hash"] = self.manifest_hash
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "U0NativeManifestV1":
        expected = {
            "schema", "manifest_id", "source_scope", "split_id", "split_hash",
            "native_state_set_hash", "matcher_provider_id",
            "matcher_checkpoint_hash", "teacher_recipe_hash", "teacher_metric",
            "probes_per_state", "states", "manifest_hash",
        }
        _require_exact_fields(value, expected, "U0 native manifest")
        if value["source_scope"] != "native_only":
            raise ValueError("U0 manifest source scope must be native_only")
        states_value = value["states"]
        if not isinstance(states_value, list):
            raise ValueError("U0 manifest states must be a list")
        result = cls(
            schema=value["schema"],
            manifest_id=value["manifest_id"],
            split_id=value["split_id"],
            matcher_provider_id=value["matcher_provider_id"],
            matcher_checkpoint_hash=value["matcher_checkpoint_hash"],
            teacher_recipe_hash=value["teacher_recipe_hash"],
            teacher_metric=U0TeacherMetricV1(value["teacher_metric"]),
            probes_per_state=value["probes_per_state"],
            states=tuple(U0NativeStateV1.from_dict(row) for row in states_value),
            manifest_hash=value["manifest_hash"],
        )
        if (
            value["split_hash"] != result.split_hash
            or value["native_state_set_hash"] != result.native_state_set_hash
        ):
            raise ValueError("U0 manifest derived binding drift")
        return result


@dataclass(frozen=True)
class U0CheckpointReceiptV1:
    checkpoint_id: str
    training_manifest_hash: str
    native_state_set_hash: str
    training_state_count: int
    split_hash: str
    feature_schema_hash: str
    normalizer_hash: str
    teacher_recipe_hash: str
    teacher_metric: U0TeacherMetricV1
    model_architecture_hash: str
    model_state_hash: str
    training_code_hash: str
    claim_semantics: U0ClaimSemanticsV1 = U0ClaimSemanticsV1.SELF_CONSISTENCY_SURROGATE
    schema: str = U0_NATIVE_CHECKPOINT_SCHEMA_V1
    receipt_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if self.schema != U0_NATIVE_CHECKPOINT_SCHEMA_V1:
            raise ValueError("U0 checkpoint receipt schema drift")
        _text(self.checkpoint_id, "U0 checkpoint id")
        for value, name in (
            (self.training_manifest_hash, "training manifest hash"),
            (self.native_state_set_hash, "native state-set hash"),
            (self.split_hash, "split hash"),
            (self.feature_schema_hash, "feature schema hash"),
            (self.normalizer_hash, "normalizer hash"),
            (self.teacher_recipe_hash, "teacher recipe hash"),
            (self.model_architecture_hash, "model architecture hash"),
            (self.model_state_hash, "model state hash"),
            (self.training_code_hash, "training code hash"),
        ):
            _digest(value, name)
        _positive_integer(self.training_state_count, "training state count")
        if not isinstance(self.teacher_metric, U0TeacherMetricV1):
            raise ValueError("checkpoint teacher metric must be typed")
        if self.claim_semantics is not U0ClaimSemanticsV1.SELF_CONSISTENCY_SURROGATE:
            raise ValueError("an uncalibrated U0 checkpoint is only a consistency surrogate")
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.receipt_hash and self.receipt_hash != expected:
            raise ValueError("U0 checkpoint receipt hash drift")
        object.__setattr__(self, "receipt_hash", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": self.schema,
            "checkpoint_id": self.checkpoint_id,
            "training_data_scope": "native_only",
            "training_manifest_hash": self.training_manifest_hash,
            "native_state_set_hash": self.native_state_set_hash,
            "training_state_count": self.training_state_count,
            "split_hash": self.split_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "normalizer_hash": self.normalizer_hash,
            "teacher_recipe_hash": self.teacher_recipe_hash,
            "teacher_metric": self.teacher_metric.value,
            "model_architecture_hash": self.model_architecture_hash,
            "model_state_hash": self.model_state_hash,
            "training_code_hash": self.training_code_hash,
            "claim_semantics": self.claim_semantics.value,
        }
        if include_hash:
            result["receipt_hash"] = self.receipt_hash
        return result

    def checkpoint_binding(self) -> dict[str, object]:
        """Minimal mapping that must be stored inside the model payload."""

        return self.as_dict()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "U0CheckpointReceiptV1":
        expected = {
            "schema", "checkpoint_id", "training_data_scope",
            "training_manifest_hash", "native_state_set_hash",
            "training_state_count", "split_hash", "feature_schema_hash",
            "normalizer_hash", "teacher_recipe_hash", "teacher_metric",
            "model_architecture_hash", "model_state_hash", "training_code_hash",
            "claim_semantics", "receipt_hash",
        }
        _require_exact_fields(value, expected, "U0 checkpoint receipt")
        if value["training_data_scope"] != "native_only":
            raise ValueError("U0 checkpoint training data scope must be native_only")
        return cls(
            schema=value["schema"],
            checkpoint_id=value["checkpoint_id"],
            training_manifest_hash=value["training_manifest_hash"],
            native_state_set_hash=value["native_state_set_hash"],
            training_state_count=value["training_state_count"],
            split_hash=value["split_hash"],
            feature_schema_hash=value["feature_schema_hash"],
            normalizer_hash=value["normalizer_hash"],
            teacher_recipe_hash=value["teacher_recipe_hash"],
            teacher_metric=U0TeacherMetricV1(value["teacher_metric"]),
            model_architecture_hash=value["model_architecture_hash"],
            model_state_hash=value["model_state_hash"],
            training_code_hash=value["training_code_hash"],
            claim_semantics=U0ClaimSemanticsV1(value["claim_semantics"]),
            receipt_hash=value["receipt_hash"],
        )

    @classmethod
    def for_manifest(
        cls,
        manifest: U0NativeManifestV1,
        *,
        checkpoint_id: str,
        feature_schema_hash: str,
        normalizer_hash: str,
        model_architecture_hash: str,
        model_state_hash: str,
        training_code_hash: str,
    ) -> "U0CheckpointReceiptV1":
        if not isinstance(manifest, U0NativeManifestV1):
            raise ValueError("U0 checkpoint needs a typed native manifest")
        return cls(
            checkpoint_id=checkpoint_id,
            training_manifest_hash=manifest.manifest_hash,
            native_state_set_hash=manifest.native_state_set_hash,
            training_state_count=len(manifest.role_states(U0SplitRoleV1.FIT)),
            split_hash=manifest.split_hash,
            feature_schema_hash=feature_schema_hash,
            normalizer_hash=normalizer_hash,
            teacher_recipe_hash=manifest.teacher_recipe_hash,
            teacher_metric=manifest.teacher_metric,
            model_architecture_hash=model_architecture_hash,
            model_state_hash=model_state_hash,
            training_code_hash=training_code_hash,
        )


def verify_u0_native_checkpoint_v1(
    manifest: U0NativeManifestV1,
    receipt: U0CheckpointReceiptV1,
    checkpoint_binding: Mapping[str, object],
) -> None:
    """Verify manifest, receipt, and the binding embedded in a checkpoint.

    Requiring all three prevents a checkpoint trained on a mixed state bank
    from borrowing only the model hash or a human-readable ``native`` label.
    """

    if not isinstance(manifest, U0NativeManifestV1):
        raise ValueError("checkpoint verification needs a typed U0 manifest")
    if not isinstance(receipt, U0CheckpointReceiptV1):
        raise ValueError("checkpoint verification needs a typed U0 receipt")
    expected = {
        "schema", "checkpoint_id", "training_data_scope",
        "training_manifest_hash", "native_state_set_hash",
        "training_state_count", "split_hash", "feature_schema_hash",
        "normalizer_hash", "teacher_recipe_hash", "teacher_metric",
        "model_architecture_hash", "model_state_hash", "training_code_hash",
        "claim_semantics", "receipt_hash",
    }
    _require_exact_fields(checkpoint_binding, expected, "U0 checkpoint binding")
    if dict(checkpoint_binding) != receipt.checkpoint_binding():
        raise ValueError("checkpoint payload does not match its U0 receipt")
    if (
        receipt.training_manifest_hash != manifest.manifest_hash
        or receipt.native_state_set_hash != manifest.native_state_set_hash
        or receipt.training_state_count
        != len(manifest.role_states(U0SplitRoleV1.FIT))
        or receipt.split_hash != manifest.split_hash
        or receipt.teacher_recipe_hash != manifest.teacher_recipe_hash
        or receipt.teacher_metric is not manifest.teacher_metric
    ):
        raise ValueError("checkpoint receipt is not bound to this native U0 manifest")


@dataclass(frozen=True)
class U0CalibrationMetricV1:
    name: str
    value: float
    sample_count: int

    def __post_init__(self) -> None:
        _name(self.name, "calibration metric name")
        _finite(self.value, "calibration metric value")
        _positive_integer(self.sample_count, "calibration metric sample count")

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": float(self.value), "sample_count": self.sample_count}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "U0CalibrationMetricV1":
        _require_exact_fields(value, {"name", "value", "sample_count"}, "U0 calibration metric")
        return cls(name=value["name"], value=value["value"], sample_count=value["sample_count"])


@dataclass(frozen=True)
class U0CalibrationThresholdV1:
    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        _name(self.name, "calibration threshold name")
        _finite(self.value, "calibration threshold value")
        _name(self.unit, "calibration threshold unit")

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": float(self.value), "unit": self.unit}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "U0CalibrationThresholdV1":
        _require_exact_fields(value, {"name", "value", "unit"}, "U0 calibration threshold")
        return cls(name=value["name"], value=value["value"], unit=value["unit"])


@dataclass(frozen=True)
class U0CalibrationReceiptV1:
    calibration_id: str
    checkpoint_receipt_hash: str
    model_state_hash: str
    calibration_split_hash: str
    calibration_state_count: int
    calibration_group_count: int
    calibration_data_hash: str
    calibration_method_hash: str
    support_definition_hash: str
    target_semantics: str
    metrics: tuple[U0CalibrationMetricV1, ...]
    thresholds: tuple[U0CalibrationThresholdV1, ...]
    claim_semantics: U0ClaimSemanticsV1 = U0ClaimSemanticsV1.CALIBRATED_ENDPOINT_ERROR
    schema: str = U0_CALIBRATION_SCHEMA_V1
    receipt_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if self.schema != U0_CALIBRATION_SCHEMA_V1:
            raise ValueError("U0 calibration receipt schema drift")
        _text(self.calibration_id, "U0 calibration id")
        for value, name in (
            (self.checkpoint_receipt_hash, "checkpoint receipt hash"),
            (self.model_state_hash, "calibrated model state hash"),
            (self.calibration_split_hash, "calibration split hash"),
            (self.calibration_data_hash, "calibration data hash"),
            (self.calibration_method_hash, "calibration method hash"),
            (self.support_definition_hash, "calibration support definition hash"),
        ):
            _digest(value, name)
        _positive_integer(self.calibration_state_count, "calibration state count")
        _positive_integer(self.calibration_group_count, "calibration group count")
        target = _name(self.target_semantics, "calibration target semantics")
        metrics = tuple(self.metrics)
        thresholds = tuple(self.thresholds)
        if not metrics or any(not isinstance(row, U0CalibrationMetricV1) for row in metrics):
            raise ValueError("U0 calibration needs typed metrics")
        if not thresholds or any(not isinstance(row, U0CalibrationThresholdV1) for row in thresholds):
            raise ValueError("U0 calibration needs typed thresholds")
        metrics = tuple(sorted(metrics, key=lambda row: row.name))
        thresholds = tuple(sorted(thresholds, key=lambda row: row.name))
        if len({row.name for row in metrics}) != len(metrics):
            raise ValueError("U0 calibration metric names must be unique")
        if len({row.name for row in thresholds}) != len(thresholds):
            raise ValueError("U0 calibration threshold names must be unique")
        if self.claim_semantics is not U0ClaimSemanticsV1.CALIBRATED_ENDPOINT_ERROR:
            raise ValueError("U0 calibration receipt has the wrong claim semantics")
        object.__setattr__(self, "target_semantics", target)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "thresholds", thresholds)
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.receipt_hash and self.receipt_hash != expected:
            raise ValueError("U0 calibration receipt hash drift")
        object.__setattr__(self, "receipt_hash", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": self.schema,
            "calibration_id": self.calibration_id,
            "checkpoint_receipt_hash": self.checkpoint_receipt_hash,
            "model_state_hash": self.model_state_hash,
            "calibration_split_hash": self.calibration_split_hash,
            "calibration_state_count": self.calibration_state_count,
            "calibration_group_count": self.calibration_group_count,
            "calibration_data_hash": self.calibration_data_hash,
            "calibration_method_hash": self.calibration_method_hash,
            "support_definition_hash": self.support_definition_hash,
            "target_semantics": self.target_semantics,
            "metrics": [row.as_dict() for row in sorted(self.metrics, key=lambda item: item.name)],
            "thresholds": [
                row.as_dict() for row in sorted(self.thresholds, key=lambda item: item.name)
            ],
            "claim_semantics": self.claim_semantics.value,
        }
        if include_hash:
            result["receipt_hash"] = self.receipt_hash
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "U0CalibrationReceiptV1":
        expected = {
            "schema", "calibration_id", "checkpoint_receipt_hash",
            "model_state_hash", "calibration_split_hash",
            "calibration_state_count", "calibration_group_count",
            "calibration_data_hash", "calibration_method_hash",
            "support_definition_hash", "target_semantics", "metrics",
            "thresholds", "claim_semantics", "receipt_hash",
        }
        _require_exact_fields(value, expected, "U0 calibration receipt")
        metrics = value["metrics"]
        thresholds = value["thresholds"]
        if not isinstance(metrics, list) or not isinstance(thresholds, list):
            raise ValueError("U0 calibration metrics and thresholds must be lists")
        return cls(
            schema=value["schema"],
            calibration_id=value["calibration_id"],
            checkpoint_receipt_hash=value["checkpoint_receipt_hash"],
            model_state_hash=value["model_state_hash"],
            calibration_split_hash=value["calibration_split_hash"],
            calibration_state_count=value["calibration_state_count"],
            calibration_group_count=value["calibration_group_count"],
            calibration_data_hash=value["calibration_data_hash"],
            calibration_method_hash=value["calibration_method_hash"],
            support_definition_hash=value["support_definition_hash"],
            target_semantics=value["target_semantics"],
            metrics=tuple(U0CalibrationMetricV1.from_dict(row) for row in metrics),
            thresholds=tuple(U0CalibrationThresholdV1.from_dict(row) for row in thresholds),
            claim_semantics=U0ClaimSemanticsV1(value["claim_semantics"]),
            receipt_hash=value["receipt_hash"],
        )

    @classmethod
    def for_checkpoint(
        cls,
        manifest: U0NativeManifestV1,
        checkpoint: U0CheckpointReceiptV1,
        *,
        calibration_id: str,
        calibration_data_hash: str,
        calibration_method_hash: str,
        support_definition_hash: str,
        target_semantics: str,
        metrics: Sequence[U0CalibrationMetricV1],
        thresholds: Sequence[U0CalibrationThresholdV1],
    ) -> "U0CalibrationReceiptV1":
        verify_u0_native_checkpoint_v1(
            manifest, checkpoint, checkpoint.checkpoint_binding(),
        )
        calibration_states = manifest.role_states(U0SplitRoleV1.CALIBRATION)
        return cls(
            calibration_id=calibration_id,
            checkpoint_receipt_hash=checkpoint.receipt_hash,
            model_state_hash=checkpoint.model_state_hash,
            calibration_split_hash=manifest.role_hash(U0SplitRoleV1.CALIBRATION),
            calibration_state_count=len(calibration_states),
            calibration_group_count=len({row.group_id for row in calibration_states}),
            calibration_data_hash=calibration_data_hash,
            calibration_method_hash=calibration_method_hash,
            support_definition_hash=support_definition_hash,
            target_semantics=target_semantics,
            metrics=tuple(metrics),
            thresholds=tuple(thresholds),
        )


def verify_u0_calibration_v1(
    manifest: U0NativeManifestV1,
    checkpoint: U0CheckpointReceiptV1,
    calibration: U0CalibrationReceiptV1,
) -> None:
    verify_u0_native_checkpoint_v1(
        manifest, checkpoint, checkpoint.checkpoint_binding(),
    )
    if not isinstance(calibration, U0CalibrationReceiptV1):
        raise ValueError("calibration verification needs a typed receipt")
    states = manifest.role_states(U0SplitRoleV1.CALIBRATION)
    if (
        calibration.checkpoint_receipt_hash != checkpoint.receipt_hash
        or calibration.model_state_hash != checkpoint.model_state_hash
        or calibration.calibration_split_hash
        != manifest.role_hash(U0SplitRoleV1.CALIBRATION)
        or calibration.calibration_state_count != len(states)
        or calibration.calibration_group_count != len({row.group_id for row in states})
    ):
        raise ValueError("U0 calibration is not bound to checkpoint and calibration split")


__all__ = [
    "U0_CALIBRATION_SCHEMA_V1",
    "U0_NATIVE_CHECKPOINT_SCHEMA_V1",
    "U0_NATIVE_TRAINING_SCHEMA_V1",
    "U0CalibrationMetricV1",
    "U0CalibrationReceiptV1",
    "U0CalibrationThresholdV1",
    "U0CheckpointReceiptV1",
    "U0ClaimSemanticsV1",
    "U0NativeManifestV1",
    "U0NativeStateV1",
    "U0SplitRoleV1",
    "U0TeacherMetricV1",
    "reject_action_control_fields_v1",
    "verify_u0_calibration_v1",
    "verify_u0_native_checkpoint_v1",
]
