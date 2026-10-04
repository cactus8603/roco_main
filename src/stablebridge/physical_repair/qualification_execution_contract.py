"""Fail-closed chronology and provenance for action qualification.

The low-level qualification matrix intentionally accepts only three arrays.
That narrow surface prevents direct metadata leakage, but an ``HxWx2`` array
does not say whether it came from the native matcher, clean-pair inference, or
task ground truth.  Scientific execution therefore has to cross this wrapper.

The wrapper binds the exact controlled-corruption RGB pair to a completed
native matcher forward on that same pair.  Only the resulting native predicted
flow may be passed to action discovery.  Task ground truth is neither an input
nor an allowed flow role and remains sealed until all action decisions and
matcher outputs have been committed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

import numpy as np

from .qualification_matrix import (
    QualificationMatrixResultV1,
    build_uncapped_qualification_matrix,
    canonical_sha256,
)


QUALIFICATION_NATIVE_MATCHER_RECEIPT_SCHEMA_V1 = (
    "stablebridge-qualification-native-matcher-receipt/v1"
)
QUALIFICATION_OBSERVATION_BINDING_SCHEMA_V1 = (
    "stablebridge-qualification-observation-binding/v1"
)
NATIVE_FLOW_ROLE_V1 = (
    "NATIVE_MATCHER_PREDICTION_ON_SAME_CONTROLLED_CORRUPTED_PAIR"
)
NATIVE_STAGE_ORDER_V1 = (
    "CONTROLLED_CORRUPTION_INPUTS_SEALED",
    "NATIVE_MATCHER_STARTED",
    "NATIVE_MATCHER_OUTPUT_SEALED",
)
NEXT_STAGE_V1 = "BOUND_UNCAPPED_ACTION_DISCOVERY"

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/=-]{0,255}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _array_sha256(value: np.ndarray) -> str:
    """Match the qualification-matrix array identity exactly."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_arrays(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(first)
    second = np.asarray(second)
    raw_flow = np.asarray(flow)
    if (
        first.dtype != np.uint8
        or second.dtype != np.uint8
        or first.ndim != 3
        or first.shape != second.shape
        or first.shape[2] != 3
        or raw_flow.dtype != np.float32
        or raw_flow.shape != (*first.shape[:2], 2)
        or not np.isfinite(raw_flow).all()
    ):
        raise ValueError(
            "expected matching uint8 RGB corrupted inputs and finite float32 "
            "HxWx2 native matcher flow"
        )
    return (
        np.ascontiguousarray(first),
        np.ascontiguousarray(second),
        np.ascontiguousarray(raw_flow),
    )


def _require_id(value: str, name: str) -> str:
    value = str(value)
    if _ID_RE.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")
    return value


def _require_sha256(value: str, name: str) -> str:
    value = str(value)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")
    return value


def _without_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if key != "receipt_sha256"}


def build_native_matcher_qualification_receipt_v1(
    *,
    case_id: str,
    matcher_id: str,
    corrupted_first_rgb: np.ndarray,
    corrupted_second_rgb: np.ndarray,
    native_predicted_flow: np.ndarray,
    corruption_receipt_sha256: str,
    matcher_execution_receipt_sha256: str,
) -> dict[str, Any]:
    """Seal the only flow provenance accepted by scientific qualification.

    ``matcher_execution_receipt_sha256`` binds the backend-specific execution
    receipt; this semantic receipt adds the cross-stage role and chronology.
    It is deliberately impossible to declare GT, outcome, clean-pair flow, or
    action-arm output as a tolerated source.
    """

    first, second, flow = _validate_arrays(
        corrupted_first_rgb, corrupted_second_rgb, native_predicted_flow,
    )
    value: dict[str, Any] = {
        "schema": QUALIFICATION_NATIVE_MATCHER_RECEIPT_SCHEMA_V1,
        "case_id": _require_id(case_id, "case_id"),
        "status": "COMPLETE",
        "matcher_id": _require_id(matcher_id, "matcher_id"),
        "corruption_receipt_sha256": _require_sha256(
            corruption_receipt_sha256, "corruption_receipt_sha256",
        ),
        "matcher_execution_receipt_sha256": _require_sha256(
            matcher_execution_receipt_sha256,
            "matcher_execution_receipt_sha256",
        ),
        "inputs": {
            "first": {
                "semantic_role": "CONTROLLED_CORRUPTED_FIRST_RGB",
                "array_sha256": _array_sha256(first),
            },
            "second": {
                "semantic_role": "CONTROLLED_CORRUPTED_SECOND_RGB",
                "array_sha256": _array_sha256(second),
            },
        },
        "outputs": {
            "flow": {
                "semantic_role": NATIVE_FLOW_ROLE_V1,
                "array_sha256": _array_sha256(flow),
            },
        },
        "chronology": {
            "completed_stages": list(NATIVE_STAGE_ORDER_V1),
            "next_required_stage": NEXT_STAGE_V1,
            "action_discovery_started": False,
            "action_matcher_started": False,
            "task_ground_truth_decoded": False,
            "task_ground_truth_joined": False,
        },
        "matcher_array_inputs": [
            "controlled_corrupted_first_rgb",
            "controlled_corrupted_second_rgb",
        ],
        "ground_truth_fields_read": [],
        "outcome_fields_read": [],
        "corruption_label_fields_read": [],
        "action_candidate_fields_read": [],
        "clean_pair_flow_read": False,
        "scientific_qualification": False,
        "selector_admission": False,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def validate_native_matcher_qualification_receipt_v1(
    receipt: Mapping[str, Any],
    *,
    case_id: str,
    corrupted_first_rgb: np.ndarray,
    corrupted_second_rgb: np.ndarray,
    native_predicted_flow: np.ndarray,
) -> dict[str, Any]:
    """Validate exact role, chronology, self-hash, and array bindings."""

    first, second, flow = _validate_arrays(
        corrupted_first_rgb, corrupted_second_rgb, native_predicted_flow,
    )
    value = dict(receipt)
    if value.get("schema") != QUALIFICATION_NATIVE_MATCHER_RECEIPT_SCHEMA_V1:
        raise RuntimeError("qualification native matcher receipt schema drift")
    if value.get("status") != "COMPLETE":
        raise RuntimeError("native matcher output is not complete")
    if value.get("case_id") != _require_id(case_id, "case_id"):
        raise RuntimeError("native matcher receipt case binding drift")
    _require_id(str(value.get("matcher_id", "")), "matcher_id")
    _require_sha256(
        str(value.get("corruption_receipt_sha256", "")),
        "corruption_receipt_sha256",
    )
    _require_sha256(
        str(value.get("matcher_execution_receipt_sha256", "")),
        "matcher_execution_receipt_sha256",
    )
    declared_hash = _require_sha256(
        str(value.get("receipt_sha256", "")), "receipt_sha256",
    )
    if declared_hash != canonical_sha256(_without_self_hash(value)):
        raise RuntimeError("qualification native matcher receipt hash drift")

    inputs = value.get("inputs")
    outputs = value.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise RuntimeError("native matcher receipt omitted input/output bindings")
    expected_inputs = {
        "first": ("CONTROLLED_CORRUPTED_FIRST_RGB", _array_sha256(first)),
        "second": ("CONTROLLED_CORRUPTED_SECOND_RGB", _array_sha256(second)),
    }
    if set(inputs) != set(expected_inputs):
        raise RuntimeError("native matcher receipt input role universe drift")
    for endpoint, (role, digest) in expected_inputs.items():
        observed = inputs[endpoint]
        if (
            not isinstance(observed, Mapping)
            or observed.get("semantic_role") != role
            or observed.get("array_sha256") != digest
        ):
            raise RuntimeError(f"native matcher {endpoint} input binding drift")
    if set(outputs) != {"flow"} or not isinstance(outputs["flow"], Mapping):
        raise RuntimeError("native matcher output role universe drift")
    if outputs["flow"].get("semantic_role") != NATIVE_FLOW_ROLE_V1:
        raise RuntimeError("qualification rejects non-native or GT flow role")
    if outputs["flow"].get("array_sha256") != _array_sha256(flow):
        raise RuntimeError("native matcher flow array binding drift")

    chronology = value.get("chronology")
    if not isinstance(chronology, Mapping):
        raise RuntimeError("native matcher chronology missing")
    expected_chronology = {
        "completed_stages": list(NATIVE_STAGE_ORDER_V1),
        "next_required_stage": NEXT_STAGE_V1,
        "action_discovery_started": False,
        "action_matcher_started": False,
        "task_ground_truth_decoded": False,
        "task_ground_truth_joined": False,
    }
    if dict(chronology) != expected_chronology:
        raise RuntimeError("native matcher qualification chronology drift")
    exact_empty = (
        "ground_truth_fields_read",
        "outcome_fields_read",
        "corruption_label_fields_read",
        "action_candidate_fields_read",
    )
    if any(value.get(name) != [] for name in exact_empty):
        raise RuntimeError("forbidden evidence reached native matcher qualification")
    if value.get("clean_pair_flow_read") is not False:
        raise RuntimeError("clean-pair flow cannot stand in for corrupted-pair prediction")
    if value.get("matcher_array_inputs") != [
        "controlled_corrupted_first_rgb",
        "controlled_corrupted_second_rgb",
    ]:
        raise RuntimeError("native matcher received a non-image or unbound input")
    return value


@dataclass(frozen=True)
class BoundQualificationMatrixResultV1:
    """Low-level matrix plus its independently auditable observation binding."""

    matrix: QualificationMatrixResultV1
    binding_receipt: Mapping[str, Any]


def build_bound_uncapped_qualification_matrix_v1(
    *,
    case_id: str,
    corrupted_first_rgb: np.ndarray,
    corrupted_second_rgb: np.ndarray,
    native_predicted_flow: np.ndarray,
    native_matcher_receipt: Mapping[str, Any],
) -> BoundQualificationMatrixResultV1:
    """Run discovery only after validating same-pair native-flow provenance."""

    first, second, flow = _validate_arrays(
        corrupted_first_rgb, corrupted_second_rgb, native_predicted_flow,
    )
    native_receipt = validate_native_matcher_qualification_receipt_v1(
        native_matcher_receipt,
        case_id=case_id,
        corrupted_first_rgb=first,
        corrupted_second_rgb=second,
        native_predicted_flow=flow,
    )
    matrix = build_uncapped_qualification_matrix(first, second, flow)
    matrix_receipt = dict(matrix.receipt)
    native_reference = matrix_receipt.get("native_reference")
    expected_reference = {
        "first_sha256": _array_sha256(first),
        "second_sha256": _array_sha256(second),
        "native_flow_sha256": _array_sha256(flow),
    }
    if native_reference != expected_reference:
        raise RuntimeError("qualification matrix did not preserve native observation binding")
    matrix_hash = str(matrix_receipt.get("receipt_sha256", ""))
    _require_sha256(matrix_hash, "qualification_matrix_receipt_sha256")
    binding: dict[str, Any] = {
        "schema": QUALIFICATION_OBSERVATION_BINDING_SCHEMA_V1,
        "case_id": _require_id(case_id, "case_id"),
        "flow_semantic_role": NATIVE_FLOW_ROLE_V1,
        "corruption_receipt_sha256": native_receipt[
            "corruption_receipt_sha256"
        ],
        "native_matcher_receipt_sha256": native_receipt["receipt_sha256"],
        "native_matcher_execution_receipt_sha256": native_receipt[
            "matcher_execution_receipt_sha256"
        ],
        "qualification_matrix_receipt_sha256": matrix_hash,
        "array_bindings": expected_reference,
        "chronology": [
            *NATIVE_STAGE_ORDER_V1,
            "BOUND_UNCAPPED_ACTION_DISCOVERY_COMPLETE",
        ],
        "ground_truth_fields_read": [],
        "outcome_fields_read": [],
        "task_ground_truth_decoded": False,
        "task_ground_truth_joined": False,
        "scientific_qualification": False,
        "selector_admission": False,
    }
    binding["receipt_sha256"] = canonical_sha256(binding)
    return BoundQualificationMatrixResultV1(
        matrix=matrix,
        binding_receipt=binding,
    )


__all__ = [
    "BoundQualificationMatrixResultV1",
    "NATIVE_FLOW_ROLE_V1",
    "NATIVE_STAGE_ORDER_V1",
    "NEXT_STAGE_V1",
    "QUALIFICATION_NATIVE_MATCHER_RECEIPT_SCHEMA_V1",
    "QUALIFICATION_OBSERVATION_BINDING_SCHEMA_V1",
    "build_bound_uncapped_qualification_matrix_v1",
    "build_native_matcher_qualification_receipt_v1",
    "validate_native_matcher_qualification_receipt_v1",
]
