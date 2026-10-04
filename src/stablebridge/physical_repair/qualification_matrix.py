"""Outcome-blind, uncapped execution surface for action qualification.

The selector-facing Work-E helper intentionally ranks and caps non-native
controls.  Scientific qualification has a different estimand: every one of
the five E249 policies must receive exactly one typed row on every case, and
every eligible policy must be materialized at the descriptor-frozen strength
of one.  This module supplies that narrow surface without reading outcomes,
ground truth, dataset identity, paths, or corruption labels.

This is an execution receipt, not scientific qualification or selector
admission.  Cost, local-compositor, mechanism, harm, and CC/RR/CR/RC receipts
remain separate mandatory gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .local_case_materialization import (
    LocalCaseMaterializationReceiptV1,
    build_local_case_materialization_v1,
)
from .local_wiener_successor import (
    LOCAL_WIENER_ACTION_ID_V2,
    LOCAL_WIENER_OPERATOR_ID_V2,
    build_local_wiener_successor_v2,
)
from .local_blur_successor import (
    LOCAL_BLUR_ACTION_ID_V2,
    LOCAL_BLUR_OPERATOR_ID_V2,
    build_local_blur_successor_v2,
)
from .local_jpeg_successors import (
    LOCAL_JPEG_CODEC_ACTION_ID_V2,
    LOCAL_JPEG_CODEC_OPERATOR_ID_V2,
    LOCAL_JPEG_QCELL_ACTION_ID_V2,
    LOCAL_JPEG_QCELL_OPERATOR_ID_V2,
    build_local_jpeg_successor_v2,
)
from .blur_mechanism_evaluators import (
    BlurMechanismEvaluationReceiptV1,
    evaluate_blur_mechanism_v1,
)
from .jpeg_mechanism_evaluators import (
    JPEGMechanismEvaluationReceiptV2,
    evaluate_jpeg_mechanism_v2,
)
from .noise_mechanism_evaluators import (
    NoiseMechanismEvaluationReceiptV1,
    evaluate_noise_mechanism_v1,
)


QUALIFICATION_MATRIX_SCHEMA_V1 = "stablebridge-uncapped-qualification-matrix/v1"
QUALIFICATION_POLICY_ROW_SCHEMA_V1 = "stablebridge-qualification-policy-execution/v1"
QUALIFICATION_STRENGTH_V1 = 1.0

_ROOT = Path(__file__).resolve().parents[3]
_WORK_E = _ROOT / "research/region_aware_action_learning_20261001/work_package_E"
_EXECUTOR = _WORK_E / "executor_adapter.py"
_REGISTRY_CODE = _WORK_E / "choice_registry.py"
_REGISTRY = _WORK_E / "choice_family_registry.json"
_E249_BANK = _ROOT / "experiments/E249_legacy7_endpoint_mechanism_freeze_v1/QUALIFICATION_ACTION_BANK.json"
_E249_FREEZE = _ROOT / "experiments/E249_legacy7_endpoint_mechanism_freeze_v1/QUALIFICATION_FREEZE.json"
_E250_BANK = _ROOT / "experiments/E250_action_bank_v1_freeze/ACTION_BANK_V1.json"
_E250_SEAL = _ROOT / "experiments/E250_action_bank_v1_freeze/PACKAGE_SEAL.json"
_E251_DESCRIPTOR = _ROOT / "experiments/E251_local_wiener_successor_v2/ACTION_DESCRIPTOR.json"
_E251_FREEZE = _ROOT / "experiments/E251_local_wiener_successor_v2/SUCCESSOR_FREEZE.json"
_E253_DESCRIPTOR = _ROOT / "experiments/E253_local_blur_successor_v2/ACTION_DESCRIPTOR.json"
_E253_FREEZE = _ROOT / "experiments/E253_local_blur_successor_v2/SUCCESSOR_FREEZE.json"
_E255_DESCRIPTORS = _ROOT / "experiments/E255_local_jpeg_successors_v2/ACTION_DESCRIPTORS.json"
_E255_FREEZE = _ROOT / "experiments/E255_local_jpeg_successors_v2/SUCCESSOR_FREEZE.json"
_LOCAL_WIENER = _ROOT / "src/stablebridge/physical_repair/local_wiener_successor.py"
_LOCAL_BLUR = _ROOT / "src/stablebridge/physical_repair/local_blur_successor.py"
_LOCAL_JPEG = _ROOT / "src/stablebridge/physical_repair/local_jpeg_successors.py"
_LOCAL_CASE = _ROOT / "src/stablebridge/physical_repair/local_case_materialization.py"
_NOISE_MECHANISM = _ROOT / "src/stablebridge/physical_repair/noise_mechanism_evaluators.py"
_BLUR_MECHANISM = _ROOT / "src/stablebridge/physical_repair/blur_mechanism_evaluators.py"
_JPEG_MECHANISM = _ROOT / "src/stablebridge/physical_repair/jpeg_mechanism_evaluators.py"
_STABLEBRIDGE_INIT = _ROOT / "src/stablebridge/__init__.py"
_PHYSICAL_REPAIR_INIT = _ROOT / "src/stablebridge/physical_repair/__init__.py"
_SOURCE_SHA256 = {
    _EXECUTOR: "28b7742ad7a9baf44f7373fc5bc108d60643f8a25980dc258c42df6529e50a1e",
    _REGISTRY_CODE: "0eaed9a2a169e50333083c5a896a6fc151614779e7f4cb167c078da2992e0d0f",
    _REGISTRY: "bb06bf779ade835722b29f60c7c764c421dead0e794c2a935aa47d9c6e5102fd",
    _E249_BANK: "27a4e3e81cdf863bec1040e0f058c61adffe4932c52e0a1627b92c16d73b4893",
    _E249_FREEZE: "2f2549b39b50f84f0d086a2e43bb8c1e85ed283d70de49412b1db70d2e8eba3d",
    _E250_BANK: "c7440bc7cc715fa35f3c718e1293a623008c6e253bdecb4375c47b918a348e09",
    _E250_SEAL: "98081965539b2830e98fdfed834292fcc0547996fe4c50303ea349f6ad9b231e",
    _E251_DESCRIPTOR: "99f29c23ec392d47dc62ad2e7fef2216d8b73201b009cae3ea4316471a0e70f6",
    _E251_FREEZE: "8fa933aefdb2ee12a4036ea2e67b1fd3ca47332884122efe44a1561035d77c1f",
    _E253_DESCRIPTOR: "3e8d93c456878360a3de2584d8e5f812274ddfddc4e34605c80a5aca0eec3b95",
    _E253_FREEZE: "4cb2c9eceef5bc8fc9f84592f6743788e8939ccd1e484ca05aaec4af3fcce42a",
    _E255_DESCRIPTORS: "41d0a6abb5266c1418fd202fb8d10b6624f34f0bcfb3b55a70d103e3dd09a6d6",
    _E255_FREEZE: "3952233efd0e13e82af77f1358322d548c5444ee6494537e6a77c3e595138a3d",
    _LOCAL_WIENER: "11e23d5449c66629e31fa665e704e860831fe6a0092623937ae4836cd05ec673",
    _LOCAL_BLUR: "a95bd3716ca63d0ecda668faf0eb8025c0d7db3e59f8575438a96b68d19380b9",
    _LOCAL_JPEG: "ba675f0e5e557d3104c41f86c22616c0113f88e0837ba11a35193eb8eb3c9505",
    _LOCAL_CASE: "1046abd43afc56dba15cc0895d281251e353916cbb8bbc4c23205e32270ae3fa",
    _NOISE_MECHANISM: "2053388f49d879a8b98de81de63cb627f452a7eb8c039b35b7ca2a71fa1e5b99",
    _BLUR_MECHANISM: "3ae770a47b84ef36be1d85e67efeda18f94073d5888875f9e418215f63ac7361",
    _JPEG_MECHANISM: "ccd2ad7cd75fa7c734dd61bbabe784cfc9c71020cb68cc2bd4a50e968117a370",
    _STABLEBRIDGE_INIT: "468b8f6c6a8d0f94320b9e18908b2eed6a1bb368866eeb96d0ffe0e85e07b298",
    _PHYSICAL_REPAIR_INIT: "9be0fd9172a33dd4a2cd11c02d5d00bac7ac19f21980f18dc3912e336301732f",
}

_POLICIES = (
    (
        "paired_impulse_median3",
        "paired_impulse_median3.endpoint_supported_v1",
        "case_selected_supported",
    ),
    (
        LOCAL_WIENER_ACTION_ID_V2,
        LOCAL_WIENER_ACTION_ID_V2,
        "case_selected_supported",
    ),
    (
        LOCAL_BLUR_ACTION_ID_V2,
        LOCAL_BLUR_ACTION_ID_V2,
        "case_selected_single_supported",
    ),
    (
        LOCAL_JPEG_QCELL_ACTION_ID_V2,
        LOCAL_JPEG_QCELL_ACTION_ID_V2,
        "case_selected_supported",
    ),
    (
        LOCAL_JPEG_CODEC_ACTION_ID_V2,
        LOCAL_JPEG_CODEC_ACTION_ID_V2,
        "case_selected_supported",
    ),
)

_DESCRIPTOR_SHA256 = {
    "paired_impulse_median3.endpoint_supported_v1": "d75a7464fe8cdc370113af17dc5f157dc77c0699126edc471d96969a8b70ceab",
    LOCAL_WIENER_ACTION_ID_V2: "302021e6a55655ef0f48a8673a1df4f74b4cde6d8f153f88b1d144aec1c6aaeb",
    LOCAL_BLUR_ACTION_ID_V2: "713660cbefe238381fd4f1a934c7ce11cd29c07c21f22b9ff218fd04fe8ebffa",
    LOCAL_JPEG_QCELL_ACTION_ID_V2: "8a825bff93fcc3fce4fb73517ef7cd094fdff0ced099b5bcdf7d71c26fb6ca1e",
    LOCAL_JPEG_CODEC_ACTION_ID_V2: "c9b46bd31fc3d61076eca43e44a3cb5d0e6c54e179894a6595bd57433aa3ed60",
}

_ENDPOINT_POLICY = {
    "id": "case_selected_supported_before_only_v2",
    "allowed_runtime_inputs": [
        "observed_first_rgb",
        "observed_second_rgb",
        "observed_native_flow",
    ],
    "case_selected_supported": ["first", "second", "both"],
    "case_selected_single_supported": ["first", "second"],
    "strength": QUALIFICATION_STRENGTH_V1,
}


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        raise TypeError("arrays must be content-addressed before receipt serialization")
    if hasattr(value, "item"):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite receipt value")
    return value


def _stable_physical_receipts(value: Any) -> Any:
    """Remove replay-volatile probe timing from semantic identity.

    Runtime remains mandatory, but belongs in a separate typed cost receipt.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _stable_physical_receipts(item)
            for key, item in value.items()
            if str(key) != "measured_probe_seconds"
        }
    if isinstance(value, (tuple, list)):
        return [_stable_physical_receipts(item) for item in value]
    return _clean(value)


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        _clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


ENDPOINT_POLICY_SHA256_V1 = canonical_sha256(_ENDPOINT_POLICY)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _physical_pair_sha256(
    first: np.ndarray, second: np.ndarray, flow: np.ndarray,
) -> str:
    return hashlib.sha256(
        (
            "choice-physical-pair/v1::"
            + "::".join(
                [_array_sha256(first), _array_sha256(second), _array_sha256(flow)]
            )
        ).encode("ascii")
    ).hexdigest()


def _verify_frozen_sources() -> None:
    for path, expected in _SOURCE_SHA256.items():
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(_ROOT.resolve()):
            raise RuntimeError(f"qualification source escaped workspace: {path}")
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"qualification source drift for {path.relative_to(_ROOT)}: "
                f"expected {expected}, got {actual}"
            )


def _work_e_backend():
    _verify_frozen_sources()
    work_e = str(_WORK_E)
    if work_e not in sys.path:
        sys.path.insert(0, work_e)
    backend = importlib.import_module("executor_adapter")
    if Path(backend.__file__).resolve() != _EXECUTOR.resolve():
        raise RuntimeError("preloaded executor_adapter came from an unexpected source")
    registry = importlib.import_module("choice_registry")
    if Path(registry.__file__).resolve() != _REGISTRY_CODE.resolve():
        raise RuntimeError("preloaded choice_registry came from an unexpected source")
    return backend


def _validate_inputs(
    first: np.ndarray, second: np.ndarray, flow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(first)
    second = np.asarray(second)
    flow = np.asarray(flow, dtype=np.float32)
    if (
        first.dtype != np.uint8
        or second.dtype != np.uint8
        or first.ndim != 3
        or first.shape != second.shape
        or first.shape[2] != 3
        or flow.shape != (*first.shape[:2], 2)
        or not np.isfinite(flow).all()
    ):
        raise ValueError("expected matching uint8 RGB images and finite HxWx2 native flow")
    return (
        np.ascontiguousarray(first),
        np.ascontiguousarray(second),
        np.ascontiguousarray(flow),
    )


@dataclass(frozen=True)
class MaterializedQualificationArmV1:
    """One eligible full-strength arm; arrays are not embedded in receipts."""

    qualification_action_id: str
    exact_control_id: str
    first_rgb: np.ndarray
    second_rgb: np.ndarray
    first_support: np.ndarray
    second_support: np.ndarray


@dataclass(frozen=True)
class QualificationMatrixResultV1:
    """Complete five-row receipt plus all and only eligible materializations."""

    receipt: Mapping[str, Any]
    arms: tuple[MaterializedQualificationArmV1, ...]
    local_case_receipts: tuple[LocalCaseMaterializationReceiptV1, ...]
    mechanism_receipts: tuple[
        NoiseMechanismEvaluationReceiptV1
        | BlurMechanismEvaluationReceiptV1
        | JPEGMechanismEvaluationReceiptV2,
        ...,
    ]


def _control_receipt(control: Any) -> dict[str, Any]:
    payload = {
        "control_id": str(control.control_id),
        "family": str(control.family),
        "operator": str(control.operator),
        "authorization": str(control.authorization),
        "execution_status": str(control.execution_status),
        "applicability": str(control.applicability),
        "modified_endpoints": list(control.modified_endpoints),
        "parameters": _clean(control.parameters),
        "physical_receipts": _stable_physical_receipts(control.receipts),
        "observable_score": control.observable_score,
        "diagnostics": _clean(control.diagnostics),
    }
    forbidden = {
        "gain", "task_gain", "outcome", "ground_truth", "corruption_label",
        "dataset", "path", "scene_identity", "severity",
    }
    if forbidden & set(payload):  # defensive: payload keys are fixed above
        raise AssertionError("forbidden qualification receipt field")
    return payload


def _exact_endpoint(modified: Sequence[str], endpoint_type: str) -> str:
    endpoints = tuple(str(value) for value in modified)
    if not endpoints or len(endpoints) != len(set(endpoints)):
        raise RuntimeError("eligible control needs unique modified endpoints")
    if any(value not in {"first", "second"} for value in endpoints):
        raise RuntimeError("unknown modified endpoint")
    if endpoint_type == "case_selected_single_supported":
        if len(endpoints) != 1:
            raise RuntimeError("single-supported policy resolved to multiple endpoints")
        return endpoints[0]
    if endpoint_type != "case_selected_supported":
        raise RuntimeError("unknown endpoint policy type")
    return endpoints[0] if len(endpoints) == 1 else "both"


def _blur_branch_receipt(control: Any, exact_endpoint: str) -> dict[str, Any]:
    parameters = dict(control.parameters)
    required = {
        "winner", "family", "selected_parameter",
        "identified_endpoint", "modified_endpoint",
    }
    if not required.issubset(parameters):
        raise RuntimeError("blur control is missing a routed-branch field")
    winner = str(parameters["winner"])
    if "@" not in winner:
        raise RuntimeError("blur winner/operator binding drift")
    if str(control.control_id) == LOCAL_BLUR_ACTION_ID_V2:
        if str(control.operator) != LOCAL_BLUR_OPERATOR_ID_V2:
            raise RuntimeError("local blur successor operator drift")
    elif winner != str(control.operator):
        raise RuntimeError("legacy blur winner/operator binding drift")
    family_id, identified_from_winner = winner.rsplit("@", 1)
    family = str(parameters["family"])
    if family_id != f"common_{family}" or family not in {"disk", "gaussian", "motion"}:
        raise RuntimeError("blur family/winner binding drift")
    identified = str(parameters["identified_endpoint"])
    modified = str(parameters["modified_endpoint"])
    if identified != identified_from_winner or identified not in {"first", "second"}:
        raise RuntimeError("blur identified endpoint drift")
    expected_modified = "second" if identified == "first" else "first"
    if modified != expected_modified or modified != exact_endpoint:
        raise RuntimeError("blur must modify exactly the opposite endpoint")
    return {
        "exact_control_id": str(control.control_id),
        "winner_id": winner,
        "family": family,
        "selected_parameter_sha256": canonical_sha256(parameters["selected_parameter"]),
        "identified_endpoint": identified,
        "modified_endpoint": modified,
        "endpoint_policy_sha256": ENDPOINT_POLICY_SHA256_V1,
    }


def _eligible_row(
    *,
    control: Any,
    action_id: str,
    endpoint_type: str,
    pair: tuple[np.ndarray, np.ndarray],
    flow: np.ndarray,
    physical_pair_sha256: str,
    materialize: Callable[[Any, np.ndarray, np.ndarray], Mapping[float, tuple[np.ndarray, np.ndarray]]],
) -> tuple[
    dict[str, Any],
    MaterializedQualificationArmV1,
    LocalCaseMaterializationReceiptV1 | None,
    tuple[
        NoiseMechanismEvaluationReceiptV1
        | BlurMechanismEvaluationReceiptV1
        | JPEGMechanismEvaluationReceiptV2,
        ...,
    ],
]:
    exact_endpoint = _exact_endpoint(control.modified_endpoints, endpoint_type)
    if control.proposal is None or control.supports is None:
        raise RuntimeError("eligible control omitted proposal/support")
    if len(control.proposal) != 2 or len(control.supports) != 2:
        raise RuntimeError("eligible control needs exactly two proposals and supports")
    proposal_hashes_before = tuple(_array_sha256(value) for value in control.proposal)
    support_hashes_before = tuple(_array_sha256(value) for value in control.supports)
    materialized = materialize(control, pair[0], pair[1])
    if set(materialized) != {0.5, QUALIFICATION_STRENGTH_V1}:
        raise RuntimeError("legacy materializer strength set drifted")
    if proposal_hashes_before != tuple(_array_sha256(value) for value in control.proposal):
        raise RuntimeError("materializer mutated the frozen proposal")
    if support_hashes_before != tuple(_array_sha256(value) for value in control.supports):
        raise RuntimeError("materializer mutated the frozen support")
    outputs = tuple(np.ascontiguousarray(value) for value in materialized[QUALIFICATION_STRENGTH_V1])
    if len(outputs) != 2 or any(value.dtype != np.uint8 for value in outputs):
        raise RuntimeError("materializer returned an invalid image pair")
    proposals = tuple(np.ascontiguousarray(value) for value in control.proposal)
    raw_supports = tuple(np.asarray(value) for value in control.supports)
    if any(value.dtype != np.bool_ for value in raw_supports):
        raise RuntimeError("control supports must be boolean")
    supports = tuple(np.ascontiguousarray(value) for value in raw_supports)
    endpoint_rows = []
    changed_total = 0
    names = ("first", "second")
    for name, observed, proposal, output, support in zip(
        names, pair, proposals, outputs, supports,
    ):
        if proposal.shape != observed.shape or output.shape != observed.shape:
            raise RuntimeError("proposal/output geometry mismatch")
        if proposal.dtype != np.uint8:
            raise RuntimeError("proposal dtype must be uint8")
        if support.shape != observed.shape[:2]:
            raise RuntimeError("support geometry mismatch")
        if not np.array_equal(output, proposal):
            raise RuntimeError("strength-one materialization is not the exact proposal")
        outside_exact = bool(np.array_equal(output[~support], observed[~support]))
        if not outside_exact:
            raise RuntimeError("qualification arm escaped declared support")
        modified = name in tuple(control.modified_endpoints)
        changed = np.any(output != observed, axis=2)
        changed_pixels = int(changed.sum())
        changed_total += changed_pixels
        if modified and not bool(support.any()):
            raise RuntimeError("modified endpoint has empty support")
        if not modified and (bool(support.any()) or not np.array_equal(output, observed)):
            raise RuntimeError("unmodified endpoint changed or declared support")
        endpoint_rows.append({
            "endpoint": name,
            "modified": modified,
            "support_sha256": _array_sha256(support),
            "support_pixels": int(support.sum()),
            "output_sha256": _array_sha256(output),
            "changed_mask_sha256": _array_sha256(changed),
            "changed_pixels": changed_pixels,
            "outside_support_identity": outside_exact,
        })
    if changed_total == 0:
        raise RuntimeError("executed control has zero acting support")
    control_payload = _control_receipt(control)
    local_receipt = build_local_case_materialization_v1(
        qualification_action_id=action_id,
        exact_control_id=str(control.control_id),
        operator_id=str(control.operator),
        observed_first_rgb=pair[0],
        observed_second_rgb=pair[1],
        observed_native_flow=flow,
        output_first_rgb=outputs[0],
        output_second_rgb=outputs[1],
        endpoint_supports=supports,
        modified_endpoints=control.modified_endpoints,
        physical_pair_sha256=physical_pair_sha256,
        endpoint_policy_sha256=ENDPOINT_POLICY_SHA256_V1,
    )
    local_status = "LOCAL_OUTPUT_CONTRACT_READY_COST_AND_SCIENCE_PENDING"
    local_blocker = "PROSPECTIVE_COST_CEILING_AND_DOWNSTREAM_GATES"
    mechanism_receipts: tuple[
        NoiseMechanismEvaluationReceiptV1
        | BlurMechanismEvaluationReceiptV1
        | JPEGMechanismEvaluationReceiptV2,
        ...,
    ] = ()
    try:
        if action_id == LOCAL_BLUR_ACTION_ID_V2:
            parameters = dict(control.parameters)
            mechanism_receipts = (evaluate_blur_mechanism_v1(
                physical_pair_sha256=physical_pair_sha256,
                before_pair=pair,
                after_pair=outputs,
                endpoint_supports=supports,
                observed_native_flow=flow,
                winner_id=str(parameters["winner"]),
                family=str(parameters["family"]),
                identified_endpoint=str(parameters["identified_endpoint"]),
                selected_parameter=parameters["selected_parameter"],
                recoverable_regions=parameters["recoverable_regions"],
            ),)
        elif action_id in {
            LOCAL_JPEG_QCELL_ACTION_ID_V2,
            LOCAL_JPEG_CODEC_ACTION_ID_V2,
        }:
            jpeg_receipts = []
            for index, endpoint in enumerate(("first", "second")):
                if endpoint not in tuple(control.modified_endpoints):
                    continue
                endpoint_parameters = dict(control.parameters[endpoint])
                jpeg_receipts.append(evaluate_jpeg_mechanism_v2(
                    action_id=action_id,
                    physical_pair_sha256=physical_pair_sha256,
                    endpoint=endpoint,
                    before_endpoint=pair[index],
                    after_endpoint=outputs[index],
                    endpoint_support=supports[index],
                    estimated_quality=int(
                        endpoint_parameters["estimated_ijg_quality"]
                    ),
                    selected_strength=float(
                        endpoint_parameters["selected_strength"]
                    ),
                ))
            if len(jpeg_receipts) != len(tuple(control.modified_endpoints)):
                raise RuntimeError("JPEG mechanism endpoint coverage drift")
            mechanism_receipts = tuple(jpeg_receipts)
        else:
            mechanism_receipts = (evaluate_noise_mechanism_v1(
                action_id=action_id,
                physical_pair_sha256=physical_pair_sha256,
                before_pair=pair,
                after_pair=outputs,
                endpoint_supports=supports,
            ),)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        mechanism_receipts = ()
        mechanism_status = "EVALUATOR_FAILED_TYPED_MISSING"
        mechanism_blocker = type(exc).__name__
    else:
        mechanism_status = "IMPLEMENTED_DEVELOPMENT_ESTIMAND_READY"
        mechanism_blocker = "FRESH_SCENE_DISJOINT_INFERENCE_NOT_RUN"
    mechanism_receipt_hashes = [
        receipt.receipt_sha256 for receipt in mechanism_receipts
    ]
    row = {
        "schema": QUALIFICATION_POLICY_ROW_SCHEMA_V1,
        "qualification_action_id": action_id,
        "action_descriptor_sha256": _DESCRIPTOR_SHA256[action_id],
        "exact_control_id": str(control.control_id),
        "qualification_status": "ELIGIBLE_EXECUTED",
        "source_execution_status": str(control.execution_status),
        "source_applicability": str(control.applicability),
        "input_strength": QUALIFICATION_STRENGTH_V1,
        "half_strength_materialized_by_legacy_backend": True,
        "half_strength_persisted": False,
        "half_strength_matcher_scheduled": False,
        "endpoint_type": endpoint_type,
        "exact_endpoint": exact_endpoint,
        "endpoint_policy_sha256": ENDPOINT_POLICY_SHA256_V1,
        "parameter_sha256": canonical_sha256(control.parameters),
        "physical_receipts_sha256": canonical_sha256(
            _stable_physical_receipts(control.receipts)
        ),
        "physical_receipt_identity_policy": (
            "DROP_MEASURED_PROBE_SECONDS_ACCOUNT_RUNTIME_SEPARATELY_V1"
        ),
        "control_receipt_sha256": canonical_sha256(control_payload),
        "endpoints": endpoint_rows,
        "blur_branch": (
            _blur_branch_receipt(control, exact_endpoint)
            if control.control_id == LOCAL_BLUR_ACTION_ID_V2 else None
        ),
        "local_case_materialization_status": local_status,
        "local_case_materialization_blocker": local_blocker,
        "local_case_materialization_receipt_sha256": (
            local_receipt.receipt_sha256 if local_receipt is not None else None
        ),
        "mechanism_evaluator_status": mechanism_status,
        "mechanism_evaluator_blocker": mechanism_blocker,
        "mechanism_receipt_sha256": (
            mechanism_receipt_hashes[0]
            if len(mechanism_receipt_hashes) == 1 else
            canonical_sha256(mechanism_receipt_hashes)
            if mechanism_receipt_hashes else None
        ),
        "mechanism_receipt_sha256s": mechanism_receipt_hashes,
        "capacity_selection_used": False,
        "scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    arm = MaterializedQualificationArmV1(
        qualification_action_id=action_id,
        exact_control_id=str(control.control_id),
        first_rgb=outputs[0],
        second_rgb=outputs[1],
        first_support=supports[0],
        second_support=supports[1],
    )
    return row, arm, local_receipt, mechanism_receipts


def _missing_row(control: Any, action_id: str, endpoint_type: str) -> dict[str, Any]:
    if str(control.execution_status) == "EXECUTED":
        raise RuntimeError("executed target control is unexpectedly ineligible")
    dispositions = {
        "UNSUPPORTED": "UNSUPPORTED",
        "NOT_REACHED": "INELIGIBLE_PREOUTCOME",
        "INVALID_CERTIFICATE": "EXECUTION_FAILED_DISCOVERY",
        "IDENTITY_ONLY": "INELIGIBLE_PREOUTCOME_ZERO_ACTING_SUPPORT",
    }
    source_status = str(control.execution_status)
    if source_status not in dispositions:
        raise RuntimeError(f"unknown ineligible execution status: {source_status}")
    row = {
        "schema": QUALIFICATION_POLICY_ROW_SCHEMA_V1,
        "qualification_action_id": action_id,
        "action_descriptor_sha256": _DESCRIPTOR_SHA256[action_id],
        "exact_control_id": str(control.control_id),
        "qualification_status": "TYPED_MISSING",
        "source_execution_status": source_status,
        "typed_disposition": dispositions[source_status],
        "source_applicability": str(control.applicability),
        "missing_reason": f"{control.execution_status}:{control.applicability}",
        "input_strength": QUALIFICATION_STRENGTH_V1,
        "half_strength_materialized_by_legacy_backend": False,
        "half_strength_persisted": False,
        "half_strength_matcher_scheduled": False,
        "endpoint_type": endpoint_type,
        "exact_endpoint": None,
        "endpoint_policy_sha256": ENDPOINT_POLICY_SHA256_V1,
        "parameter_sha256": canonical_sha256(control.parameters),
        "physical_receipts_sha256": canonical_sha256(
            _stable_physical_receipts(control.receipts)
        ),
        "physical_receipt_identity_policy": (
            "DROP_MEASURED_PROBE_SECONDS_ACCOUNT_RUNTIME_SEPARATELY_V1"
        ),
        "control_receipt_sha256": canonical_sha256(_control_receipt(control)),
        "endpoints": [],
        "blur_branch": None,
        "local_case_materialization_status": "NOT_APPLICABLE_TYPED_MISSING",
        "local_case_materialization_blocker": "ACTION_NOT_EXECUTED",
        "local_case_materialization_receipt_sha256": None,
        "mechanism_evaluator_status": "NOT_APPLICABLE_TYPED_MISSING",
        "mechanism_evaluator_blocker": "ACTION_NOT_EXECUTED",
        "mechanism_receipt_sha256": None,
        "mechanism_receipt_sha256s": [],
        "capacity_selection_used": False,
        "scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    return row


def _replace_legacy_wiener_with_v2(
    controls: Iterable[Any],
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
) -> list[Any]:
    """Replace the structurally identity-prone v1 control with its frozen v2."""

    controls = list(controls)
    legacy = [
        item for item in controls
        if str(item.control_id) == "paired_additive_wiener3"
    ]
    if len(legacy) != 1:
        raise RuntimeError("legacy Wiener discovery identity drift")
    old = legacy[0]
    successor = build_local_wiener_successor_v2(first, second, flow)
    control_type = type(old)
    if successor.status == "EXECUTED_LOCAL_TILE_SIGMA_V2":
        if successor.exact_endpoint not in {"first", "second", "both"}:
            raise RuntimeError("local Wiener successor endpoint drift")
        endpoints = (
            ("first", "second")
            if successor.exact_endpoint == "both"
            else (successor.exact_endpoint,)
        )
        replacement = control_type(
            control_id=LOCAL_WIENER_ACTION_ID_V2,
            family="noise",
            operator=LOCAL_WIENER_OPERATOR_ID_V2,
            authorization=str(old.authorization),
            execution_status="EXECUTED",
            applicability="VERSIONED_LOCAL_TILE_SIGMA_V2",
            modified_endpoints=endpoints,
            parameters={
                "family": "additive",
                "endpoint": successor.exact_endpoint,
                "legacy_control_id": str(old.control_id),
                "legacy_execution_status": str(old.execution_status),
                "successor_receipt_sha256": successor.receipt["receipt_sha256"],
            },
            receipts=tuple(old.receipts),
            proposal=(successor.first_rgb, successor.second_rgb),
            supports=(successor.first_support, successor.second_support),
            diagnostics={
                "versioned_rescue": True,
                "successor_receipt_sha256": successor.receipt["receipt_sha256"],
            },
            observable_score=old.observable_score,
        )
        if not replacement.eligible:
            raise RuntimeError("executed local Wiener successor is unexpectedly ineligible")
    else:
        status = str(old.execution_status)
        if status not in {"UNSUPPORTED", "NOT_REACHED", "INVALID_CERTIFICATE"}:
            raise RuntimeError(
                "local Wiener successor abstained after an incompatible legacy status"
            )
        replacement = control_type(
            control_id=LOCAL_WIENER_ACTION_ID_V2,
            family="noise",
            operator=LOCAL_WIENER_OPERATOR_ID_V2,
            authorization=str(old.authorization),
            execution_status=status,
            applicability=str(successor.status),
            parameters={
                "legacy_control_id": str(old.control_id),
                "legacy_execution_status": status,
                "successor_receipt_sha256": successor.receipt["receipt_sha256"],
            },
            receipts=tuple(old.receipts),
            diagnostics={
                "versioned_rescue": True,
                "successor_receipt_sha256": successor.receipt["receipt_sha256"],
            },
            observable_score=old.observable_score,
        )
    return [replacement if item is old else item for item in controls]


def _replace_legacy_blur_with_v2(
    controls: Iterable[Any],
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
) -> list[Any]:
    """Replace whole-endpoint Work-E blur with its local recoverable successor."""

    controls = list(controls)
    legacy = [
        item for item in controls
        if str(item.control_id) == "blur_v3_cross_endpoint"
    ]
    if len(legacy) != 1:
        raise RuntimeError("legacy routed-blur discovery identity drift")
    old = legacy[0]
    successor = build_local_blur_successor_v2(first, second, flow)
    control_type = type(old)
    if successor.status == "EXECUTED_LOCAL_RECOVERABLE_V2":
        if not bool(old.eligible):
            raise RuntimeError(
                "local blur successor executed after legacy dispatcher abstained"
            )
        old_parameters = dict(old.parameters)
        expected = {
            "winner": successor.winner_id,
            "family": successor.family,
            "identified_endpoint": successor.identified_endpoint,
            "modified_endpoint": successor.exact_endpoint,
            "selected_parameter": list(successor.selected_parameter),
        }
        for name, value in expected.items():
            if old_parameters.get(name) != value:
                raise RuntimeError(f"local blur successor disagrees with legacy {name}")
        replacement = control_type(
            control_id=LOCAL_BLUR_ACTION_ID_V2,
            family="blur",
            operator=LOCAL_BLUR_OPERATOR_ID_V2,
            authorization=str(old.authorization),
            execution_status="EXECUTED",
            applicability="VERSIONED_LOCAL_RECOVERABLE_V2",
            modified_endpoints=(str(successor.exact_endpoint),),
            parameters={
                **expected,
                "recoverable_regions": [
                    list(value) for value in successor.recoverable_regions
                ],
                "legacy_control_id": str(old.control_id),
                "legacy_execution_status": str(old.execution_status),
                "successor_receipt_sha256": successor.receipt["receipt_sha256"],
            },
            receipts=tuple(old.receipts),
            proposal=(successor.first_rgb, successor.second_rgb),
            supports=(successor.first_support, successor.second_support),
            diagnostics={
                "versioned_localization": True,
                "successor_receipt_sha256": successor.receipt["receipt_sha256"],
                "legacy_whole_endpoint_proposal_not_retained": True,
            },
            observable_score=old.observable_score,
        )
        if not replacement.eligible:
            raise RuntimeError("executed local blur successor is unexpectedly ineligible")
    else:
        if successor.status.startswith("IDENTITY_ONLY"):
            status = "IDENTITY_ONLY"
        elif successor.status == "INVALID_CERTIFICATE_V3":
            status = "INVALID_CERTIFICATE"
        elif successor.status in {
            "UNSUPPORTED_DIRECT_WINNER_COUNT",
            "UNSUPPORTED_RECOVERABLE_REGIONS",
        }:
            status = "UNSUPPORTED"
        else:
            raise RuntimeError(
                f"unknown local blur successor abstention: {successor.status}"
            )
        legacy_detected_without_local_recoverability = bool(old.eligible)
        replacement = control_type(
            control_id=LOCAL_BLUR_ACTION_ID_V2,
            family="blur",
            operator=LOCAL_BLUR_OPERATOR_ID_V2,
            authorization=str(old.authorization),
            execution_status=status,
            applicability=str(successor.status),
            parameters={
                "legacy_control_id": str(old.control_id),
                "legacy_execution_status": str(old.execution_status),
                "legacy_detected_without_local_recoverability": (
                    legacy_detected_without_local_recoverability
                ),
                "successor_receipt_sha256": successor.receipt["receipt_sha256"],
            },
            receipts=tuple(old.receipts),
            diagnostics={
                "versioned_localization": True,
                "successor_receipt_sha256": successor.receipt["receipt_sha256"],
                "legacy_detected_without_local_recoverability": (
                    legacy_detected_without_local_recoverability
                ),
                "local_successor_status": str(successor.status),
            },
            observable_score=old.observable_score,
        )
    return [replacement if item is old else item for item in controls]


def _replace_legacy_jpeg_with_v2(
    controls: Iterable[Any],
    first: np.ndarray,
    second: np.ndarray,
) -> list[Any]:
    """Version both JPEG policies onto their before-only local successors.

    E132 and the legacy action certificate continue to authorize endpoint and
    quality.  The successor is allowed to add only localization: its full
    proposal must be byte-identical to the proposal discovered by Work-E.
    """

    controls = list(controls)
    pair = (first, second)
    definitions = (
        (
            "jpeg_qcell_v3",
            LOCAL_JPEG_QCELL_ACTION_ID_V2,
            LOCAL_JPEG_QCELL_OPERATOR_ID_V2,
            "qcell",
        ),
        (
            "jpeg_codec_path_v4",
            LOCAL_JPEG_CODEC_ACTION_ID_V2,
            LOCAL_JPEG_CODEC_OPERATOR_ID_V2,
            "codec",
        ),
    )
    replacements: dict[str, Any] = {}
    for legacy_id, action_id, operator_id, policy in definitions:
        matches = [item for item in controls if str(item.control_id) == legacy_id]
        if len(matches) != 1:
            raise RuntimeError(f"legacy JPEG discovery identity drift: {legacy_id}")
        old = matches[0]
        control_type = type(old)
        if not bool(old.eligible):
            replacements[legacy_id] = control_type(
                control_id=action_id,
                family="jpeg",
                operator=operator_id,
                authorization=str(old.authorization),
                execution_status=str(old.execution_status),
                applicability=str(old.applicability),
                parameters={
                    "legacy_control_id": legacy_id,
                    "legacy_execution_status": str(old.execution_status),
                    "successor_execution_attempted": False,
                },
                receipts=tuple(old.receipts),
                diagnostics={
                    "versioned_localization": True,
                    "legacy_diagnostics": _clean(old.diagnostics),
                },
                observable_score=old.observable_score,
            )
            continue
        if old.proposal is None or old.supports is None:
            raise RuntimeError("eligible legacy JPEG control omitted proposal/support")
        localized_outputs = [value.copy() for value in pair]
        localized_supports = [
            np.zeros(first.shape[:2], dtype=bool),
            np.zeros(first.shape[:2], dtype=bool),
        ]
        parameters: dict[str, Any] = {
            "legacy_control_id": legacy_id,
            "legacy_execution_status": str(old.execution_status),
        }
        active_endpoints = []
        successor_statuses = {}
        for endpoint in tuple(old.modified_endpoints):
            if endpoint not in {"first", "second"}:
                raise RuntimeError("legacy JPEG endpoint drift")
            index = 0 if endpoint == "first" else 1
            old_endpoint_parameters = dict(old.parameters[endpoint])
            quality = int(old_endpoint_parameters["estimated_ijg_quality"])
            successor = build_local_jpeg_successor_v2(
                pair[index],
                policy=policy,
                endpoint=endpoint,
                estimated_quality=quality,
            )
            legacy_full_hash = _array_sha256(old.proposal[index])
            if successor.receipt["full_proposal_sha256"] != legacy_full_hash:
                raise RuntimeError(
                    f"local JPEG successor changed the {legacy_id} proposal path"
                )
            if policy == "codec":
                legacy_strength = float(
                    old_endpoint_parameters["internal_dyadic_strength"]
                )
                if successor.selected_strength != legacy_strength:
                    raise RuntimeError("local codec successor strength drift")
            elif successor.selected_strength != 1.0:
                raise RuntimeError("local qcell successor strength drift")
            successor_statuses[endpoint] = successor.status
            parameters[endpoint] = {
                "estimated_ijg_quality": quality,
                "selected_strength": successor.selected_strength,
                "legacy_full_proposal_sha256": legacy_full_hash,
                "successor_receipt_sha256": successor.receipt["receipt_sha256"],
            }
            if successor.status == "EXECUTED_LOCAL_MACROBLOCK_V2":
                localized_outputs[index] = successor.output_rgb
                localized_supports[index] = successor.support
                active_endpoints.append(endpoint)
        parameters["successor_statuses"] = successor_statuses
        if active_endpoints:
            replacement = control_type(
                control_id=action_id,
                family="jpeg",
                operator=operator_id,
                authorization=str(old.authorization),
                execution_status="EXECUTED",
                applicability=(
                    "VERSIONED_LOCAL_MACROBLOCK_V2:"
                    + ",".join(active_endpoints)
                ),
                modified_endpoints=tuple(active_endpoints),
                parameters=parameters,
                receipts=tuple(old.receipts),
                proposal=tuple(localized_outputs),
                supports=tuple(localized_supports),
                diagnostics={
                    "versioned_localization": True,
                    "legacy_whole_endpoint_proposal_not_retained": True,
                    "successor_statuses": successor_statuses,
                },
                observable_score=old.observable_score,
            )
            if not replacement.eligible:
                raise RuntimeError(
                    "executed local JPEG successor is unexpectedly ineligible"
                )
        else:
            statuses = set(successor_statuses.values())
            status = (
                "IDENTITY_ONLY"
                if any(value.startswith("IDENTITY_ONLY") for value in statuses)
                else "UNSUPPORTED"
            )
            replacement = control_type(
                control_id=action_id,
                family="jpeg",
                operator=operator_id,
                authorization=str(old.authorization),
                execution_status=status,
                applicability="LOCAL_MACROBLOCK_V2_NO_ACTING_SUPPORT",
                parameters=parameters,
                receipts=tuple(old.receipts),
                diagnostics={
                    "versioned_localization": True,
                    "successor_statuses": successor_statuses,
                },
                observable_score=old.observable_score,
            )
        replacements[legacy_id] = replacement
    return [replacements.get(str(item.control_id), item) for item in controls]


def _build_uncapped_matrix(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    *,
    controls: Iterable[Any],
    physical_pair_sha256: str,
    materialize: Callable[[Any, np.ndarray, np.ndarray], Mapping[float, tuple[np.ndarray, np.ndarray]]],
) -> QualificationMatrixResultV1:
    """Internal injectable boundary used for contract tests."""

    first, second, flow = _validate_inputs(first, second, flow)
    if (
        not isinstance(physical_pair_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", physical_pair_sha256) is None
    ):
        raise ValueError("physical pair binding must be a SHA-256")
    if physical_pair_sha256 != _physical_pair_sha256(first, second, flow):
        raise RuntimeError("physical pair hash does not bind the supplied arrays")
    by_id: dict[str, Any] = {}
    for control in controls:
        control_id = str(control.control_id)
        if control_id in by_id:
            raise RuntimeError(f"duplicate discovered control: {control_id}")
        by_id[control_id] = control
    target_ids = {row[0] for row in _POLICIES}
    expected_ids = target_ids | {"radiometry_rank3_diagnostic"}
    if set(by_id) != expected_ids:
        extras = sorted(set(by_id) - expected_ids)
        absent = sorted(expected_ids - set(by_id))
        raise RuntimeError(
            f"discovery control universe drift; missing={absent}, unknown={extras}"
        )
    diagnostic = by_id["radiometry_rank3_diagnostic"]
    if (
        str(diagnostic.authorization) != "diagnostic_only"
        or bool(diagnostic.eligible)
        or diagnostic.proposal is not None
        or diagnostic.supports is not None
    ):
        raise RuntimeError("rank3 diagnostic crossed the execution boundary")

    rows: list[dict[str, Any]] = []
    arms: list[MaterializedQualificationArmV1] = []
    local_receipts: list[LocalCaseMaterializationReceiptV1] = []
    mechanism_receipts: list[
        NoiseMechanismEvaluationReceiptV1
        | BlurMechanismEvaluationReceiptV1
        | JPEGMechanismEvaluationReceiptV2
    ] = []
    source_hashes_before = (_array_sha256(first), _array_sha256(second))
    for control_id, action_id, endpoint_type in _POLICIES:
        control = by_id[control_id]
        selection_status = str(getattr(control, "selection_status", "NOT_RANKED"))
        if selection_status != "NOT_RANKED":
            raise RuntimeError(
                f"qualification forbids ranked/capacity-mutated control {control_id}: "
                f"{selection_status}"
            )
        if bool(control.eligible):
            row, arm, local_receipt, row_mechanism_receipts = _eligible_row(
                control=control,
                action_id=action_id,
                endpoint_type=endpoint_type,
                pair=(first, second),
                flow=flow,
                physical_pair_sha256=physical_pair_sha256,
                materialize=materialize,
            )
            rows.append(row)
            arms.append(arm)
            if local_receipt is not None:
                local_receipts.append(local_receipt)
            mechanism_receipts.extend(row_mechanism_receipts)
        else:
            rows.append(_missing_row(control, action_id, endpoint_type))

    if source_hashes_before != (_array_sha256(first), _array_sha256(second)):
        raise RuntimeError("qualification materialization mutated observed inputs")

    eligible = sum(row["qualification_status"] == "ELIGIBLE_EXECUTED" for row in rows)
    receipt = {
        "schema": QUALIFICATION_MATRIX_SCHEMA_V1,
        "case_id": f"physical-pair:{physical_pair_sha256}",
        "physical_pair_sha256": physical_pair_sha256,
        "source_input_sha256s": [
            _array_sha256(first), _array_sha256(second), _array_sha256(flow),
        ],
        "qualification_policy_order": [row[1] for row in _POLICIES],
        "endpoint_policy_sha256": ENDPOINT_POLICY_SHA256_V1,
        "execution_surface": "DISCOVER_ONCE_MATERIALIZE_EVERY_ELIGIBLE_AT_STRENGTH_ONE",
        "discovery_call_count": 1,
        "selector_capacity": None,
        "capacity_selection_used": False,
        "ranking_invoked": False,
        "ground_truth_fields_read": [],
        "outcome_fields_read": [],
        "native_reference": {
            "first_sha256": _array_sha256(first),
            "second_sha256": _array_sha256(second),
            "native_flow_sha256": _array_sha256(flow),
        },
        "diagnostic": {
            "control_id": "radiometry_rank3_diagnostic",
            "authorization": str(diagnostic.authorization),
            "execution_status": str(diagnostic.execution_status),
            "applicability": str(diagnostic.applicability),
            "control_receipt_sha256": canonical_sha256(_control_receipt(diagnostic)),
            "action_row": False,
            "matcher_scheduled": False,
        },
        "counts": {
            "policies": len(rows),
            "eligible_executed": eligible,
            "typed_missing": len(rows) - eligible,
            "materialized_arms": len(arms),
            "local_output_contracts_ready": len(local_receipts),
            "mechanism_evaluations_ready": len(mechanism_receipts),
        },
        "rows": rows,
        "authority": {
            "execution_receipt": True,
            "scientific_qualification": False,
            "selector_admission": False,
            "production": False,
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return QualificationMatrixResultV1(
        receipt=receipt,
        arms=tuple(arms),
        local_case_receipts=tuple(local_receipts),
        mechanism_receipts=tuple(mechanism_receipts),
    )


def build_uncapped_qualification_matrix(
    observed_first_rgb: np.ndarray,
    observed_second_rgb: np.ndarray,
    observed_native_flow: np.ndarray,
) -> QualificationMatrixResultV1:
    """Discover once and execute every eligible E249 policy without ranking.

    The public surface intentionally accepts only deployment-observable arrays.
    In particular there is no dataset/path/scene/corruption/outcome/GT input.
    """

    first, second, flow = _validate_inputs(
        observed_first_rgb, observed_second_rgb, observed_native_flow,
    )
    backend = _work_e_backend()
    controls, pair_hash = backend.discover_exact_controls(first, second, flow)
    controls = _replace_legacy_wiener_with_v2(
        controls, first, second, flow,
    )
    controls = _replace_legacy_blur_with_v2(
        controls, first, second, flow,
    )
    controls = _replace_legacy_jpeg_with_v2(
        controls, first, second,
    )
    result = _build_uncapped_matrix(
        first,
        second,
        flow,
        controls=controls,
        physical_pair_sha256=pair_hash,
        materialize=backend.materialize_control,
    )
    _verify_frozen_sources()
    return result


__all__ = [
    "ENDPOINT_POLICY_SHA256_V1",
    "MaterializedQualificationArmV1",
    "QUALIFICATION_MATRIX_SCHEMA_V1",
    "QUALIFICATION_POLICY_ROW_SCHEMA_V1",
    "QUALIFICATION_STRENGTH_V1",
    "QualificationMatrixResultV1",
    "build_uncapped_qualification_matrix",
]
