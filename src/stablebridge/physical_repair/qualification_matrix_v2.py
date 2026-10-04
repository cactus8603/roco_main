"""Six-row outcome-blind qualification matrix for foundation action bank v2.

The historical matrix remains sealed.  This version reuses its frozen Work-E
discovery only for impulse, additive-noise, JPEG endpoint/quality evidence, and
the diagnostic row.  Blur v6 is evaluated independently of the legacy blur
eligibility gate and is computed exactly once; its isotropic and motion public
arms receive separate typed rows, with at most one executed.

Every executed action is full-strength and local.  Every non-executed action
still receives a typed-missing row.  No GT, task outcome, ranking, capacity
selection, scientific qualification, or selector admission occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from . import qualification_matrix as v1
from .blur_mechanism_evaluators_v2 import (
    BlurMechanismEvaluationReceiptV2,
    evaluate_blur_mechanism_v2,
)
from .jpeg_mechanism_evaluators_v3 import (
    JPEGMechanismEvaluationReceiptV3,
    evaluate_jpeg_mechanism_v3,
)
from .local_blur_successor_v3 import (
    LOCAL_BLUR_ACTION_IDS_V3,
    build_local_blur_successor_v3,
)
from .local_case_materialization import LocalCaseMaterializationReceiptV1
from .local_case_materialization_v2 import (
    LocalCaseMaterializationReceiptV2,
    build_local_case_materialization_v2,
)
from .local_jpeg_case_materialization_v3 import (
    LocalJPEGCaseMaterializationReceiptV3,
    build_local_jpeg_case_materialization_v3,
)
from .local_jpeg_successors_v3 import (
    LOCAL_JPEG_CODEC_ACTION_ID_V3,
    LOCAL_JPEG_QCELL_ACTION_ID_V3,
    LocalJPEGSuccessorV3,
    build_local_jpeg_successor_v3,
)
from .noise_mechanism_evaluators import NoiseMechanismEvaluationReceiptV1


QUALIFICATION_MATRIX_SCHEMA_V2 = "stablebridge-uncapped-qualification-matrix/v2"
QUALIFICATION_POLICY_ROW_SCHEMA_V2 = "stablebridge-qualification-policy-execution/v2"
QUALIFICATION_STRENGTH_V2 = 1.0

_ROOT = Path(__file__).resolve().parents[3]
_E262_BANK = _ROOT / "experiments/E262_action_bank_v2_foundation_freeze/ACTION_BANK_V2.json"
_E262_SEAL = _ROOT / "experiments/E262_action_bank_v2_foundation_freeze/PACKAGE_SEAL.json"
_SOURCE_SHA256 = {
    _E262_BANK: "2014d1cd7af488893b37765993e3ced1d22aa00a94688f9c39df30d6fbef4665",
    _E262_SEAL: "a4b66cabd2a59da438fac41c5bb929a7b26a2fecb9ee8f047fd8a773520a7bac",
    _ROOT / "src/stablebridge/physical_repair/qualification_matrix.py": (
        "7cc4d6680c41f44e839dd4e30450b4b9b5993c6aeed0c86304302e80dc02c784"
    ),
    _ROOT / "src/stablebridge/physical_repair/local_blur_successor_v3.py": (
        "092f519659e247171271208c5b5e9bd3fc2c2abb40bc3eabc5419299175f9021"
    ),
    _ROOT / "src/stablebridge/physical_repair/blur_mechanism_evaluators_v2.py": (
        "f01b5d7b79109e20fc47614793bacfad4aa005827ef3d51e00b77f9b2fa626ab"
    ),
    _ROOT / "src/stablebridge/physical_repair/local_case_materialization_v2.py": (
        "3fb1b0ec18c1e921d631b08001403a6f18291e1538f971db7279dba52d9dcfd8"
    ),
    _ROOT / "src/stablebridge/physical_repair/local_jpeg_successors_v3.py": (
        "051a930cf41ecbe2db64ab871d4f0f6529c73e4a7323723ed3c7da9790138b2d"
    ),
    _ROOT / "src/stablebridge/physical_repair/jpeg_mechanism_evaluators_v3.py": (
        "e015da6706a562ac8a1b2c86f80a12dc02df8f990bbf186169677d9a89fa33a9"
    ),
    _ROOT / "src/stablebridge/physical_repair/local_jpeg_case_materialization_v3.py": (
        "3b61335e6e8209b97bb2f9a2087ac2e538ed5afa1b235369cb0f9bb8d5b3fcdf"
    ),
}

_BANK = json.loads(_E262_BANK.read_text())
_BANK_SHA256 = str(_BANK["bank_sha256"])
_DESCRIPTOR_SHA256 = {
    row["action_id"]: row["descriptor_hash"] for row in _BANK["actions"]
}

IMPULSE_ACTION_ID = "paired_impulse_median3.endpoint_supported_v1"
WIENER_ACTION_ID = "paired_additive_wiener3.local_tile_sigma_v2"
ISOTROPIC_ACTION_ID = LOCAL_BLUR_ACTION_IDS_V3["common_isotropic"]
MOTION_ACTION_ID = LOCAL_BLUR_ACTION_IDS_V3["common_motion"]

_POLICY_ORDER = (
    IMPULSE_ACTION_ID,
    WIENER_ACTION_ID,
    ISOTROPIC_ACTION_ID,
    MOTION_ACTION_ID,
    LOCAL_JPEG_QCELL_ACTION_ID_V3,
    LOCAL_JPEG_CODEC_ACTION_ID_V3,
)


def _verify_sources() -> None:
    for path, expected in _SOURCE_SHA256.items():
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(_ROOT.resolve()):
            raise RuntimeError(f"qualification v2 source escaped workspace: {path}")
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"qualification v2 source drift for {path.relative_to(_ROOT)}: "
                f"expected {expected}, got {actual}"
            )


def _row_hash(row: dict[str, Any]) -> dict[str, Any]:
    row.pop("row_sha256", None)
    row["row_sha256"] = v1.canonical_sha256(row)
    return row


def _rebind_v1_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    action_id = str(output["qualification_action_id"])
    output.update({
        "schema": QUALIFICATION_POLICY_ROW_SCHEMA_V2,
        "action_descriptor_sha256": _DESCRIPTOR_SHA256[action_id],
        "prospective_cost_status": "UNBOUND_TYPED_MISSING",
        "child_observable_status": "NOT_RUN_TYPED_MISSING",
        "child_observable_missing": "REGION_PROJECTED_CC_RR_CR_RC_RECEIPT",
    })
    return _row_hash(output)


def _typed_missing(
    *,
    action_id: str,
    source_status: str,
    source_applicability: str,
    missing_reason: str,
    parameter_payload: Any,
    control: Any | None = None,
) -> dict[str, Any]:
    control_hash = (
        v1.canonical_sha256(v1._control_receipt(control))
        if control is not None else None
    )
    row = {
        "schema": QUALIFICATION_POLICY_ROW_SCHEMA_V2,
        "qualification_action_id": action_id,
        "action_descriptor_sha256": _DESCRIPTOR_SHA256[action_id],
        "exact_control_id": action_id,
        "qualification_status": "TYPED_MISSING",
        "source_execution_status": source_status,
        "typed_disposition": "INELIGIBLE_PREOUTCOME",
        "source_applicability": source_applicability,
        "missing_reason": missing_reason,
        "input_strength": QUALIFICATION_STRENGTH_V2,
        "endpoint_type": (
            "case_selected_single_supported"
            if action_id in {ISOTROPIC_ACTION_ID, MOTION_ACTION_ID}
            else "case_selected_supported"
        ),
        "exact_endpoint": None,
        "endpoint_policy_sha256": v1.ENDPOINT_POLICY_SHA256_V1,
        "parameter_sha256": v1.canonical_sha256(parameter_payload),
        "control_receipt_sha256": control_hash,
        "endpoints": [],
        "blur_branch": None,
        "local_case_materialization_status": "NOT_APPLICABLE_TYPED_MISSING",
        "local_case_materialization_blocker": "ACTION_NOT_EXECUTED",
        "local_case_materialization_receipt_sha256": None,
        "mechanism_evaluator_status": "NOT_APPLICABLE_TYPED_MISSING",
        "mechanism_evaluator_blocker": "ACTION_NOT_EXECUTED",
        "mechanism_receipt_sha256": None,
        "mechanism_receipt_sha256s": [],
        "prospective_cost_status": "UNBOUND_TYPED_MISSING",
        "child_observable_status": "NOT_RUN_TYPED_MISSING",
        "child_observable_missing": "REGION_PROJECTED_CC_RR_CR_RC_RECEIPT",
        "capacity_selection_used": False,
        "scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
    }
    return _row_hash(row)


def _endpoint_rows(
    pair: tuple[np.ndarray, np.ndarray],
    outputs: tuple[np.ndarray, np.ndarray],
    writes: tuple[np.ndarray, np.ndarray],
    reads: tuple[np.ndarray, np.ndarray],
    modified_endpoints: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for name, observed, output, write, read in zip(
        ("first", "second"), pair, outputs, writes, reads,
    ):
        changed = np.any(output != observed, axis=2)
        if np.any(changed & ~write) or not np.array_equal(
            output[~write], observed[~write]
        ):
            raise RuntimeError("qualification v2 action escaped write support")
        if np.any(write & ~read):
            raise RuntimeError("qualification v2 write support escaped read support")
        rows.append({
            "endpoint": name,
            "modified": name in modified_endpoints,
            "write_support_sha256": v1._array_sha256(write),
            "write_support_pixels": int(write.sum()),
            "read_support_sha256": v1._array_sha256(read),
            "read_support_pixels": int(read.sum()),
            "output_sha256": v1._array_sha256(output),
            "changed_mask_sha256": v1._array_sha256(changed),
            "changed_pixels": int(changed.sum()),
            "outside_write_support_identity": True,
        })
    return rows


@dataclass(frozen=True)
class QualificationMatrixResultV2:
    receipt: Mapping[str, Any]
    arms: tuple[v1.MaterializedQualificationArmV1, ...]
    local_case_receipts: tuple[
        LocalCaseMaterializationReceiptV1
        | LocalCaseMaterializationReceiptV2
        | LocalJPEGCaseMaterializationReceiptV3,
        ...,
    ]
    mechanism_receipts: tuple[
        NoiseMechanismEvaluationReceiptV1
        | BlurMechanismEvaluationReceiptV2
        | JPEGMechanismEvaluationReceiptV3,
        ...,
    ]


def _blur_policy_rows(
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    *,
    pair_sha256: str,
    legacy_control: Any,
) -> tuple[
    list[dict[str, Any]],
    list[v1.MaterializedQualificationArmV1],
    list[LocalCaseMaterializationReceiptV2],
    list[BlurMechanismEvaluationReceiptV2],
]:
    successor = build_local_blur_successor_v3(first, second, flow)
    rows: list[dict[str, Any]] = []
    arms: list[v1.MaterializedQualificationArmV1] = []
    local_receipts: list[LocalCaseMaterializationReceiptV2] = []
    mechanism_receipts: list[BlurMechanismEvaluationReceiptV2] = []
    selected_action = (
        successor.action_id
        if successor.status == "EXECUTED_LOCAL_RECOVERABLE_V3" else None
    )
    for public_arm, action_id in (
        ("common_isotropic", ISOTROPIC_ACTION_ID),
        ("common_motion", MOTION_ACTION_ID),
    ):
        if selected_action != action_id:
            reason = (
                f"OTHER_UNIQUE_BLUR_ARM_EXECUTED:{selected_action}"
                if selected_action else successor.status
            )
            rows.append(_typed_missing(
                action_id=action_id,
                source_status="UNSUPPORTED",
                source_applicability="BLUR_V6_UNIQUE_GLOBAL_CANDIDATE",
                missing_reason=reason,
                parameter_payload={
                    "successor_receipt_sha256": successor.receipt["receipt_sha256"],
                    "public_arm": public_arm,
                },
                control=legacy_control,
            ))
            continue
        local = build_local_case_materialization_v2(
            case_id=f"physical-pair:{pair_sha256}",
            physical_pair_sha256=pair_sha256,
            endpoint_policy_sha256=v1.ENDPOINT_POLICY_SHA256_V1,
            observed_first_rgb=first,
            observed_second_rgb=second,
            observed_native_flow=flow,
            successor=successor,
        )
        try:
            mechanism = evaluate_blur_mechanism_v2(
                physical_pair_sha256=pair_sha256,
                before_pair=(first, second),
                successor=successor,
                observed_native_flow=flow,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            mechanism = None
            mechanism_status = "EVALUATOR_FAILED_TYPED_MISSING"
            mechanism_blocker = type(exc).__name__
        else:
            mechanism_receipts.append(mechanism)
            mechanism_status = "IMPLEMENTED_DEVELOPMENT_ESTIMAND_READY"
            mechanism_blocker = "FRESH_SCENE_DISJOINT_INFERENCE_NOT_RUN"
        exact_endpoint = str(successor.exact_endpoint)
        outputs = (successor.first_rgb, successor.second_rgb)
        writes = (successor.first_support, successor.second_support)
        reads = (successor.first_read_support, successor.second_read_support)
        endpoint_rows = _endpoint_rows(
            (first, second), outputs, writes, reads, (exact_endpoint,),
        )
        receipt_hashes = [mechanism.receipt_sha256] if mechanism else []
        row = {
            "schema": QUALIFICATION_POLICY_ROW_SCHEMA_V2,
            "qualification_action_id": action_id,
            "action_descriptor_sha256": _DESCRIPTOR_SHA256[action_id],
            "exact_control_id": action_id,
            "qualification_status": "ELIGIBLE_EXECUTED",
            "source_execution_status": "EXECUTED",
            "source_applicability": "BLUR_V6_UNIQUE_GLOBAL_CANDIDATE",
            "input_strength": QUALIFICATION_STRENGTH_V2,
            "endpoint_type": "case_selected_single_supported",
            "exact_endpoint": exact_endpoint,
            "endpoint_policy_sha256": v1.ENDPOINT_POLICY_SHA256_V1,
            "parameter_sha256": local.exact_parameter_sha256,
            "control_receipt_sha256": v1.canonical_sha256(
                v1._control_receipt(legacy_control)
            ),
            "endpoints": endpoint_rows,
            "blur_branch": {
                "public_arm": successor.public_arm,
                "winner_id": successor.winner_id,
                "internal_family": successor.internal_family,
                "source_action_key": successor.receipt["source_action_key"],
                "selected_parameter_sha256": successor.receipt[
                    "selected_parameter_sha256"
                ],
                "fit_support_sha256": successor.receipt["fit_support_sha256"],
                "check_support_sha256": successor.receipt["check_support_sha256"],
                "identified_endpoint": successor.identified_endpoint,
                "modified_endpoint": exact_endpoint,
            },
            "local_case_materialization_status": (
                "LOCAL_OUTPUT_CONTRACT_READY_COST_AND_CHILDREN_PENDING"
            ),
            "local_case_materialization_blocker": (
                "PROSPECTIVE_COST_AND_REGION_PROJECTED_CHILDREN"
            ),
            "local_case_materialization_receipt_sha256": local.receipt_sha256,
            "mechanism_evaluator_status": mechanism_status,
            "mechanism_evaluator_blocker": mechanism_blocker,
            "mechanism_receipt_sha256": (
                receipt_hashes[0] if receipt_hashes else None
            ),
            "mechanism_receipt_sha256s": receipt_hashes,
            "prospective_cost_status": "UNBOUND_TYPED_MISSING",
            "child_observable_status": "NOT_RUN_TYPED_MISSING",
            "child_observable_missing": "REGION_PROJECTED_CC_RR_CR_RC_RECEIPT",
            "capacity_selection_used": False,
            "scientific_qualification": False,
            "selector_admission": False,
            "production_authority": False,
        }
        rows.append(_row_hash(row))
        arms.append(v1.MaterializedQualificationArmV1(
            qualification_action_id=action_id,
            exact_control_id=action_id,
            first_rgb=successor.first_rgb,
            second_rgb=successor.second_rgb,
            first_support=successor.first_support,
            second_support=successor.second_support,
        ))
        local_receipts.append(local)
    return rows, arms, local_receipts, mechanism_receipts


def _jpeg_policy_row(
    *,
    policy: str,
    action_id: str,
    control: Any,
    first: np.ndarray,
    second: np.ndarray,
    flow: np.ndarray,
    pair_sha256: str,
) -> tuple[
    dict[str, Any],
    v1.MaterializedQualificationArmV1 | None,
    LocalJPEGCaseMaterializationReceiptV3 | None,
    list[JPEGMechanismEvaluationReceiptV3],
]:
    if not bool(control.eligible):
        return (
            _typed_missing(
                action_id=action_id,
                source_status=str(control.execution_status),
                source_applicability=str(control.applicability),
                missing_reason=(
                    f"LEGACY_BEFORE_ONLY_ENDPOINT_QUALITY_UNAVAILABLE:"
                    f"{control.execution_status}:{control.applicability}"
                ),
                parameter_payload=control.parameters,
                control=control,
            ),
            None,
            None,
            [],
        )
    if control.proposal is None or control.supports is None:
        raise RuntimeError("eligible JPEG discovery omitted proposal/support")
    pair = (first, second)
    successors: dict[str, LocalJPEGSuccessorV3] = {}
    statuses = {}
    contract_failure = None
    for endpoint in tuple(control.modified_endpoints):
        if endpoint not in {"first", "second"}:
            raise RuntimeError("JPEG v3 discovery endpoint drift")
        index = 0 if endpoint == "first" else 1
        parameters = dict(control.parameters[endpoint])
        quality = int(parameters["estimated_ijg_quality"])
        try:
            successor = build_local_jpeg_successor_v3(
                pair[index], policy=policy, endpoint=endpoint,
                estimated_quality=quality,
            )
        except (ValueError, RuntimeError) as exc:
            contract_failure = type(exc).__name__
            statuses[endpoint] = f"CONTRACT_FAILURE:{type(exc).__name__}"
            continue
        statuses[endpoint] = successor.status
        if successor.receipt["full_proposal_sha256"] != v1._array_sha256(
            control.proposal[index]
        ):
            raise RuntimeError("JPEG v3 full proposal drift from frozen discovery")
        if policy == "codec":
            if successor.selected_strength != float(parameters["internal_dyadic_strength"]):
                raise RuntimeError("JPEG v3 codec selected strength drift")
        elif successor.selected_strength != 1.0:
            raise RuntimeError("JPEG v3 qcell selected strength drift")
        if successor.status == "EXECUTED_LOCAL_BLOCK_EXACT_V3":
            successors[endpoint] = successor
    if contract_failure is not None:
        return (
            _typed_missing(
                action_id=action_id,
                source_status="INVALID_CERTIFICATE",
                source_applicability="JPEG_V3_BLOCK_EXACT_CONTRACT",
                missing_reason=f"CONTRACT_FAILURE:{contract_failure}",
                parameter_payload=statuses,
                control=control,
            ), None, None, [],
        )
    if not successors:
        return (
            _typed_missing(
                action_id=action_id,
                source_status="UNSUPPORTED",
                source_applicability="JPEG_V3_BLOCK_EXACT_NO_ACTING_SUPPORT",
                missing_reason=v1.canonical_sha256(statuses),
                parameter_payload=statuses,
                control=control,
            ), None, None, [],
        )
    local = build_local_jpeg_case_materialization_v3(
        action_id=action_id,
        case_id=f"physical-pair:{pair_sha256}",
        physical_pair_sha256=pair_sha256,
        endpoint_policy_sha256=v1.ENDPOINT_POLICY_SHA256_V1,
        observed_first_rgb=first,
        observed_second_rgb=second,
        observed_native_flow=flow,
        successors=successors,
    )
    outputs = [first.copy(), second.copy()]
    writes = [
        np.zeros(first.shape[:2], dtype=bool),
        np.zeros(first.shape[:2], dtype=bool),
    ]
    reads = [writes[0].copy(), writes[1].copy()]
    mechanisms: list[JPEGMechanismEvaluationReceiptV3] = []
    mechanism_failure = None
    for endpoint, successor in sorted(successors.items()):
        index = 0 if endpoint == "first" else 1
        outputs[index] = successor.output_rgb
        writes[index] = successor.support
        reads[index] = np.ones(first.shape[:2], dtype=bool)
        try:
            mechanisms.append(evaluate_jpeg_mechanism_v3(
                action_id=action_id,
                physical_pair_sha256=pair_sha256,
                endpoint=endpoint,
                before_endpoint=pair[index],
                after_endpoint=successor.output_rgb,
                endpoint_support=successor.support,
                estimated_quality=successor.estimated_quality,
                selected_strength=successor.selected_strength,
            ))
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            mechanism_failure = type(exc).__name__
            mechanisms = []
            break
    modified = tuple(sorted(successors))
    endpoint_rows = _endpoint_rows(
        pair, tuple(outputs), tuple(writes), tuple(reads), modified,
    )
    mechanism_hashes = [row.receipt_sha256 for row in mechanisms]
    row = {
        "schema": QUALIFICATION_POLICY_ROW_SCHEMA_V2,
        "qualification_action_id": action_id,
        "action_descriptor_sha256": _DESCRIPTOR_SHA256[action_id],
        "exact_control_id": action_id,
        "qualification_status": "ELIGIBLE_EXECUTED",
        "source_execution_status": "EXECUTED",
        "source_applicability": "JPEG_V3_ALIGNED_BLOCK_EXACT",
        "input_strength": QUALIFICATION_STRENGTH_V2,
        "endpoint_type": "case_selected_supported",
        "exact_endpoint": local.exact_endpoint,
        "endpoint_policy_sha256": v1.ENDPOINT_POLICY_SHA256_V1,
        "parameter_sha256": local.exact_parameter_sha256,
        "control_receipt_sha256": v1.canonical_sha256(
            v1._control_receipt(control)
        ),
        "endpoints": endpoint_rows,
        "blur_branch": None,
        "jpeg_endpoint_statuses": statuses,
        "local_case_materialization_status": (
            "LOCAL_OUTPUT_CONTRACT_READY_COST_AND_CHILDREN_PENDING"
        ),
        "local_case_materialization_blocker": (
            "PROSPECTIVE_COST_AND_REGION_PROJECTED_CHILDREN"
        ),
        "local_case_materialization_receipt_sha256": local.receipt_sha256,
        "mechanism_evaluator_status": (
            "IMPLEMENTED_DEVELOPMENT_ESTIMAND_READY"
            if mechanisms else "EVALUATOR_FAILED_TYPED_MISSING"
        ),
        "mechanism_evaluator_blocker": (
            "FRESH_SCENE_DISJOINT_INFERENCE_NOT_RUN"
            if mechanisms else mechanism_failure
        ),
        "mechanism_receipt_sha256": (
            mechanism_hashes[0] if len(mechanism_hashes) == 1
            else v1.canonical_sha256(mechanism_hashes)
            if mechanism_hashes else None
        ),
        "mechanism_receipt_sha256s": mechanism_hashes,
        "prospective_cost_status": "UNBOUND_TYPED_MISSING",
        "child_observable_status": "NOT_RUN_TYPED_MISSING",
        "child_observable_missing": "REGION_PROJECTED_CC_RR_CR_RC_RECEIPT",
        "capacity_selection_used": False,
        "scientific_qualification": False,
        "selector_admission": False,
        "production_authority": False,
    }
    arm = v1.MaterializedQualificationArmV1(
        qualification_action_id=action_id,
        exact_control_id=action_id,
        first_rgb=np.ascontiguousarray(outputs[0]),
        second_rgb=np.ascontiguousarray(outputs[1]),
        first_support=np.ascontiguousarray(writes[0]),
        second_support=np.ascontiguousarray(writes[1]),
    )
    return _row_hash(row), arm, local, mechanisms


def build_uncapped_qualification_matrix_v2(
    observed_first_rgb: np.ndarray,
    observed_second_rgb: np.ndarray,
    observed_native_flow: np.ndarray,
) -> QualificationMatrixResultV2:
    """Discover once and emit exactly six foundation policy rows."""
    _verify_sources()
    first, second, flow = v1._validate_inputs(
        observed_first_rgb, observed_second_rgb, observed_native_flow,
    )
    input_hashes_before = (
        v1._array_sha256(first), v1._array_sha256(second), v1._array_sha256(flow),
    )
    backend = v1._work_e_backend()
    controls, pair_sha256 = backend.discover_exact_controls(first, second, flow)
    controls = v1._replace_legacy_wiener_with_v2(
        controls, first, second, flow,
    )
    by_id = {}
    for control in controls:
        control_id = str(control.control_id)
        if control_id in by_id:
            raise RuntimeError(f"duplicate v2 discovery control: {control_id}")
        if str(getattr(control, "selection_status", "NOT_RANKED")) != "NOT_RANKED":
            raise RuntimeError("qualification v2 forbids ranked controls")
        by_id[control_id] = control
    expected = {
        "paired_impulse_median3",
        WIENER_ACTION_ID,
        "blur_v3_cross_endpoint",
        "jpeg_qcell_v3",
        "jpeg_codec_path_v4",
        "radiometry_rank3_diagnostic",
    }
    if set(by_id) != expected:
        raise RuntimeError(
            f"qualification v2 discovery universe drift: {sorted(by_id)}"
        )
    diagnostic = by_id["radiometry_rank3_diagnostic"]
    if bool(diagnostic.eligible) or diagnostic.proposal is not None:
        raise RuntimeError("qualification v2 diagnostic crossed execution boundary")
    if pair_sha256 != v1._physical_pair_sha256(first, second, flow):
        raise RuntimeError("qualification v2 physical-pair binding drift")

    rows: list[dict[str, Any]] = []
    arms: list[v1.MaterializedQualificationArmV1] = []
    local_receipts: list[Any] = []
    mechanism_receipts: list[Any] = []
    for control_id, action_id, endpoint_type in (
        ("paired_impulse_median3", IMPULSE_ACTION_ID, "case_selected_supported"),
        (WIENER_ACTION_ID, WIENER_ACTION_ID, "case_selected_supported"),
    ):
        control = by_id[control_id]
        if bool(control.eligible):
            row, arm, local, mechanisms = v1._eligible_row(
                control=control,
                action_id=action_id,
                endpoint_type=endpoint_type,
                pair=(first, second),
                flow=flow,
                physical_pair_sha256=pair_sha256,
                materialize=backend.materialize_control,
            )
            rows.append(_rebind_v1_row(row))
            arms.append(arm)
            if local is not None:
                local_receipts.append(local)
            mechanism_receipts.extend(mechanisms)
        else:
            rows.append(_rebind_v1_row(
                v1._missing_row(control, action_id, endpoint_type)
            ))

    blur_rows, blur_arms, blur_locals, blur_mechanisms = _blur_policy_rows(
        first, second, flow,
        pair_sha256=pair_sha256,
        legacy_control=by_id["blur_v3_cross_endpoint"],
    )
    rows.extend(blur_rows)
    arms.extend(blur_arms)
    local_receipts.extend(blur_locals)
    mechanism_receipts.extend(blur_mechanisms)

    for policy, action_id, legacy_id in (
        ("qcell", LOCAL_JPEG_QCELL_ACTION_ID_V3, "jpeg_qcell_v3"),
        ("codec", LOCAL_JPEG_CODEC_ACTION_ID_V3, "jpeg_codec_path_v4"),
    ):
        row, arm, local, mechanisms = _jpeg_policy_row(
            policy=policy,
            action_id=action_id,
            control=by_id[legacy_id],
            first=first,
            second=second,
            flow=flow,
            pair_sha256=pair_sha256,
        )
        rows.append(row)
        if arm is not None:
            arms.append(arm)
        if local is not None:
            local_receipts.append(local)
        mechanism_receipts.extend(mechanisms)

    if tuple(row["qualification_action_id"] for row in rows) != _POLICY_ORDER:
        raise RuntimeError("qualification v2 policy order drift")
    blur_executed = sum(
        row["qualification_status"] == "ELIGIBLE_EXECUTED"
        for row in rows[2:4]
    )
    if blur_executed > 1:
        raise RuntimeError("qualification v2 executed multiple blur public arms")
    if input_hashes_before != (
        v1._array_sha256(first), v1._array_sha256(second), v1._array_sha256(flow),
    ):
        raise RuntimeError("qualification v2 mutated observed inputs")
    eligible = sum(
        row["qualification_status"] == "ELIGIBLE_EXECUTED" for row in rows
    )
    receipt = {
        "schema": QUALIFICATION_MATRIX_SCHEMA_V2,
        "case_id": f"physical-pair:{pair_sha256}",
        "physical_pair_sha256": pair_sha256,
        "source_input_sha256s": list(input_hashes_before),
        "action_bank_sha256": _BANK_SHA256,
        "qualification_policy_order": list(_POLICY_ORDER),
        "endpoint_policy_sha256": v1.ENDPOINT_POLICY_SHA256_V1,
        "execution_surface": (
            "DISCOVER_ONCE;BLUR_CERTIFICATE_ONCE;MATERIALIZE_EVERY_ELIGIBLE_"
            "PUBLIC_ARM_AT_STRENGTH_ONE"
        ),
        "discovery_call_count": 1,
        "blur_certificate_call_count": 1,
        "blur_public_arm_executed_count": blur_executed,
        "selector_capacity": None,
        "capacity_selection_used": False,
        "ranking_invoked": False,
        "ground_truth_fields_read": [],
        "outcome_fields_read": [],
        "native_reference": {
            "first_sha256": input_hashes_before[0],
            "second_sha256": input_hashes_before[1],
            "native_flow_sha256": input_hashes_before[2],
        },
        "diagnostic": {
            "control_id": "radiometry_rank3_diagnostic",
            "authorization": str(diagnostic.authorization),
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
            "prospective_cost_receipts_ready": 0,
            "region_projected_child_quartets_ready": 0,
        },
        "rows": rows,
        "authority": {
            "execution_receipt": True,
            "scientific_qualification": False,
            "selector_admission": False,
            "production": False,
        },
    }
    receipt["receipt_sha256"] = v1.canonical_sha256(receipt)
    _verify_sources()
    return QualificationMatrixResultV2(
        receipt=receipt,
        arms=tuple(arms),
        local_case_receipts=tuple(local_receipts),
        mechanism_receipts=tuple(mechanism_receipts),
    )


__all__ = [
    "IMPULSE_ACTION_ID",
    "ISOTROPIC_ACTION_ID",
    "MOTION_ACTION_ID",
    "QUALIFICATION_MATRIX_SCHEMA_V2",
    "QUALIFICATION_POLICY_ROW_SCHEMA_V2",
    "QUALIFICATION_STRENGTH_V2",
    "QualificationMatrixResultV2",
    "WIENER_ACTION_ID",
    "build_uncapped_qualification_matrix_v2",
]
