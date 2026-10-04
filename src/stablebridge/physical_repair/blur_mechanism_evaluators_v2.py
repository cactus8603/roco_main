"""Outcome-blind mechanism receipts for the E261 two-arm blur successor.

Disk and Gaussian are internal modes of one isotropic public arm.  Their
matched-second-moment comparison is retained as a diagnostic, not as an arm
rejection rule.  Public-arm specificity compares isotropic against motion and
motion against both isotropic modes; motion also records an orthogonal-angle
control.

This evaluator reads no task GT or action outcome and grants no scientific,
selector, cost, or production authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

import numpy as np

from .blur_mechanism_evaluators import (
    _array_sha256,
    _canonical_sha256,
    _kernel_second_moment,
    _negative_control,
    _tile_discrepancies,
    _validate,
    _warp_second,
)
from .blur_parameter_certificates import TILE, _kernel
from .blur_parameter_certificates_v6 import ISOTROPIC_ARM_V6, MOTION_ARM_V6
from .local_blur_successor import _flow_support, _proposal
from .local_blur_successor_v3 import (
    LOCAL_BLUR_ACTION_IDS_V3,
    LOCAL_BLUR_OPERATOR_IDS_V3,
    LocalBlurSuccessorV3,
    _read_halo,
)
from .operators import blend_local_proposal
from .support import transport_flow_mask_to_second


BLUR_MECHANISM_RECEIPT_SCHEMA_V2 = "stablebridge-blur-mechanism-evaluation/v2"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_METRICS = {
    ISOTROPIC_ARM_V6: "isotropic_psf_endpoint_discrepancy_reduction_v2",
    MOTION_ARM_V6: "motion_psf_endpoint_discrepancy_reduction_v2",
}


def _require_sha(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _closest_disk(moment: float) -> tuple[float, ...]:
    _, radius = min(
        (
            abs(_kernel_second_moment(_kernel("disk", (float(radius),))) - moment),
            radius,
        )
        for radius in range(1, 9)
    )
    return (float(radius),)


def _matched_gaussian(moment: float) -> tuple[float, ...]:
    return (math.sqrt(max(moment, 1e-12) / 2.0),)


def _closest_motion_length(moment: float) -> float:
    _, length = min(
        (
            abs(_kernel_second_moment(
                _kernel("motion", (float(length), 0.0))
            ) - moment),
            length,
        )
        for length in (3, 7, 11, 15, 17)
    )
    return float(length)


def _control_specs(
    public_arm: str,
    internal_family: str,
    parameter: tuple[float, ...],
) -> tuple[dict[str, Any], ...]:
    moment = _kernel_second_moment(_kernel(internal_family, parameter))
    rows: list[dict[str, Any]] = []
    if public_arm == ISOTROPIC_ARM_V6:
        diagnostic_family, diagnostic_parameter, match = _negative_control(
            internal_family, parameter,
        )
        rows.append({
            "role": "SAME_PUBLIC_ARM_ISOTROPIC_MODE_DIAGNOSTIC",
            "family": diagnostic_family,
            "parameter": diagnostic_parameter,
            "construction": match,
        })
        length = _closest_motion_length(moment)
        for angle in range(0, 180, 30):
            rows.append({
                "role": "WRONG_PUBLIC_ARM_MOTION",
                "family": "motion",
                "parameter": (length, float(angle)),
                "construction": {
                    "rule": "CLOSEST_MOTION_SECOND_MOMENT_ALL_COARSE_ANGLES",
                    "selected_second_moment_px2": moment,
                    "control_second_moment_px2": _kernel_second_moment(
                        _kernel("motion", (length, float(angle)))
                    ),
                },
            })
    elif public_arm == MOTION_ARM_V6:
        for family, control_parameter in (
            ("disk", _closest_disk(moment)),
            ("gaussian", _matched_gaussian(moment)),
        ):
            rows.append({
                "role": "WRONG_PUBLIC_ARM_ISOTROPIC",
                "family": family,
                "parameter": control_parameter,
                "construction": {
                    "rule": "MATCH_SELECTED_MOTION_SECOND_MOMENT",
                    "selected_second_moment_px2": moment,
                    "control_second_moment_px2": _kernel_second_moment(
                        _kernel(family, control_parameter)
                    ),
                },
            })
        orthogonal = (parameter[0], float((parameter[1] + 90.0) % 180.0))
        rows.append({
            "role": "SAME_PUBLIC_ARM_ORTHOGONAL_MOTION_DIAGNOSTIC",
            "family": "motion",
            "parameter": orthogonal,
            "construction": {
                "rule": "SAME_LENGTH_ORTHOGONAL_ORIENTATION",
                "selected_second_moment_px2": moment,
                "control_second_moment_px2": _kernel_second_moment(
                    _kernel("motion", orthogonal)
                ),
            },
        })
    else:
        raise ValueError("unknown v2 blur public arm")
    return tuple(rows)


@dataclass(frozen=True)
class BlurMechanismEvaluationReceiptV2:
    action_id: str
    operator_id: str
    public_arm: str
    internal_family: str
    metric_id: str
    physical_pair_sha256: str
    winner_id: str
    source_action_key: str
    fit_support_sha256: str
    check_support_sha256: str
    identified_endpoint: str
    modified_endpoint: str
    selected_parameter: tuple[float, ...]
    endpoint_records: tuple[dict[str, Any], ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.public_arm not in _METRICS:
            raise ValueError("unknown blur public arm")
        if self.action_id != LOCAL_BLUR_ACTION_IDS_V3[self.public_arm]:
            raise ValueError("blur v2 mechanism action identity drift")
        if self.operator_id != LOCAL_BLUR_OPERATOR_IDS_V3[self.public_arm]:
            raise ValueError("blur v2 mechanism operator identity drift")
        if self.metric_id != _METRICS[self.public_arm]:
            raise ValueError("blur v2 mechanism metric identity drift")
        if self.winner_id != f"{self.public_arm}@{self.identified_endpoint}":
            raise ValueError("blur v2 mechanism winner identity drift")
        if self.source_action_key != (
            f"common_{self.internal_family}@{self.identified_endpoint}"
        ):
            raise ValueError("blur v2 mechanism source lineage drift")
        if self.public_arm == ISOTROPIC_ARM_V6:
            if self.internal_family not in {"disk", "gaussian"}:
                raise ValueError("isotropic mechanism needs disk/Gaussian mode")
        elif self.internal_family != "motion":
            raise ValueError("motion mechanism needs the motion mode")
        expected = "second" if self.identified_endpoint == "first" else "first"
        if self.modified_endpoint != expected:
            raise ValueError("blur v2 mechanism endpoint direction drift")
        for name in (
            "physical_pair_sha256", "fit_support_sha256", "check_support_sha256",
        ):
            _require_sha(getattr(self, name), name)
        if self.fit_support_sha256 == self.check_support_sha256:
            raise ValueError("blur v2 mechanism A/B supports must differ")
        if len(self.endpoint_records) != 1:
            raise ValueError("blur v2 mechanism needs one endpoint record")
        expected_hash = _canonical_sha256(self.as_dict(include_hash=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected_hash:
            raise ValueError("blur v2 mechanism receipt hash drift")
        object.__setattr__(self, "receipt_sha256", expected_hash)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema": BLUR_MECHANISM_RECEIPT_SCHEMA_V2,
            "action_id": self.action_id,
            "operator_id": self.operator_id,
            "public_arm": self.public_arm,
            "internal_family": self.internal_family,
            "metric_id": self.metric_id,
            "physical_pair_sha256": self.physical_pair_sha256,
            "winner_id": self.winner_id,
            "source_action_key": self.source_action_key,
            "fit_support_sha256": self.fit_support_sha256,
            "check_support_sha256": self.check_support_sha256,
            "identified_endpoint": self.identified_endpoint,
            "modified_endpoint": self.modified_endpoint,
            "selected_parameter": list(self.selected_parameter),
            "endpoint_records": list(self.endpoint_records),
            "runtime_inputs": [
                "before_pair", "typed_local_successor_v3",
                "observed_native_flow",
            ],
            "ground_truth_read": False,
            "task_outcome_read": False,
            "scientific_qualification": False,
            "selector_admission": False,
            "production_authority": False,
        }
        if include_hash:
            result["receipt_sha256"] = self.receipt_sha256
        return result


def evaluate_blur_mechanism_v2(
    *,
    physical_pair_sha256: str,
    before_pair: Sequence[np.ndarray],
    successor: LocalBlurSuccessorV3,
    observed_native_flow: np.ndarray,
) -> BlurMechanismEvaluationReceiptV2:
    """Evaluate active closure and public-arm negative controls."""
    _require_sha(physical_pair_sha256, "physical pair")
    if successor.status != "EXECUTED_LOCAL_RECOVERABLE_V3":
        raise ValueError("blur v2 mechanism needs an executed local successor v3")
    if any(value is None for value in (
        successor.action_id,
        successor.operator_id,
        successor.public_arm,
        successor.winner_id,
        successor.internal_family,
        successor.identified_endpoint,
        successor.selected_parameter,
    )):
        raise ValueError("executed local successor v3 is incomplete")
    after_pair = (successor.first_rgb, successor.second_rgb)
    supports = (successor.first_support, successor.second_support)
    before, after, supports, flow = _validate(
        before_pair, after_pair, supports, observed_native_flow,
    )
    public_arm = str(successor.public_arm)
    internal_family = str(successor.internal_family)
    identified = str(successor.identified_endpoint)
    parameter = tuple(map(float, successor.selected_parameter or ()))
    regions = tuple(successor.recoverable_regions)
    if not parameter or len(regions) < 4:
        raise RuntimeError("blur v2 mechanism lacks parameter or local regions")

    flow_weight = _flow_support(before[0].shape[:2], regions)
    if identified == "first":
        modified = "second"
        endpoint_weight, _ = transport_flow_mask_to_second(flow_weight, flow)
        expected_supports = (
            np.zeros(flow_weight.shape, dtype=bool), endpoint_weight > 0.0,
        )
        modified_index = 1
    elif identified == "second":
        modified = "first"
        endpoint_weight = flow_weight
        expected_supports = (
            endpoint_weight > 0.0, np.zeros(flow_weight.shape, dtype=bool),
        )
        modified_index = 0
    else:
        raise ValueError("invalid blur v2 identified endpoint")
    if any(
        not np.array_equal(actual, expected)
        for actual, expected in zip(supports, expected_supports)
    ):
        raise RuntimeError("blur v2 mechanism endpoint support drift")
    if not np.array_equal(before[1 - modified_index], after[1 - modified_index]):
        raise RuntimeError("blur v2 mechanism changed its identified endpoint")
    if not np.array_equal(
        before[modified_index][~supports[modified_index]],
        after[modified_index][~supports[modified_index]],
    ):
        raise RuntimeError("blur v2 mechanism output escaped support")
    kernel = _kernel(internal_family, parameter)
    expected_reads = [
        np.zeros(flow_weight.shape, dtype=bool),
        np.zeros(flow_weight.shape, dtype=bool),
    ]
    expected_reads[modified_index] = _read_halo(
        supports[modified_index], kernel,
    )
    actual_reads = (successor.first_read_support, successor.second_read_support)
    if any(
        not np.array_equal(actual, expected)
        for actual, expected in zip(actual_reads, expected_reads)
    ):
        raise RuntimeError("blur v2 mechanism read-halo drift")

    warped_native_second, valid = _warp_second(before[1], flow)
    warped_action_second, valid_action = _warp_second(after[1], flow)
    if not np.array_equal(valid, valid_action):
        raise RuntimeError("blur v2 mechanism action warp-validity drift")

    controls = []
    for spec in _control_specs(public_arm, internal_family, parameter):
        control_modified, record = blend_local_proposal(
            before[modified_index],
            _proposal(
                before[modified_index], spec["family"], spec["parameter"]
            ),
            endpoint_weight,
            operator_id=f"{successor.operator_id}.control.{spec['role'].lower()}",
            endpoint=modified,
            feather_sigma=1.0,
        )
        pair = [before[0].copy(), before[1].copy()]
        pair[modified_index] = control_modified
        warped_control_second, valid_control = _warp_second(pair[1], flow)
        if not np.array_equal(valid, valid_control):
            raise RuntimeError("blur v2 mechanism control warp-validity drift")
        controls.append((spec, pair, warped_control_second, record))

    region_rows = []
    for y0, x0 in regions:
        ys, xs = slice(y0, y0 + TILE), slice(x0, x0 + TILE)
        local_valid = valid[ys, xs]
        if float(local_valid.mean()) < 0.90:
            raise RuntimeError("blur v2 mechanism region lost valid support")
        if identified == "first":
            target = before[0][ys, xs].astype(np.float32)
            native_candidate = warped_native_second[ys, xs]
            action_candidate = warped_action_second[ys, xs]
        else:
            target = warped_native_second[ys, xs]
            native_candidate = before[0][ys, xs].astype(np.float32)
            action_candidate = after[0][ys, xs].astype(np.float32)
        control_rows = []
        active_values = []
        native_values = []
        diagnostics = None
        for spec, pair, warped_control_second, record in controls:
            control_candidate = (
                warped_control_second[ys, xs]
                if identified == "first"
                else pair[0][ys, xs].astype(np.float32)
            )
            native_signal, active_signal, control_signal, current_diagnostics = (
                _tile_discrepancies(
                    target,
                    native_candidate,
                    action_candidate,
                    control_candidate,
                    local_valid,
                )
            )
            native_values.append(native_signal)
            active_values.append(active_signal)
            diagnostics = current_diagnostics
            control_rows.append({
                "role": spec["role"],
                "family": spec["family"],
                "parameter": list(spec["parameter"]),
                "signal": control_signal,
                "construction": spec["construction"],
                "local_action": {
                    "operator_id": record.operator_id,
                    "endpoint": record.endpoint,
                    "support_fraction": record.support_fraction,
                    "changed_fraction": record.changed_fraction,
                },
            })
        if max(active_values) - min(active_values) > 1e-12:
            raise RuntimeError("blur v2 active signal changed across controls")
        if max(native_values) - min(native_values) > 1e-12:
            raise RuntimeError("blur v2 native signal changed across controls")
        region_rows.append({
            "region_y0_x0": [y0, x0],
            "native_signal": native_values[0],
            "action_signal": active_values[0],
            "control_records": control_rows,
            **(diagnostics or {}),
        })

    before_signal = float(np.mean([row["native_signal"] for row in region_rows]))
    after_signal = float(np.mean([row["action_signal"] for row in region_rows]))
    if before_signal <= 1e-12 or not all(map(
        math.isfinite, (before_signal, after_signal),
    )):
        raise RuntimeError("blur v2 mechanism active signal is unestimable")
    active_reduction = float((before_signal - after_signal) / before_signal)
    control_records = []
    roles = sorted({
        item["role"]
        for row in region_rows for item in row["control_records"]
    })
    for role in roles:
        keyed = {}
        for row in region_rows:
            for item in row["control_records"]:
                if item["role"] == role:
                    key = (item["family"], tuple(item["parameter"]))
                    keyed.setdefault(key, []).append(item["signal"])
        for (family, control_parameter), signals in sorted(keyed.items()):
            signal = float(np.mean(signals))
            reduction = float((before_signal - signal) / before_signal)
            control_records.append({
                "role": role,
                "family": family,
                "parameter": list(control_parameter),
                "signal_after": signal,
                "relative_reduction": reduction,
            })
    wrong_public = [
        row["relative_reduction"] for row in control_records
        if str(row["role"]).startswith("WRONG_PUBLIC_ARM_")
    ]
    if not wrong_public:
        raise RuntimeError("blur v2 mechanism lacks a wrong-public-arm control")
    max_wrong = float(max(wrong_public))
    endpoint_record = {
        "endpoint": modified,
        "identified_endpoint": identified,
        "support_sha256": _array_sha256(supports[modified_index]),
        "support_pixels": int(supports[modified_index].sum()),
        "read_halo_sha256": _array_sha256(actual_reads[modified_index]),
        "read_halo_pixels": int(actual_reads[modified_index].sum()),
        "recoverable_region_count": len(regions),
        "active_signal_before": before_signal,
        "active_signal_after": after_signal,
        "active_relative_reduction": active_reduction,
        "wrong_public_arm_max_relative_reduction": max_wrong,
        "public_arm_specificity_difference": active_reduction - max_wrong,
        "control_records": control_records,
        "outside_support_byte_identity": True,
        "region_records": region_rows,
    }
    receipt = successor.receipt
    return BlurMechanismEvaluationReceiptV2(
        action_id=str(successor.action_id),
        operator_id=str(successor.operator_id),
        public_arm=public_arm,
        internal_family=internal_family,
        metric_id=_METRICS[public_arm],
        physical_pair_sha256=physical_pair_sha256,
        winner_id=str(successor.winner_id),
        source_action_key=str(receipt["source_action_key"]),
        fit_support_sha256=str(receipt["fit_support_sha256"]),
        check_support_sha256=str(receipt["check_support_sha256"]),
        identified_endpoint=identified,
        modified_endpoint=modified,
        selected_parameter=parameter,
        endpoint_records=(endpoint_record,),
    )


__all__ = [
    "BLUR_MECHANISM_RECEIPT_SCHEMA_V2",
    "BlurMechanismEvaluationReceiptV2",
    "evaluate_blur_mechanism_v2",
]
