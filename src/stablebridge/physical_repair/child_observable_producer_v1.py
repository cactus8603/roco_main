"""Produce complete region-projected quartet evidence for one exact child.

``child_observables`` is deliberately only a receipt contract.  This module is
the outcome-blind numeric producer that closes that contract from actual
CC/RR/CR/RC matcher arrays.  It never invokes an action, a matcher, a selector,
or ground truth; the caller must supply three immutable states and the measured
execution/cost bindings.

The feature payload returned here is intended to be written as a sidecar.  The
receipt stores only content hashes, so numeric zero can never be confused with
an observation that was not made.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .child_observables import (
    ActionBindingV1,
    BranchChildObservableV1,
    ChildObservableAvailabilityV1,
    ChildObservableReceiptV1,
    ChildObserverPolicyV1,
    ObservableStateBindingV1,
    ObserverLevelPolicyV1,
    ObserverLevelV1,
    QUARTET_BRANCHES_V1,
    QuartetBranchV1,
    RegionIdentityV1,
    RegionProjectionBindingV1,
    RegionSetBindingV1,
    canonical_sha256,
)


CHILD_OBSERVABLE_FEATURE_SCHEMA_V1 = (
    "stablebridge-region-projected-child-feature-payload/v1"
)
FULL_AUDIT_POLICY_ID_V1 = "qualification_full_quartet_fixed_regions_v1"
FIXED_GRID_REGION_POLICY_ID_V1 = "fixed_grid_row_major_v1"
_REQUIRED_ROLES = ("flow", "risk", "info")


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and contiguous bytes without lossy conversion."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _require_rgb_pair(
    pair: Sequence[np.ndarray], *, name: str, expected_hw: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pair) != 2:
        raise ValueError(f"{name} must contain two RGB endpoints")
    first, second = (np.asarray(value) for value in pair)
    if (
        first.dtype != np.uint8
        or second.dtype != np.uint8
        or first.ndim != 3
        or first.shape != second.shape
        or first.shape[2] != 3
    ):
        raise ValueError(f"{name} must contain equal uint8 HxWx3 RGB endpoints")
    if expected_hw is not None and first.shape[:2] != expected_hw:
        raise ValueError(f"{name} lattice does not match the frozen region set")
    return np.ascontiguousarray(first), np.ascontiguousarray(second)


def _branch_key(value: object) -> QuartetBranchV1:
    if isinstance(value, QuartetBranchV1):
        return value
    try:
        return QuartetBranchV1(str(value))
    except ValueError as exc:
        raise ValueError(f"unknown quartet branch: {value}") from exc


def _normalize_outputs(
    outputs: Mapping[object, Mapping[str, np.ndarray]],
    *,
    hw: tuple[int, int],
    state_name: str,
) -> dict[QuartetBranchV1, dict[str, np.ndarray]]:
    normalized: dict[QuartetBranchV1, dict[str, np.ndarray]] = {}
    for raw_branch, raw_roles in outputs.items():
        branch = _branch_key(raw_branch)
        if branch in normalized:
            raise ValueError(f"duplicate {state_name} quartet branch")
        if set(raw_roles) != set(_REQUIRED_ROLES):
            raise ValueError(
                f"{state_name}/{branch.value} must contain flow/risk/info"
            )
        flow = np.asarray(raw_roles["flow"])
        risk = np.asarray(raw_roles["risk"])
        info = np.asarray(raw_roles["info"])
        if (
            flow.dtype != np.float32
            or risk.dtype != np.float32
            or info.dtype != np.float32
            or flow.shape != (*hw, 2)
            or risk.shape != hw
            or info.ndim != 3
            or info.shape[1:] != hw
            or info.shape[0] < 1
            or not np.isfinite(flow).all()
            or not np.isfinite(risk).all()
            or not np.isfinite(info).all()
        ):
            raise ValueError(f"invalid {state_name}/{branch.value} matcher arrays")
        normalized[branch] = {
            "flow": np.ascontiguousarray(flow),
            "risk": np.ascontiguousarray(risk),
            "info": np.ascontiguousarray(info),
        }
    if tuple(normalized) != QUARTET_BRANCHES_V1:
        # Mapping insertion order is not evidence.  Accept every complete set,
        # then canonicalize to the frozen quartet order below.
        if set(normalized) != set(QUARTET_BRANCHES_V1):
            raise ValueError(f"{state_name} needs complete CC/RR/CR/RC outputs")
    return {branch: normalized[branch] for branch in QUARTET_BRANCHES_V1}


def _observation_record(
    outputs: Mapping[QuartetBranchV1, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    branches = []
    for branch in QUARTET_BRANCHES_V1:
        arrays = outputs[branch]
        branch_record = {
            "branch": branch.value,
            "arrays": {
                role: {
                    "sha256": array_sha256(arrays[role]),
                    "dtype": str(arrays[role].dtype),
                    "shape": list(arrays[role].shape),
                }
                for role in _REQUIRED_ROLES
            },
        }
        branches.append({
            **branch_record,
            "observation_sha256": canonical_sha256(branch_record),
        })
    record = {
        "schema": "stablebridge-quartet-observation-binding/v1",
        "branches": branches,
    }
    return {**record, "observation_sha256": canonical_sha256(record)}


def _state_binding(
    *,
    state_id: str,
    pair: tuple[np.ndarray, np.ndarray],
    observation: Mapping[str, Any],
) -> ObservableStateBindingV1:
    return ObservableStateBindingV1(
        state_id=state_id,
        input_sha256=canonical_sha256({
            "first_rgb_sha256": array_sha256(pair[0]),
            "second_rgb_sha256": array_sha256(pair[1]),
        }),
        output_sha256=str(observation["observation_sha256"]),
    )


def build_fixed_grid_region_set_v1(
    height: int,
    width: int,
    *,
    tile_px: int = 64,
    region_set_id: str | None = None,
) -> tuple[RegionSetBindingV1, np.ndarray]:
    """Build the exact row-major region lattice used by qualification."""

    for value, name in ((height, "height"), (width, "width"), (tile_px, "tile_px")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    labels = np.empty((height, width), dtype=np.int32)
    regions: list[RegionIdentityV1] = []
    ordinal = 0
    for y0 in range(0, height, tile_px):
        for x0 in range(0, width, tile_px):
            y1, x1 = min(height, y0 + tile_px), min(width, x0 + tile_px)
            labels[y0:y1, x0:x1] = ordinal
            regions.append(RegionIdentityV1(
                region_id=f"r{ordinal:06d}",
                bbox_xyxy=(x0, y0, x1, y1),
                area_px=(y1 - y0) * (x1 - x0),
            ))
            ordinal += 1
    policy_id = f"{FIXED_GRID_REGION_POLICY_ID_V1}/{tile_px}px/partial_edges"
    resolved_set_id = region_set_id or f"grid:{height}x{width}:{tile_px}px"
    return (
        RegionSetBindingV1(
            region_set_id=resolved_set_id,
            region_policy_id=policy_id,
            label_map_sha256=array_sha256(labels),
            regions=tuple(regions),
        ),
        np.ascontiguousarray(labels),
    )


def build_full_audit_observer_policy_v1() -> ChildObserverPolicyV1:
    """Freeze the three-level policy while qualification executes L2 in full."""

    definitions = (
        (
            ObserverLevelV1.L0_SENTINEL,
            "FULL_FRAME_CHEAP",
            "risk_and_flow_global_sentinel_v1",
        ),
        (
            ObserverLevelV1.L1_FOCUSED,
            "FOCUSED_UNION",
            "write_read_uncertainty_union_v1",
        ),
        (
            ObserverLevelV1.L2_FULL_AUDIT,
            "FULL_FRAME_HIGH_RESOLUTION",
            "every_frozen_region_every_quartet_branch_v1",
        ),
    )
    levels = []
    feature_contract = canonical_sha256({
        "schema": CHILD_OBSERVABLE_FEATURE_SCHEMA_V1,
        "roles": list(_REQUIRED_ROLES),
        "relative_to": ["parent", "root"],
        "fixed_denominator": True,
    })
    for level, coverage, trigger in definitions:
        levels.append(ObserverLevelPolicyV1(
            level=level,
            coverage_mode=coverage,
            coverage_policy_sha256=canonical_sha256({
                "level": level.value,
                "coverage": coverage,
            }),
            trigger_policy_id=trigger,
            trigger_policy_sha256=canonical_sha256({
                "trigger_policy_id": trigger,
                "qualification_execution": (
                    level is ObserverLevelV1.L2_FULL_AUDIT
                ),
            }),
            feature_contract_sha256=feature_contract,
        ))
    skip_id = "no_skip_during_action_level_qualification_v1"
    return ChildObserverPolicyV1(
        policy_id=FULL_AUDIT_POLICY_ID_V1,
        levels=tuple(levels),
        outside_support_skip_policy_id=skip_id,
        outside_support_skip_policy_sha256=canonical_sha256({
            "policy_id": skip_id,
            "qualification": "ALL_REGIONS_OBSERVED",
        }),
        periodic_full_audit_interval_commits=1,
    )


def _describe(value: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("feature summary needs nonempty finite values")
    return {
        "n": int(vector.size),
        "mean": float(vector.mean()),
        "std": float(vector.std()),
        "min": float(vector.min()),
        "p50": float(np.quantile(vector, 0.50)),
        "p90": float(np.quantile(vector, 0.90)),
        "p99": float(np.quantile(vector, 0.99)),
        "max": float(vector.max()),
    }


def _relative_metrics(
    *,
    branch: QuartetBranchV1,
    region: RegionIdentityV1,
    mask: np.ndarray,
    reference_name: str,
    reference: Mapping[str, np.ndarray],
    child: Mapping[str, np.ndarray],
    uncertainty_threshold: float,
) -> dict[str, Any]:
    flow_change = np.linalg.norm(
        child["flow"].astype(np.float64)
        - reference["flow"].astype(np.float64),
        axis=2,
    )
    reference_flow = np.linalg.norm(reference["flow"].astype(np.float64), axis=2)
    child_flow = np.linalg.norm(child["flow"].astype(np.float64), axis=2)
    reference_risk = reference["risk"].astype(np.float64)
    child_risk = child["risk"].astype(np.float64)
    risk_delta = child_risk - reference_risk
    info_change = np.mean(
        np.abs(
            child["info"].astype(np.float64)
            - reference["info"].astype(np.float64)
        ),
        axis=0,
    )
    uncertain_reference = reference_risk >= uncertainty_threshold
    uncertain_child = child_risk >= uncertainty_threshold
    payload = {
        "schema": "stablebridge-region-relative-quartet-metrics/v1",
        "branch": branch.value,
        "region_id": region.region_id,
        "region_identity_sha256": region.identity_sha256,
        "reference_state": reference_name,
        "fixed_denominator_px": region.area_px,
        "uncertainty_threshold": uncertainty_threshold,
        "uncertainty_threshold_source": f"{reference_name}_full_frame_p90",
        "flow_magnitude_reference_px": _describe(reference_flow[mask]),
        "flow_magnitude_child_px": _describe(child_flow[mask]),
        "flow_change_px": _describe(flow_change[mask]),
        "risk_reference": _describe(reference_risk[mask]),
        "risk_child": _describe(child_risk[mask]),
        "risk_child_minus_reference": _describe(risk_delta[mask]),
        "absolute_risk_change": _describe(np.abs(risk_delta[mask])),
        "mean_absolute_info_change": _describe(info_change[mask]),
        "uncertain_reference_px": int((uncertain_reference & mask).sum()),
        "uncertain_child_px": int((uncertain_child & mask).sum()),
        "newly_uncertain_px": int(
            (uncertain_child & ~uncertain_reference & mask).sum()
        ),
        "resolved_uncertainty_px": int(
            (uncertain_reference & ~uncertain_child & mask).sum()
        ),
    }
    return payload


@dataclass(frozen=True)
class ProducedChildObservableV1:
    """Receipt plus the external numeric feature payload it hashes."""

    receipt: ChildObservableReceiptV1
    feature_payload: Mapping[str, Any]
    feature_payload_sha256: str
    label_map: np.ndarray

    def feature_record(self) -> dict[str, Any]:
        return {
            **dict(self.feature_payload),
            "feature_payload_sha256": self.feature_payload_sha256,
        }


def produce_complete_child_observable_v1(
    *,
    observation_id: str,
    action: ActionBindingV1,
    root_state_id: str,
    parent_state_id: str,
    child_state_id: str,
    root_pair: Sequence[np.ndarray],
    parent_pair: Sequence[np.ndarray],
    child_pair: Sequence[np.ndarray],
    root_outputs: Mapping[object, Mapping[str, np.ndarray]],
    parent_outputs: Mapping[object, Mapping[str, np.ndarray]],
    child_outputs: Mapping[object, Mapping[str, np.ndarray]],
    region_set: RegionSetBindingV1,
    label_map: np.ndarray,
    matcher_backend_receipt_sha256: str,
    action_execution_receipt_sha256: str,
    cost_receipt_sha256: str,
    source_evidence_sha256s: Sequence[str] = (),
    observer_policy: ChildObserverPolicyV1 | None = None,
) -> ProducedChildObservableV1:
    """Build an all-regions, all-branches outcome-blind child receipt.

    The caller must have run every root/parent/child quartet observation.  A
    partial run is rejected here instead of being silently converted to zero.
    """

    if not isinstance(action, ActionBindingV1):
        raise ValueError("action must be an exact ActionBindingV1")
    if not isinstance(region_set, RegionSetBindingV1):
        raise ValueError("region_set must be a RegionSetBindingV1")
    labels = np.asarray(label_map)
    if labels.dtype != np.int32 or labels.ndim != 2:
        raise ValueError("label_map must be an int32 HxW array")
    labels = np.ascontiguousarray(labels)
    if array_sha256(labels) != region_set.label_map_sha256:
        raise ValueError("label_map content drifted from the frozen region set")
    hw = tuple(int(value) for value in labels.shape)
    expected_labels = set(range(len(region_set.regions)))
    if set(int(value) for value in np.unique(labels)) != expected_labels:
        raise ValueError("label_map indices do not exactly cover frozen regions")
    for ordinal, region in enumerate(region_set.regions):
        mask = labels == ordinal
        if int(mask.sum()) != region.area_px:
            raise ValueError("label_map area drifted from frozen region identity")
        y, x = np.nonzero(mask)
        bbox = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
        if bbox != region.bbox_xyxy:
            raise ValueError("label_map bounding box drifted from frozen region identity")

    root_pair_value = _require_rgb_pair(root_pair, name="root_pair", expected_hw=hw)
    parent_pair_value = _require_rgb_pair(
        parent_pair, name="parent_pair", expected_hw=hw,
    )
    child_pair_value = _require_rgb_pair(
        child_pair, name="child_pair", expected_hw=hw,
    )
    normalized = {
        "root": _normalize_outputs(root_outputs, hw=hw, state_name="root"),
        "parent": _normalize_outputs(parent_outputs, hw=hw, state_name="parent"),
        "child": _normalize_outputs(child_outputs, hw=hw, state_name="child"),
    }
    observations = {
        name: _observation_record(outputs) for name, outputs in normalized.items()
    }
    state_bindings = {
        "root": _state_binding(
            state_id=root_state_id,
            pair=root_pair_value,
            observation=observations["root"],
        ),
        "parent": _state_binding(
            state_id=parent_state_id,
            pair=parent_pair_value,
            observation=observations["parent"],
        ),
        "child": _state_binding(
            state_id=child_state_id,
            pair=child_pair_value,
            observation=observations["child"],
        ),
    }
    policy = observer_policy or build_full_audit_observer_policy_v1()
    if policy.policy_id != FULL_AUDIT_POLICY_ID_V1:
        raise ValueError("qualification producer requires the frozen full-audit policy")

    projection_payloads: list[dict[str, Any]] = []
    thresholds: dict[tuple[str, QuartetBranchV1], float] = {}
    for reference_name in ("parent", "root"):
        for branch in QUARTET_BRANCHES_V1:
            thresholds[(reference_name, branch)] = float(np.quantile(
                normalized[reference_name][branch]["risk"].astype(np.float64),
                0.90,
            ))
    for branch in QUARTET_BRANCHES_V1:
        for ordinal, region in enumerate(region_set.regions):
            mask = labels == ordinal
            parent_metrics = _relative_metrics(
                branch=branch,
                region=region,
                mask=mask,
                reference_name="parent",
                reference=normalized["parent"][branch],
                child=normalized["child"][branch],
                uncertainty_threshold=thresholds[("parent", branch)],
            )
            root_metrics = _relative_metrics(
                branch=branch,
                region=region,
                mask=mask,
                reference_name="root",
                reference=normalized["root"][branch],
                child=normalized["child"][branch],
                uncertainty_threshold=thresholds[("root", branch)],
            )
            projection_payloads.append({
                "branch": branch.value,
                "region_id": region.region_id,
                "parent_relative": parent_metrics,
                "parent_relative_metrics_sha256": canonical_sha256(parent_metrics),
                "root_relative": root_metrics,
                "root_relative_metrics_sha256": canonical_sha256(root_metrics),
            })
    implementation_sha = _source_sha256()
    feature_payload = {
        "schema": CHILD_OBSERVABLE_FEATURE_SCHEMA_V1,
        "observation_id": observation_id,
        "action_binding_sha256": action.binding_sha256,
        "region_set_sha256": region_set.region_set_sha256,
        "observer_policy_sha256": policy.policy_sha256,
        "executed_level": ObserverLevelV1.L2_FULL_AUDIT.value,
        "state_observations": observations,
        "projections": projection_payloads,
        "observer_implementation_sha256": implementation_sha,
        "GT_read": False,
        "H2_read": False,
        "outcome_read": False,
    }
    feature_hash = canonical_sha256(feature_payload)
    by_projection = {
        (row["branch"], row["region_id"]): row for row in projection_payloads
    }
    extra_sources = tuple(str(value) for value in source_evidence_sha256s)
    branch_receipts = []
    for branch in QUARTET_BRANCHES_V1:
        projections = []
        for region in region_set.regions:
            row = by_projection[(branch.value, region.region_id)]
            projections.append(RegionProjectionBindingV1(
                branch=branch,
                region_id=region.region_id,
                region_identity_sha256=region.identity_sha256,
                availability=ChildObservableAvailabilityV1.AVAILABLE,
                missing_reason=None,
                parent_relative_metrics_sha256=row[
                    "parent_relative_metrics_sha256"
                ],
                root_relative_metrics_sha256=row["root_relative_metrics_sha256"],
                conservative_bound_receipt_sha256=None,
            ))
        sources = tuple(dict.fromkeys((*extra_sources, feature_hash)))
        branch_receipts.append(BranchChildObservableV1(
            branch=branch,
            availability=ChildObservableAvailabilityV1.AVAILABLE,
            missing_reason=None,
            root_observation_sha256=next(
                row["observation_sha256"]
                for row in observations["root"]["branches"]
                if row["branch"] == branch.value
            ),
            parent_observation_sha256=next(
                row["observation_sha256"]
                for row in observations["parent"]["branches"]
                if row["branch"] == branch.value
            ),
            child_observation_sha256=next(
                row["observation_sha256"]
                for row in observations["child"]["branches"]
                if row["branch"] == branch.value
            ),
            source_evidence_sha256s=sources,
            projections=tuple(projections),
        ))
    receipt = ChildObservableReceiptV1(
        observation_id=observation_id,
        action=action,
        root=state_bindings["root"],
        parent=state_bindings["parent"],
        child=state_bindings["child"],
        region_set=region_set,
        observer_policy=policy,
        branches=tuple(branch_receipts),
        availability=ChildObservableAvailabilityV1.AVAILABLE,
        missing_reason=None,
        observer_implementation_sha256=implementation_sha,
        matcher_backend_receipt_sha256=matcher_backend_receipt_sha256,
        action_execution_receipt_sha256=action_execution_receipt_sha256,
        cost_account="PROSPECTIVE_RUNTIME",
        cost_receipt_sha256=cost_receipt_sha256,
    )
    return ProducedChildObservableV1(
        receipt=receipt,
        feature_payload=feature_payload,
        feature_payload_sha256=feature_hash,
        label_map=labels,
    )


__all__ = [
    "CHILD_OBSERVABLE_FEATURE_SCHEMA_V1",
    "FIXED_GRID_REGION_POLICY_ID_V1",
    "FULL_AUDIT_POLICY_ID_V1",
    "ProducedChildObservableV1",
    "array_sha256",
    "build_fixed_grid_region_set_v1",
    "build_full_audit_observer_policy_v1",
    "produce_complete_child_observable_v1",
]
