from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from stablebridge.physical_repair.child_observables import (
    ActionBindingV1,
    ChildObservableAvailabilityV1,
    QUARTET_BRANCH_NAMES_V1,
    canonical_sha256,
)
from stablebridge.physical_repair.child_observable_producer_v1 import (
    FULL_AUDIT_POLICY_ID_V1,
    array_sha256,
    build_fixed_grid_region_set_v1,
    build_full_audit_observer_policy_v1,
    produce_complete_child_observable_v1,
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def action() -> ActionBindingV1:
    return ActionBindingV1(
        bank_id="bank-v2",
        bank_sha256=sha("bank"),
        action_id="action-v1",
        mechanism_id="mechanism-v1",
        operator_id="operator-v1",
        control_id="control-v1",
        endpoint="first",
        strength=1.0,
        exact_parameter_sha256=sha("parameter"),
        support_policy_id="support-v1",
        support_sha256=sha("support"),
        composition_id="composition-v1",
        action_descriptor_sha256=sha("descriptor"),
    )


def pair(offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    base = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    first = (base.astype(np.uint16) + offset).clip(0, 255).astype(np.uint8)
    second = np.roll(first, 1, axis=1)
    return first, second


def outputs(offset: float = 0.0):
    rows = {}
    for index, branch in enumerate(QUARTET_BRANCH_NAMES_V1):
        flow = np.zeros((8, 10, 2), dtype=np.float32)
        flow[..., 0] = np.float32(index + offset)
        flow[..., 1] = np.float32(offset / 2)
        risk = np.linspace(0, 1, 80, dtype=np.float32).reshape(8, 10)
        risk = risk + np.float32(index / 10 + offset)
        info = np.stack(
            [risk + np.float32(channel / 100) for channel in range(4)], axis=0,
        ).astype(np.float32)
        rows[branch] = {"flow": flow, "risk": risk, "info": info}
    return rows


def produce():
    regions, labels = build_fixed_grid_region_set_v1(8, 10, tile_px=4)
    root = outputs(0.0)
    parent = outputs(0.25)
    child = outputs(0.5)
    value = produce_complete_child_observable_v1(
        observation_id="case/action/child",
        action=action(),
        root_state_id="root",
        parent_state_id="parent",
        child_state_id="child",
        root_pair=pair(0),
        parent_pair=pair(1),
        child_pair=pair(2),
        root_outputs=root,
        parent_outputs=parent,
        child_outputs=child,
        region_set=regions,
        label_map=labels,
        matcher_backend_receipt_sha256=sha("backend"),
        action_execution_receipt_sha256=sha("execution"),
        cost_receipt_sha256=sha("cost"),
        source_evidence_sha256s=(sha("forward receipts"),),
    )
    return value, regions, labels


def test_complete_producer_emits_four_branches_and_all_regions():
    value, regions, labels = produce()
    receipt = value.receipt

    assert receipt.complete is True
    assert receipt.availability is ChildObservableAvailabilityV1.AVAILABLE
    assert tuple(row.branch.value for row in receipt.branches) == (
        "CC", "RR", "CR", "RC",
    )
    assert all(len(row.projections) == len(regions.regions) for row in receipt.branches)
    assert receipt.observer_policy.policy_id == FULL_AUDIT_POLICY_ID_V1
    assert value.feature_payload["executed_level"] == "L2_FULL_AUDIT"
    assert value.feature_payload_sha256 == canonical_sha256(value.feature_payload)
    assert value.feature_record()["feature_payload_sha256"] == value.feature_payload_sha256
    assert array_sha256(labels) == regions.label_map_sha256
    assert all(
        value.feature_payload_sha256 in row.source_evidence_sha256s
        for row in receipt.branches
    )


def test_metrics_are_separately_parent_and_root_relative_with_fixed_denominator():
    value, regions, _ = produce()
    first = value.feature_payload["projections"][0]

    assert first["parent_relative"]["fixed_denominator_px"] == regions.regions[0].area_px
    assert first["root_relative"]["fixed_denominator_px"] == regions.regions[0].area_px
    assert first["parent_relative_metrics_sha256"] != first["root_relative_metrics_sha256"]
    assert first["parent_relative"]["flow_change_px"]["mean"] == pytest.approx(
        np.hypot(0.25, 0.125)
    )
    assert first["root_relative"]["flow_change_px"]["mean"] == pytest.approx(
        np.hypot(0.5, 0.25)
    )


def test_producer_is_deterministic_and_child_change_changes_identity():
    first, regions, labels = produce()
    second, _, _ = produce()
    assert first.receipt.receipt_sha256 == second.receipt.receipt_sha256
    assert first.feature_payload_sha256 == second.feature_payload_sha256

    changed = outputs(0.75)
    third = produce_complete_child_observable_v1(
        observation_id="case/action/child",
        action=action(),
        root_state_id="root",
        parent_state_id="parent",
        child_state_id="child",
        root_pair=pair(0),
        parent_pair=pair(1),
        child_pair=pair(2),
        root_outputs=outputs(0.0),
        parent_outputs=outputs(0.25),
        child_outputs=changed,
        region_set=regions,
        label_map=labels,
        matcher_backend_receipt_sha256=sha("backend"),
        action_execution_receipt_sha256=sha("execution"),
        cost_receipt_sha256=sha("cost"),
        source_evidence_sha256s=(sha("forward receipts"),),
    )
    assert third.feature_payload_sha256 != first.feature_payload_sha256
    assert third.receipt.receipt_sha256 != first.receipt.receipt_sha256


def test_incomplete_nonfinite_or_region_drift_fails_closed():
    _, regions, labels = produce()
    incomplete = outputs()
    incomplete.pop("RC")
    kwargs = dict(
        observation_id="case/action/child",
        action=action(),
        root_state_id="root",
        parent_state_id="parent",
        child_state_id="child",
        root_pair=pair(),
        parent_pair=pair(1),
        child_pair=pair(2),
        root_outputs=outputs(),
        parent_outputs=outputs(0.25),
        child_outputs=incomplete,
        region_set=regions,
        label_map=labels,
        matcher_backend_receipt_sha256=sha("backend"),
        action_execution_receipt_sha256=sha("execution"),
        cost_receipt_sha256=sha("cost"),
    )
    with pytest.raises(ValueError, match="complete CC/RR/CR/RC"):
        produce_complete_child_observable_v1(**kwargs)

    bad = outputs(0.5)
    bad["CC"]["risk"][0, 0] = np.nan
    kwargs["child_outputs"] = bad
    with pytest.raises(ValueError, match="invalid child/CC"):
        produce_complete_child_observable_v1(**kwargs)

    kwargs["child_outputs"] = outputs(0.5)
    drifted = labels.copy()
    drifted[0, 0] = 1
    kwargs["label_map"] = drifted
    with pytest.raises(ValueError, match="content drifted"):
        produce_complete_child_observable_v1(**kwargs)


def test_only_frozen_full_audit_policy_is_accepted():
    _, regions, labels = produce()
    policy = build_full_audit_observer_policy_v1()
    drifted = replace(policy, policy_id="different-policy", policy_sha256="")
    with pytest.raises(ValueError, match="frozen full-audit"):
        produce_complete_child_observable_v1(
            observation_id="case/action/child",
            action=action(),
            root_state_id="root",
            parent_state_id="parent",
            child_state_id="child",
            root_pair=pair(),
            parent_pair=pair(1),
            child_pair=pair(2),
            root_outputs=outputs(),
            parent_outputs=outputs(0.25),
            child_outputs=outputs(0.5),
            region_set=regions,
            label_map=labels,
            matcher_backend_receipt_sha256=sha("backend"),
            action_execution_receipt_sha256=sha("execution"),
            cost_receipt_sha256=sha("cost"),
            observer_policy=drifted,
        )
