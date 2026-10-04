from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from stablebridge.physical_repair.child_observables import (
    ActionBindingV1,
    BranchChildObservableV1,
    ChildObservableAvailabilityV1,
    ChildObservableReceiptV1,
    ChildObserverPolicyV1,
    ObservableStateBindingV1,
    ObserverLevelPolicyV1,
    ObserverLevelV1,
    QUARTET_BRANCH_NAMES_V1,
    QUARTET_BRANCHES_V1,
    RegionIdentityV1,
    RegionProjectionBindingV1,
    RegionSetBindingV1,
    validate_region_projection_coverage,
)


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def action() -> ActionBindingV1:
    return ActionBindingV1(
        bank_id="BANK_V7",
        bank_sha256=sha("bank"),
        action_id="paired_additive_wiener3::full",
        mechanism_id="paired_additive_wiener3",
        operator_id="wiener3",
        control_id="control:wiener3:case-1",
        endpoint="BOTH",
        strength=1.0,
        exact_parameter_sha256=sha("parameters"),
        support_policy_id="paired-local-support/v1",
        support_sha256=sha("support"),
        composition_id="SINGLE",
        action_descriptor_sha256=sha("action descriptor"),
    )


def states():
    root = ObservableStateBindingV1(
        "root:case-1", sha("root input"), sha("root output"),
    )
    child = ObservableStateBindingV1(
        "child:case-1:action-1", sha("child input"), sha("child output"),
    )
    return root, root, child


def region_set() -> RegionSetBindingV1:
    return RegionSetBindingV1(
        region_set_id="case-1/native-grid",
        region_policy_id="fixed-grid-row-major/v1",
        label_map_sha256=sha("label map"),
        regions=(
            RegionIdentityV1("0", (0, 0, 64, 64), 4096),
            # Area can be smaller than the bounding box for an irregular region.
            RegionIdentityV1("1", (64, 0, 128, 64), 3000),
        ),
    )


def observer_policy() -> ChildObserverPolicyV1:
    coverage = {
        ObserverLevelV1.L0_SENTINEL: "FULL_FRAME_CHEAP",
        ObserverLevelV1.L1_FOCUSED: "FOCUSED_UNION",
        ObserverLevelV1.L2_FULL_AUDIT: "FULL_FRAME_HIGH_RESOLUTION",
    }
    levels = tuple(
        ObserverLevelPolicyV1(
            level=level,
            coverage_mode=coverage[level],
            coverage_policy_sha256=sha(f"coverage:{level.value}"),
            trigger_policy_id=f"trigger:{level.value}",
            trigger_policy_sha256=sha(f"trigger:{level.value}"),
            feature_contract_sha256=sha(f"features:{level.value}"),
        )
        for level in ObserverLevelV1
    )
    return ChildObserverPolicyV1(
        policy_id="child-observer-policy/v1",
        levels=levels,
        outside_support_skip_policy_id="certified-local-only/v1",
        outside_support_skip_policy_sha256=sha("outside-support policy"),
        periodic_full_audit_interval_commits=2,
    )


def projection(
    branch,
    region: RegionIdentityV1,
    *,
    available: bool = True,
) -> RegionProjectionBindingV1:
    if available:
        return RegionProjectionBindingV1(
            branch=branch,
            region_id=region.region_id,
            region_identity_sha256=region.identity_sha256,
            availability=ChildObservableAvailabilityV1.AVAILABLE,
            missing_reason=None,
            parent_relative_metrics_sha256=sha(
                f"parent metrics:{branch.value}:{region.region_id}"
            ),
            root_relative_metrics_sha256=sha(
                f"root metrics:{branch.value}:{region.region_id}"
            ),
            conservative_bound_receipt_sha256=None,
        )
    return RegionProjectionBindingV1(
        branch=branch,
        region_id=region.region_id,
        region_identity_sha256=region.identity_sha256,
        availability=ChildObservableAvailabilityV1.TYPED_MISSING,
        missing_reason="L1_NOT_RUN_CALIBRATED_BOUND_APPLIED",
        parent_relative_metrics_sha256=None,
        root_relative_metrics_sha256=None,
        conservative_bound_receipt_sha256=sha(
            f"bound:{branch.value}:{region.region_id}"
        ),
    )


def branch_receipt(
    branch,
    regions: RegionSetBindingV1,
    *,
    missing_region: str | None = None,
) -> BranchChildObservableV1:
    rows = tuple(
        projection(branch, region, available=region.region_id != missing_region)
        for region in regions.regions
    )
    available = missing_region is None
    return BranchChildObservableV1(
        branch=branch,
        availability=(
            ChildObservableAvailabilityV1.AVAILABLE
            if available
            else ChildObservableAvailabilityV1.TYPED_MISSING
        ),
        missing_reason=None if available else "ONE_OR_MORE_REGIONS_TYPED_MISSING",
        root_observation_sha256=sha(f"root observation:{branch.value}"),
        parent_observation_sha256=sha(f"parent observation:{branch.value}"),
        child_observation_sha256=sha(f"child observation:{branch.value}"),
        source_evidence_sha256s=(sha(f"source:{branch.value}"),),
        projections=rows,
    )


def receipt(
    *,
    branches: tuple[BranchChildObservableV1, ...] | None = None,
    availability: ChildObservableAvailabilityV1 = (
        ChildObservableAvailabilityV1.AVAILABLE
    ),
    missing_reason: str | None = None,
) -> ChildObservableReceiptV1:
    regions = region_set()
    root, parent, child = states()
    if branches is None:
        branches = tuple(branch_receipt(q, regions) for q in QUARTET_BRANCHES_V1)
    return ChildObservableReceiptV1(
        observation_id="case-1/action-1/child-observation",
        action=action(),
        root=root,
        parent=parent,
        child=child,
        region_set=regions,
        observer_policy=observer_policy(),
        branches=branches,
        availability=availability,
        missing_reason=missing_reason,
        observer_implementation_sha256=sha("observer implementation"),
        matcher_backend_receipt_sha256=sha("matcher backend receipt"),
        action_execution_receipt_sha256=sha("action execution receipt"),
        cost_account="PROSPECTIVE_RUNTIME",
        cost_receipt_sha256=sha("cost receipt"),
    )


def test_complete_receipt_binds_fixed_quartet_and_all_identities():
    value = receipt()

    assert value.complete is True
    assert QUARTET_BRANCH_NAMES_V1 == ("CC", "RR", "CR", "RC")
    assert value.receipt_record() == {
        **value._payload(),
        "receipt_sha256": value.receipt_sha256,
    }
    assert value.receipt_record()["action_binding_sha256"] == value.action.binding_sha256
    assert value.receipt_record()["root_binding_sha256"] == value.root.binding_sha256
    assert value.receipt_record()["parent_binding_sha256"] == value.parent.binding_sha256
    assert value.receipt_record()["child_binding_sha256"] == value.child.binding_sha256
    assert value.receipt_record()["region_set_sha256"] == value.region_set.region_set_sha256
    assert replace(value, receipt_sha256=value.receipt_sha256) == value


def test_exact_action_change_changes_child_receipt_identity():
    first = receipt()
    changed_action = replace(first.action, strength=0.5, binding_sha256="")
    second = replace(first, action=changed_action, receipt_sha256="")

    assert first.action.binding_sha256 != second.action.binding_sha256
    assert first.receipt_sha256 != second.receipt_sha256
    with pytest.raises(ValueError, match="action binding SHA-256 drifted"):
        replace(first.action, strength=0.5)


def test_missing_projection_is_typed_and_cannot_be_zero_imputed():
    regions = region_set()
    available = projection(QUARTET_BRANCHES_V1[0], regions.regions[0])

    with pytest.raises(ValueError, match="cannot carry numeric metric payloads"):
        replace(
            available,
            availability=ChildObservableAvailabilityV1.TYPED_MISSING,
            missing_reason="NOT_OBSERVED",
            conservative_bound_receipt_sha256=sha("bound"),
            projection_binding_sha256="",
        )
    with pytest.raises(ValueError, match="conservative-bound receipt"):
        RegionProjectionBindingV1(
            branch=QUARTET_BRANCHES_V1[0],
            region_id=regions.regions[0].region_id,
            region_identity_sha256=regions.regions[0].identity_sha256,
            availability=ChildObservableAvailabilityV1.TYPED_MISSING,
            missing_reason="NOT_OBSERVED",
            parent_relative_metrics_sha256=None,
            root_relative_metrics_sha256=None,
            conservative_bound_receipt_sha256=None,
        )


def test_partial_branch_and_partial_quartet_cannot_claim_complete():
    regions = region_set()
    missing_rows = (
        projection(QUARTET_BRANCHES_V1[0], regions.regions[0]),
        projection(
            QUARTET_BRANCHES_V1[0], regions.regions[1], available=False,
        ),
    )
    with pytest.raises(ValueError, match="partial quartet branch"):
        BranchChildObservableV1(
            branch=QUARTET_BRANCHES_V1[0],
            availability=ChildObservableAvailabilityV1.AVAILABLE,
            missing_reason=None,
            root_observation_sha256=sha("root"),
            parent_observation_sha256=sha("parent"),
            child_observation_sha256=sha("child"),
            source_evidence_sha256s=(),
            projections=missing_rows,
        )

    branches = tuple(
        branch_receipt(
            q, regions, missing_region="1" if q is QUARTET_BRANCHES_V1[0] else None,
        )
        for q in QUARTET_BRANCHES_V1
    )
    with pytest.raises(ValueError, match="partial quartet"):
        receipt(branches=branches)
    incomplete = receipt(
        branches=branches,
        availability=ChildObservableAvailabilityV1.TYPED_MISSING,
        missing_reason="CC_HAS_TYPED_MISSING_REGION",
    )
    assert incomplete.complete is False


def test_region_projection_validator_rejects_partial_duplicate_and_identity_drift():
    regions = region_set()
    branch = QUARTET_BRANCHES_V1[1]
    rows = tuple(projection(branch, region) for region in regions.regions)
    assert validate_region_projection_coverage(regions, rows, branch) == rows

    with pytest.raises(ValueError, match="coverage is partial"):
        validate_region_projection_coverage(regions, rows[:1], branch)
    with pytest.raises(ValueError, match="duplicate"):
        validate_region_projection_coverage(regions, (rows[0], rows[0]), branch)
    drifted = replace(
        rows[0], region_identity_sha256=sha("wrong region"),
        projection_binding_sha256="",
    )
    with pytest.raises(ValueError, match="does not match frozen region set"):
        validate_region_projection_coverage(regions, (drifted, rows[1]), branch)


def test_receipt_requires_canonical_region_and_quartet_order():
    value = receipt()
    wrong_region_order = replace(
        value.branches[0],
        projections=tuple(reversed(value.branches[0].projections)),
        projection_set_sha256="",
        branch_receipt_sha256="",
    )
    with pytest.raises(ValueError, match="frozen region-set order"):
        replace(
            value,
            branches=(wrong_region_order, *value.branches[1:]),
            receipt_sha256="",
        )
    with pytest.raises(ValueError, match="ordered CC/RR/CR/RC"):
        replace(value, branches=tuple(reversed(value.branches)), receipt_sha256="")


def test_l0_l1_l2_policy_and_cost_receipt_are_mandatory_and_hashed():
    policy = observer_policy()
    assert tuple(row.level for row in policy.levels) == tuple(ObserverLevelV1)

    with pytest.raises(ValueError, match="ordered L0/L1/L2"):
        replace(policy, levels=policy.levels[:-1], policy_sha256="")
    with pytest.raises(ValueError, match="coverage mode drifted"):
        replace(
            policy.levels[0], coverage_mode="FOCUSED_UNION",
            level_policy_sha256="",
        )
    with pytest.raises(ValueError, match="cost_receipt_sha256"):
        replace(receipt(), cost_receipt_sha256="not-a-sha", receipt_sha256="")
    with pytest.raises(ValueError, match="prospective-runtime"):
        replace(receipt(), cost_account="OFFLINE", receipt_sha256="")


def test_state_hashes_are_all_or_none_and_receipt_stays_outcome_blind():
    value = receipt()
    with pytest.raises(ValueError, match="all-or-none"):
        replace(
            value.branches[0], child_observation_sha256=None,
            branch_receipt_sha256="",
        )
    with pytest.raises(ValueError, match="own state identity"):
        replace(value, child=value.parent, receipt_sha256="")
    with pytest.raises(ValueError, match="outcome blind"):
        replace(value, outcome_read=True, receipt_sha256="")


def test_supplied_content_hashes_fail_closed_on_drift():
    value = receipt()
    with pytest.raises(ValueError, match="child-observable receipt SHA-256 drifted"):
        replace(value, receipt_sha256=sha("fabricated"))
    with pytest.raises(ValueError, match="region-set SHA-256 drifted"):
        replace(value.region_set, region_set_sha256=sha("fabricated"))
