"""Fail-closed receipts for region-projected speculative-child observables.

This module intentionally does not run an action, a matcher, or a selector.  It
only validates and content-addresses the evidence produced after an exact action
has materialized a speculative child.  Numeric feature payloads stay in their
producer-owned artifacts; this contract binds their SHA-256 digests and makes
missing observations structurally different from observed numeric zeroes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MISSING_DISPOSITION = "TYPED_MISSING_WITH_CONSERVATIVE_BOUND_NEVER_ZERO"
_COST_ACCOUNT = "PROSPECTIVE_RUNTIME"


class QuartetBranchV1(str, Enum):
    CC = "CC"
    RR = "RR"
    CR = "CR"
    RC = "RC"


QUARTET_BRANCHES_V1 = (
    QuartetBranchV1.CC,
    QuartetBranchV1.RR,
    QuartetBranchV1.CR,
    QuartetBranchV1.RC,
)
QUARTET_BRANCH_NAMES_V1 = tuple(branch.value for branch in QUARTET_BRANCHES_V1)


class ChildObservableAvailabilityV1(str, Enum):
    AVAILABLE = "AVAILABLE"
    TYPED_MISSING = "TYPED_MISSING"


class ObserverLevelV1(str, Enum):
    L0_SENTINEL = "L0_SENTINEL"
    L1_FOCUSED = "L1_FOCUSED"
    L2_FULL_AUDIT = "L2_FULL_AUDIT"


OBSERVER_LEVELS_V1 = (
    ObserverLevelV1.L0_SENTINEL,
    ObserverLevelV1.L1_FOCUSED,
    ObserverLevelV1.L2_FULL_AUDIT,
)
_LEVEL_COVERAGE = {
    ObserverLevelV1.L0_SENTINEL: "FULL_FRAME_CHEAP",
    ObserverLevelV1.L1_FOCUSED: "FOCUSED_UNION",
    ObserverLevelV1.L2_FULL_AUDIT: "FULL_FRAME_HIGH_RESOLUTION",
}


def canonical_sha256(value: Any) -> str:
    """Return the repository's compact, deterministic JSON SHA-256."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value: object, name: str) -> str:
    value = _text(value, name)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _optional_sha(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _sha(value, name)


def _enum(value: object, cls: type[Enum], name: str):
    if not isinstance(value, cls):
        raise ValueError(f"{name} must be a typed {cls.__name__}")
    return value


def _sealed_hash(given: str, payload: Any, name: str) -> str:
    expected = canonical_sha256(payload)
    if given not in {"", expected}:
        raise ValueError(f"{name} drifted")
    return expected


@dataclass(frozen=True)
class ActionBindingV1:
    """Exact action identity and the frozen bank descriptor that owns it."""

    bank_id: str
    bank_sha256: str
    action_id: str
    mechanism_id: str
    operator_id: str
    control_id: str
    endpoint: str
    strength: float
    exact_parameter_sha256: str
    support_policy_id: str
    support_sha256: str
    composition_id: str
    action_descriptor_sha256: str
    binding_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "bank_id", "action_id", "mechanism_id", "operator_id", "control_id",
            "endpoint", "support_policy_id", "composition_id",
        ):
            _text(getattr(self, name), name)
        for name in (
            "bank_sha256", "exact_parameter_sha256", "support_sha256",
            "action_descriptor_sha256",
        ):
            _sha(getattr(self, name), name)
        if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)):
            raise ValueError("strength must be a finite nonnegative number")
        strength = float(self.strength)
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("strength must be a finite nonnegative number")
        payload = {
            "bank_id": self.bank_id,
            "bank_sha256": self.bank_sha256,
            "action_id": self.action_id,
            "mechanism_id": self.mechanism_id,
            "operator_id": self.operator_id,
            "control_id": self.control_id,
            "endpoint": self.endpoint,
            "strength": strength,
            "exact_parameter_sha256": self.exact_parameter_sha256,
            "support_policy_id": self.support_policy_id,
            "support_sha256": self.support_sha256,
            "composition_id": self.composition_id,
            "action_descriptor_sha256": self.action_descriptor_sha256,
        }
        object.__setattr__(self, "strength", strength)
        object.__setattr__(
            self, "binding_sha256",
            _sealed_hash(self.binding_sha256, payload, "action binding SHA-256"),
        )


@dataclass(frozen=True)
class ObservableStateBindingV1:
    """Content identity for root, parent, or uncommitted child state."""

    state_id: str
    input_sha256: str
    output_sha256: str
    binding_sha256: str = ""

    def __post_init__(self) -> None:
        _text(self.state_id, "state_id")
        _sha(self.input_sha256, "state input_sha256")
        _sha(self.output_sha256, "state output_sha256")
        payload = {
            "state_id": self.state_id,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
        }
        object.__setattr__(
            self, "binding_sha256",
            _sealed_hash(self.binding_sha256, payload, "state binding SHA-256"),
        )


@dataclass(frozen=True)
class RegionIdentityV1:
    """One immutable region in the parent/root coordinate system."""

    region_id: str
    bbox_xyxy: tuple[int, int, int, int]
    area_px: int
    identity_sha256: str = ""

    def __post_init__(self) -> None:
        _text(self.region_id, "region_id")
        if (
            not isinstance(self.bbox_xyxy, tuple)
            or len(self.bbox_xyxy) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) for v in self.bbox_xyxy)
        ):
            raise ValueError("bbox_xyxy must be an integer 4-tuple")
        x0, y0, x1, y1 = self.bbox_xyxy
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("bbox_xyxy must be a nonempty nonnegative half-open box")
        if isinstance(self.area_px, bool) or not isinstance(self.area_px, int):
            raise ValueError("area_px must be an integer")
        if not 0 < self.area_px <= (x1 - x0) * (y1 - y0):
            raise ValueError("area_px must be positive and fit inside the bounding box")
        payload = {
            "region_id": self.region_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "area_px": self.area_px,
        }
        object.__setattr__(
            self, "identity_sha256",
            _sealed_hash(self.identity_sha256, payload, "region identity SHA-256"),
        )


@dataclass(frozen=True)
class RegionSetBindingV1:
    """Canonical region graph/lattice binding shared by every quartet branch."""

    region_set_id: str
    region_policy_id: str
    label_map_sha256: str
    regions: tuple[RegionIdentityV1, ...]
    region_set_sha256: str = ""

    def __post_init__(self) -> None:
        _text(self.region_set_id, "region_set_id")
        _text(self.region_policy_id, "region_policy_id")
        _sha(self.label_map_sha256, "label_map_sha256")
        rows = tuple(self.regions)
        if not rows or any(not isinstance(row, RegionIdentityV1) for row in rows):
            raise ValueError("region set needs typed region identities")
        ids = tuple(row.region_id for row in rows)
        if len(ids) != len(set(ids)):
            raise ValueError("region ids must be unique")
        payload = {
            "region_set_id": self.region_set_id,
            "region_policy_id": self.region_policy_id,
            "label_map_sha256": self.label_map_sha256,
            "regions": [
                {"region_id": row.region_id, "identity_sha256": row.identity_sha256}
                for row in rows
            ],
        }
        object.__setattr__(self, "regions", rows)
        object.__setattr__(
            self, "region_set_sha256",
            _sealed_hash(self.region_set_sha256, payload, "region-set SHA-256"),
        )


@dataclass(frozen=True)
class ObserverLevelPolicyV1:
    """Content-addressed coverage and trigger policy for one observer level."""

    level: ObserverLevelV1
    coverage_mode: str
    coverage_policy_sha256: str
    trigger_policy_id: str
    trigger_policy_sha256: str
    feature_contract_sha256: str
    level_policy_sha256: str = ""

    def __post_init__(self) -> None:
        _enum(self.level, ObserverLevelV1, "observer level")
        _text(self.coverage_mode, "coverage_mode")
        if self.coverage_mode != _LEVEL_COVERAGE[self.level]:
            raise ValueError(f"{self.level.value} coverage mode drifted")
        _text(self.trigger_policy_id, "trigger_policy_id")
        for name in (
            "coverage_policy_sha256", "trigger_policy_sha256",
            "feature_contract_sha256",
        ):
            _sha(getattr(self, name), name)
        payload = {
            "level": self.level.value,
            "coverage_mode": self.coverage_mode,
            "coverage_policy_sha256": self.coverage_policy_sha256,
            "trigger_policy_id": self.trigger_policy_id,
            "trigger_policy_sha256": self.trigger_policy_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
        }
        object.__setattr__(
            self, "level_policy_sha256",
            _sealed_hash(
                self.level_policy_sha256, payload, "observer-level policy SHA-256",
            ),
        )


@dataclass(frozen=True)
class ChildObserverPolicyV1:
    """Frozen L0/L1/L2 schedule; changing it creates a new policy identity."""

    policy_id: str
    levels: tuple[ObserverLevelPolicyV1, ...]
    outside_support_skip_policy_id: str
    outside_support_skip_policy_sha256: str
    periodic_full_audit_interval_commits: int
    missing_disposition: str = _MISSING_DISPOSITION
    policy_sha256: str = ""

    def __post_init__(self) -> None:
        _text(self.policy_id, "observer policy_id")
        levels = tuple(self.levels)
        if (
            any(not isinstance(row, ObserverLevelPolicyV1) for row in levels)
            or tuple(row.level for row in levels) != OBSERVER_LEVELS_V1
        ):
            raise ValueError("observer policy must contain ordered L0/L1/L2 policies")
        _text(self.outside_support_skip_policy_id, "outside_support_skip_policy_id")
        _sha(
            self.outside_support_skip_policy_sha256,
            "outside_support_skip_policy_sha256",
        )
        if (
            isinstance(self.periodic_full_audit_interval_commits, bool)
            or not isinstance(self.periodic_full_audit_interval_commits, int)
            or self.periodic_full_audit_interval_commits < 1
        ):
            raise ValueError("periodic full-audit interval must be a positive integer")
        if self.missing_disposition != _MISSING_DISPOSITION:
            raise ValueError("missing-observation disposition may not impute numeric zero")
        payload = {
            "policy_id": self.policy_id,
            "levels": [row.level_policy_sha256 for row in levels],
            "outside_support_skip_policy_id": self.outside_support_skip_policy_id,
            "outside_support_skip_policy_sha256": (
                self.outside_support_skip_policy_sha256
            ),
            "periodic_full_audit_interval_commits": (
                self.periodic_full_audit_interval_commits
            ),
            "missing_disposition": self.missing_disposition,
        }
        object.__setattr__(self, "levels", levels)
        object.__setattr__(
            self, "policy_sha256",
            _sealed_hash(self.policy_sha256, payload, "child-observer policy SHA-256"),
        )


@dataclass(frozen=True)
class RegionProjectionBindingV1:
    """One branch/region projection, or an explicit nonnumeric missing row."""

    branch: QuartetBranchV1
    region_id: str
    region_identity_sha256: str
    availability: ChildObservableAvailabilityV1
    missing_reason: str | None
    parent_relative_metrics_sha256: str | None
    root_relative_metrics_sha256: str | None
    conservative_bound_receipt_sha256: str | None
    projection_binding_sha256: str = ""

    def __post_init__(self) -> None:
        _enum(self.branch, QuartetBranchV1, "quartet branch")
        _text(self.region_id, "projection region_id")
        _sha(self.region_identity_sha256, "projection region_identity_sha256")
        _enum(
            self.availability, ChildObservableAvailabilityV1,
            "region projection availability",
        )
        parent_hash = _optional_sha(
            self.parent_relative_metrics_sha256,
            "parent_relative_metrics_sha256",
        )
        root_hash = _optional_sha(
            self.root_relative_metrics_sha256,
            "root_relative_metrics_sha256",
        )
        bound_hash = _optional_sha(
            self.conservative_bound_receipt_sha256,
            "conservative_bound_receipt_sha256",
        )
        if self.availability is ChildObservableAvailabilityV1.AVAILABLE:
            if self.missing_reason is not None or parent_hash is None or root_hash is None:
                raise ValueError(
                    "available projection needs parent/root metric hashes and no missing reason"
                )
            if bound_hash is not None:
                raise ValueError("available projection cannot carry a missing-data bound")
        else:
            _text(self.missing_reason, "projection missing_reason")
            if parent_hash is not None or root_hash is not None:
                raise ValueError(
                    "typed-missing projection cannot carry numeric metric payloads"
                )
            if bound_hash is None:
                raise ValueError(
                    "typed-missing projection needs a conservative-bound receipt"
                )
        payload = {
            "branch": self.branch.value,
            "region_id": self.region_id,
            "region_identity_sha256": self.region_identity_sha256,
            "availability": self.availability.value,
            "missing_reason": self.missing_reason,
            "parent_relative_metrics_sha256": parent_hash,
            "root_relative_metrics_sha256": root_hash,
            "conservative_bound_receipt_sha256": bound_hash,
        }
        object.__setattr__(
            self, "projection_binding_sha256",
            _sealed_hash(
                self.projection_binding_sha256,
                payload,
                "region projection binding SHA-256",
            ),
        )


def validate_region_projection_coverage(
    region_set: RegionSetBindingV1,
    projections: Sequence[RegionProjectionBindingV1],
    branch: QuartetBranchV1,
) -> tuple[RegionProjectionBindingV1, ...]:
    """Require exactly one correctly bound typed row for every frozen region."""

    if not isinstance(region_set, RegionSetBindingV1):
        raise ValueError("region_set must be typed")
    _enum(branch, QuartetBranchV1, "quartet branch")
    rows = tuple(projections)
    if any(not isinstance(row, RegionProjectionBindingV1) for row in rows):
        raise ValueError("region projections must be typed")
    expected = {
        region.region_id: region.identity_sha256 for region in region_set.regions
    }
    actual: dict[str, RegionProjectionBindingV1] = {}
    for row in rows:
        if row.branch is not branch:
            raise ValueError("region projection is attached to the wrong quartet branch")
        if row.region_id in actual:
            raise ValueError("duplicate region projection")
        if expected.get(row.region_id) != row.region_identity_sha256:
            raise ValueError("region projection identity does not match frozen region set")
        actual[row.region_id] = row
    if set(actual) != set(expected):
        raise ValueError("region projection coverage is partial")
    return tuple(actual[region.region_id] for region in region_set.regions)


@dataclass(frozen=True)
class BranchChildObservableV1:
    """All region projections for one of the four fixed matcher branches."""

    branch: QuartetBranchV1
    availability: ChildObservableAvailabilityV1
    missing_reason: str | None
    root_observation_sha256: str | None
    parent_observation_sha256: str | None
    child_observation_sha256: str | None
    source_evidence_sha256s: tuple[str, ...]
    projections: tuple[RegionProjectionBindingV1, ...]
    projection_set_sha256: str = ""
    branch_receipt_sha256: str = ""

    def validate_against(self, region_set: RegionSetBindingV1) -> None:
        rows = validate_region_projection_coverage(
            region_set, self.projections, self.branch,
        )
        if rows != self.projections:
            raise ValueError("region projections must follow frozen region-set order")
        all_available = all(
            row.availability is ChildObservableAvailabilityV1.AVAILABLE
            for row in rows
        )
        if (
            self.availability is ChildObservableAvailabilityV1.AVAILABLE
            and not all_available
        ):
            raise ValueError("partial quartet branch cannot be marked available")
        if (
            self.availability is ChildObservableAvailabilityV1.TYPED_MISSING
            and all_available
        ):
            raise ValueError("complete quartet branch cannot be marked typed missing")

    def __post_init__(self) -> None:
        _enum(self.branch, QuartetBranchV1, "quartet branch")
        _enum(
            self.availability, ChildObservableAvailabilityV1,
            "branch availability",
        )
        observation_hashes = (
            _optional_sha(self.root_observation_sha256, "root_observation_sha256"),
            _optional_sha(self.parent_observation_sha256, "parent_observation_sha256"),
            _optional_sha(self.child_observation_sha256, "child_observation_sha256"),
        )
        if sum(value is not None for value in observation_hashes) not in {0, 3}:
            raise ValueError("root/parent/child observation hashes are all-or-none")
        sources = tuple(self.source_evidence_sha256s)
        if len(sources) != len(set(sources)):
            raise ValueError("source evidence hashes must be unique")
        for value in sources:
            _sha(value, "source evidence SHA-256")
        rows = tuple(self.projections)
        if not rows or any(not isinstance(row, RegionProjectionBindingV1) for row in rows):
            raise ValueError("branch needs typed region-projection rows")
        if any(row.branch is not self.branch for row in rows):
            raise ValueError("region projection is attached to the wrong quartet branch")
        all_available = all(
            row.availability is ChildObservableAvailabilityV1.AVAILABLE
            for row in rows
        )
        if self.availability is ChildObservableAvailabilityV1.AVAILABLE:
            if self.missing_reason is not None or not all_available:
                raise ValueError("partial quartet branch cannot be marked available")
            if any(value is None for value in observation_hashes):
                raise ValueError(
                    "available branch needs root/parent/child observation hashes"
                )
        else:
            _text(self.missing_reason, "branch missing_reason")
            if all_available:
                raise ValueError("complete quartet branch cannot be marked typed missing")
        projection_payload = [row.projection_binding_sha256 for row in rows]
        projection_set_hash = _sealed_hash(
            self.projection_set_sha256,
            projection_payload,
            "branch projection-set SHA-256",
        )
        payload = {
            "branch": self.branch.value,
            "availability": self.availability.value,
            "missing_reason": self.missing_reason,
            "root_observation_sha256": observation_hashes[0],
            "parent_observation_sha256": observation_hashes[1],
            "child_observation_sha256": observation_hashes[2],
            "source_evidence_sha256s": list(sources),
            "projection_set_sha256": projection_set_hash,
        }
        object.__setattr__(self, "source_evidence_sha256s", sources)
        object.__setattr__(self, "projections", rows)
        object.__setattr__(self, "projection_set_sha256", projection_set_hash)
        object.__setattr__(
            self, "branch_receipt_sha256",
            _sealed_hash(
                self.branch_receipt_sha256, payload, "branch child-observable receipt SHA-256",
            ),
        )


@dataclass(frozen=True)
class ChildObservableReceiptV1:
    """Canonical action-specific, region-projected CC/RR/CR/RC receipt."""

    observation_id: str
    action: ActionBindingV1
    root: ObservableStateBindingV1
    parent: ObservableStateBindingV1
    child: ObservableStateBindingV1
    region_set: RegionSetBindingV1
    observer_policy: ChildObserverPolicyV1
    branches: tuple[BranchChildObservableV1, ...]
    availability: ChildObservableAvailabilityV1
    missing_reason: str | None
    observer_implementation_sha256: str
    matcher_backend_receipt_sha256: str
    action_execution_receipt_sha256: str
    cost_account: str
    cost_receipt_sha256: str
    GT_read: bool = False
    H2_read: bool = False
    outcome_read: bool = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _text(self.observation_id, "observation_id")
        for name in (
            "action", "root", "parent", "child", "region_set", "observer_policy",
        ):
            expected = {
                "action": ActionBindingV1,
                "root": ObservableStateBindingV1,
                "parent": ObservableStateBindingV1,
                "child": ObservableStateBindingV1,
                "region_set": RegionSetBindingV1,
                "observer_policy": ChildObserverPolicyV1,
            }[name]
            if not isinstance(getattr(self, name), expected):
                raise ValueError(f"{name} must be a typed {expected.__name__}")
        if self.child.state_id in {self.root.state_id, self.parent.state_id}:
            raise ValueError("speculative child needs its own state identity")
        if self.root.state_id == self.parent.state_id and self.root != self.parent:
            raise ValueError("equal root/parent ids must bind identical state content")
        rows = tuple(self.branches)
        if (
            any(not isinstance(row, BranchChildObservableV1) for row in rows)
            or tuple(row.branch for row in rows) != QUARTET_BRANCHES_V1
        ):
            raise ValueError("child receipt must contain ordered CC/RR/CR/RC branches")
        for row in rows:
            row.validate_against(self.region_set)
        _enum(
            self.availability, ChildObservableAvailabilityV1,
            "child-observable availability",
        )
        all_available = all(
            row.availability is ChildObservableAvailabilityV1.AVAILABLE
            for row in rows
        )
        if self.availability is ChildObservableAvailabilityV1.AVAILABLE:
            if self.missing_reason is not None or not all_available:
                raise ValueError("partial quartet cannot be marked complete/available")
        else:
            _text(self.missing_reason, "child-observable missing_reason")
            if all_available:
                raise ValueError("complete quartet cannot be marked typed missing")
        for name in (
            "observer_implementation_sha256", "matcher_backend_receipt_sha256",
            "action_execution_receipt_sha256", "cost_receipt_sha256",
        ):
            _sha(getattr(self, name), name)
        if self.cost_account != _COST_ACCOUNT:
            raise ValueError("child-observable cost must use prospective-runtime account")
        if any(value is not False for value in (self.GT_read, self.H2_read, self.outcome_read)):
            raise ValueError("child-observable receipt must remain outcome blind")
        object.__setattr__(self, "branches", rows)
        payload = self._payload()
        object.__setattr__(
            self, "receipt_sha256",
            _sealed_hash(self.receipt_sha256, payload, "child-observable receipt SHA-256"),
        )

    @property
    def complete(self) -> bool:
        return self.availability is ChildObservableAvailabilityV1.AVAILABLE

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "region-projected-child-observable-receipt/v1",
            "observation_id": self.observation_id,
            "action_binding_sha256": self.action.binding_sha256,
            "root_binding_sha256": self.root.binding_sha256,
            "parent_binding_sha256": self.parent.binding_sha256,
            "child_binding_sha256": self.child.binding_sha256,
            "region_set_sha256": self.region_set.region_set_sha256,
            "observer_policy_sha256": self.observer_policy.policy_sha256,
            "branches": [
                {
                    "branch": row.branch.value,
                    "branch_receipt_sha256": row.branch_receipt_sha256,
                }
                for row in self.branches
            ],
            "availability": self.availability.value,
            "missing_reason": self.missing_reason,
            "observer_implementation_sha256": self.observer_implementation_sha256,
            "matcher_backend_receipt_sha256": self.matcher_backend_receipt_sha256,
            "action_execution_receipt_sha256": self.action_execution_receipt_sha256,
            "cost_account": self.cost_account,
            "cost_receipt_sha256": self.cost_receipt_sha256,
            "GT_read": self.GT_read,
            "H2_read": self.H2_read,
            "outcome_read": self.outcome_read,
        }

    def receipt_record(self) -> dict[str, Any]:
        """Return a compact JSON-ready receipt; feature values remain external."""

        return {**self._payload(), "receipt_sha256": self.receipt_sha256}


__all__ = [
    "ActionBindingV1",
    "BranchChildObservableV1",
    "ChildObservableAvailabilityV1",
    "ChildObservableReceiptV1",
    "ChildObserverPolicyV1",
    "ObservableStateBindingV1",
    "OBSERVER_LEVELS_V1",
    "ObserverLevelPolicyV1",
    "ObserverLevelV1",
    "QUARTET_BRANCH_NAMES_V1",
    "QUARTET_BRANCHES_V1",
    "QuartetBranchV1",
    "RegionIdentityV1",
    "RegionProjectionBindingV1",
    "RegionSetBindingV1",
    "canonical_sha256",
    "validate_region_projection_coverage",
]
