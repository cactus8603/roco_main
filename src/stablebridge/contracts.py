"""Small, explicit contracts shared by StableBridge experiments.

Time values are arrival events (frame indices in E01), not seconds. Feature
payloads may be NumPy arrays or torch tensors; this module does not import torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping
import math


EVIDENCE_STATES = frozenset({"unresolved", "eligible_for_query", "stale", "invalidated"})
QUERY_STATES = frozenset({"trusted", "actionable", "waiting", "unresolved", "abstained"})
DECISION_OUTCOMES = frozenset({"accept", "retain", "retract"})


@dataclass(frozen=True)
class QueryIdentity:
    sequence: str
    source_view: str
    source_frame: int
    target_view: str
    target_frame: int
    native_xy: tuple[float, float] | None = None
    roi: tuple[int, int, int, int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if not self.sequence or not self.source_view or not self.target_view:
            raise ValueError("Query sequence and views must be nonempty")
        if self.source_frame < 0 or self.target_frame < 0:
            raise ValueError("Frame indices must be nonnegative")
        if self.native_xy is not None and (len(self.native_xy) != 2 or not all(math.isfinite(v) for v in self.native_xy)):
            raise ValueError("native_xy must contain two finite native pixel coordinates")
        if self.roi is not None and (len(self.roi) != 4 or self.roi[2] <= 0 or self.roi[3] <= 0):
            raise ValueError("roi is (x, y, width, height) with positive size")

    @property
    def cutoff(self) -> int:
        return max(self.source_frame, self.target_frame)

    @property
    def allowed_views(self) -> frozenset[str]:
        return frozenset((self.source_view, self.target_view))


@dataclass(frozen=True)
class TimeSnapshot:
    cutoff: float
    available: float | None = None
    mode: str = "M1"
    allowed_views: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.mode not in {"M0", "M1"}:
            raise ValueError("Only M0 pair and M1 causal snapshots are implemented")
        if not math.isfinite(self.cutoff) or (self.available is not None and not math.isfinite(self.available)):
            raise ValueError("Snapshot event values must be finite")


@dataclass(frozen=True)
class FrameEvidence:
    sequence: str
    frame: int
    view: str
    features: Any
    origin_xy: tuple[float, float] = (0.0, 0.0)
    image_hw: tuple[int, int] = (0, 0)
    quality: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    source_groups: tuple[str, ...] = ()
    availability: float | None = None
    observation_ids: tuple[str, ...] = ()
    parent_versions: tuple[str, ...] = ()
    evidence_id: str | None = None
    status: str = "eligible_for_query"

    def __post_init__(self):
        if not self.sequence or not self.view or self.frame < 0:
            raise ValueError("Evidence requires a sequence, view, and nonnegative frame")
        if self.status not in EVIDENCE_STATES:
            raise ValueError(f"Unknown evidence state: {self.status}")
        if not math.isfinite(float(self.quality)):
            raise ValueError("Evidence quality must be finite; it is a score, not a probability")
        if self.availability is not None and (not math.isfinite(self.availability) or self.availability < self.frame):
            raise ValueError("Availability cannot precede the observation arrival event")
        if len(self.origin_xy) != 2 or not all(math.isfinite(v) for v in self.origin_xy):
            raise ValueError("origin_xy must contain two finite coordinates")
        if len(self.image_hw) != 2 or any(v < 0 for v in self.image_hw):
            raise ValueError("image_hw must be a nonnegative (height, width)")
        shape = getattr(self.features, "shape", None)
        if self.features is not None and (shape is None or not (
                hasattr(self.features, "nbytes") or
                (hasattr(self.features, "numel") and hasattr(self.features, "element_size")))):
            raise ValueError("Features must be an array/tensor with measurable storage bytes")
        if shape is not None and len(shape) != 3:
            raise ValueError("Frame features must have shape [C, h, w]")


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    payload: FrameEvidence
    parent_versions: tuple[str, ...]
    observation_ids: tuple[str, ...]
    source_groups: tuple[str, ...]
    information_time: float
    availability: float
    status: str
    insertion_order: int
    reason: str = ""

    @property
    def features(self):
        return self.payload.features

    @property
    def frame(self):
        return self.payload.frame

    @property
    def view(self):
        return self.payload.view


@dataclass(frozen=True)
class ReadBudget:
    max_reads: int = 4
    max_probes: int = 0
    max_index_items: int | None = None
    max_read_bytes: int | None = None

    def __post_init__(self):
        for value in (self.max_reads, self.max_probes, self.max_index_items, self.max_read_bytes):
            if value is not None and value < 0:
                raise ValueError("Read budgets must be nonnegative")


@dataclass(frozen=True)
class ResourceCost:
    """Cost vector for one action; units are declared by the run manifest."""

    compute: float = 0.0
    read_bytes: int = 0
    candidates: int = 0
    revisions: int = 0
    delay_events: float = 0.0

    def __post_init__(self):
        values = (self.compute, self.read_bytes, self.candidates,
                  self.revisions, self.delay_events)
        if any(not math.isfinite(float(value)) or value < 0 for value in values):
            raise ValueError("Action costs must be finite and nonnegative")


@dataclass(frozen=True)
class ActionBudget:
    """Remaining resources at one query-state revision."""

    compute: float
    read_bytes: int
    candidates: int
    revisions: int
    delay_events: float

    def __post_init__(self):
        values = (self.compute, self.read_bytes, self.candidates,
                  self.revisions, self.delay_events)
        if any(not math.isfinite(float(value)) or value < 0 for value in values):
            raise ValueError("Remaining budgets must be finite and nonnegative")

    def can_afford(self, cost: ResourceCost) -> bool:
        return (cost.compute <= self.compute and
                cost.read_bytes <= self.read_bytes and
                cost.candidates <= self.candidates and
                cost.revisions <= self.revisions and
                cost.delay_events <= self.delay_events)

    def spend(self, cost: ResourceCost) -> "ActionBudget":
        if not self.can_afford(cost):
            raise ValueError("Action exceeds the remaining query budget")
        return ActionBudget(
            compute=self.compute - cost.compute,
            read_bytes=self.read_bytes - cost.read_bytes,
            candidates=self.candidates - cost.candidates,
            revisions=self.revisions - cost.revisions,
            delay_events=self.delay_events - cost.delay_events,
        )


@dataclass(frozen=True)
class RiskEstimate:
    """Calibrated risk of the current answer at a declared error threshold."""

    error_probability: float
    threshold_native_px: float
    calibration_id: str
    expected_error: float | None = None
    tail_probability: float | None = None
    tail_threshold_native_px: float | None = None

    def __post_init__(self):
        if not 0.0 <= self.error_probability <= 1.0:
            raise ValueError("error_probability must be in [0, 1]")
        if not math.isfinite(self.threshold_native_px) or self.threshold_native_px <= 0:
            raise ValueError("The error threshold must be a positive native-pixel value")
        if not self.calibration_id:
            raise ValueError("Risk estimates require a calibration/model version")
        if self.expected_error is not None and (
                not math.isfinite(self.expected_error) or self.expected_error < 0):
            raise ValueError("expected_error must be finite and nonnegative")
        if self.tail_probability is not None and not 0.0 <= self.tail_probability <= 1.0:
            raise ValueError("tail_probability must be in [0, 1]")
        if (self.tail_probability is None) != (self.tail_threshold_native_px is None):
            raise ValueError("Tail probability and threshold must be declared together")
        if self.tail_threshold_native_px is not None and (
                not math.isfinite(self.tail_threshold_native_px) or
                self.tail_threshold_native_px <= self.threshold_native_px):
            raise ValueError("The tail threshold must exceed the main error threshold")


@dataclass(frozen=True)
class ActionAssessment:
    """Observable, action-conditioned outcome estimate for one query state."""

    action_id: str
    action_kind: str
    based_on_revision: int
    evidence_version: str
    success_probability: float
    expected_gain: float
    harm_probability: float
    expected_tail_harm: float
    cost: ResourceCost = field(default_factory=ResourceCost)
    required_evidence_ids: tuple[str, ...] = ()
    calibration_id: str = "uncalibrated"
    candidate_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.action_id or not self.action_kind or not self.evidence_version:
            raise ValueError("Actions require IDs, a kind, and an evidence version")
        if self.based_on_revision < 0:
            raise ValueError("Action state revisions must be nonnegative")
        for name, value in (("success_probability", self.success_probability),
                            ("harm_probability", self.harm_probability)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.expected_gain):
            raise ValueError("expected_gain must be finite and may be negative")
        if not math.isfinite(self.expected_tail_harm) or self.expected_tail_harm < 0:
            raise ValueError("expected_tail_harm must be finite and nonnegative")
        if not self.calibration_id:
            raise ValueError("Action estimates require a calibration/model version")


@dataclass(frozen=True)
class QueryState:
    """Versioned state; no status is an intrinsic property of the pixel."""

    query: QueryIdentity
    revision: int
    evidence_version: str
    prediction_id: str
    risk: RiskEstimate
    budget: ActionBudget
    status: str
    ambiguity: float = 0.0
    evidence_ids: tuple[str, ...] = ()
    admissible_action_ids: tuple[str, ...] = ("retain",)
    information_cutoff: float | None = None
    decision_event: float | None = None
    deadline_event: float | None = None
    predecessor_revision: int | None = None

    def __post_init__(self):
        if self.revision < 0:
            raise ValueError("Query revisions must be nonnegative")
        if self.predecessor_revision is not None and self.predecessor_revision >= self.revision:
            raise ValueError("A predecessor must be older than the current revision")
        if not self.evidence_version or not self.prediction_id:
            raise ValueError("Query states require evidence and prediction versions")
        if self.status not in QUERY_STATES:
            raise ValueError(f"Unknown query state: {self.status}")
        if not math.isfinite(self.ambiguity) or self.ambiguity < 0:
            raise ValueError("ambiguity must be finite and nonnegative")
        if not self.admissible_action_ids or len(set(self.admissible_action_ids)) != len(self.admissible_action_ids):
            raise ValueError("Each query state requires a nonempty, unique action set")
        if not all(self.admissible_action_ids):
            raise ValueError("Admissible action IDs must be nonempty")
        for value in (self.information_cutoff, self.decision_event, self.deadline_event):
            if value is not None and not math.isfinite(value):
                raise ValueError("Query timing events must be finite")
        if (self.decision_event is not None and self.deadline_event is not None and
                self.decision_event > self.deadline_event):
            raise ValueError("The query decision event cannot exceed its deadline")


@dataclass(frozen=True)
class QueryTransition:
    """Auditable result of executing one assessed action."""

    query: QueryIdentity
    from_revision: int
    to_revision: int
    action_id: str
    outcome: str
    before_prediction_id: str
    after_prediction_id: str
    before_evidence_version: str
    after_evidence_version: str
    before_status: str
    after_status: str
    cost: ResourceCost
    added_evidence_ids: tuple[str, ...] = ()
    accepted_candidate_id: str | None = None
    support_eligible: bool = False
    support_reason: str = ""
    reason: str = ""

    def __post_init__(self):
        if self.to_revision != self.from_revision + 1:
            raise ValueError("A query transition must advance exactly one revision")
        if self.outcome not in DECISION_OUTCOMES:
            raise ValueError(f"Unknown decision outcome: {self.outcome}")
        if self.outcome == "accept":
            if not self.accepted_candidate_id:
                raise ValueError("Accepted updates require a candidate ID")
            if self.after_prediction_id == self.before_prediction_id:
                raise ValueError("Accepted updates must create a new prediction version")
        elif self.outcome == "retain":
            if self.accepted_candidate_id is not None:
                raise ValueError("Retain cannot name an accepted candidate")
            if self.after_prediction_id != self.before_prediction_id:
                raise ValueError("Retain must preserve the current prediction")
        elif self.after_prediction_id == self.before_prediction_id:
            raise ValueError("Retraction must select a different prediction version")
        if self.support_eligible and not self.support_reason:
            raise ValueError("Support eligibility requires a query-specific reason")


def advance_query_state(
        state: QueryState,
        action: ActionAssessment,
        *,
        outcome: str,
        next_status: str,
        next_risk: RiskEstimate | None = None,
        next_ambiguity: float | None = None,
        next_prediction_id: str | None = None,
        next_evidence_version: str | None = None,
        next_admissible_action_ids: tuple[str, ...] | None = None,
        next_decision_event: float | None = None,
        added_evidence_ids: tuple[str, ...] = (),
        accepted_candidate_id: str | None = None,
        support_eligible: bool = False,
        support_reason: str = "",
        reason: str = "",
) -> tuple[QueryState, QueryTransition]:
    """Apply an action without allowing stale estimates or hidden free resources."""

    if action.based_on_revision != state.revision:
        raise ValueError("Action assessment is stale for the current query revision")
    if action.evidence_version != state.evidence_version:
        raise ValueError("Action assessment used a different evidence snapshot")
    if action.action_id not in state.admissible_action_ids:
        raise ValueError("Action is not admissible in the current query state")
    if not set(action.required_evidence_ids).issubset(state.evidence_ids):
        raise ValueError("Action requires evidence absent from the current query state")
    if accepted_candidate_id is not None and accepted_candidate_id not in action.candidate_ids:
        raise ValueError("The accepted candidate was not produced by the action")
    budget = state.budget.spend(action.cost)
    prediction_id = next_prediction_id or state.prediction_id
    evidence_version = next_evidence_version or state.evidence_version
    if added_evidence_ids and evidence_version == state.evidence_version:
        raise ValueError("New evidence requires a new evidence version")
    transition = QueryTransition(
        query=state.query,
        from_revision=state.revision,
        to_revision=state.revision + 1,
        action_id=action.action_id,
        outcome=outcome,
        before_prediction_id=state.prediction_id,
        after_prediction_id=prediction_id,
        before_evidence_version=state.evidence_version,
        after_evidence_version=evidence_version,
        before_status=state.status,
        after_status=next_status,
        cost=action.cost,
        added_evidence_ids=added_evidence_ids,
        accepted_candidate_id=accepted_candidate_id,
        support_eligible=support_eligible,
        support_reason=support_reason,
        reason=reason,
    )
    next_state = replace(
        state,
        revision=state.revision + 1,
        predecessor_revision=state.revision,
        evidence_version=evidence_version,
        prediction_id=prediction_id,
        risk=next_risk or state.risk,
        budget=budget,
        status=next_status,
        ambiguity=state.ambiguity if next_ambiguity is None else next_ambiguity,
        evidence_ids=tuple(dict.fromkeys((*state.evidence_ids, *added_evidence_ids))),
        admissible_action_ids=(state.admissible_action_ids if next_admissible_action_ids is None
                               else next_admissible_action_ids),
        decision_event=(state.decision_event if next_decision_event is None
                        else next_decision_event),
    )
    return next_state, transition


@dataclass(frozen=True)
class SupportCheck:
    eligible: bool
    reason: str
    source_groups: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()

    def __bool__(self):
        return self.eligible


@dataclass(frozen=True)
class RetrievalResult:
    records: tuple[EvidenceRecord, ...]
    costs: Mapping[str, int]
    independent_groups: tuple[str, ...]

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)
