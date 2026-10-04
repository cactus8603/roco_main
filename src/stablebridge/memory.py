"""Bounded causal feature memory with explicit source/version accounting.

Metadata ranking reads only already-written summaries. An optional probe is
charged as a payload access, even when the probed record is subsequently rejected.
Costs are measured operations/bytes, not claimed FLOPs or measured wall time.
Support legality is not a claim that the support is geometrically applicable.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Callable, Any
import copy
import math
import uuid

from .contracts import (EvidenceRecord, FrameEvidence, QueryIdentity, ReadBudget,
                        RetrievalResult, SupportCheck, TimeSnapshot)


def _nbytes(value) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    return int(getattr(value, "nbytes", 0))


def _cosine(a, b) -> float:
    # Compact descriptors are index metadata paid for at insertion, never
    # recomputed from full feature payloads during free metadata selection.
    if a is None or b is None:
        return 0.0
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        return 0.0
    if not all(math.isfinite(float(v)) for v in a + b):
        return 0.0
    norm = math.sqrt(sum(float(x) ** 2 for x in a) * sum(float(x) ** 2 for x in b))
    return sum(float(x) * float(y) for x, y in zip(a, b)) / norm if norm else 0.0


class EvidenceStore:
    """Finite payload and provenance capacities; reset explicitly across sequences.

    An evicted payload leaves a bounded metadata tombstone. If a needed
    tombstone expires, dependent live records become stale rather than having
    their ancestry silently erased. New revisions need freshly validated parent
    versions; revision_of is history, never an independent observation.
    """

    SCHEMA_VERSION = 1

    def __init__(self, capacity: int = 8, tombstone_capacity: int = 32,
                 mode: str = "M1", max_lineage: int = 64,
                 max_feature_bytes: int | None = None):
        if capacity < 0 or tombstone_capacity < 0 or max_lineage < 1:
            raise ValueError("Memory capacities must be nonnegative and max_lineage positive")
        if mode not in {"M0", "M1"}:
            raise ValueError("Only M0 and M1 are supported")
        if max_feature_bytes is not None and max_feature_bytes < 0:
            raise ValueError("max_feature_bytes must be nonnegative or None")
        self.capacity = int(capacity)
        self.tombstone_capacity = int(tombstone_capacity)
        self.mode = mode
        self.max_lineage = int(max_lineage)
        self.max_feature_bytes = max_feature_bytes
        self.reset()

    def reset(self, sequence: str | None = None) -> None:
        self.sequence = sequence
        self._records: OrderedDict[str, EvidenceRecord] = OrderedDict()
        self._tombstones: OrderedDict[str, EvidenceRecord] = OrderedDict()
        self._revision_queue: OrderedDict[str, None] = OrderedDict()
        # Bounded exact revocation memory survives feature/tombstone eviction.
        # Saturation is fail-closed until sequence reset, never silent forgetting.
        self._revoked_groups: set[str] = set()
        self._revocation_overflow = False
        self._serial = 0
        self._namespace = uuid.uuid4().hex
        self._costs = dict(index_items=0, probe_items=0, read_items=0,
                           read_bytes=0, write_items=0, write_bytes=0,
                           evictions=0, revisions=0)

    def __len__(self):
        return len(self._records)

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return copy.deepcopy(tuple(self._records.values()))

    @property
    def revision_queue(self) -> tuple[str, ...]:
        return tuple(self._revision_queue)

    @property
    def tombstone_count(self) -> int:
        return len(self._tombstones)

    @property
    def feature_bytes(self) -> int:
        """Resident feature payload bytes; metadata is separately count-bounded."""
        return sum(_nbytes(record.features) for record in self._records.values())

    def _lookup(self, evidence_id: str):
        return self._records.get(evidence_id, self._tombstones.get(evidence_id))

    def _depends_on(self, record: EvidenceRecord, evidence_id: str) -> bool:
        pending, visited = list(record.parent_versions), set()
        while pending:
            parent_id = pending.pop()
            if parent_id == evidence_id:
                return True
            if parent_id in visited:
                continue
            visited.add(parent_id)
            parent = self._lookup(parent_id)
            if parent is not None:
                pending.extend(parent.parent_versions)
        return False

    def _set_stale(self, evidence_id: str, reason: str):
        record = self._records.get(evidence_id)
        if record is not None and record.status != "invalidated":
            self._records[evidence_id] = replace(record, status="stale", reason=reason)
            self._revision_queue[evidence_id] = None

    def _evict(self):
        while len(self._records) > self.capacity or (self.max_feature_bytes is not None and
                                                     self.feature_bytes > self.max_feature_bytes):
            evidence_id, record = self._records.popitem(last=False)
            self._revision_queue.pop(evidence_id, None)
            # Feature payload is released; retain only bounded lineage metadata.
            tomb = replace(record, payload=replace(record.payload, features=None, metadata={}))
            self._tombstones[evidence_id] = tomb
            self._costs["evictions"] += 1
        while len(self._tombstones) > self.tombstone_capacity:
            expired_id = next(iter(self._tombstones))
            affected = [r.evidence_id for r in self._records.values() if self._depends_on(r, expired_id)]
            for evidence_id in affected:
                self._set_stale(evidence_id, "required provenance expired")
            self._tombstones.pop(expired_id)

    def add(self, frame: FrameEvidence) -> EvidenceRecord:
        if self.sequence is not None and frame.sequence != self.sequence:
            raise ValueError("Reset memory before changing sequence")
        if self.max_feature_bytes is not None and _nbytes(frame.features) > self.max_feature_bytes:
            raise ValueError("Single feature payload exceeds max_feature_bytes")
        evidence_id = frame.evidence_id or f"{self._namespace}:{self._serial}"
        if self._lookup(evidence_id) is not None:
            raise ValueError("Evidence versions are immutable; duplicate evidence_id")
        if evidence_id in frame.parent_versions:
            raise ValueError("Evidence cannot depend on its own version")
        parents = tuple(dict.fromkeys(frame.parent_versions))
        if any(self._lookup(parent_id) is not None and
               self._depends_on(self._lookup(parent_id), evidence_id) for parent_id in parents):
            raise ValueError("Evidence version dependencies cannot form a cycle")
        groups, observations = set(frame.source_groups), set(frame.observation_ids)
        inherited_groups, inherited_observations = set(), set()
        information_time = float(frame.frame)
        availability = float(frame.frame if frame.availability is None else frame.availability)
        status, reason = frame.status, ""
        for parent_id in parents:
            parent = self._lookup(parent_id)
            if parent is None:
                status, reason = "stale", "unknown parent version"
                continue
            groups.update(parent.source_groups)
            observations.update(parent.observation_ids)
            inherited_groups.update(parent.source_groups)
            inherited_observations.update(parent.observation_ids)
            information_time = max(information_time, parent.information_time)
            availability = max(availability, parent.availability)
            if not self.check_support(parent_id):
                status, reason = "stale", "parent support is not eligible"
        if not parents:
            root = f"{frame.sequence}/{frame.view}/{frame.frame}"
            observations = observations or {root}
            groups = groups or observations.copy()
        elif set(frame.source_groups) - inherited_groups and not set(frame.observation_ids) - inherited_observations:
            raise ValueError("Derived copies cannot introduce independent groups without new observations")
        if max(len(parents), len(groups), len(observations)) > self.max_lineage:
            raise ValueError("Lineage exceeds configured bounded provenance capacity")
        if self._revoked_groups.intersection(groups):
            status, reason = "stale", "source group is invalidated"
        if self._revocation_overflow:
            status, reason = "stale", "revocation capacity exceeded; reset required"
        # Caller retains its inputs; copy so later in-place feature/metadata edits
        # cannot rewrite a stored version or invalidate resume reproducibility.
        stored_frame = copy.deepcopy(replace(frame, evidence_id=evidence_id,
                                             parent_versions=parents,
                                             source_groups=tuple(sorted(groups)),
                                             observation_ids=tuple(sorted(observations))))
        record = EvidenceRecord(evidence_id, stored_frame, parents,
                                tuple(sorted(observations)), tuple(sorted(groups)),
                                information_time, availability, status, self._serial, reason)
        self.sequence = frame.sequence
        self._serial += 1
        self._records[evidence_id] = record
        self._costs["write_items"] += 1
        self._costs["write_bytes"] += _nbytes(stored_frame.features)
        if status == "stale":
            self._revision_queue[evidence_id] = None
        self._evict()
        return copy.deepcopy(record)

    def _snapshot(self, query: QueryIdentity, snapshot: TimeSnapshot | None):
        snapshot = snapshot or TimeSnapshot(query.cutoff, mode=self.mode)
        if snapshot.mode != self.mode:
            raise ValueError("Store mode and snapshot mode must match")
        if snapshot.cutoff > query.cutoff:
            raise ValueError("M0/M1 snapshot cannot access events beyond original query cutoff")
        if snapshot.allowed_views is not None and not set(snapshot.allowed_views).issubset(query.allowed_views):
            raise ValueError("Snapshot grants views outside the query protocol")
        return snapshot

    def check_support(self, evidence_id: str, query: QueryIdentity | None = None,
                      snapshot: TimeSnapshot | None = None) -> SupportCheck:
        record = self._lookup(evidence_id)
        if record is None:
            return SupportCheck(False, "unknown or expired version")
        if record.status != "eligible_for_query":
            return SupportCheck(False, record.reason or record.status, record.source_groups, record.observation_ids)
        # Parent eligibility is checked recursively. IDs are immutable, and add
        # only accepts already-existing parents, so stored graphs are acyclic.
        for parent_id in record.parent_versions:
            parent_check = self.check_support(parent_id, query, snapshot)
            if not parent_check:
                return SupportCheck(False, "parent provenance is not eligible", record.source_groups, record.observation_ids)
        if query is not None:
            snap = self._snapshot(query, snapshot)
            if record.payload.sequence != query.sequence:
                return SupportCheck(False, "different sequence")
            if record.information_time > snap.cutoff:
                return SupportCheck(False, "future observation")
            if snap.available is not None and record.availability > snap.available:
                return SupportCheck(False, "feature computation not yet available")
            allowed_views = set(snap.allowed_views or query.allowed_views)
            if record.view not in allowed_views:
                return SupportCheck(False, "view is outside the information protocol")
            if snap.mode == "M0" and (record.view, record.frame) not in {
                (query.source_view, query.source_frame), (query.target_view, query.target_frame)
            }:
                return SupportCheck(False, "M0 cannot read historical evidence")
        return SupportCheck(True, "legal source; geometric applicability still requires matching",
                            record.source_groups, record.observation_ids)

    def retrieve(self, query: QueryIdentity, snapshot: TimeSnapshot | None = None,
                 budget: ReadBudget | None = None, policy: str = "fifo",
                 max_items: int | None = None,
                 probe: Callable[[FrameEvidence, QueryIdentity], float] | None = None) -> RetrievalResult:
        snapshot = self._snapshot(query, snapshot)
        budget = budget or ReadBudget()
        if policy not in {"fifo", "quality", "diversity", "query_conditioned"}:
            raise ValueError(f"Unknown read policy: {policy}")
        if max_items is not None and max_items < 0:
            raise ValueError("max_items must be nonnegative")
        before = self.cost_summary()
        eligible = []
        for index, record in enumerate(self._records.values()):
            if budget.max_index_items is not None and index >= budget.max_index_items:
                break
            self._costs["index_items"] += 1
            if self.check_support(record.evidence_id, query, snapshot):
                eligible.append(record)
        descriptor = query.metadata.get("descriptor")
        def score(record):
            if policy == "query_conditioned":
                return (_cosine(descriptor, record.payload.metadata.get("descriptor")),
                        record.payload.quality, record.insertion_order)
            if policy in {"quality", "diversity"}:
                return (record.payload.quality, record.insertion_order)
            # FIFO is insertion order for reads as well as eviction.
            return (-record.insertion_order,)
        eligible.sort(key=score, reverse=True)
        if probe is not None and budget.max_probes:
            probed = []
            for record in eligible[:budget.max_probes]:
                size = _nbytes(record.features)
                used_bytes = self._costs["read_bytes"] - before["read_bytes"]
                if budget.max_read_bytes is not None and used_bytes + size > budget.max_read_bytes:
                    continue
                self._costs["probe_items"] += 1
                self._costs["read_bytes"] += size
                value = float(probe(copy.deepcopy(record.payload), query))
                if not math.isfinite(value):
                    value = -math.inf
                probed.append((value, record))
            probed.sort(key=lambda pair: pair[0], reverse=True)
            probed_ids = {r.evidence_id for _, r in probed}
            eligible = [r for _, r in probed] + [r for r in eligible if r.evidence_id not in probed_ids]
        chosen, groups, observations = [], set(), set()
        limit = min(budget.max_reads, max_items) if max_items is not None else budget.max_reads
        while eligible and len(chosen) < limit:
            if policy == "diversity":
                eligible.sort(key=lambda r: (len(set(r.source_groups) - groups),
                                             r.payload.quality, r.insertion_order), reverse=True)
            record = eligible.pop(0)
            # A duplicated/derived copy of exactly the same observation never
            # consumes another read or increases independent evidence count.
            if set(record.observation_ids).issubset(observations):
                continue
            size = _nbytes(record.features)
            used_bytes = self._costs["read_bytes"] - before["read_bytes"]
            if budget.max_read_bytes is not None and used_bytes + size > budget.max_read_bytes:
                continue
            chosen.append(record)
            groups.update(record.source_groups)
            observations.update(record.observation_ids)
            self._costs["read_items"] += 1
            self._costs["read_bytes"] += size
        costs = {k: self._costs[k] - before[k] for k in self._costs}
        return RetrievalResult(copy.deepcopy(tuple(chosen)), costs, tuple(sorted(groups)))

    def invalidate(self, evidence_id: str | None = None, *, source_group: str | None = None,
                   reason: str = "source invalidated") -> tuple[str, ...]:
        if evidence_id is None and source_group is None:
            raise ValueError("Provide evidence_id or source_group")
        direct = set()
        revoked = {source_group} if source_group is not None else set()
        for mapping in (self._records, self._tombstones):
            for key, record in tuple(mapping.items()):
                if key == evidence_id or (source_group is not None and source_group in record.source_groups):
                    # Group membership inherited from a parent invalidates its
                    # support, not the child's independent observations.
                    is_root = key == evidence_id or not record.parent_versions
                    mapping[key] = replace(record, status="invalidated" if is_root else "stale", reason=reason)
                    if key == evidence_id and not record.parent_versions:
                        revoked.update(record.source_groups)
                    direct.add(key)
                    if key in self._records and not is_root:
                        self._revision_queue[key] = None
        revocation_capacity = max(1, self.tombstone_capacity) * self.max_lineage
        for group in sorted(revoked):
            if len(self._revoked_groups) >= revocation_capacity and group not in self._revoked_groups:
                self._revocation_overflow = True
                break
            self._revoked_groups.add(group)
        if self._revocation_overflow:
            for record in tuple(self._records.values()):
                self._set_stale(record.evidence_id, "revocation capacity exceeded; reset required")
        affected = set(direct)
        for record in tuple(self._records.values()):
            if any(self._depends_on(record, key) for key in direct):
                self._set_stale(record.evidence_id, reason)
                affected.add(record.evidence_id)
        return tuple(sorted(affected))

    def complete_revision(self, evidence_id: str, replacement: FrameEvidence,
                          support_ids: tuple[str, ...] = ()) -> EvidenceRecord:
        if self._lookup(evidence_id) is None:
            raise KeyError(evidence_id)
        if replacement.evidence_id == evidence_id:
            raise ValueError("A revision must have a new immutable version ID")
        if any(not self.check_support(parent) for parent in support_ids):
            raise ValueError("Revision supporters must be valid current versions")
        metadata = dict(replacement.metadata)
        metadata["revision_of"] = evidence_id
        record = self.add(replace(replacement, parent_versions=tuple(support_ids), metadata=metadata))
        self._revision_queue.pop(evidence_id, None)
        self._costs["revisions"] += 1
        return record

    def cost_summary(self) -> dict[str, int]:
        return dict(self._costs)

    def state_dict(self) -> dict[str, Any]:
        """Python state, suitable for torch.save/pickle; tensors are preserved."""
        return copy.deepcopy(dict(schema_version=self.SCHEMA_VERSION,
                                  config=dict(capacity=self.capacity, tombstone_capacity=self.tombstone_capacity,
                                              mode=self.mode, max_lineage=self.max_lineage,
                                              max_feature_bytes=self.max_feature_bytes),
                                  sequence=self.sequence, serial=self._serial, namespace=self._namespace,
                                  records=list(self._records.items()), tombstones=list(self._tombstones.items()),
                                  revision_queue=list(self._revision_queue), costs=self._costs,
                                  revoked_groups=sorted(self._revoked_groups),
                                  revocation_overflow=self._revocation_overflow))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = dict(capacity=self.capacity, tombstone_capacity=self.tombstone_capacity,
                        mode=self.mode, max_lineage=self.max_lineage,
                        max_feature_bytes=self.max_feature_bytes)
        if state.get("schema_version") != self.SCHEMA_VERSION or state.get("config") != expected:
            raise ValueError("Resume requires the identical memory schema and configuration")
        state = copy.deepcopy(state)
        records, tombstones = OrderedDict(state["records"]), OrderedDict(state["tombstones"])
        if len(records) > self.capacity or len(tombstones) > self.tombstone_capacity:
            raise ValueError("Resume state exceeds configured capacity")
        if set(records).intersection(tombstones):
            raise ValueError("Duplicate live/tombstone versions in resume state")
        if self.max_feature_bytes is not None and sum(_nbytes(r.features) for r in records.values()) > self.max_feature_bytes:
            raise ValueError("Resume state exceeds feature byte budget")
        revoked_groups = set(state["revoked_groups"])
        if len(revoked_groups) > max(1, self.tombstone_capacity) * self.max_lineage:
            raise ValueError("Resume revocation index exceeds configured capacity")
        self.sequence, self._serial, self._namespace = state["sequence"], state["serial"], state["namespace"]
        self._records, self._tombstones = records, tombstones
        self._revision_queue = OrderedDict((key, None) for key in state["revision_queue"] if key in records)
        self._costs = dict(state["costs"])
        self._revoked_groups = revoked_groups
        self._revocation_overflow = bool(state["revocation_overflow"])
