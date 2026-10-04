"""Measured prospective-runtime cost receipts for action qualification.

These receipts record what was actually incurred for one outcome-blind
``before/child x CC/RR/CR/RC`` audit.  They are observations, not selector
ceilings.  A later scene-disjoint calibration step may derive and validate a
ceiling, but this module intentionally cannot promote a historical maximum to
a prospective guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

from .child_observables import QUARTET_BRANCHES_V1, QuartetBranchV1


QUALIFICATION_COST_SCHEMA_V1 = "stablebridge-action-qualification-cost/v1"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_STATE_ORDER = ("before", "child")
_ACCOUNT = "PROSPECTIVE_RUNTIME"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _sha(value: object, name: str) -> str:
    value = _text(value, name)
    if _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class QuartetForwardCostV1:
    state: str
    branch: QuartetBranchV1
    actual_forward: bool
    cache_state: str
    reuse_receipt_sha256: str | None
    input_binding_sha256: str
    output_binding_sha256: str
    forward_receipt_sha256: str
    cpu_seconds: float
    gpu_seconds: float
    wall_seconds: float
    bytes_moved: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    row_sha256: str = ""

    def __post_init__(self) -> None:
        if self.state not in _STATE_ORDER:
            raise ValueError("cost row state must be before or child")
        if not isinstance(self.branch, QuartetBranchV1):
            raise ValueError("cost row branch must be typed")
        if not isinstance(self.actual_forward, bool):
            raise ValueError("actual_forward must be boolean")
        allowed_cache = {"cold", "warm"} if self.actual_forward else {"reused"}
        if self.cache_state not in allowed_cache:
            raise ValueError("forward cache_state does not match actual/reused state")
        for name in (
            "input_binding_sha256", "output_binding_sha256",
            "forward_receipt_sha256",
        ):
            _sha(getattr(self, name), name)
        if self.actual_forward:
            if self.reuse_receipt_sha256 is not None:
                raise ValueError("actual forward cannot carry a reuse receipt")
        else:
            _sha(self.reuse_receipt_sha256, "reuse_receipt_sha256")
        for name in ("cpu_seconds", "gpu_seconds", "wall_seconds"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for name in ("bytes_moved", "peak_allocated_bytes", "peak_reserved_bytes"):
            _nonnegative_int(getattr(self, name), name)
        if self.peak_allocated_bytes > self.peak_reserved_bytes:
            raise ValueError("allocated VRAM cannot exceed reserved VRAM")
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.row_sha256 not in {"", expected}:
            raise ValueError("quartet forward cost row hash drift")
        object.__setattr__(self, "row_sha256", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "state": self.state,
            "branch": self.branch.value,
            "actual_forward": self.actual_forward,
            "cache_state": self.cache_state,
            "reuse_receipt_sha256": self.reuse_receipt_sha256,
            "input_binding_sha256": self.input_binding_sha256,
            "output_binding_sha256": self.output_binding_sha256,
            "forward_receipt_sha256": self.forward_receipt_sha256,
            "cpu_seconds": self.cpu_seconds,
            "gpu_seconds": self.gpu_seconds,
            "wall_seconds": self.wall_seconds,
            "bytes_moved": self.bytes_moved,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }
        if include_hash:
            value["row_sha256"] = self.row_sha256
        return value


@dataclass(frozen=True)
class QualificationCostReceiptV1:
    case_id: str
    action_id: str
    action_binding_sha256: str
    action_execution_receipt_sha256: str
    matcher_backend_receipt_sha256: str
    cache_state: str
    action_cpu_seconds: float
    action_wall_seconds: float
    action_bytes_moved: int
    forward_costs: tuple[QuartetForwardCostV1, ...]
    cost_account: str = _ACCOUNT
    GT_read: bool = False
    H2_read: bool = False
    outcome_read: bool = False
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        _text(self.action_id, "action_id")
        for name in (
            "action_binding_sha256", "action_execution_receipt_sha256",
            "matcher_backend_receipt_sha256",
        ):
            _sha(getattr(self, name), name)
        if self.cache_state not in {"cold", "warm", "mixed"}:
            raise ValueError("cache_state must be cold, warm, or mixed")
        object.__setattr__(
            self, "action_cpu_seconds",
            _nonnegative(self.action_cpu_seconds, "action_cpu_seconds"),
        )
        object.__setattr__(
            self, "action_wall_seconds",
            _nonnegative(self.action_wall_seconds, "action_wall_seconds"),
        )
        _nonnegative_int(self.action_bytes_moved, "action_bytes_moved")
        rows = tuple(self.forward_costs)
        expected_order = tuple(
            (state, branch)
            for state in _STATE_ORDER
            for branch in QUARTET_BRANCHES_V1
        )
        if (
            len(rows) != 8
            or any(not isinstance(row, QuartetForwardCostV1) for row in rows)
            or tuple((row.state, row.branch) for row in rows) != expected_order
        ):
            raise ValueError(
                "qualification cost needs ordered before/child x CC/RR/CR/RC"
            )
        if self.cost_account != _ACCOUNT:
            raise ValueError("qualification cost account must be prospective-runtime")
        if any(value is not False for value in (self.GT_read, self.H2_read, self.outcome_read)):
            raise ValueError("qualification cost receipt must remain outcome blind")
        object.__setattr__(self, "forward_costs", rows)
        actual_cache_states = {
            row.cache_state for row in rows if row.actual_forward
        }
        expected_cache_state = (
            next(iter(actual_cache_states))
            if len(actual_cache_states) == 1 else "mixed"
        )
        if self.cache_state != expected_cache_state:
            raise ValueError("aggregate cache_state disagrees with actual forwards")
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.receipt_sha256 not in {"", expected}:
            raise ValueError("qualification cost receipt hash drift")
        object.__setattr__(self, "receipt_sha256", expected)

    @property
    def natural_forwards(self) -> int:
        return 8

    @property
    def actual_forwards(self) -> int:
        return sum(row.actual_forward for row in self.forward_costs)

    @property
    def reused_forwards(self) -> int:
        return self.natural_forwards - self.actual_forwards

    @property
    def wall_seconds(self) -> float:
        return self.action_wall_seconds + sum(
            row.wall_seconds for row in self.forward_costs if row.actual_forward
        )

    @property
    def gpu_seconds(self) -> float:
        return sum(
            row.gpu_seconds for row in self.forward_costs if row.actual_forward
        )

    @property
    def cpu_seconds(self) -> float:
        return self.action_cpu_seconds + sum(
            row.cpu_seconds for row in self.forward_costs if row.actual_forward
        )

    @property
    def peak_reserved_bytes(self) -> int:
        return max((row.peak_reserved_bytes for row in self.forward_costs), default=0)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema": QUALIFICATION_COST_SCHEMA_V1,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "action_binding_sha256": self.action_binding_sha256,
            "action_execution_receipt_sha256": (
                self.action_execution_receipt_sha256
            ),
            "matcher_backend_receipt_sha256": self.matcher_backend_receipt_sha256,
            "cache_state": self.cache_state,
            "action_cpu_seconds": self.action_cpu_seconds,
            "action_wall_seconds": self.action_wall_seconds,
            "action_bytes_moved": self.action_bytes_moved,
            "action_bytes_moved_scope": "EXTERNAL_PARENT_READ_PLUS_CHILD_WRITE",
            "forward_costs": [row.as_dict() for row in self.forward_costs],
            "natural_forwards": 8,
            "actual_forwards": sum(row.actual_forward for row in self.forward_costs),
            "reused_forwards": sum(not row.actual_forward for row in self.forward_costs),
            "matcher_trajectories": sum(
                row.actual_forward for row in self.forward_costs
            ),
            "cpu_seconds": self.action_cpu_seconds + sum(
                row.cpu_seconds for row in self.forward_costs if row.actual_forward
            ),
            "gpu_seconds": sum(
                row.gpu_seconds for row in self.forward_costs if row.actual_forward
            ),
            "wall_seconds": self.action_wall_seconds + sum(
                row.wall_seconds for row in self.forward_costs if row.actual_forward
            ),
            "bytes_moved": self.action_bytes_moved + sum(
                row.bytes_moved for row in self.forward_costs if row.actual_forward
            ),
            "peak_allocated_bytes": max(
                (row.peak_allocated_bytes for row in self.forward_costs), default=0,
            ),
            "peak_reserved_bytes": max(
                (row.peak_reserved_bytes for row in self.forward_costs), default=0,
            ),
            "cost_account": self.cost_account,
            "ceiling_authority": False,
            "scientific_qualification": False,
            "selector_admission": False,
            "GT_read": self.GT_read,
            "H2_read": self.H2_read,
            "outcome_read": self.outcome_read,
        }
        if include_hash:
            value["receipt_sha256"] = self.receipt_sha256
        return value


__all__ = [
    "QUALIFICATION_COST_SCHEMA_V1",
    "QualificationCostReceiptV1",
    "QuartetForwardCostV1",
]
