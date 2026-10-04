from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from stablebridge.physical_repair.child_observables import QUARTET_BRANCHES_V1
from stablebridge.physical_repair.qualification_costs_v1 import (
    QualificationCostReceiptV1,
    QuartetForwardCostV1,
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def row(state, branch, *, actual=True):
    return QuartetForwardCostV1(
        state=state,
        branch=branch,
        actual_forward=actual,
        cache_state="cold" if actual else "reused",
        reuse_receipt_sha256=None if actual else sha(f"reuse:{state}:{branch.value}"),
        input_binding_sha256=sha(f"input:{state}:{branch.value}"),
        output_binding_sha256=sha(f"output:{state}:{branch.value}"),
        forward_receipt_sha256=sha(f"forward:{state}:{branch.value}"),
        cpu_seconds=0.1 if actual else 0.0,
        gpu_seconds=0.2 if actual else 0.0,
        wall_seconds=0.3 if actual else 0.0,
        bytes_moved=100 if actual else 0,
        peak_allocated_bytes=1000 if actual else 0,
        peak_reserved_bytes=2000 if actual else 0,
    )


def receipt(*, reuse_before_cc=False):
    rows = tuple(
        row(state, branch, actual=not (
            reuse_before_cc and state == "before" and branch.value == "CC"
        ))
        for state in ("before", "child")
        for branch in QUARTET_BRANCHES_V1
    )
    return QualificationCostReceiptV1(
        case_id="case-1",
        action_id="action-1",
        action_binding_sha256=sha("action"),
        action_execution_receipt_sha256=sha("execution"),
        matcher_backend_receipt_sha256=sha("backend"),
        cache_state="cold",
        action_cpu_seconds=0.4,
        action_wall_seconds=0.5,
        action_bytes_moved=50,
        forward_costs=rows,
    )


def test_cost_receipt_accounts_natural_actual_and_reused_forwards():
    value = receipt(reuse_before_cc=True)
    assert value.natural_forwards == 8
    assert value.actual_forwards == 7
    assert value.reused_forwards == 1
    assert value.wall_seconds == pytest.approx(2.6)
    assert value.cpu_seconds == pytest.approx(1.1)
    assert value.gpu_seconds == pytest.approx(1.4)
    assert value.peak_reserved_bytes == 2000
    assert value.as_dict()["ceiling_authority"] is False
    assert value.as_dict()["matcher_trajectories"] == 7


def test_cost_receipt_requires_exact_eight_row_order_and_outcome_blindness():
    value = receipt()
    with pytest.raises(ValueError, match="ordered before/child"):
        replace(value, forward_costs=value.forward_costs[:-1], receipt_sha256="")
    with pytest.raises(ValueError, match="ordered before/child"):
        replace(
            value,
            forward_costs=tuple(reversed(value.forward_costs)),
            receipt_sha256="",
        )
    with pytest.raises(ValueError, match="outcome blind"):
        replace(value, outcome_read=True, receipt_sha256="")


def test_reuse_and_actual_forward_bindings_are_mutually_exclusive():
    value = row("before", QUARTET_BRANCHES_V1[0])
    with pytest.raises(ValueError, match="cannot carry a reuse"):
        replace(value, reuse_receipt_sha256=sha("reuse"), row_sha256="")
    with pytest.raises(ValueError, match="reuse_receipt_sha256"):
        replace(
            value,
            actual_forward=False,
            cache_state="reused",
            reuse_receipt_sha256=None,
            row_sha256="",
        )


def test_cache_state_is_measured_per_forward_and_aggregate_must_match():
    value = receipt()
    rows = list(value.forward_costs)
    rows[1] = replace(rows[1], cache_state="warm", row_sha256="")
    with pytest.raises(ValueError, match="aggregate cache_state"):
        replace(value, forward_costs=tuple(rows), receipt_sha256="")
    mixed = replace(
        value, cache_state="mixed", forward_costs=tuple(rows), receipt_sha256="",
    )
    assert mixed.as_dict()["cache_state"] == "mixed"
    with pytest.raises(ValueError, match="cache_state"):
        replace(rows[0], cache_state="reused", row_sha256="")


def test_cost_hashes_fail_closed_on_drift():
    value = receipt()
    with pytest.raises(ValueError, match="receipt hash drift"):
        replace(value, receipt_sha256=sha("fabricated"))
