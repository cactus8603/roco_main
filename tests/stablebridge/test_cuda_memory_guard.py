from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stablebridge.physical_repair.cuda_memory_guard import (
    CudaMemoryReservationPolicyV1,
    CudaMemoryReservationV1,
)


def test_cpu_guard_is_inert_and_retry_target_is_bounded():
    policy = CudaMemoryReservationPolicyV1(
        enabled=True,
        target_fraction=0.88,
        minimum_headroom_mib=3072,
        chunk_mib=256,
        retry_fraction_decrement=0.05,
        minimum_target_fraction=0.65,
    )
    guard = CudaMemoryReservationV1(policy, torch.device("cpu"), retry_count=10)
    assert guard.effective_target_fraction == pytest.approx(0.65)
    assert guard.prime() == {
        "enabled": True, "active": False, "device_type": "cpu",
    }


def test_memory_policy_requires_exact_serialized_fields():
    value = {
        "enabled": True,
        "target_fraction": 0.88,
        "minimum_headroom_mib": 3072,
        "chunk_mib": 256,
        "retry_fraction_decrement": 0.05,
        "minimum_target_fraction": 0.65,
    }
    assert CudaMemoryReservationPolicyV1.from_mapping(value).enabled
    with pytest.raises(ValueError, match="fields drift"):
        CudaMemoryReservationPolicyV1.from_mapping({**value, "unknown": 1})
