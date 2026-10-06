"""Process-local CUDA cache reservation for long shared-GPU training jobs.

The guard deliberately leaves a configurable safety margin for CUDA library
workspaces.  Temporary tensors are released without calling ``empty_cache``;
the PyTorch allocator therefore keeps the pages available to this process and
can reuse them for later training peaks.  A scheduler retry lowers the target
fraction through ``STABLEBRIDGE_RESOURCE_RETRY_COUNT`` so a too-aggressive
reservation cannot create an infinite OOM loop.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from typing import Any

import torch


@dataclass(frozen=True)
class CudaMemoryReservationPolicyV1:
    enabled: bool = False
    target_fraction: float = 0.88
    minimum_headroom_mib: int = 3072
    chunk_mib: int = 256
    retry_fraction_decrement: float = 0.05
    minimum_target_fraction: float = 0.65

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("CUDA memory reservation enabled must be boolean")
        for name in (
            "target_fraction", "retry_fraction_decrement",
            "minimum_target_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 < self.minimum_target_fraction <= self.target_fraction < 1.0:
            raise ValueError("CUDA reservation fractions must satisfy 0 < minimum <= target < 1")
        if not 0.0 <= self.retry_fraction_decrement < 1.0:
            raise ValueError("retry fraction decrement must lie in [0,1)")
        for name in ("minimum_headroom_mib", "chunk_mib"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_mapping(
        cls, value: dict[str, Any] | None,
    ) -> "CudaMemoryReservationPolicyV1":
        if value is None:
            return cls()
        expected = set(asdict(cls()))
        if set(value) != expected:
            raise ValueError("CUDA memory reservation policy fields drift")
        return cls(**value)


class CudaMemoryReservationV1:
    """Reserve reusable allocator cache up to a safe, retry-aware target."""

    def __init__(
        self,
        policy: CudaMemoryReservationPolicyV1,
        device: torch.device,
        *,
        retry_count: int | None = None,
    ) -> None:
        self.policy = policy
        # ``torch.device("cuda")`` deliberately leaves the index implicit,
        # but CUDA APIs such as ``set_device`` require a concrete logical
        # index.  The scheduler exposes exactly one physical GPU through
        # CUDA_VISIBLE_DEVICES, so logical device zero is the correct default.
        self.device = (
            torch.device("cuda", 0)
            if device.type == "cuda" and device.index is None
            else device
        )
        if retry_count is None:
            raw_retry = os.environ.get("STABLEBRIDGE_RESOURCE_RETRY_COUNT", "0")
            try:
                retry_count = int(raw_retry)
            except ValueError as exc:
                raise ValueError(
                    "STABLEBRIDGE_RESOURCE_RETRY_COUNT must be an integer"
                ) from exc
        if isinstance(retry_count, bool) or retry_count < 0:
            raise ValueError("resource retry count must be nonnegative")
        self.retry_count = retry_count
        self.effective_target_fraction = max(
            policy.minimum_target_fraction,
            policy.target_fraction - retry_count * policy.retry_fraction_decrement,
        )
        self.last_report: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        return self.policy.enabled and self.device.type == "cuda"

    def prime(self) -> dict[str, Any]:
        """Initialize CUDA/cuBLAS, then occupy the configured allocator pool."""

        if not self.active:
            self.last_report = {
                "enabled": self.policy.enabled,
                "active": False,
                "device_type": self.device.type,
            }
            return dict(self.last_report)
        torch.cuda.set_device(self.device)
        torch.cuda.init()
        # Initialize the cuBLAS handle before filling the cache; this avoids
        # starving the library's own small non-allocator workspace.
        probe = torch.ones((1, 1), device=self.device)
        _ = probe @ probe
        del probe
        return self.refill()

    def refill(self) -> dict[str, Any]:
        """Top up cached pages while preserving both fraction and headroom caps."""

        if not self.active:
            return self.prime()
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        headroom_bytes = self.policy.minimum_headroom_mib * 1024 * 1024
        target_used = min(
            int(total_bytes * self.effective_target_fraction),
            max(0, total_bytes - headroom_bytes),
        )
        observed_used = total_bytes - free_bytes
        requested = max(0, target_used - observed_used)
        chunk_bytes = self.policy.chunk_mib * 1024 * 1024
        blocks: list[torch.Tensor] = []
        remaining = requested
        try:
            while remaining > 0:
                size = min(chunk_bytes, remaining)
                blocks.append(torch.empty(size, dtype=torch.uint8, device=self.device))
                remaining -= size
        except torch.OutOfMemoryError:
            # A concurrent allocation may race the last nvidia-smi snapshot.
            # Keep whatever was obtained and leave the configured headroom.
            remaining = max(0, remaining)
        finally:
            del blocks
        free_after, _ = torch.cuda.mem_get_info(self.device)
        self.last_report = {
            "enabled": True,
            "active": True,
            "retry_count": self.retry_count,
            "configured_target_fraction": self.policy.target_fraction,
            "effective_target_fraction": self.effective_target_fraction,
            "minimum_headroom_mib": self.policy.minimum_headroom_mib,
            "requested_reservation_mib": requested // (1024 * 1024),
            "unfilled_reservation_mib": remaining // (1024 * 1024),
            "allocator_reserved_mib": torch.cuda.memory_reserved(self.device) // (1024 * 1024),
            "allocator_allocated_mib": torch.cuda.memory_allocated(self.device) // (1024 * 1024),
            "device_free_mib": free_after // (1024 * 1024),
            "device_total_mib": total_bytes // (1024 * 1024),
        }
        return dict(self.last_report)

    def release_for_failure(self) -> dict[str, Any]:
        """Return cached pages before propagating a resource failure."""

        if self.active:
            torch.cuda.empty_cache()
        report = dict(self.last_report or {})
        report["released_for_failure"] = True
        return report


__all__ = [
    "CudaMemoryReservationPolicyV1",
    "CudaMemoryReservationV1",
]
