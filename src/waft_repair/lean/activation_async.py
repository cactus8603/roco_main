"""Ordered asynchronous CPU activation offload with version-aware deduplication.

CUDA transfers retain the same shape, values and dense layout.
There are no CUDA calls when packing CPU tensors.  CUDA packing copies to a
fresh pinned CPU snapshot on the current stream, records that stream on the
source allocation, and records a ready event.  CUDA unpacking waits for that
event on its own current stream before asynchronously restoring the snapshot.

The host allocation is created by PyTorch's pinned caching allocator.  In the
installed Torch 2.5.1, ``copy_kernel_cuda`` records its transfer stream through
``CachingHostAllocator_recordEvent``: deleting a graph before its DMA completes
therefore cannot recycle that host allocation early.  A Python reference to the
GPU source is never retained.  The snapshot/cache lifetime and version guards
are otherwise the same as ``DeduplicatedSaveOnCPU``.

Callers must obey the ordinary PyTorch stream contract: the pack current stream
must already wait for the stream that produced the saved tensor, and no stream
may concurrently mutate it.  An event orders copies; it cannot repair an
unsynchronized producer or writer.  Run ``torch.cuda.synchronize`` before
reading copy timings or starting the next resource admission decision.

The cap measures graph-owned snapshot storage, not allocator-reserved pinned
memory or outstanding DMA.  The pinned allocator can retain freed blocks, so
an independent RSS/cgroup/headroom gate remains necessary.
"""
from __future__ import annotations

import torch

from .activation import (
    DeduplicatedSaveOnCPU,
    _Snapshot,
)


class _AsyncSnapshot(_Snapshot):
    def __init__(self, cpu, source, version, guards, device, owner, ready_event):
        super().__init__(cpu, source, version, guards, device, owner)
        self.ready_event = ready_event


class AsyncDeduplicatedSaveOnCPU(DeduplicatedSaveOnCPU):
    """Share live, versioned snapshots and use ordered async CUDA transfers.

    The default and maximum accepted graph-owned storage budget is 8 GiB.
    Smaller budgets are useful for testing or a stricter admitted workload.
    There is no asynchronous device reload cache and no GPU source strong ref.

    ``async_d2h=False`` is a diagnostic control: it preserves the same CPU
    layouts and async H2D behavior, but blocks in D2H packing.  This can separate
    a layout change from a transfer-ordering change in a fixed GPU comparison.
    """

    MAX_GRAPH_BYTES = 8 * 1024 ** 3

    def __init__(self, *, max_resident_cpu_bytes=MAX_GRAPH_BYTES, async_d2h=True):
        if (type(max_resident_cpu_bytes) is not int or
                not 0 < max_resident_cpu_bytes <= self.MAX_GRAPH_BYTES):
            raise ValueError('snapshot budget must be a positive integer at most 8 GiB')
        if type(async_d2h) is not bool:
            raise TypeError('async_d2h must be bool')
        super().__init__(pin_memory=True, max_resident_cpu_bytes=max_resident_cpu_bytes)
        self.async_d2h = async_d2h
        self._async_cuda_snapshots = 0
        self._cuda_event_waits = 0

    def stats(self):
        result = super().stats()
        with self._lock:
            result.update(
                schema='flow-feature-repair-activation-offload/async-v1',
                preserve_format=True,
                async_d2h=self.async_d2h,
                async_cuda_snapshots=self._async_cuda_snapshots,
                cuda_event_waits=self._cuda_event_waits,
            )
        return result

    def _pack(self, tensor):
        if tensor.device.type != 'cuda':
            # In particular, pinning CPU-only tests must not create CUDA state.
            return super()._pack(tensor)
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise TypeError('Activation offload requires nonquantized strided tensors')
        with self._lock:
            version = tensor._version
            key = (id(tensor), version, tensor.device, tensor.dtype,
                   tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(),
                   tensor.is_conj(), tensor.is_neg())
            self._counters.saved_calls += 1
            self._counters.requested_bytes += tensor.numel() * tensor.element_size()
            existing = self._cache.get(key)
            if existing is not None and existing.source() is tensor:
                self._check_versions(existing)
                self._counters.deduplicated_saves += 1
                return existing
            guards = self._guards(tensor)
            required = tensor.numel() * tensor.element_size()
            if self._counters.resident_cpu_bytes + required > self.max_resident_cpu_bytes:
                raise MemoryError('CPU activation snapshot budget would be exceeded')

            # Match the original non-pinned to(copy=True, preserve_format) path.
            # torch.empty(shape) would silently make transpose/channels-last
            # snapshots contiguous and can change backward kernel numerics.
            cpu = torch.empty_like(tensor, device='cpu', pin_memory=True,
                                   memory_format=torch.preserve_format)
            byte_count = cpu.untyped_storage().nbytes()
            if self._counters.resident_cpu_bytes + byte_count > self.max_resident_cpu_bytes:
                raise MemoryError('CPU activation snapshot storage exceeds its budget')
            with torch.cuda.device(tensor.device):
                stream = torch.cuda.current_stream(tensor.device)
                # Protect the source allocation after its Python tensor dies.
                tensor.record_stream(stream)
                cpu.copy_(tensor.detach(), non_blocking=self.async_d2h)
                ready_event = None
                if self.async_d2h:
                    ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                    ready_event.record(stream)
            if any(ref() is not None and ref()._version != expected for ref, expected in guards):
                raise RuntimeError('Tensor changed while copying an activation snapshot')
            self._counters.unique_snapshots += 1
            self._counters.copied_cpu_bytes += byte_count
            self._counters.resident_cpu_bytes += byte_count
            self._counters.peak_resident_cpu_bytes = max(
                self._counters.peak_resident_cpu_bytes, self._counters.resident_cpu_bytes)
            snapshot = _AsyncSnapshot(cpu, tensor, version, guards, tensor.device,
                                      self, ready_event)
            self._async_cuda_snapshots += int(self.async_d2h)
            self._cache[key] = snapshot
            return snapshot

    def _unpack(self, snapshot):
        if snapshot.device.type != 'cuda':
            return super()._unpack(snapshot)
        with self._lock:
            self._check_versions(snapshot)
            self._counters.unpack_calls += 1
            self._counters.restored_device_bytes += (
                snapshot.cpu.numel() * snapshot.cpu.element_size())
            with torch.cuda.device(snapshot.device):
                stream = torch.cuda.current_stream(snapshot.device)
                if snapshot.ready_event is not None:
                    stream.wait_event(snapshot.ready_event)
                    self._cuda_event_waits += 1
                # The pinned allocator tracks this transfer stream as well.
                return snapshot.cpu.to(snapshot.device, non_blocking=True,
                                       memory_format=torch.preserve_format)
