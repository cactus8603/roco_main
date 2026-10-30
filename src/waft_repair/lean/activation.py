"""Bounded CPU offload for tensors repeatedly saved by chunked correspondence.

``save_on_cpu`` copies a CUDA input every time an operation saves it. A target
feature map used by 147 ``grid_sample`` chunks consequently makes 147 host
copies. Here the graph shares one independent CPU snapshot per *live Tensor
object and version*. Different views are deliberately separate snapshots:
their shape, strides and storage offset never become accidental cache aliases.

The cache owns only weak references. It keeps neither source CUDA tensors nor
CPU snapshots alive after the graph releases them. Unpacking makes no persistent
CUDA cache, so GPU memory does not grow with the number of offloaded tensors.
Value/shape/dtype/device are preserved, as required by saved-tensor hooks;
storage aliasing and the original storage offset are not reconstructed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
import weakref

import torch


@dataclass
class _Counters:
    saved_calls: int = 0
    requested_bytes: int = 0
    unique_snapshots: int = 0
    deduplicated_saves: int = 0
    copied_cpu_bytes: int = 0
    resident_cpu_bytes: int = 0
    peak_resident_cpu_bytes: int = 0
    unpack_calls: int = 0
    restored_device_bytes: int = 0


class _Snapshot:
    """Graph-owned snapshot; no strong reference to its source tensor."""

    def __init__(self, cpu, source, version, guards, device, owner):
        self.cpu = cpu
        self.source = weakref.ref(source)
        self.version = version
        self.guards = guards
        self.device = device
        byte_count = cpu.untyped_storage().nbytes()
        weakref.finalize(self, owner._release_bytes, byte_count)


class DeduplicatedSaveOnCPU:
    """Offload saved activations with weak, version-aware snapshot reuse.

    Wrap one training forward/backward, exactly as with ``save_on_cpu``::

        with DeduplicatedSaveOnCPU(pin_memory=False) as offload:
            loss = model(...)
            loss.backward()
        telemetry = offload.stats()

    Copies are synchronous before a snapshot can be reused. ``pin_memory=True``
    uses pinned storage only for CUDA inputs; CPU-only checks initialize no CUDA
    context. CPU inputs also get independent snapshots, which makes the same
    deduplication and mutation checks testable without a GPU.

    A live source or view base changing its PyTorch version after saving raises
    on unpack, rather than silently reusing stale values. Snapshots remain
    correct if the source wrapper dies: immutable CPU values are graph-owned.
    As with native autograd, untracked writes through ``.data`` or foreign code
    are unsupported. Use a fresh instance for every update; instances cannot be
    reentered or reused. Nested hook contexts obey PyTorch's innermost-hook rule.

    ``max_resident_cpu_bytes`` optionally rejects a new snapshot before its
    allocation would exceed the declared snapshot-storage budget. This is not
    a process-memory gate: callers must additionally budget model/input data,
    Python overhead, CUDA reloads, and allocator reservations. Telemetry's
    ``peak_resident_cpu_bytes`` has the same snapshot-storage scope.
    """

    def __init__(self, *, pin_memory=False, max_resident_cpu_bytes=None):
        if type(pin_memory) is not bool:
            raise TypeError('pin_memory must be bool')
        if (max_resident_cpu_bytes is not None and
                (type(max_resident_cpu_bytes) is not int or max_resident_cpu_bytes <= 0)):
            raise ValueError('max_resident_cpu_bytes must be a positive integer or None')
        self.pin_memory = pin_memory
        self.max_resident_cpu_bytes = max_resident_cpu_bytes
        self._cache = weakref.WeakValueDictionary()
        self._counters = _Counters()
        self._lock = threading.RLock()
        self._entered = False
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)

    def __enter__(self):
        if self._entered:
            raise RuntimeError('Use a fresh DeduplicatedSaveOnCPU context per update')
        self._entered = True
        self._hooks.__enter__()
        return self

    def __exit__(self, *args):
        return self._hooks.__exit__(*args)

    def stats(self):
        """Return JSON-safe counters; this method never retains saved tensors."""
        with self._lock:
            result = asdict(self._counters)
            result.update(schema='flow-feature-repair-activation-offload/v1',
                          pin_memory=self.pin_memory,
                          max_resident_cpu_bytes=self.max_resident_cpu_bytes,
                          live_cache_entries=len(self._cache))
            return result

    def _release_bytes(self, byte_count):
        with self._lock:
            self._counters.resident_cpu_bytes -= byte_count

    @staticmethod
    def _guards(tensor):
        guards = []
        source = tensor
        while source is not None:
            guards.append((weakref.ref(source), source._version))
            source = source._base
        return tuple(guards)

    @staticmethod
    def _check_versions(snapshot):
        for reference, version in snapshot.guards:
            live = reference()
            if live is not None and live._version != version:
                raise RuntimeError('A tensor saved for backward was modified in-place after CPU offload')

    def _copy_to_cpu(self, tensor):
        """Single instrumentation point for copy-count and CPU semantic tests."""
        if self.pin_memory and tensor.device.type == 'cuda':
            cpu = torch.empty(tensor.shape, dtype=tensor.dtype, device='cpu', pin_memory=True)
            cpu.copy_(tensor.detach(), non_blocking=False)
            return cpu
        return tensor.detach().to(device='cpu', copy=True, memory_format=torch.preserve_format)

    def _pack(self, tensor):
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise TypeError('Activation offload requires nonquantized strided tensors')
        with self._lock:
            version = tensor._version
            # Object identity is checked again on cache lookup, so an expired
            # Tensor's recycled id (or storage pointer) cannot reuse its values.
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
            if (self.max_resident_cpu_bytes is not None and
                    self._counters.resident_cpu_bytes + required > self.max_resident_cpu_bytes):
                raise MemoryError('CPU activation snapshot budget would be exceeded')
            cpu = self._copy_to_cpu(tensor)
            if any(ref() is not None and ref()._version != expected for ref, expected in guards):
                raise RuntimeError('Tensor changed while copying an activation snapshot')
            byte_count = cpu.untyped_storage().nbytes()
            if (self.max_resident_cpu_bytes is not None and
                    self._counters.resident_cpu_bytes + byte_count > self.max_resident_cpu_bytes):
                raise MemoryError('CPU activation snapshot storage exceeds its budget')
            self._counters.unique_snapshots += 1
            self._counters.copied_cpu_bytes += byte_count
            self._counters.resident_cpu_bytes += byte_count
            self._counters.peak_resident_cpu_bytes = max(
                self._counters.peak_resident_cpu_bytes, self._counters.resident_cpu_bytes)
            snapshot = _Snapshot(cpu, tensor, version, guards, tensor.device, self)
            self._cache[key] = snapshot
            return snapshot

    def _unpack(self, snapshot):
        with self._lock:
            self._check_versions(snapshot)
            self._counters.unpack_calls += 1
            if snapshot.device.type != 'cpu':
                self._counters.restored_device_bytes += snapshot.cpu.numel() * snapshot.cpu.element_size()
            # No strong source reference and no CUDA reload cache. Retain-graph
            # backward can unpack again from the same immutable host snapshot.
            return snapshot.cpu.to(snapshot.device, non_blocking=self.pin_memory)
