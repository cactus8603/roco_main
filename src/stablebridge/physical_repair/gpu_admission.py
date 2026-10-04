"""Pure, provider-neutral GPU admission policy.

The module deliberately does not execute ``nvidia-smi``, launch a worker, or
touch the filesystem.  A provider supplies two independently collected GPU
snapshots; this module parses/normalizes them and makes a deterministic,
fail-closed admission decision.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
import posixpath
from pathlib import PurePosixPath
import re
from typing import Mapping, Sequence


NVIDIA_SMI_COLUMNS = (
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "temperature.gpu",
)

_COLUMN_ALIASES = {
    "gpu_uuid": "uuid",
    "memory_total": "memory.total",
    "memory_used": "memory.used",
    "memory_free": "memory.free",
    "utilization_gpu": "utilization.gpu",
    "temperature_gpu": "temperature.gpu",
    "compute_processes": "compute_process_count",
}
_INTEGER_PREFIX = re.compile(r"^[+]?(\d+)(?:\.0+)?(?:\s|$)")
_FORBIDDEN_WRITE_ROOTS = (PurePosixPath("/tmp"), PurePosixPath("/ssd8"))


class GPUAdmissionError(RuntimeError):
    """The supplied evidence cannot safely authorize GPU execution."""


@dataclass(frozen=True)
class GPUSnapshot:
    """One normalized physical GPU observation.

    ``compute_process_count=None`` means the provider did not observe process
    evidence.  Such a GPU is never classified as empty, but it may still be a
    shared-capacity candidate.
    """

    index: int
    uuid: str
    name: str
    total_memory_mib: int
    used_memory_mib: int
    free_memory_mib: int
    utilization_percent: int
    temperature_celsius: int
    compute_process_count: int | None = None

    def __post_init__(self) -> None:
        integer_fields = {
            "index": self.index,
            "total_memory_mib": self.total_memory_mib,
            "used_memory_mib": self.used_memory_mib,
            "free_memory_mib": self.free_memory_mib,
            "utilization_percent": self.utilization_percent,
            "temperature_celsius": self.temperature_celsius,
        }
        for field, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if not self.uuid.strip():
            raise ValueError("GPU uuid must be non-empty")
        if self.used_memory_mib + self.free_memory_mib > self.total_memory_mib:
            raise ValueError("used plus free GPU memory exceeds total memory")
        if self.utilization_percent > 100:
            raise ValueError("GPU utilization must be at most 100 percent")
        count = self.compute_process_count
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 0
        ):
            raise ValueError("compute_process_count must be non-negative or None")

    @property
    def is_empty(self) -> bool:
        return self.compute_process_count == 0


@dataclass(frozen=True)
class GPUAdmissionPolicy:
    """Frozen resource requirements for one worker."""

    allowed_uuids: frozenset[str]
    peak_memory_mib: int
    safety_margin_mib: int

    def __post_init__(self) -> None:
        if not self.allowed_uuids or any(not item.strip() for item in self.allowed_uuids):
            raise ValueError("allowed_uuids must be a non-empty UUID allowlist")
        for field in ("peak_memory_mib", "safety_margin_mib"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.peak_memory_mib == 0:
            raise ValueError("peak_memory_mib must be positive")

    @property
    def required_free_memory_mib(self) -> int:
        return self.peak_memory_mib + self.safety_margin_mib


@dataclass(frozen=True)
class GPUAdmissionDecision:
    """Deterministic decision derived from one snapshot."""

    gpu: GPUSnapshot
    mode: str
    required_free_memory_mib: int

    def __post_init__(self) -> None:
        if self.mode not in {"empty", "shared"}:
            raise ValueError("admission mode must be 'empty' or 'shared'")


@dataclass(frozen=True)
class ValidatedWriteRoots:
    """Lexically normalized roots validated without filesystem access."""

    runtime_root: str
    output_root: str
    cache_roots: tuple[tuple[str, str], ...]


def _canonical_column(value: str) -> str:
    # ``nvidia-smi --format=csv`` appends units to headers unless ``nounits``
    # is requested (for example, ``memory.free [MiB]``).
    normalized = re.sub(r"\s*\[[^]]+\]\s*$", "", value.strip().lower())
    normalized = normalized.replace(" ", "_")
    return _COLUMN_ALIASES.get(normalized, normalized)


def _parse_nonnegative_integer(value: str, *, field: str) -> int:
    match = _INTEGER_PREFIX.match(value.strip())
    if match is None:
        raise ValueError(f"invalid {field} value: {value!r}")
    parsed = int(match.group(1))
    suffix = value.strip()[match.end():].strip().lower()
    if suffix not in {"", "mib", "%", "c"}:
        raise ValueError(f"invalid {field} unit: {value!r}")
    return parsed


def parse_nvidia_smi_csv(
    text: str,
    *,
    process_counts_by_uuid: Mapping[str, int] | None = None,
    columns: Sequence[str] = NVIDIA_SMI_COLUMNS,
) -> tuple[GPUSnapshot, ...]:
    """Parse ``nvidia-smi --format=csv`` output into generic snapshots.

    Both header and ``noheader`` output are accepted.  The standard eight GPU
    query columns are given by :data:`NVIDIA_SMI_COLUMNS`.  A provider may add
    a ``compute_process_count`` column, or join results from
    ``--query-compute-apps`` through ``process_counts_by_uuid``.  Missing
    process evidence remains unknown rather than being treated as zero.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("nvidia-smi CSV must be non-empty text")
    canonical_columns = tuple(_canonical_column(item) for item in columns)
    required = set(NVIDIA_SMI_COLUMNS)
    if not required.issubset(canonical_columns):
        missing = sorted(required.difference(canonical_columns))
        raise ValueError(f"nvidia-smi columns missing required fields: {missing}")
    if len(set(canonical_columns)) != len(canonical_columns):
        raise ValueError("nvidia-smi columns contain duplicates")

    rows = [row for row in csv.reader(StringIO(text)) if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("nvidia-smi CSV has no rows")
    first = tuple(_canonical_column(item) for item in rows[0])
    if required.issubset(first):
        active_columns = first
        data_rows = rows[1:]
    else:
        active_columns = canonical_columns
        data_rows = rows
    if not data_rows:
        raise ValueError("nvidia-smi CSV has no GPU data rows")

    normalized_counts: dict[str, int] | None = None
    if process_counts_by_uuid is not None:
        normalized_counts = {}
        for uuid, count in process_counts_by_uuid.items():
            if not str(uuid).strip():
                raise ValueError("process count UUID must be non-empty")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("process counts must be non-negative integers")
            normalized_counts[str(uuid).strip()] = count

    snapshots: list[GPUSnapshot] = []
    seen_uuids: set[str] = set()
    seen_indices: set[int] = set()
    for row_number, row in enumerate(data_rows, start=2 if data_rows is not rows else 1):
        if len(row) != len(active_columns):
            raise ValueError(
                f"nvidia-smi CSV row {row_number} has {len(row)} fields; "
                f"expected {len(active_columns)}"
            )
        values = dict(zip(active_columns, (item.strip() for item in row)))
        uuid = values["uuid"]
        index = _parse_nonnegative_integer(values["index"], field="index")
        if uuid in seen_uuids or index in seen_indices:
            raise ValueError("nvidia-smi CSV contains duplicate GPU UUID or index")
        seen_uuids.add(uuid)
        seen_indices.add(index)
        inline_count = values.get("compute_process_count")
        if inline_count is not None:
            process_count: int | None = _parse_nonnegative_integer(
                inline_count, field="compute_process_count",
            )
            if normalized_counts is not None and uuid in normalized_counts:
                if process_count != normalized_counts[uuid]:
                    raise ValueError("conflicting compute process counts")
        elif normalized_counts is not None:
            process_count = normalized_counts.get(uuid)
        else:
            process_count = None
        snapshots.append(GPUSnapshot(
            index=index,
            uuid=uuid,
            name=values["name"],
            total_memory_mib=_parse_nonnegative_integer(
                values["memory.total"], field="memory.total",
            ),
            used_memory_mib=_parse_nonnegative_integer(
                values["memory.used"], field="memory.used",
            ),
            free_memory_mib=_parse_nonnegative_integer(
                values["memory.free"], field="memory.free",
            ),
            utilization_percent=_parse_nonnegative_integer(
                values["utilization.gpu"], field="utilization.gpu",
            ),
            temperature_celsius=_parse_nonnegative_integer(
                values["temperature.gpu"], field="temperature.gpu",
            ),
            compute_process_count=process_count,
        ))
    return tuple(snapshots)


def choose_gpu(
    snapshots: Sequence[GPUSnapshot],
    policy: GPUAdmissionPolicy,
) -> GPUAdmissionDecision:
    """Choose one admissible GPU, preferring empty cards deterministically."""

    if not snapshots:
        raise GPUAdmissionError("GPU snapshot is empty")
    uuids = [item.uuid for item in snapshots]
    indices = [item.index for item in snapshots]
    if len(set(uuids)) != len(uuids) or len(set(indices)) != len(indices):
        raise GPUAdmissionError("GPU snapshot contains duplicate UUID or index")
    candidates = [
        item for item in snapshots
        if item.uuid in policy.allowed_uuids
        and item.free_memory_mib >= policy.required_free_memory_mib
    ]
    if not candidates:
        raise GPUAdmissionError(
            "no allowlisted GPU has the required free memory "
            f"({policy.required_free_memory_mib} MiB)"
        )
    # Empty first, then most free memory, lowest utilization/temperature, and
    # finally stable physical identity.  UUID provides a final total ordering.
    chosen = min(candidates, key=lambda item: (
        0 if item.is_empty else 1,
        -item.free_memory_mib,
        item.utilization_percent,
        item.temperature_celsius,
        item.index,
        item.uuid,
    ))
    return GPUAdmissionDecision(
        gpu=chosen,
        mode="empty" if chosen.is_empty else "shared",
        required_free_memory_mib=policy.required_free_memory_mib,
    )


def verify_second_snapshot(
    initial: GPUAdmissionDecision,
    second_snapshots: Sequence[GPUSnapshot],
    policy: GPUAdmissionPolicy,
) -> GPUAdmissionDecision:
    """Fail closed unless a fresh snapshot reproduces the initial decision."""

    if initial.required_free_memory_mib != policy.required_free_memory_mib:
        raise GPUAdmissionError("initial decision does not bind the active memory policy")
    expected_initial_mode = "empty" if initial.gpu.is_empty else "shared"
    if initial.mode != expected_initial_mode:
        raise GPUAdmissionError("initial decision mode contradicts its GPU snapshot")
    if initial.gpu.uuid not in policy.allowed_uuids:
        raise GPUAdmissionError("initial decision GPU is not allowlisted")
    if initial.gpu.free_memory_mib < policy.required_free_memory_mib:
        raise GPUAdmissionError("initial decision lacks its required free memory")
    refreshed = choose_gpu(second_snapshots, policy)
    if refreshed.gpu.uuid != initial.gpu.uuid:
        raise GPUAdmissionError("fresh snapshot selected a different GPU")
    if refreshed.gpu.index != initial.gpu.index:
        raise GPUAdmissionError("GPU UUID moved to a different physical index")
    if (
        refreshed.gpu.name != initial.gpu.name
        or refreshed.gpu.total_memory_mib != initial.gpu.total_memory_mib
    ):
        raise GPUAdmissionError("GPU physical identity changed between snapshots")
    if refreshed.mode != initial.mode:
        raise GPUAdmissionError("GPU admission mode changed between snapshots")
    return refreshed


def admit_gpu(
    first_snapshots: Sequence[GPUSnapshot],
    second_snapshots: Sequence[GPUSnapshot],
    policy: GPUAdmissionPolicy,
) -> GPUAdmissionDecision:
    """Select from the first snapshot and verify against an independent second."""

    return verify_second_snapshot(choose_gpu(first_snapshots, policy), second_snapshots, policy)


def _validate_write_root(path: str | PurePosixPath, *, label: str) -> str:
    raw = str(path)
    if not raw or "\x00" in raw:
        raise ValueError(f"{label} must be a non-empty path")
    normalized = posixpath.normpath(raw)
    # POSIX permits implementation-defined semantics for exactly two leading
    # slashes.  Treat all repeated leading slashes as the ordinary root so
    # ``//tmp`` and ``//ssd8`` cannot bypass the denylist.
    if normalized.startswith("/"):
        normalized = "/" + normalized.lstrip("/")
    candidate = PurePosixPath(normalized)
    if not candidate.is_absolute():
        raise ValueError(f"{label} must be absolute")
    for forbidden in _FORBIDDEN_WRITE_ROOTS:
        if candidate == forbidden or forbidden in candidate.parents:
            raise ValueError(f"{label} points inside forbidden root {forbidden}")
    return str(candidate)


def validate_write_roots(
    *,
    runtime_root: str | PurePosixPath,
    output_root: str | PurePosixPath,
    cache_roots: Mapping[str, str | PurePosixPath] | None = None,
) -> ValidatedWriteRoots:
    """Validate runtime/output/cache roots without resolving or touching them.

    This is intentionally lexical.  A launcher that accepts symlinks must pass
    separately provider-resolved paths through this same function before use.
    """

    caches: list[tuple[str, str]] = []
    for name, path in sorted((cache_roots or {}).items()):
        if not name:
            raise ValueError("cache root name must be non-empty")
        caches.append((name, _validate_write_root(path, label=f"cache root {name!r}")))
    return ValidatedWriteRoots(
        runtime_root=_validate_write_root(runtime_root, label="runtime root"),
        output_root=_validate_write_root(output_root, label="output root"),
        cache_roots=tuple(caches),
    )


# Concise compatibility name for callers that frame admission as selection.
select_gpu = choose_gpu


__all__ = [
    "GPUAdmissionDecision",
    "GPUAdmissionError",
    "GPUAdmissionPolicy",
    "GPUSnapshot",
    "NVIDIA_SMI_COLUMNS",
    "ValidatedWriteRoots",
    "admit_gpu",
    "choose_gpu",
    "parse_nvidia_smi_csv",
    "select_gpu",
    "validate_write_roots",
    "verify_second_snapshot",
]
