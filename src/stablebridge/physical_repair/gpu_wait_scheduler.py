"""Durable single-job GPU wait scheduler.

The scheduler polls ``nvidia-smi``, chooses the least busy card satisfying a
free-memory and utilization threshold, acquires an atomic per-GPU lock, checks
the card a second time, and launches one command with ``CUDA_VISIBLE_DEVICES``.
It is intentionally small: this is a durable launcher for one experiment, not
a cluster scheduler.

``start`` detaches the watchdog from the caller.  A user systemd transient
service is preferred; a ``start_new_session`` watchdog is the fallback.
``run`` is the foreground watchdog used by those backends.  ``status`` and
``dry-run`` never launch the training command.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid as uuid_module


SCHEDULER_SCHEMA_V1 = "stablebridge-gpu-wait-scheduler/v1"
_NVIDIA_QUERY = (
    "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu"
)
_RETRYABLE_RESOURCE_FAILURES = (
    ("cublas_status_alloc_failed", "CUBLAS_STATUS_ALLOC_FAILED"),
    ("cudnn_status_alloc_failed", "CUDNN_STATUS_ALLOC_FAILED"),
    ("torch.outofmemoryerror", "TORCH_CUDA_OUT_OF_MEMORY"),
    ("cuda out of memory", "CUDA_OUT_OF_MEMORY"),
    ("cuda error: out of memory", "CUDA_OUT_OF_MEMORY"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def classify_retryable_resource_failure(text: str) -> str | None:
    """Return a stable reason for explicit CUDA allocation failures only."""

    lowered = text.lower()
    for needle, reason in _RETRYABLE_RESOURCE_FAILURES:
        if needle in lowered:
            return reason
    return None


def _read_text_suffix(path: Path, offset: int) -> str:
    """Read log bytes written after ``offset``, tolerating rotation/truncation."""

    if not path.is_file():
        return ""
    size = path.stat().st_size
    start = offset if 0 <= offset <= size else 0
    with path.open("rb") as stream:
        stream.seek(start)
        return stream.read().decode("utf-8", errors="replace")


@dataclass(frozen=True)
class GPUStatusV1:
    index: int
    uuid: str
    name: str
    total_memory_mib: int
    used_memory_mib: int
    free_memory_mib: int
    utilization_percent: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "total_memory_mib": self.total_memory_mib,
            "used_memory_mib": self.used_memory_mib,
            "free_memory_mib": self.free_memory_mib,
            "utilization_percent": self.utilization_percent,
        }


def _integer(value: str, field: str) -> int:
    stripped = value.strip()
    if not re.fullmatch(r"\d+", stripped):
        raise ValueError(f"invalid nvidia-smi {field}: {value!r}")
    return int(stripped)


def parse_gpu_csv(text: str) -> tuple[GPUStatusV1, ...]:
    """Parse the exact no-header/no-units query used by the watchdog."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("nvidia-smi returned no GPU rows")
    rows = [row for row in csv.reader(StringIO(text)) if any(cell.strip() for cell in row)]
    statuses: list[GPUStatusV1] = []
    seen_indices: set[int] = set()
    seen_uuids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 7:
            raise ValueError(f"nvidia-smi row {row_number} has {len(row)} columns, expected 7")
        index = _integer(row[0], "index")
        gpu_uuid = row[1].strip()
        if not gpu_uuid:
            raise ValueError("nvidia-smi GPU UUID is empty")
        if index in seen_indices or gpu_uuid in seen_uuids:
            raise ValueError("nvidia-smi returned duplicate GPU identity")
        seen_indices.add(index)
        seen_uuids.add(gpu_uuid)
        total = _integer(row[3], "memory.total")
        used = _integer(row[4], "memory.used")
        free = _integer(row[5], "memory.free")
        utilization = _integer(row[6], "utilization.gpu")
        if used + free > total or utilization > 100:
            raise ValueError("nvidia-smi returned inconsistent GPU capacity")
        statuses.append(GPUStatusV1(
            index=index,
            uuid=gpu_uuid,
            name=row[2].strip(),
            total_memory_mib=total,
            used_memory_mib=used,
            free_memory_mib=free,
            utilization_percent=utilization,
        ))
    return tuple(statuses)


def choose_gpu(
    statuses: Sequence[GPUStatusV1],
    *,
    minimum_free_mib: int,
    maximum_utilization_percent: int,
    allowed_indices: frozenset[int] | None = None,
) -> GPUStatusV1 | None:
    """Choose most-free, then least-utilized, then lowest-index GPU."""

    if minimum_free_mib <= 0:
        raise ValueError("minimum free memory must be positive")
    if not 0 <= maximum_utilization_percent <= 100:
        raise ValueError("maximum utilization must lie in [0,100]")
    candidates = [
        item for item in statuses
        if item.free_memory_mib >= minimum_free_mib
        and item.utilization_percent <= maximum_utilization_percent
        and (allowed_indices is None or item.index in allowed_indices)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (
        -item.free_memory_mib, item.utilization_percent, item.index, item.uuid,
    ))


def build_child_environment(
    base: Mapping[str, str], selected: GPUStatusV1, *, resource_retry_count: int = 0,
) -> dict[str, str]:
    if isinstance(resource_retry_count, bool) or resource_retry_count < 0:
        raise ValueError("resource retry count must be nonnegative")
    result = dict(base)
    result["CUDA_VISIBLE_DEVICES"] = str(selected.index)
    result["STABLEBRIDGE_PHYSICAL_GPU_UUID"] = selected.uuid
    result["STABLEBRIDGE_PHYSICAL_GPU_INDEX"] = str(selected.index)
    result["STABLEBRIDGE_RESOURCE_RETRY_COUNT"] = str(resource_retry_count)
    result.setdefault("PYTHONUNBUFFERED", "1")
    return result


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class AtomicProcessLock:
    """O_EXCL lock with PID/token ownership and conservative stale recovery."""

    def __init__(self, path: Path):
        self.path = path
        self.token = uuid_module.uuid4().hex
        self.owned = False

    def _payload(self) -> dict[str, Any]:
        return {"pid": os.getpid(), "token": self.token, "created_utc": _utc_now()}

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(2):
            try:
                descriptor = os.open(
                    self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                )
            except FileExistsError:
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                    existing_pid = int(existing["pid"])
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    # An unreadable lock may still be under construction.
                    return False
                if _pid_alive(existing_pid):
                    return False
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            try:
                os.write(descriptor, (json.dumps(self._payload()) + "\n").encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.owned = True
            return True
        return False

    def release(self) -> None:
        if not self.owned:
            return
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if existing.get("token") == self.token and int(existing.get("pid", -1)) == os.getpid():
                self.path.unlink(missing_ok=True)
        finally:
            self.owned = False

    def __enter__(self) -> "AtomicProcessLock":
        if not self.acquire():
            raise FileExistsError(f"active lock already exists: {self.path}")
        return self

    def __exit__(self, *_args: Any) -> None:
        self.release()


@dataclass(frozen=True)
class SchedulerOptionsV1:
    state_dir: Path
    minimum_free_mib: int = 16_000
    maximum_utilization_percent: int = 10
    poll_seconds: float = 30.0
    allowed_indices: frozenset[int] | None = None
    gpu_lock_dir: Path | None = None
    max_resource_retries: int = 0

    def __post_init__(self) -> None:
        if self.minimum_free_mib <= 0:
            raise ValueError("minimum free memory must be positive")
        if not 0 <= self.maximum_utilization_percent <= 100:
            raise ValueError("maximum utilization must lie in [0,100]")
        if not 1 <= self.poll_seconds <= 3600:
            raise ValueError("poll interval must lie in [1,3600] seconds")
        if self.allowed_indices is not None and (
            not self.allowed_indices or any(index < 0 for index in self.allowed_indices)
        ):
            raise ValueError("allowed GPU indices must be a non-empty non-negative set")
        if not 0 <= self.max_resource_retries <= 10:
            raise ValueError("maximum resource retries must lie in [0,10]")

    @property
    def resolved_gpu_lock_dir(self) -> Path:
        if self.gpu_lock_dir is not None:
            return self.gpu_lock_dir
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        root = Path(runtime) if runtime else Path("/tmp") / f"stablebridge-{os.getuid()}"
        return root / "gpu-locks"


def query_gpus(*, timeout_seconds: float = 15.0) -> tuple[GPUStatusV1, ...]:
    process = subprocess.run(
        [
            "nvidia-smi", f"--query-gpu={_NVIDIA_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "unknown error"
        raise RuntimeError(f"nvidia-smi failed: {detail}")
    return parse_gpu_csv(process.stdout)


class GPUWaitSchedulerV1:
    def __init__(self, options: SchedulerOptionsV1, command: Sequence[str]):
        if not command or any(not isinstance(item, str) or not item or "\0" in item for item in command):
            raise ValueError("training command needs non-empty, NUL-free argv")
        self.options = options
        self.command = tuple(command)
        self.status_path = options.state_dir / "status.json"
        self.scheduler_lock = AtomicProcessLock(options.state_dir / "scheduler.lock")
        self.stop_requested: int | None = None
        self.child: subprocess.Popen[str] | None = None

    def _write_status(self, state: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "schema": SCHEDULER_SCHEMA_V1,
            "state": state,
            "updated_utc": _utc_now(),
            "scheduler_pid": os.getpid(),
            "command": list(self.command),
            "minimum_free_mib": self.options.minimum_free_mib,
            "maximum_utilization_percent": self.options.maximum_utilization_percent,
            "poll_seconds": self.options.poll_seconds,
            "max_resource_retries": self.options.max_resource_retries,
            "allowed_indices": (
                None if self.options.allowed_indices is None
                else sorted(self.options.allowed_indices)
            ),
        }
        payload.update(extra)
        _atomic_json(self.status_path, payload)

    def _signal(self, signum: int, _frame: Any) -> None:
        if self.stop_requested is None:
            self.stop_requested = signum
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signum)

    def _sleep(self) -> None:
        deadline = time.monotonic() + self.options.poll_seconds
        while self.stop_requested is None and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))

    def run(self) -> int:
        self.options.state_dir.mkdir(parents=True, exist_ok=True)
        if not self.scheduler_lock.acquire():
            # Do not overwrite the authoritative status owned by the active
            # watchdog merely because a duplicate invocation was attempted.
            _atomic_json(self.options.state_dir / f"duplicate-{os.getpid()}.json", {
                "schema": SCHEDULER_SCHEMA_V1,
                "state": "duplicate_rejected",
                "updated_utc": _utc_now(),
                "scheduler_pid": os.getpid(),
                "error": "another watchdog owns scheduler.lock",
            })
            return 73
        previous: dict[int, Any] = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, self._signal)
        try:
            self._write_status("waiting")
            resource_retry_count = 0
            while self.stop_requested is None:
                try:
                    snapshots = query_gpus()
                    selected = choose_gpu(
                        snapshots,
                        minimum_free_mib=self.options.minimum_free_mib,
                        maximum_utilization_percent=self.options.maximum_utilization_percent,
                        allowed_indices=self.options.allowed_indices,
                    )
                except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                    self._write_status("waiting", last_query_error=str(error))
                    self._sleep()
                    continue
                if selected is None:
                    self._write_status("waiting", observed_gpus=[item.as_dict() for item in snapshots])
                    self._sleep()
                    continue
                safe_uuid = re.sub(r"[^A-Za-z0-9_.-]", "_", selected.uuid)
                gpu_lock = AtomicProcessLock(
                    self.options.resolved_gpu_lock_dir / f"{safe_uuid}.lock",
                )
                if not gpu_lock.acquire():
                    self._write_status("waiting", candidate_gpu=selected.as_dict(), reason="gpu_lock_busy")
                    self._sleep()
                    continue
                try:
                    # Re-query after lock acquisition; fail closed if identity or
                    # capacity changed between observation and reservation.
                    second = query_gpus()
                    verified = next((item for item in second if item.uuid == selected.uuid), None)
                    if verified is None or choose_gpu(
                        [verified],
                        minimum_free_mib=self.options.minimum_free_mib,
                        maximum_utilization_percent=self.options.maximum_utilization_percent,
                        allowed_indices=self.options.allowed_indices,
                    ) is None:
                        self._write_status("waiting", candidate_gpu=selected.as_dict(), reason="second_snapshot_rejected")
                        gpu_lock.release()
                        self._sleep()
                        continue
                    environment = build_child_environment(
                        os.environ, verified,
                        resource_retry_count=resource_retry_count,
                    )
                    stderr_path = self.options.state_dir / "watchdog.stderr.log"
                    stderr_offset = (
                        stderr_path.stat().st_size if stderr_path.is_file() else 0
                    )
                    self._write_status(
                        "launching", selected_gpu=verified.as_dict(),
                        resource_retry_count=resource_retry_count,
                    )
                    self.child = subprocess.Popen(
                        list(self.command), env=environment, stdin=subprocess.DEVNULL,
                        text=True, start_new_session=False,
                    )
                    self._write_status(
                        "running", selected_gpu=verified.as_dict(), child_pid=self.child.pid,
                        resource_retry_count=resource_retry_count,
                    )
                    while self.child.poll() is None:
                        time.sleep(1.0)
                    returncode = int(self.child.returncode)
                    retry_reason = classify_retryable_resource_failure(
                        _read_text_suffix(stderr_path, stderr_offset)
                    )
                    if (
                        returncode != 0
                        and self.stop_requested is None
                        and retry_reason is not None
                        and resource_retry_count < self.options.max_resource_retries
                    ):
                        resource_retry_count += 1
                        self.child = None
                        self._write_status(
                            "waiting", reason="retryable_resource_failure",
                            last_retry_reason=retry_reason,
                            last_child_returncode=returncode,
                            previous_gpu=verified.as_dict(),
                            resource_retry_count=resource_retry_count,
                        )
                        # Do not hold the GPU reservation during backoff.  The
                        # next pass re-runs both admission snapshots.
                        gpu_lock.release()
                        self._sleep()
                        continue
                    state = "completed" if returncode == 0 else (
                        "stopped" if self.stop_requested is not None else "failed"
                    )
                    self._write_status(
                        state, selected_gpu=verified.as_dict(), child_pid=self.child.pid,
                        child_returncode=returncode,
                        resource_retry_count=resource_retry_count,
                        retryable_resource_failure=retry_reason,
                    )
                    return returncode
                except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                    self._write_status("waiting", candidate_gpu=selected.as_dict(), launch_error=str(error))
                    gpu_lock.release()
                    self._sleep()
                finally:
                    gpu_lock.release()
            self._write_status("stopped", signal=self.stop_requested)
            return 128 + int(self.stop_requested)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            self.scheduler_lock.release()


def build_watchdog_argv(
    script_path: Path,
    options: SchedulerOptionsV1,
    command: Sequence[str],
) -> list[str]:
    argv = [
        # Preserve a virtual-environment interpreter path.  ``resolve()`` can
        # collapse its python symlink to /usr/bin/python and silently drop the
        # environment that contains torch/project dependencies.
        str(Path(sys.executable).absolute()), str(script_path.resolve()), "run",
        "--state-dir", str(options.state_dir.resolve()),
        "--minimum-free-mib", str(options.minimum_free_mib),
        "--maximum-utilization-percent", str(options.maximum_utilization_percent),
        "--poll-seconds", str(options.poll_seconds),
        "--max-resource-retries", str(options.max_resource_retries),
    ]
    if options.gpu_lock_dir is not None:
        argv.extend(["--gpu-lock-dir", str(options.gpu_lock_dir.resolve())])
    if options.allowed_indices is not None:
        argv.extend(["--allowed-indices", ",".join(str(item) for item in sorted(options.allowed_indices))])
    argv.extend(["--", *command])
    return argv


def _systemd_user_available() -> bool:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _tmux_available() -> bool:
    executable = shutil.which("tmux")
    if executable is None:
        return False
    try:
        result = subprocess.run(
            [executable, "list-sessions"], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    # Exit 1 also means tmux is installed but no server/sessions exist yet.
    return result.returncode in {0, 1}


def start_detached(
    options: SchedulerOptionsV1,
    command: Sequence[str],
    *,
    script_path: Path,
    backend: str = "auto",
) -> dict[str, Any]:
    """Start the watchdog independently and return a persistent launch record."""

    if backend not in {"auto", "systemd", "tmux", "setsid"}:
        raise ValueError("backend must be auto, systemd, tmux, or setsid")
    options.state_dir.mkdir(parents=True, exist_ok=True)
    existing = read_status(options.state_dir)
    if existing and existing.get("state") in {"waiting", "launching", "running"}:
        pid = int(existing.get("scheduler_pid", -1))
        if _pid_alive(pid):
            raise FileExistsError(f"scheduler is already active with PID {pid}")
    watchdog = build_watchdog_argv(script_path, options, command)
    selected_backend = backend
    if backend == "auto":
        selected_backend = (
            "systemd" if _systemd_user_available()
            else "tmux" if _tmux_available()
            else "setsid"
        )
    stdout_path = options.state_dir / "watchdog.stdout.log"
    stderr_path = options.state_dir / "watchdog.stderr.log"
    launch: dict[str, Any] = {
        "schema": SCHEDULER_SCHEMA_V1,
        "created_utc": _utc_now(),
        "backend": selected_backend,
        "watchdog_argv": watchdog,
        "training_command": list(command),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    if selected_backend == "systemd":
        digest = hashlib.sha256(str(options.state_dir.resolve()).encode()).hexdigest()[:12]
        unit = f"stablebridge-u0-gpu-wait-{digest}.service"
        launch_argv = [
            "systemd-run", "--user", f"--unit={unit}", "--collect",
            "--service-type=exec", "--property=KillMode=control-group",
            f"--property=StandardOutput=append:{stdout_path}",
            f"--property=StandardError=append:{stderr_path}", "--", *watchdog,
        ]
        result = subprocess.run(launch_argv, text=True, capture_output=True, check=False, timeout=30)
        launch.update({
            "unit": unit, "launcher_returncode": result.returncode,
            "launcher_stdout": result.stdout, "launcher_stderr": result.stderr,
            "status": "submitted" if result.returncode == 0 else "launch_failed",
        })
        if result.returncode != 0:
            _atomic_json(options.state_dir / "launch.json", launch)
            raise RuntimeError(f"systemd-run failed: {result.stderr.strip()}")
    elif selected_backend == "tmux":
        digest = hashlib.sha256(str(options.state_dir.resolve()).encode()).hexdigest()[:12]
        session = f"stablebridge-u0-{digest}"
        launch_argv = ["tmux", "new-session", "-d", "-s", session, *watchdog]
        result = subprocess.run(
            launch_argv, text=True, capture_output=True, check=False, timeout=30,
        )
        launch.update({
            "tmux_session": session,
            "launcher_returncode": result.returncode,
            "launcher_stdout": result.stdout,
            "launcher_stderr": result.stderr,
            "status": "submitted" if result.returncode == 0 else "launch_failed",
        })
        if result.returncode == 0:
            subprocess.run(
                ["tmux", "set-option", "-t", session, "remain-on-exit", "on"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, check=False, timeout=10,
            )
        else:
            _atomic_json(options.state_dir / "launch.json", launch)
            raise RuntimeError(f"tmux launch failed: {result.stderr.strip()}")
    else:
        with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open("ab", buffering=0) as stderr:
            child = subprocess.Popen(
                watchdog, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                cwd=str(Path.cwd()), start_new_session=True, close_fds=True,
            )
        launch.update({"watchdog_pid": child.pid, "status": "submitted"})
    _atomic_json(options.state_dir / "launch.json", launch)
    return launch


def read_status(state_dir: Path) -> dict[str, Any] | None:
    path = state_dir / "status.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("scheduler status must be a JSON object")
    pid = int(value.get("scheduler_pid", -1))
    value["scheduler_pid_alive"] = _pid_alive(pid)
    return value


def _allowed_indices(text: str | None) -> frozenset[int] | None:
    if text is None:
        return None
    try:
        values = frozenset(int(item) for item in text.split(",") if item != "")
    except ValueError as error:
        raise argparse.ArgumentTypeError("allowed indices must be comma-separated integers") from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("allowed indices must be non-negative")
    return values


def _options(args: argparse.Namespace) -> SchedulerOptionsV1:
    return SchedulerOptionsV1(
        state_dir=args.state_dir.expanduser().resolve(),
        minimum_free_mib=args.minimum_free_mib,
        maximum_utilization_percent=args.maximum_utilization_percent,
        poll_seconds=args.poll_seconds,
        max_resource_retries=args.max_resource_retries,
        allowed_indices=_allowed_indices(args.allowed_indices),
        gpu_lock_dir=(
            None if args.gpu_lock_dir is None
            else args.gpu_lock_dir.expanduser().resolve()
        ),
    )


def _command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a training command is required after --")
    return command


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--minimum-free-mib", type=int, default=16_000)
    parser.add_argument("--maximum-utilization-percent", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--max-resource-retries", type=int, default=0,
        help="retry only explicit CUDA allocation failures this many times",
    )
    parser.add_argument("--allowed-indices", help="comma-separated physical GPU indices")
    parser.add_argument(
        "--gpu-lock-dir", type=Path,
        help="shared lock directory; defaults to a per-user runtime directory",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("start", "run", "dry-run"):
        child = subparsers.add_parser(operation)
        _add_common(child)
        if operation == "start":
            child.add_argument(
                "--backend", choices=("auto", "systemd", "tmux", "setsid"),
                default="auto",
            )
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.operation == "status":
        status = read_status(args.state_dir.expanduser().resolve())
        print(json.dumps(status or {"state": "not_started"}, indent=2, sort_keys=True))
        return 0
    options = _options(args)
    command = _command(args, parser)
    if args.operation == "run":
        return GPUWaitSchedulerV1(options, command).run()
    if args.operation == "dry-run":
        try:
            statuses = query_gpus()
            selected = choose_gpu(
                statuses,
                minimum_free_mib=options.minimum_free_mib,
                maximum_utilization_percent=options.maximum_utilization_percent,
                allowed_indices=options.allowed_indices,
            )
            result = {
                "state": "would_launch" if selected is not None else "would_wait",
                "selected_gpu": None if selected is None else selected.as_dict(),
                "command": command,
                "watchdog_argv": build_watchdog_argv(Path(sys.argv[0]), options, command),
            }
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            result = {"state": "query_failed", "error": str(error), "command": command}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["state"] != "query_failed" else 1
    launch = start_detached(
        options, command, script_path=Path(sys.argv[0]), backend=args.backend,
    )
    print(json.dumps(launch, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AtomicProcessLock",
    "GPUStatusV1",
    "GPUWaitSchedulerV1",
    "SCHEDULER_SCHEMA_V1",
    "SchedulerOptionsV1",
    "build_child_environment",
    "build_watchdog_argv",
    "classify_retryable_resource_failure",
    "choose_gpu",
    "main",
    "parse_gpu_csv",
    "query_gpus",
    "read_status",
    "start_detached",
]
