from __future__ import annotations

import json
import os

from stablebridge.physical_repair.gpu_wait_scheduler import (
    AtomicProcessLock,
    SchedulerOptionsV1,
    build_child_environment,
    build_watchdog_argv,
    choose_gpu,
    classify_retryable_resource_failure,
    parse_gpu_csv,
)


GPU_CSV = """0, GPU-a, NVIDIA A, 24576, 1000, 23576, 12
1, GPU-b, NVIDIA B, 49152, 2000, 47152, 3
2, GPU-c, NVIDIA C, 49152, 1000, 48152, 80
"""


def test_parse_and_choose_by_threshold_then_free_memory():
    statuses = parse_gpu_csv(GPU_CSV)
    selected = choose_gpu(
        statuses, minimum_free_mib=16_000, maximum_utilization_percent=20,
    )
    assert selected is not None
    assert selected.uuid == "GPU-b"
    allowed = choose_gpu(
        statuses,
        minimum_free_mib=16_000,
        maximum_utilization_percent=20,
        allowed_indices=frozenset({0}),
    )
    assert allowed is not None and allowed.index == 0
    assert choose_gpu(
        statuses, minimum_free_mib=48_500, maximum_utilization_percent=100,
    ) is None


def test_atomic_process_lock_prevents_duplicate_and_recovers_stale(tmp_path):
    path = tmp_path / "job.lock"
    first = AtomicProcessLock(path)
    second = AtomicProcessLock(path)
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()

    path.write_text(json.dumps({"pid": 999_999_999, "token": "stale"}))
    recovered = AtomicProcessLock(path)
    assert recovered.acquire()
    recovered.release()


def test_child_environment_and_watchdog_argv_are_shell_free(tmp_path):
    selected = parse_gpu_csv(GPU_CSV)[1]
    environment = build_child_environment(
        {"PATH": os.environ.get("PATH", "")}, selected,
        resource_retry_count=2,
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert environment["STABLEBRIDGE_PHYSICAL_GPU_UUID"] == "GPU-b"
    assert environment["STABLEBRIDGE_RESOURCE_RETRY_COUNT"] == "2"

    options = SchedulerOptionsV1(
        state_dir=tmp_path / "state with spaces",
        minimum_free_mib=20_000,
        maximum_utilization_percent=5,
        poll_seconds=17,
        allowed_indices=frozenset({1, 3}),
        max_resource_retries=1,
    )
    command = ["python", "train.py", "--literal", "$HOME and spaces"]
    argv = build_watchdog_argv(tmp_path / "launcher.py", options, command)
    separator = argv.index("--")
    assert argv[separator + 1 :] == command
    assert argv[argv.index("--allowed-indices") + 1] == "1,3"
    assert argv[argv.index("--max-resource-retries") + 1] == "1"


def test_resource_retry_classification_is_narrow():
    assert classify_retryable_resource_failure(
        "RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED"
    ) == "CUBLAS_STATUS_ALLOC_FAILED"
    assert classify_retryable_resource_failure(
        "torch.OutOfMemoryError: CUDA out of memory"
    ) == "TORCH_CUDA_OUT_OF_MEMORY"
    assert classify_retryable_resource_failure(
        "RuntimeError: invalid tensor shape"
    ) is None


def test_parser_rejects_inconsistent_or_duplicate_rows():
    try:
        parse_gpu_csv("0, GPU-a, A, 10, 8, 8, 0\n")
    except ValueError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError("inconsistent memory was accepted")
    try:
        parse_gpu_csv("0, GPU-a, A, 10, 1, 9, 0\n0, GPU-b, B, 10, 1, 9, 0\n")
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate GPU identity was accepted")
