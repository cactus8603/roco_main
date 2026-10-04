from __future__ import annotations

import unittest

from stablebridge.physical_repair.gpu_admission import (
    GPUAdmissionError,
    GPUAdmissionPolicy,
    GPUSnapshot,
    admit_gpu,
    choose_gpu,
    parse_nvidia_smi_csv,
    validate_write_roots,
    verify_second_snapshot,
)


def gpu(
    index: int,
    uuid: str,
    *,
    free: int,
    processes: int | None,
    utilization: int = 0,
    temperature: int = 40,
) -> GPUSnapshot:
    return GPUSnapshot(
        index=index,
        uuid=uuid,
        name="Test GPU",
        total_memory_mib=24_576,
        used_memory_mib=24_576 - free,
        free_memory_mib=free,
        utilization_percent=utilization,
        temperature_celsius=temperature,
        compute_process_count=processes,
    )


class NvidiaSmiParsingTests(unittest.TestCase):
    def test_parses_header_units_and_external_process_counts(self) -> None:
        text = """index, uuid, name, memory.total [MiB], memory.used [MiB], memory.free [MiB], utilization.gpu [%], temperature.gpu
0, GPU-a, RTX 3090, 24576 MiB, 13 MiB, 24563 MiB, 0 %, 39 C
1, GPU-b, RTX 3090, 24576 MiB, 8000 MiB, 16576 MiB, 99 %, 75 C
"""
        snapshots = parse_nvidia_smi_csv(
            text, process_counts_by_uuid={"GPU-a": 0, "GPU-b": 3},
        )
        self.assertEqual([item.uuid for item in snapshots], ["GPU-a", "GPU-b"])
        self.assertTrue(snapshots[0].is_empty)
        self.assertFalse(snapshots[1].is_empty)
        self.assertEqual(snapshots[1].compute_process_count, 3)

    def test_no_process_evidence_is_unknown_not_empty(self) -> None:
        row = "0, GPU-a, RTX 3090, 24576, 13, 24563, 0, 39\n"
        snapshot = parse_nvidia_smi_csv(row)[0]
        self.assertIsNone(snapshot.compute_process_count)
        self.assertFalse(snapshot.is_empty)

    def test_inline_process_count_and_malformed_rows(self) -> None:
        columns = (
            "index", "uuid", "name", "memory.total", "memory.used",
            "memory.free", "utilization.gpu", "temperature.gpu",
            "compute_process_count",
        )
        snapshot = parse_nvidia_smi_csv(
            "5,GPU-five,RTX 3090,24576,13,24563,0,36,0\n",
            columns=columns,
        )[0]
        self.assertTrue(snapshot.is_empty)
        with self.assertRaisesRegex(ValueError, "fields"):
            parse_nvidia_smi_csv("0,GPU-a,too,few\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_nvidia_smi_csv(
                "0,GPU-a,RTX,24576,13,24563,0,39\n"
                "1,GPU-a,RTX,24576,13,24563,0,39\n"
            )


class GPUSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = GPUAdmissionPolicy(
            allowed_uuids=frozenset({"GPU-0", "GPU-1", "GPU-2"}),
            peak_memory_mib=8_000,
            safety_margin_mib=2_000,
        )

    def test_allowlist_and_empty_gpu_have_priority_over_busier_free_gpu(self) -> None:
        decision = choose_gpu([
            gpu(0, "GPU-0", free=23_000, processes=2),
            gpu(1, "GPU-1", free=12_000, processes=0),
            gpu(2, "GPU-not-allowed", free=24_000, processes=0),
        ], self.policy)
        self.assertEqual(decision.gpu.uuid, "GPU-1")
        self.assertEqual(decision.mode, "empty")

    def test_shared_gpu_requires_peak_plus_margin(self) -> None:
        decision = choose_gpu([
            gpu(0, "GPU-0", free=9_999, processes=1),
            gpu(1, "GPU-1", free=10_000, processes=4, utilization=90),
        ], self.policy)
        self.assertEqual(decision.gpu.uuid, "GPU-1")
        self.assertEqual(decision.mode, "shared")
        self.assertEqual(decision.required_free_memory_mib, 10_000)
        with self.assertRaisesRegex(GPUAdmissionError, "required free memory"):
            choose_gpu([gpu(0, "GPU-0", free=9_999, processes=0)], self.policy)

    def test_deterministic_tie_break(self) -> None:
        cards = [
            gpu(2, "GPU-2", free=20_000, processes=0, temperature=40),
            gpu(1, "GPU-1", free=20_000, processes=0, temperature=40),
            gpu(0, "GPU-0", free=18_000, processes=0, temperature=35),
        ]
        self.assertEqual(choose_gpu(cards, self.policy).gpu.uuid, "GPU-1")
        self.assertEqual(choose_gpu(list(reversed(cards)), self.policy).gpu.uuid, "GPU-1")

    def test_unknown_process_count_is_shared_not_empty(self) -> None:
        decision = choose_gpu([
            gpu(0, "GPU-0", free=22_000, processes=None),
            gpu(1, "GPU-1", free=11_000, processes=0),
        ], self.policy)
        self.assertEqual(decision.gpu.uuid, "GPU-1")
        self.assertEqual(decision.mode, "empty")

    def test_second_snapshot_must_reproduce_uuid_index_and_mode(self) -> None:
        first = [
            gpu(0, "GPU-0", free=19_000, processes=0),
            gpu(1, "GPU-1", free=18_000, processes=1),
        ]
        second = [
            gpu(0, "GPU-0", free=18_000, processes=0),
            gpu(1, "GPU-1", free=18_000, processes=1),
        ]
        verified = admit_gpu(first, second, self.policy)
        self.assertEqual(verified.gpu.uuid, "GPU-0")
        initial = choose_gpu(first, self.policy)
        with self.assertRaisesRegex(GPUAdmissionError, "different GPU"):
            verify_second_snapshot(initial, [
                gpu(0, "GPU-0", free=9_000, processes=0),
                gpu(1, "GPU-1", free=18_000, processes=1),
            ], self.policy)
        with self.assertRaisesRegex(GPUAdmissionError, "mode changed"):
            verify_second_snapshot(initial, [
                gpu(0, "GPU-0", free=18_000, processes=1),
                gpu(1, "GPU-1", free=17_000, processes=1),
            ], self.policy)


class WriteRootValidationTests(unittest.TestCase):
    def test_accepts_allowed_absolute_roots_and_sorts_caches(self) -> None:
        roots = validate_write_roots(
            runtime_root="/ssd6/cactus8603/run/../run/runtime",
            output_root="/ssd6/cactus8603/run/output",
            cache_roots={
                "torch": "/ssd6/cactus8603/run/cache/torch",
                "hf": "/ssd1/cactus8603/cache/hf",
            },
        )
        self.assertEqual(roots.runtime_root, "/ssd6/cactus8603/run/runtime")
        self.assertEqual([name for name, _ in roots.cache_roots], ["hf", "torch"])

    def test_rejects_tmp_ssd8_traversal_and_relative_paths(self) -> None:
        bad_paths = (
            "/tmp", "/tmp/job", "/ssd8", "/ssd8/cactus8603/job",
            "//tmp/job", "//ssd8/job",
            "/ssd6/cactus8603/../../tmp/job", "relative/job",
        )
        for path in bad_paths:
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_write_roots(
                    runtime_root=path,
                    output_root="/ssd6/cactus8603/output",
                )

    def test_rejects_forbidden_cache_even_when_other_roots_are_safe(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden root /ssd8"):
            validate_write_roots(
                runtime_root="/ssd6/cactus8603/runtime",
                output_root="/ssd6/cactus8603/output",
                cache_roots={"torch": "/ssd8/cactus8603/cache"},
            )


if __name__ == "__main__":
    unittest.main()
