"""CPU fixtures for bounded raw reuse and unchanged memory-release ordering."""
import copy
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image
import torch

from waft_repair.lean import common, data, memory


class RawCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        h, w = self.hw = 4, 5
        pixels = np.arange(h*w*3, dtype=np.uint8).reshape(h, w, 3)
        for name, image in [('one.png', pixels), ('two.png', 255-pixels)]:
            Image.fromarray(image).save(self.root/name)
        flow = np.arange(h*w*2, dtype=np.float32).reshape(h, w, 2)/8-1
        (self.root/'flow.flo').write_bytes(struct.pack('<fii', 202021.25, w, h)+flow.tobytes())
        encoded = np.zeros((h, w, 3), dtype=np.uint16)
        encoded[..., 2] = ((flow[..., 0]*64)+32768).astype(np.uint16)
        encoded[..., 1] = ((flow[..., 1]*64)+32768).astype(np.uint16)
        encoded[..., 0] = 1
        self.assertTrue(cv2.imwrite(str(self.root/'flow.png'), encoded))

    def row(self, dataset, case_id=None):
        row = dict(dataset=dataset, case_id=case_id or dataset,
                   native_hw=list(self.hw), valid_pixels=self.hw[0]*self.hw[1],
                   RGB_first=common.descriptor(self.root/'one.png'),
                   RGB_second=common.descriptor(self.root/'two.png'))
        row['raw_gt' if dataset == 'kitti' else 'gt'] = common.descriptor(
            self.root/('flow.png' if dataset == 'kitti' else 'flow.flo'))
        return row

    def assert_same_bytes(self, one, two):
        for first, second in zip((*one[0], *one[1:]), (*two[0], *two[1:])):
            self.assertEqual(first.dtype, second.dtype)
            self.assertEqual(first.shape, second.shape)
            self.assertEqual(first.stride(), second.stride())
            self.assertEqual(first.numpy().tobytes(), second.numpy().tobytes())

    def test_complete_cache_preserves_both_formats_and_never_decodes_on_get(self):
        cases = [self.row('kitti'), self.row('sintel')]
        expected = [data.load(row) for row in cases]
        with patch.object(data, 'load', wraps=data.load) as decode:
            cache = data.RawCaseCache(cases)
            self.assertTrue(cache.cached)
            self.assertEqual(decode.call_count, 2)
            for row, before in zip(cases, expected):
                first = cache.get(row)
                self.assert_same_bytes(before, first)
                self.assertIs(first, cache.get(row))
            self.assertEqual(decode.call_count, 2)
        self.assertEqual(cache.resident_bytes, 2*4*5*33)
        self.assertEqual(cache.receipt['unique_raw_files'], 4)
        self.assertFalse(torch.cuda.is_initialized())

    def test_oversized_population_streams_every_case_without_partial_cache(self):
        cases = [self.row('kitti'), self.row('sintel')]
        with patch.object(data, 'load', wraps=data.load) as decode:
            cache = data.RawCaseCache(cases, max_bytes=4*5*33)
            self.assertFalse(cache.cached)
            self.assertEqual(cache.resident_bytes, 0)
            self.assertFalse(cache._values)
            self.assertEqual(decode.call_count, 2)  # Startup validation remains.
            for row in cases:
                self.assert_same_bytes(data.load(row), cache.get(row))
            self.assertEqual(decode.call_count, 6)
        advice = SimpleNamespace(advise_readonly=Mock())
        self.assertEqual(cache.release_files(advice), 0)
        advice.advise_readonly.assert_not_called()

    def test_cache_advises_unique_original_files_once_and_binds_case_metadata(self):
        cases = [self.row('sintel', 'clean'), self.row('sintel', 'final')]
        with patch.object(data, 'verify', wraps=data.verify) as verify:
            cache = data.RawCaseCache(cases)
        self.assertEqual(verify.call_count, 3)
        advice = SimpleNamespace(advise_readonly=Mock())
        self.assertEqual(cache.release_files(advice), 3)
        self.assertEqual(cache.release_files(advice), 0)
        self.assertEqual(advice.advise_readonly.call_count, 3)
        altered = copy.deepcopy(cases[0])
        altered['RGB_first']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'startup binding'):
            cache.get(altered)

    def test_both_modes_reject_bad_startup_supervision_or_file_sha(self):
        for limit in (0, 128 << 20):
            bad = self.row('kitti')
            bad['valid_pixels'] -= 1
            with self.assertRaisesRegex(ValueError, 'population'):
                data.RawCaseCache([bad], max_bytes=limit)
            bad = self.row('sintel')
            bad['gt']['sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                data.RawCaseCache([bad], max_bytes=limit)


class MemoryReleaseTests(unittest.TestCase):
    def test_cuda_release_keeps_required_order_without_telemetry_or_libc_reload(self):
        events = []
        fake = SimpleNamespace(cuda=SimpleNamespace(
            is_initialized=lambda: True,
            synchronize=lambda: events.append('synchronize')),
            _C=SimpleNamespace(_host_emptyCache=lambda: events.append('pinned')))
        with patch.object(memory.gc, 'collect', side_effect=lambda: events.append('gc')), \
             patch.object(memory, '_MALLOC_TRIM', side_effect=lambda value: events.append('trim') or 1) as trim, \
             patch.object(memory.Path, 'read_text', side_effect=AssertionError('Unused telemetry')), \
             patch.object(memory.ctypes, 'CDLL', side_effect=AssertionError('Repeated libc binding')):
            result = memory.release_unused(fake)
        self.assertEqual(events, ['gc', 'synchronize', 'pinned', 'trim'])
        trim.assert_called_once_with(0)
        self.assertEqual(result, dict(cuda_synchronized=True,
            unused_pinned_host_cache_released=True, malloc_trim_result=1))

    def test_cpu_release_never_calls_cuda_cleanup_and_missing_pinned_hook_fails(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False,
            synchronize=Mock(side_effect=AssertionError('CPU-only release'))), _C=SimpleNamespace())
        with patch.object(memory.gc, 'collect'), patch.object(memory, '_MALLOC_TRIM', return_value=0) as trim:
            result = memory.release_unused(fake)
        trim.assert_called_once_with(0)
        self.assertFalse(result['cuda_synchronized'])
        fake.cuda.is_initialized = lambda: True
        fake.cuda.synchronize = Mock()
        with patch.object(memory.gc, 'collect'), patch.object(memory, '_MALLOC_TRIM') as trim:
            with self.assertRaisesRegex(RuntimeError, 'pinned-host cache release'):
                memory.release_unused(fake)
        fake.cuda.synchronize.assert_called_once_with()
        trim.assert_not_called()

    def test_file_advice_keeps_no_follow_regular_owner_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'raw.bin'
            path.write_bytes(b'raw')
            with patch.object(memory.os, 'posix_fadvise') as advise:
                result = memory.advise_readonly(path)
            advise.assert_called_once()
            self.assertEqual(result['bytes'], 3)
            self.assertTrue(result['read_only'])
            link = Path(temporary)/'link.bin'
            link.symlink_to(path)
            with self.assertRaises(OSError):
                memory.advise_readonly(link)


if __name__ == '__main__':
    unittest.main()
