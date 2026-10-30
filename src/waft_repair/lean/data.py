"""Raw RGB/GT input. Bind immutable files once, decode without repeated hashes."""
import copy
import struct

import cv2
import numpy as np
from PIL import Image
import torch

from .common import verify


def load(row, device='cpu'):
    images = []
    h, w = row['native_hw']
    for key in ('RGB_first', 'RGB_second'):
        with Image.open(row[key]['path']) as image:
            if image.mode != 'RGB' or image.size != (w, h):
                raise ValueError('RGB geometry/mode differs')
            array = np.array(image, dtype=np.uint8, copy=True)
        images.append(torch.from_numpy(array).permute(2, 0, 1)[None].float().contiguous().to(device))
    if row['dataset'] == 'kitti':
        encoded = cv2.imread(row['raw_gt']['path'], cv2.IMREAD_UNCHANGED)
        if encoded is None or encoded.shape != (h, w, 3) or encoded.dtype != np.uint16:
            raise ValueError('KITTI GT encoding differs')
        gt = ((encoded[..., [2, 1]].astype(np.float32)-32768)/64).transpose(2, 0, 1)[None].copy()
        valid = (encoded[..., 0] > 0)[None]
    else:
        with open(row['gt']['path'], 'rb') as stream:
            header = stream.read(12)
            magic, width, height = struct.unpack('<fii', header)
            if magic != 202021.25 or (height, width) != (h, w):
                raise ValueError('Sintel GT header differs')
            payload = stream.read()
        if len(payload) != 8*h*w:
            raise ValueError('Sintel GT payload differs')
        gt = np.frombuffer(payload, dtype='<f4').reshape(h, w, 2).transpose(2, 0, 1)[None].copy()
        valid = np.isfinite(gt).all(1) & (np.abs(gt) < 1000).all(1)
    return tuple(images), torch.from_numpy(np.ascontiguousarray(gt)).to(device), torch.from_numpy(
        np.ascontiguousarray(valid)).bool().to(device)


def _verify_cases(cases, *, retain=False):
    """Verify and decode each case once, optionally retaining the raw tensors."""
    seen = set()
    retained = {}
    for row in cases:
        for key in ('RGB_first', 'RGB_second', 'raw_gt', 'gt'):
            if key in row and row[key]['path'] not in seen:
                verify(row[key])
                seen.add(row[key]['path'])
        pair, gt, valid = load(row)
        selected = gt.permute(0, 2, 3, 1)[valid]
        if not selected.numel() or not bool(torch.isfinite(selected).all()):
            raise ValueError('Nonempty finite raw supervision required')
        if 'valid_pixels' in row and int(valid.sum()) != row['valid_pixels']:
            raise ValueError('Raw valid population differs')
        if retain:
            retained[row['case_id']] = pair, gt, valid
        del pair, gt, valid, selected
    return dict(cases=len(cases), unique_raw_files=len(seen)), retained


def verify_cases(cases):
    """One startup pass over raw descriptors/GT, before any optimizer update."""
    receipt, _ = _verify_cases(cases)
    return receipt


class RawCaseCache:
    """Bounded CPU raw inputs; encoder features are always computed elsewhere.

    ``get`` lends read-only tensors to the caller. If the complete population
    cannot fit, retain no cases and preserve the original streaming loader.
    """
    def __init__(self, cases, max_bytes=128 << 20):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('A nonnegative integer raw-cache budget is required')
        cases = tuple(cases)
        self._rows = {row['case_id']: copy.deepcopy(row) for row in cases}
        if len(self._rows) != len(cases):
            raise ValueError('Unique raw case IDs are required')
        self.estimated_bytes = 0
        paths = {}
        for row in cases:
            hw = row['native_hw']
            if len(hw) != 2 or any(type(value) is not int or value <= 0 for value in hw):
                raise ValueError('Positive integer raw geometry is required')
            # Two float32 RGB images, float32 XY GT, and one bool validity map.
            self.estimated_bytes += hw[0] * hw[1] * 33
            for key in ('RGB_first', 'RGB_second', 'raw_gt', 'gt'):
                if key in row:
                    paths.setdefault(row[key]['path'], None)
        self.filepaths = tuple(paths)
        self.cached = self.estimated_bytes <= max_bytes
        receipt, self._values = _verify_cases(cases, retain=self.cached)
        self.resident_bytes = self.estimated_bytes if self.cached else 0
        self.receipt = dict(receipt, cached=self.cached,
                            estimated_bytes=self.estimated_bytes,
                            resident_bytes=self.resident_bytes, max_bytes=max_bytes)
        self._released_paths = set()

    def get(self, row):
        if row['case_id'] not in self._rows or row != self._rows[row['case_id']]:
            raise ValueError('Raw input case differs from its startup binding')
        return self._values[row['case_id']] if self.cached else load(row)

    def release_files(self, memory):
        """Once raw tensors are resident, advise each original file only once."""
        if not self.cached:
            return 0
        released = 0
        for path in self.filepaths:
            if path not in self._released_paths:
                memory.advise_readonly(path)
                self._released_paths.add(path)
                released += 1
        return released
