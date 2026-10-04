from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from stablebridge.physical_repair.evidence_pooling import CaseEvidencePooler


ROOT = Path(__file__).resolve().parents[2]
E233_COMMON = (
    ROOT
    / "experiments/E233_p05_native_cycle_geometry_preexecution_v2/common.py"
)


def _legacy_pool_region():
    spec = importlib.util.spec_from_file_location(
        "e233_common_pooling_reference", E233_COMMON,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pool_region


def _case(height: int = 79, width: int = 91):
    rng = np.random.default_rng(20261003)
    native = rng.normal(0.0, 1.0, (height, width, 2)).astype(np.float32)
    full = (native + rng.normal(0.0, 0.35, native.shape)).astype(np.float32)
    verifier = rng.normal(0.0, 1.0, native.shape).astype(np.float32)
    cycle_valid = rng.random((height, width)) > 0.17
    cycle_valid[2:8, 3:11] = False
    forward_nonfinite = rng.random((height, width)) < 0.02
    coordinate_invalid = rng.random((height, width)) < 0.04
    interpolation_incomplete = (~cycle_valid) & ~(
        forward_nonfinite | coordinate_invalid
    )
    jacobian_det = rng.normal(1.0, 0.4, (height, width))
    jacobian_finite = rng.random((height, width)) > 0.03
    jacobian_det[~jacobian_finite] = np.nan
    fold = jacobian_finite & (jacobian_det <= 0.0)
    near_singular = jacobian_finite & (np.abs(jacobian_det) < 0.05)
    ambiguity = (~cycle_valid) | fold | near_singular
    cycle = {
        "verifier": verifier,
        "cycle_valid": np.ascontiguousarray(cycle_valid, dtype=bool),
        "forward_nonfinite": np.ascontiguousarray(
            forward_nonfinite, dtype=bool,
        ),
        "coordinate_invalid": np.ascontiguousarray(
            coordinate_invalid, dtype=bool,
        ),
        "interpolation_incomplete": np.ascontiguousarray(
            interpolation_incomplete, dtype=bool,
        ),
    }
    g0 = {
        "jacobian_det": np.ascontiguousarray(jacobian_det, dtype=np.float64),
        "jacobian_finite": np.ascontiguousarray(jacobian_finite, dtype=bool),
        "fold": np.ascontiguousarray(fold, dtype=bool),
        "near_singular": np.ascontiguousarray(near_singular, dtype=bool),
        "ambiguity": np.ascontiguousarray(ambiguity, dtype=bool),
    }
    return native, full, cycle, g0


def test_multi_region_multi_fold_is_field_exact_to_sealed_e233_reference():
    legacy = _legacy_pool_region()
    native, full, cycle, g0 = _case()
    pooler = CaseEvidencePooler(native, full, cycle, g0)
    regions = (
        [0, 0, 64, 64],
        [13, 7, 78, 69],
        [64, 32, 91, 79],
        [31, 31, 65, 66],
        [2, 2, 12, 9],
    )
    fit_scales = {
        "0": -0.2,
        "1": 0.0,
        "2": 0.15,
        "3": float("nan"),
        "4": float("inf"),
    }
    for bbox in regions:
        batch = pooler.pool_region_fit_scales(bbox, fit_scales)
        assert batch.unscaled == legacy(
            native, full, cycle, g0, bbox, fit_scale=None,
        )
        for fold, scale in fit_scales.items():
            assert batch.by_fit_scale[fold] == legacy(
                native, full, cycle, g0, bbox, fit_scale=scale,
            )


def test_typed_missing_region_is_exact_to_e233_reference():
    legacy = _legacy_pool_region()
    native, full, cycle, g0 = _case()
    cycle["cycle_valid"][20:24, 30:35] = False
    pooler = CaseEvidencePooler(native, full, cycle, g0)
    bbox = [30, 20, 35, 24]
    batch = pooler.pool_region_fit_scales(bbox, {"0": 0.1, "1": None})
    assert batch.unscaled == legacy(
        native, full, cycle, g0, bbox, fit_scale=None,
    )
    assert batch.by_fit_scale["0"] == legacy(
        native, full, cycle, g0, bbox, fit_scale=0.1,
    )
    assert batch.by_fit_scale["1"] == legacy(
        native, full, cycle, g0, bbox, fit_scale=None,
    )


def test_case_cache_is_owned_float64_and_stable_after_input_mutation():
    native, full, cycle, g0 = _case(height=16, width=20)
    pooler = CaseEvidencePooler(native, full, cycle, g0)
    expected_bytes = 3 * 16 * 20 * 2 * np.dtype(np.float64).itemsize
    assert pooler.float64_flow_cache_nbytes == expected_bytes
    before = pooler.pool_region([0, 0, 20, 16], fit_scale=0.1)
    native[...] = 999
    full[...] = -999
    cycle["verifier"][...] = 333
    cycle["cycle_valid"][...] = False
    g0["jacobian_det"][...] = -333
    after = pooler.pool_region([0, 0, 20, 16], fit_scale=0.1)
    assert after == before


@pytest.mark.parametrize(
    "bbox",
    ([0, 0, 0, 3], [-1, 0, 4, 3], [0, 0, 100, 3], [0.5, 0, 4, 3]),
)
def test_invalid_bbox_fails_closed(bbox):
    native, full, cycle, g0 = _case(height=8, width=9)
    pooler = CaseEvidencePooler(native, full, cycle, g0)
    with pytest.raises(ValueError, match="bbox"):
        pooler.pool_region(bbox)
