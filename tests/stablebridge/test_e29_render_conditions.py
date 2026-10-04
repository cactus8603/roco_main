from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "experiments/E29_external_joint_panel/render_conditions.py"
SPEC = importlib.util.spec_from_file_location("e29_render_conditions_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_endpoint_exposure_and_determinism() -> None:
    yy, xx = np.mgrid[:32, :40]
    first = np.stack((xx * 5, yy * 7, (xx + yy) * 3), axis=-1).clip(0, 255).astype(np.uint8)
    second = np.flip(first, axis=1).copy()
    for condition in MODULE.CONDITIONS:
        variants = MODULE.render_condition_variants(
            first, second, sample_key="unit:000000", condition=condition
        )
        repeated = MODULE.render_condition_variants(
            first, second, sample_key="unit:000000", condition=condition
        )
        assert np.array_equal(variants["first"].second, second)
        assert np.array_equal(variants["second"].first, first)
        assert np.array_equal(variants["both"].first, variants["first"].first)
        assert np.array_equal(variants["both"].second, variants["second"].second)
        assert not np.array_equal(variants["first"].first, first)
        assert not np.array_equal(variants["second"].second, second)
        for exposure in MODULE.EXPOSURES:
            assert np.array_equal(variants[exposure].first, repeated[exposure].first)
            assert np.array_equal(variants[exposure].second, repeated[exposure].second)
            assert variants[exposure].metadata["changes_geometry"] is False
