from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "experiments/E32_flow_deployment_aggregation/runtime.py"
SPEC = importlib.util.spec_from_file_location("test_e32_runtime_module", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_conservative_plurality_falls_back_on_even_ties() -> None:
    votes = np.asarray([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=np.int8)
    assert MODULE.conservative_plurality(votes, axis=1).tolist() == [0, 1]


def test_fold_then_seed_hierarchy_is_not_flat_vote() -> None:
    # Seed 1 has six RR votes; the other two seeds each conservatively choose
    # CC.  Hierarchical deployment therefore returns CC despite a flat 10/18
    # RR majority.
    folds = np.asarray(
        [[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]],
        dtype=np.int8,
    )
    seeds = MODULE.conservative_plurality(folds, axis=1)
    assert seeds.tolist() == [1, 0, 0]
    assert int(MODULE.conservative_plurality(seeds, axis=0)) == 0
    assert int(MODULE.conservative_plurality(folds.reshape(-1), axis=0)) == 1


def test_plurality_rejects_nonbinary_or_empty_inputs() -> None:
    with pytest.raises(ValueError):
        MODULE.conservative_plurality(np.asarray([], dtype=np.int8))
    with pytest.raises(ValueError):
        MODULE.conservative_plurality(np.asarray([0, 2], dtype=np.int8))


def test_probability_shift_is_identity_at_zero_and_monotonic() -> None:
    value = np.asarray([0.1, 0.5, 0.9])
    np.testing.assert_allclose(MODULE.shift_probability(value, 0.0), value, rtol=1e-6)
    assert np.all(MODULE.shift_probability(value, 1.0) > value)
    assert np.all(MODULE.shift_probability(value, -1.0) < value)
