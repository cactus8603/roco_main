from __future__ import annotations

import math

import numpy as np
import pytest

from stablebridge.physical_repair.uncertainty_evaluation import (
    action_uncertainty_associations_v1,
    binary_auroc_v1,
    evaluate_log_scale_uncertainty_v1,
    grouped_bootstrap_mean_v1,
    spearman_v1,
)


def test_perfect_ranking_has_zero_ause_and_unit_severe_auroc():
    error = np.array([0.1, 0.2, 1.0, 3.0], np.float64)
    result = evaluate_log_scale_uncertainty_v1(
        np.log(error), error, np.ones(4, bool), severe_threshold=0.8,
    )
    assert result.valid_count == 4
    assert result.spearman_error == pytest.approx(1.0)
    assert result.ause == pytest.approx(0.0, abs=1e-12)
    assert result.severe_auroc == pytest.approx(1.0)


def test_reverse_ranking_is_worse_than_oracle():
    error = np.array([0.1, 0.2, 1.0, 3.0], np.float64)
    result = evaluate_log_scale_uncertainty_v1(
        -np.log(error), error, np.ones(4, bool), severe_threshold=0.8,
    )
    assert result.spearman_error == pytest.approx(-1.0)
    assert result.ause > 0
    assert result.severe_auroc == pytest.approx(0.0)


def test_exponential_coverage_math_and_masking():
    # q=0.5 upper bound is b*log(2).  One of two selected errors is covered.
    scale = np.ones(3, np.float64)
    error = np.array([0.5, 0.8, 100.0])
    result = evaluate_log_scale_uncertainty_v1(
        np.log(scale), error, np.array([True, True, False]),
        severe_threshold=1.0, calibration_coverages=(0.5,),
    )
    assert result.empirical_coverages == pytest.approx((0.5,))
    assert result.coverage_calibration_mae == pytest.approx(0.0)
    assert math.isnan(result.severe_auroc)


def test_rank_metrics_handle_ties_and_missing_classes():
    assert spearman_v1([1, 1, 2], [1, 1, 3]) == pytest.approx(1.0)
    assert binary_auroc_v1([0.1, 0.9], [False, True]) == pytest.approx(1.0)
    assert math.isnan(binary_auroc_v1([0.1, 0.9], [True, True]))


def test_group_bootstrap_resamples_groups_not_rows():
    result = grouped_bootstrap_mean_v1(
        [0.0, 0.0, 10.0, 10.0], ["a", "a", "b", "b"],
        resamples=200, seed=7,
    )
    assert result.point == pytest.approx(5.0)
    assert result.group_count == 2
    assert result.lower <= result.point <= result.upper


def test_action_association_is_conditioned_on_exact_control():
    rows = action_uncertainty_associations_v1(
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        ["a", "a", "b", "b"],
        severe_harm_threshold=0.5,
    )
    assert [row.exact_control_id for row in rows] == ["a", "b"]
    assert rows[0].uncertainty_gain_spearman == pytest.approx(1.0)
    assert rows[1].uncertainty_gain_spearman == pytest.approx(-1.0)

