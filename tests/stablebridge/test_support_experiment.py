"""Leakage, scoring and training-output contracts for the S04 worker."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from stablebridge.support_experiment import (validate_config, prepare_model_inputs,
                                            target_tensors, native_min4_loss, cases_for)


def tiny_arrays():
    h, w = 32, 32
    return {'image0': np.zeros((h, w, 3), np.uint8), 'image1': np.zeros((h, w, 3), np.uint8),
            'source_features': np.zeros((8, 2, 2), np.float32), 'target_features': np.zeros((8, 2, 2), np.float32),
            'initial': np.zeros((2, h, w), np.float32), 'support_xy': np.array([[1, 1], [20, 1], [1, 20], [20, 20]], np.float32),
            'support_raw': np.arange(8, dtype=np.float32).reshape(4, 2),
            'support_processed': np.arange(8, dtype=np.float32).reshape(4, 2)+1,
            'support_valid': np.ones(4, bool)}


def test_inputs_closed_and_wrong_values_preserve_marginal():
    arrays = tiny_arrays()
    query = np.array([[8, 8]], np.float32)
    correct = prepare_model_inputs(arrays, query, 'processed', 'cpu')
    wrong = prepare_model_inputs(arrays, query, 'wrong', 'cpu')
    raw = prepare_model_inputs(arrays, query, 'raw', 'cpu')
    assert torch.equal(correct['supports_xy'], wrong['supports_xy'])
    assert torch.equal(correct['support_source'], wrong['support_source'])
    assert torch.equal(correct['support_source'], raw['support_source'])
    assert torch.equal(correct['supports_uv'].sort(dim=1).values, wrong['supports_uv'].sort(dim=1).values)
    arrays.update({'gt4': np.full((4, 2, 32, 32), 9876), 'fixed_q': np.ones((32, 32), bool)})
    other = prepare_model_inputs(arrays, query, 'processed', 'cpu')
    for key in correct:
        assert torch.equal(correct[key], other[key]) if isinstance(correct[key], torch.Tensor) else correct[key] == other[key]
    assert 'gt4' not in other and 'fixed_q' not in other


def test_privileged_support_requires_explicit_separate_values():
    with pytest.raises(ValueError):
        prepare_model_inputs(tiny_arrays(), np.array([[8, 8]], np.float32), 'gt_diagnostic', 'cpu')
    result = prepare_model_inputs(tiny_arrays(), np.array([[8, 8]], np.float32), 'no_support', 'cpu')
    assert not result['support_valid'].any()
    assert not result['supports_uv'].any()


def test_target_indexing_and_whole_vector_min4_loss():
    gt = np.full((4, 2, 3, 3), np.inf, np.float32)
    gt[0, :, 1, 2] = [1, 2]
    gt[1, :, 1, 2] = [3, 4]
    labels = {'gt4': gt, 'gt_valid': np.ones((3, 3), bool)}
    targets, branches, valid = target_tensors(labels, np.array([[2, 1]], np.float32), 'cpu')
    assert targets.shape == (1, 1, 4, 2)
    assert targets[0, 0, 0].tolist() == [1, 2]
    assert branches[0, 0].tolist() == [True, True, False, False]
    output = torch.tensor([[[1., 4.]]], requires_grad=True)
    result = {'output_uv': output, 'refined_candidate_uv': torch.tensor([[[[1., 2.], [3., 4.]]]]),
              'candidate_valid': torch.ones(1, 1, 2, dtype=torch.bool), 'logits': torch.zeros(1, 1, 2, requires_grad=True)}
    loss, _ = native_min4_loss(result, targets, branches, valid, .1)
    assert float(loss) == pytest.approx(2+.1*np.log(2), abs=1e-5)
    loss.backward()
    assert torch.isfinite(output.grad).all()


def test_registered_split_budget_and_leakage_rejection():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root/'configs/stablebridge/e01_s04_support_rematch_v1.json').read_text())
    validate_config(config)
    assert len(cases_for(config, 'train')) == 144
    assert len(cases_for(config, 'confirmation')) == 24
    bad = copy.deepcopy(config)
    bad['splits']['confirmation']['0001'] = [60]
    with pytest.raises(ValueError):
        validate_config(bad)
