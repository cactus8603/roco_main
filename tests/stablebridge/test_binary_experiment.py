import copy
import json
from pathlib import Path

import numpy as np
import pytest

from stablebridge.binary_data import project_two_answers
from stablebridge.binary_experiment import baseline_scores, validate_config, verify_calibration_lock, scientific_decision
from stablebridge.binary_reporting import evaluate_case, aggregate
from stablebridge.util import ROOT, save_json, sha256


def test_projection_uses_only_registered_candidate_and_fallback():
    initial = np.ones((2, 2, 2), dtype=np.float32)
    bank = np.stack([initial, initial*3, initial*999])
    valid = np.ones((3, 2, 2), bool)
    valid[1, 0, 0] = False
    arrays = {'initial': initial, 'e0_bank': bank, 'e0_valid': valid,
              'image0': np.zeros((2,2,3), np.uint8), 'image1': np.zeros((2,2,3), np.uint8),
              'source_features': initial, 'target_features': initial, 'raw_uncertainty': initial[0],
              'fixed_q': np.ones((2,2), bool), 'processed_support': initial*999}
    result = project_two_answers(arrays, 1)
    assert 'fixed_q' not in result and 'processed_support' not in result and 'e0_bank' not in result
    assert np.all(result['candidate'][:, 0, 0] == 1)
    assert np.all(result['candidate'][:, 1, 1] == 3)


def test_rgb_baseline_score_prefers_lower_candidate_cost():
    scores = baseline_scores(np.array([[[-.2]]]), {'uncertainty': np.ones((1,1)), 'delta_norm': np.ones((1,1))},
                             {'feature_names': ['candidate_minus_initial_patch_rgb_l1']})
    assert scores['rgb_match'].item() == .2


def test_exposure_overlap_is_rejected():
    config = json.loads((ROOT/'configs/stablebridge/e01_s05_binary_revision_v1.json').read_text())
    validate_config(config)
    config['splits']['confirmation']['0015'] = [60]
    with pytest.raises(ValueError, match='already exposed'):
        validate_config(config)


def test_calibration_locks_both_policies_and_checkpoint(tmp_path):
    (tmp_path/'checkpoints').mkdir()
    ckpt = tmp_path/'checkpoints/model.pth'
    ckpt.write_bytes(b'final checkpoint')
    save_json(tmp_path/'policies.json', {'threshold': 1})
    save_json(tmp_path/'calibration_complete.json', {'policy_sha256': sha256(tmp_path/'policies.json'),
              'checkpoints': {'model': sha256(ckpt)}})
    verify_calibration_lock(tmp_path)
    ckpt.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        verify_calibration_lock(tmp_path)


def test_no_good_denominator_returns_inconclusive_not_crash():
    config = json.loads((ROOT/'configs/stablebridge/e01_s05_binary_revision_v1.json').read_text())
    rows = [evaluate_case(np.array([2.]), np.array([1.]), np.array([True]), np.array([True]),
                          np.array([False]), {'scene':str(s),'task':'stereo','split':'confirmation'}) for s in range(4)]
    summary = aggregate(rows)
    result = {'summary': summary, 'screen': {'passed': False}}
    names = ['identity', 'allcandidate', 'uncertainty', 'small_change', 'rgb_match']
    names += [f'{info}_{objective}_s{s}' for info in ('geometry','full') for objective in ('gain','errors') for s in config['learning']['seeds']]
    decision = scientific_decision({n:result for n in names}, config)
    assert decision['best_harm_feasible_simple_baseline'] is None
    assert not decision['any_registered_learned_arm_passed']
