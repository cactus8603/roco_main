"""Committed-cursor recovery and native CLI boundary checks; no neural work."""
import json
from pathlib import Path

import pytest

from waft_repair.cli import parser
from waft_repair.motion.io import atomic_torch
from waft_repair.native_pipeline import _package_sources,_reconcile_history


def test_history_recovery_preserves_only_uncommitted_tail(tmp_path):
    atomic_torch(tmp_path/'checkpoint.pt',{'completed_updates':2})
    prefix=''.join(json.dumps({'completed_updates':i})+'\n' for i in (1,2))
    tail=json.dumps({'completed_updates':3})+'\n'+'{"partial":'
    (tmp_path/'history.jsonl').write_text(prefix+tail)
    _reconcile_history(tmp_path,2)
    assert (tmp_path/'history.jsonl').read_text()==prefix
    backups=list(tmp_path.glob('history-uncommitted-*.jsonl'))
    assert len(backups)==1 and backups[0].read_text()==tail


def test_older_checkpoint_cannot_rewind_existing_run(tmp_path):
    atomic_torch(tmp_path/'checkpoint.pt',{'completed_updates':4})
    with pytest.raises(ValueError,match='latest committed'):_reconcile_history(tmp_path,2)


def test_missing_or_short_committed_history_rejected(tmp_path):
    atomic_torch(tmp_path/'checkpoint.pt',{'completed_updates':2})
    with pytest.raises(ValueError,match='missing'):_reconcile_history(tmp_path,2)
    (tmp_path/'history.jsonl').write_text('{"completed_updates":1}\n')
    with pytest.raises(ValueError,match='shorter'):_reconcile_history(tmp_path,2)


def test_native_cli_defaults_and_no_label_input_to_inference():
    fit=parser().parse_args(['native-train','--manifest','m','--output','o',
        '--waft-source','s','--waft-checkpoint','w'])
    assert fit.epochs==32 and fit.seed==20261028 and fit.warmup_updates==160
    assert fit.activation_offload=='cpu' and fit.checkpoint_every==32
    with pytest.raises(SystemExit):parser().parse_args(['native-infer','--repair-checkpoint','r',
        '--image1','a','--image2','b','--output','o','--waft-source','s','--waft-checkpoint','w','--gt','g'])


def test_source_binding_covers_strict_loader_labels_and_native_consumer():
    pins=_package_sources()
    assert {'waft_repair.backend','waft_repair.motion.consumer','waft_repair.motion.training',
            'waft_repair.ground_truth','waft_repair.native_cli','waft_repair.motion.checkpoint'}<=pins.keys()
    assert all(len(v)==64 for v in pins.values())
