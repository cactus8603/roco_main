"""Real read-only cache advice and allocator lifetime ordering; no CUDA work."""
import hashlib
import os
from types import SimpleNamespace

import pytest
import torch

from waft_repair import memory_hygiene as M


def test_cache_advice_preserves_content_inode_and_mtime(tmp_path):
    path=tmp_path/'input';path.write_bytes(b'original RGB bytes'*4000)
    before=path.stat();digest=hashlib.sha256(path.read_bytes()).digest()
    result=M.advise_readonly(path);after=path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    assert hashlib.sha256(path.read_bytes()).digest()==digest
    assert result['read_only'] and result['bytes']==before.st_size


def test_symlink_and_fifo_rejected_without_following_or_waiting(tmp_path):
    original=tmp_path/'original';original.write_bytes(b'keep')
    link=tmp_path/'link';link.symlink_to(original)
    with pytest.raises(OSError):M.advise_readonly(link)
    pipe=tmp_path/'fifo';os.mkfifo(pipe)
    with pytest.raises(ValueError,match='regular'):M.advise_readonly(pipe)


def test_foreign_owned_files_rejected(tmp_path,monkeypatch):
    p=tmp_path/'p';p.write_bytes(b'keep')
    original=os.getuid();monkeypatch.setattr(M.os,'getuid',lambda:original+1)
    with pytest.raises(ValueError,match='owner'):M.advise_readonly(p)


def test_sync_precedes_unused_pinned_release_and_trim(monkeypatch):
    events=[]
    trim=SimpleNamespace()
    class Function:
        def __call__(self,*args):events.append('trim');return 1
    trim.malloc_trim=Function()
    monkeypatch.setattr(M.ctypes,'CDLL',lambda _:trim)
    fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True,
        synchronize=lambda:events.append('sync')),_C=SimpleNamespace(_host_emptyCache=lambda:events.append('host-empty')))
    result=M.release_unused(fake)
    assert events==['sync','host-empty','trim']
    assert result['cuda_synchronized'] and result['unused_pinned_host_cache_released']


def test_missing_pinned_release_api_fails_explicitly():
    fake=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True,synchronize=lambda:None),_C=SimpleNamespace())
    with pytest.raises(RuntimeError,match='verified pinned-host'):M.release_unused(fake)


def test_cpu_boundary_keeps_live_tensors_and_writes_telemetry(tmp_path):
    tensor=torch.arange(23);before=tensor.clone();path=tmp_path/'input';path.write_bytes(b'x')
    result=M.case_boundary(torch,paths=[path,path],log_path=tmp_path/'memory.jsonl',completed_updates=8)
    assert torch.equal(tensor,before) and not torch.cuda.is_initialized()
    assert len(result['files'])==1 and result['context']['completed_updates']==8
    assert 'VmRSS' in result['after']
