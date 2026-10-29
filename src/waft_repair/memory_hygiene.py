"""Release unused process caches at completed case boundaries, without data edits."""
import ctypes
import gc
import json
import os
from pathlib import Path
import stat


def advise_readonly(path):
    """Evict clean file-cache pages through a no-follow read-only descriptor."""
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
            raise ValueError('Only this owner regular files may receive cache advice')
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('Bound read-only file changed during cache advice')
        return dict(path=str(path), bytes=before.st_size, inode=before.st_ino,
                    advice='POSIX_FADV_DONTNEED', read_only=True)
    finally:
        os.close(fd)


def telemetry():
    result = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:', 'VmSwap:', 'VmLck:')):
            key, value = line.split(':', 1)
            result[key] = int(value.split()[0]) * 1024
    group = next((line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                  if line.startswith('0::')), None)
    if group is None:
        return result
    base = Path('/sys/fs/cgroup') / group.lstrip('/')
    for name in ('memory.current', 'memory.peak'):
        p = base/name
        if p.is_file():
            result[name] = int(p.read_text())
    for name in ('memory.stat', 'memory.events'):
        p = base/name
        if p.is_file():
            values = dict(line.split() for line in p.read_text().splitlines())
            keys = ('anon', 'file', 'file_mapped', 'file_dirty', 'unevictable', 'kernel') if name == 'memory.stat' else values
            result[name] = {key: int(values[key]) for key in keys if key in values}
    return result


def release_unused(torch_module=None):
    """Synchronize first; free only unused pinned and libc allocator blocks."""
    before = telemetry()
    gc.collect()
    pinned_cleared = False
    if torch_module is not None and torch_module.cuda.is_initialized():
        torch_module.cuda.synchronize()
        clear = getattr(torch_module._C, '_host_emptyCache', None)
        if not callable(clear):
            raise RuntimeError('Installed Torch must provide the verified pinned-host cache release')
        clear()
        pinned_cleared = True
    libc = ctypes.CDLL(None)
    trim = libc.malloc_trim
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    trimmed = int(trim(0))
    return dict(before=before, after=telemetry(), cuda_synchronized=pinned_cleared,
                unused_pinned_host_cache_released=pinned_cleared, malloc_trim_result=trimmed)


def case_boundary(torch_module=None, *, paths=(), log_path=None, **context):
    """Call after dropping the previous case's tensors and completing its work.

    Linux/glibc and PyTorch's verified pinned-host cache API are required for
    CUDA execution. Live tensors and CUDA allocator blocks are never freed.
    Files must be owned regular files; cache advice changes no file content.
    """
    record=release_unused(torch_module)
    record['after_allocator_release']=record['after']
    record['files']=[advise_readonly(path) for path in dict.fromkeys(map(Path,paths))]
    record['after']=telemetry()
    record['context']=context
    if log_path is not None:
        with Path(log_path).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
    return record
