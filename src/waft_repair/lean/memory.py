"""Release unused process caches without collecting unused telemetry."""
import ctypes
import gc
import os
from pathlib import Path
import stat


_LIBC = ctypes.CDLL(None)
_MALLOC_TRIM = _LIBC.malloc_trim
_MALLOC_TRIM.argtypes = [ctypes.c_size_t]
_MALLOC_TRIM.restype = ctypes.c_int


def advise_readonly(path):
    """Evict clean file-cache pages through a no-follow read-only descriptor."""
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
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


def release_unused(torch_module=None):
    """Keep per-boundary GC, CUDA ordering, pinned-host release, and libc trim."""
    gc.collect()
    pinned_cleared = False
    if torch_module is not None and torch_module.cuda.is_initialized():
        torch_module.cuda.synchronize()
        clear = getattr(torch_module._C, '_host_emptyCache', None)
        if not callable(clear):
            raise RuntimeError('Installed Torch must provide the verified pinned-host cache release')
        clear()
        pinned_cleared = True
    trimmed = int(_MALLOC_TRIM(0))
    return dict(cuda_synchronized=pinned_cleared,
                unused_pinned_host_cache_released=pinned_cleared,
                malloc_trim_result=trimmed)
