#!/usr/bin/env python3
"""Generic durable GPU admission wrapper for U0/U1/U2 training commands."""

from pathlib import Path
import importlib.util
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "stablebridge" / "physical_repair" / "gpu_wait_scheduler.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "stablebridge_gpu_wait_scheduler", _MODULE_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load GPU scheduler from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


if __name__ == "__main__":
    raise SystemExit(main())
