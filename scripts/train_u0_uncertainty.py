#!/usr/bin/env python3
"""CLI wrapper for the native-only U0 uncertainty trainer."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stablebridge.physical_repair.u0_uncertainty_trainer import main


if __name__ == "__main__":
    raise SystemExit(main())
