#!/usr/bin/env python3
"""Reject project write destinations that lexically or physically use /ssd8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath


FORBIDDEN = PurePosixPath("/ssd8")


def is_under_ssd8(value: str) -> bool:
    lexical = PurePosixPath(value)
    if not lexical.is_absolute():
        lexical = PurePosixPath(str(Path.cwd())) / lexical
    if lexical == FORBIDDEN or FORBIDDEN in lexical.parents:
        return True

    # strict=False also exposes an existing parent symlink into /ssd8 while
    # allowing callers to validate a destination before it is created.
    resolved = PurePosixPath(str(Path(value).expanduser().resolve(strict=False)))
    return resolved == FORBIDDEN or FORBIDDEN in resolved.parents


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed when a proposed project write path uses /ssd8."
    )
    parser.add_argument("paths", nargs="+", help="proposed file or directory destinations")
    args = parser.parse_args()
    results = [
        {"path": value, "allowed": not is_under_ssd8(value)}
        for value in args.paths
    ]
    print(json.dumps({"policy": "NO_PROJECT_STORAGE_ON_SSD8", "results": results}, sort_keys=True))
    return 0 if all(item["allowed"] for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
