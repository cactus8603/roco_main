#!/usr/bin/env python3
"""Materialize traced SAM masks, then launch the configured U2 training run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

from stablebridge.physical_repair.sam_semantic_smoothness import (
    SAM_FULL_SEGMENTATION_SCHEMA_V1,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--sam-output-root", type=Path, required=True)
    parser.add_argument("--sam-source", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="auto")
    return parser.parse_args()


def _complete_manifest(root: Path, checkpoint_sha256: str) -> bool:
    path = root / "manifest.json"
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, Mapping)
        and value.get("schema") == SAM_FULL_SEGMENTATION_SCHEMA_V1
        and value.get("complete") is True
        and value.get("sam_checkpoint_sha256") == checkpoint_sha256
    )


def main() -> int:
    args = _args()
    repository_root = Path(__file__).resolve().parents[1]
    checkpoint = args.sam_checkpoint.expanduser().resolve()
    checkpoint_sha256 = _sha256(checkpoint)
    output_root = args.sam_output_root.expanduser().resolve()
    if not _complete_manifest(output_root, checkpoint_sha256):
        generation = [
            sys.executable,
            str(repository_root / "scripts" / "generate_sintel_sam_full_segmentation.py"),
            "--data-config", str(args.data_config.expanduser().resolve()),
            "--output-root", str(output_root),
            "--sam-source", str(args.sam_source.expanduser().resolve()),
            "--checkpoint", str(checkpoint),
            "--model-type", str(args.sam_model_type),
            "--device", str(args.device),
            "--roles", "fit", "validation",
        ]
        completed = subprocess.run(generation, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    training = [
        sys.executable,
        str(repository_root / "scripts" / "train_integrated_uncertainty_flow.py"),
        "--config", str(args.trainer_config.expanduser().resolve()),
        "--device", str(args.device),
        "--resume", str(args.resume),
    ]
    return int(subprocess.run(training, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
