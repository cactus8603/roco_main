#!/usr/bin/env python3
"""Train one recurrent uncertainty-aware SEA-RAFT ablation."""
from __future__ import annotations

import argparse
from pathlib import Path

from stablebridge.physical_repair.integrated_uncertainty_trainer import (
    IntegratedTrainerConfigV2,
    IntegratedUncertaintyTrainerV2,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default="none", help="none, auto, or a checkpoint path")
    args = parser.parse_args()
    config = IntegratedTrainerConfigV2.from_json(args.config)
    trainer = IntegratedUncertaintyTrainerV2(config, device=args.device)
    if args.resume == "auto":
        if trainer.latest_path.is_file():
            trainer.resume(trainer.latest_path)
    elif args.resume != "none":
        trainer.resume(Path(args.resume))
    return trainer.train()


if __name__ == "__main__":
    raise SystemExit(main())
