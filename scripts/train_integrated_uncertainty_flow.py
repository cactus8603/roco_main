#!/usr/bin/env python3
"""Train one recurrent uncertainty-aware SEA-RAFT ablation."""
from __future__ import annotations

import argparse
from pathlib import Path

from stablebridge.physical_repair.integrated_uncertainty_trainer import (
    IntegratedTrainerConfigV2,
    IntegratedUncertaintyTrainerV2,
)
from stablebridge.physical_repair.cuda_memory_guard import (
    CudaMemoryReservationPolicyV1,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default="none", help="none, auto, or a checkpoint path")
    parser.add_argument(
        "--reserve-vram", action=argparse.BooleanOptionalAction, default=None,
        help="runtime-only CUDA cache reservation override; does not alter checkpoint lineage",
    )
    parser.add_argument("--target-vram-fraction", type=float, default=0.88)
    parser.add_argument("--vram-headroom-mib", type=int, default=3072)
    args = parser.parse_args()
    config = IntegratedTrainerConfigV2.from_json(args.config)
    memory_override = None
    if args.reserve_vram is not None:
        memory_override = CudaMemoryReservationPolicyV1(
            enabled=args.reserve_vram,
            target_fraction=args.target_vram_fraction,
            minimum_headroom_mib=args.vram_headroom_mib,
            chunk_mib=256,
            retry_fraction_decrement=0.05,
            minimum_target_fraction=0.65,
        )
    trainer = IntegratedUncertaintyTrainerV2(
        config,
        device=args.device,
        cuda_memory_reservation_override=memory_override,
    )
    if args.resume == "auto":
        if trainer.latest_path.is_file():
            trainer.resume(trainer.latest_path)
    elif args.resume != "none":
        trainer.resume(Path(args.resume))
    return trainer.train()


if __name__ == "__main__":
    raise SystemExit(main())
