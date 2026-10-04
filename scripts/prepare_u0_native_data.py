"""Build the hash-bound, label-unopened U0 native flow data plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stablebridge.physical_repair.u0_native_data import (
    materialize_u0_native_data_plan_v1,
    verify_u0_native_data_plan_file_v1,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/stablebridge/u0_native_flow_data_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = materialize_u0_native_data_plan_v1(args.config, args.output)
    verified = verify_u0_native_data_plan_file_v1(args.output)
    if verified.plan_hash != plan.plan_hash:
        raise RuntimeError("serialized U0 input plan failed exact reload")
    print(json.dumps({
        "status": "PASS_NATIVE_INPUT_PLAN",
        "output": str(args.output.resolve()),
        "plan_hash": plan.plan_hash,
        "split_hash": plan.split_hash,
        "probe_recipe_hash": plan.probe_recipe_hash,
        "rows": len(plan.rows),
        "groups": len({row.group_id for row in plan.rows}),
        "labels_opened": False,
        "matcher_outputs_materialized": False,
        "training_started": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
