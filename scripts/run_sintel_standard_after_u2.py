#!/usr/bin/env python3
"""Wait for U2-main completion, then submit its standard Sintel evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _completed(metrics: Path) -> bool:
    if not metrics.is_file():
        return False
    with metrics.open("rb") as stream:
        for raw in stream:
            if b'"event": "training_completed"' in raw:
                return True
    return False


def _state(path: Path) -> Mapping[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sintel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--u2-scheduler-state-dir", type=Path, required=True)
    parser.add_argument("--evaluation-state-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--minimum-free-mib", type=int, default=10000)
    parser.add_argument("--allowed-indices", default="0,1,2,3,4,5,6")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    while not _completed(args.metrics):
        scheduler = _state(args.u2_scheduler_state_dir / "status.json")
        if scheduler and scheduler.get("state") in {"failed", "cancelled", "interrupted"}:
            raise RuntimeError(f"U2 scheduler terminated before training completion: {scheduler}")
        print(json.dumps({
            "event": "waiting_for_u2",
            "metrics": str(args.metrics),
            "utc": _utc_now(),
        }, sort_keys=True), flush=True)
        time.sleep(args.poll_seconds)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    repository = Path(__file__).resolve().parents[1]
    scheduler = repository / "scripts" / "run_gpu_training_when_free.py"
    evaluator = repository / "scripts" / "evaluate_sintel_standard.py"
    command = [
        sys.executable,
        str(scheduler),
        "start",
        "--state-dir", str(args.evaluation_state_dir),
        "--minimum-free-mib", str(args.minimum_free_mib),
        "--maximum-utilization-percent", "100",
        "--poll-seconds", "30",
        "--max-resource-retries", "2",
        "--allowed-indices", str(args.allowed_indices),
        "--",
        "/usr/bin/env",
        f"PYTHONPATH={repository / 'src'}:{repository}",
        "PYTHONDONTWRITEBYTECODE=1",
        sys.executable,
        str(evaluator),
        "--config", str(args.config),
        "--checkpoint", str(args.checkpoint),
        "--sintel-root", str(args.sintel_root),
        "--output", str(args.output),
        "--device", "cuda",
        "--iters", "4",
        "--scale", "0",
    ]
    completed = subprocess.run(command, cwd=repository, text=True, capture_output=True, check=False)
    print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
