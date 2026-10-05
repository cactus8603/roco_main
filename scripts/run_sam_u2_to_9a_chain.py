#!/usr/bin/env python3
"""Continue completed U2-SAM into 9A-SAM and standardized Sintel evaluation.

The controller is durable and idempotent.  It does not own a GPU: each GPU
stage is submitted to the existing admission scheduler, so the next stage can
start as soon as any allowed card has enough free memory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
TRAINING_PYTHON = Path(
    "/ssd7/cactus8603/roco_spring/optical-flow-track/.venv/bin/python"
)
SCHEDULER = REPOSITORY / "scripts/run_gpu_training_when_free.py"
TRAINER = REPOSITORY / "scripts/train_integrated_uncertainty_flow.py"
EVALUATOR = REPOSITORY / "scripts/evaluate_sintel_standard.py"
U2_CONFIG = REPOSITORY / "configs/stablebridge/u2_sintel_searaft_uncertainty_refinement_sam_u0ft_v1.json"
NINE_A_CONFIG = REPOSITORY / "configs/stablebridge/u2_sintel_searaft_action_bank_training_sam_u0ft_v1.json"
U2_STATE = REPOSITORY / "operations/U2_sintel_searaft_uncertainty_refinement_sam_u0ft_v1_seed11"
NINE_A_STATE = REPOSITORY / "operations/U2_sintel_searaft_action_bank_training_sam_u0ft_v1_seed11"
EVALUATION_STATE = REPOSITORY / "operations/Sintel_standard_evaluation_9A_SAM_seed11"
SINTEL_ROOT = Path("/ssd7/cactus8603/roco_sintel_external_20260929/extracted")
CHAIN_SCHEMA = "stablebridge-sam-u2-to-9a-chain/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _config_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_completed_metric(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return any(b'"event": "training_completed"' in line for line in stream)


def _run_dir(config_path: Path) -> Path:
    config = _load_json(config_path)
    if config is None or not isinstance(config.get("run_dir"), str):
        raise ValueError(f"config has no run_dir: {config_path}")
    return Path(config["run_dir"]).expanduser().resolve()


def _scheduler_state(state_dir: Path) -> dict[str, Any] | None:
    return _load_json(state_dir / "status.json")


def _training_stage(config_path: Path, state_dir: Path) -> dict[str, Any]:
    run_dir = _run_dir(config_path)
    status = _scheduler_state(state_dir)
    if status is None:
        return {"state": "not_submitted", "run_dir": str(run_dir)}
    scheduler_state = status.get("state")
    result = {
        "state": scheduler_state,
        "run_dir": str(run_dir),
        "scheduler_state_dir": str(state_dir),
    }
    if scheduler_state == "completed":
        if status.get("child_returncode") != 0:
            return {**result, "state": "failed", "error": "nonzero child return code"}
        if not (run_dir / "best.pt").is_file():
            return {**result, "state": "failed", "error": "best checkpoint is missing"}
        if not _has_completed_metric(run_dir / "metrics.jsonl"):
            return {**result, "state": "failed", "error": "training_completed metric is missing"}
    elif scheduler_state in {"failed", "cancelled", "interrupted"}:
        result["state"] = "failed"
        result["error"] = status.get("error", f"scheduler state {scheduler_state}")
    elif scheduler_state not in {"waiting", "launching", "running"}:
        result["state"] = "failed"
        result["error"] = f"unexpected scheduler state {scheduler_state!r}"
    return result


def _validate_warm_start(source_config_path: Path, target_config_path: Path) -> dict[str, Any]:
    # Delayed import keeps the long-running polling process lightweight.
    import torch

    source_config = _load_json(source_config_path)
    target_config = _load_json(target_config_path)
    if source_config is None or target_config is None:
        raise ValueError("source/target config is missing")
    source_checkpoint = _run_dir(source_config_path) / "best.pt"
    configured = Path(str(target_config.get("model_initialization_checkpoint", ""))).resolve()
    if configured != source_checkpoint.resolve():
        raise ValueError("9A-SAM warm-start path does not point to U2-SAM best.pt")
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("U2-SAM checkpoint is not a mapping")
    if payload.get("schema") != "stablebridge-integrated-uncertainty-trainer/v2":
        raise ValueError("U2-SAM checkpoint schema drift")
    if payload.get("config_digest") != _config_digest(source_config):
        raise ValueError("U2-SAM checkpoint/config digest drift")
    if source_config.get("model") != target_config.get("model"):
        raise ValueError("U2-SAM and 9A-SAM model definitions differ")
    if source_config.get("sam_homography") != target_config.get("sam_homography"):
        raise ValueError("U2-SAM and 9A-SAM SAM objectives differ")
    return {
        "checkpoint": str(source_checkpoint),
        "checkpoint_sha256": _sha256_file(source_checkpoint),
        "source_config_digest": str(payload["config_digest"]),
        "source_best_validation_epe": payload.get("best_validation_epe"),
    }


def _submit_training(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        str(TRAINING_PYTHON), str(SCHEDULER), "start",
        "--state-dir", str(NINE_A_STATE),
        "--minimum-free-mib", str(args.training_minimum_free_mib),
        "--maximum-utilization-percent", "100",
        "--poll-seconds", str(args.gpu_poll_seconds),
        "--max-resource-retries", "3",
        "--allowed-indices", args.allowed_indices,
        "--", "/usr/bin/env",
        f"PYTHONPATH={REPOSITORY / 'src'}:{REPOSITORY}",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "PYTHONDONTWRITEBYTECODE=1",
        str(TRAINING_PYTHON), str(TRAINER),
        "--config", str(NINE_A_CONFIG), "--device", "cuda", "--resume", "auto",
    ]
    return _submit_scheduler(command, "9A-SAM")


def _submit_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _run_dir(NINE_A_CONFIG)
    command = [
        str(TRAINING_PYTHON), str(SCHEDULER), "start",
        "--state-dir", str(EVALUATION_STATE),
        "--minimum-free-mib", str(args.evaluation_minimum_free_mib),
        "--maximum-utilization-percent", "100",
        "--poll-seconds", str(args.gpu_poll_seconds),
        "--max-resource-retries", "2",
        "--allowed-indices", args.allowed_indices,
        "--", "/usr/bin/env",
        f"PYTHONPATH={REPOSITORY / 'src'}:{REPOSITORY}",
        "PYTHONDONTWRITEBYTECODE=1",
        str(TRAINING_PYTHON), str(EVALUATOR),
        "--config", str(NINE_A_CONFIG),
        "--checkpoint", str(run_dir / "best.pt"),
        "--sintel-root", str(SINTEL_ROOT),
        "--output", str(run_dir / "sintel_standard_evaluation.json"),
        "--device", "cuda", "--iters", "4", "--scale", "0",
    ]
    return _submit_scheduler(command, "9A-SAM standard evaluation")


def _submit_scheduler(command: list[str], label: str) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=REPOSITORY, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"failed to submit {label}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    launch = json.loads(completed.stdout)
    if launch.get("status") not in {"submitted", "already_active"}:
        raise RuntimeError(f"unexpected {label} launch result: {launch}")
    return launch


def _evaluation_stage() -> dict[str, Any]:
    run_dir = _run_dir(NINE_A_CONFIG)
    output = run_dir / "sintel_standard_evaluation.json"
    status = _scheduler_state(EVALUATION_STATE)
    if status is None:
        return {"state": "not_submitted", "output": str(output)}
    scheduler_state = status.get("state")
    result = {"state": scheduler_state, "output": str(output)}
    if scheduler_state == "completed":
        if status.get("child_returncode") != 0 or not output.is_file():
            return {**result, "state": "failed", "error": "evaluation output is missing or failed"}
    elif scheduler_state in {"failed", "cancelled", "interrupted"}:
        result["state"] = "failed"
        result["error"] = status.get("error", f"scheduler state {scheduler_state}")
    elif scheduler_state not in {"waiting", "launching", "running"}:
        result["state"] = "failed"
        result["error"] = f"unexpected scheduler state {scheduler_state!r}"
    return result


def advance_once(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": CHAIN_SCHEMA,
        "state": "running",
        "updated_utc": _utc_now(),
    }
    u2 = _training_stage(U2_CONFIG, U2_STATE)
    result["u2_sam"] = u2
    if u2["state"] == "failed":
        return {**result, "state": "failed", "failed_stage": "u2_sam"}
    if u2["state"] != "completed":
        result["active_stage"] = "u2_sam"
        return result

    warm_start = _validate_warm_start(U2_CONFIG, NINE_A_CONFIG)
    result["warm_start"] = warm_start
    nine_a = _training_stage(NINE_A_CONFIG, NINE_A_STATE)
    if nine_a["state"] == "not_submitted":
        result["training_launch"] = _submit_training(args)
        nine_a = _training_stage(NINE_A_CONFIG, NINE_A_STATE)
    result["nine_a_sam"] = nine_a
    if nine_a["state"] == "failed":
        return {**result, "state": "failed", "failed_stage": "nine_a_sam"}
    if nine_a["state"] != "completed":
        result["active_stage"] = "nine_a_sam"
        return result

    evaluation = _evaluation_stage()
    if evaluation["state"] == "not_submitted":
        result["evaluation_launch"] = _submit_evaluation(args)
        evaluation = _evaluation_stage()
    result["standard_evaluation"] = evaluation
    if evaluation["state"] == "failed":
        return {**result, "state": "failed", "failed_stage": "standard_evaluation"}
    if evaluation["state"] == "completed":
        result["state"] = "completed"
        result["active_stage"] = None
    else:
        result["active_stage"] = "standard_evaluation"
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir", type=Path,
        default=REPOSITORY / "operations/SAM_U2_to_9A_seed11",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=15.0)
    parser.add_argument("--training-minimum-free-mib", type=int, default=14000)
    parser.add_argument("--evaluation-minimum-free-mib", type=int, default=10000)
    parser.add_argument("--allowed-indices", default="0,1,2,3,4,5,6")
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.poll_seconds < 1 or args.gpu_poll_seconds < 1:
        raise SystemExit("poll intervals must be at least one second")
    status_path = args.state_dir.expanduser().resolve() / "status.json"
    while True:
        try:
            status = advance_once(args)
        except BaseException as error:
            status = {
                "schema": CHAIN_SCHEMA,
                "state": "failed",
                "updated_utc": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        _atomic_json(status_path, status)
        if status["state"] in {"completed", "failed"}:
            return 0 if status["state"] == "completed" else 1
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
