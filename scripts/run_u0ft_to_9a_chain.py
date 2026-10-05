#!/usr/bin/env python3
"""Advance the seed-11 U0-finetune lineage through U1, U2, and 9A.

The controller never reserves a GPU itself.  It waits for both jobs in a
stage to finish successfully, verifies their basic completion artifacts, and
then submits the next pair to the durable GPU scheduler.  Any failed or
malformed upstream job stops the chain instead of silently skipping ahead.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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
CHAIN_SCHEMA = "stablebridge-u0ft-to-9a-chain/v1"


@dataclass(frozen=True)
class Job:
    name: str
    config: Path
    state_dir: Path


STAGES = (
    (
        "u1",
        (
            Job(
                "U1-main",
                REPOSITORY / "configs/stablebridge/u1_sintel_searaft_recurrent_refiner_u0ft_v2.json",
                REPOSITORY / "operations/U1_sintel_searaft_recurrent_refiner_u0ft_v2_seed11",
            ),
            Job(
                "U1-dummy",
                REPOSITORY / "configs/stablebridge/u1_sintel_searaft_recurrent_refiner_no_uncertainty_u0ft_v2.json",
                REPOSITORY / "operations/U1_sintel_searaft_recurrent_refiner_no_uncertainty_u0ft_v2_seed11",
            ),
        ),
    ),
    (
        "u2",
        (
            Job(
                "U2-main",
                REPOSITORY / "configs/stablebridge/u2_sintel_searaft_uncertainty_refinement_u0ft_v2.json",
                REPOSITORY / "operations/U2_sintel_searaft_uncertainty_refinement_u0ft_v2_seed11",
            ),
            Job(
                "U2-dummy",
                REPOSITORY / "configs/stablebridge/u2_sintel_searaft_refinement_no_uncertainty_u0ft_v2.json",
                REPOSITORY / "operations/U2_sintel_searaft_refinement_no_uncertainty_u0ft_v2_seed11",
            ),
        ),
    ),
    (
        "9a",
        (
            Job(
                "9A-main",
                REPOSITORY / "configs/stablebridge/u2_sintel_searaft_action_bank_training_u0ft_v1.json",
                REPOSITORY / "operations/U2_sintel_searaft_action_bank_training_u0ft_v1_seed11",
            ),
            Job(
                "9A-dummy",
                REPOSITORY / "configs/stablebridge/u2_sintel_searaft_action_bank_no_uncertainty_u0ft_v1.json",
                REPOSITORY / "operations/U2_sintel_searaft_action_bank_no_uncertainty_u0ft_v1_seed11",
            ),
        ),
    ),
)


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


def _run_dir(job: Job) -> Path:
    config = _load_json(job.config)
    if config is None or not isinstance(config.get("run_dir"), str):
        raise ValueError(f"job has no valid run_dir: {job.config}")
    return Path(config["run_dir"]).expanduser().resolve()


def _has_completed_metric(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("event") == "training_completed":
            return True
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"metrics row {line_number} is not an object: {path}")
        rows.append(value)
    if not rows:
        raise ValueError(f"metrics file is empty: {path}")
    return rows


def _audit_alternating_schedule(
    config: Mapping[str, Any], rows: list[dict[str, Any]], *, job_name: str,
) -> dict[str, Any] | None:
    schedule = config.get("u2_schedule")
    if schedule is None:
        return None
    if not isinstance(schedule, Mapping):
        raise ValueError(f"{job_name} has an invalid U2 schedule")
    observer_epochs = int(schedule["observer_epochs_per_round"])
    flow_epochs = int(schedule["flow_epochs_per_round"])
    epochs_per_round = observer_epochs + flow_epochs
    action_bank = config.get("action_bank")
    training = config.get("training")
    if not isinstance(action_bank, Mapping) or not isinstance(training, Mapping):
        raise ValueError(f"{job_name} has invalid action/training config")
    action_ids = tuple(str(item) for item in action_bank["action_ids"])
    epochs = int(training["epochs"])
    steps = [row for row in rows if row.get("event") == "train_step"]
    by_epoch: dict[int, list[dict[str, Any]]] = {}
    for row in steps:
        by_epoch.setdefault(int(row["epoch"]), []).append(row)
    if set(by_epoch) != set(range(epochs)):
        raise ValueError(f"{job_name} train-step epoch coverage is incomplete")
    observed_pairs: set[tuple[str, str]] = set()
    for epoch in range(epochs):
        round_index = epoch // epochs_per_round
        position = epoch % epochs_per_round
        expected_phase = "observer" if position < observer_epochs else "flow"
        action_epoch = round_index if action_bank["mode"] == "cycle" else epoch
        expected_action = action_ids[action_epoch % len(action_ids)]
        signatures = {
            (str(row.get("phase")), int(row.get("round_index", -1)), str(row.get("action_id")))
            for row in by_epoch[epoch]
        }
        expected = {(expected_phase, round_index, expected_action)}
        if signatures != expected:
            raise ValueError(
                f"{job_name} epoch {epoch} action/phase drift: "
                f"expected {expected}, observed {signatures}"
            )
        observed_pairs.add((expected_action, expected_phase))
    expected_pairs = {
        (action_id, phase)
        for action_id in action_ids
        for phase in ("observer", "flow")
    }
    if observed_pairs != expected_pairs:
        raise ValueError(f"{job_name} did not observe every configured action in both phases")
    validation_epochs = {
        int(row["epoch"])
        for row in rows if row.get("event") == "validation_epoch"
    }
    validation_every = int(training["validation_every_epochs"])
    expected_validation_epochs = {
        epoch for epoch in range(epochs) if (epoch + 1) % validation_every == 0
    }
    if validation_epochs != expected_validation_epochs:
        raise ValueError(f"{job_name} validation epoch coverage is incomplete")
    return {
        "epochs": epochs,
        "action_ids": list(action_ids),
        "observed_action_phase_pairs": len(observed_pairs),
        "train_steps": len(steps),
    }


def _audit_checkpoint(
    path: Path,
    *,
    config: Mapping[str, Any],
    expected_digest: str,
    source_hashes: dict[Path, str],
    require_final_epoch: bool,
) -> dict[str, Any]:
    # Delayed import keeps the long-lived controller lightweight until the
    # whole chain is otherwise complete.
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    if payload.get("schema") != "stablebridge-integrated-uncertainty-trainer/v2":
        raise ValueError(f"checkpoint schema drift: {path}")
    checkpoint_config = payload.get("config")
    if (
        not isinstance(checkpoint_config, Mapping)
        or _config_digest(checkpoint_config) != expected_digest
        or payload.get("config_digest") != expected_digest
    ):
        raise ValueError(f"checkpoint config digest drift: {path}")
    checkpoint_bank = payload.get("action_bank_lineage")
    configured_bank = config.get("action_bank")
    if (
        not isinstance(checkpoint_bank, Mapping)
        or not isinstance(configured_bank, Mapping)
        or _config_digest(checkpoint_bank) != _config_digest(configured_bank)
    ):
        raise ValueError(f"checkpoint action-bank lineage drift: {path}")

    def check_source(config_key: str, lineage_key: str) -> Mapping[str, Any] | None:
        configured = config.get(config_key)
        lineage = payload.get(lineage_key)
        if configured is None:
            if lineage is not None:
                raise ValueError(f"unexpected {lineage_key}: {path}")
            return None
        source = Path(str(configured)).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"missing initialization source {source}")
        if not isinstance(lineage, Mapping):
            raise ValueError(f"missing {lineage_key}: {path}")
        if Path(str(lineage.get("checkpoint"))).expanduser().resolve() != source:
            raise ValueError(f"{lineage_key} path drift: {path}")
        source_hash = source_hashes.get(source)
        if source_hash is None:
            source_hash = _sha256_file(source)
            source_hashes[source] = source_hash
        if lineage.get("checkpoint_sha256") != source_hash:
            raise ValueError(f"{lineage_key} SHA-256 drift: {path}")
        return lineage

    uncertainty_lineage = check_source(
        "uncertainty_initialization_checkpoint", "uncertainty_initialization_lineage",
    )
    refiner_lineage = check_source(
        "refiner_initialization_checkpoint", "refiner_initialization_lineage",
    )
    model_lineage = check_source(
        "model_initialization_checkpoint", "model_initialization_lineage",
    )
    if (
        refiner_lineage is not None
        and refiner_lineage.get("uncertainty_initialization_lineage") != uncertainty_lineage
    ):
        raise ValueError(f"refiner and uncertainty initialization lineages differ: {path}")
    if model_lineage is not None and (
        uncertainty_lineage is not None or refiner_lineage is not None
    ):
        raise ValueError(f"full and partial initialization lineages coexist: {path}")
    if require_final_epoch:
        epochs = int(config["training"]["epochs"])
        if int(payload.get("epoch", -1)) != epochs or int(payload.get("next_batch_index", -1)) != 0:
            raise ValueError(f"latest checkpoint has an incomplete epoch cursor: {path}")
    if int(payload.get("global_step", 0)) <= 0:
        raise ValueError(f"checkpoint has no completed optimizer steps: {path}")
    return {
        "path": str(path),
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "config_digest": str(payload["config_digest"]),
    }


def audit_completed_chain() -> dict[str, Any]:
    source_hashes: dict[Path, str] = {}
    reports: list[dict[str, Any]] = []
    for _stage_name, jobs in STAGES:
        for job in jobs:
            run_dir = _run_dir(job)
            config = _load_json(job.config)
            resolved = _load_json(run_dir / "resolved_config.json")
            if config is None or resolved is None or resolved != config:
                raise ValueError(f"{job.name} resolved config differs from submitted config")
            rows = _read_metrics(run_dir / "metrics.jsonl")
            if rows[-1].get("event") != "training_completed":
                raise ValueError(f"{job.name} metrics do not end in training_completed")
            expected_digest = _config_digest(resolved)
            started_digests = {
                row.get("config_digest")
                for row in rows if row.get("event") == "training_started"
            }
            if started_digests != {expected_digest}:
                raise ValueError(f"{job.name} training-start config digest drift")
            schedule_report = _audit_alternating_schedule(
                resolved, rows, job_name=job.name,
            )
            best = _audit_checkpoint(
                run_dir / "best.pt", config=resolved, expected_digest=expected_digest,
                source_hashes=source_hashes, require_final_epoch=False,
            )
            latest = _audit_checkpoint(
                run_dir / "latest.pt", config=resolved, expected_digest=expected_digest,
                source_hashes=source_hashes, require_final_epoch=True,
            )
            reports.append({
                "name": job.name,
                "run_dir": str(run_dir),
                "best": best,
                "latest": latest,
                "schedule": schedule_report,
            })
    return {
        "audited_utc": _utc_now(),
        "jobs": reports,
        "source_checkpoint_sha256": {
            str(path): digest for path, digest in sorted(source_hashes.items())
        },
    }


def inspect_job(job: Job) -> dict[str, Any]:
    status = _load_json(job.state_dir / "status.json")
    if status is None:
        return {"name": job.name, "state": "not_submitted"}
    state = status.get("state")
    result: dict[str, Any] = {"name": job.name, "state": state}
    if state == "completed":
        run_dir = _run_dir(job)
        returncode = status.get("child_returncode")
        result.update({
            "child_returncode": returncode,
            "best_checkpoint": str(run_dir / "best.pt"),
            "metrics": str(run_dir / "metrics.jsonl"),
        })
        if returncode != 0:
            result["state"] = "failed"
            result["error"] = "completed scheduler recorded a nonzero child return code"
        elif not (run_dir / "best.pt").is_file():
            result["state"] = "failed"
            result["error"] = "completed training has no best checkpoint"
        elif not _has_completed_metric(run_dir / "metrics.jsonl"):
            result["state"] = "failed"
            result["error"] = "completed training has no training_completed metric"
    elif state in {"failed", "cancelled", "interrupted"}:
        result["error"] = status.get("error", f"scheduler entered terminal state {state}")
    elif state not in {"waiting", "launching", "running"}:
        result["state"] = "failed"
        result["error"] = f"unexpected scheduler state {state!r}"
    return result


def _submit(job: Job, args: argparse.Namespace) -> dict[str, Any]:
    command = [
        str(TRAINING_PYTHON), str(SCHEDULER), "start",
        "--state-dir", str(job.state_dir),
        "--minimum-free-mib", str(args.minimum_free_mib),
        "--maximum-utilization-percent", str(args.maximum_utilization_percent),
        "--poll-seconds", str(args.gpu_poll_seconds),
        "--max-resource-retries", str(args.max_resource_retries),
        "--allowed-indices", args.allowed_indices,
        "--",
        "/usr/bin/env",
        f"PYTHONPATH={REPOSITORY / 'src'}:{REPOSITORY}",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "PYTHONDONTWRITEBYTECODE=1",
        str(TRAINING_PYTHON), str(TRAINER),
        "--config", str(job.config), "--device", "cuda", "--resume", "auto",
    ]
    process = subprocess.run(
        command, cwd=REPOSITORY, text=True, capture_output=True, check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"failed to submit {job.name}: {process.stderr.strip() or process.stdout.strip()}"
        )
    launch = json.loads(process.stdout)
    if launch.get("status") not in {"submitted", "already_active"}:
        raise RuntimeError(f"unexpected launch result for {job.name}: {launch}")
    return launch


def advance_once(args: argparse.Namespace) -> dict[str, Any]:
    stage_rows = []
    active_stage: str | None = None
    completed_stages: list[str] = []
    for stage_name, jobs in STAGES:
        rows = [inspect_job(job) for job in jobs]
        if any(row["state"] == "failed" for row in rows):
            return {
                "schema": CHAIN_SCHEMA,
                "state": "failed",
                "updated_utc": _utc_now(),
                "failed_stage": stage_name,
                "stages": stage_rows + [{"stage": stage_name, "jobs": rows}],
            }
        if all(row["state"] == "completed" for row in rows):
            completed_stages.append(stage_name)
            stage_rows.append({"stage": stage_name, "state": "completed", "jobs": rows})
            continue
        active_stage = stage_name
        launches = []
        for job, row in zip(jobs, rows):
            if row["state"] == "not_submitted":
                launches.append({"name": job.name, "launch": _submit(job, args)})
        refreshed = [inspect_job(job) for job in jobs]
        stage_rows.append({
            "stage": stage_name,
            "state": "active",
            "jobs": refreshed,
            "launches": launches,
        })
        break
    chain_state = "completed" if len(completed_stages) == len(STAGES) else "running"
    result = {
        "schema": CHAIN_SCHEMA,
        "state": chain_state,
        "updated_utc": _utc_now(),
        "active_stage": active_stage,
        "completed_stages": completed_stages,
        "stages": stage_rows,
    }
    if chain_state == "completed":
        result["completion_audit"] = audit_completed_chain()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=REPOSITORY / "operations/U0FT_to_9A_seed11",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=15.0)
    parser.add_argument("--minimum-free-mib", type=int, default=16000)
    parser.add_argument("--maximum-utilization-percent", type=int, default=10)
    parser.add_argument("--max-resource-retries", type=int, default=1)
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
