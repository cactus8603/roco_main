"""Sequential E01 worker intended to be launched by the detached service script.

Process success means that the registered stages completed. Scientific effects,
including negative results and a calibrated reject-all policy, remain in each
stage's reports and are never inferred from an exit code.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .util import ROOT, save_json, sha256


@dataclass(frozen=True)
class Stage:
    stage_id: str
    command: tuple[str, ...]
    required_files: tuple[str, ...]
    completion_status: str | None = None


def build_plan(output_dir, *, device="cuda:1", epochs=30,
               include_full_temporal=False, root=ROOT) -> dict:
    """Read existing registration only; no directories, jobs, or GPU are created."""
    root = Path(root).resolve()
    output = Path(output_dir)
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    workflow_name = output.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", workflow_name):
        raise ValueError("Workflow directory name must use letters, digits, underscores, or hyphens")
    registrations = [
        ("smoke", "e01_smoke_v1.json", "S01_spatial_repair"),
        ("s01_spatial", "e01_s01_spatial_v1.json", "S01_spatial_repair"),
        ("s02_temporal_pilot", "e01_s02_temporal_pilot_v1.json", "S02_temporal_evidence"),
        ("s03_source_control", "e01_s03_source_control_v1.json", "S03_source_control"),
    ]
    if include_full_temporal:
        registrations.append(("s02_temporal_full", "e01_s02_temporal_full_v1.json", "S02_temporal_evidence"))
    inputs = set((root / "src/stablebridge").glob("*.py"))
    inputs.add(root / "configs/stablebridge/data_models_v1.json")
    stages, runs = [], {}
    settings = {}
    for stage_id, config_name, study in registrations:
        config_path = root / "configs/stablebridge" / config_name
        registration = json.loads(config_path.read_text())
        inputs.update((config_path, root / registration["clips_manifest"]))
        settings[stage_id] = registration
        run_dir = root / "experiments/E01_evidence_mechanism/studies" / study / "runs" / f"{workflow_name}_{stage_id}"
        runs[stage_id] = run_dir
        frozen_config = output / "snapshot" / config_path.relative_to(root)
        stages.append(Stage(stage_id,
                            (sys.executable, "-m", "stablebridge.cli", "run", "--config", str(frozen_config),
                             "--run-dir", str(run_dir), "--device", device),
                            tuple(str(run_dir / name) for name in ("manifest.json", "metrics.jsonl", "status.json")),
                            str(run_dir / "status.json")))
    # Calibration eligibility must be the policy actually deployed in both runs.
    policies = [(settings[key].get("matching", {}).get("max_update_px", 32.0),
                 settings[key].get("matching", {}).get("require_baseline_support", True))
                for key in ("s01_spatial", "s02_temporal_pilot")]
    if policies[0] != policies[1]:
        raise ValueError("S01 and S02 training sources must use identical candidate eligibility policies")
    dataset, checkpoint = output / "supervision.npz", output / "utility.pt"
    stages.append(Stage("collect_supervision",
                        (sys.executable, "-m", "stablebridge.cli", "collect-supervision", "--runs",
                         str(runs["s01_spatial"]), str(runs["s02_temporal_pilot"]), "--output", str(dataset)),
                        (str(dataset), str(dataset) + ".json")))
    train_command = [sys.executable, "-m", "stablebridge.learning", "--dataset", str(dataset),
                     "--output", str(checkpoint), "--epochs", str(epochs),
                     "--max-update-px", str(policies[0][0])]
    if not policies[0][1]:
        train_command.append("--allow-unsupported-baseline")
    stages.append(Stage("train_utility", tuple(train_command), (str(checkpoint), str(checkpoint) + ".json")))
    for key, study in (("s01_spatial", "S01_spatial_repair"), ("s02_temporal_pilot", "S02_temporal_evidence")):
        replay_dir = root / "experiments/E01_evidence_mechanism/studies" / study / "runs" / f"{workflow_name}_{key}_learned"
        stages.append(Stage("replay_" + key,
                            (sys.executable, "-m", "stablebridge.replay", "--run-dir", str(runs[key]),
                             "--acceptor", str(checkpoint), "--output-dir", str(replay_dir)),
                            tuple(str(replay_dir / name) for name in ("manifest.json", "decisions.jsonl", "results.json", "results.md"))))
    input_records = []
    for path in sorted(inputs):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Registered configuration/source is outside project root: {path}")
        input_records.append({"path": str(path), "relative_path": str(path.relative_to(root)), "sha256": sha256(path)})
    return {"schema_version": 1, "workflow_name": workflow_name, "root": str(root), "output_dir": str(output),
            "device": device, "epochs": epochs, "include_full_temporal": bool(include_full_temporal),
            "python_executable": sys.executable, "inputs": input_records,
            "stages": [asdict(stage) for stage in stages],
            "scientific_success_claim": False,
            "completion_semantics": "Registered processes and artifacts completed; consult scientific reports separately."}


def _journal(output: Path, kind: str, **fields):
    entry = {"event": kind, "unix_time": time.time(), "pid": os.getpid(), **fields}
    with (output / "journal.jsonl").open("a") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False, allow_nan=False), flush=True)


def _verify_inputs(plan):
    for item in plan["inputs"]:
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen workflow input changed: {path}; create a new workflow directory")


def _artifact_hashes(stage: Stage) -> dict[str, str]:
    if stage.completion_status is not None:
        status_path = Path(stage.completion_status)
        if not status_path.is_file() or json.loads(status_path.read_text()).get("state") != "completed":
            raise RuntimeError(f"Stage {stage.stage_id} did not publish a completed status")
    artifacts = {}
    for name in stage.required_files:
        path = Path(name)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Stage {stage.stage_id} is missing its required artifact: {path}")
        artifacts[name] = sha256(path)
    if stage.stage_id.startswith("replay_"):
        report_path = next(Path(name) for name in stage.required_files if Path(name).name == "results.json")
        report = json.loads(report_path.read_text())
        if report.get("kind") != "frozen_candidate_query_replay" or report.get("whole_image_result") is not False \
                or report.get("closed_loop_result") is not False:
            raise RuntimeError("Replay artifact has an unexpected or overstated evaluation scope")
    return artifacts


@contextmanager
def _worker_lock(output: Path):
    with (output / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another worker owns this workflow directory") from error
        yield


def execute_plan(plan: dict) -> None:
    """Run child processes serially; stop at the first failed/unverifiable stage.

    The caller detaches this worker. We never start/stop unrelated services, and
    subprocess argument lists are passed directly without shell interpolation.
    """
    # JSON normalization makes tuples written to a manifest equal on resume.
    plan = json.loads(json.dumps(plan))
    output, root = Path(plan["output_dir"]), Path(plan["root"])
    output.mkdir(parents=True, exist_ok=True)
    with _worker_lock(output):
        manifest = output / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != plan:
                raise ValueError("Workflow manifest hashes/settings changed; use a new workflow directory")
        else:
            _verify_inputs(plan)
            for item in plan["inputs"]:
                destination = output / "snapshot" / item["relative_path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item["path"], destination)
                if sha256(destination) != item["sha256"]:
                    raise ValueError("Workflow source changed while creating its snapshot")
            save_json(manifest, plan)
        progress_path = output / "stages.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        worker = {"pid": os.getpid(), "ppid": os.getppid(),
                  "systemd_invocation_id": os.environ.get("INVOCATION_ID"),
                  "scientific_success_claim": False}
        active_stage = None
        try:
            _verify_inputs(plan)
            save_json(output / "status.json", {**worker, "state": "running", "started_unix": time.time()})
            _journal(output, "workflow_started", stages=len(plan["stages"]))
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(root / "src") + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
            environment["PYTHONUNBUFFERED"] = "1"
            for stage_data in plan["stages"]:
                stage = Stage(**stage_data)
                active_stage = stage.stage_id
                _verify_inputs(plan)
                # Frozen config copies are executable inputs, not decorative backups.
                for item in plan["inputs"]:
                    copied = output / "snapshot" / item["relative_path"]
                    if not copied.is_file() or sha256(copied) != item["sha256"]:
                        raise ValueError(f"Workflow snapshot changed: {copied}")
                previous = progress.get(stage.stage_id, {})
                if previous.get("state") == "completed":
                    if previous.get("command") != list(stage.command) or previous.get("artifacts") != _artifact_hashes(stage):
                        raise ValueError(f"Completed stage artifacts changed: {stage.stage_id}; refusing to silently rerun")
                    _journal(output, "stage_resumed", stage=stage.stage_id)
                    continue
                attempt = int(previous.get("attempt", 0)) + 1
                log_path = output / "logs" / f"{stage.stage_id}.attempt{attempt:03d}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                stage_state = {"state": "running", "attempt": attempt, "command": list(stage.command),
                               "started_unix": time.time(), "log": str(log_path), "worker_pid": os.getpid()}
                progress[stage.stage_id] = stage_state
                save_json(progress_path, progress)
                save_json(output / "status.json", {**worker, "state": "running", "stage": stage.stage_id,
                                                     "log": str(log_path)})
                process = None
                try:
                    with log_path.open("a") as log:
                        process = subprocess.Popen(stage.command, cwd=root, env=environment,
                                                   stdout=log, stderr=subprocess.STDOUT)
                        stage_state["child_pid"] = process.pid
                        save_json(progress_path, progress)
                        _journal(output, "stage_started", stage=stage.stage_id, child_pid=process.pid,
                                 command=list(stage.command), log=str(log_path))
                        returncode = process.wait()
                    stage_state.update(returncode=returncode, finished_unix=time.time())
                    if returncode != 0:
                        raise RuntimeError(f"Stage {stage.stage_id} exited with code {returncode}; see {log_path}")
                    _verify_inputs(plan)
                    artifacts = _artifact_hashes(stage)
                except BaseException as error:
                    if process is not None and process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    stage_state.update(state="failed", error=repr(error), finished_unix=time.time())
                    save_json(progress_path, progress)
                    _journal(output, "stage_failed", stage=stage.stage_id, error=repr(error),
                             returncode=stage_state.get("returncode"), log=str(log_path))
                    raise
                stage_state.update(state="completed", artifacts=artifacts)
                save_json(progress_path, progress)
                _journal(output, "stage_completed", stage=stage.stage_id, returncode=returncode)
            save_json(output / "status.json", {**worker, "state": "completed", "finished_unix": time.time(),
                                                 "completed_stages": len(plan["stages"]),
                                                 "scientific_conclusion": "not_inferred_from_process_completion"})
            _journal(output, "workflow_completed", scientific_success_claim=False)
        except BaseException as error:
            save_json(output / "status.json", {**worker, "state": "failed", "stage": active_stage,
                                                 "error": repr(error), "finished_unix": time.time()})
            _journal(output, "workflow_failed", stage=active_stage, error=repr(error))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="operations/e01_workflow_v1")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--include-full-temporal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(args.output_dir, device=args.device, epochs=args.epochs,
                      include_full_temporal=args.include_full_temporal)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return
    execute_plan(plan)


if __name__ == "__main__":
    main()
