#!/usr/bin/env python3
"""Launch an independently managed user service, never a session-bound process.

Usage: python scripts/run_stablebridge_detached.py --unit NAME --output-dir DIR
       [--dry-run] -- /absolute/python -m stablebridge.cli ...

Arguments are encoded and exec'ed without a shell, protecting literal $, %,
backticks, whitespace and newlines from systemd's ExecStart substitutions too.
The small trampoline becomes the requested process through os.execvpe.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys


_TRAMPOLINE = ("import base64,json,os,sys; "
               "p=json.loads(base64.b64decode(sys.argv[1])); "
               "os.chdir(p['cwd']); "
               "e=os.environ.copy(); e.update(p['environment']); "
               "os.execvpe(p['command'][0],p['command'],e)")


def build_launch(unit,output_dir,command,*,cwd=None,pythonpath=None,launcher_python=None):
    """Return a launch record with safe argv, without creating files/services."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}",unit):
        raise ValueError("Unit name must contain only letters, digits, underscore, dot and hyphen")
    unit = unit if unit.endswith(".service") else unit+".service"
    if not command or any(not isinstance(value,str) or "\0" in value for value in command):
        raise ValueError("Command must contain non-NUL string arguments")
    if not command[0]:
        raise ValueError("Command executable cannot be empty")
    root = Path(cwd or Path(__file__).resolve().parents[1]).resolve()
    output = Path(output_dir).resolve()
    source = Path(pythonpath or root/"src").resolve()
    python = str(Path(launcher_python or sys.executable).resolve())
    if any("\n" in str(path) or "\r" in str(path) for path in (root,output,source)):
        raise ValueError("Service directory paths cannot contain line breaks")
    if not root.is_dir() or not source.is_dir():
        raise ValueError("Working directory and Python import directory must exist")
    payload = {"command":list(command),"cwd":str(root),
               "environment":{"PYTHONPATH":str(source),"PYTHONUNBUFFERED":"1"}}
    encoded = base64.b64encode(json.dumps(payload,ensure_ascii=False).encode()).decode("ascii")
    # Unit properties expand % specifiers; %% preserves a literal percent.
    stdout = str(output/"stdout.log").replace("%","%%")
    stderr = str(output/"stderr.log").replace("%","%%")
    argv = ["systemd-run","--user",f"--unit={unit}","--remain-after-exit","--service-type=exec",
            "--property=KillMode=control-group",f"--property=StandardOutput=append:{stdout}",
            f"--property=StandardError=append:{stderr}","--",python,"-c",_TRAMPOLINE,encoded]
    return {"launcher_version":"stablebridge_systemd_detached_v1","unit":unit,"output_dir":str(output),
            "command":list(command),"working_directory":str(root),"environment":payload["environment"],
            "systemd_argv":argv,"shell":False,"remain_after_exit":True,
            "stdout_log":str(output/"stdout.log"),"stderr_log":str(output/"stderr.log"),
            "independence":"user_service_owned_by_systemd_not_codex_session",
            "time_limit":"none_by_launcher; worker config determines any job-specific budget"}


def _save(path,record):
    temporary = path.with_name(path.name+f".tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def launch(record,*,dry_run=False):
    """Create logs/launch record; launch only after complete validation."""
    output = Path(record["output_dir"])
    output.mkdir(parents=True,exist_ok=True)
    path = output/"launch.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("status") != "planned" or previous.get("systemd_argv") != record["systemd_argv"]:
            raise FileExistsError("Existing launch record is not an identical unlaunched plan; use a new output directory")
    result = {**record,"created_utc":datetime.now(timezone.utc).isoformat(),
              "status":"planned" if dry_run else "launching"}
    _save(path,result)
    if dry_run:
        return result
    try:
        process = subprocess.run(record["systemd_argv"],cwd=record["working_directory"],
                                 text=True,capture_output=True,check=False,timeout=30)
        result.update({"status":"submitted" if process.returncode==0 else "launch_failed",
                       "launch_returncode":process.returncode,"launch_stdout":process.stdout,
                       "launch_stderr":process.stderr})
    except (OSError,subprocess.TimeoutExpired) as error:
        # A timeout can occur after submission; do not imply it is safe to
        # relaunch or that the job failed. The named unit must be inspected.
        result.update({"status":"submission_state_unknown" if isinstance(error,subprocess.TimeoutExpired) else "launch_failed",
                       "launch_error":str(error)})
    _save(path,result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit",required=True)
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--cwd",type=Path)
    parser.add_argument("--pythonpath",type=Path)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1]==["--"] else args.command
    try:
        record = build_launch(args.unit,args.output_dir,command,cwd=args.cwd,pythonpath=args.pythonpath)
        result = launch(record,dry_run=args.dry_run)
    except (ValueError,FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({"unit":result["unit"],"status":result["status"],
                      "launch_record":str(Path(result["output_dir"])/"launch.json")},ensure_ascii=False))
    return 0 if result["status"] in ("planned","submitted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
