"""Verified, resumable acquisition of the two official CroCo main checkpoints.

Official checkpoints contain an argparse.Namespace, so inspecting them requires
``weights_only=False``.  Only bytes acquired from the pinned HTTPS publisher URL
(or previously recorded official legacy hashes) are admitted to that operation.
The first download records a digest; it is not a publisher-signed checksum.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable
from urllib.request import Request, urlopen


OFFICIAL_BASE = "https://download.europe.naverlabs.com/ComputerVision/CroCo/StereoFlow_models/"
OFFICIAL_FILES = {"stereo": "crocostereo.pth", "flow": "crocoflow.pth"}
LEGACY_SHA256 = {
    "stereo": "123273a722a58134c6816efe5122c427815c92f82896be4d484e6d21b52da575",
    "flow": "1fdab7edbae9e13a171a85972376d08e6d0650003197f8eaaeabd92a7d415eb6",
}
TRAINING_DESCRIPTION = "https://github.com/naver/croco/blob/master/stereoflow/README.MD"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def _locked(path: Path):
    with path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _task(task: str) -> str:
    if task not in OFFICIAL_FILES:
        raise ValueError(f"Expected stereo or flow, got {task!r}")
    return task


def _check_digest(actual: str, expected: str | None) -> None:
    if expected is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError("expected_sha256 must be a 64-character SHA256 digest")
        if actual != expected.lower():
            raise ValueError(f"Checkpoint SHA256 mismatch: expected {expected}, got {actual}")


def verify_checkpoint_provenance(
    checkpoint: str | Path, task: str, expected_sha256: str | None = None
) -> dict:
    """Authenticate a recorded artifact before unpickling; never infer from a name."""
    task = _task(task)
    checkpoint = Path(checkpoint).resolve(strict=True)
    actual = sha256_file(checkpoint)
    _check_digest(actual, expected_sha256)
    if actual == LEGACY_SHA256[task]:
        return {"path": str(checkpoint), "sha256": actual, "task": task,
                "source_url": OFFICIAL_BASE + f"croco{task}_finetune_spring.pth",
                "role": "legacy_spring", "spring_supervised_finetuning": True,
                "upstream_pretraining_overlap": "not_fully_audited"}
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if not sidecar.exists():
        raise ValueError("Unregistered checkpoint: use ensure_official_checkpoint; filename alone is not provenance")
    metadata = json.loads(sidecar.read_text())
    if (metadata.get("source_url") != OFFICIAL_BASE + OFFICIAL_FILES[task]
            or metadata.get("task") != task or metadata.get("sha256") != actual
            or metadata.get("size_bytes") != checkpoint.stat().st_size):
        raise ValueError("Checkpoint bytes do not match the recorded official artifact")
    return {**metadata, "path": str(checkpoint)}


def load_official_checkpoint(checkpoint: str | Path, task: str,
                             expected_sha256: str | None = None):
    """Return (checkpoint mapping, provenance), checking origin before torch.load."""
    import torch
    provenance = verify_checkpoint_provenance(checkpoint, task, expected_sha256)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    _validate_state(state, task)
    return state, provenance


def _validate_state(state, task: str) -> None:
    if not isinstance(state, dict) or "args" not in state or "model" not in state:
        raise ValueError("Not an official downstream CroCo checkpoint")
    args = state["args"]
    if getattr(args, "task", None) != task:
        raise ValueError(f"Checkpoint task mismatch: expected {task}")
    if not str(getattr(args, "criterion", "")).startswith("LaplacianLoss"):
        raise ValueError("Unsupported checkpoint output/uncertainty convention")
    if not isinstance(getattr(args, "croco_args", None), dict):
        raise ValueError("Missing CroCo architecture arguments")


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return repr(value)


def _download(url: str, part: Path, state_path: Path,
              progress: Callable[[int, int | None], None] | None = None) -> dict:
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous and previous.get("source_url") != url:
        raise ValueError("Partial artifact belongs to another URL")
    offset = part.stat().st_size if part.exists() else 0
    # If publication metadata was interrupted after all bytes reached disk,
    # revalidate those complete bytes without requesting an unsatisfiable range.
    if offset and previous.get("total_bytes") == offset:
        return previous
    # Partial bytes without a publisher validator cannot safely be concatenated
    # with a possibly changed remote object. Restart in that uncommon case.
    validator = previous.get("etag") or previous.get("last_modified")
    headers = {"User-Agent": "StableBridge-research-artifact/1", "Accept-Encoding": "identity"}
    if offset and validator:
        headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
    else:
        offset = 0
    with urlopen(Request(url, headers=headers), timeout=90) as response:
        if response.geturl() != url:
            raise ValueError("Unexpected artifact redirect; verify the publisher URL before downloading")
        status = response.status
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if match is None or int(match[1]) != offset:
                raise ValueError("Invalid HTTP resume range")
            total = int(match[3])
            if int(match[2]) < offset or int(match[2]) >= total:
                raise ValueError("Invalid HTTP resume extent")
            mode = "ab" if offset else "wb"
        elif status == 200:
            offset, mode = 0, "wb"
            length = response.headers.get("Content-Length")
            total = int(length) if length is not None else None
        else:
            raise ValueError(f"Unexpected download response: {status}")
        state = {"source_url": url, "total_bytes": total,
                 "etag": response.headers.get("ETag"),
                 "last_modified": response.headers.get("Last-Modified")}
        with part.open(mode) as stream:
            # Truncate a restarted object before recording its new validator;
            # otherwise a crash could attach new provenance to old partial bytes.
            _atomic_json(state_path, state)
            received = offset
            while chunk := response.read(4 * 1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
                if progress:
                    progress(received, total)
            stream.flush()
            os.fsync(stream.fileno())
        if total is not None and received != total:
            raise IOError(f"Incomplete artifact ({received} of {total} bytes); rerun to resume")
        if received == 0:
            raise IOError("Publisher returned an empty artifact")
        state["total_bytes"] = received
        _atomic_json(state_path, state)
        return state


def ensure_official_checkpoint(task: str, directory: str | Path,
                               expected_sha256: str | None = None,
                               progress: Callable[[int, int | None], None] | None = None) -> dict:
    """Download exactly one named main model, inspect it and atomically publish it.

    Interrupted downloads remain as ``.part`` plus resume metadata. The published
    checkpoint never denotes partial bytes. A changed/missing provenance record
    is an error, not permission to fall back to a different checkpoint.
    """
    import torch
    task = _task(task)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / OFFICIAL_FILES[task]
    sidecar = destination.with_suffix(destination.suffix + ".json")
    with _locked(destination.with_suffix(destination.suffix + ".lock")):
        if destination.exists():
            return verify_checkpoint_provenance(destination, task, expected_sha256)
        url = OFFICIAL_BASE + OFFICIAL_FILES[task]
        part = destination.with_suffix(destination.suffix + ".part")
        resume = part.with_suffix(part.suffix + ".json")
        transport = _download(url, part, resume, progress)
        digest = sha256_file(part)
        _check_digest(digest, expected_sha256)
        # Only the fixed official HTTPS source above reaches unsafe-pickle load.
        state = torch.load(part, map_location="cpu", weights_only=False, mmap=True)
        _validate_state(state, task)
        args = state["args"]
        metadata = {
            "schema_version": 1, "task": task, "role": "E01_scientific_main",
            "path": str(destination), "source_url": url, "sha256": digest,
            "size_bytes": part.stat().st_size,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "checksum_basis": "caller_pinned" if expected_sha256 else "first_https_download_recorded_locally",
            "publisher_checksum_independently_verified": False,
            "crop": _json_safe(getattr(args, "crop", args.croco_args.get("img_size"))),
            "criterion": str(args.criterion), "croco_args": _json_safe(args.croco_args),
            "checkpoint_args": _json_safe(vars(args)),
            "spring_supervised_finetuning": False,
            "exposure_basis": "official_main_training_command_does_not_list_Spring",
            "exposure_source": TRAINING_DESCRIPTION,
            "upstream_pretraining_overlap": "not_fully_audited",
            "transport": transport,
        }
        del state
        # Record provenance before publication; if interrupted between these two
        # atomic replacements the next run re-inspects the complete .part file.
        _atomic_json(sidecar, metadata)
        part.replace(destination)
        resume.unlink(missing_ok=True)
        return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(OFFICIAL_FILES))
    parser.add_argument("--directory", required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    print(json.dumps(ensure_official_checkpoint(args.task, args.directory, args.expected_sha256),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
