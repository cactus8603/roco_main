"""Portable launcher for the SAM2.1, joint-action, no-homography mainline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .cuda_memory_guard import CudaMemoryReservationPolicyV1
from .integrated_uncertainty_trainer import (
    IntegratedTrainerConfigV2,
    IntegratedUncertaintyTrainerV2,
)
from .sam_semantic_smoothness import SAM_FULL_SEGMENTATION_SCHEMA_V1
from .sea_raft_loader import (
    verify_pinned_sea_raft_identity,
)


PORTABLE_SAM21_TRAINING_SCHEMA_V1 = "roco-portable-sam21-training/v1"
SAM21_TRAINING_MASK_SCHEMA_V1 = "stablebridge-sam21-training-masks/v1"
SAM21_CHECKPOINT_SHA256 = (
    "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
)
INITIALIZATION_CHECKPOINT_SHA256 = (
    "452eb049594d9940882b15d8b15cd166f27f4db248cf742c1f9e71c86957ab6f"
)
INITIALIZATION_CHECKPOINT_BYTES = 238_305_806
RESNET34_CHECKPOINT_SHA256 = (
    "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"
)
EXPECTED_SINTEL_FRAMES = 2_128
EXPECTED_MINIMUM_KEY_OBJECTS = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_stable_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(value), indent=2, sort_keys=True, allow_nan=False,
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"refusing to change an existing launch input: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _completed_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"initialization metrics.jsonl must accompany best.pt: {path}"
        )
    completed: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and row.get("event") == "training_completed":
            completed = row
    if completed is None:
        raise ValueError("initialization metrics has no training_completed event")
    return completed


def _resolve_full_segmentation_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if (root / "full_seg" / "manifest.json").is_file():
        return (root / "full_seg").resolve()
    if (root / "manifest.json").is_file():
        return root
    raise FileNotFoundError(
        f"SAM2.1 output has no full_seg/manifest.json: {root}"
    )


def validate_sam21_masks(
    *, sam_mask_root: Path, sintel_root: Path,
) -> tuple[Path, dict[str, Any]]:
    full_root = _resolve_full_segmentation_root(sam_mask_root)
    manifest_path = full_root / "manifest.json"
    manifest = _json(manifest_path)
    records = manifest.get("records")
    required = {
        "schema": SAM_FULL_SEGMENTATION_SCHEMA_V1,
        "extended_schema": SAM21_TRAINING_MASK_SCHEMA_V1,
        "complete": True,
        "dataset": "sintel",
        "sam_checkpoint_sha256": SAM21_CHECKPOINT_SHA256,
        "sam_model_type": "sam2.1_hiera_large",
        "inventory_frame_count": EXPECTED_SINTEL_FRAMES,
        "inventory_truncated": False,
        "selected_frame_count": EXPECTED_SINTEL_FRAMES,
        "frame_count": EXPECTED_SINTEL_FRAMES,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"SAM2.1 manifest {field} drift: "
                f"{manifest.get(field)!r} != {expected!r}"
            )
    if not isinstance(records, Mapping) or len(records) != EXPECTED_SINTEL_FRAMES:
        raise ValueError("SAM2.1 manifest records do not cover full Sintel inventory")
    source_root = Path(str(manifest.get("source_root", ""))).resolve()
    if source_root != sintel_root.resolve():
        raise ValueError(
            "SAM2.1 source_root differs from --sintel-root; republish the "
            "manifest by rerunning the generator with the same parameters"
        )
    expected_keys = {
        str(path.resolve().relative_to(sintel_root.resolve()))
        for render_pass in ("clean", "final")
        for path in (sintel_root / "training" / render_pass).rglob("*.png")
    }
    if len(expected_keys) != EXPECTED_SINTEL_FRAMES:
        raise ValueError(
            "Sintel RGB inventory drift: "
            f"{len(expected_keys)} != {EXPECTED_SINTEL_FRAMES}"
        )
    record_keys = {str(key) for key in records}
    if record_keys != expected_keys:
        raise ValueError(
            "SAM2.1 records differ from the complete Sintel clean/final inventory"
        )
    key_root = Path(str(manifest.get("key_object_root", ""))).resolve()
    if not key_root.is_dir():
        raise FileNotFoundError(
            "SAM2.1 key_object_root is unavailable; republish the manifest "
            "after moving masks"
        )
    missing_full: list[str] = []
    missing_key: list[str] = []
    key_object_count = 0
    nonempty_frames = 0
    for raw_key, raw_record in records.items():
        key = str(raw_key)
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"invalid SAM2.1 record: {key}")
        source_path = Path(str(raw_record.get("source_path", ""))).resolve()
        if source_path != (sintel_root / key).resolve():
            raise ValueError(f"SAM2.1 source path drift: {key}")
        if not (full_root / key).is_file() and len(missing_full) < 10:
            missing_full.append(key)
        key_path = key_root / Path(key).with_suffix(".npz")
        if not key_path.is_file() and len(missing_key) < 10:
            missing_key.append(key)
        count = raw_record.get("key_object_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"invalid SAM2.1 key_object_count: {key}")
        key_object_count += count
        nonempty_frames += int(count > 0)
    if missing_full or missing_key:
        raise FileNotFoundError(
            "SAM2.1 output is incomplete; "
            f"missing_full={missing_full}, missing_key={missing_key}"
        )
    return full_root, {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "frame_count": len(records),
        "key_object_root": str(key_root),
        "key_object_count": key_object_count,
        "nonempty_frames": nonempty_frames,
        "sam_checkpoint_sha256": SAM21_CHECKPOINT_SHA256,
    }


def _semantic_input_preflight(config: IntegratedTrainerConfigV2) -> dict[str, Any]:
    """Prove that exact plus full-seg fallback masks can fill the cache."""

    from .sam_semantic_augmentation import select_sam_object_masks_v1
    from .u0_sintel_dataset import build_sintel_integrated_dataset_from_config_v2

    policy = config.sam_semantic_augmentation
    if not policy.enabled:
        raise ValueError("SAM2.1 mainline requires semantic augmentation")
    dataset = build_sintel_integrated_dataset_from_config_v2(
        role=config.dataset["fit_split"], **dict(config.dataset["kwargs"]),
    )
    usable = 0
    exact = 0
    fallback = 0
    checked = 0
    for sample in dataset:
        checked += 1
        if bool(sample.get("sam_key_object_present", False)):
            exact += 1
            usable += 1
        elif policy.fallback_to_full_segmentation:
            segment_ids = sample.get("sam_segment_ids")
            if segment_ids is None:
                raise ValueError("SAM semantic preflight is missing segment ids")
            _mask, present = select_sam_object_masks_v1(
                segment_ids[None], policy=policy,
            )
            if bool(present.item()):
                fallback += 1
                usable += 1
        if usable >= policy.cache_size:
            break
    if usable < policy.cache_size:
        raise ValueError(
            "SAM2.1 fit split cannot fill the semantic cache: "
            f"{usable}/{policy.cache_size} usable objects"
        )
    return {
        "state": "passed", "checked_fit_samples": checked,
        "usable_objects": usable, "exact_objects": exact,
        "fallback_objects": fallback, "required_objects": policy.cache_size,
        "fallback_to_full_segmentation": policy.fallback_to_full_segmentation,
    }


def _validate_sintel_root(root: Path) -> None:
    for relative in (
        "training/clean", "training/final", "training/flow",
        "training/invalid", "training/occlusions",
    ):
        target = root / relative
        if not target.is_dir():
            raise FileNotFoundError(target)


def _validate_initialization_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != INITIALIZATION_CHECKPOINT_BYTES:
        raise ValueError("U0ft initialization checkpoint size drift")
    digest = _sha256(path)
    if digest != INITIALIZATION_CHECKPOINT_SHA256:
        raise ValueError("U0ft initialization checkpoint digest drift")
    metrics_path = path.parent / "metrics.jsonl"
    completed = _completed_metrics(metrics_path)
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "checkpoint_bytes": path.stat().st_size,
        "metrics": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "completed": completed,
    }


def _validate_torch_home(path: Path) -> dict[str, Any]:
    checkpoint = path / "hub/checkpoints/resnet34-b627a593.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "pinned ResNet34 initialization is missing from TORCH_HOME: "
            f"{checkpoint}"
        )
    digest = _sha256(checkpoint)
    if digest != RESNET34_CHECKPOINT_SHA256:
        raise ValueError("pinned ResNet34 initialization digest drift")
    return {"torch_home": str(path), "resnet34_sha256": digest}


def prepare_portable_sam21_training(
    *,
    repository_root: Path,
    sintel_root: Path,
    sam_mask_root: Path,
    sea_raft_root: Path,
    sea_raft_checkpoint: Path,
    initialization_checkpoint: Path,
    torch_home: Path,
    run_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    repository_root = repository_root.expanduser().resolve()
    sintel_root = sintel_root.expanduser().resolve()
    sea_raft_root = sea_raft_root.expanduser().resolve()
    sea_raft_checkpoint = sea_raft_checkpoint.expanduser().resolve()
    initialization_checkpoint = initialization_checkpoint.expanduser().resolve()
    torch_home = torch_home.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    if run_dir == sintel_root or sintel_root in run_dir.parents:
        raise ValueError("run directory must not be inside the Sintel dataset")
    _validate_sintel_root(sintel_root)
    full_root, sam_report = validate_sam21_masks(
        sam_mask_root=sam_mask_root, sintel_root=sintel_root,
    )
    sea_config = sea_raft_root / "config/eval/spring-M.json"
    sea_report = verify_pinned_sea_raft_identity(
        vendor_root=sea_raft_root,
        config_path=sea_config,
        checkpoint=sea_raft_checkpoint,
    )
    initialization_report = _validate_initialization_checkpoint(
        initialization_checkpoint,
    )
    torch_report = _validate_torch_home(torch_home)

    data_template_path = (
        repository_root / "configs/stablebridge/u0_sintel_u2flow_data_v1.json"
    )
    training_template_path = repository_root / (
        "configs/stablebridge/"
        "u2_sintel_searaft_action_bank_joint_sam21_nohg_v2.json"
    )
    data_config = _json(data_template_path)
    data_config["root"] = str(sintel_root)
    launch_root = run_dir / "portable_launch"
    data_config_path = launch_root / "sintel_data.json"
    _atomic_stable_json(data_config_path, data_config)

    training_config = _json(training_template_path)
    training_config["run_dir"] = str(run_dir)
    dataset_kwargs = training_config["dataset"]["kwargs"]
    dataset_kwargs["config_path"] = str(data_config_path)
    dataset_kwargs["sam_full_segmentation_root"] = str(full_root)
    dataset_kwargs["sam_checkpoint_sha256"] = SAM21_CHECKPOINT_SHA256
    model_kwargs = training_config["model"]["kwargs"]
    model_kwargs["vendor_root"] = str(sea_raft_root)
    model_kwargs["config_path"] = str(sea_config)
    model_kwargs["checkpoint"] = str(sea_raft_checkpoint)
    training_config["model_initialization_checkpoint"] = str(
        initialization_checkpoint
    )
    training_config_path = launch_root / "training_config.json"
    _atomic_stable_json(training_config_path, training_config)
    parsed = IntegratedTrainerConfigV2.from_json(training_config_path)
    if parsed.sam_homography.enabled or not parsed.sam_semantic_augmentation.enabled:
        raise ValueError("portable mainline must use SAM semantic augmentation without HG")
    if parsed.u2_schedule is None or parsed.u2_schedule.mode != "joint_decoupled":
        raise ValueError("portable mainline requires the joint U2 schedule")
    if parsed.action_bank.mode != "balanced_batches":
        raise ValueError("portable mainline requires balanced action batches")
    semantic_preflight = _semantic_input_preflight(parsed)

    report = {
        "schema": PORTABLE_SAM21_TRAINING_SCHEMA_V1,
        "repository_root": str(repository_root),
        "run_dir": str(run_dir),
        "training_config": str(training_config_path),
        "training_config_digest": parsed.digest,
        "sintel_root": str(sintel_root),
        "sam21": sam_report,
        "sam_semantic_preflight": semantic_preflight,
        "sea_raft": sea_report,
        "initialization": initialization_report,
        "torch": torch_report,
        "training_contract": {
            "u2_schedule": parsed.u2_schedule.mode,
            "action_schedule": parsed.action_bank.mode,
            "sam_semantic_enabled": parsed.sam_semantic_augmentation.enabled,
            "fallback_to_full_segmentation": (
                parsed.sam_semantic_augmentation.fallback_to_full_segmentation
            ),
            "homography_enabled": parsed.sam_homography.enabled,
            "epochs": parsed.training.epochs,
            "batch_size": parsed.training.batch_size,
            "gradient_accumulation_steps": (
                parsed.training.gradient_accumulation_steps
            ),
        },
    }
    _atomic_stable_json(launch_root / "preflight.json", report)
    return training_config_path, report


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sintel-root", type=Path, required=True)
    parser.add_argument("--sam-mask-root", type=Path, required=True)
    parser.add_argument("--sea-raft-root", type=Path, required=True)
    parser.add_argument("--sea-raft-checkpoint", type=Path, required=True)
    parser.add_argument("--initialization-checkpoint", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume", default="auto",
        help="auto, none, or an exact checkpoint path",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--reserve-vram", action=argparse.BooleanOptionalAction, default=None,
        help="runtime-only CUDA cache reservation override",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    repository_root = Path(__file__).resolve().parents[3]
    os.environ["TORCH_HOME"] = str(args.torch_home.expanduser().resolve())
    config_path, report = prepare_portable_sam21_training(
        repository_root=repository_root,
        sintel_root=args.sintel_root,
        sam_mask_root=args.sam_mask_root,
        sea_raft_root=args.sea_raft_root,
        sea_raft_checkpoint=args.sea_raft_checkpoint,
        initialization_checkpoint=args.initialization_checkpoint,
        torch_home=args.torch_home,
        run_dir=args.run_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if args.prepare_only:
        return 0
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    config = IntegratedTrainerConfigV2.from_json(config_path)
    memory_override = None
    if args.reserve_vram is not None:
        default = CudaMemoryReservationPolicyV1()
        memory_override = CudaMemoryReservationPolicyV1(
            **{**asdict(default), "enabled": args.reserve_vram},
        )
    trainer = IntegratedUncertaintyTrainerV2(
        config, device=args.device,
        cuda_memory_reservation_override=memory_override,
    )
    if args.resume == "auto":
        if trainer.latest_path.is_file():
            trainer.resume(trainer.latest_path)
    elif args.resume != "none":
        trainer.resume(Path(args.resume).expanduser().resolve())
    elif trainer.metrics_path.exists() or trainer.latest_path.exists():
        raise ValueError("--resume none requires a fresh training run directory")
    return trainer.train()


__all__ = [
    "EXPECTED_MINIMUM_KEY_OBJECTS",
    "EXPECTED_SINTEL_FRAMES",
    "INITIALIZATION_CHECKPOINT_BYTES",
    "INITIALIZATION_CHECKPOINT_SHA256",
    "PORTABLE_SAM21_TRAINING_SCHEMA_V1",
    "RESNET34_CHECKPOINT_SHA256",
    "SAM21_CHECKPOINT_SHA256",
    "main",
    "prepare_portable_sam21_training",
    "validate_sam21_masks",
]
