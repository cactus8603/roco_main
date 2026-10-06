from __future__ import annotations

import json
from pathlib import Path

from stablebridge.physical_repair.integrated_uncertainty_trainer import (
    IntegratedTrainerConfigV2,
)
from stablebridge.physical_repair import sam21_training_launcher as launcher


def _sintel_root(tmp_path: Path) -> Path:
    root = tmp_path / "sintel"
    for relative in (
        "training/clean", "training/final", "training/flow",
        "training/invalid", "training/occlusions",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def _sam_output(
    tmp_path: Path, sintel_root: Path, *, key_object_count: int,
) -> Path:
    output = tmp_path / "sam21"
    full_root = output / "full_seg"
    key_root = output / "key_objects"
    records = {}
    for index in range(2):
        key = f"training/clean/scene/frame_{index + 1:04d}.png"
        source_path = sintel_root / key
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(b"rgb")
        full_path = full_root / key
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(b"full")
        key_path = key_root / Path(key).with_suffix(".npz")
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(b"key")
        records[key] = {
            "source_path": str(source_path),
            "key_object_count": key_object_count,
            "sha256": "a" * 64,
            "key_objects_sha256": "b" * 64,
        }
    manifest = {
        "schema": "stablebridge-sam-full-segmentation/v1",
        "extended_schema": "stablebridge-sam21-training-masks/v1",
        "complete": True,
        "dataset": "sintel",
        "sam_checkpoint_sha256": launcher.SAM21_CHECKPOINT_SHA256,
        "sam_model_type": "sam2.1_hiera_large",
        "inventory_frame_count": 2,
        "inventory_truncated": False,
        "selected_frame_count": 2,
        "frame_count": 2,
        "source_root": str(sintel_root),
        "key_object_root": str(key_root),
        "records": records,
    }
    full_root.mkdir(parents=True, exist_ok=True)
    (full_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    return output


def test_mask_preflight_accepts_complete_lineage_with_enough_key_objects(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(launcher, "EXPECTED_SINTEL_FRAMES", 2)
    sintel_root = _sintel_root(tmp_path)
    output = _sam_output(tmp_path, sintel_root, key_object_count=50)
    full_root, report = launcher.validate_sam21_masks(
        sam_mask_root=output, sintel_root=sintel_root,
    )
    assert full_root == (output / "full_seg").resolve()
    assert report["frame_count"] == 2
    assert report["key_object_count"] == 100


def test_mask_preflight_allows_sparse_exact_objects_for_fullseg_fallback(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(launcher, "EXPECTED_SINTEL_FRAMES", 2)
    sintel_root = _sintel_root(tmp_path)
    output = _sam_output(tmp_path, sintel_root, key_object_count=1)
    _root, report = launcher.validate_sam21_masks(
        sam_mask_root=output, sintel_root=sintel_root,
    )
    assert report["key_object_count"] == 2


def test_portable_prepare_rewrites_every_machine_path_and_is_resume_stable(
    tmp_path, monkeypatch,
):
    repository = Path(__file__).resolve().parents[2]
    sintel_root = _sintel_root(tmp_path)
    sam_full_root = tmp_path / "sam21/full_seg"
    sam_full_root.mkdir(parents=True)
    sea_root = tmp_path / "SEA-RAFT"
    sea_root.mkdir()
    sea_checkpoint = tmp_path / "model.safetensors"
    sea_checkpoint.write_bytes(b"sea")
    initialization = tmp_path / "initial/best.pt"
    initialization.parent.mkdir()
    initialization.write_bytes(b"initial")
    torch_home = tmp_path / "torch-home"
    torch_home.mkdir()
    run_dir = tmp_path / "run"

    monkeypatch.setattr(
        launcher, "validate_sam21_masks",
        lambda **_kwargs: (sam_full_root.resolve(), {"frame_count": 2128}),
    )
    monkeypatch.setattr(
        launcher, "verify_pinned_sea_raft_identity",
        lambda **_kwargs: {"vendor_commit": "pinned"},
    )
    monkeypatch.setattr(
        launcher, "_validate_initialization_checkpoint",
        lambda _path: {"checkpoint_sha256": launcher.INITIALIZATION_CHECKPOINT_SHA256},
    )
    monkeypatch.setattr(
        launcher, "_validate_torch_home",
        lambda _path: {"resnet34_sha256": launcher.RESNET34_CHECKPOINT_SHA256},
    )
    monkeypatch.setattr(
        launcher, "_semantic_input_preflight",
        lambda _config: {
            "state": "passed", "usable_objects": 100,
            "fallback_objects": 100,
        },
    )

    arguments = {
        "repository_root": repository,
        "sintel_root": sintel_root,
        "sam_mask_root": sam_full_root,
        "sea_raft_root": sea_root,
        "sea_raft_checkpoint": sea_checkpoint,
        "initialization_checkpoint": initialization,
        "torch_home": torch_home,
        "run_dir": run_dir,
    }
    config_path, first_report = launcher.prepare_portable_sam21_training(
        **arguments,
    )
    second_path, second_report = launcher.prepare_portable_sam21_training(
        **arguments,
    )
    assert second_path == config_path
    assert second_report == first_report
    config = IntegratedTrainerConfigV2.from_json(config_path)
    assert config.run_dir == run_dir.resolve()
    assert config.model_initialization_checkpoint == initialization.resolve()
    assert config.dataset["kwargs"]["sam_full_segmentation_root"] == str(
        sam_full_root.resolve()
    )
    assert config.model["kwargs"] == {
        "variant": "refinement_with_uncertainty",
        "trainable_scope": "all",
        "refinement_channels": 64,
        "maximum_update_px": 1.0,
        "vendor_root": str(sea_root.resolve()),
        "config_path": str(
            (sea_root / "config/eval/spring-M.json").resolve()
        ),
        "checkpoint": str(sea_checkpoint.resolve()),
    }
    assert config.sam_semantic_augmentation.enabled
    assert config.sam_semantic_augmentation.fallback_to_full_segmentation
    assert not config.sam_homography.enabled
    assert first_report["sam_semantic_preflight"]["usable_objects"] == 100
