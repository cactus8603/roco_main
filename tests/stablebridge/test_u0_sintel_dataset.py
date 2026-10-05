from __future__ import annotations

from pathlib import Path
import shutil
import struct
import uuid

import numpy as np
from PIL import Image
import pytest
import torch

from stablebridge.physical_repair.u0_sintel_dataset import (
    SintelU0UncertaintyDatasetV1,
    build_sintel_integrated_dataset_from_config_v2,
    collate_sintel_u0_samples_v1,
    make_sintel_u0_dataloader_v1,
    photometric_only_u0_profile_v1,
    read_sintel_flow_v1,
)
from stablebridge.physical_repair.u2flow_augmentations import (
    U2FlowAugmentationProfileV1,
)
from stablebridge.physical_repair.u2flow_training_data import (
    build_u2flow_training_manifest_v1,
)


@pytest.fixture
def sintel_root():
    repository = Path(__file__).resolve().parents[2]
    path = repository / ".runtime_tmp" / f"u0-sintel-dataset-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _splits() -> dict[str, list[str]]:
    return {
        "fit": ["fit_scene"],
        "validation": ["validation_scene"],
        "calibration": ["calibration_scene"],
        "evaluation": ["evaluation_scene"],
    }


def _write_flo(path: Path, flow: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = flow.shape
    assert channels == 2
    with path.open("wb") as stream:
        stream.write(struct.pack("<fii", 202021.25, width, height))
        stream.write(np.asarray(flow, dtype="<f4").tobytes())


def _populate(root: Path) -> None:
    height, width = 8, 12
    for pass_index, render_pass in enumerate(("clean", "final")):
        for scene_index, scene in enumerate(sum(_splits().values(), [])):
            for frame in (1, 2):
                yy, xx = np.indices((height, width))
                rgb = np.stack((
                    (xx + frame * 10) % 255,
                    (yy + scene_index * 20) % 255,
                    np.full_like(xx, pass_index * 100),
                ), axis=-1).astype(np.uint8)
                path = root / "training" / render_pass / scene / f"frame_{frame:04d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb, mode="RGB").save(path)
    for scene_index, scene in enumerate(sum(_splits().values(), [])):
        flow = np.zeros((height, width, 2), dtype=np.float32)
        flow[..., 0] = scene_index + 1
        flow[..., 1] = -2
        _write_flo(root / "training" / "flow" / scene / "frame_0001.flo", flow)
        invalid = np.zeros((height, width), dtype=np.uint8)
        invalid[2, 3] = 255
        occlusion = np.zeros((height, width), dtype=np.uint8)
        occlusion[4, 5] = 255
        for folder, value in (("invalid", invalid), ("occlusions", occlusion)):
            path = root / "training" / folder / scene / "frame_0001.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(value, mode="L").save(path)


def _profile() -> U2FlowAugmentationProfileV1:
    return U2FlowAugmentationProfileV1(
        profile_id="synthetic-u0-v1",
        crop_hw=(6, 8),
        crop_provenance="synthetic-test",
        horizontal_flip_probability=1.0,
        vertical_flip_probability=1.0,
        swap_probability=1.0,
        affine_probability=1.0,
        brightness_range=(0.5, 1.5),
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        gaussian_blur_probability=0.5,
        last_frame_erasing_probability=0.5,
    )


def _manifest(root: Path):
    profile = _profile()
    config = {
        "manifest_id": "synthetic-u0-sintel-v1",
        "stage": "finetune",
        "root": str(root),
        "splits": _splits(),
        "crop_recipe": {"height": 6, "width": 8},
        "augmentation_recipe": {
            "profile": profile.as_dict(),
            "profile_hash": profile.profile_hash,
        },
    }
    return build_u2flow_training_manifest_v1(config)


def test_photometric_profile_disables_unsupported_geometry():
    safe = photometric_only_u0_profile_v1(_profile())
    assert safe.horizontal_flip_probability == 0
    assert safe.vertical_flip_probability == 0
    assert safe.swap_probability == 0
    assert safe.affine_probability == 0
    assert safe.brightness_range == (0.5, 1.5)
    assert safe.crop_hw == (6, 8)


def test_fit_is_aligned_deterministic_and_never_opens_ground_truth(sintel_root, monkeypatch):
    _populate(sintel_root)
    dataset = SintelU0UncertaintyDatasetV1(
        _manifest(sintel_root), role="fit", fit_profile=_profile(), master_seed=17
    )
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".flo" or "invalid" in path.parts or "occlusions" in path.parts:
            raise AssertionError("fit attempted to open Sintel ground truth")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    first = dataset[0]
    replay = dataset[0]
    assert first["ground_truth_flow"] is None
    assert first["ground_truth_valid"] is None
    assert first["native_frames"].shape == (2, 3, 6, 8)
    assert first["augmented_frames"].shape == (2, 3, 6, 8)
    assert first["augmentation_valid"].all()
    assert first["recipe_hash"] == replay["recipe_hash"]
    assert torch.equal(first["native_frames"], replay["native_frames"])
    assert torch.equal(first["augmented_frames"], replay["augmented_frames"])

    dataset.set_epoch(1)
    changed = dataset[0]
    assert changed["recipe_hash"] != first["recipe_hash"]


def test_fit_rejects_ground_truth_request(sintel_root):
    _populate(sintel_root)
    with pytest.raises(ValueError, match="fit role must not read"):
        SintelU0UncertaintyDatasetV1(
            _manifest(sintel_root),
            role="fit",
            fit_profile=_profile(),
            master_seed=17,
            include_ground_truth=True,
        )


def test_u1_explicitly_opts_into_fit_ground_truth(sintel_root):
    _populate(sintel_root)
    dataset = SintelU0UncertaintyDatasetV1(
        _manifest(sintel_root),
        role="fit",
        fit_profile=_profile(),
        master_seed=17,
        include_ground_truth=True,
        allow_fit_ground_truth=True,
    )
    sample = dataset[0]
    assert sample["ground_truth_flow"].shape == (2, 6, 8)
    assert sample["ground_truth_valid"].dtype is torch.bool


def test_integrated_dataset_emits_exact_spatial_transport_metadata(sintel_root, tmp_path):
    _populate(sintel_root)
    profile = _profile()
    config = {
        "manifest_id": "synthetic-integrated-sintel-v2",
        "stage": "finetune",
        "root": str(sintel_root),
        "base_frame_count": 2,
        "fusion_context_frames": 3,
        "fusion_context_enabled": False,
        "splits": _splits(),
        "crop_recipe": {"height": 6, "width": 8},
        "augmentation_recipe": {
            "master_seed": 17,
            "profile": profile.as_dict(),
            "profile_hash": profile.profile_hash,
        },
    }
    path = tmp_path / "integrated.json"
    path.write_text(__import__("json").dumps(config), encoding="utf-8")
    sample = build_sintel_integrated_dataset_from_config_v2(
        path, role="fit",
    )[0]
    forward = sample["native_to_augmented"].double().numpy()
    inverse = sample["augmented_to_native"].double().numpy()
    np.testing.assert_allclose(forward @ inverse, np.eye(3), atol=1e-5)
    assert sample["swap_endpoints"] is True
    assert not sample["augmentation_valid"].all()
    # Native GT support is independent of the augmented affine support.
    assert int(sample["ground_truth_valid"].sum()) == 47


def test_held_out_reads_flow_masks_and_is_epoch_invariant(sintel_root):
    _populate(sintel_root)
    dataset = SintelU0UncertaintyDatasetV1(
        _manifest(sintel_root),
        role="validation",
        fit_profile=_profile(),
        master_seed=17,
    )
    first = dataset[0]
    dataset.set_epoch(99)
    replay = dataset[0]
    assert first["recipe_hash"] == replay["recipe_hash"]
    assert not torch.equal(first["native_frames"], first["augmented_frames"])
    assert first["ground_truth_flow"].shape == (2, 6, 8)
    assert first["ground_truth_valid"].shape == (1, 6, 8)
    assert first["invalid_mask"].shape == (1, 6, 8)
    assert first["occlusion_mask"].shape == (1, 6, 8)
    assert torch.all(first["ground_truth_flow"][0] == 2)
    assert torch.all(first["ground_truth_flow"][1] == -2)
    assert torch.equal(first["ground_truth_valid"], ~first["invalid_mask"])

    calibration = SintelU0UncertaintyDatasetV1(
        _manifest(sintel_root),
        role="calibration",
        fit_profile=_profile(),
        master_seed=17,
    )[0]
    assert torch.equal(calibration["native_frames"], calibration["augmented_frames"])


def test_loader_collates_optional_ground_truth(sintel_root):
    _populate(sintel_root)
    manifest = _manifest(sintel_root)
    fit = SintelU0UncertaintyDatasetV1(
        manifest, role="fit", fit_profile=_profile(), master_seed=4
    )
    fit_batch = next(iter(make_sintel_u0_dataloader_v1(
        fit, batch_size=2, num_workers=0, shuffle=False, pin_memory=False
    )))
    assert fit_batch["native_frames"].shape == (2, 2, 3, 6, 8)
    assert fit_batch["ground_truth_flow"] is None

    held_out = SintelU0UncertaintyDatasetV1(
        manifest, role="calibration", fit_profile=_profile(), master_seed=4
    )
    held_out_batch = collate_sintel_u0_samples_v1([held_out[0]])
    assert held_out_batch["ground_truth_flow"].shape == (1, 2, 6, 8)
    assert held_out_batch["ground_truth_valid"].dtype is torch.bool


def test_flo_reader_rejects_truncated_payload(sintel_root):
    path = sintel_root / "bad.flo"
    path.write_bytes(struct.pack("<fii", 202021.25, 2, 2) + b"\0" * 4)
    with pytest.raises(ValueError, match="payload size mismatch"):
        read_sintel_flow_v1(path)
