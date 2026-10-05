"""Train-ready Sintel samples for uncertainty and flow refinement.

The original U0 path remains photometric-only and never opens fit labels.  The
integrated v2 path additionally emits a hash-bound affine mapping from the
native crop to the augmented lattice plus endpoint-order metadata, allowing a
trainer to transport detached teacher flow exactly instead of pretending that
spatial augmentation preserves pixel/vector coordinates.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import struct
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from stablebridge.physical_repair.u2flow_augmentations import (
    U2FlowAugmentationProfileV1,
    apply_u2flow_recipe_v1,
    derive_u2flow_view_seed_v1,
    sample_u2flow_recipe_v1,
    sample_u2flow_role_recipe_v1,
)
from stablebridge.physical_repair.u2flow_training_data import (
    U2FlowSceneRoleV1,
    U2FlowTrainingManifestV1,
    U2FlowTrainingRowV1,
    U2FlowTrainingStageV1,
    build_u2flow_training_manifest_v1,
)


_HELD_OUT_ROLES = frozenset({
    U2FlowSceneRoleV1.VALIDATION,
    U2FlowSceneRoleV1.CALIBRATION,
    U2FlowSceneRoleV1.EVALUATION,
})
_SINTEL_FLO_MAGIC = 202021.25


def photometric_only_u0_profile_v1(
    profile: U2FlowAugmentationProfileV1,
) -> U2FlowAugmentationProfileV1:
    """Disable every v0 transform that changes flow geometry or direction."""

    if not isinstance(profile, U2FlowAugmentationProfileV1):
        raise ValueError("profile must be a typed U2Flow augmentation profile")
    return replace(
        profile,
        profile_id=f"{profile.profile_id}-u0-photometric-only",
        crop_provenance=f"{profile.crop_provenance}-u0-photometric-only",
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        swap_probability=0.0,
        affine_probability=0.0,
        rotation_degrees_range=(0.0, 0.0),
        scale_range=(1.0, 1.0),
        translate_x_fraction_range=(0.0, 0.0),
        translate_y_fraction_range=(0.0, 0.0),
        profile_hash="",
    )


def read_sintel_flow_v1(path: str | Path) -> np.ndarray:
    """Decode one Middlebury/Sintel ``.flo`` file as float32 ``HxWx2``."""

    target = Path(path)
    with target.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError(f"truncated Sintel flow header: {target}")
        magic, width, height = struct.unpack("<fii", header)
        if magic != _SINTEL_FLO_MAGIC:
            raise ValueError(f"invalid Sintel flow magic: {target}")
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid Sintel flow dimensions: {target}")
        expected = width * height * 2
        payload = np.frombuffer(stream.read(), dtype="<f4")
    if payload.size != expected:
        raise ValueError(
            f"Sintel flow payload size mismatch: expected {expected}, "
            f"observed {payload.size}: {target}"
        )
    return payload.reshape(height, width, 2).astype(np.float32, copy=False)


def _read_rgb(path: str | Path, expected_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Sintel frame must remain RGB: {path}")
        result = np.asarray(image, dtype=np.uint8).copy()
    if result.shape != (expected_hw[0], expected_hw[1], 3):
        raise ValueError(f"Sintel RGB geometry drift: {path}")
    return result


def _read_binary_mask(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        result = np.asarray(image.convert("L"), dtype=np.uint8)
    if result.shape != expected_hw:
        raise ValueError(f"Sintel mask geometry drift: {path}")
    return result != 0


def _chw_float(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)


def _chw_bool(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(mask[None])).bool()


def _role(value: U2FlowSceneRoleV1 | str) -> U2FlowSceneRoleV1:
    try:
        return value if isinstance(value, U2FlowSceneRoleV1) else U2FlowSceneRoleV1(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("role must be fit, validation, calibration, or evaluation") from exc


class SintelU0UncertaintyDatasetV1(Dataset[dict[str, object]]):
    """Materialize aligned native/photometric pairs for one scene-disjoint role."""

    def __init__(
        self,
        manifest: U2FlowTrainingManifestV1,
        *,
        role: U2FlowSceneRoleV1 | str,
        fit_profile: U2FlowAugmentationProfileV1,
        master_seed: int,
        include_ground_truth: bool | None = None,
        validation_appearance_probe: bool = True,
        allow_fit_ground_truth: bool = False,
        allow_spatial_augmentation: bool = False,
    ) -> None:
        if not isinstance(manifest, U2FlowTrainingManifestV1):
            raise ValueError("manifest must be a typed U2Flow training manifest")
        selected_role = _role(role)
        if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
            raise ValueError("master_seed must be a nonnegative integer")
        if include_ground_truth is None:
            include_ground_truth = selected_role in _HELD_OUT_ROLES
        if not isinstance(include_ground_truth, bool):
            raise ValueError("include_ground_truth must be bool or None")
        if not isinstance(validation_appearance_probe, bool):
            raise ValueError("validation_appearance_probe must be bool")
        if not isinstance(allow_fit_ground_truth, bool):
            raise ValueError("allow_fit_ground_truth must be bool")
        if not isinstance(allow_spatial_augmentation, bool):
            raise ValueError("allow_spatial_augmentation must be bool")
        if (
            selected_role is U2FlowSceneRoleV1.FIT
            and include_ground_truth
            and not allow_fit_ground_truth
        ):
            raise ValueError("fit role must not read Sintel ground truth")
        if include_ground_truth and manifest.stage is not U2FlowTrainingStageV1.FINETUNE:
            raise ValueError("Sintel ground truth is available only for finetune-stage data")
        rows = tuple(row for row in manifest.rows if row.split_role is selected_role)
        if not rows:
            raise ValueError(f"manifest has no rows for role {selected_role.value}")

        self.manifest = manifest
        self.role = selected_role
        self.fit_profile = (
            fit_profile
            if allow_spatial_augmentation
            else photometric_only_u0_profile_v1(fit_profile)
        )
        self.master_seed = master_seed
        self.include_ground_truth = include_ground_truth
        self.validation_appearance_probe = validation_appearance_probe
        self.allow_fit_ground_truth = allow_fit_ground_truth
        self.allow_spatial_augmentation = allow_spatial_augmentation
        self.rows = rows
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def _ground_truth_paths(self, row: U2FlowTrainingRowV1) -> tuple[Path, Path, Path]:
        first = Path(row.base_inputs[0].path)
        stem = first.stem
        root = Path(self.manifest.root)
        return (
            root / "training" / "flow" / row.scene_id / f"{stem}.flo",
            root / "training" / "invalid" / row.scene_id / f"{stem}.png",
            root / "training" / "occlusions" / row.scene_id / f"{stem}.png",
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        expected_hw = (row.base_inputs[0].height, row.base_inputs[0].width)
        first = _read_rgb(row.base_inputs[0].path, expected_hw)
        second = _read_rgb(row.base_inputs[1].path, expected_hw)
        if (
            self.role is U2FlowSceneRoleV1.VALIDATION
            and self.validation_appearance_probe
        ):
            # Validation needs a non-trivial, but epoch-invariant, consistency
            # target.  Calibration/evaluation remain identity views so their
            # GT metrics characterize the native matcher without probe drift.
            selected_profile = self.fit_profile
            recipe = sample_u2flow_recipe_v1(
                expected_hw,
                seed=derive_u2flow_view_seed_v1(
                    row.row_id, epoch=0, master_seed=self.master_seed
                ),
                profile=selected_profile,
                crop_mode="center",
            )
        else:
            recipe, selected_profile = sample_u2flow_role_recipe_v1(
                expected_hw,
                row_id=row.row_id,
                split_role=self.role.value,
                epoch=self.epoch,
                master_seed=self.master_seed,
                fit_profile=self.fit_profile,
            )
        spatial = recipe.spatial
        if (
            spatial.affine_enabled
            or spatial.horizontal_flip
            or spatial.vertical_flip
            or spatial.swap_endpoints
        ) and not self.allow_spatial_augmentation:
            raise RuntimeError("U0 v0 geometry safety contract drift")
        top, left = spatial.crop_top, spatial.crop_left
        height, width = spatial.crop_hw
        native_first = np.ascontiguousarray(first[top : top + height, left : left + width])
        native_second = np.ascontiguousarray(second[top : top + height, left : left + width])
        augmented = apply_u2flow_recipe_v1(
            first, second, recipe, profile=selected_profile
        )

        erase_mask = np.zeros((height, width), dtype=bool)
        erasing = recipe.last_frame_erasing
        if erasing.enabled:
            erase_mask[
                erasing.top : erasing.top + erasing.height,
                erasing.left : erasing.left + erasing.width,
            ] = True

        flow_tensor: torch.Tensor | None = None
        valid_tensor: torch.Tensor | None = None
        invalid_tensor: torch.Tensor | None = None
        occlusion_tensor: torch.Tensor | None = None
        if self.include_ground_truth:
            flow_path, invalid_path, occlusion_path = self._ground_truth_paths(row)
            for path in (flow_path, invalid_path, occlusion_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            flow = read_sintel_flow_v1(flow_path)
            if flow.shape != (expected_hw[0], expected_hw[1], 2):
                raise ValueError(f"Sintel flow geometry drift: {flow_path}")
            invalid = _read_binary_mask(invalid_path, expected_hw)
            occlusion = _read_binary_mask(occlusion_path, expected_hw)
            flow = np.ascontiguousarray(flow[top : top + height, left : left + width])
            invalid = np.ascontiguousarray(invalid[top : top + height, left : left + width])
            occlusion = np.ascontiguousarray(occlusion[top : top + height, left : left + width])
            finite = np.isfinite(flow).all(axis=-1)
            # This validity belongs to the native crop.  Augmented geometric
            # support is a distinct mask and is applied only after teacher-flow
            # transport by the integrated trainer.
            valid = (~invalid) & finite
            flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1).copy()).float()
            valid_tensor = _chw_bool(valid)
            invalid_tensor = _chw_bool(invalid)
            occlusion_tensor = _chw_bool(occlusion)

        full_to_native = np.asarray(
            ((1.0, 0.0, -float(left)), (0.0, 1.0, -float(top)), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        full_to_augmented = np.asarray(spatial.input_to_output, dtype=np.float64).reshape(3, 3)
        native_to_augmented = full_to_augmented @ np.linalg.inv(full_to_native)
        augmented_to_native = np.linalg.inv(native_to_augmented)

        return {
            "row_id": row.row_id,
            "scene_id": row.scene_id,
            "group_id": row.group_id,
            "split_role": row.split_role.value,
            "render_pass": row.render_pass,
            "native_frames": torch.stack((_chw_float(native_first), _chw_float(native_second))),
            "augmented_frames": torch.stack((_chw_float(augmented.first), _chw_float(augmented.second))),
            "augmentation_valid": _chw_bool(augmented.valid_support),
            "erasure_mask": _chw_bool(erase_mask),
            "ground_truth_flow": flow_tensor,
            "ground_truth_valid": valid_tensor,
            "invalid_mask": invalid_tensor,
            "occlusion_mask": occlusion_tensor,
            "view_seed": recipe.seed,
            "recipe_hash": recipe.recipe_hash,
            "crop_top_left": torch.tensor((top, left), dtype=torch.int64),
            "input_hw": torch.tensor(expected_hw, dtype=torch.int64),
            "native_to_augmented": torch.from_numpy(native_to_augmented).float(),
            "augmented_to_native": torch.from_numpy(augmented_to_native).float(),
            "swap_endpoints": spatial.swap_endpoints,
        }


def collate_sintel_u0_samples_v1(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Collate samples while preserving the all-present/all-absent GT contract."""

    if not samples:
        raise ValueError("cannot collate an empty Sintel batch")
    keys = tuple(samples[0])
    if any(tuple(sample) != keys for sample in samples):
        raise ValueError("Sintel batch sample fields drift")
    optional = {"ground_truth_flow", "ground_truth_valid", "invalid_mask", "occlusion_mask"}
    result: dict[str, object] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        if key in optional:
            present = [value is not None for value in values]
            if any(present) and not all(present):
                raise ValueError("mixed ground-truth availability in one batch")
            result[key] = None if not any(present) else torch.stack(values)  # type: ignore[arg-type]
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)  # type: ignore[arg-type]
        elif isinstance(values[0], int):
            integer_dtype = (
                torch.uint64
                if any(value > torch.iinfo(torch.int64).max for value in values)
                else torch.int64
            )
            result[key] = torch.tensor(values, dtype=integer_dtype)
        else:
            result[key] = tuple(values)
    return result


def make_sintel_u0_dataloader_v1(
    dataset: SintelU0UncertaintyDatasetV1,
    *,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool | None = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a deterministic loader; workers restart each epoch after ``set_epoch``."""

    if not isinstance(dataset, SintelU0UncertaintyDatasetV1):
        raise ValueError("dataset must be a SintelU0UncertaintyDatasetV1")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    selected_shuffle = dataset.role is U2FlowSceneRoleV1.FIT if shuffle is None else shuffle
    if not isinstance(selected_shuffle, bool):
        raise ValueError("shuffle must be bool or None")
    generator = torch.Generator()
    generator.manual_seed(dataset.master_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=selected_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        generator=generator,
        collate_fn=collate_sintel_u0_samples_v1,
    )


def build_sintel_u0_dataset_from_config_v1(
    config_path: str | Path,
    *,
    role: U2FlowSceneRoleV1 | str,
    include_ground_truth: bool | None = None,
    validation_appearance_probe: bool = True,
    allow_fit_ground_truth: bool = False,
    allow_spatial_augmentation: bool = False,
) -> SintelU0UncertaintyDatasetV1:
    """Load the checked JSON config, inventory its RGB data, and bind one role."""

    config = json.loads(Path(config_path).read_text())
    if not isinstance(config, Mapping):
        raise ValueError("Sintel U0 config must contain a JSON object")
    augmentation = config.get("augmentation_recipe")
    if not isinstance(augmentation, Mapping):
        raise ValueError("config needs augmentation_recipe")
    profile_payload = augmentation.get("profile")
    if not isinstance(profile_payload, Mapping):
        raise ValueError("config needs a typed augmentation profile payload")
    profile = U2FlowAugmentationProfileV1(**dict(profile_payload))
    if augmentation.get("profile_hash") != profile.profile_hash:
        raise ValueError("configured augmentation profile hash drift")
    master_seed = augmentation.get("master_seed")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise ValueError("augmentation_recipe.master_seed must be an integer")
    return SintelU0UncertaintyDatasetV1(
        build_u2flow_training_manifest_v1(config),
        role=role,
        fit_profile=profile,
        master_seed=master_seed,
        include_ground_truth=include_ground_truth,
        validation_appearance_probe=validation_appearance_probe,
        allow_fit_ground_truth=allow_fit_ground_truth,
        allow_spatial_augmentation=allow_spatial_augmentation,
    )


def build_sintel_u1_dataset_from_config_v1(
    config_path: str | Path,
    *,
    role: U2FlowSceneRoleV1 | str,
    validation_appearance_probe: bool = False,
) -> SintelU0UncertaintyDatasetV1:
    """Build the supervised U1/U2 view without weakening U0's data boundary.

    U0 continues to reject ground-truth reads from its fit split by default.
    The flow-changing U1/U2 stages have an explicitly different scientific
    role and therefore opt into Sintel flow labels for fit and validation.
    """

    return build_sintel_u0_dataset_from_config_v1(
        config_path,
        role=role,
        include_ground_truth=True,
        validation_appearance_probe=validation_appearance_probe,
        allow_fit_ground_truth=True,
    )


def build_sintel_integrated_dataset_from_config_v2(
    config_path: str | Path,
    *,
    role: U2FlowSceneRoleV1 | str,
    validation_appearance_probe: bool = True,
) -> SintelU0UncertaintyDatasetV1:
    """Build labelled native views plus full spatial consistency metadata.

    Labels provide the task carrier for the SEA-RAFT adaptation experiment;
    uncertainty itself remains supervised only by augmentation consistency.
    """

    return build_sintel_u0_dataset_from_config_v1(
        config_path,
        role=role,
        include_ground_truth=True,
        validation_appearance_probe=validation_appearance_probe,
        allow_fit_ground_truth=True,
        allow_spatial_augmentation=True,
    )


__all__ = [
    "SintelU0UncertaintyDatasetV1",
    "build_sintel_u0_dataset_from_config_v1",
    "build_sintel_u1_dataset_from_config_v1",
    "build_sintel_integrated_dataset_from_config_v2",
    "collate_sintel_u0_samples_v1",
    "make_sintel_u0_dataloader_v1",
    "photometric_only_u0_profile_v1",
    "read_sintel_flow_v1",
]
