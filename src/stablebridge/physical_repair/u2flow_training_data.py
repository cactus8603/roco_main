"""Sintel RGB inventory for U²Flow-style uncertainty training.

The inventory unit is every consecutive temporal RGB pair, grouped by scene.
An opt-in three-frame fusion context adds only an auxiliary RGB identity; the
base sample remains exactly two frames.  This module never discovers, opens,
decodes, or records flow ground truth and carries no action/control/strength
label.  Its only future-policy surface is an empty, unjoined outcome namespace
reserved on each row.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence

from PIL import Image


U2FLOW_TRAINING_DATA_SCHEMA_V1 = "stablebridge-u2flow-sintel-rgb-inventory/v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCENE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FRAME = re.compile(r"^(.*?)(\d+)\.png$", re.IGNORECASE)
_FORBIDDEN_FIELD_TOKENS = frozenset({
    "action", "actions", "arm", "arms", "candidate", "candidates",
    "control", "controls", "family", "families", "strength", "strengths",
})
_FORBIDDEN_WRITE_ROOTS = (PurePosixPath("/tmp"), PurePosixPath("/ssd8"))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty canonical text")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain NUL")
    return value


def _digest(value: object, name: str) -> str:
    result = _text(value, name)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _field_tokens(value: str) -> set[str]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}


def _reject_training_labels(value: object, *, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} contains a non-string field name")
            forbidden = _field_tokens(key) & _FORBIDDEN_FIELD_TOKENS
            if forbidden:
                raise ValueError(
                    f"RGB inventory forbids action/control/strength label {location}.{key}"
                )
            _reject_training_labels(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_training_labels(child, location=f"{location}[{index}]")


def _safe_root(value: object) -> Path:
    raw = _text(value, "Sintel root")
    lexical = _safe_absolute_path(raw, name="Sintel root")
    # Check the lexical path before resolution so a direct forbidden path is
    # never touched.  Then reject an allowed-looking symlink into one.
    resolved = Path(str(lexical)).resolve(strict=True)
    _safe_absolute_path(str(resolved), name="resolved Sintel root")
    if not resolved.is_dir():
        raise ValueError("Sintel root must be a directory")
    return resolved


def _safe_absolute_path(value: object, *, name: str) -> PurePosixPath:
    raw = _text(value, name)
    if not PurePosixPath(raw).is_absolute():
        raise ValueError(f"{name} must be absolute")
    lexical_string = os.path.abspath(os.path.normpath(raw))
    lexical = PurePosixPath("/" + lexical_string.lstrip("/"))
    for forbidden in _FORBIDDEN_WRITE_ROOTS:
        if lexical == forbidden or forbidden in lexical.parents:
            raise ValueError(f"{name} must not be inside {forbidden}")
    return lexical


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class U2FlowTrainingStageV1(str, Enum):
    RAW = "raw"
    FINETUNE = "finetune"


class U2FlowSceneRoleV1(str, Enum):
    FIT = "fit"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    EVALUATION = "evaluation"


@dataclass(frozen=True)
class RGBFrameIdentityV1:
    path: str
    render_pass: str
    frame_index: int
    width: int
    height: int
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _safe_absolute_path(self.path, name="RGB path")
        if self.render_pass not in {"raw", "clean", "final"}:
            raise ValueError("unknown Sintel RGB render pass")
        for value, name in (
            (self.frame_index, "frame index"),
            (self.width, "RGB width"),
            (self.height, "RGB height"),
            (self.size_bytes, "RGB size"),
        ):
            _positive_integer(value, name)
        _digest(self.sha256, "RGB SHA-256")

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "render_pass": self.render_pass,
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class U2FlowTrainingRowV1:
    row_id: str
    scene_id: str
    group_id: str
    split_role: U2FlowSceneRoleV1
    stage: U2FlowTrainingStageV1
    render_pass: str
    base_frame_count: int
    base_inputs: tuple[RGBFrameIdentityV1, RGBFrameIdentityV1]
    fusion_context_frames: int
    fusion_context_enabled: bool
    fusion_context: RGBFrameIdentityV1 | None
    crop_recipe_hash: str
    augmentation_recipe_hash: str
    future_action_bank_outcome_namespace: str
    row_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        _text(self.row_id, "training row id")
        if _SCENE.fullmatch(_text(self.scene_id, "scene id")) is None:
            raise ValueError("scene id is not canonical")
        if self.group_id != f"scene:{self.scene_id}":
            raise ValueError("scene group id drift")
        if not isinstance(self.split_role, U2FlowSceneRoleV1):
            raise ValueError("scene split role must be typed")
        if not isinstance(self.stage, U2FlowTrainingStageV1):
            raise ValueError("training stage must be typed")
        expected_passes = {"raw"} if self.stage is U2FlowTrainingStageV1.RAW else {"clean", "final"}
        if self.render_pass not in expected_passes:
            raise ValueError("render pass does not belong to the training stage")
        if self.base_frame_count != 2:
            raise ValueError("U2Flow base frame count must remain two")
        inputs = tuple(self.base_inputs)
        if len(inputs) != 2 or any(not isinstance(item, RGBFrameIdentityV1) for item in inputs):
            raise ValueError("training row needs exactly two typed base RGB identities")
        if inputs[1].frame_index != inputs[0].frame_index + 1:
            raise ValueError("base RGB inputs must be a consecutive temporal pair")
        if any(item.render_pass != self.render_pass for item in inputs):
            raise ValueError("base RGB render-pass drift")
        if (inputs[0].width, inputs[0].height) != (inputs[1].width, inputs[1].height):
            raise ValueError("base RGB dimensions differ")
        if self.fusion_context_frames != 3:
            raise ValueError("optional fusion context must be a three-frame recipe")
        if not isinstance(self.fusion_context_enabled, bool):
            raise ValueError("fusion context enabled must be boolean")
        context = self.fusion_context
        if not self.fusion_context_enabled and context is not None:
            raise ValueError("disabled fusion context must not carry an identity")
        if context is not None:
            if not isinstance(context, RGBFrameIdentityV1):
                raise ValueError("fusion context must be a typed RGB identity")
            if context.render_pass != self.render_pass:
                raise ValueError("fusion context render-pass drift")
            allowed_context_indices = {
                inputs[0].frame_index,
                inputs[0].frame_index - 1,
            }
            if context.frame_index not in allowed_context_indices:
                raise ValueError(
                    "fusion context must be the previous frame or duplicate the first "
                    "frame at a scene boundary"
                )
            if (context.width, context.height) != (inputs[0].width, inputs[0].height):
                raise ValueError("fusion context RGB dimensions differ")
        _digest(self.crop_recipe_hash, "crop recipe hash")
        _digest(self.augmentation_recipe_hash, "augmentation recipe hash")
        _text(self.future_action_bank_outcome_namespace, "future outcome namespace")
        object.__setattr__(self, "base_inputs", inputs)
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.row_hash and self.row_hash != expected:
            raise ValueError("U2Flow training row hash drift")
        object.__setattr__(self, "row_hash", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result = {
            "row_id": self.row_id,
            "scene_id": self.scene_id,
            "group_id": self.group_id,
            "split_role": self.split_role.value,
            "stage": self.stage.value,
            "render_pass": self.render_pass,
            "base_frame_count": self.base_frame_count,
            "base_inputs": [item.as_dict() for item in self.base_inputs],
            "fusion_context_frames": self.fusion_context_frames,
            "fusion_context_enabled": self.fusion_context_enabled,
            "fusion_context": (
                None if self.fusion_context is None else self.fusion_context.as_dict()
            ),
            "crop_recipe_hash": self.crop_recipe_hash,
            "augmentation_recipe_hash": self.augmentation_recipe_hash,
            "future_action_bank_outcome_namespace": (
                self.future_action_bank_outcome_namespace
            ),
            "future_outcome_joined": False,
        }
        if include_hash:
            result["row_hash"] = self.row_hash
        return result


@dataclass(frozen=True)
class U2FlowTrainingManifestV1:
    manifest_id: str
    stage: U2FlowTrainingStageV1
    root: str
    config_hash: str
    crop_recipe_hash: str
    augmentation_recipe_hash: str
    base_frame_count: int
    fusion_context_frames: int
    fusion_context_enabled: bool
    rows: tuple[U2FlowTrainingRowV1, ...]
    schema: str = U2FLOW_TRAINING_DATA_SCHEMA_V1
    manifest_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if self.schema != U2FLOW_TRAINING_DATA_SCHEMA_V1:
            raise ValueError("U2Flow training manifest schema drift")
        _text(self.manifest_id, "manifest id")
        if not isinstance(self.stage, U2FlowTrainingStageV1):
            raise ValueError("training manifest stage must be typed")
        _text(self.root, "training manifest root")
        for value, name in (
            (self.config_hash, "config hash"),
            (self.crop_recipe_hash, "crop recipe hash"),
            (self.augmentation_recipe_hash, "augmentation recipe hash"),
        ):
            _digest(value, name)
        if self.base_frame_count != 2 or self.fusion_context_frames != 3:
            raise ValueError("training frame-count contract drift")
        if not isinstance(self.fusion_context_enabled, bool):
            raise ValueError("fusion context enabled must be boolean")
        rows = tuple(self.rows)
        if not rows or any(not isinstance(row, U2FlowTrainingRowV1) for row in rows):
            raise ValueError("training manifest needs typed rows")
        if len({row.row_id for row in rows}) != len(rows):
            raise ValueError("duplicate training row id")
        if len({row.future_action_bank_outcome_namespace for row in rows}) != len(rows):
            raise ValueError("future outcome namespace collision")
        roles_by_scene: dict[str, U2FlowSceneRoleV1] = {}
        passes_by_scene: dict[str, set[str]] = {}
        for row in rows:
            if (
                row.stage is not self.stage
                or row.base_frame_count != self.base_frame_count
                or row.fusion_context_frames != self.fusion_context_frames
                or row.fusion_context_enabled is not self.fusion_context_enabled
                or row.crop_recipe_hash != self.crop_recipe_hash
                or row.augmentation_recipe_hash != self.augmentation_recipe_hash
            ):
                raise ValueError("training row contract drift")
            previous = roles_by_scene.setdefault(row.scene_id, row.split_role)
            if previous is not row.split_role:
                raise ValueError("scene leaks across split roles")
            passes_by_scene.setdefault(row.scene_id, set()).add(row.render_pass)
        if set(roles_by_scene.values()) != set(U2FlowSceneRoleV1):
            raise ValueError("training manifest needs all four scene roles")
        expected_passes = {"raw"} if self.stage is U2FlowTrainingStageV1.RAW else {"clean", "final"}
        if any(value != expected_passes for value in passes_by_scene.values()):
            raise ValueError("scene render-pass coverage drift")
        object.__setattr__(self, "rows", rows)
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.manifest_hash and self.manifest_hash != expected:
            raise ValueError("U2Flow training manifest hash drift")
        object.__setattr__(self, "manifest_hash", expected)

    @property
    def split_hash(self) -> str:
        return _canonical_sha256({
            role.value: sorted({
                row.scene_id for row in self.rows if row.split_role is role
            })
            for role in U2FlowSceneRoleV1
        })

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result = {
            "schema": self.schema,
            "manifest_id": self.manifest_id,
            "stage": self.stage.value,
            "root": self.root,
            "config_hash": self.config_hash,
            "split_hash": self.split_hash,
            "crop_recipe_hash": self.crop_recipe_hash,
            "augmentation_recipe_hash": self.augmentation_recipe_hash,
            "base_frame_count": self.base_frame_count,
            "fusion_context_frames": self.fusion_context_frames,
            "fusion_context_enabled": self.fusion_context_enabled,
            "flow_ground_truth_decoded": False,
            "inputs_contain_action_control_strength_labels": False,
            "future_outcomes_joined": False,
            "rows": [row.as_dict() for row in self.rows],
        }
        if include_hash:
            result["manifest_hash"] = self.manifest_hash
        return result


def _parse_config(config: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise ValueError("U2Flow training config must be a mapping")
    _reject_training_labels(config)
    allowed = {
        "manifest_id", "stage", "root", "splits", "crop_recipe",
        "augmentation_recipe", "base_frame_count", "fusion_context_frames",
        "fusion_context_enabled",
    }
    required = {
        "manifest_id", "stage", "root", "splits", "crop_recipe",
        "augmentation_recipe",
    }
    observed = set(config)
    if not required.issubset(observed) or not observed.issubset(allowed):
        raise ValueError(
            "U2Flow training config fields drifted; "
            f"missing={sorted(required-observed)}, extra={sorted(observed-allowed)}"
        )
    result = dict(config)
    result.setdefault("base_frame_count", 2)
    result.setdefault("fusion_context_frames", 3)
    result.setdefault("fusion_context_enabled", False)
    _text(result["manifest_id"], "manifest id")
    try:
        result["stage"] = U2FlowTrainingStageV1(result["stage"])
    except (TypeError, ValueError) as exc:
        raise ValueError("stage must be raw or finetune") from exc
    if result["base_frame_count"] != 2:
        raise ValueError("base_frame_count must remain two")
    if result["fusion_context_frames"] != 3:
        raise ValueError("fusion_context_frames must remain three")
    if not isinstance(result["fusion_context_enabled"], bool):
        raise ValueError("fusion_context_enabled must be boolean")
    for name in ("crop_recipe", "augmentation_recipe"):
        if not isinstance(result[name], Mapping) or not result[name]:
            raise ValueError(f"{name} must be a nonempty mapping")
    augmentation = result["augmentation_recipe"]
    if "profile" in augmentation or "profile_hash" in augmentation:
        if "profile" not in augmentation or "profile_hash" not in augmentation:
            raise ValueError("augmentation profile and profile_hash must appear together")
        from stablebridge.physical_repair.u2flow_augmentations import (
            U2FlowAugmentationProfileV1,
        )

        profile_payload = augmentation["profile"]
        if not isinstance(profile_payload, Mapping):
            raise ValueError("augmentation profile must be a mapping")
        profile = U2FlowAugmentationProfileV1(**dict(profile_payload))
        if augmentation["profile_hash"] != profile.profile_hash:
            raise ValueError("configured augmentation profile hash drift")
        crop = result["crop_recipe"]
        if "height" in crop or "width" in crop:
            if (crop.get("height"), crop.get("width")) != profile.crop_hw:
                raise ValueError("crop recipe and augmentation profile dimensions drift")
    splits = result["splits"]
    if not isinstance(splits, Mapping) or set(splits) != {
        role.value for role in U2FlowSceneRoleV1
    }:
        raise ValueError("splits need exact fit/validation/calibration/evaluation roles")
    scenes: list[str] = []
    for role in U2FlowSceneRoleV1:
        assigned = splits[role.value]
        if not isinstance(assigned, (list, tuple)) or not assigned:
            raise ValueError(f"{role.value} needs a nonempty scene sequence")
        for scene in assigned:
            if not isinstance(scene, str) or _SCENE.fullmatch(scene) is None:
                raise ValueError("scene ids must be canonical strings")
        scenes.extend(assigned)
    if len(scenes) != len(set(scenes)):
        raise ValueError("scene appears in more than one split role")
    result["splits"] = {
        role.value: tuple(splits[role.value]) for role in U2FlowSceneRoleV1
    }
    return result


def _frame_identity(path: Path, render_pass: str, root: Path) -> RGBFrameIdentityV1:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("RGB input escaped the approved Sintel root")
    match = _FRAME.fullmatch(resolved.name)
    if match is None:
        raise ValueError(f"Sintel RGB filename has no frame index: {resolved.name}")
    with Image.open(resolved) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("Sintel input must be an RGB PNG")
        width, height = image.size
        image.verify()
    return RGBFrameIdentityV1(
        path=str(resolved),
        render_pass=render_pass,
        frame_index=int(match.group(2)),
        width=width,
        height=height,
        size_bytes=resolved.stat().st_size,
        sha256=_file_sha256(resolved),
    )


def _scene_frames(directory: Path, render_pass: str, root: Path) -> tuple[RGBFrameIdentityV1, ...]:
    resolved_directory = directory.resolve(strict=True)
    if not resolved_directory.is_relative_to(root):
        raise ValueError("Sintel scene directory escaped the approved root")
    if not resolved_directory.is_dir():
        raise ValueError(f"missing Sintel scene directory: {directory}")
    identities = tuple(sorted(
        (
            _frame_identity(path, render_pass, root)
            for path in resolved_directory.glob("*.png")
        ),
        key=lambda item: (item.frame_index, item.path),
    ))
    if len(identities) < 2:
        raise ValueError("Sintel scene needs at least two RGB frames")
    indices = tuple(item.frame_index for item in identities)
    if len(set(indices)) != len(indices):
        raise ValueError("Sintel scene has duplicate frame indices")
    if indices != tuple(range(indices[0], indices[0] + len(indices))):
        raise ValueError("Sintel scene frame indices must be consecutive")
    dimensions = {(item.width, item.height) for item in identities}
    if len(dimensions) != 1:
        raise ValueError("Sintel scene RGB dimensions drift")
    return identities


def _context_for_pair(
    frames: Sequence[RGBFrameIdentityV1], index: int,
) -> RGBFrameIdentityV1:
    # U²Flow fusion forms [previous, current, next].  At a scene boundary
    # the first frame is duplicated, yielding [frame0, frame0, frame1].
    return frames[index - 1] if index > 0 else frames[index]


def build_u2flow_training_manifest_v1(
    config: Mapping[str, object],
) -> U2FlowTrainingManifestV1:
    """Hash RGB inputs from a mapping config without reading flow labels."""

    parsed = _parse_config(config)
    root = _safe_root(parsed["root"])
    stage = parsed["stage"]
    assigned_scenes = {
        scene for scenes in parsed["splits"].values() for scene in scenes
    }
    roles = {
        scene: role
        for role in U2FlowSceneRoleV1
        for scene in parsed["splits"][role.value]
    }
    if stage is U2FlowTrainingStageV1.RAW:
        pass_roots = {"raw": root / "scene"}
    else:
        pass_roots = {
            "clean": root / "training" / "clean",
            "final": root / "training" / "final",
        }
    validated_pass_roots = {}
    for render_pass, render_root in pass_roots.items():
        resolved_render_root = render_root.resolve(strict=True)
        if not resolved_render_root.is_relative_to(root):
            raise ValueError("Sintel render root escaped the approved root")
        if not resolved_render_root.is_dir():
            raise ValueError(f"missing Sintel render root: {render_root}")
        validated_pass_roots[render_pass] = resolved_render_root
        observed = {path.name for path in resolved_render_root.iterdir() if path.is_dir()}
        if observed != assigned_scenes:
            raise ValueError(
                "Sintel scene assignment does not exactly cover render root; "
                f"missing={sorted(observed-assigned_scenes)}, "
                f"unknown={sorted(assigned_scenes-observed)}"
            )

    crop_hash = _canonical_sha256(parsed["crop_recipe"])
    augmentation_hash = _canonical_sha256(parsed["augmentation_recipe"])
    rows: list[U2FlowTrainingRowV1] = []
    geometry_by_scene: dict[str, tuple[tuple[int, int, int], ...]] = {}
    for render_pass, render_root in validated_pass_roots.items():
        for scene in sorted(assigned_scenes):
            frames = _scene_frames(render_root / scene, render_pass, root)
            geometry = tuple(
                (item.frame_index, item.width, item.height) for item in frames
            )
            previous = geometry_by_scene.setdefault(scene, geometry)
            if previous != geometry:
                raise ValueError("clean/final temporal geometry drift")
            for index in range(len(frames) - 1):
                first, second = frames[index], frames[index + 1]
                row_id = (
                    f"{stage.value}:{render_pass}:{scene}:"
                    f"{first.frame_index:04d}-{second.frame_index:04d}"
                )
                context = (
                    _context_for_pair(frames, index)
                    if parsed["fusion_context_enabled"] else None
                )
                rows.append(U2FlowTrainingRowV1(
                    row_id=row_id,
                    scene_id=scene,
                    group_id=f"scene:{scene}",
                    split_role=roles[scene],
                    stage=stage,
                    render_pass=render_pass,
                    base_frame_count=2,
                    base_inputs=(first, second),
                    fusion_context_frames=3,
                    fusion_context_enabled=parsed["fusion_context_enabled"],
                    fusion_context=context,
                    crop_recipe_hash=crop_hash,
                    augmentation_recipe_hash=augmentation_hash,
                    future_action_bank_outcome_namespace=(
                        f"u2flow-future-outcomes/v1/{row_id}"
                    ),
                ))
    canonical_config = {
        **parsed,
        "stage": stage.value,
        "root": str(root),
        "splits": {
            role.value: list(parsed["splits"][role.value])
            for role in U2FlowSceneRoleV1
        },
    }
    return U2FlowTrainingManifestV1(
        manifest_id=parsed["manifest_id"],
        stage=stage,
        root=str(root),
        config_hash=_canonical_sha256(canonical_config),
        crop_recipe_hash=crop_hash,
        augmentation_recipe_hash=augmentation_hash,
        base_frame_count=2,
        fusion_context_frames=3,
        fusion_context_enabled=parsed["fusion_context_enabled"],
        rows=tuple(sorted(rows, key=lambda row: row.row_id)),
    )


__all__ = [
    "RGBFrameIdentityV1",
    "U2FLOW_TRAINING_DATA_SCHEMA_V1",
    "U2FlowSceneRoleV1",
    "U2FlowTrainingManifestV1",
    "U2FlowTrainingRowV1",
    "U2FlowTrainingStageV1",
    "build_u2flow_training_manifest_v1",
]
