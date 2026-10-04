"""Action-free native data planning for the U0 uncertainty observer.

This module prepares immutable input identities before any matcher forward or
label decode.  It deliberately stops before :class:`U0NativeManifestV1`: that
training manifest additionally requires hashes of materialized native flow,
risk maps, and augmentation-probe receipts produced by a later GPU worker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from stablebridge.data import SpringProvider, file_sha256
from stablebridge.physical_repair.uncertainty_training_contracts import (
    U0SplitRoleV1,
    reject_action_control_fields_v1,
)
from stablebridge.util import ROOT, save_json


U0_NATIVE_DATA_PLAN_SCHEMA_V1 = "stablebridge-u0-native-data-plan/v1"
U0_NATIVE_DATA_CONTRACT_V1 = "CSB-U0-NATIVE-FLOW-DATA-v1-20261005"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")
_FRAME = re.compile(r"_(\d{4})\.png$")
_TRANSFORMS = frozenset({"shared_translation", "shared_exposure", "shared_gamma"})


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty canonical text")
    return value


def _name(value: object, name: str) -> str:
    result = _text(value, name)
    if _NAME.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase canonical name")
    return result


def _digest(value: object, name: str) -> str:
    result = _text(value, name)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact_fields(value: Mapping[str, object], expected: set[str], name: str) -> None:
    reject_action_control_fields_v1(value, location=name)
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{name} fields drifted; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )


@dataclass(frozen=True)
class U0ProbeSpecV1:
    """One deterministic native-pair perturbation and inverse geometry rule."""

    probe_id: str
    transform: str
    shift_xy: tuple[int, int] = (0, 0)
    scalar: float = 1.0
    probe_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        _name(self.probe_id, "U0 probe id")
        if self.transform not in _TRANSFORMS:
            raise ValueError("unsupported U0 probe transform")
        shift = tuple(self.shift_xy)
        if len(shift) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in shift):
            raise ValueError("U0 probe shift must contain two integers")
        scalar = _finite(self.scalar, "U0 probe scalar")
        if self.transform == "shared_translation":
            if shift == (0, 0) or scalar != 1.0:
                raise ValueError("translation probe needs a nonzero shift and unit scalar")
        elif shift != (0, 0):
            raise ValueError("photometric probe cannot change geometry")
        elif self.transform == "shared_exposure" and not 0.0 < scalar <= 2.0:
            raise ValueError("exposure scalar must lie in (0,2]")
        elif self.transform == "shared_gamma" and not 0.0 < scalar <= 3.0:
            raise ValueError("gamma scalar must lie in (0,3]")
        object.__setattr__(self, "shift_xy", shift)
        object.__setattr__(self, "scalar", scalar)
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.probe_hash and self.probe_hash != expected:
            raise ValueError("U0 probe hash drift")
        object.__setattr__(self, "probe_hash", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result = {
            "probe_id": self.probe_id,
            "transform": self.transform,
            "shift_xy": list(self.shift_xy),
            "scalar": self.scalar,
            "flow_restore": "inverse_translation" if self.transform == "shared_translation" else "identity",
            "wrapped_border_masked": self.transform == "shared_translation",
        }
        if include_hash:
            result["probe_hash"] = self.probe_hash
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "U0ProbeSpecV1":
        expected = {
            "probe_id", "transform", "shift_xy", "scalar", "flow_restore",
            "wrapped_border_masked", "probe_hash",
        }
        _exact_fields(value, expected, "U0 probe")
        result = cls(
            probe_id=value["probe_id"],
            transform=value["transform"],
            shift_xy=tuple(value["shift_xy"]),
            scalar=value["scalar"],
            probe_hash=value["probe_hash"],
        )
        if value["flow_restore"] != result.as_dict()["flow_restore"]:
            raise ValueError("U0 probe flow restore drift")
        if value["wrapped_border_masked"] != result.as_dict()["wrapped_border_masked"]:
            raise ValueError("U0 probe support policy drift")
        return result


@dataclass(frozen=True)
class U0NativeDataRowV1:
    row_id: str
    group_id: str
    split_role: U0SplitRoleV1
    dataset_provider_id: str
    dataset_version: str
    task: str
    dataset_split: str
    scene_id: str
    source_frame: int
    target_frame: int
    source_view: str
    target_view: str
    context_hw: tuple[int, int]
    roi_xyhw: tuple[int, int, int, int]
    input_paths: tuple[str, str]
    input_hashes: tuple[str, str]
    input_size_bytes: tuple[int, int]
    label_path: str
    label_present: bool
    probe_hashes: tuple[str, ...]
    row_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        _name(self.row_id, "U0 row id")
        _name(self.group_id, "U0 row group id")
        if not isinstance(self.split_role, U0SplitRoleV1):
            raise ValueError("U0 row split role must be typed")
        _text(self.dataset_provider_id, "dataset provider id")
        _text(self.dataset_version, "dataset version")
        if self.task != "flow" or self.dataset_split != "train":
            raise ValueError("U0 native data v1 supports Spring training flow only")
        if not re.fullmatch(r"\d{4}", self.scene_id):
            raise ValueError("U0 scene id must have four digits")
        source = _positive_int(self.source_frame, "source frame")
        target = _positive_int(self.target_frame, "target frame")
        if target != source + 1:
            raise ValueError("flow target frame must be source frame plus one")
        if self.source_view != "left" or self.target_view != "left":
            raise ValueError("U0 native flow v1 is left-view only")
        context = tuple(self.context_hw)
        roi = tuple(self.roi_xyhw)
        if len(context) != 2 or any(_positive_int(v, "context extent") % 32 for v in context):
            raise ValueError("U0 context must contain positive multiples of 32")
        if len(roi) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in roi):
            raise ValueError("U0 ROI must contain four integers")
        if roi[2:] != context or min(roi[:2]) < 0:
            raise ValueError("U0 ROI must bind the declared context")
        paths = tuple(self.input_paths)
        hashes = tuple(self.input_hashes)
        sizes = tuple(self.input_size_bytes)
        if len(paths) != 2 or len(hashes) != 2 or len(sizes) != 2:
            raise ValueError("U0 row needs exactly two input identities")
        for value in hashes:
            _digest(value, "U0 input hash")
        if any(_positive_int(value, "U0 input size") <= 0 for value in sizes):
            raise ValueError("U0 input sizes must be positive")
        _text(self.label_path, "U0 label path")
        if self.label_present is not True:
            raise ValueError("U0 plan requires a present but unopened label sidecar")
        probes = tuple(self.probe_hashes)
        if not probes or len(probes) != len(set(probes)):
            raise ValueError("U0 row needs unique probe identities")
        for value in probes:
            _digest(value, "U0 probe hash")
        object.__setattr__(self, "context_hw", context)
        object.__setattr__(self, "roi_xyhw", roi)
        object.__setattr__(self, "input_paths", paths)
        object.__setattr__(self, "input_hashes", hashes)
        object.__setattr__(self, "input_size_bytes", sizes)
        object.__setattr__(self, "probe_hashes", probes)
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.row_hash and self.row_hash != expected:
            raise ValueError("U0 data row hash drift")
        object.__setattr__(self, "row_hash", expected)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result = {
            "row_id": self.row_id,
            "group_id": self.group_id,
            "split_role": self.split_role.value,
            "source_scope": "native_only",
            "dataset_provider_id": self.dataset_provider_id,
            "dataset_version": self.dataset_version,
            "task": self.task,
            "dataset_split": self.dataset_split,
            "scene_id": self.scene_id,
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "source_view": self.source_view,
            "target_view": self.target_view,
            "context_hw": list(self.context_hw),
            "roi_xyhw": list(self.roi_xyhw),
            "input_paths": list(self.input_paths),
            "input_hashes": list(self.input_hashes),
            "input_size_bytes": list(self.input_size_bytes),
            "label_path": self.label_path,
            "label_present": self.label_present,
            "label_opened": False,
            "probe_hashes": list(self.probe_hashes),
        }
        if include_hash:
            result["row_hash"] = self.row_hash
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "U0NativeDataRowV1":
        expected = {
            "row_id", "group_id", "split_role", "source_scope",
            "dataset_provider_id", "dataset_version", "task", "dataset_split",
            "scene_id", "source_frame", "target_frame", "source_view", "target_view",
            "context_hw", "roi_xyhw", "input_paths", "input_hashes",
            "input_size_bytes", "label_path", "label_present", "label_opened",
            "probe_hashes", "row_hash",
        }
        _exact_fields(value, expected, "U0 data row")
        if value["source_scope"] != "native_only" or value["label_opened"] is not False:
            raise ValueError("U0 data row must remain native-only and label-unopened")
        return cls(
            row_id=value["row_id"],
            group_id=value["group_id"],
            split_role=U0SplitRoleV1(value["split_role"]),
            dataset_provider_id=value["dataset_provider_id"],
            dataset_version=value["dataset_version"],
            task=value["task"],
            dataset_split=value["dataset_split"],
            scene_id=value["scene_id"],
            source_frame=value["source_frame"],
            target_frame=value["target_frame"],
            source_view=value["source_view"],
            target_view=value["target_view"],
            context_hw=tuple(value["context_hw"]),
            roi_xyhw=tuple(value["roi_xyhw"]),
            input_paths=tuple(value["input_paths"]),
            input_hashes=tuple(value["input_hashes"]),
            input_size_bytes=tuple(value["input_size_bytes"]),
            label_path=value["label_path"],
            label_present=value["label_present"],
            probe_hashes=tuple(value["probe_hashes"]),
            row_hash=value["row_hash"],
        )


@dataclass(frozen=True)
class U0NativeDataPlanV1:
    plan_id: str
    split_id: str
    config_hash: str
    registry_hash: str
    dataset_provider_id: str
    dataset_version: str
    matcher_provider_id: str
    matcher_checkpoint_path: str
    matcher_checkpoint_hash: str
    matcher_checkpoint_size_bytes: int
    task: str
    context_hw: tuple[int, int]
    crop_policy: str
    frames_per_scene: int
    probes: tuple[U0ProbeSpecV1, ...]
    rows: tuple[U0NativeDataRowV1, ...]
    scientific_scope: str
    schema: str = U0_NATIVE_DATA_PLAN_SCHEMA_V1
    plan_hash: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if self.schema != U0_NATIVE_DATA_PLAN_SCHEMA_V1:
            raise ValueError("U0 data plan schema drift")
        _name(self.plan_id, "U0 data plan id")
        _name(self.split_id, "U0 data split id")
        _digest(self.config_hash, "U0 data config hash")
        _digest(self.registry_hash, "U0 registry hash")
        _text(self.dataset_provider_id, "dataset provider id")
        _text(self.dataset_version, "dataset version")
        _name(self.matcher_provider_id, "matcher provider id")
        _text(self.matcher_checkpoint_path, "matcher checkpoint path")
        _digest(self.matcher_checkpoint_hash, "matcher checkpoint hash")
        _positive_int(self.matcher_checkpoint_size_bytes, "matcher checkpoint size")
        if self.task != "flow":
            raise ValueError("U0 data plan v1 supports flow only")
        context = tuple(self.context_hw)
        if len(context) != 2 or any(_positive_int(v, "context extent") % 32 for v in context):
            raise ValueError("U0 context must contain positive multiples of 32")
        if self.crop_policy != "central_native_no_resize":
            raise ValueError("U0 data plan requires central native crops")
        count = _positive_int(self.frames_per_scene, "frames per scene")
        probes = tuple(self.probes)
        rows = tuple(self.rows)
        if not probes or any(not isinstance(item, U0ProbeSpecV1) for item in probes):
            raise ValueError("U0 data plan needs typed probes")
        if len({item.probe_id for item in probes}) != len(probes):
            raise ValueError("U0 data plan has duplicate probe ids")
        if not rows or any(not isinstance(item, U0NativeDataRowV1) for item in rows):
            raise ValueError("U0 data plan needs typed rows")
        if len({item.row_id for item in rows}) != len(rows):
            raise ValueError("U0 data plan has duplicate row ids")
        expected_probes = tuple(item.probe_hash for item in probes)
        roles_by_group: dict[str, U0SplitRoleV1] = {}
        counts_by_group: dict[str, int] = {}
        for row in rows:
            if row.dataset_provider_id != self.dataset_provider_id:
                raise ValueError("U0 data row provider drift")
            if row.dataset_version != self.dataset_version or row.context_hw != context:
                raise ValueError("U0 data row version or context drift")
            if row.probe_hashes != expected_probes:
                raise ValueError("U0 data row probe recipe drift")
            previous = roles_by_group.setdefault(row.group_id, row.split_role)
            if previous is not row.split_role:
                raise ValueError("U0 scene group leaks across split roles")
            counts_by_group[row.group_id] = counts_by_group.get(row.group_id, 0) + 1
        if set(row.split_role for row in rows) != set(U0SplitRoleV1):
            raise ValueError("U0 data plan needs all four split roles")
        if set(counts_by_group.values()) != {count}:
            raise ValueError("U0 data plan frame count drift across scenes")
        _text(self.scientific_scope, "U0 scientific scope")
        object.__setattr__(self, "context_hw", context)
        object.__setattr__(self, "frames_per_scene", count)
        object.__setattr__(self, "probes", probes)
        object.__setattr__(self, "rows", rows)
        expected = _canonical_sha256(self.as_dict(include_hash=False))
        if self.plan_hash and self.plan_hash != expected:
            raise ValueError("U0 data plan hash drift")
        object.__setattr__(self, "plan_hash", expected)
        reject_action_control_fields_v1(self.as_dict(), location="U0 data plan")

    @property
    def probe_recipe_hash(self) -> str:
        return _canonical_sha256([item.as_dict() for item in self.probes])

    @property
    def split_hash(self) -> str:
        return _canonical_sha256({
            role.value: [
                {"row_id": row.row_id, "group_id": row.group_id, "row_hash": row.row_hash}
                for row in sorted(self.rows, key=lambda item: item.row_id)
                if row.split_role is role
            ]
            for role in U0SplitRoleV1
        })

    def as_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        role_counts = {
            role.value: sum(row.split_role is role for row in self.rows)
            for role in U0SplitRoleV1
        }
        group_counts = {
            role.value: len({row.group_id for row in self.rows if row.split_role is role})
            for role in U0SplitRoleV1
        }
        result = {
            "schema": self.schema,
            "plan_id": self.plan_id,
            "source_scope": "native_only",
            "labels_opened": False,
            "matcher_outputs_materialized": False,
            "training_started": False,
            "split_id": self.split_id,
            "split_hash": self.split_hash,
            "config_hash": self.config_hash,
            "registry_hash": self.registry_hash,
            "dataset_provider_id": self.dataset_provider_id,
            "dataset_version": self.dataset_version,
            "matcher_provider_id": self.matcher_provider_id,
            "matcher_checkpoint_path": self.matcher_checkpoint_path,
            "matcher_checkpoint_hash": self.matcher_checkpoint_hash,
            "matcher_checkpoint_size_bytes": self.matcher_checkpoint_size_bytes,
            "task": self.task,
            "context_hw": list(self.context_hw),
            "crop_policy": self.crop_policy,
            "frames_per_scene": self.frames_per_scene,
            "probe_recipe_hash": self.probe_recipe_hash,
            "probes": [item.as_dict() for item in self.probes],
            "role_row_counts": role_counts,
            "role_group_counts": group_counts,
            "rows": [item.as_dict() for item in sorted(self.rows, key=lambda item: item.row_id)],
            "scientific_scope": self.scientific_scope,
        }
        if include_hash:
            result["plan_hash"] = self.plan_hash
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "U0NativeDataPlanV1":
        expected = {
            "schema", "plan_id", "source_scope", "labels_opened",
            "matcher_outputs_materialized", "training_started", "split_id",
            "split_hash", "config_hash", "registry_hash", "dataset_provider_id",
            "dataset_version", "matcher_provider_id", "matcher_checkpoint_path",
            "matcher_checkpoint_hash", "matcher_checkpoint_size_bytes", "task",
            "context_hw", "crop_policy", "frames_per_scene", "probe_recipe_hash",
            "probes", "role_row_counts", "role_group_counts", "rows",
            "scientific_scope", "plan_hash",
        }
        _exact_fields(value, expected, "U0 data plan")
        if value["source_scope"] != "native_only":
            raise ValueError("U0 data plan source scope must be native_only")
        for field_name in ("labels_opened", "matcher_outputs_materialized", "training_started"):
            if value[field_name] is not False:
                raise ValueError(f"U0 data plan {field_name} must remain false")
        probes_value, rows_value = value["probes"], value["rows"]
        if not isinstance(probes_value, list) or not isinstance(rows_value, list):
            raise ValueError("U0 data plan probes and rows must be lists")
        result = cls(
            schema=value["schema"],
            plan_id=value["plan_id"],
            split_id=value["split_id"],
            config_hash=value["config_hash"],
            registry_hash=value["registry_hash"],
            dataset_provider_id=value["dataset_provider_id"],
            dataset_version=value["dataset_version"],
            matcher_provider_id=value["matcher_provider_id"],
            matcher_checkpoint_path=value["matcher_checkpoint_path"],
            matcher_checkpoint_hash=value["matcher_checkpoint_hash"],
            matcher_checkpoint_size_bytes=value["matcher_checkpoint_size_bytes"],
            task=value["task"],
            context_hw=tuple(value["context_hw"]),
            crop_policy=value["crop_policy"],
            frames_per_scene=value["frames_per_scene"],
            probes=tuple(U0ProbeSpecV1.from_dict(item) for item in probes_value),
            rows=tuple(U0NativeDataRowV1.from_dict(item) for item in rows_value),
            scientific_scope=value["scientific_scope"],
            plan_hash=value["plan_hash"],
        )
        observed = result.as_dict()
        for name in ("split_hash", "probe_recipe_hash", "role_row_counts", "role_group_counts"):
            if value[name] != observed[name]:
                raise ValueError(f"U0 data plan derived {name} drift")
        return result


def evenly_spaced_frames_v1(frames: Sequence[int], count: int) -> tuple[int, ...]:
    """Choose a deterministic, endpoint-inclusive subset without data values."""

    count = _positive_int(count, "frame count")
    available = tuple(sorted(set(int(value) for value in frames)))
    if len(available) < count:
        raise ValueError("scene has fewer valid frames than requested")
    if count == 1:
        return (available[len(available) // 2],)
    indices = [round(index * (len(available) - 1) / (count - 1)) for index in range(count)]
    selected = tuple(available[index] for index in indices)
    if len(set(selected)) != count:
        raise RuntimeError("even frame selection produced duplicates")
    return selected


def _parse_probe(value: Mapping[str, object]) -> U0ProbeSpecV1:
    _exact_fields(value, {"probe_id", "transform", "shift_xy", "scalar"}, "U0 probe config")
    return U0ProbeSpecV1(
        probe_id=value["probe_id"], transform=value["transform"],
        shift_xy=tuple(value["shift_xy"]), scalar=value["scalar"],
    )


def _load_config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    expected = {
        "contract", "plan_id", "registry", "dataset_key", "model_profile",
        "matcher_provider_id", "task", "dataset_split", "view", "context_hw",
        "crop_policy", "frames_per_scene", "split_id", "splits", "probes",
        "scientific_scope",
    }
    _exact_fields(value, expected, "U0 data config")
    if value["contract"] != U0_NATIVE_DATA_CONTRACT_V1:
        raise ValueError("unexpected U0 native data contract")
    if value["dataset_key"] != "spring" or value["model_profile"] != "main":
        raise ValueError("U0 native flow v1 requires Spring and the main matcher")
    if value["task"] != "flow" or value["dataset_split"] != "train" or value["view"] != "left":
        raise ValueError("U0 native data v1 is train-split left-view flow only")
    splits = value["splits"]
    if not isinstance(splits, Mapping) or set(splits) != {role.value for role in U0SplitRoleV1}:
        raise ValueError("U0 data config needs exact fit/validation/calibration/evaluation splits")
    scenes = []
    for role in U0SplitRoleV1:
        assigned = splits[role.value]
        if not isinstance(assigned, list) or not assigned:
            raise ValueError(f"U0 {role.value} split must be a nonempty scene list")
        for scene in assigned:
            if not isinstance(scene, str) or re.fullmatch(r"\d{4}", scene) is None:
                raise ValueError("U0 scene ids must have four digits")
        scenes.extend(assigned)
    if len(scenes) != len(set(scenes)):
        raise ValueError("U0 scene appears in more than one split")
    probes = value["probes"]
    if not isinstance(probes, list) or not probes:
        raise ValueError("U0 data config needs probe definitions")
    return value


def _resolve(root: Path, path: object) -> Path:
    if not isinstance(path, (str, Path)):
        raise ValueError("path must be text or Path")
    raw = str(path)
    if not raw or raw != raw.strip():
        raise ValueError("path must be nonempty canonical text")
    result = Path(raw)
    return result if result.is_absolute() else root / result


def _valid_flow_frames(provider: SpringProvider, scene: str, view: str) -> tuple[int, ...]:
    directory = provider.rgb_root / "train" / scene / f"frame_{view}"
    frames = []
    for path in directory.glob(f"frame_{view}_*.png"):
        match = _FRAME.search(path.name)
        if match is None:
            continue
        frame = int(match.group(1))
        if (
            path.exists()
            and provider.image_path(scene, frame + 1, view, "train").exists()
            and provider.gt_path("flow", scene, frame, view, "train").exists()
        ):
            frames.append(frame)
    return tuple(sorted(set(frames)))


def build_u0_native_data_plan_v1(
    config_path: str | Path,
    *,
    root: str | Path = ROOT,
) -> U0NativeDataPlanV1:
    """Inventory and hash native RGB endpoints without opening any label file."""

    root_path = Path(root).resolve()
    config_file = _resolve(root_path, config_path).resolve(strict=True)
    config = _load_config(config_file)
    registry_path = _resolve(root_path, config["registry"]).resolve(strict=True)
    registry = json.loads(registry_path.read_text())
    reject_action_control_fields_v1(config, location="U0 data config")
    dataset = registry["datasets"][config["dataset_key"]]
    provider = SpringProvider.from_registry(registry_path)
    dataset_version = dataset.get("dataset_version", provider.dataset_version)
    dataset_provider_id = f"spring.{dataset_version}.train.flow.left"
    model = registry["models"][config["model_profile"]][config["task"]]
    checkpoint_path = Path(model["local_path"]).resolve(strict=True)
    expected_size = int(model.get("metadata", {}).get("size_bytes", checkpoint_path.stat().st_size))
    if checkpoint_path.stat().st_size != expected_size:
        raise ValueError("registered matcher checkpoint size drift")
    checkpoint_hash = _digest(model["sha256"], "registered matcher checkpoint hash")
    if file_sha256(checkpoint_path) != checkpoint_hash:
        raise ValueError("registered matcher checkpoint content drift")
    context = tuple(config["context_hw"])
    if len(context) != 2 or any(_positive_int(value, "context extent") % 32 for value in context):
        raise ValueError("U0 context must contain positive multiples of 32")
    probes = tuple(_parse_probe(item) for item in config["probes"])
    probe_hashes = tuple(item.probe_hash for item in probes)
    rows = []
    input_cache: dict[Path, tuple[str, int, tuple[int, int]]] = {}

    def input_identity(path: Path) -> tuple[str, int, tuple[int, int]]:
        cached = input_cache.get(path)
        if cached is not None:
            return cached
        with Image.open(path) as image:
            dimensions = tuple(image.size)
        result = (file_sha256(path), path.stat().st_size, dimensions)
        input_cache[path] = result
        return result

    for role in U0SplitRoleV1:
        for scene in config["splits"][role.value]:
            frames = evenly_spaced_frames_v1(
                _valid_flow_frames(provider, scene, config["view"]),
                config["frames_per_scene"],
            )
            for frame in frames:
                first = provider.image_path(scene, frame, config["view"], "train").resolve(strict=True)
                second = provider.image_path(scene, frame + 1, config["view"], "train").resolve(strict=True)
                label = provider.gt_path("flow", scene, frame, config["view"], "train").resolve(strict=True)
                first_hash, first_size, first_wh = input_identity(first)
                second_hash, second_size, second_wh = input_identity(second)
                if first_wh != second_wh:
                    raise ValueError("U0 endpoint image dimensions differ")
                full_w, full_h = first_wh
                height, width = context
                if height > full_h or width > full_w:
                    raise ValueError("U0 context exceeds source image")
                x, y = (full_w - width) // 2, (full_h - height) // 2
                row_id = f"flow-{scene}-{frame:04d}-left"
                rows.append(U0NativeDataRowV1(
                    row_id=row_id,
                    group_id=f"scene-{scene}",
                    split_role=role,
                    dataset_provider_id=dataset_provider_id,
                    dataset_version=dataset_version,
                    task="flow",
                    dataset_split="train",
                    scene_id=scene,
                    source_frame=frame,
                    target_frame=frame + 1,
                    source_view="left",
                    target_view="left",
                    context_hw=context,
                    roi_xyhw=(x, y, height, width),
                    input_paths=(str(first), str(second)),
                    input_hashes=(first_hash, second_hash),
                    input_size_bytes=(first_size, second_size),
                    label_path=str(label),
                    label_present=True,
                    probe_hashes=probe_hashes,
                ))
    return U0NativeDataPlanV1(
        plan_id=config["plan_id"],
        split_id=config["split_id"],
        config_hash=file_sha256(config_file),
        registry_hash=file_sha256(registry_path),
        dataset_provider_id=dataset_provider_id,
        dataset_version=dataset_version,
        matcher_provider_id=config["matcher_provider_id"],
        matcher_checkpoint_path=str(checkpoint_path),
        matcher_checkpoint_hash=checkpoint_hash,
        matcher_checkpoint_size_bytes=expected_size,
        task="flow",
        context_hw=context,
        crop_policy=config["crop_policy"],
        frames_per_scene=config["frames_per_scene"],
        probes=probes,
        rows=tuple(rows),
        scientific_scope=config["scientific_scope"],
    )


def materialize_u0_native_data_plan_v1(
    config_path: str | Path,
    output_path: str | Path,
    *,
    root: str | Path = ROOT,
) -> U0NativeDataPlanV1:
    plan = build_u0_native_data_plan_v1(config_path, root=root)
    save_json(output_path, plan.as_dict())
    return plan


def verify_u0_native_data_plan_file_v1(
    path: str | Path,
    *,
    verify_input_bytes: bool = False,
) -> U0NativeDataPlanV1:
    """Reload a plan and optionally re-hash RGB bytes without opening labels."""

    value = json.loads(Path(path).read_text())
    plan = U0NativeDataPlanV1.from_dict(value)
    checkpoint = Path(plan.matcher_checkpoint_path).resolve(strict=True)
    if checkpoint.stat().st_size != plan.matcher_checkpoint_size_bytes:
        raise ValueError("U0 matcher checkpoint size drift")
    for row in plan.rows:
        label = Path(row.label_path)
        if not label.exists():
            raise ValueError("U0 label sidecar disappeared")
        for input_path, expected_hash, expected_size in zip(
            row.input_paths, row.input_hashes, row.input_size_bytes,
        ):
            source = Path(input_path).resolve(strict=True)
            if source.stat().st_size != expected_size:
                raise ValueError("U0 input size drift")
            if verify_input_bytes and file_sha256(source) != expected_hash:
                raise ValueError("U0 input content drift")
    return plan


def apply_u0_probe_v1(
    first: np.ndarray,
    second: np.ndarray,
    probe: U0ProbeSpecV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a frozen probe and return two uint8 images plus valid support."""

    if not isinstance(probe, U0ProbeSpecV1):
        raise ValueError("U0 probe must be typed")
    images = tuple(np.asarray(value) for value in (first, second))
    if any(value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8 for value in images):
        raise ValueError("U0 probe inputs must be uint8 HxWx3")
    if images[0].shape != images[1].shape:
        raise ValueError("U0 probe endpoints must have matching shapes")
    height, width = images[0].shape[:2]
    support = np.ones((height, width), dtype=bool)
    if probe.transform == "shared_translation":
        dx, dy = probe.shift_xy
        if abs(dx) * 2 >= width or abs(dy) * 2 >= height:
            raise ValueError("U0 translation is too large for the image")
        output = tuple(np.roll(value, (dy, dx), axis=(0, 1)).copy() for value in images)
        if dx:
            support[:, :abs(dx)] = False
            support[:, -abs(dx):] = False
        if dy:
            support[:abs(dy)] = False
            support[-abs(dy):] = False
    elif probe.transform == "shared_exposure":
        output = tuple(
            np.rint(np.clip(value.astype(np.float32) * probe.scalar, 0, 255)).astype(np.uint8)
            for value in images
        )
    elif probe.transform == "shared_gamma":
        output = tuple(
            np.rint(np.power(value.astype(np.float32) / 255.0, probe.scalar) * 255.0).astype(np.uint8)
            for value in images
        )
    else:  # pragma: no cover - dataclass validation is exhaustive
        raise ValueError("unsupported U0 probe transform")
    return output[0], output[1], support


def restore_u0_probe_flow_v1(
    displacement: np.ndarray,
    probe: U0ProbeSpecV1,
) -> np.ndarray:
    """Map a probe prediction back onto the native source lattice."""

    value = np.asarray(displacement)
    if value.ndim != 3 or value.shape[0] != 2 or not np.isfinite(value).all():
        raise ValueError("U0 probe flow must be finite 2xHxW")
    if probe.transform != "shared_translation":
        return value.astype(np.float32, copy=True)
    dx, dy = probe.shift_xy
    return np.roll(value, (-dy, -dx), axis=(1, 2)).astype(np.float32, copy=True)


__all__ = [
    "U0_NATIVE_DATA_CONTRACT_V1",
    "U0_NATIVE_DATA_PLAN_SCHEMA_V1",
    "U0NativeDataPlanV1",
    "U0NativeDataRowV1",
    "U0ProbeSpecV1",
    "apply_u0_probe_v1",
    "build_u0_native_data_plan_v1",
    "evenly_spaced_frames_v1",
    "materialize_u0_native_data_plan_v1",
    "restore_u0_probe_flow_v1",
    "verify_u0_native_data_plan_file_v1",
]
