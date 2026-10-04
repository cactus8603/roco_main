from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image
import pytest

from stablebridge.physical_repair.u2flow_training_data import (
    U2FlowSceneRoleV1,
    U2FlowTrainingStageV1,
    build_u2flow_training_manifest_v1,
)


@pytest.fixture
def allowed_root():
    repository = Path(__file__).resolve().parents[2]
    path = repository / ".runtime_tmp" / f"u2flow-data-test-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_rgb(path: Path, value: int, *, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (8, 12, 3) if mode == "RGB" else (8, 12)
    Image.fromarray(np.full(shape, value, np.uint8), mode=mode).save(path)


def splits() -> dict[str, list[str]]:
    return {
        "fit": ["alley_1"],
        "validation": ["ambush_2"],
        "calibration": ["bamboo_1"],
        "evaluation": ["cave_2"],
    }


def config(root: Path, stage: str, **updates) -> dict[str, object]:
    value = {
        "manifest_id": f"synthetic-sintel-{stage}-v1",
        "stage": stage,
        "root": str(root),
        "splits": splits(),
        "crop_recipe": {"kind": "random_crop", "height": 368, "width": 768},
        "augmentation_recipe": {"kind": "u2flow_photometric_v1", "seed": 17},
    }
    value.update(updates)
    return value


def populate_raw(root: Path, frame_count: int = 4) -> None:
    for scene_index, scene in enumerate(sum(splits().values(), [])):
        for frame in range(1, frame_count + 1):
            write_rgb(
                root / "scene" / scene / f"frame_{frame:04d}.png",
                scene_index * 10 + frame,
            )


def populate_finetune(root: Path, frame_count: int = 3) -> None:
    for pass_index, render_pass in enumerate(("clean", "final")):
        for scene_index, scene in enumerate(sum(splits().values(), [])):
            for frame in range(1, frame_count + 1):
                write_rgb(
                    root / "training" / render_pass / scene / f"frame_{frame:04d}.png",
                    pass_index * 50 + scene_index * 10 + frame,
                )


def test_raw_inventory_uses_every_consecutive_pair_and_scene_groups(allowed_root):
    populate_raw(allowed_root, frame_count=4)
    manifest = build_u2flow_training_manifest_v1(config(allowed_root, "raw"))
    assert manifest.stage is U2FlowTrainingStageV1.RAW
    assert manifest.base_frame_count == 2
    assert manifest.fusion_context_frames == 3
    assert manifest.fusion_context_enabled is False
    assert len(manifest.rows) == 4 * 3
    assert all(row.render_pass == "raw" for row in manifest.rows)
    assert all(row.fusion_context is None for row in manifest.rows)
    assert {
        row.split_role for row in manifest.rows
    } == set(U2FlowSceneRoleV1)
    for row in manifest.rows:
        assert row.group_id == f"scene:{row.scene_id}"
        assert row.base_inputs[1].frame_index == row.base_inputs[0].frame_index + 1
        assert len(row.base_inputs) == 2
        assert len(row.crop_recipe_hash) == len(row.augmentation_recipe_hash) == 64
        assert row.future_action_bank_outcome_namespace.startswith(
            "u2flow-future-outcomes/v1/"
        )
    serialized = manifest.as_dict()
    assert serialized["flow_ground_truth_decoded"] is False
    assert serialized["inputs_contain_action_control_strength_labels"] is False
    assert serialized["future_outcomes_joined"] is False


def test_finetune_inventory_has_clean_and_final_rows_with_same_scene_role(allowed_root):
    populate_finetune(allowed_root, frame_count=3)
    manifest = build_u2flow_training_manifest_v1(config(allowed_root, "finetune"))
    assert manifest.stage is U2FlowTrainingStageV1.FINETUNE
    assert len(manifest.rows) == 4 * 2 * 2
    assert {row.render_pass for row in manifest.rows} == {"clean", "final"}
    roles = {}
    for row in manifest.rows:
        assert roles.setdefault(row.scene_id, row.split_role) is row.split_role
    per_scene_passes = {
        scene: {row.render_pass for row in manifest.rows if row.scene_id == scene}
        for scene in roles
    }
    assert all(value == {"clean", "final"} for value in per_scene_passes.values())


def test_opt_in_fusion_adds_only_optional_third_identity(allowed_root):
    populate_raw(allowed_root, frame_count=3)
    manifest = build_u2flow_training_manifest_v1(config(
        allowed_root, "raw", fusion_context_enabled=True,
    ))
    assert all(len(row.base_inputs) == 2 for row in manifest.rows)
    for scene in splits()["fit"]:
        rows = [row for row in manifest.rows if row.scene_id == scene]
        assert [row.fusion_context.frame_index for row in rows] == [1, 1]
        assert rows[0].fusion_context == rows[0].base_inputs[0]
        assert rows[1].fusion_context.frame_index == rows[1].base_inputs[0].frame_index - 1


def test_two_frame_scene_duplicates_first_frame_for_optional_context(allowed_root):
    populate_raw(allowed_root, frame_count=2)
    manifest = build_u2flow_training_manifest_v1(config(
        allowed_root, "raw", fusion_context_enabled=True,
    ))
    assert len(manifest.rows) == 4
    assert all(row.fusion_context == row.base_inputs[0] for row in manifest.rows)
    assert all(len(row.base_inputs) == 2 for row in manifest.rows)


def test_scene_role_leakage_and_unassigned_disk_scene_fail_closed(allowed_root):
    populate_raw(allowed_root, frame_count=2)
    leaked = config(allowed_root, "raw")
    leaked["splits"]["evaluation"] = ["alley_1"]
    with pytest.raises(ValueError, match="more than one split"):
        build_u2flow_training_manifest_v1(leaked)

    write_rgb(allowed_root / "scene" / "unassigned" / "frame_0001.png", 1)
    write_rgb(allowed_root / "scene" / "unassigned" / "frame_0002.png", 2)
    with pytest.raises(ValueError, match="exactly cover"):
        build_u2flow_training_manifest_v1(config(allowed_root, "raw"))


def test_action_control_strength_labels_and_frame_count_drift_are_rejected(allowed_root):
    populate_raw(allowed_root, frame_count=2)
    for forbidden in ("action_label", "control_id", "strength_target"):
        payload = config(allowed_root, "raw")
        payload[forbidden] = "forbidden"
        with pytest.raises(ValueError, match="forbids action/control/strength"):
            build_u2flow_training_manifest_v1(payload)
    with pytest.raises(ValueError, match="base_frame_count"):
        build_u2flow_training_manifest_v1(config(
            allowed_root, "raw", base_frame_count=3,
        ))
    with pytest.raises(ValueError, match="fusion_context_frames"):
        build_u2flow_training_manifest_v1(config(
            allowed_root, "raw", fusion_context_frames=4,
        ))


def test_inventory_never_reads_flow_ground_truth(allowed_root, monkeypatch):
    populate_finetune(allowed_root, frame_count=2)
    flow = allowed_root / "training" / "flow" / "alley_1" / "frame_0001.flo"
    flow.parent.mkdir(parents=True)
    flow.write_bytes(b"must-not-be-opened")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix in {".flo", ".pfm"}:
            raise AssertionError("flow GT was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    manifest = build_u2flow_training_manifest_v1(config(allowed_root, "finetune"))
    assert manifest.as_dict()["flow_ground_truth_decoded"] is False
    assert all(
        all(identity.path.endswith(".png") for identity in row.base_inputs)
        for row in manifest.rows
    )


def test_rgb_bytes_recipe_and_manifest_drift_change_or_reject_identity(allowed_root):
    populate_raw(allowed_root, frame_count=2)
    first = build_u2flow_training_manifest_v1(config(allowed_root, "raw"))
    target = allowed_root / "scene" / "alley_1" / "frame_0001.png"
    write_rgb(target, 199)
    second = build_u2flow_training_manifest_v1(config(allowed_root, "raw"))
    assert first.manifest_hash != second.manifest_hash
    assert first.rows[0].base_inputs[0].sha256 != second.rows[0].base_inputs[0].sha256

    changed_recipe = build_u2flow_training_manifest_v1(config(
        allowed_root, "raw",
        crop_recipe={"kind": "random_crop", "height": 320, "width": 640},
    ))
    assert changed_recipe.crop_recipe_hash != second.crop_recipe_hash
    with pytest.raises(ValueError, match="manifest hash drift"):
        replace(second, manifest_hash="0" * 64)


def test_non_rgb_missing_frames_and_clean_final_drift_fail_closed(allowed_root):
    populate_raw(allowed_root, frame_count=2)
    write_rgb(
        allowed_root / "scene" / "alley_1" / "frame_0002.png", 3, mode="L",
    )
    with pytest.raises(ValueError, match="RGB PNG"):
        build_u2flow_training_manifest_v1(config(allowed_root, "raw"))

    shutil.rmtree(allowed_root)
    allowed_root.mkdir()
    populate_finetune(allowed_root, frame_count=3)
    (allowed_root / "training" / "final" / "alley_1" / "frame_0002.png").unlink()
    with pytest.raises(ValueError, match="consecutive|temporal geometry drift"):
        build_u2flow_training_manifest_v1(config(allowed_root, "finetune"))


def test_tmp_and_ssd8_roots_are_rejected_before_inventory_access():
    for root in ("/tmp/u2flow-forbidden", "/ssd8/cactus8603/u2flow-forbidden"):
        with pytest.raises(ValueError, match="must not be inside"):
            build_u2flow_training_manifest_v1(config(Path(root), "raw"))
