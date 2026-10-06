from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import Dataset

from stablebridge.physical_repair.integrated_uncertainty_trainer import (
    INTEGRATED_TRAINER_SCHEMA_V2,
    IntegratedTrainerConfigV2,
    IntegratedUncertaintyTrainerV2,
)
from stablebridge.physical_repair.candidate_action_bank import (
    OPTICAL_FLOW_CAPACITY_BANK_HASH,
    OPTICAL_FLOW_CAPACITY_SOURCE_MANIFEST_SHA256,
    OPTICAL_NATIVE_ACTION_ID,
)
from stablebridge.physical_repair.sam_semantic_augmentation import (
    SamSemanticAugmentationPolicyV1,
)


class TinyIntegratedDataset(Dataset):
    def __init__(
        self, *, split: str, length: int = 3, fusion_context: bool = False,
        sam_context: bool = False, **_kwargs,
    ):
        self.split = split
        self.length = length
        self.fusion_context = fusion_context
        self.sam_context = sam_context
        self.epoch = 0
        self.action_id = OPTICAL_NATIVE_ACTION_ID

    def __len__(self):
        return self.length

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def set_action_id(self, action_id: str):
        self.action_id = action_id

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(index + (100 if self.split == "validation" else 0))
        native = torch.rand(2, 3, 4, 5, generator=generator)
        augmented = (native * 0.9).clamp(0, 1)
        truth = torch.stack((native[0, 0] * 0.1, native[0, 1] * -0.1))
        result = {
            "row_id": f"{self.split}-row-{index}",
            "scene_id": f"scene-{index % 2}",
            "native_frames": native,
            "augmented_frames": augmented,
            "augmentation_valid": torch.ones(1, 4, 5, dtype=torch.bool),
            "ground_truth_flow": truth,
            "ground_truth_valid": torch.ones(1, 4, 5, dtype=torch.bool),
            "native_to_augmented": torch.eye(3),
            "swap_endpoints": False,
            "action_id": self.action_id,
            "matcher_iterations_override": (
                8 if self.action_id.endswith("P4-iters8") else None
            ),
        }
        if self.fusion_context:
            result["fusion_frames"] = torch.cat((native[:1], native), dim=0)
        if self.sam_context:
            result["sam_segment_ids"] = torch.ones(1, 4, 5, dtype=torch.int64)
            result["sam_key_object_mask"] = torch.ones(1, 4, 5, dtype=torch.bool)
            result["sam_key_object_present"] = True
        return result


def make_tiny_integrated_dataset(
    *, split: str, length: int = 3, fusion_context: bool = False,
    sam_context: bool = False, **kwargs,
):
    del kwargs
    return TinyIntegratedDataset(
        split=split, length=length, fusion_context=fusion_context,
        sam_context=sam_context,
    )


def collate_tiny_integrated(samples):
    result = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if torch.is_tensor(values[0]):
            result[key] = torch.stack(values)
        elif isinstance(values[0], (bool, int)):
            result[key] = torch.tensor(values)
        else:
            result[key] = tuple(values)
    return result


class TinyIntegratedModel(nn.Module):
    def __init__(self, *, variant: str):
        super().__init__()
        self.variant = variant
        self.network = nn.Conv2d(6, 3, 1)
        self.recurrent_head = nn.Module()
        self.recurrent_head.uncertainty_head = nn.Conv2d(1, 1, 1)
        self.recurrent_head.refinement_head = nn.Conv2d(3, 2, 1)
        nn.init.zeros_(self.recurrent_head.refinement_head.weight)
        nn.init.zeros_(self.recurrent_head.refinement_head.bias)
        self.refinement_enabled = True

    @contextmanager
    def refinement_disabled(self):
        previous = self.refinement_enabled
        self.refinement_enabled = False
        try:
            yield
        finally:
            self.refinement_enabled = previous

    def forward(self, first, second, *, iters, test_mode):
        del test_mode
        prediction = self.network(torch.cat((first, second), dim=1) / 255.0)
        alpha = self.recurrent_head.uncertainty_head(prediction[:, 2:3])
        flow = prediction[:, :2]
        if self.variant != "head_only" and self.refinement_enabled:
            evidence = alpha.detach() if self.variant == "refinement_with_uncertainty" else torch.zeros_like(alpha)
            flow = flow + self.recurrent_head.refinement_head(torch.cat((flow, evidence), dim=1))
        flows = [flow * (index + 1) / (iters + 1) for index in range(iters + 1)]
        alphas = [alpha for _ in range(iters + 1)]
        return {"flow": flows, "uncertainty_log_variance": alphas, "final": flows[-1]}


def make_tiny_integrated_model(
    *, device: str, variant: str, trainable_scope: str, **_kwargs,
):
    model = TinyIntegratedModel(variant=variant).to(device)
    if trainable_scope != "all":
        model.requires_grad_(False)
    if trainable_scope in {"uncertainty_only", "refinement_heads"}:
        model.recurrent_head.uncertainty_head.requires_grad_(True)
    if trainable_scope in {"refiner_only", "refinement_heads"}:
        model.recurrent_head.refinement_head.requires_grad_(True)
    return model


def _config(run_dir: Path) -> IntegratedTrainerConfigV2:
    return IntegratedTrainerConfigV2.from_mapping({
        "schema": INTEGRATED_TRAINER_SCHEMA_V2,
        "run_dir": str(run_dir),
        "dataset": {
            "factory": f"{__name__}:make_tiny_integrated_dataset",
            "kwargs": {"length": 3},
            "fit_split": "fit",
            "validation_split": "validation",
            "split_argument": "split",
            "collate_fn": f"{__name__}:collate_tiny_integrated",
        },
        "model": {
            "factory": f"{__name__}:make_tiny_integrated_model",
            "kwargs": {
                "variant": "refinement_with_uncertainty",
                "trainable_scope": "all",
            },
        },
        "loss": {
            "task_weight": 1.0,
            "augmentation_weight": 0.02,
            "uncertainty_weight": 0.005,
            "gamma": 0.8,
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "num_workers": 0,
            "gradient_accumulation_steps": 1,
            "checkpoint_every_steps": 1,
            "validation_every_epochs": 1,
            "amp": False,
            "seed": 3,
            "gradient_clip_norm": 1.0,
            "pin_memory": False,
            "augmentation_start_epoch": 0,
        },
        "iters": 1,
        "severe_error_threshold": 0.1,
    })


def test_integrated_trainer_runs_validates_and_resumes(tmp_path):
    config = _config(tmp_path / "run")
    trainer = IntegratedUncertaintyTrainerV2(config, device="cpu")
    before = {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}
    assert trainer.train() == 0
    assert any(not torch.equal(before[name], value) for name, value in trainer.model.state_dict().items())
    assert trainer.latest_path.is_file()
    assert trainer.best_path.is_file()
    rows = [json.loads(line) for line in trainer.metrics_path.read_text().splitlines()]
    validation = next(row for row in rows if row["event"] == "validation_epoch")
    assert set(("epe", "ause", "spearman", "severe_auroc", "calibration_mae")) <= set(validation)
    assert rows[-1]["event"] == "training_completed"

    resumed = IntegratedUncertaintyTrainerV2(config, device="cpu")
    resumed.resume(trainer.latest_path)
    assert resumed.epoch == 1
    assert resumed.global_step == trainer.global_step


def test_resume_loads_checkpoint_on_cpu_for_rng_contract(tmp_path, monkeypatch):
    config = _config(tmp_path / "cpu-map-resume")
    trainer = IntegratedUncertaintyTrainerV2(config, device="cpu")
    trainer.save()
    resumed = IntegratedUncertaintyTrainerV2(config, device="cpu")
    real_load = torch.load
    observed = []

    def recording_load(*args, **kwargs):
        observed.append(kwargs.get("map_location"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(
        "stablebridge.physical_repair.integrated_uncertainty_trainer.torch.load",
        recording_load,
    )
    resumed.resume(trainer.latest_path)
    assert observed == ["cpu"]


def test_sam_homography_requires_traced_dataset_and_runs_in_flow_phase(tmp_path):
    value = _config(tmp_path / "sam-u2").serializable()
    value["sam_homography"] = {
        "enabled": True,
        "weight": 0.1,
        "maximum_regions": 6,
        "uncertainty_variance_threshold": 2.0,
        "minimum_reliable_points": 4,
        "minimum_reliable_fraction": 0.2,
        "ransac_reprojection_threshold": 3.0,
        "minimum_inlier_fraction": 0.5,
        "per_region_loss_cap": 0.5,
    }
    with pytest.raises(ValueError, match="traced full-segmentation"):
        IntegratedTrainerConfigV2.from_mapping(value)
    value["dataset"]["kwargs"].update({
        "sam_context": True,
        "sam_full_segmentation_root": "/not-read-by-test-double",
        "sam_checkpoint_sha256": "a" * 64,
    })
    trainer = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(value), device="cpu",
    )
    raw = next(iter(trainer._loader(trainer.fit_dataset, training=True, epoch=0)))
    losses, _native, _base = trainer._forward_loss(
        trainer._move(raw), augmentation_enabled=True, phase="flow",
    )
    assert losses.sam_homography.ndim == 0
    assert losses.sam_candidate_regions >= losses.sam_fitted_regions >= 0


def test_sam_semantic_augmentation_prefers_exact_key_objects(tmp_path):
    value = _config(tmp_path / "sam-semantic-u2").serializable()
    value["sam_semantic_augmentation"] = asdict(
        SamSemanticAugmentationPolicyV1(
            enabled=True, activation_epoch=0, cache_size=2,
            objects_per_batch=1,
        )
    )
    value["dataset"]["kwargs"].update({
        "sam_context": True,
        "sam_full_segmentation_root": "/not-read-by-test-double",
        "sam_checkpoint_sha256": "a" * 64,
    })
    trainer = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(value), device="cpu",
    )
    raw = next(iter(trainer._loader(trainer.fit_dataset, training=True, epoch=0)))
    batch = trainer._move(raw)
    first, _native, _base = trainer._forward_loss(
        batch, augmentation_enabled=True, phase="joint",
        semantic_augmentation_enabled=False,
        semantic_cache_update_enabled=True,
    )
    assert first.sam_semantic_objects == 0
    assert trainer.sam_semantic_cache is not None
    assert trainer.sam_semantic_cache.ready
    second, _native, _base = trainer._forward_loss(
        batch, augmentation_enabled=True, phase="joint",
        semantic_augmentation_enabled=True,
        semantic_cache_update_enabled=True,
    )
    assert second.sam_semantic_objects == 2
    assert second.sam_semantic_object_pixels > 0
    assert torch.isfinite(second.sam_semantic)


def test_optional_three_frame_fusion_reports_separate_validation_metrics(tmp_path):
    value = _config(tmp_path / "fusion-validation").serializable()
    value["dataset"]["kwargs"]["fusion_context"] = True
    value["fusion"] = {
        "enabled": True,
        "optimization_steps": 1,
        "uncertainty_variance_threshold": 45.0,
        "learning_rate": 0.001,
        "learning_rate_decay": 0.8,
        "variance_minimum": 0.001,
        "variance_maximum": 200.0,
        "random_seed": 17,
    }
    trainer = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(value), device="cpu",
    )
    metrics = trainer.validate(0)
    assert metrics["fusion_enabled"] is True
    assert isinstance(metrics["fusion_epe"], float)
    assert 0.0 <= metrics["fusion_replaced_fraction"] <= 1.0
    assert trainer.selection_metric == "epe"


def test_head_only_config_must_freeze_matcher(tmp_path):
    value = _config(tmp_path / "run").serializable()
    value["model"]["kwargs"]["variant"] = "head_only"
    with pytest.raises(ValueError, match="must freeze"):
        IntegratedTrainerConfigV2.from_mapping(value)


def test_u1_requires_refiner_only_scope_zero_decoupled_loss_and_u0(tmp_path):
    value = _config(tmp_path / "u1").serializable()
    value["model"]["kwargs"]["trainable_scope"] = "refiner_only"
    value["uncertainty_initialization_checkpoint"] = str(tmp_path / "u0.pt")
    value["loss"].update({
        "task_weight": 0.0,
        "augmentation_weight": 0.0,
        "uncertainty_weight": 0.0,
    })
    value["refiner_loss"] = {
        "task_weight": 1.0,
        "harm_weight": 2.0,
        "anchor_weight": 0.05,
        "smoothness_weight": 0.01,
        "harm_margin_px": 0.0,
        "charbonnier_epsilon": 0.001,
    }
    parsed = IntegratedTrainerConfigV2.from_mapping(value)
    assert parsed.refiner_loss is not None

    value["model"]["kwargs"]["trainable_scope"] = "refinement_heads"
    with pytest.raises(ValueError, match="configured together"):
        IntegratedTrainerConfigV2.from_mapping(value)


def test_u2_schedule_requires_u1_checkpoint_all_scope_and_complete_rounds(tmp_path):
    value = _config(tmp_path / "u2-contract").serializable()
    value["uncertainty_initialization_checkpoint"] = str(tmp_path / "u0.pt")
    value["u2_schedule"] = {
        "mode": "alternating_decoupled",
        "observer_epochs_per_round": 1,
        "flow_epochs_per_round": 1,
    }
    with pytest.raises(ValueError, match="configured together"):
        IntegratedTrainerConfigV2.from_mapping(value)

    value["refiner_initialization_checkpoint"] = str(tmp_path / "u1.pt")
    value["training"]["epochs"] = 3
    with pytest.raises(ValueError, match="complete alternating rounds"):
        IntegratedTrainerConfigV2.from_mapping(value)

    value["training"]["epochs"] = 2
    value["model"]["kwargs"]["trainable_scope"] = "refinement_heads"
    with pytest.raises(ValueError, match="all parameters"):
        IntegratedTrainerConfigV2.from_mapping(value)


def test_head_only_selects_best_checkpoint_by_uncertainty(tmp_path):
    value = _config(tmp_path / "head-only").serializable()
    value["model"]["kwargs"].update({
        "variant": "head_only",
        "trainable_scope": "uncertainty_only",
    })
    trainer = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(value), device="cpu",
    )
    assert trainer.selection_metric == "uncertainty"
    assert trainer.train() == 0
    checkpoint = torch.load(trainer.best_path, map_location="cpu", weights_only=False)
    assert checkpoint["selection_metric"] == "uncertainty"
    assert checkpoint["best_validation_score"] == pytest.approx(
        next(
            row["selection_score"]
            for row in map(json.loads, trainer.metrics_path.read_text().splitlines())
            if row.get("event") == "validation_epoch" and row.get("best")
        )
    )


def test_u2_loads_completed_native_head_only_initialization(tmp_path):
    source_value = _config(tmp_path / "source-u0").serializable()
    source_value["model"]["kwargs"].update({
        "variant": "head_only",
        "trainable_scope": "uncertainty_only",
    })
    source = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(source_value), device="cpu",
    )
    assert source.train() == 0
    payload = torch.load(source.best_path, map_location="cpu", weights_only=False)
    for key in tuple(payload["model"]):
        if key.startswith("recurrent_head.uncertainty_head."):
            payload["model"][key] = torch.full_like(payload["model"][key], 0.375)
    torch.save(payload, source.best_path)

    target_value = _config(tmp_path / "target-u2").serializable()
    target_value["uncertainty_initialization_checkpoint"] = str(source.best_path)
    target = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(target_value), device="cpu",
    )
    assert target.initialization_lineage is not None
    assert target.initialization_lineage["checkpoint"] == str(source.best_path)
    assert all(
        torch.equal(parameter, torch.full_like(parameter, 0.375))
        for parameter in target.model.recurrent_head.uncertainty_head.state_dict().values()
    )


def test_new_stage_warm_starts_full_model_with_traced_lineage(tmp_path):
    source = IntegratedUncertaintyTrainerV2(
        _config(tmp_path / "source-full"), device="cpu",
    )
    assert source.train() == 0
    payload = torch.load(source.best_path, map_location="cpu", weights_only=False)
    for key in tuple(payload["model"]):
        payload["model"][key] = torch.full_like(payload["model"][key], 0.375)
    torch.save(payload, source.best_path)

    target_value = _config(tmp_path / "target-full").serializable()
    target_value["model_initialization_checkpoint"] = str(source.best_path)
    target = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(target_value), device="cpu",
    )
    assert target.initialization_lineage is None
    assert target.refiner_initialization_lineage is None
    assert target.model_initialization_lineage is not None
    assert target.model_initialization_lineage["checkpoint"] == str(source.best_path)
    assert all(
        torch.equal(parameter, torch.full_like(parameter, 0.375))
        for parameter in target.model.state_dict().values()
    )
    target.save()
    checkpoint = torch.load(target.latest_path, map_location="cpu", weights_only=False)
    assert checkpoint["model_initialization_lineage"] == (
        target.model_initialization_lineage
    )
    resumed = IntegratedUncertaintyTrainerV2(target.config, device="cpu")
    resumed.resume(target.latest_path)


def test_full_model_initialization_rejects_ambiguous_partial_sources(tmp_path):
    value = _config(tmp_path / "ambiguous-initialization").serializable()
    value["model_initialization_checkpoint"] = str(tmp_path / "full.pt")
    value["uncertainty_initialization_checkpoint"] = str(tmp_path / "u0.pt")
    with pytest.raises(ValueError, match="cannot be combined"):
        IntegratedTrainerConfigV2.from_mapping(value)


def test_u2_loads_matching_u1_and_alternates_disjoint_gradient_phases(tmp_path):
    u0_value = _config(tmp_path / "u0").serializable()
    u0_value["model"]["kwargs"].update({
        "variant": "head_only",
        "trainable_scope": "uncertainty_only",
    })
    u0 = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(u0_value), device="cpu",
    )
    assert u0.train() == 0

    u1_value = _config(tmp_path / "u1").serializable()
    u1_value["model"]["kwargs"]["trainable_scope"] = "refiner_only"
    u1_value["uncertainty_initialization_checkpoint"] = str(u0.best_path)
    u1_value["loss"].update({
        "task_weight": 0.0,
        "augmentation_weight": 0.0,
        "uncertainty_weight": 0.0,
    })
    u1_value["refiner_loss"] = {
        "task_weight": 1.0,
        "harm_weight": 2.0,
        "anchor_weight": 0.05,
        "smoothness_weight": 0.01,
        "harm_margin_px": 0.0,
        "charbonnier_epsilon": 0.001,
    }
    u1 = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(u1_value), device="cpu",
    )
    assert u1.train() == 0

    u2_value = _config(tmp_path / "u2").serializable()
    u2_value["training"]["epochs"] = 4
    u2_value["uncertainty_initialization_checkpoint"] = str(u0.best_path)
    u2_value["refiner_initialization_checkpoint"] = str(u1.best_path)
    u2_value["u2_schedule"] = {
        "mode": "alternating_decoupled",
        "observer_epochs_per_round": 1,
        "flow_epochs_per_round": 1,
    }
    u2_value["action_bank"] = {
        "mode": "cycle",
        "action_ids": [
            OPTICAL_NATIVE_ACTION_ID,
            "CSB/OF/SEA-RAFT/action/P4-iters8",
        ],
        "validation_action_id": OPTICAL_NATIVE_ACTION_ID,
        "bank_hash": OPTICAL_FLOW_CAPACITY_BANK_HASH,
        "source_manifest_sha256": OPTICAL_FLOW_CAPACITY_SOURCE_MANIFEST_SHA256,
    }
    u2 = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(u2_value), device="cpu",
    )
    assert u2.refiner_initialization_lineage is not None
    source = torch.load(u1.best_path, map_location="cpu", weights_only=False)
    for name, parameter in u2.model.recurrent_head.refinement_head.state_dict().items():
        assert torch.equal(
            parameter,
            source["model"][f"recurrent_head.refinement_head.{name}"],
        )

    bad_dir = tmp_path / "u1-wrong-lineage"
    bad_dir.mkdir()
    bad_source = copy.deepcopy(source)
    bad_source["uncertainty_initialization_lineage"]["checkpoint_sha256"] = "0" * 64
    bad_checkpoint = bad_dir / "best.pt"
    torch.save(bad_source, bad_checkpoint)
    (bad_dir / "metrics.jsonl").write_text(
        json.dumps({"event": "training_completed"}) + "\n",
        encoding="utf-8",
    )
    bad_value = copy.deepcopy(u2_value)
    bad_value["refiner_initialization_checkpoint"] = str(bad_checkpoint)
    with pytest.raises(ValueError, match="different U0 observer"):
        IntegratedUncertaintyTrainerV2(
            IntegratedTrainerConfigV2.from_mapping(bad_value), device="cpu",
        )

    phase, round_index = u2._configure_training_phase(0)
    assert (phase, round_index) == ("observer", 0)
    assert all(
        parameter.requires_grad
        for parameter in u2.model.recurrent_head.uncertainty_head.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in u2.model.recurrent_head.refinement_head.parameters()
    )
    assert all(not parameter.requires_grad for parameter in u2.model.network.parameters())

    raw = next(iter(u2._loader(u2.fit_dataset, training=True, epoch=0)))
    batch = u2._move(raw)
    observer_losses, _native, _base = u2._forward_loss(
        batch, augmentation_enabled=True, phase="observer",
    )
    observer_losses.total.backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in u2.model.recurrent_head.uncertainty_head.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in u2.model.recurrent_head.refinement_head.parameters()
    )
    assert all(parameter.grad is None for parameter in u2.model.network.parameters())
    u2.optimizer.zero_grad(set_to_none=True)

    phase, round_index = u2._configure_training_phase(1)
    assert (phase, round_index) == ("flow", 0)
    assert all(
        not parameter.requires_grad
        for parameter in u2.model.recurrent_head.uncertainty_head.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in u2.model.recurrent_head.refinement_head.parameters()
    )
    assert all(parameter.requires_grad for parameter in u2.model.network.parameters())

    flow_losses, _native, _base = u2._forward_loss(
        batch, augmentation_enabled=True, phase="flow",
    )
    flow_losses.total.backward()
    assert all(
        parameter.grad is None
        for parameter in u2.model.recurrent_head.uncertainty_head.parameters()
    )
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in u2.model.recurrent_head.refinement_head.parameters()
    )
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in u2.model.network.parameters()
    )
    u2.optimizer.zero_grad(set_to_none=True)

    assert u2.train() == 0
    rows = [json.loads(line) for line in u2.metrics_path.read_text().splitlines()]
    phases = {row.get("phase") for row in rows if row.get("event") == "train_step"}
    assert phases == {"observer", "flow"}
    steps = [row for row in rows if row.get("event") == "train_step"]
    epoch_actions = {
        row["epoch"]: (row["phase"], row["action_id"])
        for row in steps
    }
    assert epoch_actions == {
        0: ("observer", OPTICAL_NATIVE_ACTION_ID),
        1: ("flow", OPTICAL_NATIVE_ACTION_ID),
        2: ("observer", "CSB/OF/SEA-RAFT/action/P4-iters8"),
        3: ("flow", "CSB/OF/SEA-RAFT/action/P4-iters8"),
    }
    checkpoint = torch.load(u2.latest_path, map_location="cpu", weights_only=False)
    assert checkpoint["refiner_initialization_lineage"] == (
        u2.refiner_initialization_lineage
    )


def test_action_bank_cycle_drives_materialization_and_matcher_iterations(tmp_path):
    value = _config(tmp_path / "action-cycle").serializable()
    value["training"]["epochs"] = 2
    value["action_bank"] = {
        "mode": "cycle",
        "action_ids": [
            OPTICAL_NATIVE_ACTION_ID,
            "CSB/OF/SEA-RAFT/action/P4-iters8",
        ],
        "validation_action_id": OPTICAL_NATIVE_ACTION_ID,
        "bank_hash": OPTICAL_FLOW_CAPACITY_BANK_HASH,
        "source_manifest_sha256": OPTICAL_FLOW_CAPACITY_SOURCE_MANIFEST_SHA256,
    }
    trainer = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(value), device="cpu",
    )
    assert trainer.train() == 0
    rows = [json.loads(line) for line in trainer.metrics_path.read_text().splitlines()]
    steps = [row for row in rows if row["event"] == "train_step"]
    by_epoch = {row["epoch"]: (row["action_id"], row["matcher_iterations"]) for row in steps}
    assert by_epoch[0] == (OPTICAL_NATIVE_ACTION_ID, 1)
    assert by_epoch[1] == ("CSB/OF/SEA-RAFT/action/P4-iters8", 8)
    checkpoint = torch.load(trainer.latest_path, map_location="cpu", weights_only=False)
    assert checkpoint["action_bank_lineage"]["bank_hash"] == (
        OPTICAL_FLOW_CAPACITY_BANK_HASH
    )


def test_joint_decoupled_u2_updates_flow_and_uncertainty_in_every_step(tmp_path):
    source = IntegratedUncertaintyTrainerV2(
        _config(tmp_path / "joint-source"), device="cpu",
    )
    assert source.train() == 0

    value = _config(tmp_path / "joint-target").serializable()
    value["model_initialization_checkpoint"] = str(source.best_path)
    value["u2_schedule"] = {"mode": "joint_decoupled"}
    value["gradient_diagnostics"] = {
        "enabled": True,
        "every_steps": 1,
        "maximum_parameter_elements": 100_000,
        "epsilon": 1e-12,
    }
    trainer = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(value), device="cpu",
    )

    phase, round_index = trainer._configure_training_phase(0)
    assert (phase, round_index) == ("joint", 0)
    assert all(parameter.requires_grad for parameter in trainer.model.parameters())

    raw = next(iter(trainer._loader(trainer.fit_dataset, training=True, epoch=0)))
    losses, _native, _base = trainer._forward_loss(
        trainer._move(raw), augmentation_enabled=True, phase=phase,
    )
    losses.total.backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in trainer.model.recurrent_head.uncertainty_head.parameters()
    )
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in trainer.model.recurrent_head.refinement_head.parameters()
    )
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in trainer.model.network.parameters()
    )
    trainer.optimizer.zero_grad(set_to_none=True)

    assert trainer.train() == 0
    rows = [json.loads(line) for line in trainer.metrics_path.read_text().splitlines()]
    assert {
        row.get("phase") for row in rows if row.get("event") == "train_step"
    } == {"joint"}
    steps = [row for row in rows if row.get("event") == "train_step"]
    assert all(row["gradient_norm_task"] > 0.0 for row in steps)
    assert all(row["gradient_norm_uncertainty"] > 0.0 for row in steps)
    assert all("gradient_conflict_count" in row for row in steps)
    assert all(
        row["gradient_diagnostic_scope"] == "current_microbatch_before_backward"
        for row in steps
    )
    assert all(row["optimizer_gradient_accumulation_steps"] == 1 for row in steps)
    assert all(
        row["gradient_diagnostic_parameter_elements"] > 0 for row in steps
    )


def test_balanced_batches_interleave_actions_within_each_epoch(tmp_path):
    value = _config(tmp_path / "balanced-actions").serializable()
    value["dataset"]["kwargs"]["length"] = 4
    value["training"]["batch_size"] = 1
    value["action_bank"] = {
        "mode": "balanced_batches",
        "action_ids": [
            OPTICAL_NATIVE_ACTION_ID,
            "CSB/OF/SEA-RAFT/action/P4-iters8",
        ],
        "validation_action_id": OPTICAL_NATIVE_ACTION_ID,
        "bank_hash": OPTICAL_FLOW_CAPACITY_BANK_HASH,
        "source_manifest_sha256": OPTICAL_FLOW_CAPACITY_SOURCE_MANIFEST_SHA256,
    }
    value["action_shadow_probe"] = {
        "enabled": True,
        "every_epochs": 1,
        "maximum_batches": 2,
        "bootstrap_repetitions": 100,
        "confidence_level": 0.95,
        "seed": 17,
    }
    trainer = IntegratedUncertaintyTrainerV2(
        IntegratedTrainerConfigV2.from_mapping(value), device="cpu",
    )
    assert trainer.train() == 0
    rows = [json.loads(line) for line in trainer.metrics_path.read_text().splitlines()]
    steps = [row for row in rows if row.get("event") == "train_step"]
    assert {row["epoch"] for row in steps} == {0}
    assert {row["action_id"] for row in steps} == {
        OPTICAL_NATIVE_ACTION_ID,
        "CSB/OF/SEA-RAFT/action/P4-iters8",
    }
    assert {
        row["action_id"]: row["matcher_iterations"] for row in steps
    } == {
        OPTICAL_NATIVE_ACTION_ID: 1,
        "CSB/OF/SEA-RAFT/action/P4-iters8": 8,
    }
    shadow = [row for row in rows if row.get("event") == "action_shadow_probe"]
    assert {row["action_id"] for row in shadow} == {
        OPTICAL_NATIVE_ACTION_ID,
        "CSB/OF/SEA-RAFT/action/P4-iters8",
    }
    assert all(row["case_count"] == 2 and row["scene_count"] == 2 for row in shadow)
    assert all(row["authorized_for_selector_training"] is False for row in shadow)
    assert all(row["multiple_comparison_correction"] == "bonferroni" for row in shadow)
    assert all(row["familywise_confidence_level"] == 0.95 for row in shadow)
    assert all(row["hypothesis_count"] == 1 for row in shadow)
    assert all("mean_action_case_epe" in row for row in shadow)
    assert all("harmed_case_fraction" in row for row in shadow)
    case_rows = [
        json.loads(line)
        for line in trainer.action_shadow_path.read_text().splitlines()
    ]
    assert len(case_rows) == 4
    assert all(row["authorized_for_selector_training"] is False for row in case_rows)
    assert all(row["context_is_outcome_blind"] is True for row in case_rows)
    assert all(row["context_requires_ground_truth"] is False for row in case_rows)
    assert all(row["context_source_action_id"] == OPTICAL_NATIVE_ACTION_ID for row in case_rows)
    assert all("native_log_scale_mean" in row for row in case_rows)
    assert all("native_photometric_l1_mean" in row for row in case_rows)
