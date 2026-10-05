from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import Dataset

from stablebridge.physical_repair.uncertainty_aware_flow import (
    DecoupledUncertaintyHeadV1,
)
from stablebridge.physical_repair.u0_uncertainty_trainer import TRAINER_SCHEMA_V1
from stablebridge.physical_repair.uncertainty_flow_trainer import (
    FLOW_TRAINER_SCHEMA_V1,
    SeaRaftRefinementBatchPreparerV1,
    UncertaintyFlowTrainerConfigV1,
    UncertaintyFlowTrainerV1,
)
from stablebridge.physical_repair import sea_raft_uncertainty_adapter


class TinyFlowDataset(Dataset):
    def __init__(self, *, split: str, include_consistency: bool, length: int = 3):
        self.split = split
        self.include_consistency = include_consistency
        self.length = length
        self.epochs: list[int] = []

    def __len__(self):
        return self.length

    def set_epoch(self, epoch: int):
        self.epochs.append(epoch)

    def __getitem__(self, index: int):
        offset = 0 if self.split == "fit" else 100
        generator = torch.Generator().manual_seed(index + offset)
        features = torch.rand(3, 4, 5, generator=generator)
        base = torch.zeros(2, 4, 5)
        truth = torch.stack((features[0] * 0.2, features[1] * -0.1))
        sample = {
            "features": features,
            "base_flow": base,
            "ground_truth_flow": truth,
            "valid_mask": torch.ones(1, 4, 5, dtype=torch.bool),
        }
        if self.include_consistency:
            sample.update({
                "reference_flow": base,
                "restored_augmented_flow": base + 0.1 + features[:2] * 0.05,
                "consistency_valid_mask": torch.ones(1, 4, 5, dtype=torch.bool),
            })
        return sample


def make_tiny_flow_dataset(
    *, split: str, include_consistency: bool, length: int = 3,
):
    return TinyFlowDataset(
        split=split, include_consistency=include_consistency, length=length,
    )


def _write_u0_checkpoint(path: Path) -> Path:
    torch.manual_seed(3)
    model = DecoupledUncertaintyHeadV1(feature_channels=3, hidden_channels=4)
    payload = {
        "schema": TRAINER_SCHEMA_V1,
        "config": {
            "model": {
                "factory": (
                    "stablebridge.physical_repair.uncertainty_aware_flow:"
                    "DecoupledUncertaintyHeadV1"
                ),
                "kwargs": {"feature_channels": 3, "hidden_channels": 4},
            }
        },
        "model": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _config(
    run_dir: Path,
    observer: Path,
    *,
    stage: str,
    include_consistency: bool,
    refiner_checkpoint: Path | None = None,
) -> UncertaintyFlowTrainerConfigV1:
    return UncertaintyFlowTrainerConfigV1.from_mapping({
        "schema": FLOW_TRAINER_SCHEMA_V1,
        "run_dir": str(run_dir),
        "stage": stage,
        "dataset": {
            "factory": f"{__name__}:make_tiny_flow_dataset",
            "kwargs": {"include_consistency": include_consistency, "length": 3},
            "fit_split": "fit",
            "validation_split": "validation",
        },
        "observer_checkpoint": str(observer),
        "require_completed_sources": False,
        "refiner": {
            "factory": (
                "stablebridge.physical_repair.uncertainty_flow_trainer:"
                "make_bounded_uncertainty_flow_refiner_v1"
            ),
            "kwargs": {
                "feature_channels": 3,
                "hidden_channels": 4,
                "maximum_update_px": 0.2,
                "evidence_role": "base_flow",
            },
        },
        "refiner_checkpoint": (
            None if refiner_checkpoint is None else str(refiner_checkpoint)
        ),
        "loss": {
            "task_weight": 1.0,
            "harm_weight": 2.0,
            "anchor_weight": 0.05,
            "smoothness_weight": 0.01,
            "harm_margin_px": 0.0,
            "charbonnier_epsilon": 0.001,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 0.001,
            "num_workers": 0,
            "checkpoint_every_steps": 1,
            "validation_every_phase_epochs": 1,
            "amp": False,
            "seed": 17,
            "u1_epochs": 1,
            "u2_rounds": 1,
            "u2_observer_epochs_per_round": 1,
            "u2_refiner_epochs_per_round": 1,
        },
    })


def _state_clone(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _changed(before, after):
    return any(not torch.equal(before[name], after[name]) for name in before)


def test_u1_updates_only_refiner_and_writes_resume_checkpoint(tmp_path):
    observer = _write_u0_checkpoint(tmp_path / "u0.pt")
    config = _config(
        tmp_path / "u1", observer,
        stage="frozen_uncertainty_refiner", include_consistency=False,
    )
    trainer = UncertaintyFlowTrainerV1(config, device="cpu")
    observer_before = _state_clone(trainer.observer)
    refiner_before = _state_clone(trainer.refiner)
    assert trainer.train() == 0
    assert not _changed(observer_before, trainer.observer.state_dict())
    assert _changed(refiner_before, trainer.refiner.state_dict())
    assert trainer.latest_path.is_file()
    assert trainer.best_refiner_path.is_file()
    assert not trainer.best_observer_path.exists()
    events = [json.loads(line) for line in trainer.metrics_path.read_text().splitlines()]
    assert events[-1]["event"] == "training_completed"
    assert events[-1]["observer_steps"] == 0
    assert events[-1]["refiner_steps"] > 0

    resumed = UncertaintyFlowTrainerV1(config, device="cpu")
    resumed.resume(trainer.latest_path)
    assert resumed.phase_index == len(resumed.schedule)
    assert resumed.train() == 0


def test_u2_alternates_detached_observer_and_refiner_updates(tmp_path):
    observer = _write_u0_checkpoint(tmp_path / "u0.pt")
    u1_config = _config(
        tmp_path / "u1", observer,
        stage="frozen_uncertainty_refiner", include_consistency=False,
    )
    u1 = UncertaintyFlowTrainerV1(u1_config, device="cpu")
    assert u1.train() == 0

    u2_config = _config(
        tmp_path / "u2", observer,
        stage="alternating_decoupled", include_consistency=True,
        refiner_checkpoint=u1.best_refiner_path,
    )
    u2 = UncertaintyFlowTrainerV1(u2_config, device="cpu")
    observer_before = _state_clone(u2.observer)
    refiner_before = _state_clone(u2.refiner)
    assert u2.train() == 0
    assert _changed(observer_before, u2.observer.state_dict())
    assert _changed(refiner_before, u2.refiner.state_dict())
    assert (u2_config.run_dir / "round_00.pt").is_file()
    events = [json.loads(line) for line in u2.metrics_path.read_text().splitlines()]
    phases = [row["phase"] for row in events if row["event"] == "train_step"]
    assert phases[:2] == ["observer", "observer"]
    assert phases[2:] == ["refiner", "refiner"]
    completed = events[-1]
    assert completed["observer_steps"] == 2
    assert completed["refiner_steps"] == 2


def test_u2_rejects_refiner_trained_from_another_observer(tmp_path):
    first_observer = _write_u0_checkpoint(tmp_path / "first.pt")
    u1_config = _config(
        tmp_path / "u1", first_observer,
        stage="frozen_uncertainty_refiner", include_consistency=False,
    )
    u1 = UncertaintyFlowTrainerV1(u1_config, device="cpu")
    assert u1.train() == 0
    second_observer = _write_u0_checkpoint(tmp_path / "second.pt")
    checkpoint = torch.load(second_observer, weights_only=False)
    first_key = next(iter(checkpoint["model"]))
    checkpoint["model"][first_key] = checkpoint["model"][first_key] + 1.0
    torch.save(checkpoint, second_observer)
    u2_config = _config(
        tmp_path / "u2", second_observer,
        stage="alternating_decoupled", include_consistency=True,
        refiner_checkpoint=u1.best_refiner_path,
    )
    with pytest.raises(ValueError, match="different U0 observer"):
        UncertaintyFlowTrainerV1(u2_config, device="cpu")


def test_real_runs_require_completed_source_lifecycle(tmp_path):
    observer = _write_u0_checkpoint(tmp_path / "u0" / "best.pt")
    config = _config(
        tmp_path / "u1", observer,
        stage="frozen_uncertainty_refiner", include_consistency=False,
    )
    value = config.serializable()
    value["require_completed_sources"] = True
    guarded = UncertaintyFlowTrainerConfigV1.from_mapping(value)
    with pytest.raises(RuntimeError, match="has not completed|no metrics"):
        UncertaintyFlowTrainerV1(guarded, device="cpu")

    metrics = observer.parent / "metrics.jsonl"
    metrics.write_text(
        '{"event":"training_completed"}\n{"event":"training_started"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="has not completed"):
        UncertaintyFlowTrainerV1(guarded, device="cpu")


def test_refinement_preparer_keeps_matcher_frozen_and_emits_both_targets(monkeypatch):
    class Predictor:
        def __call__(self, first, second):
            flow = torch.stack((first[:, 0], second[:, 0]), dim=1).div(255.0)
            risk = torch.full_like(flow[:, :1], 0.2)
            return type("Prediction", (), {
                "flow": flow,
                "uncertainty": type("U", (), {"upstream_heatmap_log_scale": risk})(),
            })()

        def predict_flow(self, first, second):
            return self(first, second).flow + 0.05

    monkeypatch.setattr(
        sea_raft_uncertainty_adapter,
        "load_pinned_sea_raft_predictor",
        lambda **_kwargs: (Predictor(), {"provider": "fake"}),
    )
    native = torch.full((2, 2, 3, 4, 5), 0.5)
    augmented = native.clone()
    augmented[:, 1] *= 0.9
    prepared = SeaRaftRefinementBatchPreparerV1(
        include_augmented_teacher=True,
    )({
        "native_frames": native,
        "augmented_frames": augmented,
        "augmentation_valid": torch.ones(2, 1, 4, 5, dtype=torch.bool),
        "ground_truth_flow": torch.zeros(2, 2, 4, 5),
        "ground_truth_valid": torch.ones(2, 1, 4, 5, dtype=torch.bool),
    }, device=torch.device("cpu"), training=True)
    assert prepared["features"].shape == (2, 12, 4, 5)
    assert prepared["base_flow"].shape == (2, 2, 4, 5)
    assert prepared["ground_truth_flow"].shape == (2, 2, 4, 5)
    assert prepared["valid_mask"].all()
    assert prepared["consistency_valid_mask"].any()
    assert not prepared["base_flow"].requires_grad
