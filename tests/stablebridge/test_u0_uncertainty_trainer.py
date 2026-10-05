from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from stablebridge.physical_repair import sea_raft_uncertainty_adapter
from stablebridge.physical_repair.u0_uncertainty_trainer import (
    SeaRaftConsistencyBatchPreparerV1,
    U0TrainerConfigV1,
    U0UncertaintyTrainerV1,
)


class TinyConsistencyDataset(Dataset):
    def __init__(self, *, split: str, length: int = 3):
        self.split = split
        self.length = length
        self.epochs: list[int] = []

    def __len__(self):
        return self.length

    def set_epoch(self, epoch: int):
        self.epochs.append(epoch)

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(index + (0 if self.split == "fit" else 100))
        reference = torch.rand(2, 4, 5, generator=generator)
        return {
            "features": torch.rand(12, 4, 5, generator=generator),
            "reference_flow": reference,
            "restored_augmented_flow": reference + 0.1,
            "valid_mask": torch.ones(1, 4, 5, dtype=torch.bool),
        }


def make_tiny_dataset(*, split: str, length: int = 3):
    return TinyConsistencyDataset(split=split, length=length)


def _config(run_dir: Path) -> U0TrainerConfigV1:
    return U0TrainerConfigV1.from_mapping({
        "run_dir": str(run_dir),
        "dataset": {
            "factory": f"{__name__}:make_tiny_dataset",
            "kwargs": {"length": 3},
            "fit_split": "fit",
            "validation_split": "validation",
        },
        "model": {
            "factory": (
                "stablebridge.physical_repair.uncertainty_aware_flow:"
                "DecoupledUncertaintyHeadV1"
            ),
            "kwargs": {"feature_channels": 12, "hidden_channels": 4},
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "amp": False,
            "checkpoint_every_steps": 1,
            "validation_every_epochs": 1,
            "seed": 17,
        },
    })


def test_cpu_training_writes_resume_checkpoint_and_jsonl(tmp_path):
    config = _config(tmp_path / "run")
    trainer = U0UncertaintyTrainerV1(config, device="cpu")
    assert trainer.train() == 0
    assert trainer.latest_path.is_file()
    assert trainer.best_path.is_file()
    events = [json.loads(line) for line in trainer.metrics_path.read_text().splitlines()]
    assert events[0]["event"] == "training_started"
    assert events[-1]["event"] == "training_completed"
    assert any(row["event"] == "validation_epoch" for row in events)

    resumed = U0UncertaintyTrainerV1(config, device="cpu")
    resumed.resume(trainer.latest_path)
    assert resumed.epoch == 1
    assert resumed.global_step == trainer.global_step
    assert resumed.train() == 0


def test_sea_raft_preparer_uses_255_matcher_and_native_risk(monkeypatch):
    observed_maxima: list[float] = []

    class Predictor:
        def __call__(self, first, second):
            observed_maxima.append(float(first.max()))
            flow = torch.stack((first[:, 0] / 255.0, second[:, 0] / 255.0), dim=1)
            risk = torch.full_like(flow[:, :1], 0.25)
            return SimpleNamespace(
                flow=flow,
                uncertainty=SimpleNamespace(upstream_heatmap_log_scale=risk),
            )

        def predict_flow(self, first, second):
            observed_maxima.append(float(first.max()))
            return torch.stack((first[:, 0] / 255.0, second[:, 0] / 255.0), dim=1)

    def fake_loader(**kwargs):
        assert kwargs["device"] == "cpu"
        return Predictor(), {"provider": "fake"}

    monkeypatch.setattr(
        sea_raft_uncertainty_adapter, "load_pinned_sea_raft_predictor", fake_loader,
    )
    native = torch.full((2, 2, 3, 4, 5), 0.5)
    augmented = native.clone()
    augmented[:, 0, 0] = 0.25
    prepared = SeaRaftConsistencyBatchPreparerV1()(
        {
            "native_frames": native,
            "augmented_frames": augmented,
            "augmentation_valid": torch.ones(2, 1, 4, 5, dtype=torch.bool),
        },
        device=torch.device("cpu"),
        training=True,
    )
    assert prepared["features"].shape == (2, 12, 4, 5)
    assert prepared["valid_mask"].any()
    assert not prepared["valid_mask"].all()
    assert torch.all(prepared["features"][:, 8] == 0.25)
    assert max(observed_maxima) == 127.5
    assert not torch.equal(
        prepared["reference_flow"], prepared["restored_augmented_flow"],
    )


def test_config_supports_sintel_role_and_collate_symbols(tmp_path):
    value = _config(tmp_path / "run").serializable()
    value["dataset"]["split_argument"] = "role"
    value["dataset"]["collate_fn"] = (
        "stablebridge.physical_repair.u0_sintel_dataset:"
        "collate_sintel_u0_samples_v1"
    )
    parsed = U0TrainerConfigV1.from_mapping(value)
    assert parsed.dataset["split_argument"] == "role"
