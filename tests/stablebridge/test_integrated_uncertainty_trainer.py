from __future__ import annotations

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


class TinyIntegratedDataset(Dataset):
    def __init__(self, *, split: str, length: int = 3):
        self.split = split
        self.length = length
        self.epoch = 0

    def __len__(self):
        return self.length

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(index + (100 if self.split == "validation" else 0))
        native = torch.rand(2, 3, 4, 5, generator=generator)
        augmented = (native * 0.9).clamp(0, 1)
        truth = torch.stack((native[0, 0] * 0.1, native[0, 1] * -0.1))
        return {
            "native_frames": native,
            "augmented_frames": augmented,
            "augmentation_valid": torch.ones(1, 4, 5, dtype=torch.bool),
            "ground_truth_flow": truth,
            "ground_truth_valid": torch.ones(1, 4, 5, dtype=torch.bool),
            "native_to_augmented": torch.eye(3),
            "swap_endpoints": False,
        }


def make_tiny_integrated_dataset(*, split: str, length: int = 3):
    return TinyIntegratedDataset(split=split, length=length)


def collate_tiny_integrated(samples):
    return {
        key: torch.stack([
            value if torch.is_tensor(value) else torch.tensor(value)
            for value in (sample[key] for sample in samples)
        ])
        for key in samples[0]
    }


class TinyIntegratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Conv2d(6, 3, 1)

    def forward(self, first, second, *, iters, test_mode):
        del test_mode
        prediction = self.network(torch.cat((first, second), dim=1) / 255.0)
        flows = [prediction[:, :2] * (index + 1) / (iters + 1) for index in range(iters + 1)]
        alphas = [prediction[:, 2:3] for _ in range(iters + 1)]
        return {"flow": flows, "uncertainty_log_variance": alphas, "final": flows[-1]}


def make_tiny_integrated_model(*, device: str, variant: str, trainable_scope: str):
    del variant, trainable_scope
    return TinyIntegratedModel().to(device)


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


def test_head_only_config_must_freeze_matcher(tmp_path):
    value = _config(tmp_path / "run").serializable()
    value["model"]["kwargs"]["variant"] = "head_only"
    with pytest.raises(ValueError, match="must freeze"):
        IntegratedTrainerConfigV2.from_mapping(value)
