"""End-to-end trainer for recurrent uncertainty-aware SEA-RAFT ablations."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .integrated_uncertainty_flow import (
    DecoupledFlowLossPolicyV2,
    RecurrentUncertaintyVariantV2,
    TrainableScopeV2,
    decoupled_uncertainty_flow_loss_v2,
    transform_flow_affine_v2,
)
from .uncertainty_evaluation import evaluate_log_scale_uncertainty_v1


INTEGRATED_TRAINER_SCHEMA_V2 = "stablebridge-integrated-uncertainty-trainer/v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _symbol(path: str):
    module_name, separator, name = path.partition(":")
    if not separator:
        raise ValueError("factory must use module:symbol syntax")
    return getattr(importlib.import_module(module_name), name)


def _factory(spec: Mapping[str, Any], **extra: Any):
    if set(spec) != {"factory", "kwargs"}:
        raise ValueError("factory spec needs exact factory/kwargs fields")
    kwargs = dict(spec["kwargs"])
    kwargs.update(extra)
    return _symbol(str(spec["factory"]))(**kwargs)


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class IntegratedTrainingOptionsV2:
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 2.5e-5
    weight_decay: float = 1e-6
    num_workers: int = 4
    gradient_accumulation_steps: int = 2
    checkpoint_every_steps: int = 100
    validation_every_epochs: int = 1
    amp: bool = True
    seed: int = 11
    gradient_clip_norm: float = 1.0
    pin_memory: bool = True
    augmentation_start_epoch: int = 0

    def __post_init__(self) -> None:
        for name in (
            "epochs", "batch_size", "num_workers", "gradient_accumulation_steps",
            "checkpoint_every_steps", "validation_every_epochs", "augmentation_start_epoch",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if min(
            self.epochs, self.batch_size, self.gradient_accumulation_steps,
            self.checkpoint_every_steps, self.validation_every_epochs,
        ) <= 0:
            raise ValueError("positive training counters cannot be zero")
        for name in ("learning_rate", "weight_decay", "gradient_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("learning rate and gradient clip norm must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.amp, bool) or not isinstance(self.pin_memory, bool):
            raise ValueError("amp and pin_memory must be boolean")


@dataclass(frozen=True)
class IntegratedTrainerConfigV2:
    run_dir: Path
    dataset: Mapping[str, Any]
    model: Mapping[str, Any]
    loss: DecoupledFlowLossPolicyV2
    training: IntegratedTrainingOptionsV2
    iters: int = 4
    severe_error_threshold: float = 3.0
    schema: str = INTEGRATED_TRAINER_SCHEMA_V2

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IntegratedTrainerConfigV2":
        expected = {
            "schema", "run_dir", "dataset", "model", "loss", "training", "iters",
            "severe_error_threshold",
        }
        if set(value) != expected or value["schema"] != INTEGRATED_TRAINER_SCHEMA_V2:
            raise ValueError("integrated trainer config schema/fields drift")
        dataset = dict(value["dataset"])
        if set(dataset) != {"factory", "kwargs", "fit_split", "validation_split", "split_argument", "collate_fn"}:
            raise ValueError("integrated dataset spec fields drift")
        model = dict(value["model"])
        if set(model) != {"factory", "kwargs"}:
            raise ValueError("integrated model spec fields drift")
        model_kwargs = dict(model["kwargs"])
        variant = RecurrentUncertaintyVariantV2(model_kwargs.get("variant"))
        scope = TrainableScopeV2(model_kwargs.get("trainable_scope"))
        if variant is RecurrentUncertaintyVariantV2.HEAD_ONLY and scope is not TrainableScopeV2.UNCERTAINTY_ONLY:
            raise ValueError("head-only ablation must freeze the matcher")
        iters = value["iters"]
        if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
            raise ValueError("iters must be a positive integer")
        threshold = float(value["severe_error_threshold"])
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("severe error threshold must be finite and nonnegative")
        return cls(
            run_dir=Path(value["run_dir"]).expanduser().resolve(),
            dataset=dataset,
            model=model,
            loss=DecoupledFlowLossPolicyV2(**dict(value["loss"])),
            training=IntegratedTrainingOptionsV2(**dict(value["training"])),
            iters=iters,
            severe_error_threshold=threshold,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "IntegratedTrainerConfigV2":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("integrated trainer JSON must contain an object")
        return cls.from_mapping(payload)

    def serializable(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_dir": str(self.run_dir),
            "dataset": dict(self.dataset),
            "model": dict(self.model),
            "loss": asdict(self.loss),
            "training": asdict(self.training),
            "iters": self.iters,
            "severe_error_threshold": self.severe_error_threshold,
        }

    @property
    def digest(self) -> str:
        return _digest(self.serializable())


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _freeze_batch_norm_statistics(module: torch.nn.Module) -> None:
    # U2Flow replaces BN with instance normalization.  The pinned SEA-RAFT
    # checkpoint cannot be changed compatibly, so small-batch adaptation keeps
    # pretrained BN statistics fixed while allowing affine weights to train.
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


class IntegratedUncertaintyTrainerV2:
    def __init__(self, config: IntegratedTrainerConfigV2, *, device: str = "auto") -> None:
        self.config = config
        self.options = config.training
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        _seed_all(self.options.seed)
        split_argument = str(config.dataset["split_argument"])
        common = {"factory": config.dataset["factory"], "kwargs": config.dataset["kwargs"]}
        self.fit_dataset = _factory(common, **{split_argument: config.dataset["fit_split"]})
        self.validation_dataset = _factory(common, **{split_argument: config.dataset["validation_split"]})
        if not len(self.fit_dataset) or not len(self.validation_dataset):
            raise ValueError("fit and validation datasets must be nonempty")
        self.collate_fn = _symbol(str(config.dataset["collate_fn"]))
        self.model = _factory(config.model, device=str(self.device)).to(self.device)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("integrated model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable, lr=self.options.learning_rate, weight_decay=self.options.weight_decay,
        )
        self.trainable = tuple(trainable)
        amp = self.options.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp)
        self.amp_enabled = amp
        self.epoch = 0
        self.next_batch_index = 0
        self.global_step = 0
        self.best_validation_epe = float("inf")
        self.run_dir = config.run_dir
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.latest_path = self.run_dir / "latest.pt"
        self.best_path = self.run_dir / "best.pt"

    def _loader(self, dataset, *, training: bool, epoch: int) -> DataLoader:
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(self.options.seed + epoch + (0 if training else 100000))
        return DataLoader(
            dataset,
            batch_size=self.options.batch_size,
            shuffle=training,
            num_workers=self.options.num_workers,
            pin_memory=self.options.pin_memory,
            persistent_workers=False,
            generator=generator,
            collate_fn=self.collate_fn,
        )

    def _autocast(self):
        if hasattr(torch, "amp"):
            return torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled)
        return nullcontext()

    def _move(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in batch.items():
            result[key] = value.to(self.device, non_blocking=self.device.type == "cuda") if torch.is_tensor(value) else value
        required = {
            "native_frames", "augmented_frames", "augmentation_valid", "ground_truth_flow",
            "ground_truth_valid", "native_to_augmented", "swap_endpoints",
        }
        if required - set(result) or any(result[key] is None for key in required):
            raise ValueError("integrated batch is missing required tensors")
        return result

    @staticmethod
    def _flows(output: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
        flows = output.get("flow")
        if not isinstance(flows, (tuple, list)) or not flows:
            raise ValueError("model output needs a nonempty flow sequence")
        return tuple(flows)

    @staticmethod
    def _alphas(output: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
        values = output.get("uncertainty_log_variance")
        if not isinstance(values, (tuple, list)) or not values:
            raise ValueError("model output needs recurrent log variances")
        return tuple(values)

    def _forward_loss(self, batch: Mapping[str, Any], *, augmentation_enabled: bool):
        native = batch["native_frames"].float() * 255.0
        augmented = batch["augmented_frames"].float() * 255.0
        native_output = self.model(
            native[:, 0], native[:, 1], iters=self.config.iters, test_mode=True,
        )
        native_flows = self._flows(native_output)
        swap = batch["swap_endpoints"].to(dtype=torch.bool).reshape(-1, 1, 1, 1)
        teacher = native_flows[-1].detach()
        if bool(swap.any()):
            with torch.no_grad():
                reverse_output = self.model(
                    native[:, 1], native[:, 0], iters=self.config.iters, test_mode=True,
                )
                reverse = self._flows(reverse_output)[-1]
            teacher = torch.where(swap, reverse.detach(), teacher)
        transformed_teacher, consistency_valid = transform_flow_affine_v2(
            teacher,
            batch["native_to_augmented"].float(),
            augmented.shape[-2:],
            augmented_valid=batch["augmentation_valid"].bool(),
        )
        augmented_output = self.model(
            augmented[:, 0], augmented[:, 1], iters=self.config.iters, test_mode=True,
        )
        selected_policy = self.config.loss
        if not augmentation_enabled:
            selected_policy = DecoupledFlowLossPolicyV2(
                task_weight=selected_policy.task_weight,
                augmentation_weight=0.0,
                uncertainty_weight=0.0,
                gamma=selected_policy.gamma,
            )
        losses = decoupled_uncertainty_flow_loss_v2(
            native_flows,
            self._flows(augmented_output),
            self._alphas(augmented_output),
            batch["ground_truth_flow"].float(),
            batch["ground_truth_valid"].bool(),
            transformed_teacher,
            consistency_valid,
            policy=selected_policy,
        )
        return losses, native_output

    def _write(self, row: Mapping[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")

    def _checkpoint(self) -> dict[str, Any]:
        return {
            "schema": INTEGRATED_TRAINER_SCHEMA_V2,
            "config": self.config.serializable(),
            "config_digest": self.config.digest,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "next_batch_index": self.next_batch_index,
            "global_step": self.global_step,
            "best_validation_epe": self.best_validation_epe,
            "rng": _capture_rng(),
            "saved_utc": _utc_now(),
        }

    def save(self, path: Path | None = None) -> Path:
        target = self.latest_path if path is None else path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        try:
            torch.save(self._checkpoint(), temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def resume(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("schema") != INTEGRATED_TRAINER_SCHEMA_V2 or payload.get("config_digest") != self.config.digest:
            raise ValueError("integrated checkpoint config/schema mismatch")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.epoch = int(payload["epoch"])
        self.next_batch_index = int(payload["next_batch_index"])
        self.global_step = int(payload["global_step"])
        self.best_validation_epe = float(payload["best_validation_epe"])
        _restore_rng(payload["rng"])

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float | None]:
        self.model.eval()
        totals = {"total": 0.0, "task": 0.0, "augmentation": 0.0, "uncertainty": 0.0}
        count = 0
        all_alpha, all_error, all_valid = [], [], []
        for raw in self._loader(self.validation_dataset, training=False, epoch=epoch):
            batch = self._move(raw)
            with self._autocast():
                losses, native_output = self._forward_loss(batch, augmentation_enabled=True)
            batch_size = int(batch["native_frames"].shape[0])
            count += batch_size
            for name in totals:
                totals[name] += float(getattr(losses, name).detach().cpu()) * batch_size
            final_flow = self._flows(native_output)[-1]
            final_alpha = self._alphas(native_output)[-1]
            endpoint_error = torch.linalg.vector_norm(
                final_flow - batch["ground_truth_flow"].float(), dim=1,
            )
            valid = batch["ground_truth_valid"].bool()[:, 0]
            # Eq. 9 corresponds to b=exp(alpha/2)/sqrt(2).
            log_scale = 0.5 * final_alpha[:, 0] - 0.5 * math.log(2.0)
            all_alpha.append(log_scale.detach().float().cpu().numpy())
            all_error.append(endpoint_error.detach().float().cpu().numpy())
            all_valid.append(valid.detach().cpu().numpy())
        if count == 0:
            raise ValueError("validation loader is empty")
        evaluated = evaluate_log_scale_uncertainty_v1(
            np.concatenate(all_alpha), np.concatenate(all_error), np.concatenate(all_valid),
            severe_threshold=self.config.severe_error_threshold,
        )
        result = {name: value / count for name, value in totals.items()}
        def finite_or_none(value: float) -> float | None:
            return float(value) if math.isfinite(float(value)) else None

        result.update({
            "epe": evaluated.mean_endpoint_error,
            "ause": evaluated.ause,
            "spearman": finite_or_none(evaluated.spearman_error),
            "severe_auroc": finite_or_none(evaluated.severe_auroc),
            "calibration_mae": evaluated.coverage_calibration_mae,
        })
        return result

    def train(self) -> int:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        resolved = self.run_dir / "resolved_config.json"
        if not resolved.exists():
            resolved.write_text(
                json.dumps(self.config.serializable(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self._write({"event": "training_started", "utc": _utc_now(), "config_digest": self.config.digest})
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.epoch, self.options.epochs):
            self.model.train()
            _freeze_batch_norm_statistics(self.model)
            loader = self._loader(self.fit_dataset, training=True, epoch=epoch)
            for batch_index, raw in enumerate(loader):
                if epoch == self.epoch and batch_index < self.next_batch_index:
                    continue
                batch = self._move(raw)
                with self._autocast():
                    losses, _native_output = self._forward_loss(
                        batch,
                        augmentation_enabled=epoch >= self.options.augmentation_start_epoch,
                    )
                    scaled_loss = losses.total / self.options.gradient_accumulation_steps
                self.scaler.scale(scaled_loss).backward()
                end_accumulation = (
                    (batch_index + 1) % self.options.gradient_accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if end_accumulation:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable, self.options.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                self.next_batch_index = batch_index + 1
                if end_accumulation:
                    self._write({
                        "event": "train_step", "utc": _utc_now(), "epoch": epoch,
                        "global_step": self.global_step,
                        "loss": float(losses.total.detach().cpu()),
                        "task": float(losses.task.detach().cpu()),
                        "augmentation": float(losses.augmentation.detach().cpu()),
                        "uncertainty": float(losses.uncertainty.detach().cpu()),
                    })
                    if self.global_step % self.options.checkpoint_every_steps == 0:
                        self.save()
            self.epoch = epoch + 1
            self.next_batch_index = 0
            if self.epoch % self.options.validation_every_epochs == 0:
                metrics = self.validate(epoch)
                validation_epe = metrics["epe"]
                assert isinstance(validation_epe, float)
                improved = validation_epe < self.best_validation_epe
                if improved:
                    self.best_validation_epe = validation_epe
                self._write({
                    "event": "validation_epoch", "utc": _utc_now(), "epoch": epoch,
                    "global_step": self.global_step, "best": improved, **metrics,
                })
                self.save()
                if improved:
                    self.save(self.best_path)
        self._write({
            "event": "training_completed", "utc": _utc_now(),
            "epochs": self.options.epochs, "global_step": self.global_step,
            "best_validation_epe": self.best_validation_epe,
        })
        self.save()
        return 0


__all__ = [
    "INTEGRATED_TRAINER_SCHEMA_V2",
    "IntegratedTrainerConfigV2",
    "IntegratedTrainingOptionsV2",
    "IntegratedUncertaintyTrainerV2",
]
