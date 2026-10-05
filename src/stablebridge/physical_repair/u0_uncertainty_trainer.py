"""Resume-capable trainer for the native-only U0 uncertainty observer.

The training loop deliberately knows nothing about SEA-RAFT or a particular
dataset layout.  A dataset factory returns ordinary PyTorch samples and a
``batch_preparer`` converts a collated batch into the four tensors required by
the augmentation-consistency objective.  This keeps the frozen matcher out of
the optimizer and makes cached and online teacher implementations equivalent
to the trainer.

The JSON configuration has this minimal shape::

    {
      "run_dir": "/path/to/run",
      "dataset": {
        "factory": "package.module:make_dataset",
        "kwargs": {},
        "fit_split": "fit",
        "validation_split": "validation"
      },
      "batch_preparer": {
        "factory": "package.module:make_preparer",
        "kwargs": {}
      },
      "model": {
        "factory": "stablebridge.physical_repair.uncertainty_aware_flow:DecoupledUncertaintyHeadV1",
        "kwargs": {"feature_channels": 12}
      },
      "training": {"epochs": 20, "batch_size": 4}
    }

Factories are trusted local Python entry points.  The dataset factory is
called with ``split=...``.  The optional preparer is called as
``preparer(batch, device=device, training=bool)`` and must return a mapping
containing ``features``, ``reference_flow``, ``restored_augmented_flow``, and
``valid_mask``.  If it is omitted, the collated batch must already contain
those keys.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import signal
import sys
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .uncertainty_aware_flow import augmentation_consistency_laplace_loss_v1
from .uncertainty_features import build_searaft_observable_features_v1


TRAINER_SCHEMA_V1 = "stablebridge-u0-uncertainty-trainer/v1"
_REQUIRED_BATCH_KEYS = frozenset({
    "features", "reference_flow", "restored_augmented_flow", "valid_mask",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _load_symbol(reference: str) -> Any:
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("factory reference must have form package.module:symbol")
    module_name, symbol_name = reference.split(":", 1)
    if not module_name or not symbol_name or symbol_name.startswith("_"):
        raise ValueError("factory reference must name a public symbol")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as error:
        raise ValueError(f"factory symbol does not exist: {reference}") from error


def _factory(spec: Mapping[str, Any], *, extra: Mapping[str, Any] | None = None) -> Any:
    expected = {"factory", "kwargs"}
    if set(spec) != expected:
        raise ValueError(
            f"factory spec fields drifted; expected={sorted(expected)}, got={sorted(spec)}"
        )
    kwargs = dict(spec["kwargs"])
    if extra:
        overlap = set(kwargs) & set(extra)
        if overlap:
            raise ValueError(f"factory kwargs duplicate injected fields: {sorted(overlap)}")
        kwargs.update(extra)
    return _load_symbol(spec["factory"])(**kwargs)


@dataclass(frozen=True)
class TrainingOptionsV1:
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 4
    gradient_accumulation_steps: int = 1
    checkpoint_every_steps: int = 500
    validation_every_epochs: int = 1
    amp: bool = True
    seed: int = 8603
    pin_memory: bool = True
    gradient_clip_norm: float | None = 1.0

    def __post_init__(self) -> None:
        integers = (
            "epochs", "batch_size", "gradient_accumulation_steps",
            "checkpoint_every_steps", "validation_every_epochs",
        )
        for name in integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"training.{name} must be a positive integer")
        if isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ValueError("training.num_workers must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("training.seed must be a non-negative integer")
        for name in ("learning_rate", "weight_decay"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"training.{name} has an invalid value")
        if self.gradient_clip_norm is not None:
            value = float(self.gradient_clip_norm)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("training.gradient_clip_norm must be positive or null")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingOptionsV1":
        allowed = set(cls.__dataclass_fields__)
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"unknown training fields: {sorted(extra)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class U0TrainerConfigV1:
    run_dir: Path
    dataset: Mapping[str, Any]
    model: Mapping[str, Any]
    training: TrainingOptionsV1
    batch_preparer: Mapping[str, Any] | None = None
    schema: str = TRAINER_SCHEMA_V1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "U0TrainerConfigV1":
        allowed = {"schema", "run_dir", "dataset", "model", "batch_preparer", "training"}
        extra = set(value) - allowed
        missing = {"run_dir", "dataset", "model", "training"} - set(value)
        if missing or extra:
            raise ValueError(f"trainer config fields drifted; missing={sorted(missing)}, extra={sorted(extra)}")
        schema = value.get("schema", TRAINER_SCHEMA_V1)
        if schema != TRAINER_SCHEMA_V1:
            raise ValueError("unsupported trainer config schema")
        dataset = dict(value["dataset"])
        required_dataset = {"factory", "kwargs", "fit_split", "validation_split"}
        allowed_dataset = required_dataset | {"split_argument", "collate_fn"}
        if not required_dataset.issubset(dataset) or set(dataset) - allowed_dataset:
            raise ValueError(
                "dataset spec requires factory, kwargs, fit_split, validation_split; "
                "split_argument and collate_fn are optional"
            )
        for key in ("fit_split", "validation_split"):
            if not isinstance(dataset[key], str) or not dataset[key]:
                raise ValueError(f"dataset.{key} must be non-empty text")
        if not isinstance(dataset.get("split_argument", "split"), str):
            raise ValueError("dataset.split_argument must be text")
        if "collate_fn" in dataset and not isinstance(dataset["collate_fn"], str):
            raise ValueError("dataset.collate_fn must be a symbol reference")
        model = dict(value["model"])
        _validate_factory_spec(model, "model")
        preparer = value.get("batch_preparer")
        if preparer is not None:
            preparer = dict(preparer)
            _validate_factory_spec(preparer, "batch_preparer")
        return cls(
            run_dir=Path(value["run_dir"]).expanduser().resolve(),
            dataset=dataset,
            model=model,
            batch_preparer=preparer,
            training=TrainingOptionsV1.from_mapping(value["training"]),
            schema=schema,
        )

    def serializable(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_dir": str(self.run_dir),
            "dataset": dict(self.dataset),
            "model": dict(self.model),
            "batch_preparer": None if self.batch_preparer is None else dict(self.batch_preparer),
            "training": asdict(self.training),
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.serializable(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _validate_factory_spec(value: Mapping[str, Any], name: str) -> None:
    if set(value) != {"factory", "kwargs"}:
        raise ValueError(f"{name} spec requires exactly factory and kwargs")
    if not isinstance(value["factory"], str) or not isinstance(value["kwargs"], Mapping):
        raise ValueError(f"{name} factory must be text and kwargs must be a mapping")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is preferred to benchmark speed for the observer training.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _loader(
    dataset: Any,
    options: TrainingOptionsV1,
    *,
    epoch: int,
    training: bool,
    collate_fn: Callable[..., Any] | None = None,
) -> DataLoader:
    set_epoch = getattr(dataset, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)
    generator = torch.Generator()
    generator.manual_seed(options.seed + epoch * 1_000_003 + (0 if training else 500_009))
    return DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=training,
        num_workers=options.num_workers,
        pin_memory=options.pin_memory and torch.cuda.is_available(),
        persistent_workers=options.num_workers > 0,
        worker_init_fn=_worker_seed,
        generator=generator,
        drop_last=False,
        collate_fn=collate_fn,
    )


def _move_tensor(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: _move_tensor(child, device) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensor(child, device) for child in value)
    if isinstance(value, list):
        return [_move_tensor(child, device) for child in value]
    return value


def _prepare_batch(
    batch: Any,
    *,
    preparer: Callable[..., Mapping[str, Any]] | None,
    device: torch.device,
    training: bool,
) -> Mapping[str, Any]:
    if preparer is None:
        prepared = _move_tensor(batch, device)
    else:
        prepared = preparer(batch, device=device, training=training)
    if not isinstance(prepared, Mapping):
        raise TypeError("batch_preparer must return a mapping")
    missing = _REQUIRED_BATCH_KEYS - set(prepared)
    if missing:
        raise ValueError(f"prepared batch is missing keys: {sorted(missing)}")
    result = dict(prepared)
    for key in _REQUIRED_BATCH_KEYS:
        if not torch.is_tensor(result[key]):
            raise TypeError(f"prepared batch {key} must be a tensor")
        if result[key].device != device:
            result[key] = result[key].to(device, non_blocking=device.type == "cuda")
    return result


def _model_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping) and torch.is_tensor(output.get("log_scale")):
        return output["log_scale"]
    raise TypeError("uncertainty model must return a tensor or {'log_scale': tensor}")


class SeaRaftConsistencyBatchPreparerV1:
    """Lazy frozen SEA-RAFT teacher and 12-channel observable producer.

    The Sintel dataset emits RGB in [0,1].  SEA-RAFT receives float32 [0,255]
    while feature construction deliberately keeps [0,1] RGB.  Native and
    appearance-augmented predictions share the same crop lattice, so their
    endpoint discrepancy needs no geometric inverse transform in v0.
    """

    def __init__(
        self,
        *,
        iters: int | None = None,
        vendor_root: str | None = None,
        config_path: str | None = None,
        checkpoint: str | None = None,
    ) -> None:
        self.loader_kwargs = {
            "iters": iters,
            "vendor_root": None if vendor_root is None else Path(vendor_root),
            "config_path": None if config_path is None else Path(config_path),
            "checkpoint": None if checkpoint is None else Path(checkpoint),
        }
        self.predictor: Any | None = None
        self.load_report: Mapping[str, object] | None = None
        self.device: torch.device | None = None

    def _ensure_predictor(self, device: torch.device) -> None:
        if self.predictor is not None:
            if self.device != device:
                raise RuntimeError("SEA-RAFT preparer cannot change device after loading")
            return
        from .sea_raft_uncertainty_adapter import load_pinned_sea_raft_predictor

        self.predictor, self.load_report = load_pinned_sea_raft_predictor(
            device=str(device), **self.loader_kwargs,
        )
        self.device = device

    def __call__(
        self, batch: Mapping[str, Any], *, device: torch.device, training: bool,
    ) -> Mapping[str, torch.Tensor]:
        del training
        if not isinstance(batch, Mapping):
            raise TypeError("SEA-RAFT preparer expects a collated mapping")
        for key in ("native_frames", "augmented_frames", "augmentation_valid"):
            if not torch.is_tensor(batch.get(key)):
                raise TypeError(f"SEA-RAFT batch requires tensor {key}")
        native = batch["native_frames"].to(device, dtype=torch.float32, non_blocking=True)
        augmented = batch["augmented_frames"].to(
            device, dtype=torch.float32, non_blocking=True,
        )
        if native.ndim != 5 or native.shape[1:3] != (2, 3) or augmented.shape != native.shape:
            raise ValueError("native/augmented frames must have shape B,2,3,H,W")
        if bool((native < 0).any()) or bool((native > 1).any()):
            raise ValueError("native RGB must lie in [0,1]")
        if bool((augmented < 0).any()) or bool((augmented > 1).any()):
            raise ValueError("augmented RGB must lie in [0,1]")
        self._ensure_predictor(device)
        assert self.predictor is not None
        native_prediction = self.predictor(native[:, 0] * 255.0, native[:, 1] * 255.0)
        augmented_flow = self.predictor.predict_flow(
            augmented[:, 0] * 255.0, augmented[:, 1] * 255.0,
        ).detach().clone()
        # SEA-RAFT runs under inference_mode. Clone its outputs back to normal
        # tensors before the trainable head needs to save inputs for backward.
        native_flow = native_prediction.flow.detach().clone()
        native_risk = (
            native_prediction.uncertainty.upstream_heatmap_log_scale.detach().clone()
        )
        features, warp_valid = build_searaft_observable_features_v1(
            native[:, 0],
            native[:, 1],
            native_flow,
            native_risk,
        )
        features = features.clone()
        augmentation_valid = batch["augmentation_valid"].to(
            device, dtype=torch.bool, non_blocking=True,
        )
        if augmentation_valid.ndim == 3:
            augmentation_valid = augmentation_valid[:, None]
        if augmentation_valid.shape != native_flow[:, :1].shape:
            raise ValueError("augmentation_valid must have shape B,1,H,W")
        finite = torch.isfinite(native_flow).all(dim=1, keepdim=True)
        finite &= torch.isfinite(augmented_flow).all(dim=1, keepdim=True)
        # The observer feature itself contains a warp-validity channel, but
        # pixels whose native correspondence leaves the image must not become
        # teacher targets merely because both solver outputs are finite.
        common_valid = augmentation_valid & warp_valid & finite
        if not bool(common_valid.any()):
            raise ValueError("SEA-RAFT native/augmented flow has empty common support")
        return {
            "features": features,
            "reference_flow": native_flow,
            "restored_augmented_flow": augmented_flow,
            "valid_mask": common_valid,
        }


def make_sea_raft_consistency_preparer_v1(**kwargs: Any) -> SeaRaftConsistencyBatchPreparerV1:
    """Factory entry point suitable for the trainer JSON config."""

    return SeaRaftConsistencyBatchPreparerV1(**kwargs)


class _StopState:
    requested_signal: int | None = None

    def handler(self, signum: int, _frame: Any) -> None:
        if self.requested_signal is None:
            self.requested_signal = signum


class U0UncertaintyTrainerV1:
    """Stateful trainer with exact optimizer/RNG resume checkpoints."""

    def __init__(self, config: U0TrainerConfigV1, *, device: str = "auto"):
        self.config = config
        self.options = config.training
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch cannot access CUDA")
        _seed_everything(self.options.seed)
        split_argument = config.dataset.get("split_argument", "split")
        self.fit_dataset = _factory(
            {"factory": config.dataset["factory"], "kwargs": config.dataset["kwargs"]},
            extra={split_argument: config.dataset["fit_split"]},
        )
        self.validation_dataset = _factory(
            {"factory": config.dataset["factory"], "kwargs": config.dataset["kwargs"]},
            extra={split_argument: config.dataset["validation_split"]},
        )
        if len(self.fit_dataset) == 0 or len(self.validation_dataset) == 0:
            raise ValueError("fit and validation datasets must be non-empty")
        self.preparer = None if config.batch_preparer is None else _factory(config.batch_preparer)
        if self.preparer is not None and not callable(self.preparer):
            raise TypeError("batch_preparer factory must return a callable")
        self.collate_fn = None
        if "collate_fn" in config.dataset:
            self.collate_fn = _load_symbol(config.dataset["collate_fn"])
            if not callable(self.collate_fn):
                raise TypeError("dataset.collate_fn must resolve to a callable")
        self.model = _factory(config.model).to(self.device)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("uncertainty model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.options.learning_rate,
            weight_decay=self.options.weight_decay,
        )
        amp_enabled = self.options.amp and self.device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except AttributeError:  # pragma: no cover - old PyTorch compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.epoch = 0
        self.next_batch_index = 0
        self.global_step = 0
        self.best_validation_loss = float("inf")
        self.metrics_path = config.run_dir / "metrics.jsonl"
        self.latest_path = config.run_dir / "latest.pt"
        self.best_path = config.run_dir / "best.pt"

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema": TRAINER_SCHEMA_V1,
            "config_digest": self.config.digest,
            "config": self.config.serializable(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "next_batch_index": self.next_batch_index,
            "global_step": self.global_step,
            "best_validation_loss": self.best_validation_loss,
            "rng": _capture_rng(),
            "saved_utc": _utc_now(),
        }

    def save_checkpoint(self, path: Path | None = None) -> Path:
        path = path or self.latest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            torch.save(self._checkpoint_payload(), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def resume(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("schema") != TRAINER_SCHEMA_V1:
            raise ValueError("checkpoint trainer schema does not match")
        if checkpoint.get("config_digest") != self.config.digest:
            raise ValueError("checkpoint config differs from current config")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.epoch = int(checkpoint["epoch"])
        self.next_batch_index = int(checkpoint["next_batch_index"])
        self.global_step = int(checkpoint["global_step"])
        self.best_validation_loss = float(checkpoint["best_validation_loss"])
        _restore_rng(checkpoint["rng"])

    def _autocast(self):
        if hasattr(torch, "amp"):
            return torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled)
        if self.device.type == "cuda":  # pragma: no cover
            return torch.cuda.amp.autocast(enabled=self.amp_enabled)
        return nullcontext()

    def _loss(self, prepared: Mapping[str, Any]) -> torch.Tensor:
        log_scale = _model_output(self.model(prepared["features"]))
        return augmentation_consistency_laplace_loss_v1(
            log_scale,
            prepared["reference_flow"],
            prepared["restored_augmented_flow"],
            prepared["valid_mask"],
        )

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        loader = _loader(
            self.validation_dataset, self.options, epoch=epoch, training=False,
            collate_fn=self.collate_fn,
        )
        for batch in loader:
            prepared = _prepare_batch(
                batch, preparer=self.preparer, device=self.device, training=False,
            )
            with self._autocast():
                loss = self._loss(prepared)
            batch_size = int(prepared["features"].shape[0])
            total += float(loss.detach().cpu()) * batch_size
            count += batch_size
        if count == 0:
            raise RuntimeError("validation loader produced no samples")
        return total / count

    def train(self, *, stop: _StopState | None = None) -> int:
        stop = stop or _StopState()
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.config.run_dir / "resolved_config.json", self.config.serializable())
        _append_jsonl(self.metrics_path, {
            "event": "training_started", "utc": _utc_now(), "device": str(self.device),
            "amp_enabled": self.amp_enabled, "epoch": self.epoch,
            "global_step": self.global_step,
        })
        while self.epoch < self.options.epochs:
            epoch = self.epoch
            loader = _loader(
                self.fit_dataset, self.options, epoch=epoch, training=True,
                collate_fn=self.collate_fn,
            )
            if self.next_batch_index >= len(loader):
                self.epoch += 1
                self.next_batch_index = 0
                continue
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_batches = 0
            for batch_index, batch in enumerate(loader):
                if batch_index < self.next_batch_index:
                    continue
                prepared = _prepare_batch(
                    batch, preparer=self.preparer, device=self.device, training=True,
                )
                with self._autocast():
                    raw_loss = self._loss(prepared)
                    loss = raw_loss / self.options.gradient_accumulation_steps
                if not bool(torch.isfinite(raw_loss.detach())):
                    raise FloatingPointError("non-finite uncertainty training loss")
                self.scaler.scale(loss).backward()
                accumulated_loss += float(raw_loss.detach().cpu())
                accumulated_batches += 1
                last_batch = batch_index + 1 == len(loader)
                update = accumulated_batches == self.options.gradient_accumulation_steps or last_batch
                if not update:
                    continue
                if self.options.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.options.gradient_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self.next_batch_index = batch_index + 1
                _append_jsonl(self.metrics_path, {
                    "event": "train_step", "utc": _utc_now(), "epoch": epoch,
                    "next_batch_index": self.next_batch_index,
                    "global_step": self.global_step,
                    "loss": accumulated_loss / accumulated_batches,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                })
                accumulated_loss = 0.0
                accumulated_batches = 0
                if self.global_step % self.options.checkpoint_every_steps == 0:
                    self.save_checkpoint()
                if stop.requested_signal is not None:
                    self.save_checkpoint()
                    _append_jsonl(self.metrics_path, {
                        "event": "training_interrupted", "utc": _utc_now(),
                        "signal": stop.requested_signal, "epoch": epoch,
                        "next_batch_index": self.next_batch_index,
                        "global_step": self.global_step,
                    })
                    return 128 + stop.requested_signal
            self.epoch = epoch + 1
            self.next_batch_index = 0
            if self.epoch % self.options.validation_every_epochs == 0:
                validation_loss = self.validate(epoch)
                improved = validation_loss < self.best_validation_loss
                if improved:
                    self.best_validation_loss = validation_loss
                _append_jsonl(self.metrics_path, {
                    "event": "validation_epoch", "utc": _utc_now(), "epoch": epoch,
                    "global_step": self.global_step, "loss": validation_loss,
                    "best": improved,
                })
                self.save_checkpoint()
                if improved:
                    self.save_checkpoint(self.best_path)
            else:
                self.save_checkpoint()
        _append_jsonl(self.metrics_path, {
            "event": "training_completed", "utc": _utc_now(),
            "epochs": self.options.epochs, "global_step": self.global_step,
            "best_validation_loss": self.best_validation_loss,
        })
        return 0


def load_config(path: Path) -> U0TrainerConfigV1:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("trainer config root must be a JSON object")
    return U0TrainerConfigV1.from_mapping(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument(
        "--resume", default="auto",
        help="auto resumes run_dir/latest.pt; none starts fresh; otherwise checkpoint path",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    trainer = U0UncertaintyTrainerV1(config, device=args.device)
    if args.resume == "auto":
        if trainer.latest_path.exists():
            trainer.resume(trainer.latest_path)
    elif args.resume != "none":
        trainer.resume(Path(args.resume).expanduser().resolve())
    stop = _StopState()
    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, stop.handler)
    try:
        return trainer.train(stop=stop)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TRAINER_SCHEMA_V1",
    "SeaRaftConsistencyBatchPreparerV1",
    "TrainingOptionsV1",
    "U0TrainerConfigV1",
    "U0UncertaintyTrainerV1",
    "load_config",
    "main",
    "make_sea_raft_consistency_preparer_v1",
]
