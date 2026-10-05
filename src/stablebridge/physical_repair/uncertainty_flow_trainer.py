"""Resume-capable U1 and U2 uncertainty-guided flow training.

U1 freezes both SEA-RAFT and a completed U0 observer and trains only a
pointwise-bounded residual refiner against labelled flow.  U2 alternates an
observer-only consistency phase with a refiner-only supervised phase.  The
two paths never share gradients, and every checkpoint binds the immutable U0
and optional U1 source checkpoints used to initialise the run.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import importlib
import json
import math
import os
from pathlib import Path
import random
import signal
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .uncertainty_aware_flow import (
    BoundedUncertaintyFlowRefinerV1,
    UncertaintyEvidenceRoleV1,
    UncertaintyTrainingStageV1,
    augmentation_consistency_laplace_loss_v1,
)
from .uncertainty_features import build_searaft_observable_features_v1
from .uncertainty_refinement_training import (
    UncertaintyRefinerLossPolicyV1,
    uncertainty_refiner_loss_v1,
)
from .u0_uncertainty_trainer import TRAINER_SCHEMA_V1 as U0_TRAINER_SCHEMA_V1


FLOW_TRAINER_SCHEMA_V1 = "stablebridge-uncertainty-flow-trainer/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_symbol(reference: str) -> Any:
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("factory references must use module:symbol")
    module_name, symbol_name = reference.split(":", 1)
    return getattr(importlib.import_module(module_name), symbol_name)


def _validate_factory_spec(value: Mapping[str, Any], name: str) -> None:
    if set(value) != {"factory", "kwargs"}:
        raise ValueError(f"{name} must contain exactly factory and kwargs")
    if not isinstance(value["factory"], str) or not isinstance(value["kwargs"], Mapping):
        raise ValueError(f"{name} factory/kwargs are invalid")


def _factory(spec: Mapping[str, Any], *, extra: Mapping[str, Any] | None = None) -> Any:
    _validate_factory_spec(spec, "factory spec")
    kwargs = dict(spec["kwargs"])
    if extra:
        overlap = set(kwargs) & set(extra)
        if overlap:
            raise ValueError(f"factory kwargs duplicate injected fields: {sorted(overlap)}")
        kwargs.update(extra)
    return _load_symbol(spec["factory"])(**kwargs)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


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


def _worker_seed(worker_id: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class FlowTrainingOptionsV1:
    batch_size: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 4
    gradient_accumulation_steps: int = 1
    checkpoint_every_steps: int = 100
    validation_every_phase_epochs: int = 1
    amp: bool = True
    seed: int = 8603
    pin_memory: bool = True
    gradient_clip_norm: float | None = 1.0
    u1_epochs: int = 20
    u2_rounds: int = 5
    u2_observer_epochs_per_round: int = 1
    u2_refiner_epochs_per_round: int = 1

    def __post_init__(self) -> None:
        for name in (
            "batch_size", "gradient_accumulation_steps", "checkpoint_every_steps",
            "validation_every_phase_epochs", "u1_epochs", "u2_rounds",
            "u2_observer_epochs_per_round", "u2_refiner_epochs_per_round",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"training.{name} must be a positive integer")
        if isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ValueError("training.num_workers must be nonnegative")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("training.seed must be nonnegative")
        for name in ("learning_rate", "weight_decay"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"training.{name} is invalid")
        if self.gradient_clip_norm is not None:
            value = float(self.gradient_clip_norm)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("training.gradient_clip_norm must be positive or null")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FlowTrainingOptionsV1":
        extra = set(value) - set(cls.__dataclass_fields__)
        if extra:
            raise ValueError(f"unknown training fields: {sorted(extra)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class UncertaintyFlowTrainerConfigV1:
    run_dir: Path
    stage: UncertaintyTrainingStageV1
    dataset: Mapping[str, Any]
    observer_checkpoint: Path
    refiner: Mapping[str, Any]
    loss: UncertaintyRefinerLossPolicyV1
    training: FlowTrainingOptionsV1
    batch_preparer: Mapping[str, Any] | None = None
    refiner_checkpoint: Path | None = None
    require_completed_sources: bool = True
    schema: str = FLOW_TRAINER_SCHEMA_V1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UncertaintyFlowTrainerConfigV1":
        allowed = {
            "schema", "run_dir", "stage", "dataset", "observer_checkpoint",
            "refiner", "refiner_checkpoint", "loss", "batch_preparer", "training",
            "require_completed_sources",
        }
        required = {
            "run_dir", "stage", "dataset", "observer_checkpoint", "refiner", "loss", "training",
        }
        missing, extra = required - set(value), set(value) - allowed
        if missing or extra:
            raise ValueError(
                f"flow trainer config drift; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if value.get("schema", FLOW_TRAINER_SCHEMA_V1) != FLOW_TRAINER_SCHEMA_V1:
            raise ValueError("unsupported flow trainer schema")
        try:
            stage = UncertaintyTrainingStageV1(value["stage"])
        except (TypeError, ValueError) as exc:
            raise ValueError("flow trainer stage is invalid") from exc
        if stage is UncertaintyTrainingStageV1.UNCERTAINTY_HEAD_ONLY:
            raise ValueError("use the U0 trainer for uncertainty_head_only")
        dataset = dict(value["dataset"])
        dataset_required = {"factory", "kwargs", "fit_split", "validation_split"}
        dataset_allowed = dataset_required | {"split_argument", "collate_fn"}
        if not dataset_required.issubset(dataset) or set(dataset) - dataset_allowed:
            raise ValueError("dataset spec has missing or unknown fields")
        _validate_factory_spec(
            {"factory": dataset["factory"], "kwargs": dataset["kwargs"]}, "dataset"
        )
        refiner = dict(value["refiner"])
        _validate_factory_spec(refiner, "refiner")
        preparer = value.get("batch_preparer")
        if preparer is not None:
            preparer = dict(preparer)
            _validate_factory_spec(preparer, "batch_preparer")
        refiner_checkpoint = value.get("refiner_checkpoint")
        if stage is UncertaintyTrainingStageV1.ALTERNATING_DECOUPLED and not refiner_checkpoint:
            raise ValueError("U2 requires a qualified U1 initialization checkpoint")
        loss = UncertaintyRefinerLossPolicyV1(**dict(value["loss"]))
        require_completed_sources = value.get("require_completed_sources", True)
        if not isinstance(require_completed_sources, bool):
            raise ValueError("require_completed_sources must be bool")
        return cls(
            run_dir=Path(value["run_dir"]).expanduser().resolve(),
            stage=stage,
            dataset=dataset,
            observer_checkpoint=Path(value["observer_checkpoint"]).expanduser().resolve(),
            refiner=refiner,
            refiner_checkpoint=(
                None if refiner_checkpoint is None
                else Path(refiner_checkpoint).expanduser().resolve()
            ),
            loss=loss,
            batch_preparer=preparer,
            training=FlowTrainingOptionsV1.from_mapping(value["training"]),
            require_completed_sources=require_completed_sources,
        )

    def serializable(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_dir": str(self.run_dir),
            "stage": self.stage.value,
            "dataset": dict(self.dataset),
            "observer_checkpoint": str(self.observer_checkpoint),
            "refiner": dict(self.refiner),
            "refiner_checkpoint": (
                None if self.refiner_checkpoint is None else str(self.refiner_checkpoint)
            ),
            "loss": asdict(self.loss),
            "batch_preparer": (
                None if self.batch_preparer is None else dict(self.batch_preparer)
            ),
            "training": asdict(self.training),
            "require_completed_sources": self.require_completed_sources,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.serializable(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def make_bounded_uncertainty_flow_refiner_v1(
    *,
    feature_channels: int,
    hidden_channels: int = 32,
    maximum_update_px: float = 1.0,
    evidence_role: str = "base_flow",
    minimum_update_fraction: float = 0.0,
) -> BoundedUncertaintyFlowRefinerV1:
    """JSON-friendly factory for the typed bounded refiner."""

    return BoundedUncertaintyFlowRefinerV1(
        feature_channels=feature_channels,
        hidden_channels=hidden_channels,
        maximum_update_px=maximum_update_px,
        evidence_role=UncertaintyEvidenceRoleV1(evidence_role),
        minimum_update_fraction=minimum_update_fraction,
    )


class SeaRaftRefinementBatchPreparerV1:
    """Frozen SEA-RAFT features plus labelled targets for U1/U2."""

    def __init__(
        self,
        *,
        include_augmented_teacher: bool,
        iters: int | None = None,
        vendor_root: str | None = None,
        config_path: str | None = None,
        checkpoint: str | None = None,
    ) -> None:
        if not isinstance(include_augmented_teacher, bool):
            raise ValueError("include_augmented_teacher must be bool")
        self.include_augmented_teacher = include_augmented_teacher
        self.loader_kwargs = {
            "iters": iters,
            "vendor_root": None if vendor_root is None else Path(vendor_root),
            "config_path": None if config_path is None else Path(config_path),
            "checkpoint": None if checkpoint is None else Path(checkpoint),
        }
        self.predictor: Any | None = None
        self.device: torch.device | None = None

    def _ensure_predictor(self, device: torch.device) -> None:
        if self.predictor is not None:
            if self.device != device:
                raise RuntimeError("SEA-RAFT preparer cannot change device")
            return
        from .sea_raft_uncertainty_adapter import load_pinned_sea_raft_predictor

        self.predictor, _ = load_pinned_sea_raft_predictor(
            device=str(device), **self.loader_kwargs,
        )
        self.device = device

    def __call__(
        self, batch: Mapping[str, Any], *, device: torch.device, training: bool,
    ) -> Mapping[str, torch.Tensor]:
        del training
        required = ("native_frames", "ground_truth_flow", "ground_truth_valid")
        if any(not torch.is_tensor(batch.get(key)) for key in required):
            raise TypeError("U1/U2 batch needs native RGB and ground-truth flow tensors")
        native = batch["native_frames"].to(
            device, dtype=torch.float32, non_blocking=True,
        )
        truth = batch["ground_truth_flow"].to(
            device, dtype=torch.float32, non_blocking=True,
        )
        valid = batch["ground_truth_valid"].to(
            device, dtype=torch.bool, non_blocking=True,
        )
        if native.ndim != 5 or native.shape[1:3] != (2, 3):
            raise ValueError("native frames must have shape B,2,3,H,W")
        if truth.shape != (native.shape[0], 2, native.shape[3], native.shape[4]):
            raise ValueError("ground-truth flow must match native frames")
        if valid.shape == truth[:, 0].shape:
            valid = valid[:, None]
        if valid.shape != truth[:, :1].shape:
            raise ValueError("ground-truth validity must have shape B,1,H,W")
        self._ensure_predictor(device)
        assert self.predictor is not None
        prediction = self.predictor(native[:, 0] * 255.0, native[:, 1] * 255.0)
        base = prediction.flow.detach().clone()
        upstream_risk = prediction.uncertainty.upstream_heatmap_log_scale.detach().clone()
        features, warp_valid = build_searaft_observable_features_v1(
            native[:, 0], native[:, 1], base, upstream_risk,
        )
        valid = valid & torch.isfinite(truth).all(dim=1, keepdim=True)
        valid &= torch.isfinite(base).all(dim=1, keepdim=True)
        if not bool(valid.any()):
            raise ValueError("U1/U2 batch has empty labelled support")
        result = {
            "features": features.clone(),
            "base_flow": base,
            "ground_truth_flow": truth,
            "valid_mask": valid,
        }
        if self.include_augmented_teacher:
            if not torch.is_tensor(batch.get("augmented_frames")) or not torch.is_tensor(
                batch.get("augmentation_valid")
            ):
                raise TypeError("U2 observer phase needs augmented frames and validity")
            augmented = batch["augmented_frames"].to(
                device, dtype=torch.float32, non_blocking=True,
            )
            augmented_valid = batch["augmentation_valid"].to(
                device, dtype=torch.bool, non_blocking=True,
            )
            if augmented.shape != native.shape:
                raise ValueError("augmented frames must match native frames")
            if augmented_valid.ndim == 3:
                augmented_valid = augmented_valid[:, None]
            if augmented_valid.shape != valid.shape:
                raise ValueError("augmentation validity must have shape B,1,H,W")
            augmented_flow = self.predictor.predict_flow(
                augmented[:, 0] * 255.0, augmented[:, 1] * 255.0,
            ).detach().clone()
            consistency_valid = augmented_valid & warp_valid
            consistency_valid &= torch.isfinite(base).all(dim=1, keepdim=True)
            consistency_valid &= torch.isfinite(augmented_flow).all(dim=1, keepdim=True)
            if not bool(consistency_valid.any()):
                raise ValueError("U2 consistency support is empty")
            result.update({
                "reference_flow": base,
                "restored_augmented_flow": augmented_flow,
                "consistency_valid_mask": consistency_valid,
            })
        return result


def make_sea_raft_refinement_preparer_v1(
    **kwargs: Any,
) -> SeaRaftRefinementBatchPreparerV1:
    return SeaRaftRefinementBatchPreparerV1(**kwargs)


def _observer_from_u0_checkpoint(
    path: Path, device: torch.device,
) -> tuple[torch.nn.Module, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(payload), map_location=device, weights_only=False)
    if checkpoint.get("schema") != U0_TRAINER_SCHEMA_V1:
        raise ValueError("observer checkpoint is not a U0 trainer checkpoint")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("model"), Mapping):
        raise ValueError("U0 checkpoint does not embed its model factory")
    model = _factory(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, hashlib.sha256(payload).hexdigest()


def _require_completed_run(checkpoint_path: Path, *, stage: str) -> None:
    metrics_path = checkpoint_path.parent / "metrics.jsonl"
    if not metrics_path.is_file():
        raise RuntimeError(f"{stage} source run has no metrics.jsonl completion record")
    lifecycle_event: str | None = None
    with metrics_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{stage} source metrics are malformed") from exc
            if row.get("event") in {
                "training_started", "training_interrupted", "training_completed",
            }:
                lifecycle_event = str(row["event"])
    if lifecycle_event != "training_completed":
        raise RuntimeError(f"{stage} source training has not completed")


def _load_refiner_initialization(
    path: Path, refiner: torch.nn.Module, device: torch.device,
) -> tuple[str, Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(payload), map_location=device, weights_only=False)
    if checkpoint.get("schema") != FLOW_TRAINER_SCHEMA_V1:
        raise ValueError("refiner checkpoint is not a U1/U2 trainer checkpoint")
    source_stage = checkpoint.get("stage")
    if source_stage not in {
        UncertaintyTrainingStageV1.FROZEN_UNCERTAINTY_REFINER.value,
        UncertaintyTrainingStageV1.ALTERNATING_DECOUPLED.value,
    }:
        raise ValueError("refiner checkpoint has an invalid source stage")
    refiner.load_state_dict(checkpoint["refiner"])
    source_bindings = checkpoint.get("source_bindings")
    if not isinstance(source_bindings, Mapping):
        raise ValueError("refiner checkpoint does not bind its frozen sources")
    return hashlib.sha256(payload).hexdigest(), dict(source_bindings)


@dataclass(frozen=True)
class _PhaseV1:
    kind: str
    round_index: int
    epoch_in_round: int
    round_complete: bool


class _StopState:
    requested_signal: int | None = None

    def handler(self, signum: int, _frame: Any) -> None:
        if self.requested_signal is None:
            self.requested_signal = signum


class UncertaintyFlowTrainerV1:
    """Train U1 or alternate U2 with exact phase/RNG resume."""

    def __init__(self, config: UncertaintyFlowTrainerConfigV1, *, device: str = "auto"):
        self.config = config
        self.options = config.training
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        _seed_everything(self.options.seed)
        if config.require_completed_sources:
            _require_completed_run(config.observer_checkpoint, stage="U0 observer")
            if config.refiner_checkpoint is not None:
                _require_completed_run(config.refiner_checkpoint, stage="U1 refiner")
        split_argument = config.dataset.get("split_argument", "split")
        dataset_spec = {
            "factory": config.dataset["factory"], "kwargs": config.dataset["kwargs"]
        }
        self.fit_dataset = _factory(
            dataset_spec, extra={split_argument: config.dataset["fit_split"]},
        )
        self.validation_dataset = _factory(
            dataset_spec, extra={split_argument: config.dataset["validation_split"]},
        )
        if len(self.fit_dataset) == 0 or len(self.validation_dataset) == 0:
            raise ValueError("fit and validation datasets must be nonempty")
        self.collate_fn = None
        if "collate_fn" in config.dataset:
            self.collate_fn = _load_symbol(config.dataset["collate_fn"])
        self.preparer = None if config.batch_preparer is None else _factory(config.batch_preparer)
        if self.preparer is not None and not callable(self.preparer):
            raise TypeError("batch_preparer must be callable")
        self.observer, observer_hash = _observer_from_u0_checkpoint(
            config.observer_checkpoint, self.device,
        )
        self.refiner = _factory(config.refiner).to(self.device)
        if not isinstance(self.refiner, torch.nn.Module):
            raise TypeError("refiner factory must return a torch module")
        refiner_hash = None
        refiner_source_bindings: Mapping[str, Any] | None = None
        if config.refiner_checkpoint is not None:
            refiner_hash, refiner_source_bindings = _load_refiner_initialization(
                config.refiner_checkpoint, self.refiner, self.device,
            )
            if (
                refiner_source_bindings.get("observer_checkpoint_sha256")
                != observer_hash
            ):
                raise ValueError(
                    "U1 refiner initialization was trained with a different U0 observer"
                )
        self.source_bindings = {
            "observer_checkpoint": str(config.observer_checkpoint),
            "observer_checkpoint_sha256": observer_hash,
            "refiner_checkpoint": (
                None if config.refiner_checkpoint is None else str(config.refiner_checkpoint)
            ),
            "refiner_checkpoint_sha256": refiner_hash,
            "refiner_source_bindings": refiner_source_bindings,
        }
        self.observer_optimizer = torch.optim.AdamW(
            self.observer.parameters(), lr=self.options.learning_rate,
            weight_decay=self.options.weight_decay,
        )
        self.refiner_optimizer = torch.optim.AdamW(
            self.refiner.parameters(), lr=self.options.learning_rate,
            weight_decay=self.options.weight_decay,
        )
        amp_enabled = self.options.amp and self.device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except AttributeError:  # pragma: no cover
            self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.schedule = self._build_schedule()
        self.phase_index = 0
        self.next_batch_index = 0
        self.global_step = 0
        self.observer_steps = 0
        self.refiner_steps = 0
        self.best_observer_validation = float("inf")
        self.best_refiner_validation = float("inf")
        self.metrics_path = config.run_dir / "metrics.jsonl"
        self.latest_path = config.run_dir / "latest.pt"
        self.best_observer_path = config.run_dir / "best_observer.pt"
        self.best_refiner_path = config.run_dir / "best_refiner.pt"

    def _build_schedule(self) -> tuple[_PhaseV1, ...]:
        if self.config.stage is UncertaintyTrainingStageV1.FROZEN_UNCERTAINTY_REFINER:
            return tuple(
                _PhaseV1("refiner", 0, epoch, epoch + 1 == self.options.u1_epochs)
                for epoch in range(self.options.u1_epochs)
            )
        result = []
        for round_index in range(self.options.u2_rounds):
            for epoch in range(self.options.u2_observer_epochs_per_round):
                result.append(_PhaseV1("observer", round_index, epoch, False))
            for epoch in range(self.options.u2_refiner_epochs_per_round):
                result.append(_PhaseV1(
                    "refiner", round_index, epoch,
                    epoch + 1 == self.options.u2_refiner_epochs_per_round,
                ))
        return tuple(result)

    def _loader(self, dataset: Any, *, phase_index: int, training: bool) -> DataLoader:
        setter = getattr(dataset, "set_epoch", None)
        if callable(setter):
            setter(phase_index)
        generator = torch.Generator()
        generator.manual_seed(
            self.options.seed + phase_index * 1_000_003 + (0 if training else 500_009)
        )
        return DataLoader(
            dataset,
            batch_size=self.options.batch_size,
            shuffle=training,
            num_workers=self.options.num_workers,
            pin_memory=self.options.pin_memory,
            persistent_workers=False,
            generator=generator,
            worker_init_fn=_worker_seed,
            collate_fn=self.collate_fn,
        )

    def _prepare(self, batch: Any, *, training: bool) -> Mapping[str, Any]:
        prepared = (
            _move(batch, self.device)
            if self.preparer is None
            else self.preparer(batch, device=self.device, training=training)
        )
        if not isinstance(prepared, Mapping):
            raise TypeError("prepared batch must be a mapping")
        required = {"features", "base_flow", "ground_truth_flow", "valid_mask"}
        missing = required - set(prepared)
        if missing:
            raise ValueError(f"prepared U1/U2 batch is missing {sorted(missing)}")
        return {key: _move(value, self.device) for key, value in prepared.items()}

    def _autocast(self):
        if hasattr(torch, "amp"):
            return torch.amp.autocast(
                device_type=self.device.type, enabled=self.amp_enabled,
            )
        if self.device.type == "cuda":  # pragma: no cover
            return torch.cuda.amp.autocast(enabled=self.amp_enabled)
        return nullcontext()

    def _set_phase(self, kind: str) -> tuple[torch.optim.Optimizer, Sequence[torch.nn.Parameter]]:
        observer_trainable = kind == "observer"
        self.observer.train(observer_trainable)
        self.refiner.train(not observer_trainable)
        self.observer.requires_grad_(observer_trainable)
        self.refiner.requires_grad_(not observer_trainable)
        if observer_trainable:
            return self.observer_optimizer, tuple(self.observer.parameters())
        return self.refiner_optimizer, tuple(self.refiner.parameters())

    def _observer_loss(self, prepared: Mapping[str, Any]) -> torch.Tensor:
        required = {"reference_flow", "restored_augmented_flow", "consistency_valid_mask"}
        missing = required - set(prepared)
        if missing:
            raise ValueError(f"U2 observer phase is missing {sorted(missing)}")
        log_scale = self.observer(prepared["features"])
        return augmentation_consistency_laplace_loss_v1(
            log_scale,
            prepared["reference_flow"],
            prepared["restored_augmented_flow"],
            prepared["consistency_valid_mask"],
        )

    def _refiner_loss(self, prepared: Mapping[str, Any]):
        with torch.no_grad():
            # U0 predicts log Laplace scale; log variance is exactly twice it.
            log_variance = 2.0 * self.observer(prepared["features"])
        refined = self.refiner(
            prepared["features"], prepared["base_flow"],
            log_variance.detach(), prepared["valid_mask"],
        )
        loss = uncertainty_refiner_loss_v1(
            refined,
            prepared["base_flow"],
            prepared["ground_truth_flow"],
            prepared["valid_mask"],
            policy=self.config.loss,
        )
        return loss, refined

    @staticmethod
    def _epe(flow: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if valid.ndim == 4:
            valid = valid[:, 0]
        return torch.linalg.vector_norm(flow - truth, dim=1)[valid].mean()

    def validate(self, phase_index: int) -> dict[str, float | None]:
        self.observer.eval()
        self.refiner.eval()
        totals: dict[str, float] = {
            "refiner_total": 0.0, "task": 0.0, "harm": 0.0,
            "anchor": 0.0, "smoothness": 0.0, "base_epe": 0.0,
            "refined_epe": 0.0, "observer_consistency": 0.0,
        }
        count = 0
        observer_count = 0
        with torch.no_grad():
            for batch in self._loader(
                self.validation_dataset, phase_index=phase_index, training=False,
            ):
                prepared = self._prepare(batch, training=False)
                with self._autocast():
                    loss, refined = self._refiner_loss(prepared)
                    base_epe = self._epe(
                        prepared["base_flow"], prepared["ground_truth_flow"],
                        prepared["valid_mask"],
                    )
                    refined_epe = self._epe(
                        refined, prepared["ground_truth_flow"], prepared["valid_mask"],
                    )
                    observer_loss = None
                    if {
                        "reference_flow", "restored_augmented_flow",
                        "consistency_valid_mask",
                    }.issubset(prepared):
                        observer_loss = self._observer_loss(prepared)
                batch_size = int(prepared["features"].shape[0])
                count += batch_size
                for name in ("total", "task", "harm", "anchor", "smoothness"):
                    key = "refiner_total" if name == "total" else name
                    totals[key] += float(getattr(loss, name).detach().cpu()) * batch_size
                totals["base_epe"] += float(base_epe.detach().cpu()) * batch_size
                totals["refined_epe"] += float(refined_epe.detach().cpu()) * batch_size
                if observer_loss is not None:
                    totals["observer_consistency"] += (
                        float(observer_loss.detach().cpu()) * batch_size
                    )
                    observer_count += batch_size
        if count == 0:
            raise RuntimeError("validation loader produced no samples")
        result = {name: value / count for name, value in totals.items()}
        result["observer_consistency"] = (
            None if observer_count == 0
            else totals["observer_consistency"] / observer_count
        )
        return result

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema": FLOW_TRAINER_SCHEMA_V1,
            "stage": self.config.stage.value,
            "config_digest": self.config.digest,
            "config": self.config.serializable(),
            "source_bindings": self.source_bindings,
            "observer": self.observer.state_dict(),
            "refiner": self.refiner.state_dict(),
            "observer_optimizer": self.observer_optimizer.state_dict(),
            "refiner_optimizer": self.refiner_optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "phase_index": self.phase_index,
            "next_batch_index": self.next_batch_index,
            "global_step": self.global_step,
            "observer_steps": self.observer_steps,
            "refiner_steps": self.refiner_steps,
            "best_observer_validation": self.best_observer_validation,
            "best_refiner_validation": self.best_refiner_validation,
            "rng": _capture_rng(),
            "saved_utc": _utc_now(),
        }

    def save_checkpoint(self, path: Path | None = None) -> Path:
        target = self.latest_path if path is None else path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        try:
            torch.save(self._checkpoint_payload(), temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def resume(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("schema") != FLOW_TRAINER_SCHEMA_V1:
            raise ValueError("resume checkpoint schema differs")
        if checkpoint.get("stage") != self.config.stage.value:
            raise ValueError("resume checkpoint training stage differs")
        if checkpoint.get("config_digest") != self.config.digest:
            raise ValueError("resume checkpoint config differs")
        if checkpoint.get("source_bindings") != self.source_bindings:
            raise ValueError("frozen source checkpoint identity drifted")
        self.observer.load_state_dict(checkpoint["observer"])
        self.refiner.load_state_dict(checkpoint["refiner"])
        self.observer_optimizer.load_state_dict(checkpoint["observer_optimizer"])
        self.refiner_optimizer.load_state_dict(checkpoint["refiner_optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        for name in (
            "phase_index", "next_batch_index", "global_step",
            "observer_steps", "refiner_steps",
        ):
            setattr(self, name, int(checkpoint[name]))
        self.best_observer_validation = float(checkpoint["best_observer_validation"])
        self.best_refiner_validation = float(checkpoint["best_refiner_validation"])
        _restore_rng(checkpoint["rng"])

    def train(self, *, stop: _StopState | None = None) -> int:
        stop = stop or _StopState()
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        resolved = self.config.serializable()
        resolved["source_bindings"] = self.source_bindings
        _atomic_json(self.config.run_dir / "resolved_config.json", resolved)
        _append_jsonl(self.metrics_path, {
            "event": "training_started", "utc": _utc_now(),
            "stage": self.config.stage.value, "device": str(self.device),
            "amp_enabled": self.amp_enabled, "phase_index": self.phase_index,
            "global_step": self.global_step, **self.source_bindings,
        })
        while self.phase_index < len(self.schedule):
            phase = self.schedule[self.phase_index]
            optimizer, parameters = self._set_phase(phase.kind)
            optimizer.zero_grad(set_to_none=True)
            loader = self._loader(
                self.fit_dataset, phase_index=self.phase_index, training=True,
            )
            accumulated = 0.0
            accumulated_batches = 0
            for batch_index, batch in enumerate(loader):
                if batch_index < self.next_batch_index:
                    continue
                prepared = self._prepare(batch, training=True)
                with self._autocast():
                    if phase.kind == "observer":
                        raw_loss = self._observer_loss(prepared)
                        components = {"observer_consistency": float(raw_loss.detach().cpu())}
                    else:
                        refiner_loss, _ = self._refiner_loss(prepared)
                        raw_loss = refiner_loss.total
                        components = {
                            name: float(getattr(refiner_loss, name).detach().cpu())
                            for name in ("task", "harm", "anchor", "smoothness")
                        }
                    loss = raw_loss / self.options.gradient_accumulation_steps
                if not bool(torch.isfinite(raw_loss.detach())):
                    raise FloatingPointError("non-finite U1/U2 loss")
                self.scaler.scale(loss).backward()
                accumulated += float(raw_loss.detach().cpu())
                accumulated_batches += 1
                last_batch = batch_index + 1 == len(loader)
                update = (
                    accumulated_batches == self.options.gradient_accumulation_steps
                    or last_batch
                )
                if not update:
                    continue
                if self.options.gradient_clip_norm is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        parameters, self.options.gradient_clip_norm,
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if phase.kind == "observer":
                    self.observer_steps += 1
                else:
                    self.refiner_steps += 1
                self.next_batch_index = batch_index + 1
                _append_jsonl(self.metrics_path, {
                    "event": "train_step", "utc": _utc_now(),
                    "phase": phase.kind, "phase_index": self.phase_index,
                    "round_index": phase.round_index,
                    "epoch_in_round": phase.epoch_in_round,
                    "next_batch_index": self.next_batch_index,
                    "global_step": self.global_step,
                    "loss": accumulated / accumulated_batches,
                    **components,
                })
                accumulated = 0.0
                accumulated_batches = 0
                if self.global_step % self.options.checkpoint_every_steps == 0:
                    self.save_checkpoint()
                if stop.requested_signal is not None:
                    self.save_checkpoint()
                    _append_jsonl(self.metrics_path, {
                        "event": "training_interrupted", "utc": _utc_now(),
                        "signal": stop.requested_signal,
                        "phase_index": self.phase_index,
                        "next_batch_index": self.next_batch_index,
                        "global_step": self.global_step,
                    })
                    return 128 + stop.requested_signal
            completed_phase_index = self.phase_index
            self.phase_index += 1
            self.next_batch_index = 0
            next_phase = (
                None if self.phase_index == len(self.schedule)
                else self.schedule[self.phase_index]
            )
            phase_kind_complete = (
                next_phase is None
                or next_phase.kind != phase.kind
                or next_phase.round_index != phase.round_index
            )
            should_validate = phase_kind_complete or (
                (phase.epoch_in_round + 1)
                % self.options.validation_every_phase_epochs == 0
            )
            if should_validate:
                metrics = self.validate(completed_phase_index)
                observer_value = metrics["observer_consistency"]
                observer_improved = (
                    observer_value is not None
                    and math.isfinite(observer_value)
                    and observer_value < self.best_observer_validation
                )
                refiner_improved = metrics["refiner_total"] < self.best_refiner_validation
                if observer_improved:
                    self.best_observer_validation = observer_value
                if refiner_improved:
                    self.best_refiner_validation = metrics["refiner_total"]
                _append_jsonl(self.metrics_path, {
                    "event": "validation_phase", "utc": _utc_now(),
                    "phase": phase.kind, "phase_index": completed_phase_index,
                    "round_index": phase.round_index,
                    "observer_best": observer_improved,
                    "refiner_best": refiner_improved,
                    **metrics,
                })
                self.save_checkpoint()
                if observer_improved:
                    self.save_checkpoint(self.best_observer_path)
                if refiner_improved:
                    self.save_checkpoint(self.best_refiner_path)
            else:
                self.save_checkpoint()
            if phase.round_complete:
                self.save_checkpoint(
                    self.config.run_dir / f"round_{phase.round_index:02d}.pt"
                )
        _append_jsonl(self.metrics_path, {
            "event": "training_completed", "utc": _utc_now(),
            "stage": self.config.stage.value, "global_step": self.global_step,
            "observer_steps": self.observer_steps,
            "refiner_steps": self.refiner_steps,
            "best_observer_validation": (
                self.best_observer_validation
                if math.isfinite(self.best_observer_validation) else None
            ),
            "best_refiner_validation": (
                self.best_refiner_validation
                if math.isfinite(self.best_refiner_validation) else None
            ),
        })
        return 0


def load_config(path: Path) -> UncertaintyFlowTrainerConfigV1:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("flow trainer config root must be an object")
    return UncertaintyFlowTrainerConfigV1.from_mapping(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default="auto", help="auto, none, or checkpoint path")
    args = parser.parse_args(argv)
    config = load_config(args.config.expanduser().resolve())
    trainer = UncertaintyFlowTrainerV1(config, device=args.device)
    if args.resume == "auto":
        if trainer.latest_path.exists():
            trainer.resume(trainer.latest_path)
    elif args.resume != "none":
        trainer.resume(Path(args.resume).expanduser().resolve())
    stop = _StopState()
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, stop.handler)
    try:
        return trainer.train(stop=stop)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


__all__ = [
    "FLOW_TRAINER_SCHEMA_V1",
    "FlowTrainingOptionsV1",
    "SeaRaftRefinementBatchPreparerV1",
    "UncertaintyFlowTrainerConfigV1",
    "UncertaintyFlowTrainerV1",
    "load_config",
    "main",
    "make_bounded_uncertainty_flow_refiner_v1",
    "make_sea_raft_refinement_preparer_v1",
]
