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
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .candidate_action_bank import (
    OPTICAL_FLOW_CAPACITY_ACTION_IDS,
    OPTICAL_FLOW_CAPACITY_BANK,
    OPTICAL_FLOW_CAPACITY_BANK_HASH,
    OPTICAL_FLOW_CAPACITY_SOURCE_MANIFEST_SHA256,
    OPTICAL_NATIVE_ACTION_ID,
)
from .cuda_memory_guard import (
    CudaMemoryReservationPolicyV1,
    CudaMemoryReservationV1,
)
from .integrated_uncertainty_flow import (
    DecoupledFlowLossPolicyV2,
    RecurrentUncertaintyVariantV2,
    TrainableScopeV2,
    decoupled_uncertainty_flow_loss_v2,
    transform_flow_affine_v2,
)
from .uncertainty_evaluation import evaluate_log_scale_uncertainty_v1
from .uncertainty_features import build_searaft_observable_features_v1
from .uncertainty_refinement_training import (
    UncertaintyRefinerLossPolicyV1,
    uncertainty_refiner_loss_v1,
)
from .u2flow_bidirectional_fusion import (
    U2FlowBidirectionalFusionPolicyV1,
    fuse_u2flow_triplet_prediction_v1,
)
from .sam_semantic_smoothness import (
    SamHomographySmoothnessPolicyV1,
    sam_homography_smoothness_loss_v1,
)
from .sam_semantic_augmentation import (
    SamSemanticAugmentationPolicyV1,
    SamSemanticObjectCacheV1,
    compose_sam_semantic_augmentation_v1,
    sam_semantic_sequence_loss_v1,
    select_sam_object_masks_v1,
)


INTEGRATED_TRAINER_SCHEMA_V2 = "stablebridge-integrated-uncertainty-trainer/v2"


@dataclass(frozen=True)
class _ConservativeU1LossV2:
    total: torch.Tensor
    task: torch.Tensor
    augmentation: torch.Tensor
    uncertainty: torch.Tensor
    harm: torch.Tensor
    anchor: torch.Tensor
    smoothness: torch.Tensor
    sam_homography: torch.Tensor | None = None
    sam_candidate_regions: int = 0
    sam_fitted_regions: int = 0
    sam_semantic: torch.Tensor | None = None
    sam_semantic_objects: int = 0
    sam_semantic_object_pixels: int = 0


@dataclass(frozen=True)
class _IntegratedLossV2:
    total: torch.Tensor
    task: torch.Tensor
    augmentation: torch.Tensor
    uncertainty: torch.Tensor
    sam_homography: torch.Tensor
    sam_candidate_regions: int
    sam_fitted_regions: int
    sam_semantic: torch.Tensor
    sam_semantic_objects: int
    sam_semantic_object_pixels: int


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ActionBankTrainingPolicyV1:
    """Action schedule with immutable E292 lineage.

    Legacy ``cycle`` uses one action for a whole loader epoch.  The preferred
    ``balanced_batches`` mode interleaves the whole bank within each epoch while
    keeping every batch homogeneous, which is required by recurrent-iteration
    arms. Validation defaults to native so model selection remains comparable.
    """

    mode: str = "native_only"
    action_ids: tuple[str, ...] = (OPTICAL_NATIVE_ACTION_ID,)
    validation_action_id: str = OPTICAL_NATIVE_ACTION_ID
    bank_hash: str = OPTICAL_FLOW_CAPACITY_BANK_HASH
    source_manifest_sha256: str = OPTICAL_FLOW_CAPACITY_SOURCE_MANIFEST_SHA256

    def __post_init__(self) -> None:
        if self.mode not in {
            "native_only", "fixed", "cycle", "balanced_batches",
        }:
            raise ValueError(
                "action-bank training mode must be native_only, fixed, cycle, "
                "or balanced_batches"
            )
        actions = tuple(self.action_ids)
        allowed = {OPTICAL_NATIVE_ACTION_ID, *OPTICAL_FLOW_CAPACITY_ACTION_IDS}
        if not actions or len(actions) != len(set(actions)) or not set(actions) <= allowed:
            raise ValueError("action-bank schedule must contain unique E292/native actions")
        if self.mode == "native_only" and actions != (OPTICAL_NATIVE_ACTION_ID,):
            raise ValueError("native_only schedule must contain only native")
        if self.mode == "fixed" and len(actions) != 1:
            raise ValueError("fixed action-bank schedule must contain one action")
        if self.validation_action_id not in allowed:
            raise ValueError("validation action must be native or an E292 anchor")
        if self.bank_hash != OPTICAL_FLOW_CAPACITY_BANK_HASH:
            raise ValueError("E292 action-bank hash drift")
        if self.source_manifest_sha256 != OPTICAL_FLOW_CAPACITY_SOURCE_MANIFEST_SHA256:
            raise ValueError("E292 source-manifest hash drift")
        object.__setattr__(self, "action_ids", actions)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ActionBankTrainingPolicyV1":
        if value is None:
            return cls()
        expected = {
            "mode", "action_ids", "validation_action_id", "bank_hash",
            "source_manifest_sha256",
        }
        if set(value) != expected:
            raise ValueError("action-bank training policy fields drift")
        return cls(
            mode=str(value["mode"]),
            action_ids=tuple(str(item) for item in value["action_ids"]),
            validation_action_id=str(value["validation_action_id"]),
            bank_hash=str(value["bank_hash"]),
            source_manifest_sha256=str(value["source_manifest_sha256"]),
        )

    def action_for_epoch(self, epoch: int) -> str:
        return self.action_ids[epoch % len(self.action_ids)]


@dataclass(frozen=True)
class AlternatingU2ScheduleV2:
    """Epoch-level U2 schedule with mutually exclusive gradient phases."""

    mode: str = "alternating_decoupled"
    observer_epochs_per_round: int = 1
    flow_epochs_per_round: int = 1

    def __post_init__(self) -> None:
        if self.mode != "alternating_decoupled":
            raise ValueError("U2 schedule mode must be alternating_decoupled")
        for name in ("observer_epochs_per_round", "flow_epochs_per_round"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "AlternatingU2ScheduleV2 | None":
        if value is None:
            return None
        expected = {"mode", "observer_epochs_per_round", "flow_epochs_per_round"}
        if set(value) != expected:
            raise ValueError("U2 schedule fields drift")
        return cls(
            mode=str(value["mode"]),
            observer_epochs_per_round=int(value["observer_epochs_per_round"]),
            flow_epochs_per_round=int(value["flow_epochs_per_round"]),
        )

    @property
    def epochs_per_round(self) -> int:
        return self.observer_epochs_per_round + self.flow_epochs_per_round

    def phase_for_epoch(self, epoch: int) -> str:
        position = epoch % self.epochs_per_round
        return (
            "observer"
            if position < self.observer_epochs_per_round
            else "flow"
        )

    def round_for_epoch(self, epoch: int) -> int:
        return epoch // self.epochs_per_round


@dataclass(frozen=True)
class JointDecoupledU2ScheduleV2:
    """Joint flow/uncertainty learning with loss-level gradient decoupling.

    Both objectives are evaluated for every batch and optimized by one
    backward pass.  Decoupling is supplied by the detached consistency teacher,
    the detached discrepancy in the uncertainty likelihood, and the detached
    uncertainty input to the flow-refinement branch.
    """

    mode: str = "joint_decoupled"

    def __post_init__(self) -> None:
        if self.mode != "joint_decoupled":
            raise ValueError("joint U2 schedule mode must be joint_decoupled")

    @property
    def epochs_per_round(self) -> int:
        return 1

    def phase_for_epoch(self, epoch: int) -> str:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        return "joint"

    def round_for_epoch(self, epoch: int) -> int:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        return epoch


def _u2_schedule_from_mapping(
    value: Mapping[str, Any] | None,
) -> AlternatingU2ScheduleV2 | JointDecoupledU2ScheduleV2 | None:
    if value is None:
        return None
    mode = value.get("mode")
    if mode == "alternating_decoupled":
        return AlternatingU2ScheduleV2.from_mapping(value)
    if mode == "joint_decoupled":
        if set(value) != {"mode"}:
            raise ValueError("joint U2 schedule needs only the mode field")
        return JointDecoupledU2ScheduleV2(mode=str(mode))
    raise ValueError(
        "U2 schedule mode must be alternating_decoupled or joint_decoupled"
    )


class _ActionScheduledDatasetV1(Dataset):
    """Bind an action to an index without sharing mutable state across workers."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key: tuple[int, str]):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("scheduled dataset index must contain index and action id")
        index, action_id = key
        setter = getattr(self.dataset, "set_action_id", None)
        if not callable(setter):
            raise ValueError(
                "balanced action-bank training requires dataset.set_action_id"
            )
        setter(str(action_id))
        return self.dataset[int(index)]


class _BalancedActionBatchSamplerV1:
    """Deterministic, resume-stable, batch-homogeneous action interleaving."""

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        action_ids: tuple[str, ...],
        seed: int,
        epoch: int,
    ) -> None:
        if dataset_size <= 0 or batch_size <= 0 or not action_ids:
            raise ValueError("balanced action sampler inputs must be nonempty")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.action_ids = tuple(action_ids)
        self.seed = int(seed)
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        sample_generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            self.dataset_size, generator=sample_generator,
        ).tolist()
        action_generator = torch.Generator().manual_seed(self.seed + 7919)
        action_order = [
            self.action_ids[index]
            for index in torch.randperm(
                len(self.action_ids), generator=action_generator,
            ).tolist()
        ]
        # The offset also covers all arms across successive epochs when a tiny
        # dataset has fewer batches than actions.
        offset = (self.epoch * len(self)) % len(action_order)
        for batch_index, start in enumerate(
            range(0, self.dataset_size, self.batch_size)
        ):
            action_id = action_order[(offset + batch_index) % len(action_order)]
            yield [
                (int(index), action_id)
                for index in indices[start : start + self.batch_size]
            ]


@dataclass(frozen=True)
class GradientDiagnosticsPolicyV1:
    """Read-only gradient conflict measurements for the joint objective."""

    enabled: bool = False
    every_steps: int = 100
    maximum_parameter_elements: int = 262_144
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("gradient diagnostics enabled must be boolean")
        for name in ("every_steps", "maximum_parameter_elements"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        epsilon = float(self.epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("gradient diagnostics epsilon must be positive")
        object.__setattr__(self, "epsilon", epsilon)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None,
    ) -> "GradientDiagnosticsPolicyV1":
        if value is None:
            return cls()
        if set(value) != set(asdict(cls())):
            raise ValueError("gradient diagnostics policy fields drift")
        return cls(**dict(value))


@dataclass(frozen=True)
class ActionShadowProbePolicyV1:
    """Paired, held-out action evaluation that never changes optimization."""

    enabled: bool = False
    every_epochs: int = 5
    maximum_batches: int = 24
    bootstrap_repetitions: int = 10_000
    confidence_level: float = 0.95
    seed: int = 20_261_006

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("action shadow probe enabled must be boolean")
        for name in ("every_epochs", "maximum_batches", "bootstrap_repetitions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        confidence = float(self.confidence_level)
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("shadow confidence level must lie in (0,1)")
        object.__setattr__(self, "confidence_level", confidence)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("shadow seed must be a nonnegative integer")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None,
    ) -> "ActionShadowProbePolicyV1":
        if value is None:
            return cls()
        if set(value) != set(asdict(cls())):
            raise ValueError("action shadow probe policy fields drift")
        return cls(**dict(value))


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
    action_bank: ActionBankTrainingPolicyV1
    refiner_loss: UncertaintyRefinerLossPolicyV1 | None = None
    uncertainty_initialization_checkpoint: Path | None = None
    refiner_initialization_checkpoint: Path | None = None
    model_initialization_checkpoint: Path | None = None
    u2_schedule: (
        AlternatingU2ScheduleV2 | JointDecoupledU2ScheduleV2 | None
    ) = None
    fusion: U2FlowBidirectionalFusionPolicyV1 | None = None
    sam_homography: SamHomographySmoothnessPolicyV1 = (
        SamHomographySmoothnessPolicyV1()
    )
    sam_semantic_augmentation: SamSemanticAugmentationPolicyV1 = (
        SamSemanticAugmentationPolicyV1()
    )
    cuda_memory_reservation: CudaMemoryReservationPolicyV1 = (
        CudaMemoryReservationPolicyV1()
    )
    gradient_diagnostics: GradientDiagnosticsPolicyV1 = (
        GradientDiagnosticsPolicyV1()
    )
    action_shadow_probe: ActionShadowProbePolicyV1 = (
        ActionShadowProbePolicyV1()
    )
    metric_spatial_stride: int = 8
    iters: int = 4
    severe_error_threshold: float = 3.0
    schema: str = INTEGRATED_TRAINER_SCHEMA_V2

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IntegratedTrainerConfigV2":
        expected = {
            "schema", "run_dir", "dataset", "model", "loss", "training", "iters",
            "severe_error_threshold",
        }
        optional = {
            "action_bank", "uncertainty_initialization_checkpoint",
            "refiner_initialization_checkpoint", "model_initialization_checkpoint",
            "metric_spatial_stride",
            "refiner_loss", "u2_schedule", "fusion",
            "sam_homography", "sam_semantic_augmentation",
            "cuda_memory_reservation",
            "gradient_diagnostics", "action_shadow_probe",
        }
        if not expected <= set(value) or set(value) - expected - optional or value["schema"] != INTEGRATED_TRAINER_SCHEMA_V2:
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
        refiner_loss_value = value.get("refiner_loss")
        refiner_loss = (
            None
            if refiner_loss_value is None
            else UncertaintyRefinerLossPolicyV1(**dict(refiner_loss_value))
        )
        if (scope is TrainableScopeV2.REFINER_ONLY) != (refiner_loss is not None):
            raise ValueError("refiner_only scope and conservative U1 loss must be configured together")
        if refiner_loss is not None:
            if variant is RecurrentUncertaintyVariantV2.HEAD_ONLY:
                raise ValueError("U1 refiner training requires a refinement variant")
            decoupled = DecoupledFlowLossPolicyV2(**dict(value["loss"]))
            if any((decoupled.task_weight, decoupled.augmentation_weight, decoupled.uncertainty_weight)):
                raise ValueError("U1 uses only the conservative refiner loss")
            if value.get("uncertainty_initialization_checkpoint") is None:
                raise ValueError("U1 refiner training requires a completed U0 initialization")
        u2_schedule = _u2_schedule_from_mapping(value.get("u2_schedule"))
        refiner_checkpoint_value = value.get("refiner_initialization_checkpoint")
        model_checkpoint_value = value.get("model_initialization_checkpoint")
        partial_initialization = any(
            value.get(key) is not None
            for key in (
                "uncertainty_initialization_checkpoint",
                "refiner_initialization_checkpoint",
            )
        )
        if model_checkpoint_value is not None and partial_initialization:
            raise ValueError(
                "full-model initialization cannot be combined with U0/U1 partial initialization"
            )
        if u2_schedule is None and refiner_checkpoint_value is not None:
            raise ValueError(
                "U1 refiner initialization requires a U2 schedule"
            )
        if (
            u2_schedule is not None
            and model_checkpoint_value is None
            and refiner_checkpoint_value is None
        ):
            raise ValueError(
                "U2 and U1 refiner initialization must be configured "
                "together unless a full-model initialization is provided"
            )
        if u2_schedule is not None:
            if scope is not TrainableScopeV2.ALL:
                raise ValueError("U2 requires all parameters in its optimizer")
            if variant is RecurrentUncertaintyVariantV2.HEAD_ONLY:
                raise ValueError("head-only model cannot run U2")
            if refiner_loss is not None:
                raise ValueError("U2 cannot use the frozen-U0 U1 objective")
            if (
                model_checkpoint_value is None
                and value.get("uncertainty_initialization_checkpoint") is None
            ):
                raise ValueError(
                    "U2 requires a completed U0 or full-model initialization"
                )
        iters = value["iters"]
        if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
            raise ValueError("iters must be a positive integer")
        threshold = float(value["severe_error_threshold"])
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("severe error threshold must be finite and nonnegative")
        metric_stride = value.get("metric_spatial_stride", 8)
        if isinstance(metric_stride, bool) or not isinstance(metric_stride, int) or metric_stride < 1:
            raise ValueError("metric_spatial_stride must be a positive integer")
        training = IntegratedTrainingOptionsV2(**dict(value["training"]))
        if u2_schedule is not None and training.epochs % u2_schedule.epochs_per_round:
            raise ValueError("U2 epochs must contain complete alternating rounds")
        fusion = U2FlowBidirectionalFusionPolicyV1.from_mapping(value.get("fusion"))
        sam_homography = SamHomographySmoothnessPolicyV1.from_mapping(
            value.get("sam_homography")
        )
        sam_semantic_augmentation = SamSemanticAugmentationPolicyV1.from_mapping(
            value.get("sam_semantic_augmentation")
        )
        cuda_memory_reservation = CudaMemoryReservationPolicyV1.from_mapping(
            value.get("cuda_memory_reservation")
        )
        gradient_diagnostics = GradientDiagnosticsPolicyV1.from_mapping(
            value.get("gradient_diagnostics")
        )
        action_shadow_probe = ActionShadowProbePolicyV1.from_mapping(
            value.get("action_shadow_probe")
        )
        if sam_homography.enabled or sam_semantic_augmentation.enabled:
            dataset_kwargs = dataset.get("kwargs")
            if not isinstance(dataset_kwargs, Mapping) or not {
                "sam_full_segmentation_root", "sam_checkpoint_sha256",
            } <= set(dataset_kwargs):
                raise ValueError(
                    "enabled SAM objective requires traced full-segmentation data"
                )
        action_bank = ActionBankTrainingPolicyV1.from_mapping(value.get("action_bank"))
        if (
            fusion is not None
            and fusion.enabled
            and action_bank.validation_action_id != OPTICAL_NATIVE_ACTION_ID
        ):
            raise ValueError("U2Flow fusion currently requires native validation action")
        return cls(
            run_dir=Path(value["run_dir"]).expanduser().resolve(),
            dataset=dataset,
            model=model,
            loss=DecoupledFlowLossPolicyV2(**dict(value["loss"])),
            training=training,
            action_bank=action_bank,
            refiner_loss=refiner_loss,
            uncertainty_initialization_checkpoint=(
                None
                if value.get("uncertainty_initialization_checkpoint") is None
                else Path(str(value["uncertainty_initialization_checkpoint"])).expanduser().resolve()
            ),
            refiner_initialization_checkpoint=(
                None
                if refiner_checkpoint_value is None
                else Path(str(refiner_checkpoint_value)).expanduser().resolve()
            ),
            model_initialization_checkpoint=(
                None
                if model_checkpoint_value is None
                else Path(str(model_checkpoint_value)).expanduser().resolve()
            ),
            u2_schedule=u2_schedule,
            fusion=fusion,
            sam_homography=sam_homography,
            sam_semantic_augmentation=sam_semantic_augmentation,
            cuda_memory_reservation=cuda_memory_reservation,
            gradient_diagnostics=gradient_diagnostics,
            action_shadow_probe=action_shadow_probe,
            metric_spatial_stride=metric_stride,
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
        result = {
            "schema": self.schema,
            "run_dir": str(self.run_dir),
            "dataset": dict(self.dataset),
            "model": dict(self.model),
            "loss": asdict(self.loss),
            "training": asdict(self.training),
            "action_bank": asdict(self.action_bank),
            "uncertainty_initialization_checkpoint": (
                None
                if self.uncertainty_initialization_checkpoint is None
                else str(self.uncertainty_initialization_checkpoint)
            ),
            "metric_spatial_stride": self.metric_spatial_stride,
            "iters": self.iters,
            "severe_error_threshold": self.severe_error_threshold,
        }
        if self.refiner_loss is not None:
            result["refiner_loss"] = asdict(self.refiner_loss)
        if self.refiner_initialization_checkpoint is not None:
            result["refiner_initialization_checkpoint"] = str(
                self.refiner_initialization_checkpoint
            )
        if self.model_initialization_checkpoint is not None:
            result["model_initialization_checkpoint"] = str(
                self.model_initialization_checkpoint
            )
        if self.u2_schedule is not None:
            result["u2_schedule"] = asdict(self.u2_schedule)
        if self.fusion is not None:
            result["fusion"] = asdict(self.fusion)
        # Omit disabled defaults so checkpoints written before these optional
        # mechanisms retain the exact same config digest and remain resumable.
        if self.sam_homography != SamHomographySmoothnessPolicyV1():
            result["sam_homography"] = asdict(self.sam_homography)
        if self.sam_semantic_augmentation != SamSemanticAugmentationPolicyV1():
            result["sam_semantic_augmentation"] = asdict(
                self.sam_semantic_augmentation
            )
        if self.cuda_memory_reservation != CudaMemoryReservationPolicyV1():
            result["cuda_memory_reservation"] = asdict(
                self.cuda_memory_reservation
            )
        if self.gradient_diagnostics != GradientDiagnosticsPolicyV1():
            result["gradient_diagnostics"] = asdict(self.gradient_diagnostics)
        if self.action_shadow_probe != ActionShadowProbePolicyV1():
            result["action_shadow_probe"] = asdict(self.action_shadow_probe)
        return result

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
    cpu_state = state["torch"]
    if not isinstance(cpu_state, torch.Tensor):
        raise TypeError("checkpoint CPU RNG state must be a tensor")
    torch.set_rng_state(cpu_state.detach().to(device="cpu", dtype=torch.uint8))
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = state["cuda"]
        if not isinstance(cuda_states, (tuple, list)) or not all(
            isinstance(item, torch.Tensor) for item in cuda_states
        ):
            raise TypeError("checkpoint CUDA RNG states must be a tensor sequence")
        torch.cuda.set_rng_state_all([
            item.detach().to(device="cpu", dtype=torch.uint8)
            for item in cuda_states
        ])


def _freeze_batch_norm_statistics(module: torch.nn.Module) -> None:
    # U2Flow replaces BN with instance normalization.  The pinned SEA-RAFT
    # checkpoint cannot be changed compatibly, so small-batch adaptation keeps
    # pretrained BN statistics fixed while allowing affine weights to train.
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


class IntegratedUncertaintyTrainerV2:
    def __init__(
        self,
        config: IntegratedTrainerConfigV2,
        *,
        device: str = "auto",
        cuda_memory_reservation_override: CudaMemoryReservationPolicyV1 | None = None,
    ) -> None:
        self.config = config
        self.options = config.training
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        _seed_all(self.options.seed)
        memory_policy = (
            config.cuda_memory_reservation
            if cuda_memory_reservation_override is None
            else cuda_memory_reservation_override
        )
        self.cuda_memory_guard = CudaMemoryReservationV1(
            memory_policy, self.device,
        )
        self.cuda_memory_report = self.cuda_memory_guard.prime()
        split_argument = str(config.dataset["split_argument"])
        common = {"factory": config.dataset["factory"], "kwargs": config.dataset["kwargs"]}
        self.fit_dataset = _factory(common, **{split_argument: config.dataset["fit_split"]})
        self.validation_dataset = _factory(common, **{split_argument: config.dataset["validation_split"]})
        if not len(self.fit_dataset) or not len(self.validation_dataset):
            raise ValueError("fit and validation datasets must be nonempty")
        self.collate_fn = _symbol(str(config.dataset["collate_fn"]))
        self.variant = RecurrentUncertaintyVariantV2(config.model["kwargs"]["variant"])
        self.trainable_scope = TrainableScopeV2(config.model["kwargs"]["trainable_scope"])
        self.model = _factory(config.model, device=str(self.device)).to(self.device)
        if self.cuda_memory_guard.active:
            self.cuda_memory_report = self.cuda_memory_guard.refill()
        self.model_initialization_lineage = self._initialize_full_model()
        if self.model_initialization_lineage is None:
            self.initialization_lineage = self._initialize_uncertainty_head()
            self.refiner_initialization_lineage = self._initialize_refiner_head()
        else:
            self.initialization_lineage = None
            self.refiner_initialization_lineage = None
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
        self.selection_metric = (
            "uncertainty"
            if self.variant is RecurrentUncertaintyVariantV2.HEAD_ONLY
            else "epe"
        )
        self.best_validation_score = float("inf")
        self.best_validation_epe = float("inf")
        self.run_dir = config.run_dir
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.action_shadow_path = self.run_dir / "action_shadow_cases.jsonl"
        self.latest_path = self.run_dir / "latest.pt"
        self.best_path = self.run_dir / "best.pt"
        self.sam_semantic_cache = (
            SamSemanticObjectCacheV1(
                self.config.sam_semantic_augmentation.cache_size
            )
            if self.config.sam_semantic_augmentation.enabled
            else None
        )

    def _initialize_full_model(self) -> dict[str, Any] | None:
        """Warm-start a new stage without importing optimizer or epoch state."""

        path = self.config.model_initialization_checkpoint
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != INTEGRATED_TRAINER_SCHEMA_V2:
            raise ValueError("model initialization must be an integrated v2 checkpoint")
        source_config = payload.get("config")
        if not isinstance(source_config, Mapping):
            raise ValueError("model initialization checkpoint has no source config")
        if payload.get("config_digest") != _digest(source_config):
            raise ValueError("model initialization config digest drift")
        if source_config.get("model") != self.config.serializable().get("model"):
            raise ValueError("model initialization architecture/config drift")
        if int(source_config.get("iters", -1)) != self.config.iters:
            raise ValueError("model initialization recurrent iteration count drift")
        metrics_path = path.parent / "metrics.jsonl"
        if not metrics_path.is_file() or not any(
            json.loads(line).get("event") == "training_completed"
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ):
            raise ValueError("model initialization source training is incomplete")
        source_state = payload.get("model")
        if not isinstance(source_state, Mapping):
            raise ValueError("model initialization state is invalid")
        self.model.load_state_dict(source_state, strict=True)
        return {
            "checkpoint": str(path),
            "checkpoint_sha256": _sha256_file(path),
            "source_config_digest": str(payload["config_digest"]),
            "source_best_validation_score": payload.get("best_validation_score"),
            "source_best_validation_epe": payload.get("best_validation_epe"),
            "source_action_bank_lineage": payload.get("action_bank_lineage"),
            "source_model_initialization_lineage": payload.get(
                "model_initialization_lineage"
            ),
            "source_uncertainty_initialization_lineage": payload.get(
                "uncertainty_initialization_lineage"
            ),
            "source_refiner_initialization_lineage": payload.get(
                "refiner_initialization_lineage"
            ),
        }

    def _initialize_uncertainty_head(self) -> dict[str, Any] | None:
        path = self.config.uncertainty_initialization_checkpoint
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != INTEGRATED_TRAINER_SCHEMA_V2:
            raise ValueError("uncertainty initialization must be an integrated v2 checkpoint")
        source_config = payload.get("config")
        if not isinstance(source_config, Mapping):
            raise ValueError("uncertainty initialization checkpoint has no source config")
        if payload.get("config_digest") != _digest(source_config):
            raise ValueError("uncertainty initialization config digest drift")
        source_model = source_config.get("model")
        source_kwargs = source_model.get("kwargs") if isinstance(source_model, Mapping) else None
        if not isinstance(source_kwargs, Mapping):
            raise ValueError("uncertainty initialization model config is invalid")
        if (
            source_kwargs.get("variant") != RecurrentUncertaintyVariantV2.HEAD_ONLY.value
            or source_kwargs.get("trainable_scope") != TrainableScopeV2.UNCERTAINTY_ONLY.value
        ):
            raise ValueError("uncertainty initialization must come from head-only U0")
        source_bank = source_config.get("action_bank")
        if not isinstance(source_bank, Mapping) or (
            source_bank.get("mode") != "native_only"
            or tuple(source_bank.get("action_ids", ())) != (OPTICAL_NATIVE_ACTION_ID,)
        ):
            raise ValueError("uncertainty initialization must be native-only")
        if int(source_config.get("iters", -1)) != self.config.iters:
            raise ValueError("uncertainty initialization recurrent iteration count drift")
        source_backbone = {
            key: value for key, value in source_kwargs.items()
            if key not in {
                "variant", "trainable_scope", "refinement_channels", "maximum_update_px",
            }
        }
        target_kwargs = dict(self.config.model["kwargs"])
        target_backbone = {
            key: value for key, value in target_kwargs.items()
            if key not in {
                "variant", "trainable_scope", "refinement_channels", "maximum_update_px",
            }
        }
        if source_backbone != target_backbone:
            raise ValueError("uncertainty initialization matcher lineage drift")
        metrics_path = path.parent / "metrics.jsonl"
        if not metrics_path.is_file() or not any(
            json.loads(line).get("event") == "training_completed"
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ):
            raise ValueError("uncertainty initialization source training is incomplete")
        target_head = getattr(getattr(self.model, "recurrent_head", None), "uncertainty_head", None)
        if not isinstance(target_head, torch.nn.Module):
            raise ValueError("target model does not expose a recurrent uncertainty head")
        source_state = payload.get("model")
        if not isinstance(source_state, Mapping):
            raise ValueError("uncertainty initialization model state is invalid")
        prefix = "recurrent_head.uncertainty_head."
        head_state = {
            str(key)[len(prefix):]: value
            for key, value in source_state.items()
            if str(key).startswith(prefix)
        }
        if not head_state:
            raise ValueError("source checkpoint has no recurrent uncertainty-head state")
        target_head.load_state_dict(head_state, strict=True)
        return {
            "checkpoint": str(path),
            "checkpoint_sha256": _sha256_file(path),
            "source_config_digest": str(payload["config_digest"]),
            "source_best_validation_score": payload.get("best_validation_score"),
        }

    def _initialize_refiner_head(self) -> dict[str, Any] | None:
        path = self.config.refiner_initialization_checkpoint
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != INTEGRATED_TRAINER_SCHEMA_V2:
            raise ValueError("refiner initialization must be an integrated v2 checkpoint")
        source_config = payload.get("config")
        if not isinstance(source_config, Mapping):
            raise ValueError("refiner initialization checkpoint has no source config")
        if payload.get("config_digest") != _digest(source_config):
            raise ValueError("refiner initialization config digest drift")
        source_model = source_config.get("model")
        source_kwargs = source_model.get("kwargs") if isinstance(source_model, Mapping) else None
        if not isinstance(source_kwargs, Mapping):
            raise ValueError("refiner initialization model config is invalid")
        target_kwargs = dict(self.config.model["kwargs"])
        if source_kwargs.get("trainable_scope") != TrainableScopeV2.REFINER_ONLY.value:
            raise ValueError("refiner initialization must come from refiner-only U1")
        if source_kwargs.get("variant") != target_kwargs.get("variant"):
            raise ValueError("U1 and U2 refinement variants differ")
        if not isinstance(source_config.get("refiner_loss"), Mapping):
            raise ValueError("refiner initialization source has no conservative U1 loss")
        if int(source_config.get("iters", -1)) != self.config.iters:
            raise ValueError("refiner initialization recurrent iteration count drift")
        source_backbone = {
            key: value for key, value in source_kwargs.items()
            if key not in {
                "variant", "trainable_scope", "refinement_channels", "maximum_update_px",
            }
        }
        target_backbone = {
            key: value for key, value in target_kwargs.items()
            if key not in {
                "variant", "trainable_scope", "refinement_channels", "maximum_update_px",
            }
        }
        if source_backbone != target_backbone:
            raise ValueError("refiner initialization matcher lineage drift")
        if int(source_kwargs.get("refinement_channels", 64)) != int(
            target_kwargs.get("refinement_channels", 64)
        ):
            raise ValueError("refiner initialization channel count drift")
        if float(source_kwargs.get("maximum_update_px", 1.0)) != float(
            target_kwargs.get("maximum_update_px", 1.0)
        ):
            raise ValueError("refiner initialization update bound drift")
        if payload.get("uncertainty_initialization_lineage") != self.initialization_lineage:
            raise ValueError("U1 refiner was trained with a different U0 observer")
        metrics_path = path.parent / "metrics.jsonl"
        if not metrics_path.is_file() or not any(
            json.loads(line).get("event") == "training_completed"
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ):
            raise ValueError("refiner initialization source training is incomplete")
        target_head = getattr(getattr(self.model, "recurrent_head", None), "refinement_head", None)
        if not isinstance(target_head, torch.nn.Module):
            raise ValueError("target model does not expose a recurrent refinement head")
        source_state = payload.get("model")
        if not isinstance(source_state, Mapping):
            raise ValueError("refiner initialization model state is invalid")
        prefix = "recurrent_head.refinement_head."
        head_state = {
            str(key)[len(prefix):]: value
            for key, value in source_state.items()
            if str(key).startswith(prefix)
        }
        if not head_state:
            raise ValueError("source checkpoint has no recurrent refinement-head state")
        target_head.load_state_dict(head_state, strict=True)
        return {
            "checkpoint": str(path),
            "checkpoint_sha256": _sha256_file(path),
            "source_config_digest": str(payload["config_digest"]),
            "source_best_validation_epe": payload.get("best_validation_epe"),
            "uncertainty_initialization_lineage": self.initialization_lineage,
        }

    def _loader(self, dataset, *, training: bool, epoch: int) -> DataLoader:
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(
            self.options.seed + epoch + (0 if training else 100000)
        )
        if training and self.config.action_bank.mode == "balanced_batches":
            if not callable(getattr(dataset, "set_action_id", None)):
                raise ValueError(
                    "balanced action-bank training requires dataset.set_action_id"
                )
            scheduled = _ActionScheduledDatasetV1(dataset)
            batch_sampler = _BalancedActionBatchSamplerV1(
                dataset_size=len(dataset),
                batch_size=self.options.batch_size,
                action_ids=self.config.action_bank.action_ids,
                seed=self.options.seed,
                epoch=epoch,
            )
            return DataLoader(
                scheduled,
                batch_sampler=batch_sampler,
                num_workers=self.options.num_workers,
                pin_memory=self.options.pin_memory,
                persistent_workers=False,
                generator=generator,
                collate_fn=self.collate_fn,
            )
        action_epoch = epoch
        if (
            training
            and self.config.u2_schedule is not None
            and self.config.action_bank.mode == "cycle"
        ):
            # Keep both detached U2 phases on the same action. Advancing once
            # per complete round prevents phase identity from being confounded
            # with disjoint action-bank subsets.
            action_epoch = self.config.u2_schedule.round_for_epoch(epoch)
        action_id = (
            self.config.action_bank.action_for_epoch(action_epoch)
            if training else self.config.action_bank.validation_action_id
        )
        if hasattr(dataset, "set_action_id"):
            dataset.set_action_id(action_id)
        elif action_id != OPTICAL_NATIVE_ACTION_ID:
            raise ValueError("configured action-bank training requires dataset.set_action_id")
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
            # Head-only U0 performs a frozen native forward before the
            # trainable augmented forward inside the same autocast region.
            # Autocast's weight cache can otherwise reuse a cast created under
            # no_grad and silently detach the second use of the shared head.
            return torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.amp_enabled,
                cache_enabled=False,
            )
        return nullcontext()

    def _move(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in batch.items():
            result[key] = value.to(self.device, non_blocking=self.device.type == "cuda") if torch.is_tensor(value) else value
        required = {
            "native_frames", "augmented_frames", "augmentation_valid",
            "native_to_augmented", "swap_endpoints",
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

    def _batch_action(self, batch: Mapping[str, Any]) -> tuple[str, int]:
        raw_actions = batch.get("action_id", (OPTICAL_NATIVE_ACTION_ID,))
        actions = (
            (raw_actions,)
            if isinstance(raw_actions, str)
            else tuple(str(item) for item in raw_actions)
        )
        if len(set(actions)) != 1:
            raise ValueError("one training batch cannot mix action-bank arms")
        action_id = actions[0]
        raw_overrides = batch.get("matcher_iterations_override", (None,))
        if torch.is_tensor(raw_overrides):
            overrides = tuple(int(item) for item in raw_overrides.detach().cpu().tolist())
        elif raw_overrides is None or isinstance(raw_overrides, int):
            overrides = (raw_overrides,)
        else:
            overrides = tuple(raw_overrides)
        if len(set(overrides)) != 1:
            raise ValueError("one training batch cannot mix matcher iteration counts")
        override = overrides[0]
        if action_id == OPTICAL_NATIVE_ACTION_ID:
            expected_override = None
        else:
            try:
                arm = OPTICAL_FLOW_CAPACITY_BANK[action_id]
            except KeyError as exc:
                raise ValueError("batch action is outside the E292 capacity bank") from exc
            parameters = dict(arm.operator_parameters)
            expected_override = (
                int(parameters["iterations"])
                if arm.operator == "matcher_iteration_override" else None
            )
        if override != expected_override:
            raise ValueError("batch matcher iteration override does not match its action")
        iters = self.config.iters if override is None else int(override)
        if iters < 1:
            raise ValueError("matcher iterations must be positive")
        return action_id, iters

    def _configure_training_phase(self, epoch: int) -> tuple[str | None, int | None]:
        schedule = self.config.u2_schedule
        if schedule is None:
            self.model.train()
            _freeze_batch_norm_statistics(self.model)
            return None, None
        phase = schedule.phase_for_epoch(epoch)
        round_index = schedule.round_for_epoch(epoch)
        uncertainty_head = getattr(
            getattr(self.model, "recurrent_head", None), "uncertainty_head", None,
        )
        if not isinstance(uncertainty_head, torch.nn.Module):
            raise ValueError("U2 model has no uncertainty head")
        if phase == "observer":
            self.model.requires_grad_(False)
            uncertainty_head.requires_grad_(True)
            self.model.eval()
            uncertainty_head.train()
        elif phase == "flow":
            self.model.requires_grad_(True)
            uncertainty_head.requires_grad_(False)
            self.model.train()
            _freeze_batch_norm_statistics(self.model)
        elif phase == "joint":
            self.model.requires_grad_(True)
            self.model.train()
            _freeze_batch_norm_statistics(self.model)
        else:  # pragma: no cover - schedule owns this closed vocabulary
            raise ValueError("unknown U2 phase")
        return phase, round_index

    def _forward_loss(
        self,
        batch: Mapping[str, Any],
        *,
        augmentation_enabled: bool,
        semantic_augmentation_enabled: bool = False,
        semantic_cache_update_enabled: bool = False,
        phase: str | None = None,
    ):
        _action_id, iters = self._batch_action(batch)
        native = batch["native_frames"].float() * 255.0
        augmented = batch["augmented_frames"].float() * 255.0
        if self.config.refiner_loss is not None:
            disable_refinement = getattr(self.model, "refinement_disabled", None)
            if not callable(disable_refinement):
                raise TypeError("U1 model must expose refinement_disabled()")
            with torch.no_grad(), disable_refinement():
                base_output = self.model(
                    native[:, 0], native[:, 1], iters=iters, test_mode=True,
                )
            native_output = self.model(
                native[:, 0], native[:, 1], iters=iters, test_mode=True,
            )
            ground_truth = batch.get("ground_truth_flow")
            valid = batch.get("ground_truth_valid")
            if ground_truth is None or valid is None:
                raise ValueError("U1 refiner training requires ground-truth flow")
            conservative = uncertainty_refiner_loss_v1(
                self._flows(native_output)[-1],
                self._flows(base_output)[-1],
                ground_truth.float(),
                valid.bool(),
                policy=self.config.refiner_loss,
            )
            zero = conservative.total * 0.0
            losses = _ConservativeU1LossV2(
                total=conservative.total,
                task=conservative.task,
                augmentation=zero,
                uncertainty=zero,
                harm=conservative.harm,
                anchor=conservative.anchor,
                smoothness=conservative.smoothness,
            )
            return losses, native_output, base_output
        native_context = (
            torch.no_grad()
            if self.variant is RecurrentUncertaintyVariantV2.HEAD_ONLY or phase == "observer"
            else nullcontext()
        )
        with native_context:
            native_output = self.model(
                native[:, 0], native[:, 1], iters=iters, test_mode=True,
            )
        native_flows = self._flows(native_output)
        swap = batch["swap_endpoints"].to(dtype=torch.bool).reshape(-1, 1, 1, 1)
        teacher = native_flows[-1].detach()
        if bool(swap.any()):
            with torch.no_grad():
                reverse_output = self.model(
                    native[:, 1], native[:, 0], iters=iters, test_mode=True,
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
            augmented[:, 0], augmented[:, 1], iters=iters, test_mode=True,
        )
        selected_policy = self.config.loss
        if phase == "observer":
            selected_policy = DecoupledFlowLossPolicyV2(
                task_weight=0.0,
                augmentation_weight=0.0,
                uncertainty_weight=1.0,
                gamma=selected_policy.gamma,
            )
        elif phase == "flow":
            selected_policy = DecoupledFlowLossPolicyV2(
                task_weight=selected_policy.task_weight,
                augmentation_weight=selected_policy.augmentation_weight,
                uncertainty_weight=0.0,
                gamma=selected_policy.gamma,
            )
        elif phase not in {None, "joint"}:
            raise ValueError("unknown U2 phase")
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
            (
                None
                if batch.get("ground_truth_flow") is None
                else batch["ground_truth_flow"].float()
            ),
            (
                None
                if batch.get("ground_truth_valid") is None
                else batch["ground_truth_valid"].bool()
            ),
            transformed_teacher,
            consistency_valid,
            policy=selected_policy,
        )
        semantic_policy = self.config.sam_semantic_augmentation
        semantic_loss = losses.total * 0.0
        semantic_objects = 0
        semantic_object_pixels = 0
        if (
            semantic_policy.enabled
            and (semantic_augmentation_enabled or semantic_cache_update_enabled)
            and phase != "observer"
        ):
            segment_ids = batch.get("sam_segment_ids")
            if not isinstance(segment_ids, torch.Tensor):
                raise ValueError(
                    "enabled SAM semantic augmentation requires batch segment ids"
                )
            if self.sam_semantic_cache is None:
                raise RuntimeError("SAM semantic augmentation cache is unavailable")
            if semantic_augmentation_enabled and self.sam_semantic_cache.ready:
                cached_masks, cached_images, cached_motions = (
                    self.sam_semantic_cache.sample(
                        int(native.shape[0]),
                        objects_per_batch=semantic_policy.objects_per_batch,
                        device=native.device,
                        dtype=native.dtype,
                        policy=semantic_policy,
                    )
                )
                valid_value = batch.get("ground_truth_valid")
                semantic_valid = (
                    torch.ones_like(native_flows[-1][:, :1], dtype=torch.bool)
                    if not isinstance(valid_value, torch.Tensor)
                    else valid_value.bool()
                )
                semantic_batch = compose_sam_semantic_augmentation_v1(
                    native,
                    native_flows[-1].detach(),
                    semantic_valid,
                    cached_masks,
                    cached_images,
                    cached_motions,
                )
                semantic_output = self.model(
                    semantic_batch.frames[:, 0],
                    semantic_batch.frames[:, 1],
                    iters=iters,
                    test_mode=True,
                )
                semantic_loss = sam_semantic_sequence_loss_v1(
                    self._flows(semantic_output),
                    semantic_batch.target_flow,
                    semantic_batch.valid,
                    gamma=self.config.loss.gamma,
                )
                semantic_objects = semantic_batch.object_count
                semantic_object_pixels = semantic_batch.object_pixel_count
            exact_object_mask = batch.get("sam_key_object_mask")
            exact_object_present = batch.get("sam_key_object_present")
            if isinstance(exact_object_mask, torch.Tensor) and isinstance(
                exact_object_present, torch.Tensor
            ):
                object_masks = exact_object_mask.bool()
                object_present = exact_object_present.bool().reshape(-1)
                if object_masks.shape != segment_ids.shape:
                    raise ValueError(
                        "SAM key-object masks must match the segment-id grid"
                    )
                if object_present.shape != (segment_ids.shape[0],):
                    raise ValueError("SAM key-object presence shape drift")
            else:
                object_masks, object_present = select_sam_object_masks_v1(
                    segment_ids, policy=semantic_policy,
                )
            if bool(object_present.any()):
                float_masks = object_masks.to(dtype=native_flows[-1].dtype)
                denominator = float_masks.sum(dim=(2, 3)).clamp_min(1.0)
                mean_motion = (
                    native_flows[-1].detach() * float_masks
                ).sum(dim=(2, 3)) / denominator
                self.sam_semantic_cache.push(
                    object_masks[object_present],
                    native[object_present, 0],
                    mean_motion[object_present],
                )
        configured_sam_policy = self.config.sam_homography
        sam_policy = (
            SamHomographySmoothnessPolicyV1()
            if phase == "observer"
            else configured_sam_policy
        )
        if sam_policy.enabled and not isinstance(batch.get("sam_segment_ids"), torch.Tensor):
            raise ValueError("enabled SAM homography loss requires batch segment ids")
        sam_result = sam_homography_smoothness_loss_v1(
            native_flows[-1],
            (
                batch["sam_segment_ids"]
                if sam_policy.enabled
                else torch.zeros_like(native_flows[-1][:, :1], dtype=torch.int64)
            ),
            self._alphas(native_output)[-1],
            policy=sam_policy,
        )
        combined = _IntegratedLossV2(
            total=(
                losses.total
                + semantic_policy.weight * semantic_loss
                + configured_sam_policy.weight * sam_result.loss
            ),
            task=losses.task,
            augmentation=losses.augmentation,
            uncertainty=losses.uncertainty,
            sam_homography=sam_result.loss,
            sam_candidate_regions=sam_result.candidate_regions,
            sam_fitted_regions=sam_result.fitted_regions,
            sam_semantic=semantic_loss,
            sam_semantic_objects=semantic_objects,
            sam_semantic_object_pixels=semantic_object_pixels,
        )
        return combined, native_output, None

    def _gradient_diagnostics(self, losses: Any) -> dict[str, Any]:
        """Measure weighted objective alignment without changing ``.grad``."""

        policy = self.config.gradient_diagnostics
        candidates = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
            and "uncertainty_head" not in name
            and "refinement_head" not in name
        ]
        if not candidates:
            raise ValueError("gradient diagnostics found no shared parameters")
        # Prefer the recurrent/shared trunk while bounding retained diagnostic
        # vectors.  This is a measurement path only; optimization still uses
        # every trainable parameter.
        candidates.sort(key=lambda item: (
            0 if any(token in item[0] for token in ("update", "backbone", "network")) else 1,
            item[0],
        ))
        selected: list[tuple[str, torch.nn.Parameter]] = []
        selected_elements = 0
        for name, parameter in candidates:
            elements = int(parameter.numel())
            if selected_elements + elements <= policy.maximum_parameter_elements:
                selected.append((name, parameter))
                selected_elements += elements
        if not selected:
            name, parameter = min(candidates, key=lambda item: item[1].numel())
            selected = [(name, parameter)]
            selected_elements = int(parameter.numel())
        parameters = tuple(parameter for _name, parameter in selected)
        loss_policy = self.config.loss
        components: dict[str, torch.Tensor] = {}
        for name, weight in (
            ("task", loss_policy.task_weight),
            ("augmentation", loss_policy.augmentation_weight),
            ("uncertainty", loss_policy.uncertainty_weight),
        ):
            component = getattr(losses, name, None)
            if (
                weight > 0.0
                and isinstance(component, torch.Tensor)
                and component.requires_grad
            ):
                components[name] = float(weight) * component
        sam_component = getattr(losses, "sam_homography", None)
        if (
            self.config.sam_homography.enabled
            and isinstance(sam_component, torch.Tensor)
            and sam_component.requires_grad
        ):
            components["sam_homography"] = (
                self.config.sam_homography.weight * sam_component
            )
        semantic_component = getattr(losses, "sam_semantic", None)
        if (
            self.config.sam_semantic_augmentation.enabled
            and isinstance(semantic_component, torch.Tensor)
            and semantic_component.requires_grad
            and float(semantic_component.detach()) != 0.0
        ):
            components["sam_semantic"] = (
                self.config.sam_semantic_augmentation.weight * semantic_component
            )
        vectors: dict[str, torch.Tensor] = {}
        for name, component in components.items():
            gradients = torch.autograd.grad(
                component,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            vectors[name] = torch.cat([
                (
                    torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
                    if gradient is None
                    else gradient.detach().float().reshape(-1)
                )
                for parameter, gradient in zip(parameters, gradients)
            ])
        result: dict[str, Any] = {
            "gradient_diagnostic_scope": "current_microbatch_before_backward",
            "optimizer_gradient_accumulation_steps": (
                self.options.gradient_accumulation_steps
            ),
            "gradient_diagnostic_parameter_elements": selected_elements,
            "gradient_diagnostic_parameter_tensors": [
                name for name, _parameter in selected
            ],
        }
        norms = {
            name: torch.linalg.vector_norm(vector)
            for name, vector in vectors.items()
        }
        for name, norm in norms.items():
            result[f"gradient_norm_{name}"] = float(norm.cpu())
        conflict_count = 0
        names = tuple(vectors)
        for first_index, first in enumerate(names):
            for second in names[first_index + 1 :]:
                denominator = norms[first] * norms[second]
                cosine: float | None
                if float(denominator.detach().cpu()) <= policy.epsilon:
                    cosine = None
                else:
                    cosine = float(
                        (
                            torch.dot(vectors[first], vectors[second])
                            / denominator
                        ).detach().cpu()
                    )
                    conflict_count += int(cosine < 0.0)
                result[f"gradient_cosine_{first}__{second}"] = cosine
        result["gradient_conflict_count"] = conflict_count
        return result

    def _shadow_indices(self) -> tuple[int, ...]:
        maximum = min(
            len(self.validation_dataset),
            self.config.action_shadow_probe.maximum_batches
            * self.options.batch_size,
        )
        rows = getattr(self.validation_dataset, "rows", None)
        if isinstance(rows, (tuple, list)) and len(rows) == len(self.validation_dataset):
            buckets: dict[str, list[int]] = {}
            for index, row in enumerate(rows):
                scene_id = str(getattr(row, "scene_id", ""))
                buckets.setdefault(scene_id, []).append(index)
            selected: list[int] = []
            depth = 0
            scenes = tuple(sorted(buckets))
            while len(selected) < maximum:
                added = False
                for scene in scenes:
                    if depth < len(buckets[scene]):
                        selected.append(buckets[scene][depth])
                        added = True
                        if len(selected) == maximum:
                            break
                if not added:
                    break
                depth += 1
            return tuple(selected)
        if maximum == len(self.validation_dataset):
            return tuple(range(maximum))
        # Evenly cover generic datasets instead of taking a prefix.
        return tuple(
            int(index)
            for index in np.linspace(
                0, len(self.validation_dataset) - 1, maximum,
            ).round().astype(np.int64)
        )

    def _shadow_loader(self, action_id: str, epoch: int) -> DataLoader:
        setter = getattr(self.validation_dataset, "set_action_id", None)
        if not callable(setter):
            raise ValueError("action shadow probe requires dataset.set_action_id")
        setter(action_id)
        set_epoch = getattr(self.validation_dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        generator = torch.Generator().manual_seed(
            self.options.seed + epoch + 700_001
        )
        return DataLoader(
            Subset(self.validation_dataset, self._shadow_indices()),
            batch_size=self.options.batch_size,
            shuffle=False,
            num_workers=self.options.num_workers,
            pin_memory=self.options.pin_memory,
            persistent_workers=False,
            generator=generator,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def _scene_cluster_interval(
        gains: np.ndarray,
        scenes: tuple[str, ...],
        *,
        repetitions: int,
        confidence_level: float,
        seed: int,
    ) -> tuple[float | None, float | None, float]:
        grouped: dict[str, list[float]] = {}
        for gain, scene in zip(gains.tolist(), scenes):
            grouped.setdefault(scene, []).append(float(gain))
        scene_means = np.asarray([
            np.mean(grouped[scene], dtype=np.float64)
            for scene in sorted(grouped)
        ], dtype=np.float64)
        mean = float(scene_means.mean())
        if scene_means.size < 2:
            return None, None, mean
        generator = np.random.default_rng(seed)
        sampled = generator.integers(
            0, scene_means.size,
            size=(repetitions, scene_means.size),
        )
        bootstrap = scene_means[sampled].mean(axis=1)
        tail = (1.0 - confidence_level) / 2.0
        low, high = np.quantile(bootstrap, (tail, 1.0 - tail))
        return float(low), float(high), mean

    @staticmethod
    def _selector_observable_context(
        native_frames: torch.Tensor,
        flow: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[dict[str, float | bool | str | None], ...]:
        """Summarize P0-only signals that are available before action choice."""

        log_scale = 0.5 * alpha.float() - 0.5 * math.log(2.0)
        features, warp_valid = build_searaft_observable_features_v1(
            native_frames[:, 0].float(),
            native_frames[:, 1].float(),
            flow.float(),
            log_scale,
        )
        magnitude = torch.linalg.vector_norm(flow.float(), dim=1)
        grayscale = native_frames.float().mean(dim=2)
        spatial_dx = torch.abs(grayscale[..., 1:] - grayscale[..., :-1])
        spatial_dy = torch.abs(grayscale[..., 1:, :] - grayscale[..., :-1, :])
        contexts: list[dict[str, float | bool | str | None]] = []
        for index in range(native_frames.shape[0]):
            support = warp_valid[index, 0]
            supported_log_scale = log_scale[index, 0][support]
            supported_magnitude = magnitude[index][support]
            photometric = features[index, 9][support]
            flow_gradient = features[index, 10][support]
            spatial_gradient = torch.cat((
                spatial_dx[index].reshape(-1),
                spatial_dy[index].reshape(-1),
            ))
            contexts.append({
                "observable_context_schema": (
                    "stablebridge-action-selector-observable-context/v1"
                ),
                "context_source_action_id": OPTICAL_NATIVE_ACTION_ID,
                "context_is_outcome_blind": True,
                "context_requires_ground_truth": False,
                "native_log_scale_mean": (
                    None
                    if not supported_log_scale.numel()
                    else float(supported_log_scale.double().mean().cpu())
                ),
                "native_log_scale_p90": (
                    None
                    if not supported_log_scale.numel()
                    else float(torch.quantile(supported_log_scale.float(), 0.9).cpu())
                ),
                "native_flow_magnitude_mean": (
                    None
                    if not supported_magnitude.numel()
                    else float(supported_magnitude.double().mean().cpu())
                ),
                "native_flow_magnitude_p90": (
                    None
                    if not supported_magnitude.numel()
                    else float(torch.quantile(supported_magnitude.float(), 0.9).cpu())
                ),
                "native_photometric_l1_mean": (
                    None
                    if not photometric.numel()
                    else float(photometric.double().mean().cpu())
                ),
                "native_flow_gradient_mean": (
                    None
                    if not flow_gradient.numel()
                    else float(flow_gradient.double().mean().cpu())
                ),
                "native_warp_valid_fraction": float(
                    support.double().mean().cpu()
                ),
                "native_frame_intensity_mean": float(
                    native_frames[index].double().mean().cpu()
                ),
                "native_frame_intensity_standard_deviation": float(
                    native_frames[index].double().std(unbiased=False).cpu()
                ),
                "native_temporal_l1_mean": float(
                    torch.abs(
                        native_frames[index, 0] - native_frames[index, 1]
                    ).double().mean().cpu()
                ),
                "native_spatial_gradient_mean": float(
                    0.0
                    if not spatial_gradient.numel()
                    else spatial_gradient.double().mean().cpu()
                ),
            })
        return tuple(contexts)

    @torch.no_grad()
    def action_shadow_probe(self, epoch: int) -> tuple[dict[str, Any], ...]:
        """Run paired actions at one checkpoint; never feed results to training."""

        policy = self.config.action_shadow_probe
        if not policy.enabled:
            return ()
        self.model.eval()
        action_ids = (OPTICAL_NATIVE_ACTION_ID,) + tuple(
            action_id
            for action_id in self.config.action_bank.action_ids
            if action_id != OPTICAL_NATIVE_ACTION_ID
        )
        outcomes: dict[str, dict[str, tuple[str, float, int]]] = {}
        observable_contexts: dict[str, dict[str, float | bool | str | None]] = {}
        elapsed: dict[str, float] = {}
        matcher_iterations: dict[str, int] = {}
        try:
            for action_id in action_ids:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                per_case: dict[str, tuple[str, float, int]] = {}
                for raw in self._shadow_loader(action_id, epoch):
                    batch = self._move(raw)
                    native = batch["native_frames"].float() * 255.0
                    _observed_action, iterations = self._batch_action(batch)
                    with self._autocast():
                        output = self.model(
                            native[:, 0], native[:, 1],
                            iters=iterations,
                            test_mode=True,
                        )
                    flow = self._flows(output)[-1]
                    if action_id == OPTICAL_NATIVE_ACTION_ID:
                        contexts = self._selector_observable_context(
                            batch["native_frames"].float(),
                            flow,
                            self._alphas(output)[-1],
                        )
                    truth = batch.get("ground_truth_flow")
                    valid_value = batch.get("ground_truth_valid")
                    if not isinstance(truth, torch.Tensor) or not isinstance(
                        valid_value, torch.Tensor,
                    ):
                        raise ValueError("action shadow probe requires held-out flow")
                    error = torch.linalg.vector_norm(flow - truth.float(), dim=1)
                    valid = valid_value.bool()[:, 0]
                    row_ids = tuple(str(item) for item in batch.get("row_id", ()))
                    scene_ids = tuple(str(item) for item in batch.get("scene_id", ()))
                    if len(row_ids) != error.shape[0] or len(scene_ids) != error.shape[0]:
                        raise ValueError("action shadow probe needs row and scene identities")
                    for index, (row_id, scene_id) in enumerate(
                        zip(row_ids, scene_ids)
                    ):
                        support = valid[index]
                        count = int(support.sum().cpu())
                        if count == 0:
                            raise ValueError("action shadow case has empty valid support")
                        per_case[row_id] = (
                            scene_id,
                            float(error[index][support].double().mean().cpu()),
                            count,
                        )
                        if action_id == OPTICAL_NATIVE_ACTION_ID:
                            observable_contexts[row_id] = contexts[index]
                    matcher_iterations[action_id] = iterations
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                elapsed[action_id] = time.perf_counter() - started
                outcomes[action_id] = per_case
        finally:
            setter = getattr(self.validation_dataset, "set_action_id", None)
            if callable(setter):
                setter(self.config.action_bank.validation_action_id)
        native = outcomes[OPTICAL_NATIVE_ACTION_ID]
        native_ids = tuple(sorted(native))
        summaries: list[dict[str, Any]] = []
        # The policy confidence is interpreted as family-wise confidence for
        # the non-native action bank.  Without this correction, checking many
        # actions at 95% independently would overstate benefit/harm evidence.
        hypothesis_count = max(len(action_ids) - 1, 1)
        per_action_confidence = 1.0 - (
            (1.0 - policy.confidence_level) / hypothesis_count
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.action_shadow_path.open("a", encoding="utf-8") as stream:
            for action_index, action_id in enumerate(action_ids):
                current = outcomes[action_id]
                if tuple(sorted(current)) != native_ids:
                    raise ValueError("action shadow cases drifted across actions")
                if tuple(sorted(observable_contexts)) != native_ids:
                    raise ValueError("action shadow observable contexts drifted")
                gains = np.asarray([
                    native[row_id][1] - current[row_id][1]
                    for row_id in native_ids
                ], dtype=np.float64)
                scenes = tuple(native[row_id][0] for row_id in native_ids)
                low, high, scene_mean = self._scene_cluster_interval(
                    gains,
                    scenes,
                    repetitions=policy.bootstrap_repetitions,
                    confidence_level=per_action_confidence,
                    seed=policy.seed + epoch * 10_007 + action_index,
                )
                native_epes = np.asarray([
                    native[row_id][1] for row_id in native_ids
                ], dtype=np.float64)
                action_epes = np.asarray([
                    current[row_id][1] for row_id in native_ids
                ], dtype=np.float64)
                for row_id, gain in zip(native_ids, gains.tolist()):
                    stream.write(json.dumps({
                        "schema": "stablebridge-action-shadow-case/v1",
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "role": "validation_diagnostic_only",
                        "authorized_for_selector_training": False,
                        "row_id": row_id,
                        "scene_id": native[row_id][0],
                        "action_id": action_id,
                        "native_epe": native[row_id][1],
                        "action_epe": current[row_id][1],
                        "gain_epe": gain,
                        "valid_pixel_count": current[row_id][2],
                        **observable_contexts[row_id],
                    }, sort_keys=True, allow_nan=False) + "\n")
                summary = {
                    "event": "action_shadow_probe",
                    "utc": _utc_now(),
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "role": "validation_diagnostic_only",
                    "authorized_for_selector_training": False,
                    "action_id": action_id,
                    "matcher_iterations": matcher_iterations[action_id],
                    "case_count": len(native_ids),
                    "scene_count": len(set(scenes)),
                    "mean_native_case_epe": float(native_epes.mean()),
                    "mean_action_case_epe": float(action_epes.mean()),
                    "mean_case_gain_epe": float(gains.mean()),
                    "median_case_gain_epe": float(np.median(gains)),
                    "gain_standard_deviation_epe": float(gains.std()),
                    "best_case_gain_epe": float(gains.max()),
                    "worst_case_gain_epe": float(gains.min()),
                    "improved_case_fraction": float(np.mean(gains > 0.0)),
                    "harmed_case_fraction": float(np.mean(gains < 0.0)),
                    "mean_scene_gain_epe": scene_mean,
                    "scene_cluster_ci_low": low,
                    "scene_cluster_ci_high": high,
                    "benefit_supported": low is not None and low > 0.0,
                    "harm_supported": high is not None and high < 0.0,
                    "familywise_confidence_level": policy.confidence_level,
                    "per_action_confidence_level": per_action_confidence,
                    "multiple_comparison_correction": "bonferroni",
                    "hypothesis_count": hypothesis_count,
                    "runtime_seconds": elapsed[action_id],
                    "milliseconds_per_case": (
                        1_000.0 * elapsed[action_id] / len(native_ids)
                    ),
                }
                self._write(summary)
                summaries.append(summary)
        return tuple(summaries)

    def _write(self, row: Mapping[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")

    def _checkpoint(self) -> dict[str, Any]:
        return {
            "schema": INTEGRATED_TRAINER_SCHEMA_V2,
            "config": self.config.serializable(),
            "config_digest": self.config.digest,
            "action_bank_lineage": asdict(self.config.action_bank),
            "uncertainty_initialization_lineage": self.initialization_lineage,
            "refiner_initialization_lineage": self.refiner_initialization_lineage,
            "model_initialization_lineage": self.model_initialization_lineage,
            "selection_metric": self.selection_metric,
            "best_validation_score": self.best_validation_score,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "next_batch_index": self.next_batch_index,
            "global_step": self.global_step,
            "best_validation_epe": self.best_validation_epe,
            "sam_semantic_cache": (
                None
                if self.sam_semantic_cache is None
                else self.sam_semantic_cache.state_dict()
            ),
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
        # RNG state is defined as a CPU ByteTensor by torch.  Loading the whole
        # checkpoint directly onto CUDA corrupts that contract before restore.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != INTEGRATED_TRAINER_SCHEMA_V2 or payload.get("config_digest") != self.config.digest:
            raise ValueError("integrated checkpoint config/schema mismatch")
        if payload.get("action_bank_lineage") != asdict(self.config.action_bank):
            raise ValueError("integrated checkpoint action-bank lineage mismatch")
        if payload.get("uncertainty_initialization_lineage") != self.initialization_lineage:
            raise ValueError("integrated checkpoint uncertainty initialization drift")
        if payload.get("refiner_initialization_lineage") != self.refiner_initialization_lineage:
            raise ValueError("integrated checkpoint refiner initialization drift")
        if payload.get("model_initialization_lineage") != self.model_initialization_lineage:
            raise ValueError("integrated checkpoint model initialization drift")
        if payload.get("selection_metric") != self.selection_metric:
            raise ValueError("integrated checkpoint selection metric drift")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in tuple(optimizer_state.items()):
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(self.device)
        self.scaler.load_state_dict(payload["scaler"])
        self.epoch = int(payload["epoch"])
        self.next_batch_index = int(payload["next_batch_index"])
        self.global_step = int(payload["global_step"])
        self.best_validation_score = float(payload["best_validation_score"])
        self.best_validation_epe = float(payload["best_validation_epe"])
        cache_payload = payload.get("sam_semantic_cache")
        if self.sam_semantic_cache is None:
            if cache_payload is not None:
                raise ValueError("checkpoint unexpectedly contains a SAM semantic cache")
        else:
            if not isinstance(cache_payload, Mapping):
                raise ValueError("SAM semantic checkpoint cache is missing")
            self.sam_semantic_cache.load_state_dict(cache_payload)
        _restore_rng(payload["rng"])

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float | None]:
        self.model.eval()
        totals = {
            "total": 0.0, "task": 0.0, "augmentation": 0.0,
            "uncertainty": 0.0, "harm": 0.0, "anchor": 0.0,
            "smoothness": 0.0, "sam_homography": 0.0,
            "sam_semantic": 0.0,
        }
        count = 0
        all_alpha, all_error, all_valid = [], [], []
        endpoint_error_sum = 0.0
        endpoint_error_count = 0
        base_endpoint_error_sum = 0.0
        base_endpoint_error_count = 0
        fusion_endpoint_error_sum = 0.0
        fusion_endpoint_error_count = 0
        fusion_replacement_count = 0
        fusion_pixel_count = 0
        for raw in self._loader(self.validation_dataset, training=False, epoch=epoch):
            batch = self._move(raw)
            with self._autocast():
                losses, native_output, base_output = self._forward_loss(
                    batch, augmentation_enabled=True,
                )
            batch_size = int(batch["native_frames"].shape[0])
            count += batch_size
            for name in totals:
                component = getattr(losses, name, None)
                if component is not None:
                    totals[name] += float(component.detach().cpu()) * batch_size
            final_flow = self._flows(native_output)[-1]
            final_alpha = self._alphas(native_output)[-1]
            if batch.get("ground_truth_flow") is None or batch.get("ground_truth_valid") is None:
                raise ValueError("validation requires held-out ground truth")
            endpoint_error = torch.linalg.vector_norm(
                final_flow - batch["ground_truth_flow"].float(), dim=1,
            )
            valid = batch["ground_truth_valid"].bool()[:, 0]
            endpoint_error_sum += float(endpoint_error[valid].double().sum().cpu())
            endpoint_error_count += int(valid.sum().cpu())
            fusion_policy = self.config.fusion
            if fusion_policy is not None and fusion_policy.enabled:
                fusion_frames = batch.get("fusion_frames")
                if not isinstance(fusion_frames, torch.Tensor):
                    raise ValueError(
                        "enabled U2Flow fusion requires dataset fusion_frames"
                    )
                if not torch.equal(fusion_frames[:, 1:], batch["native_frames"]):
                    raise ValueError(
                        "fusion current/next frames differ from native validation pair"
                    )
                _action_id, matcher_iterations = self._batch_action(batch)
                fused = fuse_u2flow_triplet_prediction_v1(
                    self.model,
                    fusion_frames.float(),
                    final_flow,
                    final_alpha,
                    matcher_iterations=matcher_iterations,
                    policy=fusion_policy,
                )
                fusion_error = torch.linalg.vector_norm(
                    fused.fused_flow - batch["ground_truth_flow"].float(), dim=1,
                )
                fusion_endpoint_error_sum += float(
                    fusion_error[valid].double().sum().cpu()
                )
                fusion_endpoint_error_count += int(valid.sum().cpu())
                replacement = fused.replacement_mask[:, 0] & valid
                fusion_replacement_count += int(replacement.sum().cpu())
                fusion_pixel_count += int(valid.sum().cpu())
            if base_output is not None:
                base_error = torch.linalg.vector_norm(
                    self._flows(base_output)[-1] - batch["ground_truth_flow"].float(),
                    dim=1,
                )
                base_endpoint_error_sum += float(base_error[valid].double().sum().cpu())
                base_endpoint_error_count += int(valid.sum().cpu())
            # Eq. 9 corresponds to b=exp(alpha/2)/sqrt(2).
            log_scale = 0.5 * final_alpha[:, 0] - 0.5 * math.log(2.0)
            stride = self.config.metric_spatial_stride
            all_alpha.append(log_scale[..., ::stride, ::stride].detach().float().cpu().numpy())
            all_error.append(endpoint_error[..., ::stride, ::stride].detach().float().cpu().numpy())
            all_valid.append(valid[..., ::stride, ::stride].detach().cpu().numpy())
        if count == 0:
            raise ValueError("validation loader is empty")
        evaluated = evaluate_log_scale_uncertainty_v1(
            np.concatenate(all_alpha), np.concatenate(all_error), np.concatenate(all_valid),
            severe_threshold=self.config.severe_error_threshold,
        )
        result = {
            name: value / count
            for name, value in totals.items()
            if self.config.refiner_loss is not None
            or name not in {"harm", "anchor", "smoothness"}
        }
        def finite_or_none(value: float) -> float | None:
            return float(value) if math.isfinite(float(value)) else None

        if endpoint_error_count == 0:
            raise ValueError("validation has empty ground-truth support")
        result.update({
            "epe": endpoint_error_sum / endpoint_error_count,
            "ause": evaluated.ause,
            "spearman": finite_or_none(evaluated.spearman_error),
            "severe_auroc": finite_or_none(evaluated.severe_auroc),
            "calibration_mae": evaluated.coverage_calibration_mae,
            "metric_valid_count": evaluated.valid_count,
            "metric_spatial_stride": self.config.metric_spatial_stride,
        })
        if base_endpoint_error_count:
            result.update({
                "base_epe": base_endpoint_error_sum / base_endpoint_error_count,
                "harm": totals["harm"] / count,
                "anchor": totals["anchor"] / count,
                "smoothness": totals["smoothness"] / count,
            })
        if fusion_endpoint_error_count:
            result.update({
                "fusion_epe": fusion_endpoint_error_sum / fusion_endpoint_error_count,
                "fusion_replaced_fraction": (
                    fusion_replacement_count / fusion_pixel_count
                    if fusion_pixel_count else 0.0
                ),
                "fusion_enabled": True,
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
        self._write({
            "event": "training_started",
            "utc": _utc_now(),
            "config_digest": self.config.digest,
            "u2_schedule_mode": (
                None
                if self.config.u2_schedule is None
                else self.config.u2_schedule.mode
            ),
            "action_schedule_mode": self.config.action_bank.mode,
            "gradient_diagnostics": asdict(self.config.gradient_diagnostics),
            "action_shadow_probe": asdict(self.config.action_shadow_probe),
            "sam_semantic_augmentation": asdict(
                self.config.sam_semantic_augmentation
            ),
            "sam_homography": asdict(self.config.sam_homography),
        })
        if self.cuda_memory_guard.active:
            self._write({
                "event": "cuda_memory_reserved", "utc": _utc_now(),
                **self.cuda_memory_report,
            })
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.epoch, self.options.epochs):
            phase, round_index = self._configure_training_phase(epoch)
            self.optimizer.zero_grad(set_to_none=True)
            loader = self._loader(self.fit_dataset, training=True, epoch=epoch)
            for batch_index, raw in enumerate(loader):
                if epoch == self.epoch and batch_index < self.next_batch_index:
                    continue
                batch = self._move(raw)
                with self._autocast():
                    losses, _native_output, _base_output = self._forward_loss(
                        batch,
                        augmentation_enabled=epoch >= self.options.augmentation_start_epoch,
                        semantic_augmentation_enabled=(
                            self.config.sam_semantic_augmentation.enabled
                            and epoch
                            >= self.config.sam_semantic_augmentation.activation_epoch
                        ),
                        semantic_cache_update_enabled=(
                            self.config.sam_semantic_augmentation.enabled
                        ),
                        phase=phase,
                    )
                    scaled_loss = losses.total / self.options.gradient_accumulation_steps
                end_accumulation = (
                    (batch_index + 1) % self.options.gradient_accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                diagnostics: dict[str, Any] = {}
                diagnostic_policy = self.config.gradient_diagnostics
                if (
                    diagnostic_policy.enabled
                    and phase in {None, "joint"}
                    and end_accumulation
                    and (self.global_step + 1) % diagnostic_policy.every_steps == 0
                ):
                    diagnostics = self._gradient_diagnostics(losses)
                self.scaler.scale(scaled_loss).backward()
                optimizer_step_skipped = False
                optimizer_scale_before: float | None = None
                optimizer_scale_after: float | None = None
                if end_accumulation:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable, self.options.gradient_clip_norm)
                    optimizer_scale_before = float(self.scaler.get_scale())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    optimizer_scale_after = float(self.scaler.get_scale())
                    # GradScaler skips the underlying optimizer step when it
                    # observes non-finite gradients and decreases its scale.
                    # Such an attempt must not advance the scientific update
                    # counter or checkpoint cadence.
                    optimizer_step_skipped = (
                        self.amp_enabled
                        and optimizer_scale_after < optimizer_scale_before
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    if not optimizer_step_skipped:
                        self.global_step += 1
                    if self.cuda_memory_guard.active:
                        self.cuda_memory_report = self.cuda_memory_guard.refill()
                self.next_batch_index = batch_index + 1
                if end_accumulation:
                    action_id, matcher_iters = self._batch_action(batch)
                    row = {
                        "event": "train_step", "utc": _utc_now(), "epoch": epoch,
                        "global_step": self.global_step,
                        "loss": float(losses.total.detach().cpu()),
                        "task": float(losses.task.detach().cpu()),
                        "augmentation": float(losses.augmentation.detach().cpu()),
                        "uncertainty": float(losses.uncertainty.detach().cpu()),
                        "action_id": action_id,
                        "matcher_iterations": matcher_iters,
                        "optimizer_step_skipped": optimizer_step_skipped,
                        "optimizer_scale_before": optimizer_scale_before,
                        "optimizer_scale_after": optimizer_scale_after,
                        **diagnostics,
                    }
                    if phase is not None:
                        row.update({"phase": phase, "round_index": round_index})
                    for name in ("harm", "anchor", "smoothness"):
                        component = getattr(losses, name, None)
                        if component is not None:
                            row[name] = float(component.detach().cpu())
                    sam_component = getattr(losses, "sam_homography", None)
                    if sam_component is not None:
                        row.update({
                            "sam_homography": float(sam_component.detach().cpu()),
                            "sam_candidate_regions": int(
                                getattr(losses, "sam_candidate_regions", 0)
                            ),
                            "sam_fitted_regions": int(
                                getattr(losses, "sam_fitted_regions", 0)
                            ),
                        })
                    semantic_component = getattr(losses, "sam_semantic", None)
                    if semantic_component is not None:
                        row.update({
                            "sam_semantic": float(
                                semantic_component.detach().cpu()
                            ),
                            "sam_semantic_objects": int(
                                getattr(losses, "sam_semantic_objects", 0)
                            ),
                            "sam_semantic_object_pixels": int(
                                getattr(losses, "sam_semantic_object_pixels", 0)
                            ),
                            "sam_semantic_cache_count": (
                                0
                                if self.sam_semantic_cache is None
                                else self.sam_semantic_cache.count
                            ),
                        })
                    self._write(row)
                    if (
                        not optimizer_step_skipped
                        and self.global_step % self.options.checkpoint_every_steps == 0
                    ):
                        self.save()
            self.epoch = epoch + 1
            self.next_batch_index = 0
            if self.epoch % self.options.validation_every_epochs == 0:
                metrics = self.validate(epoch)
                validation_epe = metrics["epe"]
                selection_score = metrics[self.selection_metric]
                assert isinstance(validation_epe, float)
                assert isinstance(selection_score, float)
                self.best_validation_epe = min(self.best_validation_epe, validation_epe)
                improved = selection_score < self.best_validation_score
                if improved:
                    self.best_validation_score = selection_score
                self._write({
                    "event": "validation_epoch", "utc": _utc_now(), "epoch": epoch,
                    "global_step": self.global_step, "best": improved,
                    "selection_metric": self.selection_metric,
                    "selection_score": selection_score,
                    **(
                        {}
                        if phase is None
                        else {"completed_phase": phase, "round_index": round_index}
                    ),
                    **metrics,
                })
                shadow_policy = self.config.action_shadow_probe
                if (
                    shadow_policy.enabled
                    and self.epoch % shadow_policy.every_epochs == 0
                ):
                    self.action_shadow_probe(epoch)
                self.save()
                if improved:
                    self.save(self.best_path)
        self._write({
            "event": "training_completed", "utc": _utc_now(),
            "epochs": self.options.epochs, "global_step": self.global_step,
            "selection_metric": self.selection_metric,
            "best_validation_score": self.best_validation_score,
            "best_validation_epe": self.best_validation_epe,
        })
        self.save()
        return 0


__all__ = [
    "INTEGRATED_TRAINER_SCHEMA_V2",
    "ActionShadowProbePolicyV1",
    "ActionBankTrainingPolicyV1",
    "AlternatingU2ScheduleV2",
    "GradientDiagnosticsPolicyV1",
    "JointDecoupledU2ScheduleV2",
    "IntegratedTrainerConfigV2",
    "IntegratedTrainingOptionsV2",
    "IntegratedUncertaintyTrainerV2",
]
