"""Optional WAFT provider for cross-backbone uncertainty/refinement checks.

This adapter is intentionally not selected by any active schedule.  It gives
WAFT the same trainer-facing contract as the SEA-RAFT path: recurrent flow
predictions, one log-variance map per prediction, a zero-initialized bounded
repair head, and independently trainable observer/flow phases.
"""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import torch
from torch import nn

from .integrated_uncertainty_flow import (
    RecurrentUncertaintyVariantV2,
    TrainableScopeV2,
)


class WaftUncertaintyRefinementHeadV1(nn.Module):
    def __init__(self, *, hidden_channels: int, maximum_update_px: float) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("WAFT adapter hidden channels must be positive")
        if not math.isfinite(float(maximum_update_px)) or maximum_update_px <= 0.0:
            raise ValueError("WAFT maximum update must be finite and positive")
        self.maximum_update_px = float(maximum_update_px)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(6, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )
        self.refinement_head = nn.Sequential(
            nn.Conv2d(5, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.refinement_head[-1].weight)
        nn.init.zeros_(self.refinement_head[-1].bias)

    def observe(self, flow: torch.Tensor, info: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError("WAFT flow must have shape N2HW")
        if info.ndim != 4 or info.shape[1] != 4 or info.shape[0] != flow.shape[0]:
            raise ValueError("WAFT info must have shape N4HW")
        if info.shape[-2:] != flow.shape[-2:]:
            raise ValueError("WAFT flow and info grids must match")
        return self.uncertainty_head(torch.cat((flow, info), dim=1))

    def refine(self, flow: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        detached_alpha = alpha.detach()
        reliability = torch.sigmoid(-detached_alpha)
        raw = self.refinement_head(torch.cat(
            (flow, flow * reliability, detached_alpha), dim=1,
        ))
        magnitude = torch.linalg.vector_norm(raw, dim=1, keepdim=True)
        scale = torch.clamp(
            self.maximum_update_px / magnitude.clamp_min(1e-12), max=1.0,
        )
        return flow + raw * scale


class UncertaintyAwareWaftV1(nn.Module):
    """Post-recurrent WAFT adapter with the integrated trainer contract."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        variant: RecurrentUncertaintyVariantV2 | str,
        trainable_scope: TrainableScopeV2 | str = TrainableScopeV2.ALL,
        refinement_channels: int = 64,
        maximum_update_px: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("WAFT backbone must be a torch module")
        self.backbone = backbone
        self.variant = RecurrentUncertaintyVariantV2(variant)
        self.trainable_scope = TrainableScopeV2(trainable_scope)
        self.recurrent_head = WaftUncertaintyRefinementHeadV1(
            hidden_channels=refinement_channels,
            maximum_update_px=maximum_update_px,
        )
        self.refinement_enabled = True
        self._apply_trainable_scope()

    def _apply_trainable_scope(self) -> None:
        self.requires_grad_(self.trainable_scope is TrainableScopeV2.ALL)
        if self.trainable_scope in {
            TrainableScopeV2.UNCERTAINTY_ONLY,
            TrainableScopeV2.REFINEMENT_HEADS,
        }:
            self.recurrent_head.uncertainty_head.requires_grad_(True)
        if self.trainable_scope in {
            TrainableScopeV2.REFINER_ONLY,
            TrainableScopeV2.REFINEMENT_HEADS,
        }:
            self.recurrent_head.refinement_head.requires_grad_(
                self.variant is not RecurrentUncertaintyVariantV2.HEAD_ONLY
            )

    @contextmanager
    def refinement_disabled(self):
        previous = self.refinement_enabled
        self.refinement_enabled = False
        try:
            yield
        finally:
            self.refinement_enabled = previous

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.trainable_scope is not TrainableScopeV2.ALL:
            self.backbone.eval()
        return self

    def forward(
        self, image1: torch.Tensor, image2: torch.Tensor, *,
        iters: int | None = None, test_mode: bool | None = None,
    ) -> dict[str, object]:
        del test_mode
        output = self.backbone(image1, image2, iters=iters)
        if not isinstance(output, Mapping):
            raise TypeError("WAFT output must be a mapping")
        flows = output.get("flow")
        infos = output.get("info")
        if (
            not isinstance(flows, (tuple, list)) or not flows
            or not isinstance(infos, (tuple, list)) or len(infos) != len(flows)
        ):
            raise ValueError("WAFT flow/info sequences must be nonempty and aligned")
        refined: list[torch.Tensor] = []
        alphas: list[torch.Tensor] = []
        for flow, info in zip(flows, infos):
            if not isinstance(flow, torch.Tensor) or not isinstance(info, torch.Tensor):
                raise TypeError("WAFT flow/info predictions must be tensors")
            alpha = self.recurrent_head.observe(flow, info)
            alphas.append(alpha)
            if (
                self.variant is RecurrentUncertaintyVariantV2.HEAD_ONLY
                or not self.refinement_enabled
            ):
                refined.append(flow)
            else:
                refined.append(self.recurrent_head.refine(flow, alpha))
        result = dict(output)
        result["flow"] = tuple(refined)
        result["uncertainty_log_variance"] = tuple(alphas)
        result["final"] = refined[-1]
        return result


def load_official_waft_model_v1(
    *, vendor_root: Path, config_path: Path, checkpoint: Path, device: str,
) -> nn.Module:
    """Load an explicit official WAFT checkout/config/checkpoint tuple."""

    vendor_root = vendor_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    for path in (vendor_root / "model" / "__init__.py", config_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("WAFT config must contain an object")
    args = argparse.Namespace(**dict(config))
    previous_model = sys.modules.get("model")
    if previous_model is not None:
        module_file = Path(str(getattr(previous_model, "__file__", ""))).resolve()
        if vendor_root not in module_file.parents:
            raise RuntimeError("top-level module name 'model' is already owned by another provider")
    sys.path.insert(0, str(vendor_root))
    try:
        from model import fetch_model
        model = fetch_model(args)
    finally:
        sys.path.remove(str(vendor_root))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("WAFT checkpoint must contain a state mapping")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "WAFT checkpoint is not exact: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model.to(torch.device(device))


def make_pinned_uncertainty_aware_waft_v1(
    *, device: str, variant: str, trainable_scope: str = "all",
    refinement_channels: int = 64, maximum_update_px: float = 1.0,
    vendor_root: str, config_path: str, checkpoint: str,
) -> UncertaintyAwareWaftV1:
    backbone = load_official_waft_model_v1(
        vendor_root=Path(vendor_root), config_path=Path(config_path),
        checkpoint=Path(checkpoint), device=device,
    )
    return UncertaintyAwareWaftV1(
        backbone,
        variant=variant,
        trainable_scope=trainable_scope,
        refinement_channels=refinement_channels,
        maximum_update_px=maximum_update_px,
    ).to(torch.device(device))


__all__ = [
    "UncertaintyAwareWaftV1",
    "WaftUncertaintyRefinementHeadV1",
    "load_official_waft_model_v1",
    "make_pinned_uncertainty_aware_waft_v1",
]
