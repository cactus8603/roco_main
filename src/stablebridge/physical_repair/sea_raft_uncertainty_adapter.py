"""Frozen SEA-RAFT inference and its native mixture uncertainty.

The portable pinned Spring-M loader lives beside this module in
``sea_raft_loader.py``.  This module keeps that identity/load boundary and
adds only a small training-facing adapter.

SEA-RAFT emits four ``info`` channels per pixel: two unnormalised mixture
weights followed by the raw log scales of a large- and a small-scale Laplace
component.  The decoder below follows the likelihood implementation in the
official ``core/raft.py``.  Its scalar baseline is the model-implied expected
L1 error of the two flow coordinates.  Since pointwise EPE is no larger than
L1 error, this is deliberately more conservative than the upstream demo's
weighted-mean-log-scale visualisation.  It is not a calibrated coverage
bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class SeaRaftMixtureUncertainty:
    """Decoded native SEA-RAFT mixture quantities, all on the input grid."""

    probabilities: torch.Tensor
    log_scales: torch.Tensor
    scales: torch.Tensor
    expected_absolute_axis_error: torch.Tensor
    conservative_l1_error: torch.Tensor
    upstream_heatmap_log_scale: torch.Tensor


@dataclass(frozen=True)
class SeaRaftNativePrediction:
    """Final frozen native prediction and unmodified SEA-RAFT ``info``."""

    flow: torch.Tensor
    info: torch.Tensor
    uncertainty: SeaRaftMixtureUncertainty


def decode_sea_raft_mixture_uncertainty(
    info: torch.Tensor,
    *,
    var_min: float,
    var_max: float,
) -> SeaRaftMixtureUncertainty:
    """Decode ``info`` exactly as SEA-RAFT's mixture-Laplace likelihood does.

    ``expected_absolute_axis_error`` is ``sum_k pi_k * b_k`` because a
    zero-centred Laplace distribution with scale ``b`` has ``E|X| = b``.
    SEA-RAFT uses the same two-component marginal for each flow coordinate,
    so twice that value is the expected L1 flow error.  This L1 quantity is a
    conservative scalar proxy for EPE under the model distribution.
    """

    if not isinstance(info, torch.Tensor):
        raise TypeError("SEA-RAFT info must be a torch tensor")
    if info.ndim != 4 or info.shape[1] != 4:
        raise ValueError("SEA-RAFT info must have shape [B,4,H,W]")
    if not info.is_floating_point():
        raise TypeError("SEA-RAFT info must have a floating dtype")
    if not bool(torch.isfinite(info).all()):
        raise ValueError("SEA-RAFT info must be finite")
    if not (float(var_min) <= 0.0 <= float(var_max)):
        raise ValueError("SEA-RAFT scale bounds require var_min <= 0 <= var_max")

    logits = info[:, :2]
    raw_log_scales = info[:, 2:]
    probabilities = torch.softmax(logits, dim=1)
    # This asymmetric clamp is the official loss semantics: component zero
    # is the large-b component and component one is the small-b component.
    large_log_scale = raw_log_scales[:, 0:1].clamp(min=0.0, max=float(var_max))
    small_log_scale = raw_log_scales[:, 1:2].clamp(min=float(var_min), max=0.0)
    log_scales = torch.cat((large_log_scale, small_log_scale), dim=1)
    scales = torch.exp(log_scales)
    expected_absolute_axis_error = (probabilities * scales).sum(dim=1, keepdim=True)
    conservative_l1_error = 2.0 * expected_absolute_axis_error
    upstream_heatmap_log_scale = (
        probabilities * log_scales
    ).sum(dim=1, keepdim=True)
    return SeaRaftMixtureUncertainty(
        probabilities=probabilities,
        log_scales=log_scales,
        scales=scales,
        expected_absolute_axis_error=expected_absolute_axis_error,
        conservative_l1_error=conservative_l1_error,
        upstream_heatmap_log_scale=upstream_heatmap_log_scale,
    )


def _validate_image_pair(image1: torch.Tensor, image2: torch.Tensor) -> None:
    if not isinstance(image1, torch.Tensor) or not isinstance(image2, torch.Tensor):
        raise TypeError("SEA-RAFT images must be torch tensors")
    if image1.shape != image2.shape or image1.ndim != 4 or image1.shape[1] != 3:
        raise ValueError("expected matching SEA-RAFT images [B,3,H,W]")
    if image1.dtype != torch.float32 or image2.dtype != torch.float32:
        raise TypeError("SEA-RAFT images must use float32")
    if image1.device != image2.device:
        raise ValueError("SEA-RAFT images must share a device")
    if not bool(torch.isfinite(image1).all()) or not bool(torch.isfinite(image2).all()):
        raise ValueError("SEA-RAFT images must be finite")


def _final_tensor(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"SEA-RAFT output {name!r} must be a non-empty sequence")
    final = value[-1]
    if not isinstance(final, torch.Tensor):
        raise TypeError(f"SEA-RAFT final {name!r} must be a torch tensor")
    return final


class FrozenSeaRaftPredictor:
    """Callable frozen SEA-RAFT native predictor for U0 teacher generation.

    Calling the object returns flow, raw info, and decoded uncertainty.
    ``predict_flow`` is the narrower callable expected by augmentation-
    consistency target builders; it performs the same frozen native forward
    and returns only the final flow.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        iters: int,
        var_min: float,
        var_max: float,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("SEA-RAFT model must be a torch module")
        if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
            raise ValueError("SEA-RAFT iters must be a positive integer")
        if not (float(var_min) <= 0.0 <= float(var_max)):
            raise ValueError("SEA-RAFT scale bounds require var_min <= 0 <= var_max")
        self.model = model.eval()
        self.iters = iters
        self.var_min = float(var_min)
        self.var_max = float(var_max)
        self.model.requires_grad_(False)

    def __call__(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> SeaRaftNativePrediction:
        _validate_image_pair(image1, image2)
        # Reassert eval mode in case an enclosing training harness called
        # ``train()`` recursively on all owned modules.
        self.model.eval()
        with torch.inference_mode():
            output = self.model(
                image1,
                image2,
                iters=self.iters,
                test_mode=True,
            )
        if not isinstance(output, Mapping):
            raise TypeError("SEA-RAFT output must be a mapping")
        flow = _final_tensor(output.get("flow"), name="flow")
        info = _final_tensor(output.get("info"), name="info")
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError("SEA-RAFT final flow must have shape [B,2,H,W]")
        if info.ndim != 4 or info.shape[1] != 4:
            raise ValueError("SEA-RAFT final info must have shape [B,4,H,W]")
        if flow.shape[0] != info.shape[0] or flow.shape[-2:] != info.shape[-2:]:
            raise ValueError("SEA-RAFT final flow/info grids must match")
        if flow.shape[0] != image1.shape[0] or flow.shape[-2:] != image1.shape[-2:]:
            raise ValueError("SEA-RAFT output must be unpadded to the input grid")
        if not flow.is_floating_point() or not bool(torch.isfinite(flow).all()):
            raise ValueError("SEA-RAFT final flow must be finite floating point")
        uncertainty = decode_sea_raft_mixture_uncertainty(
            info,
            var_min=self.var_min,
            var_max=self.var_max,
        )
        final_alias = output.get("final")
        if final_alias is not None:
            if not isinstance(final_alias, torch.Tensor) or final_alias.shape != flow.shape:
                raise ValueError("SEA-RAFT final alias does not match final flow")
            if not torch.equal(final_alias, flow):
                raise ValueError("SEA-RAFT final alias differs from flow[-1]")
        return SeaRaftNativePrediction(flow=flow, info=info, uncertainty=uncertainty)

    def predict_flow(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Frozen flow-only callable for native or augmented teacher pairs."""

        return self(image1, image2).flow


def load_pinned_sea_raft_model(
    *,
    device: str,
    vendor_root: Path | None = None,
    config_path: Path | None = None,
    checkpoint: Path | None = None,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Load the exact E58-pinned official SEA-RAFT model without freezing it.

    Training wrappers use this boundary before installing their recurrent
    uncertainty-aware flow head.  Frozen inference remains the responsibility
    of :func:`load_pinned_sea_raft_predictor`.
    """

    # Keep heavyweight vendor imports and safetensors loading outside module
    # import so CPU contract tests need neither the checkpoint nor CUDA.
    if vendor_root is None or config_path is None or checkpoint is None:
        raise ValueError(
            "portable SEA-RAFT loading requires explicit vendor, config, "
            "and checkpoint paths"
        )
    from .sea_raft_loader import load_pinned_sea_raft_official_model

    model, _corr_factory, _utils, report = load_pinned_sea_raft_official_model(
        vendor_root=vendor_root,
        config_path=config_path,
        checkpoint=checkpoint,
        device=device,
    )
    return model, report


def load_pinned_sea_raft_predictor(
    *,
    device: str,
    iters: int | None = None,
    vendor_root: Path | None = None,
    config_path: Path | None = None,
    checkpoint: Path | None = None,
) -> tuple[FrozenSeaRaftPredictor, dict[str, object]]:
    """Load the exact E58-pinned official SEA-RAFT model as a frozen adapter."""

    model, report = load_pinned_sea_raft_model(
        device=device,
        vendor_root=vendor_root,
        config_path=config_path,
        checkpoint=checkpoint,
    )
    args = model.args
    resolved_iters = int(args.iters) if iters is None else iters
    if bool(getattr(args, "use_var", True)):
        var_min = float(args.var_min)
        var_max = float(args.var_max)
    else:
        var_min = var_max = 0.0
    predictor = FrozenSeaRaftPredictor(
        model,
        iters=resolved_iters,
        var_min=var_min,
        var_max=var_max,
    )
    return predictor, report


__all__ = [
    "FrozenSeaRaftPredictor",
    "SeaRaftMixtureUncertainty",
    "SeaRaftNativePrediction",
    "decode_sea_raft_mixture_uncertainty",
    "load_pinned_sea_raft_model",
    "load_pinned_sea_raft_predictor",
]
