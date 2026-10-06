"""CPU-only tests for the frozen SEA-RAFT uncertainty adapter."""

from __future__ import annotations

from argparse import Namespace
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from stablebridge.physical_repair.sea_raft_uncertainty_adapter import (
    FrozenSeaRaftPredictor,
    decode_sea_raft_mixture_uncertainty,
    load_pinned_sea_raft_predictor,
)


class FakeSeaRaft(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.args = Namespace(iters=4, use_var=True, var_min=-2, var_max=3)
        self.calls: list[tuple[int, bool, bool]] = []

    def forward(self, image1, image2, *, iters, test_mode):
        self.calls.append((iters, test_mode, torch.is_inference_mode_enabled()))
        flow = (image2[:, :2] - image1[:, :2]) * self.weight
        logits = torch.cat((torch.zeros_like(flow[:, :1]), torch.ones_like(flow[:, :1])), 1)
        raw_log_b = torch.cat((torch.full_like(flow[:, :1], 9), torch.full_like(flow[:, :1], -9)), 1)
        info = torch.cat((logits, raw_log_b), 1)
        return {"final": flow, "flow": [flow * 0, flow], "info": [info * 0, info], "nf": None}


def test_decoder_follows_official_mixture_laplace_semantics():
    # p = [1/4, 3/4], with official asymmetric clamps yielding b = [e^2, e^-1].
    logits = torch.tensor([math.log(1 / 4), math.log(3 / 4)]).view(1, 2, 1, 1)
    raw = torch.tensor([8.0, -8.0]).view(1, 2, 1, 1)
    decoded = decode_sea_raft_mixture_uncertainty(
        torch.cat((logits, raw), 1), var_min=-1, var_max=2,
    )
    expected_axis = 0.25 * math.exp(2) + 0.75 * math.exp(-1)
    assert decoded.probabilities[:, 0].item() == pytest.approx(0.25)
    assert decoded.log_scales.flatten().tolist() == [2.0, -1.0]
    assert decoded.expected_absolute_axis_error.item() == pytest.approx(expected_axis)
    assert decoded.conservative_l1_error.item() == pytest.approx(2 * expected_axis)
    assert decoded.upstream_heatmap_log_scale.item() == pytest.approx(-0.25)


def test_predictor_is_frozen_and_exposes_flow_info_and_flow_only_callable():
    model = FakeSeaRaft().train()
    predictor = FrozenSeaRaftPredictor(
        model, iters=5, var_min=-2, var_max=3,
    )
    assert not model.training
    assert not model.weight.requires_grad
    image1 = torch.zeros(2, 3, 4, 5, dtype=torch.float32, requires_grad=True)
    image2 = torch.ones_like(image1)
    prediction = predictor(image1, image2)
    assert prediction.flow.shape == (2, 2, 4, 5)
    assert prediction.info.shape == (2, 4, 4, 5)
    assert prediction.uncertainty.conservative_l1_error.shape == (2, 1, 4, 5)
    assert not prediction.flow.requires_grad
    assert model.calls[-1] == (5, True, True)
    assert callable(predictor.predict_flow)
    torch.testing.assert_close(predictor.predict_flow(image1, image2), prediction.flow)


def test_predictor_fails_closed_on_bad_native_output():
    class BadInfo(FakeSeaRaft):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            output["info"][-1] = output["info"][-1][:, :3]
            return output

    predictor = FrozenSeaRaftPredictor(BadInfo(), iters=4, var_min=0, var_max=10)
    images = torch.zeros(1, 3, 3, 4)
    with pytest.raises(ValueError, match="final info"):
        predictor(images, images)


@pytest.mark.parametrize(
    "info,error",
    [
        (torch.zeros(1, 3, 2, 2), "shape"),
        (torch.zeros(1, 4, 2, 2, dtype=torch.int64), "floating"),
        (torch.full((1, 4, 2, 2), float("nan")), "finite"),
    ],
)
def test_decoder_rejects_invalid_info(info, error):
    with pytest.raises((TypeError, ValueError), match=error):
        decode_sea_raft_mixture_uncertainty(info, var_min=0, var_max=10)


def test_pinned_loader_uses_portable_identity_boundary_and_explicit_paths():
    from stablebridge.physical_repair import sea_raft_loader

    model = FakeSeaRaft()
    report = {"vendor_commit": sea_raft_loader.SEA_RAFT_VENDOR_COMMIT}
    vendor = Path("/portable/vendor/SEA-RAFT")
    config = vendor / "config/eval/spring-M.json"
    checkpoint = Path("/portable/checkpoints/model.safetensors")
    with patch.object(
        sea_raft_loader,
        "load_pinned_sea_raft_official_model",
        return_value=(model, object(), object(), report),
    ) as loader:
        predictor, actual_report = load_pinned_sea_raft_predictor(
            device="cpu", vendor_root=vendor, config_path=config,
            checkpoint=checkpoint,
        )
    assert actual_report is report
    assert predictor.model is model
    assert predictor.iters == 4
    assert predictor.var_min == -2
    assert predictor.var_max == 3
    loader.assert_called_once_with(
        vendor_root=vendor,
        config_path=config,
        checkpoint=checkpoint,
        device="cpu",
    )
