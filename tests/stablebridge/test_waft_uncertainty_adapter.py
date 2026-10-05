from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from stablebridge.physical_repair.waft_uncertainty_adapter import (
    UncertaintyAwareWaftV1,
)


class TinyWaft(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow = nn.Conv2d(6, 2, 1)
        self.info = nn.Conv2d(6, 4, 1)

    def forward(self, first, second, *, iters=None):
        count = 2 if iters is None else iters
        features = torch.cat((first, second), dim=1) / 255.0
        flow = self.flow(features)
        info = self.info(features)
        return {
            "flow": [flow * (index + 1) / count for index in range(count)],
            "info": [info for _ in range(count)],
        }


def test_waft_adapter_preserves_base_at_initialization_and_exposes_contract():
    model = UncertaintyAwareWaftV1(
        TinyWaft(), variant="refinement_with_uncertainty",
        trainable_scope="all", refinement_channels=8, maximum_update_px=1.0,
    )
    first = torch.rand(2, 3, 5, 6) * 255.0
    second = torch.rand(2, 3, 5, 6) * 255.0
    base = model.backbone(first, second, iters=3)
    output = model(first, second, iters=3, test_mode=True)
    assert len(output["flow"]) == len(output["uncertainty_log_variance"]) == 3
    assert torch.equal(output["flow"][-1], base["flow"][-1])
    assert output["uncertainty_log_variance"][-1].shape == (2, 1, 5, 6)
    assert torch.equal(output["final"], output["flow"][-1])


def test_waft_adapter_head_only_scope_freezes_provider():
    model = UncertaintyAwareWaftV1(
        TinyWaft(), variant="head_only", trainable_scope="uncertainty_only",
        refinement_channels=8,
    )
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.recurrent_head.uncertainty_head.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.recurrent_head.refinement_head.parameters()
    )


def test_waft_provider_manifest_is_explicitly_disabled():
    root = Path(__file__).resolve().parents[2]
    value = json.loads(
        (root / "configs/stablebridge/waft_cross_backbone_disabled_v1.json")
        .read_text(encoding="utf-8")
    )
    assert value["enabled"] is False
    assert value["upstream_commit"] == "b152ff1cad1af8c185ee7b141997c48ff3334c87"
    assert "not_scheduled" in value["activation_status"]
