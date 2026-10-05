from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn
import torch.nn.functional as F

from stablebridge.physical_repair.integrated_uncertainty_flow import (
    DecoupledFlowLossPolicyV2,
    UncertaintyAwareSeaRaftFlowHeadV2,
    UncertaintyAwareSeaRaftV2,
    decoupled_uncertainty_flow_loss_v2,
    transform_flow_affine_v2,
)


class TinySeaRaft(nn.Module):
    hidden_channels = 4

    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)
        self.flow_head = nn.Conv2d(4, 6, 1)

    def forward(self, image1, image2, *, iters=2, test_mode=True):
        del image2, test_mode
        hidden = self.encoder(image1 / 255.0)
        flows, infos = [], []
        for _ in range(iters + 1):
            output = self.flow_head(hidden)
            flows.append(output[:, :2])
            infos.append(output[:, 2:])
            hidden = hidden + 0.01
        return {"flow": flows, "info": infos, "final": flows[-1]}


def test_installing_recurrent_head_is_exactly_flow_preserving():
    torch.manual_seed(4)
    base = TinySeaRaft()
    image = torch.rand(2, 3, 5, 7) * 255
    expected = base(image, image, iters=2)["flow"]
    wrapped = UncertaintyAwareSeaRaftV2(
        base, variant="refinement_with_uncertainty", trainable_scope="all",
        refinement_channels=8,
    )
    observed = wrapped(image, image, iters=2, test_mode=True)
    assert len(observed["uncertainty_log_variance"]) == 3
    for before, after in zip(expected, observed["flow"]):
        assert torch.equal(before, after)


def test_uncertainty_feedback_is_detached_from_flow_objective():
    torch.manual_seed(5)
    head = UncertaintyAwareSeaRaftFlowHeadV2(
        nn.Conv2d(4, 6, 1), hidden_channels=4, refinement_channels=8,
        variant="refinement_with_uncertainty",
    )
    nn.init.normal_(head.refinement_head[-1].weight)
    hidden = torch.randn(2, 4, 5, 7, requires_grad=True)
    flow = head(hidden)[:, :2]
    flow.square().mean().backward()
    assert all(parameter.grad is None for parameter in head.uncertainty_head.parameters())
    assert any(parameter.grad is not None for parameter in head.refinement_head.parameters())
    assert hidden.grad is not None


def test_head_only_freezes_matcher_and_never_changes_flow():
    torch.manual_seed(6)
    base = TinySeaRaft()
    wrapped = UncertaintyAwareSeaRaftV2(
        base, variant="head_only", trainable_scope="uncertainty_only",
        refinement_channels=8,
    )
    assert all(not parameter.requires_grad for parameter in wrapped.recurrent_head.base_head.parameters())
    assert all(parameter.requires_grad for parameter in wrapped.recurrent_head.uncertainty_head.parameters())
    assert all(not parameter.requires_grad for parameter in wrapped.recurrent_head.refinement_head.parameters())


def test_affine_flow_transport_transforms_vectors_and_support():
    flow = torch.zeros(1, 2, 4, 5)
    flow[:, 0] = 1.0
    # Horizontal flip about x=2 maps (1,0) to (-1,0).
    matrix = torch.tensor([[[-1.0, 0.0, 4.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    transformed, valid = transform_flow_affine_v2(flow, matrix, (4, 5))
    assert torch.allclose(transformed[:, 0][valid[:, 0]], torch.full_like(transformed[:, 0][valid[:, 0]], -1.0))
    assert torch.allclose(transformed[:, 1][valid[:, 0]], torch.zeros_like(transformed[:, 1][valid[:, 0]]))
    # Native endpoint x+1 must remain in bounds, so one output column is invalid.
    assert int(valid.sum()) == 16


def test_decoupled_likelihood_cannot_train_flow_through_discrepancy():
    native = torch.zeros(1, 2, 3, 4, requires_grad=True)
    augmented = torch.full((1, 2, 3, 4), 0.4, requires_grad=True)
    alpha = torch.zeros(1, 1, 3, 4, requires_grad=True)
    truth = torch.zeros_like(native)
    teacher = torch.zeros_like(augmented, requires_grad=True)
    valid = torch.ones(1, 1, 3, 4, dtype=torch.bool)
    only_uncertainty = DecoupledFlowLossPolicyV2(
        task_weight=0.0, augmentation_weight=0.0, uncertainty_weight=1.0,
    )
    loss = decoupled_uncertainty_flow_loss_v2(
        [native], [augmented], [alpha], truth, valid, teacher, valid,
        policy=only_uncertainty,
    ).total
    loss.backward()
    assert alpha.grad is not None and float(alpha.grad.abs().sum()) > 0
    assert augmented.grad is None or float(augmented.grad.abs().sum()) == 0
    assert teacher.grad is None


def test_augmentation_loss_is_the_explicit_flow_gradient_path():
    native = torch.zeros(1, 2, 3, 4, requires_grad=True)
    augmented = torch.full((1, 2, 3, 4), 0.4, requires_grad=True)
    alpha = torch.zeros(1, 1, 3, 4, requires_grad=True)
    truth = torch.zeros_like(native)
    teacher = torch.zeros_like(augmented)
    valid = torch.ones(1, 1, 3, 4, dtype=torch.bool)
    only_ar = DecoupledFlowLossPolicyV2(
        task_weight=0.0, augmentation_weight=1.0, uncertainty_weight=0.0,
    )
    parts = decoupled_uncertainty_flow_loss_v2(
        [native], [augmented], [alpha], truth, valid, teacher, valid,
        policy=only_ar,
    )
    parts.total.backward()
    assert augmented.grad is not None and float(augmented.grad.abs().sum()) > 0
    assert alpha.grad is None or float(alpha.grad.abs().sum()) == 0
    assert float(parts.uncertainty.detach()) == pytest.approx(
        math.sqrt(2.0) * 0.8, rel=1e-6,
    )
