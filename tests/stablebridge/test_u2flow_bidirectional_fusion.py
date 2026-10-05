from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from stablebridge.physical_repair.u2flow_bidirectional_fusion import (
    U2FlowBidirectionalFusionPolicyV1,
    predict_u2flow_triplet_fusion_v1,
    u2flow_bidirectional_fusion_v1,
)


class CountingFlowProvider(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.last_flow = None

    def forward(self, first, second, *, iters, test_mode):
        del test_mode
        self.calls += 1
        flow = (second[:, :2] - first[:, :2]) / 255.0
        self.last_flow = flow
        alpha = torch.zeros_like(flow[:, :1])
        return {
            "flow": [flow for _ in range(iters)],
            "uncertainty_log_variance": [alpha for _ in range(iters)],
        }


def _policy(*, enabled: bool) -> U2FlowBidirectionalFusionPolicyV1:
    return U2FlowBidirectionalFusionPolicyV1(
        enabled=enabled,
        optimization_steps=2,
        uncertainty_variance_threshold=2.0,
        learning_rate=0.001,
        learning_rate_decay=0.8,
        variance_minimum=1e-3,
        variance_maximum=200.0,
        random_seed=17,
    )


def test_disabled_triplet_path_runs_only_forward_provider_and_is_exact():
    model = CountingFlowProvider()
    triplet = torch.rand(2, 3, 3, 4, 5)
    result = predict_u2flow_triplet_fusion_v1(
        model, triplet, matcher_iterations=2, policy=_policy(enabled=False),
    )
    assert model.calls == 1
    assert result.fusion_applied is False
    assert torch.equal(result.fused_flow, model.last_flow)
    assert not result.replacement_mask.any()


def test_enabled_triplet_path_runs_forward_and_backward_provider():
    model = CountingFlowProvider()
    triplet = torch.rand(1, 3, 3, 4, 5)
    result = predict_u2flow_triplet_fusion_v1(
        model, triplet, matcher_iterations=2, policy=_policy(enabled=True),
    )
    assert model.calls == 2
    assert result.fusion_applied is True
    assert result.forward_reliable is not None
    assert result.backward_reliable is not None
    assert torch.isfinite(result.fused_flow).all()


def test_fusion_replaces_only_forward_unreliable_backward_reliable_pixels():
    forward = torch.zeros(1, 2, 4, 5)
    backward = torch.ones(1, 2, 4, 5)
    forward_alpha = torch.zeros(1, 1, 4, 5)
    backward_alpha = torch.zeros(1, 1, 4, 5)
    forward_alpha[..., 1, 2] = math.log(10.0)
    result = u2flow_bidirectional_fusion_v1(
        forward,
        backward,
        forward_alpha,
        backward_alpha,
        policy=_policy(enabled=True),
    )
    expected_replacement = torch.zeros(1, 1, 4, 5, dtype=torch.bool)
    expected_replacement[..., 1, 2] = True
    assert torch.equal(result.replacement_mask, expected_replacement)
    preserved = ~expected_replacement.expand_as(forward)
    assert torch.equal(result.fused_flow[preserved], forward[preserved])
    assert torch.isfinite(result.fused_flow).all()


def test_no_joint_reliable_support_fails_closed_to_forward_flow():
    forward = torch.randn(1, 2, 3, 4)
    backward = torch.randn(1, 2, 3, 4)
    unreliable = torch.full((1, 1, 3, 4), math.log(100.0))
    reliable = torch.zeros(1, 1, 3, 4)
    result = u2flow_bidirectional_fusion_v1(
        forward,
        backward,
        unreliable,
        reliable,
        policy=_policy(enabled=True),
    )
    assert result.fusion_applied is True
    assert not result.replacement_mask.any()
    assert torch.equal(result.fused_flow, forward)


def test_tiny_motion_fit_does_not_advance_global_rng():
    forward = torch.zeros(1, 2, 3, 4)
    backward = torch.ones(1, 2, 3, 4)
    reliable = torch.zeros(1, 1, 3, 4)
    torch.manual_seed(1234)
    state = torch.random.get_rng_state().clone()
    u2flow_bidirectional_fusion_v1(
        forward,
        backward,
        reliable,
        reliable,
        policy=_policy(enabled=True),
    )
    assert torch.equal(torch.random.get_rng_state(), state)
