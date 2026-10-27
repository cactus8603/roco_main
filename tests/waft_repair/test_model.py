"""Portable CPU checks for feature repair's public data and gradient contract."""
from unittest.mock import patch

import pytest
import torch

from waft_repair.model import FeatureRepair, features


@pytest.fixture(autouse=True)
def small_cpu_execution():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def example():
    generator = torch.Generator().manual_seed(17)
    original = torch.randn(1, 2, 64, 8, 12, generator=generator)
    original[..., 0, 0] = -0.0
    sigma = torch.linspace(.5, 1.5, 64)
    mask = torch.zeros(1, 1, 1, 8, 12, dtype=torch.bool)
    mask[..., 1:7, 1:10] = True
    return original, sigma, mask


def test_constructor_preserves_rng_and_uses_cpu_float32():
    before = torch.get_rng_state().clone()
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("unexpected CUDA")):
        model = FeatureRepair()
    assert torch.equal(torch.get_rng_state(), before)
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == 3_923_200
    assert len(model.state_dict()) == 90
    assert model.configuration()["local_parameter_count"] == 38_848


def test_initial_output_preserves_signed_zero_and_has_live_readout_gradients():
    original, sigma, mask = example()
    model = FeatureRepair()
    prediction = features(model, original, sigma, mask)
    assert torch.equal(prediction.detach().contiguous().view(torch.int32), original.view(torch.int32))
    (prediction - original - .1).square().mean().backward()
    assert model.readout.weight.grad.abs().sum() > 0
    assert model.local_readout.weight.grad.abs().sum() > 0
    assert original.grad is None


def test_raw_residual_covers_both_frames_and_padding_without_sigma_output_scaling():
    original, sigma, mask = example()
    model = FeatureRepair()
    with torch.no_grad():
        model.readout.bias.fill_(2)
        model.local_readout.bias.fill_(1)
        prediction = model(original, sigma, mask)
    # A large correction also checks that the old sigma box/tanh is absent.
    expected = original + 3
    assert torch.equal(prediction, expected)
    expanded_mask = mask.expand_as(original)
    assert torch.equal(prediction[~expanded_mask], expected[~expanded_mask])


def test_all_branches_receive_gradients_after_readouts_are_active():
    original, sigma, mask = example()
    model = FeatureRepair()
    generator = torch.Generator().manual_seed(99)
    with torch.no_grad():
        model.readout.weight.copy_(torch.randn(model.readout.weight.shape, generator=generator) * .001)
        model.local_readout.weight.copy_(torch.randn(model.local_readout.weight.shape, generator=generator) * .001)
    output = model(original, sigma, mask)
    (output - original - .01).square().mean().backward()
    branches = ("input_projection", "body.convs", "body.mambas", "local_body", "local_readout")
    for prefix in branches:
        gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_target_is_not_part_of_forward_and_live_original_is_rejected():
    original, sigma, mask = example()
    model = FeatureRepair()
    with pytest.raises(ValueError, match="detached"):
        model(original.requires_grad_(), sigma, mask)
    with pytest.raises(TypeError):
        model(original.detach(), sigma, mask, target=original.detach())


def test_bad_normalization_and_empty_native_support_fail_early():
    original, sigma, mask = example()
    model = FeatureRepair()
    with pytest.raises(ValueError, match="positive finite"):
        model(original, sigma * 0, mask)
    with pytest.raises(ValueError, match="nonempty per example"):
        model(original, sigma, torch.zeros_like(mask))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast disabled"):
            model(original, sigma, mask)


def test_reference_guard_rejects_large_grid_before_context_execution():
    model = FeatureRepair(max_reference_tokens=4)
    original, sigma, mask = example()
    with patch.object(model.input_projection, "forward", side_effect=AssertionError("projection ran")):
        with pytest.raises(RuntimeError, match="native"):
            model(original, sigma, mask)
