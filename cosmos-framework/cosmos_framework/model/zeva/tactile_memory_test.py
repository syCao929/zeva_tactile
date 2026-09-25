# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.zeva.tactile_memory import (
    TactileBehaviorHead,
    TactileBIT,
    TactileBITConfig,
)


def test_tactile_bit_keeps_a_causal_window_and_finger_tokens() -> None:
    bit = TactileBIT(TactileBITConfig(memory_steps=3))
    tokens = torch.randn(2, 5, 5, 256)
    output = bit(tokens)
    assert output.shape == (2, 3, 5, 256)


def test_tactile_bit_online_step_truncates_and_resets() -> None:
    torch.manual_seed(7)
    bit = TactileBIT(TactileBITConfig(memory_steps=2))
    frame = torch.randn(2, 5, 256)
    first = bit.step(frame)
    bit.step(torch.randn(2, 5, 256))
    bit.step(torch.randn(2, 5, 256))
    assert bit.online_length == 2
    bit.reset()
    assert bit.online_length == 0
    reset_first = bit.step(frame)
    assert torch.allclose(first, reset_first)


def test_tactile_bit_rejects_invalid_shapes() -> None:
    bit = TactileBIT()
    with pytest.raises(ValueError, match="Expected tactile tokens"):
        bit(torch.zeros(2, 5, 256))
    with pytest.raises(ValueError, match="Expected current tactile tokens"):
        bit.step(torch.zeros(2, 4, 256))


def test_tactile_bit_is_trainable() -> None:
    bit = TactileBIT()
    output = bit(torch.randn(2, 4, 5, 256))
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in bit.parameters())


def test_tactile_behavior_head_returns_zeva_spaces() -> None:
    head = TactileBehaviorHead()
    phase, effect, confidence = head(torch.randn(2, 5, 256))
    assert phase.shape == (2, 128)
    assert effect.shape == (2, 128)
    assert confidence.shape == (2, 1)


@pytest.mark.parametrize("module_type", [TactileBIT, TactileBehaviorHead])
def test_tactile_initialization_overwrites_materialized_storage(module_type) -> None:
    """Construction on meta cannot initialize storage later allocated by FSDP."""
    with torch.device("meta"):
        module = module_type()
    module.to_empty(device="cpu")
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(float("nan"))
    module.reset_parameters()
    assert all(torch.isfinite(parameter).all() for parameter in module.parameters())
    for child in module.modules():
        if isinstance(child, torch.nn.LayerNorm):
            assert torch.equal(child.weight, torch.ones_like(child.weight))
            assert torch.equal(child.bias, torch.zeros_like(child.bias))


def test_left_padding_preserves_current_tactile_memory() -> None:
    torch.manual_seed(7)
    bit = TactileBIT(TactileBITConfig(memory_steps=4))
    current = torch.randn(1, 1, 5, 256)
    padded = torch.cat((torch.randn(1, 3, 5, 256), current), dim=1)
    valid = torch.tensor([[False, False, False, True]])
    torch.testing.assert_close(bit(padded, valid)[:, -1], bit(current)[:, -1])


def test_tactile_branch_learns_before_first_visual_effect() -> None:
    from cosmos_framework.model.zeva.policy_injection import (
        PolicyInjectionConfig,
        PolicyInjectionPrior,
        gaussian_prior_nll,
    )
    from cosmos_framework.model.zeva.tactile_encoder import FrozenTactileEncoderWithProjector

    torch.manual_seed(7)
    encoder_projector = FrozenTactileEncoderWithProjector()
    bit = TactileBIT(TactileBITConfig(memory_steps=2))
    head = TactileBehaviorHead()
    prior = PolicyInjectionPrior(PolicyInjectionConfig(hidden_dim=32, horizon=4, action_dim=2))
    gate = torch.nn.Parameter(torch.zeros(1))
    force = torch.randn(1, 2, 5, 120, 3)
    tactile_valid = torch.tensor([[False, True]])
    global_context, phase = torch.randn(1, 256), torch.randn(1, 128)
    effects, effect_valid = torch.zeros(1, 4, 128), torch.zeros(1, 4, dtype=torch.bool)
    target = torch.randn(1, 4, 2)

    def forward():
        tokens = encoder_projector(force)
        _, effect, _ = head(bit(tokens, tactile_valid)[:, -1])
        return prior(global_context, phase, effects, effect_valid, current_effect_residual=torch.tanh(gate) * effect)

    # A zero gate preserves the baseline but must learn even before a visual
    # effect completes. Upstream tactile weights begin learning once it opens.
    baseline = prior(global_context, phase, effects, effect_valid)
    mean, std = forward()
    assert torch.equal(mean, baseline[0])
    assert torch.equal(std, baseline[1])
    gaussian_prior_nll(target, mean, std).backward()
    assert gate.grad is not None and torch.isfinite(gate.grad).all() and gate.grad.abs().sum() > 0
    with torch.no_grad():
        gate.fill_(0.1)
    for module in (encoder_projector, bit, head, prior):
        module.zero_grad(set_to_none=True)
    mean, std = forward()
    gaussian_prior_nll(target, mean, std).backward()
    for module in (encoder_projector.projector, bit, head.finger_projection, head.effect_head):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert all(p.grad is None for p in encoder_projector.encoder.parameters())
    for module in (head.phase_head, head.confidence_head):
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
