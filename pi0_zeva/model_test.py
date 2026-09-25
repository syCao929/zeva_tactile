from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pi0_zeva.model import build_policy, load_pretrained_backbone


class FakeBackbone(nn.Module):
    """Small differentiable implementation of the official suffix/flow API."""

    def __init__(self, horizon=3):
        super().__init__()
        self.config = SimpleNamespace(
            action_horizon=horizon, action_dim=32, pi05=False, pytorch_compile_mode=None
        )
        self.action_in_proj = nn.Linear(32, 12)
        self.state_proj = nn.Linear(32, 12)
        self.action_out_proj = nn.Linear(12, 32)
        self.preprocess_modes = []
        self.suffix_calls = []

    def _preprocess_observation(self, observation, *, train=True):
        self.preprocess_modes.append(train)
        return observation

    def embed_suffix(self, state, noisy_actions, timestep):
        state_token = self.state_proj(state)[:, None]
        actions = self.action_in_proj(noisy_actions) + timestep[:, None, None]
        embeddings = torch.cat((state_token, actions), dim=1)
        padding = torch.ones(
            embeddings.shape[:2], device=state.device, dtype=torch.bool
        )
        attention = torch.zeros_like(padding)
        attention[:, :2] = True
        return embeddings, padding, attention, None

    def velocity(self, observation, actions, time):
        embeddings, padding, attention, _ = self.embed_suffix(
            observation.state, actions, time
        )
        self.suffix_calls.append(
            (
                embeddings.detach().clone(),
                padding.detach().clone(),
                attention.detach().clone(),
            )
        )
        return self.action_out_proj(torch.tanh(embeddings[:, 1:] + embeddings[:, :1]))

    def forward(self, observation, actions, noise=None, time=None):
        observation = self._preprocess_observation(observation, train=True)
        noise = torch.zeros_like(actions) if noise is None else noise
        time = (
            torch.full((actions.shape[0],), 0.5, device=actions.device)
            if time is None
            else time
        )
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        return (self.velocity(observation, x_t, time) - (noise - actions)).square()

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        observation = self._preprocess_observation(observation, train=False)
        shape = (
            observation.state.shape[0],
            self.config.action_horizon,
            self.config.action_dim,
        )
        actions = torch.zeros(shape, device=device) if noise is None else noise.clone()
        for step in range(num_steps):
            time = torch.full((shape[0],), 1 - step / num_steps, device=device)
            actions = actions - self.velocity(observation, actions, time) / num_steps
        return actions


@pytest.fixture
def inputs():
    torch.manual_seed(3)
    return (
        SimpleNamespace(state=torch.randn(2, 32)),
        torch.randn(2, 3, 32),
        {
            "global": torch.randn(2, 256),
            "phase": torch.randn(2, 128),
            "effect": torch.randn(2, 4, 128),
            "effect_valid": torch.tensor([[False] * 4, [False, False, True, True]]),
        },
    )


def policy(mode="zeva", **kwargs):
    return build_policy(
        mode, backbone=FakeBackbone(), horizon=3, require_pretrained=False, **kwargs
    )


def test_zero_adapter_preserves_baseline_loss_sampling_state_and_masks(inputs):
    observation, actions, behavior = inputs
    original = FakeBackbone()
    baseline = build_policy(
        backbone=copy.deepcopy(original), horizon=3, require_pretrained=False
    ).eval()
    zeva = build_policy(
        "zeva", backbone=original, horizon=3, require_pretrained=False
    ).eval()
    noise, time = torch.randn_like(actions), torch.tensor([0.2, 0.7])
    expected = baseline(observation, actions, noise=noise, time=time)
    actual = zeva(observation, actions, behavior, noise=noise, time=time)
    assert torch.equal(actual["action_flow_loss"], expected["action_flow_loss"])
    assert torch.equal(
        actual["loss"], actual["action_flow_loss"] + 0.01 * actual["prior_nll"]
    )
    for actual_tensor, expected_tensor in zip(
        zeva.backbone.suffix_calls[-1], baseline.backbone.suffix_calls[-1]
    ):
        assert torch.equal(actual_tensor, expected_tensor)
    assert torch.equal(
        zeva.sample_actions(observation, behavior, noise=noise, num_steps=3),
        baseline.sample_actions(observation, noise=noise, num_steps=3),
    )


def test_training_and_sampling_inject_identical_residual_and_ignore_padded_loss(inputs):
    observation, actions, behavior = inputs
    model = policy().eval()
    nn.init.normal_(model.prior_adapter.weight, std=0.1)
    noise = torch.randn_like(actions)
    metrics = model(observation, actions, behavior, noise=noise, time=torch.ones(2))
    train_suffix = model.backbone.suffix_calls[-1][0]
    model.backbone.suffix_calls.clear()
    model.sample_actions(observation, behavior, noise=noise, num_steps=3)
    assert torch.equal(train_suffix, model.backbone.suffix_calls[0][0])
    residual, _, _ = model._memory(behavior, None, None)
    with model._condition(residual):
        channel_loss = model.backbone(
            observation, actions, noise=noise, time=torch.ones(2)
        )
    assert torch.equal(metrics["action_flow_loss"], channel_loss[..., :18].mean())
    assert not torch.equal(metrics["action_flow_loss"], channel_loss.mean())


def test_memory_changes_samples_and_stage2_preserves_frozen_backbone_gradients(inputs):
    observation, actions, behavior = inputs
    model = policy().train()
    assert not model.backbone.training
    nn.init.normal_(model.prior_adapter.weight, std=0.1)
    perturbed = {**behavior, "phase": -behavior["phase"], "effect": -behavior["effect"]}
    noise = torch.randn_like(actions)
    a = model.sample_actions(observation, behavior, noise=noise, num_steps=2)
    b = model.sample_actions(observation, perturbed, noise=noise, num_steps=2)
    assert not torch.allclose(a, b)
    metrics = model(
        observation, actions, behavior, noise=noise, time=torch.full((2,), 0.5)
    )
    metrics["loss"].backward()
    assert all(
        not p.requires_grad and p.grad is None for p in model.backbone.parameters()
    )
    assert model.prior_adapter.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.prior.parameters()
    )
    assert not any(model.backbone.preprocess_modes)
    assert model._suffix_context.get() is None


def test_eval_disables_official_forced_training_augmentation(inputs):
    observation, actions, _ = inputs
    model = policy("baseline")
    model.train()
    model(observation, actions)
    model.eval()
    model(observation, actions)
    assert model.backbone.preprocess_modes == [True, False]


def test_unloaded_weights_fail_and_tied_safetensors_load_strictly(tmp_path, inputs):
    from safetensors.torch import save_model

    observation, actions, _ = inputs
    source = FakeBackbone()
    source.tied_action_projection = source.action_in_proj
    target = FakeBackbone()
    target.tied_action_projection = target.action_in_proj
    model = build_policy(backbone=target, horizon=3)
    with pytest.raises(RuntimeError, match="Load pretrained"):
        model(observation, actions)
    save_model(source, tmp_path / "model.safetensors")
    load_pretrained_backbone(model, tmp_path)
    assert model.pretrained_loaded
    assert torch.equal(
        model.backbone.action_in_proj.weight, source.action_in_proj.weight
    )
    assert torch.isfinite(model(observation, actions)["loss"])
    broken = nn.Linear(2, 2)
    save_model(broken, tmp_path / "broken.safetensors")
    with pytest.raises(RuntimeError):
        load_pretrained_backbone(model, tmp_path / "broken.safetensors")
    assert not model.pretrained_loaded


@pytest.fixture
def tactile_checkpoint(tmp_path):
    from cosmos_framework.model.zeva.tactile_encoder import (
        FrozenPatchInformedTactileEncoder,
    )

    path = tmp_path / "encoder.pt"
    torch.save(FrozenPatchInformedTactileEncoder().encoder.state_dict(), path)
    return path


def test_paired_stage2_seed_preserves_shared_initial_weights_and_outputs(
    inputs, tactile_checkpoint
):
    """Extra tactile modules must not shift the shared prior's initialization."""
    torch.manual_seed(42)
    visual = policy("zeva").eval()
    torch.manual_seed(42)
    tactile = policy("zeva_tactile", tactile_checkpoint=tactile_checkpoint).eval()
    visual_state, tactile_state = visual.state_dict(), tactile.state_dict()
    for name, value in visual_state.items():
        assert torch.equal(value, tactile_state[name]), name

    observation, actions, behavior = inputs
    noise, time = torch.randn_like(actions), torch.tensor([0.2, 0.7])
    force = torch.randn(2, 3, 1972)
    valid = torch.ones(2, 3, dtype=torch.bool)
    expected = visual(observation, actions, behavior, noise=noise, time=time)
    actual = tactile(
        observation, actions, behavior, force, valid, noise=noise, time=time
    )
    for name in ("loss", "action_flow_loss", "prior_nll"):
        assert torch.equal(actual[name], expected[name]), name
    assert torch.equal(
        visual.sample_actions(observation, behavior, noise=noise, num_steps=2),
        tactile.sample_actions(
            observation, behavior, force, valid, noise=noise, num_steps=2
        ),
    )


def test_tactile_startup_gradients_padding_freezing_and_causality(
    inputs, tactile_checkpoint
):
    observation, actions, behavior = inputs
    model = policy("zeva_tactile", tactile_checkpoint=tactile_checkpoint)
    behavior = {**behavior, "effect_valid": torch.zeros(2, 4, dtype=torch.bool)}
    state = torch.randn(2, 30, 1972)
    valid = torch.zeros(2, 30, dtype=torch.bool)
    valid[:, -1] = True
    metrics = model(observation, actions, behavior, state, valid)
    metrics["loss"].backward()
    assert (
        model.tactile_effect_gate.grad is not None
        and model.tactile_effect_gate.grad.abs().sum() > 0
    )
    with torch.no_grad():
        model.tactile_effect_gate.fill_(0.2)
        nn.init.normal_(model.prior_adapter.weight, std=0.1)
    reference = model._tactile_effect(state, valid)
    changed_padding = state.clone()
    changed_padding[:, :-1] = float("nan")
    torch.testing.assert_close(
        reference, model._tactile_effect(changed_padding, valid), rtol=0, atol=0
    )
    torch.testing.assert_close(
        reference,
        model._tactile_effect(state[:, -1:], valid[:, -1:]),
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.equal(
        model._tactile_effect(state, torch.zeros_like(valid)), torch.zeros(2, 128)
    )
    model.zero_grad(set_to_none=True)
    model(observation, actions, behavior, state, valid)["loss"].backward()
    for module in (
        model.tactile_encoder_projector.projector,
        model.tactile_bit,
        model.tactile_behavior_head.effect_head,
    ):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert all(
        p.grad is None and not p.requires_grad
        for p in model.tactile_encoder_projector.encoder.parameters()
    )
    for module in (
        model.tactile_behavior_head.phase_head,
        model.tactile_behavior_head.confidence_head,
    ):
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    future = torch.randn(2, 4, 5, 256)
    history = torch.randn(2, 2, 5, 256)
    torch.testing.assert_close(
        model.tactile_bit(torch.cat((history, future), 1))[:, :2],
        model.tactile_bit(history),
    )


def test_invalid_tactile_inputs_are_rejected(inputs, tactile_checkpoint):
    observation, actions, behavior = inputs
    model = policy("zeva_tactile", tactile_checkpoint=tactile_checkpoint)
    with pytest.raises(ValueError, match="requires tactile_state"):
        model(observation, actions, behavior)
    with pytest.raises(ValueError, match="left padding"):
        model._tactile_effect(
            torch.randn(2, 3, 1972),
            torch.tensor([[True, False, True], [True, True, True]]),
        )
    with pytest.raises(ValueError, match="finite"):
        model._tactile_effect(
            torch.full((2, 1, 1972), float("nan")), torch.ones(2, 1, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="1 <= T <= 30"):
        model._tactile_effect(
            torch.randn(2, 31, 1972), torch.ones(2, 31, dtype=torch.bool)
        )


def test_suffix_context_is_cleared_on_backbone_failure(inputs, monkeypatch):
    observation, actions, behavior = inputs
    model = policy()

    def fail(*args, **kwargs):
        raise RuntimeError("intentional")

    monkeypatch.setattr(model.backbone, "forward", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        model(observation, actions, behavior)
    assert model._suffix_context.get() is None


@pytest.mark.skipif(
    os.environ.get("PI0_ZEVA_TEST_OPENPI") != "1",
    reason="Run explicitly in the prepared OpenPI environment",
)
def test_official_openpi_small_cpu_bridge(monkeypatch):
    """Exercise real OpenPI flow and KV sampling with small transformer configs."""
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch import gemma_pytorch, pi0_pytorch

    original_paligemma = gemma_pytorch.PaliGemmaForConditionalGeneration
    original_gemma = gemma_pytorch.GemmaForCausalLM

    def small_paligemma(config):
        config.text_config.vocab_size = 64
        config.text_config.num_hidden_layers = 1
        config.hidden_size = config.text_config.hidden_size
        config.projection_dim = config.text_config.hidden_size
        config.vision_config.hidden_size = 32
        config.vision_config.projection_dim = config.text_config.hidden_size
        config.vision_config.intermediate_size = 64
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 4
        config.vision_config.image_size = 28
        config.vision_config.patch_size = 14
        return original_paligemma(config=config)

    def small_gemma(config):
        config.vocab_size = 64
        config.num_hidden_layers = 1
        return original_gemma(config=config)

    monkeypatch.setattr(
        gemma_pytorch, "PaliGemmaForConditionalGeneration", small_paligemma
    )
    monkeypatch.setattr(gemma_pytorch, "GemmaForCausalLM", small_gemma)
    # Keep preprocessing out of this small-model integration test: the official
    # processor always resizes to 224, while this fixture has a 28px vision stem.
    monkeypatch.setattr(
        pi0_pytorch._preprocessing,
        "preprocess_observation_pytorch",
        lambda obs, **kwargs: obs,
    )
    cfg = Pi0Config(
        action_horizon=3,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        pytorch_compile_mode=None,
    )
    backbone = pi0_pytorch.PI0Pytorch(cfg)
    baseline = build_policy(
        backbone=copy.deepcopy(backbone), horizon=3, require_pretrained=False
    ).eval()
    zeva = build_policy(
        "zeva", backbone=backbone, horizon=3, require_pretrained=False
    ).eval()
    observation = SimpleNamespace(
        state=torch.randn(1, 32),
        images={"base_0_rgb": torch.randn(1, 3, 28, 28)},
        image_masks={"base_0_rgb": torch.ones(1, dtype=torch.bool)},
        tokenized_prompt=torch.tensor([[1, 2, 3]]),
        tokenized_prompt_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    actions, noise = torch.randn(1, 3, 32), torch.randn(1, 3, 32)
    behavior = {
        "global": torch.randn(1, 256),
        "phase": torch.randn(1, 128),
        "effect": torch.randn(1, 4, 128),
        "effect_valid": torch.zeros(1, 4, dtype=torch.bool),
    }
    expected = baseline(observation, actions, noise=noise, time=torch.ones(1))
    actual = zeva(observation, actions, behavior, noise=noise, time=torch.ones(1))
    assert torch.equal(expected["action_flow_loss"], actual["action_flow_loss"])
    assert torch.equal(
        baseline.sample_actions(observation, noise=noise, num_steps=2),
        zeva.sample_actions(observation, behavior, noise=noise, num_steps=2),
    )
    actual["loss"].backward()
    assert zeva.prior_adapter.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in zeva.backbone.parameters())
