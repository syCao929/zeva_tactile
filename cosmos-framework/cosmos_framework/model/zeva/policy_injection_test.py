import torch

from cosmos_framework.model.zeva.brief_interaction_trace import BriefInteractionTrace
from cosmos_framework.model.zeva.policy_injection import (
    CausalPromptPolicyAdapter,
    PolicyInjectionConfig,
    PolicyInjectionPrior,
    gaussian_prior_nll,
)


def test_policy_injection_shapes_nll_and_zero_projection() -> None:
    prior = PolicyInjectionPrior(PolicyInjectionConfig(horizon=32, action_dim=8))
    effect = torch.randn(3, 4, 128)
    valid = torch.tensor([[False, False, False, False], [False, True, True, True], [True, True, True, True]])
    mean, std = prior(torch.randn(3, 256), torch.randn(3, 128), effect, valid)
    assert mean.shape == std.shape == (3, 32, 8)
    assert torch.isfinite(gaussian_prior_nll(torch.randn_like(mean), mean, std))
    adapter = CausalPromptPolicyAdapter(action_dim=8, hidden_dim=16)
    adapter.init_weights()
    assert not hasattr(adapter, "residual_gate")
    assert torch.equal(adapter(mean, action_length=33), torch.zeros(3, 33, 16))


def test_zero_projection_has_direct_action_gradient() -> None:
    """The faithful zero projection must grow without a scalar-gate warmup."""
    adapter = CausalPromptPolicyAdapter(action_dim=8, hidden_dim=16)
    adapter.init_weights()
    prior = torch.randn(2, 32, 8, requires_grad=True)
    target = torch.randn(2, 33, 16)
    loss = (adapter(prior, action_length=33) - target).square().mean()
    loss.backward()
    weight_grad = adapter.prior_to_action_embedding.weight.grad
    assert weight_grad is not None
    assert torch.count_nonzero(weight_grad) > 0


def test_brief_interaction_trace_tensor_path_is_exact() -> None:
    torch.manual_seed(7)
    prior = PolicyInjectionPrior(PolicyInjectionConfig(horizon=4, action_dim=2))
    task_context = torch.randn(2, 256)
    phase = torch.randn(2, 128)
    effects = torch.randn(2, 4, 128)
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    tensor_path = prior(task_context, phase, effects, valid)
    paper_named = prior(
        task_context,
        phase,
        BriefInteractionTrace(effects=effects, valid=valid),
    )
    assert torch.equal(tensor_path[0], paper_named[0])
    assert torch.equal(tensor_path[1], paper_named[1])


def test_zero_current_effect_residual_preserves_baseline_with_any_visual_mask() -> None:
    torch.manual_seed(7)
    prior = PolicyInjectionPrior(PolicyInjectionConfig(horizon=4, action_dim=2))
    context, phase = torch.randn(3, 256), torch.randn(3, 128)
    effects = torch.randn(3, 4, 128)
    valid = torch.tensor([[False, False, False, False], [False, False, True, True], [True, True, True, True]])
    baseline = prior(context, phase, effects, valid)
    actual = prior(context, phase, effects, valid, current_effect_residual=torch.zeros(3, 128))
    assert torch.equal(actual[0], baseline[0])
    assert torch.equal(actual[1], baseline[1])


def test_current_effect_residual_is_added_after_bos_substitution() -> None:
    torch.manual_seed(7)
    prior = PolicyInjectionPrior(PolicyInjectionConfig(horizon=4, action_dim=2))
    context, phase = torch.randn(2, 256), torch.randn(2, 128)
    effects = torch.randn(2, 4, 128)
    valid = torch.tensor([[False, False, False, False], [False, False, True, True]])
    residual = torch.randn(2, 128)
    actual = prior(context, phase, effects, valid, current_effect_residual=residual)
    # The equivalent baseline input explicitly contains BOS for missing visual
    # history, plus current evidence in its last slot.
    explicit_effects = torch.where(valid.unsqueeze(-1), effects, prior.bos_effect).clone()
    explicit_effects[:, -1] += residual
    expected = prior(context, phase, explicit_effects, torch.ones_like(valid))
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
