"""CPU-only audit reproductions; does not modify project files or checkpoints.

Run from zeva-work:
  CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' PYTHONPATH=cosmos-framework \
    envs/zeva/bin/python audit-artifacts/20260925/zeva_tactile_fix_cpu_20260925.py

The initialization probe poisons storage after to_empty, then invokes the current
network's actual init_weights AST. Remaining poison proves a missing initializer,
not that an existing training checkpoint necessarily contains NaN.
The gradient probe uses normal CPU construction and deliberately nonzero gate to
isolate masking behavior; it is not an FSDP or production-checkpoint validation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch import nn

from cosmos_framework.model.zeva.policy_injection import (
    CausalPromptPolicyAdapter,
    PolicyInjectionConfig,
    PolicyInjectionPrior,
)
from cosmos_framework.model.zeva.tactile_encoder import (
    FrozenTactileEncoderWithProjector,
    TactileEncoderAdapter,
)
from cosmos_framework.model.zeva.tactile_memory import (
    TactileBIT,
    TactileBITConfig,
    TactileBehaviorHead,
)


REPO = Path.cwd()
MODEL = REPO / "cosmos-framework/cosmos_framework/model"
NETWORK = MODEL / "generator/mot/cosmos3_vfm_network.py"
OMNI = MODEL / "generator/omni_mot_model.py"


def source_method(path: Path, class_name: str, method_name: str):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[]))
    ns = {"torch": torch, "nn": nn, "math": math}
    exec(compile(module, str(path), "exec"), ns)
    return ns[method_name]


def all_finite(module: nn.Module) -> bool:
    return all(bool(p.isfinite().all()) for p in module.parameters())


def init_probe() -> dict:
    with torch.device("meta"):
        parts = nn.ModuleDict({
            "ep": FrozenTactileEncoderWithProjector(),
            "bit": TactileBIT(),
            "head": TactileBehaviorHead(),
            "global": nn.Linear(256, 256),
            "prior": PolicyInjectionPrior(PolicyInjectionConfig(action_dim=18)),
            "adapter": CausalPromptPolicyAdapter(action_dim=18, hidden_dim=256),
        })
    parts.to_empty(device="cpu")
    with torch.no_grad():
        for p in parts.parameters():
            p.fill_(float("nan"))
    obj = SimpleNamespace(
        config=SimpleNamespace(action_gen=False, vision_gen=False, sound_gen=False),
        behavior_pbd=parts["prior"], behavior_adapter=parts["adapter"],
        behavior_global_projector=parts["global"], behavior_pim_encoder=None,
        behavior_pim_projector=None, behavior_pim_gate=None, behavior_online_projector=None,
        tactile_bit=parts["bit"], tactile_behavior_head=parts["head"],
        tactile_encoder_projector=parts["ep"], tactile_encoder_checkpoint=None,
        tactile_phase_gate=nn.Parameter(torch.zeros(1)),
        tactile_effect_gate=nn.Parameter(torch.zeros(1)), proprio_projector=None,
        language_model=SimpleNamespace(init_weights=lambda **kwargs: None),
    )
    source_method(NETWORK, "Cosmos3VFMNetwork", "init_weights")(obj, buffer_device=torch.device("cpu"))
    missing = [n for n, p in parts["head"].named_parameters() if not bool(p.isfinite().all())]
    expected = [f"{head}.0.{kind}" for head in ("phase_head", "effect_head", "confidence_head") for kind in ("weight", "bias")]
    assert missing == [], missing
    assert all_finite(parts["bit"])
    assert all_finite(parts["prior"])
    assert all_finite(parts["adapter"])
    assert all_finite(parts["global"])
    norms = {
        n: bool(torch.equal(m.weight, torch.ones_like(m.weight)) and torch.equal(m.bias, torch.zeros_like(m.bias)))
        for n, m in parts["prior"].named_modules() if isinstance(m, nn.LayerNorm)
    }
    assert all(norms.values())
    return {
        "tactile_head_parameters_uninitialized": missing,
        "tactile_head_output_finite": [bool(t.isfinite().all()) for t in parts["head"](torch.randn(1, 5, 256))],
        "tactile_bit_all_parameters_finite": all_finite(parts["bit"]),
        "baseline_prior_all_parameters_finite": all_finite(parts["prior"]),
        "baseline_layernorms_exact_one_zero": norms,
        "baseline_attention_all_parameters_finite": {
            "effect_attention": all_finite(parts["prior"].effect_attention),
            "progress_attention": all_finite(parts["prior"].progress_attention),
        },
        "baseline_adapter_all_parameters_finite": all_finite(parts["adapter"]),
        "baseline_global_projector_all_parameters_finite": all_finite(parts["global"]),
    }


def grad_l1(module: nn.Module) -> float:
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def grad_probe() -> list[dict]:
    torch.manual_seed(25)
    net = nn.Module()
    net.tactile_encoder_adapter = TactileEncoderAdapter()
    net.tactile_encoder_projector = FrozenTactileEncoderWithProjector()
    state = torch.load(REPO / "models/zeva/tactile_patch_encoder_19999.pt", map_location="cpu", weights_only=True)
    net.tactile_encoder_projector.encoder.encoder.load_state_dict(state, strict=True)
    net.tactile_bit = TactileBIT(TactileBITConfig(memory_steps=2))
    net.tactile_behavior_head = TactileBehaviorHead()
    net.tactile_effect_gate = nn.Parameter(torch.tensor([0.3]))
    net.tactile_phase_gate = nn.Parameter(torch.tensor([0.]), requires_grad=False)
    net.encode_tactile_behavior = MethodType(source_method(NETWORK, "Cosmos3VFMNetwork", "encode_tactile_behavior"), net)
    attach = source_method(OMNI, "OmniMoTModel", "_attach_stage2_behavior")
    cfg = SimpleNamespace(enabled=True, leading_condition_steps=0, horizon=32, action_dim=18, pim_memory_enabled=False, online_memory_enabled=False)
    obj = SimpleNamespace(config=SimpleNamespace(behavior_stage2=cfg), net=net, precision=torch.float32)
    prior = PolicyInjectionPrior(PolicyInjectionConfig(action_dim=18, horizon=32))
    prior.init_weights()
    class ActionProjection(nn.Linear):
        def forward(self, values, domain):
            return super().forward(values)
    net.behavior_pbd = prior
    net.behavior_adapter = CausalPromptPolicyAdapter(action_dim=18, hidden_dim=32)
    net.behavior_adapter.init_weights()
    net.behavior_prior_leading_condition_steps = 0
    net.behavior_prior_dropout_rate = 0.
    net.behavior_prior_inference_guidance_scale = 0.5
    net.action2llm = ActionProjection(18, 32)
    net.action_modality_embed = nn.Parameter(torch.zeros(32))
    net.pack_action = lambda tokens, shapes, domain: (torch.cat(tokens), None)
    encode_action = source_method(NETWORK, "Cosmos3VFMNetwork", "_encode_action")
    encode_action.__globals__["has_noisy_tokens"] = lambda action: False
    batch = {
        "behavior_global": torch.randn(1, 256), "behavior_phase": torch.randn(1, 128),
        "behavior_effect": torch.randn(1, 4, 128), "behavior_effect_valid": torch.zeros(1, 4, dtype=torch.bool),
        "tactile_state": torch.randn(1, 2, 1972), "tactile_valid": torch.ones(1, 2, dtype=torch.bool),
    }
    clean = SimpleNamespace(x0_tokens_action=[torch.randn(32, 18)])
    outputs = []
    for is_valid, gate_value in ((False, 0.), (False, 0.3), (True, 0.), (True, 0.3)):
        net.zero_grad()
        prior.zero_grad()
        with torch.no_grad():
            net.tactile_effect_gate.fill_(gate_value)
        batch["behavior_effect_valid"][0, -1] = is_valid
        packed = SimpleNamespace(text_ids=torch.zeros(1, dtype=torch.long))
        attach(obj, packed, batch, clean, [0])
        baseline = prior(packed.behavior_global, packed.behavior_phase, packed.behavior_effect, packed.behavior_effect_valid)
        assert torch.equal(packed.behavior_effect, batch["behavior_effect"])
        assert torch.equal(packed.behavior_effect_valid, batch["behavior_effect_valid"])
        packed.action = SimpleNamespace(tokens=[torch.randn(32, 18)], token_shapes=[(32,)], domain_id=[], sequence_indexes=torch.arange(32), timesteps=torch.zeros(32), mse_loss_indexes=torch.arange(32))
        encode_action(net, packed, torch.zeros(32, 32), torch.float32)
        mean, std = packed.behavior_prior_mean, packed.behavior_prior_std
        baseline_exact = torch.equal(mean, baseline[0]) and torch.equal(std, baseline[1])
        assert baseline_exact if gate_value == 0 else not baseline_exact
        (mean.square().mean() + std.mean()).backward()
        result = {
            "last_effect_slot_valid": is_valid,
            "gate_value": gate_value,
            "baseline_exact": baseline_exact,
            "gate_grad": net.tactile_effect_gate.grad.tolist(),
            "projector_grad_l1": grad_l1(net.tactile_encoder_projector.projector),
            "bit_grad_l1": grad_l1(net.tactile_bit),
            "effect_head_grad_l1": grad_l1(net.tactile_behavior_head.effect_head),
            "phase_head_grad_tensors": sum(p.grad is not None for p in net.tactile_behavior_head.phase_head.parameters()),
            "confidence_head_grad_tensors": sum(p.grad is not None for p in net.tactile_behavior_head.confidence_head.parameters()),
            "encoder_grad_tensors": sum(p.grad is not None for p in net.tactile_encoder_projector.encoder.parameters()),
            "phase_gate_has_grad": net.tactile_phase_gate.grad is not None,
        }
        vals = [abs(result["gate_grad"][0]), result["projector_grad_l1"], result["bit_grad_l1"], result["effect_head_grad_l1"]]
        assert vals[0] > 0
        assert all(v > 0 for v in vals[1:]) if gate_value else all(v == 0 for v in vals[1:])
        assert result["phase_head_grad_tensors"] == result["confidence_head_grad_tensors"] == result["encoder_grad_tensors"] == 0
        outputs.append(result)
    all_missing = net.encode_tactile_behavior(batch["tactile_state"], torch.zeros_like(batch["tactile_valid"]))
    assert all(torch.equal(x, torch.zeros_like(x)) for x in all_missing)
    return outputs


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(25)
    print(json.dumps({
        "torch_version": torch.__version__,
        "device": "cpu",
        "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (NETWORK, OMNI, MODEL / "zeva/policy_injection.py")},
        "initialization": init_probe(),
        "gradients": grad_probe(),
    }, indent=2))
