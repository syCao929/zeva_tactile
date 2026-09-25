"""Minimal Zeva transfer to the official OpenPI PyTorch pi0 action expert.

This bridge adds a zero-initialized projection of the action prior to existing
action tokens. It does not add prefix tokens, change attention masks, or add to
the final robot command. It is not a reproduction of Cosmos' full Zeva prefix
path. Training and Euler sampling use the same per-instance suffix wrapper.

OpenPI is imported only when constructing an official backbone. Put its ``src``
and this repository's ``cosmos-framework`` on PYTHONPATH in the OpenPI environment.
Production calls require a successfully loaded pretrained backbone; constructing
a model does not silently authorize training random backbone weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MethodType
from typing import Any, Literal

import torch
from torch import Tensor, nn


PolicyMode = Literal["baseline", "zeva", "zeva_tactile"]


class Pi0ZevaPolicy(nn.Module):
    """Wrap pi0 with a trainable action-prior adapter and optional tactile BIT.

    ``actions`` and the returned samples use the backbone's padded, normalized
    action space. Only the leading ``real_action_dim`` channels enter the flow
    loss or prior NLL. ``behavior`` contains global/phase/effect/effect_valid;
    the existing behavior_global/etc. dataset keys are also accepted.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        mode: PolicyMode = "baseline",
        horizon: int = 32,
        action_dim: int = 32,
        real_action_dim: int = 18,
        prior_loss_weight: float = 0.01,
        tactile_checkpoint: str | Path | None = None,
        freeze_backbone: bool | None = None,
        require_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ("baseline", "zeva", "zeva_tactile"):
            raise ValueError(f"Unknown policy mode: {mode!r}")
        if horizon < 1 or not 0 < real_action_dim <= action_dim:
            raise ValueError(
                "Require horizon > 0 and 0 < real_action_dim <= action_dim"
            )
        if prior_loss_weight < 0:
            raise ValueError("prior_loss_weight must be nonnegative")
        if mode == "zeva_tactile" and tactile_checkpoint is None:
            raise ValueError("zeva_tactile requires a pretrained tactile_checkpoint")
        cfg = backbone.config
        if getattr(cfg, "pi05", False):
            raise ValueError(
                "This bridge targets pi0 with its continuous state token, not pi05"
            )
        if cfg.action_horizon != horizon or cfg.action_dim != action_dim:
            raise ValueError("Backbone action_horizon/action_dim must match the bridge")
        if getattr(cfg, "pytorch_compile_mode", None) is not None:
            raise ValueError(
                "Construct the OpenPI backbone with pytorch_compile_mode=None"
            )
        if hasattr(backbone, "_zeva_suffix_context"):
            raise ValueError("This backbone already belongs to a Pi0ZevaPolicy")

        self.backbone = backbone
        self.mode = mode
        self.horizon = horizon
        self.action_dim = action_dim
        self.real_action_dim = real_action_dim
        self.prior_loss_weight = float(prior_loss_weight)
        self.freeze_backbone = (
            mode != "baseline" if freeze_backbone is None else bool(freeze_backbone)
        )
        self.require_pretrained = bool(require_pretrained)
        self.pretrained_loaded = False
        self.pretrained_source: str | None = None
        self.tactile_checkpoint = (
            str(tactile_checkpoint) if tactile_checkpoint is not None else None
        )
        self.backbone.requires_grad_(not self.freeze_backbone)
        self.prior: nn.Module | None = None
        self.prior_adapter: nn.Linear | None = None
        self.tactile_encoder_projector: nn.Module | None = None
        self.tactile_bit: nn.Module | None = None
        self.tactile_behavior_head: nn.Module | None = None
        self.tactile_effect_gate: nn.Parameter | None = None
        self.tactile_phase_gate: nn.Parameter | None = None

        if mode != "baseline":
            from cosmos_framework.model.zeva.policy_injection import (
                PolicyInjectionConfig,
                PolicyInjectionPrior,
            )

            self.prior = PolicyInjectionPrior(
                PolicyInjectionConfig(action_dim=real_action_dim, horizon=horizon)
            )
            self.prior.init_weights()
            self.prior_adapter = nn.Linear(
                real_action_dim, backbone.action_in_proj.out_features
            )
            nn.init.zeros_(self.prior_adapter.weight)
            nn.init.zeros_(self.prior_adapter.bias)
        if mode == "zeva_tactile":
            from cosmos_framework.model.zeva.tactile_encoder import (
                FrozenTactileEncoderWithProjector,
                TactileEncoderAdapter,
            )
            from cosmos_framework.model.zeva.tactile_memory import (
                TactileBIT,
                TactileBehaviorHead,
            )

            self.tactile_encoder_adapter = TactileEncoderAdapter()
            self.tactile_encoder_projector = FrozenTactileEncoderWithProjector()
            state = torch.load(
                tactile_checkpoint, map_location="cpu", weights_only=True
            )
            self.tactile_encoder_projector.encoder.encoder.load_state_dict(
                state, strict=True
            )
            self.tactile_encoder_projector.encoder.requires_grad_(False)
            self.tactile_bit = TactileBIT()
            self.tactile_behavior_head = TactileBehaviorHead()
            self.tactile_behavior_head.phase_head.requires_grad_(False)
            self.tactile_behavior_head.confidence_head.requires_grad_(False)
            self.tactile_effect_gate = nn.Parameter(torch.zeros(1))
            # Preserve the existing effect-only tactile checkpoint convention.
            self.tactile_phase_gate = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Context-local data avoids retaining previous requests/graphs and makes
        # the suffix addition identical for forward and all denoising steps.
        self._suffix_context: ContextVar[Tensor | None] = ContextVar(
            "pi0_zeva_residual", default=None
        )
        context = self._suffix_context
        original_suffix = backbone.embed_suffix

        def embed_suffix(instance, state, noisy_actions, timestep):
            embs, padding, attention, adarms = original_suffix(
                state, noisy_actions, timestep
            )
            residual = context.get()
            if residual is not None:
                expected = (embs.shape[0], horizon, embs.shape[-1])
                if tuple(residual.shape) != expected or embs.shape[1] < horizon:
                    raise ValueError(
                        f"Action-token residual must have shape {expected}"
                    )
                action_tokens = embs[:, -horizon:] + residual.to(
                    device=embs.device, dtype=embs.dtype
                )
                embs = torch.cat((embs[:, :-horizon], action_tokens), dim=1)
            return embs, padding, attention, adarms

        backbone.embed_suffix = MethodType(embed_suffix, backbone)
        backbone._zeva_suffix_context = context
        if hasattr(backbone, "_preprocess_observation"):
            original_preprocess = backbone._preprocess_observation

            def preprocess(instance, observation, *, train=True):
                # Official forward otherwise augments even after .eval().
                return original_preprocess(
                    observation, train=train and instance.training
                )

            backbone._preprocess_observation = MethodType(preprocess, backbone)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        if self.tactile_encoder_projector is not None:
            self.tactile_encoder_projector.encoder.eval()
        return self

    def mark_backbone_loaded(self, source: str | Path) -> None:
        """Call only after a strict external checkpoint restore succeeds."""
        if not str(source):
            raise ValueError("A restored backbone must have a checkpoint source")
        self.pretrained_loaded = True
        self.pretrained_source = str(source)

    def _check_pretrained(self) -> None:
        if self.require_pretrained and not self.pretrained_loaded:
            raise RuntimeError(
                "Load pretrained backbone weights with load_pretrained_backbone before training or sampling"
            )

    @contextmanager
    def _condition(self, residual: Tensor | None):
        token = self._suffix_context.set(residual)
        try:
            yield
        finally:
            self._suffix_context.reset(token)

    def _tactile_effect(self, state: Tensor | None, valid: Tensor | None) -> Tensor:
        if state is None or valid is None:
            raise ValueError("zeva_tactile requires tactile_state and tactile_valid")
        if state.ndim != 3 or state.shape[-1] != 1972 or not 1 <= state.shape[1] <= 30:
            raise ValueError("Expected tactile_state [B,T,1972] with 1 <= T <= 30")
        if valid.shape != state.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Expected boolean tactile_valid [B,T]")
        if torch.any(valid[:, :-1] & ~valid[:, 1:]):
            raise ValueError("tactile_valid may only have invalid left padding")
        assert (
            self.tactile_bit is not None and self.tactile_encoder_projector is not None
        )
        assert (
            self.tactile_behavior_head is not None
            and self.tactile_effect_gate is not None
        )
        device = next(self.tactile_bit.parameters()).device
        state, valid = state.to(device=device), valid.to(device=device)
        if torch.any(~torch.isfinite(state) & valid.unsqueeze(-1)):
            raise ValueError("Valid tactile frames must contain finite values")
        state = torch.where(valid.unsqueeze(-1), state, torch.zeros_like(state))
        force = self.tactile_encoder_adapter.prepare_current_frame(state)
        tokens = self.tactile_encoder_projector(force)
        tokens = tokens.to(dtype=self.tactile_bit.input_norm.weight.dtype)
        latest = self.tactile_bit(tokens, valid_mask=valid)[:, -1]
        _, effect, _ = self.tactile_behavior_head(latest)
        effect = torch.where(
            valid.any(dim=1, keepdim=True), effect, torch.zeros_like(effect)
        )
        return torch.tanh(self.tactile_effect_gate) * effect

    def _memory(
        self,
        behavior: Mapping[str, Tensor] | None,
        tactile_state: Tensor | None,
        tactile_valid: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if self.mode == "baseline":
            return None, None, None
        if behavior is None:
            raise ValueError(
                "Zeva modes require cached global/phase/effect/effect_valid features"
            )
        assert self.prior is not None and self.prior_adapter is not None
        parameter = next(self.prior.parameters())
        features = []
        for key in ("global", "phase", "effect", "effect_valid"):
            value = behavior.get(key, behavior.get(f"behavior_{key}"))
            if not isinstance(value, Tensor):
                raise ValueError(f"behavior requires tensor {key!r}")
            dtype = torch.bool if key == "effect_valid" else parameter.dtype
            if key != "effect_valid" and not torch.isfinite(value).all():
                raise ValueError(f"behavior {key!r} must be finite")
            features.append(value.to(device=parameter.device, dtype=dtype))
        tactile = (
            self._tactile_effect(tactile_state, tactile_valid)
            if self.mode == "zeva_tactile"
            else None
        )
        mean, std = self.prior(*features, current_effect_residual=tactile)
        return self.prior_adapter(mean), mean, std

    def forward(
        self,
        observation: Any,
        actions: Tensor,
        behavior: Mapping[str, Tensor] | None = None,
        tactile_state: Tensor | None = None,
        tactile_valid: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        self._check_pretrained()
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"Expected padded normalized actions [B,{self.horizon},{self.action_dim}]"
            )
        residual, mean, std = self._memory(behavior, tactile_state, tactile_valid)
        with self._condition(residual):
            per_channel_loss = self.backbone(
                observation, actions, noise=noise, time=time
            )
        if (
            not isinstance(per_channel_loss, Tensor)
            or per_channel_loss.shape != actions.shape
        ):
            raise ValueError(
                "PI0 backbone must return unreduced flow loss [B,H,action_dim]"
            )
        action_flow_loss = per_channel_loss[..., : self.real_action_dim].mean()
        prior_nll = action_flow_loss.new_zeros(())
        if mean is not None:
            from cosmos_framework.model.zeva.policy_injection import gaussian_prior_nll

            prior_nll = gaussian_prior_nll(
                actions[..., : self.real_action_dim].to(mean.dtype), mean, std
            )
        return {
            "loss": action_flow_loss + self.prior_loss_weight * prior_nll,
            "action_flow_loss": action_flow_loss,
            "prior_nll": prior_nll,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        observation: Any,
        behavior: Mapping[str, Tensor] | None = None,
        tactile_state: Tensor | None = None,
        tactile_valid: Tensor | None = None,
        noise: Tensor | None = None,
        num_steps: int = 10,
    ) -> Tensor:
        """Return normalized padded [B,H,action_dim] commands, without decoding."""
        self._check_pretrained()
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        residual, _, _ = self._memory(behavior, tactile_state, tactile_valid)
        with self._condition(residual):
            actions = self.backbone.sample_actions(
                observation.state.device, observation, noise=noise, num_steps=num_steps
            )
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("PI0 sampler returned an incompatible action shape")
        return actions


def build_policy(
    mode: PolicyMode = "baseline",
    *,
    horizon: int = 32,
    prior_loss_weight: float = 0.01,
    tactile_checkpoint: str | Path | None = None,
    freeze_backbone: bool | None = None,
    backbone: nn.Module | None = None,
    action_dim: int = 32,
    real_action_dim: int = 18,
    backbone_config: Any = None,
    backbone_dtype: str = "bfloat16",
    device: str | torch.device | None = None,
    require_pretrained: bool = True,
) -> Pi0ZevaPolicy:
    """Build on CPU by default; this never downloads or converts weights.

    ``backbone_dtype`` controls OpenPI's own mixed-dtype initialization. Do not
    cast the entire wrapper to bf16: pi0 keeps projections and selected norms
    in fp32. ``require_pretrained=False`` is for explicit synthetic tests only.
    """
    if mode == "zeva_tactile" and tactile_checkpoint is None:
        raise ValueError("zeva_tactile requires a pretrained tactile_checkpoint")
    if mode == "zeva_tactile" and not Path(tactile_checkpoint).is_file():
        raise FileNotFoundError(
            "tactile_checkpoint must be the converted PyTorch encoder file, not an Orbax directory"
        )
    if backbone is None:
        try:
            from openpi.models.pi0_config import Pi0Config
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        except ImportError as exc:
            raise ImportError(
                "Use the prepared OpenPI environment with openpi/src on PYTHONPATH"
            ) from exc
        cfg = backbone_config or Pi0Config(
            action_dim=action_dim,
            action_horizon=horizon,
            pi05=False,
            pytorch_compile_mode=None,
            dtype=backbone_dtype,
        )
        backbone = PI0Pytorch(cfg)
    policy = Pi0ZevaPolicy(
        backbone,
        mode=mode,
        horizon=horizon,
        action_dim=action_dim,
        real_action_dim=real_action_dim,
        prior_loss_weight=prior_loss_weight,
        tactile_checkpoint=tactile_checkpoint,
        freeze_backbone=freeze_backbone,
        require_pretrained=require_pretrained,
    )
    return policy.to(device=device) if device is not None else policy


def load_pretrained_backbone(policy: Pi0ZevaPolicy, path: str | Path) -> Path:
    """Strictly load an OpenPI save_model safetensors file, including tied keys.

    A directory resolves to ``model.safetensors``. JAX/Orbax params are not Torch
    weights and must be converted separately; no implicit conversion is done.
    """
    from safetensors.torch import load_model

    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file() or path.suffix != ".safetensors":
        raise FileNotFoundError(f"Expected converted OpenPI model.safetensors: {path}")
    # Clear readiness before attempting a load: strict failure can leave some
    # tensors modified, which must never be mistaken for a usable checkpoint.
    policy.pretrained_loaded = False
    policy.pretrained_source = None
    load_model(policy.backbone, str(path), strict=True, device="cpu")
    policy.mark_backbone_loaded(path.resolve())
    return path
