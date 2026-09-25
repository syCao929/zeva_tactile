# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Causal tactile memory for the current Zeva attempt.

The frozen patch encoder and its projector produce one token per finger for
each 15 Hz tactile frame.  This module owns the short online memory: it keeps
the latest configurable 1--2 seconds of projected tokens and applies a shared
causal GRU independently to each finger.  It does not read future frames or
perform cross-attempt retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TactileBITConfig:
    """Shape and causal-window configuration for tactile BIT."""

    token_dim: int = 256
    memory_dim: int = 256
    num_fingers: int = 5
    memory_steps: int = 30  # 2 seconds at the 15 Hz XHand rate.

    def __post_init__(self) -> None:
        if self.token_dim <= 0 or self.memory_dim <= 0:
            raise ValueError("tactile BIT dimensions must be positive")
        if self.num_fingers <= 0 or self.memory_steps <= 0:
            raise ValueError("tactile BIT fingers and memory_steps must be positive")


class TactileBIT(nn.Module):
    """Windowed causal memory over projected per-finger tactile tokens.

    ``forward`` is the training/evaluation path and accepts a complete causal
    window ``[B,T,F,D]``.  ``step`` is the online path: it appends one current
    frame, truncates the internal buffer to ``memory_steps``, and returns the
    latest memory state ``[B,F,M]``.  Call ``reset`` at episode/attempt reset.
    """

    def __init__(self, config: TactileBITConfig | None = None) -> None:
        super().__init__()
        self.config = config or TactileBITConfig()
        cfg = self.config
        self.input_norm = nn.LayerNorm(cfg.token_dim)
        self.input_projection = nn.Linear(cfg.token_dim, cfg.memory_dim)
        self.temporal = nn.GRU(cfg.memory_dim, cfg.memory_dim, batch_first=True)
        self.output_norm = nn.LayerNorm(cfg.memory_dim)
        self._online_tokens: Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every trainable tensor, including after meta materialization."""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        for norm in (self.input_norm, self.output_norm):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        for parameter in self.temporal.parameters():
            if parameter.ndim >= 2:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)

    def _validate_tokens(self, tokens: Tensor) -> None:
        cfg = self.config
        if tokens.ndim != 4 or tuple(tokens.shape[2:]) != (
            cfg.num_fingers,
            cfg.token_dim,
        ):
            raise ValueError(
                f"Expected tactile tokens [B,T,{cfg.num_fingers},{cfg.token_dim}], got {tuple(tokens.shape)}"
            )
        if tokens.shape[1] <= 0:
            raise ValueError("tactile BIT requires at least one frame")

    def forward(self, tokens: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        """Encode a causal window, returning ``[B,T_window,F,memory_dim]``."""

        tokens = torch.as_tensor(tokens)
        self._validate_tokens(tokens)
        # Preserve the incoming autocast dtype.  The BIT is trained under the
        # policy's mixed precision context; forcing fp32 here makes LayerNorm
        # receive a dtype different from its FSDP-managed parameters.
        tokens = tokens[:, -self.config.memory_steps :]
        if valid_mask is not None:
            valid_mask = torch.as_tensor(valid_mask, device=tokens.device, dtype=torch.bool)
            if valid_mask.ndim != 2 or valid_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"Expected tactile valid_mask {tuple(tokens.shape[:2])}, got {tuple(valid_mask.shape)}"
                )
        batch, steps, fingers, _ = tokens.shape
        x = self.input_projection(self.input_norm(tokens))
        x = x.permute(0, 2, 1, 3).reshape(batch * fingers, steps, self.config.memory_dim)
        if valid_mask is None:
            y, _ = self.temporal(x)
            y = self.output_norm(y)
        else:
            # A packed sequence would handle right padding efficiently, but
            # tactile windows use left padding at episode start.  Sequential
            # updates let invalid warm-up rows leave the GRU state untouched.
            hidden = x.new_zeros((1, batch * fingers, self.config.memory_dim))
            outputs = []
            for step in range(steps):
                candidate, candidate_hidden = self.temporal(x[:, step : step + 1], hidden)
                valid = valid_mask[:, step].repeat_interleave(fingers)
                hidden = torch.where(valid[None, :, None], candidate_hidden, hidden)
                outputs.append(torch.where(valid[:, None, None], candidate, torch.zeros_like(candidate)))
            y = self.output_norm(torch.cat(outputs, dim=1))
        return y.reshape(batch, fingers, steps, self.config.memory_dim).permute(0, 2, 1, 3)

    def reset(self) -> None:
        """Clear the online buffer at an episode or attempt boundary."""

        self._online_tokens = None

    @property
    def online_length(self) -> int:
        """Number of frames currently retained by the online buffer."""

        return 0 if self._online_tokens is None else int(self._online_tokens.shape[1])

    @torch.no_grad()
    def step(self, current_tokens: Tensor) -> Tensor:
        """Append one ``[B,F,D]`` frame and return the latest memory state."""

        current_tokens = torch.as_tensor(current_tokens)
        if current_tokens.ndim != 3 or tuple(current_tokens.shape[1:]) != (
            self.config.num_fingers,
            self.config.token_dim,
        ):
            raise ValueError(
                f"Expected current tactile tokens [B,{self.config.num_fingers},{self.config.token_dim}], "
                f"got {tuple(current_tokens.shape)}"
            )
        current_tokens = current_tokens.to(dtype=next(self.parameters()).dtype)
        if self._online_tokens is None:
            self._online_tokens = current_tokens[:, None]
        else:
            if self._online_tokens.shape[0] != current_tokens.shape[0]:
                raise ValueError("online tactile BIT batch size changed; call reset first")
            self._online_tokens = torch.cat((self._online_tokens, current_tokens[:, None]), dim=1)
            self._online_tokens = self._online_tokens[:, -self.config.memory_steps :]
        return self.forward(self._online_tokens)[:, -1]


@dataclass(frozen=True)
class TactileBehaviorConfig:
    """Projection from per-finger BIT state to Zeva phase/effect spaces."""

    memory_dim: int = 256
    num_fingers: int = 5
    hidden_dim: int = 256
    phase_dim: int = 128
    effect_dim: int = 128


class TactileBehaviorHead(nn.Module):
    """Produce tactile phase/effect candidates from the latest BIT state."""

    def __init__(self, config: TactileBehaviorConfig | None = None) -> None:
        super().__init__()
        self.config = config or TactileBehaviorConfig()
        cfg = self.config
        self.finger_projection = nn.Linear(cfg.memory_dim, cfg.hidden_dim)
        self.phase_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.phase_dim),
        )
        self.effect_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.effect_dim),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, 1),
        )
        # Retain these checkpoint keys while this recipe trains effect only.
        self.phase_head.requires_grad_(False)
        self.confidence_head.requires_grad_(False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore Linear and LayerNorm parameters after ``to_empty``."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, latest_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        if latest_tokens.ndim != 3 or tuple(latest_tokens.shape[1:]) != (
            cfg.num_fingers,
            cfg.memory_dim,
        ):
            raise ValueError(
                f"Expected latest tactile BIT state [B,{cfg.num_fingers},{cfg.memory_dim}], "
                f"got {tuple(latest_tokens.shape)}"
            )
        pooled = torch.nn.functional.silu(self.finger_projection(latest_tokens)).mean(dim=1)
        phase = torch.nn.functional.normalize(self.phase_head(pooled), dim=-1)
        effect = torch.nn.functional.normalize(self.effect_head(pooled), dim=-1)
        confidence = torch.sigmoid(self.confidence_head(pooled))
        return phase, effect, confidence
