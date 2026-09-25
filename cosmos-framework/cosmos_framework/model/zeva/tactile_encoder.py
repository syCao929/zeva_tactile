# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Input contract and PyTorch implementation for the frozen XHand encoder.

The pretrained encoder lives in the external FactileLDM JAX/Flax repository.
The policy-facing path is deliberately frame based: the external tokenizer is a
spatial encoder whose steps are independent, so temporal memory can be added by
the policy's tactile memory module later.  The causal history helpers remain
available for checkpoint inspection and ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch import Tensor


TACTILE_STATE_DIM = 1972
TACTILE_BLOCK_START = 52
TACTILE_SENSOR_COUNT = 5
TACTILE_BLOCK_SIZE = 384
TACTILE_RAW_FORCE_OFFSET = 24
TACTILE_RAW_FORCE_POINTS = 120
TACTILE_RAW_FORCE_DIM = 3
TACTILE_RAW_FORCE_WIDTH = TACTILE_RAW_FORCE_POINTS * TACTILE_RAW_FORCE_DIM
TACTILE_REQUIRED_STATE_DIM = TACTILE_BLOCK_START + TACTILE_SENSOR_COUNT * TACTILE_BLOCK_SIZE

# These are the effort statistics stored with the external taskall-2
# checkpoint. They are defaults for that checkpoint only; callers should pass
# dataset-specific statistics when the target data distribution differs.
TASKALL2_EFFORT_MEAN = (-0.0178368445, -0.0056814761, 0.6617453098)
TASKALL2_EFFORT_STD = (0.5177429318, 0.6578912139, 11.0228757858)


@dataclass(frozen=True)
class TactileEncoderConfig:
    """Shape and normalization contract for the pretrained encoder."""

    # The production contract is one current frame.  Ten-frame preprocessing
    # is retained by ``prepare_history`` for comparison with stage-1 training.
    history_frames: int = 1
    sample_hz: float = 15.0
    num_fingers: int = TACTILE_SENSOR_COUNT
    points_per_finger: int = TACTILE_RAW_FORCE_POINTS
    dim_per_point: int = TACTILE_RAW_FORCE_DIM
    encoder_width: int = 1024
    encoder_hidden_dim: int = 256
    num_patches: int = 5
    contact_threshold: float = 0.5
    contact_temperature: float = 0.5
    effort_mean: tuple[float, float, float] = TASKALL2_EFFORT_MEAN
    effort_std: tuple[float, float, float] = TASKALL2_EFFORT_STD

    def __post_init__(self) -> None:
        if self.history_frames <= 0:
            raise ValueError("history_frames must be positive")
        if self.sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        if self.encoder_width <= 0 or self.encoder_hidden_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        if self.num_patches != 5:
            raise ValueError("the external XHand checkpoint requires five patches")
        if self.contact_temperature <= 0:
            raise ValueError("contact_temperature must be positive")
        if (self.num_fingers, self.points_per_finger, self.dim_per_point) != (
            TACTILE_SENSOR_COUNT,
            TACTILE_RAW_FORCE_POINTS,
            TACTILE_RAW_FORCE_DIM,
        ):
            raise ValueError("the external XHand checkpoint requires [5, 120, 3] tactile input")
        if len(self.effort_mean) != self.dim_per_point or len(self.effort_std) != self.dim_per_point:
            raise ValueError("effort_mean and effort_std must match dim_per_point")
        if any(float(std) <= 0 for std in self.effort_std):
            raise ValueError("effort_std entries must be positive")


class TactileEncoderAdapter:
    """Prepare XHand state for the frozen tactile encoder.

    The adapter accepts either ``[T, 1972]`` or ``[B, T, 1972]`` state history.
    It never pads from a future frame: asking for a history before the first
    available state raises ``ValueError`` so an online caller can handle warmup
    explicitly.
    """

    def __init__(self, config: TactileEncoderConfig | None = None) -> None:
        self.config = config or TactileEncoderConfig()
        self._mean = torch.tensor(self.config.effort_mean, dtype=torch.float32)
        self._std = torch.tensor(self.config.effort_std, dtype=torch.float32)

    @property
    def history_times(self) -> Tensor:
        """Return causal offsets for the optional history preprocessing path."""

        return torch.arange(
            -self.config.history_frames + 1,
            1,
            dtype=torch.float32,
        ) / self.config.sample_hz

    def prepare_current_frame(self, state: Tensor) -> Tensor:
        """Extract and normalize one current state.

        ``state`` may have any leading batch dimensions, followed by the XHand
        state dimension.  The result has shape ``[..., 5, 120, 3]`` and is the
        main input contract for the frozen encoder.
        """

        return self.normalize_raw_force(self.extract_raw_force(state))

    @staticmethod
    def extract_raw_force(states: Tensor) -> Tensor:
        """Extract ``[..., 5, 120, 3]`` raw force vectors from XHand state."""

        states = torch.as_tensor(states)
        if states.ndim < 1 or states.shape[-1] < TACTILE_REQUIRED_STATE_DIM:
            raise ValueError(
                "XHand tactile extraction requires state[..., 1972] or wider, "
                f"got {tuple(states.shape)}"
            )
        states = states.to(dtype=torch.float32)
        chunks = []
        for sensor_id in range(TACTILE_SENSOR_COUNT):
            start = TACTILE_BLOCK_START + sensor_id * TACTILE_BLOCK_SIZE + TACTILE_RAW_FORCE_OFFSET
            stop = start + TACTILE_RAW_FORCE_WIDTH
            chunks.append(states[..., start:stop].reshape(*states.shape[:-1], TACTILE_RAW_FORCE_POINTS, 3))
        return torch.stack(chunks, dim=-3)

    def normalize_raw_force(self, raw_force: Tensor) -> Tensor:
        """Apply checkpoint-compatible per-force-component normalization."""

        raw_force = torch.as_tensor(raw_force, dtype=torch.float32)
        expected = (self.config.num_fingers, self.config.points_per_finger, self.config.dim_per_point)
        if raw_force.ndim < 3 or tuple(raw_force.shape[-3:]) != expected:
            raise ValueError(f"Expected raw force [...,5,120,3], got {tuple(raw_force.shape)}")
        # The policy network is constructed under ``torch.device('meta')``.
        # Plain tensors created by the adapter would therefore also be meta
        # tensors; rebuild the tiny constants on the actual input device when
        # needed instead of attempting to copy them out of meta.
        if self._mean.is_meta or self._std.is_meta:
            mean = torch.tensor(self.config.effort_mean, device=raw_force.device, dtype=raw_force.dtype)
            std = torch.tensor(self.config.effort_std, device=raw_force.device, dtype=raw_force.dtype)
        else:
            mean = self._mean.to(device=raw_force.device, dtype=raw_force.dtype)
            std = self._std.to(device=raw_force.device, dtype=raw_force.dtype)
        # Match FactileLDM's transforms.Normalize exactly. The epsilon is part
        # of the checkpoint input contract, rather than only a divide-by-zero
        # safeguard for this dataset.
        return (raw_force - mean) / (std + 1e-6)

    def prepare_history(self, states: Tensor, current_index: int | None = None) -> Tensor:
        """Extract and normalize a history ending at ``current_index``.

        For ``states=[B,T,D]`` the result is ``[B,history,5,120,3]``. For
        ``states=[T,D]`` the batch dimension is preserved as absent.
        """

        states = torch.as_tensor(states)
        if states.ndim < 2:
            raise ValueError(f"Expected state history [T,D] or [B,T,D], got {tuple(states.shape)}")
        time_steps = states.shape[-2]
        end = time_steps - 1 if current_index is None else int(current_index)
        if not 0 <= end < time_steps:
            raise ValueError(f"current_index must be in [0,{time_steps}), got {end}")
        start = end - self.config.history_frames + 1
        if start < 0:
            raise ValueError(
                f"need {self.config.history_frames} causal states before index {end}, "
                f"but only {end + 1} are available"
            )
        raw = self.extract_raw_force(states[..., start : end + 1, :])
        return self.normalize_raw_force(raw)

    def validate_history_tokens(self, tokens: Tensor) -> Tensor:
        """Validate structured or flattened history tokens and return unchanged.

        FactileLDM's low-level ``_encode_steps`` returns ``[...,T,5,D]`` while
        its ``encode_history`` convenience method flattens this to
        ``[...,T*5,D]``.  Both forms are accepted for checkpoint comparisons.
        """

        tokens = torch.as_tensor(tokens)
        structured = (self.config.history_frames, self.config.num_fingers, self.config.encoder_width)
        flattened = (self.config.history_frames * self.config.num_fingers, self.config.encoder_width)
        if (tokens.ndim < 3 or tuple(tokens.shape[-3:]) != structured) and (
            tokens.ndim < 2 or tuple(tokens.shape[-2:]) != flattened
        ):
            raise ValueError(
                f"Expected tactile tokens [...,{structured[0]},5,{structured[2]}] or "
                f"[...,{flattened[0]},{flattened[1]}], got {tuple(tokens.shape)}"
            )
        return tokens

    def validate_current_tokens(self, tokens: Tensor) -> Tensor:
        """Validate current-frame encoder output ``[..., 5, 1024]``."""

        tokens = torch.as_tensor(tokens)
        expected = (self.config.num_fingers, self.config.encoder_width)
        if tokens.ndim not in (2, 3) or tuple(tokens.shape[-2:]) != expected:
            raise ValueError(
                f"Expected current tactile tokens [...,{expected[0]},{expected[1]}], "
                f"got {tuple(tokens.shape)}"
            )
        return tokens


def _continuous_time_embedding(times_seconds: Tensor, dim: int) -> Tensor:
    """Match FactileLDM's sinusoidal time embedding for parity checks."""

    half = dim // 2
    if half == 0:
        return torch.zeros((*times_seconds.shape, dim), dtype=torch.float32, device=times_seconds.device)
    frequencies = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(1.0, device=times_seconds.device)),
            torch.log(torch.tensor(1000.0, device=times_seconds.device)),
            half,
            dtype=torch.float32,
            device=times_seconds.device,
        )
    )
    angles = times_seconds[..., None].float() * (2.0 * torch.pi) * frequencies
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


class PatchInformedFingerTokenizerTorch(nn.Module):
    """Torch equivalent of FactileLDM's patch-informed spatial tokenizer.

    This class intentionally exposes only the encoder forward pass.  It accepts
    either ``[B, 5, 120, 3]`` (the policy path) or ``[B, T, 5, 120, 3]`` and
    returns ``[B, 5, 1024]`` or ``[B, T, 5, 1024]`` respectively.  The
    implementation has no temporal attention; ``T`` only indexes independent
    spatial encoding steps.
    """

    _THUMB_PATCH_IDS = (
        2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0,
        2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        2, 2, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0,
        2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    )
    _OTHER_PATCH_IDS = (
        2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0,
        2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0,
        2, 2, 2, 1, 1, 3, 3, 3, 0, 0, 0, 0,
        2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        2, 2, 2, 1, 1, 4, 4, 4, 0, 0, 0, 0,
        2, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0,
        2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0,
    )

    def __init__(self, config: TactileEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or TactileEncoderConfig()
        c = self.config
        d, h = c.encoder_width, c.encoder_hidden_dim
        self.force_proj_in = nn.Linear(c.dim_per_point, h)
        self.force_proj_out = nn.Linear(h, d)
        self.time_proj = nn.Linear(64, d)
        # These four inherited RawTactileSpatialTokenizer parameters are not
        # read by PatchInformedFingerTokenizer._encode_steps, but keeping them
        # makes the complete Flax ``patch_encoder`` subtree loadable.
        self.contact_proj = nn.Linear(2, d)
        self.point_score = nn.Linear(d, 1)
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.patch_stat_proj_in = nn.Linear(2 * c.dim_per_point + 2, h)
        self.patch_stat_proj_out = nn.Linear(h, d)
        self.patch_score = nn.Linear(d, 1)
        self.patch_norm = nn.LayerNorm(d, eps=1e-6)
        self.finger_embedding = nn.Parameter(torch.empty(c.num_fingers, d))
        self.point_embedding = nn.Parameter(torch.empty(c.points_per_finger, d))
        self.patch_embedding = nn.Parameter(torch.empty(c.num_patches, d))
        self.type_embedding = nn.Parameter(torch.empty(2, d))
        self.segment_pool_logits = nn.Parameter(torch.zeros(4))
        # Keep standalone construction finite for unit tests and CPU shape
        # checks; all four embeddings are overwritten by the external
        # checkpoint before training.
        for embedding in (
            self.finger_embedding,
            self.point_embedding,
            self.patch_embedding,
            self.type_embedding,
        ):
            nn.init.normal_(embedding, mean=0.0, std=0.02)
        self.register_buffer(
            "point_patch_ids",
            torch.tensor(
                [self._THUMB_PATCH_IDS] + [self._OTHER_PATCH_IDS] * (c.num_fingers - 1),
                dtype=torch.long,
            ),
            persistent=False,
        )

    def reset_patch_map(self) -> None:
        """Restore the non-persistent patch map after meta ``to_empty``."""
        values = torch.tensor(
            [self._THUMB_PATCH_IDS] + [self._OTHER_PATCH_IDS] * (self.config.num_fingers - 1),
            dtype=torch.long,
            device=self.point_patch_ids.device,
        )
        self.point_patch_ids.copy_(values)

    def forward(
        self,
        forces: Tensor,
        times_seconds: Tensor | None = None,
        *,
        future: bool = False,
        include_temporal: bool = False,
    ) -> Tensor:
        forces = torch.as_tensor(forces).to(dtype=torch.float32)
        squeeze_time = forces.ndim == 4
        if squeeze_time:
            forces = forces.unsqueeze(1)
        if forces.ndim != 5:
            raise ValueError(f"Expected tactile force [B,T,F,P,C] or [B,F,P,C], got {tuple(forces.shape)}")
        c = self.config
        if tuple(forces.shape[2:]) != (c.num_fingers, c.points_per_finger, c.dim_per_point):
            raise ValueError(
                "Expected tactile finger/point shape "
                f"{(c.num_fingers, c.points_per_finger, c.dim_per_point)}, got {tuple(forces.shape[2:])}"
            )
        batch_size, time_steps = forces.shape[:2]
        if times_seconds is None:
            times_seconds = torch.zeros(time_steps, dtype=torch.float32, device=forces.device)
        times_seconds = torch.as_tensor(times_seconds, dtype=torch.float32, device=forces.device)
        if tuple(times_seconds.shape) != (time_steps,):
            raise ValueError(f"Expected {time_steps} time offsets, got {tuple(times_seconds.shape)}")

        forces_f32 = forces.float()
        # Spell this out to match jnp.linalg.norm's sum/sqrt order during
        # JAX-to-Torch parity checks (vector_norm may use a different kernel).
        magnitude = torch.sqrt(torch.sum(torch.square(forces_f32), dim=-1))
        gate = torch.sigmoid((magnitude - c.contact_threshold) / max(c.contact_temperature, 1e-6))
        patch_masks = torch.nn.functional.one_hot(self.point_patch_ids, c.num_patches).permute(0, 2, 1).float()
        patch_counts = patch_masks.sum(dim=-1).clamp_min(1.0)
        masked_gate = gate.unsqueeze(-2) * patch_masks[None, None]
        gate_sum = masked_gate.sum(dim=-1)
        gated_force_mean = torch.einsum(
            "btfrp,btfpc->btfrc", masked_gate, forces_f32
        ) / gate_sum.clamp_min(1e-6)[..., None]
        abs_forces = forces_f32.abs()
        patch_abs_max = torch.where(
            patch_masks[None, None, ..., None] > 0,
            abs_forces[:, :, :, None],
            torch.zeros_like(abs_forces[:, :, :, None]),
        ).amax(dim=-2)
        contact_area = gate_sum / patch_counts[None, None]
        patch_strength = torch.where(
            patch_masks[None, None] > 0,
            magnitude[:, :, :, None],
            torch.zeros_like(magnitude[:, :, :, None]),
        ).amax(dim=-1)
        patch_stats = torch.cat(
            [gated_force_mean, patch_abs_max, contact_area[..., None], patch_strength[..., None]], dim=-1
        ).to(dtype=self.patch_stat_proj_in.weight.dtype)

        patch_tokens = torch.nn.functional.silu(self.patch_stat_proj_in(patch_stats))
        patch_tokens = self.patch_stat_proj_out(patch_tokens)
        patch_tokens = patch_tokens + self.finger_embedding[None, None, :, None]
        patch_tokens = patch_tokens + self.patch_embedding[None, None, None, :]
        patch_scores = self.patch_score(torch.nn.functional.silu(patch_tokens)).squeeze(-1).float()
        patch_scores = patch_scores + torch.log(contact_area + 1e-6)
        patch_weights = torch.softmax(patch_scores, dim=-1).to(patch_tokens.dtype)
        finger_tokens = torch.einsum("btfr,btfrd->btfd", patch_weights, patch_tokens)
        if include_temporal:
            time_feature = self.time_proj(_continuous_time_embedding(times_seconds, 64))
            finger_tokens = (
                finger_tokens
                + time_feature[None, :, None]
                + self.type_embedding[int(future)][None, None, None]
            )
        result = self.patch_norm(finger_tokens)
        return result[:, 0] if squeeze_time else result

    @staticmethod
    def _leaf(params: Mapping[str, Any], *names: str) -> Any:
        value: Any = params
        for name in names:
            if not isinstance(value, Mapping) or name not in value:
                raise KeyError("missing Flax parameter " + ".".join(names))
            value = value[name]
        if isinstance(value, Mapping) and "value" in value:
            value = value["value"]
        return value

    @staticmethod
    def _tensor_copy(value: Any, *, dtype: torch.dtype) -> Tensor:
        """Materialize JAX/NumPy leaves before copying into Torch parameters."""

        return torch.from_numpy(np.array(value, copy=True)).to(dtype=dtype)

    def load_flax_params(
        self, params: Mapping[str, Any], *, strict: bool = True
    ) -> "PatchInformedFingerTokenizerTorch":
        """Copy the ``patch_encoder`` subtree from a restored Flax PyTree.

        Flax ``Linear.kernel`` is ``[in, out]`` while Torch stores
        ``Linear.weight`` as ``[out, in]``.  The explicit mapping keeps this
        conversion auditable and avoids silently loading decoder heads.
        """

        root: Mapping[str, Any] = params
        if "params" in root:
            root = root["params"]
        if "patch_encoder" in root:
            root = root["patch_encoder"]
        elif "force_proj_in" not in root and strict:
            raise KeyError("Flax PyTree does not contain params.patch_encoder")

        def linear(module: nn.Linear, name: str) -> None:
            with torch.no_grad():
                kernel = self._tensor_copy(self._leaf(root, name, "kernel"), dtype=module.weight.dtype)
                bias = self._tensor_copy(self._leaf(root, name, "bias"), dtype=module.bias.dtype)
                if tuple(kernel.T.shape) != tuple(module.weight.shape) or tuple(bias.shape) != tuple(
                    module.bias.shape
                ):
                    raise ValueError(f"shape mismatch for Flax parameter {name}")
                module.weight.copy_(kernel.T)
                module.bias.copy_(bias)

        for module, name in (
            (self.force_proj_in, "force_proj_in"),
            (self.force_proj_out, "force_proj_out"),
            (self.time_proj, "time_proj"),
            (self.contact_proj, "contact_proj"),
            (self.point_score, "point_score"),
            (self.patch_stat_proj_in, "patch_stat_proj_in"),
            (self.patch_stat_proj_out, "patch_stat_proj_out"),
            (self.patch_score, "patch_score"),
        ):
            linear(module, name)
        with torch.no_grad():
            for parameter, name in (
                (self.finger_embedding, "finger_embedding"),
                (self.point_embedding, "point_embedding"),
                (self.patch_embedding, "patch_embedding"),
                (self.type_embedding, "type_embedding"),
            ):
                value = self._tensor_copy(self._leaf(root, name), dtype=parameter.dtype)
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"shape mismatch for Flax parameter {name}")
                parameter.copy_(value)
            self.segment_pool_logits.copy_(
                self._tensor_copy(self._leaf(root, "segment_pool_logits"), dtype=self.segment_pool_logits.dtype)
            )
            self.norm.weight.copy_(self._tensor_copy(self._leaf(root, "norm", "scale"), dtype=self.norm.weight.dtype))
            self.norm.bias.copy_(self._tensor_copy(self._leaf(root, "norm", "bias"), dtype=self.norm.bias.dtype))
            self.patch_norm.weight.copy_(
                self._tensor_copy(self._leaf(root, "patch_norm", "scale"), dtype=self.patch_norm.weight.dtype)
            )
            self.patch_norm.bias.copy_(
                self._tensor_copy(self._leaf(root, "patch_norm", "bias"), dtype=self.patch_norm.bias.dtype)
            )
        return self


class FrozenPatchInformedTactileEncoder(nn.Module):
    """Inference wrapper that guarantees the external encoder stays frozen."""

    def __init__(self, encoder: PatchInformedFingerTokenizerTorch | None = None) -> None:
        super().__init__()
        self.encoder = encoder or PatchInformedFingerTokenizerTorch()
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_flax_params(
        cls, params: Mapping[str, Any], config: TactileEncoderConfig | None = None
    ) -> "FrozenPatchInformedTactileEncoder":
        encoder = PatchInformedFingerTokenizerTorch(config)
        encoder.load_flax_params(params)
        return cls(encoder)

    def forward(self, current_force: Tensor) -> Tensor:
        with torch.no_grad():
            # The Cosmos network is materialized in bf16, so the frozen
            # encoder's replicated weights may be bf16 as well. Match its
            # compute dtype while keeping the projector trainable.
            parameter = next(self.encoder.parameters())
            return self.encoder(current_force.to(dtype=parameter.dtype))


@dataclass(frozen=True)
class TactileProjectionConfig:
    """Trainable projection dimensions after the frozen tactile encoder."""

    input_width: int = 1024
    output_width: int = 256

    def __post_init__(self) -> None:
        if self.input_width <= 0 or self.output_width <= 0:
            raise ValueError("tactile projection dimensions must be positive")


class TactileTokenProjector(nn.Module):
    """Project per-finger frozen encoder tokens into the policy width.

    The projector is intentionally a single affine layer.  It preserves the
    leading batch, time, and finger axes, so it can be trained before adding a
    temporal tactile memory module.
    """

    def __init__(self, config: TactileProjectionConfig | None = None) -> None:
        super().__init__()
        self.config = config or TactileProjectionConfig()
        self.projection = nn.Linear(self.config.input_width, self.config.output_width)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim < 2 or tokens.shape[-1] != self.config.input_width:
            raise ValueError(
                f"Expected tactile tokens [...,{self.config.input_width}], got {tuple(tokens.shape)}"
            )
        return self.projection(tokens)


class FrozenTactileEncoderWithProjector(nn.Module):
    """Frozen 1024-wide encoder followed by a trainable 256-wide projector."""

    def __init__(
        self,
        encoder: FrozenPatchInformedTactileEncoder | None = None,
        projector: TactileTokenProjector | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or FrozenPatchInformedTactileEncoder()
        self.projector = projector or TactileTokenProjector()
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @classmethod
    def from_flax_params(
        cls, params: Mapping[str, Any], config: TactileEncoderConfig | None = None
    ) -> "FrozenTactileEncoderWithProjector":
        return cls(FrozenPatchInformedTactileEncoder.from_flax_params(params, config))

    def forward(self, current_force: Tensor) -> Tensor:
        return self.projector(self.encoder(current_force))
