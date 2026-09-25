# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.zeva.tactile_encoder import (
    TACTILE_BLOCK_START,
    TACTILE_BLOCK_SIZE,
    TACTILE_RAW_FORCE_OFFSET,
    FrozenPatchInformedTactileEncoder,
    FrozenTactileEncoderWithProjector,
    PatchInformedFingerTokenizerTorch,
    TactileEncoderConfig,
    TactileEncoderAdapter,
    TactileTokenProjector,
)


def _states(time_steps: int = 10) -> torch.Tensor:
    state = torch.zeros(time_steps, 1972)
    for sensor_id in range(5):
        start = TACTILE_BLOCK_START + sensor_id * TACTILE_BLOCK_SIZE + TACTILE_RAW_FORCE_OFFSET
        state[:, start : start + 360] = 100.0 + sensor_id
    return state


def test_extracts_five_structured_force_sensors() -> None:
    force = TactileEncoderAdapter.extract_raw_force(_states())
    assert force.shape == (10, 5, 120, 3)
    assert torch.all(force[:, 0] == 100.0)
    assert torch.all(force[:, 4] == 104.0)


def test_history_is_causal_and_uses_checkpoint_normalization() -> None:
    adapter = TactileEncoderAdapter(TactileEncoderConfig(history_frames=10))
    states = _states(12)
    states[0, TACTILE_BLOCK_START + TACTILE_RAW_FORCE_OFFSET] = 999.0
    history = adapter.prepare_history(states, current_index=10)
    assert history.shape == (10, 5, 120, 3)
    expected = (100.0 - adapter.config.effort_mean[0]) / (adapter.config.effort_std[0] + 1e-6)
    assert history[0, 0, 0, 0].item() == pytest.approx(expected)


def test_rejects_history_that_would_require_future_padding() -> None:
    adapter = TactileEncoderAdapter(TactileEncoderConfig(history_frames=10))
    with pytest.raises(ValueError, match="need 10 causal states"):
        adapter.prepare_history(_states(9))


def test_validates_encoder_token_contract() -> None:
    adapter = TactileEncoderAdapter(TactileEncoderConfig(history_frames=10))
    tokens = torch.zeros(2, 10, 5, 1024)
    assert adapter.validate_history_tokens(tokens) is tokens
    with pytest.raises(ValueError, match="Expected tactile tokens"):
        adapter.validate_history_tokens(torch.zeros(2, 10, 1024))


def test_current_frame_is_the_primary_adapter_contract() -> None:
    adapter = TactileEncoderAdapter()
    frame = adapter.prepare_current_frame(_states(1)[0])
    assert frame.shape == (5, 120, 3)
    assert adapter.history_times.shape == (1,)
    assert adapter.validate_current_tokens(torch.zeros(2, 5, 1024)).shape == (2, 5, 1024)
    with pytest.raises(ValueError, match="Expected current tactile tokens"):
        adapter.validate_current_tokens(torch.zeros(2, 1, 5, 1024))


def test_torch_patch_encoder_is_frozen_and_preserves_frame_contract() -> None:
    encoder = FrozenPatchInformedTactileEncoder()
    forces = torch.randn(2, 5, 120, 3)
    tokens = encoder(forces)
    assert tokens.shape == (2, 5, 1024)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_torch_patch_encoder_supports_independent_time_steps() -> None:
    encoder = PatchInformedFingerTokenizerTorch()
    forces = torch.randn(2, 3, 5, 120, 3)
    tokens = encoder(forces)
    assert tokens.shape == (2, 3, 5, 1024)
    assert torch.allclose(tokens[:, 0], encoder(forces[:, 0]), atol=2e-5, rtol=2e-5)


def test_official_xhand_patch_map_regression() -> None:
    encoder = PatchInformedFingerTokenizerTorch()
    # The official T16 map assigns taxel 21 to patch 3.  This catches a subtle
    # one-entry map drift that changes the loaded encoder output measurably.
    assert encoder.point_patch_ids[1, 21].item() == 3


def test_projector_preserves_leading_axes_and_is_trainable() -> None:
    projector = TactileTokenProjector()
    tokens = torch.randn(2, 5, 1024, requires_grad=False)
    projected = projector(tokens)
    assert projected.shape == (2, 5, 256)
    projected.square().mean().backward()
    assert all(parameter.grad is not None for parameter in projector.parameters())


def test_encoder_projector_only_updates_projector() -> None:
    model = FrozenTactileEncoderWithProjector()
    output = model(torch.randn(2, 5, 120, 3))
    assert output.shape == (2, 5, 256)
    output.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is not None for parameter in model.projector.parameters())
