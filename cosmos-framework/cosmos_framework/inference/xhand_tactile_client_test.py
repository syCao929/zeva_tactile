# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from cosmos_framework.inference.xhand_tactile_client import TactileClientWindow, adapt_factile_client


def test_client_collects_every_tick_and_does_not_fabricate_startup_history():
    history = TactileClientWindow("first")
    history.append(0, np.ones(1972))
    packet = history.packet(0)
    assert packet["tactile_valid"].sum() == 1
    assert not packet["tactile_state"][:-1].any()
    with pytest.raises(ValueError, match="expected frame 1"):
        history.append(4, np.ones(1972))
    history.reset("second")
    history.append(0, np.zeros(1972))
    assert not history.packet(0)["tactile_state"].any()
    assert history.packet(0)["episode_id"] == "second"


def _legacy_client():
    client = ModuleType("legacy_client")
    client.StateHistoryBuffer = object
    client.main = lambda: 0
    client.parse_args = lambda: SimpleNamespace(
        fps=20,
        query_frequency=48,
        policy_input_mode="auto",
        cached_vlm_async_ae=True,
        smoothing_alpha=0.3,
        action_scale=0.5,
    )

    def build_pi0_observation(*, env_state, state_history, frame_idx):
        return {
            "observation/state": state_history.sample(frame_idx),
            "observation/cam_left_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation/cam_front_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation/cam_right_image": np.zeros((2, 2, 3), dtype=np.uint8),
        }

    client.build_pi0_observation = build_pi0_observation
    client.request_action_chunk = lambda **kwargs: (np.zeros((32, 18)), {}, 0.0)
    return client


def test_legacy_adapter_emits_server_protocol_and_pins_control_settings():
    client = _legacy_client()
    adapt_factile_client(client)
    args = client.parse_args()
    assert args.fps == 15 and args.query_frequency == 4
    assert not args.cached_vlm_async_ae
    assert args.smoothing_alpha == args.action_scale == 1
    history = client.StateHistoryBuffer(tuple(range(-29, 1)))
    for index in range(5):
        history.append(index, np.full(1972, index, dtype=np.float32))
    observation = client.build_pi0_observation(
        env_state=np.full(1972, 4, dtype=np.float32),
        state_history=history,
        frame_idx=4,
    )
    assert observation["tactile_valid"].sum() == 5
    assert observation["observation/state"].shape == (1972,)
    assert "observation.images.cam_left" in observation
    assert "observation.images.cam_front" in observation
    assert "observation.images.cam_right" in observation
    np.testing.assert_array_equal(observation["tactile_frame_indices"][-5:], np.arange(5))


def test_legacy_adapter_stops_after_inference_error_instead_of_mislabeling_hold_actions():
    client = _legacy_client()

    def fail(**kwargs):
        raise OSError("disconnected")

    client.request_action_chunk = fail
    adapt_factile_client(client)
    with pytest.raises(SystemExit, match="fresh episode"):
        client.request_action_chunk()
