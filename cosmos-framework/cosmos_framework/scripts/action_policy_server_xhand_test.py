# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the XHand server's server-side CTE history reconstruction.

The interesting property is the *attribution*: with a chunk-cadence client the
server must reconstruct exactly the ``(boundary_frames, transition_actions)``
pair a history-sending client would have produced. These tests pin that down
against a hand-computed expectation rather than against the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.datasets.xhand_lerobot_dataset import _STATE_INDICES
from cosmos_framework.scripts.action_policy_server_xhand import _PROPRIO_INDICES, _BoundaryBuffer

STRIDE = 4
ACTION_DIM = 18
CHUNK = 32


def _chunk(tag: int) -> np.ndarray:
    """A distinguishable action chunk; row i, col j == tag*1000 + i*10 + j."""
    return np.arange(CHUNK * ACTION_DIM, dtype=np.float32).reshape(CHUNK, ACTION_DIM) + tag * 1000.0


def _frame(tag: int) -> torch.Tensor:
    return torch.full((1, 48, 30, 52), float(tag))


def test_no_history_before_second_frame() -> None:
    buf = _BoundaryBuffer(stride=STRIDE, action_dim=ACTION_DIM, max_frames=64)
    assert buf.as_cte_inputs() is None

    buf.observe(_frame(0))
    assert buf.num_frames == 1
    # A single observation still supplies a causal phase for PIM retrieval.
    latents, transitions = buf.as_cte_inputs()
    assert latents.shape[1] == 1 and transitions.shape[1] == 0


def test_transitions_trail_frames_by_exactly_one() -> None:
    """The client executes the actions returned on the *previous* query.

    So transition t->t+1 must carry query t's actions, never query t+1's.
    """
    buf = _BoundaryBuffer(stride=STRIDE, action_dim=ACTION_DIM, max_frames=64)

    for q in range(5):
        buf.observe(_frame(q))  # request q arrives with its boundary frame
        prepared = buf.as_cte_inputs()
        if q == 0:
            assert prepared[1].shape[1] == 0, "the initial phase sees no executed controls"
        else:
            latents, transitions = prepared
            assert latents.shape == (1, q + 1, 1, 48, 30, 52)
            assert transitions.shape == (1, q, STRIDE, ACTION_DIM)
            # Query q-1's chunk is what was executed over [q-1, q).
            for t in range(q):
                expected = _chunk(t)[:STRIDE]
                np.testing.assert_allclose(transitions[0, t], expected)
        # The server then generates and returns a chunk for query q.
        buf.record_returned_actions(_chunk(q))

    prepared = buf.as_cte_inputs()
    assert prepared is not None
    latents, transitions = prepared
    assert latents.shape[1] == 5
    assert transitions.shape[:2] == (1, 4)
    np.testing.assert_allclose(transitions[0, 3], _chunk(3)[:STRIDE])


def test_frame_records_do_not_leak_into_transitions() -> None:
    """Latent values must not be contaminated by action bookkeeping."""
    buf = _BoundaryBuffer(stride=STRIDE, action_dim=ACTION_DIM, max_frames=64)
    for q in range(3):
        buf.observe(_frame(q))
        buf.record_returned_actions(_chunk(q))
    latents, _ = buf.as_cte_inputs()
    for q in range(3):
        assert torch.all(latents[0, q] == float(q))


def test_reset_clears_pending_actions() -> None:
    """After a reset the next query must not inherit the previous run's chunk."""
    buf = _BoundaryBuffer(stride=STRIDE, action_dim=ACTION_DIM, max_frames=64)
    buf.observe(_frame(0))
    buf.record_returned_actions(_chunk(99))

    buf.reset()
    assert buf.num_frames == 0
    assert buf.as_cte_inputs() is None

    # The real cadence after a reset: first frame, generate, then next frame.
    buf.observe(_frame(1))
    buf.record_returned_actions(_chunk(1))
    buf.observe(_frame(2))
    prepared = buf.as_cte_inputs()
    assert prepared is not None
    _, transitions = prepared
    np.testing.assert_allclose(transitions[0, 0], _chunk(1)[:STRIDE])


def test_lost_request_restarts_the_window() -> None:
    """A dropped request must not leave transitions permanently out of step.

    Without recovery the frame count would run ahead of the transition count by
    two forever, and ``as_cte_inputs`` would starve for the rest of the episode.
    """
    buf = _BoundaryBuffer(stride=STRIDE, action_dim=ACTION_DIM, max_frames=64)
    for q in range(3):
        buf.observe(_frame(q))
        buf.record_returned_actions(_chunk(q))
    assert buf.as_cte_inputs() is not None

    # Query 3 arrives and is buffered, but generation fails, so no chunk is ever
    # recorded for it — the pending slot is left empty.
    buf.observe(_frame(3))
    assert buf.num_frames == 4

    # Query 4 then finds no pending chunk despite a non-empty window: the
    # attribution of everything buffered is now untrustworthy, so restart.
    buf.observe(_frame(4))
    assert buf.num_frames == 1, "stale window must be dropped, not carried forward"
    assert buf.as_cte_inputs()[1].shape[1] == 0

    # The episode recovers on the next exchange.
    buf.record_returned_actions(_chunk(4))
    buf.observe(_frame(5))
    prepared = buf.as_cte_inputs()
    assert prepared is not None
    latents, transitions = prepared
    assert latents.shape[1] == 2
    np.testing.assert_allclose(transitions[0, 0], _chunk(4)[:STRIDE])


def test_proprio_indices_match_training() -> None:
    """The server's proprio slice must be the one training used.

    If these drift apart the policy silently receives the wrong conditioning
    channels — no shape error, just a quietly worse policy.
    """
    assert tuple(_PROPRIO_INDICES) == _STATE_INDICES["joint18"]


def test_proprio_covers_the_action_space() -> None:
    """Proprio must describe the joints the policy is commanding.

    Both reference implementations hold to this: pi0/pi0.5 build `state_proj` as
    `Linear(action_dim, width)` (openpi pi0.py:159), and Zeva's proprio is its
    action's DOF set (end-effector + gripper) in absolute form.

    The XHand policy observes all 18 arm and hand joint positions it commands.
    """
    from cosmos_framework.data.generator.action.datasets.xhand_lerobot_dataset import (
        _ACTION_INDICES,
        _HAND_POS,
    )

    assert tuple(_PROPRIO_INDICES) == tuple(range(6)) + _HAND_POS
    assert len(_PROPRIO_INDICES) == len(_ACTION_INDICES["full18"]) == 18
    # The hand half must be the position lanes, not the interleaved torque lanes.
    assert set(_HAND_POS).isdisjoint(range(29, 52, 2))


def test_composite_geometry_matches_training_loader() -> None:
    from types import SimpleNamespace

    from cosmos_framework.data.generator.action.xhand_camera import CAMERA_KEYS, compose_xhand_views
    from cosmos_framework.scripts.action_policy_server_xhand import XHandPolicyService

    views = {name: torch.full((1, 3, 480, 640), level / 255.0)
             for name, level in (("front", 32), ("left", 96), ("wrist", 224))}
    service = XHandPolicyService.__new__(XHandPolicyService)
    service.xargs = SimpleNamespace(camera_view_size=256)
    obs = {CAMERA_KEYS[name]: (image[0].permute(1, 2, 0) * 255).byte().numpy()
           for name, image in views.items()}
    actual = service._compose_client_view(obs)
    expected = (compose_xhand_views(views)[0].permute(1, 2, 0) * 255).byte().numpy()
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (576, 512, 3)
    assert actual[100, 100, 0] == 224  # wrist above
    assert actual[450, 100, 0] == 32   # front below left
    assert actual[450, 400, 0] == 96   # external left below right
    del obs["observation.images.cam_right"]
    with pytest.raises(ValueError, match="cam_right"):
        service._compose_client_view(obs)


def test_window_is_bounded() -> None:
    """Frames and transitions stay aligned once the deque starts evicting."""
    max_frames = 4
    buf = _BoundaryBuffer(stride=STRIDE, action_dim=ACTION_DIM, max_frames=max_frames)
    for q in range(10):
        buf.observe(_frame(q))
        buf.record_returned_actions(_chunk(q))

    assert buf.num_frames == max_frames
    prepared = buf.as_cte_inputs()
    assert prepared is not None
    latents, transitions = prepared
    assert latents.shape[1] == max_frames
    assert transitions.shape[:2] == (1, max_frames - 1)
    # Frames kept are the newest ones; transitions still trail by one.
    assert torch.all(latents[0, -1] == 9.0)
    np.testing.assert_allclose(transitions[0, -1], _chunk(8)[:STRIDE])


def _tactile_request(history, frame: int, *, start: int = 0) -> dict:
    for index in range(start, frame + 1):
        state = np.full(1972, float(index), dtype=np.float32)
        history.append(index, state)
    return {
        **history.packet(frame),
        "observation/state": np.full(1972, float(frame), dtype=np.float32),
        "prompt": "Press the button four times.",
    }


def _cpu_tactile_service():
    import threading
    from types import SimpleNamespace

    from cosmos_framework.scripts.action_policy_server_xhand import XHandPolicyService

    service = XHandPolicyService.__new__(XHandPolicyService)
    service.cfg = SimpleNamespace(
        action_chunk_size=32,
        action_dim=18,
        conditioning_fps=15,
        domain_name="ur7e-xhand",
        resolution="256",
        history_length=0,
        guidance=1.0,
        num_steps=1,
        shift=1.0,
    )
    service.model = SimpleNamespace(
        config=SimpleNamespace(
            behavior_stage2=SimpleNamespace(tactile_enabled=True, tactile_memory_steps=30),
        )
    )
    service._pim_context = None
    service._zeva_enabled = True
    service._init_tactile_input()
    service._episode_reset_pending = True
    service._compose_client_view = lambda obs: np.zeros((2, 4, 3), dtype=np.uint8)
    service._transform = lambda sample, resolution: dict(sample)
    service._lock = threading.Lock()
    service._cte_buffer = _BoundaryBuffer(stride=4, action_dim=18, max_frames=17)
    service._update_cte_buffer = lambda obs, image: service._cte_buffer.observe(torch.zeros(1, 2, 2))
    service._causal_interaction_features_from_buffer = lambda: (
        torch.zeros(1, 256),
        torch.zeros(1, 128),
        torch.zeros(1, 4, 128),
        torch.zeros(1, 4, dtype=torch.bool),
    )
    service.xargs = SimpleNamespace(require_cte_history=False)
    service._next_seed = lambda: 0
    service._denormalize = lambda action: action
    service.observed_batches = []

    def generate(data_batch, **kwargs):
        service.observed_batches.append(data_batch)
        return {"action": [torch.zeros(32, 18)]}

    service.model.generate_samples_from_batch = generate
    return service


def test_tactile_inference_preserves_dense_frames_between_cte_queries_and_resets() -> None:
    from cosmos_framework.inference.xhand_tactile_client import TactileClientWindow

    service = _cpu_tactile_service()
    history = TactileClientWindow("attempt-1")
    first = _tactile_request(history, 0)
    assert service.infer(first)["actions"].shape == (32, 18)
    second = _tactile_request(history, 4, start=1)
    service.infer(second)
    batch = service.observed_batches[-1]
    assert batch["tactile_state"][0].shape == (1, 30, 1972)
    assert batch["tactile_valid"][0].shape == (1, 30)
    assert batch["tactile_valid"][0].sum() == 5
    torch.testing.assert_close(batch["tactile_state"][0][0, -5:, 0], torch.arange(5, dtype=torch.float32))
    assert service._cte_buffer.num_frames == 2

    history.reset("attempt-2")
    service.infer(_tactile_request(history, 0))
    assert service._cte_buffer.num_frames == 1
    assert service.observed_batches[-1]["tactile_valid"][0].sum() == 1
    assert torch.count_nonzero(service.observed_batches[-1]["tactile_state"][0]) == 0


def test_tactile_window_rejects_sparse_query_samples() -> None:
    from cosmos_framework.inference.xhand_tactile import XHandTactileInput
    from cosmos_framework.inference.xhand_tactile_client import TactileClientWindow

    request = _tactile_request(TactileClientWindow(), 4)
    request["tactile_valid"][-4:-1] = False
    request["tactile_frame_indices"][-4:-1] = -1
    with pytest.raises(ValueError, match="every available 15 Hz frame"):
        XHandTactileInput().prepare(request, reset=True)


@pytest.mark.parametrize("field", ["tactile_state", "tactile_valid", "tactile_frame_indices", "tactile_fps"])
def test_tactile_window_requires_explicit_sensor_history(field) -> None:
    from cosmos_framework.inference.xhand_tactile import XHandTactileInput
    from cosmos_framework.inference.xhand_tactile_client import TactileClientWindow

    request = _tactile_request(TactileClientWindow(), 0)
    del request[field]
    with pytest.raises(ValueError, match="missing"):
        XHandTactileInput().prepare(request, reset=True)


@pytest.mark.parametrize("problem", ["nan", "current", "fps", "indices", "mask", "future"])
def test_tactile_window_rejects_invalid_observations(problem) -> None:
    from cosmos_framework.inference.xhand_tactile import XHandTactileInput
    from cosmos_framework.inference.xhand_tactile_client import TactileClientWindow

    request = _tactile_request(TactileClientWindow(), 4)
    if problem == "nan":
        request["tactile_state"][-2, 52] = np.nan
    elif problem == "current":
        request["observation/state"][100] += 1
    elif problem == "fps":
        request["tactile_fps"] = 15 / 4
    elif problem == "indices":
        request["tactile_frame_indices"][-2] = 0
    elif problem == "mask":
        request["tactile_valid"] = request["tactile_valid"].astype(np.int64)
    else:
        request["tactile_frame_indices"][-1] = 5
    with pytest.raises(ValueError):
        XHandTactileInput().prepare(request, reset=True)


def test_tactile_window_cadence_episode_identity_and_reconnect() -> None:
    from cosmos_framework.inference.xhand_tactile import XHandTactileInput
    from cosmos_framework.inference.xhand_tactile_client import TactileClientWindow

    adapter = XHandTactileInput()
    history = TactileClientWindow("attempt-1")
    adapter.commit(adapter.prepare(_tactile_request(history, 0), reset=True))
    wrong_cadence = _tactile_request(history, 8, start=1)
    with pytest.raises(ValueError, match="exactly 4"):
        adapter.prepare(wrong_cadence, reset=False)
    reconnected = adapter.prepare(wrong_cadence, reset=True)
    assert reconnected.valid.sum() == 9
    adapter.commit(reconnected)
    new_attempt = _tactile_request(TactileClientWindow("attempt-2"), 0)
    with pytest.raises(ValueError, match="new episode"):
        adapter.prepare(new_attempt, reset=False)
    # Old history must not survive a new attempt's frame-zero request.
    new_attempt["tactile_state"] = wrong_cadence["tactile_state"]
    new_attempt["tactile_valid"] = wrong_cadence["tactile_valid"]
    new_attempt["tactile_frame_indices"] = wrong_cadence["tactile_frame_indices"]
    with pytest.raises(ValueError, match="every available"):
        adapter.prepare(new_attempt, reset=True)


def test_tactile_short_window_is_left_padded_and_long_window_keeps_last_30() -> None:
    from cosmos_framework.inference.xhand_tactile import XHandTactileInput
    from cosmos_framework.inference.xhand_tactile_client import TactileClientWindow

    request = _tactile_request(TactileClientWindow(), 0)
    for key in ("tactile_state", "tactile_valid", "tactile_frame_indices"):
        request[key] = request[key][-1:]
    padded = XHandTactileInput().prepare(request, reset=True)
    assert padded.state.shape == (30, 1972)
    assert padded.valid.sum() == 1
    request = _tactile_request(TactileClientWindow(), 36)
    window = XHandTactileInput().prepare(request, reset=True)
    torch.testing.assert_close(window.state[:, 0], torch.arange(7, 37, dtype=torch.float32))


def test_tactile_serving_is_enabled_only_by_the_loaded_model_config() -> None:
    service = _cpu_tactile_service()
    service.model.config.behavior_stage2.tactile_enabled = False
    service._init_tactile_input()
    assert service._tactile_input is None
    assert "tactile_state" not in service._build_client_sample(
        {
            "prompt": "Press the button",
            "observation/state": np.zeros(1972, dtype=np.float32),
        }
    )
    service.model.config.behavior_stage2.tactile_enabled = True
    service.cfg.conditioning_fps = 20
    with pytest.raises(ValueError, match="15 Hz"):
        service._init_tactile_input()
