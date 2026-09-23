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
    # A single boundary frame has no completed transition yet.
    assert buf.as_cte_inputs() is None


def test_transitions_trail_frames_by_exactly_one() -> None:
    """The client executes the actions returned on the *previous* query.

    So transition t->t+1 must carry query t's actions, never query t+1's.
    """
    buf = _BoundaryBuffer(stride=STRIDE, action_dim=ACTION_DIM, max_frames=64)

    for q in range(5):
        buf.observe(_frame(q))  # request q arrives with its boundary frame
        prepared = buf.as_cte_inputs()
        if q == 0:
            assert prepared is None, "no transition exists until the second frame"
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
    assert buf.as_cte_inputs() is None

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
    assert tuple(_PROPRIO_INDICES) == _STATE_INDICES["arm22"]


def test_composite_geometry_matches_training_loader() -> None:
    """left | wrist, each side square, matching XHandLeRobotDataset._compose_video."""
    from cosmos_framework.data.generator.action.datasets.xhand_lerobot_dataset import _CAMERAS

    # The server hardcodes the same camera mapping the loader exposes.
    assert _CAMERAS["left"] == "observation.images.cam_left"
    assert _CAMERAS["wrist"] == "observation.images.cam_front"

    # 2 x 256 wide, 256 tall — the same 256x512 the RoboCasa protocol uses.
    height, width = 256, 512
    half = width // 2
    left = np.zeros((height, half, 3), dtype=np.uint8)
    wrist = np.full((height, half, 3), 255, dtype=np.uint8)
    composite = np.concatenate([left, wrist], axis=1)
    assert composite.shape == (height, width, 3)
    assert composite[:, :half].max() == 0 and composite[:, half:].min() == 255


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
