# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate client-sampled tactile windows without inventing intermediate observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

TACTILE_FPS = 15
TACTILE_MEMORY_STEPS = 30
XHAND_STATE_DIM = 1972


@dataclass(frozen=True)
class TactileWindow:
    episode_id: str
    frame_index: int
    state: torch.Tensor
    valid: torch.Tensor


class XHandTactileInput:
    """Check dense control-frame history and the cadence of successful policy queries.

    The client owns the 15 Hz ring buffer. This object remembers only the last
    successful query's episode/frame identity; no sensor state survives a reset.
    Call ``commit`` only after inference succeeds, so a failed request is retryable.
    """

    def __init__(self, query_stride: int = 4) -> None:
        self.query_stride = query_stride
        self._episode_id: str | None = None
        self._frame_index: int | None = None

    def prepare(self, obs: dict[str, Any], *, reset: bool) -> TactileWindow:
        required = (
            "tactile_state",
            "tactile_valid",
            "tactile_frame_indices",
            "tactile_fps",
            "control_frame_index",
            "episode_id",
            "observation/state",
        )
        missing = [key for key in required if key not in obs]
        if missing:
            raise ValueError(f"Tactile policy requires client-sampled 15 Hz history; missing {missing}")
        episode_id = obs["episode_id"]
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a nonempty string, changed for each new attempt")
        frame_index = obs["control_frame_index"]
        if isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, (int, np.integer)):
            raise ValueError("control_frame_index must be an integer control-frame counter, not a query counter")
        frame_index = int(frame_index)
        if frame_index < 0:
            raise ValueError("control_frame_index must be nonnegative")
        fps = np.asarray(obs["tactile_fps"])
        if fps.ndim != 0 or fps.dtype.kind not in "iuf" or not np.isfinite(fps) or float(fps) != TACTILE_FPS:
            raise ValueError("tactile_fps must be 15; capture every control frame, including between queries")
        if not reset and (self._episode_id is None or episode_id != self._episode_id):
            raise ValueError("A new episode or reconnection requires cte_reset or reset_tactile_memory")
        if not reset and frame_index != self._frame_index + self.query_stride:
            raise ValueError(
                f"CTE queries must advance exactly {self.query_stride} control frames; reset for a new attempt"
            )

        state = np.asarray(obs["tactile_state"], dtype=np.float32)
        valid = np.asarray(obs["tactile_valid"])
        indices = np.asarray(obs["tactile_frame_indices"])
        if state.ndim != 2 or state.shape[1] != XHAND_STATE_DIM or not 1 <= len(state) <= TACTILE_MEMORY_STEPS:
            raise ValueError("tactile_state must have shape [T,1972] with 1 <= T <= 30")
        if valid.shape != (len(state),) or valid.dtype.kind != "b":
            raise ValueError("tactile_valid must be a boolean vector with one entry per tactile_state row")
        if indices.shape != valid.shape or indices.dtype.kind not in "iu":
            raise ValueError("tactile_frame_indices must be an integer vector with one entry per tactile_state row")
        if not np.isfinite(state).all():
            raise ValueError("tactile_state contains nonfinite values")
        expected_count = min(frame_index + 1, TACTILE_MEMORY_STEPS)
        if valid.sum() != expected_count or not valid[-expected_count:].all():
            raise ValueError(
                "Provide every available 15 Hz frame (up to 30); only episode-start left padding may be invalid"
            )
        expected_indices = np.arange(frame_index - expected_count + 1, frame_index + 1)
        if not np.array_equal(indices[valid], expected_indices) or np.any(indices[~valid] != -1):
            raise ValueError(
                "Tactile frame indices must be consecutive through the current frame; padding indices must be -1"
            )
        current = np.asarray(obs["observation/state"], dtype=np.float32)
        if current.ndim == 2 and len(current):
            current = current[-1]
        if current.shape != (XHAND_STATE_DIM,) or not np.isfinite(current).all():
            raise ValueError("A tactile policy requires the current finite raw observation/state[1972]")
        if not np.allclose(state[-1], current, rtol=1e-5, atol=1e-6):
            raise ValueError("The latest tactile_state row must match the current observation/state")

        padded = np.zeros((TACTILE_MEMORY_STEPS, XHAND_STATE_DIM), dtype=np.float32)
        padded[-expected_count:] = state[valid]
        padded_valid = np.zeros(TACTILE_MEMORY_STEPS, dtype=np.bool_)
        padded_valid[-expected_count:] = True
        return TactileWindow(episode_id, frame_index, torch.from_numpy(padded), torch.from_numpy(padded_valid))

    def commit(self, window: TactileWindow) -> None:
        self._episode_id = window.episode_id
        self._frame_index = window.frame_index
