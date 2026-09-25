# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""NumPy-only adapter for FactileLDM's existing robot client; no model imports."""

from __future__ import annotations

import argparse
import collections
import importlib.util
import inspect
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


class TactileClientWindow:
    """Collect every control tick, even though the policy is queried every four ticks."""

    def __init__(self, episode_id: str | None = None) -> None:
        self.reset(episode_id)

    def reset(self, episode_id: str | None = None) -> None:
        self.episode_id = episode_id or uuid.uuid4().hex
        self._frames: collections.deque[tuple[int, np.ndarray]] = collections.deque(maxlen=30)

    def append(self, frame_idx: int, state: np.ndarray) -> None:
        expected = self._frames[-1][0] + 1 if self._frames else 0
        if frame_idx != expected:
            raise ValueError(f"Capture every 15 Hz control frame: expected frame {expected}, received {frame_idx}")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (1972,) or not np.isfinite(state).all():
            raise ValueError("The tactile client needs finite raw observation.state[1972] in dataset order")
        self._frames.append((frame_idx, state.copy()))

    def packet(self, frame_idx: int, *, reset: bool = False) -> dict[str, Any]:
        if not self._frames or self._frames[-1][0] != frame_idx:
            raise ValueError("Append the current raw observation before building a tactile request")
        state = np.zeros((30, 1972), dtype=np.float32)
        valid = np.zeros(30, dtype=np.bool_)
        indices = np.full(30, -1, dtype=np.int64)
        count = len(self._frames)
        state[-count:] = np.stack([value for _, value in self._frames])
        valid[-count:] = True
        indices[-count:] = [index for index, _ in self._frames]
        return {
            "episode_id": self.episode_id,
            "control_frame_index": frame_idx,
            "tactile_fps": 15,
            "tactile_state": state,
            "tactile_valid": valid,
            "tactile_frame_indices": indices,
            "cte_reset": reset or frame_idx == 0,
        }

    def sample(self, frame_idx: int) -> np.ndarray:
        return self.packet(frame_idx)["tactile_state"]


def adapt_factile_client(client: ModuleType, *, pim_episode_id: str | None = None, pim_attempt_id: int = 0) -> None:
    """Install narrow hooks on the original client without changing its file."""
    required = ("parse_args", "StateHistoryBuffer", "build_pi0_observation", "request_action_chunk", "main")
    if missing := [name for name in required if not hasattr(client, name)]:
        raise ValueError(f"Unsupported Factile client API; missing {missing}")
    original_parse = client.parse_args
    original_observation = client.build_pi0_observation
    original_request = client.request_action_chunk
    observation_signature = inspect.signature(original_observation)

    class DenseHistory(TactileClientWindow):
        def __init__(self, offsets: tuple[int, ...]) -> None:
            if tuple(offsets) != tuple(range(-29, 1)):
                raise ValueError("The Zeva tactile adapter requires dense offsets -29,...,0")
            super().__init__()

    def parse_args():
        args = original_parse()
        args.fps = 15
        args.query_frequency = 4
        args.policy_input_mode = "structured"
        args.structured_history_offsets = ",".join(str(index) for index in range(-29, 1))
        args.cached_vlm_async_ae = False
        args.smoothing_alpha = 1.0
        args.action_scale = 1.0
        args.max_action_chunk_size = 32
        print(
            "Zeva tactile adapter: 15 Hz, query every 4 controls, 30 real history frames, async off, smoothing/scale=1"
        )
        return args

    def build_observation(*args, **kwargs):
        bound = observation_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        history = bound.arguments["state_history"]
        if not isinstance(history, DenseHistory):
            raise ValueError("The tactile adapter needs its per-control-frame DenseHistory")
        frame_idx = bound.arguments["frame_idx"]
        observation = original_observation(*args, **kwargs)
        observation.update(history.packet(frame_idx))
        if pim_episode_id is not None:
            observation.update(pim_episode_id=pim_episode_id, pim_attempt_id=pim_attempt_id)
        observation["observation/state"] = np.asarray(bound.arguments["env_state"], dtype=np.float32)
        for camera in ("cam_front", "cam_left", "cam_right"):
            observation[f"observation.images.{camera}"] = observation.pop(f"observation/{camera}_image")
        return observation

    def request_action_chunk(**kwargs):
        # The original loop catches Exception and executes hold actions. Continuing
        # after that would mislabel CTE history as the previously returned commands.
        try:
            result = original_request(**kwargs)
            actions = result[0]
            if len(actions) < 4 or not np.isfinite(actions).all():
                raise ValueError("The server must return at least four finite actions")
            return result
        except Exception as exc:
            raise SystemExit(
                f"Stopping the attempt after inference failure; restart with a fresh episode: {exc}"
            ) from exc

    client.StateHistoryBuffer = DenseHistory
    client.parse_args = parse_args
    client.build_pi0_observation = build_observation
    client.request_action_chunk = request_action_chunk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-source", type=Path, required=True, help="Path to ur7e_xhand_deploy_pi0_client.py")
    parser.add_argument("--pim-episode-id", default=None, help="Same scene ID across retries; change for new scenes")
    parser.add_argument("--pim-attempt-id", type=int, default=0, help="0 initially, then increment on each retry")
    parser.add_argument("client_args", nargs=argparse.REMAINDER, help="Original client arguments after --")
    args = parser.parse_args(argv)
    source = args.client_source.resolve()
    spec = importlib.util.spec_from_file_location("zeva_factile_robot_client", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load client source {source}")
    client = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = client
    sys.path.insert(0, str(source.parent))
    spec.loader.exec_module(client)
    adapt_factile_client(client, pim_episode_id=args.pim_episode_id, pim_attempt_id=args.pim_attempt_id)
    forwarded = args.client_args[1:] if args.client_args[:1] == ["--"] else args.client_args
    previous_argv = sys.argv
    try:
        sys.argv = [str(source), *forwarded]
        return int(client.main() or 0)
    finally:
        sys.argv = previous_argv
