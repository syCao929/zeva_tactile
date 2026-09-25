# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LeRobot-v2.1 adapter for the UR7e + XHand (TactileTTT) datasets.

Same on-disk conventions as :mod:`robocasa365_lerobot_dataset` — a directory of
independent LeRobot v2.1 roots laid out as ``<root>/<category>/<task>/<x>/lerobot``
— but a different embodiment: the source records joint-position commands for a
6-DoF UR7e arm and a 12-DoF XHand, not RoboCasa's end-effector deltas.

Two consequences worth stating plainly:

* **Camera mapping.** The source has ``cam_front`` / ``cam_left`` / ``cam_right``
  cameras. ``cam_right`` is wrist-mounted; front and left are external views.
* **fps.** The source runs at 15 Hz, so ``fps`` must be passed explicitly; the
  metadata check rejects a root whose ``info.json`` disagrees.

State layout of ``observation.state`` (1972-D) for this embodiment::

    [0:6]     arm_joint_0..5.pos
    [6:12]    arm_joint_0..5.vel
    [12:28]   arm_ee_pose.00..15
    [28:52]   hand_joint_0..11 pos/torque, interleaved (pos at even offsets)
    [52:1972] hand_tactile_sensor_* force / temperature channels
"""

from __future__ import annotations

import bisect
import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq
import torch
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_processing import ActionNormalizer, resolve_action_normalization
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import split_episode_ids
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.xhand_camera import (
    CAMERA_KEYS as _CAMERAS,
)
from cosmos_framework.data.generator.action.xhand_camera import (
    CAMERA_LAYOUTS as _CAMERA_LAYOUTS,
)
from cosmos_framework.data.generator.action.xhand_camera import (
    DEFAULT_CAMERA_LAYOUT,
    VIEW_DESCRIPTION,
    compose_xhand_views,
)

_ACTION_INDICES = {
    # 6 arm joints + 12 hand joints, all joint positions.
    "full18": tuple(range(18)),
    "arm6": tuple(range(6)),
}

# hand_joint_N.pos sits at even offsets from 28, torque at the odd one.
_HAND_POS = tuple(range(28, 52, 2))
_HAND_TORQUE = tuple(range(29, 52, 2))

_STATE_INDICES = {
    "arm6": tuple(range(6)),
    # 18 == exactly the `full18` action space: the arm + hand joint positions the policy
    # commands, in the same absolute parameterization it commands them in. State ==
    # "where the joints are now", action == "where they should go next"; in this dataset
    # `action[t]` tracks `state[t]` to within ~0.01 rad (cosine 0.99 arm / 0.96 hand).
    #
    # This is the shape the two reference implementations both use. pi0/pi0.5 hard-code
    # it -- `state_proj = nnx.Linear(config.action_dim, width)` (openpi pi0.py:159) --
    # so a policy there *cannot* be built with a state of any other width. Zeva's own
    # proprio (RoboCasa `arm9` = relative EEF pose 7 + gripper qpos 2) instead matches
    # its `arm7` action's DOF set (end-effector + gripper) in absolute form, which is the
    # same principle: state and action describe the same joints.
    "joint18": tuple(range(6)) + _HAND_POS,
    "arm34": tuple(range(6)) + tuple(range(12, 28)) + _HAND_POS,
    "full52": tuple(range(52)),
}


def _humanize_task(task: str) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", task).strip()
    return (words[:1].upper() + words[1:].lower() + ".") if words else task


@dataclass(frozen=True)
class XHandEpisode:
    root: Path
    category: str
    task_name: str
    episode_id: int
    length: int
    command: str
    parquet_path: Path
    video_paths: dict[str, Path]


class XHandLeRobotDataset(Dataset):
    """Cosmos policy windows over UR7e + XHand LeRobot-v2.1 roots."""

    EMBODIMENT_TYPE = "ur7e-xhand"

    def __init__(
        self,
        root: str,
        *,
        fps: float = 15.0,
        chunk_length: int = 32,
        split: str = "train",
        split_seed: int = 42,
        split_val_ratio: float = 0.03,
        mode: str = "wam",
        viewpoint: str = "concat_view",
        use_state: bool = False,
        use_tactile: bool = False,
        tactile_memory_steps: int = 30,
        use_image_augmentation: bool = False,
        emit_behavior_metadata: bool = False,
        task_names: Sequence[str] | None = None,
        camera_layout: str = DEFAULT_CAMERA_LAYOUT,
        action_mode: str = "full18",
        state_mode: str = "joint18",
        action_normalization: str | None = "minmax",
        action_stats_path: str | None = None,
        view_size: int = 256,
        parquet_cache_size: int = 8,
        max_roots: int | None = None,
    ) -> None:
        if viewpoint not in {"concat_view", "third_person_view"}:
            raise ValueError(f"Unsupported XHand viewpoint={viewpoint!r}")
        if split not in {"train", "val", "full"}:
            raise ValueError("split must be train, val or full")
        if camera_layout not in _CAMERA_LAYOUTS:
            raise ValueError(f"Unsupported XHand camera_layout={camera_layout!r}")
        if action_mode not in _ACTION_INDICES:
            raise ValueError(f"Unsupported XHand action_mode={action_mode!r}")
        if state_mode not in _STATE_INDICES:
            raise ValueError(f"Unsupported XHand state_mode={state_mode!r}")

        self.root = Path(root)
        self.fps = float(fps)
        self.chunk_length = int(chunk_length)
        self.split = split
        self.split_seed = int(split_seed)
        self.split_val_ratio = float(split_val_ratio)
        self.mode = mode
        self.viewpoint = viewpoint
        self.use_state = bool(use_state)
        # Keep the full raw state sequence separate from ``proprio``.  The
        # latter is normalized/padded by the policy input path; tactile values
        # must reach the dedicated encoder with their checkpoint normalization.
        self.use_tactile = bool(use_tactile)
        self.tactile_memory_steps = int(tactile_memory_steps)
        if self.tactile_memory_steps <= 0:
            raise ValueError("tactile_memory_steps must be positive")
        self.use_image_augmentation = bool(use_image_augmentation)
        self.emit_behavior_metadata = bool(emit_behavior_metadata)
        self.task_names = frozenset(task_names) if task_names is not None else None
        self.camera_layout = camera_layout
        self.action_mode = action_mode
        self.action_indices = _ACTION_INDICES[action_mode]
        self.state_mode = state_mode
        self.state_indices = _STATE_INDICES[state_mode]
        self.camera_names = _CAMERA_LAYOUTS[camera_layout]
        self.view_size = int(view_size)
        self.domain_id = get_domain_id(self.EMBODIMENT_TYPE)
        self.action_dim = len(self.action_indices)
        self._parquet_cache_size = int(parquet_cache_size)
        self._parquet_cache: OrderedDict[Path, dict[str, torch.Tensor]] = OrderedDict()
        self._normalizer: ActionNormalizer | None = None

        if action_normalization is not None:
            stats_path = Path(action_stats_path) if action_stats_path else None
            if stats_path is None or not stats_path.is_file():
                raise FileNotFoundError(
                    "XHand action normalization requires an action statistics JSON; "
                    f"missing {stats_path}. Run tools/convert_xhand_dataset.py to generate "
                    "'meta/feature_stats_compact.json', or pass action_normalization=None."
                )
            raw = json.loads(stats_path.read_text())
            raw = raw.get("action", raw)
            index = torch.tensor(self.action_indices, dtype=torch.long)
            missing = [k for k in ("min", "max") if k not in raw]
            if missing:
                raise KeyError(f"action stats {stats_path} lacks {missing}")
            stats = {k: torch.tensor(raw[k], dtype=torch.float32).index_select(0, index) for k in ("min", "max")}
            self._normalizer = resolve_action_normalization(action_normalization, stats)

        roots = sorted(path.parent.parent for path in self.root.glob("*/*/*/lerobot/meta/info.json"))
        roots = [path for path in roots if self._include_root(path)]
        if max_roots is not None:
            roots = roots[:max_roots]
        if not roots:
            raise FileNotFoundError(f"No XHand LeRobot-v2.1 roots under {self.root}")

        self.training_episode_ids: set[int] = set()
        self.episodes: list[XHandEpisode] = []
        for root_path in roots:
            self._index_root(root_path)

        # A length-H command needs H action rows and H+1 observation frames.
        self._window_lengths = [max(0, episode.length - self.chunk_length) for episode in self.episodes]
        self._cum_ends: list[int] = []
        total = 0
        for length in self._window_lengths:
            total += length
            self._cum_ends.append(total)
        self._num_windows = total

    def _include_root(self, path: Path) -> bool:
        _category, task_name = path.relative_to(self.root).parts[:2]
        return self.task_names is None or task_name in self.task_names

    def _index_root(self, path: Path) -> None:
        info = json.loads((path / "meta" / "info.json").read_text())
        if info["codebase_version"] != "v2.1":
            raise ValueError(f"Unexpected XHand metadata version {info['codebase_version']!r} in {path}")
        if float(info["fps"]) != self.fps:
            raise ValueError(f"XHand root {path} is {info['fps']} fps but dataset was built for {self.fps} fps")
        category, task_name = path.relative_to(self.root).parts[:2]
        records = [json.loads(line) for line in (path / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
        self.training_episode_ids.update(split_episode_ids(len(records), self.split_seed, self.split_val_ratio, "train"))
        for episode_id in sorted(split_episode_ids(len(records), self.split_seed, self.split_val_ratio, self.split)):
            record = records[episode_id]
            length = int(record["length"])
            if length <= self.chunk_length:
                continue
            command = str(record.get("tasks", [""])[0]).strip() or _humanize_task(task_name)
            video_paths = {
                name: path / f"videos/chunk-{episode_id // 1000:03d}/{key}/episode_{episode_id:06d}.mp4"
                for name, key in _CAMERAS.items()
                if name in self.camera_names
            }
            parquet_path = path / f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet"
            if not parquet_path.is_file() or any(not video.is_file() for video in video_paths.values()):
                raise FileNotFoundError(f"Incomplete XHand episode {path}#{episode_id}")
            self.episodes.append(
                XHandEpisode(path, category, task_name, episode_id, length, command, parquet_path, video_paths)
            )

    def __len__(self) -> int:
        return self._num_windows

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        start = 0
        for length in self._window_lengths:
            blocks.append((start, length))
            start += length
        return blocks

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self._cum_ends, index)
        start = 0 if episode_index == 0 else self._cum_ends[episode_index - 1]
        return episode_index, index - start

    def _load_low_dim(self, episode: XHandEpisode) -> dict[str, torch.Tensor]:
        cached = self._parquet_cache.pop(episode.parquet_path, None)
        if cached is None:
            table = pq.read_table(episode.parquet_path, columns=["action", "observation.state"])
            payload = table.to_pydict()
            cached = {
                "action": torch.tensor(payload["action"], dtype=torch.float32),
                "state": torch.tensor(payload["observation.state"], dtype=torch.float32),
            }
        self._parquet_cache[episode.parquet_path] = cached
        while len(self._parquet_cache) > self._parquet_cache_size:
            self._parquet_cache.popitem(last=False)
        return cached

    def _decode(self, path: Path, start: int) -> torch.Tensor:
        timestamps = [(start + offset) / self.fps for offset in range(self.chunk_length + 1)]
        return decode_video_frames(path, timestamps, tolerance_s=2e-4, backend="torchcodec")

    def _compose_video(self, episode: XHandEpisode, start: int) -> torch.Tensor:
        views = {name: self._decode(path, start) for name, path in episode.video_paths.items()}
        if self.use_image_augmentation:
            contrast, brightness = random.uniform(0.85, 1.15), random.uniform(0.85, 1.15)
            for name, frames in views.items():
                mean = frames.mean(dim=(-2, -1), keepdim=True)
                views[name] = ((frames - mean) * contrast + mean).mul(brightness).clamp(0, 1)
        if self.viewpoint == "third_person_view":
            return views["front"] if "front" in views else views["left"]
        return compose_xhand_views(views, view_size=self.view_size, layout=self.camera_layout)

    def get_action_normalizer(self, _sample: dict[str, Any] | None = None) -> ActionNormalizer | None:
        return self._normalizer

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, frame_offset = self._resolve_index(index)
        episode = self.episodes[episode_index]
        low_dim = self._load_low_dim(episode)
        action = low_dim["action"][frame_offset : frame_offset + self.chunk_length, self.action_indices]
        initial_state = low_dim["state"][frame_offset, self.state_indices] if self.use_state else None
        video = self._compose_video(episode, frame_offset)
        result: dict[str, Any] = {
            "ai_caption": episode.command,
            "video": (video * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3),
            "action": action,
            "conditioning_fps": torch.tensor(int(self.fps), dtype=torch.long),
            "mode": self.mode,
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "viewpoint": self.viewpoint,
            "additional_view_description": (
                VIEW_DESCRIPTION if self.camera_layout == DEFAULT_CAMERA_LAYOUT
                else "The left panel is external cam_left and the right panel is wrist-mounted cam_right."
            ),
            "task_cluster": episode.task_name,
            "task_category": episode.category,
        }
        if initial_state is not None:
            result["proprio"] = initial_state
        if self.use_tactile:
            # The tactile branch is causal: each policy window receives the
            # current frame and only the preceding 15 Hz frames.  At an
            # episode boundary we left-pad with zeros and expose a validity
            # mask so BIT can leave its recurrent state untouched during warmup.
            history_start = max(0, frame_offset - self.tactile_memory_steps + 1)
            history = low_dim["state"][history_start : frame_offset + 1].clone()
            left_pad = self.tactile_memory_steps - history.shape[0]
            if left_pad:
                history = torch.cat((torch.zeros((left_pad, *history.shape[1:]), dtype=history.dtype), history), dim=0)
            result["tactile_state"] = history
            result["tactile_valid"] = torch.cat(
                (
                    torch.zeros(left_pad, dtype=torch.bool),
                    torch.ones(self.tactile_memory_steps - left_pad, dtype=torch.bool),
                )
            )
        if self.emit_behavior_metadata:
            result["behavior_source_index"] = torch.tensor(episode_index, dtype=torch.long)
            result["behavior_episode_id"] = torch.tensor(episode.episode_id, dtype=torch.long)
            result["behavior_task_cluster"] = episode.task_name
            result["behavior_frame_offset"] = torch.tensor(frame_offset, dtype=torch.long)
        return result
