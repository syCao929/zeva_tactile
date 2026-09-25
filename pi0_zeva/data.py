"""Current-observation XHand data for PyTorch pi0, independent of OpenPI/JAX.

State/action normalization is fitted on unique frames of training episodes only.
The split and ``length - horizon`` windows deliberately match the Cosmos XHand
baseline. Three OpenPI slots receive front/left external views and right wrist.
Heavy dependencies are imported only when their functionality is requested.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from pi0_zeva.camera import CAMERAS, require_camera_contract

STATE_INDICES = tuple(range(6)) + tuple(range(28, 52, 2))
JOINT_DIM = 18
PADDED_DIM = 32
FPS = 15


def index_feature_cache(directory: Path) -> dict[int, Path]:
    """Validate the declared three-view cache and every episode's camera version."""
    import numpy as np

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require_camera_contract(manifest, manifest_path)
    if any(
        manifest.get(key) != value
        for key, value in (
            ("phase_dim", 128),
            ("effect_dim", 128),
            ("effect_history", 4),
        )
    ):
        raise ValueError(f"CTE feature dimensions mismatch: {manifest_path}")
    files = {}
    for entry in manifest["episodes"]:
        path = directory / entry["file"]
        key = int(entry["episode_id"])
        if key in files:
            raise ValueError(f"Duplicate CTE episode {key} in {directory}")
        with np.load(path, allow_pickle=False) as data:
            require_camera_contract(data, path)
            if int(data["episode_id"]) != key:
                raise ValueError(f"CTE episode ID differs from manifest: {path}")
        files[key] = path
    if not files:
        raise ValueError(f"Empty CTE feature cache: {directory}")
    return files


@dataclass(frozen=True)
class Episode:
    root: Path
    episode_id: int
    length: int
    task_name: str
    prompt: str
    parquet_path: Path
    video_paths: dict[str, Path]


def _split_ids(count: int, seed: int, ratio: float, split: str) -> list[int]:
    # Exact algorithm of Cosmos split_episode_ids without importing its stack.
    import torch

    ids = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = int(round(count * ratio))
    return ids[n_val:] if split == "train" else ids[:n_val] if split == "val" else ids


def _index(root: str | Path, split: str, seed: int, ratio: float, horizon: int):
    if split not in {"train", "val", "full"}:
        raise ValueError("split must be train, val or full")
    if not 0 <= ratio < 1 or horizon <= 0:
        raise ValueError("split_val_ratio must be in [0, 1), horizon must be positive")
    root = Path(root).resolve()
    roots = (
        [root]
        if (root / "meta/info.json").is_file()
        else sorted(p.parent.parent for p in root.glob("*/*/*/lerobot/meta/info.json"))
    )
    if not roots:
        raise FileNotFoundError(f"No XHand LeRobot roots under {root}")
    episodes = []
    fingerprint = hashlib.sha256()
    for source in roots:
        relative = str(source.relative_to(root))
        info_bytes = (source / "meta/info.json").read_bytes()
        records_bytes = (source / "meta/episodes.jsonl").read_bytes()
        fingerprint.update(relative.encode() + b"\0" + info_bytes + records_bytes)
        info = json.loads(info_bytes)
        if info.get("codebase_version") != "v2.1" or float(info.get("fps", 0)) != FPS:
            raise ValueError(f"Expected LeRobot v2.1 at {FPS} Hz: {source}")
        features = info.get("features", {})
        if features.get("action", {}).get("shape") != [JOINT_DIM]:
            raise ValueError(f"Expected absolute XHand action18 metadata: {source}")
        if features.get("observation.state", {}).get("shape") != [1972]:
            raise ValueError(f"Expected raw XHand state1972 metadata: {source}")
        action_names = features["action"].get("names")
        state_names = features["observation.state"].get("names")
        if (
            action_names
            and state_names
            and action_names != [state_names[i] for i in STATE_INDICES]
        ):
            raise ValueError(f"State/action joint order disagrees: {source}")
        records = [
            json.loads(line) for line in records_bytes.splitlines() if line.strip()
        ]
        selected = set(_split_ids(len(records), seed, ratio, split))
        task = source.parents[1].name
        for position, record in enumerate(records):
            episode_id = int(record["episode_index"])
            if episode_id != position:
                raise ValueError(
                    "Episode IDs must be contiguous to match the Cosmos split"
                )
            chunk = episode_id // int(info.get("chunks_size", 1000))
            fmt = {"episode_chunk": chunk, "episode_index": episode_id}
            parquet = source / info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            ).format(**fmt)
            # Include backing-file identity without scanning all raw values during startup.
            stat = parquet.stat()
            fingerprint.update(
                f"{relative}/{episode_id}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            )
            length = int(record["length"])
            if position not in selected or length <= horizon:
                continue
            fallback = re.sub(r"(?<!^)(?=[A-Z])", " ", task).strip()
            fallback = fallback[:1].upper() + fallback[1:].lower() + "."
            tasks = record.get("tasks") or [""]
            prompt = str(tasks[0]).strip() or fallback
            video_template = info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
            videos = {
                slot: source / video_template.format(**fmt, video_key=key)
                for slot, key in CAMERAS.items()
            }
            episodes.append(
                Episode(source, episode_id, length, task, prompt, parquet, videos)
            )
    provenance = {
        "root": str(root),
        "dataset_fingerprint": fingerprint.hexdigest(),
        "split_seed": seed,
        "split_val_ratio": ratio,
        "horizon": horizon,
        "fps": FPS,
        "state_indices": list(STATE_INDICES),
        "action_indices": list(range(JOINT_DIM)),
        "action_representation": "absolute_joint_position",
        "frame_weighting": "unique_frames_of_training_episodes",
    }
    return episodes, provenance


def _read_episode(episode: Episode):
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(episode.parquet_path, columns=["action", "observation.state"])
    if table.num_rows != episode.length:
        raise ValueError(f"Invalid state/action shapes in {episode.parquet_path}")
    values = []
    for key, width in (("observation.state", 1972), ("action", JOINT_DIM)):
        column = table[key].combine_chunks()
        lengths = pc.list_value_length(column).to_numpy(zero_copy_only=False)
        if column.null_count or not np.all(lengths == width):
            raise ValueError(f"Invalid {key} row width in {episode.parquet_path}")
        values.append(
            pc.list_flatten(column)
            .to_numpy(zero_copy_only=False)
            .reshape(episode.length, width)
            .astype(np.float32, copy=True)
        )
    state, actions = values
    if not np.isfinite(actions).all() or not np.isfinite(state).all():
        raise ValueError(f"Nonfinite state/action values in {episode.parquet_path}")
    return state, actions


class Normalizer:
    """Z-score the 18 real joints; padding is handled by the dataset/model input."""

    def __init__(self, stats: dict[str, Any]):
        if stats.get("version") != 1 or stats.get("normalization") != "zscore":
            raise ValueError("Expected pi0 XHand z-score statistics version 1")
        for key in ("state", "action"):
            mean, std = stats[key]["mean"], stats[key]["std"]
            if len(mean) != JOINT_DIM or len(std) != JOINT_DIM:
                raise ValueError(
                    f"{key} statistics must have 18 dimensions before padding"
                )
            if not all(math.isfinite(v) for v in mean + std) or any(
                v <= 0 for v in std
            ):
                raise ValueError(f"Invalid {key} normalization statistics")
        self.stats = stats
        self.provenance = stats["provenance"]

    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        return cls(json.loads(Path(path).read_text()))

    def _transform(self, value, key: str, inverse: bool = False):
        import torch

        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32)
        if value.ndim < 1 or value.shape[-1] not in (
            {JOINT_DIM, PADDED_DIM} if inverse else {JOINT_DIM}
        ):
            raise ValueError(
                "Normalize exactly 18 joints before padding; inverse accepts 18 or 32"
            )
        if not value.is_floating_point():
            value = value.float()
        value = value[..., :JOINT_DIM]
        mean = value.new_tensor(self.stats[key]["mean"])
        std = value.new_tensor(self.stats[key]["std"])
        return value * std + mean if inverse else (value - mean) / std

    def normalize_state(self, value):
        return self._transform(value, "state")

    def normalize_actions(self, value):
        return self._transform(value, "action")

    def denormalize_actions(self, value):
        return self._transform(value, "action", inverse=True)


def compute_normalization(
    root, output_path, split_seed=42, split_val_ratio=0.03, horizon=32
):
    """Stream training episodes once; validation frames never enter the moments."""
    import numpy as np

    episodes, provenance = _index(root, "train", split_seed, split_val_ratio, horizon)
    if not episodes:
        raise ValueError(
            "No training episodes are long enough for the requested horizon"
        )
    count = 0
    means = {key: np.zeros(JOINT_DIM, dtype=np.float64) for key in ("state", "action")}
    m2 = {key: np.zeros(JOINT_DIM, dtype=np.float64) for key in means}
    for episode in episodes:
        state, actions = _read_episode(episode)
        n = len(actions)
        for key, values in (("state", state[:, STATE_INDICES]), ("action", actions)):
            values = values.astype(np.float64)
            batch_mean = values.mean(axis=0)
            delta = batch_mean - means[key]
            m2[key] += ((values - batch_mean) ** 2).sum(
                axis=0
            ) + delta**2 * count * n / (count + n)
            means[key] += delta * n / (count + n)
        count += n
    provenance["training_episodes"] = [
        {
            "root": str(e.root.relative_to(Path(root).resolve())),
            "episode_id": e.episode_id,
        }
        for e in episodes
    ]
    stats = {
        "version": 1,
        "normalization": "zscore",
        "count": count,
        "provenance": provenance,
    }
    for key in means:
        stats[key] = {
            "mean": means[key].tolist(),
            "std": np.maximum(np.sqrt(m2[key] / count), 1e-6).tolist(),
        }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(stats, indent=2) + "\n")
    temporary.replace(output_path)
    return stats


def _decode_current(path: Path, frame: int, image_size: int):
    import av
    import torch
    import torch.nn.functional as F

    if frame < 0:
        raise ValueError("frame index must be nonnegative")
    target = Fraction(frame, FPS)
    tolerance = Fraction(1, 5000)  # Same 2e-4 seconds as the Cosmos decoder.
    image = None
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        start = stream.start_time or 0
        time_base = Fraction(stream.time_base)
        container.seek(
            start + int(target / time_base),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        # Decode forward from the preceding keyframe, select only the requested
        # presentation timestamp, and stop immediately. Codec reference frames
        # never become additional policy observations.
        for decoded in container.decode(stream):
            if decoded.pts is None:
                continue
            timestamp = Fraction(decoded.pts - start) * time_base
            if timestamp < target - tolerance:
                continue
            if abs(timestamp - target) <= tolerance:
                image = (
                    torch.from_numpy(decoded.to_ndarray(format="rgb24"))
                    .permute(2, 0, 1)[None]
                    .float()
                    / 255.0
                )
            break
    if image is None:
        raise ValueError(
            f"No video frame within 2e-4 seconds of frame {frame} at {FPS} Hz: {path}"
        )
    return (
        F.interpolate(
            image, size=(image_size, image_size), mode="bilinear", align_corners=False
        )[0].clamp(0, 1)
        * 2
        - 1
    )


def _task_context(task: str):
    # Bit-identical to Cosmos ZevaBehaviorWrapper.task_context_vector.
    import torch

    digest = hashlib.sha256(task.encode()).digest()
    raw = (digest * (256 // len(digest) + 1))[:256]
    value = (
        torch.frombuffer(bytearray(raw), dtype=torch.uint8).float() - 127.5
    ) / 127.5
    return value / value.norm().clamp_min(1e-6)


class XHandPi0Dataset:
    """Map-style dataset with current images and causal optional Zeva features."""

    def __init__(
        self,
        root,
        stats_path,
        split="train",
        horizon=32,
        feature_cache=None,
        tactile=False,
        tactile_memory_steps=30,
        split_seed=42,
        split_val_ratio=0.03,
        image_size=224,
    ):
        if image_size <= 0 or tactile_memory_steps <= 0:
            raise ValueError("image_size and tactile_memory_steps must be positive")
        self.episodes, provenance = _index(
            root, split, split_seed, split_val_ratio, horizon
        )
        self.normalizer = Normalizer.load(stats_path)
        for key, value in provenance.items():
            if self.normalizer.provenance.get(key) != value:
                raise ValueError(
                    f"Normalization provenance mismatch for {key}; recompute training statistics"
                )
        self.horizon = horizon
        self.image_size = image_size
        self.tactile = tactile
        self.tactile_memory_steps = tactile_memory_steps
        self._ends = []
        total = 0
        for episode in self.episodes:
            total += episode.length - horizon
            self._ends.append(total)
            for path in episode.video_paths.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
        self._parquet_cache = OrderedDict()
        self._feature_cache = OrderedDict()
        self._feature_files = None
        if feature_cache is not None:
            self._index_features(Path(feature_cache))

    def _index_features(self, directory: Path):
        if len({e.root for e in self.episodes}) > 1:
            raise ValueError(
                "Legacy CTE caches lack root IDs; use one source root to avoid episode collisions"
            )
        if not directory.is_dir():
            raise FileNotFoundError(f"No CTE feature cache at {directory}")
        self._feature_files = index_feature_cache(directory)
        missing = {e.episode_id for e in self.episodes} - self._feature_files.keys()
        if missing:
            raise ValueError(f"CTE cache missing episodes: {sorted(missing)}")

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def _resolve_index(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self._ends, index)
        return episode, index - (self._ends[episode - 1] if episode else 0)

    def get_shuffle_blocks(self):
        return [
            (end - e.length + self.horizon, e.length - self.horizon)
            for e, end in zip(self.episodes, self._ends)
        ]

    def _behavior(self, episode: Episode, frame: int):
        import numpy as np
        import torch

        key = episode.episode_id
        cached = self._feature_cache.pop(key, None)
        if cached is None:
            with np.load(self._feature_files[key], allow_pickle=False) as data:
                cached = {
                    name: data[name].copy()
                    for name in ("phase", "effect", "effect_valid", "boundary_frame")
                }
                if (
                    "task_cluster" in data
                    and str(data["task_cluster"].item()) != episode.task_name
                ):
                    raise ValueError(f"CTE task mismatch for episode {key}")
            boundary = cached["boundary_frame"]
            n = len(boundary)
            if not n or not np.array_equal(boundary, np.arange(n) * 4):
                raise ValueError(
                    f"CTE boundaries must be contiguous four-frame intervals: episode {key}"
                )
            expected = {
                "phase": (n, 128),
                "effect": (n, 4, 128),
                "effect_valid": (n, 4),
            }
            for name, shape in expected.items():
                if cached[name].shape != shape or not np.isfinite(cached[name]).all():
                    raise ValueError(f"Invalid CTE {name} in episode {key}")
        self._feature_cache[key] = cached
        while len(self._feature_cache) > 8:
            self._feature_cache.popitem(last=False)
        row = frame // 4
        if row >= len(cached["boundary_frame"]):
            raise ValueError(f"Incomplete CTE cache for episode {key}, frame {frame}")
        return {
            "behavior_global": _task_context(episode.task_name),
            "behavior_phase": torch.from_numpy(cached["phase"][row].copy()).float(),
            "behavior_effect": torch.from_numpy(cached["effect"][row].copy()).float(),
            "behavior_effect_valid": torch.from_numpy(
                cached["effect_valid"][row].copy()
            ).bool(),
        }

    def __getitem__(self, index):
        import torch
        import torch.nn.functional as F

        episode_index, frame = self._resolve_index(index)
        episode = self.episodes[episode_index]
        cached = self._parquet_cache.pop(episode.parquet_path, None)
        if cached is None:
            cached = tuple(torch.from_numpy(value) for value in _read_episode(episode))
        self._parquet_cache[episode.parquet_path] = cached
        while len(self._parquet_cache) > 2:
            self._parquet_cache.popitem(last=False)
        state, actions = cached
        images = {
            slot: _decode_current(path, frame, self.image_size)
            for slot, path in episode.video_paths.items()
        }
        result = {
            "images": images,
            "image_masks": {slot: slot in CAMERAS for slot in images},
            "state": F.pad(
                self.normalizer.normalize_state(state[frame, STATE_INDICES]),
                (0, PADDED_DIM - JOINT_DIM),
            ),
            "actions": F.pad(
                self.normalizer.normalize_actions(
                    actions[frame : frame + self.horizon]
                ),
                (0, PADDED_DIM - JOINT_DIM),
            ),
            "prompt": episode.prompt,
            "episode_id": episode.episode_id,
            "episode_root": str(episode.root),
            "frame_index": frame,
        }
        if self._feature_files is not None:
            result.update(self._behavior(episode, frame))
        if self.tactile:
            start = max(0, frame - self.tactile_memory_steps + 1)
            history = state[start : frame + 1].clone()
            pad = self.tactile_memory_steps - len(history)
            result["tactile_state"] = F.pad(history, (0, 0, pad, 0))
            result["tactile_valid"] = torch.arange(self.tactile_memory_steps) >= pad
        return result
