# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CTE training windows over the cached Wan latents.

Produces exactly what ``CausalTransitionEncoder.forward`` and
``causal_transition_encoder_loss`` consume (see ``cte_losses.py:110``):

    frames              [T, C, 30, 52]   float32   (C=48, from the VAE cache)
    transition_actions  [T-1, 4, A]      float32   (raw joint positions, A=18)
    valid_mask          [T]              bool
    transition_valid    [T-1, 4]         bool
    semantic_ids        scalar           int64     (drives the task-clustering loss)

The cadence is the crux. The cache stores one latent per 4 raw controls, so
latent ``i`` sits at raw frame ``4i`` and the transition between latents ``i``
and ``i+1`` is exactly raw actions ``[4i, 4i+4)`` — the "four-action transition"
the encoder's docstring describes. Consecutive transitions then group into effect
windows of ``effect_window_transitions`` (4), i.e. 16 raw controls per effect,
which is what makes an effect observable as a visual change
(``causal_transition_encoder.py:30-31``).

Why the default window is 17 latent frames: the model carries a 4-slot effect
history (``behavior_stage2.effect_history_length = 4``), and
``effect_windows = (T-1) // effect_window_transitions``. So T-1 must be at least
16 for a window to fill all four slots. RoboCasa trained at 9 latents (2 slots),
which is also valid — the model masks unused slots — so this is exposed as
``window_latents`` rather than hardcoded.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# One latent frame per 4 raw control steps (see vae_cache / probe_vae).
RAW_PER_LATENT = 4
# Keep in sync with CausalTransitionEncoderConfig.effect_window_transitions.
ACTION_PER_TRANSITION = 4


class CTECacheWindowDataset(Dataset):
    """Sliding windows over the per-episode latent cache written by ``vae_cache``."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        window_latents: int = 17,
        split: str = "train",
        val_ratio: float = 0.05,
        split_seed: int = 42,
        latent_cache_size: int = 8,
        episodes: list[int] | None = None,
    ) -> None:
        if window_latents < 2:
            raise ValueError("window_latents must be >= 2 (the encoder needs a completed transition)")
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"no cache dir at {self.cache_dir}; run vae_cache first")
        self.window_latents = int(window_latents)
        self._latent_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._latent_cache_size = int(latent_cache_size)

        # Build the episode list from the DIRECTORY, not from manifest.json.
        # The manifest is only rewritten when vae_cache finishes a whole run, so
        # an interrupted encode (or a `kill` mid-run) leaves it stale — listing
        # fewer episodes than are actually on disk. Scanning is cheap: the
        # per-episode scalars live inside each npz and npz loads arrays lazily.
        manifest_path = self.cache_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        entries = []
        for f in sorted(self.cache_dir.glob("episode_*.npz")):
            with np.load(f) as z:
                entries.append(
                    {
                        "file": f.name,
                        "episode_id": int(z["episode_id"]),
                        "task_cluster": str(z["task_cluster"]),
                        "raw_frames": int(z["length"]),
                        "latent_frames": int(z["latents"].shape[1]),
                    }
                )
        if not entries:
            raise FileNotFoundError(f"no episode_*.npz under {self.cache_dir}; run vae_cache first")
        entries.sort(key=lambda e: e["episode_id"])
        if episodes is not None:
            keep = set(episodes)
            entries = [e for e in entries if e["episode_id"] in keep]

        # Deterministic episode-level split: windows from one episode never span
        # train and val, which would otherwise leak.
        order = np.random.default_rng(split_seed).permutation(len(entries))
        n_val = max(1, int(round(len(entries) * val_ratio))) if val_ratio > 0 else 0
        if split == "val":
            chosen = [entries[i] for i in sorted(order[:n_val])]
        elif split == "train":
            chosen = [entries[i] for i in sorted(order[n_val:])]
        elif split == "full":
            chosen = list(entries)
        else:
            raise ValueError(f"split must be train/val/full, got {split!r}")

        # One entry per window: (episode file, start latent index)
        self.windows: list[tuple[str, int]] = []
        self.episodes: list[dict] = []
        for e in chosen:
            length = int(e["raw_frames"])
            t_lat = int(e["latent_frames"])
            # Two bounds have to hold for a window at start s (T = window_latents):
            #   latents:   s + T <= t_lat                       -> s <= t_lat - T
            #   actions:   4(s + T - 1) <= length               -> s <= length//4 - T + 1
            # Both are needed, and for `length % 4 == 0` episodes the action bound
            # is the LARGER one (t_lat == length//4 there), so using it alone asks
            # for latents that do not exist and `__getitem__` raises. 26 of the 101
            # episodes have length % 4 == 0, which is why this only blew up at eval
            # time — the training split happened to miss them.
            max_start = min(
                t_lat - self.window_latents,
                length // RAW_PER_LATENT - self.window_latents + 1,
            )
            starts = range(0, max_start + 1)
            if max_start < 0:
                continue
            self.episodes.append(e)
            self.windows.extend((e["file"], s) for s in starts)

        self._task_ids = {
            name: i for i, name in enumerate(sorted({x["task_cluster"] for x in entries}))
        }
        if not self.windows:
            raise RuntimeError(
                f"no windows: {len(chosen)} episodes, window_latents={self.window_latents}; "
                "the episodes may be shorter than one window"
            )

    # ------------------------------------------------------------------ loading
    def _latents(self, filename: str) -> np.ndarray:
        cached = self._latent_cache.pop(filename, None)
        if cached is None:
            with np.load(self.cache_dir / filename) as z:
                cached = z["latents"]  # [C,T_lat,30,52] fp16
            self._latent_cache[filename] = cached
            while len(self._latent_cache) > self._latent_cache_size:
                self._latent_cache.popitem(last=False)
        else:
            self._latent_cache[filename] = cached
        return cached

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        filename, start = self.windows[index]
        with np.load(self.cache_dir / filename) as z:
            latents = torch.from_numpy(z["latents"]).float()  # [C,T_lat,30,52]
            actions = torch.from_numpy(z["actions"]).float()  # [N_raw,A]
            task_cluster = str(z["task_cluster"])

        t = self.window_latents
        frames = latents[:, start : start + t]  # [C,T,30,52]
        if frames.shape[1] != t:
            # Report the total, not the slice length: the slice length is 0 here
            # whenever start is past the end, which reads as "the episode is tiny".
            raise RuntimeError(
                f"{filename}: window [{start},{start+t}) does not fit in {latents.shape[1]} latents"
            )

        # Transition j (between latent start+j and start+j+1) is the raw action
        # block [4*(start+j), 4*(start+j)+4).
        base = start * RAW_PER_LATENT
        blocks = actions[base : base + (t - 1) * ACTION_PER_TRANSITION]
        transitions = blocks.reshape(t - 1, ACTION_PER_TRANSITION, -1)

        return {
            # forward() wants [B,T,C,H,W]; add the batch dim later in the collate.
            "frames": frames.permute(1, 0, 2, 3).contiguous(),  # [T,C,30,52]
            "transition_actions": transitions.contiguous(),  # [T-1,4,A]
            "valid_mask": torch.ones(t, dtype=torch.bool),
            "transition_valid": torch.ones(t - 1, ACTION_PER_TRANSITION, dtype=torch.bool),
            "semantic_id": torch.tensor(self._task_ids[task_cluster], dtype=torch.long),
            "task_cluster": task_cluster,
        }


def collate_cte(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack windows into a CTE batch (``[B,...]`` tensors plus ``semantic_ids [B]``)."""
    return {
        "frames": torch.stack([b["frames"] for b in batch]),  # [B,T,C,H,W]
        "transition_actions": torch.stack([b["transition_actions"] for b in batch]),  # [B,T-1,4,A]
        "valid_mask": torch.stack([b["valid_mask"] for b in batch]),  # [B,T]
        "transition_valid": torch.stack([b["transition_valid"] for b in batch]),  # [B,T-1,4]
        "semantic_ids": torch.stack([b["semantic_id"] for b in batch]),  # [B]
    }
