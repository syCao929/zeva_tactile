# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Supplies BIT and independent-demonstration PIM features for Zeva injection.

This is the wrapper the release ships without.  ``_attach_stage2_behavior``
(``omni_mot_model.py:714-716``) hard-``KeyError``s unless ``behavior_global``,
``behavior_phase``, ``behavior_effect`` and ``behavior_effect_valid`` are all in the
batch, and nothing under ``data/**`` produces them; the loader only emits the three
*index* fields (``xhand_lerobot_dataset.py:315-319``) that say where to look.

What each tensor is, per ``omni_mot_model.py:714-829``:

    behavior_global        [256]      opaque task-context conditioning vector
    behavior_phase         [128]      CTE phase at the chunk's start frame
    behavior_effect        [4,128]    last four completed CTE effect tokens
    behavior_effect_valid  [4]  bool  which of those slots are real

**Where the numbers come from.**  ``cte_features.py`` ran the trained CTE over every
boundary (one per 4 raw controls) of every episode and cached the phase and the
right-aligned effect history.  A training chunk starting at raw frame ``f`` is
described by exactly the state a robot would have had at that moment, i.e. the row
for boundary ``4 * (f // 4)``.

**Why the wrapper resolves indices itself instead of reading them off the sample.**
``ActionTransformPipeline`` owns the sample dict and may drop unknown keys, so
depending on the loader's ``emit_behavior_metadata`` fields surviving the transform
would be fragile.  ``ActionSFTDataset`` passes ``idx`` straight through to the inner
dataset, so re-deriving ``(episode, frame_offset)`` from the same ``idx`` is exact
and costs one arithmetic call.

**On ``behavior_global``.**  At inference this value is *retrieved*, not computed:
the server encodes the initial observation, runs the stage-3 head, looks the key up
in a task-context bank and feeds back the stored 256-d value
(``action_policy_server_robocasa365_zeva.py:706-719``).  Nothing in the model
supervises it (``_attach_stage2_behavior`` only projects it), so its *semantics* are
whatever the bank stores — the single hard constraint is that training must feed
values from the same space.  With one task cluster there is exactly one bank entry,
so this wrapper emits that same constant vector for every sample, derived
deterministically from the task name by :func:`task_context_vector`.
``build_task_context_bank.py`` uses the identical function, so train and inference
agree by construction.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.xhand_camera import require_camera_contract
from cosmos_framework.model.zeva.demonstration_memory import demonstration_memory

# One cached row per latent frame = per 4 raw controls (see cte_features).
RAW_PER_LATENT = 4
GLOBAL_DIM = 256


def task_context_vector(task_cluster: str, dim: int = GLOBAL_DIM) -> torch.Tensor:
    """Deterministic 256-d unit vector for a task cluster name.

    A hash rather than a learned embedding, deliberately: with a single task the
    stage-3 retrieval head cannot be trained contrastively (every sample is a
    positive of every other, the same degeneracy that killed the CTE's
    ``loss_task``), so a learned per-task embedding would have no gradient signal.
    A stable fingerprint keeps train and inference bit-identical and needs no
    training, while still exercising the bank/retrieval machinery end to end.

    Swap this out for a learned embedding once there is more than one task cluster.
    """
    digest = hashlib.sha256(task_cluster.encode("utf-8")).digest()
    # Repeat the 32-byte digest until there are `dim` bytes, then read one byte per
    # output element. (Slicing to `dim * 4` and casting uint8 -> float32 instead
    # would yield four times as many elements — the stage-2 projector expects 256.)
    raw = (digest * (dim // len(digest) + 1))[:dim]
    vec = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.float32)
    vec = (vec - 127.5) / 127.5
    return vec / vec.norm().clamp_min(1e-6)


class ZevaBehaviorWrapper(torch.utils.data.Dataset):
    """Wrap an ``ActionSFTDataset`` and attach the Zeva ``behavior_*`` tensors.

    ``feature_cache`` is a ``cte_features.py`` output directory.  Samples whose
    episode is missing from the cache raise immediately rather than silently
    training on zeros — a partially built cache is a configuration error, not
    something to paper over.
    """

    def __init__(self, sft, feature_cache: str | Path, latent_cache_size: int = 8, *,
                 pim_training: bool = False, pim_top_k: int = 4, pim_context_dropout: float = 0.2) -> None:
        self.sft = sft
        self.inner = sft._dataset  # the XHandLeRobotDataset behind the transform
        self.cache_dir = Path(feature_cache)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"no CTE feature cache at {self.cache_dir}; run cte_features first")
        manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        require_camera_contract(manifest, str(self.cache_dir))
        self.pim_training = pim_training
        self.pim_top_k = int(pim_top_k)
        self.pim_context_dropout = float(pim_context_dropout)
        if not 0 <= self.pim_context_dropout <= 1:
            raise ValueError("pim_context_dropout must be in [0,1]")
        self._pim_cache: OrderedDict = OrderedDict()
        self._tasks: dict[int, str] = {}
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._cache_size = int(latent_cache_size)
        self._files: dict[int, Path] = {}
        for f in sorted(self.cache_dir.glob("features_*.npz")):
            with np.load(f) as z:
                require_camera_contract(z, str(f))
                episode_id = int(z["episode_id"])
                if episode_id in self._files:
                    raise ValueError(f"Duplicate feature-cache episode ID: {episode_id}")
                self._files[episode_id] = f
                self._tasks[episode_id] = str(z["task_cluster"])
        self._support_ids: list[int] = []
        if pim_training:
            # Supplied by the raw dataset from the same per-root split, even when
            # this wrapper serves val queries. Validation trajectories never support PIM.
            self._support_ids = sorted(self.inner.training_episode_ids.intersection(self._files))
            for episode in self.inner.episodes:
                if not self.support_candidates(episode.episode_id, episode.task_name):
                    raise ValueError(f"No independent training demonstration for episode {episode.episode_id}")


    def support_candidates(self, query_episode: int, task: str) -> list[int]:
        return [key for key in self._support_ids if key != query_episode and self._tasks[key] == task]

    def _pim_features(self, episode, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = self.support_candidates(episode.episode_id, episode.task_name)
        training = self.inner.split == "train"
        support_id = candidates[int(torch.randint(len(candidates), ()).item())] if training else candidates[0]
        memory = self._pim_cache.pop(support_id, None)
        if memory is None:
            memory = demonstration_memory(self._episode_row(support_id), episode.task_name, top_k=self.pim_top_k)
        self._pim_cache[support_id] = memory
        while len(self._pim_cache) > self._cache_size:
            self._pim_cache.popitem(last=False)
        phases, effects, valid, _ = memory.query_tensors(phase, top_k=self.pim_top_k)
        if training and float(torch.rand(())) < self.pim_context_dropout:
            phases.zero_()
            effects.zero_()
            valid.zero_()
        return phases, effects, valid

    # ---------------------------------------------------------------- plumbing
    def __len__(self) -> int:
        return len(self.sft)

    def get_shuffle_blocks(self):
        return self.inner.get_shuffle_blocks()

    def _episode_row(self, episode_id: int) -> dict:
        cached = self._cache.pop(episode_id, None)
        if cached is None:
            path = self._files.get(episode_id)
            if path is None:
                raise KeyError(
                    f"episode {episode_id} is not in the CTE feature cache at {self.cache_dir}; "
                    "rebuild it with cte_features (it can resume)"
                )
            with np.load(path) as z:
                cached = {
                    "phase": torch.from_numpy(z["phase"]),  # [T_lat,128] fp16
                    "effect": torch.from_numpy(z["effect"]),  # [T_lat,4,128] fp16
                    "effect_valid": torch.from_numpy(z["effect_valid"]),  # [T_lat,4] bool
                }
            self._cache[episode_id] = cached
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache[episode_id] = cached
        return cached

    def __getitem__(self, idx: int) -> dict:
        sample = self.sft[idx]
        episode_index, frame_offset = self.inner._resolve_index(idx)
        episode = self.inner.episodes[episode_index]
        rows = self._episode_row(episode.episode_id)

        # The chunk starts at raw frame `frame_offset`; the most recent boundary the
        # robot would have observed is the largest multiple of 4 at or before it.
        row = int(frame_offset) // RAW_PER_LATENT
        n = rows["phase"].shape[0]
        if row >= n:
            raise IndexError(
                f"episode {episode.episode_id}: frame_offset {frame_offset} maps to cached row {row}, "
                f"but the cache only has {n} rows"
            )

        sample["behavior_global"] = task_context_vector(episode.task_name)
        sample["behavior_phase"] = rows["phase"][row].float()  # [128]
        sample["behavior_effect"] = rows["effect"][row].float()  # [4,128]
        sample["behavior_effect_valid"] = rows["effect_valid"][row]  # [4] bool
        if self.pim_training:
            phases, effects, valid = self._pim_features(episode, sample["behavior_phase"])
            sample["behavior_pim_phase"] = phases
            sample["behavior_pim_effect"] = effects
            sample["behavior_pim_valid"] = valid
        return sample
