"""XHand PIM lifecycle; explicit scene IDs keep memory across retries safely."""

from __future__ import annotations

import numpy as np

from cosmos_framework.data.generator.action.xhand_camera import require_camera_contract
from cosmos_framework.model.zeva.demonstration_memory import demonstration_memory
from cosmos_framework.model.zeva.persistent_interaction_memory import (
    PersistentInteractionMemory,
    PersistentInteractionMemoryConfig,
)


class XHandPIMContext:
    def __init__(self, *, top_k: int = 4, demonstration: str | None = None) -> None:
        self.top_k = top_k
        self.demo = None
        if demonstration:
            with np.load(demonstration) as data:
                require_camera_contract(data, demonstration)
                self.demo = {name: data[name].copy() for name in ("phase", "effect", "effect_valid")}
        self.memory = None
        self.episode_id = None
        self.task = None
        self.attempt_id = 0
        self.boundary = 0

    def prepare(self, observation: dict, *, reset: bool) -> bool:
        """Return whether the current attempt's CTE/BIT state must be reset.

        Without explicit PIM IDs, a client reset always starts a new scene.
        With IDs, a retry increments pim_attempt_id while preserving pim_episode_id.
        Tactile episode_id may change per attempt and is intentionally independent.
        """
        explicit = "pim_episode_id" in observation
        if explicit != ("pim_attempt_id" in observation):
            raise ValueError("Provide pim_episode_id and pim_attempt_id together")
        episode_id = str(observation["pim_episode_id"]) if explicit else None
        attempt = int(observation["pim_attempt_id"]) if explicit else 0
        task = str(observation["prompt"])
        new_scene = self.memory is None or episode_id != self.episode_id or (reset and not explicit)
        if new_scene:
            if attempt != 0:
                raise ValueError("A new PIM scene must start at pim_attempt_id=0")
            self.memory = (
                demonstration_memory(self.demo, task, top_k=self.top_k)
                if self.demo is not None
                else PersistentInteractionMemory(PersistentInteractionMemoryConfig(top_k=self.top_k))
            )
            if self.demo is None:
                self.memory.reset_episode(task)
            self.episode_id, self.task, self.attempt_id = episode_id, task, 0
            self.boundary = 0
            return True
        if task != self.task:
            raise ValueError("Changed prompt requires a new pim_episode_id")
        if attempt != self.attempt_id:
            self.memory.begin_attempt(attempt)
            self.attempt_id, self.boundary = attempt, 0
            return True
        if reset and self.boundary:
            raise ValueError("A retry must increment pim_attempt_id; a new scene must change pim_episode_id")
        return reset

    def observe_and_query(self, phase, effect, effect_valid):
        # At boundaries 4,8,..., the latest effect covers a fresh 16-control block.
        if self.boundary > 0 and self.boundary % 4 == 0 and bool(effect_valid[-1]):
            self.memory.append_completed(
                task_cluster=self.task,
                phase=phase,
                effect=effect[-1],
                attempt_id=self.attempt_id,
                metadata={"source": "online", "boundary": self.boundary},
            )
        self.boundary += 1
        phases, effects, valid, _ = self.memory.query_tensors(phase, top_k=self.top_k)
        return phases, effects, valid
