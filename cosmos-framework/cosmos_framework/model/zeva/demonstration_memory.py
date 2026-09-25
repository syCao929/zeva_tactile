"""Build PIM from a completed demonstration's cached CTE features.

The demonstration is support context, not a claimed previous failed attempt.
Training must exclude the query trajectory and every held-out trajectory.
"""

from __future__ import annotations

import torch

from cosmos_framework.model.zeva.persistent_interaction_memory import (
    PersistentInteractionMemory,
    PersistentInteractionMemoryConfig,
)


def demonstration_memory(rows, task: str, *, top_k: int = 4) -> PersistentInteractionMemory:
    memory = PersistentInteractionMemory(PersistentInteractionMemoryConfig(top_k=top_k))
    memory.reset_episode(task)
    phase = torch.as_tensor(rows["phase"]).float()
    effect = torch.as_tensor(rows["effect"]).float()
    valid = torch.as_tensor(rows["effect_valid"]).bool()
    # Every fourth latent boundary ends a disjoint 16-control effect window.
    # Incomplete windows and repeatedly cached copies are never written.
    for boundary in range(4, phase.shape[0], 4):
        if bool(valid[boundary, -1]):
            memory.append_completed(
                task_cluster=task,
                phase=phase[boundary],
                effect=effect[boundary, -1],
                attempt_id=0,
                metadata={"source": "demonstration", "boundary": boundary},
            )
    if not len(memory):
        raise ValueError("PIM demonstration has no completed effect windows")
    return memory
