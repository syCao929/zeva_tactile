# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build the task-context bank the inference server retrieves ``behavior_global`` from.

At serving time the value fed to ``behavior_global`` is **not** computed from the
observation.  ``action_policy_server_robocasa365_zeva.py:706-719`` encodes the initial
observation, runs the stage-3 head over the frozen policy's readout, retrieves from
this bank, and feeds back the stored 256-d ``behavior_value``.  The model itself never
supervises that vector (``_attach_stage2_behavior`` only projects it), so its
*semantics* are whatever the bank stores — the one hard invariant is that training
fed values from the same space.

That invariant is enforced here: this script imports :func:`task_context_vector` from
the training wrapper, so the bank holds exactly the vector ``ZevaBehaviorWrapper``
emitted for each task during stage-2 training.  A separate copy of the function would
be free to drift; sharing it makes train/serve agreement structural.

**Single-task data takes the simple path.**  With one task cluster there is nothing to
retrieve *between*, and the stage-3 contrastive head cannot be trained at all — every
sample is a positive of every other, the same degeneracy that made the CTE's
``loss_task`` a dead objective (see the README).  The server therefore runs with
``--task-context-instruction <cluster>`` and no ``--static-task-context-checkpoint``
(``action_policy_server_robocasa365_zeva.py:607-609`` requires exactly one of the two),
which averages the ``behavior_value`` of every entry whose ``instruction`` matches.
With one entry that is that entry.  ``retrieval_key`` is still written, so the file
also works in retrieval mode once there is more than one task.

Usage::

    PYTHONPATH=. python -m cosmos_framework.zeva_training.build_task_context_bank \\
        --feature-cache "$ZEVA_WORK/datasets/xhand_cte_features" \\
        --output        "$ZEVA_WORK/datasets/xhand_task_context_bank.pt" \\
        --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from cosmos_framework.data.generator.action.datasets.zeva_behavior_wrapper import (
    task_context_vector,
)

RETRIEVAL_DIM = 128


def retrieval_key(task_cluster: str) -> torch.Tensor:
    """Deterministic ``[128]`` unit key for a task cluster.

    Distinguishable per cluster and stable across runs, which is all retrieval needs.
    It is *not* a learned embedding: with a single task the stage-3 head has no
    contrastive signal, so there is nothing to learn. Replace this (and train the
    head) once multiple clusters exist.
    """
    import hashlib

    digest = hashlib.sha256(f"key::{task_cluster}".encode("utf-8")).digest()
    raw = (digest * (RETRIEVAL_DIM // len(digest) + 1))[:RETRIEVAL_DIM]
    vec = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.float32)
    vec = (vec - 127.5) / 127.5
    return vec / vec.norm().clamp_min(1e-6)


def collect_clusters(feature_cache: Path) -> list[str]:
    """Every task cluster present in a ``cte_features.py`` cache, sorted and unique."""
    manifest = feature_cache / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"no manifest at {manifest}; run cte_features first")
    payload = json.loads(manifest.read_text())
    clusters = sorted({str(e["task_cluster"]) for e in payload.get("episodes", [])})
    if not clusters:
        raise ValueError(f"{manifest} lists no episodes")
    return clusters


def build(feature_cache: Path) -> dict:
    clusters = collect_clusters(feature_cache)
    return {
        "entries": [
            {
                "retrieval_key": retrieval_key(name),
                "behavior_value": task_context_vector(name),
                "instruction": name,
            }
            for name in clusters
        ]
    }


def check(bank: dict, path: Path) -> int:
    """Re-load the way the server does and assert the train/serve invariant."""
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    entries = loaded["entries"]
    print(f"bank: {len(entries)} entries")
    for entry in entries:
        key, value, instruction = entry["retrieval_key"], entry["behavior_value"], entry["instruction"]
        expected = task_context_vector(instruction)
        same = torch.allclose(value.float(), expected, atol=1e-6)
        print(f"  {instruction!r:<24} key {tuple(key.shape)}  value {tuple(value.shape)}  "
              f"matches training wrapper: {same}")
        if not same:
            print("  ERROR: bank value differs from what stage-2 training fed the model", file=sys.stderr)
            return 1
        if tuple(key.shape) != (RETRIEVAL_DIM,):
            print(f"  ERROR: retrieval_key must be [{RETRIEVAL_DIM}]", file=sys.stderr)
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-cache", required=True, help="cte_features.py output directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--check", action="store_true", help="reload like the server and verify the invariant")
    args = ap.parse_args()

    bank = build(Path(args.feature_cache))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, out)
    print(f"wrote {len(bank['entries'])} entries to {out}  ({out.stat().st_size / 1e3:.1f} kB)")
    print("serve with: --task-context-bank <this file> "
          f"--task-context-instruction {bank['entries'][0]['instruction']!r}")
    return check(bank, out) if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
