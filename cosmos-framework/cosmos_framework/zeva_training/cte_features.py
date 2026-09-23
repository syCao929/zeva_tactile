# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run the trained CTE over every boundary of every episode and cache its outputs.

This is the bridge between stage 1 (CTE) and stage 2 (policy injection).  The
dataset loader already emits the three index fields that identify *where* in an
episode a training chunk starts (``behavior_episode_id`` / ``behavior_task_cluster``
/ ``behavior_frame_offset``, ``xhand_lerobot_dataset.py:315-319``), but nothing in
the release turns those indices into the four tensors ``_attach_stage2_behavior``
demands.  This module produces the CTE half of that lookup table; the wrapper in
``zeva_behavior_wrapper.py`` does the lookup.

**What is cached, and why exactly this**

At inference the server (``action_policy_server_robocasa365_zeva.py:735-767``) runs
the CTE over the boundary frames the client sends and takes two things:

    phase      encoded["phase"][:, -1]                    [128]   current phase
    effect     last 4 completed effect_post, right-aligned [4,128] + [4] bool

So the cache stores exactly those, one row per boundary frame.

**The window rule (the part that must match the server bit for bit)**

The CTE is a GRU stack with no positional embedding, so its state accumulates from
the first frame it is given and ``phase[:, -1]`` therefore depends on *how many*
frames preceded it.  Training always used ``window_latents=17``, so the cache and
the server must both use a 17-latent window ending at the boundary in question.
For the first 16 boundaries of an episode that window does not exist yet; rather
than pad (which the CTE never saw in training, and which a GRU would treat as a
real zero-observation), the window simply starts at the episode's first frame and
is shorter.  ``_BoundaryBuffer`` in ``action_policy_server_xhand.py`` implements the
same rule client-side, so deployment reproduces these numbers.

**Cadence**

One cached row per *latent* frame, i.e. per 4 raw controls — the same cadence the
server sees, since ``cte_boundary_images`` holds observations at raw offsets
0,4,8,... .  A training chunk starting at raw frame ``f`` therefore reads the row
for ``4 * (f // 4)``.

Usage::

    PYTHONPATH=. python -m cosmos_framework.zeva_training.cte_features \\
        --cte-checkpoint "$ZEVA_WORK/runs/zeva_cte/cte-v2-20260921/cte_step_003000.pt" \\
        --latent-cache   "$ZEVA_WORK/datasets/xhand_cte_cache" \\
        --output         "$ZEVA_WORK/datasets/xhand_cte_features"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

_CACHE_VERSION = 1
RAW_PER_LATENT = 4


def load_cte(checkpoint: Path, device: str):
    """Load a CTE checkpoint exactly the way the inference server does.

    Mirrors ``action_policy_server_robocasa365_zeva.py:667-679``: ``weights_only=False``,
    ``normalize_cte_state_dict``, ``use_mamba`` inferred from the key names, then a
    strict ``load_state_dict``.  Keeping this in one place means a checkpoint that
    the cache builder accepts is a checkpoint the server accepts.
    """
    from cosmos_framework.model.zeva import CausalTransitionEncoder, CausalTransitionEncoderConfig
    from cosmos_framework.model.zeva.checkpoint_io import normalize_cte_state_dict

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(payload["model_config"])
    state = normalize_cte_state_dict(payload["model"])
    cfg["use_mamba"] = any("mamba" in key for key in state)
    model = CausalTransitionEncoder(CausalTransitionEncoderConfig(**cfg))
    model.load_state_dict(state)  # strict
    model = model.to(device).eval()
    model.float()
    return model, model.cfg


@torch.no_grad()
def encode_window(model, cfg, latents: torch.Tensor, actions: torch.Tensor, start: int, length: int):
    """One CTE forward over latents ``[start, start+length)`` -> (phase[128], effect[4,128], valid[4]).

    ``latents`` is the episode's full ``[C,T,30,52]`` cache and ``actions`` its
    ``[N_raw,A]`` raw joint positions.  ``length`` must be passed explicitly: the
    window ends at ``start + length - 1``, NOT at the end of the episode.
    """
    win = latents[:, start : start + length]  # [C,t,30,52]
    t = win.shape[1]
    if t != length:
        raise RuntimeError(f"window [{start},{start + length}) runs past the episode ({latents.shape[1]} latents)")
    device = latents.device
    # Transition j (between latent start+j and start+j+1) is raw block [4*(start+j), +4).
    base = start * RAW_PER_LATENT
    if t > 1:
        block = actions[base : base + (t - 1) * RAW_PER_LATENT]
        transitions = block.reshape(t - 1, RAW_PER_LATENT, -1).unsqueeze(0)
    else:
        transitions = torch.zeros((1, 0, RAW_PER_LATENT, cfg.action_dim), dtype=torch.float32, device=device)
    frames = win.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1,t,C,H,W]
    valid = torch.ones((1, t), dtype=torch.bool, device=device)
    transition_valid = torch.ones(transitions.shape[:-1], dtype=torch.bool, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(frames, transitions, valid, transition_valid)

    phase = out["phase"][0, -1].float().cpu()  # [128]
    # Right-align the completed effect tokens into the 4 history slots, exactly as
    # the server does (`action_policy_server_robocasa365_zeva.py:757-762`).
    completed = out["effect_post"][0][out["effect_complete"][0]].float().cpu()  # [n,128]
    effect = torch.zeros((4, cfg.effect_dim), dtype=torch.float32)
    effect_valid = torch.zeros(4, dtype=torch.bool)
    take = min(4, completed.shape[0])
    if take:
        effect[-take:] = completed[-take:]
        effect_valid[-take:] = True
    return phase, effect, effect_valid


@torch.no_grad()
def encode_episode(model, cfg, latents: torch.Tensor, actions: torch.Tensor, window_latents: int):
    """Features for every boundary (latent index) of one episode.

    Boundaries with fewer than ``window_latents`` preceding latents are grouped by
    their exact window length and encoded together — batching across different
    lengths would require padding, and a GRU would consume that padding as real
    input, silently shifting every feature in the batch.
    """
    device = next(model.parameters()).device
    t_lat = latents.shape[1]
    latents = latents.to(device, dtype=torch.float32)
    actions = actions.to(device, dtype=torch.float32)

    phases, effects, valids = [], [], []
    for length in range(1, min(window_latents, t_lat) + 1):
        # Window of exactly `length` latents ending at each k >= length-1.
        ks = [k for k in range(t_lat) if min(k + 1, window_latents) == length]
        if not ks:
            continue
        # All these windows share a length, so they may share a batch.
        rows = [encode_window(model, cfg, latents, actions, k - length + 1, length) for k in ks]
        for phase, effect, ev in rows:
            phases.append(phase)
            effects.append(effect)
            valids.append(ev)

    return (
        torch.stack(phases).to(torch.float16).numpy(),  # [T_lat,128]
        torch.stack(effects).to(torch.float16).numpy(),  # [T_lat,4,128]
        torch.stack(valids).numpy(),  # [T_lat,4]
    )


def rebuild_manifest(out_dir: Path, args: argparse.Namespace) -> int:
    """Regenerate ``manifest.json`` from the npz files on disk (no GPU)."""
    entries = []
    for f in sorted(out_dir.glob("features_*.npz")):
        with np.load(f) as z:
            entries.append(
                {
                    "episode_id": int(z["episode_id"]),
                    "task_cluster": str(z["task_cluster"]),
                    "latent_frames": int(z["phase"].shape[0]),
                    "file": f.name,
                }
            )
    entries.sort(key=lambda e: e["episode_id"])
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": _CACHE_VERSION,
                "cte_checkpoint": str(args.cte_checkpoint),
                "latent_cache": str(args.latent_cache),
                "window_latents": args.window_latents,
                "raw_per_latent": RAW_PER_LATENT,
                "phase_dim": 128,
                "effect_dim": 128,
                "effect_history": 4,
                "num_episodes": len(entries),
                "num_boundaries": sum(e["latent_frames"] for e in entries),
                "episodes": entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"rebuilt manifest: {len(entries)} episodes, "
          f"{sum(e['latent_frames'] for e in entries):,} boundary frames")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cte-checkpoint", required=True)
    ap.add_argument("--latent-cache", required=True, help="the vae_cache output directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--window-latents", type=int, default=17, help="must match CTE training and the server")
    ap.add_argument("--limit", type=int, default=0, help="encode only the first N episodes (0 = all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rebuild-manifest", action="store_true", help="rescan the output dir and exit (no GPU)")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.rebuild_manifest:
        return rebuild_manifest(out_dir, args)

    model, cfg = load_cte(Path(args.cte_checkpoint), args.device)
    if cfg.image_channels != 48:
        raise RuntimeError(f"CTE expects image_channels={cfg.image_channels}, the latent cache stores 48")

    cache_dir = Path(args.latent_cache)
    files = sorted(cache_dir.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"no episode_*.npz under {cache_dir}; run vae_cache first")
    if args.limit > 0:
        files = files[: args.limit]
    print(f"CTE: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"window_latents={args.window_latents}")
    print(f"encoding {len(files)} episodes -> {out_dir}")

    by_id: dict[int, dict] = {}
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        for e in json.loads(manifest_path.read_text()).get("episodes", []):
            by_id[int(e["episode_id"])] = e

    t0 = time.perf_counter()
    done = skipped = 0
    for f in files:
        with np.load(f) as z:
            episode_id = int(z["episode_id"])
            task_cluster = str(z["task_cluster"])
            latents = torch.from_numpy(z["latents"])
            actions = torch.from_numpy(z["actions"])
        out_file = out_dir / f"features_{episode_id:06d}.npz"
        if out_file.exists() and not args.overwrite:
            skipped += 1
            continue

        phase, effect, effect_valid = encode_episode(model, cfg, latents, actions, args.window_latents)
        np.savez_compressed(
            out_file,
            phase=phase,
            effect=effect,
            effect_valid=effect_valid,
            # Raw frame of each row: row k is the boundary at raw frame 4k.
            boundary_frame=(np.arange(phase.shape[0], dtype=np.int64) * RAW_PER_LATENT),
            episode_id=np.int64(episode_id),
            task_cluster=np.array(task_cluster),
        )
        by_id[episode_id] = {
            "episode_id": episode_id,
            "task_cluster": task_cluster,
            "latent_frames": int(phase.shape[0]),
            "file": out_file.name,
        }
        # Rewrite after every episode: a `kill` mid-run must not leave a manifest
        # that silently under-reports what is on disk (that bug already bit once).
        entries = sorted(by_id.values(), key=lambda e: e["episode_id"])
        manifest_path.write_text(
            json.dumps(
                {
                    "version": _CACHE_VERSION,
                    "cte_checkpoint": str(args.cte_checkpoint),
                    "latent_cache": str(args.latent_cache),
                    "window_latents": args.window_latents,
                    "raw_per_latent": RAW_PER_LATENT,
                    "phase_dim": 128,
                    "effect_dim": 128,
                    "effect_history": 4,
                    "num_episodes": len(entries),
                    "num_boundaries": sum(e["latent_frames"] for e in entries),
                    "episodes": entries,
                },
                indent=2,
            )
            + "\n"
        )
        done += 1
        if done % 10 == 0 or done == len(files):
            el = time.perf_counter() - t0
            rate = done / el if el else 0
            eta = (len(files) - done - skipped) / rate if rate else 0
            print(f"  [{done}/{len(files)}] ep{episode_id} {phase.shape[0]} boundaries "
                  f"({rate:.2f} ep/s, ETA {eta/60:.1f} min)", flush=True)

    el = time.perf_counter() - t0
    total = sum(e["latent_frames"] for e in by_id.values())
    print()
    print(f"encoded {done} episodes ({skipped} skipped), {total:,} boundary frames, in {el/60:.1f} min")
    print(f"features: {out_dir}  ({sum(f.stat().st_size for f in out_dir.glob('*.npz'))/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
