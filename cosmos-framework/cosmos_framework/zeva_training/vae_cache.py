# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Encode the training videos to Wan latents once and cache them.

The CTE consumes *latents*, not pixels (see ``probe_vae``), and encoding is by
far the most expensive part of CTE data preparation. So the encode pass runs
once here and everything downstream reads the cache.

Two properties of the tokenizer make the cache straightforward:

1. **It is causal in time.** Encoding a whole episode at once yields exactly the
   same latents as encoding each prefix separately, so we never have to worry
   about context windows matching between cache time and train time.
2. **Latent frame *i* corresponds to raw frame 4i** (``T_lat = 1 + (T_raw-1)//4``,
   confirmed by ``probe_vae``: 9 raw -> 3 latents, 33 raw -> 9). That is precisely
   the CTE's boundary cadence, so the cache stores the latents at exactly the
   resolution the CTE's ``transition_actions`` need — latents ``i`` and ``i+1``
   are separated by raw actions ``[4i, 4i+4)``.

What lands on disk, one file per episode (``[C,T,H,W]`` — the VAE's own layout,
batch dim dropped; ``cte_dataset`` permutes when it builds ``[B,T,C,H,W]`` windows):

    latents      [48, T_lat, 30, 52]  float16   (VAE output is fp32; see note below)
    actions      [N_raw, A]           float32   (raw joint positions, un-normalized)
    episode_id   scalar
    task_cluster str
    length       scalar               (raw frames; T_lat == 1 + (length-1)//4)

``T_lat`` varies per episode (the CTE pads/ masks), so this stores full episodes
rather than fixed windows and lets ``cte_dataset`` cut windows.

Note on dtype: ``WanVAE.encode`` casts its result back to the input dtype, but the
conv stack runs in bfloat16, so the fp32 output carries only bf16 precision. We
store fp16 to halve cache I/O; ``cte_dataset`` promotes to fp32 before the CTE,
which is what ``_FrozenVAEDeltaTarget`` does internally anyway.

Usage::

    PYTHONPATH=. python -m cosmos_framework.zeva_training.vae_cache \\
        --vae-path "$WAN_VAE_PATH" \\
        --dataset-root "$XHAND_DATA_ROOT" \\
        --output "$ZEVA_WORK/datasets/xhand_cte_cache"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Must match the inference path — see probe_vae for why.
_VISION_HEIGHT = 480
_VISION_WIDTH = 832
_CACHE_VERSION = 2

from cosmos_framework.data.generator.action.xhand_camera import (
    CAMERA_CONTRACT,
    compose_xhand_views,
    require_camera_contract,
)

# One Wan latent frame per this many raw control steps — the VAE's temporal
# compression factor, and the cadence serving sees (the client re-queries every 4
# steps and sends one image).
RAW_PER_LATENT = 4


def _compose_frames(video: torch.Tensor) -> torch.Tensor:
    """``[C,T,H,W]`` uint8 composite (already three-view) -> ``[1,C,T,H',W']`` in [-1,1].

    Mirrors ``action_policy_server_robocasa365_zeva.py:695-703``: upscale to
    (480, 832), then map [0,255] -> [-1,1]. ``WanVAE.encode`` does *not*
    normalize its input (it normalizes the output latent), so this has to happen
    here.
    """
    x = video.float()  # [C,T,H,W]
    c, t = x.shape[:2]
    x = x.permute(1, 0, 2, 3).reshape(t * c, 1, *x.shape[-2:])  # fold T into batch
    x = F.interpolate(x, size=(_VISION_HEIGHT, _VISION_WIDTH), mode="bilinear", align_corners=False)
    x = x.reshape(t, c, _VISION_HEIGHT, _VISION_WIDTH).permute(1, 0, 2, 3)  # [C,T,H,W]
    x = x.div_(127.5).sub_(1.0)
    return x.unsqueeze(0)  # [1,C,T,H,W]


def encode_padded(vae, x: torch.Tensor) -> torch.Tensor:
    """Encode ``[1,C,T,H,W]`` when ``T`` is not of the form ``4n+1``.

    ``WanVAE_.encode`` asserts ``T == 1 or (T-1) % 4 == 0`` *before* its own
    internal padding runs, so callers with an arbitrary episode length have to
    pad themselves (the error message points at a ``pad_video_batch`` helper that
    is not actually present in this release).

    Padding is safe here **because the tokenizer is causal in time**: appending
    frames at the end cannot change the latents of earlier frames. We pad to the
    next ``4n+1``, encode, then truncate to the number of latents that correspond
    to real frames::

        valid_latents = 1 + (T_raw - 1) // 4

    which is the formula the encoder itself documents one line above its assert
    (``latent_T = 1 + (T - 1) // 4``).
    """
    t_raw = x.shape[2]
    valid = 1 + (t_raw - 1) // 4
    if t_raw == 1 or (t_raw - 1) % 4 == 0:
        return vae.encode(x)
    t_pad = 1 + ((t_raw - 1) // 4 + 1) * 4
    x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, t_pad - t_raw))
    latent = vae.encode(x)
    return latent[:, :, :valid]


def encode_per_frame(vae, x: torch.Tensor) -> torch.Tensor:
    """Encode each **boundary frame independently**, exactly as serving does.

    The serving path sees one observation per request -- the client re-queries every
    ``RAW_PER_LATENT`` control steps and sends a single image -- so the only latent it
    can ever produce is ``_encode_cte_frame``'s single-frame one (``T=1``). A
    whole-episode causal encode gives a *different* latent for the same frame: measured
    on real content, cosine 0.72-0.85 and roughly **half the magnitude**, because the
    causal encoder conditions on every preceding frame.

    Training the CTE on the sequence encoding therefore makes every serving request out
    of distribution. Encode per frame here so cache and server agree by construction,
    and take only the boundary frames (index ``4k``) that serving actually observes --
    ``1 + (T-1) // 4`` of them, the same count ``encode_padded`` produced.

    This also matches the released RoboCasa server, which is the only inference
    implementation the authors shipped.
    """
    frames = x[0].permute(1, 0, 2, 3)  # [C,T,H,W] -> [T,C,H,W]
    lats = [
        # frames[t:t+1] is [1,C,H,W]; unsqueeze(2) inserts the temporal axis -> [1,C,1,H,W],
        # which is exactly what the server's `_encode_cte_frame` feeds the VAE.
        vae.encode(frames[t : t + 1].unsqueeze(2))[0, :, 0]  # [C,h,w]
        for t in range(0, frames.shape[0], RAW_PER_LATENT)
    ]
    return torch.stack(lats, dim=1).unsqueeze(0)  # [1,C,T_lat,30,52]


def _decode_full_episode(dataset, episode) -> torch.Tensor:
    """Decode every frame of one episode as the shared three-view composite.

    Reuses the loader's own decode helpers (``XHandLeRobotDataset._decode`` /
    ``_compose_video``) so caching and training see identical pixels — the only
    difference is that we ask for the whole episode instead of a 33-frame window.
    Building a fresh dataset per episode just to widen the window would re-index
    the metadata 101 times, so instead we call the underlying decoder directly
    with one timestamp per raw frame.
    """
    from lerobot.datasets.video_utils import decode_video_frames

    n = int(episode.length)
    timestamps = [i / dataset.fps for i in range(n)]
    views = {
        name: decode_video_frames(path, timestamps, tolerance_s=2e-4, backend="torchcodec")
        for name, path in episode.video_paths.items()
    }
    composite = compose_xhand_views(views, view_size=dataset.view_size, layout=dataset.camera_layout)
    return (composite * 255.0).clamp_(0, 255).to(torch.uint8)


def rebuild_manifest(out_dir: Path, args: argparse.Namespace) -> int:
    """Regenerate ``manifest.json`` from the npz files actually on disk.

    The manifest is advisory — ``cte_dataset`` reads episode geometry straight
    from each npz — but a stale or wrong one is actively misleading (it once
    listed 20 of 101 episodes, with 16 of those carrying a wrong latent count).
    Reading the truth back out of the files costs seconds and needs no GPU.
    """
    entries = []
    for f in sorted(out_dir.glob("episode_*.npz")):
        with np.load(f) as z:
            require_camera_contract(z, str(f))
            latents = z["latents"]  # [C,T,H,W]
            entries.append(
                {
                    "episode_id": int(z["episode_id"]),
                    "task_cluster": str(z["task_cluster"]),
                    "raw_frames": int(z["length"]),
                    "latent_frames": int(latents.shape[1]),  # T, not shape[2] (H)
                    "file": f.name,
                }
            )
    entries.sort(key=lambda e: e["episode_id"])
    bad = [e for e in entries if e["latent_frames"] != 1 + (e["raw_frames"] - 1) // 4]
    if bad:
        print(f"WARNING: {len(bad)} episodes have an unexpected latent count:")
        for e in bad[:5]:
            print(f"  ep{e['episode_id']}: {e['latent_frames']} latents for {e['raw_frames']} raw frames")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": _CACHE_VERSION,
                "camera_contract": CAMERA_CONTRACT,
                # Kept identical to the incremental writer below so the two paths
                # cannot drift; nothing reads this key today — `cte_dataset` reads
                # the npz files directly and treats the manifest as advisory.
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "vae_path": args.vae_path,
                "image_channels": 48,
                "latent_hw": [30, 52],
                "vision_input_hw": [_VISION_HEIGHT, _VISION_WIDTH],
                "fps": args.fps,
                "num_episodes": len(entries),
                "num_latent_frames": sum(e["latent_frames"] for e in entries),
                "episodes": entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"rebuilt manifest: {len(entries)} episodes, "
          f"{sum(e['latent_frames'] for e in entries):,} latent frames")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--chunk-length", type=int, default=32, help="only used to find the dataset's root layout")
    ap.add_argument("--action-stats-path", default=None, help="unused here; actions are cached raw")
    ap.add_argument("--limit", type=int, default=0, help="encode only the first N episodes (0 = all)")
    ap.add_argument("--overwrite", action="store_true", help="re-encode episodes whose cache file exists")
    ap.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="rebuild manifest.json by scanning the cache directory, then exit (no GPU work)",
    )
    ap.add_argument("--shard-index", type=int, default=int(os.environ.get("RANK", "0")))
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    args = ap.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        ap.error("shard-index must be in [0, num-shards)")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_manifest:
        return rebuild_manifest(out_dir, args)

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    from cosmos_framework.data.generator.action.datasets.xhand_lerobot_dataset import XHandLeRobotDataset
    from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

    # action_normalization=None: we want raw joint positions on disk. The CTE
    # trains on real action magnitudes, and normalizing here would silently bake
    # the minmax stats into the cache.
    dataset = XHandLeRobotDataset(
        args.dataset_root,
        fps=args.fps,
        chunk_length=args.chunk_length,
        use_state=False,
        action_mode="full18",
        state_mode="joint18",  # inert here (use_state=False); the cache holds no state at all
        action_normalization=None,
        split="full",  # cache every episode; train/val split happens downstream
        parquet_cache_size=1,
    )
    if len({ep.episode_id for ep in dataset.episodes}) != len(dataset.episodes):
        raise ValueError("Cache requires globally unique episode IDs; merge/reindex the dataset first")
    print(f"dataset: {len(dataset.episodes)} episodes under {args.dataset_root}")

    vae = Wan2pt2VAEInterface(
        vae_path=args.vae_path,
        spatial_compression_factor=16,
        temporal_compression_factor=4,
        causal=True,
    )

    episodes = dataset.episodes
    if args.limit > 0:
        episodes = episodes[: args.limit]
    episodes = episodes[args.shard_index::args.num_shards]

    # Seed from any previous run so re-running after an interruption keeps every
    # already-encoded episode in the manifest.
    manifest_path = out_dir / ("manifest.json" if args.num_shards == 1 else f"manifest.rank{args.shard_index}.json")
    by_id: dict[int, dict] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        require_camera_contract(manifest, str(manifest_path))
        for e in manifest.get("episodes", []):
            by_id[int(e["episode_id"])] = e

    def write_manifest() -> None:
        """Rewrite the manifest from the current state.

        Called after *every* episode, because the previous version only wrote it
        on clean exit — a `kill` mid-encode left it listing a stale subset, and
        downstream consumers happily trained on that subset.
        """
        entries = sorted(by_id.values(), key=lambda e: e["episode_id"])
        manifest_path.write_text(
            json.dumps(
                {
                    "version": _CACHE_VERSION,
                "camera_contract": CAMERA_CONTRACT,
                    "dataset_root": str(Path(args.dataset_root).resolve()),
                    "vae_path": args.vae_path,
                    "image_channels": 48,
                    "latent_hw": [30, 52],
                    "vision_input_hw": [_VISION_HEIGHT, _VISION_WIDTH],
                    "fps": args.fps,
                    "num_episodes": len(entries),
                    "num_latent_frames": sum(e["latent_frames"] for e in entries),
                    "episodes": entries,
                },
                indent=2,
            )
            + "\n"
        )

    if by_id:
        write_manifest()

    t_start = time.perf_counter()
    done = skipped = 0
    total_latents = 0

    for ep in episodes:
        out_file = out_dir / f"episode_{ep.episode_id:06d}.npz"
        if out_file.exists() and not args.overwrite:
            with np.load(out_file) as existing:
                require_camera_contract(existing, str(out_file))
            skipped += 1
            continue

        # Full-episode video: [T,C,256,512] uint8, and the raw actions alongside.
        # The tokenizer is causal, so this equals encoding every prefix.
        video = _decode_full_episode(dataset, ep).permute(1, 0, 2, 3)  # [C,T,256,512]
        actions = dataset._load_low_dim(ep)["action"].numpy()  # [T,18] raw joint positions

        x = _compose_frames(video).cuda()
        with torch.no_grad():
            # Per-frame, NOT `encode_padded`: serving can only produce single-frame
            # latents (one image per request), so the cache must match it. See
            # `encode_per_frame`.
            latent = encode_per_frame(vae, x)  # [1,48,T_lat,30,52]

        # Latent i corresponds to raw frame 4i, so the cache should carry exactly
        # ceil((length-1)/4)+1 of them — verify rather than trust.
        expected = 1 + (ep.length - 1) // 4
        if latent.shape[2] != expected:
            raise RuntimeError(
                f"episode {ep.episode_id}: got {latent.shape[2]} latents for {ep.length} raw frames, "
                f"expected {expected}"
            )

        latent = latent[0].to(torch.float16).cpu().numpy()
        np.savez_compressed(
            out_file,
            camera_contract=np.array(CAMERA_CONTRACT),
            latents=latent,
            actions=actions,
            episode_id=np.int64(ep.episode_id),
            task_cluster=np.array(ep.task_name),
            length=np.int64(ep.length),
        )
        # NOTE: `latent` is [C,T,H,W] here (batch dim already dropped above), so the
        # temporal extent is shape[1] — NOT shape[2], which is height. (The 5-D
        # tensor in the verification above uses shape[2] for T; that is a different
        # rank, which is exactly how this got mis-indexed once.)
        n_latent = int(latent.shape[1])
        by_id[int(ep.episode_id)] = {
            "episode_id": ep.episode_id,
            "task_cluster": ep.task_name,
            "raw_frames": ep.length,
            "latent_frames": n_latent,
            "file": out_file.name,
        }
        write_manifest()
        total_latents += n_latent
        done += 1

        if done % 10 == 0 or done == len(episodes):
            el = time.perf_counter() - t_start
            rate = done / el if el else 0
            eta = (len(episodes) - done - skipped) / rate if rate else 0
            print(
                f"  [{done}/{len(episodes)}] ep{ep.episode_id} "
                f"{ep.length} raw -> {n_latent} latent  "
                f"({rate:.2f} ep/s, ETA {eta/60:.1f} min)",
                flush=True,  # else the redirect to a log file swallows progress until exit
            )

    write_manifest()  # already current; this just guarantees a final write

    el = time.perf_counter() - t_start
    print()
    print(f"encoded {done} episodes ({skipped} skipped), {total_latents} latent frames, in {el/60:.1f} min")
    print(f"cache:   {out_dir}  ({sum(f.stat().st_size for f in out_dir.glob('*.npz'))/1e9:.2f} GB)")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
