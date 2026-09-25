# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Measure the Wan VAE's latent geometry before building a CTE.

Why this exists as a separate step: ``CausalTransitionEncoderConfig.image_channels``
must equal the VAE's latent channel count *exactly*. The inference server asserts
it at runtime (``action_policy_server_robocasa365_zeva.py:702``), so getting it
wrong does not fail loudly at training time — it fails when the trained CTE is
first loaded for inference. The encoder's own docstring is also stale (it says
``[B,T,3,H,W]`` while the server feeds latents), so the only trustworthy answer
is to encode a real frame and look.

It also times single vs batched encoding, which sizes the ``vae_cache`` step:
encoding is the expensive part of CTE training data prep, so the cache exists
purely to pay that cost once.

Usage::

    PYTHONPATH=. python -m cosmos_framework.zeva_training.probe_vae \\
        --vae-path "$WAN_VAE_PATH" \\
        --dataset-root "$XHAND_DATA_ROOT"
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

# Both of these mirror the inference path exactly:
#   - the server resizes every frame to (480, 832) before encoding
#     (action_policy_server_robocasa365_zeva.py:700), and 480 is a VAE encode
#     bucket key, so any other size lands in a different bucket and shifts the
#     latent distribution;
#   - WanVAE.encode only casts dtype — it does NOT normalize the input
#     (tokenizers/wan2pt2_vae_4x16x16.py:1192-1207 normalizes the *output*
#     latent). The caller must map [0,255] -> [-1,1] first.
_VISION_HEIGHT = 480
_VISION_WIDTH = 832


def _load_real_frames(dataset_root: str, count: int) -> torch.Tensor:
    """Return ``[count,3,H,W]`` uint8 frames straight from the training dataset.

    Using real frames rather than noise keeps the measured latent statistics
    meaningful (a constant image can collapse to a degenerate latent).
    """
    from cosmos_framework.data.generator.action.datasets.xhand_lerobot_dataset import XHandLeRobotDataset

    ds = XHandLeRobotDataset(
        dataset_root,
        fps=15.0,
        chunk_length=32,
        use_state=False,
        action_mode="full18",
        state_mode="joint18",  # inert here (use_state=False)
        action_normalization=None,
    )
    # One window gives us a whole clip of composite frames; take the first few.
    sample = ds[0]
    video = sample["video"]  # [C,T,H,W] uint8, already the left|wrist composite
    frames = video.permute(1, 0, 2, 3)[:count]  # [count,C,H,W]
    if frames.shape[0] < count:
        raise RuntimeError(f"dataset window only yielded {frames.shape[0]} frames, wanted {count}")
    return frames.contiguous()


def _preprocess(frames: torch.Tensor) -> torch.Tensor:
    """``[N,C,H,W]`` uint8 -> ``[1,C,N,H',W']`` float in [-1,1], server-identical."""
    x = frames.float()
    x = F.interpolate(x, size=(_VISION_HEIGHT, _VISION_WIDTH), mode="bilinear", align_corners=False)
    x = x.div_(127.5).sub_(1.0)
    return x.permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,N,H,W]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-frames", type=int, default=9, help="frames per encoded clip (33 raw -> 9 latent)")
    args = ap.parse_args()

    # NOTE: Wan2pt2VAEInterface is a plain wrapper, not an nn.Module -- it has no
    # .to(). It builds its inner WanVAE on the module-level DEVICE
    # (cosmos_framework/utils/flags.py:58, overridable via $COSMOS_DEVICE) and
    # already calls .eval().requires_grad_(False) on the inner module.
    print(f"loading Wan VAE from {args.vae_path}  (COSMOS_DEVICE={os.environ.get('COSMOS_DEVICE', 'cuda')})")
    vae = Wan2pt2VAEInterface(
        vae_path=args.vae_path,
        spatial_compression_factor=16,
        temporal_compression_factor=4,
        causal=True,
    )

    frames = _load_real_frames(args.dataset_root, args.num_frames)
    x = _preprocess(frames).to(args.device)
    print(f"input  {tuple(x.shape)}  (uint8 source, -> [-1,1], resized to {_VISION_HEIGHT}x{_VISION_WIDTH})")

    with torch.no_grad():
        t0 = time.perf_counter()
        latent = vae.encode(x)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    print(f"latent {tuple(latent.shape)}  dtype={latent.dtype}")

    channels = latent.shape[1]
    print()
    print("=" * 62)
    print(f"  image_channels = {channels}")
    print(f"  spatial        = {latent.shape[-2]}x{latent.shape[-1]}  "
          f"(= {_VISION_HEIGHT}/16 x {_VISION_WIDTH}/16)")
    print(f"  temporal       = {latent.shape[2]} latent frames for {args.num_frames} input frames "
          f"(compression {args.num_frames / max(latent.shape[2], 1):.1f}x)")
    print("=" * 62)
    print(f"\n  -> set CausalTransitionEncoderConfig.image_channels={channels}")
    print(f"  -> one clip ({args.num_frames} frames) encodes in {dt:.2f}s on {args.device}")

    # Sanity checks against what the inference server would assert.
    if latent.ndim != 5:
        print(f"\n  !! expected 5-D latent, got {latent.ndim}-D", flush=True)
        return 1
    if latent.shape[-2:] != (_VISION_HEIGHT // 16, _VISION_WIDTH // 16):
        print(f"\n  !! unexpected spatial size {tuple(latent.shape[-2:])}", flush=True)
        return 1
    if float(latent.std()) == 0.0:
        print("\n  !! latent has zero std — something is wrong with the input path", flush=True)
        return 1

    lat_flat = latent.float().flatten().cpu()
    print(f"  latent stats: mean={lat_flat.mean():.4f} std={lat_flat.std():.4f} "
          f"min={lat_flat.min():.3f} max={lat_flat.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
