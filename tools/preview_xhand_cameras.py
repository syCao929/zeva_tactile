#!/usr/bin/env python3
"""Preview the actual shared three-view compositor from a LeRobot episode."""

import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cosmos-framework"))
from cosmos_framework.data.generator.action.xhand_camera import (
    CAMERA_CONTRACT,
    CAMERA_KEYS,
    compose_xhand_views,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True, help="LeRobot root containing videos/")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--time", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    views = {}
    for name, key in CAMERA_KEYS.items():
        path = (
            args.episode_root
            / f"videos/chunk-{args.episode_id // 1000:03d}"
            / key
            / f"episode_{args.episode_id:06d}.mp4"
        )
        frame = subprocess.check_output(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(args.time),
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ]
        )
        pixels = np.array(Image.open(io.BytesIO(frame)).convert("RGB"))
        views[name] = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float() / 255
        print(f"{name}: {path}")
    pixels = (compose_xhand_views(views)[0].permute(1, 2, 0) * 255).clamp(0, 255).byte().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.output)
    print(f"{CAMERA_CONTRACT}: {pixels.shape}; wrist on top, front/left below -> {args.output}")


if __name__ == "__main__":
    main()
