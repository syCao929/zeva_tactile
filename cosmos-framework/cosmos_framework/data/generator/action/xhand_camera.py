"""Shared XHand camera geometry for policy training, CTE caching, and serving."""

from __future__ import annotations

import torch
from torch.nn import functional as F

CAMERA_KEYS = {
    "front": "observation.images.cam_front",
    "left": "observation.images.cam_left",
    "wrist": "observation.images.cam_right",
}
CAMERA_LAYOUTS = {
    "three_view_grid": ("front", "left", "wrist"),
    "left_wrist_horizontal": ("left", "wrist"),
}
DEFAULT_CAMERA_LAYOUT = "three_view_grid"
CAMERA_CONTRACT = "xhand_three_view_v2_wrist_right"
VIEW_DESCRIPTION = (
    "The top panel is the wrist-mounted cam_right camera. "
    "The bottom-left panel is the external cam_front camera and "
    "the bottom-right panel is the external cam_left camera."
)


def compose_xhand_views(
    views: dict[str, torch.Tensor], *, view_size: int = 256, layout: str = DEFAULT_CAMERA_LAYOUT
) -> torch.Tensor:
    """Compose [T,C,H,W] float images in [0,1], using the same resize everywhere.

    Three-view grid: a large wrist view over two external views. All panels keep
    the source 4:3 aspect ratio. At view_size=256 the result is 576x512 (H,W).
    """
    if view_size < 4 or view_size % 4:
        raise ValueError("XHand view_size must be a positive multiple of four")
    names = CAMERA_LAYOUTS[layout]
    for name in names:
        if name not in views:
            raise ValueError(f"Missing XHand camera: {CAMERA_KEYS[name]}")
    if layout == "left_wrist_horizontal":
        return torch.cat(
            [
                F.interpolate(views[name], size=(view_size, view_size), mode="bilinear", align_corners=False)
                for name in names
            ],
            dim=-1,
        )
    small = (view_size * 3 // 4, view_size)
    front, left = [
        F.interpolate(views[name], size=small, mode="bilinear", align_corners=False) for name in ("front", "left")
    ]
    wrist = F.interpolate(views["wrist"], size=(small[0] * 2, small[1] * 2), mode="bilinear", align_corners=False)
    return torch.cat((wrist, torch.cat((front, left), dim=-1)), dim=-2)


def require_camera_contract(metadata, source: str) -> None:
    if str(metadata.get("camera_contract", "")) != CAMERA_CONTRACT:
        raise ValueError(
            f"{source}: expected {CAMERA_CONTRACT}. Old two-view caches/checkpoints cannot be "
            "mixed with three-view training. Rebuild into a new directory."
        )
