"""Versioned XHand camera inputs; no training dependencies."""

CAMERA_CONTRACT = "xhand_three_view_v2_wrist_right"
# OpenPI slot names do not imply the physical mounting of the camera.
CAMERAS = {
    "base_0_rgb": "observation.images.cam_front",
    "left_wrist_0_rgb": "observation.images.cam_left",
    "right_wrist_0_rgb": "observation.images.cam_right",
}


def require_camera_contract(metadata, source):
    if metadata.get("camera_contract") != CAMERA_CONTRACT:
        raise ValueError(
            f"Camera contract mismatch: {source}; requires {CAMERA_CONTRACT}. "
            "Old or unversioned two-view artifacts must be regenerated."
        )


def require_policy_camera(config, source):
    # Check the saved dictionary before TrainConfig can fill in defaults.
    require_camera_contract(config, source)
    if config.get("camera_mapping") != CAMERAS:
        raise ValueError(f"Camera mapping mismatch: {source}; requires {CAMERAS}")
