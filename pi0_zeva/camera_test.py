"""Reject camera-version mixing before starting expensive training."""

import json
from pathlib import Path

import numpy as np
import pytest

from pi0_zeva.camera import CAMERA_CONTRACT, CAMERAS, require_policy_camera
from pi0_zeva.config import TrainConfig, load_config
from pi0_zeva.data import index_feature_cache


def test_policy_contract_records_mapping_and_matches_shared_cte():
    from cosmos_framework.data.generator.action.xhand_camera import (
        CAMERA_CONTRACT as shared,
    )

    assert CAMERA_CONTRACT == shared
    config = TrainConfig().as_dict()
    require_policy_camera(config, "new checkpoint")
    assert config["camera_mapping"] == CAMERAS
    config["camera_mapping"]["right_wrist_0_rgb"] = "wrong_camera"
    with pytest.raises(ValueError, match="Camera mapping mismatch"):
        TrainConfig(**config)
    assert TrainConfig().camera_mapping == CAMERAS


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "manifest_missing",
        "manifest_old",
        "episode_missing",
        "episode_old",
        "dimensions",
        "missing_file",
    ],
)
def test_feature_cache_validates_manifest_and_episode_versions(tmp_path, invalid):
    path = tmp_path / "features_000000.npz"
    manifest = {
        "camera_contract": CAMERA_CONTRACT,
        "phase_dim": 128,
        "effect_dim": 128,
        "effect_history": 4,
        "episodes": [{"episode_id": 0, "file": path.name}],
    }
    episode = {"camera_contract": CAMERA_CONTRACT, "episode_id": 0}
    if invalid == "manifest_missing":
        manifest.pop("camera_contract")
    elif invalid == "manifest_old":
        manifest["camera_contract"] = "two_view"
    elif invalid == "episode_missing":
        episode.pop("camera_contract")
    elif invalid == "episode_old":
        episode["camera_contract"] = "two_view"
    elif invalid == "dimensions":
        manifest["effect_history"] = 8
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    if invalid != "missing_file":
        np.savez(path, **episode)
    if invalid is None:
        assert index_feature_cache(tmp_path) == {0: path}
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            index_feature_cache(tmp_path)


def test_default_configs_and_cache_override():
    root = Path(__file__).resolve().parents[1]
    for name in ("xhand_zeva", "xhand_zeva_tactile"):
        path = root / f"configs/pi0/{name}.json"
        config = load_config(path)
        assert config.feature_cache == "datasets/xhand_cte_features_threeview"
        assert "v2-threeview" in config.init_checkpoint
        assert (
            load_config(path, ["feature_cache=/new/cte_features"]).feature_cache
            == "/new/cte_features"
        )
