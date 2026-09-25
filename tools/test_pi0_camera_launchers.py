"""CPU launcher checks: custom caches reach both commands and pair metadata."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pi0_zeva.camera import CAMERA_CONTRACT
from pi0_zeva.config import TrainConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["baseline", "tactile"])
def test_custom_cache_reaches_launcher_command(tmp_path, mode):
    scripts = tmp_path / "tools"
    scripts.mkdir()
    common = (ROOT / "tools/pi0-comparison-common.sh").read_text()
    # Bypass resources/GPU checks only; exercise the real parser and final command.
    common += '\npi0_prepare_pair_contract() { PI0_RUN_DIR="/test/run"; }\npi0_prepare_gpu() { :; }\n'
    (scripts / "pi0-comparison-common.sh").write_text(common)
    launcher = scripts / f"train-pi0-{mode}.sh"
    launcher.write_text((ROOT / "tools" / launcher.name).read_text())
    encoder = tmp_path / "encoder.pt"
    encoder.touch()
    cache = str(tmp_path / "custom features")
    output = subprocess.run(
        [
            "bash",
            str(launcher),
            "--base-checkpoint",
            "/test/base",
            "--cte-cache",
            cache,
            "--dry-run",
        ],
        env={**os.environ, "PI0_TACTILE_CHECKPOINT": str(encoder)},
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tokens = shlex.split(output)
    assert f"feature_cache={cache}" in tokens
    assert "xhand_cte_features_v4" not in output


@pytest.mark.parametrize("old_base", [False, True])
def test_pair_payload_validates_camera_and_uses_custom_cache(tmp_path, old_base):
    workspace = tmp_path / "workspace"
    (workspace / "datasets").mkdir(parents=True)
    (workspace / "datasets/pi0_xhand_norm.json").write_text("{}")
    (tmp_path / "hf_weight").mkdir()
    (tmp_path / "hf_weight/paligemma_tokenizer.model").write_bytes(b"tokenizer")
    base = workspace / "step_00005000"
    base.mkdir()
    config = TrainConfig().as_dict()
    if old_base:
        config.pop("camera_contract")
    (base / "manifest.json").write_text(json.dumps({"config": config}))
    (base / "backbone.safetensors").write_bytes(b"weights")
    cache = workspace / "new features"
    cache.mkdir()
    np.savez(
        cache / "features_000000.npz", episode_id=0, camera_contract=CAMERA_CONTRACT
    )
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "camera_contract": CAMERA_CONTRACT,
                "phase_dim": 128,
                "effect_dim": 128,
                "effect_history": 4,
                "episodes": [{"episode_id": 0, "file": "features_000000.npz"}],
            }
        )
    )
    source = (ROOT / "tools/pi0-comparison-common.sh").read_text()
    code = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    output = workspace / "pair.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(sys.path),
        "PI0_WORKSPACE": str(workspace),
        "PI0_BASE_CHECKPOINT": str(base),
        "PI0_FEATURE_CACHE": str(cache),
        "PI0_OPENPI_ROOT": str(tmp_path / "openpi"),
        "PI0_NPROC": "8",
        "PI0_SEED": "42",
        "PI0_BATCH_SIZE": "2",
        "PI0_GRAD_ACCUM": "7",
        "PI0_STEPS": "2000",
        "PI0_LEARNING_RATE": "0.0002",
        "PI0_WARMUP_STEPS": "100",
    }
    result = subprocess.run(
        [sys.executable, "-", str(output)],
        input=code,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if old_base:
        assert result.returncode != 0 and "Camera contract mismatch" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        payload = json.loads(output.read_text())
        assert payload["feature_cache"] == str(cache)
        assert payload["camera_contract"] == CAMERA_CONTRACT
        assert payload["camera_mapping"] == config["camera_mapping"]
