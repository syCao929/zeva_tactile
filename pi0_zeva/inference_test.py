import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pi0_zeva import inference
from pi0_zeva.config import TrainConfig
from pi0_zeva.data import STATE_INDICES, Normalizer
from pi0_zeva.runtime import IMAGE_KEYS


class Tokenizer:
    def tokenize(self, prompt):
        return np.zeros(48, dtype=np.int64), np.ones(48, dtype=np.bool_)


class Policy(torch.nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.value = value
        self.calls = []

    def sample_actions(self, observation, **kwargs):
        self.calls.append(kwargs)
        return torch.full((len(observation.state), 32, 32), self.value)


def _stats():
    return {
        "version": 1,
        "normalization": "zscore",
        "state": {"mean": [0.0] * 18, "std": [1.0] * 18},
        "action": {"mean": [3.0] * 18, "std": [2.0] * 6 + [4.0] * 12},
        "provenance": {
            "state_indices": list(STATE_INDICES),
            "action_indices": list(range(18)),
            "action_representation": "absolute_joint_position",
            "horizon": 32,
            "fps": 15,
            "split_seed": 42,
            "split_val_ratio": 0.03,
        },
    }


def _batch():
    return {
        "images": {key: torch.zeros(2, 3, 16, 16) for key in IMAGE_KEYS},
        "image_masks": {key: torch.ones(2, dtype=torch.bool) for key in IMAGE_KEYS},
        "state": torch.zeros(2, 32),
        "prompt": ["Press."] * 2,
        "actions": torch.zeros(2, 32, 32),
        "behavior_phase": torch.zeros(2, 128),
        "tactile_state": torch.zeros(2, 30, 1972),
        "tactile_valid": torch.ones(2, 30, dtype=torch.bool),
    }


def _predictor(policy=None):
    return inference.Predictor(
        policy or Policy(),
        Tokenizer(),
        Normalizer(_stats()),
        config=TrainConfig(),
        checkpoint=Path("/test/run/checkpoints/step_00000001"),
        metadata={"step": 1},
        stats_path=Path("/test/run/norm_stats.json"),
        device="cpu",
    )


def test_predict_decodes_absolute_joints_and_preserves_memory_inputs():
    predictor, batch = _predictor(), _batch()
    actions = predictor.predict(batch, num_steps=3)
    assert actions.shape == (2, 32, 18)
    torch.testing.assert_close(actions[..., :6], torch.full((2, 32, 6), 5.0))
    torch.testing.assert_close(actions[..., 6:], torch.full((2, 32, 12), 7.0))
    call = predictor.policy.calls[0]
    assert call["behavior"] is batch
    assert call["tactile_state"] is batch["tactile_state"]
    assert call["tactile_valid"] is batch["tactile_valid"]
    assert call["num_steps"] == 3
    assert not predictor.policy.training


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_predict_rejects_nonfinite_commands(value):
    with pytest.raises(ValueError, match="nonfinite"):
        _predictor(Policy(value)).predict(_batch())


def test_predict_rejects_wrong_output_shape():
    class WrongShape(Policy):
        def sample_actions(self, observation, **kwargs):
            return torch.zeros(2, 32, 18)

    with pytest.raises(ValueError, match="shaped"):
        _predictor(WrongShape()).predict(_batch())


def test_offline_metrics_use_radians_separate_arm_hand_and_fixed_noise():
    batch = _batch()
    sample = {
        key: (
            {name: value[0] for name, value in values.items()}
            if isinstance(values, dict)
            else values[0]
        )
        for key, values in batch.items()
    }
    predictor = _predictor()
    result = inference.evaluate_dataset(predictor, [sample] * 7, max_samples=3, seed=17)
    noises = [call["noise"].clone() for call in predictor.policy.calls]
    assert result["sample_indices"] == [0, 3, 6]
    assert result["metrics"] == {
        "arm_mae_rad": 2.0,
        "arm_rmse_rad": 2.0,
        "hand_mae_rad": 4.0,
        "hand_rmse_rad": 4.0,
    }
    predictor.policy.calls.clear()
    assert (
        inference.evaluate_dataset(predictor, [sample] * 7, max_samples=3, seed=17)
        == result
    )
    for previous, call in zip(noises, predictor.policy.calls, strict=True):
        torch.testing.assert_close(previous, call["noise"], rtol=0, atol=0)


def _checkpoint(tmp_path, **metadata_overrides):
    directory = tmp_path / "checkpoints" / "step_00000001"
    directory.mkdir(parents=True)
    stats = json.dumps(_stats()).encode()
    (tmp_path / "norm_stats.json").write_bytes(stats)
    metadata = {
        "format_version": 1,
        "step": 1,
        "config": TrainConfig().as_dict(),
        "norm_sha256": hashlib.sha256(stats).hexdigest(),
        "backbone": "backbone.safetensors",
        "memory": None,
    } | metadata_overrides
    (directory / "manifest.json").write_text(json.dumps(metadata))
    (directory / "backbone.safetensors").write_bytes(b"fake-test-weights")
    return directory


@pytest.mark.parametrize(
    "invalid", ["hash", "mode", "provenance", "unversioned", "camera", "mapping"]
)
def test_load_rejects_manifest_or_normalization_before_model_creation(
    tmp_path, monkeypatch, invalid
):
    directory = _checkpoint(
        tmp_path, **({"memory": "memory.safetensors"} if invalid == "mode" else {})
    )
    stats_path = tmp_path / "norm_stats.json"
    if invalid == "hash":
        stats_path.write_text(stats_path.read_text() + " ")
    elif invalid == "provenance":
        stats = _stats()
        stats["provenance"]["state_indices"] = list(range(18))
        stats_path.write_text(json.dumps(stats))
        metadata_path = directory / "manifest.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["norm_sha256"] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata))
    elif invalid in {"unversioned", "camera", "mapping"}:
        metadata_path = directory / "manifest.json"
        metadata = json.loads(metadata_path.read_text())
        if invalid == "unversioned":
            metadata["config"].pop("camera_contract")
        elif invalid == "camera":
            metadata["config"]["camera_contract"] = "old_two_view"
        else:
            metadata["config"]["camera_mapping"]["right_wrist_0_rgb"] = (
                "observation.images.cam_front"
            )
        metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(
        inference,
        "build_policy",
        lambda **kw: pytest.fail("Must reject before building a policy"),
    )
    with pytest.raises(ValueError):
        inference.load_policy(directory, device="cpu")


def test_load_restores_backbone_and_memory_before_returning_predictor(
    tmp_path, monkeypatch
):
    directory = _checkpoint(tmp_path)
    calls = []
    policy = Policy()
    monkeypatch.setattr(
        inference, "PromptTokenizer", lambda *args, **kwargs: Tokenizer()
    )
    monkeypatch.setattr(inference, "configure_openpi", lambda root: {})
    monkeypatch.setattr(
        inference,
        "inspect_dependency_environment",
        lambda: {"ready": True, "errors": []},
    )
    monkeypatch.setattr(inference, "build_policy", lambda **kwargs: policy)
    monkeypatch.setattr(
        inference,
        "load_pretrained_backbone",
        lambda model, path: calls.append(("backbone", path)),
    )
    monkeypatch.setattr(
        inference.checkpoint_io,
        "load_memory",
        lambda model, path: calls.append(("memory", path)),
    )
    predictor = inference.load_policy(directory, device="cpu")
    assert predictor.policy is policy
    assert calls == [
        ("backbone", directory / "backbone.safetensors"),
        ("memory", directory),
    ]
    assert predictor.predict(_batch()).shape == (2, 32, 18)
