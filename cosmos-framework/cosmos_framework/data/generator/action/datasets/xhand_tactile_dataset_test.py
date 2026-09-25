from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.datasets.xhand_lerobot_dataset import (
    _ACTION_INDICES,
    _STATE_INDICES,
    XHandLeRobotDataset,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline


def _dataset() -> tuple[XHandLeRobotDataset, list[dict[str, torch.Tensor]]]:
    """Use the real indexing/sample builder without video files or robot hardware."""
    dataset = object.__new__(XHandLeRobotDataset)
    dataset.episodes = [
        SimpleNamespace(episode_id=i, command="press", task_name="press", category="xhand") for i in range(2)
    ]
    dataset._cum_ends = [40, 80]
    dataset._num_windows = 80
    dataset.chunk_length = 32
    dataset.fps = 15.0
    dataset.mode = "wam"
    dataset.domain_id = 23
    dataset.viewpoint = "concat_view"
    dataset.camera_layout = "left_wrist_horizontal"
    dataset.use_state = True
    dataset.use_tactile = True
    dataset.tactile_memory_steps = 30
    dataset.emit_behavior_metadata = True
    dataset.action_indices = _ACTION_INDICES["full18"]
    dataset.state_indices = _STATE_INDICES["joint18"]
    episodes = [
        {
            "state": (torch.arange(72, dtype=torch.float32) + i * 100).view(-1, 1).expand(-1, 1972).clone(),
            "action": torch.zeros(72, 18),
        }
        for i in range(2)
    ]
    dataset._load_low_dim = lambda episode: episodes[episode.episode_id]
    dataset._compose_video = lambda episode, start: torch.zeros(33, 3, 32, 64)
    return dataset, episodes


@pytest.mark.parametrize("frame", [0, 4, 29, 35])
def test_tactile_sample_is_a_dense_causal_window(frame: int) -> None:
    dataset, episodes = _dataset()
    sample = dataset[frame]
    assert sample["tactile_state"].shape == (30, 1972)
    valid = sample["tactile_valid"]
    assert valid.sum().item() == min(frame + 1, 30)
    assert valid[-1]
    assert torch.count_nonzero(sample["tactile_state"][~valid]) == 0
    observed = sample["tactile_state"][valid, 0]
    assert observed[-1].item() == frame
    assert torch.all(observed.diff() == 1)
    torch.testing.assert_close(sample["proprio"], sample["tactile_state"][-1, list(_STATE_INDICES["joint18"])])

    # Future sensor data must have no influence on this policy input.
    episodes[0]["state"][frame + 1 :] = 99999
    torch.testing.assert_close(dataset[frame]["tactile_state"], sample["tactile_state"])


def test_new_episode_does_not_inherit_tactile_from_previous_episode() -> None:
    dataset, _ = _dataset()
    dataset[39]
    start = dataset[40]
    assert start["behavior_episode_id"].item() == 1
    assert start["tactile_valid"].sum().item() == 1
    assert torch.count_nonzero(start["tactile_state"][:-1]) == 0
    assert torch.all(start["tactile_state"][-1] == 100)


def test_action_transform_preserves_tactile_and_validity() -> None:
    dataset, _ = _dataset()
    sample = dataset[4]
    expected = sample["tactile_state"].clone()
    valid = sample["tactile_valid"].clone()
    pipeline = ActionTransformPipeline(tokenizer_config=None, max_action_dim=64, format_prompt_as_json=True)
    result = pipeline(sample, "256")
    torch.testing.assert_close(result["tactile_state"], expected)
    torch.testing.assert_close(result["tactile_valid"], valid)


def test_client_server_and_training_use_identical_windows_across_queries_and_attempts() -> None:
    from cosmos_framework.inference.xhand_tactile import XHandTactileInput
    from cosmos_framework.inference.xhand_tactile_client import TactileClientWindow

    dataset, episodes = _dataset()
    server = XHandTactileInput()
    for episode in range(2):
        client = TactileClientWindow(episode_id=f"attempt-{episode}")
        for frame in range(37):
            raw_state = episodes[episode]["state"][frame].numpy()
            client.append(frame, raw_state)
            if frame % 4:
                continue
            packet = client.packet(frame)
            packet["observation/state"] = raw_state
            window = server.prepare(packet, reset=packet["cte_reset"])
            training_sample = dataset[episode * 40 + frame]
            torch.testing.assert_close(window.state, training_sample["tactile_state"], rtol=0, atol=0)
            torch.testing.assert_close(window.valid, training_sample["tactile_valid"])
            server.commit(window)
