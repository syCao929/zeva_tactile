"""Data-contract checks using small real Parquet fixtures and mocked video IO."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from fractions import Fraction

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pi0_zeva import data


@pytest.fixture
def dataset_files(tmp_path):
    root = tmp_path / "dataset"
    source = root / "xhand/PressButton4Times/lerobot_v21/lerobot"
    (source / "meta").mkdir(parents=True)
    action_names = [f"arm_joint_{i}.pos" for i in range(6)] + [
        f"hand_joint_{i}.pos" for i in range(12)
    ]
    state_names = [f"unused_{i}" for i in range(1972)]
    for i, name in zip(data.STATE_INDICES, action_names):
        state_names[i] = name
    info = {
        "codebase_version": "v2.1",
        "fps": 15,
        "features": {
            "action": {"shape": [18], "names": action_names},
            "observation.state": {"shape": [1972], "names": state_names},
        },
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    records = []
    for episode_id in range(6):
        n = 40
        records.append(
            {"episode_index": episode_id, "length": n, "tasks": ["Press four times."]}
        )
        state = (
            np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], (n, 1972)).copy()
            + episode_id * 100
        )
        action = state[:, data.STATE_INDICES] + np.arange(18)[None, :]
        parquet = source / f"data/chunk-000/episode_{episode_id:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({"action": action.tolist(), "observation.state": state.tolist()}),
            parquet,
        )
        for key in data.CAMERAS.values():
            video = source / f"videos/chunk-000/{key}/episode_{episode_id:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.touch()
    (source / "meta/episodes.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records)
    )
    stats = tmp_path / "stats.json"
    data.compute_normalization(root, stats, split_val_ratio=1 / 3)
    return root, source, stats


def test_import_is_lazy():
    source = "import sys; import pi0_zeva.data; assert not {'torch', 'numpy', 'pyarrow', 'lerobot', 'jax', 'openpi'} & sys.modules.keys()"
    subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )


def test_split_matches_cosmos_and_statistics_exclude_validation(dataset_files):
    root, _, stats_path = dataset_files
    # Execute only the upstream helper so parity doesn't import the Cosmos stack.
    upstream = (
        Path(__file__).parents[1]
        / "cosmos-framework/cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py"
    )
    tree = ast.parse(upstream.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "split_episode_ids"
    )
    scope = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(upstream), "exec"),
        scope,
    )
    for split in ("train", "val", "full"):
        assert data._split_ids(6, 42, 1 / 3, split) == scope["split_episode_ids"](
            6, 42, 1 / 3, split
        )
    train = data.XHandPi0Dataset(root, stats_path, split_val_ratio=1 / 3)
    val = data.XHandPi0Dataset(root, stats_path, split="val", split_val_ratio=1 / 3)
    train_ids = {e.episode_id for e in train.episodes}
    val_ids = {e.episode_id for e in val.episodes}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(range(6))
    stats = json.loads(stats_path.read_text())
    assert stats["count"] == 4 * 40
    expected = np.concatenate([np.arange(40) + i * 100 for i in sorted(train_ids)])
    np.testing.assert_allclose(stats["state"]["mean"], expected.mean())
    np.testing.assert_allclose(stats["state"]["std"], expected.std())
    assert len(train) == 4 * (40 - 32)


def test_current_frame_only_action_order_padding_and_roundtrip(
    dataset_files, monkeypatch
):
    root, _, stats_path = dataset_files
    calls = []

    def decode(path, frame, image_size):
        calls.append((str(path), frame))
        return torch.full((3, image_size, image_size), 0.5)

    monkeypatch.setattr(data, "_decode_current", decode)
    dataset = data.XHandPi0Dataset(root, stats_path, split_val_ratio=1 / 3)
    sample = dataset[3]
    assert len(calls) == 2
    assert all(frame == 3 for _, frame in calls)
    assert "cam_left" in calls[0][0] and "cam_front" in calls[1][0]
    assert sample["image_masks"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": False,
    }
    assert sample["images"]["base_0_rgb"].shape == (3, 224, 224)
    torch.testing.assert_close(
        sample["images"]["base_0_rgb"], torch.full((3, 224, 224), 0.5)
    )
    assert sample["state"].shape == (32,)
    assert sample["actions"].shape == (32, 32)
    assert not sample["state"][18:].any()
    assert not sample["actions"][:, 18:].any()
    _, raw_actions = data._read_episode(dataset.episodes[0])
    torch.testing.assert_close(
        dataset.normalizer.denormalize_actions(sample["actions"]),
        torch.from_numpy(raw_actions[3:35]),
    )
    assert sample["frame_index"] == 3
    assert sample["prompt"] == "Press four times."


def test_tactile_is_causal_and_resets_between_episodes(dataset_files, monkeypatch):
    root, _, stats_path = dataset_files
    monkeypatch.setattr(data, "_decode_current", lambda *_: torch.zeros(3, 8, 8))
    dataset = data.XHandPi0Dataset(
        root, stats_path, split_val_ratio=1 / 3, tactile=True, horizon=32
    )
    for index in (0, 3, 8):
        sample = dataset[index]
        episode, frame = dataset._resolve_index(index)
        history = sample["tactile_state"]
        valid = sample["tactile_valid"]
        assert history.shape == (30, 1972)
        assert valid.dtype == torch.bool
        assert int(valid.sum()) == frame + 1
        assert not history[~valid].any()
        raw_state, _ = data._read_episode(dataset.episodes[episode])
        torch.testing.assert_close(
            history[valid], torch.from_numpy(raw_state[: frame + 1])
        )


@pytest.mark.parametrize(
    "argument,value", [("split_seed", 43), ("split_val_ratio", 0.1), ("horizon", 16)]
)
def test_statistics_provenance_rejects_wrong_partition(dataset_files, argument, value):
    root, _, stats_path = dataset_files
    kwargs = {"split_val_ratio": 1 / 3, argument: value}
    with pytest.raises(ValueError, match="provenance mismatch"):
        data.XHandPi0Dataset(root, stats_path, **kwargs)


def test_statistics_rejects_changed_data_and_wrong_padding(dataset_files):
    root, source, stats_path = dataset_files
    parquet = source / "data/chunk-000/episode_000000.parquet"
    stat = parquet.stat()
    os.utime(parquet, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        data.XHandPi0Dataset(root, stats_path, split_val_ratio=1 / 3)
    normalizer = data.Normalizer.load(stats_path)
    with pytest.raises(ValueError, match="before padding"):
        normalizer.normalize_actions(torch.zeros(32, 32))


def test_cte_uses_latest_completed_boundary_and_exact_task_context(
    dataset_files, monkeypatch, tmp_path
):
    root, _, stats_path = dataset_files
    monkeypatch.setattr(data, "_decode_current", lambda *_: torch.zeros(3, 8, 8))
    cache = tmp_path / "features"
    cache.mkdir()
    for episode_id in range(6):
        np.savez(
            cache / f"features_{episode_id:06d}.npz",
            episode_id=episode_id,
            task_cluster="PressButton4Times",
            boundary_frame=np.arange(10) * 4,
            phase=np.broadcast_to(np.arange(10)[:, None], (10, 128)),
            effect=np.zeros((10, 4, 128)),
            effect_valid=np.zeros((10, 4), dtype=bool),
        )
    dataset = data.XHandPi0Dataset(
        root, stats_path, split_val_ratio=1 / 3, feature_cache=cache
    )
    assert not dataset[3]["behavior_phase"].any()
    assert torch.equal(dataset[4]["behavior_phase"], torch.ones(128))
    assert not dataset[4]["behavior_effect_valid"].any()
    upstream = (
        Path(__file__).parents[1]
        / "cosmos-framework/cosmos_framework/data/generator/action/datasets/zeva_behavior_wrapper.py"
    )
    tree = ast.parse(upstream.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "task_context_vector"
    )
    scope = {"torch": torch, "hashlib": data.hashlib, "GLOBAL_DIM": 256}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(upstream), "exec"),
        scope,
    )
    torch.testing.assert_close(
        dataset[4]["behavior_global"],
        scope["task_context_vector"]("PressButton4Times"),
        atol=0,
        rtol=0,
    )


def test_invalid_joint_metadata_fails(dataset_files):
    root, source, stats_path = dataset_files
    metadata = source / "meta/info.json"
    info = json.loads(metadata.read_text())
    info["features"]["action"]["names"][0] = "wrong_joint"
    metadata.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="joint order"):
        data.XHandPi0Dataset(root, stats_path, split_val_ratio=1 / 3)


def test_nonfinite_training_values_fail_before_writing_stats(dataset_files, tmp_path):
    root, _, _ = dataset_files
    episodes, _ = data._index(root, "train", 42, 1 / 3, 32)
    episode = episodes[0]
    state, actions = data._read_episode(episode)
    state[0, 0] = float("nan")
    pq.write_table(
        pa.table({"action": actions.tolist(), "observation.state": state.tolist()}),
        episode.parquet_path,
    )
    output = tmp_path / "bad_stats.json"
    with pytest.raises(ValueError, match="Nonfinite"):
        data.compute_normalization(root, output, split_val_ratio=1 / 3)
    assert not output.exists()


def test_pyav_decoder_accounts_for_start_pts_and_selects_only_current_frame(
    monkeypatch,
):
    stream = types.SimpleNamespace(
        start_time=40000,
        time_base=Fraction(1, 15000),
        codec_context=types.SimpleNamespace(thread_count=0),
    )
    calls = []

    class Container:
        streams = types.SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def seek(self, offset, **kwargs):
            assert offset == 42000
            assert kwargs == {"stream": stream, "backward": True, "any_frame": False}

        def decode(self, selected_stream):
            assert selected_stream is stream
            for pts in (41000, 42000):
                calls.append(pts)
                yield types.SimpleNamespace(
                    pts=pts,
                    to_ndarray=lambda format: np.full((4, 4, 3), 191, np.uint8),
                )
            raise AssertionError("Decoder must stop after selecting the current frame")

    monkeypatch.setitem(
        sys.modules, "av", types.SimpleNamespace(open=lambda path: Container())
    )
    image = data._decode_current(Path("camera.mp4"), 2, 8)
    assert image.shape == (3, 8, 8)
    torch.testing.assert_close(image, torch.full((3, 8, 8), 191 / 255 * 2 - 1))
    assert calls == [41000, 42000, "closed"]
    assert stream.codec_context.thread_count == 1


def test_pyav_decoder_rejects_missing_exact_frame(monkeypatch):
    stream = types.SimpleNamespace(
        start_time=0,
        time_base=Fraction(1, 15000),
        codec_context=types.SimpleNamespace(),
    )

    class Container:
        streams = types.SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, *args, **kwargs):
            pass

        def decode(self, *args):
            yield types.SimpleNamespace(pts=1000)
            yield types.SimpleNamespace(pts=3000)

    monkeypatch.setitem(
        sys.modules, "av", types.SimpleNamespace(open=lambda path: Container())
    )
    with pytest.raises(ValueError, match="No video frame within"):
        data._decode_current(Path("missing_frame.mp4"), 2, 8)
