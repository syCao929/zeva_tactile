"""CPU regression checks for camera routing, memory leakage, and PIM gradients.

Run: PYTHONPATH=cosmos-framework python -m pytest -q tools/test_xhand_threeview_pim.py
"""

import __future__

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from cosmos_framework.data.generator.action.xhand_camera import (
    CAMERA_CONTRACT,
    CAMERA_KEYS,
    compose_xhand_views,
    require_camera_contract,
)
from cosmos_framework.inference.xhand_pim import XHandPIMContext
from cosmos_framework.model.zeva.persistent_interaction_memory import (
    CausalPromptEncoder,
    inject_causal_prompt,
)

# The wrapper only needs Torch/Numpy; avoid importing unrelated robot dataset
# packages through datasets/__init__.py in a CPU development environment.
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "xhand_behavior_wrapper_test_target",
    ROOT / "cosmos-framework/cosmos_framework/data/generator/action/datasets/zeva_behavior_wrapper.py",
)
wrapper_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper_module)


def test_three_cameras_are_present_and_right_is_the_large_wrist_panel():
    assert CAMERA_KEYS["wrist"] == "observation.images.cam_right"
    views = {
        name: torch.full((2, 3, 480, 640), value) for name, value in (("front", 0.2), ("left", 0.5), ("wrist", 0.9))
    }
    image = compose_xhand_views(views)
    assert image.shape == (2, 3, 576, 512)
    torch.testing.assert_close(image[..., :384, :], torch.full((2, 3, 384, 512), 0.9))
    torch.testing.assert_close(image[..., 384:, :256], torch.full((2, 3, 192, 256), 0.2))
    torch.testing.assert_close(image[..., 384:, 256:], torch.full((2, 3, 192, 256), 0.5))
    del views["wrist"]
    with pytest.raises(ValueError, match="cam_right"):
        compose_xhand_views(views)


def test_old_camera_cache_is_rejected():
    with pytest.raises(ValueError, match="two-view"):
        require_camera_contract({"version": 1}, "old cache")
    require_camera_contract({"camera_contract": CAMERA_CONTRACT}, "new cache")


def _source_method(relative_path, class_name, method_name, globals_):
    """Load real preprocessing methods without importing CUDA serving dependencies."""
    tree = ast.parse((ROOT / "cosmos-framework/cosmos_framework" / relative_path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {**globals_, "__builtins__": __builtins__}
    exec(
        compile(ast.fix_missing_locations(module), relative_path, "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace[method_name]


def test_training_cache_and_server_use_identical_pixels(monkeypatch):
    # Non-uniform images expose resize / range / camera-order drift that solid
    # color geometry tests cannot detect. Execute the actual three entry points.
    rng = np.random.default_rng(7)
    images = {name: rng.integers(0, 256, (48, 64, 3), dtype=np.uint8) for name in CAMERA_KEYS}
    decoded = {
        name: torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255 for name, image in images.items()
    }
    dataset = SimpleNamespace(
        _decode=lambda path, start: decoded[path],
        use_image_augmentation=False,
        viewpoint="concat_view",
        view_size=256,
        camera_layout="three_view_grid",
        fps=15,
    )
    episode = SimpleNamespace(video_paths={name: name for name in CAMERA_KEYS}, length=1)
    compose_training = _source_method(
        "data/generator/action/datasets/xhand_lerobot_dataset.py",
        "XHandLeRobotDataset",
        "_compose_video",
        {"compose_xhand_views": compose_xhand_views},
    )
    expected = (compose_training(dataset, episode, 0) * 255).clamp(0, 255).byte()
    decoder = ModuleType("lerobot.datasets.video_utils")
    decoder.decode_video_frames = lambda path, *args, **kwargs: decoded[path]
    monkeypatch.setitem(sys.modules, "lerobot.datasets.video_utils", decoder)
    from cosmos_framework.zeva_training.vae_cache import _decode_full_episode

    torch.testing.assert_close(_decode_full_episode(dataset, episode), expected)
    compose_server = _source_method(
        "scripts/action_policy_server_xhand.py",
        "XHandPolicyService",
        "_compose_client_view",
        {"torch": torch, "CAMERA_KEYS": CAMERA_KEYS, "compose_xhand_views": compose_xhand_views},
    )
    obs = {CAMERA_KEYS[name]: image for name, image in images.items()}
    service = SimpleNamespace(_client_image=lambda obs, key: obs.get(key), xargs=SimpleNamespace(camera_view_size=256))
    actual = compose_server(service, obs)
    np.testing.assert_array_equal(actual, expected[0].permute(1, 2, 0).numpy())


def test_initial_cte_boundary_supplies_phase_without_future_actions():
    tree = ast.parse((ROOT / "cosmos-framework/cosmos_framework/scripts/action_policy_server_xhand.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_BoundaryBuffer")
    import collections

    namespace = {"torch": torch, "np": np, "collections": collections}
    exec(
        compile(
            ast.Module(body=[cls], type_ignores=[]),
            "boundary_buffer",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    buffer = namespace["_BoundaryBuffer"](stride=4, action_dim=18, max_frames=17)
    assert buffer.as_cte_inputs() is None
    buffer.observe(torch.zeros(48, 30, 52))
    latents, actions = buffer.as_cte_inputs()
    assert latents.shape == (1, 1, 48, 30, 52)
    assert actions.shape == (1, 0, 4, 18)


def _features(marker: int):
    phase = np.zeros((9, 128), np.float32)
    phase[:, marker] = 1
    effects = np.zeros((9, 4, 128), np.float32)
    effects[:, -1, marker] = 1
    valid = np.zeros((9, 4), bool)
    valid[4:, -1] = True
    return dict(phase=phase, effect=effects, effect_valid=valid)


class FakeSFT:
    def __init__(self, query_id, split):
        self._dataset = SimpleNamespace(
            episodes=[SimpleNamespace(episode_id=query_id, task_name="press")],
            training_episode_ids={0, 1},
            split=split,
            _resolve_index=lambda idx: (0, idx),
        )

    def __getitem__(self, idx):
        return {"action": torch.zeros(32, 18)}

    def __len__(self):
        return 33


def _cache(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"camera_contract": CAMERA_CONTRACT}))
    for episode in range(3):
        np.savez(
            tmp_path / f"features_{episode:06d}.npz",
            episode_id=episode,
            task_cluster="press",
            camera_contract=CAMERA_CONTRACT,
            **_features(episode),
        )


def test_training_support_excludes_self_and_validation_and_uses_no_query_future(tmp_path):
    _cache(tmp_path)
    wrapper = wrapper_module.ZevaBehaviorWrapper(
        FakeSFT(0, "train"), tmp_path, pim_training=True, pim_context_dropout=0
    )
    assert wrapper.support_candidates(0, "press") == [1]
    sample = wrapper[0]  # Query has no completed actions, but an independent demo is allowed.
    assert sample["behavior_pim_valid"].any()
    assert not sample["behavior_effect_valid"].any()
    assert sample["behavior_pim_effect"][sample["behavior_pim_valid"], 1].eq(1).all()
    # Changing the query's future and held-out trajectory must not change memory.
    wrapper._episode_row(0)["effect"][1:] = 99
    wrapper._episode_row(2)["effect"][:] = 99
    torch.testing.assert_close(wrapper[0]["behavior_pim_effect"], sample["behavior_pim_effect"])


def test_validation_uses_training_support_and_context_dropout_is_training_only(tmp_path):
    _cache(tmp_path)
    val = wrapper_module.ZevaBehaviorWrapper(FakeSFT(2, "val"), tmp_path, pim_training=True, pim_context_dropout=1)
    assert val.support_candidates(2, "press") == [0, 1]
    assert val[0]["behavior_pim_valid"].any()
    train = wrapper_module.ZevaBehaviorWrapper(FakeSFT(0, "train"), tmp_path, pim_training=True, pim_context_dropout=1)
    assert not train[0]["behavior_pim_valid"].any()
    assert not train[0]["behavior_pim_effect"].any()


def test_cte_default_split_matches_the_policy_split(tmp_path):
    from cosmos_framework.zeva_training.cte_dataset import CTECacheWindowDataset

    (tmp_path / "manifest.json").write_text(json.dumps({"camera_contract": CAMERA_CONTRACT}))
    for episode in range(101):
        np.savez(
            tmp_path / f"episode_{episode:06d}.npz", episode_id=episode, task_cluster="press",
            camera_contract=CAMERA_CONTRACT, length=69,
            latents=np.zeros((48, 18, 1, 1), np.float16), actions=np.zeros((69, 18), np.float32),
        )
    val = CTECacheWindowDataset(tmp_path, split="val")
    train = CTECacheWindowDataset(tmp_path, split="train")
    expected = set(torch.randperm(101, generator=torch.Generator().manual_seed(42))[:3].tolist())
    assert {row["episode_id"] for row in val.episodes} == expected
    assert not {row["episode_id"] for row in train.episodes}.intersection(expected)


def test_memory_lifecycle_preserves_retries_but_clears_new_scenes():
    session = XHandPIMContext()
    obs = dict(prompt="press", pim_episode_id="scene-a", pim_attempt_id=0)
    assert session.prepare(obs, reset=True)
    phase, effects = torch.eye(128)[0], torch.eye(128)[:4]
    valid = torch.ones(4, dtype=torch.bool)
    for _ in range(4):
        session.observe_and_query(phase, effects, valid)
    assert len(session.memory) == 0  # No complete 16-control interval until boundary 4.
    session.observe_and_query(phase, effects, valid)
    assert len(session.memory) == 1
    assert session.prepare({**obs, "pim_attempt_id": 1}, reset=True)
    assert len(session.memory) == 1 and session.boundary == 0
    assert session.prepare({**obs, "pim_episode_id": "scene-b"}, reset=True)
    assert len(session.memory) == 0


def test_without_explicit_ids_reset_never_retains_another_scene():
    session = XHandPIMContext()
    session.prepare({"prompt": "press"}, reset=True)
    phase = torch.eye(128)[0]
    session.memory.append_completed(task_cluster="press", phase=phase, effect=phase, attempt_id=0)
    session.prepare({"prompt": "press"}, reset=True)
    assert len(session.memory) == 0


def test_policy_loss_reaches_pim_encoder_after_zero_gate_warmup():
    torch.manual_seed(12)
    encoder = CausalPromptEncoder()
    projector = torch.nn.Linear(256, 32)
    gate = torch.nn.Parameter(torch.zeros(1))
    parameters = [*encoder.parameters(), *projector.parameters(), gate]
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    inputs = (
        torch.randn(2, 256),
        torch.randn(2, 128),
        torch.randn(2, 4, 128),
        torch.ones(2, 4, dtype=torch.bool),
        torch.randn(2, 4, 128),
        torch.randn(2, 4, 128),
        torch.ones(2, 4, dtype=torch.bool),
    )
    baseline, target = torch.randn(2, 32), torch.randn(2, 32)
    for step in range(2):
        optimizer.zero_grad()
        prompt = projector(encoder(*inputs))
        output = inject_causal_prompt(baseline, prompt, gate, inputs[-1])
        if step == 0:
            torch.testing.assert_close(output, baseline)
        (output - target).square().mean().backward()
        assert gate.grad.abs().sum() > 0
        if step == 1:
            assert sum(float(p.grad.abs().sum()) for p in encoder.parameters() if p.grad is not None) > 0
        optimizer.step()
    torch.testing.assert_close(inject_causal_prompt(baseline, prompt, gate, torch.zeros_like(inputs[-1])), baseline)
