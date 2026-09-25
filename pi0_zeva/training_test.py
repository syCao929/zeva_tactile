"""CPU checks for training state, validation, and checkpoint retention."""

import copy
import json
import random
import socket

import numpy as np
import pytest
import torch
from safetensors.torch import load_model
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from pi0_zeva.camera import CAMERA_CONTRACT, CAMERAS
from pi0_zeva.checkpoint import (
    backbone_path,
    capture_rng,
    load_memory,
    pin_backbone,
    prune_checkpoints,
    resolve_checkpoint,
    restore_rng,
    save_checkpoint,
)
from pi0_zeva.train import BatchStream, evaluate


class TinyPolicy(nn.Module):
    def __init__(self, memory=False):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        if memory:
            self.backbone.requires_grad_(False)
            self.memory = nn.Linear(2, 2)

    def training_step(self, optimizer):
        value = torch.randn(3, 2) + random.random() + float(np.random.random())
        output = self.backbone(value)
        if hasattr(self, "memory"):
            output = self.memory(output)
        optimizer.zero_grad()
        loss = output.square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()


def save(policy, optimizer, directory, step, mode="baseline", cursor=None):
    return save_checkpoint(
        policy,
        optimizer,
        directory,
        step=step,
        config={
            "mode": mode,
            "horizon": 32,
            "camera_contract": CAMERA_CONTRACT,
            "camera_mapping": CAMERAS,
        },
        loader_state=cursor or {"epoch": 2, "offset": 3},
        rng_states=[capture_rng()],
        norm_sha256="test_norm_sha256",
    )


def test_checkpoint_restores_weights_optimizer_rng_and_cursor(tmp_path):
    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)
    policy = TinyPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    policy.training_step(optimizer)
    path = save(policy, optimizer, tmp_path, 1)
    expected_loss = policy.training_step(optimizer)
    expected_weights = copy.deepcopy(policy.state_dict())
    restored = TinyPolicy()
    load_model(restored.backbone, str(backbone_path(path)), strict=True)
    assert load_memory(restored, path)["norm_sha256"] == "test_norm_sha256"
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.8)
    state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    restored_optimizer.load_state_dict(state["optimizer"])
    assert state["loader"] == {"epoch": 2, "offset": 3}
    restore_rng(state["rng_by_rank"][0])
    actual_loss = restored.training_step(restored_optimizer)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    for key, expected in expected_weights.items():
        torch.testing.assert_close(restored.state_dict()[key], expected, atol=0, rtol=0)
    assert resolve_checkpoint(tmp_path / "checkpoints/latest.json") == path
    with pytest.raises(FileExistsError):
        save(restored, restored_optimizer, tmp_path, 1)


def test_pruning_baseline_and_stage2_keeps_pinned_backbone_and_memory(tmp_path):
    base_run, memory_run = tmp_path / "baseline", tmp_path / "stage2"
    memory_run.mkdir()
    base = TinyPolicy()
    base_optimizer = torch.optim.AdamW(base.parameters())
    old_base = save(base, base_optimizer, base_run, 1)
    pinned = pin_backbone(backbone_path(old_base), memory_run)
    pinned_bytes = pinned.read_bytes()
    with pytest.raises(FileExistsError):
        pin_backbone(backbone_path(old_base), memory_run)
    base.training_step(base_optimizer)
    save(base, base_optimizer, base_run, 2)
    prune_checkpoints(base_run, keep=1)
    assert not old_base.exists()
    assert pinned.read_bytes() == pinned_bytes
    policy = TinyPolicy(memory=True)
    load_model(policy.backbone, str(pinned), strict=True)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad])
    previous = save(policy, optimizer, memory_run, 5, mode="zeva")
    policy.training_step(optimizer)
    latest = save(policy, optimizer, memory_run, 6, mode="zeva")
    prune_checkpoints(memory_run, keep=1)
    assert not previous.exists()
    assert pinned.read_bytes() == pinned_bytes
    assert backbone_path(latest) == pinned.resolve()
    restored = TinyPolicy(memory=True)
    load_model(restored.backbone, str(backbone_path(latest)), strict=True)
    metadata = load_memory(restored, latest)
    assert metadata["step"] == 6 and metadata["world_size"] == 1
    for key, expected in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected, atol=0, rtol=0)
    assert not (latest / "backbone.safetensors").exists()


def test_checkpoint_rejects_wrong_memory_contract_and_latest_escape(tmp_path):
    policy = TinyPolicy()
    path = save(policy, torch.optim.AdamW(policy.parameters()), tmp_path, 1)
    with pytest.raises(ValueError, match="cannot resume a Stage-2"):
        load_memory(TinyPolicy(memory=True), path)
    latest = tmp_path / "checkpoints/latest.json"
    latest.write_text(json.dumps({"checkpoint": "../elsewhere"}))
    with pytest.raises(ValueError, match="sibling"):
        resolve_checkpoint(latest)


def make_stream(rank, num_workers, **cursor):
    dataset = TensorDataset(torch.arange(25))
    sampler = DistributedSampler(
        dataset, num_replicas=2, rank=rank, seed=19, drop_last=True
    )
    loader = DataLoader(
        dataset,
        batch_size=3,
        sampler=sampler,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers else None,
        timeout=30 if num_workers else 0,
        drop_last=True,
        generator=torch.Generator().manual_seed(19),
    )
    return BatchStream(loader, sampler, **cursor)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("cut", [2, 4, 7])
def test_batch_stream_resume_matches_across_epochs_without_rng_drift(rank, cut):
    uninterrupted = make_stream(rank, 0)
    for _ in range(cut):
        next(uninterrupted)
    cursor = uninterrupted.state_dict()
    expected = [next(uninterrupted)[0].clone() for _ in range(9)]
    restored = make_stream(rank, 0, **cursor)
    before = torch.get_rng_state().clone()
    actual = [next(restored)[0].clone() for _ in range(9)]
    assert torch.equal(before, torch.get_rng_state())
    for result, reference in zip(actual, expected):
        assert torch.equal(result, reference)
    assert restored.state_dict() == uninterrupted.state_dict()


def test_batch_stream_worker_count_change_preserves_deterministic_dataset():
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except PermissionError:
        pytest.skip(
            "Sandbox disallows the Unix sockets required for DataLoader tensor IPC"
        )
    source = make_stream(0, 0)
    next(source)
    cursor = source.state_dict()
    expected = [next(source)[0].clone() for _ in range(6)]
    restored = make_stream(0, 1, **cursor)
    for reference in expected:
        assert torch.equal(next(restored)[0], reference)


class EvaluationPolicy(nn.Module):
    def forward(self, observation, actions, noise=None, time=None):
        loss = (actions - noise).square().mean() + time.mean() + torch.rand(())
        return {"loss": loss, "action_flow_loss": loss, "prior_nll": loss * 0}


def test_validation_repeats_nonempty_and_preserves_training_rng(monkeypatch):
    from pi0_zeva import runtime

    monkeypatch.setattr(
        runtime, "make_observation", lambda batch, tokenizer, device: batch
    )
    samples = [{"actions": torch.full((3, 32), float(i))} for i in range(5)]
    loader = DataLoader(
        samples, batch_size=2, generator=torch.Generator().manual_seed(123)
    )
    policy = EvaluationPolicy().train()
    before = torch.get_rng_state().clone()
    first = evaluate(policy, loader, None, torch.device("cpu"), max_batches=20, seed=7)
    second = evaluate(policy, loader, None, torch.device("cpu"), max_batches=20, seed=7)
    assert first == second
    assert first["samples"] == 5
    assert policy.training
    assert torch.equal(before, torch.get_rng_state())
    limited = evaluate(policy, loader, None, torch.device("cpu"), max_batches=1, seed=7)
    assert limited["samples"] == 2


def test_validation_rejects_empty_and_nonfinite_and_restores_mode(monkeypatch):
    from pi0_zeva import runtime

    monkeypatch.setattr(
        runtime, "make_observation", lambda batch, tokenizer, device: batch
    )
    policy = EvaluationPolicy().train()
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="zero samples"):
        evaluate(policy, [], None, torch.device("cpu"), max_batches=2, seed=7)
    assert policy.training
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(FloatingPointError, match="Non-finite validation"):
        evaluate(
            policy,
            [{"actions": torch.full((1, 3, 32), float("nan"))}],
            None,
            torch.device("cpu"),
            max_batches=2,
            seed=7,
        )
    assert policy.training
    assert torch.equal(before, torch.get_rng_state())
