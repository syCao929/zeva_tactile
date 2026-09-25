# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from itertools import count

import pytest
import torch

from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import instantiate


def _sample(index):
    return {"video": torch.zeros(3, 1, 16, 16), "sample_id": torch.tensor(index)}


class _FiniteSamples(torch.utils.data.Dataset):
    def __init__(self):
        self.reads = 0

    def __len__(self):
        return 5

    def __getitem__(self, index):
        self.reads += 1
        return _sample(index)


class _InfiniteSamples(torch.utils.data.IterableDataset):
    def __len__(self):
        return 5

    def __iter__(self):
        for index in count():
            yield _sample(index)


def _loader(monkeypatch, dataset, *, restart_on_iter=False, num_workers=0, persistent_workers=False):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    return instantiate(
        {
            "_target_": PackingDataLoader,
            "dataloader": {
                "_target_": RankPartitionedDataLoader,
                "datasets": {"test": {"dataset": dataset, "ratio": 1}},
                "batch_size": 3,
                "num_workers": num_workers,
                "persistent_workers": persistent_workers,
                **({"multiprocessing_context": "spawn", "timeout": 30} if num_workers else {}),
            },
            "tokenizer_spatial_compression_factor": 16,
            "tokenizer_temporal_compression_factor": 4,
            "patch_spatial": 1,
            "max_samples_per_batch": 2,
            "restart_on_iter": restart_on_iter,
        }
    )


def _ids(batch):
    return [value.item() for value in batch["sample_id"]]


@pytest.mark.parametrize(("num_workers", "persistent_workers"), [(0, False), (1, False), (1, True)])
def test_validation_can_repeat_complete_passes(monkeypatch, num_workers, persistent_workers):
    loader = _loader(
        monkeypatch,
        _FiniteSamples(),
        restart_on_iter=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    expected = [[0, 1], [2, 3], [4]]
    assert [_ids(batch) for batch in loader] == expected
    assert [_ids(batch) for batch in loader] == expected


def test_validation_restarts_after_partial_pass_without_stale_buffer(monkeypatch):
    loader = _loader(monkeypatch, _FiniteSamples(), restart_on_iter=True)
    first_pass = iter(loader)
    assert _ids(next(first_pass)) == [0, 1]
    assert len(loader.buffers[0]) == 1
    first_pass.close()
    assert [_ids(batch) for batch in loader] == [[0, 1], [2, 3], [4]]


def test_first_iteration_keeps_prewarmed_samples_and_restored_step(monkeypatch):
    dataset = _FiniteSamples()
    loader = _loader(monkeypatch, dataset, restart_on_iter=True)
    assert dataset.reads == 3
    loader.set_start_iteration(4000)
    assert _ids(next(iter(loader))) == [0, 1]
    assert dataset.reads == 3
    assert loader.global_id == 4001


def test_default_keeps_infinite_training_stream_and_restored_step(monkeypatch):
    loader = _loader(monkeypatch, _InfiniteSamples())
    loader.set_start_iteration(4000)
    first_pass = iter(loader)
    assert _ids(next(first_pass)) == [0, 1]
    first_pass.close()
    # A new outer iterator must retain both the prewarmed residual and source cursor.
    second_pass = iter(loader)
    assert _ids(next(second_pass)) == [2, 3]
    assert _ids(next(second_pass)) == [4, 5]
    assert loader.global_id == 4003
