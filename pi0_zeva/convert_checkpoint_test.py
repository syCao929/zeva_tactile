"""Prevent silent acceptance of incomplete or corrupted pretrained weights."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from pi0_zeva.convert_checkpoint import (
    _EMBEDDING,
    _SourceArray,
    _TIED_HEAD,
    _UNUSED_HEAD,
    _from_numpy,
    load_slicers,
    validate_and_complete,
)


@pytest.mark.parametrize("failure", ["missing", "unexpected", "shape", "nan"])
def test_reject_incomplete_or_corrupted_prediction_weights(failure):
    model = nn.Linear(3, 2)
    tensors = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    if failure == "missing":
        del tensors["weight"]
    elif failure == "unexpected":
        tensors["typo.weight"] = tensors["weight"]
    elif failure == "shape":
        tensors["weight"] = torch.zeros(3, 2)
    else:
        tensors["weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        validate_and_complete(model, tensors)


def _policy_with_language_heads():
    root = nn.Module()
    for name, dimensions in (
        (_EMBEDDING, (7, 3)),
        (_TIED_HEAD, (7, 3)),
        (_UNUSED_HEAD, (7, 2)),
    ):
        current = root
        *parents, leaf = name.split(".")
        for part in parents:
            if not hasattr(current, part):
                current.add_module(part, nn.Module())
            current = getattr(current, part)
        current.register_parameter(leaf, nn.Parameter(torch.randn(*dimensions)))
    root.paligemma_with_expert.paligemma.lm_head.weight = (
        root.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
    )
    return root


def test_checkpoint_embedding_populates_tied_head_and_unused_head_is_not_random():
    model = _policy_with_language_heads()
    embedding = torch.randn(7, 3)
    report = validate_and_complete(model, {_EMBEDDING: embedding})
    state = model.state_dict()
    assert torch.equal(state[_EMBEDDING], embedding)
    assert torch.equal(state[_TIED_HEAD], embedding)
    assert state[_UNUSED_HEAD].count_nonzero() == 0
    assert report["explicitly_initialized_unused"] == [_UNUSED_HEAD]
    assert report["missing_prediction_weights"] == []


def test_disagreeing_tied_weights_are_rejected():
    model = _policy_with_language_heads()
    with pytest.raises(ValueError, match="differs"):
        validate_and_complete(
            model, {_EMBEDDING: torch.zeros(7, 3), _TIED_HEAD: torch.ones(7, 3)}
        )


def test_finite_source_that_overflows_target_dtype_is_rejected():
    model = nn.Linear(3, 2).to(torch.bfloat16)
    tensors = {
        "weight": torch.full((2, 3), torch.finfo(torch.float32).max),
        "bias": torch.zeros(2),
    }
    with pytest.raises(ValueError, match="Non-finite tensor after dtype"):
        validate_and_complete(model, tensors)


def test_source_provenance_survives_official_numpy_operations():
    values = _SourceArray(np.arange(24, dtype=np.float32).reshape(2, 3, 4), "params/q")
    tensor = _from_numpy(values[1].transpose().reshape(12))
    assert tensor._pi0_source == "params/q"
    torch.testing.assert_close(
        tensor,
        torch.tensor(
            [12.0, 16.0, 20.0, 13.0, 17.0, 21.0, 14.0, 18.0, 22.0, 15.0, 19.0, 23.0]
        ),
    )


def test_unreviewed_upstream_mapping_is_never_executed(tmp_path: Path):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "convert_jax_model_to_pytorch.py").write_text(
        "raise RuntimeError('must not execute')"
    )
    with pytest.raises(ValueError, match="Unreviewed"):
        load_slicers(tmp_path)
