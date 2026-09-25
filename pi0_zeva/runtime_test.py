from dataclasses import replace
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from pi0_zeva import runtime


@pytest.fixture
def tokenizer_model(tmp_path):
    sentencepiece = pytest.importorskip("sentencepiece")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "press the button now\nmove the hand slowly\npress button four times\n"
    )
    prefix = tmp_path / "tokenizer"
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=32,
        model_type="char",
        hard_vocab_limit=False,
        minloglevel=2,
    )
    return prefix.with_suffix(".model")


@pytest.mark.parametrize("max_len", [3, 48])
def test_prompt_matches_sentencepiece_pi0_format(tokenizer_model, max_len):
    import sentencepiece

    direct = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_model))
    expected = direct.encode("press the button now", add_bos=True) + direct.encode("\n")
    tokenizer = runtime.PromptTokenizer(tokenizer_model, max_length=max_len)
    tokens, mask = tokenizer.tokenize("  press_the\nbutton_now  ")
    kept = min(len(expected), max_len)
    assert tokens.tolist() == expected[:kept] + [0] * (max_len - kept)
    assert mask.tolist() == [True] * kept + [False] * (max_len - kept)
    assert mask.dtype == np.bool_


def test_tokenizer_requires_local_asset(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local SentencePiece"):
        runtime.PromptTokenizer(tmp_path / "missing.model")


def _batch():
    return {
        "images": {
            key: torch.full((2, 3, 16, 16), -0.25) for key in runtime.IMAGE_KEYS
        },
        "image_masks": {key: torch.tensor([True, False]) for key in runtime.IMAGE_KEYS},
        "state": torch.zeros(2, 32),
        "prompt": ["press button", "move hand"],
    }


def test_observation_is_torch_dataclass_and_preserves_masks(tokenizer_model):
    batch = _batch()
    observation = runtime.make_observation(
        batch, runtime.PromptTokenizer(tokenizer_model), "cpu"
    )
    assert observation.state.shape == (2, 32)
    assert observation.tokenized_prompt.shape == (2, 48)
    assert observation.tokenized_prompt.dtype == torch.long
    assert observation.tokenized_prompt_mask.dtype == torch.bool
    assert observation.token_ar_mask is None and observation.token_loss_mask is None
    for key in runtime.IMAGE_KEYS:
        torch.testing.assert_close(observation.images[key], batch["images"][key])
        assert observation.image_masks[key].tolist() == [True, False]
    replaced = replace(observation, state=torch.ones(2, 32))
    assert observation.state.sum() == 0
    assert replaced.state.sum() == 64


@pytest.mark.parametrize("invalid", ["state", "pixels", "mask", "prompt", "camera"])
def test_observation_rejects_mismatched_data_contract(tokenizer_model, invalid):
    batch = _batch()
    key = runtime.IMAGE_KEYS[0]
    if invalid == "state":
        batch["state"] = torch.zeros(2, 18)
    elif invalid == "pixels":
        batch["images"][key] = torch.full((2, 3, 16, 16), 128, dtype=torch.uint8)
    elif invalid == "mask":
        batch["image_masks"][key] = torch.ones(2)
    elif invalid == "prompt":
        batch["prompt"] = ["one prompt only"]
    else:
        del batch["images"][key]
    with pytest.raises(ValueError):
        runtime.make_observation(batch, runtime.PromptTokenizer(tokenizer_model), "cpu")


def test_configure_checks_source_without_importing_openpi(tmp_path, monkeypatch):
    reference = {}
    for relative in runtime._REFERENCE_SHA256:
        path = tmp_path / "src" / "openpi" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "raise RuntimeError('source verification must not import this')\n"
        )
        reference[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(runtime, "_REFERENCE_SHA256", reference)
    monkeypatch.delitem(sys.modules, "openpi", raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    info = runtime.configure_openpi(str(tmp_path))
    assert info["source_compatible"]
    assert info["reference_commit"] == runtime.OPENPI_REFERENCE_COMMIT
    assert "openpi" not in sys.modules
    changed = tmp_path / "src" / "openpi" / next(iter(reference))
    changed.write_text("different source\n")
    with pytest.raises(ValueError, match="differs from OpenPI"):
        runtime.configure_openpi(str(tmp_path))


def test_environment_probe_is_isolated_and_cpu_only(monkeypatch):
    expected = {"ready": False, "errors": ["Missing dependency: flax"]}

    def run(command, **kwargs):
        assert command[0] == sys.executable
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["JAX_PLATFORMS"] == "cpu"
        assert kwargs["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert kwargs["timeout"] == 45
        return subprocess.CompletedProcess(
            command, 0, "OPENPI_PREFLIGHT_JSON=" + json.dumps(expected), ""
        )

    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime.inspect_dependency_environment() == expected


def test_environment_probe_reports_timeout(monkeypatch):
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(runtime.subprocess, "run", run)
    result = runtime.inspect_dependency_environment()
    assert not result["ready"]
    assert "CPU dependency probe failed" in result["errors"][0]
