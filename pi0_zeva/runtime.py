"""Local OpenPI source checks and the pi0 observation boundary.

This module does not import OpenPI's JAX model definitions, download assets, or
construct a policy. Dependency probes run in a separate process with GPUs hidden.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch


OPENPI_REFERENCE_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
_REFERENCE_SHA256 = {
    "models/pi0_config.py": "d321858c30b398126035ea2b6da09e1c422eeeebb5ebf8df631daf9dc77647fa",
    "models_pytorch/pi0_pytorch.py": "dcd7b6f508808ce4b7170a1720a4c62c6da04edae3652d16c94a469dca924e16",
}
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def configure_openpi(openpi_root: str | None = None) -> dict[str, Any]:
    """Select a local checkout whose two policy interface files match the reference.

    Pass the repository directory, or set OPENPI_ROOT. This checks the two named
    files only; it does not certify the whole checkout or its Python environment.
    """
    selected = openpi_root or os.environ.get("OPENPI_ROOT")
    if selected is None:
        sibling = Path(__file__).resolve().parents[2] / "openpi-3d-tactile"
        if sibling.is_dir():
            selected = str(sibling)
        else:
            spec = importlib.util.find_spec("openpi")
            if spec is not None and spec.submodule_search_locations:
                selected = str(
                    Path(next(iter(spec.submodule_search_locations))).parent.parent
                )
    if selected is None:
        raise FileNotFoundError(
            "Provide --openpi-root or OPENPI_ROOT pointing to a local OpenPI checkout."
        )
    root = Path(selected).expanduser().resolve()
    package = root / "src" / "openpi"
    actual = {}
    for relative, expected in _REFERENCE_SHA256.items():
        path = package / relative
        if not path.is_file():
            raise FileNotFoundError(f"OpenPI source file is missing: {path}")
        actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[relative] != expected:
            raise ValueError(
                f"{path} differs from OpenPI {OPENPI_REFERENCE_COMMIT}; "
                "review the policy interface before using this checkout."
            )
    loaded = sys.modules.get("openpi")
    if loaded is not None:
        locations = [Path(p).resolve() for p in getattr(loaded, "__path__", ())]
        if package.resolve() not in locations:
            raise RuntimeError(
                "A different OpenPI package is already imported; start a fresh process."
            )
    source_dir = str(package.parent)
    sys.path[:] = [source_dir, *(entry for entry in sys.path if entry != source_dir)]
    importlib.invalidate_caches()
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        checkout_commit = revision.stdout.strip() if revision.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        checkout_commit = None
    return {
        "root": str(root),
        "source_dir": source_dir,
        "reference_commit": OPENPI_REFERENCE_COMMIT,
        "checkout_commit": checkout_commit,
        "source_compatible": True,
        "source_sha256": actual,
    }


_CPU_IMPORT_PROBE = r"""
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
sys.path[:] = json.loads(sys.argv[1])
report = {"python": sys.executable, "dependencies": {}, "errors": []}
for name in ("torch", "transformers", "safetensors", "sentencepiece", "jax", "flax", "jaxtyping", "beartype", "einops"):
    try:
        found = importlib.util.find_spec(name) is not None
        version = importlib.metadata.version(name) if found else None
    except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
        found, version = False, None
    report["dependencies"][name] = {"installed": found, "version": version}
    if not found:
        report["errors"].append(f"Missing dependency: {name}")
try:
    check = importlib.import_module("transformers.models.siglip.check")
    patched = bool(check.check_whether_transformers_replace_is_installed_correctly())
    # OpenPI's check function only verifies the transformers version. Verify
    # the actual replacement files too, so a partial patch cannot pass preflight.
    source = Path(next(iter(importlib.util.find_spec("openpi").submodule_search_locations)))
    patch_dir = source / "models_pytorch" / "transformers_replace"
    installed_dir = Path(importlib.import_module("transformers").__file__).parent
    patch_files = {}
    for relative in (
        "models/siglip/check.py", "models/siglip/modeling_siglip.py",
        "models/paligemma/modeling_paligemma.py", "models/gemma/configuration_gemma.py",
        "models/gemma/modeling_gemma.py",
    ):
        expected, installed = patch_dir / relative, installed_dir / relative
        patch_files[relative] = bool(
            expected.is_file() and installed.is_file()
            and hashlib.sha256(expected.read_bytes()).digest() == hashlib.sha256(installed.read_bytes()).digest()
        )
    patched = patched and all(patch_files.values())
    report["transformers_patch"] = {"ok": patched, "files_match_source": patch_files}
    if not patched:
        report["errors"].append("OpenPI transformers_replace patch check or file verification failed")
except Exception as exc:
    report["transformers_patch"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report["errors"].append("OpenPI transformers_replace patch is unavailable or incompatible")
try:
    policy_module = importlib.import_module("openpi.models_pytorch.pi0_pytorch")
    config_module = importlib.import_module("openpi.models.pi0_config")
    getattr(policy_module, "PI0Pytorch")
    getattr(config_module, "Pi0Config")
    report["openpi"] = {"importable": True, "policy_file": policy_module.__file__, "config_file": config_module.__file__}
except Exception as exc:
    report["openpi"] = {"importable": False, "error": f"{type(exc).__name__}: {exc}"}
    report["errors"].append("OpenPI policy/config imports failed")
report["ready"] = not report["errors"]
print("OPENPI_PREFLIGHT_JSON=" + json.dumps(report))
"""


def inspect_dependency_environment() -> dict[str, Any]:
    """Report import and patch readiness without constructing models or using GPUs.

    Call configure_openpi first to select the checkout. An importable package does
    not imply available weights; weight provenance and conversion are separate.
    """
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        JAX_PLATFORMS="cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CPU_IMPORT_PROBE, json.dumps(sys.path)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "python": sys.executable,
            "ready": False,
            "errors": [f"CPU dependency probe failed: {exc}"],
        }
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("OPENPI_PREFLIGHT_JSON="):
            return json.loads(line.split("=", 1)[1])
    return {
        "python": sys.executable,
        "ready": False,
        "errors": [
            f"CPU dependency probe exited {result.returncode}: {result.stderr[-2000:]}"
        ],
    }


class PromptTokenizer:
    """Official pi0 prompt tokenization using an explicitly supplied local model."""

    def __init__(self, model_path: str | Path, max_length: int = 48):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local SentencePiece model not found: {path}")
        import sentencepiece

        self.max_length = max_length
        self.model_path = str(path)
        self._tokenizer = sentencepiece.SentencePieceProcessor(
            model_proto=path.read_bytes()
        )

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(prompt, str):
            raise TypeError("Each prompt must be a string")
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self._tokenizer.encode(cleaned, add_bos=True) + self._tokenizer.encode(
            "\n"
        )
        if len(tokens) > self.max_length:
            logging.getLogger(__name__).warning(
                "Prompt has %d tokens; truncating to %d", len(tokens), self.max_length
            )
        length = min(len(tokens), self.max_length)
        padded = np.zeros(self.max_length, dtype=np.int64)
        padded[:length] = tokens[:length]
        mask = np.arange(self.max_length) < length
        return padded, mask


@dataclass
class Observation:
    """Torch-only observation; compatible with dataclasses.replace and OpenPI preprocessing."""

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None


def make_observation(
    batch: Mapping[str, Any], tokenizer: PromptTokenizer, device: str | torch.device
) -> Observation:
    """Adapt a batched, normalized dataset record without changing its values.

    Images must already be float BCHW in [-1, 1]; state must already be padded
    to 32 dimensions. Missing camera slots must be explicit with a false mask.
    """
    state = torch.as_tensor(batch["state"])
    if state.ndim != 2 or state.shape[1] != 32 or not state.is_floating_point():
        raise ValueError("state must be a floating tensor shaped [batch, 32]")
    if not torch.isfinite(state).all():
        raise ValueError("state contains nonfinite values")
    batch_size = state.shape[0]
    if batch_size == 0:
        raise ValueError("Observation batches must not be empty")
    prompts = batch["prompt"]
    if (
        isinstance(prompts, str)
        or not isinstance(prompts, Sequence)
        or len(prompts) != batch_size
    ):
        raise ValueError("prompt must contain one string per batch element")
    images, image_masks = {}, {}
    for key in IMAGE_KEYS:
        if key not in batch["images"] or key not in batch["image_masks"]:
            raise ValueError(f"Missing image or image mask for {key}")
        image = torch.as_tensor(batch["images"][key])
        if (
            image.ndim != 4
            or image.shape[:2] != (batch_size, 3)
            or min(image.shape[2:]) < 1
            or not image.is_floating_point()
        ):
            raise ValueError(
                f"{key} must be a floating image tensor shaped [batch, 3, height, width]"
            )
        if not torch.isfinite(image).all() or image.abs().max() > 1.0001:
            raise ValueError(f"{key} must contain finite pixels normalized to [-1, 1]")
        mask = torch.as_tensor(batch["image_masks"][key])
        if mask.dtype != torch.bool or mask.shape != (batch_size,):
            raise ValueError(f"{key} image mask must be bool with shape [batch]")
        images[key] = image.to(device=device, dtype=torch.float32)
        image_masks[key] = mask.to(device=device)
    tokenized = [tokenizer.tokenize(prompt) for prompt in prompts]
    return Observation(
        images=images,
        image_masks=image_masks,
        state=state.to(device=device, dtype=torch.float32),
        tokenized_prompt=torch.as_tensor(
            np.stack([x[0] for x in tokenized]), device=device, dtype=torch.long
        ),
        tokenized_prompt_mask=torch.as_tensor(
            np.stack([x[1] for x in tokenized]), device=device, dtype=torch.bool
        ),
    )
