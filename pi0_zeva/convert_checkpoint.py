"""Strict CPU conversion of the official pi0_base Orbax checkpoint.

Only the two audited slicing functions are loaded from OpenPI's CLI script; its
training configuration imports (including LeRobot) are unnecessary here. Every
prediction weight must come from the checkpoint. The one unused expert language
head has no JAX counterpart and is explicitly zeroed, never randomly retained.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from pi0_zeva.runtime import configure_openpi, inspect_dependency_environment


_CONVERTER_SHA256 = "62980fda1a1b1a09167d413232a2a9295db8370e52b58a70cfba52ef5d88c440"
_GEMMA_SHA256 = "225092e06509da0ae37bb9dca55c00d2804255bb1496f7c8700304b4c52a95eb"
_EMBEDDING = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
_TIED_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
_UNUSED_HEAD = "paligemma_with_expert.gemma_expert.lm_head.weight"
_PROJECTIONS = (
    "state_proj",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class _SourceArray(np.ndarray):
    """Carry a source leaf through the official NumPy reshape/transpose code."""

    def __new__(cls, value, source):
        result = np.asarray(value).view(cls)
        result.source = source
        return result

    def __array_finalize__(self, parent):
        self.source = getattr(parent, "source", None)


def _from_numpy(value):
    tensor = torch.from_numpy(value)
    tensor._pi0_source = getattr(value, "source", None)
    if tensor._pi0_source is None:
        raise ValueError("Conversion lost the checkpoint source of a tensor")
    return tensor


def load_slicers(root: Path):
    path = root / "examples/convert_jax_model_to_pytorch.py"
    if sha256(path) != _CONVERTER_SHA256:
        raise ValueError(f"Unreviewed OpenPI conversion script: {path}")
    # Do not execute imports or CLI code from the upstream example. The full
    # source hash fixes the exact NumPy mappings whose definitions are selected.
    names = {"slice_paligemma_state_dict", "slice_gemma_state_dict"}
    definitions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(definitions) != len(names):
        raise ValueError("Official conversion helpers were not found")
    namespace = {"torch": SimpleNamespace(Tensor=torch.Tensor, from_numpy=_from_numpy)}
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["slice_paligemma_state_dict"], namespace["slice_gemma_state_dict"]


def validate_and_complete(model, tensors: dict) -> dict:
    """Reject missing/extra/invalid prediction weights before strict loading."""
    expected = model.state_dict()
    origins = {
        name: getattr(value, "_pi0_source", "provided")
        for name, value in tensors.items()
    }
    if _TIED_HEAD in expected and _TIED_HEAD not in tensors and _EMBEDDING in tensors:
        # A tied alias is a checkpoint weight, not an initialization exception.
        if expected[_TIED_HEAD].data_ptr() != expected[_EMBEDDING].data_ptr():
            raise ValueError("PaliGemma's language head is unexpectedly untied")
        tensors[_TIED_HEAD] = tensors[_EMBEDDING]
        origins[_TIED_HEAD] = origins[_EMBEDDING]
    initialized = []
    if _UNUSED_HEAD in expected and _UNUSED_HEAD not in tensors:
        tensors[_UNUSED_HEAD] = torch.zeros_like(expected[_UNUSED_HEAD])
        origins[_UNUSED_HEAD] = "explicit_zero: unused language head, absent from JAX"
        initialized.append(_UNUSED_HEAD)
    missing = sorted(expected.keys() - tensors.keys())
    unexpected = sorted(tensors.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(
            f"Incomplete conversion: missing={missing}, unexpected={unexpected}"
        )
    coverage = {}
    for name, tensor in tensors.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(
                f"Shape mismatch for {name}: {tensor.shape} != {expected[name].shape}"
            )
        if tensor.device.type != "cpu" or not torch.isfinite(tensor).all().item():
            raise ValueError(f"Non-finite or non-CPU tensor: {name}")
        coverage[name] = {
            "source": origins[name],
            "shape": list(tensor.shape),
            "source_dtype": str(tensor.dtype),
            "saved_dtype": str(expected[name].dtype),
        }
    if _TIED_HEAD in tensors and not torch.equal(
        tensors[_TIED_HEAD], tensors[_EMBEDDING]
    ):
        raise ValueError("Tied PaliGemma head differs from the checkpoint embedding")
    model.load_state_dict(tensors, strict=True)
    # Verify exact source values after the intentional per-parameter dtype cast.
    for name, actual in model.state_dict().items():
        if not torch.isfinite(actual).all().item():
            raise ValueError(f"Non-finite tensor after dtype conversion: {name}")
        if not torch.equal(actual, tensors[name].to(dtype=actual.dtype)):
            raise ValueError(f"Loaded tensor differs from converted checkpoint: {name}")
    return {
        "coverage": coverage,
        "explicitly_initialized_unused": initialized,
        "missing_prediction_weights": [],
        "unexpected_weights": [],
        "finite": True,
        "source_values_equal_after_dtype_cast": True,
    }


def convert(
    checkpoint_dir: Path, output_dir: Path, openpi_root: str, precision: str
) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError(
            "Run conversion with CUDA_VISIBLE_DEVICES='' to reserve GPUs for training"
        )
    os.environ["JAX_PLATFORMS"] = "cpu"
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    checkpoint_dir, output_dir = checkpoint_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite conversion output: {output_dir}")
    params_dir = checkpoint_dir / "params"
    for filename in ("_METADATA", "manifest.ocdbt"):
        if not (params_dir / filename).is_file():
            raise FileNotFoundError(f"Not an Orbax checkpoint root: {checkpoint_dir}")
    source = configure_openpi(openpi_root)
    root = Path(source["root"])
    if sha256(root / "src/openpi/models_pytorch/gemma_pytorch.py") != _GEMMA_SHA256:
        raise ValueError(
            "Unreviewed OpenPI expert model; recheck unused language-head assumption"
        )
    readiness = inspect_dependency_environment()
    if not readiness["ready"]:
        raise RuntimeError(f"OpenPI environment is not ready: {readiness}")
    slice_paligemma, slice_gemma = load_slicers(root)
    from flax.nnx import traversals
    import safetensors
    import safetensors.torch
    from openpi.models import gemma, model as jax_model
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    print("Restoring all Orbax arrays on CPU as float32", flush=True)
    restored = jax_model.restore_params(
        params_dir, restore_type=np.ndarray, dtype="float32"
    )
    required_roots = {"PaliGemma", *_PROJECTIONS}
    if set(restored) != required_roots:
        raise ValueError(
            f"Unexpected pi0 parameter groups: {set(restored) ^ required_roots}"
        )
    flat = traversals.flatten_mapping(restored["PaliGemma"], sep="/")
    traced = {
        key: _SourceArray(value, f"params/PaliGemma/{key}")
        for key, value in flat.items()
    }
    config = Pi0Config(
        action_dim=32,
        action_horizon=32,
        pi05=False,
        pytorch_compile_mode=None,
        dtype=precision,
    )
    print("Building the official CPU model for shape and alias validation", flush=True)
    policy = PI0Pytorch(config)
    paligemma, expert = slice_paligemma(
        traced, policy.paligemma_with_expert.paligemma.config
    )
    expert = slice_gemma(
        expert,
        gemma.get_config("gemma_300m"),
        num_expert=1,
        checkpoint_dir="pi0_base",
        pi05=False,
    )
    tensors = {**paligemma, **expert}
    for name in _PROJECTIONS:
        if set(restored[name]) != {"kernel", "bias"}:
            raise ValueError(f"Unknown projection parameters: {name}")
        for key, target in (("kernel", "weight"), ("bias", "bias")):
            value = _SourceArray(restored[name][key], f"params/{name}/{key}")
            tensors[f"{name}.{target}"] = _from_numpy(
                value.T if key == "kernel" else value
            )
    print(
        "Checking every prediction weight, shape, finite value and loaded value",
        flush=True,
    )
    report = validate_and_complete(policy, tensors)
    report.update(
        {
            "source_checkpoint": str(checkpoint_dir),
            "openpi": source,
            "converter_source_sha256": _CONVERTER_SHA256,
            "expert_source_sha256": _GEMMA_SHA256,
            "source_metadata_sha256": {
                name: sha256(params_dir / name)
                for name in ("_METADATA", "manifest.ocdbt", "_CHECKPOINT_METADATA")
                if (params_dir / name).is_file()
            },
            "parameter_count": sum(p.numel() for p in policy.parameters()),
            "dtype_parameter_counts": dict(
                Counter(
                    {
                        dtype: sum(
                            p.numel()
                            for p in policy.parameters()
                            if str(p.dtype) == dtype
                        )
                        for dtype in {str(p.dtype) for p in policy.parameters()}
                    }
                )
            ),
            "precision": precision,
            "precision_note": "Official mixed dtype: selected norms, vision embeddings and action projections stay float32",
            "source_array_count": len(flat) + 2 * len(_PROJECTIONS),
        }
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        path = temporary / "model.safetensors"
        print("Saving and reading back all safetensors entries", flush=True)
        safetensors.torch.save_model(policy, str(path))
        expected = policy.state_dict()
        with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
            saved_keys = set(handle.keys())
            aliases = handle.metadata() or {}
            for name, tensor in expected.items():
                saved_name = name if name in saved_keys else aliases.get(name)
                if saved_name not in saved_keys or not torch.equal(
                    handle.get_tensor(saved_name), tensor
                ):
                    raise ValueError(f"Saved tensor failed exact readback: {name}")
        report["saved_weights_equal"] = True
        report["model_sha256"] = sha256(path)
        report["model_bytes"] = path.stat().st_size
        (temporary / "conversion_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        (temporary / "config.json").write_text(
            json.dumps(
                {
                    "action_dim": 32,
                    "action_horizon": 32,
                    "pi05": False,
                    "paligemma_variant": "gemma_2b",
                    "action_expert_variant": "gemma_300m",
                    "precision": precision,
                    "preserve_official_mixed_dtype": True,
                },
                indent=2,
            )
            + "\n"
        )
        if (checkpoint_dir / "assets").is_dir():
            shutil.copytree(checkpoint_dir / "assets", temporary / "assets")
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"Strict conversion complete: {output_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openpi-root", default="../openpi-3d-tactile")
    parser.add_argument(
        "--precision", choices=("bfloat16", "float32"), default="bfloat16"
    )
    arguments = parser.parse_args()
    convert(
        arguments.checkpoint_dir,
        arguments.output_dir,
        arguments.openpi_root,
        arguments.precision,
    )


if __name__ == "__main__":
    main()
