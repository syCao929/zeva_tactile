"""Atomic checkpoints; Stage 2 stores its memory separately from the frozen base."""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file, save_model

from pi0_zeva.camera import require_policy_camera

FORMAT_VERSION = 1


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path).resolve()
    if path.name == "latest.json":
        relative = Path(json.loads(path.read_text())["checkpoint"])
        if relative.is_absolute() or relative.name != str(relative):
            raise ValueError("latest.json must point to a sibling checkpoint directory")
        path = path.parent / relative
    metadata = json.loads((path / "manifest.json").read_text())
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported π0 Zeva checkpoint: {path}")
    require_policy_camera(metadata["config"], path)
    return path


def backbone_path(path: str | Path) -> Path:
    path = resolve_checkpoint(path)
    metadata = json.loads((path / "manifest.json").read_text())
    result = (path / metadata["backbone"]).resolve()
    if not result.is_file():
        raise FileNotFoundError(f"Checkpoint's frozen backbone is missing: {result}")
    return result


def pin_backbone(source: str | Path, output_dir: str | Path) -> Path:
    """Keep the baseline alive even if its original run prunes old checkpoints."""
    source = Path(source).resolve()
    destination = Path(output_dir) / "base_backbone.safetensors"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace an existing pinned base: {destination}"
        )
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    return destination


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("A CUDA training checkpoint requires CUDA to resume")
        torch.cuda.set_rng_state(state["cuda"].cpu())


def save_checkpoint(
    policy,
    optimizer,
    output_dir: str | Path,
    *,
    step: int,
    config: dict,
    loader_state: dict,
    rng_states: list[dict],
    norm_sha256: str,
    artifact_hashes: dict | None = None,
) -> Path:
    require_policy_camera(config, "checkpoint being saved")
    directory = Path(output_dir) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"step_{step:08d}"
    final = directory / name
    temporary = directory / f".{name}.partial"
    if final.exists() or temporary.exists():
        raise FileExistsError(f"Checkpoint already exists or needs inspection: {final}")
    temporary.mkdir()
    stage2 = config["mode"] != "baseline"
    if stage2:
        if not (Path(output_dir) / "base_backbone.safetensors").is_file():
            raise FileNotFoundError(
                "Stage 2 requires a pinned base_backbone.safetensors"
            )
        state = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in policy.state_dict().items()
            if not key.startswith("backbone.")
        }
        save_file(state, temporary / "memory.safetensors")
        base = "../../base_backbone.safetensors"
    else:
        save_model(policy.backbone, str(temporary / "backbone.safetensors"))
        base = "backbone.safetensors"
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "loader": loader_state,
            "rng_by_rank": rng_states,
        },
        temporary / "training.pt",
    )
    metadata = {
        "format_version": FORMAT_VERSION,
        "step": step,
        "config": config,
        "world_size": len(rng_states),
        "norm_sha256": norm_sha256,
        "artifact_hashes": artifact_hashes or {},
        "backbone": base,
        "memory": "memory.safetensors" if stage2 else None,
    }
    (temporary / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.rename(final)
    latest_tmp = directory / ".latest.json.tmp"
    latest_tmp.write_text(json.dumps({"checkpoint": name}) + "\n")
    latest_tmp.replace(directory / "latest.json")
    return final


def load_memory(policy, checkpoint: str | Path) -> dict:
    directory = resolve_checkpoint(checkpoint)
    metadata = json.loads((directory / "manifest.json").read_text())
    if metadata["memory"] is not None:
        weights = load_file(str(directory / metadata["memory"]), device="cpu")
        expected = {
            key for key in policy.state_dict() if not key.startswith("backbone.")
        }
        if set(weights) != expected:
            raise ValueError(
                "Memory checkpoint keys do not match the selected π0 Zeva model"
            )
        result = policy.load_state_dict(weights, strict=False)
        if result.unexpected_keys or any(
            not key.startswith("backbone.") for key in result.missing_keys
        ):
            raise ValueError("Incomplete π0 memory checkpoint")
    elif any(not key.startswith("backbone.") for key in policy.state_dict()):
        raise ValueError(
            "Baseline checkpoint cannot resume a Stage-2 model; use init_checkpoint"
        )
    return metadata


def prune_checkpoints(output_dir: str | Path, keep: int) -> None:
    if keep < 1:
        raise ValueError("At least one checkpoint must be retained")
    directory = Path(output_dir) / "checkpoints"
    checkpoints = sorted(
        path for path in directory.glob("step_[0-9]*") if path.is_dir()
    )
    for path in checkpoints[:-keep]:
        # Only remove directories created by this writer, after a new save succeeded.
        metadata = json.loads((path / "manifest.json").read_text())
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Refusing to prune an unknown checkpoint: {path}")
        shutil.rmtree(path)
