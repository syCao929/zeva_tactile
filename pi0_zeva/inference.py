"""Shared pi0 checkpoint inference and offline XHand validation.

The predictor accepts the already normalized batch contract of ``data.py`` and
returns absolute joint positions in radians. This is not a robot client: camera
decoding, raw state normalization and online CTE state are the caller's job.
Importing this module does not import OpenPI or construct a model.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

from pi0_zeva import checkpoint as checkpoint_io
from pi0_zeva.config import TrainConfig
from pi0_zeva.data import Normalizer, STATE_INDICES, XHandPi0Dataset
from pi0_zeva.model import build_policy, load_pretrained_backbone
from pi0_zeva.runtime import (
    PromptTokenizer,
    configure_openpi,
    inspect_dependency_environment,
    make_observation,
)


class Predictor:
    def __init__(
        self,
        policy,
        tokenizer,
        normalizer: Normalizer,
        *,
        config: TrainConfig,
        checkpoint: Path,
        metadata: dict,
        stats_path: Path,
        device: str | torch.device,
    ):
        self.policy = policy.eval()
        self.policy.requires_grad_(False)
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.config = config
        self.checkpoint = checkpoint
        self.metadata = metadata
        self.stats_path = stats_path
        self.device = torch.device(device)

    @torch.inference_mode()
    def predict(
        self,
        batch: Mapping[str, Any],
        *,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode normalized batch inputs to finite absolute actions [B,32,18]."""
        if (
            not isinstance(num_steps, int)
            or isinstance(num_steps, bool)
            or num_steps < 1
        ):
            raise ValueError("num_steps must be a positive integer")
        observation = make_observation(batch, self.tokenizer, self.device)
        normalized = self.policy.sample_actions(
            observation,
            behavior=batch.get("behavior", batch),
            tactile_state=batch.get("tactile_state"),
            tactile_valid=batch.get("tactile_valid"),
            noise=noise,
            num_steps=num_steps,
        )
        expected = (observation.state.shape[0], self.config.horizon, 32)
        if (
            not isinstance(normalized, torch.Tensor)
            or tuple(normalized.shape) != expected
        ):
            raise ValueError(f"Policy must return normalized actions shaped {expected}")
        if not normalized.is_floating_point() or not torch.isfinite(normalized).all():
            raise ValueError("Policy returned nonfinite or nonfloating actions")
        actions = self.normalizer.denormalize_actions(normalized.float())
        if not torch.isfinite(actions).all():
            raise ValueError("Action denormalization produced nonfinite commands")
        return actions


def load_policy(
    checkpoint: str | Path,
    openpi_root: str | None = None,
    tokenizer_path: str | None = None,
    device: str | torch.device = "cuda",
) -> Predictor:
    """Strictly restore a saved baseline or memory model and its normalization."""
    directory = checkpoint_io.resolve_checkpoint(checkpoint)
    metadata = json.loads((directory / "manifest.json").read_text())
    config = TrainConfig(**metadata["config"])
    expected_memory = None if config.mode == "baseline" else "memory.safetensors"
    if metadata.get("memory") != expected_memory:
        raise ValueError("Checkpoint mode and memory manifest disagree")
    stats_path = directory.parent.parent / "norm_stats.json"
    stats_bytes = stats_path.read_bytes()
    if hashlib.sha256(stats_bytes).hexdigest() != metadata.get("norm_sha256"):
        raise ValueError("Run norm_stats.json does not match the checkpoint SHA256")
    normalizer = Normalizer(json.loads(stats_bytes))
    expected_provenance = {
        "state_indices": list(STATE_INDICES),
        "action_indices": list(range(18)),
        "action_representation": "absolute_joint_position",
        "horizon": config.horizon,
        "fps": config.fps,
        "split_seed": config.split_seed,
        "split_val_ratio": config.split_val_ratio,
    }
    for key, value in expected_provenance.items():
        if normalizer.provenance.get(key) != value:
            raise ValueError(f"Normalization and checkpoint config disagree for {key}")
    weights = checkpoint_io.backbone_path(directory)
    if expected_memory and not (directory / expected_memory).is_file():
        raise FileNotFoundError(directory / expected_memory)
    tactile_checkpoint = (
        config.path("tactile_checkpoint") if config.mode == "zeva_tactile" else None
    )
    if tactile_checkpoint is not None and not tactile_checkpoint.is_file():
        raise FileNotFoundError(
            f"Tactile construction requires the encoder recorded in the manifest: {tactile_checkpoint}"
        )
    tokenizer_file = Path(tokenizer_path or config.path("tokenizer_path"))
    artifacts = metadata.get("artifact_hashes", {})
    for name, path in (
        ("tokenizer", tokenizer_file),
        ("tactile_encoder", tactile_checkpoint),
    ):
        if name in artifacts and path is not None:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != artifacts[name]:
                raise ValueError(f"Checkpoint input artifact changed: {name}")
    tokenizer = PromptTokenizer(tokenizer_file, max_length=48)
    configure_openpi(openpi_root or str(config.path("openpi_root")))
    readiness = inspect_dependency_environment()
    if not readiness["ready"]:
        raise RuntimeError(
            "OpenPI runtime is not ready: " + "; ".join(readiness["errors"])
        )
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested, but CUDA is unavailable")
    policy = build_policy(
        mode=config.mode,
        horizon=config.horizon,
        prior_loss_weight=config.prior_loss_weight,
        tactile_checkpoint=tactile_checkpoint,
        backbone_dtype=config.backbone_dtype,
    )
    load_pretrained_backbone(policy, weights)
    checkpoint_io.load_memory(policy, directory)
    policy.to(device=device)
    return Predictor(
        policy,
        tokenizer,
        normalizer,
        config=config,
        checkpoint=directory,
        metadata=metadata,
        stats_path=stats_path,
        device=device,
    )


def evaluate_dataset(
    predictor: Predictor, dataset, *, max_samples=32, num_steps=10, seed=42
) -> dict:
    """Uniform validation windows; deterministic initial noise; metrics in radians."""
    if max_samples < 1 or len(dataset) < 1:
        raise ValueError("Evaluation requires a nonempty dataset and max_samples > 0")
    indices = np.linspace(
        0, len(dataset) - 1, min(max_samples, len(dataset)), dtype=np.int64
    )
    generator = torch.Generator(device=predictor.device).manual_seed(seed)
    sums = {"arm": [0.0, 0.0, 0], "hand": [0.0, 0.0, 0]}
    for index in indices:
        batch = default_collate([dataset[int(index)]])
        noise = torch.randn(
            (1, predictor.config.horizon, 32),
            generator=generator,
            device=predictor.device,
        )
        prediction = (
            predictor.predict(batch, num_steps=num_steps, noise=noise).cpu().double()
        )
        target = (
            predictor.normalizer.denormalize_actions(batch["actions"]).cpu().double()
        )
        if target.shape != prediction.shape or not torch.isfinite(target).all():
            raise ValueError(
                "Validation target has an invalid shape or nonfinite values"
            )
        error = prediction - target
        for name, part in (("arm", error[..., :6]), ("hand", error[..., 6:18])):
            sums[name][0] += part.abs().sum().item()
            sums[name][1] += part.square().sum().item()
            sums[name][2] += part.numel()
    metrics = {}
    for name, (absolute, squared, count) in sums.items():
        metrics[f"{name}_mae_rad"] = absolute / count
        metrics[f"{name}_rmse_rad"] = (squared / count) ** 0.5
    return {
        "checkpoint": str(predictor.checkpoint),
        "step": predictor.metadata["step"],
        "mode": predictor.config.mode,
        "split": "val",
        "seed": seed,
        "num_steps": num_steps,
        "sample_count": len(indices),
        "sample_indices": indices.tolist(),
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--openpi-root")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.max_samples < 1 or args.num_steps < 1:
        parser.error("--max-samples and --num-steps must be positive")
    torch.manual_seed(args.seed)
    predictor = load_policy(
        args.checkpoint, args.openpi_root, args.tokenizer_path, args.device
    )
    config = predictor.config
    if config.feature_cache:
        expected = predictor.metadata.get("artifact_hashes", {}).get("cte_manifest")
        manifest = config.path("feature_cache") / "manifest.json"
        if expected and hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
            raise ValueError("CTE feature cache differs from the one used for training")
    dataset = XHandPi0Dataset(
        config.path("data_root"),
        predictor.stats_path,
        split="val",
        horizon=config.horizon,
        feature_cache=config.path("feature_cache")
        if config.mode != "baseline"
        else None,
        tactile=config.mode == "zeva_tactile",
        tactile_memory_steps=config.tactile_memory_steps,
        split_seed=config.split_seed,
        split_val_ratio=config.split_val_ratio,
    )
    result = evaluate_dataset(
        predictor,
        dataset,
        max_samples=args.max_samples,
        num_steps=args.num_steps,
        seed=args.seed,
    )
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
