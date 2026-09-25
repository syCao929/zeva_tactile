"""Explicit, serializable experiment configuration for the π0 comparison."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path

from pi0_zeva.camera import CAMERA_CONTRACT, CAMERAS, require_policy_camera

WORKSPACE = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    mode: str = "baseline"
    camera_contract: str = CAMERA_CONTRACT
    camera_mapping: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(CAMERAS)
    )
    data_root: str = "datasets/press_button_4_times_merged_filtered"
    norm_stats: str = "datasets/pi0_xhand_norm.json"
    openpi_root: str = "../openpi-3d-tactile"
    tokenizer_path: str = "../hf_weight/paligemma_tokenizer.model"
    pretrained: str = "models/pi0_base_pytorch/model.safetensors"
    # Stage 2 must start from a π0 XHand baseline, never from a Cosmos checkpoint.
    init_checkpoint: str | None = None
    feature_cache: str | None = None
    tactile_checkpoint: str | None = None
    output_dir: str = "runs/pi0/action_xhand/v2-threeview-joint18"
    horizon: int = 32
    tactile_memory_steps: int = 30
    fps: float = 15.0
    seed: int = 42
    split_seed: int = 42
    split_val_ratio: float = 0.03
    batch_size: int = 1
    grad_accum: int = 14
    num_workers: int = 2
    max_steps: int = 5000
    learning_rate: float = 2.5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    prior_loss_weight: float = 0.01
    backbone_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 20
    save_every: int = 500
    keep_checkpoints: int = 2

    def __post_init__(self) -> None:
        require_policy_camera(self.as_dict(), "training configuration")
        if self.mode not in {"baseline", "zeva", "zeva_tactile"}:
            raise ValueError("mode must be baseline, zeva or zeva_tactile")
        for name in (
            "horizon",
            "tactile_memory_steps",
            "batch_size",
            "grad_accum",
            "max_steps",
            "log_every",
            "eval_every",
            "eval_batches",
            "save_every",
            "keep_checkpoints",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.horizon != 32 or self.fps != 15.0:
            raise ValueError("This comparison uses 32 action steps at 15 Hz")
        if self.tactile_memory_steps != 30:
            raise ValueError("The initial π0 tactile bridge uses a 30-frame BIT window")
        if self.num_workers < 0 or self.warmup_steps < 0:
            raise ValueError("num_workers and warmup_steps must be nonnegative")
        if not 0 < self.split_val_ratio < 1:
            raise ValueError("split_val_ratio must be between 0 and 1")
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "prior_loss_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0,1]")
        if self.backbone_dtype not in {"bfloat16", "float32"}:
            raise ValueError("backbone_dtype must be bfloat16 or float32")
        if self.mode != "baseline" and (
            not self.feature_cache or not self.init_checkpoint
        ):
            raise ValueError(
                "Zeva requires feature_cache and a trained π0 init_checkpoint"
            )
        if self.mode == "zeva_tactile" and not self.tactile_checkpoint:
            raise ValueError("zeva_tactile requires tactile_checkpoint")
        if self.mode == "baseline" and (self.feature_cache or self.tactile_checkpoint):
            raise ValueError("baseline cannot consume Zeva or tactile features")

    def path(self, name: str) -> Path:
        value = getattr(self, name)
        if value is None:
            raise ValueError(f"{name} is not configured")
        expanded = os.path.expandvars(os.path.expanduser(value))
        if "$" in expanded:
            raise ValueError(f"Unresolved environment variable in {name}: {value}")
        path = Path(expanded)
        return path.resolve() if path.is_absolute() else (WORKSPACE / path).resolve()

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_config(path: str | Path, overrides: list[str] = ()) -> TrainConfig:
    def read(filename: Path, seen: frozenset[Path] = frozenset()) -> dict:
        filename = filename.resolve()
        if filename in seen:
            raise ValueError("Configuration inheritance cycle")
        data = json.loads(filename.read_text())
        parent = data.pop("extends", None)
        return (
            read(filename.parent / parent, seen | {filename}) if parent else {}
        ) | data

    values = read(Path(path))
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"Override must be key=value: {item}")
        try:
            values[key] = json.loads(value)
        except json.JSONDecodeError:
            values[key] = value
    return TrainConfig(**values)
