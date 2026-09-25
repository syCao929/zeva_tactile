"""π0 XHand baseline and Zeva Stage-2 training, independent of Cosmos trainers."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time

from pi0_zeva.config import WORKSPACE, TrainConfig, load_config


class BatchStream:
    """A resumable, deterministic distributed sampler with an explicit epoch cursor."""

    def __init__(self, loader, sampler, *, epoch: int = 0, offset: int = 0):
        if not len(loader):
            raise ValueError(
                "Training loader is empty; reduce batch_size or check the split"
            )
        if not 0 <= offset <= len(loader):
            raise ValueError("Invalid checkpoint dataloader offset")
        self.loader, self.sampler = loader, sampler
        self.epoch, self.offset = epoch, offset
        self.iterator = None

    def _start(self):
        self.sampler.set_epoch(self.epoch)
        if self.loader.generator is not None:
            self.loader.generator.manual_seed(self.sampler.seed + self.epoch)
        self.iterator = iter(self.loader)
        for _ in range(self.offset):
            next(self.iterator)

    def __next__(self):
        if self.iterator is None:
            self._start()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.offset = 0
            self._start()
            batch = next(self.iterator)
        self.offset += 1
        return batch

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "offset": self.offset}


def learning_rate_at_step(config: TrainConfig, step: int) -> float:
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(
        1, config.max_steps - config.warmup_steps
    )
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return config.learning_rate * (
        config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine
    )


def memory_kwargs(batch: dict, device) -> dict:
    keys = (
        "behavior_global",
        "behavior_phase",
        "behavior_effect",
        "behavior_effect_valid",
    )
    result = {}
    if any(key in batch for key in keys):
        if not all(key in batch for key in keys):
            raise ValueError("Incomplete Zeva behavior inputs")
        result["behavior"] = {key: batch[key].to(device) for key in keys}
    if "tactile_state" in batch:
        result["tactile_state"] = batch["tactile_state"].to(device)
        result["tactile_valid"] = batch["tactile_valid"].to(device)
    return result


def evaluate(policy, loader, tokenizer, device, *, max_batches: int, seed: int) -> dict:
    """Repeatable action-flow validation, without training augmentations or RNG drift."""
    import torch

    from pi0_zeva.runtime import make_observation

    was_training = policy.training
    policy.eval()
    totals = {key: 0.0 for key in ("loss", "action_flow_loss", "prior_nll")}
    count = 0
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            for index, batch in enumerate(loader):
                if index >= max_batches:
                    break
                actions = batch["actions"].to(device)
                noise = torch.randn(actions.shape, device=device, generator=generator)
                # Fixed evaluation distribution shared by baseline and memory variants.
                times = torch.rand(actions.shape[0], device=device, generator=generator)
                times = (times * 0.998 + 0.001).clamp(0.001, 0.999)
                output = policy(
                    make_observation(batch, tokenizer, device),
                    actions,
                    noise=noise,
                    time=times,
                    **memory_kwargs(batch, device),
                )
                for key in totals:
                    value = float(output[key].detach())
                    if not math.isfinite(value):
                        raise FloatingPointError(f"Non-finite validation {key}")
                    totals[key] += value * len(actions)
                count += len(actions)
    finally:
        policy.train(was_training)
    if not count:
        raise ValueError(
            "Validation processed zero samples; refusing to report a zero loss"
        )
    return {key: value / count for key, value in totals.items()} | {"samples": count}


def _dataset(config: TrainConfig, split: str):
    from pi0_zeva.data import XHandPi0Dataset

    return XHandPi0Dataset(
        str(config.path("data_root")),
        str(config.path("norm_stats")),
        split=split,
        horizon=config.horizon,
        split_seed=config.split_seed,
        split_val_ratio=config.split_val_ratio,
        feature_cache=str(config.path("feature_cache"))
        if config.feature_cache
        else None,
        tactile=config.mode == "zeva_tactile",
        tactile_memory_steps=config.tactile_memory_steps,
    )


def _preflight(config: TrainConfig) -> dict:
    import importlib.util

    from pi0_zeva.runtime import configure_openpi, inspect_dependency_environment

    errors = []
    try:
        upstream = configure_openpi(str(config.path("openpi_root")))
    except (ImportError, ValueError, FileNotFoundError, RuntimeError) as exc:
        upstream = None
        errors.append(str(exc))
    environment = inspect_dependency_environment()
    errors.extend(environment.get("errors", []))
    for module in ("pyarrow", "av"):
        if importlib.util.find_spec(module) is None:
            errors.append(f"Missing data dependency: {module}")
    required = ["data_root", "norm_stats", "tokenizer_path"]
    required += ["init_checkpoint"] if config.init_checkpoint else ["pretrained"]
    required += [
        name
        for name in ("feature_cache", "tactile_checkpoint")
        if getattr(config, name)
    ]
    for name in required:
        if not config.path(name).exists():
            errors.append(f"Missing {name}: {config.path(name)}")
        elif (
            name in {"norm_stats", "tokenizer_path", "pretrained", "tactile_checkpoint"}
            and not config.path(name).is_file()
        ):
            errors.append(f"Expected a file for {name}: {config.path(name)}")
    if config.init_checkpoint and config.path("init_checkpoint").exists():
        from pi0_zeva.checkpoint import backbone_path

        try:
            backbone_path(config.path("init_checkpoint"))
        except (ValueError, OSError, KeyError) as exc:
            errors.append(str(exc))
    if config.feature_cache and config.path("feature_cache").exists():
        from pi0_zeva.data import index_feature_cache

        try:
            index_feature_cache(config.path("feature_cache"))
        except (ValueError, OSError, KeyError) as exc:
            errors.append(str(exc))
    return {
        "ok": not errors,
        "errors": errors,
        "openpi": upstream,
        "environment": environment,
        "config": config.as_dict(),
        "global_batch_at_8_gpus": config.batch_size * config.grad_accum * 8,
    }


def _run(config: TrainConfig, resume: str | None, eval_only: str | None) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler, Subset

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
    from pi0_zeva.model import build_policy, load_pretrained_backbone
    from pi0_zeva.runtime import (
        PromptTokenizer,
        configure_openpi,
        inspect_dependency_environment,
        make_observation,
    )

    if str(WORKSPACE / "cosmos-framework") not in sys.path:
        sys.path.insert(0, str(WORKSPACE / "cosmos-framework"))
    configure_openpi(str(config.path("openpi_root")))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    readiness = inspect_dependency_environment()
    if not readiness["ready"]:
        raise RuntimeError(
            "OpenPI runtime is not ready: " + "; ".join(readiness["errors"])
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training requires CUDA; use --data-smoke or the CPU tests to check the code"
        )
    rank, world_size = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")

    def on_rank0(function):
        status = [None]
        if rank == 0:
            try:
                status[0] = {"result": function()}
            except Exception as exc:
                status[0] = {"error": f"{type(exc).__name__}: {exc}"}
        if world_size > 1:
            dist.broadcast_object_list(status, src=0)
        if "error" in status[0]:
            raise RuntimeError(status[0]["error"])
        return status[0]["result"]

    try:
        random.seed(config.seed + rank)
        np.random.seed(config.seed + rank)
        torch.manual_seed(config.seed + rank)
        torch.cuda.manual_seed(config.seed + rank)
        train_data, val_data = _dataset(config, "train"), _dataset(config, "val")
        if not len(train_data) or not len(val_data):
            raise ValueError("Both training and validation splits must contain windows")
        tokenizer = PromptTokenizer(str(config.path("tokenizer_path")))
        output_dir = config.path("output_dir")
        norm_sha256 = hashlib.sha256(config.path("norm_stats").read_bytes()).hexdigest()
        artifact_hashes = {
            "tokenizer": hashlib.sha256(
                config.path("tokenizer_path").read_bytes()
            ).hexdigest()
        }
        if config.feature_cache:
            manifest = config.path("feature_cache") / "manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"CTE cache requires its provenance manifest: {manifest}"
                )
            artifact_hashes["cte_manifest"] = hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest()
        if config.tactile_checkpoint:
            artifact_hashes["tactile_encoder"] = hashlib.sha256(
                config.path("tactile_checkpoint").read_bytes()
            ).hexdigest()
        restored = (
            resolve_checkpoint(resume or eval_only) if (resume or eval_only) else None
        )
        if resume and output_dir != restored.parent.parent.resolve():
            raise ValueError(
                "--resume must use the original run's output_dir; cross-run resume is unsupported"
            )
        metadata = (
            json.loads((restored / "manifest.json").read_text()) if restored else None
        )
        if metadata:
            if metadata["norm_sha256"] != norm_sha256:
                raise ValueError(
                    "Checkpoint normalization differs from the current stats file"
                )
            if metadata.get("artifact_hashes", {}) != artifact_hashes:
                raise ValueError(
                    "Checkpoint tokenizer/CTE/tactile input artifacts have changed"
                )
            saved_config = metadata["config"]
            mutable = {
                "output_dir",
                "log_every",
                "eval_every",
                "eval_batches",
                "save_every",
                "keep_checkpoints",
                "num_workers",
            }
            if eval_only:
                mutable |= {
                    "batch_size",
                    "grad_accum",
                    "max_steps",
                    "warmup_steps",
                    "learning_rate",
                    "weight_decay",
                }
            changed = [
                key
                for key, value in config.as_dict().items()
                if key not in mutable and saved_config.get(key) != value
            ]
            if changed:
                raise ValueError(f"Checkpoint configuration mismatch: {changed}")
            if resume and metadata["world_size"] != world_size:
                raise ValueError("Exact resume requires the original world size")
        initialization = (
            resolve_checkpoint(config.path("init_checkpoint"))
            if config.init_checkpoint and not restored
            else None
        )
        if initialization:
            initial = json.loads((initialization / "manifest.json").read_text())
            if (
                initial["config"]["mode"] != "baseline"
                or initial["norm_sha256"] != norm_sha256
            ):
                raise ValueError(
                    "Stage 2 must initialize from a π0 baseline with identical normalization"
                )
            if initial["config"]["horizon"] != config.horizon:
                raise ValueError("Baseline and Stage-2 horizons differ")
        source = (
            backbone_path(restored or initialization)
            if (restored or initialization)
            else config.path("pretrained")
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"PyTorch π0 weights not found: {source}; convert the JAX checkpoint first"
            )
        policy = build_policy(
            mode=config.mode,
            horizon=config.horizon,
            prior_loss_weight=config.prior_loss_weight,
            tactile_checkpoint=str(config.path("tactile_checkpoint"))
            if config.tactile_checkpoint
            else None,
            backbone_dtype=config.backbone_dtype,
        )
        load_pretrained_backbone(policy, source)
        if restored:
            load_memory(policy, restored)
        policy.to(device)
        if config.gradient_checkpointing:
            policy.backbone.gradient_checkpointing_enable()
        parameters = [
            parameter for parameter in policy.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("No trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        sampler = DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            seed=config.seed,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            multiprocessing_context="spawn" if config.num_workers else None,
            generator=torch.Generator().manual_seed(config.seed),
        )
        # A fresh iterator is created on every validation; only rank 0 evaluates.
        # Identical examples/noise across all variants, independent of training batch size.
        val_count = min(len(val_data), config.eval_batches)
        val_indices = (
            torch.linspace(0, len(val_data) - 1, steps=val_count).long().tolist()
        )
        val_loader = DataLoader(
            Subset(val_data, val_indices),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            generator=torch.Generator().manual_seed(config.seed + 10000),
        )
        stream_state, step, rng = {}, 0, None
        if resume:
            state = torch.load(
                restored / "training.pt", map_location="cpu", weights_only=False
            )
            optimizer.load_state_dict(state["optimizer"])
            stream_state, step = state["loader"], metadata["step"]
            rng = state["rng_by_rank"][rank]
        stream = BatchStream(train_loader, sampler, **stream_state)
        if eval_only:
            result = on_rank0(
                lambda: evaluate(
                    policy,
                    val_loader,
                    tokenizer,
                    device,
                    max_batches=config.eval_batches,
                    seed=config.seed + 10000,
                )
            )
            if rank == 0:
                print(
                    json.dumps(
                        {"event": "validation", "step": metadata["step"], **result}
                    ),
                    flush=True,
                )
            return

        def prepare_output():
            if resume:
                if not (output_dir / "config.json").is_file():
                    raise ValueError(
                        "Resume output_dir must be the existing run directory"
                    )
            else:
                if output_dir.exists() and any(output_dir.iterdir()):
                    raise FileExistsError(
                        f"Nonempty output directory requires --resume: {output_dir}"
                    )
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "config.json").write_text(
                    json.dumps(config.as_dict(), indent=2) + "\n"
                )
                (output_dir / "norm_stats.json").write_bytes(
                    config.path("norm_stats").read_bytes()
                )
                if config.mode != "baseline":
                    pin_backbone(source, output_dir)

        on_rank0(prepare_output)
        model = (
            DistributedDataParallel(
                policy,
                device_ids=[local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            if world_size > 1
            else policy
        )
        if rng is not None:
            restore_rng(rng)
        logfile = output_dir / "metrics.jsonl"

        def log(record):
            if rank == 0:
                record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}
                line = json.dumps(record, allow_nan=False)
                with logfile.open("a") as handle:
                    handle.write(line + "\n")
                print(line, flush=True)

        log(
            {
                "event": "start",
                "mode": config.mode,
                "camera_contract": config.camera_contract,
                "camera_mapping": config.camera_mapping,
                "feature_cache": str(config.path("feature_cache"))
                if config.feature_cache
                else None,
                "step": step,
                "world_size": world_size,
                "global_batch": config.batch_size * config.grad_accum * world_size,
                "train_windows": len(train_data),
                "val_windows": len(val_data),
                "trainable_parameters": sum(p.numel() for p in parameters),
                "backbone_source": str(source),
                "gpu_name": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
            }
        )
        if not resume:
            result = on_rank0(
                lambda: evaluate(
                    policy,
                    val_loader,
                    tokenizer,
                    device,
                    max_batches=config.eval_batches,
                    seed=config.seed + 10000,
                )
            )
            log({"event": "validation", "step": 0, **result})
        policy.train()
        while step < config.max_steps:
            started = time.monotonic()
            lr = learning_rate_at_step(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            values = torch.zeros(3, device=device)
            for microstep in range(config.grad_accum):
                batch = next(stream)
                observation = make_observation(batch, tokenizer, device)
                sync = (
                    contextlib.nullcontext()
                    if world_size == 1 or microstep == config.grad_accum - 1
                    else model.no_sync()
                )
                with sync:
                    output = model(
                        observation,
                        batch["actions"].to(device),
                        **memory_kwargs(batch, device),
                    )
                    finite = torch.isfinite(output["loss"]).to(torch.int32)
                    if world_size > 1:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not bool(finite):
                        raise FloatingPointError(
                            f"Non-finite training loss at step {step + 1}"
                        )
                    (output["loss"] / config.grad_accum).backward()
                values += (
                    torch.stack(
                        [
                            output[key].detach().float()
                            for key in ("loss", "action_flow_loss", "prior_nll")
                        ]
                    )
                    / config.grad_accum
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            if world_size > 1:
                dist.all_reduce(values)
                values /= world_size
            if step == 1 or step % config.log_every == 0:
                log(
                    {
                        "event": "train",
                        "step": step,
                        "loss": float(values[0]),
                        "action_flow_loss": float(values[1]),
                        "prior_nll": float(values[2]),
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "seconds": time.monotonic() - started,
                        "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(
                            device
                        )
                        / 1024**3,
                        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(
                            device
                        )
                        / 1024**3,
                    }
                )
            if step % config.eval_every == 0 or step == config.max_steps:
                result = on_rank0(
                    lambda: evaluate(
                        policy,
                        val_loader,
                        tokenizer,
                        device,
                        max_batches=config.eval_batches,
                        seed=config.seed + 10000,
                    )
                )
                log({"event": "validation", "step": step, **result})
            if step % config.save_every == 0 or step == config.max_steps:
                all_rng = [None] * world_size
                if world_size > 1:
                    dist.all_gather_object(all_rng, capture_rng())
                else:
                    all_rng[0] = capture_rng()

                def save():
                    path = save_checkpoint(
                        policy,
                        optimizer,
                        output_dir,
                        step=step,
                        config=config.as_dict(),
                        loader_state=stream.state_dict(),
                        rng_states=all_rng,
                        norm_sha256=norm_sha256,
                        artifact_hashes=artifact_hashes,
                    )
                    prune_checkpoints(output_dir, config.keep_checkpoints)
                    return str(path)

                path = on_rank0(save)
                log({"event": "checkpoint", "step": step, "path": path})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(WORKSPACE / "configs/pi0/xhand_baseline.json")
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--compute-stats", action="store_true")
    group.add_argument("--data-smoke", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--resume")
    group.add_argument("--eval-only")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.set)
    if args.compute_stats:
        from pi0_zeva.data import compute_normalization

        compute_normalization(
            str(config.path("data_root")),
            str(config.path("norm_stats")),
            split_seed=config.split_seed,
            split_val_ratio=config.split_val_ratio,
            horizon=config.horizon,
        )
        print(json.dumps({"norm_stats": str(config.path("norm_stats"))}))
    elif args.data_smoke:
        import torch

        def describe(value):
            if isinstance(value, torch.Tensor):
                return {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "finite": bool(torch.isfinite(value).all()),
                }
            return (
                {key: describe(item) for key, item in value.items()}
                if isinstance(value, dict)
                else value
            )

        for split in ("train", "val"):
            dataset = _dataset(config, split)
            print(
                json.dumps(
                    {
                        "split": split,
                        "windows": len(dataset),
                        "sample": describe(dataset[0]),
                    }
                )
            )
    elif args.preflight:
        result = _preflight(config)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["ok"] else 1
    else:
        _run(config, args.resume, args.eval_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
