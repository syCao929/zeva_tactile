# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stage 1: train the Causal Transition Encoder (CTE).

The release ships the CTE model and its loss but no training loop —
``causal_transition_encoder_loss`` has exactly one caller in the whole tree, a
unit test. This is that loop.

Deliberately a **standalone script** rather than a recipe in the main trainer:

- the CTE is ~9.5M params; it needs no FSDP, no sequence packing, no CP;
- the loss wants ``semantic_ids``, which the framework's ``training_step`` does
  not produce;
- the framework's ``net`` / ``net_ema`` conventions would collide with the CTE's
  own EMA target encoder (``update_ema_target``).

Precision is not a style choice here:

- **Parameters stay fp32.** ``ema_decay=0.996`` lerps at 0.004; in bf16 that
  rounds away entirely and the EMA target never moves.
- **The forward runs under bf16 autocast**, matching inference
  (``action_policy_server_robocasa365_zeva.py:751-752``).
- **The loss runs in fp32.** ``effect_temperature=0.07`` / ``temperature=0.1``
  put the contrastive logits at ±14; bf16 carries ~3 decimal digits, which is
  not enough for the NCE / clustering / VICReg terms.

The checkpoint is written in the exact shape the inference server reads
(``action_policy_server_robocasa365_zeva.py:667-682``)::

    {"model_config": <CausalTransitionEncoderConfig as dict>,
     "model":        <state_dict, run through normalize_cte_state_dict>}

Note the server loads with ``weights_only=False`` and derives ``use_mamba`` from
the state dict's key names, and its ``load_state_dict`` is strict — so the saved
dict must be *complete* (``target_visual_encoder.*`` and the
``frozen_effect_target.projection`` buffer included) and must not carry DDP or
``torch.compile`` prefixes.

Usage::

    PYTHONPATH=. python -m cosmos_framework.zeva_training.train_cte \\
        --cache-dir "$ZEVA_WORK/datasets/xhand_cte_cache" \\
        --output    "$ZEVA_WORK/runs/zeva_cte" \\
        [--resume] [--batch-size 8] [--steps 500]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cosmos_framework.model.zeva import (
    CTELossConfig,
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    causal_transition_encoder_loss,
)
from cosmos_framework.model.zeva.checkpoint_io import normalize_cte_state_dict
from cosmos_framework.zeva_training.cte_dataset import CTECacheWindowDataset, collate_cte

# Must match the Wan VAE's latent channel count — measured, not assumed; see
# probe_vae (the encoder's docstring is stale and the server asserts at runtime).
IMAGE_CHANNELS = 48
ACTION_DIM = 18


def build_encoder() -> CausalTransitionEncoder:
    cfg = CausalTransitionEncoderConfig(
        action_dim=ACTION_DIM,
        image_channels=IMAGE_CHANNELS,
        # transition_steps / effect_window_transitions / effect_target_grid are
        # coupled to the checkpoint layout and the effect-window cadence; leave
        # them at their defaults.
    )
    return CausalTransitionEncoder(cfg)


def save_cte(model: CausalTransitionEncoder, path: Path) -> None:
    payload = {
        "model_config": model.cfg.to_dict(),
        # Idempotent key-name compatibility shim; a no-op for current names.
        "model": normalize_cte_state_dict(model.state_dict()),
    }
    for key, value in payload["model"].items():
        if key.startswith(("module.", "_orig_mod.")):
            raise RuntimeError(f"state dict carries a wrapper prefix ({key}); the server loads strict")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True, help="output directory for checkpoints and logs")
    ap.add_argument("--window-latents", type=int, default=17)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="continue from <output>/cte_latest.pt")
    ap.add_argument("--device", default="cuda")
    # Effect-loss weights. All default to None = keep CTELossConfig's own value, so
    # omitting them reproduces the published behaviour exactly.
    #
    # They exist because `effect_variance` (the VICReg term that is supposed to stop
    # `effect_post_raw` collapsing) contributes only ~1/48 of the gradient norm that
    # `effect_contrastive` does, so the injected code's scale sits wherever it lands
    # instead of at the target std of 1. Measured on the 3000-step checkpoint, with
    # every entry of each code centred: contrastive grad norm 6.64, variance 0.137.
    ap.add_argument("--effect-diversity-weight", type=float, default=None,
                    help="penalty on mean pairwise cosine of the injected codes; 0 = published behaviour")
    ap.add_argument("--effect-variance-weight", type=float, default=None)
    ap.add_argument("--effect-covariance-weight", type=float, default=None)
    ap.add_argument("--effect-contrastive-weight", type=float, default=None)
    ap.add_argument("--effect-align-weight", type=float, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    latest_path = out_dir / "cte_latest.pt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    train_ds = CTECacheWindowDataset(
        args.cache_dir, window_latents=args.window_latents, split="train", val_ratio=args.val_ratio
    )
    val_ds = CTECacheWindowDataset(
        args.cache_dir, window_latents=args.window_latents, split="val", val_ratio=args.val_ratio
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_cte, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_cte,
    )

    model = build_encoder().to(args.device)
    # Keep the module in fp32; only the forward is autocast (see module docstring).
    model.float()
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    if args.resume and latest_path.is_file():
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
        log(f"resumed from {latest_path} at step {start_step}")
    elif args.resume:
        log(f"--resume given but {latest_path} does not exist; starting fresh")

    log(
        f"cache={args.cache_dir} window_latents={args.window_latents} "
        f"train_windows={len(train_ds)} val_windows={len(val_ds)} "
        f"params={n_params/1e6:.2f}M device={args.device} start_step={start_step}"
    )

    loss_cfg = CTELossConfig()
    for _flag, _field in (
        (args.effect_diversity_weight, "effect_diversity_weight"),
        (args.effect_variance_weight, "effect_variance_weight"),
        (args.effect_covariance_weight, "effect_covariance_weight"),
        (args.effect_contrastive_weight, "effect_contrastive_weight"),
        (args.effect_align_weight, "effect_align_weight"),
    ):
        if _flag is not None:
            setattr(loss_cfg, _field, _flag)
    log(
        "loss weights: " + "  ".join(
            f"{f}={getattr(loss_cfg, f):g}"
            for f in ("effect_contrastive_weight", "effect_diversity_weight",
                      "effect_variance_weight", "effect_covariance_weight",
                      "effect_align_weight")
        )
    )

    def run_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frames = batch["frames"].to(args.device, non_blocking=True)
        actions = batch["transition_actions"].to(args.device, non_blocking=True)
        valid = batch["valid_mask"].to(args.device, non_blocking=True)
        tv = batch["transition_valid"].to(args.device, non_blocking=True)
        sids = batch["semantic_ids"].to(args.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            outputs = model(frames, actions, valid, tv)
        # Loss in fp32 — the contrastive temperatures make bf16 unusable here.
        outputs = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in outputs.items()}
        return causal_transition_encoder_loss(outputs, actions, valid, sids, loss_cfg)

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        model.eval()
        agg: dict[str, float] = {}
        n = 0
        for batch in val_loader:
            losses = run_batch(batch)
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
        model.train()
        return {k: v / max(n, 1) for k, v in agg.items()}

    model.train()
    step = start_step
    t_start = time.perf_counter()
    data_iter = iter(train_loader)

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            continue

        losses = run_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        # Once per optimizer step, after the weights move and before the next
        # target is built (causal_transition_encoder.py:243-247).
        model.update_ema_target()

        step += 1
        if step % args.log_every == 0 or step == start_step + 1:
            el = time.perf_counter() - t_start
            done = step - start_step
            log(
                f"step {step}/{args.steps}  total={float(losses['total']):.4f} "
                f"action={float(losses['action']):.4f} vision={float(losses['vision']):.4f} "
                f"task={float(losses['loss_task']):.4f} phase={float(losses['loss_phase']):.4f} "
                f"effect={float(losses['loss_effect']):.4f} "
                f"(contrast={float(losses['effect_contrastive']):.3f} "
                f"var={float(losses['effect_variance']):.3f})  "
                f"{el/done:.2f}s/step"
            )

        if step % args.save_every == 0 or step == args.steps:
            ckpt = out_dir / f"cte_step_{step:06d}.pt"
            save_cte(model, ckpt)
            payload = {"model_config": model.cfg.to_dict(),
                       "model": normalize_cte_state_dict(model.state_dict()),
                       "optimizer": optimizer.state_dict(), "step": step}
            torch.save(payload, latest_path)
            val = evaluate()
            log(f"  saved {ckpt.name}  val_total={val.get('total', float('nan')):.4f}")
            with (out_dir / "metrics.jsonl").open("a") as fh:
                fh.write(json.dumps({"step": step, **{k: float(v) for k, v in val.items()}}) + "\n")

    log(f"done. final checkpoint: {out_dir / f'cte_step_{step:06d}.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
