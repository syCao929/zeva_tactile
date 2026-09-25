"""Read-only CPU audit of the retained CTE v4 and its data (writes this audit's JSON only)."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from cosmos_framework.zeva_training.cte_dataset import CTECacheWindowDataset
from cosmos_framework.zeva_training.cte_features import load_cte
from cosmos_framework.model.zeva import CTELossConfig, causal_transition_encoder_loss

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "datasets/xhand_cte_cache_v2"
FEATURES = ROOT / "datasets/xhand_cte_features_v4"
RUN = ROOT / "runs/zeva_cte/cte-v4-20260924"
OUT = Path(__file__).with_suffix(".json")
torch.set_num_threads(2)
torch.manual_seed(42)
report = {"device": "cpu", "forward_dtype": "float32", "checks": {}}


def save():
    OUT.write_text(json.dumps(report, indent=2) + "\n")


def geometry(x):
    x = x.float()
    z = F.normalize(x, dim=-1)
    n = len(z)
    mean_cos = (z.sum(0).square().sum() - z.square().sum()) / (n * (n - 1))
    sv = torch.linalg.svdvals(x - x.mean(0))
    p = sv / sv.sum().clamp_min(1e-12)
    return {"count": n, "mean_pairwise_cosine": float(mean_cos),
            "centered_singular_entropy_rank": float((-(p * p.clamp_min(1e-12).log()).sum()).exp()),
            "mean_dimension_std": float(x.std(0).mean())}


train = CTECacheWindowDataset(CACHE, split="train")
val = CTECacheWindowDataset(CACHE, split="val")
train_ids = {e["episode_id"] for e in train.episodes}
val_ids = {e["episode_id"] for e in val.episodes}
assert not train_ids & val_ids
report["split"] = {"train_episodes": len(train_ids), "val_episodes": sorted(val_ids),
                   "train_windows": len(train), "val_windows": len(val), "overlap": []}
manifest = json.loads((CACHE / "manifest.json").read_text())
data_root = Path(manifest["dataset_root"]) / "xhand/PressButton4Times/lerobot_v21/lerobot"
val_data = {}
phase_samples, effect_samples = [], []
finite, actions_match, masks_match = True, True, True
total_frames = 0
for entry in manifest["episodes"]:
    ep = entry["episode_id"]
    with np.load(CACHE / entry["file"]) as a, np.load(FEATURES / f"features_{ep:06d}.npz") as b:
        lat, act = a["latents"], a["actions"]
        t = lat.shape[1]
        assert lat.shape == (48, 1 + (len(act) - 1) // 4, 30, 52)
        assert act.shape == (int(a["length"]), 18)
        assert t == entry["latent_frames"] and len(act) == entry["raw_frames"]
        finite &= bool(np.isfinite(lat).all() and np.isfinite(act).all())
        raw = pq.read_table(data_root / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet", columns=["action"])
        expected = np.asarray(raw["action"].to_pylist(), dtype=np.float32)
        actions_match &= np.array_equal(act, expected)
        assert np.array_equal(b["boundary_frame"], np.arange(t) * 4)
        ev = b["effect_valid"]
        want = np.arange(4)[None, :] >= 4 - (np.minimum(np.arange(t), 16) // 4)[:, None]
        masks_match &= np.array_equal(ev, want)
        assert b["phase"].shape == (t, 128) and b["effect"].shape == (t, 4, 128)
        finite &= bool(np.isfinite(b["phase"]).all() and np.isfinite(b["effect"]).all())
        assert np.count_nonzero(b["effect"][~ev]) == 0
        phase_samples.append(torch.from_numpy(b["phase"][16::16].astype(np.float32)))
        effect_samples.append(torch.from_numpy(b["effect"][16::16, -1].astype(np.float32)))
        total_frames += t
        if ep in val_ids:
            val_data[entry["file"]] = (torch.from_numpy(lat).float(), torch.from_numpy(act).float())
report["checks"].update(all_cached_values_finite=finite, all_raw_actions_match_source=actions_match,
                         all_feature_masks_match_cadence=masks_match, latent_frames=total_frames)
report["cached_features_spaced_16_boundaries"] = {"phase": geometry(torch.cat(phase_samples)),
                                                 "effect_post": geometry(torch.cat(effect_samples))}
print("Data scan:", json.dumps(report), flush=True)
save()


def batch_from_windows(windows):
    frames, actions = [], []
    for filename, start in windows:
        lat, act = val_data[filename]
        frames.append(lat[:, start:start + 17].permute(1, 0, 2, 3))
        actions.append(act[4 * start:4 * start + 64].reshape(16, 4, 18))
    return torch.stack(frames), torch.stack(actions)


model, cfg = load_cte(RUN / "cte_step_003000.pt", "cpu")
latest = torch.load(RUN / "cte_latest.pt", map_location="cpu", weights_only=False)
report["checkpoint"] = {"config": cfg.to_dict(), "step": latest["step"],
    "all_weights_match_latest": all(torch.equal(v, latest["model"][k]) for k, v in model.state_dict().items()),
    "all_weights_finite": all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),
    "optimizer": [{k: v for k, v in g.items() if k != "params"} for g in latest["optimizer"]["param_groups"]],
    "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
    "optimizer_step_range": [min(float(s["step"]) for s in latest["optimizer"]["state"].values()),
                             max(float(s["step"]) for s in latest["optimizer"]["state"].values())]}
del latest
loss_cfg = CTELossConfig(effect_diversity_weight=1.0)
aggregate = {}
batch_count = 0
with torch.no_grad():
    for i in range(0, len(val.windows), 8):
        frames, actions = batch_from_windows(val.windows[i:i + 8])
        valid = torch.ones(frames.shape[:2], dtype=torch.bool)
        out = model(frames, actions, valid)
        losses = causal_transition_encoder_loss(out, actions, valid, torch.zeros(len(frames), dtype=torch.long), loss_cfg)
        for k, v in losses.items():
            aggregate[k] = aggregate.get(k, 0.) + float(v)
        batch_count += 1
report["full_validation_cpu_fp32"] = {k: v / batch_count for k, v in aggregate.items()}
print("Validation:", report["full_validation_cpu_fp32"], flush=True)
save()

# Spaced windows reduce overlap; every checkpoint is compared on the exact same held-out inputs.
spaced = [(name, s) for name, s in val.windows if s % 17 == 0]
report["checkpoint_diagnostics"] = {}
with torch.no_grad():
    for step in (500, 1000, 1500, 2000, 2500, 3000):
        model, cfg = load_cte(RUN / f"cte_step_{step:06d}.pt", "cpu")
        outputs = {}
        for i in range(0, len(spaced), 8):
            frames, actions = batch_from_windows(spaced[i:i + 8])
            out = model(frames, actions)
            for key in ("phase", "effect_post", "effect_pre", "effect_delta_target"):
                outputs.setdefault(key, []).append(out[key][:, -1])
        outputs = {k: torch.cat(v) for k, v in outputs.items()}
        diag = {k: geometry(v) for k, v in outputs.items()}
        sims = [F.normalize(outputs[k], dim=-1) @ F.normalize(outputs[k], dim=-1).T
                for k in ("effect_post", "effect_delta_target")]
        idx = torch.triu_indices(len(spaced), len(spaced), offset=1)
        diag["effect_vs_visual_delta_similarity_pearson"] = float(torch.corrcoef(torch.stack([s[idx[0], idx[1]] for s in sims]))[0, 1])
        report["checkpoint_diagnostics"][str(step)] = diag
        print("Checkpoint", step, "phase cosine", diag["phase"]["mean_pairwise_cosine"],
              "effect cosine", diag["effect_post"]["mean_pairwise_cosine"],
              "delta correlation", diag["effect_vs_visual_delta_similarity_pearson"], flush=True)
        save()

    # Future perturbations must leave phase and action prediction at boundary 8 unchanged.
    frames, actions = batch_from_windows(spaced[:2])
    base = model(frames, actions)
    altered_frames, altered_actions = frames.clone(), actions.clone()
    altered_frames[:, 9:] = torch.randn_like(altered_frames[:, 9:])
    altered_actions[:, 8:] = torch.randn_like(altered_actions[:, 8:])
    perturbed = model(altered_frames, altered_actions)
    report["causality_max_abs_difference"] = {
        "phase_through_boundary8": float((base["phase"][:, :9] - perturbed["phase"][:, :9]).abs().max()),
        "next_action_through_boundary8": float((base["next_action"][:, :9] - perturbed["next_action"][:, :9]).abs().max()),
        "completed_effects_through_boundary8": float((base["effect_post"][:, :2] - perturbed["effect_post"][:, :2]).abs().max())}
    # Recompute sampled feature rows using CPU fp32; stored features used CUDA bf16 + fp16 storage.
    comparisons = []
    for filename, start in spaced[::max(1, len(spaced) // 5)]:
        fr, ac = batch_from_windows([(filename, start)])
        o = model(fr, ac)
        ep = int(filename.removeprefix("episode_").removesuffix(".npz"))
        with np.load(FEATURES / f"features_{ep:06d}.npz") as z:
            row = start + 16
            phase = torch.from_numpy(z["phase"][row].astype(np.float32))
            effect = torch.from_numpy(z["effect"][row].astype(np.float32))
        comparisons.append({"episode": ep, "boundary": row,
            "phase_cosine": float(F.cosine_similarity(phase, o["phase"][0, -1], dim=0)),
            "min_effect_cosine": float(F.cosine_similarity(effect, o["effect_post"][0], dim=-1).min())})
    report["recomputed_vs_cached_features"] = comparisons
save()
print("Finished:", OUT, flush=True)
