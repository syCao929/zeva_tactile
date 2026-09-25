"""Isolate the effect of sequential versus shuffled held-out batches on CTE loss."""
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.model.zeva import CTELossConfig, causal_transition_encoder_loss
from cosmos_framework.zeva_training.cte_dataset import CTECacheWindowDataset
from cosmos_framework.zeva_training.cte_features import load_cte

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "datasets/xhand_cte_cache_v2"
torch.set_num_threads(2)
ds = CTECacheWindowDataset(CACHE, split="val")
data = {}
for entry in ds.episodes:
    with np.load(CACHE / entry["file"]) as z:
        data[entry["file"]] = torch.from_numpy(z["latents"]).float(), torch.from_numpy(z["actions"]).float()
model, _ = load_cte(ROOT / "runs/zeva_cte/cte-v4-20260924/cte_step_003000.pt", "cpu")
saved_outputs, saved_actions = {}, []
with torch.no_grad():
    for i in range(0, len(ds), 8):
        frames, actions = [], []
        for name, start in ds.windows[i:i+8]:
            lat, act = data[name]
            frames.append(lat[:, start:start+17].permute(1,0,2,3))
            actions.append(act[4*start:4*start+64].reshape(16,4,18))
        actions = torch.stack(actions)
        out = model(torch.stack(frames), actions)
        for key, value in out.items():
            saved_outputs.setdefault(key, []).append(value)
        saved_actions.append(actions)
    outputs = {k: torch.cat(v) for k,v in saved_outputs.items()}
    actions = torch.cat(saved_actions)
    result = {"note": "Same 519 held-out windows, identical CPU-fp32 outputs; only batch composition changes.", "orders": {}}
    for seed in (None, 42, 43, 44):
        order = torch.arange(len(ds)) if seed is None else torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))
        total, count = {}, 0
        for i in range(0, len(ds), 8):
            ix = order[i:i+8]
            loss = causal_transition_encoder_loss({k: v[ix] for k,v in outputs.items()}, actions[ix],
                torch.ones(len(ix),17,dtype=torch.bool), torch.zeros(len(ix),dtype=torch.long),
                CTELossConfig(effect_diversity_weight=1.))
            for k,v in loss.items():
                total[k] = total.get(k,0.) + float(v)
            count += 1
        result["orders"]["sequential" if seed is None else f"shuffle_seed{seed}"] = {k:v/count for k,v in total.items()}
Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result), flush=True)
