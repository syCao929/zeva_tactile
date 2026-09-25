"""Re-encode one real boundary observation on CPU and compare with retained CTE cache."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cosmos_framework.data.generator.action.datasets.xhand_lerobot_dataset import XHandLeRobotDataset
from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface
from cosmos_framework.zeva_training.vae_cache import _compose_frames
from lerobot.datasets.video_utils import decode_video_frames

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "datasets/xhand_cte_cache_v2"
manifest = json.loads((CACHE / "manifest.json").read_text())
torch.set_num_threads(2)
dataset = XHandLeRobotDataset(manifest["dataset_root"], fps=15., chunk_length=32,
    use_state=False, action_mode="full18", state_mode="joint18", action_normalization=None, split="full")
ep = dataset.episodes[0]
raw_frame = 16
views = {k: decode_video_frames(v, [raw_frame / 15.], tolerance_s=2e-4, backend="torchcodec")
         for k, v in ep.video_paths.items()}
composite = torch.cat([F.interpolate(views[k], size=(256, 256), mode="bilinear", align_corners=False)
                       for k in ("left", "wrist")], dim=-1)
uint8 = (composite * 255.).clamp(0, 255).to(torch.uint8)
x = _compose_frames(uint8.permute(1, 0, 2, 3))
server_x = F.interpolate(uint8.float(), size=(480, 832), mode="bilinear", align_corners=False).unsqueeze(2) / 127.5 - 1
assert torch.equal(x, server_x)
print("Decoded raw frame", raw_frame, "range", (float(x.min()), float(x.max())), flush=True)
vae = Wan2pt2VAEInterface(vae_path=manifest["vae_path"], spatial_compression_factor=16,
                         temporal_compression_factor=4, causal=True)
print("VAE loaded on", next(vae.model.model.parameters()).device, flush=True)
with torch.no_grad():
    latent = vae.encode(x)[0, :, 0].float()
with np.load(CACHE / f"episode_{ep.episode_id:06d}.npz") as z:
    cached = torch.from_numpy(z["latents"][:, raw_frame // 4].astype(np.float32))
result = {"device": "cpu", "vae_dtype": str(vae.dtype),
          "episode": ep.episode_id, "raw_frame": raw_frame, "shape": list(latent.shape),
          "preprocessing_matches_server_formula": True,
          "input_range": [float(x.min()), float(x.max())],
          "cached_vs_reencoded_cosine": float(F.cosine_similarity(latent.flatten(), cached.flatten(), dim=0)),
          "relative_l2_error": float((latent-cached).norm()/cached.norm()),
          "cached_std": float(cached.std()), "reencoded_std": float(latent.std())}
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
