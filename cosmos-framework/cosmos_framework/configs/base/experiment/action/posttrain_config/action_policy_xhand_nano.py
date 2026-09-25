# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_xhand_nano`` — UR7e + XHand action policy SFT recipe.

Same stack as ``action_policy_droid_nano`` (PackingDataLoader +
RankPartitionedDataLoader, FSDP, bf16) but feeds the UR7e + XHand dataset
(18-dim joint-position commands, 15 fps) instead of DROID.

Two departures from the DROID recipe, both deliberate:

1. **Base checkpoint.** DROID trains its action heads from scratch off the bare
   ``nvidia/Cosmos3-Nano`` base, so it skips the action-head tensors on load.
   This recipe is meant to start from a *post-trained policy*
   (``nvidia/Cosmos3-Nano-Policy-DROID``), whose action heads are already
   trained. Inheriting DROID's skip list would silently discard them, so the
   skip list is narrowed to the unused EMA namespace only.

2. **A fresh domain slot.** ``ur7e-xhand`` owns domain id 23
   (``data/generator/action/domain_utils.py``), which no released checkpoint has
   seen. Its rows in ``action2llm`` / ``llm2action`` / ``action_modality_embed``
   start from whatever the base initialized them to and **must stay trainable**,
   or the policy can never emit this embodiment's actions.

Usage (1 node, 8 GPU)::

    XHAND_DATA_ROOT=/path/to/press_button_4_times_merged_filtered \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano-Policy-DROID DCP dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_xhand_nano.toml
"""

import copy

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_xhand_sft_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


action_policy_xhand_nano = copy.deepcopy(action_policy_droid_nano)

action_policy_xhand_nano["job"].update(
    project="zeva",
    group="action_xhand",
    name="action_policy_xhand_nano",
)

# Robot state as a projected prefix token, mirroring the shipped Zeva recipe
# (action_policy_robocasa365_atomic5_zeva.py:41-45). `input_dim` must equal
# len(_STATE_INDICES["joint18"]) == 18 or `_attach_proprio_condition` raises
# `Expected proprio [18], got (...)`.
#
# Without this the policy is pure video -> action: the arm commands are absolute joint
# positions, so a policy that cannot see its own configuration has to infer it from
# pixels on every query, and consecutive chunks carry no continuity signal at all.
#
# 18 is the action space itself -- the arm + hand joint positions the policy commands,
# in the absolute parameterization it commands them in. That is not a coincidence: it
# is the same constraint pi0/pi0.5 build into the architecture
# (`state_proj = nnx.Linear(config.action_dim, width)`, openpi pi0.py:159), and the same
# principle behind Zeva's own proprio (RoboCasa `arm9` = relative EEF pose + gripper
# qpos, i.e. the DOF set its `arm7` action commands, in absolute form).
action_policy_xhand_nano["model"]["config"]["proprio_condition"] = dict(
    enabled=True,
    input_dim=18,
    prefix_tokens=1,
)

# Keep the post-trained policy's action heads (see module docstring).
action_policy_xhand_nano["checkpoint"]["keys_to_skip_loading"] = ["net_ema.", "proprio_projector"]

# `keys_to_select` is a substring ALLOWLIST over parameter names relative to `model.net`
# (utils/generator/optimizer.py:173-175): a parameter matching none of the entries gets
# `requires_grad=False`. `proprio_projector` matches none of the DROID entries, so
# without this line the freshly-initialized projector would be frozen at its xavier init
# and injected into the prefix token forever -- silently, with a perfectly normal loss
# curve. The log line to check is `selected tensors`:
#   v1-20260921 logged "Total tensors: 809, trainable tensors: 809, selected tensors: 410"
#   this recipe must log 412 (the projector's .weight and .bias).
action_policy_xhand_nano["optimizer"]["keys_to_select"].append("proprio_projector")

# The dataset entry is keyed by name; rename `droid` -> `xhand` so logs read
# correctly and the PIM/behavior wrapper (added later) can target it.
_train = action_policy_xhand_nano["dataloader_train"]
_train["dataset_name"] = "action_xhand"
# The DROID default (16 workers/rank) assumes data on local disk. This dataset
# lives on a shared NFS mount, where 16 x 8 ranks = 128 concurrent readers stalls
# individual reads long enough that one rank never finishes dataloader pre-warm
# and every other rank deadlocks on the pre-warm barrier. 4/rank is verified
# working; raise toward 8 if GPU utilization sags. Overridable per-run via
# `-- dataloader_train.dataloader.num_workers=N`.
_train["dataloader"]["num_workers"] = 4
_train["dataloader"]["prefetch_factor"] = 1
_train["dataloader"]["datasets"] = dict(
    xhand=dict(
        ratio=1,
        dataset=L(get_action_xhand_sft_dataset)(
            root="${oc.env:XHAND_DATA_ROOT}",
            # Must equal meta/info.json's fps or XHandLeRobotDataset rejects the root.
            fps=15.0,
            chunk_length=32,
            action_mode="full18",  # 6 arm + 12 hand joint positions
            state_mode="joint18",  # arm + hand joint positions == the action space; feeds proprio_condition (18 dims)
            camera_layout="three_view_grid",
            viewpoint="concat_view",
            view_size=256,
            action_normalization="minmax",
            action_stats_path="${oc.env:XHAND_ACTION_STATS_PATH}",
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            use_image_augmentation=True,
            use_state=True,
            # Emit the behavior_* index fields. Harmless until the Zeva wrapper
            # consumes them, and switching it on later would invalidate caches.
            emit_behavior_metadata=True,
            # Three-view composite: wrist above front/left, preserving each panel's 4:3 ratio.
            # The policy resolution remains an explicit compute-budget choice.
            resolution="256",
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            format_prompt_as_json=True,
        ),
    ),
)

# ------------------------------------------------------------------- train/val split
#
# The loader's own defaults are `split="train"` + `split_val_ratio=0.03`, and this recipe
# never overrode them -- so Phase 1 has always trained on 98 of the 101 episodes
# (`split_episode_ids(101, 42, 0.03, "train")` holds out ids 38 / 68 / 99). Those three
# were then never evaluated either, because `dataloader_val` inherited DROID's `None`.
#
# Wiring the val loader up therefore costs NO training data -- the split is already in
# effect -- and keeps this run comparable with v1-20260921 episode-for-episode, while
# giving a held-out signal to pick the best iteration from.
#
# `split_val_ratio` is pinned explicitly on BOTH entries so they cannot drift apart.
action_policy_xhand_nano["dataloader_train"]["dataloader"]["datasets"]["xhand"]["dataset"].update(
    split="train",
    split_seed=42,
    split_val_ratio=0.03,
)

action_policy_xhand_nano["dataloader_val"] = copy.deepcopy(action_policy_xhand_nano["dataloader_train"])
action_policy_xhand_nano["dataloader_val"]["dataloader"]["datasets"]["xhand"]["dataset"].update(
    split="val",
    use_image_augmentation=False,  # never augment the held-out episodes
    iterable_shuffle=False,  # 3 episodes; sequential streaming, no need to shuffle
)
# Each validation starts a new finite pass. PackingDataLoader otherwise retains
# its exhausted source iterator and yields no batches after the first validation;
# changing worker counts alone does not fix that. Training keeps stream continuity.
action_policy_xhand_nano["dataloader_val"]["restart_on_iter"] = True
# Keep validation decoding in the main process, as in videophy2_sft_nano.
action_policy_xhand_nano["dataloader_val"]["dataloader"].update(
    num_workers=0,
    persistent_workers=False,
    prefetch_factor=None,  # required by torch when num_workers == 0
)

# The val loss is what picks the best iteration. It surfaces in the run log as
# `avg_final_loss: <x>` and `[val] ... avg_final_loss: <x>`
# (callbacks/wandb_log_eval.py:104-110); validation runs under the EMA scope, like the
# reference eval. `max_val_iter` stays at its None default = score the whole val split.
action_policy_xhand_nano["trainer"] = dict(
    run_validation=True,
    run_validation_on_start=True,  # an iter-0 baseline, to see how far training moved
    validation_iter=500,  # == the launcher's SAVE_ITER, so every checkpoint gets a score
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_xhand_nano",
    node=action_policy_xhand_nano,
)
