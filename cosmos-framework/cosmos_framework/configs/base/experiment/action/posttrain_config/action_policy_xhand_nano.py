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

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_xhand_sft_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()


action_policy_xhand_nano = copy.deepcopy(action_policy_droid_nano)

action_policy_xhand_nano["job"].update(
    project="zeva",
    group="action_xhand",
    name="action_policy_xhand_nano",
)

# Keep the post-trained policy's action heads (see module docstring).
action_policy_xhand_nano["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]

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
            state_mode="arm22",  # arm joints + ee pose; unused while proprio is disabled
            camera_layout="left_wrist_horizontal",
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
            # VideoResize resizes aspect-preservingly then reflection-pads to the
            # closest target in this tier. Our composite is left|wrist = 1:2,
            # which is not in the ratio table, so it letterboxes into the 16:9
            # box (192x320 at tier 256). Tune via TOML if that wastes too much.
            resolution="256",
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            format_prompt_as_json=True,
        ),
    ),
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_xhand_nano",
    node=action_policy_xhand_nano,
)
