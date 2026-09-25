# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Proprietary

"""Zeva stage-2 XHand recipe with the causal tactile residual branch."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_xhand_zeva import (
    action_policy_xhand_zeva,
)

cs = ConfigStore.instance()
action_policy_xhand_zeva_tactile = copy.deepcopy(action_policy_xhand_zeva)
action_policy_xhand_zeva_tactile["job"].update(name="action_policy_xhand_zeva_tactile")

behavior = action_policy_xhand_zeva_tactile["model"]["config"]["behavior_stage2"]
behavior.update(
    tactile_enabled=True,
    tactile_encoder_checkpoint="${oc.env:TACTILE_ENCODER_CHECKPOINT}",
    tactile_memory_steps=30,
    tactile_token_dim=256,
)
for loader_name in ("dataloader_train", "dataloader_val"):
    dataset = action_policy_xhand_zeva_tactile[loader_name]["dataloader"]["datasets"]["xhand"]["dataset"]
    dataset.update(use_tactile=True, tactile_memory_steps=30)

action_policy_xhand_zeva_tactile["optimizer"]["keys_to_select"].extend(
    [
        "tactile_encoder_projector.projector",
        "tactile_bit",
        # This experiment learns an effect residual. Keep the unused phase and
        # confidence heads out of the optimizer until they have an objective.
        "tactile_behavior_head.finger_projection",
        "tactile_behavior_head.effect_head",
        "tactile_effect_gate",
    ]
)
action_policy_xhand_zeva_tactile["optimizer"]["lr_multipliers"].update(
    tactile_encoder_projector=5.0,
    tactile_bit=5.0,
    tactile_behavior_head=5.0,
    tactile_effect_gate=5.0,
)
action_policy_xhand_zeva_tactile["checkpoint"]["keys_to_skip_loading"] += [
    "tactile_encoder_projector",
    "tactile_bit",
    "tactile_behavior_head",
    "tactile_phase_gate",
    "tactile_effect_gate",
]

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_xhand_zeva_tactile",
    node=action_policy_xhand_zeva_tactile,
)
