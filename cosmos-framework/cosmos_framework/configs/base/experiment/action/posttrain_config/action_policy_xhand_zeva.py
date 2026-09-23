# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_xhand_zeva`` — stage-2 Zeva injection training on UR7e + XHand.

Starts from the Phase-1 policy (``action_policy_xhand_nano``) and trains *only* the
Zeva behavior modules against the CTE features cached by ``zeva_training.cte_features``,
leaving the policy frozen.  This is the recipe the release never shipped: its own Zeva
experiments set ``dataloader_train = None`` (see
``action_policy_robocasa365_atomic5_zeva.py:57-58``), which is why nothing in the
snapshot can be trained.

Three things differ from the Phase-1 recipe:

1. **``behavior_stage2`` is enabled.**  This is what makes ``_attach_stage2_behavior``
   (``omni_mot_model.py:711-716``) demand the four ``behavior_*`` tensors from every
   batch, and what makes the model allocate ``behavior_pbd`` / ``behavior_adapter`` /
   ``behavior_global_projector``.

2. **Only the Zeva modules are trainable.**  ``keys_to_select`` is a *substring*
   match over parameter names relative to ``model.net``
   (``utils/generator/optimizer.py:176-178``); everything not matching is frozen in
   place.  Replacing the list (rather than extending it) is what freezes the policy.

3. **The dataset emits the four tensors.**  ``zeva_feature_cache`` switches on
   ``ZevaBehaviorWrapper`` inside ``get_action_xhand_sft_dataset``.

``action_dim`` is 18 (6 arm + 12 hand joints), not DROID/RoboCasa's 7.  ``horizon``
must equal the dataset's ``chunk_length`` (32) and ``leading_condition_steps`` is 0,
because ``_encode_action`` requires the action-token count to be
``horizon + leading_condition_steps`` (``cosmos3_vfm_network.py:1012-1014``).

Usage (1 node, 8 GPU)::

    ZEVA_FEATURE_CACHE=$ZEVA_WORK/datasets/xhand_cte_features \\
    tools/run-xhand-zeva-train.sh start zeva-v1-20260921
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_xhand_nano import (
    action_policy_xhand_nano,
)

cs = ConfigStore.instance()

action_policy_xhand_zeva = copy.deepcopy(action_policy_xhand_nano)

action_policy_xhand_zeva["job"].update(
    project="zeva",
    group="zeva_xhand",
    name="action_policy_xhand_zeva",
)

action_policy_xhand_zeva["model"]["config"]["behavior_stage2"] = dict(
    enabled=True,
    global_dim=256,
    phase_dim=128,
    effect_dim=128,
    effect_history_length=4,
    action_dim=18,  # 6 arm + 12 hand, NOT RoboCasa's 7
    horizon=32,  # == the dataset's chunk_length
    num_anchors=8,
    hidden_dim=256,
    num_heads=4,
    prior_loss_weight=0.01,
    prior_dropout_rate=0.4,
    prior_inference_guidance_scale=0.5,
    global_prefix_tokens=1,
    leading_condition_steps=0,
)

# Train the Zeva modules only; the policy stays frozen. `keys_to_select` is matched
# as a substring against parameter names relative to `model.net`, so these prefixes
# catch PolicyInjectionPrior / CausalPromptPolicyAdapter / the 256-d prefix projector.
_zeva_keys = action_policy_xhand_zeva["optimizer"]["keys_to_select"]
_zeva_keys[:] = ["behavior_pbd", "behavior_adapter", "behavior_global_projector"]
action_policy_xhand_zeva["optimizer"]["lr_multipliers"].update(
    behavior_pbd=5.0,
    behavior_adapter=5.0,
    behavior_global_projector=5.0,
)

# These modules do not exist in the Phase-1 checkpoint; without the skip list the
# strict load would fail on three missing prefixes.
action_policy_xhand_zeva["checkpoint"]["keys_to_skip_loading"] += [
    "behavior_pbd",
    "behavior_adapter",
    "behavior_global_projector",
]

# The `xhand` dataset entry already carries `emit_behavior_metadata=True`; this adds
# the CTE feature lookup that turns those indices into the four behavior_* tensors.
#
# The `,null` default matters: the inference server composes this experiment to get
# the *model* config, and a bare `${oc.env:...}` makes it fail with
# "Environment variable 'ZEVA_FEATURE_CACHE' not found" even though serving never
# touches the dataset. With the default it resolves to None, and the dataset factory
# treats a falsy value as "no wrapper" — training still gets one when the variable
# is exported.
action_policy_xhand_zeva["dataloader_train"]["dataloader"]["datasets"]["xhand"]["dataset"][
    "zeva_feature_cache"
] = "${oc.env:ZEVA_FEATURE_CACHE,null}"

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_xhand_zeva",
    node=action_policy_xhand_zeva,
)
