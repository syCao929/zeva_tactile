#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for UR7e + XHand Zeva stage-2 injection training.
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_xhand_zeva.toml (selects the registered
# `action_policy_xhand_zeva` experiment; trains only the Zeva behavior modules on
# top of a frozen Phase-1 policy, conditioned on cached CTE features).
#
# Env vars (override for your filesystem):
#   XHAND_DATA_ROOT         Root of the converted LeRobot tree.
#   XHAND_ACTION_STATS_PATH meta/feature_stats_compact.json of that dataset.
#   ZEVA_POLICY_CHECKPOINT  Phase-1 DCP checkpoint directory to continue from
#                           (runs/zeva/action_xhand/<run>/checkpoints/iter_XXXXXXXX).
#   ZEVA_FEATURE_CACHE      cte_features.py output directory.
#   WAN_VAE_PATH            Wan2.2 VAE .pth (Wan-AI/Wan2.2-TI2V-5B)
#   NPROC_PER_NODE          torchrun --nproc_per_node (default 8)
#   TAIL_OVERRIDES          bash array of Hydra CLI overrides (see _sft_launcher_common.sh)
#
# Single-node smoke (config/data sanity, a few iters):
#   TAIL_OVERRIDES=(trainer.max_iter=5 checkpoint.save_iter=5 \
#                   dataloader_train.max_samples_per_batch=2) \
#     bash examples/launch_sft_action_policy_xhand_zeva.sh
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_policy_xhand_zeva.toml"

: "${XHAND_DATA_ROOT:?Set XHAND_DATA_ROOT to the converted LeRobot tree}"
: "${XHAND_ACTION_STATS_PATH:?Set XHAND_ACTION_STATS_PATH to meta/feature_stats_compact.json}"
: "${ZEVA_POLICY_CHECKPOINT:?Set ZEVA_POLICY_CHECKPOINT to the Phase-1 DCP checkpoint dir}"
: "${ZEVA_FEATURE_CACHE:?Set ZEVA_FEATURE_CACHE to the cte_features output dir}"
: "${WAN_VAE_PATH:?Set WAN_VAE_PATH to the Wan2.2 VAE checkpoint}"

export XHAND_DATA_ROOT XHAND_ACTION_STATS_PATH ZEVA_POLICY_CHECKPOINT ZEVA_FEATURE_CACHE WAN_VAE_PATH

EXTRA_DATASET_CHECK='
# NOTE: use [ ] not [[ ]] for the glob — [[ ]] suppresses pathname expansion,
# so the pattern would stay literal and the test would always fail.
compgen -G "$XHAND_DATA_ROOT"/*/*/*/lerobot/meta/info.json >/dev/null || {
  echo "ERROR: no <root>/<category>/<task>/<x>/lerobot/meta/info.json under $XHAND_DATA_ROOT" >&2
  echo "       run: python tools/convert_xhand_dataset.py" >&2
  exit 1
}
[ -f "$XHAND_ACTION_STATS_PATH" ] || { echo "ERROR: missing $XHAND_ACTION_STATS_PATH" >&2; exit 1; }
[ -f "$WAN_VAE_PATH" ] || { echo "ERROR: missing $WAN_VAE_PATH" >&2; exit 1; }
# load_path is the directory the DCP loader appends `model/` to, so this must be an
# iteration directory (…/checkpoints/iter_XXXXXXXX), not the checkpoints/ root.
[ -f "$ZEVA_POLICY_CHECKPOINT/model/.metadata" ] || {
  echo "ERROR: $ZEVA_POLICY_CHECKPOINT/model/.metadata not found" >&2
  echo "       point ZEVA_POLICY_CHECKPOINT at a Phase-1 checks/iter_XXXXXXXX dir" >&2
  exit 1
}
[ -f "$ZEVA_FEATURE_CACHE/manifest.json" ] || {
  echo "ERROR: no CTE feature cache manifest at $ZEVA_FEATURE_CACHE" >&2
  echo "       run: python -m cosmos_framework.zeva_training.cte_features ... " >&2
  exit 1
}
'

# TAIL_OVERRIDES is a bash *array*, and arrays are not exported to child processes,
# so `TAIL_OVERRIDES=(...) bash this_script.sh` silently drops them (the common
# launcher then defaults it to empty and the run proceeds with TOML values). Accept a
# whitespace-separated string instead so a caller in another shell can still override:
#   ZEVA_TAIL_OVERRIDES="trainer.max_iter=5 checkpoint.save_iter=5" bash this_script.sh
if [ -n "${ZEVA_TAIL_OVERRIDES:-}" ]; then
  read -r -a TAIL_OVERRIDES <<< "$ZEVA_TAIL_OVERRIDES"
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
