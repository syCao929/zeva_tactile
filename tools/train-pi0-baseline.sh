#!/usr/bin/env bash
# Train the π0 visual Zeva Stage-2 branch.
set -euo pipefail
PI0_DEFAULT_STEPS=${PI0_DEFAULT_STEPS:-2000}
PI0_DEFAULT_BATCH=${PI0_DEFAULT_BATCH:-2}
PI0_LEARNING_RATE=${PI0_LEARNING_RATE:-2e-4}
PI0_WARMUP_STEPS=${PI0_WARMUP_STEPS:-100}
PI0_RUN_NAME=${PI0_RUN_NAME:-pi0-baseline}
source "$(dirname "${BASH_SOURCE[0]}")/pi0-comparison-common.sh"
pi0_parse_common "$@"
[[ -n "$PI0_BASE_CHECKPOINT" ]] || { echo "Stage 2 requires --base-checkpoint <concrete baseline step dir>" >&2; exit 2; }
pi0_prepare_pair_contract
ARGS=(--set mode=zeva --set init_checkpoint="$PI0_BASE_CHECKPOINT" --set feature_cache="$PI0_FEATURE_CACHE" --set output_dir="$PI0_RUN_DIR" --set max_steps="$PI0_STEPS" --set batch_size="$PI0_BATCH_SIZE" --set grad_accum="$PI0_GRAD_ACCUM" --set seed="$PI0_SEED" --set learning_rate="$PI0_LEARNING_RATE" --set warmup_steps="$PI0_WARMUP_STEPS")
if [[ -n "$PI0_RESUME" ]]; then ARGS+=(--resume "$PI0_RESUME"); fi
pi0_prepare_gpu
pi0_print_or_exec "$PI0_WORKSPACE/configs/pi0/xhand_zeva.json" "${ARGS[@]}"
