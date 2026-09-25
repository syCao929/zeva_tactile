#!/usr/bin/env bash
# Train the π0 XHand Stage-1 base. This is an initialization model, not a final
# baseline in the four-model tactile comparison.
set -euo pipefail
PI0_DEFAULT_STEPS=${PI0_DEFAULT_STEPS:-5000}
PI0_DEFAULT_BATCH=${PI0_DEFAULT_BATCH:-1}
PI0_LEARNING_RATE=${PI0_LEARNING_RATE:-2.5e-5}
PI0_WARMUP_STEPS=${PI0_WARMUP_STEPS:-100}
PI0_RUN_NAME=${PI0_RUN_NAME:-pi0-base}
PI0_SEED=${PI0_SEED:-0}
source "$(dirname "${BASH_SOURCE[0]}")/pi0-comparison-common.sh"
pi0_parse_common "$@"
PI0_OUTPUT_ROOT=$(realpath -m -- "$PI0_OUTPUT_ROOT")
PI0_RUN_DIR="$PI0_OUTPUT_ROOT/$PI0_RUN_NAME"
[[ ! -e "$PI0_RUN_DIR" || -n "$PI0_RESUME" ]] || { echo "Refusing to reuse nonempty run directory: $PI0_RUN_DIR" >&2; exit 2; }
if [[ -n "$PI0_RESUME" ]]; then
  pi0_validate_resume_checkpoint "$PI0_RESUME"
  local_resume=$(realpath -e -- "$PI0_RESUME")
  PI0_RUN_DIR=$(cd -- "$(dirname -- "$local_resume")/.." && pwd -P)
fi
pi0_prepare_gpu
ARGS=(--set mode=baseline --set output_dir="$PI0_RUN_DIR" --set max_steps="$PI0_STEPS" --set batch_size="$PI0_BATCH_SIZE" --set grad_accum="$PI0_GRAD_ACCUM" --set seed="$PI0_SEED" --set learning_rate="$PI0_LEARNING_RATE" --set warmup_steps="$PI0_WARMUP_STEPS")
if [[ -n "$PI0_RESUME" ]]; then ARGS+=(--resume "$PI0_RESUME"); fi
pi0_print_or_exec "$PI0_WORKSPACE/configs/pi0/xhand_baseline.json" "${ARGS[@]}"
