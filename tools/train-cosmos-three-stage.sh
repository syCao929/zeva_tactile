#!/usr/bin/env bash
# Linux GPU host: serial three-view Cosmos training; use nohup for background runs.
set -euo pipefail
workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${COSMOS_PYTHON:-$workspace/envs/zeva/bin/python}
# Help and command preview are CPU-only and do not need the training environment.
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h || "$arg" == --dry-run ]]; then
    python_bin=${COSMOS_PYTHON:-python}
    exec "$python_bin" "$workspace/tools/cosmos_three_stage.py" "$@"
  fi
done
saved_data=${XHAND_DATA_ROOT:-}; saved_stats=${XHAND_ACTION_STATS_PATH:-}
saved_vae=${WAN_VAE_PATH:-}; saved_qwen=${QWEN_VLM_PATH:-}; saved_base=${BASE_CHECKPOINT_PATH:-}
export ZEVA_WORK="$workspace" ZEVA_SKIP_EGL_WARNING=1
source "$workspace/env.sh"
[[ -z "$saved_data" ]] || export XHAND_DATA_ROOT="$saved_data"
[[ -z "$saved_stats" ]] || export XHAND_ACTION_STATS_PATH="$saved_stats"
[[ -z "$saved_vae" ]] || export WAN_VAE_PATH="$saved_vae"
[[ -z "$saved_qwen" ]] || export QWEN_VLM_PATH="$saved_qwen"
[[ -z "$saved_base" ]] || export BASE_CHECKPOINT_PATH="$saved_base"
export COSMOS_PYTHON="$python_bin" PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=''
export PYTHONPATH="$workspace/cosmos-framework" TMPDIR=/tmp OMP_NUM_THREADS=1
exec "$python_bin" "$workspace/tools/cosmos_three_stage.py" "$@"
