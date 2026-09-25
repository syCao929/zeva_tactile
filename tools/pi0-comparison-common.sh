#!/usr/bin/env bash
# Shared argument parsing and immutable pair-contract checks for π0 runs.
set -euo pipefail

PI0_WORKSPACE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PI0_OPENPI_ROOT=${OPENPI_ROOT:-"$PI0_WORKSPACE/../openpi-3d-tactile"}
PI0_OUTPUT_ROOT=${PI0_OUTPUT_ROOT:-"$PI0_WORKSPACE/runs/pi0/comparison"}
PI0_PAIR_NAME=${PI0_PAIR_NAME:-main-s42}
PI0_GPUS=${PI0_GPUS:-0,1,2,3,4,5,6,7}
PI0_ALLOW_BUSY_GPUS=${PI0_ALLOW_BUSY_GPUS:-0}
PI0_DRY_RUN=${PI0_DRY_RUN:-0}
PI0_RESUME=${PI0_RESUME:-}
PI0_RUN_NAME=${PI0_RUN_NAME:-}
PI0_BASE_CHECKPOINT=${PI0_BASE_CHECKPOINT:-}
PI0_FEATURE_CACHE=${PI0_FEATURE_CACHE:-"$PI0_WORKSPACE/datasets/xhand_cte_features_threeview"}
PI0_STEPS=${PI0_STEPS:-}
PI0_BATCH_SIZE=${PI0_BATCH_SIZE:-}
PI0_GRAD_ACCUM=${PI0_GRAD_ACCUM:-}
PI0_SEED=${PI0_SEED:-42}
PI0_LEARNING_RATE=${PI0_LEARNING_RATE:-}
PI0_WARMUP_STEPS=${PI0_WARMUP_STEPS:-}

pi0_usage_common() {
  cat <<'HELP'
Options:
  --gpus LIST              Physical GPU indices, default 0,1,2,3,4,5,6,7
  --allow-busy-gpus        Explicitly allow sharing selected GPUs
  --steps N                Optimizer updates (base default 5000, Stage 2 default 2000)
  --batch-size N           Per-GPU batch; default chosen for global batch 112
  --grad-accum N           Gradient accumulation; default chosen for global batch 112
  --seed N                 Training seed (Stage 2 default 42)
  --run-name NAME          Output run name
  --output-root DIR        Root for comparison runs
  --pair-name NAME         Immutable Stage-2 pair contract name
  --base-checkpoint DIR    Concrete baseline step directory (Stage 2 only)
  --cte-cache DIR          Three-view CTE features (default: datasets/xhand_cte_features_threeview)
  --learning-rate LR       Override learning rate
  --warmup-steps N         Override warmup steps
  --resume CHECKPOINT      Resume this concrete checkpoint directory
  --dry-run                Print resolved command and contract without training
  -h, --help               Show this help
HELP
}

pi0_parse_common() {
  while (($#)); do
    case "$1" in
      --gpus) PI0_GPUS=$2; shift 2;;
      --allow-busy-gpus) PI0_ALLOW_BUSY_GPUS=1; shift;;
      --steps) PI0_STEPS=$2; shift 2;;
      --batch-size) PI0_BATCH_SIZE=$2; shift 2;;
      --grad-accum) PI0_GRAD_ACCUM=$2; shift 2;;
      --seed) PI0_SEED=$2; shift 2;;
      --run-name) PI0_RUN_NAME=$2; shift 2;;
      --output-root) PI0_OUTPUT_ROOT=$2; shift 2;;
      --pair-name) PI0_PAIR_NAME=$2; shift 2;;
      --base-checkpoint) PI0_BASE_CHECKPOINT=$2; shift 2;;
      --cte-cache) PI0_FEATURE_CACHE=$2; shift 2;;
      --learning-rate) PI0_LEARNING_RATE=$2; shift 2;;
      --warmup-steps) PI0_WARMUP_STEPS=$2; shift 2;;
      --resume) PI0_RESUME=$2; shift 2;;
      --dry-run) PI0_DRY_RUN=1; shift;;
      -h|--help) pi0_usage_common; exit 0;;
      --) shift; break;;
      *) echo "Unknown option: $1" >&2; pi0_usage_common >&2; exit 2;;
    esac
  done
  (($# == 0)) || { echo "Unexpected arguments: $*" >&2; exit 2; }
  [[ "$PI0_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "--gpus must be comma-separated physical GPU indices" >&2; exit 2; }
  local commas=${PI0_GPUS//[^,]}
  PI0_NPROC=$(( ${#commas} + 1 ))
  [[ -n "$PI0_STEPS" ]] || PI0_STEPS=$PI0_DEFAULT_STEPS
  [[ -n "$PI0_BATCH_SIZE" ]] || PI0_BATCH_SIZE=$PI0_DEFAULT_BATCH
  [[ -n "$PI0_GRAD_ACCUM" ]] || {
    ((112 % (PI0_NPROC * PI0_BATCH_SIZE) == 0)) || { echo "Cannot choose grad_accum for global batch 112; pass --batch-size/--grad-accum" >&2; exit 2; }
    PI0_GRAD_ACCUM=$((112 / (PI0_NPROC * PI0_BATCH_SIZE)))
  }
  [[ "$PI0_STEPS" =~ ^[1-9][0-9]*$ && "$PI0_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$PI0_GRAD_ACCUM" =~ ^[1-9][0-9]*$ ]] || { echo "steps, batch-size and grad-accum must be positive integers" >&2; exit 2; }
  (( PI0_NPROC * PI0_BATCH_SIZE * PI0_GRAD_ACCUM == 112 )) || { echo "Effective global batch must be 112; got $PI0_NPROC*$PI0_BATCH_SIZE*$PI0_GRAD_ACCUM" >&2; exit 2; }
  [[ "$PI0_PAIR_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid --pair-name" >&2; exit 2; }
}

pi0_hash() { sha256sum "$1" | awk '{print $1}'; }

pi0_validate_base_checkpoint() {
  local checkpoint=$1
  [[ "$checkpoint" != */latest.json && "$checkpoint" != */checkpoints ]] || { echo "Use a concrete π0 step directory, not latest.json or checkpoints/" >&2; exit 2; }
  [[ -d "$checkpoint" && ! -L "$checkpoint" ]] || { echo "π0 base checkpoint must be a concrete directory: $checkpoint" >&2; exit 2; }
  [[ "$(basename "$checkpoint")" =~ ^step_[0-9]{8}$ ]] || { echo "π0 base checkpoint must be named step_XXXXXXXX: $checkpoint" >&2; exit 2; }
  [[ -f "$checkpoint/manifest.json" && -f "$checkpoint/backbone.safetensors" ]] || { echo "Invalid π0 baseline checkpoint (need manifest.json and backbone.safetensors): $checkpoint" >&2; exit 2; }
  PI0_BASE_CHECKPOINT=$(cd -- "$checkpoint" && pwd -P)
  PI0_BASE_MANIFEST_SHA=$(pi0_hash "$checkpoint/manifest.json")
  PI0_BASE_BACKBONE_SHA=$(pi0_hash "$checkpoint/backbone.safetensors")
}

pi0_validate_resume_checkpoint() {
  local checkpoint=$1
  [[ -e "$checkpoint" && ! -L "$checkpoint" ]] || { echo "--resume must point to an existing, non-symlink checkpoint: $checkpoint" >&2; exit 2; }
  if [[ "$checkpoint" == */latest.json ]]; then
    [[ -f "$checkpoint" ]] || { echo "Invalid π0 latest.json: $checkpoint" >&2; exit 2; }
  else
    [[ -d "$checkpoint" && "$(basename "$checkpoint")" =~ ^step_[0-9]{8}$ ]] || { echo "--resume must point to step_XXXXXXXX or latest.json: $checkpoint" >&2; exit 2; }
    [[ -f "$checkpoint/manifest.json" && ( -f "$checkpoint/backbone.safetensors" || -f "$checkpoint/memory.safetensors" ) ]] || { echo "Invalid π0 checkpoint (missing manifest/backbone or memory): $checkpoint" >&2; exit 2; }
  fi
}

pi0_prepare_pair_contract() {
  pi0_validate_base_checkpoint "$PI0_BASE_CHECKPOINT"
  [[ "$PI0_FEATURE_CACHE" == /* ]] || PI0_FEATURE_CACHE="$PI0_WORKSPACE/$PI0_FEATURE_CACHE"
  PI0_FEATURE_CACHE=$(realpath -m -- "$PI0_FEATURE_CACHE")
  [[ -z "$PI0_RESUME" ]] || pi0_validate_resume_checkpoint "$PI0_RESUME"
  PI0_OUTPUT_ROOT=$(realpath -m -- "$PI0_OUTPUT_ROOT")
  PI0_RUN_DIR="$PI0_OUTPUT_ROOT/$PI0_PAIR_NAME/$PI0_RUN_NAME"
  if [[ -n "$PI0_RESUME" ]]; then
    local resume_abs
    resume_abs=$(realpath -e -- "$PI0_RESUME")
    local pair_root
    pair_root=$(realpath -m -- "$PI0_OUTPUT_ROOT/$PI0_PAIR_NAME")
    [[ "$resume_abs" == "$pair_root"/*/checkpoints/* ]] || { echo "--resume must belong to the selected output-root/pair-name: $resume_abs" >&2; exit 2; }
    PI0_RUN_DIR=$(cd -- "$(dirname -- "$resume_abs")/.." && pwd -P)
    PI0_RUN_NAME=$(basename "$PI0_RUN_DIR")
  fi
  PI0_CONTRACT="$PI0_OUTPUT_ROOT/$PI0_PAIR_NAME/pair-contract.json"
  [[ ! -e "$PI0_RUN_DIR" || -n "$PI0_RESUME" ]] || { echo "Refusing to reuse nonempty run directory: $PI0_RUN_DIR (choose --run-name or --resume)" >&2; exit 2; }
  for required in \
    "$PI0_WORKSPACE/datasets/press_button_4_times_merged_filtered" \
    "$PI0_WORKSPACE/datasets/pi0_xhand_norm.json" \
    "$PI0_FEATURE_CACHE/manifest.json" \
    "$PI0_WORKSPACE/../hf_weight/paligemma_tokenizer.model"; do
    [[ -e "$required" ]] || { echo "Missing π0 comparison input: $required" >&2; exit 2; }
  done
  local payload_file
  payload_file=$(mktemp)
  export PI0_WORKSPACE PI0_BASE_CHECKPOINT PI0_FEATURE_CACHE PI0_OPENPI_ROOT PI0_NPROC PI0_SEED PI0_BATCH_SIZE PI0_GRAD_ACCUM PI0_STEPS PI0_LEARNING_RATE PI0_WARMUP_STEPS
  PYTHONPATH="$PI0_WORKSPACE:$PI0_WORKSPACE/tools" "$PI0_WORKSPACE/envs/pi0/bin/python" - "$payload_file" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from pi0_zeva.camera import CAMERA_CONTRACT, CAMERAS, require_policy_camera
from pi0_zeva.data import index_feature_cache
out=Path(sys.argv[1]); workspace=Path(os.environ["PI0_WORKSPACE"]).resolve(); base=Path(os.environ["PI0_BASE_CHECKPOINT"]).resolve()
cache=Path(os.environ["PI0_FEATURE_CACHE"])
require_policy_camera(json.loads((base/"manifest.json").read_text())["config"], base)
index_feature_cache(cache)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()
payload={"schema":2,"camera_contract":CAMERA_CONTRACT,"camera_mapping":CAMERAS,"backbone":"pi0","mode":"zeva_stage2","base_checkpoint":str(base),"base_manifest_sha256":sha(base/"manifest.json"),"base_backbone_sha256":sha(base/"backbone.safetensors"),"data_root":str((workspace/"datasets/press_button_4_times_merged_filtered").resolve()),"norm_stats":str((workspace/"datasets/pi0_xhand_norm.json").resolve()),"norm_stats_sha256":sha(workspace/"datasets/pi0_xhand_norm.json"),"feature_cache":str(cache),"cte_manifest_sha256":sha(cache/"manifest.json"),"tokenizer_path":str((workspace/"../hf_weight/paligemma_tokenizer.model").resolve()),"tokenizer_sha256":sha(workspace/"../hf_weight/paligemma_tokenizer.model"),"openpi_root":str(Path(os.environ["PI0_OPENPI_ROOT"]).resolve()),"horizon":32,"fps":15.0,"split_seed":42,"split_val_ratio":0.03,"seed":int(os.environ["PI0_SEED"]),"world_size":int(os.environ["PI0_NPROC"]),"batch_size":int(os.environ["PI0_BATCH_SIZE"]),"grad_accum":int(os.environ["PI0_GRAD_ACCUM"]),"global_batch":int(os.environ["PI0_NPROC"])*int(os.environ["PI0_BATCH_SIZE"])*int(os.environ["PI0_GRAD_ACCUM"]),"max_steps":int(os.environ["PI0_STEPS"]),"learning_rate":float(os.environ["PI0_LEARNING_RATE"]),"warmup_steps":int(os.environ["PI0_WARMUP_STEPS"])}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
  PYTHONPATH="$PI0_WORKSPACE:$PI0_WORKSPACE/tools" "$PI0_WORKSPACE/envs/pi0/bin/python" - "$PI0_CONTRACT" "$payload_file" "$PI0_DRY_RUN" <<'PY'
import json,sys
from pathlib import Path
from comparison_contract import enforce_pair_contract
target,payload,dry=Path(sys.argv[1]),json.loads(Path(sys.argv[2]).read_text()),int(sys.argv[3])
enforce_pair_contract(target,payload,create=not dry)
print(json.dumps({"pair_contract":str(target),"payload":payload},indent=2,sort_keys=True))
PY
  rm -f "$payload_file"
}

pi0_prepare_gpu() {
  local allow_arg=()
  # A dry run is allowed while another job occupies the selected cards; it still
  # verifies that every requested device exists.  Real training requires the
  # explicit --allow-busy-gpus override before sharing a card.
  ((PI0_ALLOW_BUSY_GPUS || PI0_DRY_RUN)) && allow_arg=(--allow-busy-gpus)
  PYTHONPATH="$PI0_WORKSPACE:$PI0_WORKSPACE/tools" "$PI0_WORKSPACE/envs/pi0/bin/python" - "$PI0_GPUS" "${allow_arg[@]}" <<'PY'
import sys
from comparison_contract import ensure_gpus_available
ensure_gpus_available(sys.argv[1], "--allow-busy-gpus" in sys.argv[2:])
PY
}

pi0_print_or_exec() {
  local config=$1; shift
  local -a command=(bash "$PI0_WORKSPACE/tools/run-pi0-xhand.sh" --config "$config" "$@")
  printf 'Resolved: CUDA_VISIBLE_DEVICES=%s PI0_NPROC=%s ' "$PI0_GPUS" "$PI0_NPROC"
  printf '%q ' "${command[@]}"; printf '\n'
  ((PI0_DRY_RUN)) && return 0
  CUDA_VISIBLE_DEVICES="$PI0_GPUS" PI0_NPROC="$PI0_NPROC" OPENPI_ROOT="$PI0_OPENPI_ROOT" exec "${command[@]}"
}
