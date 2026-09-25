#!/usr/bin/env bash
# Single-node, foreground launchers for the four-model comparison. Dry-run only
# performs lightweight resource/file checks; fresh launches never auto-resume.
set -euo pipefail

mode=${1:?Use a train-cosmos launcher}; shift
[[ "$mode" =~ ^(base|baseline|tactile)$ ]] || { echo "Invalid mode: $mode" >&2; exit 2; }
workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cf_root="$workspace/cosmos-framework"
usage() {
  cat <<EOF
Usage: tools/train-cosmos-$mode.sh [options]
  --gpus IDS               Visible GPU indices (default: 0,1,2,3,4,5,6,7)
  --base-checkpoint PATH   Stage 2: required fixed Stage 1 iter_XXXXXXXXX DCP
                          Stage 1: optional original Cosmos DCP override
  --run-name NAME         New output name (default: cosmos-$mode)
  --pair-name NAME        Stage 2 matched-pair contract (default: main-s42)
  --output-root PATH      Output is PATH/zeva/comparison_cosmos/NAME
  --steps N               Optimizer updates (base: 5000; Stage 2: 2000)
  --global-batch N        Effective samples/update (default: 112)
  --batch-size N          Samples/GPU/microbatch (default: largest divisor <= 16)
  --grad-accum N          Default: derive exactly from global batch and GPU count
  --lr FLOAT              Base optimizer LR (default: 2e-4; Stage 2 heads use x5)
  --seed N                Model seed (base: 0; Stage 2: 42; split always 42)
  --save-every N          Checkpoint interval (default: 500)
  --workers N             DataLoader workers/GPU (default: 4, minimum 1)
  --cte-cache PATH        Default: datasets/xhand_cte_features_v4
  --tactile-encoder PATH  Default: models/zeva/tactile_patch_encoder_19999.pt
  --resume                Resume same run; requires existing launch manifest/DCP
  --allow-busy-gpus       Explicitly permit sharing GPUs with existing processes
  --dry-run               Check inputs and print command/manifest; write nothing
  --help

Set COSMOS_PYTHON to use another compatible environment. Paths are resolved from
the workspace. Use tmux for foreground training; train.log and launch_manifest.json
are saved inside the run. Pair baseline/tactile checkpoint, seed and budgets.
EOF
}
dry_run=0; resume=0; allow_busy=0
gpus=${COSMOS_GPUS:-0,1,2,3,4,5,6,7}
steps=2000; seed=42
[[ "$mode" != base ]] || { steps=5000; seed=0; }
run_name="cosmos-$mode"
pair_name=main-s42
output_root="$workspace/runs"
global_batch=112; batch_size=''; grad_accum=''; save_every=500; workers=4; lr=2e-4
base_checkpoint=${COSMOS_BASE_CHECKPOINT:-}
[[ "$mode" != base ]] || base_checkpoint=${BASE_CHECKPOINT_PATH:-}
cte_cache=${ZEVA_FEATURE_CACHE:-$workspace/datasets/xhand_cte_features_v4}
tactile_encoder=${TACTILE_ENCODER_CHECKPOINT:-$workspace/models/zeva/tactile_patch_encoder_19999.pt}
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run) dry_run=1; shift ;;
    --resume) resume=1; shift ;;
    --allow-busy-gpus) allow_busy=1; shift ;;
    --gpus|--steps|--run-name|--pair-name|--output-root|--global-batch|--batch-size|--grad-accum|--lr|--seed|--save-every|--workers|--base-checkpoint|--cte-cache|--tactile-encoder)
      (($# >= 2)) || { echo "Missing value: $1" >&2; exit 2; }
      case "$1" in
        --gpus) gpus=$2 ;; --steps) steps=$2 ;; --run-name) run_name=$2 ;;
        --pair-name) pair_name=$2 ;;
        --output-root) output_root=$2 ;; --global-batch) global_batch=$2 ;;
        --batch-size) batch_size=$2 ;; --grad-accum) grad_accum=$2 ;;
        --lr) lr=$2 ;; --seed) seed=$2 ;; --save-every) save_every=$2 ;;
        --workers) workers=$2 ;; --base-checkpoint) base_checkpoint=$2 ;;
        --cte-cache) cte_cache=$2 ;; --tactile-encoder) tactile_encoder=$2 ;;
      esac
      shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid run name' >&2; exit 2; }
[[ "$pair_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid pair name' >&2; exit 2; }
[[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo '--gpus requires comma-separated GPU indices' >&2; exit 2; }
IFS=',' read -r -a gpu_ids <<< "$gpus"
nproc=${#gpu_ids[@]}
for key in steps global_batch save_every workers; do
  value=${!key}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $key: $value" >&2; exit 2; }
done
[[ "$seed" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'Invalid seed' >&2; exit 2; }
if [[ -z "$batch_size" ]]; then
  # Preserve the target global batch at 7 cards (16/card), 8 cards (14/card),
  # or fewer cards with accumulation; no silently rounded sample budgets.
  for ((candidate=16; candidate>=1; candidate--)); do
    if ((global_batch % (nproc * candidate) == 0)); then batch_size=$candidate; break; fi
  done
fi
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { echo 'No integral batch decomposition; set --batch-size/--global-batch' >&2; exit 2; }
if [[ -z "$grad_accum" ]]; then
  ((global_batch % (nproc * batch_size) == 0)) || { echo 'Global batch is not divisible by GPUs * batch-size' >&2; exit 2; }
  grad_accum=$((global_batch / (nproc * batch_size)))
fi
[[ "$grad_accum" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid grad-accum' >&2; exit 2; }
((nproc * batch_size * grad_accum == global_batch)) || { echo 'GPUs * batch-size * grad-accum must equal --global-batch' >&2; exit 2; }

# env.sh sets workspace paths unconditionally. Preserve explicit user resources.
saved_data=${XHAND_DATA_ROOT:-}; saved_stats=${XHAND_ACTION_STATS_PATH:-}
saved_vae=${WAN_VAE_PATH:-}; saved_qwen=${QWEN_VLM_PATH:-}
export ZEVA_WORK="$workspace" ZEVA_SKIP_EGL_WARNING=1
# shellcheck disable=SC1091
source "$workspace/env.sh"
[[ -z "$saved_data" ]] || export XHAND_DATA_ROOT="$saved_data"
[[ -z "$saved_stats" ]] || export XHAND_ACTION_STATS_PATH="$saved_stats"
[[ -z "$saved_vae" ]] || export WAN_VAE_PATH="$saved_vae"
[[ -z "$saved_qwen" ]] || export QWEN_VLM_PATH="$saved_qwen"
if [[ "$mode" == base ]]; then
  base_checkpoint=${base_checkpoint:-$BASE_CHECKPOINT_PATH}
  recipe=action_policy_xhand_nano
else
  [[ -n "$base_checkpoint" ]] || { echo 'Stage 2 requires --base-checkpoint pointing at one fixed Stage 1 iter_XXXXXXXXX directory' >&2; exit 2; }
  recipe=action_policy_xhand_zeva
  [[ "$mode" != tactile ]] || recipe=action_policy_xhand_zeva_tactile
fi
python_bin=${COSMOS_PYTHON:-$workspace/envs/zeva/bin/python}
[[ -x "$python_bin" ]] || { echo "Python unavailable: $python_bin" >&2; exit 2; }
cd "$workspace"
[[ "${base_checkpoint,,}" != *latest* ]] || { echo 'Use a fixed checkpoint path, not latest' >&2; exit 2; }
[[ ! -L "$base_checkpoint" ]] || { echo 'Checkpoint must be a concrete directory, not a mutable symlink' >&2; exit 2; }
base_checkpoint=$(realpath -m -- "$base_checkpoint")
output_root=$(realpath -m -- "$output_root")
cte_cache=$(realpath -m -- "$cte_cache")
tactile_encoder=$(realpath -m -- "$tactile_encoder")
run_dir="$output_root/zeva/comparison_cosmos/$run_name"
toml="$cf_root/examples/toml/sft_config/$recipe.toml"
export BASE_CHECKPOINT_PATH="$base_checkpoint" ZEVA_POLICY_CHECKPOINT="$base_checkpoint"
export ZEVA_FEATURE_CACHE="$cte_cache" TACTILE_ENCODER_CHECKPOINT="$tactile_encoder"
export CUDA_VISIBLE_DEVICES="$gpus" LD_LIBRARY_PATH='' PYTHONHASHSEED="$seed"
export IMAGINAIRE_OUTPUT_ROOT="$output_root" PYTHONPATH="$cf_root"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} COSMOS_EXIT_WITHOUT_FINALIZE=1
overrides=(
  "job.project=zeva" "job.group=comparison_cosmos" "job.name=$run_name"
  "trainer.max_iter=$steps" "trainer.seed=$seed" "trainer.grad_accum_iter=$grad_accum"
  "scheduler.cycle_lengths=[$steps]" "optimizer.lr=$lr" "checkpoint.save_iter=$save_every"
  "model.config.parallelism.data_parallel_shard_degree=$nproc"
  "model.config.parallelism.data_parallel_replicate_degree=1"
  "dataloader_train.max_samples_per_batch=$batch_size" "dataloader_train.max_sequence_length=null"
  "dataloader_train.dataloader.batch_size=$batch_size" "dataloader_train.dataloader.num_workers=$workers"
  "dataloader_val.dataloader.num_workers=$workers"
)
command=("$python_bin" -m torch.distributed.run --standalone --nnodes=1 "--nproc_per_node=$nproc"
  -m cosmos_framework.scripts.train "--sft-toml=$toml" -- "${overrides[@]}")

# CPU-only resource/contract checks. Manifest hashes small source artifacts and
# metadata, and inventories DCP shards (not a full hash of multi-GB tensor files).
CUDA_VISIBLE_DEVICES='' "$python_bin" -B - "$mode" "$run_dir" "$base_checkpoint" "$toml" "$steps" "$global_batch" \
  "$batch_size" "$grad_accum" "$seed" "$save_every" "$workers" "$lr" "$gpus" \
  "$dry_run" "$resume" "$workspace" "$pair_name" "$allow_busy" "${command[@]}" <<'PY'
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

mode, run_s, base_s, toml_s = sys.argv[1:5]
steps, global_batch, batch, accum, seed, save, workers = map(int, sys.argv[5:12])
lr, gpus = float(sys.argv[12]), sys.argv[13].split(',')
dry, resume = map(int, sys.argv[14:16])
workspace = Path(sys.argv[16])
pair_name, allow_busy = sys.argv[17], bool(int(sys.argv[18]))
command = sys.argv[19:]
sys.path.insert(0, str(workspace / 'tools'))
from comparison_contract import enforce_pair_contract, ensure_gpus_available
run, base, toml = map(Path, (run_s, base_s, toml_s))

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def digest(path):
    path = Path(path)
    require(path.is_file(), f'Missing file: {path}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path.resolve()), 'sha256': h.hexdigest()}

def dcp_inventory(path, components=('model',)):
    from torch.distributed.checkpoint import FileSystemReader
    result = {}
    for component in components:
        directory = path / component
        meta = digest(directory / '.metadata')
        shards = sorted(directory.glob('*.distcp'))
        require(shards and all(p.stat().st_size > 0 for p in shards), f'No complete {component} shards: {path}')
        metadata = FileSystemReader(directory).read_metadata()
        require(metadata.storage_data, f'Empty DCP metadata: {directory}')
        for entry in metadata.storage_data.values():
            shard = directory / entry.relative_path
            require(shard.is_file() and shard.stat().st_size >= entry.offset + entry.length,
                    f'Missing or truncated DCP shard referenced by metadata: {shard}')
        result[component] = {'metadata': meta, 'shards': [
            {'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in shards]}
    return result

require(len(set(gpus)) == len(gpus), 'Duplicate GPU indices')
require(math.isfinite(lr) and lr > 0, '--lr must be positive and finite')
require(not any('latest' in part.lower() for part in base.parts), 'Use a fixed checkpoint directory, not latest')
require(not base.is_symlink(), 'Checkpoint must be a concrete directory, not a mutable symlink')
base_identity = dcp_inventory(base)
if mode != 'base':
    require(re.fullmatch(r'iter_\d{9}', base.name), 'Stage 2 base must be a concrete iter_XXXXXXXXX directory')
    source_config = base.parent.parent / 'config.yaml'
    import yaml
    config = yaml.safe_load(source_config.read_text())
    ds = config['dataloader_train']['dataloader']['datasets']['xhand']['dataset']
    proprio = config['model']['config'].get('proprio_condition', {})
    behavior = config['model']['config'].get('behavior_stage2') or {}
    require(ds.get('state_mode') == 'joint18' and ds.get('action_mode') == 'full18',
            'Stage 2 needs a joint18/full18 Stage 1 base')
    require(proprio.get('enabled') and proprio.get('input_dim') == 18,
            'Stage 1 checkpoint config must have 18-dimensional proprio')
    require(not behavior.get('enabled', False), 'Choose Stage 1, not an already-trained Stage 2 model')
    base_identity['source_config'] = digest(source_config)
data = Path(os.environ['XHAND_DATA_ROOT']).resolve()
infos = sorted(data.glob('*/*/*/lerobot/meta/info.json'))
require(infos, f'Missing converted XHand data: {data}')
resources = {'base': base_identity, 'action_stats': digest(os.environ['XHAND_ACTION_STATS_PATH']),
             'dataset_info': [digest(p) for p in infos]}
for name in ('WAN_VAE_PATH', 'QWEN_VLM_PATH'):
    require(Path(os.environ[name]).exists(), f'Missing {name}: {os.environ[name]}')
if mode != 'base':
    cache = Path(os.environ['ZEVA_FEATURE_CACHE'])
    cache_info = json.loads((cache / 'manifest.json').read_text())
    require(cache_info.get('phase_dim') == 128 and cache_info.get('effect_dim') == 128,
            'CTE cache dimensions do not match the Zeva recipe')
    require(cache_info.get('effect_history') == 4, 'CTE effect history must be 4')
    resources['cte_manifest'] = digest(cache / 'manifest.json')
    resources['cte_checkpoint'] = digest(cache_info['cte_checkpoint'])
    require(all((cache / item['file']).is_file() for item in cache_info['episodes']), 'Incomplete CTE cache')
if mode == 'tactile':
    encoder = Path(os.environ['TACTILE_ENCODER_CHECKPOINT'])
    require(encoder.suffix == '.pt', 'Tactile encoder must be the converted Torch .pt file, not an Orbax directory')
    resources['tactile_encoder'] = digest(encoder)
source_paths = [toml, workspace / 'tools/cosmos-comparison-common.sh',
                workspace / 'cosmos-framework/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_xhand_nano.py']
if mode != 'base':
    source_paths += [workspace / 'cosmos-framework/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_xhand_zeva.py']
if mode == 'tactile':
    source_paths += [workspace / 'cosmos-framework/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_xhand_zeva_tactile.py']
contract = {
    'mode': mode, 'base_checkpoint': str(base), 'run_dir': str(run),
    'steps': steps, 'global_batch': global_batch, 'batch_size': batch, 'grad_accum': accum,
    'world_size': len(gpus), 'seed': seed, 'split_seed': 42, 'split_val_ratio': 0.03,
    'save_every': save, 'workers': workers, 'optimizer_lr': lr,
    'shared_head_lr_multiplier': 1 if mode == 'base' else 5,
    'planned_window_exposures': steps * global_batch,
    'resources': resources, 'sources': [digest(p) for p in source_paths],
    'environment': {name: os.environ.get(name, '') for name in (
        'XHAND_DATA_ROOT', 'XHAND_ACTION_STATS_PATH', 'WAN_VAE_PATH', 'QWEN_VLM_PATH',
        'ZEVA_FEATURE_CACHE', 'TACTILE_ENCODER_CHECKPOINT', 'LD_LIBRARY_PATH', 'PYTHONPATH',
        'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'TMPDIR', 'HF_ENDPOINT', 'HF_HOME')},
    'python': sys.executable,
    'pair_name': pair_name if mode != 'base' else None,
}
manifest_path = run / 'launch_manifest.json'
resume_from = None
if resume:
    require(manifest_path.is_file(), 'Resume requires launch_manifest.json from this launcher in the same run')
    prior = json.loads(manifest_path.read_text())
    require(prior['contract'] == contract,
            'Resume contract differs; keep the same budgets, source files and inputs (no scheduler reset)')
    latest = (run / 'checkpoints/latest_checkpoint.txt').read_text().strip()
    require(re.fullmatch(r'iter_\d{9}', latest), 'Invalid latest_checkpoint.txt')
    resume_from = str(run / 'checkpoints' / latest)
    dcp_inventory(Path(resume_from), ('model', 'optim', 'scheduler', 'trainer'))
else:
    require(not run.exists() or not any(run.iterdir()), 'Run is not empty: choose a new --run-name or explicit --resume')
if not dry:
    ensure_gpus_available(','.join(gpus), allow_busy=allow_busy)
pair_path = None
if mode != 'base':
    pair_path = run.parent / 'pairs' / f'{pair_name}.json'
    pair_payload = {key: value for key, value in contract.items()
                    if key not in ('mode', 'run_dir', 'resources', 'sources', 'environment')}
    pair_payload['resources'] = {key: value for key, value in resources.items() if key != 'tactile_encoder'}
    pair_payload['sources'] = [digest(workspace / 'cosmos-framework/cosmos_framework/configs/base/experiment/action/posttrain_config' / name)
                               for name in ('action_policy_xhand_nano.py', 'action_policy_xhand_zeva.py')]
    pair_payload['environment'] = {key: value for key, value in contract['environment'].items()
                                   if key != 'TACTILE_ENCODER_CHECKPOINT'}
    enforce_pair_contract(pair_path, pair_payload, create=not dry)
manifest = {'schema_version': 1, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'contract': contract, 'cuda_visible_devices': gpus, 'command': command,
            'resume_from': resume_from,
            'pair_contract': str(pair_path) if pair_path else None,
            'integrity_scope': 'SHA256 of metadata/config/small resources; DCP shard size+mtime inventory',
            'paired_initialization': 'Same source checkpoint and seed; full GPU equality probe not run by launcher'}
if dry:
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
else:
    run.mkdir(parents=True, exist_ok=True)
    if not resume:
        with manifest_path.open('x') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    with (run / 'launch_history.jsonl').open('a') as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False) + '\n')
PY
printf '\nRun: %s\nGlobal batch: %s GPUs * %s samples * %s accumulation = %s\n' \
  "$run_dir" "$nproc" "$batch_size" "$grad_accum" "$global_batch"
printf 'Command: CUDA_VISIBLE_DEVICES=%q ' "$gpus"
printf '%q ' "${command[@]}"
printf '\n'
((!dry_run)) || exit 0
cd "$cf_root"
"${command[@]}" 2>&1 | tee -a "$run_dir/train.log"
