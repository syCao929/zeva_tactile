#!/usr/bin/env bash
# 评测单个 joint18 基座检查点在留出集（val split）上的损失。
#
# 当前配方通过 PackingDataLoader.restart_on_iter=True 为每轮验证重建底层
# iterator，长跑训练也能重复验证。此脚本用于单独比较保存的基座检查点：
# 在 iteration 0 评估留出集，再执行 1 步训练后退出，不保存模型。
#
# 此入口使用 action_policy_xhand_nano；Stage-2/触觉检查点须使用对应实验配方。
#
# 用法:
#   tools/eval-checkpoint.sh <检查点 iter 目录> [标签]
#   NPROC_PER_NODE=1 tools/eval-checkpoint.sh ...      # 单卡（显存够时）
#
# 环境变量:
#   EVAL_OUT_ROOT  输出根目录（默认 /tmp/zeva-eval，不污染 runs/）
#   NPROC_PER_NODE 卡数（默认 8）
#   EVAL_VAL_WORKERS  验证 loader 的 worker 数（默认 4）
#
# EVAL_VAL_WORKERS 只控制视频解码并行度；是否重新遍历验证集由
# restart_on_iter 决定。比较检查点时应保持数据划分、预处理和评估设置一致。
set -u

ZEVA_WORK=${ZEVA_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CF_ROOT="$ZEVA_WORK/cosmos-framework"
CKPT="${1:?用法: eval-checkpoint.sh <检查点 iter 目录> [标签]}"
TAG="${2:-$(basename "$CKPT")}"
OUT_ROOT="${EVAL_OUT_ROOT:-/tmp/zeva-eval}"

if [ ! -f "$CKPT/model/.metadata" ]; then
  echo "错误：$CKPT/model/.metadata 不存在（要指向 iter_XXXXXXXX 目录本身）" >&2
  exit 2
fi
# 必须转绝对路径：启动器 `_sft_launcher_common.sh` 会 `cd "$WORKDIR"`（进
# cosmos-framework/），相对路径到那里就失效了 —— 实测会报
# `Checkpoint path <rel> does not exist`。
CKPT="$(cd "$CKPT" && pwd)"
mkdir -p "$OUT_ROOT"

cd "$CF_ROOT" || exit 2
# shellcheck disable=SC1090
source "$ZEVA_WORK/env.sh" >/dev/null 2>&1

# 评估作业的输出全部丢到临时目录，别污染 runs/
export OUTPUT_ROOT="$OUT_ROOT"
export IMAGINAIRE_OUTPUT_ROOT="$OUT_ROOT"
export LOG_FILENAME="eval-$TAG.log"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-50123}"

# 关键覆盖项：
#   load_path=<ckpt>          从该检查点 warm start
#   keys_to_skip_loading=[]   必须清空！配方里默认跳过 proprio_projector（那是为
#                             基座准备的），评估自己的检查点时跳过它会让投影层
#                             回到随机初始化
#   max_iter=1 + on_start     评估一次，再执行 1 步训练后退出
#   save_iter 极大             不写检查点（否则又是 137 GB）
#   max_samples_per_batch=8   那一步训练走得快一点
TAIL_OVERRIDES=(
  "job.name=eval-$TAG"
  trainer.max_iter=1
  trainer.run_validation=True
  trainer.run_validation_on_start=True
  "checkpoint.load_path=$CKPT"
  "checkpoint.keys_to_skip_loading=[]"
  checkpoint.save_iter=999999999
  dataloader_train.max_samples_per_batch=8
  # 按 EVAL_VAL_WORKERS 设置验证视频解码并行度。
  "dataloader_val.dataloader.num_workers=${EVAL_VAL_WORKERS:-4}"
  dataloader_val.dataloader.persistent_workers=False
)

echo "== 评测 $CKPT =="
echo "   标签: $TAG"
echo "   日志: $OUT_ROOT/logs/eval-$TAG.log"
echo "   卡数: $NPROC_PER_NODE"

# 必须 source（不是 exec）——TAIL_OVERRIDES 是 bash 数组，无法 export。
# shellcheck disable=SC1091
source "$CF_ROOT/examples/launch_sft_action_policy_xhand_nano.sh"
EXIT=$?

echo
if [ $EXIT -ne 0 ]; then
  echo "❌ 评估失败（exit $EXIT）" >&2
  exit $EXIT
fi

LOSS=$(grep -oE "Validation loss \(iteration [0-9]+\): [-0-9a-z.]+" "$OUT_ROOT/logs/eval-$TAG.log" | tail -1)
if [ -z "$LOSS" ]; then
  echo "❌ 日志里没有 Validation loss —— 验证没有真正跑起来" >&2
  echo "   请检查验证集是否非空、启动日志和 run_validation_on_start 配置。" >&2
  exit 3
fi
echo "✅ $TAG  $LOSS"
