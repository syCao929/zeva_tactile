#!/usr/bin/env bash
# Phase 5：Zeva stage2 注入训练（脱离终端，断线/关窗不中断）。
#
# 冻结 Phase 1 的策略，只训 behavior_pbd / behavior_adapter /
# behavior_global_projector，条件是 CTE 特征缓存。
#
# 用法:
#   tools/run-xhand-zeva-train.sh start [run名]   启动
#   tools/run-xhand-zeva-train.sh status          状态 / 进度 / GPU
#   tools/run-xhand-zeva-train.sh tail            跟踪日志
#   tools/run-xhand-zeva-train.sh stop            停止（不存检查点！见 README）
#
# 环境变量:
#   STAGE2_POLICY_CHECKPOINT=<Phase1 的 iter_XXXXXXXX 目录>  （必需）
#   ZEVA_FEATURE_CACHE=<cte_features 输出目录>                （必需）
#   MAX_ITER=2000  SAVE_ITER=500  NPROC_PER_NODE=8
set -u

ZEVA_WORK=${ZEVA_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CF_ROOT="$ZEVA_WORK/cosmos-framework"
RUNDIR="$ZEVA_WORK/runs/zeva/zeva_xhand/action_policy_xhand_zeva"
LOG_DIR="$ZEVA_WORK/logs"
LOG_FILE="$LOG_DIR/zeva-stage2.log"
PID_FILE="$LOG_DIR/zeva-stage2.pid"
: "${MAX_ITER:=2000}"
: "${SAVE_ITER:=500}"
: "${STAGE2_POLICY_CHECKPOINT:=${ZEVA_POLICY_CHECKPOINT:-}}"
: "${ZEVA_FEATURE_CACHE:=}"

alive() {
  [ -f "$PID_FILE" ] || return 1
  local p; p=$(cat "$PID_FILE" 2>/dev/null)
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"
}

case "${1:-}" in
start)
  alive >/dev/null && { echo "已在运行 (PID $(cat $PID_FILE))" >&2; exit 1; }
  [ -n "$STAGE2_POLICY_CHECKPOINT" ] || { echo "需要 STAGE2_POLICY_CHECKPOINT" >&2; exit 2; }
  [ -n "$ZEVA_FEATURE_CACHE" ] || { echo "需要 ZEVA_FEATURE_CACHE" >&2; exit 2; }
  [ -f "$STAGE2_POLICY_CHECKPOINT/model/.metadata" ] || {
    echo "$STAGE2_POLICY_CHECKPOINT/model/.metadata 不存在" >&2
    echo "要指向 checkpoints/iter_XXXXXXXX 目录本身" >&2; exit 2; }
  [ -f "$ZEVA_FEATURE_CACHE/manifest.json" ] || { echo "$ZEVA_FEATURE_CACHE/manifest.json 不存在" >&2; exit 2; }

  echo "== 启动 stage2 (max_iter=$MAX_ITER save_iter=$SAVE_ITER) =="
  echo "   策略: $STAGE2_POLICY_CHECKPOINT"
  echo "   特征: $ZEVA_FEATURE_CACHE"
  echo "   日志: $LOG_FILE"
  cd "$CF_ROOT" || exit 2
  # NOTE: 不要用 timeout 包这个命令 —— timeout 会杀掉整个进程组（含 8 个 rank），
  # 而且外层 shell 退出码是 0，看起来像"正常结束"。教训。
  setsid nohup bash -c "
    set -u
    cd '$CF_ROOT'
    source '$ZEVA_WORK/env.sh'
    export ZEVA_POLICY_CHECKPOINT='$STAGE2_POLICY_CHECKPOINT'
    export ZEVA_FEATURE_CACHE='$ZEVA_FEATURE_CACHE'
    ZEVA_TAIL_OVERRIDES='trainer.max_iter=$MAX_ITER checkpoint.save_iter=$SAVE_ITER' \
      source examples/launch_sft_action_policy_xhand_zeva.sh
  " >> "$LOG_FILE" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  sleep 10
  if alive >/dev/null; then echo "   已启动 PID $(cat $PID_FILE)"; else
    echo "   启动失败:" >&2; tail -20 "$LOG_FILE" >&2; exit 1; fi
  ;;
status)
  if p=$(alive); then echo "运行中 (PID $p)，已运行 $(ps -o etime= -p "$p" | tr -d ' ')"; else echo "未在运行"; fi
  echo "--- 进度 ---"
  ls -1 "$RUNDIR/checkpoints" 2>/dev/null | grep '^iter_' | tail -3 || echo "  尚无检查点"
  grep -oE "\[RANK 0\] [0-9]+ : iter_speed" "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/^/  最新: /'
  grep -oE "behavior_prior_nll=[0-9.]+ \(iteration [0-9]+\)" "$LOG_FILE" 2>/dev/null | tail -3 | sed 's/^/  /'
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -3 | sed 's/^/  /'
  ;;
tail)
  echo "(Ctrl-C 只退 tail)"; tail -f "$LOG_FILE"
  ;;
stop)
  p=$(alive) || { echo "未在运行" >&2; exit 1; }
  echo "向进程组 -$p 发送 SIGTERM（不存检查点）"
  kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null
  rm -f "$PID_FILE"
  ;;
*) sed -n '2,12p' "$0"; exit 1;;
esac
