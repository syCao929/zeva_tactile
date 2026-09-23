#!/usr/bin/env bash
# 第一阶段：在自有数据上微调 action policy（脱离终端，断线/关窗不中断）。
#
# 只管 policy。CTE 训练用另一个脚本 tools/run-cte-train.sh，两者独立、可并发。
#
# run 名由你指定（第 2 个参数）。它决定检查点与日志的落盘位置：
#     runs/zeva/action_xhand/<run名>/checkpoints/iter_XXXXXXXX/
#     logs/<run名>.log
# **同名 = 续训，换名 = 新开一个 run**。所以：
#     首次训练   tools/run-xhand-train.sh start v1-20260921
#     中断后续训 tools/run-xhand-train.sh start v1-20260921   ← 同一个名字即可
#     另起实验   tools/run-xhand-train.sh start v2-20260922
#
# 用法:
#   tools/run-xhand-train.sh start [run名]    启动 / 续训
#   tools/run-xhand-train.sh status [run名]   状态、进度、最新 loss、GPU
#   tools/run-xhand-train.sh tail [run名]     跟踪日志（Ctrl-C 只退 tail）
#   tools/run-xhand-train.sh stop             停止（SIGTERM；⚠️ 不存检查点，最多丢 SAVE_ITER 步）
#   tools/run-xhand-train.sh fresh [run名]    归档该 run，下次从基座重训
#   tools/run-xhand-train.sh runs             列出所有 run 及占用
#   tools/run-xhand-train.sh rm <run名>       删除指定 run（二次确认）
#
# 环境变量:
#   MAX_ITER=5000  SAVE_ITER=500  NPROC_PER_NODE=8  RUN_NAME=xxx
#   EXTRA_OVERRIDES="optimizer.lr=1e-4"
set -u

ZEVA_WORK=${ZEVA_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CF_ROOT="$ZEVA_WORK/cosmos-framework"
NAME_PREFIX="action_policy_xhand_nano"
RUNDIR_ROOT="$ZEVA_WORK/runs/zeva/action_xhand"
LOG_DIR="$ZEVA_WORK/logs"
PID_FILE="$LOG_DIR/xhand-policy.pid"
ACTIVE_FILE="$LOG_DIR/xhand-policy-active.txt"
CKPT_GLOB="iter_*"
: "${MAX_ITER:=5000}"
: "${SAVE_ITER:=500}"
: "${NPROC_PER_NODE:=8}"
: "${EXTRA_OVERRIDES:=}"
: "${RUN_NAME:=}"

mkdir -p "$LOG_DIR"

usage_hint() { sed -n '2,24p' "$0"; }

running_pid() {
  [ -f "$PID_FILE" ] || return 1
  local pid; pid=$(cat "$PID_FILE" 2>/dev/null)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo "$pid"
}

resolve_name() {
  local given="${1:-}"
  if [ -n "$given" ]; then echo "$given"; return; fi
  if [ -n "$RUN_NAME" ]; then echo "$RUN_NAME"; return; fi
  if pid=$(running_pid) && [ -f "$ACTIVE_FILE" ]; then cat "$ACTIVE_FILE"; return; fi
  echo "$NAME_PREFIX-$(date '+%Y%m%d')"
}

CMD="${1:-}"   # 不带子命令时打用法，绝不默认开跑（误触会启动 137GB 检查点的训练）

case "$CMD" in
start)
  if pid=$(running_pid); then echo "已在运行 (PID $pid)。先 stop 再重启。" >&2; exit 1; fi
  RUN_NAME=$(resolve_name "${2:-}")
  RUN_DIR="$RUNDIR_ROOT/$RUN_NAME"
  LOG_FILE="$LOG_DIR/$RUN_NAME.log"

  echo "== 检查环境 =="
  # shellcheck disable=SC1090
  source "$ZEVA_WORK/env.sh"
  for v in BASE_CHECKPOINT_PATH QWEN_VLM_PATH WAN_VAE_PATH XHAND_DATA_ROOT XHAND_ACTION_STATS_PATH; do
    eval "p=\$$v"
    [ -e "$p" ] || { echo "  缺少 $v -> $p" >&2; exit 2; }
    printf "  ✅ %-24s\n" "$v"
  done

  echo "== run: $RUN_NAME =="
  echo "   检查点目录: $RUN_DIR"
  echo "   日志文件:   $LOG_FILE"
  if [ -f "$RUN_DIR/checkpoints/latest_checkpoint.txt" ]; then
    echo "   续训自: $(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")"
  else
    echo "   新 run，从 BASE_CHECKPOINT_PATH 基座开始"
  fi
  echo "   （续训请用同一个名字: $0 start $RUN_NAME）"

  {
    echo ""
    echo "################ 会话开始 $(date '+%F %T')  ################"
    echo "# run=$RUN_NAME  max_iter=$MAX_ITER  save_iter=$SAVE_ITER  nproc=$NPROC_PER_NODE"
    [ -n "$EXTRA_OVERRIDES" ] && echo "# extra: $EXTRA_OVERRIDES"
  } >> "$LOG_FILE"
  echo "$RUN_NAME" > "$ACTIVE_FILE"

  echo "== 启动 (max_iter=$MAX_ITER save_iter=$SAVE_ITER nproc=$NPROC_PER_NODE) =="
  cd "$CF_ROOT" || exit 2
  setsid nohup bash -c "
    set -u
    cd '$CF_ROOT'
    source '$ZEVA_WORK/env.sh'
    TAIL_OVERRIDES=(
      job.name=$RUN_NAME
      trainer.max_iter=$MAX_ITER
      checkpoint.save_iter=$SAVE_ITER
      $EXTRA_OVERRIDES
    )
    source examples/launch_sft_action_policy_xhand_nano.sh
  " >> "$LOG_FILE" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  sleep 5
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "   已启动 PID $(cat "$PID_FILE")"
    echo "   日志: $LOG_FILE   (tail: $0 tail | 停止: $0 stop)"
  else
    echo "   启动失败，看日志:" >&2; tail -20 "$LOG_FILE" >&2; exit 1
  fi
  ;;

status)
  RUN_NAME=$(resolve_name "${2:-}")
  RUN_DIR="$RUNDIR_ROOT/$RUN_NAME"
  LOG_FILE="$LOG_DIR/$RUN_NAME.log"
  if pid=$(running_pid); then
    echo "运行中 (PID $pid)，已运行 $(ps -o etime= -p "$pid" | tr -d ' ')"
  else
    echo "未在运行"
  fi
  echo "run: $RUN_NAME"
  echo
  echo "== 进度 =="
  if [ -f "$RUN_DIR/checkpoints/latest_checkpoint.txt" ]; then
    echo "  最新检查点: $(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")"
    ls -1 "$RUN_DIR/checkpoints" 2>/dev/null | grep '^iter_' | tail -5 | sed 's/^/    /'
    echo "  占用: $(du -sh "$RUN_DIR/checkpoints" 2>/dev/null | cut -f1)  (每个约 137GB)"
  else
    echo "  尚无检查点"
  fi
  echo
  echo "== 最近 loss =="
  if [ -s "$LOG_FILE" ]; then
    tr '\r' '\n' < "$LOG_FILE" 2>/dev/null | grep "stage2_loss_components" | tail -3 | sed 's/^/  /'
  else
    echo "  (无日志)"
  fi
  echo
  echo "== GPU =="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -8 | sed 's/^/  /'
  ;;

tail)
  RUN_NAME=$(resolve_name "${2:-}")
  LOG_FILE="$LOG_DIR/$RUN_NAME.log"
  [ -f "$LOG_FILE" ] || { echo "日志不存在: $LOG_FILE" >&2; exit 1; }
  echo "(Ctrl-C 只退出 tail，训练不受影响)"
  echo "  文件: $LOG_FILE"
  tail -f "$LOG_FILE"
  ;;

stop)
  if ! pid=$(running_pid); then echo "未在运行" >&2; exit 1; fi
  # 负 PID 杀整个进程组：setsid 让启动进程成为会话/组长，torchrun 及其 8 个 rank
  # 都在组内。只杀组长会留下孤儿进程继续占 GPU、继续写检查点。
  #
  # NOTE: 这**不会**存检查点。`termination_signal_checkpoint` 只认 SIGUSR1，
  # 且靠 Slurm 哨兵文件 $SLURM_LOG_DIR/SIGUSR1_RECEIVED 触发；本机无 Slurm，
  # 该路径为空，回调直接 return。它注册的 SIGTERM 处理器只打日志。
  # 想不丢进度就别 stop，让它跑到下一个 SAVE_ITER。
  echo "向进程组 -$pid 发送 SIGTERM（注意：不会存检查点，最多丢 SAVE_ITER 步）"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 90); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if pgrep -g "$pid" >/dev/null 2>&1; then
    echo "  组长已退出但组内仍有进程（可能仍在存检查点）。用 $0 status 观察。"
    echo "  若确认卡死: kill -9 -- -$pid"
  else
    echo "  全部已退出"; rm -f "$PID_FILE"
  fi
  ;;

fresh)
  pid=$(running_pid) && { echo "训练在运行，先 stop" >&2; exit 1; }
  RUN_NAME=$(resolve_name "${2:-}")
  RUN_DIR="$RUNDIR_ROOT/$RUN_NAME"
  [ -d "$RUN_DIR" ] || { echo "$RUN_DIR 不存在，无需归档。下次 start 即为新 run。" >&2; exit 0; }
  ARCHIVED="$RUN_NAME-archived-$(date '+%Y%m%d-%H%M%S')"
  echo "归档: $RUN_NAME -> $ARCHIVED  ($(du -sh "$RUN_DIR" 2>/dev/null | cut -f1))"
  mv "$RUN_DIR" "$RUNDIR_ROOT/$ARCHIVED"
  echo "再用同名 start 即可从基座重新训练（或换个新名字）"
  ;;

runs)
  echo "目录: $RUNDIR_ROOT"
  [ -f "$ACTIVE_FILE" ] && echo "最近启动: $(cat "$ACTIVE_FILE")"
  echo
  printf "  %-52s %10s  %s\n" "RUN" "大小" "最后检查点"
  total=0
  for d in "$RUNDIR_ROOT"/*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    sz=$(du -sm "$d" 2>/dev/null | cut -f1); total=$((total + sz))
    last=$(ls -1 "$d/checkpoints" 2>/dev/null | grep '^iter_' | tail -1)
    printf "  %-52s %9sM  %s\n" "$n" "$sz" "${last:-（无）}"
  done
  echo
  echo "  合计: $((total / 1024)) GB"
  ;;

rm)
  target="${2:-}"
  [ -n "$target" ] || { echo "用法: $0 rm <run名>   （用 $0 runs 查看）" >&2; exit 1; }
  if pid=$(running_pid) && [ "$target" = "$(cat "$ACTIVE_FILE" 2>/dev/null)" ]; then
    echo "不能删正在运行的 run" >&2; exit 1
  fi
  d="$RUNDIR_ROOT/$target"
  [ -d "$d" ] || { echo "不存在: $d" >&2; exit 1; }
  echo "将删除 $d（$(du -sh "$d" 2>/dev/null | cut -f1)）"
  read -r -p "确认？输入 yes 继续: " ans
  [ "$ans" = "yes" ] || { echo "已取消"; exit 1; }
  rm -rf "$d"; echo "已删除"
  ;;

*) usage_hint; exit 1;;
esac
