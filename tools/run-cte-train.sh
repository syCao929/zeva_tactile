#!/usr/bin/env bash
# 第二阶段：训练 Causal Transition Encoder (CTE)（脱离终端，断线/关窗不中断）。
#
# 与 tools/run-xhand-train.sh（policy 训练）完全独立：CTE 只读 VAE latent 缓存，
# 不依赖 policy 的任何产物，两者可以并发跑。两个脚本各自记 PID / run 名，互不干扰。
#
# run 名由你指定（第 2 个参数）。它决定检查点与日志的落盘位置：
#     runs/zeva_cte/<run名>/cte_step_XXXXXX.pt
#     logs/<run名>.log
# **同名 = 续训，换名 = 新开一个 run**。
#
# 前置：VAE latent 缓存（由 vae_cache 生成，可断点续跑）
#   PYTHONPATH=. python -m cosmos_framework.zeva_training.vae_cache \
#     --vae-path "$WAN_VAE_PATH" --dataset-root "$XHAND_DATA_ROOT" \
#     --output "$ZEVA_WORK/datasets/xhand_cte_cache"
#
# 用法:
#   tools/run-cte-train.sh start [run名]    启动 / 续训
#   tools/run-cte-train.sh status [run名]   状态、进度、最新 loss
#   tools/run-cte-train.sh tail [run名]     跟踪日志
#   tools/run-cte-train.sh stop             优雅停止
#   tools/run-cte-train.sh fresh [run名]    归档该 run，下次从零重训
#   tools/run-cte-train.sh runs             列出所有 run
#   tools/run-cte-train.sh rm <run名>       删除指定 run（二次确认）
#
# 环境变量:
#   STEPS=500  SAVE_EVERY=100  BATCH_SIZE=8  RUN_NAME=xxx
#   CTE_CACHE=<latent缓存目录>  CTE_WINDOW_LATENTS=17  CTE_NUM_WORKERS=4
#   CTE_LR=1e-4
set -u

ZEVA_WORK=${ZEVA_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CF_ROOT="$ZEVA_WORK/cosmos-framework"
NAME_PREFIX="cte"
RUNDIR_ROOT="$ZEVA_WORK/runs/zeva_cte"
LOG_DIR="$ZEVA_WORK/logs"
PID_FILE="$LOG_DIR/xhand-cte.pid"
ACTIVE_FILE="$LOG_DIR/xhand-cte-active.txt"
CKPT_GLOB="cte_step_*"
: "${STEPS:=500}"
: "${SAVE_EVERY:=100}"
: "${BATCH_SIZE:=8}"
: "${CTE_CACHE:=$ZEVA_WORK/datasets/xhand_cte_cache}"
: "${CTE_WINDOW_LATENTS:=17}"
: "${CTE_NUM_WORKERS:=4}"
: "${CTE_LR:=1e-4}"
: "${RUN_NAME:=}"

mkdir -p "$LOG_DIR"

usage_hint() { sed -n '2,28p' "$0"; }

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

CMD="${1:-start}"

case "$CMD" in
start)
  if pid=$(running_pid); then echo "已在运行 (PID $pid)。先 stop 再重启。" >&2; exit 1; fi
  RUN_NAME=$(resolve_name "${2:-}")
  RUN_DIR="$RUNDIR_ROOT/$RUN_NAME"
  LOG_FILE="$LOG_DIR/$RUN_NAME.log"

  echo "== 检查环境 =="
  # shellcheck disable=SC1090
  source "$ZEVA_WORK/env.sh"
  if [ ! -f "$CTE_CACHE/manifest.json" ]; then
    echo "  缺少 $CTE_CACHE/manifest.json" >&2
    echo "  先跑 vae_cache 生成 latent 缓存（见脚本头部注释）" >&2
    exit 2
  fi
  n_ep=$(python3 -c "import json;print(json.load(open('$CTE_CACHE/manifest.json'))['num_episodes'])" 2>/dev/null || echo "?")
  printf "  ✅ %-24s (%s 个 episode)\n" "CTE latent 缓存" "$n_ep"

  echo "== run: $RUN_NAME =="
  echo "   检查点目录: $RUN_DIR"
  echo "   日志文件:   $LOG_FILE"
  if [ -f "$RUN_DIR/cte_latest.pt" ]; then
    echo "   续训自: cte_latest.pt"
  else
    echo "   新 run，从随机初始化开始"
  fi
  echo "   （续训请用同一个名字: $0 start $RUN_NAME）"

  {
    echo ""
    echo "################ 会话开始 $(date '+%F %T')  [CTE] ################"
    echo "# run=$RUN_NAME  steps=$STEPS  save_every=$SAVE_EVERY  batch=$BATCH_SIZE  lr=$CTE_LR"
    echo "# cache=$CTE_CACHE  window_latents=$CTE_WINDOW_LATENTS"
  } >> "$LOG_FILE"
  echo "$RUN_NAME" > "$ACTIVE_FILE"

  echo "== 启动 (steps=$STEPS save_every=$SAVE_EVERY batch=$BATCH_SIZE lr=$CTE_LR) =="
  cd "$CF_ROOT" || exit 2
  setsid nohup bash -c "
    set -u
    cd '$CF_ROOT'
    source '$ZEVA_WORK/env.sh'
    # env.sh points TMPDIR at the shared NFS volume. multiprocessing's
    # resource_tracker puts its socket in \$TMPDIR/pymp-* and then fails to
    # unlink it on NFS at shutdown (EBUSY), printing a spurious traceback and
    # leaking one directory per run (536 had piled up). CTE needs no big temp
    # space, so send it to local disk instead.
    export TMPDIR=/tmp
    exec '$ZEVA_WORK/envs/zeva/bin/python' -m cosmos_framework.zeva_training.train_cte \
      --cache-dir '$CTE_CACHE' \
      --output    '$RUN_DIR' \
      --steps $STEPS --save-every $SAVE_EVERY --batch-size $BATCH_SIZE \
      --lr $CTE_LR --window-latents $CTE_WINDOW_LATENTS --num-workers $CTE_NUM_WORKERS \
      --resume
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
  n=$(ls -1 "$RUN_DIR"/$CKPT_GLOB 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then
    echo "  最新检查点: $(ls -1 "$RUN_DIR"/$CKPT_GLOB 2>/dev/null | tail -1 | xargs basename)"
    echo "  共 $n 个, 占用 $(du -sh "$RUN_DIR" 2>/dev/null | cut -f1)  (每个约 39MB)"
  else
    echo "  尚无检查点（首个在 step $SAVE_EVERY）"
  fi
  echo
  echo "== 最近 loss =="
  if [ -s "$LOG_FILE" ]; then
    grep -E "^\[.*step [0-9]+/" "$LOG_FILE" | tail -3 | sed 's/^/  /'
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
  # 负 PID 杀整个进程组（理由同 policy 脚本）
  echo "向进程组 -$pid 发送 SIGTERM"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 60); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if pgrep -g "$pid" >/dev/null 2>&1; then
    echo "  组内仍有进程。用 $0 status 观察；确认卡死: kill -9 -- -$pid"
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
  echo "再用同名 start 即可从零重新训练（或换个新名字）"
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
    last=$(ls -1 "$d"/$CKPT_GLOB 2>/dev/null | tail -1 | xargs -r basename)
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
