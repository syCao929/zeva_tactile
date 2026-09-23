#!/usr/bin/env bash
# 从 hf-mirror.com 拉取 Zeva 运行所需的全部权重。
# 用 curl 直连 + 断点续传，不依赖 envs/zeva（该环境可能正在安装中）。
# 可重复执行：已完整下载的文件会跳过，未完成的会续传。
set -u

ZEVA_WORK=${ZEVA_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EP=${HF_ENDPOINT:-https://hf-mirror.com}
MODELS="$ZEVA_WORK/models"
LOG="$ZEVA_WORK/logs/fetch-weights.log"

mkdir -p "$MODELS" "$(dirname "$LOG")"
: > "$LOG"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 单文件下载：$1=repo $2=仓库内路径 $3=本地绝对路径
fetch() {
  local repo="$1" path="$2" out="$3"
  local url="$EP/$repo/resolve/main/$path"
  mkdir -p "$(dirname "$out")"

  if [ -f "$out" ]; then
    # 用 Content-Length 判断是否已完整（HEAD 请求）
    local remote local_sz
    remote=$(curl -sSIL "$url" 2>/dev/null | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{gsub(/\r/,"",v);print v}')
    local_sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ -n "$remote" ] && [ "$local_sz" = "$remote" ]; then
      log "跳过(已完整) $path  ($((local_sz/1024/1024)) MB)"
      return 0
    fi
    log "续传 $path  本地 ${local_sz}B / 远端 ${remote:-?}B"
  else
    log "开始 $path"
  fi

  local attempt=0
  while [ $attempt -lt 8 ]; do
    attempt=$((attempt+1))
    if curl -fL -C - --retry 5 --retry-delay 5 --retry-all-errors \
         --connect-timeout 30 --speed-limit 10240 --speed-time 120 \
         -o "$out" "$url" >>"$LOG" 2>&1; then
      log "完成 $path  ($(stat -c%s "$out") B)"
      return 0
    fi
    log "第 $attempt 次失败，重试 $path"
    sleep 5
  done
  log "!!! 放弃 $path"
  return 1
}

# 列出仓库文件（走镜像 API）并逐个小文件下载
fetch_repo_small() {
  local repo="$1" outdir="$2" pattern="${3:-.}"
  local files
  files=$(curl -sS "$EP/api/models/$repo?blobs=true" 2>/dev/null | python3 -c "
import json,sys,re
d=json.load(sys.stdin)
pat=re.compile(r'''$pattern''')
for f in d.get('siblings') or []:
    n=f.get('rfilename','')
    if pat.search(n): print(n)
" 2>/dev/null)
  [ -z "$files" ] && { log "!! 无法列出 $repo 文件"; return 1; }
  echo "$files" | while read -r f; do
    [ -n "$f" ] && fetch "$repo" "$f" "$outdir/$f"
  done
}

log "================ 开始拉取权重 ================"

# --- 1. Zeva release（裸权重仓库 chen123fu/zeva-robocasa）---
# 注意：仓库实际布局与 docs/reproduce.md 写的不同，稍后建软链做映射。
Z="$MODELS/zeva"
fetch chen123fu/zeva-robocasa "weights/stage1/zeva_cte.pt"                "$Z/weights/stage1/zeva_cte.pt"
fetch chen123fu/zeva-robocasa "weights/stage1/train_memory_effect_v3.pt"  "$Z/weights/stage1/train_memory_effect_v3.pt"
fetch chen123fu/zeva-robocasa "weights/stage3/best.pt"                    "$Z/weights/stage3/best.pt"
fetch chen123fu/zeva-robocasa "weights/stage3/readouts/train.pt"          "$Z/weights/stage3/readouts/train.pt"
fetch chen123fu/zeva-robocasa "weights/stage3/readouts/val.pt"            "$Z/weights/stage3/readouts/val.pt"
fetch chen123fu/zeva-robocasa "weights/pim/pim_adapter_delta.pt"          "$Z/weights/pim/pim_adapter_delta.pt"
fetch chen123fu/zeva-robocasa "weights/pim/stage3_case_study.pt"          "$Z/weights/pim/stage3_case_study.pt"
fetch chen123fu/zeva-robocasa "RELEASE_INFO.json"                         "$Z/RELEASE_INFO.json"
fetch chen123fu/zeva-robocasa "README.md"                                 "$Z/README.md"
fetch chen123fu/zeva-robocasa "LICENSE"                                   "$Z/LICENSE"
fetch chen123fu/zeva-robocasa "NOTICE"                                    "$Z/NOTICE"
fetch chen123fu/zeva-robocasa "reproducibility/inference_seed_manifest.json" "$Z/reproducibility/inference_seed_manifest.json"

log "--- 小文件完成，开始 stage2 DCP（91 GB，最耗时）---"
fetch chen123fu/zeva-robocasa "weights/stage2/model/.metadata" "$Z/weights/stage2/model/.metadata"
for i in 0 1 2 3 4 5 6 7; do
  fetch chen123fu/zeva-robocasa "weights/stage2/model/__${i}_0.distcp" "$Z/weights/stage2/model/__${i}_0.distcp"
done

# --- 2. Wan2.2 VAE ---
log "--- Wan2.2 VAE ---"
fetch Wan-AI/Wan2.2-TI2V-5B "Wan2.2_VAE.pth" "$MODELS/Wan2.2/Wan2.2_VAE.pth"

# --- 3. Qwen3-VL-8B-Instruct（tokenizer + 权重）---
log "--- Qwen3-VL-8B-Instruct ---"
fetch_repo_small "Qwen/Qwen3-VL-8B-Instruct" "$MODELS/Qwen3-VL-8B-Instruct" '.'

log "================ 全部下载任务结束 ================"
du -sh "$MODELS"/* 2>/dev/null | tee -a "$LOG"
