#!/usr/bin/env bash
# 安装进度可视化
# 用法: tools/watch-install.sh [requirements文件] [日志文件]
set -u
ZEVA_WORK=${ZEVA_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY="$ZEVA_WORK/envs/zeva/bin/python"
REQ="${1:-/tmp/zeva-reqs-lean.txt}"
LOG="${2:-$ZEVA_WORK/logs/install-uv-lean.log}"

installed=$(mktemp); wanted=$(mktemp)

# 包名按 PEP 503 归一化：小写，-/_/. 统一成 -（否则 flash_attn 和 flash-attn 对不上）
norm() { sed 's/[=<>!].*//' | tr 'A-Z' 'a-z' | tr '_' '-' | sed 's/\./-/g; s/-\+/-/g'; }

"$PY" -m pip list --format=freeze 2>/dev/null | norm | sort -u > "$installed"
grep -E "^[A-Za-z0-9._-]+==" "$REQ" 2>/dev/null | norm | sort -u > "$wanted"

total=$(wc -l < "$wanted")
if [ "$total" -eq 0 ]; then
  echo "无法解析目标清单: $REQ"; rm -f "$installed" "$wanted"; exit 1
fi

done_n=$(comm -12 "$installed" "$wanted" | wc -l)
pct=$(( done_n * 100 / total ))

width=30
filled=$(( pct * width / 100 ))
empty=$(( width - filled ))
bar=""
for ((i=0; i<filled; i++)); do bar+="█"; done
for ((i=0; i<empty;  i++)); do bar+="░"; done

cache_mb=$(du -sm "$ZEVA_WORK/cache/uv" 2>/dev/null | cut -f1)

echo "  目标清单 : $REQ  (共 $total 个包)"
printf "  [%s] %3d%%   %d/%d 已就位\n" "$bar" "$pct" "$done_n" "$total"
echo "  uv 缓存  : ${cache_mb:-?} MB"
# 注意：不能用 pgrep -f "reqs-..."，本脚本自己的命令行里就含这个串，会自匹配
if ps -eo cmd 2>/dev/null | grep -E "^uv pip install .*reqs-" >/dev/null 2>&1; then
  echo "  uv 进程  : 仍在运行"
else
  echo "  uv 进程  : 已结束"
fi
echo
echo "  最近日志 :"
tr '\r' '\n' < "$LOG" 2>/dev/null | grep -v '^$' | tail -6 | sed 's/^/    /'

rm -f "$installed" "$wanted"
