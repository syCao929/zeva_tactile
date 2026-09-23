#!/usr/bin/env bash
# 把 HF 仓库 chen123fu/zeva-robocasa 的实际布局，映射成 docs/reproduce.md
# 与 docs/zeva_pim.md 里写的那套路径，这样文档里的命令可以原样执行。
#
# 仓库实际布局                      docs 里写的路径
#   weights/stage1/zeva_cte.pt   -> weights/stage1/cte_step_000500.pt
#   weights/stage2/model/        -> weights/stage2_iter_000005000/
#   weights/stage3/best.pt       -> weights/stage3_iter_000005000/best.pt
#   weights/stage1/train_memory_effect_v3.pt  （两边一致，无需映射）
#
# 用软链而非复制：不占额外空间，且权重更新后映射自动跟随。
# 可重复执行；已存在的软链会重建。
set -u

ZEVA_WORK=${ZEVA_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
Z=${ZEVA_RELEASE:-$ZEVA_WORK/models/zeva}

if [ ! -d "$Z" ]; then
  echo "错误：找不到 $Z —— 请先把 chen123fu/zeva-robocasa 下载到该目录。" >&2
  exit 2
fi

link() {  # $1=目标软链路径  $2=真实路径（相对 $Z）
  local dst="$Z/$1" src="$Z/$2"
  if [ ! -e "$src" ]; then
    echo "  跳过（源不存在）: $2"
    return 1
  fi
  mkdir -p "$(dirname "$dst")"
  [ -L "$dst" ] && rm -f "$dst"
  ln -s "$src" "$dst"
  echo "  $1 -> $2"
  return 0
}

echo "映射 $Z"
link "weights/stage1/cte_step_000500.pt"        "weights/stage1/zeva_cte.pt"
link "weights/stage2_iter_000005000"            "weights/stage2/model"
link "weights/stage3_iter_000005000/best.pt"    "weights/stage3/best.pt"

echo
echo "缺失检查（docs 流程需要但仓库里没有的）:"
for p in weights/stage1/train_memory_effect_v3.pt weights/stage1/zeva_cte.pt \
         weights/stage2/model/.metadata weights/stage3/best.pt; do
  if [ -e "$Z/$p" ]; then
    printf "  [有] %s\n" "$p"
  else
    printf "  [缺] %s\n" "$p"
  fi
done

echo
echo "外部权重（不在本仓库，需另行获取）:"
[ -e "${QWEN_VLM_PATH:-/nonexistent}" ] && echo "  [有] QWEN_VLM_PATH=$QWEN_VLM_PATH" || echo "  [缺] QWEN_VLM_PATH=${QWEN_VLM_PATH:-未设置}  (HF: Qwen/Qwen3-VL-8B-Instruct)"
[ -e "${WAN_VAE_PATH:-/nonexistent}" ]  && echo "  [有] WAN_VAE_PATH=$WAN_VAE_PATH"   || echo "  [缺] WAN_VAE_PATH=${WAN_VAE_PATH:-未设置}  (HF: Wan-AI/Wan2.2-TI2V-5B 里的 Wan2.2_VAE.pth)"
[ -d "${ROBOCASA365_ROOT:-/nonexistent}" ] && echo "  [有] ROBOCASA365_ROOT=$ROBOCASA365_ROOT" || echo "  [缺] ROBOCASA365_ROOT=${ROBOCASA365_ROOT:-未设置}  (GitHub: robocasa/robocasa + 资产)"

echo
echo "stage2 DCP 分片完整性:"
n=$(ls "$Z/weights/stage2/model/"__*_0.distcp 2>/dev/null | wc -l)
if [ "$n" -eq 8 ]; then
  echo "  8/8 分片齐全"
elif [ "$n" -eq 0 ]; then
  echo "  0/8 —— 尚未下载"
else
  echo "  $n/8 分片（下载可能未完成）"
fi
