#!/usr/bin/env bash
# 环境自检。分三档，由浅入深：
#   [A] 现在就能跑 —— 不需要任何权重（依赖 / CUDA / 单测 / 配置组装 / 数据加载）
#   [B] 需要权重   —— 检查权重是否就位
#   [C] 端到端     —— 真正起 policy server（需要全部权重）
#
# 用法:
#   bash tools/verify-env.sh          # 跑 A（+ B 的清单）
#   bash tools/verify-env.sh --all    # 跑到 C
set -u

ZEVA_WORK=${ZEVA_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY="$ZEVA_WORK/envs/zeva/bin/python"
PASS=0; FAIL=0

if [ ! -x "$PY" ]; then
  echo "找不到 $PY —— 请先 source env.sh" >&2; exit 2
fi

ok()   { printf "  \033[32m[OK]\033[0m   %s\n" "$1"; PASS=$((PASS+1)); }
bad()  { printf "  \033[31m[FAIL]\033[0m %s\n" "$1"; FAIL=$((FAIL+1)); }
skip() { printf "  \033[33m[--]\033[0m   %s\n" "$1"; }

echo "=============================================="
echo " A. 不需要权重的检查"
echo "=============================================="

# --- A1 依赖导入 -------------------------------------------------------------
"$PY" - <<'PY' && ok "依赖导入" || bad "依赖导入"
import importlib, sys
mods = ["torch","torchvision","torchcodec","flash_attn","transformers","diffusers",
        "hydra","omegaconf","mujoco","lerobot","openpi_client","pyarrow","cv2","numpy"]
bad = []
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: bad.append(f"{m}({type(e).__name__})")
if bad:
    print("     缺失:", ", ".join(bad)); sys.exit(1)
PY

# --- A2 CUDA -----------------------------------------------------------------
"$PY" - <<'PY' && ok "CUDA 可用" || bad "CUDA 可用"
import torch, sys
assert torch.cuda.is_available(), "CUDA 不可用"
t = torch.randn(1024, 1024, device="cuda")
_ = (t @ t).sum().item()          # 真跑一次，确认不是假可用
print(f"     {torch.cuda.device_count()} × {torch.cuda.get_device_name(0)}")
PY

# --- A3 cosmos_framework 可导入 ----------------------------------------------
"$PY" -c "import cosmos_framework" 2>/dev/null && ok "cosmos_framework 导入" || bad "cosmos_framework 导入"

# --- A4 zeva 单元测试 --------------------------------------------------------
if (cd "$ZEVA_WORK/cosmos-framework" && "$PY" -m pytest cosmos_framework/model/zeva/ -q >/tmp/zeva_ut.log 2>&1); then
  ok "zeva 单元测试 ($(tail -1 /tmp/zeva_ut.log | tr -d '\n'))"
else
  bad "zeva 单元测试（详见 /tmp/zeva_ut.log）"
fi

# --- A5 配置组装（覆盖所有实验配置注册，含 zeva）------------------------------
"$PY" - <<'PY' && ok "Hydra 配置组装 (make_config)" || bad "Hydra 配置组装"
import sys
from cosmos_framework.configs.base.config import make_config
make_config()
PY

# --- A6 数据加载器 -----------------------------------------------------------
"$PY" - <<'PY' && ok "XHand 数据加载器" || bad "XHand 数据加载器"
import os, sys
R = os.environ.get("XHAND_DATA_ROOT")
if not R:
    print("     XHAND_DATA_ROOT 未设置"); sys.exit(1)
from cosmos_framework.data.generator.action.datasets.xhand_lerobot_dataset import XHandLeRobotDataset
ds = XHandLeRobotDataset(
    R, fps=15.0, use_state=True,
    action_stats_path=R + "/xhand/PressButton4Times/lerobot_v21/lerobot/meta/feature_stats_compact.json",
)
s = ds[0]
assert s["video"].shape == (3, 33, 256, 512), s["video"].shape
assert s["action"].shape == (32, 18), s["action"].shape
print(f"     {len(ds.episodes)} episodes / {len(ds)} windows / video{tuple(s['video'].shape)} action{tuple(s['action'].shape)}")
PY

echo
echo "=============================================="
echo " B. 权重就位检查"
echo "=============================================="
check_path() { [ -e "$2" ] && ok "$1" || { printf "  \033[31m[缺]\033[0m %s -> %s\n" "$1" "$2"; FAIL=$((FAIL+1)); }; }

echo " 基础训练/推理（缺这些就跑不起来）："
check_path "基座 DCP (BASE_CHECKPOINT_PATH)" "${BASE_CHECKPOINT_PATH:-/x}/model"
# HF 快照只在你打算重新转换时才需要；已有 DCP 就不必再下。
if [ -f "$ZEVA_WORK/models/cosmos3-nano-policy-droid/checkpoint.json" ] \
   || ls -d "$HF_HOME"/models--nvidia--Cosmos3-Nano* >/dev/null 2>&1; then
  ok "基座 HF 快照（备选基座，非必需）"
else
  printf "  \033[33m[--]\033[0m   基座 HF 快照（可选：只有要重新转换时才需要）\n"
fi
check_path "Qwen3-VL-8B-Instruct"   "${QWEN_VLM_PATH:-/x}/config.json"
check_path "Wan2.2 VAE"             "${WAN_VAE_PATH:-/x}"
check_path "XHand 训练数据"         "${XHAND_DATA_ROOT:-/x}"
check_path "XHand action 统计"      "${XHAND_ACTION_STATS_PATH:-/x}"

echo " 仅复现 RoboCasa 才需要："
check_path "RoboCasa365"            "${ROBOCASA365_ROOT:-/x}"
check_path "Zeva stage1 CTE"        "${ZEVA_RELEASE:-/x}/weights/stage1/zeva_cte.pt"
check_path "Zeva stage1 context"    "${ZEVA_RELEASE:-/x}/weights/stage1/train_memory_effect_v3.pt"
check_path "Zeva stage3 best.pt"    "${ZEVA_RELEASE:-/x}/weights/stage3/best.pt"

# stage2 DCP 分片
n=$(ls "${ZEVA_RELEASE:-/x}/weights/stage2/model/"__*_0.distcp 2>/dev/null | wc -l)
if [ "$n" -eq 8 ]; then ok "Zeva stage2 DCP (8/8 分片)"
elif [ "$n" -eq 0 ]; then printf "  \033[31m[缺]\033[0m Zeva stage2 DCP (0/8 —— 91GB，最耗时)\n"; FAIL=$((FAIL+1))
else printf "  \033[33m[..]\033[0m Zeva stage2 DCP ($n/8 下载中)\n"; FAIL=$((FAIL+1)); fi

# libEGL
ldconfig -p 2>/dev/null | grep -q "libEGL\.so" && ok "libEGL（无头渲染）" || printf "  \033[31m[缺]\033[0m libEGL -> apt-get install -y libegl1 libgl1\n"

echo
echo "=============================================="
echo " 结果: $PASS 通过, $FAIL 待办"
echo "=============================================="

if [ "${1:-}" != "--all" ]; then
  echo
  echo "权重齐全后跑端到端：bash tools/verify-env.sh --all"
  exit 0
fi

echo
echo "=============================================="
echo " C. 端到端（起 policy server）"
echo "=============================================="
cd "$ZEVA_WORK/cosmos-framework" || exit 2
export PYTHONPATH="$ZEVA_WORK/cosmos-framework"
# 33 秒内应看到 "ready"；未 ready 就杀掉
timeout 300 "$PY" -u -m cosmos_framework.scripts.action_policy_server_robocasa365_zeva \
  --checkpoint-path "$ZEVA_RELEASE/weights/stage2_iter_000005000" \
  --allow-dcp-checkpoint \
  --experiment action_policy_robocasa365_atomic5_zeva \
  --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
  --task-context-bank "$ZEVA_RELEASE/weights/stage1/train_memory_effect_v3.pt" \
  --cte-checkpoint "$ZEVA_RELEASE/weights/stage1/cte_step_000500.pt" \
  --static-task-context-checkpoint "$ZEVA_RELEASE/weights/stage3_iter_000005000/best.pt" \
  --static-task-context-top-k 5 \
  --domain-name robocasa-panda-omron \
  --host 127.0.0.1 --port 8300 \
  --num-steps 30 --guidance 3.0 --shift 5.0 \
  --action-dim 7 --action-chunk-size 32 --conditioning-fps 20 \
  --image-height 256 --image-width 512 \
  --format-prompt-as-json --no-use-state --history-length 0 \
  --output-dir /tmp/zeva_server 2>&1 | tee /tmp/zeva_server.log | grep -iE "ready|error|traceback" &
SRV=$!
sleep 300 && kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
