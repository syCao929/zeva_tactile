# workspace 根目录 = 本仓库根（env.sh 就在根下）；可用 ZEVA_WORK=... 覆盖。
# 目录约定见 README_ZEVA_ENV.md §1：
#   $ZEVA_WORK/cosmos-framework/   模型代码（NVIDIA cosmos-framework 的复现改动）
#   $ZEVA_WORK/tools/              安装与训练脚本
#   $ZEVA_WORK/{envs,models,datasets,runs,cache,tmp}/   环境、权重、数据、训练输出
export ZEVA_WORK="${ZEVA_WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# uv 工具、下载的 Python 和缓存
export UV_INSTALL_DIR="$ZEVA_WORK/tools/uv"
export UV_PYTHON_INSTALL_DIR="$ZEVA_WORK/tools/python"
export UV_TOOL_DIR="$ZEVA_WORK/tools/uv-tools"
export UV_TOOL_BIN_DIR="$ZEVA_WORK/tools/bin"
export UV_CACHE_DIR="$ZEVA_WORK/cache/uv"

# Python 虚拟环境
# 注：本项目实际装在 conda 环境 envs/zeva（cosmos-framework/.venv 未使用）。
# uv sync 仍指向 .venv，但环境是用 uv pip 装进 conda 的，两者不要混用。
export ZEVA_ENV="$ZEVA_WORK/envs/zeva"
export UV_PROJECT_ENVIRONMENT="$ZEVA_WORK/cosmos-framework/.venv"

# HuggingFace：huggingface.co 在本机不可达（DNS 被污染 + 直连超时），
# 实测 https://hf-mirror.com 可达，故统一走镜像。
# 大文件下载建议用 hf CLI（支持断点续传）并设 HF_HUB_DISABLE_XET=1。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

# 模型及其他缓存
export HF_HOME="$ZEVA_WORK/cache/huggingface"
export XDG_CACHE_HOME="$ZEVA_WORK/cache"
export PIP_CACHE_DIR="$ZEVA_WORK/cache/pip"
export TORCH_HOME="$ZEVA_WORK/cache/torch"
export TORCH_EXTENSIONS_DIR="$ZEVA_WORK/cache/torch_extensions"
export TRITON_CACHE_DIR="$ZEVA_WORK/cache/triton"
export TMPDIR="$ZEVA_WORK/tmp"

# 数据、权重和训练输出
export XHAND_DATA_ROOT="$ZEVA_WORK/datasets/press_button_4_times_merged_filtered"
# 转换脚本生成的精简 action 统计（min/max），驱动 minmax 归一化
export XHAND_ACTION_STATS_PATH="$XHAND_DATA_ROOT/xhand/PressButton4Times/lerobot_v21/lerobot/meta/feature_stats_compact.json"
# 基座：训练起点，必须是 DCP 目录。
# models/cosmos3-nano 已经是 nvidia/Cosmos3-Nano 转好的 DCP（30.35GB/814张量，
# 6 分片与元数据一致，通过 _validate_checkpoint），所以不需要再跑 convert_model_to_dcp。
# 若改用 nvidia/Cosmos3-Nano-Policy-DROID（后训练策略，action heads 已训过），
# 转出来后把它指到 models/base_checkpoint_dcp。
export BASE_CHECKPOINT_PATH="$ZEVA_WORK/models/cosmos3-nano"
export ZEVA_RELEASE="$ZEVA_WORK/models/zeva"
export QWEN_VLM_PATH="$ZEVA_WORK/models/Qwen3-VL-8B-Instruct"
# Wan2.2_VAE.pth 就在整个 TI2V-5B 仓库里（同仓库其余 ~29GB 是 TI2V-5B 模型本身，用不到）
export WAN_VAE_PATH="$ZEVA_WORK/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
# RoboCasa365 模拟器（eval 用）。sim 源码 + 资产需单独获取，见 README_ZEVA_ENV.md
export ROBOCASA365_ROOT="$ZEVA_WORK/sim/robocasa365"

# RoboCasa 无头渲染（docs/reproduce.md 要求 egl）。
# 但本机系统里没有 libEGL（连 NVIDIA 的 libEGL_nvidia.so.0 都没有），
# 一旦设了 MUJOCO_GL=egl 就会让 `import mujoco` 直接抛
# AttributeError: 'NoneType' object has no attribute 'eglQueryString'。
# 因此只在 libEGL 确实存在时才启用，否则留空（模拟器可 import，但无法离屏渲染）。
if ldconfig -p 2>/dev/null | grep -q "libEGL\.so"; then
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
else
  unset MUJOCO_GL PYOPENGL_PLATFORM 2>/dev/null || true
  : "${ZEVA_SKIP_EGL_WARNING:=}"
  if [ -z "$ZEVA_SKIP_EGL_WARNING" ]; then
    echo "[env.sh] 警告: 未找到 libEGL，MUJOCO_GL 未启用。" >&2
    echo "         需 apt-get install -y libegl1 libgl1（或装带 EGL 的 NVIDIA 驱动）" >&2
    echo "         后才能做 RoboCasa 无头渲染。静默此提示: export ZEVA_SKIP_EGL_WARNING=1" >&2
  fi
fi

export IMAGINAIRE_OUTPUT_ROOT="$ZEVA_WORK/runs"
export WANDB_DIR="$ZEVA_WORK/logs"
export WANDB_CACHE_DIR="$ZEVA_WORK/cache/wandb"

export PYTHONPATH="$ZEVA_WORK/cosmos-framework"
export PATH="$ZEVA_WORK/envs/zeva/bin:$ZEVA_WORK/tools/uv:$ZEVA_WORK/tools/bin:$PATH"

# 本机私有路径（源数据所在位置等），不入库。模板见 env.local.sh.example。
if [ -f "$ZEVA_WORK/env.local.sh" ]; then
  # shellcheck disable=SC1091
  . "$ZEVA_WORK/env.local.sh"
fi
