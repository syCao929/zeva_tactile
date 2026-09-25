# Zeva on UR7e + XHand

在自有机器人数据上复现 **Zeva（In-Context Causal Learning）** 的完整训练与部署流水线。

Zeva 让策略记住"之前几次尝试里我的动作造成了什么后果"，并把这段因果记忆喂回给策略。
本仓库补上了官方 release **完全没有开源**的训练链路——从 CTE 训练循环到数据契约层，
再到策略注入与真机服务端。

| | |
|---|---|
| 机器人 | UR7e + XHand（18 自由度：6 臂关节 + 12 手关节）|
| 任务 | `press_button_4_times`（101 episodes / 15 fps）|
| 硬件 | 1 节点 8×A800-80G |
| 基座 | `nvidia/Cosmos3-Nano`（30 GB DCP）+ Qwen3-VL-8B + Wan2.2 VAE |

> **2026-09-25 Cosmos 当前入口：**基础策略使用 `v3-joint18-20260925`（18 维臂/手关节 proprio）。
> V2 策略及其派生 Stage-2 已退役；新的 Cosmos baseline / 触觉 Stage-2 均须从选定的已保存 V3 检查点训练。
> CTE v4、对应缓存与 task-context bank 继续共享，具体状态见第 9 节。
> 新增的 **π0 基座、Zeva 和触觉对照版本**见 [PI0_ZEVA.md](PI0_ZEVA.md)。该入口使用独立环境和 π0 权重，本轮完成代码与CPU验证，尚未启动正式训练。
>
> **本仓库只包含复现所需的改动**，不含 Cosmos Framework 上游代码之外的第三方资源。
> 2026-09-21 之前的详细排障记录（权重校验、网络限制、逐个 bug 的定位过程）
> 存档在 [`NOTES_archive_20260921.md`](NOTES_archive_20260921.md)。

---

## 0. 快速开始

### 0.1 这个仓库里有什么

```
cosmos-framework/     上游 Cosmos Framework + 本仓库全部复现改动
env.sh                唯一的环境入口（所有路径从这里来）
tools/                数据转换、三个阶段的启动脚本、环境自检
examples/             训练 launcher 与 TOML
NOTES_archive_*.md    排障记录（踩过的坑、逐 bug 定位过程）
```

**仓库里没有**：权重、Python 环境、数据集、训练产物——都在 `.gitignore` 里，
需要按 [1.3](#13-权重清单) 自行准备。

### 0.2 拿到代码

```bash
git clone https://github.com/syCao929/zeva_tactile.git zeva-work
cd zeva-work
```

仓库根目录就是 `$ZEVA_WORK`。`env.sh` 用 `BASH_SOURCE` 自定位，**换机器不用改任何路径**——
只要保持内部的相对布局（`cosmos-framework/`、`models/`、`datasets/`、`runs/`、`envs/`）。

### 0.3 还要自备三样

| 缺什么 | 怎么来 |
|---|---|
| **权重** | 见 [1.3](#13-权重清单)。`tools/fetch-weights.sh` 能拉一部分，但注意它会下载 91 GB 的官方 release——**那份对本复现没用**，见 1.3 的说明 |
| **Python 环境** | `envs/zeva/`（Python 3.13 + torch 2.10.0+cu128 + flash_attn）。同构机器可直接拷，否则按 [1.2](#12-激活环境) 重建 |
| **数据** | 自己的 LeRobot v2.1 数据集，按 [1.4](#14-数据) 转成训练布局 |

### 0.4 端到端最短路径

```bash
source env.sh
bash tools/verify-env.sh                 # 环境自检（A 档不需要权重）

python tools/convert_xhand_dataset.py    # 数据 → 训练布局（零拷贝，只建软链）

# 三个阶段，每个都有自己的启动脚本，可独立停止
tools/run-xhand-train.sh   start v3-joint18-YYYYMMDD       # ① 基础策略      见第 3 节
CTE_CACHE="$ZEVA_WORK/datasets/xhand_cte_cache_v2" tools/run-cte-train.sh start cte-YYYYMMDD   # ② CTE           见第 4 节
# ③ 特征缓存 + 注入训练                          见第 5 节

# 部署
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_xhand ...
```

**没时间细读的话**：第 3、4、5 节各自是「启动命令 → 可调参数 → 判读日志」，
第 6 节是三种部署方式和一条验证命令。第 8 节全是踩过的坑，遇到怪问题先翻那里。

---

## 目录

- [0. 快速开始](#0-快速开始)
- [1. 环境](#1-环境)
- [2. 方法概览](#2-方法概览)
- [3. 阶段一：基础策略微调](#3-阶段一基础策略微调)
- [4. 阶段二：CTE 训练](#4-阶段二cte-训练)
- [5. 阶段三：特征缓存与注入训练](#5-阶段三特征缓存与注入训练)
- [6. 部署](#6-部署)
- [7. 可调参数速查表](#7-可调参数速查表)
- [8. 已知问题](#8-已知问题)
- [9. 现状与局限](#9-现状与局限)

---

## 1. 环境

### 1.1 目录结构

所有路径都挂在 `$ZEVA_WORK`（仓库根目录，本机为 `/workspace/mnt/sqzhang26/zeva-work`）。

```
$ZEVA_WORK/
├── env.sh                  ← 唯一的入口，所有变量从这里来
├── cosmos-framework/       Cosmos Framework + 本仓库的全部改动
├── envs/zeva/              conda 环境（Python 3.13.15，422 包）
├── models/                 权重
├── datasets/               数据与中间产物
├── runs/                   训练输出（检查点、配置快照）
├── logs/                   训练日志
└── tools/                  启动脚本
```

### 1.2 激活环境

```bash
cd /workspace/mnt/sqzhang26/zeva-work
source env.sh
```

`env.sh` 会设置好 `ZEVA_WORK` / `PYTHONPATH` / `PATH`、HF 镜像、各级缓存目录，
以及后面所有命令要用的环境变量：

| 变量 | 指向 |
|---|---|
| `XHAND_DATA_ROOT` | 转换后的 LeRobot 数据树 |
| `XHAND_ACTION_STATS_PATH` | `meta/feature_stats_compact.json`（驱动 minmax 归一化）|
| `BASE_CHECKPOINT_PATH` | 基座 DCP |
| `WAN_VAE_PATH` | Wan2.2 VAE 权重 |
| `QWEN_VLM_PATH` | Qwen3-VL-8B |
| `IMAGINAIRE_OUTPUT_ROOT` | 训练输出根（`runs/`）|

> `env.sh` 可重复 source。若设置了 `ZEVA_SKIP_EGL_WARNING=1` 可静默 libEGL 提示。

**在新机器上重建环境**（本机版本：Python 3.13.15 / torch 2.10.0+cu128 / transformers 4.57.6）：

```bash
conda create -p envs/zeva python=3.13 -y
tools/uv pip install --python envs/zeva/bin/python \
  --group cu128 --group policy-server -e cosmos-framework
# mujoco 不在 policy-server 组里，需要时单独补：
tools/uv pip install --python envs/zeva/bin/python --group libero -e cosmos-framework
```

装完用自检确认：

```bash
source env.sh && bash tools/verify-env.sh
```

自检分三档——[A] 不需要任何权重（依赖 / CUDA / 单测 / 配置组装 / 数据加载），
[B] 检查权重就位，[C] 真起 policy server。**只有 A 档全绿才说明环境没问题。**

> ⚠️ `cosmos-framework/.venv` 是早期 uv 留下的空壳（只有 `_virtualenv.py`），
> `env.sh` 里的 `UV_PROJECT_ENVIRONMENT` 仍指向它。实际环境是 `envs/zeva`，
> 两者别混用——上面的命令都显式 `--python envs/zeva/bin/python`。

### 1.3 权重清单

| 项 | 路径 | 大小 | 训练 | 部署 |
|---|---|---:|---|---|
| 基座（DCP）| `models/cosmos3-nano/` | 30 GB | 必需 | — |
| Qwen3-VL-8B | `models/Qwen3-VL-8B-Instruct/` | 41 GB | 必需 | **只用 tokenizer（11 MB）** |
| Wan2.2 VAE | `models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | 2.82 GB | 必需 | 必需 |
| Zeva release | `models/zeva/` | 91 GB | **不需要** | **不需要** |

> **部署只需要 Qwen 的 tokenizer，不需要那 41 GB 权重。** 策略的视觉通路是
> 「图像 → Wan VAE → latent → `vae2llm`」，**不经过 Qwen 的视觉塔**；Qwen 只负责
> 把文字指令编码成 token。实测：服务端启动日志里 `Qwen3-VL-8B-Instruct/model-*.safetensors`
> 的出现次数是 **0**。迁移部署只需带那 7 个 tokenizer 文件（`tokenizer.json` /
> `vocab.json` / `merges.txt` / `tokenizer_config.json` / `chat_template.json` /
> `generation_config.json` / `config.json`，共 11 MB）——已用 `HF_HUB_OFFLINE=1`
> 验证过。启动时要用覆盖参数指到本地目录，见 [6.2(#6-部署)。

> ⚠️ **官方 91 GB 的 Zeva release 权重对本复现没有用**。它是 RoboCasa 专用的推理产物，
> 里面的 CTE 是在 RoboCasa 上训的、stage2 模块绑定 RoboCasa 的动作维度。
> 我们要在自己的数据上重新训这两者，所以只需要基座 + 两个 tokenizer 权重。
> （`models/zeva/weights/` 是空目录。）
> `tools/fetch-weights.sh` 会拉这份 91 GB——**对本复现是纯浪费**，跳过它。

> ⚠️ **VAE 别选错仓库**：必须是 `Wan2.2-TI2V-5B` 里的 `Wan2.2_VAE.pth`。
> `Wan2.2-T2V-A14B` / `I2V-A14B` 里是 `Wan2.1_VAE.pth`（旧版，压缩率不同，加载会失败）。

> **基座说明**：当前 `BASE_CHECKPOINT_PATH` 指向 `models/cosmos3-nano`，
> 即 `nvidia/Cosmos3-Nano` 的原始基座（已转 DCP）。
> 实验配置的本意是改用后训练版 `nvidia/Cosmos3-Nano-Policy-DROID`（动作头已训过）——
> 若换过去，把它转成 DCP 后覆盖 `BASE_CHECKPOINT_PATH` 即可，配置不用改。

### 1.4 数据

源数据（`TactileTTT/data/press_button_4_times`，36 GB）**只读**，与其他项目共用。
转换脚本不复制、不改写，只新建目录树 + 软链：

```bash
source env.sh
python tools/convert_xhand_dataset.py
```

实际占用 **211 KB**。产出布局：

```
datasets/press_button_4_times_merged_filtered/
└── xhand/PressButton4Times/lerobot_v21/lerobot/
    ├── meta/{info.json, episodes.jsonl, tasks.jsonl, feature_stats_compact.json}
    ├── data   -> 软链源数据
    └── videos -> 软链源数据
```

语义映射（在加载器里定义，不动数据）：

| 项 | 源 | 目标 |
|---|---|---|
| embodiment | `ur7e_xhand` | `ur7e-xhand`（domain_id 23）|
| action | `[18]` 关节位置 | `full18`（6 臂 + 12 手）|
| state | `[1972]` | `joint18`（6 臂关节 + 12 手关节位置，与动作顺序一致）|
| 相机 | `cam_front` / `cam_left` / `cam_right` | 外部 / 外部 / **腕部**；三路全部输入，腕部在上、两个外部视角在下 |
| fps | 15 | 15（非加载器默认的 20，必须显式传）|

> `joint18` 是 proprio 输入。触觉版另取原始 state 的连续 30 帧窗口，经冻结 encoder 和 BIT 注入策略；
> baseline 不启用该分支。触觉部署协议见 [触觉服务文档](cosmos-framework/docs/xhand_tactile_serving.md)。

2026-09-25 三视角版本：旧的 `v3-joint18` 策略与 CTE v4 使用的是两路外部相机，
不能直接作为新布局的训练结果。新版本依次训练三视角 Stage 1、生成
`datasets/xhand_cte_cache_threeview`、训练 CTE、生成
`datasets/xhand_cte_features_threeview`，再用原有 `tools/train-cosmos-baseline.sh`
和 `tools/train-cosmos-tactile.sh` 训练 Stage 3，且使用新的 run/pair 名称。
Stage 3 默认启用 PIM；每个查询使用同任务的另一条训练示范作为支持记忆，排除
自身与验证集。该设定训练示范条件策略，不把独立示范冒充同场景失败重试。
PIM 神经模块由策略损失训练，非参数记忆的写入、合并、检索不做梯度更新。

相机输入预览（与训练共用拼图函数）：

```bash
python tools/preview_xhand_cameras.py \
  --episode-root datasets/press_button_4_times/press_button_0 \
  --episode-id 0 --time 3 --output plots/xhand-threeview-preview.png
```

---

## 2. 方法概览

Zeva 由三个训练阶段组成，产物逐级依赖：

```
 ┌─ 阶段一 ────────────────────────────────────────────────┐
 │  Cosmos3-Nano VLA 微调                                    │  137 GB/检查点
 │  loss: 视频流匹配 ×1 + 动作流匹配 ×10                      │
 └─────────────────────────────────────────────────────────┘
                            ↓ 冻结
 ┌─ 阶段三：注入 ──────────────────────────────────────────┐
 │  只训 behavior_pbd / behavior_adapter /                    │  137 GB/检查点
 │       behavior_global_projector                            │
 │  loss: behavior_prior_nll ×0.01                            │
 └─────────────────────────────────────────────────────────┘
                            ↑ 提供 behavior_* 张量
 ┌─ 阶段二 ────────────────────────────────────────────────┐
 │  CausalTransitionEncoder（9.49M 参数，从零训）             │  39 MB/检查点
 │  loss: 5 项复合，主力是 effect 对比损失                     │
 └─────────────────────────────────────────────────────────┘
```

**三者训的是完全不同的东西**：

| 阶段 | 网络 | 在学什么 | 规模 |
|---|---|---|---|
| 一 | Cosmos3-Nano VLA | 「看画面 + 听指令 → 出动作」的本能 | ~30 GB 权重 |
| 二 | CausalTransitionEncoder | 「刚才那 16 步造成了什么后果」 | 9.49M 参数 |
| 三 | 三个小模块（策略冻结）| 「把因果记录翻译成策略听得懂的话」 | 极小 |

### 推理时的数据流

```
初始观测 ─→ VAE ─→ 冻结策略 VLM readout ─→ [stage3 head] ─→ task-context bank
                                                                   │
                                                         behavior_global[256]
                                                                   │
边界帧序列 ─→ CTE ─┬─→ behavior_phase[128]                         │
                   └─→ behavior_effect[4,128] + valid              │
                              │                                    │
                              └──────────────┬─────────────────────┘
                                             ▼
                                stage2 模块（behavior_pbd / adapter）
                                             │
                                ┌────────────┴────────────┐
                                ▼                         ▼
                          prefix token            动作先验（高斯）
                                │                         │
                                └──────→ 冻结策略 ←────────┘
                                            │
                                       输出动作块
```

> **单任务不需要 stage3**。全仓库 `bidirectional_supervised_contrastive_loss` 零调用点，
> 官方也没训过它。单任务下 bank 只有 1 条 entry，检索是恒等映射，
> 服务端用 `--task-context-instruction` 直接取回即可。多任务时才需要训。

---

## 3. 阶段一：基础策略微调

在自有数据上微调 Cosmos3-Nano，得到一个能用的 VLA。所有后续阶段都建立在它之上。

### 启动

```bash
cd $ZEVA_WORK
tools/run-xhand-train.sh start v3-joint18-20260925
```

**run 名同时决定检查点和日志的位置**，**同名 = 续训，换名 = 新开一个 run**：

```
runs/zeva/action_xhand/<run名>/checkpoints/iter_XXXXXXXX/
logs/<run名>.log
```

```bash
# 管理
tools/run-xhand-train.sh status [名]   # 进度 / loss / GPU
tools/run-xhand-train.sh tail   [名]   # 跟踪日志（Ctrl-C 只退 tail）
tools/run-xhand-train.sh stop          # 停止（⚠️ 不存检查点，见 8.2）
tools/run-xhand-train.sh runs          # 列出所有 run 及占用
tools/run-xhand-train.sh fresh [名]     # 归档该 run，下次从基座重训
tools/run-xhand-train.sh rm <名>        # 删除（二次确认）
```

### 可调参数

```bash
MAX_ITER=20000 SAVE_ITER=1000 tools/run-xhand-train.sh start v3-joint18-20260925
EXTRA_OVERRIDES="optimizer.lr=1e-4" tools/run-xhand-train.sh start v3-joint18-20260925
```

| 变量 | 默认 | 说明 |
|---|---:|---|
| `MAX_ITER` | 5000 | 训练步数 |
| `SAVE_ITER` | 500 | 检查点间隔（每个 137 GB，存档约 2 分钟）|
| `NPROC_PER_NODE` | 8 | 卡数 |
| `EXTRA_OVERRIDES` | — | 透传给 Hydra 的额外覆盖 |

### 判读日志

```
[RANK 0] 4136 : iter_speed 5.50 seconds per iteration | Loss: 0.2549
stage2_loss_components: flow_matching_loss_vision=0.038, flow_matching_loss_action=0.0005
```

- `flow_matching_loss_action` 是主角（权重 10），应从 2.6 降到 1e-3 量级
- `flow_matching_loss_vision` 权重 1，通常稳定在 0.01~0.1

### 实测

| iteration | vision loss | action loss |
| ---: | ---: | ---: |
| 0 | 0.1415 | **2.6169** |
| 9 | 0.0622 | **0.9053** |
| 4136 | — | ~0.25（总）|

---

## 4. 阶段二：CTE 训练

训练 `CausalTransitionEncoder`——一个 9.49M 参数的小网络，把「视觉变化 + 动作」压成
描述因果后果的码。**它不依赖策略，可以和阶段一并发跑。**

### 前置：VAE latent 缓存

编码是最贵的一步，只做一次：

```bash
source env.sh && cd cosmos-framework
PYTHONPATH=. python -m cosmos_framework.zeva_training.vae_cache \
  --vae-path "$WAN_VAE_PATH" --dataset-root "$XHAND_DATA_ROOT" \
  --output "$ZEVA_WORK/datasets/xhand_cte_cache_v2"
```

101 episodes → 11,883 latent 帧，608 MB，约 30 分钟。**可断点续跑**，已编码的会跳过。

> ⚠️ 编码会显著抢占 GPU（实测让策略训练的迭代从 3.5s 掉到 7s）。
> manifest 与磁盘不一致时（例如中途被 `kill`），加 `--rebuild-manifest` 免 GPU 修复。

### 启动

```bash
cd $ZEVA_WORK
CTE_CACHE="$ZEVA_WORK/datasets/xhand_cte_cache_v2" tools/run-cte-train.sh start cte-v4-20260924     # 同名即续训
```

产物：

```
runs/zeva_cte/<run名>/cte_step_XXXXXX.pt   # 39 MB，推理端可直接加载
runs/zeva_cte/<run名>/cte_latest.pt        # 含优化器状态，供续训
logs/<run名>.log
```

### 可调参数

```bash
STEPS=3000 SAVE_EVERY=500 CTE_EXTRA="--effect-diversity-weight 1" \
  CTE_CACHE="$ZEVA_WORK/datasets/xhand_cte_cache_v2" tools/run-cte-train.sh start cte-v4-20260924
```

| 变量 | 默认 | 说明 |
|---|---:|---|
| `STEPS` | 500 | 训练步数（同 run 名续训时填新的总步数）|
| `SAVE_EVERY` | 100 | 检查点间隔。**每次存档顺带跑一遍完整验证集（约 55 秒）**，长跑要调大 |
| `BATCH_SIZE` | 8 | |
| `CTE_LR` | 1e-4 | |
| `CTE_WINDOW_LATENTS` | 17 | 窗口长度。**改了必须同步服务端的 `_CTE_MAX_BOUNDARY_FRAMES`** |
| `CTE_NUM_WORKERS` | 4 | NFS 上别调高（见 8.1）|
| `CTE_EXTRA` | 空 | 透传给训练脚本的额外参数，用于损失权重（见下）|

**损失权重**（`CTE_EXTRA` 透传，默认全为原值）：

```bash
STEPS=3000 CTE_EXTRA="--effect-diversity-weight 1" \
  CTE_CACHE="$ZEVA_WORK/datasets/xhand_cte_cache_v2" tools/run-cte-train.sh start cte-v4-20260924
```

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--effect-diversity-weight` | 0 | **注入端防坍缩，强烈建议开成 1**，见 [8.5](#85-effect_post-方向坍缩已修) |
| `--effect-variance-weight` | 1.0 | VICReg 方差项。**调大没用**——实测调到 100 余弦只降 2%，有效秩反而掉 3 倍 |
| `--effect-contrastive-weight` | 1.0 | 对比损失，是维持多样性的主力，**别调小**（调到 0.2 余弦反而升到 0.99）|
| `--effect-covariance-weight` | 0.04 | VICReg 协方差项 |
| `--effect-align-weight` | 0.1 | pre/post 余弦一致 |

### 判读日志

```
step 3000/3000  total=0.4040 action=0.0089 vision=0.0028 task=0.0000
                phase=0.0007 effect=1.5686 (contrast=0.612 var=0.924)
loss weights: effect_contrastive_weight=1  effect_diversity_weight=1  effect_variance_weight=1 ...
```

约 **0.2~3 秒/步**，3000 步约 12–15 分钟。开头的 `loss weights:` 一行会回显实际生效的权重
（确认 `CTE_EXTRA` 有没有传进去）。

---

## 5. 阶段三：特征缓存与注入训练

把 CTE 的输出变成策略能吃的 `behavior_*` 张量，然后冻结策略、只训 Zeva 模块。
**这三步是整个 release 里完全缺失的那条链路。**

### 5.1 CTE 特征缓存

当前共享资源是 CTE v4 @3000 和 `xhand_cte_features_v4`。`xhand_cte_cache_v2` 是其 VAE latent 缓存，
名称中的 v2 与已退役的 V2 策略无关；这些共享资源继续保留，已有完整缓存时无需重复生成。

```bash
cd $ZEVA_WORK/cosmos-framework && source $ZEVA_WORK/env.sh
PYTHONPATH=. python -m cosmos_framework.zeva_training.cte_features \
  --cte-checkpoint "$ZEVA_WORK/runs/zeva_cte/cte-v4-20260924/cte_step_003000.pt" \
  --latent-cache   "$ZEVA_WORK/datasets/xhand_cte_cache_v2" \
  --output         "$ZEVA_WORK/datasets/xhand_cte_features_v4"
```

约 2.5 分钟，产物 5.5 MB。每个 boundary（每 4 个原始控制步）存一行 `phase[128]` +
最近 4 个 `effect_post[4,128]`（右对齐）+ valid。

### 5.2 task-context bank

```bash
PYTHONPATH=. python -m cosmos_framework.zeva_training.build_task_context_bank \
  --feature-cache "$ZEVA_WORK/datasets/xhand_cte_features_v4" \
  --output        "$ZEVA_WORK/datasets/xhand_task_context_bank.pt" --check
```

`--check` 会重新加载并断言 bank 里的 256 维向量与训练时 wrapper 产出的**完全一致**——
这是 train/serve 一致性的唯一保证。

### 5.3 注入训练

新的 baseline 和触觉 Stage-2 尚待从同一份选定的 V3 检查点训练。先查看
`runs/zeva/action_xhand/v3-joint18-20260925/checkpoints/`，把 `V3_POLICY_CHECKPOINT`
设为其中实际保存完整的 `iter_XXXXXXXX` 目录；不要假定存在 4000 步检查点。
以下是 baseline 的启动命令；触觉版使用对应 tactile recipe，保持基座、CTE、数据和 seed 一致。

```bash
cd $ZEVA_WORK
: "${V3_POLICY_CHECKPOINT:?请先选择已保存的 v3-joint18 检查点目录}"
test -f "$V3_POLICY_CHECKPOINT/model/.metadata" || exit 1
STAGE2_POLICY_CHECKPOINT="$V3_POLICY_CHECKPOINT" \
ZEVA_FEATURE_CACHE="$ZEVA_WORK/datasets/xhand_cte_features_v4" \
  tools/run-xhand-zeva-train.sh start
```

> ⚠️ `STAGE2_POLICY_CHECKPOINT` 要指向 **`iter_XXXXXXXX` 目录本身**，不是 `checkpoints/`。
> DCP loader 会自己往后拼 `model/`。

| 变量 | 默认 | 说明 |
|---|---:|---|
| `MAX_ITER` | 2000 | 训练步数 |
| `SAVE_ITER` | 500 | 检查点间隔（同样 137 GB/个，即使只训了几个小模块）|

产物：`runs/zeva/zeva_xhand/action_policy_xhand_zeva/checkpoints/iter_XXXXXXXX/`

### 判读日志

```
behavior_prior_nll = 2.203 → 1.008 → … → 0.906    (iter 0→10)
```

`behavior_prior_nll`（权重 0.01）是**整个 Zeva 里唯一直接监督"记忆有没有用"的地方**：
看着记忆猜接下来该怎么动，猜得准不准就是它。

---

## 6. 部署

服务端是 `cosmos_framework.scripts.action_policy_server_xhand`，走 openpi 的
websocket + msgpack 协议。V3 基座需要正确的 18 维 proprio；Stage-2 需要连续控制动作与 CTE 历史一致，
触觉版还要求每个控制帧的触觉窗口。客户端适配与检查命令见
[触觉服务文档](cosmos-framework/docs/xhand_tactile_serving.md)。

### 6.1 方式一：VLA 基础策略

使用阶段一 V3 joint18 检查点，输入包含视觉和 proprio。先按第 5.3 节设置 `V3_POLICY_CHECKPOINT`。

```bash
cd $ZEVA_WORK/cosmos-framework && source $ZEVA_WORK/env.sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $ZEVA_WORK/envs/zeva/bin/python \
  -m cosmos_framework.scripts.action_policy_server_xhand \
  --checkpoint-path "${V3_POLICY_CHECKPOINT:?请先选择已保存的 v3-joint18 检查点目录}" \
  --allow-dcp-checkpoint \
  --experiment action_policy_xhand_nano \
  --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
  --action-stats-path "$XHAND_ACTION_STATS_PATH" \
  --domain-name ur7e-xhand \
  --resolution 256 --action-dim 18 --conditioning-fps 15 --proprio-dim 18 \
  --image-height 256 --image-width 512 --action-chunk-size 32 --history-length 0 \
  --num-steps 30 --guidance 3.0 --shift 5.0 \
  --host 0.0.0.0 --port 8990
```

### 6.2 方式二：Zeva（CTE + bank + Stage-2）

**先完成新的 V3 Stage-2 训练，再设置 `V3_STAGE2_CHECKPOINT` 为其实际保存的检查点。**
当前没有在此文档中指定已可部署的 V3 Stage-2 产物；基础策略检查点不能代替它。

```bash
cd $ZEVA_WORK/cosmos-framework && source $ZEVA_WORK/env.sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $ZEVA_WORK/envs/zeva/bin/python \
  -m cosmos_framework.scripts.action_policy_server_xhand \
  --checkpoint-path "${V3_STAGE2_CHECKPOINT:?请先训练并选择 V3 Stage-2 检查点}" \
  --allow-dcp-checkpoint \
  --experiment action_policy_xhand_zeva \
  --experiment-overrides \
      "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
      "model.config.vlm_config.tokenizer.pretrained_model_name=$ZEVA_WORK/models/Qwen3-VL-8B-Instruct" \
  --action-stats-path "$XHAND_ACTION_STATS_PATH" \
  --domain-name ur7e-xhand \
  --resolution 256 --action-dim 18 --conditioning-fps 15 --proprio-dim 18 \
  --image-height 256 --image-width 512 --action-chunk-size 32 --history-length 0 \
  --num-steps 30 --guidance 3.0 --shift 5.0 \
  --cte-checkpoint "$ZEVA_WORK/runs/zeva_cte/cte-v4-20260924/cte_step_003000.pt" \
  --task-context-bank "$ZEVA_WORK/datasets/xhand_task_context_bank.pt" \
  --task-context-instruction PressButton4Times \
  --host 0.0.0.0 --port 8990
```

日志出现 `[robolab-policy-server] ready` 即可接客户端。

> **第二个 `--experiment-overrides` 别漏。** 配置里 Qwen 的 `pretrained_model_name` 默认是
> HF 仓库名 `Qwen/Qwen3-VL-8B-Instruct`，服务端会去联网解析。如果目标机连不上 HF、
> 也没有 HF 缓存，**服务端起不来**。这一行把它指到本地目录即可——已用
> `HF_HUB_OFFLINE=1` 验证过本地 tokenizer 目录可用（见 [1.3](#13-权重清单)）。

> **`--task-context-instruction` 和 `--static-task-context-checkpoint` 必须二选一**
> （服务端强制）。单任务走前者；多任务时训好 stage3 head 后改用后者。

### 6.3 方式三：消融（同一份检查点，开关切换）

Zeva 的核心主张是"跨尝试的因果记忆有效"。验证它只需在同一次部署里改开关：

| 开关 | 作用 |
|---|---|
| `--disable-policy-injection` | 置零 policy adapter 和 task-context projector（等价于注入被关）|
| `--bit-mode zero` | effect 历史清零，但保留时间可用性（掩码不变）|
| `--bit-mode shuffled` | 反转已完成的 effect 槽位，保留右对齐的因果位置 |

当前 V3 触觉方案只启用 attempt 内 BIT；PIM（跨尝试记忆）不计入这轮已完成的训练与部署链路（见 9）。

> ⚠️ `--bit-mode` 和服务端的 `disable_policy_injection` 都是**诊断用途**，
> 不要用在正式跑批里。

### 6.4 部署前验证

一条命令验证整条服务链路（启动服务端 → 走真实协议发请求 → 断言 CTE 路径活着）：

```bash
cd $ZEVA_WORK/cosmos-framework && source $ZEVA_WORK/env.sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m cosmos_framework.zeva_training.verify_serving \
  --checkpoint "${V3_STAGE2_CHECKPOINT:?请先训练并选择 V3 Stage-2 检查点}" \
  --cte-checkpoint "$ZEVA_WORK/runs/zeva_cte/cte-v4-20260924/cte_step_003000.pt" \
  --task-context-bank "$ZEVA_WORK/datasets/xhand_task_context_bank.pt" \
  --task-context-instruction PressButton4Times
```

新 V3 Stage-2 部署后应检查的输出形状（不是已完成的验证记录）：

```
server ready
  request 0: actions (32, 18)
  ...
boundary_frames per request: [1, 2, 3, 4, 5, 6]
action shapes:               [(32, 18) × 6]

服务端加载所选 V3 Stage-2、协议返回 (32, 18)、CTE 路径活跃
```

**`boundary_frames` 递增这一条最关键**：服务端在边界缓冲建不起来时会**静默回退**
到全零特征，客户端拿到的动作看起来完全正常。只有查这个计数才能确认特征是真的。

---

## 7. 可调参数速查表

### 训练步数

| 阶段 | 参数 | 默认 | 实测耗时 | 建议 |
|---|---|---:|---|---|
| 一 | `MAX_ITER` | 5000 | 约 4~6 秒/步 | 5000 起；曲线还在降就加倍 |
| 二 | `STEPS` | 500 | 约 0.2~3 秒/步 | 3000（15 分钟，很便宜）|
| 三 | `MAX_ITER` | 2000 | 约 6 秒/步 | 2000；`behavior_prior_nll` 变负就该停 |

### 检查点间隔

| 阶段 | 参数 | 默认 | 单个大小 | 存档耗时 |
|---|---|---:|---:|---|
| 一 | `SAVE_ITER` | 500 | 137 GB | 约 2 分钟 |
| 二 | `SAVE_EVERY` | 100 | 39 MB | < 1 秒 + 55 秒验证 |
| 三 | `SAVE_ITER` | 500 | 137 GB | 约 2 分钟 |

> 阶段一和三的检查点是 DCP **全量存**——即使阶段三只训了三个小模块，也还是 137 GB/个。
> 跑满会迅速吃掉几 TB：阶段一 5000 步 = 1.37 TB，阶段三 2000 步 = 550 GB。

### 学习率

| 阶段 | 位置 | 值 |
|---|---|---:|
| 一 | `examples/toml/sft_config/action_policy_xhand_nano.toml` → `[optimizer] lr` | 2e-4 |
| 二 | `CTE_LR` 环境变量 | 1e-4 |
| 三 | TOML `[optimizer] lr` + 配置里的 `lr_multipliers`（Zeva 模块 ×5）| 2e-4 |

### 其他常用覆盖

```bash
# 阶段一/三：透传任意 Hydra 覆盖
EXTRA_OVERRIDES="optimizer.lr=1e-4 dataloader_train.max_samples_per_batch=8" \
  tools/run-xhand-train.sh start v3-joint18-20260925

# 阶段一：NFS 上 num_workers 别超过 4（见 8.1）
EXTRA_OVERRIDES="dataloader_train.dataloader.num_workers=8" ...

# 阶段三：用字符串传覆盖（bash 数组不会导出到子进程，见 8.2）
ZEVA_TAIL_OVERRIDES="trainer.max_iter=5 checkpoint.save_iter=5" \
  bash examples/launch_sft_action_policy_xhand_zeva.sh

# 阶段二：损失权重（默认全为原值；diversity 建议开成 1，见 8.5）
CTE_EXTRA="--effect-diversity-weight 1" CTE_CACHE="$ZEVA_WORK/datasets/xhand_cte_cache_v2" tools/run-cte-train.sh start cte-v4-20260924
```

> **特别注意各阶段的开关传法不一样**：阶段一/三是 `EXTRA_OVERRIDES`（Hydra 覆盖），
> 阶段三是 `ZEVA_TAIL_OVERRIDES`（字符串形式的 bash 数组），阶段二是 `CTE_EXTRA`
> （直接透传给 argparse）。传错位置的参数会被**静默忽略**——命令照跑，用的却是默认值。

---

## 8. 已知问题

### 8.1 NFS 上的 dataloader 死锁

官方配方默认 `num_workers=16`（注释写着 "assumes data on local disk"）。
我们的数据在共享 NFS 上，16 × 8 rank = **128 个并发读**会把个别读请求拖死。
症状很隐蔽：**某一个 rank 永远不报 dataloader ready**，其余 7 个卡在 pre-warm barrier，
CPU 几乎为 0、GPU 100%、无任何报错。

已改成 `num_workers=4` / `prefetch_factor=1`。GPU 利用率低时可往上调到 8。

### 8.2 `stop` 不存检查点

`termination_signal_checkpoint` 回调**只处理 SIGUSR1**，且靠 Slurm 哨兵文件
`$SLURM_LOG_DIR/SIGUSR1_RECEIVED` 触发；本机没有 Slurm，回调初始化时就把哨兵路径
置空直接 return，它注册的 SIGTERM 处理器**只打一行日志**。

**所以 `stop` 等于直接杀进程，从上一次定期存档之后的进度全丢。**
想不丢进度就别 stop，让它跑到下一个 `SAVE_ITER`。

### 8.3 代理会劫持 localhost 的 websocket

本机全局设了 `http_proxy`/`https_proxy` 且没有 `no_proxy`，`websockets` 会把
`127.0.0.1` 的连接也走代理并收到 403。客户端需要 `no_proxy=127.0.0.1,localhost`
（`verify_serving.py` 里已自动设置；真机客户端若也配了代理需自己加）。

### 8.4 CTE 的任务聚类目标是死目标

101 个 episode 的 `task_cluster` 全是同一个，`semantic_ids` 恒为 0。
多正样本对比损失在这种情况下**每个样本都是其它所有样本的正样本**，
理论下界是 `log(N-1)`，唯一的下降方向是把所有 embedding 压成一个方向。

已修（`cte_losses.py`：batch 内语义 id 少于 2 个时返回 0，不产生梯度）。
**多任务数据不受影响**——分支只在少于两个不同 id 时触发。

### 8.5 `effect_post` 方向坍缩（已修）

**症状**：`effect_post` 是服务端唯一注入的 effect 张量（`behavior_effect`）。
如果它的跨样本余弦接近 1，说明所有"刚才发生了什么"的报告长得一样——
**策略收到的注入信号是个常数，等价于没注入**，Zeva 的整个主张无从谈起。

实测第 3000 步：

| 张量 | 角色 | cos | 有效秩 |
|---|---|---:|---:|
| `effect_delta_target` | 输入（冻结 VAE 视觉差分）| **−0.000** | 31.3 |
| `effect_outcome_post` | 被 NCE 训练的那侧 | +0.115 | 26.1 |
| `effect_post` | **服务端实际注入的** | **+0.829** | 19.1 |

而且训得越久越糟（500 步 0.963 → 3000 步 0.980）。

**根因不在权重，在结构**。梯度归因（对 `effect_post_raw` 逐项反向测梯度范数）：

```
effect_contrastive   grad norm 6.643   ← 占 99.8%
effect_variance      grad norm 0.137   ← VICReg 只有它的 1/48
方差方向：contrastive ↑变大  variance ↑变大  align ↓变小  covariance ↓变小
```

没有任何一项在主动压小方差——VICReg 自己也在推大，只是信号被对比损失淹没了。

但真正的原因是 `effect_outcome_head = LayerNorm(128) → Linear(128,256)`：
**LayerNorm 按样本重新归一化**，于是对比损失可以靠放大 `effect_post_raw` 里极小的差异
来满足自己，**从来不需要让被注入的那个张量本身变得多样**。

两条错路（都实测过）：

| 尝试 | 结果 |
|---|---|
| `--effect-variance-weight 100` | cos 只从 0.963 降到 0.945，有效秩反而 17.1 → **5.2**（VICReg 靠撑大少数方向就满足了"每维 std ≥ 1"）|
| `--effect-contrastive-weight 0.2` | cos 反而升到 **0.990**——对比损失是维持多样性的主力，动不得 |

**修法**：直接惩罚跨样本余弦（`_effect_diversity`），加在 `effect_post` / `effect_pre` 上。
这个目标没有逃逸口——L2 归一化去掉了尺度，只能靠真正指向不同方向来降低。

```bash
CTE_EXTRA="--effect-diversity-weight 1"
```

**效果**（960 个 effect 样本）：

| 检查点 | 注入码 cos | 相关性 r |
|---|---:|---:|
| v2 @3000（旧） | +0.837 | 0.546 |
| **v3 @2000（新）** | **+0.044** | **0.960** |

- **cos** 越低说明注入码越有区分度
- **r** = 注入码之间的相似度矩阵与真实视觉差异之间的相关性，**越高说明"散开得越有意义"**
  （单纯换成噪声也能压低 cos，但 r 会很低——所以两个指标必须一起看）

@2000 的 r 最高；@3000 是 0.914，说明 diversity 正则在 2000 步后开始过度优化，
所以**选 2000 步而不是 3000 步**。

> release 代码里作者自己记过同类问题（`causal_transition_encoder.py:229-230`），
> 说明这个头本身是脆的。

---

## 9. 现状与局限

### 各阶段状态

截至 2026-09-25 本次文件核验，正式策略入口是 **V3 joint18**。运行状态以进程和最新日志为准；
下表只列已核实保存的资源，不把启动记录当作训练完成。

| 阶段 | 状态 | 产物 |
|---|---|---|
| 一：策略微调 | V3，18 维 proprio；核验时最新已保存 iter 2500 | `runs/zeva/action_xhand/v3-joint18-20260925/checkpoints/` |
| 二：CTE | v4，3000 步；共享保留 | `runs/zeva_cte/cte-v4-20260924/cte_step_003000.pt` |
| 二：VAE latent 缓存 | 共享保留，与策略 V2 无关 | `datasets/xhand_cte_cache_v2/` |
| 三：CTE 特征缓存 | 101 episodes / 11,883 boundary；共享保留 | `datasets/xhand_cte_features_v4/` |
| 三：bank | 1 entry；共享保留 | `datasets/xhand_task_context_bank.pt` |
| 三：baseline / 触觉注入训练 | 待从同一个选定 V3 检查点训练 | 尚未指定新的可部署检查点 |
| 部署 | 触觉链路已做 CPU 语义与协议检查；新的 V3 Stage-2 待训练及部署验证 | [触觉服务文档](cosmos-framework/docs/xhand_tactile_serving.md) |

V2 策略及其派生 Stage-2 已退役，不再用于训练起点、对照或部署。
早期 V1、旧 CTE 实验的数字只作为历史记录；第 8 节的 CTE v2/v3 消融结果不是当前策略版本状态。

### 尚待验证

| 项目 | 状态 |
|---|---|
| 新 baseline / 触觉对照 | 需在同一 V3 基座上训练，再比较实机表现 |
| PIM（跨尝试记忆） | 当前触觉方案只使用 attempt 内 BIT；不将 PIM 计入已完成的触觉链路 |
| stage3 检索头 | 单任务下不作为当前训练步骤 |
| 触觉部署 | CPU 测试通过不代表八卡训练或机器人测试完成 |

### 当前触觉与 proprio 输入

基础策略与触觉版都使用 `joint18` proprio（6 臂关节 + 12 手关节位置）。触觉版另输入
当前时刻及过去的 30 帧原始 state，15 Hz；episode 开头左补零并提供有效掩码。
冻结的逐帧 encoder 输出经 projector、BIT 和 effect head 注入策略，历史只由 BIT 建模。
注入在视觉 effect 的 BOS 替换之后执行：零 gate 保持 baseline 等价，开始接触时也能学习。
phase/confidence 分支保留检查点兼容但冻结，目前不承担训练目标。

### 单任务的限制

数据只有一个 task cluster，所以：

- CTE 的任务聚类目标退化（8.4）
- task-context bank 只有 1 条 entry，检索是恒等映射
- **Zeva 主张的"跨任务检索"这条线无法验证**

但**跨尝试记忆（PIM）在单任务下仍然可测**——`press_button_4_times` 本身就是
重复尝试的任务结构，这正是该任务适合验证 Zeva 的地方。
