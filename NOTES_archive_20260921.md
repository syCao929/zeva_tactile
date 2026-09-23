# zeva-work 环境说明

Zeva（In-Context Causal Learning）在 Cosmos Framework 上的复现环境。
生成时间：2026-09-20。

## 1. 目录职责

| 路径 | 说明 |
| --- | --- |
| `env.sh` | **入口**。`source env.sh` 后所有变量与 PATH 就位 |
| `envs/zeva/` | 实际使用的 Python 环境（conda 建，Python 3.13.15，422 个包） |
| `cosmos-framework/` | Cosmos Framework @ `ee58e41`，**已 rsync 叠加 Zeva overlay** |
| `Zeva/` | Zeva overlay 原始仓库（只读参照，不要在这里改代码） |
| `tools/uv`, `tools/python` | uv 及它下载的 CPython |
| `cache/` | uv/pip/conda 缓存（约 10GB） |
| `models/`, `datasets/`, `runs/` | 权重、数据、训练输出 |

> `cosmos-framework/.venv` 是早期 uv 建的**空壳**（只有 `_virtualenv.py`），
> 实际环境是 `envs/zeva`。`env.sh` 里的 `UV_PROJECT_ENVIRONMENT` 仍指向它，
> 但环境是用 `uv pip install --python envs/zeva/bin/python` 装的，两者别混用。

## 2. 已完成

- **Python 依赖**：`envs/zeva` 装好 422 个包（`policy-server` + `cu128` 组）。
  清单里剩 17 个全是 darwin/win32/py<3.11 条件包，Linux 上本就无需安装。
  验证通过：torch 2.10.0+cu128、CUDA 8 卡 A800-80G、flash_attn 2.7.4、
  torchcodec 0.10、transformers 4.57.6、cosmos_framework 及全部 zeva 子模块可 import。
- **mujoco 3.3.2**：不在 `policy-server` 组（属 `libero` 组），已单独补装，RoboCasa 模拟器需要。
- **env.sh**：新增 `HF_ENDPOINT`（HF 被墙）、`ROBOCASA365_ROOT`、`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`。
- **数据转换**：TactileTTT `press_button_4_times` → Cosmos 训练布局，零拷贝、源数据未改动。
- **加载器**：`cosmos_framework/data/generator/action/datasets/xhand_lerobot_dataset.py`
  与 embodiment `ur7e-xhand`（domain_id 23）。
- **训练跑通**：8×A800 实测，loss 正常下降（见第 6 节）。
- **真机 server**：`scripts/action_policy_server_xhand.py`，TactileTTT 客户端零代码改动可对接（见第 8 节）。

## 3. 网络限制（重要）

- `huggingface.co` **不可达**（DNS 被污染 + 直连超时）。`hf-mirror.com` 可达但很慢
  （实测 2 KB/s ~ 100 KB/s）。机器全局走内网 HTTP 代理（地址见运维配置）。
- **91GB 的 stage2 权重按此速度需 2~10 天**，不建议在本机拉取。
- pypi / github 经代理尚可（uv 装 5.6GB 包耗时约 1 小时）。

## 4. 权重就位情况

### 4.1 已有（训练所需全部就位）

| 项 | 路径 | 来源 | 大小 | 校验 |
| --- | --- | --- | ---: | --- |
| 基座 DCP | `models/cosmos3-nano/` | `nvidia/Cosmos3-Nano` 已转 DCP | 30.4 GB | ✅ 814 张量 / 6 分片与元数据一致 / 通过 `_validate_checkpoint` |
| Qwen3-VL-8B | `models/Qwen3-VL-8B-Instruct/` | `Qwen/Qwen3-VL-8B-Instruct` | 17.5 GB | ✅ 4 分片 / 750 张量齐全；`AutoConfig`+`AutoTokenizer`+`AutoProcessor` 实测可加载 |
| Wan2.2 VAE | `models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | `Wan-AI/Wan2.2-TI2V-5B` | 2.82 GB | ✅ 字节数与 HF 完全一致；合法 torch 归档 |

> `Wan2.2-TI2V-5B/` 是整个仓库（32 GB），其中只有 `Wan2.2_VAE.pth` 是我们需要的，
> 其余是 TI2V-5B 模型本身。**仓库别选错**：`Wan2.2-T2V-A14B` / `I2V-A14B` 里是
> `Wan2.1_VAE.pth`（旧版 VAE，压缩率不同，加载会失败）。

### 4.2 可选（只有复现 RoboCasa 才需要）

| 项 | 目标路径 | 来源 | 大小 |
| --- | --- | --- | ---: |
| Zeva release | `models/zeva/` | HF `chen123fu/zeva-robocasa` | **91 GB** |
| RoboCasa365 | `sim/robocasa365/` | GitHub `robocasa/robocasa` + 资产 | 数 GB |

> ⚠️ `Zeva/README.md` 里写的权重仓库 `chen123fu/zeva` **不存在或是私有的**（返回 401），
> 公开可用的是 **`chen123fu/zeva-robocasa`**。
>
> 另外：**训练自己的数据完全不需要这 91 GB**——它是推理产物，
> 里面的 Zeva 模块是 RoboCasa 专用的。

### 4.2 仓库布局 ≠ 文档布局

`docs/reproduce.md` 写的路径和 HF 仓库实际布局不一致。下载完成后执行
`bash tools/link-zeva-release.sh` 建软链映射即可让文档命令原样可用：

| 文档里的路径 | 仓库实际路径 |
| --- | --- |
| `weights/stage1/cte_step_000500.pt` | `weights/stage1/zeva_cte.pt` |
| `weights/stage2_iter_000005000/` | `weights/stage2/model/` |
| `weights/stage3_iter_000005000/best.pt` | `weights/stage3/best.pt` |

stage2 是 **8 个真实分片**（元数据 1696 个张量合计 91.05GB），不是副本，
8 个 `__N_0.distcp` 一个都不能少。其中 `net`(bf16) 30.4GB、`net_ema`(fp32) 60.7GB。

### 4.3 系统库：缺 libEGL

本机只有 `libGL.so.1`，**完全没有 libEGL**（连 NVIDIA 的 `libEGL_nvidia.so.0` 都没有）。
`docs/reproduce.md` 要求 `MUJOCO_GL=egl` 做无头渲染，但在缺 libEGL 时设了它会让
`import mujoco` 直接抛 `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`。

`env.sh` 已改成**条件启用**（检测到 libEGL 才设），并会打印一行提示。修复方式二选一：

```bash
apt-get install -y libegl1 libgl1        # 简单，需要 root/网络
# 或重装带 EGL 的 NVIDIA 驱动
```

静默提示：`export ZEVA_SKIP_EGL_WARNING=1`。
sim 源码本身可正常 `import mujoco`，受影响的只有离屏渲染。

### 4.4 可选

- 训练数据集若要用 `robocasa365` 流程，还需 RoboCasa365 的 Atomic-5 数据集。
- 预训练基座 `nvidia/Cosmos3-Nano-Policy-DROID`（若不做全量训练则不需要）。

## 5. 数据转换怎么做的

```bash
source env.sh
python tools/convert_xhand_dataset.py          # 默认源/目标已配好
python tools/convert_xhand_dataset.py --help   # 过滤、链接方式等选项
```

**方法**：不复制、不改写源数据，只新建目录树 + 软链。

```
datasets/press_button_4_times_merged_filtered/
└── xhand/PressButton4Times/lerobot_v21/lerobot/
    ├── meta/{info.json, episodes.jsonl, tasks.jsonl, feature_stats_compact.json}
    ├── data   -> 软链 TactileTTT/data/chunk-000
    └── videos -> 软链 TactileTTT/videos/chunk-000
```

实际占用 **211 KB**（源数据 36GB 原样引用）。`episodes_stats.jsonl`(4GB) 与
`stats.json`(103MB) 也走软链——加载器不读它们。

**语义映射**（在加载器里定义，不动数据）：

| 项 | 源 | 目标 |
| --- | --- | --- |
| embodiment | `ur7e_xhand` | `ur7e-xhand`，domain_id **23** |
| action | `[18]` 关节位置 | `full18`（全部）或 `arm6`（仅臂） |
| state | `[1972]` | `arm6`/`arm22`/`arm34`/`full52` 可选，默认 `arm22` |
| 相机 left | `observation.images.cam_left` | 同 |
| 相机 wrist | `observation.images.cam_front` | **按约定充当 wrist**（源无腕部相机） |
| fps | 15 | 15（加载器需显式传 `fps=15`，非默认 20） |

state 布局：`[0:6]`臂关节pos `[6:12]`臂关节vel `[12:28]`ee_pose `[28:52]`手关节
pos/torque **交错** `[52:1972]`触觉（未使用）。

**验证**：98 个训练 episode 共 42673 个窗口；抽查样本 `video=(3,33,256,512)`
`action=(32,18)` `proprio=(22,)`；98/98 个 episode 末窗口边界全部通过；
`proprio` 与 `action` 的手臂关节值一致。

## 6. 训练：已实测跑通

2026-09-21 用 8×A800 跑通，loss 正常下降：

| iteration | vision loss | action loss |
| ---: | ---: | ---: |
| 0 | 0.1415 | **2.6169** |
| 2 | 0.1941 | 1.4038 |
| 7 | 0.1629 | 0.9771 |
| 9 | 0.0622 | **0.9053** |

### 第一阶段：policy 微调（`run-xhand-train.sh`）

run 名由你指定，它同时决定**检查点和日志的落盘位置**：

```
runs/zeva/action_xhand/<run名>/checkpoints/iter_XXXXXXXX/
logs/<run名>.log
```

**同名 = 续训，换名 = 新开一个 run。**

```bash
cd $ZEVA_WORK

# 首次训练（名字自己定，建议带日期）
tools/run-xhand-train.sh start v1-20260921

# 中断后续训 —— 输入同一个名字即可
tools/run-xhand-train.sh start v1-20260921

# 另起一个实验/改超参后重训 —— 换个名字
tools/run-xhand-train.sh start v2-20260922

# 管理
tools/run-xhand-train.sh status [名]   # 状态 / 进度 / 最新 loss / GPU（省略名字用最近启动的）
tools/run-xhand-train.sh tail [名]     # 跟踪日志（Ctrl-C 只退 tail）
tools/run-xhand-train.sh stop          # 停止整个进程组（SIGTERM，不存检查点！见下）
tools/run-xhand-train.sh runs          # 列出所有 run 及占用
tools/run-xhand-train.sh fresh [名]     # 归档该 run（改名带时间戳），下次从基座重训
tools/run-xhand-train.sh rm <名>        # 删除指定 run（二次确认）
```

可覆盖的参数（环境变量，放在命令前）：

```bash
MAX_ITER=20000 SAVE_ITER=1000 tools/run-xhand-train.sh start v1-20260921
EXTRA_OVERRIDES="optimizer.lr=1e-4" tools/run-xhand-train.sh start v1-20260921
```

### 第二阶段：CTE 训练（`run-cte-train.sh`）

**CTE 完全独立于 policy**：它只读 VAE latent 缓存，不碰 policy 的任何产物，
所以**两个阶段的训练可以并发跑**，各用各的 PID 文件和 run 名，互不干扰。

```bash
cd $ZEVA_WORK
tools/run-cte-train.sh start cte-v1-20260921    # 启动（同名即续训）
```

启动前脚本会自检并打印缓存状态，正常应看到：

```
== 检查环境 ==
  ✅ CTE latent 缓存          (101 个 episode)
== run: cte-v1-20260921 ==
   检查点目录: $ZEVA_WORK/runs/zeva_cte/cte-v1-20260921
   日志文件:   $ZEVA_WORK/logs/cte-v1-20260921.log
   新 run，从随机初始化开始
```

#### 管理

```bash
tools/run-cte-train.sh status cte-v1-20260921
tools/run-cte-train.sh tail   cte-v1-20260921
tools/run-cte-train.sh stop
tools/run-cte-train.sh runs
```

落盘位置（命名规则与第一阶段对应）：

```
runs/zeva_cte/<run名>/cte_step_XXXXXX.pt     # 可直接被推理端加载
runs/zeva_cte/<run名>/cte_latest.pt          # 含优化器状态，供续训
runs/zeva_cte/<run名>/train.log
logs/<run名>.log
```

可覆盖：

```bash
STEPS=2000 SAVE_EVERY=200 BATCH_SIZE=16 tools/run-cte-train.sh start cte-v1-20260921
```

跑起来的日志长这样（实跑数据，`STEPS=2`）：

```
[17:32:22] cache=... window_latents=17 train_windows=9748 val_windows=519 params=9.49M device=cuda
[17:32:25] step 1/2  total=2.8892 action=0.9105 vision=0.3574 task=1.9463 phase=0.0035 effect=4.9270 (contrast=3.843 var=0.975)  3.04s/step
[17:33:18]   saved cte_step_000001.pt  val_total=2.4906
[17:34:12]   saved cte_step_000002.pt  val_total=2.2741
[17:34:12] done. final checkpoint: .../cte_step_000002.pt
```

**速度**：约 **3.0 秒/步**（batch=8），每次存检查点要顺带跑一遍验证集（约 55 秒）。
默认 `STEPS=500 SAVE_EVERY=100` 大约 **30 分钟**跑完，产物约 200 MB。

健康信号：`val_total` 逐次下降；`effect_contrastive`、`effect_align` 逐次下降。

> ⚠️ `effect_variance` 是**损失项**（VICReg 方差项 `relu(1-std).mean()`），
> **目标是 0**（std ≥ 1 时归零），不是"越大越好"。实测 500 步稳定在 0.958，
> 意味着 effect 表示每个维度的 std 只有约 0.04～0.08——**这一项没被优化动**。
> 详见下面"CTE 500 步实测结论"。

#### 两次 CTE 训练的实测对比（2026-09-21）

| | v1（修前）| v2（修后）|
|---|---|---|
| run / 步数 | `cte-v1-20260921` / 500 | `cte-v2-20260921` / **3000** |
| 最终 `val_total` | 1.3795（@100）→ 1.1945 | 0.8108 → **0.5856** |
| 最终 `effect_contrastive` | 2.16 | **1.37** |
| `effect_align` | 0.194 | 0.078（已平）|
| `effect_variance` | 0.958 | 0.932（基本没动）|

**同一步数对比**（都是第 500 步，唯一变量是那个修复）：
`effect_contrastive` **2.1618 → 1.7080**。去掉退化目标后，
有效目标的梯度变纯粹，学得更快——这不只是把 0.39 从 total 里减掉。

**⚠️ 但注入端在退化——训得越久越糟：**

| `effect_post` 跨窗口余弦 | v1@500 | v2@500 | v2@3000 |
|---|---:|---:|---:|
| | +0.963 | +0.963 | **+0.980** |

而同一批窗口上各张量的几何（val 前 48 个窗口）：

| 张量 | 角色 | 平均 cos | 有效秩 |
|---|---|---:|---:|
| `effect_delta_target` | **输入**（冻结 VAE 视觉差分）| **−0.000** | 31.3 |
| `effect_outcome_post` | **被 NCE 训练** | **+0.115** | 26.1 |
| `effect_post` | **server 实际注入的** | **+0.829** | 19.1 |

也就是说：**判别性表示确实学出来了，但不在被注入的那个张量上。**
输入侧是健康的（cos 0、秩 31），所以这是**目标函数问题，不是数据问题**。

注意二者维度不同：`effect_outcome_head` 输出 **256**（`causal_transition_encoder.py:239-241`），
而 server 注入端要 **128**，**不能直接互换**。

release 代码里作者自己留了同类问题的记录（`causal_transition_encoder.py:229-230`）：
> `# ... Passing executed actions here lets post simply echo the action-conditioned`
> `# pre code and caused v2's rank-one collapse.`

说明这个头本身是脆的。我们遇到的更轻（秩 19 而非秩 1），但方向一致。

#### 前置条件：VAE latent 缓存（**已就绪，通常不用管**）

缓存已经编码完成并落在共享盘上：

```
datasets/xhand_cte_cache/episode_XXXXXX.npz   # 101 个，共 608 MB
datasets/xhand_cte_cache/manifest.json
```

编码是最贵的一步（要跑 VAE），**只做一次**。只有在缓存缺失或想换分辨率的
情况下才需要重跑；在已有缓存上重复执行不会重算，已编码的 episode 会跳过：

```bash
source env.sh && cd cosmos-framework
PYTHONPATH=. python -m cosmos_framework.zeva_training.vae_cache \
  --vae-path "$WAN_VAE_PATH" --dataset-root "$XHAND_DATA_ROOT" \
  --output "$ZEVA_WORK/datasets/xhand_cte_cache"
```

> ⚠️ **编码会显著抢占 GPU**（实测让 policy 训练的迭代从 3.5s 掉到 7s）。
> 建议等第一阶段训练跑完再编码；两者也可以并发，但都要变慢。

manifest 与磁盘上的 npz 不一致时（例如编码被中途 `kill` 过），不需要 GPU 即可修复：

```bash
source env.sh && cd cosmos-framework
PYTHONPATH=. python -m cosmos_framework.zeva_training.vae_cache \
  --rebuild-manifest --output "$ZEVA_WORK/datasets/xhand_cte_cache" \
  --vae-path "$WAN_VAE_PATH" --dataset-root "$XHAND_DATA_ROOT"
```

### 第三阶段：CTE 特征缓存 + stage2 注入训练

把 CTE 的输出变成策略能吃的 `behavior_*` 张量，然后冻结策略、只训 Zeva 模块。
这三步是 release 里**完全缺失**的那条链路（它的 zeva 配置把 `dataloader_train` 设成 `None`）。

```bash
cd $ZEVA_WORK/cosmos-framework && source $ZEVA_WORK/env.sh

# 3a. 用训好的 CTE 对每个 boundary 前向，落盘 phase + 最近4个 effect_post
#     101 个 episode / 11,883 个 boundary，约 2.3 分钟，产物 5.5 MB
PYTHONPATH=. python -m cosmos_framework.zeva_training.cte_features \
  --cte-checkpoint "$ZEVA_WORK/runs/zeva_cte/cte-v2-20260921/cte_step_003000.pt" \
  --latent-cache   "$ZEVA_WORK/datasets/xhand_cte_cache" \
  --output         "$ZEVA_WORK/datasets/xhand_cte_features"

# 3b. 构建 task-context bank（单任务 = 1 条 entry），--check 会校验
#     train/serve 用的是同一个 256 维向量
PYTHONPATH=. python -m cosmos_framework.zeva_training.build_task_context_bank \
  --feature-cache "$ZEVA_WORK/datasets/xhand_cte_features" \
  --output        "$ZEVA_WORK/datasets/xhand_task_context_bank.pt" --check

# 3c. stage2 注入训练（冻结策略，只训 behavior_pbd / behavior_adapter /
#     behavior_global_projector）
export ZEVA_POLICY_CHECKPOINT="$ZEVA_WORK/runs/zeva/action_xhand/v1-20260921/checkpoints/iter_000004000"
export ZEVA_FEATURE_CACHE="$ZEVA_WORK/datasets/xhand_cte_features"
bash examples/launch_sft_action_policy_xhand_zeva.sh
```

数据契约（`omni_mot_model.py:714-829`，wrapper 见 `zeva_behavior_wrapper.py`）：

| batch key | shape | 来源 |
|---|---|---|
| `behavior_global` | `[B,256]` | task-context bank 检索出的 `behavior_value` |
| `behavior_phase` | `[B,128]` | CTE `phase[:, -1]` |
| `behavior_effect` | `[B,4,128]` | 最近 4 个已完成的 `effect_post`，**右对齐** |
| `behavior_effect_valid` | `[B,4]` bool | 哪几个槽位是真的 |

**真正被监督的只有 `behavior_prior_nll`**（权重 0.01）：把先验预测的
`[B,32,18]` 动作分布回归到 ground-truth 动作块。`behavior_global` 本身
**没有任何 loss 约束**——它是不透明条件向量，语义由 bank 里存什么决定。

> ⚠️ **`ZEVA_POLICY_CHECKPOINT` 要指向 `iter_XXXXXXXX` 目录本身**，不是
> `checkpoints/`。DCP loader 会自己往后拼 `model/`。
>
> ⚠️ 这个 launcher 用 `ZEVA_TAIL_OVERRIDES`（**字符串**）传覆盖参数，
> 不能用 `TAIL_OVERRIDES=(...)`——bash 数组不会导出到子进程，会被静默丢掉，
> 命令照样跑但用的是 TOML 里的值：
> ```bash
> ZEVA_TAIL_OVERRIDES="trainer.max_iter=5 checkpoint.save_iter=5" \
>   bash examples/launch_sft_action_policy_xhand_zeva.sh
> ```

### 两阶段通用

**脱离终端**：脚本用 `setsid` + `nohup` 启动，实测进程 `PPID=1`、独立 `SID`、
`TT=?`（无控制终端）。关窗口、断 SSH、退出 Claude Code 都不影响。

**日志**：`logs/<run名>.log`。同名续训会**追加**到同一份日志，每次启动写入一行
会话分隔头（含时间戳和参数），所以一个 run 的完整历史都在一个文件里。

### 续训原理（两阶段通用）

**同名即续训，换名即新 run。** 两个阶段各自实现：

- **policy**：框架内建自动续训（`cosmos_framework/utils/checkpointer.py:174-182`）——
  每次加载先找 run 目录下的 `checkpoints/latest_checkpoint.txt`，找到就自动恢复
  **模型 + 优化器 + 调度器 + 迭代数**；找不到才用 `BASE_CHECKPOINT_PATH` 做基座初始化。
  run 名通过 `job.name=` 覆盖传给训练（`train.py:282` 的 trailing overrides 在 TOML
  之后应用，所以优先）。已实测：检查点正确落到对应名字的目录，迭代计数接着走。
- **CTE**：`train_cte.py` 自己读 `<run目录>/cte_latest.pt`（含模型+优化器+step），
  脚本固定传 `--resume`，文件不存在就从头开始。

> 顺带一提：policy 训练即使 `save_iter` 设得很大，**正常结束时框架仍会存一次最终检查点**。

### 检查点间隔怎么定

| | policy | CTE |
|---|---|---|
| 参数 | `SAVE_ITER` 默认 **500** | `SAVE_EVERY` 默认 **100** |
| 单个检查点 | **137 GB**（`model/` 85G + `optim/` 53G）| **39 MB**（+110MB 的 latest 含优化器）|
| 存档耗时 | 约 2 分钟 | < 1 秒 |
| 单步耗时 | 约 4 秒（batch 16，8 卡）| 约 0.6 秒（batch 8）|

**policy 用 500**：每 ~33 分钟存一次，5000 步共 10 个 / 1.37 TB，崩溃最多丢 33 分钟。
不建议更密——每次存档会**暂停训练约 2 分钟**，500 以下收益递减。

**CTE 用 100**：检查点小，但**每次存档要顺带跑一遍完整验证集（约 55 秒）**，
所以长跑时反而要调大——3000 步用 `SAVE_EVERY=500` 比默认 100 省下约 20 分钟。

磁盘充裕（131 TB 可用）时按默认即可。**长跑时注意 policy 检查点会累积，框架没有自动清理。**

> ⚠️ **`stop` 不会存检查点**（2026-09-21 实测确认）。
> `termination_signal_checkpoint` 回调**只处理 SIGUSR1**，而且靠一个 Slurm 哨兵文件
> `$SLURM_LOG_DIR/SIGUSR1_RECEIVED` 触发；本机没有 Slurm，`SLURM_LOG_DIR` 为空，
> 回调在 `__init__` 里就把哨兵路径置空、直接 return。它注册的 SIGTERM 处理器
> **只是打一行日志**（`termination_signal_checkpoint.py:_log_sigterm`）。
>
> 所以 `stop` 等于直接杀进程，**从上一次定期存档之后的进度全丢**。
> 实例：`v1-20260921` 停在 4136 步，最后可用检查点是 `iter_000004000`，丢了 136 步。
>
> 要用这个机制，得让 `start` 导出 `SLURM_LOG_DIR`、`stop` 先 `touch` 哨兵再等待，
> 并把 `min_save_fraction`（默认 1/3，即 `SAVE_ITER/3` 步内不存）调 0。
> **尚未实现**——因为无法廉价验证（要跑满 8 卡并存一次 137 GB 检查点）。

### 磁盘现状

`runs/zeva/action_xhand/` 目前是**空的**——调试产生的检查点已清理。可以直接开始正式训练。

> 如果中途想丢弃某个 run 重训：`tools/run-xhand-train.sh fresh <名>` 会把它改名归档
> （不是删除），确认无用后再用 `rm <名>` 删掉。

### 跑通过程中修掉的几个阻塞

**1. dataloader 死锁（NFS 并发读）**

官方 DROID recipe 默认 `num_workers=16`，注释写着 "assumes data on local disk"。
我们的数据在共享 NFS 上，16 × 8 rank = **128 个并发读进程**会把个别读请求拖死，
表现为**某一个 rank 永远不报 dataloader ready**，其余 7 个 rank 在 pre-warm barrier
上无限等待（CPU 几乎为 0，纯粹在等）。

已在 experiment 配置里改成 `num_workers=4` / `prefetch_factor=1`。
GPU 利用率偏低时可往上调（`-- dataloader_train.dataloader.num_workers=8`）。

**2. zeva overlay 的版本不匹配（已打补丁）**

```
TypeError: compute_flow_matching_loss() got an unexpected keyword argument 'sample_weights'
```

`omni_mot_model.py` 被 overlay 加了 379 行、引入了 `sample_weights` 管道，
但 `flow_matching.py` **没被 overlay 改过**、签名里没有这个参数。

根因：zeva 的 `RELEASE_INFO.json` 写 `source_base_commit: db7d5194`，
而 `Zeva/README.md` 让你 checkout `ee58e414`——**两个 commit 的 `flow_matching.py`
不一样，且 `db7d5194` 在公开仓库里不存在**。也就是说这份 release 按它自己的
说明走是训不起来的（`compute_teacher_anchor_loss` 同样被调用但全仓库无定义，
只是它躲在默认关闭的开关后面）。

补丁：给 `compute_flow_matching_loss` 加了带默认值的 `sample_weights` 参数。
**对复现无影响**——`r3_event_weighted` 默认 False（`model_config.py:258`），
传进去的值恒为 `None`，新加的 `if` 分支被跳过，数值路径逐位相同。
已用"权重全 1 应等于不加权"的单测验证过这一点。

**3. CTE 数据集窗口越界（自己写的数据层，2026-09-21 修）**

```
RuntimeError: episode_000018.npz: window [111,128) does not fit in 127 latents
```

`cte_dataset` 把窗口起点上界写成了**动作**约束（`length//4 - T + 1`），
但真正的约束是**latent**约束（`t_lat - T`）。当 `length % 4 == 0` 时
`t_lat == length//4`，动作约束比 latent 约束**大 1**，于是最后一个窗口会
要求一段不存在的 latent。101 个 episode 里有 **26 个**踩中。

之所以训练时没报、只在验证时报：那 26 个在 train/val 里的分布是随机的，
第一次冒烟测试正好在 val 上撞到。现在两个约束都取 `min`，
并逐窗口校验过 train/val/full 三个 split（越界 0，所有 episode 的末窗都能取到）。

副作用：窗口数从 9772 变成 9748（train）、521 变成 519（val）——
删掉的正是那些本来就不该存在的越界窗口。

**4. multiprocessing 在 NFS 上留垃圾目录（已规避）**

训练正常跑完之后会多打一段 traceback：

```
OSError: [Errno 16] Device or resource busy: '.../tmp/pymp-p9t8oraf'
```

`env.sh` 把 `TMPDIR` 指向共享 NFS，而 multiprocessing 的 resource_tracker
在退出时删不掉自己放在 `$TMPDIR/pymp-*` 里的 socket，于是每次运行留一个目录
（已经攒了 536 个）。**不影响训练结果**，但看着像崩了。

`run-cte-train.sh` 里已经在 `source env.sh` 之后把 `TMPDIR` 改回本机 `/tmp`。
CTE 不需要大临时空间，policy 那条路径没动（它可能真的需要大 TMPDIR）。

**5. CTE 的任务聚类目标是死目标，拿掉后反而学得更快（2026-09-21 修）**

101 个 episode 的 `task_cluster` **全是同一个 `PressButton4Times`**，所以
`semantic_ids` 恒为 0。`_task_identity_clustering_loss`（`cte_losses.py:39`）是
多正样本对比损失：同类即正样本。只有一个类时**每个样本都是其它所有样本的正样本**，
损失下界变成 `log(N-1)`，唯一的下降方向是把所有 embedding 压成一个方向。

实测三条独立证据：

1. `loss_task` 从第 1 步 1.9463 到第 500 步 1.9435 —— **正好卡在 log(7)=1.9459 这个下界上**
2. `retrieval` 跨窗口平均余弦 = **0.999 ~ 1.000**（48 个互相隔开的 val 窗口，step 100/300/500 都是），即完全塌成一个常数
3. 该项权重 0.2 × 1.9435 = 0.389，占第 500 步 total（1.1945）的 **33%**

而且 `retrieval_head(z)` 和 phase/effect 头**共用同一个 trunk `z`**，所以这条退化梯度
一直在往共享主干上灌。

修法（`cte_losses.py:_task_identity_clustering_loss`）：batch 内语义 id 少于 2 个时直接返回 0，
不产生梯度。**多任务数据完全不受影响**——分支只在少于两个不同 id 时触发，已用单测覆盖
（单任务→0 且无梯度；两类→目标照常下降 2.77→1.10；混合 batch→正常）。

移除后的效果（同为第 500 步）：

| | v1（修前）| v2（修后）|
|---|---:|---:|
| `total` | 1.1945 | 0.6968 |
| `effect_contrastive` | 2.1618 | **1.7080** |
| `loss_task` | 1.9435（死）| 0.0000 |

`total` 掉 0.39 是把死目标拿掉的算术结果（**不代表收敛变好**，看曲线时别被误导）；
但 `effect_contrastive` 从 2.16 降到 1.71 是**真实改善**——有效目标的梯度变纯粹了。

### 已知无害噪音

训练结束时会刷一批 `OSError: [Errno 16] Device or resource busy:
tmp/pymp-*` —— 这是 multiprocessing 的 resource tracker 在 **NFS 上**清理
临时文件失败。发生在训练完成之后，不影响结果（退出码 0）。
要消除的话把 `TMPDIR` 指到本地盘（非 NFS）即可。

### 磁盘规划

**单个检查点 137 GB**（`model/` 85G + `optim/` 53G）。
`save_iter=500` 即每 500 步 137 GB；5000 步的 run 约 1.4 TB。
当前 131 TB 可用，够用，但长跑时注意别把检查点留太多。

## 7. 代码改动清单

新增/修改的文件：

| 文件 | 作用 |
| --- | --- |
| `data/.../datasets/xhand_lerobot_dataset.py` | UR7e+XHand 的 LeRobot 加载器（新增） |
| `data/.../datasets/action_sft_dataset.py` | 加 `get_action_xhand_sft_dataset` 工厂 |
| `configs/.../posttrain_config/action_policy_xhand_nano.py` | 新 experiment 配置 |
| `examples/toml/sft_config/action_policy_xhand_nano.toml` | 运行配置 |
| `examples/launch_sft_action_policy_xhand_nano.sh` | launcher |
| `tools/run-xhand-train.sh` | 第一阶段（policy）训练管理脚本（新增） |
| `zeva_training/probe_vae.py` | VAE latent 几何探针（新增） |
| `zeva_training/vae_cache.py` | VAE 批量编码 + latent 缓存（新增） |
| `zeva_training/cte_dataset.py` | CTE 训练窗口（新增） |
| `zeva_training/train_cte.py` | CTE 训练循环（新增） |
| `tools/run-cte-train.sh` | 第二阶段（CTE）训练管理脚本（新增） |
| `model/generator/algorithm/loss/flow_matching.py` | 加 `sample_weights` 参数（修 release 缺陷）|
| `model/generator/omni_mot_model.py` | overlay 自带，未再改 |
| `data/generator/action/domain_utils.py` | 注册 `ur7e-xhand`（domain 23）|

两个关键设计点：

1. **`keys_to_skip_loading = ["net_ema."]`** —— 不从裸 Cosmos3-Nano 起训，
   而是从后训练过的 policy 起，所以**保留它的 action heads**。
   直接抄 DROID recipe 的 skip 列表会静默丢弃这些权重。
2. **domain id 23 是全新槽位** —— `action2llm` / `llm2action` /
   `action_modality_embed` 里属于它的行没被任何已发布权重训过，必须保持可训练。

另有 `scripts/action_policy_server_xhand.py`（真机 server，见下一节）。

## 8. 真机部署（对齐 TactileTTT 客户端）

新增 `scripts/action_policy_server_xhand.py`，让 **TactileTTT 的机器人客户端零代码改动**即可对接。

**为什么能零改动**：两边传输层是同一个——都是 openpi 的 `WebsocketPolicyServer`（websocket + msgpack）。
而且客户端是**数据驱动**的：它从数据集 `meta/info.json` 读 `action.names` / `state.names` / 相机名，
按名字发观测（`observation/<camera>_image`）、按名字解析动作（`resp["actions"]`，形状 `[T, len(action.names)]`）。
所以 18 维关节位置和 `cam_front/left/right` 都是自动适配的。

服务端做了三件 RoboCasa server 没做的事：

1. **视图合成** —— 客户端分开发三路相机，训练时喂给策略的是 `left | wrist` 横向拼接
   （`wrist = cam_front`，因为源数据没有腕部相机）。服务端复刻
   `XHandLeRobotDataset._compose_video` 的几何。
2. **动作反归一化** —— 训练用了 `minmax` 归一化，而**真实关节范围不是 ±1**
   （如 `arm_joint_1 ∈ [-2.124, -1.477]`）。RoboCasa 的 arm7 通道恰好是 ±1，
   它的 server 才能跳过反归一化；照抄会让机器人收到 0.3 这种归一化值当弧度执行。
   已用真实统计数据验证：服务端与训练侧的 normalizer 逐位一致，往返误差 7e-08。
3. **服务端 CTE 历史重建** —— Zeva 需要每 4 个控制步一帧 + 对应的 4 步已执行动作。
   客户端一次 infer 执行一个 chunk，所以让它用 `--query-frequency 4` 启动
   （**只是命令行参数**），服务端即可自行重建历史。动作归属规则见 `_BoundaryBuffer`：
   请求 *t+1* 到达时，*t* 时刻返回的 chunk 就是帧 *t*→*t+1* 之间的转移。
   已用 8 个单测钉住（含丢包自愈、窗口滚动、与训练侧 proprio 索引一致性）。

**已知假设**：重建假设机器人原样按序执行了服务端返回的动作。若客户端有动作限幅/丢弃逻辑，
重建的转移会与物理实际不符——那时应改为由客户端发送历史（它已经为触觉状态做了同样的事：
`StateHistoryBuffer`）。

### 8.1 部署基础策略（纯 VLA，无 Zeva）—— 现在就能用

部署第一阶段训出的检查点。**这是纯模仿学习策略**：视觉 + 指令 → 动作，
不含任何 Zeva 记忆机制。

判断依据（三条都确认过）：`ZevaPolicyConfig.enabled` 默认 `False`
（`model_config.py:118-126`，文档明说 "disabled by default, preserving every existing
Cosmos recipe"）；我们的训练配置实际生效值也是 `False`；未传 `--cte-checkpoint` 时
服务端置 `_zeva_enabled = False`（`server:601`）。

**实证**：这个检查点的 `net.*` 里**没有** `behavior_pbd` / `behavior_adapter` /
`behavior_global_projector` —— 模块压根没被构造，所以权重里也不会有。

```bash
cd $ZEVA_WORK/cosmos-framework
source $ZEVA_WORK/env.sh

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
python -m cosmos_framework.scripts.action_policy_server_xhand \
  --checkpoint-path "$ZEVA_WORK/runs/zeva/action_xhand/v1-20260921/checkpoints/iter_000003000" \
  --allow-dcp-checkpoint \
  --experiment action_policy_xhand_nano \
  --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
  --action-stats-path "$XHAND_ACTION_STATS_PATH" \
  --domain-name ur7e-xhand \
  --resolution 256 \
  --action-dim 18 \
  --conditioning-fps 15 \
  --proprio-dim 22 \
  --image-height 256 --image-width 512 \
  --action-chunk-size 32 \
  --num-steps 30 --guidance 3.0 --shift 5.0 \
  --host 0.0.0.0 --port 8990
```

日志出现 `ready` 即加载完成。

**⚠️ 下面四个参数不覆盖就一定错**——服务端默认是 RoboCasa/DROID 的配置：

| 参数 | 服务端默认 | 必须设为 | 原因 |
| --- | --- | --- | --- |
| `--resolution` | `480` | **256** | 我们训练用的 tier 256 |
| `--action-dim` | `7` | **18** | UR7e + XHand 是 18 维关节位置 |
| `--proprio-dim` | `9` | **22** | 对应 `state_mode=arm22` |
| `--conditioning-fps` | `20` | **15** | 数据是 15 fps |

**⚠️ 显存**：训练在跑时每卡已占 ~42GB/80GB，服务端还要 ~30GB，可能 OOM。
建议等训练结束后再起服务端。

**⚠️ 先小幅试跑**：训练没有验证集，action loss 低到 0.0012 有过拟合嫌疑
（数据集仅 98 个训练 episode）。上真机时先小幅度验证，别直接全量操作。

后台运行（关窗不断）：

```bash
mkdir -p $ZEVA_WORK/logs
cd $ZEVA_WORK/cosmos-framework
source $ZEVA_WORK/env.sh

setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  $ZEVA_WORK/envs/zeva/bin/python \
  -m cosmos_framework.scripts.action_policy_server_xhand \
  --checkpoint-path "$ZEVA_WORK/runs/zeva/action_xhand/v1-20260921/checkpoints/iter_000003000" \
  --allow-dcp-checkpoint \
  --experiment action_policy_xhand_nano \
  --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
  --action-stats-path "$XHAND_ACTION_STATS_PATH" \
  --domain-name ur7e-xhand \
  --resolution 256 --action-dim 18 --conditioning-fps 15 --proprio-dim 22 \
  --image-height 256 --image-width 512 --action-chunk-size 32 \
  --num-steps 30 --guidance 3.0 --shift 5.0 \
  --host 0.0.0.0 --port 8990 \
  > $ZEVA_WORK/logs/xhand-server.log 2>&1 < /dev/null &
```

机器人侧**不改代码**，只加参数（`--query-frequency 4` 是 Zeva 阶段服务端重建
CTE 历史所必需的；纯策略部署时保留也无害）：

```bash
python TactileTTT_client_multi.py --server-port 8990 --query-frequency 4 ...
```

### 8.2 部署完整 Zeva

需要三个产物：**stage2 检查点 + CTE 检查点 + task-context bank**。

```bash
cd $ZEVA_WORK/cosmos-framework && source $ZEVA_WORK/env.sh
PYTHONPATH=. $ZEVA_WORK/envs/zeva/bin/python -m cosmos_framework.scripts.action_policy_server_xhand \
  --checkpoint-path "<stage2 的 DCP，即 runs/zeva/zeva_xhand/…/checkpoints/iter_XXXXXXXX>" --allow-dcp-checkpoint \
  --experiment action_policy_xhand_zeva \
  --experiment-overrides model.config.tokenizer.vae_path="$WAN_VAE_PATH" \
  --action-stats-path "$XHAND_ACTION_STATS_PATH" \
  --domain-name ur7e-xhand \
  --resolution 256 --action-dim 18 --conditioning-fps 15 --proprio-dim 22 \
  --cte-checkpoint "$ZEVA_WORK/runs/zeva_cte/cte-v2-20260921/cte_step_003000.pt" \
  --task-context-bank "$ZEVA_WORK/datasets/xhand_task_context_bank.pt" \
  --task-context-instruction PressButton4Times \
  --host 0.0.0.0 --port 8990
```

> **`--task-context-bank` 和 `--task-context-instruction` / `--static-task-context-checkpoint`
> 必须二选一**（`server:607-609` 强制）。单任务数据走 `--task-context-instruction` 这条路：
> bank 里只有 1 条 entry，直接按键取回该任务的 `behavior_value`。
> 多任务时需要训 stage3 head 并改用 `--static-task-context-checkpoint`。

> ⚠️ **`_CTE_MAX_BOUNDARY_FRAMES` 必须等于 CTE 训练的 `window_latents`（17）。**
> CTE 的时间混合器是 GRU、没有位置编码，`phase[:, -1]` 依赖它前面有多少帧。
> 服务端留 64 帧的话，模型会看到训练时没见过的状态，注入的 phase/effect
> 会**静默偏离** stage2 拟合时用的缓存特征。已改成 17 并加了注释说明。
> 参考实现（RoboCasa server）没有这个问题也不受这个约束，因为它是每次请求
> 现算特征，而不是查缓存。

## 9. Zeva 三阶段训练的现状

**这份 release 是推理 release，不含 Zeva 自身的训练代码。** 已查证：

- `causal_transition_encoder_loss` 全仓库唯一调用点是单测
- `bidirectional_supervised_contrastive_loss` 零调用点
- 模型训练需要的 `behavior_global/phase/effect/effect_valid` 四个张量，
  在全仓库 `data/**` 里零产出——消费它们的 wrapper 不在快照内
- 四个 zeva experiment 配置的 `dataloader_train` 全是 `None`

要在自有数据上复现 Zeva，需要自己补：CTE 训练循环、CTE feature cache 构建、
task-context bank 构建、stage3 retrieval head 训练、PIM training bank 构建，
以及上面那个 `behavior_*` 数据集包装层。完整方案见
`~/.claude-cli/.claude/plans/vectorized-floating-zephyr.md`。

## 10. 下一步

按计划文件（`~/.claude-cli/.claude/plans/vectorized-floating-zephyr.md`）的分阶段顺序：

- 🟡 **Phase 1 — base policy 微调**：跑通并训到 **iter 4136 / 5000**，
  最新可用检查点 `iter_000004000`（`runs/zeva/action_xhand/v1-20260921/`）。
  **已按你的要求停止**。续训：`tools/run-xhand-train.sh start v1-20260921`
  （⚠️ 先看第 6 节"检查点间隔怎么定"——`stop` 不存检查点）。
- ✅ **Phase 2 — CTE 训练**：代码完成并实测，VAE 缓存 101/101（11,883 latent，
  608 MB），两次训练各 500 / 3000 步，检查点全部通过推理端加载验收。
  **当前结论：目标函数还没配对**——判别性表示学在了 `effect_outcome_post` 上，
  而 server 注入的是 `effect_post`，训越久后者越塌（见第 6 节对比表）。
- ⬜ **Phase 3~5 — 中间产物 + 数据契约层 + stage2 注入训练**。

  **`behavior_value` 的疑问已经缩小**（查证 `action_policy_server_robocasa365_zeva.py:706-719`）：
  它不是 CTE 的输出，而是**从 task-context bank 里检索出来的**。完整链路：

  ```
  初始帧 → VAE 编码 → 冻结 policy 的 VLM readout (extract_batch)
          → stage3 head → 在 bank 中检索 → 取回 256 维 value = behavior_global
  ```

  而模型侧 `_attach_stage2_behavior`（`omni_mot_model.py:714-829`）**对 `behavior_global`
  没有任何监督**——它只是个不透明条件向量，喂给 `behavior_global_projector`
  和 `PolicyInjectionPrior.global_to_anchors`。同一条路径里真正被监督的是
  `behavior_prior_nll`：把先验预测的 `[B,32,A]` 动作分布回归到 ground-truth 动作块。

  **所以 `behavior_value` 的语义是自由的**，只需满足一条：**训练时喂什么，bank 里就得存什么**。
  这从"必须看论文"降级成"设计选择 + 一致性要求"。

  仍未定的：CTE 只提供 `phase`（最后一帧）和 `effect`（最近 4 个 `effect_post`，右对齐）；
  四个 `behavior_*` 张量在全仓库 `data/**` 里**零产出**，消费它们的 wrapper 必须自己写
  （`xhand_lerobot_dataset.py:315-319` 已经输出了 `behavior_episode_id` /
  `behavior_task_cluster` / `behavior_frame_offset` 三个索引字段，无人消费）。
- ⬜ **Phase 6~7 — 真机部署与消融**：server 已写好（第 8 节），
  还差真机客户端联调，然后做 PIM 开/关消融。

**建议的下一步顺序**：先修 CTE 的 effect 目标（Phase 2 收尾，见第 6 节），
再打通 Phase 3~5 的部署链路。CTE 再训一轮只要约 15 分钟，不急。
