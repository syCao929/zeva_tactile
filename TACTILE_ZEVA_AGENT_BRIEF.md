# Tactile-ZeVa 分支交接说明

本文档给接下来负责实现触觉分支的 agent 使用。它汇总了当前仓库、Zeva 功能审计、XHand 数据、触觉 encoder、研究动机和建议的代码改造顺序。

当前任务先整理设计和实现边界，**不要在没有核验的情况下认定 tactile encoder checkpoint 已经确认**。encoder 的真实接口、框架、输入输出和远程权重需要由新 agent 首先核验。

## 1. 项目和目标

仓库：

```text
/Users/babyna/zeva_tactile
https://github.com/syCao929/zeva_tactile.git
```

仓库是在 Cosmos Framework 上复现和扩展 Zeva（In-Context Causal Learning）的 XHand 项目。当前代码已经包含视觉 CTE、stage2 行为注入、XHand 数据适配器和真机/服务端入口，但还没有把触觉接入成完整训练和部署链路。

Zeva 论文：

```text
https://fe-1301391939.cos.ap-shanghai.myqcloud.com/zeva/zeva.pdf
```

用户要加入 tactile 分支，核心研究主张是：

> 触觉记忆和视觉记忆形成互补的双分支，共同协作，使联合模型产生 1+1>2 的效果。

视觉可以观察物体、场景、姿态和可见阶段变化；触觉可以记录接触是否发生、压力、滑移、卡阻和视觉无法区分的物理结果。两种记忆应共享时间边界、动作转移和任务阶段，但保留模态专属的 effect 表征。

## 2. 当前数据和运行环境

当前实验是单任务：

```text
TactileTTT/data/press_button_4_times
```

数据事实：

- UR7e + XHand。
- 动作是 18 维关节位置：6 个机械臂关节 + 12 个手关节。
- 15 Hz，101 episodes，LeRobot v2.1。
- 原始 `observation.state` 是 1972 维。
- `[0:52]` 是机械臂、末端位姿和手部关节/力矩信息。
- `[52:1972]` 是 XHand 五指触觉相关数据，共 1920 维；当前加载器没有使用这部分。
- 当前使用 `cam_left` 和 `cam_front`；代码约定把 `cam_front` 作为 wrist 视角，虽然它实际上不是腕部相机。
- `press_button_4_times` 表示一个任务中按压四次的阶段，不等于四次 Zeva attempt。一次 attempt 是一整次执行，失败后重新开始才算新的 attempt。

主要文件：

```text
README_ZEVA_ENV.md
tools/convert_xhand_dataset.py
cosmos-framework/cosmos_framework/data/generator/action/datasets/xhand_lerobot_dataset.py
```

数据转换脚本默认只建立软链接，不复制约 36 GB 的原始数据。

## 3. 当前 Zeva 实现审计

### 3.1 已经实现的视觉链路

现有视觉链路为：

```text
视频
→ Wan VAE latent
→ CausalTransitionEncoder（CTE）
→ visual phase/effect
→ stage2 policy injection
```

相关文件：

```text
cosmos-framework/cosmos_framework/zeva_training/vae_cache.py
cosmos-framework/cosmos_framework/zeva_training/train_cte.py
cosmos-framework/cosmos_framework/zeva_training/cte_features.py
cosmos-framework/cosmos_framework/data/generator/action/datasets/zeva_behavior_wrapper.py
cosmos-framework/cosmos_framework/model/zeva/causal_transition_encoder.py
cosmos-framework/cosmos_framework/model/zeva/policy_injection.py
```

XHand stage2 当前有这些行为字段：

```text
behavior_global
behavior_phase
behavior_effect
behavior_effect_valid
```

配置文件：

```text
cosmos-framework/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_xhand_zeva.py
```

该配置只训练 `behavior_pbd`、`behavior_adapter` 和 `behavior_global_projector`，Cosmos 基础策略冻结。

### 3.2 PIM 是否已经完成

结论：

> PIM 的底层模块存在，RoboCasa 的 PIM server/config 存在，但 XHand 的训练和部署链路没有接入 PIM。

已有模块：

```text
cosmos-framework/cosmos_framework/model/zeva/persistent_interaction_memory.py
cosmos-framework/cosmos_framework/model/zeva/phase_conditioned_retrieval.py
cosmos-framework/cosmos_framework/model/zeva/causal_prompt.py
cosmos-framework/cosmos_framework/scripts/action_policy_server_robocasa365_zeva_pim.py
cosmos-framework/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robocasa365_atomic5_zeva_pim.py
```

PIM 底层协议包括：

- `reset_episode()`：清空一个 episode 的 memory。
- `begin_attempt()`：保留 PIM，开始下一次 attempt。
- `append_completed()`：写入并合并 phase/effect。
- `query_phase()` / `query_tensors()`：按 phase 检索经验。

但 `action_policy_server_xhand.py` 当前只维护视觉 CTE buffer；它没有跨 attempt 的 XHand PIM。代码中的 `reset_tactile_memory` 名字容易误导，当前实际清空的是视觉 CTE buffer，触觉 memory 尚未实现。

RoboCasa PIM 配置本身也有训练链路问题：`action_policy_robocasa365_zeva.py` 把 `dataloader_train` 设为 `None`，其 PIM 配置虽然声明了 `behavior_pim_training_bank`，但没有形成可复用的 XHand PIM training path。

已有 PIM 单测通过：

```text
persistent_interaction_memory_test.py
attempt_protocol_test.py
policy_injection_test.py

12 passed
```

这些测试只证明 PIM 基础模块可用，不证明 XHand PIM end-to-end 已完成。

### 3.3 当前触觉实际上没有进入模型

`xhand_lerobot_dataset.py` 的 state mode 最宽只到 `full52`，不会读取 `[52:1972]`。`__getitem__()` 只返回 video、action、proprio 和任务元数据。

`vae_cache.py` 只缓存 visual latent、raw action 和 episode metadata，也没有 tactile。

因此当前模型实际是：

```text
video → action
```

而不是：

```text
video + tactile → action
```

## 4. 触觉 encoder 信息

### 4.1 用户指定的真实 encoder

本地参考代码：

```text
/Users/babyna/reTouch/tactilepatchencoder
```

远程服务器上可能存在的 checkpoint：

```text
/workspace/mnt/sqzhang26/FactileLDM/checkpoints/xhand_patch_tactile_encoder_pretrain/patch_informed_full_heads_taskall2_encoder_final_20k_0722/19999
```

这条路径目前只作为用户提供的候选路径记录。新 agent 必须确认：

1. `reTouch` 中 encoder 的真实 class 名、import 路径和加载方式。
2. checkpoint 是否确实存在于上述远程路径。
3. checkpoint 使用 PyTorch、JAX/Flax 还是其他框架。
4. raw tactile 的输入 shape、预处理、归一化和时间窗口。
5. encoder 输出 shape、dtype、维度以及每个 token 的语义。
6. 在线推理是否只依赖当前及历史 tactile，是否会读取未来帧。
7. 是否可以稳定导出到 Cosmos 当前使用的 PyTorch runtime。

在这些事实确认前，不要把 checkpoint 路径、输出维度或框架写死进训练配置。

### 4.2 旧的本地参考接口

另一个本地项目 `/Users/babyna/TactileTTT` 中有 patch tactile encoder 参考实现：

```text
/Users/babyna/TactileTTT/src/openpi/models/patch_tactile_pretrain.py
/Users/babyna/TactileTTT/src/openpi/models/tactile_tokenizer.py
/Users/babyna/TactileTTT/src/openpi/models/pi0_config.py
/Users/babyna/TactileTTT/src/openpi/policies/xhand_policy.py
```

参考接口的 raw force 输入大致是：

```text
[B, T, 5, 120, 3]
```

含义是 batch、时间、五根手指、每根手指 120 个点、每个点 3 维力向量。参考配置使用 10 帧 tactile history、15 Hz、5 根手指、120 点、3 维力，patch tokenizer 每根手指输出一个 token，默认 embedding width 为 1024。

XHand raw force 在完整 state 中的参考常量：

```python
TACTILE_SENSOR_COUNT = 5
TACTILE_BLOCK_SIZE = 384
TACTILE_BLOCK_START = 52
TACTILE_RAW_FORCE_OFFSET = 24
TACTILE_RAW_FORCE_POINTS = 120
```

这些只是 TactileTTT 的参考，不能替代对 `reTouch/tactilepatchencoder` 的实际核验。

## 5. 第一阶段范围：单任务先不接 PIM

当前目标是单任务训练，先做 attempt 内的双模态经验：

```text
当前执行过程中的视觉 BIT + 触觉 BIT
```

第一阶段建议：

- 不接跨 attempt PIM。
- 保留现有视觉 CTE/BIT。
- 新增 tactile encoder、tactile temporal transition encoder 和双模态 fusion。
- 冻结预训练 tactile encoder。
- 冻结 Cosmos 基础策略。
- 只训练触觉时序适配器、视觉/触觉融合层、policy injection adapter 和 action prior head。

只有在需要“失败 → 恢复初始状态 → 再次尝试 → 利用上一次经验改进”时，才进入第二阶段 PIM。届时需要明确 episode、attempt、环境 seed、失败结果和 memory reset 协议。

## 6. 推荐的双模态方法

### 6.1 两条分支

视觉分支继续使用现有 CTE：

```text
image/VAE latent + executed action
→ visual phase
→ visual effect
```

触觉分支建议为：

```text
raw tactile
→ pretrained tactile patch encoder
→ tactile finger/patch tokens
→ tactile temporal/action-conditioned encoder
→ tactile phase/effect/confidence
```

不要强制视觉 effect 和触觉 effect 使用相同语义空间。建议先使用两个模态专属表示，例如：

```text
visual_effect  ∈ R^128
tactile_effect ∈ R^128
```

两者共享：

```text
episode id
frame/boundary id
action transition
task phase
```

### 6.2 经验单位

双模态经验可以抽象为：

```text
(
    task/phase,
    executed_action,
    visual_effect,
    tactile_effect,
    validity/confidence,
    outcome
)
```

触觉 confidence/valid mask 很重要，因为没有接触时触觉信号可能没有有效信息；融合层应学会在无接触或信号异常时降低 tactile 分支权重。

### 6.3 策略融合

建议不要大改 Cosmos 基础策略，而是在现有 policy injection 上增加：

```text
visual effect attention
tactile effect attention
cross-modal attention 或 gated fusion
接触 confidence / valid mask
→ fused action prior
→ frozen Cosmos policy
```

第一版可以用 learned gate：

```text
gate = f(visual_phase, tactile_phase, tactile_confidence, valid_mask)
fused = visual + gate * tactile
```

然后输出现有 injection 所需的 `prior_mean`、`prior_std` 或等价 action prior 字段。

视觉 CTE 当前约每 4 个 raw control frame 对齐一个 latent boundary。触觉第一版可沿用同一 boundary；patch encoder 内部保留其自己的 tactile history（例如 10 帧），但不得让当前 boundary 使用未来 tactile。

## 7. 建议代码改造顺序

### 步骤 1：扩展 XHand 数据加载器

修改：

```text
cosmos-framework/cosmos_framework/data/generator/action/datasets/xhand_lerobot_dataset.py
```

增加类似参数：

```text
emit_tactile
tactile_mode = "raw_force"
tactile_history_length
```

训练样本最好按帧输出：

```text
[chunk_length + 1, 5, 120, 3]
```

并带上：

```text
tactile_valid
episode_id
frame_offset
```

不要只输出一个当前 tactile，因为 CTE/BIT 需要历史窗口。

### 步骤 2：实现 tactile cache

新增：

```text
cosmos-framework/cosmos_framework/zeva_training/tactile_cache.py
```

职责：读取 raw tactile，调用 `reTouch` tactilepatchencoder，保存每个 episode 的 tactile tokens。缓存至少应包含：

```text
episode_id
frame_index
tokens
valid
sensor/contact confidence
encoder config hash
```

如果 encoder 是 JAX/Flax 而 Cosmos 是 PyTorch，需要三选一：

1. 转成 PyTorch adapter；
2. 导出 ONNX/其他可验证格式；
3. 训练阶段离线预计算，部署阶段另写运行时。

优先考虑 PyTorch adapter 或严格验证过的导出版本。导出后必须用随机输入比较原实现和导出实现的输出误差。

### 步骤 3：实现触觉时序 encoder

新增：

```text
cosmos-framework/cosmos_framework/model/zeva/tactile_transition_encoder.py
```

输入：

```text
tactile_tokens
executed_actions
valid_mask
```

输出：

```text
tactile_phase
tactile_effect
tactile_effect_valid
tactile_confidence
```

必须保证严格因果：当前 boundary 的 feature 不能使用未来 tactile。

### 步骤 4：实现 tactile feature wrapper

新增：

```text
cosmos-framework/cosmos_framework/zeva_training/tactile_features.py
```

参考：

```text
cosmos-framework/cosmos_framework/zeva_training/cte_features.py
```

统一输出 tactile phase/effect/valid/confidence，并处理 boundary 和 episode 对齐。

### 步骤 5：扩展 Zeva behavior wrapper

修改：

```text
cosmos-framework/cosmos_framework/data/generator/action/datasets/zeva_behavior_wrapper.py
```

新增字段：

```text
behavior_tactile_phase
behavior_tactile_effect
behavior_tactile_effect_valid
behavior_tactile_confidence
```

保留已有视觉字段，不要让触觉字段覆盖视觉字段。

### 步骤 6：连接 stage2 batch

修改：

```text
cosmos-framework/cosmos_framework/model/generator/omni_mot_model.py
```

重点函数：

```text
_attach_stage2_behavior()
```

需要处理 tactile 字段、batch 维度、`action_sample_indices`、valid mask、confidence 以及 packed sequence 的传递。

### 步骤 7：实现双模态 policy injection

修改：

```text
cosmos-framework/cosmos_framework/model/zeva/policy_injection.py
```

当前模块只处理 global/phase/effect/effect_valid。建议增加视觉注意力、触觉注意力、跨模态注意力或 gate，并根据 tactile valid/confidence 动态融合。输出仍保持现有 action prior 接口，从而减少对 Cosmos 基础策略的修改。

### 步骤 8：增加 XHand 双模态训练配置

新增：

```text
cosmos-framework/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_xhand_zeva_tactile.py
```

继承：

```text
action_policy_xhand_zeva
```

开启 tactile behavior stage 和 dual-modal policy injection，只训练新增模块和 adapter。

### 步骤 9：扩展在线 XHand server

修改：

```text
cosmos-framework/cosmos_framework/scripts/action_policy_server_xhand.py
```

需要增加：

```text
从 state 提取 raw tactile
维护 tactile history
调用 tactile encoder
生成 tactile BIT feature
与视觉 feature 融合
将双模态 feature 注入策略
```

reset 语义必须区分：

```text
attempt reset：清空当前 attempt 的 BIT
episode reset：未来接入 PIM 时清空视觉/触觉 PIM
normal next request：保留 BIT
```

现有 `reset_tactile_memory` 命名应重新检查，因为它目前并没有清空真正的触觉 memory。

## 8. 新 agent 首先要做的核验

按优先级执行：

1. 阅读 `/Users/babyna/reTouch/tactilepatchencoder`，确认真实 class/import/checkpoint API。
2. 在远程服务器确认候选 checkpoint 的存在性、文件结构和框架。
3. 用一个真实样本确认 raw tactile 输入 shape、预处理和输出 shape。
4. 设计最小 `TactileEncoderAdapter`，支持离线缓存和在线单步/历史窗口推理。
5. 用随机输入或固定样本比较原始 encoder 与 adapter/export 的输出误差。
6. 再扩展 XHand loader 输出 tactile。
7. 最后连接 tactile feature、fusion、policy injection 和 server。

以下默认值可以先用于代码骨架：

```text
单任务
不接 PIM
15 Hz
每 4 个 raw frame 对齐一个 interaction boundary
冻结 pretrained tactile encoder
冻结 Cosmos base policy
只训练 tactile temporal adapter + fusion + policy adapter
```

不阻塞代码骨架但需要远程服务器补充确认的事项：

- 数据是否包含多次失败/重试；
- 是否有 success/failure/outcome 标签；
- 触觉、图像和动作时间戳是否严格对齐；
- 远程是否已有 XHand stage1/stage2 权重；
- 服务器是否安装 JAX/Flax；
- 最终部署是否要求纯 PyTorch runtime。

## 9. 必须避免的错误

1. 不要把 `press_button_4_times` 写成四次 attempt；它通常是一次任务里的四个动作阶段。
2. 不要第一版就加入 PIM。单任务和 attempt 内 memory 足以验证双模态是否有效。
3. 不要让触觉分支读未来帧。预训练任务可以有辅助标签，在线记忆必须 causal。
4. 不要把视觉和触觉 effect 强行压成一个 latent；先保留模态专属 effect，通过 fusion 协作。
5. 不要只在 config 里加字段，必须贯通 dataset → cache → feature wrapper → batch attach → packed sequence → injection → server。
6. 不要把 `xhand_task_context_bank.pt` 当成 PIM；它是固定的单任务 task context，不是跨 attempt 经验库。
7. 不要把候选 checkpoint 路径或 TactileTTT 的参考维度当成 `reTouch` 的已确认接口。

## 10. 验证计划

第一阶段至少比较：

```text
no-memory baseline
vision-only BIT
tactile-only BIT
vision+tactile BIT
```

建议额外比较：

```text
当前 tactile 输入但不使用 tactile history
简单 concat fusion
gated/cross-attention fusion
```

固定数据划分、基础策略、训练步数、随机种子、action horizon 和 inference steps。报告：

```text
success rate
按压完成率
接触成功率
失败类型
平均执行步数/尝试步数
```

可以用以下交互增益衡量双模态是否真的有协同：

```text
Delta_interaction
= S(vision+tactile)
 - S(vision-only)
 - S(tactile-only)
 + S(no-memory)
```

只有联合模型稳定优于两种单路模型，并且交互增益为正，才能支持“视觉记忆 + 触觉记忆产生超加性协同”的主张。

## 11. 重要文件索引

```text
README_ZEVA_ENV.md
NOTES_archive_20260921.md
env.sh
tools/convert_xhand_dataset.py
tools/run-xhand-train.sh
tools/run-cte-train.sh
tools/run-xhand-zeva-train.sh

cosmos-framework/cosmos_framework/data/generator/action/datasets/xhand_lerobot_dataset.py
cosmos-framework/cosmos_framework/data/generator/action/datasets/zeva_behavior_wrapper.py
cosmos-framework/cosmos_framework/zeva_training/vae_cache.py
cosmos-framework/cosmos_framework/zeva_training/cte_features.py
cosmos-framework/cosmos_framework/zeva_training/train_cte.py
cosmos-framework/cosmos_framework/model/zeva/causal_transition_encoder.py
cosmos-framework/cosmos_framework/model/zeva/policy_injection.py
cosmos-framework/cosmos_framework/model/zeva/persistent_interaction_memory.py
cosmos-framework/cosmos_framework/model/generator/omni_mot_model.py
cosmos-framework/cosmos_framework/scripts/action_policy_server_xhand.py
cosmos-framework/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_xhand_zeva.py
```

完成上述核验后，再开始写正式实现；第一份可 review 的代码应是 encoder adapter 和最小单元测试，而不是直接修改整个 server。
