# Zeva / XHand 训练与代码审计 — 2026-09-25

> 退役更新：用户随后决定停止使用并删除 V2。V2 基座、其视觉 Stage-2、评测副本和运行日志已清理，旧服务已停止；删除清单见 `audit-artifacts/20260925/v2_cleanup.json`。以下是删除前的历史审计快照，V2 路径不再存在，保留 V2 测试的建议也已撤销。后续统一使用 V3 joint18。

审计快照：2026-09-25 14:25（Asia/Shanghai）。仅查询进程、日志、配置和权重，并运行 CPU 检查；没有修改训练/推理代码、停止作业、删除或替换模型。当前源码有多人未提交修改，运行中进程可能持有修改前的源码；下文明确区分。

**当前结论：正在训练的是 v3 joint18 Stage-1 基座，7 张 A800；另外 1 张卡运行 v2 基座推理服务。没有正在训练的视觉 Stage-2 或触觉作业。现存策略 checkpoint 全部没有触觉参数，无法认定触觉正式对照已完成。**

## 实际模型清单

下列路径以 `/workspace/mnt/sqzhang26/zeva-work` 为根。

| 模型/目录 | 实际输入及来源 | 已确认进度与权重 | 当前判定 |
| --- | --- | --- | --- |
| `runs/zeva/action_xhand/v2-proprio-20260924` | `models/cosmos3-nano` 起训；22维 state：6臂关节＋16维末端矩阵；18维关节 action；无 CTE/触觉 | 日志到4000，现存 `checkpoints/iter_000004000`；配置目标5000，当前未训练 | 缺少12个手关节，但不因此自动判废；属于可保留测试的旧输入版本，须用22维匹配的推理接口 |
| `runs/zeva/zeva_xhand/action_policy_xhand_zeva`（目前目录内容） | 从 v2-4000 起训，22维 state，CTE v4 特征，触觉关闭 | 日志实际到800；只保存 `checkpoints/iter_000000500`；当前未训练 | 师弟说的第二阶段500步模型；是视觉 Zeva，不能当作触觉版本 |
| `runs/zeva/action_xhand/v3-joint18-20260925` | 从 `models/cosmos3-nano` 重新起训；18维 state＝6臂＋12手关节，顺序对应18维 action；无 CTE/触觉 | 14:24:51最新进度1870/5000；已保存500/1000/1500 | 当前唯一策略训练作业，7卡；不是从v2继续训练 |
| `runs/zeva_cte/cte-v4-20260924` | 基于 `datasets/xhand_cte_cache_v2` 训练CTE | 完成3000步；生成 `datasets/xhand_cte_features_v4` | 当前视觉 Stage-2 实际使用的独立编码器 |
| 历史 v1 基座 | 9/21 run | 历史日志证明保存4000，随后到4136；当前目录不在 | 不属于当前可用权重；历史归一化/输入问题需要按当时源码追溯 |
| 历史 v1 的视觉 Stage-2 | 曾使用相同 `action_policy_xhand_zeva` 输出路径 | 历史日志证明保存2000；当前该权重不在 | 该路径已被新v2 Stage-2复用，不能把旧2000与现500混算 |
| 触觉 smoke | 9/24调试作业 | 到iteration4（第5次更新）后NCCL保存失败 | 仅证明跑过调试，不能当作可部署模型 |
| 历史 `compare_formal_20260924*` 触觉/基线对照 | 此前会话中的启动记录 | 本次在 `runs/zeva` 中找不到对应目录或权重 | 不能证明最终完成；此前“已启动”不等于已训成 |

证据：

- v2配置：`runs/zeva/action_xhand/v2-proprio-20260924/config.yaml:29`（起点）、`:77`（arm22）、`:291`（22维）、`:618`（5000目标）。保存4000见 `logs/v2-proprio-20260924.log:8873`。
- v3配置：`runs/zeva/action_xhand/v3-joint18-20260925/config.yaml:29`（起点）、`:77`（joint18）、`:227`（tactile=false）、`:291`（18维）。7卡启动见 `logs/v3-joint18-launch.log:12`；1870步见 `logs/v3-joint18-20260925.log:3333`；1500保存见 `:2969`。
- 现 Stage-2 配置：`runs/zeva/zeva_xhand/action_policy_xhand_zeva/config.yaml:31`（v2-4000）、`:79`（arm22）、`:90`（CTE v4）、`:229`（tactile=false）。500保存见 `logs/zeva-stage2.log:12043`；800进度见 `:12361`；历史2000保存见 `:9315`。
- CTE v4完成见 `logs/cte-v4-20260924.log:636`；特征来源见 `datasets/xhand_cte_features_v4/manifest.json:3`。
- v1保存及最后进度见 `logs/v1-20260921.log:9437`、`:10807`。
- 触觉smoke进度与保存错误见 `cosmos-framework/outputs/train/logs/action_policy_xhand_zeva_tactile_sft.log:1330`、`:1436`。

实时进程：torchrun PID 131922（7 rank，目标5000）；v2 server PID 126596，端口8990。NVIDIA查询确认该server约31GB，其余7个训练进程各约45–47GB。服务加载成功见 `logs/serve-stage1.log:18`，22维初始化见 `:24`，监听见 `:25`。没有该服务已成功执行机器人请求的证据。5090远端环境、客户端和磁盘内容未核实。

## 已确认的问题与影响范围

**P1：重复验证返回0个batch，当前baseline的验证分数不可用于选模型。**

`cosmos-framework/cosmos_framework/data/generator/joint_dataloader.py:381` 仅在构造时创建底层iterator，`:661`一直读它；`PackingDataLoader.__iter__` 在`:1093`不重建iterator，`:1115`遇耗尽退出，`:1147`空batch直接返回。`cosmos-framework/cosmos_framework/trainer/__init__.py:481`只是再次遍历同一loader。把workers改为0不能解决此问题。

v2、v3、Stage-2日志均存在后续空验证：`logs/v2-proprio-20260924.log:8890`、`logs/v3-joint18-20260925.log:2973`、`logs/zeva-stage2.log:12047`。这些位置的0/NaN不证明训练发散，也不证明loss已降到0。v3训练loss仍有限，但成功率尚无证据。`tools/eval-checkpoint.sh:86`独立进程评估可避开iterator复用，却固定使用当前18维基座recipe，不能直接用于22维v2或视觉Stage-2。

**P1：当前磁盘代码重新启动v2及其Stage-2时有22/18维兼容错误。**

`cosmos-framework/cosmos_framework/scripts/action_policy_server_xhand.py:133`、`:319`硬编码joint18切片；当前nano recipe也构建18维proprio projector。v2及其Stage-2的checkpoint实际为 `[4096,22]`。只设置 `--proprio-dim 22` 不能同时恢复22维构图和arm22切片，必须让模型配置、state选择、权重一致。

例外是本机目前活着的v2服务：11:03启动，11:04:58已成功加载22维模型；server文件和nano配方在11:05之后才改为18维。因此不能把“当前代码重启不兼容”误写成“现有服务已经加载失败”。

**P1：触觉behavior head遗漏LayerNorm初始化。**

`cosmos-framework/cosmos_framework/model/generator/omni_mot_model.py:315`先 `to_empty`，再执行网络初始化；`cosmos-framework/cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:424`只初始化三个head的Linear，遗漏 `cosmos-framework/cosmos_framework/model/zeva/tactile_memory.py:158`、`:162`、`:166`的LayerNorm。新触觉训练会使用未初始化的LN weight/bias。

CPU复现使用真实模块和从当前源码提取的初始化方法：meta构造、to_empty、NaN哨兵后执行初始化，三组LN共六个tensor仍为NaN，head输出NaN；BIT参数已正确初始化。这证明源码缺陷，不证明某个已丢失的历史checkpoint必定含NaN。修复并验证前不应重新启动正式触觉对照。

**P1：触觉validation和在线server缺少输入。**

`cosmos-framework/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_xhand_zeva_tactile.py:25`仅为train开启触觉，val继承baseline设置。模型在 `cosmos-framework/cosmos_framework/model/generator/omni_mot_model.py:775`要求 `tactile_state/tactile_valid`，当前配方开启validation会报错。

server在 `cosmos-framework/cosmos_framework/scripts/action_policy_server_xhand.py:341`构建样本，`:425`补CTE字段，但没有触觉窗口及mask，无法直接服务触觉模型。CTE reset字段里出现tactile字样不代表BIT接入完成。

**P2：episode起始约1.07秒触觉被CTE有效性mask屏蔽。**

`cosmos-framework/cosmos_framework/model/generator/omni_mot_model.py:823`把触觉加到最后一个CTE effect slot，但保留原 `effect_valid`。`cosmos-framework/cosmos_framework/model/zeva/policy_injection.py:107`会把invalid slot整体替成BOS。实测当前CTE v4缓存101集、11883边界中404个边界的最后slot无效：每集frame0、4、8、12；映射后raw frame0–15均受影响。

CPU梯度链验证：mask=false时projector/BIT/effect head/gate梯度全为0；mask=true且gate非零时这四部分均有非零梯度。修复时还应保留gate=0时与baseline一致的行为，不能只粗暴将全部mask设true。

**P2：触觉phase/confidence分支没有参与学习。**

`cosmos-framework/cosmos_framework/model/generator/omni_mot_model.py:799`返回phase/effect/confidence，后续仅使用effect；phase gate也未使用。CPU backward验证phase/confidence没有梯度。若实验目标仅为effect residual，这是待清理的冗余；不能称已完成三者联合学习。

**CTE部署协议仍需客户端配合。**

server将返回动作前4步记录为执行历史（`cosmos-framework/cosmos_framework/scripts/action_policy_server_xhand.py:184`），并依赖attempt reset（`:406`）。本地参考客户端 `/workspace/mnt/sqzhang26/FactileLDM/ur7e_xhand_deploy_pi0_client.py:404`默认query-frequency=48，`:726`未发送对应reset。若直接沿用该默认客户端，CTE历史将错位。实际5090客户端未读取，不能推定它就是此文件。第二阶段测试须确认每次执行4步、attempt reset和真实执行动作一致。

## 已核实正常的部分和验证边界

- v3真实数据索引：state为 `[0:6] + [28:52:2]`，即6臂＋12手关节位置，未混入torque；与action元数据18个名称逐项一致。见 `cosmos-framework/cosmos_framework/data/generator/action/datasets/xhand_lerobot_dataset.py:68`、`:84`、`:89`。
- v2缺手关节是可观测性不足，但state和action必须等维/同表示并非一般要求。v3更适合作为此任务后续统一基座。v2仍可用于验证增加手部本体信息的收益。
- 当前action minmax归一化训练/推理链一致；state保持原始单位也在两端一致。`cosmos-framework/cosmos_framework/data/generator/action/datasets/action_sft_dataset.py:54`传入normalizer；Stage-2 wrapper在变换之外附加特征，未丢失归一化。Git中9/21的对象及当前HEAD仍未透传normalizer，修复仅在工作区，旧v1应列为归一化不一致的高风险版本。但没有各run的完整源码快照，不能确定修复进入运行环境的精确时间，也不能仅凭当前mtime或config断言v2训练时已经/尚未修复。v2 provenance仅保存diff哈希。
- CPU读取现存5个policy checkpoint的DCP metadata，model/optim/scheduler/trainer引用分片均存在且文件长度覆盖记录范围；trainer迭代数与目录一致。结构完整不代表整个大模型所有数值和推理正确。
- 仅读取约6.36MB小权重：v2-4000与Stage-2-500的 `net.proprio_projector` weight/bias完全相同且有限；Stage-2全部37个 `net.behavior_*` tensor均有限。此项未检查net_ema或全部约91GB模型参数。
- 冻结触觉encoder的本地转换权重存在，25 keys、740614参数，严格加载且全有限；CPU fp32/bf16输出有限。逐帧与时间维批量编码最大差约4.77e-7。本轮未重新做JAX对照。
- 当前触觉encoder/BIT既有单测15通过；XHand server单测9通过。它们未覆盖发现的meta初始化、真实checkpoint端到端推理和机器人成功率；额外CPU初始化/梯度探针覆盖前两项源码缺陷。
- 同一meta/to_empty初始化探针确认baseline的PolicyInjectionPrior、adapter、global projector均有限；baseline的5组LayerNorm为weight=1/bias=0，两组attention参数均有限，没有触觉head的同类遗漏。
- 未进行8卡触觉回归。自定义forward已注册FSDP，不能仅因gate在方法外读取就认定FSDP错误；此次CPU梯度链正常。

## 建议执行顺序

1. 保留正在训练的v3，优先修复重复验证和按checkpoint恢复state契约；从非空验证数据及真实执行结果评估v2/v3，避免依据NaN/0选模型。
2. 修好触觉head初始化、validation输入和episode开头注入mask，补能覆盖这些缺陷的测试，再补在线触觉窗口和reset接口。
3. 以同一个经验证的v3 checkpoint分别训练视觉Zeva Stage-2与触觉Stage-2，保持数据、CTE版本、seed、更新次数和有效batch一致；单独目录保存，避免复用路径覆盖来源。
4. 5090测试先确认可用服务与客户端协议，再迁移具体checkpoint。现有v2及其Stage-2仍有诊断价值，不应因22维输入就删除。

机器可读检查结果见同目录 `audit-artifacts/20260925/checkpoints.json`、`audit-artifacts/20260925/small-weights.json` 与 `audit-artifacts/20260925/tactile_cpu_probe.json`。CPU复现脚本保存在 `audit-artifacts/20260925/tactile_cpu_probe.py`，从仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' PYTHONPATH=cosmos-framework envs/zeva/bin/python audit-artifacts/20260925/tactile_cpu_probe.py
```

此次没有自动开展训练修复或新实验。
