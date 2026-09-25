# 触觉链路修复记录 — 2026-09-25

> 后续清理更新：按用户指示，V2 专属权重、运行日志及评测副本已删除，V2 推理服务已停止并释放 GPU 0。详见 `V2_CLEANUP_20260925.md`。下面的训练进度和服务状态为修复完成时的历史快照。

按本轮决定，后续正式基线采用 v3 joint18；v2 不再作为正式对照起点。本次没有删除 v2 权重或重启正在训练的 v3。

**训练状态快照（15:21，Asia/Shanghai）**：v3 到2470/5000步，已运行4小时13分，最近每步约5.6秒，剩余训练计算约4小时，另加保存时间。现存完整checkpoint为500/1000/1500/2000。torchrun PID131922使用7卡；第八张卡运行v2-4000推理服务PID126596。证据：`logs/v3-joint18-20260925.log:3919`、`logs/v3-joint18-launch.log:12`、`logs/serve-stage1.log:18`。

## 已实施

- **完整初始化**：BIT和触觉head提供统一reset_parameters，模型在to_empty之后调用；覆盖Linear、GRU及全部LayerNorm。phase/confidence暂保留checkpoint结构但冻结，optimizer仅选择实际用于effect residual的模块。见 `cosmos-framework/cosmos_framework/model/zeva/tactile_memory.py` 与 `model/generator/mot/cosmos3_vfm_network.py`。
- **修复触觉被视觉mask屏蔽**：当前触觉使用独立 `behavior_tactile_effect` 字段；在FSDP forward内乘gate，PolicyInjectionPrior先完成视觉BOS替换，再加入触觉。零gate时包含全无效视觉历史的场景均与baseline精确一致；episode开始时gate可学习。见 `cosmos-framework/cosmos_framework/model/zeva/policy_injection.py`。
- **训练与验证输入一致**：触觉recipe的train和val都开启30帧窗口，并继承joint18 state及同一CTE缓存配置。见 `cosmos-framework/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_xhand_zeva_tactile.py`。
- **修复重复验证为空**：PackingDataLoader新增默认关闭的 `restart_on_iter`，只在验证loader打开，重建耗尽iterator并清空残留buffer；训练无限数据流和恢复步数语义不变。见 `cosmos-framework/cosmos_framework/data/generator/joint_dataloader.py`。
- **在线服务贯通**：服务按模型配置启用触觉，接收客户端采集的连续原始窗口、mask及帧编号，检查当前帧、episode/reset、查询间隔及finite值，左补零到30帧后传入模型。不会从每4步的稀疏查询伪造15Hz触觉历史。见 `cosmos-framework/cosmos_framework/inference/xhand_tactile.py` 与 `scripts/action_policy_server_xhand.py`。
- **可运行客户端适配**：`tools/run_xhand_tactile_client.py`加载原Factile客户端，不修改原文件；每个控制tick采集，四步查询，发送完整窗口，桥接相机字段，推理失败终止attempt。使用说明见 `cosmos-framework/docs/xhand_tactile_serving.md`。

## 验证

共67项相关CPU单测通过，未使用训练GPU：

| 范围 | 数量 | 覆盖 |
| --- | ---: | --- |
| encoder / BIT / policy injection | 24 | 初始化、因果窗口、零gate等价、BOS后注入 |
| 验证loader | 6 | 0/1 worker、persistent worker、重复及中断遍历、预热与训练流连续性 |
| server / client | 27 | 输入贯通、缺失/稀疏/未来/非法数据拒绝、reset、重连和客户端适配 |
| recipe / dataset | 10 | train/val一致、optimizer范围、训练与线上窗口逐元素一致、跨episode隔离 |

此外，用真实 `tactile_patch_encoder_19999.pt`、真实episode_000000.parquet及对应CTE v4特征，调用当前源码的encode、attach、action injection路径：

- frame0：29帧左补零＋1帧真实触觉，视觉effect全无效；gate=0与baseline完全相同且gate梯度非零；gate=0.3时projector、BIT、effect head均有非零有限梯度。
- frame40：30帧真实窗口、有效视觉历史；同样满足上述条件。
- 冻结encoder无梯度，未使用的phase/confidence分支冻结，初始化后不再存在未覆盖的NaN哨兵。

证据保存在 `audit-artifacts/20260925/zeva_tactile_real_cpu_20260925.json` 及同名脚本；随机输入及完整源码接线探针为 `zeva_tactile_fix_cpu_20260925.json`。结果中的hash对应测试时源码，之后仅对两个模块做了格式整理。

原Factile客户端通过新wrapper实际执行 `--check-config`，确认1972维state、18维action；该模式未连接服务或机器人。相关新模块及小范围修改的Ruff、Python编译、git diff检查通过。两个大型模型文件仍有既有import顺序和可选分支缺失符号 `bounded_action_residual` / `compute_teacher_anchor_loss` 的lint问题；此次未扩大修改这些未启用的路径。

## 生效范围与后续运行

这些修改对新启动的进程生效。正在运行的v3仍持有旧代码，重复验证修复不会热更新进去；其checkpoint应在修复后的独立评测进程中补评。当前v3的7卡运行继续保持，未中途改变world size或有效batch。

本次没有启动新触觉训练或真实checkpoint的GPU推理，也没有操作5090或机器人。CPU链路及真实数据梯度检查通过不等于8卡FSDP、checkpoint保存/恢复、完整策略推理和实机成功率已验证。后续以选定的同一个v3 checkpoint启动视觉/触觉两组Stage-2，统一卡数、有效batch、数据和更新次数，使用独立输出目录。

客户端适配保留硬件安全限幅。原客户端同步等待推理会延长查询处的实际时间间隔，因此15Hz配置和连续控制帧契约不代表连续墙钟15Hz已经实测；若硬件改写命令，精确CTE动作归因仍需要执行动作反馈。
