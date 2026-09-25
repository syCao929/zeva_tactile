# π0 独立环境与真实训练验证 · 2026-09-25

结果：环境、真实权重转换、完整模型训练、保存恢复和动作采样均通过。本轮按用户确认只验证训练，没有启动正式长训。

## 可直接使用的资源

所有路径以 `/workspace/mnt/sqzhang26/zeva-work` 为根目录：

| 资源 | 路径 |
|---|---|
| 独立Python | `envs/pi0/bin/python` |
| 转换后π0权重 | `models/pi0_base_pytorch/model.safetensors` |
| 参数来源与转换报告 | `models/pi0_base_pytorch/conversion_report.json` |
| 训练归一化统计 | `datasets/pi0_xhand_norm.json` |
| 短训练配置 | `audit-artifacts/20260925/pi0_training/smoke-baseline.json` |
| 短训练最终检查点 | `runs/pi0/smoke-baseline-20260925/checkpoints/step_00000003` |
| 训练事件与loss | `runs/pi0/smoke-baseline-20260925/metrics.jsonl` |

Python 3.11.9；PyTorch 2.7.1+cu126；Transformers 4.53.2加官方5文件补丁；NumPy 1.26.4；JAX 0.5.3；Flax 0.10.2；Orbax 0.11.13。完整环境见 [environment-freeze.txt](environment-freeze.txt)。73个包的依赖检查通过。

原 `FactileLDM/env/.venv` 与 `openpi-3d-tactile/.venv` 是同一环境，缺少补丁、torchvision、pyarrow、av等，不能直接运行本项目。新环境复制63个精确匹配包的独立文件并安装10个包，原环境的依赖及补丁保持原样。审计见 [source_environment_audit.json](source_environment_audit.json)，复制来源见 [reused_packages.json](reused_packages.json)。

## 实际验证

| 检查 | 结果 | 证据 |
|---|---|---|
| 环境与资源preflight | 通过，全部5个补丁匹配 | [preflight.json](preflight.json) |
| 真实数据 | 42673训练窗口、1492验证窗口；双图、状态、动作均有限 | [data-smoke.jsonl](data-smoke.jsonl) |
| Orbax → PyTorch | 50个JAX参数叶、778模型键；预测参数无缺失，保存读回一致 | [conversion_summary.json](conversion_summary.json) |
| CPU测试 | 64 passed，含官方OpenPI组件集成 | [pytest-final.log](pytest-final.log) |
| GPU训练 | 完整π0预训练权重+真实XHand数据，完成3次Adam更新 | [baseline-train.log](baseline-train.log) |
| GPU恢复训练 | 从step 1继续到step 3，与连续训练逐值相同 | [baseline-resume.log](baseline-resume.log) |
| 权重确实更新 | 623,045,227个元素发生变化；动作/状态/时间投影及expert均更新 | [resume_verification.json](resume_verification.json) |
| 恢复状态完整 | 模型、Adam、数据游标、四类RNG精确相同 | [resume_verification.json](resume_verification.json) |
| 保存后动作采样 | 2个验证窗口、10步flow采样，输出有限18维关节位置 | [inference.json](inference.json) |

单卡batch 1、梯度累积1，训练loss为0.783455、0.601223、0.745997，梯度范数均有限且非零。固定2窗口验证loss为2.434408 → 1.050485。峰值allocated显存29.99 GiB、reserved 30.38 GiB；计算每步约0.75–1.32秒，检查点保存另需约30秒。PyTorch/CUDA版本与GPU记录见 [cuda-runtime.json](cuda-runtime.json)。

这3步用于证明训练链路真实可用。它们不构成收敛、任务成功率或与Cosmos对照的结论。推理arm/hand MAE约0.2005/0.4794 rad也仅来自2个窗口。

## 本轮使用的命令

基座短训练（输出目录已存在，重新从头测试时需要新目录）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  bash tools/run-pi0-xhand.sh \
  --config audit-artifacts/20260925/pi0_training/smoke-baseline.json
```

恢复验证先将连续训练的step 2/3和原始metrics保留到本目录的 `uninterrupted/`，再从原run的step 1继续，以避免覆盖已有checkpoint。使用相同配置和max_steps：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  bash tools/run-pi0-xhand.sh \
  --config audit-artifacts/20260925/pi0_training/smoke-baseline.json \
  --resume runs/pi0/smoke-baseline-20260925/checkpoints/step_00000001
```

上述恢复已经完成，当前step 2/3再次存在，不能原样重复这个覆盖旧步骤的测试。正常续训应使用 `checkpoints/latest.json`。

GPU1–7的Cosmos V3保持运行；本轮测试仅使用GPU0并已退出。八卡DDP和完整Stage-2 GPU训练尚未验证；π0机器人客户端尚未接入。正式训练配置仍为 `configs/pi0/xhand_baseline.json`，不会加载这次短训练的权重。
