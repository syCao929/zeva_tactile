# π0 / Cosmos 双骨干实验

`pi0_zeva/` 是独立的 PyTorch π0 训练、Zeva 记忆适配和离线推理入口。Cosmos 仍使用原来的训练入口和检查点格式。

四个最终模型的定义、成对训练预算与真机评测协议见 [四模型对照方案](MODEL_COMPARISON_PLAN.md)。其中正式比较的“baseline”是视觉 Zeva Stage 2；本页配置名 `xhand_baseline.json` 指其前置 π0 Stage 1 基座。

后续直接训练可使用 `tools/train-pi0-base.sh`、`tools/train-pi0-baseline.sh` 和 `tools/train-pi0-tactile.sh`。这些入口默认使用 global batch 112；Stage 2 必须指定具体的 π0 baseline step 目录，并用相同 `--pair-name` 启动 baseline/触觉分支。`--dry-run` 会检查输入、GPU 是否存在、pair contract 和最终命令；正式启动时还会拒绝已被占用的 GPU，除非显式传入 `--allow-busy-gpus`。

第一版迁移 CTE/BIT → PBD 动作先验 → π0 action expert 的路径。它不新增 prefix token，也不迁移 Cosmos 的全局 prefix、PIM 或未来视频生成路径。因此这里的 `zeva` 是 **Zeva 的动作先验迁移版本**，不能将其标为完整原始 Zeva 的逐项复现。

## 数据和损失

- 同一份 `press_button_4_times_merged_filtered` 数据，同一 episode 划分：seed 42，验证比例 0.03。
- 当前 `cam_left` → `base_0_rgb`；当前 `cam_front` → `left_wrist_0_rgb`。后者只是模型槽位名称，源设备没有腕部相机。第三相机槽置零、mask=false。
- 只解码当前画面；π0 不读取未来视频。PyAV 按 PTS 定位，检查时间误差不超过 0.0002 秒。
- state 为 `state[:6] + state[28:52:2]`，动作是相同18关节的绝对位置，单位弧度。
- state/action 分别用**训练 episode** 的逐帧 mean/std 做 z-score，再补零至32维。动作输出裁回18维并反归一化。不能拿 Cosmos 的 minmax stats 代替。
- 32步动作块，15 Hz。与 Cosmos 保持相同的 `episode.length - 32` 窗口集合。
- 基座 loss：真实18维的 action flow matching MSE。Stage 2：该 MSE + `0.01 * prior_nll`。prior 在同一个归一化动作空间训练；不含未来视频 loss。
- 触觉保持冻结逐帧 encoder + 30帧因果 BIT，完整1972维原始状态另行送入触觉分支；无效左填充不更新记忆。

## 配置

| 配置 | 可训练部分 | 初始化 |
|---|---|---|
| `configs/pi0/xhand_baseline.json` | π0基座 | 转换后的官方 π0 base |
| `configs/pi0/xhand_zeva.json` | PBD、动作适配器 | 本项目 π0 XHand 基座检查点 |
| `configs/pi0/xhand_zeva_tactile.json` | 上述模块 + 触觉 projector/BIT/effect head/gate | 同一 π0 XHand 基座 + 冻结触觉 encoder |

Stage 2 冻结 π0 权重，但保留经过它的反向传播，以训练记忆适配器。新的动作投影零初始化，关闭新增动作条件时与同一个 π0 基线等价。触觉 gate 零初始化时与视觉 Zeva 分支等价。

触觉配置使用已转换、与 Cosmos 共享的 `models/zeva/tactile_patch_encoder_19999.pt`，不能把外部 FactileLDM 的 Orbax 检查点目录直接交给 PyTorch 加载。

所有相对资源路径以本仓库根目录为基准。配置支持 `extends` 和 `--set key=value`；输出配置、数据统计、损失分项和实际全局 batch 会保存到 run 目录。

## 准备依赖与权重

需要 Python 3.11 的独立环境。不能把 OpenPI 要求的 `transformers==4.53.2` 和其补丁装入正在使用的 Cosmos 环境。

```bash
bash tools/setup-pi0-env.sh
```

脚本只修改带有 `.zeva-pi0-environment` 标记的新环境，默认 `envs/pi0`。`OPENPI_ROOT` 默认指向同级 `openpi-3d-tactile`。源码接口固定为官方 commit `215abfb217dbac7d5f1273282331b9b1866c0479` 的 `pi0_config.py` / `pi0_pytorch.py`；入口检查这两个文件，环境检查另外逐文件核对五个 transformers 补丁。并不宣称本机 OpenPI 的所有其他文件均未修改。

2026-09-25 已建好 `envs/pi0`：Python 3.11.9、PyTorch 2.7.1+cu126、Transformers 4.53.2、NumPy 1.26.4、JAX 0.5.3。73个依赖的兼容性检查、官方补丁检查和GPU BF16前向/反向通过。`../openpi-3d-tactile/.venv` 实际链接到 `../FactileLDM/env/.venv`；该原环境缺少补丁和数据依赖，不能直接用于这套训练。本次按解析后的精确版本复制63个匹配包的独立文件，再安装10个缺失或不兼容的包；未改动原环境。安装脚本使用copy模式，并通过新文件替换补丁，避免修改共享硬链接。完整版本表在 `audit-artifacts/20260925/pi0_training/environment-freeze.txt`。

训练只接受严格加载成功的 PyTorch safetensors，默认位置：

```text
models/pi0_base_pytorch/model.safetensors
```

已将 `../hf_weight/pi0_base` 的 JAX/Orbax 权重转换到上述路径，文件约7.01 GB。`pi0_zeva.convert_checkpoint` 固定校验官方转换源码，复用其参数映射，严格检查每个预测参数的来源、形状、有限值、加载后的值和保存读回一致性。50个JAX参数叶映射为778个模型键；PaliGemma输出头共享embedding，另一个从未参与动作预测的expert语言输出头显式清零。保留官方混合精度，norm和动作projection等仍为FP32。逐参数报告在 `models/pi0_base_pytorch/conversion_report.json`。

如需转换其他副本，使用一个尚不存在的输出目录；转换器拒绝覆盖已有产物。输入必须指向 **pi0_base目录，而不是params子目录**：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu LD_LIBRARY_PATH='' \
  envs/pi0/bin/python -m pi0_zeva.convert_checkpoint \
  --checkpoint-dir ../hf_weight/pi0_base \
  --output-dir models/pi0_base_pytorch
```

禁止用 Cosmos checkpoint 初始化 π0；默认也禁止未加载权重就开始训练。测试中的随机小模型是显式的独立测试 fixture。

## 数据检查与训练

以下命令均在本仓库根目录运行。训练统计已生成在 `datasets/pi0_xhand_norm.json`；数据集变动后需重新生成并开启新 run。

```bash
bash tools/run-pi0-xhand.sh --compute-stats
bash tools/run-pi0-xhand.sh --data-smoke
bash tools/run-pi0-xhand.sh --preflight
```

`--preflight` 只检查依赖和资源，不加载大模型、不占用训练GPU。缺少转换后的权重时会明确失败。当前独立环境中的preflight和真实数据smoke均已通过。

正式训练前先用已准备的权重进行小规模 GPU 检查，包括保存恢复：

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/run-pi0-xhand.sh \
  --set max_steps=2 --set warmup_steps=0 --set grad_accum=1 \
  --set eval_batches=2 --set save_every=1 \
  --set output_dir=runs/pi0/smoke-baseline
```

分配好GPU后，八卡基座训练入口是：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PI0_NPROC=8 \
  bash tools/run-pi0-xhand.sh --config configs/pi0/xhand_baseline.json
```

默认每卡batch 1、累积14次，八卡全局batch 112，与当前七卡 Cosmos 的16×7样本数匹配；目标5000次 optimizer 更新。这里使用 PyTorch DDP，没有假定 OpenPI 自带 FSDP 或 LoRA 支持。完整基座在单张A800上、batch 1的实测峰值allocated显存约30 GiB；八卡DDP通信与累积后的显存、吞吐仍需另行验证。

基座完成后，两个 Stage-2 配置都从同一基座加载：

```bash
PI0_NPROC=8 bash tools/run-pi0-xhand.sh --config configs/pi0/xhand_zeva.json
PI0_NPROC=8 bash tools/run-pi0-xhand.sh --config configs/pi0/xhand_zeva_tactile.json
```

这些是各自独立的启动命令，不应在同一批GPU上同时执行。默认Stage 2每卡batch 2、累积7次，目标2000步。冻结基座保持eval模式时，官方activation checkpointing不生效，因此Stage 2的实际显存也要实测。

## 验证、保存和恢复

- `metrics.jsonl` 的训练loss是所有rank、所有累积microbatch的均值；记录 action flow、prior NLL、总loss、梯度范数和学习率，同时记录当前rank的峰值allocated/reserved显存，启动事件包含GPU型号和Torch版本。
- 验证每次新建迭代器，在验证窗口集合中均匀选取固定20个样本，使用相同噪声和均匀采样的时间步；关闭图像训练增强。零batch或非有限loss直接报错。
- 每500步及训练结束保存，默认保留最新两个checkpoint。更新 `latest.json` 前先原子发布完整目录。
- Stage 2仅保存记忆模块和优化器；run目录固定保存一份基座，优先硬链接，避免原基座run清理旧checkpoint导致Stage 2不可恢复。
- checkpoint包含逐rank RNG、数据epoch/位置、优化器、配置、norm SHA256及tokenizer/CTE manifest/触觉encoder哈希。训练恢复要求原run目录、原world size及一致输入契约。

```bash
PI0_NPROC=8 bash tools/run-pi0-xhand.sh \
  --resume runs/pi0/action_xhand/v1-joint18/checkpoints/latest.json
```

独立动作采样评测会输出机械臂/手关节各自的 MAE、RMSE，单位为弧度。训练的flow loss不等同于这些误差，也不能把不同骨干的总loss直接比较。

```bash
PYTHONPATH=.:cosmos-framework JAX_PLATFORMS=cpu LD_LIBRARY_PATH='' \
  envs/pi0/bin/python -m pi0_zeva.inference \
  --checkpoint runs/pi0/action_xhand/v1-joint18/checkpoints/latest.json \
  --max-samples 32 --num-steps 10 --output plots/pi0_joint18_eval.json
```

`inference.load_policy(...).predict(batch)` 接收与数据集一致的规范化batch，输出 `[B,32,18]` 绝对关节位置；它不是机器人客户端。在线 CTE 更新、触觉采集和硬件控制尚未接入这个新入口。

## 已完成的验证（2026-09-25）

- 真实数据：98个训练episode、3个验证episode（38/68/99），42673/1492个训练/验证窗口；统计只使用45809个训练帧。
- 真实双相机、CTEv4和触觉样本读取；PyAV在两相机frame 0/40的输出与原TorchCodec逐像素一致。
- 真实触觉encoder、真实parquet和CTEv4接入新π0桥接层的CPU检查通过：frame 0/40有效触觉长度为1/30，gate有非零梯度，encoder与基座保持冻结。该检查的π0骨干使用小型测试替身，结果在 `audit-artifacts/20260925/pi0_real_tactile_cpu.json`。
- CPU测试覆盖数据划分、未来信息隔离、18维归一化/补齐/反演、零adapter等价、触觉mask、冻结参数/梯度、检查点与RNG恢复、验证重入、推理输出契约。
- 独立 `envs/pi0` 的64项CPU测试全部通过，没有跳过官方OpenPI小型集成测试；覆盖真实forward、KV-cache采样、baseline activation-checkpoint backward、冻结Stage-2 adapter backward及严格转换的异常拒绝。此前两进程CPU/Gloo的baseline和Zeva梯度累积检查也通过，更新后三步参数在两rank逐位一致。
- 官方π0 base已完成真实Orbax转换；所有预测权重完整，777个保存张量全部有限，加载/保存读回均逐值一致。
- GPU0（A800 80GB）上完整预训练π0、真实双相机/18关节数据的基座训练完成3步，batch 1、无梯度累积。训练loss为0.783455、0.601223、0.745997；固定2个验证窗口的loss从2.434408变为1.050485。峰值allocated显存29.99 GiB，reserved为30.38 GiB；每步计算约0.75–1.32秒，不含约30秒/次的检查点保存。这只是链路检查，不是收敛或完整验证集指标。
- 从第1步恢复并重新训练至第3步：模型参数、完整Adam状态、Python/NumPy/Torch/CUDA RNG及数据位置与连续训练逐值一致。相比初始权重，623,045,227个参数元素确有更新，动作、状态、时间投影和action expert均更新。结果见 `audit-artifacts/20260925/pi0_training/resume_verification.json`。
- 最终检查点在2个验证窗口完成10步动作采样，输出有限的 `[1,32,18]` 绝对关节位置。arm/hand MAE分别约0.2005/0.4794 rad，仅用于确认保存后的推理链路；不能当作已训练策略的性能结论。结果见 `audit-artifacts/20260925/pi0_training/inference.json`。
- 本轮仅完成环境和真实训练验证，未启动5000步正式训练。GPU1–7的Cosmos V3继续运行。八卡GPU DDP、完整Stage-2模型的GPU训练以及机器人部署仍待验证。短训练产物独立保存在 `runs/pi0/smoke-baseline-20260925`，不会被默认正式训练或Stage-2配置误用。

本轮环境、命令和证据索引见 [验证记录](audit-artifacts/20260925/pi0_training/README.md)。

常规CPU检查：

```bash
PYTHONPATH=.:cosmos-framework:../openpi-3d-tactile/src \
  CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' JAX_PLATFORMS=cpu \
  OMP_NUM_THREADS=4 PI0_ZEVA_TEST_OPENPI=1 \
  envs/pi0/bin/python -m pytest -q pi0_zeva
```

`PI0_ZEVA_TEST_OPENPI=1` 启用官方小模型集成检查，当前独立环境已满足依赖。
