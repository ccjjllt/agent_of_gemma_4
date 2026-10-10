# P005：6032-token 显存对照、长样本容量与 SFT

日期：2026-10-10。源规格：[抽样计划合并审查节](005-sampling-sft-pair.md#合并审查后的研究安排2026-10-10)。
工作分支：`p005-save-on-cpu`。不要改 `main`。

目标：用同一 6,032-token 完整样本定位显存峰值，对照 backbone `save_on_cpu`；有效后再测 9,841／12,832，并在容量与数据都支持时做一个真实 SFT 和八题 T0 配对。

研究优先：阶段峰值、traceback、对照差异。不要新建门控平台。

## Global Constraints

- 不覆盖 P002 冻结变体、历史结果、heldout、v1/v2 普查文件。
- 不正式提交比赛。不截断历史。不把诊断 adapter 当修复收益。
- 训练路径：Python 3.13 + `ccjjllt/p005-cpu-nf4-cache` + `ccjjllt/p005-training-cpu-stage`。不重量化 62.5GB，不装 41 个推理 wheel，不重跑四步诊断。
- Kernel 用 CLI `kaggle kernels push`。从**当前仓库**重新生成载体，不要沿用已删除 worktree 的绝对路径。
- 本轮第一项 GPU 建议最多 **0.6 个额度小时**。4×L4 按两倍 wall 计费。启动前重读配额。上次快照已用 12,002.806s（14:10:30 UTC），以启动时读数为准。
- NF4、LoRA rank8/alpha16、标签、注意力实现、CE chunk、同一 6032-token 输入在对照两臂保持固定。只改变是否对 backbone 保存张量启用 `torch.autograd.graph.save_on_cpu(pin_memory=True)`。
- 容量阶段用过的 select 更新须丢弃；正式 SFT 从干净 adapter 初始化。
- 12,832 成功 ≠ 16,384 已验证。
- 预留八题及其 repo＋commit 不得进 train/select：`fastapi_13713`、`requests_6592`、`requests_6644`、`requests_7309`、`requests_7505`、`rich_3130`、`rich_3296`、`rich_3469`。
- 实现者自己 commit。

## Task 1: 同一样本的显存阶段诊断与 save_on_cpu 对照

优先改 `training/memory.py` 的 backbone 调用与容量 worker 的阶段记录，不先重写 FSDP。

### CPU

1. 微型 Gemma 上核对：原配方与 `save_on_cpu(pin_memory=True)` 包住 backbone 后，loss 与 LoRA 梯度一致（容差与现有 `test_memory.py` 同级）。
2. 在 `selected_token_forward` 里对 `base.model(...)` 增加可选上下文：仅保存张量走 CPU（`pin_memory=True`）。默认关闭，不改变已证明的 2048 诊断路径除非显式打开。

### GPU（≤0.6h 额度）

1. 新 kernel slug，例如 `ccjjllt/p005-save-on-cpu-6032`。挂 NF4 缓存与 7-wheel。输入必须是同一完整交易 `fastapi_11194-3f51fb716edd-tx01`（6,032 tokens）。
2. 两臂：`baseline`（现配方）与 `save_on_cpu`。若 baseline OOM，**结束该分布式 worker** 后再跑对照臂，不复用失败状态的 DDP reducer。
3. 每臂记录：forward / backward / optimizer 边界（optimizer 若未到达则标明）、最后执行的 decoder 层（能取到就记）、完整 traceback、逐阶段 allocated/reserved 峰值及耗时。只保留必要诊断，避免 profiler 放大显存。
4. 不自动连续重试 16K。不截断。不把容量更新当正式 adapter。
5. 证据目录：`research/005/save-on-cpu-6032-20261010/`。历史 `p005-capacity-sft-v2` v1 报告不改写。

### 完成标准

- CPU 微型核对有数字。
- GPU 有 baseline 与 offload 的阶段峰值或明确 OOM 阶段。
- 结论只能来自实测：节省量、速度、是否撑过首步（含 Adam 状态创建）。不能用 3.62 GiB 估算代替。

## Task 2: 若 offload 有效，测 9841／12832 并扩充训练 issue

仅当 Task 1 的 offload 臂完成了含首轮 Adam 的优化步。否则记录 BLOCKED，把 traceback 指向下一刀（分块/融合、SDPA/KV、或 FSDP），**不**自动消耗额度去冲 16K。

### 容量

同一加载或新 kernel（仍计本轮额度，启动前重读）。完整 train 样本 9,841 与 12,832 tokens 各做一步含 Adam 创建。成功后才能声称这两个真实长度的容量。丢弃容量阶段 adapter。

### 数据

CPU 继续扩充独立训练 issue 与真实恢复事件：来源、失败观察、成功动作、resolved。研究自然较短且上下文完整的交易。不把裁剪后的 v2 冒充完整前缀。预留八题及同 repo＋commit 仍排除。输出新目录，不覆盖 v2 普查。

## Task 3: 小规模真实 SFT 与八题 T0 配对

仅当 Task 2 证明至少一个完整 train 长度可反传，且有独立于预留八题的可训样本。否则跳过。

SFT：rank8/alpha16、LR 5e-5、≤1 epoch、`norms_fp32`、`selected_chunked`、干净初始化、一次加载。loss ≠ 修复增益。

配对：八题固定 T0/六工具/预算，一次 vLLM，base vs 该 adapter。不提交比赛。无 adapter 则只写 skip 记录。
