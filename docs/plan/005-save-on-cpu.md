# P005：6032-token 显存对照、长样本容量与 SFT

日期：2026-10-10。源规格：[抽样计划合并审查节](005-sampling-sft-pair.md#合并审查后的研究安排2026-10-10)。
原执行分支：`p005-save-on-cpu`。训练模式问题及修复后的研究顺序见文末合并审查。

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

## 2026-10-11 合并审查与下一步

本轮四个提交交付了可选 backbone CPU 保存、两臂显存阶段记录和完整交易扫描。**合并审查发现实验 worker 漏了 `model.train()`：backbone/decoder 仍在 eval 模式，decoder checkpoint 未实际执行。** 因此旧两臂观测不能回答正常 checkpoint 训练配方下 `save_on_cpu` 是否有效，不能据此直接转向 MLP/FSDP 改造。

本地加载 tiny Gemma＋PEFT 复现：外层 PEFT 的 `training=True`，但 base 和两个 decoder 的 `training=False`；checkpoint 标志虽开，实际调用为 0。显式 `model.train()` 后两个 decoder 都执行 checkpoint。官方实现的触发条件是 `gradient_checkpointing and self.training`，见 [Transformers 5.13.1 checkpoint 实现](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/modeling_layers.py)及[模型加载实现](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/modeling_utils.py)。

### 已修正并可合并的内容

- worker 在前向前显式设训练态，并记录/校验 base、decoder 的 training、checkpoint 开关及函数；不能只检查外层 PEFT 的 training。
- CPU 数值核对同时覆盖 checkpoint 关闭和开启，观察实际 decoder checkpoint 调用。开启时两臂 loss 与全部 20 个 LoRA 梯度差为 0；原关闭场景最大梯度差仍为 8.94e-8。两臂梯度均有限且非零。运行时 `save_on_cpu` 仍默认关闭。
- 扫描器在真正测长前排除预留 issue 及相同 repo＋commit，并用历史 v2 问题集计算新增问题，避免拿本轮 prepare 集合减自身。重算复现全部历史汇总：214 个完整前缀、9 个合格问题，8K 内 24 条、其中有标签 6 条且都是局部编辑，无恢复事件、无新独立问题。
- 共 85 项本地检查通过：训练/编码/数据 71、载体 6、扫描 7、配对记录 1。没有启动新的 Kaggle GPU 作业。

原始 `kaggle-output/`、遥测、配额与旧 CPU 数字不改写；派生摘要和下一步指引已更正，并附[机器可读审查记录](../../research/005/save-on-cpu-6032-20261010/review-20261011.json)。审查前源码/结论可在提交 `3cde33a` 回溯。修复后的 kernel 尚未远程运行；使用前从当前仓库重新 build，避免旧 worktree 绝对路径。

### 第一优先：修正训练态后的同样本对照

用同一 6,032-token 完整交易，保持 NF4、LoRA、loss、标签、注意力和优化器设置一致，两臂都明确启用训练态与 decoder checkpoint；唯一比较变量为 backbone `save_on_cpu`。每臂用新的分布式 worker，尝试一个完整优化步（含第一次 Adam 状态创建）。保留训练态记录、forward/backward/optimizer 边界、层号、完整 traceback、GPU 峰值和耗时；不再把开关配置本身当作 checkpoint 实际执行的证据。

同时记录各 worker 的 RSS、主机可用内存，以及可取得的 cgroup `memory.events`；不可读取时注明缺失。旧 SIGKILL 只说明进程被杀，尚未证明主机 OOM 或 pinned-memory 耗尽。旧 baseline 确实在 decoder 索引 1 的 gate_proj/F.linear 分配 248 MiB 失败，但那是在 checkpoint 未激活时的现象；这一次先确定真实训练态下的失败阶段。

建议这项仍最多 **0.6 个 GPU 额度小时**，设置整个对照会话的总时间上限，避免只限单臂；不自动连续重试。若失败就保留证据，再按结果选择修改，不能先承诺 offload 节省多少显存。最近保存的账号快照为北京时间 **2026-10-11 00:14**：已用 19,451.299 秒，剩约 **24.597h**；启动前重读，快照不是当前余额。

### 第二优先：有效后才扩展长度和候选训练

若纠正后的 offload 臂完成首个优化步，再验证完整 train 的 9,841／12,832 tokens，各做一步含 Adam；成功只能证明相应真实长度，不能声称已测 16,384。容量阶段的 select 更新必须丢弃，正式 SFT 从干净 adapter 初始化。

若仍失败，依据实际失败阶段决定下一项：主机内存增长则检查保存了哪些张量及复制规模；训练态局部算子峰值才研究相应分块/融合；注意力路径则检查实际 kernel/mask/KV。只有证据支持时才投入 FSDP-QLoRA 和缓存格式改造。此前“Live MLP/dequant 已定位”“host pin_memory 导致 SIGKILL”的确定性表述已撤回。

### 数据与收益研究并行推进

现有本地数据没有增加独立问题，反复选更短前缀不会产生新的恢复训练分布。下一轮优先获取更多非预留问题的、协议兼容的真实修复/恢复轨迹，记录来源和最终 resolved；CPU 先做去重、来源与长度统计，再安排训练。不得使用 heldout 内容或预留八题及其相同快照。若研究证据摘要/状态化上下文，应另建数据表示实验，与原完整前缀样本分别记录，不能以裁剪冒充原样本。

得到真实 SFT adapter 后才做固定 T0／六工具／预算的八题 base-adapter 配对，记录新增修复、回退、耗时。现阶段尚无修复收益证据；训练支线以小预算验证可行性，避免仅因技术路径能运行就扩大投入。

### 清理时保留的材料

旧 worktree 的 17 项非缓存忽略文件已逐项校验，存于 `training/artifacts/worktree-archives/p005-save-on-cpu-3cde33a-20261011/ignored-files.tar.gz`，旁边 `manifest.json` 记录哈希。5 个容量输入/日志/遥测文件已复制到主目录原相对路径。归档不进 Git，删除旧 worktree 后仍可恢复过程材料。

## 2026-10-11 本次实测（训练态 6032 对照）

私有 4×L4 kernel **COMPLETE** v1。**Live URL slug** 为 `ccjjllt/p005-save-on-cpu-6032-train-mode`；metadata id 仍为 `ccjjllt/p005-save-on-cpu-6032-tm`。同一交易 `fastapi_11194-3f51fb716edd-tx01`（6032 tokens）。证据目录 `research/005/save-on-cpu-6032-trainmode-20261011/`。

对照对 checkpoint 训练有效：两臂 `outer_training=true`、`base_training=true`，60 层 `decoder_checkpointing_active=true`；实际调用 baseline 48/60（前向中途失败）、offload 60/60（前向完成）。

**baseline：** 前向 CUDA OOM，层 47，`sdpa_attention_forward` / `scaled_dot_product_attention`，申请 378 MiB，进程占用 21.69 GiB / 22.03。`optimizer_reached=false`。主机 cgroup `oom_kill=0`。

**save_on_cpu：** 已安装。前向完成（15.38s，loss 0.0603，末层 59，峰值 allocated 18.988 GiB，调用 60/60），随后反向 CUDA OOM，层 53，申请 124 MiB，占用 21.91 GiB。`optimizer_reached=false`，无 adam_state。主机 RSS 约 6.29 GiB，MemAvailable 约 166 GiB，`oom_kill=0`，不是 SIGKILL。

实测差异：offload 完成了 baseline 未完成的前向，两臂都没有完成第一次 Adam。不能据此承诺 9841/12832 或训练可节省若干 GiB。Wrapper `success=true` 只表示两臂都记录了 oom/completed，不是训练成功。3.62 GiB 激活估算不作结论。

9841/12832 **BLOCKED**（`did_not_push_9841` / `did_not_push_12832` / `did_not_retry_16k` / `did_not_rewrite_fsdp`，`gpu_launched=false`）。真实 SFT 与八题 T0 配对跳过：没有完整 train 长度反传，也没有新的独立非预留样本。诊断 adapter 不是修复收益臂。未比赛提交。

下一刀按实际失败阶段：checkpoint 下的 live 算子是层 47 的 SDPA；offload 失败在反向层 53 的 CUDA，不是主机 OOM。已撤回的 v1 表述（“Live MLP/dequant 已定位”“host pin_memory 导致 SIGKILL”）仍不成立。历史 `research/005/save-on-cpu-6032-20261010/result.json` 状态保持 `invalid_checkpointed_training_comparison`。配额 Δ 612.944s = 0.170262h（<0.6h）。后续轮询用 live slug `p005-save-on-cpu-6032-train-mode`。
