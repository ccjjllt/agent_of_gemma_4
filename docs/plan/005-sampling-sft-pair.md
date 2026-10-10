# P005：修正抽样后容量、SFT 与八题配对

日期：2026-10-10。源规格：[下一轮计划合并审查节](005-next-round-sft.md#合并审查后的下一轮工作2026-10-10)。
原执行分支：`p005-sampling-sft`。本轮结果、合并修复与后续研究安排见文末。

目标：先修掉丢掉全部恢复候选的抽样偏差，再按新完整上下文长度做容量与一个真实 SFT，最后在 8 个固定 T0 题上比较 base/adapter 修复数。

研究优先：交付可复算的选样、长度、训练和配对证据。不要新建门控平台，不要为了报告通过率堆测试。实现者用少量真实片段确认接口即可。

## Global Constraints

- 不覆盖 P002 冻结变体、历史结果、heldout 内容。
- 不正式提交比赛。
- 不截断历史观察。超长交易整段拒绝。
- 旧 v1 普查与 `training/artifacts/repair-sft-v1/` 保留，不覆盖。
- 诊断 adapter 只用于兼容，不当修复收益臂。
- 训练过的 issue 及相同 repo＋commit 不能当独立收益证据。
- 配对固定 T0 提示、官方六工具、预算；不要同时换 prompt 和 adapter。
- 推理镜像钉 `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`。Python 3.13 上不装 v30 cp312 wheel。
- 训练走 Python 3.13 + `ccjjllt/p005-cpu-nf4-cache` + `ccjjllt/p005-training-cpu-stage`。不要重量化 62.5GB，不要装 41 个推理 wheel，不要重跑已证明的四步诊断。
- Kernel 用 CLI `kaggle kernels push`。MCP `save_notebook` 挂不上数据源。
- 建议 GPU 额度：容量 0.6h、SFT 1.0h、八题配对 3.5h，合计 5.1h。4×L4 按两倍 wall 计费。启动前重读配额。账号快照约已用 8442.630s、剩 27.655h（2026-10-10 09:47 UTC），以启动时读数为准。超过建议上限则停止后续 GPU。
- 协议 `p005-harness-tools-v1`。
- 实现者自己 commit。

预留配对 8 题（v2 训练和 select 都必须排除它们及相同 repo＋commit）：

`fastapi_13713`、`requests_6592`、`requests_6644`、`requests_7309`、`requests_7505`、`rich_3130`、`rich_3296`、`rich_3469`

## Task 1: CPU 修正行为标签与跨轨迹抽样并重算长度

第一优先。只做 CPU。先完成这项再决定 GPU。

偏差：39 条 trace → 214 交易，标签 194 `local_edit` / 20 `tool_recovery`；每 issue 先到先得 3 条，9/9 issue 都取第一条 trace 的 tx01–tx03。27 条全是 `local_edit`，20 个恢复候选全丢。`local_edit` 是 `prepare_local.py` 默认标签。`transactions.py` 按同工具名连 error→ok，20 个候选里 23 条关联只有 10 个不同 trace＋failed_call。

### 要做的

1. 改 `training/data/{prepare_local,transactions,split_registry}.py` 和普查入口（可新建 `training/data/sample_behaviors.py` 或扩展 `length_census.py`，不要覆盖 v1 报告）。
2. 按实际工具动作标记 `local_edit` / `tool_recovery` / `verify_close`（或文档等价名）。失败调用只作上下文，不作正向监督。恢复必须关联同一文件/操作和失败观察；后续成功结束该失败事件，避免重复监督同一恢复。
3. 每 issue 仍最多 3 条。先按恢复事件去重，再跨 trace、按真实行为选样，固定种子（建议 20261010）消除文件顺序偏差。有证据时覆盖编辑、恢复、验证；缺失类别写入报告，不能捏造。
4. 保留完整因果前缀、工具 JSON、来源 trace、调用 ID、失败/成功观察和最终 resolved。`action_succeeded` 与 `task_resolved` 分开记录。按 issue 与 repo＋commit 去重，train/select 分离。上述 8 题及其 repo＋commit 不得进入 train 或 select。
5. 输出到 `training/artifacts/repair-sft-v2/` 和 `research/005/length-census-v2-20261010/`。重算 4K/8K/16K 覆盖、独立 issue/事件数、监督 token 数及密度、行为分布。完整模板编码，`--on-oversize skip`。Tokenizer sha `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`（32169626 字节）。
6. 决策：多数能 4096 则选 4096；多数超 4096 但 8192 覆盖主集合则选 8192（4K 未证明）；主样本超窗则 `trainable_ok=false`，不截断。2048 只作对照。当前五个训练 issue 只支持方法探索。
7. 交付可复算选样清单（每条：issue、trace、tx id、behavior、raw length、split、为何入选）。少量真实片段确认标签与去重即可，不建测试平台。

### 完成标准

- v2 报告含行为计数、恢复候选保留数（对比旧 0/20）、各窗口覆盖、chosen_max_seq_len、trainable_ok、train_issues。
- 旧 v1 文件未改。heldout 未读。8 预留题不在 train/select。

## Task 2: 针对真实长度的容量实验与一个 SFT 候选

仅当 Task 1 完成。若 `trainable_ok=false` 且没有足够完整交易，仍可做容量（用能代表新数据的完整长度），但正式 SFT 候选在没有可训完整样本时停止并记录，不截断、不拿诊断合成 2K 当窗口。

### 容量

1. 启动前重读配额。建议 ≤0.6h 额度。NvidiaL4，`--timeout` 保守。钉训练依赖路径（NF4 缓存 + 7-wheel），不要装 v30 41 wheel。
2. 同一次模型加载中检查能代表新数据的完整长度（按 Task 1 分位数选，可能是 4K/8K/16K）。输入用完整交易，不要再只测稀疏标签合成 2K。
3. 记录每卡峰值 allocated、有效监督 tokens/s、步耗时。独立输出目录，临时更新产物不当正式候选。
4. DDP 放不下时根据峰值判断激活 vs 权重，决定更省内存策略或是否值得 FSDP-QLoRA；不自动连续重试。改变部署协议的上下文构造要另设对照，不能只删训练历史变短。
5. CPU 能编码 ≠ GPU 能反传。

### SFT 候选

容量与数据可用后，只做一个 rank8/alpha16、LR 5e-5、一个 epoch 以内的真实 SFT。`norms_fp32`、`selected_chunked`、长度分桶。步数由独立样本数决定，100 是上限不是凑数目标。一次加载后连续训练。导出 PEFT、数据指纹、监督 token 统计。禁止诊断合成重复文本。明确 loss ≠ 修复增益。

Kernel 建议新 slug，例如 `ccjjllt/p005-capacity-sft-v2`。证据：`research/005/capacity-sft-v2-20261010/`。

### 完成标准

- 容量报告：尝试的长度、是否反传成功、峰值、步耗时。
- 若训练：云端 adapter 指纹 + training_manifest。若未训练：写明数据或容量原因，`gpu_sft=false`。

## Task 3: 八题固定 T0 的 base/adapter 配对

仅当 Task 2 产出真实数据 SFT adapter，且 Task 1 的 8 预留题仍不在 train/select。否则跳过并写原因，不跑空配对，不用诊断 adapter 当收益臂。

### 要做的

1. 启动前重读配额。建议 ≤3.5h 额度。4×L4，一次 vLLM 加载，同一服务上 base 与 adapter。推理 docker 用已验证 3.12 Starter pin，v30 wheelhouse + CT v2。
2. 固定 T0 提示、官方六工具、预算 5min/50/80/60s、`max_output_tokens 16384`。不要用 single_state。
3. 八题：`fastapi_13713`、`requests_6592`、`requests_6644`、`requests_7309`、`requests_7505`、`rich_3130`、`rich_3296`、`rich_3469`。Base `n_adapters=0`；adapter 为 Task 2 导出。
4. 记录每题 resolved、exit、duration、patch 非空、tool_calls、adapter 指纹。汇总新增修复、回退、耗时。八题只决定是否扩大，不外推金牌。
5. 载体从 joint v30 派生，另存 `notebooks/005-base-adapter-pair-single-cell.py`，不覆盖冻结文件。结果：`research/005/base-adapter-pair-v2-20261010/`。
6. 不提交比赛 zip。

### 完成标准

- 配对表完整，train/eval 不相交。
- 结论只来自 resolved 差异。

## 合并审查后的研究安排（2026-10-10）

上文是原执行计划。本轮已完成 v2 抽样、长度普查及一次容量实验；正式 SFT 和八题配对未运行。原始实验数据保留，合并后的探针修复未重新提交 Kaggle。本次审查与归档只使用本地 CPU。

### 当前已证实的事实

- v2 从 214 个交易中选出 23 个：13 个编辑、5 个恢复、5 个验证收尾；恢复候选不再全部丢失，入选 7 个不同的恢复事件。原报告可在 `research/005/length-census-v2-20261010/` 复查。
- train 为 11 条、4 个 issue，完整模板长度中位数 18,729；8K 覆盖 0 条，16K 仅覆盖 4 条、3 个 issue。那 4 条合计 47,514 输入 tokens、945 个监督 tokens。扩大显存也不会自动解决样本规模与代表性问题。
- 实际容量输入是 select 中一个 **6,032-token** 完整交易；四卡每卡加载后 allocated 约 17.21 GiB，首步路径峰值约 21.47 GiB 后 OOM。**16K 未尝试，正式 SFT 未开始，没有 adapter**。不能将提交标题或窗口名称解读为已测满 8K/16K。
- 旧异常分支只记录错误文本，未保存失败栈或前向/反向/优化器阶段。报告中 activation/optimizer 归因尚未被证明；目前只能确定加载成功后首步路径耗尽可用显存。
- 账号最后快照是 2026-10-10 14:10:30 UTC：已用 12,002.806 秒，剩约 26.666h，预留 0。前后账号增量为 1,247.151 秒（0.346h）；wrapper 239.884 秒，二者口径不同，不能只按 worker 时长重算实际扣费。下一次启动前重新读取配额。

### 合并时修正的代码问题

修正引号内测试命令文本的误标、合法文件名被裁剪后错误关联恢复、重叠恢复事件重复选样，以及独立 register 入口对预留题同 repo＋commit 的排除。完整历史、工具 JSON 和监督目标不因这些修正被裁剪。修复后重建历史 214 个交易逐字段一致，23 条选样及 7 个恢复事件不变；76 项本地检查通过（68 项训练/编码/数据、7 项容量载体、1 项配对归档），含真实 tokenizer。

容量脚本现在为 OOM 保存 traceback；正式 SFT/加载失败返回非零，wrapper 只将明确的容量负结果当作完成的探查，不能再把正式训练失败报成成功。当前生成器和 kernel 源一致；新版本尚未远程运行。历史运行源码、bundle 和原始报告可从提交 `c128536` 回溯，后续重新生成载体再推送，避免沿用已删除 worktree 的绝对路径。

### 下一项：定位内存峰值，做一个有明确对照的小实验

优先改动 `training/memory.py` 的 backbone 调用与容量 worker 的阶段记录，不先重写 FSDP。先用微型 Gemma 在 CPU 核对 loss/LoRA 梯度，再为真实模型准备同一 6,032-token 输入的两种设置：原配方与仅对 backbone 保存张量启用 `torch.autograd.graph.save_on_cpu(pin_memory=True)`。NF4、LoRA、标签、注意力实现、CE chunk 和数据全部固定。

CPU offload 是待验证假说：若 60 层各保留一个 BF16 hidden 边界，6,032×5,376×2×60 约为 3.62 GiB；CPU 保存检查点输入可能释放其中一部分，但共享 KV、其他引用、重算临时量和传输开销均要实测，不能承诺节省量或速度。[PyTorch 2.11 save_on_cpu 文档](https://docs.pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.save_on_cpu)和[checkpoint 源码](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/utils/checkpoint.py)是实现参考。

实验需记录 forward、backward、optimizer 的边界、最后执行的 decoder 层、完整 traceback、逐阶段显存峰值及耗时。只保留必要诊断，避免 profiler 本身放大显存。先做一轮有解释力的比较，建议最多 **0.6 个 GPU 额度小时**；若基线 OOM，结束该分布式 worker 后再运行比较臂，不复用处于失败状态的 DDP reducer。

若 offload 成功，继续测试现有完整 train 中 9,841 与 12,832 tokens 的样本，各完成包含首轮 Adam 状态创建的优化步；成功后才能谈这两个真实长度的容量。不要将 12,832 成功写成 16,384 已验证。容量阶段使用过的 select 更新须丢弃，正式 SFT 从干净 adapter 初始化开始。

若仍失败，依 traceback 决定下一项：归一化临时量则研究局部分块/融合；注意力路径则检查实际 SDPA kernel、mask 与 KV；确需权重分片时再评估 FSDP-QLoRA 及缓存格式、通信成本。报错申请 378 MiB 与某些 q_norm FP32 临时量尺寸接近，仅可作为定位线索，不是根因结论。

### 同时推进的数据研究与最终判断

继续扩充独立训练 issue 和真实恢复事件，保留来源、失败观察、成功动作及最终 resolved 标签。研究自然较短、上下文完整的交易；若另做证据摘要或状态化上下文，必须明确它是新的数据表示和部署协议实验，并与完整前缀版本分别比较，不能把裁剪后的 v2 冒充原完整样本。

只有容量与数据都支持时，做一个小规模真实 SFT；现有四条数据最多用于方法探索。产出 adapter 后，沿用预留八题、固定 T0/六工具/预算、一次 vLLM 加载，对比新增修复、回退和耗时。此次八题尚未运行，后续数据扩充仍需排除它们及同 repo＋commit。更大的 SFT 和配对额度在诊断结果返回后分配，不因“GPU 可以跑”直接消耗剩余周额度。

### 本地研究材料保存

旧 worktree 的 34 项非缓存忽略文件已逐项校验并归档到 `training/artifacts/worktree-archives/p005-sampling-sft-c128536-20261010/ignored-files.tar.gz`，旁边 `manifest.json` 保存文件哈希和符号链接信息。20 个训练/容量输入/遥测/日志文件已复制到主目录原相对路径；`results` 链接只保留链接本身，不跟随打包，主目录实际结果不受清理影响。这份本地归档不进 Git。
