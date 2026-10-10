# P005：修正抽样后容量、SFT 与八题配对

日期：2026-10-10。源规格：[下一轮计划合并审查节](005-next-round-sft.md#合并审查后的下一轮工作2026-10-10)。
工作分支：`p005-sampling-sft`。不要改 `main`。

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
