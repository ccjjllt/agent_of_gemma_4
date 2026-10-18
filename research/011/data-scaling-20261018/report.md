# P011：从 8 题探针转为 139 题连续训练输入

2026-10-18。此目录实际完成 CPU 物化和正式 12B CT 模板编码；没有 GPU、付费教师、完整仓库测试或比赛评测调用。P009 v1/v2 文件保持原指纹。

## 已冻结、可直接使用的训练输入

主文件：`encoded-train-12b.jsonl`，SHA256 `ebfa8d1357c413e467fc71295c371f178f08932b10fb179b95a2014b1c8638b0`。契约和来源分层在 `dataset-manifest.json`，逐来源文件指纹在 `admitted-source-manifest.json`。此后不得覆盖该版本；生产入口已拒绝对冻结目录再次物化。

| 数据层 | 不同问题 | 仓库 | 输入 tokens | 监督 tokens | 已证明范围 |
|---|---:|---:|---:|---:|---|
| 新增局部编辑事务 | 128 | 57 | 404,880 | 93,323 | 固定 buggy 生产文件上真实 read/edit/submit，修改后语法可解析、提交补丁非空 |
| 原 P009 v1 + v2 短事务 | 11 | 7 | 47,094 | 8,500 | 原有真实局部 API/静态类型修复前后证据，保留其原验证范围 |
| 合并去重 | **139** | **58** | **451,974** | **101,823** | 每个 repo/PR 仅一条，不把局部编辑记为整题成功 |

新 128 独立文件为 `encoded-new128-12b.jsonl`，SHA256 `e4e1256f296016cccad1b99e05fb9e4c57f9aa801b2ad32f85465e68ebb24de7`。所有新条目 `behavior_verified=false`、`task_resolved=null`；旧 11 条同样没有整题 resolved 宣称。

完整会话长度：最短 1,009、中位 2,847、P90 5,929、最长 **8,080** tokens；128/139 条不超过 6,144，139/139 不超过 8,192。没有截断、裁掉失败上下文、截短工具 JSON 或静默丢弃动作。P009 v2 的 10,310-token mypy PR12943 整条未混入；旧三条短事务按原始已编码行复制。相同 tokenizer/template/tools/protocol 指纹见 manifest。

在 global batch 4、2 epochs 的现有连续训练方案下，139 个样本对应每 epoch 35 个更新、共 70 个更新。这里仅提供数据输入，是否训练完成由独立 GPU 实验记录证明。

## 从哪里获得足够多的不同问题

使用 SWE-bench/SWE-smith-trajectories 固定 revision `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`，只取 `tool` 的八个分片。XML/ticks/train 是其他表示，不作为额外独立问题相加。官方文件列表原件见 `hf-files.json`，八个分片共 **1,077,271,863 bytes**，文件 SHA256 全部对应 HF LFS 元数据；其中原有 77,754,558-byte 分片复用缓存。

官方来源：[固定版本目录](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/tree/08e109b4a59eaeebf80e4675cd125d42e7ac99a4/data)。许可证数据卡保存为 `dataset-README.pinned.md`。`shard-manifest.json` 是实际下载文件指纹；大 parquet 是本地缓存，不提交 Git。

首次统计仅投影 `file_row_number/instance_id/resolved/model/traj_id`，没有读取保留仓库的 messages/patch。全部统计在 `metadata-census-full.json`：

| 范围 | 来源行 | 独立问题 | 仓库 |
|---|---:|---:|---:|
| tool 全分片全部类型 | 24,100 | 10,637 | 129 |
| 真实 PR、全部 resolved 标签 | 8,802 | 2,127 | 88 |
| 非保留仓库、来源 resolved=true 的真实 PR | 2,457 | 849 | 71 |
| 上项中原来分配的 8 个 train 仓库 | 497 | 159 | 8 |
| 排除此前已读问题后的新候选 | 2,374 | **836** | **70** |

原先单分片、原 8 个 train 仓库只有 39 个可用独立 PR，去除已读仅 26，无法提供 64/128 个新问题。单分片放开非保留 repo 也只有 168 个独立 PR、155 个未读候选，不能承诺转换后有 128，且不可能达 512。全量源解决了候选数量不足：128 已实际交付；512 在原始候选层面有空间，但尚不能把 836 候选宣称为 836 条有效或已验证训练数据。

`data-usage-full.json` 在读取正文前冻结了全量候选次序及 70 个 train 仓库归属。整仓排除原 external-select 八仓库、既有 select/heldout、公开奖励评测四仓库，以及 pydicom、click、MarkupSafe。具体 15 个排除 repo 及此前 13 个已读问题见该清单；未来 select 替补必须跳过新增 train 仓库。只对清单中明确分配 train 的问题读取正文。

## 实际物化方法及证据边界

`materialize.py` 是数据生产入口，不是测试/门控平台。流程为：元数据按 repo 内固定哈希排序、repo 名称轮转；按精确 source file/row/instance/traj 原子取回已分配 train 的 messages；解析 JSON 工具调用和 XML 参数中的真实 `str_replace`；解析对应 SWE-smith instance 分支到完整 40 位 buggy commit；只下载被编辑的生产 Python 文件及仓库 LICENSE；在临时目录建立该真实源码基线。

随后加载已归档的官方 v31 `read_file/edit_file/submit_patch` 原函数，通过 macOS 本地 transport 实际执行。每个 edit 前的新 read 必须含完整 exact old_string，不能靠未展示上下文猜编辑。source editor 参数只构成要执行的动作，响应全部新产生；旧 assistant 推理、旧工具观察和旧未来对话均未拼接。修改后只做必要的 Python 语法解析，实际 `submit_patch` 返回非空补丁才保留。最终 diff、前后源码 SHA、工具事件、命令和许可证均在每个 `outputs/<instance>/` 下。

监督目标为实际成功的 **edit_file + submit_patch**；read、工具结果和角色头不监督。正式 encoder 用完整模板 render、差分 span 和真实 token offset 对齐；没有用字符长度充当 tokens。

这是有答案定位的编辑监督：read 行号来自来源编辑位置，不证明模型学会自主检索或定位。源 shell、create/insert、仓库测试和未知安装脚本不执行；因此新事务明确是生产文件局部编辑子集，user 中也限定本轮编辑文件范围。即使所有选定 replacement 都成功，也不能宣称完整原 trajectory 重放、原 issue 修复或官方评测通过。修改后的 Python 可解析不能证明逻辑正确；来源 `resolved=true` 单独保留但未被采信为本轮行为验证。

数据卡 MIT 与代码许可证分别保存。例如 yamllint 是 GPL-3.0，不能把数据卡 MIT 当作所有仓库代码均为 MIT；原许可证随源码指纹保留。此轮不调用外部教师，也不把来源模型名改成本地生成模型。

## 真实吞吐、丢弃原因与固定入选次序

最终生产批按元数据固定顺序处理前 **210** 候选，8 个 CPU 网络 worker，实际 wall time **75.58 秒**。149 条真实局部应用成功；完整 12B 编码后 132 条不超过 8,192 tokens，17 条超长整条排除。按冻结元数据顺序取前 128，达到目标的终点是第 **201** 个候选 `spulec__freezegun.5f171db0.pr_542`；没有按线程完成顺序、文件遍历顺序或人工挑高成功率题目入选。

剩余 61 个未成为成功事务的原因：41 个完整 edit 上下文未能在当前源码/read 中精确展示；8 个固定源文件 HTTP404；6 个没有可重放的结构化生产 replacement；3 个在限定许可证文件名中未找到原件；1 个原动作超过 20；1 个实际 edit 失败；1 个官方 submit 空/失败。它们保留在 `production-results.json` 和相应实际事件中。超过 20 的限制是首批生产范围，未把部分动作剪成完整修复样本。

最初 `materialize-first.log` 是生产器工具状态字段处理错误的开发诊断，不能当成来源模型失败率。最终统计只使用修正后的 `p011-local-2`、`materialize-batch.log` 和 `production-results.json`。短复核已确认 yamllint582/gunicorn3003 的真实 read 覆盖 old_string、submit 被监督、无答案预注入或旧 future 拼接；没有因此扩建测试套件。

## 发现的源数据错配

必须禁用 parquet 的 `patch` 列。单条 SQL 原子取回 `file_row_number,instance_id,messages,patch,traj_id` 已确认：tool00004 的 yamllint PR582 messages 是该 YAML directive 问题，patch 文件头却是 sqlfluff；pydantic PR7736 messages 是 defer_build/TypeAdapter，patch 文件头却是 moto。不是分别查询后 zip 或并发读取错配。最小证据及原文件 SHA 在 `source-patch-mismatch-evidence.json`。

本批不依赖该列，也不把错仓库 patch 做自动应用。新数据只来自同一身份的 messages 编辑参数与固定 buggy 实际源码，真实失败保留。此前已观察到来源 resolved 标签也可能伴随未真正完成的源轨迹，故继续区分“来源声称成功”“本次 edit 应用成功”“本次行为验证成功”三个层次。

## 下一批的可执行扩量入口

下载、物化、编码分别由 `download_shards.py`、`materialize.py`、`assemble.py` 完成；这版被冻结，继续扩量应新建版本目录并复用固定缓存/分配清单，而非重跑覆盖本目录。当前 210 候选得 132 条完整短事务，转换率 62.86%；将该比例直接外推到全部 836 候选约 525 条只是粗估，不是已交付 512。后续优先沿同一冻结顺序处理剩余候选，记录真实转化；不要为了达到 512 将切片、重复轨迹或长事务截断计成新问题。

若扩量转换不足，下一项有价值的工作是支持来源确实可恢复的多种编辑形式或更长上下文的独立批次，而非让每个问题先手写一套测试。当前 139 题已经可用于连续训练，研究收益仍必须由独立推理/比赛评测结果判断。
