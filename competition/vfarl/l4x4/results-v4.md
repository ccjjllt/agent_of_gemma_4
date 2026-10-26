# VFARL 第四轮 L4×4：实际结果

2026-10-26。已重新读取 Kaggle 完成状态和发布后的原始文件。**99 道当前策略真实任务均完成完整验收，6 成功、93 失败；5 个同状态组的 15 条分支全部奖励 0，实际 optimizer step 为 0。** 本轮跑通了采样、声明范围内的状态恢复、分支执行和完整验收，但尚未完成奖励学习更新闭环。

实际作业为 [qingyu1232/vfarl-l4x4-two-gpu-policy-training/1](https://www.kaggle.com/code/qingyu1232/vfarl-l4x4-two-gpu-policy-training)，载体 SHA `06b6a226fd2f4923a1935675294780ac8126de6db5eaf954f5afd0189720be03`。两个采样模型各占两张 L4，4K 上下文、4-bit NF4、BF16、LoRA r32、thinking 上限 1024、每响应上限 1536。全部样本来自冻结 S600 train100/v4；这里的 6/99（6.06%）是初始策略在本训练子集的结果，不能解释为 RL 提升或 heldout 成绩。

## 完成范围与真实修复

| 项目 | 实际结果 |
| --- | ---: |
| 固定训练池 | 100 题 |
| 本轮真实 original rollout | 99 题 |
| 完整验收成功 / 失败 / unknown | 6 / 93 / 0 |
| 因上下文停止 | 86 / 99，其中失败 80、成功 6 |
| 原生工具格式错误停止 | 10 / 99 |
| 模型文本终止 | 3 / 99 |
| 非空 evaluated.patch | 63，其中成功 6、失败 57 |
| 显式提交 / 官方工作区 fallback | 1 / 98 |

DVC7628 在本轮载体冻结时的 CPU 报告中尚不可用，因此记录为未采样，不能把 99 写成 100。其后完整 CPU 对照已通过，当前训练池累计 **100/100 环境可用**，但 DVC 的 GPU 补跑尚未执行。[CPU 来源索引](kaggle/cpu-evidence-index-v4.json)。非空补丁也可能仅含模型复现脚本，不等于 63 次有效生产修复。

下列六题均已逐一核对生产源码补丁、原始 `original-result.json` 和 JUnit：`resolved=true`、完整验收、exit 0、0 skipped。附带复现脚本是模型真实运行产物，未用参考修复代替模型行为。

| 任务 | 生产修复 | 原始验收 | 补丁获取 | 原件 |
| --- | --- | --- | --- | --- |
| `Suor__funcy.207a7810.func_basic__4zw03ae8` | without: value in items 改为 value not in items | 203 passed，0 skipped | 显式 submit | [补丁](kaggle/train-evidence-v4/vfarl-train/worker-0/Suor__funcy.207a7810.func_basic__4zw03ae8/draw-0/original/evaluated.patch) · [验收](kaggle/train-evidence-v4/vfarl-train/worker-0/Suor__funcy.207a7810.func_basic__4zw03ae8/draw-0/original-result.json) · [JUnit](kaggle/train-evidence-v4/vfarl-train/worker-0/Suor__funcy.207a7810.func_basic__4zw03ae8/draw-0/original-1-_swegemma_junit_9f1a282b53a4.xml) |
| `Suor__funcy.207a7810.func_basic__fqlqaceh` | cut_suffix: 修正后缀切片并处理空后缀 | 203 passed，0 skipped | 工作区 fallback | [补丁](kaggle/train-evidence-v4/vfarl-train/worker-0/Suor__funcy.207a7810.func_basic__fqlqaceh/draw-0/original/evaluated.patch) · [验收](kaggle/train-evidence-v4/vfarl-train/worker-0/Suor__funcy.207a7810.func_basic__fqlqaceh/draw-0/original-result.json) · [JUnit](kaggle/train-evidence-v4/vfarl-train/worker-0/Suor__funcy.207a7810.func_basic__fqlqaceh/draw-0/original-1-_swegemma_junit_c48d0c2c5150.xml) |
| `kayak__pypika.1c9646f0.func_basic__wqdn07ef` | Term.eq: != 改为 == | 1061 passed，0 skipped | 工作区 fallback | [补丁](kaggle/train-evidence-v4/vfarl-train/worker-1/kayak__pypika.1c9646f0.func_basic__wqdn07ef/draw-0/original/evaluated.patch) · [验收](kaggle/train-evidence-v4/vfarl-train/worker-1/kayak__pypika.1c9646f0.func_basic__wqdn07ef/draw-0/original-result.json) · [JUnit](kaggle/train-evidence-v4/vfarl-train/worker-1/kayak__pypika.1c9646f0.func_basic__wqdn07ef/draw-0/original-1-_swegemma_junit_702c215fd44e.xml) |
| `Suor__funcy.207a7810.func_basic__dto91avw` | re_finder: match 改 search，未匹配返回 None | 203 passed，0 skipped | 工作区 fallback | [补丁](kaggle/train-evidence-v4/vfarl-train/worker-1/Suor__funcy.207a7810.func_basic__dto91avw/draw-0/original/evaluated.patch) · [验收](kaggle/train-evidence-v4/vfarl-train/worker-1/Suor__funcy.207a7810.func_basic__dto91avw/draw-0/original-result.json) · [JUnit](kaggle/train-evidence-v4/vfarl-train/worker-1/Suor__funcy.207a7810.func_basic__dto91avw/draw-0/original-1-_swegemma_junit_f1ff60bd7c3d.xml) |
| `pytest-dev__iniconfig.16793ead.func_basic__vb9u5hga` | ParseError.__str__: lineno - 1 改为 lineno + 1 | 42 passed，0 skipped | 工作区 fallback | [补丁](kaggle/train-evidence-v4/vfarl-train/worker-1/pytest-dev__iniconfig.16793ead.func_basic__vb9u5hga/draw-0/original/evaluated.patch) · [验收](kaggle/train-evidence-v4/vfarl-train/worker-1/pytest-dev__iniconfig.16793ead.func_basic__vb9u5hga/draw-0/original-result.json) · [JUnit](kaggle/train-evidence-v4/vfarl-train/worker-1/pytest-dev__iniconfig.16793ead.func_basic__vb9u5hga/draw-0/original-1-_swegemma_junit_d3b81b2d5773.xml) |
| `Suor__funcy.207a7810.func_basic__zkdyaxut` | cut_prefix: s[:len(prefix)] 改为 s[len(prefix):] | 203 passed，0 skipped | 工作区 fallback | [补丁](kaggle/train-evidence-v4/vfarl-train/worker-1/Suor__funcy.207a7810.func_basic__zkdyaxut/draw-0/original/evaluated.patch) · [验收](kaggle/train-evidence-v4/vfarl-train/worker-1/Suor__funcy.207a7810.func_basic__zkdyaxut/draw-0/original-result.json) · [JUnit](kaggle/train-evidence-v4/vfarl-train/worker-1/Suor__funcy.207a7810.func_basic__zkdyaxut/draw-0/original-1-_swegemma_junit_ead8b40ea973.xml) |

五次成功没有调用 submit_patch，而是由官方未提交工作区 fallback 捕获真实修改；这一行为避免把已完成的生产修复误记为空补丁。六题全部来自初始零 B LoRA 策略，本轮未更新参数。

## 上下文与动作预算

完整日志共有 **522 条 action 事件：517 次真实新生成、5 次原动作强制重放**。新生成包括 original 的 459 次动作和分支中的 58 次动作；新生成 completion 共 77,857 tokens，其中真实 thought 46,349。original 部分为 69,358 completion / 41,480 thought tokens。thinking 1024 是上限：全轮仅一次动作达到该上限，original 平均每动作约 90.37 thought tokens。

99 条 original 最终历史逐题计一次，累计 429,353 tokens：初始 prompt 135,487（31.56%）、助手输出 69,358（16.15%）、序列化工具回复 224,508（52.29%）。工具部分按“最终历史 − 初始 prompt − 全部实际生成 completion”求得。这是终态历史的组成，不是重复处理的总 input tokens、API 计费量或 GPU token 吞吐。

模型实际调用的最大 prefix 为 4069 tokens，prefix＋completion 最大 4096；末次工具输出可把保存历史推到 7169 后直接停止，**并没有让模型在 >4K 的历史上继续推理**。所有 original 的输出预算最大仅 1684，远未达到 episode 8192 输出上限，86 个 context 类终止确属联合上下文限制。最高单卡 allocated 为 14,175,890,944 bytes（13.20 GiB），本轮没有 collection_error 或 OOM。

十个格式错误均来自真实 read_file 动作把 native 参数写成混合 JSON 形式，例如 `start_line":151`；不能归入上下文失败，也不能自动补写成另一条动作。下一轮可让模型看到真实协议错误反馈并尝试恢复，需保留原始无效动作。

## 五个状态实际都是初始状态：P0 选择缺陷

15 条分支均完整验证，包含 5 条原动作重放和 10 条独立当前策略采样。每组奖励均为 `[0,0,0]`，同状态 sampled 优势均为 0，`complete_declared_scope` 的文件系统及运行时恢复记录通过。

但逐个核对 `branch-1/restore.json` 后发现：**全部五个恢复账本都是 `turns=0, output_tokens=0`，完整 token 前缀与 original 的首动作前缀逐项相同。** 长度分别为 1424、1420、1342、1387、1443。它们是“最终失败 episode 的初始状态再采样”，尚未证明“在已经观察到真实失败之后的关键状态恢复”。

flake8 w0t 组虽标 `checkpoint_observed_tool_error=true`，该字段描述的是保存快照之后执行的动作结果，不能说明恢复前缀已见到该错误。当前排序偏向短前缀，使五组都选择了 pre-turn-0；应改进状态选择和字段语义，再取得真正含失败观察及剩余恢复预算的状态。真实恢复数据保留，不能重命名为失败后恢复收益。

## 概率数值差异：P0 尚待校准

本轮 10 条独立 sampled focal action 共 2283 个 loss-mask=1 token。实际行为 old 与独立 reference 的绝对 logprob 差如下：

| MAE | P95 | P99 | 最大差 | >0.08 token | action 最大差 >0.08 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00386141 | 0.0154423 | 0.1055289 | 0.4105685 | 34 / 2283 | 10 / 10 |

初始零 B 在数学上等价于基座，但实际自回归、分块参考和训练前向的计算路径不能因此视为数值严格一致。`exp(ref−old)` 只是代理，不能替代尚未实测的 FSDP `exp(current−old)`。本轮训练入口因零异奖组提前返回，**没有执行 0.08 对齐检查，也不是被该阈值卡住**。不应重写 old logprob 或直接增大容差以凑出更新；真实 actor/FSDP 零更新校准尚未完成。

## 参数更新与结论边界

[update.log](kaggle/train-evidence-v4/vfarl-train/update.log) 中四个进程均返回 `optimizer_steps=0`、`mixed_group_count=0`；[carrier.json](kaggle/train-evidence-v4/vfarl-train/carrier.json) 记录训练进程 exit 0、`one_update_completed=false`。没有导出新 adapter，也没有更新后 rollout。实际 31B FSDP 梯度、KL 约束效果、LoRA 参数更新及奖励一致的学习方向均尚未验证。

已有的机制证据是：双卡 NF4/r32 采样能够执行真实 thinking 与六工具任务，得到完整验收和六个生产修复；声明范围内快照可恢复并运行分支。当前主要研究缺口是 4K 工具历史容量、pre-turn-0 状态选择，以及 actor/learner 概率执行差异；五组同奖不能证明 VFARL 无效，也不能算有效学习信号。

8K 对照、DVC GPU 补题和真实 CUDA 数值校准均未执行。后续应使用既定训练白名单和冻结比较规则，保留本轮全部同奖及失败数据，取得真实异奖组后才完成一次实际更新。

## 资源与来源

本轮载体 wall time 为 **16,669.05 秒（4 小时 37 分 49 秒）**。qingyu1232 的平台配额本轮增加 **9.26 小时**，累计已用 **10.78 小时**，剩余 **19.22 小时**；采用平台读数，不按四张卡自行换算。[完成状态与配额原件](kaggle/completed-v4-readback.json)。

结果由发布后原件重新汇总，实时日志在 56 题后失联的截断不再作为最终计数。下载期间的 179 个 SSL EOF 已重取恢复，当前 [下载 manifest](kaggle/train-evidence-v4/download-manifest.json) 无错误；没有把下载异常误记为任务 unknown。结构化汇总及原件 SHA 在 [result-summary.json](kaggle/train-evidence-v4/result-summary.json)。

主要依据：[worker-0 collection](kaggle/train-evidence-v4/vfarl-train/worker-0/collection.jsonl)、[worker-1 collection](kaggle/train-evidence-v4/vfarl-train/worker-1/collection.jsonl)、[focal.jsonl](kaggle/train-evidence-v4/vfarl-train/focal.jsonl)、[worker-0 状态组](kaggle/train-evidence-v4/vfarl-train/worker-0/state-results.jsonl)、[worker-1 状态组](kaggle/train-evidence-v4/vfarl-train/worker-1/state-results.jsonl)、[worker-0 原始日志](kaggle/train-evidence-v4/vfarl-train/worker-0.log)、[worker-1 原始日志](kaggle/train-evidence-v4/vfarl-train/worker-1.log)。


本地后续修正：collector 已禁止在没有真实工具响应的初始前缀采焦点状态，并将“恢复前缀已经包含的工具错误”与“候选动作执行后才出现的错误”分开记录。保存的账本增加工具观察计数和最后一次真实反馈状态，恢复后继续保留这些字段；源码 SHA `6df83ae1bcfaf3e8760da12d0aed91e27ede34ee6fe1ad1ca5b2ecda066fd5ff`。一次性原始证据核对确认本轮五个 turn0 起点均会被新规则排除。此修正尚未 GPU 复跑，不计入本轮结果，也尚未证明新状态选择产生非零优势。
