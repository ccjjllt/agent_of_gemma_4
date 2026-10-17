# 2026-10-17 公开高分源码增量研究

本轮已找到并核实 **0.22 V10 与 0.20 V1 的完整公开源码**。0.22 来源可作为新的公开强参考，但我们尚未重跑，更没有证明其泛化提升。0.20 的四个 agent 文件与此前保存的 0.18 flex V9 **逐字节相同**，所以不能把这 0.02 解释成机制进步。

只读检索 Kaggle `ScoreDescending`、`DateRun` 各第一页 40 条，读取八份新建/更新 Notebook 完整源码，用浏览器官方评分徽章核对七份版本。未运行外部代码、GPU、训练或 submission；未下载任务内容、参考 patch 或预留题。前轮背景见 [10/13 报告](../../006/high-score-notebooks-20261013/report.md)。

## 1. 分数必须绑定版本

| 来源 | 本轮当前源码 | 已核实分数及绑定 | 与 10/13 相比的新信息 |
|---|---|---|---|
| [flexonafft Budget-Aware](https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-v2?scriptVersionId=357955943) | V10 / **357955943** | 当前 **0.22**；Best 0.22 V10 | V9 当时 0.18；V10 修改导航与编辑恢复提示，配置三文件不变 |
| [Jvnhonglei SuperAgent Reproduction](https://www.kaggle.com/code/jvnhonglei/gemma-4-superagent-reproduction?scriptVersionId=358224038) | V1 / **358224038** | 当前/Best **0.20 V1** | 四文件与旧 flex V9 完全相同，是同方案另一公开观测 |
| [Bounded Rescue](https://www.kaggle.com/code/lynnsakurai/gemma-4-budget-fit-agent-w-bounded-rescue?scriptVersionId=358677131) | V3 / **358677131** | 当前 **0.15**；Best **0.18 属于 V2 / 358320692** | 相比 0.20 源码只多一句精确匹配失败后重读重试；不是已证实收益 |
| [Dmitrii Public Ideas](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-public-agent-ideas-and-evidence?scriptVersionId=359075575) | V2 / **359075575** | Best **0.18 V1**；未见 V2 当前得分 | 新 31 题配对研究；V2 把采样温度改为 0.4，其 31 题研究仍是 0.2 |
| [Single-Agent Tuning](https://www.kaggle.com/code/lambdaderta/single-agent-tuning?scriptVersionId=359129363) | V3 / **359129363** | Best **0.17 属于 V2 / 358766411**；未见 V3 当前得分 | 当前六工具、实时预算、行为验证、温度 1.0、48 calls；不能继承 0.17，更不是其 manifest 中的目标 0.20 |
| [Hong 的 LB 0.24 标题](https://www.kaggle.com/code/honghanhhh/gemma-4-baseline-lb-0-24?scriptVersionId=358348855) | V5 / **358348855** | Best **0.18 属于 V3 / 355856877**；未见 V5 当前得分 | 当前沿用 0.22 提示并加保护路径、并行实现与未跟踪文件检查；标题 0.24 不是分数证据 |
| [Offline QLoRA](https://www.kaggle.com/code/denpugovkin/gemma-4-offline-qlora-train-verify-submit?scriptVersionId=358921141) | V4 / **358921141** | 当前/Best **0.03 V4** | 真实训练与产物；运行 1310.6 秒成功，仍无 base/adapter 同配方修复对照 |

以上页面实际 licence 均为 Apache 2.0。六个分数/最佳版本不能互相继承。`get_notebook_info.last_run_time` 与列表日期冲突，本报告以检索日期、版本号、`scriptVersionId` 和徽章为身份依据。源代码中“尚未评分”的旧文字与当前 0.22、0.03 徽章冲突时，以精确版本徽章为准。当前检索没有取得 **0.24+ 的可核实完整源码**，这不代表全榜不存在更高方案。

证据：[浏览器徽章记录](raw/browser-score-evidence.json)、[检索清单和源文件 SHA](retrieval-manifest.json)、[高分列表原件](raw/search-score.json)、[最新列表原件](raw/search-latest.json)。

## 2. 0.22 到底改了什么

四文件的语义内容用 Python AST 读取字面量，没有执行 Notebook。

| 对比 | agent.yaml | sampling.yaml | eval_config.yaml | system.md |
|---|---|---|---|---|
| BudgetFit V5 → Jvn 0.20 V1 | 相同 | 相同 | 相同 | 不同：明确 issue 规格、分阶段行动、近邻模式、最终审阅 |
| flex 0.18 V9 → Jvn 0.20 V1 | 相同 | 相同 | 相同 | **相同** |
| flex V9 → flex 0.22 V10 | 相同 | 相同 | 相同 | 仅导航与编辑恢复两节有改动 |
| Jvn V1 → Rescue V3 0.15 | 相同 | 相同 | 相同 | 只增加一条 anchor 失败恢复规则 |

V10 仍为单 agent、官方 31B CT；声明 `run_command/read_file/edit_file/write_file/get_status/submit_patch` 加三图工具，但提示明确禁止使用图工具。温度 **0.2**、top-p **0.95**、输出上限 **8192**、thinking **4096**、include_thoughts=true；**8 分钟/28 counted calls/80 turns/240 秒命令超时**。没有技能、子 agent 或 adapter；Notebook 自身仅 CPU 打包，CPU 运行成本不能代替评分耗时。

相对 V9，V10 的实际变化：

1. 把已知输入的独立搜索合并为一个 `run_command`；依赖上一步结果的搜索仍顺序执行，每项约 20 行输出。
2. 找到目标后同时读实现与一个 caller/既有 test；每区约 40 行、每命令最多三个区段，复用已经找到的路径。
3. `edit_file` 使用当前文本的唯一短 anchor；失败先重读，不原样重试；两次匹配失败后允许带唯一性断言的 Python heredoc 替换。

其 call 12 前编辑、call 20 后停止探索、最多两次验证、8 分钟墙钟均未变。不能把它简化为“只增加错误恢复”；0.22 是多个 prompt 改动加一次公开评测的联合观测。其强制 best-guess edit 与最多两次验证也可能在歧义任务上产生错误补丁，需观测机制而非照单全收。

逐字 diff：[file-comparisons.json](extracted/file-comparisons.json)。0.20 与旧 V9 的身份哈希：[public020-v1-identity.json](extracted/public020-v1-identity.json)。

已保存 [public022_v10 原样候选](../candidates/public022_v10/README.md)，`agent/` 内恰好原四文件，来源与许可在外部 README/manifest。保留原 **8 分钟**，没有打包、运行或提交；prompt SHA256 为 `7f8dde62578a247ffbfc1a519b712ae65849fcfae1ee70c49e0628cab74cbef6`。0.20 不重复建候选。**任何缩成 5 分钟的派生版均不能继承 0.22。**

## 3. 新增反证：31 题配对没有分出提示或技能优势

[Public Ideas V2](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-public-agent-ideas-and-evidence?scriptVersionId=359075575) 报告四配置各跑同 31 个开发题，共 124 outcomes，seed=1，四仓库此前未在该项目评测过；空 patch 0/31、参考 patch 31/31。读取的是汇总与 [frozen protocol](raw/fresh_screen_protocol.json)，没有读取任务内容或参考 patch。

协议实际固定 31B CT、32k context、温度 0.2、top-p .95、8192/4096 token、28 calls、8 分钟、80 turns、240 秒命令超时。软件 Python3.12.14/vLLM0.19.1/adk_submission0.2.13/swegemma0.2.11。单 A100、TP1、八并发，**不是四 L4 串行 scorer**。耗时包括 agent/setup/teardown，不含 grading 与共享模型启动。

| 配置 | 修复数 | 平均秒 | P95秒 | agent timeout | 其他错误 | 120×mean 小时（只作算术） |
|---|---:|---:|---:|---:|---:|---:|
| BudgetFit | 9/31 | 349.0 | 489.3 | 6 | 0 | 11.63 |
| BudgetAware | 9/31 | 359.6 | 489.8 | 6 | 1 | 11.99 |
| BudgetFit + edit fallback/file shortlist | 8/31 | 373.7 | 489.6 | 9 | 0 | 12.46 |
| BudgetAware + 两工具 | 8/31 | 390.5 | 491.7 | 13 | 0 | 13.02 |

BudgetAware 对 BudgetFit 独胜/独败 **2/2**；两个工具臂分别 **1/2、2/3**，exact two-sided sign test 均 p=1。只有一次 seed，且按四仓库聚类/多重比较未调整，不能据此证明等价；但也没有证据宣布 0.22 提示在新任务上胜出。第二 seed 与独立确认均未执行，候选选择已使用这批开发结果。

工具实际使用尤为重要：BudgetFit 工具臂 8 题加载 shortlist，全部先传错根路径，再成功重试，共 8 错+8 成功；BudgetAware 工具臂两个工具均未使用；edit fallback 没有被调用。因此不能说“fallback 机制已证伪”，能说的是**额外声明/加载的工具没有形成收益，且路径错误与超时增加**。此时继续扩技能框架的优先级应低于已知有效的直接工具调用。

原文还列出同 coder–analyzer archive 的 0.12（原作者报告）、0.10、0.06（该项目运行），混有日期和 scorer 条件变化。结合本轮逐字相同的 0.18/0.20，足够否定“单次多 0.01–0.02 即为机制胜利”，但不能用这些数估纯采样噪声分布。

## 4. 其他新源码给出的边界

**训练确实完成，不等于训练有效。** [QLoRA V4](https://www.kaggle.com/code/denpugovkin/gemma-4-offline-qlora-train-verify-submit?scriptVersionId=358921141) 的 [实际汇总输出](raw/qlora_result.json) 记录 56 训练/12 loss 评估，r8/alpha8、1 epoch、seq1024、batch1、accum4、LR8e-5；训练 804.7 秒、T4 双卡峰值 9.34/11.91GiB，评估 loss 0.8420→0.6128，保存 820 个 attention/MLP LoRA 矩阵。训练目标是 source-only unified diff，非完整工具闭环轨迹；训练 base 是 Unsloth bnb4bit31B，而评分声明官方 QAT31B。实际版本公榜 **0.03**；纯 schema/tensor/ZIP 审计与 badge 都不能单独证明 adapter 在评分推理中实际生效。又因 agent 为 3 分钟/40 calls/.15/8192/3072，不能把低分单因果归给 LoRA。此方案提供可运行训练工程参考，**不提供修复收益证据**。

**自适应验证提示值得研究，但还没分数支持。** [Tuning V3 源码](notebooks/lambdaderta__single-agent-tuning.txt) 要求公共接口的可证伪行为检查、验证运行的实际 import 路径、零效果 probe 不能作为证据、约 60 秒/4 calls 收尾预留，并显式写出六工具参数。它同时把温度 .2→1.0、calls28→48、取消固定阶段和两次检查上限，属于复合新候选；历史 0.17 属于 V2。更长预算与更高随机性暂不应作为已证实推荐。

**审计日志不等于 agent 修复机制。** [Chronos V2 源码](notebooks/segamega__chronos-completion-authority-separated-agent-harn.txt) 的 claim/proof、hash-chain、可选 gate 都在外层本地评测，最后 ZIP 仍从 `sample_submission` agent 目录打包，外层日志/裁判 gate 不进入 agent；默认 gate=False。没有展示其优于 starter 的修复对照，不值得移植这种工程层来争取比赛收益。

## 5. 三个优先实现/否决结论

1. **采用原样 0.22 V10 作为新的公开强参考，而非立刻叠加改造。** 已冻结四文件，下一次授权实验应保留原 8 分钟完整配方，和我们冻结方案比较同预算总体成本及修复率；重点记录首个有效生产编辑、失败 payload/anchor 恢复、验证耗时与收尾。这个参考是来源分数可核实、代码可复现；本项目复现和新题泛化仍未完成。
2. **下一项具体机制应是有区分力的行为证据配合及时生产编辑，并将导航合并/编辑恢复拆开归因。** V10 合并搜索与受保护 heredoc 有清楚可迁移实现；31题平局、Rescue0.15、我们已有失败类型都不支持“多一段恢复提示便保证涨分”。不做多agent/技能扩张，也不把高温48calls方案当成已胜出方案。
3. **否决用训练 loss、产物存在或审计数量作为研究胜利。** 新 0.03 QLoRA 是直接反例。训练路线应先证明官方 serving 真实使用 adapter，再在原样强参考与同时间预算上比较行为和修复；这次只保存公开原件，未发起训练或新评测。
