# P007：10月17日官方规则、部署与数据更新

核查截止：2026-10-17 02:30 UTC（北京时间10:30）。对照[10月13日报告](../../006/official-data-20261013/report.md)及P006计划；只读官方页面、公开讨论、模型和文件元数据。未读取hidden/heldout/预留题内容，未批量下载任务、下载权重、调用教师、启动GPU或提交。配额、榜单和正式成绩由主代理独立核查，本报告不重复请求。

## 1. 对下一轮真正有影响的变化

| 事项 | 截至10月17日的证据 | 对研究的影响 |
| --- | --- | --- |
| 四模型规则 | Overview已列出31B/12B/E4B/E2B CT；不再与10月11日公告冲突 | 12B合法性已写入正式概述；一个提交所有agent仍须同一基座 |
| LoRA生效争议 | 10月15日官方确认示例adapter已作用；相同贪心文本不能证明未加载 | 要观察有区分度的动作/分布；单层随机stub不能代替训练后全adapter实验 |
| LoRA显存 | 官方指出入门Notebook的CUDA graph显存估算环境设置；已有rank64成功提交 | 保留已成功的Py3.12 + v31路径；训练能放下不等于部署并发/KV容量足够 |
| 评测总预算 | 新提案仅获“会考虑”答复，没有改变12小时规则 | 继续控制有效修复事务耗时，不能假设未完成题自动记0即可安全结束 |
| 隐藏反馈 | 10月16日官方明确不会公开本赛测试轨迹 | 不等待隐藏trace来补训练集，继续有来源的公开跨仓库研究 |
| 外部教师 | Claude/GPT轨迹帖子尚无官方答复，服务商条款另有约束 | 不把参赛者推测当许可；本轮无需新教师，继续公开来源和真实重放 |
| 软件和数据更新 | wheelhouse仍v31，模型variation、wheel文件列表、公开数据摘要与10月13日完全相同 | 无需以“新包”为由推翻P006成对结果或重建环境 |

“完全相同”指本次API返回的结构化元数据逐字段比较，不是重新下载全部权重、wheel或任务后验hash。源文件见本目录JSON；比较结果见[metadata-comparison.json](metadata-comparison.json)。

## 2. LoRA与运行镜像：新的官方证据

### 2.1 不能从相同文本推断adapter被忽略

[749528，官方回复3539167](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749528#3539167)，Ryan，2026-10-15 13:44 UTC：

- 官方说scorer使用同一patched vLLM 0.19.1 wheelhouse，示例adapter已经在前向过程起作用。
- 示例只作用于第0层；只看很简单、top-1概率接近1的贪心输出，对adapter影响不敏感。
- 这解决“平台一定忽略所有LoRA”的泛化说法，**没有验证本项目尚未完成的12B训练adapter**，也不证明对真实修复有增益。

官方回复包含一段其agent生成的分析，不能全部当作数学证明：从单个top-1 logprob只能得到相对“其余词合计概率”的log odds，不能唯一反推出runner-up margin或某个raw logit改变了1.6。不同批处理设置的贪心逐bit一致性也不能在本项目未测时直接假定。可用结论是“文本相同不足以否定adapter生效”，不是“两个小数已完全证明训练部署正确”。

### 2.2 启动显存问题已有明确操作建议

[749191，官方回复3539170](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749191#3539170)，Ryan，2026-10-15 14:05 UTC：沿用Getting Started与scorer相同配置，关键设置为：

```python
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "1"
```

官方已见过最高rank64的成功LoRA提交。这是成功观察，**不是官方rank上限，也不是任意rank64、多adapter、上下文配置的容量保证**。

同帖参赛者报告了关闭LoRA专用graph capture、降低max_num_seqs等替代办法和KV容量数字；这是各自环境的观察，不替代本项目部署实测。其全层r16超时被推测为KV不足，尚无官方归因。下一次真实训练后的验证应同时回答：adapter是否改变目标动作、是否能完成修复、同预算吞吐和KV容量如何。不能只报告导出/加载成功。

### 2.3 镜像问题要区分host与任务sandbox

[749505，官方回复3539175](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749505#3539175)，Ryan，2026-10-15 14:09 UTC：从Getting Started直接fork，保留原Python3.12镜像。该问题来自参赛者CPU Notebook的Python3.13与cp312 wheel不兼容；官方未建议在线升级任意PyTorch/vLLM解决。

[748287新回复3540037](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287#3540037)在10月16日由普通参赛者报告12B CT v2 + v31在Py3.12 L4本地HTTP推理成功，明确未验证compiler/scorer。它支持镜像匹配的重要性，不能当成12B正式提交已成功的官方证明。另[749259](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749259)的12B正式失败报告尚无官方解释。

模型服务host的Py3.12与任务sandbox的Py3.13是两个环境，不能据前者把官方任务容器改成Py3.12。本项目P006已有v31结果可继续复用，不为这些帖子盲目重跑。

## 3. 软件、模型与协议没有新版本

wheelhouse最新版本仍为**31**，更新时间仍为**2026-10-12 19:51:31 UTC**，元信息总大小880,842,601 bytes。[wheel-info.json](wheel-info.json)、[wheel-files.json](wheel-files.json)及[官方数据集](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)。

本次wheel文件元数据列表与10月13日快照完全相同，因此没有再下载小wheel。10月13日已经查明的v30→v31变化仍适用：swegemma 0.2.11把Phase 2验证超时分离为verification_timeout_seconds，默认300秒；adk_submission 0.2.13归档hash变化但ZIP内容相同。**这是上一轮的变化，不是本轮新修正**。P006固定v31、两臂同300秒的实验不受版本更新干扰；不能把当前正式0.13归因给此变化。

当前四个合法部署路径均为：
`google/gemma-4/Other/<variant>/2`，variant分别为：

- `gemma-4-31b-it-qat-w4a16-ct`
- `gemma-4-12b-it-qat-w4a16-ct`
- `gemma-4-e4b-it-qat-w4a16-ct`
- `gemma-4-e2b-it-qat-w4a16-ct`

同提交所有agent/subagent同一变体；可分配不同LoRA。4×L4和12小时总限制没有变化。[Overview](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview)、[官方公告](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287)、[model-variations.json](model-variations.json)。

整个模型variation返回与10月13日一致，Kaggle 12B未量化QAT目录仍记录总3,128,668,544 bytes，不能据目录名推断已修复为完整BF16存档。本轮未再下载或加载它。P006已固定的Google HF完整QAT元数据来源仍是既有研究依赖；12B真实训练加载、梯度、导出、CT部署效应均不能由本次目录核查代替。

工具仍是官方9工具、agent_tool子agent以及sandbox技能；本项目六工具是研究选择。10月16日[官方3539965](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743573#3539965)确认技能会按ADK机制被发现。[ADK文档](https://adk.dev/skills/#system-instructions-for-using-skills)说明先发现元数据、按需load_skill，资源与脚本分别经load_skill_resource/run_skill_script；不构成放开host任意工具注册的授权。技能脚本仍与run_command共享持久容器和预算。

未找到新的官方thinking或compaction修正公告。[744692](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744692)10月14日的阈值/overflow追问没有官方新答复；[748801](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748801)嵌入生成方法追问亦无官方答复。不能把参赛者的overflow发生率当本项目观测，也不能把文档14336阈值或论坛质疑直接推定为隐藏scorer的实际失分原因。

## 4. 预算、反馈与公开数据边界

[749597，官方3539198](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749597#3539198)，10月15日15:04 UTC：简单把未完成题计0可能造成public/private分数不一致，主办方会考虑其他方法。**没有宣布改为任意全局预算，也未承诺超时部分结果保留。** 当前Evaluation仍为12小时完成所有补丁，计入sandbox准备，不计最终补丁验证；可自设每题限制。

[749953，官方3539810](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749953#3539810)，10月16日15:02 UTC：本赛不会公开测试回放，原因是会泄漏测试内容并可能被主动导出。P006报告的“尚未找到开放”现在可升级为“官方明确不开放本赛测试回放”。

[747519](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747519)中：

- 10月13日的约14小时队列/约24小时完成只是当时估计，不是10月17日服务承诺。
- 10月15日官方3539366说interactive-session提交故障已修复；completed saved version路径此前可工作。
- 10月16日3539786说明原广泛queue故障早已修复，后续KaggleError可能原因不同；3539805提及压缩包多余pycache/pyc可能造成格式错误。
- 早先自动恢复仅针对当时特定队列故障，不能假设今后每次错误都会恢复名额。本轮不对现有提交作归因。

公开数据摘要与10月13日完全相同，仍129题，来自FastAPI/Rich/Requests/HTTPX。官方数据页说明隐藏约120题，public/private约各半，来自私有仓库。10月9日官方已明确四个公开仓库不在隐藏测试中；本轮[747248](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747248)只有参赛者催更，无新官方修订。这里没有下载tasks.jsonl或任何隐藏内容。[当前数据说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)、[data-summary.json](data-summary.json)。

因此P006少量已暴露开发题的4/6→5/6是局部机制证据，不能外推“私有仓库提升16.7个百分点”。跨仓库泛化、正确补丁和单位时间修复数仍需独立证据。没有新公开资料支持把公开gold/依赖问题等同于正式0.13原因。

## 5. 外部数据与教师许可：来源事实与推论分开

Kaggle规则2.6允许公开、对参赛者平等可获得的外部数据，或满足主办方reasonableness标准的来源；2.5/2.8仍要求可复现的获奖代码、训练与推理说明和相应许可。规则文本与10月13日未变。它没有替第三方服务商授予额外权利。[本赛规则](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules)、[pages.json](pages.json)。

[749585](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749585)10月15日提出Claude/GPT修复轨迹训练问题。截至本次读取，只有一条普通参赛者答复，**没有ADMIN官方许可或禁止裁定**。“task-specific所以应当可以”是提问者推测。

本次用一手页面核对以下独立事实，不根据论坛评论制造新门控：

| 来源 | 核实内容 | 本计划可得结论 |
| --- | --- | --- |
| [OpenAI Services Agreement](https://openai.com/policies/services-agreement/) | 当前页标为2026-01-01生效；3.3(e)限制用Output开发竞争模型；定义中的例外为不分发/商用的分类组织模型及OpenAI服务内微调 | 没有给Gemma代码修复轨迹蒸馏一个明示通用例外；不能仅用“专用任务”当许可 |
| [Anthropic Commercial Terms](https://www.anthropic.com/legal/commercial-terms) | 当前页标为2025-06-17生效；D.4限制利用服务构建竞争产品/模型，明示批准除外 | 输出所有权不取消使用限制 |
| [Anthropic训练FAQ](https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model) | 2026-03-16页同时列出非竞争专用分类等可用场景，以及竞争、开放生成、训练target等限制 | FAQ不能被截取成“任何专用coding模型均允许” |
| [SWE-smith数据卡](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/blob/main/README.md) | 当前仍MIT，tool split 24,100行；卡明确轨迹由SWE-agent + Claude3.7 Sonnet产生 | 公开数据授权声明与上游模型生成来源都要保留；不同表示不能相加当独立问题数 |

本轮没有新教师调用，也没有裁定已有SWE-smith公开数据非法。应保留数据集许可、上游代码许可、教师来源和本地重放记录，不能把数据卡MIT自动说成已替所有服务方作法律保证。若以后主动购买教师生成新训练targets，需要按实际服务/合同处理；这与当前已经授权的CPU公开数据研究是不同动作。

**可立即继续的具体路线**：以现有固定buggy实例为起点，扩展不同公开仓库的官方工具真实恢复事务，优先保留“exact编辑成功但行为仍错→依据新观察修正”的正负过程，并测同预算真实修复。只重放来源明确的任务源码动作，不拼接旧未来；local behavior pass与whole issue resolved分别记录。暂不花GPU或教师费用扩大未经协议转换的长轨迹。

本报告未改变既有[用途冻结](../../006/data-replay-20261013/data-usage.json)：click仅train容量样本；pydicom仅select诊断；markupsafe负例和语义续修仍为未分配candidate，allow_parameter_update=false。集合归档不能整体作为训练集。3个repo/局部事务目前远不足以支持大规模SFT或独立泛化收益主张，但可为机制实验和资源论文提供真实例证。

## 6. 截止与奖励没有变化

截至API本次快照为2996队，awards_points=true。队伍数量不等于最终有效参赛数，不能据此算固定金牌分数；金牌与现金门槛也不同。

| 项目 | 官方截止/奖金 | 北京时间 |
| --- | --- | --- |
| 主赛加入/组队合并 | 2026-11-25 23:59 UTC | 11月26日07:59 |
| 主赛最终提交 | 2026-12-02 23:59 UTC | 12月3日07:59 |
| 主赛现金 | 前三名37,000 / 18,000 / 10,000美元 | 按最终规则与私榜定奖 |
| 论文赛最终提交 | 2026-11-12 23:59 UTC | 11月13日07:59 |
| 论文奖 | 最佳论文15,000；新资源10,000；新应用10,000美元 | 与主赛奖项分开 |

[官方时间与奖金](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview)、[论文赛](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)。paper.json与paper-tracks.json同10月13日完全相同：Kaggle Writeup，原创未发表，3,000词上限；Novelty/Quality/Relevance/Verifiability/Clarity等权。主赛参与不是论文赛先决条件；同分按更早提交。资源方向仍须证明新见解、真实性、可复现性与可迁移价值，不能把样本数量或工程适配本身当论文创新。

## 7. 下一轮优先项与证据边界

1. 保留当前v31与已成功镜像；下一次LoRA研究同时记录训练的真实参数更新、CT部署差异、推理容量和修复事务结果。rank64成功官方陈述只减少“平台完全不支持LoRA”的不确定性。
2. 总预算和隐藏反馈均无放宽，继续从公开跨repo真实事务研究正确修复/耗时，不等待隐藏轨迹，不据单一开发集比例估算正式分数。
3. 12B仍是合法候选，模型目录没更新；按既有完整QAT来源和CT模板准备，不重新把Kaggle异常小存档当完整训练权重，也不把12B本地HTTP成功等同正式编译评分成功。
4. 论文赛比主赛早20天截止。围绕已发生的exact-match与语义正确性差异、真实恢复事务、预算下修复能力形成实证，比新增没有结果的工具包装更可支撑研究。

全部Kaggle查询结果保存在本目录对应JSON（包括普通参赛者帖子，已明确区分ADMIN）。[source-index.json](source-index.json)记录来源类别和核查时间；[web-source-notes.json](web-source-notes.json)仅保存外站一手页面的简短事实摘录及链接，不伪称完整原文快照。没有新脚本、测试文件、权重或评测产物。

