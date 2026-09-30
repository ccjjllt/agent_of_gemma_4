# 运行与可提交能力审查（2026-09-30）

本审查仅做源码、官方资料和 CPU 合同探针；未运行 GPU、提交比赛、修改 Agent 或实验载体。目的是给后续实现交接，避免把无法提交的本地改造或未生效参数当作研究收益。

主审随后恢复全部面板轨迹，确认一次上下文超限及1437步输出统计，已将最大输出16384→8192的单变量容量对照提前到技能实验之前。最终执行顺序以[统一交接](../../../docs/plan/002-next-stage-handoff.md)为准，本文后面的E1/E2/E3保留各实验的技术范围。

## 结论

下一步仍应研究定位，但 P002 应定义为“可提交的定位能力改进”，而不预设官方 embedding/图工具会赢。优先构建可信评估门槛，再比较结构化文本/符号检索与按需图扩展。允许的技能脚本是值得利用的确定性能力；仅继续增加行为提示词的证据不足。P003 的 5 分钟是控制变量，不能误当比赛规定；资源配置应在定位策略稳定后单独研究。

不能把修主机 Harness、注册自定义工具、Python callback 硬停止、替换自动压缩器等本地改造当作可提交方案。技能脚本、声明式串行子 Agent、输出长度与思考开关则有正式提交入口，但仍须验证端到端实际行为。

## 资料与新旧版本边界

- 9 月 30 日通过 Kaggle MCP 重新读取 wheelhouse 完整文件清单和官方 Starter；Starter 仍为 v2，挂载固定模型 v2，源码归档在 [official-starter-source.txt](official-starter-source.txt)，元数据见 [official-notebook-metadata.json](official-notebook-metadata.json)。
- 实际重新下载公开 `adk_submission 0.2.11`、`google_adk 1.36.1`、`swegemma 0.2.7`，与 9 月 28 日已解包源码对比。分别有 20、542、36 个已有 Python 文件逐字节一致，零变更。SHA-256 见 [current-wheel-comparison.json](current-wheel-comparison.json)；文件创建时间仍为 9 月 27 日，见 [wheelhouse-file-metadata.json](wheelhouse-file-metadata.json)。未安装这些 wheel。
- 源码证据摘录保留原文件名、SHA-256 和原始行号，见 [source-excerpts.md](source-excerpts.md)。
- 最新 HARNESS_README 下载链接取得，但实际下载返回 HTTP 429；README 说明来自 9 月 28 日归档，SHA-256 `3d6e57a13234cb4e783ba24eaab486459af76e0923ea4c6607cd41cc8961bbbb`，不冒充新下载版本。
- 当前公开三份 wheel 未变，不足以证明正式评分服务没有独立部署补丁。未下载 433 MB vLLM wheel，也未取得评分服务启动代码或最新模型 chat template，因此服务器侧状态单独列为待确认。
- 主办方新公告与规则以主审的 `research/competition-check-20260930/` 取证为准；不能把社区推测当作已部署事实。

官方入口：[评分与 12 小时规则](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/evaluation)、[工具与技能规则](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/model-selection-budget-and-harness-rules)、[wheelhouse](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)、[Starter](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent)。

## 1. 预算实际上限制什么

| 参数/口径 | 当前实现 | 对研究的含义 |
| --- | --- | --- |
| `max_time_minutes: 5` | 初始化仓库之后 `start_agent_session()`，Agent 循环套 `asyncio.timeout(300)`；不含 A 准备与 B 裁判 | 可以提交调整，并非官方强制 5 分钟。不能用整题 duration 当模型推理耗时 |
| `max_tool_calls: 50` | 计费工具进入前检查，正常调用递增；工具失败也消耗调用 | 查看是否实际碰到 50，不能因失败就认定调用上限是瓶颈 |
| `max_turns: 80` | `RunConfig(max_llm_calls=remaining_turns)` 与上下文计数共同限制；所有子 Agent 共用任务上下文 | 多 Agent 不获得额外 80 轮/5 分钟，切分角色有额外成本 |
| `get_status` / `submit_patch` | 不增加普通工具调用数，且不受 count-gated 的工具/turn/token 检查；仍受任务时间检查与外层 LLM 轮数约束 | 免费不等于超时后仍可用；不能在最后一秒才依赖其成功 |
| `timeout_seconds: 60` | `run_command` 用 `min(60, max(5, int(remaining)))`；B 终局 `pytest` 也用同一个 60 秒 | 不能简单把 124 当模型错误。诊断裁判超时可冻结补丁单独调整，不同时加 Agent 时间 |
| 正式 12 小时 | 官方 Evaluation 明确：含所有任务补丁生成及 sandbox setup，排除 patch validation | 本地 A+B 完整面板耗时与正式 12 小时分母不同，吞吐需分别建账 |

源码：`agent_runner.py:428-441,493-516,635-673`；`context.py:467-544`；`execution.py:15-47,71-98`；`verification.py:482-505`，均见源码摘录。

这里“60 秒也限制 B 裁判”已由我方 `Evaluator.evaluate_task` 同一个 EvalConfig 路径确认。README 明确包内预算由正式 Stage 1 inference 加载，但尚未取得当前 Stage 2 `metric.py`；不能据此保证提交包的 `timeout_seconds` 一定决定正式评分的验证时限。改变本地裁判时限首先是诊断措施，是否能作为正式分数改进需要官方评分入口证据。

正式预算评估应汇总“服务与计入的初始化 + 各题 A 准备 + Agent + 生成收尾”，并预留启动/异常余量。B 裁判时间另列。主审取得9月29日官方744258回复：总隐藏测试约120题，公私约各60题，同一次12小时运行；精确题数与具体评分初始化计时起点仍未知，不能推导保证满足12小时的每题上限。

包内 `eval_config.yaml` 是静态四字段配置，没有任务难度回调或跨题余额调度入口。可以通过提示词/声明式流程依据本题 `get_status()` 少做或多做检查，但这是软策略，不能声称具有可编程的跨任务自适应预算。任意主机调度器需要主办方入口支持；当前未发现。

## 2. 已发现的请求传递问题

### 思考预算数字不是已证实的推理 token 上限

`adk_submission/resolvers/generation.py:61-126` 的桥接函数将 `thinking_budget > 0` / `include_thoughts` 转成 `extra_body.chat_template_kwargs.enable_thinking`；它没有把 4096 作为数字上限发送。`google/adk/models/lite_llm.py:1943-1965` 只转采样、输出长度等字段，没有转 `thinking_config`。

CPU 直接执行已取得源码中的桥接函数：`4096,true` 与 `1024,true` 输出完全相同；`4096,false` 输出 `enable_thinking:false`。结果见 [cpu-contract-probes.json](cpu-contract-probes.json)。这验证的是函数映射，不是 GPU 模型行为。

因此当前不应安排“4096 对 1024”作为有效推理预算消融。可辨别的候选是思考开/关与 `max_output_tokens`，仍需检查实际请求和最终 chat template。`thinking_budget:0` 也不能照 README 直接写：当前 compiler schema 的 `ThinkingConfig` 要求 `ge=1`；关闭思考应首先测试 `include_thoughts:false`。

同样，YAML schema 允许 `seed`，但上述 LiteLLM 参数转发白名单未包含它。固定 YAML seed 不能作为已经控制随机性的证据；需看真实请求。temperature 0 也不等于整个服务完全确定。

### 双 JSON 链仍存在于当前公开源码

`swegemma/tools/base.py:11-31` 的工具返回 `json.dumps(dict)` 字符串；`google/adk/flows/llm_flows/functions.py:1116-1117` 将非 dict 包成 `{'result': string}`；`lite_llm.py:804-809` 再序列化外层。因此工具内容中换行和引号会被多层转义。这解释一种输入质量风险，不等于已经量化其成功率损失。

### 思考历史的转发边界

`lite_llm.py:879-940` 将 thought 与普通 content 分开，只把 thought 送入 `reasoning_content`。它是否被服务端 chat template 保留需要验证；主办方关于 vLLM 丢弃该字段的公告不能替代部署状态检查。修复后需要新的冻结基线，不能将修复前后运行混算成策略收益。

## 3. 哪些改进能够随提交包落地

| 能力 | 可提交性及证据 | 实现注意点 |
| --- | --- | --- |
| 提示词、四预算字段、被转发的生成字段 | 有支持 | 新变量单独记录；不要假定 schema 接受就一定送达模型 |
| 文本/AST/符号定位技能 | 有支持：`skills` 加载目录，`SkillToolset` 使用官方 sandbox executor | 使用源码构造索引，可涵盖 `ast.AsyncFunctionDef`；输出候选位置和证据；不把答案、gold 修改位置、其他组轨迹写入脚本 |
| 技能脚本做补丁审查 | 有支持的可研究路径 | 读取 `git diff`、列出临时文件/测试修改和空源码改动；报告给 Agent。它必须被调用，不能包装成自动强制钩子 |
| 任意自定义 host Python 工具 | 无支持 | compiler 只从 host registry 解析工具名；不能上传 `agent.py` 并执行 |
| YAML callbacks | schema 有字段，但当前 harness 不传 `callback_registry`；resolver 遇 None 直接返回空配置 | 不能据此宣称可实现强制防循环、自动提交、工具拦截 |
| 串行 `SequentialAgent` / `agent_tool` | 声明式支持，同一基座、同一中央预算 | 可以隔离只读定位角色与修复角色、用 `output_key`/`include_contents` 控制交接；不得给分析角色 submit 能力导致提前结束。先证实单 Agent 定位增强，再研究拆分的增量 |
| 并行多 Agent | compiler 存在类并不等于比赛允许并发模型请求 | 依当前主办方关于不支持并行模型调用的规定，不将并行作为方案 |
| 自定义压缩器 / App compaction 参数 | 未发现提交 YAML 或四预算字段入口 | 本地能修改 Notebook 不等于正式服务使用这些设置 |
| 自建索引/图 | 运行时通过被允许技能分析本题源码是合理路径；替换官方 host 图目录没有已证提交入口 | 首轮做轻量符号索引而非预建全仓服务；不要把官方 graph API 路径当可写可替换能力 |

技能支持证据：compiler 通过 `compile_submission(..., code_executor=..., script_timeout=...)`；`resolvers/tools.py:184-221,253-260` 加载 `skills/` 并注入 sandbox executor。技能脚本不是绕过预算的入口。

当前 A0 原有“不要把复现文件放入 diff”的软指令不能保证补丁干净。技能检查可以把问题变成可见、稳定的反馈，但即便这样仍需测量调用率与执行率，不能宣称约束已被强制执行。审查自身工作树是正常用途，不应更改评分器、隐藏测试或规避反篡改检查。

## 4. 图工具真实能力与风险

- `search_similar_code` 是“已有节点向量之间找相似节点”，没有任意自然语言编码模型。`embedding_utils.py:201-214` 首先精确匹配，随后返回第一个满足短名后缀的键。短名可歧义，必须尽量采用完整标识。
- `get_code_neighbors` 返回进出邻居合并后的名字列表，`graph_utils.py:341-363` 用集合后按字母排序并截断；默认 50 不是相关性前 50，返回值本身不标注调用方向。可以读源码或用子图检查关系，不应将名字邻接直接当作真实执行链。
- 相似检索会返回节点完整代码，当前 wrapper 没有统一的源码字符截断；盲目 k=10 与邻居 50 可能增加上下文成本。需要测返回字数、重复代码、空结果及找到相关源码所耗时间。
- 索引可能缺少异步函数、嵌套动态结构或相关定义。旧公告不能作为 9 月 30 日最新覆盖状态；应按被选开发快照做源码节点对照。不能提前读取保留题答案。
- 没有“图无效”或“图必有效”的本项目证据。6/24 题现有研究均未给 Agent 注册图工具；路径存在不等于能力已测。

所以比较对象至少包含强化的文本/符号基线，而不是让薄弱 grep 基线与复杂图策略比赛后便将差异归功于 embedding。可以先离线判断图覆盖与 issue-only 查询召回，再决定是否值得 GPU 对照。

## 5. 上下文与评估可信性

官方 Starter v2 当前源码 `compaction_interval=15`，我方载体相同；归档 HARNESS_README 对正式 `scripts/inference.py` 描述的是 5。两者可能只是示例/评分默认差异，尚无当前评分入口字节核实。必须记录这种差异，不能把我方配置泛化成正式环境固定值。

可提交的上下文改善包括缩小源码阅读窗口、给长函数做有证据的分段、工具输出只列有限候选、避免大段重复返回、在普通文字/技能结果中留下简短已知事实与下一步依据。若思考内容在历史中丢失，单纯依赖隐藏 thought 保存跨步结论不可靠；这一假设需要随主办方修复重新验证。`include_contents:none` 配合结构化交接是可提交候选，但可能丢掉修复必须的信息，不宜先作为默认。

`read_file` 的 150 行/10000 字符、`run_command` 的 5000 字符都是 host 限制，当前没有正式提交字段调整。应优化查询输出，不靠绕过限制发送完整仓库。

Docker 与 subprocess 不可等同：官方 README 描述 Docker 为无网络、4 GiB、2 vCPU；当前 `SubprocessManager` 默认 `system_site_packages=True`，并将宿主 site-packages 写进虚拟环境 `.pth`，ensurepip 失败时也会继续回退。依赖/插件可能继承宿主状态，且资源隔离不同。requests_6592 的 fixture 递归只证明对应环境的验证失真，不能推断正式隐藏集不可解。

下一轮可信门槛：

1. 冻结官方包/模型/脚本指纹；查明主办方推理历史与 JSON 修复是否已反映在真实消息渲染中。
2. 对每个拟纳入开发比较的任务，在 Agent 不可见的验证流程里执行原始补丁/空补丁与 gold 对照。gold 必须通过、空补丁应暴露目标失败；另分类本来就通过、setup 错误、gold 失败和超时。优先 CPU 重放，避免浪费 GPU。
3. 保留全量原始研究分母并另报“经过验证的可解任务”分母，预先冻结筛选规则，不根据某组得分临时剔题。
4. 同题各组必须相同 A 环境与 B 裁判；先冻结补丁判断环境影响，再重跑 Agent。不要把 pip/fixture 修复后的本地分数当正式分数。
5. Docker 可复现时优先做对齐检查；暂不可复现时明示 subprocess 代理环境局限。任务选择与最终裁判输出不能进入 Agent 输入；保留集继续不参与方案设计。

## 6. 三个最高价值实验（交用户实现）

### E1：评分与请求合同验收，优先 CPU

目的：确认之后的差异来自可提交 Agent，而非版本变动/错误裁判/未生效参数。

- 复用冻结补丁，对拟选开发任务执行 gold/空补丁和必要的历史 Agent 补丁验证；记录 60 秒与诊断宽限下的测试结果。若已有 gold/setup 问题，先归类再选对照任务。
- 通过最小本地/mock 请求回放检查工具 JSON、thinking history、生成字段实际渲染；正式服务修复状态另核查，不自行改包并混入正式收益。
- 输出版本清单、eligible manifest、校验结果、已知环境排除原因、实际有效的参数清单。
- 停止条件：如果关键消息仍被破坏，先暂停扩大 GPU 面板；材料分析与定位 CPU 验证继续。修复部署后冻结新的 A0，保留 P003 为历史证据。

### E2：定位工具对照，强化文本基线并按需加入图

目的：减少“没有经源码确认的位置却开始写复现/编辑”的失败。

- 在 P003 已暴露开发案例上实现可提交导航技能：issue 提取的路径/符号/异常字符串 → 文本/AST 候选 → 有行号的有限源码上下文。支持同步/异步函数，禁止 gold 进入查询。
- CPU 比较官方相似/图结果的覆盖、空结果、短名歧义、输出长度及新增相关位置；只有覆盖与查询质量过关的方式进入 GPU。
- 同预算比较 A0 文本基线、导航技能、导航技能加按需图。其余提示、模型、采样、工作区保持一致；不把技能和全新补丁生成策略同时变更。
- 首先少量预登记配对与重复验证工具确实被用，再用新的开发题确认；若图只增加阅读没有新增位置，保留技能版本并停止扩图。
- 主要终点为通过率与每题 A 阶段用时；定位时间/召回/源码修改率解释机制。11 次定位失败是尝试数，实施前按任务和快照去重，不能视为 11 个独立题。

### E3：真实有效的资源配置，而非伪 thinking-budget 消融

目的：在 12 小时生成预算内提高解决数，而非固守研究用 5 分钟或无条件加时。

- 先从轨迹统计达到时间/调用/轮数/输出截断的各类数量。固定 E2 胜出定位方案和可信裁判。
- 一次只试真实生效的因素：优先 `max_output_tokens` 较低候选与 16384，或思考开/关；不使用当前等价的 4096→1024。低输出上限必须监控 `<|tool_call>` 截断和补丁完整性。
- 对确有持续源码进展但被 5 分钟截断的任务，预登记少量 5 分钟对较宽限额的配对，再决定是否扩大；不要给纯循环样例盲目加时。
- 报告相同验证条件下解决率、A 准备/Agent/收尾分段、生成吞吐及按假设题量推算的 12 小时可行域。预算/思考两个方向分批而非全因子铺开。
- 串行只读定位子 Agent 留为后续结构消融，只有单 Agent 的上下文/定位耗时已证瓶颈且技能增强仍不足时再上；暂不新增训练工作。

以上是交接实验设计，不是已经实施的 P002 或新的成功率结论。
