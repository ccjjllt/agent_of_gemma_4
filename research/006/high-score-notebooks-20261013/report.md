# P006：公开高分源码与新证据更新

检索日期：2026-10-13，北京时间。起点为 P005 2026-10-10 公开方案研究。本轮读取 Kaggle ScoreDescending 前 30、DateRun 前 30、10 份完整 Notebook 源码和 2 个新讨论；用 Chrome 页面可见评分徽章绑定分数与版本。只读研究，没有运行外部代码、Notebook、模型、GPU、提交或发帖，没有读取本项目 heldout 内容。源码仅作为研究材料，未执行；嵌入的字面配置通过 ast.literal_eval 提取。完整来源清单见 [retrieval-manifest.json](retrieval-manifest.json)。

## 核心结论

1. **仍未找到可核实、能完整对应 0.20+ 的公开 Agent 源码。** ScoreDescending 搜索不返回 score 字段；本次页面核实的最高完整源码仍为 0.18。不能从标题、排名、作者队伍最高分或点赞推断代码成绩。0.24 榜首的具体机制仍未知。
2. **0.18 的可靠公开路线仍是无训练单 Agent。** 共同特征是较少工具动作、8192 输出预留、8 分钟/28 调用、直接局部编辑与较短验证流程。不是三技能状态平台，也没有经核实的高分 adapter。各配置、工具与提示同时改变，不能据外部分数单独认证某个参数。
3. **最有价值的新反证是跨仓库结果。** 更新至 V18 的研究报告四工具 coder 在开发集 27/70 对 coder–analyzer 16/70，但 Rich 确认集双方都是 12/44；两 reviewer 为 13/70 对 16/70；开发选择的预算日程在 Rich 为 7/44 对 12/44。额外角色和更复杂调度目前没有获得收益支持。
4. **当前最值得做的是让证据尽快变成生产修改，并保持足够推理上下文。** 我们的 single_state 三技能在四题实验中完全未被调用；新公开资料又记录大量 edit/write 参数畸形与重复重发。轻量、单字符串编辑入口有明确机制动机，但仍须真实修复配对证明，不能把工具成功率等同于比赛收益。
5. **不能继续沿用“图工具普遍返回空”的旧事实。** 10 月 11 日的 CPU 重放显示新构建明显改变返回率；是否提高解决率仍未测，较大回包会增加上下文负担。图应作为有问题驱动的局部增量，不作无条件工作流。

## 分数、版本与源码绑定

下面的页面分数均由本轮浏览器可见徽章确认；原文摘录和精确 scriptVersionId 在 [browser-score-evidence.json](raw/browser-score-evidence.json)。分数是对应版本证据，不能转移到最新改动后的 bundle。

| 方案 | 本轮版本 | 可核实分数 | 解释 |
|---|---:|---|---|
| [Budget-Fit](https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent?scriptVersionId=355310852) | V5 / 355310852 | **0.18 V5** | MCP 同时确认 team 16981679 的公开 submission 56839138 为 0.18；完整四文件配置可读 |
| [Budget-Aware](https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-v2?scriptVersionId=356722956) | V9 / 356722956 | **0.18 V9** | SuperAgent 派生、单 Agent、无 adapter；源码文字仍说未评分，当前徽章已提供 V9 分数，按更强页面证据记录 |
| [Task Soundness, Serving and Evaluation](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-task-soundness-serving-and-evaluation) | V18 / 357877283 | **0.18 属 V12 / 356256255** | 当前 V18 的 issue-contract 改版明确没有得分或本地质量优势声明；0.18 绑定未改动 Budget-Fit 旧包 |
| [Sam Budget-Fit](https://www.kaggle.com/code/samvalladares/gemma-4-agent-budget-fit-single-agent) | V4 / 357438539 | **0.18 属 V1 / 355976064** | V4 新提示不能继承 V1 成绩；MCP 指定 V1 源码返回 Not found，所以只分析当前代码，不做精确 V1 归因 |
| [Apex Silver LB 0.24](https://www.kaggle.com/code/hitarthjain0/gemma-4-apex-silver-agent-lb-0-24) | V1 / 357610384 | 页面无评分徽章 | 标题写 0.24～0.26+，正文却明确为未测路径；把 honghanhhh 的标题错当 0.24 实绩，不能用作高分证据 |
| [Contract-first 1-agent](https://www.kaggle.com/code/nilanjansaha123/gemma-4-dev-agent-contract-first-1-agent-bundle) | V1 | 本轮未核实评分版本 | 可读源码，主要为 Budget-Fit 的提示/工具删减派生；不能借其父本 0.18 |
| [Single Agent Anytime Patch](https://www.kaggle.com/code/leoprovorov/single-agent-anytime-patch) | V4 | 作者明确未实测排行榜 | 依据其他 bundle 的结果提出 prompt 变化，不是新增成绩 |
| [Gemma + Gemini](https://www.kaggle.com/code/matterhorn3838/gemma-gemini-the-ultimate-duo) | V4 | 本轮未核实当前版本分数 | 完整代码声明的模型全是 Gemma；标题不代表调用 Gemini。混有多份历史报告和候选，不能把旧正文得分套到当前生成包 |
| [Tactical Inference Budgeting](https://www.kaggle.com/code/noormamhidayat/tactical-inference-budgeting-swe-agent-gemma4) | V1 | 未核实 | 实际代码仅打印硬编码 JSON，未含 Agent、评测、训练；“129 holdout、100% sweep、4× faster”等无运行原件支持 |
| [Gold Submit](https://www.kaggle.com/code/ozertuu/gemma-4-gold-submit) | V2 | 未核实 | 仅复制 dataset 后打包；数据目录清单是四个配置/提示文件，无 adapter。标题 gold、正文 0.24 winner 不构成得分证据，本轮未下载该 dataset 正文 |

浏览器同时看到当前榜首两队 0.24，3–8 名 0.22，随后 0.20；同为 0.20，15 名显示 gold、16 名显示 silver。这是当前榜单快照，既不是最终奖牌保证，也没有提供对应源码。正式 0.13 与这些成绩不能按我们四题开发集的 2/4 换算。

## 实际配置比较

共同模型均为这些方案所用的 31B CT，不代表现行比赛仍只允许 31B；新模型许可由主研究依据官方 748287 单独确认。

| 配置 | 工具/角色 | temperature / output / thinking | timeout / calls / minutes / turns | 训练 |
|---|---|---|---|---|
| 我们正式 single_state | 六基础工具 + 三技能的 SkillToolset；单角色 | .2 / 16384 / 4096，保留 thoughts | 60 / 50 / 5 / 80 | 无 adapter |
| 0.18 Budget-Fit V5 | 九官方工具，提示禁用图；单角色 | .2 / 8192 / 4096，保留 thoughts | 240 / 28 / 8 / 80 | 无 adapter |
| 0.18 Budget-Aware V9 | 同九工具，提示禁图；单角色 | .2 / 8192 / 4096，保留 thoughts | 240 / 28 / 8 / 80 | 无 adapter |
| Sam 当前 V4 | 六工具，图未注册；单角色 | .2 / 8192 / 4096，保留 thoughts | 240 / 28 / 8 / 80 | 无 adapter；V4 未绑定 0.18 |
| 公开冻结 four-tool V4 | run_command、read_file、edit_file、submit_patch；没有 get_status/write_file/skill | .2 / 8192 / 2048，保留 thoughts | 60 / 20 / 5 / 60 | 无 adapter |
| Apex/Anytime 当前版本 | 六工具，单角色，无 skills | .2 / 8192 / 4096，保留 thoughts | 240 / 28 / 8 / 80 | 无 adapter；无确证新成绩 |
| Gemma+Gemini 中 FILES 初始候选 | coder + 两只读 reviewer + reconciliation，同一Gemma | coder .2/8192/4096且include_thoughts:false；review .2/2048/0；reconcile .2/1024/0 | 初始4分钟/60工具/80turn/45s；后文还修改并构建不同预算 | 无训练权重；不能把初始字典当最终导出物 |

所有这些方案复用官方基础工具 schema，未提供新的执行器实现；除我们技能和旧 v19 自定义编辑器外，公开高分组合仍依赖官方 edit_file/write_file 多字段调用。源码明示 top_p 均 .95。旧文章对 thinking 数值和 include_thoughts 的解释随官方桥接版本失效；只记录字面配置，实际条件需对应 wheel 构建。

### 精确四工具 prompt：不是现成的“单命令 Agent”

[完整抽取](extracted/four-tool-v4-full-prompt-and-config.md)来自 Gemma+Gemini Notebook 当前 V4 的历史 `V4_FILES` 字面配置；也可在 [原始源码](notebooks/matterhorn3838__gemma-gemini-the-ultimate-duo.txt)内复核。它要求：从 issue 路径/符号定位；明确单文件问题最多三次探索调用后 `edit_file`；连续完成必要生产修改后立即 submit；禁止编辑后继续复现、查测试或执行命令，将判分交给独立 verifier；新生产文件才用 run_command。

这与计划中的 `minimal_command` 有实质差异：后者移除 edit_file、保留 get_status、以单字符串 shell/heredoc 编辑，保留目标行为确认，且维持我们原采样和 50 调用。它是新的复合候选，不能给它挂上公开 four-tool 的 27/70 或 43.4% 节时成绩。

### 高分提示值得借鉴的部分与不能照抄的部分

- Budget-Fit/预算派生把操作约束写在很近的位置：定位约前 10 调用、12 调用前编辑、20 调用后停止新探索、一次针对性检查和一次修正。其作用是减少高频无进展动作，不是增加任务信息。
- Budget-Aware 增加明确任务契约、sibling feature 的 imports/exports/dispatch 注册点，直接定位行为产生处。对应我们“懂了原因但迟迟不改”的已见失败，值得保留简短证据→动作纪律。
- “每题必须制造非空修改”“第 8/12 调用必须猜改”可能把困难题的必要阅读截断；历史数据中短轨迹更成功，可能是题容易，而不是强行变短使它成功。不要把这种相关性当因果。
- Sam V4 记录 V3 将调用 28→36 后整个评分超 12 小时，回退到 28。其作者“这一预算所有历史都能完成”的措辞不能推广为保证。工具数量上限和时间上限共同决定尾部成本。
- Sam prompt 写 submit 可多次、最后一个有效；我们历史源码与观察曾是第一次提交冻结。当前语义应由当前官方源码核定，不能复制外部指令替代事实。
- 三技能并不因代码存在就产生作用。我们四题候选没有任何 load_skill/run_skill_script，重写提示时应先减少进技能的选择成本、避免再加必须维护的状态对象，而不是继续堆技能实现。

## 10月10日之后的新研究证据

### 1. 精简 coder 的开发增益没有变成 Rich 增益

[Dmitrii V18](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-task-soundness-serving-and-evaluation?scriptVersionId=357877283)新增 692 个结果，八配置、70 开发题及部分配置 44 Rich 题。作者只发布聚合与任务结果，没有本轮需要读取的题文/补丁。报告四工具 27/70 vs coder–analyzer16/70（14胜3负）；Rich12/44 vs12/44（5胜5负）；Rich平均215.65s vs381.22s，约少43.4%。两reviewer 13/70 vs16/70（3胜6负）；开发选择的六分钟预算日程Rich7/44 vs12/44（1胜6负）。

这是完整设计对比，prompt/tools/budgets一起变；A100 TP1八路并发测得的单题延迟不等于正式4×L4串行吞吐。Rich此前已经被观察过，不是真正全新独立holdout。Gemma+Gemini还重复引用同一来源，不算第二份独立复现。适合推导“暂不优先多评审/复杂日程”，不适合推导“4工具一定泛化更好”。

V18的训练负实验仍是先前30步、412turn/23任务的LoRA：NLL改善、base4/32 vsadapter2/32。没有新正向修复证据，没有可核实高分训练权重。其当前issue-contract候选是无训练prompt改版且无质量声明。

### 2. 参数格式失败仍是实际瓶颈；不是只避开三引号即可

[748323，10月11日](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748323)作者报告9月30日与10月9日构建下的历史轨迹，当前Budget-Fit112任务：524次edit_file/write_file中105次因必需参数缺失被ADK拒绝，分布15题；6题占85次，最多14次相同重发；47次出现代码片段变成参数名，只有2次含三引号。原因位于模型生成或工具解析哪一侧尚不能区分。

这与我们 fastapi_5077 的缺filepath循环一致，优先级高于重写完整工作流。可迁移策略是同一参数错误只纠正一次，其后换等价单字符串入口；保留真正最小修改和简短原始错误，不让模型反复转义整段源码。公开 robust-edit 的三轮6/16对6/16反证仍有效：降低格式失败不保证语义修复提升。

### 3. 图工具回包变了，旧禁令不是永久事实

同帖用314条已录图调用在四构建重放：有结果82→11→270→270，后两构建是10月8日0.2.10与10月9日0.2.11；含参考补丁文件节点的调用122次，top3内87次。所有314调用在原运行构建上重放与历史空/非空结果一致。中位回包5763字符，async仍缺失。作者10月12日补充公开 [instruments kit](https://www.kaggle.com/code/kimdonggyu1255/gemma-4-agent-instruments-kit)。这是工具返回证据，不是解决率实验；本轮未执行重放，也未读图背后的本项目保留题。

潜在增量：只在文本定位有真实歧义、已有可查符号时取少量邻居，并控制输出；不要因为新工具会返回内容就恢复三个图工具并默认调用。图可能更耗上下文，正好与我们16384预留的故障耦合，必须分开解释。

### 4. 新“Anytime”是有根据的假说，不是已获收益

[Anytime Patch V4](https://www.kaggle.com/code/leoprovorov/single-agent-anytime-patch)从其他bundle的692结果归纳194无patch、75上下文溢出、52无编辑超时、53未编辑主动停止；提出8调用前首改、80行读取、小stdout、搜索时短思考。作者承认不知75溢出案例是未编辑还是编辑后丢patch，不能从空patch反推全无编辑。其当前代码只有prompt变化和打包。应迁移短输出与及时行动的假说，不能用该相关性直接增加强制猜改。

[748168](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748168)自报28调用/5分钟得0.15，24调用/5分钟0.13，32调用/8分钟超时，同时变了策略/skills/推理开关；没有精确bundle匹配或重复，不足以选择最优预算。只支持预算尾部确会影响整次运行。

## 对我们下一轮的优先级建议

**P0：保留正式0.13为真实基线，先做可解释的执行候选。** 主研究在账户只剩3.28749额度小时后，选择先准备closure_prompt：保留当前single_state的全部工具/技能和采样预算，仅附加一次错误纠正/等价工具替换、源码证据满足即局部编辑、已证目标行为后收尾三条提示；候选已准备但未运行。去掉未用技能、改为单字符串编辑的minimal_command仍是后续未实现设计，会同时改变工具与提示，不能与本轮单文件变化混称。两者都不能借用公开four-tool的开发成绩。

**P1：把上下文生存空间独立处理。** 8192输出预留和有限stdout可能减少硬溢出，公开0.18参考均用8192；但我们历史0C16384 7/12对8192 6/12，不能宣称8192已证提分。与行为提示/工具入口拆开，记录真实截断、畸形调用、生产编辑及resolved。

**P1：训练数据优先支持行动与纠错。** 没有新的“训练即获高分”公开证据。收集真实当时信息下的下一行动、参数失败后正确恢复、短编辑与必要邻近行为检查；区分任务正确性与格式正确性。不能把参考答案反推的短diff或重复历史题当跨仓库经验。

**P2：局部图增量或一次有目的的独立复核。** 只针对真实剩余失败类别，不默认增加双reviewer；先明确要新增哪条不同证据。公开多评审已出现负结果，只有不同角色名字的相关判断可能增加成本而不增加信息。

**不建议现在优先做：** 追随0.24标题；复制Apex/Gold所谓赢家；强制全题8分钟；扩大任意多Agent；为了更高动作成功率继续扩建编辑器；把loss下降、编译通过、patch非空、局部样本翻倍换算为金牌概率。

## 证据限制与交付

- [raw/](raw/)保存完整MCP响应、两帖和浏览器评分摘要；[notebooks/](notebooks/)保存10份完整源码逐cell展开；[extracted/](extracted/)保存安全提取的字面配置与四工具完整prompt。
- MCP get_notebook_info返回的last_run_time多为同一个10月10日窗口，与search卡片和页面时间冲突。使用本次检索日期、精确版本和scriptVersionId，不把该字段当实验发生时间。
- ScoreDescending首30仅是搜索覆盖，不证明互联网没有更高公开方案。已核实上界只表述为“本轮未找到对应0.20+完整源码”。
- source文件中的“held_out”可能指作者自己的Rich划分；本轮只读论文式聚合说明，没有打开本项目heldout题文、参考答案、轨迹或训练素材。
- 本轮没有产生新模型修复成绩，研究收益是排除错误高分归因、更新机制假说和形成可执行候选依据。
