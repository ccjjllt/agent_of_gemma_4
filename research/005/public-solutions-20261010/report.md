# P005：公开方案、可核实成绩与训练代码研究

日期：2026-10-10（北京时间）。先阅读了本项目 2026-10-01 研究机会文档，再通过 Kaggle MCP 搜索、取得 8 个 notebook 的完整源码，精读其中的配置、提示、编辑器、执行追踪及训练数据生成逻辑；浏览器仅用于核实 MCP 未提供的分数徽章。没有运行任何 notebook、训练、提交或下载模型。研究集中于能改变修复能力的机制。

## 1. 研究结论

1. **可靠编辑仍值得做，但已有公开实验直接显示：减少编辑失败并不自动增加修复数。** 我们的 T2 应作为基础能力，冲奖的新增研究重心应前移到“触发条件 → 执行路径 → 根因假说 → 精确修改”，即加强 T3，而不是继续只优化工具成功率。
2. **可核实且能完整读到配置的竞争参考是 0.18 单 Agent 和 0.17 v19，不是标题中的 0.24。** 本轮未建立榜首 0.24 与某份公开源码/LoRA 的对应关系。
3. **公开训练代码质量差异很大。** 目前读到的真训练实验包括负结果和不同量化基座的短 diff SFT；没有找到经本轮核实、依靠训练赢得本赛高分的公开 adapter。
4. **最有研究价值的新组件是轻量执行路径证据和行动定向训练数据。** 可在既有 T2/T3 上扩展，没必要先建立更复杂的多 Agent 平台。

## 2. 分数与源码是否真正对应

| 来源 | 本轮核实的分数 | 源码与含义 |
|---|---|---|
| [Budget-Fit Single Agent V5](https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent?scriptVersionId=355310852) | Kaggle 页面 Public Score / Best Score **0.18 V5**；MCP team 16981679 的公开 submission 56839138 亦为 0.18 | V5 完整源码可读；单 Agent、无 adapter |
| [v19 原始 V1](https://www.kaggle.com/code/yiyu0716/g4-pack-yiyu-v19-fixed-trace-t7p5?scriptVersionId=356344238) | Kaggle 页面 Public Score / Best Score **0.17 V1** | [11 文件完整源码镜像](https://www.kaggle.com/code/yiyu0716/g4-v19-complete-harness-source)明确声明逐字节导出原 ZIP；本轮读了完整镜像，不执行其导出代码 |
| [Gemma 4 Baseline | LB 0.24](https://www.kaggle.com/code/honghanhhh/gemma-4-baseline-lb-0-24) | 实际页面 Best Score **0.13 V2**；当前源码是 V3 | **标题的 0.24 是当时公开榜首，不是这份代码的成绩**；正文明确路线尚非实测得分，不能称“0.24 高分方案” |
| [Measure Before You Tune](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-measure-before-you-tune) | 页面 Best Score **0.18 V12**；当前源码 V13 | V13 打包公开无 adapter 的 Budget-Fit/SuperAgent 衍生方案，同时报告训练负实验；0.18 不能归因于 LoRA 或当前源码中的任意实验 |
| [当前官方排行榜](https://www.kaggle.com/competitions/gemma-4-developer-agent/leaderboard) | 前两名 **0.24**，第三 **0.22**，随后多个 **0.20** | 这是官方队伍成绩。没有匹配到榜首训练方法或完整提交源码，不猜测它们使用 LoRA、图检索或某个热门 notebook |

ScoreDescending 搜索返回的 notebooks 不含 score 字段，不能据排序或点赞确认得分。Kaggle notebook 的金色奖章是代码内容奖章，也不等于本赛金牌。浏览器分数证据见 [browser-score-evidence.json](raw/browser-score-evidence.json)。

## 3. 两份已核实成绩的完整代码

### 3.1 0.18 Budget-Fit：较少动作、较长单题上限

实际配置：temperature 0.2，top_p 0.95，max_output_tokens 8192，thinking_budget 4096，include_thoughts true；8 分钟、28 次工具调用、80 turns、command timeout 240 秒。仍注册 9 个工具，但提示明确不要使用三个图工具。没有 LoRA、没有额外角色。

提示策略：从 issue 中提取精确符号或报错，一次 grep 后读取窄区域；前约 10 个调用定位，12 次调用前编辑；一次目标检查后结束；不修复与当前 issue 无关的环境/测试故障。作者将“晚期仍只读不改”视为低收益轨迹。

**可迁移判断：** 单 Agent 是值得保留的强对照，清晰的任务契约（精确异常、字符串、参数与边界条件）可直接进入我们的 T3。8 分钟和 28 次不是通用最优：120×8 分钟最坏 16 小时；作者仅称某次正式运行在 12 小时内完成。强制“12 次前猜一个修改”是相关性驱动的启发式，可能损害确需多文件理解的问题，不建议原样照搬。

8192 的外部经验主要关乎请求窗口余量；我们历史 0C 仍是 16384 7/12 对 8192 6/12。外部代码不能抹掉本地结果，应将“动态可用输出容量/上下文增长”视为独立问题，而不是直接翻转冻结参数。

### 3.2 0.17 v19：实际执行路径 + 单字符串编辑

配置为 7.5 分钟、100 工具、120 turns、command timeout 300 秒；两个角色均 temperature 1.0 / top_p 0.95 / top_k 64 / output 2048。YAML 同时写 thinking_budget 4096 和 include_thoughts false，作者历史实验标为 thinking Off；新轮子语义已变，不能把 YAML 的字面值当作今天的实际推理配置。

root 使用 run_command、read_file、get_status、submit_patch，并加载 g4-edit；一次转给同 Runner 中的只读 analyzer，结果通过 analysis_plan 返回。analyzer 最多 10 个调用，约 250 字报告 LOCATION / ROOT CAUSE / FIX PLAN / RELATED / TESTS / CONFIDENCE / TRACE。

最值得迁移的是 g4edit 的 `--trace`：
- 通过 Python `sys.setprofile` 和 `threading.setprofile` 捕获函数调用，仅保留仓库路径、真实函数，排除模块导入噪声。
- 按首次调用顺序列出函数名、首行号、调用次数；异常回溯最多保留最后 8 个仓库栈帧。
- 每文件最多显示 10 个函数，路径主体约 3300 字符，复现输出只留最后 600 字符。
- 它让模型确认“这个复现到底执行了哪些仓库函数”，比按名字猜修改位置更具体。

编辑通过单个 run_command 的 heredoc SEARCH/REPLACE，避免多个 JSON 参数丢失；Python 文件原先可编译时才检查修改后的编译；匹配失败反馈最近位置。其工具同时含宽松缩进匹配和修改清理逻辑，**本项目不应整段无分析地替换 T2**。我们既有 span handle 可保留精确原文语义，另借鉴单字符串传输和简短失败反馈。

执行路径并不等于根因。尤其我们 fastapi_9425 的 None/'None'、fastapi_11194 的 bytes/UploadFile 反例，即使跑到相关函数，仍可能没满足触发条件。更合理的迁移是：T3 的最小差异输入先规定预期差异，再用 trace 确认分支证据；不要每题无条件强制复现，也不要盲抄整套 analyzer。

作者 [10 月 9 日讨论](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747696) 报告：v19 本地 15 分钟两轮 44/129、49/129，8 分钟 44/129；移除 analyzer 的 v20 为 37/129。预算、采样、硬件和对照关系复杂，只说明单 Agent 与 analyzer 都应有证据，不证明 analyzer 必须存在。正式 0.17 与本地约 34–38% 不能换算。

## 4. 最重要的反证：robust-edit 减少错误但没有多修题

[10 月 10 日原作者帖](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747923)与[完整 A/B notebook](https://www.kaggle.com/code/magicsword001/robust-edit-reliable-edits-for-gemma-4-agents)：

- 指定 31B，4×L4，16 题×3 轮，两臂每题交替先后；40 调用、20 分钟，远长于我们的预算。
- 无工具时 edit_file 107/246 次失败（43%）；有工具时仍混用 edit_file，19/66 次失败（29%），替代 editor 8/75 次失败（11%）。
- **每一轮修复数都是 6/16 对 6/16。** 这是作者自报本地 A/B，非官方榜分；公开 notebook 源码能看到实验生成与计数逻辑，但下载源码的输出栏为空，未独立复跑。
- 单字符串 run_command 没出现其观察到的参数丢失；安装采用单独 SequentialAgent 步骤，47/48 次成功，纯“先调用X”的提示曾被 11/16 次跳过。
- 编辑后拒绝破坏 Python 语法、展示最近匹配行、提示不要重复相同失败，均可迁移。它还采用 skip-worktree 隐去测试文件变化；这与编辑机制是不同干预，不应混为一项“编辑更可靠”的收益。
- 脚本路径不能硬编码 /workspace；ADK 运行技能时会切到临时技能目录，这一点与本项目 10 月 1 日分析一致。

**对本项目的影响：** T2 不取消，但从“首要潜在提分点”调整为必要的行动基础。大规模新实现优先给触发条件推理、真实执行证据、可用于训练的决策片段；不要只因为 edit success rate 更好就继续扩建编辑器。

## 5. 模型训练：读到的真实方法与可借鉴之处

### 5.1 原生轨迹 LoRA 的负实验

[Measure Before You Tune](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-measure-before-you-tune)详细报告：
- rank 16，alpha 32，LR 1e-4，seed 42，30 optimizer steps，gradient accumulation 8，序列上限 8192。
- 23 个 development tasks 的 412 turns；监督 token 中 thought 97,829、action 19,464，约 5:1。
- Rich action NLL 降 40.7%，thought NLL 降 25.5%；但 Rich 成功轨迹 overall NLL 降 27.1%，失败轨迹也降 25.5%，说明广泛强化原有风格。
- 16 Rich tasks×2 seeds、每题两分钟、A100 TP1 八路并发的局部对比：base 4/32，adapter 2/32。它不能证明所有 LoRA 无效，但足以否定“loss 好看所以扩训练”的推断。

它分析了历史 ADK/vLLM reasoning_content 与 reasoning 字段错配。正确组合工具历史和 content parts 后 936 个 turns 的 prompt 长度匹配；只有 5 次 live probe 检查了 token identity。官方后续已修过该问题，不能沿用旧 renderer。研究价值是**训练条件应来自实际送入模型的 prompt，而非分析者臆造的全轨迹格式**。

对我们的建议：以行动选择、短修复、读后编辑、错误回包后转向为优先监督目标；thought/action 分别采样或加权是待研究假说，而非既有结论。不要把 27 题多次重复运行伪装成大规模独立经验。

### 5.2 真 31B QLoRA：能作资源配方，不能作获奖证据

[Den Pugovkin 的 QLoRA Study](https://www.kaggle.com/code/denpugovkin/can-gemma-4-fix-an-unseen-repo-31b-qlora-study)完整训练源码使用 Unsloth 31B bnb-4bit，`device_map='balanced'` 跨两张 T4；不是比赛指定 QAT 权重。

参数：r=8 / alpha=8 / dropout=0，attention+MLP LoRA，Unsloth gradient checkpointing，sequence=1024，batch=1，accumulation=4，LR=8e-5，AdamW 8bit，一 epoch，seed 3407，assistant response-only loss。

训练输入为 issue + 从参考 diff 重建的当前代码片段，输出 source-only unified diff；Rich 留出，最多 12 条用于 loss，最多 8 题的 base/adapter 配对 draft，每次最多生成 256 tokens。检查的是 diff 可解析性、路径和 changed-line F1，**没有执行补丁和完整 Agent 修复评价**；作者明确本版尚不声称 GPU 结果或榜分。

因此只可借鉴低显存配方和 response-only 训练，不能用 1024-token、已知正确位置的 diff SFT 代替我们所需的长上下文决策、定位与工具行动。训练/服务基座差异也是实质迁移问题。

### 5.3 标题是“Gemma 4 TPU”的示例实际不能用于本赛训练

[Nur Srijan 的 KerasHub TPU notebook](https://www.kaggle.com/code/nursrijan/gemma-4-devagent-kerashub-lora-training-on-tpu)在源码里设置 `MODEL_PRESET = 'gemma2_instruct_2b_en'`。若加载失败，还会创建前两层随机 `(rank,64)/(64,rank)` 矩阵，却把 adapter_config 的 base_model_name_or_path 写成比赛 31B。这些是演示占位数据，不是 31B 微调权重，也没有本赛效果证据。不要投入 TPU 配额复用此代码。

### 5.4 CPU oracle 轨迹：适合格式示范，不宜冒充真实推理

[Validated Oracle Trajectories](https://www.kaggle.com/code/kevindiazcordova/gemma-4-validated-oracle-trajectories)覆写 Evaluator 的模型执行阶段，由参考 patch 决定 graph 节点、read 范围、edit hunk，再调用真实 harness 工具并保存真实回包。作者报告修复依赖后的 gold 105/129，oracle 103/105；CPU 约一小时，Apache 2.0。源码可读，本轮未执行。

它能廉价生产与比赛格式一致的动作和工具观测，优于纯虚构 JSON。局限是位置与动作路径由答案反推、thought 用短模板，不能学出从 issue 自主找到根因的能力，也不是模型成功率。适合少量工具格式与精确编辑 warm-up；语义推理训练仍要真实信息状态下的搜索、假设推翻和修复经验。旧模板“会丢历史 thought”的说明已过时，需以当前官方服务语义为准。

## 6. 对推理预算的新证据

[746250：作者 129 题本地扫描](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746250)：
- reference temperature .2/thinking 4096 为 48/129；temperature .4 和 1.0 都是 51/129；thinking off 23/129。
- 摘要关闭、thinking 4096 为 35/129，涉及历史 bridge 语义，不可当成今天单纯隐藏展示文本的因果效应。
- 50 工具的手动提示 50/129；25 工具 36/129，median 时间约低 29%。
- 图集成 50/129 对历史对照 55/129；21 次 semantic query 全空。多 Agent 50→51 不具决定性。
- 同配置重复 55→48，说明小幅均值变化不能解释得过满。

[747857：最新校准](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747857)把先前“有并发 B40 比顺序 L4”改为每 replica 仅一个活跃 Agent；约 B40 4.5 分钟对应 Kaggle 5.5 分钟。129 题 local cap 7 / 4.5 / 4 分钟分别 43 / 47 / 43，均单次采样。空 system prompt 在匹配条件节省约 15–18% 总 model tokens，但冲突增加，无明确修复率优势。这支持削减冗余指导语，不支持完全拿掉行为提示。

[747914](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747914)报告 0.2.13 中正 thinking_budget 会覆盖 include_thoughts:false；[747791](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747791)报告 32768 context / 14336 compaction / 16384 output 的请求仅有 2048 token 增长余量。二者是参与者报告，当前源码核对由主研究进行。它们提醒我们研究的是实际模型条件，而不是 YAML 标签。

## 7. 面向下一步实现的研究规格

给实现交接文档的建议增量（本报告不实现）：

1. **行为契约对象**：原样记录 issue 要求的输入子条件、预期输出/异常、边界值；必须关联当时可见 issue 文本，避免只复述泛化的“修 bug”。
2. **复现与路径证据**：T3 可调用短 trace，返回实际仓库函数/异常栈/是否触发预期差异，不输出大调用图。它应回答“当前最小输入是否真的进入相关路径”。
3. **精确行动**：保留 T2 span handle；研究单字符串调用入口以降低多参数编码失败。不要为了与公开代码一致改成宽松模糊匹配。
4. **训练样本**：每条包含 served prompt、候选动作、工具回包、下一决策、最终 outcome；重点保存“原假说被何种证据推翻”的节点。oracle demonstration 与真实 Agent rollout 必须分开标记。
5. **两个不同训练目标**：短代码修改/工具格式可以用 response-only SFT；语义修复需要更广仓库的真实可执行任务与转向数据。rank 8/16、LR 5e-5–1e-4 可作小规模起点范围，不能冒充已证实本赛最优参数。
6. **简洁框架先行**：公开 0.18 已说明无训练单 Agent 可有竞争力，公开 0.17 说明执行证据/工具机制可组合。既不把多 Agent 当前提，也不把纯提示优化当作冲击榜首的全部空间。

## 8. 可复核文件与访问边界

- [notebooks/](notebooks/)：8 份源码按 cell 展开；[raw/](raw/)：完整 MCP 原始响应与分数核实摘要。
- [retrieval-manifest.json](retrieval-manifest.json)：检索范围和证据分类。
- web.open 无法打开三个 Kaggle notebook URL；改用浏览器只读页面核实 score badge。没有因此猜造分数。
- get_competition_submission 返回 submissions.get 权限拒绝，只保留已公开可见 team/submission/score；未寻求绕过。
- 原始 v19 指定 versionLabel=1 的 MCP 读取返回 Not found；通过原始网页 V1 徽章和作者完整镜像分别取得分数与源码。
- 模型检索“gemma-4”返回大量 Gemma 2/3 及无关条目，检索精度不足；**“本轮未找到经核实的高分 adapter”不等于不存在**。
- 原始 notebook metadata 的 last_run_time 与搜索卡片不一致，研究日期使用本次 2026-10-10 获取日期和明确版本，不用该字段反推实验发生时间。

