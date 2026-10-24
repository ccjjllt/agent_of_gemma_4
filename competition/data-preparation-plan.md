# 四个 adapter 的训练数据准备方案

2026-10-24（Asia/Shanghai）。沿用用户选定结构：G=r128、文本q/o+MLP五投影；S/F/R各r32、七种文本投影，训练计算与adapter导出均BF16。

**根据用户选定的Verified Focal Agent RL，当前先做96题的小批采集，研究真实六工具冷启动、同状态关键动作与多步恢复、当前策略rollout及预算信号。** 原1,400题改为采集产率成立后的扩展目标；成功SFT轨迹数量不能代替已验证动作对数量。主线见 [vfarl/README.md](vfarl/README.md) 及 [数据契约](vfarl/data-contract.md)。

首批96题现已补齐来源与机制审阅：G36/S20/F20/R20，43个canonical仓库；S/F/R各五个子机制均4题。81题来自Smith-py、13题来自SWE-Gym、2题来自SWE-rebench-V2，均固定完整执行commit。已实际推进官方工具和验证器的CPU重放，成功示范经官方Gemma模板、推理prefix和assistant loss mask导出；当前本地结果及剩余数量见 [96题清单](vfarl/pilot96/manifest.json) 和 [实际重放](vfarl/replay/replay-summary.json)。这是人工off-policy冷启动材料；没有模型on-policy数据、已验证动作对或GPU训练，本地Subprocess后端尚不等价于比赛Docker环境。最初48条来源轨迹仍按历史种子记录在 [vfarl/seed-inventory.json](vfarl/seed-inventory.json)，不与96题混加。

机器可读预算、子机制和证据字段见 [data-preparation-plan.json](data-preparation-plan.json)。来源原始统计及指纹见 [sources/data-preparation/provenance.json](sources/data-preparation/provenance.json)。

## 1. 分别需要多少数据

分开记三本账：

- **独立任务**：issue/PR或修复问题，经精确/近重复修复家族去重。同题多次采样、改写题面、轨迹分窗都不增加独立任务数。同一snapshot可有不同问题，但相关修复链和snapshot组必须保持用途隔离。
- **成功轨迹**：真实操作和观察，加可追溯的验证。先每题选一条成功轨迹；多次成功另记，不能拿轨迹行数冒充任务数。
- **assistant监督tokens**：实际进入loss的助手说明和工具调用。system/user/工具输出默认只作条件输入，不参与loss；输入上下文与GPU处理tokens另记。

### 后续扩展：1,400个独立任务（原SFT算术方案）

下表保留为后续成功过程课程的规模刻度，不是当前FocalRL首批任务或已获得库存。当前先按G主责36、S/F/R各20，共96题，采192条基线attempt、最多128个失败checkpoint和512条续跑，观察完整结果对的真实产率。

| 数据池 | 新增独立任务 | 默认成功轨迹 | assistant监督tokens预算刻度 |
| --- | ---: | ---: | ---: |
| C 共同修复课程 | 400 | 400 | 1.6M |
| G 广覆盖增量 | 400 | 400 | 1.6M |
| S 专科增量 | 200 | 200 | 0.8M |
| F 专科增量 | 200 | 200 | 0.8M |
| R 专科增量 | 200 | 200 | 0.8M |
| **全局去重总量** | **1,400** | **1,400** | **5.6M唯一监督tokens** |

换成每个adapter的课程：**G=400共同+400增量=800题；S/F/R各为400共同+200专科=600题。** 共同课程在三位专家之间复用，不是三份新的400题。G共同训练按r128五投影进行；专家共同训练按r32七投影另外进行一次，再复制r32状态给S/F/R。

这里使用每条成功轨迹约4k assistant监督tokens作算术刻度，并非已测到的数据集平均值。先对100条候选用实际Gemma tokenizer与loss mask统计，再调整预算；不能为了凑tokens延长解释，也不能从这些数直接推GPU小时。

专科续训可按70%专科、30%通用监督tokens起步：每位专家新增0.8M专科tokens，对应约0.343M共同重放，总续训约1.143M。计入G、一次专家共同起点与三条分支，首轮优化器监督暴露约 **8.229M**；它不同于5.6M唯一数据tokens。该比例是待比较的起点。

### 更大扩展：约6,000个独立任务

C增至1,000题，G额外2,000题，S/F/R各额外1,000题。G累计3,000题，每位专家累计2,000题（含相同的1,000共同课程）。按4k刻度，全局24M唯一监督tokens；加共同训练复用与70/30重放，优化器暴露约33.143M。

扩展以首轮实际结果和筛选产率决定。目前尚不知道源池中最终有多少合格S/F/R样本，不能将6,000目标写成已有供给。r128提供更大容量，不构成“必须四倍数据”的定律。

## 2. 数据从哪里来

### 先取现成成功轨迹

以下数字来自本轮固定revision的Parquet元数据列统计，仅扫描ID、成功标记、模型等列，没有全量下载轨迹正文；成功是来源环境的结果，不是已在本比赛框架复现。

| 来源 | 所选轨迹行数 | 成功轨迹 | 有成功轨迹的不同任务ID | 用途 |
| --- | ---: | ---: | ---: | --- |
| [Nebius SWE-rebench OpenHands](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) `train` | 67,074 | 32,161 | **3,792** | 优先真实issue过程池；G定位/工具行为与各专科候选 |
| [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) `tool` | 24,100 | 9,427 | **5,497** | 广仓库合成故障与机制候选；仅选一种工具协议 |
| [SWE-Gym SFT](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories)及其原始runs | 6,055原始runs | 491 | **294** | 小规模真实过程补充；SFT导出只有messages，须另保留任务身份 |

前三行跨来源的重叠尚未排除，**3,792和5,497不能直接相加成独立可训练任务数**。还要排除比赛四仓/保留任务、语言不符、协议无法转换、超长或机制不匹配的样本。

Nebius新版由Qwen3-Coder-480B与OpenHands0.54采集，结构化工具调用较适合转换。其平均64.3 turns、报告训练长度131,072，本比赛上下文上限32,768，因此不能直接照搬完整数据或其训练token量。Smith的tool/xml/ticks是不同格式，不应相加当三个独立训练集；旧README仍写最初5,017条，与当前文件规模不同。

许可来源要分开保存：Smith轨迹MIT，Nebius新版CC-BY-4.0；还需保留原始仓库与教师输出条款。Gym的MIT明确见SFT导出卡，raw卡本轮未声明许可，不能拿项目代码的Apache许可替代。首轮不把未澄清的raw许可当已解决。

### 专科缺口再用可执行任务池补

| 任务池 | 本轮核实规模 | 如何使用 |
| --- | ---: | --- |
| [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2) `train` Python | **7,243**（全部语言32,079） | 真实issue/PR、base_commit、金补丁、验证补丁、依赖配置和镜像；按S/F/R缺口选题后获取实际修复轨迹 |
| [SWE-smith-py](https://huggingface.co/datasets/SWE-bench/SWE-smith-py) `train` | **50,908** | 合成故障补特定机制；保留生成方式和原始代码来源 |
| [SWE-Gym任务集](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) `train` | **2,438** | 11个Python仓库的真实任务；便于先摸清环境恢复与采样成本 |

**SWE-smith-py的`patch`是制造缺陷的diff，不是修复答案。** 官方[验证文档](https://swesmith.com/guides/harnesses/)明确区分缺陷补丁与预测修复。应使用其真实成功修复轨迹，或在缺陷分支重新修复并验证；若反向处理diff也必须核对版本与执行结果。不能直接将任务行的patch拼成SFT回答。

V2卡明确只排除了原SWE-bench仓库，并未过滤所有其他benchmarks，所以它不是天然无污染训练集。旧`princeton-nlp/SWE-bench`虽有19,008个train记录，但数据卡许可和完整执行环境保障不足，首轮不计为现成训练产能。旧SWE-rebench V1的split叫test/filtered，不自动作train摄入。

固定revision、列统计方法与未知项分别见 [轨迹核查](sources/data-preparation/trajectory-sources.json) 和 [任务核查](sources/data-preparation/task-sources.json)。

## 3. 怎样确认每题匹配它要训练的能力

这些来源没有现成的G/S/F/R标签。题面关键词、仓库名称、patch长短只能召回候选，不能确认机制。

每个专科细分五组，首轮每组约40题作为覆盖目标；不足时登记真实缺口，从任务池或原仓PR继续补，不以相似主题的题塞满配额。

| 专家/子机制 | 入池所需的具体证据 | 应与相邻机制区分 |
| --- | --- | --- |
| S1 编码/解析/序列化 | 变换前后有值、长度、线格式或往返语义差异 | 版本改变下游编码契约，可能主标R |
| S2 单位/坐标/跨度 | 字节、码点、grapheme、cell、span的边界与对应关系 | 仅默认宽度常数变化通常交G |
| S3 集合/多值/别名 | 顺序、重复项、基数、字段命名空间在转换中保真 | 共享列表被其他请求污染是F |
| S4 扫描/切分/遍历 | 终止、进度、覆盖或宽度/数量守恒的反例 | 没有结构推理的一次if变化通常交G |
| S5 嵌套结构/Schema | 树、语法、嵌套映射的合法变换与重建 | ForwardRef的运行时求值是R |
| F1 参数/配置传播 | 入口→中间层→消费者的缺边或覆盖优先级 | 单函数默认值变化不算跨层 |
| F2 引用/所有权隔离 | 真实对象身份、共享修改、复制时点与污染路径 | 纯容器表示变换是S |
| F3 缓存/作用域失效 | 操作前后序列、缓存键或跨请求隔离反例 | 单纯对象能力识别是R |
| F4 资源/退出/取消 | 资源所有者、使用期、yield/取消及释放时点 | 只识别async generator是R |
| F5 启停/流式/事件 | 生产、消费、结束的时序因果链 | SSE字符串转义是S |
| R1 注解/命名空间/泛型 | globals/locals/ForwardRef等导致真实反射差异 | Schema字段命名变换是S |
| R2 callable/wrapper/partial | 类、实例、装饰器和真实被调对象的区别 | 同一资源退出早晚是F |
| R3 对象协议/代理/MRO | 声明与能力、动态属性、继承分派之间的真实区别 | HTTP编码本身是S |
| R4 Python/依赖版本 | 支持版本之间的契约变化及明确适用条件 | 只有requirements数字变更不算 |
| R5 导入/运行时可用性 | 导入顺序、lazy import、可选模块等因果关系 | 网络/pip安装失败是环境问题 |

每题建立一条简短机制证据链：**可见触发条件 → base中的错误行为 → 负责符号和因果改动 → 区分错误修法的观察 → 修复后的验证证据**。标签依据实际生效的生产改动与需求，不把金补丁中所有附带变化都视为必要能力。

例子只用已研究的公共题解释边界，不作为训练输入：`requests_6589`的字节长度转换偏S，`requests_6757`必须按urllib3版本选择语义偏R；`fastapi_14301`少传scope而改变资源退出时点偏F。出现“async”或“JSON”本身不能替代这些证据。

数据可信度分开标记：

- **来源验证**：保存publisher的resolved/test_result及版本，是原环境成功，不冒充本轮复现。
- **目标工具已重放**：转换后用实际工具执行，观察和结果可追溯，才能称本协议成功过程。
- **只有patch/静态解释**：可作辅助机制或代码编辑材料，不计作完整成功agent轨迹。

需要本地确认时使用已有上游测试和必要的一次性Shell/Python命令；不新建专用测试文件、tests目录或check/gate脚本。安装/架构/网络失败单独说明，不作为技能负例；无论元信息或采样模型怎么宣称，都不伪造执行成功。已有验证冲突未解决的样本不能充当高可信成功过程。

## 4. 统一成比赛实际使用的训练协议

主候选是SWE-agent/OpenHands数据，目标是Gemma+ADK，转换是实质工作。

1. 保留原始轨迹ID、hash、教师、验证出处，另外产生目标协议副本。 同名instance_id连接任务表时还要核对任务版本、镜像/bug分支和补丁hash，不把旧Smith轨迹盲目接到新版任务定义。
2. 对齐实际工具schema：OpenHands持久shell/poll/editor与本比赛run_command/read_file/edit_file/submit_patch不能仅靠改名字等同。cwd/env、行号/截断、替换唯一性与错误返回都需要正确处理。
3. 不支持或不等价动作要实际重放获得真实观察，或剔除该完整过程。不能补写不存在的status/exit_code，也不能把源端工具输出伪装成本工具实跑结果。原材料可降级为静态辅助资料。
4. 以真实Gemma tokenizer/chat template序列化工具调用，不把Qwen XML或SWE-agent旧协议直接塞成目标格式。只对assistant目标做loss，观察作为输入。
5. 超长轨迹按决策点选择此前真实证据或重新采样短过程；摘要只来自当时前缀，不能偷看未来patch/结果。分窗后保留完成任务所需证据和来源链接，不能直接取前32k并声称完整成功。实际训练窗口由显存方案确定，上限不超过32,768总tokens。
6. 不把源轨迹修改测试/运行配置的行为学成评分策略；本比赛会保护这些文件。新图工具调用和多agent行为若源数据没有，必须另采真实调用，不能从旧搜索日志编造。

G还需学习**何时自己修、何时委派、给专家哪些已知证据、何时接回**。现成单agent轨迹不自动具有这些行为。G增量400题中可安排约100题在真实任务环境采集handoff/不handoff过程；这是400题的组成部分，尚未获取，也不在本轮启动付费或GPU采样。未取得真实协作日志时，先用提示词定义路由，不声称已经通过SFT教会路由。

## 5. 去重与隔离

129公共开发题本轮继续用于能力画像和开发比较，不进入这份训练计划。第一轮训练来源按canonical repo排除FastAPI、Rich、Requests、HTTPX及fork/历史别名，以便观察跨仓迁移。

先通过允许的ID/来源清单确定用途，再抓取正文或生成轨迹；不为查重读取heldout题面或解答。按原issue/PR、revert/backport/cherry-pick、近重复代码修复家族去重；相关snapshot/修复链整组留在同一用途。SWE-smith的swesmith/fork与合成ID要追溯到原始repo和commit。外部独立开发比较材料按仓库隔离，不能把同一函数的近邻变异分到训练和比较两边。

保留多标签，分配主标签用于采样；不能一题打S/F/R三个标签就算三题。先按独立任务扩多样性，再考虑同题多策略轨迹。

## 6. 扩展到1,400题前的准备工作

1. 固定上述版本与许可、元信息，先做四仓/预留ID排除及跨来源去重。
2. S/F/R各抽20个候选，逐条人工建立机制证据锚点；通用池同时检查定位、修改、恢复和提交过程。
3. 用100条候选测实际Gemma长度、动作转换保留率和来源成功证据。需要恢复环境时按repo/镜像批量复用缓存，不全量拉数百Docker。
4. 先完成96题的小批采集，按真实命中率、环境恢复率、协议可保留率与已验证对产率，决定是否补足400共同+400G+3×200专科的扩展目标。记录已获得数量和缺口，不能将候选总量当完成量。
5. 先比较G单独与专家系统的真实修复收益和时间成本；之后决定是否扩到6,000题。loss下降或标签看起来准确都不能单独证明有效。

训练数据、完整镜像和日志不放入比赛submission；现有2.949GiB纯adapter计划剩余空间仅供头信息、配置、提示词和skills。数据准备与训练费用另按后续会话授权和预算执行。

## 7. 本轮实际完成与尚未知

完成：四adapter课程/数据量设计；96题逐题固定来源、代码身份、许可证据、因果机制及原始F2P/P2P，重点四个子机制各4题；运行原版官方六工具、路径保护、干净Phase B和JUnit验收；成功的人工示范按真实ADK声明与固定Gemma模板导出推理prefix及loss mask。48条来源种子、原生候选整理与字段长度测量单独保留。论文机制核对及采集契约仍用于后续Focal/预算数据采集。

实际样本发现：Smith种子18个非空`row.patch`均跨仓错配，6个为空；其中一条还从固定SHA的原始Parquet复核，确认不是datasets-server视图问题，因此整列隔离。Nebius的24条种子中23条`model_patch`含辅助debug/test/issue等产物，不能直接当生产修复正例。来源质量审阅见 [vfarl/sources/seed/source-content-review.json](vfarl/sources/seed/source-content-review.json)。

尚未完成：96题的全部环境实例化、依赖/image digest冻结和成功过程；比赛Docker环境下的等价复现；其他system prompt/thinking预算的版本化导出；当前策略失败及同状态动作分叉。现有成功示范的真实Gemma监督tokens已统计，但不能外推全部96题平均值；采集保留率、已验证动作对产率和训练净收益仍未知。parse保护冲突、Freezegun必需节点skip等实际问题已单独记录，没有靠删required凑成功数。没有整理出1,400题训练集，没有新模型采样或训练，也没有读取隐藏/预留评测内容。
