# P006：官方约束、公开数据与真实训练路线更新

读取时间：2026-10-13（Asia/Shanghai）。本轮只读 Kaggle MCP 官方页面/讨论/模型目录，下载小配置、README 与两版小 wheel 以比较源码；未下载模型权重、未运行 GPU/评测/提交、未发帖、未读取隐藏任务或本地 heldout 原件。submission 57091482 和排行榜由主代理另行核查。

## 1. 决定性变更：12B 已成为合法且值得研究的基座

**10 月 11 日官方置顶公告已允许四个变体**：

- `gemma-4-31b-it-qat-w4a16-ct`
- `gemma-4-12b-it-qat-w4a16-ct`
- `gemma-4-e4b-it-qat-w4a16-ct`
- `gemma-4-e2b-it-qat-w4a16-ct`

同一提交的所有 agent/subagent 必须同一个变体，可以使用不同 LoRA。主办方确认小模型与 31B 使用相同硬件、12 小时总限制和运行预算，并指向 Google 官方 Transformers 未量化变体作为 LoRA 训练来源。当前模型目录中四个指定 CT 变体均为 v2。

重要文档冲突：本次实时读取的 Overview “Model Selection, Budget, and Harness Rules” 仍说只能 31B；更新的官方置顶公告明确扩大名单。不能继续把旧 Overview 当作禁止小模型研究的依据，也不能把允许四模型误读为可在一个提交里混用。

来源：[官方新模型公告](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287)，首帖 2026-10-11 11:48 UTC；Ryan 回复 3536992（同条件/训练权重）、3536939（12B服务参数）、3537606（最新wheel修复）。原始快照：[topic-748287.json](topic-748287.json)、[pages.json](pages.json)、[model-variations.json](model-variations.json)。

**12B 文件身份已读到，但训练基座还有一个真实的存档异常。**

| 项目 | 已核实内容 |
|---|---|
| 部署 CT | Google / gemma-4 / Other / gemma-4-12b-it-qat-w4a16-ct / 2；model.safetensors 10,264,229,896 bytes |
| QAT 未量化目录 | Google / gemma-4 / Transformers / gemma-4-12b-it-qat-q4_0-unquantized / 2；READY，Apache 2.0，小 config/README 可访问 |
| 普通 IT | Google / gemma-4 / Transformers / gemma-4-12b-it / 2；约 23.95 GB；不能静默冒充 QAT 训练源 |
| QAT 目录异常 | 文件 API 仅列一个 model.safetensors，3,096,444,928 bytes；总 3,128,668,544 bytes。官方卡称未量化为半精度、11.95B 参数，这些信息不能直接一致地解释为完整 BF16 权重，尚未加载或判断成因 |
| 架构 | 三份实际小 config 都为 Gemma4UnifiedForConditionalGeneration / gemma4_unified；48层、hidden 3840、16 Q heads、8 KV heads、global KV 1、head_dim 256、global_head_dim 512、MLP 15360、vocab 262144 |

**随后找到可用的官方替代来源（仅核实元信息，未取权重）**：Hugging Face `google/gemma-4-12B-it-qat-q4_0-unquantized`，非gated，固定revision `b6ed86275a6a5735884e208bfed95b445a684ca2`。其safetensors元数据为 **BF16 11,959,730,224参数**，单个模型文件 **23,919,549,408 bytes**，LFS SHA256 `26f2cee4292298a3f9f92209643c37c80e34e011381e22434088870d9439a0a0`。这是官方完整12B QAT训练源的明确元数据证据，Kaggle3.096GB记录异常无需阻塞CPU适配；未来下载按此固定来源和会话预算处理，仍不宣称本项目已加载训练。来源：[Google官方QAT仓库](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized/tree/b6ed86275a6a5735884e208bfed95b445a684ca2)，原始API快照：[models/12b-qat-hf-metadata.json](models/12b-qat-hf-metadata.json)。

HF固定revision的config已实际下载，与Kaggle QAT config逐字节相同，SHA256均为 `a323d02f68420f6fa3a3548130a0d36356075a4047a622e57148558f8eee7077`。但HF当前chat_template文件为18,683 bytes，Kaggle CT v2为17,466 bytes；不能跟随HF最新模板而静默改变部署协议。训练编码应显式固定实际部署CT模板与工具schema，再核对tokenizer指纹。元信息归纳：[models/12b-qat-source-summary.json](models/12b-qat-source-summary.json)。

当前训练入口仍需要真实适配：`training/train_sft.py` 写死31B/v2来源并要求model_type=gemma4，`training/memory.py`强制Gemma4ForConditionalGeneration；因此不能只改模型字段。已有31B NF4缓存不能重命名后拿来训练12B。

来源：[QAT目录](https://www.kaggle.com/models/google/gemma-4/Transformers/gemma-4-12b-it-qat-q4_0-unquantized)、[CT目录](https://www.kaggle.com/models/google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct)、[普通IT目录](https://www.kaggle.com/models/google/gemma-4/Transformers/gemma-4-12b-it)。快照：[12b-qat-files.json](12b-qat-files.json)、[12b-ct-files.json](12b-ct-files.json)、[models/12b-qat-config.json](models/12b-qat-config.json)、[models/12b-qat-README.md](models/12b-qat-README.md)。

主办方给12B的vLLM配置为 `--hf-overrides '{"architectures":["Gemma4ForCausalLM"],"model_type":"gemma4"}'`，并说明 scorer 已加入。随后仍出现 vision_embedder 权重加载错误，10月12日19:56 UTC官方回复最新 wheelhouse 已修复。这里仅确认官方说明；没有在本项目实跑12B或检查433MB vLLM wheel。公告提到 ParallelAgent，但随后主办方承认当前ADK的并发问题、认为迁移ADK v2风险太高；不能把并发当已验证的收益路线。

## 2. v31 的实质源码变化：分离验证超时

wheelhouse 已从 v30（10月9日）更新到 **v31（10月12日19:51 UTC，北京时间10月13日03:51）**。两个包名字没变，归档SHA变了：

- swegemma 0.2.11：119,520 → 119,670 bytes；v31 SHA256 `9ad876581f3acde8ec3c369858f0834e2b9f41a3cffce3ca42f09cb090aea9d3`。
- adk_submission 0.2.13：仍66,528 bytes；v31 SHA256 `f156f0bd84580ccb9334726d114d7a5481f16d3a6e2887cc4d276ced0ef8d99d`；逐ZIP条目比较内容完全相同，不能凭归档hash变化就声称逻辑更新。

swegemma只有5个Python文件变化：

- `budget.py: HarnessLimits.verification_timeout_seconds` 新增，默认300秒。
- `config.py: EvaluationConfig.verification_timeout_seconds` 新增平面入口；未显式传时继承 harness 的300秒。
- `cli.py` 新增 `--verification-timeout-seconds`。
- `context.py` 传播该字段。
- `harness/verification.py` 的Phase 2 pytest改用 verification_timeout_seconds；其值为None时才回退 command_timeout_seconds。

旧 `timeout_seconds` 仍映射 command_timeout_seconds（单个工具命令），不再同时决定验证pytest的默认上限。P006新的成对实验两臂应固定同一v31包及验证300秒；这是减少环境混杂，不是另外建设门控。当前0.13来自哪个服务版本还未查明，**不能将分数归因于v31或该修正**。

实际源差异：[wheel-source-diff.patch](wheel-source-diff.patch)、[wheel-source-diff.json](wheel-source-diff.json)、[wheel-manifest.json](wheel-manifest.json)。全量下载仅4个小wheel（两版各2个），未安装。官方元信息：[wheel-info.json](wheel-info.json)、[wheel-files.json](wheel-files.json)。

## 3. 为什么公开 FastAPI/Rich 结果不能代表0.13的正式私有仓库成绩

官方数据页仍是129个公开开发任务，来自 FastAPI/Rich/Requests/HTTPX，公开patch/test_patch供本地训练和研究。隐藏测试约120题，公榜/私榜约各半，来自私有仓库。**10月9日主办方明确：这四个公开仓库都不在隐藏测试集，隐藏测试gold patch验证100%通过。**

10月12日主办方进一步说明：

- 隐藏集有不同规模、不同领域的多个仓库，各仓库题数不等；
- 任务必须gold前失败、gold后通过；用强商业模型简单配置试跑，只保留成功或只差一个断言的题；
- 相同sandbox，但预缓存依赖wheel不同；
- 两个sandbox里被测包及其缓存测试依赖都可就地导入，agent可运行pytest；环境准备无网络超时拖延；
- 正式评分只接收补丁，验证pytest固定300秒，gold在其内完成；用户eval_config作用于agent阶段，不改变正式评分阶段的验证超时。

因此：本地暴露过的9题、旧环境FastAPI导入失败、公开集gold失败率都不能解释或预测隐藏集0.13。它们可以帮助研究工具语义、状态保持和修复动作，但应以外部仓库分组对照检验迁移，再将正式分数看作跨分布校验。不能因两个百分比碰巧接近就推断正式scorer有相同环境bug。

隐藏内容限制未变；没有公开完整仓库身份、准确任务分母或领域比例。官方仍说隐藏修复不触及async def、私有仓库async函数最多1.8%；故专攻公开FastAPI异步图遗漏不是当前强优先项。

来源：[官方数据说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)、[Gold/仓库不重叠说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747248#3535945)、[10月12日评分与sandbox答复](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748428)、[async说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745998)。原件：[topic-747248.json](topic-747248.json)、[topic-748428.json](topic-748428.json)、[data-summary.json](data-summary.json)。

## 4. 思考、压缩、技能与反馈：事实和未证实分开

- `include_thoughts=false` 现在管理跨轮携带的思考文本，不等价关闭thinking。Ryan在10月10日明确说该改动是有意的；关闭应显式 `thinking_budget: 0`，正预算独立作为研究变量。v31的adk逻辑与v30相同。
- 公开Overview仍允许官方9个工具、agent_tool子agent以及sandbox内run_skill_script/load_skill_resource。技能执行共享持久容器与预算；不能在host执行任意控制器。训练目前六工具协议是本项目选择，不是官方只允许六个工具。
- 压缩阈值14336以及摘要遗漏工具调用/回包有公开讨论；10月13日对阈值的答复来自普通COMMENT，不能作为新官方承诺。没有查到允许自定义scorer摘要器的新官方接口。v31小wheel也未改compaction。
- 12小时包括agent与sandbox准备，不包括最终补丁验证；主办方10月10日确认完整隐藏集在同一次运行生成补丁、队列时间不计入12小时。未完成题自动记0而非全提交超时的改造，主办方说可能不做。
- 仍为每天最多1次提交，并另有1个pending限制。没有在本轮所读新讨论中找到开放隐藏逐题trace或任务级反馈的官方更新；请求日志帖仍无答复。API只读成绩信息不意味着能看到隐藏执行轨迹。
- 社区“0.18上限只能LoRA突破/榜首都训练”的帖子是猜测，没有提交产物对应证据；不能拿来声称训练必然超过当前prompt路线。

来源：[thinking官方确认](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747914#3536450)、[压缩讨论](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747186)、[新阈值问答](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748584)、[预算官方回复](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747519#3536449)、[日志请求](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747763)、[LoRA猜测帖](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748051)。

## 5. 一条具体路线：12B上的完整修复事务与恢复行为SFT

研究假设：在统一12小时预算下，较小的合法12B让完整10K–16K修复前缀训练及更多有效推理轮次更可行；能否抵消其较低的基础推理能力，必须用实际成功修复数/耗时回答。官方卡的LiveCodeBench v6为12B 72%、31B80%，这不是本赛SWE分数，只说明存在能力—成本权衡，不能宣称12B一定更好。

1. **复用现有来源，不先扩大下载。** SWE-smith已固定分片3012行、1255上游resolved行/105来源repo，抽20条跨repo，其中7条结构化。7条的规范repo、完整commit、根LICENSE已补齐；优先用BSD来源click的短完整轨迹，以及pydicom/markupsafe恢复链。
2. **CPU补真实工具语义。** 在对应repo/base状态重放有差异的动作，保留实际观察。bash不能只改名；编辑匹配规则、view范围/截断、create覆写都不同；末尾submit缺回包不能伪造。失败动作保留上下文，仅监督已明确成功的后续纠正。可先形成最小可用的“定位→编辑→验证”真实完整事务，并单独保留恢复样本。
3. **重算正式训练长度并扩大独立repo。** 现20源对话≤8K仅1、≤16K6；5次错误来自2题，markupsafe两错误共用一次成功，不能重复算恢复。三个pydicom后续成功完整前缀约10,734/11,796/16,052，markupsafe约16,775，且尚未带官方schema；不截断历史凑8K。形成首批真实规范样本后，再从已下载分片的其余成功repo按身份/许可列筛选，扩独立任务，train/select按repo分组。
4. **适配12B训练加载后做一个真实小规模实验。** 采用已固定的Google官方HF完整BF16 QAT来源，支持unified架构、LoRA模块和导出CT对应；目标先得到能加载且确实训练过真实事务的adapter，再与12B base、31B当前候选比较相同时间预算下的外部仓库修复成功数。GPU和正式提交仍按主会话预算执行；本轮没有启动。
5. **把结果沉淀为有出处的资源。** 公开资源方向可以是“保留完整因果历史、可重放的跨工具修复/恢复事务”，但是否新颖、是否优于现有robust-edit和SWE轨迹工作，需要实际重放与修复增益。不能只交格式转换器或更多校验数。

这条路线允许先并行推进CPU数据/架构准备与现有31B推理改进，不因12B权重异常停掉全部研究，也不因小模型开放而抛弃已有31B基线。

规则明确允许公开、对参赛者平等可获得的外部数据用于开发/测试；训练来源与原代码许可记录分别保留。SWE-smith数据卡当前仍为MIT、tool split24100行，但不能把不同表示相加为独立题数；7条repo根许可不等于每个触及文件都已核查。SWE-Gym491条精简SFT缺精确身份/成功映射，R2E的数据许可仍未确认，当前不优先花转换成本。来源与历史实测：[P005外部数据计划](../../005/data-expansion-cpu-20261011/external-source-plan.md)、[身份/许可补全](../../005/data-expansion-cpu-20261011/structured-source-provenance.json)、[SWE-smith官方卡](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/blob/main/README.md)、本次规则快照[pages.json](pages.json)。

## 6. 金牌与奖金简表

本次MCP参赛队数2610，awards_points=true。主赛现金仅前三名37,000/18,000/10,000美元；金牌与现金不是同一门槛。最终名次按私榜，截止北京时间12月3日07:59；不从当前公榜推算固定金牌分数。官方奖牌计算页本次只返回空前端页面，未新增一个“已核实金牌名额”的数字。

论文赛截止北京时间11月13日07:59，3000词上限，原创未发表；Novelty/Quality/Relevance/Verifiability/Clarity等权。奖项15,000最佳论文、10,000新资源、10,000新应用，论文赛与主赛现金独立。可复现的真实修复事务资源与跨基座实验证据有机会支撑论文，但仅有工程适配并不自动构成新研究。

来源：[主赛时间/奖金](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview)、[论文赛](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)、[competition.json](competition.json)、[paper.json](paper.json)、[paper-tracks.json](paper-tracks.json)。

## 本轮边界与下一步依赖

已经核实：小模型资格、官方HF完整12B BF16 QAT元数据、v31真实小包源码差异、隐藏评测的公开分布/环境事实、现有外部来源的可落实路径。尚未证实：12B实际权重加载、当前vLLM加载成功、12B真实长序列容量、外部轨迹官方工具重放、任何新SFT收益。只下载了小资料并写此目录，没有训练、提交或消耗GPU额度。

