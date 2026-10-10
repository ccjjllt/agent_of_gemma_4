# 005：推理机制与LoRA实现交接

日期：2026-10-10。配套[研究决策](005-medal-and-prize-research.md)。本文保留原始实现规格。后续已完成single_state载体、技能接线、真实编码/SFT/导出，以及31B四卡LoRA诊断训练；实际命令以[载体说明](../../notebooks/005-single-state.README.md)和[训练说明](../../training/README.md)为准，验证边界见[最新进展](../progress/005-medal-and-prize-research.md)。已完成官方 CT 上诊断 adapter 的加载、路由与工具格式兼容验证；尚未完成真实修复语料 SFT、修复收益配对或比赛提交，DPO仍后置。最新数据瓶颈、合并归档及后续资源安排见[下一轮计划](005-next-round-sft.md#合并审查后的下一轮工作2026-10-10)。

## 1. 先实现什么

两条线可独立并行：

- **推理线：**T2最小编辑能力 → T3行为约束和两种假设 → 按需执行路径证据 → 单次真实交接。
- **训练线：**公开轨迹统一格式 → 真实局部修复/恢复片段 → Gemma工具协议编码 → 一个LoRA SFT → 独立PEFT导出。

不要为了训练而等待所有推理实验结束。第一批数据先以当前可部署的官方工具协议为主；后续T2/T3接口稳定后，新样本用明确protocol_id区分，首个adapter只训练一种选定协议，避免混淆调用方式。

下列内容不在首版：多角色辩论、第二个在线模型、全仓神经向量索引、从零搭RL集群、动态host callbacks、自定义评分器、全量通用测试框架。完成一个可运行研究版本比扩展这些功能更有价值。

## 2. 继承哪些现有资产

| 资产 | 路径 | 用法 |
|---|---|---|
| 当前冻结对照 | `research/002/variants/t0/agent/` | 当前无adapter基线；不要误用根`agent/` |
| 导航技能 | `research/002/variants/t1/agent/skills/repo_navigation/` | 提供符号/范围读取能力；保留历史冻结文件，新候选中再改 |
| 精确编辑原型 | `research/002/candidates/t2_span_edit/agent/skills/source_edit/` | 复用inspect/span_id/apply及哈希校验，不再发明新的模糊匹配器 |
| T3短提示 | `research/002/candidates/t3_contrastive_repro/` | 扩展为修复层次与两个假设，不嵌公开题答案 |
| 分层读取原型 | `research/002/candidates/t4_layered_context/` | 取outline/locate/完整范围读取，服务当前决策 |
| 历史行为索引 | `research/002/candidates/training_prep/fragments_index.json` | 回查真实片段的起点，不能当作完整训练数据 |
| 联合评测载体 | `notebooks/002-joint-explore-single-cell.py` | 复用单次模型加载/逐题归档思想；新版本另存，不覆盖旧冻结结果 |
| 新版接口源码 | `research/005/official-20261010/wheels-source/` | 阅读0.2.13/0.2.11实际可提交能力，避免照搬旧版限制 |

已有T2/T4是工作区未提交原型，没有本地模型效果。冻结T1及T2复制的导航read_range会直接切断序列化JSON；在新候选中改为完整JSON预算即可，不因此扩建全套测试工程，也不悄悄改变冻结T1。

建议新文件组织（目录为待创建，不是本轮已有代码）：

```text
research/005/candidates/
  single_state/agent/               # 单Agent+行为约束+当前状态
  one_handoff/agent/                # 同工具，一次研究→执行交接
  source_context/                  # 共用源码、状态、执行路径组件
training/
  data/{catalog,canonicalize,transactions,split_registry}.py
  encoding/gemma_tool_chat.py
  configs/sft_qat_r8.yaml
  train_sft.py
  export_peft.py
  preferences/                     # 第二轮再实现
```

每个候选使用完整自包含agent目录，以对应比赛压缩包结构；共用组件可以开发时引用，但打包时应包含真实脚本，不依赖仓库外路径或symlink。

## 3. 推理工作包

### I1. T2接入与不可变编辑事实（较小实现）

接口保持：

```text
inspect(path, start_line, end_line)
  -> span_id, file_sha256, exact_text, actual_range, truncated
apply(span_id, replacement)
  -> status, compact_diff, post_sha256, patch_id
```

- source_edit已有大部分实现。模型只生成replacement；失败后回读，不猜旧串、不扩大替换。
- 完整源码、该组件执行得到的原始输出与diff由工具保管；模型写的“已修改”不作为修改成功事实。普通skill不会自动拦截原生read_file/run_command回包；需要保存的观察通过自己的inspect/probe包装入口捕获，其他观察由record显式传入并标注来源，不能冒充自动取证。
- 可增加最多3–4份同题diff快照，记录parent_patch_id与文件哈希。恢复由明确动作发起，不自动选择所谓最佳补丁。
- 用单个run_command字符串或稳定的skill入口传输参数，避免多层转义；比较调用协议时单独记录它的变化。不要抄公开工具的skip-worktree测试隐藏动作。
- 状态写入本题沙箱临时目录，绑定真实任务工作区；不进入最终patch。路径与题目身份复用T2已有逻辑，接入时按实际runner映射处理。

交付是一个可供Gemma连续调用的编辑skill；不是大型编辑器，也不把编辑成功率当修复率。

### I2. 行为约束和两假设状态（首要算法工作）

在准备修改或原假设被推翻时更新一次，避免每步维护长表。建议文件`scripts/working_state.py`提供`record/show`，不自行推断根因。

```json
{
  "schema_version": 1,
  "issue_contract": {
    "trigger": "题目中明确的输入/配置条件",
    "expected": "预期返回、异常、状态或格式",
    "preserve": "应保持不变的相邻正常行为"
  },
  "facts": [{"id": "f1", "source_ref": "span或原始工具回包", "claim": "由该证据支持的事实"}],
  "hypotheses": [
    {"id": "h1", "layer": "producer|transformer|consumer|boundary", "claim": "...", "support": ["f1"], "contradiction": []},
    {"id": "h2", "layer": "...", "claim": "...", "support": [], "contradiction": []}
  ],
  "chosen": "h1",
  "unknown": "哪一条未知事实会改变选择",
  "next_action": {"kind": "inspect|probe|edit", "target": "...", "purpose": "..."},
  "current_patch_id": "..."
}
```

行为流程：

1. 根据issue提取契约，已有信息不足可以留unknown，不能为填表虚构预期。
2. 从当前代码识别产生/转换/消费关系，保留最多两个合理解释；只有一个明显解释就直接推进。
3. 若两个解释都成立，选择一次信息量高的读取/轻量探针；若已足以支持修改，直接用I1编辑。
4. 实际回包与预测不符时更新或舍弃假设，下一动作必须回应这个差异；不能只是改一个复现脚本文件名。
5. 局部检查只说明对应行为，完整修复仍由最终结果回答；普通打印success不能自动标记契约满足。

初版system提示只增加这几个规则，控制在短段落；不把结构JSON要求变成每轮冗长模型输出。工具可以只展示changed_fields和当前next_action。

### I3. 决策导向源码读取与按需路径探针（中等实现）

复用T4，入口足够简单：

```text
locate(queries[1..4], scope?, question)
inspect(targets[1..3], question, mode="exact|outline")
probe(command, question, timeout_seconds)
```

`question`说明当前要决定什么，不要求模型展开长推理。locate返回少量候选及命中依据；inspect返回完整小函数或可继续展开的范围。截断片段不能签发可编辑句柄。关系边来自源码引用时标textual_reference，实际运行捕获时才标observed_call。

probe只在需要确认触发条件/执行路径时使用。可借鉴公开v19的Python sys.setprofile/threading.setprofile，过滤到任务仓库，输出有序函数、调用数和短异常栈；模型自行决定下一步。

```json
{
  "observation_id": "...",
  "question": "...",
  "workspace_patch_id": "...",
  "exit_kind": "ok|assertion|exception|timeout|setup_error",
  "calls": [{"path": "...", "function": "...", "first_line": 1, "count": 1}],
  "assertions": [{"name": "...", "observed": "..."}],
  "stdout_tail": "...",
  "omitted": true
}
```

只有探针实际具名断言才填assertions；退出0不能机械推断issue已复现或修复。调用函数命中也不能替代根因推理。脚本放临时目录；路径追踪不收集宿主文件、不输出庞大完整调用图。无需为了这个组件补齐整仓依赖，更不每题强制运行。

### I4. 一次真实交接（第二个推理版本）

复用当前声明式SequentialAgent/AgentTool能力，所有LlmAgent仍使用指定31B基座。schema已确认有output_key、include_contents:none和agent_tool.skip_summarization；初版不设计未被提交接口支持的host回调。

推荐三段职责：

1. **入口保存问题。** 探索者在带原始任务的上下文中读取问题，通过record_issue把可见问题文本及必要预算字段传给skill；不能凭空使用`{problem_statement}`等未确认变量，也不能让skill读取宿主tasks答案。此路径的问题文本由模型传入，不宣称是host自动逐字节复制；保留其实际传入内容和来源，长问题不要无声摘要成一句话。
2. **工具生成工作状态文件。** skill在本题沙箱临时目录写handoff.json：issue字段来自上一步，代码从已登记路径/范围重新读取，diff从当前工作树采集；假设/反证/下一动作由探索者显式传入。普通skill不写ADK session.state。探索者的最终文本只输出handoff文件handle/路径，由output_key保存为repair_context；output_key保存的是该最终文本，不是自动执行的JSON合并器。
3. **执行者使用新上下文。** include_contents:none，在指令中引用repair_context中的handle，首个动作通过同一个skill读取handoff.json；取得原issue、真实源码和diff后执行。自己的新工具调用和回包需正常进入后续动作。最多这一次交接，不回流辩论。文件保存在共享题目沙箱，不能放在每次调用即回收的skill临时释放目录。

上下文包含原始issue全文；关键函数保持原文；只压缩已经排除的搜索。初始目标是紧凑到约2–4K tokens的工作状态，但这是建议值，无法完整容纳时应保留必要代码并减少无关叙述，不强行把关键分支截掉。

注意：配置支持none只证明有入口，不自动证明实现已正确传递原issue及执行阶段新观察。实现者应查看实际消息完成接入；若最终只是增加状态卡，名称与研究记录如实写状态卡。研究不需要为此重写runner。

软时间安排起点为探索约总时长1/3，剩余给执行；这是prompt软目标，不是host强制保证。总工具/时间与单Agent共享，不给每个角色额外五分钟。可先用同sampling比较交接，再研究探索4096/机械执行1024等差异，避免混合解释。

## 4. 训练工作包

具体原理及一手来源见[训练研究](../../research/005/training-20261010/report.md)。本节给首版可开发交付，不要求先实现完整偏好学习或RL。

### D1. 数据目录与统一协议

`catalog.py`只需记录source_uri/license/revision/teacher/issue/repo/commit/content_hash；先小批取得公开SWE-Gym/SWE-smith轨迹，再扩量。维护公开外部、Gemma自采样、人工/teacher纠正和oracle演示四种来源类型。

`canonicalize.py`输出统一角色消息、tool_call_id、工具名、完整参数、回包和可见状态。工具作用相同但名称不同可映射；作用不同不能仅改名。若要训练本项目工具，应从相同初始仓库重放转换后的动作得到真实回包，或者在第一批只选协议兼容动作。tool_schema_hash和protocol_id用于明确训练分布。

`transactions.py`按完整因果片段切分，典型2–6个assistant动作：已获证据→修改→检查，或错误→回读/改变条件→成功动作。必须包含决定动作所需观察，不能剪掉错误却保留依赖它的后文；参考答案标注单独保存，不进入模型输入。

建议首批工程规模为500–1500个不同外部issue、1000–3000条片段、约4–12M实际处理tokens；这不是效果阈值。每题最多3条并限制单repo权重，数据不足就缩小规模，不靠重复切片达到目标。

训练数据来源的主办方明确答复：[允许开源teacher和现成公开非Gemma轨迹](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746046)。外部API发送比赛任务的特定问题尚无明确答复，首版可用公开外部题或本地开源teacher继续，不必等待该路径。

### D2. 明确训练与选择集

本地27个已暴露任务可用于诊断与行为设计。**一旦拿其中任务训练，原24题开发面板就不能再充当独立训练收益证据。** 保留28题最终heldout；从未暴露开发池或外部多仓库任务提前留出选择集，按issue/repo+commit和近重复分组。若目标检验跨仓库迁移，优先另留整repo，而不是只随机分片段。

这个工作只需一个split_registry文件，不建复杂门控系统。汇报全体结果及任务类别，训练loss不替代修复结果。

### D3. 单条样本与编码

```json
{
  "sample_id": "...",
  "origin_kind": "student|teacher|corrected|oracle",
  "group": {"issue": "...", "repo": "...", "commit": "...", "split": "train"},
  "source": {"uri": "...", "license": "...", "sha256": "..."},
  "protocol_id": "p005-native-tools-v1",
  "messages": [],
  "target_assistant_steps": [12, 14],
  "behavior": "hypothesis_revision|local_edit|tool_recovery|state_handoff",
  "observed_outcome": {"action_succeeded": true, "task_resolved": null},
  "offline_labels": {}
}
```

这里messages为空只为展示schema，真实样本必须含真实按序消息。`gemma_tool_chat.py`使用部署版本tokenizer/chat template和工具schema，生成input_ids、attention_mask、labels与片段元数据。

默认只有选定assistant动作算loss，system/user/tool观察和离线标签为-100。关键工具JSON不跨截断边界；长片段按完整交易分桶。过去未保存的thinking不人工补写，也不把缺失作为“应该不思考”的目标；若新增有真实reasoning的样本，单独记录其来源和监督权重，避免长思考压倒动作监督。

初始行为混合比例可用外部多样局部过程60%、真实恢复/反例25%、完整紧凑解题15%；它是研究起点，实际缺哪类就记录缺口。oracle动作最多只作有限格式热身，不作为语义推理的大头。

### T1. 一个SFT配方

```yaml
# 待实现的研究起点；不是已运行配置
base_family: gemma-4-31b-it-qat-q4_0-unquantized
base_revision: kaggle-v2  # 实际加载器须记录对应精确revision
training_mode: lora_bf16_or_same_source_nf4
target_text_modules: [q_proj, o_proj, gate_proj, up_proj, down_proj]
rank: 8
alpha: 16
dropout: 0
learning_rate: 5.0e-5
max_sequence_length: 4096
micro_batch_size: 1
gradient_accumulation_steps: 8
epochs: 1
warmup_ratio: 0.03
max_grad_norm: 1.0
gradient_checkpointing: true
loss: selected_assistant_tokens
```

module names必须从模型实际文本模块展开，不能误训视觉塔；K=V与打包键由导出映射处理，不假定每层都有v_proj。初始q/o+MLP是本项目建议；进一步加k/v、rank16、8192序列各自独立改动，不同时搜索全组合。

优先同源QAT未压缩权重。资源不足时在该权重上用NF4 QLoRA，记录NF4→CT量化差异；普通IT+NF4是备用参照，不改adapter_config名称冒充QAT同源。指定CT压缩权重本身不是通用QLoRA可直接训练格式。

算力首选可用的CUDA路径。48GB卡做QLoRA、80GB卡做BF16-LoRA或更宽裕QLoRA是资源推断，非本机实测；2×T4/4×L4分片可研究但不要把显存简单相加。完整吞吐/峰值显存由第一版实际执行回报，本轮不承诺时长、费用或提升点数。

### T2. 导出与部署

`export_peft.py`产物：

```text
adapters/repair_lora/
  adapter_config.json
  adapter_model.safetensors
training_manifest.json  # 研究归档；按允许扩展名/目录决定是否随提交
```

manifest记录QAT来源、量化后端、目标模块、rank、模板/工具协议、数据及adapter指纹。保留独立adapter，正式推理继续使用固定CT基座，不合并后替换基座。

训练条件关系要写清：BF16-LoRA是`W_QAT+ΔW`，同源NF4训练是`Q_NF4(W_QAT)+ΔW`，比赛执行是`dequant(Q_CT(W_QAT))+ΔW`。性能迁移需要实际结果回答；训练加载器上的loss或短生成不能替代CT工具任务表现。

第一版结果只需清楚分开：adapter未加载、推理异常、成功执行但未修复、修复成功。不为此搭一套新裁判，复用现有逐题归档。

### T3. 后续偏好训练（不用先做）

有同一消息前缀和源码状态下的真实chosen/rejected后，再实现DPO。优先错误回包后的决策，而非整条失败轨迹。chosen必须执行后继续生成，不能拼接另一条成功轨迹未来；只有都正确时才以耗时偏好短路径。

初始范围：300–1000对、beta0.05–0.2、lr5e-6–2e-5、0.5–1epoch，均为候选超参。共用SFT编码/导出，新增paired loss；在线RL单独分支，不阻塞首个SFT版本。

## 5. 实验只回答几个必要问题

无需给每个helper做完整测试矩阵。实现过程中用少量真实片段确认接口工作即可；研究主要记录终局修复与机制变化。

建议最多先做下列核心比较，而不是全部排列组合：

| 比较 | 回答的问题 |
|---|---|
| 同版本T0 vs 已有T1 | 原有导航技能是否被使用且有帮助；完成历史问题 |
| 共同编辑底座 vs +行为约束/假设状态 | 语义决策是否提高正确修复 |
| 同状态单Agent vs 一次真实交接 | 去除旧探索历史是否改善后续动作 |
| 完全相同工具/提示下base vs SFT adapter | 训练是否增加跨题修复能力 |

每次记录task、环境/协议版本、adapter指纹、最终patch、原始工具动作、实际预算和resolved即可。分析解释指标：假设是否被新证据更新、是否修改正确层、好状态是否保留、交接后的首个动作是否回应下一步。不要用更早编辑、更多提交或更低NLL代替终局结果。

复用现有联合载体节约模型启动；小探索用于筛明显方向，候选有依据再扩大。先独立观察各项，主赛最终可以组合，论文仍报告各增量。此次不要求重跑已完成0C，也不以无限preflight阻塞推进。

## 6. 开发者交付给研究者的内容

1. 可运行、自包含的single_state与one_handoff agent包，各自说明新增行为和入口。
2. canonical数据示例、split_registry、protocol_id及能追溯回原始观察的片段集；不包含heldout。
3. 首个SFT训练配置、运行入口、PEFT导出入口；实际开始训练后再补权重和资源记录。
4. 小规模结果表与新增成功/回退的机制案例；无结果的模块标“实现完成，尚未运行”。
5. 简短偏差说明：是否只实现状态卡、是否使用普通IT备用基座、是否缺少某类恢复样本。直接写事实，不以占位物冒充完整功能。

不需要交付大型门控系统、泛化到所有训练框架的兼容层，或一套为了报告通过率而写的形式化测试。首要交付是让研究问题真正可运行、可解释。

## 7. 可直接转交的任务描述

> 在不覆盖P002冻结变体与历史结果的前提下，实现P005两条研究线。推理线复用T2/T4，增加issue行为约束、最多两个带来源的修复假设、按需仓库执行路径观测及一次真实研究→执行交接；先交单Agent版，再交同工具同总预算的交接版。训练线将公开SWE轨迹统一为本项目工具协议，导出完整局部修复和真实错误恢复片段，使用官方同源QAT未压缩基座完成rank8 LoRA SFT与独立PEFT导出。所有新目录和接口按本文实现；大工程分包交付，优先可运行版本，不扩建门控/测试平台。训练、GPU运行和正式提交按用户后续资源安排执行；实现阶段如没有实测结果，明确记录未运行。

## 8. 每周30小时配额下的训练资源交接（2026-10-10）

用户已授权用Kaggle CLI做训练探查。私有CPU准备会话`ccjjllt/p005-training-cpu-stage`已完成：Python3.13.15、stock torch2.11.0+cpu，Transformers5.13.1/PEFT0.21.2/accelerate1.15.0/bitsandbytes0.50.2及官方tokenizer导入成功。官方QAT未量化v2的两个权重分片完整，权重文件合计62,546,338,248字节。训练需要的7个wheel已作为该CPU Notebook输出保存，后续用`kernel_sources`挂载。不要安装比赛推理用41个wheel，也不要在GPU会话重新下载torch或模型。

现有`training/train_sft.py`仅支持单设备；默认BF16并不适合单L4，NF4路径也将全模型放到单GPU。普通DDP不会合并显存。官方4×L4会话按两倍额度扣费，因此30额度小时约对应15小时L4会话；实际账号余额以`kaggle quota --format json`为准，必须给推理实验留预算。

GPU探针载体及完整证据保存在`research/005/kaggle-training-probe-20261010/`。只有服务器1200秒、训练子进程900秒的一次短探针，输入明确标为诊断合成数据，不能把产物当修复SFT模型。GPU是否成功及消耗以该目录最终report和配额快照为准，不以Notebook的COMPLETE状态代替模型训练成功。

**实际结果已返回：原样单卡路径失败。** 4×L4中只有GPU0被使用，小型bnb反传通过；31B NF4加载839.30秒后，在PEFT将非4bit权重转FP32时申请5.25 GiB、但仅余4.63 GiB而OOM。尚未注入LoRA或执行31B训练更新，0步、无adapter。脚本885.34秒，账户净扣额0.50h，剩28.20h。不要原样重跑；先解决下述转换峰值和多卡利用。此次外层COMPLETE是wrapper未传播worker退出码导致，生成器已修正但未重新提交；已执行v1源码原样保留在gpu-probe/probe.py。详细记录见同目录result-summary.json。

下一版正式训练实现需要解决的具体问题：

1. **限制非量化权重的上转。** PEFT的kbit准备会将共享embedding/head从BF16转FP32；262144×5376参数使其由2.625 GiB增至5.25 GiB，另有转换峰值。研究保持共享权重BF16、norm FP32的路径，并保留实际数值结果；当前探针保持原实现以定位真实瓶颈。
2. **减少无监督位置的词表投影。** 4096×262144的BF16 logits约2 GiB，默认CE的FP32 logits约4 GiB；`labels=-100`只屏蔽损失，不省这些分配。按下一token对齐选择有效监督位置，并分块计算词表投影/CE，保留Gemma的logit softcap和完整上下文。有效监督tokens/配额小时比输入tokens/秒更能比较当前稀疏监督数据的训练收益。
3. **先四卡DDP，内存不足再考虑FSDP。** 后续已实现`training/train_ddp.py`：每rank固定一张GPU和一份NF4基座，正常DDP forward、全局监督token归一化、累积结束才同步LoRA梯度，rank0导出adapter，各rank保存恢复RNG。两进程真实CPU数值比较已通过，包括不等监督量、尾部空rank和恢复；真实31B四卡诊断也已通过2048长度，尚未验证更长上下文。约5304万LoRA的FP32梯度为202 MiB，通信远小于每microbatch重复收集约17 GiB基座的FSDP；这是通信量分析，不是DDP/FSDP的速度对照实测。更长上下文若每卡仍放不下，再实现FSDP-QLoRA；当前uint8 NF4缓存不能用普通数值转换冒充BF16 quant_storage，需按[官方FSDP-QLoRA文档](https://huggingface.co/docs/bitsandbytes/fsdp_qlora)处理。
4. **把准备与计算分开。** 数据清洗、协议编码、长度统计、wheel缓存均在CPU完成；GPU只做模型加载、训练及必要推理。同一加载周期内顺序完成有限实验并及时退出，避免反复冷启动。只有短序列成功时，仍不能宣称4K/16K可行；先根据实际峰值与步耗时确定一个正式长度和步数。

这些是资源利用改造，不增加大型门控或测试平台；现有本地诊断数据也不足以支持完整高质量SFT，应继续扩充真实修复/恢复片段。

### 后续实现与运行入口

**更新：新路径已通过真实31B四卡诊断。** 256、256、1024、2048四步全部完成，四rank adapter更新且一致，step4 adapter已存Kaggle；缓存加载每rank约181秒，2048更新15.89秒，每卡峰allocated20.398 GiB，脚本264.48秒，账号扣0.15h后剩28.05h。代码与缓存可以进入真实数据准备和小规模SFT阶段。当前只证明诊断容量与数值执行，没有4K/16K、真实修复质量或CT推理迁移结论。完整数据见`research/005/kaggle-training-probe-20261010/ddp-result-summary.json`。

- `training/memory.py`已实现只提升norm及checkpoint分块监督位置CE，保持正常PEFT/DDP调用链。新单卡配方和DDP配方均显式选择`selected_chunked`；`reference_fp32`仅作为原实现对照选项。
- `training/configs/sft_qat_nf4_ddp.yaml`为四卡候选配方，读取带manifest的同源NF4缓存。`train_ddp.py`拥有真实加载、优化、全局指标、断点恢复和导出，不是torchrun包裹旧单卡脚本的占位实现。
- 私有CPU准备`ccjjllt/p005-cpu-nf4-cache`已在Kaggle Python3.13/torch2.11.0+cpu/bnb0.50.2验证微型Gemma4的22个4bit模块保存与重载，并完成完整官方31B的601个量化模块缓存。18.30GB缓存已归档可复用，训练GPU不再直接挂载62.5GB未压缩基座。
- `research/005/kaggle-training-probe-20261010/build_ddp_probe.py`生成独立新载体，已实际执行成功；快照打包真实训练代码，只挂载CPU NF4缓存与7-wheel缓存。四步按256、256、1024、2048长度执行，每rank每步64个监督token，诊断合成数据明确禁止作为修复训练语料；每步保留adapter，后段失败保留最近完整产物。1200秒子进程硬上限、1500秒服务器上限，无自动重试。不要原样重跑已经证明的四步。
- 本地数字证据在`followup-local-verification.json`：分块loss FP32误差0、adapter梯度最大误差8.20e-8；两rank对单进程全局batch的参数差3.54e-8，两rank参数一致、恢复后逐位一致。实际GPU结果必须另行归档，不能把这些CPU证据称为31B训练成功。

下一轮真实SFT应先准备长度合适、上下文完整的真实修复样本，按长度组织批次以减少DDP进程等待；一次加载后连续执行有效训练步，摊薄当前仍约181秒的加载成本。诊断中的重复文本loss和两个100%利用率采样点均不能当作真实数据的质量或长期吞吐保证。

### 下一轮研究顺序（四卡成功后的修订）

1. 先在CPU准备与官方v30匹配的推理环境，解决已观察到的Python3.13/cp312 ABI差异；用现成诊断adapter完成一次指定CT基座上的加载、实际路由与工具调用验证。诊断adapter只用于兼容性，不用于修复收益或正式提交。vLLM通用LoRA支持不能代替比赛冻结版本的实际证据。
2. 同时在CPU整理真实数据并统计完整模板后的长度。新增真实tokenizer测量：六工具schema增量332 tokens；T0 system加极短消息及工具共1158；再加入官方build_agent_prompt固定说明/预算、极短占位问题后1435（关闭sandbox testing时1483），尚无真实issue/目录树/源码/工具观察。2K只剩约565–613 tokens，因此不能将裸合成2K成功直接定成正式窗口。若代表性数据主要超限，再做一次针对4K的容量研究或与部署一致的协议精简；不截断历史观察来伪造短轨迹。
3. 数据和CT执行均可用后，做一个rank8小规模真实SFT候选，先一个epoch以内、最多50–100个更新作为候选上限，实际按样本量和耗时调整。保持prompt、工具协议和基座固定，沿用已生成NF4缓存，按长度分桶。
4. 在同一版本服务中，对8–12个已暴露开发任务做base/adapter配对；核心记录新增修复、原有修复回退及耗时，再决定是否扩大验证。推理机制候选另作独立比较，避免同时换prompt和adapter造成归因混淆。

建议第一轮研究包先控制在6个GPU额度小时以内（资源上限建议，不是运行时长承诺），多数预算用于真实修复对照；CPU数据整理与依赖准备不占GPU配额。最近一次余额28.05h仅为已记录快照，启动前重新读取。上述计划尚未启动新GPU运行。
