# Verified Focal Agent RL：数据契约与小批采集方案

2026-10-24。研究设计，不表示已经采集、验证或训练。沿用 G=r128（q/o+MLP 五投影）、S/F/R=r32（文本七投影），计算和导出 BF16。真实工具只有 run_command、read_file、edit_file、write_file、get_status、submit_patch。

## 1. 与原成功轨迹计划的变化

原来的成功轨迹仍可提供冷启动候选，但不能直接充当“同状态分叉”RL数据。来源轨迹没有可恢复环境、策略版本和行为 logprob 时，只能进入来源可追溯的 SFT/静态资料池。必须在本工具环境中重新执行，才取得本工具真实观察；必须实际保存并恢复完整状态，才有资格产生同状态比较。

四份数据不是按“编辑失败 / 思考太长”分工。G/S/F/R是要修复的仓库机制；错误定位、无效编辑、反馈利用和预算管理是四个 adapter 都应练的共同能力。G可覆盖所有机制；S/F/R按题目的因果机制增加采样权重，保留通用回放。六工具中没有委派工具，不生成虚构 handoff 调用；若运行时外部选择 adapter，选择结果只能作为真实运行元数据。

## 2. 分为四个可组合的真实数据池

1. **cold_start**：目标六工具下的真实成功过程，用于 SFT。保留失败尝试和工具错误作为上下文，只有真实助手动作进入相应 loss；工具观察不伪造，不用未来答案补写过去的思考。
2. **critical_action_pairs**：同一已恢复状态中，只替换下一个动作，分别继续执行并完整验证。保留原动作重放和替代动作分支，记录观测胜负。只有一次分叉时称“本次观察到的关键动作收益”，不宣称普遍因果定律。
3. **recovery_segments**：同状态起点允许多步短恢复，例如 read→edit→run。它能证明恢复策略有效，不能伪装成仅改一个动作的 CSO 证据。源轨迹里“失败后自己又修好了”也不是与早期失败动作的同状态偏好对。
4. **on_policy_rollouts**：由当前训练策略采集的完整 episode / 同状态分叉；保留每个输出 token 的行为 logprob、mask、策略版本和采样配置，供实际实现的 PPO/GRPO/GiGPO 类目标使用。没有这些字段的旧轨迹不改名成 on-policy RL。

SFT、off-policy偏好和on-policy RL可以引用同一任务，但用途及权重必须显式区分。

SFT的assistant mask也要审阅。成功轨迹中的无效重复、错误编辑与事后解释不自动成为正向监督；它们可保留为真实历史条件，错误动作的正向loss置零。经实际重放且有证据支持的恢复动作才进入相应正向span。不能从一条最终成功轨迹推导“每步都正确”。

## 3. 什么才叫同一个状态

`state_id = SHA256(canonical(snapshot_manifest))`。manifest 必须绑定下列内容的内容摘要和恢复方式，不能仅使用 Git commit / diff / 文件 hash：

- **完整模型可见前缀**：system/user/assistant/tool消息、工具调用ID和顺序、已见错误和截断、预算提示、工具schema、chat template、上下文压缩结果。若历史被摘要，两个分支必须看到同一份当时可得的摘要；不能让其中一支多看一次报错或未来答案。
- **代码与文件系统**：tracked、untracked、相关 ignored 文件、git index、权限/符号链接、工作目录、工具生成的临时文件和写到仓库外的任务文件。仅 `git reset` 无法复原未跟踪文件，也不等于完整恢复。
- **工具与运行环境**：工具实现版本、编辑锚点规则、输出截断、shell session、cwd、环境变量、Python/依赖/镜像版本、当前缓存和临时目录。秘密只用受控引用或摘要，不能把原始凭证写进训练数据。
- **隐藏可影响状态**：后台进程、打开文件/连接、退出栈、服务内存、随机状态、外部服务副作用、时间相关状态。能够可靠 freeze/restore 的记录快照证据；不能恢复的明确列入 `unrestorable_components`。
- **原始预算账本**：起始与已消耗的 wall time、输出tokens、tool calls、turns，以及剩余额度。两个分支的恢复状态和剩余额度相同，不是各拿一份新整题预算。
- **策略身份**：绑定base model、continuation adapter权重hash、训练step、tokenizer/template、system prompt、routing配置和解码参数。分支seed及提议动作的教师身份另记branch，不用不同seed制造不同环境state_id；比较双方应使用同一continuation策略。KV cache可以从相同前缀重新构建，但模型和序列化必须一致。

同状态比较使用等量的**逻辑剩余预算**。宿主机的绝对时间、真实网络和外部随机性未必可以复原；不能谎称它们相同。优先在进程静止、无未完成工具请求、无外部写副作用的边界采checkpoint，并以已冻结的本地环境执行。

恢复范围分开记录：

- `complete_declared_scope`：声明范围内的完整可恢复状态，适合高可信同状态比较；“完整”必须附组件清单，不意味着忽略未声明的外部因子。
- `observable_replay_only`：只复原消息/文件等可观察部分，隐藏运行状态不可靠；可用于一般恢复研究，不能计作高可信单动作因果偏好。
- `unreplayable`：环境或外部副作用不可恢复，仅保留原始轨迹资料。

比较前实际核对恢复得到的文件、消息、工具状态和预算manifest，并保存恢复日志；这是一条样本的证据，不建设额外门控脚本或测试套件。

## 4. 单动作反事实的实际操作

1. 从真实失败 episode 中提出候选动作 t；候选来自可见工具错误、重复动作、定位偏差或预算耗尽前兆，不能用模型解释本身判定错误。
2. 保存动作 t 之前的状态。若只有动作 t 之后的快照，则只能研究“已看见这次错误以后如何恢复”，不能倒称动作 t 的原状态。
3. 从同一快照重放原动作 a−。原失败轨迹不能自动替代这次基线重放，因为源环境、策略及剩余预算可能不同。
4. 从同一快照采样替代动作 a1…ak。**单动作比较只改一个助手动作**，该动作可以是一个真实工具调用或一个真实助手决策；随后所有分支使用同版本 continuation policy、同种解码配置和匹配的剩余预算继续执行。共同随机seed有助于降低噪声，但不消除环境随机性。
5. 提前限制恢复段长度用于采集效率；若短段结束还没有完成整题，只记local evidence。需要resolved证据时，仍必须让续跑走到提交和完整验证。
6. 完整评估通过且没有通过修改评分测试/配置作弊，才标 `task_resolved_verified=true`。基线和替代都完整验证后，才形成完整结果对。中断、安装失败、评估器错误是unknown，不自动标失败。
7. 重复若干种子能估计动作作用的稳定性；同一状态的十个分支仍只有一个checkpoint，不算十个独立问题。

动作 t 的原策略与 continuation policy、替代动作提议策略可以不同，但必须分别记录。教师生成的局部优胜动作可作off-policy偏好；它不因此成为当前学生的on-policy样本。

## 5. 奖励与优势的边界

- **整题结果**：`R_task=1` 仅对应本训练任务的完整真实验证通过；验证完整完成且未通过可记0；没有完整执行或环境失效用null/unknown，不补0。验证目标和命令版本固定，保留退出码和原始日志。
- **局部进展**：读到正确文件、唯一锚点成功、补丁可应用、某条相关上游检查改善，分别存为类型化 observation / local_signal。它们可以用于独立的局部偏好、辅助loss或有上界的 shaping，不能改写为resolved。
- **环境错误**：镜像缺失、网络故障、依赖安装失败单独记录，不作为adapter能力负例；源任务测试缺陷需保存说明，不用参考patch“看起来正确”代替执行。
- **成本**：使用原任务起点账本，`C_total = C_prefix + C_suffix`。分叉时prefix已花费成本保留，剩余预算不重置；数据去重统计只计一次prefix，实际计算费用则计每次重放和每条branch。采样/验证的CPU费用也另外记账。
- **预算信号**：先把verified completion置于主要目标，再研究同成功等级内的成本偏好，或明确阈值/权重的约束目标。不能让省tokens的失败自然压过真正成功。预算倍数基准必须从训练任务的独立成功轨迹估计，不能偷用当前候选的最短分支或公共/heldout答案逐题定价。
- **短预算条件**：policy输入中真实暴露剩余额度或离散预算档位，训练和部署保持一致；不能只给reward成本而不记录模型当时知道的预算。

建议保留未经压缩的原始信号，不提前合成为唯一浮点reward。后续可研究用户的联合优势，但要分别说明两个均值的组：task均值来自**同任务/同预算**的完整rollout；branch均值来自**同state/同剩余预算/同验证定义**的分支。不能把不同任务难度的均值直接混在同一组。

公式中两个分量若都包含同一个最终成功标签会双重计权；先明确 `R_branch` 是终局结果还是局部证据，并保留二者的统计，alpha只是待研究超参。局部观测与整题回报量纲不同，不能无标定直接相加。参考KL和通用回放的目标由训练器记录，不能把BF16/rank设计当作稳定性保证。

## 6. off-policy 与 on-policy 的数据要求

| 用途 | 最小可用证据 | 不能声称什么 |
| --- | --- | --- |
| 六工具SFT | 真实执行消息、任务身份、验证出处、可用assistant mask | 来源轨迹成功不等于已在本工具重放 |
| 局部DPO/偏好 | 相同真实prefix、恢复范围、chosen/rejected动作/段、双方执行与结果、reference身份 | 不需要旧策略token logprob，不意味着它是on-policy RL |
| PPO/GRPO类 | 当前行为策略版本、采样分布、逐token behavior logprob与mask、完整回报/组、预算 | 没有logprob的旧教师轨迹不能事后补出“采样时概率” |
| GiGPO式状态组优势 | 以上on-policy字段，另有真实状态相同的group和正确的group成员关系 | 文件hash相同/题目相同不足以认定same-state |

从旧样本在当前模型下重新算logprob只得到当前/参考策略概率，不能还原未知行为策略概率。是否允许策略陈旧、importance ratio怎样截断属于具体算法实现，必须记录；不在数据层假定任何旧样本都可直接混用。

## 7. 小批试采：96个独立任务，先测产率

本地官方重放已确认三种来源适配风险：生产doctest被测试路径保护剥离（parse）、来源必需依赖在当前Python版本被排除而required skip（Freezegun）、参数化节点被官方matcher按同函数合并而可选skip污染required（Dask9573）。保存原始JUnit、原始要求及matcher返回，不修改官方工具/验证器或删除要求取得成功。命令退出0、指定F2P单独passed、局部行为改善都不能替代原版整题resolved。环境/采集失败和模型策略失败分账；这些人工尝试不构成Focal负动作或同状态偏好对。

96题的来源/机制库现已汇齐，见 [pilot96/manifest.json](pilot96/manifest.json)：G主责36，S/F/R各20，每个专家子机制4题。下表的96条冷启动成功、192条当前策略attempt和分叉产率仍是采集目标，**不是已完成的过程数量**。实际人工off-policy成功另见 [重放清单](replay/replay-summary.json)。任务来自排除比赛四仓及fork的允许train来源；G包括普通规则、组合修复及跨专家边界。机制难度、项目、错误类型交叉分布；同修补链、近重复变异整组隔离。

| 阶段 | 数据量目标 | 实际产物与停止/调整依据 |
| --- | ---: | --- |
| 六工具冷启动 | 96题各至少1条目标协议成功过程为目标 | 不可恢复题/验证冲突换题并保留真实原因；成功轨迹可超过96但不增加独立题数 |
| 当前策略基线 | 96题×2次=192条完整attempt | 全部保留成败与资源账；真正失败才触发恢复候选 |
| 失败checkpoint | 最多128个，通常每题最多2个 | 真实失败不足128时接受不足；不人为损坏轨迹凑数 |
| 关键动作续跑 | 每state原动作重放1条＋替代3条，最多512条continuation | 同state及同剩余预算；可先短段筛选，只有完整续跑并验证才进入resolved对 |
| 完整结果对 | 希望获得32–64个有完整验证的不同state对 | 是产率假设，不是保证。报告resolved/unresolved、both fail、both succeed、unknown各自数量 |
| 多步恢复段 | 另希望有16–32个不同state的真实恢复段 | 与单动作对分开；能重用同state但不重复算checkpoint/独立任务 |

先看环境恢复率、真实失败比例、完整resolved对产率、单动作相对多步恢复的收益，以及每单位预算的新成功数。若32个完整对都难获得，先修采集和选题，不直接扩到1400题。不能按adapter硬凑同数“正例”；失败稀少可能是题太易，失败全不可救可能是题太难/工具不适配。

所有适配器均要包含共同工具恢复，专科则按S/F/R实际机制偏采。一个任务即使为多个adapter提供数据，独立任务账仍只计一次；每个adapter实际loss暴露另计。

## 8. 成本只做透明算术，不冒充实测

用作预算刻度的假设：每条初始attempt生成3000个assistant tokens，每条checkpoint continuation生成1500个assistant tokens。

- 基线：192×3000=576k sampled assistant tokens。
- 分支：128×4×1500=768k sampled assistant tokens。
- 合计：约1.344M generated assistant tokens，不包含冷启动教师成本、工具输出tokens、反向传播或输入prompt重复计算。
- 若每次分支启动的prefix为12k，512次起点单是重复prefix即约6.144M input tokens；续跑各turn的额外输入、缓存命中/未命中另记，不能把6.144M称为全部input。
- 若初始attempt平均8分钟、分支平均3分钟，192×8+512×3=3072 environment-minutes，即51.2 environment-hours。这是容器占用墙钟小时示例，**不是CPU core-hours或GPU-hours**。例如每环境预留2cores只说明102.4 reserved core-hours容量；实际CPU消耗按worker cpu_seconds累计。真实并行度、GPU/服务采样速度、验证时长尚未知。

原prefix成本在每个分支的任务奖励中保留一次，计算账本中又要如实计入实际重复处理；这是不同账目，不应混成一个token总量。保存cached_input、uncached_input、generated、loss_tokens、tool_output、verifier_cpu_seconds等分项。

## 9. 最小可追溯记录

无需为每个未执行状态堆叠状态JSON。可采用一个任务索引、内容寻址的消息/快照/原始日志和按attempt追加的事件记录；字段是研究证据，不是建设新门控体系。

**task**：canonical_task_id、origin_dataset/revision/split、origin_repo/base_commit、license来源、repair_family_id、用途隔离依据（只用允许ID，不读heldout正文）、problem_blob_hash、机制标签及因果证据、upstream_verifier版本。

**checkpoint**：state_id、snapshot_manifest_hash、原episode/turn、messages_prefix_hash、model_visible_budget、fs_snapshot_ref、tracked/untracked/ignored范围、tools/session/cwd/env/dependency manifest、process/exit-stack/external-state处理、restoration_scope、unrestorable_components、restore_log、original/consumed/remaining budgets、policy/base/adapter/tokenizer/template/schema hashes。

**branch**：branch_id、state_id、kind（original-replay/single-action/multi-step）、proposal_policy和continuation_policy版本、seed/decoding、intervened_action_range、真实messages/actions/observations、逐token behavior logprobs及mask（仅已实际采集者）、max_horizon、terminal_reason、tool/errors、耗时/turn/tool/token细账、patch_hash、privileged_verifier_log_ref、verifier_exit/status、task_resolved_verified（true/false/null）、typed local observations、environment_failure。

**pair/group**：state_id、成员branch IDs、是否完整验证、same-state恢复证据、单动作/多步类型、预算匹配、observed_outcome、chosen/rejected理由、重复种子稳定性、训练用途（SFT/local-preference/on-policy）、reference-policy hash、sample_weight和supervised action span。

验证输出可以由collector保存，但不得作为早期prefix的工具观察泄漏给policy；金补丁、训练任务验证patch同样与policy上下文分离。只运行已有上游验证和必要一次性手动命令，不新增测试文件、tests目录或改名的gate/check框架。
