# 原100题：16K常驻VFARL研究

2026-10-28最新：[算法与固定100题深审](optimization-overnight/README.md)。已复现并修复动作长度归一化偏置、固定cohort结果依赖丢样、实际生产修改漏记、Adam状态错接及训练运行时依赖脱节；100题身份及原版本/历史奖励保留，累计39条题面澄清另存[新版本](question-revision-r3/README.md)。Addict56、Hamming81、Bag82、SVG84、OSA85、Marshmallow88及iniconfig95的具体原oracle盲点暂停未来RL，正确SFT仍保留；[最新课程r7](question-revision-r3/curriculum-r7.json)待采59题为32共同＋27专科。另4个历史runtime unknown已在真实CPU修复为有效0／1，并[私有导出925MB运行时](cpu-runtime-repair/execution-r7-20261028/README.md)，尚未替换课程和默认采样运行时。四轮[零更新GPU诊断](numerics-20261028/README.md)均已结束：R4b/R6/R7正常完成，R5部分完成后取消确认；0 optimizer更新。已定位持续grad/no-grad差异，最后checkpoint绕过干预没有改善数值，因此未推广。正式训练关闭，没有新的模型或比赛收益结论。以下各日期的旧参数、课程及原运行结果保留为历史说明。

[100题监督r6](question-revision-r3/supervision-r6/README.md)继承此前来源／CRLF／API修正，另补Dominate0修前真实异常文案及Mistune39公开MarkdownRenderer两端行为；当前805条／78,924监督tokens，98题789行与监督r5逐字相同。训练器消费[trainer-input-r6](question-revision-r3/supervision-r6/trainer-input-r6/README.md)及匹配工具／协议配置。51题本轮真实重放、49题历史来源，706工具及99最终回答完整保留；没有改变题目、原修复／oracle或历史奖励。旧版本不覆盖，旧r1禁止训练；未启动正式训练。

另已补齐[R3本地任务材料](question-revision-r3/material-local-r1/material-build.json)，约125MB，包含39份历史task原件。历史100行加CPU R7四行在实际消费中得到62项严格匹配／38项题面兼容，四题使用新结果；这不是100项新Linux验收。该材料尚未上传，无远端dataset版本；它是任务与来源材料，SFT标签继续使用独立的trainer-input-r6。

2026-10-28 09:40，共享Codex额度显示100%，当时停止新增实验并收尾CPU5。CPU5实际COMPLETE，但4题8次reward均null、缺依赖问题仍在，原件保留。账户窗口随后变化，工具读到2%，用户明确继续新一轮额度；CPU6及运行时导出R7均取得四题完整0／1。新运行时为`jiyujia/vfarl-current-four-cpu-runtime-export-r7/1`的`vfarl-cpu-stage.bin`，SHA `f13237d48dd0adecb3556334106836002290efa2b02987061f17b606771be44c`。本任务没有购买或执行额度重置，旧0提交记录仍作为09:40时点事实保留。详细来源、已修缺陷与剩余研究边界见[总记录](optimization-overnight/README.md)。

R3冻结载体实际为32K上下文、单响应1536、6活动任务与B4，原16K／768描述仅属早期计划；[501个实际序列核对](optimization-overnight/r3-actual-shape-evidence.json)已将两者区分。新有界响应片段目标与16K小cohort均为显式未运行方案。

## 2026-10-27：停止长采样后的本地优化

用户停止R3，要求先深度优化。本节是新协议的实现与研究计划，尚未启动新的GPU实验；下文原100题参数保留为历史运行说明。

新的显式协议为 `--collection-protocol live-original-pair`。焦点仍来自当前策略实际遇到的工具失败，奖励仍是完整终局验证，训练仍只更新焦点动作的PPO＋KL目标，没有改成成功轨迹SFT或CSO的DPO。

- 在首个预算充足的真实工具失败后、下一动作采样前固定快照。原始动作当时真实保存token、行为概率及独立reference评分；不按未来动作或奖励更换焦点。
- 把这个真实原始动作及其原始后缀奖励作为第一个样本，只增加一个从同状态恢复的独立分支。强制回放不进入训练。原来焦点之后的原始续跑加三条分支，改为原始续跑加一条分支：完整后缀数量从四条减为两条，不承诺墙钟必然减半。
- 每批固定最多两题、每个双卡副本一个活动任务、最多两个焦点状态；一个完整异奖组就允许更新。完成该小批并关闭采样服务后再更新、同步，避免同一旧轨迹中途更换权重。仍可能等同批另一题完成，尚未实现跨版本异步训练。
- 原始动作绑定实际快照、前缀、预算、原始动作行与行为概率、原始终局及独立分支恢复。缺失对端或评分无效不补失败。旧三分支协议和历史数据继续保留，不把已有强制回放改名为独立样本。
- 新协议中，真实工具异常返回结构化错误；格式问题、截断或未提交结束可获得最多三次连续继续提示，预算不重置。提交后在剩余原预算内等待模型最终回答，不再发送继续提示。此前“官方提交后立即结束”的判断已被[pinned官方调用链](semantic-audit/post-submit-terminal-scope-r1/README.md)推翻；当前采集器已移除该提前退出，CPU实跑另存来源。live现可在同一预算内继续工具及再次提交，最终评估取最后一次显式提交；historical保留旧guard。严格解析及整体沙箱仍不能称比赛Docker／ADK等价，见[独立多提交CPU证据](semantic-audit/post-submit-cpu-multisubmit-r1/README.md)。

训练数值侧保持原目标，k3 KL改为 `expm1(x)-x`，减少近零相消。加载Adam状态后显式采用本轮声明的学习率，同时保留moment与step；记录实际学习率。沿现有更新后测量增加参数步RMS、裁剪系数、一阶目标变化预测与实际变化，不增加模型前向次数，也不通过回滚反复训练同一旧批。

[新小轮配置](../../../../training/configs/vfarl_l4x4_live_pair.yaml)声明16K上限、固定学习率1e-6、单次更新；手动learner没有warmup scheduler，配置明确warmup为0。降低学习率是待验证的研究选择，不能称已修复旧更新方向异常。CLI的`--context`仍决定载体实际上下文，容量与反向长度须分别实测。

旧short-r1实测目标 `0.235803419→0.236187304` 略差。旧首步变化元素RMS约4.98e-6，接近Adam学习率5e-6，支持有限步长／低精度影响的假设，尚未定位唯一根因。一次性CPU核对中，pinned融合输出层对比普通softmax的梯度余弦约1；稳定KL在±1e-5处保留约5e-11，而旧FP32表达式舍入为0。这些不证明31B GPU反向或修复收益。

本地已核对真实旧focal数据兼容、模板反馈拼接、Adam状态续接、模块语法与载体冻结接线；未新增测试文件或启动GPU。后续先用新的真实异奖组做一次小更新，记录方向，再观察独立训练研究题。若两个动作终局同奖，当前目标仍无优势；精简调度不能保证每批必有更新。

R3取消后的日志与发布状态见[cancelled-r3/recovery-summary.json](cancelled-r3/recovery-summary.json)。完整动作文件发布前，不把实时日志摘要当作可训练token数据；旧快照跨session续采也尚无自动等价恢复保证。

## 100题课程与采样优先级

保留原100题的名单、题面、源码快照、任务哈希与参考修复。原selection SHA仍为 `ee270560ab7b27a0322eec8c604a53832d87f491cb6cdd71f680ff38a065d946`。新的[课程清单](curriculum.json)仅用于host侧调度，不作为模型提示或新的奖励标签；由 `python -m competition.vfarl.l4x4.curriculum` 从冻结元数据及真实日志构建。

修正了原`mixed_course_order`把适配器G当作共同题的错误。真正的课程身份是`role`：67共同＋33专科；适配器主责实际为G34/F19/R10/S37，不能代替课程身份。普通mixed-course现在按role做两共同一专科，再轮转仓库与机制。

三轮resident合计领取35道不同任务，实际全是共同题。剩余65道在这三轮未领取，包含32共同＋33专科；这不表示它们在整个项目从未采样。evidence-course因此按剩余配额近1:1交错，补足此前专科覆盖，完整100题池仍为67/33。

- 4道评分无效题暂停新采，先查明验证原因，不把null当失败，也不据此判定题坏。
- Tenacity `pvngsy7s`的原始0与独立分支1列为优先回收核对候选；没有精确原件前不标成现成异奖组，也不自动重新整题采样。
- 已观察到全成功的焦点状态降低复采优先级。这只描述当时状态与策略，不永久删除整题。
- 高成本轨迹和旧格式／截断失败分别保存；旧回复预算与反馈语义不同，单次旧0分不直接当作难度标签。历史示范长度只是成本先验。
- 4道独立研究验证题的ID仅用于host侧禁止训练与自适应选题，均不在原100题中；未读取其题目内容。

新课程保留100题完整排序，并单独声明默认新采65题。已领取35题、评分暂停题、待回收候选和预留验证题不会自动进入新采队列。以后若需在新策略下复访，须显式形成新的课程版本，不能复用已提交行为批冒充新样本。

载体接线使用 `--resident-task-order evidence-course --resident-curriculum competition/vfarl/l4x4/round100/curriculum.json`，可与`--collection-protocol live-original-pair`组合。课程文件冻结进载体，运行时解析为解包后的绝对路径，resident再次绑定原selection与任务身份。已用真实元数据和本地载体冻结核对，未启动GPU；课程调整能否提高异奖产率仍须实际采样研究。

## 后续本地优化：焦点触发、KV所有权与旧数据复用

新增显式 `--focal-trigger execution-recovery`，仅与live-original-pair组合；旧any-tool-failure仍保留作对照。新触发器读取实际反馈的`error_type`及完整stdout/stderr，区分工具执行错误、初次诊断测试失败、修改观察后的测试失败、普通搜索空结果和含糊搜索管道失败。成功写入只是一条修改观察，不代表修复正确；当前计数已读取真实工作树内容指纹，run_command／shell和失败命令留下的已知生产修改也会纳入；无法归类的新路径仍记unknown，不作为正面修改证据。普通诊断测试失败、搜索空结果和含糊搜索失败不直接触发，预算／超时反馈也不充作恢复动作证据。新触发要求至少剩4个动作回合、3次工具调用及原token余量；两分支预算不重置。

分类发生在真实工具反馈到达后，下一动作采样前固定快照；终局验证与奖励不变。分类依据保存在原有tool-events、ledger和选样记录中，没有新增逐状态门控文件。[R3离线反馈分析](cancelled-r3/feedback-census.json)包含265条真实反馈，其中首次诊断测试失败1、搜索空结果8、含糊搜索失败4；3题的首个触发点因此移动或暂缓。该分析不预测新的异奖产率或GPU速度。

单会话cache路径增加B1所有权转移：没有padding和后续dummy行时，不再用单元素cat合并KV，也不在停止时复制整个KV。多行路径仍保留停止时复制以保护滑窗。前向顺序、position和mask不变，源码hash仍绑定真实实现。一次性CPU小Gemma对比旧路径，下一token与多turn增量预填充logits差为0；这不等于实际Unsloth GPU已经测得提速。

R3完整输出已发布。只读取约82.4MB有界Range，回收443个研究文件、501条真实动作；未下载整个约6.49GB运行时包。完整输出确认optimizer为0步，16条原始轨迹结束：9成功、3有效失败、4评分无效。

Tenacity的原始0与独立branch1为1已完成[原件来源核对](cancelled-r3/tenacity-live-original-review.json)：同2316-token前缀、状态、策略与预算，两个真实焦点动作分别99／97 tokens且不同。原始动作未做reference评分。新的[待评分数据](recovered-tenacity-r3/pair-pending-reference.jsonl)保留真实tokens、行为概率和采样身份，未把回放作为独立样本，也未伪造旧precommit文件。

`competition.vfarl.l4x4.recovered.build_pair()`通过已审冻结R3源码和实际快照／恢复／终局记录建立显式旧来源派生契约。`complete_reference(policy, pending_path, out_path)`仅接受外部明确提供且已独占的模型，真实补算缺失reference，记录新评分路径后恢复原模式；行为概率永不重算或替换。原pending数据仍不变且被训练器拒绝。2026-10-28已在真实31B零更新诊断中补齐，另存[新两行](numerics-20261028/execution-r4b-resumed-20261028/actual-output/vfarl-train/numerics/focal-reference-completed.jsonl)，SHA `a9e2e282ebfab39e2f625b5ddc77c251c8ba9c8a0bd86d4cfa39b25582fee5bb`；原行为／token／mask／reward未改，branch1全行不变。当前CPU消费器[实际读入](numerics-20261028/execution-r4b-resumed-20261028/current-consumer-reference-admission.json)确认2行／1异奖组／完整动作分母2，尚无optimizer更新。

载体已支持给非resident的回收Unsloth更新传入经SHA核对的初始adapter。当前仅完成真实reference及零更新梯度／head／暂态方向诊断；这批行为未消耗训练机会。首动作模式差异与有限差分偏离仍待定位，正式GPU训练保持关闭，没有新的独立任务收益。

## 原常驻实验说明

用户确认沿用v4的67共同＋33专科题，并授权原124,593,514字节材料上传到jiyujia私有数据集。selection及材料SHA见plan.json。全部来自train，不读取heldout。

两份Unsloth模型各使用两张L4，常驻采样。每小批最多4题、最多4个focal状态；累计2组真实异奖状态即做一次VFARL PPO＋KL更新，保留Adam对象及其矩，再把已提交LoRA同步到另一采样模型。小批异奖不足则在同一策略下继续积累，不反复训练已提交行为。初始LoRA来自已完成的短轮；该短轮未导出Adam，因此本轮初始Adam是新的。

上下文16,384；单轮输出768、思考预算256、每个episode输出总额8,192、32次对话/工具预算。新任务领取预算7小时，结束后保留在途结果及最后更新。100题是固定覆盖目标；以实际配额和实际完成数量为准。

CPU环境证据按原任务哈希合并，100/100均有真实baseline 0、参考修复1的历史记录；来源见cpu-evidence-sources.json，不重复整套CPU执行。

上一短轮确实完成1步Unsloth更新，但测得奖励方向略负，尚无泛化收益证据。本轮保存逐步目标、重要性权重、梯度与参数变化。16K配置不等于16K反向传播已被实测，须报告实际序列长度。该100题全部是训练题，其任务收益不能当作独立验证。
