# VFARL算法与固定100题的深审

2026-10-28。用户要求持续优化算法与固定100题，先不正式GPU训练，允许短GPU验证。本轮保持原100题身份、原题版本与历史奖励，并将39条有依据的题面澄清冻结为独立新版本；没有读取heldout或预留4题正文，没有新增测试文件。下面区分已复现的缺陷、实现修复与仍未测得的研究收益。

**最新执行：** 第一轮共享额度显示100%后曾停止新增实验；账户计量窗口随后变化、工具读到2%，用户再次明确“继续优化，仍不启动正式训练”。这一窗口已完成下述研究并在额度终点收尾。[CPU6](../cpu-runtime-repair/execution-r6-20261028/README.md)及[R7运行时导出](../cpu-runtime-repair/execution-r7-20261028/README.md)均已取得当前四题完整baseline0／参考修复1；CPU5的8个环境unknown原件保留。首个[真实31B零更新GPU诊断](../numerics-20261028/execution-r4b-resumed-20261028/README.md)完整结束：0 optimizer、adapter前后hash一致，缺失reference真实补齐。后续[模式对照r5](../numerics-20261028/execution-r5-mode-order-20261028/README.md)完成第一进程13次前向，第二进程未完成并按预算取消。实际两动作均有持续G／N差异，同模式重复稳定，热身未消除；尚未定位唯一根因或取得训练收益。旧停止时0提交记录保留为历史时点。

**当前监督消费版本：** [supervision-r6](../question-revision-r3/supervision-r6/README.md)包含805条／78,924监督tokens。最新仅替换Dominate0及Mistune39的16行，补真实修前故障／公开API行为，其余98题789行与r5逐字相同；此前CRLF、Bottle目录保存、PyPika SQL、Bag频次及Hamming truncate修正保留。51题本轮新teacher、49题历史来源，库存100题未换。直接输入为[trainer-input-r6](../question-revision-r3/supervision-r6/trainer-input-r6/README.md)，805个实际CT prefix及原IDs/mask映射为labels/attention，reader／CPUcollator消费一致；源action-view不能直接传给SFT trainer。706工具及99最终回答完整保留，最大上下文4649、19行动／7题超过4K，未截断。旧版本和失败证据保留，r1跨度错误及无参schema元数据草稿继续弃用，未启动正式训练。

最新数值结论：[R6一次backward对照](../numerics-20261028/execution-r6-backward-mode-20261028/README.md)确认本例实际offload全部保持BF16，没有FP32→BF16压缩；no-grad评分确实分配offload槽并改变GC计数。最后[R7干预](../numerics-20261028/execution-r7-no-grad-checkpoint-20261028/README.md)已正常完成21次前向／1次backward／0 optimizer。临时绕过N／R的checkpoint消除了额外GC状态变化，但legacy／candidate／restored的LP及hidden逐值相同，模式差异没有改善，因此不推广为训练修复。R7的A前后N/R/G也全相同，未重现R6的N变化；R6中间还插入了B调用，不能单归因backward。现有metadata没有实际生成forward的源码身份，编译路径仍只是未证实候选。主训练默认保持不变，正式训练关闭，所有本轮GPU作业已终态，最后剩余3.463413 GPU小时、预留0。

当前可复用产物分别为：

| 产物 | 当前内容与执行状态 |
| --- | --- |
| [监督r6及trainer输入](../question-revision-r3/supervision-r6/trainer-input-r6/README.md) | 原100题，805行／78,924标签；尚未训练 |
| [R3任务材料](../question-revision-r3/material-local-r1/material-build.json) | 125,224,607字节，含39份历史task原件；本地已冻结，未上传，无远端dataset版本 |
| [CPU R7运行时](../cpu-runtime-repair/execution-r7-20261028/README.md) | 私有导出925,108,005字节，四题新0／1；其他题不称新环境全通过 |
| [课程r7](../question-revision-r3/curriculum-r7.json) | 7项弱oracle及4项历史runtime暂停仍保留；默认采样尚未切换新材料／运行时 |
| [数值研究R4b–R7](../numerics-20261028/README.md) | 全部已终态、0 optimizer；模式差异与跨运行差异尚无唯一根因 |

最后补齐的[R3材料消费](../question-revision-r3/material-local-r1/consumption-observation.json)实际读取归档中的39份原件，并执行现有报告选择／适用性函数。合并历史100行与R7四行后为62项严格task匹配、38项仅verification兼容，四条新记录原样选中。它避免下一次仅重打carrier却遗漏原件，未新增Linux验收、模型运行或训练收益。

尚待解决的数值问题有实际量级：[R7概率复算](../numerics-20261028/anchor-adam-review-20261028.json)中，两动作最大token差均低于当前0.08容限，但N/G锚点的动作importance仍相差12.5617%／1.6137%。通过逐token容限不代表整个动作权重等价，直接换锚点也会改变目标。此轮没有凭这些结果改loss、学习率或默认阈值。

2026-10-28 13:57（UTC05:57）第二个已授权窗口的共享Codex额度也显示100%，已按用户要求停止新增研究／提交，见[两次额度终点](quota-end-20261028.json)。本任务未重置或购买额度，全部在途CPU／GPU已取得终态；正式训练及本窗口optimizer更新均为0。研究尚未解决所有数值问题，没有新模型或比赛收益结论。Kaggle最后记录剩余3.463413 GPU小时，属于另一配额口径。

## 已形成的新证据及修复

- **动作长度归一化会改变学习方向。** 每动作token均值并非对终局成功率梯度的统一缩放。有限动作模型精确枚举给出旧方向与真实成功率梯度内积为负的反例。新增显式`action_sum_fixed_scale`，PG与sampled k3均按选中token求和后除固定128；历史配置仍走原`.mean()`。反例及估计目标边界见[目标审阅](../objective-audit-20261027/README.md)。这不是对旧short-r1负更新的唯一根因解释，也没有新的31B训练收益。
- **只按异奖动作数平均仍可反转方向。** 即使动作等长，过滤同奖组后再除mixed行数也让分母依赖终局结果。新模式保留当前固定cohort中完整同奖组的零PG质量；live-pair不再将前几批全同奖记录拼入下一批的随机分母。有限两任务枚举中，真实平均成功率梯度为+0.01，旧分母期望为−0.0125，新完整组质量为+0.005（组均值基线的1/2）。unknown与缺失对端不补零，不编造动作质量。
- **固定cohort会按别题结果丢样。** 首题先产生异奖时，旧队列拒绝较慢已预选状态的分支。新live-pair固定cohort完成已领取任务的配对，claim截止时间仍禁止新任务。真实生产内容指纹还补上shell写文件和失败命令留下的修改；报告产物/无法归类的新增文件保留unknown，不作为恢复触发证据。一次性机制证据及成本见[collector审阅](collector-cohort-evidence.md)。
- **仅自然结束的响应仍有完成筛选偏置。** 达到length上限的真实片段可以继续执行并取得终局奖励，但旧learner丢掉该片段及其配对质量。有限MDP枚举中旧方向可与真实收益反号。新增独立、显式启用的[bounded_response_segment模式](../objective-audit-20261027/bounded-response-segments.md)，以真实预声明cap作为动作边界，绑定原始／恢复两端的prefix、ledger和采样上限；不为旧length补证据、不补unknown奖励。默认旧模式保留。它仍受终局验收覆盖、off-policy续写与TIS截断影响，不是全任务无偏目标，也没有真实模型收益。
- **Adam裸状态存在同形状错接风险。** 新保存格式绑定实际adapter tensor hash、optimizer有序参数槽、shape/dtype、冻结模型/协议及live LoRA scaling；加载时核对再恢复moment/step，采用当前声明学习率。旧裸状态明确拒绝，不能静默续接。CPU证据见[optimizer绑定](../objective-audit-20261027/optimizer-binding.json)。
- **真实Adam执行不能与adapter成功提交混为一谈。** native在step已返回、导出失败时原本可能被外层记成0步；carrier还可能将Unsloth已有partial的1步当成可复用更新。现保留rank局部执行事实，未确认全局完成就不称全局提交；carrier另核不可变commit、behavior SHA及实际adapter文件SHA/bytes后才允许后续rollout。真实微Adam与注入I/O故障、实际reader文件消费见[异常边界](../objective-audit-20261027/gradient-update-boundary-review-20261028.md)。缺记录保持unknown，partial不是checkpoint；loss／梯度未变，未验证完整FSDP或31B。
- **有限精度的下降预测可能失效。** 真实CPU BF16小LoRA、零B起步的受控例子中，autograd与实际参数步点积为负，同一grad-enabled前向的有限步loss却上升；同一实际参数步的FP32控制下降。更小步可能完全落在舍入平台。见[精确输入与边界](../objective-audit-20261027/bf16-finite-step-boundary.md)。这说明不能单凭有限loss上升判定backward代码错误，也不是31B旧负更新的根因结论或学习率建议。
- **历史CPU通过没有绑定到实际采样环境。** R3四个unknown来自moto缺sure、DVC缺colorama、markdownify缺bs4。进一步核对发现非stock的marshmallow也可能丢tzdata及源包元数据。维护中runner按实际dist-info及必要来源绑定，拒绝错误环境回退；原null不补0。CPU修复载体和实跑边界见[运行时修正](../cpu-runtime-repair/README.md)。
- **源码改了但执行值可能仍旧。** 真实Moto在控制相同mtime/大小后，新Python进程继续读旧pyc；仅-B也不禁读。新增显式`isolated-prefix-no-write-v1`任务环境，用空任务cacheprefix＋禁写绕过旧缓存而不删除任何共享缓存。真实official manager/API观察为7→7（源码6）→6→6，显式-E又返回7，边界与编译成本均保留。默认历史模式不变，新policy不能借用旧CPU结果。见[新配置及实际证据](../semantic-audit/source-bytecode-profile-r1/README.md)。

## 固定数据池的研究边界

100题包含81道合成变异，75题的参考补丁增删合计不超过4行；100个repair family仅来自43个snapshot。没有证实需要删题的family重复，未按模型输赢换题。首轮结构审阅后，下面的源语义复核进一步发现了题面与示范缺陷。R3原始终止episode最终上下文中位10,472.5、最大19,108；全部501次实际采样序列中位7390、最大19108。R3冻结载体实际为32K／单响应1536／6活动任务／B4，原16K／768说明不能当作R3配置；新16K小cohort配置是另外的未运行方案。见[实际载体与模型序列证据](r3-actual-shape-evidence.json)，不能用历史短示范长度替代真实策略成本。详见[100题数据审阅](../data-review.md)。

此前35道resident领取任务全部为共同课的错误已在课程中纠正；v4课程原新采65题保留32共同+33专科。最新[版本化课程r7](../question-revision-r3/curriculum-r7.json)绑定39题面修订，依据7项具体弱oracle反例暂停未来原奖励RL，另4 runtime unknown仍暂停，待采59题为32共同+27专科。Addict56原已claimed，暂停后不再减pending；仅未claimed Hamming81从r6队列移出。实际loader独立消费与旧observations／100身份保留见[r7复核](../question-revision-r3/manual-curriculum-r7-independent-review.json)。库存100题及正确SFT仍在；原35领取与历史奖励不改，实际新版CPU适用性独立决定能否采集。原65／64／63／62／60题及r4/r5/r6课程保留为历史版本。课程改动尚未产生新的异奖产率证据。

本轮已进一步分四段审阅全部100题的题面、实际生产修复及源契约，分别保留静态、局部执行与完整oracle证据等级。已找到真实API/Expected冲突、不能触发故障的MRE与Actual转写噪声，不能统称100题都无效。split_by已完成独立题面revision及同一原oracle的真实0/1核对。OSA发现已验收示范的空串距离错误，且故意常数0也通过全部18个原验收节点；原奖励不改，弱oracle单列。修正后的OSA已重新取得真实六工具示范和原官方0/1，见[索引75–99审阅](../semantic-audit/075-099/README.md)。这属于训练材料修正，不是模型收益。

另一题内奖励盲点来自qrcode SVG84：仅返回`ET.Element('not-svg')`也获原81节点全通过，实际XML根违背题面的proper SVG element与standalone SVG契约。正确参考控制生成正常SVG根。该错误探针未导出为teacher/SFT，原oracle及历史奖励保留，见[原始受控验收](../semantic-audit/oracle-strength-20261028/qrcode84/oracle-observations.json)。iniconfig95另有题目内盲点：显示行号恒1的实现通过全部28要求节点展开的42用例，实际lineno7／13却应显示8／14。三项均有错误语义与正确参考的原完整oracle控制，见[审查范围](../semantic-audit/oracle-strength-20261028/README.md)；其余97题未被笼统宣称可靠。MySQL42的modifier共享则单列同一copy契约的边界增强，原oracle能抓目标mutation，未将它当作OSA式奖励失效或据此暂停。

后续又确认[Bag82](../semantic-audit/oracle-strength-20261028/bag82/README.md)与[Marshmallow88](../semantic-audit/075-099/marshmallow-bytes-oracle-r1/README.md)。前者自然set去重修复通过完整430cases／原verify_task，却把aab／ab的1算成0；使用明确依赖预载CPU profile解除首次Hypothesis导入超时，原失败控制、8–9秒成本和环境边界保留。后者自然剥bytes表示外壳通过完整1231cases及actual JUnit validator，却仍留下换行／引号／UTF8转义；该证据为direct pytest＋validator，未冒称新verify_task或模型reward。两项独立来源复核均支持未来RL暂停，正确SFT与旧reward保留；88现teacher已测中文bytes，无需替换。Bottle5 no-op reset则被原365scope明确抓住，候选关闭，没有扩大暂停。

R3真实501条raw output用原frozen严格parser复算，10个失败确属坏原生语法；这不等于比赛服务端也必须立即结束。后续[pinned官方组件差异](r3-official-parser-semantics/README.md)显示：483个原合法call的name/参数一致，8个无call不变，10个坏语法中官方组件仍提取8个call、2个文本。坏edit提取后仍缺old_string，不可补gold后冒称已修；坏submit仅提交既有workingtree，原reward不改。固定ADK非流请求和预算内最多3次nudge的源码链已单独[钉来源与边界](r3-official-request-path-r1/README.md)，未声称完整服务或新模型实跑。当前live-pair已有严格parse错误后的有限反馈，仍不是这套官方容错／结构化history的端到端等价。另从精确模板发现合法nullable参数会渲染为`None`，旧parser仅收`null/none`；已最小支持精确`None`，错拼仍拒绝，R3严格分类不变。此前[严格解析器证据](r3-parser-review/README.md)作为其范围内结论保留。

[38题批量revision](../question-revision-r2/README.md)首先冻结，62个其余task对象和question行保留原始字节；当时旧Linux CPU记录仅匹配62个未变task，38个新版本不追认旧usable。split_by按新版真实重放并导出[8条新决策、628监督tokens](../question-revision-r1/teacher-sft-r3/README.md)，OSA另有8条修正决策、954监督tokens；原S600/F600仍不覆盖。此后直接运行题面又发现[PyPika40导入缺陷](../semantic-audit/025-049/pypika-period-sql-teacher-r1/README.md)：修复生产代码后原MRE仍ImportError。新版只改正确public import，真实DELETE／UPDATE及原1061用例0/1完成，当前[question-revision-r3](../question-revision-r3/README.md)累计39题面修订，其余61题原问题不变。

后续进一步分离了prompt身份与固定补丁验收身份：[真实字段消费链](../semantic-audit/prompt-only-cpu-evidence-r1/README.md)确认当前官方verify不读取problem_statement。新兼容派生只接受SHA绑定的真实旧task对象与当前对象除该单一字段外完全相同，其他source／patch／isolation／bytecode及依赖profile判别不放宽。实际39份旧原件消费后，61项仍严格task匹配、39项仅verification兼容；旧行及奖励未改，没有把旧prompt称为新prompt，也没有声称新运行或runtime等价。缺原件或任何其他字段改变仍不兼容，避免单纯为了改写题面重复39次原oracle。

此前对当时剩余53历史teacher的实际messages／recipe／反馈作[行为复核](historical-teacher-behavior-review-r1/README.md)，没有发现纯import/compile两端成功的同类缺陷；Bottle4确实遗漏主目录分支，已补同一原patch／365用例0/1与真实保存行为。当前51题413条新决策＋49题392条历史来源构成[监督r6](../question-revision-r3/supervision-r6/README.md)，805决策／78,924监督tokens；48题384行仍为原S600字节，原formatter的CRLF读取修复持续生效，历史Hydra8行保留更正。最大context4649，19条动作／7题超4K，完整保留。此前对当时其余51篇MRE另作源码/API阅读，实际运行33／59／94后相关疑点已关闭，未宣称当时51题全量MRE实测；见[明确范围](../semantic-audit/current-supervision-source-review/retained-historical-MRE-API-review-20261028.json)。这是off-policy材料，不是Gemma行为／VFARL焦点对。

来源可用性复核发现11个原始source task blob缺失。8个moto从既有明确训练缓存单行精确恢复，另外3个按固定revision只读获取各自公开训练行，全部原SHA匹配。现100个原始来源pin可读，已额外冻结[357KB原始来源归档](source-pin-archive/manifest.json)，避免后续缓存清理再次丢失；这是来源修复，不是新验收或模型收益。8个moto旧示范的run_command仅检查导入/语法，baseline也exit0；现已全部补真实API行为断言及新的完整原oracle0/1，旧记录保留。

旧short-r1的混合grad/no-grad诊断已用其真正保存的pre-no-grad向量重建：[原始公式复算](../objective-audit-20261027/short-r1-mode-reconstruction.md)显示匹配no-grad锚点后仍为负方向，该模式差异只解释原loss上升量约0.01579%。缺post-grad向量，不能将BF16或backward认作已定根因。另用[真实Gemma4小模型CPU](../objective-audit-20261027/real-gemma4-cache-boundary-20261028.json)执行生产sliding/full cache pack/extract、停行退休和严格prefix复用，该边界selected logprob最大差2.38e-7，未发现新cache缺陷；完整sample_batch仍在CPU的CUDA显存统计处真实失败，不能外推Unsloth／BF16／31B。

对三条真实R3正确调用又完成[部署历史roundtrip](r3-deployment-roundtrip-r1/README.md)与[参数preprocess补链](r3-deployment-preprocess-r1/README.md)：ADK输出的JSON-string参数经同commit上游真实函数转mapping后，CT模板会重排参数并补thought末尾换行，下一输入各比原rawappend多1 token。该CT模板不读取preserve_thinking，保留reasoning字段时thought仍在，不能认定官方默认丢thought。这是静态组件输入差异，不是端到端比赛或模型收益；原500余次采样tokens/probs未改。

真实小Gemma4另证明[前缀KV恢复机制](../objective-audit-20261027/real-gemma4-sliding-prefix-backup-20261028.json)：pre-action只备份sliding尾部、full cache裁回P后可准确prefill不同canonical tail；错误地只改sliding逻辑长度会不符。full crop保留backing storage，未虚报立即释放显存。显式opt-in [v5缓存接口及CLI](../objective-audit-20261027/session-prefix-rewind-v5.md)已接线且默认关闭，原v3/v4 profile保留。canonical history helper及collector新模式已完成[真实CPU工具循环](canonical-history-cpu-episode-r1/README.md)：原oracle baseline 0，修复后1；坏调用保留原sample并经实际nudge退回raw，后续正常修复/提交。精确loaded-source证据绑定run.py的b23f1fa7…版本；此后另纳入深嵌套解析异常反馈及提交后最终回答机会，均单列来源，旧CPU运行不追认为新源码实跑。该工具循环没有模型采样或Snapshot分叉；另有[真实observable Snapshot恢复](canonical-history-boundary-review-r1/README.md)证明nested ledger／ctx／prefix相等，明确不可充作训练pair。默认仍raw，没有31B cache实跑或完整官方部署等价结论。

另复现两条构造输出可触发的解析边界：`1e309`经JSON成为inf，会在后续严格JSON归档崩溃；563-token完整调用中的1100层嵌套会抛RecursionError。前者现按非有限值直接产生解析反馈，后者纳入既有有限parse反馈，不修改原sample或增加奖励；真实501条R3输出分类与参数全部不变。见[数值溢出](../objective-audit-20261027/native-numeric-overflow-20261028.json)及[深嵌套反馈](../objective-audit-20261027/native-recursion-feedback-20261028.json)。这两项不是已观察的Gemma错误频率或模型收益。

三例[题面错误与实际R3模型轨迹关联](../semantic-audit/r3-prompt-error-linkage-r1/README.md)进一步说明修订依据：25/26 Tenacity原MRE的不适用API被真实模型采用并报错；6 Bottle旧操作顺序不触发，模型调整顺序才复现。只统计实际历史动作/token开销，不将其当作新版节省或修复率收益。

还形成了独立[恢复-only辅助视图](../semantic-audit/execution-recovery-prefix-r1/recovery-only-sft-r1/README.md)：两条Codex受控错误前缀后12个真实恢复决策、1060监督tokens。错误前缀只作输入，完整历史保留，不能按全episode监督；现已另导[独立trainer输入](../semantic-audit/execution-recovery-prefix-r1/recovery-only-sft-r1/trainer-input-r1/README.md)，真实reader／CT协议／CPUcollator消费一致。独立protocol为vfarl-controlled-prefix-recovery-v1；未混入100题主SFT，未伪造Gemma样本／焦点对／干预状态奖励。

## 实验执行状态

本地手动机制实验和上述macOS CPU真实工具／原oracle重放已运行。用户已明确允许jiyujia私有CPU及短GPU零更新验证；实际只使用该账号专属凭据，默认CLI的另一账号未用于提交。此前自动审批拒绝作为历史记录保留，新授权后执行没有再次受该拒绝阻断。

当前四题私有CPU r4真实提交后，在任何题目验收前因实际输入运行时缺少`stage/runtime/bin/uv`失败，不能记成模型或修复0分，见[r4实际失败](../cpu-runtime-repair/execution-r4-20261028/README.md)。r5实际完成并恢复UV，但stock依赖cache缺失导致准备失败；旧prevalidate继续沿用原runtime，四题8次原验收仍缺sure／colorama／bs4，全部reward=null。导出文件存在不代表runtime修复成功，见[r5实际结果](../cpu-runtime-repair/execution-r5-20261028/README.md)。其[真实Linux进程预算观察](../cpu-runtime-repair/execution-r5-20261028/budget-probe-live-observation.json)已确认子进程及独立session孙进程在1.570秒内被终止、回收，timeout返回124，cleanup confirmed、无survivors；这个CPU观测只支持远端预算机制，不是GPU数值／任务修复收益。原4个unknown和课程暂停继续保留。

已准备[零optimizer数值研究r4](../numerics-20261028/README.md)，用于补reference、测真实head和backbone梯度；原first4只覆盖thought开头，现两动作至多8行按结构覆盖thought/tool/尾部。另冻结20分钟总预算的监督wrapper，原37项源和数值payload未改。2026-10-28 09:40（UTC01:40）共享Codex额度显示100%，在任何GPU push前撤下提交计划，仅收尾已经运行的CPU r5；实际新增GPU提交为0，未使用额度重置或购买。尚无新的31B探针、参数更新方向或独立修复收益。

第一轮额度终点的待办已在用户明确恢复后继续推进：CPU6修复验证环境，R7在明确私有导出授权后保存925,108,005字节运行时，SHA `f13237d48dd0adecb3556334106836002290efa2b02987061f17b606771be44c`。reference及零更新数值实验完成。当前优先复现首动作执行模式差异；新运行时实际接入前，不解除仍消费旧0a8运行时的采样暂停。正式训练保持关闭。截断恢复、同状态逻辑预算与focal-only估计偏置仍是研究边界，不将实现完成当作研究收益。

方法对照阅读：[verl rollout correction](https://github.com/verl-project/verl/blob/main/docs/algo/rollout_corr.md)将采样/学习执行差异与参数更新漂移分开处理；这里保留实际行为概率，不把重评分当作采样时概率。[GiGPO原论文](https://arxiv.org/abs/2505.10978)同时组织episode与状态组优势；本项目的显式分叉、focal-only目标与其设计不同，不能照搬其无额外rollout的成本结论。

新增[当前论文与目标对照](../objective-audit-20261027/current-methods-comparison-20261028.md)：SPROUT的指导式分叉、完整trajectory-return组中心化和future-KL目标均有独立采样／恢复／覆盖条件。两步精确CPU枚举只说明它们与当前focal-k3/TIS目标不同；没有据论文更名目标或声称当前实现获得其保证，也未盲改loss。

另用真实reducer作[异奖筛选k3的三动作精确枚举](../objective-audit-20261027/mixed-group-k3-selection-20261028.md)：自己的mixed固定数据正则下降时，两种完整reference KL仍可同时上升。该证据与现声明一致，不是backward错误或旧short-r1根因；全cohort方案还涉及同奖cohort跳过、Adam时钟和成本，尚未据此直接改loss。数据生产另补[可选selection-bound prompt配置](prompt-selection-profile-r1/README.md)，保留int60／float30.0原类型并在工具开始前绑定实际prompt，防止有效执行却落在错误上下文版本；旧默认及冻结载体不回写。

## 持续研究记录

截至2026-10-28 05:32（UTC前一日21:32），账号共享周窗口已用80%，本轮起始读取46%；这不是对本chat独占token成本的精确归因。未使用额度重置、未新增Kaggle作业／GPU消耗，正式训练仍关闭。当前100题推荐已更新到r3及其真实trainer输入；独立跨版本复核确认旧R3行为身份仍被识别为不同执行路径、pending pair仍因缺真实reference拒绝，冻结r4不会自动混入后来v5/history源码，见[来源与消费链](../objective-audit-20261027/cross-version-profile-and-trainer-r3-review-20261028.json)。默认采样历史仍raw，新模式没有31B／比赛收益结论。后续继续寻找贴题的明显错误实现是否会被原oracle误收；尚未跑出的候选不增加弱oracle题数或暂停名单。


提交结束语义的进一步审阅推翻了此前“官方提交后立即结束”的判断：[pinned官方／ADK链及805行census](../semantic-audit/post-submit-terminal-scope-r1/README.md)表明普通submit响应仍可触发最终回答。99条监督最终回答均来自真实成功submit反馈，应继续保留。当前live采集器已移除提前break，在原预算内请求后续回答，no-call不再nudge；[新源码真实CPU raw及canonical循环](../semantic-audit/post-submit-cpu-episode-r1/README.md)均为5工具＋提交后第6次final采样，完整原oracle0→1。随后[多提交CPU证据](../semantic-audit/post-submit-cpu-multisubmit-r1/README.md)又确认live可继续工具／再次submit：第一次错误提交的原oracle0，修复并再次提交后1；另一case修改workingtree却不resubmit仍评估已保存的正确提交。11条真实R3合法postsubmit命令原被guard停止，原reward均1，未据此声称增益。historical guard保留，严格解析仍不同于完整ADK；旧episode不追认新源码。

当前31B官方QAT-v2 NF4 cache已有SFT loader路径；此前怀疑的12B限制并不适用于31B。新[SFT单步配置草稿](../../../../../training/configs/sft_vfarl_current100_31b_nf4_r32_draft.README.md)显式r32／alpha64／adapter_init_seed=0并读取监督r6；single／FSDP／DDP只在显式声明时传入现有loader，VFARL继续用model.seed且拒绝该SFT专用键。真实小Gemma4+PEFT CPU证明独立adapter seed跨base随机消耗仍复现，resume保留已存adapter；没有31B本地完整权重加载、反向容量或新阶段训练结果，草稿不构成启动授权。

另基于冻结监督r4对重复历史形成了[固定cohort前缀共享证据](../objective-audit-20261027/sft-prefix-sharing-20261028.md)：705条决策是各自最长行的严格prefix；100个首动作另有prompt-only空thought控制，不能删除。内存200行候选保留全部78,148个目标，输入位置总数从2,054,401降到476,745；真实小Gemma4 CPU的一题8→2行对照，固定token-pooled目标的loss差1.49e−7、最大梯度差1.68e−7。当前SFT按实际accumulation window归一化，直接换200行会改变随机组批与Adam时钟，所以未替换推荐数据、修改训练器或宣称31B速度收益。


2026-10-28 07:02（UTC前一日23:02）继续记录：账号共享周窗口已用88%，起始46%；没有使用重置。推荐监督已到r4／805行／78,148目标，当前课程仍r6／5项已证实弱oracle。新完成native执行步与adapter提交的分离、提交后正常final机会及推理异常时的真实patch／验收诊断保留；后者仍fatal且不训练，见[独立异常CPU证据](../semantic-audit/post-submit-cpu-sampling-error-r1/README.md)。新源码与旧正常proof分别冻结，不追认。原r4零更新GPU载体没有被本地后续改动重打包或提交；当前本地研究继续，尚未测得新的模型或比赛收益。


另两项完整原oracle盲点已独立核对：[Addict56](../semantic-audit/050-074/addict-setdefault-oracle-r1/README.md)的truthiness实现会覆盖已有None／0／False／空值，却通过全部128用例；[Hamming81](../semantic-audit/075-099/hamming-truncate-oracle-r1/independent-review.json)忽略题面truncate=True仍通过430用例／427required。二者均有真实baseline0／correct1／wrong1及源绑定。Addict正确teacher本来就能捕捉错误，实际R3补丁也使用key membership而非该wrong探针，故不改原奖励或SFT；Hamming另补同一正确patch的True分支观察。wrong探针不导入训练，当前七题暂停只针对未来原oracle RL。


同一前缀优化又按现有seed0真实shuffle逐个保留optimizer window做了[输入位置成本核算](../objective-audit-20261027/sft-prefix-sharing-optimizer-window-20261028.json)：当前单步草稿是micro1／accum1，窗内没有可复用前缀，不能拿全100题的76.8%位置减少冒称其收益。只有在同一原窗口内合并严格prefix且目标位置不重叠，才可保留原采样／Adam边界；尚未实现运行时合并，selected-head工作也不减少。


当前单次更新的数值边界又已核清：[PPO anchor审阅](../objective-audit-20261027/ppo-current-anchor-overflow-boundary-20261028.md)表明有效首步的token log-ratio受±0.08约束，ratio约0.923–1.083，在当前epsilon0.2内；这个首步梯度不会受到PPO clipping，不能用调epsilon保证实际Adam步的有限loss下降。新native只补NaN repeat可能绕过max比较的有限值处理，在任何backward前一致拒绝；没有改PPO／KL／IS。GPU数值探针仍未执行。


旧short-r1又完成[真实概率向量的clipping归因](../objective-audit-20261027/short-r1-clipping-attribution-20261028.md)：同一frozen anchor／reference／FP32 rho／动作均值下，匹配no-grad目标上升约0.000383946；去掉clip作诊断后仍上升0.000334480。真正active仅545个selected tokens中的2个，clip保守项占原上升12.8836%；其余是unclipped PG及同KL。pre-grad没有active clip，post诊断clip不解释已观察的负logp方向或更新梯度。缺post-grad，仍不能定位backward／有限步根因，更不能据此调epsilon或重复训练旧批。

另将真实R3 Glom49的[早停样本作恢复机会对照](r3-glom49-early-termination-recovery-r1/README.md)：旧v4题面及1294＋59真实token/µ原件不变，historical forced replay后直接结束、原178用例得0；当前live产生1条真实nudge，再执行脚本化正确suffix后178全通过。它保留的是旧问题版本，未冒充当前修订题面或新的Gemma采样；新facade全部mask0、referenceNone，无pair／主SFT导出。第一次CPU重建漏了原glom CLI入口的环境失败原件保留，不作有效模型0。

[冻结r4载体差异复核](../numerics-20261028/carrier-r4-freeze-delta-20261028.json)再次确认37项原内嵌文件与payload SHA不变，工作区后来已有10项源差异；该零更新包不包含后续postsubmit／canonical／v5／native有限值与partial commit／当前SFT数据等新路径。没有重打包或提交，也没有补成真实reference／31B结果。[SFT r6单步草稿实际dry-run](../objective-audit-20261027/sft-draft-r6-dry-run-20261028.json)只读805行／78,924目标，未加载权重或训练依赖。


基于冻结r5的[命令脚手架审阅](sft-command-scaffolding-review-20261028.json)显示200个run_command占50,641标签；纯来源打印8,220，约全部监督10.48%，但来源包络含必要import／路径断言，不能整体当噪声。另明确0／39修前只import、39修后只私有helper，已用新真实teacher补齐。[Dominate0](../semantic-audit/000-024/dominate-visible-reproduction-teacher-r1/README.md)同API修前观察错误IndexError文案、修后正确文案，原65例0→1；[Mistune39 r2](../semantic-audit/025-049/mistune-public-renderer-teacher-r2/README.md)在所选命令里仅捕捉预期IndexError并报告真实源码末帧，再统一断言，因此原945例仍0→1。独立公开MRE候选r1的完整trace保留；r2单题输入34,368→18,450、最大5623→2998，但标签551→885。r6相对r5总输入反增4,626至2,062,458，不冒称整体缩小或模型提速。

2026-10-28 09:40（UTC01:40）达到用户指定的额度终点：[账户读取](quota-end-20261028.json)显示共享周窗口100%，本轮起点46%，不是本chat独占消耗归因。停止新增研究与GPU／CPU提交，只收尾已运行CPU5并保存其真实结果。当前推荐监督r6、课程r7及全部历史来源不再自动变更；没有重置额度、正式GPU训练或新模型收益结论。

随后用户明确继续新一轮额度，已产生两项真实结果。CPU6只为四题准备原锁定的三个stock环境，联网安装后恢复UV_OFFLINE=1，实际chroot验收全部baseline0／repair1：Moto5348为7用例、Moto6159为69、DVC为34、Markdownify为75，修前失败分别1／1／31／1，修后全部通过。任务／source／patch及新版54公开prompt均绑定；历史null未改。维护中的TaskEnvironment现在在沙箱启动前拒绝明确准备失败，正常官方exit1规则不动；[最小修复证据](../cpu-runtime-repair/runtime-consumption-repair-20261028/result.json)与真正Linux结果分别保存。

GPU零更新实跑的[原始数值](../numerics-20261028/execution-r4b-resumed-20261028/actual-output/vfarl-train/numerics/numerics-study.json)使用两条2415／2413-token真实动作，完整backbone非零梯度norm0.423103。8个结构选位的head对照中，fused对分块FP32的logprob最大差9.06e−6、hidden梯度相对L2差0.002759；普通BF16 head对应相对L2差0.09642。它支持该局部fused实现较贴近FP32，不能证明整个backbone反向正确或直接改用普通BF16 head。

同一次实跑首动作grad／no-grad logprob最大差0.070849，第二动作仅9.59e−6，需独立复现模式／首次调用／reference零scaling切换的影响。沿实测未裁剪梯度的负sign方向，暂态epsilon1e−6预测目标变化−0.00242085、实际+0.00042729；epsilon5e−6实际−0.00091673。探针anchor与原grad-forward目标完全一致，adapter参数最后精确恢复。这不是实际Adam更新：其epsilon、裁剪及已有moment会改变方向，不能据此挑选学习率或称旧short-r1根因已定。实际GPU作业在20分钟界限内自行结束，无取消，额度差0.548826小时，最终预留0／剩余4.74402小时；诊断没有消耗该行为批的训练机会。
