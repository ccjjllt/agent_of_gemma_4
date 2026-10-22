# 006进展：完成强参考对照、三仓库真实重放与12B CPU适配

> 目录整理（2026-10-22）：本页历史载体路径保留当时记录；当前文件位置和重建入口见 [Notebook 目录导航](../../notebooks/README.md)。

2026-10-13，北京时间。用户授权实施后，已恢复10月11日的执行收尾实验，完成E2的12次真实GPU对照、三个外部仓库的CPU工具重放与12B Unified代码适配。E2两臂结果为exec_closure **4/6**、budget_fit_5m **5/6**，完整归档已取回。没有教师调用、新正式submission、真实12B预训练权重训练或heldout读取，没有新增测试套件。

## 已完成的研究与结果恢复

- 官方API确认submission57091482为COMPLETE，公开分0.13、私榜未返回；其无adapter的single_state提交包保持冻结。没有隐藏逐题反馈，不能把总分归因于单一机制。
- 核对31B/12B/E4B/E2B新规则、同提交同变体约束，取得官方12B来源/部署小资产和v31实际源码差异。公开完整源码可核实参考仍为0.18，没有找到对应0.20+的完整方案。
- 完成10份Notebook源码与得分版本核对，以及Sentry、SWE-Replay、SWE-AGILE、SWE-Lego、Agentic Rubrics的一手资料研究。外部分数和论文涨幅不视作本项目效果。
- 远端发现并恢复`p005-exec-closure-pair`：**10月11日已经完成**12次，A single_state为3/6、B旧exec_closure为4/6；新增fastapi_5077仅在B第二次成功，没有成功回退，rich_3518节时方向不一致。三题各两次，不是六个独立任务；12条轨迹均未调用新增技能。[原件与分析](../../research/006/recovered-exec-closure-20261013/results.md)。

恢复实验的A15文件与正式0.13包逐字节一致，B仅改prompt。10月13日新closure_prompt措辞不同，仍是未运行草稿，不赋予旧B的4/6成绩。为避免重复机制筛选，不再运行原计划的近似提示E1；≤1.5额度小时转用于E2强参考比较。恢复旧结果没有产生本日新GPU费用。

## E2已完成：执行提示版4/6，五分钟公开参考5/6

A使用旧exec_closure冻结ZIP原样；B从精确Budget-Fit V5（355310852）取四文件，仅将8分钟改为5分钟并同步两处prompt预算句。B保留28调用/240秒命令/8192输出预留，A保留50调用/60秒命令/16384输出预留；均thinking4096、80turn、Agent5分钟。**这是整体方案对照，修改后的五分钟参考没有继承原0.18成绩。**

[生成器与载体](../../notebooks/006/006-reference-pair.README.md)已完成必要手动核对：两份冻结来源、12个唯一attempt、每题每臂两次且r1/r2交换AB/BA、内外代码语法及Notebook内容一致。v31关键wheel均固定实下载指纹；adk-eval-core归档hash变了，google-adk实算保持原值。精确V5的Apache 2.0页面许可、来源及修改记录随研究产物保留。

任务只来自已归档四题子集中的fastapi_14873、fastapi_5077、rich_3518三行，运行不读取或哈希官方完整任务文件，不动heldout28题、预留八题及其快照。一次31B CT v2加载、最多两个隔离Evaluator，共12次；v31最终验证显式300秒，两臂命令时限各自保留。45分钟GPU会话按4×L4两倍系数估算上限1.5额度小时；内部评测2400秒截止、coordinator2460秒强停、2535秒归档截止，目标2550秒内结束，失败不自动续开。

私有CPU准备V1真实完成：157.03秒、两臂构建成功、模型请求/裁判均0。GPU `ccjjllt/p006-reference-pair` V1随后完成12/12次，wall2134.73秒（35.58分钟），模型只加载一次，没有超出2700秒硬限或追加会话。[实际结果、原件与额度](../../research/006/reference-pair-20261013/results.md)。

逐题A/B：fastapi_14873为1/2对2/2，fastapi_5077双方1/2但成功repeat相反，Rich双方2/2。B净增1次，具体为2次配对新增、1次回退，不能说没有回退。Rich的B两次均更快；所有3次失败都达到Agent五分钟时限。12条trace仍未调用新增skills或图工具，不把收益归给它们。B作为下一轮跨仓库优先参考，A保留；不再用这三题继续选择提示。

[逐轨迹机制分析](../../research/006/reference-pair-20261013/mechanism-analysis.md)把三次失败区分为：弱断言把ForwardRef当成功后只改复现脚本；持续重读/改复现而没有生产修改；生产文件只增加DEBUG输出而没有修复。Rich四份正确补丁完全相同，却因错误的复现判据及编辑参数循环耗时不同。因此优先研究能区分目标行为的局部观察与证据成立后的及时行动，不能把首次编辑更早或自建案例变绿当作收益本身。

本实验按wall两倍估算1.186额度小时，账户总变化1.600小时；同时期另有ARC3项目GPU会话运行，不能将总差全部归本实验。完成后读数剩1.688额度小时，为全账户瞬时余额。此轮没有新正式提交，公榜仍0.13。

## CPU数据线已覆盖三个独立仓库

[数据重放报告](../../research/006/data-replay-20261013/report.md)记录两个独立仓库/问题、一例真实工具恢复：

- click：固定buggy源码下，stdout/stderr缺失的两种行为修改前抛AttributeError、修改后不抛；显式StringIO输出保持正确，官方submit_patch实际捕获一个生产文件修改。
- pydicom：从明确的中间状态开始，官方edit_file真实拒绝多处匹配且文件不变；读取当前源码、补充docstring锚点后exact一处成功，并实际回读。这是局部恢复，不能当原完整轨迹无损裁短或自主定位成功。

两条在31B/12B部署CT模板下完整编码，输入/监督tokens分别为3368/519和2055/226，共745个监督tokens，无截断；这两个样本的token IDs恰相同，不能外推模板普遍等价。两条`task_resolved`都为null，未执行完整benchmark；上游resolved标签、局部行为检查、整题修复和训练收益分别记录。

已纠正SWE-smith初始状态理解：profile commit只是环境身份，loader实际checkout instance_id。本轮固定了真实buggy instance commit与源码/许可，而非使用已包含修复的profile源码。已冻结click仅作train容量、pydicom仅作select诊断，分别保存编码文件；原合并文件不作为训练输入。当前尚未用于真实训练或独立泛化评价。

等待GPU期间，进一步完成[MarkupSafe实际重放](../../research/006/data-replay-20261013/markupsafe-report.md)：两次过期old_string失败后刷新源码，工具exact编辑成功，但程序仍报SystemError并破坏安全HTML语义。根据新观察继续修改后，同5个inline行为案例从2/5变为5/5，官方submit捕获一个生产文件。保存负过程和包含真实负前缀的新继续，分别4153/295及6968/688输入/监督tokens。两阶段是同一个问题、一个恢复事件，都为candidate且禁止自动参数更新；所有外部记录仍task_resolved=null。这是局部行为收益，不能写成官方整题通过或比赛涨分。

## 12B Unified CPU适配已经实现，真实12B训练尚未运行

[适配结果](../../research/006/12b-cpu-adaptation-20261013/results.md)包括Unified加载、selected loss、RMSNorm、文本LoRA导出及固定来源配置；官方CT v2模板/tokenizer已获取并固定，12B拒绝复用31B缓存。

一次性手动CPU计算使用随机微型模型：loss及LoRA梯度与原HF路径相符，完成两步Adam、checkpoint恢复和独立PEFT导出，基座未变。终审另修复checkpoint未绑定基座身份的问题：同一已校验基座换路径可恢复，manifest或权重变化会拒绝恢复；单卡与DDP在写元数据前核对，DDP尚未多进程实跑。完整12B配置仅在meta device构造，得到48层、240个q/o+MLP目标、r8共28,737,536参数。没有下载/加载12B预训练权重，没有CUDA/NF4容量结果，也没有官方CT adapter推理收益。真实数据只做协议与长度核对，没有训练。

## 下一步与边界

本轮唯一GPU实验E2已结束，不追加12B GPU训练。下一阶段先用冻结B作跨仓库强参考，保留A对照，再安排固定12B真权重上的单样本容量与独立导出。CPU继续扩大可重放的独立任务。E2三题净增1次只支持扩大研究，不能换算为0.18、私榜成绩或奖牌概率。旧0.13之外没有新正式分数，历史3/6→4/6、局部工具恢复和微型CPU优化都不能混称新模型修复收益。

完整来源、预算及后续跨仓库安排见[006计划](../plan/006-from-013-to-competitive-agent.md)。
