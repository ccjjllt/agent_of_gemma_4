# 零更新数值研究

2026-10-28。本轮明确不做正式GPU训练。`--numerics-only`是实际模型数值实验入口：最多接收一个已验证异奖组的两条真实动作，补齐缺失的冻结reference，做完整焦点目标的零更新反向，并隔离真实词表输出层比较融合实现、普通BF16实现与分块FP32计算。没有创建optimizer、没有step、没有新rollout，也不导出新adapter。

最新执行：账户额度窗口变化后，用户明确授权继续新一轮研究。UTC02:38单次私有提交的`jiyujia/vfarl-focal-numerics-zero-update-r4-bounded/1`已于02:57自行COMPLETE，见[实际结果](execution-r4b-resumed-20261028/README.md)。0 optimizer／0 rollout，adapter前后hash相同，缺失reference真实补齐，backbone梯度非零；观察到首动作grad／no-grad差异以及epsilon1e−6负sign梯度扰动使目标上升。完整原件与边界已保存，未据此选学习率或宣称训练收益。实际4×L4分配、worker只用其中2卡，20分钟界限内结束，无取消；最终GPU剩余4.74402小时、预留0。下文第一轮额度终点的0提交记录是历史时点，不代表此次新执行。

后续[r5模式观察](execution-r5-mode-order-20261028/README.md)取得A-first进程13次真实前向，B-first在N0开始时触及外部观察预算并取消，最终CANCEL_ACKNOWLEDGED；不称26次完整顺序对照。A、B两动作的G／N最大logprob差为0.060098／0.055105，hidden相对L2差约1.78%／1.88%；同模式重复的LP及hidden完全相同，reference重复及插入前后也相同。[独立分析](execution-r5-mode-order-20261028/independent-a-first-analysis.json)证明本次差异持续存在，不能用热身一次解释，也不能只归因head。零backward／optimizer，adapter哈希不变。最终GPU剩余4.24765小时、预留0；取消后的平台确认／发布延迟如实记录，不能宣称20分钟内已确认停止或取得wrapper cleanup证据。

新的[checkpoint源审阅](checkpoint-source-review-20261028.md)确认首次backward会改变GC计数／FIRST_PASS等持久状态，r4日志也记录双缓冲启用。G前向后直接detach不会恢复这些计数，故旧方向探针缺少固定权重重复控制；原测量值保留，不能把有限差分全部归因于参数扰动。后续R6已完成一次真实backward前后观察，并读取实际offload dtype，结果如下。

后续[R6一次backward实跑](execution-r6-backward-mode-20261028/README.md)已正常COMPLETE：13次前向、1次原A目标backward、0 optimizer，adapter哈希前后相同。A的G及reference前后完全一致，A的N0→后续N1最大LP差0.130252；后测N/G差仍为A0.052665、B0.168945。A后测隔着B的5次前向，不能单独归因于backward；一次backward也没有使模式相等。[独立复算](execution-r6-backward-mode-20261028/independent-backward-mode-analysis.json)同时核对了原scalar及rho，A-only梯度范数不与旧整pair直接比较。

R6的真实graph暴露60个Unsloth checkpoint节点，已offload的节点全部BF16→BF16；post B为59／60 offload，仍全同dtype。本例没有触发FP32→BF16保存路径，故不采用该精度开关作为本例修复。相反，N0／R0在外层no-grad时仍各分配一个offload槽并增加GC计数，G0后LAST为62；首次post B的N调用还开启双缓冲。这是实际副作用证据，但不证明其唯一解释概率差异。

[R7窄干预](execution-r7-no-grad-checkpoint-20261028/README.md)已于UTC05:25正常COMPLETE：21次前向、1次原A目标backward、0 optimizer／参数扰动，adapter hash不变，cleanup确认成功。只在候选N／R作用域绕过60个text层的checkpoint，每个scope实际120次direct调用、GC完整状态前后相同；G前精确恢复原callable。两动作的legacy／candidate／restored对应LP向量及hidden SHA完全一致，N/G最大差始终为A0.060098、B0.055105。**候选没有改善数值差异，不推广为训练修复。** A在本次backward前后对应N/R/G也全部相同，没有重现R6的N变化；R6还有中间B调用的混杂。见[独立分析](execution-r7-no-grad-checkpoint-20261028/independent-ng-checkpoint-analysis.json)。

[跨运行原件比较](cross-run-mode-comparison-20261028.json)确认R5与R7六个A/B N/R/G结果逐值一致，R6对应结果均不同；同一已记录base／adapter／依赖身份不等于同一实际执行图。现有metadata没有Gemma4实际生成forward的路径或SHA，不能据固定依赖文件SHA定位编译根因。R7干预发生在原A backward之后，未消除此前全部状态历史，也不能排除所有GC相关机制。当前主训练默认、loss、精度及学习率均未改变。最后GPU剩余3.463413小时、预留0；R4b/R6/R7正常完成，R5取消已确认。没有新GPU作业或正式训练。

[实际权重复算](anchor-adam-review-20261028.json)进一步量化其影响：R5/R7的A、B最大token差0.06010／0.05511均低于现有0.08容限，但sum(G−N)为0.118332／0.016008。按真实行为概率、原cap2和FP32落地规则，N锚点的动作importance为0.946509／0.967813，G锚点为1.065407／0.983430，相对变化12.5617%／1.6137%，均未触发cap。逐token容限不约束动作权重误差；改用G锚点会改变现有目标，不能称代数等价修复。R6的B超过0.08，生产入口会拒绝，所列权重仅是已保存诊断向量的解析结果。没有因此改容限、锚点或loss。

输入为已回收的Tenacity R3原始0/独立分支1。行为tokens、old logprob、mask、策略身份及终局奖励不重算、不改写。补reference使用独立真实前向；原件保持pending，新评分另存。使用R3实际初始adapter（short-r1已导出权重SHA `cc45f7d1e6bf7a40f98d1d92e27e9a1022ea110a67565aba0c908aec77abbfe6`），源数据SHA `aa988529c4b40947172cd42fa64ff23bf0a2b861c7903b4fb3575196301f9611`。本次研究不消费这批数据的训练机会。

词表诊断采用真实梯度前向的结构分层行：每动作至多4行、两动作总计至多8行，覆盖thought body、native tool body及响应尾部，按源token结构选位；不按奖励、概率变化或梯度选点。待发两条动作的selected索引为[3,37,86,98]和[3,35,82,96]。原先first4全为channel／thought／换行／首词，不能代表工具部分。hidden、target与同一完整RL目标的实际上游系数使用完全同一索引，汇总后一次head诊断；保留原冻结tied head，不用另一次no-grad hidden。FP32参照逐4096词分块，不复制整张FP32词表；临时关闭TF32、使用highest精度后恢复原设置。源码与只读实际输入核对见[结构选位机制](../objective-audit-20261027/head-source-strata-mechanism-20261028.json)。完整backbone反向与head隔离分别报告；8行形状仍不同于完整焦点head，不代替31B NF4证据。

R4b反向后另做两个暂态中心差分探测：沿真实未裁剪梯度的负sign方向，分别扰动FP32 LoRA存储±1e-6、±5e-6，在冻结old/reference/IS下重算同一标量目标。探针沿与反向相同的grad-enabled前向路径计算，立即detach释放图，绝不额外backward；不能仅凭两条路径在anchor数值相同就将no-grad有限差分称作反向核对。该方向接近新Adam首步的epsilon→0极限，不代表续接Adam的moment方向。每次从原始tensor重新放置参数，最终在`finally`中恢复并核对精确hash；即使前向中断也执行恢复。不构造optimizer，不接受、选择、导出或训练任何探测参数。分别保存探针anchor与反向anchor差、实际有限差分与一阶预测；BF16不光滑时不将它解释为训练收益。最终再次核对adapter真实tensor hash未变。后续源码和状态观察补充了限制：这些前向未恢复GC／编译状态，也没有固定参数重复与恢复anchor控制，参数恢复不能代替执行状态等价。

本地一次性CPU实验使用随机48维、8201词词表，目标跨4096分块边界，覆盖softcap=0/5及非零bias。解析分块梯度对完整FP32 autograd的相对误差最大约1.27e-6；pinned融合CPU梯度余弦约0.999999。实际结果见`cpu-head-study.json`。这没有运行31B、CUDA/Triton或证明奖励方向改善；不新增测试文件。

`cpu-directional-probe.json`记录微型CPU策略对有限差分、前向故障后精确恢复及非FP32存储拒绝的机制核对；`cpu-binding-math-review.json`记录真实回收pair的learner后端归因修正、三档数学设置恢复和故意区分grad/no-grad hidden的输入位置核对。它们不替代真实模型验证。

`cpu-probe-mode-confound.json`保留两条前向在anchor数值相等、导数却反号的CPU反例；修后实际探针函数的有限差分与预测同号，正常及中断后所有参数和`.grad`字节均保持原样。

`carrier-r1/`为较早冻结的草稿，源SHA `2f1a01548d90425115c485aca2991332eb6b9288dc715898cd23b5097003d1ac`，尚不包含上述全部暂态探测和复核修复，不应作为最新诊断代码提交。8K仅为完整覆盖该约2.4K序列的运行上限，不作16K反向证据。计划平台会话上限20分钟，启动前须读取实际剩余配额；不重复提交未知结果的请求，也不切换账号或付费机器。

`carrier-r2/`已冻结上述数值修正，源SHA `f68707c02828d4f1062b6929e7ea057b3dbac167d09a87806e4dc13977ca000e`，预定私有kernel为`jiyujia/vfarl-focal-numerics-zero-update-r2`。它不含后续课程／CPU来源／segment接线修复，作为中间草稿保留；最新冻结见下方r3，所有载体均未提交。

`carrier-r3/`在生成当时冻结36份源／配置／原始输入，源SHA `7f61bc37bfc35b45443f45d5001a5dd869b9645326490027f8ed9223607286d4`，预定私有kernel为`jiyujia/vfarl-focal-numerics-zero-update-r3`。压缩bundle逐项hash与当前源及真实prevalidation别名核对一致，NUMERICS_ONLY=True、原始pair SHA不变；详见[本地冻结核对](carrier-r3-local-inspection.json)。r1/r2保留为更早草稿。此后源码已新增bytecode配置与消费适配等修正，r3不再代表当前完整源码；下一冻结版正在准备，所有旧载体均未提交。

原冻结数值payload为`carrier-r4/`：源SHA `ca576bc72a5dd8a41b88b80d7594b045e636240d0add259cb42c753573145ac0`，私有kernel计划为`jiyujia/vfarl-focal-numerics-zero-update-r4`，冻结当时37份源／配置／原始输入并逐项核对。包含两动作8点分层head探测及已有bytecode模块导入依赖，原pending pair、缓存NF4/runtime/adapter均不变；[冻结核对](carrier-r4-local-inspection.json)。原r1/r2/r3保持草稿，r4之后的可选cache研究不回写这个已固定payload。

启动辅助函数现在对numerics_only请求强制至多1200秒，采用本次短诊断所需配额再加5分钟预留，不再继承旧resident七小时timeout／最低配额；较短显式时限保留。已有receipt还必须绑定同一源SHA才复用。实际函数只用模拟API完成[离线预算核对](launch-budget-evidence.json)，没有connect、凭据读取、网络请求或GPU；本地SDK确将timeout写入session_timeout_seconds，平台实际接受／回读尚未发生。builder此前显示的raw CLI缺timeout，现numerics建议命令也显式追加--timeout 1200；实际CLI源码支持该秒数参数。r4最初命令metadata另存carrier-r4-initial-command-manifest.json，修正只影响命令说明、run.py／bundle SHA未变；配额读取和5分钟预留仍优先通过bounded launcher完成。

用户随后已明确允许jiyujia私有CPU／短GPU诊断提交，原自动审批拒绝仅作为历史记录，不再是当前阻断条件。实际云端CPU暴露了原运行时缺UV及离线cache不足；这些没有被改记成模型失败。MODE=train的GPU数值启动路径不调用该CPU修复分支，原数值payload未因此重写。

最新预算载体为[carrier-r4-bounded](carrier-r4-bounded/)，run.py SHA `6796849539e4bdd27d53e0425a052dfb7184a96a9dc775eeb84c5c311cbbddf3`。其stdlib wrapper精确内嵌上述ca576原件，37项源、输入及缓存依赖不变；1180秒执行＋20秒cleanup预算，使用Linux subreaper、pidfd和实际后代身份回收独立session子孙。可读control SHA `fd25b6732986acd99ecfe8cb3d0d5f192a994362d8e8739297e258eb8c9412d4`。相同control在已运行的CPU5中[实际观察](../cpu-runtime-repair/execution-r5-20261028/budget-probe-live-observation.json)：子进程及setsid孙进程超时后被回收，返回124、cleanup confirmed、无survivors，外层1.570秒小于预声明3秒。该证据不包含模型、CUDA、GPU数值或训练更新。

第一窗口历史记录：2026-10-28 09:40（UTC01:40）共享Codex额度显示100%，达到用户设定的持续研究终点，因而当时在GPU push前停止新增实验。[当时收尾](execution-r4-20261028/final-no-gpu-submission.json)确认实际GPU提交0、observer未启动、没有在途GPU作业需要取消；jiyujia当时剩余5.29285 GPU小时与Codex额度是不同口径。该时点没有重置／购买额度、GPU消耗、新reference、31B梯度结果或optimizer更新。用户后来明确继续新窗口，产生本文开头的R4b–R7实际结果，旧时点记录不覆盖后续事实。
