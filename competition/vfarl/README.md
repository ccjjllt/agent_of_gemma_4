# Verified Focal Agent RL 数据准备

2026-10-28最新：[VFARL算法与固定100题深审](l4x4/round100/optimization-overnight/README.md)。已保留native与Unsloth各一次真实单步更新，尚无独立修复率／比赛收益结论。本轮持续修复算法、题面、监督及运行环境，正式GPU训练关闭；下文各日期的0更新、旧课程与GPU配额均为对应历史实验事实。

2026-10-26 新实验入口：[Kaggle L4×4 训练验证](l4x4/README.md)。已创建独立训练验证分支并固定 S600 中的 100 题，开始补齐真实当前策略采样、状态恢复及 PPO/KL 更新路径。下文的“未启动采样或训练”是此前数据准备阶段的事实；新实验只以该入口保存的实际结果为准。

S600 与 F600 均已完成 **400共同＋200专科＝600题** 的题目、真实工具证据与冷启动对白准备。当前F专科复核与修正见[F600说明](f600/README.md)及[课程清单](f600/course-manifest.json)；S的数据保持冻结。两课共享相同400共同题，合计800个不同训练任务；未启动模型采样或GPU训练。

以下81题是S600扩容开始前的已验收起点，原96题的61正例和35未入池历史不改；新增批次以最新准入和课程清单为准。

2026-10-25。当前共有81条完整验收并完成Gemma对白导出的off-policy冷启动示范：G28、S16、F18、R19。本轮新增14题。原96题全部已实跑，其中61题可用；另有20题明确补充，原失败记录未替换。原96仍有35题未入池，数据尚未全部准备完毕。

重点S2/F2/R2/R5各4条可用；本轮将F1/F3/F4/R1/R3补到各4条，R4增至3条。已导出716条决策（641工具调用＋75实际终止回答）、89,774个监督tokens。四个adapter主责数据和共享课程已分别导出，采样比例未设定。S3/F5各2条、S4/S5/R4各3条仍需扩充。

入口：[每机制数量与边界](pilot96/README.md)、[原96逐题目录](pilot96/catalog.md)、[补充来源目录](acquisition/supplemental/catalog.json)、[真实结果和原始证据](replay/canonical-results-manifest.json)、[对白清单](replay/gemma-sft-manifest.json)、[四份主责/共享课程](replay/adapter-curricula-manifest.json)。

G r128五投影、S/F/R r32七投影、BF16计算/导出方案不变。示范由Codex指定和审阅、通过真实官方六工具执行，不是Gemma当前策略采样；没有编造思考、logprob、关键动作对或训练收益。两条基线F2P标签与当前运行结果有差异的任务、一个保守unittest ID映射样本均单独标注，原修后验收要求全部保留。4道原题面为空的任务采用固定审阅hash的行为澄清题面，已用该题面重新实跑，明确标为curator版本；77道保留原题面。

当前使用官方SubprocessManager的CPU本地后端，尚未证明比赛Docker等价。只使用允许train，未读四比赛仓/129公共开发题或heldout正文。模型rollout、同状态已验证关键动作对和GPU训练均为0。设计与后续采集假设见[data-contract.md](data-contract.md)及[collection-plan.json](collection-plan.json)，不能把冷启动示范视为整个RL数据已齐备。
