# Verified Focal Agent RL 数据准备

2026-10-25。当前共有81条完整验收并完成Gemma对白导出的off-policy冷启动示范：G28、S16、F18、R19。本轮新增14题。原96题全部已实跑，其中61题可用；另有20题明确补充，原失败记录未替换。原96仍有35题未入池，数据尚未全部准备完毕。

重点S2/F2/R2/R5各4条可用；本轮将F1/F3/F4/R1/R3补到各4条，R4增至3条。已导出716条决策（641工具调用＋75实际终止回答）、89,774个监督tokens。四个adapter主责数据和共享课程已分别导出，采样比例未设定。S3/F5各2条、S4/S5/R4各3条仍需扩充。

入口：[每机制数量与边界](pilot96/README.md)、[原96逐题目录](pilot96/catalog.md)、[补充来源目录](acquisition/supplemental/catalog.json)、[真实结果和原始证据](replay/canonical-results-manifest.json)、[对白清单](replay/gemma-sft-manifest.json)、[四份主责/共享课程](replay/adapter-curricula-manifest.json)。

G r128五投影、S/F/R r32七投影、BF16计算/导出方案不变。示范由Codex指定和审阅、通过真实官方六工具执行，不是Gemma当前策略采样；没有编造思考、logprob、关键动作对或训练收益。两条基线F2P标签与当前运行结果有差异的任务、一个保守unittest ID映射样本均单独标注，原修后验收要求全部保留。4道原题面为空的任务采用固定审阅hash的行为澄清题面，已用该题面重新实跑，明确标为curator版本；77道保留原题面。

当前使用官方SubprocessManager的CPU本地后端，尚未证明比赛Docker等价。只使用允许train，未读四比赛仓/129公共开发题或heldout正文。模型rollout、同状态已验证关键动作对和GPU训练均为0。设计与后续采集假设见[data-contract.md](data-contract.md)及[collection-plan.json](collection-plan.json)，不能把冷启动示范视为整个RL数据已齐备。
