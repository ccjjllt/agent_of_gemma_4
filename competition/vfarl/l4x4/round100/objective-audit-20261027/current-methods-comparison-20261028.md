# 当前论文与VFARL的可迁移边界

2026-10-28，直接阅读原论文正文；没有采用论文算法、启动模型或新增训练收益。此处只记录与当前缺陷研究相关的三项方法，避免把相似术语当成同一目标。

[SPROUT，2026-10-09，§3.3–3.5](https://arxiv.org/html/2610.14405v1)配对恢复执行环境与上下文，利用验收诊断规划分叉；其带指导分支训练明确属于蒸馏式surrogate。它支持重视环境恢复和共享前缀信用分配，不能作为本项目focal-only无偏或成本收益的证据。

本项目自己的实现边界是：Snapshot复制完整可写task root（含venv/tmp），外部共享运行时只读并绑定来源，时钟/熵不在完整声明范围；不是全磁盘microVM。当前live-pair在观察失败后、下一动作与结果出现前固定状态。若以后用终局oracle诊断生成首动作，采样分布与所需行为来源会改变，不能沿用当前同策略pair声明。本文未据论文增加指导角色或把诊断写入主100的模型上下文。

[Anatomy of Off-Policy Policy Gradient，2026-10-19，§4与Remark1](https://arxiv.org/html/2610.25725v1)分析完整轨迹regularized return的组中心化路线；其路径常数机制依赖确定性转移，有限样本结论另有覆盖等条件。这不等于“任何组均值加任意KL”都获得同一结论。

[Proximal KL，2026-10-13，§3 Eq3](https://arxiv.org/html/2610.18738v1)使用当前token重要性权重与完整未来log-ratio代价，在覆盖和局部表达条件下研究stationary target；该结论不声称token-wise估计无偏，也不能直接移给截断窗口或现有k3。

本项目另作[两步精确CPU枚举](proximal-stationarity-two-step-20261028.md)：Gibbs proximal最优点上，完整future-KL及trajectory-return组中心化场约零，local-only信用仍非零。当前实际fixed-scale reducer加PPO/k3/mixed-group/TIS算式在该点也非零，说明这些目标不同；这不是现实现的新增bug。当前reference是冻结base，focal续写使用actor，KL只在mixed不同动作组上评估，不能借上述论文补成全轨迹或proximal保证。

下一步算法比较仍须先取得真实概率与数值路径证据：固定同一批原prefix/token/behavior，记录真实reference、执行差异和梯度；随后才有依据选择新的目标或更新实验。已有tiny CPU结果不能替代31B NF4／CUDA，论文结果也不能填补本项目尚缺的独立修复率证据。正式GPU训练保持关闭。
