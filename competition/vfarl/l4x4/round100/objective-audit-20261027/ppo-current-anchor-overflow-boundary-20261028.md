# 当前单步PPO ratio范围与非有限误差处理

2026-10-28。普通有限值路径的PPO overflow候选关闭，没有修改exp、clipping或目标。当前全部VFARL配置的误差阈值为0.08；Unsloth在ratio计算前校验current-grad与冻结learner-old差的有限性及上限，native的finite strict/corrected路径分别约束实际ratio的actor-old/learner-old。所有动作的反向先累积，最后才有唯一一次optimizer.step；下一resident更新重新计算自己的learner-old，不是假想多epoch更新。因此普通路径ratio在约0.923116至1.083287之间。

另确认了native原有一致性检查的一处窄缺口：`errors.max()`遇NaN后，与阈值比较为False。明确构造的CPU例中，current=-1、learner-old=-101、repeat=NaN，使100的实际差被NaN隐藏。正优势PPO的forward被clip成有限值，k3也有限，而FP32 exp的反向产生NaN。原grad-norm检查仍会阻止optimizer，这不是已观测31B故障或有效finite数据可达的overflow证据。

最小修复只在既有MAX/误差判定前加入`isfinite(errors)`的跨rank MIN；任一rank非有限就全部在backward前拒绝，不填补概率或NaN，不尝试挽救该样本。真实CPU张量与实际源码predicate核对了该MIN代数及finite分支；这是单CPU进程的四rank误差向量模拟，不声称实际Gloo/NCCL/FSDP验证。移除新增四行后，其余源码字节精确回到旧`772c6b09…`；此前post-step证据保持原件。

[完整输入、实际表达式、配置SHA及结果](ppo-current-anchor-overflow-boundary-20261028.json)。没有新增测试文件、模型、GPU、正式训练、optimizer操作或收益结论。
