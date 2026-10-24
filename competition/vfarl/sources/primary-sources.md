# Verified Focal Agent RL：原始来源核对

核查日期：2026-10-24（Asia/Shanghai）。只读研究；未调用 GPU、付费模型或读取比赛 heldout。下列机制是原始来源所述；末节明确标注我们对数据设计的推论。

## FocalRL

- 官方：[microsoft/FocalRL](https://github.com/microsoft/FocalRL)。当前核实到唯一初始化提交 [bfdf5f9aee6aedb00a9c66d7217421373bb753c2](https://github.com/microsoft/FocalRL/commit/bfdf5f9aee6aedb00a9c66d7217421373bb753c2)，提交时间 **2026-10-09 06:01:02 UTC**（来自[官方 commit feed](https://github.com/microsoft/FocalRL/commits/main.atom)，提交时间不能当作论文发表日期）。本轮未定位可核验 arXiv 论文链接，不编造论文编号或发表日期。
- 官方 README 方法：完整 rollout → 从失败轨迹及成功参照定位 consequential error → 在错误前恢复并分叉 → 短 continuation → 同时优化整题奖励与局部 repair 奖励。README 示例局部窗口最多五个 assistant turns；局部 rubric 为仍犯错 0、避开错但未完成指定修正 0.5、指定修正完成 1。这些局部分数不等于整题 resolved。
- [SWE 文档](https://github.com/microsoft/FocalRL/blob/main/docs/swe.md)区分了 full trajectory 的 test verifier 与 local continuation 的 repair rubric；不是每个短分支都运行到最终成功。
- [SWE replay 实现](https://github.com/microsoft/FocalRL/blob/main/adaptive_branching/src/swe/lightning_replay.py)：`build_replay` 保存动作前消息前缀及前序命令 ledger。`restore_replay` 重放命令并核对 return code；虽 ledger 包含 output SHA，restore 函数并未比较它。v3 无 filesystem boundaries（legacy 版本有 `/testbed`、`/tmp` 指纹字段）。所以不能声称该实现已证明完整执行状态、剩余时间完全等价。

## Verified Critical Step Optimization（CSO）

- 原论文：[arXiv:2602.03412v2](https://arxiv.org/html/2602.03412v2)，首次提交 **2026-02-03**，修订 **2026-04-29**，论文标注 ACL 2026 Findings；[作者代码](https://github.com/kiaia/CSO)。
- §3.2：从当前 policy 的失败轨迹出发，PRM 比较 policy 动作与专家候选，只把低质量原动作、高质量替代动作当作候选。用替代动作替换该步后，**后续由当前 policy 自己执行直到终止**；只有整题 ground-truth outcome 成功才成为 verified critical step。训练 pair 为共享 `s_t` 的 `(a_plus, a_minus)`，§3.3 用带 reference policy 的 DPO。迭代时重新采集当前 policy 的失败。
- 易误读：专家给的动作、编辑工具返回成功、PRM 高分都不是最终验证。一次替代分支成功是该实验中的操作性反事实证据，不证明动作在所有随机续轨迹中必然改变结果，也不是把后续所有动作强行固定。原论文验证的是 GAIA-Text-103 / XBench-DeepSearch，不是 Gemma 4 仓库修复收益。

## GiGPO

- 原论文：[arXiv:2505.10978v3](https://arxiv.org/html/2505.10978v3)，首次提交 **2025-05-16**，当前修订 **2025-10-28**，NeurIPS 2025；[会议版 PDF](https://papers.nips.cc/paper_files/paper/2025/file/420c9f777c0b4f78d515e53cf74d58b2-Paper-Conference.pdf)；[作者代码 verl-agent](https://github.com/langfengQ/verl-agent)。
- §4.1 同任务、同初始环境采完整轨迹组，按整轨迹 return 算 episode advantage。§4.2 在已有轨迹里回溯匹配相同 environment state 构成 anchor group，比较从该步开始的 discounted returns；不是在每个 state 额外分叉采样。无匹配组则 step 信号退化为零。
- 论文 Eq.8 是 `A(a_t^i)=A^E(tau_i)+omega*A^S(a_t^i)`，不是用户的 `alpha*(R_task-mean_task)+(1-alpha)*(R_branch-mean_branch)`。用户公式可以作为本项目待研究设计，应标明自己的组定义、归一化、样本量和随机性处理，不能称为 GiGPO 原公式。其 state 定义也不自动保证相同完整消息前缀与相同剩余时间。

## Kimi K3 / K3.1

- [官方开放公告](https://www.kimi.com/news/kimi-k3-open-source)更新于 **2026-07-27**，链接到[官方仓库和技术报告](https://github.com/MoonshotAI/Kimi-K3)。本轮实读[官方 PDF](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)，47 页，§4.1.2 / PDF 第 13 页；本地临时副本 `/private/tmp/k3_tech_report.pdf`，PDF metadata 创建/修改时间 2026-08-06 10:14:55 UTC，不能把它混同首次发布日期。
- Reasoning Effort RL：cold-start 模型估计每题初始 token 预算 `b0(x)`；当 `T(y)>tau*b0(x)`，**把任务奖励覆盖为 -1**。普通题 T 计 thinking；agent 题 T 计累计全部输出，包括推理与工具参数。先训练较宽 tau 的 max，再按领域人工指导退火 tau 得 high/low。报告没公开具体 tau 值和 b0 完整估计流程；墙钟平均/尾部约束是我们的扩展，不是该报告的原公式。
- [K3.1 Pro 官方博客](https://www.kimi.com/en/blog/kimi-k3-1-pro)页面 metadata `dateModified=2026-10-24`。博客说明跨数百领域扩大 joint RL 及系统优化，但没有公开可直接复现的训练代码、完整奖励、tau/退火参数；不能把 K3.1 的概览当成一个已公开算法配置。

## 对本项目准备数据的推论（不是论文原结论）

1. 分离三种证据：工具动作有效 / 局部机制修复 / 整题成功；仅最后一种能产生严格 CSO 正例。局部正例可供 Focal 风格局部奖励，但保留独立字段，不升格为 resolved。
2. 可恢复状态至少需要：任务/版本/容器摘要，源码快照与未跟踪文件，完整消息前缀，六工具 schema 和实现版本，工作目录与环境，剩余时间/token/步数。pair 要从同一状态产生；旧公开轨迹没有这些证据时只能作 cold-start 候选，不能伪造为反事实 pair。
3. 每个分支记录行为 policy/checkpoint、采样设置与随机种子、首次替代动作、真实 observations、后续 rollout、局部 rubric、终局 verifier 和成本。DPO 负例来自当前 policy 的实失败，不应人工编造“坏动作”。
4. 预算数据应存原始计数与测时，再研究 reward；按同题基准比较，不能把所有长输出都当错，也不能因短而把未完成任务当好。T 需使用比赛 tokenizer 的全累计输出，不只统计最后一次答案或 thinking 字段。
5. 一次成功对照不能估计可靠成功概率；高随机性或高成本的临界样本应标明 sample count、branch variance、已验证范围。对无法证明状态相同的分支，不组成严格局部偏好。

本轮没有运行任何训练、分支采样或任务验证；以上是原始来源核对与数据合同要求。
