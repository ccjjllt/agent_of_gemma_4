# 研究索引

截至 2026-10-24，**P016首轮正式主训练已完成**：537个完整问题、68仓、7,119,739输入/1,291,819监督tokens、68/68更新、1 epoch，10月23日17:40正常退出；最终adapter和全部七个checkpoint已完成Drive备份，真续训已在A40实测。见 [P016执行记录](016/README.md)。原677题语料不是此次实际训练版本。

本阶段 [P017权重对照](../docs/plan/017-kaggle-weight-transfer-and-score-handoff.md) 已使用jiyujia完成，见 [结果与下一步](017/inference/results-and-next-step.md) 和 [执行记录](017/README.md)。旧12题同期Base6/12、Step-68 6/12，W1/L1、净0；新16题/7仓单活动Agent同期Base14/16、Step-68 9/16，W0/L5、净−5。全部配对有效，当前保留Base。GPU累计配额差额6.53h，剩23.47h；两作业均结束。正式最好成绩仍为 **0.15**；最近已保存正式提交 `57471497` COMPLETE，无新正式提交。[正式回执](014/submission/runtime-v2/README.md)

## P015 运行基础

先读 [执行记录](015/README.md)，再按需要进入 [研究计划](../docs/plan/015-31b-posttraining-and-score-handoff.md)、[数据准备](015/data-preparation/README.md)、[开发面板与实际基线](015/evaluation/README.md) 和 [权重 A/B 载体](015/inference/weight-ab/README.md)。

- 已完成 31B FSDP 三次真实优化器更新、最长 6,846-token 容量实验、独立 adapter 导出与 CT 服务路由确认。这些是容量与运行证据，尚未证明任务修复收益。
- 冻结的训练外开发面板，base 实际为 **5/12**；另行统计的 train 采样为首批 **1/3**、新增八题 **8/16 draws** resolved。draw 数不等于独立问题数，开发比例不换算为正式分数。
- 首轮 action RFT 输入只有 **3 个独立问题、14 个动作、1,014 个监督 tokens**。P016已回收 `zhuating/p015-31b-task-rft/1`，实际为取消，只确认第一步完成，未证实8,165-token第二步更新结束或最终导出；[回收证据](016/p015-recovery/actual-result.json)。权重 A/B 尚无结果。

这里概述本地证据的最新状态，不代替远端状态查询；后续实际结果继续追加到对应轮次的执行记录。

## 历史轮次

编号用于对齐计划、代码、冻结输入和运行证据。`002/`–`015/` 的既有路径继续保留，避免破坏载体和来源指纹中的引用。旧文档中的 PENDING、待训练等描述属于其写作时点，阅读时以同轮后续结果和当前执行记录为准。

| 轮次 | 研究用途与已知结果边界 | 主要入口 |
| --- | --- | --- |
| 002 | 代码定位、精确编辑、上下文与运行条件；容量对照保留 16K，后续候选与资源路线有冻结记录 | [进展与证据](../docs/progress/002-code-localization.md)、[研究交接](../docs/plan/002-research-opportunities-20261001.md) |
| 003 | 时间/调用预算与上下文行为；主面板 A0/B 均 3/24，无净增，保存原始轨迹及归因更正 | [进展与原始轨迹复核](../docs/progress/003-budget-and-context.md) |
| 005 | 公开方案、机制候选、早期真实训练和 single-state 载体；正式 0.13 为历史结果 | [进展](../docs/progress/005-medal-and-prize-research.md)、[训练工程](../training/README.md) |
| 006 | 强参考配对、三仓真实工具重放、12B CPU 适配 | [进展](../docs/progress/006-from-013-to-competitive-agent.md) |
| 007 | 公开 12 题配对、12B QAT→NF4 缓存、真实样本首步训练和 CT 诊断 | [进展与后续更正](../docs/progress/007-public022-and-transfer.md)、[12B 实验说明](007/12b-qat-cache-and-train-20261017/README.md) |
| 008 | 既有工作复核和正式提交；0.15 的历史起点 | [进展与提交](../docs/progress/008-reviewed-submission-and-real-sft.md) |
| 009 | 多仓真实 SFT 与完整 Agent 对照；base/adapter 均 1/3 | [进展](../docs/progress/009-real-sft-and-agent-transfer.md)、[实际运行报告](009/execution-20261018/report-v1.md) |
| 010 | 复用 P009 adapter 研究响应预算；base 2/3、adapter 1/3，未见 adapter 净收益 | [预算对照](../docs/progress/010-budget-transfer.md) |
| 011 | 139 个独立问题的两轮连续 SFT，实际完成 70 步和独立导出；训练完成不等于修复收益 | [进展](../docs/progress/011-batch-sft.md)、[训练结果与资产](011/training-run-20261018/README.md) |
| 012 | 多角色 Agent、修复角色 adapter 路由与完整矩阵；保存负结果和候选打捞缺失说明 | [执行记录](012/README.md) |
| 013 | 31B 修复诊断、行为 SFT 与后续评测准备；后续事实由 P014 更新 | [执行记录](013/README.md) |
| 014 | 31B/12B 修复对照、真实提交与整体超时修复；runtime-v2 正式 0.15，无提分 | [执行记录](014/README.md)、[最终正式结果](014/submission/runtime-v2/README.md) |
| 015 | 31B 容量、同协议采样与权重 A/B 准备；小RFT作业已取消，无提分证据 | [执行记录](015/README.md) |
| 016 | 537题/68仓/68步主训练及完整备份完成，官方CT迁移与修复收益待评估 | [执行记录](016/README.md)、[历史交接方案](../docs/plan/016-large-scale-posttraining-handoff.md) |
| 017 | 同期对照已完成；D6/12对6/12，C14/16对9/16，W0/L5净−5；保留Base，完整轨迹已回收 | [结果与下一步](017/inference/results-and-next-step.md)、[执行记录](017/README.md)、[交接方案](../docs/plan/017-kaggle-weight-transfer-and-score-handoff.md) |

P004 只有 [针对失败行为的 LoRA 计划](../docs/plan/004-targeted-lora.md)，没有独立 `research/004/`；P001 的早期基线见 [基线进展](../docs/progress/001-baseline.md) 和根目录的 `results/`。

## 文件保留与使用

- **研究证据**：保留真实结果、失败日志、源码版本、来源与 SHA256 清单、正式提交回执以及配额消耗记录。旧失败和负结果用于约束结论，不能当作垃圾删除。
- **数据与权重**：保留训练输入、冻结划分、原始采样输出、adapter 和必要 checkpoint。大文件通常受 `.gitignore` 管理；未进 Git 不代表可删除，也不能仅凭文件名或大小推断重复。
- **任务划分**：[task_splits.json](task_splits.json) 仅用于核对任务身份、分组和划分元数据。不要读取 heldout/预留评测题正文，不把它们转为训练数据。
- **上游判据**：快照和来源包里的 `tests/`、`test-patches/` 等可能是实验使用的上游既有判据，不是项目新建的测试套件，不能按名称统一清除。项目约定见 [AGENTS.md](../AGENTS.md)：不新增专用测试文件，相关核对采用一次性手动命令。
- **可重建内容**：`__pycache__/`、工具缓存、临时安装的依赖目录和执行工作树可在核对来源及保存证据后清理。归档快照、离线 wheelhouse、原始模型输出和权重继续按实验用途保留；不要把所有 `output/`、`data/` 或 ignored 目录都视为临时文件。

跨轮计划在 [docs/plan/](../docs/plan/README.md)，早期执行进展在 `docs/progress/`。每轮运行载体、局部说明和原始证据仍放在对应编号目录中。
